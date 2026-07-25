"""Numba-jit'd 2D simplex noise primitives. Bit-identical to noise.snoise2(x,y,1,0.5,2)."""
import math
import numpy as np
import numba


_PERM = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
], dtype=np.int32)
# Doubled for wrap-around (ii + perm[jj] can reach 510).
_PERM2 = np.concatenate([_PERM, _PERM])
# 12 gradient directions for 2D simplex (Gustavson reference).
_GRAD2 = np.array([
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [1, 0], [-1, 0],
    [0, 1], [0, -1], [0, 1], [0, -1],
], dtype=np.float32)
_F2 = 0.3660254037844386   # 0.5 * (sqrt(3) - 1)
_G2 = 0.21132486540518713  # (3 - sqrt(3)) / 6


@numba.njit(cache=True, fastmath=True)
def _snoise2_scalar(x, y, perm, grad2):
    """Single-point 2D simplex noise. Bit-identical to noise.snoise2(x,y,1,0.5,2)."""
    s = (x + y) * _F2
    # int(math.floor(...)) rounds toward -inf (correct for negative coords);
    # numba's int() on a floored float is safe since floor already returned
    # a whole number. Avoids numba.int32() which confuses Pyright's type stub.
    i = int(math.floor(x + s))
    j = int(math.floor(y + s))
    t = (i + j) * _G2
    x0 = x - (i - t)
    y0 = y - (j - t)
    if x0 > y0:
        i1, j1 = 1, 0
    else:
        i1, j1 = 0, 1
    x1 = x0 - i1 + _G2
    y1 = y0 - j1 + _G2
    x2 = x0 - 1.0 + 2.0 * _G2
    y2 = y0 - 1.0 + 2.0 * _G2
    ii = i & 255
    jj = j & 255
    gi0 = perm[ii + perm[jj]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1]] % 12
    gi2 = perm[ii + 1 + perm[jj + 1]] % 12
    t0 = 0.5 - x0 * x0 - y0 * y0
    n0 = 0.0
    if t0 > 0.0:
        t0 *= t0
        n0 = t0 * t0 * (grad2[gi0, 0] * x0 + grad2[gi0, 1] * y0)
    t1 = 0.5 - x1 * x1 - y1 * y1
    n1 = 0.0
    if t1 > 0.0:
        t1 *= t1
        n1 = t1 * t1 * (grad2[gi1, 0] * x1 + grad2[gi1, 1] * y1)
    t2 = 0.5 - x2 * x2 - y2 * y2
    n2 = 0.0
    if t2 > 0.0:
        t2 *= t2
        n2 = t2 * t2 * (grad2[gi2, 0] * x2 + grad2[gi2, 1] * y2)
    return 70.0 * (n0 + n1 + n2)


@numba.njit(cache=True, fastmath=True, parallel=True)
def _snoise2_grid(x, y, out, perm, grad2):
    """Single-octave simplex noise over a 2D grid. Writes into out, returns out."""
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = _snoise2_scalar(x[i, j], y[i, j], perm, grad2)
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _fbm_grid(x, y, out, octaves, persistence, lacunarity, perm, grad2):
    """Plain fBm over a 2D grid (replaces _vnoise2(arr, arr, oct, p, l)).

    Normalized by sum of amplitudes to match noise.snoise2's multi-octave
    behavior (single-octave norm=1.0 so output is unchanged for octaves=1)."""
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            amp = 1.0
            total = 0.0
            norm = 0.0
            xi = x[i, j]
            yi = y[i, j]
            for _ in range(octaves):
                total += amp * _snoise2_scalar(xi, yi, perm, grad2)
                norm += amp
                amp *= persistence
                xi *= lacunarity
                yi *= lacunarity
            out[i, j] = total / max(norm, 1e-12)
    return out
