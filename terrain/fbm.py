"""Fractal Brownian Motion: derivative-eroded fBm, ridged fBm, and height field composition."""
import numpy as np
import numba
from terrain._noise_core import _snoise2_scalar, _fbm_grid, _PERM2, _GRAD2
from terrain.biomes import _biome_modulate, _BIOME_AMPLITUDE, _BIOME_FLATNESS, _BIOME_RIDGE, _BIOME_WARP, _BIOME_DETAIL, _BIOME_CONTINENTAL
from terrain.continental import (
    _continental_mask, _continental_elevation,
    _coastal_ridge_weight, _ocean_flatness,
)

# Per-octave rotation matrix ~37° (det=1) is inlined in _ridged_fbm_grid.


def _fbm_eroded(x, z, octaves, persistence, lacunarity, k=2.0):
    """Slope-aware fBm with derivative-erosion weighting (numba-jit'd).

    Suppresses high-frequency detail on steep slopes via 1/(1+k*|d|^2),
    preventing the bed-of-nails artifact from raw octave stacking. Uses
    forward-difference gradients (3 snoise2 samples per octave). Inputs x, z
    are pre-scaled (freq already applied); the function handles lacunarity
    scaling internally. d is accumulated in the first octave's sample frame
    (no rotation), so gradients align coherently on steep slopes → large |d|
    → strong suppression; cancel on flats → small |d| → full detail.
    """
    out = np.zeros_like(x, dtype=np.float32)
    _fbm_eroded_grid(x, z, out, int(octaves), float(persistence),
                     float(lacunarity), float(k), _PERM2, _GRAD2)
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _fbm_eroded_grid(x, z, out, octaves, persistence, lacunarity, k, perm, grad2):
    """Per-point octave loop in compiled code. ~10x faster than the np.vectorize
    version because the 7*3=21 noise evaluations per cell run without Python
    dispatch and the outer grid loop is parallelized across CPU cores."""
    eps = 0.1
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            amp = 1.0
            total = 0.0
            dx = 0.0
            dz = 0.0
            xi = x[i, j]
            zi = z[i, j]
            for _ in range(octaves):
                v = _snoise2_scalar(xi, zi, perm, grad2)
                gx = (_snoise2_scalar(xi + eps, zi, perm, grad2) - v) / eps
                gz = (_snoise2_scalar(xi, zi + eps, perm, grad2) - v) / eps
                weight = 1.0 / (1.0 + k * (dx * dx + dz * dz))
                total += amp * v * weight
                dx += amp * gx
                dz += amp * gz
                amp *= persistence
                xi *= lacunarity
                zi *= lacunarity
            out[i, j] = total
    return out


def _ridged_fbm_rotated(x, z, octaves, persistence, lacunarity):
    """Ridged fBm with per-octave rotation and smootherstep peaks.

    Returns a ridge field in [0, 1]. Smootherstep (6t^5-15t^4+10t^3) softens
    ridge crests more than smoothstep, reducing spike density. Per-octave
    rotation breaks axis-aligned ridge banding.
    """
    out = np.zeros_like(x, dtype=np.float32)
    _ridged_fbm_grid(x, z, out, int(octaves), float(persistence),
                     float(lacunarity), _PERM2, _GRAD2)
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _ridged_fbm_grid(x, z, out, octaves, persistence, lacunarity, perm, grad2):
    """Per-point ridged fBm with per-octave rotation and smootherstep peaks,
    all in compiled code. Rotation matrix [[0.8,-0.6],[0.6,0.8]] (det=1)."""
    r00, r01 = 0.8, -0.6
    r10, r11 = 0.6, 0.8
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            amp = 1.0
            total = 0.0
            norm = 0.0
            xi = x[i, j]
            yi = z[i, j]
            for _ in range(octaves):
                v = _snoise2_scalar(xi, yi, perm, grad2)
                r = 1.0 - abs(v)
                if r < 0.0:
                    r = 0.0
                elif r > 1.0:
                    r = 1.0
                total += amp * r
                norm += amp
                amp *= persistence
                # Rotate (xi, yi) by [[r00,r01],[r10,r11]] then scale by lacunarity
                nx = (r00 * xi + r01 * yi) * lacunarity
                ny = (r10 * xi + r11 * yi) * lacunarity
                xi = nx
                yi = ny
            ridge = total / max(norm, 1e-6)
            if ridge < 0.0:
                ridge = 0.0
            elif ridge > 1.0:
                ridge = 1.0
            # Smootherstep: 6t^5 - 15t^4 + 10t^3
            s = ridge * ridge * ridge * (ridge * (ridge * 6.0 - 15.0) + 10.0)
            # Sharpen peaks: pow(s, 0.7) boosts values near 1 → jagged crests.
            out[i, j] = s ** 0.7
    return out


def _ridged_fbm_detail(x, z, octaves, persistence, lacunarity):
    """High-frequency ridged fBm with a different rotation angle for sub-peak detail."""
    out = np.zeros_like(x, dtype=np.float32)
    _ridged_fbm_detail_grid(x, z, out, int(octaves), float(persistence),
                            float(lacunarity), _PERM2, _GRAD2)
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _ridged_fbm_detail_grid(x, z, out, octaves, persistence, lacunarity, perm, grad2):
    """Ridged fBm with rotation ~53° (different from main ridge) for jagged spires."""
    r00, r01 = 0.6, -0.8
    r10, r11 = 0.8, 0.6
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            amp = 1.0
            total = 0.0
            norm = 0.0
            xi = x[i, j]
            yi = z[i, j]
            for _ in range(octaves):
                v = _snoise2_scalar(xi, yi, perm, grad2)
                r = 1.0 - abs(v)
                if r < 0.0:
                    r = 0.0
                elif r > 1.0:
                    r = 1.0
                total += amp * r
                norm += amp
                amp *= persistence
                nx = (r00 * xi + r01 * yi) * lacunarity
                ny = (r10 * xi + r11 * yi) * lacunarity
                xi = nx
                yi = ny
            ridge = total / max(norm, 1e-6)
            if ridge < 0.0:
                ridge = 0.0
            elif ridge > 1.0:
                ridge = 1.0
            s = ridge * ridge * ridge * (ridge * (ridge * 6.0 - 15.0) + 10.0)
            out[i, j] = s ** 0.6
    return out


def _compute_height(
    xx,
    zz,
    seed: int = 0,
    freq: float = 0.008,
    scale: float = 25.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    warp_scale: float = 0.3,
    warp_amp: float = 3.0,
    ridge_weight: float = 0.35,
    detail_weight: float = 0.12,
    ridge_detail_weight: float = 0.08,
    biome=None,
    continental_freq: float = 0.0002,
    sea_level: float = 0.0,
    ocean_depth: float = 18.0,
    land_boost: float = 12.0,
    coastal_peak: float = 0.65,
    coastal_width: float = 0.18,
    coastal_mountain_strength: float = 0.7,
    ocean_transition: float = 0.06,
    ocean_detail_floor: float = 0.15,
):
    """Return a height field for given world x/z coordinates.

    The shape uses domain warping for natural-looking valleys, a ridged
    noise layer for peaks, a high-frequency detail layer, and an extra
    micro-detail layer for sub-cell roughness.

    If ``biome`` is given (a (..., 4) array of tundra/mountain/desert/forest
    weights), per-biome amplitude / ridge weight / warp strength / detail
    weight are blended across the field so each biome has distinct geometry
    (mountains are tall and jagged, deserts are low and dune-like, etc.).
    """
    x = xx * freq + seed * 1000.0
    z = zz * freq + seed * 1000.0

    # Biome-modulated parameters. Fall back to scalar defaults when no biome
    # weights are supplied (keeps the function usable as a plain noise field).
    if biome is not None:
        amp = _biome_modulate(biome, _BIOME_AMPLITUDE)
        flat = _biome_modulate(biome, _BIOME_FLATNESS)
        rw = _biome_modulate(biome, _BIOME_RIDGE)
        wp = _biome_modulate(biome, _BIOME_WARP)
        dw = _biome_modulate(biome, _BIOME_DETAIL)
        cont = _biome_modulate(biome, _BIOME_CONTINENTAL)
    else:
        amp = np.float32(1.0)
        flat = np.float32(1.0)
        rw = np.float32(ridge_weight)
        wp = np.float32(warp_amp)
        dw = np.float32(detail_weight)
        cont = np.float32(0.4)

    # Two-octave domain warp: low-frequency for valleys + a higher-frequency
    # secondary warp to break up ridge regularity.
    if warp_scale != 0.0:
        wx1 = np.zeros_like(x, dtype=np.float32)
        wz1 = np.zeros_like(x, dtype=np.float32)
        wx2 = np.zeros_like(x, dtype=np.float32)
        wz2 = np.zeros_like(x, dtype=np.float32)
        _fbm_grid(x * warp_scale, z * warp_scale, wx1,
                  max(1, octaves - 1), persistence, lacunarity, _PERM2, _GRAD2)
        _fbm_grid(x * warp_scale + 100.0, z * warp_scale + 100.0, wz1,
                  max(1, octaves - 1), persistence, lacunarity, _PERM2, _GRAD2)
        _fbm_grid(x * warp_scale * 3.0, z * warp_scale * 3.0, wx2,
                  2, persistence, lacunarity, _PERM2, _GRAD2)
        _fbm_grid(x * warp_scale * 3.0 + 200.0, z * warp_scale * 3.0 + 200.0, wz2,
                  2, persistence, lacunarity, _PERM2, _GRAD2)
        x = x + (wx1 + 0.3 * wx2) * wp
        z = z + (wz1 + 0.3 * wz2) * wp

    # --- Continental mask: large-scale land/ocean pattern ---
    # Computed on the pre-warp coordinates so it's not distorted by domain
    # warp. This drives ocean depth, coastal mountains, and terrain flatness.
    cont_mask = _continental_mask(xx, zz, continental_freq, seed)
    cont_elev = _continental_elevation(cont_mask, sea_level, ocean_depth, land_boost)
    coastal_rw = _coastal_ridge_weight(
        cont_mask, coastal_peak, coastal_width, coastal_mountain_strength)
    # Ocean flatness ramps from ocean_detail_floor (not 0) to 1.0 so the
    # seafloor has gentle terrain detail instead of being a flat plane
    # that creates a hard visual line against detailed land terrain.
    ocean_flat = _ocean_flatness(cont_mask, 0.40, ocean_transition, ocean_detail_floor)

    # Regional low-frequency variation on land (broad rolling hills). The
    # continental mask handles oceans/mountains; this adds interior variety.
    continental = np.zeros_like(xx, dtype=np.float32)
    _fbm_grid(
        xx * freq * 0.25 + seed * 250.0,
        zz * freq * 0.25 + seed * 250.0,
        continental, 3, persistence, lacunarity, _PERM2, _GRAD2,
    )

    # Base fractal terrain with derivative-erosion: accumulate per-octave
    # gradients and suppress detail on steep slopes via 1/(1+k*|d|^2). This
    # is the key fix for the bed-of-nails artifact — without it, high octaves
    # add full amplitude on already-steep slopes, producing dense spikes.
    h = _fbm_eroded(x, z, octaves, persistence, lacunarity, k=2.0) * flat

    # Ridged layer with per-octave rotation and smootherstep peaks. The
    # smootherstep (6t^5-15t^4+10t^3) softens ridge crests more than the
    # previous smoothstep, reducing spike density. Per-octave rotation
    # breaks axis-aligned ridge banding from the noise lattice.
    ridge = _ridged_fbm_rotated(x * 2.0, z * 2.0, octaves, persistence, lacunarity)

    # High-frequency ridged detail layer (jagged rock spires) with a different
    # rotation angle to break up the main ridge pattern.
    ridge_detail = _ridged_fbm_detail(x * 5.0, z * 5.0, max(3, octaves - 2),
                                      persistence * 0.8, lacunarity)

    # Fine detail without changing large-scale shape, plus a micro layer for
    # sub-cell roughness that the fragment shader's normal perturbation can
    # pick up on.
    detail = np.zeros_like(x, dtype=np.float32)
    micro = np.zeros_like(x, dtype=np.float32)
    _fbm_grid(x * 4.0, z * 4.0, detail,
              max(3, octaves - 2), persistence * 0.85, lacunarity, _PERM2, _GRAD2)
    _fbm_grid(x * 9.0, z * 9.0, micro,
              3, persistence * 0.6, lacunarity, _PERM2, _GRAD2)

    # --- Compose final heightfield ---
    # Terrain detail (fBm + ridges + detail) is flattened underwater by
    # ocean_flat (0 in deep ocean → smooth seafloor). Ridge weight gets an
    # extra coastal multiplier so mountains form chains along continental
    # edges rather than uniformly everywhere. The continental elevation
    # provides the ocean/land base, and the regional fBm adds interior variety.
    terrain = (h + (rw + coastal_rw) * ridge
               + ridge_detail_weight * ridge_detail
               + dw * detail + 0.03 * micro)
    return (cont_elev + ocean_flat * terrain * scale * amp
            + cont * continental * 0.4 * scale)
