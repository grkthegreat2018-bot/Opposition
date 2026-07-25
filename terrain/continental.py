"""Continental-scale terrain: land/ocean mask, coastal mountains, biome regions.

All functions are pure functions of world position → seam-safe across chunks.
The continental mask is a very low-frequency noise field in [0, 1] that
determines large-scale landmasses:

    mask < 0.40  → deep ocean (flat seafloor below sea level)
    0.40-0.55    → continental shelf / coast (beaches, shallow water)
    0.55-0.75    → coastal highlands (mountain chains)
    mask > 0.75  → continental interior (rolling hills, plains)

Mountain ranges form along the coast-to-interior transition (mask ~0.65),
mimicking tectonic plate boundary mountain chains. The deep interior is
flatter with gentler hills, like continental cratons.
"""
import numpy as np
import numba
from terrain._noise_core import _snoise2_scalar, _PERM2, _GRAD2


def _continental_mask(xx, zz, freq, seed):
    """Return a continental mask in [0, 1] for the given world coordinates.

    Uses 3-octave fBm at a very low frequency (default ~40x lower than the
    base terrain) to produce large landmasses ~500-2000m across. The raw
    noise in [-1, 1] is remapped to [0, 1] and smoothed.
    """
    out = np.zeros_like(xx, dtype=np.float32)
    _continental_mask_grid(
        xx * freq + seed * 5000.0,
        zz * freq + seed * 5000.0,
        out, 3, 0.5, 2.0, _PERM2, _GRAD2,
    )
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _continental_mask_grid(x, z, out, octaves, persistence, lacunarity, perm, grad2):
    """3-octave fBm remapped to [0, 1] with smoothstep for gentle coastlines."""
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            amp = 1.0
            total = 0.0
            norm = 0.0
            xi = x[i, j]
            yi = z[i, j]
            for _ in range(octaves):
                v = _snoise2_scalar(xi, yi, perm, grad2)
                total += amp * v
                norm += amp
                amp *= persistence
                xi *= lacunarity
                yi *= lacunarity
            # Normalize to [-1, 1] then remap to [0, 1].
            t = total / max(norm, 1e-6)
            t = t * 0.5 + 0.5
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            # Smoothstep for softer coastline transitions.
            out[i, j] = t * t * (3.0 - 2.0 * t)


@numba.njit(cache=True, fastmath=True, parallel=True)
def _continental_elevation(mask, sea_level, ocean_depth, land_boost):
    """Map a [0, 1] mask to a base elevation field.

    Below the sea threshold the floor drops to ``sea_level - ocean_depth``
    (flat seafloor). Above it, elevation rises smoothly with the mask,
    boosted by ``land_boost`` to lift the continental interior above the
    coastal plains. The transition uses smoothstep so coastlines are
    gradual, not cliffs.
    """
    out = np.empty_like(mask)
    # Sea threshold: mask values below this are underwater.
    sea_t = 0.40
    for i in numba.prange(mask.shape[0]):
        for j in range(mask.shape[1]):
            m = mask[i, j]
            if m <= sea_t:
                # Deep ocean: smooth descent from sea_level to ocean floor.
                # At m=0 → sea_level - ocean_depth; at m=sea_t → sea_level.
                t = m / sea_t  # [0, 1]
                t = t * t * (3.0 - 2.0 * t)  # smoothstep
                out[i, j] = sea_level - ocean_depth * (1.0 - t)
            else:
                # Land: smooth rise from sea_level to land_boost.
                t = (m - sea_t) / (1.0 - sea_t)  # [0, 1]
                t = t * t * (3.0 - 2.0 * t)  # smoothstep
                out[i, j] = sea_level + land_boost * t
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _coastal_ridge_weight(mask, peak, width, strength):
    """Map a [0, 1] continental mask to a ridge-weight multiplier.

    Mountains peak at ``mask = peak`` (default ~0.65, the coast-to-interior
    transition) with a Gaussian-like falloff of ``width``. Deep ocean and
    deep interior both get near-zero ridge weight, producing:
      - Flat seafloors (no mountains underwater)
      - Coastal mountain chains (tall ridges along continental edges)
      - Gentle interior (rolling hills, not jagged peaks)

    ``strength`` scales the peak weight (0 = no coastal mountains, 1 = full).
    """
    out = np.empty_like(mask)
    inv_w2 = 1.0 / (width * width)
    for i in numba.prange(mask.shape[0]):
        for j in range(mask.shape[1]):
            m = mask[i, j]
            d = m - peak
            # Gaussian falloff, clamped to [0, 1].
            w = np.exp(-d * d * inv_w2)
            if w > 1.0:
                w = 1.0
            out[i, j] = w * strength
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _ocean_flatness(mask, sea_t, transition, floor=0.0):
    """Map a [0, 1] mask to a terrain-amplitude multiplier.

    Returns ``floor`` in deep ocean (gentle underwater terrain detail
    instead of a completely flat seafloor), 1.0 on land, with a smooth
    transition zone of width ``transition`` around the sea threshold.
    ``floor=0`` gives a flat seafloor; ``floor=0.15`` gives gentle hills.
    """
    out = np.empty_like(mask)
    for i in numba.prange(mask.shape[0]):
        for j in range(mask.shape[1]):
            m = mask[i, j]
            if m <= sea_t - transition:
                out[i, j] = floor
            elif m >= sea_t + transition:
                out[i, j] = 1.0
            else:
                t = (m - (sea_t - transition)) / (2.0 * transition)
                t = t * t * (3.0 - 2.0 * t)
                out[i, j] = floor + (1.0 - floor) * t
    return out
