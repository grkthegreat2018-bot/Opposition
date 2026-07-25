"""Biome weight computation and biome-modulated parameter tables."""
import numpy as np
from terrain._noise_core import _fbm_grid, _PERM2, _GRAD2


_BIOME_AMPLITUDE = np.array([0.45, 2.80, 0.50, 0.85], dtype=np.float32)
_BIOME_FLATNESS = np.array([0.18, 1.00, 0.55, 0.72], dtype=np.float32)
_BIOME_RIDGE = np.array([0.02, 0.50, 0.20, 0.10], dtype=np.float32)
_BIOME_WARP = np.array([0.06, 0.35, 0.12, 0.18], dtype=np.float32)
_BIOME_DETAIL = np.array([0.02, 0.07, 0.09, 0.08], dtype=np.float32)
_BIOME_CONTINENTAL = np.array([0.25, 0.55, 0.30, 0.40], dtype=np.float32)


def _compute_biome(xx, zz, seed: int = 0, biome_freq: float = 0.0015):
    """Return per-cell biome weights as a (..., 4) float32 array.

    Channel order is (tundra, mountain, desert, forest). Weights are in
    [0, 1] and sum to 1 per cell. Temperature/humidity are driven by
    low-frequency simplex noise so biome regions are large and continuous.
    """
    s = seed * 13.37
    temp = np.zeros_like(xx, dtype=np.float32)
    humid = np.zeros_like(xx, dtype=np.float32)
    _fbm_grid(xx * biome_freq + s, zz * biome_freq, temp, 3, 0.5, 2.0, _PERM2, _GRAD2)
    _fbm_grid(
        xx * biome_freq + 777.0, zz * biome_freq + 777.0, humid, 3, 0.5, 2.0,
        _PERM2, _GRAD2,
    )
    t = np.clip(temp * 0.5 + 0.5, 0.0, 1.0)
    u = np.clip(humid * 0.5 + 0.5, 0.0, 1.0)
    # Smoothstep for softer biome transitions (avoids hard biome seams).
    t = t * t * (3.0 - 2.0 * t)
    u = u * u * (3.0 - 2.0 * u)
    w_tundra = (1.0 - t) * (1.0 - u)
    w_mountain = (1.0 - t) * u
    w_desert = t * (1.0 - u)
    w_forest = t * u
    return np.stack([w_tundra, w_mountain, w_desert, w_forest], axis=-1).astype(
        np.float32
    )


def _biome_modulate(biome, table):
    """Reduce a (..., 4) biome weight array against a length-4 table to (...,)."""
    return (
        biome[..., 0] * table[0]
        + biome[..., 1] * table[1]
        + biome[..., 2] * table[2]
        + biome[..., 3] * table[3]
    )
