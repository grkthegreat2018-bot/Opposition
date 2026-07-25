"""Large-scale terrain features: rivers, plateaus, canyons, craters. All seam-safe."""
import math
import numpy as np
import numba
from terrain._noise_core import _snoise2_scalar, _PERM2, _GRAD2


@numba.njit(cache=True, fastmath=True, parallel=True)
def _apply_rivers(xx, zz, h, scale, seed, perm, grad2,
                  river_freq=0.004, river_width=4.0, river_depth=6.0):
    """Carve V-valleys along a noise-based river network.

    Uses a low-frequency ridge noise as the 'river mask' — its zero crossings
    form a dendritic network. Distance to the nearest zero crossing is
    approximated by |noise| / |gradient|, converted to world units by dividing
    by freq. Domain-warped for meandering. Seam-safe: pure function of world
    position.
    """
    out = h.copy()
    for i in numba.prange(xx.shape[0]):
        for j in range(xx.shape[1]):
            wx = xx[i, j]
            wz = zz[i, j]
            # Domain warp for meandering (low-freq noise offset).
            mx = _snoise2_scalar(wx * 0.0008 + 11.0, wz * 0.0008, perm, grad2) * 80.0
            mz = _snoise2_scalar(wx * 0.0008, wz * 0.0008 + 11.0, perm, grad2) * 80.0
            # River structure noise (low freq → large river basins).
            s = (wx + mx) * river_freq + seed * 7.0
            t = (wz + mz) * river_freq + seed * 7.0
            r = _snoise2_scalar(s, t, perm, grad2)
            # Gradient in noise-input space.
            eps = 0.5
            gx = (_snoise2_scalar(s + eps, t, perm, grad2) - r) / eps
            gz = (_snoise2_scalar(s, t + eps, perm, grad2) - r) / eps
            gm = math.sqrt(gx * gx + gz * gz) + 1e-6
            # Distance to zero crossing in WORLD units = |r| / (|grad| * freq).
            dist = abs(r) / (gm * river_freq)
            # Channel mask: 1 in river, 0 outside. Width in world units.
            w = river_width
            mask = max(0.0, 1.0 - dist / w)
            mask = mask * mask * (3.0 - 2.0 * mask)  # smoothstep
            # V-valley profile: deepest at center, fades to 0 at edges.
            # Scale depth with terrain height so rivers don't carve into oceans.
            depth = river_depth * mask * max(0.0, min(1.0, h[i, j] / (scale * 0.5)))
            out[i, j] = h[i, j] - depth
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _apply_plateaus(xx, zz, h, scale, seed, perm, grad2,
                    mesa_freq=0.0012, plateau_base=0.70, cliff_sharp=0.04,
                    strength=1.0):
    """Clip terrain above a noise-modulated threshold to create flat-top mesas.

    The threshold varies slowly across the world (low-freq noise) so some
    regions have mesas and others don't. Sharp transition (small cliff_sharp)
    produces cliff edges; the plateau top gets a tiny noise perturbation so
    it's not perfectly flat. `strength` controls how much of the map has mesas
    (mask threshold). `plateau_base` is the fraction of scale where the flat
    top sits (high = only the tallest terrain becomes mesas). Seam-safe.
    """
    out = h.copy()
    h_scale = scale
    # Threshold for mesa mask scales with strength (higher strength = more mesas).
    mask_lo = 0.55 - 0.15 * strength
    mask_hi = 0.75 - 0.15 * strength
    for i in numba.prange(xx.shape[0]):
        for j in range(xx.shape[1]):
            wx = xx[i, j]
            wz = zz[i, j]
            # Mesa mask: where do mesas exist at all? Low-freq noise.
            mesa_mask = max(0.0, min(1.0,
                _snoise2_scalar(wx * mesa_freq + seed * 3.0,
                                wz * mesa_freq + seed * 3.0, perm, grad2) * 0.5 + 0.5))
            mesa_mask = smoothstep_numba(mask_lo, mask_hi, mesa_mask)
            if mesa_mask < 0.01:
                continue
            # Plateau height (noise-modulated). In world height units.
            ph_n = _snoise2_scalar(wx * mesa_freq * 2.0 + 99.0,
                                   wz * mesa_freq * 2.0 + 99.0, perm, grad2)
            plateau_h = (plateau_base + 0.15 * ph_n) * h_scale
            # Tiny roughness on plateau top.
            rough = _snoise2_scalar(wx * 0.05, wz * 0.05, perm, grad2) * 0.4
            cur = h[i, j]
            # Smoothstep transition: below plateau_h → keep terrain; above → flat top.
            t = smoothstep_numba(plateau_h - cliff_sharp * h_scale,
                                 plateau_h + cliff_sharp * h_scale, cur)
            new_h = cur * (1.0 - t) + (plateau_h + rough) * t
            out[i, j] = cur * (1.0 - mesa_mask) + new_h * mesa_mask
    return out


@numba.njit(cache=True, fastmath=True, parallel=True)
def _apply_canyons(xx, zz, h, scale, seed, perm, grad2,
                   canyon_freq=0.003, canyon_width=5.0, canyon_depth=10.0):
    """Carve deep narrow canyons with layered walls along a noise-guided path.

    A low-frequency noise field defines the canyon centerline (its zero
    crossing). Distance to the centerline approximated by |noise|/|grad|,
    converted to world units by dividing by freq. The canyon carves down with
    a smooth U-shape; walls are left steep. Layered banding is left to the
    fragment shader (rock_strata) — here we only carve geometry. Seam-safe.
    """
    out = h.copy()
    # Floor clamp: don't carve below a fraction of negative scale.
    floor = -scale * 0.15
    for i in numba.prange(xx.shape[0]):
        for j in range(xx.shape[1]):
            wx = xx[i, j]
            wz = zz[i, j]
            # Domain warp for meandering canyon path.
            mx = _snoise2_scalar(wx * 0.0006 + 22.0, wz * 0.0006, perm, grad2) * 100.0
            mz = _snoise2_scalar(wx * 0.0006, wz * 0.0006 + 22.0, perm, grad2) * 100.0
            s = (wx + mx) * canyon_freq + seed * 11.0
            t = (wz + mz) * canyon_freq + seed * 11.0
            c = _snoise2_scalar(s, t, perm, grad2)
            eps = 0.5
            gx = (_snoise2_scalar(s + eps, t, perm, grad2) - c) / eps
            gz = (_snoise2_scalar(s, t + eps, perm, grad2) - c) / eps
            gm = math.sqrt(gx * gx + gz * gz) + 1e-6
            # Distance to zero crossing in WORLD units.
            dist = abs(c) / (gm * canyon_freq)
            # Canyon mask: narrow channel, wider influence for wall carving.
            w = canyon_width
            mask = max(0.0, 1.0 - dist / (w * 2.5))
            mask = mask * mask * (3.0 - 2.0 * mask)
            # Carve down, clamped to floor.
            cur = h[i, j]
            carve = canyon_depth * mask * mask  # squared for gentler edges
            new_h = max(cur - carve, floor)
            out[i, j] = new_h
    return out


@numba.njit(cache=True, fastmath=True)
def _hash2(x, y):
    """Deterministic 2D hash → uint32. Used for crater center placement."""
    h = np.uint64(x) * np.uint64(73856093) ^ np.uint64(y) * np.uint64(19349663)
    h = h * np.uint64(83492791) + np.uint64(2654435761)
    return np.uint32((h >> np.uint64(32)) & np.uint64(0xFFFFFFFF))


@numba.njit(cache=True, fastmath=True, parallel=True)
def _apply_craters(xx, zz, h, scale, seed,
                   grid_size=80.0, crater_prob=0.18, min_r=5.0, max_r=18.0):
    """Add impact/volcanic craters with raised rims and ejecta blankets.

    Crater centers are placed on a jittered grid (deterministic hash). Each
    cell may or may not have a crater (hash threshold). Profile: bowl-shaped
    depression + raised rim + power-law ejecta blanket. Seam-safe: same hash
    on both sides of a chunk boundary → same craters.
    """
    out = h.copy()
    for i in numba.prange(xx.shape[0]):
        for j in range(xx.shape[1]):
            wx = xx[i, j]
            wz = zz[i, j]
            # Check the 3x3 neighborhood of grid cells (for craters near borders).
            for gx_off in range(-1, 2):
                for gz_off in range(-1, 2):
                    gx = int(math.floor(wx / grid_size)) + gx_off
                    gz = int(math.floor(wz / grid_size)) + gz_off
                    cell_hash = _hash2(gx + seed * 31, gz + seed * 17)
                    if (cell_hash % 1000) / 1000.0 >= crater_prob:
                        continue
                    # Crater parameters from hash.
                    R = min_r + (cell_hash >> 10) % 1000 / 1000.0 * (max_r - min_r)
                    depth = (R * 0.18) * (0.5 + (cell_hash >> 20) % 100 / 100.0)
                    rim = depth * 0.35
                    # Jittered center within the grid cell.
                    jx = (cell_hash >> 5) % 100 / 100.0 * grid_size
                    jz = (cell_hash >> 15) % 100 / 100.0 * grid_size
                    cx = gx * grid_size + jx
                    cz = gz * grid_size + jz
                    dx = wx - cx
                    dz = wz - cz
                    r = math.sqrt(dx * dx + dz * dz)
                    if r > R * 3.0:
                        continue
                    r_norm = r / R
                    # Crater profile (piecewise).
                    if r_norm < 0.2:
                        crater_h = -depth
                    elif r_norm < 0.8:
                        t = (r_norm - 0.2) / 0.6
                        crater_h = -depth + (depth + rim) * t * t
                    elif r_norm < 1.0:
                        t = (r_norm - 0.8) / 0.2
                        crater_h = rim * (1.0 - t)
                    else:
                        # Ejecta blanket: power-law decay.
                        crater_h = rim * (R / r) * (R / r) * (R / r) * 0.3
                    # Smooth blend at influence edge.
                    blend = max(0.0, min(1.0, (R * 3.0 - r) / (R * 0.5)))
                    blend = blend * blend * (3.0 - 2.0 * blend)
                    out[i, j] = out[i, j] * (1.0 - blend) + (h[i, j] + crater_h) * blend
    return out


@numba.njit(cache=True, fastmath=True)
def smoothstep_numba(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


@numba.njit(cache=True, fastmath=True, parallel=True)
def _apply_glacial_valleys(xx, zz, h, scale, seed, perm, grad2, strength=1.0):
    """Carve U-shaped glacial valleys in high-altitude terrain.

    Uses a low-frequency noise field to determine glacier flow paths. Where
    flow is strong and terrain is high, carves a wide U-valley (truncated
    parabola cross-section: widest at top, narrow at bottom). Seam-safe:
    pure function of world position.
    """
    out = h.copy()
    glacier_freq = 0.0018
    for i in numba.prange(xx.shape[0]):
        for j in range(xx.shape[1]):
            wx = xx[i, j]
            wz = zz[i, j]
            cur = h[i, j]
            h_norm = cur / scale
            if h_norm < 0.3:
                continue
            # Glacier flow noise (low freq → broad valley corridors).
            g = _snoise2_scalar(wx * glacier_freq + seed * 19.0,
                                wz * glacier_freq + seed * 19.0, perm, grad2)
            g = max(0.0, min(1.0, g * 0.5 + 0.5))
            # Height mask: only carve above 40% of scale.
            h_mask = smoothstep_numba(0.3, 0.6, h_norm)
            # Flow mask: only where glacier flow is strong.
            g_mask = smoothstep_numba(0.4, 0.7, g)
            carve = strength * scale * 0.15 * g_mask * h_mask
            # Truncated parabola: wider at top, narrow at bottom.
            # Cross-section factor based on height fraction within carved zone.
            top_frac = max(0.0, min(1.0, (h_norm - 0.3) / 0.4))
            u_shape = 1.0 - (1.0 - top_frac) ** 2
            out[i, j] = cur - carve * u_shape
    return out
