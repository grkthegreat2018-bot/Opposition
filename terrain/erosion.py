"""Erosion and smoothing post-processing for 2D heightfields."""
import numpy as np
import numba


def _thermal_erode(h, iters: int = 8, talus: float = 1.2, factor: float = 0.35):
    """Thermal (talus) erosion on a 2D heightfield in world units.

    Each iteration moves material from cells whose slope to a 4-neighbour
    exceeds ``talus``. Talus is in world height units per cell. Result is
    a new array; input is untouched.
    """
    if iters <= 0:
        return h
    h = h.astype(np.float32).copy()
    for _ in range(iters):
        up = np.empty_like(h)
        up[1:] = h[:-1]
        up[0] = h[0]
        dn = np.empty_like(h)
        dn[:-1] = h[1:]
        dn[-1] = h[-1]
        lf = np.empty_like(h)
        lf[:, 1:] = h[:, :-1]
        lf[:, 0] = h[:, 0]
        rt = np.empty_like(h)
        rt[:, :-1] = h[:, 1:]
        rt[:, -1] = h[:, -1]
        o_up = np.maximum(h - up - talus, 0.0) * factor
        o_dn = np.maximum(h - dn - talus, 0.0) * factor
        o_lf = np.maximum(h - lf - talus, 0.0) * factor
        o_rt = np.maximum(h - rt - talus, 0.0) * factor
        out = o_up + o_dn + o_lf + o_rt
        i_up = np.zeros_like(h)
        i_up[:-1] = o_up[1:]
        i_dn = np.zeros_like(h)
        i_dn[1:] = o_dn[:-1]
        i_lf = np.zeros_like(h)
        i_lf[:, :-1] = o_lf[:, 1:]
        i_rt = np.zeros_like(h)
        i_rt[:, 1:] = o_rt[:, :-1]
        h = h - out + (i_up + i_dn + i_lf + i_rt)
    return h


@numba.njit(cache=True, fastmath=True)
def _hydraulic_erode_numba(h, sediment, n_droplets, rng_seed,
                           inertia=0.05, capacity=4.0, deposition_rate=0.3,
                           erosion_rate=0.3, evaporation=0.01,
                           gravity=4.0, min_slope=0.01, max_lifetime=30,
                           initial_water=1.0, initial_speed=1.0):
    """Particle-based hydraulic erosion (in-place on h, accumulates in sediment).

    Ported from xarray-spatial/erosion.py and SebLague/Hydraulic-Erosion.
    Uses a local LCG for deterministic per-chunk results. Bilinear splat
    for erosion/deposition over 4 neighbouring cells (radius-1 brush).
    """
    H, W = h.shape
    state = np.uint64(rng_seed)
    for _ in range(n_droplets):
        # LCG random position
        state = (state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407))
        pos_x = float(state >> 33) / float(1 << 31) * (W - 3) + 1.0
        state = (state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407))
        pos_y = float(state >> 33) / float(1 << 31) * (H - 3) + 1.0
        dir_x = 0.0
        dir_y = 0.0
        speed = initial_speed
        water = initial_water
        sed = 0.0
        for _step in range(max_lifetime):
            node_x = int(pos_x)
            node_y = int(pos_y)
            if node_x < 1 or node_x >= W - 2 or node_y < 1 or node_y >= H - 2:
                break
            fx = pos_x - node_x
            fy = pos_y - node_y
            h00 = h[node_y, node_x]
            h10 = h[node_y, node_x + 1]
            h01 = h[node_y + 1, node_x]
            h11 = h[node_y + 1, node_x + 1]
            grad_x = (h10 - h00) * (1.0 - fy) + (h11 - h01) * fy
            grad_y = (h01 - h00) * (1.0 - fx) + (h11 - h10) * fx
            h_old = h00 * (1.0 - fx) * (1.0 - fy) + h10 * fx * (1.0 - fy) + h01 * (1.0 - fx) * fy + h11 * fx * fy
            dir_x = dir_x * inertia - grad_x * (1.0 - inertia)
            dir_y = dir_y * inertia - grad_y * (1.0 - inertia)
            dir_len = (dir_x * dir_x + dir_y * dir_y) ** 0.5
            if dir_len < 1e-10:
                break
            dir_x /= dir_len
            dir_y /= dir_len
            new_x = pos_x + dir_x
            new_y = pos_y + dir_y
            if new_x < 1 or new_x >= W - 2 or new_y < 1 or new_y >= H - 2:
                break
            n_nx = int(new_x)
            n_ny = int(new_y)
            n_fx = new_x - n_nx
            n_fy = new_y - n_ny
            h_new = (h[n_ny, n_nx] * (1.0 - n_fx) * (1.0 - n_fy) +
                     h[n_ny, n_nx + 1] * n_fx * (1.0 - n_fy) +
                     h[n_ny + 1, n_nx] * (1.0 - n_fx) * n_fy +
                     h[n_ny + 1, n_nx + 1] * n_fx * n_fy)
            h_diff = h_new - h_old
            speed_sq = speed * speed + (-h_diff) * gravity
            if speed_sq < 0.01:
                speed_sq = 0.01
            speed = speed_sq ** 0.5
            water *= (1.0 - evaporation)
            sed_cap = max(-h_diff, min_slope) * speed * water * capacity
            if sed > sed_cap or h_diff > 0:
                if h_diff > 0:
                    amount = min(h_diff, sed)
                else:
                    amount = (sed - sed_cap) * deposition_rate
                # Deposit: raise terrain, track sediment
                h[node_y, node_x] += amount * (1.0 - fx) * (1.0 - fy)
                h[node_y, node_x + 1] += amount * fx * (1.0 - fy)
                h[node_y + 1, node_x] += amount * (1.0 - fx) * fy
                h[node_y + 1, node_x + 1] += amount * fx * fy
                sediment[node_y, node_x] += amount * (1.0 - fx) * (1.0 - fy)
                sediment[node_y, node_x + 1] += amount * fx * (1.0 - fy)
                sediment[node_y + 1, node_x] += amount * (1.0 - fx) * fy
                sediment[node_y + 1, node_x + 1] += amount * fx * fy
                sed -= amount
            else:
                # Erode: lower terrain (no sediment tracking)
                amount = min((sed_cap - sed) * erosion_rate, h_old)
                h[node_y, node_x] -= amount * (1.0 - fx) * (1.0 - fy)
                h[node_y, node_x + 1] -= amount * fx * (1.0 - fy)
                h[node_y + 1, node_x] -= amount * (1.0 - fx) * fy
                h[node_y + 1, node_x + 1] -= amount * fx * fy
                sed += amount
            pos_x = new_x
            pos_y = new_y


def _hydraulic_erode(h, n_droplets=2000, seed=0, **kwargs):
    """Run particle-based hydraulic erosion on a 2D heightfield.

    The erosion algorithm assumes slope magnitudes on the order of ~1 per
    cell. Real-world heightfields with large vertical ranges (tens of
    meters per ~0.7 m cell) cause runaway erosion/deposition and spiky
    artifacts, so we normalize the heightfield to a unit range before
    eroding and scale back afterwards.

    Returns (eroded_height, sediment_deposition) arrays. Input is untouched.
    """
    h_out = h.astype(np.float32).copy()
    sediment = np.zeros_like(h_out)
    h_range = 1.0  # default; overwritten if erosion runs
    if n_droplets > 0:
        # Normalize to unit height range to keep erosion parameters stable.
        h_min = float(h_out.min())
        h_max = float(h_out.max())
        h_range = max(h_max - h_min, 1.0)
        h_out = (h_out - h_min) / h_range
        _hydraulic_erode_numba(h_out, sediment, n_droplets, np.uint64(seed), **kwargs)
        h_out = h_out * h_range + h_min
    return h_out, sediment * h_range


def _wind_erode(h, desert_mask, iters=3, wind_angle=0.5, strength=0.15):
    """Anisotropic smoothing along a wind axis for desert dune formation.

    Elongates terrain features along the wind direction, producing
    dune-like ridges instead of isotropic noise. Only affects cells
    where ``desert_mask`` > 0. Input is untouched.
    """
    if iters <= 0:
        return h
    h = h.astype(np.float32).copy()
    wx = int(round(np.cos(wind_angle) * 2.0))
    wz = int(round(np.sin(wind_angle) * 2.0))
    if wx == 0 and wz == 0:
        return h
    mask = desert_mask.astype(np.float32)
    for _ in range(iters):
        h_fwd = np.roll(h, shift=(wx, wz), axis=(0, 1))
        h_bwd = np.roll(h, shift=(-wx, -wz), axis=(0, 1))
        h_avg = (h_fwd + h_bwd) * 0.5
        h = h + (h_avg - h) * strength * mask
    return h


def smooth_terrain(h, strength=0.3, iters=1):
    """Gaussian smoothing pass to flatten tiny spikes and sharp edges.
    
    Applies a 3x3 Gaussian kernel ([1,2,1],[2,4,2],[1,2,1]/16) `iters` times,
    then blends the result with the original by `strength`. Seam-safe when
    run on the extended grid (padding absorbs roll-wrap artifacts at edges).
    """
    if strength <= 0.0 or iters <= 0:
        return h
    h_out = h.astype(np.float32).copy()
    for _ in range(iters):
        # Separable Gaussian: horizontal then vertical pass.
        # Horizontal: [1, 2, 1] / 4
        h_l = np.roll(h_out, 1, axis=1)
        h_r = np.roll(h_out, -1, axis=1)
        h_x = (h_l + 2.0 * h_out + h_r) * 0.25
        # Vertical: [1, 2, 1] / 4
        h_u = np.roll(h_x, 1, axis=0)
        h_d = np.roll(h_x, -1, axis=0)
        h_blur = (h_u + 2.0 * h_x + h_d) * 0.25
        h_out = h_out * (1.0 - strength) + h_blur * strength
    return h_out
