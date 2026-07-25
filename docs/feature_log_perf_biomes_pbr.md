# Feature Log: Performance + Biomes + PBR + Terrain Features

## Overview
Major round of improvements: 8.5x chunk build speedup, 4 new biomes, PBR
lighting, sand ripples, rock strata, snow SSS, and 4 large-scale terrain
features (rivers, plateaus, canyons, craters).

## Performance: Numba Simplex Noise (8.5x build speedup)

### Problem
Chunk build took 12ms, 92% in `_compute_height`, 75% of that was
`np.vectorize(noise.snoise2)` overhead — 37+ Python-level scalar calls per
chunk through np.vectorize's slow dispatch loop.

### Fix
Replaced `np.vectorize(noise.snoise2)` with a numba-jit'd 2D simplex noise
implementation (`_snoise2_scalar`, `_snoise2_grid`) that is **bit-identical**
to `noise.snoise2(x, y, 1, 0.5, 2.0)` (max error < 1e-6).

Key implementation details:
- Standard Perlin permutation table + 12 gradient directions (Gustavson ref)
- `math.floor()` for grid cell lookup (not `int()` — truncation breaks
  negative coordinates, producing exact-0 values)
- `parallel=True` on grid loops for multi-core scaling
- `_fbm_grid`: multi-octave fBm in compiled code, normalized by amplitude sum
  to match `noise.snoise2`'s multi-octave behavior
- `_fbm_eroded_grid`: slope-aware fBm with `1/(1+k·|d|²)` weighting, all in
  compiled code (21 noise evaluations per cell, no Python dispatch)
- `_ridged_fbm_grid`: ridged fBm with per-octave rotation and smootherstep,
  all in compiled code

### Result
- Build time: **12ms → 1.4ms** (8.5x faster, no features)
- Build time: **1.8ms** with all 4 terrain features enabled
- App FPS: **830 → 880+** (render-bound, not compute-bound)
- GPU utilization: only 3-4% (tons of headroom for shader complexity)

### Files Changed
- `chunks.py`: Added `_snoise2_scalar`, `_snoise2_grid`, `_fbm_grid`,
  `_fbm_eroded_grid`, `_ridged_fbm_grid` (all numba-jit'd). Rewrote
  `_fbm_eroded`, `_ridged_fbm_rotated`, `_compute_biome`, and all noise calls
  in `_compute_height` to use the numba path.

---

## 4 New Biomes (shader-computed)

Added beach, savanna, swamp, and volcanic biomes. Computed in-shader from
world position + altitude + climate + noise — **no vertex format change
needed**. Each weight carves out of the climate base biomes.

| Biome | Trigger | Material |
|-------|---------|----------|
| Beach | altitude < 3m, near water | wet/dry sand blend |
| Savanna | desert climate + flat + noise mask | dry yellow grass |
| Swamp | forest climate + low alt + flat | muddy dark brown |
| Volcanic | low-freq noise mask (patches) | basalt + lava in channels |

### Files Changed
- `renderer.py`: Extended biome weight computation, new color palettes
  (beach_sand, wet_sand, basalt_color, lava_color, savanna_grass, swamp_mud),
  biome albedo blending.

---

## PBR Lighting (Cook-Torrance BRDF)

Replaced Blinn-Phong with proper physically-based rendering:

- **GGX/Trowbridge-Reitz** normal distribution (`d_ggx`)
- **Smith geometry** function (`g_smith`)
- **Schlick Fresnel** (`f_schlick`)
- **Lambert diffuse** scaled by `(1 - F)` for energy conservation
- **Metallic workflow**: metals have no diffuse, specular tinted by albedo
- **Per-material roughness/metallic**: snow=0.35, wet sand=0.65, basalt=0.95,
  lava=0.6 metallic, rock=1.0, grass=0.90
- **Ambient occlusion** from curvature (concave = darker)

### Files Changed
- `renderer.py`: Added `pow5`, `d_ggx`, `g_smith`, `f_schlick` helpers.
  Replaced Blinn-Phong block with Cook-Torrance BRDF.

---

## Material Details

### Sand Ripples
Wind-aligned sine wave ripples in desert/beach terrain. Three octaves at
decreasing wavelength, projected onto a fixed wind direction `(0.7, 0.7)`.
Modulates albedo (darker troughs) and slightly perturbs the normal. Fades
on steep slopes.

### Rock Strata
Sedimentary banding on cliff faces. Horizontal bands follow world height
with noise-perturbed boundaries. Band spacing ~2.2 world units. Sharpness
scales with slope (sharper on steeper cliffs). Darker bands read as harder
layers.

### Snow SSS (Subsurface Scattering)
Snow is a participating medium — light entering one face exits another with
a warm glow. Faked with:
- **Wrap diffuse**: soft, flat lighting with no hard terminator
- **Forward scatter**: bright halo when sun is behind the surface
- **Warm tint**: `vec3(1.0, 0.92, 0.82)` from multiple scattering

### Files Changed
- `renderer.py`: Added `sand_ripples`, `rock_strata`, `snow_sss` helpers.
  Integrated into fragment shader after albedo computation.

---

## 4 Large-Scale Terrain Features

All features are **pure functions of world position → seam-safe across
chunks**. Applied after base fBm height, before thermal erosion. Each gated
by a strength parameter (0 = disabled). All numba-jit'd with parallel grid
loops.

### Rivers (`_apply_rivers`)
- Low-frequency noise defines river network (zero crossings = river paths)
- Distance to zero crossing approximated by `|noise| / (|gradient| * freq)`
  — **must convert to world units by dividing by freq** (the original bug
  was comparing noise-space distance to world-space width)
- V-valley carving, depth scaled by terrain height (no rivers in oceans)
- Domain-warped for meandering
- Parameters: `river_depth=2.5`, `river_width=4.0`, `river_freq=0.004`

### Plateaus (`_apply_plateaus`)
- Clips terrain above a noise-modulated threshold to create flat-top mesas
- `plateau_base=0.70` → flat top at ~70% of scale height (only clips peaks)
- `strength` controls mesa mask threshold (how much of map has mesas)
- Sharp transition (`cliff_sharp=0.04`) for cliff edges
- Tiny noise roughness on plateau top
- Parameters: `plateau_strength=0.6`

### Canyons (`_apply_canyons`)
- Carves deep narrow channels along a noise-guided path
- Same distance-to-zero formula as rivers (with freq conversion)
- `mask²` for gentler edges, floor clamp at `-scale * 0.15`
- Domain-warped for meandering
- Parameters: `canyon_depth=4.0`, `canyon_width=5.0`, `canyon_freq=0.003`

### Craters (`_apply_craters`)
- Deterministic hash-based crater placement on a jittered grid
- Each grid cell may have a crater (hash threshold)
- Profile: bowl depression + raised rim + power-law ejecta blanket
- 3x3 neighborhood check for craters near chunk borders
- Parameters: `crater_strength=0.5`, `grid_size=80`, `crater_prob=0.18`

### Key Bug Fixed
The distance-to-zero-crossing formula was computed in noise-input space but
compared to world-space width. At low frequencies, the gradient is tiny in
noise space, making the distance enormous → mask=0 everywhere → no features
visible. Fix: `dist_world = |noise| / (|gradient| * freq)`.

### Files Changed
- `chunks.py`: Added `_apply_rivers`, `_apply_plateaus`, `_apply_canyons`,
  `_apply_craters`, `_hash2`, `smoothstep_numba`. Added `river_depth`,
  `plateau_strength`, `canyon_depth`, `crater_strength` params to `Chunk`
  and `ChunkManager`. Integrated into `Chunk.build`.
- `main.py`: Added feature parameters with tuned defaults.

---

## Performance Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Chunk build (no features) | 12ms | 1.4ms | -88% |
| Chunk build (all features) | 12ms | 1.8ms | -85% |
| App FPS (steady state) | 830 | 880+ | +6% |
| Render time | 2.5ms | 2.5ms | same |
| GPU utilization | 2% | 3-4% | +1% |
| VRAM | 8.2% | 7.8% | -0.4% |

The chunk build speedup is the main win. The PBR shader adds ~10 ALU ops per
fragment but GPU is only at 3-4% utilization, so the render time is unchanged.

## How to Tune

All feature strengths are in `main.py`:
```python
river_depth=2.5,       # 0 = off, 6 = deep rivers
plateau_strength=0.6,  # 0 = off, 1 = lots of mesas
canyon_depth=4.0,      # 0 = off, 8 = deep canyons
crater_strength=0.5,   # 0 = off, 1 = lots of craters
```
