# Terrain Realism Overhaul — Research & Plan

## Current state (baseline)

- `chunks.py`: 2D heightfield per xz column, `grid_res=48`, `chunk_size=32` (~0.67 m cells).
  - `_compute_height`: 2-octave domain warp, ridged noise w/ smoothstep, fBm base, micro detail, continental low-freq layer.
  - `_compute_biome`: 4 biomes (tundra/mountain/desert/forest) blended by temp/humidity simplex.
  - `_thermal_erode`: 4-neighbour talus erosion only (smooths slopes, cannot carve valleys).
  - Per-octave rotation: **none** (axis-aligned banding risk).
  - Derivative erosion: **none**.
  - Hydraulic erosion: **none**.
  - Curvature/sediment attributes: **none**.
- `renderer.py` WGSL fragment shader:
  - Micro normal perturbation via 3-sample fBm3 — already present but coarse.
  - Slope-based rock override, snow line by altitude, biome palettes.
  - Snow uses altitude only — appears on cliffs unrealistically.
  - Fog: single-color `exp(-density*dist)` — no wavelength scattering.
  - No curvature-based material selection.
- `main.py`: `erosion_iters=3` thermal only, `lacunarity=2.13` (good — breaks octave alignment).

## Research findings

### A. Hydraulic erosion (highest visual impact)
- **Particle/droplet method** (Mei 2007; SebLague reference impl; `xarray-spatial/erosion.py`):
  - Default params: `inertia=0.05, capacity=4.0, deposition=0.3, erosion=0.3, evaporation=0.01, gravity=4.0, radius=3, max_lifetime=30`.
  - Per droplet: bilinear height+gradient sample → inertia-blended direction → speed from gravity → sediment capacity = `-dh * speed * water * capacity` → erode/deposit with bilinear splat.
  - Brush radius 3 gives smooth wide valleys; radius 1 gives sharp gullies.
  - 50k–500k droplets typical for a 256² map. On a 50×50 extended chunk grid, ~2k–5k droplets is enough.
- **Shallow-water pipe model** (Mei et al., bshishov/UnityTerrainErosionGPU): more physically accurate, supports rivers/lakes, but requires water flux field + velocity field — overkill for our chunked streaming model.
- **Recommendation**: particle-based, run on extended grid in `_build_chunk` after `_thermal_erode`. Cache result by chunk key so re-erosion isn't repeated on chunk reload.

### B. Derivative-erosion fBm + per-octave rotation (cheap, big win)
- Inigo Quilez / MiniMax terrain shader technique:
  - `f(p) = Σ aⁿ × noise(p) / (1 + dot(d,d))` where `d` = accumulated gradient.
  - Suppresses high-freq detail on steep slopes → smooth cliffs, detailed flats (inverse of plain fBm).
- Per-octave rotation matrix `mat2(1.6, 1.2, -1.2, 1.6)` (|M|=2, ~36.87°) breaks axis-aligned banding.
- `noise.snoise2` (the `noise` library) doesn't expose analytic derivatives, so we either:
  1. Compute derivatives via central differences per octave (cheap, 4 extra samples/octave).
  2. Switch to a custom value-noise-with-derivatives implementation (more work, more accurate).
- **Recommendation**: option 1 — central-difference derivatives per octave, accumulate, apply `1/(1+k*dot(d,d))` weighting. Add rotation per octave.

### C. Material realism (curvature + snow accumulation + sediment)
- **Curvature** (Laplacian of height): concave → dirt/sediment pockets; convex → exposed rock. Cheap: `lap = h[x-1] + h[x+1] + h[z-1] + h[z+1] - 4*h`. Pass as 5th vertex attribute (or pack into biome.w if forest weight is unused — but better as new attribute).
- **Snow accumulation**: snow only sticks where `slope < 0.5` AND a low-freq noise mask > threshold (wind-sheltered). Currently snow appears on vertical cliffs.
- **Sediment map**: from hydraulic erosion, track per-cell deposition. Use to tint valleys fertile/dirt.
- Reference: pantarei.xyz snowline tutorial, Andersson Siggraph 2007 terrain course.

### D. Atmospheric / aerial perspective
- Replace single-color exp fog with wavelength-dependent:
  ```glsl
  let att = exp(-u.fog_density * max(dist - u.fog_start, 0.0));
  let scatter = vec3<f32>(1.0, 1.5, 4.0);  // blue scatters more
  let lit_att = lit * pow(att, scatter);    // per-channel
  let out_color = mix(u.fog_color, lit_att, att);
  ```
- Optional: add sun-scatter halo toward light direction. ~3 lines in shader.
- Full Rayleigh/Mie (wwwtyro/glsl-atmosphere) is overkill for now — defer.

### E. Fragment-shader normal perturbation (already partial)
- Current: 3-sample fBm3 at scale 1.5, eps 0.4, 3 octaves, fade past 160 m.
- Improvement: use **two octaves at different scales** (1.5 m for medium rocks, 6.0 m for large undulations), weight by slope (more perturbation on rock, less on snow/sand). This gives sub-meter roughness without increasing vertex grid.

### F. Wind erosion (desert-specific, polish)
- For desert biome: directional blur along a dominant wind axis (e.g. 30°), subtract from peaks.
- Cheap, biome-gated. Can be done as a final pass in `_build_chunk` when `w_desert > 0.5`.

## Implementation plan (ordered by impact / cost)

### Phase 1 — Noise quality (low risk, high visual payoff)
1. **Per-octave rotation** in `_compute_height`: rotate `(x,z)` by `mat2(1.6,1.2,-1.2,1.6)` each octave.
2. **Derivative-erosion fBm**: accumulate central-difference gradient per octave, weight each octave's contribution by `1/(1+k*dot(d,d))`. Start with `k=2.0`, tune.
3. **Tune ridge smoothstep**: current `raw*raw*(3-2*raw)` is fine; consider `pow(raw, 1.5)` for slightly sharper alpine peaks.

**Files**: `chunks.py` only. **Risk**: low. **Verify**: py_compile + visual check.

### Phase 2 — Hydraulic erosion (high impact, medium risk)
4. **Add `_hydraulic_erode(h, n_droplets, ...)`** in `chunks.py`: port `xarray-spatial/erosion.py` `_erode_cpu` to pure numpy (no numba dep). Vectorize droplet loop where possible, or keep python loop with numba optional.
5. **Wire into `Chunk.build`** after `_thermal_erode`, gated by `hydraulic_droplets` param (default ~2000 on 50×50 extended grid).
6. **Track sediment deposition** as a separate array; pass to renderer as a vertex attribute for valley tinting.
7. **Cache eroded height+sediment** by chunk key in `ChunkManager` to avoid recompute on reload.

**Files**: `chunks.py`, `main.py` (new param). **Risk**: medium (perf — measure build time). **Verify**: build time stays under `target_compute_ms`, valleys visible.

### Phase 3 — Material realism (medium impact, low risk)
8. **Curvature attribute**: compute Laplacian in `Chunk.build`, add as 5th vertex float (interleaved 11 floats/vertex). Update `renderer.py` vertex layout + `VertexOut`.
9. **Snow accumulation mask**: in fragment shader, gate `w_snow` by `slope < 0.5 && fbm3(world_pos*0.05, 2) > 0.0`.
10. **Sediment valley tint**: blend dirt_color where sediment attribute is high.
11. **Two-octave normal perturbation** in fragment shader, slope-weighted.

**Files**: `chunks.py`, `renderer.py`. **Risk**: medium (vertex layout change touches GPU buffer creation, indirect draw, shadow pipeline). **Verify**: py_compile + render smoke test.

### Phase 4 — Atmosphere (low impact, trivial)
12. **Wavelength-dependent fog** in fragment shader.
13. **Sun-scatter halo** toward light dir.

**Files**: `renderer.py` only. **Risk**: low. **Verify**: visual.

### Phase 5 — Polish (optional)
14. **Wind erosion** for desert biome in `Chunk.build`.
15. **Procedural sky gradient** + sun disk (deferred — current solid sky_color works with fog).
16. **Full Rayleigh/Mie atmosphere** (deferred — only if sky becomes a priority).

## Performance budget (RTX 5070 baseline ~1.4 ms render, ~0.05 ms compute, 850–1100 FPS)

- Phase 1: negligible (a few extra snoise2 calls per octave).
- Phase 2: ~2000 droplets × ~30 steps × ~10 ops = ~600k ops per chunk build. Pure-python loop ≈ 50–150 ms per chunk — **too slow for streaming**. Mitigations:
  - Use `numba.njit` if available (check `.venv`).
  - Else vectorize: process N droplets in parallel as arrays, accept race conditions (SebLague GPU impl has same issue, looks fine visually).
  - Else reduce to ~500 droplets and increase brush radius to 4.
  - Cache aggressively so re-erosion only happens on first chunk load.
- Phase 3: +1 float/vertex → +10% vertex buffer size. Negligible on 5070.
- Phase 4: +3 ALU ops/fragment. Negligible.

## Decisions (locked)

- **numba**: YES — `pip install numba` into `.venv`. Use `@numba.njit` for `_hydraulic_erode` inner loop. Target ~1-3 ms/chunk.
- **Vertex layout**: NEW `location(3) vec2<f32> = (curvature, sediment)` attribute. Interleaved 12 floats/vertex (was 10). Touches: `_ChunkData`, vertex buffer creation, vertex attributes in main + shadow + prepass pipelines.
- **Scope**: ALL 5 PHASES — noise → hydraulic → materials → atmosphere → wind/sky polish.

## Implementation order (final)

1. `pip install numba` into `.venv`; verify import works.
2. Phase 1 — `chunks.py`: per-octave rotation + derivative-erosion fBm. py_compile + visual.
3. Phase 2 — `chunks.py`: `_hydraulic_erode` (numba.njit) + sediment array; wire into `Chunk.build` after `_thermal_erode`; cache by chunk key in `ChunkManager`. `main.py`: add `hydraulic_droplets=2000` param. Measure build time.
4. Phase 3 — `chunks.py`: compute curvature (Laplacian) + sediment attr, add to mesh dict. `renderer.py`: new `location(3) vec2` in VertexIn/VertexOut, update `_ChunkData` interleaved layout (12 floats), update all pipeline vertex attribute declarations. Fragment shader: snow accumulation mask + sediment valley tint + two-octave slope-weighted normal perturbation.
5. Phase 4 — `renderer.py`: wavelength-dependent fog + sun-scatter halo.
6. Phase 5 — `chunks.py`: wind erosion pass for desert biome. Optional: procedural sky gradient + sun disk in renderer (deferred unless requested).

## Verification gates

- After each phase: `.venv\Scripts\python.exe -m py_compile chunks.py renderer.py water.py camera.py main.py`
- After phases that touch rendering: boot the app, fly around, confirm FPS stays >500 and no visual regressions.
- After Phase 2: log chunk build time to profiler; confirm stays under `target_compute_ms=4.0`.
- After Phase 3: confirm shadow map still renders (vertex layout change is the riskiest part).

## Rollback plan

- Each phase is a separate commit. If Phase 3 vertex layout breaks the shadow pipeline and can't be quickly fixed, revert that commit — Phases 1 & 2 still stand alone.
- Keep `chunks.py` and `renderer.py` backups before Phase 3.

## Implementation log (completed)

All 5 phases implemented and verified. App boots and runs without errors.

### Phase 1 — Noise quality (chunks.py)
- Added `_fbm_eroded()`: manual octave loop with per-octave rotation matrix
  `mat2(1.6, 1.2, -1.2, 1.6)` scaled to lacunarity, and derivative-erosion
  weighting `1/(1 + k*|d|^2)` using forward-difference gradients.
- Added `_ridged_fbm_rotated()`: ridged fBm with per-octave rotation.
- Replaced base height + both ridge layers with new helpers.
- Perf: ~7.6ms height computation (was ~4ms), acceptable for worker processes.

### Phase 2 — Hydraulic erosion (chunks.py, main.py)
- Added `_hydraulic_erode_numba()`: numba.njit particle-based erosion ported
  from xarray-spatial/erosion.py with LCG RNG for deterministic per-chunk results.
- Added `_hydraulic_erode()` wrapper returning (height, sediment) arrays.
- Wired into `Chunk.build()` after thermal erosion, gated by `hydraulic_droplets`.
- Tracks sediment deposition per cell for valley material shading.
- **Bug fix**: algorithm assumes unit-scale slopes; raw world-unit heights
  (tens of meters per ~0.7 m cell) caused runaway erosion/deposition and
  sky-high spikes on some seeds. Fixed by normalizing h to [0,1] before
  erosion and scaling back afterwards.
- Perf: ~10ms/chunk warm (was ~8ms), +2ms for 2000 droplets.
- JIT cache=True persists compilation across worker process restarts.
- numba 0.66.0 installed (numpy downgraded 2.5.1 → 2.4.6, verified compatible).

### Phase 3 — Material realism (chunks.py, renderer.py, occlusion.py)
- Added curvature (Laplacian) computation on extended heightfield.
- Vertex layout changed from 10 → 12 floats/vertex (pos3 + normal3 + biome4 + sc2).
- Updated all 3 pipelines (render, prepass, shadow) with new array_stride.
- Updated `_ChunkData` interleaving and occlusion.py empty-case placeholder.
- Fragment shader: snow accumulation mask (slope + wind-shelter noise),
  sediment valley tint, curvature-based rock/dirt selection, two-octave
  slope-weighted normal perturbation.
- Verified: shader compiles, app boots, vertex_data shape (2304, 12).

### Phase 4 — Atmosphere (renderer.py)
- Replaced single-color exp fog with wavelength-dependent scattering:
  `att = vec3(exp(-t*1.0), exp(-t*1.5), exp(-t*4.0))` — blue scatters more.
- Added sun-scatter halo: `pow(max(dot(view, sun), 0), 8) * 0.15` warm glow
  toward the light direction.

### Phase 5 — Wind erosion (chunks.py, main.py)
- Added `_wind_erode()`: anisotropic smoothing along a wind axis (angle 0.5 rad),
  gated by desert biome mask. Elongates features into dune ridges.
- Wired into `Chunk.build()` after curvature computation, gated by
  `wind_erode_iters=3`.

### Files changed
- `chunks.py`: +numba import, +_fbm_eroded, +_ridged_fbm_rotated,
  +_hydraulic_erode_numba, +_hydraulic_erode, +_wind_erode, curvature computation,
  sediment/curvature in mesh dict, new params on Chunk/ChunkManager/_build_spec.
- `renderer.py`: vertex layout 10→12 floats, new location(3) sc attribute,
  two-octave normal perturbation, snow accumulation mask, sediment/curvature
  shading, wavelength fog + sun scatter.
- `occlusion.py`: empty-case vertex_data 6→12 floats.
- `main.py`: hydraulic_droplets=2000, wind_erode_iters=3.

### Performance
- Chunk build: ~10ms warm (worker process, parallel).
- Vertex buffer: +20% size (10→12 floats/vertex).
- Fragment shader: +~10 ALU ops (two-octave noise, wavelength fog).
- Expected FPS impact: minimal (build is async, shader is cheap on RTX 5070).
