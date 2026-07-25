# Project Notes

## Environment

- Use the project virtual environment: `.venv\Scripts\python.exe`.
- System `python` may not have the required dependencies installed.

## Running

```powershell
.venv\Scripts\python.exe main.py
```

- The renderer forces the Vulkan backend.
- The app opens a fullscreen Vulkan window.

## Verification

Before running, compile changed modules:

```powershell
.venv\Scripts\python.exe -m py_compile renderer.py shaders.py chunk_data.py occlusion.py water.py camera.py main.py
```

## Recent Architecture

- `terrain/` package builds 2D heightfield chunks (one chunk per xz column, y=0) instead of dense marching-cubes volumes.
  - `terrain/chunk.py`: `Chunk.build()` — heightfield mesh + skirts (vectorized perimeter loop).
  - `terrain/fbm.py`: derivative-eroded fBm, ridged fBm with per-octave rotation, height field composition.
  - `terrain/features.py`: rivers, plateaus, canyons, craters, glacial valleys (all seam-safe, numba-jit'd).
  - `terrain/erosion.py`: thermal, hydraulic (numba), wind erosion + Gaussian smoothing.
  - `terrain/manager.py`: `ChunkManager` — async process-pool chunk building, streaming, caching.
  - `terrain/biomes.py`, `terrain/_noise_core.py`: biome blending + numba simplex noise core.
  - **Numba-jit'd 2D simplex noise** (`_snoise2_scalar`, `_snoise2_grid`, `_fbm_grid`) — bit-identical to `noise.snoise2`, ~10x faster than `np.vectorize`. See `docs/feature_log_perf_biomes_pbr.md`.
  - Derivative-erosion fBm (`_fbm_eroded` / `_fbm_eroded_grid`) with slope-aware weighting `1/(1+k*|d|^2)` to suppress high-frequency detail on steep slopes.
  - Ridged fBm (`_ridged_fbm_rotated` / `_ridged_fbm_grid`) with per-octave rotation and smootherstep peaks for mountain ridges.
  - **4 large-scale terrain features** (all seam-safe, pure functions of world position, numba-jit'd):
    - Rivers (`_apply_rivers`): V-valley carving along noise-based river network
    - Plateaus (`_apply_plateaus`): flat-top mesas with cliff edges
    - Canyons (`_apply_canyons`): deep narrow channels with layered walls
    - Craters (`_apply_craters`): deterministic hash-placed impact craters with raised rims
  - Thermal (`_thermal_erode`) erosion on padded extended grid (seam-safe). Hydraulic and wind erosion disabled — visual effects handled in fragment shader.
  - 4-biome blending (tundra/mountain/desert/forest) via temperature/humidity simplex noise. 4 additional biomes (beach/savanna/swamp/volcanic) computed in-shader.
  - Per-vertex (packed, 32 bytes): position(float32x3) + normal(snorm8x4) + biome(unorm8x4) + sediment_curvature(float16x2) + 8 bytes pad. Was 48 bytes (12 floats).
  - Domain warp amplitude must stay under ~1.0 noise-input units (see `docs/bug_bed_of_nails.md`).
- `shaders.py`: WGSL shader source (`SHADER` = terrain PBR shader, `BBOX_SHADER` = position-only depth shader). Extracted from renderer.py.
- `chunk_data.py`: `_ChunkData` — CPU-side chunk mesh data used to build merged GPU buffers. Extracted from renderer.py.
- `renderer.py`: `TerrainRenderer` — pipeline creation, uniform updates, shadow MVP, draw orchestration. Imports shaders and chunk data from their own modules.
  - **PBR lighting** (Cook-Torrance BRDF): GGX normal distribution, Smith geometry, Schlick Fresnel, Lambert diffuse with energy conservation, metallic workflow.
  - Per-material roughness/metallic/AO: snow smooth, basalt rough, lava metallic, rock rough.
  - **Material details**: sand ripples (wind-aligned, desert/beach), rock strata (sedimentary banding on cliffs), snow SSS (wrap diffuse + forward scatter).
  - Two-octave slope-weighted normal perturbation, snow accumulation mask, sediment valley tint, curvature-based rock/dirt, wavelength-dependent aerial perspective fog with sun-scatter halo.
- `occlusion.py`: `Occlusion` — GPU-driven HZB occlusion culling (prepass → HZB build → cull compute → main pass). Multi-line WGSL shaders (COPY/REDUCE/CULL). Main pass clears depth (not load) to avoid prepass depth leaking culled-chunk geometry into visible chunks (see `docs/bug_occlusion_seam_holes.md`).
- `water.py` adds a large, transparent, camera-following water plane with procedural waves.
- `camera.py` includes `orthographic()` used for the light projection matrix.

## Dependencies

- `numba` (0.66.0): JIT-compiled simplex noise, fBm, erosion, and terrain features. Requires numpy < 2.5.
- First chunk build per worker process incurs ~3.8s JIT compilation (all numba functions); subsequent
  builds use disk cache (`cache=True`). Warm builds: 1.4ms (no features) / 1.8ms (all features).
- **Async prewarm**: the main process pre-compiles all numba functions on a background thread
  (`_prewarm_numba` in `main.py`) while the window shows a loading screen. Worker processes
  then load from disk cache. Chunk manager creation is deferred until prewarm completes.

## Performance Baseline (RTX 5070, 32GB RAM)

- After numba noise + PBR + terrain features: ~880-950 FPS, ~2ms render time, ~0.05ms compute time, ~8% VRAM, ~3-4% GPU.
- Chunk build: 1.4ms warm (no features), 1.8ms warm (all 4 features). Was 12ms before numba.
- Vertex buffer: 32 bytes/vertex (packed: pos f32x3 + normal snorm8x4 + biome unorm8x4 + sc f16x2 + pad). Was 48 bytes. Fragment shader: PBR adds ~15 ALU ops, material details add ~10.
