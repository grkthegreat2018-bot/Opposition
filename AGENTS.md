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
.venv\Scripts\python.exe -m py_compile render\renderer.py render\shaders.py render\chunk_data.py render\occlusion.py render\water.py core\camera.py main.py
```

## Recent Architecture

- `terrain/` package builds 2D heightfield chunks (one chunk per xz column, y=0) instead of dense marching-cubes volumes.
  - `terrain/chunk.py`: `Chunk.build()` — heightfield mesh + skirts (vectorized perimeter loop).
  - `terrain/fbm.py`: derivative-eroded fBm, ridged fBm with per-octave rotation, height field composition.
  - `terrain/features.py`: rivers, plateaus, canyons, craters, glacial valleys (all seam-safe, numba-jit'd).
  - `terrain/erosion.py`: thermal, hydraulic (numba), wind erosion + Gaussian smoothing.
  - `terrain/manager.py`: `ChunkManager` — async process-pool chunk building, streaming, caching. Supports uniform-grid and quadtree LOD modes (`use_lod` in `StreamingConfig`).
  - `terrain/quadtree.py`: `select_quadtree()` — distance-based quadtree LOD selection. Level-0 chunks (32m, 48x48 verts) near camera; coarser levels (up to 512m at level 4) far away. ~42% fewer verts than uniform grid for same render distance. Camera position quantized to 32m grid to prevent re-selection churn. Stale chunks retained until replacements built (prevents LOD-transition gaps).
  - `terrain/biomes.py`, `terrain/_noise_core.py`: biome blending + numba simplex noise core.
  - `terrain/continental.py`: continental-scale land/ocean mask, coastal mountain chains, ocean flatness (all seam-safe, numba-jit'd).
  - **Numba-jit'd 2D simplex noise** (`_snoise2_scalar`, `_snoise2_grid`, `_fbm_grid`) — bit-identical to `noise.snoise2`, ~10x faster than `np.vectorize`. See `docs/feature_log_perf_biomes_pbr.md`.
  - Derivative-erosion fBm (`_fbm_eroded` / `_fbm_eroded_grid`) with slope-aware weighting `1/(1+k*|d|^2)` to suppress high-frequency detail on steep slopes.
  - Ridged fBm (`_ridged_fbm_rotated` / `_ridged_fbm_grid`) with per-octave rotation and smootherstep peaks for mountain ridges.
  - **Continental multi-scale** (`terrain/continental.py`): very low-frequency mask [0,1] drives oceans (mask<0.4 → flat seafloor below sea level), coastal mountain chains (Gaussian peak at mask~0.65), and interior plains. All seam-safe, numba-jit'd.
  - **4 large-scale terrain features** (all seam-safe, pure functions of world position, numba-jit'd):
    - Rivers (`_apply_rivers`): V-valley carving along noise-based river network
    - Plateaus (`_apply_plateaus`): flat-top mesas with cliff edges
    - Canyons (`_apply_canyons`): deep narrow channels with layered walls
    - Craters (`_apply_craters`): deterministic hash-placed impact craters with raised rims
  - Thermal (`_thermal_erode`) erosion on padded extended grid (seam-safe). Hydraulic and wind erosion disabled — visual effects handled in fragment shader.
  - 4-biome blending (tundra/mountain/desert/forest) via temperature/humidity simplex noise. 4 additional biomes (beach/savanna/swamp/volcanic) computed in-shader.
  - Per-vertex (packed, 32 bytes): position(float32x3) + normal(snorm8x4) + biome(unorm8x4) + sediment_curvature(float16x2) + 8 bytes pad. Was 48 bytes (12 floats).
  - Domain warp amplitude must stay under ~1.0 noise-input units (see `docs/bug_bed_of_nails.md`).
- `render/shaders.py`: WGSL shader source (`SHADER` = terrain PBR shader, `BBOX_SHADER` = position-only depth shader). Extracted from renderer.py.
- `render/chunk_data.py`: `_ChunkData` — CPU-side chunk mesh data used to build merged GPU buffers. Extracted from renderer.py.
- `render/renderer.py`: `TerrainRenderer` — pipeline creation, uniform updates, shadow MVP, draw orchestration. Imports shaders and chunk data from their own modules.
  - **PBR lighting** (Cook-Torrance BRDF): GGX normal distribution, Smith geometry, Schlick Fresnel, Lambert diffuse with energy conservation, metallic workflow.
  - Per-material roughness/metallic/AO: snow smooth, basalt rough, lava metallic, rock rough.
  - **Material details**: sand ripples (wind-aligned, desert/beach), rock strata (sedimentary banding on cliffs), snow SSS (wrap diffuse + forward scatter).
  - Two-octave slope-weighted normal perturbation, snow accumulation mask, sediment valley tint, curvature-based rock/dirt, wavelength-dependent aerial perspective fog with sun-scatter halo.
- `render/occlusion.py`: `Occlusion` — GPU-driven HZB occlusion culling (prepass → HZB build → cull compute → main pass). Multi-line WGSL shaders (COPY/REDUCE/CULL). Main pass clears depth (not load) to avoid prepass depth leaking culled-chunk geometry into visible chunks (see `docs/bug_occlusion_seam_holes.md`).
- `render/water.py` adds a large, transparent, camera-following water plane with procedural waves.
- `core/camera.py` includes `orthographic()` used for the light projection matrix.
- `core/config.py`: frozen dataclass configuration (NoiseConfig, ErosionConfig, FeatureConfig, StreamingConfig, CameraConfig, FogConfig, DisplayConfig, TimeConfig).
- `core/profiler.py`: `PerformanceProfiler` — frame timing, memory tracking, chunk build stats.

## Dependencies

- `numba` (0.66.0): JIT-compiled simplex noise, fBm, erosion, and terrain features. Requires numpy < 2.5.
- First chunk build per worker process incurs ~3.8s JIT compilation (all numba functions); subsequent
  builds use disk cache (`cache=True`). Warm builds: 1.4ms (no features) / 1.8ms (all features).
- **Async prewarm**: the main process pre-compiles all numba functions on a background thread
  (`_prewarm_numba` in `main.py`) while the window shows a loading screen. Worker processes
  then load from disk cache. Chunk manager creation is deferred until prewarm completes.

## Performance Baseline (RTX 5070, 32GB RAM)

- After GPU arena + packed vertices + continental worldgen: ~1100-1300 FPS avg, ~350 FPS min (chunk streaming), ~0.7ms render time, ~0.15ms compute time, ~8% VRAM, ~8-9% GPU, ~420MB proc mem.
- After quadtree LOD (Phase 4a-c): ~1150-1250 FPS avg, ~310-360 FPS min, ~1600 peak, ~0.70-0.75ms render, ~0.07-0.20ms compute (LOD selection), ~9.1% VRAM, ~418MB proc mem. ~42% fewer verts than uniform grid for same render distance.
- Chunk build: 1.4ms warm (no features), 1.8ms warm (all 4 features). Was 12ms before numba.
- Vertex buffer: 32 bytes/vertex (packed: pos f32x3 + normal snorm8x4 + biome unorm8x4 + sc f16x2 + pad). Was 48 bytes. Fragment shader: PBR adds ~15 ALU ops, material details add ~10.
- Numba prewarm: ~0.36s cached, ~6.5s cold start. Runs async on a background thread with a loading screen.
