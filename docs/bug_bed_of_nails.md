# Bug Log: Bed-of-Nails Terrain

## Symptom
Terrain appeared as a "bed of nails" — dense, spiky peaks covering the entire
landscape instead of natural rolling hills and mountains.

## Metrics (48x48 chunk, seed 42, main.py defaults)
| Metric          | Before  | After   | Change |
|-----------------|---------|---------|--------|
| mean slope      | 1.68    | 1.03    | -39%   |
| max slope       | 6.44    | 2.84    | -56%   |
| % cells slope>2 | 32.5%   | 2.6%    | -92%   |

A mean slope of 1.68 means the average cell was steeper than 45°, and one-third
of cells were steeper than 63°. After the fix, only 2.6% of cells exceed 63°.

## Root Causes (3 confirmed)

### 1. Domain warp ~10x too strong (PRIMARY cause)
`_compute_height` applied a two-octave domain warp before sampling the base
noise. The warp displacement was `(wx1 + 0.3*wx2) * wp` where:
- `wx1` = 6-octave fBm, range ~[-2, 2]
- `wx2` = 2-octave fBm, range ~[-1.5, 1.5]
- `wp` = biome-modulated `_BIOME_WARP` (mountain=2.4) or `warp_amp` arg (4.0)

Total displacement: up to 2.45 * 2.4 = **5.9 noise-input units** (biome) or
2.45 * 4.0 = **9.8 noise-input units** (main.py default). The base noise has
wavelength ~1.0 in noise-input space, so the warp displaced samples by **6-10
full wavelengths**. Adjacent world-space points sampled completely uncorrelated
noise values, producing white-noise-level chaos — the bed of nails.

No amount of derivative erosion or thermal erosion could fix this because the
chaos was baked in at the sample-coordinate level, before any octave stacking.

**Fix:**
- `_BIOME_WARP`: `[0.6, 2.4, 1.2, 1.8]` → `[0.06, 0.24, 0.12, 0.18]` (10x reduction)
- `main.py` `warp_amp`: `4.0` → `0.4` (10x reduction)

Now max displacement is ~0.59 noise-input units (biome) or ~0.98 (arg) — under
one wavelength, producing gentle warping instead of chaos.

### 2. Base fBm had no derivative-erosion (SECONDARY cause)
`_compute_height` used `_vnoise2(x, z, 7, 0.5, 2.13)` — plain fBm with 7
octaves. High-frequency octaves added full amplitude on already-steep slopes,
accumulating sharp features. The project docs (`AGENTS.md`,
`terrain_overhaul_plan.md`) claimed `_fbm_eroded` was already implemented, but
it did not exist in the source.

**Fix:** Added `_fbm_eroded(x, z, octaves, persistence, lacunarity, k=2.0)`:
- Manual octave loop calling `_vnoise2(..., 1, ...)` per octave
- Forward-difference gradients (eps=0.1) accumulated into `d`
- Per-octave weight `1/(1 + k*|d|^2)` suppresses detail on steep slopes
- No per-octave rotation (rotation broke gradient alignment in testing —
  gradients from differently-rotated frames don't coherently accumulate on
  steep slopes, so suppression becomes uniform instead of slope-targeted)

### 3. Thermal erosion was disabled (TERTIARY cause)
`Chunk.build` had a comment: "Per-chunk erosion (thermal/hydraulic/wind) was
removed because it breaks chunk seams." The erosion functions existed but were
never called. `main.py` still passed `erosion_iters=3` etc., but the params
were dead code. Thermal erosion is the classic cure for bed-of-nails (Musgrave,
World Machine) — it rounds sharp peaks and produces natural talus slopes.

**Fix:** Re-enabled thermal erosion on a padded extended grid:
- `pad = max(erosion_iters, 4) + 1` cells of padding on each side
- Erode the full extended grid, then slice the interior
- Seam-safe because `_compute_height` is a pure function of world position and
  the padding is large enough that mesh-border erosion doesn't reach the
  clamped edges (both chunks compute identical values on shared border cells)
- Hydraulic and wind erosion remain disabled (hydraulic uses random
  per-chunk droplets that can't match across seams; wind's reach exceeds
  practical padding). Their visual effects stay in the fragment shader.

## Other Changes

### Ridged fBm: smootherstep + per-octave rotation
Replaced the inline `1-|noise|` + smoothstep ridge with `_ridged_fbm_rotated`:
- Per-octave rotation breaks axis-aligned ridge banding
- Smootherstep `6t^5-15t^4+10t^3` softens crests more than smoothstep
- Normalized by sum of amplitudes for consistent [0,1] range

### Reduced _BIOME_RIDGE
`[0.02, 0.38, 0.30, 0.12]` → `[0.02, 0.24, 0.20, 0.10]` — softer ridges to
further reduce spike density.

### Fragment shader micro-roughness toned down
`renderer.py`:
- Micro normal perturbation: `0.15/0.08` → `0.10/0.05`
- Gully normal perturbation: `0.08` → `0.05`

Reduces shimmer on close-up surfaces from sub-lattice noise perturbation.

## Files Changed
- `chunks.py`: Added `_fbm_eroded`, `_ridged_fbm_rotated`; rewrote base/ridge
  layers in `_compute_height`; reduced `_BIOME_WARP` and `_BIOME_RIDGE`;
  re-enabled thermal erosion in `Chunk.build` with padded extended grid.
- `main.py`: `warp_amp` 4.0 → 0.4.
- `renderer.py`: Reduced micro/gully normal perturbation amplitudes.

## Verification
- `py_compile` passes on all modules.
- 48x48 chunk slope stats: mean 1.03, max 2.84, 2.6% >2 (was 1.68/6.44/32.5%).
- Thermal erosion seam-safety: pure height function + padding >= iters+1
  guarantees identical border values across neighbouring chunks.
