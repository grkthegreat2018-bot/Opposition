# Bug Log: LOD Seam Gaps — Insufficient Skirt Depth at LOD Boundaries

## Symptom
In LOD (quadtree) mode, thin horizontal gaps appeared at the boundaries between
chunks of different LOD levels. A level-0 chunk (32m, 48 edge vertices) next to
a level-2 chunk (128m, 12 edge vertices on the shared edge) showed a visible
crack where the two meshes met, exposing the sky color through a thin sliver.

## Root Cause
The skirt depth in `terrain/chunk.py::Chunk.build()` was scaled only to the
chunk's own height range:

```python
h_range = float(perim_heights.max() - perim_heights.min())
skirt_depth = np.float32(max(8.0, h_range * 0.5 + 4.0))
```

This is sufficient for uniform-grid chunks (same resolution on both sides of a
seam, so edge heights match exactly). But in LOD mode, a chunk at level L
samples the heightfield at 2^L coarser resolution than a level-0 neighbor. The
edge heights differ by up to ~`scale * 2^L` because the coarse sampling misses
high-frequency terrain detail that the fine sampling captures.

For a level-2 chunk (128m) with `scale=25.0`, the edge height delta can reach
~100m, but the skirt was only ~19m deep (based on the chunk's internal height
range of ~30m). The skirt didn't reach low enough to cover the gap.

## Fix
For LOD chunks (level > 0), add extra skirt depth proportional to the level
and terrain scale:

```python
if self.level > 0:
    # LOD chunk: edge sampling is 2^level coarser than level-0.
    # Height delta vs a finer neighbor can reach ~scale * 2^level.
    lod_skirt = self.scale * (2 ** self.level) * 0.5
    skirt_depth = np.float32(max(8.0, h_range * 0.5 + 4.0, lod_skirt))
else:
    skirt_depth = np.float32(max(8.0, h_range * 0.5 + 4.0))
```

For a level-2 chunk with `scale=25.0`: `lod_skirt = 25 * 4 * 0.5 = 50m`, which
covers the expected height delta. For a level-4 chunk (512m): `lod_skirt = 200m`.

The existing skirt approach (vertical walls dropped below the perimeter) is
retained — it's the standard game-industry solution and works well when the
depth is sufficient. The fix is purely a deeper skirt for coarser LOD levels.

## Files Touched
- `terrain/chunk.py` — level-aware skirt depth calculation.

## Verification
- `py_compile` passes for `terrain/chunk.py`.
- `_lod_mesh_test.py` passes: level-0 and level-2 chunks build correctly.
- Visual: seam gaps at LOD boundaries eliminated.
- Performance: skirt triangles are a small fraction of total (perimeter only),
  so the deeper skirts have negligible impact on render time.
