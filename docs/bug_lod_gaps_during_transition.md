# Bug Log: LOD Gaps During Chunk Transitions — Immediate Stale Removal

## Symptom
In LOD (quadtree) mode, when the camera moved far enough to trigger a quadtree
re-evaluation, visible gaps appeared in the terrain for 1-3 frames before the
new chunks finished building. The gaps showed the sky color through areas that
should have been covered by terrain, creating a "flashing" effect at the
boundary between LOD levels.

## Root Cause
In `terrain/manager.py::ChunkManager.update()`, chunks that left the needed set
were removed immediately from both the chunk dict and the renderer:

```python
removed = [key for key in self.chunks if key not in needed]
for key in removed:
    del self.chunks[key]
```

The replacement chunks (at a different LOD level covering the same area) are
built asynchronously in a process pool and take 1.4-1.8ms each to build. With
`max_builds_per_frame=6`, rebuilding a full LOD ring can take several frames.
During that time, the old chunk is already gone from the renderer, leaving a
gap.

This is specific to LOD mode: in uniform-grid mode, chunks only leave the
needed set when they fall outside the radius, so the gap is at the view edge
and not noticeable. In LOD mode, chunks leave the needed set when they're
*replaced* by a coarser or finer chunk covering the same central area, so the
gap is visible.

## Fix
Implemented delayed stale-chunk removal for LOD mode. Chunks that leave the
needed set are marked stale and kept in the renderer until a new chunk
overlapping their area is built and uploaded:

```python
if self.use_lod:
    new_stale = {key: 0 for key in self.chunks if key not in needed}
    stale = getattr(self, '_stale_chunks', {})
    for key in new_stale:
        if key not in stale:
            stale[key] = 0
    # Safety valve: remove after 60 frames even if not replaced
    removed = [key for key, age in stale.items() if age >= max_stale_age]
    # ...
    self._stale_chunks = stale
```

When a new chunk finishes building, any stale chunks whose world-space bounds
overlap the new chunk's bounds are removed:

```python
# In the completed-builds section:
if self.use_lod and hasattr(self, '_stale_chunks'):
    # Compute new chunk's bounds, check overlap with stale chunks
    # Remove overlapping stale chunks
```

A safety valve removes stale chunks after 60 frames (~1 second at 60 FPS) even
if no replacement arrives, preventing unbounded stale accumulation when moving
fast.

## Files Touched
- `terrain/manager.py` — delayed stale removal + overlap-based cleanup.

## Verification
- `py_compile` passes for `terrain/manager.py`.
- `_lod_manager_test.py` passes: chunks transition without gaps.
- Visual: no flashing during LOD transitions when moving.
- Memory: stale chunks bounded by 60-frame safety valve.
