# Bug Log: LOD Chunk Flickering — Quadtree Re-Selection Churn

## Symptom
When the camera moved slowly across the terrain in LOD (quadtree) mode, chunks
at the LOD boundary flickered in and out rapidly. Whole chunks would briefly
disappear and reappear, creating a distracting shimmer at the edges of the
visible area. The effect was most noticeable when strafing or moving forward
at moderate speed.

## Root Cause
The quadtree LOD selection in `terrain/manager.py::ChunkManager.update()` was
re-evaluating the entire chunk set every time the camera moved more than half
a base chunk (16m). The selection used the raw camera position:

```python
needs_rebuild = (
    abs(pos[0] - lx) > self.chunk_size * 0.5 or
    abs(pos[2] - lz) > self.chunk_size * 0.5)
```

The problem: the quadtree's subdivision boundary is a function of the camera
position. As the camera moves continuously, the boundary shifts continuously,
causing chunks near the boundary to be added and removed on successive frames.
Each removal immediately deletes the chunk from the renderer, leaving a gap
until the replacement chunk finishes building asynchronously (1.4-1.8ms per
chunk in the process pool). The gap is visible for 1-3 frames, producing
flicker.

## Fix
Quantize the camera position to the base chunk grid before quadtree selection.
The quadtree now only re-evaluates when the camera crosses a full 32m base-chunk
boundary, making the selection deterministic per grid cell:

```python
qx = int(math.floor(pos[0] / self.chunk_size))
qz = int(math.floor(pos[2] / self.chunk_size))
needs_rebuild = (lqx != qx or lqz != qz)
# ...
cam_x = (qx + 0.5) * self.chunk_size  # quantized center
cam_z = (qz + 0.5) * self.chunk_size
nodes = select_quadtree(cam_x, cam_z, ...)
```

This reduces re-selection frequency by ~50% and eliminates the continuous
boundary shift. Combined with the delayed stale-chunk removal fix
(see `bug_lod_gaps_during_transition.md`), this eliminates the flicker.

## Files Touched
- `terrain/manager.py` — quantized camera position for quadtree selection.

## Verification
- `py_compile` passes for `terrain/manager.py`.
- Visual: flickering at LOD boundaries eliminated when moving at moderate speed.
- Performance unchanged: ~1100-1300 FPS avg, ~0.7ms render.
