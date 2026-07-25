# Bug Log: Occlusion Draw Crash When No Chunks Loaded

## Symptom
On startup in LOD (quadtree) mode, the first few frames crashed with:

```
AttributeError: 'NoneType' object has no attribute 'size'
```

at `render/occlusion.py` line 317:

```python
prepass.set_vertex_buffer(0, self.vertex_buffer, 0, self.vertex_buffer.size)
```

The error repeated for several frames until chunks finished building, then
rendering proceeded normally. In uniform-grid mode the crash did not occur
because chunks built fast enough to populate the arena before the first draw.

## Root Cause
The GPU mesh arena (`gpu_arena.py`) lazily allocates its vertex and index
buffers on the first chunk upload. Before any chunk is uploaded,
`Occlusion.vertex_buffer` and `Occlusion.index_buffer` are `None`.

`Occlusion.draw()` did not check for this state and unconditionally called
`set_vertex_buffer` / `set_index_buffer` with the None buffers, causing the
AttributeError when accessing `.size`.

In LOD mode, chunks take longer to appear on the first frame because:
1. The quadtree selects ~100 chunks on the first update.
2. With `max_builds_per_frame=6`, only 6 start building immediately.
3. The first build per worker process incurs ~3.8s numba JIT compilation
   (or ~0.37s if disk-cached).
4. Until the first chunk completes, the arena has no buffers.

## Fix
Added an early return in `Occlusion.draw()` when the arena buffers are not yet
allocated:

```python
def draw(self, encoder, color_view):
    # If no chunks have been uploaded yet, the arena buffers are None.
    # Skip the draw entirely; the renderer will clear the framebuffer.
    if self.vertex_buffer is None or self.index_buffer is None:
        return
    max_count = max(1, self.chunk_count)
    # ...
```

The shadow pass in `renderer.py` was already guarded by
`if self.occlusion.chunk_count > 0:`, so it did not need changes.

## Files Touched
- `render/occlusion.py` — early return when arena buffers are None.

## Verification
- `py_compile` passes for `render/occlusion.py`.
- App starts cleanly in LOD mode: no AttributeError on startup.
- First few frames show sky color (no terrain) until chunks load, then
  terrain fades in normally.
