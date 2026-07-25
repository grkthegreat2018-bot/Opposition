# Bug Log: Prewarm Crash — LOD kwargs Passed to Chunk.__init__

## Symptom
On startup with LOD enabled, the numba prewarm thread crashed with:

```
TypeError: Chunk.__init__() got an unexpected keyword argument 'use_lod'
```

The prewarm thread failed, but the app continued (the main thread caught the
exception). Worker processes still loaded numba from disk cache, so chunk
building worked, but the prewarm optimization was lost.

## Root Cause
`core/config.py::Config.chunk_manager_kwargs()` returns a flat dict of all
streaming config fields via `**asdict(s)`, including the new LOD fields:
`use_lod`, `lod_factor`, `lod_max_level`, `lod_render_distance`.

`Config.prewarm_chunk_kwargs()` builds its kwargs from
`chunk_manager_kwargs()` and pops the streaming-specific fields:

```python
for key in ("radius", "min_radius", "max_radius", "y_radius",
            "max_builds_per_frame", "target_compute_ms"):
    kwargs.pop(key, None)
```

The LOD fields were not in this pop list, so they remained in the kwargs dict
and were passed to `Chunk(**spec)`, which does not accept them.

## Fix
Added the LOD fields to the pop list in `prewarm_chunk_kwargs()`, and added
`level=0` as a default since prewarm builds a uniform-grid chunk:

```python
for key in ("radius", "min_radius", "max_radius", "y_radius",
            "max_builds_per_frame", "target_compute_ms",
            "use_lod", "lod_factor", "lod_max_level",
            "lod_render_distance"):
    kwargs.pop(key, None)
kwargs["size"] = kwargs.pop("chunk_size")
kwargs["grid_res"] = 8
kwargs.setdefault("level", 0)
```

## Files Touched
- `core/config.py` — pop LOD fields in prewarm_chunk_kwargs.

## Verification
- `py_compile` passes for `core/config.py`.
- App starts cleanly: "Numba pre-warm: 0.37s (cached to disk for workers)".
- No TypeError in prewarm thread.
