# Bug Log: Occlusion Prepass Depth Conflict → Chunk-Border Seam Holes

## Symptom
Visible holes/seams appeared along chunk borders, resembling "marching-cube skirt
seams". The gaps exposed the sky color through what should have been continuous
terrain, particularly noticeable where chunk skirts (the vertical walls dropped
`skirt_depth=8.0` below the perimeter to hide sub-pixel seams) met neighboring
chunks. The artifact appeared after occlusion culling was re-enabled.

## Root Cause
A depth-buffer conflict between the occlusion culling prepass and the main color
pass in `occlusion.py::Occlusion.draw()`.

The render flow was:
1. **Prepass** — drew **ALL** chunks (including soon-to-be-culled ones) into the
   depth buffer, with color cleared to the sky color. This seeds the HZB.
2. **HZB build + cull** — the depth pyramid was reduced from the prepass depth,
   and the compute cull pass marked some chunks invisible.
3. **Main pass** — drew only the *visible* chunks, but **loaded** the prepass
   depth (`depth_load_op: "load"`) and used `depth_compare: "less-equal"`.

The flaw: the prepass depth buffer still contained geometry from chunks that the
cull pass had just marked invisible — including their skirts. When a *visible*
chunk tried to rasterize a pixel where a *culled* chunk's skirt or surface had
been written closer in the prepass, the visible fragment **failed** the
`less-equal` depth test. The pixel retained the prepass's sky-clear color,
producing a hole along the shared border. Because skirts run the full perimeter,
the artifact manifested as continuous seam lines between chunks — visually
identical to broken marching-cube skirt stitching.

The skirt generation itself (`terrain/chunk.py`) was correct: all four perimeter
edges form a closed loop, and the heightfield is seam-safe (a pure function of
world position with extended-grid padding for erosion). The bug was purely a
rendering-side depth-state leak from the occlusion system.

## Fix
Two changes in `occlusion.py`:

### 1. Clear depth on the main color pass (primary fix)
Changed the main pass's `depth_load_op` from `"load"` to `"clear"`. The prepass
depth is only needed to build the HZB, which is already complete by the time the
main pass begins. Clearing depth ensures visible chunks render without competing
against culled chunks' prepass geometry. The color attachment still uses
`load_op: "load"` so the sky color written by the prepass is preserved for the
sky/cloud passes that follow.

```python
# Before
depth_stencil_attachment={"view":self.renderer.depth_view,
                          "depth_load_op":"load","depth_store_op":"store"}
# After
depth_stencil_attachment={"view":self.renderer.depth_view,
                          "depth_clear_value":1.0,
                          "depth_load_op":"clear","depth_store_op":"store"}
```

### 2. Loosen HZB depth tolerance (secondary fix)
Increased the cull shader's depth slack from `0.001` to `0.005`:
```wgsl
// Before
if (min_depth <= max_hiz + 0.001) { visible = true; }
// After
if (min_depth <= max_hiz + 0.005) { visible = true; }
```
This reduces false-negative culling at chunk borders, which would otherwise
expose neighboring chunks' skirts as visible vertical walls when a chunk is
barely occluded.

## Files Touched
- `occlusion.py` — main pass depth clear + HZB tolerance.

## Verification
- `py_compile` passes for `occlusion.py`, `renderer.py`, `terrain/chunk.py`, `main.py`.
- Visual: seam holes along chunk borders eliminated; skirts now correctly hidden
  behind neighboring terrain.
