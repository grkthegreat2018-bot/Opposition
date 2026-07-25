"""Test that LOD chunks actually build meshes correctly (no rendering)."""
import sys
sys.path.insert(0, '.')
import numpy as np
from terrain.chunk import Chunk

# Build a level-0 chunk (32m) and a level-2 chunk (128m) with same grid_res.
# Both should produce valid meshes; the level-2 chunk covers 4x the area
# with the same vertex count (lower resolution).
spec_base = dict(
    cx=0, cy=0, cz=0, size=32.0, grid_res=48, level=0,
    seed=42, scale=25.0, freq=0.008, octaves=6, persistence=0.5,
    lacunarity=2.0, warp_scale=0.3, warp_amp=3.0,
    ridge_weight=0.3, detail_weight=0.15, biome_freq=0.003,
    continental_freq=0.0015, sea_level=8.0, ocean_depth=12.0,
    land_boost=1.3, coastal_peak=0.65, coastal_width=0.08,
    coastal_mountain_strength=1.2, ocean_transition=0.12,
    ocean_detail_floor=0.3, erosion_iters=3, erosion_talus=0.6,
    erosion_factor=0.3, hydraulic_droplets=0, wind_erode_iters=0,
    river_depth=2.5, plateau_strength=0.6, canyon_depth=4.0,
    crater_strength=0.5, smooth_strength=0.0, glacial_strength=0.8,
)

# Level 0: 32m chunk.
c0 = Chunk(**spec_base)
m0 = c0.build()
print(f"Level 0 (32m): verts={m0['vertices'].shape[0]}, tris={m0['indices'].shape[0]}, build={c0.build_time*1000:.2f}ms")
print(f"  key={c0.key()}, bounds={c0.bounds()}")

# Level 2: 128m chunk (4x larger area, same grid_res).
spec_l2 = dict(spec_base)
spec_l2.update(size=128.0, level=2)
c2 = Chunk(**spec_l2)
m2 = c2.build()
print(f"Level 2 (128m): verts={m2['vertices'].shape[0]}, tris={m2['indices'].shape[0]}, build={c2.build_time*1000:.2f}ms")
print(f"  key={c2.key()}, bounds={c2.bounds()}")

# Verify both meshes have the same vertex count (same grid_res).
assert m0['vertices'].shape[0] == m2['vertices'].shape[0], "vertex count should match"
print(f"\nVertex count match: {m0['vertices'].shape[0]}")

# Verify heights are in a reasonable range.
v0 = m0['vertices']
v2 = m2['vertices']
print(f"Level 0 height range: [{v0[:,1].min():.2f}, {v0[:,1].max():.2f}]")
print(f"Level 2 height range: [{v2[:,1].min():.2f}, {v2[:,1].max():.2f}]")

# Verify the level-2 chunk covers 4x the xz extent.
x0_range = v0[:,0].max() - v0[:,0].min()
x2_range = v2[:,0].max() - v2[:,0].min()
print(f"Level 0 x-extent: {x0_range:.1f}m, Level 2 x-extent: {x2_range:.1f}m")
assert abs(x2_range - 128.0) < 1.0, f"level 2 should span 128m, got {x2_range}"
assert abs(x0_range - 32.0) < 1.0, f"level 0 should span 32m, got {x0_range}"

print("\nLOD MESH BUILD TEST PASSED")
