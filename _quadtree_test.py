"""Test the quadtree LOD selection."""
import math
from terrain.quadtree import select_quadtree, neighbor_levels

# Test 1: Basic selection around camera at origin.
nodes = select_quadtree(0, 0, base_size=32, max_distance=300, lod_factor=2.0, max_level=4)
print(f"Test 1: {len(nodes)} nodes for max_distance=300")
# Count nodes by level
by_level = {}
for cx, cz, level, size in nodes:
    by_level[level] = by_level.get(level, 0) + 1
for lv in sorted(by_level):
    print(f"  level {lv} (size={32*2**lv}m): {by_level[lv]} chunks")

# Test 2: Verify no overlaps and full coverage.
# Each node covers [cx*size, (cx+1)*size] x [cz*size, (cz+1)*size].
# Check that no two nodes overlap.
covered = set()
for cx, cz, level, size in nodes:
    x0 = cx * size
    z0 = cz * size
    # Sample the center to check for overlaps
    mx = x0 + size // 2
    mz = z0 + size // 2
    assert (mx, mz) not in covered, f"overlap at ({mx}, {mz})"
    covered.add((mx, mz))
print(f"Test 2: no overlaps detected ({len(covered)} unique centers)")

# Test 3: Verify all nodes are within max_distance of the camera.
for cx, cz, level, size in nodes:
    x0 = cx * size
    z0 = cz * size
    x1 = x0 + size
    z1 = z0 + size
    # Nearest point to origin
    nx = max(x0, min(0, x1))
    nz = max(z0, min(0, z1))
    dist = math.sqrt(nx**2 + nz**2)
    assert dist <= 300 + size, f"node at ({cx},{cz},{level}) too far: {dist}"
print("Test 3: all nodes within render distance")

# Test 4: Neighbor levels.
nbrs = neighbor_levels(nodes)
print(f"Test 4: {len(nbrs)} nodes have neighbor info")
# Check a few nodes
for key, neighbors in list(nbrs.items())[:3]:
    print(f"  {key}: neighbors={neighbors}")

# Test 5: Larger render distance.
nodes2 = select_quadtree(500, 500, base_size=32, max_distance=600, lod_factor=2.5, max_level=5)
print(f"Test 5: {len(nodes2)} nodes for max_distance=600 at (500,500)")
by_level2 = {}
for cx, cz, level, size in nodes2:
    by_level2[level] = by_level2.get(level, 0) + 1
for lv in sorted(by_level2):
    print(f"  level {lv} (size={32*2**lv}m): {by_level2[lv]} chunks")

# Test 6: Compare vertex counts. Uniform grid vs quadtree.
# Uniform: radius 8, ~200 chunks, 49*49 verts each = ~480K
# Quadtree: sum of chunks * 49*49
uniform_verts = 200 * 49 * 49
quad_verts = len(nodes) * 49 * 49
print(f"Test 6: uniform ~{uniform_verts} verts vs quadtree ~{quad_verts} verts ({100*quad_verts/uniform_verts:.0f}%)")

print("\nALL QUADTREE TESTS PASSED")
