"""Quadtree LOD selection for terrain chunks.

Replaces the uniform-grid chunk selection with a distance-based quadtree.
Near the camera, chunks are small (level 0 = 32m, grid_res=48). Far from
the camera, chunks are larger (level 1 = 64m, level 2 = 128m, etc.) with
the same grid_res, so distant terrain uses fewer vertices per area.

The quadtree is evaluated each frame:
  1. Start with a square region centered on the camera, sized to cover
     the desired render distance.
  2. Recursively subdivide nodes that are too close for their size.
  3. The result is a set of non-overlapping (cx, cz, level) nodes that
     tile the visible area without gaps.

LOD level for a node at distance d from the camera:
  level = clamp(floor(log2(d / (base_size * lod_factor))), 0, max_level)

A node of level L has size = base_size * 2^L. It should be subdivided if
its nearest point to the camera is closer than size * lod_factor, i.e.
if a lower level (smaller chunk) would be appropriate.

Seam handling: when a high-res chunk (level L) is adjacent to a low-res
chunk (level L+1 or higher), the high-res chunk has more edge vertices.
The simplest approach is extended skirts (the high-res skirt drops lower
to cover the height delta). A better approach is vertex thinning (drop
every other edge vertex on the high-res side to match the low-res edge
resolution). This module handles selection; seam handling is in chunk.py.
"""
import math
import numpy as np


def select_quadtree(cam_x, cam_z, base_size, max_distance, lod_factor,
                    max_level=4):
    """Select a set of quadtree nodes that cover the visible area.

    Returns a list of (cx, cz, level, size) tuples where:
      - (cx, cz) is the node's grid coordinate at its level
      - level is the LOD level (0 = finest, max_level = coarsest)
      - size is the node's world-space size in meters (= base_size * 2^level)

    The nodes tile the area without overlap. Each node's nearest point to
    the camera determines whether it should be subdivided.

    Parameters:
      cam_x, cam_z: camera world position
      base_size: level-0 chunk size in meters (e.g. 32.0)
      max_distance: render distance in meters (how far to draw terrain)
      lod_factor: controls how aggressively distant terrain is simplified.
        Higher = more aggressive (larger chunks closer to camera).
        Typical: 1.5-3.0. At 2.0, a chunk is subdivided only if its nearest
        point is within 2x its size from the camera.
      max_level: maximum LOD level (coarsest chunks). At level 4 with
        base_size=32, the largest chunk is 32*16 = 512m.
    """
    # The quadtree root covers a square region centered on the camera.
    # Its size must be a power-of-2 multiple of base_size, large enough to
    # cover max_distance. We use the camera's base-size grid cell as the
    # center and expand outward.
    root_size = base_size * (2 ** max_level)
    # Camera's position in root-level grid coordinates.
    root_cx = int(math.floor(cam_x / root_size))
    root_cz = int(math.floor(cam_z / root_size))

    # How many root-level nodes do we need to cover the view? A square
    # of (2*half+1) x (2*half+1) root nodes centered on the camera.
    half = int(math.ceil(max_distance / root_size)) + 1

    nodes = []
    for rz in range(root_cz - half, root_cz + half + 1):
        for rx in range(root_cx - half, root_cx + half + 1):
            _subdivide(rx, rz, max_level, base_size, lod_factor,
                       cam_x, cam_z, max_distance, nodes)

    return nodes


def _subdivide(cx, cz, level, base_size, lod_factor,
               cam_x, cam_z, max_distance, out):
    """Recursively subdivide a node if it's too close for its size.

    A node at level L has size = base_size * 2^L. It should be subdivided
    if its nearest point to the camera is within size * lod_factor AND
    level > 0 (can't subdivide below level 0).
    """
    size = base_size * (2 ** level)
    # Node's world-space bounds.
    x0 = cx * size
    z0 = cz * size
    x1 = x0 + size
    z1 = z0 + size

    # Nearest point on the node's AABB to the camera.
    nx = max(x0, min(cam_x, x1))
    nz = max(z0, min(cam_z, z1))
    dx = cam_x - nx
    dz = cam_z - nz
    dist = math.sqrt(dx * dx + dz * dz)

    # If the node is entirely beyond the render distance, skip it.
    # Use the farthest corner to check.
    fx = x0 if cam_x > x1 else x1
    fz = z0 if cam_z > z1 else z1
    far_dist = math.sqrt((cam_x - fx) ** 2 + (cam_z - fz) ** 2)
    if far_dist > max_distance + size:
        return

    # Should we subdivide? Subdivide if the nearest point is within
    # size * lod_factor of the camera (i.e. this chunk is "too close"
    # for its resolution).
    if level > 0 and dist < size * lod_factor:
        # Subdivide into 4 children.
        child_level = level - 1
        _subdivide(cx * 2, cz * 2, child_level, base_size, lod_factor,
                   cam_x, cam_z, max_distance, out)
        _subdivide(cx * 2 + 1, cz * 2, child_level, base_size, lod_factor,
                   cam_x, cam_z, max_distance, out)
        _subdivide(cx * 2, cz * 2 + 1, child_level, base_size, lod_factor,
                   cam_x, cam_z, max_distance, out)
        _subdivide(cx * 2 + 1, cz * 2 + 1, child_level, base_size, lod_factor,
                   cam_x, cam_z, max_distance, out)
    else:
        # Keep this node. Check if any part is within render distance.
        if far_dist <= max_distance or dist <= max_distance:
            out.append((cx, cz, level, size))


def neighbor_levels(nodes):
    """Build a map from (cx, cz, level) -> neighbor levels for seam handling.

    For each node, returns a dict of {direction: neighbor_level} where
    direction is 'N', 'S', 'E', 'W'. The neighbor level is the LOD level
    of the adjacent node in that direction. If no neighbor exists (edge
    of the selected area), the neighbor level is None.

    This is used by the chunk builder to thin edge vertices where a
    high-res chunk meets a low-res chunk.
    """
    node_set = set()
    for cx, cz, level, size in nodes:
        node_set.add((cx, cz, level))

    # For each node, find the neighbor in each direction at the same
    # or different level. The neighbor might be at a coarser level
    # (covering a larger area), so we need to check if any ancestor
    # of the adjacent position is in the set.
    result = {}
    for cx, cz, level, size in nodes:
        neighbors = {}
        for direction, (dx, dz) in [('E', (1, 0)), ('W', (-1, 0)),
                                     ('N', (0, -1)), ('S', (0, 1))]:
            # The neighbor at our level:
            ncx = cx + dx
            ncz = cz + dz
            nlevel = _find_neighbor_level(ncx, ncz, level, node_set)
            neighbors[direction] = nlevel
        result[(cx, cz, level)] = neighbors
    return result


def _find_neighbor_level(ncx, ncz, level, node_set):
    """Find the LOD level of the node at (ncx, ncz) at or above `level`.

    The neighbor might be at a coarser level (larger chunk). We search
    from our level up to the root, checking if any ancestor of the
    neighbor position exists in the node set.
    """
    # First check if there's a node at our exact level.
    if (ncx, ncz, level) in node_set:
        return level
    # Search coarser levels: at level L+1, the neighbor's coordinate
    # is (ncx // 2, ncz // 2).
    cx, cz, lv = ncx, ncz, level
    while lv < 64:  # safety limit
        lv += 1
        cx = cx // 2 if cx >= 0 else -((-cx - 1) // 2) - 1
        cz = cz // 2 if cz >= 0 else -((-cz - 1) // 2) - 1
        if (cx, cz, lv) in node_set:
            return lv
    return None
