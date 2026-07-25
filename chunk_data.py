"""Internal CPU-side chunk mesh data used to build merged GPU buffers."""
import numpy as np


class ChunkRecord:
    """What the renderer keeps for a chunk once its mesh is on the GPU.

    Only the bounding box is still needed (debug raycast, shadow bounds); the
    vertex/index arrays are handed to the GPU arena and dropped, which is worth
    a few hundred MB of resident memory at typical view distances.
    """

    __slots__ = ("bbox",)

    def __init__(self, bbox):
        self.bbox = bbox


class _ChunkData:
    """Internal CPU data for one chunk, used to build merged GPU buffers."""

    def __init__(self, mesh, bbox=None):
        if bbox is None:
            vertices = mesh["vertices"]
            vmin = vertices.min(axis=0)
            vmax = vertices.max(axis=0)
            self.bbox = (
                float(vmin[0]),
                float(vmin[1]),
                float(vmin[2]),
                float(vmax[0]),
                float(vmax[1]),
                float(vmax[2]),
            )
        else:
            self.bbox = bbox  # (min_x, min_y, min_z, max_x, max_y, max_z)
        vertices = mesh["vertices"]
        normals = mesh["normals"]
        indices = mesh["indices"]
        # Biome weights are optional so we can still upload legacy meshes that
        # predate the biome system (defaults to all-zero -> shader treats as
        # no biome contribution, which the caller must handle).
        biome = mesh.get("biome")
        sediment = mesh.get("sediment")
        curvature = mesh.get("curvature")
        self.index_count = indices.size
        self.vertex_count = vertices.shape[0]
        # Interleave position (3) + normal (3) + biome (4) + sc (2) = 12 floats/vertex.
        vertex_data = np.zeros((self.vertex_count, 12), dtype=np.float32)
        vertex_data[:, :3] = vertices
        vertex_data[:, 3:6] = normals
        if biome is not None:
            vertex_data[:, 6:10] = biome
        if sediment is not None:
            vertex_data[:, 10] = sediment
        if curvature is not None:
            vertex_data[:, 11] = curvature
        self.vertex_data = vertex_data
        self.index_data = np.ascontiguousarray(indices, dtype=np.uint32)
