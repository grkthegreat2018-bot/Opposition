"""Internal CPU-side chunk mesh data used to build merged GPU buffers."""
import numpy as np


class ChunkRecord:
    """What the renderer keeps for a chunk once its mesh is on the GPU.

    Only the bounding box is still needed (debug raycast, shadow bounds); the
    vertex/index arrays are handed to the GPU arena and dropped. Measured at
    radius 10 / grid_res 48 that is ~57 MB of resident memory (469 -> 413 MB);
    the remaining footprint is Python/numba/wgpu runtime, not chunk data.
    """

    __slots__ = ("bbox",)

    def __init__(self, bbox):
        self.bbox = bbox


# Packed vertex stride (bytes). Layout:
#   pos    : float32x3   (12 bytes, offset 0)  -- world space, full precision
#   normal : snorm8x4    ( 4 bytes, offset 12) -- xyz unit vector, w unused
#   biome  : unorm8x4    ( 4 bytes, offset 16) -- 4 biome weights, sum to 1
#   sc     : float16x2   ( 4 bytes, offset 20) -- sediment, curvature
#   pad    :             ( 8 bytes, offset 24) -- align to 32 for cache lines
# Total: 32 bytes/vertex (was 48). Saves ~33% VRAM and vertex fetch BW.
VERTEX_STRIDE = 32


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
        self.vertex_data = _pack_vertices(
            vertices, normals, biome, sediment, curvature, self.vertex_count)
        self.index_data = np.ascontiguousarray(indices, dtype=np.uint32)


def _pack_vertices(vertices, normals, biome, sediment, curvature, count):
    """Build the packed 32-byte/vertex byte buffer.

    Returns a contiguous uint8 array of length count*VERTEX_STRIDE ready for
    queue.write_buffer. Position keeps full float32; normals are quantized to
    snorm8 (precision ~1/127), biome weights to unorm8 (~1/255), and the
    sediment/curvature pair to float16 (~3 decimal digits, range +-65504).
    """
    out = np.zeros((count, VERTEX_STRIDE), dtype=np.uint8)
    # pos: float32x3 at offset 0
    pos_view = out[:, :12].view(np.float32).reshape(count, 3)
    pos_view[:] = vertices
    # normal: snorm8x4 at offset 12. Clip to [-1,1] then map to [-127,127].
    n = np.clip(normals, -1.0, 1.0)
    nq = np.empty((count, 4), dtype=np.int8)
    nq[:, :3] = np.round(n * 127.0).astype(np.int8)
    nq[:, 3] = 0
    out[:, 12:16] = nq.view(np.uint8)
    # biome: unorm8x4 at offset 16. Clip to [0,1] then map to [0,255].
    if biome is not None:
        b = np.clip(biome, 0.0, 1.0)
        bq = np.round(b * 255.0).astype(np.uint8)
    else:
        bq = np.zeros((count, 4), dtype=np.uint8)
    out[:, 16:20] = bq
    # sc: float16x2 at offset 20.
    sc = np.zeros((count, 2), dtype=np.float16)
    if sediment is not None:
        sc[:, 0] = sediment.astype(np.float16)
    if curvature is not None:
        sc[:, 1] = curvature.astype(np.float16)
    out[:, 20:24] = sc.view(np.uint8).reshape(count, 4)
    # bytes 24..31 left zero (pad to 32-byte stride).
    return np.ascontiguousarray(out)
