"""Persistent sub-allocated GPU buffers for streamed chunk meshes.

The previous design concatenated every loaded chunk and recreated all GPU
buffers whenever a single chunk streamed in or out. At radius 10 / grid_res 48
that is ~37 MB of vertices plus ~18 MB of indices rebuilt per streaming event,
which showed up as ~21 ms frame spikes during camera panning.

Instead we keep one long-lived vertex buffer and one long-lived index buffer,
divided into fixed-size slots. Adding a chunk writes only that chunk's own
region; removing one just returns its slot to a free list. Nothing else moves.

Two independent indices are in play and must not be confused:

  * geometry slot  - where a chunk's vertices/indices physically live. Stable
                     for the lifetime of the chunk so its data never has to be
                     re-uploaded.
  * dense index    - the chunk's position in the AABB / indirect-draw arrays.
                     These must stay gap-free because the cull compute shader
                     dispatches over `chunk_count` consecutive entries, so a
                     removal is handled by swapping the last entry down.
"""

import numpy as np
import wgpu

# Draw-call metadata written per chunk, matching wgpu's indexed-indirect layout:
# (index_count, instance_count, first_index, base_vertex, first_instance).
_INDIRECT_FIELDS = 5
# vec4 min + vec4 max, matching `array<vec4<f32>>` in the cull shader.
_AABB_FIELDS = 8


class MeshArena:
    """Slot-allocated vertex/index storage plus dense cull metadata."""

    def __init__(self, device, vertex_floats: int = 12, initial_slots: int = 384):
        self.device = device
        self.vertex_floats = vertex_floats
        self.vertex_itemsize = vertex_floats * 4
        self.capacity = max(1, int(initial_slots))

        # Per-slot capacity, sized from the first chunk we see. Uniform grid_res
        # means every chunk is the same size in practice; a larger one triggers
        # a re-layout rather than silently corrupting neighbouring slots.
        self.slot_verts = 0
        self.slot_indices = 0

        self.free_slots = list(range(self.capacity))
        self.dense_of_key = {}      # chunk key -> dense index
        self.key_of_dense = []      # dense index -> chunk key
        self.slot_of_dense = []     # dense index -> geometry slot
        self.bbox_of_key = {}       # chunk key -> bbox tuple (for raycast/shadow)

        self.aabb_cpu = np.zeros((self.capacity, _AABB_FIELDS), dtype=np.float32)
        self.indirect_cpu = np.zeros((self.capacity, _INDIRECT_FIELDS), dtype=np.uint32)
        self._count_cpu = np.zeros(1, dtype=np.uint32)

        # Dirty range over dense indices, flushed once per frame.
        self._dirty_lo = None
        self._dirty_hi = None
        self._count_dirty = True
        # Bumped whenever a buffer object is replaced, so the owner knows to
        # rebuild any bind groups that referenced the old buffer.
        self.generation = 0

        self.vertex_buffer = None
        self.index_buffer = None
        self._create_meta_buffers()

    # -- properties ---------------------------------------------------------

    @property
    def chunk_count(self):
        return len(self.key_of_dense)

    # -- buffer management --------------------------------------------------

    def _create_meta_buffers(self):
        self.aabb_buffer = self.device.create_buffer(
            size=self.capacity * _AABB_FIELDS * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.indirect_buffers = [
            self.device.create_buffer(
                size=self.capacity * _INDIRECT_FIELDS * 4,
                usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_DST,
            ) for _ in range(2)
        ]
        self.prepass_indirect_buffer = self.device.create_buffer(
            size=self.capacity * _INDIRECT_FIELDS * 4,
            usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST,
        )
        self.count_buffer = self.device.create_buffer(
            size=4, usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST,
        )
        self.generation += 1

    def _create_geometry_buffers(self, slot_verts, slot_indices, capacity):
        """Allocate geometry storage, migrating existing slots on the GPU.

        Old contents are moved with copy_buffer_to_buffer so we never need a
        CPU-side copy of the mesh data to survive a resize.
        """
        old_vb, old_ib = self.vertex_buffer, self.index_buffer
        old_slot_verts, old_slot_indices = self.slot_verts, self.slot_indices

        new_vb = self.device.create_buffer(
            size=capacity * slot_verts * self.vertex_itemsize,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC,
        )
        new_ib = self.device.create_buffer(
            size=capacity * slot_indices * 4,
            usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC,
        )

        if old_vb is not None and self.slot_of_dense:
            encoder = self.device.create_command_encoder()
            for slot in self.slot_of_dense:
                encoder.copy_buffer_to_buffer(
                    old_vb, slot * old_slot_verts * self.vertex_itemsize,
                    new_vb, slot * slot_verts * self.vertex_itemsize,
                    old_slot_verts * self.vertex_itemsize,
                )
                encoder.copy_buffer_to_buffer(
                    old_ib, slot * old_slot_indices * 4,
                    new_ib, slot * slot_indices * 4,
                    old_slot_indices * 4,
                )
            self.device.queue.submit([encoder.finish()])

        self.vertex_buffer = new_vb
        self.index_buffer = new_ib
        self.slot_verts = slot_verts
        self.slot_indices = slot_indices
        self.generation += 1

        # Slot stride changed, so every existing entry's base_vertex and
        # first_index now point at the wrong place. Recompute them (index_count
        # is per-chunk and unaffected).
        if (slot_verts, slot_indices) != (old_slot_verts, old_slot_indices):
            for dense, slot in enumerate(self.slot_of_dense):
                self.indirect_cpu[dense, 2] = slot * slot_indices
                self.indirect_cpu[dense, 3] = slot * slot_verts
            if self.slot_of_dense:
                self._mark_dirty(0, len(self.slot_of_dense) - 1)

    def _grow(self, need_verts, need_indices):
        """Expand slot capacity and/or per-slot size to fit a new chunk."""
        slot_verts = max(self.slot_verts, int(need_verts))
        slot_indices = max(self.slot_indices, int(need_indices))
        capacity = self.capacity
        if not self.free_slots:
            capacity = max(1, capacity * 2)

        if capacity != self.capacity:
            self.free_slots.extend(range(self.capacity, capacity))
            aabb = np.zeros((capacity, _AABB_FIELDS), dtype=np.float32)
            aabb[:self.capacity] = self.aabb_cpu
            self.aabb_cpu = aabb
            indirect = np.zeros((capacity, _INDIRECT_FIELDS), dtype=np.uint32)
            indirect[:self.capacity] = self.indirect_cpu
            self.indirect_cpu = indirect
            self.capacity = capacity
            self._create_meta_buffers()
            # Metadata buffers were replaced, so re-upload everything in them.
            self._mark_dirty(0, max(0, self.chunk_count - 1))
            self._count_dirty = True

        if (slot_verts != self.slot_verts or slot_indices != self.slot_indices
                or self.vertex_buffer is None
                or capacity * slot_verts * self.vertex_itemsize > self.vertex_buffer.size):
            self._create_geometry_buffers(slot_verts, slot_indices, capacity)

    # -- dirty tracking -----------------------------------------------------

    def _mark_dirty(self, lo, hi):
        if self._dirty_lo is None:
            self._dirty_lo, self._dirty_hi = lo, hi
        else:
            self._dirty_lo = min(self._dirty_lo, lo)
            self._dirty_hi = max(self._dirty_hi, hi)

    def _write_dense(self, dense, slot, index_count, bbox):
        self.aabb_cpu[dense] = 0.0
        self.aabb_cpu[dense, :3] = bbox[:3]
        self.aabb_cpu[dense, 4:7] = bbox[3:]
        self.indirect_cpu[dense] = (
            index_count, 1, slot * self.slot_indices, slot * self.slot_verts, 0,
        )
        self._mark_dirty(dense, dense)

    # -- public API ---------------------------------------------------------

    def add(self, key, vertex_data, index_data, bbox):
        """Upload one chunk's mesh into a free slot.

        `vertex_data` must be (V, vertex_floats) float32 and `index_data` a
        flat uint32 array. Neither is retained after this call.
        """
        if key in self.dense_of_key:
            self.remove(key)

        vert_count = vertex_data.shape[0]
        index_count = index_data.size
        if (not self.free_slots or vert_count > self.slot_verts
                or index_count > self.slot_indices):
            self._grow(vert_count, index_count)

        slot = self.free_slots.pop()
        dense = len(self.key_of_dense)
        self.dense_of_key[key] = dense
        self.key_of_dense.append(key)
        self.slot_of_dense.append(slot)
        self.bbox_of_key[key] = bbox

        queue = self.device.queue
        queue.write_buffer(
            self.vertex_buffer, slot * self.slot_verts * self.vertex_itemsize,
            np.ascontiguousarray(vertex_data, dtype=np.float32),
        )
        queue.write_buffer(
            self.index_buffer, slot * self.slot_indices * 4,
            np.ascontiguousarray(index_data, dtype=np.uint32),
        )
        self._write_dense(dense, slot, index_count, bbox)
        self._count_dirty = True

    def remove(self, key):
        """Free a chunk's slot, swapping the last dense entry into its place."""
        dense = self.dense_of_key.pop(key, None)
        if dense is None:
            return
        self.bbox_of_key.pop(key, None)
        self.free_slots.append(self.slot_of_dense[dense])
        last = len(self.key_of_dense) - 1

        if dense != last:
            # Move the tail entry down so the dense range stays gap-free.
            moved_key = self.key_of_dense[last]
            moved_slot = self.slot_of_dense[last]
            self.key_of_dense[dense] = moved_key
            self.slot_of_dense[dense] = moved_slot
            self.dense_of_key[moved_key] = dense
            self.aabb_cpu[dense] = self.aabb_cpu[last]
            self.indirect_cpu[dense] = self.indirect_cpu[last]
            self._mark_dirty(dense, dense)

        self.key_of_dense.pop()
        self.slot_of_dense.pop()
        self._count_dirty = True

    def flush(self):
        """Upload pending metadata changes. Call once per frame before drawing."""
        if self._dirty_lo is not None:
            lo, hi = self._dirty_lo, min(self._dirty_hi, self.capacity - 1)
            if hi >= lo:
                queue = self.device.queue
                aabb = np.ascontiguousarray(self.aabb_cpu[lo:hi + 1])
                queue.write_buffer(self.aabb_buffer, lo * _AABB_FIELDS * 4, aabb)
                indirect = np.ascontiguousarray(self.indirect_cpu[lo:hi + 1])
                offset = lo * _INDIRECT_FIELDS * 4
                for buf in self.indirect_buffers:
                    queue.write_buffer(buf, offset, indirect)
                queue.write_buffer(self.prepass_indirect_buffer, offset, indirect)
            self._dirty_lo = self._dirty_hi = None

        if self._count_dirty:
            self._count_cpu[0] = self.chunk_count
            self.device.queue.write_buffer(self.count_buffer, 0, self._count_cpu)
            self._count_dirty = False
