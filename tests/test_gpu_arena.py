"""Correctness check for MeshArena slot/dense bookkeeping.

Runs against a fake device so it needs no GPU. Covers the variable-slot-size
re-layout path that per-chunk LOD resolutions will exercise.

    .venv\\Scripts\\python.exe tests\\test_gpu_arena.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gpu_arena
from gpu_arena import MeshArena, _INDIRECT_FIELDS, _AABB_FIELDS


class FakeBuffer:
    def __init__(self, size):
        self.size = size
        self.data = bytearray(size)


class FakeQueue:
    def write_buffer(self, buf, offset, data):
        raw = np.ascontiguousarray(data).tobytes()
        assert offset % 4 == 0, f"unaligned offset {offset}"
        assert offset + len(raw) <= buf.size, (
            f"overflow: offset {offset} + {len(raw)} > {buf.size}")
        buf.data[offset:offset + len(raw)] = raw

    def submit(self, _):
        pass


class FakeEncoder:
    def __init__(self):
        self.copies = []

    def copy_buffer_to_buffer(self, src, so, dst, do, size):
        assert so + size <= src.size and do + size <= dst.size, "copy out of range"
        dst.data[do:do + size] = src.data[so:so + size]
        self.copies.append((so, do, size))

    def finish(self):
        return self


class FakeDevice:
    def __init__(self):
        self.queue = FakeQueue()

    def create_buffer(self, size, usage=0, **kw):
        return FakeBuffer(size)

    def create_command_encoder(self):
        return FakeEncoder()


# wgpu.BufferUsage flags are ints in the real API; stub them out.
class _Usage:
    def __getattr__(self, name):
        return 1


gpu_arena.wgpu = type("m", (), {"BufferUsage": _Usage()})()


def make_chunk(vid, nverts=10, nidx=18):
    v = np.full((nverts, 12), float(vid), dtype=np.float32)
    i = np.arange(nidx, dtype=np.uint32)
    bbox = (vid, 0.0, vid, vid + 1.0, 5.0, vid + 1.0)
    return v, i, bbox


def check_invariants(a, label):
    n = a.chunk_count
    assert len(a.key_of_dense) == n and len(a.slot_of_dense) == n, label
    assert set(a.dense_of_key.values()) == set(range(n)), f"{label}: dense not gap-free"
    for k, d in a.dense_of_key.items():
        assert a.key_of_dense[d] == k, f"{label}: dense map mismatch"
    slots = a.slot_of_dense
    assert len(set(slots)) == len(slots), f"{label}: duplicate slot allocation!"
    assert not (set(slots) & set(a.free_slots)), f"{label}: slot both used and free!"
    # Every active chunk's indirect entry must point at its own slot.
    for d, slot in enumerate(slots):
        assert a.indirect_cpu[d, 3] == slot * a.slot_verts, f"{label}: bad base_vertex"
        assert a.indirect_cpu[d, 2] == slot * a.slot_indices, f"{label}: bad first_index"
        assert a.indirect_cpu[d, 1] == 1, f"{label}: instance_count"


dev = FakeDevice()
a = MeshArena(dev, vertex_floats=12, initial_slots=4)

# 1. Fill past initial capacity to force growth.
for i in range(10):
    a.add(("c", i), *make_chunk(i))
    a.flush()
check_invariants(a, "after 10 adds")
assert a.chunk_count == 10
print(f"grew to capacity={a.capacity}, generation={a.generation}")

# 2. Remove from the middle - exercises the swap-down path.
a.remove(("c", 3))
a.remove(("c", 0))
a.flush()
check_invariants(a, "after middle removals")
assert a.chunk_count == 8
assert ("c", 3) not in a.dense_of_key and ("c", 0) not in a.dense_of_key

# 3. Re-add: must reuse freed slots, not grow.
cap_before = a.capacity
a.add(("c", 100), *make_chunk(100))
a.add(("c", 101), *make_chunk(101))
a.flush()
check_invariants(a, "after re-add")
assert a.capacity == cap_before, "should have reused freed slots"

# 4. Verify vertex data actually landed in the right slot for every chunk.
for key, dense in a.dense_of_key.items():
    slot = a.slot_of_dense[dense]
    off = slot * a.slot_verts * a.vertex_itemsize
    got = np.frombuffer(bytes(a.vertex_buffer.data[off:off + 12 * 4]), dtype=np.float32)
    assert np.allclose(got, float(key[1])), (
        f"slot {slot} holds {got[0]} but {key} expected {key[1]}")
print("vertex payloads verified in-slot for all chunks")

# 5. Oversized chunk forces a per-slot re-layout; existing data must survive.
a.add(("big",), *make_chunk(999, nverts=40, nidx=60))
a.flush()
check_invariants(a, "after oversized add")
for key, dense in a.dense_of_key.items():
    slot = a.slot_of_dense[dense]
    off = slot * a.slot_verts * a.vertex_itemsize
    got = np.frombuffer(bytes(a.vertex_buffer.data[off:off + 12 * 4]), dtype=np.float32)
    expect = 999.0 if key == ("big",) else float(key[1])
    assert np.allclose(got, expect), f"{key} corrupted by re-layout: {got[0]} != {expect}"
print("data survived oversized re-layout (slot_verts=%d)" % a.slot_verts)

# 6. Full drain.
for key in list(a.dense_of_key):
    a.remove(key)
a.flush()
check_invariants(a, "after drain")
assert a.chunk_count == 0
assert len(set(a.free_slots)) == a.capacity, "leaked slots on drain"

# 7. Churn: simulates streaming, the case that caused the 21ms spikes.
rng = np.random.default_rng(0)
live = {}
for step in range(400):
    if live and rng.random() < 0.45:
        k = list(live)[rng.integers(len(live))]
        a.remove(k)
        del live[k]
    else:
        k = ("s", int(rng.integers(10_000)))
        if k not in live:
            a.add(k, *make_chunk(k[1]))
            live[k] = True
    a.flush()
check_invariants(a, "after churn")
assert a.chunk_count == len(live)
print(f"churn ok: {len(live)} live, capacity={a.capacity}, generation={a.generation}")
print("\nALL ARENA TESTS PASSED")
