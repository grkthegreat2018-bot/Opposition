"""Test ChunkManager in LOD mode (no rendering)."""
import sys
sys.path.insert(0, '.')
from terrain import ChunkManager
from core.config import Config

# Build a config with LOD enabled.
cfg = Config()
print(f"Default use_lod: {cfg.streaming.use_lod}")
print(f"LOD factor: {cfg.streaming.lod_factor}, max_level: {cfg.streaming.lod_max_level}, render_dist: {cfg.streaming.lod_render_distance}")

# Enable LOD.
import dataclasses
streaming = dataclasses.replace(cfg.streaming, use_lod=True, lod_render_distance=200.0, lod_factor=2.0, lod_max_level=3)
cfg = dataclasses.replace(cfg, streaming=streaming)

# Override executor to None (no process pool) so we don't actually build.
kwargs = cfg.chunk_manager_kwargs()
print(f"kwargs has use_lod: {kwargs.get('use_lod')}")
print(f"kwargs has lod_factor: {kwargs.get('lod_factor')}")

# Create manager without executor (we just want to test selection).
mgr = ChunkManager(**kwargs)
mgr._executor = None  # disable async building
print(f"Manager use_lod={mgr.use_lod}, factor={mgr.lod_factor}, max_level={mgr.lod_max_level}, dist={mgr.lod_render_distance}")

# Run update at origin.
changed, new_chunks, removed = mgr.update((0.0, 50.0, 0.0))
print(f"Update 1: needed={len(mgr._needed)} chunks, queue={len(mgr._build_queue)}")
by_level = {}
for key in mgr._needed:
    level = key[2]
    by_level[level] = by_level.get(level, 0) + 1
for lv in sorted(by_level):
    print(f"  level {lv}: {by_level[lv]} chunks")

# Move camera.
changed, new_chunks, removed = mgr.update((50.0, 50.0, 50.0))
print(f"Update 2 (moved): needed={len(mgr._needed)} chunks, queue={len(mgr._build_queue)}")

# Move back.
changed, new_chunks, removed = mgr.update((0.0, 50.0, 0.0))
print(f"Update 3 (back): needed={len(mgr._needed)} chunks, queue={len(mgr._build_queue)}")

print("\nLOD MANAGER TEST PASSED")
