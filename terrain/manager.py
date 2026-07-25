"""Chunk lifecycle management: async building, caching, streaming around camera."""
import atexit
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from terrain.chunk import Chunk, _build_chunk


class ChunkManager:
    """Generates and caches chunks around the camera."""

    def __init__(
        self,
        chunk_size: float = 32.0,
        grid_res: int = 32,
        cell_size: float | None = None,
        radius: int = 8,
        min_radius: int = 8,
        max_radius: int = 12,
        y_radius: int = 2,
        seed: int = 42,
        scale: float = 25.0,
        freq: float = 0.008,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        warp_scale: float = 0.3,
        warp_amp: float = 3.0,
        ridge_weight: float = 0.35,
        detail_weight: float = 0.12,
        biome_freq: float = 0.0015,
        continental_freq: float = 0.0002,
        sea_level: float = 0.0,
        ocean_depth: float = 18.0,
        land_boost: float = 12.0,
        coastal_peak: float = 0.65,
        coastal_width: float = 0.18,
        coastal_mountain_strength: float = 0.7,
        ocean_transition: float = 0.06,
        ocean_detail_floor: float = 0.15,
        erosion_iters: int = 0,
        erosion_talus: float = 1.5,
        erosion_factor: float = 0.25,
        hydraulic_droplets: int = 0,
        wind_erode_iters: int = 0,
        river_depth: float = 0.0,
        plateau_strength: float = 0.0,
        canyon_depth: float = 0.0,
        crater_strength: float = 0.0,
        smooth_strength: float = 0.0,
        glacial_strength: float = 0.0,
        max_builds_per_frame: int = 3,
        target_compute_ms: float = 4.0,
    ):
        self.chunk_size = chunk_size
        if cell_size is not None:
            grid_res = int(round(chunk_size / cell_size)) + 1
        self.grid_res = grid_res
        self.cell_size = chunk_size / (grid_res - 1)
        self.radius = radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.y_radius = y_radius
        self.seed = seed
        self.scale = scale
        self.freq = freq
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.warp_scale = warp_scale
        self.warp_amp = warp_amp
        self.ridge_weight = ridge_weight
        self.detail_weight = detail_weight
        self.biome_freq = biome_freq
        self.continental_freq = continental_freq
        self.sea_level = sea_level
        self.ocean_depth = ocean_depth
        self.land_boost = land_boost
        self.coastal_peak = coastal_peak
        self.coastal_width = coastal_width
        self.coastal_mountain_strength = coastal_mountain_strength
        self.ocean_transition = ocean_transition
        self.ocean_detail_floor = ocean_detail_floor
        self.erosion_iters = erosion_iters
        self.erosion_talus = erosion_talus
        self.erosion_factor = erosion_factor
        self.hydraulic_droplets = hydraulic_droplets
        self.wind_erode_iters = wind_erode_iters
        self.river_depth = river_depth
        self.plateau_strength = plateau_strength
        self.canyon_depth = canyon_depth
        self.crater_strength = crater_strength
        self.smooth_strength = smooth_strength
        self.glacial_strength = glacial_strength
        self.max_builds_per_frame = max_builds_per_frame
        self.target_compute_ms = target_compute_ms
        self.chunks: dict = {}
        self._build_queue = deque()
        self._build_queue_keys = set()
        self._pending_futures: dict = {}
        self._pending_keys: set = set()
        self._last_cx = None
        self._last_cy = None
        self._last_cz = None
        self._last_radius = radius
        self._needed: set = set()
        # Cache the y range of the terrain surface for each horizontal chunk column.
        self._surface_y_cache: dict = {}

        workers = min(max(self.max_builds_per_frame, 4), os.cpu_count() or 1)
        self._executor = ProcessPoolExecutor(max_workers=max(workers, 1))
        atexit.register(self.shutdown)

    def adjust_radius(self, compute_ms: float):
        """Grow or shrink render distance based on compute budget."""
        if compute_ms > self.target_compute_ms * 1.2 and self.radius > self.min_radius:
            self.radius -= 1
        elif compute_ms < self.target_compute_ms * 0.5 and self.radius < self.max_radius:
            self.radius += 1

    def chunk_coord(self, value: float) -> int:
        return int(np.floor(value / self.chunk_size))

    def _dist_to_cam(self, key, cx, cy, cz):
        # Squared 3-D distance for priority; height matters for ordering but does
        # not cull the chunk from being loaded.
        dx = key[0] - cx
        dy = key[1] - cy
        dz = key[2] - cz
        return dx * dx + dy * dy + dz * dz

    def _build_spec(self, key):
        return {
            "cx": key[0],
            "cy": key[1],
            "cz": key[2],
            "size": self.chunk_size,
            "grid_res": self.grid_res,
            "seed": self.seed,
            "scale": self.scale,
            "freq": self.freq,
            "octaves": self.octaves,
            "persistence": self.persistence,
            "lacunarity": self.lacunarity,
            "warp_scale": self.warp_scale,
            "warp_amp": self.warp_amp,
            "ridge_weight": self.ridge_weight,
            "detail_weight": self.detail_weight,
            "biome_freq": self.biome_freq,
            "continental_freq": self.continental_freq,
            "sea_level": self.sea_level,
            "ocean_depth": self.ocean_depth,
            "land_boost": self.land_boost,
            "coastal_peak": self.coastal_peak,
            "coastal_width": self.coastal_width,
            "coastal_mountain_strength": self.coastal_mountain_strength,
            "ocean_transition": self.ocean_transition,
            "ocean_detail_floor": self.ocean_detail_floor,
            "erosion_iters": self.erosion_iters,
            "erosion_talus": self.erosion_talus,
            "erosion_factor": self.erosion_factor,
            "hydraulic_droplets": self.hydraulic_droplets,
            "wind_erode_iters": self.wind_erode_iters,
            "river_depth": self.river_depth,
            "plateau_strength": self.plateau_strength,
            "canyon_depth": self.canyon_depth,
            "crater_strength": self.crater_strength,
            "smooth_strength": self.smooth_strength,
            "glacial_strength": self.glacial_strength,
        }

    def _queue_key(self, entry):
        return entry if isinstance(entry, tuple) else entry.key()

    def _surface_y_range(self, cx, cz):
        """Return the chunk y-range for column (cx, cz).

        With a single heightfield chunk per xz column, vertical chunking is no
        longer used; each column gets exactly one chunk at cy=0.
        """
        return (0, 0)

    def update(self, pos):
        """Update loaded chunks for the given camera position.

        Chunks are built asynchronously in a process pool. The main thread no
        longer blocks waiting for ``max_builds_per_frame`` builds each frame;
        completed builds are collected and new tasks are submitted up to the
        in-flight budget. Returns (changed, new_chunks, removed_keys).
        """
        cx = self.chunk_coord(pos[0])
        cy = self.chunk_coord(pos[1])
        cz = self.chunk_coord(pos[2])
        removed = []

        if (cx, cy, cz) != (self._last_cx, self._last_cy, self._last_cz) or self.radius != self._last_radius:
            self._last_cx, self._last_cy, self._last_cz = cx, cy, cz
            self._last_radius = self.radius

            needed = set()
            current_columns = set()
            for dx in range(-self.radius, self.radius + 1):
                for dz in range(-self.radius, self.radius + 1):
                    if dx * dx + dz * dz > self.radius * self.radius:
                        continue
                    col_cx = cx + dx
                    col_cz = cz + dz
                    current_columns.add((col_cx, col_cz))
                    # One heightfield chunk per xz column.
                    needed.add((col_cx, 0, col_cz))

            # Keep the surface y cache bounded to columns currently in view.
            self._surface_y_cache = {
                k: v for k, v in self._surface_y_cache.items() if k in current_columns
            }

            # Remove chunks that are no longer needed
            removed = [key for key in self.chunks if key not in needed]
            for key in removed:
                del self.chunks[key]

            # Cancel pending builds that are no longer needed
            for f in list(self._pending_futures):
                key = self._pending_futures[f]
                if key not in needed:
                    f.cancel()
                    del self._pending_futures[f]
                    self._pending_keys.discard(key)

            # Build a fresh queue of chunks that still need to be loaded
            self._build_queue = deque()
            self._build_queue_keys = set()
            for key in needed:
                if key in self.chunks or key in self._pending_keys:
                    continue
                self._build_queue.append(key)
                self._build_queue_keys.add(key)

            # Sort the whole queue by camera distance
            work = list(self._build_queue)
            work.sort(key=lambda e: self._dist_to_cam(self._queue_key(e), cx, cy, cz))
            self._build_queue = deque(work)

            self._needed = needed

        # Collect any chunks that finished building since the last frame
        new_chunks = []
        completed = [f for f in list(self._pending_futures) if f.done()]
        for f in completed:
            key = self._pending_futures.pop(f)
            self._pending_keys.discard(key)
            if key not in self._needed:
                continue
            try:
                _, mesh, build_time = f.result()
            except Exception:
                continue
            c = Chunk(**self._build_spec(key))
            c.mesh = mesh
            c.build_time = build_time
            if mesh is not None:
                c.tri_count = mesh["indices"].shape[0]
                c.vert_count = mesh["vertices"].shape[0]
            self.chunks[key] = c
            new_chunks.append(c)

        # Keep the process pool saturated up to the in-flight budget
        executor = self._executor
        while (
            executor is not None
            and len(self._pending_futures) < self.max_builds_per_frame
            and self._build_queue
        ):
            key = self._build_queue.popleft()
            self._build_queue_keys.discard(key)
            spec = self._build_spec(key)
            f = executor.submit(_build_chunk, spec)
            self._pending_futures[f] = key
            self._pending_keys.add(key)

        changed = len(new_chunks) > 0 or len(removed) > 0
        return changed, new_chunks, removed

    def shutdown(self):
        """Clean up the process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def get_meshes(self):
        """Return a list of meshes for all loaded chunks."""
        return [c.mesh for c in self.chunks.values() if c.mesh is not None]

    def stats(self):
        loaded = len(self.chunks)
        meshed = sum(1 for c in self.chunks.values() if getattr(c, "tri_count", 0) > 0)
        triangles = sum(getattr(c, "tri_count", 0) for c in self.chunks.values())
        return {"loaded": loaded, "meshed": meshed, "triangles": triangles}
