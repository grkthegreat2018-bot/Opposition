"""Central tuning configuration.

Previously these values were literals inside `App.__init__`, duplicated again
as `ChunkManager` defaults and a third time in `_prewarm_numba`. The copies had
drifted apart (grid_res 48 vs 32, lacunarity 2.13 vs 2.0, warp_amp 0.4 vs 3.0),
so the defaults were dead weight that would silently produce different terrain
if anything ever constructed a ChunkManager without passing every argument.

Everything tunable now lives here as frozen dataclasses, and the call sites
derive their arguments from a single instance.
"""

from dataclasses import dataclass, field, asdict, replace
import numpy as np


@dataclass(frozen=True)
class NoiseConfig:
    """Base heightfield noise."""

    seed: int = 42
    scale: float = 25.0
    freq: float = 0.008
    octaves: int = 7
    persistence: float = 0.5
    # Non-integer lacunarity breaks the perfect 2x octave grid alignment that
    # produces visible repeating noise patterns at distance.
    lacunarity: float = 2.13
    warp_scale: float = 0.3
    warp_amp: float = 0.4
    ridge_weight: float = 0.35
    detail_weight: float = 0.12
    biome_freq: float = 0.0015


@dataclass(frozen=True)
class ContinentalConfig:
    """Large-scale landmasses: oceans, coastal mountains, interior plains.

    The continental mask is a very low-frequency noise field in [0, 1]:
      mask < 0.40  → ocean (flat seafloor below sea_level)
      0.40-0.55    → continental shelf / coast
      0.55-0.75    → coastal highlands (mountain chains)
      mask > 0.75  → continental interior (gentler terrain)

    All parameters are seam-safe (pure functions of world position).
    """

    # Frequency of the continental mask noise. ~40x lower than base terrain
    # → landmasses are 500-2000m across at the default settings.
    continental_freq: float = 0.0002
    # Height of the ocean surface. Terrain below this is underwater.
    sea_level: float = 0.0
    # How far below sea_level the deep ocean floor drops. Keep moderate so
    # the ocean-to-land transition doesn't create severe cliffs at chunk
    # boundaries (the skirt system can only cover limited height deltas).
    ocean_depth: float = 10.0
    # Base elevation boost for land above sea_level (continental interior).
    land_boost: float = 10.0
    # Coastal mountain chain parameters. Mountains peak at ``coastal_peak``
    # (mask value) with a Gaussian falloff of ``coastal_width``.
    coastal_peak: float = 0.65
    coastal_width: float = 0.18
    # Strength of the coastal mountain effect (0 = no coastal mountains,
    # 1 = full ridge weight at the coast). Scales the ridge multiplier.
    coastal_mountain_strength: float = 0.4
    # Width of the ocean-to-land amplitude transition (in mask units).
    # Within this band, terrain amplitude ramps from the ocean floor
    # amplitude to 1. Wider = smoother coastline but less defined shore.
    ocean_transition: float = 0.12
    # Minimum terrain amplitude in the ocean (0 = flat seafloor, 0.15 =
    # gentle underwater hills). Prevents the ocean from looking like a
    # flat plane abutting detailed land terrain.
    ocean_detail_floor: float = 0.15


@dataclass(frozen=True)
class ErosionConfig:
    """Post-process passes applied to the heightfield."""

    # Light thermal erosion rounds off sharp polygon edges and produces
    # natural-looking talus slopes without flattening the terrain.
    iters: int = 3
    talus: float = 1.2
    factor: float = 0.22
    # Hydraulic and wind erosion are accepted by Chunk but not currently
    # applied: both are seam-unsafe (random per-chunk droplets / a blur reach
    # exceeding practical padding). Their visual effect is approximated in the
    # fragment shader instead. See terrain/chunk.py for the full rationale.
    hydraulic_droplets: int = 2000
    wind_iters: int = 3
    # Gaussian smoothing pass to flatten tiny spikes/edges. 0=off, 0.3=gentle.
    smooth_strength: float = 0.35


@dataclass(frozen=True)
class FeatureConfig:
    """Large-scale landforms. All seam-safe; strength 0 disables each."""

    river_depth: float = 2.5
    plateau_strength: float = 0.6
    canyon_depth: float = 4.0
    crater_strength: float = 0.5
    # Glacial valley carving: U-shaped valleys in high-altitude terrain.
    glacial_strength: float = 0.8


@dataclass(frozen=True)
class StreamingConfig:
    """Chunk sizing and the streaming budget."""

    chunk_size: float = 32.0
    grid_res: int = 48
    radius: int = 8
    min_radius: int = 8
    max_radius: int = 10
    y_radius: int = 0
    max_builds_per_frame: int = 6
    target_compute_ms: float = 4.0
    # Quadtree LOD parameters. When use_lod is True, chunks are selected
    # via a distance-based quadtree instead of a uniform grid.
    use_lod: bool = True
    # LOD factor: controls how aggressively distant terrain is simplified.
    # A node of size S is subdivided only if its nearest point to the camera
    # is within S * lod_factor. Higher = more aggressive simplification.
    # Typical: 1.5-3.0. At 2.0, a 128m chunk is kept until the camera is
    # within 256m of it.
    lod_factor: float = 2.0
    # Maximum LOD level (coarsest chunk size = chunk_size * 2^max_level).
    # At level 4 with chunk_size=32, the largest chunk is 512m.
    lod_max_level: int = 4
    # Render distance in meters for LOD mode (replaces radius * chunk_size).
    # Set to cover roughly the same area as radius=8 (256m) but extend
    # further since LOD keeps vertex count bounded.
    lod_render_distance: float = 400.0


@dataclass(frozen=True)
class CameraConfig:
    start_pos: tuple = (2000.0, 80.0, 2000.0)
    start_yaw: float = -np.pi / 2
    start_pitch: float = 0.0
    fov: float = 75.0
    near: float = 0.1
    # Far plane is derived per-frame from render distance, but never below this
    # or distant terrain gets clipped when the camera is high up.
    min_far: float = 2500.0
    speed: float = 10.0
    shift_mult: float = 2.0
    sensitivity: float = 0.002


@dataclass(frozen=True)
class FogConfig:
    """Altitude-aware distance fog.

    fog_start is kept close to the loaded-terrain edge so chunks that ARE
    loaded stay visible; an earlier 0.6x start fogged out the outer ~25% of
    loaded chunks and made it look like chunks weren't loading at all.
    """

    # Altitude at which the widening starts, and the span over which it ramps.
    alt_ref: float = 20.0
    alt_span: float = 130.0
    # fog_start spans start_near..start_far x render_distance as altitude rises.
    start_near: float = 0.8
    start_far: float = 0.95
    end_near: float = 1.4
    end_far: float = 1.8

    def params(self, render_distance: float, altitude: float):
        """Return (fog_density, fog_start) for the current view."""
        alt_factor = min(max((altitude - self.alt_ref) / self.alt_span, 0.0), 1.0)
        start = render_distance * (self.start_near
                                   + (self.start_far - self.start_near) * alt_factor)
        end = render_distance * (self.end_near
                                 + (self.end_far - self.end_near) * alt_factor)
        # 4.605 = -ln(0.01): density placing ~99% extinction at fog_end.
        density = 4.605 / max(end - start, 1.0)
        return density, start


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 1280
    height: int = 720
    title: str = "Oposition - Vulkan Terrain"
    fullscreen: bool = True
    shadow_map_size: int = 2048


@dataclass(frozen=True)
class TimeConfig:
    """Day/night cycle. time_of_day is in hours [0, 24)."""

    start_hour: float = 9.0  # mid-morning light
    # Real seconds per full 24-hour cycle. 600s = 10 min/day, slow enough to
    # enjoy each phase without being static.
    day_length_sec: float = 600.0
    # Below this sun intensity the moon becomes the key light.
    night_threshold: float = 0.01


@dataclass(frozen=True)
class Config:
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    continental: ContinentalConfig = field(default_factory=ContinentalConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    fog: FogConfig = field(default_factory=FogConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    profiler_log_interval: float = 1.0

    def chunk_manager_kwargs(self) -> dict:
        """Flatten into the keyword arguments ChunkManager expects."""
        n, c, e, f, s = (self.noise, self.continental, self.erosion,
                         self.features, self.streaming)
        return {
            **asdict(s),
            **asdict(n),
            "continental_freq": c.continental_freq,
            "sea_level": c.sea_level,
            "ocean_depth": c.ocean_depth,
            "land_boost": c.land_boost,
            "coastal_peak": c.coastal_peak,
            "coastal_width": c.coastal_width,
            "coastal_mountain_strength": c.coastal_mountain_strength,
            "ocean_transition": c.ocean_transition,
            "ocean_detail_floor": c.ocean_detail_floor,
            "erosion_iters": e.iters,
            "erosion_talus": e.talus,
            "erosion_factor": e.factor,
            "hydraulic_droplets": e.hydraulic_droplets,
            "wind_erode_iters": e.wind_iters,
            "smooth_strength": e.smooth_strength,
            **asdict(f),
        }

    def prewarm_chunk_kwargs(self) -> dict:
        """Chunk arguments for the numba pre-warm build.

        Must exercise the same jitted code paths as a real chunk, so it shares
        every tuning value and only shrinks the grid.
        """
        kwargs = self.chunk_manager_kwargs()
        for key in ("radius", "min_radius", "max_radius", "y_radius",
                    "max_builds_per_frame", "target_compute_ms",
                    "use_lod", "lod_factor", "lod_max_level",
                    "lod_render_distance"):
            kwargs.pop(key, None)
        kwargs["size"] = kwargs.pop("chunk_size")
        kwargs["grid_res"] = 8
        kwargs.setdefault("level", 0)
        return {"cx": 0, "cy": 0, "cz": 0, **kwargs}


DEFAULT = Config()

__all__ = [
    "Config", "NoiseConfig", "ContinentalConfig", "ErosionConfig",
    "FeatureConfig", "StreamingConfig", "CameraConfig", "FogConfig",
    "DisplayConfig", "TimeConfig", "DEFAULT", "replace",
]
