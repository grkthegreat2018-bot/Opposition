"""Rendering subsystem: pipelines, GPU buffers, sky/water/clouds, debug HUD."""
from .renderer import TerrainRenderer
from .shaders import SHADER, BBOX_SHADER
from .occlusion import Occlusion
from .sky import SkyRenderer, compute_sky_params
from .clouds import CloudRenderer
from .water import WaterRenderer
from .debug_hud import DebugHUD
from .chunk_data import _ChunkData, ChunkRecord, VERTEX_STRIDE
from .gpu_arena import MeshArena

__all__ = [
    "TerrainRenderer", "SHADER", "BBOX_SHADER", "Occlusion",
    "SkyRenderer", "compute_sky_params", "CloudRenderer", "WaterRenderer",
    "DebugHUD", "_ChunkData", "ChunkRecord", "VERTEX_STRIDE", "MeshArena",
]
