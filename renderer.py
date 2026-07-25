"""wgpu-based Vulkan renderer for chunked terrain."""

import time

import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import multi_draw_indexed_indirect_count

from camera import look_at, orthographic
from chunk_data import _ChunkData, ChunkRecord
from occlusion import Occlusion
from shaders import SHADER, BBOX_SHADER
from water import WaterRenderer
from sky import SkyRenderer
from clouds import CloudRenderer


class TerrainRenderer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync(
            label="Terrain renderer device",
            required_features=["multi-draw-indirect-count"],
        )
        self.context = canvas.get_wgpu_context()
        self.format = self.context.get_preferred_format(self.adapter)
        self.context.configure(device=self.device, format=self.format)

        self._create_pipeline()
        self._create_depth_texture()
        self._create_shadow_texture()
        self._create_occlusion_pipeline()
        self._create_prepass_pipeline()
        self._create_shadow_pipeline()

        self.sky_color = np.array([0.53, 0.81, 0.98, 1.0], dtype=np.float32)
        # mvp + light_mvp + light/camera/fog/sun_color = 52 floats (208 bytes)
        self.uniform_buffer = self.device.create_buffer(
            size=52 * 4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="uniform buffer",
        )
        # Reusable staging buffer for update_uniforms (avoids per-frame alloc).
        self._uniform_staging = np.zeros(52, dtype=np.float32)
        # Reusable scratch arrays for _compute_shadow_mvp (avoids per-frame alloc).
        self._shadow_world_min = np.zeros(3, dtype=np.float32)
        self._shadow_world_max = np.zeros(3, dtype=np.float32)
        self._shadow_target = np.zeros(3, dtype=np.float32)
        self._shadow_eye = np.zeros(3, dtype=np.float32)
        self._shadow_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._shadow_up_alt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._shadow_corners = np.zeros((8, 4), dtype=np.float32)
        self._shadow_view_mat = np.zeros((4, 8), dtype=np.float32)
        # Cached bboxes array + world bounds, rebuilt only when chunks change.
        self._bboxes_cache = np.zeros((0, 6), dtype=np.float32)
        self._shadow_bounds_dirty = True
        self._shadow_world_min_cached = np.zeros(3, dtype=np.float32)
        self._shadow_world_max_cached = np.zeros(3, dtype=np.float32)
        self._shadow_half_size_cached = 256.0

        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.uniform_buffer, "offset": 0, "size": self.uniform_buffer.size}},
                {"binding": 1, "resource": self.shadow_view},
                {"binding": 2, "resource": self.shadow_sampler},
            ],
            label="terrain bind group",
        )

        self.shadow_bind_group = self.device.create_bind_group(
            layout=self.shadow_bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.uniform_buffer, "offset": 0, "size": self.uniform_buffer.size}}
            ],
            label="shadow bind group",
        )

        self.water = WaterRenderer(self)
        self.sky = SkyRenderer(self)
        self.clouds = CloudRenderer(self)

        self.chunk_meshes = {}
        self.occlusion = Occlusion(self)

    def _create_pipeline(self):
        shader = self.device.create_shader_module(code=SHADER, label="terrain shader")

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "view_dimension": "2d"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "comparison"},
                },
            ],
            label="terrain bind group layout",
        )

        self.pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout],
            label="terrain pipeline layout",
        )

        self.render_pipeline = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        # pos (3) + normal (3) + biome (4) + sc (2) = 12 floats/vertex
                        "array_stride": 12 * 4,
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 3 * 4, "shader_location": 1},
                            {"format": "float32x4", "offset": 6 * 4, "shader_location": 2},
                            {"format": "float32x2", "offset": 10 * 4, "shader_location": 3},
                        ],
                    }
                ],
            },
            primitive={
                "topology": "triangle-list",
                "front_face": "ccw",
                "cull_mode": "back",
            },
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": True,
                "depth_compare": "less-equal",
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.format}],
            },
            label="terrain render pipeline",
        )

    def _create_occlusion_pipeline(self):
        """Pipeline used to rasterize chunk bounding boxes for occlusion queries.

        Color writes are disabled; only the depth test and query counter matter.
        """
        shader = self.device.create_shader_module(code=BBOX_SHADER, label="bbox shader")

        self.occlusion_pipeline = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 3 * 4,
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                        ],
                    }
                ],
            },
            primitive={
                "topology": "triangle-list",
                "front_face": "ccw",
                "cull_mode": "none",
            },
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": False,
                "depth_compare": "less-equal",
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.format, "write_mask": 0}],
            },
            label="occlusion bbox pipeline",
        )

    def _create_prepass_pipeline(self):
        """Depth-only pipeline used to seed the occlusion query depth buffer."""
        # Use the cheap position-only shader; color writes are masked off.
        shader = self.device.create_shader_module(code=BBOX_SHADER, label="prepass shader")

        self.prepass_pipeline = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        # Reads only pos from the interleaved pos+normal+biome+sc buffer.
                        "array_stride": 12 * 4,
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                        ],
                    }
                ],
            },
            primitive={
                "topology": "triangle-list",
                "front_face": "ccw",
                "cull_mode": "back",
            },
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": True,
                "depth_compare": "less",
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.format, "write_mask": 0}],
            },
            label="terrain depth prepass pipeline",
        )

    def _create_shadow_pipeline(self):
        """Depth-only pipeline for rendering the shadow map."""
        shader = self.device.create_shader_module(code=BBOX_SHADER, label="shadow shader")

        self.shadow_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": "uniform"},
                }
            ],
            label="shadow bind group layout",
        )
        self.shadow_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.shadow_bind_group_layout],
            label="shadow pipeline layout",
        )

        self.shadow_pipeline = self.device.create_render_pipeline(
            layout=self.shadow_pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        # Reads only pos from the interleaved pos+normal+biome+sc buffer.
                        "array_stride": 12 * 4,
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                        ],
                    }
                ],
            },
            primitive={
                "topology": "triangle-list",
                "front_face": "ccw",
                "cull_mode": "back",
            },
            depth_stencil={
                "format": "depth32float",
                "depth_write_enabled": True,
                "depth_compare": "less",
            },
            label="terrain shadow pipeline",
        )

    def _create_depth_texture(self):
        self.depth_texture = None
        self.depth_view = None
        self._resize_depth()

    def _resize_depth(self):
        width, height = self.get_physical_size()
        width, height = max(1, width), max(1, height)
        if (
            self.depth_texture is not None
            and self.depth_texture.size[0] == width
            and self.depth_texture.size[1] == height
        ):
            return
        self.depth_texture = self.device.create_texture(
            size=(width, height, 1),
            dimension="2d",
            format="depth24plus",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            label="depth texture",
        )
        self.depth_view = self.depth_texture.create_view(label="depth texture view")

    def _create_shadow_texture(self, size: int = 2048):
        self.shadow_texture = self.device.create_texture(
            size=(size, size, 1),
            dimension="2d",
            format="depth32float",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            label="shadow map",
        )
        self.shadow_view = self.shadow_texture.create_view(label="shadow map view")
        self.shadow_sampler = self.device.create_sampler(
            compare="less",
            mag_filter="linear",
            min_filter="linear",
            label="shadow sampler",
        )

    def get_physical_size(self):
        w, h = self.canvas.get_logical_size()
        return int(w * self.canvas.get_pixel_ratio()), int(h * self.canvas.get_pixel_ratio())

    def update_uniforms(self, mvp, light_mvp, light_dir, camera_pos, fog_density, fog_start, fog_color, time=0.0, sun_color=None):
        data = self._uniform_staging
        data.fill(0)
        data[:16] = mvp.T.flatten()
        data[16:32] = light_mvp.T.flatten()
        data[32:35] = light_dir
        data[36:39] = camera_pos
        data[40:43] = fog_color
        data[44] = fog_density
        data[45] = fog_start
        data[46] = time
        if sun_color is not None:
            data[48:51] = sun_color
        else:
            data[48:51] = [1.0, 0.96, 0.88]
        self.device.queue.write_buffer(self.uniform_buffer, 0, data)

    def _rebuild_shadow_bounds(self):
        """Recompute cached world-space AABB of all loaded chunks.

        Called only when chunks are added or removed, not every frame.
        """
        if not self.chunk_meshes:
            self._bboxes_cache = np.zeros((0, 6), dtype=np.float32)
            return
        self._bboxes_cache = np.array(
            [c.bbox for c in self.chunk_meshes.values()], dtype=np.float32
        )
        b = self._bboxes_cache
        wmin = self._shadow_world_min_cached
        wmax = self._shadow_world_max_cached
        wmin[0] = b[:, 0].min()
        wmin[1] = b[:, 1].min()
        wmin[2] = b[:, 2].min()
        wmax[0] = b[:, 3].max()
        wmax[1] = b[:, 4].max()
        wmax[2] = b[:, 5].max()
        half = max(wmax[0] - wmin[0], wmax[2] - wmin[2]) * 0.5 + 20.0
        self._shadow_half_size_cached = max(half, 256.0)

    def _compute_shadow_mvp(self, camera_pos, light_dir):
        """Build an orthographic light view-projection covering visible chunks."""
        if self._shadow_bounds_dirty:
            self._rebuild_shadow_bounds()
            self._shadow_bounds_dirty = False

        if self.chunk_meshes:
            world_min = self._shadow_world_min_cached
            world_max = self._shadow_world_max_cached
            target = self._shadow_target
            target[:] = (world_min + world_max) * 0.5
            half_size = self._shadow_half_size_cached

            corners = self._shadow_corners
            corners[:, 3] = 1.0
            corners[0, :3] = (world_min[0], world_min[1], world_min[2])
            corners[1, :3] = (world_min[0], world_min[1], world_max[2])
            corners[2, :3] = (world_min[0], world_max[1], world_min[2])
            corners[3, :3] = (world_min[0], world_max[1], world_max[2])
            corners[4, :3] = (world_max[0], world_min[1], world_min[2])
            corners[5, :3] = (world_max[0], world_min[1], world_max[2])
            corners[6, :3] = (world_max[0], world_max[1], world_min[2])
            corners[7, :3] = (world_max[0], world_max[1], world_max[2])
        else:
            target = self._shadow_target
            target[:] = camera_pos.astype(np.float32)
            half_size = 256.0
            corners = self._shadow_corners
            corners[:, 3] = 1.0
            corners[0, :3] = (-half_size, -100.0, -half_size)
            corners[1, :3] = (-half_size, -100.0, half_size)
            corners[2, :3] = (-half_size, 100.0, -half_size)
            corners[3, :3] = (-half_size, 100.0, half_size)
            corners[4, :3] = (half_size, -100.0, -half_size)
            corners[5, :3] = (half_size, -100.0, half_size)
            corners[6, :3] = (half_size, 100.0, -half_size)
            corners[7, :3] = (half_size, 100.0, half_size)

        eye = self._shadow_eye
        eye[:] = target - light_dir * (half_size + 50.0)
        up = self._shadow_up
        if abs(np.dot(light_dir, up)) > 0.99:
            up = self._shadow_up_alt
        view = look_at(eye, target, up)

        # Tighten near/far planes using chunk AABB corners in light view space.
        view_mat = self._shadow_view_mat
        np.dot(view, corners.T, out=view_mat)
        view_z = view_mat[2, :]
        near = -float(view_z.max())
        far = -float(view_z.min())

        near = max(near, 1.0)
        far = max(far, near + 1.0)
        proj = orthographic(
            -half_size, half_size, -half_size, half_size, near, far
        )
        return proj @ view

    def add_chunks(self, chunks):
        """Upload a list of chunk meshes to GPU."""
        for c in chunks:
            mesh = c.mesh
            if mesh is not None and mesh["indices"].size > 0:
                key = c.key()
                cd = _ChunkData(mesh, getattr(c, "bbox", None))
                # Hand the interleaved data straight to the arena, which copies
                # it into its own slot; nothing is retained on the CPU here.
                self.occlusion.add_chunk(key, cd.vertex_data, cd.index_data, cd.bbox)
                self.chunk_meshes[key] = ChunkRecord(cd.bbox)
            c.mesh = None
        self._shadow_bounds_dirty = True

    def remove_chunks(self, keys):
        """Remove GPU buffers for the given chunk keys."""
        for key in keys:
            if self.chunk_meshes.pop(key, None) is not None:
                self.occlusion.remove_chunk(key)
        self._shadow_bounds_dirty = True

    def draw(self, camera_view, camera_proj, light_dir=(0.0, 1.0, 0.0), camera_pos=None,
             fog_density=0.0, fog_start=0.0, fog_color=None, sun_color=None, sky_params=None,
             debug_hud=None, inv_view_proj=None):
        self._resize_depth()
        # Flushes any pending arena writes; a no-op when no chunks changed.
        self.occlusion.sync()

        width, height = self.get_physical_size()
        self.occlusion.resize_depth_pyramid(width, height)

        mvp = camera_proj @ camera_view
        if fog_color is None:
            fog_color = self.sky_color[:3]
        # light_dir may arrive as a tuple or list; convert once if needed.
        if not isinstance(light_dir, np.ndarray):
            light_dir = np.asarray(light_dir, dtype=np.float32)
        elif light_dir.dtype != np.float32:
            light_dir = light_dir.astype(np.float32)
        norm = np.linalg.norm(light_dir)
        if norm > 0.0:
            light_dir = light_dir / norm
        # camera_pos is already np.float32 from Camera.pos; avoid redundant copy.
        if camera_pos is not None and not isinstance(camera_pos, np.ndarray):
            camera_pos = np.asarray(camera_pos, dtype=np.float32)
        light_mvp = self._compute_shadow_mvp(camera_pos, light_dir)

        current = self.context.get_current_texture()
        view = current.create_view(label="main color view")

        encoder = self.device.create_command_encoder(label="shadow and hzb encoder")

        # Render the shadow map from the light's point of view.
        if self.occlusion.chunk_count > 0:
            self.update_uniforms(light_mvp, light_mvp, light_dir, camera_pos, fog_density, fog_start, fog_color, sun_color=sun_color)
            shadow_pass = encoder.begin_render_pass(
                color_attachments=[],
                depth_stencil_attachment={
                    "view": self.shadow_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": "clear",
                    "depth_store_op": "store",
                },
                label="shadow pass",
            )
            shadow_pass.set_pipeline(self.shadow_pipeline)
            shadow_pass.set_bind_group(0, self.shadow_bind_group)
            shadow_pass.set_vertex_buffer(0, self.occlusion.vertex_buffer, 0, self.occlusion.vertex_buffer.size)
            shadow_pass.set_index_buffer(self.occlusion.index_buffer, "uint32", 0, self.occlusion.index_buffer.size)
            multi_draw_indexed_indirect_count(  # type: ignore
                shadow_pass,  # type: ignore
                self.occlusion.prepass_indirect_buffer,
                count_buffer=self.occlusion.count_buffer,
                max_count=max(1, self.occlusion.chunk_count),
            )
            shadow_pass.end()

        # Main pass with camera MVP and light MVP for shadow sampling.
        current_time = time.time()
        self.update_uniforms(mvp, light_mvp, light_dir, camera_pos, fog_density, fog_start, fog_color, time=current_time, sun_color=sun_color)
        self.occlusion.update(mvp, width, height)
        self.occlusion.draw(encoder, view)

        # Sky pass: draws sky gradient + sun + moon + stars where no terrain.
        # Cloud pass uses the same inv_view_proj, so compute it once here.
        if (self.sky is not None or self.clouds is not None) and sky_params is not None:
            if inv_view_proj is None:
                inv_view_proj = np.linalg.inv(mvp)
            if self.sky is not None:
                self.sky.update_uniforms(inv_view_proj, camera_pos, sky_params, current_time)
                self.sky.draw(encoder, view, self.depth_view)

            # Cloud pass: alpha-blended clouds, depth-tested against terrain.
            if self.clouds is not None:
                self.clouds.update_uniforms(inv_view_proj, camera_pos, sky_params, current_time)
                self.clouds.draw(encoder, view, self.depth_view)

        # Transparent water pass.
        if self.water is not None:
            self.water.draw(encoder, view, self.depth_view)

        # Debug HUD overlay (text + chunk highlights).
        if debug_hud is not None:
            debug_hud.draw(encoder, view, self.depth_view, camera_view, camera_proj, width, height)

        self.device.queue.submit([encoder.finish()])

        self.occlusion.swap()
