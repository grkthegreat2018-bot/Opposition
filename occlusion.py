"""GPU-driven HZB occlusion culling helper for the terrain renderer."""
import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import multi_draw_indexed_indirect_count
from gpu_arena import MeshArena

COPY_SHADER = """
@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var dst_hiz: texture_storage_2d<r32float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(dst_hiz));
    let coord = vec2<i32>(id.xy);
    if (coord.x >= dims.x || coord.y >= dims.y) { return; }
    let src_dims = vec2<i32>(textureDimensions(src_depth));
    if (coord.x >= src_dims.x || coord.y >= src_dims.y) {
        textureStore(dst_hiz, coord, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }
    let d = textureLoad(src_depth, coord, 0);
    textureStore(dst_hiz, coord, vec4<f32>(d, 0.0, 0.0, 0.0));
}
"""

REDUCE_SHADER = """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_dims = vec2<i32>(textureDimensions(dst));
    let coord = vec2<i32>(id.xy);
    if (coord.x >= dst_dims.x || coord.y >= dst_dims.y) { return; }
    let src0 = vec2<i32>(coord * 2);
    let d0 = textureLoad(src, src0, 0).r;
    let d1 = textureLoad(src, src0 + vec2<i32>(1, 0), 0).r;
    let d2 = textureLoad(src, src0 + vec2<i32>(0, 1), 0).r;
    let d3 = textureLoad(src, src0 + vec2<i32>(1, 1), 0).r;
    let m = max(max(d0, d1), max(d2, d3));
    textureStore(dst, coord, vec4<f32>(m, 0.0, 0.0, 0.0));
}
"""

CULL_SHADER = """
struct CullUniforms { mvp: mat4x4<f32>, screen: vec2<f32>, params: vec4<u32> };
@group(0) @binding(0) var<uniform> u: CullUniforms;
@group(0) @binding(1) var<storage, read> aabb_buf: array<vec4<f32>>;
@group(0) @binding(2) var hiz: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> indirect_buf: array<u32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.params.y) { return; }
    let a = aabb_buf[idx * 2u];
    let b = aabb_buf[idx * 2u + 1u];
    let bmin = a.xyz;
    let bmax = b.xyz;
    var any_w_pos = false;
    var any_w_neg = false;
    var min_ndc = vec3<f32>(1.0e10);
    var max_ndc = vec3<f32>(-1.0e10);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let p = vec3<f32>(
            select(bmin.x, bmax.x, (i & 1u) != 0u),
            select(bmin.y, bmax.y, (i & 2u) != 0u),
            select(bmin.z, bmax.z, (i & 4u) != 0u),
        );
        let clip = u.mvp * vec4<f32>(p, 1.0);
        if (clip.w > 0.0) {
            any_w_pos = true;
            let ndc = clip.xyz / clip.w;
            min_ndc = min(min_ndc, ndc);
            max_ndc = max(max_ndc, ndc);
        } else {
            any_w_neg = true;
        }
    }
    var visible = false;
    if (!any_w_pos) {
        visible = false;
    } else if (any_w_neg) {
        visible = true;
    } else if (max_ndc.x < -1.0 || min_ndc.x > 1.0
            || max_ndc.y < -1.0 || min_ndc.y > 1.0
            || max_ndc.z < 0.0  || min_ndc.z > 1.0) {
        visible = false;
    } else {
        let min_depth = clamp(min_ndc.z, 0.0, 1.0);
        let min_uv = clamp(min_ndc.xy * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
        let max_uv = clamp(max_ndc.xy * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
        let min_px = min_uv * u.screen;
        let max_px = max_uv * u.screen;
        var tex_w = max_px.x - min_px.x;
        var tex_h = max_px.y - min_px.y;
        if (tex_w < 1.0) { tex_w = 1.0; }
        if (tex_h < 1.0) { tex_h = 1.0; }
        // Tiny distant edge chunks are prone to HZB sample noise; just draw them.
        if (tex_w <= 2.0 && tex_h <= 2.0) {
            visible = true;
        } else {
            let mip_f = floor(log2(max(tex_w, tex_h)));
            var mip = u32(mip_f);
            let max_mip_level = u.params.x - 1u;
            if (mip > max_mip_level) { mip = max_mip_level; }
            let mip_scale = f32(1u << mip);
            let mip_dims = vec2<f32>(textureDimensions(hiz, i32(mip)));
            let min_mip = vec2<i32>(min_px / mip_scale);
            let max_mip = vec2<i32>(max_px / mip_scale);
            let mn = clamp(min_mip, vec2<i32>(0), vec2<i32>(mip_dims) - vec2<i32>(1));
            let mx = clamp(max_mip, vec2<i32>(0), vec2<i32>(mip_dims) - vec2<i32>(1));
            let d0 = textureLoad(hiz, vec2<i32>(mn.x, mn.y), i32(mip)).r;
            let d1 = textureLoad(hiz, vec2<i32>(mn.x, mx.y), i32(mip)).r;
            let d2 = textureLoad(hiz, vec2<i32>(mx.x, mn.y), i32(mip)).r;
            let d3 = textureLoad(hiz, vec2<i32>(mx.x, mx.y), i32(mip)).r;
            let max_hiz = max(max(d0, d1), max(d2, d3));
            if (min_depth <= max_hiz + 0.005) { visible = true; }
        }
    }
    indirect_buf[idx * 5u + 1u] = select(0u, 1u, visible);
}
"""

class Occlusion:
    def __init__(self, renderer):
        self.renderer = renderer
        self.device = renderer.device
        self.arena = MeshArena(self.device, vertex_stride=32)
        self._bg_generation = -1
        self.current_index = 0
        self.hiz_width = self.hiz_height = self.hiz_mip_count = 1
        self._prev_size = (0, 0)
        self._create_resources()
        self.resize_depth_pyramid(*renderer.get_physical_size())
        self.sync()

    # Geometry storage lives in the arena; expose it under the names the
    # render passes already use.
    @property
    def chunk_count(self):
        return self.arena.chunk_count

    @property
    def vertex_buffer(self):
        return self.arena.vertex_buffer

    @property
    def index_buffer(self):
        return self.arena.index_buffer

    @property
    def aabb_buffer(self):
        return self.arena.aabb_buffer

    @property
    def indirect_buffers(self):
        return self.arena.indirect_buffers

    @property
    def prepass_indirect_buffer(self):
        return self.arena.prepass_indirect_buffer

    @property
    def count_buffer(self):
        return self.arena.count_buffer

    def _create_resources(self):
        self.cull_uniform_buffer = self.device.create_buffer(
            size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        # Reusable staging buffer for update() (avoids per-frame alloc).
        self._cull_uniform_staging = np.zeros(24, dtype=np.float32)
        self.copy_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": "depth", "view_dimension": "2d"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": "write-only", "format": "r32float", "view_dimension": "2d"}},
        ])
        self.reduce_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": "write-only", "format": "r32float", "view_dimension": "2d"}},
        ])
        self.cull_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
        ])
        copy_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.copy_bgl])
        reduce_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.reduce_bgl])
        cull_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.cull_bgl])
        self.copy_pipeline = self.device.create_compute_pipeline(
            layout=copy_layout,
            compute={"module": self.device.create_shader_module(code=COPY_SHADER), "entry_point": "main"},
        )
        self.reduce_pipeline = self.device.create_compute_pipeline(
            layout=reduce_layout,
            compute={"module": self.device.create_shader_module(code=REDUCE_SHADER), "entry_point": "main"},
        )
        self.cull_pipeline = self.device.create_compute_pipeline(
            layout=cull_layout,
            compute={"module": self.device.create_shader_module(code=CULL_SHADER), "entry_point": "main"},
        )

    def resize_depth_pyramid(self, width, height):
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == self._prev_size:
            return
        self._prev_size = (width, height)

        def next_power_of_2(v):
            v = max(1, int(v))
            v -= 1
            v |= v >> 1
            v |= v >> 2
            v |= v >> 4
            v |= v >> 8
            v |= v >> 16
            return v + 1

        self.hiz_width = next_power_of_2(width)
        self.hiz_height = next_power_of_2(height)
        max_dim = max(self.hiz_width, self.hiz_height)
        self.hiz_mip_count = max(1, int(np.floor(np.log2(max_dim))) + 1)
        self.hiz_texture = self.device.create_texture(
            size=(self.hiz_width, self.hiz_height, 1), dimension="2d",
            format="r32float", mip_level_count=self.hiz_mip_count,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.hiz_full_view = self.hiz_texture.create_view()
        self.hiz_mip_views = [
            self.hiz_texture.create_view(base_mip_level=i, mip_level_count=1)
            for i in range(self.hiz_mip_count)
        ]
        self.depth_compute_view = self.renderer.depth_texture.create_view(aspect="depth-only")
        self.copy_bg = self.device.create_bind_group(
            layout=self.copy_bgl,
            entries=[{"binding": 0, "resource": self.depth_compute_view},
                     {"binding": 1, "resource": self.hiz_mip_views[0]}],
        )
        self.reduce_bgs = []
        for i in range(1, self.hiz_mip_count):
            self.reduce_bgs.append(self.device.create_bind_group(
                layout=self.reduce_bgl,
                entries=[{"binding": 0, "resource": self.hiz_mip_views[i - 1]},
                         {"binding": 1, "resource": self.hiz_mip_views[i]}],
            ))
        if self.arena.vertex_buffer is not None:
            self._create_cull_bind_groups()

    def add_chunk(self, key, vertex_data, index_data, bbox):
        """Stream one chunk's mesh into the arena (no full rebuild)."""
        self.arena.add(key, vertex_data, index_data, bbox)

    def remove_chunk(self, key):
        self.arena.remove(key)

    def sync(self):
        """Flush pending arena writes and refresh bind groups if buffers moved.

        Cheap enough to call every frame: when nothing changed it does no GPU
        work at all.
        """
        self.arena.flush()
        if self.arena.generation != self._bg_generation:
            self._create_cull_bind_groups()
            self._bg_generation = self.arena.generation

    def _create_cull_bind_groups(self):
        self.cull_bgs = []
        for buf in self.indirect_buffers:
            self.cull_bgs.append(self.device.create_bind_group(
                layout=self.cull_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self.cull_uniform_buffer,
                                                "offset": 0, "size": self.cull_uniform_buffer.size}},
                    {"binding": 1, "resource": {"buffer": self.aabb_buffer,
                                                "offset": 0, "size": self.aabb_buffer.size}},
                    {"binding": 2, "resource": self.hiz_full_view},
                    {"binding": 3, "resource": {"buffer": buf, "offset": 0, "size": buf.size}},
                ],
            ))

    def update(self, mvp, width, height):
        data = self._cull_uniform_staging
        data.fill(0)
        data[:16] = mvp.T.flatten()
        data[16] = float(self.hiz_width)
        data[17] = float(self.hiz_height)
        data_u = data.view(np.uint32)
        data_u[20] = self.hiz_mip_count
        data_u[21] = self.chunk_count
        self.device.queue.write_buffer(self.cull_uniform_buffer, 0, data)

    def draw(self, encoder, color_view):
        max_count = max(1, self.chunk_count)
        curr_buf = self.indirect_buffers[self.current_index]
        # Seed the HZB with a complete depth prepass of all chunks so visibility
        # for the current frame is computed against the current view, not the
        # previous frame's cull result. This eliminates temporal feedback loops
        # that cause chunks at the render distance edge to flicker in and out.
        prepass = encoder.begin_render_pass(
            color_attachments=[{
                "view": color_view, "resolve_target": None,
                "clear_value": (self.renderer.sky_color[0], self.renderer.sky_color[1],
                                self.renderer.sky_color[2], self.renderer.sky_color[3]),
                "load_op": "clear", "store_op": "store",
            }],
            depth_stencil_attachment={
                "view": self.renderer.depth_view, "depth_clear_value": 1.0,
                "depth_load_op": "clear", "depth_store_op": "store",
            },
        )
        prepass.set_pipeline(self.renderer.prepass_pipeline)
        prepass.set_bind_group(0, self.renderer.bind_group)
        prepass.set_vertex_buffer(0, self.vertex_buffer, 0, self.vertex_buffer.size)
        prepass.set_index_buffer(self.index_buffer, "uint32", 0, self.index_buffer.size)
        multi_draw_indexed_indirect_count(prepass, self.prepass_indirect_buffer, count_buffer=self.count_buffer, max_count=max_count)  # type: ignore
        prepass.end()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.copy_pipeline)
        compute.set_bind_group(0, self.copy_bg)
        compute.dispatch_workgroups((self.hiz_width + 7) // 8, (self.hiz_height + 7) // 8)
        for i in range(1, self.hiz_mip_count):
            w = max(1, self.hiz_width >> i)
            h = max(1, self.hiz_height >> i)
            compute.set_pipeline(self.reduce_pipeline)
            compute.set_bind_group(0, self.reduce_bgs[i - 1])
            compute.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        if self.chunk_count > 0:
            compute.set_pipeline(self.cull_pipeline)
            compute.set_bind_group(0, self.cull_bgs[self.current_index])
            compute.dispatch_workgroups((self.chunk_count + 63) // 64)
        compute.end()
        # Clear depth (not load) for the main pass: the prepass depth was only
        # needed to build the HZB above. Loading it here would leave culled
        # chunks' geometry (including their skirts) in the depth buffer, causing
        # visible chunks to fail the less-equal depth test at shared borders and
        # producing holes/seams along chunk boundaries.
        color_pass = encoder.begin_render_pass(
            color_attachments=[{
                "view": color_view, "resolve_target": None,
                "load_op": "load", "store_op": "store",
            }],
            depth_stencil_attachment={
                "view": self.renderer.depth_view, "depth_clear_value": 1.0,
                "depth_load_op": "clear", "depth_store_op": "store",
            },
        )
        color_pass.set_pipeline(self.renderer.render_pipeline)
        color_pass.set_bind_group(0, self.renderer.bind_group)
        color_pass.set_vertex_buffer(0, self.vertex_buffer, 0, self.vertex_buffer.size)
        color_pass.set_index_buffer(self.index_buffer, "uint32", 0, self.index_buffer.size)
        multi_draw_indexed_indirect_count(color_pass, curr_buf, count_buffer=self.count_buffer, max_count=max_count)  # type: ignore
        color_pass.end()

    def swap(self):
        self.current_index = 1 - self.current_index
