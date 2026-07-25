"""Self-contained debug HUD: bitmap text overlay + chunk highlight boxes for wgpu Vulkan renderer."""
import numpy as np
import wgpu

_FONT = {
    ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    '!': [0x04,0x04,0x04,0x04,0x04,0x00,0x04],
    '"': [0x0A,0x0A,0x0A,0x00,0x00,0x00,0x00],
    '#': [0x0A,0x1F,0x0A,0x1F,0x0A,0x00,0x00],
    '&': [0x0E,0x11,0x0E,0x11,0x15,0x12,0x0D],
    '%': [0x18,0x18,0x04,0x08,0x10,0x0C,0x0C],
    "'": [0x04,0x04,0x04,0x00,0x00,0x00,0x00],
    '(': [0x08,0x04,0x02,0x02,0x02,0x04,0x08],
    ')': [0x02,0x04,0x08,0x08,0x08,0x04,0x02],
    '*': [0x00,0x11,0x0A,0x1F,0x0A,0x11,0x00],
    '+': [0x00,0x04,0x04,0x1F,0x04,0x04,0x00],
    ',': [0x00,0x00,0x00,0x00,0x0C,0x04,0x08],
    '-': [0x00,0x00,0x00,0x1F,0x00,0x00,0x00],
    '.': [0x00,0x00,0x00,0x00,0x00,0x0C,0x0C],
    '/': [0x00,0x10,0x08,0x04,0x02,0x01,0x00],
    '0': [0x0E,0x11,0x19,0x15,0x13,0x11,0x0E],
    '1': [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E],
    '2': [0x0E,0x11,0x10,0x08,0x04,0x02,0x1F],
    '3': [0x1F,0x08,0x04,0x08,0x10,0x11,0x0E],
    '4': [0x08,0x0C,0x0A,0x09,0x1F,0x08,0x08],
    '5': [0x1F,0x01,0x0F,0x10,0x10,0x11,0x0E],
    '6': [0x0C,0x02,0x01,0x0F,0x11,0x11,0x0E],
    '7': [0x1F,0x10,0x08,0x04,0x02,0x02,0x02],
    '8': [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E],
    '9': [0x0E,0x11,0x11,0x1E,0x10,0x08,0x0C],
    ':': [0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00],
    ';': [0x00,0x0C,0x0C,0x00,0x0C,0x04,0x08],
    '<': [0x08,0x04,0x02,0x01,0x02,0x04,0x08],
    '=': [0x00,0x00,0x1F,0x00,0x1F,0x00,0x00],
    '>': [0x02,0x04,0x08,0x10,0x08,0x04,0x02],
    '?': [0x0E,0x11,0x10,0x08,0x04,0x00,0x04],
    'A': [0x0E,0x11,0x11,0x1F,0x11,0x11,0x11],
    'B': [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E],
    'C': [0x0E,0x11,0x01,0x01,0x01,0x11,0x0E],
    'D': [0x1C,0x12,0x11,0x11,0x11,0x12,0x1C],
    'E': [0x1F,0x01,0x01,0x0E,0x01,0x01,0x1F],
    'F': [0x1F,0x01,0x01,0x0E,0x01,0x01,0x01],
    'G': [0x0E,0x11,0x01,0x0D,0x11,0x11,0x0E],
    'H': [0x11,0x11,0x11,0x1F,0x11,0x11,0x11],
    'I': [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E],
    'J': [0x10,0x10,0x10,0x10,0x10,0x11,0x0E],
    'K': [0x11,0x09,0x05,0x03,0x05,0x09,0x11],
    'L': [0x01,0x01,0x01,0x01,0x01,0x01,0x1F],
    'M': [0x11,0x1B,0x15,0x15,0x11,0x11,0x11],
    'N': [0x11,0x11,0x19,0x15,0x13,0x11,0x11],
    'O': [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E],
    'P': [0x1E,0x11,0x11,0x1E,0x01,0x01,0x01],
    'Q': [0x0E,0x11,0x11,0x11,0x15,0x09,0x16],
    'R': [0x1E,0x11,0x11,0x1E,0x05,0x09,0x11],
    'S': [0x0E,0x11,0x01,0x0E,0x10,0x11,0x0E],
    'T': [0x1F,0x04,0x04,0x04,0x04,0x04,0x04],
    'U': [0x11,0x11,0x11,0x11,0x11,0x11,0x0E],
    'V': [0x11,0x11,0x11,0x11,0x11,0x0A,0x04],
    'W': [0x11,0x11,0x11,0x15,0x15,0x15,0x0A],
    'X': [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11],
    'Y': [0x11,0x11,0x11,0x0A,0x04,0x04,0x04],
    'Z': [0x1F,0x10,0x08,0x04,0x02,0x01,0x1F],
    '[': [0x0E,0x02,0x02,0x02,0x02,0x02,0x0E],
    '\\':[0x00,0x01,0x02,0x04,0x08,0x10,0x00],
    ']': [0x0E,0x08,0x08,0x08,0x08,0x08,0x0E],
    '_': [0x00,0x00,0x00,0x00,0x00,0x00,0x1F],
}

_TEXT_WGSL = r"""
struct TextUniform { screen: vec2f, _pad: vec2f }
@group(0) @binding(0) var<uniform> u: TextUniform;
@group(0) @binding(1) var<storage, read> instances: array<u32>;
@group(0) @binding(2) var s_font: sampler;
@group(0) @binding(3) var t_font: texture_2d<f32>;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
    let corners = array<vec2f,6>(
        vec2f(0.0,0.0), vec2f(1.0,0.0), vec2f(0.0,1.0),
        vec2f(0.0,1.0), vec2f(1.0,0.0), vec2f(1.0,1.0));
    let cp = corners[vi];
    let col = f32(instances[ii * 3u]);
    let row = f32(instances[ii * 3u + 1u]);
    let char_code = instances[ii * 3u + 2u];
    let char_idx = f32(char_code - 32u);
    let cell = vec2f(6.0, 8.0);
    let line_h = 10.0;
    let screen_pos = vec2f(col * cell.x + 2.0, row * line_h + 2.0) + cp * cell;
    let ndc = vec2f(screen_pos.x / u.screen.x, screen_pos.y / u.screen.y) * 2.0 - vec2f(1.0);
    var out: VSOut;
    out.pos = vec4f(ndc.x, -ndc.y, 0.0, 1.0);
    // Atlas is 320x7 pixels. Normalize UVs to [0,1].
    let atlas_w = 320.0;
    let atlas_h = 7.0;
    out.uv = vec2f((char_idx * 5.0 + cp.x * 5.0) / atlas_w, cp.y * 7.0 / atlas_h);
    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let sample = textureSample(t_font, s_font, uv);
    return vec4f(0.1, 0.9, 0.2, sample.r * 0.85);
}
"""

_BOX_WGSL = r"""
struct BoxUniform { mvp: mat4x4f, color: vec4f }
@group(0) @binding(0) var<uniform> u: BoxUniform;

@vertex
fn vs_main(@location(0) pos: vec3f) -> @builtin(position) vec4f {
    return u.mvp * vec4f(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return u.color;
}
"""

_MAX_LINES = 20
_MAX_CHARS = 80
_MAX_INSTANCES = _MAX_LINES * _MAX_CHARS


def _build_font_atlas():
    """Build uint8 texture atlas (width=5*64, height=7) from _FONT dict."""
    atlas = np.zeros((7, 5 * 64), dtype=np.uint8)
    for ch, rows in _FONT.items():
        idx = ord(ch) - 32
        if idx < 0 or idx >= 64:
            continue
        for r in range(7):
            bits = rows[r]
            for c in range(5):
                if bits & (1 << (4 - c)):
                    atlas[r, idx * 5 + c] = 255
    return atlas


def _box_solid_verts(bbox):
    """36 vertices (12 tris) for a solid box."""
    x0, y0, z0, x1, y1, z1 = bbox
    v = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
         (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    faces = [[0,1,2, 0,2,3], [4,6,5, 4,7,6],
             [0,4,5, 0,5,1], [1,5,6, 1,6,2],
             [2,6,7, 2,7,3], [0,3,7, 0,7,4]]
    verts = []
    for f in faces:
        for i in f:
            verts.append(v[i])
    return np.array(verts, dtype=np.float32)


def _box_line_verts(bbox):
    """24 vertices (12 line segments) for wireframe box."""
    x0, y0, z0, x1, y1, z1 = bbox
    c = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
         (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    verts = []
    for a, b in edges:
        verts.append(c[a]); verts.append(c[b])
    return np.array(verts, dtype=np.float32)


class DebugHUD:
    """Debug overlay: text + chunk highlight boxes."""

    def __init__(self, device, format, sample_count=1):
        self.device = device
        self.format = format
        self.sample_count = sample_count
        self._text_instances = np.zeros(_MAX_INSTANCES * 3, dtype=np.uint32)
        self._text_count = 0
        self._selected_bbox = None
        self._hovered_bbox = None
        self._build_font_texture()
        self._build_buffers()
        self._build_pipelines()

    def _build_font_texture(self):
        atlas = _build_font_atlas()
        w, h = atlas.shape[1], atlas.shape[0]
        self.font_tex = self.device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.font_view = self.font_tex.create_view()
        self.device.queue.write_texture(
            {"texture": self.font_tex, "origin": (0, 0, 0)},
            atlas.tobytes(),
            {"bytes_per_row": w},
            (w, h, 1),
        )
        self.font_sampler = self.device.create_sampler(
            min_filter=wgpu.FilterMode.nearest,
            mag_filter=wgpu.FilterMode.nearest,
        )

    def _build_buffers(self):
        self.text_instance_buf = self.device.create_buffer(
            size=_MAX_INSTANCES * 3 * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.text_uniform_buf = self.device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.box_uniform_buf = self.device.create_buffer(
            size=80, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        # Pre-allocate vertex buffers for box rendering (reused each frame)
        self.solid_vbuf = self.device.create_buffer(
            size=36 * 12, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        self.line_vbuf = self.device.create_buffer(
            size=24 * 12, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

    def _build_pipelines(self):
        text_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
        ])
        text_mod = self.device.create_shader_module(code=_TEXT_WGSL)
        blend = {"alpha": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
                 "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"}}
        self.text_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[text_bgl]),
            vertex={"module": text_mod, "entry_point": "vs_main"},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": text_mod, "entry_point": "fs_main",
                      "targets": [{"format": self.format, "blend": blend}]},
            depth_stencil=None,
        )
        self.text_bgl = text_bgl

        box_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        box_mod = self.device.create_shader_module(code=_BOX_WGSL)
        box_vbuf = {"array_stride": 12, "attributes": [
            {"shader_location": 0, "format": wgpu.VertexFormat.float32x3, "offset": 0}]}
        ds = {"format": wgpu.TextureFormat.depth24plus,
              "depth_write_enabled": False,
              "depth_compare": wgpu.CompareFunction.less_equal}
        self.solid_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[box_bgl]),
            vertex={"module": box_mod, "entry_point": "vs_main", "buffers": [box_vbuf]},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": box_mod, "entry_point": "fs_main",
                      "targets": [{"format": self.format, "blend": blend}]},
            depth_stencil=ds,
        )
        self.line_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[box_bgl]),
            vertex={"module": box_mod, "entry_point": "vs_main", "buffers": [box_vbuf]},
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            fragment={"module": box_mod, "entry_point": "fs_main",
                      "targets": [{"format": self.format, "blend": blend}]},
            depth_stencil=ds,
        )
        self.box_bgl = box_bgl

    def update_text(self, lines):
        """Update text buffer with new lines (max 20 lines, 80 chars each)."""
        inst = np.zeros(_MAX_INSTANCES * 3, dtype=np.uint32)
        n = 0
        for row, line in enumerate(lines[:_MAX_LINES]):
            for col, ch in enumerate(line.upper()[:_MAX_CHARS]):
                code = ord(ch) if ch in _FONT else (ord('?') if '?' in _FONT else ord(' '))
                inst[n * 3] = col
                inst[n * 3 + 1] = row
                inst[n * 3 + 2] = code
                n += 1
        self._text_count = n
        self._text_instances = inst
        self.device.queue.write_buffer(self.text_instance_buf, 0, inst.tobytes())

    def set_selected_chunk(self, key=None, bbox=None):
        """Set/clear the selected (red solid) chunk."""
        self._selected_bbox = bbox

    def set_hovered_chunk(self, key=None, bbox=None):
        """Set/clear the hovered (yellow wireframe) chunk."""
        self._hovered_bbox = bbox

    def _write_box_uniform(self, mvp, color):
        data = np.zeros(20, dtype=np.float32)
        data[0:16] = mvp.flatten()
        data[16:20] = color
        self.device.queue.write_buffer(self.box_uniform_buf, 0, data.tobytes())

    def draw(self, encoder, color_view, depth_view, camera_view, camera_proj, screen_w, screen_h):
        """Draw text overlay + chunk highlights after terrain pass."""
        vp = (camera_proj @ camera_view).astype(np.float32)
        if self._selected_bbox is not None:
            verts = _box_solid_verts(self._selected_bbox)
            self.device.queue.write_buffer(self.solid_vbuf, 0, verts.tobytes())
            self._write_box_uniform(vp, np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32))
            bg = self.device.create_bind_group(
                layout=self.box_bgl,
                entries=[{"binding": 0, "resource": {"buffer": self.box_uniform_buf}}])
            pass_enc = encoder.begin_render_pass(
                color_attachments=[{"view": color_view, "load_op": "load", "store_op": "store"}],
                depth_stencil_attachment={"view": depth_view,
                                          "depth_load_op": "load", "depth_store_op": "store"})
            pass_enc.set_pipeline(self.solid_pipeline)
            pass_enc.set_bind_group(0, bg)
            pass_enc.set_vertex_buffer(0, self.solid_vbuf)
            pass_enc.draw(36)
            pass_enc.end()
        if self._hovered_bbox is not None:
            verts = _box_line_verts(self._hovered_bbox)
            self.device.queue.write_buffer(self.line_vbuf, 0, verts.tobytes())
            self._write_box_uniform(vp, np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32))
            bg = self.device.create_bind_group(
                layout=self.box_bgl,
                entries=[{"binding": 0, "resource": {"buffer": self.box_uniform_buf}}])
            pass_enc = encoder.begin_render_pass(
                color_attachments=[{"view": color_view, "load_op": "load", "store_op": "store"}],
                depth_stencil_attachment={"view": depth_view,
                                          "depth_load_op": "load", "depth_store_op": "store"})
            pass_enc.set_pipeline(self.line_pipeline)
            pass_enc.set_bind_group(0, bg)
            pass_enc.set_vertex_buffer(0, self.line_vbuf)
            pass_enc.draw(24)
            pass_enc.end()
        if self._text_count > 0:
            screen_data = np.array([float(screen_w), float(screen_h), 0.0, 0.0], dtype=np.float32)
            self.device.queue.write_buffer(self.text_uniform_buf, 0, screen_data.tobytes())
            bg = self.device.create_bind_group(
                layout=self.text_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self.text_uniform_buf}},
                    {"binding": 1, "resource": {"buffer": self.text_instance_buf}},
                    {"binding": 2, "resource": self.font_sampler},
                    {"binding": 3, "resource": self.font_view},
                ])
            pass_enc = encoder.begin_render_pass(
                color_attachments=[{"view": color_view, "load_op": "load", "store_op": "store"}])
            pass_enc.set_pipeline(self.text_pipeline)
            pass_enc.set_bind_group(0, bg)
            pass_enc.draw(6, self._text_count)
            pass_enc.end()
