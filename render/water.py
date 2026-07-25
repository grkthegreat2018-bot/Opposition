"""Transparent water plane with procedural waves."""

import numpy as np
import wgpu

WATER_SHADER = """
struct Uniforms {
    mvp: mat4x4<f32>,
    light_mvp: mat4x4<f32>,
    light_dir: vec3<f32>,
    _pad: f32,
    camera_pos: vec3<f32>,
    _pad2: f32,
    fog_color: vec3<f32>,
    _pad3: f32,
    fog_density: f32,
    fog_start: f32,
    time: f32,
    _pad4: f32,
    sun_color: vec3<f32>,
    _pad5: f32,
};

@binding(0) @group(0)
var<uniform> u: Uniforms;

@binding(1) @group(0)
var t_shadow: texture_depth_2d;

@binding(2) @group(0)
var s_shadow: sampler_comparison;

struct VertexIn {
    @location(0) pos: vec3<f32>,
};

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

fn wave_height(p: vec2<f32>, t: f32) -> f32 {
    var h = 0.0;
    h += 0.3 * sin(p.x * 0.5 + t * 1.0);
    h += 0.2 * sin(p.y * 0.4 + t * 0.8);
    h += 0.15 * sin((p.x + p.y) * 0.3 + t * 1.2);
    return h;
}

fn wave_deriv(p: vec2<f32>, t: f32) -> vec2<f32> {
    var d = vec2<f32>(0.0);
    d.x += 0.3 * 0.5 * cos(p.x * 0.5 + t * 1.0);
    d.y += 0.2 * 0.4 * cos(p.y * 0.4 + t * 0.8);
    let d3 = 0.15 * 0.3 * cos((p.x + p.y) * 0.3 + t * 1.2);
    d.x += d3;
    d.y += d3;
    return d;
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let WATER_LEVEL = 0.0;
    var world = vec3<f32>(in.pos.x, WATER_LEVEL, in.pos.z);
    // Keep the water patch centered under the camera.
    world.x += u.camera_pos.x;
    world.z += u.camera_pos.z;
    world.y += wave_height(world.xz, u.time);
    let deriv = wave_deriv(world.xz, u.time);
    let n = normalize(vec3<f32>(-deriv.x, 1.0, -deriv.y));
    var out: VertexOut;
    out.pos = u.mvp * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.normal = n;
    return out;
}

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    let t = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

fn procedural_sky_color(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y, 0.0, 1.0);
    let horizon = u.fog_color;
    let zenith = u.fog_color * 0.5 + vec3<f32>(0.1, 0.15, 0.3);
    return mix(horizon, zenith, pow(up, 0.5));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let view_dir = normalize(u.camera_pos - in.world_pos);
    let sun_dir = normalize(u.light_dir);
    let sun_color = u.sun_color;

    // Directional shadow.
    let light_clip = u.light_mvp * vec4<f32>(in.world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    var shadow = 1.0;
    if (all(abs(light_ndc.xy) <= vec2<f32>(1.0))) {
        let shadow_uv = vec2<f32>(light_ndc.x * 0.5 + 0.5, -light_ndc.y * 0.5 + 0.5);
        shadow = textureSampleCompare(t_shadow, s_shadow, shadow_uv, light_ndc.z - 0.005);
    }

    // High-frequency normal perturbation for sparkly micro-reflections.
    let np = in.world_pos.xz * 2.0 + u.time * 0.5;
    let nx = noise2(np) - 0.5;
    let nz = noise2(np + vec2<f32>(3.7, 1.3)) - 0.5;
    let perturbed_n = normalize(n + vec3<f32>(nx * 0.1, 0.0, nz * 0.1));

    let ndotl = max(dot(perturbed_n, sun_dir), 0.0);

    // Schlick Fresnel (water IOR 1.33 -> R0 = 0.02).
    let facing = max(dot(perturbed_n, view_dir), 0.0);
    let R0 = 0.02;
    let fresnel = R0 + (1.0 - R0) * pow(1.0 - facing, 5.0);

    // Reflection direction across perturbed normal.
    let reflect_dir = reflect(-view_dir, perturbed_n);

    // Procedural sky reflection.
    let sky_reflection = procedural_sky_color(reflect_dir);

    // Broad sun glitter streak on the waves.
    let sun_glitter = sun_color * pow(max(dot(reflect_dir, sun_dir), 0.0), 64.0) * 2.0 * shadow;

    // Deeper, more realistic water body color.
    let deep = vec3<f32>(0.01, 0.03, 0.08);
    let shallow = vec3<f32>(0.05, 0.25, 0.35);
    let water_body = mix(deep, shallow, fresnel)
        * (vec3<f32>(0.1, 0.12, 0.15) + sun_color * ndotl * shadow);

    let lit = mix(water_body, sky_reflection + sun_glitter, fresnel);

    let dist = length(in.world_pos - u.camera_pos);
    let fog = clamp(exp(-u.fog_density * max(dist - u.fog_start, 0.0)), 0.0, 1.0);
    let out_color = mix(u.fog_color, lit, fog);
    return vec4<f32>(out_color, 0.8);
}
"""


class WaterRenderer:
    """A large, transparent, camera-following water plane."""

    def __init__(self, renderer, size: float = 1024.0, res: int = 128):
        self.renderer = renderer
        self.device = renderer.device
        self.format = renderer.format
        self.size = size
        self.res = res
        self.pipeline = self._create_pipeline()
        self.vertex_buffer, self.index_buffer, self.index_count = self._create_mesh()

    def _create_mesh(self):
        n = self.res + 1
        xs = np.linspace(-self.size * 0.5, self.size * 0.5, n, dtype=np.float32)
        zs = np.linspace(-self.size * 0.5, self.size * 0.5, n, dtype=np.float32)
        pos = np.zeros((n * n, 3), dtype=np.float32)
        for i in range(n):
            base = i * n
            z = zs[i]
            for j in range(n):
                pos[base + j] = (xs[j], 0.0, z)

        indices = []
        for i in range(self.res):
            for j in range(self.res):
                a = i * n + j
                b = a + 1
                c = (i + 1) * n + j
                d = c + 1
                indices.extend([a, c, b, b, c, d])
        indices = np.array(indices, dtype=np.uint32)

        vb = self.device.create_buffer_with_data(
            data=pos,
            usage=wgpu.BufferUsage.VERTEX,
            label="water vertex buffer",
        )
        ib = self.device.create_buffer_with_data(
            data=indices,
            usage=wgpu.BufferUsage.INDEX,
            label="water index buffer",
        )
        return vb, ib, indices.size

    def _create_pipeline(self):
        shader = self.device.create_shader_module(code=WATER_SHADER, label="water shader")

        return self.device.create_render_pipeline(
            layout=self.renderer.pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 3 * 4,
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0}
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
                "depth_compare": "less",
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.format,
                        "blend": {
                            "color": {
                                "src_factor": "src-alpha",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                            "alpha": {
                                "src_factor": "one",
                                "dst_factor": "one-minus-src-alpha",
                                "operation": "add",
                            },
                        },
                    }
                ],
            },
            label="water pipeline",
        )

    def draw(self, encoder, color_view, depth_view):
        """Draw the water plane after opaque terrain."""
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": color_view,
                    "resolve_target": None,
                    "load_op": "load",
                    "store_op": "store",
                }
            ],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_load_op": "load",
                "depth_store_op": "store",
            },
            label="water pass",
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.renderer.bind_group)
        render_pass.set_vertex_buffer(0, self.vertex_buffer, 0, self.vertex_buffer.size)
        render_pass.set_index_buffer(self.index_buffer, "uint32", 0, self.index_buffer.size)
        render_pass.draw_indexed(self.index_count, 1, 0, 0, 0)
        render_pass.end()
