"""Volumetric raymarched cloud renderer (full-screen, no mesh).

Renders a full-screen triangle and reconstructs the world-space view ray per
pixel using ``inv_view_proj``.  The ray is intersected with a bounded cloud
layer (altitude 20-80, centred on the camera xz) and marched in 32 steps,
sampling 3D FBM noise for density.  Light attenuation toward the sun is
estimated with a short 4-step light march (self-shadowing), and scattered
radiance is accumulated via Beer-Lambert transmittance and a
Henyey-Greenstein phase function for forward scattering.

Reference: jeantimex/procedural-clouds (WebGPU/WGSL).
"""

import numpy as np
import wgpu

CLOUD_SHADER = r"""
struct SkyUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
    sun_dir: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    _pad2: f32,
    sky_zenith: vec3<f32>,
    _pad3: f32,
    sky_horizon: vec3<f32>,
    _pad4: f32,
    moon_dir: vec3<f32>,
    moon_intensity: f32,
    moon_color: vec3<f32>,
    star_intensity: f32,
    cloud_time: f32,
    _pad5: f32,
    _pad6: f32,
};

@binding(0) @group(0) var<uniform> u: SkyUniforms;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// --- 3D hash-based value noise + fbm ---
fn hash3(p: vec3<f32>) -> f32 {
    let q = vec3<f32>(dot(p, vec3<f32>(127.1, 311.7, 74.7)),
                      dot(p, vec3<f32>(269.5, 183.3, 246.1)),
                      dot(p, vec3<f32>(113.5, 271.9, 124.6)));
    return fract(sin(dot(q, vec3<f32>(1.0, 1.0, 1.0))) * 43758.5453);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let a = hash3(i);
    let b = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let f1 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let g = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let h = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let t = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(a, b, t.x), mix(c, d, t.x), t.y),
        mix(mix(e, f1, t.x), mix(g, h, t.x), t.y),
        t.z
    );
}

fn fbm3(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var fp = p;
    for (var i = 0; i < 5; i = i + 1) {
        v += a * noise3(fp);
        fp *= 2.0;
        a *= 0.5;
    }
    return v;
}

// Henyey-Greenstein phase function for forward scattering.
fn hgPhase(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

// Interleaved gradient noise for ray-start dithering (reduces banding).
fn ign(uv: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

// Cloud density at a world position.  The cloud layer lives between
// CLOUD_BOTTOM and CLOUD_TOP; an altitude gradient fades the edges so clouds
// have soft tops and bottoms.  Wind drifts the FBM pattern over time.
fn cloud_density(pos: vec3<f32>) -> f32 {
    let CLOUD_BOTTOM = 20.0;
    let CLOUD_TOP = 80.0;
    let alt_frac = (pos.y - CLOUD_BOTTOM) / (CLOUD_TOP - CLOUD_BOTTOM);
    let alt_gradient = smoothstep(0.0, 0.2, alt_frac) * smoothstep(1.0, 0.8, alt_frac);
    let wind = vec3<f32>(u.cloud_time * 0.8, 0.0, u.cloud_time * 0.3);
    let n = fbm3(pos * 0.02 + wind);
    let coverage = smoothstep(0.45, 0.65, n);
    return coverage * alt_gradient;
}

// Ray-box intersection for the cloud volume.  The box is centred on the
// camera xz but uses a fixed world altitude range.  Returns (tNear, tFar)
// or (-1, -1) if the ray misses the box.
fn intersect_cloud_box(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    let box_min = vec3<f32>(ro.x - 500.0, 20.0, ro.z - 500.0);
    let box_max = vec3<f32>(ro.x + 500.0, 80.0, ro.z + 500.0);
    let invRd = 1.0 / rd;
    let t0 = (box_min - ro) * invRd;
    let t1 = (box_max - ro) * invRd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let tNear = max(max(tmin.x, tmin.y), tmin.z);
    let tFar = min(min(tmax.x, tmax.y), tmax.z);
    if (tFar < max(tNear, 0.0)) {
        return vec2<f32>(-1.0, -1.0);
    }
    return vec2<f32>(max(tNear, 0.0), tFar);
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    // Full-screen triangle: 3 vertices cover the entire screen with no
    // vertex buffer needed.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOut;
    let p = positions[vid];
    out.pos = vec4<f32>(p, 1.0, 1.0);
    out.uv = p;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Reconstruct the world-space ray from the camera through this pixel.
    let near = u.inv_view_proj * vec4<f32>(in.uv, 0.0, 1.0);
    let far = u.inv_view_proj * vec4<f32>(in.uv, 1.0, 1.0);
    let ro = u.camera_pos;
    let rd = normalize(far.xyz / far.w - near.xyz / near.w);

    // Intersect the ray with the bounded cloud volume.
    let hit = intersect_cloud_box(ro, rd);
    if (hit.x < 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let tEntry = hit.x;
    let tExit = hit.y;
    let NUM_STEPS = 32;
    let stepSize = (tExit - tEntry) / f32(NUM_STEPS);
    let dither = ign(vec2<f32>(in.pos.x, in.pos.y));

    var pos = ro + rd * (tEntry + stepSize * dither);
    var transmittance = 1.0;
    var color = vec3<f32>(0.0);

    let sun_dir = normalize(u.sun_dir);
    let cosTheta = dot(rd, sun_dir);
    let phase = mix(1.0, hgPhase(cosTheta, 0.4), 0.7);

    let ambient = u.sky_horizon * 0.4;
    let sun_color = u.sun_color * u.sun_intensity;

    for (var i = 0; i < NUM_STEPS; i = i + 1) {
        let d = cloud_density(pos);
        if (d > 0.01) {
            // Light march toward the sun (short, 4 steps) for self-shadowing.
            var shadow = 0.0;
            let lightStep = 2.0;
            for (var j = 1; j <= 4; j = j + 1) {
                let lp = pos + sun_dir * f32(j) * lightStep;
                shadow += cloud_density(lp) * lightStep;
            }
            let light_transmittance = exp(-shadow * 0.8);

            let step_trans = exp(-d * stepSize * 0.3);
            let scattering = light_transmittance * phase * (1.0 - step_trans);
            let lit_color = sun_color * scattering + ambient * (1.0 - step_trans) * 0.3;
            color += transmittance * (1.0 - step_trans) * lit_color;
            transmittance *= step_trans;

            if (transmittance < 0.01) {
                break;
            }
        }
        pos += rd * stepSize;
    }

    // Sky colour shows through where clouds are thin.
    let sky_dir_y = clamp(rd.y, 0.0, 1.0);
    let sky_color = mix(u.sky_horizon, u.sky_zenith, smoothstep(0.0, 0.5, sky_dir_y));
    let final_color = color + transmittance * sky_color;

    // Alpha = how much the clouds blocked the sky.
    let alpha = 1.0 - transmittance;
    return vec4<f32>(final_color, alpha * (u.sun_intensity * 0.9 + 0.1));
}
"""


class CloudRenderer:
    """Full-screen volumetric raymarched cloud renderer.

    Uses its own bind-group / pipeline layout (same SkyUniforms struct as the
    sky renderer) so it can be drawn independently in its own render pass.
    No mesh is needed — a full-screen triangle is generated in the vertex
    shader via ``@builtin(vertex_index)``.
    """

    # SkyUniforms layout: 48 floats = 192 bytes (struct is 16-byte aligned).
    UNIFORM_FLOATS = 48

    def __init__(self, renderer):
        self.renderer = renderer
        self.device = renderer.device
        self.format = renderer.format

        # Own uniform buffer (same layout as SkyRenderer's SkyUniforms).
        self.uniform_buffer = self.device.create_buffer(
            size=self.UNIFORM_FLOATS * 4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="cloud uniform buffer",
        )
        # Reusable staging buffer (avoids per-frame alloc).
        self._uniform_staging = np.zeros(self.UNIFORM_FLOATS, dtype=np.float32)

        self._create_pipeline()

    # -------------------------------------------------------------- pipeline
    def _create_pipeline(self):
        shader = self.device.create_shader_module(code=CLOUD_SHADER, label="cloud shader")

        # Own bind-group layout: single uniform buffer, visible to both stages.
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform", "has_dynamic_offset": False},
                }
            ],
            label="cloud bind group layout",
        )
        self.pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout],
            label="cloud pipeline layout",
        )

        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": self.uniform_buffer.size,
                    },
                }
            ],
            label="cloud bind group",
        )

        # Clouds: depth-tested against terrain (less) but never write depth.
        # No vertex buffers — the full-screen triangle is generated in the
        # vertex shader.  Alpha blend for semi-transparent cloud edges.
        self.pipeline = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [],
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
            label="cloud pipeline",
        )

    # ------------------------------------------------------------- uniforms
    def update_uniforms(self, inv_view_proj, camera_pos, sky_params, current_time):
        """Write the 48-float SkyUniforms buffer.

        ``inv_view_proj`` is the clip->world matrix (same as the sky renderer
        receives).  The volumetric shader needs it directly to reconstruct
        world-space rays per pixel, so it is stored as-is.
        """
        data = self._uniform_staging
        data.fill(0)
        # [0:16]  inv_view_proj (clip -> world)
        data[:16] = np.asarray(inv_view_proj, dtype=np.float32).T.flatten()
        # [16:19] camera_pos + [19] pad
        data[16:19] = np.asarray(camera_pos, dtype=np.float32)[:3]
        # [20:23] sun_dir + [23] sun_intensity
        data[20:23] = np.asarray(sky_params["sun_dir"], dtype=np.float32)[:3]
        data[23] = float(sky_params["sun_intensity"])
        # [24:27] sun_color + [27] pad2
        data[24:27] = np.asarray(sky_params["sun_color"], dtype=np.float32)[:3]
        # [28:31] sky_zenith + [31] pad3
        data[28:31] = np.asarray(sky_params["sky_zenith"], dtype=np.float32)[:3]
        # [32:35] sky_horizon + [35] pad4
        data[32:35] = np.asarray(sky_params["sky_horizon"], dtype=np.float32)[:3]
        # [36:39] moon_dir + [39] moon_intensity
        data[36:39] = np.asarray(sky_params["moon_dir"], dtype=np.float32)[:3]
        data[39] = float(sky_params["moon_intensity"])
        # [40:43] moon_color + [43] star_intensity
        data[40:43] = np.asarray(sky_params["moon_color"], dtype=np.float32)[:3]
        data[43] = float(sky_params["star_intensity"])
        # [44] cloud_time + [45:47] padding
        data[44] = float(current_time)
        self.device.queue.write_buffer(self.uniform_buffer, 0, data)

    # ------------------------------------------------------------------ draw
    def draw(self, encoder, color_view, depth_view):
        """Draw the volumetric clouds after the sky pass, before the water pass."""
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
            label="cloud pass",
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
