"""Full-screen sky renderer with sun, moon, stars, and day/night cycle."""

import math

import numpy as np
import wgpu


# ---------------------------------------------------------------------------
# Day/night parameter computation
# ---------------------------------------------------------------------------

def _smoothstep(edge0, edge1, x):
    """GLSL-style smoothstep that also handles edge0 > edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# Module-level constant color arrays (avoid per-frame allocation in
# compute_sky_params, which is called every frame).
_SKY_WHITE = np.array([1.0, 0.96, 0.88], dtype=np.float32)
_SKY_ORANGE = np.array([1.0, 0.5, 0.2], dtype=np.float32)
_SKY_BLACK = np.array([0.0, 0.0, 0.0], dtype=np.float32)
_SKY_DAY_ZENITH = np.array([0.25, 0.45, 0.85], dtype=np.float32)
_SKY_SUNSET_ZENITH = np.array([0.15, 0.20, 0.40], dtype=np.float32)
_SKY_NIGHT_ZENITH = np.array([0.02, 0.03, 0.08], dtype=np.float32)
_SKY_DAY_HORIZON = np.array([0.55, 0.75, 0.95], dtype=np.float32)
_SKY_SUNSET_HORIZON = np.array([0.95, 0.45, 0.20], dtype=np.float32)
_SKY_NIGHT_HORIZON = np.array([0.05, 0.08, 0.15], dtype=np.float32)
_SKY_MOON_COLOR = np.array([0.7, 0.75, 0.85], dtype=np.float32)


def compute_sky_params(time_of_day):
    """Compute sky parameters from time of day in hours [0, 24).

    Returns a dict of np.float32 arrays/scalars describing the sun, moon,
    stars, and sky gradient colors for the given time of day.
    """
    # Sun angle: 0 rad at 6am (rising in +x), pi/2 at noon (zenith),
    # pi at 6pm (setting in -x), -pi/2 at midnight (below horizon).
    angle = (time_of_day - 6.0) * (math.pi / 12.0)

    sun_dir = np.array(
        [math.cos(angle), math.sin(angle), 0.0], dtype=np.float32
    )
    norm = float(np.linalg.norm(sun_dir))
    if norm > 0.0:
        sun_dir = sun_dir / norm

    sun_elevation = float(sun_dir[1])  # sin(angle), range [-1, 1]

    # Moon is always opposite the sun.
    moon_dir = (-sun_dir).astype(np.float32)
    moon_elevation = -sun_elevation

    # --- Sun color: black below horizon, orange at horizon, white at zenith ---
    t_above = _smoothstep(-0.05, 0.05, sun_elevation)  # 0 below, 1 above
    t_high = _smoothstep(0.0, 0.5, sun_elevation)  # 0 at horizon, 1 high
    day_color = _SKY_ORANGE * (1.0 - t_high) + _SKY_WHITE * t_high
    sun_color = _SKY_BLACK * (1.0 - t_above) + day_color * t_above

    # --- Intensities ---
    sun_intensity = np.float32(_smoothstep(0.0, 0.1, sun_elevation))
    moon_intensity = np.float32(_smoothstep(0.0, 0.5, moon_elevation) * 0.15)
    star_intensity = np.float32(_smoothstep(0.0, -0.3, sun_elevation))

    # --- Sky gradient colors ---
    t_day = _smoothstep(0.0, 0.3, sun_elevation)  # 1 in day, 0 at/below horizon
    t_night = _smoothstep(0.0, -0.3, sun_elevation)  # 1 at night, 0 at/above

    zenith_day_sunset = _SKY_SUNSET_ZENITH * (1.0 - t_day) + _SKY_DAY_ZENITH * t_day
    sky_zenith = (zenith_day_sunset * (1.0 - t_night) + _SKY_NIGHT_ZENITH * t_night).astype(
        np.float32
    )

    horizon_day_sunset = _SKY_SUNSET_HORIZON * (1.0 - t_day) + _SKY_DAY_HORIZON * t_day
    sky_horizon = (
        horizon_day_sunset * (1.0 - t_night) + _SKY_NIGHT_HORIZON * t_night
    ).astype(np.float32)

    moon_color = _SKY_MOON_COLOR

    return {
        "sun_dir": sun_dir,
        "sun_color": sun_color.astype(np.float32),
        "sun_intensity": sun_intensity,
        "moon_dir": moon_dir,
        "moon_color": moon_color,
        "moon_intensity": moon_intensity,
        "star_intensity": star_intensity,
        "sky_zenith": sky_zenith,
        "sky_horizon": sky_horizon,
        "milky_way_intensity": np.float32(0.8),
    }


# ---------------------------------------------------------------------------
# WGSL shader
# ---------------------------------------------------------------------------

SKY_SHADER = """
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
    milky_way_intensity: f32,
    _pad6: f32,
};

@binding(0) @group(0)
var<uniform> u: SkyUniforms;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) view_dir: vec3<f32>,
};

fn hash3(p: vec3<f32>) -> f32 {
    let q = vec3<f32>(dot(p, vec3<f32>(127.1, 311.7, 74.7)),
                      dot(p, vec3<f32>(269.5, 183.3, 246.1)),
                      dot(p, vec3<f32>(113.5, 271.9, 124.6)));
    return fract(sin(q.x + q.y + q.z) * 43758.5453);
}

// 3D hash returning a vec3 of independent floats (for per-cell randomness).
fn hash33(p: vec3<f32>) -> vec3<f32> {
    let q = vec3<f32>(dot(p, vec3<f32>(127.1, 311.7, 74.7)),
                      dot(p, vec3<f32>(269.5, 183.3, 246.1)),
                      dot(p, vec3<f32>(113.5, 271.9, 124.6)));
    return fract(sin(q) * vec3<f32>(43758.5453, 22578.1459, 19312.7317));
}

// 2D value noise for Milky Way clumping.
fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let a = hash33(vec3<f32>(i, 0.0)).x;
    let b = hash33(vec3<f32>(i + vec2<f32>(1.0, 0.0), 0.0)).x;
    let c = hash33(vec3<f32>(i + vec2<f32>(0.0, 1.0), 0.0)).x;
    let d = hash33(vec3<f32>(i + vec2<f32>(1.0, 1.0), 0.0)).x;
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 2D fractal Brownian motion (3 octaves) for Milky Way clumping.
fn fbm2(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pp = p;
    for (var i = 0; i < 3; i = i + 1) {
        v = v + a * noise2(pp);
        pp = pp * 2.0;
        a = a * 0.5;
    }
    return v;
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    // Full-screen triangle: 3 vertices cover the entire screen with no
    // vertex buffer needed.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    var out: VertexOut;
    let p = positions[vid];
    // Place at the far plane (z=1 in NDC) so depth test only passes where
    // no terrain has been drawn (depth buffer still at clear value 1.0).
    out.pos = vec4<f32>(p, 1.0, 1.0);
    // Unproject the clip-space corner to world space to obtain the view
    // direction for this fragment.
    let world = u.inv_view_proj * vec4<f32>(p, 1.0, 1.0);
    let world_pos = world.xyz / world.w;
    out.view_dir = normalize(world_pos - u.camera_pos);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let view_dir = normalize(in.view_dir);
    let t = u.cloud_time;

    // --- 1. Three-color sky gradient (horizon -> middle -> zenith) ---
    let middle_color = mix(u.sky_horizon, u.sky_zenith, 0.5) * vec3<f32>(1.05, 1.0, 0.95);
    let middle_threshold = smoothstep(0.0, 0.35, view_dir.y);
    let top_threshold = smoothstep(0.35, 0.8, view_dir.y);
    var sky_gradient = mix(u.sky_horizon, middle_color, middle_threshold);
    sky_gradient = mix(sky_gradient, u.sky_zenith, top_threshold);

    // --- 2. Sun disc + multi-layer glow + atmospheric halo ---
    var sun_total = vec3<f32>(0.0);
    if (u.sun_intensity > 0.01) {
        let sd = max(dot(view_dir, u.sun_dir), 0.0);
        let sun_disc = u.sun_color * u.sun_intensity * pow(sd, 250.0);
        let sun_glow = u.sun_color * u.sun_intensity
            * (pow(sd, 4.0) * 0.3 + pow(sd, 16.0) * 0.5 + pow(sd, 64.0) * 0.8);
        // Wider warm halo near the horizon (atmospheric scattering).
        let atmo_halo = u.sun_color * pow(sd, 2.0) * (1.0 - view_dir.y)
            * u.sun_intensity * 0.5;
        sun_total = sun_disc + sun_glow + atmo_halo;
    }

    // --- 3. Moon disc + glow + earthshine ---
    var moon_total = vec3<f32>(0.0);
    if (u.moon_intensity > 0.01) {
        let md = max(dot(view_dir, u.moon_dir), 0.0);
        let moon_disc = u.moon_color * u.moon_intensity * pow(md, 800.0);
        let moon_glow = u.moon_color * u.moon_intensity * pow(md, 32.0) * 0.15;
        // Earthshine: faint ambient on the dark side of the moon.
        let earthshine = u.moon_color * u.moon_intensity * pow(md, 2000.0) * 0.0
            + smoothstep(0.998, 0.999, md) * 0.05;
        moon_total = moon_disc + moon_glow + earthshine;
    }

    // --- 4. Stars: blackbody colors, magnitude brightness, twinkle, spikes ---
    var stars = vec3<f32>(0.0);
    if (u.star_intensity > 0.0) {
        let scale = 350.0;
        let hp = floor(view_dir * scale);
        let h = hash3(hp);
        let h3 = hash33(hp);
        // Cell-local position for soft-edged stars.
        let cell_pos = fract(view_dir * scale);
        let d2 = dot(cell_pos - 0.5, cell_pos - 0.5);
        let dist_to_center = sqrt(d2);
        // Star candidates: ~0.5% of cells.
        let is_star = step(0.9955, h);
        // Magnitude-based brightness: pow(h,4) skews so most stars are dim.
        let magnitude = pow(h, 4.0);
        let bright_star_factor = smoothstep(0.7, 1.0, magnitude);
        // Soft circular star with smoothstep falloff.
        let soft = smoothstep(0.5, 0.0, dist_to_center);
        // Multi-frequency twinkle (3 sine waves, hash-seeded phases).
        let p1 = h3.x * 6.2831;
        let p2 = h3.y * 6.2831;
        let p3 = h3.z * 6.2831;
        let twinkle = 0.7
            + 0.15 * sin(t * 1.3 + p1)
            + 0.1 * sin(t * 2.7 + p2)
            + 0.05 * sin(t * 5.1 + p3);
        // Diffraction spikes on bright stars (4-point cross).
        let dx = abs(cell_pos.x - 0.5);
        let dy = abs(cell_pos.y - 0.5);
        let spike = max(dx, dy);
        let spike_intensity = pow(1.0 - spike, 8.0) * bright_star_factor;
        // Blackbody spectral color based on hash class index.
        let class_idx = h3.x * 7.0;
        var star_color = vec3<f32>(1.0);
        if (class_idx < 1.0) {
            star_color = vec3<f32>(0.6, 0.7, 1.0);  // O: blue
        } else if (class_idx < 2.0) {
            star_color = vec3<f32>(0.8, 0.9, 1.0);  // B: blue-white
        } else if (class_idx < 3.0) {
            star_color = vec3<f32>(1.0, 1.0, 1.0);  // A: white
        } else if (class_idx < 4.0) {
            star_color = vec3<f32>(1.0, 1.0, 0.9);  // F: yellow-white
        } else if (class_idx < 5.0) {
            star_color = vec3<f32>(1.0, 0.95, 0.8); // G: yellow (Sun)
        } else if (class_idx < 6.0) {
            star_color = vec3<f32>(1.0, 0.8, 0.6);  // K: orange
        } else {
            star_color = vec3<f32>(1.0, 0.6, 0.4);  // M: red
        }
        let brightness = is_star * (magnitude + spike_intensity) * soft * twinkle;
        stars = star_color * brightness * u.star_intensity;
    }

    // --- 5. Milky Way band with FBM clumping and dust lane ---
    var milky_way = vec3<f32>(0.0);
    if (u.milky_way_intensity > 0.0 && u.star_intensity > 0.0) {
        let milky_way_dir = normalize(vec3<f32>(1.0, 0.0, 0.3));
        let band_normal = normalize(cross(milky_way_dir, vec3<f32>(0.0, 1.0, 0.0)));
        let band_dist = abs(dot(view_dir, band_normal));
        let band = exp(-band_dist * band_dist * 8.0);
        // FBM clumping along the band.
        let clump = fbm2(view_dir.xz * 3.0);
        // Dust lane: subtract a thinner darker band.
        let dust = exp(-band_dist * band_dist * 30.0) * 0.3;
        let band_glow = (band * clump - dust) * u.milky_way_intensity;
        milky_way = max(band_glow, 0.0) * vec3<f32>(0.15, 0.12, 0.1) * u.star_intensity;
        // Extra dense stars in the band region (finer grid weighted by band).
        let mw_scale = 600.0;
        let mhp = floor(view_dir * mw_scale);
        let mh = hash3(mhp);
        let mcell = fract(view_dir * mw_scale);
        let md2 = dot(mcell - 0.5, mcell - 0.5);
        let msoft = smoothstep(0.5, 0.0, sqrt(md2));
        let mis_star = step(0.992, mh);
        let mtw = 0.7 + 0.3 * sin(t * 2.0 + mh * 6.2831);
        let mw_stars = mis_star * msoft * mtw * band * u.milky_way_intensity;
        milky_way = milky_way + vec3<f32>(0.8, 0.85, 1.0) * mw_stars * u.star_intensity;
    }

    // --- 6. Combine ---
    let color = sky_gradient
        + sun_total
        + moon_total
        + stars * (1.0 - u.sun_intensity)
        + milky_way * (1.0 - u.sun_intensity);

    return vec4<f32>(color, 1.0);
}
"""


# ---------------------------------------------------------------------------
# SkyRenderer
# ---------------------------------------------------------------------------

class SkyRenderer:
    """Full-screen sky background rendered behind the terrain.

    Draws a single full-screen triangle (no vertex buffer) with depth
    compare set to ``less_equal`` and depth writes disabled, so the sky
    only appears where the depth buffer is still at the clear value
    (i.e. where no terrain was drawn).
    """

    def __init__(self, renderer):
        self.renderer = renderer
        self.device = renderer.device
        self.format = renderer.format

        # 48 floats = 192 bytes (mat4x4 + packed vec3/f32 pairs).
        self.uniform_buffer = self.device.create_buffer(
            size=48 * 4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="sky uniform buffer",
        )
        # Reusable staging buffer (avoids per-frame alloc).
        self._uniform_staging = np.zeros(48, dtype=np.float32)

        self.pipeline = self._create_pipeline()

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
            label="sky bind group",
        )

    def _create_pipeline(self):
        shader = self.device.create_shader_module(
            code=SKY_SHADER, label="sky shader"
        )

        # The sky shader uses its own uniform struct, so it needs its own
        # bind group layout — distinct from the terrain renderer's shared one.
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX
                    | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": "uniform",
                        "has_dynamic_offset": False,
                    },
                }
            ],
            label="sky bind group layout",
        )
        self.pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout],
            label="sky pipeline layout",
        )

        return self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                # No vertex buffer — the full-screen triangle is generated
                # entirely from @builtin(vertex_index).
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
                "depth_compare": "less-equal",
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.format,
                        "blend": {
                            "color": {
                                "src_factor": "one",
                                "dst_factor": "zero",
                                "operation": "add",
                            },
                            "alpha": {
                                "src_factor": "one",
                                "dst_factor": "zero",
                                "operation": "add",
                            },
                        },
                    }
                ],
            },
            label="sky pipeline",
        )

    def update_uniforms(self, inv_view_proj, camera_pos, sky_params, cloud_time):
        """Write the 48-float uniform buffer.

        Layout (float indices into the 192-byte buffer)::

            0-15  inv_view_proj (mat4x4, column-major)
            16-18 camera_pos
            19    _pad
            20-22 sun_dir
            23    sun_intensity
            24-26 sun_color
            27    _pad2
            28-30 sky_zenith
            31    _pad3
            32-34 sky_horizon
            35    _pad4
            36-38 moon_dir
            39    moon_intensity
            40-42 moon_color
            43    star_intensity
            44    cloud_time
            45    milky_way_intensity
            46    _pad6
            47    (implicit WGSL struct padding)
        """
        data = self._uniform_staging
        data.fill(0)
        data[:16] = np.asarray(inv_view_proj, dtype=np.float32).T.flatten()
        data[16:19] = np.asarray(camera_pos, dtype=np.float32)
        data[20:23] = np.asarray(sky_params["sun_dir"], dtype=np.float32)
        data[23] = float(sky_params["sun_intensity"])
        data[24:27] = np.asarray(sky_params["sun_color"], dtype=np.float32)
        data[28:31] = np.asarray(sky_params["sky_zenith"], dtype=np.float32)
        data[32:35] = np.asarray(sky_params["sky_horizon"], dtype=np.float32)
        data[36:39] = np.asarray(sky_params["moon_dir"], dtype=np.float32)
        data[39] = float(sky_params["moon_intensity"])
        data[40:43] = np.asarray(sky_params["moon_color"], dtype=np.float32)
        data[43] = float(sky_params["star_intensity"])
        data[44] = float(cloud_time)
        data[45] = float(sky_params.get("milky_way_intensity", 0.8))
        self.device.queue.write_buffer(self.uniform_buffer, 0, data)

    def draw(self, encoder, color_view, depth_view):
        """Draw the sky after the opaque terrain pass.

        Uses ``load_op='load'`` for both color and depth so existing
        terrain rendering is preserved.  The full-screen triangle only
        writes where the depth buffer is still 1.0 (no terrain).
        """
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
            label="sky pass",
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
