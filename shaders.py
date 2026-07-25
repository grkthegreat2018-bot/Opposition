"""WGSL shader source for the terrain renderer.

Extracted from renderer.py so the shader code is editable independently
of the Python pipeline/draw orchestration.
"""

SHADER = """
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
    @location(1) normal: vec3<f32>,
    @location(2) biome: vec4<f32>,
    @location(3) sc: vec2<f32>,
};

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) biome: vec4<f32>,
    @location(4) sc: vec2<f32>,
};

// --- procedural detail noise ---
fn hash3(p: vec3<f32>) -> f32 {
    let h = fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    return h;
}

// 2D hash → vec2. Used for Voronoi cell centers in rock_crack.
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let h = fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
    let g = fract(sin(dot(p, vec2<f32>(269.5, 183.3))) * 12345.6789);
    return vec2<f32>(h, g);
}

fn value_noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let v000 = hash3(i);
    let v100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let v010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let v110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let v001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let v101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let v011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let v111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));

    let x0 = mix(mix(v000, v100, u.x), mix(v010, v110, u.x), u.y);
    let x1 = mix(mix(v001, v101, u.x), mix(v011, v111, u.x), u.y);
    return mix(x0, x1, u.z);
}

fn fbm3(p: vec3<f32>, octaves: i32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var f = 1.0;
    for (var i=0; i<octaves; i++) {
        v += a * value_noise3(p * f);
        f *= 2.0;
        a *= 0.5;
    }
    return v;
}

// --- PBR helper functions (GGX / Smith / Schlick Fresnel) ---
// Reference: Real-Time Rendering 4th ed., Karis "Real Shading in UE4".
fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

// GGX/Trowbridge-Reitz normal distribution.
fn d_ggx(nh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * d * d, 1e-6);
}

// Smith geometry (combined with Schlick-GGX k = a/2).
fn g_smith(nv: f32, nl: f32, a: f32) -> f32 {
    let k = a * 0.5;
    let gv = nv / (nv * (1.0 - k) + k);
    let gl = nl / (nl * (1.0 - k) + k);
    return gv * gl;
}

// Schlick Fresnel: f0 at normal incidence, blends to 1 at grazing angles.
fn f_schlick(hv: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow5(1.0 - hv);
}

// --- Material detail helpers ---

// Wind-aligned sand ripples. Projects world xz onto a fixed wind direction
// and stacks sine waves at decreasing wavelength. Returns a vec2:
//   x = primary ripple (wind-aligned), y = cross-hatch ripple (30° offset).
// The caller combines them: x modulates albedo, y perturbs the normal.
fn sand_ripples(p: vec3<f32>) -> vec2<f32> {
    let wind_dir = vec2<f32>(0.7, 0.7);  // normalized below
    let wd = normalize(wind_dir);
    let proj = dot(p.xz, wd);
    // Slow noise modulates ripple amplitude so some areas have strong
    // ripples and others are nearly flat sand (wind exposure varies).
    let amp_mod = 0.4 + 0.6 * fbm3(p * 0.05, 2);
    // Four octaves of ripples. Spacing ~1.5 / 0.6 / 0.25 / 0.1 world units.
    // The 4th octave is very high frequency for fine grain texture.
    let r1 = sin(proj * 4.2 + fbm3(p * 0.3, 2) * 3.0);
    let r2 = sin(proj * 10.5 + fbm3(p * 0.7, 2) * 5.0);
    let r3 = sin(proj * 26.0 + fbm3(p * 1.5, 2) * 7.0);
    let r4 = sin(proj * 62.8 + fbm3(p * 3.0, 2) * 9.0);  // ~0.1 unit spacing
    // Weighted sum, normalized to ~[-1, 1], amplitude-modulated.
    let primary = (r1 * 0.50 + r2 * 0.27 + r3 * 0.15 + r4 * 0.08) * amp_mod;
    // Cross-hatch: second ripple set at 30° offset from wind dir, lower
    // amplitude — simulates wind from multiple directions over time.
    let cross_dir = vec2<f32>(
        wd.x * 0.866 - wd.y * 0.5,   // rotate +30°
        wd.x * 0.5   + wd.y * 0.866,
    );
    let proj_c = dot(p.xz, cross_dir);
    let c1 = sin(proj_c * 8.0 + fbm3(p * 0.4, 2) * 4.0);
    let c2 = sin(proj_c * 20.0 + fbm3(p * 1.0, 2) * 6.0);
    let cross = (c1 * 0.6 + c2 * 0.4) * amp_mod * 0.5;  // half strength
    return vec2<f32>(primary, cross);
}

// Sedimentary rock strata: horizontal banding on cliff faces. Bands follow
// world height with noise-perturbed boundaries. Returns a vec2:
//   x = band tint intensity in [0,1] for albedo tinting,
//   y = roughness modifier (harder bands smoother, softer bands rougher).
// Per-band color variation and fine vertical cracking are included.
fn rock_strata(p: vec3<f32>, slope: f32) -> vec2<f32> {
    // Band spacing ~2.2 world units, perturbed by low-freq noise for wavy
    // strata. Quantize to discrete bands, then smoothstep for soft edges.
    let band_freq = 0.45;
    let perturb = fbm3(p * 0.08, 3) * 1.5;
    let h = p.y * band_freq + perturb;
    let band = abs(fract(h) - 0.5) * 2.0;  // 0 at band center, 1 at edge
    // Sharper bands on steeper terrain; fade out on flats.
    let sharp = mix(0.7, 0.95, smoothstep(0.4, 0.8, slope));
    let band_intensity = smoothstep(sharp, 1.0, band);
    // Per-band color variation: hash the band index so some bands are
    // darker/harder and some lighter/softer. This gives each sedimentary
    // layer a distinct character.
    let band_idx = floor(h);
    let band_hash = hash3(vec3<f32>(band_idx, 0.0, 0.0));
    // Darker bands (low hash) are harder → smoother; lighter (high hash)
    // are softer → rougher. Roughness modifier in [-0.15, +0.15].
    let rough_mod = (band_hash - 0.5) * 0.30;
    // Very fine vertical cracking: thin dark lines perpendicular to strata
    // (i.e. along world y). High-frequency noise threshold creates cracks
    // that only appear within softer (rougher) bands.
    let crack_noise = fbm3(p * 8.0, 2);
    let crack = smoothstep(0.78, 0.82, crack_noise) * (0.5 + band_hash);
    // Cracks darken the band tint locally.
    let tint_with_crack = clamp(band_intensity + crack * 0.4, 0.0, 1.0);
    return vec2<f32>(tint_with_crack, rough_mod);
}

// Fine grain noise for sand surfaces. Very high frequency (scale ~15.0)
// hash-based noise that modulates albedo brightness slightly (±10%).
// Only visible on flat sand — the caller weights it by sand biome weights
// and a slope fade so it vanishes where rock takes over.
fn sand_grain(p: vec3<f32>) -> f32 {
    let s = p * 15.0;
    // Hash-based value noise at fine scale for individual grain sparkle.
    let i = floor(s);
    let f = fract(s);
    let u = f * f * (3.0 - 2.0 * f);
    let g000 = hash3(i);
    let g100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let g010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let g110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let g001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let g101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let g011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let g111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x0 = mix(mix(g000, g100, u.x), mix(g010, g110, u.x), u.y);
    let x1 = mix(mix(g001, g101, u.x), mix(g011, g111, u.x), u.y);
    return mix(x0, x1, u.z);  // [0, 1]
}

// Procedural crack pattern on rock faces using Voronoi-like cell noise.
// Uses hash2 to create cell centers, finds distance to nearest cell edge.
// Cracks are dark thin lines where distance to edge < threshold. Fades out
// on very steep or very flat terrain — cracks show best on moderate slopes.
// Returns 0 = no crack, 1 = fully dark crack.
fn rock_crack(p: vec3<f32>, slope: f32) -> f32 {
    // Project onto the rock face plane using xz. Scale controls crack density.
    let scale = 3.0;
    let sp = p.xz * scale;
    let cell = floor(sp);
    let fcell = fract(sp);
    // Find distance to nearest cell edge (F1 Voronoi edge distance).
    var min_dist = 1.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let neighbor = vec2<f32>(f32(i), f32(j));
            let center = hash2(cell + neighbor) * 0.8 + 0.1;  // jitter in [0.1,0.9]
            let d = length(neighbor + center - fcell);
            min_dist = min(min_dist, d);
        }
    }
    // Crack threshold: thin dark lines near cell edges.
    let crack = 1.0 - smoothstep(0.02, 0.08, min_dist);
    // Fade out on very steep (cliff overhangs) and very flat terrain.
    // Cracks show best on moderate slopes (0.3–0.7).
    let slope_fade = smoothstep(0.15, 0.35, slope) * (1.0 - smoothstep(0.7, 0.9, slope));
    return crack * slope_fade;
}

// Snow subsurface scattering approximation. Snow is a participating medium;
// light entering one face exits another with a warm, soft glow. We fake it
// with a wrap-lighting diffuse plus a forward-scatter halo when the sun is
// behind the surface. Returns an additive warm radiance term.
fn snow_sss(n: vec3<f32>, l: vec3<f32>, v: vec3<f32>, w_snow: f32) -> vec3<f32> {
    if (w_snow < 0.01) {
        return vec3<f32>(0.0);
    }
    // Wrap diffuse: soft, flat lighting with no hard terminator.
    let wrap = 0.5;
    let ndotl_wrap = (dot(n, l) + wrap) / (1.0 + wrap);
    let diffuse_wrap = max(ndotl_wrap, 0.0);
    // Forward scatter: bright halo when view aligns with light through snow.
    let back = -v;  // direction from viewer through the surface
    let scatter = pow(max(dot(back, l), 0.0), 3.0);
    // Warm tint from multiple scattering in the snow volume.
    let sss_color = vec3<f32>(1.0, 0.92, 0.82);
    return sss_color * (diffuse_wrap * 0.15 + scatter * 0.25) * w_snow;
}

// Analytic gully erosion: overlays multi-octave gully patterns aligned to
// the steepest-descent direction. Pure function of world position + slope
// gradient — no neighbour reads, no chunk dependencies, zero seams.
// Inspired by korbindeman/bevy_erosion_filter. Returns a scalar in
// roughly [-1, 1] that modulates albedo (darker in gullies) and can
// perturb the normal for fake carved detail.
fn gully_erode(p: vec3<f32>, slope_grad: vec2<f32>, slope_mag: f32) -> f32 {
    // Steepest-descent direction (2D in xz plane). Guard against flat.
    var dir = vec2<f32>(0.0);
    let m = slope_mag;
    if (m > 1e-4) {
        dir = -slope_grad / m;
    }
    // Project position onto descent direction and perpendicular.
    // Gullies form as ridges perpendicular to the descent direction.
    let perp = vec2<f32>(-dir.y, dir.x);
    let pz = dot(p.xz, dir);
    let pp = dot(p.xz, perp);
    // Multi-octave gully pattern. Each octave's strength depends on the
    // slope magnitude (no gullies on flat terrain) and the previous octave
    // (branching gullies). Fade in only in the slope band.
    let slope_band = smoothstep(0.15, 0.6, m);
    var v = 0.0;
    var a = 0.5;
    var freq = 1.0;
    var prev = 0.0;
    for (var i=0; i<4; i++) {
        let n = value_noise3(vec3<f32>(pp * freq + 17.3, pz * freq * 0.3, 0.0));
        let g = (n - 0.5) * 2.0;
        // Branching: higher octaves concentrate where lower octaves are dark.
        let branch = 1.0 - abs(prev) * 0.5;
        v += a * g * branch;
        prev = g;
        freq *= 2.0;
        a *= 0.5;
    }
    return v * slope_band;
}

// --- Triplanar mapping for seam-free material details on any surface ---
// Returns vec3 of weights for x, y, z axis projections based on the surface
// normal. Dominant axis gets most weight; blending zone uses smoothstep.
fn triplanar_weights(n: vec3<f32>) -> vec3<f32> {
    let an = abs(n);
    var w = vec3<f32>(0.0);
    if (an.x >= an.y && an.x >= an.z) { w = vec3<f32>(1.0, 0.0, 0.0); }
    else if (an.y >= an.x && an.y >= an.z) { w = vec3<f32>(0.0, 1.0, 0.0); }
    else { w = vec3<f32>(0.0, 0.0, 1.0); }
    let soft = mix(an * an * an, w, 0.5);
    let s = soft.x + soft.y + soft.z + 1e-6;
    return soft / s;
}

// Triplanar FBM: sample 3D noise on each axis plane and blend by normal.
fn triplanar_fbm(p: vec3<f32>, n: vec3<f32>, scale: f32, octaves: i32) -> f32 {
    let w = triplanar_weights(n);
    let px = fbm3(vec3<f32>(p.yz * scale, 0.0), octaves);
    let py = fbm3(vec3<f32>(p.xz * scale, 0.0), octaves);
    let pz = fbm3(vec3<f32>(p.xy * scale, 0.0), octaves);
    return w.x * px + w.y * py + w.z * pz;
}

// Triplanar sand ripples: project wind-aligned ripples on each axis.
fn triplanar_sand_ripples(p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let w = triplanar_weights(n);
    // Y-projection (flat ground) uses the original sand_ripples
    let ry = sand_ripples(p);
    // X/Z projections: ripples on vertical faces, lower amplitude
    let rx = sand_ripples(vec3<f32>(p.y, 0.0, p.z));
    let rz = sand_ripples(vec3<f32>(p.x, 0.0, p.y));
    let primary = w.x * rx.x + w.y * ry.x + w.z * rz.x;
    let cross_h = w.x * rx.y + w.y * ry.y + w.z * rz.y;
    return vec2<f32>(primary, cross_h);
}

// Triplanar rock strata: sedimentary banding follows each axis projection.
fn triplanar_rock_strata(p: vec3<f32>, n: vec3<f32>, slope: f32) -> vec2<f32> {
    let w = triplanar_weights(n);
    let sx = rock_strata(vec3<f32>(p.y, p.z, p.x), slope);
    let sy = rock_strata(p, slope);
    let sz = rock_strata(vec3<f32>(p.x, p.y, p.z), slope);
    return vec2<f32>(
        w.x * sx.x + w.y * sy.x + w.z * sz.x,
        w.x * sx.y + w.y * sy.y + w.z * sz.y,
    );
}

// Triplanar rock cracks: Voronoi cracks projected on each axis.
fn triplanar_rock_crack(p: vec3<f32>, n: vec3<f32>, slope: f32) -> f32 {
    let w = triplanar_weights(n);
    let cx = rock_crack(vec3<f32>(p.y, p.z, p.x), slope);
    let cy = rock_crack(p, slope);
    let cz = rock_crack(vec3<f32>(p.x, p.y, p.z), slope);
    return w.x * cx + w.y * cy + w.z * cz;
}

// --- Geological texture: weathering stains ---
// Dark streaks running downhill from rock edges. Simulates water staining
// and mineral leaching. Uses slope direction + height-based fade.
fn weathering_stains(p: vec3<f32>, n: vec3<f32>, slope: f32) -> f32 {
    if (slope < 0.2) { return 0.0; }
    // Streak pattern: high-freq noise modulated by downhill flow
    let flow_dir = vec2<f32>(-n.x, -n.z);
    let fd = normalize(flow_dir + vec2<f32>(0.001));
    let proj = dot(p.xz, fd);
    // Streaks are thin dark bands perpendicular to flow, spaced ~3-8 units
    let streak = sin(proj * 1.5 + fbm3(p * 0.1, 2) * 4.0);
    let streak_mask = smoothstep(0.6, 0.9, abs(streak));
    // Fade with height: more staining lower down (water collects below)
    let h_fade = 1.0 - smoothstep(0.0, 30.0, p.y);
    // Noise variation: not all surfaces stained equally
    let variation = fbm3(p * 0.05, 3);
    return streak_mask * h_fade * smoothstep(0.3, 0.7, variation) * 0.3;
}

// --- Geological texture: moss/lichen coverage ---
// Green tinting on shaded, moist surfaces. Favors north-facing slopes,
// crevices (high curvature), and low altitudes. Fades on very steep rock.
fn moss_coverage(p: vec3<f32>, n: vec3<f32>, curvature: f32, h: f32) -> f32 {
    // North-facing bias: surfaces facing away from sun (negative x in our
    // world) get more moss. Use normal.x as a simple aspect proxy.
    let aspect = smoothstep(0.3, -0.3, n.x);
    // Crevices: concave curvature traps moisture
    let crevice = smoothstep(0.1, 0.5, curvature);
    // Low altitude: more moisture at lower elevations
    let alt_fade = 1.0 - smoothstep(5.0, 40.0, h);
    // Noise patchiness: moss grows in patches, not uniformly
    let moss_patch = fbm3(p * 0.08, 3);
    let patchy = smoothstep(0.35, 0.65, moss_patch);
    // Gentle slopes only: moss can't cling to sheer cliffs
    let slope_ok = 1.0 - smoothstep(0.5, 0.8, 1.0 - n.y);
    return aspect * 0.3 + crevice * 0.4 + alt_fade * 0.2 * patchy * slope_ok;
}

// Voronoi vein helper for mineral_veins (veins at cell centers, not edges).
fn vein_axis(sp: vec2<f32>) -> f32 {
    let scale = 4.0;
    let cell = floor(sp * scale);
    let fcell = fract(sp * scale);
    var min_d = 1.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let neighbor = vec2<f32>(f32(i), f32(j));
            let center = hash2(cell + neighbor) * 0.8 + 0.1;
            let d = length(neighbor + center - fcell);
            min_d = min(min_d, d);
        }
    }
    return 1.0 - smoothstep(0.0, 0.06, min_d);
}

// --- Geological texture: mineral veins ---
// Thin lighter-colored lines through rock, like quartz veins. Uses Voronoi
// cell centers at medium frequency. Only on rock surfaces.
fn mineral_veins(p: vec3<f32>, n: vec3<f32>, slope: f32) -> f32 {
    let w = triplanar_weights(n);
    let vx = vein_axis(p.yz);
    let vy = vein_axis(p.xz);
    let vz = vein_axis(p.xy);
    return (w.x * vx + w.y * vy + w.z * vz) * smoothstep(0.3, 0.7, slope);
}

// --- Biome edge softening: widen transitions to prevent sharp lines ---
// When biome weights are near-equal (transition zone), add extra noise-based
// blending to break up hard biome boundaries. Returns a blend factor.
fn biome_edge_blend(p: vec3<f32>, wb: vec4<f32>) -> f32 {
    // Find how close the two largest biome weights are — near-equal = edge
    let sorted = sort4(wb);
    let edge_proximity = 1.0 - smoothstep(0.0, 0.15, sorted.x - sorted.y);
    // Noise breaks up the edge into irregular patches
    let noise = fbm3(p * 0.03, 3);
    return edge_proximity * (0.5 + 0.5 * noise);
}

fn sort4(v: vec4<f32>) -> vec4<f32> {
    var a = v;
    if (a.x < a.y) { let t = a.x; a.x = a.y; a.y = t; }
    if (a.z < a.w) { let t = a.z; a.z = a.w; a.w = t; }
    if (a.x < a.z) { let t = a.x; a.x = a.z; a.z = t; }
    if (a.y < a.w) { let t = a.y; a.y = a.w; a.w = t; }
    if (a.y < a.z) { let t = a.y; a.y = a.z; a.z = t; }
    return a;
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.pos = u.mvp * vec4<f32>(in.pos, 1.0);
    out.world_pos = in.pos;
    out.normal = in.normal;
    out.color = vec3<f32>(0.0);
    out.biome = in.biome;
    out.sc = in.sc;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let base_normal = normalize(in.normal);

    // --- Two-octave procedural normal perturbation for micro-roughness ---
    // High-freq octave (scale 1.5): fine rock detail. Low-freq octave
    // (scale 6.0): broad undulations. Both fade with distance so far
    // terrain uses the per-vertex normal. The high-freq octave is
    // slope-weighted: more perturbation on rock, less on flat snow/sand.
    let micro_dist = length(in.world_pos - u.camera_pos);
    let micro_fade = clamp(1.0 - (micro_dist - 40.0) / 120.0, 0.0, 1.0);
    let slope_pre = 1.0 - base_normal.y;

    let detail_p = in.world_pos * 1.5;
    let eps = 0.4;
    let h0 = fbm3(detail_p, 3);
    let hx = fbm3(detail_p + vec3<f32>(eps, 0.0, 0.0), 3) - h0;
    let hy = fbm3(detail_p + vec3<f32>(0.0, eps, 0.0), 3) - h0;
    let hz = fbm3(detail_p + vec3<f32>(0.0, 0.0, eps), 3) - h0;
    let micro_hi = vec3<f32>(-hx, -hy, -hz) * (0.4 + 0.6 * smoothstep(0.2, 0.7, slope_pre));

    let detail_p2 = in.world_pos * 6.0;
    let eps2 = 0.15;
    let h0b = fbm3(detail_p2, 2);
    let hxb = fbm3(detail_p2 + vec3<f32>(eps2, 0.0, 0.0), 2) - h0b;
    let hyb = fbm3(detail_p2 + vec3<f32>(0.0, eps2, 0.0), 2) - h0b;
    let hzb = fbm3(detail_p2 + vec3<f32>(0.0, 0.0, eps2), 2) - h0b;
    let micro_lo = vec3<f32>(-hxb, -hyb, -hzb);

    let n = normalize(base_normal + (micro_hi * 0.10 + micro_lo * 0.05) * micro_fade);

    let view_dir = normalize(u.camera_pos - in.world_pos);
    let sun_dir = normalize(u.light_dir);
    let sun_color = u.sun_color;
    let ambient = vec3<f32>(0.12, 0.15, 0.20);

    // Directional shadow.
    let light_clip = u.light_mvp * vec4<f32>(in.world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    var shadow = 1.0;
    if (all(abs(light_ndc.xy) <= vec2<f32>(1.0))) {
        let shadow_uv = vec2<f32>(light_ndc.x * 0.5 + 0.5, -light_ndc.y * 0.5 + 0.5);
        let current_depth = light_ndc.z;
        let bias = 0.005;
        shadow = textureSampleCompare(t_shadow, s_shadow, shadow_uv, current_depth - bias);
    }

    // Large-scale material variation.
    let macro_noise = fbm3(in.world_pos * 0.2, 3);

    // Material weights based on altitude, slope, noise, biome, curvature.
    let slope = 1.0 - base_normal.y;

    // Analytic gully erosion: pure function of world position + slope
    // gradient. No neighbour reads → zero chunk seams. Darkens albedo
    // in gullies and perturbs the normal for fake carved detail.
    let slope_grad = vec2<f32>(-base_normal.x, -base_normal.z);
    let slope_mag = length(slope_grad);
    let gully = gully_erode(in.world_pos, slope_grad, slope_mag);
    // Perturb normal along the gully direction for fake carved valleys.
    let gully_n = normalize(n + vec3<f32>(slope_grad.x, 0.0, slope_grad.y) * gully * 0.05 * micro_fade);
    let h = in.world_pos.y;

    // Sediment + curvature from vertex attribute.
    let sediment = in.sc.x;
    let curvature = in.sc.y;  // >0 concave (valley), <0 convex (ridge)

    // Biome weights: x=tundra, y=mountain, z=desert, w=forest.
    let wb = in.biome;
    let w_tundra = wb.x;
    let w_mountain = wb.y;
    let w_desert = wb.z;
    let w_forest = wb.w;

    // --- Extended biome weights (beach, savanna, swamp, volcanic) ---
    // Computed in-shader from world position + altitude + climate + noise so
    // no vertex format change is needed. Each weight carves out of the climate
    // biomes, keeping the total weight sum ~1.

    // Volcanic: low-frequency noise mask. Dark basalt + craters in patches.
    let volcanic_noise = fbm3(in.world_pos * 0.012 + vec3<f32>(500.0, 0.0, 500.0), 4);
    let w_volcanic = smoothstep(0.55, 0.75, volcanic_noise);

    // Swamp: low altitude + humid (forest climate proxy) + flat terrain.
    let w_swamp = smoothstep(2.0, -2.0, h) * w_forest * (1.0 - smoothstep(0.3, 0.6, slope));

    // Beach: very low altitude near water. Sand material regardless of climate.
    let w_beach = smoothstep(0.0, -3.0, h) * (1.0 - w_volcanic);

    // Savanna: hot + dry (desert climate proxy) + flat + mid altitude.
    let savanna_noise = fbm3(in.world_pos * 0.04 + vec3<f32>(300.0, 0.0, 300.0), 3);
    let w_savanna = w_desert * (1.0 - smoothstep(0.4, 0.7, slope)) * smoothstep(0.2, 0.6, savanna_noise);

    // Per-biome base palettes.
    let sand_color        = vec3<f32>(0.76, 0.70, 0.50);
    let beach_sand        = vec3<f32>(0.82, 0.74, 0.52);
    let wet_sand          = vec3<f32>(0.55, 0.48, 0.35);
    let snow_color        = vec3<f32>(0.92, 0.94, 0.97);
    let rock_color        = vec3<f32>(0.30, 0.28, 0.26);
    let dark_rock_color   = vec3<f32>(0.22, 0.20, 0.19);
    let basalt_color      = vec3<f32>(0.14, 0.13, 0.13);
    let lava_color        = vec3<f32>(0.85, 0.32, 0.08);
    let grass_color       = vec3<f32>(0.16, 0.42, 0.12);
    let savanna_grass     = vec3<f32>(0.58, 0.52, 0.24);
    let forest_grass      = vec3<f32>(0.10, 0.30, 0.08);
    let tundra_grass      = vec3<f32>(0.42, 0.46, 0.36);
    let swamp_mud         = vec3<f32>(0.28, 0.26, 0.18);
    let dirt_color        = vec3<f32>(0.42, 0.32, 0.20);

    // Snow line: tundra is flat and cold (snow at low alt), mountains need
    // real peaks (snow only near the top), desert/forest are temperate.
    let snow_line = 10.0 + 22.0 * w_mountain - 6.0 * w_tundra;
    // Snow accumulation mask: snow only sticks on gentle slopes AND in
    // wind-sheltered areas (low-freq noise mask). Prevents snow on cliffs.
    let wind_shelter = smoothstep(0.0, 0.4, fbm3(in.world_pos * 0.05, 2));
    let snow_slope = smoothstep(0.5, 0.25, slope);
    let w_snow = smoothstep(snow_line, snow_line + 8.0, h + macro_noise * 5.0) * snow_slope * wind_shelter;

    // Per-biome base color before slope/rock override.
    var biome_albedo = vec3<f32>(0.0);
    biome_albedo += w_tundra  * mix(tundra_grass, snow_color, w_snow);
    biome_albedo += w_mountain * mix(rock_color, snow_color, w_snow);
    biome_albedo += w_desert  * mix(sand_color, rock_color, smoothstep(0.10, 0.45, slope));
    biome_albedo += w_forest  * mix(forest_grass, grass_color, macro_noise * 0.5 + 0.5);
    // Low-altitude dirt blending for forest/grass transition zones.
    biome_albedo = mix(biome_albedo, mix(biome_albedo, dirt_color, 0.5), smoothstep(-2.0, 4.0, h) * (w_forest + w_tundra) * 0.3);

    // Extended biome blending: carve new biomes out of the climate base.
    // Savanna: blend desert portion toward dry grass.
    biome_albedo = mix(biome_albedo, savanna_grass, w_savanna);
    // Swamp: muddy lowlands.
    biome_albedo = mix(biome_albedo, swamp_mud, w_swamp);
    // Beach: wet/dry sand near water.
    let beach_wet = smoothstep(-1.0, 1.0, h);
    biome_albedo = mix(biome_albedo, mix(wet_sand, beach_sand, beach_wet), w_beach);
    // Volcanic: basalt rock + lava in deep channels.
    let lava_mask = smoothstep(0.6, 0.8, fbm3(in.world_pos * 0.1, 3)) * smoothstep(0.4, 0.7, slope);
    let volcanic_rock = mix(basalt_color, dark_rock_color, macro_noise * 0.5 + 0.5);
    biome_albedo = mix(biome_albedo, mix(volcanic_rock, lava_color, lava_mask * 0.3), w_volcanic);

    // Steep slopes AND convex ridges (negative curvature) become rock.
    let w_rock_slope = smoothstep(0.55, 0.85, slope);
    let w_rock_ridge = smoothstep(0.3, 0.8, -curvature);
    let w_rock = max(w_rock_slope, w_rock_ridge * 0.6);
    var albedo = mix(biome_albedo, mix(rock_color, dark_rock_color, macro_noise * 0.5 + 0.5), w_rock);
    // Snow caps beat everything on high, flat-to-moderate terrain.
    albedo = mix(albedo, snow_color, w_snow);
    // Sediment deposition in valleys -> fertile dirt tint.
    albedo = mix(albedo, mix(albedo, dirt_color, 0.5), sediment * 0.4);
    // Concave curvature (valleys, gullies) -> subtle dirt accumulation.
    albedo = mix(albedo, mix(albedo, dirt_color, 0.3), smoothstep(0.2, 0.6, curvature) * 0.5);

    // Subtle hue/noise variation.
    albedo = albedo * (0.9 + 0.2 * macro_noise);
    // Gully erosion darkening: gullies (negative gully value) collect
    // dirt and appear darker; ridges between gullies stay bright.
    albedo = albedo * (1.0 + gully * 0.25);

    // --- Material detail: triplanar sand ripples (desert + beach) ---
    // Uses triplanar projection so ripples appear correctly on any surface
    // orientation, not just flat ground. Fades on steep slopes.
    let ripples = triplanar_sand_ripples(in.world_pos, base_normal);
    let ripple_primary = ripples.x;
    let ripple_cross = ripples.y;
    let ripple_w = (w_desert + w_beach) * (1.0 - smoothstep(0.15, 0.55, slope));
    albedo = mix(albedo, albedo * (0.85 + 0.30 * (ripple_primary * 0.5 + 0.5)), ripple_w);
    let ripple_n = normalize(gully_n + vec3<f32>(0.7, 0.0, 0.7) * ripple_primary * 0.04 * ripple_w * micro_fade
                                       + vec3<f32>(0.5, 0.0, 0.5) * ripple_cross * 0.03 * ripple_w * micro_fade);
    let shade_n = ripple_n;

    // --- Material detail: fine sand grain sparkle (triplanar) ---
    let grain_w = (w_desert + w_beach) * (1.0 - smoothstep(0.15, 0.55, slope));
    let grain_n = triplanar_fbm(in.world_pos, base_normal, 15.0, 2);
    albedo = mix(albedo, albedo * (0.9 + 0.2 * grain_n), grain_w);

    // --- Material detail: triplanar sedimentary rock strata (cliffs) ---
    let strata_res = triplanar_rock_strata(in.world_pos, base_normal, slope);
    let strata = strata_res.x;
    let strata_rough = strata_res.y;
    let strata_w = w_rock * smoothstep(0.35, 0.85, slope);
    let strata_tint = mix(vec3<f32>(0.18, 0.16, 0.14), vec3<f32>(0.42, 0.36, 0.28), strata);
    albedo = mix(albedo, strata_tint, strata_w * strata);

    // --- Material detail: triplanar procedural rock cracks ---
    albedo = mix(albedo, vec3<f32>(0.08, 0.07, 0.06), triplanar_rock_crack(in.world_pos, base_normal, slope) * w_rock * 0.7);

    // --- Geological texture: weathering stains on rock ---
    let stain = weathering_stains(in.world_pos, base_normal, slope);
    albedo = mix(albedo, albedo * (1.0 - stain), w_rock * 0.8);

    // --- Geological texture: moss/lichen coverage ---
    let moss = moss_coverage(in.world_pos, base_normal, curvature, h);
    let moss_color = vec3<f32>(0.12, 0.22, 0.08);
    albedo = mix(albedo, mix(albedo, moss_color, 0.4), moss * (w_forest + w_tundra) * 0.5);

    // --- Geological texture: mineral veins (quartz in rock) ---
    let veins = mineral_veins(in.world_pos, base_normal, slope);
    let vein_color = vec3<f32>(0.65, 0.62, 0.55);
    albedo = mix(albedo, mix(albedo, vein_color, 0.5), veins * w_rock * 0.6);

    // --- Biome edge softening: break up sharp biome boundaries ---
    let edge_blend = biome_edge_blend(in.world_pos, wb);
    albedo = mix(albedo, albedo * (0.85 + 0.3 * fbm3(in.world_pos * 0.1, 2)), edge_blend * 0.3);

    // --- PBR parameters (roughness, metallic, AO) per material ---
    var roughness = 0.85;
    var metallic = 0.0;
    // Snow is the smoothest surface (polished ice crystals).
    roughness = mix(roughness, 0.35, w_snow);
    // Wet beach sand near water is slightly smoother.
    roughness = mix(roughness, 0.65, w_beach * (1.0 - beach_wet));
    // Volcanic basalt is rough; lava is metallic and glossy.
    roughness = mix(roughness, 0.95, w_volcanic * (1.0 - lava_mask));
    metallic = mix(metallic, 0.6, w_volcanic * lava_mask);
    // Savanna grass is slightly rougher than forest.
    roughness = mix(roughness, 0.90, w_savanna);
    // Rock on steep slopes is rougher than dirt.
    roughness = mix(roughness, 1.0, w_rock * 0.5);
    // Sedimentary strata roughness variation: harder bands smoother,
    // softer bands rougher. strata_rough is in [-0.15, +0.15].
    roughness = mix(roughness, clamp(roughness + strata_rough, 0.05, 1.0), strata_w);
    // Sand ripples slightly reduce perceived roughness.
    roughness = mix(roughness, roughness * 0.9, ripple_w * 0.5);
    roughness = clamp(roughness, 0.05, 1.0);

    // Ambient occlusion from curvature: concave (positive curvature) darker.
    let ao_curve = mix(0.85, 1.0, smoothstep(-0.3, 0.3, curvature));
    let ao = min(ao_curve, mix(1.0, 0.75, sediment * 0.5));

    // --- PBR direct lighting (Cook-Torrance BRDF) ---
    let pn = shade_n;
    let pl = sun_dir;
    let pv = view_dir;
    let pnl = max(dot(pn, pl), 0.0);
    let pnv = max(dot(pn, pv), 1e-4);
    let ph_vec = normalize(pl + pv);
    let pnh = max(dot(pn, ph_vec), 0.0);
    let phv = max(dot(ph_vec, pv), 0.0);
    // GGX roughness → alpha. Square roughness for perceptually-linear control.
    let a = roughness * roughness;
    let f0_base = mix(0.04, 0.06, w_volcanic * (1.0 - lava_mask));
    let f0 = mix(f0_base, 0.55, metallic);
    // Specular term: D * G * F / (4 * nv * nl).
    let d_term = d_ggx(pnh, a);
    let g_term = g_smith(pnv, pnl, a);
    let f_term = f_schlick(phv, f0);
    let spec_brdf = (d_term * g_term * f_term) / max(4.0 * pnv * pnl, 1e-6);
    // Diffuse term: Lambert, scaled by (1 - fresnel) for energy conservation.
    // Metallic surfaces have no diffuse.
    let kd = (1.0 - f_term) * (1.0 - metallic);
    // Specular is tinted by f0 for metals, white for dielectrics.
    let spec_color = mix(vec3<f32>(1.0), albedo, metallic);
    // Direct sun + sky ambient, both modulated by AO and shadow.
    let direct = (kd * albedo / 3.14159265 + spec_brdf * spec_color) * sun_color * pnl * shadow;
    let ambient_lit = albedo * ambient * ao;
    var lit = direct + ambient_lit;

    // --- Snow subsurface scattering (additive warm glow) ---
    lit += snow_sss(shade_n, sun_dir, view_dir, w_snow) * shadow;

    // Distance fog / atmospheric haze with wavelength-dependent scattering.
    // Blue scatters more than red, so distant terrain shifts toward blue
    // and loses warm tones — mimicking real aerial perspective.
    let dist = length(in.world_pos - u.camera_pos);
    let fog_t = u.fog_density * max(dist - u.fog_start, 0.0);
    let att = vec3<f32>(exp(-fog_t * 1.0), exp(-fog_t * 1.5), exp(-fog_t * 4.0));
    // Sun-scatter halo: distant terrain picks up warm light scattering
    // toward the camera along the view-sun plane.
    let sun_scatter = pow(max(dot(view_dir, sun_dir), 0.0), 8.0) * 0.15;
    let lit_att = lit * att + sun_color * sun_scatter * (1.0 - att.r);
    let out_color = mix(u.fog_color, lit_att, att.r);
    return vec4<f32>(out_color, 1.0);
}
"""

BBOX_SHADER = """
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

struct VertexIn {
    @location(0) pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(in.pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
"""
