// The scene: a gradient backdrop, the model lit by a key light, a sky and a warm floor bounce,
// a grid floor that takes its shadow, and points and lines.

struct Frame {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // Toward the key light; w its strength.
    light_dir: vec4<f32>,
    // Floor height, the model's radius, grid spacing, 1 when the floor is drawn.
    ground: vec4<f32>,
    // The model's center.
    center: vec4<f32>,
    // Width, height, 1 / width, 1 / height.
    viewport: vec4<f32>,
    // Exposure, shadow strength, time, a shadow texel.
    params: vec4<f32>,
    // A point's size in the world, the focal length in pixels, its least and most pixels.
    point: vec4<f32>,
};

struct Material {
    color: vec4<f32>,
    // w: 1 when the color comes from a texture.
    emissive: vec4<f32>,
    // Metallic, roughness, the alpha cutoff (below 0 for none), 1 for unlit.
    params: vec4<f32>,
    extra: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var color_texture: texture_2d<f32>;
@group(1) @binding(2) var color_sampler: sampler;

@group(2) @binding(0) var<uniform> bones: array<mat4x4<f32>, 256>;

const PI: f32 = 3.14159265;
const SKY: vec3<f32> = vec3<f32>(0.40, 0.45, 0.56);
const BOUNCE: vec3<f32> = vec3<f32>(0.20, 0.13, 0.09);
const KEY: vec3<f32> = vec3<f32>(1.0, 0.95, 0.88);
const EMBER: vec3<f32> = vec3<f32>(1.0, 0.42, 0.12);

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

// ACES, fitted (Narkowicz).
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let x = c * frame.params.x;
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// How lit a point is by the key light, 0 in shadow to 1, filtered over 3 by 3 texels.
fn lit(world: vec3<f32>, n: vec3<f32>) -> f32 {
    let p = frame.light_view_proj * vec4<f32>(world + n * frame.ground.y * 0.004, 1.0);
    let ndc = p.xyz / p.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let depth = ndc.z - 0.0015;
    let texel = frame.params.w;
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(f32(x), f32(y)) * texel, depth);
        }
    }
    let inside = all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0)) && ndc.z < 1.0;
    return select(1.0, sum / 9.0, inside);
}

// The key light (GGX, Schlick, Smith), the sky and floor around, and a warm rim.
fn shade(base: vec3<f32>, metallic: f32, roughness: f32, n: vec3<f32>, v: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let l = normalize(frame.light_dir.xyz);
    let h = normalize(l + v);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let r = clamp(roughness, 0.04, 1.0);
    let a2 = r * r * r * r;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd);
    let k = (r + 1.0) * (r + 1.0) / 8.0;
    let g = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vh, 5.0);
    let specular = d * g * f / max(4.0 * nv * nl, 1e-4);
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * base / PI;
    let shadow = mix(1.0, lit(world, n), frame.params.y);
    let direct = (diffuse + specular) * KEY * frame.light_dir.w * nl * shadow;

    let up = n.y * 0.5 + 0.5;
    let ambient = mix(BOUNCE, SKY, up) * base * (1.0 - metallic);
    let rv = reflect(-v, n);
    let env = mix(mix(BOUNCE * 0.8, SKY * 1.3, smoothstep(-0.2, 0.4, rv.y)), mix(BOUNCE, SKY, up), r);
    let fe = f0 + (max(vec3<f32>(1.0 - r), f0) - f0) * pow(1.0 - nv, 5.0);
    let reflection = env * fe * (1.0 - 0.5 * r);
    let rim = EMBER * pow(1.0 - nv, 4.0) * 0.18 * (0.4 + 0.6 * up);
    return direct + ambient + reflection + rim;
}

// The backdrop: a cool gradient with a faint ember glow low in the frame.

struct Backdrop {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_backdrop(@builtin(vertex_index) i: u32) -> Backdrop {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var out: Backdrop;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

@fragment
fn fs_backdrop(in: Backdrop) -> @location(0) vec4<f32> {
    let far = frame.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let near = frame.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);
    let top = vec3<f32>(0.010, 0.012, 0.020);
    let horizon = vec3<f32>(0.040, 0.036, 0.044);
    let below = vec3<f32>(0.018, 0.015, 0.016);
    var c = mix(horizon, top, smoothstep(0.0, 0.7, dir.y));
    c = mix(c, below, smoothstep(0.0, -0.5, dir.y));
    let aspect = frame.viewport.x * frame.viewport.w;
    let q = vec2<f32>(in.ndc.x * aspect, in.ndc.y + 0.55);
    c += EMBER * 0.035 * exp(-dot(q, q) * 1.6);
    let vignette = 1.0 - 0.35 * dot(in.ndc * 0.7, in.ndc * 0.7);
    return vec4<f32>(c * vignette, 1.0);
}

// The floor: a fading grid at the model's feet that takes its shadow.

struct Floor {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn vs_floor(@builtin(vertex_index) i: u32) -> Floor {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[i];
    let extent = frame.ground.y * 6.0;
    let world = vec3<f32>(frame.center.x + c.x * extent, frame.ground.x, frame.center.z + c.y * extent);
    var out: Floor;
    out.world = world;
    out.clip = frame.view_proj * vec4<f32>(world, 1.0);
    return out;
}

fn grid(p: vec2<f32>, spacing: f32) -> f32 {
    let coord = p / spacing;
    let w = fwidth(coord);
    let g = abs(fract(coord - 0.5) - 0.5) / max(w, vec2<f32>(1e-6));
    // Lines thinner than a pixel fade rather than shimmer.
    let fade = 1.0 - smoothstep(0.25, 0.6, max(w.x, w.y));
    return (1.0 - min(min(g.x, g.y), 1.0)) * fade;
}

@fragment
fn fs_floor(in: Floor) -> @location(0) vec4<f32> {
    let p = in.world.xz - frame.center.xz;
    let radius = frame.ground.y;
    let spacing = frame.ground.z;
    let minor = grid(p, spacing);
    let major = grid(p, spacing * 5.0);
    let d = length(p) / radius;
    let shadow = mix(1.0, lit(in.world, vec3<f32>(0.0, 1.0, 0.0)), frame.params.y);
    let fade = 1.0 - smoothstep(1.2, 5.5, d);
    var c = vec3<f32>(0.020, 0.020, 0.024);
    c = mix(c, vec3<f32>(0.045, 0.043, 0.05), minor * 0.4);
    c = mix(c, vec3<f32>(0.085, 0.075, 0.08), major * 0.55);
    c += EMBER * 0.05 * exp(-d * d * 1.5);
    c *= mix(0.25, 1.0, shadow);
    let alpha = fade * mix(0.92, 0.6, smoothstep(0.0, 4.0, d)) * frame.ground.w;
    return vec4<f32>(tonemap(c * 2.0), alpha);
}

// Meshes.

struct Surface {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

fn surface(world: vec3<f32>, normal: vec3<f32>, uv: vec2<f32>, color: vec4<f32>) -> Surface {
    var out: Surface;
    out.world = world;
    out.normal = normal;
    out.uv = uv;
    out.color = color;
    out.clip = frame.view_proj * vec4<f32>(world, 1.0);
    return out;
}

@vertex
fn vs_mesh(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> Surface {
    return surface(position, normal, uv, color);
}

@vertex
fn vs_skinned(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
) -> Surface {
    let total = weights.x + weights.y + weights.z + weights.w;
    var m = bones[joints.x] * weights.x + bones[joints.y] * weights.y + bones[joints.z] * weights.z
        + bones[joints.w] * weights.w;
    // A vertex no joint moves stays where it is.
    if total < 1e-4 {
        m = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
    let world = (m * vec4<f32>(position, 1.0)).xyz;
    let n = (m * vec4<f32>(normal, 0.0)).xyz;
    return surface(world, n, uv, color);
}

fn shade_surface(in: Surface, front: bool) -> vec4<f32> {
    let texel = textureSample(color_texture, color_sampler, in.uv);
    let dx = dpdx(in.world);
    let dy = dpdy(in.world);
    let base = material.color * in.color * texel;
    if material.params.z >= 0.0 && base.a < material.params.z {
        discard;
    }
    let v = normalize(frame.camera_pos.xyz - in.world);
    var facet = normalize(cross(dx, dy));
    facet = select(facet, -facet, dot(facet, v) < 0.0);
    let smooth_n = select(-in.normal, in.normal, front);
    let has_normal = dot(in.normal, in.normal) > 0.01;
    let n = select(facet, normalize(smooth_n), has_normal);
    var c: vec3<f32>;
    if material.params.w > 0.5 {
        c = base.rgb;
    } else {
        c = shade(base.rgb, material.params.x, material.params.y, n, v, in.world) + material.emissive.rgb;
    }
    // Opaque and cut-out surfaces cover what's behind them.
    let alpha = select(1.0, base.a, material.extra.x > 0.5);
    return vec4<f32>(tonemap(c), alpha);
}

@fragment
fn fs_mesh(in: Surface, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return shade_surface(in, front);
}

// Lines, and the skeleton drawn over everything.

struct Stroke {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_line(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> Stroke {
    var out: Stroke;
    out.clip = frame.view_proj * vec4<f32>(position, 1.0);
    out.color = vec4<f32>(srgb_to_linear(color.rgb), color.a);
    return out;
}

@fragment
fn fs_line(in: Stroke) -> @location(0) vec4<f32> {
    return vec4<f32>(tonemap(in.color.rgb * 1.6), in.color.a);
}

// Points: a disc facing the camera for each, sized in the world, shaded as a little sphere.

struct Dot {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) corner: vec2<f32>,
};

@vertex
fn vs_point(
    @builtin(vertex_index) i: u32,
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
) -> Dot {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[i];
    var clip = frame.view_proj * vec4<f32>(position, 1.0);
    let pixels = clamp(frame.point.x * frame.point.y / max(clip.w, 1e-6), frame.point.z, frame.point.w);
    clip = vec4<f32>(clip.xy + corner * pixels * frame.viewport.zw * clip.w, clip.zw);
    var out: Dot;
    out.clip = clip;
    out.color = srgb_to_linear(color.rgb);
    out.corner = corner;
    return out;
}

@fragment
fn fs_point(in: Dot) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    if r2 > 1.0 {
        discard;
    }
    let n = vec3<f32>(in.corner * vec2<f32>(1.0, -1.0), sqrt(1.0 - r2));
    let light = 0.55 + 0.45 * max(dot(n, normalize(vec3<f32>(-0.4, 0.5, 0.8))), 0.0);
    return vec4<f32>(tonemap(in.color * light * 1.4), 1.0);
}
