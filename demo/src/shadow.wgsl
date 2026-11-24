// The shadow map: the model's depth seen from the key light.

// The start of the scene's frame uniform (shader.wgsl), as far as the shadow pass reads it.
struct Frame {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    ground: vec4<f32>,
    center: vec4<f32>,
    viewport: vec4<f32>,
    params: vec4<f32>,
    point: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> bones: array<mat4x4<f32>, 256>;

@vertex
fn vs_shadow(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return frame.light_view_proj * vec4<f32>(position, 1.0);
}

@vertex
fn vs_shadow_skinned(
    @location(0) position: vec3<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    let total = weights.x + weights.y + weights.z + weights.w;
    var p = (bones[joints.x] * weights.x + bones[joints.y] * weights.y + bones[joints.z] * weights.z
        + bones[joints.w] * weights.w) * vec4<f32>(position, 1.0);
    if total < 1e-4 {
        p = vec4<f32>(position, 1.0);
    }
    return frame.light_view_proj * vec4<f32>(p.xyz, 1.0);
}

// A point's square, as wide as it's drawn, facing the light (whose view is orthographic, 1.6 radii across).
@vertex
fn vs_shadow_point(
    @builtin(vertex_index) i: u32,
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let clip = frame.light_view_proj * vec4<f32>(position, 1.0);
    let half = frame.point.x * 0.5 / (frame.ground.y * 1.6);
    return vec4<f32>(clip.xy + corners[i] * half, clip.zw);
}
