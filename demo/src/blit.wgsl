// Halves a texture into its next mipmap level.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: Out;
    out.clip = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

@fragment
fn fs_blit(in: Out) -> @location(0) vec4<f32> {
    return textureSample(source, src_sampler, in.uv);
}
