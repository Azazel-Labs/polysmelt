//! Draws a flattened model on a staged floor with wgpu: the same code on WebGPU, WebGL 2 and,
//! for the README's images, native GPUs.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use polysmelt::mesh::{Material, Mesh};
use wgpu::util::DeviceExt;

/// Multisampling of the scene.
pub const SAMPLES: u32 = 4;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
const SHADOW: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_SIZE: u32 = 2048;
/// The most points drawn: beyond, every nth.
pub const MAX_POINTS: usize = 3_000_000;
/// glTF's nearest filter.
const NEAREST: u32 = 9728;
/// The most bones [`Renderer::set_bones`] takes.
pub const MAX_BONES: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [f32; 16],
    light_view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    ground: [f32; 4],
    center: [f32; 4],
    viewport: [f32; 4],
    params: [f32; 4],
    point: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4],
    extra: [f32; 4],
}

/// A line or point vertex: a position and an sRGB color.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [u8; 4],
}

/// Where the scene is seen from.
#[derive(Debug, Clone, Copy)]
pub struct View {
    pub view_proj: Mat4,
    pub eye: Vec3,
    /// Radians around +Y the camera looks from: the key light comes from over its shoulder.
    pub yaw: f32,
    pub fov: f32,
}

/// What's staged around the model.
#[derive(Debug, Clone, Copy)]
pub struct Stage {
    /// The model's center and radius.
    pub center: Vec3,
    pub radius: f32,
    /// The floor's height (the bottom of the model).
    pub floor: f32,
    pub show_floor: bool,
    pub show_skeleton: bool,
    pub exposure: f32,
}

/// A triangle primitive's buffers.
struct Draw {
    /// Positions, normals, texture coordinates, colors; then joints and weights when skinned.
    buffers: Vec<wgpu::Buffer>,
    index: wgpu::Buffer,
    count: u32,
    vertices: usize,
    material: usize,
}

struct MaterialGpu {
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    image: Option<usize>,
    sampler: SamplerKey,
    blend: bool,
}

/// Wrap S, wrap T, nearest magnification.
type SamplerKey = (wgpu::AddressMode, wgpu::AddressMode, bool);

struct Lines {
    buffer: wgpu::Buffer,
    count: u32,
}

#[derive(Default)]
struct Scene {
    draws: Vec<Draw>,
    skinned: bool,
    lines: Option<Lines>,
    points: Option<Lines>,
    overlay: Option<Lines>,
    materials: Vec<MaterialGpu>,
    textures: Vec<Option<wgpu::TextureView>>,
    /// A point's diameter in the world.
    point_size: f32,
}

struct Pipelines {
    backdrop: wgpu::RenderPipeline,
    floor: wgpu::RenderPipeline,
    mesh: [wgpu::RenderPipeline; 2],
    skinned: [wgpu::RenderPipeline; 2],
    line: wgpu::RenderPipeline,
    overlay: wgpu::RenderPipeline,
    point: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    shadow_skinned: wgpu::RenderPipeline,
    shadow_point: wgpu::RenderPipeline,
    blit: wgpu::RenderPipeline,
}

/// The scene's GPU side.
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    size: (u32, u32),
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    shadow: wgpu::TextureView,
    frame: wgpu::Buffer,
    frame_group: wgpu::BindGroup,
    shadow_frame_group: wgpu::BindGroup,
    material_layout: wgpu::BindGroupLayout,
    blit_layout: wgpu::BindGroupLayout,
    bones: wgpu::Buffer,
    bones_group: wgpu::BindGroup,
    pipelines: Pipelines,
    samplers: HashMap<SamplerKey, wgpu::Sampler>,
    linear: wgpu::Sampler,
    white: wgpu::TextureView,
    scene: Scene,
    /// The largest texture side the device takes.
    pub max_texture: u32,
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

const fn attr(location: u32, format: wgpu::VertexFormat) -> [wgpu::VertexAttribute; 1] {
    [wgpu::VertexAttribute {
        format,
        offset: 0,
        shader_location: location,
    }]
}

const POSITION: [wgpu::VertexAttribute; 1] = attr(0, wgpu::VertexFormat::Float32x3);
const NORMAL: [wgpu::VertexAttribute; 1] = attr(1, wgpu::VertexFormat::Float32x3);
const UV: [wgpu::VertexAttribute; 1] = attr(2, wgpu::VertexFormat::Float32x2);
const COLOR: [wgpu::VertexAttribute; 1] = attr(3, wgpu::VertexFormat::Float32x4);
const JOINTS: [wgpu::VertexAttribute; 1] = attr(4, wgpu::VertexFormat::Uint32x4);
const WEIGHTS: [wgpu::VertexAttribute; 1] = attr(5, wgpu::VertexFormat::Float32x4);
const LINE: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4];

fn layout(stride: u64, attributes: &'static [wgpu::VertexAttribute]) -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: stride,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes,
    }
}

fn texture_size(w: u32, h: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    }
}

/// An sRGB byte color from a linear one.
fn srgb8(c: [f32; 4]) -> [u8; 4] {
    let e = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    };
    [e(c[0]), e(c[1]), e(c[2]), (c[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8]
}

fn address(wrap: u32) -> wgpu::AddressMode {
    match wrap {
        polysmelt::mesh::CLAMP => wgpu::AddressMode::ClampToEdge,
        polysmelt::mesh::MIRROR => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::Repeat,
    }
}

impl Renderer {
    /// A renderer drawing into `format` targets (an sRGB format) of `width` by `height`.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let both = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[
                uniform_entry(0, both),
                texture_entry(1, wgpu::TextureSampleType::Depth),
                sampler_entry(2, wgpu::SamplerBindingType::Comparison),
            ],
        });
        let shadow_frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow frame"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(2, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let bones_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bones"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            ],
        });

        let frame = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bones = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bones"),
            size: (MAX_BONES * 64) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("shadow"),
                size: texture_size(SHADOW_SIZE, SHADOW_SIZE),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SHADOW,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });
        let shadow_frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow frame"),
            layout: &shadow_frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.as_entire_binding(),
            }],
        });
        let bones_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bones"),
            layout: &bones_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bones.as_entire_binding(),
            }],
        });
        queue.write_buffer(&bones, 0, bytemuck::cast_slice(&[Mat4::IDENTITY; MAX_BONES]));

        let white = device
            .create_texture_with_data(
                &queue,
                &wgpu::TextureDescriptor {
                    label: Some("white"),
                    size: texture_size(1, 1),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255; 4],
            )
            .create_view(&wgpu::TextureViewDescriptor::default());

        let pipelines = Self::pipelines(
            &device,
            format,
            [
                &frame_layout,
                &shadow_frame_layout,
                &material_layout,
                &bones_layout,
                &blit_layout,
            ],
        );
        let (color, depth) = Self::targets(&device, format, width, height);
        let max_texture = device.limits().max_texture_dimension_2d;
        Self {
            device,
            queue,
            format,
            size: (width, height),
            color,
            depth,
            shadow,
            frame,
            frame_group,
            shadow_frame_group,
            material_layout,
            blit_layout,
            bones,
            bones_group,
            pipelines,
            samplers: HashMap::new(),
            linear,
            white,
            scene: Scene::default(),
            max_texture,
        }
    }

    fn pipelines(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        layouts: [&wgpu::BindGroupLayout; 5],
    ) -> Pipelines {
        struct Spec<'a> {
            layout: &'a wgpu::PipelineLayout,
            module: &'a wgpu::ShaderModule,
            vs: &'a str,
            fs: Option<&'a str>,
            buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
            topology: wgpu::PrimitiveTopology,
            /// Depth test and write; `None` for the shadow and blit passes' own targets.
            depth: Option<(wgpu::CompareFunction, bool)>,
            blend: Option<wgpu::BlendState>,
            shadow: bool,
            target: Option<wgpu::TextureFormat>,
        }
        let [frame, shadow_frame, material, bones, blit] = layouts;
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let shadow_shader = device.create_shader_module(wgpu::include_wgsl!("shadow.wgsl"));
        let blit_shader = device.create_shader_module(wgpu::include_wgsl!("blit.wgsl"));
        let pipeline_layout = |groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        let plain = pipeline_layout(&[Some(frame)]);
        let lit = pipeline_layout(&[Some(frame), Some(material)]);
        let skinned = pipeline_layout(&[Some(frame), Some(material), Some(bones)]);
        let shadow = pipeline_layout(&[Some(shadow_frame)]);
        let shadow_skinned = pipeline_layout(&[Some(shadow_frame), Some(bones)]);
        let blit_layout = pipeline_layout(&[Some(blit)]);

        let make = |s: Spec| {
            let targets = [s.target.map(|format| wgpu::ColorTargetState {
                format,
                blend: s.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(s.vs),
                layout: Some(s.layout),
                vertex: wgpu::VertexState {
                    module: s.module,
                    entry_point: Some(s.vs),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: s.buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: s.topology,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: if s.shadow {
                    Some(wgpu::DepthStencilState {
                        format: SHADOW,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState {
                            constant: 2,
                            slope_scale: 2.0,
                            clamp: 0.0,
                        },
                    })
                } else {
                    s.depth.map(|(compare, write)| wgpu::DepthStencilState {
                        format: DEPTH,
                        depth_write_enabled: Some(write),
                        depth_compare: Some(compare),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    })
                },
                multisample: wgpu::MultisampleState {
                    count: if s.depth.is_some() { SAMPLES } else { 1 },
                    ..Default::default()
                },
                fragment: s.fs.map(|fs| wgpu::FragmentState {
                    module: s.module,
                    entry_point: Some(fs),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let surface = [
            Some(layout(12, &POSITION)),
            Some(layout(12, &NORMAL)),
            Some(layout(8, &UV)),
            Some(layout(16, &COLOR)),
        ];
        let skinned_buffers = [
            surface[0].clone(),
            surface[1].clone(),
            surface[2].clone(),
            surface[3].clone(),
            Some(layout(16, &JOINTS)),
            Some(layout(16, &WEIGHTS)),
        ];
        let line_buffer = [Some(layout(16, &LINE))];
        let point_buffer = [Some(wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &LINE,
        })];
        let triangles = wgpu::PrimitiveTopology::TriangleList;
        let less = wgpu::CompareFunction::LessEqual;
        let alpha = Some(wgpu::BlendState::ALPHA_BLENDING);
        let scene = |vs, fs, layout, buffers, topology, depth, blend| Spec {
            layout,
            module: &shader,
            vs,
            fs: Some(fs),
            buffers,
            topology,
            depth: Some(depth),
            blend,
            shadow: false,
            target: Some(format),
        };
        let shadow_spec = |vs, layout, buffers| Spec {
            layout,
            module: &shadow_shader,
            vs,
            fs: None,
            buffers,
            topology: triangles,
            depth: None,
            blend: None,
            shadow: true,
            target: None,
        };
        let shadow_skinned_buffers = [
            surface[0].clone(),
            None,
            None,
            None,
            Some(layout(16, &JOINTS)),
            Some(layout(16, &WEIGHTS)),
        ];
        Pipelines {
            backdrop: make(scene(
                "vs_backdrop",
                "fs_backdrop",
                &plain,
                &[],
                triangles,
                (wgpu::CompareFunction::Always, false),
                None,
            )),
            floor: make(scene(
                "vs_floor",
                "fs_floor",
                &plain,
                &[],
                triangles,
                (less, false),
                alpha,
            )),
            mesh: [
                make(scene(
                    "vs_mesh",
                    "fs_mesh",
                    &lit,
                    &surface,
                    triangles,
                    (less, true),
                    None,
                )),
                make(scene(
                    "vs_mesh",
                    "fs_mesh",
                    &lit,
                    &surface,
                    triangles,
                    (less, false),
                    alpha,
                )),
            ],
            skinned: [
                make(scene(
                    "vs_skinned",
                    "fs_mesh",
                    &skinned,
                    &skinned_buffers,
                    triangles,
                    (less, true),
                    None,
                )),
                make(scene(
                    "vs_skinned",
                    "fs_mesh",
                    &skinned,
                    &skinned_buffers,
                    triangles,
                    (less, false),
                    alpha,
                )),
            ],
            line: make(scene(
                "vs_line",
                "fs_line",
                &plain,
                &line_buffer,
                wgpu::PrimitiveTopology::LineList,
                (less, true),
                alpha,
            )),
            overlay: make(scene(
                "vs_line",
                "fs_line",
                &plain,
                &line_buffer,
                wgpu::PrimitiveTopology::LineList,
                (wgpu::CompareFunction::Always, false),
                alpha,
            )),
            point: make(scene(
                "vs_point",
                "fs_point",
                &plain,
                &point_buffer,
                triangles,
                (less, true),
                None,
            )),
            shadow: make(shadow_spec("vs_shadow", &shadow, &surface[..1])),
            shadow_skinned: make(shadow_spec(
                "vs_shadow_skinned",
                &shadow_skinned,
                &shadow_skinned_buffers,
            )),
            shadow_point: make(shadow_spec("vs_shadow_point", &shadow, &point_buffer)),
            blit: make(Spec {
                layout: &blit_layout,
                module: &blit_shader,
                vs: "vs_blit",
                fs: Some("fs_blit"),
                buffers: &[],
                topology: triangles,
                depth: None,
                blend: None,
                shadow: false,
                target: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            }),
        }
    }

    fn targets(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::TextureView, wgpu::TextureView) {
        let make = |format, label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: texture_size(width.max(1), height.max(1)),
                    mip_level_count: 1,
                    sample_count: SAMPLES,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        (make(format, "color"), make(DEPTH, "depth"))
    }

    /// Resizes the targets the scene is drawn into.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) != self.size {
            self.size = (width, height);
            (self.color, self.depth) = Self::targets(&self.device, self.format, width, height);
        }
    }

    fn sampler(&mut self, key: SamplerKey) -> &wgpu::Sampler {
        self.samplers.entry(key).or_insert_with(|| {
            self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("material"),
                address_mode_u: key.0,
                address_mode_v: key.1,
                mag_filter: if key.2 {
                    wgpu::FilterMode::Nearest
                } else {
                    wgpu::FilterMode::Linear
                },
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            })
        })
    }

    fn material_group(&mut self, uniform: &wgpu::Buffer, image: Option<usize>, key: SamplerKey) -> wgpu::BindGroup {
        let view = image
            .and_then(|i| self.scene.textures.get(i).cloned().flatten())
            .unwrap_or_else(|| self.white.clone());
        let sampler = self.sampler(key).clone();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &self.material_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    }

    fn material(&mut self, m: &Material, has_texture: bool) -> MaterialGpu {
        let blend = m.alpha_cutoff.is_none() && m.color[3] < 0.999;
        let uniform = MaterialUniform {
            color: m.color,
            emissive: [
                m.emissive[0],
                m.emissive[1],
                m.emissive[2],
                f32::from(u8::from(has_texture)),
            ],
            params: [
                m.metallic,
                m.roughness,
                m.alpha_cutoff.unwrap_or(-1.0),
                f32::from(u8::from(m.unlit)),
            ],
            extra: [f32::from(u8::from(blend)), 0.0, 0.0, 0.0],
        };
        let uniform = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let wrap = m.sampling.color.wrap;
        let key = (
            address(wrap[0]),
            address(wrap[1]),
            m.sampling.color.filter[0] == Some(NEAREST),
        );
        let group = self.material_group(&uniform, m.color_image, key);
        MaterialGpu {
            uniform,
            group,
            image: m.color_image,
            sampler: key,
            blend,
        }
    }

    /// Shows `mesh`: in its bind pose with joints and weights for GPU skinning when `skinned`.
    /// Its textures are white until [`Renderer::set_texture_rgba`] (or `set_texture_bitmap`) fills them.
    pub fn set_mesh(&mut self, mesh: Option<&Mesh>, skinned: bool, radius: f32) {
        self.scene = Scene {
            skinned,
            textures: vec![None; mesh.map_or(0, |m| m.images.len())],
            ..Scene::default()
        };
        let Some(mesh) = mesh else {
            return;
        };
        let mut materials = Vec::with_capacity(mesh.materials.len().max(1));
        for m in &mesh.materials {
            materials.push(self.material(m, m.color_image.is_some()));
        }
        if materials.is_empty() {
            materials.push(self.material(&Material::default(), false));
        }
        self.scene.materials = materials;
        self.set_geometry(mesh, radius);
    }

    /// Replaces the geometry with `mesh`'s (a new pose of the same model), keeping materials and textures.
    pub fn update_mesh(&mut self, mesh: &Mesh, radius: f32) {
        let same = {
            let mut tri = mesh.prims.iter().filter(|p| !p.indices.is_empty());
            self.scene
                .draws
                .iter()
                .all(|d| tri.next().is_some_and(|p| p.positions.len() == d.vertices))
                && tri.next().is_none()
        };
        if same {
            for (d, p) in self
                .scene
                .draws
                .iter()
                .zip(mesh.prims.iter().filter(|p| !p.indices.is_empty()))
            {
                self.queue
                    .write_buffer(&d.buffers[0], 0, bytemuck::cast_slice(&p.positions));
                if p.normals.len() == p.positions.len() {
                    self.queue
                        .write_buffer(&d.buffers[1], 0, bytemuck::cast_slice(&p.normals));
                }
            }
            self.set_lines_and_points(mesh, radius);
        } else {
            self.set_geometry(mesh, radius);
        }
    }

    fn set_geometry(&mut self, mesh: &Mesh, radius: f32) {
        let device = &self.device;
        let buffer = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: usage | wgpu::BufferUsages::COPY_DST,
            })
        };
        let vertex = wgpu::BufferUsages::VERTEX;
        let materials = self.scene.materials.len();
        let mut draws = Vec::new();
        for p in mesh
            .prims
            .iter()
            .filter(|p| !p.indices.is_empty() && !p.positions.is_empty())
        {
            let n = p.positions.len();
            let normals: Vec<[f32; 3]> = if p.normals.len() == n {
                p.normals.clone()
            } else {
                vec![[0.0; 3]; n]
            };
            let uvs: Vec<[f32; 2]> = if p.uvs.len() == n {
                p.uvs.clone()
            } else {
                vec![[0.0; 2]; n]
            };
            let colors: Vec<[f32; 4]> = if p.colors.len() == n {
                p.colors.clone()
            } else {
                vec![[1.0; 4]; n]
            };
            let mut buffers = vec![
                buffer("positions", bytemuck::cast_slice(&p.positions), vertex),
                buffer("normals", bytemuck::cast_slice(&normals), vertex),
                buffer("uvs", bytemuck::cast_slice(&uvs), vertex),
                buffer("colors", bytemuck::cast_slice(&colors), vertex),
            ];
            if self.scene.skinned {
                let (joints, weights) = if p.skinned() {
                    (p.skin_joints.clone(), p.skin_weights.clone())
                } else {
                    (vec![[0; 4]; n], vec![[0.0; 4]; n])
                };
                buffers.push(buffer("joints", bytemuck::cast_slice(&joints), vertex));
                buffers.push(buffer("weights", bytemuck::cast_slice(&weights), vertex));
            }
            // Indices naming a missing vertex would read past the buffers: drop their triangles.
            let indices: Vec<u32> = p
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter(|t| t.iter().all(|&i| (i as usize) < n))
                .flatten()
                .copied()
                .collect();
            if indices.is_empty() {
                continue;
            }
            draws.push(Draw {
                buffers,
                index: buffer("indices", bytemuck::cast_slice(&indices), wgpu::BufferUsages::INDEX),
                count: indices.len() as u32,
                vertices: n,
                material: p.material.min(materials - 1),
            });
        }
        self.scene.draws = draws;
        self.set_lines_and_points(mesh, radius);
    }

    fn set_lines_and_points(&mut self, mesh: &Mesh, radius: f32) {
        let color = |p: &polysmelt::mesh::Prim, i: usize| {
            let m = mesh.materials.get(p.material).map_or([0.8, 0.8, 0.8, 1.0], |m| m.color);
            let c = p.colors.get(i).copied().unwrap_or([1.0; 4]);
            srgb8([m[0] * c[0], m[1] * c[1], m[2] * c[2], 1.0])
        };
        let mut lines = Vec::new();
        for p in &mesh.prims {
            for s in p.lines.as_chunks::<2>().0 {
                if let (Some(a), Some(b)) = (p.positions.get(s[0] as usize), p.positions.get(s[1] as usize)) {
                    lines.push(Vertex {
                        position: *a,
                        color: color(p, s[0] as usize),
                    });
                    lines.push(Vertex {
                        position: *b,
                        color: color(p, s[1] as usize),
                    });
                }
            }
        }
        let total: usize = mesh.prims.iter().map(|p| p.points.len()).sum();
        let step = total.div_ceil(MAX_POINTS).max(1);
        let mut points = Vec::with_capacity(total / step + 1);
        let mut k = 0usize;
        for p in &mesh.prims {
            for &i in &p.points {
                k += 1;
                if !(k - 1).is_multiple_of(step) {
                    continue;
                }
                if let Some(pos) = p.positions.get(i as usize) {
                    points.push(Vertex {
                        position: *pos,
                        color: color(p, i as usize),
                    });
                }
            }
        }
        // About the spacing of the points, were they spread over a surface the model's size.
        self.scene.point_size =
            (radius * 3.2 / (points.len().max(1) as f32).sqrt()).clamp(radius * 0.002, radius * 0.05);
        self.scene.lines = self.vertices("lines", &lines, false);
        self.scene.points = self.vertices("points", &points, false);
    }

    fn vertices(&self, label: &str, v: &[Vertex], dynamic: bool) -> Option<Lines> {
        (!v.is_empty()).then(|| Lines {
            buffer: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(v),
                usage: wgpu::BufferUsages::VERTEX
                    | if dynamic {
                        wgpu::BufferUsages::COPY_DST
                    } else {
                        wgpu::BufferUsages::empty()
                    },
            }),
            count: v.len() as u32,
        })
    }

    /// The skeleton drawn over the model: two points per bone.
    pub fn set_overlay(&mut self, segments: &[[f32; 3]]) {
        let color = [255, 138, 61, 230];
        let v: Vec<Vertex> = segments.iter().map(|&position| Vertex { position, color }).collect();
        match &mut self.scene.overlay {
            Some(o) if o.buffer.size() >= (v.len() * size_of::<Vertex>()) as u64 && !v.is_empty() => {
                self.queue.write_buffer(&o.buffer, 0, bytemuck::cast_slice(&v));
                o.count = v.len() as u32;
            }
            _ => self.scene.overlay = self.vertices("skeleton", &v, true),
        }
    }

    /// The bone matrices skinned meshes are drawn with.
    pub fn set_bones(&self, bones: &[Mat4]) {
        let n = bones.len().min(MAX_BONES);
        if n > 0 {
            self.queue
                .write_buffer(&self.bones, 0, bytemuck::cast_slice(&bones[..n]));
        }
    }

    /// How many images the materials may read.
    pub fn image_slots(&self) -> usize {
        self.scene.textures.len()
    }

    fn new_texture(&self, width: u32, height: u32) -> (wgpu::Texture, u32) {
        let levels = 32 - width.max(height).max(1).leading_zeros();
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image"),
            size: texture_size(width, height),
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        (texture, levels)
    }

    /// Fills each mipmap level below the first from the one above, then uses the image for every material that reads it.
    fn finish_texture(&mut self, image: usize, texture: &wgpu::Texture, levels: u32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mipmaps") });
        let level = |l: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: l,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        for l in 1..levels {
            let source = level(l - 1);
            let target = level(l);
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mipmap"),
                layout: &self.blit_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.linear),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mipmap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines.blit);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        if let Some(slot) = self.scene.textures.get_mut(image) {
            *slot = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        }
        for m in 0..self.scene.materials.len() {
            if self.scene.materials[m].image == Some(image) {
                let (uniform, key) = (self.scene.materials[m].uniform.clone(), self.scene.materials[m].sampler);
                self.scene.materials[m].group = self.material_group(&uniform, Some(image), key);
            }
        }
    }

    /// Image `image` of the mesh, decoded: RGBA, sRGB, `width` by `height`.
    pub fn set_texture_rgba(&mut self, image: usize, width: u32, height: u32, rgba: &[u8]) {
        if image >= self.scene.textures.len() || width == 0 || height == 0 {
            return;
        }
        let (texture, levels) = self.new_texture(width, height);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            texture_size(width, height),
        );
        self.finish_texture(image, &texture, levels);
    }

    /// Image `image` of the mesh, as the browser decoded it.
    #[cfg(target_arch = "wasm32")]
    pub fn set_texture_bitmap(&mut self, image: usize, bitmap: &web_sys::ImageBitmap) {
        let (width, height) = (bitmap.width(), bitmap.height());
        if image >= self.scene.textures.len() || width == 0 || height == 0 {
            return;
        }
        let (texture, levels) = self.new_texture(width, height);
        self.queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source: wgpu::ExternalImageSource::ImageBitmap(bitmap.clone()),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            texture_size(width, height),
        );
        self.finish_texture(image, &texture, levels);
    }

    /// Draws the scene into `target`, a view of the format the renderer was made for, its size.
    pub fn render(&self, target: &wgpu::TextureView, view: &View, stage: &Stage) {
        let (w, h) = (self.size.0 as f32, self.size.1 as f32);
        let r = stage.radius;
        let c = stage.center;
        // The key light: high, from over the camera's left shoulder.
        let yaw = view.yaw + 0.75;
        let l = Vec3::new(yaw.sin() * 0.55, 0.85, yaw.cos() * 0.55).normalize();
        let reach = r * 1.6;
        let light_view = glam::camera::rh::view::look_at_mat4(c + l * r * 4.0, c, Vec3::Y);
        let light_proj = glam::camera::rh::proj::directx::orthographic(-reach, reach, -reach, reach, r * 0.5, r * 8.0);
        let focal = h / 2.0 / (view.fov / 2.0).tan();
        let uniform = FrameUniform {
            view_proj: view.view_proj.to_cols_array(),
            light_view_proj: (light_proj * light_view).to_cols_array(),
            inv_view_proj: view.view_proj.inverse().to_cols_array(),
            camera_pos: view.eye.extend(1.0).to_array(),
            light_dir: [l.x, l.y, l.z, 2.0],
            ground: [stage.floor, r, grid_spacing(r), f32::from(u8::from(stage.show_floor))],
            center: c.extend(0.0).to_array(),
            viewport: [w, h, 1.0 / w, 1.0 / h],
            params: [stage.exposure, 0.85, 0.0, 1.0 / SHADOW_SIZE as f32],
            point: [self.scene.point_size, focal, 1.5, 24.0],
        };
        self.queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&uniform));

        let scene = &self.scene;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.shadow_frame_group, &[]);
            if scene.skinned {
                pass.set_pipeline(&self.pipelines.shadow_skinned);
                pass.set_bind_group(1, &self.bones_group, &[]);
            } else {
                pass.set_pipeline(&self.pipelines.shadow);
            }
            for d in &scene.draws {
                pass.set_vertex_buffer(0, d.buffers[0].slice(..));
                if scene.skinned {
                    pass.set_vertex_buffer(4, d.buffers[4].slice(..));
                    pass.set_vertex_buffer(5, d.buffers[5].slice(..));
                }
                pass.set_index_buffer(d.index.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..d.count, 0, 0..1);
            }
            if let Some(p) = &scene.points {
                pass.set_pipeline(&self.pipelines.shadow_point);
                pass.set_vertex_buffer(0, p.buffer.slice(..));
                pass.draw(0..6, 0..p.count);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color,
                    depth_slice: None,
                    resolve_target: Some(target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.frame_group, &[]);
            pass.set_pipeline(&self.pipelines.backdrop);
            pass.draw(0..3, 0..1);

            let meshes = |pass: &mut wgpu::RenderPass<'_>, blend: bool| {
                let pipelines = if scene.skinned {
                    &self.pipelines.skinned
                } else {
                    &self.pipelines.mesh
                };
                pass.set_pipeline(&pipelines[usize::from(blend)]);
                if scene.skinned {
                    pass.set_bind_group(2, &self.bones_group, &[]);
                }
                for d in scene
                    .draws
                    .iter()
                    .filter(|d| scene.materials[d.material].blend == blend)
                {
                    pass.set_bind_group(1, &scene.materials[d.material].group, &[]);
                    for (slot, b) in d.buffers.iter().enumerate() {
                        pass.set_vertex_buffer(slot as u32, b.slice(..));
                    }
                    pass.set_index_buffer(d.index.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..d.count, 0, 0..1);
                }
            };
            meshes(&mut pass, false);
            if let Some(p) = &scene.points {
                pass.set_pipeline(&self.pipelines.point);
                pass.set_vertex_buffer(0, p.buffer.slice(..));
                pass.draw(0..6, 0..p.count);
            }
            if let Some(l) = &scene.lines {
                pass.set_pipeline(&self.pipelines.line);
                pass.set_vertex_buffer(0, l.buffer.slice(..));
                pass.draw(0..l.count, 0..1);
            }
            if stage.show_floor {
                pass.set_pipeline(&self.pipelines.floor);
                pass.draw(0..6, 0..1);
            }
            meshes(&mut pass, true);
            if let (true, Some(o)) = (stage.show_skeleton, &scene.overlay) {
                pass.set_pipeline(&self.pipelines.overlay);
                pass.set_vertex_buffer(0, o.buffer.slice(..));
                pass.draw(0..o.count, 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
    }
}

/// A grid spacing for a model of `radius`: a power of ten, so that a few dozen lines cross it.
fn grid_spacing(radius: f32) -> f32 {
    10f32.powf((radius * 0.25).max(1e-6).log10().round())
}
