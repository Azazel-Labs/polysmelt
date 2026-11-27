//! Renders models with the demo's renderer, offscreen, to PNG: the stage for the README hero and the social
//! preview, which `media/compose.sh` lays the wordmark over.
//!
//! ```text
//! cargo run --release --features media --bin media -- OUT.png WIDTH HEIGHT [--yaw DEG] [--pitch DEG]
//!     [--zoom K] [--shift X] [--lift Y] MODEL[#FINISH][@TIME][:SPIN] ...
//! ```
//!
//! Each model is scaled to the same height, stood on the floor and set beside the one before; `@TIME` poses it at
//! that time in its first animation, `:SPIN` turns it that many degrees about the vertical, and `#FINISH` (gold,
//! copper, steel, ember) gives its untextured materials that metal.

use std::path::Path;

use glam::{Mat3, Vec3};
use polysmelt::assets::Memory;
use polysmelt::mesh::{Material, Mesh, Prim};
use polysmelt_demo::camera::Orbit;
use polysmelt_demo::model::Model;
use polysmelt_demo::render::{Renderer, Stage, View};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("media: {message}");
    std::process::exit(1)
}

/// Reads `path` with every file of its folder beside it.
fn load(path: &Path) -> Model {
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut files = Memory::new();
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| fail(format!("{}: {e}", dir.display()))) {
        let p = entry.unwrap_or_else(|e| fail(e)).path();
        if p.is_file() {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_owned();
            files.insert(
                &name,
                std::fs::read(&p).unwrap_or_else(|e| fail(format!("{}: {e}", p.display()))),
            );
        }
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    Model::load(files, name, &|| 0.0).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())))
}

/// `model` posed at `time` (at rest without), its skeleton as orange lines when it has nothing else to draw.
fn posed(model: &Model, time: Option<f32>) -> Mesh {
    let at = time.filter(|_| !model.rig.animations.is_empty()).map(|t| (0, t));
    let mut mesh = model.flatten(at).unwrap_or_else(|e| fail(e)).unwrap_or_default();
    if model.bones_only() {
        let segments = model.skeleton(at);
        let material = mesh.materials.len();
        mesh.materials.push(Material::plain("skeleton", [1.0, 0.32, 0.06, 1.0]));
        mesh.prims.push(Prim {
            lines: (0..segments.len() as u32).collect(),
            positions: segments,
            material,
            ..Prim::default()
        });
    }
    mesh
}

/// Gives `mesh`'s untextured materials the metal `finish` names.
fn finish(mesh: &mut Mesh, finish: &str) {
    let (color, roughness, glow) = match finish {
        "gold" => ([1.0, 0.71, 0.29], 0.28, [0.0; 3]),
        "copper" => ([0.98, 0.48, 0.24], 0.3, [0.0; 3]),
        "steel" => ([0.62, 0.62, 0.66], 0.36, [0.0; 3]),
        "ember" => ([0.9, 0.35, 0.12], 0.4, [1.6, 0.42, 0.08]),
        _ => fail(format!("unknown finish {finish}")),
    };
    for m in &mut mesh.materials {
        if m.color_image.is_none() {
            m.color = [color[0], color[1], color[2], 1.0];
            m.metallic = 1.0;
            m.roughness = roughness;
            m.emissive = glow;
        }
    }
}

fn bounds(mesh: &Mesh) -> Option<(Vec3, Vec3)> {
    let mut b: Option<(Vec3, Vec3)> = None;
    for p in &mesh.prims {
        for &q in &p.positions {
            let q = Vec3::from(q);
            b = Some(b.map_or((q, q), |(lo, hi)| (lo.min(q), hi.max(q))));
        }
    }
    b
}

/// Adds `part` to `scene`, turned `spin` degrees, scaled to `height`, its left side at `x`; gives its right side.
fn place(scene: &mut Mesh, mut part: Mesh, spin: f32, height: f32, x: f32) -> f32 {
    let turn = Mat3::from_rotation_y(spin.to_radians());
    for p in &mut part.prims {
        for q in &mut p.positions {
            *q = (turn * Vec3::from(*q)).to_array();
        }
        for n in &mut p.normals {
            *n = (turn * Vec3::from(*n)).to_array();
        }
        p.skin_joints.clear();
        p.skin_weights.clear();
    }
    let Some((lo, hi)) = bounds(&part) else {
        return x;
    };
    let size = hi - lo;
    let scale = height / size.y.max(size.x.max(size.z) * 0.35).max(1e-6);
    let offset = Vec3::new(x - lo.x * scale, -lo.y * scale, -(lo.z + hi.z) / 2.0 * scale);
    let (materials, images) = (scene.materials.len(), scene.images.len());
    for mut p in part.prims {
        for q in &mut p.positions {
            *q = (Vec3::from(*q) * scale + offset).to_array();
        }
        p.material += materials;
        scene.prims.push(p);
    }
    for mut m in part.materials {
        for i in [
            &mut m.color_image,
            &mut m.normal_image,
            &mut m.metallic_roughness_image,
            &mut m.occlusion_image,
            &mut m.emissive_image,
        ]
        .into_iter()
        .flatten()
        {
            *i += images;
        }
        scene.materials.push(m);
    }
    scene.images.extend(part.images);
    x + size.x * scale
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        fail("usage: media OUT.png WIDTH HEIGHT [--yaw DEG] [--pitch DEG] [--zoom K] [--shift X] [--lift Y] MODEL...");
    }
    let out = &args[0];
    let width: u32 = args[1].parse().unwrap_or_else(|_| fail("WIDTH"));
    let height: u32 = args[2].parse().unwrap_or_else(|_| fail("HEIGHT"));
    let (mut yaw, mut pitch, mut zoom, mut shift, mut lift) = (30.0f32, 14.0f32, 1.0f32, 0.0f32, 0.0f32);
    let mut scene = Mesh::default();
    let mut x = 0.0;
    let mut rest = args[3..].iter();
    while let Some(a) = rest.next() {
        let mut number = || -> f32 { rest.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| fail(a)) };
        match a.as_str() {
            "--yaw" => yaw = number(),
            "--pitch" => pitch = number(),
            "--zoom" => zoom = number(),
            "--shift" => shift = number(),
            "--lift" => lift = number(),
            spec => {
                let (spec, spin) = spec
                    .rsplit_once(':')
                    .filter(|(_, s)| s.parse::<f32>().is_ok())
                    .map_or((spec, 0.0), |(p, s)| (p, s.parse().unwrap_or(0.0)));
                let (spec, time) = spec.rsplit_once('@').map_or((spec, None), |(p, t)| {
                    (p, Some(t.parse::<f32>().unwrap_or_else(|_| fail(spec))))
                });
                let (path, metal) = spec.rsplit_once('#').map_or((spec, None), |(p, f)| (p, Some(f)));
                let model = load(Path::new(path));
                let mut part = posed(&model, time);
                if let Some(metal) = metal {
                    finish(&mut part, metal);
                }
                let gap = if scene.prims.is_empty() { 0.0 } else { 0.25 };
                x = place(&mut scene, part, spin, 1.0, x + gap);
            }
        }
    }
    let (lo, hi) = bounds(&scene).unwrap_or_else(|| fail("nothing to draw"));
    let center = (lo + hi) / 2.0;
    let radius = ((hi - lo).length() / 2.0).max(1e-4);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .unwrap_or_else(|e| fail(format!("no GPU adapter: {e}")));
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("media"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))
    .unwrap_or_else(|e| fail(format!("no GPU device: {e}")));
    let mut renderer = Renderer::new(device, queue, FORMAT, width, height);
    renderer.set_mesh(Some(&scene), false, radius);
    for (i, image) in scene.images.iter().enumerate() {
        if let Ok(decoded) = image::load_from_memory(&image.bytes) {
            let rgba = decoded.to_rgba8();
            renderer.set_texture_rgba(i, rgba.width(), rgba.height(), rgba.as_raw());
        }
    }

    let mut camera = Orbit::framing(center, radius);
    camera.yaw = yaw.to_radians();
    camera.pitch = pitch.to_radians();
    // Positive shift moves the models right in the frame, positive lift moves them down.
    let right = Vec3::new(camera.yaw.cos(), 0.0, -camera.yaw.sin());
    camera.target += (Vec3::Y * lift - right * shift) * radius;
    camera.distance /= zoom;
    let view = View {
        view_proj: camera.view_proj(width as f32 / height as f32),
        eye: camera.eye(),
        yaw: camera.yaw,
        fov: camera.fov,
    };
    let stage = Stage {
        center,
        radius,
        floor: lo.y,
        show_floor: true,
        show_skeleton: false,
        exposure: 1.0,
    };
    let device = &renderer.device;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    renderer.render(
        &target.create_view(&wgpu::TextureViewDescriptor::default()),
        &view,
        &stage,
    );

    let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    renderer.queue.submit([encoder.finish()]);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap_or_else(|e| fail(e)));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap_or_else(|e| fail(e));
    let data = readback.slice(..).get_mapped_range().unwrap_or_else(|e| fail(e));
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let start = (y * row) as usize;
        for px in data[start..start + (width * 4) as usize].as_chunks::<4>().0 {
            pixels.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
    }
    image::save_buffer(out, &pixels, width, height, image::ColorType::Rgba8).unwrap_or_else(|e| fail(e));
    println!("{out}: {width}x{height}, {} triangles", scene.triangles());
}
