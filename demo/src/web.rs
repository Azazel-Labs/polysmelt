//! What the page calls: a viewer on a canvas that reads dropped files and draws them.

use std::sync::Arc;

use polysmelt::assets::Memory;
use polysmelt::mesh::Mesh;
use wasm_bindgen::prelude::*;

use crate::camera::Orbit;
use crate::model::{self, Model, Motion};
use crate::render::{Renderer, Stage, View};

fn now() -> f64 {
    web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now())
}

fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// The model on show and how it's playing.
struct Showing {
    model: Model,
    /// The animation playing, if any.
    animation: Option<usize>,
    time: f32,
    playing: bool,
    /// How long the last pose took to flatten, in milliseconds, and when it was made: slow models pose less often.
    pose_ms: f64,
    posed_at: f64,
}

/// A model viewer on a canvas.
#[wasm_bindgen]
pub struct Viewer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    view_format: wgpu::TextureFormat,
    renderer: Renderer,
    backend: &'static str,
    files: Memory,
    showing: Option<Showing>,
    camera: Orbit,
    home: Orbit,
    speed: f32,
    show_floor: bool,
    show_skeleton: bool,
    /// When the last frame was drawn, in milliseconds.
    last: Option<f64>,
    /// Whether the next frame needs drawing though nothing moves.
    dirty: bool,
    /// Counts the models shown, so images decoded for one that's gone are dropped.
    generation: u32,
}

#[wasm_bindgen]
impl Viewer {
    /// A viewer drawing on `canvas`: with WebGPU where the browser has it, else WebGL 2.
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<Viewer, JsValue> {
        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let desc = wgpu::InstanceDescriptor::new_without_display_handle();
        let instance = wgpu::util::new_instance_with_webgpu_detection(desc).await;
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| error(format!("can't draw on the canvas: {e}")))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| error(format!("no GPU adapter: {e}")))?;
        let backend = match adapter.get_info().backend {
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "WebGL 2",
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("polysmelt demo"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| error(format!("no GPU device: {e}")))?;
        device.on_uncaptured_error(Arc::new(|e| {
            web_sys::console::error_1(&format!("GPU error: {e}").into());
        }));
        let caps = surface.get_capabilities(&adapter);
        // Drawn through an sRGB view: of the canvas's plain format where views may differ from it (WebGPU), else
        // of an sRGB format the canvas offers itself (WebGL 2).
        let views = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::SURFACE_VIEW_FORMATS);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f.remove_srgb_suffix(),
                    wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
                ) && (views || f.is_srgb())
            })
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| error("the canvas offers no format"))?;
        let format = if views { format.remove_srgb_suffix() } else { format };
        let view_format = if views { format.add_srgb_suffix() } else { format };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: if view_format == format {
                vec![]
            } else {
                vec![view_format]
            },
        };
        surface.configure(&device, &config);
        let renderer = Renderer::new(device, queue, view_format, width, height);
        let home = Orbit::framing(glam::Vec3::ZERO, 1.0);
        Ok(Viewer {
            surface,
            config,
            view_format,
            renderer,
            backend,
            files: Memory::new(),
            showing: None,
            camera: home,
            home,
            speed: 1.0,
            show_floor: true,
            show_skeleton: false,
            last: None,
            dirty: true,
            generation: 0,
        })
    }

    /// `WebGPU` or `WebGL 2`.
    pub fn backend(&self) -> String {
        self.backend.into()
    }

    /// Forgets the files added.
    pub fn clear_files(&mut self) {
        self.files = Memory::new();
    }

    /// Adds a file the model may be, or may refer to, at its path in the folder dropped.
    pub fn add_file(&mut self, path: &str, bytes: Vec<u8>) {
        self.files.insert(path, bytes);
    }

    /// The file among those added most likely to be the model.
    pub fn pick_main(&self) -> Option<String> {
        model::pick_main(
            self.files
                .paths()
                .map(|p| (p, self.files.get(p).map_or(&[][..], |b| &b[..]))),
        )
    }

    /// Reads `name` from the files added and shows it, in place of what was shown. Gives what the panel shows, as
    /// JSON. The files are forgotten after.
    pub fn load(&mut self, name: &str) -> Result<String, JsValue> {
        let files = std::mem::replace(&mut self.files, Memory::new());
        let model = Model::load(files, name, &now).map_err(error)?;
        self.generation += 1;
        let (center, radius) = model.sphere();
        self.home = Orbit::framing(center, radius);
        self.camera = self.home;
        let gpu = model.motion == Motion::Gpu;
        self.renderer.set_mesh(model.mesh.as_ref(), gpu, radius);
        let animation = (!model.rig.animations.is_empty()).then_some(0);
        self.show_skeleton = model.bones_only();
        let mut info = model.info.clone();
        info["file"] = name.into();
        info["generation"] = self.generation.into();
        let time = animation.map_or(0.0, |a| model.span(a).0);
        self.showing = Some(Showing {
            model,
            animation,
            time,
            playing: animation.is_some(),
            pose_ms: 0.0,
            posed_at: f64::NEG_INFINITY,
        });
        self.pose(true);
        self.dirty = true;
        Ok(info.to_string())
    }

    /// Which models' images are wanted: the one `load` last showed.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// How many images the model's materials read.
    pub fn image_count(&self) -> usize {
        self.showing
            .as_ref()
            .and_then(|s| s.model.mesh.as_ref())
            .map_or(0, |m| m.images.len())
    }

    /// Image `i`'s encoded bytes, for the browser to decode.
    pub fn image_bytes(&self, i: usize) -> Vec<u8> {
        self.image(i).map_or_default(|m| m.bytes.to_vec())
    }

    /// Image `i`'s type: `image/png`, `image/jpeg` or `image/webp`.
    pub fn image_mime(&self, i: usize) -> String {
        self.image(i).map_or(String::new(), |m| m.mime.into())
    }

    /// The largest texture side the GPU takes: bigger images are scaled to it before they're given.
    pub fn max_texture_size(&self) -> u32 {
        self.renderer.max_texture
    }

    /// Image `i` of the model `generation` names, decoded by the browser.
    pub fn set_image(&mut self, generation: u32, i: usize, bitmap: web_sys::ImageBitmap) {
        if generation == self.generation {
            self.renderer.set_texture_bitmap(i, &bitmap);
            self.dirty = true;
        }
    }

    /// The model as a GLB file.
    pub fn to_glb(&self) -> Result<Vec<u8>, JsValue> {
        let s = self.showing.as_ref().ok_or_else(|| error("nothing is loaded"))?;
        s.model.doc.to_glb().map_err(|e| error(format!("{e:#}")))
    }

    /// Resizes the drawing to `width` by `height` pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) != (self.config.width, self.config.height) {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.renderer.device, &self.config);
            self.renderer.resize(width, height);
            self.dirty = true;
        }
    }

    /// Turns the camera by a drag of `dx`, `dy` pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.camera.rotate(dx, dy);
        self.dirty = true;
    }

    /// Slides the view by a drag of `dx`, `dy` pixels.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.camera.pan(dx, dy, self.config.height as f32);
        self.dirty = true;
    }

    /// Moves in (`factor` below 1) or out.
    pub fn zoom(&mut self, factor: f32) {
        self.camera.zoom(factor);
        self.dirty = true;
    }

    /// Frames the model again.
    pub fn reset_view(&mut self) {
        self.camera = self.home;
        self.dirty = true;
    }

    /// Plays animation `index`, or none (the rest pose) when it's negative.
    pub fn set_animation(&mut self, index: i32) {
        if let Some(s) = &mut self.showing {
            s.animation = usize::try_from(index)
                .ok()
                .filter(|&i| i < s.model.rig.animations.len());
            s.time = s.animation.map_or(0.0, |a| s.model.span(a).0);
            self.pose(true);
        }
    }

    pub fn set_playing(&mut self, playing: bool) {
        if let Some(s) = &mut self.showing {
            s.playing = playing;
        }
        self.dirty = true;
    }

    /// Jumps to `time` seconds into the animation.
    pub fn set_time(&mut self, time: f32) {
        if let Some(s) = &mut self.showing {
            s.time = time;
            self.pose(true);
        }
    }

    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    pub fn set_floor(&mut self, on: bool) {
        self.show_floor = on;
        self.dirty = true;
    }

    pub fn set_skeleton(&mut self, on: bool) {
        self.show_skeleton = on;
        if on {
            self.pose(true);
        }
        self.dirty = true;
    }

    /// Where the animation is, in seconds.
    pub fn time(&self) -> f32 {
        self.showing.as_ref().map_or(0.0, |s| s.time)
    }

    /// Moves the animation on to `now` (milliseconds) and draws, when anything changed.
    pub fn frame(&mut self, now: f64) {
        let dt = self.last.map_or(0.0, |l| ((now - l) / 1000.0).clamp(0.0, 0.25)) as f32;
        self.last = Some(now);
        let mut moved = false;
        if let Some(s) = &mut self.showing
            && let (true, Some(a)) = (s.playing, s.animation)
        {
            let (t0, t1) = s.model.span(a);
            let length = t1 - t0;
            s.time = if length > 0.0 {
                t0 + (s.time - t0 + dt * self.speed).rem_euclid(length)
            } else {
                t0
            };
            moved = true;
        }
        if moved {
            self.pose(false);
        }
        if self.dirty {
            self.draw();
            self.dirty = false;
        }
    }
}

impl Viewer {
    fn image(&self, i: usize) -> Option<&polysmelt::mesh::Image> {
        self.showing.as_ref()?.model.mesh.as_ref()?.images.get(i)
    }

    /// Poses the model where its animation is. A CPU-posed model that takes long to flatten is posed less often
    /// while it plays, unless `force`d.
    fn pose(&mut self, force: bool) {
        let Some(s) = &mut self.showing else {
            return;
        };
        let at = s.animation.map(|a| (a, s.time));
        let t = now();
        match s.model.motion {
            Motion::Gpu => self.renderer.set_bones(&s.model.bone_matrices(at)),
            Motion::Cpu if force || t - s.posed_at >= s.pose_ms * 3.0 => {
                match s.model.flatten(at) {
                    Ok(Some(mesh)) => self.renderer.update_mesh(&mesh, s.model.sphere().1),
                    Ok(None) => self.renderer.update_mesh(&Mesh::default(), s.model.sphere().1),
                    Err(e) => web_sys::console::warn_1(&format!("can't pose: {e}").into()),
                }
                let done = now();
                s.pose_ms = done - t;
                s.posed_at = done;
            }
            _ => {}
        }
        if self.show_skeleton && s.model.has_skeleton() {
            self.renderer.set_overlay(&s.model.skeleton(at));
        }
        self.dirty = true;
    }

    fn draw(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.renderer.device, &self.config);
                self.dirty = true;
                return;
            }
            _ => return,
        };
        let target = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });
        let (center, radius, floor) = self.showing.as_ref().map_or((glam::Vec3::ZERO, 1.0, 0.0), |s| {
            let (c, r) = s.model.sphere();
            (c, r, s.model.framed.map_or(0.0, |b| b.min[1]))
        });
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view = View {
            view_proj: self.camera.view_proj(aspect),
            eye: self.camera.eye(),
            yaw: self.camera.yaw,
            fov: self.camera.fov,
        };
        let stage = Stage {
            center,
            radius,
            floor,
            show_floor: self.show_floor && self.showing.is_some(),
            show_skeleton: self.show_skeleton,
            exposure: 1.0,
        };
        self.renderer.render(&target, &view, &stage);
        self.renderer.queue.present(frame);
    }
}
