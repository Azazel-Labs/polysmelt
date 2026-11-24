//! A model read with polysmelt, and what the viewer needs from it: the mesh to draw, how it
//! moves, where it is, and what the file says about itself.

use std::fmt::Display;

use glam::{Mat4, Vec3};
use polysmelt::assets::Memory;
use polysmelt::document::{Bounds, Document, Keep, Pose};
use polysmelt::formats;
use polysmelt::mesh::Mesh;
use polysmelt::rig::{Rig, Trs};
use polysmelt::source::Fact;
use serde_json::{Map, Value, json};

/// The most bones the GPU skinning path takes: a 16 KiB uniform buffer of matrices, WebGL 2's least.
pub const MAX_BONES: usize = 256;

/// How the model moves when an animation plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion {
    /// It has no animation.
    Still,
    /// Every primitive is skinned and nothing else animates: the mesh stays in its bind pose on the
    /// GPU and only the bone matrices change.
    Gpu,
    /// Anything else (morph targets, animated nodes, a mix): the whole mesh is flattened again in
    /// each pose.
    Cpu,
}

/// A model read and ready to draw.
pub struct Model {
    pub doc: Document<'static>,
    pub rig: Rig,
    /// What's drawn: in the bind pose for [`Motion::Gpu`], else at rest.
    pub mesh: Option<Mesh>,
    pub motion: Motion,
    /// The box around the model at rest (the skeleton's in the first frame, for a model that has no mesh).
    pub bounds: Option<Bounds>,
    /// The box the camera frames: around the rest pose and where the first animation takes the model.
    pub framed: Option<Bounds>,
    /// What the panel shows: the source, counts, size, animations, timings.
    pub info: Value,
    /// For each node, whether a skeleton line runs from it to its parent.
    bone_lines: Vec<bool>,
}

/// How many poses of an animation are looked at to frame it.
const FRAMED_POSES: u32 = 8;

/// The box around both.
fn union(a: Option<Bounds>, b: Option<Bounds>) -> Option<Bounds> {
    match (a, b) {
        (Some(a), Some(b)) => Some(Bounds {
            min: std::array::from_fn(|c| a.min[c].min(b.min[c])),
            max: std::array::from_fn(|c| a.max[c].max(b.max[c])),
        }),
        (a, b) => a.or(b),
    }
}

/// Extensions the readers take, best first for choosing a model among the files of a folder:
/// models (which may refer to the others), then motion with its own skeleton, then animation
/// alone, which needs a model to mean much.
const MODELS: &[&str] = &[
    "gltf", "glb", "vrm", "fbx", "dae", "zae", "kmz", "usdz", "usd", "usda", "usdc", "3mf", "skp", "obj", "3ds", "x3d",
    "x3dv", "x3dz", "wrl", "wrz", "amf", "lwo", "ac", "flt", "ogex", "x", "pmx", "pmd", "md2", "md3", "md5mesh", "iqm",
    "ms3d", "b3d", "mesh", "smd", "stl", "ply", "off", "dxf", "vox", "splat", "spz", "las", "laz", "pcd", "e57", "xyz",
    "pts", "ptx", "asc",
];
const MOTION: &[&str] = &["asf", "bvh", "c3d", "trc", "htr"];
const ANIMATION: &[&str] = &["vmd", "md5anim", "amc", "vta", "vrma", "skeleton"];
/// Files that only ever come with a model.
const SIDE_FILES: &[&str] = &[
    "mtl", "bin", "material", "png", "jpg", "jpeg", "webp", "tga", "bmp", "gif", "tif", "tiff", "psd", "dds", "ktx2",
    "rgb", "rgba", "sgi", "int", "inta", "bw", "skin", "shader", "txt", "md", "json", "html",
];

/// The file's extension, lowercase, with OGRE's `.mesh.xml` and `.skeleton.xml` as `mesh` and `skeleton`.
#[must_use]
pub fn extension(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    for (double, single) in [(".mesh.xml", "mesh"), (".skeleton.xml", "skeleton")] {
        if name.ends_with(double) {
            return single.into();
        }
    }
    name.rsplit_once('.').map_or_default(|(_, e)| e.to_owned())
}

/// Whether polysmelt names this extension among its formats but this build can't read it.
#[must_use]
pub fn unsupported(ext: &str) -> bool {
    matches!(ext, "usd" | "usda" | "usdc" | "usdz")
}

/// The file among `files` (path and bytes) most likely to be the model: by extension, then nearest the top of the
/// folder, then by name. Files whose extension says nothing are sniffed.
#[must_use]
pub fn pick_main<'a>(files: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> Option<String> {
    let mut best: Option<((usize, usize, usize), &str)> = None;
    for (path, bytes) in files {
        let ext = extension(path);
        let rank = if let Some(i) = MODELS.iter().position(|e| *e == ext) {
            (0, i)
        } else if let Some(i) = MOTION.iter().position(|e| *e == ext) {
            (1, i)
        } else if let Some(i) = ANIMATION.iter().position(|e| *e == ext) {
            (2, i)
        } else if ext == "xml" {
            (3, 0)
        } else if SIDE_FILES.contains(&ext.as_str()) || formats::detect(bytes).is_none() {
            continue;
        } else {
            (4, 0)
        };
        let key = (rank.0, path.matches('/').count(), rank.1);
        if best.is_none_or(|(k, p)| key < k || (key == k && path < p)) {
            best = Some((key, path));
        }
    }
    best.map(|(_, p)| p.to_owned())
}

/// A [`Fact`] for the panel: its value and whether the file declared it.
fn fact<T: Display>(f: &Fact<T>) -> Value {
    match f {
        Fact::Declared(v) => json!({ "value": v.to_string(), "declared": true }),
        Fact::Assumed(v) => json!({ "value": v.to_string(), "declared": false }),
        Fact::Unknown => Value::Null,
    }
}

fn count(json: &Map<String, Value>, key: &str) -> usize {
    json.get(key).and_then(Value::as_array).map_or(0, Vec::len)
}

impl Model {
    /// Reads `name` from `files` (where the files it refers to are found too) and flattens it. `now` is a clock in
    /// milliseconds, for the timings the panel shows.
    pub fn load(files: &Memory, name: &str, now: &dyn Fn() -> f64) -> Result<Self, String> {
        let bytes = files.get(name).ok_or_else(|| format!("{name} isn't among the files"))?;
        let ext = extension(name);
        if unsupported(&ext) {
            return Err(format!(
                "USD (.{ext}) needs OpenUSD, which doesn't build for the web yet. Every other format works here, \
                 and the polysmelt crate reads USD natively."
            ));
        }
        let t0 = now();
        let doc = formats::load_bytes(bytes, name, &ext, files, Keep::ALL)
            .map_err(|e| format!("{e:#}"))?
            .into_owned();
        let t1 = now();
        let rig = doc.rig().map_err(|e| format!("{e:#}"))?;
        let animated = !rig.animations.is_empty();
        let rest = doc.flatten(Pose::Rest).map_err(|e| format!("{e:#}"))?;
        let mut motion = if animated { Motion::Cpu } else { Motion::Still };
        let mut mesh = rest;
        if animated && gpu_skinnable(&doc, &rig, mesh.as_ref()) {
            let bind = doc.flatten(Pose::Bind).map_err(|e| format!("{e:#}"))?;
            if let Some(b) =
                bind.filter(|b| b.skeleton.len() <= MAX_BONES && b.prims.iter().all(polysmelt::mesh::Prim::skinned))
            {
                mesh = Some(b);
                motion = Motion::Gpu;
            }
        }
        let t2 = now();

        // A model drawn as nothing but nodes (motion capture) shows its skeleton.
        let joints: Vec<bool> = {
            let mut j = vec![false; rig.nodes.len()];
            for s in &rig.skins {
                for &n in &s.joints {
                    j[n] = true;
                }
            }
            j
        };
        let bare = mesh.as_ref().is_none_or(Mesh::is_empty_mesh);
        // Bones alone: every link but those from a still root (a wrapper at the origin, not a hip).
        let moving: Vec<bool> = {
            let mut m = vec![false; rig.nodes.len()];
            for c in rig.animations.iter().flat_map(|a| &a.channels) {
                if let Some(m) = m.get_mut(c.node) {
                    *m = true;
                }
            }
            m
        };
        let bone_lines: Vec<bool> = rig
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                n.parent
                    .is_some_and(|p| (joints[i] && joints[p]) || (bare && (rig.nodes[p].parent.is_some() || moving[p])))
            })
            .collect();
        let mut bounds = doc.bounds(Pose::Rest).map_err(|e| format!("{e:#}"))?;
        let mut model = Model {
            doc,
            rig,
            mesh,
            motion,
            bounds: None,
            framed: None,
            info: Value::Null,
            bone_lines,
        };
        let (start, end) = model.span(0);
        let at = |k: u32, of: u32| animated.then_some((0, start + (end - start) * k as f32 / of as f32));
        let around = |segs: Vec<[f32; 3]>| {
            segs.iter()
                .fold(None, |b, p| union(b, Some(Bounds { min: *p, max: *p })))
        };
        let mut framed = bounds;
        if bounds.is_none() {
            // Bones alone: the box around the joints, framed all through the first animation so that motion capture
            // that walks stays in view.
            bounds = around(model.skeleton(at(0, 1)));
            framed = if animated {
                around((0..=32).flat_map(|k| model.skeleton(at(k, 32))).collect())
            } else {
                bounds
            };
        } else if animated && (t2 - t1) * f64::from(FRAMED_POSES) < 150.0 {
            // Animated and quick to pose: frame what the first animation reaches, not the rest pose alone.
            for (index, time) in (0..=FRAMED_POSES).filter_map(|k| at(k, FRAMED_POSES)) {
                if let Ok(b) = model.doc.bounds(Pose::Animation { index, time }) {
                    framed = union(framed, b);
                }
            }
        }
        model.bounds = bounds;
        model.framed = framed;
        model.info = model.describe(t1 - t0, t2 - t1);
        Ok(model)
    }

    /// Animation `i`'s first and last key times.
    #[must_use]
    pub fn span(&self, i: usize) -> (f32, f32) {
        self.rig
            .animations
            .get(i)
            .map_or((0.0, 0.0), polysmelt::rig::Animation::span)
    }

    /// Whether skeleton lines can be drawn.
    #[must_use]
    pub fn has_skeleton(&self) -> bool {
        self.bone_lines.iter().any(|&b| b)
    }

    /// Whether there's nothing to draw but the skeleton.
    #[must_use]
    pub fn bones_only(&self) -> bool {
        self.mesh.as_ref().is_none_or(Mesh::is_empty_mesh)
    }

    /// The local transforms in animation `anim` at its time, or at rest.
    fn pose(&self, anim: Option<(usize, f32)>) -> Vec<Trs> {
        let (a, t) = anim.map_or((None, 0.0), |(a, t)| (Some(a), t));
        self.rig.pose(a, t)
    }

    /// The skeleton's lines, two points each, in a pose.
    #[must_use]
    pub fn skeleton(&self, anim: Option<(usize, f32)>) -> Vec<[f32; 3]> {
        let world = self.rig.world(&self.pose(anim));
        let at = |i: usize| [world[i][12], world[i][13], world[i][14]];
        let mut out = Vec::new();
        for (i, n) in self.rig.nodes.iter().enumerate() {
            if let (true, Some(p)) = (self.bone_lines[i], n.parent) {
                out.push(at(p));
                out.push(at(i));
            }
        }
        out
    }

    /// The bone matrices for [`Motion::Gpu`] in a pose: each takes a vertex from the bind pose to the pose.
    #[must_use]
    pub fn bone_matrices(&self, anim: Option<(usize, f32)>) -> Vec<Mat4> {
        let Some(mesh) = &self.mesh else {
            return Vec::new();
        };
        let world = self.rig.world(&self.pose(anim));
        mesh.skeleton
            .iter()
            .map(|b| {
                let w = world.get(b.node).map_or(Mat4::IDENTITY, Mat4::from_cols_array);
                w * Mat4::from_cols_array(&b.matrix).inverse()
            })
            .collect()
    }

    /// The mesh flattened in a pose, for [`Motion::Cpu`].
    pub fn flatten(&self, anim: Option<(usize, f32)>) -> Result<Option<Mesh>, String> {
        let pose = anim.map_or(Pose::Rest, |(index, time)| Pose::Animation { index, time });
        self.doc.flatten(pose).map_err(|e| format!("{e:#}"))
    }

    /// The center of the framed box and its radius.
    #[must_use]
    pub fn sphere(&self) -> (Vec3, f32) {
        self.framed.map_or((Vec3::ZERO, 1.0), |b| {
            let size = Vec3::from(b.size());
            (Vec3::from(b.center()), (size.length() / 2.0).max(1e-4))
        })
    }

    /// What the panel shows, as JSON.
    fn describe(&self, parse_ms: f64, flatten_ms: f64) -> Value {
        let src = self.doc.source_info();
        let json = &self.doc.json;
        let counts = self.doc.counts();
        let meters = src.output_meters_per_unit();
        let size = self.bounds.map(|b| {
            let k = meters.value().copied().unwrap_or(1.0);
            b.size().map(|s| f64::from(s) * k)
        });
        let animations: Vec<Value> = self
            .rig
            .animations
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let (t0, t1) = a.span();
                json!({
                    "name": a.name.clone().unwrap_or_else(|| format!("Animation {}", i + 1)),
                    "start": t0,
                    "end": t1,
                    "frameRate": a.frame_rate,
                    "morph": a.channels.iter().any(|c| c.path == polysmelt::rig::Path::Weights),
                })
            })
            .collect();
        let lights = json
            .get("extensions")
            .and_then(|e| e.pointer("/KHR_lights_punctual/lights"))
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let mesh = self.mesh.as_ref();
        json!({
            "format": src.format,
            "version": src.version,
            "tool": src.tool,
            "author": src.author,
            "title": src.title,
            "copyright": src.copyright,
            "created": src.created,
            "unit": src.unit,
            "metersPerUnit": fact(&src.meters_per_unit),
            "upAxis": fact(&src.up_axis.map(polysmelt::source::Axis::as_str)),
            "handedness": fact(&src.handedness.map(|h| format!("{h:?}").to_lowercase())),
            "frameRate": fact(&src.frame_rate),
            "converted": {
                "units": src.converted.units,
                "upAxis": src.converted.up_axis,
                "handedness": src.converted.handedness,
            },
            "unitsKnown": meters.is_known(),
            "triangles": counts.triangles,
            "vertices": counts.vertices,
            "points": counts.points,
            "lines": counts.lines,
            "size": size,
            "materials": count(json, "materials"),
            "textures": count(json, "textures"),
            "images": count(json, "images"),
            "meshes": count(json, "meshes"),
            "nodes": count(json, "nodes"),
            "skins": count(json, "skins"),
            "cameras": count(json, "cameras"),
            "lights": lights,
            "animations": animations,
            "bones": mesh.map_or(0, |m| m.skeleton.len()),
            "motion": match self.motion {
                Motion::Still => "still",
                Motion::Gpu => "gpu",
                Motion::Cpu => "cpu",
            },
            "skeleton": self.has_skeleton(),
            "bonesOnly": self.bones_only(),
            "parseMs": parse_ms,
            "flattenMs": flatten_ms,
            "facts": self.doc.facts,
        })
    }
}

/// Whether skinning on the GPU poses the model as flattening it would: every primitive drawn is skinned, nothing
/// sets morph weights, and nothing is drawn as lines or points (which the GPU path doesn't skin).
fn gpu_skinnable(doc: &Document<'_>, rig: &Rig, rest: Option<&Mesh>) -> bool {
    let Some(rest) = rest else {
        return false;
    };
    let morphs = rig
        .animations
        .iter()
        .any(|a| a.channels.iter().any(|c| c.path == polysmelt::rig::Path::Weights));
    let counts = doc.counts();
    !rig.skins.is_empty()
        && !morphs
        && counts.points == 0
        && counts.lines == 0
        && !rest.prims.is_empty()
        && rest.prims.iter().all(polysmelt::mesh::Prim::skinned)
}

/// Whether a mesh draws nothing.
trait EmptyMesh {
    fn is_empty_mesh(&self) -> bool;
}

impl EmptyMesh for Mesh {
    fn is_empty_mesh(&self) -> bool {
        self.prims.iter().all(polysmelt::mesh::Prim::is_empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(path: &str) -> (Memory, String) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/corpus");
        let dir = root.join(path).parent().unwrap().to_path_buf();
        let mut files = Memory::new();
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_file() {
                let name = p.file_name().unwrap().to_str().unwrap().to_owned();
                files.insert(&name, std::fs::read(&p).unwrap());
            }
        }
        let name = std::path::Path::new(path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        (files, name)
    }

    fn clock() -> f64 {
        0.0
    }

    #[test]
    fn picks_the_model_among_its_files() {
        let files: Vec<(&str, &[u8])> = vec![
            ("scene/scene.bin", b"\0\0\0\0"),
            ("scene/textures/a.png", b"\x89PNG"),
            ("scene/scene.gltf", b"{}"),
            ("readme.txt", b"hello"),
        ];
        assert_eq!(pick_main(files).as_deref(), Some("scene/scene.gltf"));
        let files: Vec<(&str, &[u8])> = vec![("a.mtl", b"newmtl x"), ("a.obj", b"v 0 0 0")];
        assert_eq!(pick_main(files).as_deref(), Some("a.obj"));
        let files: Vec<(&str, &[u8])> = vec![("walk.md5anim", b""), ("body.md5mesh", b"")];
        assert_eq!(pick_main(files).as_deref(), Some("body.md5mesh"));
        assert_eq!(extension("Rig.MESH.xml"), "mesh");
    }

    #[test]
    fn gpu_skinning_poses_as_flattening_does() {
        let (files, name) = corpus("gltf/skin_animated/skin.glb");
        let model = Model::load(&files, &name, &clock).unwrap();
        assert_eq!(model.motion, Motion::Gpu);
        let bind = model.mesh.as_ref().unwrap();
        let (t0, t1) = model.span(0);
        for k in 0..=8 {
            let t = t0 + (t1 - t0) * k as f32 / 8.0;
            let bones = model.bone_matrices(Some((0, t)));
            let flat = model.flatten(Some((0, t))).unwrap().unwrap();
            for (pb, pf) in bind.prims.iter().zip(&flat.prims) {
                assert_eq!(pb.positions.len(), pf.positions.len());
                for (v, (p, want)) in pb.positions.iter().zip(&pf.positions).enumerate() {
                    let mut got = Vec3::ZERO;
                    for (j, w) in pb.skin_joints[v].iter().zip(pb.skin_weights[v]) {
                        got += bones[*j as usize].transform_point3(Vec3::from(*p)) * w;
                    }
                    let want = Vec3::from(*want);
                    assert!((got - want).length() < 1e-3, "t {t} vertex {v}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn motion_capture_frames_its_skeleton() {
        let (files, name) = corpus("bvh/humanoid/humanoid.bvh");
        let model = Model::load(&files, &name, &clock).unwrap();
        assert!(model.bones_only());
        assert!(model.has_skeleton());
        assert!(model.bounds.is_some());
        assert!(!model.skeleton(Some((0, 0.5))).is_empty());
    }

    #[test]
    fn morph_animation_moves_vertices() {
        let (files, name) = corpus("fbx/blender_shape_key_animated/shape_key_animated.fbx");
        let model = Model::load(&files, &name, &clock).unwrap();
        assert_eq!(model.motion, Motion::Cpu);
        let (t0, t1) = model.span(0);
        let a = model.flatten(Some((0, t0))).unwrap().unwrap();
        let b = model.flatten(Some((0, (t0 + t1) / 2.0))).unwrap().unwrap();
        assert_ne!(a.prims[0].positions, b.prims[0].positions);
    }
}
