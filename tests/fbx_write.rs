//! The FBX writer, checked by reading back what it writes: scenes built to cover each thing it writes, then the
//! conformance corpus. Every file written is parsed by an independent FBX parser too.
#![cfg(all(feature = "fbx", feature = "gltf"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the scenes built here are small: counts, indices and grid steps fit any of these types"
)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::io::Cursor;

use polysmelt::Document;
use polysmelt::document::Pose;
use polysmelt::mesh::Mesh;
use serde_json::{Map, Value, json};

// ---------------------------------------------------------------------------------------------------------------
// Building glTF scenes to write.

/// A GLB being built: its binary chunk, buffer views and accessors.
#[derive(Default)]
struct Glb {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Glb {
    fn view(&mut self, bytes: &[u8]) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        self.views
            .push(json!({"buffer": 0, "byteOffset": self.bin.len(), "byteLength": bytes.len()}));
        self.bin.extend_from_slice(bytes);
        self.views.len() - 1
    }

    fn accessor(&mut self, bytes: &[u8], component: u32, count: usize, kind: &str) -> usize {
        let view = self.view(bytes);
        self.accessors
            .push(json!({"bufferView": view, "componentType": component, "count": count, "type": kind}));
        self.accessors.len() - 1
    }

    fn floats(&mut self, data: &[f32], kind: &str) -> usize {
        let width = match kind {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            _ => unreachable!(),
        };
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let a = self.accessor(&bytes, 5126, data.len() / width, kind);
        if kind == "VEC3" || kind == "SCALAR" {
            let mut min = vec![f32::INFINITY; width];
            let mut max = vec![f32::NEG_INFINITY; width];
            for c in data.chunks_exact(width) {
                for k in 0..width {
                    min[k] = min[k].min(c[k]);
                    max[k] = max[k].max(c[k]);
                }
            }
            self.accessors[a]["min"] = json!(min);
            self.accessors[a]["max"] = json!(max);
        }
        a
    }

    fn u16s(&mut self, data: &[u16], kind: &str) -> usize {
        let width = if kind == "VEC4" { 4 } else { 1 };
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.accessor(&bytes, 5123, data.len() / width, kind)
    }

    fn build(self, mut gltf: Value) -> Document {
        gltf["asset"] = json!({"version": "2.0"});
        let mut bin = self.bin;
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        gltf["buffers"] = json!([{"byteLength": bin.len()}]);
        gltf["bufferViews"] = Value::Array(self.views);
        gltf["accessors"] = Value::Array(self.accessors);
        let mut text = serde_json::to_vec(&gltf).unwrap();
        while !text.len().is_multiple_of(4) {
            text.push(b' ');
        }
        let total = 12 + 8 + text.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&u32::try_from(total).unwrap().to_le_bytes());
        glb.extend_from_slice(&u32::try_from(text.len()).unwrap().to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&text);
        glb.extend_from_slice(&u32::try_from(bin.len()).unwrap().to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);
        polysmelt::read(glb, "scene.glb").unwrap()
    }
}

/// A small PNG, a gradient so its bytes are its own.
fn png(seed: u8) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(4, 4, |x, y| {
        image::Rgba([seed.wrapping_mul(40), (x * 60) as u8, (y * 60) as u8, 255])
    });
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// A quaternion `[x, y, z, w]` turning `degrees` about unit `axis`.
fn quat(axis: [f32; 3], degrees: f32) -> [f32; 4] {
    let h = degrees.to_radians() / 2.0;
    [axis[0] * h.sin(), axis[1] * h.sin(), axis[2] * h.sin(), h.cos()]
}

// ---------------------------------------------------------------------------------------------------------------
// Writing and reading back.

/// Top-level records every file holds, in order.
const TOP: [&str; 11] = [
    "FBXHeaderExtension",
    "FileId",
    "CreationTime",
    "Creator",
    "GlobalSettings",
    "Documents",
    "References",
    "Definitions",
    "Objects",
    "Connections",
    "Takes",
];

/// Parses `bytes` with an independent FBX parser: the whole tree and the footer.
fn fbxcel_parses(bytes: &[u8]) -> Result<(), String> {
    match fbxcel::tree::any::AnyTree::from_seekable_reader(Cursor::new(bytes)) {
        Ok(fbxcel::tree::any::AnyTree::V7400(version, tree, footer)) => {
            if version != fbxcel::low::FbxVersion::V7_4 {
                return Err(format!("version {version:?}"));
            }
            footer.map_err(|e| format!("footer: {e}"))?;
            let names: Vec<String> = tree.root().children().map(|c| c.name().to_owned()).collect();
            if names != TOP {
                return Err(format!("top-level records {names:?}"));
            }
            Ok(())
        }
        Ok(_) => Err("an unknown version".to_owned()),
        Err(e) => Err(e.to_string()),
    }
}

/// Writes `doc` both ways, checks they agree and parse, and reads the file back. With `FBX_WRITE_OUT` set to a
/// folder, the file and the GLB it came from are kept there as `name`, to look at in other tools.
fn round_trip(doc: &Document, name: &str) -> (Vec<u8>, Document) {
    let fbx = doc.to_fbx().expect("writes");
    if let Ok(dir) = std::env::var("FBX_WRITE_OUT") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(format!("{name}.fbx")), &fbx).unwrap();
        std::fs::write(dir.join(format!("{name}.glb")), doc.to_glb().unwrap()).unwrap();
    }
    let mut streamed = Vec::new();
    doc.write_fbx(&mut streamed).unwrap();
    assert!(streamed == fbx, "write_fbx and to_fbx differ");
    assert!(doc.to_fbx().unwrap() == fbx, "output isn't deterministic");
    fbxcel_parses(&fbx).unwrap();
    let back = polysmelt::read(fbx.clone(), "scene.fbx").expect("reads back");
    (fbx, back)
}

// ---------------------------------------------------------------------------------------------------------------
// Comparing what went in with what came back.

/// What a material looks like, for comparing.
#[derive(Clone, Debug)]
struct Look {
    color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    double_sided: bool,
    cutoff: Option<f32>,
    image: Option<u64>,
}

fn hash(bytes: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

fn looks(mesh: &Mesh) -> Vec<Look> {
    mesh.materials
        .iter()
        .map(|m| Look {
            color: m.color,
            metallic: m.metallic,
            roughness: m.roughness,
            emissive: m.emissive,
            double_sided: m.double_sided,
            cutoff: m.alpha_cutoff,
            // A WebP image is compared as the PNG every reader but glTF's turns it into.
            image: m.color_image.and_then(|i| mesh.images.get(i)).map(|i| {
                i.clone()
                    .into_png_or_jpeg()
                    .map_or_else(|| hash(&i.bytes), |i| hash(&i.bytes))
            }),
        })
        .collect()
}

/// Which ways two looks differ.
fn look_diff(a: &Look, b: &Look, out: &mut BTreeSet<&'static str>) {
    let near = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(x, y)| (x - y).abs() <= 2e-3);
    if !near(&a.color, &b.color)
        || !near(&[a.metallic, a.roughness], &[b.metallic, b.roughness])
        || !near(&a.emissive, &b.emissive)
        || a.double_sided != b.double_sided
        || a.cutoff.is_some() != b.cutoff.is_some()
        || a.cutoff.zip(b.cutoff).is_some_and(|(x, y)| (x - y).abs() > 1e-3)
    {
        out.insert("material");
    }
    if a.image != b.image {
        out.insert("texture");
    }
}

/// A triangle corner and what it carries.
#[derive(Clone, Debug)]
struct Corner {
    p: [f32; 3],
    n: Option<[f32; 3]>,
    uv: Option<[f32; 2]>,
    color: Option<[f32; 4]>,
    /// Every other UV and color set, flattened.
    sets: Vec<f32>,
}

#[derive(Clone, Debug)]
struct Tri {
    k: [Corner; 3],
    look: usize,
}

fn corner(prim: &polysmelt::mesh::Prim, v: usize) -> Corner {
    let mut sets = Vec::new();
    for s in prim.uv_sets.iter().filter(|s| !s.is_empty()) {
        sets.extend_from_slice(&s[v]);
    }
    for s in prim.color_sets.iter().filter(|s| !s.is_empty()) {
        sets.extend_from_slice(&s[v]);
    }
    Corner {
        p: prim.positions[v],
        n: prim.normals.get(v).copied(),
        uv: prim.uvs.get(v).copied(),
        color: prim.colors.get(v).copied(),
        sets,
    }
}

fn cmp3(a: &[f32; 3], b: &[f32; 3]) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0])
        .then(a[1].total_cmp(&b[1]))
        .then(a[2].total_cmp(&b[2]))
}

/// A mesh's triangles, each turned to start at its least corner (keeping its winding), sorted.
fn tris(mesh: &Mesh) -> Vec<Tri> {
    let mut out = Vec::new();
    for prim in &mesh.prims {
        for t in prim.indices.chunks_exact(3) {
            let k = [0, 1, 2].map(|c| corner(prim, t[c] as usize));
            let first = (0..3).min_by(|&a, &b| cmp3(&k[a].p, &k[b].p)).unwrap();
            let k = [0, 1, 2].map(|c| k[(first + c) % 3].clone());
            out.push(Tri { k, look: prim.material });
        }
    }
    out.sort_by(|a, b| {
        cmp3(&a.k[0].p, &b.k[0].p)
            .then(cmp3(&a.k[1].p, &b.k[1].p))
            .then(cmp3(&a.k[2].p, &b.k[2].p))
    });
    out
}

/// The size of a rig's nodes at rest: what an angle off at a joint moves the joints past it by.
fn rig_extent(rig: &polysmelt::rig::Rig) -> f32 {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for w in rig.world(&rig.pose(None, 0.0)) {
        for k in 0..3 {
            lo[k] = lo[k].min(w[12 + k]);
            hi[k] = hi[k].max(w[12 + k]);
        }
    }
    let d: f32 = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f32>().sqrt();
    if d.is_finite() { d.max(1.0) } else { 1.0 }
}

fn extent(mesh: &Mesh) -> f32 {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in mesh.prims.iter().flat_map(|p| &p.positions) {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let d: f32 = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f32>().sqrt();
    if d.is_finite() { d.max(1.0) } else { 1.0 }
}

fn same_place(a: &Tri, b: &Tri, tol: f32) -> bool {
    (0..3).all(|c| (0..3).all(|k| (a.k[c].p[k] - b.k[c].p[k]).abs() <= tol))
}

/// Which ways two triangles at the same place differ.
fn tri_diff(a: &Tri, b: &Tri, la: &[Look], lb: &[Look], out: &mut BTreeSet<&'static str>) {
    let white = |c: &[f32; 4]| c.iter().all(|v| (v - 1.0).abs() < 2e-3);
    for c in 0..3 {
        let (x, y) = (&a.k[c], &b.k[c]);
        if let (Some(n), Some(m)) = (x.n, y.n) {
            let dot = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
            let len = (n.iter().map(|v| v * v).sum::<f32>() * m.iter().map(|v| v * v).sum::<f32>()).sqrt();
            if len > 1e-6 && dot / len < 0.995 {
                out.insert("normal");
            }
        }
        match (x.uv, y.uv) {
            (Some(u), Some(v)) if (u[0] - v[0]).abs() > 1e-3 || (u[1] - v[1]).abs() > 1e-3 => {
                out.insert("uv");
            }
            (Some(_), None) => {
                out.insert("uv");
            }
            _ => {}
        }
        match (x.color, y.color) {
            (Some(u), Some(v)) if u.iter().zip(&v).any(|(u, v)| (u - v).abs() > 3e-3) => {
                out.insert("color");
            }
            (Some(u), None) | (None, Some(u)) if !white(&u) => {
                out.insert("color");
            }
            _ => {}
        }
        // An FBX layer covers the whole geometry: a primitive without a set another has comes back with one.
        if !x.sets.is_empty()
            && (x.sets.len() != y.sets.len() || x.sets.iter().zip(&y.sets).any(|(u, v)| (u - v).abs() > 3e-3))
        {
            out.insert("uv/color sets");
        }
    }
    if let (Some(x), Some(y)) = (la.get(a.look), lb.get(b.look)) {
        look_diff(x, y, out);
    }
}

/// Which ways two flattened meshes differ: the triangles, where they are and what they carry, and the lines.
///
/// `tol` is how far apart points may be, relative to the mesh's size.
fn mesh_diff(a: &Mesh, b: &Mesh, places_only: bool, tol: f32) -> BTreeSet<&'static str> {
    let mut out = BTreeSet::new();
    let tol = tol * extent(a);
    let (ta, tb) = (tris(a), tris(b));
    let (la, lb) = (looks(a), looks(b));
    if ta.len() != tb.len() {
        out.insert("triangles");
    }
    let lines = |m: &Mesh| m.prims.iter().map(|p| p.lines.len()).sum::<usize>();
    if lines(a) != lines(b) {
        out.insert("lines");
    }
    let mut unmatched_a = Vec::new();
    let mut unmatched_b = Vec::new();
    if ta.len() == tb.len() {
        for (x, y) in ta.iter().zip(&tb) {
            if same_place(x, y, tol) {
                if !places_only {
                    tri_diff(x, y, &la, &lb, &mut out);
                }
            } else {
                unmatched_a.push(x);
                unmatched_b.push(y);
            }
        }
    } else {
        unmatched_a.extend(&ta);
        unmatched_b.extend(&tb);
    }
    // Triangles sorted a little differently: match them one by one, when there are few enough.
    if !unmatched_a.is_empty() {
        if unmatched_a.len() > 3000 || unmatched_b.len() > 3000 {
            out.insert("position");
            return out;
        }
        let mut used = vec![false; unmatched_b.len()];
        for x in unmatched_a {
            let found = unmatched_b.iter().enumerate().position(|(i, y)| {
                !used[i]
                    && (0..3).any(|r| {
                        let turned = Tri {
                            k: [0, 1, 2].map(|c| y.k[(c + r) % 3].clone()),
                            look: y.look,
                        };
                        same_place(x, &turned, tol)
                    })
            });
            match found {
                Some(i) => {
                    used[i] = true;
                    let y = unmatched_b[i];
                    let r = (0..3)
                        .find(|&r| (0..3).all(|c| (0..3).all(|k| (x.k[c].p[k] - y.k[(c + r) % 3].p[k]).abs() <= tol)))
                        .unwrap();
                    let turned = Tri {
                        k: [0, 1, 2].map(|c| y.k[(c + r) % 3].clone()),
                        look: y.look,
                    };
                    if !places_only {
                        tri_diff(x, &turned, &la, &lb, &mut out);
                    }
                }
                None => {
                    out.insert("position");
                }
            }
        }
    }
    out
}

/// A rig's nodes' world matrices by name, for names only one node has.
fn worlds(rig: &polysmelt::rig::Rig, animation: Option<usize>, time: f32) -> HashMap<String, [f32; 16]> {
    let world = rig.world(&rig.pose(animation, time));
    let mut count: HashMap<&str, usize> = HashMap::new();
    for n in &rig.nodes {
        if let Some(name) = n.name.as_deref() {
            *count.entry(name).or_default() += 1;
        }
    }
    rig.nodes
        .iter()
        .zip(world)
        .filter_map(|(n, w)| {
            let name = n.name.as_deref()?;
            (count[name] == 1).then(|| (name.to_owned(), w))
        })
        .collect()
}

/// How close points of a still scene come back, relative to its size.
const STILL: f32 = 1e-4;

/// How close points of an animated scene come back, relative to its size: rotations are written as Euler curves
/// within 0.05 degrees of the quaternions, which moves a point at the scene's edge by up to 0.05 degrees' worth.
const MOVING: f32 = 1e-3;

fn matrix_close(a: &[f32; 16], b: &[f32; 16], tol: f32) -> bool {
    let scale = a[..12].iter().fold(1.0f32, |m, v| m.max(v.abs()));
    (0..12).all(|k| (a[k] - b[k]).abs() <= 3e-3 * scale) && (12..15).all(|k| (a[k] - b[k]).abs() <= tol)
}

/// Names of nodes holding a camera or light, which the file stores turned to face the way FBX's do.
fn turned(doc: &Document) -> BTreeSet<String> {
    let json = doc.json();
    json.get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|n| n.get("camera").is_some() || n.pointer("/extensions/KHR_lights_punctual").is_some())
        .filter_map(|n| n.get("name").and_then(Value::as_str).map(str::to_owned))
        .collect()
}

/// The times to compare an animation at: its keys, and evenly between, at most `most`.
fn times(a: &polysmelt::rig::Animation, most: usize) -> Vec<f32> {
    let (first, last) = a.span();
    if !first.is_finite() || !last.is_finite() {
        return vec![0.0];
    }
    let mut t: Vec<f32> = a.key_times();
    if t.len() > most / 2 {
        let step = t.len().div_ceil(most / 2);
        t = t.into_iter().step_by(step).collect();
    }
    let n = most - t.len();
    t.extend((0..n).map(|k| first + (last - first) * k as f32 / (n.max(2) - 1) as f32));
    t.push(last);
    t.sort_by(f32::total_cmp);
    t.dedup();
    t
}

/// Which ways two documents' rigs and animations differ: the number of clips, then each clip's nodes' world
/// transforms, and the posed, skinned and morphed meshes, over time.
fn motion_diff(a: &Document, b: &Document, most: usize, mesh_times: usize) -> BTreeSet<&'static str> {
    let mut out = BTreeSet::new();
    let (Ok(ra), Ok(rb)) = (a.rig(), b.rig()) else {
        out.insert("rig");
        return out;
    };
    let skip = turned(a);
    let size = a.flatten(Pose::Rest).ok().flatten().map_or(1.0, |m| extent(&m));
    let tol = size.max(rig_extent(&ra)) * MOVING;
    // A clip that holds every node at rest is written, but read back as nothing to play: the clips compared are
    // the ones that move something.
    let rest = worlds(&ra, None, 0.0);
    let moving: Vec<usize> = (0..ra.animations.len())
        .filter(|&k| {
            ra.animations.len() == rb.animations.len()
                || times(&ra.animations[k], most).iter().any(|&t| {
                    worlds(&ra, Some(k), t)
                        .iter()
                        .any(|(name, m)| rest.get(name).is_some_and(|r| !matrix_close(m, r, 1e-6)))
                })
        })
        .collect();
    if moving.len() != rb.animations.len() {
        out.insert("animation count");
        return out;
    }
    for (j, &k) in moving.iter().enumerate() {
        let anim = &ra.animations[k];
        let ts = times(anim, most);
        for &t in &ts {
            let (wa, wb) = (worlds(&ra, Some(k), t), worlds(&rb, Some(j), t));
            for (name, m) in &wa {
                if skip.contains(name) {
                    continue;
                }
                if let Some(n) = wb.get(name)
                    && !matrix_close(m, n, tol.max(1e-4))
                {
                    out.insert("animated nodes");
                }
            }
        }
        let step = (ts.len() / mesh_times.max(1)).max(1);
        for &t in ts.iter().step_by(step) {
            let pose = Pose::Animation { index: k, time: t };
            let back = Pose::Animation { index: j, time: t };
            match (a.flatten(pose), b.flatten(back)) {
                (Ok(Some(x)), Ok(Some(y))) => {
                    if !mesh_diff(&x, &y, true, MOVING).is_empty() {
                        out.insert("animated mesh");
                    }
                }
                (Ok(None), Ok(None)) => {}
                _ => {
                    out.insert("animated mesh");
                }
            }
        }
    }
    out
}

/// Which ways a document and what came back from writing it differ: the rest and bind poses, and its motion.
fn diff(a: &Document, b: &Document) -> BTreeSet<&'static str> {
    let mut out = BTreeSet::new();
    for pose in [Pose::Rest, Pose::Bind] {
        match (a.flatten(pose), b.flatten(pose)) {
            (Ok(Some(x)), Ok(Some(y))) => {
                let d = mesh_diff(&x, &y, pose == Pose::Bind, STILL);
                if pose == Pose::Bind {
                    if !d.is_empty() {
                        out.insert("bind pose");
                    }
                } else {
                    out.extend(d);
                }
            }
            (Ok(None), Ok(None)) => {}
            (Ok(Some(x)), Ok(None)) if x.prims.iter().all(|p| p.indices.is_empty() && p.lines.is_empty()) => {}
            _ => {
                out.insert("mesh");
            }
        }
    }
    out.extend(motion_diff(a, b, 24, 4));
    out
}

fn node_named<'a>(doc: &'a Document, name: &str) -> Option<(usize, &'a Value)> {
    doc.json()
        .get("nodes")?
        .as_array()?
        .iter()
        .enumerate()
        .find(|(_, n)| n.get("name").and_then(Value::as_str) == Some(name))
}

fn num(v: &Value, pointer: &str) -> f64 {
    v.pointer(pointer)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("no {pointer} in {v}"))
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

// ---------------------------------------------------------------------------------------------------------------
// Scenes.

/// A quad in the XY plane, `size` across, with normals, two UV sets and two color sets.
fn quad(g: &mut Glb, size: f32, material: usize) -> Value {
    let s = size / 2.0;
    let pos = g.floats(&[-s, -s, 0.0, s, -s, 0.0, s, s, 0.0, -s, s, 0.0], "VEC3");
    let nrm = g.floats(&[0.0, 0.0, 1.0].repeat(4), "VEC3");
    let uv0 = g.floats(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], "VEC2");
    let uv1 = g.floats(&[0.5, 0.5, 0.75, 0.5, 0.75, 0.25, 0.5, 0.25], "VEC2");
    let c0 = g.floats(
        &[
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0,
        ],
        "VEC4",
    );
    let c1 = g.floats(&[0.25, 0.5, 0.75, 1.0].repeat(4), "VEC4");
    let idx = g.u16s(&[0, 1, 2, 0, 2, 3], "SCALAR");
    json!({
        "attributes": {"POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": uv0, "TEXCOORD_1": uv1,
                       "COLOR_0": c0, "COLOR_1": c1},
        "indices": idx, "material": material
    })
}

/// Two materials, one textured and one metal; a mesh of two primitives drawn by three nodes, one placed by a
/// matrix; a mesh of lines.
fn static_scene() -> Document {
    let mut g = Glb::default();
    let quad_prim = quad(&mut g, 2.0, 0);
    let tri_pos = g.floats(&[0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 0.5], "VEC3");
    let tri_nrm = g.floats(&[0.0, 0.0, 1.0].repeat(3), "VEC3");
    let line_pos = g.floats(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], "VEC3");
    let line_idx = g.u16s(&[0, 1, 1, 2], "SCALAR");
    let img0 = g.view(&png(1));
    let img1 = g.view(&png(2));
    g.build(json!({
        "scene": 0,
        "scenes": [{"nodes": [0, 5]}],
        "nodes": [
            {"name": "Root", "children": [1, 2, 3, 4], "translation": [0.0, 1.0, 0.0]},
            {"name": "Left", "mesh": 0, "translation": [-3.0, 0.0, 0.0]},
            {"name": "Right", "mesh": 0, "translation": [3.0, 0.0, 0.5],
             "rotation": quat([0.0, 0.6, 0.8], 70.0), "scale": [1.0, 2.0, 0.5]},
            {"name": "Placed", "mesh": 0, "matrix": [
                0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -4.0, 1.0, 1.0]},
            {"name": "Wire", "mesh": 1, "translation": [0.0, 3.0, 0.0]},
            {"name": "Empty", "translation": [5.0, 5.0, 5.0],
             "extras": {"note": "hello", "count": 3, "ratio": 0.5}}
        ],
        "meshes": [
            {"name": "Panel", "primitives": [
                quad_prim,
                {"attributes": {"POSITION": tri_pos, "NORMAL": tri_nrm}, "material": 1}
            ]},
            {"name": "Lines", "primitives": [{"attributes": {"POSITION": line_pos}, "indices": line_idx, "mode": 1}]}
        ],
        "materials": [
            {"name": "Painted", "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.6, 0.4, 1.0],
                "baseColorTexture": {"index": 0, "texCoord": 0,
                    "extensions": {"KHR_texture_transform": {"offset": [0.25, 0.5], "scale": [2.0, 3.0]}}},
                "metallicFactor": 0.0, "roughnessFactor": 0.9},
             "normalTexture": {"index": 1, "texCoord": 1},
             "emissiveFactor": [0.0, 0.0, 0.0]},
            {"name": "Metal", "pbrMetallicRoughness": {
                "baseColorFactor": [0.9, 0.9, 1.0, 1.0], "metallicFactor": 0.7, "roughnessFactor": 0.2},
             "emissiveFactor": [0.5, 0.25, 0.0], "doubleSided": true, "alphaMode": "MASK", "alphaCutoff": 0.3}
        ],
        "textures": [{"source": 0, "sampler": 0}, {"source": 1, "sampler": 1}],
        "samplers": [{"wrapS": 10497, "wrapT": 33071}, {"wrapS": 33648, "wrapT": 10497}],
        "images": [{"bufferView": img0, "mimeType": "image/png"}, {"bufferView": img1, "mimeType": "image/png"}]
    }))
}

/// Column-major translation.
fn translation(t: [f32; 3]) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, t[0], t[1], t[2], 1.0,
    ]
}

/// A column of quads skinned to a chain of three bones, with three clips: one linear and turning far, one stepped,
/// one cubic.
fn skinned_scene() -> Document {
    let mut g = Glb::default();
    let (mut pos, mut nrm, mut joints, mut weights, mut idx) = (vec![], vec![], vec![], vec![], vec![]);
    let rows = 7;
    for r in 0..rows {
        let y = r as f32 * 0.5;
        for x in [-0.25f32, 0.25] {
            pos.extend_from_slice(&[x, y, 0.0]);
            nrm.extend_from_slice(&[0.0, 0.0, 1.0]);
            // Blend between the bones at 0, 1 and 2 units up.
            let b = (y / 1.0).min(2.0);
            let j = (b.floor() as u16).min(1);
            let w = (b - f32::from(j)).clamp(0.0, 1.0);
            joints.extend_from_slice(&[j, j + 1, 0, 0]);
            weights.extend_from_slice(&[1.0 - w, w, 0.0, 0.0]);
        }
    }
    for r in 0..rows - 1 {
        let a = (r * 2) as u16;
        idx.extend_from_slice(&[a, a + 1, a + 3, a, a + 3, a + 2]);
    }
    let pos = g.floats(&pos, "VEC3");
    let nrm = g.floats(&nrm, "VEC3");
    let joints = g.u16s(&joints, "VEC4");
    let weights = g.floats(&weights, "VEC4");
    let idx = g.u16s(&idx, "SCALAR");
    let ibm: Vec<f32> = [0.0, 1.0, 2.0]
        .iter()
        .flat_map(|&y| translation([0.0, -y - 0.5, 0.0]))
        .collect();
    let ibm = g.floats(&ibm, "MAT4");
    // Linear: the spine turns all the way round, past where quaternions change sign.
    let walk_t = g.floats(&[0.0, 0.5, 1.0, 1.5, 2.0], "SCALAR");
    let spin: Vec<f32> = [0.0, 100.0, 200.0, 300.0, 359.0]
        .iter()
        .flat_map(|&d| quat([0.0, 0.0, 1.0], d))
        .collect();
    let spin = g.floats(&spin, "VEC4");
    let tilt: Vec<f32> = [0.0, 45.0, -60.0, 120.0, 10.0]
        .iter()
        .flat_map(|&d| quat([0.6, 0.0, 0.8], d))
        .collect();
    let tilt = g.floats(&tilt, "VEC4");
    let hips = g.floats(
        &[
            0.0, 0.5, 0.0, 0.3, 0.6, 0.0, 0.0, 0.5, 0.2, -0.3, 0.4, 0.0, 0.0, 0.5, 0.0,
        ],
        "VEC3",
    );
    // Stepped: the arm snaps between poses, and scales.
    let wave_t = g.floats(&[0.0, 0.4, 0.8, 1.2], "SCALAR");
    let wave: Vec<f32> = [0.0, 80.0, -80.0, 30.0]
        .iter()
        .flat_map(|&d| quat([1.0, 0.0, 0.0], d))
        .collect();
    let wave = g.floats(&wave, "VEC4");
    let grow = g.floats(&[1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0], "VEC3");
    // Cubic: in-tangent, value, out-tangent for each key.
    let bend_t = g.floats(&[0.0, 1.0, 2.0], "SCALAR");
    let mut bend = Vec::new();
    for (d, tangent) in [(0.0f32, 0.0f32), (90.0, 0.4), (30.0, -0.2)] {
        let q = quat([1.0, 0.0, 0.0], d);
        let tan = [tangent, 0.0, 0.0, 0.0];
        bend.extend_from_slice(&tan);
        bend.extend_from_slice(&q);
        bend.extend_from_slice(&tan);
    }
    let bend = g.floats(&bend, "VEC4");
    let slide = g.floats(
        &[
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, //
            0.5, 0.0, 0.0, 0.2, 0.8, 0.0, -0.5, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0,
        ],
        "VEC3",
    );
    g.build(json!({
        "scene": 0,
        "scenes": [{"nodes": [0, 3]}],
        "nodes": [
            {"name": "Hips", "translation": [0.0, 0.5, 0.0], "children": [1]},
            {"name": "Spine", "translation": [0.0, 1.0, 0.0], "children": [2]},
            {"name": "Arm", "translation": [0.0, 1.0, 0.0]},
            {"name": "Body", "mesh": 0, "skin": 0}
        ],
        "meshes": [{"name": "Column", "primitives": [{"attributes": {
            "POSITION": pos, "NORMAL": nrm, "JOINTS_0": joints, "WEIGHTS_0": weights}, "indices": idx}]}],
        "skins": [{"joints": [0, 1, 2], "inverseBindMatrices": ibm, "skeleton": 0}],
        "animations": [
            {"name": "Walk", "samplers": [
                {"input": walk_t, "output": spin},
                {"input": walk_t, "output": tilt},
                {"input": walk_t, "output": hips}
             ], "channels": [
                {"sampler": 0, "target": {"node": 1, "path": "rotation"}},
                {"sampler": 1, "target": {"node": 2, "path": "rotation"}},
                {"sampler": 2, "target": {"node": 0, "path": "translation"}}
             ]},
            {"name": "Wave", "samplers": [
                {"input": wave_t, "output": wave, "interpolation": "STEP"},
                {"input": wave_t, "output": grow, "interpolation": "STEP"}
             ], "channels": [
                {"sampler": 0, "target": {"node": 2, "path": "rotation"}},
                {"sampler": 1, "target": {"node": 2, "path": "scale"}}
             ]},
            {"name": "Bend", "samplers": [
                {"input": bend_t, "output": bend, "interpolation": "CUBICSPLINE"},
                {"input": bend_t, "output": slide, "interpolation": "CUBICSPLINE"}
             ], "channels": [
                {"sampler": 0, "target": {"node": 1, "path": "rotation"}},
                {"sampler": 1, "target": {"node": 0, "path": "translation"}}
             ]}
        ]
    }))
}

/// A quad with two morph targets, one moving normals too, and two clips of their weights.
/// A bone and an empty node, both skin joints, and a strip whose far end is on the empty node alone: what this
/// crate's reader makes of a skinned mesh whose model is placed (and its geometry offset) away from the origin.
fn mesh_model_scene() -> Document {
    let mut g = Glb::default();
    let pos = g.floats(
        &[
            -0.2, 0.0, 0.0, 0.2, 0.0, 0.0, -0.2, 1.0, 0.0, 0.2, 1.0, 0.0, -0.2, 2.0, 0.0, 0.2, 2.0, 0.0,
        ],
        "VEC3",
    );
    let joints = g.u16s(
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "VEC4",
    );
    let weights = g.floats(
        &[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
        ],
        "VEC4",
    );
    let idx = g.u16s(&[0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4], "SCALAR");
    // The empty node's inverse bind matrix isn't the inverse of where it is: the geometry sits offset in it.
    let mut ibm: Vec<f32> = translation([0.5, -1.0, 0.0]).to_vec();
    ibm.extend_from_slice(&[
        0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0, 0.25, 3.0, 1.0,
    ]);
    let ibm = g.floats(&ibm, "MAT4");
    let t = g.floats(&[0.0, 1.0], "SCALAR");
    let turn: Vec<f32> = [0.0, 60.0].iter().flat_map(|&d| quat([0.0, 0.0, 1.0], d)).collect();
    let turn = g.floats(&turn, "VEC4");
    let slide = g.floats(&[0.0, 1.0, 0.0, 0.5, 1.5, 0.0], "VEC3");
    g.build(json!({
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2]}],
        "nodes": [
            {"name": "Bone", "translation": [-0.5, 1.0, 0.0]},
            {"name": "Body", "translation": [0.0, 1.0, 0.0], "rotation": quat([0.0, 0.0, 1.0], 45.0)},
            {"name": "Body", "mesh": 0, "skin": 0}
        ],
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": pos, "JOINTS_0": joints, "WEIGHTS_0": weights}, "indices": idx}]}],
        "skins": [{"joints": [0, 1], "inverseBindMatrices": ibm}],
        "animations": [{"name": "Sway", "samplers": [
            {"input": t, "output": turn}, {"input": t, "output": slide}], "channels": [
            {"sampler": 0, "target": {"node": 0, "path": "rotation"}},
            {"sampler": 1, "target": {"node": 1, "path": "translation"}}]}]
    }))
}

fn morph_scene() -> Document {
    let mut g = Glb::default();
    let prim = quad(&mut g, 2.0, 0);
    let smile = g.floats(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0, 0.5, 0.2], "VEC3");
    let blink = g.floats(&[0.0, -0.3, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "VEC3");
    let blink_n = g.floats(
        &[0.0, 0.2, -0.02, 0.0, 0.2, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "VEC3",
    );
    let mut prim = prim;
    prim["targets"] = json!([{"POSITION": smile}, {"POSITION": blink, "NORMAL": blink_n}]);
    let talk_t = g.floats(&[0.0, 0.5, 1.0, 2.0], "SCALAR");
    let talk = g.floats(&[0.0, 0.0, 1.0, 0.0, 0.3, 1.0, 0.0, 0.0], "SCALAR");
    let blink_t = g.floats(&[0.0, 0.25, 0.5], "SCALAR");
    let blink_w = g.floats(&[0.0, 1.0, 1.0, 0.0, 0.0, 0.5], "SCALAR");
    g.build(json!({
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Face", "mesh": 0, "translation": [0.0, 1.5, 0.0]}],
        "meshes": [{"name": "Face", "primitives": [prim], "weights": [0.25, 0.0],
                    "extras": {"targetNames": ["Smile", "Blink"]}}],
        "materials": [{"name": "Skin", "pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.7, 0.6, 1.0]}}],
        "animations": [
            {"name": "Talk", "samplers": [{"input": talk_t, "output": talk}],
             "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}]},
            {"name": "Blink", "samplers": [{"input": blink_t, "output": blink_w, "interpolation": "STEP"}],
             "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}]}
        ]
    }))
}

/// Two cameras, three lights, a mesh under a light, and extras.
fn staging_scene() -> Document {
    let mut g = Glb::default();
    let prim = quad(&mut g, 0.5, 0);
    g.build(json!({
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2, 3, 5]}],
        "extensionsUsed": ["KHR_lights_punctual"],
        "extensions": {"KHR_lights_punctual": {"lights": [
            {"name": "Sun", "type": "directional", "color": [1.0, 0.9, 0.8], "intensity": 3.0},
            {"name": "Cone", "type": "spot", "color": [0.2, 0.4, 1.0], "intensity": 50.0,
             "spot": {"innerConeAngle": 0.2, "outerConeAngle": 0.5}},
            {"name": "Bulb", "type": "point", "intensity": 20.0}
        ]}},
        "nodes": [
            {"name": "Cam", "camera": 0, "translation": [0.0, 2.0, 8.0], "rotation": quat([1.0, 0.0, 0.0], -10.0),
             "extras": {"lens": "wide", "stops": 4}},
            {"name": "Top", "camera": 1, "translation": [0.0, 10.0, 0.0], "rotation": quat([1.0, 0.0, 0.0], -90.0)},
            {"name": "Sun", "translation": [0.0, 5.0, 0.0], "rotation": quat([0.6, 0.8, 0.0], -45.0),
             "extensions": {"KHR_lights_punctual": {"light": 0}}},
            {"name": "Cone", "translation": [2.0, 3.0, 1.0], "rotation": quat([0.0, 1.0, 0.0], 30.0),
             "extensions": {"KHR_lights_punctual": {"light": 1}}, "children": [4]},
            {"name": "Shade", "mesh": 0, "translation": [0.0, 0.0, -1.0]},
            {"name": "Lamp", "translation": [-2.0, 1.0, 0.0], "mesh": 0,
             "extensions": {"KHR_lights_punctual": {"light": 2}}}
        ],
        "cameras": [
            {"name": "Cam", "type": "perspective",
             "perspective": {"yfov": 0.7, "aspectRatio": 1.5, "znear": 0.1, "zfar": 200.0},
             "extras": {"lens": "wide"}},
            {"name": "Top", "type": "orthographic",
             "orthographic": {"xmag": 4.0, "ymag": 2.0, "znear": 0.01, "zfar": 50.0}}
        ],
        "meshes": [{"name": "Shade", "primitives": [prim]}],
        "materials": [{"name": "Paper"}]
    }))
}

// ---------------------------------------------------------------------------------------------------------------
// Tests.

#[test]
fn static_meshes_materials_and_textures_come_back() {
    let doc = static_scene();
    let (_, back) = round_trip(&doc, "static");
    let d = diff(&doc, &back);
    assert!(d.is_empty(), "differs in {d:?}");
    // Each mesh is one geometry, however many models draw it.
    let geometries = back.json()["meshes"].as_array().unwrap().len();
    assert!(geometries <= 2, "{geometries} meshes");
    // The node tree and names.
    let rig_a = doc.rig().unwrap();
    let rig_b = back.rig().unwrap();
    for name in ["Root", "Left", "Right", "Placed", "Wire", "Empty"] {
        let (a, b) = (rig_a.find(name).unwrap(), rig_b.find(name).expect(name));
        let pa = rig_a.nodes[a].parent.and_then(|p| rig_a.nodes[p].name.clone());
        let pb = rig_b.nodes[b].parent.and_then(|p| rig_b.nodes[p].name.clone());
        assert_eq!(pa, pb, "{name}'s parent");
    }
    let (wa, wb) = (worlds(&rig_a, None, 0.0), worlds(&rig_b, None, 0.0));
    for (name, m) in &wa {
        assert!(matrix_close(m, &wb[name], 1e-4), "{name}: {m:?} vs {:?}", wb[name]);
    }
    // Extras as user properties.
    let (_, empty) = node_named(&back, "Empty").unwrap();
    assert_eq!(empty.pointer("/extras/note").and_then(Value::as_str), Some("hello"));
    assert_eq!(empty.pointer("/extras/count").and_then(Value::as_f64), Some(3.0));
    assert_eq!(empty.pointer("/extras/ratio").and_then(Value::as_f64), Some(0.5));
    // Material settings beyond what the flattened mesh shows.
    let mats = back.json()["materials"].as_array().unwrap();
    let metal = mats.iter().find(|m| m["name"] == "Metal").expect("the metal material");
    assert_eq!(metal["alphaMode"], "MASK");
    assert_eq!(metal["doubleSided"], true);
    let painted = mats.iter().find(|m| m["name"] == "Painted").unwrap();
    assert!(painted.pointer("/normalTexture/index").is_some(), "{painted}");
    assert_eq!(
        painted.pointer("/normalTexture/texCoord").and_then(Value::as_u64),
        Some(1)
    );
    let tex = &back.json()["textures"][painted
        .pointer("/pbrMetallicRoughness/baseColorTexture/index")
        .and_then(Value::as_u64)
        .unwrap() as usize];
    let sampler = &back.json()["samplers"][tex["sampler"].as_u64().unwrap() as usize];
    assert_eq!(sampler["wrapT"], 33071, "{sampler}");
}

#[test]
fn skins_bind_poses_and_clips_come_back() {
    let doc = skinned_scene();
    let (_, back) = round_trip(&doc, "skinned");
    let d = diff(&doc, &back);
    assert!(d.is_empty(), "differs in {d:?}");
    let rig = back.rig().unwrap();
    assert_eq!(rig.skins.len(), 1);
    // A joint per cluster, then the reader's own joint for the mesh's node, standing for unweighted vertices.
    let joints: Vec<_> = rig.skins[0]
        .joints
        .iter()
        .map(|&j| rig.nodes[j].name.as_deref())
        .collect();
    assert_eq!(joints, [Some("Hips"), Some("Spine"), Some("Arm"), Some("Body")]);
    let names: Vec<_> = rig
        .animations
        .iter()
        .map(|a| a.name.clone().unwrap_or_default())
        .collect();
    assert_eq!(names, ["Walk", "Wave", "Bend"]);
    // Bone world poses, finely sampled.
    let src = doc.rig().unwrap();
    for (k, anim) in src.animations.iter().enumerate() {
        let (first, last) = anim.span();
        for s in 0..=200 {
            let t = first + (last - first) * s as f32 / 200.0;
            let (wa, wb) = (worlds(&src, Some(k), t), worlds(&rig, Some(k), t));
            for bone in ["Hips", "Spine", "Arm"] {
                // A stepped clip jumps at its keys: compare just after rather than on them.
                if names[k] == "Wave" && anim.key_times().iter().any(|&kt| (kt - t).abs() < 1e-3) {
                    continue;
                }
                assert!(
                    matrix_close(&wa[bone], &wb[bone], MOVING * 4.0),
                    "{} {bone} at {t}: {:?} vs {:?}",
                    names[k],
                    wa[bone],
                    wb[bone]
                );
            }
        }
    }
}

/// The names of a document's nodes and its first skin's joints.
fn layout(doc: &Document) -> (Vec<String>, Vec<String>) {
    let json = doc.json();
    let names: Vec<String> = json["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.get("name").and_then(Value::as_str).unwrap_or("").to_owned())
        .collect();
    let joints = json["skins"][0]["joints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| names[j.as_u64().unwrap() as usize].clone())
        .collect();
    (names, joints)
}

#[test]
fn a_read_skinned_mesh_writes_back_as_one_model() {
    // The reader keeps the mesh's model as a joint and draws the mesh from a root node of its own: written again,
    // the two are one model, so reading and writing over and over neither adds nodes nor joints.
    let (_, once) = round_trip(&skinned_scene(), "skinned");
    let (_, twice) = round_trip(&once, "skinned_again");
    assert_eq!(layout(&once), layout(&twice));
    let d = diff(&once, &twice);
    assert!(d.is_empty(), "differs in {d:?}");
    let (_, thrice) = round_trip(&twice, "skinned_thrice");
    assert_eq!(layout(&twice), layout(&thrice));
}

#[test]
fn a_mesh_folded_onto_its_joint_keeps_its_geometric_offset() {
    let doc = mesh_model_scene();
    let (_, back) = round_trip(&doc, "mesh_model");
    assert_eq!(layout(&doc), layout(&back));
    let d = diff(&doc, &back);
    assert!(d.is_empty(), "differs in {d:?}");
}

#[test]
fn morph_targets_and_their_weights_come_back() {
    let doc = morph_scene();
    let (_, back) = round_trip(&doc, "morphs");
    let d = diff(&doc, &back);
    assert!(d.is_empty(), "differs in {d:?}");
    let mesh = &back.json()["meshes"][0];
    let weights: Vec<f64> = mesh["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_f64().unwrap())
        .collect();
    assert!(close(weights[0], 0.25, 1e-5) && weights[1] == 0.0, "{weights:?}");
    let names = mesh.pointer("/extras/targetNames").unwrap();
    assert_eq!(names, &json!(["Smile", "Blink"]));
    let rig = back.rig().unwrap();
    assert_eq!(rig.animations.len(), 2);
    // Sampled finely: the posed mesh follows the weights.
    for k in 0..2 {
        let (first, last) = doc.rig().unwrap().animations[k].span();
        for s in 0..=40 {
            let t = first + (last - first) * (s as f32 + 0.5) / 41.0;
            let pose = Pose::Animation { index: k, time: t };
            let (a, b) = (
                doc.flatten(pose).unwrap().unwrap(),
                back.flatten(pose).unwrap().unwrap(),
            );
            assert!(mesh_diff(&a, &b, true, MOVING).is_empty(), "clip {k} at {t}");
        }
    }
}

/// Three materials that between them use everything the writer carries: specular and glossiness in place of
/// metal and roughness; blending with every extension and a texture on every slot; and an unlit one.
fn material_scene() -> Document {
    let mut g = Glb::default();
    let prims = [quad(&mut g, 2.0, 0), quad(&mut g, 1.0, 1), quad(&mut g, 0.5, 2)];
    let images: Vec<Value> = (0..3)
        .map(|k| json!({"bufferView": g.view(&png(10 + k)), "mimeType": "image/png"}))
        .collect();
    let tex = |index: usize| json!({"index": index});
    g.build(json!({
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Panel", "mesh": 0}],
        "meshes": [{"name": "Panel", "primitives": prims}],
        "materials": [
            {"name": "Glossy", "extensions": {"KHR_materials_pbrSpecularGlossiness": {
                "diffuseFactor": [0.5, 0.25, 0.125, 1.0], "specularFactor": [0.25, 0.5, 0.75],
                "glossinessFactor": 0.75, "diffuseTexture": tex(0)}}},
            {"name": "Glass",
             "pbrMetallicRoughness": {
                "baseColorFactor": [0.25, 0.5, 0.75, 0.5], "baseColorTexture": tex(0),
                "metallicFactor": 0.25, "roughnessFactor": 0.5, "metallicRoughnessTexture": tex(1)},
             "normalTexture": {"index": 2, "scale": 0.5},
             "occlusionTexture": {"index": 1, "strength": 0.75},
             "emissiveTexture": tex(0),
             "emissiveFactor": [1.0, 0.5, 0.25],
             "alphaMode": "BLEND",
             "doubleSided": true,
             "extensions": {
                "KHR_materials_emissive_strength": {"emissiveStrength": 4.0},
                "KHR_materials_ior": {"ior": 1.25},
                "KHR_materials_transmission": {"transmissionFactor": 0.75, "transmissionTexture": tex(1)},
                "KHR_materials_clearcoat": {
                    "clearcoatFactor": 0.5, "clearcoatRoughnessFactor": 0.25, "clearcoatTexture": tex(1),
                    "clearcoatRoughnessTexture": tex(1), "clearcoatNormalTexture": tex(2)},
                "KHR_materials_sheen": {
                    "sheenColorFactor": [0.125, 0.25, 0.5], "sheenRoughnessFactor": 0.375,
                    "sheenColorTexture": tex(0), "sheenRoughnessTexture": tex(1)},
                "KHR_materials_specular": {
                    "specularFactor": 0.875, "specularColorFactor": [1.0, 0.5, 0.5],
                    "specularTexture": tex(1), "specularColorTexture": tex(0)},
                "KHR_materials_volume": {
                    "thicknessFactor": 2.0, "attenuationDistance": 5.0, "attenuationColor": [0.75, 0.5, 0.25],
                    "thicknessTexture": tex(1)},
                "KHR_materials_anisotropy": {
                    "anisotropyStrength": 0.25, "anisotropyRotation": 1.25, "anisotropyTexture": tex(2)},
                "KHR_materials_iridescence": {
                    "iridescenceFactor": 0.5, "iridescenceIor": 1.5, "iridescenceThicknessMinimum": 150.0,
                    "iridescenceThicknessMaximum": 500.0, "iridescenceTexture": tex(1),
                    "iridescenceThicknessTexture": tex(1)},
                "KHR_materials_dispersion": {"dispersion": 0.25}
             }},
            {"name": "Flat", "pbrMetallicRoughness": {"baseColorFactor": [0.0, 1.0, 0.0, 1.0]},
             "extensions": {"KHR_materials_unlit": {}}}
        ],
        "textures": [{"source": 0}, {"source": 1}, {"source": 2}],
        "images": images
    }))
}

/// Every number under `at` in `want` is the same number in `got`, and every texture there names the same image.
/// What differs is added to `out` by its path.
fn same_values(doc: (&Document, &Document), want: &Value, got: Option<&Value>, at: &str, out: &mut Vec<String>) {
    let image = |d: &Document, info: &Value| -> Option<u64> {
        let t = info.get("index")?.as_u64()?;
        let source = d.json()["textures"][t as usize]["source"].as_u64()?;
        let image = &d.json()["images"][source as usize];
        let view = &d.json()["bufferViews"][image["bufferView"].as_u64()? as usize];
        let start = view.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let bytes = d.buffer(view["buffer"].as_u64()? as usize).ok()?;
        Some(hash(&bytes[start..start + view["byteLength"].as_u64()? as usize]))
    };
    match want {
        Value::Object(o) if at.ends_with("Texture") => {
            let same = got.is_some_and(|g| image(doc.0, want).is_some() && image(doc.0, want) == image(doc.1, g));
            if !same {
                out.push(format!("{at}: the image"));
            }
            for key in ["scale", "strength"] {
                if let Some(w) = o.get(key) {
                    same_values(doc, w, got.and_then(|g| g.get(key)), &format!("{at}/{key}"), out);
                }
            }
        }
        Value::Object(o) => {
            for (k, w) in o {
                same_values(doc, w, got.and_then(|g| g.get(k)), &format!("{at}/{k}"), out);
            }
        }
        Value::Array(a) => {
            for (k, w) in a.iter().enumerate() {
                same_values(doc, w, got.and_then(|g| g.get(k)), &format!("{at}/{k}"), out);
            }
        }
        Value::Number(w) => {
            let same = got
                .and_then(Value::as_f64)
                .is_some_and(|g| close(w.as_f64().unwrap(), g, 1e-5));
            if !same {
                out.push(format!("{at}: {w}, came back {got:?}"));
            }
        }
        other => {
            if got != Some(other) {
                out.push(format!("{at}: {other}, came back {got:?}"));
            }
        }
    }
}

#[test]
fn material_extensions_and_their_textures_come_back() {
    let doc = material_scene();
    let (_, back) = round_trip(&doc, "materials");
    let material = |d: &Document, name: &str| -> Value {
        let all = d.json()["materials"].as_array().unwrap();
        all.iter()
            .find(|m| m["name"] == name)
            .unwrap_or_else(|| panic!("no {name}"))
            .clone()
    };
    let mut wrong = Vec::new();
    // Blending, and every extension's numbers and textures, exactly as they went in.
    let (want, got) = (material(&doc, "Glass"), material(&back, "Glass"));
    same_values((&doc, &back), &want, Some(&got), "Glass", &mut wrong);
    let (want, got) = (material(&doc, "Flat"), material(&back, "Flat"));
    same_values((&doc, &back), &want, Some(&got), "Flat", &mut wrong);
    // Specular and glossiness come back as metal and roughness: the diffuse color and texture as the base color's,
    // no metal, and roughness one less the glossiness.
    let glossy = material(&back, "Glossy");
    let want = json!({"pbrMetallicRoughness": {
        "baseColorFactor": [0.5, 0.25, 0.125, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.25,
        "baseColorTexture": {"index": 0}}});
    same_values((&doc, &back), &want, Some(&glossy), "Glossy", &mut wrong);
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn cameras_lights_and_extras_come_back() {
    let doc = staging_scene();
    let (_, back) = round_trip(&doc, "staging");
    let d = diff(&doc, &back);
    assert!(d.is_empty(), "differs in {d:?}");
    let json = back.json();
    let rig_a = doc.rig().unwrap();
    let rig_b = back.rig().unwrap();
    let world_a = rig_a.world(&rig_a.pose(None, 0.0));
    let world_b = rig_b.world(&rig_b.pose(None, 0.0));
    // The node holding each camera and light stands where the source's did.
    let holder = |what: &str, index: usize| -> usize {
        json["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .position(|n| match what {
                "camera" => n["camera"].as_u64() == Some(index as u64),
                _ => {
                    n.pointer("/extensions/KHR_lights_punctual/light")
                        .and_then(Value::as_u64)
                        == Some(index as u64)
                }
            })
            .unwrap_or_else(|| panic!("no node holds {what} {index}"))
    };
    for (name, what, index) in [
        ("Cam", "camera", 0),
        ("Top", "camera", 1),
        ("Sun", "light", 0),
        ("Cone", "light", 1),
        ("Lamp", "light", 2),
    ] {
        let a = node_named(&doc, name).unwrap().0;
        let b = holder(what, index);
        assert!(
            matrix_close(&world_a[a], &world_b[b], 1e-4),
            "{name}: {:?} vs {:?}",
            world_a[a],
            world_b[b]
        );
    }
    let cams = json["cameras"].as_array().unwrap();
    let persp = &cams[json["nodes"][holder("camera", 0)]["camera"].as_u64().unwrap() as usize];
    assert!(close(num(persp, "/perspective/yfov"), 0.7, 1e-5), "{persp}");
    assert!(close(num(persp, "/perspective/aspectRatio"), 1.5, 1e-5), "{persp}");
    assert!(close(num(persp, "/perspective/znear"), 0.1, 1e-5), "{persp}");
    assert!(close(num(persp, "/perspective/zfar"), 200.0, 1e-5), "{persp}");
    let ortho = &cams[1];
    assert_eq!(ortho["type"], "orthographic");
    assert!(close(num(ortho, "/orthographic/xmag"), 4.0, 1e-5), "{ortho}");
    assert!(close(num(ortho, "/orthographic/ymag"), 2.0, 1e-5), "{ortho}");
    assert!(close(num(ortho, "/orthographic/znear"), 0.01, 1e-5), "{ortho}");
    assert!(close(num(ortho, "/orthographic/zfar"), 50.0, 1e-5), "{ortho}");
    let lights = json["extensions"].pointer("/KHR_lights_punctual/lights").unwrap();
    let sun = &lights[0];
    assert_eq!(sun["type"], "directional");
    assert!(close(num(sun, "/intensity"), 3.0, 1e-5), "{sun}");
    for (k, want) in [1.0, 0.9, 0.8].iter().enumerate() {
        assert!(close(num(sun, &format!("/color/{k}")), *want, 1e-3), "{sun}");
    }
    let cone = &lights[1];
    assert_eq!(cone["type"], "spot");
    assert!(close(num(cone, "/spot/innerConeAngle"), 0.2, 1e-5), "{cone}");
    assert!(close(num(cone, "/spot/outerConeAngle"), 0.5, 1e-5), "{cone}");
    assert!(close(num(cone, "/intensity"), 50.0, 1e-5), "{cone}");
    assert_eq!(lights[2]["type"], "point");
    // The mesh under the spot stays put.
    let (shade_a, shade_b) = (rig_a.find("Shade").unwrap(), rig_b.find("Shade").unwrap());
    assert!(matrix_close(&world_a[shade_a], &world_b[shade_b], 1e-4));
    // Extras on a node, as user properties.
    let (_, cam) = node_named(&back, "Cam").unwrap();
    assert_eq!(
        cam.pointer("/extras/lens").and_then(Value::as_str),
        Some("wide"),
        "{cam}"
    );
    assert_eq!(cam.pointer("/extras/stops").and_then(Value::as_f64), Some(4.0), "{cam}");
}

/// Four grids, each big enough on its own that writing deflates them across threads, on nodes a long clip turns.
fn heavy_scene() -> Document {
    const SIDE: usize = 120;
    const KEYS: usize = 240;
    let mut g = Glb::default();
    let mut idx = Vec::new();
    for y in 0..SIDE - 1 {
        for x in 0..SIDE - 1 {
            let i = (y * SIDE + x) as u16;
            let (r, d) = (i + 1, i + SIDE as u16);
            idx.extend_from_slice(&[i, r, d, r, d + 1, d]);
        }
    }
    let idx = g.u16s(&idx, "SCALAR");
    let times: Vec<f32> = (0..KEYS).map(|k| k as f32 / 30.0).collect();
    let times = g.floats(&times, "SCALAR");
    let (mut nodes, mut meshes, mut samplers, mut channels) = (vec![], vec![], vec![], vec![]);
    for m in 0..4 {
        let mut pos = Vec::with_capacity(SIDE * SIDE * 3);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let (fx, fy) = (x as f32, y as f32);
                pos.extend_from_slice(&[fx, fy, (fx * 0.37 + fy * 0.11 + m as f32).sin()]);
            }
        }
        let pos = g.floats(&pos, "VEC3");
        let mut turns = Vec::with_capacity(KEYS * 4);
        for k in 0..KEYS {
            let k = k as f32;
            turns.extend_from_slice(&quat([(k * 0.1).sin(), 1.0, (k * 0.07).cos()], k * (3.0 + m as f32)));
        }
        let turns = g.floats(&turns, "VEC4");
        nodes.push(json!({"name": format!("Grid{m}"), "mesh": m, "translation": [m as f32 * 200.0, 0.0, 0.0]}));
        meshes.push(json!({"name": format!("Sheet{m}"), "primitives": [
            {"attributes": {"POSITION": pos}, "indices": idx}]}));
        samplers.push(json!({"input": times, "output": turns}));
        channels.push(json!({"sampler": m, "target": {"node": m, "path": "rotation"}}));
    }
    g.build(json!({
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2, 3]}],
        "nodes": nodes,
        "meshes": meshes,
        "animations": [{"name": "Tumble", "samplers": samplers, "channels": channels}]
    }))
}

#[test]
fn writing_on_more_threads_writes_the_same_bytes() {
    let doc = heavy_scene();
    let one = doc.to_fbx().unwrap();
    for threads in [0, 2, 3, 8] {
        let many = doc
            .to_fbx_with(&polysmelt::FbxOptions::default().with_threads(threads))
            .unwrap();
        assert!(one == many, "{threads} threads wrote other bytes");
    }
    // Too many alike triangles for `diff` to pair up once they move: the rest pose is compared, then how far each
    // node turns from where it should.
    let (_, back) = round_trip(&doc, "heavy");
    let (rest, rest_back) = (
        doc.flatten(Pose::Rest).unwrap().unwrap(),
        back.flatten(Pose::Rest).unwrap().unwrap(),
    );
    let d = mesh_diff(&rest, &rest_back, false, STILL);
    assert!(d.is_empty(), "differs in {d:?}");
    let (ra, rb) = (doc.rig().unwrap(), back.rig().unwrap());
    for i in 0..=2390 {
        let t = i as f32 / 300.0;
        let (wa, wb) = (worlds(&ra, Some(0), t), worlds(&rb, Some(0), t));
        for (name, m) in &wa {
            let n = &wb[name];
            let trace: f64 = (0..3)
                .flat_map(|c| (0..3).map(move |r| c * 4 + r))
                .map(|k| f64::from(m[k]) * f64::from(n[k]))
                .sum();
            let degrees = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
            assert!(degrees < 0.051, "{name} at {t}: {degrees} degrees off");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The corpus.

#[cfg(feature = "fs")]
mod corpus {
    use std::path::{Path, PathBuf};

    use polysmelt::Loader;
    use polysmelt::assets::Folder;

    use super::{BTreeMap, BTreeSet, Map, Pose, Value, diff, fbxcel_parses};

    /// Whether a document draws nothing: no triangles, lines or points in any pose.
    fn empty(doc: &polysmelt::Document) -> bool {
        matches!(doc.flatten(Pose::Rest), Ok(None))
    }

    const MODELS: &[&str] = &[
        "stl", "ply", "vox", "obj", "3ds", "fbx", "3mf", "usd", "usda", "usdc", "usdz", "gltf", "glb", "skp", "bvh",
        "dae", "zae", "kmz", "off", "vrm", "vrma", "x3d", "x3dv", "x3dz", "wrl", "wrz", "dxf", "splat", "spz", "sog",
        "las", "laz", "pcd", "xyz", "pts", "ptx", "asc", "e57", "amf", "lwo", "ac", "pmx", "pmd", "vmd", "x", "md2",
        "md3", "md5mesh", "md5anim", "iqm", "smd", "vta", "ms3d", "b3d", "asf", "amc", "c3d", "trc", "htr", "mesh",
        "skeleton", "xml", "flt", "ogex", "irrmesh", "irr", "xgl", "zgl", "vtk", "vtp", "vtu", "b3dm", "i3dm", "pnts",
        "cmpt", "msh", "meshb",
    ];

    fn ext(p: &Path) -> String {
        p.extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    }

    /// Each case's folder and model file, skipping cases that expect an error.
    fn cases() -> Vec<(String, PathBuf, PathBuf)> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
        let mut out = Vec::new();
        let mut formats: Vec<PathBuf> = std::fs::read_dir(&root).unwrap().flatten().map(|e| e.path()).collect();
        formats.sort();
        for format in formats.into_iter().filter(|p| p.is_dir()) {
            let mut dirs: Vec<PathBuf> = std::fs::read_dir(&format)
                .unwrap()
                .flatten()
                .map(|e| e.path())
                .collect();
            dirs.sort();
            for dir in dirs.into_iter().filter(|p| p.is_dir()) {
                let Ok(text) = std::fs::read_to_string(dir.join("expect.json")) else {
                    continue;
                };
                let expect: Map<String, Value> = serde_json::from_str(&text).unwrap();
                if expect.contains_key("error") {
                    continue;
                }
                let file = if let Some(f) = expect.get("file").and_then(Value::as_str) {
                    dir.join(f)
                } else {
                    let models: Vec<PathBuf> = std::fs::read_dir(&dir)
                        .unwrap()
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| p.is_file() && MODELS.contains(&ext(p).as_str()))
                        .collect();
                    let [one] = &models[..] else { continue };
                    one.clone()
                };
                let name = format.file_name().unwrap().to_string_lossy().into_owned();
                out.push((name, dir, file));
            }
        }
        out
    }

    /// Writes every corpus model that loads as FBX, has an independent parser parse it, reads it back and compares,
    /// printing a table of how many came back the same, and in which ways the rest differ. Writing and parsing must
    /// never fail; differences are reported.
    #[test]
    #[ignore = "reads the whole corpus: run with --ignored --nocapture for the report"]
    fn corpus_round_trip() {
        let only = std::env::var("FBX_CORPUS_ONLY").ok();
        let mut table: BTreeMap<String, (usize, usize, BTreeMap<&'static str, usize>)> = BTreeMap::new();
        let mut failures = Vec::new();
        let mut details = Vec::new();
        let mut empty_cases = Vec::new();
        for (format, dir, file) in cases() {
            if only.as_deref().is_some_and(|o| o != format) {
                continue;
            }
            let Ok(doc) = Loader::new().assets(Folder::new(&dir)).open(&file) else {
                continue;
            };
            let row = table.entry(format.clone()).or_default();
            row.0 += 1;
            let case = format!("{format}/{}", dir.file_name().unwrap().to_string_lossy());
            let fbx = match doc.to_fbx() {
                Ok(f) => f,
                Err(e) => {
                    failures.push(format!("{case}: writing: {e}"));
                    continue;
                }
            };
            if let Err(e) = fbxcel_parses(&fbx) {
                failures.push(format!("{case}: parsing: {e}"));
                continue;
            }
            let back = match polysmelt::read(fbx, "back.fbx") {
                Ok(b) => b,
                // The reader turns down an FBX file with no geometry at all: so be it, when there was none.
                Err(e) if e.to_string().contains("no mesh geometry") && empty(&doc) => {
                    row.1 += 1;
                    empty_cases.push(case);
                    continue;
                }
                Err(e) => {
                    failures.push(format!("{case}: reading back: {e}"));
                    continue;
                }
            };
            let d: BTreeSet<&'static str> = diff(&doc, &back);
            if d.is_empty() {
                row.1 += 1;
            } else {
                for f in &d {
                    *row.2.entry(f).or_default() += 1;
                }
                details.push(format!("{case}: {d:?}"));
            }
        }
        println!("\n{:<10} {:>6} {:>6}  differences", "format", "cases", "clean");
        let (mut cases, mut clean) = (0, 0);
        let mut totals: BTreeMap<&str, usize> = BTreeMap::new();
        for (format, (n, c, f)) in &table {
            cases += n;
            clean += c;
            for (k, v) in f {
                *totals.entry(k).or_default() += v;
            }
            let f: Vec<String> = f.iter().map(|(k, v)| format!("{k} {v}")).collect();
            println!("{format:<10} {n:>6} {c:>6}  {}", f.join(", "));
        }
        let t: Vec<String> = totals.iter().map(|(k, v)| format!("{k} {v}")).collect();
        println!("{:<10} {cases:>6} {clean:>6}  {}", "total", t.join(", "));
        println!("with nothing to draw, written and turned down by the reader as such: {empty_cases:?}");
        if std::env::var("FBX_CORPUS_DETAILS").is_ok() {
            for d in &details {
                println!("{d}");
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
