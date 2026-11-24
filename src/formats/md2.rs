//! MD2 (the Quake II engine's model format) as a glTF document: one mesh whose animation frames are morph targets.
//!
//! The file is little-endian binary: a 68-byte header (`IDP2`, version 8, the skin's size in pixels, the frame
//! size, then the counts and offsets of the skin names, texture coordinates, triangles, frames and GL commands),
//! and the lists it points to. Each frame stores every vertex position as three bytes scaled and offset by the
//! frame's own `scale` and `translate`, plus an index into the engine's table of 162 normals. Each triangle names
//! three positions and, separately, three texture coordinates (`s`, `t` in skin pixels), so a vertex here is each
//! distinct pair of a position and a texture coordinate. Texture coordinates are `s` and `t` over the skin's width
//! and height, with the origin at the top left as glTF's; with no skin size they're dropped. Triangles are wound
//! clockwise seen from the front, so each is turned around for glTF.
//!
//! The first frame is the mesh; every later frame is a morph target (position and normal offsets from the first),
//! named after its frame (the names are the mesh's `extras.targetNames`). Frames come in runs named by an action
//! and a number (`run1`, `run2`, ... then `pain101`, ...): each run of frames sharing a name once the trailing
//! digits are taken off becomes a linear animation of that name, a weights channel with a key per frame that turns
//! that frame's target fully on (the first frame's key turns them all off). The format has no frame rate; frames
//! are taken at 10 a second, the rate the engine's game logic runs at.
//!
//! The first skin name (a path in the game's folders, such as `models/monsters/tank/skin.pcx`) is the material's
//! texture, looked for relative to the model and by file name (the assets find the same name in other image
//! formats); with no skin, or one not found, a texture named for the model is looked for. Further skins are listed
//! in the source facts. GL commands (the triangle strips and fans the engine draws with) repeat the triangles and
//! are skipped.
//!
//! The format is Z up by the engine's convention, so the mesh node is turned for glTF's Y up (the source info's
//! `upAxis` is an assumed `+Z`, converted); game units are kept unscaled, and their length unknown. The assumed
//! frame rate is recorded when there are frames to animate.
//!
//! A file cut short, an offset past its end, a negative count, or a triangle naming a position or texture
//! coordinate past its list is an error. Counts are checked against the file's size before anything is allocated
//! for them, and a file whose frames would make more morph data than a model of the engine can hold is refused.

use std::collections::HashMap;
use std::ops::Range;

use serde_json::json;

use super::Input;
use super::scene::{
    self, Animation, Bin, Channel, Interp, Node, Path, Primitive, Scene, SceneMesh, Target, Z_UP_TO_Y_UP,
};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::{FastMap, Material};
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact};

/// Frames a second: the rate the engine's game logic runs at, as the format stores none.
const RATE: f32 = 10.0;

/// The most frames read; the engine holds 512.
const MAX_FRAMES: usize = 1024;

/// The most frame-vertices (frames times vertices) read; the engine's largest model has about six million.
const MAX_FRAME_VERTICES: usize = 1 << 24;

/// The engine's vertex normals, which each frame vertex names by index.
#[rustfmt::skip]
#[allow(clippy::unreadable_literal, reason = "the table as the engine gives it")]
static NORMALS: [[f32; 3]; 162] = [
    [-0.525731, 0.000000, 0.850651], [-0.442863, 0.238856, 0.864188], [-0.295242, 0.000000, 0.955423],
    [-0.309017, 0.500000, 0.809017], [-0.162460, 0.262866, 0.951056], [0.000000, 0.000000, 1.000000],
    [0.000000, 0.850651, 0.525731], [-0.147621, 0.716567, 0.681718], [0.147621, 0.716567, 0.681718],
    [0.000000, 0.525731, 0.850651], [0.309017, 0.500000, 0.809017], [0.525731, 0.000000, 0.850651],
    [0.295242, 0.000000, 0.955423], [0.442863, 0.238856, 0.864188], [0.162460, 0.262866, 0.951056],
    [-0.681718, 0.147621, 0.716567], [-0.809017, 0.309017, 0.500000], [-0.587785, 0.425325, 0.688191],
    [-0.850651, 0.525731, 0.000000], [-0.864188, 0.442863, 0.238856], [-0.716567, 0.681718, 0.147621],
    [-0.688191, 0.587785, 0.425325], [-0.500000, 0.809017, 0.309017], [-0.238856, 0.864188, 0.442863],
    [-0.425325, 0.688191, 0.587785], [-0.716567, 0.681718, -0.147621], [-0.500000, 0.809017, -0.309017],
    [-0.525731, 0.850651, 0.000000], [0.000000, 0.850651, -0.525731], [-0.238856, 0.864188, -0.442863],
    [0.000000, 0.955423, -0.295242], [-0.262866, 0.951056, -0.162460], [0.000000, 1.000000, 0.000000],
    [0.000000, 0.955423, 0.295242], [-0.262866, 0.951056, 0.162460], [0.238856, 0.864188, 0.442863],
    [0.262866, 0.951056, 0.162460], [0.500000, 0.809017, 0.309017], [0.238856, 0.864188, -0.442863],
    [0.262866, 0.951056, -0.162460], [0.500000, 0.809017, -0.309017], [0.850651, 0.525731, 0.000000],
    [0.716567, 0.681718, 0.147621], [0.716567, 0.681718, -0.147621], [0.525731, 0.850651, 0.000000],
    [0.425325, 0.688191, 0.587785], [0.864188, 0.442863, 0.238856], [0.688191, 0.587785, 0.425325],
    [0.809017, 0.309017, 0.500000], [0.681718, 0.147621, 0.716567], [0.587785, 0.425325, 0.688191],
    [0.955423, 0.295242, 0.000000], [1.000000, 0.000000, 0.000000], [0.951056, 0.162460, 0.262866],
    [0.850651, -0.525731, 0.000000], [0.955423, -0.295242, 0.000000], [0.864188, -0.442863, 0.238856],
    [0.951056, -0.162460, 0.262866], [0.809017, -0.309017, 0.500000], [0.681718, -0.147621, 0.716567],
    [0.850651, 0.000000, 0.525731], [0.864188, 0.442863, -0.238856], [0.809017, 0.309017, -0.500000],
    [0.951056, 0.162460, -0.262866], [0.525731, 0.000000, -0.850651], [0.681718, 0.147621, -0.716567],
    [0.681718, -0.147621, -0.716567], [0.850651, 0.000000, -0.525731], [0.809017, -0.309017, -0.500000],
    [0.864188, -0.442863, -0.238856], [0.951056, -0.162460, -0.262866], [0.147621, 0.716567, -0.681718],
    [0.309017, 0.500000, -0.809017], [0.425325, 0.688191, -0.587785], [0.442863, 0.238856, -0.864188],
    [0.587785, 0.425325, -0.688191], [0.688191, 0.587785, -0.425325], [-0.147621, 0.716567, -0.681718],
    [-0.309017, 0.500000, -0.809017], [0.000000, 0.525731, -0.850651], [-0.525731, 0.000000, -0.850651],
    [-0.442863, 0.238856, -0.864188], [-0.295242, 0.000000, -0.955423], [-0.162460, 0.262866, -0.951056],
    [0.000000, 0.000000, -1.000000], [0.295242, 0.000000, -0.955423], [0.162460, 0.262866, -0.951056],
    [-0.442863, -0.238856, -0.864188], [-0.309017, -0.500000, -0.809017], [-0.162460, -0.262866, -0.951056],
    [0.000000, -0.850651, -0.525731], [-0.147621, -0.716567, -0.681718], [0.147621, -0.716567, -0.681718],
    [0.000000, -0.525731, -0.850651], [0.309017, -0.500000, -0.809017], [0.442863, -0.238856, -0.864188],
    [0.162460, -0.262866, -0.951056], [0.238856, -0.864188, -0.442863], [0.500000, -0.809017, -0.309017],
    [0.425325, -0.688191, -0.587785], [0.716567, -0.681718, -0.147621], [0.688191, -0.587785, -0.425325],
    [0.587785, -0.425325, -0.688191], [0.000000, -0.955423, -0.295242], [0.000000, -1.000000, 0.000000],
    [0.262866, -0.951056, -0.162460], [0.000000, -0.850651, 0.525731], [0.000000, -0.955423, 0.295242],
    [0.238856, -0.864188, 0.442863], [0.262866, -0.951056, 0.162460], [0.500000, -0.809017, 0.309017],
    [0.716567, -0.681718, 0.147621], [0.525731, -0.850651, 0.000000], [-0.238856, -0.864188, -0.442863],
    [-0.500000, -0.809017, -0.309017], [-0.262866, -0.951056, -0.162460], [-0.850651, -0.525731, 0.000000],
    [-0.716567, -0.681718, -0.147621], [-0.716567, -0.681718, 0.147621], [-0.525731, -0.850651, 0.000000],
    [-0.500000, -0.809017, 0.309017], [-0.238856, -0.864188, 0.442863], [-0.262866, -0.951056, 0.162460],
    [-0.864188, -0.442863, 0.238856], [-0.809017, -0.309017, 0.500000], [-0.688191, -0.587785, 0.425325],
    [-0.681718, -0.147621, 0.716567], [-0.442863, -0.238856, 0.864188], [-0.587785, -0.425325, 0.688191],
    [-0.309017, -0.500000, 0.809017], [-0.147621, -0.716567, 0.681718], [-0.425325, -0.688191, 0.587785],
    [-0.162460, -0.262866, 0.951056], [0.442863, -0.238856, 0.864188], [0.162460, -0.262866, 0.951056],
    [0.309017, -0.500000, 0.809017], [0.147621, -0.716567, 0.681718], [0.000000, -0.525731, 0.850651],
    [0.425325, -0.688191, 0.587785], [0.587785, -0.425325, 0.688191], [0.688191, -0.587785, 0.425325],
    [-0.955423, 0.295242, 0.000000], [-0.951056, 0.162460, 0.262866], [-1.000000, 0.000000, 0.000000],
    [-0.850651, 0.000000, 0.525731], [-0.955423, -0.295242, 0.000000], [-0.951056, -0.162460, 0.262866],
    [-0.864188, 0.442863, -0.238856], [-0.951056, 0.162460, -0.262866], [-0.809017, 0.309017, -0.500000],
    [-0.864188, -0.442863, -0.238856], [-0.951056, -0.162460, -0.262866], [-0.809017, -0.309017, -0.500000],
    [-0.681718, 0.147621, -0.716567], [-0.681718, -0.147621, -0.716567], [-0.850651, 0.000000, -0.525731],
    [-0.688191, 0.587785, -0.425325], [-0.587785, 0.425325, -0.688191], [-0.425325, 0.688191, -0.587785],
    [-0.425325, -0.688191, -0.587785], [-0.587785, -0.425325, -0.688191], [-0.688191, -0.587785, -0.425325],
];

/// Reads an `.md2` model.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let data = input.read()?;
    let model = parse(&data, keep.morph_targets)?;
    let stem = input
        .path
        .file_stem()
        .map_or_else(|| "md2".to_owned(), |s| s.to_string_lossy().into_owned());

    let mut scene = Scene::default();
    let mut images = Images::new(!keep.image_bytes, input.dir());
    let textured = !model.prim.uvs.is_empty();
    let skin = model.skins.first().filter(|s| !s.is_empty());
    let found = skin.and_then(|s| assets.find(s, input.dir()).map(|p| (p, s.clone())));
    let found = found.or_else(|| {
        let p = assets.color_texture_for(input.path).filter(|_| textured)?;
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        Some((p, name))
    });
    let name = skin.cloned().unwrap_or_else(|| stem.clone());
    let mut material = Material::plain(&name, [1.0, 1.0, 1.0, 1.0]);
    material.color_image = found.and_then(|(p, r)| images.file(assets, p, &r));
    scene.materials.push(material);
    scene.images = images.list;

    let mut node = Node::named(stem.clone(), None);
    node.mesh = Some(0);
    let mesh_node = scene.add(node);
    let targets = model.prim.targets.len();
    if targets > 0 {
        for (name, frames) in groups(&model.names) {
            let mut times = Vec::with_capacity(frames.len());
            let mut values = vec![0.0; frames.len() * targets];
            for (k, f) in frames.enumerate() {
                times.push(f32_of(f64::from(u32_of(k))) / RATE);
                if let Some(Some(t)) = model.frames.get(f) {
                    values[k * targets + t] = 1.0;
                }
            }
            scene.animations.push(Animation {
                name,
                channels: vec![Channel {
                    node: mesh_node,
                    path: Path::Weights,
                    times,
                    values,
                    interp: Interp::Linear,
                }],
                extras: None,
            });
        }
    }
    let mut prim = model.prim;
    prim.material = Some(0);
    scene.meshes.push(SceneMesh {
        name: stem,
        prims: vec![prim],
        target_names: model
            .frames
            .iter()
            .zip(&model.names)
            .filter(|(t, _)| t.is_some())
            .map(|(_, n)| n.clone())
            .collect(),
        ..SceneMesh::default()
    });

    scene.facts.insert("frames".into(), json!(model.names.len()));
    scene.facts.insert("skins".into(), json!(model.skins.len()));
    scene.info.format = Some("MD2".into());
    scene.info.version = Some(model.version.to_string());
    scene.info.up_axis = Fact::Assumed(Axis::PosZ);
    if model.names.len() > 1 {
        scene.info.frame_rate = Fact::Assumed(f64::from(RATE));
    }
    if !model.skins.is_empty() {
        scene.source.insert("skins".into(), json!(model.skins));
    }
    scene.rotate_roots(Z_UP_TO_Y_UP);
    scene.finish(keep)
}

/// What a file holds, ready for the scene.
struct Model {
    version: i32,
    skins: Vec<String>,
    /// Each frame's name.
    names: Vec<String>,
    /// The first frame's mesh, with the later frames as targets when they're kept.
    prim: Primitive,
    /// Each kept frame's target; `None` for the first frame and any frame that doesn't move from it.
    frames: Vec<Option<usize>>,
}

/// A header count or offset, which can't be negative.
fn size(v: i32, what: &str) -> Result<usize> {
    usize::try_from(v).map_err(|_| invalid!("MD2 header's {what} is negative ({v})"))
}

/// Reads the file; `morphs` says whether frames after the first are kept as targets.
fn parse(data: &[u8], morphs: bool) -> Result<Model> {
    ensure!(data.starts_with(b"IDP2"), "not an MD2 file: no IDP2 signature");
    let mut b = Bin::new(data, "MD2");
    b.skip(4)?;
    let mut h = [0i32; 16];
    for v in &mut h {
        *v = b.i32()?;
    }
    let [
        version,
        skin_w,
        skin_h,
        frame_size,
        n_skins,
        n_xyz,
        n_st,
        n_tris,
        _,
        n_frames,
        ofs_skins,
        ofs_st,
        ofs_tris,
        ofs_frames,
        _,
        _,
    ] = h;
    let n_skins = size(n_skins, "skin count")?;
    let n_xyz = size(n_xyz, "vertex count")?;
    let n_st = size(n_st, "texture coordinate count")?;
    let n_tris = size(n_tris, "triangle count")?;
    let n_frames = size(n_frames, "frame count")?;
    let frame_size = size(frame_size, "frame size")?;

    let mut skins = Vec::new();
    if n_skins > 0 {
        b.seek(size(ofs_skins, "skin offset")?)?;
        for _ in 0..b.count(n_skins as u64, 64)? {
            skins.push(scene::text(b.fixed_str(64)?));
        }
    }
    let mut st = Vec::new();
    if n_st > 0 {
        b.seek(size(ofs_st, "texture coordinate offset")?)?;
        for _ in 0..b.count(n_st as u64, 4)? {
            st.push([b.i16()?, b.i16()?]);
        }
    }
    ensure!(n_tris > 0, "MD2 file has no triangles");
    b.seek(size(ofs_tris, "triangle offset")?)?;
    let mut tris = Vec::with_capacity(b.count(n_tris as u64, 12)?);
    for t in 0..n_tris {
        let v = [b.u16()?, b.u16()?, b.u16()?];
        let s = [b.u16()?, b.u16()?, b.u16()?];
        ensure!(
            v.iter().all(|&i| usize::from(i) < n_xyz),
            "MD2 triangle {t} names a vertex past the {n_xyz} each frame holds"
        );
        // With no texture coordinates at all, their indices mean nothing.
        ensure!(
            n_st == 0 || s.iter().all(|&i| usize::from(i) < n_st),
            "MD2 triangle {t} names a texture coordinate past the {n_st} there are"
        );
        tris.push((v, if n_st == 0 { [0; 3] } else { s }));
    }

    // A vertex per distinct position and texture coordinate pair, in the order the triangles name them.
    let mut slot: FastMap<(u16, u16), u32> = FastMap::default();
    let mut verts: Vec<(u16, u16)> = Vec::new();
    let mut indices = Vec::with_capacity(tris.len() * 3);
    for (v, s) in &tris {
        // Clockwise front faces in the file; glTF's are counterclockwise.
        for k in [2, 1, 0] {
            let key = (v[k], s[k]);
            let i = *slot.entry(key).or_insert_with(|| {
                verts.push(key);
                u32_of(verts.len() - 1)
            });
            indices.push(i);
        }
    }

    ensure!(n_frames > 0, "MD2 file has no frames");
    ensure!(
        n_frames <= MAX_FRAMES,
        "MD2 file has {n_frames} frames, more than {MAX_FRAMES}"
    );
    let need = n_xyz.checked_mul(4).and_then(|n| n.checked_add(40));
    ensure!(
        need.is_some_and(|n| frame_size >= n),
        "MD2 frames are {frame_size} bytes, too few for {n_xyz} vertices"
    );
    let kept = if morphs { n_frames } else { 1 };
    ensure!(
        kept.checked_mul(verts.len()).is_some_and(|n| n <= MAX_FRAME_VERTICES),
        "MD2 file's {n_frames} frames of {} vertices are more than a model can hold",
        verts.len()
    );
    let ofs_frames = size(ofs_frames, "frame offset")?;
    b.seek(ofs_frames)?;
    b.count(n_frames as u64, frame_size)?;

    let mut names = Vec::with_capacity(n_frames);
    let mut frames = Vec::with_capacity(kept);
    let mut prim = Primitive::default();
    let mut frame_pos = vec![[0.0f32; 3]; n_xyz];
    let mut frame_nrm = vec![[0.0f32; 3]; n_xyz];
    for f in 0..n_frames {
        // In range: the count above checked all the frames fit.
        b.seek(ofs_frames + f * frame_size)?;
        let scale = b.vec3()?;
        let translate = b.vec3()?;
        names.push(scene::text(b.fixed_str(16)?));
        if f >= kept {
            continue;
        }
        let packed = b.bytes(n_xyz * 4)?.chunks_exact(4);
        for ((p, n), c) in frame_pos.iter_mut().zip(&mut frame_nrm).zip(packed) {
            *p = [0, 1, 2].map(|k| f32::from(c[k]) * scale[k] + translate[k]);
            // An index past the table has no normal; the writer gives the vertex its faces'.
            *n = NORMALS.get(usize::from(c[3])).copied().unwrap_or([0.0; 3]);
        }
        if f == 0 {
            prim.positions = verts.iter().map(|&(v, _)| frame_pos[usize::from(v)]).collect();
            prim.normals = verts.iter().map(|&(v, _)| frame_nrm[usize::from(v)]).collect();
            frames.push(None);
        } else {
            frames.push(target(&mut prim, &verts, &frame_pos, &frame_nrm));
        }
    }
    if n_st > 0 && skin_w > 0 && skin_h > 0 {
        let (w, h) = (f32_of(f64::from(skin_w)), f32_of(f64::from(skin_h)));
        prim.uvs = vec![
            verts
                .iter()
                .map(|&(_, s)| {
                    let [u, v] = st[usize::from(s)];
                    [f32::from(u) / w, f32::from(v) / h]
                })
                .collect(),
        ];
    }
    prim.indices = indices;
    Ok(Model {
        version,
        skins,
        names,
        prim,
        frames,
    })
}

/// Adds a frame (its positions and normals per file vertex, which `verts` name) as a target of `prim` (holding
/// the first frame) and gives its index, or `None` when the frame doesn't move a vertex: then it's the first frame
/// again, with no target of its own.
#[allow(
    clippy::needless_bitwise_bool,
    reason = "branchless, so the loop over every vertex vectorizes"
)]
fn target(prim: &mut Primitive, verts: &[(u16, u16)], pos: &[[f32; 3]], nrm: &[[f32; 3]]) -> Option<usize> {
    // Each vertex's offset from the first frame, or `None` when none moves.
    let offsets = |frame: &[[f32; 3]], first: &[[f32; 3]]| {
        let mut moved = false;
        let d: Vec<[f32; 3]> = verts
            .iter()
            .zip(first)
            .map(|(&(v, _), b)| {
                let a = frame[usize::from(v)];
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                moved |= (d[0] != 0.0) | (d[1] != 0.0) | (d[2] != 0.0);
                d
            })
            .collect();
        moved.then_some(d)
    };
    let positions = offsets(pos, &prim.positions)?;
    // A target needn't carry what it leaves alone.
    let normals = offsets(nrm, &prim.normals).unwrap_or_default();
    prim.targets.push(Target { positions, normals });
    Some(prim.targets.len() - 1)
}

/// The runs of frames that make each animation: consecutive frames whose names match once trailing digits are
/// taken off, named that way (`frames` when nothing is left; a name used again gets a number).
fn groups(names: &[String]) -> Vec<(String, Range<usize>)> {
    let mut out: Vec<(String, Range<usize>)> = Vec::new();
    for (i, n) in names.iter().enumerate() {
        let p = n.trim_end_matches(|c: char| c.is_ascii_digit());
        let p = if p.is_empty() { "frames" } else { p };
        match out.last_mut() {
            Some((q, r)) if q == p => r.end = i + 1,
            _ => out.push((p.to_owned(), i..i + 1)),
        }
    }
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (name, _) in &mut out {
        let n = seen.entry(name.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            *name = format!("{name}_{n}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_group_by_name_without_digits() {
        let names: Vec<String> = ["stand01", "stand02", "run1", "run2", "run3", "7", "stand1"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let g = groups(&names);
        let got: Vec<(&str, Range<usize>)> = g.iter().map(|(n, r)| (n.as_str(), r.clone())).collect();
        assert_eq!(
            got,
            vec![("stand", 0..2), ("run", 2..5), ("frames", 5..6), ("stand_2", 6..7)]
        );
    }

    #[test]
    fn hostile_headers_are_errors() {
        assert!(parse(b"IDP", true).is_err());
        assert!(parse(b"XXXX", true).is_err());
        let mut h = b"IDP2".to_vec();
        for v in [8i32, 8, 8, 44, 0, 1, 0, 1, 0, 1, 0, 0, 68, 80, 0, 0] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        // Triangle 0 0 0 / 0 0 0, then a frame cut short.
        h.extend_from_slice(&[0; 12]);
        h.extend_from_slice(&[0; 20]);
        assert!(parse(&h, true).is_err());
        // Now whole.
        h.extend_from_slice(&[0; 24]);
        let m = parse(&h, true).unwrap();
        assert_eq!(m.prim.indices, [0, 0, 0]);
        // A negative count.
        h[4 + 4 * 7..4 + 4 * 8].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(parse(&h, true).is_err());
    }

    /// Loads every prefix of `bytes`, and `bytes` with each byte overwritten by a few values, through `load`;
    /// any result but a panic passes.
    fn sweep(bytes: &[u8], load: impl Fn(&[u8])) {
        for end in 0..bytes.len() {
            load(&bytes[..end]);
        }
        let mut copy = bytes.to_vec();
        for at in 0..bytes.len() {
            for value in [0x00, 0x7F, 0x80, 0xFF] {
                copy[at] = value;
                load(&copy);
            }
            copy[at] = bytes[at];
        }
    }

    #[test]
    fn hostile_bytes_never_panic() {
        let bytes = include_bytes!("../../tests/corpus/md2/morph_frame_groups/quad.md2");
        assert!(
            load(
                Input::memory(bytes, "quad.md2"),
                &crate::assets::Memory::new(),
                Keep::ALL
            )
            .is_ok()
        );
        sweep(bytes, |b| {
            let _ = load(Input::memory(b, "quad.md2"), &crate::assets::Memory::new(), Keep::ALL);
        });
    }

    #[test]
    fn source_info_is_game_units_z_up_at_ten_frames_a_second() {
        let bytes = include_bytes!("../../tests/corpus/md2/morph_frame_groups/quad.md2");
        let doc = load(
            Input::memory(bytes, "quad.md2"),
            &crate::assets::Memory::new(),
            Keep::ALL,
        )
        .unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("MD2"));
        assert_eq!(info.version.as_deref(), Some("8"));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
        assert_eq!(info.frame_rate, Fact::Assumed(10.0));
        assert!(info.converted.up_axis && !info.converted.units);
    }
}
