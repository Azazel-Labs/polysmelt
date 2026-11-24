//! IQM (Inter-Quake Model) as a glTF document: skinned meshes, a skeleton and its animations.
//!
//! The file is little-endian binary: a 124-byte header (`INTERQUAKEMODEL`, version 2, the file's size, then the
//! counts and offsets of each list), a text block that every name is an offset into, and the lists. Vertices are
//! held as vertex arrays, one per attribute, each with its own number format (8, 16 and 32-bit integers, signed or
//! not, and 16, 32 and 64-bit floats) and component count, all the same length; triangles index them from the start
//! of the arrays, and each mesh is a range of vertices and of triangles with a name and a material name.
//!
//! Each mesh is a node and a mesh of one primitive, named after it. Positions, normals, texture coordinates and
//! colors come from their arrays: every texture coordinate array is a set (origin at the top left, as glTF's), and
//! every color array a color set (integers read as fractions of their range, sRGB as the format's tools write them,
//! so turned linear; floats taken as linear). Blend index and weight arrays give each vertex its joints and weights
//! (integer weights as fractions of their range); with more than one pair, each pair adds its influences.
//! Triangles are wound clockwise seen from the front, so each is turned around for glTF. Tangents, adjacency,
//! custom arrays (types from 16 on), bounds and extensions are dropped. A mesh with no triangles is left out. The
//! comment block's strings are the source info's `comments`.
//!
//! The material name is the texture's file (`Body.jpg`), looked for as written and, when it has no extension, as a
//! `.png` (the assets find the same name in other image formats, and by file name in other folders); each distinct
//! name is one material, named after it.
//!
//! Joints give a name, a parent (earlier in the list) and a translation, rotation and scale relative to it: they
//! become nodes, and a skin over all of them that the skinned meshes share, whose inverse bind matrices undo each
//! joint's place in the model. Poses (one per joint) say how frames move them: each of a joint's ten channels
//! (translation X, Y, Z, rotation X, Y, Z, W, scale X, Y, Z) is its offset, plus, when its bit is set in the mask,
//! the next 16-bit number of the frame times its scale. Each animation is a range of frames with a name and a frame
//! rate (24 when it isn't a positive number): a linear animation with translation, rotation and scale channels for
//! each joint, leaving out those that hold the joint's rest value in every frame. Looping animations say so in their
//! extras (`loop`). The source info's `frameRate` is the first animation's rate that is a positive number (assumed 24
//! when none is); when the animations' rates differ, each gives its own in its extras (`frameRate`).
//!
//! The format is Z up by its tools' convention, so the scene's roots are turned for glTF's Y up (the source info's
//! `upAxis` is an assumed `+Z`, converted); units are kept unscaled, and their length unknown.
//!
//! A file cut short, a list past its end, a name outside the text block, an unknown number format, vertex arrays
//! that don't all hold every vertex, a triangle naming a vertex outside its mesh, a weighted influence naming a
//! joint there isn't, a joint whose parent isn't earlier in the list, poses that aren't one per joint, frames
//! that don't hold the channels the poses say, or an animation past the frames is an error. Counts are checked
//! against the file's size before anything is allocated for them.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::Input;
use super::scene::{
    self, Animation, Bin, Channel, Interp, M4, Node, Path, Primitive, Scene, SceneMesh, Skin, Z_UP_TO_Y_UP,
};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::{Material, srgb_to_linear};
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact, Handedness};

/// The frame rate taken when an animation gives none that's usable.
const DEFAULT_RATE: f32 = 24.0;

/// The most pose values (frames times joints) decoded.
const MAX_POSE_FRAMES: usize = 1 << 22;

/// The header's size.
const HEADER: usize = 124;

/// Vertex array types.
const POSITION: u32 = 0;
const TEXCOORD: u32 = 1;
const NORMAL: u32 = 2;
const BLENDINDEXES: u32 = 4;
const BLENDWEIGHTS: u32 = 5;
const COLOR: u32 = 6;

/// Reads an `.iqm` model.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    let data = input.read()?;
    let file = parse(&data, keep.animations)?;
    let stem = input
        .path
        .file_stem()
        .map_or_else(|| "iqm".to_owned(), |s| s.to_string_lossy().into_owned());
    let mut scene = build(&file, &stem, &input, assets, keep)?;
    scene.info.format = Some("IQM".into());
    scene.info.version = Some("2".into());
    scene.info.up_axis = Fact::Assumed(Axis::PosZ);
    scene.info.handedness = Fact::Assumed(Handedness::Right);
    scene.info.comments.clone_from(&file.comments);
    // The first animation's rate that is one; with none, the rate frames are taken at.
    let rates = file.anims.iter().map(|a| a.rate);
    if let Some(r) = rates.clone().find(|r| *r > 0.0 && r.is_finite()) {
        scene.info.frame_rate = Fact::Declared(f64::from(r));
    } else if !file.anims.is_empty() {
        scene.info.frame_rate = Fact::Assumed(f64::from(DEFAULT_RATE));
    }
    scene.rotate_roots(Z_UP_TO_Y_UP);
    scene.finish(keep)
}

/// One vertex array: its type, component count, and values as floats, `size` per vertex.
struct Array {
    kind: u32,
    size: usize,
    values: Vec<f32>,
}

struct Mesh {
    name: String,
    material: String,
    first_vertex: usize,
    num_vertexes: usize,
    first_triangle: usize,
    num_triangles: usize,
}

struct Joint {
    name: String,
    parent: i32,
    t: [f32; 3],
    r: [f32; 4],
    s: [f32; 3],
}

struct Pose {
    mask: u32,
    offset: [f32; 10],
    scale: [f32; 10],
}

struct Anim {
    name: String,
    first: usize,
    count: usize,
    rate: f32,
    looping: bool,
}

/// What a file holds.
struct File {
    vertices: usize,
    arrays: Vec<Array>,
    triangles: Vec<[u32; 3]>,
    meshes: Vec<Mesh>,
    joints: Vec<Joint>,
    poses: Vec<Pose>,
    anims: Vec<Anim>,
    /// Each frame's ten channels for every pose, laid end to end.
    frames: Vec<f32>,
    num_frames: usize,
    /// The comment block's strings.
    comments: Vec<String>,
}

/// A header field as a count or offset.
fn usize_of(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

/// The name at `ofs` in the text block.
fn name(text: &[u8], ofs: u32) -> Result<String> {
    let ofs = usize_of(ofs);
    ensure!(
        ofs < text.len() || (ofs == 0 && text.is_empty()),
        Corrupt,
        "IQM name at {ofs} is outside the {}-byte text block",
        text.len()
    );
    let s = text.get(ofs..).unwrap_or_default();
    Ok(scene::text(&s[..s.iter().position(|&c| c == 0).unwrap_or(s.len())]))
}

/// The bytes of one number of `format`, or `None` for one that isn't a format.
fn format_size(format: u32) -> Option<usize> {
    Some(match format {
        0 | 1 => 1,
        2 | 3 | 6 => 2,
        4 | 5 | 7 => 4,
        8 => 8,
        _ => return None,
    })
}

/// A half-precision float.
fn half(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 if frac == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// The numbers of `format` in `raw`, end to end; integers are fractions of their range when `normalized`.
fn numbers(raw: &[u8], format: u32, normalized: bool) -> Vec<f32> {
    fn ints<const N: usize>(raw: &[u8], range: f64, normalized: bool, int: impl Fn([u8; N]) -> f64) -> Vec<f32> {
        raw.chunks_exact(N)
            .map(|c| {
                let v = int(c.try_into().unwrap_or([0; N]));
                f32_of(if normalized { (v / range).max(-1.0) } else { v })
            })
            .collect()
    }
    match format {
        0 => ints(raw, 127.0, normalized, |b: [u8; 1]| f64::from(i8::from_le_bytes(b))),
        1 => ints(raw, 255.0, normalized, |b: [u8; 1]| f64::from(b[0])),
        2 => ints(raw, 32_767.0, normalized, |b| f64::from(i16::from_le_bytes(b))),
        3 => ints(raw, 65_535.0, normalized, |b| f64::from(u16::from_le_bytes(b))),
        4 => ints(raw, 2_147_483_647.0, normalized, |b| f64::from(i32::from_le_bytes(b))),
        5 => ints(raw, 4_294_967_295.0, normalized, |b| f64::from(u32::from_le_bytes(b))),
        6 => raw
            .chunks_exact(2)
            .map(|c| half(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        7 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => raw
            .chunks_exact(8)
            .map(|c| f32_of(f64::from_le_bytes(c.try_into().unwrap_or([0; 8]))))
            .collect(),
    }
}

/// The file's lists; its frames only when `frames`.
fn parse(data: &[u8], frames: bool) -> Result<File> {
    ensure!(
        data.starts_with(b"INTERQUAKEMODEL\0"),
        UnknownFormat,
        "not an IQM file: no INTERQUAKEMODEL signature"
    );
    let mut b = Bin::new(data, "IQM");
    b.skip(16)?;
    let mut h = [0u32; 27];
    for v in &mut h {
        *v = b.u32()?;
    }
    let [
        version,
        filesize,
        _flags,
        num_text,
        ofs_text,
        num_meshes,
        ofs_meshes,
        num_vertexarrays,
        num_vertexes,
        ofs_vertexarrays,
        num_triangles,
        ofs_triangles,
        _ofs_adjacency,
        num_joints,
        ofs_joints,
        num_poses,
        ofs_poses,
        num_anims,
        ofs_anims,
        num_frames,
        num_framechannels,
        ofs_frames,
        _ofs_bounds,
        num_comment,
        ofs_comment,
        ..,
    ] = h;
    ensure!(
        version == 2,
        Unsupported,
        "IQM version {version} isn't supported, only 2"
    );
    ensure!(
        usize_of(filesize) <= data.len(),
        Truncated,
        "IQM file is cut short: {} bytes of {filesize}",
        data.len()
    );
    // Lists are read from the start of the file to the size it gives.
    let data = &data[..usize_of(filesize).max(HEADER)];
    let mut b = Bin::new(data, "IQM");

    let text: &[u8] = if num_text > 0 {
        b.seek(usize_of(ofs_text))?;
        b.bytes(usize_of(num_text))?
    } else {
        &[]
    };

    let vertices = usize_of(num_vertexes);
    let mut arrays = Vec::new();
    if num_vertexarrays > 0 {
        b.seek(usize_of(ofs_vertexarrays))?;
        let n = b.count(u64::from(num_vertexarrays), 20)?;
        let mut heads = Vec::with_capacity(n);
        for _ in 0..n {
            heads.push([b.u32()?, b.u32()?, b.u32()?, b.u32()?, b.u32()?]);
        }
        for (i, [kind, _flags, format, size, offset]) in heads.into_iter().enumerate() {
            let Some(bytes) = format_size(format) else {
                bail!(Corrupt, "IQM vertex array {i} has unknown number format {format}");
            };
            // Tangents and custom arrays aren't kept.
            if !matches!(kind, POSITION | TEXCOORD | NORMAL | BLENDINDEXES | BLENDWEIGHTS | COLOR) {
                continue;
            }
            let size = usize_of(size);
            ensure!(size > 0, Corrupt, "IQM vertex array {i} has no components");
            let mut a = Bin::new(data, "IQM");
            a.seek(usize_of(offset))?;
            let total = vertices
                .checked_mul(size)
                .ok_or_else(|| err!(LimitExceeded, "IQM vertex array {i} is too large"))?;
            a.count(total as u64, bytes)?;
            let normalized = matches!(kind, TEXCOORD | BLENDWEIGHTS | COLOR);
            let mut values = numbers(a.bytes(total * bytes)?, format, normalized);
            // Integer colors are sRGB; the alpha, the fourth component, isn't.
            if kind == COLOR && format <= 5 {
                for (k, v) in values.iter_mut().enumerate() {
                    if k % size < 3 {
                        *v = srgb_to_linear(*v);
                    }
                }
            }
            arrays.push(Array { kind, size, values });
        }
    }

    let mut triangles = Vec::new();
    if num_triangles > 0 {
        b.seek(usize_of(ofs_triangles))?;
        let n = b.count(u64::from(num_triangles), 12)?;
        triangles = b
            .bytes(n * 12)?
            .chunks_exact(12)
            .map(|c| std::array::from_fn(|k| u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])))
            .collect();
    }

    let mut meshes = Vec::new();
    if num_meshes > 0 {
        b.seek(usize_of(ofs_meshes))?;
        let n = b.count(u64::from(num_meshes), 24)?;
        for _ in 0..n {
            let [nm, mt, fv, nv, ft, nt] = [b.u32()?, b.u32()?, b.u32()?, b.u32()?, b.u32()?, b.u32()?];
            meshes.push(Mesh {
                name: name(text, nm)?,
                material: name(text, mt)?,
                first_vertex: usize_of(fv),
                num_vertexes: usize_of(nv),
                first_triangle: usize_of(ft),
                num_triangles: usize_of(nt),
            });
        }
    }

    let mut joints = Vec::new();
    if num_joints > 0 {
        b.seek(usize_of(ofs_joints))?;
        let n = b.count(u64::from(num_joints), 48)?;
        for _ in 0..n {
            let nm = b.u32()?;
            let parent = b.i32()?;
            joints.push(Joint {
                name: name(text, nm)?,
                parent,
                t: b.vec3()?,
                r: b.vec4()?,
                s: b.vec3()?,
            });
        }
    }

    let mut poses = Vec::new();
    if num_poses > 0 {
        b.seek(usize_of(ofs_poses))?;
        let n = b.count(u64::from(num_poses), 88)?;
        for _ in 0..n {
            let _parent = b.i32()?;
            let mask = b.u32()?;
            let mut offset = [0.0; 10];
            let mut scale = [0.0; 10];
            for v in &mut offset {
                *v = b.f32()?;
            }
            for v in &mut scale {
                *v = b.f32()?;
            }
            poses.push(Pose { mask, offset, scale });
        }
    }

    let mut anims = Vec::new();
    if num_anims > 0 {
        b.seek(usize_of(ofs_anims))?;
        let n = b.count(u64::from(num_anims), 20)?;
        for _ in 0..n {
            let nm = b.u32()?;
            anims.push(Anim {
                name: name(text, nm)?,
                first: usize_of(b.u32()?),
                count: usize_of(b.u32()?),
                rate: b.f32()?,
                looping: b.u32()? & 1 != 0,
            });
        }
    }

    let num_frames = usize_of(num_frames);
    let read_frames = frames;
    let mut frames = Vec::new();
    if read_frames && num_frames > 0 && !poses.is_empty() {
        let channels: u32 = poses.iter().map(|p| (p.mask & 0x3ff).count_ones()).sum();
        ensure!(
            channels == num_framechannels,
            Corrupt,
            "IQM poses animate {channels} channels, but frames hold {num_framechannels}"
        );
        ensure!(
            num_frames
                .checked_mul(poses.len())
                .is_some_and(|n| n <= MAX_POSE_FRAMES),
            LimitExceeded,
            "IQM file has {num_frames} frames of {} joints, more than can be read",
            poses.len()
        );
        b.seek(usize_of(ofs_frames))?;
        let values = (num_frames as u64).saturating_mul(u64::from(num_framechannels));
        let values = b.count(values, 2)?;
        // As many as the poses animate, frame after frame, as checked above.
        let mut words = b
            .bytes(values * 2)?
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]));
        frames.reserve(num_frames * poses.len() * 10);
        for _ in 0..num_frames {
            for p in &poses {
                for c in 0..10 {
                    let mut v = p.offset[c];
                    if p.mask & (1 << c) != 0 {
                        v += f32::from(words.next().unwrap_or(0)) * p.scale[c];
                    }
                    frames.push(v);
                }
            }
        }
    }
    // The comment block: text, its strings ended by NULs. One past the file's end is skipped, not an error.
    let comments = usize_of(ofs_comment)
        .checked_add(usize_of(num_comment))
        .and_then(|end| data.get(usize_of(ofs_comment)..end))
        .filter(|_| num_comment > 0)
        .map(|block| {
            block
                .split(|&c| c == 0)
                .map(|s| scene::text(s).trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    Ok(File {
        vertices,
        arrays,
        triangles,
        meshes,
        joints,
        poses,
        anims,
        frames,
        num_frames,
        comments,
    })
}

/// The texture a material name gives: as written, then as a `.png` when it has no extension.
fn texture(name: &str, input: &Input<'_>, assets: &dyn Assets, images: &mut Images) -> Option<usize> {
    if name.is_empty() {
        return None;
    }
    let mut tries = vec![name.to_owned()];
    if std::path::Path::new(name).extension().is_none() {
        tries.push(format!("{name}.png"));
    }
    tries
        .iter()
        .find_map(|t| assets.find(t, input.dir()).map(|p| (p, t)))
        .and_then(|(p, t)| images.file(assets, p, t))
}

fn build(f: &File, stem: &str, input: &Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Scene> {
    let mut scene = Scene::default();
    let n = f.joints.len();
    let mut worlds: Vec<M4> = Vec::with_capacity(n);
    for (i, j) in f.joints.iter().enumerate() {
        let parent = usize::try_from(j.parent).ok();
        ensure!(
            j.parent >= -1 && parent.is_none_or(|p| p < i),
            Corrupt,
            "IQM joint {i} has parent {}, which isn't an earlier joint",
            j.parent
        );
        let mut node = Node::named(j.name.clone(), parent);
        node.translation = j.t;
        node.rotation = scene::normalize(j.r.map(f64::from));
        node.scale = j.s;
        let local = node.matrix();
        worlds.push(parent.map_or(local, |p| scene::mul(&worlds[p], &local)));
        scene.add(node);
    }

    let mut images = Images::new(!keep.image_bytes, input.dir());
    let mut materials: HashMap<String, usize> = HashMap::new();
    let pairs: Vec<(&Array, &Array)> = f
        .arrays
        .iter()
        .filter(|a| a.kind == BLENDINDEXES)
        .zip(f.arrays.iter().filter(|a| a.kind == BLENDWEIGHTS))
        .collect();
    let skinned = n > 0 && !pairs.is_empty();
    let first = |kind: u32| f.arrays.iter().find(|a| a.kind == kind);
    let Some(positions) = first(POSITION).filter(|a| a.size >= 3) else {
        ensure!(
            f.meshes.iter().all(|m| m.num_triangles == 0),
            Corrupt,
            "IQM file has meshes but no positions"
        );
        return finish_anims(scene, f, keep.animations);
    };
    let normals = first(NORMAL).filter(|a| a.size >= 3);
    for (mi, m) in f.meshes.iter().enumerate() {
        let verts = m.first_vertex..m.first_vertex.saturating_add(m.num_vertexes);
        ensure!(
            verts.end <= f.vertices,
            Corrupt,
            "IQM mesh {mi}'s vertices run past the {} there are",
            f.vertices
        );
        let tris = m.first_triangle..m.first_triangle.saturating_add(m.num_triangles);
        ensure!(
            tris.end <= f.triangles.len(),
            Corrupt,
            "IQM mesh {mi}'s triangles run past the {} there are",
            f.triangles.len()
        );
        if tris.is_empty() || verts.is_empty() {
            continue;
        }
        let mut indices = Vec::with_capacity(tris.len() * 3);
        for (ti, t) in f.triangles[tris].iter().enumerate() {
            let local = t.map(|v| usize_of(v).checked_sub(verts.start).filter(|&v| v < verts.len()));
            let [Some(a), Some(b), Some(c)] = local else {
                bail!(Corrupt, "IQM mesh {mi}'s triangle {ti} names a vertex outside the mesh");
            };
            // Clockwise front faces in the file; glTF's are counterclockwise.
            indices.extend([a, c, b].map(u32_of));
        }
        let take3 = |a: &Array| -> Vec<[f32; 3]> {
            verts
                .clone()
                .map(|v| std::array::from_fn(|k| a.values[v * a.size + k]))
                .collect()
        };
        let mut prim = Primitive {
            positions: take3(positions),
            normals: normals.map(take3).unwrap_or_default(),
            indices,
            ..Primitive::default()
        };
        for a in f.arrays.iter().filter(|a| a.kind == TEXCOORD) {
            prim.uvs.push(
                verts
                    .clone()
                    .map(|v| {
                        [
                            a.values[v * a.size],
                            a.values
                                .get(v * a.size + 1)
                                .copied()
                                .filter(|_| a.size > 1)
                                .unwrap_or(0.0),
                        ]
                    })
                    .collect(),
            );
        }
        for a in f.arrays.iter().filter(|a| a.kind == COLOR) {
            prim.colors.push(
                verts
                    .clone()
                    .map(|v| color(&a.values[v * a.size..(v + 1) * a.size]))
                    .collect(),
            );
        }
        if skinned {
            let influences: usize = pairs.iter().map(|(i, w)| i.size.min(w.size)).sum();
            prim.influences = influences;
            prim.joints.reserve(verts.len() * influences);
            prim.weights.reserve(verts.len() * influences);
            for v in verts.clone() {
                for (ia, wa) in &pairs {
                    for k in 0..ia.size.min(wa.size) {
                        let w = wa.values[v * wa.size + k];
                        let j = ia.values[v * ia.size + k];
                        let weighted = w > 0.0 && w.is_finite();
                        ensure!(
                            !weighted || (j >= 0.0 && f64::from(j) < f64::from(u32::try_from(n).unwrap_or(u32::MAX))),
                            Corrupt,
                            "IQM vertex {v} is weighted to joint {j}, but there are {n}"
                        );
                        prim.joints.push(if weighted { joint_of(j) } else { 0 });
                        prim.weights.push(if weighted { w } else { 0.0 });
                    }
                }
            }
        }
        let material = if let Some(&i) = materials.get(&m.material) {
            i
        } else {
            let name = if m.material.is_empty() {
                format!("material{mi}")
            } else {
                m.material.clone()
            };
            let mut mat = Material::plain(&name, [1.0, 1.0, 1.0, 1.0]);
            mat.color_image = texture(&m.material, input, assets, &mut images);
            scene.materials.push(mat);
            materials.insert(m.material.clone(), scene.materials.len() - 1);
            scene.materials.len() - 1
        };
        prim.material = Some(material);
        let name = if m.name.is_empty() {
            format!("{stem}{mi}")
        } else {
            m.name.clone()
        };
        let mut node = Node::named(name.clone(), None);
        node.mesh = Some(scene.meshes.len());
        if skinned {
            node.skin = Some(0);
        }
        scene.add(node);
        scene.meshes.push(SceneMesh {
            name,
            prims: vec![prim],
            ..SceneMesh::default()
        });
    }
    scene.images = images.list;
    if skinned && !scene.meshes.is_empty() {
        scene.skins.push(Skin {
            name: stem.to_owned(),
            joints: (0..n).collect(),
            inverse_binds: worlds.iter().map(scene::invert).collect(),
        });
    }
    scene.facts.insert("meshes".into(), json!(f.meshes.len()));
    finish_anims(scene, f, keep.animations)
}

/// A blend index, checked to be a joint, as one.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "checked to be a whole number of joints, fewer than 2^32"
)]
fn joint_of(j: f32) -> u32 {
    j as u32
}

/// The time of frame `k` at `rate` frames a second.
fn frame_time(k: usize, rate: f32) -> f32 {
    f32_of(f64::from(u32::try_from(k).unwrap_or(u32::MAX))) / rate
}

/// A color from an array's components: RGB, or RGBA.
fn color(c: &[f32]) -> [f32; 4] {
    let get = |k: usize, or: f32| c.get(k).copied().unwrap_or(or);
    [get(0, 0.0), get(1, 0.0), get(2, 0.0), get(3, 1.0)]
}

/// Adds the skeleton's facts and, when `keep`, the animations.
fn finish_anims(mut scene: Scene, f: &File, keep: bool) -> Result<Scene> {
    let n = f.joints.len();
    scene.facts.insert("joints".into(), json!(n));
    scene.facts.insert("frames".into(), json!(f.num_frames));
    if f.anims.is_empty() {
        return Ok(scene);
    }
    if !keep {
        // Their frames weren't read.
        scene.facts.insert("animations".into(), json!(f.anims.len()));
        return Ok(scene);
    }
    ensure!(
        f.poses.len() == n,
        Corrupt,
        "IQM file has {} poses for {n} joints",
        f.poses.len()
    );
    let rate_of = |a: &Anim| {
        if a.rate > 0.0 && a.rate.is_finite() {
            a.rate
        } else {
            DEFAULT_RATE
        }
    };
    // When animations differ in rate, each says its own in its extras.
    let rates_differ = f
        .anims
        .windows(2)
        .any(|w| rate_of(&w[0]).to_bits() != rate_of(&w[1]).to_bits());
    for (ai, a) in f.anims.iter().enumerate() {
        let frames = a.first..a.first.saturating_add(a.count);
        ensure!(
            frames.end <= f.num_frames,
            Corrupt,
            "IQM animation {ai} runs past the {} frames there are",
            f.num_frames
        );
        if frames.is_empty() {
            continue;
        }
        let rate = rate_of(a);
        let times: Vec<f32> = (0..frames.len()).map(|k| frame_time(k, rate)).collect();
        let mut channels = Vec::new();
        for (j, joint) in f.joints.iter().enumerate() {
            let value = |fr: usize, c: usize| f.frames[(fr * n + j) * 10 + c];
            let rest = scene::normalize(joint.r.map(f64::from));
            let t: Vec<f32> = frames.clone().flat_map(|fr| [0, 1, 2].map(|c| value(fr, c))).collect();
            let r: Vec<f32> = frames
                .clone()
                .flat_map(|fr| scene::normalize([3, 4, 5, 6].map(|c| f64::from(value(fr, c)))))
                .collect();
            let s: Vec<f32> = frames.clone().flat_map(|fr| [7, 8, 9].map(|c| value(fr, c))).collect();
            let same = |v: &[f32], rest: &[f32]| v.chunks(rest.len()).all(|k| close(k, rest));
            let same_rot = |v: &[f32]| v.chunks(4).all(|k| close(k, &rest) || close(k, &rest.map(|c| -c)));
            if !same(&t, &joint.t) {
                channels.push(channel(j, Path::Translation, &times, t));
            }
            if !same_rot(&r) {
                channels.push(channel(j, Path::Rotation, &times, r));
            }
            if !same(&s, &joint.s) {
                channels.push(channel(j, Path::Scale, &times, s));
            }
        }
        let name = if a.name.is_empty() {
            format!("anim{ai}")
        } else {
            a.name.clone()
        };
        scene.animations.push(Animation {
            name,
            channels,
            extras: (a.looping || rates_differ).then(|| {
                let mut m = serde_json::Map::new();
                if a.looping {
                    m.insert("loop".into(), Value::Bool(true));
                }
                if rates_differ {
                    m.insert("frameRate".into(), json!(rate));
                }
                Value::Object(m)
            }),
        });
    }
    scene.facts.insert("animations".into(), json!(f.anims.len()));
    Ok(scene)
}

/// Whether two values match to a millionth of a unit.
fn close(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-6)
}

fn channel(node: usize, path: Path, times: &[f32], values: Vec<f32>) -> Channel {
    Channel {
        node,
        path,
        times: times.to_vec(),
        values,
        interp: Interp::Linear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let bytes = include_bytes!("../../tests/corpus/iqm/skinned_animated/rig.iqm");
        assert!(
            load(
                Input::memory(bytes, "rig.iqm"),
                &crate::assets::Memory::new(),
                Keep::ALL
            )
            .is_ok()
        );
        sweep(bytes, |b| {
            let _ = load(Input::memory(b, "rig.iqm"), &crate::assets::Memory::new(), Keep::ALL);
        });
    }

    #[test]
    fn source_info_has_the_first_rate_and_the_comments() {
        let mut bytes = include_bytes!("../../tests/corpus/iqm/skinned_animated/rig.iqm").to_vec();
        let at = u32::try_from(bytes.len()).unwrap();
        let comment = b"made by hand\0exported\0";
        bytes.extend(comment);
        let set =
            |b: &mut Vec<u8>, field: usize, v: u32| b[16 + 4 * field..20 + 4 * field].copy_from_slice(&v.to_le_bytes());
        let len = u32::try_from(bytes.len()).unwrap();
        set(&mut bytes, 1, len);
        set(&mut bytes, 23, u32::try_from(comment.len()).unwrap());
        set(&mut bytes, 24, at);
        let doc = load(
            Input::memory(&bytes, "rig.iqm"),
            &crate::assets::Memory::new(),
            Keep::ALL,
        )
        .unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("IQM"));
        assert_eq!(info.comments, ["made by hand", "exported"]);
        // "wave" at 10 frames a second, "hold" with none (24): the first that has one, and each its own.
        assert_eq!(info.frame_rate, Fact::Declared(10.0));
        assert_eq!(doc.json["animations"][0]["extras"]["frameRate"], 10.0);
        assert_eq!(doc.json["animations"][1]["extras"]["frameRate"], 24.0);
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert!(info.converted.up_axis);
    }
}
