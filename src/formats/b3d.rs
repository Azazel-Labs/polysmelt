//! `Blitz3D` (`.b3d`), a binary tree of tagged chunks: textures (`TEXS`), brushes (`BRUS`) and a hierarchy of nodes
//! (`NODE`), each with a position, scale and rotation and optionally a mesh (`MESH`, holding vertices `VRTS` and
//! triangle lists `TRIS`), bone weights on the mesh above it (`BONE`), keyframes (`KEYS`) and the animation's
//! length and rate (`ANIM`).
//!
//! The nodes become glTF nodes, each triangle list a primitive with its brush as the material (color, alpha,
//! shininess as roughness, the first texture as the color image, full-bright as unlit, two-sided), and the vertices'
//! normals, colors (when the brush asks for them) and texture coordinate sets are kept. Bones make a skin over the
//! mesh they weigh; keys, at frame numbers divided by the animation's frames per second, an animation. `Blitz3D` is Y
//! up and left-handed, so the scene is mirrored through the XY plane, and its quaternions turn row vectors, so each is
//! read as its conjugate. In the source info the handedness is declared left (the format defines it) and converted,
//! the up axis an assumed `+Y`, the units' length unknown; the frame rate is the `ANIM` chunk's, or an assumed 60
//! when keys need one and it gives none.

use std::collections::HashMap;

use serde_json::json;

use super::scene::{self, Animation, Channel, Interp, M4, Node, Path as Key, Primitive, Scene, SceneMesh, Skin};
use crate::Result;
use crate::assets::Assets;
use crate::bytes::Bin;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::{Material, srgb_to_linear};
use crate::num::{f32_of, u32_of};
use crate::quat;
use crate::source::{Axis, Fact, Handedness};

/// How deep nodes may nest.
const MAX_DEPTH: usize = 256;
/// The frame rate when the file gives none.
const DEFAULT_FPS: f32 = 60.0;

/// A texture: its file.
struct Texture {
    file: String,
}

/// A brush: a material.
struct Brush {
    name: String,
    color: [f32; 4],
    shininess: f32,
    fx: i32,
    textures: Vec<i32>,
}

/// A mesh: its vertices and triangle lists (brush, triangles).
#[derive(Default)]
struct Mesh {
    brush: i32,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    has_normals: bool,
    colors: Vec<[f32; 4]>,
    has_colors: bool,
    uvs: Vec<Vec<[f32; 2]>>,
    lists: Vec<(i32, Vec<[i32; 3]>)>,
}

/// Keys at frame numbers.
#[derive(Default)]
struct Keys {
    positions: Vec<(i32, [f32; 3])>,
    scales: Vec<(i32, [f32; 3])>,
    rotations: Vec<(i32, [f32; 4])>,
}

/// A node as the file has it.
struct FileNode {
    name: String,
    parent: Option<usize>,
    translation: [f32; 3],
    scale: [f32; 3],
    /// glTF order, x, y, z, w, already turned to act on column vectors.
    rotation: [f32; 4],
    mesh: Option<usize>,
    bones: Vec<(i32, f32)>,
    keys: Keys,
}

/// What a file holds.
#[derive(Default)]
struct B3d {
    version: i32,
    textures: Vec<Texture>,
    brushes: Vec<Brush>,
    meshes: Vec<Mesh>,
    nodes: Vec<FileNode>,
    /// The animation's flags, frame count and frames per second.
    anim: Option<(i32, i32, f32)>,
    /// Texture coordinate transforms, blend modes and other brush and texture settings left out.
    unsupported: usize,
}

/// Reads a `.b3d` file.
pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let bytes = input.read()?;
    let f = parse(&bytes)?;
    let stem = input
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut scene = Scene::default();
    let mut facts = serde_json::Map::new();

    // Brushes.
    let dir = input.dir();
    let mut images = Images::new(!keep.image_bytes, dir);
    for b in &f.brushes {
        let unit = |c: f32| if c.is_finite() { c.clamp(0.0, 1.0) } else { 1.0 };
        let c = b.color.map(unit);
        let mut m = Material::plain(
            &b.name,
            [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]],
        );
        m.roughness = 1.0 - unit(b.shininess.max(0.0)).sqrt();
        m.unlit = b.fx & 1 != 0;
        m.double_sided = b.fx & 0x10 != 0;
        let texture = b
            .textures
            .first()
            .and_then(|&t| usize::try_from(t).ok())
            .filter(|&t| t < f.textures.len());
        if keep.textures
            && let Some(t) = texture
        {
            m.color_image = images.find(assets, &[&f.textures[t].file], &[dir]);
        }
        scene.materials.push(m);
    }
    scene.images = images.list;

    // Nodes.
    for n in &f.nodes {
        let mut node = Node::named(n.name.clone(), n.parent);
        node.translation = n.translation;
        node.scale = n.scale;
        node.rotation = n.rotation;
        scene.add(node);
    }
    let world = scene.world();

    // Which mesh each bone weighs: the nearest one at or above it.
    let owner = |mut i: usize| -> Option<(usize, usize)> {
        for _ in 0..=f.nodes.len() {
            if let Some(m) = f.nodes[i].mesh {
                return Some((i, m));
            }
            i = f.nodes[i].parent?;
        }
        None
    };
    let mut bones_of: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut stray = 0usize;
    for (i, n) in f.nodes.iter().enumerate() {
        if n.bones.is_empty() {
            continue;
        }
        match owner(i) {
            Some((_, m)) => bones_of.entry(m).or_default().push(i),
            None => stray += n.bones.len(),
        }
    }

    // Meshes.
    let mut dropped = 0usize;
    for (ni, n) in f.nodes.iter().enumerate() {
        let Some(mi) = n.mesh else { continue };
        let mesh = &f.meshes[mi];
        let count = mesh.positions.len();
        // The skin: the bones weighing this mesh, then the mesh's own node for vertices no bone moves.
        let bones = bones_of.remove(&mi).unwrap_or_default();
        let mut links: Vec<Vec<(u32, f32)>> = Vec::new();
        if !bones.is_empty() {
            links = vec![Vec::new(); count];
            for (j, &b) in bones.iter().enumerate() {
                for &(v, w) in &f.nodes[b].bones {
                    match usize::try_from(v).ok().and_then(|v| links.get_mut(v)) {
                        Some(l) if w.is_finite() && w > 0.0 => l.push((u32_of(j), w)),
                        Some(_) => {}
                        None => stray += 1,
                    }
                }
            }
        }
        let rest = u32_of(bones.len());
        let mut unbound = false;
        // A primitive per brush.
        let mut by_brush: Vec<(i32, Vec<[i32; 3]>)> = Vec::new();
        for (brush, tris) in &mesh.lists {
            let brush = if *brush < 0 { mesh.brush } else { *brush };
            match by_brush.iter_mut().find(|(b, _)| *b == brush) {
                Some((_, t)) => t.extend_from_slice(tris),
                None => by_brush.push((brush, tris.clone())),
            }
        }
        let mut prims = Vec::new();
        for (brush, tris) in by_brush {
            let material = usize::try_from(brush).ok().filter(|&b| b < f.brushes.len());
            let colors = mesh.has_colors && material.is_some_and(|b| f.brushes[b].fx & 2 != 0);
            let mut p = Primitive {
                material,
                ..Primitive::default()
            };
            p.uvs = vec![Vec::new(); mesh.uvs.len()];
            if colors {
                p.colors = vec![Vec::new()];
            }
            let mut map: HashMap<usize, u32> = HashMap::new();
            let mut used = Vec::new();
            for t in &tris {
                let Some(t) = t
                    .iter()
                    .map(|&v| usize::try_from(v).ok().filter(|&v| v < count))
                    .collect::<Option<Vec<usize>>>()
                else {
                    dropped += 1;
                    continue;
                };
                for v in t {
                    let next = u32_of(used.len());
                    let k = *map.entry(v).or_insert(next);
                    if k == next {
                        used.push(v);
                    }
                    p.indices.push(k);
                }
            }
            if p.indices.is_empty() {
                continue;
            }
            for &v in &used {
                p.positions.push(mesh.positions[v]);
                if mesh.has_normals {
                    p.normals.push(mesh.normals[v]);
                }
                if colors {
                    let c = mesh.colors[v];
                    p.colors[0].push([srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]]);
                }
                for (k, set) in mesh.uvs.iter().enumerate() {
                    p.uvs[k].push(set[v]);
                }
            }
            if !links.is_empty() {
                let width = used
                    .iter()
                    .map(|&v| links[v].len())
                    .max()
                    .unwrap_or(0)
                    .clamp(1, scene::MAX_INFLUENCES);
                p.influences = width;
                p.rest_joint = rest;
                for &v in &used {
                    unbound |= links[v].is_empty();
                    let mut l = links[v].clone();
                    l.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for k in 0..width {
                        let (j, w) = l.get(k).copied().unwrap_or((0, 0.0));
                        p.joints.push(j);
                        p.weights.push(w);
                    }
                }
            }
            prims.push(p);
        }
        if prims.is_empty() {
            continue;
        }
        let index = scene.meshes.len();
        scene.meshes.push(SceneMesh {
            name: n.name.clone(),
            prims,
            ..SceneMesh::default()
        });
        scene.nodes[ni].mesh = Some(index);
        if !bones.is_empty() {
            // Vertices are in the mesh node's space: each inverse bind takes them there first.
            let bind = |j: usize| -> M4 { scene::mul(&scene::invert(&world[j]), &world[ni]) };
            let mut joints = bones.clone();
            if unbound {
                joints.push(ni);
            }
            scene.skins.push(Skin {
                name: n.name.clone(),
                inverse_binds: joints.iter().map(|&j| bind(j)).collect(),
                joints,
            });
            scene.nodes[ni].skin = Some(scene.skins.len() - 1);
        }
    }
    if dropped > 0 {
        facts.insert("dropped_triangles".into(), json!(dropped));
    }
    if stray > 0 {
        facts.insert("dropped_weights".into(), json!(stray));
    }

    // Keys.
    let fps = match f.anim {
        Some((_, _, fps)) if fps.is_finite() && fps > 0.0 => fps,
        _ => DEFAULT_FPS,
    };
    let time = |frame: i32| f32_of(f64::from(frame.max(0)) / f64::from(fps));
    let mut channels = Vec::new();
    for (i, n) in f.nodes.iter().enumerate() {
        let k = &n.keys;
        if !k.positions.is_empty() {
            channels.push(Channel {
                node: i,
                path: Key::Translation,
                times: k.positions.iter().map(|x| time(x.0)).collect(),
                values: k.positions.iter().flat_map(|x| x.1).collect(),
                interp: Interp::Linear,
            });
        }
        if !k.rotations.is_empty() {
            channels.push(Channel {
                node: i,
                path: Key::Rotation,
                times: k.rotations.iter().map(|x| time(x.0)).collect(),
                values: k.rotations.iter().flat_map(|x| x.1).collect(),
                interp: Interp::Linear,
            });
        }
        if !k.scales.is_empty() {
            channels.push(Channel {
                node: i,
                path: Key::Scale,
                times: k.scales.iter().map(|x| time(x.0)).collect(),
                values: k.scales.iter().flat_map(|x| x.1).collect(),
                interp: Interp::Linear,
            });
        }
    }
    if !channels.is_empty() {
        scene.animations.push(Animation {
            name: stem,
            channels,
            extras: None,
        });
    }
    ensure!(!scene.nodes.is_empty(), Empty, "B3D file has no nodes");

    facts.insert("nodes".into(), json!(f.nodes.len()));
    facts.insert("brushes".into(), json!(f.brushes.len()));
    if f.unsupported > 0 {
        facts.insert("unsupported_settings".into(), json!(f.unsupported));
    }
    let mut source = serde_json::Map::new();
    scene.info.format = Some("B3D".into());
    scene.info.version = Some(f.version.to_string());
    scene.info.up_axis = Fact::Assumed(Axis::PosY);
    scene.info.handedness = Fact::Declared(Handedness::Left);
    if let Some((_, frames, rate)) = f.anim {
        source.insert("frames".into(), json!(frames));
        scene.info.frame_rate = if rate.is_finite() && rate > 0.0 {
            Fact::Declared(f64::from(rate))
        } else {
            Fact::Assumed(f64::from(DEFAULT_FPS))
        };
    } else if !scene.animations.is_empty() {
        scene.info.frame_rate = Fact::Assumed(f64::from(DEFAULT_FPS));
    }
    scene.source = source;
    scene.facts = facts;
    scene.mirror_z();
    scene.finish(keep)
}

/// The chunks in `data`, each tag and body; a chunk claiming more than `data` holds is an error.
fn chunks(data: &[u8]) -> impl Iterator<Item = Result<([u8; 4], &[u8])>> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        if at >= data.len() {
            return None;
        }
        let rest = &data[at..];
        if rest.len() < 8 {
            at = data.len();
            return Some(Err(err!(Truncated, "B3D file is cut short in a chunk header")));
        }
        let tag: [u8; 4] = [rest[0], rest[1], rest[2], rest[3]];
        let size = i32::from_le_bytes([rest[4], rest[5], rest[6], rest[7]]);
        let Some(size) = usize::try_from(size).ok().filter(|&s| s <= rest.len() - 8) else {
            at = data.len();
            return Some(Err(err!(
                Truncated,
                "B3D {} chunk runs past the end of what holds it: it claims {size} bytes, and there is room for {}",
                String::from_utf8_lossy(&tag),
                rest.len() - 8
            )));
        };
        at += 8 + size;
        Some(Ok((tag, &rest[8..8 + size])))
    })
}

/// A NUL-terminated string.
fn cstring(b: &mut Bin<'_>) -> Result<String> {
    let rest = &b.data[b.at..];
    let Some(end) = rest.iter().position(|&c| c == 0) else {
        bail!(Truncated, "B3D file is cut short in a name");
    };
    let s = scene::text(&rest[..end]);
    b.skip(end + 1)?;
    Ok(s)
}

/// Reads the file.
fn parse(data: &[u8]) -> Result<B3d> {
    ensure!(
        data.len() >= 8 && &data[..4] == b"BB3D",
        UnknownFormat,
        "isn't a Blitz3D file: it doesn't start with BB3D"
    );
    let mut f = B3d::default();
    let mut top = chunks(data);
    let (_, body) = top.next().unwrap_or_else(|| Err(err!(Empty, "B3D file is empty")))?;
    let mut b = Bin::new(body, "B3D");
    f.version = b.i32()?;
    for c in chunks(&body[4..]) {
        let (tag, body) = c?;
        match &tag {
            b"TEXS" => texs(body, &mut f)?,
            b"BRUS" => brus(body, &mut f)?,
            b"NODE" => node(body, None, 0, &mut f)?,
            _ => {}
        }
    }
    Ok(f)
}

fn texs(body: &[u8], f: &mut B3d) -> Result<()> {
    let mut b = Bin::new(body, "B3D");
    while b.left() > 0 {
        let file = cstring(&mut b)?.replace('\\', "/");
        let _flags = b.i32()?;
        let blend = b.i32()?;
        let pos = b.vec2()?;
        let scale = b.vec2()?;
        let rot = b.f32()?;
        let moved = pos
            .iter()
            .chain(&scale)
            .zip([0.0, 0.0, 1.0, 1.0])
            .any(|(&c, d)| (c - d).abs() > 1e-6);
        if moved || rot.abs() > 1e-6 || !(blend == 1 || blend == 2) {
            f.unsupported += 1;
        }
        f.textures.push(Texture { file });
    }
    Ok(())
}

fn brus(body: &[u8], f: &mut B3d) -> Result<()> {
    let mut b = Bin::new(body, "B3D");
    let n = b.i32()?;
    ensure!(
        (0..=8).contains(&n),
        Corrupt,
        "B3D brushes claim {n} textures each, more than the 8 the format allows"
    );
    let n = usize::try_from(n).unwrap_or(0);
    while b.left() > 0 {
        let name = cstring(&mut b)?;
        let color = b.vec4()?;
        let shininess = b.f32()?;
        let blend = b.i32()?;
        let fx = b.i32()?;
        let mut textures = Vec::with_capacity(n);
        for _ in 0..n {
            textures.push(b.i32()?);
        }
        if blend > 1 || textures.iter().skip(1).any(|&t| t >= 0) {
            f.unsupported += 1;
        }
        f.brushes.push(Brush {
            name,
            color,
            shininess,
            fx,
            textures,
        });
    }
    Ok(())
}

fn node(body: &[u8], parent: Option<usize>, depth: usize, f: &mut B3d) -> Result<()> {
    ensure!(
        depth < MAX_DEPTH,
        LimitExceeded,
        "B3D nodes nest deeper than {MAX_DEPTH}"
    );
    let mut b = Bin::new(body, "B3D");
    let name = cstring(&mut b)?;
    let translation = b.vec3()?;
    let scale = b.vec3()?;
    let [w, x, y, z] = b.vec4()?;
    let index = f.nodes.len();
    f.nodes.push(FileNode {
        name,
        parent,
        translation,
        scale,
        rotation: turn([w, x, y, z]),
        mesh: None,
        bones: Vec::new(),
        keys: Keys::default(),
    });
    for c in chunks(&body[b.at..]) {
        let (tag, body) = c?;
        match &tag {
            b"MESH" => {
                // One mesh per node; a second one is read and dropped.
                let m = mesh(body)?;
                if f.nodes[index].mesh.is_none() {
                    f.nodes[index].mesh = Some(f.meshes.len());
                    f.meshes.push(m);
                }
            }
            b"BONE" => {
                let mut b = Bin::new(body, "B3D");
                let n = b.left() / 8;
                let bones = &mut f.nodes[index].bones;
                bones.reserve(n);
                for _ in 0..n {
                    bones.push((b.i32()?, b.f32()?));
                }
            }
            b"KEYS" => keys(body, &mut f.nodes[index].keys)?,
            b"ANIM" => {
                let mut b = Bin::new(body, "B3D");
                let anim = (b.i32()?, b.i32()?, b.f32()?);
                f.anim.get_or_insert(anim);
            }
            b"NODE" => node(body, Some(index), depth + 1, f)?,
            _ => {}
        }
    }
    let k = &mut f.nodes[index].keys;
    k.positions.sort_by_key(|x| x.0);
    k.scales.sort_by_key(|x| x.0);
    k.rotations.sort_by_key(|x| x.0);
    Ok(())
}

/// A rotation as the file stores it (w first, turning row vectors) in glTF's order, turning column vectors.
fn turn([w, x, y, z]: [f32; 4]) -> [f32; 4] {
    let q = [-x, -y, -z, w].map(f64::from);
    quat::normalize(q)
}

fn keys(body: &[u8], k: &mut Keys) -> Result<()> {
    let mut b = Bin::new(body, "B3D");
    let flags = b.i32()?;
    let stride = 4
        + if flags & 1 != 0 { 12 } else { 0 }
        + if flags & 2 != 0 { 12 } else { 0 }
        + if flags & 4 != 0 { 16 } else { 0 };
    let n = b.left() / stride;
    for _ in 0..n {
        let frame = b.i32()?;
        if flags & 1 != 0 {
            k.positions.push((frame, b.vec3()?));
        }
        if flags & 2 != 0 {
            k.scales.push((frame, b.vec3()?));
        }
        if flags & 4 != 0 {
            let [w, x, y, z] = b.vec4()?;
            k.rotations.push((frame, turn([w, x, y, z])));
        }
    }
    Ok(())
}

fn mesh(body: &[u8]) -> Result<Mesh> {
    let mut b = Bin::new(body, "B3D");
    let mut m = Mesh {
        brush: b.i32()?,
        ..Mesh::default()
    };
    for c in chunks(&body[4..]) {
        let (tag, body) = c?;
        match &tag {
            b"VRTS" => vrts(body, &mut m)?,
            b"TRIS" => {
                let mut b = Bin::new(body, "B3D");
                let brush = b.i32()?;
                let n = b.left() / 12;
                let mut tris = Vec::with_capacity(n);
                for _ in 0..n {
                    tris.push([b.i32()?, b.i32()?, b.i32()?]);
                }
                m.lists.push((brush, tris));
            }
            _ => {}
        }
    }
    Ok(m)
}

fn vrts(body: &[u8], m: &mut Mesh) -> Result<()> {
    let mut b = Bin::new(body, "B3D");
    let flags = b.i32()?;
    let sets = b.i32()?;
    let size = b.i32()?;
    let (Some(sets), Some(size)) = (
        usize::try_from(sets).ok().filter(|&s| s <= 8),
        usize::try_from(size).ok().filter(|&s| s <= 4),
    ) else {
        bail!(
            Corrupt,
            "B3D vertices claim {sets} texture coordinate sets of {size}, more than the format allows (8 of 4)"
        );
    };
    let normals = flags & 1 != 0;
    let colors = flags & 2 != 0;
    let stride = 12 + if normals { 12 } else { 0 } + if colors { 16 } else { 0 } + sets * size * 4;
    let n = b.left() / stride;
    // Earlier vertices without what these have get defaults; these without what earlier ones had, too.
    let before = m.positions.len();
    if m.uvs.len() < sets {
        m.uvs.resize(sets, vec![[0.0; 2]; before]);
    }
    m.has_normals |= normals;
    m.has_colors |= colors;
    m.positions.reserve(n);
    for _ in 0..n {
        m.positions.push(b.vec3()?);
        m.normals.push(if normals { b.vec3()? } else { [0.0; 3] });
        m.colors.push(if colors { b.vec4()? } else { [1.0; 4] });
        for set in m.uvs.iter_mut().take(sets) {
            let mut t = [0.0f32; 4];
            for c in t.iter_mut().take(size) {
                *c = b.f32()?;
            }
            set.push([t[0], t[1]]);
        }
        for set in m.uvs.iter_mut().skip(sets) {
            set.push([0.0; 2]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;

    fn chunk(tag: [u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = tag.to_vec();
        out.extend(i32::try_from(body.len()).unwrap().to_le_bytes());
        out.extend(body);
        out
    }

    fn f(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn i(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn node_chunk(name: &str, t: [f32; 3], q: [f32; 4], kids: &[u8]) -> Vec<u8> {
        let mut body = name.as_bytes().to_vec();
        body.push(0);
        body.extend(f(&t));
        body.extend(f(&[1.0, 1.0, 1.0]));
        body.extend(f(&q));
        body.extend(kids);
        chunk(*b"NODE", &body)
    }

    /// A triangle on a root node, weighed by a bone that turns a quarter about Y between frames 0 and 10.
    fn sample() -> Vec<u8> {
        let mut texs = b"skin.png\0".to_vec();
        texs.extend(i(&[1, 2]));
        texs.extend(f(&[0.0, 0.0, 1.0, 1.0, 0.0]));
        let mut brus = i(&[1]);
        brus.extend(b"red\0");
        brus.extend(f(&[1.0, 0.0, 0.0, 1.0, 0.5]));
        brus.extend(i(&[1, 0x10 | 2, 0]));
        let mut vrts = i(&[3, 1, 2]);
        for (p, uv) in [
            ([0.0, 0.0, 0.0], [0.0, 0.0]),
            ([1.0, 0.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 1.0]),
        ] {
            vrts.extend(f(&p));
            vrts.extend(f(&[0.0, 0.0, -1.0]));
            vrts.extend(f(&[0.5, 0.5, 0.5, 1.0]));
            vrts.extend(f(&uv));
        }
        let mut mesh = i(&[0]);
        mesh.extend(chunk(*b"VRTS", &vrts));
        let mut tris = i(&[-1]);
        tris.extend(i(&[0, 1, 2]));
        mesh.extend(chunk(*b"TRIS", &tris));
        let mut bone = Vec::new();
        bone.extend(chunk(*b"BONE", &[i(&[0]), f(&[1.0]), i(&[1]), f(&[1.0])].concat()));
        let mut keys = i(&[4]);
        let half = std::f32::consts::FRAC_1_SQRT_2;
        keys.extend(i(&[0]));
        keys.extend(f(&[1.0, 0.0, 0.0, 0.0]));
        keys.extend(i(&[10]));
        keys.extend(f(&[half, 0.0, half, 0.0]));
        bone.extend(chunk(*b"KEYS", &keys));
        let mut kids = chunk(*b"MESH", &mesh);
        kids.extend(chunk(*b"ANIM", &[i(&[0, 10]), f(&[10.0])].concat()));
        kids.extend(node_chunk("bone", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], &bone));
        let root = node_chunk("root", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], &kids);
        let mut body = i(&[1]);
        body.extend(chunk(*b"TEXS", &texs));
        body.extend(chunk(*b"BRUS", &brus));
        body.extend(root);
        chunk(*b"BB3D", &body)
    }

    fn read(bytes: &[u8]) -> Result<Document> {
        load(super::super::Input::memory(bytes, "t.b3d"), &Memory::new(), Keep::ALL)
    }

    #[test]
    fn a_skinned_triangle_with_a_brush_and_keys() {
        let doc = read(&sample()).unwrap();
        let j = &doc.json();
        assert_eq!(doc.facts["triangles"], json!(1));
        assert_eq!(j["materials"][0]["name"], "red");
        assert_eq!(j["materials"][0]["doubleSided"], true);
        // The brush asks for vertex colors.
        assert!(j["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"].is_number());
        // The bone, then the mesh's node for the vertex no bone weighs.
        assert_eq!(j["skins"][0]["joints"].as_array().unwrap().len(), 2);
        let rig = doc.rig().unwrap();
        let bone = rig.find("bone").unwrap();
        assert_eq!(rig.nodes[bone].parent, rig.find("root"));
        let a = &rig.animations[0];
        assert_eq!(a.channels.len(), 1);
        assert!((a.channels[0].times[1] - 1.0).abs() < 1e-6);
        // Mirrored: the file's normal toward -Z faces +Z, and the winding turns with it.
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert!(mesh.prims[0].normals.iter().all(|n| n[2] > 0.99));
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("B3D"));
        assert_eq!(info.version.as_deref(), Some("1"));
        assert_eq!(info.handedness, Fact::Declared(Handedness::Left));
        assert!(info.converted.handedness && !info.converted.up_axis && !info.converted.units);
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(info.frame_rate, Fact::Declared(10.0));
    }

    #[test]
    fn row_vector_quaternions_are_conjugated_then_mirrored() {
        // The file's quarter turn about Y (w first): in Blitz3D it takes +X to +Z, which in the mirrored scene is
        // -Z; glTF's quarter turn about +Y takes +X to -Z.
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let q = scene::mirror_quat(turn([half, 0.0, half, 0.0]));
        let v = quat::rotate(q.map(f64::from), [1.0, 0.0, 0.0]);
        assert!((v[2] + 1.0).abs() < 1e-6, "{v:?}");
    }

    #[test]
    fn cut_short_and_broken_files() {
        let good = sample();
        for k in 0..good.len() {
            let _ = read(&good[..k]);
        }
        assert!(read(b"").unwrap_err().to_string().contains("isn't"));
        assert!(read(b"BB3Dxxxx").is_err());
        // A chunk claiming more than its parent.
        let mut v = good.clone();
        v[4..8].copy_from_slice(&i32::MAX.to_le_bytes());
        let err = read(&v).unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::Truncated);
        assert!(err.to_string().contains("runs past the end of what holds it"), "{err}");
        // Too many texture coordinate sets.
        let at = v.windows(4).position(|w| w == b"VRTS").unwrap();
        let mut v = good.clone();
        v[at + 12..at + 16].copy_from_slice(&99i32.to_le_bytes());
        assert!(read(&v).unwrap_err().to_string().contains("texture coordinate sets"));
        // A triangle index past the vertices is dropped.
        let mut v = good.clone();
        let at = v.windows(4).position(|w| w == b"TRIS").unwrap();
        v[at + 20..at + 24].copy_from_slice(&7i32.to_le_bytes());
        let doc = read(&v);
        assert!(doc.is_err() || doc.unwrap().facts["dropped_triangles"] == json!(1));
    }

    #[test]
    fn nodes_nested_too_deep_are_an_error() {
        let mut inner = node_chunk("n", [0.0; 3], [1.0, 0.0, 0.0, 0.0], &[]);
        for _ in 0..MAX_DEPTH + 4 {
            inner = node_chunk("n", [0.0; 3], [1.0, 0.0, 0.0, 0.0], &inner);
        }
        let mut body = i(&[1]);
        body.extend(inner);
        let err = read(&chunk(*b"BB3D", &body)).unwrap_err();
        assert!(err.to_string().contains("nest"), "{err}");
    }
}
