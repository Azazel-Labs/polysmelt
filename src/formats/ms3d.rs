//! `MilkShape` 3D (`.ms3d`), a binary model of vertices, triangles with a normal and texture coordinates at each
//! corner, groups of triangles sharing a material, and a skeleton of joints with keyframes.
//!
//! The groups become the primitives of one mesh (their names kept in its `extras.groups`), the materials glTF
//! materials (diffuse color with the transparency as alpha, emissive color, shininess as roughness, the texture as
//! the color image), and the joints nodes posed by Euler angles in radians applied X, then Y, then Z. Each vertex
//! moves with its joint, or with up to four joints by the weights the format's later versions add. Keyframes, in
//! seconds, turn and move a joint from where it stands at rest, in its rest frame (a move is turned by the rest
//! rotation). `MilkShape` is Y up and right-handed, as glTF is
//! (declared in the source info: the program's space is every file's); its units' length is unknown. The model's comment is
//! the source info's `comments`; the groups', materials' and joints' comments are skipped.

use std::collections::HashMap;

use serde_json::json;

use super::scene::{self, Animation, Channel, Interp, Node, Path as Key, Primitive, Scene, SceneMesh, Skin};
use crate::Result;
use crate::assets::Assets;
use crate::bytes::Bin;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::{Material, srgb_to_linear};
use crate::num::{f32_of, u32_of};
use crate::quat;
use crate::source::{Axis, Fact, Handedness};

const MAGIC: &[u8; 10] = b"MS3D000000";

/// A vertex: its position, joint (negative for none), and the extra joints and weights later versions add.
struct Vertex {
    position: [f32; 3],
    joint: i8,
    extra_joints: [i8; 3],
    extra_weights: [u8; 3],
}

/// A triangle: vertex indices, and a normal and texture coordinates at each corner.
struct Triangle {
    corners: [u16; 3],
    normals: [[f32; 3]; 3],
    uvs: [[f32; 2]; 3],
}

/// A group: its name, triangles (indices into the file's) and material (negative for none).
struct Group {
    name: String,
    triangles: Vec<u16>,
    material: i8,
}

/// A joint: its name, its parent's name, its rest rotation (Euler angles) and position, and its keys (time in
/// seconds, then value).
struct Joint {
    name: String,
    parent: String,
    rotation: [f32; 3],
    position: [f32; 3],
    turns: Vec<(f32, [f32; 3])>,
    moves: Vec<(f32, [f32; 3])>,
}

/// A material as the file has it.
struct Mat {
    name: String,
    diffuse: [f32; 4],
    emissive: [f32; 4],
    shininess: f32,
    transparency: f32,
    texture: String,
    alphamap: String,
}

/// What a file holds.
struct Ms3d {
    version: i32,
    vertices: Vec<Vertex>,
    triangles: Vec<Triangle>,
    groups: Vec<Group>,
    materials: Vec<Mat>,
    fps: f32,
    frames: i32,
    joints: Vec<Joint>,
    /// From the model's extra data: how transparency is drawn (1 is cut out at `alpha_ref`).
    transparency_mode: i32,
    alpha_ref: f32,
    /// The extra weights' version, when the file has them.
    weights_version: Option<i32>,
    /// The model's comment, from the extra data.
    comment: Option<String>,
}

/// Reads a `.ms3d` file.
pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let bytes = input.read()?;
    let m = parse(&bytes)?;
    let stem = input
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut scene = Scene::default();
    let mut facts = serde_json::Map::new();

    // Materials.
    let dir = input.dir();
    let mut images = Images::new(!keep.image_bytes, dir);
    for mat in &m.materials {
        // Colors as the file has them are display values; glTF's are linear.
        let unit = |c: f32, default: f32| if c.is_finite() { c.clamp(0.0, 1.0) } else { default };
        let alpha = unit(mat.transparency, 1.0);
        let d = mat.diffuse.map(|c| srgb_to_linear(unit(c, 1.0)));
        let mut out = Material::plain(&mat.name, [d[0], d[1], d[2], alpha]);
        out.emissive = std::array::from_fn(|k| srgb_to_linear(unit(mat.emissive[k], 0.0)));
        out.roughness = 1.0 - (unit(mat.shininess / 128.0, 0.0)).sqrt();
        if keep.textures && !mat.texture.is_empty() {
            out.color_image = images.find(assets, &[&mat.texture], &[dir]);
            // The texture tints the glow as it does the rest of the lighting.
            if out.emissive.iter().any(|&c| c > 0.0) {
                out.emissive_image = out.color_image;
            }
        }
        // Cut out where the texture's alpha falls below the model's alpha reference.
        let textured = !mat.texture.is_empty() || !mat.alphamap.is_empty();
        if textured && m.transparency_mode == 1 && m.alpha_ref.is_finite() {
            out.alpha_cutoff = Some(m.alpha_ref.clamp(0.0, 1.0));
        }
        scene.materials.push(out);
    }
    scene.images = images.list;

    // Joints.
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (i, j) in m.joints.iter().enumerate() {
        by_name.entry(j.name.as_str()).or_insert(i);
    }
    for j in &m.joints {
        let mut n = Node::named(j.name.clone(), by_name.get(j.parent.as_str()).copied());
        n.translation = j.position;
        n.rotation = quat::normalize(euler(j.rotation));
        scene.add(n);
    }
    let joints = m.joints.len();

    // The groups' triangles.
    let mut prims = Vec::new();
    let mut groups = Vec::new();
    let mut dropped = 0usize;
    let mut unbound = false;
    for g in &m.groups {
        let mut p = Primitive {
            material: usize::try_from(g.material).ok().filter(|&k| k < m.materials.len()),
            uvs: vec![Vec::new()],
            ..Primitive::default()
        };
        let mut links: Vec<Vec<(u32, f32)>> = Vec::new();
        let mut seen: HashMap<[u32; 6], u32> = HashMap::new();
        for &t in &g.triangles {
            let Some(t) = m.triangles.get(usize::from(t)) else {
                dropped += 1;
                continue;
            };
            if t.corners.iter().any(|&c| usize::from(c) >= m.vertices.len()) {
                dropped += 1;
                continue;
            }
            for k in 0..3 {
                let c = t.corners[k];
                let (nn, uv) = (t.normals[k], t.uvs[k]);
                let key = [
                    u32::from(c),
                    nn[0].to_bits(),
                    nn[1].to_bits(),
                    nn[2].to_bits(),
                    uv[0].to_bits(),
                    uv[1].to_bits(),
                ];
                let next = u32_of(p.positions.len());
                let index = *seen.entry(key).or_insert(next);
                if index == next {
                    let v = &m.vertices[usize::from(c)];
                    p.positions.push(v.position);
                    p.normals.push(nn);
                    p.uvs[0].push(uv);
                    let l = weights(v, joints, m.weights_version.is_some());
                    unbound |= l.is_empty();
                    links.push(l);
                }
                p.indices.push(index);
            }
        }
        if joints > 0 {
            // A vertex no joint moves stays where it is: bound to a joint of its own, after the file's.
            let width = links.iter().map(Vec::len).max().unwrap_or(0).max(1);
            p.influences = width;
            p.rest_joint = u32_of(joints);
            for l in &links {
                for k in 0..width {
                    let (j, w) = l.get(k).copied().unwrap_or((0, 0.0));
                    p.joints.push(j);
                    p.weights.push(w);
                }
            }
        }
        prims.push(p);
        groups.push(json!(g.name));
    }
    if dropped > 0 {
        facts.insert("dropped_triangles".into(), json!(dropped));
    }
    let has_mesh = prims.iter().any(|p| !p.indices.is_empty());
    if has_mesh {
        scene.meshes.push(SceneMesh {
            name: stem.clone(),
            prims,
            extras: Some(json!({ "groups": groups })),
            ..SceneMesh::default()
        });
        let mut node = Node::named(stem.clone(), None);
        node.mesh = Some(0);
        if joints > 0 {
            let mut world = scene.world();
            world.truncate(joints);
            let mut skin = Skin {
                name: stem.clone(),
                joints: (0..joints).collect(),
                inverse_binds: world.iter().map(scene::invert).collect(),
            };
            if unbound {
                let fixed = scene.add(Node::named("unbound", None));
                skin.joints.push(fixed);
                skin.inverse_binds.push(scene::IDENTITY);
            }
            scene.skins.push(skin);
            node.skin = Some(0);
        }
        scene.add(node);
    }

    // Keys are in the joint's rest frame, as MilkShape's own viewer plays them: the joint's local matrix is its
    // rest matrix times the key's (a turn, then a move). So a key's turn follows the rest rotation, and its move
    // is turned by the rest rotation before it is added to the rest position.
    let mut channels = Vec::new();
    for (i, j) in m.joints.iter().enumerate() {
        let rest = euler(j.rotation);
        if !j.turns.is_empty() {
            let mut values = Vec::with_capacity(j.turns.len() * 4);
            for (_, r) in &j.turns {
                values.extend(quat::normalize(quat::mul(rest, euler(*r))));
            }
            channels.push(Channel {
                node: i,
                path: Key::Rotation,
                times: j.turns.iter().map(|k| k.0.max(0.0)).collect(),
                values,
                interp: Interp::Linear,
            });
        }
        if !j.moves.is_empty() {
            let values = j
                .moves
                .iter()
                .flat_map(|(_, v)| {
                    let v = quat::rotate(rest, v.map(f64::from));
                    std::array::from_fn::<f32, 3, _>(|k| f32_of(f64::from(j.position[k]) + v[k]))
                })
                .collect();
            channels.push(Channel {
                node: i,
                path: Key::Translation,
                times: j.moves.iter().map(|k| k.0.max(0.0)).collect(),
                values,
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
    ensure!(!scene.nodes.is_empty(), Empty, "MS3D file has no triangles or joints");

    facts.insert("joints".into(), json!(joints));
    facts.insert("groups".into(), json!(m.groups.len()));
    let mut source = serde_json::Map::new();
    scene.info.format = Some("MS3D".into());
    scene.info.version = Some(m.version.to_string());
    scene.info.up_axis = Fact::Declared(Axis::PosY);
    scene.info.handedness = Fact::Declared(Handedness::Right);
    if m.fps.is_finite() && m.fps > 0.0 {
        scene.info.frame_rate = Fact::Declared(f64::from(m.fps));
    }
    scene.info.comments.extend(m.comment);
    source.insert("frames".into(), json!(m.frames));
    if let Some(v) = m.weights_version {
        source.insert("weightsVersion".into(), json!(v));
    }
    let alphamaps: Vec<&str> = m
        .materials
        .iter()
        .map(|x| x.alphamap.as_str())
        .filter(|a| !a.is_empty())
        .collect();
    if !alphamaps.is_empty() {
        source.insert("alphamaps".into(), json!(alphamaps));
    }
    scene.source = source;
    scene.facts = facts;
    scene.finish(keep)
}

/// The rotation Euler angles make: about X, then Y, then Z.
fn euler(r: [f32; 3]) -> [f64; 4] {
    quat::euler_xyz(r.map(f64::from))
}

/// A vertex's joints and weights. Without extra weights, its joint takes all; with them, its joint and the first
/// two extra joints take the three weights (out of 100, or of 255 when they sum past 100) and the third extra joint
/// what's left, unless all three weights are 0, when its joint takes all. Joints out of range are left out.
fn weights(v: &Vertex, joints: usize, extra: bool) -> Vec<(u32, f32)> {
    let valid = |j: i8| usize::try_from(j).ok().filter(|&j| j < joints).map(u32_of);
    if !extra || v.extra_weights == [0; 3] {
        return valid(v.joint).map(|j| vec![(j, 1.0)]).unwrap_or_default();
    }
    let sum: u32 = v.extra_weights.iter().map(|&w| u32::from(w)).sum();
    let scale = if sum > 100 { 255.0 } else { 100.0 };
    let w = v.extra_weights.map(|w| f32::from(w) / scale);
    let ids = [v.joint, v.extra_joints[0], v.extra_joints[1], v.extra_joints[2]];
    let ws = [w[0], w[1], w[2], (1.0 - w[0] - w[1] - w[2]).max(0.0)];
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(4);
    for (j, w) in ids.into_iter().zip(ws) {
        if let Some(j) = valid(j)
            && w > 0.0
        {
            match out.iter_mut().find(|l| l.0 == j) {
                Some(l) => l.1 += w,
                None => out.push((j, w)),
            }
        }
    }
    out
}

/// A fixed-size name field as text.
fn field(b: &mut Bin<'_>, n: usize) -> Result<String> {
    Ok(crate::bytes::name(b.fixed_str(n)?).trim_start().to_owned())
}

/// Reads the file.
fn parse(data: &[u8]) -> Result<Ms3d> {
    let mut b = Bin::new(data, "MS3D");
    ensure!(
        data.len() >= 14 && &data[..10] == MAGIC,
        UnknownFormat,
        "isn't a MilkShape 3D file: it doesn't start with MS3D000000"
    );
    b.skip(10)?;
    let version = b.i32()?;
    ensure!(
        (3..=4).contains(&version),
        Unsupported,
        "MS3D version {version} isn't one this reader knows (3 or 4)"
    );

    let n = b.u16()?;
    let n = b.count(n.into(), 15)?;
    let mut vertices = Vec::with_capacity(n);
    for _ in 0..n {
        b.skip(1)?;
        let position = b.vec3()?;
        let joint = b.i8()?;
        b.skip(1)?;
        vertices.push(Vertex {
            position,
            joint,
            extra_joints: [-1; 3],
            extra_weights: [0; 3],
        });
    }

    let n = b.u16()?;
    let n = b.count(n.into(), 70)?;
    let mut triangles = Vec::with_capacity(n);
    for _ in 0..n {
        b.skip(2)?;
        let corners = [b.u16()?, b.u16()?, b.u16()?];
        let normals = [b.vec3()?, b.vec3()?, b.vec3()?];
        let s = b.vec3()?;
        let t = b.vec3()?;
        b.skip(2)?;
        triangles.push(Triangle {
            corners,
            normals,
            uvs: [[s[0], t[0]], [s[1], t[1]], [s[2], t[2]]],
        });
    }

    let n = b.u16()?;
    let n = b.count(n.into(), 36)?;
    let mut groups = Vec::with_capacity(n);
    for _ in 0..n {
        b.skip(1)?;
        let name = field(&mut b, 32)?;
        let k = b.u16()?;
        let k = b.count(k.into(), 2)?;
        let mut tris = Vec::with_capacity(k);
        for _ in 0..k {
            tris.push(b.u16()?);
        }
        let material = b.i8()?;
        groups.push(Group {
            name,
            triangles: tris,
            material,
        });
    }

    let mut m = Ms3d {
        version,
        vertices,
        triangles,
        groups,
        materials: Vec::new(),
        fps: 0.0,
        frames: 0,
        joints: Vec::new(),
        transparency_mode: 0,
        alpha_ref: 0.0,
        weights_version: None,
        comment: None,
    };
    // A file may end after its groups or its materials.
    if b.left() == 0 {
        return Ok(m);
    }
    let n = b.u16()?;
    let n = b.count(n.into(), 361)?;
    for _ in 0..n {
        let name = field(&mut b, 32)?;
        let _ambient = b.vec4()?;
        let diffuse = b.vec4()?;
        let _specular = b.vec4()?;
        let emissive = b.vec4()?;
        let shininess = b.f32()?;
        let transparency = b.f32()?;
        b.skip(1)?;
        let texture = name_path(&mut b)?;
        let alphamap = name_path(&mut b)?;
        m.materials.push(Mat {
            name,
            diffuse,
            emissive,
            shininess,
            transparency,
            texture,
            alphamap,
        });
    }
    if b.left() == 0 {
        return Ok(m);
    }
    m.fps = b.f32()?;
    let _current = b.f32()?;
    m.frames = b.i32()?;
    let n = b.u16()?;
    let n = b.count(n.into(), 93)?;
    for _ in 0..n {
        b.skip(1)?;
        let name = field(&mut b, 32)?;
        let parent = field(&mut b, 32)?;
        let rotation = b.vec3()?;
        let position = b.vec3()?;
        let nr = b.u16()?;
        let np = b.u16()?;
        let nr = b.count(nr.into(), 16)?;
        let mut turns = Vec::with_capacity(nr);
        for _ in 0..nr {
            turns.push((b.f32()?, b.vec3()?));
        }
        let np = b.count(np.into(), 16)?;
        let mut moves = Vec::with_capacity(np);
        for _ in 0..np {
            moves.push((b.f32()?, b.vec3()?));
        }
        m.joints.push(Joint {
            name,
            parent,
            rotation,
            position,
            turns,
            moves,
        });
    }
    // The optional parts after: a broken one ends the reading without spoiling what came before.
    let _ = extras(&mut b, &mut m);
    Ok(m)
}

/// A 128-byte path field, with `\` read as `/`.
fn name_path(b: &mut Bin<'_>) -> Result<String> {
    Ok(field(b, 128)?.replace('\\', "/"))
}

/// The optional parts after the joints: comments, the extra weights, the joints' colors and the model's
/// transparency settings.
fn extras(b: &mut Bin<'_>, m: &mut Ms3d) -> Result<()> {
    if b.left() < 4 || b.i32()? != 1 {
        return Ok(());
    }
    // Comments on groups, materials and joints, then the model's.
    for _ in 0..3 {
        let n = b.i32()?;
        let n = b.count(u64::try_from(n).unwrap_or(u64::MAX), 8)?;
        for _ in 0..n {
            b.skip(4)?;
            let len = b.i32()?;
            b.skip(usize::try_from(len).unwrap_or(usize::MAX))?;
        }
    }
    if b.i32()? != 0 {
        let len = b.i32()?;
        let text = b.bytes(usize::try_from(len).unwrap_or(usize::MAX))?;
        let text = scene::text(text.split(|&c| c == 0).next().unwrap_or_default());
        let text = text.trim();
        if !text.is_empty() {
            m.comment = Some(text.to_owned());
        }
    }
    if b.left() < 4 {
        return Ok(());
    }
    let version = b.i32()?;
    if !(1..=3).contains(&version) {
        return Ok(());
    }
    let stride = 6 + 4 * usize::try_from(version - 1).unwrap_or(0);
    b.count(m.vertices.len() as u64, stride)?;
    for v in &mut m.vertices {
        v.extra_joints = [b.i8()?, b.i8()?, b.i8()?];
        v.extra_weights = [b.u8()?, b.u8()?, b.u8()?];
        b.skip(stride - 6)?;
    }
    m.weights_version = Some(version);
    // The joints' colors.
    if b.left() < 4 {
        return Ok(());
    }
    let version = b.i32()?;
    if version == 1 || version == 2 {
        b.skip(m.joints.len().saturating_mul(12))?;
    } else {
        return Ok(());
    }
    // The model's joint size, transparency mode and alpha reference.
    if b.left() < 4 || b.i32()? != 1 {
        return Ok(());
    }
    let _joint_size = b.f32()?;
    m.transparency_mode = b.i32()?;
    m.alpha_ref = b.f32()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;

    /// Builds a file by hand, header first.
    struct Writer {
        out: Vec<u8>,
    }

    impl Writer {
        fn new() -> Self {
            let mut out = MAGIC.to_vec();
            out.extend(4i32.to_le_bytes());
            Self { out }
        }
        fn u16(&mut self, v: u16) -> &mut Self {
            self.out.extend(v.to_le_bytes());
            self
        }
        fn i32(&mut self, v: i32) -> &mut Self {
            self.out.extend(v.to_le_bytes());
            self
        }
        fn f(&mut self, v: &[f32]) -> &mut Self {
            for x in v {
                self.out.extend(x.to_le_bytes());
            }
            self
        }
        fn s(&mut self, s: &str, n: usize) -> &mut Self {
            let mut b = s.as_bytes().to_vec();
            b.resize(n, 0);
            self.out.extend(b);
            self
        }
    }

    /// A triangle bound to joint 0, a group and a material, a parent and a child joint, with keys.
    fn sample(extra_weights: bool) -> Vec<u8> {
        sample_with(extra_weights, [0.0; 3], [0.0, 2.0, 0.0])
    }

    /// [`sample`] with the hip's rest rotation (Euler angles) and the move of its one position key.
    fn sample_with(extra_weights: bool, hip_rest: [f32; 3], hip_move: [f32; 3]) -> Vec<u8> {
        let mut w = Writer::new();
        w.u16(3);
        for (p, j) in [([0.0, 0.0, 0.0], 0i8), ([1.0, 0.0, 0.0], 1), ([0.0, 1.0, 0.0], -1)] {
            w.out.push(0);
            w.f(&p);
            w.out.push(j.to_le_bytes()[0]);
            w.out.push(0);
        }
        w.u16(1).u16(0).u16(0).u16(1).u16(2);
        for _ in 0..3 {
            w.f(&[0.0, 0.0, 1.0]);
        }
        w.f(&[0.0, 1.0, 0.0]).f(&[0.0, 0.0, 1.0]);
        w.out.extend([1, 0]);
        w.u16(1);
        w.out.push(0);
        w.s("body", 32).u16(1).u16(0);
        w.out.push(0);
        w.u16(1);
        w.s("skin", 32)
            .f(&[0.2, 0.2, 0.2, 1.0])
            .f(&[1.0, 0.5, 0.25, 1.0])
            .f(&[0.0; 4])
            .f(&[0.0; 4])
            .f(&[0.0, 0.5]);
        w.out.push(0);
        w.s("", 128).s("", 128);
        w.f(&[24.0, 0.0]).i32(10);
        w.u16(2);
        w.out.push(0);
        w.s("hip", 32).s("", 32).f(&hip_rest).f(&[0.0, 1.0, 0.0]).u16(2).u16(1);
        w.f(&[0.0, 0.0, 0.0, 0.0])
            .f(&[1.0, 0.0, 0.0, std::f32::consts::FRAC_PI_2]);
        w.f(&[0.5]).f(&hip_move);
        w.out.push(0);
        w.s("knee", 32)
            .s("hip", 32)
            .f(&[0.0; 3])
            .f(&[1.0, 0.0, 0.0])
            .u16(0)
            .u16(0);
        if extra_weights {
            w.i32(1).i32(0).i32(0).i32(0).i32(0);
            w.i32(1);
            // Vertex 0: joint 0 at 50, joint 1 at 50.
            w.out.extend([1, 255, 255, 50, 50, 0]);
            w.out.extend([255, 255, 255, 0, 0, 0]);
            w.out.extend([255, 255, 255, 0, 0, 0]);
        }
        w.out
    }

    fn read(bytes: &[u8]) -> Result<Document> {
        load(super::super::Input::memory(bytes, "t.ms3d"), &Memory::new(), Keep::ALL)
    }

    #[test]
    fn a_skinned_triangle_with_a_material_and_keys() {
        let doc = read(&sample(false)).unwrap();
        let j = &doc.json();
        assert_eq!(doc.facts["joints"], json!(2));
        assert_eq!(doc.facts["triangles"], json!(1));
        assert_eq!(j["materials"][0]["name"], "skin");
        assert_eq!(j["meshes"][0]["extras"]["groups"], json!(["body"]));
        // Two joints, and the one the unbound vertex is held by.
        assert_eq!(j["skins"][0]["joints"].as_array().unwrap().len(), 3);
        let rig = doc.rig().unwrap();
        let knee = rig.find("knee").unwrap();
        assert_eq!(rig.nodes[knee].parent, rig.find("hip"));
        let a = &rig.animations[0];
        assert_eq!(a.channels.len(), 2);
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(mesh.prims[0].positions.len(), 3);
        // Texture coordinates as they are: the file's origin is top-left.
        assert!(mesh.prims[0].uvs.iter().any(|uv| uv.iter().all(|&c| c.abs() < 1e-9)));
    }

    #[test]
    fn a_position_key_moves_in_the_joints_rest_frame() {
        let moved = |rest: [f32; 3], by: [f32; 3]| {
            let doc = read(&sample_with(false, rest, by)).unwrap();
            let rig = doc.rig().unwrap();
            let hip = rig.find("hip").unwrap();
            let c = rig.animations[0]
                .channels
                .iter()
                .find(|c| c.node == hip && c.property == crate::rig::Property::Translation)
                .unwrap();
            [c.values[0], c.values[1], c.values[2]]
        };
        let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        // Unturned at rest, the move is added to the rest position (0, 1, 0) as it is.
        assert!(near(moved([0.0; 3], [1.0, 0.0, 0.0]), [1.0, 1.0, 0.0]));
        // Turned a quarter about Z at rest, the joint's own X is the parent's Y.
        let quarter = std::f32::consts::FRAC_PI_2;
        let got = moved([0.0, 0.0, quarter], [1.0, 0.0, 0.0]);
        assert!(near(got, [0.0, 2.0, 0.0]), "{got:?}");
        // And a quarter about X takes its Y to the parent's Z.
        let got = moved([quarter, 0.0, 0.0], [0.0, 3.0, 0.0]);
        assert!(near(got, [0.0, 1.0, 3.0]), "{got:?}");
    }

    #[test]
    fn extra_weights_split_a_vertex() {
        let doc = read(&sample(true)).unwrap();
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        let p = &mesh.prims[0];
        let w = p.skin_weights[0];
        assert!((w[0] - 0.5).abs() < 1e-3 && (w[1] - 0.5).abs() < 1e-3, "{w:?}");
        assert_eq!(doc.json()["asset"]["extras"]["source"]["weightsVersion"], json!(1));
    }

    #[test]
    fn source_info_has_the_rate_and_the_model_comment() {
        let mut w = Writer { out: sample(false) };
        w.i32(1).i32(0).i32(0).i32(0).i32(1).i32(12);
        w.out.extend(b"made by hand");
        let doc = read(&w.out).unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("MS3D"));
        assert_eq!(info.version.as_deref(), Some("4"));
        assert_eq!(info.frame_rate, Fact::Declared(24.0));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosY));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(info.comments, ["made by hand"]);
        assert_eq!(info.converted, crate::source::Converted::default());
    }

    #[test]
    fn weights_follow_the_format() {
        let v = |j, e, w| Vertex {
            position: [0.0; 3],
            joint: j,
            extra_joints: e,
            extra_weights: w,
        };
        assert_eq!(weights(&v(1, [-1; 3], [0; 3]), 2, true), vec![(1, 1.0)]);
        assert_eq!(weights(&v(5, [-1; 3], [0; 3]), 2, true), vec![]);
        let l = weights(&v(0, [1, -1, 1], [60, 0, 0]), 2, true);
        assert_eq!(l.len(), 2);
        assert!((l[0].1 - 0.6).abs() < 1e-6 && (l[1].1 - 0.4).abs() < 1e-6);
        let l = weights(&v(0, [1, -1, -1], [128, 127, 0]), 2, true);
        assert!((l[0].1 - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn cut_short_and_broken_files() {
        let good = sample(true);
        for k in 0..good.len() {
            let _ = read(&good[..k]);
        }
        assert!(read(b"MS3D000000").unwrap_err().to_string().contains("isn't"));
        assert!(read(b"not a model at all").unwrap_err().to_string().contains("isn't"));
        let mut v = good.clone();
        v[10] = 9;
        assert!(read(&v).unwrap_err().to_string().contains("version"));
        // A vertex count past the end.
        let mut v = good[..14].to_vec();
        v.extend([0xFF, 0xFF]);
        assert!(read(&v).unwrap_err().to_string().contains("more than it holds"));
        // Triangle and vertex indices out of range are dropped.
        let mut v = good.clone();
        let tri = 14 + 2 + 45 + 2 + 2;
        v[tri..tri + 2].copy_from_slice(&9u16.to_le_bytes());
        let doc = read(&v).unwrap();
        assert_eq!(doc.facts["dropped_triangles"], json!(1));
    }

    #[test]
    fn a_joint_that_is_its_own_parent_does_not_hang() {
        let mut v = sample(false);
        let at = v.windows(4).rposition(|w| w == b"hip\0").unwrap();
        v[at..at + 4].copy_from_slice(b"knee");
        let doc = read(&v).unwrap();
        assert!(doc.rig().is_ok());
    }
}
