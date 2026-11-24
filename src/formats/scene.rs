//! A scene the game, character and motion capture readers fill in, then write as a glTF document: nodes with their
//! transforms, meshes with any number of texture coordinate, color and joint sets, morph targets, skins and
//! animations.
//!
//! A reader builds the scene in its format's own coordinates, then converts it whole: [`Scene::mirror_z`] for a
//! left-handed source (positions, normals, transforms, inverse bind matrices and keys mirrored, and triangles wound
//! the other way), [`Scene::rotate_roots`] for a source that isn't Y up (only the root nodes and their keys turn,
//! so skins and morph targets stay as they are), and [`Scene::scale`] for a source in other units.
//!
//! [`Scene::finish`] writes what glTF requires and the validator checks: unit normals and rotations, positions'
//! bounds, joints and weights merged, sorted, cut to sixteen and summing to 1, key times strictly increasing,
//! quaternion keys on one side, and every skin's joints under one root. A node can't be its own ancestor: a parent
//! link that would close a loop is cut.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::Result;
use crate::document::{Builder, Document, Keep, Part};
use crate::images::Source;
use crate::mesh::Material;
use crate::num::{f32_of, u32_of};
use crate::source::SourceInfo;

/// The most influences a vertex keeps: four sets of joints and weights.
pub(crate) const MAX_INFLUENCES: usize = 16;

/// A node: a name, a parent, a transform, and what it carries.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub name: String,
    pub parent: Option<usize>,
    pub translation: [f32; 3],
    /// A unit quaternion, `[x, y, z, w]`.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    pub extras: Option<Value>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            name: String::new(),
            parent: None,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            mesh: None,
            skin: None,
            extras: None,
        }
    }
}

impl Node {
    /// A node named `name` under `parent`, at rest.
    pub fn named(name: impl Into<String>, parent: Option<usize>) -> Self {
        Self {
            name: name.into(),
            parent,
            ..Self::default()
        }
    }

    /// Its local transform, column-major.
    pub fn matrix(&self) -> M4 {
        trs(self.translation, self.rotation, self.scale)
    }
}

/// A morph target: position offsets, and normal offsets when the source has them, one per vertex of its primitive.
#[derive(Debug, Clone, Default)]
pub(crate) struct Target {
    pub positions: Vec<[f32; 3]>,
    /// Empty when the target doesn't move the normals.
    pub normals: Vec<[f32; 3]>,
}

/// One material's triangles (or points) and their vertices.
#[derive(Debug, Clone, Default)]
pub(crate) struct Primitive {
    pub positions: Vec<[f32; 3]>,
    /// Empty, or one per vertex; zero or broken normals are replaced by their faces' normal.
    pub normals: Vec<[f32; 3]>,
    /// Empty, or one per vertex: `[x, y, z, w]` with `w` the bitangent's sign. Written only beside normals.
    pub tangents: Vec<[f32; 4]>,
    /// Texture coordinate sets, glTF convention (origin top-left), each empty or one per vertex.
    pub uvs: Vec<Vec<[f32; 2]>>,
    /// Vertex color sets, linear RGBA, each empty or one per vertex.
    pub colors: Vec<Vec<[f32; 4]>>,
    /// `influences` joints (indices into the skin's joints) and weights per vertex, laid end to end; empty when
    /// the primitive isn't skinned. Zero weights are padding.
    pub joints: Vec<u32>,
    pub weights: Vec<f32>,
    pub influences: usize,
    /// The joint a vertex no joint moves is bound to.
    pub rest_joint: u32,
    /// Triangles, three indices each (or points, one each, when `points` is set, or line segments, two each,
    /// when `lines` is set).
    pub indices: Vec<u32>,
    pub points: bool,
    pub lines: bool,
    pub material: Option<usize>,
    pub targets: Vec<Target>,
}

/// A mesh: primitives sharing morph targets.
#[derive(Debug, Clone, Default)]
pub(crate) struct SceneMesh {
    pub name: String,
    pub prims: Vec<Primitive>,
    /// The morph targets' default weights; empty for all zero.
    pub weights: Vec<f32>,
    /// The morph targets' names, written as the mesh's `extras.targetNames`.
    pub target_names: Vec<String>,
    pub extras: Option<Value>,
}

/// A skin: its joints (nodes) and the matrix that takes the mesh to each joint's space where it was bound.
#[derive(Debug, Clone, Default)]
pub(crate) struct Skin {
    pub name: String,
    pub joints: Vec<usize>,
    /// Column-major, one per joint.
    pub inverse_binds: Vec<M4>,
}

/// What an animation channel sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Path {
    Translation,
    Rotation,
    Scale,
    Weights,
}

/// How values between keys are found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Interp {
    Linear,
}

/// One animated property of one node.
#[derive(Debug, Clone)]
pub(crate) struct Channel {
    pub node: usize,
    pub path: Path,
    /// Seconds.
    pub times: Vec<f32>,
    /// A value per key: 3 floats, 4 for a rotation, or the target count for weights.
    pub values: Vec<f32>,
    pub interp: Interp,
}

/// A named set of channels.
#[derive(Debug, Clone, Default)]
pub(crate) struct Animation {
    pub name: String,
    pub channels: Vec<Channel>,
    pub extras: Option<Value>,
}

/// The whole scene.
#[derive(Default)]
pub(crate) struct Scene {
    pub nodes: Vec<Node>,
    pub meshes: Vec<SceneMesh>,
    pub materials: Vec<Material>,
    pub images: Vec<Source>,
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
    /// The format's own details, written as `asset.extras.source` with [`Scene::info`].
    pub source: Map<String, Value>,
    /// What the file says about itself (its units, up axis, author...), written into `asset.extras.source`. What
    /// [`Scene::scale`], [`Scene::rotate_roots`] and [`Scene::mirror_z`] convert is marked converted.
    pub info: SourceInfo,
    pub facts: Map<String, Value>,
}

/// A 4×4 matrix, column-major.
pub(crate) type M4 = [f64; 16];

pub(crate) const IDENTITY: M4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

impl Scene {
    /// Adds a node; returns its index.
    pub fn add(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// Each node's world transform at rest (a parent link that would close a loop counts as none).
    pub fn world(&self) -> Vec<M4> {
        let parents = self.parents();
        let mut out = vec![None; self.nodes.len()];
        for i in 0..self.nodes.len() {
            let mut chain = vec![i];
            while let Some(p) = parents[*chain.last().unwrap_or(&i)] {
                if out[p].is_some() {
                    break;
                }
                chain.push(p);
            }
            for &n in chain.iter().rev() {
                if out[n].is_none() {
                    let local = self.nodes[n].matrix();
                    out[n] = Some(match parents[n].and_then(|p| out[p]) {
                        Some(pw) => mul(&pw, &local),
                        None => local,
                    });
                }
            }
        }
        out.into_iter().map(|m| m.unwrap_or(IDENTITY)).collect()
    }

    /// Each node's parent, with links out of range or closing a loop cut.
    fn parents(&self) -> Vec<Option<usize>> {
        let n = self.nodes.len();
        let mut parents: Vec<Option<usize>> = self.nodes.iter().map(|x| x.parent.filter(|&p| p < n)).collect();
        // 0 unvisited, 1 on the current path, 2 done.
        let mut state = vec![0u8; n];
        for start in 0..n {
            let mut path = Vec::new();
            let mut at = start;
            loop {
                if state[at] == 2 {
                    break;
                }
                if state[at] == 1 {
                    // `at` is its own ancestor: cut the link that reached it again.
                    if let Some(&last) = path.last() {
                        parents[last] = None;
                    }
                    break;
                }
                state[at] = 1;
                path.push(at);
                match parents[at] {
                    Some(p) => at = p,
                    None => break,
                }
            }
            for &p in &path {
                state[p] = 2;
            }
        }
        parents
    }

    /// Mirrors the scene through the XY plane (Z negated), as a left-handed source needs: vertex data, node
    /// transforms, inverse bind matrices and keys, with every triangle wound the other way so faces keep facing out.
    pub fn mirror_z(&mut self) {
        self.info.converted.handedness = true;
        let v = |p: &mut [f32; 3]| p[2] = -p[2];
        for m in &mut self.meshes {
            for p in &mut m.prims {
                p.positions.iter_mut().for_each(v);
                p.normals.iter_mut().for_each(v);
                for t in &mut p.tangents {
                    t[2] = -t[2];
                    t[3] = -t[3];
                }
                for t in &mut p.targets {
                    t.positions.iter_mut().for_each(v);
                    t.normals.iter_mut().for_each(v);
                }
                if !p.points && !p.lines {
                    for t in p.indices.chunks_exact_mut(3) {
                        t.swap(1, 2);
                    }
                }
            }
        }
        for n in &mut self.nodes {
            v(&mut n.translation);
            n.rotation = mirror_quat(n.rotation);
        }
        for s in &mut self.skins {
            for m in &mut s.inverse_binds {
                *m = mirror_matrix(m);
            }
        }
        for a in &mut self.animations {
            for c in &mut a.channels {
                match c.path {
                    Path::Translation => c.values.chunks_exact_mut(3).for_each(|k| k[2] = -k[2]),
                    Path::Rotation => c.values.chunks_exact_mut(4).for_each(|k| {
                        k[0] = -k[0];
                        k[1] = -k[1];
                    }),
                    Path::Scale | Path::Weights => {}
                }
            }
        }
    }

    /// Scales lengths by `s`: positions, morph offsets, translations, keys and inverse bind matrices.
    pub fn scale(&mut self, s: f32) {
        if (s - 1.0).abs() > f32::EPSILON {
            self.info.converted.units = true;
        }
        let v = |p: &mut [f32; 3]| *p = p.map(|c| c * s);
        for m in &mut self.meshes {
            for p in &mut m.prims {
                p.positions.iter_mut().for_each(v);
                for t in &mut p.targets {
                    t.positions.iter_mut().for_each(v);
                }
            }
        }
        for n in &mut self.nodes {
            v(&mut n.translation);
        }
        let sd = f64::from(s);
        for sk in &mut self.skins {
            for m in &mut sk.inverse_binds {
                m[12] *= sd;
                m[13] *= sd;
                m[14] *= sd;
            }
        }
        for a in &mut self.animations {
            for c in &mut a.channels {
                if c.path == Path::Translation {
                    c.values.iter_mut().for_each(|x| *x *= s);
                }
            }
        }
    }

    /// Turns the whole scene by unit quaternion `q`: each root node's transform and keys are turned, and what hangs
    /// from them follows. Skinned vertices follow their joints, so nothing else changes.
    pub fn rotate_roots(&mut self, q: [f32; 4]) {
        self.info.converted.up_axis = true;
        let parents = self.parents();
        let qd = q.map(f64::from);
        let roots: Vec<bool> = parents.iter().map(Option::is_none).collect();
        for (n, &root) in self.nodes.iter_mut().zip(&roots) {
            // A skinned mesh's node doesn't place it, so it stays as it is.
            if !root || n.skin.is_some() {
                continue;
            }
            n.translation = rotate(qd, n.translation.map(f64::from)).map(f32_of);
            n.rotation = normalize(quat_mul(qd, n.rotation.map(f64::from)));
        }
        for a in &mut self.animations {
            for c in &mut a.channels {
                if c.node >= roots.len() || !roots[c.node] || self.nodes[c.node].skin.is_some() {
                    continue;
                }
                match c.path {
                    Path::Translation => c.values.chunks_exact_mut(3).for_each(|k| {
                        let r = rotate(qd, [k[0], k[1], k[2]].map(f64::from)).map(f32_of);
                        k.copy_from_slice(&r);
                    }),
                    Path::Rotation => c.values.chunks_exact_mut(4).for_each(|k| {
                        let r = normalize(quat_mul(qd, [k[0], k[1], k[2], k[3]].map(f64::from)));
                        k.copy_from_slice(&r);
                    }),
                    Path::Scale | Path::Weights => {}
                }
            }
        }
    }

    /// The triangles the scene draws.
    pub fn triangles(&self) -> usize {
        self.meshes
            .iter()
            .flat_map(|m| &m.prims)
            .filter(|p| !p.points && !p.lines)
            .map(|p| p.indices.len() / 3)
            .sum()
    }

    /// Writes the scene as a glTF document, keeping what `keep` says; more triangles than it allows is an error.
    pub fn finish(mut self, keep: Keep) -> Result<Document<'static>> {
        let limit = keep.triangles();
        let tris = self.triangles();
        ensure!(tris <= limit, "too many triangles to convert (more than {limit})");
        let parents = self.parents();
        let mut b = Builder::default();
        let mut json = Map::new();

        // Every skin's joints under one root: the tops of their trees gathered under a new node when there are
        // several.
        let mut extra_root = None;
        if keep.skins {
            let mut tops = Vec::new();
            for s in &self.skins {
                let mut mine: Vec<usize> = Vec::new();
                for &j in &s.joints {
                    let mut at = j;
                    let mut steps = 0;
                    while let Some(p) = parents.get(at).copied().flatten() {
                        at = p;
                        steps += 1;
                        if steps > parents.len() {
                            break;
                        }
                    }
                    if !mine.contains(&at) {
                        mine.push(at);
                    }
                }
                if mine.len() > 1 {
                    for t in mine {
                        if !tops.contains(&t) {
                            tops.push(t);
                        }
                    }
                }
            }
            if !tops.is_empty() {
                extra_root = Some((self.nodes.len(), tops));
            }
        }

        let n = self.nodes.len();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        let mut roots = Vec::new();
        for (i, p) in parents.iter().enumerate() {
            match p {
                Some(p) => children[*p].push(i),
                None => match &extra_root {
                    Some((r, tops)) if tops.contains(&i) => children[*r].push(i),
                    _ => roots.push(i),
                },
            }
        }
        if let Some((r, _)) = &extra_root {
            roots.push(*r);
        }

        // Meshes.
        let joint_counts: Vec<Option<usize>> = (0..self.meshes.len())
            .map(|m| {
                let skin = self.nodes.iter().find(|x| x.mesh == Some(m)).and_then(|x| x.skin);
                skin.and_then(|s| self.skins.get(s)).map(|s| s.joints.len())
            })
            .collect();
        let material_count = self.materials.len();
        let mut meshes_json = Vec::new();
        let mut mesh_index = vec![None; self.meshes.len()];
        // Joints and weights are written after every mesh's other data, so a read without skins numbers the
        // rest as a full read does.
        let mut skinning: Vec<(usize, usize, Skinning)> = Vec::new();
        // Morph targets too, before the joints and weights, so a read without them numbers every mesh's own data
        // as a full read does.
        let mut morphing: Vec<(usize, usize, usize, Vec<Target>)> = Vec::new();
        for (mi, m) in std::mem::take(&mut self.meshes).into_iter().enumerate() {
            let joints = if keep.skins { joint_counts[mi] } else { None };
            let targets = m.prims.iter().map(|p| p.targets.len()).max().unwrap_or(0);
            let morphs = keep.morph_targets && targets > 0 && m.prims.iter().all(|p| p.targets.len() == targets);
            let mut prims = Vec::new();
            for mut p in m.prims {
                let targets = if morphs {
                    std::mem::take(&mut p.targets)
                } else {
                    Vec::new()
                };
                let n = p.positions.len();
                if let Some((v, skin)) = write_prim(&mut b, p, joints, material_count) {
                    if let Some(skin) = skin {
                        skinning.push((meshes_json.len(), prims.len(), skin));
                    }
                    if !targets.is_empty() {
                        morphing.push((meshes_json.len(), prims.len(), n, targets));
                    }
                    prims.push(v);
                }
            }
            if prims.is_empty() {
                continue;
            }
            let mut mj = json!({ "primitives": prims });
            if !m.name.is_empty() {
                mj["name"] = json!(m.name);
            }
            if morphs {
                let w: Vec<f32> = (0..targets)
                    .map(|k| m.weights.get(k).copied().filter(|w| w.is_finite()).unwrap_or(0.0))
                    .collect();
                if w.iter().any(|&w| w != 0.0) {
                    mj["weights"] = json!(w);
                }
                if m.target_names.len() == targets {
                    mj["extras"] = json!({ "targetNames": m.target_names });
                }
            }
            if let Some(e) = m.extras {
                match mj.get_mut("extras").and_then(Value::as_object_mut) {
                    Some(x) => {
                        if let Value::Object(e) = e {
                            x.extend(e);
                        }
                    }
                    None => mj["extras"] = e,
                }
            }
            mesh_index[mi] = Some(meshes_json.len());
            meshes_json.push(mj);
        }
        for (mi, pi, n, targets) in morphing {
            let targets: Vec<Value> = targets
                .into_iter()
                .map(|t| {
                    let mut tj = json!({ "POSITION": sparse_vec3(&mut b, t.positions, n, true) });
                    if t.normals.len() == n {
                        tj["NORMAL"] = json!(sparse_vec3(&mut b, t.normals, n, false));
                    }
                    tj
                })
                .collect();
            meshes_json[mi]["primitives"][pi]["targets"] = json!(targets);
        }
        for (mi, pi, skin) in skinning {
            let attributes = &mut meshes_json[mi]["primitives"][pi]["attributes"];
            for (key, a) in skin.write(&mut b) {
                attributes[key] = json!(a);
            }
        }
        let morph_counts: Vec<usize> = meshes_json
            .iter()
            .map(|m| {
                m["primitives"][0]
                    .get("targets")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len)
            })
            .collect();

        // Skins.
        let mut skins_json = Vec::new();
        let mut skin_index = vec![None; self.skins.len()];
        if keep.skins {
            for (si, s) in self.skins.iter().enumerate() {
                if s.joints.is_empty() || s.joints.iter().any(|&j| j >= n) {
                    continue;
                }
                let mut ibm = Vec::with_capacity(s.joints.len() * 16);
                for k in 0..s.joints.len() {
                    let mut m = s.inverse_binds.get(k).copied().unwrap_or(IDENTITY);
                    if m.iter().any(|v| !v.is_finite()) {
                        m = IDENTITY;
                    }
                    m[3] = 0.0;
                    m[7] = 0.0;
                    m[11] = 0.0;
                    m[15] = 1.0;
                    ibm.extend(m.map(f32_of));
                }
                let a = b.accessor(Part::f32(ibm), 5126, s.joints.len(), "MAT4", None);
                let mut sj = json!({ "joints": s.joints, "inverseBindMatrices": a });
                if !s.name.is_empty() {
                    sj["name"] = json!(s.name);
                }
                skin_index[si] = Some(skins_json.len());
                skins_json.push(sj);
            }
        }

        // Nodes.
        let mut nodes_json = Vec::with_capacity(n + 1);
        for (i, node) in self.nodes.iter().enumerate() {
            let mut nj = json!({});
            if !node.name.is_empty() {
                nj["name"] = json!(node.name);
            }
            let mesh = node.mesh.and_then(|m| mesh_index.get(m).copied().flatten());
            let skin = node.skin.and_then(|s| skin_index.get(s).copied().flatten());
            let skinned = mesh.is_some() && skin.is_some();
            if !skinned {
                let t = node.translation;
                if t.iter().all(|c| c.is_finite()) && t.iter().any(|&c| c != 0.0) {
                    nj["translation"] = json!(t);
                }
                let r = normalize(node.rotation.map(f64::from));
                #[allow(clippy::float_cmp, reason = "only the exact identity is left out")]
                if r != [0.0, 0.0, 0.0, 1.0] {
                    nj["rotation"] = json!(r);
                }
                let s = node.scale;
                #[allow(clippy::float_cmp, reason = "only the exact unit scale is left out")]
                if s.iter().all(|c| c.is_finite()) && s != [1.0; 3] {
                    nj["scale"] = json!(s);
                }
            }
            if let Some(m) = mesh {
                nj["mesh"] = json!(m);
                if let Some(s) = skin {
                    nj["skin"] = json!(s);
                }
            }
            if !children[i].is_empty() {
                nj["children"] = json!(children[i]);
            }
            if keep.extras
                && let Some(e) = &node.extras
            {
                nj["extras"] = e.clone();
            }
            nodes_json.push(nj);
        }
        if let Some((r, _)) = &extra_root {
            nodes_json.push(json!({ "name": "Skeleton", "children": children[*r] }));
        }

        // Materials and images, before the animations, so a read without animations numbers the images' views as
        // a full read does.
        let images = if keep.textures { self.images } else { Vec::new() };
        let mut materials = self.materials;
        if !keep.textures {
            for m in &mut materials {
                m.color_image = None;
                m.normal_image = None;
                m.metallic_roughness_image = None;
                m.occlusion_image = None;
                m.emissive_image = None;
            }
        }
        b.materials(materials, images, &mut json);

        // Animations.
        let mut animations_json = Vec::new();
        if keep.animations {
            // Each distinct set of times is written once, by a hash of their bits: channels usually share theirs.
            let mut inputs: HashMap<u64, Vec<(Vec<f32>, usize)>> = HashMap::new();
            // Every key is written to one buffer view, each accessor at its offset into it: one part and one view,
            // where a view per accessor would cost far more than the keys of a short channel.
            let mut data: Vec<f32> = Vec::new();
            let mut placed: Vec<usize> = Vec::new();
            let mut place = |b: &mut Builder, v: &[f32], count: usize, kind: &str| {
                b.accessors.push(json!({
                    "byteOffset": data.len() * 4, "componentType": 5126, "count": count, "type": kind
                }));
                data.extend_from_slice(v);
                placed.push(b.accessors.len() - 1);
                b.accessors.len() - 1
            };
            for a in std::mem::take(&mut self.animations) {
                let mut channels = Vec::new();
                let mut samplers = Vec::new();
                for c in a.channels {
                    let width = match c.path {
                        Path::Translation | Path::Scale => 3,
                        Path::Rotation => 4,
                        Path::Weights => {
                            let Some(m) = self.nodes.get(c.node).and_then(|x| x.mesh) else {
                                continue;
                            };
                            match mesh_index.get(m).copied().flatten() {
                                Some(mi) if keep.morph_targets && morph_counts[mi] > 0 => morph_counts[mi],
                                _ => continue,
                            }
                        }
                    };
                    if c.node >= n {
                        continue;
                    }
                    let (node, cpath, cinterp) = (c.node, c.path, c.interp);
                    let Some((times, values)) = clean_keys(c, width) else {
                        continue;
                    };
                    let count = times.len();
                    let hash = times.iter().fold(times.len() as u64, |h, t| {
                        (h.rotate_left(5) ^ u64::from(t.to_bits())).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    });
                    let list = inputs.entry(hash).or_default();
                    let found = list
                        .iter()
                        .find(|(t, _)| t.iter().map(|x| x.to_bits()).eq(times.iter().map(|x| x.to_bits())));
                    let input = if let Some(&(_, i)) = found {
                        i
                    } else {
                        let i = place(&mut b, &times, count, "SCALAR");
                        // Strictly increasing, so the first and last are the bounds.
                        b.accessors[i]["min"] = json!([times[0]]);
                        b.accessors[i]["max"] = json!([times[count - 1]]);
                        list.push((times, i));
                        i
                    };
                    let (path, kind) = match cpath {
                        Path::Translation => ("translation", "VEC3"),
                        Path::Rotation => ("rotation", "VEC4"),
                        Path::Scale => ("scale", "VEC3"),
                        Path::Weights => ("weights", "SCALAR"),
                    };
                    let output = if cpath == Path::Weights {
                        place(&mut b, &values, count * width, "SCALAR")
                    } else {
                        place(&mut b, &values, count, kind)
                    };
                    let interp = match cinterp {
                        Interp::Linear => "LINEAR",
                    };
                    channels.push(json!({ "sampler": samplers.len(), "target": { "node": node, "path": path } }));
                    samplers.push(json!({ "input": input, "output": output, "interpolation": interp }));
                }
                if channels.is_empty() {
                    continue;
                }
                let mut aj = json!({ "channels": channels, "samplers": samplers });
                if !a.name.is_empty() {
                    aj["name"] = json!(a.name);
                }
                if keep.extras
                    && let Some(e) = a.extras
                {
                    aj["extras"] = e;
                }
                animations_json.push(aj);
            }
            if !data.is_empty() {
                let view = b.view(Part::f32(data), None);
                for a in placed {
                    b.accessors[a]["bufferView"] = json!(view);
                }
            }
        }

        self.info.write(&mut self.source);
        let mut asset = json!({ "extras": { "source": Value::Object(self.source) } });
        if let Some(c) = self.info.copyright {
            asset["copyright"] = json!(c);
        }
        json.insert("asset".into(), asset);
        json.insert("scene".into(), json!(0));
        json.insert("scenes".into(), json!([{ "nodes": roots }]));
        if !nodes_json.is_empty() {
            json.insert("nodes".into(), Value::Array(nodes_json));
        }
        if !meshes_json.is_empty() {
            json.insert("meshes".into(), Value::Array(meshes_json));
        }
        if !skins_json.is_empty() {
            json.insert("skins".into(), Value::Array(skins_json));
        }
        if !animations_json.is_empty() {
            json.insert("animations".into(), Value::Array(animations_json));
        }
        let mut facts = self.facts;
        facts.entry("triangles").or_insert(json!(tris));
        Ok(b.finish(json, facts))
    }
}

/// A channel's keys as glTF wants them: finite, times strictly increasing (a key at or before the last moves just
/// after it), rotations unit length and each on the same side as the last. `None` when no key is left. Keys that
/// are already in order are cleaned where they are.
fn clean_keys(c: Channel, width: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    let Channel {
        mut times,
        mut values,
        path,
        ..
    } = c;
    let width = width.max(1);
    let keys = times.len().min(values.len() / width);
    times.truncate(keys);
    values.truncate(keys * width);
    let tidy = times.iter().all(|t| t.is_finite())
        && times.windows(2).all(|w| w[0] < w[1])
        && values.iter().all(|x| x.is_finite());
    if !tidy {
        let mut kept: Vec<f32> = Vec::with_capacity(keys);
        let mut out = Vec::with_capacity(keys * width);
        for (&t, v) in times.iter().zip(values.chunks_exact(width)) {
            if !t.is_finite() || v.iter().any(|x| !x.is_finite()) {
                continue;
            }
            kept.push(match kept.last() {
                Some(&p) if t <= p => p.next_up(),
                _ => t,
            });
            out.extend_from_slice(v);
        }
        times = kept;
        values = out;
    }
    if path == Path::Rotation {
        let mut last = [0.0f64, 0.0, 0.0, 1.0];
        for v in values.chunks_exact_mut(4) {
            let mut q: [f64; 4] = std::array::from_fn(|i| f64::from(v[i]));
            // Already unit length to within half a float's step: normalizing wouldn't change it.
            let unit = (q.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() <= 1e-7;
            if !unit {
                q = normalize(q).map(f64::from);
            }
            let dot: f64 = (0..4).map(|i| q[i] * last[i]).sum();
            if dot < 0.0 {
                q = q.map(|x| -x);
            }
            last = q;
            if !unit || dot < 0.0 {
                for (o, x) in v.iter_mut().zip(q) {
                    *o = f32_of(x);
                }
            }
        }
    }
    (!times.is_empty()).then_some((times, values))
}

/// A primitive's joints and weights, to be written after the meshes.
struct Skinning {
    joints: Vec<u32>,
    weights: Vec<f32>,
    sets: usize,
    vertices: usize,
    count: usize,
}

impl Skinning {
    /// Writes the accessors, returning the attributes they make.
    fn write(self, b: &mut Builder) -> Vec<(String, usize)> {
        let (n, count) = (self.vertices, self.count);
        let per = n * 4;
        let mut out = Vec::new();
        for s in 0..self.sets {
            let jpart = &self.joints[s * per..(s + 1) * per];
            let j = if count <= 256 {
                let v: Vec<u8> = jpart.iter().map(|&j| u8::try_from(j).unwrap_or(0)).collect();
                b.accessor(Part::bytes(v), 5121, n, "VEC4", Some(34962))
            } else {
                let v = jpart.iter().map(|&j| u16::try_from(j).unwrap_or(0)).collect();
                b.accessor(Part::u16(v), 5123, n, "VEC4", Some(34962))
            };
            let w = b.accessor(
                Part::f32(self.weights[s * per..(s + 1) * per].to_vec()),
                5126,
                n,
                "VEC4",
                Some(34962),
            );
            out.push((format!("JOINTS_{s}"), j));
            out.push((format!("WEIGHTS_{s}"), w));
        }
        out
    }
}

/// Writes a primitive's attributes and indices, and gives its joints and weights to write later (the caller
/// writes its targets); `None` when it draws nothing.
fn write_prim(
    b: &mut Builder,
    mut p: Primitive,
    joints: Option<usize>,
    materials: usize,
) -> Option<(Value, Option<Skinning>)> {
    let n = p.positions.len();
    if n == 0 || p.indices.is_empty() {
        return None;
    }
    if p.indices.iter().any(|&i| i as usize >= n) {
        let width = if p.points {
            1
        } else if p.lines {
            2
        } else {
            3
        };
        let idx: Vec<u32> = p
            .indices
            .chunks_exact(width)
            .filter(|t| t.iter().all(|&i| (i as usize) < n))
            .flatten()
            .copied()
            .collect();
        p.indices = idx;
        if p.indices.is_empty() {
            return None;
        }
    }
    for v in &mut p.positions {
        if v.iter().any(|c| !c.is_finite()) {
            *v = [0.0; 3];
        }
    }
    let mut attributes = Map::new();
    if p.normals.len() == n && !p.points && !p.lines {
        fix_normals(&mut p.normals, &p.positions, &p.indices);
        let a = b.accessor(
            Part::f32(std::mem::take(&mut p.normals).into_flattened()),
            5126,
            n,
            "VEC3",
            Some(34962),
        );
        attributes.insert("NORMAL".into(), json!(a));
    } else if p.normals.len() == n {
        let mut ok = true;
        for v in &mut p.normals {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 0.0 && l.is_finite() {
                *v = v.map(|c| c / l);
            } else {
                ok = false;
            }
        }
        if ok {
            let a = b.accessor(
                Part::f32(std::mem::take(&mut p.normals).into_flattened()),
                5126,
                n,
                "VEC3",
                Some(34962),
            );
            attributes.insert("NORMAL".into(), json!(a));
        }
    }
    // glTF wants tangents only beside normals, unit length with a w of ±1.
    if attributes.contains_key("NORMAL") && p.tangents.len() == n {
        let t: Vec<f32> = p
            .tangents
            .iter()
            .flat_map(|t| {
                let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                let w = if t[3] < 0.0 { -1.0 } else { 1.0 };
                if len > 0.0 && len.is_finite() {
                    [t[0] / len, t[1] / len, t[2] / len, w]
                } else {
                    [1.0, 0.0, 0.0, w]
                }
            })
            .collect();
        let a = b.accessor(Part::f32(t), 5126, n, "VEC4", Some(34962));
        attributes.insert("TANGENT".into(), json!(a));
    }
    let pos = b.positions(std::mem::take(&mut p.positions));
    attributes.insert("POSITION".into(), json!(pos));
    let mut k = 0;
    for set in std::mem::take(&mut p.uvs) {
        if set.len() != n {
            continue;
        }
        let mut set = set.into_flattened();
        for x in &mut set {
            if !x.is_finite() {
                *x = 0.0;
            }
        }
        let a = b.accessor(Part::f32(set), 5126, n, "VEC2", Some(34962));
        attributes.insert(format!("TEXCOORD_{k}"), json!(a));
        k += 1;
    }
    let mut k = 0;
    for set in std::mem::take(&mut p.colors) {
        if set.len() != n {
            continue;
        }
        let mut set = set.into_flattened();
        for x in &mut set {
            *x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 1.0 };
        }
        let a = b.accessor(Part::f32(set), 5126, n, "VEC4", Some(34962));
        attributes.insert(format!("COLOR_{k}"), json!(a));
        k += 1;
    }
    let mut skinning = None;
    if let Some(count) = joints
        && p.influences > 0
        && p.joints.len() == n * p.influences
        && p.weights.len() == n * p.influences
    {
        let (joints, weights, sets) = skin_sets(&p, count);
        skinning = Some(Skinning {
            joints,
            weights,
            sets,
            vertices: n,
            count,
        });
    }
    let count = p.indices.len();
    let narrow = (n < usize::from(u16::MAX))
        .then(|| {
            p.indices
                .iter()
                .map(|&i| u16::try_from(i).ok())
                .collect::<Option<Vec<u16>>>()
        })
        .flatten();
    let indices = match narrow {
        Some(v) => b.accessor(Part::u16(v), 5123, count, "SCALAR", Some(34963)),
        None => b.accessor(
            Part::u32(std::mem::take(&mut p.indices)),
            5125,
            count,
            "SCALAR",
            Some(34963),
        ),
    };
    let mut prim = json!({ "attributes": attributes, "indices": indices });
    if p.points {
        prim["mode"] = json!(0);
    } else if p.lines {
        prim["mode"] = json!(1);
    }
    if let Some(m) = p.material
        && materials > 0
    {
        prim["material"] = json!(m.min(materials - 1));
    }
    Some((prim, skinning))
}

/// A `VEC3` float accessor of `n` offsets from `v` (missing ones zero): sparse over zeros when few are nonzero.
/// `bounds` adds `min` and `max`, which glTF requires of position offsets.
#[allow(
    clippy::needless_bitwise_bool,
    reason = "branchless, so the loop over every vertex vectorizes"
)]
fn sparse_vec3(b: &mut Builder, mut v: Vec<[f32; 3]>, n: usize, bounds: bool) -> usize {
    // Every value as it's written, in one pass: one past the list, or not finite, is zero.
    v.resize(n, [0.0; 3]);
    let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut moved = 0;
    // Without branching per component (`&` and `|`, not `&&` and `||`), so the loop vectorizes.
    for x in &mut v {
        if !(x[0].is_finite() & x[1].is_finite() & x[2].is_finite()) {
            *x = [0.0; 3];
        }
        moved += usize::from((x[0] != 0.0) | (x[1] != 0.0) | (x[2] != 0.0));
        // Finite by now, so a plain comparison does; zero vectors count, as the bounds take in zero
        // whenever there's one.
        for k in 0..3 {
            min[k] = if x[k] < min[k] { x[k] } else { min[k] };
            max[k] = if x[k] > max[k] { x[k] } else { max[k] };
        }
    }
    let a = if moved * 4 < n {
        let mut nonzero: Vec<u32> = Vec::with_capacity(moved.max(1));
        let mut values: Vec<[f32; 3]> = Vec::with_capacity(moved.max(1));
        for (i, x) in v.iter().enumerate() {
            if x.iter().any(|&c| c != 0.0) {
                nonzero.push(u32_of(i));
                values.push(*x);
            }
        }
        // An accessor with neither a view nor sparse values is all zeros to glTF, but not every reader takes it:
        // a target that moves nothing stores one zero offset.
        if nonzero.is_empty() {
            nonzero.push(0);
            values.push([0.0; 3]);
        }
        let count = nonzero.len();
        let iv = b.view(Part::u32(nonzero), None);
        let vv = b.view(Part::vec3(values), None);
        let a = json!({
            "componentType": 5126,
            "count": n,
            "type": "VEC3",
            "sparse": {
                "count": count,
                "indices": { "bufferView": iv, "componentType": 5125 },
                "values": { "bufferView": vv },
            },
        });
        b.accessors.push(a);
        b.accessors.len() - 1
    } else {
        b.accessor(Part::vec3(v), 5126, n, "VEC3", Some(34962))
    };
    if bounds {
        b.accessors[a]["min"] = json!(min);
        b.accessors[a]["max"] = json!(max);
    }
    a
}

/// Each vertex's joints and weights in sets of four, as glTF wants them: joints in range and merged where
/// repeated, the strongest first, at most [`MAX_INFLUENCES`], summing to 1, and a vertex with none bound wholly to
/// the primitive's rest joint. Returns the joints and weights set after set, and the number of sets.
fn skin_sets(p: &Primitive, count: usize) -> (Vec<u32>, Vec<f32>, usize) {
    let n = p.positions.len().max(p.joints.len() / p.influences.max(1));
    let w = p.influences;
    // Every vertex's list, end to end, and where each ends: no list of its own for each vertex.
    let mut flat: Vec<(u32, f32)> = Vec::with_capacity(n * w.clamp(1, 4));
    let mut ends: Vec<usize> = Vec::with_capacity(n);
    let mut list: Vec<(u32, f32)> = Vec::with_capacity(w);
    let mut most = 1;
    let rest = if (p.rest_joint as usize) < count {
        p.rest_joint
    } else {
        0
    };
    for v in 0..n {
        list.clear();
        for k in 0..w {
            let (j, wt) = (p.joints[v * w + k], p.weights[v * w + k]);
            if !(wt > 0.0 && wt.is_finite()) || j as usize >= count {
                continue;
            }
            match list.iter_mut().find(|e| e.0 == j) {
                Some(e) => e.1 += wt,
                None => list.push((j, wt)),
            }
        }
        list.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        list.truncate(MAX_INFLUENCES);
        let sum: f32 = list.iter().map(|e| e.1).sum();
        if list.is_empty() || sum <= 0.0 || !sum.is_finite() {
            flat.push((rest, 1.0));
        } else {
            flat.extend(list.iter().map(|&(j, wt)| (j, wt / sum)));
        }
        most = most.max(flat.len() - ends.last().copied().unwrap_or(0));
        ends.push(flat.len());
    }
    let sets = most.div_ceil(4);
    let mut js = vec![0u32; sets * n * 4];
    let mut ws = vec![0f32; sets * n * 4];
    let starts = std::iter::once(0).chain(ends.iter().copied());
    for (v, (start, end)) in starts.zip(ends.iter().copied()).enumerate() {
        for (k, &(j, wt)) in flat[start..end].iter().enumerate() {
            let at = (k / 4) * n * 4 + v * 4 + k % 4;
            js[at] = j;
            ws[at] = wt;
        }
    }
    (js, ws, sets)
}

/// Normalizes `normals`, replacing any that is zero or not finite by the normal of the faces around it (or +Y).
fn fix_normals(normals: &mut [[f32; 3]], positions: &[[f32; 3]], indices: &[u32]) {
    let mut bad = Vec::new();
    for (i, v) in normals.iter_mut().enumerate() {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if l > 0.0 && l.is_finite() {
            *v = v.map(|c| c / l);
        } else {
            bad.push(i);
        }
    }
    if bad.is_empty() {
        return;
    }
    let mut sum = vec![[0.0f32; 3]; normals.len()];
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| positions[i as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let f = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        for &i in t {
            let s = &mut sum[i as usize];
            for k in 0..3 {
                s[k] += f[k];
            }
        }
    }
    for i in bad {
        let s = sum[i];
        let l = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        normals[i] = if l > 0.0 && l.is_finite() {
            s.map(|c| c / l)
        } else {
            [0.0, 1.0, 0.0]
        };
    }
}

// Math, in double precision.

/// The Hamilton product `a·b` of quaternions `[x, y, z, w]`: rotating by `b`, then by `a`.
pub(crate) fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// `q` at unit length, in single precision; the identity for a zero or broken one.
pub(crate) fn normalize(q: [f64; 4]) -> [f32; 4] {
    let l = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    if l > 0.0 && l.is_finite() {
        q.map(|c| f32_of(c / l))
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// The rotation by `radians` about axis `axis` (0, 1 or 2 for X, Y or Z).
pub(crate) fn about(axis: usize, radians: f64) -> [f64; 4] {
    let (s, c) = (radians * 0.5).sin_cos();
    let mut q = [0.0, 0.0, 0.0, c];
    q[axis] = s;
    q
}

/// `v` rotated by unit quaternion `q`.
pub(crate) fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z, w] = q;
    let [a, b, c] = v;
    let t = [2.0 * (y * c - z * b), 2.0 * (z * a - x * c), 2.0 * (x * b - y * a)];
    [
        a + w * t[0] + (y * t[2] - z * t[1]),
        b + w * t[1] + (z * t[0] - x * t[2]),
        c + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

/// The quaternion mirrored through the XY plane: the same turn seen in a Z-negated world.
pub(crate) fn mirror_quat(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], q[2], q[3]]
}

/// The matrix mirrored through the XY plane: `S·m·S` with `S` negating Z.
pub(crate) fn mirror_matrix(m: &M4) -> M4 {
    let mut out = *m;
    // Column-major: element (row r, column c) is at c * 4 + r; negate where exactly one of r, c is 2.
    for c in 0..4 {
        for r in 0..4 {
            if (r == 2) != (c == 2) {
                out[c * 4 + r] = -out[c * 4 + r];
            }
        }
    }
    out
}

/// `a·b`.
pub(crate) fn mul(a: &M4, b: &M4) -> M4 {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    out
}

/// The transform of translation `t`, rotation `q` and scale `s`.
pub(crate) fn trs(t: [f32; 3], q: [f32; 4], s: [f32; 3]) -> M4 {
    let r = quat_matrix(q.map(f64::from));
    let s = s.map(f64::from);
    let mut m = IDENTITY;
    for c in 0..3 {
        for row in 0..3 {
            m[c * 4 + row] = r[c * 3 + row] * s[c];
        }
    }
    m[12] = f64::from(t[0]);
    m[13] = f64::from(t[1]);
    m[14] = f64::from(t[2]);
    m
}

/// The 3×3 rotation of unit quaternion `q`, column-major.
pub(crate) fn quat_matrix(q: [f64; 4]) -> [f64; 9] {
    let [x, y, z, w] = q;
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// The quaternion of a 3×3 rotation, column-major.
pub(crate) fn matrix_quat(m: &[f64; 9]) -> [f64; 4] {
    let at = |r: usize, c: usize| m[c * 3 + r];
    let trace = at(0, 0) + at(1, 1) + at(2, 2);
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (at(2, 1) - at(1, 2)) / s,
            (at(0, 2) - at(2, 0)) / s,
            (at(1, 0) - at(0, 1)) / s,
            0.25 * s,
        ]
    } else if at(0, 0) > at(1, 1) && at(0, 0) > at(2, 2) {
        let s = (1.0 + at(0, 0) - at(1, 1) - at(2, 2)).sqrt() * 2.0;
        [
            0.25 * s,
            (at(0, 1) + at(1, 0)) / s,
            (at(0, 2) + at(2, 0)) / s,
            (at(2, 1) - at(1, 2)) / s,
        ]
    } else if at(1, 1) > at(2, 2) {
        let s = (1.0 + at(1, 1) - at(0, 0) - at(2, 2)).sqrt() * 2.0;
        [
            (at(0, 1) + at(1, 0)) / s,
            0.25 * s,
            (at(1, 2) + at(2, 1)) / s,
            (at(0, 2) - at(2, 0)) / s,
        ]
    } else {
        let s = (1.0 + at(2, 2) - at(0, 0) - at(1, 1)).sqrt() * 2.0;
        [
            (at(0, 2) + at(2, 0)) / s,
            (at(1, 2) + at(2, 1)) / s,
            0.25 * s,
            (at(1, 0) - at(0, 1)) / s,
        ]
    };
    let l = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    if l > 0.0 && l.is_finite() {
        q.map(|c| c / l)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// A transform split into translation, rotation and scale. A mirroring transform takes a negative X scale; shear is
/// lost.
pub(crate) fn decompose(m: &M4) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let t = [m[12], m[13], m[14]].map(f32_of);
    let col = |c: usize| [m[c * 4], m[c * 4 + 1], m[c * 4 + 2]];
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let (c0, c1, c2) = (col(0), col(1), col(2));
    let det = c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
        + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1]);
    let mut s = [len(c0), len(c1), len(c2)];
    if det < 0.0 {
        s[0] = -s[0];
    }
    let mut r = [0.0; 9];
    for (c, v) in [c0, c1, c2].iter().enumerate() {
        let k = if s[c] != 0.0 && s[c].is_finite() { s[c] } else { 1.0 };
        for row in 0..3 {
            r[c * 3 + row] = v[row] / k;
        }
    }
    let q = matrix_quat(&r);
    (t, normalize(q), s.map(f32_of))
}

/// The inverse of `m`; the identity when it has none.
pub(crate) fn invert(m: &M4) -> M4 {
    let mut inv = [0.0; 16];
    inv[0] =
        m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11]
            - m[13] * m[7] * m[10];
    inv[4] =
        -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11]
            + m[12] * m[7] * m[10];
    inv[8] =
        m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11]
            - m[12] * m[7] * m[9];
    inv[12] =
        -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10]
            + m[12] * m[6] * m[9];
    inv[1] =
        -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11]
            + m[13] * m[3] * m[10];
    inv[5] =
        m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11]
            - m[12] * m[3] * m[10];
    inv[9] =
        -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11]
            + m[12] * m[3] * m[9];
    inv[13] =
        m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10]
            - m[12] * m[2] * m[9];
    inv[2] =
        m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7]
            - m[13] * m[3] * m[6];
    inv[6] =
        -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7]
            + m[12] * m[3] * m[6];
    inv[10] =
        m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7]
            - m[12] * m[3] * m[5];
    inv[14] =
        -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6]
            + m[12] * m[2] * m[5];
    inv[3] =
        -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7]
            + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7]
        - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7]
        + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6]
        - m[8] * m[2] * m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if det == 0.0 || !det.is_finite() {
        return IDENTITY;
    }
    inv.map(|v| v / det)
}

/// The quarter turn about X that takes a Z-up world to glTF's Y up: +Z goes to +Y, +Y to -Z.
pub(crate) const Z_UP_TO_Y_UP: [f32; 4] = [
    -std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    0.0,
    std::f32::consts::FRAC_1_SQRT_2,
];

/// A little-endian reader over a file's bytes that checks every read against the end.
#[derive(Clone, Copy)]
pub(crate) struct Bin<'a> {
    pub data: &'a [u8],
    pub at: usize,
    /// The format's name, for errors.
    pub what: &'static str,
}

#[allow(dead_code, reason = "each binary reader uses the parts its format needs")]
impl<'a> Bin<'a> {
    pub fn new(data: &'a [u8], what: &'static str) -> Self {
        Self { data, at: 0, what }
    }

    /// Bytes left.
    pub fn left(&self) -> usize {
        self.data.len().saturating_sub(self.at)
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        match self.at.checked_add(n).filter(|&e| e <= self.data.len()) {
            Some(end) => {
                let s = &self.data[self.at..end];
                self.at = end;
                Ok(s)
            }
            None => bail!("{} file is cut short", self.what),
        }
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }

    /// Moves to `off` from the start.
    pub fn seek(&mut self, off: usize) -> Result<()> {
        ensure!(off <= self.data.len(), "{} file points past its end", self.what);
        self.at = off;
        Ok(())
    }

    pub fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.bytes(N)?);
        Ok(out)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    pub fn i8(&mut self) -> Result<i8> {
        Ok(i8::from_le_bytes(self.array()?))
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    pub fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.array()?))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.array()?))
    }

    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    pub fn vec2(&mut self) -> Result<[f32; 2]> {
        Ok([self.f32()?, self.f32()?])
    }

    pub fn vec3(&mut self) -> Result<[f32; 3]> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }

    pub fn vec4(&mut self) -> Result<[f32; 4]> {
        Ok([self.f32()?, self.f32()?, self.f32()?, self.f32()?])
    }

    /// A count of items of `size` bytes each that must fit in what's left: refused otherwise, so nothing is
    /// allocated for a count the file can't hold.
    pub fn count(&self, n: u64, size: usize) -> Result<usize> {
        let n = usize::try_from(n).unwrap_or(usize::MAX);
        match n.checked_mul(size.max(1)) {
            Some(b) if b <= self.left() => Ok(n),
            _ => bail!("{} file claims {n} items, more than it holds", self.what),
        }
    }

    /// A fixed-size string field: up to its first NUL.
    pub fn fixed_str(&mut self, n: usize) -> Result<&'a [u8]> {
        let b = self.bytes(n)?;
        Ok(&b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())])
    }
}

/// Text as Rust wants it: UTF-8 as it is, anything else read as Windows-1252/Latin-1 byte by byte.
pub(crate) fn text(b: &[u8]) -> String {
    match std::str::from_utf8(b) {
        Ok(s) => s.to_owned(),
        Err(_) => b.iter().map(|&c| char::from(c)).collect(),
    }
}

/// Text in Shift-JIS (code page 932), as Japanese tools write it. A byte sequence that isn't Shift-JIS becomes
/// U+FFFD.
pub(crate) fn shift_jis(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c < 0x80 {
            out.push(char::from(c));
            i += 1;
        } else if (0xA1..=0xDF).contains(&c) {
            // Half-width katakana.
            out.push(char::from_u32(0xFF61 + u32::from(c - 0xA1)).unwrap_or('\u{FFFD}'));
            i += 1;
        } else if i + 1 < b.len()
            && let Some(ch) = sjis_pair(c, b[i + 1])
        {
            out.push(ch);
            i += 2;
        } else {
            out.push('\u{FFFD}');
            i += 1;
        }
    }
    out
}

/// Code page 932's double-byte characters: for lead bytes 0x81 to 0x9F then 0xE0 to 0xFC, and trail bytes 0x40 to
/// 0xFC, each character as a little-endian UTF-16 unit, 0 for none. Made from the code page's published mapping
/// (Python's `cp932` codec, pair by pair).
static CP932: &[u8; 22680] = include_bytes!("cp932.bin");

/// The character for a Shift-JIS lead and trail byte.
fn sjis_pair(a: u8, b: u8) -> Option<char> {
    let lead = match a {
        0x81..=0x9F => usize::from(a - 0x81),
        0xE0..=0xFC => usize::from(a - 0xE0) + 31,
        _ => return None,
    };
    if !(0x40..=0xFC).contains(&b) {
        return None;
    }
    let at = (lead * 189 + usize::from(b - 0x40)) * 2;
    let u = u16::from_le_bytes([CP932[at], CP932[at + 1]]);
    if u == 0 { None } else { char::from_u32(u32::from(u)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices_and_quaternions_round_trip() {
        let q = normalize([0.1, 0.7, -0.2, 0.6]);
        let m = trs([1.0, 2.0, 3.0], q, [2.0, 2.0, 2.0]);
        let (t, r, s) = decompose(&m);
        assert_eq!(t, [1.0, 2.0, 3.0]);
        let dot: f32 = (0..4).map(|k| r[k] * q[k]).sum();
        assert!((dot.abs() - 1.0).abs() < 1e-6);
        assert!(s.iter().all(|&c| (c - 2.0).abs() < 1e-5));
        let i = mul(&m, &invert(&m));
        for (k, v) in i.iter().enumerate() {
            assert!((v - IDENTITY[k]).abs() < 1e-9);
        }
    }

    #[test]
    fn mirroring_a_matrix_matches_mirroring_its_parts() {
        let q = normalize([0.3, -0.1, 0.5, 0.8]);
        let m = trs([1.0, 2.0, 3.0], q, [1.0; 3]);
        let mm = mirror_matrix(&m);
        let expect = trs([1.0, 2.0, -3.0], mirror_quat(q), [1.0; 3]);
        for k in 0..16 {
            assert!((mm[k] - expect[k]).abs() < 1e-6, "{k}");
        }
    }

    #[test]
    fn loops_in_parents_are_cut() {
        let mut s = Scene::default();
        s.add(Node::named("a", Some(1)));
        s.add(Node::named("b", Some(0)));
        s.add(Node::named("c", Some(9)));
        let p = s.parents();
        assert!(p[0].is_none() || p[1].is_none());
        assert_eq!(p[2], None);
        let _ = s.world();
    }

    #[test]
    fn skin_sets_merge_sort_and_normalize() {
        let p = Primitive {
            positions: vec![[0.0; 3]; 2],
            joints: vec![1, 1, 2, 7, 0, 0, 0, 0],
            weights: vec![0.25, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            influences: 4,
            rest_joint: 2,
            ..Primitive::default()
        };
        let (j, w, sets) = skin_sets(&p, 3);
        assert_eq!(sets, 1);
        assert_eq!(&j[..4], &[2, 1, 0, 0]);
        assert!((w[0] - 2.0 / 3.0).abs() < 1e-6 && (w[1] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(&j[4..8], &[2, 0, 0, 0]);
        assert_eq!(&w[4..8], &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn shift_jis_kana_and_ascii() {
        // "ミク" in katakana, then "a".
        assert_eq!(shift_jis(&[0x83, 0x7E, 0x83, 0x4E, b'a']), "ミクa");
        assert_eq!(shift_jis(&[0x82, 0xA0]), "あ");
        assert_eq!(shift_jis(&[0xB1]), "ｱ");
        // "右腕" (right arm), as MMD names a bone.
        assert_eq!(shift_jis(&[0x89, 0x45, 0x98, 0x72]), "右腕");
        assert_eq!(shift_jis(&[0x89]), "\u{FFFD}");
    }

    /// A scene with what the writer must fix: two skeletons under one skin, joints repeated and out of range,
    /// unnormalized rotations, keys out of order, a sparse morph target and a broken normal.
    pub(crate) fn awkward() -> Scene {
        let mut s = Scene::default();
        let a = s.add(Node::named("a", None));
        let mut bn = Node::named("b", None);
        bn.translation = [0.0, 1.0, 0.0];
        bn.rotation = [0.0, 0.0, 2.0, 2.0];
        let b = s.add(bn);
        let mut mesh = Node::named("mesh", None);
        mesh.mesh = Some(0);
        mesh.skin = Some(0);
        s.add(mesh);
        let positions: Vec<[f32; 3]> = (0..8)
            .map(|i: u8| [f32::from(i % 2), f32::from(i / 2 % 2), f32::from(i / 4)])
            .collect();
        let mut moved = vec![[0.0; 3]; 8];
        moved[7] = [0.0, 0.5, 0.0];
        s.meshes.push(SceneMesh {
            name: "m".into(),
            prims: vec![Primitive {
                normals: (0..8)
                    .map(|i| if i == 3 { [0.0; 3] } else { [0.0, 2.0, 0.0] })
                    .collect(),
                uvs: vec![vec![[0.5, 0.5]; 8], vec![]],
                joints: (0..8).flat_map(|_| [0, 0, 1, 9]).collect(),
                weights: (0..8)
                    .flat_map(|i| [0.25, 0.25, if i == 0 { 0.0 } else { 1.0 }, 1.0])
                    .collect(),
                influences: 4,
                positions,
                indices: vec![0, 1, 2, 2, 1, 3, 4, 6, 5, 5, 6, 7, 0, 9, 1],
                material: Some(5),
                targets: vec![Target {
                    positions: moved,
                    normals: Vec::new(),
                }],
                ..Primitive::default()
            }],
            weights: vec![0.25],
            target_names: vec!["up".into()],
            extras: None,
        });
        s.skins.push(Skin {
            name: "skin".into(),
            joints: vec![a, b],
            inverse_binds: vec![IDENTITY; 2],
        });
        s.materials.push(Material::plain("red", [1.0, 0.0, 0.0, 1.0]));
        s.animations.push(Animation {
            name: "take".into(),
            channels: vec![
                Channel {
                    node: b,
                    path: Path::Rotation,
                    times: vec![0.0, 0.0, 1.0],
                    values: vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                    interp: Interp::Linear,
                },
                Channel {
                    node: a,
                    path: Path::Translation,
                    times: vec![0.0, 0.5],
                    values: vec![0.0; 6],
                    interp: Interp::Linear,
                },
                Channel {
                    node: 2,
                    path: Path::Weights,
                    times: vec![0.0, 1.0],
                    values: vec![0.0, 1.0],
                    interp: Interp::Linear,
                },
            ],
            extras: None,
        });
        s
    }

    #[test]
    fn an_awkward_scene_writes_and_flattens() {
        let mut s = awkward();
        s.mirror_z();
        s.rotate_roots(Z_UP_TO_Y_UP);
        s.scale(0.5);
        let doc = s.finish(Keep::ALL).unwrap();
        let skins = doc.json["skins"].as_array().unwrap();
        assert_eq!(skins.len(), 1);
        // The two skeleton roots are gathered under one.
        assert_eq!(doc.json["nodes"].as_array().unwrap().len(), 4);
        let prim = &doc.json["meshes"][0]["primitives"][0];
        assert!(prim["attributes"].get("TEXCOORD_1").is_none());
        assert_eq!(prim["material"], 0);
        assert!(
            doc.json["accessors"][usize::try_from(prim["targets"][0]["POSITION"].as_u64().unwrap()).unwrap()]
                .get("sparse")
                .is_some()
        );
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(mesh.triangles(), 4);
        assert!(doc.to_glb().unwrap().starts_with(b"glTF"));
    }
}
