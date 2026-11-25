//! OGRE meshes (`.mesh`, `.mesh.xml`) and skeletons (`.skeleton`, `.skeleton.xml`), with their material scripts.
//!
//! The binary layouts of every 1.x serializer are read in either byte order: v1.10 to v1.30's fixed geometry
//! chunks, the vertex declarations of v1.30 on (every element type, half floats and packed colors included), the
//! level of detail layouts of v1.40, v1.41, v1.8 and v1.100, and the edge lists and extremes they may carry
//! (skipped). The XML layouts are read as OGRE's converter writes them. A `.xml` file is told apart by its top
//! element. OGRE models are right-handed and Y up with counter-clockwise faces, as glTF's are, so nothing is turned
//! (the source info assumes both, as the engine's convention; the units' length is unknown).
//!
//! Each submesh of level of detail 0 becomes a primitive; one on the shared geometry gets only the shared vertices
//! it uses. Strips, fans and line strips become lists; a submesh without indices draws its vertices in order.
//! Positions, normals, tangents (their bitangent's sign from the binormals), eight texture coordinate sets, and
//! the diffuse and specular colors (as `COLOR_0` and `COLOR_1`) are kept. Elements glTF has no place for (blend
//! weights and indices in the vertex data, which OGRE itself fills from the bone assignments, extra sets of a
//! semantic, third and fourth texture coordinates) are noted in the facts.
//!
//! The skeleton the mesh links to is found among the assets, binary or XML. Its bones become nodes in their bind
//! pose; the bone assignments become joints and weights, the four heaviest of each vertex scaled to sum to 1;
//! inverse bind matrices come from the bind pose. Skeleton animations' keys are relative to the bind pose and
//! are written as the poses they make; animations of linked skeletons are added, their translations scaled. A
//! submesh without bone assignments goes in a second mesh without the skin, as OGRE draws it unskinned; the split
//! is made even when the skin is not kept or its skeleton is missing.
//!
//! Poses become morph targets, and so does each keyframe of a morph animation (its positions less the mesh's);
//! mesh animations become weights animations. The material scripts beside the mesh, those named after it or its
//! materials, and those they import give the materials (see [`material`]). Other levels of detail, edge lists,
//! bounds and the like are noted in the facts.

mod binary;
mod material;
mod xml;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::Input;
use super::scene::{
    self, Animation, Channel, Interp, M4, Node, Path as KeyPath, Primitive, Scene, SceneMesh, Skin, Target,
};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::{Material, srgb_to_linear};
use crate::quat;
use crate::source::{Axis, Fact, Handedness};

/// The most morph target and weights animation floats kept: offsets for every target's vertices, and every key's
/// weights.
const MORPH_BUDGET: usize = 1 << 22;
/// The most files looked up by name for materials not yet found.
const MATERIAL_LOOKUPS: usize = 16;
/// The most linked skeletons whose animations are read.
const MAX_LINKS: usize = 8;

/// One set of vertices: shared by submeshes, or a submesh's own.
#[derive(Default, Debug)]
pub(super) struct Geometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// `w` is NaN where the source has none (then the binormals give the bitangent's sign).
    pub tangents: Vec<[f32; 4]>,
    pub binormals: Vec<[f32; 3]>,
    /// RGBA as stored (sRGB).
    pub diffuse: Vec<[f32; 4]>,
    pub specular: Vec<[f32; 4]>,
    pub uvs: Vec<Vec<[f32; 2]>>,
}

/// A vertex's assignment to a bone.
#[derive(Clone, Copy, Debug)]
pub(super) struct Assign {
    pub vertex: u32,
    pub bone: u16,
    pub weight: f32,
}

/// A submesh: its material, its vertices and how they're drawn.
#[derive(Default, Debug)]
pub(super) struct SubMesh {
    pub material: String,
    /// Drawn from the shared geometry.
    pub shared: bool,
    /// OGRE's operation: 1 points, 2 lines, 3 a line strip, 4 triangles, 5 a strip, 6 a fan.
    pub operation: u16,
    /// `None` draws the vertices in order.
    pub indices: Option<Vec<u32>>,
    pub geometry: Option<Geometry>,
    pub bones: Vec<Assign>,
    /// Texture aliases: alias, texture.
    pub aliases: Vec<(String, String)>,
    pub name: Option<String>,
}

/// A pose's offset of one vertex.
#[derive(Debug)]
pub(super) struct PoseOffset {
    pub index: u32,
    pub offset: [f32; 3],
    pub normal: Option<[f32; 3]>,
}

/// A pose: offsets of some vertices of one geometry (`target` 0 the shared one, `k` submesh `k - 1`'s).
#[derive(Debug)]
pub(super) struct Pose {
    pub name: String,
    pub target: u16,
    pub offsets: Vec<PoseOffset>,
}

/// A morph keyframe: every vertex's position (and normal, when given) at a time.
#[derive(Debug)]
pub(super) struct MorphKey {
    pub time: f32,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

/// A pose keyframe: poses and their influence at a time.
#[derive(Debug)]
pub(super) struct PoseKey {
    pub time: f32,
    pub refs: Vec<(u16, f32)>,
}

/// A track's keys.
#[derive(Debug)]
pub(super) enum Keys {
    Morph(Vec<MorphKey>),
    Pose(Vec<PoseKey>),
}

/// A mesh animation's track on one geometry.
#[derive(Debug)]
pub(super) struct Track {
    pub target: u16,
    pub keys: Keys,
}

/// A mesh animation.
#[derive(Debug)]
pub(super) struct MeshAnimation {
    pub name: String,
    pub length: f32,
    /// The animation and time an additive animation is relative to.
    pub base: Option<(String, f32)>,
    pub tracks: Vec<Track>,
}

/// A level of detail past the first: when it's used, and the mesh it is when made by hand.
#[derive(Debug)]
pub(super) struct LodLevel {
    pub usage: f32,
    pub manual: Option<String>,
}

/// A whole mesh file.
#[derive(Default, Debug)]
pub(super) struct MeshFile {
    pub version: String,
    pub big_endian: bool,
    pub skeletally_animated: bool,
    pub shared: Option<Geometry>,
    pub submeshes: Vec<SubMesh>,
    pub skeleton: Option<String>,
    pub shared_bones: Vec<Assign>,
    pub lod_strategy: Option<String>,
    pub lods: Vec<LodLevel>,
    /// Minimum, maximum and radius.
    pub bounds: Option<([f32; 3], [f32; 3], f32)>,
    pub edge_lists: usize,
    pub extremes: usize,
    pub poses: Vec<Pose>,
    pub animations: Vec<MeshAnimation>,
    /// Vertex elements glTF has no place for.
    pub dropped_elements: Vec<String>,
    /// Texture coordinate sets with a third or fourth component, cut to two.
    pub texcoord_w: usize,
}

/// A bone in its bind pose.
#[derive(Debug)]
pub(super) struct Bone {
    pub name: String,
    pub handle: u16,
    pub position: [f32; 3],
    /// `[x, y, z, w]`.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub parent: Option<u16>,
}

/// A bone's key, relative to its bind pose.
#[derive(Debug)]
pub(super) struct Key {
    pub time: f32,
    pub rotation: [f32; 4],
    pub translation: [f32; 3],
    pub scale: [f32; 3],
}

/// A bone's keys.
#[derive(Debug)]
pub(super) struct BoneTrack {
    pub bone: u16,
    pub keys: Vec<Key>,
}

/// A skeleton animation.
#[derive(Debug)]
pub(super) struct SkeletonAnimation {
    pub name: String,
    pub length: f32,
    pub base: Option<(String, f32)>,
    pub tracks: Vec<BoneTrack>,
}

/// A whole skeleton file.
#[derive(Default, Debug)]
pub(super) struct SkeletonFile {
    pub version: String,
    pub big_endian: bool,
    pub blend_mode: Option<&'static str>,
    pub bones: Vec<Bone>,
    pub animations: Vec<SkeletonAnimation>,
    /// Skeletons whose animations this one shares, and the scale of their translations.
    pub links: Vec<(String, f32)>,
}

enum Parsed {
    Mesh(Box<MeshFile>),
    Skeleton(SkeletonFile),
}

/// An XML file's text: UTF-16 converted, UTF-8 without its byte order mark, or bytes as Latin-1.
fn xml_text(data: &[u8]) -> std::borrow::Cow<'_, str> {
    crate::formats::xml::text(data)
}

fn parse(data: &[u8]) -> Result<Parsed> {
    if let Some((_, line)) = binary::header(data) {
        if line.starts_with("[MeshSerializer") {
            return binary::mesh(data).map(|m| Parsed::Mesh(Box::new(m)));
        }
        if line.starts_with("[Serializer") {
            return binary::skeleton(data).map(Parsed::Skeleton);
        }
        bail!(Unsupported, "unknown OGRE file version {line}");
    }
    let text = xml_text(data);
    let doc = xml::Xml::parse(&text)?;
    match doc.root().name() {
        "mesh" => xml::mesh(&doc).map(|m| Parsed::Mesh(Box::new(m))),
        "skeleton" => xml::skeleton(&doc).map(Parsed::Skeleton),
        other => bail!(UnknownFormat, "not an OGRE mesh or skeleton: its XML is a <{other}>"),
    }
}

/// Reads an OGRE mesh or skeleton, binary or XML.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let data = input.read()?;
    let name = input
        .path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let lower = name.to_ascii_lowercase();
    let stem = [".mesh.xml", ".skeleton.xml", ".mesh", ".skeleton", ".xml"]
        .iter()
        .find_map(|s| lower.ends_with(s).then(|| name[..name.len() - s.len()].to_owned()))
        .unwrap_or(name);
    match parse(&data)? {
        Parsed::Mesh(file) => mesh_document(&file, &stem, &input, assets, keep),
        Parsed::Skeleton(skel) => {
            let mut scene = Scene::default();
            let mut facts = Map::new();
            let bones = add_bones(&mut scene, &skel);
            let links = linked(&skel, &input, assets, &mut facts);
            add_skeleton_animations(&mut scene, &skel, &bones, &links, &mut facts);
            convention(&mut scene, &skel.version);
            scene.source.insert("kind".into(), json!("skeleton"));
            if skel.big_endian {
                scene.source.insert("byteOrder".into(), json!("big-endian"));
            }
            if let Some(m) = skel.blend_mode {
                scene.source.insert("blendMode".into(), json!(m));
            }
            facts.insert("joints".into(), json!(skel.bones.len()));
            facts.insert("animations".into(), json!(skel.animations.len()));
            scene.facts.extend(facts);
            scene.finish(keep)
        }
    }
}

/// The source info every OGRE file shares: its format and version, and the engine's right-handed, Y up convention.
fn convention(scene: &mut Scene, version: &str) {
    scene.info.format = Some("OGRE".into());
    scene.info.version = Some(version.to_owned());
    scene.info.up_axis = Fact::Assumed(Axis::PosY);
    scene.info.handedness = Fact::Assumed(Handedness::Right);
}

/// The bones' nodes, in the skeleton's order, each found by its handle.
struct Bones {
    nodes: Vec<usize>,
    by_handle: HashMap<u16, usize>,
}

fn add_bones(scene: &mut Scene, skel: &SkeletonFile) -> Bones {
    let mut nodes = Vec::with_capacity(skel.bones.len());
    let mut by_handle = HashMap::new();
    for (k, b) in skel.bones.iter().enumerate() {
        let mut node = Node::named(b.name.clone(), None);
        node.translation = b.position;
        node.rotation = quat::normalize(b.rotation.map(f64::from));
        node.scale = b.scale;
        nodes.push(scene.add(node));
        by_handle.entry(b.handle).or_insert(k);
    }
    for (k, b) in skel.bones.iter().enumerate() {
        if let Some(&p) = b.parent.and_then(|p| by_handle.get(&p))
            && p != k
        {
            scene.nodes[nodes[k]].parent = Some(nodes[p]);
        }
    }
    Bones { nodes, by_handle }
}

/// Reads a file the model names, trying `names` in turn from the model's folder.
fn read_named(names: &[String], input: &Input<'_>, assets: &dyn Assets) -> Option<(PathBuf, Vec<u8>)> {
    names.iter().find_map(|n| {
        let p = assets.find(n, input.dir())?;
        let b = assets.read(&p)?.into_owned();
        Some((p, b))
    })
}

/// The names a skeleton named `link` may have been saved under.
fn skeleton_names(link: &str) -> Vec<String> {
    let mut out = vec![link.to_owned(), format!("{link}.xml")];
    if let Some(base) = link.rsplit(['/', '\\']).next().filter(|b| *b != link) {
        out.push(base.to_owned());
        out.push(format!("{base}.xml"));
    }
    out.dedup();
    out
}

/// The skeletons `skel` links to, read, with their translation scales.
fn linked(
    skel: &SkeletonFile,
    input: &Input<'_>,
    assets: &dyn Assets,
    facts: &mut Map<String, Value>,
) -> Vec<(SkeletonFile, f32)> {
    let mut out = Vec::new();
    let mut missing = Vec::new();
    for (name, scale) in skel.links.iter().take(MAX_LINKS) {
        let found = read_named(&skeleton_names(name), input, assets).and_then(|(_, b)| match parse(&b) {
            Ok(Parsed::Skeleton(s)) => Some(s),
            _ => None,
        });
        match found {
            Some(s) => out.push((s, *scale)),
            None => missing.push(json!(name)),
        }
    }
    if !missing.is_empty() {
        facts.insert("missing_linked_skeletons".into(), Value::Array(missing));
    }
    out
}

/// The skeleton's animations, and its linked skeletons' (matched by bone name), as node channels.
fn add_skeleton_animations(
    scene: &mut Scene,
    skel: &SkeletonFile,
    bones: &Bones,
    links: &[(SkeletonFile, f32)],
    facts: &mut Map<String, Value>,
) {
    let by_name: HashMap<&str, usize> = skel
        .bones
        .iter()
        .enumerate()
        .rev()
        .map(|(k, b)| (b.name.as_str(), k))
        .collect();
    let mut unknown = 0usize;
    let own = skel.animations.iter().map(|a| (a, None, 1.0));
    let others = links
        .iter()
        .flat_map(|(s, scale)| s.animations.iter().map(move |a| (a, Some(s), *scale)));
    for (anim, from, scale) in own.chain(others) {
        let mut channels = Vec::new();
        for t in &anim.tracks {
            let bone = match from {
                None => bones.by_handle.get(&t.bone).copied(),
                Some(s) => s
                    .bones
                    .iter()
                    .find(|b| b.handle == t.bone)
                    .and_then(|b| by_name.get(b.name.as_str()).copied()),
            };
            let Some(k) = bone else {
                unknown += 1;
                continue;
            };
            if t.keys.is_empty() {
                continue;
            }
            let b = &skel.bones[k];
            let node = bones.nodes[k];
            let times: Vec<f32> = t.keys.iter().map(|key| key.time).collect();
            let translation: Vec<f32> = t
                .keys
                .iter()
                .flat_map(|key| [0, 1, 2].map(|i| b.position[i] + key.translation[i] * scale))
                .collect();
            let bind = b.rotation.map(f64::from);
            let rotation: Vec<f32> = t
                .keys
                .iter()
                .flat_map(|key| quat::normalize(quat::mul(bind, key.rotation.map(f64::from))))
                .collect();
            #[allow(clippy::float_cmp, reason = "a scale of exactly 1 everywhere needs no channel")]
            let scaled = t.keys.iter().any(|key| key.scale != [1.0; 3]);
            channels.push(channel(node, KeyPath::Translation, times.clone(), translation));
            channels.push(channel(node, KeyPath::Rotation, times.clone(), rotation));
            if scaled {
                let s: Vec<f32> = t
                    .keys
                    .iter()
                    .flat_map(|key| [0, 1, 2].map(|i| b.scale[i] * key.scale[i]))
                    .collect();
                channels.push(channel(node, KeyPath::Scale, times, s));
            }
        }
        let mut extras = json!({ "length": anim.length });
        if let Some((name, time)) = &anim.base {
            extras["baseAnimation"] = json!(name);
            extras["baseKeyframeTime"] = json!(time);
        }
        if from.is_some() {
            extras["linked"] = json!(true);
        }
        scene.animations.push(Animation {
            name: anim.name.clone(),
            channels,
            extras: Some(extras),
        });
    }
    if unknown > 0 {
        facts.insert("tracks_on_unknown_bones".into(), json!(unknown));
    }
}

fn channel(node: usize, path: KeyPath, times: Vec<f32>, values: Vec<f32>) -> Channel {
    Channel {
        node,
        path,
        times,
        values,
        interp: Interp::Linear,
    }
}

/// A primitive's source: which geometry (0 shared, `k` submesh `k - 1`'s), and the vertices of it used, in order
/// (`None` for all of them).
struct Source {
    geometry: u16,
    used: Option<Vec<u32>>,
}

/// A submesh's indices as glTF's lists: triangles, segments or points.
fn lists(sub: &SubMesh, n: usize) -> (Vec<u32>, bool, bool) {
    let all: Vec<u32>;
    let idx: &[u32] = if let Some(i) = &sub.indices {
        i
    } else {
        all = (0..crate::num::u32_of(n)).collect();
        &all
    };
    match sub.operation {
        1 => (idx.to_vec(), true, false),
        2 => (idx[..idx.len() & !1].to_vec(), false, true),
        3 => (idx.windows(2).flat_map(|w| [w[0], w[1]]).collect(), false, true),
        5 => {
            let mut out = Vec::with_capacity(idx.len().saturating_sub(2) * 3);
            for (k, w) in idx.windows(3).enumerate() {
                if w[0] == w[1] || w[1] == w[2] || w[0] == w[2] {
                    continue;
                }
                if k % 2 == 0 {
                    out.extend_from_slice(w);
                } else {
                    out.extend_from_slice(&[w[1], w[0], w[2]]);
                }
            }
            (out, false, false)
        }
        6 => {
            let mut out = Vec::with_capacity(idx.len().saturating_sub(2) * 3);
            if let Some(&c) = idx.first() {
                for w in idx[1.min(idx.len())..].windows(2) {
                    out.extend_from_slice(&[c, w[0], w[1]]);
                }
            }
            (out, false, false)
        }
        _ => (idx[..idx.len() - idx.len() % 3].to_vec(), false, false),
    }
}

/// `v`'s items at `used` (all of them for `None`).
fn gather<T: Copy>(v: &[T], used: Option<&[u32]>) -> Vec<T> {
    match used {
        None => v.to_vec(),
        Some(u) => u.iter().filter_map(|&i| v.get(i as usize).copied()).collect(),
    }
}

/// Each vertex's four heaviest bone assignments, as joints (indices into the bones) and weights summing to 1.
fn influences(assigns: &[Assign], n: usize, bones: &Bones, unknown: &mut usize) -> Vec<[(u32, f32); 4]> {
    let mut out = vec![[(0u32, 0.0f32); 4]; n];
    for a in assigns {
        let Some(slot) = out.get_mut(a.vertex as usize) else {
            *unknown += 1;
            continue;
        };
        let Some(&joint) = bones.by_handle.get(&a.bone) else {
            *unknown += 1;
            continue;
        };
        if !(a.weight > 0.0 && a.weight.is_finite()) {
            continue;
        }
        let joint = crate::num::u32_of(joint);
        // The lightest slot, unless a slot already holds this joint.
        if let Some(s) = slot.iter_mut().find(|s| s.0 == joint && s.1 > 0.0) {
            s.1 += a.weight;
            continue;
        }
        let (k, min) = slot
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
            .map_or((0, 0.0), |(k, s)| (k, s.1));
        if a.weight > min {
            slot[k] = (joint, a.weight);
        }
    }
    for slot in &mut out {
        let sum: f32 = slot.iter().map(|s| s.1).sum();
        if sum > 0.0 {
            for s in slot.iter_mut() {
                s.1 /= sum;
            }
        }
    }
    out
}

/// A target being built: its name, its geometry and its offsets of every vertex of that geometry.
struct TargetDef {
    name: String,
    geometry: u16,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
}

/// What a material needs after the document is written: glTF settings the shared writer doesn't make.
struct MaterialPost {
    blend: bool,
    color_uv: u32,
    normal_uv: u32,
    extras: Map<String, Value>,
}

/// Finds a texture a material names: from its script's folder, then the model's.
fn texture(t: &material::TextureRef, input: &Input<'_>, assets: &dyn Assets, images: &mut Images) -> Option<usize> {
    images.find(assets, &[&t.file], &[&t.dir, input.dir()])
}

/// Reads the material scripts the mesh may use.
fn library(file: &MeshFile, stem: &str, input: &Input<'_>, assets: &dyn Assets) -> material::Library {
    let mut lib = material::Library::default();
    let mut queue: Vec<(PathBuf, String)> = Vec::new();
    for p in assets.files_with_suffix(input.dir(), ".material") {
        queue.push((p, String::new()));
    }
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut load_queue = |lib: &mut material::Library, queue: &mut Vec<(PathBuf, String)>| {
        while let Some((p, _)) = queue.pop() {
            if seen.contains(&p) || seen.len() >= material::MAX_SCRIPTS {
                continue;
            }
            seen.push(p.clone());
            let Some(bytes) = assets.read(&p) else {
                continue;
            };
            let text = xml_text(&bytes);
            let dir = p.parent().map(Path::to_path_buf).unwrap_or_default();
            for import in lib.add(p, &text) {
                if let Some(found) = assets.find(&import, &dir) {
                    queue.push((found, import));
                }
            }
        }
    };
    queue.reverse();
    load_queue(&mut lib, &mut queue);
    let mut names: Vec<String> = vec![format!("{stem}.material")];
    for s in &file.submeshes {
        if !lib.has(&s.material) && !s.material.is_empty() {
            names.push(format!("{}.material", s.material));
            if let Some(last) = s.material.rsplit(['/', '\\']).next() {
                names.push(format!("{last}.material"));
            }
        }
    }
    let mut tried = Vec::new();
    for n in names {
        if tried.len() >= MATERIAL_LOOKUPS || tried.contains(&n) {
            continue;
        }
        if file.submeshes.iter().all(|s| lib.has(&s.material)) {
            break;
        }
        if let Some(p) = assets.find(&n, input.dir()) {
            queue.push((p, n.clone()));
            load_queue(&mut lib, &mut queue);
        }
        tried.push(n);
    }
    lib
}

#[allow(
    clippy::too_many_lines,
    reason = "one pass over the mesh, in the order glTF wants it"
)]
fn mesh_document(file: &MeshFile, stem: &str, input: &Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let mut scene = Scene::default();
    let mut facts = Map::new();

    // The skeleton.
    let skeleton = file
        .skeleton
        .as_ref()
        .filter(|_| keep.skins || keep.animations)
        .map(|link| {
            let found = read_named(&skeleton_names(link), input, assets).and_then(|(_, b)| match parse(&b) {
                Ok(Parsed::Skeleton(s)) => Some(s),
                _ => None,
            });
            if found.is_none() {
                facts.insert("skeleton_missing".into(), json!(true));
            }
            found
        });
    let skeleton = skeleton.flatten();

    let mesh_node = scene.add(Node::named(stem, None));
    let bones = skeleton.as_ref().map(|s| add_bones(&mut scene, s));

    // Materials.
    // The scripts are read, and listed, whether or not the materials are kept.
    let lib = library(file, stem, input, assets);
    let mut images = Images::new(!keep.image_bytes, input.dir());
    let mut material_of: HashMap<(String, Vec<(String, String)>), usize> = HashMap::new();
    let mut posts: Vec<MaterialPost> = Vec::new();
    let mut missing_materials: Vec<String> = Vec::new();
    let mut missing_textures: Vec<String> = Vec::new();

    // Each geometry's influences, worked out once.
    let mut unknown_bones = 0usize;
    let geometry = |g: u16| -> Option<&Geometry> {
        if g == 0 {
            file.shared.as_ref()
        } else {
            file.submeshes.get(usize::from(g) - 1)?.geometry.as_ref()
        }
    };
    let mut skin_cache: HashMap<u16, Vec<[(u32, f32); 4]>> = HashMap::new();

    let mut skinned_prims: Vec<Primitive> = Vec::new();
    let mut static_prims: Vec<Primitive> = Vec::new();
    let mut skinned_sources: Vec<Source> = Vec::new();
    let mut static_sources: Vec<Source> = Vec::new();
    let mut dropped_submeshes = 0usize;
    let mut has_bitangent_from_binormal = false;
    for (si, sub) in file.submeshes.iter().enumerate() {
        let gid = if sub.shared {
            0
        } else {
            u16::try_from(si + 1).unwrap_or(u16::MAX)
        };
        let Some(g) = geometry(gid) else {
            dropped_submeshes += 1;
            continue;
        };
        let n = g.positions.len();
        let (mut indices, points, lines) = lists(sub, n);
        let used = if sub.shared {
            // Only the shared vertices this submesh uses, numbered in the order it first uses them.
            let mut remap = vec![u32::MAX; n];
            let mut used = Vec::new();
            for i in &mut indices {
                let Some(r) = remap.get_mut(*i as usize) else {
                    *i = u32::MAX;
                    continue;
                };
                if *r == u32::MAX {
                    *r = crate::num::u32_of(used.len());
                    used.push(*i);
                }
                *i = *r;
            }
            Some(used)
        } else {
            None
        };
        let u = used.as_deref();
        let mut prim = Primitive {
            positions: gather(&g.positions, u),
            indices,
            points,
            lines,
            ..Primitive::default()
        };
        let count = prim.positions.len();
        if g.normals.len() == n {
            prim.normals = gather(&g.normals, u);
        }
        if g.tangents.len() == n {
            let mut t = gather(&g.tangents, u);
            if t.iter().any(|t| t[3].is_nan()) {
                let b = (g.binormals.len() == n && prim.normals.len() == count).then(|| gather(&g.binormals, u));
                if b.is_some() {
                    has_bitangent_from_binormal = true;
                }
                for (k, t) in t.iter_mut().enumerate() {
                    if !t[3].is_nan() {
                        continue;
                    }
                    t[3] = match &b {
                        Some(b) => {
                            let nm = prim.normals[k];
                            let c = [
                                nm[1] * t[2] - nm[2] * t[1],
                                nm[2] * t[0] - nm[0] * t[2],
                                nm[0] * t[1] - nm[1] * t[0],
                            ];
                            if c[0] * b[k][0] + c[1] * b[k][1] + c[2] * b[k][2] < 0.0 {
                                -1.0
                            } else {
                                1.0
                            }
                        }
                        None => 1.0,
                    };
                }
            }
            prim.tangents = t;
        }
        prim.uvs = g.uvs.iter().filter(|s| s.len() == n).map(|s| gather(s, u)).collect();
        let linear = |c: [f32; 4]| [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]];
        if g.diffuse.len() == n {
            prim.colors
                .push(gather(&g.diffuse, u).into_iter().map(linear).collect());
            if g.specular.len() == n {
                prim.colors
                    .push(gather(&g.specular, u).into_iter().map(linear).collect());
            }
        }

        // Material.
        let key = (sub.material.clone(), sub.aliases.clone());
        let index = if let Some(&i) = material_of.get(&key) {
            i
        } else {
            let mapped = keep
                .materials
                .then(|| material::map(&lib, &sub.material, &sub.aliases))
                .flatten();
            let mut post = MaterialPost {
                blend: false,
                color_uv: 0,
                normal_uv: 0,
                extras: Map::new(),
            };
            let mat = if let Some(m) = mapped {
                let mut mat = m.material;
                if let Some(t) = &m.color {
                    mat.color_image = texture(t, input, assets, &mut images);
                    if mat.color_image.is_none() {
                        missing_textures.push(t.file.clone());
                    }
                    mat.sampling.color = t.sampling;
                    post.color_uv = t.uv_set;
                }
                if let Some(t) = &m.normal {
                    mat.normal_image = texture(t, input, assets, &mut images);
                    if mat.normal_image.is_none() {
                        missing_textures.push(t.file.clone());
                    }
                    mat.sampling.normal = t.sampling;
                    post.normal_uv = t.uv_set;
                }
                post.blend = m.blend;
                post.extras = m.extras;
                mat
            } else {
                if keep.materials && !missing_materials.contains(&sub.material) {
                    missing_materials.push(sub.material.clone());
                }
                Material::plain(&sub.material, [1.0; 4])
            };
            scene.materials.push(mat);
            posts.push(post);
            material_of.insert(key, scene.materials.len() - 1);
            scene.materials.len() - 1
        };
        prim.material = Some(index);

        // Skinning.
        let assigns = if sub.shared { &file.shared_bones } else { &sub.bones };
        // A submesh is split off as skinned by its assignments alone, so that the meshes are the same whether or
        // not the skin is kept or its skeleton found.
        let skinned = file.skeleton.is_some() && !assigns.is_empty();
        if skinned && let Some(b) = &bones {
            let inf = skin_cache
                .entry(gid)
                .or_insert_with(|| influences(assigns, n, b, &mut unknown_bones));
            let mut joints = Vec::with_capacity(count * 4);
            let mut weights = Vec::with_capacity(count * 4);
            let mut push = |v: &[(u32, f32); 4]| {
                for s in v {
                    joints.push(s.0);
                    weights.push(s.1);
                }
            };
            match u {
                None => inf.iter().for_each(&mut push),
                Some(u) => u.iter().filter_map(|&i| inf.get(i as usize)).for_each(&mut push),
            }
            prim.joints = joints;
            prim.weights = weights;
            prim.influences = 4;
        }
        let source = Source { geometry: gid, used };
        if skinned {
            skinned_prims.push(prim);
            skinned_sources.push(source);
        } else {
            static_prims.push(prim);
            static_sources.push(source);
        }
    }
    scene.images = images.list;

    // Morph targets: the poses, then each morph keyframe.
    let mut targets: Vec<TargetDef> = Vec::new();
    let mut pose_target: Vec<Option<usize>> = vec![None; file.poses.len()];
    let mut morph_target: Vec<Vec<Vec<Option<usize>>>> = Vec::new();
    let mut budget = MORPH_BUDGET;
    let mut dropped_targets = 0usize;
    let mut dropped_keys = 0usize;
    if keep.morph_targets {
        for (k, p) in file.poses.iter().enumerate() {
            let Some(g) = geometry(p.target) else {
                dropped_targets += 1;
                continue;
            };
            let n = g.positions.len();
            let with_normals = p.offsets.iter().any(|o| o.normal.is_some());
            let cost = n * if with_normals { 6 } else { 3 };
            if cost > budget {
                dropped_targets += 1;
                continue;
            }
            budget -= cost;
            let mut positions = vec![[0.0; 3]; n];
            let mut normals = if with_normals { vec![[0.0; 3]; n] } else { Vec::new() };
            for o in &p.offsets {
                if let Some(v) = positions.get_mut(o.index as usize) {
                    *v = o.offset;
                }
                if let (Some(v), Some(nm)) = (normals.get_mut(o.index as usize), o.normal) {
                    *v = nm;
                }
            }
            pose_target[k] = Some(targets.len());
            targets.push(TargetDef {
                name: p.name.clone(),
                geometry: p.target,
                positions,
                normals,
            });
        }
        for a in &file.animations {
            let mut per_track = Vec::new();
            for t in &a.tracks {
                let mut ids = Vec::new();
                if let Keys::Morph(keys) = &t.keys {
                    let g = geometry(t.target);
                    for (k, key) in keys.iter().enumerate() {
                        let Some(g) = g.filter(|g| g.positions.len() == key.positions.len()) else {
                            dropped_keys += 1;
                            ids.push(None);
                            continue;
                        };
                        let n = g.positions.len();
                        let with_normals = key.normals.len() == n && g.normals.len() == n;
                        let cost = n * if with_normals { 6 } else { 3 };
                        if cost > budget {
                            dropped_keys += 1;
                            ids.push(None);
                            continue;
                        }
                        budget -= cost;
                        let sub = |a: &[[f32; 3]], b: &[[f32; 3]]| -> Vec<[f32; 3]> {
                            a.iter()
                                .zip(b)
                                .map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
                                .collect()
                        };
                        ids.push(Some(targets.len()));
                        targets.push(TargetDef {
                            name: format!("{}/{k}", a.name),
                            geometry: t.target,
                            positions: sub(&key.positions, &g.positions),
                            normals: if with_normals {
                                sub(&key.normals, &g.normals)
                            } else {
                                Vec::new()
                            },
                        });
                    }
                }
                per_track.push(ids);
            }
            morph_target.push(per_track);
        }
    }

    // Each primitive's share of every target.
    let give_targets = |prims: &mut [Primitive], sources: &[Source]| {
        for (p, s) in prims.iter_mut().zip(sources) {
            let n = p.positions.len();
            let normals = targets
                .iter()
                .any(|t| t.geometry == s.geometry && !t.normals.is_empty());
            p.targets = targets
                .iter()
                .map(|t| {
                    if t.geometry != s.geometry {
                        return Target {
                            positions: Vec::new(),
                            normals: if normals { vec![[0.0; 3]; n] } else { Vec::new() },
                        };
                    }
                    let u = s.used.as_deref();
                    Target {
                        positions: gather(&t.positions, u),
                        normals: if t.normals.is_empty() {
                            if normals { vec![[0.0; 3]; n] } else { Vec::new() }
                        } else {
                            gather(&t.normals, u)
                        },
                    }
                })
                .collect();
        }
    };
    if !targets.is_empty() {
        give_targets(&mut skinned_prims, &skinned_sources);
        give_targets(&mut static_prims, &static_sources);
    }
    let target_names: Vec<String> = targets.iter().map(|t| t.name.clone()).collect();
    drop(targets);

    // The meshes: skinned submeshes with the skin, the rest without.
    let mut mesh_nodes = Vec::new();
    let skin_joints = bones.as_ref().filter(|_| !skinned_prims.is_empty());
    let mut extras = Map::new();
    if let Some(names) = submesh_names(file) {
        extras.insert("submeshNames".into(), names);
    }
    let both = !skinned_prims.is_empty() && !static_prims.is_empty();
    for (prims, skinned) in [(skinned_prims, true), (static_prims, false)] {
        if prims.is_empty() {
            continue;
        }
        let node = if mesh_nodes.is_empty() {
            mesh_node
        } else {
            scene.add(Node::named(format!("{stem} (unskinned)"), None))
        };
        scene.nodes[node].mesh = Some(scene.meshes.len());
        if skinned && skin_joints.is_some() {
            scene.nodes[node].skin = Some(0);
        }
        mesh_nodes.push(node);
        let name = if both && !skinned {
            format!("{stem} (unskinned)")
        } else {
            stem.to_owned()
        };
        scene.meshes.push(SceneMesh {
            name,
            prims,
            weights: Vec::new(),
            target_names: target_names.clone(),
            extras: (!extras.is_empty()).then(|| Value::Object(extras.clone())),
        });
    }
    if let Some(b) = skin_joints {
        let world = scene.world();
        scene.skins.push(Skin {
            name: String::new(),
            joints: b.nodes.clone(),
            inverse_binds: b.nodes.iter().map(|&j| scene::invert(&world[j])).collect::<Vec<M4>>(),
        });
    }

    // Animations.
    if let (Some(skel), Some(b)) = (&skeleton, &bones) {
        let links = linked(skel, input, assets, &mut facts);
        add_skeleton_animations(&mut scene, skel, b, &links, &mut facts);
    }
    let count = target_names.len();
    let mut dropped_weights = 0usize;
    if count > 0 {
        for (a, ids) in file.animations.iter().zip(&morph_target) {
            match weights_channel(a, ids, &pose_target, count, &mut budget) {
                Some((times, values)) => {
                    let channels = mesh_nodes
                        .iter()
                        .map(|&n| channel(n, KeyPath::Weights, times.clone(), values.clone()))
                        .collect();
                    let mut extras = json!({ "length": a.length });
                    if let Some((name, time)) = &a.base {
                        extras["baseAnimation"] = json!(name);
                        extras["baseKeyframeTime"] = json!(time);
                    }
                    scene.animations.push(Animation {
                        name: a.name.clone(),
                        channels,
                        extras: Some(extras),
                    });
                }
                None => dropped_weights += 1,
            }
        }
    }

    // Source and facts.
    convention(&mut scene, &file.version);
    if file.big_endian {
        scene.source.insert("byteOrder".into(), json!("big-endian"));
    }
    if let Some(s) = &file.skeleton {
        scene.source.insert("skeleton".into(), json!(s));
    }
    if let Some(skel) = &skeleton {
        scene.source.insert("skeletonVersion".into(), json!(skel.version));
        if let Some(m) = skel.blend_mode {
            scene.source.insert("blendMode".into(), json!(m));
        }
        facts.insert("joints".into(), json!(skel.bones.len()));
        facts.insert("animations".into(), json!(skel.animations.len()));
    }
    if !lib.scripts.is_empty() {
        let names: Vec<String> = lib
            .scripts
            .iter()
            .map(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        scene.source.insert("materialScripts".into(), json!(names));
    }
    facts.insert("submeshes".into(), json!(file.submeshes.len()));
    let mut put = |k: &str, n: usize| {
        if n > 0 {
            facts.insert(k.into(), json!(n));
        }
    };
    put("lod_levels", file.lods.len());
    put("edge_lists", file.edge_lists);
    put("extremes", file.extremes);
    put("poses", file.poses.len());
    put("mesh_animations", file.animations.len());
    put("morph_targets", target_names.len());
    put("dropped_submeshes", dropped_submeshes);
    put("dropped_morph_targets", dropped_targets);
    put("dropped_morph_keys", dropped_keys);
    put("dropped_mesh_animations", dropped_weights);
    put("unknown_bone_assignments", unknown_bones);
    put("texcoords_cut_to_2d", file.texcoord_w);
    if !file.lods.is_empty() {
        let manual: Vec<&str> = file.lods.iter().filter_map(|l| l.manual.as_deref()).collect();
        if !manual.is_empty() {
            facts.insert("lod_meshes".into(), json!(manual));
        }
        let usage: Vec<f32> = file.lods.iter().map(|l| l.usage).collect();
        facts.insert("lod_usage".into(), json!(usage));
        if let Some(s) = &file.lod_strategy {
            facts.insert("lod_strategy".into(), json!(s));
        }
    }
    if let Some((min, max, radius)) = file.bounds {
        facts.insert("bounds".into(), json!({ "min": min, "max": max, "radius": radius }));
    }
    if !file.dropped_elements.is_empty() {
        facts.insert("dropped_vertex_elements".into(), json!(file.dropped_elements));
    }
    if has_bitangent_from_binormal {
        facts.insert("tangent_sign_from_binormals".into(), json!(true));
    }
    if !missing_materials.is_empty() {
        facts.insert("missing_materials".into(), json!(missing_materials));
    }
    missing_textures.sort();
    missing_textures.dedup();
    if !missing_textures.is_empty() {
        facts.insert("missing_textures".into(), json!(missing_textures));
    }
    scene.facts.extend(facts);

    let mut doc = scene.finish(keep)?;
    finish_materials(&mut doc, &posts, keep);
    Ok(doc)
}

/// The submesh names, when any is named: index to name.
fn submesh_names(file: &MeshFile) -> Option<Value> {
    let names: Map<String, Value> = file
        .submeshes
        .iter()
        .enumerate()
        .filter_map(|(k, s)| s.name.as_ref().map(|n| (k.to_string(), json!(n))))
        .collect();
    (!names.is_empty()).then_some(Value::Object(names))
}

/// Linear weights between two keys of a track at `time`: the earlier key's index, the later's, and the later's
/// share.
fn between(times: &[f32], time: f32) -> (usize, usize, f32) {
    let last = times.len() - 1;
    if time <= times[0] {
        return (0, 0, 0.0);
    }
    if time >= times[last] {
        return (last, last, 0.0);
    }
    let k = times.partition_point(|&t| t <= time).clamp(1, last);
    let (a, b) = (times[k - 1], times[k]);
    let f = if b > a { (time - a) / (b - a) } else { 0.0 };
    (k - 1, k, f)
}

/// A mesh animation as one weights channel: at every key time of any track, each target's weight (morph
/// keyframes blending into the next, poses at their keys' influences).
fn weights_channel(
    a: &MeshAnimation,
    ids: &[Vec<Option<usize>>],
    pose_target: &[Option<usize>],
    count: usize,
    budget: &mut usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let mut times: Vec<f32> = a
        .tracks
        .iter()
        .flat_map(|t| -> Vec<f32> {
            match &t.keys {
                Keys::Morph(k) => k.iter().map(|k| k.time).collect(),
                Keys::Pose(k) => k.iter().map(|k| k.time).collect(),
            }
        })
        .filter(|t| t.is_finite())
        .collect();
    times.sort_by(f32::total_cmp);
    times.dedup();
    let cost = times.len().checked_mul(count)?;
    if times.is_empty() || cost > *budget {
        return None;
    }
    *budget -= cost;
    let mut values = vec![0.0f32; cost];
    for (t, track_ids) in a.tracks.iter().zip(ids.iter().map(Some).chain(std::iter::repeat(None))) {
        match &t.keys {
            Keys::Morph(keys) => {
                let Some(track_ids) = track_ids else { continue };
                let key_times: Vec<f32> = keys.iter().map(|k| k.time).collect();
                if key_times.is_empty() {
                    continue;
                }
                for (ti, &time) in times.iter().enumerate() {
                    let (k0, k1, f) = between(&key_times, time);
                    let row = &mut values[ti * count..(ti + 1) * count];
                    if let Some(Some(id)) = track_ids.get(k0) {
                        row[*id] += 1.0 - f;
                    }
                    if f > 0.0
                        && let Some(Some(id)) = track_ids.get(k1)
                    {
                        row[*id] += f;
                    }
                }
            }
            Keys::Pose(keys) => {
                let key_times: Vec<f32> = keys.iter().map(|k| k.time).collect();
                if key_times.is_empty() {
                    continue;
                }
                let weight = |k: usize, pose: usize| -> f32 {
                    keys[k]
                        .refs
                        .iter()
                        .filter(|r| usize::from(r.0) == pose)
                        .map(|r| r.1)
                        .sum::<f32>()
                };
                let poses: Vec<usize> = {
                    let mut p: Vec<usize> = keys
                        .iter()
                        .flat_map(|k| k.refs.iter().map(|r| usize::from(r.0)))
                        .collect();
                    p.sort_unstable();
                    p.dedup();
                    p
                };
                for (ti, &time) in times.iter().enumerate() {
                    let (k0, k1, f) = between(&key_times, time);
                    let row = &mut values[ti * count..(ti + 1) * count];
                    for &pose in &poses {
                        let Some(Some(id)) = pose_target.get(pose) else {
                            continue;
                        };
                        let w = weight(k0, pose) * (1.0 - f) + if f > 0.0 { weight(k1, pose) * f } else { 0.0 };
                        if w.is_finite() {
                            row[*id] += w;
                        }
                    }
                }
            }
        }
    }
    Some((times, values))
}

/// Sets what the shared material writer doesn't: blending by a texture's alpha, texture coordinate sets past the
/// first, and the scripts' other properties as extras.
fn finish_materials(doc: &mut Document, posts: &[MaterialPost], keep: Keep) {
    let Some(Value::Array(mats)) = doc.base_mut().get_mut("materials") else {
        return;
    };
    for (m, post) in mats.iter_mut().zip(posts) {
        if post.blend && m.get("alphaMode").is_none() {
            m["alphaMode"] = json!("BLEND");
        }
        if post.color_uv > 0
            && let Some(t) = m.pointer_mut("/pbrMetallicRoughness/baseColorTexture")
        {
            t["texCoord"] = json!(post.color_uv);
        }
        if post.normal_uv > 0
            && let Some(t) = m.get_mut("normalTexture")
        {
            t["texCoord"] = json!(post.normal_uv);
        }
        if keep.extras && !post.extras.is_empty() {
            m["extras"] = json!({ "ogre": post.extras });
        }
    }
}

#[cfg(test)]
mod tests;
