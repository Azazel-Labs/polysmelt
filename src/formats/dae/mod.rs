//! COLLADA (`.dae`, versions 1.4 and 1.5) as a glTF document, and the `.zae` and `.kmz` archives that hold one.
//!
//! The visual scene becomes the node hierarchy, each node's transforms (`matrix`, `translate`, `rotate`, `scale`,
//! `lookat`, `skew`) composed into its local transform; `instance_node` places a copy of a node's subtree (a node
//! that instances itself is cut off). The document's unit and up axis become a root node that carries the scene to
//! glTF's Y-up meters, and are recorded in `asset.extras.source` (see [`crate::source`]) with what else the
//! `<asset>` says: its contributors, title, subject, keywords and dates; a node's own `<asset>` changes them below it. A
//! document with no scene shows its geometries. URLs into other documents (geometry, nodes, materials, lights and
//! cameras in other `.dae` files, beside the model or in the archive) are followed, each document read once and
//! with its own unit, up axis and image folder.
//!
//! Meshes keep every primitive that draws triangles (`triangles`, `polylist`, `polygons` with their holes,
//! `trifans`, `tristrips`) or lines (`lines`, and `linestrips` opened into segments), with normals, every texture coordinate and color set, and tangents. A vertex is one of
//! each distinct combination of the indices a corner reads, so attributes indexed apart are kept apart; texture
//! coordinates are flipped to glTF's top-left origin and colors made linear. A geometry is shared by the instances
//! that bind it to the same materials.
//!
//! Materials come from `profile_COMMON` (`constant`, `lambert`, `phong`, `blinn`): the diffuse color or texture,
//! emission, a roughness from the shininess, the specular color (`KHR_materials_specular`) and index of refraction
//! (`KHR_materials_ior`), transparency in each of its opaque modes, the bump map the common extras name (a height
//! map is left out), texture placement extras as `KHR_texture_transform`, and two-sidedness from the effect's or the
//! geometry's extras. `<setparam>` in an `<instance_effect>` overrides the effect's parameters. Textures are found
//! through their samplers and surfaces (both versions' styles), read from the archive, embedded, or found beside
//! the model.
//!
//! Skin controllers become glTF skins (up to the sixteen strongest influences of each vertex, in as many sets of
//! joints and weights as they take, the bind shape folded into the inverse bind matrices), morph controllers morph
//! targets with their default weights (positions, and the normals a target changes when it has the base's
//! topology). Animations of transforms (whole, or one component of one) and of morph
//! weights are sampled into translation, rotation, scale and weight tracks, Bézier, Hermite, B-spline and cardinal
//! curves and long turns cut into short steps; animation clips become glTF animations of their own. Cameras and
//! lights (`KHR_lights_punctual`) come across with their nodes; an ambient light, which glTF lacks, becomes the
//! node's extras, as do a light's attenuation and falloff. The `<extra>` of nodes, geometries, effects, materials,
//! cameras and lights is kept as their extras, by profile, while `Keep::extras` is set.

mod animation;
mod effect;
mod geometry;
#[cfg(test)]
mod tests;
mod xml;

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Cursor, Read as _};
use std::path::Path;

use serde_json::{Map, Value, json};
use zip::ZipArchive;

use super::Input;
use super::xform::{IDENTITY, M4, inverse, mul, scale, set_transform, to_column_major, translate};
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep, Part};
use crate::images::Images;
use crate::material::Full;
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::vec3::{self, cross, dot, sub};
use crate::{Context as _, ErrorKind, Result};
use xml::Dom;

/// How deeply nodes nest, `instance_node` included.
const MAX_DEPTH: usize = 128;

/// The most nodes a document makes: instanced subtrees can multiply.
const MAX_NODES: usize = 1 << 20;

/// The largest archive entry read.
const MAX_ENTRY: u64 = 1 << 31;

use crate::mesh::MAX_INFLUENCES;

/// The most other documents a document's URLs bring in, theirs included.
const MAX_DOCUMENTS: usize = 64;

/// The most elements of `<extra>` kept as an element's extras.
const MAX_EXTRA_VALUES: usize = 4096;

/// The document in a `.dae` file or an archive, as glTF.
pub(super) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let raw = input.read()?;
    let owned: Vec<u8>;
    let mut archive = None;
    let mut path = input
        .path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let text: &[u8] = if raw.starts_with(b"PK\x03\x04") {
        let mut zip = ZipArchive::new(Cursor::new(&raw[..])).context(ErrorKind::Corrupt, "bad ZIP archive")?;
        let entry = root_entry(&mut zip).context(ErrorKind::UnknownFormat, "the archive holds no COLLADA document")?;
        let mut bytes = Vec::new();
        zip.by_name(&entry)
            .with_context(ErrorKind::Corrupt, || format!("reading {entry}"))?
            .take(MAX_ENTRY)
            .read_to_end(&mut bytes)?;
        let (dir, name) = entry.rsplit_once('/').map_or_else(
            || (String::new(), entry.clone()),
            |(d, n)| (format!("{d}/"), n.to_owned()),
        );
        path = name;
        archive = Some(Archive { zip, dir });
        owned = bytes;
        &owned
    } else {
        &raw
    };
    let text = decode(text);
    let mut dom = Dom::parse(&text);
    let root = dom
        .root()
        .filter(|&r| dom.is(r, "COLLADA"))
        .context(ErrorKind::UnknownFormat, "not a COLLADA document")?;
    let home = archive
        .as_ref()
        .map_or_else(|| input.dir().to_string_lossy().replace('\\', "/"), |a| a.dir.clone());
    dom.set_home(&path, &home);
    // The other documents its URLs lead into, and theirs in turn.
    let (mut from, mut documents) = (0, 0);
    loop {
        let wanted = dom.wanted(from);
        from = dom.len();
        if wanted.is_empty() || documents >= MAX_DOCUMENTS {
            break;
        }
        for path in wanted.into_iter().take(MAX_DOCUMENTS - documents) {
            documents += 1;
            let bytes = read_document(&path, archive.as_mut(), assets, input.dir()).unwrap_or_default();
            let text = decode(&bytes);
            let other = Dom::parse(&text);
            if other.root().is_some_and(|r| other.is(r, "COLLADA")) {
                dom.absorb(&path, other);
            } else {
                dom.absorb(&path, Dom::parse(b""));
            }
        }
    }
    let dom = dom;

    let asset = dom.child(root, "asset");
    let meters = asset.and_then(|a| unit(&dom, a)).unwrap_or(1.0);
    let up = asset
        .and_then(|a| dom.child_string(a, "up_axis"))
        .unwrap_or("Y_UP")
        .to_owned();

    let mut r = Reader {
        dom: &dom,
        assets,
        dir: input.dir(),
        base: dom.attr(root, "base").map(xml::percent_decode),
        keep,
        meters,
        archive,
        b: Builder::default(),
        sources: HashMap::new(),
        nodes: Vec::new(),
        parents: Vec::new(),
        instances: HashMap::new(),
        meshes: Vec::new(),
        mesh_triangles: Vec::new(),
        mesh_of: HashMap::new(),
        drawn_as: HashMap::new(),
        points_of: HashMap::new(),
        normals_of: HashMap::new(),
        materials: Vec::new(),
        material_of: HashMap::new(),
        images: Images::new(!keep.image_bytes, input.dir()),
        image_of: HashMap::new(),
        archive_images: HashMap::new(),
        skins: Vec::new(),
        cameras: Vec::new(),
        camera_of: HashMap::new(),
        lights: Vec::new(),
        light_of: HashMap::new(),
        pending: Vec::new(),
        morphs: Vec::new(),
        weighted: HashMap::new(),
        skinned_roots: Vec::new(),
        drawn: 0,
        limit: input.limit,
        asset_of: HashMap::new(),
    };
    for e in 0..dom.len() {
        if dom.is(e, "asset")
            && let Some(p) = dom.parent(e)
        {
            r.asset_of.entry(p).or_insert(e);
        }
    }

    let scene = dom
        .child(root, "scene")
        .and_then(|s| dom.child(s, "instance_visual_scene"))
        .and_then(|i| dom.link(i, "url"))
        .filter(|&v| dom.is(v, "visual_scene"))
        .or_else(|| {
            dom.kids(root, "library_visual_scenes")
                .find_map(|l| dom.child(l, "visual_scene"))
        });
    let mut roots = Vec::new();
    if let Some(vs) = scene {
        let mut stack = Vec::new();
        for c in dom.children(vs) {
            // An `<instance_node>` straight under the scene, where only nodes belong, places its node as a root.
            let el = match dom.name(c) {
                b"node" => Some(c),
                b"instance_node" => dom.link(c, "url").filter(|&t| dom.is(t, "node")),
                _ => None,
            };
            if let Some(el) = el
                && let Some(n) = r.node(el, root, 0, &mut stack)?
            {
                roots.push(n);
            }
        }
    } else {
        // No scene: each geometry at the origin.
        for lib in dom.kids(root, "library_geometries") {
            for g in dom.kids(lib, "geometry") {
                let n = r.push_node(json!({}), None);
                if let Some(name) = dom.attr(g, "name").or_else(|| dom.attr(g, "id")) {
                    r.nodes[n]["name"] = json!(name);
                }
                let mut children = Vec::new();
                r.attach(g, g, &[], n, &[], &mut children)?;
                r.adopt(n, &children);
                roots.push(n);
            }
        }
    }
    r.skin_meshes();
    r.morph_meshes();
    if r.nodes.is_empty() {
        return Ok(Document::empty(Map::new()));
    }
    let animations = if keep.animations { r.animations() } else { Vec::new() };

    let basis = basis(meters, &up);
    let mut scene_nodes = roots;
    #[allow(clippy::float_cmp, reason = "only the exact identity needs no root")]
    if basis != IDENTITY {
        let mut root_node = json!({ "name": "RootNode", "children": scene_nodes });
        set_transform(&mut root_node, &basis);
        r.nodes.push(root_node);
        scene_nodes = vec![r.nodes.len() - 1];
    }
    scene_nodes.append(&mut r.skinned_roots);

    let source = source_info(&dom, root, asset, meters, &up);
    let mut asset_json = json!({});
    if let Some(c) = source.get("copyright") {
        asset_json["copyright"] = c.clone();
    }
    asset_json["extras"] = json!({ "source": source });
    let mut json = Map::new();
    json.insert("asset".into(), asset_json);
    json.insert("scene".into(), json!(0));
    let mut scene_json = json!({ "nodes": scene_nodes });
    if let Some(name) = scene.and_then(|s| dom.attr(s, "name").or_else(|| dom.attr(s, "id"))) {
        scene_json["name"] = json!(name);
    }
    json.insert("scenes".into(), json!([scene_json]));
    json.insert("nodes".into(), Value::Array(std::mem::take(&mut r.nodes)));
    if !r.meshes.is_empty() {
        json.insert("meshes".into(), Value::Array(std::mem::take(&mut r.meshes)));
    }
    if !r.skins.is_empty() {
        json.insert("skins".into(), Value::Array(std::mem::take(&mut r.skins)));
    }
    if !animations.is_empty() {
        json.insert("animations".into(), Value::Array(animations));
    }
    if !r.cameras.is_empty() {
        json.insert("cameras".into(), Value::Array(std::mem::take(&mut r.cameras)));
    }
    if !r.lights.is_empty() {
        json.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
        json.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": std::mem::take(&mut r.lights) } }),
        );
    }
    let materials = std::mem::take(&mut r.materials);
    let images = std::mem::take(&mut r.images);
    let mut b = r.b;
    b.materials(materials, images.list, &mut json);
    Ok(b.finish(json, Map::new()))
}

/// The entry of an archive that holds its document: the one `manifest.xml` names (a `.zae`), or the first model a
/// KML document links to (a `.kmz`), else the shallowest `.dae`.
fn root_entry(zip: &mut ZipArchive<Cursor<&[u8]>>) -> Option<String> {
    let names: Vec<String> = zip.file_names().filter_map(Result::ok).map(Cow::into_owned).collect();
    let mut read = |name: &str| {
        let mut bytes = Vec::new();
        zip.by_name(name)
            .ok()
            .and_then(|f| f.take(1 << 24).read_to_end(&mut bytes).ok())
            .map(|_| bytes)
    };
    // The entry a path names, from the folder `dir` (empty or with a trailing `/`), any case.
    let entry = |dir: &str, path: &str| {
        let path = xml::percent_decode(path.trim());
        let path = path.split(['#', '?']).next().unwrap_or("");
        let full = effect::normalize(&format!("{dir}{path}"));
        names
            .iter()
            .find(|e| **e == full)
            .or_else(|| names.iter().find(|e| e.eq_ignore_ascii_case(&full)))
            .cloned()
    };
    if let Some(manifest) = names.iter().find(|n| n.eq_ignore_ascii_case("manifest.xml"))
        && let Some(bytes) = read(manifest)
    {
        let dom = Dom::parse(&bytes);
        let named = dom.root().and_then(|r| {
            if dom.is(r, "dae_root") {
                Some(r)
            } else {
                dom.descendant(r, "dae_root", 4)
            }
        });
        if let Some(found) = named.and_then(|e| entry("", dom.string(e))) {
            return Some(found);
        }
    }
    let kml = names
        .iter()
        .filter(|n| n.to_ascii_lowercase().ends_with(".kml"))
        .min_by_key(|n| (n.matches('/').count(), n.len(), n.as_str()));
    if let Some(kml) = kml
        && let Some(bytes) = read(kml)
    {
        let dir = kml.rsplit_once('/').map_or_else(String::new, |(d, _)| format!("{d}/"));
        let text = decode(&bytes);
        let dom = Dom::parse(&text);
        let href = dom.root().and_then(|r| {
            let model = dom.descendant(r, "Model", 16)?;
            dom.descendant(model, "href", 2)
        });
        if let Some(found) = href.and_then(|h| entry(&dir, dom.string(h))) {
            return Some(found);
        }
    }
    names
        .iter()
        .filter(|n| n.to_ascii_lowercase().ends_with(".dae"))
        .min_by_key(|n| (n.matches('/').count(), n.len(), n.as_str()))
        .cloned()
}

/// The bytes of the document at `path` (from the first document's folder): an entry of the archive the first one
/// came in, else a file beside it.
fn read_document(path: &str, archive: Option<&mut Archive<'_>>, assets: &dyn Assets, dir: &Path) -> Option<Vec<u8>> {
    if let Some(a) = archive {
        let full = xml::join(&a.dir, path);
        let names: Vec<String> = a.zip.file_names().filter_map(Result::ok).map(Cow::into_owned).collect();
        let name = names
            .iter()
            .find(|n| **n == full)
            .or_else(|| names.iter().find(|n| n.eq_ignore_ascii_case(&full)));
        if let Some(name) = name {
            let mut bytes = Vec::new();
            a.zip.by_name(name).ok()?.take(MAX_ENTRY).read_to_end(&mut bytes).ok()?;
            return Some(bytes);
        }
    }
    let found = assets.find(path, dir)?;
    assets.read(&found).map(Cow::into_owned)
}

/// What the document's `<asset>` says of it, as `asset.extras.source`: its unit and up axis (COLLADA's meter and
/// Y up when it doesn't say), every contributor's author, tool, copyright and comments, its title, subject,
/// keywords, and when it was made and changed. `revision`, a 1.5 document's `geographic_location` (as
/// `georeference`), and each contributor in full when there are several (`contributors`) are kept too.
fn source_info(dom: &Dom<'_>, root: usize, asset: Option<usize>, meters: f64, up: &str) -> Map<String, Value> {
    let text = |e: usize, name: &str| {
        dom.child(e, name)
            .map(|c| xml::unescape(dom.text(c)).trim().to_owned())
            .filter(|s| !s.is_empty())
    };
    let declared_unit = asset.and_then(|a| unit(dom, a)).is_some();
    let up_given = asset.and_then(|a| dom.child_string(a, "up_axis")).is_some();
    let axis = match up {
        "Z_UP" => Axis::PosZ,
        "X_UP" => Axis::PosX,
        _ => Axis::PosY,
    };
    let mut info = SourceInfo {
        format: Some("COLLADA".into()),
        version: dom.attr(root, "version").map(str::to_owned),
        meters_per_unit: if declared_unit {
            Fact::Declared(meters)
        } else {
            Fact::Assumed(meters)
        },
        unit: asset
            .and_then(|a| dom.child(a, "unit"))
            .filter(|_| declared_unit)
            .and_then(|u| dom.attr(u, "name"))
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(str::to_owned)
            .or_else(|| ((meters - 1.0).abs() < 1e-12).then(|| "meter".to_owned())),
        up_axis: if up_given {
            Fact::Declared(axis)
        } else {
            Fact::Assumed(axis)
        },
        // The specification fixes it.
        handedness: Fact::Declared(Handedness::Right),
        ..SourceInfo::default()
    };
    info.converted.units = (meters - 1.0).abs() > 1e-12;
    info.converted.up_axis = axis != Axis::PosY;
    let mut details = Map::new();
    if let Some(a) = asset {
        let (mut tools, mut authors, mut copyrights, mut contributors) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for c in dom.kids(a, "contributor") {
            let mut one = Map::new();
            for field in [
                "author",
                "author_email",
                "author_website",
                "authoring_tool",
                "comments",
                "copyright",
                "source_data",
            ] {
                if let Some(v) = text(c, field) {
                    match field {
                        "author" => authors.push(v.clone()),
                        "authoring_tool" => tools.push(v.clone()),
                        "copyright" => copyrights.push(v.clone()),
                        "comments" => info.comments.push(v.clone()),
                        _ => {}
                    }
                    one.insert(field.into(), json!(v));
                }
            }
            if !one.is_empty() {
                contributors.push(Value::Object(one));
            }
        }
        let joined = |mut v: Vec<String>| {
            let mut seen = std::collections::HashSet::new();
            v.retain(|s| seen.insert(s.clone()));
            (!v.is_empty()).then(|| v.join("; "))
        };
        info.tool = joined(tools);
        info.author = joined(authors);
        info.copyright = joined(copyrights);
        if contributors.len() > 1 {
            details.insert("contributors".into(), Value::Array(contributors));
        }
        info.title = text(a, "title");
        info.description = text(a, "subject");
        info.created = text(a, "created");
        info.modified = text(a, "modified");
        info.keywords = text(a, "keywords")
            .map(|k| {
                let list: Vec<String> = if k.contains([',', ';']) {
                    k.split([',', ';'])
                        .map(str::trim)
                        .filter(|k| !k.is_empty())
                        .map(str::to_owned)
                        .collect()
                } else {
                    k.split_whitespace().map(str::to_owned).collect()
                };
                list
            })
            .unwrap_or_default();
        if let Some(r) = text(a, "revision") {
            details.insert("revision".into(), json!(r));
        }
        if let Some(g) = dom.path(a, &["coverage", "geographic_location"]) {
            let mut geo = Map::new();
            for k in ["longitude", "latitude"] {
                if let Some(v) = text(g, k).and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite()) {
                    geo.insert(k.into(), json!(v));
                }
            }
            if let Some(alt) = dom.child(g, "altitude") {
                if let Ok(v) = dom.string(alt).parse::<f64>()
                    && v.is_finite()
                {
                    geo.insert("altitude".into(), json!(v));
                }
                if let Some(mode) = dom.attr(alt, "mode") {
                    geo.insert("altitudeMode".into(), json!(mode));
                }
            }
            if !geo.is_empty() {
                info.georeference = Some(Value::Object(geo));
            }
        }
    }
    info.write(&mut details);
    details
}

/// The meters in the unit an `<asset>` gives.
fn unit(dom: &Dom<'_>, asset: usize) -> Option<f64> {
    dom.child(asset, "unit")
        .and_then(|u| dom.attr_num::<f64>(u, "meter"))
        .filter(|m| m.is_finite() && *m > 0.0)
}

/// The transform that carries a frame in `meters` with up axis `up` to glTF's Y-up meters.
fn basis(meters: f64, up: &str) -> M4 {
    let turn = match up {
        "Z_UP" => [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]],
        "X_UP" => [[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
        _ => IDENTITY,
    };
    mul(&turn, &scale([meters; 3]))
}

/// The document's text as UTF-8: a byte order mark dropped, UTF-16 converted.
fn decode(bytes: &[u8]) -> Cow<'_, [u8]> {
    crate::formats::xml::utf8(bytes)
}

/// The archive a document came in, for the textures beside it.
struct Archive<'a> {
    zip: ZipArchive<Cursor<&'a [u8]>>,
    /// The document's folder in the archive, with a trailing `/` (or empty).
    dir: String,
}

/// A material an instance binds to a symbol, and the coordinate sets it binds texture semantics to.
struct Binding {
    symbol: String,
    material: Option<usize>,
    inputs: Vec<(String, u32)>,
}

/// A mesh that a skin moves, to bind once every node is made.
struct PendingSkin {
    mesh: usize,
    skin: usize,
    /// The instance's `<skeleton>` roots, where joints named by `sid` are looked for first.
    roots: Vec<usize>,
    /// The node the instance is on, which holds the mesh if the skin can't be bound.
    node: usize,
    name: Option<String>,
}

/// What a mesh's material is looked up by: the material and the coordinate sets its textures read.
type MaterialKey = (usize, Vec<(String, u32)>, bool);

/// What a mesh is shared by: what's instanced (geometry or controller) and how its symbols are bound.
type MeshKey = (usize, Vec<(String, Option<usize>, Vec<(String, u32)>)>);

/// A mesh drawn for a static geometry, and each of its primitives' material symbol and texture
/// coordinate sets.
type Drawn = (usize, Vec<(Option<String>, Vec<u32>)>);

pub(super) struct Reader<'a> {
    dom: &'a Dom<'a>,
    assets: &'a dyn Assets,
    /// The model's folder.
    dir: &'a Path,
    /// The document's `xml:base`, which relative image paths are under.
    base: Option<String>,
    keep: Keep,
    meters: f64,
    archive: Option<Archive<'a>>,
    b: Builder,
    sources: geometry::Sources,
    nodes: Vec<Value>,
    parents: Vec<Option<usize>>,
    /// The nodes made for each `<node>` element, the first one first.
    instances: HashMap<usize, Vec<usize>>,
    meshes: Vec<Value>,
    mesh_triangles: Vec<usize>,
    mesh_of: HashMap<MeshKey, Option<usize>>,
    /// A static geometry's first mesh and each of its primitives' material symbol and coordinate sets: another
    /// binding of the geometry is a mesh on the same accessors.
    drawn_as: HashMap<usize, Drawn>,
    /// Each skinned or morphed mesh's primitives' vertices, as indices into its geometry's `<vertices>`.
    points_of: HashMap<usize, Vec<Vec<u32>>>,
    /// Each morphed mesh's primitives' normals (empty for one without), which its targets' normals are taken from.
    normals_of: HashMap<usize, Vec<Vec<[f32; 3]>>>,
    materials: Vec<Full>,
    material_of: HashMap<MaterialKey, usize>,
    images: Images,
    image_of: HashMap<usize, Option<usize>>,
    archive_images: HashMap<String, Option<usize>>,
    skins: Vec<Value>,
    cameras: Vec<Value>,
    camera_of: HashMap<usize, Option<usize>>,
    lights: Vec<Value>,
    light_of: HashMap<usize, Option<usize>>,
    pending: Vec<PendingSkin>,
    /// Morphed meshes and their `<morph>` elements, for the targets written after every mesh and the animations of
    /// their weights.
    morphs: Vec<(usize, usize)>,
    /// The skinned meshes that have their joints and weights, by the `<skin>` they're from.
    weighted: HashMap<usize, usize>,
    skinned_roots: Vec<usize>,
    drawn: usize,
    limit: usize,
    /// Each element's own `<asset>`, looked for on every ancestor of what's placed.
    asset_of: HashMap<usize, usize>,
}

impl Reader<'_> {
    fn push_node(&mut self, node: Value, parent: Option<usize>) -> usize {
        self.nodes.push(node);
        self.parents.push(parent);
        self.nodes.len() - 1
    }

    /// Makes `children` the children of `parent`.
    fn adopt(&mut self, parent: usize, children: &[usize]) {
        if children.is_empty() {
            return;
        }
        for &c in children {
            self.parents[c] = Some(parent);
        }
        let list = self.nodes[parent]
            .as_object_mut()
            .map(|o| o.entry("children").or_insert_with(|| json!([])));
        if let Some(Value::Array(list)) = list {
            list.extend(children.iter().map(|&c| json!(c)));
        }
    }

    /// The node for `<node>` element `el` placed from element `from` (its parent, or what instances it), with its
    /// subtree; `None` past the depth or node limits.
    fn node(&mut self, el: usize, from: usize, depth: usize, stack: &mut Vec<usize>) -> Result<Option<usize>> {
        if depth > MAX_DEPTH || self.nodes.len() >= MAX_NODES || stack.contains(&el) {
            return Ok(None);
        }
        let dom = self.dom;
        let mut node = json!({});
        if let Some(name) = dom
            .attr(el, "name")
            .or_else(|| dom.attr(el, "id"))
            .or_else(|| dom.attr(el, "sid"))
        {
            node["name"] = json!(name);
        }
        let mut local = self.local(el);
        if let Some(c) = self.correction(from, el) {
            local = mul(&c, &local);
        }
        set_transform(&mut node, &local);
        if let Some(e) = self.extras_of(&[el]) {
            node["extras"] = e;
        }
        let n = self.push_node(node, None);
        self.instances.entry(el).or_default().push(n);
        stack.push(el);
        let mut children = Vec::new();
        for c in dom.children(el) {
            match dom.name(c) {
                b"node" => {
                    if let Some(k) = self.node(c, el, depth + 1, stack)? {
                        children.push(k);
                    }
                }
                b"instance_node" => {
                    if let Some(t) = dom.link(c, "url").filter(|&t| dom.is(t, "node"))
                        && let Some(k) = self.node(t, el, depth + 1, stack)?
                    {
                        children.push(k);
                    }
                }
                b"instance_geometry" | b"instance_controller" => {
                    if let Some(target) = dom.link(c, "url") {
                        let binds = self.bindings(c);
                        let roots: Vec<usize> = dom
                            .kids(c, "skeleton")
                            .filter_map(|s| dom.url_at(s, dom.string(s)))
                            .collect();
                        self.attach(el, target, &binds, n, &roots, &mut children)?;
                    }
                }
                b"instance_camera" if self.keep.cameras => {
                    if let Some(cam) = dom.link(c, "url").and_then(|t| self.camera(t)) {
                        let holder = self.holder(n, "camera", &mut children);
                        self.nodes[holder]["camera"] = json!(cam);
                    }
                }
                b"instance_light" if self.keep.lights => {
                    if let Some(t) = dom.link(c, "url") {
                        self.light(t, n, &mut children);
                    }
                }
                _ => {}
            }
        }
        stack.pop();
        self.adopt(n, &children);
        Ok(Some(n))
    }

    /// Node `n` when it doesn't have `key` yet, else a new child of it.
    fn holder(&mut self, n: usize, key: &str, children: &mut Vec<usize>) -> usize {
        if self.nodes[n].get(key).is_none() {
            return n;
        }
        let c = self.push_node(json!({}), Some(n));
        if let Some(name) = self.nodes[n].get("name").cloned() {
            self.nodes[c]["name"] = name;
        }
        children.push(c);
        c
    }

    /// The frame `e` is written in, as the transform to glTF's Y-up meters: the unit and the up axis of the
    /// nearest `<asset>` around it that gives each (a node's own, its document's, ...).
    fn frame(&self, e: usize) -> M4 {
        let dom = self.dom;
        let (mut meters, mut up) = (None, None);
        let mut at = Some(e);
        for _ in 0..MAX_DEPTH * 4 {
            let Some(a) = at else { break };
            if let Some(&asset) = self.asset_of.get(&a) {
                meters = meters.or_else(|| unit(dom, asset));
                up = up.or_else(|| dom.child_string(asset, "up_axis"));
            }
            if meters.is_some() && up.is_some() {
                break;
            }
            at = dom.parent(a);
        }
        basis(meters.unwrap_or(1.0), up.unwrap_or("Y_UP"))
    }

    /// What carries `inner`'s frame into `outer`'s, when they differ.
    fn correction(&self, outer: usize, inner: usize) -> Option<M4> {
        let (a, b) = (self.frame(outer), self.frame(inner));
        #[allow(clippy::float_cmp, reason = "the same unit and axis give the same matrix")]
        (a != b).then(|| mul(&inverse(&a), &b))
    }

    /// A node's local transform: its transform elements composed in order.
    fn local(&self, el: usize) -> M4 {
        let dom = self.dom;
        let mut m = IDENTITY;
        for c in dom.children(el) {
            if let Some(kind) = Transform::of(dom.name(c)) {
                let mut v = Vec::new();
                xml::floats(dom.text(c), &mut v);
                m = mul(&m, &kind.matrix(&v));
            }
        }
        m
    }

    /// The materials an instance binds.
    fn bindings(&self, inst: usize) -> Vec<Binding> {
        let dom = self.dom;
        let Some(tc) = dom.path(inst, &["bind_material", "technique_common"]) else {
            return Vec::new();
        };
        dom.kids(tc, "instance_material")
            .map(|im| Binding {
                symbol: dom.attr(im, "symbol").unwrap_or("").to_owned(),
                material: dom.link(im, "target").filter(|&m| dom.is(m, "material")),
                inputs: dom
                    .kids(im, "bind_vertex_input")
                    .filter_map(|b| {
                        let semantic = dom.attr(b, "semantic")?.to_owned();
                        Some((semantic, dom.attr_num::<u32>(b, "input_set").unwrap_or(0)))
                    })
                    .collect(),
            })
            .collect()
    }

    /// The geometry under a geometry or controller, and the skin and morph on the way.
    fn controller(&self, el: usize) -> (Option<usize>, Option<usize>, Option<usize>) {
        let dom = self.dom;
        let (mut skin, mut morph) = (None, None);
        let mut at = el;
        for _ in 0..8 {
            if dom.is(at, "geometry") {
                return (Some(at), skin, morph);
            }
            if !dom.is(at, "controller") {
                break;
            }
            let next = if let Some(s) = dom.child(at, "skin") {
                skin.get_or_insert(s);
                dom.link(s, "source")
            } else if let Some(m) = dom.child(at, "morph") {
                morph.get_or_insert(m);
                dom.link(m, "source")
            } else {
                None
            };
            let Some(next) = next else { break };
            at = next;
        }
        (None, skin, morph)
    }

    /// Draws geometry or controller `target` on node `n` (of element `from`), bound to `binds`.
    fn attach(
        &mut self,
        from: usize,
        target: usize,
        binds: &[Binding],
        n: usize,
        roots: &[usize],
        children: &mut Vec<usize>,
    ) -> Result<()> {
        let dom = self.dom;
        let (geometry, skin, morph) = self.controller(target);
        let Some(geometry) = geometry else {
            return Ok(());
        };
        let key: MeshKey = (
            target,
            binds
                .iter()
                .map(|b| (b.symbol.clone(), b.material, b.inputs.clone()))
                .collect(),
        );
        let mesh = if let Some(&m) = self.mesh_of.get(&key) {
            m
        } else {
            let m = if skin.is_none() && morph.is_none() && self.drawn_as.contains_key(&geometry) {
                Some(self.rebind(geometry, binds))
            } else {
                self.build_mesh(geometry, binds, skin, morph)?
            };
            self.mesh_of.insert(key, m);
            m
        };
        let Some(mesh) = mesh else {
            return Ok(());
        };
        self.drawn += self.mesh_triangles[mesh];
        ensure!(
            self.drawn <= self.limit,
            LimitExceeded,
            "too many triangles (over {})",
            self.limit
        );
        if self.keep.skins
            && let Some(skin) = skin
        {
            let name = self.nodes[n]
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| dom.attr(target, "name").map(str::to_owned));
            self.pending.push(PendingSkin {
                mesh,
                skin,
                roots: roots.to_vec(),
                node: n,
                name,
            });
            return Ok(());
        }
        // A geometry in another unit or up axis than the node, as another document may be, is carried into its frame.
        let holder = match self.correction(from, geometry) {
            Some(c) => {
                let h = self.push_node(json!({}), Some(n));
                if let Some(name) = self.nodes[n].get("name").cloned() {
                    self.nodes[h]["name"] = name;
                }
                set_transform(&mut self.nodes[h], &c);
                children.push(h);
                h
            }
            None => self.holder(n, "mesh", children),
        };
        self.nodes[holder]["mesh"] = json!(mesh);
        Ok(())
    }

    /// The glTF mesh of `geometry`'s primitives; `None` when it draws nothing.
    fn build_mesh(
        &mut self,
        geometry: usize,
        binds: &[Binding],
        skin: Option<usize>,
        morph: Option<usize>,
    ) -> Result<Option<usize>> {
        let dom = self.dom;
        let Some(mesh_el) = dom
            .child(geometry, "mesh")
            .or_else(|| dom.child(geometry, "convex_mesh"))
        else {
            return Ok(None);
        };
        let budget = self.limit.saturating_sub(self.drawn);
        let Ok(pieces) = geometry::mesh(dom, &mut self.sources, mesh_el, budget) else {
            bail!(LimitExceeded, "too many triangles (over {})", self.limit);
        };
        let mut prims = Vec::new();
        let mut points = Vec::new();
        let mut normals = Vec::new();
        let mut symbols = Vec::new();
        let mut triangles = 0;
        let morphed = morph.is_some() && self.keep.morph_targets;
        for p in pieces {
            let material = if self.keep.materials {
                p.symbol
                    .as_deref()
                    .and_then(|s| self.material_for(geometry, s, binds, &p.sets))
            } else {
                None
            };
            // A piece draws triangles or lines, never both, so it is one glTF primitive.
            let base = if morphed && p.prim.normals.len() == p.prim.positions.len() {
                p.prim.normals.clone()
            } else {
                Vec::new()
            };
            let Some(mut prim) = self.b.primitives(p.prim).pop() else {
                continue;
            };
            symbols.push((p.symbol, p.sets));
            normals.push(base);
            if let Some(m) = material {
                prim["material"] = json!(m);
            }
            prims.push(prim);
            points.push(p.points);
            triangles += p.triangles;
        }
        if prims.is_empty() {
            return Ok(None);
        }
        let mut mesh = json!({ "primitives": prims });
        if let Some(name) = dom.attr(geometry, "name").or_else(|| dom.attr(geometry, "id")) {
            mesh["name"] = json!(name);
        }
        if let Some(e) = self.extras_of(&[geometry]) {
            mesh["extras"] = e;
        }
        self.meshes.push(mesh);
        self.mesh_triangles.push(triangles);
        let m = self.meshes.len() - 1;
        if skin.is_none() && morph.is_none() {
            self.drawn_as.insert(geometry, (m, symbols));
        }
        let morph = morph.filter(|_| self.keep.morph_targets);
        if (skin.is_some() && self.keep.skins) || morph.is_some() {
            self.points_of.insert(m, points);
        }
        if let Some(morph) = morph {
            self.morphs.push((m, morph));
            self.normals_of.insert(m, normals);
        }
        Ok(Some(m))
    }

    /// Static `geometry`, drawn already, as a mesh on the same accessors with the materials `binds` give.
    fn rebind(&mut self, geometry: usize, binds: &[Binding]) -> usize {
        let (first, symbols) = self.drawn_as[&geometry].clone();
        let mut mesh = self.meshes[first].clone();
        if let Some(prims) = mesh.get_mut("primitives").and_then(Value::as_array_mut) {
            for (prim, (symbol, sets)) in prims.iter_mut().zip(&symbols) {
                let material = symbol
                    .as_deref()
                    .filter(|_| self.keep.materials)
                    .and_then(|s| self.material_for(geometry, s, binds, sets));
                match (material, prim.as_object_mut()) {
                    (Some(m), Some(o)) => drop(o.insert("material".into(), json!(m))),
                    (None, Some(o)) => drop(o.remove("material")),
                    _ => {}
                }
            }
        }
        self.meshes.push(mesh);
        self.mesh_triangles.push(self.mesh_triangles[first]);
        self.meshes.len() - 1
    }

    /// The material bound to `symbol`, its textures reading the primitive's coordinate sets (`sets`, in the
    /// order of its `TEXCOORD_n`) that the binding names. An unbound symbol is taken for the id of a material in the
    /// document of `at`.
    fn material_for(&mut self, at: usize, symbol: &str, binds: &[Binding], sets: &[u32]) -> Option<usize> {
        let dom = self.dom;
        let bind = binds.iter().find(|b| b.symbol == symbol);
        let mat = bind
            .and_then(|b| b.material)
            .or_else(|| dom.by_id_at(at, symbol).filter(|&m| dom.is(m, "material")))?;
        let uv: Vec<(String, u32)> = bind
            .map(|b| {
                b.inputs
                    .iter()
                    .map(|(semantic, set)| {
                        let k = sets.iter().position(|s| s == set).map_or(0, u32_of);
                        (semantic.clone(), k)
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Two-sidedness a geometry's extras give its faces.
        let two = dom.is(at, "geometry")
            && dom.kids(at, "extra").any(|x| {
                dom.descendant(x, "double_sided", 4)
                    .is_some_and(|d| matches!(dom.string(d), "1" | "true"))
            });
        let key = (mat, uv, two);
        if let Some(&m) = self.material_of.get(&key) {
            return Some(m);
        }
        let mut full = self.material(mat, &key.1);
        full.double_sided |= two;
        self.materials.push(full);
        let m = self.materials.len() - 1;
        self.material_of.insert(key, m);
        Some(m)
    }

    /// Skins for the skinned meshes, now that every node is made.
    fn skin_meshes(&mut self) {
        for p in std::mem::take(&mut self.pending) {
            let Some((joints, count)) = self.joints(p.skin, &p.roots) else {
                // The joints aren't in the scene: the mesh goes unskinned where it is.
                let mut children = Vec::new();
                let holder = self.holder(p.node, "mesh", &mut children);
                self.nodes[holder]["mesh"] = json!(p.mesh);
                self.adopt(p.node, &children);
                continue;
            };
            if !self.weighted.contains_key(&p.mesh) {
                self.weigh(p.mesh, p.skin, count);
                self.weighted.insert(p.mesh, p.skin);
            }
            let ibms = self.inverse_binds(p.skin, count);
            let acc = self.b.floats(ibms, "MAT4", false);
            let mut skin = json!({ "joints": joints, "inverseBindMatrices": acc });
            if let Some(s) = self.common_ancestor(&joints) {
                skin["skeleton"] = json!(s);
            }
            self.skins.push(skin);
            let mut node = json!({ "mesh": p.mesh, "skin": self.skins.len() - 1 });
            if let Some(name) = p.name {
                node["name"] = json!(name);
            }
            let n = self.push_node(node, None);
            self.skinned_roots.push(n);
        }
    }

    /// The `<source>` an input of `el` with `semantic` reads.
    fn input(&self, el: usize, semantic: &str) -> Option<usize> {
        let dom = self.dom;
        dom.kids(el, "input")
            .find(|&i| dom.attr(i, "semantic") == Some(semantic))
            .and_then(|i| dom.link(i, "source"))
    }

    /// The nodes of a skin's joints, each a different node (a joint that isn't found, or is found twice, gets a
    /// node of its own under the first that is), and how many there are; `None` when none is found.
    fn joints(&mut self, skin: usize, roots: &[usize]) -> Option<(Vec<usize>, usize)> {
        let dom = self.dom;
        let joints_el = dom.child(skin, "joints")?;
        let source = self.input(joints_el, "JOINT")?;
        let idref = dom.children(source).any(|c| dom.is(c, "IDREF_array"));
        let names = geometry::names(dom, source);
        if names.is_empty() {
            return None;
        }
        // Nodes by sid and by name, the first in document order when several share one.
        let mut by_sid: HashMap<&str, usize> = HashMap::new();
        let mut by_name: HashMap<&str, usize> = HashMap::new();
        let mut made: Vec<usize> = self.instances.keys().copied().collect();
        made.sort_unstable();
        for &el in &made {
            if let Some(s) = dom.attr(el, "sid") {
                by_sid.entry(s).or_insert(el);
            }
            if let Some(s) = dom.attr(el, "name") {
                by_name.entry(s).or_insert(el);
            }
        }
        let found: Vec<Option<usize>> = names
            .iter()
            .map(|name| {
                let el = if idref {
                    dom.by_id_at(skin, name)
                } else {
                    roots
                        .iter()
                        .find_map(|&r| {
                            if dom.attr(r, "sid") == Some(name.as_str()) {
                                return Some(r);
                            }
                            dom.by_sid(r, name).filter(|&e| dom.is(e, "node"))
                        })
                        .or_else(|| by_sid.get(name.as_str()).copied())
                        .or_else(|| dom.by_id_at(skin, name).filter(|&e| dom.is(e, "node")))
                        .or_else(|| by_name.get(name.as_str()).copied())
                };
                el.and_then(|e| self.instances.get(&e)).and_then(|v| v.first().copied())
            })
            .collect();
        let first = found.iter().flatten().copied().next()?;
        let mut used = std::collections::HashSet::new();
        let mut joints = Vec::with_capacity(found.len());
        for (k, f) in found.into_iter().enumerate() {
            if let Some(j) = f.filter(|j| used.insert(*j)) {
                joints.push(j);
            } else {
                let n = self.push_node(json!({ "name": names[k] }), Some(first));
                self.adopt(first, &[n]);
                used.insert(n);
                joints.push(n);
            }
        }
        let count = joints.len();
        Some((joints, count))
    }

    /// The deepest node every one of `nodes` is under (or is).
    fn common_ancestor(&self, nodes: &[usize]) -> Option<usize> {
        let chain = |mut n: usize| {
            let mut c = vec![n];
            while let Some(p) = self.parents[n] {
                c.push(p);
                n = p;
                if c.len() > MAX_NODES {
                    break;
                }
            }
            c.reverse();
            c
        };
        let mut common = chain(*nodes.first()?);
        for &n in &nodes[1..] {
            let other = chain(n);
            let same = common.iter().zip(&other).take_while(|(a, b)| a == b).count();
            common.truncate(same);
            if common.is_empty() {
                return None;
            }
        }
        common.last().copied()
    }

    /// Writes the joints and weights of a skinned mesh's primitives.
    fn weigh(&mut self, mesh: usize, skin: usize, count: usize) {
        let dom = self.dom;
        let influences = self.influences(skin);
        let Some(points) = self.points_of.get(&mesh) else {
            return;
        };
        let last = u32_of(count.saturating_sub(1));
        // Four influences to a set of joints and weights, as many sets as the most-influenced point needs.
        let width = influences.most.div_ceil(4).max(1) * 4;
        let mut written = Vec::new();
        for pts in points {
            let n = pts.len();
            let mut js = Vec::with_capacity(n * width);
            let mut ws = Vec::with_capacity(n * width);
            let (mut j, mut w) = (vec![0u32; width], vec![0f32; width]);
            for &p in pts {
                j.fill(0);
                w.fill(0.0);
                for (k, &(jj, ww)) in influences.of(p as usize).iter().enumerate() {
                    j[k] = jj;
                    w[k] = ww;
                }
                settle(&mut j, &mut w, last);
                js.extend_from_slice(&j);
                ws.extend_from_slice(&w);
            }
            let mut sets = Vec::new();
            for set in 0..width / 4 {
                let set_js = quad(&js, width, set);
                let set_ws = quad(&ws, width, set);
                let j = if count <= 256 {
                    let b: Vec<u8> = set_js.iter().map(|&j| u8::try_from(j).unwrap_or(0)).collect();
                    self.b.accessor(Part::bytes(b), 5121, n, "VEC4", Some(34962))
                } else {
                    let b = set_js.iter().map(|&j| u16::try_from(j).unwrap_or(0)).collect();
                    self.b.accessor(Part::u16(b), 5123, n, "VEC4", Some(34962))
                };
                let w = self.b.accessor(Part::f32(set_ws), 5126, n, "VEC4", Some(34962));
                sets.push((j, w));
            }
            written.push(sets);
        }
        let _ = dom;
        if let Some(prims) = self.meshes[mesh]["primitives"].as_array_mut() {
            for (prim, sets) in prims.iter_mut().zip(written) {
                for (k, (j, w)) in sets.into_iter().enumerate() {
                    prim["attributes"][format!("JOINTS_{k}")] = json!(j);
                    prim["attributes"][format!("WEIGHTS_{k}")] = json!(w);
                }
            }
        }
    }

    /// Each point's strongest joints (up to sixteen) and their weights, from a skin's `<vertex_weights>`.
    fn influences(&self, skin: usize) -> Influences {
        let dom = self.dom;
        let mut out = Influences {
            starts: vec![0],
            pairs: Vec::new(),
            most: 0,
        };
        let Some(vw) = dom.child(skin, "vertex_weights") else {
            return out;
        };
        let mut joint_offset = 0;
        let mut weight_offset = 1;
        let mut weights = Vec::new();
        let mut stride = 1;
        for i in dom.kids(vw, "input") {
            let offset = dom.attr_num::<usize>(i, "offset").unwrap_or(0).min(64);
            stride = stride.max(offset + 1);
            match dom.attr(i, "semantic") {
                Some("JOINT") => joint_offset = offset,
                Some("WEIGHT") => {
                    weight_offset = offset;
                    if let Some(s) = dom.link(i, "source") {
                        weights = geometry::read(dom, s, 1, xml::floats32).data;
                    }
                }
                _ => {}
            }
        }
        let counts = dom
            .child(vw, "vcount")
            .map(|c| xml::indices(dom.text(c)))
            .unwrap_or_default();
        let v = dom.child(vw, "v").map(|c| xml::ints(dom.text(c))).unwrap_or_default();
        // Each point takes one number of `v` at least, which bounds how many points there can be.
        out.starts.reserve(counts.len().min(v.len()));
        let mut at = 0usize;
        let mut all: Vec<(u32, f32)> = Vec::new();
        for &n in &counts {
            all.clear();
            for _ in 0..n {
                let (Some(&j), Some(&w)) = (v.get(at + joint_offset), v.get(at + weight_offset)) else {
                    break;
                };
                at += stride;
                // A joint of -1 is the bind shape itself, which moves with no joint.
                let (Ok(j), Ok(wi)) = (u32::try_from(j), usize::try_from(w)) else {
                    continue;
                };
                let w = weights.get(wi).copied().unwrap_or(0.0);
                if w > 0.0 && w.is_finite() {
                    all.push((j, w));
                }
            }
            if at > v.len() {
                at = v.len();
            }
            all.sort_by(|a, b| b.1.total_cmp(&a.1));
            all.truncate(MAX_INFLUENCES);
            out.most = out.most.max(all.len());
            out.pairs.extend_from_slice(&all);
            out.starts.push(u32_of(out.pairs.len()));
        }
        out
    }

    /// A skin's inverse bind matrices, each with the bind shape matrix folded in, column-major.
    fn inverse_binds(&self, skin: usize, count: usize) -> Vec<f32> {
        let dom = self.dom;
        let bind_shape = dom
            .child(skin, "bind_shape_matrix")
            .map_or(IDENTITY, |b| matrix_of(&floats64(dom.text(b))));
        let ibm = dom
            .child(skin, "joints")
            .and_then(|j| self.input(j, "INV_BIND_MATRIX"))
            .map(|s| geometry::read(dom, s, 16, floats64))
            .filter(|a| a.width == 16);
        let mut out = Vec::with_capacity(count * 16);
        for k in 0..count {
            let m = ibm.as_ref().and_then(|a| a.get(u32_of(k))).map_or(IDENTITY, matrix_of);
            let mut m = mul(&m, &bind_shape);
            m[3] = [0.0, 0.0, 0.0, 1.0];
            out.extend(to_column_major(&m).map(f32_of));
        }
        out
    }

    /// Morph targets for the morphed meshes.
    fn morph_meshes(&mut self) {
        let dom = self.dom;
        for (mesh, morph) in self.morphs.clone() {
            let base = dom
                .link(morph, "source")
                .and_then(|g| self.controller(g).0)
                .and_then(|g| dom.child(g, "mesh").or_else(|| dom.child(g, "convex_mesh")))
                .and_then(|m| geometry::positions(dom, &mut self.sources, m));
            let relative = dom.attr(morph, "method") == Some("RELATIVE");
            let Some(targets_el) = dom.child(morph, "targets") else {
                continue;
            };
            let Some(target_src) = self.input(targets_el, "MORPH_TARGET") else {
                continue;
            };
            let ids = geometry::names(dom, target_src);
            if ids.is_empty() {
                continue;
            }
            let weights = self
                .input(targets_el, "MORPH_WEIGHT")
                .map(|s| geometry::read(dom, s, 1, xml::floats32).data)
                .unwrap_or_default();
            let Some(points) = self.points_of.get(&mesh).cloned() else {
                continue;
            };
            let base_normals = self.normals_of.remove(&mesh).unwrap_or_default();
            let mut names = Vec::new();
            let mut targets = Vec::new();
            let mut target_normals = Vec::new();
            for id in &ids {
                let g = dom.by_id_at(morph, id).filter(|&g| dom.is(g, "geometry"));
                names.push(g.and_then(|g| dom.attr(g, "name")).unwrap_or(id).to_owned());
                let target_mesh = g.and_then(|g| dom.child(g, "mesh").or_else(|| dom.child(g, "convex_mesh")));
                targets.push(target_mesh.and_then(|m| geometry::positions(dom, &mut self.sources, m)));
                // A relative target's normals would be offsets, which a normal can't be read as.
                let normals = target_mesh
                    .filter(|_| !relative && base_normals.iter().any(|n| !n.is_empty()))
                    .and_then(|m| self.target_normals(m, mesh, &points, &base_normals));
                target_normals.push(normals);
            }
            let at = |acc: Option<&geometry::Acc<f32>>, p: u32| -> [f32; 3] {
                acc.and_then(|a| a.get(p)).map_or([0.0; 3], |v| {
                    [v[0], v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)]
                })
            };
            let mut per_prim = Vec::new();
            for pts in &points {
                let mut list = Vec::new();
                for t in &targets {
                    let deltas: Vec<[f32; 3]> = pts
                        .iter()
                        .map(|&p| {
                            let Some(t) = t.as_deref() else { return [0.0; 3] };
                            let v = at(Some(t), p);
                            if relative || t.get(p).is_none() {
                                if t.get(p).is_none() { [0.0; 3] } else { v }
                            } else {
                                let b = at(base.as_deref(), p);
                                [v[0] - b[0], v[1] - b[1], v[2] - b[2]]
                            }
                        })
                        .collect();
                    list.push(deltas);
                }
                per_prim.push(list);
            }
            let mut written = Vec::new();
            for (k, list) in per_prim.into_iter().enumerate() {
                let mut accs = Vec::new();
                for (d, normals) in list.into_iter().zip(&mut target_normals) {
                    let mut target = json!({ "POSITION": self.b.positions(d) });
                    // A target that leaves a primitive's normals as they are has no normals of its own.
                    let normals = normals
                        .as_mut()
                        .and_then(|n| n.get_mut(k))
                        .map(std::mem::take)
                        .unwrap_or_default();
                    if normals.iter().any(|n| n.iter().any(|&c| c != 0.0)) {
                        target["NORMAL"] = json!(self.b.floats(normals.concat(), "VEC3", false));
                    }
                    accs.push(target);
                }
                written.push(accs);
            }
            let w: Vec<f32> = (0..ids.len())
                .map(|k| weights.get(k).copied().filter(|w| w.is_finite()).unwrap_or(0.0))
                .collect();
            let m = &mut self.meshes[mesh];
            if let Some(prims) = m["primitives"].as_array_mut() {
                for (prim, t) in prims.iter_mut().zip(written) {
                    prim["targets"] = json!(t);
                }
            }
            m["weights"] = json!(w);
            m["extras"]["targetNames"] = json!(names);
        }
    }

    /// The changes a morph target's `<mesh>` makes to the normals of each of `mesh`'s primitives, when it draws
    /// what the base does over the same vertices (as targets do); `None` when it doesn't, or has no normals.
    fn target_normals(
        &mut self,
        target: usize,
        mesh: usize,
        points: &[Vec<u32>],
        base: &[Vec<[f32; 3]>],
    ) -> Option<Vec<Vec<[f32; 3]>>> {
        let budget = self.mesh_triangles.get(mesh).copied().unwrap_or(0);
        let pieces = geometry::mesh(self.dom, &mut self.sources, target, budget).ok()?;
        if pieces.len() != points.len() || pieces.iter().zip(points).any(|(p, pts)| p.points != *pts) {
            return None;
        }
        let mut out = Vec::with_capacity(pieces.len());
        for (p, b) in pieces.iter().zip(base) {
            let n = &p.prim.normals;
            if b.is_empty() || n.len() != b.len() {
                out.push(Vec::new());
                continue;
            }
            out.push(
                n.iter()
                    .zip(b)
                    .map(|(t, b)| [t[0] - b[0], t[1] - b[1], t[2] - b[2]])
                    .collect(),
            );
        }
        Some(out)
    }

    /// A camera's index, made once for each `<camera>`.
    fn camera(&mut self, el: usize) -> Option<usize> {
        if let Some(&c) = self.camera_of.get(&el) {
            return c;
        }
        let c = self.make_camera(el).map(|cam| {
            self.cameras.push(cam);
            self.cameras.len() - 1
        });
        self.camera_of.insert(el, c);
        c
    }

    fn make_camera(&self, el: usize) -> Option<Value> {
        let dom = self.dom;
        let tc = dom.path(el, &["optics", "technique_common"])?;
        let num = |e: usize, name: &str| {
            dom.child(e, name)
                .map(|c| xml::float(dom.text(c)))
                .filter(|v| v.is_finite())
        };
        let m = self.meters;
        let mut cam = if let Some(p) = dom.child(tc, "perspective") {
            let (xfov, yfov, aspect) = (
                num(p, "xfov").filter(|v| *v > 0.0 && *v < 180.0),
                num(p, "yfov").filter(|v| *v > 0.0 && *v < 180.0),
                num(p, "aspect_ratio").filter(|v| *v > 0.0),
            );
            let half = |deg: f64| (deg.to_radians() / 2.0).tan();
            let (yfov, aspect) = match (xfov, yfov, aspect) {
                (_, Some(y), Some(a)) => (y.to_radians(), Some(a)),
                (Some(x), Some(y), None) => (y.to_radians(), Some(half(x) / half(y))),
                (Some(x), None, Some(a)) => (2.0 * (half(x) / a).atan(), Some(a)),
                (Some(x), None, None) => (x.to_radians(), None),
                (None, Some(y), None) => (y.to_radians(), None),
                (None, None, a) => (std::f64::consts::FRAC_PI_4, a),
            };
            let znear = num(p, "znear").filter(|v| *v > 0.0).unwrap_or(0.1) * m;
            let mut persp = json!({ "yfov": yfov, "znear": znear });
            if let Some(a) = aspect.filter(|a| a.is_finite() && *a > 0.0) {
                persp["aspectRatio"] = json!(a);
            }
            if let Some(f) = num(p, "zfar").map(|f| f * m).filter(|f| *f > znear) {
                persp["zfar"] = json!(f);
            }
            json!({ "type": "perspective", "perspective": persp })
        } else {
            let o = dom.child(tc, "orthographic")?;
            let aspect = num(o, "aspect_ratio").filter(|v| *v > 0.0);
            let (xmag, ymag) = (
                num(o, "xmag").filter(|v| *v != 0.0),
                num(o, "ymag").filter(|v| *v != 0.0),
            );
            let (xmag, ymag) = match (xmag, ymag, aspect) {
                (Some(x), Some(y), _) => (x, y),
                (Some(x), None, a) => (x, x / a.unwrap_or(1.0)),
                (None, Some(y), a) => (y * a.unwrap_or(1.0), y),
                (None, None, a) => (a.unwrap_or(1.0), 1.0),
            };
            let znear = num(o, "znear").filter(|v| *v >= 0.0).unwrap_or(0.0) * m;
            let zfar = num(o, "zfar")
                .map(|f| f * m)
                .filter(|f| *f > znear)
                .unwrap_or(znear + 1000.0 * m);
            json!({ "type": "orthographic", "orthographic": {
                "xmag": xmag * m, "ymag": ymag * m, "znear": znear, "zfar": zfar } })
        };
        if let Some(name) = dom.attr(el, "name").or_else(|| dom.attr(el, "id")) {
            cam["name"] = json!(name);
        }
        if let Some(e) = self.extras_of(&[el, tc]) {
            cam["extras"] = e;
        }
        Some(cam)
    }

    /// Puts light `el` on node `n` (or a child of it): an ambient light as the node's extras.
    fn light(&mut self, el: usize, n: usize, children: &mut Vec<usize>) {
        let dom = self.dom;
        let Some(tc) = dom.child(el, "technique_common") else {
            return;
        };
        let Some(kind) = dom.children(tc).next() else {
            return;
        };
        let color = dom
            .child(kind, "color")
            .map(|c| xml::floats32(dom.text(c)))
            .map_or([1.0; 3], |c| {
                [0, 1, 2].map(|k| c.get(k).copied().filter(|v| v.is_finite()).unwrap_or(0.0).max(0.0))
            });
        if dom.is(kind, "ambient") {
            let holder = self.holder(n, "extras", children);
            self.nodes[holder]["extras"] = json!({ "ambientLight": color });
            return;
        }
        let index = if let Some(&l) = self.light_of.get(&el) {
            l
        } else {
            let l = self.make_light(el, kind, color).map(|light| {
                self.lights.push(light);
                self.lights.len() - 1
            });
            self.light_of.insert(el, l);
            l
        };
        let Some(index) = index else {
            return;
        };
        let holder = self.holder(n, "extensions", children);
        self.nodes[holder]["extensions"] = json!({ "KHR_lights_punctual": { "light": index } });
    }

    fn make_light(&self, el: usize, kind: usize, color: [f32; 3]) -> Option<Value> {
        let dom = self.dom;
        let ty = match dom.name(kind) {
            b"directional" => "directional",
            b"point" => "point",
            b"spot" => "spot",
            _ => return None,
        };
        let peak = color.iter().copied().fold(0.0f32, f32::max);
        let (color, intensity) = if peak > 1.0 {
            (color.map(|c| c / peak), peak)
        } else {
            (color, 1.0)
        };
        let mut light = json!({ "type": ty, "color": color, "intensity": intensity });
        if ty == "spot" {
            let falloff = dom
                .child(kind, "falloff_angle")
                .map(|c| xml::float(dom.text(c)))
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(180.0);
            let outer = (falloff.to_radians() / 2.0).clamp(1e-3, std::f64::consts::FRAC_PI_2);
            light["spot"] = json!({ "innerConeAngle": 0.0, "outerConeAngle": outer });
        }
        if let Some(name) = dom.attr(el, "name").or_else(|| dom.attr(el, "id")) {
            light["name"] = json!(name);
        }
        if self.keep.extras {
            // How the light fades, which glTF fixes at the inverse square: kept as the source gives it.
            let mut fade = Map::new();
            for name in [
                "constant_attenuation",
                "linear_attenuation",
                "quadratic_attenuation",
                "falloff_exponent",
            ] {
                if let Some(v) = dom
                    .child(kind, name)
                    .map(|c| xml::float(dom.text(c)))
                    .filter(|v| v.is_finite())
                {
                    fade.insert(name.to_owned(), json!(v));
                }
            }
            if !fade.is_empty() {
                light["extras"] = Value::Object(fade);
            }
            if let Some(Value::Object(e)) = self.extras_of(&[el, dom.parent(kind).unwrap_or(el)]) {
                for (k, v) in e {
                    light["extras"][k] = v;
                }
            }
        }
        Some(light)
    }
}

impl Reader<'_> {
    /// The `<extra>` techniques of the elements `els` as JSON, under `extra` and by profile, when extras are kept:
    /// an element that holds text as its text, one that holds elements as an object of them, and a repeated one as
    /// a list. A later element's technique replaces an earlier one's of the same profile.
    fn extras_of(&self, els: &[usize]) -> Option<Value> {
        if !self.keep.extras {
            return None;
        }
        let dom = self.dom;
        let mut out = Map::new();
        let mut budget = MAX_EXTRA_VALUES;
        for &el in els {
            for extra in dom.kids(el, "extra") {
                for t in dom.children(extra) {
                    let key = dom
                        .attr(t, "profile")
                        .map_or_else(|| String::from_utf8_lossy(dom.name(t)).into_owned(), str::to_owned);
                    let v = extra_json(dom, t, 0, &mut budget);
                    out.insert(key, v);
                }
            }
        }
        (!out.is_empty()).then(|| json!({ "extra": out }))
    }
}

/// Element `e` of an `<extra>` as JSON, taking from `budget` for each element below it.
fn extra_json(dom: &Dom<'_>, e: usize, depth: usize, budget: &mut usize) -> Value {
    if depth >= MAX_DEPTH || dom.children(e).next().is_none() {
        return json!(dom.string(e));
    }
    let mut out = Map::new();
    for c in dom.children(e) {
        if *budget == 0 {
            break;
        }
        *budget -= 1;
        let v = extra_json(dom, c, depth + 1, budget);
        let key = String::from_utf8_lossy(dom.name(c)).into_owned();
        match out.get_mut(&key) {
            Some(Value::Array(list)) => list.push(v),
            Some(old) => *old = json!([old.take(), v]),
            None => {
                out.insert(key, v);
            }
        }
    }
    Value::Object(out)
}

/// A vertex's joints and weights as glTF wants them: joints in range, weights summing to 1, a vertex no joint moves
/// bound to joint 0, and a slot with no weight naming joint 0.
fn settle(j: &mut [u32], w: &mut [f32], last: u32) {
    for x in w.iter_mut() {
        if !(*x > 0.0 && x.is_finite()) {
            *x = 0.0;
        }
    }
    for x in j.iter_mut() {
        *x = (*x).min(last);
    }
    let sum: f32 = w.iter().sum();
    if sum > 0.0 {
        for x in w.iter_mut() {
            *x /= sum;
        }
        let top = (0..w.len()).fold(0, |a, k| if w[k] > w[a] { k } else { a });
        w[top] += 1.0 - w.iter().sum::<f32>();
    } else {
        j.fill(0);
        w.fill(0.0);
        w[0] = 1.0;
    }
    for (x, &v) in j.iter_mut().zip(w.iter()) {
        if v == 0.0 {
            *x = 0;
        }
    }
}

/// The `set`th four of each `width` numbers.
fn quad<T: Copy>(v: &[T], width: usize, set: usize) -> Vec<T> {
    v.chunks_exact(width)
        .flat_map(|c| c[set * 4..set * 4 + 4].iter().copied())
        .collect()
}

/// Each point's influences, strongest first.
struct Influences {
    /// Where each point's influences start in `pairs`, and where the last one's end.
    starts: Vec<u32>,
    pairs: Vec<(u32, f32)>,
    /// The most influences a point has.
    most: usize,
}

impl Influences {
    /// The influences of point `p`; none for a point the skin doesn't list.
    fn of(&self, p: usize) -> &[(u32, f32)] {
        match (self.starts.get(p), self.starts.get(p + 1)) {
            (Some(&a), Some(&b)) => &self.pairs[a as usize..b as usize],
            _ => &[],
        }
    }
}

pub(super) fn floats64(text: &[u8]) -> Vec<f64> {
    let mut v = Vec::new();
    xml::floats(text, &mut v);
    v
}

/// A row-major 4x4 matrix from 16 numbers (missing ones from the identity).
pub(super) fn matrix_of(v: &[f64]) -> M4 {
    let mut m = IDENTITY;
    for (k, &x) in v.iter().take(16).enumerate() {
        if x.is_finite() {
            m[k / 4][k % 4] = x;
        }
    }
    m
}

/// The kinds of transform element.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Transform {
    Matrix,
    Translate,
    Rotate,
    Scale,
    Lookat,
    Skew,
}

impl Transform {
    pub(super) fn of(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"matrix" => Transform::Matrix,
            b"translate" => Transform::Translate,
            b"rotate" => Transform::Rotate,
            b"scale" => Transform::Scale,
            b"lookat" => Transform::Lookat,
            b"skew" => Transform::Skew,
            _ => return None,
        })
    }

    /// How many numbers it takes.
    pub(super) fn len(self) -> usize {
        match self {
            Transform::Matrix => 16,
            Transform::Translate | Transform::Scale => 3,
            Transform::Rotate => 4,
            Transform::Lookat => 9,
            Transform::Skew => 7,
        }
    }

    /// Its matrix for values `v` (missing ones taken as the identity's).
    pub(super) fn matrix(self, v: &[f64]) -> M4 {
        let at = |k: usize, d: f64| v.get(k).copied().filter(|x| x.is_finite()).unwrap_or(d);
        match self {
            Transform::Matrix => matrix_of(v),
            Transform::Translate => translate([at(0, 0.0), at(1, 0.0), at(2, 0.0)]),
            Transform::Scale => scale([at(0, 1.0), at(1, 1.0), at(2, 1.0)]),
            Transform::Rotate => rotation([at(0, 0.0), at(1, 0.0), at(2, 1.0)], at(3, 0.0).to_radians()),
            Transform::Lookat => {
                let eye = [at(0, 0.0), at(1, 0.0), at(2, 0.0)];
                let target = [at(3, 0.0), at(4, 0.0), at(5, -1.0)];
                let up = [at(6, 0.0), at(7, 1.0), at(8, 0.0)];
                let z = vec3::unit(sub(eye, target)).unwrap_or([0.0, 0.0, 1.0]);
                let x = vec3::unit(cross(up, z))
                    .or_else(|| vec3::unit(cross([0.0, 0.0, 1.0], z)))
                    .unwrap_or([1.0, 0.0, 0.0]);
                let y = cross(z, x);
                [
                    [x[0], y[0], z[0], eye[0]],
                    [x[1], y[1], z[1], eye[1]],
                    [x[2], y[2], z[2], eye[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
            Transform::Skew => {
                // Shears along the rotation axis's part across the translation axis, by the angle's tangent.
                let angle = at(0, 0.0).to_radians().clamp(-1.5, 1.5);
                let a = [at(1, 0.0), at(2, 0.0), at(3, 0.0)];
                let Some(b) = vec3::unit([at(4, 0.0), at(5, 0.0), at(6, 0.0)]) else {
                    return IDENTITY;
                };
                let d = dot(a, b);
                let Some(a) = vec3::unit([a[0] - d * b[0], a[1] - d * b[1], a[2] - d * b[2]]) else {
                    return IDENTITY;
                };
                let t = angle.tan();
                let mut m = IDENTITY;
                for (i, row) in m.iter_mut().enumerate().take(3) {
                    for (j, x) in row.iter_mut().enumerate().take(3) {
                        *x += t * a[i] * b[j];
                    }
                }
                m
            }
        }
    }
}

/// A rotation of `angle` radians about `axis`.
fn rotation(axis: [f64; 3], angle: f64) -> M4 {
    let Some([x, y, z]) = vec3::unit(axis) else {
        return IDENTITY;
    };
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
