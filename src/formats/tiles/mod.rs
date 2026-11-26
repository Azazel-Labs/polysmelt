//! OGC 3D Tiles: the tile formats of 3D Tiles 1.0 (`.b3dm`, `.i3dm`, `.pnts`, `.cmpt`) and a
//! `tileset.json` of them or of glTF.
//!
//! A tile is read by its first bytes, not its extension: the four magics, `glTF` (3D Tiles 1.1
//! content), or JSON, which is a tileset when it has a `root` or a `geometricError` and a glTF
//! otherwise.
//!
//! - **b3dm** (Batched 3D Model): the embedded binary glTF (2.0, or 1.0 through the glTF
//!   reader's upgrade) is the document. The header may be either legacy layout. A `BATCHID`
//!   attribute takes glTF 2.0's name, `_BATCHID`.
//! - **i3dm** (Instanced 3D Model): the glTF, embedded or named by a URI the [`Assets`] read,
//!   with its instances as `EXT_mesh_gpu_instancing` on each node that draws a mesh (see
//!   [`i3dm`]).
//! - **pnts** (Point Cloud): one node drawing `POINTS` (see [`pnts`]).
//! - **cmpt** (Composite): its tiles under one node, composites inside composites too.
//! - **`tileset.json`**: the tiles at the finest level of detail, each under a node with its
//!   tile's transform (see [`tileset`]).
//!
//! # Coordinates
//!
//! 3D Tiles is right-handed with Z up, in meters, and a tile's glTF is Y up: a viewer turns it
//! with the Y-up-to-Z-up rotation before placing it. A document is glTF, Y up, so everything in
//! tile space (an `RTC_CENTER`, instance and point positions, normals, tile transforms) is
//! turned the other way, `(x, y, z)` to `(x, z, -y)`, and the glTF stays as it is. A tileset's
//! `asset.gltfUpAxis` (`Z` or `X`, from before 1.0) turns the glTF instead. The source info
//! says +Z up, converted.
//!
//! Positions on the globe are millions of meters from the origin, more than single precision
//! holds with any detail. So an `RTC_CENTER` (and a glTF's `CESIUM_RTC` center, in either glTF
//! version) is never added to positions: it is the translation of a node over the tile's
//! content, in double precision. Quantized positions keep their volume's offset the same way,
//! and points or instances stored far from the origin without either are measured from a whole
//! number of meters near their middle.
//!
//! # Batch tables
//!
//! A batch table is kept in the extras of the node over its tile's content: `batchLength`, and
//! `batchTable` with a list of one value for each feature under each property's name. Properties
//! held in the binary body are decoded into those lists (a vector as a list of its components),
//! and the table's own `extensions` and `extras` are kept as they are. A feature is what a
//! vertex's `_BATCHID` names in a b3dm, an instance (or its `_BATCHID` instance attribute) in an
//! i3dm, and a point's `_BATCHID` in a pnts; a point cloud without batch ids has per-point
//! properties, which become custom attributes instead.
//!
//! Not read: 3D Tiles 1.1 implicit tiling and its `.subtree` files, the vector and geometry tile
//! formats (`.vctr`, `.geom`), and metadata schemas, which are refused or left out as each
//! module says.

mod b3dm;
mod i3dm;
mod pnts;
mod table;
#[cfg(test)]
mod tests;
mod tileset;

use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value, json};

use super::xform::{self, IDENTITY, M4};
use super::{Input, gltf, graft};
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Bytes, Result};

/// How deep tiles nest: composites inside composites, and tilesets inside tilesets.
const MAX_DEPTH: usize = 16;

/// Tile space (Z up) to glTF's (Y up): `(x, y, z)` to `(x, z, -y)`.
const Z_TO_Y: M4 = [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]];
/// glTF's space (Y up) to tile space (Z up), the specification's Y-up-to-Z-up transform.
const Y_TO_Z: M4 = [[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]];
/// A glTF written X up to tile space.
const X_TO_Z: M4 = [[0., 0., -1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]];

/// Whether `p` is the origin: nothing was given to measure from.
fn unset(p: [f64; 3]) -> bool {
    p.iter().all(|&c| c == 0.0)
}

/// A point or direction in tile space, in glTF's axes.
fn y_up(p: [f64; 3]) -> [f64; 3] {
    // Taken from zero, so that no y is a zero, not a negative one.
    [p[0], p[2], 0.0 - p[1]]
}

/// The unit vector an oct-encoded pair stands for, each of the pair in `0..=range`: the pair
/// is a point on an octahedron's faces unfolded onto a square, as `CesiumJS` decodes it.
fn oct_decode(x: f64, y: f64, range: f64) -> [f64; 3] {
    let signed = |v: f64| v.clamp(0.0, range) / range * 2.0 - 1.0;
    let sign = |v: f64| if v < 0.0 { -1.0 } else { 1.0 };
    let (mut x, mut y) = (signed(x), signed(y));
    let z = 1.0 - (x.abs() + y.abs());
    if z < 0.0 {
        let old = x;
        x = (1.0 - y.abs()) * sign(old);
        y = (1.0 - old.abs()) * sign(y);
    }
    let len = (x * x + y * y + z * z).sqrt();
    [x / len, y / len, z / len]
}

/// Which way is up in a tile's glTF.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Up {
    X,
    /// glTF's own, and the specification's.
    Y,
    Z,
}

impl Up {
    /// The glTF's space to tile space.
    fn to_tile(self) -> M4 {
        match self {
            Up::X => X_TO_Z,
            Up::Y => Y_TO_Z,
            Up::Z => IDENTITY,
        }
    }
}

/// What a tile is read with.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    /// The file the tile is in: its references are found from its folder, and what a document
    /// keeps of its bytes is shared from them.
    input: Input<'a>,
    assets: &'a dyn Assets,
    keep: Keep,
    up: Up,
    /// How many composites and tilesets the tile is inside.
    depth: usize,
}

/// What was read, for the facts.
#[derive(Default)]
struct Stats {
    /// Each kind of content read, by magic, in the order first met.
    kinds: Vec<&'static str>,
    tiles: usize,
    instances: usize,
    features: u64,
    /// Tile contents a tileset names that aren't among the assets.
    missing: Vec<String>,
    /// The tileset's version, when one was read.
    version: Option<String>,
    /// Whether a tileset left tiles unrefined to stay under its cap.
    capped: bool,
}

impl Stats {
    fn kind(&mut self, kind: &'static str) {
        if !self.kinds.contains(&kind) {
            self.kinds.push(kind);
        }
    }
}

/// Reads a tile or a tileset.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let data = input.read()?;
    let ctx = Ctx {
        input,
        assets,
        keep,
        up: Up::Y,
        depth: 0,
    };
    let mut stats = Stats::default();
    let mut doc = content(&ctx, &data, &mut stats)?;

    let mut source = Map::new();
    source.insert("content".into(), json!(stats.kinds));
    if !stats.missing.is_empty() {
        source.insert("missingTiles".into(), json!(stats.missing));
    }
    let asset = doc.json().get("asset");
    let tool = asset
        .and_then(|a| a.get("generator"))
        .and_then(Value::as_str)
        .filter(|g| *g != "polysmelt")
        .map(str::to_owned);
    let mut info = SourceInfo {
        format: Some("3D Tiles".into()),
        version: Some(stats.version.take().unwrap_or_else(|| "1.0".into())),
        meters_per_unit: Fact::Declared(1.0),
        unit: Some("meter".into()),
        up_axis: Fact::Declared(Axis::PosZ),
        handedness: Fact::Declared(Handedness::Right),
        tool,
        ..SourceInfo::default()
    };
    info.converted.up_axis = true;
    info.write(&mut source);
    let asset = &mut doc.base_mut()["asset"];
    if !asset.is_object() {
        *asset = json!({ "version": "2.0" });
    }
    if !asset["extras"].is_object() {
        asset["extras"] = json!({});
    }
    asset["extras"]["source"] = Value::Object(source);

    let counts = doc.counts();
    let facts = &mut doc.facts;
    facts.insert("triangles".into(), json!(counts.triangles));
    if counts.points > 0 {
        facts.insert("points".into(), json!(counts.points));
    }
    if counts.lines > 0 {
        facts.insert("lines".into(), json!(counts.lines));
    }
    facts.insert("tiles".into(), json!(stats.tiles));
    if stats.instances > 0 {
        facts.insert("instances".into(), json!(stats.instances));
    }
    if stats.features > 0 {
        facts.insert("features".into(), json!(stats.features));
    }
    if !stats.missing.is_empty() {
        facts.insert("missing_tiles".into(), json!(stats.missing.len()));
    }
    if stats.capped {
        facts.insert("unrefined_tiles".into(), json!(true));
    }
    Ok(doc)
}

/// The document of one tile's content, in glTF's axes with the tile's origin at its own.
fn content(ctx: &Ctx<'_>, data: &[u8], stats: &mut Stats) -> Result<Document> {
    match data.get(..4) {
        Some(b"b3dm") => b3dm::read(ctx, data, stats),
        Some(b"i3dm") => i3dm::read(ctx, data, stats),
        Some(b"pnts") => pnts::read(ctx, data, stats),
        Some(b"cmpt") => composite(ctx, data, stats),
        Some(b"glTF") => plain(ctx, data, stats),
        Some(b"vctr" | b"geom") => bail!(
            Unsupported,
            "3D Tiles vector and geometry tiles (.vctr, .geom) aren't read"
        ),
        Some(b"subt") => bail!(Unsupported, "3D Tiles implicit tiling subtrees aren't read"),
        _ => {
            let text = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(data).trim_ascii_start();
            ensure!(
                text.starts_with(b"{"),
                Corrupt,
                "not a 3D Tiles tile or tileset: it starts with neither a tile's magic nor JSON"
            );
            // A tileset has a root tile and a geometric error; any other JSON here is a glTF.
            match serde_json::from_slice::<Value>(text) {
                Ok(Value::Object(json)) if json.contains_key("root") || json.contains_key("geometricError") => {
                    tileset::read(ctx, &json, stats)
                }
                Ok(_) => plain(ctx, data, stats),
                Err(e) => bail!(Corrupt, "3D Tiles tileset isn't JSON: {e}"),
            }
        }
    }
}

/// A glTF as a tile's whole content, as 3D Tiles 1.1 has it.
fn plain(ctx: &Ctx<'_>, data: &[u8], stats: &mut Stats) -> Result<Document> {
    let (mut doc, center) = model(ctx, data)?;
    let m = place(ctx.up, center);
    if m != IDENTITY {
        wrap(&mut doc, &m, "glTF", None);
    }
    stats.kind("glTF");
    stats.tiles += 1;
    Ok(doc)
}

/// A composite: a header of magic, version, byte length and tile count, then the tiles, each
/// saying its own length.
fn composite(ctx: &Ctx<'_>, data: &[u8], stats: &mut Stats) -> Result<Document> {
    ensure!(
        ctx.depth < MAX_DEPTH,
        LimitExceeded,
        "3D Tiles composites nest more than {MAX_DEPTH} deep"
    );
    let u32_at = |at: usize| crate::bytes::u32_at(data, at).map(|v| v as usize);
    let (Some(version), Some(declared), Some(count)) = (u32_at(4), u32_at(8), u32_at(12)) else {
        bail!(Truncated, "cmpt tile is cut short");
    };
    ensure!(version == 1, Unsupported, "cmpt version {version} isn't 1");
    ensure!(
        declared <= data.len(),
        Truncated,
        "cmpt tile is cut short: it holds {} bytes but declares {declared}",
        data.len()
    );
    let data = &data[..declared.max(16)];
    let inner = Ctx {
        depth: ctx.depth + 1,
        ..*ctx
    };
    let mut out = Merged::new("composite");
    let mut at = 16;
    // Each tile takes at least its twelve bytes of header, so the count is bounded by the data.
    for i in 0..count {
        let len = crate::bytes::u32_at(data, at + 8).map(|v| v as usize);
        let tile = len
            .filter(|&len| len >= 12)
            .and_then(|len| data.get(at..at.checked_add(len)?));
        let Some(tile) = tile else {
            bail!(
                Truncated,
                "cmpt tile is cut short: tile {i} of {count} runs past its end"
            );
        };
        at += tile.len();
        let doc = content(&inner, tile, stats)?;
        out.add(doc, &IDENTITY, None, ctx.input.limit)?;
    }
    stats.kind("cmpt");
    Ok(out.doc)
}

/// Documents joined under one root node.
struct Merged {
    doc: Document,
    copyrights: Vec<String>,
}

impl Merged {
    fn new(name: &str) -> Self {
        let mut doc = Document::default();
        let json = doc.base_mut();
        json.insert("scene".into(), json!(0));
        json.insert("scenes".into(), json!([{ "nodes": [0] }]));
        json.insert("nodes".into(), json!([{ "name": name }]));
        Self {
            doc,
            copyrights: Vec::new(),
        }
    }

    /// Adds `doc` under the root, placed by `m` (in glTF's axes) under a node named `name`
    /// when either is given. Refuses more than `limit` triangles in all.
    fn add(&mut self, mut doc: Document, m: &M4, name: Option<&str>, limit: usize) -> Result<()> {
        if let Some(c) = doc
            .json()
            .get("asset")
            .and_then(|a| a.get("copyright"))
            .and_then(Value::as_str)
            && !self.copyrights.iter().any(|x| x == c)
        {
            self.copyrights.push(c.to_owned());
            self.doc.base_mut()["asset"]["copyright"] = json!(self.copyrights.join("; "));
        }
        if *m != IDENTITY || name.is_some() {
            wrap(&mut doc, m, name.unwrap_or("tile"), None);
        }
        graft::graft(&mut self.doc, doc, 0, None);
        ensure!(
            self.doc.counts().triangles <= limit,
            LimitExceeded,
            "too many triangles to convert (more than {limit})"
        );
        Ok(())
    }
}

/// The glTF in `bytes` (binary or JSON, either version) as a document, with the center its
/// `CESIUM_RTC` extension gives, in tile space. The center is taken out of the document: the
/// caller places the content at it.
fn model(ctx: &Ctx<'_>, bytes: &[u8]) -> Result<(Document, [f64; 3])> {
    let shared = ctx.input.share(bytes);
    let sub = Input {
        limit: ctx.input.limit,
        ..Input::shared(&shared, ctx.input.path)
    };
    let mut doc = gltf::load(sub, ctx.assets, ctx.keep)?;
    let json = doc.json_mut();
    let mut center = [0.0; 3];
    let three = |v: &Value| -> Option<[f64; 3]> {
        let c: Vec<f64> = v.as_array()?.iter().filter_map(Value::as_f64).collect();
        (c.len() == 3 && c.iter().all(|x| x.is_finite())).then(|| [c[0], c[1], c[2]])
    };
    if let Some(c) = json.get("extensions").and_then(|e| e.pointer("/CESIUM_RTC/center")) {
        // glTF 2.0 with the extension: nothing has applied it yet.
        let Some(c) = three(c) else {
            bail!(Corrupt, "glTF CESIUM_RTC center isn't three numbers");
        };
        center = c;
        if let Some(Value::Object(e)) = json.get_mut("extensions") {
            e.remove("CESIUM_RTC");
            if e.is_empty() {
                json.remove("extensions");
            }
        }
        for key in ["extensionsUsed", "extensionsRequired"] {
            if let Some(Value::Array(list)) = json.get_mut(key) {
                list.retain(|x| x != "CESIUM_RTC");
                if list.is_empty() {
                    json.remove(key);
                }
            }
        }
    } else if let Some(c) = json
        .get("asset")
        .and_then(|a| a.pointer("/extras/source/extensions/CESIUM_RTC/center"))
        .and_then(three)
    {
        // glTF 1.0: the upgrade added the center to every root node, as a glTF on its own has
        // it. In a tile the center is in tile space, so it comes off again and is placed there.
        center = c;
        let n = list(json, "nodes").len();
        let mut root = vec![true; n];
        for node in list(json, "nodes") {
            for k in indices(node, "children") {
                if let Some(r) = root.get_mut(k) {
                    *r = false;
                }
            }
        }
        if let Some(Value::Array(nodes)) = json.get_mut("nodes") {
            for (node, _) in nodes.iter_mut().zip(root).filter(|(_, root)| *root) {
                let at = |v: &Value, i: usize| v.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                if let Some(m) = node
                    .get_mut("matrix")
                    .filter(|m| m.as_array().is_some_and(|m| m.len() == 16))
                {
                    for (k, x) in c.iter().enumerate() {
                        m[12 + k] = json!(at(m, 12 + k) - x);
                    }
                } else if let Some(t) = node.get("translation").cloned() {
                    node["translation"] = json!([at(&t, 0) - c[0], at(&t, 1) - c[1], at(&t, 2) - c[2]]);
                }
            }
        }
    }
    // A glTF 2.0 from before the name was settled calls its batch ids `BATCHID`.
    if let Some(Value::Array(meshes)) = json.get_mut("meshes") {
        for p in meshes
            .iter_mut()
            .filter_map(|m| m.get_mut("primitives")?.as_array_mut())
            .flatten()
        {
            if let Some(Value::Object(a)) = p.get_mut("attributes")
                && !a.contains_key("_BATCHID")
                && let Some(id) = a.remove("BATCHID")
            {
                a.insert("_BATCHID".into(), id);
            }
        }
    }
    Ok((doc, center))
}

/// Where a glTF written `up` up sits in a document when its origin is at `center` in tile space.
fn place(up: Up, center: [f64; 3]) -> M4 {
    xform::mul(&Z_TO_Y, &xform::mul(&xform::translate(center), &up.to_tile()))
}

/// `json[key]` as a list.
fn list<'v>(json: &'v Map<String, Value>, key: &str) -> &'v [Value] {
    json.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

/// The indices in the list at `v[key]`.
fn indices<'v>(v: &'v Value, key: &str) -> impl Iterator<Item = usize> + 'v {
    v.get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .filter_map(|i| usize::try_from(i).ok())
}

/// The nodes the document draws from: its default scene's, or with no scenes every node that
/// is no other's child.
fn roots(json: &Map<String, Value>) -> Vec<usize> {
    let nodes = list(json, "nodes");
    let scenes = list(json, "scenes");
    if scenes.is_empty() {
        let mut root = vec![true; nodes.len()];
        for node in nodes {
            for k in indices(node, "children") {
                if let Some(r) = root.get_mut(k) {
                    *r = false;
                }
            }
        }
        return (0..nodes.len()).filter(|&i| root[i]).collect();
    }
    let scene = json
        .get("scene")
        .and_then(Value::as_u64)
        .and_then(|i| usize::try_from(i).ok())
        .unwrap_or(0);
    scenes
        .get(scene)
        .map(|s| indices(s, "nodes").filter(|&i| i < nodes.len()).collect())
        .unwrap_or_default()
}

/// Puts what the document draws under one new node, named `name` and placed by `m`, with
/// `extras`; returns the node.
fn wrap(doc: &mut Document, m: &M4, name: &str, extras: Option<Value>) -> Option<usize> {
    let json = doc.json_mut();
    let children = roots(json);
    let mut node = json!({ "name": name });
    if !children.is_empty() {
        node["children"] = json!(children);
    }
    if *m != IDENTITY {
        xform::set_transform(&mut node, m);
    }
    if let Some(extras) = extras {
        node["extras"] = extras;
    }
    let nodes = json.entry("nodes").or_insert_with(|| json!([]));
    let Value::Array(nodes) = nodes else {
        return None;
    };
    let index = nodes.len();
    nodes.push(node);
    let scene = json.get("scene").and_then(Value::as_u64).unwrap_or(0);
    if let Some(scenes) = json
        .get_mut("scenes")
        .and_then(Value::as_array_mut)
        .filter(|s| !s.is_empty())
    {
        if let Some(s) = usize::try_from(scene).ok().and_then(|i| scenes.get_mut(i)) {
            s["nodes"] = json!([index]);
        }
    } else {
        json.insert("scenes".into(), json!([{ "nodes": [index] }]));
        json.insert("scene".into(), json!(0));
    }
    Some(index)
}

/// The extras of the node over a tile's content: what kind of tile it is, how many features it
/// has, and its batch table with `count` values to a property.
fn batch_extras(kind: &str, batch: &table::Table<'_>, length: u64, count: usize) -> Result<Value> {
    let mut extras = json!({ "tile": kind, "batchLength": length });
    let listed = batch.listed(count)?;
    if !listed.is_empty() {
        extras["batchTable"] = Value::Object(listed);
    }
    Ok(extras)
}

/// The path of the file a tile names by `uri`, relative to the file it's in. Only a relative
/// path is followed: no scheme, root, drive or share, though it may climb with `..` as tilesets
/// that share their tiles do. A query or fragment is left off.
fn reference(ctx: &Ctx<'_>, uri: &str) -> Result<PathBuf> {
    let uri = uri.split(['?', '#']).next().unwrap_or(uri);
    let rel = crate::gltf::percent_decode(uri);
    let safe = !rel.is_empty()
        && !rel.contains(':')
        && Path::new(&rel)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir | Component::ParentDir));
    ensure!(safe, Unsupported, "refusing to read {rel:?}: not a relative path");
    Ok(ctx.input.dir().join(rel))
}

/// The bytes of the file at `path`, which a tile named as `uri`.
fn fetch(ctx: &Ctx<'_>, path: &Path, uri: &str) -> Result<Bytes> {
    ctx.assets
        .read_shared(path)
        .ok_or_else(|| crate::Error::missing(format!("3D Tiles file {uri:?} isn't among the assets")).at(path))
}

/// A node's own transform.
fn local(node: &Value) -> M4 {
    let numbers = |key: &str, n: usize| -> Option<Vec<f64>> {
        let v: Vec<f64> = node.get(key)?.as_array()?.iter().filter_map(Value::as_f64).collect();
        (v.len() == n && v.iter().all(|x| x.is_finite())).then_some(v)
    };
    if let Some(m) = numbers("matrix", 16) {
        let mut flat = [0.0; 16];
        flat.copy_from_slice(&m);
        return xform::from_column_major(&flat);
    }
    let mut m = IDENTITY;
    if let Some(s) = numbers("scale", 3) {
        m = xform::scale([s[0], s[1], s[2]]);
    }
    if let Some(r) = numbers("rotation", 4) {
        let q = [r[0], r[1], r[2], r[3]];
        if crate::quat::length(q) > 0.0 {
            m = xform::mul(&xform::rotation(crate::quat::unit_unchecked(q)), &m);
        }
    }
    if let Some(t) = numbers("translation", 3) {
        m = xform::mul(&xform::translate([t[0], t[1], t[2]]), &m);
    }
    m
}

/// Every node's transform from the scene it's drawn in, for the nodes under `roots`; `None`
/// for one that isn't.
fn worlds(json: &Map<String, Value>, roots: &[usize]) -> Vec<Option<M4>> {
    let nodes = list(json, "nodes");
    let mut world: Vec<Option<M4>> = vec![None; nodes.len()];
    let mut stack: Vec<(usize, M4)> = roots.iter().map(|&r| (r, IDENTITY)).collect();
    while let Some((i, parent)) = stack.pop() {
        // A node met twice is in a cycle or under two parents: it keeps its first place.
        let Some(slot @ None) = world.get_mut(i) else {
            continue;
        };
        let m = xform::mul(&parent, &local(&nodes[i]));
        *slot = Some(m);
        stack.extend(indices(&nodes[i], "children").map(|k| (k, m)));
    }
    world
}
