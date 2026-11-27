//! [`Document::decode`]: a compressed or quantized document rewritten as plain glTF 2.0.

use std::collections::{BTreeMap, HashMap};

use serde_json::{Map, Value, json};

use super::{Reader, index, items, list};
use crate::document::Document;
use crate::gltf::MESHOPT_EXTENSIONS;
use crate::{Context as _, ErrorKind, Result};

const DRACO: &str = "KHR_draco_mesh_compression";
const QUANTIZATION: &str = "KHR_mesh_quantization";

/// Attributes `KHR_mesh_quantization` lets be integers, which plain glTF has as floats.
fn quantizable(name: &str) -> bool {
    matches!(name, "POSITION" | "NORMAL" | "TANGENT") || name.starts_with("TEXCOORD_")
}

/// An accessor's new elements, tightly packed.
struct Data {
    bytes: Vec<u8>,
    ctype: u64,
    count: usize,
    /// Its new `min` and `max`, when it had them or is a `POSITION`.
    bounds: Option<(Vec<f32>, Vec<f32>)>,
    /// Whether it holds indices (the view's `target`).
    indices: bool,
}

/// What decoding changes: meshopt views' decoded bytes, and accessors' new elements.
#[derive(Default)]
struct Plan {
    views: BTreeMap<usize, Vec<u8>>,
    accessors: BTreeMap<usize, Data>,
}

impl Document {
    /// Rewrites the document as plain glTF 2.0 that any reader takes:
    ///
    /// - meshopt-compressed buffer views (`EXT_meshopt_compression`, `KHR_meshopt_compression`)
    ///   decoded;
    /// - Draco-compressed primitives (`KHR_draco_mesh_compression`, with the `draco` feature)
    ///   decoded into accessors of their own, attributes as floats (integer ones, such as
    ///   `JOINTS_n`, in their declared component type) and indices as declared;
    /// - quantized `POSITION`, `NORMAL`, `TANGENT` and `TEXCOORD_n` accessors
    ///   (`KHR_mesh_quantization`), morph targets' included, converted to floats with the same
    ///   values a reader would see, and their bounds recomputed. Node transforms are kept, so a
    ///   mesh quantized with a scaling node draws where it did;
    /// - indices holding their type's largest value (255 as bytes, 65535 as shorts), which glTF
    ///   forbids and WebGL 2 reads as a primitive restart, widened to the next type.
    ///
    /// Those extensions leave `extensionsUsed` and `extensionsRequired`, and buffers nothing
    /// points into any more (meshopt fallbacks, buffers that held only compressed data) are
    /// dropped. Decoded data goes into a new buffer; compressed data sharing a buffer with
    /// other views stays where it was. Everything else (animation, skins, materials, unknown
    /// extensions and extras, required ones too) is untouched. A document without these is left
    /// as it is.
    ///
    /// # Errors
    ///
    /// When compressed data doesn't decode, an accessor is malformed, or the document uses Draco
    /// without the `draco` feature (unless it carries uncompressed data beside it). The
    /// document is unchanged then.
    pub fn decode(&mut self) -> Result<()> {
        let plan = read(self)?;
        let draco = has_draco(&self.base);
        if !plan.views.is_empty() || !plan.accessors.is_empty() || draco {
            // What a reader built has neither: only a file's JSON is changed here.
            self.json_mut();
            apply(self, &plan);
            drop_unused_buffers(self);
        }
        for key in ["extensionsUsed", "extensionsRequired"] {
            if !self.base.contains_key(key) {
                continue;
            }
            let json = self.base_mut();
            if let Some(a) = json.get_mut(key).and_then(Value::as_array_mut) {
                a.retain(|e| {
                    e.as_str()
                        .is_none_or(|e| !(e == DRACO || e == QUANTIZATION || MESHOPT_EXTENSIONS.contains(&e)))
                });
                if a.is_empty() {
                    json.remove(key);
                }
            }
        }
        Ok(())
    }
}

fn primitives(json: &Map<String, Value>) -> impl Iterator<Item = &Value> {
    list(json, "meshes").iter().flat_map(|m| items(m, "primitives"))
}

fn has_draco(json: &Map<String, Value>) -> bool {
    primitives(json).any(|p| p.get("extensions").and_then(|e| e.get(DRACO)).is_some())
}

/// Decodes everything [`Document::decode`] replaces, changing nothing yet.
fn read(doc: &Document) -> Result<Plan> {
    let json = &doc.base;
    // Only accessors are read: a required extension decoding leaves alone stays required.
    let mut r = Reader::bare(doc);
    let mut plan = Plan::default();
    for i in 0..r.tabs.views() {
        r.meshopt(i)?;
    }

    // Draco: which stream each accessor was decoded from, so one shared by primitives decodes once.
    let mut source: HashMap<usize, usize> = HashMap::new();
    for p in primitives(json) {
        let Some(ext) = p.get("extensions").and_then(|e| e.get(DRACO)) else {
            continue;
        };
        let attrs = p.get("attributes").cloned().unwrap_or(Value::Null);
        let view = index(ext, "bufferView").context(ErrorKind::Corrupt, "glTF Draco primitive has no bufferView")?;
        #[cfg(feature = "draco")]
        {
            r.draco.clear();
            r.decode_draco(ext, &attrs, index(p, "indices"))?;
            for (acc, bytes, _, n) in std::mem::take(&mut r.draco) {
                match source.insert(acc, view) {
                    Some(v) if v != view => {
                        bail!(Corrupt, "glTF accessor {acc} is decoded from two Draco streams")
                    }
                    Some(_) => continue,
                    None => {}
                }
                let is_indices = index(p, "indices") == Some(acc);
                let declared = r.get("accessors", acc)?;
                plan.accessors
                    .insert(acc, from_draco(declared, &bytes, n, is_indices, &attrs, acc));
            }
        }
        #[cfg(not(feature = "draco"))]
        {
            // Readable only from the uncompressed data some files carry beside the stream.
            let _ = (&mut source, view);
            let names = ext.get("attributes").and_then(Value::as_object);
            let accs = names
                .into_iter()
                .flat_map(|n| n.keys())
                .filter_map(|n| index(&attrs, n))
                .chain(index(p, "indices"));
            for a in accs {
                ensure!(
                    r.acc(a)?.view.is_some(),
                    FeatureDisabled,
                    "glTF uses Draco compression, which needs Polysmelt's `draco` feature"
                );
            }
        }
    }

    // Quantized attributes, morph targets' included.
    for p in primitives(json) {
        let sets = p.get("attributes").into_iter().chain(items(p, "targets"));
        for set in sets.filter_map(Value::as_object) {
            for (name, a) in set {
                let Some(a) = a.as_u64().map(crate::num::usize_of) else {
                    continue;
                };
                if !quantizable(name) || plan.accessors.contains_key(&a) {
                    continue;
                }
                let acc = r.acc(a)?;
                if acc.ctype == 5126 {
                    continue;
                }
                let values = r.read_acc(&acc, [0.0f32; 4])?;
                let declared = r.tabs.declares_bounds(a);
                let floats: Vec<f32> = values.iter().flat_map(|v| v[..acc.n].iter().copied()).collect();
                plan.accessors
                    .insert(a, floats_data(declared, floats, acc.n, name == "POSITION"));
            }
        }
    }
    // Indices holding their type's largest value, which glTF forbids and WebGL 2 reads as a
    // primitive restart: widened to the next type.
    for p in primitives(json) {
        let Some(a) = index(p, "indices") else { continue };
        if plan.accessors.contains_key(&a) {
            continue;
        }
        let acc = r.acc(a)?;
        let (top, wider) = match acc.ctype {
            5121 => (255.0, 5123),
            5123 => (65535.0, 5125),
            _ => continue,
        };
        let values = r.read_acc(&acc, [0.0f32; 4])?;
        if !values.iter().any(|v| v[0] >= top) {
            continue;
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "u8 and u16 indices read as floats exactly"
            )]
            let i = v[0] as u32;
            if wider == 5123 {
                bytes.extend_from_slice(&u16::try_from(i).unwrap_or(u16::MAX).to_le_bytes());
            } else {
                bytes.extend_from_slice(&i.to_le_bytes());
            }
        }
        let count = values.len();
        plan.accessors.insert(
            a,
            Data {
                bytes,
                ctype: wider,
                count,
                bounds: None,
                indices: true,
            },
        );
    }
    plan.views = r.meshopt.drain().collect();
    Ok(plan)
}

/// Float elements as an accessor's new data, for one that declares bounds when `declared`.
fn floats_data(declared: bool, floats: Vec<f32>, n: usize, position: bool) -> Data {
    let count = floats.len() / n.max(1);
    let bounds = (position || declared).then(|| bounds(&floats, n));
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    Data {
        bytes,
        ctype: 5126,
        count,
        bounds,
        indices: false,
    }
}

fn bounds(floats: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut min = vec![f32::INFINITY; n];
    let mut max = vec![f32::NEG_INFINITY; n];
    for e in floats.chunks_exact(n) {
        for c in 0..n {
            min[c] = min[c].min(e[c]);
            max[c] = max[c].max(e[c]);
        }
    }
    if floats.is_empty() {
        (vec![0.0; n], vec![0.0; n])
    } else {
        (min, max)
    }
}

/// A Draco-decoded accessor's new data. Indices come as `u32`s, attributes as floats (already
/// normalized where the accessor says so); integer attributes go back to their declared type.
#[cfg(feature = "draco")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "decoded integer attributes are written back in the type they were declared as"
)]
fn from_draco(declared: &Value, bytes: &[u8], n: usize, is_indices: bool, attrs: &Value, acc: usize) -> Data {
    let ctype = declared.get("componentType").and_then(Value::as_u64).unwrap_or(5126);
    let normalized = declared.get("normalized").and_then(Value::as_bool).unwrap_or(false);
    if is_indices {
        let idx: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let max = idx.iter().copied().max().unwrap_or(0);
        let ctype = match ctype {
            // The type's largest value restarts the strip in WebGL 2 and glTF forbids it.
            5121 if max < 0xFF => 5121,
            5121 | 5123 if max < 0xFFFF => 5123,
            _ => 5125,
        };
        let mut out = Vec::with_capacity(idx.len() * 4);
        for i in &idx {
            match ctype {
                5121 => out.push(*i as u8),
                5123 => out.extend_from_slice(&(*i as u16).to_le_bytes()),
                _ => out.extend_from_slice(&i.to_le_bytes()),
            }
        }
        return Data {
            bytes: out,
            ctype,
            count: idx.len(),
            bounds: None,
            indices: true,
        };
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let position = attrs.get("POSITION").and_then(Value::as_u64) == Some(acc as u64);
    if ctype == 5126 || normalized {
        let declared = declared.get("min").is_some() || declared.get("max").is_some();
        return floats_data(declared, floats, n, position);
    }
    let mut out = Vec::new();
    for f in &floats {
        let f = f.round();
        match ctype {
            5120 => out.push((f as i8).cast_unsigned()),
            5121 => out.push(f as u8),
            5122 => out.extend_from_slice(&(f as i16).to_le_bytes()),
            5123 => out.extend_from_slice(&(f as u16).to_le_bytes()),
            _ => out.extend_from_slice(&(f as u32).to_le_bytes()),
        }
    }
    Data {
        bytes: out,
        ctype,
        count: floats.len() / n.max(1),
        bounds: None,
        indices: false,
    }
}

/// Writes the plan into the document: the decoded bytes into a new buffer, views and accessors
/// pointed at them, Draco and meshopt extensions removed.
fn apply(doc: &mut Document, plan: &Plan) {
    let buffer = doc.buffers.len();
    let mut out: Vec<u8> = Vec::new();
    let mut place = |bytes: &[u8]| {
        out.resize(out.len().next_multiple_of(4), 0);
        let at = out.len();
        out.extend_from_slice(bytes);
        at
    };

    let views = doc.base.get_mut("bufferViews").and_then(Value::as_array_mut);
    for (i, v) in views.into_iter().flatten().enumerate() {
        let Some(v) = v.as_object_mut() else { continue };
        if let Some(bytes) = plan.views.get(&i) {
            v.insert("buffer".into(), json!(buffer));
            v.insert("byteOffset".into(), json!(place(bytes)));
            v.insert("byteLength".into(), json!(bytes.len()));
        }
        strip(v, &MESHOPT_EXTENSIONS);
    }

    let mut new_views = Vec::new();
    for (a, d) in &plan.accessors {
        let at = place(&d.bytes);
        let mut view = json!({ "buffer": buffer, "byteOffset": at, "byteLength": d.bytes.len() });
        view["target"] = json!(if d.indices { 34963 } else { 34962 });
        new_views.push((*a, view));
    }
    for (a, view) in new_views {
        let v = doc.push_view(view);
        let d = &plan.accessors[&a];
        let Some(acc) = doc
            .base
            .get_mut("accessors")
            .and_then(Value::as_array_mut)
            .and_then(|l| l.get_mut(a))
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        acc.insert("bufferView".into(), json!(v));
        acc.insert("componentType".into(), json!(d.ctype));
        acc.insert("count".into(), json!(d.count));
        for k in ["byteOffset", "normalized", "sparse"] {
            acc.remove(k);
        }
        if let Some((min, max)) = &d.bounds {
            acc.insert("min".into(), json!(min));
            acc.insert("max".into(), json!(max));
        }
    }

    let meshes = doc.base.get_mut("meshes").and_then(Value::as_array_mut);
    let prims = meshes
        .into_iter()
        .flatten()
        .filter_map(|m| m.get_mut("primitives").and_then(Value::as_array_mut))
        .flatten();
    for p in prims.filter_map(Value::as_object_mut) {
        strip(p, &[DRACO]);
    }

    if !out.is_empty() {
        out.resize(out.len().next_multiple_of(4), 0);
        doc.push_buffer(out);
    }
}

/// Removes the named extensions from an object, and `extensions` itself once empty.
fn strip(obj: &mut Map<String, Value>, names: &[&str]) {
    let Some(exts) = obj.get_mut("extensions").and_then(Value::as_object_mut) else {
        return;
    };
    for n in names {
        exts.remove(*n);
    }
    if exts.is_empty() {
        obj.remove("extensions");
    }
}

/// Drops buffers no view points into, renumbering the views'.
fn drop_unused_buffers(doc: &mut Document) {
    let views = list(&doc.base, "bufferViews");
    let mut used = vec![false; doc.buffers.len()];
    for b in views.iter().filter_map(|v| index(v, "buffer")) {
        if let Some(u) = used.get_mut(b) {
            *u = true;
        }
    }
    if used.iter().all(|&u| u) {
        return;
    }
    let mut renumber = Vec::with_capacity(used.len());
    let mut next = 0;
    for &u in &used {
        renumber.push(next);
        next += usize::from(u);
    }
    let mut keep = used.iter().copied();
    doc.buffers.retain(|_| keep.next().unwrap_or(true));
    if let Some(list) = doc.base.get_mut("buffers").and_then(Value::as_array_mut) {
        let mut keep = used.iter().copied();
        list.retain(|_| keep.next().unwrap_or(true));
        for b in list.iter_mut().filter_map(Value::as_object_mut) {
            strip(b, &MESHOPT_EXTENSIONS);
        }
    }
    let views = doc.base.get_mut("bufferViews").and_then(Value::as_array_mut);
    for v in views.into_iter().flatten() {
        if let Some(b) = index(v, "buffer") {
            v["buffer"] = json!(renumber.get(b).copied().unwrap_or(b));
        }
    }
    if doc.buffers.is_empty() {
        doc.base.remove("buffers");
    }
}
