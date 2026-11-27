//! Instanced 3D Model: one glTF drawn many times, each instance with a position, an
//! orientation and a scale.
//!
//! - The glTF is embedded, or named by a URI found beside the tile through the
//!   [`Assets`](crate::assets::Assets).
//! - Positions are `POSITION`, or `POSITION_QUANTIZED` times `QUANTIZED_VOLUME_SCALE / 65535`
//!   plus `QUANTIZED_VOLUME_OFFSET`.
//! - An orientation is `NORMAL_UP` and `NORMAL_RIGHT` (or their `_OCT32P` forms, decoded): the
//!   model's X axis goes to right, its Y to up and its Z to right cross up. Vectors that aren't
//!   quite at right angles (decoded ones seldom are) keep right and the plane of the two.
//!   Without them `EAST_NORTH_UP` stands each instance on the WGS84 ellipsoid, its X east, Y
//!   north and Z up at its position in the tile (with the `RTC_CENTER`; a tileset's transform
//!   over the tile isn't looked at). With neither, instances aren't turned.
//! - The scale is `SCALE` times `SCALE_NON_UNIFORM`, along the model's axes before it's turned.
//!
//! All of that is in tile space, Z up, where the glTF arrives through the Y-up-to-Z-up
//! rotation. In the document each node that draws a mesh carries the instances as
//! `EXT_mesh_gpu_instancing`, the way the rest of the crate reads and bakes them: the instance
//! transforms are rewritten for where the node stands in the glTF, as translation, rotation and
//! scale in single precision. They're measured from the `RTC_CENTER` (or the quantized volume's
//! offset, or for plain positions far from the origin a whole number of meters near their
//! middle), which is the translation of a node over the glTF, in double precision.
//! `BATCH_ID` becomes the `_BATCHID` instance attribute, and the batch table, a value for each
//! instance, is in that node's extras.
//!
//! What the extension can't hold:
//! - A node whose instances would shear (a non-uniform scale under a turned node) draws them
//!   as nodes of its own instead: for each, one where the instance stands over one where the node stood.
//! - Animations are dropped: an instanced node's motion would move its instances about the
//!   node, not each about itself.
//! - A skinned mesh, and a node that already has instances, is left as the glTF has it, drawn
//!   once.

use serde_json::{Value, json};

use super::table::Component;
use super::xform::{self, IDENTITY, M4};
use super::{
    Ctx, Stats, batch_extras, fetch, indices, list, model, oct_decode, place, reference, roots, table, unset, worlds,
    wrap,
};
use crate::document::Document;
use crate::formats::Input;
use crate::formats::points;
use crate::num::{f32_of, usize_of};
use crate::{Context as _, ErrorKind, Result, Within as _};

const EXTENSION: &str = "EXT_mesh_gpu_instancing";

/// Where a node's instance attributes start in the buffer made for them.
struct Attributes {
    node: usize,
    translation: usize,
    rotation: Option<usize>,
    scale: Option<usize>,
}

pub(super) fn read(ctx: &Ctx<'_>, data: &[u8], stats: &mut Stats) -> Result<Document> {
    let tile = table::parse(data, "i3dm")?;
    let ft = &tile.feature;
    let count = ft
        .count("INSTANCES_LENGTH")?
        .context(ErrorKind::Corrupt, "i3dm tile has no INSTANCES_LENGTH")?;
    let n = usize_of(count);

    // Positions in tile space, measured from `base`.
    let rtc = ft.vec3("RTC_CENTER")?.unwrap_or([0.0; 3]);
    let mut base = rtc;
    let mut positions: Vec<[f64; 3]>;
    if let Some(p) = ft.floats("POSITION", n, 3, Component::F32, false)? {
        positions = p.chunks_exact(3).map(|p| [0, 1, 2].map(|k| f64::from(p[k]))).collect();
        if unset(base) {
            let mut min = [f64::INFINITY; 3];
            let mut max = [f64::NEG_INFINITY; 3];
            for p in positions.iter().filter(|p| p.iter().all(|c| c.is_finite())) {
                for k in 0..3 {
                    min[k] = min[k].min(p[k]);
                    max[k] = max[k].max(p[k]);
                }
            }
            base = points::origin(min, max);
            for p in &mut positions {
                *p = [0, 1, 2].map(|k| p[k] - base[k]);
            }
        }
    } else if let Some(q) = ft.floats("POSITION_QUANTIZED", n, 3, Component::U16, false)? {
        let (Some(offset), Some(scale)) = (ft.vec3("QUANTIZED_VOLUME_OFFSET")?, ft.vec3("QUANTIZED_VOLUME_SCALE")?)
        else {
            bail!(
                Corrupt,
                "i3dm tile has POSITION_QUANTIZED without QUANTIZED_VOLUME_OFFSET and QUANTIZED_VOLUME_SCALE"
            );
        };
        base = [0, 1, 2].map(|k| rtc[k] + offset[k]);
        positions = q
            .chunks_exact(3)
            .map(|q| [0, 1, 2].map(|k| f64::from(q[k]) * scale[k] / 65535.0))
            .collect();
    } else {
        bail!(Corrupt, "i3dm tile has neither POSITION nor POSITION_QUANTIZED");
    }

    // Each instance's right and up.
    let vectors = |key: &str| -> Result<Option<Vec<[f64; 3]>>> {
        if let Some(v) = ft.floats(key, n, 3, Component::F32, false)? {
            return Ok(Some(
                v.chunks_exact(3).map(|v| [0, 1, 2].map(|k| f64::from(v[k]))).collect(),
            ));
        }
        let packed = ft.floats(&format!("{key}_OCT32P"), n, 2, Component::U16, false)?;
        Ok(packed.map(|v| {
            v.chunks_exact(2)
                .map(|v| oct_decode(f64::from(v[0]), f64::from(v[1]), 65535.0))
                .collect()
        }))
    };
    let frames = vectors("NORMAL_UP")?.zip(vectors("NORMAL_RIGHT")?);
    let east_north_up = frames.is_none() && ft.flag("EAST_NORTH_UP");
    let uniform = ft.floats("SCALE", n, 1, Component::F32, false)?;
    let stretch = ft.floats("SCALE_NON_UNIFORM", n, 3, Component::F32, false)?;
    let ids = ft.floats("BATCH_ID", n, 1, Component::U16, true)?;

    // Each instance's transform in tile space, from `base`.
    let instances: Vec<M4> = (0..n)
        .map(|i| {
            let p = positions[i];
            let axes = match &frames {
                Some((up, right)) => frame(right[i], up[i]),
                None if east_north_up => east_north_up_at([0, 1, 2].map(|k| p[k] + base[k])),
                None => [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            };
            let mut s = [uniform.as_ref().map_or(1.0, |s| f64::from(s[i])); 3];
            if let Some(v) = &stretch {
                for k in 0..3 {
                    s[k] *= f64::from(v[i * 3 + k]);
                }
            }
            let mut m = IDENTITY;
            for row in 0..3 {
                for col in 0..3 {
                    m[row][col] = axes[col][row] * s[col];
                }
                m[row][3] = p[row];
            }
            m
        })
        .collect();

    let (mut doc, center) = match tile.gltf_format {
        1 => {
            ensure!(
                !tile.body.is_empty(),
                Truncated,
                "i3dm tile has no glTF after its tables"
            );
            model(ctx, tile.body).within("reading an i3dm tile's glTF")?
        }
        0 => {
            let end = tile.body.iter().position(|&b| b == 0).unwrap_or(tile.body.len());
            let uri = crate::bytes::text(&tile.body[..end]);
            let uri = uri.trim();
            ensure!(!uri.is_empty(), Corrupt, "i3dm tile names no glTF");
            let path = reference(ctx, uri)?;
            let bytes = fetch(ctx, &path, uri)?;
            let sub = Ctx {
                input: Input {
                    limit: ctx.input.limit,
                    ..Input::shared(&bytes, &path)
                },
                ..*ctx
            };
            model(&sub, &bytes).map_err(|e| e.at(&path))?
        }
        f => bail!(Corrupt, "i3dm gltfFormat {f} is neither 0 (a URI) nor 1 (embedded)"),
    };

    // The glTF's own space to tile space.
    let to_tile = xform::mul(&xform::translate(center), &ctx.up.to_tile());
    let json = doc.json_mut();
    json.remove("animations");
    let top = roots(json);
    let world = worlds(json, &top);
    let drawn: Vec<(usize, M4)> = list(json, "nodes")
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            node.get("mesh").is_some_and(Value::is_u64)
                && node.get("skin").is_none()
                && node.pointer("/extensions/EXT_mesh_gpu_instancing").is_none()
        })
        .filter_map(|(i, _)| Some((i, xform::mul(&to_tile, &world[i]?))))
        .collect();
    let limit = ctx.input.limit;
    ensure!(
        n.saturating_mul(drawn.len()) <= limit,
        LimitExceeded,
        "i3dm tile draws {} meshes {n} times each, more than the {limit} allowed",
        drawn.len()
    );

    // Instance attributes, all in one new buffer: (offset, type) by node.
    let mut blob: Vec<u8> = Vec::new();
    let mut push = |values: &[f32]| {
        let at = blob.len();
        for v in values {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        at
    };
    let mut attributes = Vec::new();
    // Nodes drawn in place of a node's instances, where they would shear: the mesh, where the
    // instance stands, and where the node stood in it. Together they shear, which no one node
    // of a glTF may; each alone does not.
    let mut loose: Vec<(Value, M4, M4)> = Vec::new();
    let mut emptied = Vec::new();
    for (node, to) in &drawn {
        let back = xform::inverse(to);
        let parts: Vec<xform::Trs> = instances
            .iter()
            .map(|m| xform::decompose(&xform::mul(&back, &xform::mul(m, to))))
            .collect();
        if n == 0 {
            emptied.push(*node);
        } else if parts.iter().all(|p| p.exact) {
            let flat = |f: &dyn Fn(&xform::Trs) -> Vec<f64>| -> Vec<f32> {
                parts.iter().flat_map(|p| f(p).into_iter().map(f32_of)).collect()
            };
            let turned = parts.iter().any(|p| (p.r[3].abs() - 1.0).abs() > 1e-12);
            let scaled = parts.iter().any(|p| p.s.iter().any(|s| (s - 1.0).abs() > 1e-12));
            attributes.push(Attributes {
                node: *node,
                translation: push(&flat(&|p| p.t.to_vec())),
                rotation: turned.then(|| push(&flat(&|p| p.r.to_vec()))),
                scale: scaled.then(|| push(&flat(&|p| p.s.to_vec()))),
            });
        } else {
            let mesh = json["nodes"][*node]["mesh"].clone();
            let from_tile = xform::inverse(&to_tile);
            loose.extend(instances.iter().map(|m| (mesh.clone(), xform::mul(&from_tile, m), *to)));
            emptied.push(*node);
        }
    }
    let batch_ids = ids.filter(|_| !attributes.is_empty()).map(|ids| push(&ids));

    if let Some(Value::Array(nodes)) = json.get_mut("nodes") {
        for &i in &emptied {
            if let Some(node) = nodes[i].as_object_mut() {
                node.remove("mesh");
                node.remove("weights");
            }
        }
    }
    if !attributes.is_empty() {
        let buffer = doc.push_buffer(blob);
        let accessor = |doc: &mut Document, at: usize, kind: &str, width: usize| {
            let view = doc.push_view(json!({ "buffer": buffer, "byteOffset": at, "byteLength": n * width * 4 }));
            let json = doc.json_mut();
            let list = json.entry("accessors").or_insert_with(|| json!([]));
            let index = list.as_array().map_or(0, Vec::len);
            if let Some(list) = list.as_array_mut() {
                list.push(json!({ "bufferView": view, "componentType": 5126, "count": n, "type": kind }));
            }
            index
        };
        let batch_ids = batch_ids.map(|at| accessor(&mut doc, at, "SCALAR", 1));
        for a in &attributes {
            let mut names = json!({ "TRANSLATION": accessor(&mut doc, a.translation, "VEC3", 3) });
            if let Some(at) = a.rotation {
                names["ROTATION"] = json!(accessor(&mut doc, at, "VEC4", 4));
            }
            if let Some(at) = a.scale {
                names["SCALE"] = json!(accessor(&mut doc, at, "VEC3", 3));
            }
            if let Some(ids) = batch_ids {
                names["_BATCHID"] = json!(ids);
            }
            let node = &mut doc.json_mut()["nodes"][a.node];
            if !node["extensions"].is_object() {
                node["extensions"] = json!({});
            }
            node["extensions"][EXTENSION] = json!({ "attributes": names });
        }
        let used = doc.json_mut().entry("extensionsUsed").or_insert_with(|| json!([]));
        if let Some(used) = used.as_array_mut()
            && !used.iter().any(|x| x == EXTENSION)
        {
            used.push(json!(EXTENSION));
        }
    }

    let extras = if ctx.keep.extras {
        Some(batch_extras("i3dm", &tile.batch, count, n)?)
    } else {
        None
    };
    let m = place(ctx.up, [0, 1, 2].map(|k| base[k] + center[k]));
    let over = if m != IDENTITY || extras.is_some() || !loose.is_empty() {
        wrap(&mut doc, &m, "i3dm", extras)
    } else {
        None
    };
    if let Some(over) = over.filter(|_| !loose.is_empty()) {
        let json = doc.json_mut();
        if let Some(Value::Array(nodes)) = json.get_mut("nodes") {
            let mut children: Vec<usize> = indices(&nodes[over], "children").collect();
            for (mesh, instance, node) in &loose {
                let at = nodes.len();
                nodes.push(json!({ "matrix": xform::to_column_major(instance), "children": [at + 1] }));
                nodes.push(json!({ "mesh": mesh, "matrix": xform::to_column_major(node) }));
                children.push(at);
            }
            nodes[over]["children"] = json!(children);
        }
    }
    stats.kind("i3dm");
    stats.tiles += 1;
    stats.instances += n;
    Ok(doc)
}

/// The axes an instance's X, Y and Z go to, from its right and up: right, up and right cross
/// up, made to stand at right angles. The model's own axes when the two give no frame.
pub(super) fn frame(right: [f64; 3], up: [f64; 3]) -> [[f64; 3]; 3] {
    let unit = |v: [f64; 3]| {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (len.is_finite() && len > 1e-12).then(|| v.map(|c| c / len))
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let found = unit(right).and_then(|right| {
        let forward = unit(cross(right, up))?;
        Some([right, cross(forward, right), forward])
    });
    found.unwrap_or([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
}

/// East, north and up on the WGS84 ellipsoid at `p`, meters from the Earth's center: up is the
/// ellipsoid's normal there, east is at right angles to the axis, and at the poles and the
/// center (where east has no meaning) the frame `CesiumJS` uses stands in.
pub(super) fn east_north_up_at(p: [f64; 3]) -> [[f64; 3]; 3] {
    const EQUATOR: f64 = 6_378_137.0;
    const POLE: f64 = 6_356_752.314_245_179;
    let tiny = |v: f64| v.abs() <= 1e-14;
    if !p.iter().all(|c| c.is_finite()) || (tiny(p[0]) && tiny(p[1]) && tiny(p[2])) {
        return [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]];
    }
    if tiny(p[0]) && tiny(p[1]) {
        let sign = p[2].signum();
        return [[0., 1., 0.], [-sign, 0., 0.], [0., 0., sign]];
    }
    let unit = |v: [f64; 3]| {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        v.map(|c| c / len)
    };
    let up = unit([
        p[0] / (EQUATOR * EQUATOR),
        p[1] / (EQUATOR * EQUATOR),
        p[2] / (POLE * POLE),
    ]);
    let east = unit([-p[1], p[0], 0.0]);
    let north = [
        up[1] * east[2] - up[2] * east[1],
        up[2] * east[0] - up[0] * east[2],
        up[0] * east[1] - up[1] * east[0],
    ];
    [east, north, up]
}
