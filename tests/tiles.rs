//! 3D Tiles against a reference decoder. Beside each case's model under `tests/corpus/tiles/`, `reference.json`
//! holds what `tools/corpus/tiles.py` reads from the file, by the specification and without Polysmelt:
//!
//! - a point cloud's positions, hashed (64-bit FNV-1a) as the single-precision values the file stores, with the
//!   node translation its `RTC_CENTER` or quantized volume gives; its colors, hashed as the levels the file
//!   stores; its normals as sums and samples; and each attribute, hashed;
//! - each instance's transform to the scene, as a matrix in double precision;
//! - a batched model's positions, hashed as its glTF stores them, and the translation its centers give;
//! - a batch table, decoded;
//! - each tile's placement in a tileset.
//!
//! What Polysmelt reads must be the same: bit for bit where the file's values pass through, and to within
//! single precision of the document's own coordinates where they're worked out. A Draco tile is held to the
//! uncompressed tile with the same cloud.

#![cfg(all(feature = "tiles", feature = "fs"))]
#![allow(
    clippy::needless_range_loop,
    reason = "a component's index reads plainer than a zip of three lists"
)]

use std::path::{Path, PathBuf};

use polysmelt::document::Document;
use serde_json::Value;

type M4 = [[f64; 4]; 4];

const IDENTITY: M4 = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];

fn mul(a: &M4, b: &M4) -> M4 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
}

fn fnv(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut h = 0xCBF2_9CE4_8422_2325_u64;
    for b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
    }
    format!("{h:016x}")
}

fn floats(v: &[f32]) -> impl Iterator<Item = u8> + '_ {
    v.iter().flat_map(|x| x.to_le_bytes())
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn index(v: &Value) -> Option<usize> {
    v.as_u64().and_then(|i| usize::try_from(i).ok())
}

/// A node's own transform.
fn local(node: &Value) -> M4 {
    let m = numbers(&node["matrix"]);
    if m.len() == 16 {
        return std::array::from_fn(|r| std::array::from_fn(|c| m[c * 4 + r]));
    }
    let or = |key: &str, default: &[f64]| {
        let v = numbers(&node[key]);
        if v.len() == default.len() { v } else { default.to_vec() }
    };
    trs(
        &or("translation", &[0.0; 3]),
        &or("rotation", &[0.0, 0.0, 0.0, 1.0]),
        &or("scale", &[1.0; 3]),
    )
}

/// Translation, rotation (x, y, z, w) and scale as a matrix.
fn trs(t: &[f64], q: &[f64], s: &[f64]) -> M4 {
    let len = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    let [x, y, z, w] = [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut m = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = r[i][j] * s[j];
        }
        m[i][3] = t[i];
    }
    m
}

/// Each node the default scene draws from, with its transform from the scene.
fn placed_nodes(doc: &Document) -> Vec<(usize, M4)> {
    let json = doc.json();
    let nodes = json["nodes"].as_array().cloned().unwrap_or_default();
    let scene = json.get("scene").and_then(index).unwrap_or(0);
    let mut stack: Vec<(usize, M4)> = json["scenes"][scene]["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(index)
        .map(|i| (i, IDENTITY))
        .collect();
    let mut out = Vec::new();
    while let Some((i, parent)) = stack.pop() {
        let m = mul(&parent, &local(&nodes[i]));
        out.push((i, m));
        for child in nodes[i]["children"].as_array().into_iter().flatten().filter_map(index) {
            stack.push((child, m));
        }
    }
    out
}

fn read(doc: &Document, accessor: &Value) -> Result<(usize, Vec<f32>), String> {
    let i = index(accessor).ok_or("an attribute without an accessor")?;
    doc.read_floats(i).map_err(|e| format!("accessor {i}: {e}"))
}

/// Every mesh drawn, once for each instance: the mesh and its transform to the scene.
fn placements(doc: &Document) -> Result<Vec<(usize, M4)>, String> {
    let nodes = doc.json()["nodes"].as_array().cloned().unwrap_or_default();
    let mut out = Vec::new();
    for (i, world) in placed_nodes(doc) {
        let Some(mesh) = nodes[i].get("mesh").and_then(index) else {
            continue;
        };
        let Some(attributes) = nodes[i].pointer("/extensions/EXT_mesh_gpu_instancing/attributes") else {
            out.push((mesh, world));
            continue;
        };
        let column = |name: &str| attributes.get(name).map(|a| read(doc, a)).transpose();
        let (t, r, s) = (column("TRANSLATION")?, column("ROTATION")?, column("SCALE")?);
        let count = t.as_ref().map_or(0, |(_, v)| v.len() / 3);
        for k in 0..count {
            let part = |c: &Option<(usize, Vec<f32>)>, width: usize, default: &[f64]| -> Vec<f64> {
                c.as_ref().map_or(default.to_vec(), |(_, v)| {
                    v[k * width..(k + 1) * width].iter().map(|&x| f64::from(x)).collect()
                })
            };
            let instance = trs(
                &part(&t, 3, &[0.0; 3]),
                &part(&r, 4, &[0.0, 0.0, 0.0, 1.0]),
                &part(&s, 3, &[1.0; 3]),
            );
            out.push((mesh, mul(&world, &instance)));
        }
    }
    Ok(out)
}

/// Whether two JSON values are the same, numbers to within a part in a billion (the reference rounds its own).
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN));
            (x - y).abs() <= 1e-9 * y.abs().max(1.0)
        }
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, x)| y.get(k).is_some_and(|y| same(x, y)))
        }
        _ => a == b,
    }
}

/// The node over a tile's content: the one whose extras say it's a tile of `kind`.
fn tile_node<'d>(doc: &'d Document, kind: &str) -> Option<&'d Value> {
    doc.json()["nodes"]
        .as_array()?
        .iter()
        .find(|n| n.pointer("/extras/tile").and_then(Value::as_str) == Some(kind) || n["name"] == kind)
}

/// The batch table and its length in a tile node's extras against the reference's.
fn batch_table(node: Option<&Value>, reference: &Value) -> Result<(), String> {
    let length = &reference["batchLength"];
    let table = &reference["batchTable"];
    let empty = table.as_object().is_none_or(serde_json::Map::is_empty);
    if length.as_u64().unwrap_or(0) == 0 && empty {
        return Ok(());
    }
    let extras = node.map(|n| &n["extras"]).ok_or("no node holds the batch table")?;
    if length.is_u64() && &extras["batchLength"] != length {
        return Err(format!(
            "batchLength is {}, the reference's {length}",
            extras["batchLength"]
        ));
    }
    let got = &extras["batchTable"];
    if empty && got.is_null() {
        return Ok(());
    }
    if !same(got, table) {
        return Err(format!("the batch table is {got}, the reference's {table}"));
    }
    Ok(())
}

/// The translation of `node` against the one the file gives, to within a billionth of a meter: single
/// precision would be out by a tenth of one.
fn translation(node: Option<&Value>, want: &Value) -> Result<[f64; 3], String> {
    let got = node.map(|n| numbers(&n["translation"])).unwrap_or_default();
    let got: [f64; 3] = if got.len() == 3 {
        [got[0], got[1], got[2]]
    } else {
        [0.0; 3]
    };
    if !want.is_null() {
        let want = numbers(want);
        if (0..3).any(|k| (got[k] - want[k]).abs() > 1e-9) {
            return Err(format!("the node is translated by {got:?}, the reference's {want:?}"));
        }
    }
    Ok(got)
}

/// A point cloud's node, its translation, and each attribute of its one primitive: its width and values, the
/// values every point shares (kept in the mesh's extras) spread over the points.
struct Cloud {
    node: Value,
    origin: [f64; 3],
    count: usize,
    attributes: std::collections::BTreeMap<String, (usize, Vec<f32>)>,
}

fn cloud(doc: &Document) -> Result<Cloud, String> {
    let json = doc.json();
    let node = json["nodes"][0].clone();
    let origin = translation(Some(&node), &Value::Null)?;
    let mesh = &json["meshes"][0];
    let mut attributes = std::collections::BTreeMap::new();
    for (name, accessor) in mesh["primitives"][0]["attributes"].as_object().ok_or("no primitive")? {
        attributes.insert(name.clone(), read(doc, accessor)?);
    }
    let count = attributes.get("POSITION").map_or(0, |(_, v)| v.len() / 3);
    for (name, value) in mesh
        .pointer("/extras/constant")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "single-precision values, widened in the JSON"
        )]
        let one: Vec<f32> = match value {
            Value::Array(list) => list.iter().filter_map(Value::as_f64).map(|x| x as f32).collect(),
            other => vec![other.as_f64().unwrap_or(f64::NAN) as f32],
        };
        attributes.insert(name.clone(), (one.len(), one.repeat(count)));
    }
    Ok(Cloud {
        node,
        origin,
        count,
        attributes,
    })
}

fn linear_to_srgb(l: f64) -> f64 {
    if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

fn points(doc: &Document, reference: &Value, root: &Path) -> Result<(), String> {
    if reference["points"] == 0 {
        return match doc.json().get("meshes") {
            None => Ok(()),
            Some(_) => Err("draws something, though the file has no points".into()),
        };
    }
    let c = cloud(doc)?;
    let (_, positions) = c.attributes.get("POSITION").ok_or("no positions")?;
    if reference["points"].as_u64().is_some_and(|n| n != c.count as u64) {
        return Err(format!("{} points, the reference's {}", c.count, reference["points"]));
    }
    if let Some(want) = reference["local"].as_str() {
        // The file says where its positions are measured from: the node is there, and they're as stored.
        translation(Some(&c.node), &reference["translation"])?;
        let got = fnv(floats(positions));
        if got != want {
            return Err(format!("positions hash to {got}, the file's to {want}"));
        }
    }
    if let Some(want) = reference["absolute"].as_str() {
        // Stored absolute: wherever they're measured from now, they round back to the file's values.
        #[allow(clippy::cast_possible_truncation, reason = "back to the file's single precision")]
        let back: Vec<f32> = positions
            .iter()
            .enumerate()
            .map(|(i, &p)| match c.origin[i % 3] {
                // Nothing to add: the value as it is, a negative zero too.
                0.0 => p,
                origin => (origin + f64::from(p)) as f32,
            })
            .collect();
        let got = fnv(floats(&back));
        if got != want {
            return Err(format!(
                "positions with the node's translation hash to {got}, the file's to {want}"
            ));
        }
    }
    if let Some(colors) = reference.get("colors") {
        let (width, values) = c.attributes.get("COLOR_0").ok_or("no colors")?;
        if *width != 4 {
            return Err(format!("colors are {width} wide"));
        }
        let levels = numbers(&colors["levels"]);
        // Linear in the document; the file's are sRGB levels, and its alpha a byte.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a level in 0..=255"
        )]
        let stored = values.iter().enumerate().map(|(i, &v)| {
            let k = i % 4;
            let v = f64::from(v);
            ((if k < 3 { linear_to_srgb(v) } else { v }) * levels[k]).round() as u8
        });
        let got = fnv(stored);
        if colors["hash"] != got.as_str() {
            return Err(format!("colors hash to {got}, the file's to {}", colors["hash"]));
        }
    }
    if let Some(normals) = reference.get("normals") {
        let (_, values) = c.attributes.get("NORMAL").ok_or("no normals")?;
        let sums = numbers(&normals["sums"]);
        for k in 0..3 {
            let sum: f64 = values.iter().skip(k).step_by(3).map(|&v| f64::from(v)).sum();
            #[allow(clippy::cast_precision_loss, reason = "a tolerance")]
            if (sum - sums[k]).abs() > 1e-6 * c.count as f64 {
                return Err(format!(
                    "normals' component {k} sums to {sum}, the reference's to {}",
                    sums[k]
                ));
            }
        }
        for sample in normals["samples"].as_array().into_iter().flatten() {
            let s = numbers(sample);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a point's index")]
            let i = s[0] as usize;
            if (0..3).any(|k| (f64::from(values[i * 3 + k]) - s[1 + k]).abs() > 1e-6) {
                return Err(format!(
                    "normal {i} is {:?}, the reference's {:?}",
                    &values[i * 3..i * 3 + 3],
                    &s[1..]
                ));
            }
        }
    }
    for (name, want) in reference["attributes"].as_object().into_iter().flatten() {
        let (width, values) = c.attributes.get(name).ok_or_else(|| format!("no attribute {name}"))?;
        let got = fnv(floats(values));
        if *width != 1 || want != got.as_str() {
            return Err(format!("{name} ({width} wide) hashes to {got}, the file's to {want}"));
        }
    }
    batch_table(Some(&c.node), reference)?;
    if let Some(twin) = reference["twin"].as_str() {
        draco(&c, reference, &root.join(twin))?;
    }
    Ok(())
}

/// A Draco tile against the uncompressed tile that holds the same cloud, point for point: positions and
/// normals to within their quantization, colors and batch ids exactly.
fn draco(c: &Cloud, reference: &Value, twin: &Path) -> Result<(), String> {
    let model = model_in(twin).ok_or("the twin has no model")?;
    let other = cloud(&polysmelt::open(&model).map_err(|e| format!("the twin doesn't load: {e}"))?)?;
    if other.count != c.count {
        return Err(format!("{} points, its twin's {}", c.count, other.count));
    }
    let held: Vec<&str> = reference["draco"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let far = |name: &str, tolerance: f64, origins: Option<([f64; 3], [f64; 3])>| -> Result<(), String> {
        let (Some((width, a)), Some((_, b))) = (c.attributes.get(name), other.attributes.get(name)) else {
            return Err(format!("{name} isn't in both"));
        };
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            let (ox, oy) = origins.map_or((0.0, 0.0), |(p, q)| (p[i % 3], q[i % 3]));
            let d = (ox + f64::from(x) - oy - f64::from(y)).abs();
            if d > tolerance {
                return Err(format!(
                    "{name} of point {} is {d} from its twin's, more than {tolerance}",
                    i / width
                ));
            }
        }
        Ok(())
    };
    far(
        "POSITION",
        reference["tolerance"]["positions"].as_f64().unwrap_or(0.0),
        Some((c.origin, other.origin)),
    )?;
    for (semantic, name, tolerance) in [
        (
            "NORMAL",
            "NORMAL",
            reference["tolerance"]["normals"].as_f64().unwrap_or(0.0),
        ),
        ("RGB", "COLOR_0", 0.0),
        ("BATCH_ID", "_BATCHID", 0.0),
    ] {
        // What the twin has too: its colors may be the Draco tile's alone.
        let _ = &held;
        if other.attributes.contains_key(name) && c.attributes.contains_key(name) {
            far(name, tolerance, None)?;
        } else if held.contains(&semantic) && !c.attributes.contains_key(name) {
            return Err(format!("no {name}, which the Draco data holds as {semantic}"));
        }
    }
    batch_table(
        Some(&c.node),
        &serde_json::json!({
            "batchLength": other.node["extras"]["batchLength"],
            "batchTable": other.node["extras"]["batchTable"],
        }),
    )
}

/// Each expected matrix (the rows of a 3x4, mesh to scene) against a placement not yet matched: the linear
/// part to within `turn` of its size, the translation to within `reach`.
fn matched(
    placed: &mut Vec<(usize, M4)>,
    mesh: Option<usize>,
    matrices: &Value,
    (turn, reach): (f64, f64),
) -> Result<(), String> {
    for (k, want) in matrices.as_array().into_iter().flatten().enumerate() {
        let want = numbers(want);
        let size = want
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 != 3)
            .fold(1.0_f64, |m, (_, v)| m.max(v.abs()));
        let close = |m: &M4| {
            (0..12).all(|i| {
                let tolerance = if i % 4 == 3 { reach } else { turn * size };
                (m[i / 4][i % 4] - want[i]).abs() <= tolerance
            })
        };
        let found = placed
            .iter()
            .position(|(at, m)| mesh.is_none_or(|mesh| mesh == *at) && close(m));
        match found {
            Some(i) => {
                placed.swap_remove(i);
            }
            None => return Err(format!("no mesh is placed by instance {k}'s transform, {want:?}")),
        }
    }
    Ok(())
}

/// Every instance entry of a reference: an i3dm's own, or those of the tiles in a composite.
fn instance_entries<'r>(reference: &'r Value, out: &mut Vec<&'r Value>) {
    out.extend(reference["instances"].as_array().into_iter().flatten());
    for tile in reference["tiles"].as_array().into_iter().flatten() {
        instance_entries(tile, out);
    }
}

fn check(doc: &Document, reference: &Value, root: &Path) -> Result<(), String> {
    let kind = reference["kind"].as_str().unwrap_or("");
    // How far a placement may be from the reference's: its linear part (to within single precision, unless
    // the reference says its vectors are looser than that) and its translation.
    let within = (
        reference["turn"].as_f64().unwrap_or(2e-6),
        reference["reach"].as_f64().unwrap_or(1e-3),
    );
    match kind {
        "pnts" => points(doc, reference, root),
        "b3dm" => {
            let node = tile_node(doc, "b3dm");
            translation(node, &reference["translation"])?;
            batch_table(node, reference)?;
            let mut stored = Vec::new();
            for mesh in doc.json()["meshes"].as_array().into_iter().flatten() {
                for p in mesh["primitives"].as_array().into_iter().flatten() {
                    stored.extend(read(doc, &p["attributes"]["POSITION"])?.1);
                }
            }
            let got = fnv(floats(&stored));
            if reference["positions"] != got.as_str() {
                return Err(format!(
                    "positions hash to {got}, the glTF's to {}",
                    reference["positions"]
                ));
            }
            Ok(())
        }
        "i3dm" | "cmpt" => {
            let mut entries = Vec::new();
            instance_entries(reference, &mut entries);
            let mut placed = placements(doc)?;
            for entry in &entries {
                // A composite's meshes are renumbered as its tiles are joined.
                let mesh = entry["mesh"]
                    .as_u64()
                    .and_then(|m| usize::try_from(m).ok())
                    .filter(|_| kind == "i3dm");
                matched(&mut placed, mesh, &entry["matrices"], within)?;
            }
            if kind == "i3dm" {
                if !placed.is_empty() {
                    return Err(format!(
                        "{} meshes placed that the reference doesn't have",
                        placed.len()
                    ));
                }
                let node = tile_node(doc, "i3dm");
                batch_table(node, reference)?;
                if let Some(ids) = reference.get("batchIds") {
                    let attributes = doc.json()["nodes"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .find_map(|n| n.pointer("/extensions/EXT_mesh_gpu_instancing/attributes/_BATCHID"))
                        .ok_or("no _BATCHID instance attribute")?;
                    let got: Vec<f64> = read(doc, attributes)?.1.iter().map(|&x| f64::from(x)).collect();
                    if got != numbers(ids) {
                        return Err(format!("batch ids are {got:?}, the file's {ids}"));
                    }
                }
            }
            Ok(())
        }
        "tileset" => {
            let json = doc.json();
            let nodes = json["nodes"].as_array().cloned().unwrap_or_default();
            let mut named: Vec<(usize, M4)> = placed_nodes(doc);
            for (k, want) in reference["placements"].as_array().into_iter().flatten().enumerate() {
                let m = numbers(&want["matrix"]);
                let found = named.iter().position(|(i, world)| {
                    nodes[*i]["name"] == want["name"]
                        && (0..16).all(|e| (world[e % 4][e / 4] - m[e]).abs() <= 1e-9 * m[e].abs().max(1.0))
                });
                match found {
                    Some(i) => {
                        named.swap_remove(i);
                    }
                    None => {
                        return Err(format!(
                            "no node places tile {k}, {}, as the reference does",
                            want["name"]
                        ));
                    }
                }
            }
            Ok(())
        }
        other => Err(format!("a reference of kind {other:?}")),
    }
}

/// The model of the case in `dir`: the file its expect.json names, or its one tile.
fn model_in(dir: &Path) -> Option<PathBuf> {
    let expect: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("expect.json")).ok()?).ok()?;
    if let Some(file) = expect["file"].as_str() {
        return Some(dir.join(file));
    }
    std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).find(|p| {
        let ext = p.extension().unwrap_or_default().to_string_lossy().to_lowercase();
        matches!(ext.as_str(), "b3dm" | "i3dm" | "pnts" | "cmpt")
    })
}

#[test]
fn reads_what_the_reference_decoder_reads() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/tiles");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("reference.json").is_file())
        .collect();
    dirs.sort();
    assert!(dirs.len() >= 50, "only {} cases with a reference", dirs.len());
    let mut failures = Vec::new();
    let mut ran = 0;
    for dir in &dirs {
        let reference: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap()).unwrap();
        if reference.get("draco").is_some() && !cfg!(feature = "draco") {
            continue;
        }
        ran += 1;
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let result = model_in(dir)
            .ok_or_else(|| "no model".to_owned())
            .and_then(|model| polysmelt::open(&model).map_err(|e| format!("doesn't load: {e}")))
            .and_then(|doc| check(&doc, &reference, &root));
        if let Err(why) = result {
            failures.push(format!("{name}: {why}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {ran} cases differ from the reference decoder:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
