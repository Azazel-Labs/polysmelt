//! VTK files against VTK itself. Beside each case's model under `tests/corpus/vtk/`, `reference.json` holds
//! what VTK's own readers make of the file, worked out by `tools/corpus/vtkfile.py`: 64-bit FNV-1a hashes of the
//! points drawn (in order, bit for bit), of the triangles, lines and points drawn (as indices into them), and of
//! each attribute's values. Where cell data splits vertices, the triangles are hashed as corner positions with
//! their cells' values instead. What Polysmelt reads must hash the same.
//!
//! A file without normals gets the default ones, which add vertices after VTK's points: copies of points where
//! an edge is hard. They are told by being at the end and the same as an earlier vertex in everything but their
//! normal, and each is taken as the point it copies.

#![cfg(all(feature = "vtk", feature = "fs"))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use polysmelt::document::Document;
use serde_json::Value;

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

fn ints(v: &[u32]) -> impl Iterator<Item = u8> + '_ {
    v.iter().flat_map(|x| x.to_le_bytes())
}

/// What a document draws, its primitives' vertices one after another.
#[derive(Default)]
struct Drawn {
    positions: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    lines: Vec<[u32; 2]>,
    verts: Vec<u32>,
    /// Each attribute's width and values.
    attributes: BTreeMap<String, (usize, Vec<f32>)>,
    /// Whether the default normals added vertices, which are left out here.
    copied: bool,
}

fn drawn(doc: &Document, reference_has_normals: bool) -> Drawn {
    let mut out = Drawn::default();
    // Each vertex the default normals added, and the point it is a copy of.
    let mut copies: std::collections::HashMap<(u64, u32), u32> = std::collections::HashMap::new();
    // Primitives of one piece share their attributes: each set of positions is taken once.
    let mut blocks: Vec<(u64, u32)> = Vec::new();
    let prims = doc.json()["meshes"][0]["primitives"].as_array().unwrap().clone();
    for p in &prims {
        let position = p["attributes"]["POSITION"].as_u64().unwrap();
        let offset = if let Some(&(_, offset)) = blocks.iter().find(|(a, _)| *a == position) {
            offset
        } else {
            let offset = u32::try_from(out.positions.len()).unwrap();
            // The block's vertices, each as the bits of everything it holds but its normal.
            let mut read: Vec<(String, usize, Vec<f32>)> = Vec::new();
            for (name, accessor) in p["attributes"].as_object().unwrap() {
                let (width, values) = doc
                    .read_floats(usize::try_from(accessor.as_u64().unwrap()).unwrap())
                    .unwrap();
                read.push((name.clone(), width, values));
            }
            let count = read.iter().find(|r| r.0 == "POSITION").map_or(0, |r| r.2.len() / 3);
            let key = |v: usize| -> Vec<u32> {
                let kept = read.iter().filter(|r| r.0 != "NORMAL");
                kept.flat_map(|(_, w, values)| values[v * w..(v + 1) * w].iter().map(|x| x.to_bits()))
                    .collect()
            };
            // The copies the default normals added: at the end, each the same as an earlier vertex.
            let mut firsts: std::collections::HashMap<Vec<u32>, usize> = std::collections::HashMap::new();
            let generated = p["attributes"].get("NORMAL").is_some() && !reference_has_normals;
            let mut own = count;
            if generated {
                for v in 0..count {
                    firsts.entry(key(v)).or_insert(v);
                }
                while own > 0 && firsts[&key(own - 1)] < own - 1 {
                    own -= 1;
                }
                for v in own..count {
                    let to = u32::try_from(firsts[&key(v)]).unwrap() + offset;
                    copies.insert((position, u32::try_from(v).unwrap()), to);
                    out.copied = true;
                }
            }
            for (name, width, values) in read {
                let values = &values[..own * width];
                if name == "POSITION" {
                    assert_eq!(width, 3);
                    out.positions.extend(values.as_chunks::<3>().0);
                } else if name != "NORMAL" || !generated {
                    let entry = out.attributes.entry(name.clone()).or_insert((width, Vec::new()));
                    assert_eq!(entry.0, width, "{name} changes width between pieces");
                    entry.1.extend(values);
                }
            }
            blocks.push((position, offset));
            offset
        };
        let count = u32::try_from(out.positions.len()).unwrap() - offset;
        let indices: Vec<u32> = match p.get("indices").and_then(Value::as_u64) {
            Some(a) => doc.read_indices(usize::try_from(a).unwrap()).unwrap(),
            None => (0..count).collect(),
        };
        let indices: Vec<u32> = indices
            .iter()
            .map(|i| copies.get(&(position, *i)).copied().unwrap_or(i + offset))
            .collect();
        match p.get("mode").and_then(Value::as_u64).unwrap_or(4) {
            4 => out.tris.extend(indices.as_chunks::<3>().0),
            1 => out.lines.extend(indices.as_chunks::<2>().0),
            0 => out.verts.extend(indices),
            mode => panic!("a primitive of mode {mode}"),
        }
    }
    out
}

/// A triangle's corners turned so the least comes first, its winding kept.
fn rotated<T: PartialOrd + Copy>(t: [T; 3]) -> [T; 3] {
    let mut k = 0;
    for i in 1..3 {
        if t[i] < t[k] {
            k = i;
        }
    }
    [t[k], t[(k + 1) % 3], t[(k + 2) % 3]]
}

fn check(dir: &Path, model: &Path, reference: &Value) -> Result<(), String> {
    let doc = polysmelt::open(model).map_err(|e| format!("doesn't load: {e}"))?;
    let d = drawn(&doc, reference["attributes"].get("NORMAL").is_some());
    let same = |what: &str, got: String, want: &Value| match want.as_str() {
        Some(want) if want != got => Err(format!("{what} hash to {got}, VTK's to {want}")),
        _ => Ok(()),
    };
    same("points", fnv(floats(d.positions.as_flattened())), &reference["points"])?;
    let mut tris: Vec<[u32; 3]> = d.tris.iter().map(|&t| rotated(t)).collect();
    tris.sort_unstable();
    // Where vertices were added and two of VTK's points are alike, which one a copy stands for can't be told:
    // the triangles are then held to VTK's by their corners' positions alone, below.
    if !(d.copied && reference["soup"].is_string()) {
        same("triangles", fnv(ints(tris.as_flattened())), &reference["triangles"])?;
    }
    if reference["points"].is_string() {
        let mut lines = d.lines.clone();
        lines.sort_unstable();
        same("lines", fnv(ints(lines.as_flattened())), &reference["lines"])?;
        let mut verts = d.verts.clone();
        verts.sort_unstable();
        same("verts", fnv(ints(&verts)), &reference["verts"])?;
    }
    for (name, want) in reference["attributes"].as_object().into_iter().flatten() {
        let (width, values) = d.attributes.get(name).ok_or_else(|| format!("no attribute {name}"))?;
        if want["width"].as_u64() != Some(*width as u64) {
            return Err(format!("{name} is {width} wide, VTK's {}", want["width"]));
        }
        let list = |key: &str| -> Option<Vec<f64>> {
            Some(want.get(key)?.as_array()?.iter().filter_map(Value::as_f64).collect())
        };
        if want.get("hash").is_some() {
            same(name, fnv(floats(values)), &want["hash"])?;
        } else if let Some(theirs) = list("values") {
            // Worked out in floating point on both sides.
            let close = theirs.len() == values.len()
                && theirs
                    .iter()
                    .zip(values)
                    .all(|(a, &b)| (a - f64::from(b)).abs() <= 1e-5);
            if !close {
                return Err(format!("{name} is {values:?}, VTK's {theirs:?}"));
            }
        } else {
            // Too many to list: each component's sum and sum of squares.
            let (sums, squares) = (list("sums").unwrap(), list("squares").unwrap());
            #[allow(clippy::cast_precision_loss, reason = "a tolerance")]
            let tolerance = 1e-6 * (values.len() as f64 + 1.0);
            for k in 0..*width {
                let column = values.iter().skip(k).step_by(*width).map(|&v| f64::from(v));
                let (sum, square) = column.fold((0.0, 0.0), |(s, q), v| (s + v, q + v * v));
                if (sum - sums[k]).abs() > tolerance || (square - squares[k]).abs() > tolerance {
                    return Err(format!(
                        "{name} component {k} sums to {sum} (squares {square}), VTK's to {} ({})",
                        sums[k], squares[k]
                    ));
                }
            }
        }
    }
    if reference["soup"].is_string() {
        let names: Vec<&str> = reference["soup_attributes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        let mut records: Vec<Vec<u8>> = Vec::new();
        for t in &d.tris {
            // The corners by position, the least first; the vertex there carries the cell's values.
            let corners = t.map(|i| (d.positions[i as usize], i));
            let turned = rotated(corners.map(|(p, _)| p));
            let least = turned[0].map(f32::to_bits);
            let first = corners.iter().find(|(p, _)| p.map(f32::to_bits) == least).unwrap().1 as usize;
            let mut record: Vec<u8> = floats(turned.as_flattened()).collect();
            for name in &names {
                let (width, values) = d.attributes.get(*name).ok_or_else(|| format!("no attribute {name}"))?;
                let at = |v: usize| &values[v * width..(v + 1) * width];
                if t.iter().any(|&i| at(i as usize) != at(first)) {
                    return Err(format!("a triangle's corners differ in {name}, which its cell gives"));
                }
                record.extend(floats(at(first)));
            }
            records.push(record);
        }
        records.sort_unstable();
        same(
            "triangles with their cells' values",
            fnv(records.concat()),
            &reference["soup"],
        )?;
    }
    let _ = dir;
    Ok(())
}

#[test]
fn reads_what_vtk_reads() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/vtk");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("reference.json").is_file())
        .collect();
    dirs.sort();
    assert!(dirs.len() >= 60, "only {} cases with a reference", dirs.len());
    let mut failures = Vec::new();
    for dir in &dirs {
        let reference: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap()).unwrap();
        let model = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| {
                let ext = p.extension().unwrap_or_default().to_string_lossy().to_lowercase();
                matches!(ext.as_str(), "vtk" | "vtp" | "vtu")
            })
            .unwrap();
        if let Err(why) = check(dir, &model, &reference) {
            failures.push(format!("{}: {why}", dir.file_name().unwrap().to_string_lossy()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from VTK:\n{}",
        failures.len(),
        dirs.len(),
        failures.join("\n")
    );
}
