//! Gmsh and Medit files against other readers. Beside each case's model under `tests/corpus/msh/` and
//! `tests/corpus/medit/`, `reference.json` holds what Gmsh, meshio and VTK make of the file, worked out by
//! `tools/corpus/msh.py` and `tools/corpus/medit.py`: for each mesh (a group of the file's elements), a 64-bit
//! FNV-1a hash of its soup, which is every triangle, line and point drawn as the positions of its corners with
//! each attribute's values there, and where nothing splits vertices, a hash of the points drawn (in the file's
//! order, bit for bit). What Polysmelt reads must hash the same, mesh by mesh.
//!
//! These files have no normals, so their meshes get the default ones, which add vertices after the file's own
//! points: copies of points where an edge is hard. The points are the first `vertices` of a mesh's, and each
//! vertex after them must be a copy of one of them in everything but its normal.

#![cfg(all(feature = "fs", any(feature = "msh", feature = "medit")))]

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

/// What one mesh of a document draws.
#[derive(Default)]
struct Drawn {
    positions: Vec<[f32; 3]>,
    /// Each triangle, line and point, as vertices.
    items: Vec<Vec<u32>>,
    /// Each attribute's width and values.
    attributes: BTreeMap<String, (usize, Vec<f32>)>,
}

fn index(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}

fn drawn(doc: &Document, mesh: &Value) -> Drawn {
    let mut out = Drawn::default();
    let prims = mesh["primitives"].as_array().unwrap();
    for (k, p) in prims.iter().enumerate() {
        assert_eq!(
            p["attributes"], prims[0]["attributes"],
            "a mesh's primitives share its vertices"
        );
        if k == 0 {
            for (name, accessor) in p["attributes"].as_object().unwrap() {
                let (width, values) = doc.read_floats(index(accessor)).unwrap();
                if name == "POSITION" {
                    out.positions = values.as_chunks::<3>().0.to_vec();
                } else {
                    out.attributes.insert(name.clone(), (width, values));
                }
            }
        }
        let indices: Vec<u32> = match p.get("indices") {
            Some(a) => doc.read_indices(index(a)).unwrap(),
            None => (0..u32::try_from(out.positions.len()).unwrap()).collect(),
        };
        let per = match p.get("mode").and_then(Value::as_u64).unwrap_or(4) {
            4 => 3,
            1 => 2,
            0 => 1,
            mode => panic!("a primitive of mode {mode}"),
        };
        out.items.extend(indices.chunks_exact(per).map(<[u32]>::to_vec));
    }
    out
}

fn check_mesh(doc: &Document, mesh: &Value, want: &Value) -> Result<(), String> {
    let d = drawn(doc, mesh);
    if !want["extras"].is_null() && mesh.get("extras") != Some(&want["extras"]) {
        return Err(format!(
            "extras are {:?}, expected {}",
            mesh.get("extras"),
            want["extras"]
        ));
    }
    for (key, per) in [("triangles", 3), ("lines", 2), ("verts", 1)] {
        let n = d.items.iter().filter(|i| i.len() == per).count();
        if want[key].as_u64() != Some(n as u64) {
            return Err(format!("{n} {key}, expected {}", want[key]));
        }
    }
    let names: Vec<&str> = want["soup_attributes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let have: Vec<&str> = d
        .attributes
        .keys()
        .map(String::as_str)
        .filter(|k| *k != "NORMAL")
        .collect();
    if have != names {
        return Err(format!("attributes are {have:?}, expected {names:?}"));
    }
    // The soup: each item's corners turned so the least comes first, their order kept.
    let mut records: Vec<Vec<u8>> = Vec::new();
    for item in &d.items {
        let mut corners: Vec<Vec<u8>> = Vec::new();
        for &v in item {
            let mut record: Vec<u8> = floats(&d.positions[v as usize]).collect();
            for name in &names {
                let (width, values) = &d.attributes[*name];
                record.extend(floats(&values[v as usize * width..(v as usize + 1) * width]));
            }
            corners.push(record);
        }
        let least = (0..corners.len()).min_by_key(|&k| &corners[k]).unwrap();
        corners.rotate_left(least);
        let mut record = vec![u8::try_from(item.len()).unwrap()];
        record.extend(corners.concat());
        records.push(record);
    }
    records.sort_unstable();
    let soup = fnv(records.concat());
    if want["soup"].as_str() != Some(&soup) {
        return Err(format!("what's drawn hashes to {soup}, expected {}", want["soup"]));
    }
    // The file's own points, and how many vertices there are in all: more when the default normals added copies.
    let all = d.positions.len();
    let mut own = all;
    if let Some(points) = want["points"].as_str() {
        own = usize::try_from(want["vertices"].as_u64().unwrap_or(0)).unwrap();
        if own > all {
            return Err(format!("{all} vertices, expected {own}"));
        }
        let got = fnv(floats(d.positions[..own].as_flattened()));
        if got != points {
            return Err(format!("points hash to {got}, expected {points}"));
        }
        // What a vertex holds, its normal aside.
        let held = |v: usize| -> Vec<u32> {
            let others = d.attributes.iter().filter(|(name, _)| *name != "NORMAL");
            let values = others.flat_map(|(_, (width, values))| &values[v * width..(v + 1) * width]);
            d.positions[v].iter().chain(values).map(|x| x.to_bits()).collect()
        };
        let points: std::collections::BTreeSet<Vec<u32>> = (0..own).map(held).collect();
        if let Some(v) = (own..all).find(|&v| !points.contains(&held(v))) {
            return Err(format!(
                "vertex {v}, past the file's {own} points, is a copy of none of them"
            ));
        }
    }
    for (name, want) in want["approx"].as_object().into_iter().flatten() {
        let (width, values) = d.attributes.get(name).ok_or_else(|| format!("no attribute {name}"))?;
        let theirs: Vec<f64> = want["values"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_f64)
            .collect();
        // Worked out in floating point on both sides.
        let values = &values[..(own * width).min(values.len())];
        let close = want["width"].as_u64() == Some(*width as u64)
            && theirs.len() == values.len()
            && theirs
                .iter()
                .zip(values)
                .all(|(a, &b)| (a - f64::from(b)).abs() <= 1e-5);
        if !close {
            return Err(format!("{name} is {values:?}, expected {theirs:?}"));
        }
    }
    Ok(())
}

fn check(model: &Path, reference: &Value) -> Result<(), String> {
    let doc = polysmelt::open(model).map_err(|e| format!("doesn't load: {e}"))?;
    let meshes = doc.json()["meshes"].as_array().cloned().unwrap_or_default();
    let wanted = reference["meshes"].as_array().unwrap();
    if meshes.len() != wanted.len() {
        return Err(format!("{} meshes, expected {}", meshes.len(), wanted.len()));
    }
    for want in wanted {
        let name = want.get("name").filter(|n| !n.is_null());
        let mesh = meshes
            .iter()
            .find(|m| m.get("name") == name)
            .ok_or_else(|| format!("no mesh named {name:?}"))?;
        check_mesh(&doc, mesh, want).map_err(|why| format!("mesh {name:?}: {why}"))?;
    }
    Ok(())
}

fn reads_what_others_read(format: &str, extensions: &[&str], least: usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/corpus")
        .join(format);
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("reference.json").is_file())
        .collect();
    dirs.sort();
    assert!(dirs.len() >= least, "only {} cases with a reference", dirs.len());
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
                extensions.contains(&ext.as_str())
            })
            .unwrap();
        if let Err(why) = check(&model, &reference) {
            failures.push(format!("{}: {why}", dir.file_name().unwrap().to_string_lossy()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        dirs.len(),
        failures.join("\n")
    );
}

#[test]
#[cfg(feature = "msh")]
fn gmsh_files_read_as_gmsh_reads_them() {
    reads_what_others_read("msh", &["msh"], 60);
}

#[test]
#[cfg(feature = "medit")]
fn medit_files_read_as_meshio_reads_them() {
    reads_what_others_read("medit", &["mesh", "meshb"], 30);
}

/// Every file with the extension `mesh` under a format's corpus, with its bytes.
#[cfg(feature = "medit")]
fn mesh_files(format: &str) -> Vec<(PathBuf, Vec<u8>)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/corpus")
        .join(format);
    let mut out = Vec::new();
    for dir in std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .filter(|d| d.path().is_dir())
    {
        for file in std::fs::read_dir(dir.path()).unwrap().flatten().map(|f| f.path()) {
            if file.extension().is_some_and(|e| e.eq_ignore_ascii_case("mesh")) {
                let bytes = std::fs::read(&file).unwrap();
                out.push((file, bytes));
            }
        }
    }
    assert!(out.len() >= 10, "only {} .mesh files under {format}", out.len());
    out
}

/// `.mesh` is OGRE's extension and Medit's. No OGRE mesh in the corpus is taken for a Medit one, and every
/// Medit one that reads is told apart by its first bytes.
#[test]
#[cfg(feature = "medit")]
fn mesh_files_are_told_apart_by_their_content() {
    use polysmelt::Format;
    for (file, bytes) in mesh_files("ogre") {
        assert_eq!(Format::detect(&bytes), Some(Format::Ogre), "{}", file.display());
    }
    for (file, bytes) in mesh_files("medit") {
        let expect = std::fs::read_to_string(file.with_file_name("expect.json")).unwrap();
        if expect.contains("\"error\"") {
            continue;
        }
        assert_eq!(Format::detect(&bytes), Some(Format::Medit), "{}", file.display());
        let doc = polysmelt::open(&file).unwrap();
        assert_eq!(
            doc.json()["asset"]["extras"]["source"]["format"],
            "Medit",
            "{}",
            file.display()
        );
    }
}
