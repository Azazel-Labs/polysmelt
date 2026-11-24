//! Standards conformance: every case under `tests/corpus/<format>/<case>/` is a model file (with whatever it
//! refers to) and an `expect.json` saying which part of which standard it exercises, where it came from, and what
//! the GLB made from it must hold. Each case is loaded, written as GLB, validated as glTF 2.0, decoded, and checked
//! against its expectations. See `tests/corpus/README.md` for the layout and the expectation keys.
//!
//! Each case is also read from memory, with its folder's files as [`Memory`] assets, and must read the same.

#![cfg(feature = "fs")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "expectations are JSON doubles, compared at the GLB's single precision"
)]

mod glb;

use std::path::{Path, PathBuf};

use polysmelt::assets::{Folder, Memory};
use polysmelt::document::{Document, Keep, Pose};
use polysmelt::images::Location;
use polysmelt::mesh::Mesh;
use polysmelt::{Format, Loader};
use serde_json::{Map, Value};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

/// The cargo feature that reads `ext`, and whether this build has it.
#[allow(clippy::match_like_matches_macro)] // the arms are cfg! values that differ per build
fn enabled(ext: &str) -> bool {
    match ext {
        "stl" => cfg!(feature = "stl"),
        "ply" => cfg!(feature = "ply"),
        "vox" => cfg!(feature = "vox"),
        "obj" => cfg!(feature = "obj"),
        "3ds" => cfg!(feature = "3ds"),
        "fbx" => cfg!(feature = "fbx"),
        "3mf" => cfg!(feature = "3mf"),
        "usd" | "usda" | "usdc" | "usdz" => cfg!(feature = "usd"),
        "gltf" | "glb" | "vrm" | "vrma" => cfg!(feature = "gltf"),
        "skp" => cfg!(feature = "skp"),
        "bvh" => cfg!(feature = "bvh"),
        "dae" | "zae" | "kmz" => cfg!(feature = "dae"),
        "off" => cfg!(feature = "off"),
        "x3d" | "x3dv" | "x3dz" | "wrl" | "wrz" => cfg!(feature = "x3d"),
        "dxf" => cfg!(feature = "dxf"),
        "splat" => cfg!(feature = "splat"),
        "spz" => cfg!(feature = "spz"),
        "pmx" | "pmd" | "vmd" => cfg!(feature = "mmd"),
        "x" => cfg!(feature = "directx"),
        "md2" => cfg!(feature = "md2"),
        "md3" => cfg!(feature = "md3"),
        "md5mesh" | "md5anim" => cfg!(feature = "md5"),
        "iqm" => cfg!(feature = "iqm"),
        "smd" | "vta" => cfg!(feature = "smd"),
        "ms3d" => cfg!(feature = "ms3d"),
        "b3d" => cfg!(feature = "b3d"),
        "asf" | "amc" => cfg!(feature = "asf"),
        "c3d" => cfg!(feature = "c3d"),
        "trc" => cfg!(feature = "trc"),
        "htr" => cfg!(feature = "htr"),
        "las" => cfg!(feature = "las"),
        "laz" => cfg!(feature = "laz"),
        "pcd" => cfg!(feature = "pcd"),
        "xyz" | "pts" | "ptx" | "asc" => cfg!(feature = "xyz"),
        "e57" => cfg!(feature = "e57"),
        "amf" => cfg!(feature = "amf"),
        "lwo" => cfg!(feature = "lwo"),
        "ac" => cfg!(feature = "ac3d"),
        "mesh" | "skeleton" | "xml" => cfg!(feature = "ogre"),
        "flt" => cfg!(feature = "flt"),
        "ogex" => cfg!(feature = "ogex"),
        "irrmesh" | "irr" => cfg!(feature = "irr"),
        "xgl" | "zgl" => cfg!(feature = "xgl"),
        _ => false,
    }
}

/// Whether this build has a cargo feature a case `needs` beyond its format's own, or `png`: the
/// PNG encoder any of the features re-encoding images brings.
fn has_feature(name: &str) -> bool {
    let png = cfg!(any(
        feature = "webp",
        feature = "skp",
        feature = "tga",
        feature = "bmp",
        feature = "gif",
        feature = "tiff",
        feature = "psd",
        feature = "dds"
    ));
    let features = [
        ("png", png),
        ("draco", cfg!(feature = "draco")),
        ("webp", cfg!(feature = "webp")),
        ("tga", cfg!(feature = "tga")),
        ("bmp", cfg!(feature = "bmp")),
        ("gif", cfg!(feature = "gif")),
        ("tiff", cfg!(feature = "tiff")),
        ("psd", cfg!(feature = "psd")),
        ("dds", cfg!(feature = "dds")),
        // Formats a case needs beyond the one its extension names: a scene's mesh file, or `.xml` read by its root.
        ("obj", cfg!(feature = "obj")),
        ("irr", cfg!(feature = "irr")),
        ("xgl", cfg!(feature = "xgl")),
    ];
    features.iter().any(|&(f, on)| on && f == name)
}

/// Every model extension a full build reads, so cases are found in builds without some readers.
const ALL_EXTENSIONS: &[&str] = &[
    "stl", "ply", "vox", "obj", "3ds", "fbx", "3mf", "usd", "usda", "usdc", "usdz", "gltf", "glb", "skp", "bvh", "dae",
    "zae", "kmz", "off", "vrm", "vrma", "x3d", "x3dv", "x3dz", "wrl", "wrz", "dxf", "splat", "spz", "las", "laz",
    "pcd", "xyz", "pts", "ptx", "asc", "e57", "amf", "lwo", "ac", "pmx", "pmd", "vmd", "x", "md2", "md3", "md5mesh",
    "md5anim", "iqm", "smd", "vta", "ms3d", "b3d", "asf", "amc", "c3d", "trc", "htr", "mesh", "skeleton", "xml", "flt",
    "ogex", "irrmesh", "irr", "xgl", "zgl",
];

fn ext_of(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// A case: its folder, model file and expectations.
struct Case {
    name: String,
    dir: PathBuf,
    file: PathBuf,
    expect: Map<String, Value>,
}

fn cases(format: &str) -> Vec<Case> {
    let root = corpus().join(format);
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("no corpus for {format} at {}: {e}", root.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .map(|dir| {
            let name = format!("{format}/{}", dir.file_name().unwrap().to_string_lossy());
            let text = std::fs::read_to_string(dir.join("expect.json"))
                .unwrap_or_else(|e| panic!("{name}: no expect.json: {e}"));
            let expect: Map<String, Value> =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: expect.json doesn't parse: {e}"));
            for key in ["spec", "source"] {
                assert!(
                    expect.get(key).is_some_and(Value::is_string),
                    "{name}: expect.json needs \"{key}\""
                );
            }
            let file = if let Some(f) = expect.get("file").and_then(Value::as_str) {
                dir.join(f)
            } else {
                let models: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .unwrap()
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && ALL_EXTENSIONS.contains(&ext_of(p).as_str()))
                    .collect();
                assert!(
                    models.len() == 1,
                    "{name}: expected one model file, found {models:?}; name it with \"file\""
                );
                models.into_iter().next().unwrap()
            };
            assert!(file.is_file(), "{name}: {} is missing", file.display());
            Case {
                name,
                dir,
                file,
                expect,
            }
        })
        .collect()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).map(|f| f as f32).collect())
        .unwrap_or_default()
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

/// A path under `root`, `/`-separated and relative to it.
fn relative(p: &Path, root: &Path) -> String {
    p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/")
}

/// Reads the case from memory, with every file in its folder as an asset, and checks it reads as it did from its
/// file: the same document and facts, or an error where that failed.
fn same_from_memory(case: &Case, ext: &str, from_file: &polysmelt::Result<Loaded>) -> Result<(), String> {
    let mut memory = Memory::new();
    let mut dirs = vec![case.dir.clone()];
    while let Some(dir) = dirs.pop() {
        for p in std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()) {
            if p.is_dir() {
                dirs.push(p);
            } else {
                memory.insert(&relative(&p, &case.dir), std::fs::read(&p).unwrap());
            }
        }
    }
    let bytes = std::fs::read(&case.file).unwrap();
    let name = relative(&case.file, &case.dir);
    let _ = ext;
    let from_memory = Loader::new().assets(memory).read(bytes, &name).and_then(flattened);
    let glb = |d: &Document| d.to_glb().ok();
    match (from_file, &from_memory) {
        (Ok(a), Ok(b)) if glb(&a.0) == glb(&b.0) && a.0.facts == b.0.facts => Ok(()),
        (Ok(_), Ok(_)) => Err("reads differently from memory".into()),
        (Err(a), Err(b)) if a.kind() == b.kind() => Ok(()),
        (Err(a), Err(b)) => Err(format!("fails from memory with {b:?}, from its file with {a:?}")),
        (Ok(_), Err(err)) => Err(format!("fails to load from memory: {err}")),
        (Err(_), Ok(_)) => Err("loads from memory, though not from its file".into()),
    }
}

/// A case's document and the mesh it flattens to.
type Loaded = (Document, Option<Mesh>);

fn flattened(doc: Document) -> polysmelt::Result<Loaded> {
    let mesh = doc.flatten(Pose::Rest)?;
    Ok((doc, mesh))
}

/// `document`: how many of each thing the glTF JSON holds (`nodes`, `skins`, `animations`, `cameras`, `lights`,
/// `joints` across skins, `morph_targets` across meshes' first primitives, `channels` across animations), and
/// `node_names` and primitive `attributes` that must be there, and `source`, fields the asset's
/// `extras.source` must hold.
fn document(e: &Map<String, Value>, doc: &Document) -> Result<(), String> {
    let Some(want) = e.get("document") else { return Ok(()) };
    let j = doc.json();
    let list = |k: &str| j.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
    let sum = |k: &str, f: &dyn Fn(&Value) -> usize| list(k).iter().map(f).sum::<usize>();
    let len = |v: Option<&Value>| v.and_then(Value::as_array).map_or(0, Vec::len);
    let have = |key: &str| -> usize {
        match key {
            "lights" => len(j
                .get("extensions")
                .and_then(|x| x.pointer("/KHR_lights_punctual/lights"))),
            "joints" => sum("skins", &|s| len(s.get("joints"))),
            "morph_targets" => sum("meshes", &|m| len(m.pointer("/primitives/0/targets"))),
            "channels" => sum("animations", &|a| len(a.get("channels"))),
            k => list(k).len(),
        }
    };
    for (key, v) in want.as_object().into_iter().flatten() {
        if key == "node_names" {
            for name in v.as_array().into_iter().flatten() {
                if !list("nodes").iter().any(|n| n.get("name") == Some(name)) {
                    return Err(format!("no node is named {name}"));
                }
            }
            continue;
        }
        if key == "attributes" {
            let prims: Vec<Value> = list("meshes")
                .iter()
                .flat_map(|m| {
                    m.get("primitives")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();
            for name in v.as_array().into_iter().flatten().filter_map(Value::as_str) {
                if !prims
                    .iter()
                    .any(|p| p.pointer(&format!("/attributes/{name}")).is_some())
                {
                    return Err(format!("no primitive has attribute {name}"));
                }
            }
            continue;
        }
        if key == "source" {
            let source = j.get("asset").and_then(|a| a.pointer("/extras/source"));
            for (k, want) in v.as_object().into_iter().flatten() {
                let got = source.and_then(|s| s.get(k));
                if got != Some(want) {
                    return Err(format!("asset source {k} is {got:?}, expected {want}"));
                }
            }
            continue;
        }
        let n = have(key);
        if Some(n as u64) != v.as_u64() {
            return Err(format!("document has {n} {key}, expected {v}"));
        }
    }
    Ok(())
}

/// `json` (or another `key`): values the glTF JSON holds, by JSON pointer
/// (`/materials/0/extensions/KHR_materials_ior/ior`). Numbers match to within 1e-4 (relative), arrays and objects
/// member by member; `null` means the pointer must lead nowhere.
fn pointers(e: &Map<String, Value>, key: &str, json: &Map<String, Value>) -> Result<(), String> {
    fn same(got: &Value, want: &Value) -> bool {
        match (got, want) {
            (Value::Number(g), Value::Number(w)) => {
                let (g, w) = (g.as_f64().unwrap_or(f64::NAN), w.as_f64().unwrap_or(f64::NAN));
                (g - w).abs() <= 1e-4 * w.abs().max(1.0)
            }
            (Value::Array(g), Value::Array(w)) => g.len() == w.len() && g.iter().zip(w).all(|(g, w)| same(g, w)),
            (Value::Object(g), Value::Object(w)) => w.iter().all(|(k, w)| g.get(k).is_some_and(|g| same(g, w))),
            _ => got == want,
        }
    }
    for (pointer, want) in e.get(key).and_then(Value::as_object).into_iter().flatten() {
        // The document's JSON is a map: its first segment is a key, the rest a pointer into that member.
        let (top, rest) = pointer
            .trim_start_matches('/')
            .split_once('/')
            .unwrap_or((pointer.trim_start_matches('/'), ""));
        let got = json.get(top).and_then(|v| {
            if rest.is_empty() {
                Some(v)
            } else {
                v.pointer(&format!("/{rest}"))
            }
        });
        match (got, want) {
            (None, Value::Null) => {}
            (Some(got), Value::Null) => return Err(format!("{key} {pointer} is {got}, expected nothing")),
            (None, _) => return Err(format!("{key} {pointer} is missing, expected {want}")),
            (Some(got), _) if !same(got, want) => return Err(format!("{key} {pointer} is {got}, expected {want}")),
            _ => {}
        }
    }
    Ok(())
}

/// `poses`: the bounds of what's drawn in animation `animation` at `time` seconds.
fn poses(e: &Map<String, Value>, doc: &Document, tol: f32) -> Result<(), String> {
    for p in e.get("poses").and_then(Value::as_array).into_iter().flatten() {
        let index = p["animation"].as_u64().unwrap_or(0) as usize;
        let time = p["time"].as_f64().unwrap_or(0.0) as f32;
        let mesh = doc
            .flatten(Pose::Animation { index, time })
            .map_err(|err| format!("animation {index} at {time}: {err}"))?
            .ok_or_else(|| format!("animation {index} at {time} draws nothing"))?;
        let (want_min, want_max) = (floats(&p["bounds"]["min"]), floats(&p["bounds"]["max"]));
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in mesh.prims.iter().flat_map(|p| &p.positions) {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
        if !close(&min, &want_min, tol) || !close(&max, &want_max, tol) {
            return Err(format!(
                "animation {index} at {time}: bounds are {min:?}..{max:?}, expected {want_min:?}..{want_max:?}"
            ));
        }
    }
    Ok(())
}

/// `rig`: what `Document::rig` makes of a node, found by name. Each entry names its `node` and may give its
/// `parent` (a name, or `null` for a root), its `translation`, `rotation` (either sign) and `scale` in the rest pose,
/// or with `time` (seconds) in animation `animation` (0 by default) at that time, and its `keys`: per path
/// (`translation`, `rotation`, `scale`), the channel's key count, its key times, or `null` for no channel.
fn rig(e: &Map<String, Value>, doc: &Document, tol: f32) -> Result<(), String> {
    let Some(entries) = e.get("rig").and_then(Value::as_array) else {
        return Ok(());
    };
    let rig = doc.rig().map_err(|err| format!("no rig: {err}"))?;
    for entry in entries {
        let name = entry["node"].as_str().ok_or("a rig entry needs \"node\"")?;
        let node = rig.find(name).ok_or_else(|| format!("rig has no node {name:?}"))?;
        if let Some(want) = entry.get("parent") {
            let got = rig.nodes[node]
                .parent
                .map(|p| rig.nodes[p].name.clone().unwrap_or_default());
            if got.as_deref() != want.as_str() {
                return Err(format!("{name}: parent is {got:?}, expected {want}"));
            }
        }
        let animation = entry["animation"].as_u64().unwrap_or(0) as usize;
        let (trs, when) = match entry.get("time").and_then(Value::as_f64) {
            Some(t) => {
                if animation >= rig.animations.len() {
                    return Err(format!("{name}: no animation {animation}"));
                }
                (rig.pose(Some(animation), t as f32)[node], format!("at {t}"))
            }
            None => (rig.nodes[node].rest, "at rest".to_owned()),
        };
        let negated = trs.rotation.map(|c| -c);
        for (key, got, alt) in [
            ("translation", &trs.translation[..], None),
            ("rotation", &trs.rotation[..], Some(&negated[..])),
            ("scale", &trs.scale[..], None),
        ] {
            let Some(want) = entry.get(key) else { continue };
            let want = floats(want);
            if !close(got, &want, tol) && !alt.is_some_and(|a| close(a, &want, tol)) {
                return Err(format!("{name} {when}: {key} is {got:?}, expected {want:?}"));
            }
        }
        for (path, want) in entry.get("keys").and_then(Value::as_object).into_iter().flatten() {
            let target = match path.as_str() {
                "translation" => polysmelt::rig::Property::Translation,
                "rotation" => polysmelt::rig::Property::Rotation,
                "scale" => polysmelt::rig::Property::Scale,
                _ => return Err(format!("{name}: no such path as {path:?}")),
            };
            let channel = rig
                .animations
                .get(animation)
                .and_then(|a| a.channels.iter().find(|c| c.node == node && c.property == target));
            let ok = match (channel, want) {
                (None, Value::Null) => true,
                (Some(c), Value::Number(n)) => n.as_u64() == Some(c.times.len() as u64),
                (Some(c), Value::Array(_)) => close(&c.times, &floats(want), tol),
                _ => false,
            };
            if !ok {
                let got = channel.map(|c| c.times.len());
                return Err(format!("{name}: {path} keys are {got:?} (a count), expected {want}"));
            }
        }
    }
    Ok(())
}

/// `Document::decode`: the case decoded, written as a GLB and read back, has no compression or quantization left
/// and flattens to the mesh it did.
fn decoded(case: &Case, ext: &str, want: Option<&Mesh>, tol: f32) -> Result<(), String> {
    let _ = ext;
    let mut doc = Loader::new()
        .assets(Folder::new(&case.dir))
        .open(&case.file)
        .map_err(|e| e.to_string())?;
    doc.decode().map_err(|err| format!("doesn't decode: {err}"))?;
    // The gltf crate refuses attribute names it doesn't know, KHR_gaussian_splatting's among them.
    let splats = doc
        .json()
        .get("extensionsUsed")
        .and_then(Value::as_array)
        .is_some_and(|used| used.iter().any(|e| e == "KHR_gaussian_splatting"));
    if !splats && let Some(why) = agrees_with_gltf_crate(&doc)? {
        return Err(format!("decoded, the gltf crate can't read it: {why}"));
    }
    let text = serde_json::to_string(doc.json()).unwrap();
    for name in [
        "EXT_meshopt_compression",
        "KHR_meshopt_compression",
        "KHR_draco_mesh_compression",
        "KHR_mesh_quantization",
    ] {
        if text.contains(name) {
            return Err(format!("decoded, it still mentions {name}"));
        }
    }
    let accessors = doc
        .json()
        .get("accessors")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let prims = doc.json().get("meshes").and_then(Value::as_array).into_iter().flatten();
    for p in prims.flat_map(|m| m["primitives"].as_array().cloned().unwrap_or_default()) {
        let targets = p.get("targets").and_then(Value::as_array).cloned().unwrap_or_default();
        for set in std::iter::once(&p["attributes"]).chain(&targets) {
            for (name, a) in set.as_object().into_iter().flatten() {
                let float = matches!(name.as_str(), "POSITION" | "NORMAL" | "TANGENT") || name.starts_with("TEXCOORD_");
                let ctype = accessors[a.as_u64().unwrap() as usize]["componentType"].as_u64();
                if float && ctype != Some(5126) {
                    return Err(format!("decoded, its {name} has component type {ctype:?}"));
                }
            }
        }
    }
    let glb = doc
        .to_glb()
        .map_err(|err| format!("decoded, it doesn't write as GLB: {err}"))?;
    if !splats {
        gltf::Gltf::from_slice(&glb).map_err(|err| format!("decoded, the gltf crate rejects it: {err}"))?;
    }
    let back =
        polysmelt::read(glb, "decoded.glb").map_err(|err| format!("decoded, its GLB doesn't read back: {err}"))?;
    let got = back
        .flatten(Pose::Rest)
        .map_err(|err| format!("decoded, it doesn't flatten: {err}"))?;
    let (want, got) = match (want, got.as_ref()) {
        (None, None) => return Ok(()),
        (Some(w), Some(g)) => (w, g),
        (w, g) => {
            return Err(format!(
                "decoded, it draws {:?} triangles, not {:?}",
                g.map(Mesh::triangles),
                w.map(Mesh::triangles)
            ));
        }
    };
    if want.prims.len() != got.prims.len() {
        return Err(format!(
            "decoded, it has {} primitives, not {}",
            got.prims.len(),
            want.prims.len()
        ));
    }
    for (i, (w, g)) in want.prims.iter().zip(&got.prims).enumerate() {
        let same = w.indices == g.indices
            && w.material == g.material
            && close(w.positions.as_flattened(), g.positions.as_flattened(), tol)
            && close(w.normals.as_flattened(), g.normals.as_flattened(), tol)
            && close(w.uvs.as_flattened(), g.uvs.as_flattened(), tol)
            && close(w.colors.as_flattened(), g.colors.as_flattened(), tol);
        if !same {
            return Err(format!("decoded, its primitive {i} draws differently"));
        }
    }
    Ok(())
}

/// Writes the case's document as a GLB under `target/conformance/`, for the Khronos validator.
fn keep_glb(case: &Case, doc: &Document) -> Result<(), String> {
    let rel = relative(&case.dir, &corpus());
    let out = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("conformance")
        .join(format!("{rel}.glb"));
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    let glb = doc
        .to_glb()
        .map_err(|err| format!("the document doesn't write as GLB: {err}"))?;
    std::fs::write(out, glb).unwrap();
    Ok(())
}

/// Writes the GLB the case's flattened mesh makes beside its document's, as `<case>.mesh.glb`, so the Khronos
/// validator checks the mesh writer (skins included) too.
fn keep_mesh_glb(case: &Case, bytes: &[u8]) {
    let rel = relative(&case.dir, &corpus());
    let out = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("conformance")
        .join(format!("{rel}.mesh.glb"));
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(out, bytes).unwrap();
}

/// Every accessor with a buffer view reads the same through [`Document::read_floats`] as through the
/// gltf crate, normalized integers mapped as glTF says. A document the gltf crate rejects or
/// panics on is skipped, returning why: those are the gltf crate's to fix.
fn agrees_with_gltf_crate(doc: &Document) -> Result<Option<String>, String> {
    use gltf::accessor::{DataType, Dimensions, util::Iter};
    let Ok(root) = serde_json::from_value::<gltf::json::Root>(Value::Object(doc.json().clone())) else {
        return Ok(Some("its JSON doesn't parse as gltf::json::Root".into()));
    };
    let Ok(g) = gltf::Document::from_json(root) else {
        return Ok(Some("gltf::Document::from_json rejects it".into()));
    };
    let mut buffers = Vec::new();
    for i in 0..g.buffers().len() {
        buffers.push(doc.buffer(i).map_err(|e| format!("buffer {i}: {e}"))?.into_owned());
    }
    let get = |b: gltf::Buffer<'_>| buffers.get(b.index()).map(Vec::as_slice);
    let compressed = |a: &gltf::Accessor<'_>| {
        a.view().is_some_and(|v| {
            let j = &doc.json()["bufferViews"][v.index()];
            j.get("extensions").is_some()
        })
    };
    for a in g.accessors().filter(|a| a.view().is_some() && !compressed(a)) {
        let n = a.dimensions().multiplicity();
        let norm = a.normalized();
        macro_rules! read {
            ($t:ty, $scale:expr) => {{
                let f = |x: $t| {
                    if norm {
                        (f64::from(x) / $scale).max(-1.0) as f32
                    } else {
                        f64::from(x) as f32
                    }
                };
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Option<Vec<f32>> {
                    Some(match a.dimensions() {
                        Dimensions::Scalar => Iter::<$t>::new(a.clone(), get)?.map(f).collect(),
                        Dimensions::Vec2 => Iter::<[$t; 2]>::new(a.clone(), get)?
                            .flat_map(|e| e.map(f))
                            .collect(),
                        Dimensions::Vec3 => Iter::<[$t; 3]>::new(a.clone(), get)?
                            .flat_map(|e| e.map(f))
                            .collect(),
                        Dimensions::Vec4 => Iter::<[$t; 4]>::new(a.clone(), get)?
                            .flat_map(|e| e.map(f))
                            .collect(),
                        Dimensions::Mat4 if std::mem::size_of::<$t>() == 4 => {
                            Iter::<[[$t; 4]; 4]>::new(a.clone(), get)?
                                .flat_map(|e| e.into_iter().flatten().map(f))
                                .collect()
                        }
                        // A small matrix's columns are padded; the gltf crate reads them packed.
                        _ => return None,
                    })
                }))
            }};
        }
        let theirs = match a.data_type() {
            DataType::I8 => read!(i8, 127.0),
            DataType::U8 => read!(u8, 255.0),
            DataType::I16 => read!(i16, 32767.0),
            DataType::U16 => read!(u16, 65535.0),
            DataType::U32 => read!(u32, 4_294_967_295.0),
            DataType::F32 => read!(f32, 1.0),
        };
        let theirs = match theirs {
            Ok(Some(v)) => v,
            Ok(None) => continue,
            Err(_) => return Ok(Some(format!("the gltf crate panics reading accessor {}", a.index()))),
        };
        let (width, ours) = doc.read_floats(a.index()).map_err(|e| {
            format!(
                "accessor {} reads with the gltf crate, not with read_floats: {e}",
                a.index()
            )
        })?;
        if width != n || ours.len() != theirs.len() {
            return Err(format!(
                "accessor {}: read_floats gives {} values {width} wide, the gltf crate {} values {n} wide",
                a.index(),
                ours.len(),
                theirs.len()
            ));
        }
        if let Some(k) =
            (0..ours.len()).find(|&k| ours[k].to_bits() != theirs[k].to_bits() && (ours[k] - theirs[k]).abs() > 1e-6)
        {
            return Err(format!(
                "accessor {} value {k}: read_floats gives {}, the gltf crate {}",
                a.index(),
                ours[k],
                theirs[k]
            ));
        }
        if matches!(a.data_type(), DataType::U8 | DataType::U16 | DataType::U32) && n == 1 && !norm {
            let ints = doc
                .read_indices(a.index())
                .map_err(|e| format!("accessor {}: read_indices: {e}", a.index()))?;
            if ints
                .iter()
                .zip(&theirs)
                .any(|(&i, &f)| (i as f32).to_bits() != f.to_bits())
            {
                return Err(format!(
                    "accessor {}: read_indices disagrees with the gltf crate",
                    a.index()
                ));
            }
        }
    }
    Ok(None)
}

/// The mesh's GLB reads back and draws the same: as many triangles, the same bounds and the same
/// centroid, skinned meshes posed by their joints.
/// `Pose::Bind` draws what `Pose::Rest` does, triangle for triangle and with the same joints,
/// unskinned primitives exactly as they are.
fn bind_pose(doc: &Document, rest: Option<&Mesh>) -> Result<(), String> {
    let bind = doc.flatten(Pose::Bind).map_err(|e| format!("bind pose: {e}"))?;
    let (Some(rest), Some(bind)) = (rest, bind.as_ref()) else {
        return match (rest.is_some(), bind.is_some()) {
            (false, false) => Ok(()),
            (r, b) => Err(format!("rest pose draws something: {r}, bind pose: {b}")),
        };
    };
    if rest.prims.len() != bind.prims.len() {
        return Err(format!(
            "bind pose has {} primitives, rest {}",
            bind.prims.len(),
            rest.prims.len()
        ));
    }
    for (k, (r, b)) in rest.prims.iter().zip(&bind.prims).enumerate() {
        if r.indices != b.indices || r.skin_joints != b.skin_joints || r.skin_weights != b.skin_weights {
            return Err(format!("bind pose primitive {k}: different triangles or joints"));
        }
        if !r.is_skinned() && (r.positions != b.positions || r.normals != b.normals) {
            return Err(format!("bind pose moved unskinned primitive {k}"));
        }
    }
    Ok(())
}

fn round_trip(mesh: &Mesh, bytes: &[u8], tol: f32) -> Result<(), String> {
    let back = polysmelt::read(bytes.to_vec(), "mesh.glb")
        .map_err(|err| format!("the mesh's GLB doesn't read back: {err}"))?
        .flatten(Pose::Rest)
        .map_err(|err| format!("the mesh's GLB doesn't flatten: {err}"))?;
    let summary = |m: &Mesh| {
        let (mut min, mut max, mut sum, mut n) = ([f32::MAX; 3], [f32::MIN; 3], [0f64; 3], 0usize);
        for v in m.prims.iter().flat_map(|p| &p.positions) {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
                sum[k] += f64::from(v[k]);
            }
            n += 1;
        }
        let mean = sum.map(|s| (s / n.max(1) as f64) as f32);
        ((m.triangles(), m.lines(), m.points()), min, max, mean)
    };
    let want = summary(mesh);
    let got = back
        .as_ref()
        .map_or(((0, 0, 0), [f32::MAX; 3], [f32::MIN; 3], [0.0; 3]), summary);
    let scale = (0..3).map(|k| (want.2[k] - want.1[k]).abs()).fold(1.0f32, f32::max);
    let t = tol.max(1e-4) * scale;
    if want.0 != got.0 || !close(&want.1, &got.1, t) || !close(&want.2, &got.2, t) || !close(&want.3, &got.3, t) {
        return Err(format!(
            "the mesh's GLB reads back as {:?} triangles, lines and points in {:?}..{:?} about {:?}, not {:?} in {:?}..{:?} about {:?}",
            got.0, got.1, got.2, got.3, want.0, want.1, want.2, want.3
        ));
    }
    Ok(())
}

/// What an error says, without the path it names first: `error` is matched against this, so a
/// case's folder name can't match it.
fn message(err: &polysmelt::Error) -> String {
    let text = err.to_string();
    match err.path().map(|p| format!("{}: ", p.display())) {
        Some(prefix) => text.strip_prefix(&prefix).map_or(text.clone(), str::to_owned),
        None => text,
    }
}

/// Runs one case, returning what's wrong with it.
fn run(case: &Case) -> Result<(), String> {
    let e = &case.expect;
    let ext = ext_of(&case.file);
    let tol = e.get("tolerance").and_then(Value::as_f64).unwrap_or(1e-4) as f32;
    let loaded = Loader::new().assets(Folder::new(&case.dir)).open(&case.file);
    // An error from loading names the file it was reading: the model or one it refers to.
    let named = loaded
        .as_ref()
        .err()
        .map(|err| err.path().is_some_and(|p| p.starts_with(&case.dir)));
    let loaded = loaded.and_then(flattened);
    same_from_memory(case, &ext, &loaded)?;
    if let Some(want) = e.get("error").and_then(Value::as_str) {
        let kind = e
            .get("error_kind")
            .and_then(Value::as_str)
            .ok_or("has an `error` but no `error_kind`")?;
        return match loaded {
            Err(err) if !message(&err).to_lowercase().contains(&want.to_lowercase()) => {
                Err(format!("failed with {err:?}, expected an error containing {want:?}"))
            }
            // Without every default feature, a case can stop at the feature it needs before
            // it reaches the data the case is about.
            Err(err)
                if format!("{:?}", err.kind()) != kind
                    && (cfg!(feature = "default") || err.kind() != polysmelt::ErrorKind::FeatureDisabled) =>
            {
                Err(format!("failed with {err:?}, expected an error of kind {kind}"))
            }
            Err(err) if named == Some(false) => Err(format!(
                "failed with {err:?}, expected the error to name a file in {}",
                case.dir.display()
            )),
            Err(_) => Ok(()),
            Ok(_) => Err(format!("loaded, expected an error containing {want:?}")),
        };
    }
    let (doc, mesh) = loaded.map_err(|err| format!("failed to load: {err}"))?;
    keep_glb(case, &doc)?;
    if let Some(why) = agrees_with_gltf_crate(&doc)? {
        println!("{}: not compared with the gltf crate: {why}", case.name);
    }
    bind_pose(&doc, mesh.as_ref())?;
    if let Some(Value::Object(want)) = e.get("facts") {
        for (k, v) in want {
            if doc.facts.get(k) != Some(v) {
                return Err(format!("fact {k} is {:?}, expected {v}", doc.facts.get(k)));
            }
        }
    }
    document(e, &doc)?;
    pointers(e, "json", doc.json())?;
    poses(e, &doc, tol)?;
    rig(e, &doc, tol)?;
    decoded(case, &ext, mesh.as_ref(), tol)?;
    if e.get("no_mesh").and_then(Value::as_bool) == Some(true) {
        return match mesh {
            None => Ok(()),
            Some(_) => Err("produced a mesh, expected none".into()),
        };
    }
    let mesh = mesh.ok_or("produced no mesh")?;
    let bytes = mesh.to_glb().map_err(|err| format!("to_glb failed: {err}"))?;
    keep_mesh_glb(case, &bytes);
    round_trip(&mesh, &bytes, tol)?;
    // `flat_json`: the same, in the JSON of the flat mesh's GLB.
    if e.contains_key("flat_json") {
        let len = bytes
            .get(12..16)
            .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
        let json: Map<String, Value> = bytes
            .get(20..20 + len)
            .and_then(|j| serde_json::from_slice(j).ok())
            .ok_or("the mesh's GLB has no JSON")?;
        pointers(e, "flat_json", &json)?;
    }
    let g = glb::validate(&bytes).map_err(|err| format!("invalid GLB: {err}"))?;
    if (g.triangles(), g.lines(), g.points()) != (mesh.triangles(), mesh.lines(), mesh.points()) {
        return Err(format!(
            "GLB has {:?} triangles, lines and points, the mesh {:?}",
            (g.triangles(), g.lines(), g.points()),
            (mesh.triangles(), mesh.lines(), mesh.points())
        ));
    }
    // glTF's count is what the JSON declares, without reading indices: drawing drops triangles
    // that name a missing vertex, so it's an upper bound there and exact elsewhere.
    let declared = matches!(ext.as_str(), "gltf" | "glb");
    if let Some(t) = doc.facts.get("triangles")
        && !t
            .as_u64()
            .is_some_and(|t| t == g.triangles() as u64 || (declared && t > g.triangles() as u64))
    {
        return Err(format!("fact triangles {t} != GLB triangles {}", g.triangles()));
    }
    // Every material a primitive uses exists, and every texture's image decodes.
    for (i, (_, img)) in g.images.iter().enumerate() {
        image::load_from_memory(img).map_err(|err| format!("image {i} doesn't decode: {err}"))?;
    }

    let count = |key: &str, got: usize| -> Result<(), String> {
        match e.get(key).and_then(Value::as_u64) {
            Some(want) if want as usize != got => Err(format!("{key} is {got}, expected {want}")),
            _ => Ok(()),
        }
    };
    count("triangles", g.triangles())?;
    count("vertices", g.vertices())?;
    count("lines", g.lines())?;
    count("points", g.points())?;
    count("primitives", g.prims.len())?;
    count("material_count", g.materials.len())?;
    count("image_count", g.images.len())?;

    // Total surface area: catches polygons triangulated outside their outline.
    if let Some(want) = e.get("area").and_then(Value::as_f64) {
        let area: f64 = g
            .prims
            .iter()
            .flat_map(|p| {
                p.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| t.map(|i| p.positions[i as usize]))
            })
            .map(|[a, b, c]| {
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]].map(f64::from);
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]].map(f64::from);
                let x = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
            })
            .sum();
        if (area - want).abs() > f64::from(tol) * (1.0 + want) {
            return Err(format!("surface area is {area}, expected {want}"));
        }
    }
    if let Some(b) = e.get("bounds") {
        let (want_min, want_max) = (floats(&b["min"]), floats(&b["max"]));
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in g.prims.iter().flat_map(|p| &p.positions) {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        if !close(&min, &want_min, tol) || !close(&max, &want_max, tol) {
            return Err(format!(
                "bounds are {min:?}..{max:?}, expected {want_min:?}..{want_max:?}"
            ));
        }
    }
    let flag = |key: &str, has: bool| -> Result<(), String> {
        match e.get(key).and_then(Value::as_bool) {
            Some(want) if want != has => Err(format!("{key} is {has}, expected {want}")),
            _ => Ok(()),
        }
    };
    flag("uvs", g.prims.iter().any(|p| !p.uvs.is_empty()))?;
    flag("colors", g.prims.iter().any(|p| !p.colors.is_empty()))?;

    // Points, UVs and colors that must appear somewhere.
    let contains = |key: &str, have: &[Vec<f32>]| -> Result<(), String> {
        for want in e.get(key).and_then(Value::as_array).into_iter().flatten() {
            let want = floats(want);
            if !have.iter().any(|h| close(h, &want, tol)) {
                return Err(format!("no {key} entry matches {want:?}"));
            }
        }
        Ok(())
    };
    let all = |f: &dyn Fn(&glb::Prim) -> Vec<Vec<f32>>| -> Vec<Vec<f32>> { g.prims.iter().flat_map(f).collect() };
    contains(
        "positions_include",
        &all(&|p| p.positions.iter().map(|v| v.to_vec()).collect()),
    )?;
    contains("uvs_include", &all(&|p| p.uvs.iter().map(|v| v.to_vec()).collect()))?;
    contains(
        "colors_include",
        &all(&|p| p.colors.iter().map(|v| v.to_vec()).collect()),
    )?;
    // Every normal is one of these (a faceted model's face directions).
    if let Some(dirs) = e.get("normals_in").and_then(Value::as_array) {
        let dirs: Vec<Vec<f32>> = dirs.iter().map(floats).collect();
        // A primitive without normals is shaded flat, as the glTF spec tells clients to.
        let flat = |p: &glb::Prim| -> Vec<[f32; 3]> {
            p.indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| {
                    let [a, b, c] = t.map(|i| p.positions[i as usize]);
                    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let f = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ];
                    let len = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
                    f.map(|x| x / len)
                })
                .filter(|n| n.iter().all(|x| x.is_finite()))
                .collect()
        };
        let normals: Vec<[f32; 3]> = g
            .prims
            .iter()
            .flat_map(|p| {
                if p.normals.is_empty() {
                    flat(p)
                } else {
                    p.normals.clone()
                }
            })
            .collect();
        for n in &normals {
            if !dirs.iter().any(|d| close(n, d, 1e-3)) {
                return Err(format!("normal {n:?} is none of {dirs:?}"));
            }
        }
    }
    // Every triangle's winding agrees with its vertex normals (counter-clockwise front faces).
    if e.get("winding_matches_normals").and_then(Value::as_bool) == Some(true) {
        // Flat-shaded primitives (no normals) take their normals from the winding: nothing to
        // compare.
        for p in g.prims.iter().filter(|p| !p.normals.is_empty()) {
            for t in p.indices.as_chunks::<3>().0 {
                let [a, b, c] = t.map(|i| p.positions[i as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let face = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let n = p.normals[t[0] as usize];
                let n2 = p.normals[t[1] as usize];
                let n3 = p.normals[t[2] as usize];
                let dot = |m: [f32; 3]| face[0] * m[0] + face[1] * m[1] + face[2] * m[2];
                if dot(n) + dot(n2) + dot(n3) < 0.0 {
                    return Err(format!("triangle {t:?} winds against its normals"));
                }
            }
        }
    }

    // For a convex model: every triangle winds to face away from the middle of the bounds (catches
    // mirrored transforms and flipped files, where faces turn inside out).
    if e.get("faces_outward").and_then(Value::as_bool) == Some(true) {
        let pts = || g.prims.iter().flat_map(|p| p.positions.iter());
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for p in pts() {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let mid = [0, 1, 2].map(|k| f32::midpoint(lo[k], hi[k]));
        for p in &g.prims {
            for t in p.indices.as_chunks::<3>().0 {
                let [a, b, c] = t.map(|i| p.positions[i as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let face = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let out = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0 - mid[k]);
                if face[0] * out[0] + face[1] * out[1] + face[2] * out[2] <= 0.0 {
                    return Err(format!("triangle {t:?} faces inward"));
                }
            }
        }
    }

    for (i, want) in e
        .get("materials")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let m = match want.get("name").and_then(Value::as_str) {
            Some(name) => g.materials.iter().find(|m| m.name == name).ok_or_else(|| {
                format!(
                    "no material named {name:?} in {:?}",
                    g.materials.iter().map(|m| &m.name).collect::<Vec<_>>()
                )
            })?,
            None => g.materials.get(i).ok_or(format!("no material {i}"))?,
        };
        let label = if m.name.is_empty() {
            format!("material {i}")
        } else {
            format!("material {:?}", m.name)
        };
        if let Some(c) = want.get("color") {
            let c = floats(c);
            let tol = want.get("tolerance").and_then(Value::as_f64).map_or(2e-3, |t| t as f32);
            if !close(&m.color[..c.len()], &c, tol) {
                return Err(format!("{label} color is {:?}, expected {c:?}", m.color));
            }
        }
        if let Some(c) = want.get("emissive") {
            let c = floats(c);
            if !close(&m.emissive, &c, 2e-3) {
                return Err(format!("{label} emissive is {:?}, expected {c:?}", m.emissive));
            }
        }
        for (key, got) in [("metallic", m.metallic), ("roughness", m.roughness)] {
            if let Some(v) = want.get(key).and_then(Value::as_f64)
                && (got - v as f32).abs() > 2e-3
            {
                return Err(format!("{label} {key} is {got}, expected {v}"));
            }
        }
        for (key, got) in [
            ("textured", m.color_image.is_some()),
            ("normal_map", m.normal_image.is_some()),
            ("double_sided", m.double_sided),
        ] {
            if let Some(v) = want.get(key).and_then(Value::as_bool)
                && v != got
            {
                return Err(format!("{label} {key} is {got}, expected {v}"));
            }
        }
        if let Some(mode) = want.get("alpha_mode").and_then(Value::as_str)
            && mode != m.alpha_mode
        {
            return Err(format!("{label} alphaMode is {}, expected {mode}", m.alpha_mode));
        }
        for (key, what, image) in [
            ("pixel", "color", m.color_image),
            ("metal_rough_pixel", "metallic-roughness", m.metal_rough_image),
            ("occlusion_pixel", "occlusion", m.occlusion_image),
        ] {
            let Some(px) = want.get(key) else {
                continue;
            };
            let px = floats(px);
            let img = image.ok_or(format!("{label} has no {what} texture to sample"))?;
            let rgb = image::load_from_memory(&g.images[img].1).unwrap().to_rgb8();
            let got = rgb.get_pixel(0, 0).0.map(f32::from);
            if !close(&got, &px, 0.02) {
                return Err(format!("{label} {what} texture pixel is {got:?}, expected {px:?}"));
            }
        }
    }
    Ok(())
}

/// Runs a format's cases, reporting every failure at once.
fn check(format: &str) {
    let cases = cases(format);
    assert!(!cases.is_empty(), "no cases for {format}");
    let mut failures = Vec::new();
    let mut ran = 0;
    for case in &cases {
        let needs = case.expect.get("needs").and_then(Value::as_str);
        if !enabled(&ext_of(&case.file)) || needs.is_some_and(|f| !has_feature(f)) {
            continue;
        }
        ran += 1;
        if let Err(why) = run(case) {
            failures.push(format!("{}: {why}", case.name));
        }
    }
    println!("{format}: {ran} of {} cases ran", cases.len());
    assert!(
        failures.is_empty(),
        "{} of {ran} {format} cases failed:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn obj() {
    check("obj");
}
#[test]
fn stl() {
    check("stl");
}
#[test]
fn ply() {
    check("ply");
}
#[test]
fn vox() {
    check("vox");
}
#[test]
fn tds() {
    check("3ds");
}
#[test]
fn threemf() {
    check("3mf");
}
#[test]
fn fbx() {
    check("fbx");
}
#[test]
fn usd() {
    check("usd");
}

#[test]
fn gltf() {
    check("gltf");
}
#[test]
fn skp() {
    check("skp");
}
#[test]
fn bvh() {
    check("bvh");
}
#[test]
fn dae() {
    check("dae");
}
#[test]
fn off() {
    check("off");
}
#[test]
fn vrm() {
    check("vrm");
}
#[test]
fn x3d() {
    check("x3d");
}
#[test]
fn dxf() {
    check("dxf");
}
#[test]
fn splat() {
    check("splat");
}
#[test]
fn spz() {
    check("spz");
}
#[test]
fn mmd() {
    check("mmd");
}
#[test]
fn directx() {
    check("x");
}
#[test]
fn md2() {
    check("md2");
}
#[test]
fn md3() {
    check("md3");
}
#[test]
fn md5() {
    check("md5");
}
#[test]
fn iqm() {
    check("iqm");
}
#[test]
fn smd() {
    check("smd");
}
#[test]
fn ms3d() {
    check("ms3d");
}
#[test]
fn b3d() {
    check("b3d");
}
#[test]
fn asf() {
    check("asf");
}
#[test]
fn c3d() {
    check("c3d");
}
#[test]
fn trc() {
    check("trc");
}
#[test]
fn htr() {
    check("htr");
}
#[test]
fn las() {
    check("las");
}
#[test]
fn pcd() {
    check("pcd");
}
#[test]
fn xyz() {
    check("xyz");
}
#[test]
fn e57() {
    check("e57");
}
#[test]
fn amf() {
    check("amf");
}
#[test]
fn lwo() {
    check("lwo");
}
#[test]
fn ac3d() {
    check("ac");
}
#[test]
fn ogre() {
    check("ogre");
}
#[test]
fn flt() {
    check("flt");
}
#[test]
fn ogex() {
    check("ogex");
}

#[test]
fn irr() {
    check("irr");
}

#[test]
fn xgl() {
    check("xgl");
}

/// Every extension a [`Format`] has has cases (and `.xml`, read as the format its root names), and every format a
/// corpus folder.
#[test]
fn corpus_covers_every_extension() {
    let mut seen = std::collections::HashSet::new();
    for dir in std::fs::read_dir(corpus()).unwrap().flatten() {
        if !dir.path().is_dir() || dir.file_name() == "tools" {
            continue;
        }
        let format = dir.file_name().to_string_lossy().into_owned();
        for case in cases(&format) {
            seen.insert(ext_of(&case.file));
        }
    }
    let extensions = Format::all().flat_map(Format::extensions).copied().chain(["xml"]);
    let missing: Vec<_> = extensions.filter(|e| !seen.contains(*e)).collect();
    assert!(missing.is_empty(), "no conformance cases for {missing:?}");
}

/// A reader told to drop something comes out as a full read pruned the same way: what readers
/// skip (material libraries, texture files, shaders) changes nothing that's kept.
#[test]
fn reading_less_matches_pruning() {
    let none = Keep::ALL
        .with_animations(false)
        .with_skins(false)
        .with_morph_targets(false)
        .with_materials(false)
        .with_textures(false)
        .with_cameras(false)
        .with_lights(false)
        .with_extras(false)
        .with_image_bytes(false);
    let keeps = [
        ("no textures", Keep::ALL.with_textures(false)),
        ("no materials", Keep::ALL.with_materials(false)),
        ("meshes", Keep::MESHES),
        ("nothing", none),
    ];
    let mut failures = Vec::new();
    for dir in std::fs::read_dir(corpus()).unwrap().flatten() {
        if !dir.path().is_dir() || dir.file_name() == "tools" {
            continue;
        }
        for case in cases(&dir.file_name().to_string_lossy()) {
            let ext = ext_of(&case.file);
            let needs = case.expect.get("needs").and_then(Value::as_str);
            if !enabled(&ext) || needs.is_some_and(|f| !has_feature(f)) || case.expect.contains_key("error") {
                continue;
            }
            let loader = Loader::new().assets(Folder::new(&case.dir));
            for (label, keep) in keeps {
                let Ok(mut full) = loader.open(&case.file) else {
                    continue;
                };
                full.prune(keep);
                let less = match loader.clone().keep(keep).open(&case.file) {
                    Ok(d) => d,
                    Err(e) => {
                        failures.push(format!("{} ({label}): {e}", case.name));
                        continue;
                    }
                };
                // Skipped work leaves no data behind, where pruning leaves it unreferenced; and a
                // reader that skips skins, cameras or lights may leave fewer nodes. Both are
                // compared through the meshes they make. Without textures, a material's factors
                // are its own, not the ones a found texture calls for, so only its name is.
                let mut skip = vec!["accessors", "bufferViews", "buffers", "nodes", "scenes"];
                if !keep.textures {
                    skip.push("materials");
                }
                let differ: Vec<String> = less
                    .json()
                    .keys()
                    .chain(full.json().keys())
                    .filter(|k| !skip.contains(&k.as_str()) && less.json().get(*k) != full.json().get(*k))
                    .map(|k| {
                        let show = |d: &Document| {
                            d.json().get(k).map_or_else(
                                || "none".to_owned(),
                                |v| v.to_string().chars().take(300).collect::<String>(),
                            )
                        };
                        format!("{k}: {} vs pruned {}", show(&less), show(&full))
                    })
                    .collect();
                let names = |d: &Document| -> Vec<Value> {
                    d.json()
                        .get("materials")
                        .and_then(Value::as_array)
                        .map_or_else(Vec::new, |m| {
                            m.iter().map(|m| m.get("name").cloned().unwrap_or_default()).collect()
                        })
                };
                let flat = |d: &Document| d.flatten(Pose::Rest).ok().flatten().map(|m| format!("{:?}", m.prims));
                if !differ.is_empty() {
                    failures.push(format!("{} ({label}): {}", case.name, differ.join("; ")));
                } else if names(&less) != names(&full) {
                    failures.push(format!("{} ({label}): material names differ", case.name));
                } else if less.counts() != full.counts() {
                    failures.push(format!("{} ({label}): counts differ", case.name));
                } else if keep.skins && flat(&less) != flat(&full) {
                    // (Pruning a skin off a full read leaves the mesh in its bind space, where a
                    // reader that skips the skin places it; so without skins only the counts are
                    // compared.)
                    failures.push(format!("{} ({label}): the flat mesh differs", case.name));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} reads differ:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// A read that leaves images unread names the same ones a full read embeds: each loads, through
/// its handle, to the image the full read has, and the rest of the document is the same.
#[test]
fn unread_images_load_as_read_ones() {
    let light = Keep::ALL.with_image_bytes(false);
    let (mut failures, mut pending) = (Vec::new(), 0);
    for dir in std::fs::read_dir(corpus()).unwrap().flatten() {
        if !dir.path().is_dir() || dir.file_name() == "tools" {
            continue;
        }
        for case in cases(&dir.file_name().to_string_lossy()) {
            let ext = ext_of(&case.file);
            let needs = case.expect.get("needs").and_then(Value::as_str);
            if !enabled(&ext) || needs.is_some_and(|f| !has_feature(f)) || case.expect.contains_key("error") {
                continue;
            }
            let assets = Folder::new(&case.dir);
            let loader = Loader::new().assets(assets.clone());
            let Ok(full) = loader.open(&case.file) else {
                continue;
            };
            let mut less = match loader.keep(light).open(&case.file) {
                Ok(d) => d,
                Err(e) => {
                    failures.push(format!("{}: {e}", case.name));
                    continue;
                }
            };
            let mut bad = |why: String| failures.push(format!("{}: {why}", case.name));
            for h in less.images().filter(polysmelt::images::ImageHandle::is_pending) {
                pending += 1;
                let uri = less.json()["images"][h.index()].get("uri").and_then(Value::as_str);
                if uri.is_none() {
                    bad(format!("image {} is pending without a uri", h.index()));
                }
                if let Location::File(p) = h.location()
                    && !p.is_file()
                {
                    bad(format!("image {} names {} as its file", h.index(), p.display()));
                }
            }
            if let Err(e) = less.to_glb() {
                bad(format!("unread images don't write: {e}"));
            }
            let unread = less.embed_images(&assets);
            let read = |d: &Document| -> Vec<Option<(polysmelt::Bytes, &'static str)>> {
                d.images()
                    .map(|h| h.load(&assets).ok().map(|i| (i.bytes, i.mime)))
                    .collect()
            };
            let (a, b) = (read(&less), read(&full));
            if unread > 0 {
                // Images the assets can't embed are named all the same: those a full read did
                // read are among the rest.
                if b.iter().flatten().any(|img| !a.contains(&Some(img.clone()))) {
                    bad("an image a full read has isn't among the unread ones".into());
                }
                continue;
            }
            if a != b {
                bad("the images load differently".into());
            }
            let skip = ["accessors", "bufferViews", "buffers", "images"];
            let differ: Vec<&String> = less
                .json()
                .keys()
                .chain(full.json().keys())
                .filter(|k| !skip.contains(&k.as_str()) && less.json().get(*k) != full.json().get(*k))
                .collect();
            if !differ.is_empty() {
                bad(format!("{differ:?} differ"));
            }
        }
    }
    assert!(pending > 0, "no case left an image unread");
    assert!(
        failures.is_empty(),
        "{} reads differ:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
