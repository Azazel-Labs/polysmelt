//! glTF parity: every `.gltf` and `.glb` under a folder (such as a checkout of the Khronos
//! glTF-Sample-Assets `Models/`), loaded with the `gltf` crate and with Polysmelt, and what each
//! read compared.
//!
//! - `gltf`: `gltf::import` (document, buffers, images decoded), then every primitive's
//!   attributes, indices and morph targets, every skin's inverse bind matrices and every
//!   animation sampler's key times read through its readers. When `import` fails, the document
//!   and buffers alone (`Gltf::open` and `import_buffers`) are tried, to tell an image it can't
//!   decode from a model it can't read.
//! - Polysmelt: `formats::load` with `Keep::ALL`; the document written as a GLB; flattened at rest
//!   (`Document::flatten`) and that mesh written as a GLB; `Document::rig`; then
//!   `Document::decode` (Draco, meshopt, quantization) and every accessor read as the `gltf` side
//!   reads it.
//! - `declared`: the counts the file's JSON declares, read with `serde_json` alone, as the referee
//!   for disputed counts.
//!
//! Each side runs under `catch_unwind`, so a panic is recorded rather than ending the run.
//!
//! ```text
//! cargo run --release --features evidence --bin parity -- <models-folder> --out <glb-folder> --json parity.json
//! ```

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{Map, Value, json};

fn panic_text(e: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = e.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic".into()
    }
}

/// What one loader read from a file.
#[derive(Default, Debug, Clone)]
struct Stats {
    meshes: usize,
    primitives: usize,
    /// Primitives' `POSITION` elements, read.
    vertices: usize,
    /// Index elements, read (0 for primitives without indices).
    indices: usize,
    /// Triangles the primitives make (triangle lists, strips and fans).
    triangles: usize,
    /// Each attribute semantic, and how many primitives have it.
    attrs: BTreeMap<String, usize>,
    morph_targets: usize,
    skins: usize,
    joints: usize,
    animations: usize,
    channels: usize,
    /// Animation samplers' key times, read.
    keys: usize,
    materials: usize,
    textures: usize,
    images: usize,
    samplers: usize,
    cameras: usize,
    lights: usize,
    scenes: usize,
    nodes: usize,
    /// Sum of every position component, and of every index: whether both read the same values.
    pos_sum: f64,
    idx_sum: f64,
    /// Accessors that exist but didn't read.
    read_errors: Vec<String>,
}

impl Stats {
    fn to_json(&self) -> Value {
        json!({
            "meshes": self.meshes, "primitives": self.primitives, "vertices": self.vertices,
            "indices": self.indices, "triangles": self.triangles, "attrs": self.attrs,
            "morph_targets": self.morph_targets, "skins": self.skins, "joints": self.joints,
            "animations": self.animations, "channels": self.channels, "keys": self.keys,
            "materials": self.materials, "textures": self.textures, "images": self.images,
            "samplers": self.samplers, "cameras": self.cameras, "lights": self.lights,
            "scenes": self.scenes, "nodes": self.nodes, "pos_sum": self.pos_sum, "idx_sum": self.idx_sum,
            "read_errors": self.read_errors,
        })
    }
}

fn tri_count(mode: u64, n: usize) -> usize {
    match mode {
        4 => n / 3,
        5 | 6 => n.saturating_sub(2),
        _ => 0,
    }
}

// --- The gltf crate -------------------------------------------------------------------------

fn gltf_stats(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> (Stats, usize) {
    let get = |b: gltf::Buffer| buffers.get(b.index()).map(|d| &d.0[..]);
    let mut s = Stats {
        meshes: doc.meshes().count(),
        skins: doc.skins().count(),
        animations: doc.animations().count(),
        materials: doc.materials().count(),
        textures: doc.textures().count(),
        images: doc.images().count(),
        samplers: doc.samplers().count(),
        cameras: doc.cameras().count(),
        lights: doc.lights().map_or(0, Iterator::count),
        scenes: doc.scenes().count(),
        nodes: doc.nodes().count(),
        ..Stats::default()
    };
    let mut mesh_tris = vec![0usize; s.meshes];
    for mesh in doc.meshes() {
        for p in mesh.primitives() {
            s.primitives += 1;
            let r = p.reader(get);
            let at = format!("mesh {} prim {}", mesh.index(), p.index());
            let mut verts = 0;
            for (sem, _) in p.attributes() {
                let name = sem.to_string();
                *s.attrs.entry(name.clone()).or_default() += 1;
                let n = match sem {
                    gltf::Semantic::Positions => r.read_positions().map(|it| {
                        let mut n = 0;
                        for v in it {
                            s.pos_sum += f64::from(v[0]) + f64::from(v[1]) + f64::from(v[2]);
                            n += 1;
                        }
                        verts = n;
                        n
                    }),
                    gltf::Semantic::Normals => r.read_normals().map(Iterator::count),
                    gltf::Semantic::Tangents => r.read_tangents().map(Iterator::count),
                    gltf::Semantic::Colors(k) => r.read_colors(k).map(|c| c.into_rgba_f32().count()),
                    gltf::Semantic::TexCoords(k) => r.read_tex_coords(k).map(|c| c.into_f32().count()),
                    gltf::Semantic::Joints(k) => r.read_joints(k).map(|c| c.into_u16().count()),
                    gltf::Semantic::Weights(k) => r.read_weights(k).map(|c| c.into_f32().count()),
                    gltf::Semantic::Extras(_) => Some(0),
                };
                if n.is_none() {
                    s.read_errors.push(format!("{at} {name}"));
                }
            }
            s.vertices += verts;
            let n = if p.indices().is_some() {
                match r.read_indices() {
                    Some(it) => {
                        let mut n = 0;
                        for i in it.into_u32() {
                            s.idx_sum += f64::from(i);
                            n += 1;
                        }
                        s.indices += n;
                        n
                    }
                    None => {
                        s.read_errors.push(format!("{at} indices"));
                        0
                    }
                }
            } else {
                verts
            };
            let t = tri_count(u64::from(p.mode().as_gl_enum()), n);
            s.triangles += t;
            mesh_tris[mesh.index()] += t;
            for (k, (pos, nor, tan)) in r.read_morph_targets().enumerate() {
                s.morph_targets += 1;
                let _ = (pos.map(Iterator::count), nor.map(Iterator::count), tan.map(Iterator::count));
                let _ = k;
            }
        }
    }
    for skin in doc.skins() {
        s.joints += skin.joints().count();
        if skin.inverse_bind_matrices().is_some() && skin.reader(get).read_inverse_bind_matrices().is_none() {
            s.read_errors.push(format!("skin {} inverseBindMatrices", skin.index()));
        }
    }
    for anim in doc.animations() {
        for ch in anim.channels() {
            s.channels += 1;
            let r = ch.reader(get);
            match r.read_inputs() {
                Some(it) => s.keys += it.count(),
                None => s.read_errors.push(format!("animation {} channel input", anim.index())),
            }
            if r.read_outputs().is_none() {
                s.read_errors.push(format!("animation {} channel output", anim.index()));
            }
        }
    }
    // The triangles the default scene draws, each instance counted.
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    let mut drawn = 0;
    let mut stack: Vec<gltf::Node> = match scene {
        Some(sc) => sc.nodes().collect(),
        None => {
            let mut child = vec![false; s.nodes];
            for n in doc.nodes() {
                for c in n.children() {
                    child[c.index()] = true;
                }
            }
            doc.nodes().filter(|n| !child[n.index()]).collect()
        }
    };
    let mut steps = 0;
    while let Some(n) = stack.pop() {
        steps += 1;
        if steps > 10_000_000 {
            break;
        }
        if let Some(m) = n.mesh() {
            drawn += mesh_tris[m.index()];
        }
        stack.extend(n.children());
    }
    if s.nodes == 0 {
        drawn = mesh_tris.iter().sum();
    }
    (s, drawn)
}

fn run_gltf(path: &Path) -> Value {
    let start = Instant::now();
    let r = catch_unwind(AssertUnwindSafe(|| match gltf::import(path) {
        Ok((doc, buffers, images)) => {
            let (s, drawn) = gltf_stats(&doc, &buffers);
            Ok::<_, String>(("ok", s, drawn, images.len(), None))
        }
        Err(e) => {
            // The document and buffers alone, without decoding images.
            let full = e.to_string();
            let g = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
            let buffers =
                gltf::import_buffers(&g.document, path.parent(), g.blob.clone()).map_err(|e| e.to_string())?;
            let (s, drawn) = gltf_stats(&g.document, &buffers);
            Ok(("ok-without-images", s, drawn, 0, Some(full)))
        }
    }));
    let ms = start.elapsed().as_secs_f64() * 1e3;
    match r {
        Ok(Ok((status, s, drawn, images, err))) => json!({
            "status": status, "error": err, "ms": ms, "stats": s.to_json(), "scene_triangles": drawn,
            "images_decoded": images,
        }),
        Ok(Err(e)) => json!({ "status": "fail", "error": e, "ms": ms }),
        Err(p) => json!({ "status": "panic", "error": panic_text(&p), "ms": ms }),
    }
}

// --- Polysmelt ------------------------------------------------------------------------------

fn list<'a>(json: &'a Map<String, Value>, key: &str) -> &'a [Value] {
    json.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn idx(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|i| i as usize)
}

/// Counts from a glTF JSON alone: what it declares.
fn declared(json: &Map<String, Value>) -> Stats {
    let accessors = list(json, "accessors");
    let count = |a: Option<usize>| a.and_then(|a| accessors.get(a)).and_then(|a| idx(a, "count")).unwrap_or(0);
    let mut s = Stats {
        meshes: list(json, "meshes").len(),
        skins: list(json, "skins").len(),
        animations: list(json, "animations").len(),
        materials: list(json, "materials").len(),
        textures: list(json, "textures").len(),
        images: list(json, "images").len(),
        samplers: list(json, "samplers").len(),
        cameras: list(json, "cameras").len(),
        scenes: list(json, "scenes").len(),
        nodes: list(json, "nodes").len(),
        lights: json
            .get("extensions")
            .and_then(|e| e.get("KHR_lights_punctual"))
            .and_then(|l| l.get("lights"))
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        ..Stats::default()
    };
    for m in list(json, "meshes") {
        for p in m.get("primitives").and_then(Value::as_array).into_iter().flatten() {
            s.primitives += 1;
            let attrs = p.get("attributes").and_then(Value::as_object);
            for k in attrs.into_iter().flat_map(|a| a.keys()) {
                *s.attrs.entry(k.clone()).or_default() += 1;
            }
            let verts = count(attrs.and_then(|a| a.get("POSITION")).and_then(Value::as_u64).map(|i| i as usize));
            s.vertices += verts;
            let n = match idx(p, "indices") {
                Some(i) => {
                    let n = count(Some(i));
                    s.indices += n;
                    n
                }
                None => verts,
            };
            s.triangles += tri_count(p.get("mode").and_then(Value::as_u64).unwrap_or(4), n);
            s.morph_targets += p.get("targets").and_then(Value::as_array).map_or(0, Vec::len);
        }
    }
    for sk in list(json, "skins") {
        s.joints += sk.get("joints").and_then(Value::as_array).map_or(0, Vec::len);
    }
    for a in list(json, "animations") {
        let samplers = a.get("samplers").and_then(Value::as_array);
        for ch in a.get("channels").and_then(Value::as_array).into_iter().flatten() {
            s.channels += 1;
            let input = idx(ch, "sampler")
                .and_then(|i| samplers.and_then(|ss| ss.get(i)))
                .and_then(|sm| idx(sm, "input"));
            s.keys += count(input);
        }
    }
    s
}

/// [`declared`], with every accessor read through the document (after `Document::decode`).
fn polysmelt_stats(doc: &polysmelt::document::Document) -> Stats {
    let json = &doc.json;
    let mut s = declared(json);
    let (mut verts, mut indices) = (0, 0);
    s.pos_sum = 0.0;
    s.idx_sum = 0.0;
    let mut tris = 0;
    for (mi, m) in list(json, "meshes").iter().enumerate() {
        for (pi, p) in m.get("primitives").and_then(Value::as_array).into_iter().flatten().enumerate() {
            let at = format!("mesh {mi} prim {pi}");
            let mut v = 0;
            for (k, a) in p.get("attributes").and_then(Value::as_object).into_iter().flatten() {
                let Some(a) = a.as_u64() else { continue };
                if k.starts_with('_') {
                    continue;
                }
                match doc.read_floats(a as usize) {
                    Ok((w, vals)) => {
                        if k == "POSITION" {
                            v = vals.len() / w.max(1);
                            s.pos_sum += vals.iter().map(|&x| f64::from(x)).sum::<f64>();
                        }
                    }
                    Err(e) => s.read_errors.push(format!("{at} {k}: {e}")),
                }
            }
            verts += v;
            let n = match idx(p, "indices") {
                Some(i) => match doc.read_indices(i) {
                    Ok(ix) => {
                        s.idx_sum += ix.iter().map(|&i| f64::from(i)).sum::<f64>();
                        indices += ix.len();
                        ix.len()
                    }
                    Err(e) => {
                        s.read_errors.push(format!("{at} indices: {e}"));
                        0
                    }
                },
                None => v,
            };
            tris += tri_count(p.get("mode").and_then(Value::as_u64).unwrap_or(4), n);
            for (ti, t) in p.get("targets").and_then(Value::as_array).into_iter().flatten().enumerate() {
                for (k, a) in t.as_object().into_iter().flatten() {
                    if let Some(a) = a.as_u64()
                        && let Err(e) = doc.read_floats(a as usize)
                    {
                        s.read_errors.push(format!("{at} target {ti} {k}: {e}"));
                    }
                }
            }
        }
    }
    s.vertices = verts;
    s.indices = indices;
    s.triangles = tris;
    for (si, sk) in list(json, "skins").iter().enumerate() {
        if let Some(a) = idx(sk, "inverseBindMatrices")
            && let Err(e) = doc.read_floats(a)
        {
            s.read_errors.push(format!("skin {si} inverseBindMatrices: {e}"));
        }
    }
    let mut keys = 0;
    for (ai, a) in list(json, "animations").iter().enumerate() {
        let samplers = a.get("samplers").and_then(Value::as_array);
        for ch in a.get("channels").and_then(Value::as_array).into_iter().flatten() {
            let Some(sm) = idx(ch, "sampler").and_then(|i| samplers.and_then(|ss| ss.get(i))) else { continue };
            if let Some(i) = idx(sm, "input") {
                match doc.read_floats(i) {
                    Ok((_, v)) => keys += v.len(),
                    Err(e) => s.read_errors.push(format!("animation {ai} input: {e}")),
                }
            }
            if let Some(o) = idx(sm, "output")
                && let Err(e) = doc.read_floats(o)
            {
                s.read_errors.push(format!("animation {ai} output: {e}"));
            }
        }
    }
    s.keys = keys;
    s
}

fn step<T>(f: impl FnOnce() -> polysmelt::Result<T>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.to_string()),
        Err(p) => Err(panic_text(&p)),
    }
}

fn run_polysmelt(path: &Path, ext: &str, out: Option<&Path>) -> Value {
    use polysmelt::document::{Keep, Pose};
    let start = Instant::now();
    let dir = path.parent().unwrap();
    let loaded = step(|| polysmelt::formats::load(path, ext, &polysmelt::assets::Folder::new(dir), Keep::ALL));
    let load_ms = start.elapsed().as_secs_f64() * 1e3;
    let mut doc = match loaded {
        Ok(d) => d,
        Err(e) => {
            let status = if e.starts_with("panic") { "panic" } else { "fail" };
            return json!({ "status": status, "error": e, "ms": load_ms });
        }
    };
    let mut r = Map::new();
    r.insert("status".into(), json!("ok"));
    r.insert("ms".into(), json!(load_ms));
    // The document's GLB, as Polysmelt converts the file.
    match step(|| doc.to_glb()) {
        Ok(glb) => {
            r.insert("glb_bytes".into(), json!(glb.len()));
            if let Some(o) = out {
                std::fs::create_dir_all(o.parent().unwrap()).unwrap();
                std::fs::write(o.with_extension("glb"), glb).unwrap();
            }
        }
        Err(e) => {
            r.insert("glb_error".into(), json!(e));
        }
    }
    let t = Instant::now();
    match step(|| doc.flatten(Pose::Rest)) {
        Ok(Some(mesh)) => {
            r.insert(
                "flat".into(),
                json!({
                    "triangles": mesh.triangles(), "lines": mesh.lines(), "points": mesh.points(),
                    "vertices": mesh.prims.iter().map(|p| p.positions.len()).sum::<usize>(),
                    "prims": mesh.prims.len(), "materials": mesh.materials.len(), "images": mesh.images.len(),
                    "ms": t.elapsed().as_secs_f64() * 1e3,
                }),
            );
            match step(|| polysmelt::mesh::to_glb(&mesh)) {
                Ok(glb) => {
                    if let Some(o) = out {
                        std::fs::write(o.with_extension("mesh.glb"), glb).unwrap();
                    }
                }
                Err(e) => {
                    r.insert("mesh_glb_error".into(), json!(e));
                }
            }
        }
        Ok(None) => {
            r.insert("flat".into(), json!({ "triangles": 0, "empty": true }));
        }
        Err(e) => {
            r.insert("flat_error".into(), json!(e));
        }
    }
    match step(|| doc.rig()) {
        Ok(rig) => {
            r.insert(
                "rig".into(),
                json!({
                    "nodes": rig.nodes.len(), "skins": rig.skins.len(), "animations": rig.animations.len(),
                    "channels": rig.animations.iter().map(|a| a.channels.len()).sum::<usize>(),
                }),
            );
        }
        Err(e) => {
            r.insert("rig_error".into(), json!(e));
        }
    }
    let embedded = list(&doc.json, "images").iter().filter(|i| i.get("bufferView").is_some()).count();
    r.insert("images_embedded".into(), json!(embedded));
    let undecoded = list(&doc.json, "accessors").iter().filter(|a| a.get("bufferView").is_none() && a.get("sparse").is_none()).count();
    if let Err(e) = step(|| doc.decode()) {
        r.insert("decode_error".into(), json!(e));
    }
    let stats = catch_unwind(AssertUnwindSafe(|| polysmelt_stats(&doc)));
    match stats {
        Ok(s) => {
            r.insert("stats".into(), s.to_json());
        }
        Err(p) => {
            r.insert("stats_error".into(), json!(panic_text(&p)));
        }
    }
    r.insert("accessors_without_view".into(), json!(undecoded));
    Value::Object(r)
}

/// The JSON of a `.gltf` or `.glb` file, read with `serde_json` alone.
fn source_json(path: &Path) -> Option<Map<String, Value>> {
    let bytes = std::fs::read(path).ok()?;
    let text: &[u8] = if bytes.starts_with(b"glTF") {
        let len = u32::from_le_bytes(bytes.get(12..16)?.try_into().ok()?) as usize;
        bytes.get(20..20 + len)?
    } else {
        &bytes
    };
    match serde_json::from_slice(text).ok()? {
        Value::Object(m) => Some(m),
        _ => None,
    }
}

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            files(&p, out);
        } else if matches!(
            p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
            Some("gltf" | "glb")
        ) {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let root = PathBuf::from(args.first().expect("usage: parity <models-folder> [--out dir] [--json file] [--only substring]"));
    let out = arg("--out").map(PathBuf::from);
    let only = arg("--only");
    let mut paths = Vec::new();
    files(&root, &mut paths);
    if let Some(o) = &only {
        paths.retain(|p| p.to_string_lossy().contains(o.as_str()));
    }
    std::panic::set_hook(Box::new(|_| {}));
    let mut results = Vec::new();
    for (k, path) in paths.iter().enumerate() {
        let rel = path.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
        let ext = path.extension().unwrap().to_string_lossy().to_ascii_lowercase();
        eprintln!("[{}/{}] {rel}", k + 1, paths.len());
        let o = out.as_ref().map(|o| o.join(&rel));
        let decl = source_json(path).map(|j| declared(&j).to_json());
        let g = run_gltf(path);
        let p = run_polysmelt(path, &ext, o.as_deref());
        results.push(json!({
            "file": rel, "bytes": std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            "declared": decl, "gltf": g, "polysmelt": p,
        }));
    }
    let doc = serde_json::to_string_pretty(&Value::Array(results)).unwrap();
    match arg("--json") {
        Some(p) => std::fs::write(p, doc).unwrap(),
        None => println!("{doc}"),
    }
}
