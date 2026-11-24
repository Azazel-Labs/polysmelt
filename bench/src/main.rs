//! Load speed and peak memory of Polysmelt's readers against the leading crate for each format.
//!
//! The inputs are generated, so every run measures the same bytes: a torus of quads with
//! normals and texture coordinates, at three sizes, written as OBJ, binary and ASCII STL, and
//! binary and ASCII PLY. Each loader reads the file from disk into its own in-memory form.
//!
//! - Time: loaders take turns, in a rotating order, so drift on a shared machine hits all of
//!   them alike. The table gives the median and the fastest round.
//! - Memory: a counting global allocator gives the peak heap during a load and the heap the
//!   result keeps. These counts are exact, however noisy the machine.
//!
//! ```text
//! cargo run --release -- [--sizes small,medium,large] [--budget 3] [--json out.json] [--markdown out.md]
//! ```
//!
//! With the `baseline` feature, another commit's Polysmelt (the `polysmelt_base` dependency, which
//! the bench workflow adds) takes turns with the rest, so a regression check runs both commits
//! on one machine. Times from separate runs aren't comparable: each may get different hardware.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use serde_json::json;

// --- Allocation counting --------------------------------------------------------------------

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = CURRENT.fetch_add(by, Relaxed) + by;
    PEAK.fetch_max(now, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, new) };
        if !q.is_null() {
            if new >= layout.size() {
                grew(new - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

// --- Inputs ---------------------------------------------------------------------------------

/// A torus of `rows` × `cols` quads, with seams duplicated so texture coordinates stay continuous.
struct Torus {
    rows: usize,
    cols: usize,
}

impl Torus {
    fn vertex(&self, i: usize, j: usize) -> ([f32; 3], [f32; 3], [f32; 2]) {
        let (u, v) = (i as f32 / self.rows as f32, j as f32 / self.cols as f32);
        let (a, b) = (u * std::f32::consts::TAU, v * std::f32::consts::TAU);
        let (big, small) = (1.0f32, 0.35f32);
        let n = [a.cos() * b.cos(), a.sin() * b.cos(), b.sin()];
        let p = [
            (big + small * b.cos()) * a.cos(),
            (big + small * b.cos()) * a.sin(),
            small * b.sin(),
        ];
        (p, n, [u, v])
    }
    fn index(&self, i: usize, j: usize) -> usize {
        i * (self.cols + 1) + j
    }
    fn vertices(&self) -> impl Iterator<Item = ([f32; 3], [f32; 3], [f32; 2])> + '_ {
        (0..=self.rows).flat_map(move |i| (0..=self.cols).map(move |j| self.vertex(i, j)))
    }
    /// Quads, counter-clockwise seen from outside.
    fn quads(&self) -> impl Iterator<Item = [usize; 4]> + '_ {
        (0..self.rows).flat_map(move |i| {
            (0..self.cols).map(move |j| {
                [
                    self.index(i, j),
                    self.index(i + 1, j),
                    self.index(i + 1, j + 1),
                    self.index(i, j + 1),
                ]
            })
        })
    }
    fn triangles(&self) -> usize {
        self.rows * self.cols * 2
    }
    /// Each quad as two triangles of positions, with the facet normal.
    fn facets(&self) -> impl Iterator<Item = ([f32; 3], [[f32; 3]; 3])> + '_ {
        let pos: Vec<[f32; 3]> = self.vertices().map(|v| v.0).collect();
        self.quads().flat_map(move |[a, b, c, d]| {
            [[a, b, c], [a, c, d]].map(|t| {
                let [p, q, r] = t.map(|k| pos[k]);
                let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(f32::MIN_POSITIVE);
                (n.map(|x| x / len), [p, q, r])
            })
        })
    }
}

fn write_obj(t: &Torus, path: &Path) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "# torus {}x{}\no Torus", t.rows, t.cols)?;
    for (p, _, _) in t.vertices() {
        writeln!(f, "v {:.6} {:.6} {:.6}", p[0], p[1], p[2])?;
    }
    for (_, _, uv) in t.vertices() {
        writeln!(f, "vt {:.6} {:.6}", uv[0], uv[1])?;
    }
    for (_, n, _) in t.vertices() {
        writeln!(f, "vn {:.4} {:.4} {:.4}", n[0], n[1], n[2])?;
    }
    writeln!(f, "s 1")?;
    for q in t.quads() {
        let [a, b, c, d] = q.map(|k| k + 1);
        writeln!(f, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c} {d}/{d}/{d}")?;
    }
    f.flush()
}

fn write_stl_binary(t: &Torus, path: &Path) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(&[0u8; 80])?;
    f.write_all(&(t.triangles() as u32).to_le_bytes())?;
    for (n, tri) in t.facets() {
        for v in std::iter::once(n).chain(tri) {
            for c in v {
                f.write_all(&c.to_le_bytes())?;
            }
        }
        f.write_all(&[0, 0])?;
    }
    f.flush()
}

fn write_stl_ascii(t: &Torus, path: &Path) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "solid torus")?;
    for (n, tri) in t.facets() {
        writeln!(
            f,
            "  facet normal {:.6e} {:.6e} {:.6e}\n    outer loop",
            n[0], n[1], n[2]
        )?;
        for p in tri {
            writeln!(f, "      vertex {:.6e} {:.6e} {:.6e}", p[0], p[1], p[2])?;
        }
        writeln!(f, "    endloop\n  endfacet")?;
    }
    writeln!(f, "endsolid torus")?;
    f.flush()
}

fn write_ply(t: &Torus, path: &Path, binary: bool) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let format = if binary { "binary_little_endian" } else { "ascii" };
    let verts = (t.rows + 1) * (t.cols + 1);
    write!(
        f,
        "ply\nformat {format} 1.0\nelement vertex {verts}\nproperty float x\nproperty float y\nproperty float z\n\
         property float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\n\
         element face {}\nproperty list uchar int vertex_indices\nend_header\n",
        t.rows * t.cols
    )?;
    for (p, n, uv) in t.vertices() {
        let all = [p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]];
        if binary {
            for c in all {
                f.write_all(&c.to_le_bytes())?;
            }
        } else {
            let line: Vec<String> = all.iter().map(|c| format!("{c:.6}")).collect();
            writeln!(f, "{}", line.join(" "))?;
        }
    }
    for q in t.quads() {
        if binary {
            f.write_all(&[4])?;
            for k in q {
                f.write_all(&(k as i32).to_le_bytes())?;
            }
        } else {
            writeln!(f, "4 {} {} {} {}", q[0], q[1], q[2], q[3])?;
        }
    }
    f.flush()
}

/// The torus as Polysmelt writes a GLB: positions, normals, UVs and u32 indices.
fn write_glb(t: &Torus, path: &Path) -> std::io::Result<()> {
    let mut prim = polysmelt::mesh::Prim::default();
    for (p, n, uv) in t.vertices() {
        prim.positions.push(p);
        prim.normals.push(n);
        prim.uvs.push(uv);
    }
    for [a, b, c, d] in t.quads() {
        prim.indices.extend([a, b, c, a, c, d].map(|i| i as u32));
    }
    let mesh = polysmelt::mesh::Mesh {
        prims: vec![prim],
        materials: vec![polysmelt::mesh::Material::plain("Torus", [0.8, 0.8, 0.8, 1.0])],
        images: Vec::new(),
        skeleton: Vec::new(),
    };
    std::fs::write(path, mesh.to_glb().map_err(std::io::Error::other)?)
}

/// The torus skinned to a ring of 32 bones, each vertex weighted between the two nearest: what
/// a character's GLB asks of a loader, posed.
fn write_glb_skinned(t: &Torus, path: &Path) -> std::io::Result<()> {
    const BONES: usize = 32;
    let mut prim = polysmelt::mesh::Prim::default();
    for (i, (p, n, uv)) in (0..=t.rows)
        .flat_map(|i| (0..=t.cols).map(move |_| i))
        .zip(t.vertices())
    {
        prim.positions.push(p);
        prim.normals.push(n);
        prim.uvs.push(uv);
        let at = i as f32 / t.rows as f32 * BONES as f32;
        let k = (at.floor() as usize).min(BONES - 1);
        let w = at - k as f32;
        prim.skin_joints.push([k as u32, ((k + 1) % BONES) as u32, 0, 0]);
        prim.skin_weights.push([1.0 - w, w, 0.0, 0.0]);
    }
    for [a, b, c, d] in t.quads() {
        prim.indices.extend([a, b, c, a, c, d].map(|i| i as u32));
    }
    let skeleton = (0..BONES)
        .map(|k| {
            let a = k as f32 / BONES as f32 * std::f32::consts::TAU;
            let (s, c) = a.sin_cos();
            polysmelt::mesh::Bone {
                name: Some(format!("bone{k}")),
                parent: (k > 0).then_some(0),
                node: k,
                matrix: [c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, c, s, 0.0, 1.0],
            }
        })
        .collect();
    let mesh = polysmelt::mesh::Mesh {
        prims: vec![prim],
        materials: vec![polysmelt::mesh::Material::plain("Torus", [0.8, 0.8, 0.8, 1.0])],
        images: Vec::new(),
        skeleton,
    };
    std::fs::write(path, mesh.to_glb().map_err(std::io::Error::other)?)
}

struct Input {
    name: String,
    path: PathBuf,
    ext: &'static str,
    triangles: usize,
    bytes: u64,
}

fn inputs(dir: &Path, sizes: &[String]) -> Vec<Input> {
    std::fs::create_dir_all(dir).unwrap();
    let mut out = Vec::new();
    for size in sizes {
        let t = match size.as_str() {
            "small" => Torus { rows: 50, cols: 100 },
            "medium" => Torus { rows: 250, cols: 400 },
            "large" => Torus { rows: 1000, cols: 1000 },
            other => panic!("unknown size {other:?}: small, medium or large"),
        };
        type Writer = fn(&Torus, &Path) -> std::io::Result<()>;
        let kinds: [(&str, &str, Writer); 7] = [
            ("obj", "obj", write_obj),
            ("stl binary", "stl", write_stl_binary),
            ("stl ascii", "stl", write_stl_ascii),
            ("ply binary", "ply", |t, p| write_ply(t, p, true)),
            ("ply ascii", "ply", |t, p| write_ply(t, p, false)),
            ("glb", "glb", write_glb),
            ("glb skinned", "glb", write_glb_skinned),
        ];
        for (kind, ext, write) in kinds {
            let path = dir.join(format!("{size}_{}.{ext}", kind.replace(' ', "_")));
            write(&t, &path).unwrap();
            let bytes = std::fs::metadata(&path).unwrap().len();
            out.push(Input {
                name: format!("{kind}, {size}"),
                path,
                ext,
                triangles: t.triangles(),
                bytes,
            });
        }
    }
    out
}

// --- Loaders --------------------------------------------------------------------------------

/// A loader: reads the file and returns its result, boxed so its heap stays live until measured, and the
/// number of triangles it holds.
type Load = fn(&Input) -> (Box<dyn std::any::Any>, usize);

fn polysmelt(input: &Input) -> (Box<dyn std::any::Any>, usize) {
    use polysmelt::document::{Keep, Pose};
    let dir = input.path.parent().unwrap();
    let doc =
        polysmelt::formats::load(&input.path, input.ext, &polysmelt::assets::Folder::new(dir), Keep::ALL).unwrap();
    // A GLB is already the document, its arrays left in the file: the triangles drawn, decoded
    // (and skinned), are the like-for-like result (the gltf crate decodes them, unposed). The other formats' document
    // holds their arrays, as their competitors' results do.
    if input.ext == "glb" {
        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        let n = mesh.triangles();
        return (Box::new((doc, mesh)), n);
    }
    let n = doc.triangles();
    (Box::new(doc), n)
}

/// Polysmelt's skinned meshes left in their bind pose (`Pose::Bind`): their vertices as stored,
/// unposed, as the gltf crate gives them.
fn polysmelt_bind(input: &Input) -> (Box<dyn std::any::Any>, usize) {
    use polysmelt::document::{Keep, Pose};
    let dir = input.path.parent().unwrap();
    let doc =
        polysmelt::formats::load(&input.path, input.ext, &polysmelt::assets::Folder::new(dir), Keep::ALL).unwrap();
    let mesh = doc.flatten(Pose::Bind).unwrap().unwrap();
    let n = mesh.triangles();
    (Box::new((doc, mesh)), n)
}

/// The name the baseline commit's Polysmelt goes by in the tables.
#[cfg(feature = "baseline")]
const BASELINE: &str = "polysmelt @ base";

#[cfg(all(feature = "baseline", feature = "base-document"))]
fn polysmelt_base(input: &Input) -> (Box<dyn std::any::Any>, usize) {
    use polysmelt_base::document::{Keep, Pose};
    let dir = input.path.parent().unwrap();
    let doc = polysmelt_base::formats::load(
        &input.path,
        input.ext,
        &polysmelt_base::assets::Folder::new(dir),
        Keep::ALL,
    )
    .unwrap();
    if input.ext == "glb" {
        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        let n = mesh.triangles();
        return (Box::new((doc, mesh)), n);
    }
    let n = doc.triangles();
    (Box::new(doc), n)
}

#[cfg(all(feature = "baseline", not(feature = "base-document")))]
fn polysmelt_base(input: &Input) -> (Box<dyn std::any::Any>, usize) {
    let dir = input.path.parent().unwrap();
    let loaded =
        polysmelt_base::formats::load(&input.path, input.ext, &polysmelt_base::assets::Folder::new(dir)).unwrap();
    let mesh = loaded.mesh.unwrap();
    let n = mesh.triangles();
    (Box::new(mesh), n)
}

fn tobj(input: &Input) -> (Box<dyn std::any::Any>, usize) {
    // GPU_LOAD_OPTIONS: triangulated, one index per vertex. The form a renderer or a GLB needs, like Polysmelt's.
    let (models, _) = tobj::load_obj(&input.path, &tobj::GPU_LOAD_OPTIONS).unwrap();
    let n = models.iter().map(|m| m.mesh.indices.len() / 3).sum();
    (Box::new(models), n)
}

fn stl_io(input: &Input) -> (Box<dyn std::any::Any>, usize) {
    let mut f = BufReader::new(File::open(&input.path).unwrap());
    let mesh = stl_io::read_stl(&mut f).unwrap();
    let n = mesh.faces.len();
    (Box::new(mesh), n)
}

mod ply {
    use ply_rs::parser::Parser;
    use ply_rs::ply::{DefaultElement, Property, PropertyAccess};

    /// The typed path ply-rs recommends for speed: each element read straight into a struct.
    #[derive(Default)]
    pub struct Vertex {
        pub p: [f32; 3],
        pub n: [f32; 3],
        pub uv: [f32; 2],
    }

    impl PropertyAccess for Vertex {
        fn new() -> Self {
            Self::default()
        }
        fn set_property(&mut self, key: String, property: Property) {
            let Property::Float(v) = property else { return };
            match key.as_str() {
                "x" => self.p[0] = v,
                "y" => self.p[1] = v,
                "z" => self.p[2] = v,
                "nx" => self.n[0] = v,
                "ny" => self.n[1] = v,
                "nz" => self.n[2] = v,
                "s" => self.uv[0] = v,
                "t" => self.uv[1] = v,
                _ => {}
            }
        }
    }

    #[derive(Default)]
    pub struct Face(pub Vec<u32>);

    impl PropertyAccess for Face {
        fn new() -> Self {
            Self::default()
        }
        fn set_property(&mut self, key: String, property: Property) {
            if key == "vertex_indices" {
                self.0 = match property {
                    Property::ListInt(v) => v.into_iter().map(|i| i as u32).collect(),
                    Property::ListUInt(v) => v,
                    _ => return,
                };
            }
        }
    }

    pub fn typed(input: &super::Input) -> (Box<dyn std::any::Any>, usize) {
        let mut f = super::BufReader::new(super::File::open(&input.path).unwrap());
        let (vp, fp) = (Parser::<Vertex>::new(), Parser::<Face>::new());
        let header = vp.read_header(&mut f).unwrap();
        let (mut verts, mut faces) = (Vec::new(), Vec::new());
        for el in header.elements.values() {
            match el.name.as_str() {
                "vertex" => verts = vp.read_payload_for_element(&mut f, el, &header).unwrap(),
                "face" => faces = fp.read_payload_for_element(&mut f, el, &header).unwrap(),
                _ => {}
            }
        }
        let n = faces.iter().map(|f| f.0.len().saturating_sub(2)).sum();
        (Box::new((verts, faces)), n)
    }

    /// `read_ply` with ply-rs's generic elements (a map of properties per vertex): the quick-start path.
    pub fn generic(input: &super::Input) -> (Box<dyn std::any::Any>, usize) {
        let mut f = super::BufReader::new(super::File::open(&input.path).unwrap());
        let ply = Parser::<DefaultElement>::new().read_ply(&mut f).unwrap();
        let n = ply.payload["face"]
            .iter()
            .map(|face| match &face["vertex_indices"] {
                Property::ListInt(v) => v.len().saturating_sub(2),
                Property::ListUInt(v) => v.len().saturating_sub(2),
                _ => 0,
            })
            .sum();
        (Box::new(ply), n)
    }
}

/// The `gltf` crate as its documentation uses it: `import` (the document and its buffers), then each
/// primitive's attributes and indices read into vectors, the arrays a renderer uploads.
fn gltf_import(input: &Input) -> (Box<dyn std::any::Any>, usize) {
    // Held, not read: the arrays a renderer would upload, kept alive to be measured.
    #[allow(dead_code)]
    struct Prim {
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
        joints: Vec<[u16; 4]>,
        weights: Vec<[f32; 4]>,
    }
    let (doc, buffers, images) = gltf::import(&input.path).unwrap();
    let mut prims = Vec::new();
    for mesh in doc.meshes() {
        for p in mesh.primitives() {
            let r = p.reader(|b| Some(&buffers[b.index()]));
            prims.push(Prim {
                positions: r.read_positions().map(Iterator::collect).unwrap_or_default(),
                normals: r.read_normals().map(Iterator::collect).unwrap_or_default(),
                uvs: r.read_tex_coords(0).map(|t| t.into_f32().collect()).unwrap_or_default(),
                indices: r.read_indices().map(|i| i.into_u32().collect()).unwrap_or_default(),
                joints: r.read_joints(0).map(|j| j.into_u16().collect()).unwrap_or_default(),
                weights: r.read_weights(0).map(|w| w.into_f32().collect()).unwrap_or_default(),
            });
        }
    }
    let ibms: Vec<Vec<[[f32; 4]; 4]>> = doc
        .skins()
        .map(|s| {
            s.reader(|b| Some(&buffers[b.index()]))
                .read_inverse_bind_matrices()
                .map(Iterator::collect)
                .unwrap_or_default()
        })
        .collect();
    let n = prims.iter().map(|p| p.indices.len() / 3).sum();
    (Box::new((doc, buffers, images, prims, ibms)), n)
}

fn competitors(input: &Input) -> Vec<(&'static str, Load)> {
    let mut out: Vec<(&'static str, Load)> = vec![("polysmelt", polysmelt)];
    if input.name.contains("skinned") {
        out.push(("polysmelt (bind pose)", polysmelt_bind));
    }
    let ext = input.ext;
    #[cfg(feature = "baseline")]
    out.push((BASELINE, polysmelt_base));
    match ext {
        "obj" => out.push(("tobj", tobj)),
        "stl" => out.push(("stl_io", stl_io)),
        "glb" => out.push(("gltf", gltf_import)),
        "ply" => {
            out.push(("ply-rs (typed)", ply::typed));
            out.push(("ply-rs (generic)", ply::generic));
        }
        _ => {}
    }
    out
}

// --- Measuring ------------------------------------------------------------------------------

struct Sample {
    time: Duration,
    peak: usize,
    kept: usize,
}

fn measure(load: Load, input: &Input) -> Sample {
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    let start = Instant::now();
    let (result, triangles) = load(input);
    let time = start.elapsed();
    let (peak, kept) = (PEAK.load(Relaxed) - base, CURRENT.load(Relaxed).saturating_sub(base));
    drop(std::hint::black_box(result));
    assert_eq!(triangles, input.triangles, "{} read {triangles} triangles", input.name);
    Sample { time, peak, kept }
}

struct Row {
    loader: &'static str,
    median: Duration,
    best: Duration,
    rounds: usize,
    peak: usize,
    kept: usize,
}

fn run(input: &Input, budget: Duration) -> Vec<Row> {
    let loaders = competitors(input);
    // A warm-up round, which also gives the memory figures (identical every round).
    let first: Vec<Sample> = loaders.iter().map(|(_, l)| measure(*l, input)).collect();
    let mut times: Vec<Vec<Duration>> = loaders.iter().map(|_| Vec::new()).collect();
    let start = Instant::now();
    let mut round = 0;
    while round < 5 || (start.elapsed() < budget && round < 200) {
        for k in 0..loaders.len() {
            let i = (k + round) % loaders.len();
            times[i].push(measure(loaders[i].1, input).time);
        }
        round += 1;
    }
    loaders
        .iter()
        .zip(first)
        .zip(times)
        .map(|((&(loader, _), s), mut t)| {
            t.sort();
            Row {
                loader,
                median: t[t.len() / 2],
                best: t[0],
                rounds: t.len(),
                peak: s.peak,
                kept: s.kept,
            }
        })
        .collect()
}

// --- Reporting ------------------------------------------------------------------------------

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn mb(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let sizes: Vec<String> = arg("--sizes")
        .unwrap_or("small,medium,large".into())
        .split(',')
        .map(|s| s.trim().to_owned())
        .collect();
    let budget = Duration::from_secs_f64(arg("--budget").map_or(3.0, |b| b.parse().expect("--budget seconds")));
    let dir = std::env::temp_dir().join("polysmelt-bench");

    let inputs = inputs(&dir, &sizes);
    let mut md = String::new();
    let mut results = Vec::new();
    let (mut fastest, mut leanest) = (0, 0);
    writeln!(md, "## Polysmelt load benchmarks\n").unwrap();
    writeln!(
        md,
        "Median and fastest of interleaved rounds ({budget:?} budget per input); peak heap during the load and \
         heap the result keeps, from a counting allocator. `×` is relative to Polysmelt: above 1 means Polysmelt wins.\n"
    )
    .unwrap();
    for input in &inputs {
        eprintln!(
            "{} ({:.1} MB, {} triangles)",
            input.name,
            mb(input.bytes as usize),
            input.triangles
        );
        let rows = run(input, budget);
        let me = &rows[0];
        // Only other libraries count: the baseline and Polysmelt's own variants (bind pose) are not competitors.
        let others = || rows[1..].iter().filter(|r| !r.loader.starts_with("polysmelt"));
        if others().all(|r| me.median <= r.median) {
            fastest += 1;
        }
        if others().all(|r| me.peak <= r.peak) {
            leanest += 1;
        }
        writeln!(
            md,
            "### {} — {:.1} MB, {} triangles\n\n| Loader | Median ms | Fastest ms | MB/s | × time | Peak MB | × peak | Kept MB | Rounds |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            input.name,
            mb(input.bytes as usize),
            input.triangles
        )
        .unwrap();
        for r in &rows {
            writeln!(
                md,
                "| {} | {:.2} | {:.2} | {:.0} | {:.2} | {:.1} | {:.2} | {:.1} | {} |",
                if r.loader == "polysmelt" {
                    "**polysmelt**"
                } else {
                    r.loader
                },
                ms(r.median),
                ms(r.best),
                mb(input.bytes as usize) / r.median.as_secs_f64(),
                r.median.as_secs_f64() / me.median.as_secs_f64(),
                mb(r.peak),
                r.peak as f64 / me.peak as f64,
                mb(r.kept),
                r.rounds
            )
            .unwrap();
            results.push(json!({
                "input": input.name, "bytes": input.bytes, "triangles": input.triangles, "loader": r.loader,
                "median_ns": r.median.as_nanos() as u64, "fastest_ns": r.best.as_nanos() as u64,
                "rounds": r.rounds, "peak_bytes": r.peak, "kept_bytes": r.kept,
            }));
        }
        md.push('\n');
    }
    writeln!(
        md,
        "**Polysmelt is fastest on {fastest} of {n} inputs and leanest on {leanest} of {n}.**",
        n = inputs.len()
    )
    .unwrap();

    println!("{md}");
    if let Some(p) = arg("--markdown") {
        std::fs::write(p, &md).unwrap();
    }
    if let Some(p) = arg("--json") {
        let doc = json!({ "os": std::env::consts::OS, "arch": std::env::consts::ARCH, "results": results });
        std::fs::write(p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}
