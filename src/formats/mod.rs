//! Model formats read natively: STL (binary and ASCII), PLY (ASCII and binary), `MagicaVoxel`
//! `.vox`, OBJ (with its MTL materials and textures), FBX (binary and ASCII), 3DS, 3MF, USD,
//! glTF (`.gltf` and `.glb`) and `SketchUp` (`.skp`).
//! Each becomes a [`Document`]: glTF as it is, the others converted; textures are embedded so the
//! document stands on its own. Each format is behind a cargo feature of its name (`3ds` and
//! `3mf` for those two). A [`Keep`] says what to keep; by default, everything.
//!
//! [`load_bytes`] reads a model already in memory, with the files it refers to found through
//! [`Assets`] (such as [`Memory`](crate::assets::Memory)). [`load`] reads a file, and needs the
//! `fs` feature.

use std::borrow::Cow;
use std::io;
use std::path::Path;

use serde_json::json;

use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};

#[cfg(feature = "fbx")]
mod fbx;
#[cfg(feature = "fbx")]
pub mod fbx_ascii;
#[cfg(feature = "fbx")]
mod fbx_inflate;
#[cfg(feature = "gltf")]
mod gltf;
#[cfg(feature = "gltf")]
mod gltf1;
#[cfg(feature = "obj")]
mod obj;
#[cfg(feature = "ply")]
mod ply;
#[cfg(feature = "skp")]
mod skp;
#[cfg(feature = "stl")]
mod stl;
#[cfg(any(feature = "stl", feature = "ply", feature = "obj"))]
#[cfg_attr(
    not(all(feature = "obj", feature = "ply", feature = "stl")),
    allow(dead_code, reason = "each text reader uses its own part")
)]
mod stream;
#[cfg(feature = "3ds")]
mod tds;
#[cfg(feature = "3mf")]
mod threemf;
#[cfg(feature = "usd")]
mod usd;
#[cfg(feature = "vox")]
mod vox;
#[cfg(any(feature = "fbx", feature = "usd", feature = "3ds"))]
mod xform;

/// The extensions [`load`] handles, whether or not this build has their feature.
pub const EXTENSIONS: &[&str] = &[
    "stl", "ply", "vox", "obj", "fbx", "3ds", "3mf", "usd", "usda", "usdc", "usdz", "gltf", "glb", "skp",
];

/// Reads the model at `path`, whose format is named by `ext` (lowercase, without the dot),
/// keeping what `keep` says.
///
/// # Errors
///
/// When the file can't be read, `ext` names no reader (or one whose feature is off), or the
/// model doesn't parse.
#[cfg(feature = "fs")]
pub fn load(path: &Path, ext: &str, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    read(Input::file(path), ext, assets, keep).map(Document::into_owned)
}

/// Reads a model already in memory, whose format is named by `ext` (lowercase, without the
/// dot), keeping what `keep` says. `name` is the model's path among `assets`: the files it
/// refers to (MTL libraries, textures, glTF buffers, USD layers) are found from its folder and
/// read through [`Assets::read`]. A model that refers to nothing reads with any name and an
/// empty [`Memory`](crate::assets::Memory). The document may borrow from `bytes` and `assets`
/// (a GLB's binary chunk is used in place); [`Document::into_owned`] makes it outlive them.
///
/// # Errors
///
/// When `ext` names no reader (or one whose feature is off) or the model doesn't parse.
pub fn load_bytes<'a>(
    bytes: &'a [u8],
    name: &'a str,
    ext: &str,
    assets: &'a dyn Assets,
    keep: Keep,
) -> Result<Document<'a>> {
    read(Input::memory(bytes, name), ext, assets, keep)
}

/// A model to read: its path (or, in memory, its name), and where its bytes are.
#[derive(Clone, Copy)]
pub(crate) struct Input<'a> {
    /// The model's path or name: references are resolved from its folder.
    pub path: &'a Path,
    data: Data<'a>,
}

#[derive(Clone, Copy)]
enum Data<'a> {
    /// The file at the path, read as needed.
    #[cfg(feature = "fs")]
    File,
    /// The whole model, in memory.
    Bytes(&'a [u8]),
}

/// A stream that seeks, for zip archives.
#[cfg(any(feature = "3mf", feature = "usd"))]
pub(crate) trait ReadSeek: io::Read + io::Seek {}

#[cfg(any(feature = "3mf", feature = "usd"))]
impl<T: io::Read + io::Seek> ReadSeek for T {}

#[allow(dead_code)]
impl<'a> Input<'a> {
    /// The model in the file at `path`.
    #[cfg(feature = "fs")]
    pub fn file(path: &'a Path) -> Self {
        Self { path, data: Data::File }
    }

    /// A model in memory, named `name`.
    pub fn memory(bytes: &'a [u8], name: &'a str) -> Self {
        Self {
            path: Path::new(name),
            data: Data::Bytes(bytes),
        }
    }

    /// The model's folder.
    pub fn dir(&self) -> &'a Path {
        self.path.parent().unwrap_or(Path::new(""))
    }

    /// The model's bytes, when it's in memory.
    #[cfg_attr(
        not(feature = "fs"),
        allow(clippy::unnecessary_wraps, reason = "only reading a file fails")
    )]
    pub fn bytes(&self) -> Option<&'a [u8]> {
        match self.data {
            Data::Bytes(b) => Some(b),
            #[cfg(feature = "fs")]
            Data::File => None,
        }
    }

    /// The whole model.
    #[cfg_attr(
        not(feature = "fs"),
        allow(clippy::unnecessary_wraps, reason = "only reading a file fails")
    )]
    pub fn read(&self) -> io::Result<Cow<'a, [u8]>> {
        match self.data {
            Data::Bytes(b) => Ok(Cow::Borrowed(b)),
            #[cfg(feature = "fs")]
            Data::File => std::fs::read(self.path).map(Cow::Owned),
        }
    }

    /// Up to `n` bytes from the start of the model.
    #[cfg_attr(
        not(feature = "fs"),
        allow(clippy::unnecessary_wraps, reason = "only reading a file fails")
    )]
    pub fn head(&self, n: usize) -> io::Result<Cow<'a, [u8]>> {
        match self.data {
            Data::Bytes(b) => Ok(Cow::Borrowed(&b[..b.len().min(n)])),
            #[cfg(feature = "fs")]
            Data::File => {
                use std::io::Read as _;
                let mut head = Vec::new();
                std::fs::File::open(self.path)?.take(n as u64).read_to_end(&mut head)?;
                Ok(Cow::Owned(head))
            }
        }
    }

    /// The model a block at a time (in memory, in place).
    #[cfg(any(feature = "stl", feature = "ply", feature = "obj"))]
    #[cfg_attr(
        not(feature = "fs"),
        allow(clippy::unnecessary_wraps, reason = "only reading a file fails")
    )]
    pub fn blocks(&self) -> io::Result<stream::Blocks<'a>> {
        match self.data {
            Data::Bytes(b) => Ok(stream::Blocks::slice(b)),
            #[cfg(feature = "fs")]
            Data::File => {
                let file = std::fs::File::open(self.path)?;
                let size = file.metadata().map_or(0, |m| m.len());
                Ok(stream::Blocks::reader(file, size))
            }
        }
    }

    /// The model as a stream that seeks.
    #[cfg(any(feature = "3mf", feature = "usd"))]
    #[cfg_attr(
        not(feature = "fs"),
        allow(clippy::unnecessary_wraps, reason = "only reading a file fails")
    )]
    pub fn seekable(&self) -> io::Result<Box<dyn ReadSeek + 'a>> {
        match self.data {
            Data::Bytes(b) => Ok(Box::new(io::Cursor::new(b))),
            #[cfg(feature = "fs")]
            Data::File => Ok(Box::new(std::fs::File::open(self.path)?)),
        }
    }
}

fn read<'a>(input: Input<'a>, ext: &str, assets: &'a dyn Assets, keep: Keep) -> Result<Document<'a>> {
    let mut doc = convert(input, ext, assets, keep)?;
    doc.prune(keep);
    if matches!(
        ext,
        "gltf" | "glb" | "fbx" | "3ds" | "usd" | "usda" | "usdc" | "usdz" | "skp"
    ) {
        let c = doc.counts();
        for (key, n) in [("triangles", c.triangles), ("points", c.points), ("lines", c.lines)] {
            if n > 0 || key == "triangles" {
                doc.facts.insert(key.into(), json!(n));
            }
        }
    }
    Ok(doc)
}

fn convert<'a>(input: Input<'a>, ext: &str, assets: &'a dyn Assets, keep: Keep) -> Result<Document<'a>> {
    let _ = (input, assets, keep);
    match ext {
        #[cfg(feature = "stl")]
        "stl" => stl::stl(input),
        #[cfg(feature = "ply")]
        "ply" => ply::ply(input, assets, keep),
        #[cfg(feature = "vox")]
        "vox" => vox::vox(&input.read()?, keep),
        #[cfg(feature = "3mf")]
        "3mf" => threemf::load(input, keep),
        #[cfg(feature = "3ds")]
        "3ds" => tds::load(input, assets, keep),
        #[cfg(feature = "usd")]
        "usd" | "usda" | "usdc" | "usdz" => usd::load(input, assets, keep),
        #[cfg(feature = "obj")]
        "obj" => obj::obj(input, assets, keep),
        #[cfg(feature = "fbx")]
        "fbx" => fbx::load(input, assets, keep),
        #[cfg(feature = "gltf")]
        "gltf" | "glb" => gltf::load(input, assets, keep),
        #[cfg(feature = "skp")]
        "skp" => skp::load(input, keep),
        e if EXTENSIONS.contains(&e) => bail!("built without the `{}` feature", feature_for(e)),
        _ => bail!("not a model format handled here: .{ext}"),
    }
}

/// The cargo feature that reads extension `ext`.
fn feature_for(ext: &str) -> &str {
    match ext {
        "usda" | "usdc" | "usdz" => "usd",
        "glb" => "gltf",
        e => e,
    }
}

#[cfg(feature = "obj")]
/// Whether the file's material colors are linear values (Blender's OBJ and FBX exporters write them so) rather
/// than the sRGB other tools write.
fn linear_colors(input: Input<'_>, ext: &str) -> bool {
    let Ok(head) = input.head(65_536) else {
        return false;
    };
    if ext == "obj" {
        let text: String = String::from_utf8_lossy(&head).chars().take(500).collect();
        text.lines().any(|l| l.starts_with("# Blender"))
    } else {
        head.windows(7).any(|w| w == b"Blender")
    }
}

/// The color texture by naming convention for a model at `path` that names none, added to
/// `images`.
#[cfg(any(feature = "obj", feature = "3ds", feature = "fbx"))]
fn color_texture(path: &Path, assets: &dyn Assets, images: &mut crate::images::Images) -> Option<usize> {
    let p = assets.color_texture_for(path)?;
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    images.file(assets, p, &name)
}

/// Whether the colors of the materials a texture found by name would paint are plain enough to
/// be painted over: no color but grays, and no more than one besides white. A model colored by
/// its materials (a white body with orange trim, a dark tire on a lighter rim) keeps its colors.
#[cfg(any(feature = "obj", feature = "3ds", feature = "fbx"))]
fn plain_colors(colors: impl IntoIterator<Item = [f32; 4]>) -> bool {
    let mut other: Option<[f32; 3]> = None;
    for [r, g, b, _] in colors {
        if r.max(g).max(b) - r.min(g).min(b) > 0.05 {
            return false;
        }
        if [r, g, b].iter().all(|c| *c >= 0.999) {
            continue;
        }
        match other {
            Some(o) if o.iter().zip([r, g, b]).any(|(a, c)| (a - c).abs() > 0.01) => return false,
            _ => other = Some([r, g, b]),
        }
    }
    true
}

#[cfg(any(feature = "obj", feature = "3ds"))]
/// A model that names no color texture takes one by name: `Barrel.png` or
/// `Barrel_Albedo.png` for `Barrel.obj`, from the model's part of the pack (see
/// [`Assets::color_texture_for`]), for the `materials` that `textured` says draw triangles with
/// texture coordinates, unless their own colors say the model is colored already (see
/// [`plain_colors`]). Their color then shows as painted.
fn guess_texture(
    materials: &mut [crate::mesh::Material],
    images: &mut crate::images::Images,
    textured: &[bool],
    path: &Path,
    assets: &dyn Assets,
) {
    if !textured.contains(&true) || materials.iter().any(|m| m.color_image.is_some()) {
        return;
    }
    let painted = materials
        .iter()
        .zip(textured)
        .filter(|(_, t)| **t)
        .map(|(m, _)| m.color);
    if !plain_colors(painted) {
        return;
    }
    let Some(i) = color_texture(path, assets, images) else {
        return;
    };
    for (m, _) in materials.iter_mut().zip(textured).filter(|(_, t)| **t) {
        m.color_image = Some(i);
        m.color[..3].copy_from_slice(&[1.0, 1.0, 1.0]);
    }
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;
    use crate::assets::{Folder, Memory};
    use crate::mesh::Mesh;
    #[allow(unused_imports)]
    use crate::mesh::{GRAY, to_glb};
    use serde_json::{Map, Value};

    /// Reads `path` with assets found under `root`, as (mesh, facts); and reads it again from
    /// memory, with every file under `root` as a [`Memory`], to check the two agree.
    #[allow(dead_code)]
    fn load_in(path: &Path, ext: &str, root: &Path) -> (Option<Mesh>, Map<String, Value>) {
        let loaded = load(path, ext, &Folder::new(root), Keep::ALL).unwrap();
        let mut memory = Memory::new();
        let mut dirs = vec![root.to_path_buf()];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    dirs.push(p);
                } else {
                    let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
                    memory.insert(&rel, std::fs::read(&p).unwrap());
                }
            }
        }
        let name = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        let bytes = std::fs::read(path).unwrap();
        let in_memory = super::load_bytes(&bytes, &name, ext, &memory, Keep::ALL).unwrap();
        assert!(
            loaded.to_glb().unwrap() == in_memory.to_glb().unwrap(),
            "{name} reads differently from memory"
        );
        assert_eq!(loaded.facts, in_memory.facts, "{name} reads differently from memory");
        let mesh = loaded.flatten(crate::document::Pose::Rest).unwrap();
        (mesh, loaded.facts)
    }

    #[allow(dead_code)]
    fn load_bytes(name: &str, bytes: &[u8]) -> (Option<Mesh>, Map<String, Value>) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(name);
        std::fs::write(&p, bytes).unwrap();
        load_in(&p, name.rsplit('.').next().unwrap(), dir.path())
    }

    #[allow(dead_code)]
    fn png(rgb: [u8; 3]) -> Vec<u8> {
        let mut png = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb(rgb))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    #[allow(dead_code)]
    const QUAD: &str = "v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
";

    #[test]
    fn missing_feature_and_unknown_extension() {
        let err = |ext| {
            load(Path::new("x"), ext, &Folder::new("."), Keep::ALL)
                .unwrap_err()
                .to_string()
        };
        assert!(err("dae").contains("not a model format"));
        #[cfg(not(feature = "usd"))]
        assert!(err("usdz").contains("without the `usd` feature"));
        #[cfg(not(feature = "gltf"))]
        assert!(err("glb").contains("without the `gltf` feature"));
    }

    #[test]
    #[cfg(feature = "stl")]
    fn stl_binary_and_ascii() {
        let mut bin = vec![0u8; 80];
        bin.extend_from_slice(&1u32.to_le_bytes());
        for v in [0f32, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        bin.extend_from_slice(&[0, 0]);
        let (m, f) = load_bytes("a.stl", &bin);
        assert_eq!(f["triangles"], 1);
        let glb = to_glb(&m.unwrap()).unwrap();
        assert!(glb.starts_with(b"glTF"));

        let ascii = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        let (m, _) = load_bytes("b.stl", ascii.as_bytes());
        assert_eq!(m.unwrap().triangles(), 1);
    }

    #[test]
    #[cfg(feature = "stl")]
    fn stl_facet_colors() {
        let stl = |header: &[u8], attrs: &[u16]| {
            let mut bin = header.to_vec();
            bin.resize(80, 0);
            bin.extend_from_slice(&u32::try_from(attrs.len()).unwrap().to_le_bytes());
            for &a in attrs {
                for v in [0f32, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.] {
                    bin.extend_from_slice(&v.to_le_bytes());
                }
                bin.extend_from_slice(&a.to_le_bytes());
            }
            load_bytes("c.stl", &bin).0.unwrap()
        };
        // Magics: red in the low bits; a set top bit takes the header's color.
        let m = stl(b"COLOR=\x00\xff\x00\xff", &[0x001F, 0x8000, 0x001F]);
        assert_eq!(m.prims.len(), 2);
        assert_eq!(m.prims[0].indices.len(), 6);
        assert_eq!(m.materials[0].color, [1.0, 0.0, 0.0, 1.0]);
        let m = stl(b"COLOR=\x00\x00\xff\xff", &[0x03E0, 0x8000]);
        assert_eq!(m.materials[0].color, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(m.materials[1].color, [0.0, 0.0, 1.0, 1.0]);
        // VisCAM: the top bit marks a colored facet, blue in the low bits.
        let m = stl(b"binary", &[0x801F, 0xFC00, 0]);
        assert_eq!(
            m.materials.iter().map(|m| m.color).collect::<Vec<_>>(),
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0], GRAY]
        );
        // One color without a header is as likely junk as a color: plain gray.
        let m = stl(b"binary", &[0xFFFF, 0xFFFF]);
        assert_eq!((m.prims.len(), m.materials[0].color), (1, GRAY));
    }

    #[test]
    #[cfg(feature = "ply")]
    fn ply_ascii_quad_and_point_cloud() {
        let ply = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
                   element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n";
        let (m, f) = load_bytes("q.ply", ply.as_bytes());
        assert_eq!(f["triangles"], 2, "a quad fans into two triangles");
        assert_eq!(m.unwrap().prims[0].normals[0], [0.0, 0.0, 1.0]);

        let mut bin = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n".to_vec();
        for v in [0f32, 0., 0., 1., 2., 3.] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let (m, f) = load_bytes("p.ply", &bin);
        assert!(m.is_none());
        assert_eq!(f["points"], 2);
    }

    #[test]
    #[cfg(feature = "ply")]
    fn ply_nan_index_drops_its_face() {
        let ply = "ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
                   element face 2
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
                   3 0 1 2
3 nan 2 3
";
        let (_, f) = load_bytes("n.ply", ply.as_bytes());
        assert_eq!(f["triangles"], 1, "a NaN index is no vertex, not vertex 0");
    }

    #[test]
    #[cfg(feature = "vox")]
    fn vox_exposed_faces_by_color() {
        // Two adjacent voxels with different colors: 12 faces minus the 2 touching ones.
        let chunk = |id: &[u8], body: &[u8]| {
            let mut c = id.to_vec();
            c.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
            c.extend_from_slice(&0u32.to_le_bytes());
            c.extend_from_slice(body);
            c
        };
        let mut size = Vec::new();
        for v in [2u32, 1, 1] {
            size.extend_from_slice(&v.to_le_bytes());
        }
        let mut xyzi = 2u32.to_le_bytes().to_vec();
        xyzi.extend_from_slice(&[0, 0, 0, 1, 1, 0, 0, 2]);
        let mut rgba = vec![0u8; 1024];
        rgba[..8].copy_from_slice(&[255, 0, 0, 255, 0, 0, 255, 255]);
        let children = [chunk(b"SIZE", &size), chunk(b"XYZI", &xyzi), chunk(b"RGBA", &rgba)].concat();
        let mut file = b"VOX ".to_vec();
        file.extend_from_slice(&150u32.to_le_bytes());
        file.extend_from_slice(b"MAIN");
        file.extend_from_slice(&0u32.to_le_bytes());
        file.extend_from_slice(&u32::try_from(children.len()).unwrap().to_le_bytes());
        file.extend(children);
        let (m, f) = load_bytes("v.vox", &file);
        let m = m.unwrap();
        assert_eq!(f["voxels"], 2);
        assert_eq!(m.prims.len(), 2);
        assert_eq!(m.triangles(), 10 * 2);
        assert_eq!(m.materials[m.prims[0].material].color, [1.0, 0.0, 0.0, 1.0]);
        assert!(to_glb(&m).unwrap().starts_with(b"glTF"));
    }

    #[test]
    #[cfg(feature = "obj")]
    fn obj_mixed_faces_negative_indices_and_quads() {
        let dir = tempfile::tempdir().unwrap();
        // A quad with UVs and normals by negative index, then a triangle without UVs (a cap).
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n\
                   f -4/-4/-1 -3/-3/-1 -2/-2/-1 -1/-1/-1\nf 1 2 3\n";
        std::fs::write(dir.path().join("m.obj"), obj).unwrap();
        let (m, f) = load_in(&dir.path().join("m.obj"), "obj", dir.path());
        let m = m.unwrap();
        assert_eq!(f["triangles"], 3);
        let p = &m.prims[0];
        assert_eq!(
            p.uvs.len(),
            p.positions.len(),
            "every vertex gets a UV, missing ones zero"
        );
        assert_eq!(p.normals.len(), p.positions.len());
    }
    #[test]
    #[cfg(feature = "obj")]
    fn obj_texture_in_another_format_and_color_space() {
        let dir = tempfile::tempdir().unwrap();
        // The MTL names a TGA; the pack ships it as PNG.
        std::fs::write(dir.path().join("rock.png"), png([10, 20, 30])).unwrap();
        std::fs::write(
            dir.path().join("a.mtl"),
            "newmtl Rock\nKd 0.5 0.5 0.5\nmap_Kd rock.tga\nnewmtl Plain\nKd 0.5 0.5 0.5\n",
        )
        .unwrap();
        let obj = format!("mtllib a.mtl\n{QUAD}usemtl Rock\nf 1/1 2/2 3/3\nusemtl Plain\nf 1 3 4\n");
        std::fs::write(dir.path().join("a.obj"), &obj).unwrap();
        let (m, _) = load_in(&dir.path().join("a.obj"), "obj", dir.path());
        let m = m.unwrap();
        let rock = m.materials.iter().find(|x| x.name == "Rock").unwrap();
        assert_eq!(rock.color_image, Some(0));
        // sRGB 0.5 → linear ~0.214 for the glTF factor.
        let plain = m.materials.iter().find(|x| x.name == "Plain").unwrap();
        assert!((plain.color[0] - 0.214).abs() < 0.01, "{:?}", plain.color);

        // Blender writes linear values: kept as they are.
        std::fs::write(dir.path().join("a.obj"), format!("# Blender 4.2\n{obj}")).unwrap();
        let (m, _) = load_in(&dir.path().join("a.obj"), "obj", dir.path());
        let plain = m.unwrap().materials.into_iter().find(|x| x.name == "Plain").unwrap();
        assert!((plain.color[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "obj")]
    fn obj_without_textures_takes_one_by_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Textures")).unwrap();
        std::fs::write(dir.path().join("Textures/Barrel_Normal.png"), png([128, 128, 255])).unwrap();
        std::fs::write(dir.path().join("Textures/Barrel_Albedo.png"), png([200, 100, 50])).unwrap();
        std::fs::write(dir.path().join("Textures/Barrels.png"), png([0, 0, 0])).unwrap();
        std::fs::write(dir.path().join("Barrel.obj"), format!("{QUAD}f 1/1 2/2 3/3 4/4\n")).unwrap();
        let (m, _) = load_in(&dir.path().join("Barrel.obj"), "obj", dir.path());
        let m = m.unwrap();
        assert_eq!(m.images.len(), 1);
        assert_eq!(m.materials[0].color_image, Some(0));
        assert_eq!(m.materials[0].color, [1.0, 1.0, 1.0, 1.0]);
        let px = image::load_from_memory(&m.images[0].bytes).unwrap().to_rgb8();
        assert_eq!(px.get_pixel(0, 0).0, [200, 100, 50]);
    }

    #[test]
    #[cfg(feature = "obj")]
    fn a_texture_by_name_comes_from_the_models_part_of_the_pack() {
        let dir = tempfile::tempdir().unwrap();
        for d in ["Kit3D/Models", "Kit3D/Textures", "Kit2D"] {
            std::fs::create_dir_all(dir.path().join(d)).unwrap();
        }
        std::fs::write(dir.path().join("Kit2D/Stairs.png"), png([0, 0, 0])).unwrap();
        let model = dir.path().join("Kit3D/Models/Stairs.obj");
        std::fs::write(
            &model,
            format!(
                "{QUAD}f 1/1 2/2 3/3 4/4
"
            ),
        )
        .unwrap();
        let (m, _) = load_in(&model, "obj", dir.path());
        assert_eq!(m.unwrap().images.len(), 0, "a sprite from another kit isn't taken");

        std::fs::write(dir.path().join("Kit3D/Textures/Stairs.png"), png([200, 100, 50])).unwrap();
        let (m, _) = load_in(&model, "obj", dir.path());
        let m = m.unwrap();
        let px = image::load_from_memory(&m.images[0].bytes).unwrap().to_rgb8();
        assert_eq!(px.get_pixel(0, 0).0, [200, 100, 50]);
    }

    /// Writes each `(path, bytes)` under `root`, making folders.
    #[allow(dead_code)]
    fn put(root: &Path, files: &[(&str, &[u8])]) {
        for (rel, bytes) in files {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, bytes).unwrap();
        }
    }

    /// The color of the first texture a mesh's materials wear.
    #[allow(dead_code)]
    fn texture_color(m: &Mesh) -> Option<[u8; 3]> {
        let i = m.materials.iter().find_map(|x| x.color_image)?;
        Some(
            image::load_from_memory(&m.images[i].bytes)
                .unwrap()
                .to_rgb8()
                .get_pixel(0, 0)
                .0,
        )
    }

    #[test]
    #[cfg(feature = "obj")]
    fn a_texture_by_name_is_never_a_sprite_or_preview_beside_the_model() {
        // A Kenney kit: the 3D models, and 2D pictures of the same things beside them.
        let dir = tempfile::tempdir().unwrap();
        let obj = format!("{QUAD}f 1/1 2/2 3/3 4/4\n");
        put(
            dir.path(),
            &[
                ("Models/console.obj", obj.as_bytes()),
                ("sprites_isometric/console_NE.png", &png([255, 0, 0])),
                ("sprites_topdown/console.png", &png([0, 0, 255])),
                ("Previews/console.png", &png([0, 255, 0])),
            ],
        );
        let model = dir.path().join("Models/console.obj");
        let (m, _) = load_in(&model, "obj", dir.path());
        assert_eq!(texture_color(&m.unwrap()), None);

        put(
            dir.path(),
            &[
                ("Textures/console_Base.png", &png([1, 2, 3])),
                ("Textures/console.png", &png([200, 100, 50])),
            ],
        );
        let (m, _) = load_in(&model, "obj", dir.path());
        assert_eq!(texture_color(&m.unwrap()), Some([200, 100, 50]), "the exact name");
    }

    #[test]
    #[cfg(feature = "obj")]
    fn a_texture_named_by_another_path_is_the_one_nearest_the_model() {
        let dir = tempfile::tempdir().unwrap();
        let obj = format!("mtllib m.mtl\nusemtl m\n{QUAD}f 1/1 2/2 3/3 4/4\n");
        let red = png([255, 0, 0]);
        let blue = png([0, 0, 255]);
        put(
            dir.path(),
            &[
                ("a_kit/Models/m.obj", obj.as_bytes()),
                ("a_kit/Models/colormap.png", &red),
                ("a_kit/Textures/wood.png", &red),
                ("a_kit/Textures/stone.png", &red),
                ("a_kit/Models/stone.jpg", &red),
                ("z_kit/colormap.png", &blue),
                ("z_kit/Textures/wood.png", &blue),
                ("z_kit/Textures/stone.png", &blue),
                ("z_kit/Models/m.fbm/cobblestone.png", &blue),
                ("a_kit/Models/cobblestone.png", &red),
            ],
        );
        let model = dir.path().join("a_kit/Models/m.obj");
        for map in [
            "C:\\Users\\artist\\tex\\colormap.png",
            "/home/artist/colormap.png",
            "../Textures/wood.png",
            "stone.tga",
            "m.fbm\\cobblestone",
        ] {
            std::fs::write(
                dir.path().join("a_kit/Models/m.mtl"),
                format!("newmtl m\nKd 1 1 1\nmap_Kd {map}\n"),
            )
            .unwrap();
            let (m, _) = load_in(&model, "obj", dir.path());
            assert_eq!(texture_color(&m.unwrap()), Some([255, 0, 0]), "{map}");
        }
    }

    #[test]
    #[cfg(feature = "obj")]
    fn a_texture_named_twice_is_one_image() {
        let dir = tempfile::tempdir().unwrap();
        let obj = format!("mtllib m.mtl\n{QUAD}usemtl a\nf 1/1 2/2 3/3\nusemtl b\nf 1/1 3/3 4/4\n");
        put(
            dir.path(),
            &[
                ("m.obj", obj.as_bytes()),
                (
                    "m.mtl",
                    b"newmtl a\nKd 1 1 1\nmap_Kd tex.png\nnewmtl b\nKd 1 1 1\nmap_Kd ./tex.png\n",
                ),
                ("tex.png", &png([255, 0, 0])),
            ],
        );
        let (m, _) = load_in(&dir.path().join("m.obj"), "obj", dir.path());
        assert_eq!(m.unwrap().images.len(), 1);
    }

    #[test]
    #[cfg(feature = "fbx")]
    fn fbx_textures_naming_one_file_are_one_image() {
        // Blender writes a Texture node per material, `colormap` and `colormap.001`, for one file.
        let quad = |id: u32, model: u32, x: u32| {
            format!(
                "\tGeometry: {id}, \"Geometry::Q{id}\", \"Mesh\" {{\n\t\tVertices: *12 {{\n\t\t\ta: {x},0,0,{},0,0,{},1,0,{x},1,0\n\t\t}}\n\
                 \t\tPolygonVertexIndex: *4 {{\n\t\t\ta: 0,1,2,-4\n\t\t}}\n\t\tLayerElementUV: 0 {{\n\
                 \t\t\tMappingInformationType: \"ByPolygonVertex\"\n\t\t\tReferenceInformationType: \"IndexToDirect\"\n\
                 \t\t\tUV: *8 {{\n\t\t\t\ta: 0,0,1,0,1,1,0,1\n\t\t\t}}\n\t\t\tUVIndex: *4 {{\n\t\t\t\ta: 0,1,2,3\n\t\t\t}}\n\t\t}}\n\t}}\n\
                 \tModel: {model}, \"Model::M{model}\", \"Mesh\" {{\n\t}}\n",
                x + 1,
                x + 1
            )
        };
        let fbx = format!(
            "; FBX 7.3.0 project file\nFBXHeaderExtension:  {{\n\tFBXVersion: 7300\n}}\nObjects:  {{\n{}{}\
             \tMaterial: 30, \"Material::colormap\", \"\" {{\n\t}}\n\tMaterial: 31, \"Material::colormap-specular\", \"\" {{\n\t}}\n\
             \tTexture: 40, \"Texture::colormap\", \"\" {{\n\t\tRelativeFilename: \"tex.png\"\n\t}}\n\
             \tTexture: 41, \"Texture::colormap.001\", \"\" {{\n\t\tRelativeFilename: \"tex.png\"\n\t}}\n}}\n\
             Connections:  {{\n\tC: \"OO\",10,20\n\tC: \"OO\",11,21\n\tC: \"OO\",20,0\n\tC: \"OO\",21,0\n\tC: \"OO\",30,20\n\
             \tC: \"OO\",31,21\n\tC: \"OP\",40,30, \"DiffuseColor\"\n\tC: \"OP\",41,31, \"DiffuseColor\"\n}}\n",
            quad(10, 20, 0),
            quad(11, 21, 2)
        );
        let dir = tempfile::tempdir().unwrap();
        put(dir.path(), &[("m.fbx", fbx.as_bytes()), ("tex.png", &png([255, 0, 0]))]);
        let (m, _) = load_in(&dir.path().join("m.fbx"), "fbx", dir.path());
        let m = m.unwrap();
        assert_eq!(m.images.len(), 1);
        assert!(
            m.materials.iter().all(|x| x.color_image == Some(0)),
            "{:?}",
            m.materials
        );
    }

    #[test]
    #[cfg(feature = "fbx")]
    fn an_ascii_fbx_written_with_decimal_commas_is_refused() {
        // Asset Forge writes 2.5 as `2,5`: `*12` declared, 18 numbers there.
        let fbx = "; FBX 7.3.0 project file\nFBXHeaderExtension:  {\n\tFBXVersion: 7300\n}\nObjects:  {\n\
                   \tGeometry: 2, \"Geometry::Quad\", \"Mesh\" {\n\t\tVertices: *12 {\n\
                   \t\t\ta: 0,0,0,0,5,0,0,0,5,0,5,0,5,0,0,0,5,0\n\t\t}\n\
                   \t\tPolygonVertexIndex: *4 {\n\t\t\ta: 0,1,2,-4\n\t\t}\n\t}\n\
                   \tModel: 20, \"Model::Quad\", \"Mesh\" {\n\t}\n}\nConnections:  {\n\tC: \"OO\",2,20\n\tC: \"OO\",20,0\n}\n";
        let dir = tempfile::tempdir().unwrap();
        put(dir.path(), &[("q.fbx", fbx.as_bytes())]);
        let err = load(&dir.path().join("q.fbx"), "fbx", &Folder::new(dir.path()), Keep::ALL).unwrap_err();
        assert!(format!("{err:#}").contains("decimal commas"), "{err:#}");
    }

    #[test]
    #[cfg(feature = "obj")]
    fn a_model_colored_by_its_materials_takes_no_texture_by_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Stairs.png"), png([0, 0, 0])).unwrap();
        std::fs::write(
            dir.path().join("Stairs.mtl"),
            "newmtl Wood
Kd 0.6 0.3 0.1
newmtl Carpet
Kd 0.7 0.05 0.05
",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Stairs.obj"),
            format!(
                "mtllib Stairs.mtl
{QUAD}usemtl Wood
f 1/1 2/2 3/3
usemtl Carpet
f 1/1 3/3 4/4
"
            ),
        )
        .unwrap();
        let (m, _) = load_in(&dir.path().join("Stairs.obj"), "obj", dir.path());
        let m = m.unwrap();
        assert_eq!(m.images.len(), 0);
        assert!(m.materials.iter().all(|x| x.color_image.is_none()));
    }

    #[test]
    #[cfg(feature = "obj")]
    fn plain_colors_are_grays_and_one_besides_white() {
        let white = [1.0, 1.0, 1.0, 1.0];
        assert!(plain_colors([]));
        assert!(plain_colors([GRAY, white, GRAY]));
        assert!(!plain_colors([[0.8, 0.4, 0.1, 1.0]]));
        assert!(!plain_colors([[0.2, 0.2, 0.2, 1.0], [0.6, 0.6, 0.6, 1.0]]));
    }

    #[test]
    #[cfg(feature = "obj")]
    fn obj_with_mtl_and_texture() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Models/textures")).unwrap();
        let mut png = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb([200, 100, 50]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        // Referenced by a foreign absolute path; found by name.
        std::fs::write(dir.path().join("Models/textures/crate.png"), &png).unwrap();
        std::fs::write(
            dir.path().join("Models/crate.mtl"),
            "newmtl Wood\nKd 0.9 0.8 0.7\nmap_Kd D:\\Work\\crate.png\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Models/crate.obj"),
            "mtllib crate.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl Wood\nf 1/1 2/2 3/3 4/4\n",
        )
        .unwrap();
        let (m, f) = load_in(&dir.path().join("Models/crate.obj"), "obj", dir.path());
        let m = m.unwrap();
        assert_eq!(f["triangles"], 2);
        assert_eq!(m.materials[m.prims[0].material].name, "Wood");
        assert_eq!(m.materials[m.prims[0].material].color_image, Some(0));
        assert_eq!(m.prims[0].uvs[0], [0.0, 1.0], "V flipped");
        assert!(to_glb(&m).unwrap().starts_with(b"glTF"));
    }
}
