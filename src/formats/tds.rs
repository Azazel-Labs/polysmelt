//! Autodesk 3DS, read here rather than by Blender, whose 3DS importer has been an extension (not shipped) since 4.2.
//! A 3DS file is nested chunks (`u16` id, `u32` length counting its 6-byte header). The editor chunk holds the
//! objects (triangle meshes, cameras, lights) and the materials; the keyframer chunk beside it holds the hierarchy and
//! the animation.
//!
//! Each object becomes a node and its mesh, with a primitive per material. Mesh vertices are stored in world space
//! (the mesh matrix only records how they got there), so an object the keyframer doesn't place keeps them as they
//! are. The keyframer's object nodes become the hierarchy: their first keys are the rest pose and their tracks the
//! animation, its TCB splines as glTF cubic splines (rotations as linear, between the turns the keys add up to). A
//! placed mesh is moved into its node's space as lib3ds does (back through the mesh matrix and the pivot), unless that
//! wouldn't put it back where the editor has it at rest; then the rest pose decides. Materials have a diffuse color, a
//! transparency and maybe a texture named by file; every value they hold, maps included, is kept in `extras`. Of the
//! other maps, the specular map tints the highlight, the self-illumination map glows, and the bump map is the normal
//! map when its image is one (scaled by its amount); a bump map that is a height map, the shininess map (glossiness,
//! the inverse of glTF's roughness) and the reflection map (an environment) have no glTF slot and stay in `extras`.
//! Smoothing groups are bit masks: faces sharing a bit share normals at their common corners (by position, so a
//! texture seam doesn't crease them), and a material shaded flat ignores them. A mesh matrix that mirrors leaves its
//! faces wound inside out in world space; they are turned back. An opacity map that is the color map uses its alpha.
//! 3DS is Z up; the data stays so under a root node that turns it to glTF's Y up.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use serde_json::{Map, Value, json};

use super::xform::{self, IDENTITY, M4};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep};
use crate::images::Images;
use crate::material::{self, Alpha, Factor, Full, Specular, Tex, UvTransform};
use crate::mesh::{FastMap, GRAY, Material, Prim, face_normal, srgb_to_linear};
use crate::num::u32_of;

const MAIN: u16 = 0x4D4D;
/// A 3D Studio project file, which holds the same editor chunk.
const PROJECT: u16 = 0xC23D;
const VERSION: u16 = 0x0002;
const MASTER_SCALE: u16 = 0x0100;
const AMBIENT_LIGHT: u16 = 0x2100;
const EDITOR: u16 = 0x3D3D;
const MESH_VERSION: u16 = 0x3D3E;
const OBJECT: u16 = 0x4000;
const OBJ_HIDDEN: u16 = 0x4010;
const TRIMESH: u16 = 0x4100;
const VERTICES: u16 = 0x4110;
const FACES: u16 = 0x4120;
const FACE_MATERIAL: u16 = 0x4130;
const UVS: u16 = 0x4140;
const SMOOTHING: u16 = 0x4150;
const MESH_MATRIX: u16 = 0x4160;
const LIGHT: u16 = 0x4600;
const DL_SPOTLIGHT: u16 = 0x4610;
const DL_OFF: u16 = 0x4620;
const DL_ATTENUATE: u16 = 0x4625;
const DL_INNER_RANGE: u16 = 0x4659;
const DL_OUTER_RANGE: u16 = 0x465A;
const DL_MULTIPLIER: u16 = 0x465B;
const CAMERA: u16 = 0x4700;
const CAM_RANGES: u16 = 0x4720;
const MATERIAL: u16 = 0xAFFF;
const MAT_NAME: u16 = 0xA000;
const MAT_AMBIENT: u16 = 0xA010;
const MAT_DIFFUSE: u16 = 0xA020;
const MAT_SPECULAR: u16 = 0xA030;
const MAT_SHININESS: u16 = 0xA040;
const MAT_SHIN_STRENGTH: u16 = 0xA041;
const MAT_TRANSPARENCY: u16 = 0xA050;
const MAT_XPFALL: u16 = 0xA052;
const MAT_REFBLUR: u16 = 0xA053;
const MAT_TWO_SIDE: u16 = 0xA081;
const MAT_ADDITIVE: u16 = 0xA083;
const MAT_SELF_ILPCT: u16 = 0xA084;
const MAT_WIRE: u16 = 0xA085;
const MAT_WIRESIZE: u16 = 0xA087;
const MAT_SHADING: u16 = 0xA100;
const MAT_TEXTURE: u16 = 0xA200;
const MAT_SPECMAP: u16 = 0xA204;
const MAT_OPACMAP: u16 = 0xA210;
const MAT_REFLMAP: u16 = 0xA220;
const MAT_BUMPMAP: u16 = 0xA230;
const MAT_TEXTURE2: u16 = 0xA33A;
const MAT_SHINMAP: u16 = 0xA33C;
const MAT_SELFIMAP: u16 = 0xA33D;
const MAP_FILE: u16 = 0xA300;
const MAP_TILING: u16 = 0xA351;
const MAP_BLUR: u16 = 0xA353;
const MAP_USCALE: u16 = 0xA354;
const MAP_VSCALE: u16 = 0xA356;
const MAP_UOFFSET: u16 = 0xA358;
const MAP_VOFFSET: u16 = 0xA35A;
const MAP_ANGLE: u16 = 0xA35C;
const COLOR_F32: u16 = 0x0010;
const COLOR_U8: u16 = 0x0011;
const COLOR_U8_GAMMA: u16 = 0x0012;
const COLOR_F32_GAMMA: u16 = 0x0013;
const PERCENT_U16: u16 = 0x0030;
const PERCENT_F32: u16 = 0x0031;
const KEYFRAMER: u16 = 0xB000;
const OBJECT_NODE_TAG: u16 = 0xB002;
const KF_SEGMENT: u16 = 0xB008;
const KF_HEADER: u16 = 0xB00A;
const NODE_HDR: u16 = 0xB010;
const INSTANCE_NAME: u16 = 0xB011;
const PIVOT: u16 = 0xB013;
const POS_TRACK: u16 = 0xB020;
const ROT_TRACK: u16 = 0xB021;
const SCL_TRACK: u16 = 0xB022;
const NODE_ID: u16 = 0xB030;

/// The keyframer counts frames; 3D Studio and 3ds Max play them at 30 a second unless told otherwise, which the file
/// doesn't record.
const FRAMES_PER_SECOND: f32 = 30.0;

/// The chunks in `data`, as (id, body). A chunk claiming more than is left (a wrong length, a cut-off file) gets
/// what is left; the readers below take only whole records from it.
fn chunks(mut data: &[u8]) -> impl Iterator<Item = (u16, &[u8])> {
    std::iter::from_fn(move || {
        let id = u16::from_le_bytes(data.get(..2)?.try_into().ok()?);
        let len = u32::from_le_bytes(data.get(2..6)?.try_into().ok()?) as usize;
        if len < 6 {
            return None;
        }
        let len = len.min(data.len());
        let body = &data[6..len];
        data = &data[len..];
        Some((id, body))
    })
}

fn u16_at(d: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(i..i + 2)?.try_into().ok()?))
}

fn u32_at(d: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(i..i + 4)?.try_into().ok()?))
}

fn f32_at(d: &[u8], i: usize) -> Option<f32> {
    Some(f32::from_le_bytes(d.get(i..i + 4)?.try_into().ok()?))
}

fn vec3_at(d: &[u8], i: usize) -> Option<[f32; 3]> {
    Some([f32_at(d, i)?, f32_at(d, i + 4)?, f32_at(d, i + 8)?])
}

/// A zero-terminated name and what follows it.
fn cstr(d: &[u8]) -> (String, &[u8]) {
    let end = d.iter().position(|&c| c == 0).unwrap_or(d.len());
    (
        String::from_utf8_lossy(&d[..end]).into_owned(),
        d.get(end + 1..).unwrap_or_default(),
    )
}

/// An `f32` as the decimal it was written as (0.1, not 0.10000000149011612); null when it isn't a number.
fn clean(x: f32) -> Value {
    if x.is_finite() {
        json!(x.to_string().parse::<f64>().unwrap_or(0.0))
    } else {
        Value::Null
    }
}

fn cleans(xs: &[f32]) -> Value {
    Value::Array(xs.iter().map(|&x| clean(x)).collect())
}

/// The color chunk shown, of those in `d`. Every form holds the color picker's values, which are sRGB. 3D Studio R3
/// added "linear" twins (0x0012, 0x0013) that 3ds Max writes beside the plain ones, normally equal; the plain one is
/// the color shown, so it wins, and 8-bit wins over float.
fn color_chunk(d: &[u8]) -> Option<(u16, &[u8])> {
    let rank = |id| match id {
        COLOR_U8 => 0,
        COLOR_F32 => 1,
        COLOR_U8_GAMMA => 2,
        _ => 3,
    };
    chunks(d)
        .filter(|&(id, b)| match id {
            COLOR_U8 | COLOR_U8_GAMMA => b.len() >= 3,
            COLOR_F32 | COLOR_F32_GAMMA => b.len() >= 12,
            _ => false,
        })
        .min_by_key(|&(id, _)| rank(id))
}

fn rgb(id: u16, b: &[u8]) -> [f32; 3] {
    match id {
        COLOR_U8 | COLOR_U8_GAMMA => [b[0], b[1], b[2]].map(|v| f32::from(v) / 255.0),
        _ => vec3_at(b, 0).unwrap_or_default(),
    }
}

/// The color shown, as linear RGB.
fn color(d: &[u8]) -> Option<[f32; 3]> {
    color_chunk(d).map(|(id, b)| rgb(id, b).map(|v| srgb_to_linear(v.clamp(0.0, 1.0))))
}

/// The color shown, as written: 0 to 255, or 0 to 1 for a float chunk.
fn color_raw(d: &[u8]) -> Option<Value> {
    color_chunk(d).map(|(id, b)| match id {
        COLOR_U8 | COLOR_U8_GAMMA => json!([b[0], b[1], b[2]]),
        _ => cleans(&rgb(id, b)),
    })
}

/// A percent chunk among `d`, as written (0 to 100).
fn percent_raw(d: &[u8]) -> Option<f32> {
    chunks(d).find_map(|(id, b)| percent_of(id, b))
}

fn percent_of(id: u16, b: &[u8]) -> Option<f32> {
    match id {
        PERCENT_U16 => u16_at(b, 0).map(f32::from),
        PERCENT_F32 => f32_at(b, 0).filter(|v| v.is_finite()),
        _ => None,
    }
}

fn percent(d: &[u8]) -> Option<f32> {
    percent_raw(d).map(|p| p / 100.0)
}

struct Mat {
    name: String,
    color: [f32; 3],
    alpha: f32,
    texture: Option<String>,
    glossiness: Option<f32>,
    /// The highlight's color, and how strong it is (0 to 1).
    specular: Option<[f32; 3]>,
    shine_strength: Option<f32>,
    self_illum: f32,
    two_sided: bool,
    /// Every value the material chunk holds, by name, as written.
    raw: Map<String, Value>,
}

fn material(d: &[u8]) -> Mat {
    let mut m = Mat {
        name: String::new(),
        color: [GRAY[0], GRAY[1], GRAY[2]],
        alpha: 1.0,
        texture: None,
        glossiness: None,
        specular: None,
        shine_strength: None,
        self_illum: 0.0,
        two_sided: false,
        raw: Map::new(),
    };
    let mut maps = Map::new();
    for (id, b) in chunks(d) {
        let (key, value) = match id {
            MAT_NAME => {
                m.name = cstr(b).0;
                continue;
            }
            MAT_AMBIENT | MAT_DIFFUSE | MAT_SPECULAR => {
                if id == MAT_DIFFUSE {
                    m.color = color(b).unwrap_or(m.color);
                } else if id == MAT_SPECULAR {
                    m.specular = color(b);
                }
                let key = match id {
                    MAT_AMBIENT => "ambient",
                    MAT_DIFFUSE => "diffuse",
                    _ => "specular",
                };
                (key, color_raw(b))
            }
            MAT_SHININESS | MAT_SHIN_STRENGTH | MAT_TRANSPARENCY | MAT_XPFALL | MAT_REFBLUR | MAT_SELF_ILPCT => {
                let p = percent(b).map(|p| p.clamp(0.0, 1.0));
                let key = match id {
                    MAT_SHININESS => {
                        m.glossiness = p;
                        "shininess"
                    }
                    MAT_TRANSPARENCY => {
                        m.alpha = 1.0 - p.unwrap_or(0.0);
                        "transparency"
                    }
                    MAT_SELF_ILPCT => {
                        m.self_illum = p.unwrap_or(0.0);
                        "selfIllumination"
                    }
                    MAT_SHIN_STRENGTH => {
                        m.shine_strength = p;
                        "shininessStrength"
                    }
                    MAT_XPFALL => "transparencyFalloff",
                    _ => "reflectionBlur",
                };
                (key, percent_raw(b).map(clean))
            }
            MAT_TWO_SIDE => {
                m.two_sided = true;
                ("twoSided", Some(json!(true)))
            }
            MAT_ADDITIVE => ("additive", Some(json!(true))),
            MAT_WIRE => ("wire", Some(json!(true))),
            MAT_WIRESIZE => ("wireSize", f32_at(b, 0).map(clean)),
            MAT_SHADING => ("shading", u16_at(b, 0).map(|s| json!(s))),
            MAT_TEXTURE | MAT_TEXTURE2 | MAT_SPECMAP | MAT_OPACMAP | MAT_REFLMAP | MAT_BUMPMAP | MAT_SHINMAP
            | MAT_SELFIMAP => {
                let (file, map) = texture_map(b);
                if id == MAT_TEXTURE {
                    m.texture = file;
                }
                let key = match id {
                    MAT_TEXTURE => "texture",
                    MAT_TEXTURE2 => "texture2",
                    MAT_SPECMAP => "specular",
                    MAT_OPACMAP => "opacity",
                    MAT_REFLMAP => "reflection",
                    MAT_BUMPMAP => "bump",
                    MAT_SHINMAP => "shininess",
                    _ => "selfIllumination",
                };
                maps.insert(key.into(), Value::Object(map));
                continue;
            }
            _ => continue,
        };
        if let Some(v) = value {
            m.raw.insert(key.into(), v);
        }
    }
    if !maps.is_empty() {
        m.raw.insert("maps".into(), Value::Object(maps));
    }
    m
}

/// What a material shows beyond its flat view: how its color map is laid out, its highlight and the map that
/// tints it, its glow map, its normal map, and whether the color map's alpha cuts it.
#[derive(Default)]
struct Look {
    color: Option<Tex>,
    specular: Option<Specular<Tex>>,
    emissive: Option<Tex>,
    normal: Option<Tex>,
    alpha_source: bool,
}

/// Whether the bump map `file` is a tangent-space normal map rather than the height map 3DS bump maps were made as.
/// The image says when it can: gray is a height map, and a PNG's sampled pixels are a normal map when they are mostly
/// unit vectors leaning toward +Z (bluish, red and green near the middle). A color JPEG, or a PNG without a decoder
/// for its pixels, goes by its name (`_n`, `normal`, `nrm`, ...). Unreadable, it's neither.
fn bump_is_normal_map(assets: &dyn Assets, dir: &Path, file: &str) -> bool {
    let Some(path) = assets.find(file, dir) else {
        return false;
    };
    let Some(image) = assets.read(&path).and_then(|b| assets.image(&b, &path)) else {
        return false;
    };
    let b = &image.bytes;
    if b.starts_with(b"\x89PNG") {
        // IHDR's color type: gray, or gray and alpha.
        if matches!(b.get(25), Some(0 | 4)) {
            return false;
        }
        #[cfg(any(
            feature = "webp",
            feature = "skp",
            feature = "tga",
            feature = "bmp",
            feature = "gif",
            feature = "tiff",
            feature = "psd",
            feature = "dds"
        ))]
        if let Some(pixels) = crate::png::sample(b) {
            return normal_pixels(&pixels);
        }
    } else if jpeg_components(b) == Some(1) {
        return false;
    }
    named_normal_map(file)
}

/// Whether sampled RGB pixels read as a tangent-space normal map: on average blue well above red and green, those
/// near the middle, and three in four of them unit vectors (within 20%) facing +Z.
#[cfg_attr(
    not(any(
        feature = "webp",
        feature = "skp",
        feature = "tga",
        feature = "bmp",
        feature = "gif",
        feature = "tiff",
        feature = "psd",
        feature = "dds"
    )),
    allow(dead_code, reason = "only builds that read a PNG's pixels sample them")
)]
fn normal_pixels(pixels: &[[u8; 3]]) -> bool {
    let n = pixels.len().max(1);
    let mut sum = [0usize; 3];
    let mut unit = 0;
    for p in pixels {
        for (s, &c) in sum.iter_mut().zip(p) {
            *s += usize::from(c);
        }
        let v = p.map(|c| f32::from(c) / 127.5 - 1.0);
        let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if v[2] > 0.0 && (0.64..=1.44).contains(&len2) {
            unit += 1;
        }
    }
    let [r, g, b] = sum.map(|s| s / n);
    let middle = |c: usize| (88..=168).contains(&c);
    b >= r.max(g) + 48 && middle(r) && middle(g) && unit * 4 >= n * 3
}

/// A JPEG's color components (1 for gray), from its frame header.
fn jpeg_components(b: &[u8]) -> Option<u8> {
    let mut at = 2;
    loop {
        if *b.get(at)? != 0xFF {
            return None;
        }
        let marker = *b.get(at + 1)?;
        match marker {
            // Fill bytes.
            0xFF => at += 1,
            // Start of frame, any kind (not DHT, JPG or DAC).
            0xC0..=0xCF if !matches!(marker, 0xC4 | 0xC8 | 0xCC) => return b.get(at + 9).copied(),
            // Markers without a length, or the scan: no frame header.
            0xD0..=0xDA | 0x01 => return None,
            _ => at += 2 + usize::from(u16_at_be(b, at + 2)?),
        }
    }
}

fn u16_at_be(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*b.get(at)?, *b.get(at + 1)?]))
}

/// Whether a texture's file name marks it as a normal map: `normal`, `norm`, `nrm`, `nmap` or `ddn` in it, or `_n`,
/// `-n`, `_nm` or `_nor` ending it.
fn named_normal_map(file: &str) -> bool {
    let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
    let stem = name.rsplit_once('.').map_or(name, |(s, _)| s).to_ascii_lowercase();
    ["norm", "nrm", "nmap", "ddn"].iter().any(|k| stem.contains(k))
        || ["_n", "-n", "_nm", "_nor"].iter().any(|k| stem.ends_with(k))
}

/// Bit flags of a map's tiling: mirrored, not tiled, the image's alpha as opacity.
const TILE_MIRROR: u64 = 0x2;
const TILE_NONE: u64 = 0x10;
const TILE_ALPHA: u64 = 0x40;

/// `image` laid out as a map says: its tiling as the wrap and its scale, offset and angle as a texture transform.
/// 3DS's are taken as `uv' = rotate(angle) * scale * uv + offset` with V up, and turned into glTF's V-down UVs.
fn map_tex(map: &Map<String, Value>, image: usize) -> Tex {
    #[allow(clippy::cast_possible_truncation, reason = "they were f32s")]
    let num = |k: &str, d: f32| map.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32);
    let scale = |k: &str| Some(num(k, 1.0)).filter(|s| *s != 0.0).unwrap_or(1.0);
    let (su, sv) = (scale("uScale"), scale("vScale"));
    let (tu, tv) = (num("uOffset", 0.0), num("vOffset", 0.0));
    let angle = num("rotation", 0.0).to_radians();
    let (sin, cos) = angle.sin_cos();
    let mut t = Tex::new(image);
    let x = UvTransform {
        offset: [tu - sv * sin, 1.0 - tv - sv * cos],
        rotation: angle,
        scale: [su, sv],
    };
    // With V flipped, the identity comes back as (0, 0) offset.
    if (su, sv, tu, tv, angle) != (1.0, 1.0, 0.0, 0.0, 0.0) {
        t.transform = Some(x);
    }
    let tiling = map.get("tiling").and_then(Value::as_u64).unwrap_or(0);
    if tiling & TILE_NONE != 0 {
        t.wrap = [material::CLAMP; 2];
    } else if tiling & TILE_MIRROR != 0 {
        t.wrap = [material::MIRROR; 2];
    }
    t
}

impl Look {
    fn full(&self, m: &Material) -> Full {
        let mut f = Full::from(m);
        if let (Some(t), Some(c)) = (&mut f.base_color_tex, &self.color) {
            *t = Tex {
                image: t.image,
                ..c.clone()
            };
            if self.alpha_source {
                f.alpha = Alpha::Blend;
            }
        }
        f.specular.clone_from(&self.specular);
        f.emissive_tex.clone_from(&self.emissive);
        f.normal_tex.clone_from(&self.normal);
        f
    }
}

/// A texture map's file and every value it holds.
fn texture_map(d: &[u8]) -> (Option<String>, Map<String, Value>) {
    let mut file = None;
    let mut out = Map::new();
    for (id, b) in chunks(d) {
        let (key, value) = match id {
            MAP_FILE => {
                let f = cstr(b).0;
                let v = json!(f);
                file = Some(f).filter(|f| !f.is_empty());
                ("file", Some(v))
            }
            PERCENT_U16 | PERCENT_F32 => ("percent", percent_of(id, b).map(clean)),
            MAP_TILING => ("tiling", u16_at(b, 0).map(|t| json!(t))),
            MAP_BLUR => ("blur", f32_at(b, 0).map(clean)),
            MAP_USCALE => ("uScale", f32_at(b, 0).map(clean)),
            MAP_VSCALE => ("vScale", f32_at(b, 0).map(clean)),
            MAP_UOFFSET => ("uOffset", f32_at(b, 0).map(clean)),
            MAP_VOFFSET => ("vOffset", f32_at(b, 0).map(clean)),
            MAP_ANGLE => ("rotation", f32_at(b, 0).map(clean)),
            _ => continue,
        };
        if let Some(v) = value {
            out.insert(key.into(), v);
        }
    }
    (file, out)
}

#[derive(Default)]
struct Trimesh {
    vertices: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    faces: Vec<[u16; 3]>,
    smoothing: Vec<u32>,
    /// Material name → its faces.
    materials: Vec<(String, Vec<u16>)>,
    /// The mesh matrix as written: the X, Y and Z axes, then the origin.
    matrix: Option<[f32; 12]>,
}

fn trimesh(d: &[u8]) -> Trimesh {
    let mut t = Trimesh::default();
    for (id, b) in chunks(d) {
        match id {
            VERTICES => {
                let n = u16_at(b, 0).unwrap_or(0) as usize;
                t.vertices = (0..n).map_while(|i| vec3_at(b, 2 + i * 12)).collect();
            }
            UVS => {
                let n = u16_at(b, 0).unwrap_or(0) as usize;
                t.uvs = (0..n)
                    .map_while(|i| Some([f32_at(b, 2 + i * 8)?, f32_at(b, 6 + i * 8)?]))
                    .collect();
            }
            MESH_MATRIX => {
                let m: Option<Vec<f32>> = (0..12).map(|i| f32_at(b, i * 4)).collect();
                t.matrix = m.and_then(|m| m.try_into().ok());
            }
            FACES => {
                let n = u16_at(b, 0).unwrap_or(0) as usize;
                t.faces = (0..n)
                    .map_while(|i| {
                        let o = 2 + i * 8;
                        Some([u16_at(b, o)?, u16_at(b, o + 2)?, u16_at(b, o + 4)?])
                    })
                    .collect();
                for (id, s) in chunks(b.get(2 + n * 8..).unwrap_or_default()) {
                    match id {
                        FACE_MATERIAL => {
                            let (name, rest) = cstr(s);
                            let k = u16_at(rest, 0).unwrap_or(0) as usize;
                            let faces = (0..k).map_while(|i| u16_at(rest, 2 + i * 2)).collect();
                            t.materials.push((name, faces));
                        }
                        SMOOTHING => {
                            t.smoothing = s
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .take(n)
                                .map(|c| u32::from_le_bytes(*c))
                                .collect();
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    t
}

/// A mesh matrix as a transform.
fn mesh_matrix(m: &[f32; 12]) -> M4 {
    let mut r = IDENTITY;
    for (col, axis) in m.chunks_exact(3).enumerate() {
        for (row, &v) in axis.iter().enumerate() {
            r[row][col] = f64::from(v);
        }
    }
    r
}

struct Camera {
    position: [f32; 3],
    target: [f32; 3],
    /// Degrees about the line of sight.
    roll: f32,
    /// The focal length in millimeters, on 35 mm film.
    lens: f32,
    /// Near and far.
    range: Option<[f32; 2]>,
}

fn camera(d: &[u8]) -> Option<Camera> {
    let mut c = Camera {
        position: vec3_at(d, 0)?,
        target: vec3_at(d, 12)?,
        roll: f32_at(d, 24)?,
        lens: f32_at(d, 28)?,
        range: None,
    };
    for (id, b) in chunks(d.get(32..).unwrap_or_default()) {
        if id == CAM_RANGES {
            c.range = Some([f32_at(b, 0)?, f32_at(b, 4)?]);
        }
    }
    Some(c)
}

struct Spot {
    target: [f32; 3],
    /// The full cone angles in degrees: full brightness inside the hotspot, none past the falloff.
    hotspot: f32,
    falloff: f32,
}

struct Light {
    position: [f32; 3],
    color: Option<([f32; 3], Value)>,
    spot: Option<Spot>,
    off: bool,
    multiplier: Option<f32>,
    attenuate: bool,
    inner: Option<f32>,
    outer: Option<f32>,
}

fn light(d: &[u8]) -> Option<Light> {
    let rest = d.get(12..).unwrap_or_default();
    let mut l = Light {
        position: vec3_at(d, 0)?,
        color: color(rest).zip(color_raw(rest)),
        spot: None,
        off: false,
        multiplier: None,
        attenuate: false,
        inner: None,
        outer: None,
    };
    for (id, b) in chunks(rest) {
        match id {
            DL_SPOTLIGHT => {
                l.spot = Some(Spot {
                    target: vec3_at(b, 0)?,
                    hotspot: f32_at(b, 12)?,
                    falloff: f32_at(b, 16)?,
                });
            }
            DL_OFF => l.off = true,
            DL_ATTENUATE => l.attenuate = true,
            DL_MULTIPLIER => l.multiplier = f32_at(b, 0),
            DL_INNER_RANGE => l.inner = f32_at(b, 0),
            DL_OUTER_RANGE => l.outer = f32_at(b, 0),
            _ => {}
        }
    }
    Some(l)
}

enum Body {
    Mesh(Vec<Trimesh>),
    Camera(Camera),
    Light(Light),
    Empty,
}

struct Object {
    name: String,
    hidden: bool,
    body: Body,
}

fn object(d: &[u8]) -> Object {
    let (name, rest) = cstr(d);
    let (mut trimeshes, mut cam, mut lamp, mut hidden) = (Vec::new(), None, None, false);
    for (id, b) in chunks(rest) {
        match id {
            TRIMESH => trimeshes.push(trimesh(b)),
            CAMERA => cam = cam.or_else(|| camera(b)),
            LIGHT => lamp = lamp.or_else(|| light(b)),
            OBJ_HIDDEN => hidden = true,
            _ => {}
        }
    }
    let body = if !trimeshes.is_empty() {
        Body::Mesh(trimeshes)
    } else if let Some(c) = cam {
        Body::Camera(c)
    } else if let Some(l) = lamp {
        Body::Light(l)
    } else {
        Body::Empty
    };
    Object { name, hidden, body }
}

/// A key: its frame, its tension, continuity, bias, ease to and ease from, and its value.
struct Key<const N: usize> {
    frame: i32,
    tcb: [f32; 5],
    value: [f32; N],
}

/// A track's keys: after the flags and two unused words, the key count, then each key's frame, the flags saying
/// which spline parameters follow, those parameters, and its value.
fn track<const N: usize>(b: &[u8]) -> Vec<Key<N>> {
    let n = u32_at(b, 10).unwrap_or(0);
    let mut keys = Vec::new();
    let mut o = 14;
    for _ in 0..n {
        let (Some(frame), Some(flags)) = (u32_at(b, o), u16_at(b, o + 4)) else {
            break;
        };
        o += 6;
        let mut tcb = [0.0; 5];
        for (bit, t) in tcb.iter_mut().enumerate() {
            if flags & (1 << bit) != 0 {
                let Some(v) = f32_at(b, o) else { return keys };
                *t = v;
                o += 4;
            }
        }
        let mut value = [0.0; N];
        for v in &mut value {
            let Some(x) = f32_at(b, o) else { return keys };
            *v = x;
            o += 4;
        }
        #[allow(clippy::cast_possible_wrap, reason = "frames are signed in the file")]
        let frame = frame as i32;
        keys.push(Key { frame, tcb, value });
    }
    keys
}

/// An object node of the keyframer: the hierarchy's piece of an object, or of a dummy (`$$$DUMMY`).
struct KfNode {
    id: u16,
    name: String,
    instance: String,
    parent: Option<u16>,
    pivot: [f32; 3],
    pos: Vec<Key<3>>,
    /// Angle (radians) and axis.
    rot: Vec<Key<4>>,
    scl: Vec<Key<3>>,
}

#[derive(Default)]
struct Keyframer {
    name: String,
    revision: Option<u16>,
    length: Option<u32>,
    segment: Option<[u32; 2]>,
    nodes: Vec<KfNode>,
}

fn keyframer(d: &[u8]) -> Keyframer {
    let mut k = Keyframer::default();
    for (id, b) in chunks(d) {
        match id {
            KF_HEADER => {
                k.revision = u16_at(b, 0);
                let (name, rest) = cstr(b.get(2..).unwrap_or_default());
                k.name = name;
                k.length = u32_at(rest, 0);
            }
            KF_SEGMENT => k.segment = u32_at(b, 0).zip(u32_at(b, 4)).map(|(s, e)| [s, e]),
            OBJECT_NODE_TAG => {
                let index = u16::try_from(k.nodes.len()).unwrap_or(u16::MAX);
                k.nodes.push(kf_node(b, index));
            }
            _ => {}
        }
    }
    k
}

/// A node, its id the order it came in when it has no id chunk (as early files have).
fn kf_node(d: &[u8], index: u16) -> KfNode {
    let mut n = KfNode {
        id: index,
        name: String::new(),
        instance: String::new(),
        parent: None,
        pivot: [0.0; 3],
        pos: Vec::new(),
        rot: Vec::new(),
        scl: Vec::new(),
    };
    for (id, b) in chunks(d) {
        match id {
            NODE_ID => n.id = u16_at(b, 0).unwrap_or(index),
            NODE_HDR => {
                let (name, rest) = cstr(b);
                n.name = name;
                n.parent = u16_at(rest, 4).filter(|&p| p != u16::MAX);
            }
            INSTANCE_NAME => n.instance = cstr(b).0,
            PIVOT => n.pivot = vec3_at(b, 0).unwrap_or_default(),
            POS_TRACK => n.pos = track(b),
            ROT_TRACK => n.rot = track(b),
            SCL_TRACK => n.scl = track(b),
            _ => {}
        }
    }
    n
}

fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// A rotation key's turn. 3DS turns clockwise about the axis, as lib3ds reads it.
fn axis_angle(k: &Key<4>) -> [f64; 4] {
    let [angle, x, y, z] = k.value.map(f64::from);
    let l = (x * x + y * y + z * z).sqrt();
    if !angle.is_finite() || l.is_nan() || l < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let half = -0.5 * angle;
    let s = half.sin() / l;
    [x * s, y * s, z * s, half.cos()]
}

/// The rotation at each key: each key turns further from the one before.
fn rotations(keys: &[Key<4>]) -> Vec<[f64; 4]> {
    let mut q = [0.0, 0.0, 0.0, 1.0];
    keys.iter()
        .map(|k| {
            q = qmul(axis_angle(k), q);
            let n = q.iter().map(|c| c * c).sum::<f64>().sqrt();
            q = q.map(|c| c / n);
            q
        })
        .collect()
}

fn trs(t: [f64; 3], q: [f64; 4], s: [f64; 3]) -> M4 {
    let [x, y, z, w] = q;
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
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

/// A node's rest pose, from its first keys: translation, rotation, scale.
fn rest(n: &KfNode) -> ([f64; 3], [f64; 4], [f64; 3]) {
    let t = n.pos.first().map_or([0.0; 3], |k| k.value.map(f64::from));
    let q = rotations(&n.rot[..n.rot.len().min(1)])
        .first()
        .copied()
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let s = n.scl.first().map_or([1.0; 3], |k| k.value.map(f64::from));
    (t, q, s)
}

fn near_identity(m: &M4, scale: f64) -> bool {
    (0..3).all(|i| (0..3).all(|j| (m[i][j] - IDENTITY[i][j]).abs() < 1e-3) && m[i][3].abs() < 1e-3 * scale.max(1.0))
}

/// `prims` moved by `m`, their normals turned with them. A mirroring `m` would turn the faces inside out, so their
/// corners are put in the other order.
fn placed(mut prims: Vec<Prim>, m: &M4) -> Vec<Prim> {
    #[allow(clippy::float_cmp, reason = "only the exact identity moves nothing")]
    if *m == IDENTITY {
        return prims;
    }
    // The cofactor matrix turns normals (the inverse transpose, up to scale).
    let c: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (i1, i2, j1, j2) = ((i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3);
            m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]
        })
    });
    // The cofactors are the inverse transpose times the determinant: a mirror's points them inward.
    let mirror = xform::det3(m) < 0.0;
    let sign = if mirror { -1.0 } else { 1.0 };
    #[allow(clippy::cast_possible_truncation, reason = "back to the f32 the file had")]
    for p in &mut prims {
        if mirror {
            for t in p.indices.chunks_exact_mut(3) {
                t.swap(1, 2);
            }
        }
        for v in &mut p.positions {
            let x = v.map(f64::from);
            *v = std::array::from_fn(|i| (m[i][0] * x[0] + m[i][1] * x[1] + m[i][2] * x[2] + m[i][3]) as f32);
        }
        for n in &mut p.normals {
            let x = n.map(f64::from);
            let r: [f64; 3] = std::array::from_fn(|i| c[i][0] * x[0] + c[i][1] * x[1] + c[i][2] * x[2]);
            let len = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if len > 0.0 {
                *n = r.map(|v| (sign * v / len) as f32);
            }
        }
    }
    prims
}

/// The rotation that points a camera or spotlight (which look down their -Z, Y up) from `from` at `to`, Z up,
/// turned `roll` degrees about that line.
fn look_at(from: [f32; 3], to: [f32; 3], roll: f32) -> Option<[f64; 4]> {
    let norm = |v: [f64; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (l > 1e-12).then(|| v.map(|c| c / l))
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let f = norm(std::array::from_fn(|i| f64::from(to[i]) - f64::from(from[i])))?;
    let up = if f[2].abs() > 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let r = norm(cross(f, up))?;
    let u = cross(r, f);
    let (s, c) = f64::from(roll).to_radians().sin_cos();
    let (r, u): ([f64; 3], [f64; 3]) = (
        std::array::from_fn(|i| r[i] * c + u[i] * s),
        std::array::from_fn(|i| u[i] * c - r[i] * s),
    );
    Some(xform::quaternion(&std::array::from_fn(|i| [r[i], u[i], -f[i]])))
}

fn camera_json(c: &Camera, name: &str, extras: bool) -> Value {
    // 3D Studio's lens is on 35 mm film, 36 mm wide; at its 4:3 frame that is 27 mm high.
    let lens = if c.lens.is_finite() && c.lens > 0.0 {
        f64::from(c.lens)
    } else {
        43.456
    };
    let yfov = 2.0 * (13.5 / lens).atan();
    let mut p = json!({ "yfov": yfov, "aspectRatio": 4.0 / 3.0, "znear": 0.01 });
    if let Some([near, far]) = c.range {
        if near.is_finite() && near > 0.0 {
            p["znear"] = clean(near);
        }
        if far.is_finite() && far > near.max(0.0) {
            p["zfar"] = clean(far);
        }
    }
    let mut out = json!({ "name": name, "type": "perspective", "perspective": p });
    if extras {
        out["extras"] = json!({ "lens": clean(c.lens), "roll": clean(c.roll), "target": cleans(&c.target) });
    }
    out
}

fn light_json(l: &Light, name: &str, extras: bool) -> Value {
    let rgb = l.color.as_ref().map_or([1.0; 3], |(c, _)| *c);
    let mut out = json!({ "name": name, "type": "point", "color": rgb, "intensity": l.multiplier.map_or(1.0, |m| f64::from(m.max(0.0))) });
    if l.attenuate
        && let Some(r) = l.outer.filter(|r| r.is_finite() && *r > 0.0)
    {
        out["range"] = clean(r);
    }
    if let Some(s) = &l.spot {
        let half = std::f64::consts::FRAC_PI_2;
        let outer = (f64::from(s.falloff).to_radians() / 2.0).clamp(1e-4, half);
        let inner = (f64::from(s.hotspot).to_radians() / 2.0).clamp(0.0, outer * 0.999);
        out["type"] = json!("spot");
        out["spot"] = json!({ "innerConeAngle": inner, "outerConeAngle": outer });
    }
    if extras {
        let mut e = Map::new();
        if let Some((_, raw)) = &l.color {
            e.insert("color".into(), raw.clone());
        }
        if l.off {
            e.insert("off".into(), json!(true));
        }
        if l.attenuate {
            e.insert("attenuate".into(), json!(true));
        }
        for (k, v) in [("innerRange", l.inner), ("outerRange", l.outer)] {
            if let Some(v) = v {
                e.insert(k.into(), clean(v));
            }
        }
        if let Some(s) = &l.spot {
            e.insert(
                "spot".into(),
                json!({ "target": cleans(&s.target), "hotspot": clean(s.hotspot), "falloff": clean(s.falloff) }),
            );
        }
        if !e.is_empty() {
            out["extras"] = Value::Object(e);
        }
    }
    out
}

/// A Kochanek–Bartels track's tangents at each key as lib3ds works them out (spacing and continuity adjusted): the
/// one arriving and the one leaving, per segment.
fn tcb_tangents<const N: usize>(keys: &[&Key<N>]) -> Vec<([f32; N], [f32; N])> {
    #[allow(clippy::cast_precision_loss, reason = "frame numbers are small")]
    let frame = |k: &Key<N>| k.frame as f32;
    (0..keys.len())
        .map(|i| {
            let c = keys[i];
            let prev = i.checked_sub(1).map(|p| keys[p]);
            let next = keys.get(i + 1).copied();
            let [tens, cont, bias, ..] = c.tcb;
            let (mut fp, mut fnx) = (1.0, 1.0);
            if let (Some(p), Some(n)) = (prev, next) {
                let dt = 0.5 * (frame(n) - frame(p));
                fp = (frame(c) - frame(p)) / dt;
                fnx = (frame(n) - frame(c)) / dt;
                let a = cont.abs();
                fp = fp + a - a * fp;
                fnx = fnx + a - a * fnx;
            }
            let (cm, tm, bm) = (1.0 - cont, 0.5 * (1.0 - tens), 1.0 - bias);
            let (cp, bp) = (2.0 - cm, 2.0 - bm);
            let (ksm, ksp, kdm, kdp) = (
                tm * cm * bp * fp,
                tm * cp * bm * fp,
                tm * cp * bp * fnx,
                tm * cm * bm * fnx,
            );
            let delta = |a: &Key<N>, b: &Key<N>| -> [f32; N] { std::array::from_fn(|k| b.value[k] - a.value[k]) };
            let dm = prev.map(|p| delta(p, c));
            let dp = next.map(|n| delta(c, n));
            let (dm, dp) = match (dm, dp) {
                (Some(m), Some(p)) => (m, p),
                (Some(m), None) => (m, m),
                (None, Some(p)) => (p, p),
                (None, None) => ([0.0; N], [0.0; N]),
            };
            (
                std::array::from_fn(|k| ksm * dm[k] + ksp * dp[k]),
                std::array::from_fn(|k| kdm * dm[k] + kdp * dp[k]),
            )
        })
        .collect()
}

/// The keys of a track whose frames go up, in order (glTF needs times that only increase).
fn rising<T>(frames: impl Iterator<Item = (i32, T)>) -> Vec<T> {
    let mut last = None;
    frames
        .filter(|&(f, _)| {
            let up = last.is_none_or(|l| f > l);
            if up {
                last = Some(f);
            }
            up
        })
        .map(|(_, t)| t)
        .collect()
}

#[allow(clippy::cast_precision_loss, reason = "frame numbers are small")]
fn seconds(frame: i32) -> f32 {
    frame as f32 / FRAMES_PER_SECOND
}

/// A vector track as a cubic spline channel: each key's arriving tangent, value and leaving tangent, per second.
fn spline(b: &mut Builder, keys: &[Key<3>]) -> (usize, usize) {
    let keys: Vec<&Key<3>> = rising(keys.iter().map(|k| (k.frame, k)));
    let tangents = tcb_tangents(&keys);
    let times: Vec<f32> = keys.iter().map(|k| seconds(k.frame)).collect();
    let mut out = Vec::with_capacity(keys.len() * 9);
    for (i, (k, (arrive, leave))) in keys.iter().zip(&tangents).enumerate() {
        let before = i.checked_sub(1).map_or(0.0, |p| times[i] - times[p]);
        let after = times.get(i + 1).map_or(0.0, |n| n - times[i]);
        out.extend(arrive.map(|v| if before > 0.0 { v / before } else { 0.0 }));
        out.extend(k.value);
        out.extend(leave.map(|v| if after > 0.0 { v / after } else { 0.0 }));
    }
    (b.floats(times, "SCALAR", true), b.floats(out, "VEC3", false))
}

/// Each object's mesh, by the placement its vertices are moved by.
struct Meshes {
    prims: Vec<Option<Vec<Prim>>>,
    /// Uses left of each object's primitives, so the last takes them rather than a copy.
    left: Vec<usize>,
    of: HashMap<(usize, [u64; 16]), Option<usize>>,
    list: Vec<Value>,
}

impl Meshes {
    fn get(&mut self, b: &mut Builder, obj: usize, m: &M4) -> Option<usize> {
        let key = (obj, std::array::from_fn(|k| m[k / 4][k % 4].to_bits()));
        if let Some(&i) = self.of.get(&key) {
            return i;
        }
        self.left[obj] = self.left[obj].saturating_sub(1);
        let prims = if self.left[obj] == 0 {
            self.prims[obj].take()
        } else {
            self.prims[obj].clone()
        }?;
        let mesh = b.mesh(placed(prims, m), usize::MAX).map(|mesh| {
            self.list.push(mesh);
            self.list.len() - 1
        });
        self.of.insert(key, mesh);
        mesh
    }
}

#[allow(clippy::too_many_lines, reason = "one pass over the file's parts, in order")]
pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    let data = input.read()?;
    let main = chunks(&data)
        .next()
        .filter(|&(id, _)| id == MAIN || id == PROJECT)
        .map(|(_, b)| b);
    let Some(main) = main else { bail!("not a 3DS file") };
    let mut source = json!({ "format": "3DS", "upAxis": "Z" });
    let (mut editor, mut kf) = (&[][..], Keyframer::default());
    for (id, b) in chunks(main) {
        match id {
            VERSION => {
                if let Some(v) = u32_at(b, 0) {
                    source["version"] = json!(v);
                }
            }
            EDITOR if editor.is_empty() => editor = b,
            KEYFRAMER => kf = keyframer(b),
            _ => {}
        }
    }

    let mut materials: Vec<Material> = Vec::new();
    let mut raws: Vec<Map<String, Value>> = Vec::new();
    let mut looks: Vec<Look> = Vec::new();
    // Whether each material is shaded flat.
    let mut flat: Vec<bool> = Vec::new();
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let dir = input.dir();
    let mut images = Images::new(!keep.image_bytes, dir);
    let mut image_of: HashMap<String, Option<usize>> = HashMap::new();
    // Whether each bump map file is a normal map.
    let mut normal_of: HashMap<String, bool> = HashMap::new();
    let mut objects = Vec::new();
    for (id, b) in chunks(editor) {
        match id {
            MESH_VERSION => {
                if let Some(v) = u32_at(b, 0) {
                    source["meshVersion"] = json!(v);
                }
            }
            MASTER_SCALE => {
                if let Some(s) = f32_at(b, 0) {
                    source["masterScale"] = clean(s);
                }
            }
            AMBIENT_LIGHT => {
                if let Some(c) = color_raw(b) {
                    source["ambient"] = c;
                }
            }
            MATERIAL => {
                let m = material(b);
                let mut image_for = |f: &str| {
                    if !(keep.materials && keep.textures) {
                        return None;
                    }
                    *image_of
                        .entry(f.to_lowercase())
                        .or_insert_with(|| images.file(assets, assets.find(f, dir)?, f))
                };
                let image = m.texture.as_deref().and_then(&mut image_for);
                let maps = m.raw.get("maps").and_then(Value::as_object);
                let map = |k: &str| maps.and_then(|ms| ms.get(k)).and_then(Value::as_object);
                let mut slot = |k: &str| {
                    let map = map(k)?;
                    let image = image_for(map.get("file")?.as_str()?)?;
                    Some(map_tex(map, image))
                };
                let (spec_tex, emissive) = (slot("specular"), slot("selfIllumination"));
                let bump = map("bump").filter(|b| {
                    let Some(f) = b
                        .get("file")
                        .and_then(Value::as_str)
                        .filter(|_| keep.materials && keep.textures)
                    else {
                        return false;
                    };
                    *normal_of
                        .entry(f.to_lowercase())
                        .or_insert_with(|| bump_is_normal_map(assets, dir, f))
                });
                // The bump amount as the normal map's scale.
                #[allow(clippy::cast_possible_truncation, reason = "it was an f32")]
                let normal = bump.and_then(|_| slot("bump")).map(|mut t| {
                    let amount = bump.and_then(|b| b.get("percent")).and_then(Value::as_f64);
                    t.scale = amount.map_or(1.0, |p| (p / 100.0) as f32);
                    t
                });
                let tiling = map("texture").and_then(|t| t.get("tiling")).and_then(Value::as_u64);
                // An opacity map that is the color map: its alpha.
                let opacity = map("opacity").and_then(|o| o.get("file")).and_then(Value::as_str);
                let cut = opacity
                    .zip(m.texture.as_deref())
                    .is_some_and(|(o, t)| o.eq_ignore_ascii_case(t));
                flat.push(m.raw.get("shading").and_then(Value::as_u64) == Some(1));
                looks.push(Look {
                    color: image.and(map("texture")).map(|t| map_tex(t, 0)),
                    // Phong's highlight: its color, scaled by its strength.
                    specular: (m.specular.is_some() || m.shine_strength.is_some() || spec_tex.is_some()).then(|| {
                        Specular {
                            factor: Factor::of(m.shine_strength.unwrap_or(1.0)),
                            color: m.specular.unwrap_or([1.0; 3]),
                            color_tex: spec_tex,
                        }
                    }),
                    emissive,
                    normal: normal.clone(),
                    alpha_source: cut || tiling.is_some_and(|t| t & TILE_ALPHA != 0),
                });
                // A texture shows as painted, not tinted by the color under it.
                let rgb = if image.is_some() { [1.0; 3] } else { m.color };
                let plain = Material::plain(&m.name, [rgb[0], rgb[1], rgb[2], m.alpha]);
                by_name.insert(m.name.clone(), materials.len());
                materials.push(Material {
                    color_image: image,
                    normal_image: normal.map(|t| t.image),
                    emissive: m.color.map(|c| c * m.self_illum),
                    roughness: m.glossiness.map_or(plain.roughness, |g| 1.0 - g),
                    double_sided: m.two_sided,
                    ..plain
                });
                raws.push(m.raw);
            }
            OBJECT => objects.push(object(b)),
            _ => {}
        }
    }

    // Each mesh object's primitives, one per material.
    let mut triangles = 0usize;
    let mut prims_of = Vec::with_capacity(objects.len());
    for o in &objects {
        let Body::Mesh(ts) = &o.body else {
            prims_of.push(None);
            continue;
        };
        let mut prims: HashMap<usize, Prim> = HashMap::new();
        for t in ts {
            triangles += t.faces.len();
            ensure!(triangles <= input.limit, "more than {} triangles", input.limit);
            let mut face_mat = vec![None; t.faces.len()];
            for (name, faces) in &t.materials {
                let m = by_name.get(name).copied();
                for &f in faces {
                    if let Some(slot) = face_mat.get_mut(f as usize) {
                        *slot = m;
                    }
                }
            }
            add_faces(t, &face_mat, &flat, &mut prims, &mut materials);
        }
        let mut prims: Vec<Prim> = prims.into_values().filter(|p| !p.indices.is_empty()).collect();
        prims.sort_by_key(|p| p.material);
        prims_of.push(Some(prims).filter(|p| !p.is_empty()));
    }
    ensure!(triangles > 0, "this 3DS file has no meshes");
    let mut textured = vec![false; materials.len()];
    for p in prims_of.iter().flatten().flatten().filter(|p| !p.uvs.is_empty()) {
        textured[p.material] = true;
    }
    if keep.materials && keep.textures {
        super::guess_texture(&mut materials, &mut images, &textured, input.path, assets);
    }

    let mut b = Builder::default();
    let mut nodes: Vec<Value> = Vec::new();
    let (mut cameras, mut lights) = (Vec::new(), Vec::new());
    let mut roots = Vec::new();
    let mut meshes = Meshes {
        left: vec![0; objects.len()],
        prims: prims_of,
        of: HashMap::new(),
        list: Vec::new(),
    };
    let object_of: HashMap<&str, usize> = objects
        .iter()
        .enumerate()
        .rev()
        .map(|(i, o)| (o.name.as_str(), i))
        .collect();
    let placed_by_kf: Vec<Option<usize>> = kf
        .nodes
        .iter()
        .map(|n| object_of.get(n.name.as_str()).copied())
        .collect();
    for &o in placed_by_kf.iter().flatten() {
        meshes.left[o] += 1;
    }
    let object_extras = |o: &Object, e: &mut Map<String, Value>| {
        if o.hidden {
            e.insert("hidden".into(), json!(true));
        }
        if let Body::Mesh(ts) = &o.body
            && let Some(m) = ts.iter().find_map(|t| t.matrix)
        {
            e.insert("meshMatrix".into(), cleans(&m));
        }
    };

    // The objects the keyframer doesn't place, where the editor has them.
    for (i, o) in objects.iter().enumerate() {
        if placed_by_kf.contains(&Some(i)) {
            continue;
        }
        let mut node = json!({ "name": o.name });
        match &o.body {
            Body::Mesh(_) => {
                meshes.left[i] += 1;
                if let Some(m) = meshes.get(&mut b, i, &IDENTITY) {
                    node["mesh"] = json!(m);
                }
            }
            Body::Camera(c) if keep.cameras => {
                node["translation"] = cleans(&c.position);
                if let Some(q) = look_at(c.position, c.target, c.roll) {
                    node["rotation"] = json!(q);
                }
                cameras.push(camera_json(c, &o.name, keep.extras));
                node["camera"] = json!(cameras.len() - 1);
            }
            Body::Light(l) if keep.lights => {
                node["translation"] = cleans(&l.position);
                if let Some(q) = l.spot.as_ref().and_then(|s| look_at(l.position, s.target, 0.0)) {
                    node["rotation"] = json!(q);
                }
                lights.push(light_json(l, &o.name, keep.extras));
                node["extensions"] = json!({ "KHR_lights_punctual": { "light": lights.len() - 1 } });
            }
            Body::Camera(_) | Body::Light(_) => continue,
            Body::Empty => {}
        }
        if keep.extras {
            let mut e = Map::new();
            object_extras(o, &mut e);
            if !e.is_empty() {
                node["extras"] = Value::Object(e);
            }
        }
        roots.push(nodes.len());
        nodes.push(node);
    }

    // The keyframer's hierarchy: its nodes' rest poses, and where each is in the world at rest.
    let first = nodes.len();
    let index_of: HashMap<u16, usize> = kf.nodes.iter().enumerate().rev().map(|(i, n)| (n.id, i)).collect();
    let parent_of: Vec<Option<usize>> = kf
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| n.parent.and_then(|p| index_of.get(&p).copied()).filter(|&p| p != i))
        .collect();
    let rests: Vec<_> = kf.nodes.iter().map(rest).collect();
    let local: Vec<M4> = rests.iter().map(|&(t, q, s)| trs(t, q, s)).collect();
    let mut world: Vec<Option<M4>> = vec![None; kf.nodes.len()];
    let mut parent_ok = vec![true; kf.nodes.len()];
    for i in 0..kf.nodes.len() {
        // Up to the first node placed already, or a root; a loop makes the node a root.
        let mut chain = vec![i];
        while let Some(p) = parent_of[*chain.last().unwrap_or(&i)] {
            if world[p].is_some() {
                break;
            }
            if chain.contains(&p) {
                parent_ok[*chain.last().unwrap_or(&i)] = false;
                break;
            }
            chain.push(p);
        }
        for &n in chain.iter().rev() {
            let above = parent_of[n].filter(|_| parent_ok[n]).and_then(|p| world[p]);
            world[n] = Some(above.map_or(local[n], |w| xform::mul(&w, &local[n])));
        }
    }
    let mut placed_once = vec![false; objects.len()];
    for (i, n) in kf.nodes.iter().enumerate() {
        let obj = placed_by_kf[i];
        let name = if n.instance.is_empty() { &n.name } else { &n.instance };
        let mut node = json!({ "name": name });
        let (t, q, s) = rests[i];
        if t.iter().any(|&c| c != 0.0) {
            node["translation"] = cleans(&n.pos[0].value);
        }
        if (q[3] - 1.0).abs() > 1e-12 {
            node["rotation"] = json!(q);
        }
        #[allow(clippy::float_cmp, reason = "exactly one is no scale")]
        if s.iter().any(|&c| c != 1.0) {
            node["scale"] = cleans(&n.scl[0].value);
        }
        let w = world[i].unwrap_or(IDENTITY);
        if let Some(o) = obj
            && matches!(objects[o].body, Body::Mesh(_))
        {
            // Back through the mesh matrix and the pivot, as lib3ds does, unless the object's first node isn't then
            // where the editor has it: then its rest pose decides.
            let m = match &objects[o].body {
                Body::Mesh(ts) => ts.iter().find_map(|t| t.matrix).map_or(IDENTITY, |m| mesh_matrix(&m)),
                _ => IDENTITY,
            };
            let pivot = xform::translate(n.pivot.map(|c| -f64::from(c)));
            let lib = (xform::det3(&m).abs() > 1e-12).then(|| xform::mul(&pivot, &xform::inverse(&m)));
            let by_rest = xform::inverse(&w);
            let size = m[0][3].abs().max(m[1][3].abs()).max(m[2][3].abs());
            let into = match lib {
                Some(l) if placed_once[o] || near_identity(&xform::mul(&w, &l), size) => l,
                _ => by_rest,
            };
            placed_once[o] = true;
            if let Some(mesh) = meshes.get(&mut b, o, &into) {
                node["mesh"] = json!(mesh);
            }
        }
        if keep.extras {
            let mut e = Map::new();
            e.insert("nodeId".into(), json!(n.id));
            if !n.instance.is_empty() {
                e.insert("object".into(), json!(n.name));
            }
            if n.pivot.iter().any(|&c| c != 0.0) {
                e.insert("pivot".into(), cleans(&n.pivot));
            }
            if let Some(o) = obj {
                object_extras(&objects[o], &mut e);
            }
            node["extras"] = Value::Object(e);
        }
        nodes.push(node);
    }
    for i in 0..kf.nodes.len() {
        match parent_of[i].filter(|_| parent_ok[i]) {
            Some(p) => {
                let children = nodes[first + p]
                    .as_object_mut()
                    .map(|o| o.entry("children").or_insert_with(|| json!([])));
                if let Some(Value::Array(c)) = children {
                    c.push(json!(first + i));
                }
            }
            None => roots.push(first + i),
        }
    }

    let mut animations = Vec::new();
    if keep.animations {
        let (mut channels, mut samplers) = (Vec::new(), Vec::new());
        for (i, n) in kf.nodes.iter().enumerate() {
            let node = first + i;
            let mut add = |path: &str, (input, output): (usize, usize), interpolation: &str| {
                channels.push(json!({ "sampler": samplers.len(), "target": { "node": node, "path": path } }));
                samplers.push(json!({ "input": input, "output": output, "interpolation": interpolation }));
            };
            if n.pos.len() > 1 {
                add("translation", spline(&mut b, &n.pos), "CUBICSPLINE");
            }
            if n.rot.len() > 1 {
                let turns = rotations(&n.rot);
                let keys: Vec<(f32, [f64; 4])> =
                    rising(n.rot.iter().zip(turns).map(|(k, q)| (k.frame, (seconds(k.frame), q))));
                // Each quaternion on the same side as the last, so slerp takes the short way.
                let mut last = [0.0, 0.0, 0.0, 1.0];
                let mut values = Vec::with_capacity(keys.len() * 4);
                for &(_, mut q) in &keys {
                    if (0..4).map(|k| q[k] * last[k]).sum::<f64>() < 0.0 {
                        q = q.map(|c| -c);
                    }
                    last = q;
                    #[allow(clippy::cast_possible_truncation, reason = "accessor floats")]
                    values.extend(q.map(|c| c as f32));
                }
                let times = keys.iter().map(|&(t, _)| t).collect();
                let io = (b.floats(times, "SCALAR", true), b.floats(values, "VEC4", false));
                add("rotation", io, "LINEAR");
            }
            if n.scl.len() > 1 {
                add("scale", spline(&mut b, &n.scl), "CUBICSPLINE");
            }
        }
        if !channels.is_empty() {
            animations.push(json!({ "name": "keyframer", "channels": channels, "samplers": samplers }));
        }
    }
    if !kf.nodes.is_empty() || kf.revision.is_some() {
        let mut k = Map::new();
        if !kf.name.is_empty() {
            k.insert("name".into(), json!(kf.name));
        }
        if let Some(r) = kf.revision {
            k.insert("revision".into(), json!(r));
        }
        if let Some(l) = kf.length {
            k.insert("length".into(), json!(l));
        }
        if let Some(s) = kf.segment {
            k.insert("segment".into(), json!(s));
        }
        source["keyframer"] = Value::Object(k);
    }

    // Z-up to Y-up: a quarter turn about X.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    nodes.push(json!({ "name": "RootNode", "children": roots, "rotation": [-h, 0.0, 0.0, h] }));
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": [nodes.len() - 1] }]));
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes.list));
    if !animations.is_empty() {
        json.insert("animations".into(), Value::Array(animations));
    }
    if !cameras.is_empty() {
        json.insert("cameras".into(), Value::Array(cameras));
    }
    if !lights.is_empty() {
        json.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
        json.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": lights } }),
        );
    }
    let fulls = materials
        .iter()
        .enumerate()
        .map(|(i, m)| looks.get(i).map_or_else(|| Full::from(m), |l| l.full(m)));
    b.materials(fulls, images.list, &mut json);
    if keep.extras
        && let Some(Value::Array(list)) = json.get_mut("materials")
    {
        for (mat, raw) in list.iter_mut().zip(raws) {
            if !raw.is_empty() {
                mat["extras"] = json!({ "3ds": raw });
            }
        }
    }
    // `read` counts the triangles drawn, from the document.
    Ok(b.finish(json, Map::new()))
}

/// A trimesh's faces, into the prims of their materials. A corner's normal averages the faces around its vertex that
/// share a smoothing bit with its own face, each weighed by its angle there (so the triangulation doesn't matter); a
/// face in no group, or in a material shaded flat, is flat. Vertices at one position are one corner to smoothing.
/// Corners share a vertex when material, vertex and normal agree. Missing texture coordinates are zero.
fn add_faces(
    t: &Trimesh,
    face_mat: &[Option<usize>],
    flat: &[bool],
    prims: &mut HashMap<usize, Prim>,
    materials: &mut Vec<Material>,
) {
    let pos = &t.vertices;
    let has_uvs = !t.uvs.is_empty();
    let n = pos.len();
    let whole = |f: &[u16; 3]| f.iter().all(|&i| usize::from(i) < n);
    let group = |f: usize| {
        let flat = face_mat
            .get(f)
            .copied()
            .flatten()
            .is_some_and(|m| flat.get(m) == Some(&true));
        if flat {
            0
        } else {
            t.smoothing.get(f).copied().unwrap_or(0)
        }
    };
    // A mirroring mesh matrix leaves the faces inside out in the world space the vertices are in.
    let faces: Cow<'_, [[u16; 3]]> = if t.matrix.is_some_and(|m| xform::det3(&mesh_matrix(&m)) < 0.0) {
        t.faces.iter().map(|&[a, b, c]| [a, c, b]).collect()
    } else {
        Cow::Borrowed(&t.faces)
    };

    // Each face's unit normal, and its angle at each corner when some face is smoothed (the
    // angles weight the normals smoothing averages).
    let smoothed = faces.iter().enumerate().any(|(f, face)| whole(face) && group(f) != 0);
    let mut unit = vec![[0.0f32; 3]; faces.len()];
    let mut angle = vec![[0.0f32; 3]; if smoothed { faces.len() } else { 0 }];
    for (f, face) in faces.iter().enumerate() {
        if !whole(face) {
            continue;
        }
        let p = face.map(|i| pos[usize::from(i)]);
        let len = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let fnrm = face_normal(p[0], p[1], p[2]);
        if len(fnrm) > 0.0 {
            unit[f] = fnrm.map(|c| c / len(fnrm));
        }
        if !smoothed {
            continue;
        }
        for k in 0..3 {
            let (a, b, c) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            angle[f][k] = len(face_normal(a, b, c)).atan2(dot);
        }
    }

    // The faces around each position, as (face, corner), in compressed rows, the position named by its first vertex
    // (a texture seam splits a vertex where the surface is smooth); only built when some face is smoothed.
    let mut start = vec![0u32; n + 1];
    let mut around: Vec<(u32, u8)> = Vec::new();
    let mut canon: Vec<u32> = Vec::new();
    if smoothed {
        let mut first: FastMap<[u32; 3], u32> = FastMap::default();
        first.reserve(n);
        canon = (pos.iter().enumerate())
            .map(|(i, p)| *first.entry(p.map(|c| (c + 0.0).to_bits())).or_insert(u32_of(i)))
            .collect();
        for face in faces.iter().filter(|f| whole(f)) {
            for &i in face {
                start[canon[usize::from(i)] as usize + 1] += 1;
            }
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        around = vec![(0, 0); start[n] as usize];
        for (f, face) in faces.iter().enumerate().filter(|(_, f)| whole(f)) {
            for (k, &i) in face.iter().enumerate() {
                let slot = &mut fill[canon[usize::from(i)] as usize];
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "3DS has at most 65535 faces of 3 corners"
                )]
                let entry = (f as u32, k as u8);
                around[*slot as usize] = entry;
                *slot += 1;
            }
        }
    }

    let mut seen: FastMap<(usize, u16, [u32; 3]), u32> = FastMap::default();
    seen.reserve(n);
    // The primitive of the material faces are using, out of `prims` while it is: faces come in
    // runs of one material.
    let mut cur = (usize::MAX, Prim::default());
    for (f, face) in faces.iter().enumerate() {
        if !whole(face) {
            continue;
        }
        let mat = face_mat[f].unwrap_or_else(|| {
            if materials.last().is_none_or(|m| m.name != "default") {
                materials.push(Material::plain("default", GRAY));
            }
            materials.len() - 1
        });
        if cur.0 != mat {
            let next = prims.remove(&mat).unwrap_or_else(|| Prim {
                material: mat,
                ..Prim::default()
            });
            let (m, done) = std::mem::replace(&mut cur, (mat, next));
            if m != usize::MAX {
                prims.insert(m, done);
            }
        }
        let prim = &mut cur.1;
        let g = group(f);
        for &vi in face {
            let mut nrm = unit[f];
            if g != 0 {
                let v = canon[usize::from(vi)] as usize;
                let mut sum = [0.0f32; 3];
                for &(o, k) in &around[start[v] as usize..start[v + 1] as usize] {
                    let o = o as usize;
                    if o == f || group(o) & g != 0 {
                        let w = angle[o][usize::from(k)];
                        for c in 0..3 {
                            sum[c] += unit[o][c] * w;
                        }
                    }
                }
                let len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
                if len > 0.0 {
                    nrm = sum.map(|c| c / len);
                }
            }
            #[allow(clippy::float_cmp, reason = "exactly zero: no face in the group gave a normal")]
            if nrm == [0.0; 3] {
                nrm = [0.0, 0.0, 1.0];
            }
            // -0.0 + 0.0 is 0.0: one key for both zeros.
            let nrm = nrm.map(|c| c + 0.0);
            let idx = *seen.entry((mat, vi, nrm.map(f32::to_bits))).or_insert_with(|| {
                prim.positions.push(pos[usize::from(vi)]);
                prim.normals.push(nrm);
                if has_uvs {
                    let [s, tv] = t.uvs.get(usize::from(vi)).copied().unwrap_or_default();
                    prim.uvs.push([s, 1.0 - tv]);
                }
                u32_of(prim.positions.len() - 1)
            });
            prim.indices.push(idx);
        }
    }
    if cur.0 != usize::MAX {
        prims.insert(cur.0, cur.1);
    }
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;
    use crate::document::Pose;

    fn chunk(id: u16, body: &[u8]) -> Vec<u8> {
        let mut out = id.to_le_bytes().to_vec();
        out.extend(u32::try_from(body.len() + 6).unwrap().to_le_bytes());
        out.extend(body);
        out
    }

    fn f32s(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn u16s(xs: &[u16]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A trimesh of `verts` and `faces`, with a mesh matrix moving it to `origin` when there is one.
    fn trimesh(verts: &[[f32; 3]], faces: &[[u16; 3]], origin: Option<[f32; 3]>) -> Vec<u8> {
        let mut v = u16::try_from(verts.len()).unwrap().to_le_bytes().to_vec();
        v.extend(f32s(verts.as_flattened()));
        let mut f = u16::try_from(faces.len()).unwrap().to_le_bytes().to_vec();
        for face in faces {
            f.extend(u16s(&[face[0], face[1], face[2], 0]));
        }
        let mut t = chunk(VERTICES, &v);
        if let Some(o) = origin {
            t.extend(chunk(
                MESH_MATRIX,
                &f32s(&[1., 0., 0., 0., 1., 0., 0., 0., 1., o[0], o[1], o[2]]),
            ));
        }
        t.extend(chunk(FACES, &f));
        t
    }

    fn object(name: &str, body: &[u8]) -> Vec<u8> {
        let mut o = name.as_bytes().to_vec();
        o.push(0);
        o.extend(body);
        chunk(OBJECT, &o)
    }

    /// A square on the ground (Z up) of two triangles in one smoothing group, red, and a lone triangle standing up
    /// in no material.
    fn square() -> Vec<u8> {
        let mut mat = chunk(MAT_NAME, b"Red\0");
        mat.extend(chunk(MAT_DIFFUSE, &chunk(COLOR_U8, &[255, 0, 0])));
        mat.extend(chunk(MAT_TRANSPARENCY, &chunk(PERCENT_U16, &25u16.to_le_bytes())));
        let mut map = chunk(MAP_FILE, b"red.png\0");
        map.extend(chunk(MAP_USCALE, &f32s(&[2.0])));
        mat.extend(chunk(MAT_BUMPMAP, &map));

        let mut verts = 4u16.to_le_bytes().to_vec();
        verts.extend(f32s(&[0., 0., 0., 2., 0., 0., 2., 2., 0., 0., 2., 0.]));
        let mut uvs = 4u16.to_le_bytes().to_vec();
        uvs.extend(f32s(&[0., 0., 1., 0., 1., 1., 0., 1.]));
        let mut faces = 2u16.to_le_bytes().to_vec();
        for f in [[0u16, 1, 2, 7], [0, 2, 3, 7]] {
            faces.extend(u16s(&f));
        }
        let mut fm = b"Red\0".to_vec();
        fm.extend(2u16.to_le_bytes());
        fm.extend(u16s(&[0, 1]));
        faces.extend(chunk(FACE_MATERIAL, &fm));
        faces.extend(chunk(
            SMOOTHING,
            &[1u32, 1].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        ));
        let mut tri = chunk(VERTICES, &verts);
        tri.extend(chunk(UVS, &uvs));
        tri.extend(chunk(FACES, &faces));

        let mut editor = chunk(MATERIAL, &mat);
        editor.extend(object("Floor", &chunk(TRIMESH, &tri)));
        editor.extend(object(
            "Wall",
            &chunk(
                TRIMESH,
                &trimesh(&[[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]], &[[0, 1, 2]], None),
            ),
        ));
        chunk(MAIN, &chunk(EDITOR, &editor))
    }

    fn read(name: &str, data: &[u8]) -> Result<Document<'static>> {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(name);
        std::fs::write(&p, data).unwrap();
        load(
            crate::formats::Input::file(&p),
            &crate::assets::Folder::new(dir.path()),
            Keep::ALL,
        )
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < 1e-4)
    }

    #[test]
    fn the_triangle_limit_refuses_bigger_models() {
        let read = |max_triangles| {
            let keep = Keep {
                max_triangles,
                ..Keep::ALL
            };
            crate::formats::load_bytes(&square(), "s.3ds", "3ds", &crate::assets::Memory::new(), keep)
                .map(|d| d.flatten(crate::document::Pose::Rest).unwrap().unwrap().triangles())
        };
        let err = read(2).unwrap_err();
        assert!(format!("{err:#}").contains("more than 2 triangles"), "{err:#}");
        assert_eq!(read(3).unwrap(), 3);
    }

    #[test]
    fn meshes_materials_and_smoothing() {
        let doc = read("floor.3ds", &square()).unwrap();
        let names: Vec<&str> = doc.json["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["Floor", "Wall", "RootNode"]);
        assert_eq!(doc.json["asset"]["extras"]["source"]["upAxis"], "Z");
        let red = &doc.json["materials"][0]["extras"]["3ds"];
        assert_eq!(red["diffuse"], json!([255, 0, 0]));
        assert_eq!(red["transparency"], json!(25.0));
        assert_eq!(red["maps"]["bump"], json!({ "file": "red.png", "uScale": 2.0 }));

        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(mesh.triangles(), 3);
        assert_eq!(mesh.prims.len(), 2);
        let floor = &mesh.prims[0];
        let red = &mesh.materials[floor.material];
        assert_eq!(red.name, "Red");
        assert_eq!(red.color, [1.0, 0.0, 0.0, 0.75]);
        // One smoothing group: the shared corners are shared vertices.
        assert_eq!(floor.positions.len(), 4);
        assert!(
            floor.positions.iter().all(|p| p[1].abs() < 1e-6),
            "Z up became Y up: {:?}",
            floor.positions
        );
        assert!(floor.positions.iter().any(|&p| close(p, [2.0, 0.0, -2.0])));
        assert!(floor.normals.iter().all(|n| n[1] > 0.99), "{:?}", floor.normals);
        assert!(floor.uvs.contains(&[0.0, 1.0]), "V flipped to glTF's origin");
        let wall = &mesh.prims[1];
        assert_eq!(mesh.materials[wall.material].name, "default");
        assert_eq!(wall.positions.len(), 3);
        assert!(wall.normals.iter().all(|n| n[2].abs() > 0.99), "{:?}", wall.normals);
        assert!(crate::mesh::to_glb(&mesh).unwrap().starts_with(b"glTF"));
    }

    fn track(id: u16, keys: &[(u32, &[f32])]) -> Vec<u8> {
        let mut t = u16s(&[0, 0, 0, 0, 0]);
        t.extend(u32::try_from(keys.len()).unwrap().to_le_bytes());
        for (frame, v) in keys {
            t.extend(frame.to_le_bytes());
            t.extend(0u16.to_le_bytes());
            t.extend(f32s(v));
        }
        chunk(id, &t)
    }

    fn node(id: u16, name: &str, parent: u16, pivot: [f32; 3], tracks: &[Vec<u8>]) -> Vec<u8> {
        let mut hdr = name.as_bytes().to_vec();
        hdr.push(0);
        hdr.extend(u16s(&[0, 0, parent]));
        let mut n = chunk(NODE_ID, &id.to_le_bytes());
        n.extend(chunk(NODE_HDR, &hdr));
        n.extend(chunk(PIVOT, &f32s(&pivot)));
        for t in tracks {
            n.extend(t);
        }
        chunk(OBJECT_NODE_TAG, &n)
    }

    /// A unit triangle moved to `p`.
    fn at(p: [f32; 3]) -> Vec<[f32; 3]> {
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
            .iter()
            .map(|v| [v[0] + p[0], v[1] + p[1], v[2] + p[2]])
            .collect()
    }

    #[test]
    fn the_keyframer_places_and_animates() {
        let base = trimesh(&at([5., 0., 0.]), &[[0, 1, 2]], Some([5., 0., 0.]));
        let arm = trimesh(&at([5., 3., 0.]), &[[0, 1, 2]], Some([5., 3., 0.]));
        let mut editor = object("Base", &chunk(TRIMESH, &base));
        editor.extend(object("Arm", &chunk(TRIMESH, &arm)));
        let quarter = std::f32::consts::FRAC_PI_2;
        let mut kf = node(
            0,
            "Base",
            u16::MAX,
            [0.; 3],
            &[track(POS_TRACK, &[(0, &[5., 0., 0.]), (30, &[5., 0., 2.])])],
        );
        kf.extend(node(
            1,
            "Arm",
            0,
            [0.5, 0.5, 0.],
            &[
                track(POS_TRACK, &[(0, &[0.5, 3.5, 0.])]),
                track(ROT_TRACK, &[(0, &[0., 0., 0., 1.]), (30, &[quarter, 0., 0., 1.])]),
            ],
        ));
        let mut main = chunk(EDITOR, &editor);
        main.extend(chunk(KEYFRAMER, &kf));
        let doc = read("arm.3ds", &chunk(MAIN, &main)).unwrap();
        assert_eq!(doc.json["nodes"][1]["name"], "Arm");
        assert_eq!(doc.json["nodes"][0]["children"], json!([1]));
        assert_eq!(doc.json["animations"][0]["channels"].as_array().unwrap().len(), 2);

        // At rest, everything is where the editor has it (turned Y up).
        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        let mut got: Vec<[f32; 3]> = mesh.prims.iter().flat_map(|p| p.positions.clone()).collect();
        let mut want: Vec<[f32; 3]> = at([5., 0., 0.])
            .into_iter()
            .chain(at([5., 3., 0.]))
            .map(|[x, y, z]| [x, z, -y])
            .collect();
        #[allow(clippy::cast_possible_truncation, reason = "small test values")]
        let key = |p: &[f32; 3]| p.map(|c| (c * 1000.0).round() as i64);
        got.sort_by_key(key);
        want.sort_by_key(key);
        assert!(got.iter().zip(&want).all(|(a, b)| close(*a, *b)), "{got:?} != {want:?}");
    }

    #[test]
    fn cameras_and_lights() {
        let mut cam = f32s(&[0., -10., 0., 0., 0., 0., 0., 50.]);
        cam.extend(chunk(CAM_RANGES, &f32s(&[1., 100.])));
        let mut spot = f32s(&[0., 0., 10.]);
        spot.extend(chunk(COLOR_U8, &[255, 255, 255]));
        spot.extend(chunk(DL_SPOTLIGHT, &f32s(&[0., 0., 0., 30., 60.])));
        let mut editor = object("Tri", &chunk(TRIMESH, &trimesh(&at([0.; 3]), &[[0, 1, 2]], None)));
        editor.extend(object("Cam", &chunk(CAMERA, &cam)));
        editor.extend(object("Spot", &chunk(LIGHT, &spot)));
        let doc = read("lit.3ds", &chunk(MAIN, &chunk(EDITOR, &editor))).unwrap();
        let p = &doc.json["cameras"][0]["perspective"];
        assert_eq!(p["znear"], json!(1.0));
        assert_eq!(p["zfar"], json!(100.0));
        let light = &doc.json["extensions"]["KHR_lights_punctual"]["lights"][0];
        assert_eq!(light["type"], "spot");
        let outer = light["spot"]["outerConeAngle"].as_f64().unwrap();
        assert!((outer - 30f64.to_radians()).abs() < 1e-9);
        // The camera looks along +Y: its -Z turned onto +Y is a quarter turn about +X.
        let q: Vec<f64> = doc.json["nodes"][1]["rotation"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_f64().unwrap())
            .collect();
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((q[0] - h).abs() < 1e-6 && (q[3] - h).abs() < 1e-6, "{q:?}");
        // The spot looks straight down: -Z already.
        let q = doc.json["nodes"][2]["rotation"].as_array().unwrap();
        assert!((q[3].as_f64().unwrap().abs() - 1.0).abs() < 1e-6, "{q:?}");
    }

    /// The flattened mesh's triangles, each as its winding's normal and its corners' normals.
    fn windings(mesh: &crate::mesh::Mesh) -> Vec<([f32; 3], Vec<[f32; 3]>)> {
        let mut out = Vec::new();
        for p in &mesh.prims {
            for t in p.indices.chunks_exact(3) {
                let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
                let n = face_normal(a, b, c);
                let len = n.iter().map(|v| v * v).sum::<f32>().sqrt();
                out.push((n.map(|v| v / len), t.iter().map(|&i| p.normals[i as usize]).collect()));
            }
        }
        out
    }

    #[test]
    fn a_mirroring_mesh_matrix_doesnt_turn_faces_inside_out() {
        // Made facing +Z, then mirrored in X: the vertices are written mirrored, the faces as made.
        let mirrored = |origin: [f32; 3]| {
            let verts: Vec<[f32; 3]> = at([0.; 3]).iter().map(|v| [origin[0] - v[0], v[1], v[2]]).collect();
            let mut t = chunk(VERTICES, &{
                let mut v = 3u16.to_le_bytes().to_vec();
                v.extend(f32s(verts.as_flattened()));
                v
            });
            t.extend(chunk(
                MESH_MATRIX,
                &f32s(&[-1., 0., 0., 0., 1., 0., 0., 0., 1., origin[0], origin[1], origin[2]]),
            ));
            t.extend(chunk(FACES, &u16s(&[1, 0, 1, 2, 0])));
            chunk(TRIMESH, &t)
        };
        // Where the editor has it, and placed by the keyframer through a mirroring node.
        let alone = chunk(MAIN, &chunk(EDITOR, &object("Alone", &mirrored([0.; 3]))));
        let mut main = chunk(EDITOR, &object("Placed", &mirrored([4., 0., 0.])));
        main.extend(chunk(
            KEYFRAMER,
            &node(
                0,
                "Placed",
                u16::MAX,
                [0.; 3],
                &[
                    track(POS_TRACK, &[(0, &[4., 0., 0.])]),
                    track(SCL_TRACK, &[(0, &[-1., 1., 1.])]),
                ],
            ),
        ));
        for file in [alone, chunk(MAIN, &main)] {
            let mesh = read("m.3ds", &file).unwrap().flatten(Pose::Rest).unwrap().unwrap();
            let w = windings(&mesh);
            assert_eq!(w.len(), 1);
            // Z up is Y up.
            assert!(close(w[0].0, [0.0, 1.0, 0.0]), "{w:?}");
            assert!(w[0].1.iter().all(|&n| close(n, [0.0, 1.0, 0.0])), "{w:?}");
        }
    }

    #[test]
    fn smoothing_spans_texture_seams_and_flat_materials_ignore_it() {
        // A fold of two faces in one group, their shared edge's vertices written twice (as a texture seam does).
        let tri = |shading: Option<u16>| {
            let verts = [
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 0., 1.],
            ];
            let mut faces = 2u16.to_le_bytes().to_vec();
            faces.extend(u16s(&[0, 1, 2, 0, 3, 5, 4, 0]));
            let mut fm = b"M\0".to_vec();
            fm.extend(u16s(&[2, 0, 1]));
            faces.extend(chunk(FACE_MATERIAL, &fm));
            faces.extend(chunk(
                SMOOTHING,
                &[1u32, 1].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            ));
            let mut v = 6u16.to_le_bytes().to_vec();
            v.extend(f32s(verts.as_flattened()));
            let mut t = chunk(VERTICES, &v);
            // Too few texture coordinates: the rest are zero.
            let mut uv = 2u16.to_le_bytes().to_vec();
            uv.extend(f32s(&[0., 0., 1., 0.]));
            t.extend(chunk(UVS, &uv));
            t.extend(chunk(FACES, &faces));
            let mut mat = chunk(MAT_NAME, b"M\0");
            if let Some(s) = shading {
                mat.extend(chunk(MAT_SHADING, &s.to_le_bytes()));
            }
            let mut editor = chunk(MATERIAL, &mat);
            editor.extend(object("Fold", &chunk(TRIMESH, &t)));
            chunk(MAIN, &chunk(EDITOR, &editor))
        };
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let mesh = read("s.3ds", &tri(Some(3)))
            .unwrap()
            .flatten(Pose::Rest)
            .unwrap()
            .unwrap();
        let p = &mesh.prims[0];
        assert_eq!(p.uvs.len(), p.positions.len());
        // The fold's edge: (0, 0, 0) and (1, 0, 0), Z up turned Y up, all four of its vertices one normal.
        let edge: Vec<_> = (p.positions.iter().zip(&p.normals))
            .filter(|(q, _)| q[1].abs() < 1e-6 && q[2].abs() < 1e-6)
            .collect();
        assert_eq!(edge.len(), 4, "{p:?}");
        assert!(edge.iter().all(|(_, n)| close(**n, [0.0, h, -h])), "{edge:?}");
        // Shaded flat: each face its own normal.
        let mesh = read("s.3ds", &tri(Some(1)))
            .unwrap()
            .flatten(Pose::Rest)
            .unwrap()
            .unwrap();
        for (face, normals) in windings(&mesh) {
            assert!(normals.iter().all(|&n| close(n, face)), "{face:?} {normals:?}");
        }
    }

    #[test]
    fn an_opacity_map_that_is_the_color_map_cuts_by_its_alpha() {
        let mut mat = chunk(MAT_NAME, b"Leaf\0");
        mat.extend(chunk(MAT_TEXTURE, &chunk(MAP_FILE, b"leaf.png\0")));
        mat.extend(chunk(MAT_OPACMAP, &chunk(MAP_FILE, b"LEAF.PNG\0")));
        let mut t = trimesh(&at([0.; 3]), &[], None);
        let mut faces = 1u16.to_le_bytes().to_vec();
        faces.extend(u16s(&[0, 1, 2, 0]));
        let mut fm = b"Leaf\0".to_vec();
        fm.extend(u16s(&[1, 0]));
        faces.extend(chunk(FACE_MATERIAL, &fm));
        let mut uv = 3u16.to_le_bytes().to_vec();
        uv.extend(f32s(&[0., 0., 1., 0., 0., 1.]));
        t.extend(chunk(UVS, &uv));
        t.extend(chunk(FACES, &faces));
        let mut editor = chunk(MATERIAL, &mat);
        editor.extend(object("Leaf", &chunk(TRIMESH, &t)));
        let mut png = Vec::new();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([0, 200, 0, 100]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let assets = crate::assets::Memory::new().with("leaf.png", png);
        let file = chunk(MAIN, &chunk(EDITOR, &editor));
        let doc = crate::formats::load_bytes(&file, "l.3ds", "3ds", &assets, Keep::ALL).unwrap();
        let leaf = &doc.json["materials"][0];
        assert!(leaf["pbrMetallicRoughness"]["baseColorTexture"].is_object(), "{leaf}");
        assert_eq!(leaf["alphaMode"], "BLEND", "{leaf}");
    }

    /// A PNG of `w` × `h` RGB pixels made by `f`.
    fn rgb_png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        let mut png = Vec::new();
        image::RgbImage::from_fn(w, h, |x, y| image::Rgb(f(x, y)))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    /// A tangent-space normal map's pixels: bumps leaning every way around +Z.
    fn bumps(x: u32, y: u32) -> [u8; 3] {
        #[allow(clippy::cast_precision_loss, reason = "small test values")]
        let (a, b) = ((x as f32 * 0.7).sin() * 0.4, (y as f32 * 0.9).cos() * 0.4);
        let z = (1.0 - a * a - b * b).sqrt();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "0 to 255")]
        [a, b, z].map(|c| ((c + 1.0) * 127.5).round() as u8)
    }

    /// The start of a JPEG: its markers up to a baseline frame header with `components`.
    fn jpeg(components: u8) -> Vec<u8> {
        let mut j = vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 16];
        j.extend(b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
        j.extend([0xFF, 0xFF, 0xC0, 0, 8 + 3 * components, 8, 0, 4, 0, 4, components]);
        for c in 0..components {
            j.extend([c + 1, 0x11, 0]);
        }
        j
    }

    /// A model of one textured triangle per material, each material only a bump map (`file`, at `percent`).
    fn bumped(maps: &[(&str, &str, Option<u16>)]) -> Vec<u8> {
        let mut editor = Vec::new();
        let mut t = trimesh(&at([0.; 3]), &[], None);
        let mut faces = u16::try_from(maps.len()).unwrap().to_le_bytes().to_vec();
        for _ in maps {
            faces.extend(u16s(&[0, 1, 2, 0]));
        }
        for (i, (name, file, percent)) in maps.iter().enumerate() {
            let mut mat = chunk(MAT_NAME, format!("{name}\0").as_bytes());
            let mut map = chunk(MAP_FILE, format!("{file}\0").as_bytes());
            if let Some(p) = percent {
                map.extend(chunk(PERCENT_U16, &p.to_le_bytes()));
            }
            mat.extend(chunk(MAT_BUMPMAP, &map));
            editor.extend(chunk(MATERIAL, &mat));
            let mut fm = format!("{name}\0").into_bytes();
            fm.extend(u16s(&[1, u16::try_from(i).unwrap()]));
            faces.extend(chunk(FACE_MATERIAL, &fm));
        }
        let mut uv = 3u16.to_le_bytes().to_vec();
        uv.extend(f32s(&[0., 0., 1., 0., 0., 1.]));
        t.extend(chunk(UVS, &uv));
        t.extend(chunk(FACES, &faces));
        editor.extend(object("Wall", &chunk(TRIMESH, &t)));
        chunk(MAIN, &chunk(EDITOR, &editor))
    }

    #[test]
    fn a_bump_map_is_the_normal_map_when_its_image_is_one() {
        // Whether this build reads a PNG's pixels.
        let decodes = cfg!(any(
            feature = "webp",
            feature = "skp",
            feature = "tga",
            feature = "bmp",
            feature = "gif",
            feature = "tiff",
            feature = "psd",
            feature = "dds"
        ));
        let mut gray = Vec::new();
        image::GrayImage::from_fn(8, 8, |x, y| image::Luma([u8::try_from(x * 30 + y).unwrap()]))
            .write_to(&mut std::io::Cursor::new(&mut gray), image::ImageFormat::Png)
            .unwrap();
        let assets = crate::assets::Memory::new()
            .with("bricks.png", rgb_png(40, 30, bumps))
            .with(
                "height.png",
                rgb_png(40, 30, |x, y| [u8::try_from((x * 6 + y) % 256).unwrap(); 3]),
            )
            .with("gray_normal.png", gray)
            .with("wall_n.jpg", jpeg(3))
            .with("rough.jpg", jpeg(3))
            .with("mono_nrm.jpg", jpeg(1));
        let file = bumped(&[
            ("Pixels", "bricks.png", Some(50)),
            ("Height", "height.png", None),
            ("Gray", "gray_normal.png", None),
            ("Named", "maps\\wall_n.jpg", None),
            ("Plain", "rough.jpg", None),
            ("Mono", "mono_nrm.jpg", None),
            ("Missing", "gone_normal.png", None),
        ]);
        let doc = crate::formats::load_bytes(&file, "b.3ds", "3ds", &assets, Keep::ALL).unwrap();
        let normal = |name: &str| {
            let m = doc.json["materials"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["name"] == name)
                .unwrap();
            // Every bump map stays in extras.
            assert!(m["extras"]["3ds"]["maps"]["bump"]["file"].is_string(), "{m}");
            m.get("normalTexture").cloned()
        };
        let pixels = normal("Pixels");
        assert_eq!(pixels.is_some(), decodes);
        if decodes {
            assert_eq!(pixels.unwrap()["scale"], json!(0.5));
        }
        assert!(normal("Height").is_none());
        assert!(normal("Gray").is_none());
        assert_eq!(normal("Named").unwrap().get("scale"), None);
        assert!(normal("Plain").is_none());
        assert!(normal("Mono").is_none());
        assert!(normal("Missing").is_none());
        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(
            mesh.materials.iter().filter(|m| m.normal_image.is_some()).count(),
            1 + usize::from(decodes)
        );

        // Images cut short or damaged: never a panic.
        let png = rgb_png(40, 30, bumps);
        for len in (0..png.len()).step_by(3) {
            let mut bad = png.clone();
            bad.truncate(len);
            let assets = crate::assets::Memory::new().with("bricks.png", bad);
            let file = bumped(&[("Pixels", "bricks.png", None)]);
            let _ = crate::formats::load_bytes(&file, "b.3ds", "3ds", &assets, Keep::ALL);
        }
        let mut flipped = png;
        for i in (8..flipped.len()).step_by(5) {
            flipped[i] ^= 0x5A;
        }
        let assets = crate::assets::Memory::new().with("bricks.png", flipped);
        let file = bumped(&[("Pixels", "bricks.png", None)]);
        let _ = crate::formats::load_bytes(&file, "b.3ds", "3ds", &assets, Keep::ALL);
        for len in 0..file.len() {
            let _ = crate::formats::load_bytes(&file[..len], "b.3ds", "3ds", &assets, Keep::ALL);
        }
    }

    #[test]
    fn normal_maps_by_pixels_header_and_name() {
        let flat = vec![[128, 128, 255]; 10];
        assert!(normal_pixels(&flat));
        let mut some_gray = flat.clone();
        some_gray.extend([[200, 200, 200]; 3]);
        assert!(normal_pixels(&some_gray));
        assert!(!normal_pixels(&[[201, 201, 201]; 10]));
        assert!(!normal_pixels(&[[54, 54, 54]; 10]));
        assert!(!normal_pixels(&[[40, 40, 255]; 10]));
        assert!(!normal_pixels(&[]));
        assert_eq!(jpeg_components(&jpeg(1)), Some(1));
        assert_eq!(jpeg_components(&jpeg(3)), Some(3));
        let j = jpeg(3);
        for len in 0..j.len() {
            let _ = jpeg_components(&j[..len]);
        }
        assert_eq!(jpeg_components(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2]), None);
        for name in [
            "brick_n.png",
            "Brick_Normal.tga",
            "C:\\maps\\wall_NRM.jpg",
            "rock-n.png",
            "x_ddn.tga",
            "a_nor.png",
        ] {
            assert!(named_normal_map(name), "{name}");
        }
        for name in ["bump.png", "brick_b.jpg", "n.png", "normalmaps/brick.png", "tin.png"] {
            assert!(!named_normal_map(name), "{name}");
        }
    }

    #[test]
    fn not_3ds_is_refused() {
        assert!(read("x.3ds", b"hello").is_err());
    }
}
