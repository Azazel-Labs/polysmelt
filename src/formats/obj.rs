//! OBJ, with its MTL materials and textures.
//!
//! Read as bytes, a block at a time: lines end in `\n`, `\r\n` or a lone `\r`, statements may
//! continue over lines ending in a backslash, a leading byte order mark is skipped, and text that
//! isn't UTF-8 (names from older tools) is read as Latin-1. Polygons are triangulated in their
//! plane, so concave faces stay inside their outline. Beyond the original format, vertex colors
//! (`v x y z r g b`, or `#MRGB` comment blocks) and the MTL PBR keys (`Pr`, `Pm`, `Ps`, `Pc`, `Pcr`,
//! `aniso`, `anisor`, `Ke`, `norm`, and the `map_Pr`, `map_Pm`, `map_ORM` and `map_RMA` maps) are read.
//! glTF reads occlusion, roughness and metalness from one image's red, green and blue, so maps
//! arranged otherwise are packed into a new PNG that way, when their pixels can be read (PNG, and
//! the formats re-encoded as PNG). The maps are read for that even when the images' bytes aren't
//! kept, so the textures are the same either way. `Kd`, `Ks`, `Ns`, `d` and `illum` are the classic
//! material every reader reads one way (`legacy_material`): `Ns` the highlight's exponent, `Ks` its
//! color, none under `illum` 0 or 1. A file Blender wrote holds a roughness and a specular level
//! there instead, which are read back as Blender wrote them. Faces with no material, or one no
//! library has, take the default material.
//!
//! Corners without a normal get one made from their faces: smooth within a smoothing group
//! (`s 1`, the default), across texture seams, and flat for faces after `s off`. Corners with a
//! normal keep it.
//!
//! Each object (`o`) or group (`g`) is a node drawing its own mesh, with a primitive per material
//! and its points (`p`) and lines (`l`) as primitives of their own. Every MTL statement goes in
//! its material's extras; the file's opening comments, its libraries and the statements not
//! converted (free-form curves and surfaces) go in the asset's. An opening comment naming the
//! exporter (`# Blender 4.2.0`, `# Exported from ...`) is also the source's tool.
//!
//! OBJ names no units, so they're unknown. Its up axis is +Y by common practice (assumed, not
//! declared); the geometry is kept as written.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::io;
#[cfg(any(feature = "fs", test))]
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use serde_json::{Map, Value, json};

use super::Input;
use super::stream::{self, f32_from};

use crate::Result;
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep, Part};
use crate::images::{Images, Source, gray};
use crate::keep::spawn;
use crate::legacy_material::{self, Highlight, Phong, Rgb, Transparency};
use crate::material::{self, Alpha, Clearcoat, Factor, Full, Sheen, Specular, Tex, UvTransform};
use crate::mesh::{Image, Material, Prim, quad, smooth_normals, srgb_to_linear, triangulate};
use crate::num::u32_of;
use crate::source::{Axis, Fact, SourceInfo};
use crate::vec3;

/// Marks a corner without a texture coordinate or normal.
const NONE: u32 = u32::MAX;

/// Set in a corner's normal index when the corner has no normal: the rest is its smoothing
/// group, [`FLAT`] for a flat face.
const MADE: u32 = 0x8000_0000;

/// A corner without a normal on a flat face (`s off`): a vertex of its own.
const FLAT: u32 = MADE;

/// The bytes UTF-8 text may start with.
const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// Whether text starting with `head` is UTF-16 by its byte order mark: little-endian or not.
fn utf16(head: &[u8]) -> Option<bool> {
    match head {
        [0xFF, 0xFE, ..] => Some(true),
        [0xFE, 0xFF, ..] => Some(false),
        _ => None,
    }
}

/// UTF-16 code units from their bytes.
fn units(bytes: &[u8], le: bool) -> Vec<u16> {
    let pairs = bytes.as_chunks::<2>().0.iter();
    pairs
        .map(|&b| {
            if le {
                u16::from_le_bytes(b)
            } else {
                u16::from_be_bytes(b)
            }
        })
        .collect()
}

/// One face corner: position, texture coordinate and normal indices (the last two may be
/// [`NONE`]).
type Corner = [u32; 3];

/// One `newmtl` block of an MTL file.
#[derive(Debug, Default)]
struct MtlEntry {
    color: Option<[f32; 3]>,
    alpha: Option<f32>,
    /// Whether `alpha` is from `d`, which `Tr` doesn't override.
    dissolve: bool,
    emissive: Option<[f32; 3]>,
    roughness: Option<f32>,
    metallic: Option<f32>,
    shininess: Option<f32>,
    color_map: Option<MapRef>,
    normal_map: Option<MapRef>,
    /// `map_Ke`, `map_Ks`, `map_Ps`, and `map_Pr` and `map_Pm` (glTF's metal-roughness texture when one file).
    emissive_map: Option<MapRef>,
    specular_map: Option<MapRef>,
    sheen_map: Option<MapRef>,
    roughness_map: Option<MapRef>,
    metallic_map: Option<MapRef>,
    /// `map_d`: opacity.
    alpha_map: Option<MapRef>,
    /// `map_ORM`: occlusion, roughness and metalness in red, green and blue, as glTF packs them.
    orm_map: Option<MapRef>,
    /// `map_RMA`: roughness, metalness and occlusion in red, green and blue.
    rma_map: Option<MapRef>,
    /// `map_ao`: occlusion in red.
    occlusion_map: Option<MapRef>,
    /// `Ks`, Phong's highlight color.
    specular: Option<[f32; 3]>,
    /// `Ni`.
    ior: Option<f32>,
    /// `Ps`, `Pc`, `Pcr`, `aniso` and `anisor`: the PBR extension's sheen, clearcoat and anisotropy, its
    /// rotation a fraction of a turn.
    sheen: Option<[f32; 3]>,
    clearcoat: Option<f32>,
    clearcoat_roughness: Option<f32>,
    anisotropy: Option<f32>,
    anisotropy_rotation: Option<f32>,
    /// `illum`.
    illum: Option<f32>,
    /// Every statement as written: numbers as numbers, the rest as text.
    raw: Map<String, Value>,
}

/// Text as every reader decodes it: UTF-8 when it is, otherwise Windows-1252.
fn text(b: &[u8]) -> String {
    crate::bytes::text(b).into_owned()
}

/// The first `\n` or `\r` in `rest`. Most bytes are past `\r`, so take one compare.
#[inline]
fn eol(rest: &[u8]) -> Option<usize> {
    let mut from = 0;
    loop {
        let i = from + rest[from..].iter().position(|&b| b <= b'\r')?;
        if matches!(rest[i], b'\n' | b'\r') {
            return Some(i);
        }
        from = i + 1;
    }
}

/// Where the line starting `rest` ends, and where the next starts: after `\n`, `\r\n` or a lone
/// `\r`.
#[inline]
fn line_end(rest: &[u8]) -> (usize, usize) {
    let len = eol(rest).unwrap_or(rest.len());
    let skip = if rest.get(len..len + 2) == Some(b"\r\n") { 2 } else { 1 };
    (len, (len + skip).min(rest.len()))
}

/// Calls `f` with each logical line, ended by `\n`, `\r\n` or a lone `\r`; lines ending in a
/// backslash joined with the next.
fn for_each_line(data: &[u8], mut f: impl FnMut(&[u8])) {
    let mut joined: Vec<u8> = Vec::new();
    let mut rest = data;
    while !rest.is_empty() {
        let (len, next) = line_end(rest);
        let line = &rest[..len];
        rest = &rest[next..];
        if let Some(head) = line.trim_ascii_end().strip_suffix(b"\\") {
            joined.extend_from_slice(head);
            joined.push(b' ');
            continue;
        }
        if joined.is_empty() {
            f(line);
        } else {
            joined.extend_from_slice(line);
            f(&joined);
            joined.clear();
        }
    }
    if !joined.is_empty() {
        f(&joined);
    }
}

/// A line's first word and the rest, trimmed.
fn split_key(line: &[u8]) -> (&[u8], &[u8]) {
    let line = line.trim_ascii_start();
    let end = line.iter().position(u8::is_ascii_whitespace).unwrap_or(line.len());
    (&line[..end], line[end..].trim_ascii())
}

/// Up to `N` leading numbers of `rest`, stopping at a comment; and how many there were.
#[inline]
fn nums<const N: usize>(rest: &[u8]) -> ([f32; N], usize) {
    let mut out = [0.0; N];
    let mut n = 0;
    let mut i = 0;
    let len = rest.len();
    while n < N {
        while i < len && stream::space(rest[i]) {
            i += 1;
        }
        if i == len {
            break;
        }
        let start = i;
        while i < len && !stream::space(rest[i]) {
            i += 1;
        }
        let w = &rest[start..i];
        if w[0] == b'#' {
            break;
        }
        match f32_from(w) {
            Some(x) => out[n] = x,
            None => break,
        }
        n += 1;
    }
    (out, n)
}

/// A texture statement: its file, and the options glTF has a place for.
#[derive(Debug, Clone, PartialEq)]
struct MapRef {
    file: String,
    /// `-o` and `-s`: `uv' = uv * scale + offset`, V up.
    offset: [f32; 2],
    scale: [f32; 2],
    /// `-clamp on`.
    clamp: bool,
    /// `-bm`: a bump or normal map's strength.
    bump: f32,
    /// `-imfchan`: the channel a one-value map reads.
    channel: Pick,
}

/// The channel of an image a one-value map reads (`-imfchan`): red, green, blue, or by default its
/// luminance (which is any channel of a gray image).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Pick {
    Red,
    Green,
    Blue,
    Luma,
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
impl Pick {
    /// The value it picks from a pixel's red, green and blue.
    fn of(self, [r, g, b]: [u8; 3]) -> u8 {
        match self {
            Pick::Red => r,
            Pick::Green => g,
            Pick::Blue => b,
            // Rec. 709's weights, which add up to one: a gray pixel stays as it is.
            Pick::Luma => {
                let y = (2126 * u32::from(r) + 7152 * u32::from(g) + 722 * u32::from(b) + 5000) / 10_000;
                u8::try_from(y).unwrap_or(u8::MAX)
            }
        }
    }
}

/// Where a channel of a packed image comes from: a channel of a map's image, or one value everywhere.
#[derive(Clone, Copy)]
enum Plane<'a> {
    Map(&'a MapRef, Pick),
    Value(u8),
}

impl<'a> Plane<'a> {
    /// A map's image in the map's own channel, or with no map, full (the factor's own value).
    fn of(map: Option<&'a MapRef>) -> Self {
        map.map_or(Plane::Value(255), |m| Plane::Map(m, m.channel))
    }
}

/// A channel of a packed image, its map's file found: the key a packed image is made once by.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Channel {
    File(PathBuf, Pick),
    Value(u8),
}

/// How a material's texture is made from its map.
enum Make<'a> {
    /// The map's image as it is.
    File,
    /// The map's image as it is, when it's gray (the same in every channel); otherwise none.
    Gray,
    /// A new image, red, green and blue from these.
    Pack([Plane<'a>; 3]),
}

/// A PNG of `channels` as its red, green and blue, from the images' decoded pixels: each file read
/// and decoded once. `None` when a file can't be read or decoded (only PNG, and the formats
/// re-encoded as PNG, can), when the images aren't one size, or when no channel is a file.
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
fn pack(channels: &[Channel; 3], assets: &dyn Assets) -> Option<Image> {
    let mut out: Option<(u32, u32, Vec<[u8; 3]>)> = None;
    let mut done = [false; 3];
    for k in 0..3 {
        let Channel::File(path, _) = &channels[k] else {
            continue;
        };
        if done[k] {
            continue;
        }
        // Every channel this file gives, picked in one pass over its pixels.
        let from: [Option<Pick>; 3] = std::array::from_fn(|j| match &channels[j] {
            Channel::File(p, pick) if p == path => Some(*pick),
            _ => None,
        });
        let image = assets.load_image(path)?;
        let (w, h, px) = crate::png::decode(&image.bytes, |rgb| from.map(|f| f.map_or(0, |p| p.of(rgb))))?;
        match &mut out {
            None => out = Some((w, h, px)),
            Some((ow, oh, o)) => {
                if (*ow, *oh) != (w, h) {
                    return None;
                }
                for (a, b) in o.iter_mut().zip(px) {
                    for j in (0..3).filter(|&j| from[j].is_some()) {
                        a[j] = b[j];
                    }
                }
            }
        }
        for (d, f) in done.iter_mut().zip(from) {
            *d |= f.is_some();
        }
    }
    let (w, h, mut px) = out?;
    for (j, c) in channels.iter().enumerate() {
        if let Channel::Value(v) = c {
            for p in &mut px {
                p[j] = *v;
            }
        }
    }
    Image::from_png_or_jpeg(crate::png::encode(w, h, 3, px.as_flattened())?)
}

/// Without a PNG encoder, nothing is packed.
#[cfg(not(any(
    feature = "webp",
    feature = "skp",
    feature = "tga",
    feature = "bmp",
    feature = "gif",
    feature = "tiff",
    feature = "psd",
    feature = "dds"
)))]
fn pack(_: &[Channel; 3], _: &dyn Assets) -> Option<Image> {
    None
}

impl MapRef {
    /// The texture of `image` as this statement lays it out, in glTF's V-down UVs.
    fn tex(&self, image: usize) -> Tex {
        let mut t = Tex::new(image);
        let [su, sv] = self.scale;
        let [ou, ov] = self.offset;
        #[allow(clippy::float_cmp, reason = "the identity is written exactly")]
        let moved = self.scale != [1.0; 2] || self.offset != [0.0; 2];
        if moved {
            t.transform = Some(UvTransform {
                offset: [ou, 1.0 - sv - ov],
                rotation: 0.0,
                scale: [su, sv],
            });
        }
        if self.clamp {
            t.wrap = [material::CLAMP; 2];
        }
        t.scale = self.bump;
        t
    }

    /// Whether it lays its image out as `other` does: the same placement and wrap.
    fn same_layout(&self, other: &MapRef) -> bool {
        #[allow(clippy::float_cmp, reason = "options as written")]
        let same = self.offset == other.offset && self.scale == other.scale;
        same && self.clamp == other.clamp
    }
}

/// A texture statement: the file name after any options (`-s 1 1 1`, `-bm 0.5`, ...), and the options' values.
fn map_ref(rest: &[u8]) -> Option<MapRef> {
    let mut m = MapRef {
        file: String::new(),
        offset: [0.0; 2],
        scale: [1.0; 2],
        clamp: false,
        bump: 1.0,
        channel: Pick::Luma,
    };
    let mut rest = rest.trim_ascii();
    while rest.first() == Some(&b'-') {
        let (opt, after) = split_key(rest);
        let opt = opt.to_ascii_lowercase();
        let args = match opt.as_slice() {
            b"-blendu" | b"-blendv" | b"-bm" | b"-boost" | b"-cc" | b"-clamp" | b"-imfchan" | b"-texres" | b"-type" => {
                1
            }
            b"-mm" => 2,
            // One to three numbers.
            b"-o" | b"-s" | b"-t" => 3,
            _ => break,
        };
        rest = after;
        let mut values = Vec::new();
        for k in 0..args {
            let (w, after) = split_key(rest);
            // Optional arguments end at the first word that isn't a number.
            if k > 0 && args == 3 && f32_from(w).is_none() {
                break;
            }
            values.push(w);
            rest = after;
        }
        let n = |i: usize| values.get(i).and_then(|w| f32_from(w)).filter(|x| x.is_finite());
        match opt.as_slice() {
            b"-o" => m.offset = [n(0).unwrap_or(0.0), n(1).unwrap_or(0.0)],
            b"-s" => m.scale = [n(0).unwrap_or(1.0), n(1).unwrap_or(1.0)],
            b"-clamp" => m.clamp = values.first().is_some_and(|w| w.eq_ignore_ascii_case(b"on")),
            b"-bm" => m.bump = n(0).unwrap_or(1.0),
            b"-imfchan" => {
                m.channel = match values.first().map(|w| w.to_ascii_lowercase()).as_deref() {
                    Some(b"r") => Pick::Red,
                    Some(b"g") => Pick::Green,
                    Some(b"b") => Pick::Blue,
                    _ => Pick::Luma,
                }
            }
            _ => {}
        }
    }
    m.file = text(rest);
    (!rest.is_empty()).then_some(m)
}

/// What a material shows beyond its flat view, from the MTL statements glTF has a place for.
#[derive(Default)]
struct Look {
    color: Option<Tex>,
    normal: Option<Tex>,
    metal_rough: Option<Tex>,
    /// Whether `metal_rough` holds metalness, not one value everywhere in its place.
    metal_in_tex: bool,
    occlusion: Option<Tex>,
    emissive: Option<Tex>,
    /// Cut out by the color texture's alpha (`map_d` naming the `map_Kd` file).
    alpha: Option<Alpha>,
    /// Glass (`illum` 4, 6, 7 or 9) that is see-through.
    transmission: Option<f32>,
    specular: Option<Specular<Tex>>,
    ior: Option<f32>,
    sheen: Option<Sheen<Tex>>,
    clearcoat: Option<Clearcoat<Tex>>,
    anisotropy: Option<f32>,
    /// The anisotropy's rotation, in radians.
    anisotropy_rotation: f32,
    unlit: bool,
}

impl Look {
    fn full(&self, m: &Material) -> Full {
        let mut f = Full::from(m);
        // `guess_texture` may have found a color texture the MTL file doesn't name.
        if let (Some(t), Some(c)) = (&mut f.base_color_tex, &self.color) {
            *t = Tex {
                image: t.image,
                ..c.clone()
            };
        }
        if let (Some(t), Some(n)) = (&mut f.normal_tex, &self.normal) {
            *t = Tex {
                image: t.image,
                ..n.clone()
            };
        }
        if let Some(t) = &self.metal_rough {
            f.metallic_roughness_tex = Some(t.clone());
            // The texture is the value, unscaled, unless the file says otherwise.
            if self.metal_in_tex && m.metallic <= 0.0 {
                f.metallic = 1.0;
            }
        }
        f.occlusion_tex.clone_from(&self.occlusion);
        f.emissive_tex.clone_from(&self.emissive);
        if let Some(a) = self.alpha {
            f.alpha = a;
        }
        if let Some(t) = self.transmission {
            f.transmission = Some(Factor::of(t));
            f.alpha = Alpha::Opaque;
        }
        f.specular.clone_from(&self.specular);
        f.ior = self.ior;
        f.sheen.clone_from(&self.sheen);
        f.clearcoat.clone_from(&self.clearcoat);
        f.anisotropy = self.anisotropy.map(|a| material::Anisotropy {
            strength: Factor::of(a),
            rotation: self.anisotropy_rotation,
        });
        f.unlit = self.unlit;
        f
    }
}

/// Adds an MTL file's materials to `out`.
fn read_mtl(data: &[u8], out: &mut HashMap<String, MtlEntry>) {
    let data = data.strip_prefix(BOM).unwrap_or(data);
    let mut current: Option<String> = None;
    for_each_line(data, |line| {
        let (key, rest) = split_key(line);
        let key = key.to_ascii_lowercase();
        if key == b"newmtl" {
            let name = text(rest);
            out.entry(name.clone()).or_default();
            current = Some(name);
            return;
        }
        let Some(e) = current.as_ref().and_then(|c| out.get_mut(c)) else {
            return;
        };
        let (written, _) = split_key(line);
        if !written.is_empty() && written[0] != b'#' {
            e.raw.insert(text(written), raw_value(rest));
        }
        let (f, n) = nums::<3>(rest);
        let rgb = || (n >= 3).then_some(f).or((n == 1).then_some([f[0]; 3]));
        match key.as_slice() {
            b"kd" => e.color = rgb().or(e.color),
            b"ke" => e.emissive = rgb().or(e.emissive),
            b"d" if n > 0 => (e.alpha, e.dissolve) = (Some(f[0]), true),
            // `d` wins when both are there.
            b"tr" if n > 0 && !e.dissolve => e.alpha = Some(1.0 - f[0]),
            b"pr" if n > 0 => e.roughness = Some(f[0]),
            b"pm" if n > 0 => e.metallic = Some(f[0]),
            b"ns" if n > 0 => e.shininess = Some(f[0]),
            b"ks" => e.specular = rgb().or(e.specular),
            b"ni" if n > 0 => e.ior = Some(f[0]),
            b"ps" => e.sheen = rgb().or(e.sheen),
            b"pc" if n > 0 => e.clearcoat = Some(f[0]),
            b"pcr" if n > 0 => e.clearcoat_roughness = Some(f[0]),
            b"aniso" if n > 0 => e.anisotropy = Some(f[0]),
            b"anisor" if n > 0 => e.anisotropy_rotation = Some(f[0]),
            b"illum" if n > 0 => e.illum = Some(f[0]),
            b"map_kd" => e.color_map = map_ref(rest),
            b"map_bump" | b"bump" | b"norm" | b"map_kn" => e.normal_map = map_ref(rest),
            b"map_ke" => e.emissive_map = map_ref(rest),
            b"map_ks" => e.specular_map = map_ref(rest),
            b"map_ps" => e.sheen_map = map_ref(rest),
            b"map_pr" => e.roughness_map = map_ref(rest),
            b"map_pm" => e.metallic_map = map_ref(rest),
            b"map_d" => e.alpha_map = map_ref(rest),
            b"map_orm" => e.orm_map = map_ref(rest),
            b"map_rma" => e.rma_map = map_ref(rest),
            b"map_ao" => e.occlusion_map = map_ref(rest),
            _ => {}
        }
    });
}

/// A statement's arguments: a number, a list of numbers, or failing that the text.
fn raw_value(rest: &[u8]) -> Value {
    let words: Vec<&[u8]> = rest.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty()).collect();
    let numbers: Option<Vec<f64>> = words
        .iter()
        .map(|w| {
            std::str::from_utf8(w)
                .ok()?
                .parse()
                .ok()
                .filter(|x: &f64| x.is_finite())
        })
        .collect();
    match numbers {
        Some(n) if n.len() == 1 => json!(n[0]),
        Some(n) if !n.is_empty() => json!(n),
        _ => json!(text(rest)),
    }
}

/// The material libraries `mtllib` names: the whole rest of the line as one file (names may
/// contain spaces), or failing that each word as a file.
fn read_libraries(libs: &[String], dir: &Path, assets: &dyn Assets) -> HashMap<String, MtlEntry> {
    let mut out = HashMap::new();
    for lib in libs {
        let found = match assets.find(lib, dir) {
            Some(p) => vec![p],
            None => lib.split_whitespace().filter_map(|w| assets.find(w, dir)).collect(),
        };
        for p in found {
            if let Some(data) = assets.read(&p) {
                read_mtl(&data, &mut out);
            }
        }
    }
    out
}

/// Text up to and including the last line break that ends a statement (not one continued by a
/// backslash). A `\r` that ends the text may be the first half of a `\r\n`, so isn't cut at.
fn statements(d: &[u8]) -> Option<usize> {
    let mut end = d.len();
    loop {
        let i = d[..end].iter().rposition(|&b| b == b'\n' || b == b'\r')?;
        if d[i] == b'\r' && i + 1 == d.len() {
            end = i;
            continue;
        }
        let line = &d[..i];
        let kept = line.iter().rposition(|&b| !matches!(b, b' ' | b'\t' | b'\r' | b'\x0c'));
        if kept.is_none_or(|k| line[k] != b'\\') {
            return Some(i + 1);
        }
        end = i;
    }
}

/// An OBJ index (1-based, or negative counting back from the last) into `len` items.
#[inline]
fn resolve(w: &[u8], len: usize) -> Option<u32> {
    let n = stream::int::<i64>(w)?;
    let len = i64::try_from(len).ok()?;
    let i = if n < 0 { len + n } else { n - 1 };
    if (0..len).contains(&i) {
        u32::try_from(i).ok().filter(|&i| i != NONE)
    } else {
        None
    }
}

/// Where the line after a line break at `i` in `s` starts: past `\n`, `\r\n` or a lone `\r`.
#[inline]
fn after_break(s: &[u8], i: usize) -> usize {
    if s[i] == b'\r' && s.get(i + 1) == Some(&b'\n') {
        i + 2
    } else {
        i + 1
    }
}

/// A line's numbers from `i` on, for [`Reader::quick`]: up to `N` of them, how many, and where
/// the next line starts. Words past the `N`th are ignored, as [`nums`] does. `None` for anything
/// [`nums`] should read: a word that isn't a plain number, spacing other than spaces and tabs,
/// or a line continued by a backslash.
#[inline]
fn quick_nums<const N: usize>(s: &[u8], mut i: usize) -> Option<([f32; N], usize, usize)> {
    let mut out = [0.0; N];
    let mut n = 0;
    let len = s.len();
    loop {
        while i < len && matches!(s[i], b' ' | b'\t') {
            i += 1;
        }
        if i == len {
            return Some((out, n, len));
        }
        if matches!(s[i], b'\n' | b'\r') {
            return Some((out, n, after_break(s, i)));
        }
        if n == N {
            let end = eol(&s[i..]).map_or(len, |k| i + k);
            if s[..end].trim_ascii_end().ends_with(b"\\") {
                return None;
            }
            return Some((out, n, if end == len { len } else { after_break(s, end) }));
        }
        let (x, used) = stream::f32_lead(&s[i..])?;
        out[n] = x;
        n += 1;
        i += used;
    }
}

/// An index from `*i` on, resolved like [`resolve`]; `*i` moves past it.
#[inline]
fn quick_index(s: &[u8], i: &mut usize, len: usize) -> Option<u32> {
    let rest = s.get(*i..)?;
    let neg = rest.first() == Some(&b'-');
    let (n, digits) = stream::few_digits(&rest[usize::from(neg)..]);
    if digits == 0 || digits > 18 {
        return None;
    }
    *i += usize::from(neg) + digits;
    let len = u64::try_from(len).ok()?;
    // 1 to `len`, counting from the first or (negative) back from the last.
    if n == 0 || n > len {
        return None;
    }
    let at = if neg { len - n } else { n - 1 };
    u32::try_from(at).ok().filter(|&at| at != NONE)
}

/// A face line's corners (after the `f`), for [`Reader::quick`], and where the next line starts.
/// `lens` are how many positions, texture coordinates and normals there are so far; a corner
/// without a normal takes `smooth`. `None` for anything [`Reader::face`] should read: a corner
/// naming something that isn't there, a comment, or anything else unusual.
#[inline]
fn quick_corners(s: &[u8], lens: [usize; 3], smooth: u32, face: &mut Vec<Corner>) -> Option<usize> {
    let len = s.len();
    let ends = |i: usize| i >= len || stream::space(s[i]);
    let mut i = 1;
    loop {
        while i < len && matches!(s[i], b' ' | b'\t') {
            i += 1;
        }
        if i == len {
            return Some(len);
        }
        if matches!(s[i], b'\n' | b'\r') {
            return Some(after_break(s, i));
        }
        let p = quick_index(s, &mut i, lens[0])?;
        let (mut t, mut n) = (NONE, smooth);
        if i < len && s[i] == b'/' {
            i += 1;
            if i < len && s[i] != b'/' && !ends(i) {
                t = quick_index(s, &mut i, lens[1])?;
            }
            if i < len && s[i] == b'/' {
                i += 1;
                if !ends(i) {
                    n = quick_index(s, &mut i, lens[2])?;
                }
            }
        }
        if !ends(i) {
            return None;
        }
        face.push([p, t, n]);
    }
}

/// An index as written, from `*i` on, as a [`Piece`] keeps it: `n - 1`, as `i32` bits when
/// negative. `None` for 0, more than 18 digits, or past `i32`; `*i` moves past it.
#[inline]
fn written_index(s: &[u8], i: &mut usize) -> Option<u32> {
    let rest = s.get(*i..)?;
    let neg = rest.first() == Some(&b'-');
    let (n, digits) = stream::few_digits(&rest[usize::from(neg)..]);
    if digits == 0 || digits > 18 || n == 0 {
        return None;
    }
    *i += usize::from(neg) + digits;
    let n = i32::try_from(n).ok()?;
    Some(if neg {
        (-n - 1).cast_unsigned()
    } else {
        (n - 1).cast_unsigned()
    })
}

/// A [`written_index`] into `len` items, resolved as [`quick_index`] does.
#[inline]
fn resolve_written(w: u32, len: usize) -> Option<u32> {
    let w = w.cast_signed();
    let at = if w < 0 {
        // Back from the last: -1 is the last.
        len.checked_sub(usize::try_from(-i64::from(w) - 1).ok()?)?
    } else {
        usize::try_from(w).ok().filter(|&w| w < len)?
    };
    u32::try_from(at).ok().filter(|&at| at != NONE)
}

/// A [`written_corners`] corner resolved into `lens` positions, texture coordinates and normals,
/// as [`quick_corners`] resolves one; without a normal, it takes `smooth`. `None` when it names
/// something that isn't there.
#[inline]
fn resolved([p, t, n]: Corner, lens: [usize; 3], smooth: u32) -> Option<Corner> {
    let one = |w: u32, len: usize, none: u32| if w == NONE { Some(none) } else { resolve_written(w, len) };
    Some([
        resolve_written(p, lens[0])?,
        one(t, lens[1], NONE)?,
        one(n, lens[2], smooth)?,
    ])
}

/// A face's written corners resolved in place, as [`resolved`] resolves each; false, with the
/// face part resolved, when a corner names something that isn't there.
fn resolve_face(face: &mut [Corner], lens: [usize; 3], smooth: u32) -> bool {
    for c in face {
        match resolved(*c, lens, smooth) {
            Some(r) => *c = r,
            None => return false,
        }
    }
    true
}

/// A face line's corners as written (after the `f`), like [`quick_corners`] but with the
/// indices left for [`Reader::replay`] to resolve: as [`written_index`] keeps them, [`NONE`] where
/// a corner has none. Where the next line starts, whether any index is negative, and the most
/// positions, texture coordinates and normals the positive ones need.
#[inline]
fn written_corners(s: &[u8], face: &mut Vec<Corner>) -> Option<(usize, bool, [u32; 3])> {
    let len = s.len();
    let ends = |i: usize| i >= len || stream::space(s[i]);
    let (mut back, mut need) = (0, [0; 3]);
    let mut i = 1;
    loop {
        while i < len && matches!(s[i], b' ' | b'\t') {
            i += 1;
        }
        if i == len || matches!(s[i], b'\n' | b'\r') {
            let next = if i == len { len } else { after_break(s, i) };
            return Some((next, back != 0, need));
        }
        let mut c = [NONE; 3];
        c[0] = written_index(s, &mut i)?;
        if i < len && s[i] == b'/' {
            i += 1;
            if i < len && s[i] != b'/' && !ends(i) {
                c[1] = written_index(s, &mut i)?;
            }
            if i < len && s[i] == b'/' {
                i += 1;
                if !ends(i) {
                    c[2] = written_index(s, &mut i)?;
                }
            }
        }
        if !ends(i) {
            return None;
        }
        for k in 0..3 {
            if c[k] != NONE {
                // A negative index has the top bit set.
                back |= c[k] >> 31;
                need[k] = need[k].max(c[k].wrapping_add(1));
            }
        }
        face.push(c);
    }
}

/// One statement of a [`Piece`], or a run of them.
#[derive(Clone, Copy)]
enum Op {
    /// Positions (three numbers each), texture coordinates (two) and unit normals (three).
    V(usize),
    Vt(usize),
    Vn(usize),
    /// A position and its color: six numbers.
    Colored,
    /// A run of faces of three or more corners, `corners` in all; whether any index is negative,
    /// and the most positions, texture coordinates and normals the positive indices need.
    F {
        faces: usize,
        corners: usize,
        back: bool,
        need: [u32; 3],
    },
    /// Lines for [`Reader::chunk`] to read, from `start` to `end`.
    Lines {
        start: usize,
        end: usize,
    },
}

/// A stretch of statements read on a worker thread: the `v`, `vt`, `vn` and `f` lines read in
/// one pass, as [`Reader::quick`] would, with the rest left as lines. Offsets are from the
/// piece's start.
#[derive(Default)]
struct Piece {
    ops: Vec<Op>,
    nums: Vec<f32>,
    /// The faces' corners as [`written_corners`] keeps them, and each face's corner count.
    corners: Vec<Corner>,
    sizes: Vec<u32>,
}

impl Piece {
    fn read(&mut self, d: &[u8]) {
        self.ops.clear();
        self.nums.clear();
        self.corners.clear();
        self.sizes.clear();
        // Room for typical lines, so the workers don't all wait on the allocator as it grows.
        self.ops.reserve(d.len() / 1024);
        self.nums.reserve(d.len() / 16);
        self.corners.reserve(d.len() / 24);
        self.sizes.reserve(d.len() / 64);
        let mut at = 0;
        while at < d.len() {
            if let Some(next) = self.quick(d, at) {
                at = next;
                continue;
            }
            // A line the reader reads in full, with any lines that continue it.
            let start = at;
            loop {
                let end = eol(&d[at..]).map_or(d.len(), |i| at + i);
                let more = d[at..end].trim_ascii_end().ends_with(b"\\");
                at = (end + if d.get(end..end + 2) == Some(b"\r\n") { 2 } else { 1 }).min(d.len());
                if !more || at == d.len() {
                    break;
                }
            }
            if let Some(Op::Lines { end, .. }) = self.ops.last_mut()
                && *end == start
            {
                *end = at;
            } else {
                self.ops.push(Op::Lines { start, end: at });
            }
        }
        // The faces are kept until the whole model is read.
        trim_to_len(&mut self.corners);
        trim_to_len(&mut self.sizes);
    }

    /// A run of `op`, one longer.
    #[inline]
    fn run(&mut self, op: fn(usize) -> Op) {
        match (self.ops.last_mut(), op(0)) {
            (Some(Op::V(n)), Op::V(_)) | (Some(Op::Vt(n)), Op::Vt(_)) | (Some(Op::Vn(n)), Op::Vn(_)) => *n += 1,
            _ => self.ops.push(op(1)),
        }
    }

    /// A line [`Reader::quick`] would read, read; where the next line starts. What it would drop
    /// (a position of fewer than three numbers, a texture coordinate of none, a normal of other
    /// than three, a face of fewer than three corners) leaves nothing.
    #[inline]
    fn quick(&mut self, d: &[u8], at: usize) -> Option<usize> {
        let s = &d[at..];
        let next = match s {
            [b'v', b' ' | b'\t', ..] => {
                let (f, n, next) = quick_nums::<7>(s, 2)?;
                if n >= 6 {
                    self.nums.extend_from_slice(&f[..6]);
                    self.ops.push(Op::Colored);
                } else if n >= 3 {
                    self.nums.extend_from_slice(&f[..3]);
                    self.run(Op::V);
                }
                next
            }
            [b'v', b't', b' ' | b'\t', ..] => {
                let (f, n, next) = quick_nums::<2>(s, 3)?;
                if n > 0 {
                    self.nums.extend_from_slice(&f);
                    self.run(Op::Vt);
                }
                next
            }
            [b'v', b'n', b' ' | b'\t', ..] => {
                let (f, n, next) = quick_nums::<3>(s, 3)?;
                if n == 3 {
                    self.nums.extend_from_slice(&vec3::unit_or(f, [0.0, 1.0, 0.0]));
                    self.run(Op::Vn);
                }
                next
            }
            [b'f', b' ' | b'\t', ..] => {
                let from = self.corners.len();
                let Some((next, back, need)) = written_corners(s, &mut self.corners) else {
                    self.corners.truncate(from);
                    return None;
                };
                let n = self.corners.len() - from;
                if n < 3 {
                    self.corners.truncate(from);
                } else {
                    self.sizes.push(u32_of(n));
                    if let Some(Op::F {
                        faces,
                        corners,
                        back: b,
                        need: most,
                    }) = self.ops.last_mut()
                    {
                        *faces += 1;
                        *corners += n;
                        *b |= back;
                        for (m, k) in most.iter_mut().zip(need) {
                            *m = (*m).max(k);
                        }
                    } else {
                        self.ops.push(Op::F {
                            faces: 1,
                            corners: n,
                            back,
                            need,
                        });
                    }
                }
                next
            }
            _ => return None,
        };
        Some(at + next)
    }
}

/// Where `chunk` from `at` on is cut into pieces of about `piece` bytes: after line breaks that
/// end a statement, as [`statements`] finds them. The first cut is `at` and the last the end.
fn pieces(chunk: &[u8], at: usize, piece: usize) -> Vec<usize> {
    let mut cuts = vec![at];
    let mut from = at;
    while chunk.len() - from > piece {
        let mut to = from + piece.max(1);
        // A statement longer than a piece: look further.
        let cut = loop {
            if to >= chunk.len() {
                break chunk.len();
            }
            if let Some(n) = statements(&chunk[from..to]) {
                break from + n;
            }
            to = to.saturating_add(to - from);
        };
        if cut == chunk.len() {
            break;
        }
        cuts.push(cut);
        from = cut;
    }
    cuts.push(chunk.len());
    cuts
}

/// The bytes in a piece read on a worker thread, and in a block of the model read ahead of them.
const PIECE: usize = 1 << 20;
const THREADED_BLOCK: usize = 8 << 20;

/// A block of the model's text for [`Reader::read_threaded`], ending where a statement ends.
type Text<'a> = io::Result<Cow<'a, [u8]>>;

/// Work for a worker: a piece to read (its number, and its bytes in a block), or a piece's
/// faces to make vertices of ([`Shape::piece`]).
enum Job<'a> {
    Read(usize, Arc<Cow<'a, [u8]>>, usize, usize),
    Shape(usize, Box<Planned>),
}

/// Work a worker did: a piece read, or a piece's faces made vertices; `None` if it panicked.
enum Done {
    Read(usize, Option<Piece>),
    Shape(usize, Option<Box<Planned>>),
}

/// Things done with, kept to use again: their room is kept, so threads seldom allocate.
type Spare<T> = Mutex<Vec<T>>;

/// Something done with, if any.
fn take_spare<T: Default>(spare: &Spare<T>) -> T {
    spare.lock().ok().and_then(|mut v| v.pop()).unwrap_or_default()
}

/// `t` kept to use again.
fn put_spare<T>(spare: &Spare<T>, t: T) {
    if let Ok(mut v) = spare.lock() {
        v.push(t);
    }
}

/// The model's text sent on in blocks, for [`Reader::read_threaded`]: in place when it's in
/// memory, or read from the file at `path` into blocks from `used`.
#[cfg_attr(not(feature = "fs"), allow(unused_variables, reason = "only files have paths"))]
fn feed<'a>(bytes: Option<&'a [u8]>, path: &Path, tx: &mpsc::SyncSender<Text<'a>>, used: &Spare<Vec<u8>>) {
    if let Some(b) = bytes {
        slice_blocks(b, THREADED_BLOCK, tx);
    } else {
        #[cfg(feature = "fs")]
        match std::fs::File::open(path) {
            Ok(f) => read_blocks(f, THREADED_BLOCK, tx, used),
            Err(e) => drop(tx.send(Err(e))),
        }
    }
}

/// `d` sent on in blocks of about `block` bytes, each ending where a statement ends.
fn slice_blocks<'a>(d: &'a [u8], block: usize, tx: &mpsc::SyncSender<Text<'a>>) {
    for w in pieces(d, 0, block).windows(2) {
        if tx.send(Ok(Cow::Borrowed(&d[w[0]..w[1]]))).is_err() {
            break;
        }
    }
}

/// `source` read and sent on in blocks of about `block` bytes, each ending where a statement
/// ends, or at the end. Read into blocks from `used` where there are any.
#[cfg(any(feature = "fs", test))]
fn read_blocks(mut source: impl Read, block: usize, tx: &mpsc::SyncSender<Text<'_>>, used: &Spare<Vec<u8>>) {
    let mut rest = Vec::new();
    loop {
        let mut buf = take_spare(used);
        buf.clear();
        buf.reserve(block.max(rest.len() * 2).max(1));
        buf.append(&mut rest);
        let cut = loop {
            let want = buf.capacity() - buf.len();
            match (&mut source)
                .take(u64::try_from(want).unwrap_or(u64::MAX))
                .read_to_end(&mut buf)
            {
                Ok(n) if n < want => break None,
                Ok(_) => {}
                Err(e) => {
                    drop(tx.send(Err(e)));
                    return;
                }
            }
            if let Some(n) = statements(&buf).filter(|&n| n > 0) {
                break Some(n);
            }
            // A statement longer than the block: read more of it.
            buf.reserve(buf.len());
        };
        if let Some(n) = cut {
            rest.extend_from_slice(&buf[n..]);
            buf.truncate(n);
        }
        if (!buf.is_empty() && tx.send(Ok(Cow::Owned(buf))).is_err()) || cut.is_none() {
            return;
        }
    }
}

/// `f` called on each of `items`, on `threads` threads (this one among them), each thread with
/// a `T` of its own from `state`. Helpers that can't be started leave their share to the rest:
/// on a platform without threads, this one does it all.
fn each<I: Send, T>(items: Vec<I>, threads: usize, state: impl Fn() -> T + Sync, f: impl Fn(&mut T, I) + Sync) {
    let helpers = threads.min(items.len()).saturating_sub(1);
    let items = Mutex::new(items.into_iter());
    let run = || {
        let mut t = state();
        while let Some(i) = items.lock().ok().and_then(|mut i| i.next()) {
            f(&mut t, i);
        }
    };
    std::thread::scope(|s| {
        for _ in 0..helpers {
            if !spawn(s, run) {
                break;
            }
        }
        run();
    });
}

/// A worker: work done as it comes, and sent back.
fn work(jobs: &Mutex<mpsc::Receiver<Job<'_>>>, done: &mpsc::Sender<Done>, spare: &Spare<Piece>) {
    let mut shape = Shape::default();
    while let Some(job) = jobs.lock().ok().and_then(|j| j.recv().ok()) {
        let (sent, panic) = match job {
            Job::Read(k, text, start, end) => {
                let mut p = take_spare(spare);
                let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.read(&text[start..end])));
                drop(text);
                match read {
                    Ok(()) => (done.send(Done::Read(k, Some(p))), None),
                    Err(e) => (done.send(Done::Read(k, None)), Some(e)),
                }
            }
            Job::Shape(k, mut p) => {
                let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shape.piece(&mut p)));
                match made {
                    Ok(()) => (done.send(Done::Shape(k, Some(p))), None),
                    Err(e) => (done.send(Done::Shape(k, None)), Some(e)),
                }
            }
        };
        if let Some(e) = panic {
            // So the reader stops waiting; the scope passes the panic on.
            std::panic::resume_unwind(e);
        }
        if sent.is_err() {
            break;
        }
    }
}

/// One part's triangles in one material, built as the faces are read, and its points and lines.
struct Group {
    part: usize,
    material: usize,
    prim: Prim,
    /// Vertex colors as the file gives them, converted once the scale is known.
    raw_colors: Vec<[f32; 3]>,
    /// Vertices whose normal is made from their faces (zero until then).
    made: usize,
    /// Positions (file indices) drawn as points, and in pairs as lines.
    points: Vec<u32>,
    lines: Vec<u32>,
}

/// Faces of one group, kept while the model is read on threads: a run of a piece's written
/// faces, or faces [`Reader::face`] read in full.
struct Run {
    group: u32,
    /// The normal index of a corner without one.
    smooth: u32,
    /// How many positions, texture coordinates and normals there were before the faces.
    lens: [usize; 3],
    /// Whether every corner names something there, counting from the first.
    ok: bool,
    /// Read in full, so resolved: in the piece's `slow` and `slow_sizes`.
    slow: bool,
    /// How many faces, and where the first face and its first corner are.
    faces: usize,
    face: usize,
    corner: usize,
    /// Made on a worker: how many faces are kept, how many corners their triangles have, and
    /// how many vertices of the piece's own they make.
    kept: usize,
    indices: usize,
    vertices: usize,
    /// Of those vertices, the ones no earlier piece has, how many have normals to make, and
    /// whether any has a texture coordinate.
    fresh: usize,
    made: usize,
    uv: bool,
}

/// A piece's faces, kept while the model is read on threads and made into vertices on a worker
/// as soon as the piece is planned ([`Shape::piece`]); then, once the whole model is read, into
/// triangles ([`Reader::build`]).
#[derive(Default)]
struct Planned {
    /// Written faces, as the worker read them.
    corners: Vec<Corner>,
    sizes: Vec<u32>,
    /// Faces read in full, resolved.
    slow: Vec<Corner>,
    slow_sizes: Vec<u32>,
    runs: Vec<Run>,
    /// Sent to a worker to make vertices of, and back with them.
    away: bool,
    shaped: bool,
    /// The piece's vertices, in first-use order: position, texture coordinate, normal and group.
    vertices: Vec<[u32; 4]>,
    /// The faces kept, run after run: how many corners each has, and each corner's vertex.
    kept: Vec<u32>,
    ids: Vec<u32>,
    /// The vertices that may be shared (all but flat faces'), by range of positions: those in
    /// range `k` are `order[starts[k]..starts[k + 1]]`.
    order: Vec<u32>,
    starts: Vec<u32>,
}

impl Planned {
    /// The vertices sorted into `ranges` ranges of positions, `1 << shift` positions each.
    fn sort(&mut self, shift: u32, ranges: usize, fill: &mut Vec<u32>) {
        let Planned {
            vertices,
            order,
            starts,
            ..
        } = self;
        starts.clear();
        starts.resize(ranges + 1, 0);
        for &[at, _, n, _] in vertices.iter() {
            if n != FLAT {
                starts[(at >> shift) as usize + 1] += 1;
            }
        }
        for k in 1..starts.len() {
            starts[k] += starts[k - 1];
        }
        fill.clear();
        fill.extend_from_slice(starts);
        order.clear();
        order.resize(starts[ranges] as usize, 0);
        for (j, &[at, _, n, _]) in vertices.iter().enumerate() {
            if n != FLAT {
                let fill = &mut fill[(at >> shift) as usize];
                order[*fill as usize] = u32_of(j);
                *fill += 1;
            }
        }
    }
}

/// What a worker keeps between pieces for [`Shape::piece`].
#[derive(Default)]
struct Shape {
    /// The piece's newest vertex at each position from the lowest it uses, and each vertex's
    /// next at its position; or, for a piece whose positions are spread far apart, the vertices
    /// by position, texture coordinate, normal and group.
    head: Vec<u32>,
    next: Vec<u32>,
    map: HashMap<[u32; 4], u32>,
}

impl Shape {
    /// A piece's faces resolved, and their corners made vertices in first-use order, as
    /// [`Reader::add_face`] would make them, but counting only the piece's own.
    fn piece(&mut self, p: &mut Planned) {
        let Planned {
            corners,
            sizes,
            slow,
            slow_sizes,
            runs,
            vertices,
            kept,
            ids,
            shaped,
            ..
        } = p;
        // Resolved in place; a dropped face's first position is `NONE`.
        let (mut low, mut high, mut all) = (NONE, 0, 0);
        for r in runs.iter() {
            let (cs, ss) = if r.slow {
                (&mut slow[..], &slow_sizes[..])
            } else {
                (&mut corners[..], &sizes[..])
            };
            let mut at = r.corner;
            for &n in &ss[r.face..r.face + r.faces] {
                let face = &mut cs[at..at + n as usize];
                at += n as usize;
                if !r.slow {
                    if r.ok {
                        for c in face.iter_mut().filter(|c| c[2] == NONE) {
                            c[2] = r.smooth;
                        }
                    } else if !resolve_face(face, r.lens, r.smooth) {
                        face[0][0] = NONE;
                        continue;
                    }
                }
                for c in face.iter() {
                    low = low.min(c[0]);
                    high = high.max(c[0]);
                }
                all += face.len();
            }
        }
        let span = if low <= high { (high - low) as usize + 1 } else { 0 };
        // Positions close enough together to look up in a list, as most pieces' are: one a few
        // times the size of the corners costs less to clear than a map costs to hash into.
        let listed = span <= 32 * all + 65536;
        if listed {
            self.head.clear();
            self.head.resize(span, NONE);
            self.next.clear();
        } else {
            self.map.clear();
        }
        ids.reserve_exact(all);
        for r in runs.iter_mut() {
            let (cs, ss) = if r.slow {
                (&slow[..], &slow_sizes[..])
            } else {
                (&corners[..], &sizes[..])
            };
            let from = vertices.len();
            let mut at = r.corner;
            for &n in &ss[r.face..r.face + r.faces] {
                let face = &cs[at..at + n as usize];
                at += n as usize;
                if face[0][0] == NONE {
                    continue;
                }
                for &c in face {
                    let id = self.vertex(vertices, r.group, c, low, listed);
                    ids.push(id);
                }
                kept.push(n);
                r.kept += 1;
                r.indices += 3 * (face.len() - 2);
            }
            r.vertices = vertices.len() - from;
        }
        // Done with: the room goes back now, not once every piece is made.
        *corners = Vec::new();
        *sizes = Vec::new();
        *slow = Vec::new();
        *slow_sizes = Vec::new();
        *shaped = true;
    }

    /// The piece's vertex for a corner in group `g`, added if it's new; as [`Reader::vertex`]
    /// finds one, a flat face's corner always new. Looked up in the list from position `low` on
    /// when `listed`.
    #[inline]
    fn vertex(&mut self, vertices: &mut Vec<[u32; 4]>, g: u32, c: Corner, low: u32, listed: bool) -> u32 {
        let [p, t, n] = c;
        let want = [p, t, n, g];
        let id = u32_of(vertices.len());
        if n == FLAT {
            if listed {
                self.next.push(NONE);
            }
        } else if listed {
            let slot = &mut self.head[(p - low) as usize];
            let head = *slot;
            let mut at = head;
            while at != NONE {
                if vertices[at as usize] == want {
                    return at;
                }
                at = self.next[at as usize];
            }
            *slot = id;
            self.next.push(head);
        } else {
            match self.map.entry(want) {
                std::collections::hash_map::Entry::Occupied(e) => return *e.get(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(id);
                }
            }
        }
        vertices.push(want);
        id
    }
}

/// What a worker keeps between ranges of positions for [`Same::range`]: the newest vertex at
/// each position in the range, and for each vertex its group, texture coordinate, normal and
/// where it is among all pieces' vertices, and the next at its position.
#[derive(Default)]
struct Same {
    head: Vec<u32>,
    keys: Vec<[u32; 4]>,
    next: Vec<u32>,
}

impl Same {
    /// For each piece's vertices at positions in range `k` (from `k << shift`), the first
    /// piece's same vertex, set in `same` (by where the vertices are among all pieces', each
    /// piece's from `from`) when it's an earlier piece's.
    fn range(&mut self, k: usize, shift: u32, planned: &[Planned], from: &[usize], same: &[AtomicU32]) {
        let low = u32_of(k << shift);
        self.head.clear();
        self.head.resize(1 << shift, NONE);
        self.keys.clear();
        self.next.clear();
        for (p, &from) in planned.iter().zip(from) {
            for &j in &p.order[p.starts[k] as usize..p.starts[k + 1] as usize] {
                let [at, t, n, g] = p.vertices[j as usize];
                let want = [g, t, n];
                let slot = &mut self.head[(at - low) as usize];
                let mut id = *slot;
                while id != NONE {
                    let key = self.keys[id as usize];
                    if key[..3] == want {
                        same[from + j as usize].store(key[3], Ordering::Relaxed);
                        break;
                    }
                    id = self.next[id as usize];
                }
                if id == NONE {
                    self.next.push(*slot);
                    *slot = u32_of(self.keys.len());
                    self.keys.push([g, t, n, u32_of(from + j as usize)]);
                }
            }
        }
    }
}

/// `v` made `n` longer, with `zero` (memory the system clears, for a new vector).
fn grow<T: Clone>(v: &mut Vec<T>, n: usize, zero: T) {
    if v.is_empty() {
        *v = vec![zero; n];
    } else {
        v.resize(v.len() + n, zero);
    }
}

/// The first `n` of `s`, which keeps the rest.
#[allow(clippy::mut_mut, reason = "the slice left is the caller's")]
fn cut<'a, T>(s: &mut &'a mut [T], n: usize) -> &'a mut [T] {
    let (head, more) = std::mem::take(s).split_at_mut(n);
    *s = more;
    head
}

/// Where [`Reader::build`] writes one run's new vertices: in its group, and among all.
struct Fresh<'a> {
    positions: &'a mut [[f32; 3]],
    normals: &'a mut [[f32; 3]],
    uvs: &'a mut [[f32; 2]],
    colors: &'a mut [[f32; 3]],
    key: &'a mut [[u32; 3]],
    local: &'a mut [u32],
    /// The first one's index in its group.
    base: u32,
}

/// The statements read so far.
struct Reader {
    v: Vec<[f32; 3]>,
    /// Vertex colors, one per position once any position has one.
    colors: Vec<[f32; 3]>,
    /// Positions `#MRGB` blocks have colored so far.
    mrgb: usize,
    /// The normal index of a corner without one: [`MADE`] and the smoothing group.
    smooth: u32,
    vt: Vec<[f32; 2]>,
    vn: Vec<[f32; 3]>,
    /// Output vertices, found by position: the newest output vertex at each position, and for
    /// each output vertex the next at the same position, its group, texture coordinate and
    /// normal, and its index in its group.
    first: Vec<u32>,
    next: Vec<u32>,
    key: Vec<[u32; 3]>,
    local: Vec<u32>,
    /// Positions some output vertex uses.
    used: usize,
    libs: Vec<String>,
    /// Material names in first-use order (`""` for none), and the current one.
    materials: Vec<String>,
    material_of: HashMap<String, usize>,
    material: usize,
    /// Parts (an object's and a group's name) in first-use order, the current names, and the
    /// current part once a face needs it.
    parts: Vec<(Option<String>, Option<String>)>,
    part_of: HashMap<(Option<String>, Option<String>), usize>,
    object: Option<String>,
    group: Option<String>,
    part: Option<usize>,
    /// By part and material, in first-use order.
    groups: Vec<Group>,
    group_of: HashMap<(usize, usize), usize>,
    current: Option<usize>,
    face: Vec<Corner>,
    /// The vertex of each corner of a polygon.
    ids: Vec<u32>,
    pts: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    triangles: usize,
    /// The most triangles the file may have.
    limit: usize,
    points: usize,
    lines: usize,
    /// Faces read, and the file offset of the first, for sizing the arrays.
    faces: usize,
    face_from: usize,
    /// The comments before the first statement.
    comments: Vec<String>,
    /// Statements read but not converted, by keyword.
    skipped: Vec<(Vec<u8>, usize)>,
    /// Pieces read on workers and taken, kept to read more into.
    spare: Vec<Piece>,
    /// While read on threads: faces are kept, by piece, to be made into vertices once all are
    /// read; the most triangles they make; and whether to read again on one thread, for a file
    /// these can't stand for.
    deferring: bool,
    planned: Vec<Planned>,
    bound: usize,
    restart: bool,
}

impl Reader {
    fn new(limit: usize) -> Self {
        Reader {
            v: Vec::new(),
            colors: Vec::new(),
            mrgb: 0,
            smooth: MADE | 1,
            vt: Vec::new(),
            vn: Vec::new(),
            first: Vec::new(),
            next: Vec::new(),
            key: Vec::new(),
            local: Vec::new(),
            used: 0,
            libs: Vec::new(),
            materials: vec![String::new()],
            material_of: HashMap::from([(String::new(), 0)]),
            material: 0,
            parts: Vec::new(),
            part_of: HashMap::new(),
            object: None,
            group: None,
            part: None,
            groups: Vec::new(),
            group_of: HashMap::new(),
            current: None,
            face: Vec::new(),
            ids: Vec::new(),
            pts: Vec::new(),
            tris: Vec::new(),
            triangles: 0,
            limit,
            points: 0,
            lines: 0,
            faces: 0,
            face_from: 0,
            comments: Vec::new(),
            skipped: Vec::new(),
            spare: Vec::new(),
            deferring: false,
            planned: Vec::new(),
            bound: 0,
            restart: false,
        }
    }

    /// The statements in `chunk` from `at` on, the chunk starting `read` bytes into a file of
    /// `size`. A statement continued by a backslash collects in `joined`, maybe into the next
    /// chunk. Plain lines are read in one pass when `quick`.
    fn chunk(&mut self, chunk: &[u8], mut at: usize, read: usize, size: usize, joined: &mut Vec<u8>, quick: bool) {
        while at < chunk.len() {
            if quick
                && joined.is_empty()
                && let Some(next) = self.quick(chunk, at, read + at, size)
            {
                at = next;
                continue;
            }
            let end = eol(&chunk[at..]).map_or(chunk.len(), |i| at + i);
            let line = &chunk[at..end];
            let start = read + at;
            // A line ends at `\n`, `\r\n` or a lone `\r`.
            at = end + if chunk.get(end..end + 2) == Some(b"\r\n") { 2 } else { 1 };
            if let Some(head) = line.trim_ascii_end().strip_suffix(b"\\") {
                joined.extend_from_slice(head);
                joined.push(b' ');
                continue;
            }
            if joined.is_empty() {
                self.line(line, start, size);
            } else {
                joined.extend_from_slice(line);
                self.line(joined, start, size);
                joined.clear();
            }
        }
    }

    /// One statement, which starts `at` bytes into a file of `size`.
    #[inline]
    fn line(&mut self, line: &[u8], at: usize, size: usize) {
        let line = line.trim_ascii_start();
        let end = line.iter().position(|&b| stream::space(b)).unwrap_or(line.len());
        let (key, rest) = (&line[..end], &line[end..]);
        match key {
            b"v" => {
                let (f, n) = nums::<7>(rest);
                self.position(f, n);
            }
            b"vt" => {
                let (f, n) = nums::<2>(rest);
                if n > 0 {
                    self.vt.push([f[0], f[1]]);
                }
            }
            b"vn" => {
                let (f, n) = nums::<3>(rest);
                self.normal(f, n);
            }
            b"f" | b"fo" => self.face(rest, at, size),
            b"s" => self.smoothing(rest),
            b"mtllib" => self.libs.push(text(rest.trim_ascii())),
            b"usemtl" => {
                let next = self.materials.len();
                self.material = *self
                    .material_of
                    .entry(text(rest.trim_ascii()))
                    .or_insert_with_key(|name| {
                        self.materials.push(name.clone());
                        next
                    });
                self.current = None;
            }
            b"o" | b"g" => {
                let name = text(rest.trim_ascii());
                let name = (!name.is_empty()).then_some(name);
                if key == b"o" {
                    self.object = name;
                } else {
                    self.group = name;
                }
                (self.part, self.current) = (None, None);
            }
            b"l" | b"p" => self.loose(key == b"l", rest),
            _ => self.other(key, line),
        }
    }

    /// A statement this reader doesn't convert: a comment, kept while the file is starting, or
    /// one counted as skipped.
    #[cold]
    fn other(&mut self, key: &[u8], line: &[u8]) {
        if key.is_empty() {
            return;
        }
        if key == b"#MRGB" {
            self.mrgb(&line[key.len()..]);
            return;
        }
        if key[0] == b'#' {
            let starting = self.v.is_empty() && self.libs.is_empty() && self.parts.is_empty();
            if starting && self.object.is_none() && self.group.is_none() && self.comments.len() < 64 {
                self.comments.push(text(line[1..].trim_ascii()));
            }
            return;
        }
        if let Some(k) = self.skipped.iter_mut().find(|k| k.0 == key) {
            k.1 += 1;
        } else if self.skipped.len() < 64 {
            self.skipped.push((key.to_vec(), 1));
        }
    }

    /// Position `i`'s vertex color; the positions before it without one are white.
    fn color(&mut self, i: usize, c: [f32; 3]) {
        if self.colors.len() <= i {
            self.colors.resize(i, [1.0; 3]);
            self.colors.push(c);
        } else {
            self.colors[i] = c;
        }
    }

    /// `s`: the smoothing group of the faces that follow, `off` or `0` for flat faces.
    #[cold]
    fn smoothing(&mut self, rest: &[u8]) {
        let w = rest
            .trim_ascii()
            .split(|&b| stream::space(b))
            .next()
            .unwrap_or_default();
        let group = if w.eq_ignore_ascii_case(b"off") {
            0
        } else {
            // Other words (`on`) are a group of their own.
            stream::int::<i64>(w).map_or(1, |n| u32::try_from(n & 0x7FFF_FFFF).unwrap_or(1))
        };
        self.smooth = MADE | group;
    }

    /// A `#MRGB` block: a mask and an RGB color per position, in order, as eight hex digits
    /// each (`MMRRGGBB`).
    #[cold]
    fn mrgb(&mut self, rest: &[u8]) {
        if self.deferring && self.bound > 0 {
            // It may color a position faces have used, whose vertices take the color it had then.
            self.restart = true;
            return;
        }
        let hex = |w: &[u8]| -> Option<f32> {
            let s = std::str::from_utf8(w).ok()?;
            u8::from_str_radix(s, 16).ok().map(|b| f32::from(b) / 255.0)
        };
        for w in rest.trim_ascii().chunks_exact(8) {
            let (Some(r), Some(g), Some(b)) = (hex(&w[2..4]), hex(&w[4..6]), hex(&w[6..8])) else {
                break;
            };
            // Past the positions read so far: colored as they come.
            let i = self.mrgb;
            self.color(i, [r, g, b]);
            if self.colors.len() < self.v.len() {
                self.colors.resize(self.v.len(), [1.0; 3]);
            }
            self.mrgb += 1;
        }
    }

    /// The group for the current part and material, added if it's new.
    fn group(&mut self) -> usize {
        if let Some(g) = self.current {
            return g;
        }
        let part = if let Some(p) = self.part {
            p
        } else {
            let next = self.parts.len();
            let key = (self.object.clone(), self.group.clone());
            let p = *self.part_of.entry(key).or_insert_with_key(|k| {
                self.parts.push(k.clone());
                next
            });
            self.part = Some(p);
            p
        };
        let next = self.groups.len();
        let material = self.material;
        let g = *self.group_of.entry((part, material)).or_insert_with(|| {
            self.groups.push(Group {
                part,
                material,
                prim: Prim::default(),
                raw_colors: Vec::new(),
                made: 0,
                points: Vec::new(),
                lines: Vec::new(),
            });
            next
        });
        self.current = Some(g);
        g
    }

    /// `p` (points) or, when `lines`, `l` (a line through each position in turn); the texture
    /// coordinates a line may name aren't kept. Dropped if it names a position that isn't there.
    fn loose(&mut self, lines: bool, rest: &[u8]) {
        let mut at = std::mem::take(&mut self.face);
        at.clear();
        for w in rest.split(|&b| stream::space(b)).filter(|w| !w.is_empty()) {
            if w[0] == b'#' {
                break;
            }
            let Some(p) = w.split(|&c| c == b'/').next().and_then(|p| resolve(p, self.v.len())) else {
                at.clear();
                break;
            };
            at.push([p, NONE, NONE]);
        }
        if lines && at.len() >= 2 {
            let g = self.group();
            let out = &mut self.groups[g].lines;
            for pair in at.windows(2) {
                out.extend([pair[0][0], pair[1][0]]);
            }
            self.lines += at.len() - 1;
        } else if !lines && !at.is_empty() {
            let g = self.group();
            self.groups[g].points.extend(at.iter().map(|c| c[0]));
            self.points += at.len();
        }
        self.face = at;
    }

    fn face(&mut self, rest: &[u8], at: usize, size: usize) {
        let mut face = std::mem::take(&mut self.face);
        face.clear();
        let mut ok = true;
        let mut i = 0;
        let n = rest.len();
        loop {
            while i < n && stream::space(rest[i]) {
                i += 1;
            }
            if i == n {
                break;
            }
            let start = i;
            while i < n && !stream::space(rest[i]) {
                i += 1;
            }
            let w = &rest[start..i];
            if w[0] == b'#' {
                break;
            }
            let mut parts = w.split(|&c| c == b'/');
            let p = parts.next().and_then(|p| resolve(p, self.v.len()));
            let mut opt = |len: usize| match parts.next() {
                None | Some(b"") => Some(NONE),
                Some(s) => resolve(s, len),
            };
            let (t, nm) = (opt(self.vt.len()), opt(self.vn.len()));
            let (Some(p), Some(t), Some(nm)) = (p, t, nm) else {
                // A corner naming something that isn't there: drop the face.
                ok = false;
                break;
            };
            face.push([p, t, if nm == NONE { self.smooth } else { nm }]);
        }
        if ok {
            self.add_face(&face, at, size);
        }
        self.face = face;
    }

    /// A `v`, `vt`, `vn` or `f` statement alone on a plain line starting `at` in `d`, read in one
    /// pass; where the next line starts. `None`, with nothing read, for anything else (a comment,
    /// a backslash, other spacing, a number or index that isn't plain), which [`Reader::line`]
    /// reads instead.
    #[inline]
    fn quick(&mut self, d: &[u8], at: usize, start: usize, size: usize) -> Option<usize> {
        let s = &d[at..];
        let next = match s {
            [b'v', b' ' | b'\t', ..] => {
                let (f, n, next) = quick_nums::<7>(s, 2)?;
                self.position(f, n);
                next
            }
            [b'v', b't', b' ' | b'\t', ..] => {
                let (f, n, next) = quick_nums::<2>(s, 3)?;
                if n > 0 {
                    self.vt.push(f);
                }
                next
            }
            [b'v', b'n', b' ' | b'\t', ..] => {
                let (f, n, next) = quick_nums::<3>(s, 3)?;
                self.normal(f, n);
                next
            }
            [b'f', b' ' | b'\t', ..] => {
                let mut face = std::mem::take(&mut self.face);
                face.clear();
                let lens = [self.v.len(), self.vt.len(), self.vn.len()];
                let next = quick_corners(s, lens, self.smooth, &mut face);
                if next.is_some() {
                    self.add_face(&face, start, size);
                }
                self.face = face;
                next?
            }
            _ => return None,
        };
        Some(at + next)
    }

    /// The text `feed` sends, on a thread of its own, read as [`Reader::chunk`] reads it but on
    /// `threads` threads: cut into pieces of about `piece` bytes where statements end, read on
    /// workers a few pieces ahead, and taken here in order: on two threads, their faces added
    /// as they come; on more, the faces kept, then made into vertices and triangles on the
    /// threads too ([`Reader::build`]). The bytes read, or
    /// `None` for a file to read again on one thread: one coloring positions (`#MRGB`) after
    /// faces have used them, or one whose faces may make more triangles than the limit; or
    /// whenever threads can't be started.
    fn read_threaded<'a>(
        &mut self,
        feed: impl FnOnce(&mpsc::SyncSender<Text<'a>>, &Spare<Vec<u8>>) + Send,
        size: usize,
        threads: usize,
        piece: usize,
    ) -> Result<Option<usize>> {
        /// A block's text, where it starts in the model, where its pieces start, and its first
        /// piece's number.
        struct Block<'a> {
            text: Arc<Cow<'a, [u8]>>,
            read: usize,
            cuts: Vec<usize>,
            first: usize,
        }
        // A statement continued over lines ending in a backslash.
        let mut joined = Vec::new();
        // Pieces taken, and blocks, to read into again.
        let spare = Mutex::new(std::mem::take(&mut self.spare));
        let used = Mutex::new(Vec::new());
        let (text_tx, texts) = mpsc::sync_channel(2);
        let (job_tx, jobs) = mpsc::channel::<Job<'a>>();
        let jobs = Mutex::new(jobs);
        let (done_tx, done) = mpsc::channel();
        // With one worker, this thread has time to make the vertices as the pieces come; with
        // more, they're made on the threads too.
        let deferring = threads > 2;
        self.deferring = deferring;
        let read = std::thread::scope(|s| -> Result<usize> {
            // Dropped on returning, which ends the threads.
            let (job_tx, texts, done) = (job_tx, texts, done);
            let used = &used;
            // Where threads can't be had (a platform without them, or none left), the model is
            // read again on this one; those that did start end as the channels drop.
            for _ in 1..threads {
                let (jobs, done_tx, spare) = (&jobs, done_tx.clone(), &spare);
                if !spawn(s, move || work(jobs, &done_tx, spare)) {
                    self.restart = true;
                    return Ok(0);
                }
            }
            if !spawn(s, move || feed(&text_tx, used)) {
                self.restart = true;
                return Ok(0);
            }
            drop(done_tx);
            let ahead = 4 * threads;
            let mut blocks: VecDeque<Block<'a>> = VecDeque::new();
            let mut ready: HashMap<usize, Piece> = HashMap::new();
            let (mut pieces_in, mut sent, mut next, mut read) = (0, 0, 0, 0);
            // Pieces' faces away on workers; and those made vertices of here.
            let mut away = 0;
            let mut shape = Shape::default();
            // Work a worker did, kept; false if it panicked.
            let took = |d: Done, ready: &mut HashMap<usize, Piece>, planned: &mut [Planned], away: &mut usize| {
                match d {
                    Done::Read(k, Some(p)) => drop(ready.insert(k, p)),
                    Done::Shape(k, Some(p)) => {
                        planned[k] = *p;
                        *away -= 1;
                    }
                    _ => return false,
                }
                true
            };
            loop {
                while sent < next + ahead {
                    if let Some(b) = blocks.iter().find(|b| sent < b.first + b.cuts.len() - 1) {
                        let k = sent - b.first;
                        drop(job_tx.send(Job::Read(sent, Arc::clone(&b.text), b.cuts[k], b.cuts[k + 1])));
                        sent += 1;
                    } else if let Some(text) = if sent > next {
                        // Workers have pieces to read: take what's ready meanwhile.
                        texts.try_recv().ok()
                    } else {
                        texts.recv().ok()
                    } {
                        let text = text?;
                        let at = if read == 0 && text.starts_with(BOM) {
                            BOM.len()
                        } else {
                            0
                        };
                        let cuts = pieces(&text, at, piece);
                        let (first, len) = (pieces_in, text.len());
                        pieces_in += cuts.len() - 1;
                        blocks.push_back(Block {
                            text: Arc::new(text),
                            read,
                            cuts,
                            first,
                        });
                        read += len;
                    } else {
                        break;
                    }
                }
                let Some(b) = blocks.front() else { break };
                let mut p = loop {
                    if let Some(p) = ready.remove(&next) {
                        break p;
                    }
                    if !done
                        .recv()
                        .is_ok_and(|d| took(d, &mut ready, &mut self.planned, &mut away))
                    {
                        // A worker panicked, which the scope passes on.
                        return Ok(read);
                    }
                };
                // Blocks and pieces end where statements end, so nothing is left continued
                // between them but at the end.
                let k = next - b.first;
                self.replay(&b.text, b.cuts[k], &mut p, b.read, size, &mut joined);
                put_spare(&spare, p);
                while let Ok(d) = done.try_recv() {
                    if !took(d, &mut ready, &mut self.planned, &mut away) {
                        return Ok(read);
                    }
                }
                // Its faces made vertices of meanwhile: here, if there'd be nothing to do but wait
                // for the next piece; or else on a worker.
                if let Some(p) = self.planned.last_mut()
                    && !p.runs.is_empty()
                {
                    if ready.contains_key(&(next + 1)) {
                        let p = std::mem::replace(
                            p,
                            Planned {
                                away: true,
                                ..Planned::default()
                            },
                        );
                        drop(job_tx.send(Job::Shape(self.planned.len() - 1, Box::new(p))));
                        away += 1;
                    } else {
                        shape.piece(p);
                    }
                }
                next += 1;
                if next == b.first + b.cuts.len() - 1 {
                    if let Some(b) = blocks.pop_front()
                        && let Ok(Cow::Owned(text)) = Arc::try_unwrap(b.text)
                    {
                        put_spare(used, text);
                    }
                    // Past the limit, maybe: one thread reads no further than it.
                    self.restart |= self.bound + self.triangles > self.limit;
                }
                if self.restart {
                    return Ok(read);
                }
            }
            while away > 0 {
                let Ok(Done::Shape(k, Some(p))) = done.recv() else {
                    return Ok(read);
                };
                self.planned[k] = *p;
                away -= 1;
            }
            Ok(read)
        });
        self.spare = spare.into_inner().unwrap_or_default();
        let read = read?;
        if !joined.is_empty() && !self.restart {
            self.line(&joined, read, size);
        }
        self.deferring = false;
        if self.restart || self.bound + self.triangles > self.limit {
            return Ok(None);
        }
        if deferring {
            self.build(threads);
        }
        Ok(Some(read))
    }

    /// A piece read on a worker, starting `from` bytes into `chunk`, taken as [`Reader::chunk`]
    /// would have read it, its faces kept for [`Reader::build`].
    fn replay(&mut self, chunk: &[u8], from: usize, p: &mut Piece, read: usize, size: usize, joined: &mut Vec<u8>) {
        if self.deferring {
            self.planned.push(Planned {
                corners: std::mem::take(&mut p.corners),
                sizes: std::mem::take(&mut p.sizes),
                ..Planned::default()
            });
        }
        let mut nums = &p.nums[..];
        let mut take = |n: usize| {
            let (head, rest) = nums.split_at(n);
            nums = rest;
            head
        };
        let (mut face, mut corner) = (0, 0);
        for &op in &p.ops {
            match op {
                Op::V(n) => {
                    self.v.extend_from_slice(take(3 * n).as_chunks().0);
                    self.first.resize(self.v.len(), NONE);
                    if !self.colors.is_empty() && self.colors.len() < self.v.len() {
                        self.colors.resize(self.v.len(), [1.0; 3]);
                    }
                }
                Op::Vt(n) => self.vt.extend_from_slice(take(2 * n).as_chunks().0),
                Op::Vn(n) => self.vn.extend_from_slice(take(3 * n).as_chunks().0),
                Op::Colored => {
                    let mut f = [0.0; 7];
                    f[..6].copy_from_slice(take(6));
                    self.position(f, 6);
                }
                Op::F {
                    faces,
                    corners,
                    back,
                    need,
                } => {
                    if self.deferring {
                        self.plan(faces, corners, back, need, [face, corner]);
                    } else {
                        let (cs, ss) = (&p.corners[corner..corner + corners], &p.sizes[face..face + faces]);
                        self.written_faces(cs, ss, back, need, read + from, size);
                    }
                    face += faces;
                    corner += corners;
                }
                Op::Lines { start, end } => self.chunk(&chunk[..from + end], from + start, read, size, joined, false),
            }
        }
    }

    /// A run of the newest piece's written faces, `faces` of them with `corners` corners from
    /// the face and corner at `at`, kept for [`Reader::build`] in the current group: added at its
    /// first face whose corners are all there, as [`Reader::add_face`] adds one. `back` and `need`
    /// are as [`Op::F`] has them.
    fn plan(&mut self, faces: usize, corners: usize, back: bool, need: [u32; 3], at: [usize; 2]) {
        let lens = [self.v.len(), self.vt.len(), self.vn.len()];
        let ok = !back && (0..3).all(|k| usize::try_from(need[k]).is_ok_and(|n| n <= lens[k]));
        let Some(p) = self.planned.last() else { return };
        let some = ok || {
            let mut c = at[1];
            p.sizes[at[0]..at[0] + faces].iter().any(|&n| {
                let face = &p.corners[c..c + n as usize];
                c += n as usize;
                face.iter().all(|&w| resolved(w, lens, 0).is_some())
            })
        };
        if !some {
            return;
        }
        let group = u32_of(self.group());
        let smooth = self.smooth;
        self.bound += corners - 2 * faces;
        if let Some(p) = self.planned.last_mut() {
            p.runs.push(Run {
                group,
                smooth,
                lens,
                ok,
                slow: false,
                faces,
                face: at[0],
                corner: at[1],
                kept: 0,
                indices: 0,
                vertices: 0,
                fresh: 0,
                made: 0,
                uv: false,
            });
        }
    }

    /// A run of written faces, `corners` of them with each one's count in `sizes`, added as
    /// [`Reader::quick`] adds them, those naming something that isn't there dropped. `back` and
    /// `need` are as [`Op::F`] has them; `at` is where the piece starts in the model.
    fn written_faces(&mut self, corners: &[Corner], sizes: &[u32], back: bool, need: [u32; 3], at: usize, size: usize) {
        let lens = [self.v.len(), self.vt.len(), self.vn.len()];
        let ok = !back && (0..3).all(|k| usize::try_from(need[k]).is_ok_and(|n| n <= lens[k]));
        let smooth = self.smooth;
        let mut face = std::mem::take(&mut self.face);
        let mut c = 0;
        for &n in sizes {
            face.clear();
            face.extend_from_slice(&corners[c..c + n as usize]);
            c += n as usize;
            if ok {
                for w in face.iter_mut().filter(|w| w[2] == NONE) {
                    w[2] = smooth;
                }
            } else if !resolve_face(&mut face, lens, smooth) {
                continue;
            }
            self.add_face(&face, at, size);
        }
        self.face = face;
    }

    /// A face [`Reader::face`] read, kept for [`Reader::build`] in group `g`.
    fn defer(&mut self, g: usize, face: &[Corner]) {
        if self.planned.last().is_none_or(|p| p.away || p.shaped) {
            self.planned.push(Planned::default());
        }
        let Some(p) = self.planned.last_mut() else { return };
        self.bound += face.len() - 2;
        let group = u32_of(g);
        match p.runs.last_mut() {
            Some(r) if r.slow && r.group == group => r.faces += 1,
            _ => p.runs.push(Run {
                group,
                smooth: NONE,
                lens: [0; 3],
                ok: true,
                slow: true,
                faces: 1,
                face: p.slow_sizes.len(),
                corner: p.slow.len(),
                kept: 0,
                indices: 0,
                vertices: 0,
                fresh: 0,
                made: 0,
                uv: false,
            }),
        }
        p.slow.extend_from_slice(face);
        p.slow_sizes.push(u32_of(face.len()));
    }

    /// The faces kept while reading on threads, made into vertices and triangles as
    /// [`Reader::add_face`] would have made them in order, all on `threads` threads: each
    /// piece's own vertices found, if no worker found them yet; then, by range of positions,
    /// which of those an earlier piece has; then the new ones written, in order; and the faces
    /// triangulated.
    #[allow(clippy::too_many_lines, reason = "one pass after another, each short")]
    fn build(&mut self, threads: usize) {
        let mut planned = std::mem::take(&mut self.planned);
        // A few ranges of positions a thread.
        let mut shift = 0;
        while self.v.len() >> shift > 8 * threads {
            shift += 1;
        }
        let ranges = (self.v.len() >> shift) + 1;
        each(
            planned.iter_mut().collect(),
            threads,
            || (Shape::default(), Vec::new()),
            |(s, fill), p| {
                if !p.shaped {
                    s.piece(p);
                }
                p.sort(shift, ranges, fill);
            },
        );

        // Where each piece's vertices are among all pieces', and which earlier piece's vertex
        // each is, if any.
        let mut from = Vec::with_capacity(planned.len());
        let mut all = 0;
        for p in &planned {
            from.push(all);
            all += p.vertices.len();
        }
        let same: Vec<AtomicU32> = (0..all).map(|_| AtomicU32::new(NONE)).collect();
        each((0..ranges).collect(), threads, Same::default, |s, k| {
            s.range(k, shift, &planned, &from, &same);
        });
        each(
            planned.iter_mut().zip(&from).collect(),
            threads,
            || (),
            |(), (p, &from)| {
                let mut j = from;
                for r in &mut p.runs {
                    for &[_, t, n, _] in &p.vertices[j - from..j - from + r.vertices] {
                        if same[j].load(Ordering::Relaxed) == NONE {
                            r.fresh += 1;
                            r.made += usize::from(n & MADE != 0);
                            r.uv |= t != NONE;
                        }
                        j += 1;
                    }
                }
                p.order = Vec::new();
                p.starts = Vec::new();
            },
        );

        // Room for the new vertices and the triangles.
        let mut per_group = vec![(0, 0, 0, false); self.groups.len()];
        for r in planned.iter().flat_map(|p| &p.runs) {
            let g = &mut per_group[r.group as usize];
            g.0 += r.fresh;
            g.1 += r.indices;
            g.2 += r.made;
            g.3 |= r.uv;
            self.faces += r.kept;
            self.triangles += r.indices / 3;
        }
        let fresh = per_group.iter().map(|g| g.0).sum();
        // The vertices by position, as made normals need them.
        if per_group.iter().any(|g| g.2 > 0) || self.groups.iter().any(|g| g.made > 0) {
            self.next.reserve_exact(fresh);
            for (p, &from) in planned.iter().zip(&from) {
                for (j, &[at, _, n, _]) in p.vertices.iter().enumerate() {
                    if same[from + j].load(Ordering::Relaxed) != NONE {
                        continue;
                    }
                    let head = self.first[at as usize];
                    if head == NONE {
                        self.used += 1;
                    }
                    if n == FLAT {
                        self.next.push(NONE);
                    } else {
                        self.first[at as usize] = u32_of(self.next.len());
                        self.next.push(head);
                    }
                }
            }
        }
        let (key_from, colored) = (self.key.len(), !self.colors.is_empty());
        grow(&mut self.key, fresh, [0; 3]);
        grow(&mut self.local, fresh, 0);
        let mut ids = vec![0; all];
        let mut rest: Vec<_> = self
            .groups
            .iter_mut()
            .zip(&per_group)
            .map(|(grp, &(n, indices, made, uv))| {
                let prim = &mut grp.prim;
                let base = prim.positions.len();
                grow(&mut prim.positions, n, [0.0; 3]);
                grow(&mut prim.normals, n, [0.0; 3]);
                let uvs = if uv || !prim.uvs.is_empty() {
                    prim.uvs.resize(base, [0.0, 1.0]);
                    grow(&mut prim.uvs, n, [0.0; 2]);
                    &mut prim.uvs[base..]
                } else {
                    &mut []
                };
                let colors = if colored {
                    grp.raw_colors.resize(base, [1.0; 3]);
                    grow(&mut grp.raw_colors, n, [0.0; 3]);
                    &mut grp.raw_colors[base..]
                } else {
                    &mut []
                };
                grp.made += made;
                let from = prim.indices.len();
                grow(&mut prim.indices, indices, 0);
                let fresh = Fresh {
                    positions: &mut prim.positions[base..],
                    normals: &mut prim.normals[base..],
                    uvs,
                    colors,
                    key: &mut [],
                    local: &mut [],
                    base: u32_of(base),
                };
                (fresh, &mut prim.indices[from..])
            })
            .collect();
        // Each run's share, in order.
        let (mut key, mut local, mut ids_left) = (&mut self.key[key_from..], &mut self.local[key_from..], &mut ids[..]);
        let mut writes = Vec::with_capacity(planned.len());
        let mut fills = Vec::with_capacity(planned.len());
        for (p, &from) in planned.iter().zip(&from) {
            let (these, more) = ids_left.split_at_mut(p.vertices.len());
            ids_left = more;
            let mut runs = Vec::with_capacity(p.runs.len());
            let mut outs = Vec::with_capacity(p.runs.len());
            for r in &p.runs {
                let (f, indices) = &mut rest[r.group as usize];
                let n = r.fresh;
                // None for a group without them.
                let (uvs, colors) = (n.min(f.uvs.len()), n.min(f.colors.len()));
                runs.push(Fresh {
                    positions: cut(&mut f.positions, n),
                    normals: cut(&mut f.normals, n),
                    uvs: cut(&mut f.uvs, uvs),
                    colors: cut(&mut f.colors, colors),
                    key: cut(&mut key, n),
                    local: cut(&mut local, n),
                    base: f.base,
                });
                f.base += u32_of(n);
                outs.push(cut(indices, r.indices));
            }
            writes.push((p, from, these, runs));
            fills.push((from, outs));
        }
        let (v, vt, vn, colors) = (&self.v, &self.vt, &self.vn, &self.colors);
        each(
            writes,
            threads,
            || (),
            |(), (p, from, ids, runs)| {
                let mut j = 0;
                for (r, f) in p.runs.iter().zip(runs) {
                    let mut k = 0;
                    for &[at, t, n, g] in &p.vertices[j..j + r.vertices] {
                        if same[from + j].load(Ordering::Relaxed) == NONE {
                            f.positions[k] = v[at as usize];
                            f.normals[k] = if n & MADE == 0 { vn[n as usize] } else { [0.0; 3] };
                            if let Some(uv) = f.uvs.get_mut(k) {
                                // OBJ puts the texture origin bottom-left; glTF top-left.
                                *uv = vt.get(t as usize).map_or([0.0, 1.0], |q| [q[0], 1.0 - q[1]]);
                            }
                            if let Some(c) = f.colors.get_mut(k) {
                                *c = colors.get(at as usize).copied().unwrap_or([1.0; 3]);
                            }
                            f.key[k] = [g, t, n];
                            let id = f.base + u32_of(k);
                            f.local[k] = id;
                            ids[j] = id;
                            k += 1;
                        }
                        j += 1;
                    }
                }
            },
        );

        // Each piece's faces triangulated, its vertices numbered in their groups; and what the
        // piece kept freed on the thread that's done with it.
        let ids = &ids;
        let fills: Vec<_> = planned.into_iter().zip(fills).collect();
        each(
            fills,
            threads,
            || (Vec::new(), Vec::new(), Vec::new()),
            |(corners, pts, tris), (p, (from, outs))| {
                let (mut at, mut faces) = (0, &p.kept[..]);
                for (r, out) in p.runs.iter().zip(outs) {
                    let mut o = 0;
                    for &n in &faces[..r.kept] {
                        let face = &p.ids[at..at + n as usize];
                        at += n as usize;
                        corners.clear();
                        corners.extend(face.iter().map(|&k| {
                            let j = from + k as usize;
                            let s = same[j].load(Ordering::Relaxed);
                            ids[if s == NONE { j } else { s as usize }]
                        }));
                        let pos = |k: u32| v[p.vertices[k as usize][0] as usize];
                        if n == 3 {
                            out[o..o + 3].copy_from_slice(corners);
                            o += 3;
                        } else if let Some(t) = <&[u32; 4]>::try_from(face).ok().and_then(|q| quad(&q.map(pos))) {
                            for &k in t.iter().flatten() {
                                out[o] = corners[k as usize];
                                o += 1;
                            }
                        } else {
                            pts.clear();
                            pts.extend(face.iter().map(|&k| pos(k)));
                            tris.clear();
                            triangulate(pts, tris);
                            for &k in tris.iter().flatten() {
                                out[o] = corners[k as usize];
                                o += 1;
                            }
                        }
                    }
                    faces = &faces[r.kept..];
                }
            },
        );
    }

    /// A `v` statement's `n` numbers: a position, with a color when there are six or more.
    #[inline]
    fn position(&mut self, f: [f32; 7], n: usize) {
        if n < 3 {
            return;
        }
        self.v.push([f[0], f[1], f[2]]);
        self.first.push(NONE);
        if n >= 6 {
            self.color(self.v.len() - 1, [f[3], f[4], f[5]]);
        } else if !self.colors.is_empty() && self.colors.len() < self.v.len() {
            self.colors.resize(self.v.len(), [1.0; 3]);
        }
    }

    /// A `vn` statement's `n` numbers: a normal, made unit length, when there are three.
    #[inline]
    fn normal(&mut self, f: [f32; 3], n: usize) {
        if n == 3 {
            self.vn.push(vec3::unit_or(f, [0.0, 1.0, 0.0]));
        }
    }

    /// A face's corners, triangulated into the current group; fewer than three are dropped.
    fn add_face(&mut self, face: &[Corner], at: usize, size: usize) {
        if face.len() >= 3 {
            let g = self.group();
            if self.deferring {
                self.defer(g, face);
                return;
            }
            if face.len() == 3 {
                for &c in face {
                    let id = self.vertex(g, c);
                    self.groups[g].prim.indices.push(id);
                }
                self.triangles += 1;
            } else if let Some(t) = <&[Corner; 4]>::try_from(face)
                .ok()
                .and_then(|q| quad(&q.map(|c| self.v[c[0] as usize])))
            {
                let ids: [u32; 4] = std::array::from_fn(|k| self.vertex(g, face[k]));
                let indices = &mut self.groups[g].prim.indices;
                indices.extend(t.iter().flatten().map(|&k| ids[k as usize]));
                self.triangles += 2;
            } else {
                let mut pts = std::mem::take(&mut self.pts);
                let mut tris = std::mem::take(&mut self.tris);
                let mut ids = std::mem::take(&mut self.ids);
                pts.clear();
                pts.extend(face.iter().map(|c| self.v[c[0] as usize]));
                tris.clear();
                triangulate(&pts, &mut tris);
                ids.clear();
                for &c in face {
                    ids.push(self.vertex(g, c));
                }
                let indices = &mut self.groups[g].prim.indices;
                indices.extend(tris.iter().flatten().map(|&k| ids[k as usize]));
                self.triangles += tris.len();
                self.pts = pts;
                self.tris = tris;
                self.ids = ids;
            }
            self.faces += 1;
            if self.faces == 1 {
                self.face_from = at;
            } else if self.faces == 4096 {
                self.size(g, at, size);
            }
        }
    }

    /// Reserves the arrays once the faces so far say how big they'll get: bytes per face, and
    /// output vertices per position used.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a capacity guess from rates; the limit caps it"
    )]
    fn size(&mut self, g: usize, at: usize, size: usize) {
        let per_face = (at - self.face_from) as f64 / (self.faces - 1) as f64;
        if per_face <= 0.0 {
            return;
        }
        let more_faces = size.saturating_sub(at) as f64 / per_face;
        let tris = (more_faces * self.triangles as f64 / self.faces as f64 * 1.02) as usize + 64;
        let tris = tris.min(self.limit);
        let prim = &mut self.groups[g].prim;
        prim.indices.reserve_exact(tris * 3);
        if self.groups.len() > 1 {
            return;
        }
        // One material so far: its vertices are about every position, times how many output
        // vertices each position has needed (texture seams, hard edges).
        let per_pos = self.key.len() as f64 / self.used.max(1) as f64;
        let total = (self.v.len() as f64 * per_pos * 1.02) as usize + 64;
        let more = total.saturating_sub(self.key.len());
        let grp = &mut self.groups[0];
        let prim = &mut grp.prim;
        prim.positions.reserve_exact(more);
        if !prim.uvs.is_empty() {
            prim.uvs.reserve_exact(more);
        }
        prim.normals.reserve_exact(more);
        if !grp.raw_colors.is_empty() {
            grp.raw_colors.reserve_exact(more);
        }
        self.next.reserve_exact(more);
        self.key.reserve_exact(more);
        self.local.reserve_exact(more);
    }

    /// The group's vertex for a corner, added if it's new. A corner of a flat face always gets a
    /// vertex of its own, which no other corner finds.
    #[inline]
    fn vertex(&mut self, g: usize, c: Corner) -> u32 {
        let [p, t, n] = c;
        let want = [u32_of(g), t, n];
        let head = self.first[p as usize];
        let flat = n == FLAT;
        if !flat {
            let mut id = head;
            while id != NONE {
                if self.key[id as usize] == want {
                    return self.local[id as usize];
                }
                id = self.next[id as usize];
            }
        }
        if head == NONE {
            self.used += 1;
        }
        if flat {
            self.next.push(NONE);
        } else {
            self.first[p as usize] = u32_of(self.key.len());
            self.next.push(head);
        }
        self.key.push(want);
        let grp = &mut self.groups[g];
        let prim = &mut grp.prim;
        let k = prim.positions.len();
        self.local.push(u32_of(k));
        prim.positions.push(self.v[p as usize]);
        if t != NONE {
            if prim.uvs.len() < k {
                prim.uvs.resize(k, [0.0, 1.0]);
            }
            // OBJ puts the texture origin bottom-left; glTF top-left.
            let q = self.vt[t as usize];
            prim.uvs.push([q[0], 1.0 - q[1]]);
        } else if !prim.uvs.is_empty() {
            prim.uvs.push([0.0, 1.0]);
        }
        if n & MADE == 0 {
            prim.normals.push(self.vn[n as usize]);
        } else {
            // Made from the faces once they are all read.
            prim.normals.push([0.0; 3]);
            grp.made += 1;
        }
        if !self.colors.is_empty() {
            // Positions before the first colored one are white.
            grp.raw_colors.resize(k, [1.0; 3]);
            grp.raw_colors
                .push(self.colors.get(p as usize).copied().unwrap_or([1.0; 3]));
        }
        u32_of(k)
    }

    /// Normals for the vertices whose corners have none: the faces around each, weighted by
    /// area, summed over the vertices at its position in its group and smoothing group (so
    /// texture seams don't show); a flat face's vertices take its own.
    fn make_normals(&mut self) {
        if self.groups.iter().all(|g| g.made == 0) {
            return;
        }
        let mut made: Vec<Vec<bool>> = self
            .groups
            .iter()
            .map(|g| {
                if g.made > 0 {
                    vec![false; g.prim.positions.len()]
                } else {
                    Vec::new()
                }
            })
            .collect();
        for (k, &l) in self.key.iter().zip(&self.local) {
            if k[2] & MADE != 0
                && let Some(m) = made.get_mut(k[0] as usize).and_then(|m| m.get_mut(l as usize))
            {
                *m = true;
            }
        }
        for (grp, made) in self.groups.iter_mut().zip(&made) {
            if grp.made == 0 {
                continue;
            }
            let prim = &mut grp.prim;
            for t in prim.indices.chunks_exact(3) {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| i as usize);
                if !(made[a] || made[b] || made[c]) {
                    continue;
                }
                let n = area_normal(prim.positions[a], prim.positions[b], prim.positions[c]);
                for i in [a, b, c] {
                    if made[i] {
                        add(&mut prim.normals[i], n);
                    }
                }
            }
        }
        // One sum for the vertices at a position split by their texture coordinates.
        let mut same: Vec<[u32; 3]> = Vec::new();
        for &head in &self.first {
            if head == NONE || self.next[head as usize] == NONE {
                continue;
            }
            same.clear();
            let mut id = head;
            while id != NONE {
                let k = self.key[id as usize];
                if k[2] & MADE != 0 {
                    same.push([k[0], k[2], self.local[id as usize]]);
                }
                id = self.next[id as usize];
            }
            if same.len() < 2 {
                continue;
            }
            same.sort_unstable();
            for run in same.chunk_by(|a, b| a[..2] == b[..2]) {
                if run.len() < 2 {
                    continue;
                }
                let normals = &mut self.groups[run[0][0] as usize].prim.normals;
                let mut sum = [0.0; 3];
                for v in run {
                    add(&mut sum, normals[v[2] as usize]);
                }
                for v in run {
                    normals[v[2] as usize] = sum;
                }
            }
        }
        for (grp, made) in self.groups.iter_mut().zip(&made) {
            for (n, _) in grp.prim.normals.iter_mut().zip(made).filter(|(_, m)| **m) {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                *n = if len > 0.0 && len.is_finite() {
                    n.map(|c| c / len)
                } else {
                    [0.0, 1.0, 0.0]
                };
            }
        }
    }
}

/// A triangle's normal, as long as twice its area.
fn area_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    vec3::cross(vec3::sub(b, a), vec3::sub(c, a))
}

fn add(to: &mut [f32; 3], n: [f32; 3]) {
    for (t, n) in to.iter_mut().zip(n) {
        *t += n;
    }
}

fn trim_to_len<T>(v: &mut Vec<T>) {
    if v.capacity() > v.len() + v.len() / 16 + 16 {
        v.shrink_to_fit();
    }
}

/// Points or lines on vertices of their own: the positions they use (with their colors), and
/// for lines the pairs as indices into those.
fn loose_prim(b: &mut Builder, at: &[u32], lines: bool, v: &[[f32; 3]], colors: &[[f32; 3]], scale: f32) -> Value {
    let mut local: HashMap<u32, u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut indices = Vec::with_capacity(at.len());
    for &p in at {
        let k = *local.entry(p).or_insert_with(|| {
            positions.push(p);
            u32_of(positions.len() - 1)
        });
        indices.push(k);
    }
    let pos: Vec<[f32; 3]> = positions.iter().map(|&p| v[p as usize]).collect();
    let n = pos.len();
    let mut attributes = json!({ "POSITION": b.positions(pos) });
    if !colors.is_empty() {
        let c: Vec<f32> = positions
            .iter()
            .flat_map(|&p| {
                let c = colors.get(p as usize).copied().unwrap_or([1.0; 3]);
                let [r, g, b] = c.map(|x| srgb_to_linear((x / scale).clamp(0.0, 1.0)));
                [r, g, b, 1.0]
            })
            .collect();
        attributes["COLOR_0"] = json!(b.accessor(Part::f32(c), 5126, n, "VEC4", Some(34962)));
    }
    if lines {
        let count = indices.len();
        let i = b.accessor(Part::u32(indices), 5125, count, "SCALAR", Some(34963));
        json!({ "attributes": attributes, "indices": i, "mode": 1 })
    } else {
        json!({ "attributes": attributes, "mode": 0 })
    }
}

/// OBJ, with MTL colors and textures (base color and normal map). Faces may mix corners with
/// and without texture coordinates or normals. The file is read a block at a time, and each
/// corner becomes a vertex of its part's primitive in its material as it's read.
pub(crate) fn obj(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    // UTF-16, as some Windows tools save text: read as the UTF-8 it says.
    if let Some(text) = utf16(&input.head(2)?)
        .map(|le| input.read().map(|all| String::from_utf16_lossy(&units(&all[2..], le))))
        .transpose()?
    {
        let utf8 = super::Input {
            data: super::Data::Bytes(text.as_bytes(), None),
            ..input
        };
        return obj(utf8, assets, keep);
    }
    let mut blocks = input.blocks()?;
    let size = blocks.size_hint();
    let dir = input.dir();
    let mut r = Reader::new(input.limit);
    // Threads only for models big enough to share out.
    let threads = if size >= 2 * PIECE { keep.thread_count() } else { 1 };
    let mut threaded = None;
    if threads > 1 {
        let (bytes, path) = (input.bytes(), input.path);
        threaded = r.read_threaded(move |tx, used| feed(bytes, path, tx, used), size, threads, PIECE)?;
        if threaded.is_none() {
            // A model the threads can't read as one does: read again on this one.
            r = Reader::new(input.limit);
        }
    }
    if threaded.is_none() {
        // A statement continued over lines ending in a backslash.
        let mut joined: Vec<u8> = Vec::new();
        let mut read = 0;
        while let Some(chunk) = blocks.text(statements)? {
            let at = if read == 0 && chunk.starts_with(BOM) {
                BOM.len()
            } else {
                0
            };
            r.chunk(chunk, at, read, size, &mut joined, true);
            read += chunk.len();
            ensure!(
                r.triangles <= r.limit,
                LimitExceeded,
                "too many triangles to convert (more than {})",
                r.limit
            );
        }
        if !joined.is_empty() {
            r.line(&joined, read, size);
        }
    }
    if r.triangles + r.points + r.lines == 0 && !r.v.is_empty() {
        // Positions and nothing drawn with them: a point cloud, as scanners write it.
        let g = r.group();
        r.groups[g].points.extend(0..u32_of(r.v.len()));
        r.points = r.v.len();
    }
    if r.triangles + r.points + r.lines == 0 {
        // Lines, and none of them a statement OBJ has: some other kind of text.
        ensure!(
            r.skipped.is_empty() || !r.v.is_empty(),
            UnknownFormat,
            "not an OBJ file: none of its lines is an OBJ statement"
        );
        bail!(Empty, "OBJ has no faces or vertices");
    }
    ensure!(
        r.triangles <= r.limit,
        LimitExceeded,
        "too many triangles to convert (more than {})",
        r.limit
    );

    // Vertex colors: 0-1 floats, or 0-255 from some tools; sRGB either way.
    let has_colors = !r.colors.is_empty();
    let scale = if r.colors.iter().flatten().any(|&c| c > 1.0) {
        255.0
    } else {
        1.0
    };
    r.make_normals();
    let mut b = Builder::default();
    // Points and lines first, while the file's positions are here.
    let loose: Vec<Vec<Value>> = r
        .groups
        .iter()
        .map(|g| {
            let mut out = Vec::new();
            if !g.points.is_empty() {
                out.push(loose_prim(&mut b, &g.points, false, &r.v, &r.colors, scale));
            }
            if !g.lines.is_empty() {
                out.push(loose_prim(&mut b, &g.lines, true, &r.v, &r.colors, scale));
            }
            out
        })
        .collect();
    let Reader {
        libs,
        groups,
        materials: names,
        parts,
        triangles,
        points,
        lines,
        comments,
        skipped,
        ..
    } = r;

    // Material libraries and textures that would only be dropped aren't read.
    let textures = keep.materials && keep.textures;
    let mtl = if keep.materials {
        read_libraries(&libs, dir, assets)
    } else {
        HashMap::new()
    };
    // The materials used, in first-use order, and whether any triangles using each have texture
    // coordinates.
    let mut index_of: HashMap<usize, usize> = HashMap::new();
    let mut used: Vec<usize> = Vec::new();
    let mut textured: Vec<bool> = Vec::new();
    for (g, extra) in groups.iter().zip(&loose) {
        if g.prim.indices.is_empty() && extra.is_empty() {
            continue;
        }
        let m = *index_of.entry(g.material).or_insert_with(|| {
            used.push(g.material);
            textured.push(false);
            used.len() - 1
        });
        textured[m] |= !g.prim.uvs.is_empty();
    }
    let mut images = Images::new(!keep.image_bytes, dir);
    // The images packed so far, by what they're packed from.
    let mut packs: HashMap<[Channel; 3], Option<usize>> = HashMap::new();
    let mut out_materials = Vec::new();
    let mut looks: Vec<Look> = Vec::new();
    // Blender writes linear colors, and a roughness and specular level in Phong's fields.
    let linear = super::linear_colors(input, "obj");
    for (m, &name) in used.iter().enumerate() {
        let name = &names[name];
        let entry = mtl.get(name);
        // Faces with no material, or one no library has: the default, under the name they use.
        let mut mat = legacy_material::default_material(has_colors);
        if !name.is_empty() {
            mat.name.clone_from(name);
        }
        // The highlight's `KHR_materials_specular` numbers, when they aren't glTF's own.
        let mut highlight = None;
        if let Some(e) = entry {
            // Without a `Kd`, vertex colors show as they are; otherwise the usual 0.8 gray.
            let fallback = if has_colors { [1.0; 3] } else { [0.8; 3] };
            let mut phong = Phong::new(Rgb::of(e.color.unwrap_or(fallback), linear));
            phong.opacity = e.alpha.unwrap_or(1.0);
            phong.emissive = Rgb::of(e.emissive.unwrap_or([0.0; 3]), linear);
            let pbr = e.roughness.is_some() || e.metallic.is_some();
            // `Ns` is the specular exponent itself, 0 to 1000 (MTL specification, "Ns exponent"). Blender
            // writes its roughness there instead, as Ns = 1000 (1 - roughness)^2: read back as the
            // exponent of that roughness.
            phong.exponent = e.shininess.map(|ns| {
                let ns = ns.clamp(0.0, 1000.0);
                if linear {
                    #[allow(clippy::cast_possible_truncation, reason = "an exponent, far inside f32")]
                    let n = legacy_material::exponent_of_roughness(f64::from(1.0 - (ns / 1000.0).sqrt())) as f32;
                    n
                } else {
                    ns
                }
            });
            if !pbr {
                // `Ks` is the highlight's color. Blender writes its specular level there, where 0.5 is the
                // 4% a dielectric reflects, glTF's whole highlight.
                phong.specular = e.specular.map(|ks| {
                    if linear {
                        Rgb::Linear(ks.map(|c| c * 2.0))
                    } else {
                        Rgb::Srgb(ks)
                    }
                });
                // `illum` 0 and 1 are the models without a highlight (MTL specification, "illum").
                #[allow(clippy::float_cmp, reason = "illum is a whole number")]
                {
                    phong.highlight = !e.illum.is_some_and(|i| i == 0.0 || i == 1.0);
                }
            }
            let out = phong.pbr();
            out.set(&mut mat);
            mat.extensions = None;
            highlight = out.specular.filter(|_| !pbr);
            if pbr {
                mat.metallic = e.metallic.unwrap_or(0.0);
                mat.roughness = e.roughness.unwrap_or(mat.roughness);
            }
        }
        let mut texture = |reference: Option<&MapRef>, make: Make<'_>| -> Option<Tex> {
            if !textures || !textured[m] {
                return None;
            }
            let reference = reference?;
            let found = assets.find(&reference.file, dir)?;
            let image = match make {
                Make::File => images.file(assets, found, &reference.file)?,
                // An image already read shows whether it's gray; one only named is read to see.
                Make::Gray => match images.known(&found) {
                    Some(None) => return None,
                    Some(Some(i)) if matches!(&images.list[i], Source::Image(img) if !gray(&img.bytes)) => {
                        return None;
                    }
                    Some(Some(i)) if matches!(&images.list[i], Source::Image(_)) => i,
                    _ => {
                        let img = assets.load_image(&found).filter(|img| gray(&img.bytes))?;
                        if keep.image_bytes {
                            images.read(found, img)
                        } else {
                            images.file(assets, found, &reference.file)?
                        }
                    }
                },
                Make::Pack(planes) => {
                    let channel = |p: Plane<'_>| match p {
                        Plane::Map(r, pick) => Some(Channel::File(assets.find(&r.file, dir)?, pick)),
                        Plane::Value(v) => Some(Channel::Value(v)),
                    };
                    let key = [channel(planes[0])?, channel(planes[1])?, channel(planes[2])?];
                    (*packs.entry(key).or_insert_with_key(|key| {
                        images.list.push(Source::Image(pack(key, assets)?));
                        Some(images.list.len() - 1)
                    }))?
                }
            };
            Some(reference.tex(image))
        };
        let color = texture(entry.and_then(|e| e.color_map.as_ref()), Make::File);
        let normal = texture(entry.and_then(|e| e.normal_map.as_ref()), Make::File);
        mat.color_image = color.as_ref().map(|t| t.image);
        mat.normal_image = normal.as_ref().map(|t| t.image);
        let mut look = Look {
            color,
            normal,
            ..Look::default()
        };
        if let Some(e) = entry {
            look.emissive = texture(e.emissive_map.as_ref(), Make::File);
            mat.emissive_image = look.emissive.as_ref().map(|t| t.image);
            // The emissive texture shows as it is unless `Ke` scales it (exporters write zero
            // when it doesn't).
            if mat.emissive_image.is_some() && mat.emissive.iter().all(|&c| c <= 0.0) {
                mat.emissive = [1.0; 3];
            }
            // glTF reads roughness and metalness from one image's green and blue, and occlusion
            // from its red: `map_ORM` as it is, `map_RMA` rearranged, and `map_Pr` and `map_Pm` as
            // one (one file named for both is taken as packed that way already).
            let (rough, metal) = (e.roughness_map.as_ref(), e.metallic_map.as_ref());
            // Whether the texture holds roughness and metalness, not one value everywhere in their place.
            let mut holds = [false; 2];
            if let Some(orm) = &e.orm_map {
                look.metal_rough = texture(Some(orm), Make::File);
                look.occlusion.clone_from(&look.metal_rough);
                holds = [true; 2];
            } else if let Some(rma) = &e.rma_map {
                let planes = [
                    Plane::Map(rma, Pick::Blue),
                    Plane::Map(rma, Pick::Red),
                    Plane::Map(rma, Pick::Green),
                ];
                look.metal_rough = texture(Some(rma), Make::Pack(planes));
                look.occlusion.clone_from(&look.metal_rough);
                holds = [true; 2];
            } else if rough.is_some() && rough == metal {
                look.metal_rough = texture(rough, Make::File);
                holds = [true; 2];
            } else if rough.is_some() || metal.is_some() {
                // A gray roughness image reads the same in every channel, so it serves as it is
                // where nothing is metal.
                if let Some(r) = rough
                    && metal.is_none()
                    && r.channel == Pick::Luma
                    && e.metallic.is_none_or(|m| m <= 0.0)
                {
                    look.metal_rough = texture(Some(r), Make::Gray);
                }
                // Otherwise each map in its channel, where they lay their images out alike.
                if look.metal_rough.is_none() && rough.zip(metal).is_none_or(|(r, m)| r.same_layout(m)) {
                    let planes = [Plane::Value(255), Plane::of(rough), Plane::of(metal)];
                    look.metal_rough = texture(rough.or(metal), Make::Pack(planes));
                }
                holds = [rough.is_some(), metal.is_some()];
            }
            if look.metal_rough.is_none() {
                holds = [false; 2];
            }
            if look.occlusion.is_none() {
                look.occlusion = texture(e.occlusion_map.as_ref(), Make::File);
            }
            mat.metallic_roughness_image = look.metal_rough.as_ref().map(|t| t.image);
            mat.occlusion_image = look.occlusion.as_ref().map(|t| t.image);
            // A texture holding a value is that value, unscaled, unless the file gives a factor.
            if holds[0] && e.roughness.is_none() {
                mat.roughness = 1.0;
            }
            if holds[1] && e.metallic.is_none() {
                mat.metallic = 1.0;
            }
            look.metal_in_tex = holds[1];
            // Opacity in the color texture's alpha (`map_d` naming the `map_Kd` file), the long-standing way
            // to write cutout foliage: cut out at a half. A separate opacity file would need its pixels moved
            // into the color texture's alpha.
            if let (Some(a), Some(c)) = (&e.alpha_map, &e.color_map)
                && a.file == c.file
                && mat.color_image.is_some()
            {
                let transparency = Transparency {
                    texture_alpha: true,
                    cutout: Some(0.5),
                    ..Transparency::of(mat.color[3])
                };
                look.alpha = Some(transparency.mode());
                if let Some(Alpha::Mask(c)) = look.alpha {
                    mat.alpha_cutoff = Some(c);
                }
            }
            // Glass lets light through rather than fading out.
            #[allow(clippy::float_cmp, reason = "illum is a whole number")]
            let glass = e.illum.is_some_and(|i| [4.0, 6.0, 7.0, 9.0].contains(&i));
            if glass && let Some(d) = e.alpha.filter(|d| (0.0..1.0).contains(d)) {
                look.transmission = Some(1.0 - d);
                mat.color[3] = 1.0;
            }
            let pbr = e.roughness.is_some() || e.metallic.is_some();
            let spec_tex = texture(e.specular_map.as_ref(), Make::File);
            // No highlight has no use for a texture coloring it.
            let dark = highlight == Some(Highlight::NONE);
            if !pbr && (highlight.is_some() || spec_tex.is_some()) {
                let h = highlight.unwrap_or(Highlight {
                    factor: 1.0,
                    color: [1.0; 3],
                });
                look.specular = Some(Specular {
                    factor: Factor::of(h.factor),
                    color: h.color,
                    color_tex: spec_tex.filter(|_| !dark),
                });
            }
            // Every exporter writes Ni, often a placeholder of 1 (which would take the highlight away).
            look.ior = e.ior.filter(|i| *i > 1.0 && *i <= 5.0);
            let sheen_tex = texture(e.sheen_map.as_ref(), Make::File);
            if e.sheen.is_some() || sheen_tex.is_some() {
                look.sheen = Some(Sheen {
                    color: e.sheen.unwrap_or([1.0; 3]),
                    color_tex: sheen_tex,
                    roughness: Factor::of(0.0),
                });
            }
            if e.clearcoat.is_some() || e.clearcoat_roughness.is_some() {
                look.clearcoat = Some(Clearcoat {
                    factor: Factor::of(e.clearcoat.unwrap_or(1.0)),
                    roughness: Factor::of(e.clearcoat_roughness.unwrap_or(0.0)),
                    normal: None,
                });
            }
            look.anisotropy = e.anisotropy;
            look.anisotropy_rotation = e.anisotropy_rotation.map_or(0.0, |r| r * std::f32::consts::TAU);
            // Color on, ambient off: the color as it is, unlit.
            #[allow(clippy::float_cmp, reason = "illum is a whole number")]
            {
                look.unlit = e.illum == Some(0.0);
            }
        }
        looks.push(look);
        out_materials.push(mat);
    }
    if !linear {
        for s in looks.iter_mut().filter_map(|l| l.sheen.as_mut()) {
            s.color = s.color.map(srgb_to_linear);
        }
    }
    if textures {
        super::guess_texture(&mut out_materials, &mut images, &textured, input.path, assets);
    }

    // A mesh per part, in the order the parts turn up.
    let mut prims_of: Vec<Vec<Value>> = vec![Vec::new(); parts.len()];
    for (g, extra) in groups.into_iter().zip(loose) {
        let Group {
            part,
            material,
            mut prim,
            raw_colors,
            ..
        } = g;
        let Some(&m) = index_of.get(&material) else {
            continue;
        };
        if !prim.indices.is_empty() {
            let nv = prim.positions.len();
            if !prim.uvs.is_empty() {
                prim.uvs.resize(nv, [0.0, 1.0]);
            }
            if has_colors {
                let mut raw = raw_colors;
                raw.resize(nv, [1.0; 3]);
                prim.colors = raw
                    .iter()
                    .map(|c| {
                        let [r, g, b] = c.map(|x| srgb_to_linear((x / scale).clamp(0.0, 1.0)));
                        [r, g, b, 1.0]
                    })
                    .collect();
            }
            trim_to_len(&mut prim.positions);
            trim_to_len(&mut prim.uvs);
            trim_to_len(&mut prim.indices);
            if prim.normals.len() == nv {
                trim_to_len(&mut prim.normals);
            } else {
                prim.normals = smooth_normals(&prim.positions, &prim.indices);
            }
            if let Some(mut p) = b.primitive(prim) {
                p["material"] = json!(m);
                prims_of[part].push(p);
            }
        }
        for mut p in extra {
            p["material"] = json!(m);
            prims_of[part].push(p);
        }
    }
    let (mut nodes, mut meshes) = (Vec::new(), Vec::new());
    for ((object, group), primitives) in parts.into_iter().zip(prims_of) {
        if primitives.is_empty() {
            continue;
        }
        let mut mesh = json!({ "primitives": primitives });
        let mut node = json!({ "mesh": meshes.len() });
        // Named for its group, or failing that its object; a group inside an object keeps that
        // object's name in extras.
        if let Some(name) = group.as_ref().or(object.as_ref()) {
            mesh["name"] = json!(name);
            node["name"] = json!(name);
        }
        if let (Some(o), Some(_)) = (&object, &group) {
            node["extras"] = json!({ "object": o });
        }
        meshes.push(mesh);
        nodes.push(node);
    }

    let mut source = json!({});
    if !libs.is_empty() {
        source["mtllib"] = json!(libs);
    }
    let info = SourceInfo {
        format: Some("OBJ".into()),
        up_axis: Fact::Assumed(Axis::PosY),
        tool: comments.iter().find_map(|c| exporter(c)),
        comments,
        ..SourceInfo::default()
    };
    if let Value::Object(m) = &mut source {
        info.write(m);
    }
    if linear {
        source["linearColors"] = json!(true);
    }
    if has_colors {
        source["vertexColorScale"] = json!(scale);
    }
    if !skipped.is_empty() {
        let m: Map<String, Value> = skipped.iter().map(|(k, n)| (text(k), json!(n))).collect();
        source["skipped"] = Value::Object(m);
    }
    let mut facts = Map::new();
    if triangles > 0 {
        facts.insert("triangles".into(), json!(triangles));
    }
    if points > 0 {
        facts.insert("points".into(), json!(points));
    }
    if lines > 0 {
        facts.insert("lines".into(), json!(lines));
    }
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert(
        "scenes".into(),
        json!([{ "nodes": (0..nodes.len()).collect::<Vec<_>>() }]),
    );
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    let fulls = out_materials
        .iter()
        .enumerate()
        .map(|(i, m)| looks.get(i).map_or_else(|| Full::from(m), |l| l.full(m)));
    b.materials(fulls, images.list, &mut json);
    // Every MTL statement, in its material's extras.
    if let Some(Value::Array(list)) = json.get_mut("materials") {
        for (mat, &name) in list.iter_mut().zip(&used) {
            if let Some(e) = mtl.get(&names[name]).filter(|e| !e.raw.is_empty()) {
                mat["extras"] = json!({ "mtl": e.raw });
            }
        }
    }
    Ok(b.finish(json, facts))
}

/// The exporter an opening comment names: `Blender 4.2.0` (from `# Blender 4.2.0` or
/// `# Blender v2.93.1 OBJ File: ''`), or what follows `Exported from`, `Exported by` or `Generated by`.
fn exporter(comment: &str) -> Option<String> {
    let c = comment.trim();
    if let Some(rest) = c.strip_prefix("Blender ")
        && rest.starts_with(|ch: char| ch.is_ascii_digit() || ch == 'v')
    {
        let end = c.find(" OBJ File").or_else(|| c.find(" MTL File")).unwrap_or(c.len());
        return Some(c[..end].trim().to_owned());
    }
    let lower = c.to_ascii_lowercase();
    ["exported from ", "exported by ", "generated by "]
        .iter()
        .find_map(|p| {
            lower
                .starts_with(p)
                .then(|| c[p.len()..].trim())
                .filter(|t| !t.is_empty())
                .map(str::to_owned)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use crate::mesh::Mesh;
    use std::fmt::Write as _;

    #[test]
    fn positions_alone_are_a_point_cloud() {
        let doc = read_in(b"# scan\nv 0 0 0\nv 1 0 0\nv 0 1 0 1 0 0\n", &Memory::new());
        let m = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(m.triangles(), 0);
        assert_eq!(doc.facts["points"], 3);
        let err = obj(
            Input::memory(b"# nothing\nvn 0 0 1\n", "t.obj"),
            &Memory::new(),
            Keep::ALL,
        );
        assert!(err.err().unwrap().to_string().contains("no faces"));
    }

    #[test]
    fn opening_comments_and_the_exporter_are_the_source_info() {
        let text = b"# Blender v2.93.1 OBJ File: ''
# www.blender.org
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
# later
";
        let info = read_in(text, &Memory::new()).source_info();
        assert_eq!(info.format.as_deref(), Some("OBJ"));
        assert_eq!(info.tool.as_deref(), Some("Blender v2.93.1"));
        assert_eq!(info.comments, ["Blender v2.93.1 OBJ File: ''", "www.blender.org"]);
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
        assert_eq!(info.converted, crate::source::Converted::default());
        assert_eq!(exporter("Exported from Wings 3D 2.2"), Some("Wings 3D 2.2".into()));
        assert_eq!(exporter("a cube"), None);
        let info = read_in(
            b"v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
",
            &Memory::new(),
        )
        .source_info();
        assert!(info.tool.is_none() && info.comments.is_empty());
    }

    #[test]
    fn utf16_with_a_byte_order_mark_reads_as_utf8() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        for le in [true, false] {
            let mut bytes = if le { vec![0xFF, 0xFE] } else { vec![0xFE, 0xFF] };
            for u in text.encode_utf16() {
                bytes.extend(if le { u.to_le_bytes() } else { u.to_be_bytes() });
            }
            let m = read_in(&bytes, &Memory::new())
                .flatten(crate::document::Pose::Rest)
                .unwrap();
            assert_eq!(m.unwrap().triangles(), 1, "{le}");
        }
    }

    fn read_in(bytes: &[u8], assets: &Memory) -> Document {
        obj(Input::memory(bytes, "t.obj"), assets, Keep::ALL).unwrap()
    }

    fn mesh(text: &str) -> Mesh {
        let doc = read_in(text.as_bytes(), &Memory::new());
        doc.flatten(crate::document::Pose::Rest).unwrap().unwrap()
    }

    /// Each vertex's normal, by its position.
    fn normals_at(m: &Mesh, at: [f32; 3]) -> Vec<[f32; 3]> {
        let p = &m.prims[0];
        p.positions
            .iter()
            .zip(&p.normals)
            .filter(|(q, _)| near(**q, at))
            .map(|(_, n)| *n)
            .collect()
    }

    fn near(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    fn png() -> Vec<u8> {
        let mut png = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb([200, 100, 50]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    /// Everything a reader has read, as bits; but not where the first face is (for sizing the
    /// arrays), and with vertex colors as the model will have them, white where none was kept
    /// for a vertex made before any position had one.
    fn state(r: &Reader) -> Vec<u32> {
        let mut out = Vec::new();
        let mut put = |tag: u32, xs: &mut dyn Iterator<Item = u32>| {
            out.push(tag);
            out.extend(xs);
        };
        put(1, &mut r.v.iter().flatten().map(|x| x.to_bits()));
        put(2, &mut r.vt.iter().flatten().map(|x| x.to_bits()));
        put(3, &mut r.vn.iter().flatten().map(|x| x.to_bits()));
        put(4, &mut r.colors.iter().flatten().map(|x| x.to_bits()));
        put(5, &mut r.key.iter().flatten().copied().chain(r.local.iter().copied()));
        put(
            6,
            &mut [r.triangles, r.points, r.lines, r.faces, r.mrgb]
                .map(u32_of)
                .into_iter(),
        );
        put(7, &mut r.skipped.iter().map(|k| u32_of(k.1)));
        put(8, &mut r.comments.iter().map(|c| u32_of(c.len())));
        for g in &r.groups {
            let p = &g.prim;
            put(9, &mut [g.part, g.material, g.made].map(u32_of).into_iter());
            put(10, &mut p.positions.iter().flatten().map(|x| x.to_bits()));
            put(11, &mut p.normals.iter().flatten().map(|x| x.to_bits()));
            put(12, &mut p.uvs.iter().flatten().map(|x| x.to_bits()));
            put(13, &mut p.indices.iter().copied());
            let mut colors = g.raw_colors.clone();
            if !r.colors.is_empty() {
                colors.resize(p.positions.len(), [1.0; 3]);
            }
            put(14, &mut colors.iter().flatten().map(|x| x.to_bits()));
            put(15, &mut g.points.iter().chain(&g.lines).copied());
        }
        out
    }

    /// `text` read on `threads` threads, in blocks of about `block` bytes (read from a stream
    /// or not) and pieces of about `piece`; and whether it had to be read again on one.
    fn read_threaded<'a>(text: &'a [u8], threads: usize, piece: usize, block: usize, stream: bool) -> (Vec<u32>, bool) {
        let mut r = Reader::new(usize::MAX);
        let feed = |tx: &mpsc::SyncSender<Text<'a>>, used: &Spare<Vec<u8>>| {
            if stream {
                read_blocks(text, block, tx, used);
            } else {
                slice_blocks(text, block, tx);
            }
        };
        match r.read_threaded(feed, text.len(), threads, piece).unwrap() {
            Some(read) => {
                assert_eq!(read, text.len());
                r.make_normals();
                (state(&r), false)
            }
            None => (read_both(text).0, true),
        }
    }

    /// `text` read on threads in many ways, each read as one thread reads it; and, unless
    /// `again`, without reading it again on one thread (which two threads never do).
    fn threads_read_as_one(text: &str, again: bool) {
        let one = read_both(text.as_bytes()).0;
        for threads in [2, 3, 8] {
            for piece in [0, 1, 2, 5, 13, 64, 500, 4096] {
                for (block, stream) in [(1, true), (40, false), (333, true), (5000, false), (1 << 20, true)] {
                    let (many, restarted) = read_threaded(text.as_bytes(), threads, piece, block, stream);
                    assert!(
                        many == one,
                        "{threads} threads in pieces of {piece} bytes, blocks of {block}, differ"
                    );
                    assert_eq!(
                        restarted,
                        again && threads > 2,
                        "{threads} threads in pieces of {piece} bytes"
                    );
                }
            }
        }
    }

    /// `text` read with and without the one-pass line reader.
    fn read_both(text: &[u8]) -> (Vec<u32>, Vec<u32>) {
        let read = |quick: bool| {
            let mut r = Reader::new(usize::MAX);
            let mut joined = Vec::new();
            r.chunk(text, 0, 0, text.len(), &mut joined, quick);
            if !joined.is_empty() {
                r.line(&joined, text.len(), text.len());
            }
            r.make_normals();
            state(&r)
        };
        (read(true), read(false))
    }

    /// Lines of every kind the one-pass reader meets, plain or not.
    const LINES: &[&str] = &[
        "v 1 2 3",
        "v -0 -0.0 +0",
        "v 1e5 -1.5E-3 2.5e+2",
        "v -2.19910035e-10 1.23456789 -9.87654321e+05",
        "v 3.4028235e38 1e39 -1e-46",
        "v 1e-40 7e-11 123456789012345678901234",
        "v 16777217 0.30000000000000004 1.00000005960464477539062500",
        "v inf -inf nan",
        "v 1 2 nan",
        "v 1 2",
        "v 1 2 3 1",
        "v 1 2 3 0.5 0.25 0.125",
        "v 1 2 3 0.5 0.25 0.125 1",
        "v 1 2 3 4 5 6 7 8 9",
        "v 1 2 3 x",
        "v 1 2 3 # comment",
        "v 1 2 3#",
        "v\t4\t5\t6\t",
        "v 4 5 6   ",
        "v  7  8  9",
        "  v 1 1 1",
        "v 1 2 3\x0c",
        "v 1 2 3 \\",
        "v 1 2 \\",
        "v 1 2 3\\",
        "v 1 2 3 4 5 6 7 8 \\",
        "v 1 2 3 4 5 6 7 8 # end \\",
        "vt 0.5 0.25",
        "vt 0.5 0.25 0",
        "vt 0.5",
        "vt",
        "vt 0.5 x",
        "vt 0.1 0.2 0.3 \\",
        "vn 0 0 2",
        "vn 0 0 0",
        "vn 1 1",
        "vn 1 1 1 1",
        "vn 1e-30 1e-30 1e-30",
        "f 1 2 3",
        "f 1/1 2/2 3/3",
        "f 1//1 2//1 3//1",
        "f 1/1/1 2/2/1 3/3/1 4/4/1",
        "f -1 -2 -3",
        "f -1/-1/-1 -2/-2/-1 -3/-3/-1",
        "f 1/ 2/ 3/",
        "f 1// 2// 3//",
        "f 1/1/ 2/2/ 3/3/",
        "f 1/1/1/1 2/2/2 3/3/3",
        "f 0 1 2",
        "f -0 1 2",
        "f +1 2 3",
        "f 1 2 999999",
        "f 1 2 3 # face",
        "f 1 2 3#",
        "f 1 2",
        "f 1 2 3 4 5 6",
        "f 1 x 3",
        "f 1/x 2 3",
        "f 1 2 \\",
        "f 1 2 3 \\",
        "f 1 2 3\\",
        "f 1 2 0000000000000000003",
        "f 1 2 000000000000000003",
        "f\t1\t2\t3",
        "fo 1 2 3",
        "s off",
        "s 2",
        "usemtl other",
        "g part",
        "# a comment",
        "# ends \\",
        "l 1 2",
        "p 1",
        "",
        "   ",
    ];

    /// Every line of [`LINES`] with every ending, then `n` more picked at random; positions
    /// first so faces have something to name.
    fn mixed_lines(seed: u64, n: usize) -> String {
        let mut rng = crate::test_rng::Pcg32::new(seed);
        let ends = ["\n", "\r\n", "\r"];
        let mut text = String::new();
        for _ in 0..4 {
            text.push_str("v 0 0 0\nvt 0 0\nvn 0 0 1\n");
        }
        for line in LINES {
            for end in ends {
                text.push_str(line);
                text.push_str(end);
            }
        }
        for _ in 0..n {
            text.push_str(LINES[rng.below(u32_of(LINES.len())) as usize]);
            text.push_str(ends[rng.below(3) as usize]);
        }
        text
    }

    #[test]
    fn plain_lines_read_in_one_pass_like_any_other() {
        let text = mixed_lines(3, 3000);
        let (quick, slow) = read_both(text.as_bytes());
        assert!(quick == slow, "the one-pass reader differs");
        // And ending without a line break, in each kind of line.
        for line in LINES {
            let t = format!("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n{line}");
            let (quick, slow) = read_both(t.as_bytes());
            assert!(quick == slow, "differs at the end: {line:?}");
        }
    }

    #[test]
    fn threads_read_what_one_thread_reads() {
        let mut text = mixed_lines(5, 4000);
        // More statements the workers leave to the reader, and faces with corners of each kind.
        for i in 0..300 {
            text.push_str(match i % 8 {
                0 => "o thing\r\n",
                1 => "g part two\nf 1 2 3\ng\n",
                2 => "s 1\rf -1/-1 -2/-2 -3/-3 -4/-4\r",
                3 => "usemtl a\r\nf 1/1/1 2/2/1 3/3/1 4/4/1 5/1/1\r\n",
                4 => "f 4 3 2 1\nl 1 2 \\\r\n 3\n",
                5 => "v 0 1 0\nvt 1 1\nvn 1 0 0\nf -1/-1/-1 -2/-2/-2 -3/-3/-3\nf -4//-1 -3//-2 -2//-3 -1//-1\n",
                6 => "s off\nf 1/1 2/2 3/3 4/4\nf 2 3 4 # flat\ns 3\n",
                _ => "v 1 2 3 0.5 0.5 0.5\nf -1 -2 -3 \\\n -4\n",
            });
        }
        // Ending in a statement continued to the end of the file.
        text.push_str("f 1 2 \\");
        threads_read_as_one(&text, false);
    }

    #[test]
    fn threads_read_faces_before_their_positions_and_far_apart_as_one_does() {
        // Faces naming positions not yet there are dropped, then kept once they are.
        let mut text = String::from("usemtl early\nf 1 2 3\nf 1/1 2/1 3/1 4/1\nf -1 -2 -3\np 1\n");
        for i in 0..6000 {
            _ = writeln!(text, "v {i} {} 0", i % 7);
        }
        text.push_str("vt 0 0\nusemtl late\n");
        // Corners thousands of positions apart, in pieces of few faces; and repeated, to share.
        for i in 0..400 {
            let (a, b, c) = (1 + i, 5999 - i, 3000 + i % 50);
            _ = write!(text, "f {a} {b} {c}\nf {a}/1 {c}/1 {b}/1 {}/1\nf {c} {b} {a}\n", a + 1);
            if i % 97 == 0 {
                text.push_str("s off\nf 1 6000 3000 2\ns 1\n");
            }
        }
        // And a face continued to the end of the file, after the last piece's.
        text.push_str("f 1 2 3 \\");
        threads_read_as_one(&text, false);
    }

    #[test]
    fn threads_read_again_what_they_cant_share() {
        // Colors from `#MRGB` before any face are as good as from `v`; after one, the vertices
        // made so far would need the colors they had then.
        let base = mixed_lines(13, 1000);
        threads_read_as_one(&format!("v 1 2 3\n#MRGB ff102030ff405060\n{base}"), false);
        threads_read_as_one(&format!("{base}#MRGB ff102030ff405060\nf 1 2 3\n"), true);
    }

    #[test]
    fn threaded_loads_keep_to_the_triangle_limit() {
        let text = mixed_lines(17, 4000).repeat(40);
        assert!(text.len() > 2 * PIECE, "too small for threads");
        let load = |limit, threads| {
            let input = Input {
                limit,
                ..Input::memory(text.as_bytes(), "t.obj")
            };
            obj(input, &Memory::new(), Keep::ALL.with_threads(threads)).map(|d| d.to_glb().unwrap())
        };
        let one = load(usize::MAX, 1).unwrap();
        let triangles = mesh(&text).triangles();
        for threads in [2, 4] {
            assert!(
                load(triangles, threads).unwrap() == one,
                "a load on {threads} threads at the limit differs"
            );
            let err = load(triangles - 1, threads).unwrap_err();
            assert!(format!("{err:#}").contains("too many triangles"), "{err:#}");
        }
    }

    #[test]
    fn threaded_loads_load_what_one_thread_loads() {
        // Big enough to be read on threads, in memory and from a file; with a byte order mark.
        let text = format!("\u{feff}{}", mixed_lines(9, 4000).repeat(40));
        assert!(text.len() > 2 * PIECE, "too small for threads");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.obj");
        std::fs::write(&path, &text).unwrap();
        let keep = |threads| Keep::ALL.with_threads(threads);
        let glb = |input, keep| obj(input, &Memory::new(), keep).unwrap().to_glb().unwrap();
        let one = glb(Input::memory(text.as_bytes(), "t.obj"), keep(1));
        assert!(
            glb(Input::memory(text.as_bytes(), "t.obj"), keep(4)) == one,
            "a threaded load differs"
        );
        #[cfg(feature = "fs")]
        assert!(glb(Input::file(&path), keep(0)) == one, "a threaded file load differs");
    }

    #[test]
    fn loads_where_threads_cant_start_load_on_one() {
        // As on a platform without threads: the model is read on the calling thread instead.
        let text = mixed_lines(9, 4000).repeat(40);
        assert!(text.len() > 2 * PIECE, "too small for threads");
        let glb = |threads| {
            let keep = Keep::ALL.with_threads(threads);
            obj(Input::memory(text.as_bytes(), "t.obj"), &Memory::new(), keep)
                .unwrap()
                .to_glb()
                .unwrap()
        };
        let one = glb(1);
        crate::keep::NO_THREADS.set(true);
        let (two, four) = (glb(2), glb(4));
        let sum = std::sync::atomic::AtomicUsize::new(0);
        each(
            (1..=100).collect(),
            4,
            || (),
            |(), i| {
                sum.fetch_add(i, Ordering::Relaxed);
            },
        );
        crate::keep::NO_THREADS.set(false);
        assert!(two == one && four == one, "a load without threads differs");
        assert_eq!(sum.into_inner(), 5050, "work left undone without threads");
    }

    #[test]
    fn blocks_end_where_statements_end() {
        let text = mixed_lines(11, 2000) + "f 1 2 \\";
        let d = text.as_bytes();
        for block in [1, 2, 7, 64, 1000, 1 << 20] {
            for stream in [false, true] {
                let (tx, rx) = mpsc::sync_channel(1 << 16);
                if stream {
                    read_blocks(d, block, &tx, &Mutex::new(Vec::new()));
                } else {
                    slice_blocks(d, block, &tx);
                }
                drop(tx);
                let blocks: Vec<_> = rx.iter().map(Result::unwrap).collect();
                assert!(blocks.concat() == d, "blocks of {block} lose text");
                let mut c = 0;
                for b in &blocks[..blocks.len() - 1] {
                    c += b.len();
                    assert_cut(d, c);
                }
            }
        }
    }

    #[test]
    fn pieces_end_where_statements_end() {
        let text = mixed_lines(7, 2000);
        let d = text.as_bytes();
        for piece in [0, 1, 2, 7, 33, 1000] {
            let cuts = pieces(d, 0, piece);
            assert_eq!((cuts[0], cuts[cuts.len() - 1]), (0, d.len()));
            for w in cuts.windows(2) {
                assert!(w[0] < w[1], "pieces in order");
            }
            for &c in &cuts[1..cuts.len() - 1] {
                assert_cut(d, c);
            }
        }
    }

    /// `d` cut at `c`: after a line break, not inside a `\r\n`, and not continuing a statement.
    fn assert_cut(d: &[u8], c: usize) {
        assert!(matches!(d[c - 1], b'\n' | b'\r'), "cut mid-line at {c}");
        assert!(d[c - 1..=c] != *b"\r\n", "cut inside a line break at {c}");
        let kept = d[..c - 1]
            .iter()
            .rposition(|&b| !matches!(b, b' ' | b'\t' | b'\r' | b'\x0c'));
        assert!(
            kept.is_none_or(|k| d[k] != b'\\'),
            "a continued statement is cut at {c}"
        );
    }

    #[test]
    fn lines_may_end_in_a_lone_carriage_return() {
        let m = mesh("v 0 0 0\rv 1 0 0\r\nv 0 1 0\rf 1 2 \\\r3\r");
        assert_eq!(m.triangles(), 1);
        // A `\r` last in a block may be half of a `\r\n`.
        assert_eq!(statements(b"v 1\r\nv 2\r"), Some(5));
        assert_eq!(statements(b"v 1\rv 2\r\n"), Some(9));
        let mut lines = Vec::new();
        for_each_line(b"newmtl a\rKd 1 0 0\r\nd 1\n", |l| lines.push(l.to_vec()));
        assert_eq!(lines, [&b"newmtl a"[..], b"Kd 1 0 0", b"d 1"]);
    }

    #[test]
    fn a_byte_order_mark_is_skipped_in_obj_and_mtl() {
        let mut text = BOM.to_vec();
        text.extend_from_slice(b"mtllib t.mtl\nv 9 9 9\nv 1 0 0\nv 0 1 0\nv 0 0 1\nusemtl red\nf 2 3 4\n");
        let mut mtl = BOM.to_vec();
        mtl.extend_from_slice(b"newmtl red\nKd 1 0 0\n");
        let doc = read_in(&text, &Memory::new().with("t.mtl", mtl));
        let m = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        let p = &m.prims[0].positions;
        assert!(p.contains(&[0.0, 0.0, 1.0]) && !p.contains(&[9.0, 9.0, 9.0]), "{p:?}");
        assert_eq!(doc.json()["materials"][0]["name"], "red");
        assert_eq!(m.materials[0].color, [1.0, 0.0, 0.0, 1.0]);
    }

    /// Two triangles folded along the edge from the origin to (1, 0, 0).
    const FOLD: &str = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n";

    #[test]
    fn smoothing_groups_decide_which_faces_share_made_normals() {
        // No `s`: smooth, so the fold's edge takes the faces' average.
        let m = mesh(&format!("{FOLD}f 1 2 3\nf 1 4 2\n"));
        assert_eq!(m.prims[0].positions.len(), 4);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert!(near(normals_at(&m, [0.0; 3])[0], [0.0, h, h]));
        // Flat, or two groups: each face its own.
        for text in [
            format!("{FOLD}s off\nf 1 2 3\nf 1 4 2\n"),
            format!("{FOLD}s 1\nf 1 2 3\ns 2\nf 1 4 2\n"),
        ] {
            let m = mesh(&text);
            assert_eq!(m.prims[0].positions.len(), 6, "{text}");
            let mut at = normals_at(&m, [0.0; 3]);
            at.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(near(at[0], [0.0, 0.0, 1.0]) && near(at[1], [0.0, 1.0, 0.0]), "{at:?}");
        }
        // A flat quad's corners are one vertex each, shared by its two triangles.
        let m = mesh("s 0\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n");
        assert_eq!((m.prims[0].positions.len(), m.triangles()), (4, 2));
    }

    #[test]
    fn corners_with_normals_keep_them_beside_corners_without() {
        let m = mesh(&format!("{FOLD}vn 0 0 -1\nf 1//1 2//1 3//1\nf 1 4 2\n"));
        let mut at = normals_at(&m, [0.0; 3]);
        at.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(at, [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]);
        assert_eq!(normals_at(&m, [0.0, 1.0, 0.0]), [[0.0, 0.0, -1.0]]);
    }

    #[test]
    fn made_normals_have_no_seam_where_texture_coordinates_split_a_vertex() {
        let m = mesh(&format!(
            "{FOLD}vt 0 0\nvt 1 0\nvt 0 1\nvt 0.5 0.5\nf 1/1 2/2 3/3\nf 1/4 4/1 2/3\n"
        ));
        assert_eq!(m.prims[0].positions.len(), 6);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        for at in [[0.0; 3], [1.0, 0.0, 0.0]] {
            let n = normals_at(&m, at);
            assert_eq!(n.len(), 2);
            assert!(n.iter().all(|&n| near(n, [0.0, h, h])), "{n:?}");
        }
    }

    #[test]
    fn mrgb_comments_color_the_positions_in_order() {
        let m = mesh("v 0 0 0\nv 1 0 0\nv 0 1 0\n#MRGB ff0000ffff00ff00ffff0000\nf 1 2 3\n");
        let p = &m.prims[0];
        let color = |at: [f32; 3]| {
            let c = p.colors[p.positions.iter().position(|q| near(*q, at)).unwrap()];
            [c[0], c[1], c[2]]
        };
        assert!(near(color([0.0; 3]), [0.0, 0.0, 1.0]));
        assert!(near(color([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        assert!(near(color([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0]));
        assert!(p.colors.iter().all(|c| (c[3] - 1.0).abs() < 1e-6));
    }

    #[test]
    fn mtl_opacity_emission_packed_maps_glass_and_dissolve_over_transparency() {
        let mtl = "newmtl leaf\nKd 1 1 1\nd 1\nTr 1\nmap_Kd c.png\nmap_d c.png\nmap_Ke e.png\nmap_ORM orm.png\n\
                   newmtl glass\nillum 4\nTr 0.9\nd 0.25\n";
        let assets = Memory::new()
            .with("t.mtl", mtl)
            .with("c.png", png())
            .with("e.png", png())
            .with("orm.png", png());
        let text = "mtllib t.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n\
                    usemtl leaf\nf 1/1 2/2 3/3\nusemtl glass\nf 1/1 3/3 2/2\n";
        let doc = read_in(text.as_bytes(), &assets);
        // `map_d` naming the color texture: its alpha is the opacity, cut out at a half.
        let leaf = &doc.json()["materials"][0];
        assert_eq!(leaf["alphaMode"], "MASK", "{leaf}");
        assert!(leaf["emissiveTexture"].is_object());
        assert_eq!(leaf["emissiveFactor"], json!([1.0, 1.0, 1.0]));
        assert!(leaf["pbrMetallicRoughness"]["metallicRoughnessTexture"].is_object());
        assert!(leaf["occlusionTexture"].is_object());
        let alpha = &leaf["pbrMetallicRoughness"]["baseColorFactor"][3];
        assert!(alpha.is_null() || alpha == 1.0, "d wins over Tr: {leaf}");
        let glass = &doc.json()["materials"][1];
        assert_eq!(
            glass["extensions"]["KHR_materials_transmission"]["transmissionFactor"],
            0.75
        );
        assert!(glass["alphaMode"].is_null(), "{glass}");
    }

    // Packing needs the PNG encoder.
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
    #[test]
    fn mtl_packs_rma_and_separate_roughness_and_metalness_maps() {
        let solid = |rgb: [u8; 3], size: u32| {
            let mut png = Vec::new();
            image::RgbImage::from_pixel(size, size, image::Rgb(rgb))
                .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                .unwrap();
            png
        };
        let mtl = "newmtl rma\nmap_RMA rma.png\n\
                   newmtl split\nPm 0.5\nmap_Pr r.png\nmap_Pm -imfchan g m.png\n\
                   newmtl unalike\nmap_Pr r.png\nmap_Pm -s 2 2 1 m.png\n\
                   newmtl sizes\nmap_Pr r.png\nmap_Pm big.png\n\
                   newmtl orm\nmap_ORM rma.png\n\
                   newmtl turned\naniso 0.8\nanisor 0.25\n";
        let assets = Memory::new()
            .with("t.mtl", mtl)
            .with("rma.png", solid([200, 100, 50], 2))
            .with("r.png", solid([100, 150, 200], 2))
            .with("m.png", solid([10, 220, 30], 2))
            .with("big.png", solid([10, 220, 30], 4));
        let mut text = String::from("mtllib t.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n");
        for name in ["rma", "split", "unalike", "sizes", "orm", "turned"] {
            writeln!(text, "usemtl {name}\nf 1/1 2/2 3/3").unwrap();
        }
        let doc = |keep: Keep| obj(Input::memory(text.as_bytes(), "t.obj"), &assets, keep).unwrap();
        let m = doc(Keep::ALL).flatten(crate::document::Pose::Rest).unwrap().unwrap();
        let by = |name: &str| m.materials.iter().find(|x| x.name == name).unwrap();
        let pixel = |i: Option<usize>| {
            let img = image::load_from_memory(&m.images[i.unwrap()].bytes).unwrap().to_rgb8();
            img.get_pixel(0, 0).0
        };
        // Roughness, metalness and occlusion from the red, green and blue moved to green, blue and red.
        let rma = by("rma");
        assert_eq!(pixel(rma.metallic_roughness_image), [50, 200, 100]);
        assert_eq!(rma.occlusion_image, rma.metallic_roughness_image);
        assert_eq!((rma.roughness, rma.metallic), (1.0, 1.0));
        // Roughness as the luma of its map, metalness from the green its `-imfchan` names, scaled by `Pm`.
        let split = by("split");
        assert_eq!(pixel(split.metallic_roughness_image), [255, 143, 220]);
        assert_eq!((split.roughness, split.metallic), (1.0, 0.5));
        // Maps laid out apart, or of different sizes: no texture, the factors as they are.
        for name in ["unalike", "sizes"] {
            let x = by(name);
            assert_eq!(x.metallic_roughness_image, None, "{name}");
            assert_eq!(x.metallic, 0.0, "{name}");
        }
        assert_eq!(by("orm").roughness, 1.0);
        let turned = by("turned").extensions.as_ref().unwrap();
        let rotation = turned.anisotropy.as_ref().unwrap().rotation;
        assert!((rotation - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        // Without image bytes the maps are read all the same to pack them: the same materials, the packed image
        // embedded and the file used as it is left unread.
        let full = doc(Keep::ALL);
        let lazy = doc(Keep::ALL.with_image_bytes(false));
        assert_eq!(lazy.json()["materials"], full.json()["materials"]);
        let uris: Vec<_> = (0..3).map(|i| lazy.json()["images"][i]["uri"].as_str()).collect();
        assert_eq!(uris, [None, None, Some("rma.png")]);
    }
}
