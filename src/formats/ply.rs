//! PLY (the Stanford polygon file format), ASCII and binary in either byte order.
//!
//! Read: positions, normals (`nx` or `normal_x`), vertex colors (sRGB, as integers or floats: an
//! 8-bit channel is out of 255, a 16-bit one out of 65535, and a 32-bit whole number out of 255),
//! texture coordinates under any of their usual names, polygons (triangulated in their plane) and
//! triangle strips; `MeshLab`'s per-face texture coordinates and `comment TextureFile` (several, a
//! face picking one by its `texnumber`, a material each); per-face colors. Elements the reader
//! doesn't use are skipped. A point cloud with 3D gaussian splatting's properties (`f_dc_0` to
//! `f_dc_2`, `opacity`, `scale_0` to `scale_2`, `rot_0` to `rot_3`, and `f_rest_*` up to degree
//! 3) is read as `KHR_gaussian_splatting`: opacity through the sigmoid, scale out of its
//! logarithm, rotation normalized, the harmonics in whole degrees, and the axes (right, down,
//! forward) flipped to glTF's; [`Keep::max_points`] caps the splats. Splat-like files with only
//! some of those properties show their color: the zero-order spherical harmonics and `opacity`
//! give the vertex colors when the file has no other. `SuperSplat`'s compressed PLY (a `chunk`
//! element and packed vertices) is read as the same splats: see [`compressed`]. A zero normal
//! (splat files write nothing else) is a placeholder: a mesh's vertex takes a made one, and a
//! file of nothing but zeros has none. Made normals are smooth across vertices split only by
//! texture coordinates. Header lines may end in a lone `\r`, and a leading byte order mark is
//! skipped. Counts the header gives are checked against the bytes left before anything is
//! allocated for them.
//!
//! Nothing the vertices carry is lost: properties beyond those are custom attributes, named as
//! the file names them in capitals after an underscore (`_CONFIDENCE`, `_F_DC_0`). A file without
//! faces is a point cloud, drawn as points; an `edge` element's `vertex1` and `vertex2` are
//! lines. A point cloud's colors are normalized 16-bit linear RGBA, as the other point clouds'
//! are: that keeps every value of an 8-bit channel, made linear, apart, in half the bytes of floats.
//!
//! The header's comments are the source's comments, and one naming the program that wrote the
//! file (`Created in Blender version 4.2`, `VCGLIB generated`) its tool; `obj_info` lines, the
//! encoding and the elements skipped go in the asset's `extras.source` too. PLY names no units and
//! no up axis, so both are unknown, and the geometry is kept as written. Gaussian splats are the
//! exception: their convention is -Y up and right-handed (assumed in the source info), and the
//! flip is recorded as a converted up axis, as the other splat readers record theirs.

use std::path::Path;

use serde_json::{Map, Value, json};

use super::Input;
use super::stream::{self, Blocks};
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep, Part};
use crate::gaussian::{self, REST, Splats, degree_of, rotation, sigmoid};
use crate::images::Images;
use crate::mesh::{FastMap, Material, Mesh, Prim, quad, smooth_normals, srgb_to_linear, triangulate};
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::vec3;
use crate::{Context as _, ErrorKind, Result};

mod compressed;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Ascii,
    Le,
    Be,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &[u8]) -> Result<Ty> {
        Ok(match s {
            b"char" | b"int8" => Ty::I8,
            b"uchar" | b"uint8" => Ty::U8,
            b"short" | b"int16" => Ty::I16,
            b"ushort" | b"uint16" => Ty::U16,
            b"int" | b"int32" => Ty::I32,
            b"uint" | b"uint32" => Ty::U32,
            b"float" | b"float32" => Ty::F32,
            b"double" | b"float64" => Ty::F64,
            t => bail!(Corrupt, "unknown PLY type {}", crate::shown(t)),
        })
    }

    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }

    /// The value that means full intensity for a color channel of this type: the most an unsigned 8-bit or
    /// 16-bit channel holds (a signed one read the same), and 1 for a float. A 32-bit whole number is 0 to 255
    /// too, as the writers that give colors that type write them; more than 255 is full intensity.
    fn color_scale(self) -> f64 {
        match self {
            Ty::U8 | Ty::I8 | Ty::I32 | Ty::U32 => 255.0,
            Ty::U16 | Ty::I16 => 65535.0,
            Ty::F32 | Ty::F64 => 1.0,
        }
    }
}

#[derive(Debug, Clone)]
struct Prop {
    name: String,
    ty: Ty,
    /// For a list: the type of its count.
    count: Option<Ty>,
}

#[derive(Debug, Clone)]
struct Element {
    name: String,
    count: usize,
    props: Vec<Prop>,
}

impl Element {
    fn slot(&self, names: &[&str]) -> Option<usize> {
        names
            .iter()
            .find_map(|n| self.props.iter().position(|p| p.name == *n && p.count.is_none()))
    }

    fn list(&self, names: &[&str]) -> Option<usize> {
        names
            .iter()
            .find_map(|n| self.props.iter().position(|p| p.name == *n && p.count.is_some()))
    }

    /// The fewest bytes one item can take: its scalars' sizes and its lists' counts (binary),
    /// or a character and a separator per property (ASCII).
    fn min_size(&self, format: Format) -> usize {
        self.props
            .iter()
            .map(|p| match (format, p.count) {
                (Format::Ascii, _) => 2,
                (_, Some(c)) => c.size(),
                (_, None) => p.ty.size(),
            })
            .sum()
    }
}

/// Why a record couldn't be read from a stretch of data.
enum Stop {
    /// The stretch ends first: read more of the file and try the record again.
    Short,
    Bad(&'static str),
}

type Got<T> = std::result::Result<T, Stop>;

/// Reads values in whichever encoding the file uses, from a stretch of its data that holds
/// whole tokens (ASCII) but maybe not whole records.
struct Cursor<'a> {
    format: Format,
    d: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    #[inline]
    fn token(&mut self) -> Got<&[u8]> {
        let d = self.d;
        let mut i = self.pos;
        while i < d.len() && stream::space(d[i]) {
            i += 1;
        }
        let start = i;
        while i < d.len() && !stream::space(d[i]) {
            i += 1;
        }
        if i == start {
            return Err(Stop::Short);
        }
        self.pos = i;
        Ok(&d[start..i])
    }

    #[inline]
    fn bytes<const N: usize>(&mut self) -> Got<[u8; N]> {
        let b = self.d.get(self.pos..self.pos + N).ok_or(Stop::Short)?;
        self.pos += N;
        let mut a = [0u8; N];
        a.copy_from_slice(b);
        if self.format == Format::Be {
            a.reverse();
        }
        Ok(a)
    }

    #[inline]
    fn value(&mut self, ty: Ty) -> Got<f64> {
        if self.format == Format::Ascii {
            let t = self.token()?;
            let v = match ty {
                // Rounded once, straight to the type the file declares.
                Ty::F32 => stream::f32_from(t).map(f64::from),
                _ => stream::f64_from(t),
            };
            return v.ok_or(Stop::Bad("bad PLY number"));
        }
        Ok(match ty {
            Ty::I8 => f64::from(self.bytes::<1>()?[0].cast_signed()),
            Ty::U8 => f64::from(self.bytes::<1>()?[0]),
            Ty::I16 => f64::from(i16::from_le_bytes(self.bytes()?)),
            Ty::U16 => f64::from(u16::from_le_bytes(self.bytes()?)),
            Ty::I32 => f64::from(i32::from_le_bytes(self.bytes()?)),
            Ty::U32 => f64::from(u32::from_le_bytes(self.bytes()?)),
            Ty::F32 => f64::from(f32::from_le_bytes(self.bytes()?)),
            Ty::F64 => f64::from_le_bytes(self.bytes()?),
        })
    }

    /// A vertex index; `u32::MAX` (never valid) for a negative or NaN one.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "`as` saturates, and the vertex check refuses what's past the end"
    )]
    #[inline]
    fn index(&mut self, ty: Ty) -> Got<u32> {
        let neg = |v: i64| u32::try_from(v).unwrap_or(u32::MAX);
        match (self.format, ty) {
            (Format::Ascii, _) => {
                let t = self.token()?;
                if t.len() < 10 && t.iter().all(u8::is_ascii_digit) {
                    return Ok(t.iter().fold(0, |n, &c| n * 10 + u32::from(c - b'0')));
                }
                let v = stream::f64_from(t).ok_or(Stop::Bad("bad PLY number"))?;
                Ok(if v >= 0.0 { v as u32 } else { u32::MAX })
            }
            (_, Ty::I32) => Ok(neg(i64::from(i32::from_le_bytes(self.bytes()?)))),
            (_, Ty::U32) => Ok(u32::from_le_bytes(self.bytes()?)),
            _ => {
                let v = self.value(ty)?;
                Ok(if v >= 0.0 { v as u32 } else { u32::MAX })
            }
        }
    }

    /// Steps over a value.
    #[inline]
    fn skip(&mut self, ty: Ty, n: usize) -> Got<()> {
        if self.format == Format::Ascii {
            for _ in 0..n {
                self.token()?;
            }
            return Ok(());
        }
        let end = self.pos + ty.size() * n;
        if end > self.d.len() {
            return Err(Stop::Short);
        }
        self.pos = end;
        Ok(())
    }

    /// A list's length, or none for a list an ASCII line leaves out: the item's values so far
    /// share a line, and it ends before the list (as writers that declare a list they don't
    /// write leave it).
    #[inline]
    fn list_len(&mut self, ty: Ty, item: usize) -> Got<usize> {
        if self.format == Format::Ascii {
            let line = |c: &u8| matches!(c, b'\n' | b'\r');
            let so_far = &self.d[item..self.pos];
            if let Some(first) = so_far.iter().position(|&c| !stream::space(c))
                && !so_far[first..].iter().any(line)
                && self.d[self.pos..].iter().take_while(|&&c| stream::space(c)).any(line)
            {
                return Ok(0);
            }
        }
        self.len(ty)
    }

    /// A list's length.
    #[inline]
    fn len(&mut self, ty: Ty) -> Got<usize> {
        let v = match (self.format, ty) {
            (Format::Le | Format::Be, Ty::U8) => return Ok(usize::from(self.bytes::<1>()?[0])),
            _ => self.value(ty)?,
        };
        if !(0.0..=1e6).contains(&v) {
            return Err(Stop::Bad("bad PLY list length"));
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "checked to be in 0..=1e6"
        )]
        Ok(v as usize)
    }
}

/// A binary value at the start of `b`, which holds at least its size.
#[inline]
fn at(b: &[u8], ty: Ty, be: bool) -> f64 {
    match ty {
        Ty::I8 => f64::from(b[0].cast_signed()),
        Ty::U8 => f64::from(b[0]),
        Ty::I16 => f64::from(i16::from_le_bytes(ordered(b, be))),
        Ty::U16 => f64::from(u16::from_le_bytes(ordered(b, be))),
        Ty::I32 => f64::from(i32::from_le_bytes(ordered(b, be))),
        Ty::U32 => f64::from(u32::from_le_bytes(ordered(b, be))),
        Ty::F32 => f64::from(f32::from_le_bytes(ordered(b, be))),
        Ty::F64 => f64::from_le_bytes(ordered(b, be)),
    }
}

/// The first `N` bytes of `b` (which holds them) in little-endian order.
#[inline]
fn ordered<const N: usize>(b: &[u8], be: bool) -> [u8; N] {
    let mut a = [0u8; N];
    a.copy_from_slice(&b[..N]);
    if be {
        a.reverse();
    }
    a
}

/// What the header says besides its elements.
struct Header {
    format: Format,
    version: String,
    /// The texture files the `TextureFile` comments name, in order: a face's `texnumber`
    /// picks one.
    textures: Vec<String>,
    comments: Vec<String>,
    obj_info: Vec<String>,
}

/// The UTF-8 byte order mark some editors put first.
const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// The header, its elements, and where the data starts.
fn header(data: &[u8]) -> Result<(Header, Vec<Element>, usize)> {
    let mut format = None;
    let mut version = String::new();
    let mut elements: Vec<Element> = Vec::new();
    let mut textures = Vec::new();
    let (mut comments, mut obj_info) = (Vec::new(), Vec::new());
    let text = |b: &[u8]| crate::bytes::text(b).into_owned();
    let mut at = 0;
    let mut first = true;
    loop {
        let rest = data
            .get(at..)
            .filter(|r| !r.is_empty())
            .context(ErrorKind::Truncated, "PLY header is cut short: it has no end_header")?;
        let len = rest
            .iter()
            .position(|&c| c == b'\n' || c == b'\r')
            .unwrap_or(rest.len());
        let line = rest[..len].trim_ascii();
        // A line ends at `\n`, `\r\n` or a lone `\r`.
        at += len
            + match rest.get(len..len + 2) {
                Some(b"\r\n") => 2,
                Some(_) => 1,
                None => rest.len() - len,
            };
        if first {
            let line = line.strip_prefix(BOM).unwrap_or(line);
            ensure!(line == b"ply", UnknownFormat, "not a PLY file");
            first = false;
            continue;
        }
        let w: Vec<&[u8]> = line.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty()).collect();
        match w.as_slice() {
            [b"end_header", ..] => break,
            [b"format", f, rest @ ..] => {
                version = rest.first().map(|v| text(v)).unwrap_or_default();
                format = Some(match *f {
                    b"ascii" => Format::Ascii,
                    b"binary_little_endian" => Format::Le,
                    b"binary_big_endian" => Format::Be,
                    f => bail!(Corrupt, "unknown PLY format {}", crate::shown(f)),
                });
            }
            [b"comment", kind, ..] if kind.eq_ignore_ascii_case(b"texturefile") => {
                let name = line[b"comment".len()..].trim_ascii_start()[kind.len()..].trim_ascii();
                textures.push(text(name));
                comments.push(text(line[b"comment".len()..].trim_ascii()));
            }
            [b"comment", ..] => comments.push(text(line[b"comment".len()..].trim_ascii())),
            [b"obj_info", ..] => obj_info.push(text(line[b"obj_info".len()..].trim_ascii())),
            [b"element", name, count] => elements.push(Element {
                name: crate::bytes::text(name).into_owned(),
                count: std::str::from_utf8(count)
                    .ok()
                    .and_then(|c| c.parse().ok())
                    .context(ErrorKind::Corrupt, "bad PLY element count")?,
                props: Vec::new(),
            }),
            [b"property", b"list", c, i, name] => elements
                .last_mut()
                .context(ErrorKind::Corrupt, "PLY property before element")?
                .props
                .push(Prop {
                    name: crate::bytes::text(name).into_owned(),
                    ty: Ty::parse(i)?,
                    count: Some(Ty::parse(c)?),
                }),
            [b"property", ty, name] => elements
                .last_mut()
                .context(ErrorKind::Corrupt, "PLY property before element")?
                .props
                .push(Prop {
                    name: crate::bytes::text(name).into_owned(),
                    ty: Ty::parse(ty)?,
                    count: None,
                }),
            _ => {}
        }
    }
    let header = Header {
        format: format.context(ErrorKind::Corrupt, "PLY header names no format")?,
        version,
        textures,
        comments,
        obj_info,
    };
    Ok((header, elements, at))
}

/// Vertex data, by vertex. Normals are unit length and texture coordinates flipped to glTF's
/// convention as they're read.
#[derive(Default)]
struct Vertices {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    col: Vec<[f32; 4]>,
    uv: Vec<[f32; 2]>,
    /// The other scalar properties, a column each, in the order of `Plan::extra`.
    extra: Vec<Vec<f32>>,
}

/// Faces kept as the file gives them, for when they can't become triangles as they're read:
/// they come before the vertices, or give their corners their own texture coordinates or
/// colors. Each face's corners (vertex indices), and per corner the texture coordinate
/// (flipped) and color.
#[derive(Default)]
struct Faces {
    starts: Vec<usize>,
    corners: Vec<u32>,
    corner_uv: Vec<[f32; 2]>,
    corner_col: Vec<[f32; 4]>,
    /// Each face's texture number, when the faces give one.
    texture: Vec<u32>,
}

const X: &[&str] = &["x"];
const Y: &[&str] = &["y"];
const Z: &[&str] = &["z"];
const U: &[&str] = &["u", "s", "texture_u", "texture_s", "tx"];
const V: &[&str] = &["v", "t", "texture_v", "texture_t", "ty"];
const R: &[&str] = &["red", "diffuse_red", "r"];
const G: &[&str] = &["green", "diffuse_green", "g"];
const B: &[&str] = &["blue", "diffuse_blue", "b"];
const A: &[&str] = &["alpha", "diffuse_alpha", "a"];
const NX: &[&str] = &["nx", "normal_x", "normalx", "n_x"];
const NY: &[&str] = &["ny", "normal_y", "normaly", "n_y"];
const NZ: &[&str] = &["nz", "normal_z", "normalz", "n_z"];
/// Gaussian splats' color: the first spherical-harmonic band's coefficients, and opacity
/// before its sigmoid.
const DC: [&[&str]; 3] = [&["f_dc_0"], &["f_dc_1"], &["f_dc_2"]];
const OPACITY: &[&str] = &["opacity"];
/// The zeroth spherical-harmonic basis function's constant, `1 / (2 sqrt(pi))`.
const SH_C0: f64 = 0.282_094_791_773_878_14;
/// The properties that make a point cloud's vertices gaussian splats, as 3D gaussian splatting
/// training writes them: the base color, opacity (a logit), scale (a logarithm) and rotation
/// (w, x, y, z). The higher degrees' coefficients are `f_rest_0` on, a color channel at a time.
const GAUSSIAN: [&str; 11] = [
    "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
];

/// A color from its channels' values. `bytes` holds each byte value's channel (made linear) and
/// alpha, for binary files' `uchar` channels; it's empty otherwise.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a binary uchar's value is a whole number from 0 to 255"
)]
fn color(el: &Element, vals: &[f64], slots: &[usize; 3], alpha: Option<usize>, bytes: &[[f32; 2]]) -> [f32; 4] {
    let byte = |s: usize| {
        (el.props[s].ty == Ty::U8)
            .then(|| bytes.get(vals[s] as usize))
            .flatten()
    };
    let ch = |s: usize| match byte(s) {
        Some(b) => b[0],
        None => srgb_to_linear(f32_of((vals[s] / el.props[s].ty.color_scale()).clamp(0.0, 1.0))),
    };
    let a = alpha.map_or(1.0, |s| match byte(s) {
        Some(b) => b[1],
        None => f32_of((vals[s] / el.props[s].ty.color_scale()).clamp(0.0, 1.0)),
    });
    [ch(slots[0]), ch(slots[1]), ch(slots[2]), a]
}

/// For each byte value, the color channel it gives (made linear) and the alpha, as [`color`]
/// works them out.
fn byte_colors() -> Vec<[f32; 2]> {
    (0..=255u8)
        .map(|v| {
            let c = f32_of((f64::from(v) / 255.0).clamp(0.0, 1.0));
            [srgb_to_linear(c), c]
        })
        .collect()
}

/// A gaussian splat's color: its base spherical-harmonic band as sRGB, made linear, and its
/// opacity through a sigmoid.
fn splat_color(vals: &[f64], dc: [usize; 3], opacity: Option<usize>) -> [f32; 4] {
    let ch = |s: usize| srgb_to_linear(f32_of((0.5 + SH_C0 * vals[s]).clamp(0.0, 1.0)));
    let a = opacity.map_or(1.0, |s| f32_of(1.0 / (1.0 + (-vals[s]).exp())));
    [ch(dc[0]), ch(dc[1]), ch(dc[2]), if a.is_finite() { a } else { 1.0 }]
}

/// Whether `n` is the zero vector, the placeholder for a vertex without a normal.
#[allow(clippy::float_cmp, reason = "the placeholder is written exactly")]
fn is_zero(n: [f32; 3]) -> bool {
    n == [0.0; 3]
}

/// Where an element's properties are, and which of them the reader uses.
struct Plan {
    xyz: [usize; 3],
    n: Option<[usize; 3]>,
    rgb: Option<[usize; 3]>,
    alpha: Option<usize>,
    /// A gaussian splat's color, for vertices without one of their own; the properties stay
    /// custom attributes too.
    splat: Option<([usize; 3], Option<usize>)>,
    uv: Option<(usize, usize)>,
    index_list: Option<usize>,
    uv_list: Option<usize>,
    /// A face's texture, by its place among the `TextureFile` comments.
    texnumber: Option<usize>,
    /// An edge's two vertices.
    edge: Option<[usize; 2]>,
    /// A vertex's scalar properties beyond those above, kept as custom attributes.
    extra: Vec<usize>,
    /// The scalar properties read; the rest are stepped over.
    wanted: Vec<bool>,
    /// Binary records of fixed size (no lists): the size and each property's offset.
    fixed: Option<(usize, Vec<usize>)>,
    /// With `fixed`, the wanted properties: each one's place, offset and type.
    reads: Vec<(usize, usize, Ty)>,
    /// An element of nothing but its vertex index list, counted by a `uchar`, of `int` or
    /// `uint`: the list's type.
    lone_list: Option<Ty>,
    /// The colors byte values give, for binary files' `uchar` colors; see [`color`].
    bytes: Vec<[f32; 2]>,
}

impl Plan {
    fn new(el: &Element, kind: Kind) -> Plan {
        let tri = |a: Option<usize>, b: Option<usize>, c: Option<usize>| Some([a?, b?, c?]);
        let xyz = [el.slot(X), el.slot(Y), el.slot(Z)];
        let mut plan = Plan {
            xyz: xyz.map(|s| s.unwrap_or(usize::MAX)),
            n: tri(el.slot(NX), el.slot(NY), el.slot(NZ)),
            rgb: tri(el.slot(R), el.slot(G), el.slot(B)),
            alpha: el.slot(A),
            splat: None,
            uv: el.slot(U).zip(el.slot(V)),
            index_list: el.list(&["vertex_indices", "vertex_index"]),
            uv_list: el.list(&["texcoord"]),
            texnumber: (kind == Kind::Face).then(|| el.slot(&["texnumber"])).flatten(),
            edge: (kind == Kind::Edge)
                .then(|| Some([el.slot(&["vertex1"])?, el.slot(&["vertex2"])?]))
                .flatten(),
            extra: Vec::new(),
            wanted: vec![false; el.props.len()],
            fixed: None,
            reads: Vec::new(),
            lone_list: None,
            bytes: Vec::new(),
        };
        let used: Vec<usize> = xyz
            .into_iter()
            .flatten()
            .chain(plan.n.into_iter().flatten())
            .chain(plan.rgb.into_iter().flatten())
            .chain(plan.alpha)
            .chain(plan.uv.into_iter().flat_map(|(u, v)| [u, v]))
            .chain(plan.edge.into_iter().flatten())
            .chain(plan.texnumber)
            .collect();
        for &s in &used {
            plan.wanted[s] = true;
        }
        if kind == Kind::Vertex {
            plan.extra = (0..el.props.len())
                .filter(|&k| !used.contains(&k) && el.props[k].count.is_none())
                .collect();
            for &k in &plan.extra {
                plan.wanted[k] = true;
            }
            if plan.rgb.is_none() {
                plan.splat = tri(el.slot(DC[0]), el.slot(DC[1]), el.slot(DC[2])).map(|dc| (dc, el.slot(OPACITY)));
            }
        }
        if el.props.iter().all(|p| p.count.is_none()) {
            let mut offsets = Vec::with_capacity(el.props.len());
            let mut size = 0;
            for p in &el.props {
                offsets.push(size);
                size += p.ty.size();
            }
            plan.reads = (0..el.props.len())
                .filter(|&k| plan.wanted[k])
                .map(|k| (k, offsets[k], el.props[k].ty))
                .collect();
            plan.fixed = Some((size, offsets));
        }
        if let ([p], Some(0)) = (&el.props[..], plan.index_list)
            && p.count == Some(Ty::U8)
            && matches!(p.ty, Ty::I32 | Ty::U32)
        {
            plan.lone_list = Some(p.ty);
        }
        plan
    }
}

/// Everything read from the elements.
#[derive(Default)]
#[allow(clippy::struct_excessive_bools, reason = "independent facts about what was read")]
struct Body {
    verts: Vertices,
    /// Whether the vertices have been read (faces after them become triangles at once).
    have_verts: bool,
    /// Triangles, as vertex indices.
    tris: Vec<u32>,
    /// Faces kept for later.
    faces: Faces,
    /// Triangle strips' triangles, checked against the vertex count at the end.
    strip_tris: Vec<[u32; 3]>,
    /// Edges, as vertex indices, checked against the vertex count at the end.
    edges: Vec<u32>,
    points: usize,
    /// Vertices whose normal is zero (or broken): placeholders, filled in later.
    zero_normals: usize,
    /// The most triangles the file may have.
    limit: usize,
    /// Whether the file has no faces, so its vertices are a point cloud, and gaussian splats
    /// when they have the properties.
    points_only: bool,
    /// The most splats the file may have.
    max_splats: usize,
    /// Whether the vertices are gaussian splats.
    splats: bool,
    /// The vertex element's other scalar properties' names, for `Vertices::extra`.
    extra_names: Vec<String>,
    /// Elements read past.
    skipped: Vec<String>,
    /// Whether the data ends before the header's counts do.
    cut: bool,
    // Scratch, reused from record to record.
    vals: Vec<f64>,
    idx: Vec<u32>,
    tc: Vec<f32>,
    pts: Vec<[f32; 3]>,
    local: Vec<[u32; 3]>,
}

pub(crate) fn ply(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let path = input.path;
    let mut blocks = input.blocks()?;
    // The header, whole.
    let mut want = 4096;
    let (head, mut elements, start) = loop {
        blocks.need(want)?;
        let d = blocks.data();
        let ended = d
            .windows(11)
            .position(|w| matches!(w[0], b'\n' | b'\r') && &w[1..] == b"end_header")
            .is_some_and(|i| d[i + 11..].iter().any(|&c| c == b'\n' || c == b'\r'));
        if ended || blocks.at_end() {
            break header(d)?;
        }
        want *= 2;
    };
    if compressed::is_compressed(&elements) {
        // Packed splats are unpacked from the whole file, not a block at a time.
        drop(blocks);
        let data = input.read()?;
        return compressed::load(data.get(start..).unwrap_or(&[]), &head, &elements, keep.points());
    }
    blocks.consume(start);
    let room = blocks.size_hint().saturating_sub(start);
    let has_faces = elements
        .iter()
        .any(|e| (e.name == "face" || e.name == "tristrips") && e.count > 0);
    let mut body = Body {
        limit: input.limit,
        points_only: !has_faces,
        max_splats: keep.points(),
        ..Body::default()
    };
    // Every item takes at least its smallest record, so counts are cut to what the file can
    // hold before anything is reserved for them (the last ASCII value needs no separator).
    let mut least = 0usize;
    for el in &mut elements {
        let size = el.min_size(head.format);
        let fits = room
            .saturating_add(1)
            .saturating_sub(least)
            .checked_div(size)
            .unwrap_or(room)
            .min(room);
        if el.count > fits {
            el.count = fits;
            body.cut = true;
        }
        least = least.saturating_add(el.count.saturating_mul(size));
    }
    for el in &elements {
        read_element(&mut blocks, head.format, el, &mut body)?;
    }
    // A file cut short keeps the items it holds whole, if it holds any vertices.
    ensure!(
        !body.cut || !body.verts.pos.is_empty(),
        Truncated,
        "PLY data is cut short: the file holds fewer items than its header counts"
    );
    let mut source = json!({ "encoding": if matches!(head.format, Format::Ascii) { "ascii" } else { "binary" } });
    match head.format {
        Format::Ascii => {}
        Format::Le => source["byteOrder"] = json!("little"),
        Format::Be => source["byteOrder"] = json!("big"),
    }
    if body.cut {
        source["truncated"] = json!(true);
    }
    for (key, list) in [("objInfo", &head.obj_info), ("skipped", &body.skipped)] {
        if !list.is_empty() {
            source[key] = json!(list);
        }
    }
    let info = SourceInfo {
        format: Some("PLY".into()),
        version: Some(head.version.clone()).filter(|v| !v.is_empty()),
        tool: head.comments.iter().find_map(|c| writer(c)),
        comments: head.comments.clone(),
        ..SourceInfo::default()
    };
    if let Value::Object(m) = &mut source {
        info.write(m);
    }
    let mut facts = Map::new();
    let mut b = Builder::default();
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    let names = std::mem::take(&mut body.extra_names);
    let nv = body.verts.pos.len();
    let edges: Vec<u32> = body
        .edges
        .as_chunks::<2>()
        .0
        .iter()
        .filter(|e| e.iter().all(|&i| (i as usize) < nv))
        .flatten()
        .copied()
        .collect();
    let mut primitives = Vec::new();
    // The attributes the edges share: the triangles' when their vertices are the file's.
    let mut shared = None;
    if has_faces {
        // A texture that would only be dropped isn't read.
        let textures = if keep.materials && keep.textures {
            head.textures.as_slice()
        } else {
            &[]
        };
        let dir = path.parent().unwrap_or(Path::new(""));
        let mut images = Images::new(!keep.image_bytes, dir);
        let cut = body.cut;
        let (mesh, srcs) = build(&mut body, textures, path, assets, &mut images).map_err(|e| {
            if cut && e.kind() == ErrorKind::Empty {
                crate::Error::cut_short("PLY data is cut short before its first whole face")
            } else {
                e
            }
        })?;
        facts.insert("triangles".into(), json!(mesh.triangles()));
        facts.insert(
            "vertices".into(),
            json!(mesh.prims.iter().map(|p| p.positions.len()).sum::<usize>()),
        );
        let Mesh { prims, materials, .. } = mesh;
        for (k, p) in prims.into_iter().enumerate() {
            let n = p.positions.len();
            let material = p.material;
            let src = srcs.as_ref().and_then(|s| s.get(k));
            if let Some(mut prim) = b.primitive(p) {
                let columns = gather(&mut body.verts.extra, src.map(Vec::as_slice), n);
                custom(&mut b, &mut prim["attributes"], &names, columns);
                prim["material"] = json!(material);
                if src.is_none() {
                    shared = Some(prim["attributes"].clone());
                }
                primitives.push(prim);
            }
        }
        b.materials(&materials, images.list, &mut json);
    } else {
        facts.insert("points".into(), json!(body.points));
        let splats = if body.splats {
            splats(&mut body.verts, &names)?
        } else {
            None
        };
        if let Some(mut splats) = splats {
            let v = &mut body.verts;
            // Splats dropped for a broken position take their other properties with them.
            if let Some(kept) = splats.sanitize() {
                let n = kept.len();
                let fits = |len: usize| len == n;
                if fits(v.nrm.len()) {
                    gaussian::retain(&mut v.nrm, &kept);
                }
                if fits(v.col.len()) {
                    gaussian::retain(&mut v.col, &kept);
                }
                if fits(v.uv.len()) {
                    gaussian::retain(&mut v.uv, &kept);
                }
                for col in v.extra.iter_mut().filter(|c| fits(c.len())) {
                    gaussian::retain(col, &kept);
                }
            }
            let n = splats.len();
            if let Some(mut prim) = splats.primitive(&mut b) {
                // Right, down and forward by convention, which the file doesn't state: turned to glTF's.
                let mut turned = SourceInfo {
                    up_axis: Fact::Assumed(Axis::NegY),
                    handedness: Fact::Assumed(Handedness::Right),
                    ..SourceInfo::default()
                };
                turned.converted.up_axis = true;
                if let Some(Value::Object(source)) = json.get_mut("asset").and_then(|a| a.pointer_mut("/extras/source"))
                {
                    turned.write(source);
                }
                let has_normals = v.nrm.iter().any(|&c| !is_zero(c));
                others(&mut b, &mut prim["attributes"], v, n, has_normals, &names, "COLOR_1");
                shared = Some(json!({ "POSITION": prim["attributes"]["POSITION"] }));
                primitives.push(prim);
                gaussian::use_extension(&mut json);
            }
            facts.insert("points".into(), json!(n));
            facts.insert("splats".into(), json!(n));
        } else if nv > 0 {
            let attributes = vertex_attributes(&mut b, &mut body.verts, body.zero_normals, &names);
            primitives.push(json!({ "attributes": attributes, "mode": 0 }));
            shared = Some(attributes);
        }
    }
    if !edges.is_empty() {
        let attributes = match shared {
            Some(a) => a,
            None => json!({ "POSITION": b.positions(std::mem::take(&mut body.verts.pos)) }),
        };
        let count = edges.len();
        let indices = b.accessor(Part::u32(edges), 5125, count, "SCALAR", Some(34963));
        facts.insert("lines".into(), json!(count / 2));
        primitives.push(json!({ "attributes": attributes, "indices": indices, "mode": 1 }));
    }
    if primitives.is_empty() {
        return Ok(Document::empty(facts));
    }
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": [0] }]));
    json.insert("nodes".into(), json!([{ "mesh": 0 }]));
    json.insert("meshes".into(), json!([{ "primitives": primitives }]));
    Ok(b.finish(json, facts))
}

/// The extra columns for `n` drawn vertices: the file's own when `src` is `None`, otherwise
/// gathered by source vertex.
fn gather(extra: &mut [Vec<f32>], src: Option<&[u32]>, n: usize) -> Vec<Vec<f32>> {
    extra
        .iter_mut()
        .map(|col| match src {
            None if col.len() == n => std::mem::take(col),
            None => Vec::new(),
            Some(src) => src
                .iter()
                .map(|&v| col.get(v as usize).copied().unwrap_or(0.0))
                .collect(),
        })
        .collect()
}

/// Adds each column as a float custom attribute, `_` and the property's name in capitals.
fn custom(b: &mut Builder, attributes: &mut Value, names: &[String], columns: Vec<Vec<f32>>) {
    for (name, col) in names.iter().zip(columns) {
        if col.is_empty() {
            continue;
        }
        let key = format!("_{}", name.to_ascii_uppercase());
        if attributes.get(&key).is_some() {
            continue;
        }
        let n = col.len();
        attributes[key] = json!(b.accessor(Part::f32(col), 5126, n, "SCALAR", Some(34962)));
    }
}

/// A point cloud's attributes: everything its vertices carry.
fn vertex_attributes(b: &mut Builder, v: &mut Vertices, zero_normals: usize, names: &[String]) -> Value {
    let n = v.pos.len();
    let mut attributes = json!({ "POSITION": b.positions(std::mem::take(&mut v.pos)) });
    others(b, &mut attributes, v, n, zero_normals < n, names, "COLOR_0");
    attributes
}

/// Adds what `n` points carry beyond their positions to `attributes`: normals (unless
/// `has_normals` says they're all placeholders), texture coordinates, colors (as `color`) and
/// the other properties.
fn others(
    b: &mut Builder,
    attributes: &mut Value,
    v: &mut Vertices,
    n: usize,
    has_normals: bool,
    names: &[String],
    color: &str,
) {
    // Normals that are all zero are placeholders; a point without one faces up.
    if v.nrm.len() == n && has_normals {
        for c in &mut v.nrm {
            if is_zero(*c) {
                *c = [0.0, 1.0, 0.0];
            }
        }
        let a = b.accessor(
            Part::f32(std::mem::take(&mut v.nrm).into_flattened()),
            5126,
            n,
            "VEC3",
            Some(34962),
        );
        attributes["NORMAL"] = json!(a);
    }
    if v.uv.len() == n {
        let a = b.accessor(
            Part::f32(std::mem::take(&mut v.uv).into_flattened()),
            5126,
            n,
            "VEC2",
            Some(34962),
        );
        attributes["TEXCOORD_0"] = json!(a);
    }
    if v.col.len() == n {
        // Normalized 16-bit, as the other point clouds' colors are: an 8-bit channel's values, made linear, stay
        // apart (the closest two, 0 and 1, are some 20 steps apart), in half the bytes of floats.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 0..=1, the product is in 0..=65535"
        )]
        let colors: Vec<u16> = std::mem::take(&mut v.col)
            .into_flattened()
            .into_iter()
            .map(|c| (c.clamp(0.0, 1.0) * 65535.0).round() as u16)
            .collect();
        let a = b.accessor(Part::u16(colors), 5123, n, "VEC4", Some(34962));
        b.accessors[a].normalized = true;
        attributes[color] = json!(a);
    }
    let columns = gather(&mut v.extra, None, n);
    custom(b, attributes, names, columns);
}

/// The gaussian splats a point cloud's vertices are (see [`GAUSSIAN`]), their properties taken
/// from `v`, in glTF's axes: the file's are right, down and forward. `None` when they aren't
/// splats. Coefficients past degree 3, or of an incomplete degree, stay custom attributes.
fn splats(v: &mut Vertices, names: &[String]) -> Result<Option<Splats>> {
    let find = |name: &str| names.iter().position(|n| n == name);
    let n = v.pos.len();
    let Some(slots) = GAUSSIAN.iter().map(|&p| find(p)).collect::<Option<Vec<usize>>>() else {
        return Ok(None);
    };
    if slots.iter().any(|&k| v.extra[k].len() != n) {
        return Ok(None);
    }
    let rest: Vec<usize> = (0..).map_while(|k| find(&format!("f_rest_{k}"))).collect();
    let per = if rest.len().is_multiple_of(3) {
        rest.len() / 3
    } else {
        0
    };
    let used = if rest.iter().all(|&k| v.extra[k].len() == n) {
        REST[degree_of(per)]
    } else {
        0
    };
    let mut s = Splats::with_capacity(n, used)?;
    let cols: Vec<Vec<f32>> = slots.iter().map(|&k| std::mem::take(&mut v.extra[k])).collect();
    let [r, g, b, o, s0, s1, s2, w, x, y, z] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(|k| cols[k].as_slice());
    for i in 0..n {
        s.dc.push([r[i], g[i], b[i]]);
        s.opacities.push(sigmoid(o[i]));
        s.scales.push([s0[i].exp(), s1[i].exp(), s2[i].exp()]);
        s.rotations.push(rotation(x[i], y[i], z[i], w[i]));
    }
    drop(cols);
    for (j, out) in s.sh.iter_mut().enumerate() {
        let [r, g, b] = [0, 1, 2].map(|c| std::mem::take(&mut v.extra[rest[c * per + j]]));
        out.extend(r.iter().zip(&g).zip(&b).map(|((&r, &g), &b)| [r, g, b]));
    }
    s.positions.extend_from_slice(&v.pos);
    v.pos = Vec::new();
    s.flip([1.0, -1.0, -1.0]);
    Ok(Some(s))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Vertex,
    Face,
    Strip,
    Edge,
    Other,
}

/// Room for `n` more items, or an error where there isn't memory for them.
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<()> {
    v.try_reserve_exact(n)
        .ok()
        .context(ErrorKind::LimitExceeded, "not enough memory for the PLY data")
}

/// Reads one element's items, a block of the file at a time.
fn read_element(blocks: &mut Blocks<'_>, format: Format, el: &Element, body: &mut Body) -> Result<()> {
    let kind = match el.name.as_str() {
        "vertex" => Kind::Vertex,
        "face" => Kind::Face,
        "tristrips" => Kind::Strip,
        "edge" => Kind::Edge,
        _ => Kind::Other,
    };
    let mut plan = Plan::new(el, kind);
    if format == Format::Ascii {
        // Read as text, a uchar may hold anything.
        plan.fixed = None;
        plan.lone_list = None;
    } else if plan.rgb.is_some() {
        plan.bytes = byte_colors();
    }
    if kind == Kind::Other || kind == Kind::Edge && plan.edge.is_none() {
        body.skipped.push(el.name.clone());
    }
    if kind == Kind::Vertex {
        ensure!(
            plan.xyz.iter().all(|&s| s != usize::MAX),
            Corrupt,
            "PLY vertices have no x, y and z"
        );
        let n = el.count;
        body.splats = body.points_only && GAUSSIAN.iter().all(|&p| el.slot(&[p]).is_some());
        if body.splats {
            // The splats' colors are worked out with the rest of them.
            plan.splat = None;
            ensure!(
                n <= body.max_splats,
                LimitExceeded,
                "{n} gaussian splats, more than the {} allowed (Keep::max_points)",
                body.max_splats
            );
        }
        let v = &mut body.verts;
        reserve(&mut v.pos, n)?;
        reserve(&mut v.nrm, if plan.n.is_some() { n } else { 0 })?;
        reserve(
            &mut v.col,
            if plan.rgb.is_some() || plan.splat.is_some() {
                n
            } else {
                0
            },
        )?;
        reserve(&mut v.uv, if plan.uv.is_some() { n } else { 0 })?;
        body.extra_names = plan.extra.iter().map(|&k| el.props[k].name.clone()).collect();
        v.extra = Vec::new();
        for _ in &plan.extra {
            let mut col = Vec::new();
            reserve(&mut col, n)?;
            v.extra.push(col);
        }
    }
    let floats = if kind == Kind::Vertex {
        Floats::new(format, &plan, body)
    } else {
        None
    };
    // Faces become triangles as they're read when their vertices are known and their corners
    // share the vertices' data.
    let direct = kind == Kind::Face
        && body.have_verts
        && plan.rgb.is_none()
        && plan.uv_list.is_none()
        && plan.texnumber.is_none();
    // Sized from the first block's triangles per face once it's read.
    let mut sized = !direct;
    body.vals.resize(el.props.len(), 0.0);
    let mut done = 0;
    while done < el.count {
        blocks.fill()?;
        let all = blocks.data();
        let end = if blocks.at_end() || format != Format::Ascii {
            all.len()
        } else {
            stream::words(all).unwrap_or(0)
        };
        let mut cur = Cursor {
            format,
            d: &all[..end],
            pos: 0,
        };
        let mut kept = 0;
        let before = done;
        if let Some(f) = &floats {
            let m = f.read(cur.d, el.count - done, body);
            done += m;
            kept = m * f.size;
            cur.pos = kept;
        }
        while done < el.count {
            let r = match kind {
                Kind::Vertex => vertex(&mut cur, el, &plan, body),
                Kind::Face => face(&mut cur, el, &plan, direct, body),
                Kind::Strip => strip(&mut cur, el, &plan, body),
                Kind::Edge if plan.edge.is_some() => edge(&mut cur, el, &plan, body),
                Kind::Edge | Kind::Other => other(&mut cur, el, &plan),
            };
            match r {
                Ok(()) => {
                    done += 1;
                    kept = cur.pos;
                }
                Err(Stop::Short) => break,
                Err(Stop::Bad(m)) => bail!(Corrupt, "{m}"),
            }
        }
        let buffered = all.len();
        blocks.consume(kept);
        if done == before {
            if blocks.at_end() {
                body.cut = true;
                break;
            }
            // One record is longer than what's buffered.
            blocks.need(buffered * 2 + 64)?;
        }
        ensure!(
            body.tris.len() / 3 <= body.limit,
            LimitExceeded,
            "too many triangles to convert (more than {})",
            body.limit
        );
        if !sized && done > 0 {
            sized = true;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a capacity guess from the rate so far; the limit caps it"
            )]
            let est = (body.tris.len() as f64 / done as f64 * el.count as f64 * 1.01) as usize + 48;
            let est = est.min(body.limit.saturating_mul(3));
            let more = est.saturating_sub(body.tris.len());
            reserve(&mut body.tris, more)?;
        }
    }
    if kind == Kind::Vertex {
        body.have_verts = true;
    }
    Ok(())
}

/// Where a gaussian splat vertex's properties are in its record when every property read is a
/// little-endian `float`, as trainers write them: the record's size and the offsets of the
/// position, the normal and each custom property. Such vertices are read a column at a time,
/// without picking each value's type.
struct Floats {
    size: usize,
    xyz: [usize; 3],
    n: Option<[usize; 3]>,
    extra: Vec<usize>,
}

impl Floats {
    fn new(format: Format, plan: &Plan, body: &Body) -> Option<Floats> {
        let (size, offsets) = plan.fixed.as_ref()?;
        let plain = plan.rgb.is_none() && plan.alpha.is_none() && plan.uv.is_none() && plan.splat.is_none();
        if format != Format::Le || !body.splats || !plain || *size == 0 {
            return None;
        }
        if plan.reads.iter().any(|&(_, _, ty)| ty != Ty::F32) {
            return None;
        }
        let at = |k: usize| offsets.get(k).copied();
        Some(Floats {
            size: *size,
            xyz: [at(plan.xyz[0])?, at(plan.xyz[1])?, at(plan.xyz[2])?],
            n: match plan.n {
                Some([a, b, c]) => Some([at(a)?, at(b)?, at(c)?]),
                None => None,
            },
            extra: plan.extra.iter().map(|&k| at(k)).collect::<Option<_>>()?,
        })
    }

    /// Reads the whole records at the start of `d`, at most `left` of them, as [`vertex`]
    /// would; how many it read.
    fn read(&self, d: &[u8], left: usize, body: &mut Body) -> usize {
        let m = (d.len() / self.size).min(left);
        let rows = || d.chunks_exact(self.size).take(m);
        let f = |r: &[u8], at: usize| {
            r.get(at..)
                .and_then(<[u8]>::first_chunk::<4>)
                .map_or(0.0, |b| f32::from_le_bytes(*b))
        };
        let v = &mut body.verts;
        v.pos.extend(rows().map(|r| self.xyz.map(|at| f(r, at))));
        if let Some(n) = self.n {
            for r in rows() {
                v.nrm
                    .push(vec3::unit_above(n.map(|at| f(r, at)), 0.0).unwrap_or_else(|| {
                        body.zero_normals += 1;
                        [0.0; 3]
                    }));
            }
        }
        for (col, &at) in v.extra.iter_mut().zip(&self.extra) {
            col.extend(rows().map(|r| f(r, at)));
        }
        body.points += m;
        m
    }
}

/// Reads an item's scalar properties into `vals` (those the plan wants) and its lists into
/// `idx` (vertex indices) and `tc` (texture coordinates).
#[inline]
fn record(
    cur: &mut Cursor,
    el: &Element,
    plan: &Plan,
    vals: &mut [f64],
    idx: &mut Vec<u32>,
    tc: &mut Vec<f32>,
) -> Got<()> {
    let be = cur.format == Format::Be;
    if let Some((size, _)) = plan.fixed {
        let rec = cur.d.get(cur.pos..cur.pos + size).ok_or(Stop::Short)?;
        for &(k, off, ty) in &plan.reads {
            vals[k] = at(&rec[off..], ty, be);
        }
        cur.pos += size;
        return Ok(());
    }
    let item = cur.pos;
    idx.clear();
    tc.clear();
    if let Some(ty) = plan.lone_list {
        let d = &cur.d[cur.pos..];
        let n = usize::from(*d.first().ok_or(Stop::Short)?);
        let list = d.get(1..1 + 4 * n).ok_or(Stop::Short)?;
        for b in list.as_chunks::<4>().0 {
            let b = ordered::<4>(b, be);
            idx.push(match ty {
                Ty::I32 => u32::try_from(i32::from_le_bytes(b)).unwrap_or(u32::MAX),
                _ => u32::from_le_bytes(b),
            });
        }
        cur.pos += 1 + 4 * n;
        return Ok(());
    }
    for (k, p) in el.props.iter().enumerate() {
        match p.count {
            None if plan.wanted[k] => vals[k] = cur.value(p.ty)?,
            None => cur.skip(p.ty, 1)?,
            Some(cty) => {
                let n = cur.list_len(cty, item)?;
                if Some(k) == plan.index_list {
                    for _ in 0..n {
                        idx.push(cur.index(p.ty)?);
                    }
                } else if Some(k) == plan.uv_list {
                    for _ in 0..n {
                        tc.push(f32_of(cur.value(p.ty)?));
                    }
                } else {
                    cur.skip(p.ty, n)?;
                }
            }
        }
    }
    Ok(())
}

fn vertex(cur: &mut Cursor, el: &Element, plan: &Plan, body: &mut Body) -> Got<()> {
    record(cur, el, plan, &mut body.vals, &mut body.idx, &mut body.tc)?;
    body.points += 1;
    let vals = &body.vals;
    let v = &mut body.verts;
    let g = |s: usize| f32_of(vals[s]);
    v.pos.push(plan.xyz.map(g));
    if let Some(n) = plan.n {
        v.nrm.push(vec3::unit_above(n.map(g), 0.0).unwrap_or_else(|| {
            body.zero_normals += 1;
            [0.0; 3]
        }));
    }
    if let Some(s) = plan.rgb {
        v.col.push(color(el, vals, &s, plan.alpha, &plan.bytes));
    } else if let Some((dc, opacity)) = plan.splat {
        v.col.push(splat_color(vals, dc, opacity));
    }
    if let Some((a, b)) = plan.uv {
        v.uv.push([g(a), 1.0 - g(b)]);
    }
    for (col, &k) in v.extra.iter_mut().zip(&plan.extra) {
        col.push(g(k));
    }
    Ok(())
}

/// An edge's two vertices; one that isn't a vertex index is dropped.
fn edge(cur: &mut Cursor, el: &Element, plan: &Plan, body: &mut Body) -> Got<()> {
    record(cur, el, plan, &mut body.vals, &mut body.idx, &mut body.tc)?;
    if let Some(ends) = plan.edge {
        let ends = ends.map(|k| body.vals[k]);
        if ends
            .iter()
            .all(|&e| e >= 0.0 && e < f64::from(u32::MAX) && e.fract() == 0.0)
        {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "checked just above"
            )]
            body.edges.extend(ends.map(|e| e as u32));
        }
    }
    Ok(())
}

fn face(cur: &mut Cursor, el: &Element, plan: &Plan, direct: bool, body: &mut Body) -> Got<()> {
    record(cur, el, plan, &mut body.vals, &mut body.idx, &mut body.tc)?;
    if plan.index_list.is_none() {
        return Ok(());
    }
    let idx = &body.idx;
    if direct {
        let nv = body.verts.pos.len();
        if idx.len() < 3 || idx.iter().any(|&i| i as usize >= nv) {
            return Ok(());
        }
        if idx.len() == 3 {
            body.tris.extend_from_slice(idx);
            return Ok(());
        }
        if let &[a, b, c, d] = &idx[..] {
            let pos = &body.verts.pos;
            let q = [a, b, c, d];
            if let Some(t) = quad(&q.map(|i| pos[i as usize])) {
                body.tris.extend(t.iter().flatten().map(|&k| q[k as usize]));
                return Ok(());
            }
        }
        body.pts.clear();
        body.pts.extend(idx.iter().map(|&i| body.verts.pos[i as usize]));
        body.local.clear();
        triangulate(&body.pts, &mut body.local);
        body.tris
            .extend(body.local.iter().flat_map(|t| t.map(|k| idx[k as usize])));
        return Ok(());
    }
    let f = &mut body.faces;
    f.starts.push(f.corners.len());
    f.corners.extend_from_slice(idx);
    if plan.uv_list.is_some() {
        let t = &body.tc;
        for c in 0..idx.len() {
            let get = |k: usize| t.get(k).copied().unwrap_or(0.0);
            f.corner_uv.push([get(2 * c), 1.0 - get(2 * c + 1)]);
        }
    }
    if let Some(s) = plan.rgb {
        let c = color(el, &body.vals, &s, plan.alpha, &plan.bytes);
        f.corner_col.extend(std::iter::repeat_n(c, idx.len()));
    }
    if let Some(k) = plan.texnumber {
        let t = body.vals[k];
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "checked to be a small whole number"
        )]
        f.texture.push(if (0.0..65536.0).contains(&t) { t as u32 } else { 0 });
    }
    Ok(())
}

/// Strips separated by -1; every other triangle's order flips to keep the winding.
fn strip(cur: &mut Cursor, el: &Element, plan: &Plan, body: &mut Body) -> Got<()> {
    record(cur, el, plan, &mut body.vals, &mut body.idx, &mut body.tc)?;
    if plan.index_list.is_none() {
        return Ok(());
    }
    for run in body.idx.split(|&i| i == u32::MAX) {
        for k in 2..run.len() {
            let (a, b, c) = (run[k - 2], run[k - 1], run[k]);
            body.strip_tris.push(if k % 2 == 0 { [a, b, c] } else { [b, a, c] });
        }
    }
    Ok(())
}

/// An element the reader doesn't use: stepped over.
fn other(cur: &mut Cursor, el: &Element, plan: &Plan) -> Got<()> {
    if let (Some((size, _)), false) = (&plan.fixed, cur.format == Format::Ascii) {
        if cur.pos + size > cur.d.len() {
            return Err(Stop::Short);
        }
        cur.pos += size;
        return Ok(());
    }
    let item = cur.pos;
    for p in &el.props {
        let n = match p.count {
            None => 1,
            Some(cty) => cur.list_len(cty, item)?,
        };
        cur.skip(p.ty, n)?;
    }
    Ok(())
}

/// The primitives: shared vertices as the file has them, or split where faces give their
/// corners their own texture coordinates or colors, with a primitive per texture when faces
/// pick one of several. Also the source vertex of each drawn vertex, by primitive, when they
/// were split; `None` when they are the file's own. The vertices' positions, normals, texture
/// coordinates and colors are taken.
fn build(
    body: &mut Body,
    textures: &[String],
    path: &Path,
    assets: &dyn Assets,
    images: &mut Images,
) -> Result<(Mesh, Option<Vec<Vec<u32>>>)> {
    let limit = body.limit;
    let zero_normals = body.zero_normals;
    let verts = &mut body.verts;
    let mut tris = std::mem::take(&mut body.tris);
    let faces = std::mem::take(&mut body.faces);
    let strip_tris = std::mem::take(&mut body.strip_tris);
    let nv = verts.pos.len();
    // Faces pick one of several textures: a material and primitive each.
    let slots = if textures.len() > 1 && !faces.texture.is_empty() {
        textures.len()
    } else {
        1
    };
    // Faces kept for later become triangles, as corner numbers (into faces.corners) so
    // per-corner data follows, and each triangle's texture when there are several.
    let mut corner_tris: Vec<[usize; 3]> = Vec::new();
    let mut corner_slot: Vec<u32> = Vec::new();
    let mut local: Vec<[u32; 3]> = Vec::new();
    let mut pts: Vec<[f32; 3]> = Vec::new();
    for (f, &s) in faces.starts.iter().enumerate() {
        let end = faces.starts.get(f + 1).copied().unwrap_or(faces.corners.len());
        let face = &faces.corners[s..end];
        if face.len() < 3 || face.iter().any(|&i| i as usize >= nv) {
            continue;
        }

        if face.len() == 3 {
            corner_tris.push([s, s + 1, s + 2]);
        } else {
            pts.clear();
            pts.extend(face.iter().map(|&i| verts.pos[i as usize]));
            local.clear();
            triangulate(&pts, &mut local);
            corner_tris.extend(local.iter().map(|t| t.map(|k| s + k as usize)));
        }
        if slots > 1 {
            // A texture number past the list takes the first.
            let t = faces.texture.get(f).copied().filter(|&t| (t as usize) < slots);
            corner_slot.resize(corner_tris.len(), t.unwrap_or(0));
        }
        ensure!(
            corner_tris.len() <= limit,
            LimitExceeded,
            "too many triangles to convert (more than {limit})"
        );
    }
    let strips: Vec<[u32; 3]> = strip_tris
        .into_iter()
        .filter(|t| t.iter().all(|&i| (i as usize) < nv))
        .collect();
    let total = tris.len() / 3 + corner_tris.len() + strips.len();
    ensure!(
        total <= limit,
        LimitExceeded,
        "too many triangles to convert (more than {limit})"
    );
    ensure!(total > 0, Empty, "PLY has no faces");

    let per_corner = !faces.corner_uv.is_empty() || !faces.corner_col.is_empty() || slots > 1;
    let has_uv = !faces.corner_uv.is_empty() || verts.uv.len() == nv && nv > 0;
    let has_col = !faces.corner_col.is_empty() || verts.col.len() == nv && nv > 0;
    // Normals that are all zero are placeholders, as good as none.
    let has_nrm = verts.nrm.len() == nv && zero_normals < nv;
    let mut prims: Vec<Prim> = (0..slots)
        .map(|k| Prim {
            material: k,
            ..Prim::default()
        })
        .collect();
    let mut srcs: Vec<Vec<u32>> = vec![Vec::new(); slots];
    if per_corner {
        // Generated normals are smoothed over the file's vertices, so splitting a vertex by its
        // corners' data leaves no seam.
        let smooth = if has_nrm && zero_normals == 0 {
            Vec::new()
        } else {
            let mut by_vertex = Vec::with_capacity(total * 3);
            by_vertex.extend_from_slice(&tris);
            by_vertex.extend(corner_tris.iter().flatten().map(|&c| faces.corners[c]));
            by_vertex.extend(strips.iter().flatten());
            smooth_normals(&verts.pos, &by_vertex)
        };
        // One output vertex per distinct (texture, vertex, corner data).
        let mut seen: FastMap<(u32, u32, [u32; 2], [u32; 4]), u32> = FastMap::default();
        let mut emit = |slot: u32, v: u32, uv: Option<[f32; 2]>, col: Option<[f32; 4]>| {
            let key = (
                slot,
                v,
                uv.unwrap_or_default().map(f32::to_bits),
                col.unwrap_or_default().map(f32::to_bits),
            );
            let p = &mut prims[slot as usize];
            let id = *seen.entry(key).or_insert_with(|| {
                srcs[slot as usize].push(v);
                let at = v as usize;
                p.positions.push(verts.pos[at]);
                let own = verts.nrm.get(at).copied().filter(|&n| has_nrm && !is_zero(n));
                p.normals
                    .push(own.or_else(|| smooth.get(at).copied()).unwrap_or([0.0, 1.0, 0.0]));
                if has_uv {
                    p.uvs
                        .push(uv.or_else(|| verts.uv.get(at).copied()).unwrap_or([0.0, 1.0]));
                }
                if has_col {
                    p.colors
                        .push(col.or_else(|| verts.col.get(at).copied()).unwrap_or([1.0; 4]));
                }
                u32_of(p.positions.len() - 1)
            });
            p.indices.push(id);
        };
        for &v in &tris {
            emit(0, v, None, None);
        }
        for (k, t) in corner_tris.iter().enumerate() {
            let slot = corner_slot.get(k).copied().unwrap_or(0);
            for &c in t {
                emit(
                    slot,
                    faces.corners[c],
                    faces.corner_uv.get(c).copied(),
                    faces.corner_col.get(c).copied(),
                );
            }
        }
        for &v in strips.iter().flatten() {
            emit(0, v, None, None);
        }
    } else {
        let p = &mut prims[0];
        tris.reserve_exact(total * 3 - tris.len());
        tris.extend(corner_tris.iter().flatten().map(|&c| faces.corners[c]));
        tris.extend(strips.iter().flatten());
        if tris.capacity() > tris.len() + tris.len() / 16 + 16 {
            tris.shrink_to_fit();
        }
        p.indices = tris;
        p.positions = std::mem::take(&mut verts.pos);
        if has_nrm {
            p.normals = std::mem::take(&mut verts.nrm);
        }
        if has_uv {
            p.uvs = std::mem::take(&mut verts.uv);
        }
        if has_col {
            p.colors = std::mem::take(&mut verts.col);
        }
        if !has_nrm {
            p.normals = smooth_normals(&p.positions, &p.indices);
        } else if zero_normals > 0 {
            // Only the vertices without a normal of their own take a smooth one.
            let smooth = smooth_normals(&p.positions, &p.indices);
            for (n, s) in p.normals.iter_mut().zip(smooth) {
                if is_zero(*n) {
                    *n = s;
                }
            }
        }
    }

    let mut mesh = Mesh::default();
    // The default material: vertex colors show as they are; otherwise the usual gray.
    let base = crate::legacy_material::default_material(has_col);
    let dir = path.parent().unwrap_or(Path::new(""));
    for k in 0..slots {
        // One texture: the last the file names, as before there were several.
        let texture = if slots > 1 { textures.get(k) } else { textures.last() };
        let name = if slots > 1 {
            texture.map_or_else(|| format!("texture{k}"), Clone::clone)
        } else {
            base.name.clone()
        };
        let mut mat = Material { name, ..base.clone() };
        if has_uv
            && let Some(t) = texture
            && let Some(i) = images.find(assets, &[t], &[dir])
        {
            mat.color_image = Some(i);
            mat.color = [1.0; 4];
        }
        mesh.materials.push(mat);
    }
    let mut kept = Vec::new();
    for (p, src) in prims.into_iter().zip(srcs) {
        if !p.indices.is_empty() || slots == 1 {
            mesh.prims.push(p);
            kept.push(src);
        }
    }
    Ok((mesh, per_corner.then_some(kept)))
}

/// The program a header comment says wrote the file: `Blender 4.2.0` from `Created in Blender version 4.2.0`,
/// `VCGLIB` from `VCGLIB generated`, or what follows `Generated by`, `Exported from` or `Exported by`.
fn writer(comment: &str) -> Option<String> {
    let c = comment.trim();
    let lower = c.to_ascii_lowercase();
    let after = |p: &str| {
        lower
            .starts_with(p)
            .then(|| c[p.len()..].trim())
            .filter(|t| !t.is_empty())
    };
    if let Some(t) = after("created in ") {
        return Some(t.replacen(" version ", " ", 1));
    }
    if let Some(t) = c.strip_suffix(" generated").filter(|t| !t.contains(' ')) {
        return Some(t.to_owned());
    }
    ["generated by ", "exported from ", "exported by "]
        .iter()
        .find_map(|p| after(p))
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(bytes: &[u8]) -> Result<Document> {
        ply(Input::memory(bytes, "t.ply"), &crate::assets::Memory::new(), Keep::ALL)
    }

    fn read(text: &str) -> Document {
        load(text.as_bytes()).unwrap()
    }

    fn mesh(text: &str) -> Mesh {
        read(text).flatten(crate::document::Pose::Rest).unwrap().unwrap()
    }

    #[test]
    fn whole_number_colors_scale_by_their_type() {
        let file = |ty: &str, v: [u32; 3]| {
            format!(
                "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n\
                 property {ty} red\nproperty {ty} green\nproperty {ty} blue\nelement face 1\n\
                 property list uchar int vertex_indices\nend_header\n0 0 0 {} {} {}\n1 0 0 {} {} {}\n\
                 0 1 0 {} {} {}\n3 0 1 2\n",
                v[0], v[1], v[2], v[0], v[1], v[2], v[0], v[1], v[2]
            )
        };
        let half = srgb_to_linear(128.0 / 255.0);
        // A 32-bit channel holds 0 to 255, as the writers that use one give it.
        for ty in ["int", "uint", "int32", "uint32", "uchar", "short", "char"] {
            let c = mesh(&file(ty, [128, 0, 255])).prims[0].colors[0];
            let want = if ty == "short" {
                srgb_to_linear(128.0 / 65535.0)
            } else {
                half
            };
            assert!((c[0] - want).abs() < 1e-6, "{ty}: {c:?}");
            if ty != "short" {
                assert!(
                    c[1].abs() < 1e-9 && (c[2] - 1.0).abs() < 1e-9 && (c[3] - 1.0).abs() < 1e-9,
                    "{ty}: {c:?}"
                );
            }
        }
        let c = mesh(&file("ushort", [65535, 0, 32768])).prims[0].colors[0];
        assert!(
            (c[0] - 1.0).abs() < 1e-9 && (c[2] - srgb_to_linear(32768.0 / 65535.0)).abs() < 1e-6,
            "{c:?}"
        );
        // Past 255 a 32-bit channel is full.
        assert_eq!(mesh(&file("int", [70000, 0, 0])).prims[0].colors[0][0], 1.0);
    }

    #[test]
    fn header_comments_and_the_writer_are_the_source_info() {
        let doc = read(
            "ply
format ascii 1.0
comment Created in Blender version 4.2.0
comment a cube
obj_info a note
             element vertex 3
property float x
property float y
property float z
             element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
3 0 1 2
",
        );
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("PLY"));
        assert_eq!(info.version.as_deref(), Some("1.0"));
        assert_eq!(info.tool.as_deref(), Some("Blender 4.2.0"));
        assert_eq!(info.comments, ["Created in Blender version 4.2.0", "a cube"]);
        assert_eq!(info.meters_per_unit, crate::source::Fact::Unknown);
        assert_eq!(info.up_axis, crate::source::Fact::Unknown);
        assert_eq!(info.converted, crate::source::Converted::default());
        assert_eq!(doc.json()["asset"]["extras"]["source"]["objInfo"], json!(["a note"]));
        assert_eq!(writer("VCGLIB generated").as_deref(), Some("VCGLIB"));
        assert_eq!(writer("made by hand"), None);
    }

    #[test]
    fn a_point_clouds_byte_colors_are_normalized_16_bit() {
        let mut bytes = b"ply
format binary_little_endian 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"
        .to_vec();
        let rgb = [[255u8, 0, 0], [0, 1, 128], [12, 13, 14]];
        for (k, c) in rgb.iter().enumerate() {
            for v in [f32::from(u8::try_from(k).unwrap()), 0.0, 0.0] {
                bytes.extend(v.to_le_bytes());
            }
            bytes.extend(c);
        }
        let doc = load(&bytes).unwrap();
        let index = |v: &Value| usize::try_from(v.as_u64().unwrap()).unwrap();
        let a = &doc.json()["accessors"][index(&doc.json()["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"])];
        assert_eq!(
            (a["componentType"].as_u64(), a["type"].as_str()),
            (Some(5123), Some("VEC4"))
        );
        assert_eq!(a["normalized"], true);
        assert_eq!(doc.json()["bufferViews"][index(&a["bufferView"])]["byteLength"], 3 * 8);
        // Each byte value reads back as the byte it was, made linear.
        let m = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        for (got, want) in m.prims[0].colors.iter().zip(rgb) {
            for (g, w) in got.iter().zip(want) {
                assert!(
                    (g - srgb_to_linear(f32::from(w) / 255.0)).abs() < 1e-4,
                    "{got:?} {want:?}"
                );
            }
            assert!((got[3] - 1.0).abs() < 1e-6);
        }
    }

    fn attributes(doc: &Document) -> Vec<String> {
        let prim = &doc.json()["meshes"][0]["primitives"][0]["attributes"];
        prim.as_object().unwrap().keys().cloned().collect()
    }

    /// How many vertices the first primitive has.
    fn vertex_count(doc: &Document) -> u64 {
        let pos = doc.json()["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
            .as_u64()
            .unwrap();
        doc.json()["accessors"][usize::try_from(pos).unwrap()]["count"]
            .as_u64()
            .unwrap()
    }

    #[test]
    fn counts_times_record_size_past_the_file_are_cut_to_it_before_reserving() {
        // A thousand one-byte properties times ten million vertices in a file of a few bytes.
        let mut head = String::from("ply\nformat binary_little_endian 1.0\nelement vertex 10000000\n");
        head.push_str("property float x\nproperty float y\nproperty float z\n");
        for k in 0..1000 {
            head.push_str("property uchar p");
            head.push_str(&k.to_string());
            head.push('\n');
        }
        head.push_str("end_header\n");
        let mut bytes = head.into_bytes();
        bytes.extend(std::iter::repeat_n(0u8, 20_000_000));
        let doc = load(&bytes).unwrap();
        assert_eq!(vertex_count(&doc), 20_000_000 / 1012);
        assert_eq!(doc.json()["asset"]["extras"]["source"]["truncated"], true);
        // As ASCII: each property needs a character and a separator; two whole vertices.
        let text = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n\
                    property float z\nend_header\n0 0 0 1 0 0 0 1";
        assert_eq!(vertex_count(&load(text.as_bytes()).unwrap()), 2);
        // Nothing at all where the header counts vertices.
        let text = b"ply\nformat ascii 1.0\nelement vertex 9\nproperty float x\nproperty float y\nproperty float z\n\
                     end_header\n";
        let err = load(text).err().unwrap().to_string();
        assert!(err.contains("is cut short"), "{err}");
    }

    #[test]
    fn a_file_cut_short_keeps_its_whole_items() {
        // Four faces counted, two and a half there.
        let text = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n\
                    property float z\nelement face 4\nproperty list uchar int vertex_indices\n\
                    end_header\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n3 0 1 2\n3 1 3 2\n3 0 1";
        let doc = read(text);
        assert_eq!(doc.json()["asset"]["extras"]["source"]["truncated"], true);
        assert_eq!(mesh(text).triangles(), 2);
    }

    #[test]
    fn a_list_an_ascii_line_leaves_out_is_empty() {
        // Vertices declaring a list, written without it: a point cloud, one vertex a line.
        let text = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n\
                    property float z\nproperty list uchar uint vertex_indices\nend_header\n\
                    7.5 -2.25 0\r\n-1 2 3\r\n4 5 6\r\n";
        let doc = read(text);
        assert_eq!(vertex_count(&doc), 3);
        assert!(doc.json()["asset"]["extras"]["source"].get("truncated").is_none());
        // Items written a value a line still read, and a list on its item's line does.
        let faces = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n\
                     property float z\nelement face 1\nproperty uchar flags\n\
                     property list uchar int vertex_indices\nend_header\n0\n0\n0\n1 0 0\n0 1 0\n7 3 0 1 2\n";
        assert_eq!(mesh(faces).triangles(), 1);
    }

    #[test]
    fn errors_show_file_text_printably() {
        let text = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty \0\0\0 x\nend_header\n0\n";
        let err = load(text).err().unwrap().to_string();
        assert!(err.contains(r#""\0\0\0""#), "{err}");
        assert!(!err.chars().any(char::is_control), "{err}");
        let long = format!("ply\nformat {}\nend_header\n", "x".repeat(1000));
        let err = load(long.as_bytes()).err().unwrap().to_string();
        assert!(err.len() < 100, "{err}");
    }

    const TRIANGLE: &str = "format ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n\
                            property float z\nelement face 1\nproperty list uchar int vertex_indices\n\
                            end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";

    #[test]
    fn a_byte_order_mark_before_the_header() {
        let text = format!("\u{feff}ply\r\n{TRIANGLE}");
        assert_eq!(mesh(&text).triangles(), 1);
    }

    #[test]
    fn normals_by_their_other_names_and_zero_ones_filled_in() {
        let m = mesh(
            "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n\
             property float normal_x\nproperty float normal_y\nproperty float normal_z\n\
             element face 1\nproperty list uchar int vertex_indices\nend_header\n\
             0 0 0 0 0 2\n1 0 0 0 0 0\n0 1 0 1 0 0\n3 0 1 2\n",
        );
        let n = &m.prims[0].normals;
        assert_eq!(n[0], [0.0, 0.0, 1.0], "normal_x/y/z read and made unit");
        assert_eq!(n[1], [0.0, 0.0, 1.0], "a zero normal takes the smooth one");
        assert_eq!(n[2], [1.0, 0.0, 0.0], "the file's own normals stay");
        // All zero: as good as none, so the generated ones.
        let m = mesh(
            "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n\
             property float nx\nproperty float ny\nproperty float nz\n\
             element face 1\nproperty list uchar int vertex_indices\nend_header\n\
             0 0 0 0 0 0\n1 0 0 0 0 0\n0 1 0 0 0 0\n3 0 1 2\n",
        );
        assert!(m.prims[0].normals.iter().all(|n| (n[2] - 1.0).abs() < 1e-6));
    }

    #[test]
    fn header_lines_may_end_in_a_lone_carriage_return() {
        let text = format!("ply\r{}", TRIANGLE.replace('\n', "\r"));
        assert_eq!(mesh(&text).triangles(), 1);
    }

    #[test]
    fn faces_pick_one_of_several_textures() {
        let doc = read(
            "ply\nformat ascii 1.0\ncomment TextureFile a.png\ncomment TextureFile b.png\n\
             element vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
             element face 2\nproperty list uchar int vertex_indices\nproperty list uchar float texcoord\n\
             property int texnumber\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n\
             3 0 1 2 6 0 0 1 0 1 1 1\n3 0 2 3 6 0 0 1 1 0 1 0\n",
        );
        let prims = doc.json()["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(prims.len(), 2);
        assert_eq!(prims[0]["material"], 0);
        assert_eq!(prims[1]["material"], 1);
        assert_eq!(doc.json()["materials"][0]["name"], "a.png");
        assert_eq!(doc.json()["materials"][1]["name"], "b.png");
        let m = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(m.triangles(), 2);
    }

    #[test]
    fn generated_normals_have_no_seam_where_texture_coordinates_split_a_vertex() {
        // A folded strip: vertex 1 and 2 are split by their faces' texture coordinates, and
        // still share one smooth normal.
        let m = mesh(
            "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
             element face 2\nproperty list uchar int vertex_indices\nproperty list uchar float texcoord\n\
             end_header\n0 0 0\n1 0 0\n1 1 0\n1 0 -1\n\
             3 0 1 2 6 0 0 1 0 1 1\n3 1 3 2 6 0.5 0 1 0 0.5 1\n",
        );
        let p = &m.prims[0];
        assert_eq!(p.positions.len(), 6);
        for (k, a) in p.positions.iter().enumerate() {
            for (j, b) in p.positions.iter().enumerate() {
                if a.map(f32::to_bits) == b.map(f32::to_bits) {
                    assert!(
                        p.normals[k].map(f32::to_bits) == p.normals[j].map(f32::to_bits),
                        "one normal at {a:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn gaussian_splats_show_their_color_and_drop_zero_normals() {
        let doc = read(
            "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\n\
             property float nx\nproperty float ny\nproperty float nz\nproperty float f_dc_0\n\
             property float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nend_header\n\
             0 0 0 0 0 0 0 0 0 0\n1 0 0 0 0 0 1.7724539 -1.7724539 0 -100\n",
        );
        let names = attributes(&doc);
        assert!(names.contains(&"COLOR_0".to_owned()), "{names:?}");
        assert!(!names.contains(&"NORMAL".to_owned()), "{names:?}");
        assert!(names.contains(&"_F_DC_0".to_owned()) && names.contains(&"_OPACITY".to_owned()));
        let c = splat_color(&[0.0, 0.0, 0.0, 0.0], [0, 1, 2], Some(3));
        assert!((c[0] - srgb_to_linear(0.5)).abs() < 1e-6 && (c[3] - 0.5).abs() < 1e-6);
        let c = splat_color(&[1.772_453_9, -1.772_453_9, 0.0, -100.0], [0, 1, 2], Some(3));
        assert!((c[0] - 1.0).abs() < 1e-5 && c[1].abs() < 1e-6 && c[3] < 1e-6, "{c:?}");
    }

    /// A binary 3D gaussian splatting file of `n` splats with `rest` `f_rest_*` properties: splat
    /// `i`'s position is (i, 2i, 3i), its `f_rest_k` is `k + 100 i`, its rotation (w first) is a
    /// quarter turn about x.
    #[allow(clippy::cast_precision_loss, reason = "test files are small")]
    fn splat_file(n: usize, rest: usize) -> Vec<u8> {
        use std::fmt::Write as _;
        let mut head = String::from("ply\nformat binary_little_endian 1.0\n");
        writeln!(head, "element vertex {n}").unwrap();
        let mut names: Vec<String> = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            .map(str::to_owned)
            .to_vec();
        names.extend((0..rest).map(|k| format!("f_rest_{k}")));
        names.extend(
            [
                "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
            ]
            .map(str::to_owned),
        );
        for name in &names {
            writeln!(head, "property float {name}").unwrap();
        }
        head.push_str("end_header\n");
        let mut out = head.into_bytes();
        for i in 0..n {
            let i = i as f32;
            let mut row = vec![i, 2.0 * i, 3.0 * i, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0];
            row.extend((0..rest).map(|k| k as f32 + 100.0 * i));
            row.extend([0.0, -2.0, -3.0, 2.0f32.ln(), 1.0, 1.0, 0.0, 0.0]);
            out.extend(row.iter().flat_map(|v| v.to_le_bytes()));
        }
        out
    }

    fn floats(doc: &Document, name: &str) -> Vec<f32> {
        let a = &doc.json()["meshes"][0]["primitives"][0]["attributes"][name];
        doc.read_floats(usize::try_from(a.as_u64().unwrap()).unwrap())
            .unwrap()
            .1
    }

    #[test]
    fn gaussian_splats_read_as_khr_gaussian_splatting() {
        let doc = load(&splat_file(3, 45)).unwrap();
        assert_eq!(doc.facts["splats"], 3);
        assert_eq!(doc.json()["extensionsUsed"][0], gaussian::EXTENSION);
        let names = attributes(&doc);
        assert!(!names.iter().any(|n| n == "NORMAL" || n.starts_with('_')), "{names:?}");
        assert!(names.iter().any(|n| n.ends_with("SH_DEGREE_3_COEF_6")), "{names:?}");
        // Turned from the convention's Y down, as the other splat readers say they turn theirs.
        let info = doc.source_info();
        assert_eq!(info.up_axis, crate::source::Fact::Assumed(crate::source::Axis::NegY));
        assert_eq!(
            info.handedness,
            crate::source::Fact::Assumed(crate::source::Handedness::Right)
        );
        assert!(info.converted.up_axis && !info.converted.handedness && !info.converted.units);
        // Right, down, forward to right, up, back.
        assert_eq!(floats(&doc, "POSITION")[3..6], [1.0, -2.0, -3.0]);
        let scale = floats(&doc, "KHR_gaussian_splatting:SCALE");
        assert!((scale[0] - (-2.0f32).exp()).abs() < 1e-6 && (scale[2] - 2.0).abs() < 1e-6);
        let opacity = floats(&doc, gaussian::OPACITY);
        assert!((opacity[0] - 0.5).abs() < 1e-6);
        // A quarter turn about x stays one about x: flipping y and z twice keeps its sense.
        let q = floats(&doc, "KHR_gaussian_splatting:ROTATION");
        assert!(
            (q[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6 && (q[3] - q[0]).abs() < 1e-6,
            "{q:?}"
        );
        // Channel-major in the file (15 reds, 15 greens, 15 blues); the first degree-1
        // coefficient runs along y, which flips.
        let sh = floats(&doc, "KHR_gaussian_splatting:SH_DEGREE_1_COEF_0");
        assert_eq!(sh[3..6], [-100.0, -115.0, -130.0]);
        let sh = floats(&doc, "KHR_gaussian_splatting:SH_DEGREE_1_COEF_2");
        assert_eq!(sh[3..6], [102.0, 117.0, 132.0]);
        let m = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(m.points(), 3);
        let c = m.prims[0].colors[0];
        assert!(
            (c[0] - srgb_to_linear(0.5 + gaussian::SH_C0)).abs() < 1e-3 && (c[3] - 0.5).abs() < 1e-3,
            "{c:?}"
        );
        doc.to_glb().unwrap();
    }

    #[test]
    fn spherical_harmonics_short_of_a_degree_keep_the_whole_degrees() {
        // 30 of degree 3's 45: degree 1 and 2 (24) are kept, the rest dropped.
        let doc = load(&splat_file(2, 30)).unwrap();
        let names = attributes(&doc);
        assert!(names.iter().any(|n| n.ends_with("SH_DEGREE_2_COEF_4")), "{names:?}");
        assert!(!names.iter().any(|n| n.contains("SH_DEGREE_3")), "{names:?}");
        // A count that isn't a multiple of three: no harmonics past the base.
        let doc = load(&splat_file(2, 7)).unwrap();
        assert!(!attributes(&doc).iter().any(|n| n.contains("SH_DEGREE_1")));
    }

    #[test]
    fn gaussian_splats_past_max_points_are_refused() {
        let keep = Keep::ALL.with_max_points(2);
        let bytes = splat_file(3, 0);
        let err = ply(Input::memory(&bytes, "t.ply"), &crate::assets::Memory::new(), keep).unwrap_err();
        assert!(err.to_string().contains("max_points"), "{err}");
    }

    /// Whole floats read a column at a time give what values read one by one do, records cut by
    /// the reader's blocks included.
    #[cfg(feature = "fs")]
    #[test]
    fn float_splats_read_by_column_as_by_value() {
        let le = splat_file(3001, 45);
        let body = le
            .windows(11)
            .position(|w| {
                w == b"end_header
"
            })
            .unwrap()
            + 11;
        let mut be = String::from_utf8(le[..body].to_vec())
            .unwrap()
            .replace("binary_little_endian", "binary_big_endian")
            .into_bytes();
        be.extend(
            le[body..]
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| [c[3], c[2], c[1], c[0]]),
        );
        let mut file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut file, &le).unwrap();
        let fast = ply(Input::file(file.path()), &crate::assets::Memory::new(), Keep::ALL).unwrap();
        let slow = load(&be).unwrap();
        let names = attributes(&fast);
        assert_eq!(names, attributes(&slow));
        assert!(names.len() > 20, "{names:?}");
        for name in &names {
            assert!(floats(&fast, name) == floats(&slow, name), "{name}");
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, reason = "test files are small")]
    fn damaged_gaussian_splats_never_panic() {
        let good = splat_file(40, 9);
        let body = good.len() - 40 * (17 + 9) * 4;
        for cut in (0..good.len()).step_by(7) {
            if let Ok(doc) = load(&good[..cut]) {
                assert!(cut >= body, "a header cut at {cut} read");
                let _ = doc.flatten(crate::document::Pose::Rest);
            }
        }
        let mut rng = crate::test_rng::Pcg32::new(17);
        for _ in 0..300 {
            let mut bad = good.clone();
            for _ in 0..4 {
                let i = body + rng.below((good.len() - body) as u32) as usize;
                bad[i] ^= 1 << rng.below(8);
            }
            let doc = load(&bad).unwrap();
            doc.to_glb().unwrap();
            let _ = doc.flatten(crate::document::Pose::Rest);
        }
        for _ in 0..300 {
            let mut bad = good.clone();
            let i = rng.below(body as u32) as usize;
            bad[i] = rng.below(256) as u8;
            if let Ok(doc) = load(&bad) {
                let _ = doc.to_glb();
                let _ = doc.flatten(crate::document::Pose::Rest);
            }
        }
    }

    #[test]
    fn extra_properties_follow_vertices_split_by_face_texcoords() {
        let doc = read(
            "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
             property float quality\nelement face 2\nproperty list uchar int vertex_indices\n\
             property list uchar float texcoord\nend_header\n\
             0 0 0 1\n1 0 0 2\n1 1 0 3\n0 1 0 4\n\
             3 0 1 2 6 0 0 1 0 1 1\n3 0 2 3 6 0.5 0.5 1 1 0 1\n",
        );
        let prim = &doc.json()["meshes"][0]["primitives"][0];
        let count = |a: &Value| doc.json()["accessors"][usize::try_from(a.as_u64().unwrap()).unwrap()]["count"].clone();
        let positions = count(&prim["attributes"]["POSITION"]);
        assert_eq!(positions, 5, "vertices 0 and 2 split by their two texture coordinates");
        assert_eq!(count(&prim["attributes"]["_QUALITY"]), positions);
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(mesh.triangles(), 2);
    }
}
