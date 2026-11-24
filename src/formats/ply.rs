//! PLY (the Stanford polygon file format), ASCII and binary in either byte order.
//!
//! Read: positions, normals (`nx` or `normal_x`), vertex colors (sRGB, as integers or floats),
//! texture coordinates under any of their usual names, polygons (triangulated in their plane) and
//! triangle strips; `MeshLab`'s per-face texture coordinates and `comment TextureFile` (several, a
//! face picking one by its `texnumber`, a material each); per-face colors. Elements the reader
//! doesn't use are skipped. Gaussian splats show their color: the zero-order spherical harmonics
//! (`f_dc_0` to `f_dc_2`) and `opacity` give the vertex colors when the file has no other. A zero
//! normal (splat files write nothing else) is a placeholder: a mesh's vertex takes a made one, and
//! a file of nothing but zeros has none. Made normals are smooth across vertices split only by
//! texture coordinates. Header lines may end in a lone `\r`, and a leading byte order mark is
//! skipped. Counts the header gives are checked against the bytes left before anything is
//! allocated for them.
//!
//! Nothing the vertices carry is lost: properties beyond those are custom attributes, named as
//! the file names them in capitals after an underscore (`_CONFIDENCE`, `_F_DC_0`). A file without
//! faces is a point cloud, drawn as points; an `edge` element's `vertex1` and `vertex2` are
//! lines. The header's comments, `obj_info` lines, encoding and the elements skipped go in the
//! asset's `extras.source`.

use std::path::Path;

use serde_json::{Map, Value, json};

use super::Input;
use super::stream::{self, Blocks};
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep, Part};
use crate::images::Images;
use crate::mesh::{FastMap, Material, Mesh, Prim, quad, smooth_normals, srgb_to_linear, triangulate};
use crate::num::{f32_of, u32_of};
use crate::{Context as _, Result};

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
            t => bail!("unknown PLY type {}", String::from_utf8_lossy(t)),
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

    /// The value that means full intensity for a color channel of this type.
    fn color_scale(self) -> f64 {
        match self {
            Ty::U8 | Ty::I8 => 255.0,
            Ty::U16 | Ty::I16 => 65535.0,
            _ => 1.0,
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
    let text = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    let mut at = 0;
    let mut first = true;
    loop {
        let rest = data
            .get(at..)
            .filter(|r| !r.is_empty())
            .context("PLY header has no end_header")?;
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
            ensure!(line == b"ply", "not a PLY file");
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
                    f => bail!("unknown PLY format {}", String::from_utf8_lossy(f)),
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
                name: String::from_utf8_lossy(name).into_owned(),
                count: std::str::from_utf8(count)
                    .ok()
                    .and_then(|c| c.parse().ok())
                    .context("bad PLY element count")?,
                props: Vec::new(),
            }),
            [b"property", b"list", c, i, name] => elements
                .last_mut()
                .context("PLY property before element")?
                .props
                .push(Prop {
                    name: String::from_utf8_lossy(name).into_owned(),
                    ty: Ty::parse(i)?,
                    count: Some(Ty::parse(c)?),
                }),
            [b"property", ty, name] => elements
                .last_mut()
                .context("PLY property before element")?
                .props
                .push(Prop {
                    name: String::from_utf8_lossy(name).into_owned(),
                    ty: Ty::parse(ty)?,
                    count: None,
                }),
            _ => {}
        }
    }
    let header = Header {
        format: format.context("PLY header names no format")?,
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

/// `n` made unit length; `None` for a zero or broken one, which says nothing.
fn unit(n: [f32; 3]) -> Option<[f32; 3]> {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    (len > 0.0 && len.is_finite()).then(|| n.map(|c| c / len))
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
    /// The vertex element's other scalar properties' names, for `Vertices::extra`.
    extra_names: Vec<String>,
    /// Elements read past.
    skipped: Vec<String>,
    // Scratch, reused from record to record.
    vals: Vec<f64>,
    idx: Vec<u32>,
    tc: Vec<f32>,
    pts: Vec<[f32; 3]>,
    local: Vec<[u32; 3]>,
}

pub(crate) fn ply(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    let path = input.path;
    let mut blocks = input.blocks()?;
    // The header, whole.
    let mut want = 4096;
    let (head, elements, start) = loop {
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
    blocks.consume(start);
    let room = blocks.size_hint().saturating_sub(start);
    let has_faces = elements
        .iter()
        .any(|e| (e.name == "face" || e.name == "tristrips") && e.count > 0);
    let mut body = Body {
        limit: input.limit,
        ..Body::default()
    };
    // Every item takes at least its smallest record, so counts the file can't hold are refused
    // before anything is reserved for them (the last ASCII value needs no separator).
    let mut least = 0usize;
    for el in &elements {
        least = least.saturating_add(el.count.saturating_mul(el.min_size(head.format)));
        ensure!(
            el.count <= room && least <= room.saturating_add(1),
            "PLY data ends early: element {} counts more items than the file holds",
            el.name
        );
    }
    for el in &elements {
        read_element(&mut blocks, head.format, el, &mut body)?;
    }
    let mut source = json!({
        "format": "PLY",
        "encoding": match head.format {
            Format::Ascii => "ascii",
            Format::Le => "binary_little_endian",
            Format::Be => "binary_big_endian",
        },
        "version": head.version,
    });
    for (key, list) in [
        ("comments", &head.comments),
        ("objInfo", &head.obj_info),
        ("skipped", &body.skipped),
    ] {
        if !list.is_empty() {
            source[key] = json!(list);
        }
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
        let (mesh, srcs) = build(&mut body, textures, path, assets, &mut images)?;
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
        if nv > 0 {
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
    // Normals that are all zero are placeholders; a point without one faces up.
    if v.nrm.len() == n && zero_normals < n {
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
        let mut colors = std::mem::take(&mut v.col).into_flattened();
        for c in &mut colors {
            *c = c.clamp(0.0, 1.0);
        }
        attributes["COLOR_0"] = json!(b.accessor(Part::f32(colors), 5126, n, "VEC4", Some(34962)));
    }
    let columns = gather(&mut v.extra, None, n);
    custom(b, &mut attributes, names, columns);
    attributes
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
        .context("not enough memory for the PLY data")
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
            "PLY vertices have no x, y and z"
        );
        let v = &mut body.verts;
        let n = el.count;
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
                Err(Stop::Bad(m)) => bail!("{m}"),
            }
        }
        let buffered = all.len();
        blocks.consume(kept);
        if done == before {
            ensure!(!blocks.at_end(), "PLY data ends early");
            // One record is longer than what's buffered.
            blocks.need(buffered * 2 + 64)?;
        }
        ensure!(
            body.tris.len() / 3 <= body.limit,
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
                let n = cur.len(cty)?;
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
        v.nrm.push(unit(n.map(g)).unwrap_or_else(|| {
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
    for p in &el.props {
        let n = match p.count {
            None => 1,
            Some(cty) => cur.len(cty)?,
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
            "too many triangles to convert (more than {limit})"
        );
    }
    let strips: Vec<[u32; 3]> = strip_tris
        .into_iter()
        .filter(|t| t.iter().all(|&i| (i as usize) < nv))
        .collect();
    let total = tris.len() / 3 + corner_tris.len() + strips.len();
    ensure!(total <= limit, "too many triangles to convert (more than {limit})");
    ensure!(total > 0, "PLY has no faces");

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
    // Vertex colors show as they are; otherwise the usual gray.
    let base = if has_col { [1.0; 4] } else { crate::mesh::GRAY };
    let dir = path.parent().unwrap_or(Path::new(""));
    for k in 0..slots {
        // One texture: the last the file names, as before there were several.
        let texture = if slots > 1 { textures.get(k) } else { textures.last() };
        let name = if slots > 1 {
            texture.map_or_else(|| format!("texture{k}"), Clone::clone)
        } else {
            "default".to_owned()
        };
        let mut mat = Material::plain(&name, base);
        if has_uv
            && let Some(t) = texture
            && let Some(i) = assets.find(t, dir).and_then(|f| images.file(assets, f, t))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn load(bytes: &[u8]) -> Result<Document<'static>> {
        ply(Input::memory(bytes, "t.ply"), &crate::assets::Memory::new(), Keep::ALL)
    }

    fn read(text: &str) -> Document<'static> {
        load(text.as_bytes()).unwrap()
    }

    fn mesh(text: &str) -> Mesh {
        read(text).flatten(crate::document::Pose::Rest).unwrap().unwrap()
    }

    fn attributes(doc: &Document<'_>) -> Vec<String> {
        let prim = &doc.json["meshes"][0]["primitives"][0]["attributes"];
        prim.as_object().unwrap().keys().cloned().collect()
    }

    #[test]
    fn counts_times_record_size_past_the_file_are_refused_before_reserving() {
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
        let err = load(&bytes).err().unwrap().to_string();
        assert!(err.contains("ends early"), "{err}");
        // As ASCII: each property needs a character and a separator.
        let text = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n\
                    property float z\nend_header\n0 0 0 1 0 0 0 1";
        assert!(load(text.as_bytes()).is_err());
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
        let prims = doc.json["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(prims.len(), 2);
        assert_eq!(prims[0]["material"], 0);
        assert_eq!(prims[1]["material"], 1);
        assert_eq!(doc.json["materials"][0]["name"], "a.png");
        assert_eq!(doc.json["materials"][1]["name"], "b.png");
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

    #[test]
    fn extra_properties_follow_vertices_split_by_face_texcoords() {
        let doc = read(
            "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
             property float quality\nelement face 2\nproperty list uchar int vertex_indices\n\
             property list uchar float texcoord\nend_header\n\
             0 0 0 1\n1 0 0 2\n1 1 0 3\n0 1 0 4\n\
             3 0 1 2 6 0 0 1 0 1 1\n3 0 2 3 6 0.5 0.5 1 1 0 1\n",
        );
        let prim = &doc.json["meshes"][0]["primitives"][0];
        let count = |a: &Value| doc.json["accessors"][usize::try_from(a.as_u64().unwrap()).unwrap()]["count"].clone();
        let positions = count(&prim["attributes"]["POSITION"]);
        assert_eq!(positions, 5, "vertices 0 and 2 split by their two texture coordinates");
        assert_eq!(count(&prim["attributes"]["_QUALITY"]), positions);
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(mesh.triangles(), 2);
    }
}
