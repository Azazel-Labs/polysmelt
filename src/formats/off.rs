//! OFF, the Object File Format: a vertex list and a face list, as text or binary.
//!
//! The header keyword says what each vertex carries beyond its position, in this order:
//! `ST` texture coordinates, `C` a color, `N` a normal, `4` a homogeneous coordinate, `n` a
//! dimension other than three (`STCN4nOFF` at the most). The counts may follow on the header's
//! own line, and `#` starts a comment anywhere. A vertex is read from its line: a color on an
//! `OFF` vertex line without the `C` keyword is taken too, as many tools write one. A face is
//! its corner count, its corners, and then on the same line an optional color: three or four
//! integers (0-255) or floats (0-1); a single number, an index into a color map, is ignored.
//!
//! `OFF BINARY` is the same in big-endian 32-bit integers and floats, a face's color given by
//! its own count of floats.
//!
//! Polygons are triangulated in their plane. Faces with a corner out of range or at a point
//! that isn't finite are dropped, as are points and lines (faces of one or two corners). Face
//! colors become vertex colors, a vertex split where faces of different colors meet; without
//! normals in the file, smooth ones are made. Colors are taken as sRGB. A file that ends early
//! draws what it has.
//!
//! OFF names no units and no up axis, so both are unknown; geometry is kept as written.

use serde_json::{Map, Value, json};

use super::Input;
use super::stream;
use crate::document::Document;
use crate::mesh::{FastMap, GRAY, Material, Prim, smooth_normals, srgb_to_linear, triangulate};
use crate::num::u32_of;
use crate::source::SourceInfo;
use crate::{Context as _, ErrorKind, Result};

/// What a file's vertices carry, from its header keyword.
#[derive(Clone, Copy, Debug, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the header keyword's letters, each a yes or no"
)]
struct Layout {
    st: bool,
    color: bool,
    normal: bool,
    /// A homogeneous coordinate after the others, which they're divided by.
    homogeneous: bool,
    /// Coordinates a position has (before the homogeneous one): 3 unless `n` gives another.
    dim: usize,
    binary: bool,
}

/// The vertices and triangles read.
#[derive(Default)]
struct Model {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    /// As written, scaled to 0-1 once the whole file is read.
    colors: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    tris: Vec<[u32; 3]>,
    /// Each triangle's face color, an index into `face_colors`, when any face has one.
    tri_colors: Vec<u32>,
    face_colors: Vec<[f32; 4]>,
    face_color_of: FastMap<[u32; 4], u32>,
    /// Whether the file ended before its counts said.
    short: bool,
    /// Faces dropped: too few corners, or a corner out of range or not finite.
    dropped: usize,
}

/// Marks a triangle without a face color.
const NONE: u32 = u32::MAX;

/// The most coordinates a position may have (`nOFF`'s dimension).
const MAX_VALUES: usize = 32;

pub(crate) fn load(input: Input<'_>) -> Result<Document<'static>> {
    let data = input.read()?;
    let data = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&data);
    let mut c = Cursor { d: data, at: 0 };
    let (layout, keyword, leftover) = header(&mut c)?;
    let mut m = Model::default();
    if layout.binary {
        binary(data, c.at, layout, input.limit, &mut m)?;
    } else {
        ascii(&mut c, layout, leftover, input.limit, &mut m)?;
    }
    ensure!(!m.tris.is_empty(), Empty, "OFF has no faces");
    let triangles = m.tris.len();
    let mut source = Map::new();
    source.insert("header".into(), json!(keyword));
    source.insert("encoding".into(), json!(if layout.binary { "binary" } else { "ASCII" }));
    if m.short {
        source.insert("truncated".into(), json!(true));
    }
    if m.dropped > 0 {
        source.insert("droppedFaces".into(), json!(m.dropped));
    }
    let info = SourceInfo {
        format: Some("OFF".into()),
        ..SourceInfo::default()
    };
    info.write(&mut source);
    let colored = !m.colors.is_empty() || !m.face_colors.is_empty();
    let prim = build(m);
    let material = Material::plain("default", if colored { [1.0; 4] } else { GRAY });
    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(triangles));
    Ok(Document::from_named(
        vec![(None, vec![prim])],
        &[material],
        Vec::new(),
        Some(Value::Object(source)),
        facts,
        &[],
    ))
}

/// Reads the header keyword (and `BINARY`, and the dimension of an `nOFF`): the layout, the
/// keyword, and a count glued to it (`OFF8`, as some files have) or standing in for it.
fn header<'a>(c: &mut Cursor<'a>) -> Result<(Layout, String, Option<&'a [u8]>)> {
    let mut layout = Layout {
        dim: 3,
        ..Layout::default()
    };
    let Some(first) = c.token(true) else {
        bail!(Empty, "OFF file is empty");
    };
    let mut t = first;
    for (prefix, flag) in [
        (&b"ST"[..], &mut layout.st),
        (b"C", &mut layout.color),
        (b"N", &mut layout.normal),
        (b"4", &mut layout.homogeneous),
    ] {
        if let Some(rest) = t.strip_prefix(prefix) {
            *flag = true;
            t = rest;
        }
    }
    let ndim = t.first() == Some(&b'n');
    if ndim {
        t = &t[1..];
    }
    let (keyword, leftover) = if t.len() >= 3 && t[..3].eq_ignore_ascii_case(b"OFF") {
        let rest = &t[3..];
        ensure!(rest.iter().all(u8::is_ascii_digit), UnknownFormat, "not an OFF file");
        (
            String::from_utf8_lossy(&first[..first.len() - rest.len()]).into_owned(),
            (!rest.is_empty()).then_some(rest),
        )
    } else if int(first).is_some() {
        // No keyword: the counts come first.
        layout = Layout {
            dim: 3,
            ..Layout::default()
        };
        (String::new(), Some(first))
    } else {
        bail!(UnknownFormat, "not an OFF file");
    };
    if leftover.is_none() {
        let back = c.at;
        if c.token(false) == Some(b"BINARY") {
            layout.binary = true;
            c.end_line();
        } else {
            c.at = back;
        }
    }
    if ndim && !layout.binary {
        let n = c
            .token(true)
            .and_then(int)
            .context(ErrorKind::Corrupt, "OFF dimension missing")?;
        ensure!((1..=32).contains(&n), Unsupported, "OFF dimension {n} is out of range");
        layout.dim = usize::try_from(n).unwrap_or(3);
    } else if ndim {
        layout.dim = 0; // Read with the counts.
    }
    Ok((layout, keyword, leftover))
}

/// The ASCII body: the counts, then the vertices and faces.
fn ascii(c: &mut Cursor<'_>, layout: Layout, leftover: Option<&[u8]>, limit: usize, m: &mut Model) -> Result<()> {
    let nv = match leftover {
        Some(t) => int(t),
        None => c.token(true).and_then(int),
    }
    .context(ErrorKind::Corrupt, "OFF vertex count missing")?;
    let nf = c
        .token(true)
        .and_then(int)
        .context(ErrorKind::Corrupt, "OFF face count missing")?;
    ensure!(nv >= 0 && nf >= 0, Corrupt, "OFF counts are negative");
    // The edge count, when there is one, and anything else on the line.
    c.end_line();
    let (nv, nf) = (
        usize::try_from(nv).context(ErrorKind::LimitExceeded, "OFF vertex count is too big")?,
        usize::try_from(nf).context(ErrorKind::LimitExceeded, "OFF face count is too big")?,
    );
    // Each vertex takes at least six bytes ("0 0 0\n"), each face eight.
    m.positions.reserve(nv.min(c.d.len() / 6));
    m.tris.reserve(nf.min(c.d.len() / 8));
    let coords = layout.dim + usize::from(layout.homogeneous);
    let mut vals = [0f32; 2 * MAX_VALUES + 8];
    for _ in 0..nv {
        let mut n = 0;
        // The coordinates, over several lines when they must be.
        while n < coords {
            let Some(t) = c.token(true) else {
                m.short = true;
                break;
            };
            vals[n] = num(t);
            n += 1;
        }
        if m.short {
            break;
        }
        while n < vals.len() {
            let Some(t) = c.token(false) else { break };
            vals[n] = num(t);
            n += 1;
        }
        c.end_line();
        vertex(&vals[..n], layout, m);
    }
    let colored = !m.colors.is_empty();
    if colored {
        m.colors.resize(m.positions.len(), [1.0; 4]);
    }
    let scale = if m.colors.iter().flatten().any(|&v| v > 1.0) {
        1.0 / 255.0
    } else {
        1.0
    };
    for col in &mut m.colors {
        *col = color(col, scale);
    }
    let mut corners: Vec<u32> = Vec::new();
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut local: Vec<[u32; 3]> = Vec::new();
    for _ in 0..nf {
        if m.short {
            break;
        }
        let Some(count) = c.token(true) else {
            m.short = true;
            break;
        };
        let count = int(count).and_then(|n| usize::try_from(n).ok());
        corners.clear();
        let mut ok = count.is_some();
        for _ in 0..count.unwrap_or(0) {
            let Some(t) = c.token(true) else {
                m.short = true;
                break;
            };
            match int(t).and_then(|i| u32::try_from(i).ok()) {
                Some(i) if (i as usize) < m.positions.len() => corners.push(i),
                _ => ok = false,
            }
        }
        if m.short {
            break;
        }
        // The rest of the line: a color, or a color map index.
        let mut col = [0f32; 4];
        let (mut k, mut ints) = (0, true);
        while let Some(t) = c.token(false) {
            if k < 4 {
                col[k] = stream::f32_from(t).unwrap_or(0.0);
                ints &= !t.iter().any(|b| matches!(b, b'.' | b'e' | b'E'));
            }
            k += 1;
        }
        c.end_line();
        let face_color = (k == 3 || k == 4).then(|| {
            if k == 3 {
                col[3] = if ints { 255.0 } else { 1.0 };
            }
            color(&col, if ints { 1.0 / 255.0 } else { 1.0 })
        });
        if ok {
            face(m, &corners, face_color, &mut pts, &mut local);
        } else {
            m.dropped += 1;
        }
        ensure!(
            m.tris.len() <= limit,
            LimitExceeded,
            "too many triangles to convert (more than {limit})"
        );
    }
    Ok(())
}

/// One vertex's values: its position, and what the layout (or a vertex line with more values
/// than it says) adds.
fn vertex(v: &[f32], layout: Layout, m: &mut Model) {
    let dim = layout.dim;
    let mut p = [0f32; 3];
    for (k, x) in p.iter_mut().enumerate().take(dim.min(v.len())) {
        *x = v[k];
    }
    let mut rest = v.get(dim..).unwrap_or_default();
    if layout.homogeneous {
        let w = rest.first().copied().unwrap_or(1.0);
        rest = rest.get(1..).unwrap_or_default();
        p = p.map(|x| x / w);
    }
    m.positions.push(p);
    if layout.normal {
        let n = [0, 1, 2].map(|k| rest.get(k).copied().unwrap_or(0.0));
        m.normals.push(n);
        rest = rest.get(dim..).unwrap_or_default();
    }
    if layout.st && rest.len() >= 2 {
        let (head, st) = rest.split_at(rest.len() - 2);
        m.uvs.push([st[0], 1.0 - st[1]]);
        rest = head;
    } else if layout.st {
        m.uvs.push([0.0; 2]);
    }
    if (layout.color && rest.len() >= 3) || (!layout.color && (rest.len() == 3 || rest.len() == 4)) {
        if m.colors.is_empty() {
            m.colors.resize(m.positions.len() - 1, [1.0; 4]);
        }
        let a = rest.get(3).copied().unwrap_or(f32::NAN);
        m.colors.push([rest[0], rest[1], rest[2], a]);
    } else if !m.colors.is_empty() {
        m.colors.push([1.0; 4]);
    }
}

/// A color as written, `scale` bringing it to 0-1, as linear RGBA. A missing alpha (NaN) is
/// opaque.
fn color(c: &[f32; 4], scale: f32) -> [f32; 4] {
    let unit = |x: f32| {
        if x.is_finite() {
            (x * scale).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let a = if c[3].is_nan() { 1.0 } else { unit(c[3]) };
    [
        srgb_to_linear(unit(c[0])),
        srgb_to_linear(unit(c[1])),
        srgb_to_linear(unit(c[2])),
        a,
    ]
}

/// Adds one face's triangles.
fn face(m: &mut Model, corners: &[u32], color: Option<[f32; 4]>, pts: &mut Vec<[f32; 3]>, local: &mut Vec<[u32; 3]>) {
    let finite = |i: &u32| m.positions[*i as usize].iter().all(|x| x.is_finite());
    if corners.len() < 3 || !corners.iter().all(finite) {
        m.dropped += 1;
        return;
    }
    let before = m.tris.len();
    if corners.len() == 3 {
        m.tris.push([corners[0], corners[1], corners[2]]);
    } else {
        pts.clear();
        pts.extend(corners.iter().map(|&i| m.positions[i as usize]));
        local.clear();
        triangulate(pts, local);
        m.tris.extend(local.iter().map(|t| t.map(|k| corners[k as usize])));
    }
    let id = match color {
        Some(c) => {
            let next = u32_of(m.face_colors.len());
            let id = *m.face_color_of.entry(c.map(f32::to_bits)).or_insert(next);
            if id == next {
                m.face_colors.push(c);
            }
            id
        }
        None => NONE,
    };
    if !m.face_colors.is_empty() {
        m.tri_colors.resize(before, NONE);
        m.tri_colors.resize(m.tris.len(), id);
    }
}

/// The binary body, from `at`: big-endian counts, vertices and faces.
fn binary(d: &[u8], at: usize, mut layout: Layout, limit: usize, m: &mut Model) -> Result<()> {
    let mut r = Be { d, at };
    if layout.dim == 0 {
        let n = r.int().context(ErrorKind::Corrupt, "OFF dimension missing")?;
        ensure!((1..=32).contains(&n), Unsupported, "OFF dimension {n} is out of range");
        layout.dim = usize::try_from(n).context(ErrorKind::Unsupported, "OFF dimension is out of range")?;
    }
    let nv = usize::try_from(r.int().context(ErrorKind::Corrupt, "OFF counts missing")?)
        .context(ErrorKind::Corrupt, "OFF counts are negative")?;
    let nf = usize::try_from(r.int().context(ErrorKind::Corrupt, "OFF counts missing")?)
        .context(ErrorKind::Corrupt, "OFF counts are negative")?;
    r.int().context(ErrorKind::Corrupt, "OFF counts missing")?;
    let floats = layout.dim
        + usize::from(layout.homogeneous)
        + if layout.normal { layout.dim } else { 0 }
        + if layout.color { 4 } else { 0 }
        + if layout.st { 2 } else { 0 };
    let room = (d.len() - r.at.min(d.len())) / (floats * 4);
    if nv > room {
        m.short = true;
    }
    m.positions.reserve(nv.min(room));
    let mut vals = vec![0f32; floats];
    for _ in 0..nv.min(room) {
        for v in &mut vals {
            *v = r.float().unwrap_or(0.0);
        }
        // The binary color is four floats between the normal and the texture coordinates.
        let dim = layout.dim + usize::from(layout.homogeneous);
        let mut p = [0f32; 3];
        for (k, x) in p.iter_mut().enumerate().take(layout.dim) {
            *x = vals[k];
        }
        if layout.homogeneous {
            let w = vals[layout.dim];
            p = p.map(|x| x / w);
        }
        m.positions.push(p);
        let mut k = dim;
        if layout.normal {
            m.normals
                .push([0, 1, 2].map(|j| if j < layout.dim { vals[k + j] } else { 0.0 }));
            k += layout.dim;
        }
        if layout.color {
            m.colors
                .push(color(&[vals[k], vals[k + 1], vals[k + 2], vals[k + 3]], 1.0));
            k += 4;
        }
        if layout.st {
            m.uvs.push([vals[k], 1.0 - vals[k + 1]]);
        }
    }
    let mut corners: Vec<u32> = Vec::new();
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut local: Vec<[u32; 3]> = Vec::new();
    for _ in 0..nf {
        if m.short {
            break;
        }
        let Some(count) = r.int() else {
            m.short = true;
            break;
        };
        let count = usize::try_from(count).unwrap_or(usize::MAX);
        if count > (d.len() - r.at.min(d.len())) / 4 {
            m.short = true;
            break;
        }
        corners.clear();
        let mut ok = true;
        for _ in 0..count {
            match r.int().and_then(|i| u32::try_from(i).ok()) {
                Some(i) if (i as usize) < m.positions.len() => corners.push(i),
                _ => ok = false,
            }
        }
        let Some(nc) = r.int() else {
            m.short = true;
            break;
        };
        let mut col = [0f32; 4];
        for k in 0..usize::try_from(nc).unwrap_or(0).min(16) {
            let v = r.float().unwrap_or(0.0);
            if let Some(c) = col.get_mut(k) {
                *c = v;
            }
        }
        let face_color = matches!(nc, 3 | 4).then(|| {
            if nc == 3 {
                col[3] = 1.0;
            }
            color(&col, 1.0)
        });
        if ok {
            face(m, &corners, face_color, &mut pts, &mut local);
        } else {
            m.dropped += 1;
        }
        ensure!(
            m.tris.len() <= limit,
            LimitExceeded,
            "too many triangles to convert (more than {limit})"
        );
    }
    Ok(())
}

/// The primitive: the vertices the triangles use, split where face colors differ.
fn build(m: Model) -> Prim {
    let Model {
        positions,
        normals,
        colors,
        uvs,
        tris,
        tri_colors,
        face_colors,
        ..
    } = m;
    let n = positions.len();
    let flat: Vec<u32> = tris.into_flattened();
    let normals = if normals.len() == n {
        normals
            .into_iter()
            .map(|v| {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if len > 0.0 && len.is_finite() {
                    v.map(|x| x / len)
                } else {
                    [0.0, 1.0, 0.0]
                }
            })
            .collect()
    } else {
        smooth_normals(&positions, &flat)
    };
    let colored = colors.len() == n || !face_colors.is_empty();
    let textured = uvs.len() == n;
    let mut p = Prim::default();
    let push = |p: &mut Prim, v: usize, face: u32| {
        p.positions.push(positions[v]);
        p.normals.push(normals[v]);
        if textured {
            p.uvs.push(uvs[v]);
        }
        if colored {
            p.colors.push(match face {
                NONE if colors.len() == n => colors[v],
                NONE => GRAY,
                f => face_colors[f as usize],
            });
        }
        u32_of(p.positions.len() - 1)
    };
    p.indices.reserve(flat.len());
    if tri_colors.is_empty() {
        let mut remap = vec![NONE; n];
        for &v in &flat {
            let r = &mut remap[v as usize];
            if *r == NONE {
                *r = push(&mut p, v as usize, NONE);
            }
            p.indices.push(*r);
        }
    } else {
        let mut seen: FastMap<u64, u32> = FastMap::default();
        for (k, &v) in flat.iter().enumerate() {
            let face = tri_colors[k / 3];
            let key = (u64::from(v) << 32) | u64::from(face);
            let r = if let Some(&r) = seen.get(&key) {
                r
            } else {
                let r = push(&mut p, v as usize, face);
                seen.insert(key, r);
                r
            };
            p.indices.push(r);
        }
    }
    p
}

/// A number token; one that isn't reads as NaN, so faces using it drop.
#[inline]
fn num(t: &[u8]) -> f32 {
    stream::f32_from(t).unwrap_or(f32::NAN)
}

/// An integer as written: an optional sign and up to 18 digits.
#[inline]
fn int(w: &[u8]) -> Option<i64> {
    let (neg, d) = match w.first()? {
        b'-' => (true, &w[1..]),
        b'+' => (false, &w[1..]),
        _ => (false, w),
    };
    if d.is_empty() || d.len() > 18 {
        return None;
    }
    let mut n: i64 = 0;
    for &c in d {
        let k = c.wrapping_sub(b'0');
        if k > 9 {
            return None;
        }
        n = n * 10 + i64::from(k);
    }
    Some(if neg { -n } else { n })
}

/// Text read a token at a time, `#` comments skipped, and line ends (LF, CR LF or a lone CR)
/// seen when they matter.
struct Cursor<'a> {
    d: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    /// The next token; with `lines` false, only one before the end of the line.
    #[inline]
    fn token(&mut self, lines: bool) -> Option<&'a [u8]> {
        let d = self.d;
        let mut i = self.at;
        while i < d.len() {
            let b = d[i];
            if b == b'#' {
                while i < d.len() && d[i] != b'\n' && d[i] != b'\r' {
                    i += 1;
                }
            } else if b == b'\n' || b == b'\r' {
                if !lines {
                    self.at = i;
                    return None;
                }
                i += 1;
            } else if b <= b' ' {
                i += 1;
            } else {
                break;
            }
        }
        let start = i;
        while i < d.len() && d[i] > b' ' && d[i] != b'#' {
            i += 1;
        }
        self.at = i;
        (i > start).then(|| &d[start..i])
    }

    /// Moves past the end of the current line.
    #[inline]
    fn end_line(&mut self) {
        let d = self.d;
        let mut i = self.at;
        while i < d.len() && d[i] != b'\n' && d[i] != b'\r' {
            i += 1;
        }
        if i < d.len() {
            if d[i] == b'\r' && d.get(i + 1) == Some(&b'\n') {
                i += 1;
            }
            i += 1;
        }
        self.at = i;
    }
}

/// Big-endian binary read from `at`.
struct Be<'a> {
    d: &'a [u8],
    at: usize,
}

impl Be<'_> {
    fn word(&mut self) -> Option<[u8; 4]> {
        let w = self.d.get(self.at..self.at.checked_add(4)?)?.try_into().ok()?;
        self.at += 4;
        Some(w)
    }

    fn int(&mut self) -> Option<i32> {
        self.word().map(i32::from_be_bytes)
    }

    fn float(&mut self) -> Option<f32> {
        self.word().map(f32::from_be_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use crate::document::{Keep, Pose};
    use crate::mesh::Mesh;

    fn read_with(text: &[u8], max: usize) -> Result<Document<'static>> {
        let keep = Keep {
            max_triangles: max,
            ..Keep::ALL
        };
        super::super::load_bytes(text, "t.off", "off", &Memory::new(), keep).map(Document::into_owned)
    }

    fn mesh(text: &[u8]) -> Mesh {
        read_with(text, usize::MAX)
            .unwrap()
            .flatten(Pose::Rest)
            .unwrap()
            .unwrap()
    }

    const CUBE: &str = "OFF
# a cube
8 6 12
-1 -1 -1
 1 -1 -1
 1  1 -1
-1  1 -1
-1 -1  1
 1 -1  1
 1  1  1
-1  1  1
4 0 3 2 1
4 4 5 6 7
4 0 1 5 4
4 2 3 7 6
4 1 2 6 5
4 3 0 4 7
";

    #[test]
    fn a_cube_of_quads() {
        let doc = read_with(CUBE.as_bytes(), usize::MAX).unwrap();
        assert_eq!(doc.facts["triangles"], 12);
        let m = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(m.prims[0].positions.len(), 8);
        assert_eq!(m.prims[0].normals.len(), 8);
        assert_eq!(m.materials[0].color.map(f32::to_bits), GRAY.map(f32::to_bits));
        // Every triangle faces outward.
        let p = &m.prims[0];
        for t in p.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| p.positions[i as usize]);
            let n = crate::mesh::face_normal(a, b, c);
            let mid = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0);
            assert!(n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2] > 0.0);
        }
    }

    #[test]
    fn counts_on_the_header_line_and_glued_to_it() {
        let one = "OFF 3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
        assert_eq!(mesh(one.as_bytes()).triangles(), 1);
        let glued = "OFF3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
        assert_eq!(mesh(glued.as_bytes()).triangles(), 1);
        let bare = "3 1\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
        assert_eq!(mesh(bare.as_bytes()).triangles(), 1);
    }

    #[test]
    fn line_ends_bom_and_comments() {
        let crlf = format!("\u{feff}{}", CUBE.replace('\n', "\r\n"));
        assert_eq!(mesh(crlf.as_bytes()).triangles(), 12);
        let cr = CUBE.replace('\n', "\r");
        assert_eq!(mesh(cr.as_bytes()).triangles(), 12);
        let odd = CUBE.replace(' ', "\t \x0c").replace("4 0 3 2 1", "4 0 3 2 1 # a face");
        assert_eq!(mesh(odd.as_bytes()).triangles(), 12);
    }

    #[test]
    fn face_colors_in_integers_and_floats() {
        let text = "OFF\n4 2 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n3 0 1 2 255 0 0\n3 0 2 3 0.0 1.0 0.0 0.5\n";
        let m = mesh(text.as_bytes());
        let p = &m.prims[0];
        assert_eq!(p.positions.len(), 6, "vertices split where colors differ");
        assert_eq!(p.colors[0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(p.colors[p.indices[3] as usize], [0.0, 1.0, 0.0, 0.5]);
        assert_eq!(m.materials[0].color, [1.0; 4]);
    }

    #[test]
    fn vertex_colors_normals_and_texture_coordinates() {
        let text = "STCNOFF\n3 1 0\n0 0 0 0 0 1 255 0 0 255 0 0\n1 0 0 0 0 1 0 255 0 255 1 0\n0 1 0 0 0 1 0 0 255 255 0 1\n3 0 1 2\n";
        let p = &mesh(text.as_bytes()).prims[0];
        assert_eq!(p.normals[0], [0.0, 0.0, 1.0]);
        assert_eq!(p.colors[1], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(p.uvs[2], [0.0, 0.0]);
        assert_eq!(p.uvs[0], [0.0, 1.0]);
        // Colors on a plain OFF's vertex lines, as floats.
        let plain = "OFF\n3 1 0\n0 0 0 1 0 0\n1 0 0 0 1 0\n0 1 0 0 0 1\n3 0 1 2\n";
        assert_eq!(mesh(plain.as_bytes()).prims[0].colors[0], [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn homogeneous_and_other_dimensions() {
        let four = "4OFF\n3 1 0\n0 0 0 1\n2 0 0 2\n0 4 0 2\n3 0 1 2\n";
        let p = &mesh(four.as_bytes()).prims[0];
        assert_eq!(p.positions[1], [1.0, 0.0, 0.0]);
        assert_eq!(p.positions[2], [0.0, 2.0, 0.0]);
        let two = "nOFF\n2\n3 1 0\n0 0\n1 0\n0 1\n3 0 1 2\n";
        assert_eq!(mesh(two.as_bytes()).prims[0].positions[2], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn concave_polygons_stay_inside() {
        // An L of six corners: four triangles, all inside.
        let text = "OFF\n6 1 0\n0 0 0\n2 0 0\n2 1 0\n1 1 0\n1 2 0\n0 2 0\n6 0 1 2 3 4 5\n";
        let m = mesh(text.as_bytes());
        assert_eq!(m.triangles(), 4);
    }

    #[test]
    fn bad_faces_drop_and_short_files_draw_what_they_have() {
        let text = "OFF\n4 4 0\n0 0 0\n1 0 0\n0 1 0\nnan 0 0\n3 0 1 2\n3 0 1 9\n3 0 1 3\n2 0 1\n";
        let doc = read_with(text.as_bytes(), usize::MAX).unwrap();
        assert_eq!(doc.facts["triangles"], 1);
        assert_eq!(doc.json["asset"]["extras"]["source"]["droppedFaces"], 3);
        let short = &CUBE[..CUBE.len() - 11];
        let doc = read_with(short.as_bytes(), usize::MAX).unwrap();
        assert_eq!(doc.facts["triangles"], 10);
        assert_eq!(doc.json["asset"]["extras"]["source"]["truncated"], true);
        assert!(read_with(b"OFF\n0 0 0\n", usize::MAX).is_err());
        assert!(read_with(b"PLY\n", usize::MAX).is_err());
    }

    fn binary_cube(colors: bool) -> Vec<u8> {
        let mut b = b"OFF BINARY\n".to_vec();
        let int = |b: &mut Vec<u8>, n: i32| b.extend(n.to_be_bytes());
        int(&mut b, 4);
        int(&mut b, 2);
        int(&mut b, 0);
        for p in [[0f32, 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]] {
            for x in p {
                b.extend(x.to_be_bytes());
            }
        }
        for f in [[0, 1, 2], [0, 2, 3]] {
            int(&mut b, 3);
            for i in f {
                int(&mut b, i);
            }
            if colors {
                int(&mut b, 3);
                for x in [0f32, 0., 1.] {
                    b.extend(x.to_be_bytes());
                }
            } else {
                int(&mut b, 0);
            }
        }
        b
    }

    #[test]
    fn binary_off() {
        let doc = read_with(&binary_cube(false), usize::MAX).unwrap();
        assert_eq!(doc.facts["triangles"], 2);
        assert_eq!(doc.json["asset"]["extras"]["source"]["encoding"], "binary");
        let m = mesh(&binary_cube(true));
        assert_eq!(m.prims[0].colors[0], [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn the_triangle_limit_refuses_bigger_files() {
        for text in [CUBE.as_bytes().to_vec(), binary_cube(false)] {
            let n = read_with(&text, usize::MAX).unwrap().facts["triangles"]
                .as_u64()
                .unwrap();
            let n = usize::try_from(n).unwrap();
            let err = read_with(&text, n - 1).unwrap_err();
            assert!(format!("{err:#}").contains("triangles"));
            assert_eq!(mesh(&text).triangles(), n);
        }
    }

    /// Reading any prefix of a file, or the file with a byte changed, never panics, and keeps
    /// to the triangle limit.
    fn survives(sample: &[u8]) {
        let check = |bytes: &[u8]| {
            if let Ok(doc) = read_with(bytes, 64) {
                if let Ok(Some(m)) = doc.flatten(Pose::Rest) {
                    assert!(m.triangles() <= 64);
                }
                let _ = doc.to_glb();
            }
        };
        for n in 0..=sample.len() {
            check(&sample[..n]);
        }
        let mut bytes = sample.to_vec();
        for i in 0..bytes.len() {
            for flip in [0x01, 0x20, 0x80, 0xFF] {
                bytes[i] ^= flip;
                check(&bytes);
                bytes[i] ^= flip;
            }
        }
    }

    #[test]
    fn damaged_files_never_panic() {
        survives(CUBE.as_bytes());
        survives(b"STCN4OFF 3 1 0\r0 0 0 1 0 0 1 9 0 0 1 0 0\r1 0 0 1 0 0 1 0 9 0 1 1 0\r0 1 0 1 0 0 1 0 0 9 1 0 1\r3 0 1 2 1 1 1\r");
        survives(b"nOFF 4 3 1 0\n0 0 0 0\n1 0 0 0\n0 1 0 0\n5 0 1 2 0 1 0.5 0.5 0.5\n");
        survives(&binary_cube(true));
    }

    #[test]
    fn units_and_up_axis_are_unknown() {
        let doc = read_with(CUBE.as_bytes(), usize::MAX).unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("OFF"));
        assert_eq!(info.meters_per_unit, crate::source::Fact::Unknown);
        assert_eq!(info.up_axis, crate::source::Fact::Unknown);
        assert_eq!(info.converted, crate::source::Converted::default());
        assert_eq!(doc.json["asset"]["extras"]["source"]["encoding"], "ASCII");
    }
}
