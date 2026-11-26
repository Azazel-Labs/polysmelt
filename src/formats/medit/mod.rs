//! Medit meshes, the INRIA format libMeshb defines: `.mesh` as text and `.meshb` as binary, versions 1 to 4.
//!
//! A text file is keywords and numbers free of line ends, with `#` comments: `MeshVersionFormatted`,
//! `Dimension` (2 or 3), then `Vertices`, the element keywords and the rest, each a count and its lines. A
//! binary file starts with the integer one, which tells its byte order, and its version, which sets the widths:
//! floats are four bytes in version 1 and eight after, file positions eight bytes from version 3, and indices
//! eight bytes in version 4. Each keyword is its code and where the next one starts, so the ones not read here
//! are stepped over; in text they're passed over up to the next word.
//!
//! `Vertices`, `Edges`, `Triangles`, `Quadrilaterals`, `Tetrahedra`, `Hexahedra`, `Prisms` and `Pyramids` are
//! read, and their higher-order forms (`TrianglesP2`, `HexahedraQ2` and the like) by their corner nodes, which
//! come first. Edges draw as lines, triangles and quadrilaterals as triangles, and volume elements as the faces
//! only one of them has, wound to face outward. Vertices no drawn element uses are left out, and a file with
//! vertices alone draws them as points. A two-dimensional mesh lies in the plane z = 0.
//!
//! Each element ends with its reference. The elements of one reference and dimension are a mesh and a node of
//! their own, named `Curve 2`, `Surface 3` or `Volume 1`, with the reference and dimension in the mesh's
//! `extras`; a volume draws its own boundary. Elements of reference 0 are one mesh with no name. A file with
//! more than 1024 such groups is one mesh, the references in a `_REF` attribute. The vertices' references are
//! a `_REF` attribute when any isn't 0.
//!
//! `Normals` with `NormalAtVertices` become `NORMAL` when every vertex has one, and otherwise a `_NORMAL`
//! custom attribute, zero where a vertex has none; `Tangents` with `TangentAtVertices` are `_TANGENT` the same
//! way. Without the keyword that assigns them, as many normals or tangents as vertices go one to each.
//! Corners, ridges, required entities and solution fields aren't read.
//!
//! Medit names no units and no up axis, so both are unknown, and geometry is kept as written.

#[cfg(test)]
mod tests;

use serde_json::{Map, json};

use super::Input;
use super::cells::{self, Array, BAD, Cursor, Elements, Group, Model, Role, int_of, real_of, tag_of, word_of};
use super::stream;
use crate::document::Document;
use crate::mesh::FastMap;
use crate::num::{u32_of, usize_of};
use crate::{Context as _, ErrorKind, Result};

/// What a keyword holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Version,
    Dimension,
    Vertices,
    /// Elements: how many nodes each lists, how many of them are corners, the VTK cell type the corners draw
    /// as, and the dimension.
    Elements(usize, usize, u8, u32),
    Normals,
    NormalAtVertices,
    Tangents,
    TangentAtVertices,
    End,
}

/// The keywords read: each one's name in a text file, its code in a binary one, and what it holds.
const KEYWORDS: &[(&str, u32, Kind)] = &[
    ("MeshVersionFormatted", 1, Kind::Version),
    ("Dimension", 3, Kind::Dimension),
    ("Vertices", 4, Kind::Vertices),
    ("Edges", 5, Kind::Elements(2, 2, 3, 1)),
    ("Triangles", 6, Kind::Elements(3, 3, 5, 2)),
    ("Quadrilaterals", 7, Kind::Elements(4, 4, 9, 2)),
    ("Tetrahedra", 8, Kind::Elements(4, 4, 10, 3)),
    ("Prisms", 9, Kind::Elements(6, 6, 13, 3)),
    ("Hexahedra", 10, Kind::Elements(8, 8, 12, 3)),
    // As some writers spell it.
    ("Hexaedra", 10, Kind::Elements(8, 8, 12, 3)),
    ("Pyramids", 49, Kind::Elements(5, 5, 14, 3)),
    ("TrianglesP2", 24, Kind::Elements(6, 3, 5, 2)),
    ("EdgesP2", 25, Kind::Elements(3, 2, 3, 1)),
    ("QuadrilateralsQ2", 27, Kind::Elements(9, 4, 9, 2)),
    ("TetrahedraP2", 30, Kind::Elements(10, 4, 10, 3)),
    ("HexahedraQ2", 33, Kind::Elements(27, 8, 12, 3)),
    ("PrismsP2", 86, Kind::Elements(18, 6, 13, 3)),
    ("PyramidsP2", 87, Kind::Elements(14, 5, 14, 3)),
    ("QuadrilateralsQ3", 88, Kind::Elements(16, 4, 9, 2)),
    ("QuadrilateralsQ4", 89, Kind::Elements(25, 4, 9, 2)),
    ("TrianglesP3", 90, Kind::Elements(10, 3, 5, 2)),
    ("TrianglesP4", 91, Kind::Elements(15, 3, 5, 2)),
    ("EdgesP3", 92, Kind::Elements(4, 2, 3, 1)),
    ("EdgesP4", 93, Kind::Elements(5, 2, 3, 1)),
    ("TetrahedraP3", 96, Kind::Elements(20, 4, 10, 3)),
    ("TetrahedraP4", 97, Kind::Elements(35, 4, 10, 3)),
    ("HexahedraQ3", 98, Kind::Elements(64, 8, 12, 3)),
    ("HexahedraQ4", 99, Kind::Elements(125, 8, 12, 3)),
    ("PyramidsP3", 100, Kind::Elements(30, 5, 14, 3)),
    ("PyramidsP4", 101, Kind::Elements(55, 5, 14, 3)),
    ("PrismsP3", 102, Kind::Elements(40, 6, 13, 3)),
    ("PrismsP4", 103, Kind::Elements(75, 6, 13, 3)),
    ("Normals", 60, Kind::Normals),
    ("NormalAtVertices", 20, Kind::NormalAtVertices),
    ("Tangents", 59, Kind::Tangents),
    ("TangentAtVertices", 61, Kind::TangentAtVertices),
    ("End", 54, Kind::End),
];

/// What a group of each dimension is called.
const KINDS: [&str; 4] = ["Point", "Curve", "Surface", "Volume"];

/// A file as read.
struct Mesh {
    version: i64,
    dim: usize,
    el: Elements,
    /// Each vertex's reference.
    refs: Vec<f32>,
    /// Each group's place in the elements' groups, by dimension and reference.
    groups: FastMap<(u32, i64), u32>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 3]>,
    /// Pairs of a vertex and the normal or tangent it has, both from 1.
    normal_at: Vec<[u32; 2]>,
    tangent_at: Vec<[u32; 2]>,
    /// Keywords passed over, by name or code.
    skipped: Vec<String>,
}

impl Mesh {
    /// The elements' group for reference `reference` of dimension `dim`: [`BAD`] for reference 0.
    #[inline]
    fn group(&mut self, dim: u32, reference: i64) -> u32 {
        if reference == 0 {
            return BAD;
        }
        let next = u32_of(self.el.groups.len());
        let at = *self.groups.entry((dim, reference)).or_insert(next);
        if at == next {
            self.el.groups.push(Group {
                name: format!("{} {reference}", KINDS[dim.min(3) as usize]),
                extras: json!({ "reference": reference, "dimension": dim }),
                tag: reference,
            });
        }
        at
    }

    fn skip(&mut self, name: String) {
        if self.skipped.len() < 32 && !self.skipped.contains(&name) {
            self.skipped.push(name);
        }
    }

    /// Checks the mesh stays small enough to index.
    fn room(&self, vertices: usize) -> Result<()> {
        ensure!(
            self.el.points.len().saturating_add(vertices) < BAD as usize
                && self.el.conn.len() < BAD as usize
                && self.el.len() < BAD as usize,
            LimitExceeded,
            "Medit mesh is too big"
        );
        Ok(())
    }
}

pub(crate) fn load(input: Input<'_>) -> Result<Document> {
    let data = input.read()?;
    let mut mesh = Mesh {
        version: 1,
        dim: 0,
        el: Elements::new("ref"),
        refs: Vec::new(),
        groups: FastMap::default(),
        normals: Vec::new(),
        tangents: Vec::new(),
        normal_at: Vec::new(),
        tangent_at: Vec::new(),
        skipped: Vec::new(),
    };
    let mut source = Map::new();
    let binary = match data.first_chunk::<4>().map(|b| (int_of(b, false), int_of(b, true))) {
        Some((1, _)) => Some(false),
        Some((_, 1)) => Some(true),
        _ => None,
    };
    if let Some(big) = binary {
        self::binary(&data, big, &mut mesh)?;
        source.insert("encoding".into(), json!("binary"));
        if big {
            source.insert("byteOrder".into(), json!("big-endian"));
        }
    } else {
        let data = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&data);
        ensure!(data.iter().any(|&b| !stream::space(b)), Empty, "Medit file is empty");
        text(data, &mut mesh)?;
        source.insert("encoding".into(), json!("ASCII"));
    }
    source.insert("dimension".into(), json!(mesh.dim));
    source.insert("vertices".into(), json!(mesh.el.points.len()));
    source.insert("elements".into(), json!(mesh.el.len()));
    if !mesh.skipped.is_empty() {
        source.insert("skippedKeywords".into(), json!(mesh.skipped));
    }
    let np = mesh.el.points.len();
    // References from 1 become indices; a 0 wraps around to none.
    for i in &mut mesh.el.conn {
        *i = i.wrapping_sub(1);
    }
    if mesh.refs.len() == np && mesh.refs.iter().any(|&r| r != 0.0) {
        let refs = std::mem::take(&mut mesh.refs);
        mesh.el.point_data.push(Array::new("ref", 1, Role::Other, refs));
    }
    for (name, values, at) in [
        ("normal", &mesh.normals, &mesh.normal_at),
        ("tangent", &mesh.tangents, &mesh.tangent_at),
    ] {
        if let Some((all, vectors)) = at_vertices(values, at, np) {
            let role = if all && name == "normal" {
                Role::Normals
            } else {
                Role::Other
            };
            mesh.el.point_data.push(Array::new(name, 3, role, vectors));
        }
    }
    let mut model = Model {
        source,
        ..Model::default()
    };
    model.info.version = Some(mesh.version.to_string());
    model.pieces = mesh.el.pieces();
    cells::build(model, "Medit", input.limit)
}

/// The vector each of `np` vertices has: `values[at[k][1]]` for vertex `at[k][0]`, both from 1, or with no
/// `at`, one each in order. Returns whether every vertex has one; `None` when there are none to give.
fn at_vertices(values: &[[f32; 3]], at: &[[u32; 2]], np: usize) -> Option<(bool, Vec<f32>)> {
    if values.is_empty() || np == 0 {
        return None;
    }
    if at.is_empty() {
        return (values.len() == np).then(|| (true, values.as_flattened().to_vec()));
    }
    let mut out = vec![0.0; np * 3];
    let mut given = vec![false; np];
    for &[vertex, value] in at {
        let (v, k) = (vertex.wrapping_sub(1) as usize, value.wrapping_sub(1) as usize);
        if let (Some(slot), Some(value)) = (out.get_mut(v * 3..v * 3 + 3), values.get(k)) {
            slot.copy_from_slice(value);
            given[v] = true;
        }
    }
    given.contains(&true).then(|| (given.iter().all(|&g| g), out))
}

/// A text file.
fn text(data: &[u8], mesh: &mut Mesh) -> Result<()> {
    let mut c = Cursor::new(data, "Medit");
    let mut first = true;
    while let Some(word) = keyword(&mut c) {
        let known = KEYWORDS
            .iter()
            .find(|(name, ..)| name.as_bytes().eq_ignore_ascii_case(word));
        if first {
            ensure!(
                matches!(known, Some((_, _, Kind::Version))),
                UnknownFormat,
                "not a Medit mesh: it starts with {}",
                crate::shown(word)
            );
            first = false;
        }
        let Some(&(_, _, kind)) = known else {
            mesh.skip(String::from_utf8_lossy(word).into_owned());
            continue;
        };
        if matches!(kind, Kind::Version | Kind::Dimension) {
            let t = c.token().unwrap_or_default();
            let v = stream::int::<i64>(t).with_context(ErrorKind::Corrupt, || {
                format!(
                    "Medit {}: {} is not a whole number",
                    crate::shown(word),
                    crate::shown(t)
                )
            })?;
            if kind == Kind::Version {
                mesh.version = v;
            } else {
                ensure!(
                    v == 2 || v == 3,
                    Unsupported,
                    "Medit meshes of dimension {v} aren't read"
                );
                mesh.dim = usize_of(v.unsigned_abs());
            }
            continue;
        }
        if kind == Kind::End {
            break;
        }
        ensure!(
            mesh.dim > 0,
            Corrupt,
            "Medit {} comes before the Dimension",
            crate::shown(word)
        );
        let t = c.token().unwrap_or_default();
        let n = stream::int::<usize>(t).with_context(ErrorKind::Corrupt, || {
            format!("Medit {}: {} is not a count", crate::shown(word), crate::shown(t))
        })?;
        // Each value takes a character and a separator.
        let room = |per: usize| n.checked_mul(per).is_some_and(|values| values <= c.room());
        let short = || err!(Truncated, "Medit file is cut short in its {}", crate::shown(word));
        let dim = mesh.dim;
        match kind {
            Kind::Vertices => {
                ensure!(
                    room(dim + 1),
                    Truncated,
                    "Medit file is cut short: it claims {n} vertices, more than it holds"
                );
                mesh.room(n)?;
                mesh.el.points.reserve(n);
                mesh.refs.reserve(n);
                for _ in 0..n {
                    let mut p = [0.0; 3];
                    for v in &mut p[..dim] {
                        *v = c.float("vertex")?.ok_or_else(short)?;
                    }
                    mesh.el.points.push(p);
                    mesh.refs.push(c.float("vertex")?.ok_or_else(short)?);
                }
            }
            Kind::Elements(nodes, corners, vtk, group_dim) => {
                ensure!(
                    room(nodes + 1),
                    Truncated,
                    "Medit file is cut short: it claims {n} elements, more than it holds"
                );
                mesh.el.types.reserve(n);
                mesh.el.conn.reserve(n * corners);
                for _ in 0..n {
                    for k in 0..nodes {
                        let i = c.id("element")?.ok_or_else(short)?;
                        if k < corners {
                            mesh.el.conn.push(i);
                        }
                    }
                    let reference = reference(&mut c)?.ok_or_else(short)?;
                    let group = mesh.group(group_dim, reference);
                    mesh.el.close(vtk, group);
                }
                mesh.room(0)?;
            }
            Kind::Normals | Kind::Tangents => {
                ensure!(
                    room(dim),
                    Truncated,
                    "Medit file is cut short: it claims {n} vectors, more than it holds"
                );
                let into = if kind == Kind::Normals {
                    &mut mesh.normals
                } else {
                    &mut mesh.tangents
                };
                into.reserve(n);
                for _ in 0..n {
                    let mut v = [0.0; 3];
                    for x in &mut v[..dim] {
                        *x = c.float("vector")?.ok_or_else(short)?;
                    }
                    into.push(v);
                }
            }
            Kind::NormalAtVertices | Kind::TangentAtVertices => {
                ensure!(
                    room(2),
                    Truncated,
                    "Medit file is cut short: it claims {n} pairs, more than it holds"
                );
                let into = if kind == Kind::NormalAtVertices {
                    &mut mesh.normal_at
                } else {
                    &mut mesh.tangent_at
                };
                into.reserve(n);
                for _ in 0..n {
                    let vertex = c.id("pair")?.ok_or_else(short)?;
                    into.push([vertex, c.id("pair")?.ok_or_else(short)?]);
                }
            }
            Kind::Version | Kind::Dimension | Kind::End => {}
        }
    }
    ensure!(!first, UnknownFormat, "not a Medit mesh: it has no keywords");
    Ok(())
}

/// The next keyword: a word that starts with a letter, past the comments and whatever numbers a keyword not
/// read here left behind.
fn keyword<'a>(c: &mut Cursor<'a>) -> Option<&'a [u8]> {
    loop {
        c.skip_space();
        if c.d.get(c.at) == Some(&b'#') {
            c.skip_line();
            continue;
        }
        let word = c.token()?;
        if word[0].is_ascii_alphabetic() {
            return Some(word);
        }
    }
}

/// An element's reference: a whole number, or one written as a float.
#[inline]
fn reference(c: &mut Cursor<'_>) -> Result<Option<i64>> {
    let back = c.at;
    c.skip_space();
    let (v, n) = stream::few_digits(&c.d[c.at..]);
    if n > 0 && n < 18 && c.d.get(c.at + n).is_none_or(|&b| stream::space(b)) {
        c.at += n;
        return Ok(i64::try_from(v).ok());
    }
    c.at = back;
    let Some(t) = c.token() else {
        return Ok(None);
    };
    if let Some(v) = stream::int::<i64>(t) {
        return Ok(Some(v));
    }
    let Some(v) = stream::f64_lenient(t).filter(|v| v.is_finite()) else {
        bail!(Corrupt, "Medit reference {} is not a number", crate::shown(t));
    };
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a reference written as a float is a small whole number"
    )]
    Ok(Some(v as i64))
}

/// A binary file: the one that told its byte order, the version, then keywords.
fn binary(d: &[u8], big: bool, mesh: &mut Mesh) -> Result<()> {
    let take = |at: usize, n: usize| -> Result<&[u8]> {
        let Some(b) = at.checked_add(n).and_then(|end| d.get(at..end)) else {
            bail!(Truncated, "Medit file is cut short");
        };
        Ok(b)
    };
    let version = int_of(take(4, 4)?, big);
    ensure!(
        (1..=4).contains(&version),
        Unsupported,
        "Medit binary files of version {version} aren't read"
    );
    mesh.version = i64::from(version);
    // The widths of a float, a file position and an index.
    let real = if version == 1 { 4 } else { 8 };
    let position = if version >= 3 { 8 } else { 4 };
    let index = if version == 4 { 8 } else { 4 };
    let float = |b: &[u8]| -> f32 {
        if real == 4 {
            f32::from_bits(int_of(b, big).cast_unsigned())
        } else {
            real_of(b, big)
        }
    };
    // References are signed.
    let signed = |b: &[u8]| -> i64 {
        if index == 4 {
            i64::from(int_of(b, big))
        } else {
            word_of(b, big).cast_signed()
        }
    };
    let mut at = 8;
    while at < d.len() {
        let code = int_of(take(at, 4)?, big);
        if code == 54 {
            break;
        }
        let next = usize_of(word_of(take(at + 4, position)?, big));
        let mut body = at + 4 + position;
        let known = KEYWORDS.iter().find(|&&(_, c, _)| i64::from(c) == i64::from(code));
        // Where the next keyword starts: the position given when it leads on past this one's data, which keeps
        // any byte from being read twice.
        let onward = |end: usize| if next >= end && next <= d.len() { next } else { end };
        let Some(&(name, _, kind)) = known else {
            mesh.skip(format!("keyword {code}"));
            at = if next >= body && next <= d.len() { next } else { d.len() };
            continue;
        };
        if kind == Kind::Dimension {
            let v = int_of(take(body, 4)?, big);
            ensure!(
                v == 2 || v == 3,
                Unsupported,
                "Medit meshes of dimension {v} aren't read"
            );
            mesh.dim = usize_of(u64::from(v.unsigned_abs()));
            at = onward(body + 4);
            continue;
        }
        if kind == Kind::Version {
            at = onward(body + 4);
            continue;
        }
        ensure!(mesh.dim > 0, Corrupt, "Medit {name} comes before the Dimension");
        let dim = mesh.dim;
        let n = usize_of(word_of(take(body, index)?, big));
        body += index;
        let stride = match kind {
            Kind::Vertices => dim * real + index,
            Kind::Elements(nodes, ..) => (nodes + 1) * index,
            Kind::Normals | Kind::Tangents => dim * real,
            _ => 2 * index,
        };
        ensure!(
            n <= (d.len() - body) / stride,
            Truncated,
            "Medit file is cut short: {name} claims {n} lines, more than it holds"
        );
        let records = d[body..body + n * stride].chunks_exact(stride);
        match kind {
            Kind::Vertices => {
                mesh.room(n)?;
                mesh.el.points.reserve(n);
                mesh.refs.reserve(n);
                for record in records {
                    let mut p = [0.0; 3];
                    for (v, b) in p.iter_mut().zip(record.chunks_exact(real)).take(dim) {
                        *v = float(b);
                    }
                    mesh.el.points.push(p);
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "a reference past 2^24 rounds, as other data does"
                    )]
                    mesh.refs.push(signed(&record[dim * real..]) as f32);
                }
            }
            Kind::Elements(nodes, corners, vtk, group_dim) => {
                mesh.el.types.reserve(n);
                mesh.el.conn.reserve(n * corners);
                for record in records {
                    mesh.el.conn.extend(
                        record
                            .chunks_exact(index)
                            .take(corners)
                            .map(|b| tag_of(word_of(b, big))),
                    );
                    let group = mesh.group(group_dim, signed(&record[nodes * index..]));
                    mesh.el.close(vtk, group);
                }
                mesh.room(0)?;
            }
            Kind::Normals | Kind::Tangents => {
                let into = if kind == Kind::Normals {
                    &mut mesh.normals
                } else {
                    &mut mesh.tangents
                };
                into.reserve(n);
                for record in records {
                    let mut v = [0.0; 3];
                    for (x, b) in v.iter_mut().zip(record.chunks_exact(real)) {
                        *x = float(b);
                    }
                    into.push(v);
                }
            }
            _ => {
                let into = if kind == Kind::NormalAtVertices {
                    &mut mesh.normal_at
                } else {
                    &mut mesh.tangent_at
                };
                into.reserve(n);
                for record in records {
                    into.push([
                        tag_of(word_of(&record[..index], big)),
                        tag_of(word_of(&record[index..], big)),
                    ]);
                }
            }
        }
        at = onward(body + n * stride);
    }
    ensure!(mesh.dim > 0, Corrupt, "Medit file gives no Dimension");
    Ok(())
}
