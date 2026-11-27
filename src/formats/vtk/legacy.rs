//! Legacy `.vtk` files: a version line, a title, `ASCII` or `BINARY`, a dataset and its point and cell data.
//!
//! Keywords are lines of text in either encoding, read in any case. Numbers follow as text, free of line ends,
//! or as big-endian binary starting right after the keyword's line. Version 5 writes cells as an `OFFSETS` and a
//! `CONNECTIVITY` array where older files have one list of counted cells; either is told by what follows the
//! cell keyword, not by the version. `METADATA` blocks are skipped. Reading stops at a keyword it doesn't know,
//! which the source notes name, keeping what came before.

use serde_json::json;

use super::stream;
use super::{Array, Cells, Cursor, Kind, Model, Piece, Points, Role, Ty, check_dims, count, field_note};
use super::{floats_of, ids_of, packed};
use crate::{Context as _, ErrorKind, Result};

/// Which data the arrays being read belong to, and how many points or cells that is.
#[derive(Clone, Copy)]
enum Mode {
    Dataset,
    Points(usize),
    Cells(usize),
}

struct Reader<'a> {
    c: Cursor<'a>,
    binary: bool,
}

impl<'a> Reader<'a> {
    /// The type a keyword line names, for `n` values of it. A binary `long` is eight bytes, or four as Windows
    /// writes it: whichever ends where a line does.
    fn ty(&self, token: Option<&[u8]>, n: usize) -> Result<Ty> {
        let name = token.unwrap_or_default().to_ascii_lowercase();
        Ok(match name.as_slice() {
            b"bit" => Ty::Bit,
            // The sized names are meshio's for the widths VTK has no keyword for; VTK itself writes two of them.
            b"unsigned_char" | b"vtktypeuint8" => Ty::U8,
            b"char" | b"signed_char" | b"vtktypeint8" => Ty::I8,
            b"unsigned_short" | b"vtktypeuint16" => Ty::U16,
            b"short" | b"vtktypeint16" => Ty::I16,
            b"unsigned_int" | b"vtktypeuint32" => Ty::U32,
            b"int" | b"vtkidtype" | b"vtktypeint32" => Ty::I32,
            b"vtktypeint64" | b"long_long" => Ty::I64,
            b"vtktypeuint64" | b"unsigned_long_long" => Ty::U64,
            b"float" | b"vtktypefloat32" => Ty::F32,
            b"double" | b"vtktypefloat64" => Ty::F64,
            b"long" | b"unsigned_long" => {
                let unsigned = name[0] == b'u';
                let ends = |size: usize| {
                    let end = n.checked_mul(size).and_then(|len| self.c.at.checked_add(len));
                    end.is_some_and(|end| end == self.c.d.len() || self.c.d.get(end) == Some(&b'\n'))
                };
                match (self.binary && !ends(8) && ends(4), unsigned) {
                    (false, false) => Ty::I64,
                    (false, true) => Ty::U64,
                    (true, false) => Ty::I32,
                    (true, true) => Ty::U32,
                }
            }
            b"string" | b"utf8_string" | b"variant" => {
                bail!(Unsupported, "VTK {} arrays aren't read", crate::shown(&name))
            }
            _ => bail!(Corrupt, "unknown VTK data type {}", crate::shown(&name)),
        })
    }

    /// The bytes of `n` binary values of `ty`, checked against what's left.
    fn bytes(&mut self, ty: Ty, n: usize, what: &str) -> Result<&'a [u8]> {
        let len = if ty == Ty::Bit {
            Some(n.div_ceil(8))
        } else {
            n.checked_mul(ty.size())
        };
        let b = len
            .and_then(|len| self.c.at.checked_add(len))
            .and_then(|end| self.c.d.get(self.c.at..end))
            .with_context(ErrorKind::Truncated, || {
                format!("VTK file is cut short: {what} claims {n} values, more than it holds")
            })?;
        self.c.at += b.len();
        Ok(b)
    }

    fn floats(&mut self, ty: Ty, n: usize, what: &str) -> Result<Vec<f32>> {
        if !self.binary {
            return self.c.floats(n, what);
        }
        let b = self.bytes(ty, n, what)?;
        Ok(match ty {
            Ty::Bit => floats_of(&packed(b, n), Ty::U8, true),
            _ => floats_of(b, ty, true),
        })
    }

    fn ids(&mut self, ty: Ty, n: usize, what: &str) -> Result<Vec<u32>> {
        if !self.binary {
            return self.c.ids(n, what);
        }
        let b = self.bytes(ty, n, what)?;
        Ok(match ty {
            Ty::Bit => ids_of(&packed(b, n), Ty::U8, true),
            _ => ids_of(b, ty, true),
        })
    }

    /// Skips a `METADATA` block: to the blank line that ends it.
    fn skip_block(&mut self) {
        while let Some(line) = stream::next_line(self.c.d, &mut self.c.at) {
            if stream::trim(line).is_empty() {
                break;
            }
        }
    }

    /// A cell list of `a` and `b` as its keyword line gives them: `a` cells in a list of `b` numbers, or, when
    /// `OFFSETS` follows, `a` offsets and `b` nodes.
    fn cells(&mut self, kind: Kind, a: usize, b: usize, what: &str) -> Result<Cells> {
        let back = self.c.at;
        let mut words = stream::tokens(self.c.line().unwrap_or_default());
        if words.next().is_some_and(|w| w.eq_ignore_ascii_case(b"OFFSETS")) {
            let ty = self.ty(words.next(), a)?;
            let offsets = self.ids(ty, a, what)?;
            let mut words = stream::tokens(self.c.line().unwrap_or_default());
            ensure!(
                words.next().is_some_and(|w| w.eq_ignore_ascii_case(b"CONNECTIVITY")),
                Corrupt,
                "VTK {what} has OFFSETS without CONNECTIVITY"
            );
            let ty = self.ty(words.next(), b)?;
            let conn = self.ids(ty, b, what)?;
            return Ok(Cells {
                kind,
                conn,
                offsets,
                types: Vec::new(),
            });
        }
        self.c.at = back;
        let list = self.ids(Ty::I32, b, what)?;
        Cells::from_counted(kind, a, &list)
    }
}

/// A name as written: VTK writes a space and other awkward characters as `%` and two hex digits.
fn name(token: Option<&[u8]>) -> String {
    let t = token.unwrap_or_default();
    let mut out = Vec::with_capacity(t.len());
    let mut i = 0;
    while i < t.len() {
        let hex = |k: usize| t.get(k).and_then(|&c| char::from(c).to_digit(16));
        if t[i] == b'%'
            && let (Some(hi), Some(lo)) = (hex(i + 1), hex(i + 2))
        {
            out.push(u8::try_from(hi * 16 + lo).unwrap_or(b'_'));
            i += 3;
        } else {
            out.push(t[i]);
            i += 1;
        }
    }
    crate::bytes::text(&out).into_owned()
}

/// Three numbers from a keyword's line.
fn three(mut words: stream::Tokens<'_>, what: &str) -> Result<[f64; 3]> {
    let mut out = [0.0; 3];
    for v in &mut out {
        *v = words
            .next()
            .and_then(stream::f64_from)
            .filter(|v| v.is_finite())
            .with_context(ErrorKind::Corrupt, || format!("VTK {what} needs three numbers"))?;
    }
    Ok(out)
}

/// What a legacy file's first line starts with, before its version.
const START: &[u8] = b"# vtk DataFile Version";

pub(super) fn read(data: &[u8]) -> Result<Model> {
    let mut c = Cursor::new(data, "VTK");
    let first = c.line().unwrap_or_default();
    let version = first
        .split_at_checked(START.len())
        .filter(|(start, _)| start.eq_ignore_ascii_case(START))
        .map(|(_, version)| crate::bytes::text(stream::trim(version)).into_owned())
        .context(ErrorKind::UnknownFormat, "not a VTK file")?;
    // The title is the very next line, blank or not.
    let title = stream::next_line(c.d, &mut c.at).unwrap_or_default();
    let title = crate::bytes::text(stream::trim(title)).into_owned();
    let encoding = c.line().unwrap_or_default();
    let binary = encoding.eq_ignore_ascii_case(b"BINARY");
    ensure!(
        binary || encoding.eq_ignore_ascii_case(b"ASCII"),
        Corrupt,
        "VTK file says neither ASCII nor BINARY but {}",
        crate::shown(encoding)
    );
    let mut r = Reader { c, binary };
    let mut model = Model::default();
    let mut piece = Piece::default();
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut dataset = String::new();
    let mut dims = None;
    let mut origin = [0.0; 3];
    let mut spacing = [1.0; 3];
    let mut axes: [Vec<f32>; 3] = Default::default();
    let mut types: Option<Vec<u8>> = None;
    let mut fields = serde_json::Map::new();
    let mut mode = Mode::Dataset;
    while let Some(line) = r.c.line() {
        let mut words = stream::tokens(line);
        let Some(word) = words.next() else { continue };
        let key = word.to_ascii_uppercase();
        // How many points or cells the data arrays cover.
        let now = mode;
        let tuples = move |what: &str| match now {
            Mode::Points(n) | Mode::Cells(n) => Ok(n),
            Mode::Dataset => Err(err!(Corrupt, "VTK {what} comes before POINT_DATA or CELL_DATA")),
        };
        let mut array = None;
        match key.as_slice() {
            b"DATASET" => {
                dataset = crate::bytes::text(words.next().unwrap_or_default()).to_ascii_uppercase();
            }
            b"DIMENSIONS" => {
                let mut d = [0u32; 3];
                for v in &mut d {
                    *v = words
                        .next()
                        .and_then(stream::int::<u32>)
                        .context(ErrorKind::Corrupt, "VTK DIMENSIONS needs three counts")?;
                }
                check_dims(d)?;
                dims = Some(d);
            }
            b"ORIGIN" => origin = three(words, "ORIGIN")?,
            b"SPACING" | b"ASPECT_RATIO" => spacing = three(words, "SPACING")?,
            b"POINTS" => {
                let n = count(words.next(), "POINTS")?;
                let ty = r.ty(words.next(), n.saturating_mul(3))?;
                let n3 = n
                    .checked_mul(3)
                    .context(ErrorKind::Corrupt, "VTK POINTS count is too big")?;
                let values = r.floats(ty, n3, "POINTS")?;
                points = values.as_chunks::<3>().0.to_vec();
            }
            b"X_COORDINATES" | b"Y_COORDINATES" | b"Z_COORDINATES" => {
                let n = count(words.next(), "coordinates")?;
                let ty = r.ty(words.next(), n)?;
                axes[usize::from(key[0] - b'X')] = r.floats(ty, n, "coordinates")?;
            }
            b"VERTICES" | b"LINES" | b"POLYGONS" | b"TRIANGLE_STRIPS" | b"CELLS" => {
                let kind = match key[0] {
                    b'V' => Kind::Verts,
                    b'L' => Kind::Lines,
                    b'P' => Kind::Polys,
                    b'T' => Kind::Strips,
                    _ => Kind::Mixed,
                };
                let a = count(words.next(), "cell")?;
                let b = count(words.next(), "cell")?;
                piece.cells.push(r.cells(kind, a, b, "cell list")?);
            }
            b"CELL_TYPES" => {
                let n = count(words.next(), "CELL_TYPES")?;
                let list = r.ids(Ty::I32, n, "CELL_TYPES")?;
                // A type past the ones VTK has is one that isn't drawn.
                types = Some(list.iter().map(|&t| u8::try_from(t).unwrap_or(u8::MAX)).collect());
            }
            b"POINT_DATA" => mode = Mode::Points(count(words.next(), "POINT_DATA")?),
            b"CELL_DATA" => mode = Mode::Cells(count(words.next(), "CELL_DATA")?),
            b"FIELD" => {
                let arrays = count(words.nth(1), "FIELD")?;
                for _ in 0..arrays {
                    // An array's metadata may sit between it and the next.
                    let head = loop {
                        let Some(head) = r.c.line() else {
                            bail!(Truncated, "VTK file is cut short: a FIELD ends early");
                        };
                        if !head.eq_ignore_ascii_case(b"METADATA") {
                            break head;
                        }
                        r.skip_block();
                    };
                    let mut words = stream::tokens(head);
                    let label = name(words.next());
                    let comps = count(words.next(), "field array")?;
                    let rows = count(words.next(), "field array")?;
                    let n = comps
                        .checked_mul(rows)
                        .context(ErrorKind::Corrupt, "VTK field array is too big")?;
                    let ty = r.ty(words.next(), n)?;
                    let values = r.floats(ty, n, "field array")?;
                    match mode {
                        Mode::Dataset => {
                            if let Some(note) = field_note(&values) {
                                fields.insert(label, note);
                            }
                        }
                        Mode::Points(_) | Mode::Cells(_) => {
                            let mut a = Array::new(&label, comps, Role::Other, values);
                            if ty == Ty::U8 {
                                a.bytes_are_colors();
                            }
                            match mode {
                                Mode::Points(_) => piece.point_data.push(a),
                                _ => piece.cell_data.push(a),
                            }
                        }
                    }
                }
            }
            b"SCALARS" => {
                let label = name(words.next());
                let kind = words.next();
                let comps = match words.next() {
                    Some(t) => count(Some(t), "SCALARS component")?.max(1),
                    None => 1,
                };
                let n = tuples("SCALARS")?
                    .checked_mul(comps)
                    .context(ErrorKind::Corrupt, "VTK SCALARS are too big")?;
                // The table's name, on a line of its own before the values, when there is one.
                let back = r.c.at;
                let mut next = stream::tokens(r.c.line().unwrap_or_default());
                let table = if next.next().is_some_and(|w| w.eq_ignore_ascii_case(b"LOOKUP_TABLE")) {
                    Some(name(next.next()))
                } else {
                    r.c.at = back;
                    None
                };
                let ty = r.ty(kind, n)?;
                let mut a = Array::new(&label, comps, Role::Other, r.floats(ty, n, "SCALARS")?);
                a.table = table;
                if ty == Ty::U8 {
                    a.bytes_are_colors();
                }
                array = Some(a);
            }
            b"COLOR_SCALARS" => {
                let label = name(words.next());
                let comps = count(words.next(), "COLOR_SCALARS value")?;
                let n = tuples("COLOR_SCALARS")?
                    .checked_mul(comps)
                    .context(ErrorKind::Corrupt, "VTK COLOR_SCALARS are too big")?;
                // Floats from 0 to 1 as text, bytes in binary.
                let mut a = Array::new(&label, comps, Role::Other, r.floats(Ty::U8, n, "COLOR_SCALARS")?);
                if matches!(comps, 3 | 4) {
                    a.role = Role::Colors;
                    a.scale = if r.binary { 1.0 / 255.0 } else { 1.0 };
                }
                array = Some(a);
            }
            b"LOOKUP_TABLE" => {
                let label = name(words.next());
                let size = count(words.next(), "LOOKUP_TABLE")?;
                let n = size
                    .checked_mul(4)
                    .context(ErrorKind::Corrupt, "VTK LOOKUP_TABLE is too big")?;
                let values = r.floats(Ty::U8, n, "LOOKUP_TABLE")?;
                let scale = if r.binary { 1.0 / 255.0 } else { 1.0 };
                let colors = values.as_chunks::<4>().0.iter().map(|c| c.map(|v| v * scale)).collect();
                model.tables.push((label, colors));
            }
            b"VECTORS"
            | b"NORMALS"
            | b"TENSORS"
            | b"TENSORS6"
            | b"GLOBAL_IDS"
            | b"PEDIGREE_IDS"
            | b"EDGE_FLAGS"
            | b"TEXTURE_COORDINATES" => {
                let label = name(words.next());
                let (comps, role) = match key.as_slice() {
                    b"NORMALS" => (3, Role::Normals),
                    b"VECTORS" => (3, Role::Other),
                    b"TENSORS" => (9, Role::Other),
                    b"TENSORS6" => (6, Role::Other),
                    b"TEXTURE_COORDINATES" => {
                        let dim = count(words.next(), "TEXTURE_COORDINATES dimension")?;
                        (dim, if dim == 2 { Role::TCoords } else { Role::Other })
                    }
                    _ => (1, Role::Other),
                };
                let what = crate::bytes::text(&key).into_owned();
                let n = tuples(&what)?
                    .checked_mul(comps)
                    .with_context(ErrorKind::Corrupt, || format!("VTK {what} are too big"))?;
                let ty = r.ty(words.next(), n)?;
                array = Some(Array::new(&label, comps, role, r.floats(ty, n, &what)?));
            }
            b"METADATA" => r.skip_block(),
            _ => {
                let word = crate::bytes::text(&key[..key.len().min(32)]).into_owned();
                model.source.insert("unread".into(), json!(word));
                break;
            }
        }
        if let Some(a) = array {
            match mode {
                Mode::Points(_) => piece.point_data.push(a),
                _ => piece.cell_data.push(a),
            }
        }
    }
    match dataset.as_str() {
        "POLYDATA" | "UNSTRUCTURED_GRID" => {
            piece.points = Points::List(points);
            let types = types.unwrap_or_default();
            for cells in piece.cells.iter_mut().filter(|c| c.kind == Kind::Mixed) {
                cells.types.clone_from(&types);
            }
        }
        "STRUCTURED_GRID" | "STRUCTURED_POINTS" | "RECTILINEAR_GRID" => {
            let dims = dims.with_context(ErrorKind::Corrupt, || format!("VTK {dataset} has no DIMENSIONS"))?;
            piece.cells.clear();
            piece.dims = Some(dims);
            let total: usize = dims.iter().map(|&n| n as usize).product();
            piece.points = match dataset.as_str() {
                "STRUCTURED_GRID" => {
                    ensure!(
                        points.len() == total,
                        Corrupt,
                        "VTK STRUCTURED_GRID of {dims:?} has {} points",
                        points.len()
                    );
                    Points::List(points)
                }
                "STRUCTURED_POINTS" => Points::Image { dims, origin, spacing },
                _ => {
                    ensure!(
                        (0..3).all(|k| axes[k].len() == dims[k] as usize),
                        Corrupt,
                        "VTK RECTILINEAR_GRID of {dims:?} has other counts of coordinates"
                    );
                    Points::Rect { dims, axes }
                }
            };
        }
        "" => bail!(Empty, "VTK file has no DATASET"),
        other => bail!(Unsupported, "VTK {other} datasets aren't read"),
    }
    model
        .source
        .insert("encoding".into(), json!(if binary { "binary" } else { "ascii" }));
    model.source.insert("dataset".into(), json!(dataset));
    if !fields.is_empty() {
        model
            .source
            .insert("fieldData".into(), serde_json::Value::Object(fields));
    }
    model.info.version = Some(version).filter(|v| !v.is_empty());
    model.info.title = Some(title).filter(|t| !t.is_empty());
    model.pieces.push(piece);
    Ok(model)
}
