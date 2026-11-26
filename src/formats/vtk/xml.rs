//! XML VTK files: `.vtp` (`PolyData`) and `.vtu` (`UnstructuredGrid`).
//!
//! A `<VTKFile>` holds one dataset of `<Piece>` elements, each with its points, its cells and their data as
//! `<DataArray>` elements. An array is text (`format="ascii"`), base64 between its tags (`binary`), or a block of
//! the `<AppendedData>` at the file's end (`appended`), which is base64 or raw bytes after an underscore. A
//! binary block starts with its length in bytes, as the `header_type` says (`UInt32` unless it's `UInt64`), in
//! the file's `byte_order`. With a compressor it starts instead with a block count, the block size, the last
//! block's size and each block's compressed size, and the blocks follow, each a zlib stream. In base64 the
//! header and the data are encoded apart or as one: reading the text as a stream, a padded group ending a run,
//! takes both.
//!
//! Raw appended data isn't XML, so the file is cut at `<AppendedData` before the rest is parsed.

use std::borrow::Cow;

use flate2::{Decompress, FlushDecompress, Status};
use serde_json::json;

use super::stream;
use super::{Array, Cells, Cursor, Kind, Model, Piece, Points, Role, Ty, field_note, floats_of, ids_of};
use crate::formats::xml::{El, Xml, text};
use crate::num::usize_of;
use crate::{Context as _, ErrorKind, Result};

/// The most one array inflates to: 2 GiB, past any real array.
const MAX_INFLATED: usize = 1 << 31;

/// The most a deflate stream inflates: 258 bytes from a two-bit code.
const MAX_INFLATE_RATIO: usize = 1032;

/// Marks a byte that's no base64 digit: padding, a space, or anything else.
const PAD: u8 = 0x80;
const SPACE: u8 = 0x81;
const WRONG: u8 = 0xFF;

/// Each byte's base64 value, or one of the marks above.
const DIGITS: [u8; 256] = {
    let mut t = [WRONG; 256];
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut i = 0u8;
    while i < 64 {
        t[alphabet[i as usize] as usize] = i;
        i += 1;
    }
    t[b'=' as usize] = PAD;
    t[b' ' as usize] = SPACE;
    t[b'\n' as usize] = SPACE;
    t[b'\r' as usize] = SPACE;
    t[b'\t' as usize] = SPACE;
    t
};

/// Where a binary block's bytes come from: as they are, or out of base64 text.
struct Source<'a> {
    d: &'a [u8],
    at: usize,
    base64: bool,
    /// Bytes decoded from a group and not yet taken.
    held: [u8; 2],
    holding: usize,
}

impl<'a> Source<'a> {
    fn new(d: &'a [u8], base64: bool) -> Self {
        Self {
            d,
            at: 0,
            base64,
            held: [0; 2],
            holding: 0,
        }
    }

    /// The most bytes that can still be taken.
    fn left(&self) -> usize {
        let rest = self.d.len() - self.at;
        if self.base64 {
            (rest / 4 * 3 + 3).saturating_add(self.holding)
        } else {
            rest
        }
    }

    /// The next `n` bytes.
    fn take(&mut self, n: usize) -> Result<Cow<'a, [u8]>> {
        ensure!(
            n <= self.left(),
            Truncated,
            "VTK file is cut short: a data block claims {n} bytes, more than it holds"
        );
        if !self.base64 {
            let b = &self.d[self.at..self.at + n];
            self.at += n;
            return Ok(Cow::Borrowed(b));
        }
        let mut out = vec![0u8; n];
        let mut done = 0;
        while done < n && self.holding > 0 {
            out[done] = self.held[0];
            self.held[0] = self.held[1];
            self.holding -= 1;
            done += 1;
        }
        while done < n {
            // Whole groups of four digits, three bytes each.
            let groups = ((n - done) / 3).min((self.d.len() - self.at) / 4);
            let text = &self.d[self.at..self.at + groups * 4];
            let mut fast = 0;
            for (s, o) in text.chunks_exact(4).zip(out[done..].chunks_exact_mut(3)) {
                let q = [s[0], s[1], s[2], s[3]].map(|c| DIGITS[usize::from(c)]);
                if (q[0] | q[1] | q[2] | q[3]) & 0x80 != 0 {
                    break;
                }
                let v = u32::from(q[0]) << 18 | u32::from(q[1]) << 12 | u32::from(q[2]) << 6 | u32::from(q[3]);
                let [_, a, b, c] = v.to_be_bytes();
                o.copy_from_slice(&[a, b, c]);
                fast += 1;
            }
            self.at += fast * 4;
            done += fast * 3;
            if done == n {
                break;
            }
            // One group the slow way: spaces skipped, padding ending it.
            let mut q = [0u8; 4];
            let mut digits = 0;
            while digits < 4 {
                let Some(&c) = self.d.get(self.at) else { break };
                self.at += 1;
                match DIGITS[usize::from(c)] {
                    SPACE => {}
                    PAD if digits >= 2 => {
                        // The rest of the group is padding too, across a line break if need be.
                        while self
                            .d
                            .get(self.at)
                            .is_some_and(|&c| matches!(DIGITS[usize::from(c)], PAD | SPACE))
                        {
                            self.at += 1;
                        }
                        break;
                    }
                    PAD | WRONG => bail!(Corrupt, "VTK base64 data has a stray {}", crate::shown(&[c])),
                    v => {
                        q[digits] = v;
                        digits += 1;
                    }
                }
            }
            ensure!(digits >= 2, Truncated, "VTK file is cut short: base64 data ends early");
            let v = u32::from(q[0]) << 18 | u32::from(q[1]) << 12 | u32::from(q[2]) << 6 | u32::from(q[3]);
            let [_, a, b, c] = v.to_be_bytes();
            for &byte in &[a, b, c][..digits - 1] {
                if done < n {
                    out[done] = byte;
                    done += 1;
                } else {
                    self.held[self.holding] = byte;
                    self.holding += 1;
                }
            }
        }
        Ok(Cow::Owned(out))
    }
}

/// What a file says of how its binary blocks are written.
struct Reader<'a> {
    big: bool,
    /// Whether block headers are 64-bit.
    wide: bool,
    compressor: Option<String>,
    /// The appended data, from just past its underscore.
    appended: &'a [u8],
    appended_base64: bool,
    /// The first array's format, for the source notes.
    encoding: Option<String>,
}

impl Reader<'_> {
    /// One number of a block's header.
    fn word(&self, src: &mut Source<'_>) -> Result<usize> {
        let b = src.take(if self.wide { 8 } else { 4 })?;
        let v = match (self.wide, self.big) {
            (true, true) => u64::from_be_bytes(b[..8].try_into().unwrap_or_default()),
            (true, false) => u64::from_le_bytes(b[..8].try_into().unwrap_or_default()),
            (false, true) => u64::from(u32::from_be_bytes(b[..4].try_into().unwrap_or_default())),
            (false, false) => u64::from(u32::from_le_bytes(b[..4].try_into().unwrap_or_default())),
        };
        Ok(usize_of(v))
    }

    /// A binary block's bytes, inflated when the file is compressed.
    fn block<'s>(&self, mut src: Source<'s>) -> Result<Cow<'s, [u8]>> {
        let Some(compressor) = &self.compressor else {
            let n = self.word(&mut src)?;
            return src.take(n);
        };
        ensure!(
            compressor == "vtkZLibDataCompressor",
            Unsupported,
            "VTK data compressed with {compressor} isn't read: only vtkZLibDataCompressor is"
        );
        let blocks = self.word(&mut src)?;
        let size = self.word(&mut src)?;
        let last = self.word(&mut src)?;
        ensure!(
            blocks <= src.left() / if self.wide { 8 } else { 4 },
            Truncated,
            "VTK file is cut short: a data block claims {blocks} compressed blocks, more than it holds"
        );
        if blocks == 0 {
            return Ok(Cow::Borrowed(&[]));
        }
        let mut sizes = Vec::with_capacity(blocks);
        let mut packed_len = 0usize;
        for _ in 0..blocks {
            let n = self.word(&mut src)?;
            packed_len = packed_len.saturating_add(n);
            sizes.push(n);
        }
        let total = (blocks - 1)
            .checked_mul(size)
            .and_then(|n| n.checked_add(if last > 0 { last } else { size }))
            .filter(|&n| n <= MAX_INFLATED)
            .context(ErrorKind::LimitExceeded, "VTK compressed data inflates past 2 GiB")?;
        let packed = src.take(packed_len)?;
        ensure!(
            total <= packed.len().saturating_mul(MAX_INFLATE_RATIO),
            Corrupt,
            "VTK compressed data claims to inflate further than zlib can"
        );
        // Room is made a block at a time, each only once the ones before it have inflated, so a header that
        // claims more than its data holds costs no more than the first block that falls short.
        let mut out: Vec<u8> = Vec::new();
        let mut z = Decompress::new(true);
        let (mut from, mut to) = (0, 0);
        for (k, &len) in sizes.iter().enumerate() {
            let want = if k + 1 == blocks && last > 0 { last } else { size };
            let input = &packed[from..from + len];
            ensure!(
                want <= len.saturating_mul(MAX_INFLATE_RATIO),
                Corrupt,
                "VTK compressed data block {k} claims to inflate further than zlib can"
            );
            out.try_reserve(want)
                .ok()
                .context(ErrorKind::LimitExceeded, "not enough memory for VTK compressed data")?;
            out.resize(to + want, 0);
            let room = &mut out[to..];
            z.reset(true);
            let mut status = z.decompress(input, room, FlushDecompress::Finish).ok();
            let filled = |z: &Decompress| usize::try_from(z.total_out()).ok() == Some(want);
            if status != Some(Status::StreamEnd) && status.is_some() && filled(&z) {
                // The block is full; what's left of the stream must end it without another byte.
                let used = usize_of(z.total_in()).min(len);
                status = z.decompress(&input[used..], &mut [0; 1], FlushDecompress::Finish).ok();
            }
            ensure!(
                status == Some(Status::StreamEnd) && filled(&z),
                Corrupt,
                "VTK compressed data block {k} doesn't inflate to the {want} bytes it should"
            );
            from += len;
            to += want;
        }
        Ok(Cow::Owned(out))
    }

    /// A `<DataArray>`'s values: read from its text by `ascii`, or from its bytes by `binary`. `None` for a
    /// type that holds no numbers (strings).
    fn values<T>(
        &mut self,
        el: El<'_, '_>,
        ascii: impl FnOnce(&mut Cursor<'_>) -> Result<Vec<T>>,
        binary: impl FnOnce(&[u8], Ty, bool) -> Vec<T>,
    ) -> Result<Option<(Ty, Vec<T>)>> {
        let kind = el.attr("type").unwrap_or_default();
        let ty = match &*kind {
            "Int8" => Ty::I8,
            "UInt8" => Ty::U8,
            "Int16" => Ty::I16,
            "UInt16" => Ty::U16,
            "Int32" => Ty::I32,
            "UInt32" => Ty::U32,
            "Int64" => Ty::I64,
            "UInt64" => Ty::U64,
            "Float32" => Ty::F32,
            "Float64" => Ty::F64,
            "String" | "Bit" => return Ok(None),
            other => bail!(Corrupt, "unknown VTK data type {}", crate::shown(other.as_bytes())),
        };
        let format = el
            .attr("format")
            .map_or_else(|| "ascii".to_owned(), |f| f.to_ascii_lowercase());
        if self.encoding.is_none() {
            self.encoding = Some(match (format.as_str(), self.appended_base64) {
                ("appended", false) => "appended raw".to_owned(),
                ("appended", true) => "appended base64".to_owned(),
                _ => format.clone(),
            });
        }
        // The values come before any element an array holds (VTK's information keys).
        let raw = el.raw();
        let inline = &raw.as_bytes()[..raw.find('<').unwrap_or(raw.len())];
        let src = match format.as_str() {
            "ascii" => return Ok(Some((ty, ascii(&mut Cursor::new(inline, "VTK"))?))),
            "binary" => {
                let inline = stream::trim(inline);
                if inline.is_empty() {
                    return Ok(Some((ty, Vec::new())));
                }
                Source::new(inline, true)
            }
            "appended" => {
                let offset = el.attr("offset").and_then(|o| stream::whole::<usize>(o.as_bytes()));
                let rest = offset.and_then(|o| self.appended.get(o..)).context(
                    ErrorKind::Truncated,
                    "VTK file is cut short: its appended data ends before an array's offset",
                )?;
                Source::new(rest, self.appended_base64)
            }
            other => bail!(Corrupt, "unknown VTK array format {}", crate::shown(other.as_bytes())),
        };
        let bytes = self.block(src)?;
        Ok(Some((ty, binary(&bytes, ty, self.big))))
    }

    fn floats(&mut self, el: El<'_, '_>, expected: usize, what: &str) -> Result<Option<(Ty, Vec<f32>)>> {
        self.values(el, |c| c.all_floats(expected, what), floats_of)
    }

    fn ids(&mut self, el: El<'_, '_>, expected: usize, what: &str) -> Result<Vec<u32>> {
        let values = self.values(el, |c| c.all_ids(expected, what), ids_of)?;
        Ok(values.map(|(_, v)| v).unwrap_or_default())
    }

    /// The cells under `el`: their `offsets` (each cell's end), `connectivity` and, for a grid, `types`.
    fn cells(&mut self, el: El<'_, '_>, kind: Kind) -> Result<Cells> {
        let find = |name: &str| el.named("DataArray").find(|a| a.attr("Name").as_deref() == Some(name));
        let ends = match find("offsets") {
            Some(a) => self.ids(a, 0, "offsets")?,
            None => Vec::new(),
        };
        let nodes = ends.last().map_or(0, |&n| n as usize);
        let conn = match find("connectivity") {
            Some(a) => self.ids(a, nodes, "connectivity")?,
            None => Vec::new(),
        };
        let types = match find("types").filter(|_| kind == Kind::Mixed) {
            Some(a) => self.ids(a, ends.len(), "types")?,
            None => Vec::new(),
        };
        let mut offsets = Vec::with_capacity(ends.len() + 1);
        if !ends.is_empty() {
            offsets.push(0);
            offsets.extend(ends);
        }
        Ok(Cells {
            kind,
            conn,
            offsets,
            // A type past the ones VTK has is one that isn't drawn.
            types: types.iter().map(|&t| u8::try_from(t).unwrap_or(u8::MAX)).collect(),
        })
    }

    /// The arrays of a `<PointData>` or `<CellData>`, which names the ones that are normals and texture
    /// coordinates.
    fn data(&mut self, el: Option<El<'_, '_>>, into: &mut Vec<Array>) -> Result<()> {
        let Some(el) = el else { return Ok(()) };
        for a in el.named("DataArray") {
            let label = a.attr("Name").unwrap_or_default();
            let comps = a
                .attr("NumberOfComponents")
                .and_then(|n| stream::whole::<usize>(n.as_bytes()))
                .unwrap_or(1)
                .max(1);
            let Some((ty, values)) = self.floats(a, 0, &label)? else {
                continue;
            };
            let named = |key: &str, usual: &[&str]| {
                el.attr(key).is_some_and(|n| n == label) || usual.iter().any(|u| label.eq_ignore_ascii_case(u))
            };
            let role = if comps == 3 && named("Normals", &["Normals"]) {
                Role::Normals
            } else if comps == 2 && named("TCoords", &["TCoords", "TextureCoordinates", "Texture Coordinates"]) {
                Role::TCoords
            } else {
                Role::Other
            };
            let mut array = Array::new(&label, comps, role, values);
            if ty == Ty::U8 {
                array.bytes_are_colors();
            }
            into.push(array);
        }
        Ok(())
    }

    fn piece(&mut self, el: El<'_, '_>, poly: bool) -> Result<Piece> {
        let mut piece = Piece::default();
        let claimed = el
            .attr("NumberOfPoints")
            .and_then(|n| stream::whole::<usize>(n.as_bytes()))
            .context(ErrorKind::Corrupt, "VTK piece doesn't say how many points it has")?;
        if claimed == 0 {
            return Ok(piece);
        }
        let array = el
            .child("Points")
            .and_then(|p| p.child("DataArray"))
            .context(ErrorKind::Corrupt, "VTK piece has no points")?;
        let values = self.floats(array, claimed.saturating_mul(3), "points")?;
        let (_, values) = values.context(ErrorKind::Corrupt, "VTK points aren't numbers")?;
        let points = values.as_chunks::<3>().0;
        ensure!(
            points.len() >= claimed,
            Truncated,
            "VTK file is cut short: a piece claims {claimed} points and holds {}",
            points.len()
        );
        piece.points = Points::List(points[..claimed].to_vec());
        drop(values);
        if poly {
            let lists = [
                ("Verts", Kind::Verts),
                ("Lines", Kind::Lines),
                ("Strips", Kind::Strips),
                ("Polys", Kind::Polys),
            ];
            for (tag, kind) in lists {
                if let Some(c) = el.child(tag) {
                    piece.cells.push(self.cells(c, kind)?);
                }
            }
        } else if let Some(c) = el.child("Cells") {
            piece.cells.push(self.cells(c, Kind::Mixed)?);
        }
        self.data(el.child("PointData"), &mut piece.point_data)?;
        self.data(el.child("CellData"), &mut piece.cell_data)?;
        Ok(piece)
    }
}

/// Where `pat` first starts in `d`.
fn find(d: &[u8], pat: &[u8]) -> Option<usize> {
    let first = pat[0];
    let mut from = 0;
    while let Some(p) = d[from..].iter().position(|&b| b == first) {
        if d[from + p..].starts_with(pat) {
            return Some(from + p);
        }
        from += p + 1;
    }
    None
}

pub(super) fn read(data: &[u8]) -> Result<Model> {
    let cut = find(data, b"<AppendedData");
    let mut head = text(&data[..cut.unwrap_or(data.len())]);
    let mut appended: &[u8] = &[];
    let mut appended_base64 = true;
    if let Some(at) = cut {
        let tag = data[at..]
            .iter()
            .position(|&b| b == b'>')
            .map(|end| &data[at..=at + end])
            .context(ErrorKind::Truncated, "VTK file is cut short in its appended data tag")?;
        appended_base64 = find(tag, b"raw").is_none();
        let body = &data[at + tag.len()..];
        let start = body.iter().position(|&b| !stream::space(b)).unwrap_or(body.len());
        // The data starts after an underscore; without one there is none.
        if body.get(start) == Some(&b'_') {
            appended = &body[start + 1..];
        }
        head.to_mut().push_str("</VTKFile>");
    }
    let doc = Xml::parse(&head)?;
    let root = doc.root();
    ensure!(
        root.name() == "VTKFile",
        UnknownFormat,
        "not a VTK file: the root is <{}>",
        root.name()
    );
    let kind = root.attr("type").unwrap_or_default().into_owned();
    let poly = kind == "PolyData";
    ensure!(
        !kind.starts_with('P') || poly,
        Unsupported,
        "parallel VTK {kind} files, which name other files, aren't read"
    );
    ensure!(
        poly || kind == "UnstructuredGrid",
        Unsupported,
        "VTK XML {kind} files aren't read: only PolyData and UnstructuredGrid are"
    );
    let dataset = root
        .child(&kind)
        .with_context(ErrorKind::Corrupt, || format!("VTK file has no <{kind}>"))?;
    let mut r = Reader {
        big: root
            .attr("byte_order")
            .is_some_and(|o| o.eq_ignore_ascii_case("BigEndian")),
        wide: root
            .attr("header_type")
            .is_some_and(|h| h.eq_ignore_ascii_case("UInt64")),
        compressor: root.attr("compressor").map(Cow::into_owned).filter(|c| !c.is_empty()),
        appended,
        appended_base64,
        encoding: None,
    };
    let mut model = Model::default();
    for piece in dataset.named("Piece") {
        model.pieces.push(r.piece(piece, poly)?);
    }
    // A few values of field data (a time, say) are notes; one that doesn't read isn't worth failing for.
    let mut fields = serde_json::Map::new();
    for a in dataset
        .child("FieldData")
        .into_iter()
        .flat_map(|f| f.named("DataArray"))
    {
        if let Ok(Some((_, values))) = r.floats(a, 0, "field data")
            && let (Some(name), Some(note)) = (a.attr("Name"), field_note(&values))
        {
            fields.insert(name.into_owned(), note);
        }
    }
    if !fields.is_empty() {
        model
            .source
            .insert("fieldData".into(), serde_json::Value::Object(fields));
    }
    model.source.insert("dataset".into(), json!(kind));
    if let Some(encoding) = r.encoding {
        model.source.insert("encoding".into(), json!(encoding));
    }
    model.source.insert(
        "byteOrder".into(),
        json!(if r.big { "BigEndian" } else { "LittleEndian" }),
    );
    if let Some(c) = r.compressor {
        model.source.insert("compressor".into(), json!(c));
    }
    model.info.version = root.attr("version").map(Cow::into_owned);
    Ok(model)
}
