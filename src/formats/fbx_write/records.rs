//! Binary FBX records: a tree of named nodes, each with typed attributes and children, sized up front and written in
//! one pass. A node's attributes are encoded into its own byte buffer as they're added (arrays straight from their
//! slices), so writing is a walk that copies buffers out. Big arrays are deflated once the tree is built, across
//! threads, and kept so when that makes them smaller.
//!
//! The layout is FBX 7.4's: each record starts with the file offset of its end, its attribute count and their byte
//! length (32 bits each; 64 in 7.5, chosen when the file outgrows 4 GiB), then its name, attributes and children, and a
//! null record closes a list of children. The file ends with a null record and the footer.

use std::io::Write;

use flate2::{Compress, Compression, FlushCompress, Status};

use crate::{ErrorKind, Result};

/// The start of every binary FBX file, before its version.
const MAGIC: &[u8; 23] = b"Kaydara FBX Binary  \0\x1a\0";

/// The footer's first 16 bytes, which go with the file id and creation time written in the header.
const FOOT_ID: [u8; 16] = [
    0xfa, 0xbc, 0xab, 0x09, 0xd0, 0xc8, 0xd4, 0x66, 0xb1, 0x76, 0xfb, 0x83, 0x1c, 0xf7, 0x26, 0x7e,
];

/// The footer's last 16 bytes.
const FOOT_MAGIC: [u8; 16] = [
    0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e, 0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b,
];

/// Arrays this many bytes and longer are deflated (and kept so when that makes them smaller).
const DEFLATE_FROM: usize = 128;

/// The deflate level: fixed, so the same document always writes the same bytes, and fast, since big meshes are
/// mostly arrays.
const LEVEL: u32 = 1;

/// Below this many bytes of arrays to deflate in all, a tree is deflated on the calling thread alone.
const PARALLEL_FROM: usize = 1 << 18;

/// A record: its name, encoded attributes and children.
#[derive(Clone, Debug, Default)]
pub(crate) struct Node {
    name: &'static str,
    count: u32,
    props: Vec<u8>,
    children: Vec<Node>,
    /// Closed with a null record even without children, as the format writes some records.
    closed: bool,
    /// Where in `props` the arrays still to deflate start: each at its encoding and stored length.
    raw: Vec<usize>,
}

/// A number an FBX array holds, with its type code.
pub(crate) trait Elem: bytemuck::Pod {
    const CODE: u8;
    /// The number as little-endian bytes in memory.
    #[cfg_attr(target_endian = "little", allow(dead_code, reason = "only big-endian targets swap"))]
    fn le(self) -> Self;
}

macro_rules! elem {
    ($t:ty, $code:literal) => {
        impl Elem for $t {
            const CODE: u8 = $code;
            fn le(self) -> Self {
                <$t>::from_ne_bytes(self.to_le_bytes())
            }
        }
    };
}

elem!(f64, b'd');
elem!(f32, b'f');
elem!(i32, b'i');
elem!(i64, b'l');

impl Node {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            ..Self::default()
        }
    }

    /// Closes this record with a null record even when it has no children.
    #[must_use]
    pub(crate) fn closed(mut self) -> Self {
        self.closed = true;
        self
    }

    fn head(&mut self, code: u8) {
        self.count += 1;
        self.props.push(code);
    }

    pub(crate) fn bool(mut self, v: bool) -> Self {
        self.head(b'C');
        self.props.push(u8::from(v));
        self
    }

    pub(crate) fn i32(mut self, v: i32) -> Self {
        self.head(b'I');
        self.props.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn i64(mut self, v: i64) -> Self {
        self.head(b'L');
        self.props.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn f64(mut self, v: f64) -> Self {
        self.head(b'D');
        self.props.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn str(self, s: &str) -> Self {
        self.bytes_as(b'S', s.as_bytes())
    }

    pub(crate) fn raw(self, b: &[u8]) -> Self {
        self.bytes_as(b'R', b)
    }

    fn bytes_as(mut self, code: u8, b: &[u8]) -> Self {
        self.head(code);
        self.props.extend_from_slice(&len32(b.len()).to_le_bytes());
        self.props.extend_from_slice(b);
        self
    }

    /// An array attribute.
    pub(crate) fn arr<T: Elem>(mut self, v: &[T]) -> Self {
        self.head(T::CODE);
        self.props.extend_from_slice(&len32(v.len()).to_le_bytes());
        #[cfg(target_endian = "little")]
        let bytes: &[u8] = bytemuck::cast_slice(v);
        #[cfg(not(target_endian = "little"))]
        let owned: Vec<T> = v.iter().map(|x| x.le()).collect();
        #[cfg(not(target_endian = "little"))]
        let bytes: &[u8] = bytemuck::cast_slice(&owned);
        self.array_bytes(bytes);
        self
    }

    /// An array's encoding, its stored length and its bytes, stored as they are: [`deflate`] packs the big ones.
    fn array_bytes(&mut self, bytes: &[u8]) {
        if bytes.len() >= DEFLATE_FROM {
            self.raw.push(self.props.len());
        }
        self.props.extend_from_slice(&0u32.to_le_bytes());
        self.props.extend_from_slice(&len32(bytes.len()).to_le_bytes());
        self.props.extend_from_slice(bytes);
    }

    /// Adds a child.
    pub(crate) fn push(&mut self, child: Self) {
        self.children.push(child);
    }

    /// With a child added.
    #[must_use]
    pub(crate) fn with(mut self, child: Self) -> Self {
        self.children.push(child);
        self
    }

    /// With a child holding one attribute, as most do.
    #[must_use]
    pub(crate) fn child(self, name: &'static str, f: impl FnOnce(Self) -> Self) -> Self {
        self.with(f(Self::new(name)))
    }

    /// Whether a null record ends this one, `last` when it's the last of its siblings.
    fn ends(&self, last: bool) -> bool {
        !self.children.is_empty() || self.closed || (self.count == 0 && !last)
    }

    /// The record's size in bytes, with `wide` (64-bit) offsets.
    fn size(&self, wide: bool, last: bool) -> u64 {
        let head = if wide { 25 } else { 13 };
        let n = self.children.len();
        let kids: u64 = self
            .children
            .iter()
            .enumerate()
            .map(|(i, c)| c.size(wide, i + 1 == n))
            .sum();
        head + self.name.len() as u64 + self.props.len() as u64 + kids + if self.ends(last) { head } else { 0 }
    }

    fn write(&self, w: &mut dyn Write, at: &mut u64, wide: bool, last: bool) -> std::io::Result<()> {
        let end = *at + self.size(wide, last);
        let name = u8::try_from(self.name.len()).unwrap_or(u8::MAX);
        if wide {
            w.write_all(&end.to_le_bytes())?;
            w.write_all(&u64::from(self.count).to_le_bytes())?;
            w.write_all(&(self.props.len() as u64).to_le_bytes())?;
            *at += 25;
        } else {
            w.write_all(&len32_u64(end).to_le_bytes())?;
            w.write_all(&self.count.to_le_bytes())?;
            w.write_all(&len32(self.props.len()).to_le_bytes())?;
            *at += 13;
        }
        w.write_all(&[name])?;
        w.write_all(self.name.as_bytes())?;
        w.write_all(&self.props)?;
        *at += u64::from(name) + self.props.len() as u64;
        let n = self.children.len();
        for (i, c) in self.children.iter().enumerate() {
            c.write(w, at, wide, i + 1 == n)?;
        }
        if self.ends(last) {
            null(w, at, wide)?;
        }
        debug_assert_eq!(*at, end, "record {} sized wrong", self.name);
        Ok(())
    }
}

/// A length that fits 32 bits (anything longer is refused before writing starts).
fn len32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

fn len32_u64(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

fn null(w: &mut dyn Write, at: &mut u64, wide: bool) -> std::io::Result<()> {
    let n = if wide { 25 } else { 13 };
    w.write_all(&[0; 25][..n])?;
    *at += n as u64;
    Ok(())
}

/// A record's attributes, with where the arrays among them still to deflate start.
type Packing<'a> = (&'a mut Vec<u8>, &'a mut Vec<usize>);

/// The stored length of the array whose encoding is at `at` in `props`.
fn stored(props: &[u8], at: usize) -> usize {
    u32::from_le_bytes(props[at + 4..at + 8].try_into().expect("four bytes")) as usize
}

/// Every record's attributes in `nodes` and below with arrays to deflate.
fn to_pack<'a>(nodes: &'a mut [Node], out: &mut Vec<Packing<'a>>) {
    for n in nodes {
        let Node {
            props, raw, children, ..
        } = n;
        if !raw.is_empty() {
            out.push((props, raw));
        }
        to_pack(children, out);
    }
}

/// Attributes `props` with the arrays starting at `raw` deflated, each kept so when that makes it smaller.
fn pack(props: &[u8], raw: &[usize], z: &mut Compress) -> Vec<u8> {
    let mut out = Vec::with_capacity(props.len());
    let mut from = 0;
    for &at in raw {
        let len = stored(props, at);
        out.extend_from_slice(&props[from..at]);
        let head = out.len();
        out.extend_from_slice(&[0; 8]);
        // Room for no more than the array itself: a deflate that doesn't fit wouldn't be kept.
        out.reserve_exact(len);
        let start = out.len();
        z.reset();
        let done = matches!(
            z.compress_vec(&props[at + 8..at + 8 + len], &mut out, FlushCompress::Finish),
            Ok(Status::StreamEnd)
        );
        let packed = out.len() - start;
        if done && packed < len {
            out[head..head + 4].copy_from_slice(&1u32.to_le_bytes());
            out[head + 4..head + 8].copy_from_slice(&len32(packed).to_le_bytes());
        } else {
            out.truncate(head);
            out.extend_from_slice(&props[at..at + 8 + len]);
        }
        from = at + 8 + len;
    }
    out.extend_from_slice(&props[from..]);
    out
}

/// Deflates the big arrays in `top` and below, each kept so when that makes it smaller, over up to `threads`
/// threads. Each array deflates on its own, so the bytes are the same however many threads there are.
pub(crate) fn deflate(top: &mut [Node], threads: usize) {
    let mut records = Vec::new();
    to_pack(top, &mut records);
    let total: usize = records
        .iter()
        .map(|(props, raw)| raw.iter().map(|&at| stored(props, at)).sum::<usize>())
        .sum();
    let threads = if total < PARALLEL_FROM { 1 } else { threads };
    let packed = crate::keep::par_map_with(
        &records,
        threads,
        || Compress::new(Compression::new(LEVEL), true),
        |z, (props, raw)| pack(props, raw, z),
    );
    for ((props, raw), p) in records.iter_mut().zip(packed) {
        **props = p;
        raw.clear();
    }
}

/// The largest attribute buffer in a tree, which a 7.4 record's 32-bit length must hold.
fn largest(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|n| n.props.len().max(largest(&n.children)))
        .max()
        .unwrap_or(0)
}

/// Whether `top` needs 64-bit offsets (FBX 7.5): 7.4 holds them in 32 bits.
pub(crate) fn wide(top: &[Node]) -> bool {
    let n = top.len();
    let body: u64 = top.iter().enumerate().map(|(i, c)| c.size(false, i + 1 == n)).sum();
    27 + body + 13 > u64::from(u32::MAX)
}

/// Writes `top` (the file's top-level records) as a binary FBX file.
pub(crate) fn write_file(top: &[Node], w: &mut dyn Write) -> Result<()> {
    if largest(top) > u32::MAX as usize {
        return Err(crate::Error::from_args(
            ErrorKind::LimitExceeded,
            format_args!("an FBX record's attributes would be over 4 GiB"),
        ));
    }
    let n = top.len();
    let wide = wide(top);
    let version: u32 = if wide { 7500 } else { 7400 };
    w.write_all(MAGIC)?;
    w.write_all(&version.to_le_bytes())?;
    let mut at = 27u64;
    for (i, c) in top.iter().enumerate() {
        c.write(w, &mut at, wide, i + 1 == n)?;
    }
    null(w, &mut at, wide)?;
    w.write_all(&FOOT_ID)?;
    at += 16;
    // Zeros to the next 16-byte boundary, then four more before the version.
    let pad = (16 - (at % 16) as usize) % 16;
    w.write_all(&[0; 20][..pad + 4])?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&[0; 120])?;
    w.write_all(&FOOT_MAGIC)?;
    Ok(())
}

/// A `Properties70` record's `P` entry: name, type, label and flags, then the values `f` adds.
pub(crate) fn p(name: &str, kind: &str, label: &str, flags: &str) -> Node {
    Node::new("P").str(name).str(kind).str(label).str(flags)
}

/// A `P` entry holding a double.
pub(crate) fn p_f64(name: &str, kind: &str, flags: &str, v: f64) -> Node {
    let label = if kind == "double" { "Number" } else { "" };
    p(name, kind, label, flags).f64(v)
}

/// A `P` entry holding three doubles (a vector or color).
pub(crate) fn p_vec(name: &str, kind: &str, label: &str, flags: &str, v: [f64; 3]) -> Node {
    p(name, kind, label, flags).f64(v[0]).f64(v[1]).f64(v[2])
}

/// A `P` entry holding an integer, as `kind` ("int", "enum", "bool").
pub(crate) fn p_int(name: &str, kind: &str, v: i32) -> Node {
    let label = if kind == "int" { "Integer" } else { "" };
    p(name, kind, label, "").i32(v)
}

/// A `P` entry holding a string.
pub(crate) fn p_str(name: &str, v: &str) -> Node {
    p(name, "KString", "", "").str(v)
}

/// A `P` entry holding a time.
pub(crate) fn p_time(name: &str, v: i64) -> Node {
    p(name, "KTime", "Time", "").i64(v)
}

/// A `Properties70` record holding `entries`.
pub(crate) fn props70(entries: Vec<Node>) -> Node {
    Node {
        name: "Properties70",
        children: entries,
        ..Node::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_size_themselves_and_end_where_they_say() {
        let top = vec![
            Node::new("A").i32(1).with(Node::new("B").arr(&[1.0f64; 400])),
            Node::new("C"),
            Node::new("D").str("x"),
        ];
        let mut out = Vec::new();
        write_file(&top, &mut out).unwrap();
        assert_eq!(&out[..23], MAGIC);
        assert_eq!(u32::from_le_bytes(out[23..27].try_into().unwrap()), 7400);
        // The first record ends where its own header says.
        let end = u32::from_le_bytes(out[27..31].try_into().unwrap()) as usize;
        let size = usize::try_from(top[0].size(false, false)).unwrap();
        assert_eq!(end, 27 + size);
        // The footer ends the file with its magic, the version 120 zeros before it.
        assert_eq!(&out[out.len() - 16..], &FOOT_MAGIC);
        let v = out.len() - 16 - 120 - 4;
        assert_eq!(u32::from_le_bytes(out[v..v + 4].try_into().unwrap()), 7400);
        assert_eq!(v % 16, 4);
    }

    #[test]
    fn the_footer_pads_to_a_boundary_whatever_the_length() {
        // Every padding length, from none to fifteen bytes.
        let mut pads = std::collections::BTreeSet::new();
        for len in 0..16 {
            let top = vec![Node::new("A").str(&"x".repeat(len))];
            let mut out = Vec::new();
            write_file(&top, &mut out).unwrap();
            let id = out.windows(16).position(|w| w == FOOT_ID).unwrap() + 16;
            let v = out.len() - 16 - 120 - 4;
            assert_eq!(v % 16, 4);
            assert!(out[id..v].iter().all(|&b| b == 0));
            pads.insert(v - 4 - id);
        }
        assert_eq!(pads.len(), 16);
    }

    #[test]
    fn big_arrays_are_deflated_and_small_ones_stored() {
        let mut top = [Node::new("A").arr(&[7i32; 1000])];
        deflate(&mut top, 1);
        let [n] = top;
        // type, count, encoding, length
        assert_eq!(n.props[0], b'i');
        assert_eq!(u32::from_le_bytes(n.props[5..9].try_into().unwrap()), 1);
        let n = Node::new("A").arr(&[7i32; 4]);
        assert_eq!(u32::from_le_bytes(n.props[5..9].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(n.props[9..13].try_into().unwrap()), 16);
    }
}
