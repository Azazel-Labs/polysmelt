//! FBX 7.x as a tree of nodes, each with a name, attributes and children, read from binary FBX here and from ASCII FBX
//! in `fbx_ascii`. Names, strings and arrays borrow the file's bytes. An array is decoded only when it's read: a stored
//! one is copied out whole, a compressed one inflated straight into a buffer of its type, and one nothing reads costs
//! nothing, so a big file's vertices, normals and indices are each touched once. Nodes are read with a stack of their
//! own rather than the call stack, so nesting of any depth reads, and every count and size the file gives is checked
//! against the bytes it has before anything is reserved for it.

use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;

use flate2::{Decompress, FlushDecompress, Status};

use crate::{Context as _, Result};

/// The start of every binary FBX file, before its version.
pub(super) const BINARY_MAGIC: &[u8] = b"Kaydara FBX Binary  \0";

/// The most a deflate stream inflates: 258 bytes from a two-bit code.
const MAX_INFLATE_RATIO: usize = 1032;

/// An FBX version as the file writes it: 7400 for 7.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Version(pub(super) u32);

impl Version {
    pub(super) fn major(self) -> u32 {
        self.0 / 1000
    }

    pub(super) fn minor(self) -> u32 {
        self.0 % 1000 / 100
    }
}

/// A node's attribute (a property, in the FBX SDK's words).
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Attr<'a> {
    Bool(bool),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    /// Booleans, a byte each.
    ArrBool(Array<'a, u8>),
    ArrI32(Array<'a, i32>),
    ArrI64(Array<'a, i64>),
    ArrF32(Array<'a, f32>),
    ArrF64(Array<'a, f64>),
    Binary(Cow<'a, [u8]>),
    /// Text, with any bytes that aren't UTF-8 replaced.
    String(Cow<'a, str>),
}

/// A number type an array holds.
pub(super) trait Elem: bytemuck::Pod {
    /// The value whose little-endian bytes were read as this one (itself, on a little-endian machine).
    #[must_use]
    fn to_native(self) -> Self;
}

macro_rules! elem {
    ($($t:ty => $via:ty),*) => {$(
        impl Elem for $t {
            fn to_native(self) -> Self {
                bytemuck::cast(<$via>::from_le(bytemuck::cast(self)))
            }
        }
    )*};
}

elem!(u8 => u8, i32 => i32, i64 => i64, f32 => u32, f64 => u64);

/// An array attribute: its values, read when asked for.
#[derive(Clone, PartialEq)]
pub(super) struct Array<'a, T> {
    len: usize,
    data: Data<'a, T>,
}

#[derive(Clone, PartialEq)]
enum Data<'a, T> {
    /// Little-endian values, at least `len` of them.
    Stored(&'a [u8]),
    /// A zlib stream of `len` little-endian values.
    Zlib(&'a [u8]),
    /// Values already decoded (ASCII FBX's).
    Values(Vec<T>),
    /// An encoding not known, or fewer bytes than the values need.
    Unreadable,
}

impl<T: Elem> Array<'_, T> {
    /// An array of `len` values in `bytes`, stored (`encoding` 0) or zlib-compressed (1).
    fn raw(len: usize, encoding: u32, bytes: &[u8]) -> Array<'_, T> {
        let size = size_of::<T>();
        let (len, data) = match encoding {
            0 if len.checked_mul(size).is_some_and(|n| n <= bytes.len()) => (len, Data::Stored(bytes)),
            1 => (
                len.min(bytes.len().saturating_mul(MAX_INFLATE_RATIO) / size),
                Data::Zlib(bytes),
            ),
            _ => (0, Data::Unreadable),
        };
        Array { len, data }
    }

    /// How many values it holds (never more than its bytes could).
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Its values, or `None` if they can't be read (a compressed stream that's corrupt, or holds a different count).
    pub(super) fn to_vec(&self) -> Option<Vec<T>> {
        let mut out = match &self.data {
            Data::Values(v) => return Some(v.clone()),
            Data::Unreadable => return None,
            Data::Stored(bytes) => {
                let mut out = vec![T::zeroed(); self.len];
                let n = out.len() * size_of::<T>();
                bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(bytes.get(..n)?);
                out
            }
            Data::Zlib(packed) => inflate(packed, self.len)?,
        };
        if cfg!(target_endian = "big") {
            for v in &mut out {
                *v = v.to_native();
            }
        }
        Some(out)
    }

    /// Its values, each through `f`.
    pub(super) fn map<U>(&self, f: impl FnMut(T) -> U) -> Option<Vec<U>> {
        match &self.data {
            Data::Values(v) => Some(v.iter().copied().map(f).collect()),
            Data::Stored(bytes) => Some(
                bytes
                    .get(..self.len * size_of::<T>())?
                    .chunks_exact(size_of::<T>())
                    .map(|b| bytemuck::pod_read_unaligned::<T>(b).to_native())
                    .map(f)
                    .collect(),
            ),
            Data::Zlib(_) | Data::Unreadable => Some(self.to_vec()?.into_iter().map(f).collect()),
        }
    }
}

impl<T> From<Vec<T>> for Array<'_, T> {
    fn from(v: Vec<T>) -> Self {
        Array {
            len: v.len(),
            data: Data::Values(v),
        }
    }
}

impl<T> fmt::Debug for Array<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let how = match self.data {
            Data::Stored(_) => "stored",
            Data::Zlib(_) => "zlib",
            Data::Values(_) => "decoded",
            Data::Unreadable => "unreadable",
        };
        write!(f, "Array({} values, {how})", self.len)
    }
}

thread_local! {
    /// One decoder for every array read on a thread, reset between them.
    static ZLIB: RefCell<Decompress> = RefCell::new(Decompress::new(true));
}

/// `len` values inflated from the zlib stream `packed`, which must hold exactly that many.
fn inflate<T: Elem>(packed: &[u8], len: usize) -> Option<Vec<T>> {
    let size = len.checked_mul(size_of::<T>())?;
    if size > packed.len().saturating_mul(MAX_INFLATE_RATIO) {
        return None;
    }
    let mut out = vec![T::zeroed(); len];
    let buf = bytemuck::cast_slice_mut::<T, u8>(&mut out);
    let mut run = |z: &mut Decompress| -> Option<()> {
        z.reset(true);
        let mut status = z.decompress(packed, buf, FlushDecompress::Finish).ok()?;
        let filled = |z: &Decompress| usize::try_from(z.total_out()).ok() == Some(size);
        if status != Status::StreamEnd && filled(z) {
            // The buffer is full; what's left of the stream must end it without another byte.
            let used = usize::try_from(z.total_in()).ok()?;
            status = z
                .decompress(packed.get(used..)?, &mut [0; 1], FlushDecompress::Finish)
                .ok()?;
        }
        (status == Status::StreamEnd && filled(z)).then_some(())
    };
    ZLIB.try_with(|z| match z.try_borrow_mut() {
        Ok(mut z) => run(&mut z),
        Err(_) => run(&mut Decompress::new(true)),
    })
    .ok()
    .flatten()?;
    Some(out)
}

/// A file's nodes, under a root that has no name or attributes of its own.
pub(super) struct Tree<'a> {
    nodes: Vec<Node<'a>>,
    attrs: Vec<Attr<'a>>,
    version: Version,
}

struct Node<'a> {
    name: Cow<'a, str>,
    /// Its attributes' range in [`Tree::attrs`].
    attrs: (usize, usize),
    /// Its first child, and its next sibling, or 0 (the root, never either) for none.
    first_child: usize,
    next: usize,
}

impl<'a> Tree<'a> {
    /// The tree of an FBX file, binary 7.x or ASCII.
    ///
    /// # Errors
    ///
    /// When the file isn't FBX, is older than FBX 7, or isn't laid out as FBX is.
    pub(super) fn parse(data: &'a [u8]) -> Result<Self> {
        if super::fbx_ascii::is_ascii(data) {
            return super::fbx_ascii::parse(data).context("couldn't read this ASCII FBX file");
        }
        if let Some(v) = data.strip_prefix(BINARY_MAGIC).and_then(|r| r.get(2..6)) {
            let v = u32::from_le_bytes(v.try_into().unwrap_or_default());
            ensure!(
                v >= 7000,
                "this is FBX {}.{}, older than FBX 7, which this reader and Blender cannot read",
                v / 1000,
                v % 1000 / 100
            );
        }
        binary(data).map_err(|e| invalid!("not a readable FBX file: {e}"))
    }

    pub(super) fn version(&self) -> Version {
        self.version
    }

    pub(super) fn with_version(mut self, version: Version) -> Self {
        self.version = version;
        self
    }

    pub(super) fn root(&self) -> NodeHandle<'_> {
        NodeHandle { tree: self, i: 0 }
    }
}

/// A node in a [`Tree`].
#[derive(Clone, Copy)]
pub(super) struct NodeHandle<'a> {
    tree: &'a Tree<'a>,
    i: usize,
}

impl<'a> NodeHandle<'a> {
    fn node(self) -> &'a Node<'a> {
        &self.tree.nodes[self.i]
    }

    fn at(self, i: usize) -> Option<Self> {
        (i != 0).then_some(Self { tree: self.tree, i })
    }

    pub(super) fn name(self) -> &'a str {
        &self.node().name
    }

    pub(super) fn attributes(self) -> &'a [Attr<'a>] {
        let (start, end) = self.node().attrs;
        &self.tree.attrs[start..end]
    }

    pub(super) fn first_child(self) -> Option<Self> {
        self.at(self.node().first_child)
    }

    pub(super) fn next_sibling(self) -> Option<Self> {
        self.at(self.node().next)
    }

    pub(super) fn children(self) -> Children<'a> {
        Children(self.first_child())
    }

    pub(super) fn children_by_name(self, name: &str) -> impl Iterator<Item = Self> + use<'a, '_> {
        self.children().filter(move |c| c.name() == name)
    }

    pub(super) fn first_child_by_name(self, name: &str) -> Option<Self> {
        self.children_by_name(name).next()
    }
}

impl fmt::Debug for NodeHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeHandle({:?})", self.name())
    }
}

/// A node's children, in file order.
pub(super) struct Children<'a>(Option<NodeHandle<'a>>);

impl<'a> Iterator for Children<'a> {
    type Item = NodeHandle<'a>;

    fn next(&mut self) -> Option<NodeHandle<'a>> {
        let n = self.0?;
        self.0 = n.next_sibling();
        Some(n)
    }
}

/// Builds a [`Tree`] node by node, in file order: a node's attributes first, then the node, then its children
/// between [`Builder::open`] and [`Builder::close`].
pub(super) struct Builder<'a> {
    tree: Tree<'a>,
    /// The open nodes, the root first, each with its last child so far (0 for none).
    open: Vec<(usize, usize)>,
}

impl Default for Builder<'_> {
    fn default() -> Self {
        Self {
            tree: Tree {
                nodes: vec![Node {
                    name: Cow::Borrowed(""),
                    attrs: (0, 0),
                    first_child: 0,
                    next: 0,
                }],
                attrs: Vec::new(),
                version: Version(0),
            },
            open: vec![(0, 0)],
        }
    }
}

impl<'a> Builder<'a> {
    /// Where the next node's attributes go.
    pub(super) fn attrs(&mut self) -> &mut Vec<Attr<'a>> {
        &mut self.tree.attrs
    }

    /// Adds a node as the open node's last child, with the attributes added since there were `from` of them;
    /// returns it.
    pub(super) fn add(&mut self, name: Cow<'a, str>, from: usize) -> usize {
        let i = self.tree.nodes.len();
        self.tree.nodes.push(Node {
            name,
            attrs: (from, self.tree.attrs.len()),
            first_child: 0,
            next: 0,
        });
        // The root is always open: `close` keeps it.
        if let Some((parent, last)) = self.open.last_mut() {
            if *last == 0 {
                self.tree.nodes[*parent].first_child = i;
            } else {
                self.tree.nodes[*last].next = i;
            }
            *last = i;
        }
        i
    }

    /// Makes node `i` (just added) the one the next nodes go in.
    pub(super) fn open(&mut self, i: usize) {
        self.open.push((i, 0));
    }

    /// Closes the open node; `false` if that's the root, which stays open.
    pub(super) fn close(&mut self) -> bool {
        if self.open.len() > 1 {
            self.open.pop();
            true
        } else {
            false
        }
    }

    /// How many nodes are open, the root not counted.
    pub(super) fn depth(&self) -> usize {
        self.open.len() - 1
    }

    pub(super) fn finish(mut self, version: Version) -> Tree<'a> {
        self.tree.version = version;
        self.tree
    }
}

/// Binary FBX 7.x: the header, then nodes, each closed by its end's offset, and lists of children by a null node.
fn binary(data: &[u8]) -> std::result::Result<Tree<'_>, String> {
    let rest = data
        .strip_prefix(BINARY_MAGIC)
        .ok_or("it starts like neither binary nor ASCII FBX")?;
    let version = rest
        .get(2..6)
        .and_then(|v| v.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or("the file ends in its header")?;
    // From FBX 7.5, a node's end, attribute count and attribute bytes are 64-bit.
    let wide = version >= 7500;
    let w = if wide { 8 } else { 4 };
    let header = 3 * w + 1;
    let word = |at: usize| -> Option<usize> {
        let b = data.get(at..at + w)?;
        if wide {
            usize::try_from(u64::from_le_bytes(b.try_into().ok()?)).ok()
        } else {
            usize::try_from(u32::from_le_bytes(b.try_into().ok()?)).ok()
        }
    };
    let mut b = Builder::default();
    // Where each open node ends, the file's end for the root.
    let mut ends = vec![data.len()];
    let mut at = BINARY_MAGIC.len() + 6;
    loop {
        let end = ends.last().copied().unwrap_or(data.len());
        let top = ends.len() == 1;
        if at == end {
            if top {
                // No null node closes the file's nodes: all there is has been read.
                break;
            }
            // A node whose children fill it with no null node after them.
            ends.pop();
            b.close();
            continue;
        }
        if at + header > end {
            return Err(format!(
                "the file is cut short, or a node overruns its parent, at byte {at}"
            ));
        }
        let (Some(node_end), Some(count), Some(list)) = (word(at), word(at + w), word(at + 2 * w)) else {
            return Err(format!("a node at byte {at} has sizes too large"));
        };
        if node_end == 0 {
            // The null node ending a list of children; the rest of the parent (or the file, past the root's: its
            // footer) is padding.
            if top {
                break;
            }
            at = end;
            continue;
        }
        let name_at = at + header;
        let props = name_at + usize::from(data[at + 3 * w]);
        let props_end = props.checked_add(list).filter(|&e| e <= node_end);
        let Some(props_end) = props_end.filter(|_| node_end <= end) else {
            return Err(format!(
                "the node at byte {at} ends past its parent or the file (cut short?)"
            ));
        };
        let name = String::from_utf8_lossy(&data[name_at..props]);
        let from = b.attrs().len();
        attributes(&data[props..props_end], count, b.attrs())
            .map_err(|e| format!("in node {name:?} at byte {at}: {e}"))?;
        let node = b.add(name, from);
        if props_end < node_end {
            b.open(node);
            ends.push(node_end);
            at = props_end;
        } else {
            at = node_end;
        }
    }
    Ok(b.finish(Version(version)))
}

/// Reads `count` attributes from `data` into `out`; bytes left over are padding.
fn attributes<'a>(mut data: &'a [u8], count: usize, out: &mut Vec<Attr<'a>>) -> std::result::Result<(), String> {
    fn take<'a>(data: &mut &'a [u8], n: usize) -> std::result::Result<&'a [u8], String> {
        if n > data.len() {
            return Err("an attribute runs past the node's attribute bytes".into());
        }
        let (head, tail) = data.split_at(n);
        *data = tail;
        Ok(head)
    }
    fn u32_at(data: &mut &[u8]) -> std::result::Result<u32, String> {
        Ok(u32::from_le_bytes(take(data, 4)?.try_into().unwrap_or_default()))
    }
    fn fixed<const N: usize>(data: &mut &[u8]) -> std::result::Result<[u8; N], String> {
        Ok(take(data, N)?.try_into().unwrap_or([0; N]))
    }
    // Each attribute takes at least two bytes, so `count` can't run long past what `data` holds.
    for _ in 0..count {
        let kind = take(&mut data, 1)?[0];
        out.push(match kind {
            b'C' => Attr::Bool(fixed::<1>(&mut data)?[0] & 1 != 0),
            b'Y' => Attr::I16(i16::from_le_bytes(fixed(&mut data)?)),
            b'I' => Attr::I32(i32::from_le_bytes(fixed(&mut data)?)),
            b'L' => Attr::I64(i64::from_le_bytes(fixed(&mut data)?)),
            b'F' => Attr::F32(f32::from_le_bytes(fixed(&mut data)?)),
            b'D' => Attr::F64(f64::from_le_bytes(fixed(&mut data)?)),
            b'S' | b'R' => {
                let n = u32_at(&mut data)? as usize;
                let bytes = take(&mut data, n)?;
                if kind == b'S' {
                    Attr::String(String::from_utf8_lossy(bytes))
                } else {
                    Attr::Binary(Cow::Borrowed(bytes))
                }
            }
            b'b' | b'i' | b'l' | b'f' | b'd' => {
                let len = u32_at(&mut data)? as usize;
                let encoding = u32_at(&mut data)?;
                let n = u32_at(&mut data)? as usize;
                let bytes = take(&mut data, n)?;
                match kind {
                    b'b' => Attr::ArrBool(Array::raw(len, encoding, bytes)),
                    b'i' => Attr::ArrI32(Array::raw(len, encoding, bytes)),
                    b'l' => Attr::ArrI64(Array::raw(len, encoding, bytes)),
                    b'f' => Attr::ArrF32(Array::raw(len, encoding, bytes)),
                    _ => Attr::ArrF64(Array::raw(len, encoding, bytes)),
                }
            }
            _ => return Err(format!("an attribute of unknown type {kind:#04x}")),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use fbxcel::low::v7400::AttributeValue as Old;
    use fbxcel::tree::any::AnyTree;

    use super::super::fbx_ascii;
    use super::*;

    /// Runs `f` on a thread with a 256 KiB stack, as nesting of any depth must read on one.
    fn small_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 << 10)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    fn same_attr(old: &Old, new: &Attr<'_>) -> bool {
        match (old, new) {
            (Old::Bool(a), Attr::Bool(b)) => a == b,
            (Old::I16(a), Attr::I16(b)) => a == b,
            (Old::I32(a), Attr::I32(b)) => a == b,
            (Old::I64(a), Attr::I64(b)) => a == b,
            (Old::F32(a), Attr::F32(b)) => a.to_bits() == b.to_bits(),
            (Old::F64(a), Attr::F64(b)) => a.to_bits() == b.to_bits(),
            (Old::ArrBool(a), Attr::ArrBool(b)) => b.map(|x| x & 1 != 0).as_ref() == Some(a),
            (Old::ArrI32(a), Attr::ArrI32(b)) => b.to_vec().as_ref() == Some(a),
            (Old::ArrI64(a), Attr::ArrI64(b)) => b.to_vec().as_ref() == Some(a),
            (Old::ArrF32(a), Attr::ArrF32(b)) => b.map(f32::to_bits) == Some(a.iter().map(|x| x.to_bits()).collect()),
            (Old::ArrF64(a), Attr::ArrF64(b)) => b.map(f64::to_bits) == Some(a.iter().map(|x| x.to_bits()).collect()),
            (Old::Binary(a), Attr::Binary(b)) => a[..] == b[..],
            (Old::String(a), Attr::String(b)) => a == b,
            _ => false,
        }
    }

    /// Both trees hold the same nodes, walked with a stack so deep files don't overflow this one.
    fn same(old: fbxcel::tree::v7400::NodeHandle<'_>, new: NodeHandle<'_>) {
        let mut todo = vec![(old, new)];
        while let Some((a, b)) = todo.pop() {
            assert_eq!(a.name(), b.name());
            let (aa, ba) = (a.attributes(), b.attributes());
            assert_eq!(aa.len(), ba.len(), "in {}", a.name());
            for (x, y) in aa.iter().zip(ba) {
                assert!(same_attr(x, y), "in {}: {x:?} read as {y:?}", a.name());
            }
            let (ac, bc): (Vec<_>, Vec<_>) = (a.children().collect(), b.children().collect());
            assert_eq!(ac.len(), bc.len(), "in {}", a.name());
            todo.extend(ac.into_iter().zip(bc));
        }
    }

    fn old_tree(data: &[u8]) -> fbxcel::tree::v7400::Tree {
        let AnyTree::V7400(_, tree, _) = AnyTree::from_seekable_reader(std::io::Cursor::new(data)).unwrap() else {
            panic!("not FBX 7.4")
        };
        tree
    }

    fn corpus() -> Vec<(std::path::PathBuf, Vec<u8>)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/fbx");
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            for file in std::fs::read_dir(entry.unwrap().path()).into_iter().flatten() {
                let path = file.unwrap().path();
                if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("fbx")) {
                    let data = std::fs::read(&path).unwrap();
                    out.push((path, data));
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn every_corpus_file_reads_as_before() {
        let (mut binary, mut ascii) = (0, 0);
        for (path, data) in corpus() {
            let Ok(tree) = Tree::parse(&data) else {
                assert!(
                    fbx_ascii::is_ascii(&data) || AnyTree::from_seekable_reader(std::io::Cursor::new(&data)).is_err(),
                    "{path:?} read before"
                );
                continue;
            };
            if fbx_ascii::is_ascii(&data) {
                // As binary, it reads the same, both here and by the FBX SDK's layout.
                ascii += 1;
                let bin = fbx_ascii::to_binary(&data).unwrap();
                same(old_tree(&bin).root(), tree.root());
                let again = Tree::parse(&bin).unwrap();
                assert_eq!(again.version(), Version(7400));
                same(old_tree(&bin).root(), again.root());
            } else {
                binary += 1;
                same(old_tree(&data).root(), tree.root());
            }
        }
        assert!(binary > 50 && ascii > 0, "{binary} binary, {ascii} ASCII");
    }

    /// A binary FBX 7.4 file holding `nodes`, laid out from the header's end (byte 27).
    fn file(nodes: &[u8]) -> Vec<u8> {
        let mut out = BINARY_MAGIC.to_vec();
        out.extend_from_slice(&[0x1a, 0]);
        out.extend_from_slice(&7400u32.to_le_bytes());
        out.extend_from_slice(nodes);
        out.extend_from_slice(&[0; 13]);
        out
    }

    /// The first node of a file: `count` attributes in `attrs`, and no children.
    fn node(name: &str, count: u32, attrs: &[u8]) -> Vec<u8> {
        let end = 27 + 13 + name.len() + attrs.len();
        let mut out = Vec::new();
        out.extend_from_slice(&u32::try_from(end).unwrap().to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&u32::try_from(attrs.len()).unwrap().to_le_bytes());
        out.push(u8::try_from(name.len()).unwrap());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(attrs);
        out
    }

    /// An array attribute's bytes.
    fn array(kind: u8, len: u32, encoding: u32, data: &[u8]) -> Vec<u8> {
        let mut out = vec![kind];
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&encoding.to_le_bytes());
        out.extend_from_slice(&u32::try_from(data.len()).unwrap().to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    fn zlib(data: &[u8]) -> Vec<u8> {
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(data).unwrap();
        z.finish().unwrap()
    }

    /// Calls `test` with the attribute of a file whose one node holds just `attr`.
    fn with_attr(attr: &[u8], test: impl FnOnce(&Attr<'_>)) {
        let data = file(&node("A", 1, attr));
        let tree = Tree::parse(&data).unwrap();
        test(&tree.root().first_child().unwrap().attributes()[0]);
    }

    #[test]
    fn nesting_of_any_depth_reads_on_a_small_stack() {
        small_stack(|| {
            const DEPTH: usize = 100_000;
            // Each node is a 13-byte header and a one-byte name, then its child, then a null node.
            let mut nodes = Vec::with_capacity(DEPTH * 27);
            for d in 0..DEPTH {
                let end = 27 + 14 * d + 27 * (DEPTH - d);
                nodes.extend_from_slice(&u32::try_from(end).unwrap().to_le_bytes());
                nodes.extend_from_slice(&[0; 8]);
                nodes.extend_from_slice(b"\x01N");
            }
            nodes.resize(nodes.len() + 13 * DEPTH, 0);
            let data = file(&nodes);
            let depth = |tree: &Tree<'_>| {
                std::iter::successors(tree.root().first_child_by_name("N"), |n| n.first_child()).count()
            };
            assert_eq!(depth(&Tree::parse(&data).unwrap()), DEPTH);
            assert!(Tree::parse(&data[..data.len() / 2]).is_err(), "cut short");

            let mut text = String::from("; FBX 7.4.0 project file\nFBXHeaderExtension:  {\n}\n");
            text.push_str(&"N: {\n".repeat(DEPTH));
            text.push_str(&"}\n".repeat(DEPTH));
            let tree = Tree::parse(text.as_bytes()).unwrap();
            assert_eq!(depth(&tree), DEPTH);
            let bin = fbx_ascii::to_binary(text.as_bytes()).unwrap();
            assert_eq!(depth(&Tree::parse(&bin).unwrap()), DEPTH);
            // Unclosed, it's refused.
            let open = format!("FBXHeaderExtension:  {{\n}}\n{}", "N: {\n".repeat(DEPTH));
            assert!(Tree::parse(open.as_bytes()).is_err());
        });
    }

    #[test]
    fn cut_short_or_garbled_files_are_refused_without_panicking() {
        small_stack(|| {
            let mut files: Vec<_> = corpus()
                .into_iter()
                .map(|(_, d)| d)
                .filter(|d| d.len() > 1000)
                .collect();
            files.truncate(8);
            files.push(
                fbx_ascii::to_binary(include_bytes!(
                    "../../tests/corpus/fbx/cube_ascii7400/cube_ascii7400.fbx"
                ))
                .unwrap(),
            );
            files.push(include_bytes!("../../tests/corpus/fbx/cube_ascii7400/cube_ascii7400.fbx").to_vec());
            let read_all = |data: &[u8]| {
                if let Ok(tree) = Tree::parse(data) {
                    for a in &tree.attrs {
                        match a {
                            Attr::ArrBool(v) => drop(v.to_vec()),
                            Attr::ArrI32(v) => drop(v.map(i64::from)),
                            Attr::ArrI64(v) => drop(v.to_vec()),
                            Attr::ArrF32(v) => drop(v.map(f64::from)),
                            Attr::ArrF64(v) => drop(v.to_vec()),
                            _ => {}
                        }
                    }
                    if fbx_ascii::is_ascii(data) {
                        drop(fbx_ascii::to_binary(data));
                    }
                }
            };
            let mut seed = 0x2545_f491_4f6c_dd1d_u64;
            let mut rand = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                usize::try_from(seed >> 40).unwrap()
            };
            for data in &files {
                for len in (0..data.len()).step_by(61) {
                    read_all(&data[..len]);
                    read_all(&data[..(len + 1).min(data.len())]);
                }
                for _ in 0..300 {
                    let mut bad = data.clone();
                    for _ in 0..=rand() % 8 {
                        let at = rand() % bad.len();
                        bad[at] = u8::try_from(rand() & 0xff).unwrap();
                    }
                    read_all(&bad);
                }
            }
            for len in [0, 1, 20, 26, 27, 40, 200, 5000] {
                let mut junk = BINARY_MAGIC.to_vec();
                junk.extend((0..len).map(|_| u8::try_from(rand() & 0xff).unwrap()));
                read_all(&junk);
                let junk: Vec<u8> = (0..len).map(|_| u8::try_from(rand() & 0xff).unwrap()).collect();
                read_all(&junk);
            }
            assert!(Tree::parse(b"").is_err());
            assert!(Tree::parse(b"Kaydara FBX Binary  \0").is_err());
            assert!(Tree::parse(b"FBXHeaderExtension: {\n} \"unclosed\n").is_err());
            let huge = b"; FBX 4294967.4294967 project file\nFBXHeaderExtension: {\n}\n";
            assert_eq!(Tree::parse(huge).unwrap().version(), Version(7400));
        });
    }

    #[test]
    fn an_array_claiming_more_than_its_bytes_hold_reserves_nothing() {
        small_stack(|| {
            // Stored: four bytes can't hold four billion values.
            with_attr(&array(b'd', u32::MAX, 0, &[0; 4]), |a| {
                let Attr::ArrF64(a) = a else { panic!("{a:?}") };
                assert_eq!(a.len(), 0);
                assert_eq!(a.to_vec(), None);
            });
            // Compressed: no more than the stream could inflate to, and then only what it does.
            let packed = zlib(&[0; 16]);
            with_attr(&array(b'i', u32::MAX, 1, &packed), |a| {
                let Attr::ArrI32(a) = a else { panic!("{a:?}") };
                assert!(a.len() <= packed.len() * MAX_INFLATE_RATIO / 4);
                assert_eq!(a.to_vec(), None);
            });
            // An encoding not known.
            with_attr(&array(b'l', 2, 7, &[0; 16]), |a| {
                let Attr::ArrI64(a) = a else { panic!("{a:?}") };
                assert_eq!(a.to_vec(), None);
            });
            // A string longer than its node.
            let mut s = vec![b'S'];
            s.extend_from_slice(&u32::MAX.to_le_bytes());
            assert!(Tree::parse(&file(&node("A", 1, &s))).is_err());
            // Four billion attributes in five bytes.
            assert!(Tree::parse(&file(&node("A", u32::MAX, &[b'I', 0, 0, 0, 0]))).is_err());
            // An ASCII array declaring four billion values and holding one.
            let text = b"FBXHeaderExtension: {\n}\nVertices: *4294967295 {\n a: 1\n}\n";
            assert!(Tree::parse(text).is_err());
        });
    }

    #[test]
    fn a_compressed_array_must_hold_its_count_exactly() {
        let values: Vec<i32> = (0..1000).map(|i| i * 7 - 3000).collect();
        let packed = zlib(bytemuck::cast_slice(&values));
        let read = |len: u32, bytes: &[u8]| {
            let mut out = None;
            with_attr(&array(b'i', len, 1, bytes), |a| {
                let Attr::ArrI32(a) = a else { panic!("{a:?}") };
                out = a.to_vec();
            });
            out
        };
        assert_eq!(read(1000, &packed), Some(values.clone()));
        assert_eq!(read(999, &packed), None, "longer than declared");
        assert_eq!(read(1001, &packed), None, "shorter than declared");
        let mut bad = packed.clone();
        let mid = bad.len() / 2;
        bad[mid] ^= 0x55;
        assert_eq!(read(1000, &bad), None, "corrupt");
        assert_eq!(read(1000, &packed[..packed.len() - 3]), None, "cut short");
        with_attr(&array(b'i', 1000, 0, bytemuck::cast_slice(&values)), |a| {
            let Attr::ArrI32(a) = a else { panic!("{a:?}") };
            assert_eq!(a.to_vec(), Some(values.clone()));
            assert_eq!(a.map(i64::from), Some(values.iter().map(|&v| i64::from(v)).collect()));
        });
    }
}
