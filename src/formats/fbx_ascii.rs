//! ASCII FBX (7.x), the text form some tools write (Asset Forge, which made many of Kenney's older kits, and others
//! on request), read into the same tree as binary FBX, so it's read like any other; the tests write that tree back as
//! binary FBX (`to_binary`) to check the two read alike. Values get the types binary FBX gives them, since Blender
//! checks: object ids and connections are 64-bit, a property's values follow the type it declares, and an array takes
//! the type its node always has. Nodes nest by a stack of their own, not the call stack, so any depth reads.

use std::borrow::Cow;

use crate::{Context as _, ErrorKind, Result};

#[cfg(test)]
use super::fbx_tree::{Array, BINARY_MAGIC, Elem, NodeHandle};
use super::fbx_tree::{Attr, Builder, Tree, Version};
use super::stream;

/// Whether `data` is an ASCII FBX file.
#[must_use]
pub(crate) fn is_ascii(data: &[u8]) -> bool {
    !data.starts_with(b"Kaydara FBX Binary") && {
        let head = &data[..data.len().min(64 << 10)];
        head.windows(19).any(|w| w == b"FBXHeaderExtension:")
    }
}

/// Rewrite an ASCII FBX file as binary FBX 7.4.
///
/// # Errors
///
/// When the text isn't ASCII FBX that parses.
#[cfg(test)]
pub(crate) fn to_binary(data: &[u8]) -> Result<Vec<u8>> {
    write(&parse(data)?)
}

/// The tree of an ASCII FBX file.
pub(super) fn parse(data: &[u8]) -> Result<Tree<'_>> {
    let mut src = data;
    while let Some(rest) = src.strip_prefix("\u{feff}".as_bytes()) {
        src = rest;
    }
    let first = src.split(|&c| c == b'\n').next().unwrap_or_default();
    let first = String::from_utf8_lossy(first);
    let stated = first.trim_end_matches('\r').strip_prefix("; FBX ").map(|r| {
        let mut parts = r.split(['.', ' ']).map(|p| p.trim().parse::<u32>().ok());
        (parts.next().flatten(), parts.next().flatten())
    });
    if let Some((Some(major), _)) = stated {
        ensure!(
            major >= 7,
            Unsupported,
            "this is FBX {major}, older than FBX 7, which this reader and Blender cannot read"
        );
    }
    let mut lex = Lexer {
        src,
        i: 0,
        line: 1,
        peeked: None,
    };
    let mut b = Builder::default();
    // The open nodes' names, for the types their children's values take.
    let mut names: Vec<&str> = Vec::new();
    let mut vals = Vec::new();
    loop {
        match lex.next()? {
            None => {
                if let Some(name) = names.last() {
                    bail!(Truncated, "the file is cut short inside {name}");
                }
                break;
            }
            Some(Tok::Close) if !names.is_empty() => {
                names.pop();
                b.close();
            }
            Some(Tok::Name(name)) => {
                let block = values(&mut lex, name, &mut vals)?;
                let from = b.attrs().len();
                attributes(name, names.last().copied().unwrap_or(""), &mut vals, b.attrs());
                let node = b.add(Cow::Borrowed(name), from);
                if block {
                    b.open(node);
                    names.push(name);
                }
            }
            Some(t) if !names.is_empty() => bail!(Corrupt, "line {}: expected a node, found {}", lex.line, t.what()),
            Some(_) => bail!(Corrupt, "line {}: expected a node name", lex.line),
        }
    }
    debug_assert_eq!(b.depth(), 0);
    // The version the header extension gives, else the first line's.
    let mut tree = b.finish(Version(7400));
    let header = tree
        .root()
        .first_child_by_name("FBXHeaderExtension")
        .and_then(|h| h.first_child_by_name("FBXVersion"))
        .and_then(|v| match v.attributes().first() {
            Some(Attr::I32(v)) => u32::try_from(*v).ok(),
            _ => None,
        });
    let version = header.or_else(|| match stated {
        Some((Some(major), minor)) => major
            .checked_mul(1000)?
            .checked_add(minor.unwrap_or(0).checked_mul(100)?),
        _ => None,
    });
    if let Some(v) = version.filter(|&v| v >= 7000) {
        tree = tree.with_version(Version(v));
    }
    Ok(tree)
}

#[derive(Debug)]
enum Tok<'a> {
    /// `Name:`
    Name(&'a str),
    Str(&'a [u8]),
    Num(&'a str),
    /// A bare word: `T`, `Y`, `N`, `F`, …
    Word(&'a str),
    Comma,
    Open,
    Close,
    Star,
}

impl Tok<'_> {
    fn what(&self) -> String {
        match self {
            Tok::Name(s) => format!("{}:", crate::shown(s.as_bytes())),
            Tok::Str(s) => crate::shown(s),
            Tok::Num(s) | Tok::Word(s) => crate::shown(s.as_bytes()),
            Tok::Comma => "','".into(),
            Tok::Open => "'{'".into(),
            Tok::Close => "'}'".into(),
            Tok::Star => "'*'".into(),
        }
    }
}

struct Lexer<'a> {
    src: &'a [u8],
    i: usize,
    line: usize,
    peeked: Option<Tok<'a>>,
}

/// Where a number, word or name stops.
fn ends_token(c: u8) -> bool {
    matches!(c, b',' | b'{' | b'}' | b';') || c.is_ascii_whitespace()
}

/// Bytes known to be ASCII as text.
fn ascii(b: &[u8]) -> &str {
    std::str::from_utf8(b).unwrap_or_default()
}

impl<'a> Lexer<'a> {
    fn peek(&mut self) -> Result<Option<&Tok<'a>>> {
        if self.peeked.is_none() {
            self.peeked = self.lex()?;
        }
        Ok(self.peeked.as_ref())
    }

    fn next(&mut self) -> Result<Option<Tok<'a>>> {
        match self.peeked.take() {
            Some(t) => Ok(Some(t)),
            None => self.lex(),
        }
    }

    /// Steps over white space and comments; `false` at the end of the text.
    fn skip_space(&mut self) -> bool {
        let b = self.src;
        loop {
            match b.get(self.i) {
                None => return false,
                Some(b'\n') => {
                    self.line += 1;
                    self.i += 1;
                }
                Some(c) if c.is_ascii_whitespace() => self.i += 1,
                Some(b';') => {
                    self.i += memchr(b'\n', &b[self.i..]).unwrap_or(b.len() - self.i);
                }
                Some(_) => return true,
            }
        }
    }

    /// A number token from `self.i`, which holds its first character.
    fn number(&mut self) -> &'a str {
        let b = self.src;
        let start = self.i;
        self.i += 1;
        while self.i < b.len() && !ends_token(b[self.i]) {
            self.i += 1;
        }
        // A number that isn't ASCII reads as nothing, and so as zero.
        ascii(&b[start..self.i])
    }

    fn lex(&mut self) -> Result<Option<Tok<'a>>> {
        if !self.skip_space() {
            return Ok(None);
        }
        let b = self.src;
        let start = self.i;
        let c = b[start];
        Ok(Some(match c {
            b'{' | b'}' | b',' | b'*' => {
                self.i += 1;
                match c {
                    b'{' => Tok::Open,
                    b'}' => Tok::Close,
                    b',' => Tok::Comma,
                    _ => Tok::Star,
                }
            }
            b'"' => {
                let body = start + 1;
                let end = memchr(b'"', &b[body..])
                    .with_context(ErrorKind::Corrupt, || format!("line {}: unclosed string", self.line))?;
                let s = &b[body..body + end];
                self.line += s.split(|&c| c == b'\n').count() - 1;
                self.i = body + end + 1;
                Tok::Str(s)
            }
            b'0'..=b'9' | b'-' | b'+' | b'.' => Tok::Num(self.number()),
            c if c.is_ascii_alphabetic() || c == b'_' => {
                self.i += 1;
                while self.i < b.len() && (b[self.i].is_ascii_alphanumeric() || matches!(b[self.i], b'_' | b'|' | b'-'))
                {
                    self.i += 1;
                }
                let word = ascii(&b[start..self.i]);
                let mut j = self.i;
                while j < b.len() && matches!(b[j], b' ' | b'\t') {
                    j += 1;
                }
                if b.get(j) == Some(&b':') {
                    self.i = j + 1;
                    Tok::Name(word)
                } else {
                    Tok::Word(word)
                }
            }
            _ => {
                let c = String::from_utf8_lossy(&b[start..b.len().min(start + 4)])
                    .chars()
                    .next()
                    .unwrap_or('?');
                bail!(Corrupt, "line {}: unexpected {c:?}", self.line)
            }
        }))
    }

    /// An array's values up to its closing `}`, each as `ty` (which is never [`Ty::Auto`]).
    fn array(&mut self, ty: Ty, len: usize) -> Result<Values> {
        // No more room than the rest of the text could fill, a value and a comma at least each.
        let room = len.min((self.src.len() - self.i) / 2 + 1);
        let mut out = match ty {
            Ty::I32 => Values::I32(Vec::with_capacity(room)),
            Ty::I64 => Values::I64(Vec::with_capacity(room)),
            Ty::F32 => Values::F32(Vec::with_capacity(room)),
            Ty::F64 | Ty::Auto => Values::F64(Vec::with_capacity(room)),
        };
        loop {
            if !self.skip_space() {
                bail!(Corrupt, "line {}: unexpected value in an array", self.line);
            }
            match self.src[self.i] {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'0'..=b'9' | b'-' | b'+' | b'.' => {
                    let n = number(self.number());
                    out.push(n);
                }
                _ => bail!(Corrupt, "line {}: unexpected value in an array", self.line),
            }
        }
    }
}

fn memchr(needle: u8, hay: &[u8]) -> Option<usize> {
    hay.iter().position(|&c| c == needle)
}

#[derive(Clone, Copy)]
enum Num {
    Int(i64),
    Float(f64),
}

/// A number: an integer unless it has a point, an exponent or a `#`.
fn number(s: &str) -> Num {
    // Most numbers are short integers, read here without the general parser.
    if let Some(n) = stream::int(s.as_bytes()) {
        return Num::Int(n);
    }
    if !s.contains(['.', 'e', 'E', '#'])
        && let Ok(i) = s.parse::<i64>()
    {
        return Num::Int(i);
    }
    // `1.#INF` and the like, from old Windows exporters, have no use here.
    Num::Float(stream::f64_from(s.as_bytes()).unwrap_or(0.0))
}

/// An array's values, as the type its node always has.
enum Values {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Values {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "each value takes the width its array declares, as FBX readers expect"
    )]
    fn push(&mut self, n: Num) {
        match (self, n) {
            (Values::I32(v), Num::Int(i)) => {
                v.push(i32::try_from(i).unwrap_or(if i < 0 { i32::MIN } else { i32::MAX }));
            }
            (Values::I32(v), Num::Float(f)) => v.push(f as i32),
            (Values::I64(v), Num::Int(i)) => v.push(i),
            (Values::I64(v), Num::Float(f)) => v.push(f as i64),
            (Values::F32(v), Num::Int(i)) => v.push(i as f64 as f32),
            (Values::F32(v), Num::Float(f)) => v.push(f as f32),
            (Values::F64(v), Num::Int(i)) => v.push(i as f64),
            (Values::F64(v), Num::Float(f)) => v.push(f),
        }
    }

    fn len(&self) -> usize {
        match self {
            Values::I32(v) => v.len(),
            Values::I64(v) => v.len(),
            Values::F32(v) => v.len(),
            Values::F64(v) => v.len(),
        }
    }
}

enum Val<'a> {
    Num(Num),
    Str(Cow<'a, str>),
    Bool(bool),
    Arr(Values),
    /// Kept for `Content`, whose strings are base64 to join and decode.
    Raw(&'a [u8]),
}

/// `Geometry::Rock` → binary FBX's `Rock\0\x01Geometry`.
fn object_name(s: &[u8]) -> Cow<'_, str> {
    let mut s = String::from_utf8_lossy(s);
    if s.contains("&quot;") {
        s = Cow::Owned(s.replace("&quot;", "\""));
    }
    if let Some((class, name)) = s.split_once("::")
        && !class.is_empty()
        && class.bytes().all(|c| c.is_ascii_alphanumeric())
    {
        return Cow::Owned(format!("{name}\0\x01{class}"));
    }
    s
}

/// A node's values into `vals`, up to its children's block or the next node; whether it has a block.
fn values<'a>(lex: &mut Lexer<'a>, name: &str, vals: &mut Vec<Val<'a>>) -> Result<bool> {
    vals.clear();
    loop {
        match lex.peek()? {
            None | Some(Tok::Name(_) | Tok::Close) => return Ok(false),
            Some(Tok::Open) => {
                lex.next()?;
                return Ok(true);
            }
            Some(_) => {}
        }
        match lex.next()?.context(ErrorKind::Corrupt, "unreachable")? {
            Tok::Comma => {}
            Tok::Num(s) => vals.push(Val::Num(number(s))),
            Tok::Str(s) if name == "Content" => vals.push(Val::Raw(s)),
            Tok::Str(s) => vals.push(Val::Str(object_name(s))),
            Tok::Word(s) => vals.push(match s {
                "T" | "Y" => Val::Bool(true),
                "F" | "N" => Val::Bool(false),
                _ => Val::Str(Cow::Borrowed(s)),
            }),
            Tok::Star => {
                // `*N { a: v,v,… }`
                let n = match lex.next()? {
                    Some(Tok::Num(n)) => n.parse::<usize>().unwrap_or(0),
                    _ => bail!(Corrupt, "line {}: expected an array length", lex.line),
                };
                ensure!(
                    matches!(lex.next()?, Some(Tok::Open)),
                    Corrupt,
                    "line {}: expected an array",
                    lex.line
                );
                ensure!(
                    matches!(lex.next()?, Some(Tok::Name("a"))),
                    Corrupt,
                    "line {}: expected an array's values",
                    lex.line
                );
                let arr = lex.array(array_type(name), n)?;
                // Some exporters write `2,5` for 2.5, which can't be told from two numbers.
                ensure!(
                    arr.len() == n,
                    Corrupt,
                    "line {}: an array of {n} values holds {} (numbers written with decimal commas?)",
                    lex.line,
                    arr.len()
                );
                vals.push(Val::Arr(arr));
            }
            Tok::Name(_) | Tok::Open | Tok::Close => bail!(Corrupt, "line {}: unexpected token", lex.line),
        }
    }
}

/// How a number is stored in binary FBX.
#[derive(Clone, Copy, PartialEq)]
enum Ty {
    I32,
    I64,
    F32,
    F64,
    /// By its value: a 32-bit integer if it fits, otherwise 64-bit; a double if it has a point.
    Auto,
}

/// A `P` property's values, by the type it declares.
fn property_type(declared: &str) -> Ty {
    match declared {
        "int"
        | "Integer"
        | "enum"
        | "bool"
        | "Bool"
        | "Visibility Inheritance"
        | "Short"
        | "UShort"
        | "UInteger"
        | "ULong" => Ty::I32,
        "KTime" | "ULongLong" | "LongLong" => Ty::I64,
        "double" | "Number" | "Vector3D" | "Vector" | "Vector4D" | "ColorRGB" | "Color" | "ColorAndAlpha"
        | "ColorRGBA" | "Lcl Translation" | "Lcl Rotation" | "Lcl Scaling" | "Visibility" | "FieldOfView"
        | "FieldOfViewX" | "FieldOfViewY" | "Float" | "Real" | "Distance" | "Opacity" => Ty::F64,
        _ => Ty::Auto,
    }
}

/// An array node's element type, as binary FBX always stores it. Anything else is doubles.
fn array_type(name: &str) -> Ty {
    match name {
        "PolygonVertexIndex" | "Edges" | "UVIndex" | "NormalsIndex" | "NormalIndex" | "BinormalsIndex"
        | "BinormalIndex" | "TangentsIndex" | "TangentIndex" | "ColorIndex" | "Materials" | "Smoothing" | "Indexes"
        | "KeyAttrFlags" | "KeyAttrRefCount" | "TextureId" | "PolygonGroup" | "VertexCreaseIndex" | "PointsIndex" => {
            Ty::I32
        }
        "KeyTime" => Ty::I64,
        "KeyValueFloat" | "KeyAttrDataFloat" => Ty::F32,
        _ => Ty::F64,
    }
}

/// Node `name`'s values (taken from `vals`) as attributes typed the way binary FBX types them, into `out`. `parent`
/// decides how ids are typed.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "each value takes the width its property declares, as FBX readers expect"
)]
fn attributes<'a>(name: &str, parent: &str, vals: &mut Vec<Val<'a>>, out: &mut Vec<Attr<'a>>) {
    if name == "Content" {
        let joined: Vec<u8> = vals
            .iter()
            .filter_map(|v| if let Val::Raw(s) = v { Some(*s) } else { None })
            .flatten()
            .copied()
            .collect();
        let bytes = crate::gltf::decode_base64(&joined).unwrap_or_default();
        out.push(Attr::Binary(Cow::Owned(bytes)));
        vals.clear();
        return;
    }
    let declared = if name == "P" {
        match vals.get(1) {
            Some(Val::Str(t)) => property_type(t),
            _ => Ty::Auto,
        }
    } else {
        Ty::Auto
    };
    for (i, v) in vals.drain(..).enumerate() {
        let ty = if name == "P" && i >= 4 {
            declared
        } else if name == "C" || (parent == "Objects" && i == 0) {
            Ty::I64
        } else {
            Ty::Auto
        };
        out.push(match v {
            Val::Num(Num::Int(n)) => match ty {
                Ty::I32 => Attr::I32(n as i32),
                Ty::I64 => Attr::I64(n),
                Ty::F32 => Attr::F32(n as f32),
                Ty::F64 => Attr::F64(n as f64),
                Ty::Auto => i32::try_from(n).map_or(Attr::I64(n), Attr::I32),
            },
            Val::Num(Num::Float(f)) => match ty {
                Ty::I32 => Attr::I32(f as i32),
                Ty::I64 => Attr::I64(f as i64),
                Ty::F32 => Attr::F32(f as f32),
                Ty::F64 | Ty::Auto => Attr::F64(f),
            },
            Val::Str(s) => Attr::String(s),
            Val::Raw(s) => Attr::String(String::from_utf8_lossy(s)),
            Val::Bool(b) => Attr::Bool(b),
            Val::Arr(Values::I32(v)) => Attr::ArrI32(v.into()),
            Val::Arr(Values::I64(v)) => Attr::ArrI64(v.into()),
            Val::Arr(Values::F32(v)) => Attr::ArrF32(v.into()),
            Val::Arr(Values::F64(v)) => Attr::ArrF64(v.into()),
        });
    }
}

/// `tree` as binary FBX 7.4, arrays stored.
#[cfg(test)]
fn write(tree: &Tree<'_>) -> Result<Vec<u8>> {
    /// A node written but not yet closed: where its header is, and whether it has children or attributes.
    struct Open<'t> {
        header: usize,
        next: Option<NodeHandle<'t>>,
        children: bool,
        attrs: bool,
    }
    const NULL: [u8; 13] = [0; 13];
    let mut out = Vec::new();
    out.extend_from_slice(BINARY_MAGIC);
    out.extend_from_slice(&[0x1a, 0]);
    out.extend_from_slice(&7400u32.to_le_bytes());
    let offset = |n: usize| u32::try_from(n).map_err(|_| err!(LimitExceeded, "too big for binary FBX 7.4"));
    let mut open = vec![Open {
        header: 0,
        next: tree.root().first_child(),
        children: true,
        attrs: true,
    }];
    while let Some(top) = open.last_mut() {
        let Some(node) = top.next else {
            let done = open.pop().unwrap_or(Open {
                header: 0,
                next: None,
                children: false,
                attrs: false,
            });
            if open.is_empty() {
                break;
            }
            // The null node after children, and after a node with nothing at all, as the FBX SDK writes it.
            if done.children || !done.attrs {
                out.extend_from_slice(&NULL);
            }
            let end = offset(out.len())?;
            out[done.header..done.header + 4].copy_from_slice(&end.to_le_bytes());
            continue;
        };
        top.next = node.next_sibling();
        let header = out.len();
        let name = node.name().as_bytes();
        let name_len =
            u8::try_from(name.len()).map_err(|_| err!(LimitExceeded, "a node name is too long for binary FBX"))?;
        out.extend_from_slice(&[0; 12]);
        out.push(name_len);
        out.extend_from_slice(name);
        let start = out.len();
        let attrs = node.attributes();
        for a in attrs {
            attribute(&mut out, a)?;
        }
        let count = offset(attrs.len())?;
        let list = offset(out.len() - start)?;
        out[header + 4..header + 8].copy_from_slice(&count.to_le_bytes());
        out[header + 8..header + 12].copy_from_slice(&list.to_le_bytes());
        open.push(Open {
            header,
            next: node.first_child(),
            children: node.first_child().is_some(),
            attrs: !attrs.is_empty(),
        });
    }
    // The root's null node, then the footer the FBX SDK writes.
    out.extend_from_slice(&NULL);
    out.extend_from_slice(&[
        0xf0, 0xb1, 0xa2, 0x03, 0xd4, 0xc5, 0xd6, 0x67, 0xb8, 0x79, 0xfa, 0x8b, 0x1c, 0xfd, 0x2e, 0x7f,
    ]);
    out.resize(out.len().next_multiple_of(16), 0);
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&7400u32.to_le_bytes());
    out.extend_from_slice(&[0; 120]);
    out.extend_from_slice(&[
        0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e, 0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b,
    ]);
    Ok(out)
}

/// One attribute in binary FBX.
#[cfg(test)]
fn attribute(out: &mut Vec<u8>, a: &Attr<'_>) -> Result<()> {
    fn array<T: Elem>(out: &mut Vec<u8>, kind: u8, a: &Array<'_, T>) -> Result<()> {
        let v = a.to_vec().context(ErrorKind::Corrupt, "an array that can't be read")?;
        let bytes: &[u8] = bytemuck::cast_slice(&v);
        let len = |n: usize| u32::try_from(n).map_err(|_| err!(LimitExceeded, "an array too big for binary FBX"));
        out.push(kind);
        out.extend_from_slice(&len(v.len())?.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&len(bytes.len())?.to_le_bytes());
        if cfg!(target_endian = "big") {
            for x in &v {
                out.extend_from_slice(bytemuck::bytes_of(&x.to_native()));
            }
        } else {
            out.extend_from_slice(bytes);
        }
        Ok(())
    }
    let blob = |out: &mut Vec<u8>, kind: u8, b: &[u8]| -> Result<()> {
        out.push(kind);
        out.extend_from_slice(
            &u32::try_from(b.len())
                .map_err(|_| err!(LimitExceeded, "too big for binary FBX"))?
                .to_le_bytes(),
        );
        out.extend_from_slice(b);
        Ok(())
    };
    match a {
        Attr::Bool(b) => out.extend_from_slice(&[b'C', if *b { b'Y' } else { b'T' }]),
        Attr::I16(v) => {
            out.push(b'Y');
            out.extend_from_slice(&v.to_le_bytes());
        }
        Attr::I32(v) => {
            out.push(b'I');
            out.extend_from_slice(&v.to_le_bytes());
        }
        Attr::I64(v) => {
            out.push(b'L');
            out.extend_from_slice(&v.to_le_bytes());
        }
        Attr::F32(v) => {
            out.push(b'F');
            out.extend_from_slice(&v.to_le_bytes());
        }
        Attr::F64(v) => {
            out.push(b'D');
            out.extend_from_slice(&v.to_le_bytes());
        }
        Attr::ArrBool(v) => array(out, b'b', v)?,
        Attr::ArrI32(v) => array(out, b'i', v)?,
        Attr::ArrI64(v) => array(out, b'l', v)?,
        Attr::ArrF32(v) => array(out, b'f', v)?,
        Attr::ArrF64(v) => array(out, b'd', v)?,
        Attr::Binary(b) => blob(out, b'R', b)?,
        Attr::String(s) => blob(out, b'S', s.as_bytes())?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fbxcel::low::v7400::AttributeValue as A;

    const QUAD: &str = r#"; FBX 7.3.0 project file
; ----------------------------------------------------
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7300
	Creator: "Made using Asset Forge (www.assetforge.io)"
	SceneInfo: "SceneInfo::GlobalInfo", "UserData" {
		Type: "UserData"
	}
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",100
		P: "TimeSpanStop", "KTime", "Time", "",479181389250
	}
}
Objects:  {
	Geometry: 2295830373648, "Geometry::Quad", "Mesh" {
		Vertices: *12 {
			a: 0,0,0,100,0,0,
100,100,0,0,100,0
		}
		PolygonVertexIndex: *4 {
			a: 0,1,2,-4
		}
		LayerElementNormal: 0 {
			Version: 101
			MappingInformationType: "ByPolygonVertex"
			Normals: *0 {
				a:
			}
		}
	}
	Model: 20, "Model::Floor", "Mesh" {
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",0,0,1.5e2
		}
		Shading: T
		Culling: "CullingOff"
	}
	Video: 50, "Video::Tex", "Clip" {
		Content: , "aGVs",
		  "bG8="
	}
}
Connections:  {
	;Geometry::Quad, Model::Floor
	C: "OO",2295830373648,20
	C: "OO",20,0
}
"#;

    #[test]
    fn ascii_becomes_binary_with_binary_types() {
        assert!(is_ascii(QUAD.as_bytes()));
        let bin = to_binary(QUAD.as_bytes()).unwrap();
        assert!(!is_ascii(&bin));
        let fbxcel::tree::any::AnyTree::V7400(_, tree, _) =
            fbxcel::tree::any::AnyTree::from_seekable_reader(std::io::Cursor::new(bin)).unwrap()
        else {
            panic!("not 7.4")
        };
        let root = tree.root();
        let info = root
            .first_child_by_name("FBXHeaderExtension")
            .unwrap()
            .first_child_by_name("SceneInfo")
            .unwrap();
        assert_eq!(info.attributes()[0], A::String("GlobalInfo\0\x01SceneInfo".into()));
        let p70 = root
            .first_child_by_name("GlobalSettings")
            .unwrap()
            .first_child_by_name("Properties70")
            .unwrap();
        let props: Vec<_> = p70.children_by_name("P").map(|p| p.attributes()[4].clone()).collect();
        assert_eq!(props, [A::I32(1), A::F64(100.0), A::I64(479_181_389_250)]);

        let objects = root.first_child_by_name("Objects").unwrap();
        let geom = objects.first_child_by_name("Geometry").unwrap();
        assert_eq!(geom.attributes()[0], A::I64(2_295_830_373_648));
        assert_eq!(geom.attributes()[1], A::String("Quad\0\x01Geometry".into()));
        let verts = geom.first_child_by_name("Vertices").unwrap();
        assert_eq!(
            verts.attributes()[0],
            A::ArrF64(vec![0., 0., 0., 100., 0., 0., 100., 100., 0., 0., 100., 0.])
        );
        let idx = geom.first_child_by_name("PolygonVertexIndex").unwrap();
        assert_eq!(idx.attributes()[0], A::ArrI32(vec![0, 1, 2, -4]));
        let normals = geom.first_child_by_name("LayerElementNormal").unwrap();
        assert_eq!(normals.attributes()[0], A::I32(0));
        assert_eq!(
            normals.first_child_by_name("Normals").unwrap().attributes()[0],
            A::ArrF64(vec![])
        );
        let model = objects.first_child_by_name("Model").unwrap();
        let t = model
            .first_child_by_name("Properties70")
            .unwrap()
            .first_child_by_name("P")
            .unwrap();
        assert_eq!(&t.attributes()[4..], [A::F64(0.0), A::F64(0.0), A::F64(150.0)]);
        assert_eq!(
            model.first_child_by_name("Shading").unwrap().attributes()[0],
            A::Bool(true)
        );
        let content = objects
            .first_child_by_name("Video")
            .unwrap()
            .first_child_by_name("Content")
            .unwrap();
        assert_eq!(content.attributes()[0], A::Binary(b"hello".to_vec()));
        let c: Vec<_> = root
            .first_child_by_name("Connections")
            .unwrap()
            .children()
            .map(|c| c.attributes().to_vec())
            .collect();
        assert_eq!(c[0], [A::String("OO".into()), A::I64(2_295_830_373_648), A::I64(20)]);
    }

    #[test]
    fn fbx_6_is_refused_by_name() {
        let err = to_binary(b"; FBX 6.1.0 project file\nFBXHeaderExtension:  {\n}\n").unwrap_err();
        assert!(format!("{err}").contains("FBX 6"), "{err}");
    }
}
