//! `OpenDDL` 3.0, the text syntax `OpenGEX` is written in: a tree of structures, each either a
//! custom structure (an identifier, an optional name, optional properties and the structures
//! inside its braces) or a primitive structure (a data type, an optional array size, an
//! optional name and a list of literals, flat or in subarrays of that size).
//!
//! The whole file is read into one tree: structures in an arena linked parent, first child and
//! next sibling, their properties beside them, and the literals of every primitive structure
//! in shared arrays (numbers as `f64`, strings, references), so a mesh's thousands of values are
//! one run in one array and nothing is allocated per value. Names are indexed for references:
//! global names by name, local names by the structure that holds them.

use crate::Result;
use crate::formats::stream::{f32_from, f64_from};
use crate::mesh::FastMap;

/// How deep custom structures may nest.
const MAX_DEPTH: usize = 256;
/// The index of the tree's root, the file itself, which holds the top-level structures.
pub(super) const ROOT: u32 = 0;
/// No structure.
const NONE: u32 = u32::MAX;

/// A primitive data type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ty {
    Bool,
    /// Any signed or unsigned integer type.
    Int,
    Half,
    Float,
    Double,
    Str,
    Ref,
    Type,
    Base64,
}

/// The data type an identifier names, if it names one.
fn data_type(id: &[u8]) -> Option<Ty> {
    Some(match id {
        b"bool" | b"b" => Ty::Bool,
        b"int8" | b"i8" | b"int16" | b"i16" | b"int32" | b"i32" | b"int64" | b"i64" | b"uint8" | b"u8" | b"uint16"
        | b"u16" | b"uint32" | b"u32" | b"uint64" | b"u64" | b"unsigned_int8" | b"unsigned_int16"
        | b"unsigned_int32" | b"unsigned_int64" => Ty::Int,
        b"half" | b"h" | b"float16" | b"f16" => Ty::Half,
        b"float" | b"f" | b"float32" | b"f32" => Ty::Float,
        b"double" | b"d" | b"float64" | b"f64" => Ty::Double,
        b"string" | b"s" => Ty::Str,
        b"ref" | b"r" => Ty::Ref,
        b"type" | b"t" => Ty::Type,
        b"base64" | b"z" => Ty::Base64,
        _ => return None,
    })
}

/// A reference: names, the first global (`$`) or local (`%`), the rest local; none for `null`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Ref {
    pub global: bool,
    /// Each name's span in the file, without its `$` or `%`.
    parts: Vec<(u32, u32)>,
}

/// A property's value.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Val {
    Bool(bool),
    Num(f64),
    Str(String),
    Ref(Ref),
    Type,
}

/// A structure.
#[derive(Clone, Debug)]
pub(super) struct Node {
    /// The identifier's span, or the data type's for a primitive structure.
    ident: (u32, u32),
    /// The data type of a primitive structure.
    pub ty: Option<Ty>,
    /// Whether the name is global, and its span without its `$` or `%`.
    name: Option<(bool, u32, u32)>,
    pub parent: u32,
    first: u32,
    last: u32,
    next: u32,
    /// The range of its properties.
    props: (u32, u32),
    /// A primitive structure's subarray size; 0 for a flat list.
    pub width: u32,
    /// A primitive structure's literals: a range of the numbers, strings or references.
    data: (u32, u32),
}

/// A parsed file.
pub(super) struct Tree<'a> {
    src: &'a [u8],
    pub nodes: Vec<Node>,
    props: Vec<((u32, u32), Val)>,
    nums: Vec<f64>,
    strs: Vec<String>,
    refs: Vec<Ref>,
    globals: FastMap<&'a [u8], u32>,
    locals: FastMap<(u32, &'a [u8]), u32>,
    /// Subarrays with a state identifier in front, which carries nothing `OpenGEX` reads.
    pub states: usize,
}

/// A cursor over the file's bytes.
struct Lex<'a> {
    b: &'a [u8],
    at: usize,
}

impl Lex<'_> {
    /// Past whitespace and comments.
    fn skip(&mut self) -> Result<()> {
        loop {
            while self.b.get(self.at).is_some_and(|&c| c <= b' ') {
                self.at += 1;
            }
            if self.b.get(self.at) == Some(&b'/') {
                match self.b.get(self.at + 1) {
                    Some(b'/') => {
                        self.at += 2;
                        while self.b.get(self.at).is_some_and(|&c| c != b'\n') {
                            self.at += 1;
                        }
                        continue;
                    }
                    Some(b'*') => {
                        let rest = &self.b[self.at + 2..];
                        let Some(end) = rest.windows(2).position(|w| w == b"*/") else {
                            bail!(Truncated, "OpenDDL file is cut short inside a comment")
                        };
                        self.at += end + 4;
                        continue;
                    }
                    _ => {}
                }
            }
            return Ok(());
        }
    }

    /// The next byte after whitespace and comments.
    fn peek(&mut self) -> Result<Option<u8>> {
        self.skip()?;
        Ok(self.b.get(self.at).copied())
    }

    /// Takes `c` if it's next.
    fn eat(&mut self, c: u8) -> Result<bool> {
        let hit = self.peek()? == Some(c);
        if hit {
            self.at += 1;
        }
        Ok(hit)
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if !self.eat(c)? {
            bail!(
                Corrupt,
                "expected '{}' at byte {} of an OpenDDL file",
                char::from(c),
                self.at
            );
        }
        Ok(())
    }

    /// The identifier at the cursor, as its span.
    fn ident(&mut self) -> Option<(u32, u32)> {
        let start = self.at;
        let first = *self.b.get(start)?;
        if !(first.is_ascii_alphabetic() || first == b'_') {
            return None;
        }
        let mut end = start + 1;
        while self.b.get(end).is_some_and(|&c| c.is_ascii_alphanumeric() || c == b'_') {
            end += 1;
        }
        self.at = end;
        Some((span(start), span(end - start)))
    }

    /// A name at the cursor: whether it's global, and its identifier's span.
    fn name(&mut self) -> Result<Option<(bool, u32, u32)>> {
        let global = match self.peek()? {
            Some(b'$') => true,
            Some(b'%') => false,
            _ => return Ok(None),
        };
        self.at += 1;
        let Some((s, n)) = self.ident() else {
            bail!(
                Corrupt,
                "a name without an identifier at byte {} of an OpenDDL file",
                self.at
            )
        };
        Ok(Some((global, s, n)))
    }

    /// A reference: `null`, or names joined by `%`.
    fn reference(&mut self) -> Result<Ref> {
        if self.peek()? == Some(b'n') {
            let at = self.at;
            if self.ident().map(|(s, n)| &self.b[s as usize..(s + n) as usize]) == Some(b"null") {
                return Ok(Ref::default());
            }
            self.at = at;
        }
        let Some((global, s, n)) = self.name()? else {
            bail!(Corrupt, "expected a reference at byte {} of an OpenDDL file", self.at)
        };
        let mut parts = vec![(s, n)];
        while self.b.get(self.at) == Some(&b'%') {
            self.at += 1;
            let Some(p) = self.ident() else {
                bail!(
                    Corrupt,
                    "a reference ends in '%' at byte {} of an OpenDDL file",
                    self.at
                )
            };
            parts.push(p);
        }
        Ok(Ref { global, parts })
    }

    /// One or more string literals, joined.
    fn string(&mut self) -> Result<String> {
        let mut out = Vec::new();
        let mut any = false;
        while self.peek()? == Some(b'"') {
            any = true;
            self.at += 1;
            loop {
                let Some(&c) = self.b.get(self.at) else {
                    bail!(Truncated, "OpenDDL file is cut short inside a string")
                };
                self.at += 1;
                match c {
                    b'"' => break,
                    b'\\' => self.escape(&mut out)?,
                    _ => out.push(c),
                }
            }
        }
        ensure!(any, Corrupt, "expected a string at byte {} of an OpenDDL file", self.at);
        Ok(String::from_utf8(out).unwrap_or_else(|e| crate::bytes::text(e.as_bytes()).into_owned()))
    }

    /// The character an escape sequence (after its backslash) stands for, into `out` as UTF-8.
    fn escape(&mut self, out: &mut Vec<u8>) -> Result<()> {
        let Some(&c) = self.b.get(self.at) else {
            bail!(Corrupt, "an OpenDDL string isn't closed")
        };
        self.at += 1;
        let digits = match c {
            b'"' | b'\'' | b'?' | b'\\' => {
                out.push(c);
                return Ok(());
            }
            b'a' | b'b' | b'f' | b'n' | b'r' | b't' | b'v' => {
                out.push(match c {
                    b'a' => 7,
                    b'b' => 8,
                    b'f' => 12,
                    b'n' => b'\n',
                    b'r' => b'\r',
                    b't' => b'\t',
                    _ => 11,
                });
                return Ok(());
            }
            b'x' => 2,
            b'u' => 4,
            b'U' => 6,
            _ => bail!(Corrupt, "an unknown escape '\\{}' in an OpenDDL string", char::from(c)),
        };
        let hex = self.b.get(self.at..self.at + digits).unwrap_or_default();
        let v = std::str::from_utf8(hex)
            .ok()
            .and_then(|h| u32::from_str_radix(h, 16).ok());
        let Some(v) = v.filter(|_| hex.len() == digits) else {
            bail!(Corrupt, "a bad escape in an OpenDDL string")
        };
        self.at += digits;
        let ch = char::from_u32(v).unwrap_or(char::REPLACEMENT_CHARACTER);
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        Ok(())
    }

    /// A number literal's parts at the cursor: its sign, its base (2, 8, 10 or 16, or 0 for a
    /// character literal) and its digits' span, underscores and all.
    fn number(&mut self) -> Result<(bool, u32, usize, usize)> {
        self.skip()?;
        let mut neg = false;
        if let Some(&s @ (b'+' | b'-')) = self.b.get(self.at) {
            neg = s == b'-';
            self.at += 1;
        }
        let start = self.at;
        let at = |k: usize| self.b.get(start + k).copied().unwrap_or(0);
        if at(0) == b'\'' {
            let mut end = start + 1;
            while self.b.get(end).is_some_and(|&c| c != b'\'') {
                end += if self.b[end] == b'\\' { 2 } else { 1 };
            }
            ensure!(end < self.b.len(), Corrupt, "an OpenDDL character literal isn't closed");
            self.at = end + 1;
            return Ok((neg, 0, start + 1, end));
        }
        let base = match (at(0), at(1) | 0x20) {
            (b'0', b'x') => 16,
            (b'0', b'o') => 8,
            (b'0', b'b') => 2,
            _ => 10,
        };
        let mut end = start;
        if base == 10 {
            let digits = |end: &mut usize| {
                while self.b.get(*end).is_some_and(|&c| c.is_ascii_digit() || c == b'_') {
                    *end += 1;
                }
            };
            digits(&mut end);
            if self.b.get(end) == Some(&b'.') {
                end += 1;
                digits(&mut end);
            }
            if self.b.get(end).is_some_and(|&c| c | 0x20 == b'e') {
                end += 1;
                if self.b.get(end).is_some_and(|&c| c == b'+' || c == b'-') {
                    end += 1;
                }
                digits(&mut end);
            }
        } else {
            end += 2;
            while self.b.get(end).is_some_and(|&c| c.is_ascii_hexdigit() || c == b'_') {
                end += 1;
            }
        }
        let body = &self.b[start..end];
        ensure!(
            body.iter().any(u8::is_ascii_digit),
            Corrupt,
            "expected a number at byte {} of an OpenDDL file",
            start
        );
        self.at = end;
        Ok((neg, base, start, end))
    }
}

/// A position or length in the file as a span's `u32`; files are checked to fit.
fn span(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// A number's digits without their underscores, into `buf` when it has any.
fn plain<'b>(digits: &'b [u8], buf: &'b mut Vec<u8>) -> &'b [u8] {
    if digits.contains(&b'_') {
        buf.clear();
        buf.extend(digits.iter().filter(|&&c| c != b'_'));
        buf
    } else {
        digits
    }
}

/// The integer value of digits in `base` (2, 8 or 16, after the prefix; or 0 for a character
/// literal's characters, big-endian bytes).
fn integer(digits: &[u8], base: u32) -> Option<u64> {
    let mut v = 0u64;
    if base == 0 {
        let mut k = 0;
        while k < digits.len() {
            let mut c = digits[k];
            if c == b'\\' {
                k += 1;
                c = match digits.get(k)? {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    b'0' => 0,
                    &c => c,
                };
            }
            v = v.checked_mul(256)?.checked_add(u64::from(c))?;
            k += 1;
        }
        return Some(v);
    }
    let mut any = false;
    for &c in digits {
        if c == b'_' {
            continue;
        }
        let d = u64::from(char::from(c).to_digit(base)?);
        v = v.checked_mul(u64::from(base))?.checked_add(d)?;
        any = true;
    }
    any.then_some(v)
}

/// A half-precision float's value from its bits.
fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1F);
    let frac = f64::from(bits & 0x3FF);
    sign * match exp {
        0 => frac * 2f64.powi(-24),
        31 if frac == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

impl<'a> Tree<'a> {
    /// Parses a whole file.
    pub fn parse(src: &'a [u8]) -> Result<Self> {
        ensure!(
            u32::try_from(src.len()).is_ok_and(|n| n < u32::MAX),
            LimitExceeded,
            "the OpenDDL file is too big"
        );
        let mut t = Tree {
            src,
            nodes: vec![Node {
                ident: (0, 0),
                ty: None,
                name: None,
                parent: NONE,
                first: NONE,
                last: NONE,
                next: NONE,
                props: (0, 0),
                width: 0,
                data: (0, 0),
            }],
            props: Vec::new(),
            nums: Vec::new(),
            strs: Vec::new(),
            refs: Vec::new(),
            globals: FastMap::default(),
            locals: FastMap::default(),
            states: 0,
        };
        let mut lex = Lex { b: src, at: 0 };
        // A UTF-8 byte order mark.
        if src.starts_with(&[0xEF, 0xBB, 0xBF]) {
            lex.at = 3;
        }
        let mut open: Vec<u32> = Vec::new();
        let mut buf = Vec::new();
        while let Some(c) = lex.peek()? {
            if c == b'}' {
                ensure!(
                    open.pop().is_some(),
                    Corrupt,
                    "an unmatched '}}' at byte {} of an OpenDDL file",
                    lex.at
                );
                lex.at += 1;
                continue;
            }
            let parent = open.last().copied().unwrap_or(ROOT);
            let Some(ident) = lex.ident() else {
                bail!(
                    Corrupt,
                    "expected a structure at byte {} of an OpenDDL file, not {}",
                    lex.at,
                    crate::shown(&src[lex.at..(lex.at + 16).min(src.len())])
                )
            };
            let ty = data_type(&src[ident.0 as usize..(ident.0 + ident.1) as usize]);
            let mut node = Node {
                ident,
                ty,
                name: None,
                parent,
                first: NONE,
                last: NONE,
                next: NONE,
                props: (0, 0),
                width: 0,
                data: (0, 0),
            };
            if let Some(ty) = ty {
                let mut states = false;
                if lex.eat(b'[')? {
                    let (neg, base, s, e) = lex.number()?;
                    let n = if base == 10 {
                        std::str::from_utf8(plain(&src[s..e], &mut buf))
                            .ok()
                            .and_then(|d| d.parse().ok())
                    } else {
                        integer(&src[s + 2..e], base)
                    };
                    let n = n.and_then(|n: u64| u32::try_from(n).ok()).filter(|&n| n > 0 && !neg);
                    let Some(n) = n else {
                        bail!(Corrupt, "a bad array size in an OpenDDL file")
                    };
                    node.width = n;
                    lex.expect(b']')?;
                    states = lex.eat(b'*')?;
                }
                node.name = lex.name()?;
                lex.expect(b'{')?;
                node.data = t.data(&mut lex, ty, node.width, states, &mut buf)?;
                lex.expect(b'}')?;
            } else {
                node.name = lex.name()?;
                if lex.eat(b'(')? {
                    let from = span(t.props.len());
                    if !lex.eat(b')')? {
                        loop {
                            lex.skip()?;
                            let Some(key) = lex.ident() else {
                                bail!(Corrupt, "expected a property in an OpenDDL file")
                            };
                            let v = if lex.eat(b'=')? {
                                t.value(&mut lex, &mut buf)?
                            } else {
                                Val::Bool(true)
                            };
                            t.props.push((key, v));
                            if lex.eat(b')')? {
                                break;
                            }
                            lex.expect(b',')?;
                        }
                    }
                    node.props = (from, span(t.props.len()) - from);
                }
                lex.expect(b'{')?;
            }
            let k = span(t.nodes.len());
            ensure!(k < NONE, LimitExceeded, "too many OpenDDL structures");
            if let Some((global, s, n)) = node.name {
                let name = &src[s as usize..(s + n) as usize];
                if global {
                    t.globals.entry(name).or_insert(k);
                } else {
                    t.locals.entry((parent, name)).or_insert(k);
                }
            }
            t.nodes.push(node);
            let p = &mut t.nodes[parent as usize];
            if p.last == NONE {
                p.first = k;
            } else {
                let last = p.last;
                t.nodes[last as usize].next = k;
            }
            t.nodes[parent as usize].last = k;
            if ty.is_none() {
                open.push(k);
                ensure!(
                    open.len() <= MAX_DEPTH,
                    LimitExceeded,
                    "OpenDDL structures nest too deep"
                );
            }
        }
        ensure!(
            open.is_empty(),
            Truncated,
            "OpenDDL file is cut short: a structure isn't closed"
        );
        Ok(t)
    }

    /// A property's value.
    fn value(&mut self, lex: &mut Lex<'_>, buf: &mut Vec<u8>) -> Result<Val> {
        Ok(match lex.peek()? {
            Some(b'"') => Val::Str(lex.string()?),
            Some(b'$' | b'%') => Val::Ref(lex.reference()?),
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => {
                let at = lex.at;
                let id = lex
                    .ident()
                    .map_or(&b""[..], |(s, n)| &self.src[s as usize..(s + n) as usize]);
                match id {
                    b"true" => Val::Bool(true),
                    b"false" => Val::Bool(false),
                    b"null" => Val::Ref(Ref::default()),
                    _ if data_type(id).is_some() => Val::Type,
                    _ => bail!(Corrupt, "a bad property value at byte {at} of an OpenDDL file"),
                }
            }
            _ => {
                let (neg, base, s, e) = lex.number()?;
                // A property's type isn't declared: hex, octal and binary numbers there are
                // integers, not a float's bits.
                let ty = if base == 10 { Ty::Double } else { Ty::Int };
                let v = self.literal_number(ty, neg, base, s, e, buf)?;
                Val::Num(v)
            }
        })
    }

    /// A number literal's value as `ty` holds it.
    #[allow(
        clippy::cast_precision_loss,
        reason = "integers past 2^53 don't index anything OpenGEX has"
    )]
    fn literal_number(&self, ty: Ty, neg: bool, base: u32, s: usize, e: usize, buf: &mut Vec<u8>) -> Result<f64> {
        let digits = &self.src[s..e];
        let v = if base == 10 {
            let d = plain(digits, buf);
            let v = if ty == Ty::Float || ty == Ty::Half {
                f32_from(d).map(f64::from)
            } else {
                f64_from(d)
            };
            let Some(v) = v else {
                bail!(Corrupt, "a bad number {} in an OpenDDL file", crate::shown(digits))
            };
            v
        } else {
            let digits = if base == 0 { digits } else { &digits[2..] };
            let Some(bits) = integer(digits, base) else {
                bail!(Corrupt, "a bad number {} in an OpenDDL file", crate::shown(digits))
            };
            match ty {
                // Floats written in hex, octal or binary are their bits.
                Ty::Half => half(u16::try_from(bits).unwrap_or(0)),
                Ty::Float => f64::from(f32::from_bits(u32::try_from(bits).unwrap_or(0))),
                Ty::Double if base != 0 => f64::from_bits(bits),
                _ => bits as f64,
            }
        };
        Ok(if neg { -v } else { v })
    }

    /// A primitive structure's literals (the cursor inside its braces), into the shared
    /// arrays: their range.
    fn data(&mut self, lex: &mut Lex<'_>, ty: Ty, width: u32, states: bool, buf: &mut Vec<u8>) -> Result<(u32, u32)> {
        let from = match ty {
            Ty::Str | Ty::Type | Ty::Base64 => self.strs.len(),
            Ty::Ref => self.refs.len(),
            _ => self.nums.len(),
        };
        if lex.peek()? != Some(b'}') {
            loop {
                if width == 0 {
                    self.literal(lex, ty, buf)?;
                } else {
                    if states && lex.peek()?.is_some_and(|c| c.is_ascii_alphabetic() || c == b'_') {
                        lex.ident();
                        self.states += 1;
                    }
                    lex.expect(b'{')?;
                    for k in 0..width {
                        if k > 0 {
                            lex.expect(b',')?;
                        }
                        self.literal(lex, ty, buf)?;
                    }
                    lex.expect(b'}')?;
                }
                if !lex.eat(b',')? {
                    break;
                }
            }
        }
        let to = match ty {
            Ty::Str | Ty::Type | Ty::Base64 => self.strs.len(),
            Ty::Ref => self.refs.len(),
            _ => self.nums.len(),
        };
        Ok((span(from), span(to - from)))
    }

    /// One literal of `ty`, into its array.
    fn literal(&mut self, lex: &mut Lex<'_>, ty: Ty, buf: &mut Vec<u8>) -> Result<()> {
        match ty {
            Ty::Str => {
                let s = lex.string()?;
                self.strs.push(s);
            }
            Ty::Ref => {
                let r = lex.reference()?;
                self.refs.push(r);
            }
            Ty::Type => {
                lex.skip()?;
                let Some((s, n)) = lex.ident() else {
                    bail!(Corrupt, "expected a data type in an OpenDDL file")
                };
                let id = &self.src[s as usize..(s + n) as usize];
                self.strs.push(crate::bytes::text(id).into_owned());
            }
            Ty::Base64 => {
                lex.skip()?;
                let start = lex.at;
                while lex
                    .b
                    .get(lex.at)
                    .is_some_and(|&c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'/' | b'='))
                {
                    lex.at += 1;
                }
                self.strs
                    .push(crate::bytes::text(&self.src[start..lex.at]).into_owned());
            }
            Ty::Bool => {
                lex.skip()?;
                let at = lex.at;
                let v = match lex.ident().map(|(s, n)| &self.src[s as usize..(s + n) as usize]) {
                    Some(b"true") => 1.0,
                    Some(b"false") => 0.0,
                    Some(_) => bail!(Corrupt, "a bad boolean at byte {at} of an OpenDDL file"),
                    None => {
                        let (neg, base, s, e) = lex.number()?;
                        let v = self.literal_number(Ty::Int, neg, base, s, e, buf)?;
                        f64::from(u8::from(v != 0.0))
                    }
                };
                self.nums.push(v);
            }
            Ty::Int | Ty::Half | Ty::Float | Ty::Double => {
                let (neg, base, s, e) = lex.number()?;
                if ty == Ty::Int && base == 10 {
                    ensure!(
                        !self.src[s..e].iter().any(|&c| matches!(c, b'.' | b'e' | b'E')),
                        Corrupt,
                        "a fraction in an OpenDDL integer"
                    );
                }
                let v = self.literal_number(ty, neg, base, s, e, buf)?;
                self.nums.push(v);
            }
        }
        Ok(())
    }

    fn text(&self, s: u32, n: u32) -> &'a str {
        // Identifiers are ASCII, checked as they were read.
        std::str::from_utf8(&self.src[s as usize..(s + n) as usize]).unwrap_or_default()
    }

    /// A structure's identifier, or its data type's name.
    pub fn ident(&self, i: u32) -> &'a str {
        let (s, n) = self.nodes[i as usize].ident;
        self.text(s, n)
    }

    /// A structure's name, without its `$` or `%`.
    pub fn name(&self, i: u32) -> Option<&'a str> {
        self.nodes[i as usize].name.map(|(_, s, n)| self.text(s, n))
    }

    /// The structures inside `i`, in order.
    pub fn kids(&self, i: u32) -> Kids<'_, 'a> {
        Kids {
            t: self,
            at: self.nodes[i as usize].first,
        }
    }

    /// The first structure inside `i` with identifier `id`.
    pub fn kid(&self, i: u32, id: &str) -> Option<u32> {
        self.kids(i).find(|&k| self.ident(k) == id)
    }

    /// The value of `i`'s property `key`: the last when it's given more than once.
    pub fn prop(&self, i: u32, key: &str) -> Option<&Val> {
        let (from, n) = self.nodes[i as usize].props;
        self.props[from as usize..(from + n) as usize]
            .iter()
            .rev()
            .find(|((s, len), _)| self.text(*s, *len) == key)
            .map(|(_, v)| v)
    }

    /// A string property.
    pub fn prop_str(&self, i: u32, key: &str) -> Option<&str> {
        match self.prop(i, key)? {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }

    /// A number property (a boolean as 0 or 1).
    pub fn prop_num(&self, i: u32, key: &str) -> Option<f64> {
        match *self.prop(i, key)? {
            Val::Num(v) => Some(v),
            Val::Bool(b) => Some(f64::from(u8::from(b))),
            _ => None,
        }
    }

    /// A boolean property (a number as whether it isn't 0).
    pub fn prop_bool(&self, i: u32, key: &str) -> Option<bool> {
        match *self.prop(i, key)? {
            Val::Bool(b) => Some(b),
            Val::Num(v) => Some(v != 0.0),
            _ => None,
        }
    }

    /// A reference property.
    pub fn prop_ref(&self, i: u32, key: &str) -> Option<&Ref> {
        match self.prop(i, key)? {
            Val::Ref(r) => Some(r),
            _ => None,
        }
    }

    /// The first primitive structure inside `i`.
    pub fn data_of(&self, i: u32) -> Option<u32> {
        self.kids(i).find(|&k| self.nodes[k as usize].ty.is_some())
    }

    /// A numeric primitive structure's values; empty for any other.
    pub fn nums(&self, i: u32) -> &[f64] {
        let n = &self.nodes[i as usize];
        match n.ty {
            Some(Ty::Bool | Ty::Int | Ty::Half | Ty::Float | Ty::Double) => {
                &self.nums[n.data.0 as usize..(n.data.0 + n.data.1) as usize]
            }
            _ => &[],
        }
    }

    /// A string primitive structure's strings; empty for any other.
    pub fn strs(&self, i: u32) -> &[String] {
        let n = &self.nodes[i as usize];
        match n.ty {
            Some(Ty::Str) => &self.strs[n.data.0 as usize..(n.data.0 + n.data.1) as usize],
            _ => &[],
        }
    }

    /// A reference primitive structure's references; empty for any other.
    pub fn refs(&self, i: u32) -> &[Ref] {
        let n = &self.nodes[i as usize];
        match n.ty {
            Some(Ty::Ref) => &self.refs[n.data.0 as usize..(n.data.0 + n.data.1) as usize],
            _ => &[],
        }
    }

    /// The numbers of the first primitive structure inside `i` and its subarray size.
    pub fn values(&self, i: u32) -> (&[f64], u32) {
        self.data_of(i)
            .map_or((&[][..], 0), |d| (self.nums(d), self.nodes[d as usize].width))
    }

    /// The first string inside `i`.
    pub fn string(&self, i: u32) -> Option<&str> {
        self.data_of(i).and_then(|d| self.strs(d).first()).map(String::as_str)
    }

    /// The first reference inside `i`.
    pub fn reference(&self, i: u32) -> Option<&Ref> {
        self.data_of(i).and_then(|d| self.refs(d).first())
    }

    /// The structure reference `r` names, seen from structure `from`: a global name looked up,
    /// a local one among the structures inside `from`, then inside its parent, and so on out to
    /// the top level; each further name among the structures inside the last.
    pub fn resolve(&self, from: u32, r: &Ref) -> Option<u32> {
        let (&(s, n), rest) = r.parts.split_first()?;
        let first = &self.src[s as usize..(s + n) as usize];
        let mut at = if r.global {
            *self.globals.get(first)?
        } else {
            let mut scope = from;
            loop {
                if let Some(&k) = self.locals.get(&(scope, first)) {
                    break k;
                }
                if scope == ROOT || scope == NONE {
                    return None;
                }
                scope = self.nodes[scope as usize].parent;
            }
        };
        for &(s, n) in rest {
            at = *self.locals.get(&(at, &self.src[s as usize..(s + n) as usize]))?;
        }
        Some(at)
    }
}

/// The structures inside one, in order.
pub(super) struct Kids<'t, 'a> {
    t: &'t Tree<'a>,
    at: u32,
}

impl Iterator for Kids<'_, '_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.at == NONE {
            return None;
        }
        let k = self.at;
        self.at = self.t.nodes[k as usize].next;
        Some(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structures_properties_and_literals() {
        let src = br#"
            // A comment.
            Metric (key = "distance") {float {0.01}}
            /* A block
               comment. */
            Thing $a (flag, n = 0x10, s = "x" "y", r = %b, t = float, f = -1.5e1)
            {
                float[2] %b { {1, 0x3F800000}, {-0b0, 2_000.5} }
                double {0x3FF0000000000000, 1.25}
                half {0x3C00, 1}
                unsigned_int16 {1, 0o17, 'A', -3}
                bool {true, false, 1}
                string {"tab\t\"q\" \x41\u00e9"}
                ref {$a%b, null}
                Inner {}
            }
        "#;
        let t = Tree::parse(src).unwrap();
        let top: Vec<u32> = t.kids(ROOT).collect();
        assert_eq!(top.len(), 2);
        assert_eq!(t.ident(top[0]), "Metric");
        assert_eq!(t.prop_str(top[0], "key"), Some("distance"));
        assert_eq!(t.values(top[0]).0, &[f64::from(0.01f32)]);
        let a = top[1];
        assert_eq!(t.name(a), Some("a"));
        assert_eq!(t.prop_bool(a, "flag"), Some(true));
        assert_eq!(t.prop_num(a, "n"), Some(16.0));
        assert_eq!(t.prop_str(a, "s"), Some("xy"));
        assert_eq!(t.prop_num(a, "f"), Some(-15.0));
        assert_eq!(t.prop(a, "t"), Some(&Val::Type));
        let kids: Vec<u32> = t.kids(a).collect();
        assert_eq!(kids.len(), 8);
        assert_eq!(t.nodes[kids[0] as usize].width, 2);
        assert_eq!(t.nums(kids[0]), &[1.0, 1.0, -0.0, 2000.5]);
        assert_eq!(t.nums(kids[1]), &[1.0, 1.25]);
        assert_eq!(t.nums(kids[2]), &[1.0, 1.0]);
        assert_eq!(t.nums(kids[3]), &[1.0, 15.0, 65.0, -3.0]);
        assert_eq!(t.nums(kids[4]), &[1.0, 0.0, 1.0]);
        assert_eq!(t.strs(kids[5]), &["tab\t\"q\" A\u{e9}".to_owned()]);
        let refs = t.refs(kids[6]);
        assert_eq!(t.resolve(kids[6], &refs[0]), Some(kids[0]));
        assert_eq!(t.resolve(kids[6], &refs[1]), None);
        let r = t.prop_ref(a, "r").unwrap();
        assert_eq!(t.resolve(a, r), Some(kids[0]));
        assert_eq!(t.resolve(kids[7], r), Some(kids[0]));
        assert_eq!(t.resolve(ROOT, r), None);
    }

    #[test]
    fn states_and_last_property_wins() {
        let t = Tree::parse(b"X (a = 1, a = 2) { int32[2]* {s {1, 2}, {3, 4}} }").unwrap();
        let x = t.kids(ROOT).next().unwrap();
        assert_eq!(t.prop_num(x, "a"), Some(2.0));
        assert_eq!(t.values(x).0, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.states, 1);
    }

    #[test]
    fn broken_files_fail_without_panicking() {
        for src in [
            &b"X {"[..],
            b"}",
            b"X { float {1, } }",
            b"X { float[2] {{1}} }",
            b"X { float[0] {} }",
            b"X { int32 {1.5} }",
            b"X { string {\"open} }",
            b"X (a = ) {}",
            b"/* open",
            b"X $ {}",
            b"X { ref {$a%} }",
            b"X { string {\"\\q\"} }",
            b"1 {}",
            b"X { float {0x1FFFFFFFFFFFFFFFFF} }",
        ] {
            assert!(Tree::parse(src).is_err(), "{}", crate::bytes::text(src));
        }
        let deep = "X {".repeat(MAX_DEPTH + 1) + &"}".repeat(MAX_DEPTH + 1);
        assert!(Tree::parse(deep.as_bytes()).is_err());
        let ok = "X {".repeat(MAX_DEPTH) + &"}".repeat(MAX_DEPTH);
        assert!(Tree::parse(ok.as_bytes()).is_ok());
    }
}
