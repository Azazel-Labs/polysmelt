//! What the XML readers share: their text as UTF-8, whatever encoding it came in, and a small XML reader.
//!
//! XML is UTF-8 or UTF-16 (XML 1.0, appendix F): UTF-16 is told by its byte order mark or, without one, by the
//! zero bytes of the `<` it starts with. [`utf8`] and [`Utf8Read`] give every reader the same UTF-8, a UTF-8 byte
//! order mark dropped, so no reader has to know.
//!
//! [`Xml`] keeps elements and their attributes in two flat lists, each element linked to its first child and next
//! sibling, so a document's thousands of small elements cost no allocation each. Comments, processing
//! instructions and a document type are skipped; text is kept as the span between an element's tags, read with
//! [`El::text`], which decodes CDATA, the five named entities and numeric references (as attribute values are).
//! Tags that don't match, a tag left open, or elements nested deeper than [`MAX_DEPTH`] are errors.

use std::borrow::Cow;
use std::io::{self, BufRead, Read};

use crate::Result;

/// Whether `head`, the start of an XML document, is UTF-16: little-endian or not, and the length of its byte
/// order mark.
pub(crate) fn utf16(head: &[u8]) -> Option<(bool, usize)> {
    match head {
        [0xFF, 0xFE, ..] => Some((true, 2)),
        [0xFE, 0xFF, ..] => Some((false, 2)),
        [b'<', 0, c, 0, ..] if *c != 0 => Some((true, 0)),
        [0, b'<', 0, c, ..] if *c != 0 => Some((false, 0)),
        // Whitespace before the first tag.
        [c, 0, d, 0, ..] if c.is_ascii_whitespace() && (*d == b'<' || d.is_ascii_whitespace()) => Some((true, 0)),
        [0, c, 0, d, ..] if c.is_ascii_whitespace() && (*d == b'<' || d.is_ascii_whitespace()) => Some((false, 0)),
        _ => None,
    }
}

/// An XML document's bytes as UTF-8: UTF-16 converted (an unpaired surrogate becomes U+FFFD), a UTF-8 byte order
/// mark dropped, anything else as it is.
pub(crate) fn utf8(bytes: &[u8]) -> Cow<'_, [u8]> {
    if let Some(rest) = bytes.strip_prefix(b"\xEF\xBB\xBF") {
        return Cow::Borrowed(rest);
    }
    let Some((le, bom)) = utf16(bytes) else {
        return Cow::Borrowed(bytes);
    };
    let units = bytes[bom..].chunks_exact(2).map(|c| {
        if le {
            u16::from_le_bytes([c[0], c[1]])
        } else {
            u16::from_be_bytes([c[0], c[1]])
        }
    });
    let mut out = Vec::with_capacity(bytes.len() / 2 + 16);
    let mut buf = [0; 4];
    for c in char::decode_utf16(units) {
        let c = c.unwrap_or(char::REPLACEMENT_CHARACTER);
        if c.is_ascii() {
            out.push(c as u8);
        } else {
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    }
    Cow::Owned(out)
}

/// An XML document's text: [`utf8`], and bytes that aren't UTF-8 read as Latin-1.
pub(crate) fn text(bytes: &[u8]) -> Cow<'_, str> {
    match utf8(bytes) {
        Cow::Borrowed(b) => match std::str::from_utf8(b) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(b.iter().map(|&c| char::from(c)).collect()),
        },
        Cow::Owned(b) => {
            Cow::Owned(String::from_utf8(b).unwrap_or_else(|e| e.into_bytes().iter().map(|&c| char::from(c)).collect()))
        }
    }
}

/// A stream of an XML document as UTF-8, as [`utf8`] gives it, for readers that parse as they read.
pub(crate) struct Utf8Read<R> {
    inner: R,
    /// UTF-16, little-endian or not; `None` for bytes passed through.
    wide: Option<bool>,
    /// UTF-8 converted and not yet consumed: `out[at..]`.
    out: Vec<u8>,
    at: usize,
    /// A byte of a unit split between reads, and a leading surrogate waiting for its pair.
    odd: Option<u8>,
    high: Option<u16>,
}

impl<R: BufRead> Utf8Read<R> {
    /// Looks at the start of `inner` to tell its encoding.
    pub(crate) fn new(mut inner: R) -> io::Result<Self> {
        let head = inner.fill_buf()?;
        let (wide, skip) = if head.starts_with(b"\xEF\xBB\xBF") {
            (None, 3)
        } else {
            utf16(head).map_or((None, 0), |(le, bom)| (Some(le), bom))
        };
        inner.consume(skip);
        Ok(Self {
            inner,
            wide,
            out: Vec::new(),
            at: 0,
            odd: None,
            high: None,
        })
    }

    /// Converts the next block of UTF-16 into `out`; leaves it empty at the end of the stream.
    fn refill(&mut self, le: bool) -> io::Result<()> {
        self.out.clear();
        self.at = 0;
        while self.out.is_empty() {
            let block = self.inner.fill_buf()?;
            if block.is_empty() {
                if self.odd.take().is_some() || self.high.take().is_some() {
                    self.out.extend_from_slice("\u{FFFD}".as_bytes());
                }
                return Ok(());
            }
            let mut bytes = block;
            let first = self.odd.take().map(|b| {
                let pair = [b, bytes[0]];
                bytes = &bytes[1..];
                pair
            });
            let (pairs, rest) = bytes.as_chunks::<2>();
            if let [b] = rest {
                self.odd = Some(*b);
            }
            let unit = |c: [u8; 2]| {
                if le {
                    u16::from_le_bytes(c)
                } else {
                    u16::from_be_bytes(c)
                }
            };
            let mut buf = [0; 4];
            for u in first.into_iter().chain(pairs.iter().copied()).map(unit) {
                let c = match (self.high.take(), u) {
                    (Some(h), 0xDC00..=0xDFFF) => {
                        char::from_u32(0x10000 + ((u32::from(h) - 0xD800) << 10) + (u32::from(u) - 0xDC00))
                    }
                    (h, _) => {
                        if h.is_some() {
                            self.out.extend_from_slice("\u{FFFD}".as_bytes());
                        }
                        if (0xD800..0xDC00).contains(&u) {
                            self.high = Some(u);
                            continue;
                        }
                        char::from_u32(u32::from(u))
                    }
                };
                let c = c.unwrap_or(char::REPLACEMENT_CHARACTER);
                self.out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
            let n = block.len();
            self.inner.consume(n);
        }
        Ok(())
    }
}

impl<R: BufRead> Read for Utf8Read<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let data = self.fill_buf()?;
        let n = data.len().min(buf.len());
        buf[..n].copy_from_slice(&data[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl<R: BufRead> BufRead for Utf8Read<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        match self.wide {
            None => self.inner.fill_buf(),
            Some(le) => {
                if self.at == self.out.len() {
                    self.refill(le)?;
                }
                Ok(&self.out[self.at..])
            }
        }
    }

    fn consume(&mut self, n: usize) {
        match self.wide {
            None => self.inner.consume(n),
            Some(_) => self.at = (self.at + n).min(self.out.len()),
        }
    }
}

/// The deepest elements nest.
pub(crate) const MAX_DEPTH: usize = 256;
/// No element: the end of a sibling list.
const NONE: usize = usize::MAX;

/// A parsed XML document.
pub(crate) struct Xml<'a> {
    src: &'a str,
    nodes: Vec<XNode>,
    attrs: Vec<XAttr>,
}

#[derive(Clone, Copy)]
struct XNode {
    name: (usize, usize),
    attrs: (usize, usize),
    /// What's between the element's tags (empty for `<a/>`).
    content: (usize, usize),
    first: usize,
    last: usize,
    next: usize,
}

#[derive(Clone, Copy)]
struct XAttr {
    name: (usize, usize),
    value: (usize, usize),
}

/// An element of a parsed document.
#[derive(Clone, Copy)]
pub(crate) struct El<'d, 'a> {
    doc: &'d Xml<'a>,
    i: usize,
}

impl<'a> Xml<'a> {
    /// Parses `src`.
    pub(crate) fn parse(src: &'a str) -> Result<Self> {
        let b = src.as_bytes();
        let mut doc = Xml {
            src,
            nodes: Vec::new(),
            attrs: Vec::new(),
        };
        let mut stack: Vec<usize> = Vec::new();
        let mut at = 0;
        while let Some(p) = memchr(b'<', &b[at..]) {
            at += p;
            let rest = &b[at..];
            if rest.starts_with(b"<!--") {
                at = find(b, at + 4, b"-->")? + 3;
            } else if rest.starts_with(b"<![CDATA[") {
                at = find(b, at + 9, b"]]>")? + 3;
            } else if rest.starts_with(b"<?") {
                at = find(b, at + 2, b"?>")? + 2;
            } else if rest.starts_with(b"<!") {
                // A document type: to its closing bracket, past any internal subset and quoted strings.
                let mut depth = 0usize;
                let mut quote = 0u8;
                let mut i = at + 2;
                loop {
                    let Some(&c) = b.get(i) else {
                        bail!(Corrupt, "XML document type left open");
                    };
                    if quote != 0 {
                        if c == quote {
                            quote = 0;
                        }
                    } else if c == b'"' || c == b'\'' {
                        quote = c;
                    } else if c == b'[' {
                        depth += 1;
                    } else if c == b']' {
                        depth = depth.saturating_sub(1);
                    } else if c == b'>' && depth == 0 {
                        break;
                    }
                    i += 1;
                }
                at = i + 1;
            } else if rest.starts_with(b"</") {
                let start = at + 2;
                let end = find(b, start, b">")?;
                let name = src[start..end].trim_end();
                let Some(open) = stack.pop() else {
                    bail!(Corrupt, "XML closing tag </{name}> has no opening tag");
                };
                let (s, e) = doc.nodes[open].name;
                ensure!(
                    &src[s..e] == name,
                    Corrupt,
                    "XML tag <{}> closed by </{name}>",
                    &src[s..e]
                );
                doc.nodes[open].content.1 = at;
                at = end + 1;
            } else {
                at = doc.tag(at + 1, &mut stack)?;
            }
        }
        if let Some(&open) = stack.last() {
            let (s, e) = doc.nodes[open].name;
            bail!(Corrupt, "XML element <{}> left open", &src[s..e]);
        }
        ensure!(!doc.nodes.is_empty(), Corrupt, "XML holds no element");
        Ok(doc)
    }

    /// Reads an opening tag from just past its `<`; returns where the reader goes on.
    fn tag(&mut self, mut at: usize, stack: &mut Vec<usize>) -> Result<usize> {
        let b = self.src.as_bytes();
        let name_end = at
            + b[at..]
                .iter()
                .position(|&c| c.is_ascii_whitespace() || c == b'>' || c == b'/')
                .unwrap_or(b.len() - at);
        ensure!(name_end > at, Corrupt, "XML tag without a name at byte {at}");
        let name = (at, name_end);
        at = name_end;
        let attr_start = self.attrs.len();
        let closed;
        loop {
            while b.get(at).is_some_and(u8::is_ascii_whitespace) {
                at += 1;
            }
            match b.get(at) {
                None => bail!(Truncated, "XML tag is cut short"),
                Some(b'>') => {
                    closed = false;
                    at += 1;
                    break;
                }
                Some(b'/') => {
                    ensure!(
                        b.get(at + 1) == Some(&b'>'),
                        Corrupt,
                        "XML tag has a stray / at byte {at}"
                    );
                    closed = true;
                    at += 2;
                    break;
                }
                _ => {}
            }
            let n0 = at;
            while b
                .get(at)
                .is_some_and(|&c| !c.is_ascii_whitespace() && c != b'=' && c != b'>' && c != b'/')
            {
                at += 1;
            }
            let n1 = at;
            while b.get(at).is_some_and(u8::is_ascii_whitespace) {
                at += 1;
            }
            ensure!(
                n1 > n0 && b.get(at) == Some(&b'='),
                Corrupt,
                "XML attribute without a value at byte {n0}"
            );
            at += 1;
            while b.get(at).is_some_and(u8::is_ascii_whitespace) {
                at += 1;
            }
            let Some(&q @ (b'"' | b'\'')) = b.get(at) else {
                bail!(Corrupt, "XML attribute value without quotes at byte {at}");
            };
            let v0 = at + 1;
            let Some(len) = memchr(q, &b[v0..]) else {
                bail!(Corrupt, "XML attribute value left open");
            };
            at = v0 + len + 1;
            self.attrs.push(XAttr {
                name: (n0, n1),
                value: (v0, v0 + len),
            });
        }
        let i = self.nodes.len();
        self.nodes.push(XNode {
            name,
            attrs: (attr_start, self.attrs.len()),
            content: (at, at),
            first: NONE,
            last: NONE,
            next: NONE,
        });
        // A second top-level element isn't linked anywhere, so it's ignored.
        if let Some(&parent) = stack.last() {
            let p = &mut self.nodes[parent];
            if p.first == NONE {
                p.first = i;
            } else {
                let last = p.last;
                self.nodes[last].next = i;
            }
            self.nodes[parent].last = i;
        }
        if !closed {
            ensure!(
                stack.len() < MAX_DEPTH,
                LimitExceeded,
                "XML elements nest deeper than {MAX_DEPTH}"
            );
            stack.push(i);
        }
        Ok(at)
    }

    /// The document's top element.
    pub(crate) fn root(&self) -> El<'_, 'a> {
        El { doc: self, i: 0 }
    }
}

/// Where `pat` next starts in `b`, from `from`.
fn find(b: &[u8], from: usize, pat: &[u8]) -> Result<usize> {
    let mut at = from;
    while let Some(p) = memchr(pat[0], &b[at..]) {
        at += p;
        if b[at..].starts_with(pat) {
            return Ok(at);
        }
        at += 1;
    }
    bail!(Truncated, "XML is cut short at byte {from}")
}

fn memchr(c: u8, b: &[u8]) -> Option<usize> {
    b.iter().position(|&x| x == c)
}

/// Decodes the entities in text or an attribute value.
pub(crate) fn unescape(s: &str) -> Cow<'_, str> {
    if !s.contains('&') {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let Some(end) = rest.find(';').filter(|&e| e <= 12) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let name = &rest[1..end];
        let c = match name {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => name
                .strip_prefix("#x")
                .or_else(|| name.strip_prefix("#X"))
                .map_or_else(
                    || name.strip_prefix('#').and_then(|d| d.parse::<u32>().ok()),
                    |h| u32::from_str_radix(h, 16).ok(),
                )
                .and_then(char::from_u32),
        };
        if let Some(c) = c {
            out.push(c);
            rest = &rest[end + 1..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    Cow::Owned(out)
}

impl<'d, 'a> El<'d, 'a> {
    fn node(&self) -> XNode {
        self.doc.nodes[self.i]
    }

    /// The element's name.
    pub(crate) fn name(&self) -> &'d str {
        let (s, e) = self.node().name;
        &self.doc.src[s..e]
    }

    /// The element's name without its namespace prefix.
    pub(crate) fn local(&self) -> &'d str {
        let n = self.name();
        n.rsplit_once(':').map_or(n, |(_, l)| l)
    }

    /// Whether the element is named `name`, in any case.
    pub(crate) fn is(&self, name: &str) -> bool {
        self.local().eq_ignore_ascii_case(name)
    }

    /// An attribute's value, entities decoded.
    pub(crate) fn attr(&self, name: &str) -> Option<Cow<'d, str>> {
        let (s, e) = self.node().attrs;
        self.doc.attrs[s..e]
            .iter()
            .find(|a| &self.doc.src[a.name.0..a.name.1] == name)
            .map(|a| unescape(&self.doc.src[a.value.0..a.value.1]))
    }

    /// An attribute's value, its name in any case.
    pub(crate) fn attr_ci(&self, name: &str) -> Option<Cow<'d, str>> {
        let (s, e) = self.node().attrs;
        self.doc.attrs[s..e]
            .iter()
            .find(|a| self.doc.src[a.name.0..a.name.1].eq_ignore_ascii_case(name))
            .map(|a| unescape(&self.doc.src[a.value.0..a.value.1]))
    }

    /// An attribute as a float: 0 when it's missing or isn't a number.
    pub(crate) fn f32(&self, name: &str) -> f32 {
        self.attr(name).and_then(|v| v.trim().parse().ok()).unwrap_or(0.0)
    }

    /// An attribute as a float, `None` when it's missing.
    pub(crate) fn opt_f32(&self, name: &str) -> Option<f32> {
        self.attr(name).map(|v| v.trim().parse().unwrap_or(0.0))
    }

    /// An attribute as a whole number: `None` when it's missing or isn't one.
    pub(crate) fn u32(&self, name: &str) -> Option<u32> {
        self.attr(name).and_then(|v| v.trim().parse().ok())
    }

    /// What's between the element's tags, as it's written: child elements, comments and all.
    pub(crate) fn raw(&self) -> &'d str {
        let (s, e) = self.node().content;
        &self.doc.src[s..e.max(s)]
    }

    /// The element's text: what's between its tags with child elements, comments and processing instructions
    /// left out, CDATA kept, and entities decoded.
    pub(crate) fn text(&self) -> Cow<'d, str> {
        let raw = self.raw();
        if !raw.contains('<') {
            return unescape(raw);
        }
        let mut out = String::new();
        let mut rest = raw;
        // How deep inside child elements, whose text isn't this element's.
        let mut depth = 0usize;
        while let Some(p) = rest.find('<') {
            if depth == 0 {
                out.push_str(&unescape(&rest[..p]));
            }
            rest = &rest[p..];
            let (skip, keep) = if let Some(r) = rest.strip_prefix("<![CDATA[") {
                let e = r.find("]]>").unwrap_or(r.len());
                (9 + e + 3, Some(&r[..e]))
            } else if let Some(r) = rest.strip_prefix("<!--") {
                (4 + r.find("-->").map_or(r.len(), |e| e + 3), None)
            } else {
                let end = rest.find('>').map_or(rest.len(), |e| e + 1);
                let tag = &rest[..end];
                if tag.starts_with("</") {
                    depth = depth.saturating_sub(1);
                } else if !tag.starts_with("<?") && !tag.starts_with("<!") && !tag.ends_with("/>") {
                    depth += 1;
                }
                (end, None)
            };
            if let Some(k) = keep.filter(|_| depth == 0) {
                out.push_str(k);
            }
            rest = &rest[skip.min(rest.len())..];
        }
        if depth == 0 {
            out.push_str(&unescape(rest));
        }
        Cow::Owned(out)
    }

    /// The element's children.
    pub(crate) fn children(&self) -> impl Iterator<Item = El<'d, 'a>> + use<'d, 'a> {
        let doc = self.doc;
        let mut at = self.node().first;
        std::iter::from_fn(move || {
            (at != NONE).then(|| {
                let e = El { doc, i: at };
                at = doc.nodes[at].next;
                e
            })
        })
    }

    /// The children named `name`.
    pub(crate) fn named<'n>(&self, name: &'n str) -> impl Iterator<Item = El<'d, 'a>> + use<'d, 'a, 'n> {
        self.children().filter(move |c| c.name() == name)
    }

    /// The first child named `name`.
    pub(crate) fn child(&self, name: &str) -> Option<El<'d, 'a>> {
        self.children().find(|c| c.name() == name)
    }

    /// The siblings after this element named `name`.
    pub(crate) fn next_named(&self, name: &str) -> Option<El<'d, 'a>> {
        let mut at = self.node().next;
        while at != NONE {
            let e = El { doc: self.doc, i: at };
            if e.name() == name {
                return Some(e);
            }
            at = self.doc.nodes[at].next;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_xml() {
        let x = Xml::parse(
            "<?xml version='1.0'?><!DOCTYPE m [<!ENTITY a 'b'>]><!-- c --><m a=\"1 &amp; &#x41;&#66;\"><n/> \
             text <![CDATA[<x>]]><n b='2'>in &lt;</n ></m>",
        )
        .unwrap();
        let r = x.root();
        assert_eq!(r.name(), "m");
        assert_eq!(r.attr("a").unwrap(), "1 & AB");
        let kids: Vec<_> = r.children().collect();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[1].attr("b").unwrap(), "2");
        assert_eq!(kids[1].text(), "in <");
        assert_eq!(kids[0].text(), "");
        assert_eq!(r.text(), " text <x>");
        assert!(Xml::parse("<a><b></a>").is_err());
        assert!(Xml::parse("<a>").is_err());
        assert!(Xml::parse("<a b></a>").is_err());
        assert!(Xml::parse("").is_err());
        let deep = "<a>".repeat(MAX_DEPTH + 1);
        assert!(Xml::parse(&deep).is_err());
    }

    fn wide(s: &str, le: bool, bom: bool) -> Vec<u8> {
        let mut out = Vec::new();
        if bom {
            out.extend(if le { [0xFF, 0xFE] } else { [0xFE, 0xFF] });
        }
        for u in s.encode_utf16() {
            out.extend(if le { u.to_le_bytes() } else { u.to_be_bytes() });
        }
        out
    }

    #[test]
    fn utf16_becomes_utf8() {
        let s = "<?xml version=\"1.0\" encoding=\"UTF-16\"?>\n<a b=\"é😀\">x</a>";
        for le in [true, false] {
            for bom in [true, false] {
                let w = wide(s, le, bom);
                assert_eq!(&*utf8(&w), s.as_bytes(), "le {le} bom {bom}");
                // Streamed in blocks that split units and surrogate pairs.
                for cap in [1, 2, 3, 5, 64] {
                    let mut r = Utf8Read::new(io::BufReader::with_capacity(cap.max(4), &w[..])).unwrap();
                    let mut out = String::new();
                    r.read_to_string(&mut out).unwrap();
                    assert_eq!(out, s, "le {le} bom {bom} cap {cap}");
                }
            }
        }
        assert_eq!(&*utf8(b"\xEF\xBB\xBF<a/>"), b"<a/>");
        assert_eq!(&*utf8(b"<a/>"), b"<a/>");
        let mut r = Utf8Read::new(&b"\xEF\xBB\xBF<a/>"[..]).unwrap();
        let mut out = String::new();
        r.read_to_string(&mut out).unwrap();
        assert_eq!(out, "<a/>");
        assert_eq!(text(b"<a>\xE9</a>"), "<a>é</a>");
        // A lone surrogate, and a unit cut off at the end.
        let mut w = wide("<a>", true, true);
        w.extend([0x00, 0xD8, b'b', 0, b'c']);
        assert_eq!(String::from_utf8(utf8(&w).into_owned()).unwrap(), "<a>\u{FFFD}b");
        let mut out = String::new();
        Utf8Read::new(&w[..]).unwrap().read_to_string(&mut out).unwrap();
        assert_eq!(out, "<a>\u{FFFD}b\u{FFFD}");
    }
}
