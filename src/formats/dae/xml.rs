//! The document as a small read-only tree: element names, attributes and text, the text
//! borrowed from the file where it can be (the big number arrays always are), and fast number
//! parsing over it.

use std::borrow::Cow;
use std::collections::HashMap;

use quick_xml::Reader;
use quick_xml::events::Event;

use crate::formats::stream::{self, f32_lenient, f64_lenient, tokens};
use crate::formats::xml as shared;
use crate::gltf::percent_decode;
use crate::num::u32_of;

/// No element: the end of a list of children.
const NONE: u32 = u32::MAX;

/// Elements nested deeper than this are dropped, with what they hold.
const MAX_DEPTH: usize = 512;

/// An element.
struct El<'a> {
    /// The local name, without a namespace prefix.
    name: Cow<'a, [u8]>,
    /// Where its attributes are in the document's list of them.
    attrs: (u32, u32),
    /// The text directly inside, references resolved; untrimmed.
    text: Cow<'a, [u8]>,
    parent: u32,
    first: u32,
    last: u32,
    next: u32,
}

/// The parsed document, and the documents its URLs lead into after it.
pub(super) struct Dom<'a> {
    els: Vec<El<'a>>,
    /// Every element's attributes by local name, values unescaped, in one list, so an element
    /// costs no allocation of its own.
    attrs: Vec<Attr<'a>>,
    ids: HashMap<String, u32>,
    /// The documents in the tree, the first one first, by where their elements start.
    docs: Vec<Doc>,
    /// The names of the folders the first document is in, the innermost last, so a path that steps
    /// out of its folder and back in (`../here/x.dae`) is known for the same as `x.dae`.
    home: Vec<String>,
    /// Whether the first document stops before its XML does: elements still open at its end, or a tag the
    /// file ends inside.
    pub(super) cut: bool,
}

/// An attribute: its local name and its unescaped value.
type Attr<'a> = (Cow<'a, [u8]>, Cow<'a, str>);

/// One document of the tree.
struct Doc {
    /// Its first element.
    start: u32,
    /// Its path from the first document's folder (`/`-separated; `..` steps up).
    path: String,
    /// Its elements by id.
    ids: HashMap<String, u32>,
}

/// The children of an element.
pub(super) struct Children<'d, 'a> {
    dom: &'d Dom<'a>,
    at: u32,
}

impl Iterator for Children<'_, '_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.at == NONE {
            return None;
        }
        let e = self.at as usize;
        self.at = self.dom.els[e].next;
        Some(e)
    }
}

impl<'a> Dom<'a> {
    /// Parses `bytes` (UTF-8, no byte order mark). Malformed or cut-off XML keeps the elements
    /// read before the fault.
    pub(super) fn parse(bytes: &'a [u8]) -> Self {
        let mut dom = Dom {
            els: Vec::new(),
            attrs: Vec::new(),
            ids: HashMap::new(),
            docs: Vec::new(),
            home: Vec::new(),
            cut: false,
        };
        let mut reader = Reader::from_reader(bytes);
        let config = reader.config_mut();
        config.check_end_names = false;
        config.allow_unmatched_ends = true;
        config.check_comments = false;
        let mut stack: Vec<u32> = Vec::new();
        // Elements past the depth limit still open.
        let mut skipped = 0usize;
        loop {
            let event = match reader.read_event() {
                Ok(Event::Eof) => {
                    dom.cut = !stack.is_empty() || skipped > 0;
                    break;
                }
                Err(_) => {
                    dom.cut = true;
                    break;
                }
                Ok(e) => e,
            };
            match event {
                Event::Start(e) => {
                    if skipped > 0 || stack.len() >= MAX_DEPTH {
                        skipped += 1;
                    } else {
                        let id = dom.push(within(bytes, &e), stack.last().copied());
                        stack.push(id);
                    }
                }
                Event::Empty(e) => {
                    if skipped == 0 && stack.len() < MAX_DEPTH {
                        dom.push(within(bytes, &e), stack.last().copied());
                    }
                }
                Event::End(_) => {
                    if skipped > 0 {
                        skipped -= 1;
                    } else {
                        stack.pop();
                    }
                }
                Event::Text(t) if skipped == 0 => {
                    if let Some(&top) = stack.last() {
                        dom.append(top, t.into_inner());
                    }
                }
                Event::CData(t) if skipped == 0 => {
                    if let Some(&top) = stack.last() {
                        dom.append(top, t.into_inner());
                    }
                }
                Event::GeneralRef(r) if skipped == 0 => {
                    if let Some(&top) = stack.last() {
                        let mut text = Vec::new();
                        shared::push_reference(&r, &mut text);
                        dom.append(top, Cow::Owned(text));
                    }
                }
                _ => {}
            }
        }
        for k in 0..dom.els.len() {
            if let Some(v) = dom.attr(k, "id") {
                dom.ids.entry(v.to_owned()).or_insert(u32_of(k));
            }
        }
        dom
    }

    fn push(&mut self, raw: Cow<'a, [u8]>, parent: Option<u32>) -> u32 {
        let from = u32_of(self.attrs.len());
        let name = split_tag(raw, &mut self.attrs);
        let id = u32_of(self.els.len());
        self.els.push(El {
            name,
            attrs: (from, u32_of(self.attrs.len())),
            text: Cow::Borrowed(&[]),
            parent: parent.unwrap_or(NONE),
            first: NONE,
            last: NONE,
            next: NONE,
        });
        if let Some(p) = parent {
            let p = p as usize;
            if self.els[p].last == NONE {
                self.els[p].first = id;
            } else {
                let last = self.els[p].last as usize;
                self.els[last].next = id;
            }
            self.els[p].last = id;
        }
        id
    }

    fn append(&mut self, e: u32, chunk: Cow<'a, [u8]>) {
        let text = &mut self.els[e as usize].text;
        // The whitespace between child elements isn't kept.
        if chunk.iter().all(u8::is_ascii_whitespace) && text.iter().all(u8::is_ascii_whitespace) {
            if text.is_empty() {
                *text = chunk;
            }
            return;
        }
        if text.iter().all(u8::is_ascii_whitespace) {
            *text = chunk;
        } else {
            text.to_mut().extend_from_slice(&chunk);
        }
    }

    /// The document element, when there is one.
    pub(super) fn root(&self) -> Option<usize> {
        (!self.els.is_empty()).then_some(0)
    }

    pub(super) fn name(&self, e: usize) -> &[u8] {
        &self.els[e].name
    }

    pub(super) fn is(&self, e: usize, name: &str) -> bool {
        self.els[e].name.as_ref() == name.as_bytes()
    }

    pub(super) fn parent(&self, e: usize) -> Option<usize> {
        let p = self.els[e].parent;
        (p != NONE).then_some(p as usize)
    }

    pub(super) fn attr(&self, e: usize, name: &str) -> Option<&str> {
        self.attrs_of(e)
            .iter()
            .find(|(n, _)| n.as_ref() == name.as_bytes())
            .map(|(_, v)| v.as_ref())
    }

    /// The attributes of `e`.
    fn attrs_of(&self, e: usize) -> &[Attr<'a>] {
        let (from, to) = self.els[e].attrs;
        &self.attrs[from as usize..to as usize]
    }

    /// An attribute parsed as a number.
    pub(super) fn attr_num<T: std::str::FromStr>(&self, e: usize, name: &str) -> Option<T> {
        self.attr(e, name)?.trim().parse().ok()
    }

    pub(super) fn children(&self, e: usize) -> Children<'_, 'a> {
        Children {
            dom: self,
            at: self.els[e].first,
        }
    }

    /// The children named `name`.
    pub(super) fn kids<'s>(&'s self, e: usize, name: &'s str) -> impl Iterator<Item = usize> + 's {
        self.children(e).filter(move |&c| self.is(c, name))
    }

    /// The first child named `name`.
    pub(super) fn child(&self, e: usize, name: &str) -> Option<usize> {
        self.children(e).find(|&c| self.is(c, name))
    }

    /// Follows a path of child names.
    pub(super) fn path(&self, e: usize, names: &[&str]) -> Option<usize> {
        names.iter().try_fold(e, |at, n| self.child(at, n))
    }

    /// The first descendant named `name`, breadth first, within `depth` levels.
    pub(super) fn descendant(&self, e: usize, name: &str, depth: usize) -> Option<usize> {
        let mut level = vec![e];
        for _ in 0..depth {
            let mut next = Vec::new();
            for &at in &level {
                for c in self.children(at) {
                    if self.is(c, name) {
                        return Some(c);
                    }
                    next.push(c);
                }
            }
            if next.is_empty() {
                break;
            }
            level = next;
        }
        None
    }

    /// The trimmed text inside an element.
    pub(super) fn text(&self, e: usize) -> &[u8] {
        self.els[e].text.trim_ascii()
    }

    /// The trimmed text inside an element as a string. The document is UTF-8 by the time it's parsed
    /// ([`crate::formats::xml::utf8`]).
    pub(super) fn string(&self, e: usize) -> &str {
        std::str::from_utf8(self.text(e)).unwrap_or("")
    }

    /// The text of the child named `name`.
    pub(super) fn child_string(&self, e: usize, name: &str) -> Option<&str> {
        self.child(e, name).map(|c| self.string(c))
    }

    /// The element with id `id`.
    pub(super) fn by_id(&self, id: &str) -> Option<usize> {
        self.ids.get(id).map(|&e| e as usize)
    }

    /// The element with id `id` in the document `e` is in, else in the first document.
    pub(super) fn by_id_at(&self, e: usize, id: &str) -> Option<usize> {
        self.ids_of(e).get(id).map(|&k| k as usize).or_else(|| self.by_id(id))
    }

    /// The number of elements.
    pub(super) fn len(&self) -> usize {
        self.els.len()
    }

    /// The document after the first that `e` is in.
    fn doc_of(&self, e: usize) -> Option<&Doc> {
        let k = self.docs.partition_point(|d| d.start as usize <= e);
        k.checked_sub(1).map(|k| &self.docs[k])
    }

    /// The path of the document `e` is in; the first document's is its file name.
    pub(super) fn path_of(&self, e: usize) -> &str {
        self.doc_of(e).map_or("", |d| d.path.as_str())
    }

    /// The folder (empty, or with a trailing `/`) of the document `e` is in, from the first one's.
    pub(super) fn dir_of(&self, e: usize) -> &str {
        let path = self.path_of(e);
        path.rfind('/').map_or("", |k| &path[..=k])
    }

    /// Names the first document and its folder (`/`-separated), so URLs that name it by its path stay in it.
    pub(super) fn set_home(&mut self, path: &str, dir: &str) {
        self.home = dir
            .split('/')
            .filter(|p| !p.is_empty() && *p != ".")
            .map(str::to_owned)
            .collect();
        self.set_path(path);
    }

    fn set_path(&mut self, path: &str) {
        if self.docs.is_empty() {
            self.docs.push(Doc {
                start: 0,
                path: path.to_owned(),
                ids: HashMap::new(),
            });
        }
    }

    /// Whether the document at `path` is in the tree.
    pub(super) fn has(&self, path: &str) -> bool {
        self.docs.iter().any(|d| d.path == path)
    }

    /// Adds `other`, the document at `path`, to the tree, copying what it borrows. An empty one
    /// is kept too, so it isn't asked for again.
    pub(super) fn absorb(&mut self, path: &str, other: Dom<'_>) {
        let start = u32_of(self.els.len());
        let shift = |k: u32| if k == NONE { NONE } else { k + start };
        let base = u32_of(self.attrs.len());
        self.attrs.extend(
            other
                .attrs
                .into_iter()
                .map(|(k, v)| (Cow::Owned(k.into_owned()), Cow::Owned(v.into_owned()))),
        );
        for el in other.els {
            self.els.push(El {
                name: Cow::Owned(el.name.into_owned()),
                attrs: (el.attrs.0 + base, el.attrs.1 + base),
                text: Cow::Owned(el.text.into_owned()),
                parent: shift(el.parent),
                first: shift(el.first),
                last: shift(el.last),
                next: shift(el.next),
            });
        }
        // The first document's ids belong to it; this one's are its own.
        let ids = other.ids.into_iter().map(|(k, v)| (k, v + start)).collect();
        self.set_path("");
        self.docs.push(Doc {
            start,
            path: path.to_owned(),
            ids,
        });
    }

    /// The path of the other document a URL written in `e`'s document leads into, when it leads into one.
    pub(super) fn document(&self, e: usize, url: &str) -> Option<String> {
        let url = url.trim();
        let (path, _) = url.split_once('#')?;
        let path = percent_decode(path).replace('\\', "/");
        if path.is_empty() || path.contains("://") || path.starts_with('/') || path.as_bytes().get(1) == Some(&b':') {
            return None;
        }
        let mut full = join(self.dir_of(e), &path);
        // Out of the first document's folder and back in.
        let up = full.split('/').take_while(|p| *p == "..").count();
        if up > 0 && up <= self.home.len() {
            let parts: Vec<&str> = full.split('/').skip(up).collect();
            if parts.len() > up
                && parts[..up]
                    .iter()
                    .zip(&self.home[self.home.len() - up..])
                    .all(|(a, b)| a == b)
            {
                full = parts[up..].join("/");
            }
        }
        (full != self.path_of(e)).then_some(full)
    }

    /// The element a URL written in `e`'s document names: `#id` in that document, `path#id` in the
    /// one at that path, or a bare id.
    pub(super) fn url_at(&self, e: usize, url: &str) -> Option<usize> {
        let url = url.trim();
        let (ids, id) = match url.split_once('#') {
            Some(("", id)) => (self.ids_of(e), id),
            Some((_, id)) => match self.document(e, url) {
                None => (self.ids_of(e), id),
                Some(path) => {
                    let doc = self.docs.iter().find(|d| d.path == path)?;
                    (if doc.start == 0 { &self.ids } else { &doc.ids }, id)
                }
            },
            None => (self.ids_of(e), url),
        };
        ids.get(id).map(|&k| k as usize).or_else(|| {
            let decoded = percent_decode(id);
            (decoded != id)
                .then(|| ids.get(&decoded).map(|&k| k as usize))
                .flatten()
        })
    }

    /// The ids of the document `e` is in.
    fn ids_of(&self, e: usize) -> &HashMap<String, u32> {
        match self.doc_of(e) {
            Some(d) if d.start != 0 => &d.ids,
            _ => &self.ids,
        }
    }

    /// The element the attribute `name` of `e` points at.
    pub(super) fn link(&self, e: usize, name: &str) -> Option<usize> {
        self.url_at(e, self.attr(e, name)?)
    }

    /// The paths of the other documents that URLs in the elements from `from` on lead into, and
    /// that aren't in the tree yet, in the order they're first named.
    pub(super) fn wanted(&self, from: usize) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for e in from..self.els.len() {
            let mut add = |url: &str| {
                if let Some(path) = self.document(e, url)
                    && !self.has(&path)
                    && !out.contains(&path)
                {
                    out.push(path);
                }
            };
            for (k, v) in self.attrs_of(e) {
                if matches!(k.as_ref(), b"url" | b"source" | b"target") && v.contains('#') {
                    add(v);
                }
            }
            if self.is(e, "skeleton") {
                add(self.string(e));
            }
        }
        out
    }

    /// The descendant of `e` whose `sid` is `sid`, breadth first.
    pub(super) fn by_sid(&self, e: usize, sid: &str) -> Option<usize> {
        let mut queue = std::collections::VecDeque::from([e]);
        let mut seen = 0usize;
        while let Some(at) = queue.pop_front() {
            for c in self.children(at) {
                if self.attr(c, "sid") == Some(sid) {
                    return Some(c);
                }
                queue.push_back(c);
            }
            seen += 1;
            if seen > 1_000_000 {
                break;
            }
        }
        None
    }
}

/// The element's local name and attributes from the raw contents of its start tag.
/// A tag's bytes as a slice of the document when they are in it (as they are, read from a slice), else a copy.
fn within<'a>(doc: &'a [u8], tag: &[u8]) -> Cow<'a, [u8]> {
    let base = doc.as_ptr() as usize;
    let start = tag.as_ptr() as usize;
    match start.checked_sub(base) {
        Some(off) if off.checked_add(tag.len()).is_some_and(|end| end <= doc.len()) => {
            Cow::Borrowed(&doc[off..off + tag.len()])
        }
        _ => Cow::Owned(tag.to_vec()),
    }
}

#[allow(clippy::type_complexity, reason = "a name and its attributes")]
/// A tag's name, its attributes added to `attrs`.
fn split_tag<'a>(raw: Cow<'a, [u8]>, attrs: &mut Vec<Attr<'a>>) -> Cow<'a, [u8]> {
    match raw {
        Cow::Borrowed(b) => split_borrowed(b, attrs),
        Cow::Owned(v) => {
            let mut own = Vec::new();
            let name = split_borrowed(&v, &mut own).into_owned();
            attrs.extend(
                own.into_iter()
                    .map(|(k, v)| (Cow::Owned(k.into_owned()), Cow::Owned(v.into_owned()))),
            );
            Cow::Owned(name)
        }
    }
}

fn split_borrowed<'a>(b: &'a [u8], attrs: &mut Vec<Attr<'a>>) -> Cow<'a, [u8]> {
    let end = b.iter().position(u8::is_ascii_whitespace).unwrap_or(b.len());
    let name = local(&b[..end]);
    let mut i = end;
    let n = b.len();
    while i < n {
        while i < n && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let k0 = i;
        while i < n && b[i] != b'=' && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        let key = &b[k0..i];
        while i < n && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n || b[i] != b'=' {
            if key.is_empty() {
                break;
            }
            continue;
        }
        i += 1;
        while i < n && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        let quote = b[i];
        let value = if quote == b'"' || quote == b'\'' {
            i += 1;
            let v0 = i;
            while i < n && b[i] != quote {
                i += 1;
            }
            let v = &b[v0..i];
            i += 1;
            v
        } else {
            let v0 = i;
            while i < n && !b[i].is_ascii_whitespace() {
                i += 1;
            }
            &b[v0..i]
        };
        if !key.is_empty() {
            attrs.push((Cow::Borrowed(local(key)), shared::unescape_lossy(value)));
        }
    }
    Cow::Borrowed(name)
}

/// A name without its namespace prefix. `xml:base` keeps its prefix, so it can't be taken for
/// an attribute named `base`.
fn local(name: &[u8]) -> &[u8] {
    if name.starts_with(b"xml:") {
        return name;
    }
    match name.iter().rposition(|&c| c == b':') {
        Some(k) => &name[k + 1..],
        None => name,
    }
}

/// `path` taken from the folder `dir` (empty, or with a trailing `/`), its `.` and `..` parts
/// resolved, `..` past the top kept.
pub(super) fn join(dir: &str, path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for p in dir.split('/').chain(path.split('/')) {
        match p {
            "" | "." => {}
            ".." if parts.last().is_some_and(|l| *l != "..") => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    parts.join("/")
}

/// The numbers in whitespace-separated text, as `f64`: anything that isn't one reads as 0.
pub(super) fn floats(text: &[u8], out: &mut Vec<f64>) {
    out.extend(tokens(text).map(float));
}

/// The numbers in whitespace-separated text, as `f32`, non-finite ones as 0.
pub(super) fn floats32(text: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(text.len() / 6);
    out.extend(tokens(text).map(|t| match stream::f32_from(t) {
        Some(v) if v.is_finite() => v,
        Some(_) => 0.0,
        None => float32(t),
    }));
    out
}

/// A number in one of the rarer spellings [`f32_lenient`] takes, finite, or else 0.
#[cold]
fn float32(t: &[u8]) -> f32 {
    f32_lenient(t).filter(|v| v.is_finite()).unwrap_or(0.0)
}

/// A number, in any spelling [`f64_lenient`] takes. Not a number reads as 0.
pub(super) fn float(t: &[u8]) -> f64 {
    f64_lenient(t).filter(|v| !v.is_nan()).unwrap_or(0.0)
}

/// The whole numbers in whitespace-separated text; anything that isn't one (negative ones too)
/// reads as `u32::MAX`, which no array is long enough to index.
pub(super) fn indices(text: &[u8]) -> Vec<u32> {
    let n = text.len();
    let mut out = Vec::with_capacity(n / 3);
    let mut i = 0;
    loop {
        while i < n && stream::space(text[i]) {
            i += 1;
        }
        if i >= n {
            break;
        }
        // Nearly every one is plain digits, read as the token is found.
        let start = i;
        let mut v: u64 = 0;
        while let Some(d) = text.get(i).map(|c| c.wrapping_sub(b'0')).filter(|&d| d <= 9) {
            v = (v * 10 + u64::from(d)).min(u64::from(u32::MAX));
            i += 1;
        }
        if text.get(i).is_some_and(|&c| !stream::space(c)) {
            while i < n && !stream::space(text[i]) {
                i += 1;
            }
            out.push(index(&text[start..i]));
        } else {
            out.push(u32::try_from(v).unwrap_or(u32::MAX));
        }
    }
    out
}

/// An index in any other spelling: `+3`, `-0`, or `3.0`, which is 3; anything else is no index.
#[cold]
fn index(t: &[u8]) -> u32 {
    stream::int(t).unwrap_or_else(|| {
        let f = f64_lenient(t).unwrap_or(f64::NAN);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "checked to be a whole number in range"
        )]
        if f >= 0.0 && f.fract() == 0.0 && f < f64::from(u32::MAX) {
            f as u32
        } else {
            u32::MAX
        }
    })
}

/// The whole numbers in whitespace-separated text, signed; anything that isn't one reads as -1.
pub(super) fn ints(text: &[u8]) -> Vec<i64> {
    tokens(text).map(|t| stream::int(t).unwrap_or(-1)).collect()
}

/// The whitespace-separated words of `text`.
pub(super) fn words(text: &[u8]) -> impl Iterator<Item = &str> {
    tokens(text).map(|t| std::str::from_utf8(t).unwrap_or(""))
}
