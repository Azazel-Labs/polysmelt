//! The classic encoding, VRML97's and X3D's: nodes as `Kind { field value ... }`, lists in
//! brackets, commas as space and `#` starting a comment.
//!
//! No node's fields are declared here, so a value's type is told from how it's written: a run
//! of numbers, a string, `TRUE` or `FALSE`, a node (`Kind {`, `DEF`, `USE`, `NULL`), or a list
//! of any of them. A prototype's interface and body are read, an external prototype's
//! interface and URLs; `Script` nodes, routes and the other statements are skipped.

use std::collections::HashMap;

use super::{Graph, MAX_DEPTH, Proto, Val};
use crate::formats::stream::f64_from;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tok<'d> {
    Open,
    Close,
    LBracket,
    RBracket,
    /// A string's bytes between its quotes, escapes still in.
    Str(&'d [u8]),
    Word(&'d [u8]),
    End,
}

struct Lexer<'d> {
    d: &'d [u8],
    at: usize,
}

impl<'d> Lexer<'d> {
    #[inline]
    fn next(&mut self) -> Tok<'d> {
        let d = self.d;
        let mut at = self.at;
        loop {
            match d.get(at) {
                None => {
                    self.at = at;
                    return Tok::End;
                }
                Some(b'#') => {
                    while at < d.len() && d[at] != b'\n' && d[at] != b'\r' {
                        at += 1;
                    }
                }
                Some(&b) if b <= b' ' || b == b',' => at += 1,
                Some(_) => break,
            }
        }
        let start = at;
        at += 1;
        let tok = match d[start] {
            b'{' => Tok::Open,
            b'}' => Tok::Close,
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b'"' => {
                while let Some(&c) = d.get(at) {
                    match c {
                        b'\\' => at += 2,
                        b'"' => break,
                        _ => at += 1,
                    }
                }
                let end = at.min(d.len());
                at = (at + 1).min(d.len());
                Tok::Str(&d[start + 1..end])
            }
            _ => {
                while let Some(&c) = d.get(at) {
                    if c <= b' ' || matches!(c, b'{' | b'}' | b'[' | b']' | b'"' | b',' | b'#') {
                        break;
                    }
                    at += 1;
                }
                Tok::Word(&d[start..at])
            }
        };
        self.at = at;
        tok
    }

    fn peek(&mut self) -> Tok<'d> {
        let at = self.at;
        let t = self.next();
        self.at = at;
        t
    }

    /// The token after the next.
    fn peek2(&mut self) -> Tok<'d> {
        let at = self.at;
        self.next();
        let t = self.next();
        self.at = at;
        t
    }
}

/// Whether a word is a number (or meant as one).
#[inline]
fn numeric(w: &[u8]) -> bool {
    matches!(w.first(), Some(b'0'..=b'9' | b'-' | b'+' | b'.'))
}

/// A word as a number: decimal or `0x` hexadecimal (as images give pixels); NaN for neither.
#[inline]
fn number(w: &[u8]) -> f64 {
    if let Some(v) = f64_from(w) {
        return v;
    }
    let hex = w.strip_prefix(b"0x").or_else(|| w.strip_prefix(b"0X"));
    hex.and_then(|h| std::str::from_utf8(h).ok())
        .and_then(|h| u32::from_str_radix(h, 16).ok())
        .map_or(f64::NAN, f64::from)
}

/// A string's text, its `\"` and `\\` escapes undone.
fn unescape(s: &[u8]) -> String {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i] == b'\\' && i + 1 < s.len() {
            i += 1;
        }
        out.push(s[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn text(w: &[u8]) -> String {
    String::from_utf8_lossy(w).into_owned()
}

struct Parser<'d> {
    lx: Lexer<'d>,
    g: Graph,
    defs: HashMap<&'d [u8], usize>,
    depth: usize,
}

/// A classic file's scene graph, as much of it as reads.
pub(super) fn parse(d: &[u8]) -> Graph {
    let mut p = Parser {
        lx: Lexer { d, at: 0 },
        g: Graph::default(),
        defs: HashMap::new(),
        depth: 0,
    };
    loop {
        match p.lx.next() {
            Tok::End => break,
            Tok::Word(w) => {
                if p.statement(w) {
                    continue;
                }
                if let Some(id) = p.node(w) {
                    p.g.roots.push(id);
                }
            }
            Tok::Open | Tok::LBracket => {
                p.lx.at = p.lx.at.saturating_sub(1);
                p.skip_balanced();
            }
            _ => {}
        }
    }
    p.g
}

impl<'d> Parser<'d> {
    /// Skips the bracketed or braced block that's next, if one is.
    fn skip_balanced(&mut self) {
        if !matches!(self.lx.peek(), Tok::Open | Tok::LBracket) {
            return;
        }
        let mut depth = 0usize;
        loop {
            match self.lx.next() {
                Tok::Open | Tok::LBracket => depth += 1,
                Tok::Close | Tok::RBracket => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                Tok::End => return,
                _ => {}
            }
        }
    }

    fn word(&mut self) -> Option<&'d [u8]> {
        match self.lx.next() {
            Tok::Word(w) => Some(w),
            _ => None,
        }
    }

    /// Reads past a statement that isn't a node, for keyword `w` just read; whether it was one.
    fn statement(&mut self, w: &'d [u8]) -> bool {
        match w {
            b"PROTO" => {
                self.proto();
                return true;
            }
            b"EXTERNPROTO" => {
                let name = self.word().map(text);
                // Its own names, apart from the file's.
                let defs = std::mem::take(&mut self.defs);
                let _ = self.interface(false);
                self.defs = defs;
                let urls = match self.value() {
                    Some(Val::Strs(urls)) => urls,
                    _ => Vec::new(),
                };
                if let Some(name) = name {
                    self.g.externs.push((name, urls));
                }
                return true;
            }
            b"ROUTE" => {
                // `from.field TO to.field`
                for _ in 0..3 {
                    self.word();
                }
            }
            b"IMPORT" | b"EXPORT" => {
                self.word();
                if self.lx.peek() == Tok::Word(b"AS") {
                    self.lx.next();
                    self.word();
                }
            }
            b"PROFILE" | b"COMPONENT" => {
                self.word();
                if self.lx.peek() == Tok::Word(b":") {
                    self.lx.next();
                    self.word();
                }
            }
            b"META" => {
                let mut pair = Vec::new();
                for _ in 0..2 {
                    if let Tok::Str(s) = self.lx.peek() {
                        self.lx.next();
                        pair.push(unescape(s));
                    }
                }
                if let Ok([name, content]) = <[String; 2]>::try_from(pair) {
                    self.g.meta.push((name, content));
                }
            }
            b"UNIT" => {
                let category = self.word();
                let name = self.word();
                if let Some(f) = self.word()
                    && category == Some(b"length")
                {
                    self.g.unit = Some(number(f));
                    self.g.unit_name = name.map(text);
                }
                return true;
            }
            _ => return false,
        }
        if !matches!(w, b"PROFILE" | b"COMPONENT" | b"META") {
            self.g.skip(&text(w));
        }
        true
    }

    /// A prototype, `PROTO` just read: `Name [ interface ] { body }`.
    fn proto(&mut self) {
        let Some(name) = self.word().map(text) else {
            return;
        };
        if self.depth >= MAX_DEPTH {
            self.skip_balanced();
            self.skip_balanced();
            self.g.skip("PROTO");
            return;
        }
        self.depth += 1;
        // Its names are its own, apart from the file's.
        let defs = std::mem::take(&mut self.defs);
        let fields = self.interface(true);
        let mut body = Vec::new();
        if self.lx.peek() == Tok::Open {
            self.lx.next();
            loop {
                match self.lx.next() {
                    Tok::Close | Tok::End => break,
                    Tok::Word(w) => {
                        if self.statement(w) {
                            continue;
                        }
                        body.extend(self.node(w));
                    }
                    Tok::Open | Tok::LBracket => {
                        self.lx.at = self.lx.at.saturating_sub(1);
                        self.skip_balanced();
                    }
                    _ => {}
                }
            }
        }
        self.defs = defs;
        self.depth -= 1;
        self.g.protos.push(Proto { name, fields, body });
    }

    /// A prototype's interface, in brackets: each field's name and default (`defaults`, else
    /// none); events are passed over.
    fn interface(&mut self, defaults: bool) -> Vec<(String, Option<Val>)> {
        let mut fields = Vec::new();
        if self.lx.peek() != Tok::LBracket {
            return fields;
        }
        self.lx.next();
        loop {
            let at = self.lx.at;
            match self.lx.next() {
                Tok::RBracket | Tok::End => break,
                Tok::Word(access) => {
                    // `access type name`, then a field's default.
                    let field = matches!(access, b"field" | b"exposedField" | b"initializeOnly" | b"inputOutput");
                    let (Some(_), Some(name)) = (self.word(), self.word()) else {
                        break;
                    };
                    if field {
                        let v = if defaults { self.value() } else { None };
                        fields.push((text(name), v));
                    }
                }
                Tok::Close => {
                    // The interface was never closed.
                    self.lx.at = at;
                    break;
                }
                Tok::Open | Tok::LBracket => {
                    self.lx.at = at;
                    self.skip_balanced();
                }
                Tok::Str(_) => {}
            }
        }
        fields
    }

    /// The node that word `w` (just read) starts: `DEF name Kind {...}`, `USE name`, or
    /// `Kind {...}`; `None` for `NULL`, a node skipped, or a word that starts none.
    fn node(&mut self, w: &'d [u8]) -> Option<usize> {
        match w {
            b"DEF" => {
                let name = self.word()?;
                let kind = self.word()?;
                self.body(kind, Some(name))
            }
            b"USE" => {
                let name = self.word()?;
                self.defs.get(name).copied()
            }
            b"NULL" => None,
            kind => self.body(kind, None),
        }
    }

    /// A node's body, its kind just read.
    fn body(&mut self, kind: &'d [u8], name: Option<&'d [u8]>) -> Option<usize> {
        if self.lx.peek() != Tok::Open {
            return None;
        }
        if kind == b"Script" || self.depth >= MAX_DEPTH {
            self.skip_balanced();
            self.g.skip(&text(kind));
            return None;
        }
        self.lx.next();
        let id = self.g.add(&text(kind), name.map(text));
        if let Some(n) = name {
            self.defs.insert(n, id);
        }
        self.depth += 1;
        loop {
            match self.lx.next() {
                Tok::Close | Tok::End => break,
                Tok::Word(w) => {
                    if self.statement(w) {
                        continue;
                    }
                    if let Some(v) = self.value() {
                        self.g.set(id, &text(w), v);
                    }
                }
                Tok::Open | Tok::LBracket => {
                    self.lx.at = self.lx.at.saturating_sub(1);
                    self.skip_balanced();
                }
                _ => {}
            }
        }
        self.depth -= 1;
        Some(id)
    }

    /// A field's value, its name just read; `None` when there's none to keep.
    fn value(&mut self) -> Option<Val> {
        match self.lx.peek() {
            Tok::LBracket => {
                self.lx.next();
                Some(self.list())
            }
            Tok::Str(s) => {
                self.lx.next();
                Some(Val::Strs(vec![unescape(s)]))
            }
            Tok::Word(w) if numeric(w) => {
                let mut v = Vec::new();
                while let Tok::Word(w) = self.lx.peek()
                    && numeric(w)
                {
                    self.lx.next();
                    v.push(number(w));
                }
                Some(Val::Nums(v))
            }
            Tok::Word(b"TRUE") => {
                self.lx.next();
                Some(Val::Bools(vec![true]))
            }
            Tok::Word(b"FALSE") => {
                self.lx.next();
                Some(Val::Bools(vec![false]))
            }
            Tok::Word(b"NULL") => {
                self.lx.next();
                Some(Val::Nodes(Vec::new()))
            }
            Tok::Word(b"IS") => {
                self.lx.next();
                self.word().map(|f| Val::Is(text(f)))
            }
            // A node, or else the next field's name (this field had no value).
            Tok::Word(w) if matches!(w, b"DEF" | b"USE") || self.lx.peek2() == Tok::Open => {
                self.lx.next();
                self.node(w).map(|id| Val::Nodes(vec![id]))
            }
            _ => None,
        }
    }

    /// A bracketed list's items, its `[` just read: numbers, strings, booleans or nodes,
    /// whichever it holds.
    fn list(&mut self) -> Val {
        let mut nums = Vec::new();
        let mut strs = Vec::new();
        let mut bools = Vec::new();
        let mut nodes = Vec::new();
        loop {
            let at = self.lx.at;
            match self.lx.next() {
                Tok::RBracket | Tok::End => break,
                Tok::Word(w) if numeric(w) => nums.push(number(w)),
                Tok::Str(s) => strs.push(unescape(s)),
                Tok::Word(b"TRUE") => bools.push(true),
                Tok::Word(b"FALSE") => bools.push(false),
                Tok::Word(w) => {
                    if !self.statement(w)
                        && let Some(id) = self.node(w)
                    {
                        nodes.push(id);
                    }
                }
                // The node's end: the list was never closed.
                Tok::Close => {
                    self.lx.at = at;
                    break;
                }
                Tok::Open | Tok::LBracket => {
                    self.lx.at = at;
                    self.skip_balanced();
                }
            }
        }
        if !nodes.is_empty() {
            Val::Nodes(nodes)
        } else if !strs.is_empty() {
            Val::Strs(strs)
        } else if !bools.is_empty() {
            Val::Bools(bools)
        } else {
            Val::Nums(nums)
        }
    }
}
