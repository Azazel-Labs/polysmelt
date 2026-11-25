//! The accessors, buffer views and animations readers build, held as plain values until the
//! document is written or its JSON is asked for. A JSON object for each would be built,
//! copied, read back by repair and serialized: for a model with thousands of animation
//! channels, most of the time a load takes.
//!
//! What's written is what the JSON objects would have written, byte for byte: keys in the
//! order a [`Map`] keeps them (sorted), numbers as `serde_json` formats them.

use std::io::Write as _;

use serde_json::{Map, Value, json};

use crate::{Context as _, ErrorKind, Result};

/// An accessor's `min` or `max`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Limits {
    /// Floats: the `f32`s the data holds, widened, as JSON numbers hold them.
    Float(Vec<f64>),
    /// Integers, for integer components.
    Int(Vec<i64>),
}

impl Limits {
    pub(crate) fn floats(v: &[f32]) -> Self {
        Limits::Float(v.iter().map(|&x| f64::from(x)).collect())
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Limits::Float(v) => v.len(),
            Limits::Int(v) => v.len(),
        }
    }

    /// Component `i` as JSON reads it: `None` for a float that isn't finite (written as
    /// `null`).
    #[allow(clippy::cast_precision_loss, reason = "as `Value::as_f64` reads an integer")]
    pub(crate) fn get(&self, i: usize) -> Option<f64> {
        match self {
            Limits::Float(v) => v.get(i).copied().filter(|x| x.is_finite()),
            Limits::Int(v) => v.get(i).map(|&x| x as f64),
        }
    }

    fn to_value(&self) -> Value {
        match self {
            Limits::Float(v) => v.iter().map(|&x| Value::from(x)).collect(),
            Limits::Int(v) => v.iter().map(|&x| Value::from(x)).collect(),
        }
    }

    fn write(&self, w: &mut Vec<u8>) {
        w.push(b'[');
        match self {
            Limits::Float(v) => {
                for (i, &x) in v.iter().enumerate() {
                    if i > 0 {
                        w.push(b',');
                    }
                    float(w, x);
                }
            }
            Limits::Int(v) => {
                for (i, &x) in v.iter().enumerate() {
                    if i > 0 {
                        w.push(b',');
                    }
                    int(w, x);
                }
            }
        }
        w.push(b']');
    }
}

/// An accessor: its JSON's members.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Accessor {
    pub buffer_view: Option<usize>,
    pub byte_offset: Option<u64>,
    pub component_type: u32,
    pub count: usize,
    /// `type`: `SCALAR`, `VEC3`, ...
    pub kind: &'static str,
    pub min: Option<Limits>,
    pub max: Option<Limits>,
    /// Written only when set.
    pub normalized: bool,
    pub sparse: Option<Sparse>,
}

/// A sparse accessor's replacements: `count` elements at the indices in view `indices` (of
/// component type `index_type`), their values in view `values`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sparse {
    pub count: usize,
    pub indices: usize,
    pub index_type: u32,
    pub values: usize,
}

/// A buffer view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct View {
    pub buffer: usize,
    pub byte_offset: Option<u64>,
    pub byte_length: u64,
    pub target: Option<u32>,
}

/// An animation.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Animation {
    pub channels: Vec<Channel>,
    pub samplers: Vec<Sampler>,
    pub name: Option<String>,
    pub extras: Option<Value>,
}

/// An animation channel: sampler `sampler` drives `path` of node `node`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Channel {
    pub sampler: usize,
    pub node: usize,
    pub path: &'static str,
}

/// An animation sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sampler {
    pub input: usize,
    pub output: usize,
    pub interpolation: &'static str,
}

/// A document's accessors, buffer views and animations, when a reader built them: the JSON
/// holds none of these lists then.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Tables {
    pub accessors: Vec<Accessor>,
    pub views: Vec<View>,
    pub animations: Vec<Animation>,
}

impl Accessor {
    pub(crate) fn to_value(&self) -> Value {
        let mut m = Map::new();
        if let Some(v) = self.buffer_view {
            m.insert("bufferView".into(), json!(v));
        }
        if let Some(o) = self.byte_offset {
            m.insert("byteOffset".into(), json!(o));
        }
        m.insert("componentType".into(), json!(self.component_type));
        m.insert("count".into(), json!(self.count));
        if let Some(x) = &self.max {
            m.insert("max".into(), x.to_value());
        }
        if let Some(x) = &self.min {
            m.insert("min".into(), x.to_value());
        }
        if self.normalized {
            m.insert("normalized".into(), json!(true));
        }
        if let Some(s) = &self.sparse {
            m.insert(
                "sparse".into(),
                json!({
                    "count": s.count,
                    "indices": { "bufferView": s.indices, "componentType": s.index_type },
                    "values": { "bufferView": s.values },
                }),
            );
        }
        m.insert("type".into(), json!(self.kind));
        Value::Object(m)
    }

    fn write(&self, w: &mut Vec<u8>) {
        let mut o = Object::start(w);
        if let Some(v) = self.buffer_view {
            o.key("bufferView");
            uint(o.w, v as u64);
        }
        if let Some(x) = self.byte_offset {
            o.key("byteOffset");
            uint(o.w, x);
        }
        o.key("componentType");
        uint(o.w, u64::from(self.component_type));
        o.key("count");
        uint(o.w, self.count as u64);
        if let Some(x) = &self.max {
            o.key("max");
            x.write(o.w);
        }
        if let Some(x) = &self.min {
            o.key("min");
            x.write(o.w);
        }
        if self.normalized {
            o.key("normalized");
            o.w.extend_from_slice(b"true");
        }
        if let Some(s) = &self.sparse {
            o.key("sparse");
            let mut so = Object::start(o.w);
            so.key("count");
            uint(so.w, s.count as u64);
            so.key("indices");
            let mut io = Object::start(so.w);
            io.key("bufferView");
            uint(io.w, s.indices as u64);
            io.key("componentType");
            uint(io.w, u64::from(s.index_type));
            io.end();
            so.key("values");
            let mut vo = Object::start(so.w);
            vo.key("bufferView");
            uint(vo.w, s.values as u64);
            vo.end();
            so.end();
        }
        o.key("type");
        literal(o.w, self.kind);
        o.end();
    }
}

impl View {
    pub(crate) fn to_value(self) -> Value {
        let mut m = Map::new();
        m.insert("buffer".into(), json!(self.buffer));
        m.insert("byteLength".into(), json!(self.byte_length));
        if let Some(o) = self.byte_offset {
            m.insert("byteOffset".into(), json!(o));
        }
        if let Some(t) = self.target {
            m.insert("target".into(), json!(t));
        }
        Value::Object(m)
    }

    fn write(&self, w: &mut Vec<u8>) {
        let mut o = Object::start(w);
        o.key("buffer");
        uint(o.w, self.buffer as u64);
        o.key("byteLength");
        uint(o.w, self.byte_length);
        if let Some(x) = self.byte_offset {
            o.key("byteOffset");
            uint(o.w, x);
        }
        if let Some(t) = self.target {
            o.key("target");
            uint(o.w, u64::from(t));
        }
        o.end();
    }

    /// Moves the view to its buffer's place in a GLB, `home[buffer]` (buffer, offset), as
    /// [`rehome`](crate::gltf::rehome) moves a JSON one.
    pub(crate) fn rehome(&mut self, home: &[(usize, usize)]) -> Result<()> {
        let &(nb, base) = home
            .get(self.buffer)
            .with_context(ErrorKind::Corrupt, || format!("missing buffer {}", self.buffer))?;
        let off = self.byte_offset.unwrap_or(0);
        let at = off
            .checked_add(base as u64)
            .context(ErrorKind::Truncated, "byteOffset is too large")?;
        self.buffer = nb;
        if nb == 0 || off != 0 {
            self.byte_offset = Some(at);
        }
        Ok(())
    }
}

impl Animation {
    pub(crate) fn to_value(&self) -> Value {
        let channels: Vec<Value> = self
            .channels
            .iter()
            .map(|c| json!({ "sampler": c.sampler, "target": { "node": c.node, "path": c.path } }))
            .collect();
        let samplers: Vec<Value> = self
            .samplers
            .iter()
            .map(|s| json!({ "input": s.input, "interpolation": s.interpolation, "output": s.output }))
            .collect();
        let mut m = Map::new();
        m.insert("channels".into(), Value::Array(channels));
        if let Some(e) = &self.extras {
            m.insert("extras".into(), e.clone());
        }
        if let Some(n) = &self.name {
            m.insert("name".into(), json!(n));
        }
        m.insert("samplers".into(), Value::Array(samplers));
        Value::Object(m)
    }

    fn write(&self, w: &mut Vec<u8>) -> Result<()> {
        let mut o = Object::start(w);
        o.key("channels");
        list(o.w, &self.channels, |w, c| {
            let mut co = Object::start(w);
            co.key("sampler");
            uint(co.w, c.sampler as u64);
            co.key("target");
            let mut t = Object::start(co.w);
            t.key("node");
            uint(t.w, c.node as u64);
            t.key("path");
            literal(t.w, c.path);
            t.end();
            co.end();
        });
        if let Some(e) = &self.extras {
            o.key("extras");
            value(o.w, e)?;
        }
        if let Some(n) = &self.name {
            o.key("name");
            string(o.w, n)?;
        }
        o.key("samplers");
        list(o.w, &self.samplers, |w, s| {
            let mut so = Object::start(w);
            so.key("input");
            uint(so.w, s.input as u64);
            so.key("interpolation");
            literal(so.w, s.interpolation);
            so.key("output");
            uint(so.w, s.output as u64);
            so.end();
        });
        o.end();
        Ok(())
    }
}

impl Tables {
    /// Adds the lists to `json` as JSON. `pruned` leaves out empty ones, as repair does;
    /// otherwise the accessors and views are there whenever there are views, as a reader
    /// writes them.
    pub(crate) fn insert_into(&self, json: &mut Map<String, Value>, pruned: bool) {
        let views = !self.views.is_empty();
        if !self.accessors.is_empty() || (views && !pruned) {
            json.insert(
                "accessors".into(),
                self.accessors.iter().map(Accessor::to_value).collect(),
            );
        }
        if views {
            json.insert("bufferViews".into(), self.views.iter().map(|&v| v.to_value()).collect());
        }
        if !self.animations.is_empty() {
            json.insert(
                "animations".into(),
                self.animations.iter().map(Animation::to_value).collect(),
            );
        }
    }
}

/// Writes `json` with `tables`' lists (those not empty) among its members, in key order: what
/// writing `json` with [`Tables::insert_into`] (pruned) done to it would write.
pub(crate) fn write_json(w: &mut Vec<u8>, json: &Map<String, Value>, tables: Option<&Tables>) -> Result<()> {
    let mut typed: Vec<&str> = Vec::new();
    if let Some(t) = tables {
        // In key order.
        for (key, has) in [
            ("accessors", !t.accessors.is_empty()),
            ("animations", !t.animations.is_empty()),
            ("bufferViews", !t.views.is_empty()),
        ] {
            if has {
                typed.push(key);
            }
        }
    }
    let mut pending = typed.into_iter().peekable();
    let write_typed = |w: &mut Vec<u8>, key: &str| -> Result<()> {
        let Some(t) = tables else { return Ok(()) };
        match key {
            "accessors" => list(w, &t.accessors, |w, a| a.write(w)),
            "bufferViews" => list(w, &t.views, |w, v| v.write(w)),
            _ => {
                w.push(b'[');
                for (i, a) in t.animations.iter().enumerate() {
                    if i > 0 {
                        w.push(b',');
                    }
                    a.write(w)?;
                }
                w.push(b']');
            }
        }
        Ok(())
    };
    let mut o = Object::start(w);
    for (k, v) in json {
        while let Some(&key) = pending.peek() {
            if key >= k.as_str() {
                break;
            }
            o.key(key);
            write_typed(o.w, key)?;
            pending.next();
        }
        if pending.peek() == Some(&k.as_str()) {
            // The table's list stands for it.
            continue;
        }
        o.comma();
        string(o.w, k)?;
        o.w.push(b':');
        value(o.w, v)?;
    }
    for key in pending {
        o.key(key);
        write_typed(o.w, key)?;
    }
    o.end();
    Ok(())
}

/// A JSON object being written: commas between its members.
struct Object<'w> {
    w: &'w mut Vec<u8>,
    first: bool,
}

impl<'w> Object<'w> {
    fn start(w: &'w mut Vec<u8>) -> Self {
        w.push(b'{');
        Object { w, first: true }
    }

    fn comma(&mut self) {
        if !self.first {
            self.w.push(b',');
        }
        self.first = false;
    }

    /// A member's key, which needs no escaping.
    fn key(&mut self, k: &str) {
        self.comma();
        literal(self.w, k);
        self.w.push(b':');
    }

    fn end(self) {
        self.w.push(b'}');
    }
}

fn list<T>(w: &mut Vec<u8>, items: &[T], mut each: impl FnMut(&mut Vec<u8>, &T)) {
    w.push(b'[');
    for (i, x) in items.iter().enumerate() {
        if i > 0 {
            w.push(b',');
        }
        each(w, x);
    }
    w.push(b']');
}

/// A string that needs no escaping: a key or a name glTF defines.
fn literal(w: &mut Vec<u8>, s: &str) {
    debug_assert!(s.bytes().all(|c| c.is_ascii_graphic() && c != b'"' && c != b'\\'));
    w.push(b'"');
    w.extend_from_slice(s.as_bytes());
    w.push(b'"');
}

fn value(w: &mut Vec<u8>, v: &Value) -> Result<()> {
    serde_json::to_writer(w, v).context(ErrorKind::Invalid, "writing the glTF JSON")
}

fn string(w: &mut Vec<u8>, s: &str) -> Result<()> {
    serde_json::to_writer(w, s).context(ErrorKind::Invalid, "writing the glTF JSON")
}

fn uint(w: &mut Vec<u8>, n: u64) {
    // Writing to a `Vec` can't fail.
    let _ = write!(w, "{n}");
}

fn int(w: &mut Vec<u8>, n: i64) {
    let _ = write!(w, "{n}");
}

/// A float as `serde_json` writes one: `null` when it isn't finite.
fn float(w: &mut Vec<u8>, x: f64) {
    let _ = serde_json::to_writer(w, &x);
}
