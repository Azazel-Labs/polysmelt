//! A tile's header, and its feature table and batch table: each a JSON object and a binary body
//! the JSON's properties point into.
//!
//! Every length in the header is held against the bytes present before anything is read, and
//! every property against its body: a count the file can't hold is an error, not an
//! allocation.

use serde_json::{Map, Value, json};

use crate::bytes::Bin;
use crate::num::{f32_of, usize_of};
use crate::{Context as _, ErrorKind, Result};

/// The most batch table values kept as JSON in a node's extras; properties past that are left
/// out and named in `batchTableDropped`. A JSON number takes several times the room its bytes
/// did.
pub(super) const MAX_TABLE_VALUES: usize = 1 << 22;

/// A b3dm, i3dm or pnts tile taken apart.
pub(super) struct Tile<'a> {
    pub feature: Table<'a>,
    pub batch: Table<'a>,
    /// What follows the tables: the glTF, or a URI to it.
    pub body: &'a [u8],
    /// The batch length a legacy b3dm header gives, which has no feature table to say it.
    pub batch_length: Option<u64>,
    /// An i3dm's `gltfFormat`: 0 for a URI, 1 for an embedded binary glTF.
    pub gltf_format: u32,
}

/// The smallest value a header field holds when it is the first bytes of a batch table's JSON
/// or of a glTF, as it is in a b3dm with one of the two legacy headers: `"` or `g` in the top
/// byte. No table is that long.
const LEGACY: u32 = 0x2200_0000;

/// Takes the tile in `data` apart. `what` is its magic: `b3dm`, `i3dm` or `pnts`.
///
/// A b3dm may have either legacy header the specification describes:
/// `[batchLength][batchTableByteLength]`, or
/// `[batchTableJsonByteLength][batchTableBinaryByteLength][batchLength]`, each followed
/// directly by the batch table. They're told from the current one by the field that would be
/// the batch table's JSON or binary length holding text instead.
pub(super) fn parse<'a>(data: &'a [u8], what: &'static str) -> Result<Tile<'a>> {
    let mut r = Bin::new(data, what);
    r.skip(4)?;
    let version = r.u32()?;
    ensure!(version == 1, Unsupported, "{what} version {version} isn't 1");
    let declared = usize_of(u64::from(r.u32()?));
    ensure!(
        declared <= data.len(),
        Truncated,
        "{what} tile is cut short: it holds {} bytes but declares {declared}",
        data.len()
    );
    ensure!(
        declared >= 12,
        Corrupt,
        "{what} tile declares a length shorter than its header"
    );
    let mut r = Bin::new(&data[..declared], what);
    r.seek(12)?;
    let mut lens = [r.u32()?, r.u32()?, r.u32()?, r.u32()?];
    let mut batch_length = None;
    if what == "b3dm" {
        if lens[2] >= LEGACY {
            r.seek(20)?;
            batch_length = Some(u64::from(lens[0]));
            lens = [0, 0, lens[1], 0];
        } else if lens[3] >= LEGACY {
            r.seek(24)?;
            batch_length = Some(u64::from(lens[2]));
            lens = [0, 0, lens[0], lens[1]];
        }
    }
    let gltf_format = if what == "i3dm" { r.u32()? } else { 0 };
    let [fj, fb, bj, bb] = lens.map(|n| usize_of(u64::from(n)));
    let feature = Table::new(r.bytes(fj)?, r.bytes(fb)?, what, "feature table")?;
    let batch = Table::new(r.bytes(bj)?, r.bytes(bb)?, what, "batch table")?;
    Ok(Tile {
        feature,
        batch,
        body: r.rest(),
        batch_length,
        gltf_format,
    })
}

/// A component type, as a batch table names it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Component {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Component {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "BYTE" => Self::I8,
            "UNSIGNED_BYTE" => Self::U8,
            "SHORT" => Self::I16,
            "UNSIGNED_SHORT" => Self::U16,
            "INT" => Self::I32,
            "UNSIGNED_INT" => Self::U32,
            "FLOAT" => Self::F32,
            "DOUBLE" => Self::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// The value in `b`, which is [`size`](Self::size) bytes.
    fn read(self, b: &[u8]) -> f64 {
        let two = || [b[0], b[1]];
        let four = || [b[0], b[1], b[2], b[3]];
        match self {
            Self::I8 => f64::from(b[0].cast_signed()),
            Self::U8 => f64::from(b[0]),
            Self::I16 => f64::from(i16::from_le_bytes(two())),
            Self::U16 => f64::from(u16::from_le_bytes(two())),
            Self::I32 => f64::from(i32::from_le_bytes(four())),
            Self::U32 => f64::from(u32::from_le_bytes(four())),
            Self::F32 => f64::from(f32::from_le_bytes(four())),
            Self::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        }
    }
}

/// How many components a batch table `type` has.
fn components(name: &str) -> Option<usize> {
    Some(match name {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        _ => return None,
    })
}

/// A feature table or batch table.
pub(super) struct Table<'a> {
    pub json: Map<String, Value>,
    bin: &'a [u8],
    /// The tile's magic and the table's name, for errors.
    tile: &'static str,
    name: &'static str,
}

impl<'a> Table<'a> {
    fn new(json: &[u8], bin: &'a [u8], tile: &'static str, name: &'static str) -> Result<Self> {
        // The JSON is padded with spaces; some writers pad with NULs.
        let end = json
            .iter()
            .rposition(|&b| b != 0 && !b.is_ascii_whitespace())
            .map_or(0, |i| i + 1);
        let text = json[..end].trim_ascii_start();
        let json = if text.is_empty() {
            Map::new()
        } else {
            match serde_json::from_slice::<Value>(text) {
                Ok(Value::Object(m)) => m,
                Ok(_) => bail!(Corrupt, "{tile} {name} isn't a JSON object"),
                Err(e) => bail!(Corrupt, "{tile} {name} isn't JSON: {e}"),
            }
        };
        Ok(Self { json, bin, tile, name })
    }

    /// Whether `key` is `true`.
    pub fn flag(&self, key: &str) -> bool {
        self.json.get(key).and_then(Value::as_bool).unwrap_or(false)
    }

    /// The table's part of extension `name`.
    pub fn extension(&self, name: &str) -> Option<&Value> {
        self.json.get("extensions")?.get(name)
    }

    /// The whole binary body.
    #[cfg_attr(
        not(feature = "draco"),
        allow(dead_code, reason = "only Draco data is found by an offset of its own")
    )]
    pub fn body(&self) -> &'a [u8] {
        self.bin
    }

    /// An error that says property `key` runs past the body.
    #[cold]
    fn past(&self, key: &str) -> crate::Error {
        err!(
            Truncated,
            "{} {} property {key} runs past the binary body",
            self.tile,
            self.name
        )
    }

    /// `n` values of `size` bytes at the body offset `v` gives.
    fn at(&self, key: &str, v: &Value, n: usize, size: usize) -> Result<&'a [u8]> {
        let off = v
            .get("byteOffset")
            .and_then(Value::as_u64)
            .with_context(ErrorKind::Corrupt, || {
                format!(
                    "{} {} property {key} has no byteOffset into the binary body",
                    self.tile, self.name
                )
            })?;
        n.checked_mul(size)
            .and_then(|len| {
                let off = usize::try_from(off).ok()?;
                self.bin.get(off..off.checked_add(len)?)
            })
            .ok_or_else(|| self.past(key))
    }

    /// A global property of `n` numbers: a JSON number (for one) or list, or a reference to
    /// `n` values of `component` in the body.
    pub fn global(&self, key: &str, n: usize, component: Component) -> Result<Option<Vec<f64>>> {
        let Some(v) = self.json.get(key) else {
            return Ok(None);
        };
        let values: Vec<f64> = match v {
            Value::Number(x) if n == 1 => x.as_f64().into_iter().collect(),
            Value::Array(list) if list.len() == n => list.iter().filter_map(Value::as_f64).collect(),
            Value::Object(_) => self
                .at(key, v, n, component.size())?
                .chunks_exact(component.size())
                .map(|b| component.read(b))
                .collect(),
            _ => Vec::new(),
        };
        ensure!(
            values.len() == n && values.iter().all(|x| x.is_finite()),
            Corrupt,
            "{} {} property {key} isn't {n} number{}",
            self.tile,
            self.name,
            if n == 1 { "" } else { "s" }
        );
        Ok(Some(values))
    }

    /// A global count.
    pub fn count(&self, key: &str) -> Result<Option<u64>> {
        let Some(v) = self.global(key, 1, Component::U32)? else {
            return Ok(None);
        };
        ensure!(
            v[0] >= 0.0 && v[0].fract() == 0.0 && v[0] <= f64::from(u32::MAX),
            Corrupt,
            "{} {} property {key} isn't a count",
            self.tile,
            self.name
        );
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a whole number in a u32's range"
        )]
        Ok(Some(v[0] as u64))
    }

    /// A global point or vector of three numbers.
    pub fn vec3(&self, key: &str) -> Result<Option<[f64; 3]>> {
        Ok(self.global(key, 3, Component::F32)?.map(|v| [v[0], v[1], v[2]]))
    }

    /// A per-feature property: `n` features of `comps` components each, read from the body as
    /// floats. Its component type is `component`, or with `typed` the one it names.
    pub fn floats(
        &self,
        key: &str,
        n: usize,
        comps: usize,
        component: Component,
        typed: bool,
    ) -> Result<Option<Vec<f32>>> {
        let Some(v) = self.json.get(key) else {
            return Ok(None);
        };
        let component = match v.get("componentType").and_then(Value::as_str).filter(|_| typed) {
            Some(name) => Component::named(name).with_context(ErrorKind::Corrupt, || {
                format!(
                    "{} {} property {key} has unknown component type {name:?}",
                    self.tile, self.name
                )
            })?,
            None => component,
        };
        let total = n.checked_mul(comps).ok_or_else(|| self.past(key))?;
        let bytes = self.at(key, v, total, component.size())?;
        Ok(Some(
            bytes
                .chunks_exact(component.size())
                .map(|b| f32_of(component.read(b)))
                .collect(),
        ))
    }

    /// A batch table property held in the body: how many components each of its `count`
    /// features has, and their values. `None` for a property listed in the JSON.
    pub fn binary(&self, key: &str, v: &Value, count: usize) -> Result<Option<(usize, Vec<f64>)>> {
        if v.get("byteOffset").is_none() {
            return Ok(None);
        }
        let (comps, component) = self.kind(key, v)?;
        let total = count.checked_mul(comps).ok_or_else(|| self.past(key))?;
        let bytes = self.at(key, v, total, component.size())?;
        Ok(Some((
            comps,
            bytes
                .chunks_exact(component.size())
                .map(|b| component.read(b))
                .collect(),
        )))
    }

    /// The components and component type a binary batch table property declares.
    pub fn kind(&self, key: &str, v: &Value) -> Result<(usize, Component)> {
        let named = |k: &str| v.get(k).and_then(Value::as_str);
        let comps = named("type").and_then(components);
        let component = named("componentType").and_then(Component::named);
        match (comps, component) {
            (Some(n), Some(c)) => Ok((n, c)),
            _ => bail!(
                Corrupt,
                "{} {} property {key} needs a known type and componentType",
                self.tile,
                self.name
            ),
        }
    }

    /// The batch table as JSON for a node's extras: each property a list of `count` values,
    /// the ones held in the body decoded (a vector as a list of its components). Its
    /// `extensions` and `extras` are kept as they are.
    pub fn listed(&self, count: usize) -> Result<Map<String, Value>> {
        let mut out = Map::new();
        let mut budget = MAX_TABLE_VALUES;
        let mut dropped = Vec::new();
        for (key, v) in &self.json {
            let Some((comps, values)) = self.binary(key, v, count)? else {
                out.insert(key.clone(), v.clone());
                continue;
            };
            if values.len() > budget {
                dropped.push(json!(key));
                continue;
            }
            budget -= values.len();
            let number = |x: &f64| if x.is_finite() { json!(x) } else { Value::Null };
            let list: Vec<Value> = if comps == 1 {
                values.iter().map(number).collect()
            } else {
                values
                    .chunks_exact(comps)
                    .map(|c| Value::Array(c.iter().map(number).collect()))
                    .collect()
            };
            out.insert(key.clone(), Value::Array(list));
        }
        if !dropped.is_empty() {
            out.insert("batchTableDropped".into(), Value::Array(dropped));
        }
        Ok(out)
    }
}
