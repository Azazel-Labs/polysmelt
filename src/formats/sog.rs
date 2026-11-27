//! SOG (Spatially Ordered Gaussians), `PlayCanvas`'s compressed gaussian splat format: a
//! `meta.json` and WebP images whose pixels hold each splat's quantized values, zipped as a
//! `.sog` file or loose in a folder, where the model is its `meta.json`. Versions 1 to 3.
//!
//! Read, a splat a pixel in row order: positions (16 bits an axis over two images, between the
//! meta's bounds, out of the format's logarithm); scales (bytes between the meta's bounds in
//! version 1, through a 256-entry codebook after, out of their logarithm); rotations (the
//! three smaller of four components and a byte naming the largest in versions 1 and 2, an
//! exponential map in version 3); base color and opacity (version 1 between bounds, the
//! opacity a logit; later the color through a codebook and the opacity a byte, linear in
//! version 2 and a square root in version 3); and the higher spherical harmonics, each splat
//! naming an entry of a palette image whose width tells the bands, up to degree 3 (version 3's
//! degree 4 is dropped: `KHR_gaussian_splatting` stops at 3). Version 3's opacities above 1
//! are clamped to 1, and its blind data are custom attributes, each field `_` and its name in
//! capitals, as floats (integers past 2^24 lose their low bits, and values that aren't finite
//! are 0); fields of a type unknown here are named in the asset's `extras.source`.
//!
//! The splats are in a 3D gaussian splatting PLY's axes (right, down and forward), flipped to
//! glTF's; the source info has the version, the generator as the tool, the comments, and -Y up
//! and right-handed, assumed. The training model, the largest opacity, the meta's `extras` and
//! whether the file was zipped stay in the asset's `extras.source`.
//!
//! The count is checked against [`Keep::max_points`] before a file is read, and nothing is
//! allocated for the splats until an image is found to hold a pixel for each; an image far
//! larger than its splats need is refused, as one of a single color is a few bytes whatever
//! its size. Images must be WebP; a streamed SOG's `lod-meta.json` isn't read, though each of
//! its chunks is.

use std::borrow::Cow;
use std::f64::consts::{FRAC_PI_2, SQRT_2};
use std::io::Cursor;

use serde_json::{Map, Value, json};

use super::Input;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::gaussian::{Splats, rotation, sigmoid};
use crate::num::{f32_of, usize_of};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Context as _, ErrorKind, Result};

/// The name of the file that lists the others.
const META: &str = "meta.json";
/// Palette entries in a row of the centroids image.
const ROW: usize = 64;
/// Spherical-harmonic coefficients a channel has beyond the base one, by bands.
const COEFFS: [usize; 4] = [3, 8, 15, 24];
/// The most coefficients kept: degree 3's.
const KEPT: usize = 15;
/// The most blind textures read: the format advises 8 at most.
const MAX_BLIND: usize = 32;
/// The UTF-8 byte order mark some editors put first.
const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// A value a splat, under an attribute's name.
type Column = (String, Vec<f32>);

/// Reads the file `meta.json` names as its argument.
type Read<'a> = dyn FnMut(&str) -> Result<Vec<u8>> + 'a;

pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let max = keep.points();
    if !input.head(4)?.starts_with(b"PK\x03\x04") {
        let dir = input.dir();
        let mut read = |name: &str| {
            assets
                .find(name, dir)
                .and_then(|path| assets.read(&path))
                .map(Cow::into_owned)
                .with_context(ErrorKind::MissingAsset, || format!("SOG image {name} wasn't found"))
        };
        return decode(&input.read()?, &mut read, max, false);
    }
    let mut zip =
        zip::ZipArchive::new(input.seekable()?).context(ErrorKind::Corrupt, "this SOG's zip can't be opened")?;
    // The files sit at the archive's root; in a folder, the shallowest `meta.json` names them.
    let meta = zip
        .file_names()
        .filter_map(Result::ok)
        .filter(|n| *n == META || n.ends_with("/meta.json"))
        .min_by_key(|n| n.len())
        .map(Cow::into_owned)
        .context(ErrorKind::UnknownFormat, "not a SOG file: its zip holds no meta.json")?;
    let folder = &meta[..meta.len() - META.len()];
    let mut read = |name: &str| {
        let entry = zip
            .by_name(&format!("{folder}{name}"))
            .ok()
            .with_context(ErrorKind::MissingAsset, || format!("this SOG's zip holds no {name}"))?;
        let packed = entry.compressed_size();
        super::inflate_capped(entry, packed, name)
    };
    let json = read(META)?;
    decode(&json, &mut read, max, true)
}

/// A decoded image: its pixels in row order, 3 or 4 bytes each.
struct Pixels {
    width: usize,
    height: usize,
    /// Bytes a pixel: 4 with alpha, 3 without.
    stride: usize,
    data: Vec<u8>,
}

impl Pixels {
    /// The first `n` pixels as red, green, blue and alpha; alpha is 255 in an image without.
    fn iter(&self, n: usize) -> impl Iterator<Item = [u8; 4]> + '_ {
        let opaque = self.stride == 3;
        self.data
            .chunks_exact(self.stride)
            .take(n)
            .map(move |p| [p[0], p[1], p[2], if opaque { 255 } else { p[3] }])
    }

    /// Pixel `k`'s red, green and blue; zeros past the image's end.
    fn rgb(&self, k: usize) -> [u8; 3] {
        self.data
            .get(k * self.stride..)
            .and_then(|p| p.first_chunk::<3>())
            .copied()
            .unwrap_or_default()
    }
}

/// Room for `n` items, or an error where there isn't memory for them.
fn room<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .ok()
        .context(ErrorKind::LimitExceeded, "not enough memory for the splats")?;
    Ok(v)
}

/// The image `name` decoded, when it holds at least `need` pixels and no more than `most`
/// gives for its width: a lossless image of one color is a few bytes whatever its size, so its
/// size is held to what the splats take.
fn image(read: &mut Read<'_>, name: &str, need: usize, most: &dyn Fn(usize) -> usize) -> Result<Pixels> {
    let bytes = read(name)?;
    let bad = || format!("SOG image {name} isn't a WebP image that decodes");
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes.as_slice()))
        .ok()
        .with_context(ErrorKind::Corrupt, bad)?;
    let (w, h) = decoder.dimensions();
    let (width, height) = (usize_of(u64::from(w)), usize_of(u64::from(h)));
    let pixels = width.saturating_mul(height);
    ensure!(
        pixels >= need,
        Corrupt,
        "SOG image {name} is {w} by {h}, fewer pixels than the {need} it should hold"
    );
    ensure!(
        pixels <= most(width),
        LimitExceeded,
        "SOG image {name} is {w} by {h}, far larger than what it should hold"
    );
    let stride = if decoder.has_alpha() { 4 } else { 3 };
    let len = decoder
        .output_buffer_size()
        .filter(|len| *len >= pixels * stride)
        .with_context(ErrorKind::Corrupt, bad)?;
    let mut data = room(len)?;
    data.resize(len, 0u8);
    decoder
        .read_image(&mut data)
        .ok()
        .with_context(ErrorKind::Corrupt, bad)?;
    Ok(Pixels {
        width,
        height,
        stride,
        data,
    })
}

/// The object `name` of the meta.
fn section<'a>(meta: &'a Value, name: &str) -> Result<&'a Value> {
    meta.get(name)
        .filter(|v| v.is_object())
        .with_context(ErrorKind::Corrupt, || format!("SOG meta.json has no {name}"))
}

/// The name of file `k` of section `name`.
fn file<'a>(section: &'a Value, name: &str, k: usize) -> Result<&'a str> {
    section
        .get("files")
        .and_then(|f| f.get(k))
        .and_then(Value::as_str)
        .with_context(ErrorKind::Corrupt, || {
            format!("SOG meta.json names too few files for {name}")
        })
}

/// The first `n` numbers of the array `key` of section `name`. An entry that isn't a number
/// (JSON has no way to write one that isn't finite) is NaN.
fn numbers<const N: usize>(section: &Value, name: &str, key: &str) -> Result<[f64; N]> {
    let list = section
        .get(key)
        .and_then(Value::as_array)
        .filter(|a| a.len() >= N)
        .with_context(ErrorKind::Corrupt, || {
            format!("SOG meta.json needs {N} numbers as the {key} of {name}")
        })?;
    Ok(std::array::from_fn(|k| list[k].as_f64().unwrap_or(f64::NAN)))
}

/// A byte's value in each of its 256 steps from `min` to `max`.
fn spread(min: f64, max: f64) -> [f64; 256] {
    #[allow(clippy::cast_precision_loss, reason = "an index below 256")]
    std::array::from_fn(|b| min + (max - min) * (b as f64 / 255.0))
}

/// The rotation (x, y, z, w) a pixel of versions 1 and 2 holds: alpha, from 252 to 255, names
/// the largest of w, x, y and z, and red, green and blue are the others in that order, from
/// -1/sqrt(2) to 1/sqrt(2). Another alpha says nothing: the identity.
fn smallest_three(p: [u8; 4]) -> [f32; 4] {
    let part = |v: u8| (f64::from(v) / 255.0 * 2.0 - 1.0) / SQRT_2;
    let (a, b, c) = (part(p[0]), part(p[1]), part(p[2]));
    let m = (1.0 - (a * a + b * b + c * c)).max(0.0).sqrt();
    let [w, x, y, z] = match p[3] {
        252 => [m, a, b, c],
        253 => [a, m, b, c],
        254 => [a, b, m, c],
        255 => [a, b, c, m],
        _ => [1.0, 0.0, 0.0, 0.0],
    };
    rotation(f32_of(x), f32_of(y), f32_of(z), f32_of(w))
}

/// The rotation (x, y, z, w) a pixel of version 3 holds: red, green and blue are a point in
/// the unit ball, 128 its center, whose direction is the rotation's axis and whose length its
/// half angle over a quarter turn.
fn exponential_map(p: [u8; 4]) -> [f32; 4] {
    let [px, py, pz] = [p[0], p[1], p[2]].map(|v| (f64::from(v) - 128.0) / 127.0);
    let r = (px * px + py * py + pz * pz).sqrt();
    let half = FRAC_PI_2 * r;
    let k = if r > 0.0 { half.sin() / r } else { 0.0 };
    rotation(f32_of(px * k), f32_of(py * k), f32_of(pz * k), f32_of(half.cos()))
}

/// A half float from its bits.
fn half(bits: u16) -> f32 {
    let (exponent, fraction) = ((bits >> 10) & 31, f32::from(bits & 1023));
    let v = match exponent {
        0 => fraction * 2f32.powi(-24),
        31 if fraction == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + fraction / 1024.0) * 2f32.powi(i32::from(e) - 15),
    };
    if bits >> 15 == 0 { v } else { -v }
}

/// How a blind field's bytes are read.
#[derive(Clone, Copy)]
enum Blind {
    /// An unsigned integer.
    Integer,
    Half,
    Float,
    /// An integer's share of its largest value, between the field's bounds.
    Unorm(f64, f64),
}

/// A blind field's bytes and reading, from its description; `None` for a type unknown here.
fn blind_field(field: &Value) -> Option<(usize, Blind)> {
    let range = || {
        let bound = |k: usize, or: f64| {
            let b = field.get("range").and_then(|r| r.get(k)).and_then(Value::as_f64);
            b.unwrap_or(or)
        };
        Blind::Unorm(bound(0, 0.0), bound(1, 1.0))
    };
    Some(match field.get("type").and_then(Value::as_str)? {
        "u8" => (1, Blind::Integer),
        "u16" => (2, Blind::Integer),
        "u24" => (3, Blind::Integer),
        "u32" => (4, Blind::Integer),
        "f16" => (2, Blind::Half),
        "f32" => (4, Blind::Float),
        "unorm8" => (1, range()),
        "unorm16" => (2, range()),
        _ => return None,
    })
}

/// Version 3's blind data: each texture's fields, packed in order into a pixel's four bytes,
/// as a column named `_` and the field's name in capitals. Also the names of fields left
/// unread: those of a type unknown here or past the pixel's end, and the fields after them.
fn blind(
    meta: &Value,
    read: &mut Read<'_>,
    n: usize,
    most: &dyn Fn(usize) -> usize,
) -> Result<(Vec<Column>, Vec<String>)> {
    let (mut columns, mut skipped) = (Vec::new(), Vec::new());
    let textures = meta
        .get("blind")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for texture in textures.iter().take(MAX_BLIND) {
        let fields = texture
            .get("fields")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        let mut at = 0;
        let mut read_as = Vec::new();
        for (k, field) in fields.iter().enumerate() {
            let name = |f: &Value| f.get("name").and_then(Value::as_str).unwrap_or_default().to_owned();
            let Some((size, kind)) = blind_field(field).filter(|(size, _)| at + size <= 4) else {
                skipped.extend(fields[k..].iter().map(name));
                break;
            };
            read_as.push((name(field), at, size, kind));
            at += size;
        }
        if read_as.is_empty() {
            continue;
        }
        let pixels = image(read, file(texture, "blind", 0)?, n, most)?;
        for (name, at, size, kind) in read_as {
            let mut column = room(n)?;
            column.extend(pixels.iter(n).map(|p| {
                let mut bytes = [0u8; 4];
                bytes[..size].copy_from_slice(&p[at..at + size]);
                let code = u32::from_le_bytes(bytes);
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    reason = "a float attribute holds integers up to 2^24 exactly, and the low 16 bits are a half float's"
                )]
                let v = match kind {
                    Blind::Integer => code as f32,
                    Blind::Half => half(code as u16),
                    Blind::Float => f32::from_bits(code),
                    Blind::Unorm(min, max) => {
                        let top = if size == 1 { 255.0 } else { 65535.0 };
                        f32_of(min + f64::from(code) / top * (max - min))
                    }
                };
                // glTF holds no value that isn't finite.
                if v.is_finite() { v } else { 0.0 }
            }));
            let name: String = name
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_uppercase()
                    } else {
                        '_'
                    }
                })
                .collect();
            columns.push((format!("_{name}"), column));
        }
    }
    Ok((columns, skipped))
}

/// The splats `json` (a `meta.json`) describes, up to `max` of them, its images read by name
/// through `read`.
fn decode(json: &[u8], read: &mut Read<'_>, max: usize, zipped: bool) -> Result<Document> {
    let json = json.strip_prefix(BOM).unwrap_or(json);
    // An entry that isn't JSON at all is a broken SOG, not another format: the file's own bytes say which.
    let meta: Value = serde_json::from_slice(json).map_err(|e| match crate::Error::json(e, json, "SOG meta.json") {
        e if e.kind() == ErrorKind::UnknownFormat => e.of_kind(ErrorKind::Corrupt).within("SOG meta.json isn't JSON"),
        e => e,
    })?;
    let means = meta
        .get("means")
        .filter(|v| v.is_object())
        .context(ErrorKind::UnknownFormat, "not a SOG meta.json: it has no means")?;
    // Version 1 has no version, and counts its splats in the shape of each array.
    let version = match meta.get("version") {
        None => 1,
        Some(v) => v
            .as_u64()
            .filter(|v| (2..=3).contains(v))
            .with_context(ErrorKind::Unsupported, || {
                format!("SOG version {v} isn't supported; versions 1 to 3 are")
            })?,
    };
    let count = if version == 1 {
        means.get("shape").and_then(|s| s.get(0))
    } else {
        meta.get("count")
    };
    let n = count
        .and_then(Value::as_u64)
        .map(usize_of)
        .context(ErrorKind::Corrupt, "SOG meta.json gives no count of splats")?;
    ensure!(
        n <= max,
        LimitExceeded,
        "{n} gaussian splats, more than the {max} allowed (Keep::max_points)"
    );
    // An image of a pixel a splat may be several times larger than its splats need, and no more.
    let spare = n.saturating_mul(4).saturating_add(1 << 20);
    let most = &|_width: usize| spare;
    let codebook = |section: &Value, name: &str| numbers::<256>(section, name, "codebook");
    let mut source = Map::new();
    // Each column is made once an image is known to hold its splats.
    let mut s = Splats::default();

    // Positions: the low and high bytes of 16 bits an axis, in the logarithm's range.
    let (mins, maxs) = (
        numbers::<3>(means, "means", "mins")?,
        numbers::<3>(means, "means", "maxs")?,
    );
    let low = image(read, file(means, "means", 0)?, n, most)?;
    let high = image(read, file(means, "means", 1)?, n, most)?;
    s.positions = room(n)?;
    s.positions.extend(low.iter(n).zip(high.iter(n)).map(|(l, h)| {
        std::array::from_fn(|k| {
            let q = f64::from(u16::from_le_bytes([l[k], h[k]])) / 65535.0;
            let v = mins[k] + (maxs[k] - mins[k]) * q;
            // The transform was sign(x) ln(|x| + 1).
            f32_of(v.abs().exp_m1().copysign(v))
        })
    }));
    drop((low, high));

    let quats = section(&meta, "quats")?;
    let pixels = image(read, file(quats, "quats", 0)?, n, most)?;
    let unpack = if version == 3 { exponential_map } else { smallest_three };
    s.rotations = room(n)?;
    s.rotations.extend(pixels.iter(n).map(unpack));

    // Scales, as logarithms: version 1's between an axis's bounds, later ones a codebook's.
    let scales = section(&meta, "scales")?;
    let tables: [[f64; 256]; 3] = if version == 1 {
        let (mins, maxs) = (
            numbers::<3>(scales, "scales", "mins")?,
            numbers::<3>(scales, "scales", "maxs")?,
        );
        std::array::from_fn(|k| spread(mins[k], maxs[k]))
    } else {
        [codebook(scales, "scales")?; 3]
    };
    let tables = tables.map(|t| t.map(|v| f32_of(v).exp()));
    let pixels = image(read, file(scales, "scales", 0)?, n, most)?;
    s.scales = room(n)?;
    s.scales.extend(
        pixels
            .iter(n)
            .map(|p| std::array::from_fn(|k| tables[k][usize::from(p[k])])),
    );

    // The base color, and opacity in alpha: a logit in version 1, linear in version 2 and
    // the square root of an opacity up to 1.5625 in version 3.
    let sh0 = section(&meta, "sh0")?;
    let bytes: [f32; 256] = std::array::from_fn(|b| f32::from(u8::try_from(b).unwrap_or(255)));
    let (tables, opacity) = if version == 1 {
        let (mins, maxs) = (numbers::<4>(sh0, "sh0", "mins")?, numbers::<4>(sh0, "sh0", "maxs")?);
        let logits = spread(mins[3], maxs[3]).map(|v| sigmoid(f32_of(v)));
        (std::array::from_fn(|k| spread(mins[k], maxs[k])), logits)
    } else if version == 2 {
        ([codebook(sh0, "sh0")?; 3], bytes.map(|b| b / 255.0))
    } else {
        ([codebook(sh0, "sh0")?; 3], bytes.map(|b| (b / 204.0) * (b / 204.0)))
    };
    let tables: [[f32; 256]; 3] = tables.map(|t| t.map(f32_of));
    let pixels = image(read, file(sh0, "sh0", 0)?, n, most)?;
    (s.dc, s.opacities) = (room(n)?, room(n)?);
    for p in pixels.iter(n) {
        s.dc.push(std::array::from_fn(|k| tables[k][usize::from(p[k])]));
        s.opacities.push(opacity[usize::from(p[3])]);
    }
    drop(pixels);

    // The higher harmonics: a splat's label names a palette entry, a row of pixels in the
    // centroids image, a coefficient each.
    if let Some(shn) = meta.get("shN").filter(|v| v.is_object()) {
        // A palette has no use for many more entries than there are splats to name them, and
        // labels of 16 bits name 65,536 at most: its image has few more rows than those take.
        let declared = shn.get("count").and_then(Value::as_u64).map(usize_of);
        let entries = if version == 3 {
            declared.unwrap_or(spare).min(spare)
        } else {
            1 << 16
        };
        let rows = entries.div_ceil(ROW) + 16;
        let centroids = image(read, file(shn, "shN", 0)?, 0, &|width| rows.saturating_mul(width))?;
        // The image's width tells the bands, as PlayCanvas's engine has it: version 1 and
        // early version 2 files don't state them.
        let width = centroids.width;
        let coeffs = *COEFFS
            .iter()
            .find(|c| ROW * **c == width)
            .with_context(ErrorKind::Corrupt, || {
                format!("SOG shN centroids image is {width} wide, not 192, 512, 960 or 1536")
            })?;
        let (table, entries) = if version == 1 {
            let bound = |key: &str| {
                shn.get(key)
                    .and_then(Value::as_f64)
                    .with_context(ErrorKind::Corrupt, || {
                        format!("SOG meta.json needs a number as the {key} of shN")
                    })
            };
            (spread(bound("mins")?, bound("maxs")?), usize::MAX)
        } else {
            (codebook(shn, "shN")?, declared.unwrap_or(usize::MAX))
        };
        let table = table.map(f32_of);
        let entries = entries.min(ROW * centroids.height);
        let labels = image(read, file(shn, "shN", 1)?, n, most)?;
        let kept = coeffs.min(KEPT);
        if coeffs > kept {
            source.insert("droppedShDegree".into(), json!(4));
        }
        for _ in 0..kept {
            s.sh.push(room(n)?);
        }
        for p in labels.iter(n) {
            // 16 bits, and 24 in version 3.
            let label = usize::from(p[0]) | usize::from(p[1]) << 8 | usize::from(p[2]) << 16;
            let label = if version == 3 { label } else { label & 0xffff };
            // A label past the palette names nothing: no view-dependent color.
            let first = (label < entries).then(|| label / ROW * width + label % ROW * coeffs);
            for (j, col) in s.sh.iter_mut().enumerate() {
                col.push(first.map_or([0.0; 3], |first| {
                    centroids.rgb(first + j).map(|v| table[usize::from(v)])
                }));
            }
        }
    }

    let (custom, skipped) = if version == 3 {
        blind(&meta, read, n, most)?
    } else {
        (Vec::new(), Vec::new())
    };
    if !skipped.is_empty() {
        source.insert("skipped".into(), json!(skipped));
    }
    s.flip([1.0, -1.0, -1.0]);

    if zipped {
        source.insert("zipped".into(), json!(true));
    }
    if let Some(model) = meta.get("model").and_then(Value::as_str) {
        source.insert("model".into(), json!(model));
    }
    if let Some(top) = meta.get("opacity").and_then(|o| o.get("max")).and_then(Value::as_f64) {
        source.insert("opacityMax".into(), json!(top));
    }
    if let Some(extras) = meta.get("extras").filter(|e| !e.is_null()) {
        source.insert("extras".into(), extras.clone());
    }
    let text = |v: &Value| v.as_str().map(str::to_owned);
    // Right, down and forward, as the format has it of 3D gaussian splatting's PLY files.
    let mut info = SourceInfo {
        format: Some("SOG".into()),
        version: Some(version.to_string()),
        tool: meta.get("asset").and_then(|a| a.get("generator")).and_then(text),
        comments: meta
            .get("comments")
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |c| c.iter().filter_map(text).collect()),
        up_axis: Fact::Assumed(Axis::NegY),
        handedness: Fact::Assumed(Handedness::Right),
        ..SourceInfo::default()
    };
    info.converted.up_axis = true;
    Ok(s.document_with(&info, source, None, Map::new(), custom))
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test files are small"
)]
mod tests {
    use std::collections::HashMap;
    use std::f32::consts::FRAC_1_SQRT_2;
    use std::io::Write as _;

    use super::*;
    use crate::assets::Memory;
    use crate::document::Pose;
    use crate::test_rng::Pcg32;

    const KHR: &str = "KHR_gaussian_splatting";

    /// A lossless WebP of `pixels` (red, green, blue, alpha), `width` wide.
    fn webp(width: u32, pixels: &[[u8; 4]]) -> Vec<u8> {
        let height = pixels.len() as u32 / width;
        let mut out = Vec::new();
        image_webp::WebPEncoder::new(&mut out)
            .encode(pixels.as_flattened(), width, height, image_webp::ColorType::Rgba8)
            .unwrap();
        out
    }

    /// The same without alpha, as an image of opaque pixels may be stored.
    fn webp_rgb(width: u32, pixels: &[[u8; 4]]) -> Vec<u8> {
        let rgb: Vec<u8> = pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        let mut out = Vec::new();
        image_webp::WebPEncoder::new(&mut out)
            .encode(&rgb, width, pixels.len() as u32 / width, image_webp::ColorType::Rgb8)
            .unwrap();
        out
    }

    /// A SOG's files by name, `meta.json` among them.
    #[derive(Clone, Default)]
    struct Files(HashMap<String, Vec<u8>>);

    impl Files {
        fn put(&mut self, name: &str, bytes: Vec<u8>) -> &mut Self {
            self.0.insert(name.into(), bytes);
            self
        }

        fn meta(&mut self, meta: &Value) -> &mut Self {
            self.put(META, meta.to_string().into_bytes())
        }

        /// The meta with `change` made to it.
        fn with_meta(&self, change: impl FnOnce(&mut Value)) -> Self {
            let mut meta: Value = serde_json::from_slice(&self.0[META]).unwrap();
            change(&mut meta);
            let mut files = self.clone();
            files.meta(&meta);
            files
        }

        /// Decoded with up to `max` splats; also how many images were read.
        fn decode_max(&self, max: usize) -> (Result<Document>, usize) {
            let reads = std::cell::Cell::new(0);
            let mut read = |name: &str| {
                reads.set(reads.get() + 1);
                self.0
                    .get(name)
                    .cloned()
                    .with_context(ErrorKind::MissingAsset, || format!("SOG image {name} wasn't found"))
            };
            let doc = decode(&self.0[META], &mut read, max, false);
            (doc, reads.get())
        }

        fn decode(&self) -> Result<Document> {
            self.decode_max(1000).0
        }

        fn fails(&self, kind: ErrorKind, text: &str) {
            let e = self.decode().unwrap_err();
            assert_eq!(e.kind(), kind, "{e}");
            assert!(e.to_string().contains(text), "{e}");
        }

        /// The files zipped, each under `folder`, stored or deflated.
        fn zip(&self, folder: &str, deflate: bool) -> Vec<u8> {
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
            let method = if deflate {
                zip::CompressionMethod::Deflated
            } else {
                zip::CompressionMethod::Stored
            };
            let opts = zip::write::SimpleFileOptions::default().compression_method(method);
            let mut names: Vec<&String> = self.0.keys().collect();
            names.sort();
            for name in names {
                zip.start_file(format!("{folder}{name}"), opts).unwrap();
                zip.write_all(&self.0[name]).unwrap();
            }
            zip.finish().unwrap().into_inner()
        }
    }

    fn load_zip(bytes: &[u8]) -> Result<Document> {
        load(Input::memory(bytes, "scene.sog"), &Memory::new(), Keep::ALL)
    }

    fn floats(doc: &Document, name: &str) -> Vec<f32> {
        let a = &doc.json()["meshes"][0]["primitives"][0]["attributes"][name];
        doc.read_floats(usize::try_from(a.as_u64().unwrap_or_else(|| panic!("no {name}"))).unwrap())
            .unwrap()
            .1
    }

    fn khr(doc: &Document, name: &str) -> Vec<f32> {
        floats(doc, &format!("{KHR}:{name}"))
    }

    fn has(doc: &Document, name: &str) -> bool {
        doc.json()["meshes"][0]["primitives"][0]["attributes"]
            .get(name)
            .is_some()
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{got:?} != {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "{got:?} != {want:?}");
        }
    }

    /// A codebook whose entry `k` is `f(k)`.
    fn book(f: impl Fn(f64) -> f64) -> Vec<f64> {
        (0..256).map(|k| f(f64::from(k))).collect()
    }

    /// The low and high bytes of three splats' positions: each axis at its least, at its
    /// greatest, and at a half, a quarter and three quarters of its 16 bits.
    fn means() -> (Vec<u8>, Vec<u8>) {
        let low = [[0, 0, 0, 255], [255, 255, 255, 255], [0, 0, 0, 255], [9, 9, 9, 255]];
        let high = [
            [0, 0, 0, 255],
            [255, 255, 255, 255],
            [0x80, 0x40, 0xC0, 255],
            [9, 9, 9, 255],
        ];
        (webp_rgb(2, &low), webp_rgb(2, &high))
    }

    /// Those positions, from bounds of -1 to 1, 0 to 2 and -2 to 2 in the logarithm, in glTF's
    /// axes: e - 1 = 1.718282 and e^2 - 1 = 6.389056 at the ends.
    const POSITIONS: [f32; 9] = [
        -1.718_281_8,
        0.0,
        6.389_056,
        1.718_281_8,
        -6.389_056,
        -6.389_056,
        0.000_015_259,
        -0.648_733_85,
        -1.718_406_3,
    ];

    /// Three splats of version 2, in images two pixels wide whose fourth pixel is spare, with
    /// one band of harmonics in a palette of two entries.
    fn version_2() -> Files {
        let (low, high) = means();
        let mut files = Files::default();
        files.put("means_l.webp", low).put("means_u.webp", high);
        files.put(
            "quats.webp",
            webp(2, &[[128, 128, 128, 252], [255, 0, 128, 253], [1, 2, 3, 100], [0; 4]]),
        );
        files.put(
            "scales.webp",
            webp_rgb(2, &[[100, 110, 90, 255], [0, 255, 100, 255], [100; 4], [0; 4]]),
        );
        // The last splat is fully transparent: its color is still its own.
        files.put(
            "sh0.webp",
            webp(2, &[[128, 192, 64, 255], [0, 255, 128, 51], [200, 100, 50, 0], [0; 4]]),
        );
        // Entry 0 is all zeros, entry 1 the one read, entry 2 past the count.
        let mut palette = vec![[128, 128, 128, 255]; 192];
        palette[3..9].copy_from_slice(&[
            [255, 0, 128, 255],
            [192, 64, 128, 255],
            [0, 255, 0, 255],
            [255; 4],
            [255; 4],
            [255; 4],
        ]);
        files.put("shN_centroids.webp", webp(192, &palette));
        // Blue isn't part of a version 2 label.
        files.put(
            "shN_labels.webp",
            webp(2, &[[1, 0, 7, 255], [0, 0, 0, 255], [2, 0, 0, 255], [0; 4]]),
        );
        files.meta(&json!({
            "version": 2,
            "asset": { "generator": "splat-transform v0.16.0" },
            "count": 3,
            "means": { "mins": [-1, 0, -2], "maxs": [1, 2, 2], "files": ["means_l.webp", "means_u.webp"] },
            "scales": { "codebook": book(|k| k / 10.0 - 10.0), "files": ["scales.webp"] },
            "quats": { "files": ["quats.webp"] },
            "sh0": { "codebook": book(|k| (k - 128.0) / 64.0), "files": ["sh0.webp"] },
            "shN": {
                "count": 2,
                "bands": 1,
                "codebook": book(|k| (k - 128.0) / 128.0),
                "files": ["shN_centroids.webp", "shN_labels.webp"],
            },
        }));
        files
    }

    #[test]
    fn version_2_decodes_as_its_specification_says() {
        let doc = version_2().decode().unwrap();
        assert_eq!((&doc.facts["splats"], &doc.facts["points"]), (&json!(3), &json!(3)));
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("SOG"));
        assert_eq!(info.version.as_deref(), Some("2"));
        assert_eq!(info.tool.as_deref(), Some("splat-transform v0.16.0"));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::NegY));
        assert!(info.converted.up_axis);
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["shDegree"], 1);
        assert!(source.get("zipped").is_none());
        close(&floats(&doc, "POSITION"), &POSITIONS);
        // A part's byte b is (b / 255 * 2 - 1) / sqrt(2): 0.002773 for 128. The first splat's
        // largest is w; the second's is x, between w = 1/sqrt(2) and y = -1/sqrt(2); the
        // third's alpha names no component, so it's the identity.
        let t = 0.002_773;
        close(
            &khr(&doc, "ROTATION"),
            &[
                t,
                -t,
                -t,
                1.0,
                0.0,
                FRAC_1_SQRT_2,
                -t,
                FRAC_1_SQRT_2,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        );
        let e = std::f32::consts::E;
        close(
            &khr(&doc, "SCALE"),
            &[1.0, e, 1.0 / e, (-10.0f32).exp(), 15.5f32.exp(), 1.0, 1.0, 1.0, 1.0],
        );
        close(&khr(&doc, "OPACITY"), &[1.0, 0.2, 0.0]);
        close(
            &khr(&doc, "SH_DEGREE_0_COEF_0"),
            &[
                0.0,
                1.0,
                -1.0,
                -2.0,
                127.0 / 64.0,
                0.0,
                72.0 / 64.0,
                -28.0 / 64.0,
                -78.0 / 64.0,
            ],
        );
        // The first splat names entry 1; the second entry 0, all zeros; the third an entry
        // past the count, so it has none. Degree 1's coefficients are odd in y, z and x: the
        // first two flip with the axes.
        let top = 127.0 / 128.0;
        close(
            &khr(&doc, "SH_DEGREE_1_COEF_0"),
            &[-top, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        close(&khr(&doc, "SH_DEGREE_1_COEF_1")[..3], &[-0.5, 0.5, 0.0]);
        close(&khr(&doc, "SH_DEGREE_1_COEF_2")[..3], &[-1.0, top, -1.0]);
        assert!(!has(&doc, "KHR_gaussian_splatting:SH_DEGREE_2_COEF_0"));
        doc.to_glb().unwrap();
        assert_eq!(doc.flatten(Pose::Rest).unwrap().unwrap().points(), 3);
    }

    #[test]
    fn version_1_spreads_bytes_between_bounds() {
        let (low, high) = means();
        let mut files = Files::default();
        files.put("means_l.webp", low).put("means_u.webp", high);
        files.put("quats.webp", webp(2, &[[128, 128, 128, 252]; 4]));
        files.put("scales.webp", webp(2, &[[0, 255, 128, 255], [255; 4], [0; 4], [0; 4]]));
        files.put(
            "sh0.webp",
            webp(2, &[[0, 255, 51, 255], [255, 0, 0, 0], [51, 51, 51, 128], [0; 4]]),
        );
        // Two bands: eight coefficients an entry, an image 512 wide.
        let mut palette = vec![[0, 0, 0, 255]; 512];
        palette[8 + 4] = [0, 255, 51, 255];
        files.put("shN_centroids.webp", webp(512, &palette));
        files.put(
            "shN_labels.webp",
            webp(2, &[[1, 0, 0, 255], [0, 0, 0, 255], [64, 0, 0, 255], [0; 4]]),
        );
        files.meta(&json!({
            "means": {
                "shape": [3, 3],
                "mins": [-1, 0, -2],
                "maxs": [1, 2, 2],
                "files": ["means_l.webp", "means_u.webp"],
            },
            "scales": { "mins": [-1, -2, -3], "maxs": [1, 2, 3], "files": ["scales.webp"] },
            "quats": { "encoding": "quaternion_packed", "files": ["quats.webp"] },
            "sh0": { "mins": [-1, -1, -1, -2], "maxs": [1, 1, 1, 2], "files": ["sh0.webp"] },
            "shN": { "mins": -1, "maxs": 1, "files": ["shN_centroids.webp", "shN_labels.webp"] },
        }));
        let doc = files.decode().unwrap();
        assert_eq!(doc.source_info().version.as_deref(), Some("1"));
        close(&floats(&doc, "POSITION"), &POSITIONS);
        close(
            &khr(&doc, "SCALE"),
            &[
                (-1.0f32).exp(),
                2.0f32.exp(),
                (-3.0 + 6.0 * 128.0 / 255.0f32).exp(),
                1.0f32.exp(),
                2.0f32.exp(),
                3.0f32.exp(),
                (-1.0f32).exp(),
                (-2.0f32).exp(),
                (-3.0f32).exp(),
            ],
        );
        close(
            &khr(&doc, "SH_DEGREE_0_COEF_0"),
            &[-1.0, 1.0, -0.6, 1.0, -1.0, -1.0, -0.6, -0.6, -0.6],
        );
        // Opacity is a logit from -2 to 2: its sigmoid is 0.8807971 at 2, and half at
        // 2 * 128 / 255 * 2 - 2 = 0.007843.
        close(&khr(&doc, "OPACITY"), &[0.880_797_1, 0.119_202_92, 0.501_960_8]);
        // The first splat's entry has one coefficient set, the fifth: y z, even in a flip of
        // both. The second's is at the bounds' least; the third's label is past the image.
        close(
            &khr(&doc, "SH_DEGREE_2_COEF_1"),
            &[-1.0, 1.0, -0.6, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
        );
        close(&khr(&doc, "SH_DEGREE_2_COEF_0")[..6], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(doc.json()["asset"]["extras"]["source"]["shDegree"], 2);
    }

    /// Three splats of version 3, without harmonics.
    fn version_3() -> Files {
        let (low, high) = means();
        let mut files = Files::default();
        files.put("means_l.webp", low).put("means_u.webp", high);
        files.put(
            "quats.webp",
            webp(
                2,
                &[[128, 128, 128, 255], [255, 128, 128, 255], [128, 218, 218, 255], [0; 4]],
            ),
        );
        files.put("scales.webp", webp(2, &[[100, 110, 90, 255]; 4]));
        files.put(
            "sh0.webp",
            webp(
                2,
                &[[128, 192, 64, 204], [128, 128, 128, 102], [128, 128, 128, 255], [0; 4]],
            ),
        );
        files.meta(&json!({
            "version": 3,
            "asset": { "generator": "splat-transform v3.11.1" },
            "count": 3,
            "model": "antialiased",
            "means": { "mins": [-1, 0, -2], "maxs": [1, 2, 2], "files": ["means_l.webp", "means_u.webp"] },
            "scales": { "codebook": book(|k| k / 10.0 - 10.0), "files": ["scales.webp"] },
            "quats": { "files": ["quats.webp"] },
            "sh0": { "codebook": book(|k| (k - 128.0) / 64.0), "files": ["sh0.webp"] },
            "opacity": { "max": 1.5625 },
            "extras": { "capturedBy": "rig-04" },
            "comments": ["Postshot v1.1.69", "postshot.color.eotf=sRGB"],
        }));
        files
    }

    #[test]
    fn version_3_decodes_as_its_specification_says() {
        let doc = version_3().decode().unwrap();
        let info = doc.source_info();
        assert_eq!(info.version.as_deref(), Some("3"));
        assert_eq!(info.comments, ["Postshot v1.1.69", "postshot.color.eotf=sRGB"]);
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["model"], "antialiased");
        assert_eq!(source["opacityMax"], 1.5625);
        assert_eq!(source["extras"]["capturedBy"], "rig-04");
        close(&floats(&doc, "POSITION"), &POSITIONS);
        // The ball's center is the identity; its edge along x a half turn about x; and the
        // specification's own example, bytes 218 for y and z, just past the edge, is the half
        // turn about (0, 1, 1): w = cos(1.57425) = -0.0034535.
        close(
            &khr(&doc, "ROTATION"),
            &[
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.707_102_6,
                -0.707_102_6,
                -0.003_453_5,
            ],
        );
        // Opacity is (byte / 204) squared: 1 at 204, a quarter at 102, and 1.5625 at 255,
        // held to the 1 the extension allows.
        close(&khr(&doc, "OPACITY"), &[1.0, 0.25, 1.0]);
        close(&khr(&doc, "SH_DEGREE_0_COEF_0")[..3], &[0.0, 1.0, -1.0]);
    }

    #[test]
    fn version_3_labels_have_24_bits_and_the_fourth_band_is_dropped() {
        // A palette of 65,538 entries: 1,025 rows of 64. Entry 65,537 is the second of the
        // last row; entry 1 the second of the first.
        let mut palette = vec![[128, 128, 128, 255]; 192 * 1025];
        palette[3] = [0, 0, 0, 255];
        palette[1024 * 192 + 3] = [255, 255, 255, 255];
        let mut files = version_3();
        files.put("shN_centroids.webp", webp(192, &palette));
        files.put(
            "shN_labels.webp",
            webp(2, &[[1, 0, 1, 255], [1, 0, 0, 255], [0, 0, 0, 255], [0; 4]]),
        );
        let files = files.with_meta(|m| {
            m["shN"] = json!({
                "count": 65538,
                "bands": 1,
                "codebook": book(|k| (k - 128.0) / 128.0),
                "files": ["shN_centroids.webp", "shN_labels.webp"],
            });
        });
        let doc = files.decode().unwrap();
        let top = 127.0 / 128.0;
        close(
            &khr(&doc, "SH_DEGREE_1_COEF_0"),
            &[-top, -top, -top, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        );
        // Version 2 has 16 bits: the first label is 1 there.
        let doc = files.with_meta(|m| m["version"] = json!(2)).decode().unwrap();
        close(&khr(&doc, "SH_DEGREE_1_COEF_0")[..6], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // Four bands: 24 coefficients an entry, the image 1,536 wide. The first 15 are kept.
        let mut palette = vec![[128, 128, 128, 255]; 1536];
        for (j, p) in palette[24..48].iter_mut().enumerate() {
            *p = [128 + j as u8, 128, 128, 255];
        }
        let mut files = version_3();
        files.put("shN_centroids.webp", webp(1536, &palette));
        files.put("shN_labels.webp", webp(2, &[[1, 0, 0, 255]; 4]));
        let files = files.with_meta(|m| {
            m["shN"] = json!({
                "count": 64,
                "bands": 4,
                "codebook": book(|k| k - 128.0),
                "files": ["shN_centroids.webp", "shN_labels.webp"],
            });
        });
        let doc = files.decode().unwrap();
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(
            (&source["shDegree"], &source["droppedShDegree"]),
            (&json!(3), &json!(4))
        );
        // The last of degree 3, x (x^2 - 3 y^2), keeps its sign when y and z flip.
        close(&khr(&doc, "SH_DEGREE_3_COEF_6")[..3], &[14.0, 0.0, 0.0]);
        close(&khr(&doc, "SH_DEGREE_2_COEF_4")[..3], &[7.0, 0.0, 0.0]);
        assert!(!has(&doc, "KHR_gaussian_splatting:SH_DEGREE_4_COEF_0"));
    }

    #[test]
    fn the_centroids_width_tells_the_bands_where_the_meta_does_not() {
        // An early version 2 file: its shN has neither bands nor count.
        let files = version_2().with_meta(|m| {
            let shn = m["shN"].as_object_mut().unwrap();
            shn.remove("bands");
            shn.remove("count");
        });
        let doc = files.decode().unwrap();
        assert_eq!(doc.json()["asset"]["extras"]["source"]["shDegree"], 1);
        // Without a count, the image's entries are all there to name: the third splat's is
        // entry 2, of the top of the codebook.
        let top = 127.0 / 128.0;
        close(&khr(&doc, "SH_DEGREE_1_COEF_2")[6..], &[top, top, top]);
        // A width no band count has is refused.
        let mut odd = version_2();
        odd.put("shN_centroids.webp", webp(100, &[[0; 4]; 100]));
        odd.fails(ErrorKind::Corrupt, "100 wide");
        // A count past the image's entries stops at them.
        let mut files = version_2().with_meta(|m| m["shN"]["count"] = json!(1_000_000));
        files.put(
            "shN_labels.webp",
            webp(2, &[[64, 0, 0, 255], [255, 255, 0, 255], [1, 0, 0, 255], [0; 4]]),
        );
        let doc = files.decode().unwrap();
        close(
            &khr(&doc, "SH_DEGREE_1_COEF_0"),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -top, 1.0, 0.0],
        );
    }

    #[test]
    fn half_floats_read_from_their_bits() {
        assert_eq!(half(0x3C00), 1.0);
        assert_eq!(half(0xC000), -2.0);
        assert_eq!(half(0x0001), 2f32.powi(-24));
        assert_eq!(half(0x8000), 0.0);
        assert_eq!(half(0x7BFF), 65504.0);
        assert_eq!(half(0x7C00), f32::INFINITY);
        assert_eq!(half(0xFC00), f32::NEG_INFINITY);
        assert!(half(0x7E00).is_nan());
        assert_eq!(half(0x3555), 0.333_251_95);
    }

    #[test]
    fn version_3_blind_data_are_custom_attributes() {
        let mut files = version_3();
        // A 16-bit integer, then two bytes as shares of their ranges; alpha holds data, and
        // is zero for the first splat.
        files.put(
            "blind_0.webp",
            webp(
                2,
                &[[0x34, 0x12, 255, 0], [1, 0, 51, 255], [0xFF, 0xFF, 0, 102], [0; 4]],
            ),
        );
        // A half float, then a type unknown here, which hides the field after it.
        files.put(
            "blind_1.webp",
            webp(2, &[[0x00, 0x3C, 9, 9], [0x00, 0xC0, 9, 9], [0x01, 0x00, 9, 9], [0; 4]]),
        );
        // Three bytes of integer, then two more that don't fit the pixel.
        files.put(
            "blind_2.webp",
            webp(2, &[[1, 2, 3, 4], [0xFF, 0xFF, 0xFF, 0], [0, 0, 0, 0], [0; 4]]),
        );
        files.put(
            "blind_3.webp",
            webp(
                2,
                &[[0, 0, 0x80, 0x3F], [0, 0, 0, 0xC0], [0xFF, 0xFF, 0xFF, 0xFF], [0; 4]],
            ),
        );
        let files = files.with_meta(|m| {
            m["blind"] = json!([
                { "files": ["blind_0.webp"], "fields": [
                    { "name": "segment", "type": "u16" },
                    { "name": "weight", "type": "unorm8" },
                    { "name": "depth", "type": "unorm8", "range": [0.5, 40] },
                ] },
                { "files": ["blind_1.webp"], "fields": [
                    { "name": "heat", "type": "f16" },
                    { "name": "odd", "type": "i64" },
                    { "name": "later", "type": "u8" },
                ] },
                { "files": ["blind_2.webp"], "fields": [
                    { "name": "my id", "type": "u24" },
                    { "name": "wide", "type": "u16" },
                ] },
                { "files": ["blind_3.webp"], "fields": [{ "name": "exact", "type": "f32" }] },
                { "files": ["missing.webp"], "fields": [{ "name": "none", "type": "u128" }] },
            ]);
        });
        let doc = files.decode().unwrap();
        assert_eq!(floats(&doc, "_SEGMENT"), [4660.0, 1.0, 65535.0]);
        close(&floats(&doc, "_WEIGHT"), &[1.0, 0.2, 0.0]);
        close(&floats(&doc, "_DEPTH"), &[0.5, 40.0, 0.5 + 39.5 * 0.4]);
        assert_eq!(floats(&doc, "_HEAT"), [1.0, -2.0, 2f32.powi(-24)]);
        assert_eq!(floats(&doc, "_MY_ID"), [197_121.0, 16_777_215.0, 0.0]);
        // A value that isn't finite is 0: glTF holds none.
        assert_eq!(floats(&doc, "_EXACT"), [1.0, -2.0, 0.0]);
        let skipped = &doc.json()["asset"]["extras"]["source"]["skipped"];
        assert_eq!(skipped, &json!(["odd", "later", "wide", "none"]));
        doc.to_glb().unwrap();
        // Version 2 has no blind data.
        let doc = files.with_meta(|m| m["version"] = json!(2)).decode().unwrap();
        assert!(!has(&doc, "_SEGMENT"));
        // A splat dropped for its position takes its blind values with it.
        let far = files.with_meta(|m| m["means"]["maxs"] = json!([1, 1e300, 2]));
        let doc = far.decode().unwrap();
        assert_eq!(doc.facts["splats"], 1);
        assert_eq!(floats(&doc, "_SEGMENT"), [4660.0]);
    }

    #[test]
    fn counts_and_images_are_checked_before_anything_is_allocated() {
        let good = version_2();
        // A count past the limit is refused before a file is read.
        let (doc, reads) = good.decode_max(2);
        let e = doc.unwrap_err();
        assert_eq!((e.kind(), reads), (ErrorKind::LimitExceeded, 0), "{e}");
        assert!(e.to_string().contains("max_points"), "{e}");
        let huge = good.with_meta(|m| m["count"] = json!(u64::MAX));
        let (doc, reads) = huge.decode_max(usize::MAX);
        assert_eq!((doc.unwrap_err().kind(), reads), (ErrorKind::Corrupt, 1));
        // More splats than the images have pixels.
        good.with_meta(|m| m["count"] = json!(5))
            .fails(ErrorKind::Corrupt, "means_l.webp is 2 by 2, fewer pixels than the 5");
        // Images of different sizes: each must hold the count.
        let mut small = good.clone();
        small.put("means_u.webp", webp(1, &[[0; 4]]));
        small.fails(ErrorKind::Corrupt, "means_u.webp is 1 by 1");
        let mut small = good.clone();
        small.put("shN_labels.webp", webp(1, &[[0; 4], [0; 4]]));
        small.fails(ErrorKind::Corrupt, "shN_labels.webp is 1 by 2");
        // An image of one color is a few bytes whatever its size: far more pixels than the
        // splats take are refused, not decoded.
        let mut big = good.clone();
        big.put("scales.webp", webp(1100, &vec![[7; 4]; 1100 * 1100]));
        big.fails(ErrorKind::LimitExceeded, "scales.webp is 1100 by 1100, far larger");
        let mut big = good.clone();
        big.put("shN_centroids.webp", webp(192, &vec![[7; 4]; 192 * 8000]));
        big.fails(ErrorKind::LimitExceeded, "shN_centroids.webp is 192 by 8000");
        // Larger images within reason are read, their spare pixels ignored.
        let mut roomy = good.clone();
        roomy.put("scales.webp", webp(64, &vec![[100; 4]; 64 * 64]));
        close(&khr(&roomy.decode().unwrap(), "SCALE"), &[1.0; 9]);
    }

    #[test]
    fn a_meta_that_does_not_describe_a_sog_is_refused() {
        let good = version_2();
        let raw = |text: &str, kind: ErrorKind, has: &str| {
            let mut files = good.clone();
            files.put(META, text.as_bytes().to_vec());
            files.fails(kind, has);
        };
        raw("", ErrorKind::Corrupt, "isn't JSON");
        raw("{\"version\": 2, \"means\": {", ErrorKind::Truncated, "is cut short");
        raw(&"[".repeat(100_000), ErrorKind::Corrupt, "doesn't parse");
        raw("[]", ErrorKind::UnknownFormat, "no means");
        raw("{}", ErrorKind::UnknownFormat, "no means");
        raw("{\"means\": 5}", ErrorKind::UnknownFormat, "no means");
        for version in [json!(4), json!(0), json!(-1), json!(2.5), json!("2"), json!(null)] {
            good.with_meta(|m| m["version"] = version.clone())
                .fails(ErrorKind::Unsupported, "isn't supported");
        }
        for count in [json!(-1), json!(1.5), json!("3"), json!(null), json!([3])] {
            good.with_meta(|m| m["count"] = count.clone())
                .fails(ErrorKind::Corrupt, "no count");
        }
        good.with_meta(|m| {
            m.as_object_mut().unwrap().remove("count");
        })
        .fails(ErrorKind::Corrupt, "no count");
        // Version 1 counts in the means' shape.
        good.with_meta(|m| {
            m.as_object_mut().unwrap().remove("version");
        })
        .fails(ErrorKind::Corrupt, "no count");
        let cut = |m: &mut Value, at: &str, to: usize| {
            let list = m.pointer_mut(at).unwrap().as_array_mut().unwrap();
            list.truncate(to);
        };
        good.with_meta(|m| cut(m, "/scales/codebook", 255))
            .fails(ErrorKind::Corrupt, "256 numbers as the codebook of scales");
        good.with_meta(|m| cut(m, "/sh0/codebook", 0))
            .fails(ErrorKind::Corrupt, "codebook of sh0");
        good.with_meta(|m| cut(m, "/shN/codebook", 100))
            .fails(ErrorKind::Corrupt, "codebook of shN");
        good.with_meta(|m| cut(m, "/means/mins", 2))
            .fails(ErrorKind::Corrupt, "3 numbers as the mins of means");
        good.with_meta(|m| cut(m, "/means/files", 1))
            .fails(ErrorKind::Corrupt, "too few files for means");
        good.with_meta(|m| cut(m, "/shN/files", 1))
            .fails(ErrorKind::Corrupt, "too few files for shN");
        good.with_meta(|m| m["quats"] = json!("quats.webp"))
            .fails(ErrorKind::Corrupt, "no quats");
        good.with_meta(|m| m["sh0"]["files"] = json!([7]))
            .fails(ErrorKind::Corrupt, "too few files for sh0");
        // A codebook entry that isn't a number makes no scale: zero.
        let doc = good
            .with_meta(|m| m["scales"]["codebook"][100] = json!(null))
            .decode()
            .unwrap();
        close(
            &khr(&doc, "SCALE")[..3],
            &[0.0, std::f32::consts::E, 1.0 / std::f32::consts::E],
        );
        // Bounds no position survives drop the splats they place.
        let doc = good
            .with_meta(|m| m["means"]["maxs"] = json!([1e300, 2, 2]))
            .decode()
            .unwrap();
        assert_eq!(doc.facts["splats"], 1);
        doc.to_glb().unwrap();
    }

    #[test]
    fn missing_and_broken_images_are_errors() {
        let good = version_2();
        for name in [
            "means_l.webp",
            "means_u.webp",
            "quats.webp",
            "scales.webp",
            "sh0.webp",
            "shN_centroids.webp",
            "shN_labels.webp",
        ] {
            let mut files = good.clone();
            let bytes = files.0.remove(name).unwrap();
            files.fails(ErrorKind::MissingAsset, name);
            files.put(name, b"\x89PNG\r\n\x1a\n not a WebP image".to_vec());
            files.fails(ErrorKind::Corrupt, "isn't a WebP image that decodes");
            for cut in [0, 11, 20, bytes.len() / 2] {
                files.put(name, bytes[..cut].to_vec());
                files.fails(ErrorKind::Corrupt, name);
            }
        }
    }

    #[test]
    fn zipped_and_loose_files_read_the_same() {
        let files = version_2();
        let want = files.decode().unwrap();
        for (folder, deflate) in [("", false), ("", true), ("scene/", false)] {
            let doc = load_zip(&files.zip(folder, deflate)).unwrap();
            assert_eq!(doc.json()["asset"]["extras"]["source"]["zipped"], true);
            assert_eq!(floats(&doc, "POSITION"), floats(&want, "POSITION"));
            assert_eq!(khr(&doc, "SH_DEGREE_1_COEF_2"), khr(&want, "SH_DEGREE_1_COEF_2"));
        }
        // Loose: the model is its meta.json, told by what it holds, the images beside it.
        let mut memory = Memory::new();
        for (name, bytes) in &files.0 {
            memory.insert(&format!("scan/{name}"), bytes.clone());
        }
        let meta = &files.0[META];
        let doc = crate::formats::load_bytes(meta, "scan/meta.json", "json", &memory, Keep::ALL).unwrap();
        assert_eq!(floats(&doc, "POSITION"), floats(&want, "POSITION"));
        assert!(doc.json()["asset"]["extras"]["source"].get("zipped").is_none());
        let e = crate::formats::load_bytes(meta, "elsewhere/meta.json", "json", &Memory::new(), Keep::ALL).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::MissingAsset, "{e}");
        // A zip told by its entries, whatever its name.
        let doc = crate::formats::load_bytes(&files.zip("", false), "scene", "", &Memory::new(), Keep::ALL).unwrap();
        assert_eq!(doc.facts["splats"], 3);
        // Zips that aren't a SOG.
        let mut less = files.clone();
        less.0.remove(META);
        let e = load_zip(&less.zip("", false)).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::UnknownFormat, "{e}");
        let mut less = files.clone();
        less.0.remove("sh0.webp");
        let e = load_zip(&less.zip("", false)).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::MissingAsset, "{e}");
        let e = load_zip(b"PK\x03\x04 and nothing else").unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Corrupt, "{e}");
        let limit = Keep::ALL.with_max_points(2);
        let e = load(Input::memory(&files.zip("", false), "scene.sog"), &Memory::new(), limit).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::LimitExceeded, "{e}");
    }

    #[test]
    fn damaged_files_never_panic() {
        let files = version_2();
        let zip = files.zip("", false);
        for cut in (0..zip.len()).step_by(13) {
            assert!(load_zip(&zip[..cut]).is_err(), "a zip cut at {cut} read");
        }
        let mut rng = Pcg32::new(31);
        for _ in 0..400 {
            let mut bad = zip.clone();
            for _ in 0..3 {
                let i = rng.below(bad.len() as u32) as usize;
                bad[i] ^= 1 << rng.below(8);
            }
            if let Ok(doc) = load_zip(&bad) {
                doc.to_glb().unwrap();
                let _ = doc.flatten(Pose::Rest);
            }
        }
        // The same of each file alone, so the damage is past the zip's own checks.
        let mut names: Vec<&String> = files.0.keys().collect();
        names.sort();
        for name in names {
            for _ in 0..150 {
                let mut bad = files.clone();
                let bytes = bad.0.get_mut(name).unwrap();
                for _ in 0..2 {
                    let i = rng.below(bytes.len() as u32) as usize;
                    bytes[i] = rng.below(256) as u8;
                }
                if let Ok(doc) = bad.decode() {
                    doc.to_glb().unwrap();
                    let _ = doc.flatten(Pose::Rest);
                }
            }
        }
        // And of version 3, with its blind data.
        let mut files = version_3();
        files.put("blind_0.webp", webp(2, &[[1, 2, 3, 4]; 4]));
        let files = files.with_meta(|m| {
            m["blind"] = json!([{ "files": ["blind_0.webp"], "fields": [{ "name": "id", "type": "u32" }] }]);
        });
        files.decode().unwrap();
        for _ in 0..300 {
            let mut bad = files.clone();
            let bytes = bad.0.get_mut(META).unwrap();
            let i = rng.below(bytes.len() as u32) as usize;
            bytes[i] = b"0123456789[]{}\",:.-e "[rng.below(21) as usize];
            if let Ok(doc) = bad.decode() {
                doc.to_glb().unwrap();
            }
        }
    }
}
