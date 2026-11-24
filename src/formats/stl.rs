//! STL, binary (with facet colors) and ASCII.
//!
//! The stored facet normals are ignored: many exporters write zeros, and the vertex order is the
//! one thing the format requires to be right (counter-clockwise seen from outside). Corners at
//! the same point are shared, and the mesh carries no normals, so glTF clients shade it flat from
//! the winding, as STL means. Facets with a coordinate that isn't a finite number are dropped.
//!
//! Each ASCII `solid` is a node and mesh of its own, named as the file names it; a file of one named
//! solid takes its name as its title. A binary file's header text, when it is text, is kept in the
//! asset's `extras.source` as `header`, and is the title when it reads as words rather than an
//! exporter's `KEY=value` settings.
//!
//! STL is for printing, where Z is up (the build plate is XY), so the solids sit under a
//! `RootNode` that turns them from Z-up to glTF's Y-up: the source's up axis is `+Z` by that
//! convention (assumed, not declared), and converted. Units are left as they are and unknown: STL
//! has none, though many files are millimeters.
//!
//! The file is read a block at a time, never whole.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::Input;
use super::stream::{self, Blocks};
use crate::Result;
use crate::document::Document;
use crate::mesh::{GRAY, Material, Mesh, Prim, Welder, single, srgb_to_linear};
use crate::source::{Axis, Fact, SourceInfo};

type Tri = [[f32; 3]; 3];

/// The bytes UTF-8 text may start with.
const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

pub(crate) fn stl(input: Input<'_>) -> Result<Document> {
    let mut blocks = input.blocks()?;
    let size = blocks.size_hint() as u64;
    let head = blocks.need(1024)?;
    let Some(n) = binary_count(head, size) else {
        let solids: Vec<_> = ascii(blocks, input.limit)?
            .into_iter()
            .map(|(name, p)| (name, vec![p]))
            .collect();
        let mut info = info();
        if let [(Some(name), _)] = solids.as_slice() {
            info.title = Some(name.clone());
        }
        let mut source = Map::new();
        source.insert("encoding".into(), json!("ASCII"));
        info.write(&mut source);
        return Ok(document(solids, &single(Prim::default()), source));
    };
    ensure!(
        n <= input.limit,
        "too many triangles to convert ({n}, more than {})",
        input.limit
    );
    let header: [u8; 80] = head[..80].try_into().unwrap_or([0; 80]);
    let mut info = info();
    let mut source = Map::new();
    source.insert("encoding".into(), json!("binary"));
    if let Some(text) = header_text(&header) {
        info.title = header_title(&text);
        source.insert("header".into(), json!(text));
    }
    info.write(&mut source);
    blocks.consume(84);
    let colors = magics(&header);
    let plain = colors.is_none();
    let mut mesh = match binary(blocks, n, colors.unwrap_or(Colors::None))? {
        Some(mesh) => mesh,
        // Two colors flagged the VisCAM way: read it again with colors.
        None if plain => {
            let mut again = input.blocks()?;
            again.need(84)?;
            again.consume(84);
            binary(again, n, Colors::VisCam)?.ok_or_else(|| invalid!("STL changed while being read"))?
        }
        None => bail!("STL changed while being read"),
    };
    let prims = std::mem::take(&mut mesh.prims);
    Ok(document(vec![(None, prims)], &mesh, source))
}

/// What every STL says about itself: nothing, but for the Z up of printing, which the reader turns to Y up.
fn info() -> SourceInfo {
    let mut info = SourceInfo {
        format: Some("STL".into()),
        up_axis: Fact::Assumed(Axis::PosZ),
        ..SourceInfo::default()
    };
    info.converted.up_axis = true;
    info
}

/// A binary header's text as a title: when it holds a letter and no `=`, so it reads as words, not an
/// exporter's settings (`COLOR=... MATERIAL=...`). A leading `solid`, as some tools write, isn't part of it.
fn header_title(text: &str) -> Option<String> {
    let t = match text.split_once(' ') {
        Some((w, rest)) if w.eq_ignore_ascii_case("solid") => rest.trim_start(),
        _ => text,
    };
    (t.chars().any(char::is_alphabetic) && !t.contains('=')).then(|| t.to_owned())
}

/// The document for `meshes` using `mesh`'s materials, with its triangles counted.
fn document(meshes: Vec<(Option<String>, Vec<Prim>)>, mesh: &Mesh, source: Map<String, Value>) -> Document {
    let triangles: usize = meshes.iter().flat_map(|(_, p)| p).map(|p| p.indices.len() / 3).sum();
    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(triangles));
    // STL has no textures.
    let source = Some(Value::Object(source));
    let mut doc = Document::from_named(meshes, &mesh.materials, Vec::new(), source, facts, &[]);
    // Z-up to Y-up: a quarter turn about X.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let roots = doc.json["scenes"][0]["nodes"].take();
    if let Some(nodes) = doc.json["nodes"].as_array_mut() {
        nodes.push(json!({ "name": "RootNode", "children": roots, "rotation": [-h, 0.0, 0.0, h] }));
        let root = nodes.len() - 1;
        doc.json["scenes"][0]["nodes"] = json!([root]);
    }
    doc
}

/// A binary header's text: printable ASCII up to the first NUL, trimmed; `None` when that is
/// empty or the bytes aren't text.
fn header_text(header: &[u8; 80]) -> Option<String> {
    let text = header.split(|&b| b == 0).next().unwrap_or_default().trim_ascii();
    (!text.is_empty() && text.iter().all(|&b| b == b' ' || b.is_ascii_graphic()))
        .then(|| String::from_utf8_lossy(text).into_owned())
}

/// The facet count when the file is binary STL, from its first bytes and its size. The 80-byte
/// header may itself start with "solid", so the size is what decides: exactly 50 bytes per
/// facet, or more (some tools pad the end) when the text doesn't read as ASCII STL.
fn binary_count(head: &[u8], size: u64) -> Option<usize> {
    let n = u32::from_le_bytes(head.get(80..84)?.try_into().ok()?);
    let need = u64::from(n) * 50 + 84;
    if need == size {
        return usize::try_from(n).ok();
    }
    (need < size && !looks_ascii(head))
        .then(|| usize::try_from(n).ok())
        .flatten()
}

/// Whether the file starts like ASCII STL: `solid`, then a `facet` or `endsolid` soon after.
fn looks_ascii(head: &[u8]) -> bool {
    let head = &head[..head.len().min(1024)];
    let head = head.strip_prefix(BOM).unwrap_or(head);
    let starts = head
        .trim_ascii_start()
        .get(..5)
        .is_some_and(|w| w.eq_ignore_ascii_case(b"solid"));
    starts
        && head
            .windows(5)
            .skip(5)
            .any(|w| w.eq_ignore_ascii_case(b"facet") || w.eq_ignore_ascii_case(b"endso"))
}

/// How a binary STL's attribute bytes color its facets.
#[derive(Clone, Copy)]
enum Colors {
    /// Not at all, as far as is known yet.
    None,
    /// Materialise Magics: a `COLOR=rgba` header is the default for facets whose top bit is set,
    /// and the rest have red in the low bits (as three.js reads it).
    Magics([u8; 3]),
    /// `VisCAM` and `SolidView` set the top bit on colored facets, blue in the low bits. There's no
    /// header saying so, and other tools leave junk in those bytes, so only two colors or more
    /// count.
    VisCam,
}

fn magics(header: &[u8; 80]) -> Option<Colors> {
    let at = header
        .windows(6)
        .position(|w| w == b"COLOR=")
        .filter(|at| at + 10 <= 80)?;
    Some(Colors::Magics([header[at + 6], header[at + 7], header[at + 8]]))
}

impl Colors {
    /// A facet's sRGB color, 0-255.
    fn of(self, a: u16) -> Option<[u8; 3]> {
        let five = |v: u16| u8::try_from(u32::from(v & 0x1F) * 255 / 31).unwrap_or(u8::MAX);
        match self {
            Colors::None => None,
            Colors::Magics(default) => Some(if a & 0x8000 == 0 {
                [five(a), five(a >> 5), five(a >> 10)]
            } else {
                default
            }),
            Colors::VisCam => (a & 0x8000 != 0).then(|| [five(a >> 10), five(a >> 5), five(a)]),
        }
    }
}

/// `n` binary facets. `None` when reading without colors finds two `VisCAM` colors.
fn binary(mut blocks: Blocks<'_>, n: usize, colors: Colors) -> Result<Option<Mesh>> {
    let mut plain = Welder::with_capacity(if let Colors::None = colors { n } else { 0 });
    // Colored: a primitive per color, in order of first appearance.
    let mut groups: Vec<(Option<[u8; 3]>, Welder)> = Vec::new();
    let mut group: HashMap<Option<[u8; 3]>, usize> = HashMap::new();
    let mut flagged: Option<u16> = None;
    let mut left = n;
    while left > 0 {
        let data = blocks.need(50)?;
        let k = (data.len() / 50).min(left);
        ensure!(k > 0, "STL changed while being read");
        for f in data[..k * 50].as_chunks::<50>().0 {
            let a = u16::from_le_bytes([f[48], f[49]]);
            let t = facet(f);
            if !finite(&t) {
                continue;
            }
            if let Colors::None = colors {
                if a & 0x8000 != 0 && *flagged.get_or_insert(a) != a {
                    return Ok(None);
                }
                plain.push(t);
            } else {
                let c = colors.of(a);
                let g = *group.entry(c).or_insert_with(|| {
                    groups.push((c, Welder::with_capacity(0)));
                    groups.len() - 1
                });
                groups[g].1.push(t);
            }
        }
        blocks.consume(k * 50);
        left -= k;
    }
    if let Colors::None = colors {
        ensure!(plain.triangles() > 0, "STL has no triangles");
        return Ok(Some(single(plain.finish())));
    }
    ensure!(!groups.is_empty(), "STL has no triangles");
    let mut mesh = Mesh::default();
    for (i, (c, w)) in groups.into_iter().enumerate() {
        let mut p = w.finish();
        p.material = i;
        mesh.prims.push(p);
        let color = c.map_or(GRAY, |c| {
            let [r, g, b] = c.map(|v| srgb_to_linear(f32::from(v) / 255.0));
            [r, g, b, 1.0]
        });
        mesh.materials.push(Material::plain(&format!("color_{i}"), color));
    }
    Ok(Some(mesh))
}

#[inline]
fn facet(f: &[u8; 50]) -> Tri {
    let at = |k: usize| f32::from_le_bytes([f[k], f[k + 1], f[k + 2], f[k + 3]]);
    let v = |k: usize| [at(12 + k * 12), at(16 + k * 12), at(20 + k * 12)];
    [v(0), v(1), v(2)]
}

#[inline]
fn finite(t: &Tri) -> bool {
    t.as_flattened().iter().all(|c| c.is_finite())
}

/// ASCII STL: `solid`, then `facet normal ... outer loop`, three `vertex x y z`, `endloop
/// endfacet`, repeated; `endsolid`. Keywords in any case, any whitespace between words, and
/// any number of solids one after another, each a mesh named by the rest of its `solid` line
/// (up to a `facet` keyword, for files written on one line). Otherwise only the `vertex` lines
/// matter: a facet with more than three is a fan of triangles, and one with fewer, or a vertex
/// line that doesn't read as three numbers, is dropped. A leading byte order mark is skipped.
fn ascii(mut blocks: Blocks<'_>, limit: usize) -> Result<Vec<(Option<String>, Prim)>> {
    if blocks.need(BOM.len())?.starts_with(BOM) {
        blocks.consume(BOM.len());
    }
    let head = blocks.need(5)?;
    ensure!(
        head.trim_ascii_start()
            .get(..5)
            .is_some_and(|w| w.eq_ignore_ascii_case(b"solid")),
        "not an STL file"
    );
    // Sized from the first block's bytes per facet once it's read.
    let size = blocks.size_hint();
    let mut w = Welder::with_capacity((size / 250).min(4096));
    let mut solids: Vec<(Option<String>, Prim)> = Vec::new();
    // The current solid's name, and whether its `solid` line goes on into the next block.
    let (mut name, mut naming): (Vec<u8>, bool) = (Vec::new(), false);
    let mut seen = false;
    let mut total = 0;
    let mut read = 0;
    let mut sized = false;
    let mut tri: Tri = [[0.0; 3]; 3];
    let mut vert = [0.0; 3];
    // The corners read in this facet (three once its first triangle is out), and how many of
    // the vertex's coordinates are still to come (0 when not in a `vertex` line).
    let (mut corner, mut want) = (0, 0);
    // Whether `facet` keywords mark the facets (otherwise every three vertices are one), whether
    // this facet has a bad vertex line, and how many facets were dropped for one.
    let (mut marked, mut bad, mut dropped) = (false, false, 0usize);
    while let Some(text) = blocks.text(stream::words)? {
        if !sized && w.triangles() > 0 {
            sized = true;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a capacity guess from the rate so far; the limit caps it"
            )]
            let est = (size as f64 * w.triangles() as f64 / read as f64 * 1.02) as usize + 16;
            w.expect(est.min(limit));
        }
        read += text.len();
        let mut i = 0;
        let n = text.len();
        if naming {
            naming = !solid_name(text, &mut i, &mut name);
        }
        loop {
            while i < n && stream::space(text[i]) {
                i += 1;
            }
            if i == n {
                break;
            }
            let start = i;
            while i < n && !stream::space(text[i]) {
                i += 1;
            }
            let word = &text[start..i];
            if want > 0 {
                if let Some(x) = stream::f32_from(word) {
                    vert[3 - want] = x;
                    want -= 1;
                    if want > 0 || bad {
                        continue;
                    }
                    if corner < 3 {
                        tri[corner] = vert;
                        corner += 1;
                    } else {
                        // Past the third: a fan from the first.
                        tri[1] = tri[2];
                        tri[2] = vert;
                    }
                    if corner == 3 {
                        if !marked {
                            corner = 0;
                        }
                        if finite(&tri) {
                            w.push(tri);
                            total += 1;
                            ensure!(total <= limit, "too many triangles to convert (more than {limit})");
                        }
                    }
                    continue;
                }
                // Fewer than three numbers: the facet is dropped, and the word read as a keyword.
                want = 0;
                dropped += usize::from(!bad);
                bad = true;
            }
            if word.len() == 6 && word.eq_ignore_ascii_case(b"vertex") {
                want = 3;
            } else if word.len() == 5 && word.eq_ignore_ascii_case(b"facet") {
                // Corners left over from a facet with fewer than three are dropped.
                (corner, bad, marked) = (0, false, true);
            } else if word.len() == 7 && word.eq_ignore_ascii_case(b"endloop") {
                (corner, bad) = (0, false);
            } else if word.len() == 5 && word[0] | 0x20 == b's' && word.eq_ignore_ascii_case(b"solid") {
                (corner, bad) = (0, false);
                if seen {
                    finish_solid(&mut solids, &name, std::mem::replace(&mut w, Welder::with_capacity(0)));
                }
                seen = true;
                name.clear();
                naming = !solid_name(text, &mut i, &mut name);
            }
        }
    }
    dropped += usize::from(want > 0 && !bad);
    finish_solid(&mut solids, &name, w);
    ensure!(
        !solids.is_empty(),
        "STL has no triangles{}",
        if dropped > 0 {
            format!(" ({dropped} dropped for a bad vertex line)")
        } else {
            String::new()
        }
    );
    Ok(solids)
}

/// Reads a `solid` line's name from `text[*i..]` onto `name`, a word at a time: up to the end of
/// the line, or a `facet` or `endsolid` keyword (left for the caller to read), for files written
/// on one line. `false` when `text` ends first and the name may go on in the next block.
fn solid_name(text: &[u8], i: &mut usize, name: &mut Vec<u8>) -> bool {
    let n = text.len();
    loop {
        let gap = *i;
        while *i < n && stream::space(text[*i]) {
            *i += 1;
        }
        if text[gap..*i].iter().any(|&c| c == b'\n' || c == b'\r') {
            return true;
        }
        if *i == n {
            return false;
        }
        let start = *i;
        while *i < n && !stream::space(text[*i]) {
            *i += 1;
        }
        let word = &text[start..*i];
        if word.eq_ignore_ascii_case(b"facet") || word.eq_ignore_ascii_case(b"endsolid") {
            *i = start;
            return true;
        }
        if !name.is_empty() {
            name.push(b' ');
        }
        name.extend_from_slice(word);
    }
}

/// Adds a solid's triangles, if it has any, under its name.
fn finish_solid(solids: &mut Vec<(Option<String>, Prim)>, name: &[u8], w: Welder) {
    if w.triangles() > 0 {
        let name = (!name.is_empty()).then(|| String::from_utf8_lossy(name).into_owned());
        solids.push((name, w.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(bytes: &[u8]) -> Document {
        stl(Input::memory(bytes, "t.stl")).unwrap()
    }

    fn facet(z: f32) -> String {
        format!("facet normal 0 0 1 outer loop vertex 0 0 {z} vertex 1 0 {z} vertex 0 1 {z} endloop endfacet\n")
    }

    #[test]
    fn a_byte_order_mark_before_solid() {
        let text = format!("\u{feff}solid t\n{}endsolid t\n", facet(0.0));
        assert!(looks_ascii(text.as_bytes()));
        assert_eq!(read(text.as_bytes()).facts["triangles"], 1);
    }

    #[test]
    fn facets_with_too_many_or_too_few_vertices_dont_throw_off_the_rest() {
        let quad = "facet normal 0 0 1\nouter loop\nvertex 0 0 5\nvertex 1 0 5\nvertex 1 1 5\nvertex 0 1 5\nendloop\nendfacet\n";
        let two = "facet normal 0 0 1\nouter loop\nvertex 0 0 6\nvertex 1 0 6\nendloop\nendfacet\n";
        let bad = "facet normal 0 0 1\nouter loop\nvertex 0 0 7\nvertex 1 0\nvertex 0 1 7\nendloop\nendfacet\n";
        let text = format!("solid t\n{quad}{two}{bad}{}endsolid t\n", facet(1.0));
        let doc = read(text.as_bytes());
        // The quad as two triangles, then the good facet; the short and bad ones dropped.
        assert_eq!(doc.facts["triangles"], 3);
        let m = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        let p = &m.prims[0].positions;
        assert!(
            p.iter().all(|q| (q[1] - 6.0).abs() > 0.5 && (q[1] - 7.0).abs() > 0.5),
            "{p:?}"
        );
        let only_bad = format!("solid t\n{bad}endsolid t\n");
        let err = stl(Input::memory(only_bad.as_bytes(), "t.stl"))
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("bad vertex"), "{err}");
    }

    #[test]
    fn each_solid_is_a_named_node() {
        let text = format!(
            "solid left part\n{}endsolid left part\nsolid\n{}endsolid\nsolid empty\nendsolid empty\n",
            facet(0.0),
            facet(1.0)
        );
        let doc = read(text.as_bytes());
        let nodes = doc.json["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0]["name"], "left part");
        assert_eq!(doc.json["meshes"][0]["name"], "left part");
        assert!(nodes[1].get("name").is_none());
        assert_eq!(nodes[2]["children"], json!([0, 1]));
        assert_eq!(doc.json["scenes"][0]["nodes"], json!([2]));
        assert_eq!(doc.json["asset"]["extras"]["source"]["encoding"], "ASCII");
        assert_eq!(doc.facts["triangles"], 2);
    }

    #[test]
    fn z_up_becomes_y_up() {
        // A triangle standing 2 high along Z stands 2 high along Y.
        let text = "solid t
facet normal 0 0 1 outer loop vertex 0 0 0 vertex 1 0 0 vertex 0 0 2 endloop endfacet
";
        let doc = read(text.as_bytes());
        let info = doc.source_info();
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
        assert!(info.converted.up_axis);
        assert_eq!(info.meters_per_unit, Fact::Unknown, "STL has no units");
        assert!(info.unit.is_none());
        assert_eq!(info.title.as_deref(), Some("t"));
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        let top = mesh.prims[0].positions.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        assert!((top - 2.0).abs() < 1e-5, "{top}");
    }

    #[test]
    fn a_name_ends_at_the_first_facet_on_one_line() {
        let text = format!("solid one line {}endsolid one line", facet(0.0)).replace('\n', " ");
        assert_eq!(read(text.as_bytes()).json["nodes"][0]["name"], "one line");
    }

    #[test]
    fn binary_header_text_is_kept() {
        let mut bytes = b"made by a tool".to_vec();
        bytes.resize(80, 0);
        bytes.extend(1u32.to_le_bytes());
        for v in [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bytes.extend(v.to_le_bytes());
        }
        bytes.extend([0, 0]);
        let doc = read(&bytes);
        let source = &doc.json["asset"]["extras"]["source"];
        assert_eq!(source["encoding"], "binary");
        assert_eq!(source["header"], "made by a tool");
        assert_eq!(doc.source_info().title.as_deref(), Some("made by a tool"));
        bytes[..80].copy_from_slice(&[b' '; 80]);
        bytes[..22].copy_from_slice(b"COLOR=abcd MATERIAL=ab");
        let doc = read(&bytes);
        assert!(doc.json["asset"]["extras"]["source"]["header"].is_string());
        assert!(doc.source_info().title.is_none(), "settings aren't a title");
        bytes[..80].fill(0);
        let doc = read(&bytes);
        assert!(doc.json["asset"]["extras"]["source"].get("header").is_none());
        assert!(doc.source_info().title.is_none());
        assert_eq!(doc.source_info().format.as_deref(), Some("STL"));
    }
}
