use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::Value;

use super::super::load_bytes;
use crate::assets::Memory;
use crate::document::{Document, Keep, Pose};
use crate::{ErrorKind, Format, Result};

fn read_with(bytes: &[u8], max: usize) -> Result<Document> {
    load_bytes(bytes, "t.msh", "msh", &Memory::new(), Keep::ALL.with_max_triangles(max))
}

fn read(bytes: &[u8]) -> Document {
    read_with(bytes, usize::MAX).unwrap()
}

/// What one mesh of a document draws.
#[derive(Debug, Default, PartialEq)]
struct Got {
    positions: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    lines: Vec<[u32; 2]>,
    points: Vec<u32>,
    attributes: BTreeMap<String, Vec<f32>>,
}

/// Each mesh by name (`""` for the one with none).
fn got(doc: &Document) -> BTreeMap<String, Got> {
    let index = |v: &Value| usize::try_from(v.as_u64().unwrap()).unwrap();
    let mut all = BTreeMap::new();
    for mesh in doc.json()["meshes"].as_array().unwrap() {
        let mut out = Got::default();
        let prims = mesh["primitives"].as_array().unwrap();
        for (k, p) in prims.iter().enumerate() {
            assert_eq!(p["attributes"], prims[0]["attributes"], "a mesh shares its vertices");
            if k == 0 {
                for (name, accessor) in p["attributes"].as_object().unwrap() {
                    let (_, values) = doc.read_floats(index(accessor)).unwrap();
                    if name == "POSITION" {
                        out.positions = values.as_chunks::<3>().0.to_vec();
                    } else {
                        out.attributes.insert(name.clone(), values);
                    }
                }
            }
            let indices = match p.get("indices") {
                Some(a) => doc.read_indices(index(a)).unwrap(),
                None => (0..u32::try_from(out.positions.len()).unwrap()).collect(),
            };
            match p.get("mode").and_then(Value::as_u64).unwrap_or(4) {
                4 => out.tris = indices.as_chunks::<3>().0.to_vec(),
                1 => out.lines = indices.as_chunks::<2>().0.to_vec(),
                _ => out.points = indices,
            }
        }
        // Each vertex as the first that holds what it does, its normal aside: the vertices the default normals
        // add where an edge is hard are copies of a point, and are taken as the point they copy.
        let count = out.positions.len();
        let key = |v: usize| -> Vec<u32> {
            let others = out.attributes.iter().filter(|(name, _)| *name != "NORMAL");
            let held = others.flat_map(|(_, values)| {
                let width = values.len() / count;
                values[v * width..(v + 1) * width].iter().map(|x| x.to_bits())
            });
            out.positions[v].iter().map(|x| x.to_bits()).chain(held).collect()
        };
        let mut firsts: std::collections::HashMap<Vec<u32>, u32> = std::collections::HashMap::new();
        let mut to = Vec::with_capacity(count);
        let mut kept = Vec::new();
        for v in 0..count {
            let next = u32::try_from(kept.len()).unwrap();
            let at = *firsts.entry(key(v)).or_insert(next);
            if at == next {
                kept.push(v);
            }
            to.push(at);
        }
        if kept.len() < count {
            for values in out.attributes.values_mut() {
                let width = values.len() / count;
                *values = kept
                    .iter()
                    .flat_map(|&v| values[v * width..(v + 1) * width].to_vec())
                    .collect();
            }
            out.positions = kept.iter().map(|&v| out.positions[v]).collect();
            out.tris.iter_mut().flatten().for_each(|i| *i = to[*i as usize]);
            out.lines.iter_mut().flatten().for_each(|i| *i = to[*i as usize]);
            out.points.iter_mut().for_each(|i| *i = to[*i as usize]);
        }
        let name = mesh.get("name").and_then(Value::as_str).unwrap_or_default();
        assert!(all.insert(name.to_owned(), out).is_none(), "two meshes named {name:?}");
    }
    all
}

fn source(doc: &Document) -> Value {
    doc.json()["asset"]["extras"]["source"].clone()
}

// Two tetrahedra sharing the face of nodes 2, 3 and 4, in physical volume 7 ("body"), and a triangle under the
// first in physical surface 3, which has no name.
const POINTS: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];
/// Each element's tag, type, physical group, entity and nodes.
const ELEMENTS: [(i32, i32, i32, i32, &[i32]); 3] = [
    (1, 2, 3, 5, &[1, 3, 2]),
    (2, 4, 7, 1, &[1, 2, 3, 4]),
    (3, 4, 7, 1, &[2, 3, 4, 5]),
];

/// Writes whole numbers and doubles as text or as bytes in either order.
struct Out {
    bytes: Vec<u8>,
    binary: bool,
    big: bool,
    /// The bytes of a `size_t`.
    wide: usize,
}

impl Out {
    fn text(&mut self, s: &str) {
        self.bytes.extend_from_slice(s.as_bytes());
    }

    fn int(&mut self, v: i32) {
        if self.binary {
            self.bytes
                .extend(if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
        } else {
            write!(self.bytes_text(), "{v} ").unwrap();
        }
    }

    fn size(&mut self, v: u64) {
        if !self.binary {
            write!(self.bytes_text(), "{v} ").unwrap();
        } else if self.wide == 8 {
            self.bytes
                .extend(if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
        } else {
            let v = u32::try_from(v).unwrap();
            self.bytes
                .extend(if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
        }
    }

    fn real(&mut self, v: f64) {
        if self.binary {
            self.bytes
                .extend(if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
        } else {
            write!(self.bytes_text(), "{v} ").unwrap();
        }
    }

    /// Ends a line of text; binary records have no line ends.
    fn line(&mut self) {
        if !self.binary {
            self.bytes.push(b'\n');
        }
    }

    fn bytes_text(&mut self) -> Text<'_> {
        Text(&mut self.bytes)
    }
}

struct Text<'a>(&'a mut Vec<u8>);

impl std::fmt::Write for Text<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

fn head(version: &str, binary: bool, big: bool, wide: usize) -> Out {
    let mut o = Out {
        bytes: Vec::new(),
        binary,
        big,
        wide,
    };
    o.text(&format!("$MeshFormat\n{version} {} {wide}\n", u8::from(binary)));
    if binary {
        o.int(1);
        o.text("\n");
    }
    o.text("$EndMeshFormat\n$PhysicalNames\n1\n3 7 \"body\"\n$EndPhysicalNames\n");
    o
}

/// The mesh as a version 2.2 file.
fn v2(binary: bool, big: bool) -> Vec<u8> {
    let mut o = head("2.2", binary, big, 8);
    o.text("$Nodes\n5\n");
    for (p, tag) in POINTS.iter().zip(1..) {
        o.int(tag);
        for &c in p {
            o.real(c);
        }
        o.line();
    }
    o.text(if binary {
        "\n$EndNodes\n$Elements\n3\n"
    } else {
        "$EndNodes\n$Elements\n3\n"
    });
    // The two tetrahedra are one run in a binary file.
    let mut runs: Vec<Vec<usize>> = vec![vec![0], vec![1, 2]];
    if !binary {
        runs = vec![vec![0], vec![1], vec![2]];
    }
    for run in runs {
        let ty = ELEMENTS[run[0]].1;
        if binary {
            o.int(ty);
            o.int(i32::try_from(run.len()).unwrap());
            o.int(2);
        }
        for e in run {
            let (tag, _, physical, entity, nodes) = ELEMENTS[e];
            o.int(tag);
            if !binary {
                o.int(ty);
                o.int(2);
            }
            o.int(physical);
            o.int(entity);
            for &n in nodes {
                o.int(n);
            }
            o.line();
        }
    }
    o.text(if binary { "\n$EndElements\n" } else { "$EndElements\n" });
    o.bytes
}

/// The mesh as a version 4.1 file, or 4.0.
fn v4(v41: bool, binary: bool, big: bool, wide: usize) -> Vec<u8> {
    let mut o = head(if v41 { "4.1" } else { "4" }, binary, big, wide);
    o.text("$Entities\n");
    for n in [0, 0, 1, 1] {
        o.size(n);
    }
    o.line();
    for (entity, physical) in [(5, 3), (1, 7)] {
        o.int(entity);
        (0..6).for_each(|_| o.real(0.0));
        o.size(1);
        // The surface's group is given turned around.
        o.int(if entity == 5 { -physical } else { physical });
        o.size(0);
        o.line();
    }
    o.text(if binary {
        "\n$EndEntities\n$Nodes\n"
    } else {
        "$EndEntities\n$Nodes\n"
    });
    o.size(1);
    o.size(5);
    if v41 {
        o.size(1);
        o.size(5);
    }
    o.line();
    if v41 {
        o.int(3);
        o.int(1);
    } else {
        o.int(1);
        o.int(3);
    }
    o.int(0);
    o.size(5);
    o.line();
    if v41 {
        for tag in 1..=5 {
            o.size(tag);
            o.line();
        }
    }
    for (p, tag) in POINTS.iter().zip(1..) {
        if !v41 {
            o.int(tag);
        }
        for &c in p {
            o.real(c);
        }
        o.line();
    }
    o.text(if binary {
        "\n$EndNodes\n$Elements\n"
    } else {
        "$EndNodes\n$Elements\n"
    });
    o.size(2);
    o.size(3);
    if v41 {
        o.size(1);
        o.size(3);
    }
    o.line();
    for run in [&ELEMENTS[..1], &ELEMENTS[1..]] {
        let (_, ty, _, entity, _) = run[0];
        let dim = if ty == 2 { 2 } else { 3 };
        if v41 {
            o.int(dim);
            o.int(entity);
        } else {
            o.int(entity);
            o.int(dim);
        }
        o.int(ty);
        o.size(run.len() as u64);
        o.line();
        for &(tag, _, _, _, nodes) in run {
            if v41 {
                o.size(u64::try_from(tag).unwrap());
                for &n in nodes {
                    o.size(u64::try_from(n).unwrap());
                }
            } else {
                o.int(tag);
                for &n in nodes {
                    o.int(n);
                }
            }
            o.line();
        }
    }
    o.text(if binary { "\n$EndElements\n" } else { "$EndElements\n" });
    o.bytes
}

/// The mesh as a version 1 file, which has no names.
fn v1() -> Vec<u8> {
    let mut s = String::from("$NOD\n5\n");
    for (p, tag) in POINTS.iter().zip(1..) {
        writeln!(s, "{tag} {} {} {}", p[0], p[1], p[2]).unwrap();
    }
    s.push_str("$ENDNOD\n$ELM\n3\n");
    for (tag, ty, physical, entity, nodes) in ELEMENTS {
        write!(s, "{tag} {ty} {physical} {entity} {}", nodes.len()).unwrap();
        for n in nodes {
            write!(s, " {n}").unwrap();
        }
        s.push('\n');
    }
    s.push_str("$ENDELM\n");
    s.into_bytes()
}

/// Every encoding of the mesh, by name.
fn encodings() -> Vec<(String, Vec<u8>)> {
    let mut all = vec![("1.0".to_owned(), v1())];
    for (binary, big) in [(false, false), (true, false), (true, true)] {
        all.push((format!("2.2 binary={binary} big={big}"), v2(binary, big)));
        for v41 in [true, false] {
            for wide in [8, 4] {
                if binary || wide == 8 {
                    all.push((
                        format!("4.{} binary={binary} big={big} size_t={wide}", u8::from(v41)),
                        v4(v41, binary, big, wide),
                    ));
                }
            }
        }
    }
    all
}

#[test]
fn every_version_and_encoding_reads_the_same() {
    let text = got(&read(&v2(false, false)));
    let body = &text["body"];
    // The outside of two tetrahedra is six triangles: the face they share isn't drawn.
    assert_eq!(body.tris.len(), 6);
    assert_eq!(body.positions.len(), 5);
    assert!(body.tris.iter().all(|t| {
        let mut t = *t;
        t.sort_unstable();
        t != [1, 2, 3]
    }));
    assert_eq!(text["Physical Surface 3"].tris, [[0, 2, 1]]);
    assert_eq!(
        text["Physical Surface 3"].positions,
        [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    );
    assert_eq!(text.len(), 2);
    for (name, bytes) in encodings() {
        let mut meshes = got(&read(&bytes));
        if name == "1.0" {
            // No names in version 1.
            let body = meshes.remove("Physical Volume 7").unwrap();
            meshes.insert("body".to_owned(), body);
        }
        assert_eq!(meshes, text, "{name}");
        assert_eq!(Format::detect(&bytes), Some(Format::Gmsh), "{name}");
    }
    let doc = read(&v4(true, true, true, 4));
    assert_eq!(source(&doc)["encoding"], "binary");
    assert_eq!(source(&doc)["byteOrder"], "big");
    assert_eq!(source(&doc)["version"], "4.1");
    assert_eq!(source(&doc)["format"], "Gmsh");
}

#[test]
fn physical_groups_are_meshes_with_their_tags() {
    let doc = read(&v4(true, false, false, 8));
    let meshes = doc.json()["meshes"].as_array().unwrap().clone();
    let named = |name: &str| meshes.iter().find(|m| m["name"] == name).unwrap().clone();
    assert_eq!(
        named("body")["extras"],
        serde_json::json!({ "physical": 7, "dimension": 3 })
    );
    assert_eq!(
        named("Physical Surface 3")["extras"],
        serde_json::json!({ "physical": 3, "dimension": 2 })
    );
    let nodes = doc.json()["nodes"].as_array().unwrap().clone();
    assert_eq!(nodes.len(), 3);
    assert!(nodes.iter().any(|n| n["name"] == "body"));
    // Each group's node is under the one root node.
    assert_eq!(nodes[2]["name"], "RootNode");
    assert_eq!(doc.json()["scenes"][0]["nodes"], serde_json::json!([2]));
    // A version 2.0 name gives no dimension, and elements with no group are a mesh with no name.
    let text = String::from_utf8(v2(false, false)).unwrap();
    let old = text
        .replace("3 7 \"body\"", "3 \"base\"")
        .replace("2 4 2 7 1", "2 4 2 0 1")
        .replace("3 4 2 7 1", "3 4 2 0 1");
    let meshes = got(&read(old.as_bytes()));
    assert_eq!(meshes.keys().collect::<Vec<_>>(), ["", "base"]);
    assert_eq!(meshes[""].tris.len(), 6);
    // With no groups at all there's one mesh, of everything.
    let plain = old.replace("1 2 2 3 5", "1 2 2 0 5");
    let meshes = got(&read(plain.as_bytes()));
    assert_eq!(meshes.keys().collect::<Vec<_>>(), [""]);
    assert_eq!(meshes[""].tris.len(), 7);
}

/// A view's section as text.
fn view(kind: &str, name: &str, step: u32, comps: usize, records: &[&str]) -> String {
    let mut s = format!(
        "${kind}\n1\n\"{name}\"\n1\n0.0\n3\n{step}\n{comps}\n{}\n",
        records.len()
    );
    for r in records {
        s.push_str(r);
        s.push('\n');
    }
    writeln!(s, "$End{kind}").unwrap();
    s
}

#[test]
#[allow(clippy::float_cmp, reason = "the values are the file's, copied and never worked out")]
fn views_become_custom_attributes() {
    let mut text = String::from_utf8(v2(false, false)).unwrap();
    text.push_str(&view(
        "NodeData",
        "temp erature",
        0,
        1,
        &["1 0.5", "2 1.5", "3 2.5", "4 3.5", "5 4.5"],
    ));
    text.push_str(&view("NodeData", "push", 0, 3, &["1 1 2 3", "5 7 8 9"]));
    let meshes = got(&read(text.as_bytes()));
    assert_eq!(meshes["body"].attributes["_TEMP_ERATURE"], [0.5, 1.5, 2.5, 3.5, 4.5]);
    assert_eq!(
        meshes["Physical Surface 3"].attributes["_TEMP_ERATURE"],
        [0.5, 1.5, 2.5]
    );
    // Nodes a view leaves out are zero.
    let push = &meshes["body"].attributes["_PUSH"];
    assert_eq!(push[..3], [1.0, 2.0, 3.0]);
    assert_eq!(push[3..12], [0.0; 9]);
    assert_eq!(push[12..], [7.0, 8.0, 9.0]);

    // Element data: the two tetrahedra differ, so the points they share are a vertex for each.
    let mut text = String::from_utf8(v2(false, false)).unwrap();
    text.push_str(&view("ElementData", "zone", 0, 1, &["1 4", "2 5", "3 6"]));
    let meshes = got(&read(text.as_bytes()));
    let body = &meshes["body"];
    assert_eq!(body.positions.len(), 8);
    for t in &body.tris {
        let zone = body.attributes["_ZONE"][t[0] as usize];
        assert!(t.iter().all(|&v| body.attributes["_ZONE"][v as usize] == zone));
        // The first tetrahedron alone has the origin.
        let origin = t.iter().any(|&v| body.positions[v as usize] == [0.0; 3]);
        let apex = t.iter().any(|&v| body.positions[v as usize] == [1.0; 3]);
        assert!(!origin || zone == 5.0);
        assert!(!apex || zone == 6.0);
    }
    assert_eq!(meshes["Physical Surface 3"].attributes["_ZONE"], [4.0; 3]);

    // Element-node data: each element's own value at each of its nodes.
    let mut text = String::from_utf8(v2(false, false)).unwrap();
    text.push_str(&view(
        "ElementNodeData",
        "flux",
        0,
        1,
        &["1 3 10 30 20", "2 4 1 2 3 4", "3 4 20 30 40 50"],
    ));
    let meshes = got(&read(text.as_bytes()));
    let base = &meshes["Physical Surface 3"];
    for (p, v) in base.positions.iter().zip(&base.attributes["_FLUX"]) {
        // Node 1 at the origin, node 2 along x, node 3 along y.
        assert_eq!(*v, 10.0 + 10.0 * p[0] + 20.0 * p[1]);
    }
    let body = &meshes["body"];
    assert_eq!(body.positions.len(), 8);
    for (p, v) in body.positions.iter().zip(&body.attributes["_FLUX"]) {
        let node = POINTS.iter().position(|q| q.map(crate::num::f32_of) == *p).unwrap();
        #[allow(clippy::cast_precision_loss, reason = "a node's number")]
        let (first, second) = (node as f32 + 1.0, 10.0 * (node as f32 + 1.0));
        assert!(*v == first || *v == second, "{v} at node {node}");
    }
}

#[test]
fn binary_views_read_as_text_does() {
    for big in [false, true] {
        let mut bytes = v2(true, big);
        bytes.extend_from_slice(b"$NodeData\n1\n\"t\"\n1\n0\n3\n0\n1\n5\n");
        for tag in 1..=5i32 {
            let v = f64::from(tag) - 0.5;
            bytes.extend(if big { tag.to_be_bytes() } else { tag.to_le_bytes() });
            bytes.extend(if big { v.to_be_bytes() } else { v.to_le_bytes() });
        }
        bytes.extend_from_slice(b"\n$EndNodeData\n$ElementNodeData\n1\n\"f\"\n1\n0\n3\n0\n1\n1\n");
        let int = |v: i32| if big { v.to_be_bytes() } else { v.to_le_bytes() };
        bytes.extend(int(1));
        bytes.extend(int(3));
        for v in [10.0f64, 30.0, 20.0] {
            bytes.extend(if big { v.to_be_bytes() } else { v.to_le_bytes() });
        }
        bytes.extend_from_slice(b"\n$EndElementNodeData\n");
        let meshes = got(&read(&bytes));
        // The element's own vertices, in the order it's drawn: each is told by where it is.
        let base = &meshes["Physical Surface 3"];
        for (p, v) in base.positions.iter().zip(&base.attributes["_T"]) {
            assert_eq!(*v, 0.5 + p[0] + 2.0 * p[1]);
        }
        for (p, v) in base.positions.iter().zip(&base.attributes["_F"]) {
            assert_eq!(*v, 10.0 + 10.0 * p[0] + 20.0 * p[1]);
        }
    }
}

#[test]
fn later_time_steps_and_thin_views_are_left_out() {
    let mut text = String::from_utf8(v2(false, false)).unwrap();
    text.push_str(&view("NodeData", "t", 0, 1, &["1 1", "2 2", "3 3", "4 4", "5 5"]));
    text.push_str(&view("NodeData", "t", 1, 1, &["1 9", "2 9", "3 9", "4 9", "5 9"]));
    let doc = read(text.as_bytes());
    assert_eq!(got(&doc)["body"].attributes["_T"], [1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(source(&doc)["skippedViews"], 1);
    // A view of one node in a mesh of many isn't laid out over all of them.
    let mut text = String::from("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n40\n");
    for tag in 1..=40 {
        writeln!(text, "{tag} {tag} 0 0").unwrap();
    }
    text.push_str("$EndNodes\n");
    text.push_str(&view("NodeData", "one", 0, 1, &["3 1.5"]));
    let doc = read(text.as_bytes());
    assert_eq!(source(&doc)["skippedArrays"], serde_json::json!(["one"]));
    assert_eq!(got(&doc)[""].points.len(), 40, "nodes alone are points");
}

#[test]
fn tags_need_not_run_from_one() {
    let text = String::from_utf8(v2(false, false)).unwrap();
    // Node 5 becomes 4000000, and the elements' tags fall.
    let sparse = text
        .replace("\n5 1 1 1", "\n4000000 1 1 1")
        .replace("3 4 2 7 1 2 3 4 5", "1 4 2 7 1 2 3 4 4000000")
        .replace("1 2 2 3 5 1 3 2", "9 2 2 3 5 1 3 2");
    assert_ne!(sparse, text);
    assert_eq!(got(&read(sparse.as_bytes())), got(&read(text.as_bytes())));
    // A node no file has drops its element, which the notes count.
    let missing = text.replace("3 4 2 7 1 2 3 4 5", "3 4 2 7 1 2 3 4 77");
    let doc = read(missing.as_bytes());
    assert_eq!(got(&doc)["body"].tris.len(), 4);
    assert_eq!(source(&doc)["droppedGeometry"], 1);
}

#[test]
fn higher_orders_draw_their_corners() {
    // A ten-node tetrahedron, a three-node line and a six-node triangle: the corners come first.
    let mut text = String::from("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n10\n");
    for (p, tag) in POINTS.iter().take(4).zip(1..) {
        writeln!(text, "{tag} {} {} {}", p[0], p[1], p[2]).unwrap();
    }
    for tag in 5..=10 {
        writeln!(text, "{tag} 9 9 9").unwrap();
    }
    text.push_str("$EndNodes\n$Elements\n3\n1 11 2 0 1 1 2 3 4 5 6 7 8 9 10\n2 8 2 0 1 1 2 5\n3 9 2 0 1 1 2 3 5 6 7\n");
    text.push_str("$EndElements\n");
    let all = got(&read(text.as_bytes()));
    let mesh = &all[""];
    assert_eq!(mesh.positions.len(), 4, "only the corners are drawn");
    assert_eq!(mesh.tris.len(), 5);
    assert_eq!(mesh.lines, [[0, 1]]);
    // In a 4.1 file too, where a block gives the type.
    let block = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n\
                 $EndNodes\n$Elements\n2 2 1 2\n3 1 11 1\n1 1 2 3 4 4 4 4 4 4 4\n1 1 8 1\n2 1 2 4\n$EndElements\n";
    let all = got(&read(block.as_bytes()));
    assert_eq!(all[""].tris.len(), 4);
    assert_eq!(all[""].lines, [[0, 1]]);
}

#[test]
fn sections_and_elements_not_known_are_passed_over() {
    let text = String::from_utf8(v2(false, false)).unwrap();
    let more = text
        .replace(
            "$Nodes",
            "$Comments\nanything at all\n$Nodes\n$EndComments\n$Periodic\n0\n$EndPeriodic\n$Nodes",
        )
        .replace("$Elements\n3\n", "$Elements\n5\n8 34 2 0 1 1 2 3 4\n9 84 2 0 1 1\n");
    let doc = read(more.as_bytes());
    assert_eq!(got(&doc), got(&read(text.as_bytes())));
    assert_eq!(source(&doc)["skippedElements"], 2);
    assert_eq!(source(&doc)["elements"], 5);
    // A binary file can't be read past a type whose size isn't known.
    let mut bytes = v2(true, false);
    let at = bytes.windows(12).position(|w| w == b"$Elements\n3\n").unwrap() + 12;
    bytes[at..at + 4].copy_from_slice(&35i32.to_le_bytes());
    assert_eq!(
        read_with(&bytes, usize::MAX).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    // Versions that aren't read say so.
    let v3 = text.replace("2.2 0 8", "3 0 8");
    assert_eq!(
        read_with(v3.as_bytes(), usize::MAX).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    let narrow = text.replace("2.2 0 8", "2.2 0 4");
    assert_eq!(
        read_with(narrow.as_bytes(), usize::MAX).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
}

#[test]
fn parametric_nodes_and_line_ends() {
    let text = String::from_utf8(v4(true, false, false, 8)).unwrap();
    let parametric = text
        .replace("3 1 0 5 \n", "3 1 1 5 \n")
        .replace("1 1 1 \n", "1 1 1 0.5 0.25 0.125\n")
        .replace("0 0 1 \n", "0 0 1 0.5 0.25 0.125\n");
    assert_ne!(parametric, text);
    assert_eq!(got(&read(parametric.as_bytes())), got(&read(text.as_bytes())));
    let crlf = text.replace('\n', "\r\n");
    assert_eq!(got(&read(crlf.as_bytes())), got(&read(text.as_bytes())));
    let crlf = String::from_utf8(v2(false, false)).unwrap().replace('\n', "\r\n");
    assert_eq!(got(&read(crlf.as_bytes())), got(&read(text.as_bytes())));
}

/// Counts no file of this size could hold are errors, found before anything is allocated for them.
#[test]
fn counts_the_file_cant_hold_are_refused() {
    let cut = |bytes: &[u8]| {
        let err = read_with(bytes, usize::MAX).unwrap_err();
        assert!(
            matches!(
                err.kind(),
                ErrorKind::Truncated | ErrorKind::LimitExceeded | ErrorKind::Corrupt
            ),
            "{}: {err}",
            String::from_utf8_lossy(bytes)
        );
    };
    let text = String::from_utf8(v2(false, false)).unwrap();
    cut(text.replace("$Nodes\n5\n", "$Nodes\n4000000000\n").as_bytes());
    cut(text.replace("$Nodes\n5\n", "$Nodes\n6148914691236517205\n").as_bytes());
    cut(text.replace("$Elements\n3\n", "$Elements\n4000000000\n").as_bytes());
    cut(text.replace("$Nodes\n5\n", "$Nodes\n-5\n").as_bytes());
    let data = |records: &str, kind: &str| format!("{text}${kind}\n1\n\"v\"\n1\n0\n3\n0\n1\n{records}\n1 1\n");
    cut(data("4000000000", "NodeData").as_bytes());
    cut(data("4000000000", "ElementNodeData").as_bytes());
    cut(format!("{text}$ElementNodeData\n1\n\"v\"\n1\n0\n3\n0\n1\n1\n1 4000000000 1\n").as_bytes());
    cut(format!("{text}$NodeData\n1\n\"v\"\n1\n0\n3\n0\n4000000000\n1\n1 1\n").as_bytes());
    let four = String::from_utf8(v4(true, false, false, 8)).unwrap();
    cut(four
        .replace("$Entities\n0 0 1 1 ", "$Entities\n0 0 4000000000 1 ")
        .as_bytes());
    cut(four.replace("$Nodes\n1 5 1 5 ", "$Nodes\n1 4000000000 1 5 ").as_bytes());
    cut(four.replace("3 1 0 5 \n", "3 1 0 4000000000 \n").as_bytes());
    cut(four
        .replace("$Elements\n2 3 1 3 ", "$Elements\n2 4000000000 1 3 ")
        .as_bytes());
    cut(four.replace("3 1 4 2 \n", "3 1 4 4000000000 \n").as_bytes());
    // The same counts in binary, as bytes.
    for wide in [4, 8] {
        let whole = v4(true, true, false, wide);
        let find = |needle: &[u8]| whole.windows(needle.len()).position(|w| w == needle).unwrap() + needle.len();
        for (section, skip) in [
            (&b"$Entities\n"[..], 2),
            (b"$Nodes\n", 0),
            (b"$Nodes\n", 1),
            (b"$Elements\n", 1),
        ] {
            let mut bytes = whole.clone();
            let at = find(section) + skip * wide;
            bytes[at..at + wide].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0x7F, 0, 0, 0, 0][..wide]);
            cut(&bytes);
            bytes[at..at + wide].fill(0xFF);
            cut(&bytes);
        }
    }
    let whole = v2(true, false);
    let at = whole.windows(12).position(|w| w == b"$Elements\n3\n").unwrap() + 12;
    for field in 1..3 {
        let mut bytes = whole.clone();
        bytes[at + 4 * field..at + 4 * field + 4].copy_from_slice(&i32::MAX.to_le_bytes());
        cut(&bytes);
        bytes[at + 4 * field..at + 4 * field + 4].copy_from_slice(&(-1i32).to_le_bytes());
        cut(&bytes);
    }
}

/// Reading any prefix of a file, or the file with a byte changed, never panics, and keeps to the triangle
/// limit.
fn survives(sample: &[u8]) {
    let check = |bytes: &[u8]| {
        if let Ok(doc) = read_with(bytes, 64) {
            if let Ok(Some(m)) = doc.flatten(Pose::Rest) {
                assert!(m.triangles() <= 64);
            }
            let _ = doc.to_glb();
        }
    };
    for n in 0..=sample.len() {
        check(&sample[..n]);
    }
    let mut bytes = sample.to_vec();
    for i in 0..bytes.len() {
        for flip in [0x01, 0x20, 0x80, 0xFF] {
            bytes[i] ^= flip;
            check(&bytes);
            bytes[i] ^= flip;
        }
    }
}

#[test]
fn damaged_files_never_panic() {
    for (_, bytes) in encodings() {
        survives(&bytes);
    }
    let mut text = String::from_utf8(v2(false, false)).unwrap();
    text.push_str(&view("NodeData", "t", 0, 3, &["1 1 2 3", "2 2 3 4", "5 7 8 9"]));
    text.push_str(&view("ElementData", "z", 0, 1, &["1 4", "2 5", "3 6"]));
    text.push_str(&view("ElementNodeData", "f", 0, 1, &["1 3 10 30 20", "2 4 1 2 3 4"]));
    survives(text.as_bytes());
    let mut bytes = v2(true, true);
    bytes.extend_from_slice(b"$ElementNodeData\n1\n\"f\"\n1\n0\n3\n0\n1\n1\n");
    bytes.extend(1i32.to_be_bytes());
    bytes.extend(3i32.to_be_bytes());
    bytes.extend([1.0f64, 2.0, 3.0].iter().flat_map(|v| v.to_be_bytes()));
    bytes.extend_from_slice(b"\n$EndElementNodeData\n");
    survives(&bytes);
    survives(
        b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$PartitionedEntities\n2\n1\n4 1\n0 0 0 1\n9 3 1 1 2 0 0 0 1 1 1 1 7 0\n\
          $EndPartitionedEntities\n$Nodes\n1 4 1 4\n3 9 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n\
          $Elements\n1 1 1 1\n3 9 4 1\n1 1 2 3 4\n$EndElements\n",
    );
}

#[test]
fn partition_entities_carry_their_groups() {
    let text = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$PhysicalNames\n1\n3 7 \"body\"\n$EndPhysicalNames\n\
                $PartitionedEntities\n2\n1\n4 1\n0 0 0 1\n9 3 1 1 2 0 0 0 1 1 1 1 7 0\n$EndPartitionedEntities\n\
                $Nodes\n1 4 1 4\n3 9 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n\
                $Elements\n1 1 1 1\n3 9 4 1\n1 1 2 3 4\n$EndElements\n";
    let meshes = got(&read(text.as_bytes()));
    assert_eq!(meshes.keys().collect::<Vec<_>>(), ["body"]);
    assert_eq!(meshes["body"].tris.len(), 4);
}

#[test]
fn files_cut_short_say_so() {
    for (name, whole) in encodings() {
        for marker in [&b"$Nodes"[..], b"$NOD", b"$Elements", b"$ELM"] {
            let Some(at) = whole.windows(marker.len()).position(|w| w == marker) else {
                continue;
            };
            let cut = &whole[..at + marker.len() + 14];
            assert_eq!(
                read_with(cut, usize::MAX).unwrap_err().kind(),
                ErrorKind::Truncated,
                "{name} cut in {}",
                String::from_utf8_lossy(marker)
            );
        }
    }
    assert_eq!(read_with(b"", usize::MAX).unwrap_err().kind(), ErrorKind::Empty);
    assert_eq!(
        read_with(b"solid nothing", usize::MAX).unwrap_err().kind(),
        ErrorKind::UnknownFormat
    );
    assert_eq!(
        read_with(b"$PostFormat\n1.4 0 8\n$EndPostFormat\n", usize::MAX)
            .unwrap_err()
            .kind(),
        ErrorKind::UnknownFormat
    );
}

#[test]
fn the_triangle_limit_holds() {
    let bytes = v2(false, false);
    assert_eq!(read_with(&bytes, 6).unwrap_err().kind(), ErrorKind::LimitExceeded);
    assert!(read_with(&bytes, 7).is_ok());
}

#[test]
fn files_are_told_by_their_first_bytes() {
    assert_eq!(Format::detect(b"$MeshFormat\n4.1 0 8\n"), Some(Format::Gmsh));
    assert_eq!(Format::detect(b"  \n$NOD\n3\n"), Some(Format::Gmsh));
    assert_eq!(Format::detect(b"$PostFormat\n"), None);
    assert_eq!(Format::from_extension("MSH"), Some(Format::Gmsh));
    // With no extension to go by, the bytes say.
    let doc = load_bytes(&v2(true, true), "mesh", "", &Memory::new(), Keep::ALL).unwrap();
    assert_eq!(source(&doc)["format"], "Gmsh");
}

/// Element data in a file whose elements are gone leaves the nodes to draw as points.
#[test]
fn element_data_with_no_elements_is_dropped() {
    let mut text = String::from("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n2\n1 0 0 0\n2 1 0 0\n$EndNodes\n");
    text.push_str(&view("ElementData", "zone", 0, 1, &[]));
    text.push_str(&view("ElementNodeData", "flux", 0, 1, &[]));
    let meshes = got(&read(text.as_bytes()));
    assert_eq!(meshes[""].points, [0, 1]);
    assert!(meshes[""].attributes.is_empty());
}
