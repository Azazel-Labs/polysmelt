use std::collections::BTreeMap;

use serde_json::Value;

use super::super::load_bytes;
use crate::assets::Memory;
use crate::document::{Document, Keep, Pose};
use crate::{ErrorKind, Format, Loader, Result};

fn read_with(bytes: &[u8], max: usize) -> Result<Document> {
    load_bytes(
        bytes,
        "t.meshb",
        "meshb",
        &Memory::new(),
        Keep::ALL.with_max_triangles(max),
    )
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

// Two tetrahedra sharing a face (reference 1), a triangle under the first (reference 3) and an edge of
// reference 0, with a normal at every vertex.
const TEXT: &str = "MeshVersionFormatted 2
Dimension 3
# five vertices
Vertices
5
0 0 0 0
1 0 0 0
0 1 0 2
0 0 1 0
1 1 1 0
Tetrahedra
2
1 2 3 4 1
2 3 4 5 1
Triangles
1
1 3 2 3
Edges
1
1 5 0
Normals
5
0 0 2
0 3 0
1 0 0
0 0 1
0 1 0
NormalAtVertices
5
1 1
2 2
3 3
4 4
5 5
End
";

const POINTS: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];
const NORMALS: [[f64; 3]; 5] = [
    [0.0, 0.0, 2.0],
    [0.0, 3.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
];

/// The same mesh as a binary file of `version`, in either byte order.
fn binary(version: i32, big: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let int = |out: &mut Vec<u8>, v: i32| out.extend(if big { v.to_be_bytes() } else { v.to_le_bytes() });
    let long = |out: &mut Vec<u8>, v: i64| out.extend(if big { v.to_be_bytes() } else { v.to_le_bytes() });
    // A whole number: eight bytes in version 4.
    let whole = |out: &mut Vec<u8>, v: i32| {
        if version == 4 {
            long(out, i64::from(v));
        } else {
            int(out, v);
        }
    };
    let real = |out: &mut Vec<u8>, v: f64| {
        if version == 1 {
            let v = crate::num::f32_of(v);
            out.extend(if big { v.to_be_bytes() } else { v.to_le_bytes() });
        } else {
            out.extend(if big { v.to_be_bytes() } else { v.to_le_bytes() });
        }
    };
    let keyword = |out: &mut Vec<u8>, code: i32, body: Vec<u8>| {
        int(out, code);
        let next = i64::try_from(out.len() + if version >= 3 { 8 } else { 4 } + body.len()).unwrap();
        if version >= 3 {
            long(out, next);
        } else {
            int(out, i32::try_from(next).unwrap());
        }
        out.extend(body);
    };
    int(&mut out, 1);
    int(&mut out, version);
    let mut body = Vec::new();
    int(&mut body, 3);
    keyword(&mut out, 3, body);
    let mut body = Vec::new();
    whole(&mut body, 5);
    for (k, p) in POINTS.iter().enumerate() {
        for &c in p {
            real(&mut body, c);
        }
        whole(&mut body, if k == 2 { 2 } else { 0 });
    }
    keyword(&mut out, 4, body);
    let elements: [(i32, &[&[i32]]); 3] = [
        (8, &[&[1, 2, 3, 4, 1], &[2, 3, 4, 5, 1]]),
        (6, &[&[1, 3, 2, 3]]),
        (5, &[&[1, 5, 0]]),
    ];
    for (code, rows) in elements {
        let mut body = Vec::new();
        whole(&mut body, i32::try_from(rows.len()).unwrap());
        rows.iter().flat_map(|r| r.iter()).for_each(|&v| whole(&mut body, v));
        keyword(&mut out, code, body);
    }
    // A keyword that isn't read: corners.
    let mut body = Vec::new();
    whole(&mut body, 2);
    whole(&mut body, 1);
    whole(&mut body, 5);
    keyword(&mut out, 13, body);
    let mut body = Vec::new();
    whole(&mut body, 5);
    NORMALS.iter().flatten().for_each(|&c| real(&mut body, c));
    keyword(&mut out, 60, body);
    let mut body = Vec::new();
    whole(&mut body, 5);
    (1..=5).for_each(|k| {
        whole(&mut body, k);
        whole(&mut body, k);
    });
    keyword(&mut out, 20, body);
    int(&mut out, 54);
    if version >= 3 {
        long(&mut out, 0);
    } else {
        int(&mut out, 0);
    }
    out
}

#[test]
fn text_reads_its_elements_by_reference() {
    let doc = read(TEXT.as_bytes());
    let meshes = got(&doc);
    assert_eq!(meshes.keys().collect::<Vec<_>>(), ["", "Surface 3", "Volume 1"]);
    // The outside of two tetrahedra is six triangles.
    let volume = &meshes["Volume 1"];
    assert_eq!(volume.tris.len(), 6);
    assert_eq!(volume.positions.len(), 5);
    assert_eq!(volume.attributes["_REF"], [0.0, 0.0, 2.0, 0.0, 0.0]);
    assert_eq!(
        volume.attributes["NORMAL"],
        [
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
        ]
    );
    assert_eq!(meshes["Surface 3"].tris, [[0, 2, 1]]);
    assert_eq!(meshes["Surface 3"].attributes["_REF"], [0.0, 0.0, 2.0]);
    assert_eq!(meshes[""].lines, [[0, 1]]);
    assert_eq!(meshes[""].positions, [[0.0; 3], [1.0; 3]]);
    let named = doc.json()["meshes"].as_array().unwrap().clone();
    let volume = named.iter().find(|m| m["name"] == "Volume 1").unwrap();
    assert_eq!(volume["extras"], serde_json::json!({ "reference": 1, "dimension": 3 }));
    assert_eq!(source(&doc)["format"], "Medit");
    assert_eq!(source(&doc)["encoding"], "ascii");
    assert_eq!(source(&doc)["version"], "2");
}

#[test]
fn binary_reads_as_text_does_in_every_version() {
    let text = got(&read(TEXT.as_bytes()));
    for version in 1..=4 {
        for big in [false, true] {
            let bytes = binary(version, big);
            let doc = read(&bytes);
            assert_eq!(got(&doc), text, "version {version}, big-endian {big}");
            assert_eq!(source(&doc)["encoding"], "binary");
            assert_eq!(source(&doc)["skippedKeywords"], serde_json::json!(["keyword 13"]));
            assert_eq!(source(&doc).get("byteOrder").is_some(), big);
            assert_eq!(Format::detect(&bytes), Some(Format::Medit));
        }
    }
}

#[test]
fn text_is_read_loosely() {
    let plain = got(&read(TEXT.as_bytes()));
    let spaced = TEXT
        .replace("Dimension 3", "Dimension\n3")
        .replace("Tetrahedra", "\n\n  TETRAHEDRA")
        .replace('\n', "\r\n");
    assert_eq!(got(&read(spaced.as_bytes())), plain);
    // Keywords that aren't read are passed over, numbers and all; a reference may be written as a float.
    let more = TEXT
        .replace("Triangles", "Corners\n2\n1\n5\nRequiredVertices\n1\n3\nTriangles")
        .replace("1 3 2 3\n", "1 3 2 3.0\n")
        .replace("1 0 0 0\n", "1.0D+00 0.0d0 0 0\n")
        .replace("End\n", "");
    let doc = read(more.as_bytes());
    assert_eq!(got(&doc), plain);
    assert_eq!(
        source(&doc)["skippedKeywords"],
        serde_json::json!(["Corners", "RequiredVertices"])
    );
}

#[test]
fn normals_that_dont_cover_the_vertices_are_custom() {
    let partial = TEXT.replace(
        "NormalAtVertices\n5\n1 1\n2 2\n3 3\n4 4\n5 5\n",
        "NormalAtVertices\n2\n5 2\n1 1\n",
    );
    let meshes = got(&read(partial.as_bytes()));
    // The mesh's normals are the default ones, as for a file with none; the file's two are kept beside them.
    let volume = &meshes["Volume 1"];
    let mut want = [0.0; 15];
    want[2] = 2.0;
    want[13] = 3.0;
    assert_eq!(volume.attributes["_NORMAL"], want);
    // With no pairs, as many normals as vertices go one to each.
    let direct = TEXT.replace("NormalAtVertices\n5\n1 1\n2 2\n3 3\n4 4\n5 5\n", "");
    assert_eq!(got(&read(direct.as_bytes())), got(&read(TEXT.as_bytes())));
    // Tangents are a custom attribute, zero where a vertex has none.
    let tangents = TEXT.replace("End\n", "Tangents\n1\n0 4 0\nTangentAtVertices\n2\n2 1\n9 1\nEnd\n");
    let meshes = got(&read(tangents.as_bytes()));
    let mut want = [0.0; 15];
    want[4] = 4.0;
    assert_eq!(meshes["Volume 1"].attributes["_TANGENT"], want);
}

#[test]
fn two_dimensions_and_higher_orders() {
    let flat = "MeshVersionFormatted 1\nDimension 2\nVertices 4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n\
                Quadrilaterals 1\n1 2 3 4 0\nEdgesP2 1\n1 2 3 0\nTrianglesP2 1\n1 2 3 4 4 4 7\n";
    let doc = read(flat.as_bytes());
    let meshes = got(&doc);
    assert_eq!(
        meshes[""].positions,
        [[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    );
    assert_eq!(meshes[""].tris.len(), 2);
    // A three-node edge is its two ends.
    assert_eq!(meshes[""].lines, [[0, 1]]);
    assert_eq!(meshes["Surface 7"].tris, [[0, 1, 2]]);
    assert_eq!(source(&doc)["dimension"], 2);
    // An element at a vertex the file doesn't have is dropped.
    let short = TEXT.replace("2 3 4 5 1", "2 3 4 9 1").replace("1 5 0", "0 5 0");
    let doc = read(short.as_bytes());
    assert_eq!(got(&doc)["Volume 1"].tris.len(), 4);
    assert_eq!(source(&doc)["droppedGeometry"], 2);
}

#[test]
fn vertices_alone_are_points() {
    let cloud = "MeshVersionFormatted 2 Dimension 3 Vertices 2 0 0 0 0 1 2 3 0 End";
    let meshes = got(&read(cloud.as_bytes()));
    assert_eq!(meshes[""].points, [0, 1]);
    assert_eq!(meshes[""].positions, [[0.0; 3], [1.0, 2.0, 3.0]]);
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
            "{err}"
        );
    };
    for (keyword, count) in [
        ("Vertices\n5", "Vertices\n4000000000"),
        ("Vertices\n5", "Vertices\n6148914691236517205"),
        ("Tetrahedra\n2", "Tetrahedra\n4000000000"),
        ("Normals\n5", "Normals\n4000000000"),
        ("NormalAtVertices\n5", "NormalAtVertices\n4000000000"),
        ("Vertices\n5", "Vertices\n-5"),
    ] {
        cut(TEXT.replace(keyword, count).as_bytes());
    }
    for version in 1..=4 {
        let whole = binary(version, false);
        let position = if version >= 3 { 8 } else { 4 };
        let width = if version == 4 { 8 } else { 4 };
        // Each keyword's count in turn, followed by position.
        let mut at = 8;
        loop {
            let code = i32::from_le_bytes(whole[at..at + 4].try_into().unwrap());
            if code == 54 {
                break;
            }
            let next = if position == 8 {
                usize::try_from(u64::from_le_bytes(whole[at + 4..at + 12].try_into().unwrap())).unwrap()
            } else {
                u32::from_le_bytes(whole[at + 4..at + 8].try_into().unwrap()) as usize
            };
            if code != 3 && code != 13 {
                let count = at + 4 + position;
                let mut bytes = whole.clone();
                bytes[count..count + width].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0x7F, 0, 0, 0, 0][..width]);
                cut(&bytes);
                if width == 8 {
                    bytes[count..count + 8].fill(0xFF);
                    cut(&bytes);
                }
            }
            at = next;
        }
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
    survives(TEXT.as_bytes());
    for version in 1..=4 {
        survives(&binary(version, version % 2 == 0));
    }
    survives(
        TEXT.replace("End\n", "Tangents\n1\n0 4 0\nTangentAtVertices\n2\n2 1\n9 1\nEnd\n")
            .as_bytes(),
    );
}

#[test]
fn files_cut_short_say_so() {
    let at = TEXT.find("Tetrahedra").unwrap();
    for cut in [&TEXT[..at - 8], &TEXT[..at + 16]] {
        assert_eq!(
            read_with(cut.as_bytes(), usize::MAX).unwrap_err().kind(),
            ErrorKind::Truncated
        );
    }
    for version in 1..=4 {
        let whole = binary(version, false);
        assert_eq!(
            read_with(&whole[..60], usize::MAX).unwrap_err().kind(),
            ErrorKind::Truncated
        );
    }
    assert_eq!(read_with(b"", usize::MAX).unwrap_err().kind(), ErrorKind::Empty);
    assert_eq!(
        read_with(b"solid nothing", usize::MAX).unwrap_err().kind(),
        ErrorKind::UnknownFormat
    );
    let no_dimension = TEXT.replace("Dimension 3\n", "");
    assert_eq!(
        read_with(no_dimension.as_bytes(), usize::MAX).unwrap_err().kind(),
        ErrorKind::Corrupt
    );
    let mut newer = binary(2, false);
    newer[4] = 5;
    assert_eq!(
        read_with(&newer, usize::MAX).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
}

#[test]
fn the_triangle_limit_holds() {
    assert_eq!(
        read_with(TEXT.as_bytes(), 6).unwrap_err().kind(),
        ErrorKind::LimitExceeded
    );
    assert!(read_with(TEXT.as_bytes(), 7).is_ok());
}

/// `.mesh` is OGRE's extension too: a Medit mesh is told by how it starts, and anything else is left to OGRE.
#[test]
fn a_mesh_file_is_medit_only_when_it_starts_as_one() {
    assert_eq!(Format::detect(TEXT.as_bytes()), Some(Format::Medit));
    assert_eq!(
        Format::detect(b"# made by hand\n\n  meshversionformatted 1\n"),
        Some(Format::Medit)
    );
    assert_eq!(Format::detect(&binary(3, true)), Some(Format::Medit));
    assert_eq!(Format::from_extension("meshb"), Some(Format::Medit));
    assert_eq!(Format::from_extension("mesh"), Some(Format::Ogre));
    assert_eq!(Format::detect(b"\x00\x10[MeshSerializer_v1.100]\n"), Some(Format::Ogre));
    assert_eq!(Format::detect(b"<mesh>\n  <submeshes>"), Some(Format::Ogre));
    // Read by name: the text and the binary both open as `.mesh`, and under no name at all.
    for (bytes, name) in [
        (TEXT.as_bytes().to_vec(), "cube.mesh"),
        (binary(2, false), "cube.MESH"),
        (TEXT.as_bytes().to_vec(), "cube.meshb"),
        (binary(4, true), "cube"),
    ] {
        let doc = Loader::new().read(bytes, name).unwrap();
        assert_eq!(source(&doc)["format"], "Medit", "{name}");
    }
    // An OGRE mesh named `.mesh` still goes to OGRE's reader, whatever it makes of these few bytes, and so does
    // a Medit mesh when OGRE is named outright.
    let ogre = b"\x00\x10[MeshSerializer_v1.100]\n".to_vec();
    if let Err(e) = Loader::new().read(ogre, "x.mesh") {
        assert!(!e.to_string().contains("Medit"), "{e}");
    }
    if let Err(e) = Loader::new()
        .format(Format::Ogre)
        .read(TEXT.as_bytes().to_vec(), "x.mesh")
    {
        assert!(!e.to_string().contains("Medit"), "{e}");
    }
    let doc = Loader::new()
        .format(Format::Medit)
        .read(TEXT.as_bytes().to_vec(), "x.mesh")
        .unwrap();
    assert_eq!(source(&doc)["format"], "Medit");
}
