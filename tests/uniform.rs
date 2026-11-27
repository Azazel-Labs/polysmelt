//! What every reader does the same way: the facts it reports and how it spells them, and how it
//! decodes and makes up names.
#![cfg(feature = "default")]

use polysmelt::source::{Axis, Fact, Handedness};
use polysmelt::{Document, ErrorKind};
use serde_json::{Value, json};

fn read(bytes: &[u8], name: &str) -> Document {
    polysmelt::read(bytes.to_vec(), name).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn source(doc: &Document) -> Value {
    doc.json()["asset"]
        .pointer("/extras/source")
        .cloned()
        .unwrap_or(Value::Null)
}

/// Every name in the document: nodes', meshes' and materials'.
fn names(doc: &Document) -> Vec<String> {
    ["nodes", "meshes", "materials"]
        .iter()
        .flat_map(|k| {
            doc.json()
                .get(*k)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|v| v.get("name").and_then(Value::as_str).map(str::to_owned))
        .collect()
}

const TRIANGLE_OBJ: &[u8] = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";

const DAE_HEAD: &str = r#"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">"#;
const DAE_GEOMETRY: &str = r##"<library_geometries><geometry id="g"><mesh><source id="p"><float_array id="pa" count="9">0 0 0 1 0 0 0 1 0</float_array><technique_common><accessor source="#pa" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source><vertices id="v"><input semantic="POSITION" source="#p"/></vertices><triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles></mesh></geometry></library_geometries>"##;

/// A COLLADA triangle under a node named by `name` (raw bytes), after `declaration` and with `asset`.
fn dae(declaration: &str, asset: &str, name: &[u8]) -> Vec<u8> {
    let mut out = format!("{declaration}{DAE_HEAD}{asset}{DAE_GEOMETRY}<library_visual_scenes><visual_scene id=\"s\"><node id=\"n\" name=\"").into_bytes();
    out.extend_from_slice(name);
    out.extend_from_slice(
        b"\"><instance_geometry url=\"#g\"/></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url=\"#s\"/></scene></COLLADA>",
    );
    out
}

#[test]
fn a_name_that_is_not_utf8_is_windows_1252_in_every_reader() {
    // "caf", e acute, a space and a curly quote as Windows-1252 writes them: Latin-1 has no
    // character at 0x93.
    let raw: &[u8] = b"caf\xE9 \x93q\x94";
    let want = "caf\u{e9} \u{201c}q\u{201d}";
    let with = |before: &[u8], after: &[u8]| [before, raw, after].concat();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("a.obj", with(b"o ", b"\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")),
        (
            "a.stl",
            with(
                b"solid ",
                b"\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid\n",
            ),
        ),
        ("a.dae", dae("", "", raw)),
        (
            "a.x3d",
            with(
                b"<X3D version=\"3.3\"><Scene><Transform DEF=\"",
                b"\"><Shape><Box/></Shape></Transform></Scene></X3D>",
            ),
        ),
        (
            "a.ac",
            with(
                b"AC3Db\nMATERIAL \"m\" rgb 1 1 1 amb 0.2 0.2 0.2 emis 0 0 0 spec 0.5 0.5 0.5 shi 20 trans 0\nOBJECT world\nkids 1\nOBJECT poly\nname \"",
                b"\"\nnumvert 3\n0 0 0\n1 0 0\n0 1 0\nnumsurf 1\nSURF 0x00\nmat 0\nrefs 3\n0 0 0\n1 0 0\n2 0 0\nkids 0\n",
            ),
        ),
    ];
    for (name, bytes) in cases {
        let doc = read(&bytes, name);
        let all = names(&doc);
        assert!(all.iter().any(|n| n == want), "{name}: names are {all:?}");
        assert!(
            !all.iter().any(|n| n.contains('\u{FFFD}') || n.is_empty()),
            "{name}: names are {all:?}"
        );
    }
}

#[test]
fn xml_text_is_decoded_as_its_declaration_says() {
    // Declared Latin-1: the two bytes of a UTF-8 e acute are two characters, as the file says.
    let latin = dae(r#"<?xml version="1.0" encoding="ISO-8859-1"?>"#, "", b"caf\xC3\xA9");
    assert!(names(&read(&latin, "a.dae")).contains(&"caf\u{c3}\u{a9}".to_owned()));
    // Undeclared, the same bytes are the UTF-8 they are.
    let utf8 = dae("", "", b"caf\xC3\xA9");
    assert!(names(&read(&utf8, "a.dae")).contains(&"caf\u{e9}".to_owned()));
    // UTF-8 with one stray Windows-1252 byte beside it: each is read as what it is.
    let mixed = dae("", "", b"caf\xC3\xA9 \x99");
    assert!(names(&read(&mixed, "a.dae")).contains(&"caf\u{e9} \u{2122}".to_owned()));
}

#[test]
fn every_reader_writes_the_source_and_counts_the_same_way() {
    // glTF 2.0 says what it is, as every other format's reader does.
    let gltf = br#"{"asset":{"version":"2.0","generator":"by hand","copyright":"mine"},"scenes":[{"nodes":[0]}],"nodes":[{"name":"n"}]}"#;
    let doc = read(gltf, "a.gltf");
    let s = source(&doc);
    assert_eq!(s["format"], "glTF");
    assert_eq!(s["version"], "2.0");
    assert_eq!(s["unit"], "meter");
    assert_eq!(s["upAxis"], "+Y");
    assert_eq!(s["forwardAxis"], "+Z");
    assert_eq!(s["handedness"], "right");
    assert_eq!(s["tool"], "by hand");
    assert!(s.get("assumed").is_none(), "glTF fixes them all: {s}");
    // A source the file carries already (a model converted from another format, read back) stays.
    let converted = br#"{"asset":{"version":"2.0","extras":{"source":{"format":"FBX","upAxis":"+Z"}}},"scenes":[{"nodes":[0]}],"nodes":[{"name":"n"}]}"#;
    let s = source(&read(converted, "a.gltf"));
    assert_eq!((&s["format"], &s["upAxis"]), (&json!("FBX"), &json!("+Z")));
    assert!(s.get("forwardAxis").is_none());

    // `triangles` is always there, and `points` and `lines` when there are any, whatever the reader.
    let obj = read(TRIANGLE_OBJ, "a.obj");
    assert_eq!(obj.facts["triangles"], 1);
    assert!(obj.facts.get("points").is_none() && obj.facts.get("lines").is_none());
    let bvh = read(
        b"HIERARCHY\nROOT Hips\n{\nOFFSET 0 0 0\nCHANNELS 3 Xrotation Yrotation Zrotation\nEnd Site\n{\nOFFSET 0 1 0\n}\n}\nMOTION\nFrames: 1\nFrame Time: 0.04\n0 0 0\n",
        "a.bvh",
    );
    assert_eq!(bvh.facts["triangles"], 0, "a skeleton draws nothing, and says so");
    let cloud = read(b"0 0 0\n1 0 0\n0 1 0\n", "a.xyz");
    assert_eq!(
        (&cloud.facts["triangles"], &cloud.facts["points"]),
        (&json!(0), &json!(3))
    );
    let lines = read(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nl 1 2 3\n", "a.obj");
    assert_eq!(
        (&lines.facts["triangles"], &lines.facts["lines"]),
        (&json!(0), &json!(2))
    );
}

#[test]
fn units_are_named_one_way_and_the_files_word_kept_beside() {
    let unit = |name: &str, meter: &str| format!("<asset><unit name=\"{name}\" meter=\"{meter}\"/></asset>");
    // The usual name by the unit's length, whatever the file calls it.
    let s = source(&read(&dae("", &unit("cm", "0.01"), b"n"), "a.dae"));
    assert_eq!((&s["unit"], &s["unitAsWritten"]), (&json!("centimeter"), &json!("cm")));
    let s = source(&read(&dae("", &unit("centimeter", "0.01"), b"n"), "a.dae"));
    assert_eq!(s["unit"], "centimeter");
    assert!(s.get("unitAsWritten").is_none(), "{s}");
    // A length no common unit has: no `unit`, only the file's word.
    let s = source(&read(&dae("", &unit("hand", "0.1016"), b"n"), "a.dae"));
    assert!(s.get("unit").is_none(), "{s}");
    assert_eq!(s["unitAsWritten"], "hand");

    // AMF: a unit the format doesn't have is read as millimeters, assumed, and `unit` says what was
    // assumed, not the word that wasn't understood.
    let amf = |unit: &str| {
        format!(
            r#"<?xml version="1.0"?><amf unit="{unit}"><object id="1"><mesh><vertices><vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex></vertices><volume><triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle></volume></mesh></object></amf>"#
        )
    };
    let doc = read(amf("furlong").as_bytes(), "a.amf");
    let s = source(&doc);
    assert_eq!(
        (&s["unit"], &s["unitAsWritten"]),
        (&json!("millimeter"), &json!("furlong"))
    );
    assert_eq!(doc.source_info().meters_per_unit, Fact::Assumed(0.001));
    let s = source(&read(amf("inch").as_bytes(), "a.amf"));
    assert_eq!(s["unit"], "inch");
    assert!(s.get("unitAsWritten").is_none(), "{s}");
}

#[test]
fn what_a_format_fixes_is_declared_and_what_is_guessed_is_assumed() {
    // 3D printing's Z up is the format's, in AMF as in 3MF.
    let amf = br#"<?xml version="1.0"?><amf unit="millimeter"><object id="1"><mesh><vertices><vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex></vertices><volume><triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle></volume></mesh></object></amf>"#;
    assert_eq!(read(amf, "a.amf").source_info().up_axis, Fact::Declared(Axis::PosZ));
    // Direct3D's handedness is the .x format's; its up axis is only the usual one.
    let x = b"xof 0303txt 0032\nMesh m {\n3;\n0;0;0;,\n1;0;0;,\n0;1;0;;\n1;\n3;0,1,2;;\n}\n";
    let info = read(x, "a.x").source_info();
    assert_eq!(info.handedness, Fact::Declared(Handedness::Left));
    assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
    // X3D's standard has +Y up.
    let x3d = br#"<X3D version="3.3"><Scene><Shape><Box/></Shape></Scene></X3D>"#;
    assert_eq!(read(x3d, "a.x3d").source_info().up_axis, Fact::Declared(Axis::PosY));
    // OBJ says nothing: +Y is a guess.
    assert_eq!(
        read(TRIANGLE_OBJ, "a.obj").source_info().up_axis,
        Fact::Assumed(Axis::PosY)
    );
    // COLLADA can say and here doesn't: the default it falls back to is assumed.
    assert_eq!(
        read(&dae("", "", b"n"), "a.dae").source_info().up_axis,
        Fact::Assumed(Axis::PosY)
    );
}

#[test]
fn an_assumed_frame_rate_needs_animation() {
    // A static USD stage has no frame rate to assume.
    let usda = b"#usda 1.0\ndef Mesh \"m\"\n{\n    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n}\n";
    let doc = read(usda, "a.usda");
    assert_eq!(doc.source_info().frame_rate, Fact::Unknown, "{}", source(&doc));
    // One it declares stays, animated or not.
    let declared = b"#usda 1.0\n(\n    framesPerSecond = 30\n)\ndef Mesh \"m\"\n{\n    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n}\n";
    assert_eq!(read(declared, "a.usda").source_info().frame_rate, Fact::Declared(30.0));
}

#[test]
fn encodings_are_spelled_one_way() {
    let stl = b"solid a\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid a\n";
    let off = b"OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
    let ply = b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
    let vtk = b"# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\nPOINTS 3 float\n0 0 0 1 0 0 0 1 0\nPOLYGONS 1 4\n3 0 1 2\n";
    let dxf = b"0\nSECTION\n2\nENTITIES\n0\n3DFACE\n8\n0\n10\n0\n20\n0\n30\n0\n11\n1\n21\n0\n31\n0\n12\n0\n22\n1\n32\n0\n13\n0\n23\n1\n33\n0\n0\nENDSEC\n0\nEOF\n";
    let x = b"xof 0303txt 0032\nMesh m {\n3;\n0;0;0;,\n1;0;0;,\n0;1;0;;\n1;\n3;0,1,2;;\n}\n";
    for (name, bytes) in [
        ("a.stl", &stl[..]),
        ("a.off", &off[..]),
        ("a.ply", &ply[..]),
        ("a.vtk", &vtk[..]),
        ("a.dxf", &dxf[..]),
        ("a.x", &x[..]),
    ] {
        assert_eq!(source(&read(bytes, name))["encoding"], "ascii", "{name}");
    }
    // A binary PLY says `binary`, and its byte order beside it.
    let mut ply = b"ply\nformat binary_big_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n".to_vec();
    for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        ply.extend(v.to_be_bytes());
    }
    ply.push(3);
    for i in [0i32, 1, 2] {
        ply.extend(i.to_be_bytes());
    }
    let s = source(&read(&ply, "a.ply"));
    assert_eq!((&s["encoding"], &s["byteOrder"]), (&json!("binary"), &json!("big")));
}

#[test]
fn a_thing_with_no_name_has_no_name_key_and_made_up_names_are_kind_and_number() {
    // An X3D material has no name: no `name` key, not an empty one.
    let x3d = br#"<X3D version="3.3"><Scene><Shape><Appearance><Material diffuseColor="1 0 0"/></Appearance><Box/></Shape></Scene></X3D>"#;
    let doc = read(x3d, "a.x3d");
    let materials = doc.json()["materials"].as_array().unwrap();
    assert_eq!(materials.len(), 1);
    for list in ["nodes", "meshes", "materials"] {
        for v in doc.json().get(list).and_then(Value::as_array).into_iter().flatten() {
            assert_ne!(v.get("name"), Some(&json!("")), "{list}: {v}");
        }
    }
    // An AMF object the file numbers and doesn't name: its kind and the file's id.
    let amf = br#"<?xml version="1.0"?><amf unit="millimeter"><object id="7"><mesh><vertices><vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex></vertices><volume><triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle></volume></mesh></object></amf>"#;
    assert!(names(&read(amf, "a.amf")).contains(&"object7".to_owned()));
    // An XGL mesh and object, likewise.
    let xgl = br#"<WORLD><MESH ID="4"><P ID="0">0,0,0</P><P ID="1">1,0,0</P><P ID="2">0,1,0</P><F><FV1><PREF>0</PREF></FV1><FV2><PREF>1</PREF></FV2><FV3><PREF>2</PREF></FV3></F></MESH><OBJECT><MESHREF>4</MESHREF></OBJECT></WORLD>"#;
    let all = names(&read(xgl, "a.xgl"));
    assert!(
        all.contains(&"mesh4".to_owned()) && all.contains(&"object1".to_owned()),
        "{all:?}"
    );
}

#[test]
fn versions_the_readers_do_not_read_are_unsupported() {
    let kind = |bytes: &[u8], name: &str| polysmelt::read(bytes.to_vec(), name).map(|_| ()).map_err(|e| e.kind());
    let mut md2 = b"IDP2".to_vec();
    md2.extend(9i32.to_le_bytes());
    md2.extend([0; 60]);
    assert_eq!(kind(&md2, "a.md2"), Err(ErrorKind::Unsupported));
    let mut md3 = b"IDP3".to_vec();
    md3.extend(14i32.to_le_bytes());
    md3.extend([0; 100]);
    assert_eq!(kind(&md3, "a.md3"), Err(ErrorKind::Unsupported));
    assert_eq!(
        kind(
            b"MD5Version 11\ncommandline \"\"\nnumJoints 0\nnumMeshes 0\n",
            "a.md5mesh"
        ),
        Err(ErrorKind::Unsupported)
    );
}

#[test]
fn a_library_of_meshes_no_node_draws_is_not_empty() {
    // A glTF whose meshes no node instances: nothing is drawn, and there is still something to convert.
    let library = br#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],"buffers":[{"byteLength":36,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA"}],"bufferViews":[{"buffer":0,"byteLength":36}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}]}"#;
    let doc = read(library, "a.gltf");
    assert_eq!(doc.facts["triangles"], 0);
    assert_eq!(doc.json()["meshes"].as_array().map(Vec::len), Some(1));
    // With nothing drawable at all, it is empty.
    let nothing = br#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[]}],"materials":[{"name":"m"}]}"#;
    let e = polysmelt::read(nothing.to_vec(), "a.gltf").map(|_| ()).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Empty);
}
