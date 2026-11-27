//! What every reader does the same way: the facts it reports and how it spells them, how it
//! decodes and makes up names, and how it brings a file's axes and units to glTF's.
#![cfg(feature = "default")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::format_push_string,
    clippy::format_collect,
    reason = "the files written here are small: their counts and indices fit any of these types"
)]

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

/// One box, 100 by 200 by 300, written in each format: its corners, and its faces as quads wound
/// counter-clockwise seen from outside. Corner `i` has x in bit 0, y in bit 1, z in bit 2.
mod boxed {
    pub const SIZE: [f64; 3] = [100.0, 200.0, 300.0];
    pub const QUADS: [[usize; 4]; 6] = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];

    /// The corners, each coordinate times `scale`, as `f` arranges them.
    pub fn corners(scale: f64, f: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<[f64; 3]> {
        (0..8)
            .map(|i| f([0, 1, 2].map(|k| if i >> k & 1 == 1 { SIZE[k] * scale } else { 0.0 })))
            .collect()
    }

    /// Numbers separated by `sep`.
    pub fn list<T: ToString>(v: impl IntoIterator<Item = T>, sep: &str) -> String {
        v.into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join(sep)
    }

    pub fn flat(corners: &[[f64; 3]]) -> Vec<f64> {
        corners.iter().flatten().copied().collect()
    }

    pub fn triangles() -> Vec<[usize; 3]> {
        QUADS
            .iter()
            .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect()
    }

    /// The box with Z up, in centimeters, as FBX ASCII.
    pub fn fbx() -> Vec<u8> {
        let v = list(flat(&corners(1.0, |p| p)), ",");
        let polygons = list(
            QUADS
                .iter()
                .flat_map(|q| [q[0] as i64, q[1] as i64, q[2] as i64, -(q[3] as i64) - 1]),
            ",",
        );
        let setting = |name: &str, value: i32| format!("\t\tP: \"{name}\", \"int\", \"Integer\", \"\",{value}\n");
        format!(
            "; FBX 7.4.0 project file\nFBXHeaderExtension:  {{\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n}}\n\
             GlobalSettings:  {{\n\tVersion: 1000\n\tProperties70:  {{\n{}{}{}{}{}{}\
             \t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1\n\t}}\n}}\n\
             Objects:  {{\n\tGeometry: 10, \"Geometry::Cube\", \"Mesh\" {{\n\t\tVertices: *24 {{\n\t\t\ta: {v}\n\t\t}}\n\
             \t\tPolygonVertexIndex: *24 {{\n\t\t\ta: {polygons}\n\t\t}}\n\t}}\n\
             \tModel: 20, \"Model::Cube\", \"Mesh\" {{\n\t}}\n}}\n\
             Connections:  {{\n\tC: \"OO\",10,20\n\tC: \"OO\",20,0\n}}\n",
            setting("UpAxis", 2),
            setting("UpAxisSign", 1),
            setting("FrontAxis", 1),
            setting("FrontAxisSign", -1),
            setting("CoordAxis", 0),
            setting("CoordAxisSign", 1),
        )
        .into_bytes()
    }

    /// The box with Z up, in centimeters, as COLLADA.
    pub fn dae() -> Vec<u8> {
        let v = list(flat(&corners(1.0, |p| p)), " ");
        let p = list(triangles().into_iter().flatten(), " ");
        format!(
            r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><asset><unit name="centimeter" meter="0.01"/><up_axis>Z_UP</up_axis></asset><library_geometries><geometry id="g"><mesh><source id="p"><float_array id="pa" count="24">{v}</float_array><technique_common><accessor source="#pa" count="8" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source><vertices id="v"><input semantic="POSITION" source="#p"/></vertices><triangles count="12"><input semantic="VERTEX" source="#v" offset="0"/><p>{p}</p></triangles></mesh></geometry></library_geometries><library_visual_scenes><visual_scene id="s"><node id="n" name="Cube"><instance_geometry url="#g"/></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#s"/></scene></COLLADA>"##
        )
        .into_bytes()
    }

    /// The box with Z up, as 3DS always is, with a master scale of a hundredth.
    pub fn tds() -> Vec<u8> {
        fn chunk(id: u16, body: &[u8]) -> Vec<u8> {
            let mut out = id.to_le_bytes().to_vec();
            out.extend((body.len() as u32 + 6).to_le_bytes());
            out.extend(body);
            out
        }
        let mut vertices = 8u16.to_le_bytes().to_vec();
        for c in flat(&corners(1.0, |p| p)) {
            vertices.extend((c as f32).to_le_bytes());
        }
        let mut faces = 12u16.to_le_bytes().to_vec();
        for t in triangles() {
            for i in t {
                faces.extend((i as u16).to_le_bytes());
            }
            faces.extend(7u16.to_le_bytes());
        }
        let mesh = [chunk(0x4110, &vertices), chunk(0x4120, &faces)].concat();
        let object = [b"Cube\0".to_vec(), chunk(0x4100, &mesh)].concat();
        // 3DS names no unit: its master scale multiplies every coordinate.
        let master = 0.01f32.to_le_bytes();
        let editor = [chunk(0x0100, &master), chunk(0x4000, &object)].concat();
        chunk(
            0x4D4D,
            &[chunk(0x0002, &3u32.to_le_bytes()), chunk(0x3D3D, &editor)].concat(),
        )
    }

    /// The box as OBJ, which names no up axis and no unit.
    pub fn obj() -> Vec<u8> {
        let mut out = String::from("o Cube\n");
        for c in corners(1.0, |p| p) {
            out += &format!("v {} {} {}\n", c[0], c[1], c[2]);
        }
        for q in QUADS {
            out += &format!("f {}\n", list(q.map(|i| i + 1), " "));
        }
        out.into_bytes()
    }

    /// The box with Z up, as 3MF always is, in its millimeters.
    pub fn threemf() -> Vec<u8> {
        use std::io::Write as _;
        let vertices: String = corners(10.0, |p| p)
            .iter()
            .map(|c| format!(r#"<vertex x="{}" y="{}" z="{}"/>"#, c[0], c[1], c[2]))
            .collect();
        let triangles: String = triangles()
            .iter()
            .map(|t| format!(r#"<triangle v1="{}" v2="{}" v3="{}"/>"#, t[0], t[1], t[2]))
            .collect();
        let model = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources><object id="1" type="model" name="Cube"><mesh><vertices>{vertices}</vertices><triangles>{triangles}</triangles></mesh></object></resources><build><item objectid="1"/></build></model>"#
        );
        let types = r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>"#;
        let rels = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>"#;
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, body) in [
            ("[Content_Types].xml", types),
            ("_rels/.rels", rels),
            ("3D/3dmodel.model", model.as_str()),
        ] {
            let stored = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
            zip.start_file(name, stored).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    /// The box with Z up, in centimeters, as a USD text layer.
    pub fn usda() -> Vec<u8> {
        let points = list(
            corners(1.0, |p| p)
                .iter()
                .map(|c| format!("({}, {}, {})", c[0], c[1], c[2])),
            ", ",
        );
        let indices = list(QUADS.into_iter().flatten(), ", ");
        format!(
            "#usda 1.0\n(\n    metersPerUnit = 0.01\n    upAxis = \"Z\"\n)\ndef Mesh \"Cube\"\n{{\n    \
             int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n    int[] faceVertexIndices = [{indices}]\n    \
             point3f[] points = [{points}]\n}}\n"
        )
        .into_bytes()
    }

    /// The box in centimeters as X3D, whose standard has Y up: the same box, laid out Y up.
    pub fn x3d() -> Vec<u8> {
        let points = list(flat(&corners(1.0, |[x, y, z]| [x, z, -y])), " ");
        let indices = list(
            QUADS
                .iter()
                .flat_map(|q| [q[0] as i64, q[1] as i64, q[2] as i64, q[3] as i64, -1]),
            " ",
        );
        format!(
            r#"<X3D version="3.3"><head><unit category="length" name="centimeters" conversionFactor="0.01"/></head><Scene><Transform DEF="Cube"><Shape><IndexedFaceSet coordIndex="{indices}"><Coordinate point="{points}"/></IndexedFaceSet></Shape></Transform></Scene></X3D>"#
        )
        .into_bytes()
    }
}

/// What a document draws, in the world: its bounds, its triangle count, and whether every triangle faces away
/// from the middle of them.
fn drawn(doc: &Document) -> ([f32; 3], [f32; 3], usize, bool) {
    let mesh = doc.flatten(polysmelt::document::Pose::Rest).unwrap().unwrap();
    let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in mesh.prims.iter().flat_map(|p| &p.positions) {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    let middle = [0, 1, 2].map(|k| f32::midpoint(min[k], max[k]));
    let mut outward = true;
    for prim in &mesh.prims {
        for t in prim.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|k| prim.positions[t[k] as usize]);
            let (u, v) = ([0, 1, 2].map(|k| b[k] - a[k]), [0, 1, 2].map(|k| c[k] - a[k]));
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let out: f32 = (0..3).map(|k| n[k] * (a[k] - middle[k])).sum();
            outward &= out > 0.0;
            // A normal the reader wrote faces the way its triangle does.
            if let Some(vn) = prim.normals.get(t[0] as usize) {
                outward &= (0..3).map(|k| n[k] * vn[k]).sum::<f32>() > 0.0;
            }
        }
    }
    (min, max, mesh.triangles(), outward)
}

fn near(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() < 1e-4)
}

/// The one top-level node of a document: its index, and the names of its children.
fn root_of(doc: &Document) -> (Value, Vec<Value>) {
    let j = doc.json();
    let tops = j["scenes"][0]["nodes"].as_array().unwrap();
    assert_eq!(tops.len(), 1, "one top-level node, whatever the format: {tops:?}");
    let root = j["nodes"][tops[0].as_u64().unwrap() as usize].clone();
    let children = root["children"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| j["nodes"][c.as_u64().unwrap() as usize]["name"].clone())
        .collect();
    (root, children)
}

#[test]
fn one_box_is_the_same_document_in_every_format() {
    // Z up to Y up: a quarter turn about X.
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let quarter = json!([-h, 0.0, 0.0, h]);
    let centimeters = json!([0.01, 0.01, 0.01]);
    // (file, bytes, the root's rotation, the root's scale, what was converted, the up axis, the unit's length)
    let z = Fact::Declared(Axis::PosZ);
    #[allow(clippy::type_complexity, reason = "a table of cases")]
    let cases: Vec<(&str, Vec<u8>, Value, Value, [bool; 2], Fact<Axis>, Fact<f64>)> = vec![
        (
            "box.fbx",
            boxed::fbx(),
            quarter.clone(),
            centimeters.clone(),
            [true, true],
            z,
            Fact::Declared(0.01),
        ),
        (
            "box.dae",
            boxed::dae(),
            quarter.clone(),
            centimeters.clone(),
            [true, true],
            z,
            Fact::Declared(0.01),
        ),
        (
            "box.usda",
            boxed::usda(),
            quarter.clone(),
            centimeters.clone(),
            [true, true],
            z,
            Fact::Declared(0.01),
        ),
        // 3MF is millimeters here, its coordinates ten times the others'.
        (
            "box.3mf",
            boxed::threemf(),
            quarter.clone(),
            json!([0.001, 0.001, 0.001]),
            [true, true],
            z,
            Fact::Declared(0.001),
        ),
        // X3D's standard is Y up: its root only scales.
        (
            "box.x3d",
            boxed::x3d(),
            Value::Null,
            centimeters.clone(),
            [true, false],
            Fact::Declared(Axis::PosY),
            Fact::Declared(0.01),
        ),
    ];
    for (name, bytes, rotation, scale, [units, up], up_axis, meters) in cases {
        let doc = read(&bytes, name);
        let (root, children) = root_of(&doc);
        assert_eq!(root["name"], "RootNode", "{name}");
        assert_eq!(
            children,
            [json!("Cube")],
            "{name}: the file's one root is the root node's one child"
        );
        assert_eq!(root.get("rotation").cloned().unwrap_or(Value::Null), rotation, "{name}");
        assert_eq!(root["scale"], scale, "{name}");
        assert!(
            root.get("mesh").is_none() && root.get("translation").is_none(),
            "{name}: {root}"
        );
        let (min, max, triangles, outward) = drawn(&doc);
        assert_eq!(triangles, 12, "{name}");
        assert!(
            near(min, [0.0, 0.0, -2.0]) && near(max, [1.0, 3.0, 0.0]),
            "{name}: {min:?}..{max:?}"
        );
        assert!(outward, "{name}: every face faces out");
        let info = doc.source_info();
        assert_eq!((info.converted.units, info.converted.up_axis), (units, up), "{name}");
        assert!(!info.converted.handedness, "{name}");
        assert_eq!(info.up_axis, up_axis, "{name}");
        assert_eq!(info.meters_per_unit, meters, "{name}");
        assert_eq!(info.output_meters_per_unit(), Fact::Declared(1.0), "{name}");
    }

    // 3DS: the same shape of document. It names no unit, so none is known or converted; its master scale, a
    // hundredth here, is a factor the root node scales by.
    let doc = read(&boxed::tds(), "box.3ds");
    let (root, children) = root_of(&doc);
    assert_eq!((&root["name"], &children), (&json!("RootNode"), &vec![json!("Cube")]));
    assert_eq!(root["rotation"], quarter);
    let (min, max, triangles, outward) = drawn(&doc);
    assert!(
        near(min, [0.0, 0.0, -2.0]) && near(max, [1.0, 3.0, 0.0]),
        "{min:?}..{max:?}"
    );
    assert!(triangles == 12 && outward);
    let info = doc.source_info();
    assert_eq!(info.meters_per_unit, Fact::Unknown);
    assert!((root["scale"][0].as_f64().unwrap() - 0.01).abs() < 1e-8, "{root}");
    assert!(!info.converted.units && info.converted.up_axis);

    // OBJ names no axis and no unit: the same shape of document, with a root node that changes nothing.
    let doc = read(&boxed::obj(), "box.obj");
    let (root, children) = root_of(&doc);
    assert_eq!(root, json!({ "name": "RootNode", "children": root["children"] }));
    assert_eq!(children, [json!("Cube")]);
    let (min, max, triangles, outward) = drawn(&doc);
    assert!(
        near(min, [0.0; 3]) && near(max, [100.0, 200.0, 300.0]),
        "{min:?}..{max:?}"
    );
    assert!(triangles == 12 && outward);
    let info = doc.source_info();
    assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
    assert_eq!(info.meters_per_unit, Fact::Unknown);
    assert_eq!(info.converted, polysmelt::source::Converted::default());
    assert_eq!(info.output_meters_per_unit(), Fact::Unknown);
}

#[test]
fn sampled_data_is_converted_in_place_and_gltf_is_kept_as_it_is() {
    // A scanner's cloud is Z up, in `.xyz` and in `.pcd` alike: the same points land the same way, with no
    // root node, and say what was assumed.
    let xyz = read(b"1 2 3\n4 5 6\n", "a.xyz");
    let pcd = read(
        b"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n1 2 3\n4 5 6\n",
        "a.pcd",
    );
    for (name, doc) in [("xyz", &xyz), ("pcd", &pcd)] {
        let j = doc.json();
        assert_eq!(j["nodes"].as_array().unwrap().len(), 1, "{name}: no root node");
        assert!(j["nodes"][0].get("name").is_none() && j["nodes"][0].get("rotation").is_none());
        let info = doc.source_info();
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ), "{name}");
        assert_eq!(info.meters_per_unit, Fact::Unknown, "{name}");
        assert!(info.converted.up_axis && !info.converted.units, "{name}");
        let mesh = doc.flatten(polysmelt::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(mesh.prims[0].positions, [[1.0, 3.0, -2.0], [4.0, 6.0, -5.0]], "{name}");
    }
    // A PLY, OFF or VTK mesh says nothing of its axes and has no convention: nothing is assumed or converted,
    // and it is a scene like any other mesh.
    let off = b"OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
    let ply = b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
    let vtk = b"# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\nPOINTS 3 float\n0 0 0 1 0 0 0 1 0\nPOLYGONS 1 4\n3 0 1 2\n";
    for (name, bytes) in [("a.off", &off[..]), ("a.ply", &ply[..]), ("a.vtk", &vtk[..])] {
        let doc = read(bytes, name);
        let info = doc.source_info();
        assert_eq!(info.up_axis, Fact::Unknown, "{name}");
        assert_eq!(info.meters_per_unit, Fact::Unknown, "{name}");
        assert_eq!(info.converted, polysmelt::source::Converted::default(), "{name}");
        let (root, _) = root_of(&doc);
        assert_eq!(root["name"], "RootNode", "{name}");
        assert!(root.get("rotation").is_none() && root.get("scale").is_none(), "{name}");
    }
    // A PLY of only points is a cloud: no root node.
    let cloud = b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n4 5 6\n";
    let doc = read(cloud, "a.ply");
    assert_eq!(doc.json()["nodes"].as_array().unwrap().len(), 1);
    assert_eq!(doc.source_info().up_axis, Fact::Unknown);
    // glTF is in glTF's frame already: no node is added.
    let gltf = br#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0,1]}],"nodes":[{"name":"a"},{"name":"b"}]}"#;
    let doc = read(gltf, "a.gltf");
    assert_eq!(doc.json()["scenes"][0]["nodes"], json!([0, 1]));
    assert_eq!(doc.json()["nodes"].as_array().unwrap().len(), 2);
}

#[test]
fn a_motion_capture_file_that_names_no_unit_is_not_scaled() {
    // TRC without `Units`, as HTR without `CalibrationUnits` and BVH always: the unit is unknown, and the root
    // node has no scale. With one, the root scales to meters and the markers stay as the file has them.
    let trc = |units: &str, value: &str| {
        format!(
            "PathFileType\t4\t(X/Y/Z)\tt.trc\nDataRate\tNumFrames\tNumMarkers{units}\n10\t1\t1{value}\nFrame#\tTime\ta\t\t\n\t\tX1\tY1\tZ1\n\n1\t0\t100\t200\t300\n"
        )
    };
    let silent = read(trc("", "").as_bytes(), "a.trc");
    let (root, children) = root_of(&silent);
    assert_eq!(children, [json!("markers")]);
    assert!(root.get("scale").is_none(), "{root}");
    assert_eq!(silent.source_info().meters_per_unit, Fact::Unknown);
    assert!(!silent.source_info().converted.units);
    let declared = read(trc("\tUnits", "\tmm").as_bytes(), "a.trc");
    let (root, _) = root_of(&declared);
    assert_eq!(root["scale"], json!([0.001, 0.001, 0.001]));
    assert_eq!(declared.source_info().meters_per_unit, Fact::Declared(0.001));
    for doc in [&silent, &declared] {
        let rig = doc.rig().unwrap();
        let a = rig.find("a").unwrap();
        assert_eq!(rig.nodes[a].rest.translation, [100.0, 200.0, 300.0]);
    }
    let at = |doc: &Document| {
        let rig = doc.rig().unwrap();
        let m = rig.world(&rig.pose(None, 0.0))[rig.find("a").unwrap()];
        [m[12], m[13], m[14]]
    };
    assert!(near(at(&silent), [100.0, 200.0, 300.0]));
    assert!(near(at(&declared), [0.1, 0.2, 0.3]));
    let bvh = read(
        b"HIERARCHY\nROOT Hips\n{\nOFFSET 0 90 0\nCHANNELS 3 Xrotation Yrotation Zrotation\nEnd Site\n{\nOFFSET 0 10 0\n}\n}\nMOTION\nFrames: 1\nFrame Time: 0.04\n0 0 0\n",
        "a.bvh",
    );
    let (root, children) = root_of(&bvh);
    assert_eq!(children, [json!("Hips")]);
    assert_eq!(root, json!({ "name": "RootNode", "children": root["children"] }));
    assert_eq!(bvh.source_info().meters_per_unit, Fact::Unknown);
}
