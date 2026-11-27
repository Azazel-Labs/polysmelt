//! The geometry rules every reader follows (`src/geometry.rs`), held against the same small model written in each
//! text format: a triangle that names a missing vertex or a position that isn't finite is dropped and counted,
//! triangles with no area or a repeated corner are kept, and every triangle primitive has unit normals. The
//! document as read, its GLB and its flat mesh must agree.

#![allow(
    clippy::cast_possible_truncation,
    reason = "accessor numbers and vertex counts of models of four vertices"
)]

use polysmelt::document::Pose;
use polysmelt::{Document, Loader};
use serde_json::Value;

/// Whether this build reads files of extension `ext`.
#[allow(
    clippy::match_like_matches_macro,
    reason = "the arms are cfg! values that differ per build"
)]
fn built(ext: &str) -> bool {
    match ext {
        "obj" => cfg!(feature = "obj"),
        "off" => cfg!(feature = "off"),
        "ply" => cfg!(feature = "ply"),
        "stl" => cfg!(feature = "stl"),
        "dae" => cfg!(feature = "dae"),
        "x3d" => cfg!(feature = "x3d"),
        "amf" => cfg!(feature = "amf"),
        "x" => cfg!(feature = "directx"),
        "vtk" => cfg!(feature = "vtk"),
        "ac" => cfg!(feature = "ac3d"),
        "dxf" => cfg!(feature = "dxf"),
        "irrmesh" => cfg!(feature = "irr"),
        "xgl" => cfg!(feature = "xgl"),
        "gltf" => cfg!(feature = "gltf"),
        "fbx" => cfg!(feature = "fbx"),
        "3mf" => cfg!(feature = "3mf"),
        "mesh" => cfg!(feature = "medit"),
        _ => false,
    }
}

fn load(ext: &str, bytes: &[u8]) -> Document {
    Loader::new()
        .read(bytes.to_vec(), &format!("model.{ext}"))
        .unwrap_or_else(|e| panic!("{ext}: {e}"))
}

/// What a document's triangle primitives hold: (triangles, whether every one has normals), after checking that
/// no position is NaN or infinite, no index names a missing vertex and every normal is unit length.
fn survey(ext: &str, doc: &Document) -> (usize, bool) {
    let (mut triangles, mut normals) = (0, true);
    let meshes = doc.json().get("meshes").and_then(Value::as_array).cloned();
    for prim in meshes
        .iter()
        .flatten()
        .flat_map(|m| m["primitives"].as_array().unwrap())
    {
        let attribute = |name: &str| prim["attributes"].get(name).and_then(Value::as_u64).map(|a| a as usize);
        let (_, pos) = doc.read_floats(attribute("POSITION").unwrap()).unwrap();
        assert!(
            pos.iter().all(|x| x.is_finite()),
            "{ext}: a position isn't finite: {pos:?}"
        );
        let count = pos.len() / 3;
        let indices = match prim.get("indices").and_then(Value::as_u64) {
            Some(a) => doc.read_indices(a as usize).unwrap(),
            None => (0..count as u32).collect(),
        };
        assert!(
            indices.iter().all(|&i| (i as usize) < count),
            "{ext}: an index names no vertex: {indices:?} of {count}"
        );
        if prim.get("mode").and_then(Value::as_u64).unwrap_or(4) != 4 {
            continue;
        }
        triangles += indices.len() / 3;
        match attribute("NORMAL") {
            None => normals = false,
            Some(a) => {
                let (_, n) = doc.read_floats(a).unwrap();
                assert_eq!(n.len(), pos.len(), "{ext}: a normal for each vertex");
                for v in n.chunks_exact(3) {
                    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    assert!((l - 1.0).abs() < 1e-4, "{ext}: normal {v:?} isn't unit length");
                }
            }
        }
    }
    (triangles, normals)
}

/// The triangles of the document as read, of the GLB written from it, and of its flat mesh.
fn views(ext: &str, doc: &Document) -> [usize; 3] {
    let (read, _) = survey(ext, doc);
    let glb = doc.to_glb().unwrap();
    let back = Loader::new().read(glb, "model.glb").unwrap();
    let (written, _) = survey(ext, &back);
    let flat = doc.flatten(Pose::Rest).unwrap().map_or(0, |m| m.triangles());
    [read, written, flat]
}

fn dropped(doc: &Document) -> Option<u64> {
    doc.json()["asset"]
        .pointer("/extras/source/droppedGeometry")
        .and_then(Value::as_u64)
}

/// Runs `check` on every model this build reads, and fails with every format that failed, not only the first.
fn each(models: &[(&str, &[u8])], check: impl Fn(&str, &Document) + std::panic::RefUnwindSafe) {
    let mut failed = Vec::new();
    for &(ext, bytes) in models.iter().filter(|(ext, _)| built(ext)) {
        let run = std::panic::catch_unwind(|| check(ext, &load(ext, bytes)));
        if let Err(e) = run {
            let text = e.downcast_ref::<String>().cloned();
            failed.push(text.unwrap_or_else(|| ext.to_owned()));
        }
    }
    assert!(
        failed.is_empty(),
        "{} failed:
{}",
        failed.len(),
        failed.join(
            "
"
        )
    );
}

/// A good triangle, one with no area (three points in a line) and one with a repeated corner: all three are
/// kept, in every format.
#[test]
fn triangles_without_area_are_kept() {
    each(DEGENERATE, |ext, doc| {
        assert_eq!(views(ext, doc), [3; 3], "{ext}");
        assert_eq!(dropped(doc), None, "{ext}");
    });
}

/// A good triangle, one on a vertex whose position is NaN and (but for STL, which has no indices) one naming
/// vertex 99 of 4: only the first is drawn, in the document, its GLB and its flat mesh alike, and the others are
/// counted.
#[test]
fn triangles_that_cannot_be_drawn_are_dropped_and_counted() {
    each(BROKEN, |ext, doc| {
        // DirectX text has no word for a number that isn't one: its fourth vertex is a point like the others.
        let (drawn, want) = match ext {
            "stl" => (1, 1),
            "x" => (2, 1),
            _ => (1, 2),
        };
        assert_eq!(views(ext, doc), [drawn; 3], "{ext}");
        assert_eq!(dropped(doc), Some(want), "{ext}");
    });
}

/// The unit cube, its eight corners shared by its six faces, with no normals and nothing about smoothing.
const OBJ_CUBE: &str = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\nf 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\nf 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n";
const OFF_CUBE: &str = "OFF\n8 6 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0 0 1\n1 0 1\n1 1 1\n0 1 1\n4 0 3 2 1\n4 4 5 6 7\n4 0 1 5 4\n4 1 2 6 5\n4 2 3 7 6\n4 3 0 4 7\n";
const PLY_CUBE: &str = "ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 6\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0 0 1\n1 0 1\n1 1 1\n0 1 1\n4 0 3 2 1\n4 4 5 6 7\n4 0 1 5 4\n4 1 2 6 5\n4 2 3 7 6\n4 3 0 4 7\n";
const STL_CUBE: &str = "solid cube\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 1 0\nvertex 1 1 0\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 1 0\nvertex 1 0 0\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 1\nvertex 1 0 1\nvertex 1 1 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 1\nvertex 1 1 1\nvertex 0 1 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 1 0 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 0 1\nvertex 0 0 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 1 0 0\nvertex 1 1 0\nvertex 1 1 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 1 0 0\nvertex 1 1 1\nvertex 1 0 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 1 1 0\nvertex 0 1 0\nvertex 0 1 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 1 1 0\nvertex 0 1 1\nvertex 1 1 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 1 0\nvertex 0 0 0\nvertex 0 0 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 1 0\nvertex 0 0 1\nvertex 0 1 1\nendloop\nendfacet\nendsolid cube\n";
const DXF_CUBE: &str = "0\nSECTION\n2\nENTITIES\n0\n3DFACE\n8\n0\n10\n0\n20\n0\n30\n0\n11\n0\n21\n1\n31\n0\n12\n1\n22\n1\n32\n0\n13\n1\n23\n0\n33\n0\n0\n3DFACE\n8\n0\n10\n0\n20\n0\n30\n1\n11\n1\n21\n0\n31\n1\n12\n1\n22\n1\n32\n1\n13\n0\n23\n1\n33\n1\n0\n3DFACE\n8\n0\n10\n0\n20\n0\n30\n0\n11\n1\n21\n0\n31\n0\n12\n1\n22\n0\n32\n1\n13\n0\n23\n0\n33\n1\n0\n3DFACE\n8\n0\n10\n1\n20\n0\n30\n0\n11\n1\n21\n1\n31\n0\n12\n1\n22\n1\n32\n1\n13\n1\n23\n0\n33\n1\n0\n3DFACE\n8\n0\n10\n1\n20\n1\n30\n0\n11\n0\n21\n1\n31\n0\n12\n0\n22\n1\n32\n1\n13\n1\n23\n1\n33\n1\n0\n3DFACE\n8\n0\n10\n0\n20\n1\n30\n0\n11\n0\n21\n0\n31\n0\n12\n0\n22\n0\n32\n1\n13\n0\n23\n1\n33\n1\n0\nENDSEC\n0\nEOF\n";
const VTK_CUBE: &str = "# vtk DataFile Version 3.0\ncube\nASCII\nDATASET POLYDATA\nPOINTS 8 float\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0 0 1\n1 0 1\n1 1 1\n0 1 1\nPOLYGONS 6 30\n4 0 3 2 1\n4 4 5 6 7\n4 0 1 5 4\n4 1 2 6 5\n4 2 3 7 6\n4 3 0 4 7\n";
const MEDIT_CUBE: &str = "MeshVersionFormatted 2
Dimension 3
Vertices
8
0 0 0 0
1 0 0 0
1 1 0 0
0 1 0 0
0 0 1 0
1 0 1 0
1 1 1 0
0 1 1 0
Quadrilaterals
6
1 4 3 2 0
5 6 7 8 0
1 2 6 5 0
2 3 7 6 0
3 4 8 7 0
4 1 5 8 0
End
";
const DAE_CUBE: &str = r##"<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><library_geometries><geometry id="g"><mesh><source id="p"><float_array id="pa" count="24">0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1</float_array><technique_common><accessor source="#pa" count="8" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source><vertices id="v"><input semantic="POSITION" source="#p"/></vertices><triangles count="12"><input semantic="VERTEX" source="#v" offset="0"/><p>0 3 2 0 2 1 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7</p></triangles></mesh></geometry></library_geometries><library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"/></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#s"/></scene></COLLADA>"##;

/// A cube without normals in each format whose reader gives it the default ones, by name and extension. The
/// OBJ is there twice: small, and padded past the size its reader splits over threads. The FBX is there as its
/// binary and its ASCII reader read it; the ASCII one is a corpus cube with its normals taken out.
fn cubes() -> Vec<(&'static str, &'static str, Vec<u8>)> {
    let ascii = include_str!("corpus/fbx/cube_ascii7400/cube_ascii7400.fbx");
    let from = ascii
        .find("LayerElementNormal: 0 {")
        .expect("the ASCII cube has normals");
    let to = ascii
        .find("LayerElementUV: 0 {")
        .expect("the ASCII cube has texture coordinates");
    let fbx_ascii = format!("{}{}", &ascii[..from], &ascii[to..]);
    let padded = format!("{OBJ_CUBE}{}", "# a comment to pad the file\n".repeat(90_000));
    vec![
        ("OBJ", "obj", OBJ_CUBE.into()),
        ("OBJ on threads", "obj", padded.into_bytes()),
        ("OFF", "off", OFF_CUBE.into()),
        ("PLY", "ply", PLY_CUBE.into()),
        ("STL", "stl", STL_CUBE.into()),
        ("DXF", "dxf", DXF_CUBE.into()),
        ("VTK", "vtk", VTK_CUBE.into()),
        ("COLLADA", "dae", DAE_CUBE.into()),
        ("Medit", "mesh", MEDIT_CUBE.into()),
        (
            "FBX binary",
            "fbx",
            include_bytes!("corpus/fbx/no_normals/no_normals.fbx").to_vec(),
        ),
        ("FBX ASCII", "fbx", fbx_ascii.into_bytes()),
        ("3MF", "3mf", include_bytes!("corpus/3mf/cube/cube.3mf").to_vec()),
        ("DirectX", "x", include_bytes!("corpus/x/no_normals/cube.x").to_vec()),
    ]
}

/// What a cube's normals are worth: how many distinct pairs of a position and a normal its triangle primitives
/// hold (24 when each face has its own at each corner, 8 when each corner has one), and whether every normal is
/// along an axis, as a flat face's is.
fn normals_of(name: &str, doc: &Document) -> (usize, bool) {
    let mut pairs = std::collections::BTreeSet::new();
    let mut flat = true;
    let meshes = doc.json().get("meshes").and_then(Value::as_array).cloned();
    for prim in meshes
        .iter()
        .flatten()
        .flat_map(|m| m["primitives"].as_array().unwrap())
    {
        if prim.get("mode").and_then(Value::as_u64).unwrap_or(4) != 4 {
            continue;
        }
        let read = |key: &str| {
            let accessor = prim["attributes"].get(key).and_then(Value::as_u64);
            let accessor = accessor.unwrap_or_else(|| panic!("{name}: a triangle primitive has no {key}"));
            doc.read_floats(accessor as usize).unwrap().1
        };
        let (positions, normals) = (read("POSITION"), read("NORMAL"));
        assert_eq!(positions.len(), normals.len(), "{name}");
        for (p, n) in positions.chunks_exact(3).zip(normals.chunks_exact(3)) {
            flat &= n.iter().filter(|c| c.abs() > 0.999).count() == 1;
            // Rounded, so the same direction worked out twice is one.
            let rounded = |v: &[f32]| -> Vec<i64> { v.iter().map(|c| (f64::from(*c) * 1e4).round() as i64).collect() };
            pairs.insert((rounded(p), rounded(n)));
        }
    }
    (pairs.len(), flat)
}

/// A file with no normals and no smoothing data of its own takes the default ones at the crease angle the caller
/// asks for, in every reader: at 0 every face is flat, at the default of 30 degrees a cube's edges are hard all
/// the same, and at 180 its eight corners are smooth. The angle goes with the `Keep` to wherever a reader does
/// its work, threads included. A glTF file is left without normals whatever the angle.
#[test]
fn the_crease_angle_of_the_default_normals_is_the_callers_in_every_reader() {
    use polysmelt::Keep;
    for (name, ext, bytes) in cubes().into_iter().filter(|(_, ext, _)| built(ext)) {
        let read = |keep: Keep| -> (usize, bool) {
            let keep = keep.with_threads(4);
            let doc = Loader::new().keep(keep).read(bytes.clone(), &format!("cube.{ext}"));
            let doc = doc.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(survey(name, &doc), (12, true), "{name}");
            normals_of(name, &doc)
        };
        assert_eq!(read(Keep::ALL.with_crease_angle(0.0)), (24, true), "{name} at 0");
        assert_eq!(read(Keep::ALL), (24, true), "{name} at the default");
        assert_eq!(read(Keep::ALL.with_crease_angle(30.0)), (24, true), "{name} at 30");
        assert_eq!(read(Keep::ALL.with_crease_angle(180.0)), (8, false), "{name} at 180");
        // Out of range is the nearest end, and not a number is the default.
        assert_eq!(read(Keep::ALL.with_crease_angle(-5.0)), (24, true), "{name} below 0");
        assert_eq!(
            read(Keep::ALL.with_crease_angle(f32::INFINITY)),
            (8, false),
            "{name} past 180"
        );
        assert_eq!(read(Keep::ALL.with_crease_angle(f32::NAN)), (24, true), "{name} at NaN");
    }
    if built("gltf") {
        let (_, bytes) = DEGENERATE.iter().find(|(ext, _)| *ext == "gltf").unwrap();
        let keep = Keep::ALL.with_crease_angle(180.0);
        let doc = Loader::new().keep(keep).read(bytes.to_vec(), "model.gltf").unwrap();
        assert!(!survey("gltf", &doc).1);
    }
}

/// A glTF file is read as it is, so the document holds what the file does; its GLB and its flat mesh drop what
/// can't be drawn, as the readers of the other formats do when they read.
#[test]
fn a_gltf_files_glb_and_flat_mesh_drop_what_cannot_be_drawn() {
    if !built("gltf") {
        return;
    }
    let doc = load("gltf", BROKEN_GLTF);
    assert_eq!(doc.counts().triangles, 3);
    let back = Loader::new().read(doc.to_glb().unwrap(), "model.glb").unwrap();
    assert_eq!(survey("gltf", &back).0, 1);
    assert_eq!(doc.flatten(Pose::Rest).unwrap().unwrap().triangles(), 1);
}

/// A triangle, one on a vertex whose position is NaN, and one naming vertex 99 of 4.
const BROKEN_GLTF: &[u8] = br#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"buffers":[{"byteLength":66,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAADAfwAAAAAAAAAAAAABAAIAAAABAAMAAAABAGMA"}],"bufferViews":[{"buffer":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":18}],"accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},{"bufferView":1,"componentType":5123,"count":9,"type":"SCALAR"}]}"#;

/// Every triangle primitive a reader makes has unit normals, whether or not the file has any. A glTF file
/// without normals keeps none: glTF defines flat shading for it.
#[test]
fn triangle_primitives_have_normals() {
    each(DEGENERATE, |ext, doc| {
        let (_, normals) = survey(ext, doc);
        assert_eq!(normals, ext != "gltf", "{ext}");
    });
}

const DEGENERATE: &[(&str, &[u8])] = &[
    ("obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 2 0 0\nf 1 2 3\nf 1 2 4\nf 1 2 2\n"),
    ("off", b"OFF\n4 3 0\n0 0 0\n1 0 0\n0 1 0\n2 0 0\n3 0 1 2\n3 0 1 3\n3 0 1 1\n"),
    ("ply", b"ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 3\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n2 0 0\n3 0 1 2\n3 0 1 3\n3 0 1 1\n"),
    ("stl", b"solid a\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 2 0 0\nendloop\nendfacet\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 1 0 0\nendloop\nendfacet\nendsolid a\n"),
    ("dae", br##"<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><library_geometries><geometry id="g"><mesh><source id="p"><float_array id="pa" count="12">0 0 0 1 0 0 0 1 0 2 0 0</float_array><technique_common><accessor source="#pa" count="4" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source><vertices id="v"><input semantic="POSITION" source="#p"/></vertices><triangles count="3"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2 0 1 3 0 1 1</p></triangles></mesh></geometry></library_geometries><library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"/></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#s"/></scene></COLLADA>"##),
    ("x3d", br#"<X3D version="3.3"><Scene><Shape><IndexedFaceSet coordIndex="0 1 2 -1 0 1 3 -1 0 1 1 -1"><Coordinate point="0 0 0 1 0 0 0 1 0 2 0 0"/></IndexedFaceSet></Shape></Scene></X3D>"#),
    ("amf", br#"<?xml version="1.0"?><amf unit="millimeter"><object id="1"><mesh><vertices><vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex><vertex><coordinates><x>2</x><y>0</y><z>0</z></coordinates></vertex></vertices><volume><triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle><triangle><v1>0</v1><v2>1</v2><v3>3</v3></triangle><triangle><v1>0</v1><v2>1</v2><v3>1</v3></triangle></volume></mesh></object></amf>"#),
    ("x", b"xof 0303txt 0032\nMesh m {\n4;\n0;0;0;,\n1;0;0;,\n0;1;0;,\n2;0;0;;\n3;\n3;0,1,2;,\n3;0,1,3;,\n3;0,1,1;;\n}\n"),
    ("vtk", b"# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\nPOINTS 4 float\n0 0 0 1 0 0 0 1 0 2 0 0\nPOLYGONS 3 12\n3 0 1 2\n3 0 1 3\n3 0 1 1\n"),
    ("ac", b"AC3Db\nMATERIAL \"m\" rgb 1 1 1 amb 0.2 0.2 0.2 emis 0 0 0 spec 0.5 0.5 0.5 shi 20 trans 0\nOBJECT world\nkids 1\nOBJECT poly\nnumvert 4\n0 0 0\n1 0 0\n0 1 0\n2 0 0\nnumsurf 3\nSURF 0x00\nmat 0\nrefs 3\n0 0 0\n1 0 0\n2 0 0\nSURF 0x00\nmat 0\nrefs 3\n0 0 0\n1 0 0\n3 0 0\nSURF 0x00\nmat 0\nrefs 3\n0 0 0\n1 0 0\n1 0 0\nkids 0\n"),
    ("dxf", b"0\nSECTION\n2\nENTITIES\n0\n3DFACE\n8\n0\n10\n0\n20\n0\n30\n0\n11\n1\n21\n0\n31\n0\n12\n0\n22\n1\n32\n0\n13\n0\n23\n1\n33\n0\n0\n3DFACE\n8\n0\n10\n0\n20\n0\n30\n0\n11\n1\n21\n0\n31\n0\n12\n2\n22\n0\n32\n0\n13\n2\n23\n0\n33\n0\n0\n3DFACE\n8\n0\n10\n0\n20\n0\n30\n0\n11\n1\n21\n0\n31\n0\n12\n1\n22\n0\n32\n0\n13\n1\n23\n0\n33\n0\n0\nENDSEC\n0\nEOF\n"),
    ("irrmesh", br#"<?xml version="1.0"?><mesh xmlns="http://irrlicht.sourceforge.net/IRRMESH_09_2007" version="1.0"><buffer><material></material><vertices type="standard" vertexCount="4">
0 0 0 0 0 1 ffffffff 0 0
1 0 0 0 0 1 ffffffff 0 0
0 1 0 0 0 1 ffffffff 0 0
2 0 0 0 0 1 ffffffff 0 0
</vertices><indices indexCount="9">0 1 2 0 1 3 0 1 1</indices></buffer></mesh>"#),
    ("xgl", br#"<WORLD><MESH ID="0"><P ID="0">0,0,0</P><P ID="1">1,0,0</P><P ID="2">0,1,0</P><P ID="3">2,0,0</P><F><FV1><PREF>0</PREF></FV1><FV2><PREF>1</PREF></FV2><FV3><PREF>2</PREF></FV3></F><F><FV1><PREF>0</PREF></FV1><FV2><PREF>1</PREF></FV2><FV3><PREF>3</PREF></FV3></F><F><FV1><PREF>0</PREF></FV1><FV2><PREF>1</PREF></FV2><FV3><PREF>1</PREF></FV3></F></MESH><OBJECT><MESHREF>0</MESHREF></OBJECT></WORLD>"#),
    ("gltf", br#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"buffers":[{"byteLength":66,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAQAAAAAAAAAAAAAABAAIAAAABAAMAAAABAAEA"}],"bufferViews":[{"buffer":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":18}],"accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[2,1,0]},{"bufferView":1,"componentType":5123,"count":9,"type":"SCALAR"}]}"#),
];

const BROKEN: &[(&str, &[u8])] = &[
    ("obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv nan 0 0\nf 1 2 3\nf 1 2 4\nf 1 2 99\n"),
    ("off", b"OFF\n4 3 0\n0 0 0\n1 0 0\n0 1 0\nnan 0 0\n3 0 1 2\n3 0 1 3\n3 0 1 99\n"),
    ("ply", b"ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 3\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\nnan 0 0\n3 0 1 2\n3 0 1 3\n3 0 1 99\n"),
    ("stl", b"solid a\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex nan 0 0\nendloop\nendfacet\nendsolid a\n"),
    ("dae", br##"<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><library_geometries><geometry id="g"><mesh><source id="p"><float_array id="pa" count="12">0 0 0 1 0 0 0 1 0 NaN 0 0</float_array><technique_common><accessor source="#pa" count="4" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source><vertices id="v"><input semantic="POSITION" source="#p"/></vertices><triangles count="3"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2 0 1 3 0 1 99</p></triangles></mesh></geometry></library_geometries><library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"/></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#s"/></scene></COLLADA>"##),
    ("x3d", br#"<X3D version="3.3"><Scene><Shape><IndexedFaceSet coordIndex="0 1 2 -1 0 1 3 -1 0 1 99 -1"><Coordinate point="0 0 0 1 0 0 0 1 0 nan 0 0"/></IndexedFaceSet></Shape></Scene></X3D>"#),
    ("amf", br#"<?xml version="1.0"?><amf unit="millimeter"><object id="1"><mesh><vertices><vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates></vertex><vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex><vertex><coordinates><x>nan</x><y>0</y><z>0</z></coordinates></vertex></vertices><volume><triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle><triangle><v1>0</v1><v2>1</v2><v3>3</v3></triangle><triangle><v1>0</v1><v2>1</v2><v3>99</v3></triangle></volume></mesh></object></amf>"#),
    ("vtk", b"# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\nPOINTS 4 float\n0 0 0 1 0 0 0 1 0 nan 0 0\nPOLYGONS 3 12\n3 0 1 2\n3 0 1 3\n3 0 1 99\n"),
    ("x", b"xof 0303txt 0032\nMesh m {\n4;\n0;0;0;,\n1;0;0;,\n0;1;0;,\n2;0;0;;\n3;\n3;0,1,2;,\n3;0,1,3;,\n3;0,1,99;;\n}\n"),
    ("xgl", br#"<WORLD><MESH ID="0"><P ID="0">0,0,0</P><P ID="1">1,0,0</P><P ID="2">0,1,0</P><P ID="3">nan,0,0</P><F><FV1><PREF>0</PREF></FV1><FV2><PREF>1</PREF></FV2><FV3><PREF>2</PREF></FV3></F><F><FV1><PREF>0</PREF></FV1><FV2><PREF>1</PREF></FV2><FV3><PREF>3</PREF></FV3></F><F><FV1><PREF>0</PREF></FV1><FV2><PREF>1</PREF></FV2><FV3><PREF>99</PREF></FV3></F></MESH><OBJECT><MESHREF>0</MESHREF></OBJECT></WORLD>"#),
];
