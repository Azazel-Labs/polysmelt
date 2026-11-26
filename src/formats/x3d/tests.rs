use std::fmt::Write as _;
use std::io::Write as _;

use super::*;
use crate::assets::Memory;
use crate::document::Pose;
use crate::mesh::Mesh;

fn read_in(bytes: &[u8], name: &str, assets: &Memory, max: usize) -> Result<Document> {
    let keep = Keep::ALL.with_max_triangles(max);
    let ext = name.rsplit('.').next().unwrap_or_default().to_owned();
    super::super::load_bytes(bytes, name, &ext, assets, keep)
}

fn read(text: &str, name: &str) -> Document {
    read_in(text.as_bytes(), name, &Memory::new(), usize::MAX).unwrap()
}

fn png() -> Vec<u8> {
    let mut png = Vec::new();
    image::RgbImage::from_pixel(2, 2, image::Rgb([200, 100, 50]))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    png
}

fn mesh(doc: &Document) -> Mesh {
    doc.flatten(Pose::Rest).unwrap().unwrap()
}

fn triangles(doc: &Document) -> u64 {
    doc.facts["triangles"].as_u64().unwrap()
}

fn bounds(m: &Mesh) -> ([f32; 3], [f32; 3]) {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in m.prims.iter().flat_map(|p| &p.positions) {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

const CUBE_WRL: &str = r"#VRML V2.0 utf8
# a unit cube of quads
DEF Cube Transform {
  translation 10 0 0
  children [
    Shape {
      appearance Appearance { material Material { diffuseColor 1 0 0 } }
      geometry IndexedFaceSet {
        coord Coordinate { point [ 0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 1, 1 0 1, 1 1 1, 0 1 1 ] }
        coordIndex [ 0 3 2 1 -1, 4 5 6 7 -1, 0 1 5 4 -1, 2 3 7 6 -1, 1 2 6 5 -1, 0 4 7 3 -1 ]
      }
    }
  ]
}
";

const CUBE_X3D: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" "http://www.web3d.org/specifications/x3d-3.3.dtd">
<X3D profile="Interchange" version="3.3">
  <head><meta name="title" content="cube"/></head>
  <Scene>
    <Transform DEF="Cube" translation="10 0 0">
      <Shape>
        <Appearance><Material diffuseColor="1 0 0"/></Appearance>
        <IndexedFaceSet coordIndex="0 3 2 1 -1 4 5 6 7 -1 0 1 5 4 -1 2 3 7 6 -1 1 2 6 5 -1 0 4 7 3 -1">
          <Coordinate point="0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 1, 1 0 1, 1 1 1, 0 1 1"/>
        </IndexedFaceSet>
      </Shape>
    </Transform>
  </Scene>
</X3D>
"#;

fn check_cube(doc: &Document) {
    assert_eq!(triangles(doc), 12);
    let m = mesh(doc);
    let (lo, hi) = bounds(&m);
    assert_eq!((lo, hi), ([10.0, 0.0, 0.0], [11.0, 1.0, 1.0]));
    // Flat: no normals, one vertex a point.
    assert!(m.prims[0].normals.is_empty() || m.prims[0].positions.len() == 24);
    let red = m.materials[m.prims[0].material].color;
    assert!((red[0] - 1.0).abs() < 1e-6 && red[1] < 1e-6, "{red:?}");
    // Outward: every face's normal points away from the cube's center.
    let p = &m.prims[0];
    for t in p.indices.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| p.positions[i as usize]);
        let n = crate::mesh::face_normal(a, b, c);
        let mid = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0 - [10.5, 0.5, 0.5][k]);
        assert!(n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2] > 0.0);
    }
    assert!(doc.to_glb().is_ok());
}

#[test]
fn classic_cube() {
    let doc = read(CUBE_WRL, "cube.wrl");
    check_cube(&doc);
    let source = &doc.json()["asset"]["extras"]["source"];
    assert_eq!(source["format"], "VRML97");
    assert_eq!(source["version"], "2.0");
    assert_eq!(doc.json()["nodes"][1]["name"], "Cube");
}

#[test]
fn xml_cube() {
    let doc = read(CUBE_X3D, "cube.x3d");
    check_cube(&doc);
    let source = &doc.json()["asset"]["extras"]["source"];
    assert_eq!(source["encoding"], "XML");
    assert_eq!(source["version"], "3.3");
}

#[test]
fn gzipped() {
    let mut z = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(CUBE_WRL.as_bytes()).unwrap();
    let bytes = z.finish().unwrap();
    let doc = read_in(&bytes, "cube.wrz", &Memory::new(), usize::MAX).unwrap();
    check_cube(&doc);
    assert_eq!(doc.json()["asset"]["extras"]["source"]["gzip"], true);
    let mut z = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(CUBE_X3D.as_bytes()).unwrap();
    let doc = read_in(&z.finish().unwrap(), "cube.x3dz", &Memory::new(), usize::MAX).unwrap();
    check_cube(&doc);
}

#[test]
fn line_ends_bom_and_commas() {
    for text in [
        CUBE_WRL.replace('\n', "\r\n"),
        CUBE_WRL.replace('\n', "\r"),
        format!("\u{feff}{CUBE_WRL}"),
        CUBE_WRL.replace(' ', "\t  ,"),
    ] {
        check_cube(&read(&text, "cube.wrl"));
    }
    for text in [CUBE_X3D.replace('\n', "\r\n"), format!("\u{feff}{CUBE_X3D}")] {
        check_cube(&read(&text, "cube.x3d"));
    }
    // UTF-16 with its byte order mark.
    let mut utf16 = vec![0xff, 0xfe];
    utf16.extend(CUBE_X3D.encode_utf16().flat_map(u16::to_le_bytes));
    check_cube(&read_in(&utf16, "cube.x3d", &Memory::new(), usize::MAX).unwrap());
}

#[test]
fn use_instances_a_mesh_once() {
    let text = r"#X3D V3.3 utf8
PROFILE Interchange
Transform { children DEF S Shape { geometry Box { } } }
Transform { translation 5 0 0 children USE S }
ROUTE A.b TO C.d
";
    let doc = read(text, "a.x3dv");
    assert_eq!(doc.json()["meshes"].as_array().unwrap().len(), 1);
    assert_eq!(triangles(&doc), 24);
    let (lo, hi) = bounds(&mesh(&doc));
    assert_eq!((lo[0], hi[0]), (-1.0, 6.0));
    assert_eq!(doc.json()["asset"]["extras"]["source"]["skipped"]["ROUTE"], 1);
}

#[test]
fn a_node_inside_itself_stops() {
    let text = "#VRML V2.0 utf8\nDEF A Group { children [ Shape { geometry Sphere {} } USE A ] }\n";
    let doc = read(text, "a.wrl");
    assert_eq!(doc.json()["meshes"].as_array().unwrap().len(), 1);
    let xml = r#"<X3D><Scene><Group DEF="A"><Shape><Box/></Shape><Group USE="A"/></Group></Scene></X3D>"#;
    let doc = read(xml, "a.x3d");
    assert_eq!(triangles(&doc), 12);
}

#[test]
fn crease_angle_chooses_shading() {
    let cube = |crease: &str| {
        let text = CUBE_WRL.replace("coordIndex", &format!("creaseAngle {crease} coordIndex"));
        mesh(&read(&text, "c.wrl")).prims.remove(0)
    };
    let flat = cube("0");
    assert_eq!(flat.normals, Vec::<[f32; 3]>::new());
    assert_eq!(flat.positions.len(), 8);
    let smooth = cube("3.2");
    assert_eq!(smooth.positions.len(), 8);
    assert_eq!(smooth.normals.len(), 8);
    // Under 90 degrees: every face's own normals, so 24 vertices.
    let creased = cube("1.0");
    assert_eq!(creased.positions.len(), 24);
    for (p, n) in creased.positions.iter().zip(&creased.normals) {
        assert!(n.iter().filter(|c| c.abs() > 0.99).count() == 1, "{p:?} {n:?}");
    }
    // Past 90 degrees: averaged at the corners.
    let rounded = cube("1.6");
    assert_eq!(rounded.positions.len(), 8);
}

#[test]
fn separate_indices_and_per_face_attributes() {
    let text = r#"#VRML V2.0 utf8
Shape {
  appearance Appearance { texture ImageTexture { url "t.png" repeatS FALSE } }
  geometry IndexedFaceSet {
    solid FALSE
    ccw FALSE
    colorPerVertex FALSE
    coord Coordinate { point [ 0 0 0, 1 0 0, 1 1 0, 0 1 0 ] }
    coordIndex [ 0 1 2 -1 0 2 3 ]
    color Color { color [ 1 0 0, 0 0 1 ] }
    colorIndex [ 1 0 ]
    texCoord TextureCoordinate { point [ 0 0, 1 0, 1 1, 0 1 ] }
    texCoordIndex [ 0 1 2 -1 0 2 3 -1 ]
    normal Normal { vector [ 0 0 2 ] }
    normalIndex [ 0 0 0 -1 0 0 0 -1 ]
  }
}
"#;
    let assets = Memory::new().with("t.png", png());
    let doc = read_in(text.as_bytes(), "a.wrl", &assets, usize::MAX).unwrap();
    let m = mesh(&doc);
    let p = &m.prims[0];
    assert_eq!(p.indices.len(), 6);
    // ccw FALSE: wound the other way, away from +Z.
    let [a, b, c] = [0, 1, 2].map(|k| p.positions[p.indices[k] as usize]);
    assert!(crate::mesh::face_normal(a, b, c)[2] < 0.0);
    assert!(p.normals.iter().all(|n| (n[2] - 1.0).abs() < 1e-6));
    // Per-face colors through colorIndex: the first face blue.
    assert_eq!(p.colors[p.indices[0] as usize], [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(p.colors[p.indices[3] as usize], [1.0, 0.0, 0.0, 1.0]);
    // Texture coordinates flipped to glTF's.
    assert!(p.uvs.contains(&[1.0, 0.0]) && p.uvs.contains(&[0.0, 1.0]));
    let mat = &doc.json()["materials"][0];
    assert_eq!(mat["doubleSided"], true);
    assert_eq!(doc.json()["samplers"][0]["wrapS"], CLAMP);
    assert_eq!(doc.json()["images"].as_array().unwrap().len(), 1);
}

#[test]
fn texture_transform_is_baked() {
    let text = r#"<X3D><Scene><Shape><Appearance>
      <ImageTexture url='"t.png" "http://x/t.png"'/>
      <TextureTransform scale="2 2" translation="0.5 0"/>
    </Appearance><IndexedFaceSet coordIndex="0 1 2"><Coordinate point="0 0 0 1 0 0 0 1 0"/>
    <TextureCoordinate point="0 0 1 0 0 1"/></IndexedFaceSet></Shape></Scene></X3D>"#;
    let assets = Memory::new().with("t.png", png());
    let doc = read_in(text.as_bytes(), "a.x3d", &assets, usize::MAX).unwrap();
    let p = &mesh(&doc).prims[0];
    // (0, 0) → translated (0.5, 0) → scaled (1, 0) → flipped (1, 1).
    assert_eq!(p.uvs[0], [1.0, 1.0]);
    assert_eq!(p.uvs[1], [3.0, 1.0]);
}

#[test]
fn triangle_sets() {
    let text = r"#X3D V3.0 utf8
Shape { geometry IndexedTriangleSet { index [ 0 1 2 0 2 3 ] coord Coordinate { point [ 0 0 0 1 0 0 1 1 0 0 1 0 ] } } }
Shape { geometry TriangleSet { coord Coordinate { point [ 0 0 0 1 0 0 1 1 0 ] } normalPerVertex FALSE } }
Shape { geometry IndexedTriangleStripSet { index [ 0 1 2 3 -1 ] coord Coordinate { point [ 0 0 0 1 0 0 0 1 0 1 1 0 ] } } }
Shape { geometry IndexedTriangleFanSet { index [ 0 1 2 3 ] coord Coordinate { point [ 0 0 0 1 0 0 1 1 0 0 1 0 ] } } }
Shape { geometry TriangleFanSet { fanCount [ 4 ] coord Coordinate { point [ 0 0 0 1 0 0 1 1 0 0 1 0 ] } } }
Shape { geometry IndexedQuadSet { index [ 0 1 2 3 ] coord Coordinate { point [ 0 0 0 1 0 0 1 1 0 0 1 0 ] } } }
Shape { geometry ElevationGrid { xDimension 3 zDimension 2 height [ 0 1 0 0 1 0 ] } }
";
    let doc = read(text, "a.x3dv");
    assert_eq!(triangles(&doc), 2 + 1 + 2 + 2 + 2 + 2 + 4);
    let m = mesh(&doc);
    // Every triangle faces +Z, or up for the grid.
    for p in &m.prims {
        for t in p.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| p.positions[i as usize]);
            let n = crate::mesh::face_normal(a, b, c);
            assert!(n[2] > 0.5 || n[1] > 0.5, "{n:?}");
        }
    }
}

#[test]
fn primitives_and_their_sizes() {
    let text = r"#VRML V2.0 utf8
Shape { geometry Box { size 2 4 6 } }
Transform { translation 10 0 0 children Shape { geometry Sphere { radius 2 } } }
Transform { translation 20 0 0 children Shape { geometry Cylinder { radius 1 height 3 } } }
Transform { translation 30 0 0 children Shape { geometry Cone { bottomRadius 1 height 2 bottom FALSE } } }
";
    let doc = read(text, "a.wrl");
    let m = mesh(&doc);
    assert_eq!(m.prims.len(), 4);
    let (lo, hi) = bounds(&m);
    assert_eq!(lo, [-1.0, -2.0, -3.0]);
    assert!((hi[0] - 31.0).abs() < 1e-5 && (hi[2] - 3.0).abs() < 1e-5, "{hi:?}");
    assert!(m.prims.iter().all(|p| p.normals.len() == p.positions.len()));
}

#[test]
fn a_spot_light_takes_its_versions_default_cone() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let cone = |text: &str, name: &str| {
        let doc = read(text, name);
        let s = &doc.json()["extensions"]["KHR_lights_punctual"]["lights"][0]["spot"];
        (
            s["innerConeAngle"].as_f64().unwrap(),
            s["outerConeAngle"].as_f64().unwrap(),
        )
    };
    let near = |a: f64, b: f64| (a - b).abs() < 1e-3;
    // VRML97 and X3D to 3.2: cutOffAngle pi/4, and beamWidth pi/2, which is cut to it.
    for (text, name) in [
        ("#VRML V2.0 utf8\nSpotLight {}\n", "a.wrl"),
        ("#X3D V3.0 utf8\nPROFILE Interchange\nSpotLight {}\n", "a.x3dv"),
        (
            "<X3D profile='Interchange' version='3.2'><Scene><SpotLight/></Scene></X3D>",
            "a.x3d",
        ),
    ] {
        let (inner, outer) = cone(text, name);
        assert!(
            near(outer, FRAC_PI_4) && near(inner, FRAC_PI_4) && inner < outer,
            "{name}: {inner} {outer}"
        );
    }
    // X3D 3.3: beamWidth pi/4 in cutOffAngle pi/2. X3D 4: beamWidth 3 pi/16.
    let (inner, outer) = cone("#X3D V3.3 utf8\nPROFILE Interchange\nSpotLight {}\n", "a.x3dv");
    assert!(near(outer, FRAC_PI_2) && near(inner, FRAC_PI_4), "{inner} {outer}");
    let (inner, outer) = cone(
        "<X3D profile='Interchange' version='4.0'><Scene><SpotLight/></Scene></X3D>",
        "a.x3d",
    );
    assert!(
        near(outer, FRAC_PI_2) && near(inner, PI * 3.0 / 16.0),
        "{inner} {outer}"
    );
    // A beam wider than the cut-off is the cut-off.
    let (inner, outer) = cone(
        "#VRML V2.0 utf8\nSpotLight { cutOffAngle 0.5 beamWidth 1.2 }\n",
        "a.wrl",
    );
    assert!(near(outer, 0.5) && near(inner, 0.5) && inner < outer, "{inner} {outer}");
}

#[test]
fn text_without_a_header_or_root_is_not_x3d() {
    for (text, name) in [
        ("just some words\n", "a.wrl"),
        ("Shape { geometry Box {} }\n", "a.wrl"),
        ("just some words\n", "a.x3dv"),
        ("<html><body>hello</body></html>", "a.x3d"),
        ("<?xml version='1.0'?>\n<!-- a note -->\n<svg/>", "a.x3d"),
        ("#VRML\n", "a.wrl"),
    ] {
        let e = read_in(text.as_bytes(), name, &Memory::new(), usize::MAX).unwrap_err();
        assert_eq!(e.kind(), crate::ErrorKind::UnknownFormat, "{name}: {text}");
    }
    let e = read_in(b" \n", "a.wrl", &Memory::new(), usize::MAX).unwrap_err();
    assert_eq!(e.kind(), crate::ErrorKind::Empty);
    // The root after a declaration, a comment and a doctype, and the headers the formats give.
    let xml = "<?xml version='1.0'?>\n<!-- a note -->\n<!DOCTYPE X3D PUBLIC \"ISO//Web3D//DTD X3D 3.0//EN\" \
               \"http://www.web3d.org/specifications/x3d-3.0.dtd\">\n<X3D><Scene><Shape><Box/></Shape></Scene></X3D>";
    assert_eq!(triangles(&read(xml, "a.x3d")), 12);
    assert_eq!(
        triangles(&read("#VRML V2.0 utf8\nShape { geometry Box {} }\n", "a.wrl")),
        12
    );
    assert_eq!(
        triangles(&read("#X3D V3.0 utf8\nShape { geometry Box {} }\n", "a.x3dv")),
        12
    );
}

#[test]
fn cameras_lights_and_materials() {
    let text = r#"#VRML V2.0 utf8
WorldInfo { title "Room" }
DEF Cam Viewpoint { position 0 1 5 fieldOfView 0.5 }
DirectionalLight { direction 0 -1 0 color 1 1 1 intensity 0.5 }
PointLight { location 1 2 3 radius 10 }
SpotLight { location 0 5 0 direction 0 -1 0 cutOffAngle 0.6 beamWidth 0.3 }
PointLight { on FALSE }
Shape {
  appearance Appearance { material Material { diffuseColor 0.5 0.5 0.5 emissiveColor 1 0 0 transparency 0.25 } }
  geometry Box {}
}
"#;
    let doc = read(text, "a.wrl");
    let j = &doc.json();
    assert_eq!(j["asset"]["extras"]["source"]["title"], "Room");
    assert_eq!(j["cameras"][0]["perspective"]["yfov"], 0.5);
    let lights = j["extensions"]["KHR_lights_punctual"]["lights"].as_array().unwrap();
    assert_eq!(lights.len(), 3);
    assert_eq!(lights[2]["type"], "spot");
    let cam = j["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n.get("camera").is_some())
        .unwrap();
    assert_eq!(cam["translation"], json!([0.0, 1.0, 5.0]));
    // The directional light turned from -Z to point down.
    let sun = &j["nodes"][1];
    let q: Vec<f64> = sun["rotation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let down = rotate([q[0], q[1], q[2], q[3]], [0.0, 0.0, -1.0]);
    assert!((down[1] + 1.0).abs() < 1e-9, "{down:?}");
    let mat = &j["materials"][0];
    assert_eq!(mat["alphaMode"], "BLEND");
    let base = mat["pbrMetallicRoughness"]["baseColorFactor"].as_array().unwrap();
    assert!((base[0].as_f64().unwrap() - 0.214).abs() < 1e-3);
    assert!((base[3].as_f64().unwrap() - 0.75).abs() < 1e-6);
    assert_eq!(mat["emissiveFactor"], json!([1.0, 0.0, 0.0]));
    assert!(doc.to_glb().is_ok());
}

#[test]
fn transform_center_and_unit() {
    // A quarter turn about Z around (1, 0, 0) takes (2, 0, 0) to (1, 1, 0).
    let text = r"#X3D V4.0 utf8
UNIT length centimeters 0.01
Transform { rotation 0 0 1 1.5707963 center 1 0 0 children Shape { geometry PointSet {} } }
Transform { rotation 0 0 1 1.5707963 center 1 0 0 children Shape { geometry IndexedFaceSet {
  coord Coordinate { point [ 2 0 0, 2 0 1, 2.1 0 0 ] } coordIndex [ 0 1 2 ] } } }
";
    let doc = read(text, "a.x3dv");
    let p = &mesh(&doc).prims[0];
    let first = p.positions.iter().find(|q| q[2].abs() < 1e-6 && q[1] < 0.0105).unwrap();
    assert!(
        (first[0] - 0.01).abs() < 1e-6 && (first[1] - 0.01).abs() < 1e-6,
        "{first:?}"
    );
    // A PointSet with no points draws nothing, and isn't skipped either.
    assert!(doc.json()["asset"]["extras"]["source"].get("skipped").is_none());
}

#[test]
fn inline_reads_through_the_assets() {
    let part = "#VRML V2.0 utf8\nShape { geometry Box {} }\n";
    let assets = Memory::new().with("dir/parts/box.wrl", part).with("dir/self.wrl", "");
    let text = r#"#VRML V2.0 utf8
Inline { url [ "missing.wrl" "parts/box.wrl" ] }
Transform { translation 3 0 0 children Inline { url "parts/box.wrl" } }
Inline { url "a.wrl" }
"#;
    let doc = read_in(text.as_bytes(), "dir/a.wrl", &assets, usize::MAX).unwrap();
    assert_eq!(triangles(&doc), 24);
    assert_eq!(doc.json()["meshes"].as_array().unwrap().len(), 1);
}

#[test]
fn skipped_statements() {
    let text = r#"#VRML V2.0 utf8
PROTO Thing [ field SFFloat size 1 ] { Shape { geometry Box { size IS size } } }
EXTERNPROTO Other [ field SFFloat x ] "other.wrl#Other"
DEF T TimeSensor { loop TRUE }
DEF I PositionInterpolator { key [ 0 1 ] keyValue [ 0 0 0 1 1 1 ] }
Script { url "javascript: function f() { if (a) { b(); } }" field SFNode n USE T }
Thing { size 2 }
Shape { geometry Sphere {} }
ROUTE T.fraction_changed TO I.set_fraction
"#;
    let doc = read(text, "a.wrl");
    let skipped = &doc.json()["asset"]["extras"]["source"]["skipped"];
    for k in ["TimeSensor", "PositionInterpolator", "Script", "ROUTE"] {
        assert_eq!(skipped[k], 1, "{k}: {skipped}");
    }
    // The prototype is instanced; the external one isn't found.
    for k in ["PROTO", "EXTERNPROTO", "Thing"] {
        assert!(skipped.get(k).is_none(), "{k}: {skipped}");
    }
    assert_eq!(doc.json()["meshes"].as_array().unwrap().len(), 2);
    assert!(notes(&doc).contains("external prototype"));
}

#[test]
fn vrml_1_is_refused() {
    let err = read_in(b"#VRML V1.0 ascii\nSeparator {}\n", "a.wrl", &Memory::new(), usize::MAX).unwrap_err();
    assert!(err.to_string().contains("VRML 1.0"), "{err}");
}

#[test]
fn triangle_limit() {
    let err = read_in(CUBE_WRL.as_bytes(), "a.wrl", &Memory::new(), 11).unwrap_err();
    assert!(err.to_string().contains("triangles"), "{err}");
    // Instances count too.
    let text = "#VRML V2.0 utf8\nDEF B Shape { geometry Box {} }\nUSE B\n";
    assert!(read_in(text.as_bytes(), "a.wrl", &Memory::new(), 24).is_ok());
    let err = read_in(text.as_bytes(), "a.wrl", &Memory::new(), 23).unwrap_err();
    assert!(err.to_string().contains("triangles"), "{err}");
}

#[test]
fn deep_nesting_doesnt_overflow() {
    let n = 50_000;
    let text = format!(
        "#VRML V2.0 utf8\n{}Shape {{ geometry Box {{}} }}{}",
        "Group { children [ ".repeat(n),
        " ] }".repeat(n)
    );
    let _ = read(&text, "a.wrl");
    let xml = format!(
        "<X3D><Scene>{}<Shape><Box/></Shape>{}</Scene></X3D>",
        "<Group>".repeat(n),
        "</Group>".repeat(n)
    );
    let _ = read(&xml, "a.x3d");
}

#[test]
fn damaged_files_never_panic() {
    let samples = [
        (CUBE_WRL.to_owned(), "a.wrl"),
        (CUBE_X3D.to_owned(), "a.x3d"),
        (
            "#VRML V2.0 utf8\nDEF S Shape { appearance Appearance { texture ImageTexture { url \"t.png\" } } \
             geometry IndexedFaceSet { creaseAngle 1 coord Coordinate { point [ 0 0 0 1 0 0 1 1 0 0 1 0 ] } \
             coordIndex [ 0 1 2 3 -1 ] normalPerVertex FALSE colorPerVertex FALSE color Color { color [ 1 0 0 ] } } }\n\
             Transform { children [ USE S Shape { geometry Cone {} } ] }\nViewpoint {} SpotLight {}\n"
                .to_owned(),
            "b.wrl",
        ),
        (
            "#VRML V2.0 utf8\nPROTO P [ field SFFloat r 1 field MFNode k [ ] ] { Group { children [ \
             Shape { geometry Sphere { radius IS r } } Group { children IS k } P { } ] } }\n\
             EXTERNPROTO E [ field SFFloat r ] [ \"a.wrl#P\" \"#P\" ]\nP { r 2 k [ E { } ] }\n\
             Shape { geometry Extrusion { spine [ 0 0 0, 0 1 0, 1 1 0, 0 0 0 ] scale [ 1 1, 2 2 ] \
             orientation [ 0 1 0 1 ] crossSection [ 0 0, 1 0, 1 1, 0 0 ] creaseAngle 1 } }\n\
             Shape { geometry ArcClose2D { closureType \"CHORD\" } }\n"
                .to_owned(),
            "c.wrl",
        ),
        (
            "<X3D><Scene><ProtoDeclare name=\"P\"><ProtoInterface><field name=\"s\" type=\"SFVec2f\" \
             accessType=\"initializeOnly\" value=\"1 1\"/></ProtoInterface><ProtoBody><Shape><Rectangle2D>\
             <IS><connect nodeField=\"size\" protoField=\"s\"/></IS></Rectangle2D></Shape></ProtoBody>\
             </ProtoDeclare><ProtoInstance name=\"P\"><fieldValue name=\"s\" value=\"2 3\"/></ProtoInstance>\
             <ExternProtoDeclare name=\"Q\" url='\"a.x3d#P\"'/><ProtoInstance name=\"Q\"/></Scene></X3D>\n"
                .to_owned(),
            "c.x3d",
        ),
    ];
    let assets = Memory::new()
        .with("a.wrl", samples[3].0.clone())
        .with("a.x3d", samples[4].0.clone());
    let check = |bytes: &[u8], name: &str| {
        if let Ok(doc) = read_in(bytes, name, &assets, 64) {
            let _ = doc.flatten(Pose::Rest);
            let _ = doc.to_glb();
        }
    };
    for (text, name) in &samples {
        let sample = text.as_bytes();
        for n in 0..=sample.len() {
            check(&sample[..n], name);
        }
        let mut bytes = sample.to_vec();
        for i in 0..bytes.len() {
            for flip in [0x01, 0x20, 0x80, 0xff] {
                bytes[i] ^= flip;
                check(&bytes, name);
                bytes[i] ^= flip;
            }
        }
    }
}

#[test]
fn line_and_point_sets() {
    // A bad index breaks the polyline and -1 starts the next; a color index past the colors is
    // white. A LineSet run past the points is cut short, and a run of one point draws nothing.
    let doc = read(
        "#VRML V2.0 utf8\nShape { geometry IndexedLineSet { coord Coordinate { point [ 0 0 0, 1 0 0, 1 1 0, 0 1 0 ] } \
         coordIndex [ 0 1 7 2 3 -1 3 0 ] color Color { color [ 1 0 0 ] } colorIndex [ 0 0 0 0 0 0 0 5 ] } }\n\
         Shape { geometry PointSet { coord Coordinate { point [ 0 0 4, 1e999 0 0 ] } } }\n\
         Shape { geometry LineSet { vertexCount [ 3, 99, 1 ] coord Coordinate { point [ 0 0 5, 1 0 5, 1 1 5, 0 1 5 ] } } }\n",
        "a.wrl",
    );
    assert_eq!(triangles(&doc), 0);
    let m = mesh(&doc);
    assert_eq!((m.lines(), m.points()), (3 + 2, 1));
    let colors: Vec<[f32; 4]> = m.prims.iter().flat_map(|p| p.colors.iter().copied()).collect();
    assert!(
        colors.contains(&[1.0, 0.0, 0.0, 1.0]) && colors.contains(&[1.0; 4]),
        "{colors:?}"
    );
    let (lo, hi) = bounds(&m);
    assert_eq!((lo, hi), ([0.0; 3], [1.0, 1.0, 5.0]));
    // Lines and points are unlit.
    assert!(doc.json()["materials"][0]["extensions"]["KHR_materials_unlit"].is_object());
}

/// Whether every triangle faces away from `center`.
fn outward(m: &Mesh, center: [f32; 3]) -> bool {
    m.prims.iter().all(|p| {
        p.indices.as_chunks::<3>().0.iter().all(|t| {
            let [a, b, c] = t.map(|i| p.positions[i as usize]);
            let n = crate::mesh::face_normal(a, b, c);
            let mid = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0 - center[k]);
            n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2] >= -1e-5
        })
    })
}

fn near(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() < 1e-5)
}

fn notes(doc: &Document) -> String {
    doc.json()["asset"]["extras"]["source"]["notes"].to_string()
}

fn base_colors(doc: &Document) -> Vec<String> {
    doc.json()["materials"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["pbrMetallicRoughness"]["baseColorFactor"].to_string())
        .collect()
}

#[test]
fn prototypes_are_instanced() {
    let text = r"#VRML V2.0 utf8
PROTO Brick [
  field SFVec3f size 1 1 1
  exposedField SFColor color 1 0 0
  eventIn SFBool set_on
  field MFNode extra [ ]
] {
  Transform { children [
    Shape { appearance Appearance { material Material { diffuseColor IS color } } geometry Box { size IS size } }
    Group { children IS extra }
  ] }
  DEF Unused Shape { geometry Sphere { } }
}
PROTO Wall [ field SFColor tint 0 0 1 ] {
  Transform { translation 5 0 0 children [ Brick { color IS tint size 2 2 2 } ] }
}
DEF A Brick { size 4 1 1 }
Transform { translation 0 3 0 children Brick { extra [ Shape { geometry Box { size 1 1 1 } } ] } }
Wall { }
USE A
";
    let doc = read(text, "a.wrl");
    // A, the second Brick with its extra box, Wall's Brick, and A again.
    assert_eq!(triangles(&doc), 12 * 5);
    let (lo, hi) = bounds(&mesh(&doc));
    assert_eq!((lo, hi), ([-2.0, -1.0, -1.0], [6.0, 3.5, 1.0]));
    let colors = base_colors(&doc);
    assert!(colors.contains(&"[0.0,0.0,1.0,1.0]".to_owned()), "{colors:?}");
    assert!(colors.contains(&"[1.0,0.0,0.0,1.0]".to_owned()), "{colors:?}");
    // The instance keeps its DEF name.
    let nodes = doc.json()["nodes"].as_array().unwrap();
    assert!(nodes.iter().any(|n| n["name"] == "A"));
    assert!(doc.json()["asset"]["extras"]["source"].get("skipped").is_none());
}

#[test]
fn prototypes_in_xml() {
    let lib = r#"<X3D><Scene>
<ProtoDeclare name="Ball"><ProtoInterface><field name="r" type="SFFloat" accessType="initializeOnly" value="1"/></ProtoInterface>
<ProtoBody><Shape><Sphere><IS><connect nodeField="radius" protoField="r"/></IS></Sphere></Shape></ProtoBody></ProtoDeclare>
</Scene></X3D>"#;
    let text = r#"<X3D><Scene>
<ExternProtoDeclare name="Ball" url='"missing.x3d#Ball" "lib/balls.x3d#Ball"'><field name="r" type="SFFloat" accessType="initializeOnly"/></ExternProtoDeclare>
<ProtoDeclare name="Tile">
  <ProtoInterface>
    <field name="size" type="SFVec2f" accessType="inputOutput" value="2 2"/>
    <field name="name" type="SFString" accessType="initializeOnly" value="12"/>
    <field name="look" type="SFNode" accessType="initializeOnly"><Appearance><Material diffuseColor="0 1 0"/></Appearance></field>
    <field name="touched" type="SFTime" accessType="outputOnly"/>
  </ProtoInterface>
  <ProtoBody>
    <Transform DEF="T"><Shape DEF="S"><IS><connect nodeField="appearance" protoField="look"/></IS><Rectangle2D><IS><connect nodeField="size" protoField="size"/></IS></Rectangle2D></Shape></Transform>
  </ProtoBody>
</ProtoDeclare>
<ProtoInstance name="Tile" DEF="Floor"><fieldValue name="size" value="4 6"/></ProtoInstance>
<Transform translation="10 0 0"><ProtoInstance name="Tile"><fieldValue name="look"><Appearance><Material diffuseColor="1 1 0"/></Appearance></fieldValue></ProtoInstance></Transform>
<Transform translation="0 0 5"><ProtoInstance name="Ball" containerField="children"><fieldValue name="r" value="3"/></ProtoInstance></Transform>
<Shape USE="S"/>
</Scene></X3D>"#;
    let assets = Memory::new().with("dir/lib/balls.x3d", lib);
    let doc = read_in(text.as_bytes(), "dir/a.x3d", &assets, usize::MAX).unwrap();
    let m = mesh(&doc);
    let (lo, hi) = bounds(&m);
    // Floor's 4 by 6 tile, the other's default 2 by 2 at X = 10, and the ball of radius 3.
    assert!(
        near(lo, [-3.0, -3.0, 0.0]) && near(hi, [11.0, 3.0, 8.0]),
        "{lo:?} {hi:?}"
    );
    let colors = base_colors(&doc);
    assert!(colors.contains(&"[0.0,1.0,0.0,1.0]".to_owned()), "{colors:?}");
    assert!(colors.contains(&"[1.0,1.0,0.0,1.0]".to_owned()), "{colors:?}");
    let nodes = doc.json()["nodes"].as_array().unwrap();
    assert!(nodes.iter().any(|n| n["name"] == "Floor"));
    // The body's own names stay inside it.
    assert!(doc.json()["asset"]["extras"]["source"].get("skipped").is_none());
}

#[test]
fn external_prototypes_read_through_the_assets() {
    let lib = r#"#VRML V2.0 utf8
PROTO Leg [ field SFFloat h 1 ] { Transform { children Shape { geometry Cylinder { height IS h radius 0.1 } } } }
PROTO Stool [ field SFFloat h 2 ] { Group { children [ Leg { h IS h } Shape { geometry Box { size 1 0.1 1 } } ] } }
EXTERNPROTO Loop [ ] [ "lib.wrl#Stool" ]
"#;
    let text = r##"#VRML V2.0 utf8
EXTERNPROTO Seat [ field SFFloat h ] [ "missing.wrl#Stool" "parts/lib.wrl#Stool" ]
EXTERNPROTO First [ ] "parts/lib.wrl"
EXTERNPROTO Gone [ ] "nowhere.wrl"
EXTERNPROTO Same [ ] "#Mine"
PROTO Mine [ ] { Shape { geometry Sphere { } } }
Seat { h 4 }
Transform { translation 5 0 0 children First { } }
Gone { }
Transform { translation 0 0 -5 children Same { } }
"##;
    let assets = Memory::new().with("dir/parts/lib.wrl", lib);
    let doc = read_in(text.as_bytes(), "dir/a.wrl", &assets, usize::MAX).unwrap();
    let (lo, hi) = bounds(&mesh(&doc));
    assert!(
        near(lo, [-1.0, -2.0, -6.0]) && near(hi, [5.1, 2.0, 0.5]),
        "{lo:?} {hi:?}"
    );
    assert_eq!(doc.json()["asset"]["extras"]["source"]["skipped"]["Gone"], 1);
    assert!(notes(&doc).contains("external prototype"), "{}", notes(&doc));
}

#[test]
fn prototypes_that_instance_themselves_stop() {
    // Straight down: each level one box deeper, to the nesting limit.
    let text = "#VRML V2.0 utf8\nPROTO A [ ] { Transform { translation 1 0 0 children [ Shape { geometry Box { } } A { } ] } }\nA { }\n";
    let doc = read(text, "a.wrl");
    assert!(notes(&doc).contains("nested too deep"), "{}", notes(&doc));
    assert_eq!(doc.json()["meshes"].as_array().unwrap().len(), 32);
    // Doubling at each level: stopped by the node budget, or here the triangle limit.
    let text =
        "#VRML V2.0 utf8\nPROTO B [ ] { Group { children [ Shape { geometry Box { } } B { } B { } ] } }\nB { }\n";
    let start = std::time::Instant::now();
    match read_in(text.as_bytes(), "a.wrl", &Memory::new(), 100_000) {
        Ok(doc) => assert!(notes(&doc).contains("prototype"), "{}", notes(&doc)),
        Err(e) => assert!(e.to_string().contains("triangles"), "{e}"),
    }
    assert!(start.elapsed().as_secs() < 60);
    // A default that holds an instance of its own prototype.
    let text = "#VRML V2.0 utf8\nPROTO C [ field MFNode c [ C { } ] ] { Group { children IS c } }\nC { }\n";
    let _ = read(text, "a.wrl");
    // An external prototype from the file that declares it.
    let assets = Memory::new().with("a.wrl", "#VRML V2.0 utf8\nEXTERNPROTO D [ ] \"a.wrl#D\"\nD { }\n");
    let _ = read_in(
        b"#VRML V2.0 utf8\nEXTERNPROTO D [ ] \"a.wrl#D\"\nD { }\n",
        "a.wrl",
        &assets,
        usize::MAX,
    )
    .unwrap();
}

#[test]
fn extrusion_default_and_caps() {
    let doc = read("#VRML V2.0 utf8\nShape { geometry Extrusion { } }\n", "a.wrl");
    // The default: a square tube from Y = 0 to 1, capped.
    assert_eq!(triangles(&doc), 4 * 2 + 2 * 2);
    let m = mesh(&doc);
    assert_eq!(bounds(&m), ([-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]));
    assert!(outward(&m, [0.0, 0.5, 0.0]));
    // ccw FALSE turns every face; no caps.
    let doc = read(
        "#VRML V2.0 utf8\nShape { geometry Extrusion { ccw FALSE beginCap FALSE endCap FALSE } }\n",
        "a.wrl",
    );
    assert_eq!(triangles(&doc), 8);
    let m = mesh(&doc);
    let inward = m.prims[0].indices.as_chunks::<3>().0.iter().all(|t| {
        let [a, b, c] = t.map(|i| m.prims[0].positions[i as usize]);
        let n = crate::mesh::face_normal(a, b, c);
        let mid = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0);
        n[0] * mid[0] + n[2] * mid[2] < 0.0
    });
    assert!(inward);
    // Textured: around the section and along the spine.
    let doc = read(
        "#VRML V2.0 utf8\nShape { appearance Appearance { texture ImageTexture { url \"t.png\" } } geometry Extrusion { } }\n",
        "a.wrl",
    );
    let m = mesh(&doc);
    assert_eq!(m.prims[0].uvs.len(), m.prims[0].positions.len());
}

#[test]
fn extrusion_follows_its_spine() {
    // A straight spine along X: +Y turned onto it.
    let doc = read(
        "#VRML V2.0 utf8\nShape { geometry Extrusion { spine [ 0 0 0, 3 0 0 ] crossSection [ 1 1, 1 -1, -1 -1, -1 1, 1 1 ] } }\n",
        "a.wrl",
    );
    let m = mesh(&doc);
    let (lo, hi) = bounds(&m);
    assert!(
        near(lo, [0.0, -1.0, -1.0]) && near(hi, [3.0, 1.0, 1.0]),
        "{lo:?} {hi:?}"
    );
    assert!(outward(&m, [1.5, 0.0, 0.0]));
    // Straight down -Y, and a bend: up, then along +X.
    let doc = read(
        "#VRML V2.0 utf8\nShape { geometry Extrusion { spine [ 0 0 0, 0 -2 0 ] scale [ 0.5 0.5 ] } }\n",
        "a.wrl",
    );
    let (lo, hi) = bounds(&mesh(&doc));
    assert!(
        near(lo, [-0.5, -2.0, -0.5]) && near(hi, [0.5, 0.0, 0.5]),
        "{lo:?} {hi:?}"
    );
    let doc = read(
        "#VRML V2.0 utf8\nShape { geometry Extrusion { spine [ 0 0 0, 0 2 0, 2 2 0 ] scale [ 0.5 0.5 ] } }\n",
        "a.wrl",
    );
    let m = mesh(&doc);
    let (lo, hi) = bounds(&m);
    assert!(lo[0] > -0.6 && lo[1] < 0.01 && hi[0] > 1.99 && hi[1] < 2.6 && near([lo[2], hi[2], 0.0], [-0.5, 0.5, 0.0]));
    // A spine with coincident points, a scale step there, and an orientation per point.
    let doc = read(
        "#VRML V2.0 utf8\nShape { geometry Extrusion { spine [ 0 0 0, 0 1 0, 0 1 0, 0 2 0 ] \
         scale [ 1 1, 1 1, 2 2, 2 2 ] orientation [ 0 1 0 0, 0 1 0 0, 0 1 0 0.785398 ] } }\n",
        "a.wrl",
    );
    let (lo, hi) = bounds(&mesh(&doc));
    assert!(
        (hi[0] - 2.0 * 2f32.sqrt()).abs() < 1e-3 && (lo[1], hi[1]) == (0.0, 2.0),
        "{lo:?} {hi:?}"
    );
}

#[test]
fn extrusion_closed_spine_and_section() {
    // A torus: a closed spine around Y, a closed section, no caps; smooth by its crease angle.
    let mut spine = String::new();
    let mut section = String::new();
    for k in 0..=16 {
        let a = std::f64::consts::TAU * f64::from(k % 16) / 16.0;
        let _ = write!(spine, "{} 0 {}, ", 3.0 * a.cos(), -3.0 * a.sin());
        let _ = write!(section, "{} {}, ", 0.5 * a.cos(), 0.5 * a.sin());
    }
    let text = format!(
        "#VRML V2.0 utf8\nShape {{ geometry Extrusion {{ creaseAngle 1 spine [ {spine} ] crossSection [ {section} ] }} }}\n"
    );
    let doc = read(&text, "a.wrl");
    assert_eq!(triangles(&doc), 16 * 16 * 2);
    let m = mesh(&doc);
    let (lo, hi) = bounds(&m);
    assert!(
        (hi[0] - 3.5).abs() < 1e-4 && (hi[1] - 0.5).abs() < 0.01 && (lo[1] + 0.5).abs() < 0.01,
        "{lo:?} {hi:?}"
    );
    // The seams are shared: one vertex per point of the surface.
    assert_eq!(m.prims[0].positions.len(), 16 * 16);
    assert_eq!(m.prims[0].normals.len(), m.prims[0].positions.len());
}

#[test]
fn extrusion_respects_the_triangle_limit() {
    let mut spine = String::new();
    let mut section = String::new();
    for i in 0..2000 {
        let _ = write!(spine, "0 {i} 0 ");
        let _ = write!(section, "{} {} ", f64::from(i).cos(), f64::from(i).sin());
    }
    let text =
        format!("#VRML V2.0 utf8\nShape {{ geometry Extrusion {{ spine [ {spine} ] crossSection [ {section} ] }} }}\n");
    let err = read_in(text.as_bytes(), "a.wrl", &Memory::new(), 1_000_000).unwrap_err();
    assert!(err.to_string().contains("triangles"), "{err}");
    // Too few points draw nothing.
    let doc = read(
        "#VRML V2.0 utf8\nShape { geometry Extrusion { spine [ 0 0 0 ] } }\n",
        "a.wrl",
    );
    assert_eq!(triangles(&doc), 0);
}

#[test]
fn nodes_2d() {
    let text = r#"#VRML V2.0 utf8
Shape { geometry Rectangle2D { size 4 2 } }
Shape { geometry Disk2D { outerRadius 2 } }
Shape { geometry Disk2D { innerRadius 1 outerRadius 2 } }
Shape { geometry ArcClose2D { } }
Shape { geometry ArcClose2D { closureType "CHORD" startAngle 1 endAngle 1 } }
Shape { geometry TriangleSet2D { vertices [ 0 0, 1 0, 0 1, 5 5, 6 5, 5 6, 1e999 0 ] solid TRUE } }
Shape { geometry Disk2D { innerRadius 3 outerRadius 3 } }
Shape { geometry Circle2D { radius 4 } }
Shape { geometry Arc2D { startAngle 0 endAngle -1.5708 } }
Shape { geometry Polyline2D { lineSegments [ 0 0, 1 1, 2 0 ] } }
Shape { geometry Polypoint2D { point [ 0 0, 1 1, 2 0 ] } }
"#;
    let doc = read(text, "a.wrl");
    let m = mesh(&doc);
    // Rectangle, disk, ring, pie of a quarter turn, whole chord, the triangles.
    assert_eq!(m.triangles(), 2 + 32 + 64 + 8 + 30 + 2);
    // Circles, the three-quarter arc and the polyline.
    assert_eq!(m.lines(), 32 + 32 + 24 + 2);
    assert_eq!(m.points(), 3);
    let (lo, hi) = bounds(&m);
    assert_eq!((lo, hi), ([-4.0, -4.0, 0.0], [6.0, 6.0, 0.0]));
    // Filled ones are seen from both sides unless solid; lines and points are unlit.
    let materials = doc.json()["materials"].as_array().unwrap();
    assert!(materials.iter().any(|m| m["doubleSided"] == true));
    assert!(
        materials
            .iter()
            .any(|m| m.get("doubleSided").is_none() && m["extensions"].is_null())
    );
    assert!(
        m.prims
            .iter()
            .filter(|p| !p.indices.is_empty())
            .all(|p| p.normals.iter().all(|&n| near(n, [0.0, 0.0, 1.0])))
    );
    assert!(doc.json()["asset"]["extras"]["source"].get("skipped").is_none());
}

#[test]
fn what_the_file_says_of_itself() {
    let text = r#"#X3D V3.3 utf8
PROFILE Interchange
UNIT length centimeters 0.01
META "creator" "Ann"
META "generator" "Modeler 4"
META "copyright" "(c) Ann"
META "keywords" "room, chair"
META "reference" "https://example.com"
WorldInfo { title "Room" info [ "first", "second" ] }
Shape { geometry Box {} }
"#;
    let doc = read(text, "a.x3dv");
    let info = doc.source_info();
    assert_eq!(info.format.as_deref(), Some("X3D"));
    assert_eq!(info.version.as_deref(), Some("3.3"));
    assert_eq!(info.meters_per_unit, Fact::Declared(0.01));
    assert_eq!(info.unit.as_deref(), Some("centimeters"));
    assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
    assert!(info.converted.units && !info.converted.up_axis);
    assert_eq!(info.output_meters_per_unit(), Fact::Declared(1.0));
    assert_eq!(info.title.as_deref(), Some("Room"));
    assert_eq!(info.author.as_deref(), Some("Ann"));
    assert_eq!(info.tool.as_deref(), Some("Modeler 4"));
    assert_eq!(info.copyright.as_deref(), Some("(c) Ann"));
    assert_eq!(info.keywords, ["room", "chair"]);
    assert_eq!(info.comments, ["first", "second"]);
    assert_eq!(
        doc.json()["asset"]["extras"]["source"]["meta"]["reference"],
        "https://example.com"
    );

    let doc = read(CUBE_X3D, "cube.x3d");
    let info = doc.source_info();
    assert_eq!(info.title.as_deref(), Some("cube"));
    assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
    assert_eq!(info.unit.as_deref(), Some("meter"));
    assert_eq!(info.converted, Converted::default());
}

#[test]
fn references_are_decoded_once() {
    let xml = r#"<X3D><Scene><Transform DEF="A&amp;lt;B"><Shape><Box/></Shape></Transform></Scene></X3D>"#;
    let doc = read(xml, "a.x3d");
    let json = doc.json();
    let names: Vec<&str> = json["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n["name"].as_str())
        .collect();
    assert!(names.contains(&"A&lt;B"), "{names:?}");
}
