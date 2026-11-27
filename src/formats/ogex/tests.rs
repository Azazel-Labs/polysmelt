use super::*;
use crate::assets::Memory;
use crate::document::Pose;
use crate::formats::Input;
use crate::mesh::Mesh;
use crate::num::u32_of;
use crate::test_rng::Pcg32;

/// A 1x1 RGBA PNG.
const PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49,
    0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99, 0x3D,
    0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

fn read(text: &str) -> Result<Document> {
    load(Input::memory(text.as_bytes(), "t.ogex"), &Memory::new(), Keep::ALL)
}

fn read_with(text: &str, assets: &Memory) -> Result<Document> {
    load(Input::memory(text.as_bytes(), "t.ogex"), assets, Keep::ALL)
}

fn flat(doc: &Document) -> Mesh {
    doc.flatten(Pose::Rest).unwrap().unwrap()
}

fn at(doc: &Document, time: f32) -> Mesh {
    doc.flatten(Pose::Animation { index: 0, time }).unwrap().unwrap()
}

fn close(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
}

fn source(doc: &Document) -> &Value {
    &doc.json()["asset"]["extras"]["source"]
}

/// A geometry object of one triangle, its vertices `positions`, with whatever `more` adds
/// inside the mesh.
fn triangle(name: &str, positions: &str, more: &str) -> String {
    format!(
        "GeometryObject ${name} {{ Mesh {{
            VertexArray (attrib = \"position\") {{ float[3] {{ {positions} }} }}
            IndexArray {{ unsigned_int16[3] {{ {{0, 1, 2}} }} }}
            {more}
        }} }}"
    )
}

const TRI: &str = "{0, 0, 0}, {1, 0, 0}, {0, 1, 0}";

#[test]
fn a_cube_in_centimeters_and_z_up() {
    let text = r#"
        // A comment, and /* another */ one.
        Metric (key = "distance") { float { 0.01 } }
        Metric (key = "angle") { float { 0.017453292519943295 } }
        Metric (key = "up") { string { "z" } }
        GeometryNode $node1 {
            Name { string { "Box" } }
            ObjectRef { ref { $geometry1 } }
            MaterialRef { ref { $material1 } }
            Translation (kind = "xyz") { float[3] { {100, 0, 0} } }
            Rotation (kind = "z") { float { 90 } }
        }
        GeometryObject $geometry1 {
            Mesh (primitive = "quads") {
                VertexArray (attrib = "position") {
                    float[3] { {0, 0, 0}, {100, 0, 0}, {100, 100, 0}, {0, 100, 0} }
                }
                VertexArray (attrib = "normal") {
                    float[3] { {0, 0, 2}, {0, 0, 2}, {0, 0, 2}, {0, 0, 2} }
                }
                VertexArray (attrib = "texcoord") {
                    float[2] { {0, 0}, {1, 0}, {1, 1}, {0, 1} }
                }
                VertexArray (attrib = "color") {
                    float[3] { {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0} }
                }
                IndexArray { unsigned_int32[4] { {0, 1, 2, 3} } }
            }
        }
        Material $material1 {
            Name { string { "Red" } }
            Color (attrib = "diffuse") { float[3] { {1, 0, 0} } }
        }
    "#;
    let doc = read(text).unwrap();
    assert_eq!(doc.facts["triangles"], json!(2));
    let src = source(&doc);
    assert_eq!(src["format"], "OpenGEX");
    assert_eq!(src["upAxis"], "+Z");
    assert!((src["metersPerUnit"].as_f64().unwrap() - 0.01).abs() < 1e-12);
    let info = doc.source_info();
    assert_eq!(info.meters_per_unit, Fact::Declared(0.01));
    assert_eq!(info.unit.as_deref(), Some("centimeter"));
    assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
    assert_eq!(info.forward_axis, Fact::Unknown);
    assert_eq!(info.handedness, Fact::Declared(Handedness::Right));
    assert!(info.converted.units && info.converted.up_axis && !info.converted.handedness);
    let m = flat(&doc);
    assert_eq!(m.triangles(), 2);
    let p = &m.prims[0];
    // Rotated a quarter turn about z, moved 1 m along x, then z-up made y-up.
    let corners: Vec<[f32; 3]> = p.indices.iter().map(|&i| p.positions[i as usize]).collect();
    assert!(corners.iter().any(|&c| close(c, [1.0, 0.0, 0.0])));
    assert!(corners.iter().any(|&c| close(c, [1.0, 0.0, -1.0])));
    assert!(corners.iter().any(|&c| close(c, [0.0, 0.0, -1.0])));
    // Normals are made unit length, and point up.
    assert!(p.normals.iter().all(|&n| close(n, [0.0, 1.0, 0.0])));
    // Texture coordinates flip to glTF's top-left origin.
    assert!(p.uvs.iter().any(|&[u, v]| u.abs() < 1e-6 && (v - 1.0).abs() < 1e-6));
    assert_eq!(p.colors[0], [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(m.materials[p.material].name, "Red");
    assert_eq!(
        doc.json()["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| n["name"] == "Box")
            .count(),
        1
    );
    assert!(doc.to_glb().is_ok());
}

#[test]
fn strips_fans_lines_points_and_clockwise_faces() {
    let quad = "{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}";
    let text = format!(
        r#"
        Metric (key = "up") {{ string {{ "y" }} }}
        GeometryNode {{ ObjectRef {{ ref {{ $s }} }} }}
        GeometryNode {{ ObjectRef {{ ref {{ $c }} }} }}
        GeometryNode {{ ObjectRef {{ ref {{ $l }} }} }}
        GeometryObject $s {{ Mesh (primitive = "triangle_strip") {{
            VertexArray (attrib = "position") {{ float[3] {{ {quad} }} }}
        }} }}
        GeometryObject $c {{ Mesh (primitive = "triangles") {{
            VertexArray (attrib = "position") {{ float[3] {{ {quad} }} }}
            IndexArray (front = "cw") {{ unsigned_int8[3] {{ {{0, 2, 1}} }} }}
        }} }}
        GeometryObject $l {{ Mesh (primitive = "line_strip") {{
            VertexArray (attrib = "position") {{ float[3] {{ {quad} }} }}
        }} }}
        "#
    );
    let doc = read(&text).unwrap();
    let m = flat(&doc);
    assert_eq!(m.triangles(), 3);
    assert_eq!(m.lines(), 3);
    // Every triangle faces +z: the strip's alternate triangles and the clockwise one turned.
    for p in &m.prims {
        for t in p.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
            let z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(z > 0.0, "{t:?}");
        }
    }
    let info = doc.source_info();
    assert_eq!(info.up_axis, Fact::Declared(Axis::PosY));
    // No distance metric: a meter, by the format's default.
    assert_eq!(info.meters_per_unit, Fact::Assumed(1.0));
    assert_eq!(info.converted, Converted::default());
}

#[test]
fn the_metrics_say_what_is_assumed() {
    let doc = read(&format!(
        r#"Metric (key = "forward") {{ string {{ "-y" }} }}
        GeometryNode {{ ObjectRef {{ ref {{ $g }} }} }}
        {}"#,
        triangle("g", TRI, "")
    ))
    .unwrap();
    let info = doc.source_info();
    assert_eq!(info.format.as_deref(), Some("OpenGEX"));
    assert_eq!(info.meters_per_unit, Fact::Assumed(1.0));
    assert_eq!(info.unit.as_deref(), Some("meter"));
    assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
    assert_eq!(info.forward_axis, Fact::Declared(Axis::NegY));
    assert!(info.converted.up_axis && !info.converted.units);
}

#[test]
fn materials_and_textures() {
    let text = format!(
        r#"
        GeometryNode {{ ObjectRef {{ ref {{ $g }} }} MaterialRef (index = 0) {{ ref {{ $m }} }} }}
        {}
        Material $m {{
            Name {{ string {{ "Painted" }} }}
            Color (attrib = "diffuse") {{ float[4] {{ {{0.5, 0.5, 0.5, 1}} }} }}
            Color (attrib = "specular") {{ float[3] {{ {{1, 1, 1}} }} }}
            Param (attrib = "specular_power") {{ float {{ 98 }} }}
            Color (attrib = "emission") {{ float[3] {{ {{0, 0.5, 0}} }} }}
            Param (attrib = "opacity") {{ float {{ 0.5 }} }}
            Texture (attrib = "diffuse", texcoord = 1) {{
                string {{ "paint.png" }}
                Translation {{ float[3] {{ {{0.5, 0.25, 0}} }} }}
                Scale {{ float[3] {{ {{2, 2, 1}} }} }}
            }}
            Texture (attrib = "normal") {{ string {{ "bumps.png" }} }}
            Texture (attrib = "specular") {{ string {{ "missing.png" }} }}
        }}
        "#,
        triangle(
            "g",
            TRI,
            r#"VertexArray (attrib = "texcoord[1]") { float[2] { {0, 0}, {1, 0}, {0, 1} } }"#
        )
    );
    let assets = Memory::new()
        .with("paint.png", PNG.to_vec())
        .with("bumps.png", PNG.to_vec());
    let doc = read_with(&text, &assets).unwrap();
    let j = &doc.json();
    let mat = &j["materials"][0];
    assert_eq!(mat["name"], "Painted");
    let pbr = &mat["pbrMetallicRoughness"];
    assert_eq!(pbr["baseColorFactor"][3], json!(0.5));
    assert_eq!(mat["alphaMode"], "BLEND");
    let base = &pbr["baseColorTexture"];
    assert_eq!(base["texCoord"], json!(1));
    let tt = &base["extensions"]["KHR_texture_transform"];
    assert_eq!(tt["scale"], json!([2.0, 2.0]), "{base}");
    // u' = 2u + 0.5 and v' = 2v + 0.25, seen through glTF's flipped v.
    assert_eq!(tt["offset"], json!([0.5, -1.25]));
    assert!(mat["normalTexture"].is_object());
    // A high specular power is a smooth surface.
    assert!(pbr["roughnessFactor"].as_f64().unwrap() < 0.5);
    assert!(mat["emissiveFactor"][1].as_f64().unwrap() > 0.4);
    assert_eq!(source(&doc)["droppedTextures"], json!(1));
    // The second texture coordinate set is there for the texture, and the first padded.
    let prim = &j["meshes"][0]["primitives"][0];
    assert!(prim["attributes"]["TEXCOORD_1"].is_number());
    assert!(prim["attributes"]["TEXCOORD_0"].is_number());
    assert!(doc.to_glb().is_ok());
}

#[test]
fn shared_objects_are_instances() {
    let text = format!(
        r"
        GeometryNode $a {{ ObjectRef {{ ref {{ $g }} }} Translation {{ float[3] {{ {{5, 0, 0}} }} }}
            GeometryNode $b {{ ObjectRef {{ ref {{ $g }} }} Translation {{ float[3] {{ {{5, 0, 0}} }} }} }}
        }}
        {}
        ",
        triangle("g", TRI, "")
    );
    let doc = read(&text).unwrap();
    assert_eq!(doc.json()["meshes"].as_array().unwrap().len(), 1);
    let m = flat(&doc);
    assert_eq!(m.triangles(), 2);
    let xs: Vec<f32> = m.prims.iter().flat_map(|p| p.positions.iter().map(|q| q[0])).collect();
    assert!(xs.iter().any(|&x| (x - 10.0).abs() < 1e-4));
}

#[test]
fn morph_targets_and_weights() {
    let text = r#"
        Metric (key = "up") { string { "y" } }
        GeometryNode {
            ObjectRef { ref { $g } }
            MorphWeight %w0 (index = 0) { float { 1 } }
            MorphWeight %w1 (index = 1) { float { 0.25 } }
            Animation {
                Track (target = %w1) {
                    Time { Key { float { 0, 1 } } }
                    Value { Key { float { 0, 1 } } }
                }
            }
        }
        GeometryObject $g {
            Morph (index = 1) { Name { string { "lift" } } }
            Mesh {
                VertexArray (attrib = "position") { float[3] { {0, 0, 0}, {1, 0, 0}, {0, 1, 0} } }
                VertexArray (attrib = "position", morph = 1) { float[3] { {0, 0, 2}, {1, 0, 2}, {0, 1, 2} } }
                IndexArray { unsigned_int16[3] { {0, 1, 2} } }
            }
        }
    "#;
    let doc = read(text).unwrap();
    let j = &doc.json();
    assert_eq!(j["meshes"][0]["extras"]["targetNames"], json!(["lift"]));
    assert_eq!(j["meshes"][0]["primitives"][0]["targets"].as_array().unwrap().len(), 1);
    let m = flat(&doc);
    assert!((m.prims[0].positions[0][2] - 0.5).abs() < 1e-5);
    // The animated weight, halfway.
    let m = at(&doc, 0.5);
    assert!((m.prims[0].positions[0][2] - 1.0).abs() < 1e-4);
    let a = &j["animations"][0];
    assert_eq!(a["channels"][0]["target"]["path"], "weights");
}

#[test]
fn skins_bind_through_the_skeleton() {
    let text = r#"
        Metric (key = "up") { string { "y" } }
        BoneNode $bone { Translation { float[3] { {1, 0, 0} } } }
        GeometryNode {
            ObjectRef { ref { $g } }
            Translation { float[3] { {50, 50, 50} } }
        }
        GeometryObject $g { Mesh {
            VertexArray (attrib = "position") { float[3] { {0, 0, 0}, {1, 0, 0}, {0, 1, 0} } }
            IndexArray { unsigned_int16[3] { {0, 1, 2} } }
            Skin {
                Transform { float[16] { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1} } }
                Skeleton {
                    BoneRefArray { ref { $bone } }
                    Transform { float[16] { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1} } }
                }
                BoneCountArray { unsigned_int16 { 1, 1, 1 } }
                BoneIndexArray { unsigned_int16 { 0, 0, 0 } }
                BoneWeightArray { float { 1, 1, 1 } }
            }
        } }
    "#;
    let doc = read(text).unwrap();
    let j = &doc.json();
    assert_eq!(j["skins"].as_array().unwrap().len(), 1);
    let m = flat(&doc);
    // v' = bone * inverse(bind) * skin * v: moved 2 up by the skin and 1 along x by the bone;
    // the geometry node's own transform doesn't apply.
    assert!(
        close(m.prims[0].positions[0], [1.0, 2.0, 0.0]),
        "{:?}",
        m.prims[0].positions
    );
    assert!(doc.to_glb().is_ok());
}

/// A node holding a triangle whose `transform` structures are animated by `animation`.
fn animated(transforms: &str, animation: &str) -> String {
    format!(
        r#"
        Metric (key = "up") {{ string {{ "y" }} }}
        GeometryNode {{
            ObjectRef {{ ref {{ $g }} }}
            {transforms}
            Animation {{ {animation} }}
        }}
        {}
        "#,
        triangle("g", "{1, 0, 0}, {1, 1, 0}, {1, 0, 1}", "")
    )
}

fn sampler<'d>(doc: &'d Document, path: &str) -> &'d Value {
    let a = &doc.json()["animations"][0];
    let c = a["channels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["target"]["path"] == path)
        .unwrap();
    &a["samplers"][usize::try_from(c["sampler"].as_u64().unwrap()).unwrap()]
}

#[test]
fn bezier_tracks_become_cubic_splines() {
    // Time controls a third of the way along keep time linear.
    let text = animated(
        r#"Translation %t (kind = "x") { float { 0 } }"#,
        r#"Track (target = %t) {
            Time (curve = "bezier") {
                Key { float { 0, 3 } }
                Key (kind = "-control") { float { 0, 2 } }
                Key (kind = "+control") { float { 1, 3 } }
            }
            Value (curve = "bezier") {
                Key { float { 0, 4 } }
                Key (kind = "-control") { float { 0, 5 } }
                Key (kind = "+control") { float { 2, 4 } }
            }
        }"#,
    );
    let doc = read(&text).unwrap();
    assert_eq!(sampler(&doc, "translation")["interpolation"], "CUBICSPLINE");
    for time in [0.0f32, 0.7, 1.5, 2.2, 3.0] {
        let s = f64::from(time) / 3.0;
        let u = 1.0 - s;
        let want = 3.0 * s * u * u * 2.0 + 3.0 * s * s * u * 5.0 + s * s * s * 4.0;
        let got = at(&doc, time).prims[0].positions[0][0] - 1.0;
        #[allow(clippy::cast_possible_truncation, reason = "a test value")]
        let want = want as f32;
        assert!((got - want).abs() < 1e-4, "{time}: {got} {want}");
    }
}

#[test]
fn bezier_time_is_sampled_when_it_bends() {
    let text = animated(
        r#"Translation %t (kind = "x") { float { 0 } }"#,
        r#"Track (target = %t) {
            Time (curve = "bezier") {
                Key { float { 0, 3 } }
                Key (kind = "-control") { float { 0, 2.9 } }
                Key (kind = "+control") { float { 2.5, 3 } }
            }
            Value { Key { float { 0, 3 } } }
        }"#,
    );
    let doc = read(&text).unwrap();
    assert_eq!(source(&doc)["sampledTracks"], json!(1));
    // Halfway in time is well past halfway along the curve.
    let got = at(&doc, 1.5).prims[0].positions[0][0] - 1.0;
    let s = super::anim::solve(0.0, 2.5, 2.9, 3.0, 1.5);
    #[allow(clippy::cast_possible_truncation, reason = "a test value")]
    let want = (3.0 * s) as f32;
    assert!((got - want).abs() < 0.02, "{got} {want}");
}

#[test]
fn tcb_tracks_follow_their_tangents() {
    let text = animated(
        r#"Translation %t (kind = "x") { float { 0 } }"#,
        r#"Track (target = %t) {
            Time { Key { float { 0, 1, 2 } } }
            Value (curve = "tcb") {
                Key { float { 0, 1, 4 } }
                Key (kind = "tension") { float { 0, 0, 0 } }
                Key (kind = "continuity") { float { 0, 0, 0 } }
                Key (kind = "bias") { float { 0, 0, 0 } }
            }
        }"#,
    );
    let doc = read(&text).unwrap();
    assert_eq!(sampler(&doc, "translation")["interpolation"], "CUBICSPLINE");
    // Catmull-Rom with the ends' missing neighbors as the keys themselves.
    let got = at(&doc, 0.5).prims[0].positions[0][0] - 1.0;
    assert!((got - 0.3125).abs() < 1e-4, "{got}");
    let got = at(&doc, 1.0).prims[0].positions[0][0] - 1.0;
    assert!((got - 1.0).abs() < 1e-5);
}

#[test]
fn rotations_turn_the_long_way_when_asked() {
    let text = animated(
        r#"Rotation %r (kind = "z") { float { 0 } }"#,
        r"Track (target = %r) {
            Time { Key { float { 0, 1 } } }
            Value { Key { float { 0, 6.283185307179586 } } }
        }",
    );
    let doc = read(&text).unwrap();
    assert_eq!(sampler(&doc, "rotation")["interpolation"], "LINEAR");
    assert!(close(at(&doc, 0.25).prims[0].positions[0], [0.0, 1.0, 0.0]));
    assert!(close(at(&doc, 0.5).prims[0].positions[0], [-1.0, 0.0, 0.0]));
    assert!(close(at(&doc, 0.75).prims[0].positions[0], [0.0, -1.0, 0.0]));
}

#[test]
fn stepped_tracks_and_split_transforms() {
    let text = animated(
        r#"Translation (kind = "y") { float { 1 } }
           Translation %t (kind = "x") { float { 0 } }
           Rotation (kind = "x") { float { 0 } }
           Scale %s { float[3] { {1, 1, 1} } }"#,
        r#"Track (target = %t) {
            Time { Key { float { 0, 1 } } }
            Value (curve = "constant") { Key { float { 0, 5 } } }
        }
        Track (target = %s) {
            Time { Key { float { 0, 1 } } }
            Value { Key { float[3] { {1, 1, 1}, {2, 2, 2} } } }
        }"#,
    );
    let doc = read(&text).unwrap();
    assert_eq!(sampler(&doc, "translation")["interpolation"], "STEP");
    assert_eq!(sampler(&doc, "scale")["interpolation"], "LINEAR");
    assert!(close(at(&doc, 0.5).prims[0].positions[0], [1.5, 1.0, 0.0]));
    assert!(close(at(&doc, 1.0).prims[0].positions[0], [7.0, 1.0, 0.0]));
}

#[test]
fn animated_matrices_are_taken_apart() {
    let text = animated(
        r"Transform %m { float[16] { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1} } }",
        r"Track (target = %m) {
            Time { Key { float { 0, 2 } } }
            Value { Key { float[16] {
                {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
                {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 1}
            } } }
        }",
    );
    let doc = read(&text).unwrap();
    assert!(close(at(&doc, 1.0).prims[0].positions[0], [3.0, 0.0, 0.0]));
}

#[test]
fn clips_are_separate_animations() {
    let text = format!(
        r#"
        Metric (key = "up") {{ string {{ "y" }} }}
        Metric (key = "time") {{ float {{ 0.5 }} }}
        Clip (index = 1) {{ Name {{ string {{ "walk" }} }} Param (attrib = "rate") {{ float {{ 30 }} }} }}
        GeometryNode {{
            ObjectRef {{ ref {{ $g }} }}
            Translation %t (kind = "x") {{ float {{ 0 }} }}
            Animation (clip = 0) {{ Track (target = %t) {{
                Time {{ Key {{ float {{ 0, 2 }} }} }} Value {{ Key {{ float {{ 0, 1 }} }} }}
            }} }}
            Animation (clip = 1, begin = 0, end = 4) {{ Track (target = %t) {{
                Time {{ Key {{ float {{ 0, 4 }} }} }} Value {{ Key {{ float {{ 0, 2 }} }} }}
            }} }}
        }}
        {}
        "#,
        triangle("g", TRI, "")
    );
    let doc = read(&text).unwrap();
    let a = doc.json()["animations"].as_array().unwrap();
    assert_eq!(a.len(), 2);
    assert_eq!(a[1]["name"], "walk");
    assert_eq!(a[1]["extras"]["rate"], json!(60.0));
    assert_eq!(a[1]["extras"]["end"], json!(2.0));
    // Time is in half seconds.
    let m = doc.flatten(Pose::Animation { index: 1, time: 1.0 }).unwrap().unwrap();
    assert!((m.prims[0].positions[0][0] - 1.0).abs() < 1e-5);
}

#[test]
fn lights_and_cameras() {
    let text = format!(
        r#"
        GeometryNode {{ ObjectRef {{ ref {{ $g }} }} }}
        LightNode {{ ObjectRef {{ ref {{ $sun }} }} }}
        LightNode {{ ObjectRef {{ ref {{ $spot }} }} }}
        CameraNode {{ ObjectRef {{ ref {{ $cam }} }} }}
        LightObject $sun (type = "infinite") {{
            Color (attrib = "light") {{ float[3] {{ {{2, 1, 1}} }} }}
            Param (attrib = "intensity") {{ float {{ 3 }} }}
        }}
        LightObject $spot (type = "spot") {{
            Atten (kind = "angle") {{ Param (attrib = "begin") {{ float {{ 0.2 }} }} Param (attrib = "end") {{ float {{ 0.4 }} }} }}
        }}
        CameraObject $cam {{
            Param (attrib = "fov") {{ float {{ 0.97 }} }}
            Param (attrib = "near") {{ float {{ 1.5 }} }}
            Param (attrib = "far") {{ float {{ 150 }} }}
        }}
        {}
        "#,
        triangle("g", TRI, "")
    );
    let doc = read(&text).unwrap();
    let j = &doc.json();
    let lights = j["extensions"]["KHR_lights_punctual"]["lights"].as_array().unwrap();
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0]["type"], "directional");
    // A color brighter than white folds into the intensity.
    assert!(lights[0]["intensity"].as_f64().unwrap() > 3.0);
    assert_eq!(lights[1]["type"], "spot");
    let cam = &j["cameras"][0]["perspective"];
    // "fov" is the angle across: 0.97 across a 16:9 frame is 2 atan(tan(0.485) 9 / 16) up and down.
    let (yfov, aspect) = (cam["yfov"].as_f64().unwrap(), cam["aspectRatio"].as_f64().unwrap());
    assert!((aspect - 16.0 / 9.0).abs() < 1e-12);
    assert!((yfov - 0.576_352_569).abs() < 1e-6, "{yfov}");
    assert!((2.0 * ((yfov * 0.5).tan() * aspect).atan() - 0.97).abs() < 1e-6);
    assert_eq!(cam["znear"], json!(1.5));
    // A file of objects and no nodes places each under the root; one with nothing in it, or only an empty
    // geometry object, has no geometry.
    let lone = read("CameraObject $c { }").unwrap();
    assert_eq!(lone.json()["cameras"].as_array().map(Vec::len), Some(1));
    assert_eq!(lone.json()["nodes"][1]["camera"], json!(0));
    assert!(lone.to_glb().is_ok());
    assert!(read("").is_err());
    assert!(read("GeometryObject { Mesh { } }").is_err());
    assert!(doc.to_glb().is_ok());
}

#[test]
fn hostile_files_never_panic() {
    let good = r#"
        Metric (key = "distance") { float { 0.01 } }
        BoneNode $bone { Translation %t { float[3] { {1, 0, 0} } }
            Animation { Track (target = %t) {
                Time (curve = "bezier") { Key { float { 0, 1 } } Key (kind = "-control") { float { 0, 0.5 } }
                    Key (kind = "+control") { float { 0.5, 1 } } }
                Value (curve = "tcb") { Key { float[3] { {0, 0, 0}, {1, 1, 1} } } }
            } }
        }
        GeometryNode { ObjectRef { ref { $g } } MaterialRef { ref { $m } }
            MorphWeight %w (index = 1) { float { 0.5 } } }
        GeometryObject $g { Morph (index = 1) { } Mesh (primitive = "triangle_strip") {
            VertexArray (attrib = "position") { float[3] { {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0} } }
            VertexArray (attrib = "position", morph = 1) { float[3] { {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1} } }
            VertexArray (attrib = "texcoord") { float[2] { {0, 0}, {1, 0}, {0, 1}, {1, 1} } }
            IndexArray (restart = 65535) { unsigned_int16 { 0, 1, 2, 65535, 1, 2, 3 } }
            Skin { Skeleton { BoneRefArray { ref { $bone } }
                Transform { float[16] { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1} } } }
                BoneCountArray { unsigned_int8 { 1, 1, 1, 1 } }
                BoneIndexArray { unsigned_int8 { 0, 0, 0, 0 } }
                BoneWeightArray { float { 1, 1, 1, 1 } } }
        } }
        Material $m { Texture (attrib = "diffuse") { string { "t.png" } Rotation { float { 0.5 } } }
            Color (attrib = "diffuse") { float[3] { {0.5, 0.5, 0.5} } } }
        "#;
    let assets = Memory::new().with("t.png", PNG.to_vec());
    let doc = read_with(good, &assets).unwrap();
    assert_eq!(flat(&doc).triangles(), 2);
    let b = good.as_bytes();
    // Every truncation.
    for n in 0..b.len() {
        let _ = load(Input::memory(&b[..n], "t.ogex"), &assets, Keep::ALL);
    }
    // Random corruption.
    let mut rng = Pcg32::new(11);
    let alphabet = b"{}[]()$%,=\"'0123456789.-+exXfloat \n/*";
    for _ in 0..3000 {
        let mut c = b.to_vec();
        for _ in 0..=rng.below(6) {
            let i = rng.below(u32_of(c.len())) as usize;
            c[i] = alphabet[rng.below(u32_of(alphabet.len())) as usize];
        }
        if let Ok(doc) = load(Input::memory(&c, "t.ogex"), &assets, Keep::ALL) {
            let _ = doc.to_glb();
            let _ = doc.flatten(Pose::Rest);
        }
    }
    // Deep nesting stops at the limit rather than overflowing the stack.
    let deep = "Node {".repeat(100_000);
    assert!(read(&deep).is_err());
    let deep = format!("{}{}", "Node {".repeat(5_000), "}".repeat(5_000));
    let _ = read(&deep);
    // Huge declared array sizes and counts.
    assert!(read("GeometryObject { Mesh { VertexArray { float[4000000000] { } } } }").is_err());
    let _ = read(&format!(
        "GeometryNode {{ ObjectRef {{ ref {{ $g }} }} }} {}",
        triangle(
            "g",
            TRI,
            "Skin { BoneCountArray { unsigned_int32 { 4000000000, 1, 1 } } }"
        )
    ));
    // Garbage, bad numbers, unterminated strings and comments.
    for junk in [
        "\u{0}\u{1}\u{2}",
        "Metric { float { 1e999999 } }",
        "Name { string { \"abc",
        "/* never ends",
        "a {",
        "}",
    ] {
        let _ = read(junk);
    }
}

#[test]
fn a_quad_that_isnt_convex_keeps_its_outline() {
    // An arrowhead, its notch at the second corner: cut from the first corner it would cover three times its
    // area.
    let text = r#"
        GeometryNode $n { ObjectRef { ref { $g } } }
        GeometryObject $g { Mesh (primitive = "quads") {
            VertexArray (attrib = "position") { float[3] { {2, 0, 0}, {0.5, 0.5, 0}, {0, 2, 0}, {0, 0, 0} } }
            IndexArray { unsigned_int16[4] { {0, 1, 2, 3} } }
        } }
    "#;
    let doc = read(text).unwrap();
    let m = flat(&doc);
    assert_eq!(m.triangles(), 2);
    let mut area = 0.0;
    for p in &m.prims {
        for t in p.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
            let (u, v) = ([0, 1, 2].map(|k| b[k] - a[k]), [0, 1, 2].map(|k| c[k] - a[k]));
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            area += 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        }
    }
    assert!((area - 1.0).abs() < 1e-5, "{area}");
    // Without normals in the file, the default ones.
    assert!(
        doc.json()["meshes"][0]["primitives"][0]["attributes"]
            .get("NORMAL")
            .is_some()
    );
}
