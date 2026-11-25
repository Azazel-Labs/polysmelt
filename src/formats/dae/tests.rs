use super::*;
use crate::assets::Memory;
use crate::document::Pose;

fn read_with(text: &[u8], name: &str, assets: &Memory, keep: Keep) -> Result<Document> {
    load(Input::memory(text, name), assets, keep)
}

fn read(text: &str) -> Document {
    read_with(text.as_bytes(), "t.dae", &Memory::new(), Keep::ALL).expect("reads")
}

/// A document around library elements and a scene.
fn doc(head: &str, libraries: &str, scene: &str) -> String {
    format!(
        r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>{head}</asset>
  {libraries}
  <library_visual_scenes><visual_scene id="scene">{scene}</visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>"##
    )
}

const QUAD: &str = r##"<library_geometries><geometry id="quad" name="Quad"><mesh>
  <source id="p"><float_array id="pa" count="12">0 0 0 1 0 0 1 1 0 0 1 0</float_array>
    <technique_common><accessor source="#pa" count="4" stride="3">
      <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
    </accessor></technique_common></source>
  <source id="n"><float_array id="na" count="3">0 0 2</float_array>
    <technique_common><accessor source="#na" count="1" stride="3">
      <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
    </accessor></technique_common></source>
  <source id="t"><float_array id="ta" count="8">0 0 1 0 1 1 0 1</float_array>
    <technique_common><accessor source="#ta" count="4" stride="2">
      <param name="S" type="float"/><param name="T" type="float"/>
    </accessor></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <polylist count="1" material="mat">
    <input semantic="VERTEX" source="#v" offset="0"/>
    <input semantic="NORMAL" source="#n" offset="1"/>
    <input semantic="TEXCOORD" source="#t" offset="2" set="0"/>
    <vcount>4</vcount><p>0 0 0 1 0 1 2 0 2 3 0 3</p>
  </polylist>
</mesh></geometry></library_geometries>"##;

fn png() -> Vec<u8> {
    let mut png = Vec::new();
    image::RgbImage::from_pixel(2, 2, image::Rgb([200, 100, 50]))
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    png
}

fn flat(d: &Document) -> crate::mesh::Mesh {
    d.flatten(Pose::Rest).expect("flattens").expect("draws")
}

#[test]
fn polylist_with_normals_and_texcoords() {
    let d = read(&doc(
        "",
        QUAD,
        r##"<node id="a"><instance_geometry url="#quad"/></node>"##,
    ));
    let m = flat(&d);
    assert_eq!(m.triangles(), 2);
    let p = &m.prims[0];
    assert_eq!(p.positions.len(), 4);
    assert!(
        p.normals.iter().all(|n| (n[2] - 1.0).abs() < 1e-6),
        "normals unit length"
    );
    // V flipped to glTF's top-left origin.
    let k = p.positions.iter().position(|q| q[0] < 0.5 && q[1] > 0.5).unwrap();
    assert!(p.uvs[k][1].abs() < 1e-6);
    assert_eq!(d.json()["meshes"][0]["name"], "Quad");
    assert_eq!(d.json()["asset"]["extras"]["source"]["format"], "COLLADA");
}

#[test]
fn polygon_holes_are_bridged() {
    let lib = r##"<library_geometries><geometry id="g"><mesh>
      <source id="p"><float_array id="pa" count="24">0 0 0 4 0 0 4 4 0 0 4 0 1 1 0 1 3 0 3 3 0 3 1 0</float_array>
        <technique_common><accessor source="#pa" count="8" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common></source>
      <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
      <polygons count="1"><input semantic="VERTEX" source="#v" offset="0"/>
        <ph><p>0 1 2 3</p><h>4 5 6 7</h></ph>
      </polygons>
    </mesh></geometry></library_geometries>"##;
    let d = read(&doc("", lib, r##"<node><instance_geometry url="#g"/></node>"##));
    let m = flat(&d);
    assert_eq!(m.triangles(), 8);
    // The triangles cover the frame's area, 16 - 4, and not the hole.
    let p = &m.prims[0];
    let area: f32 = p
        .indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs() / 2.0
        })
        .sum();
    assert!((area - 12.0).abs() < 1e-4, "area {area}");
}

#[test]
fn z_up_centimeters_become_y_up_meters() {
    let d = read(&doc(
        r#"<unit meter="0.01" name="centimeter"/><up_axis>Z_UP</up_axis>"#,
        QUAD,
        r##"<node><translate>0 0 100</translate><instance_geometry url="#quad"/></node>"##,
    ));
    let m = flat(&d);
    let ys: Vec<f32> = m.prims[0].positions.iter().map(|p| p[1]).collect();
    assert!(ys.iter().all(|y| (y - 1.0).abs() < 1e-5), "{ys:?}");
    let info = d.source_info();
    assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
    assert_eq!(info.meters_per_unit, Fact::Declared(0.01));
    assert_eq!(info.unit.as_deref(), Some("centimeter"));
    assert!(info.converted.units && info.converted.up_axis && !info.converted.handedness);
    assert_eq!(info.output_meters_per_unit(), Fact::Declared(1.0));
    assert_eq!(
        d.json()["nodes"].as_array().unwrap().last().unwrap()["name"],
        "RootNode"
    );
}

#[test]
fn the_asset_says_who_made_it_and_when() {
    let head = r"<contributor><author>Ann &amp; Bo</author><authoring_tool>Modeler 3</authoring_tool>
        <comments>first cut</comments><copyright>(c) 2024 Ann</copyright></contributor>
      <contributor><author>Cy</author><authoring_tool>Fixer 1</authoring_tool><comments>cleaned up</comments></contributor>
      <created>2024-01-02T03:04:05Z</created><modified>2024-02-03T04:05:06Z</modified>
      <keywords>crate wood prop</keywords><revision>7</revision><subject>A crate</subject><title>Crate</title>";
    let d = read(&doc(head, QUAD, r##"<node><instance_geometry url="#quad"/></node>"##));
    let info = d.source_info();
    assert_eq!(info.format.as_deref(), Some("COLLADA"));
    assert_eq!(info.version.as_deref(), Some("1.4.1"));
    assert_eq!(info.tool.as_deref(), Some("Modeler 3; Fixer 1"));
    assert_eq!(info.author.as_deref(), Some("Ann & Bo; Cy"));
    assert_eq!(info.copyright.as_deref(), Some("(c) 2024 Ann"));
    assert_eq!(info.comments, ["first cut", "cleaned up"]);
    assert_eq!(info.title.as_deref(), Some("Crate"));
    assert_eq!(info.description.as_deref(), Some("A crate"));
    assert_eq!(info.keywords, ["crate", "wood", "prop"]);
    assert_eq!(info.created.as_deref(), Some("2024-01-02T03:04:05Z"));
    assert_eq!(info.modified.as_deref(), Some("2024-02-03T04:05:06Z"));
    assert_eq!(d.json()["asset"]["copyright"], "(c) 2024 Ann");
    let source = &d.json()["asset"]["extras"]["source"];
    assert_eq!(source["revision"], "7");
    assert_eq!(source["contributors"][1]["authoring_tool"], "Fixer 1");
    // No unit and no up axis: COLLADA's meter and Y up, assumed, and nothing to convert.
    assert_eq!(info.meters_per_unit, Fact::Assumed(1.0));
    assert_eq!(info.unit.as_deref(), Some("meter"));
    assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
    assert_eq!(info.handedness, Fact::Declared(Handedness::Right));
    assert_eq!(info.converted, crate::source::Converted::default());
}

#[test]
fn instance_node_copies_and_cycles_stop() {
    let scene = r##"<node id="a" name="A"><instance_geometry url="#quad"/><instance_node url="#a"/></node>
        <node id="b"><translate>5 0 0</translate><instance_node url="#a"/></node>"##;
    let d = read(&doc("", QUAD, scene));
    assert_eq!(d.triangles(), 4);
    assert_eq!(d.json()["meshes"].as_array().unwrap().len(), 1);
}

#[test]
fn transforms_compose_in_order() {
    let m = Transform::Rotate.matrix(&[0.0, 0.0, 1.0, 90.0]);
    assert!((m[0][1] + 1.0).abs() < 1e-12 && (m[1][0] - 1.0).abs() < 1e-12);
    let l = Transform::Lookat.matrix(&[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    assert!((l[2][3] - 5.0).abs() < 1e-12 && (l[2][2] - 1.0).abs() < 1e-12);
    let s = Transform::Skew.matrix(&[45.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    // y gains x's value.
    assert!((s[1][0] - 1.0).abs() < 1e-9, "{s:?}");
    assert_eq!(Transform::Skew.matrix(&[45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), IDENTITY);
}

const SKIN: &str = r##"<library_controllers><controller id="c"><skin source="#quad">
  <bind_shape_matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</bind_shape_matrix>
  <source id="j"><Name_array id="ja" count="5">j0 j1 j2 j3 j4</Name_array>
    <technique_common><accessor source="#ja" count="5"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="m"><float_array id="ma" count="80">
    1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1  1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1  1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
    1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1  1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</float_array>
    <technique_common><accessor source="#ma" count="5" stride="16"><param name="TRANSFORM" type="float4x4"/></accessor></technique_common></source>
  <source id="w"><float_array id="wa" count="6">0.1 0.2 0.3 0.4 0.5 1</float_array>
    <technique_common><accessor source="#wa" count="6"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints><input semantic="JOINT" source="#j"/><input semantic="INV_BIND_MATRIX" source="#m"/></joints>
  <vertex_weights count="4"><input semantic="JOINT" source="#j" offset="0"/><input semantic="WEIGHT" source="#w" offset="1"/>
    <vcount>5 1 1 1</vcount><v>0 0 1 1 2 2 3 3 4 4  0 5 1 5 -1 5</v></vertex_weights>
</skin></controller></library_controllers>"##;

const BONES: &str = r#"<node id="root" sid="j0" type="JOINT"><node sid="j1" type="JOINT"><node sid="j2" type="JOINT"/>
    <node sid="j3" type="JOINT"/><node sid="j4" type="JOINT"/></node></node>"#;

#[test]
fn skins_keep_up_to_sixteen_influences() {
    let scene = format!(
        r##"{BONES}<node><instance_controller url="#c"><skeleton>#root</skeleton></instance_controller></node>"##
    );
    let d = read(&doc("", &format!("{QUAD}{SKIN}"), &scene));
    assert_eq!(d.json()["skins"][0]["joints"].as_array().unwrap().len(), 5);
    let attrs = &d.json()["meshes"][0]["primitives"][0]["attributes"];
    let weights = |k: usize| {
        d.read_floats(usize::try_from(attrs[format!("WEIGHTS_{k}")].as_u64().unwrap()).unwrap())
            .unwrap()
            .1
    };
    // Five influences take two sets of four.
    let (w0, w1) = (weights(0), weights(1));
    assert!(attrs.get("WEIGHTS_2").is_none());
    for (a, b) in w0.chunks_exact(4).zip(w1.chunks_exact(4)) {
        assert!((a.iter().chain(b).sum::<f32>() - 1.0).abs() < 1e-5, "{a:?} {b:?}");
    }
    // The first vertex keeps every joint, strongest first, the weakest (0.1 of 1.5) in the second set.
    assert!((w0[0] - 0.5 / 1.5).abs() < 1e-5, "{w0:?}");
    assert!((w1[0] - 0.1 / 1.5).abs() < 1e-5, "{w1:?}");
    assert_eq!(&w1[1..4], &[0.0; 3]);
    let m = flat(&d);
    assert_eq!(m.triangles(), 2);
}

#[test]
fn rotation_animation_is_cut_into_short_turns() {
    let lib = r##"<library_animations><animation id="a">
      <source id="i"><float_array id="ia" count="2">0 1</float_array>
        <technique_common><accessor source="#ia" count="2"><param name="TIME" type="float"/></accessor></technique_common></source>
      <source id="o"><float_array id="oa" count="2">0 180</float_array>
        <technique_common><accessor source="#oa" count="2"><param name="ANGLE" type="float"/></accessor></technique_common></source>
      <source id="k"><Name_array id="ka" count="2">LINEAR LINEAR</Name_array>
        <technique_common><accessor source="#ka" count="2"><param name="INTERPOLATION" type="name"/></accessor></technique_common></source>
      <sampler id="s"><input semantic="INPUT" source="#i"/><input semantic="OUTPUT" source="#o"/><input semantic="INTERPOLATION" source="#k"/></sampler>
      <channel source="#s" target="spin/rz.ANGLE"/>
    </animation></library_animations>"##;
    let scene = r##"<node id="spin"><rotate sid="rz">0 0 1 0</rotate><instance_geometry url="#quad"/></node>"##;
    let d = read(&doc("", &format!("{QUAD}{lib}"), scene));
    let anim = &d.json()["animations"][0];
    assert_eq!(anim["channels"].as_array().unwrap().len(), 1);
    assert_eq!(anim["channels"][0]["target"]["path"], "rotation");
    let input = usize::try_from(anim["samplers"][0]["input"].as_u64().unwrap()).unwrap();
    assert!(d.read_floats(input).unwrap().1.len() >= 18);
    let m = d.flatten(Pose::Animation { index: 0, time: 1.0 }).unwrap().unwrap();
    let p = m.prims[0]
        .positions
        .iter()
        .find(|p| p[0].abs() > 0.5 && p[1].abs() < 0.5)
        .unwrap();
    assert!((p[0] + 1.0).abs() < 1e-4, "{p:?}");
}

#[test]
fn morph_targets_are_deltas() {
    let lib = r##"<library_geometries><geometry id="base"><mesh>
        <source id="bp"><float_array id="bpa" count="9">0 0 0 1 0 0 0 1 0</float_array>
          <technique_common><accessor source="#bpa" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
        <vertices id="bv"><input semantic="POSITION" source="#bp"/></vertices>
        <triangles count="1"><input semantic="VERTEX" source="#bv" offset="0"/><p>0 1 2</p></triangles>
      </mesh></geometry>
      <geometry id="up" name="Up"><mesh>
        <source id="up_p"><float_array id="upa" count="9">0 0 1 1 0 1 0 1 1</float_array>
          <technique_common><accessor source="#upa" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
        <vertices id="uv"><input semantic="POSITION" source="#up_p"/></vertices>
        <triangles count="1"><input semantic="VERTEX" source="#uv" offset="0"/><p>0 1 2</p></triangles>
      </mesh></geometry></library_geometries>
      <library_controllers><controller id="m"><morph source="#base" method="NORMALIZED">
        <source id="mt"><IDREF_array id="mta" count="1">up</IDREF_array>
          <technique_common><accessor source="#mta" count="1"><param name="MORPH_TARGET" type="IDREF"/></accessor></technique_common></source>
        <source id="mw"><float_array id="mwa" count="1">0.5</float_array>
          <technique_common><accessor source="#mwa" count="1"><param name="MORPH_WEIGHT" type="float"/></accessor></technique_common></source>
        <targets><input semantic="MORPH_TARGET" source="#mt"/><input semantic="MORPH_WEIGHT" source="#mw"/></targets>
      </morph></controller></library_controllers>"##;
    let d = read(&doc("", lib, r##"<node><instance_controller url="#m"/></node>"##));
    assert_eq!(d.json()["meshes"][0]["weights"], json!([0.5]));
    assert_eq!(d.json()["meshes"][0]["extras"]["targetNames"], json!(["Up"]));
    let m = flat(&d);
    assert!(m.prims[0].positions.iter().all(|p| (p[2] - 0.5).abs() < 1e-6));
}

#[test]
fn materials_from_common_profile() {
    let lib = r##"<library_images><image id="img"><init_from>tex%20ture.png</init_from></image></library_images>
      <library_effects><effect id="fx"><profile_COMMON>
        <newparam sid="surf"><surface type="2D"><init_from>img</init_from></surface></newparam>
        <newparam sid="samp"><sampler2D><source>surf</source><wrap_s>CLAMP</wrap_s></sampler2D></newparam>
        <technique sid="common"><phong>
          <emission><color>0.5 0 0 1</color></emission>
          <diffuse><texture texture="samp" texcoord="UVSET0"/></diffuse>
          <shininess><float>18</float></shininess>
          <transparent opaque="A_ONE"><color>1 1 1 0.5</color></transparent>
          <transparency><float>1</float></transparency>
        </phong>
        <extra><technique profile="EXPORTER"><bump><texture texture="img" texcoord="UVSET0"/></bump></technique></extra>
        </technique>
      </profile_COMMON>
      <extra><technique profile="MAX3D"><double_sided>1</double_sided></technique></extra></effect></library_effects>
      <library_materials><material id="red" name="Red"><instance_effect url="#fx"/></material></library_materials>"##;
    let scene = r##"<node><instance_geometry url="#quad"><bind_material><technique_common>
        <instance_material symbol="mat" target="#red"><bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/></instance_material>
      </technique_common></bind_material></instance_geometry></node>"##;
    let png = png();
    let assets = Memory::new().with("tex ture.png", png);
    let text = doc("", &format!("{QUAD}{lib}"), scene);
    let d = read_with(text.as_bytes(), "t.dae", &assets, Keep::ALL).unwrap();
    let m = &d.json()["materials"][0];
    assert_eq!(m["name"], "Red");
    assert_eq!(m["alphaMode"], "BLEND");
    assert_eq!(m["doubleSided"], true);
    assert!(m["pbrMetallicRoughness"]["baseColorTexture"].is_object(), "{m}");
    assert!(m["normalTexture"].is_object(), "{m}");
    assert!((m["pbrMetallicRoughness"]["roughnessFactor"].as_f64().unwrap() - 0.316_227_77).abs() < 1e-4);
    assert_eq!(d.json()["samplers"][0]["wrapS"], json!(crate::material::CLAMP));
    assert!(m["emissiveFactor"][0].as_f64().unwrap() > 0.2);
}

#[test]
fn cameras_and_lights() {
    let lib = r#"<library_cameras><camera id="cam"><optics><technique_common><perspective>
        <xfov>90</xfov><aspect_ratio>2</aspect_ratio><znear>1</znear><zfar>100</zfar>
      </perspective></technique_common></optics></camera>
      <camera id="ortho"><optics><technique_common><orthographic><ymag>3</ymag><znear>0</znear><zfar>10</zfar></orthographic></technique_common></optics></camera></library_cameras>
      <library_lights><light id="sun"><technique_common><directional><color>2 2 1</color></directional></technique_common></light>
        <light id="spot"><technique_common><spot><color>1 1 1</color><falloff_angle>60</falloff_angle></spot></technique_common></light>
        <light id="amb"><technique_common><ambient><color>0.1 0.1 0.1</color></ambient></technique_common></light></library_lights>"#;
    let scene = r##"<node><instance_camera url="#cam"/><instance_camera url="#ortho"/><instance_light url="#sun"/>
        <instance_light url="#spot"/><instance_light url="#amb"/></node>"##;
    let d = read(&doc("", lib, scene));
    let cam = &d.json()["cameras"][0]["perspective"];
    assert!((cam["yfov"].as_f64().unwrap() - 2.0 * 0.5f64.atan()).abs() < 1e-9);
    assert_eq!(d.json()["cameras"][1]["type"], "orthographic");
    let lights = &d.json()["extensions"]["KHR_lights_punctual"]["lights"];
    assert_eq!(lights[0]["intensity"], 2.0);
    assert!((lights[1]["spot"]["outerConeAngle"].as_f64().unwrap() - 30f64.to_radians()).abs() < 1e-9);
    assert_eq!(
        d.json()["nodes"][0]["extras"]["ambientLight"][0]
            .as_f64()
            .map(|v| (v * 10.0).round()),
        Some(1.0)
    );
}

#[test]
fn archives_hold_the_document_and_its_textures() {
    use std::io::Write as _;
    let lib = r##"<library_images><image id="img"><init_from>./images/t.png</init_from></image></library_images>
      <library_effects><effect id="fx"><profile_COMMON><technique sid="c"><lambert>
        <diffuse><texture texture="img" texcoord="t"/></diffuse></lambert></technique></profile_COMMON></effect></library_effects>
      <library_materials><material id="m"><instance_effect url="#fx"/></material></library_materials>"##;
    let scene = r##"<node><instance_geometry url="#quad"><bind_material><technique_common>
        <instance_material symbol="mat" target="#m"/></technique_common></bind_material></instance_geometry></node>"##;
    let text = doc("", &format!("{QUAD}{lib}"), scene);
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file("manifest.xml", opts).unwrap();
    zip.write_all(b"<dae_root>./models/scene%20one.dae</dae_root>").unwrap();
    zip.start_file("models/scene one.dae", opts).unwrap();
    zip.write_all(text.as_bytes()).unwrap();
    zip.start_file("models/images/t.png", opts).unwrap();
    zip.write_all(&png()).unwrap();
    let bytes = zip.finish().unwrap().into_inner();
    let d = read_with(&bytes, "t.zae", &Memory::new(), Keep::ALL).unwrap();
    assert_eq!(d.triangles(), 2);
    assert!(d.json()["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"].is_object());
    assert_eq!(d.json()["images"].as_array().map(Vec::len), Some(1));
}

#[test]
fn encodings_and_missing_scene() {
    let lib = QUAD;
    let text = format!("\u{feff}<COLLADA version=\"1.5.0\">{lib}</COLLADA>");
    let d = read(&text);
    assert_eq!(d.triangles(), 2);
    let utf16: Vec<u8> = text.encode_utf16().flat_map(u16::to_le_bytes).collect();
    let d = read_with(&utf16, "t.dae", &Memory::new(), Keep::ALL).unwrap();
    assert_eq!(d.triangles(), 2);
    assert!(read_with(b"<model/>", "t.dae", &Memory::new(), Keep::ALL).is_err());
}

#[test]
fn triangle_limit_is_kept() {
    let text = doc(
        "",
        QUAD,
        r##"<node><instance_geometry url="#quad"/><instance_geometry url="#quad"/></node>"##,
    );
    let input = Input {
        limit: 3,
        ..Input::memory(text.as_bytes(), "t.dae")
    };
    assert!(load(input, &Memory::new(), Keep::ALL).is_err());
    let input = Input {
        limit: 4,
        ..Input::memory(text.as_bytes(), "t.dae")
    };
    assert!(load(input, &Memory::new(), Keep::ALL).is_ok());
}

/// A document with a bit of everything, for the fuzz tests.
fn everything() -> String {
    let lib = format!(
        r##"{QUAD}{SKIN}
        <library_cameras><camera id="cam"><optics><technique_common><perspective><yfov>40</yfov></perspective></technique_common></optics></camera></library_cameras>
        <library_lights><light id="l"><technique_common><point><color>1 1 1</color></point></technique_common></light></library_lights>
        <library_effects><effect id="fx"><profile_COMMON><technique sid="c"><blinn><diffuse><color>1 0 0 1</color></diffuse>
          <transparent opaque="RGB_ZERO"><color>0.5 0.5 0.5 1</color></transparent></blinn></technique></profile_COMMON></effect></library_effects>
        <library_materials><material id="red"><instance_effect url="#fx"/></material></library_materials>
        <library_animations><animation><animation id="a">
          <source id="i"><float_array id="ia" count="3">0 0.5 1</float_array><technique_common><accessor source="#ia" count="3"><param type="float"/></accessor></technique_common></source>
          <source id="o"><float_array id="oa" count="3">0 1 0</float_array><technique_common><accessor source="#oa" count="3"><param type="float"/></accessor></technique_common></source>
          <source id="it"><float_array id="ita" count="6">0 0 0.3 1 0.8 0</float_array><technique_common><accessor source="#ita" count="3" stride="2"><param name="X" type="float"/><param name="Y" type="float"/></accessor></technique_common></source>
          <source id="k"><Name_array id="ka" count="3">BEZIER STEP HERMITE</Name_array><technique_common><accessor source="#ka" count="3"><param type="name"/></accessor></technique_common></source>
          <sampler id="s"><input semantic="INPUT" source="#i"/><input semantic="OUTPUT" source="#o"/><input semantic="INTERPOLATION" source="#k"/>
            <input semantic="IN_TANGENT" source="#it"/><input semantic="OUT_TANGENT" source="#it"/></sampler>
          <channel source="#s" target="root/t.Y"/>
        </animation></animation></library_animations>"##
    );
    let scene = r##"<node id="root" sid="j0"><translate sid="t">0 0 0</translate><node sid="j1"><node sid="j2"/><node sid="j3"/><node sid="j4"/></node></node>
        <node><matrix>1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1</matrix><instance_controller url="#c"><skeleton>#root</skeleton>
          <bind_material><technique_common><instance_material symbol="mat" target="#red"/></technique_common></bind_material></instance_controller>
          <instance_camera url="#cam"/><instance_light url="#l"/><instance_node url="#root"/></node>"##;
    doc(r#"<unit meter="0.1"/><up_axis>Z_UP</up_axis>"#, &lib, scene)
}

#[test]
fn everything_reads() {
    let d = read(&everything());
    assert!(d.json()["skins"].is_array());
    assert!(d.json()["animations"].is_array());
    assert!(d.json()["cameras"].is_array());
    assert_eq!(d.json()["materials"][0]["alphaMode"], "BLEND");
    flat(&d);
}

#[test]
fn every_prefix_reads_or_fails_cleanly() {
    let text = everything();
    let bytes = text.as_bytes();
    let assets = Memory::new();
    for n in 0..=bytes.len() {
        if let Ok(d) = read_with(&bytes[..n], "t.dae", &assets, Keep::ALL) {
            let _ = d.flatten(Pose::Rest);
        }
    }
}

#[test]
fn flipped_bytes_read_or_fail_cleanly() {
    let text = everything();
    let assets = Memory::new();
    let mut state = 0x2545_f491_u32;
    for _ in 0..3000 {
        let mut bytes = text.as_bytes().to_vec();
        for _ in 0..4 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let at = state as usize % bytes.len();
            bytes[at] = match state >> 24 {
                0..64 => b'-',
                64..128 => b'9',
                128..160 => b'<',
                160..192 => b'#',
                _ => state.to_le_bytes()[1],
            };
        }
        if let Ok(d) = read_with(&bytes, "t.dae", &assets, Keep::ALL) {
            let _ = d.flatten(Pose::Rest);
        }
    }
}

#[test]
fn keep_flags_drop_what_they_say() {
    let text = everything();
    let keep = Keep::ALL
        .with_animations(false)
        .with_skins(false)
        .with_cameras(false)
        .with_lights(false)
        .with_materials(false);
    let d = read_with(text.as_bytes(), "t.dae", &Memory::new(), keep).unwrap();
    for key in ["animations", "skins", "cameras", "materials", "extensions"] {
        assert!(d.json().get(key).is_none(), "{key}");
    }
    assert_eq!(d.triangles(), read(&text).triangles());
}

#[test]
fn lines_and_linestrips_are_segments() {
    // A pair that repeats its vertex draws nothing, and an odd index left over is dropped.
    let geometry = QUAD.replace(
        "<polylist ",
        r##"<lines count="2"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 1 1 2</p></lines>
  <linestrips count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 2 2 3</p></linestrips>
  <polylist "##,
    );
    let text = doc("", &geometry, r##"<node><instance_geometry url="#quad"/></node>"##);
    let d = read(&text);
    let m = flat(&d);
    assert_eq!((m.triangles(), m.lines(), m.points()), (2, 3, 0));
    assert_eq!(d.json()["meshes"][0]["primitives"].as_array().map(Vec::len), Some(3));
    let keep = Keep::ALL.with_lines_and_points(false);
    // The document keeps them; flattening leaves them out.
    let assets = Memory::new();
    let d = super::super::load_bytes(text.as_bytes(), "t.dae", "dae", &assets, keep).unwrap();
    assert_eq!(d.json()["meshes"][0]["primitives"].as_array().map(Vec::len), Some(3));
    let m = flat(&d);
    assert_eq!((m.triangles(), m.lines()), (2, 0));
}

#[test]
fn other_documents_load_once_and_cycles_stop() {
    // Each document instances the next one's node, the last back to the first; and a chain longer than the limit.
    let link = |k: usize, to: &str| {
        doc(
            "",
            &format!(
                r##"{QUAD}<library_nodes><node id="link"><instance_geometry url="#quad"/><instance_node url="{to}#link"/></node></library_nodes>"##
            ),
            &format!(r##"<node id="root{k}"><instance_node url="#link"/></node>"##),
        )
    };
    let mut assets = Memory::new();
    assets
        .insert("b.dae", link(1, "c.dae"))
        .insert("c.dae", link(2, "t.dae"));
    let d = read_with(link(0, "b.dae").as_bytes(), "t.dae", &assets, Keep::ALL).unwrap();
    // Three quads, then the cycle back to the first document's node stops.
    assert_eq!(d.triangles(), 6);

    let mut assets = Memory::new();
    for k in 1..200 {
        assets.insert(&format!("d{k}.dae"), link(k, &format!("d{}.dae", k + 1)));
    }
    let d = read_with(link(0, "d1.dae").as_bytes(), "t.dae", &assets, Keep::ALL).unwrap();
    assert!(d.triangles() > 0 && d.triangles() <= 2 * (MAX_DOCUMENTS + 1));
}

#[test]
fn extras_go_when_not_kept() {
    let text = doc(
        "",
        QUAD,
        r##"<node id="a"><instance_geometry url="#quad"/><extra><technique profile="X"><tag>1</tag></technique></extra></node>"##,
    );
    assert_eq!(read(&text).json()["nodes"][0]["extras"]["extra"]["X"]["tag"], "1");
    let keep = Keep::ALL.with_extras(false);
    let d = read_with(text.as_bytes(), "t.dae", &Memory::new(), keep).unwrap();
    assert!(d.json()["nodes"][0].get("extras").is_none());
}

#[test]
fn bindings_of_one_geometry_share_its_accessors() {
    let lib = r##"<library_effects><effect id="fx"><profile_COMMON><technique sid="c"><lambert><diffuse><color>1 0 0 1</color></diffuse></lambert></technique></profile_COMMON></effect>
        <effect id="gx"><profile_COMMON><technique sid="c"><lambert><diffuse><color>0 1 0 1</color></diffuse></lambert></technique></profile_COMMON></effect></library_effects>
      <library_materials><material id="red"><instance_effect url="#fx"/></material><material id="green"><instance_effect url="#gx"/></material></library_materials>"##;
    let bound = |m: &str| {
        format!(
            r##"<node><instance_geometry url="#quad"><bind_material><technique_common>
            <instance_material symbol="mat" target="#{m}"/></technique_common></bind_material></instance_geometry></node>"##
        )
    };
    let scene = format!("{}{}", bound("red"), bound("green"));
    let d = read(&doc("", &format!("{QUAD}{lib}"), &scene));
    let meshes = d.json()["meshes"].as_array().unwrap();
    assert_eq!(meshes.len(), 2);
    let (a, b) = (&meshes[0]["primitives"][0], &meshes[1]["primitives"][0]);
    assert_eq!(a["attributes"], b["attributes"]);
    assert_eq!(a["indices"], b["indices"]);
    assert_ne!(a["material"], b["material"]);
    let m = flat(&d);
    assert_eq!(m.triangles(), 4);
    let colors: Vec<f32> = m.prims.iter().map(|p| m.materials[p.material].color[1]).collect();
    assert!(colors.contains(&0.0) && colors.contains(&1.0), "{colors:?}");
}

#[test]
fn a_triangle_with_two_corners_at_one_position_is_dropped() {
    // The second triangle's first and last corners name one position with different texture
    // coordinates: it draws nothing.
    let lib = QUAD
        .replace(
            "<vcount>4</vcount><p>0 0 0 1 0 1 2 0 2 3 0 3</p>",
            "<vcount>3 3</vcount><p>0 0 0 1 0 1 2 0 2 0 0 0 1 0 1 0 0 3</p>",
        )
        .replace(r#"<polylist count="1""#, r#"<polylist count="2""#);
    let d = read(&doc("", &lib, r##"<node><instance_geometry url="#quad"/></node>"##));
    assert_eq!(flat(&d).triangles(), 1);
}

#[test]
fn a_decimal_comma_reads_as_a_point() {
    let lib = QUAD.replace("0 0 0 1 0 0 1 1 0 0 1 0", "0 0 0 1,5 0 0 1,5 1 0 0 1 0");
    let d = read(&doc("", &lib, r##"<node><instance_geometry url="#quad"/></node>"##));
    let m = flat(&d);
    let x = m.prims[0].positions.iter().map(|p| p[0]).fold(0.0f32, f32::max);
    assert!((x - 1.5).abs() < 1e-6, "{x}");
}
