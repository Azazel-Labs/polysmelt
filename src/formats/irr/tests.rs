use serde_json::Value;

use crate::assets::Memory;
use crate::document::{Document, Keep, Pose};
use crate::formats::load_bytes;
use crate::quat;

const MESH: &str = r#"<?xml version="1.0"?>
<mesh xmlns="http://irrlicht.sourceforge.net/IRRMESH_09_2007" version="1.0">
<boundingBox minEdge="0 0 0" maxEdge="1 1 0" />
<buffer>
<boundingBox minEdge="0 0 0" maxEdge="1 1 0" />
<material>
<color name="Diffuse" value="ff808080" />
<texture name="Texture1" value="wood.png" />
<bool name="BackfaceCulling" value="false" />
</material>
<vertices type="standard" vertexCount="4">
0 0 0 0 0 -1 ffffffff 0 1
1 0 0 0 0 -1 ffffffff 1 1
1 1 0 0 0 -1 ffffffff 1 0
0 1 0 0 0 -1 ffffffff 0 0
</vertices>
<indices indexCount="6">0 2 1 0 3 2</indices>
</buffer>
<buffer>
<material><enum name="Type" value="trans_vertex_alpha" /></material>
<vertices type="standard" vertexCount="3">
0 0 1 0 0 1 80ff0000 0 0
1 0 1 0 0 1 80ff0000 0 0
0 1 1 0 0 1 80ff0000 0 0
</vertices>
<indices indexCount="3">0 1 2</indices>
</buffer>
</mesh>"#;

fn load<'a>(bytes: &'a [u8], name: &'a str, assets: &'a Memory) -> Document {
    let ext = name.rsplit('.').next().unwrap();
    load_bytes(bytes, name, ext, assets, Keep::ALL).unwrap()
}

fn utf16(text: &str, big: bool) -> Vec<u8> {
    let mut out = if big { vec![0xFE, 0xFF] } else { vec![0xFF, 0xFE] };
    for u in text.encode_utf16() {
        out.extend_from_slice(&if big { u.to_be_bytes() } else { u.to_le_bytes() });
    }
    out
}

#[test]
fn reads_a_mesh() {
    let assets = Memory::new();
    let doc = load(MESH.as_bytes(), "quad.irrmesh", &assets);
    assert_eq!(doc.counts().triangles, 3);
    let j = &doc.json();
    assert_eq!(j["meshes"][0]["primitives"].as_array().unwrap().len(), 2);
    let m = &j["materials"];
    assert_eq!(m[0]["doubleSided"], Value::Bool(true));
    let c = m[0]["pbrMetallicRoughness"]["baseColorFactor"].as_array().unwrap();
    assert!((c[0].as_f64().unwrap() - 0.2158).abs() < 1e-3, "{c:?}");
    // One vertex color for every vertex: the material's, its alpha blending.
    let c = m[1]["pbrMetallicRoughness"]["baseColorFactor"].as_array().unwrap();
    assert!((c[0].as_f64().unwrap() - 1.0).abs() < 1e-6 && c[1].as_f64().unwrap() == 0.0);
    assert!((c[3].as_f64().unwrap() - 128.0 / 255.0).abs() < 1e-3);
    assert_eq!(m[1]["alphaMode"], "BLEND");
    assert_eq!(j["asset"]["extras"]["source"]["handedness"], "left");
    // Mirrored: z negated, the first quad's normal now +Z and its winding with it.
    let flat = doc.flatten(Pose::Rest).unwrap().unwrap();
    let mesh = &flat.prims[0];
    assert!(mesh.positions.iter().all(|p| p[2] <= 0.0));
    let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[mesh.indices[k] as usize]);
    let n = [
        (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
        (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
    ];
    assert!(n[2] > 0.0, "{n:?}");
}

#[test]
fn reads_utf16_both_ways() {
    let assets = Memory::new();
    let plain = load(MESH.as_bytes(), "quad.irrmesh", &assets).to_glb().unwrap();
    for big in [false, true] {
        let bytes = utf16(MESH, big);
        assert_eq!(load(&bytes, "quad.irrmesh", &assets).to_glb().unwrap(), plain);
        // Found by its root when only `.xml` names it.
        assert_eq!(load(&bytes, "quad.xml", &assets).counts().triangles, 3);
    }
}

#[test]
fn tells_an_irrlicht_mesh_from_others() {
    let assets = Memory::new();
    let ogre = br"<mesh><submeshes/></mesh>";
    assert!(load_bytes(ogre, "a.irrmesh", "irrmesh", &assets, Keep::ALL).is_err());
}

const SCENE: &str = r#"<?xml version="1.0"?>
<irr_scene>
<attributes><colorf name="AmbientLight" value="0.2, 0.2, 0.2, 1.0" /></attributes>
<node type="empty">
  <attributes>
    <string name="Name" value="root" />
    <vector3d name="Position" value="0, 0, 10" />
    <vector3d name="Rotation" value="0, 90, 0" />
  </attributes>
  <node type="cube">
    <attributes>
      <string name="Name" value="box" />
      <vector3d name="Position" value="5, 0, 0" />
      <float name="Size" value="2" />
    </attributes>
    <materials><attributes><color name="Diffuse" value="ffff0000" /></attributes></materials>
    <animators>
      <attributes><string name="Type" value="rotation" /><vector3d name="Rotation" value="0, 0.3, 0" /></attributes>
      <attributes><string name="Type" value="flyCircle" /><vector3d name="Center" value="0,0,0" />
        <float name="Radius" value="15" /><float name="Speed" value="0.001" /></attributes>
    </animators>
  </node>
</node>
<node type="sphere">
  <attributes><string name="Name" value="ball" /><float name="Radius" value="1" />
    <int name="PolyCountX" value="8" /><int name="PolyCountY" value="6" /></attributes>
  <animators>
    <attributes><string name="Type" value="flyStraight" /><vector3d name="Start" value="0,0,0" />
      <vector3d name="End" value="0,10,0" /><int name="TimeForWay" value="2000" />
      <bool name="Loop" value="true" /><bool name="PingPong" value="true" /></attributes>
  </animators>
</node>
<node type="skyBox"><attributes><string name="Name" value="sky" /></attributes></node>
<node type="billBoard"><attributes><string name="Name" value="sign" /><float name="Width" value="4" /></attributes></node>
<node type="light">
  <attributes><string name="Name" value="lamp" /><enum name="LightType" value="Spot" />
    <colorf name="DiffuseColor" value="1, 1, 1, 1" /><float name="Radius" value="50" />
    <float name="OuterCone" value="30" /><float name="InnerCone" value="10" /></attributes>
</node>
<node type="camera">
  <attributes><string name="Name" value="eye" /><vector3d name="Position" value="0, 0, -10" />
    <vector3d name="Target" value="0, 0, 0" /><float name="Fovy" value="1" /></attributes>
</node>
<node type="mesh"><attributes><string name="Name" value="quad" /><string name="Mesh" value="quad.irrmesh" /></attributes></node>
<node type="mesh"><attributes><string name="Name" value="tri" /><string name="Mesh" value="models/tri.obj" /></attributes></node>
<node type="mesh"><attributes><string name="Name" value="tri2" /><string name="Mesh" value="models/tri.obj" /></attributes></node>
<node type="mesh"><attributes><string name="Name" value="gone" /><string name="Mesh" value="nowhere.3ds" /></attributes></node>
<node type="waterSurface"><attributes><string name="Name" value="lake" /></attributes></node>
<node type="cube"><attributes><string name="Name" value="hidden" /><bool name="Visible" value="false" /></attributes>
  <node type="cube" />
</node>
</irr_scene>"#;

const OBJ: &str = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";

#[test]
#[allow(clippy::cast_possible_truncation, reason = "node indices in a small scene")]
fn reads_a_scene() {
    let assets = Memory::new()
        .with("quad.irrmesh", MESH.as_bytes())
        .with("models/tri.obj", OBJ.as_bytes());
    let doc = load(SCENE.as_bytes(), "scene.irr", &assets);
    let j = &doc.json();
    let facts = &doc.facts;
    // A cube 12, a sphere 8×6 less the poles' 16 = 80, a sky box 12, a billboard 2, the mesh 3, two OBJs 1 each.
    assert_eq!(doc.counts().triangles, 12 + 80 + 12 + 2 + 3 + 2, "{facts:?}");
    assert_eq!(facts["unsupported_nodes"]["watersurface"], 1);
    assert_eq!(facts["hidden_nodes"], 1);
    assert_eq!(j["asset"]["extras"]["source"]["missingFiles"][0], "nowhere.3ds");
    assert_eq!(j["asset"]["extras"]["source"]["format"], "Irrlicht scene");
    assert_eq!(j["cameras"].as_array().unwrap().len(), 1);
    assert_eq!(j["cameras"][0]["perspective"]["yfov"], 1.0);
    let light = &j["extensions"]["KHR_lights_punctual"]["lights"][0];
    assert_eq!(light["type"], "spot");
    assert!((light["spot"]["outerConeAngle"].as_f64().unwrap() - 30f64.to_radians()).abs() < 1e-6);
    assert_eq!(j["animations"].as_array().unwrap().len(), 2);
    // The two uses of one OBJ share its mesh.
    let names = |n: &str| {
        j["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .position(|x| x["name"] == n)
            .unwrap()
    };
    let obj_mesh = |holder: usize| {
        let child = j["nodes"][holder]["children"][0].as_u64().unwrap() as usize;
        let grand = j["nodes"][child]["children"][0].as_u64().unwrap() as usize;
        j["nodes"][grand]["mesh"].clone()
    };
    assert_eq!(obj_mesh(names("tri")), obj_mesh(names("tri2")));

    // The box: at the root's (0,0,10) turned 90° about Y, so its (5,0,0) is at (0,0,5) in Irrlicht's left-handed
    // space; mirrored, (0,0,-5).
    let world = doc.flatten(Pose::Rest).unwrap().unwrap();
    assert_ne!(world.prims.len(), 0);
    let b = names("box");
    let r = names("root");
    assert_eq!(j["nodes"][r]["children"][0], b);
    let t = &j["nodes"][r]["translation"];
    assert_eq!(t.as_array().unwrap()[2], -10.0);

    // The camera's child looks along glTF's -Z toward the target, mirrored: from z=+10 toward the origin.
    let eye = names("eye");
    let aim = j["nodes"][eye]["children"][0].as_u64().unwrap() as usize;
    let q = j["nodes"][aim]["rotation"]
        .as_array()
        .map_or([0.0, 0.0, 0.0, 1.0], |q| [0, 1, 2, 3].map(|k| q[k].as_f64().unwrap()));
    let forward = quat::rotate(q, [0.0, 0.0, -1.0]);
    assert!(forward[2] < -0.99, "{forward:?}");
    assert_eq!(j["nodes"][eye]["translation"][2], 10.0);
}

#[test]
fn reads_a_scene_in_utf16() {
    let assets = Memory::new()
        .with("quad.irrmesh", utf16(MESH, false))
        .with("models/tri.obj", OBJ.as_bytes());
    let plain = load(SCENE.as_bytes(), "scene.irr", &assets).to_glb().unwrap();
    let bytes = utf16(SCENE, true);
    assert_eq!(load(&bytes, "scene.irr", &assets).to_glb().unwrap(), plain);
    assert_eq!(load(&bytes, "scene.xml", &assets).to_glb().unwrap(), plain);
}

#[test]
fn a_mesh_file_past_the_limit_fails_the_scene() {
    let scene = |file: &str| {
        format!(
            "<?xml version=\"1.0\"?>\n<irr_scene>\n<node type=\"mesh\"><attributes><string name=\"Name\" \
             value=\"m\" /><string name=\"Mesh\" value=\"{file}\" /></attributes></node>\n</irr_scene>"
        )
    };
    let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2 4 3\nf 1 3 4\n";
    let assets = Memory::new()
        .with("quad.irrmesh", MESH.as_bytes())
        .with("three.obj", obj.as_bytes());
    let keep = Keep::ALL.with_max_triangles(2);
    for file in ["quad.irrmesh", "three.obj"] {
        let e = load_bytes(scene(file).as_bytes(), "s.irr", "irr", &assets, keep).unwrap_err();
        assert_eq!(e.kind(), crate::ErrorKind::LimitExceeded, "{file}: {e}");
        // Within the limit the same scene reads.
        let doc = load_bytes(scene(file).as_bytes(), "s.irr", "irr", &assets, Keep::ALL).unwrap();
        assert_eq!(doc.facts["triangles"], 3, "{file}");
    }
}

#[test]
fn keeps_to_the_limit() {
    let assets = Memory::new();
    let keep = Keep::ALL.with_max_triangles(2);
    assert!(load_bytes(MESH.as_bytes(), "q.irrmesh", "irrmesh", &assets, keep).is_err());
    assert!(load_bytes(SCENE.as_bytes(), "s.irr", "irr", &assets, keep).is_err());
}
