//! What every reader does the same way with skins, animations and the textures a model names: the same
//! two-joint skinned strip, turning through the same three keys, written in six formats, and the same texture
//! named three awkward ways in five.
#![cfg(feature = "default")]

use polysmelt::assets::Memory;
use polysmelt::document::Pose;
use polysmelt::rig::{Rig, Trs};
use polysmelt::source::Fact;
use polysmelt::{Document, Loader};
use serde_json::Value;

fn read(bytes: &[u8], name: &str) -> Document {
    polysmelt::read(bytes.to_vec(), name).unwrap_or_else(|e| panic!("{name}: {e}"))
}

// --- The strip ---------------------------------------------------------------------------------------------
//
// Six vertices in the XY plane, x 0 and 1 at y 0, 1 and 2, and two joints: `root` at the origin and `tip` a unit
// above it. The bottom row is root's, the top row tip's, the middle row half each. `tip` turns about +Z: 0, 45
// and 90 degrees at 1, 1.5 and 2 seconds.

/// The strip's vertices at rest, and each one's weights on (root, tip).
const STRIP: [([f32; 3], [f32; 2]); 6] = [
    ([0.0, 0.0, 0.0], [1.0, 0.0]),
    ([1.0, 0.0, 0.0], [1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.5, 0.5]),
    ([1.0, 1.0, 0.0], [0.5, 0.5]),
    ([0.0, 2.0, 0.0], [0.0, 1.0]),
    ([1.0, 2.0, 0.0], [0.0, 1.0]),
];

/// Where vertex `v` of the strip is with `tip` turned `degrees` about +Z, worked out by hand: its place as root
/// has it and as tip has it, by its weights.
fn posed(v: usize, degrees: f32) -> [f32; 3] {
    let ([x, y, _], [root, tip]) = STRIP[v];
    let (s, c) = degrees.to_radians().sin_cos();
    // About tip's origin, (0, 1, 0).
    let (dx, dy) = (x, y - 1.0);
    let turned = [c * dx - s * dy, 1.0 + s * dx + c * dy];
    [root * x + tip * turned[0], root * y + tip * turned[1], 0.0]
}

const DAE: &str = r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Y_UP</up_axis></asset>
<library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array id="pa" count="18">0 0 0 1 0 0 0 1 0 1 1 0 0 2 0 1 2 0</float_array>
    <technique_common><accessor source="#pa" count="6" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="4"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 3 0 3 2 2 3 5 2 5 4</p></triangles>
</mesh></geometry></library_geometries>
<library_controllers><controller id="c"><skin source="#g">
  <bind_shape_matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</bind_shape_matrix>
  <source id="j"><Name_array id="ja" count="2">root tip</Name_array>
    <technique_common><accessor source="#ja" count="2"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="m"><float_array id="ma" count="32">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1  1 0 0 0 0 1 0 -1 0 0 1 0 0 0 0 1</float_array>
    <technique_common><accessor source="#ma" count="2" stride="16"><param name="TRANSFORM" type="float4x4"/></accessor></technique_common></source>
  <source id="w"><float_array id="wa" count="2">1 0.5</float_array>
    <technique_common><accessor source="#wa" count="2"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints><input semantic="JOINT" source="#j"/><input semantic="INV_BIND_MATRIX" source="#m"/></joints>
  <vertex_weights count="6"><input semantic="JOINT" source="#j" offset="0"/><input semantic="WEIGHT" source="#w" offset="1"/>
    <vcount>1 1 2 2 1 1</vcount><v>0 0  0 0  0 1 1 1  0 1 1 1  1 0  1 0</v></vertex_weights>
</skin></controller></library_controllers>
<library_animations><animation id="a">
  <source id="i"><float_array id="ia" count="3">1 1.5 2</float_array>
    <technique_common><accessor source="#ia" count="3"><param name="TIME" type="float"/></accessor></technique_common></source>
  <source id="o"><float_array id="oa" count="3">0 45 90</float_array>
    <technique_common><accessor source="#oa" count="3"><param name="ANGLE" type="float"/></accessor></technique_common></source>
  <source id="k"><Name_array id="ka" count="3">LINEAR LINEAR LINEAR</Name_array>
    <technique_common><accessor source="#ka" count="3"><param name="INTERPOLATION" type="name"/></accessor></technique_common></source>
  <sampler id="s"><input semantic="INPUT" source="#i"/><input semantic="OUTPUT" source="#o"/><input semantic="INTERPOLATION" source="#k"/></sampler>
  <channel source="#s" target="tip/rz.ANGLE"/>
</animation></library_animations>
<library_visual_scenes><visual_scene id="s0">
  <node id="root" name="root" sid="root" type="JOINT">
    <node id="tip" name="tip" sid="tip" type="JOINT"><translate>0 1 0</translate><rotate sid="rz">0 0 1 0</rotate></node>
  </node>
  <node id="body" name="body"><instance_controller url="#c"><skeleton>#root</skeleton></instance_controller></node>
</visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#s0"/></scene></COLLADA>"##;

const FBX: &str = r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7400
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UpAxisSign", "int", "Integer", "",1
		P: "FrontAxis", "int", "Integer", "",2
		P: "FrontAxisSign", "int", "Integer", "",1
		P: "CoordAxis", "int", "Integer", "",0
		P: "CoordAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",100
	}
}
Objects:  {
	Model: 1, "Model::body", "Mesh" {
	}
	Geometry: 10, "Geometry::strip", "Mesh" {
		Vertices: *18 {
			a: 0,0,0,1,0,0,0,1,0,1,1,0,0,2,0,1,2,0
		}
		PolygonVertexIndex: *8 {
			a: 0,1,3,-3,2,3,5,-5
		}
	}
	Model: 11, "Model::root", "LimbNode" {
	}
	Model: 12, "Model::tip", "LimbNode" {
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",0,1,0
		}
	}
	Deformer: 40, "Deformer::skin", "Skin" {
	}
	Deformer: 41, "SubDeformer::root", "Cluster" {
		Indexes: *4 {
			a: 0,1,2,3
		}
		Weights: *4 {
			a: 1,1,0.5,0.5
		}
		Transform: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
		}
		TransformLink: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
		}
	}
	Deformer: 42, "SubDeformer::tip", "Cluster" {
		Indexes: *4 {
			a: 2,3,4,5
		}
		Weights: *4 {
			a: 0.5,0.5,1,1
		}
		Transform: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,-1,0,1
		}
		TransformLink: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1
		}
	}
	AnimationStack: 100, "AnimStack::turn", "" {
	}
	AnimationLayer: 101, "AnimLayer::base", "" {
	}
	AnimationCurveNode: 102, "AnimCurveNode::R", "" {
		Properties70:  {
			P: "d|X", "Number", "", "A",0
			P: "d|Y", "Number", "", "A",0
			P: "d|Z", "Number", "", "A",0
		}
	}
	AnimationCurve: 103, "AnimCurve::", "" {
		KeyTime: *3 {
			a: 46186158000,69279237000,92372316000
		}
		KeyValueFloat: *3 {
			a: 0,45,90
		}
		KeyAttrFlags: *1 {
			a: 4
		}
		KeyAttrDataFloat: *4 {
			a: 0,0,0,0
		}
		KeyAttrRefCount: *1 {
			a: 3
		}
	}
}
Connections:  {
	C: "OO",1,0
	C: "OO",10,1
	C: "OO",11,0
	C: "OO",12,11
	C: "OO",40,10
	C: "OO",41,40
	C: "OO",11,41
	C: "OO",42,40
	C: "OO",12,42
	C: "OO",101,100
	C: "OO",102,101
	C: "OP",102,12, "Lcl Rotation"
	C: "OP",103,102, "d|Z"
}
"#;

/// A row of a DirectX matrix key: `tip` turned `degrees` about Z, a unit up its parent (row vectors).
fn x_key(tick: u32, degrees: f32) -> String {
    let (s, c) = degrees.to_radians().sin_cos();
    format!("{tick};16;{c},{s},0,0,{ns},{c},0,0,0,0,1,0,0,1,0,1;;", ns = -s)
}

fn directx() -> String {
    format!(
        "xof 0303txt 0032
AnimTicksPerSecond {{ 30; }}
Frame root {{
 FrameTransformMatrix {{ 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;; }}
 Frame tip {{
  FrameTransformMatrix {{ 1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1;; }}
 }}
}}
Frame body {{
 FrameTransformMatrix {{ 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;; }}
 Mesh strip {{
  6;
  0;0;0;,1;0;0;,0;1;0;,1;1;0;,0;2;0;,1;2;0;;
  4;
  3;0,1,3;,3;0,3,2;,3;2,3,5;,3;2,5,4;;
  XSkinMeshHeader {{ 2;2;2; }}
  SkinWeights {{
   \"root\";
   4;
   0,1,2,3;
   1,1,0.5,0.5;
   1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;;
  }}
  SkinWeights {{
   \"tip\";
   4;
   2,3,4,5;
   0.5,0.5,1,1;
   1,0,0,0,0,1,0,0,0,0,1,0,0,-1,0,1;;
  }}
 }}
}}
AnimationSet turn {{
 Animation {{
  {{ tip }}
  AnimationKey {{
   4;
   3;
   {},
   {},
   {};
  }}
 }}
}}
",
        x_key(30, 0.0),
        x_key(45, 45.0),
        x_key(60, 90.0)
    )
}

// SMD and MD5 are Z up: the strip stands in their XZ plane, and a turn about glTF's +Z is one about their -Y.

fn smd() -> String {
    let vertex = |v: usize| {
        let ([x, y, _], [root, tip]) = STRIP[v];
        let links = match (root > 0.0, tip > 0.0) {
            (true, true) => format!("2 0 {root} 1 {tip}"),
            (true, false) => "1 0 1".to_owned(),
            _ => "1 1 1".to_owned(),
        };
        format!("0 {x} 0 {y} 0 -1 0 0 0 {links}\n")
    };
    let mut triangles = String::new();
    for t in [[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]] {
        triangles.push_str("skin\n");
        for v in t {
            triangles.push_str(&vertex(v));
        }
    }
    let frame =
        |time: u32, degrees: f32| format!("time {time}\n0 0 0 0 0 0 0\n1 0 0 1 0 {} 0\n", -degrees.to_radians());
    format!(
        "version 1\nnodes\n0 \"root\" -1\n1 \"tip\" 0\nend\nskeleton\n{}{}{}end\ntriangles\n{triangles}end\n",
        frame(30, 0.0),
        frame(45, 45.0),
        frame(60, 90.0)
    )
}

const MD5_MESH: &str = r#"MD5Version 10
commandline ""
numJoints 2
numMeshes 1
joints {
	"root" -1 ( 0 0 0 ) ( 0 0 0 )
	"tip" 0 ( 0 0 1 ) ( 0 0 0 )
}
mesh {
	shader "skin"
	numverts 6
	vert 0 ( 0 0 ) 0 1
	vert 1 ( 1 0 ) 1 1
	vert 2 ( 0 0.5 ) 2 2
	vert 3 ( 1 0.5 ) 4 2
	vert 4 ( 0 1 ) 6 1
	vert 5 ( 1 1 ) 7 1
	numtris 4
	tri 0 0 3 1
	tri 1 0 2 3
	tri 2 2 5 3
	tri 3 2 4 5
	numweights 8
	weight 0 0 1 ( 0 0 0 )
	weight 1 0 1 ( 1 0 0 )
	weight 2 0 0.5 ( 0 0 1 )
	weight 3 1 0.5 ( 0 0 0 )
	weight 4 0 0.5 ( 1 0 1 )
	weight 5 1 0.5 ( 1 0 0 )
	weight 6 1 1 ( 0 0 1 )
	weight 7 1 1 ( 1 0 1 )
}
"#;

/// An `.md5anim` has no times, only a rate: three frames at two a second. `tip`'s quaternion Y is each frame's
/// one number: a turn of -t about Y is (0, -sin(t/2), 0, cos(t/2)), which with the format's negative `w` is
/// (0, sin(t/2), 0, -cos(t/2)).
fn md5_anim() -> String {
    let y = |degrees: f32| (degrees.to_radians() / 2.0).sin();
    format!(
        "MD5Version 10\ncommandline \"\"\nnumFrames 3\nnumJoints 2\nframeRate 2\nnumAnimatedComponents 1\n\
hierarchy {{\n\t\"root\" -1 0 0\n\t\"tip\" 0 16 0\n}}\n\
bounds {{\n\t( 0 0 0 ) ( 1 0 2 )\n\t( 0 0 0 ) ( 1 0 2 )\n\t( 0 0 0 ) ( 1 0 2 )\n}}\n\
baseframe {{\n\t( 0 0 0 ) ( 0 0 0 )\n\t( 0 0 1 ) ( 0 0 0 )\n}}\n\
frame 0 {{\n\t{}\n}}\nframe 1 {{\n\t{}\n}}\nframe 2 {{\n\t{}\n}}\n",
        y(0.0),
        y(45.0),
        y(90.0)
    )
}

const OGEX: &str = r#"Metric (key = "up") { string { "y" } }
BoneNode $root {
  Name { string { "root" } }
  BoneNode $tip {
    Name { string { "tip" } }
    Translation { float[3] { {0, 1, 0} } }
    Rotation %rz (kind = "z") { float { 0 } }
    Animation { Track (target = %rz) { Time { Key { float { 1, 1.5, 2 } } } Value { Key { float { 0, 0.7853981633974483, 1.5707963267948966 } } } } }
  }
}
GeometryNode { Name { string { "body" } } ObjectRef { ref { $g } } }
GeometryObject $g { Mesh (primitive = "triangles") {
  VertexArray (attrib = "position") { float[3] { {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 2, 0}, {1, 2, 0} } }
  IndexArray { unsigned_int16[3] { {0, 1, 3}, {0, 3, 2}, {2, 3, 5}, {2, 5, 4} } }
  Skin {
    Transform { float[16] { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1} } }
    Skeleton {
      BoneRefArray { ref { $root, $tip } }
      Transform { float[16] { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1} } }
    }
    BoneCountArray { unsigned_int16 { 1, 1, 2, 2, 1, 1 } }
    BoneIndexArray { unsigned_int16 { 0, 0, 0, 1, 0, 1, 1, 1 } }
    BoneWeightArray { float { 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1 } }
  }
} }
"#;

/// A column-major matrix times a point.
fn at(m: &[f32; 16], p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r])
}

/// The inverse of a rigid transform (a turn and a move), column-major.
fn rigid_inverse(m: &[f32; 16]) -> [f32; 16] {
    let mut o = [0.0; 16];
    for r in 0..3 {
        for c in 0..3 {
            o[4 * c + r] = m[4 * r + c];
        }
    }
    for r in 0..3 {
        o[12 + r] = -(0..3).map(|c| o[4 * c + r] * m[12 + c]).sum::<f32>();
    }
    o[15] = 1.0;
    o
}

fn near(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 2e-3)
}

/// Where the world point `p`, carried by the joint `tip`, is at `time` of the document's one animation.
fn carried(rig: &Rig, time: f32, p: [f32; 3]) -> [f32; 3] {
    let tip = rig.find("tip").expect("a joint named tip");
    let rest: Vec<Trs> = rig.nodes.iter().map(|n| n.rest).collect();
    let from = rigid_inverse(&rig.world(&rest)[tip]);
    let to = rig.world(&rig.pose(Some(0), time))[tip];
    at(&to, at(&from, p))
}

/// A vertex's place and its weights by joint name, in thousandths.
type Weighted = ([i32; 3], Vec<(String, i32)>);

/// Each drawn vertex's place and its weights by joint name, sorted, from the flat mesh at rest.
fn weights(doc: &Document) -> Vec<Weighted> {
    let mesh = doc.flatten(Pose::Rest).unwrap().expect("a mesh");
    let mut out = Vec::new();
    for p in &mesh.prims {
        assert!(p.is_skinned());
        for (v, pos) in p.positions.iter().enumerate() {
            let mut w: Vec<(String, i32)> = (0..4)
                .filter(|&k| p.skin_weights[v][k] > 0.0)
                .map(|k| {
                    let bone = &mesh.skeleton[p.skin_joints[v][k] as usize];
                    #[allow(clippy::cast_possible_truncation, reason = "a weight in thousandths")]
                    (
                        bone.name.clone().unwrap_or_default(),
                        (p.skin_weights[v][k] * 1000.0).round() as i32,
                    )
                })
                .collect();
            w.sort();
            #[allow(clippy::cast_possible_truncation, reason = "a small coordinate in thousandths")]
            out.push((pos.map(|c| (c * 1000.0).round() as i32), w));
        }
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn the_same_skinned_strip_reads_the_same_from_every_format() {
    let cases: Vec<(&str, Vec<u8>, f64)> = vec![
        ("strip.dae", DAE.into(), 1.0),
        ("strip.fbx", FBX.into(), 1.0),
        ("strip.x", directx().into_bytes(), 1.0),
        ("strip.smd", smd().into_bytes(), 1.0),
        ("strip.ogex", OGEX.into(), 1.0),
        ("strip.md5mesh", MD5_MESH.into(), 0.0),
    ];
    // What the file says of the strip, worked out from `STRIP`.
    let mut want: Vec<Weighted> = STRIP
        .iter()
        .map(|(p, [root, tip])| {
            #[allow(clippy::cast_possible_truncation, reason = "small numbers in thousandths")]
            let k = |x: f32| (x * 1000.0).round() as i32;
            let mut w = vec![("root".to_owned(), k(*root)), ("tip".to_owned(), k(*tip))];
            w.retain(|e| e.1 > 0);
            (p.map(k), w)
        })
        .collect();
    want.sort();
    for (name, bytes, offset) in &cases {
        let doc = read(bytes, name);
        let rig = doc.rig().unwrap();
        // The file's two joints and no more: every vertex has a joint, so no rest joint is added.
        assert_eq!(rig.skins.len(), 1, "{name}");
        let joints: Vec<_> = rig.skins[0]
            .joints
            .iter()
            .map(|&j| rig.nodes[j].name.clone().unwrap_or_default())
            .collect();
        assert_eq!(joints, ["root", "tip"], "{name}");
        // `skeleton` is the joints' common root: the root joint.
        assert_eq!(rig.skins[0].skeleton, rig.find("root"), "{name}");
        assert_eq!(weights(&doc), want, "{name}");
        if name.ends_with("md5mesh") {
            continue;
        }
        // One animation, from 0 to 1 second, what was taken off its keys noted.
        assert_eq!(rig.animations.len(), 1, "{name}");
        let (first, last) = rig.animations[0].span();
        assert!(first == 0.0 && (last - 1.0).abs() < 1e-5, "{name}: {first}..{last}");
        let extras = &doc.json()["animations"][0]["extras"];
        assert_eq!(extras["timeOffset"].as_f64(), Some(*offset), "{name}: {extras}");
        // Half-way, tip has turned 45 degrees: the joint, and the mesh it carries.
        assert!(near(carried(&rig, 0.5, [0.0, 2.0, 0.0]), posed(4, 45.0)), "{name}");
        assert!(near(carried(&rig, 1.0, [1.0, 2.0, 0.0]), posed(5, 90.0)), "{name}");
        let mesh = doc
            .flatten(Pose::Animation { index: 0, time: 0.5 })
            .unwrap()
            .expect("a mesh");
        let mut want: Vec<[f32; 3]> = (0..6).map(|v| posed(v, 45.0)).collect();
        for p in mesh.prims.iter().flat_map(|p| &p.positions) {
            assert!(want.iter().any(|w| near(*w, *p)), "{name}: {p:?}");
            want.retain(|w| !near(*w, *p));
        }
        assert!(want.is_empty(), "{name}: no vertex at {want:?}");
    }
    // An `.md5anim` is the motion alone, with no times but a rate: it starts at 0 with nothing taken off.
    let doc = read(md5_anim().as_bytes(), "turn.md5anim");
    let rig = doc.rig().unwrap();
    assert_eq!(rig.animations[0].span(), (0.0, 1.0));
    assert!(doc.json()["animations"][0].get("extras").is_none());
    assert!(near(carried(&rig, 0.5, [0.0, 2.0, 0.0]), posed(4, 45.0)));
}

#[test]
fn an_animation_keeps_the_files_name_or_has_none() {
    let name = |bytes: &[u8], file: &str| read(bytes, file).json()["animations"][0].get("name").cloned();
    // Named in the file.
    assert_eq!(name(FBX.as_bytes(), "strip.fbx"), Some("turn".into()));
    assert_eq!(name(directx().as_bytes(), "strip.x"), Some("turn".into()));
    // Not named in the file, which holds more than a clip: no made-up name.
    assert_eq!(name(DAE.as_bytes(), "strip.dae"), None);
    // A file that is one clip is named as a library of clips names it: by its file.
    assert_eq!(name(md5_anim().as_bytes(), "turn.md5anim"), Some("turn".into()));
    assert_eq!(name(smd().as_bytes(), "wave.smd"), Some("wave".into()));
    let bvh = b"HIERARCHY\nROOT hips\n{\nOFFSET 0 0 0\nCHANNELS 3 Zrotation Xrotation Yrotation\nEnd Site\n{\nOFFSET 0 1 0\n}\n}\nMOTION\nFrames: 2\nFrame Time: 0.5\n0 0 0\n90 0 0\n";
    assert_eq!(name(bvh, "walk.bvh"), Some("walk".into()));
}

#[test]
fn nodes_keyed_from_different_times_stay_in_step() {
    const SECOND: i64 = 46_186_158_000;
    // Two FBX models moving along X: one keyed from -1 s, the other from 1 s. They move back by the same second,
    // the animation's earliest key, so the second still starts two seconds after the first.
    let curve = |id: u32, t0: i64, t1: i64| {
        format!(
            "\tAnimationCurveNode: {n}, \"AnimCurveNode::T\", \"\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"d|X\", \"Number\", \"\", \"A\",0\n\t\t\tP: \"d|Y\", \"Number\", \"\", \"A\",0\n\t\t\tP: \"d|Z\", \"Number\", \"\", \"A\",0\n\t\t}}\n\t}}\n\tAnimationCurve: {id}, \"AnimCurve::\", \"\" {{\n\t\tKeyTime: *2 {{\n\t\t\ta: {t0},{t1}\n\t\t}}\n\t\tKeyValueFloat: *2 {{\n\t\t\ta: 0,1\n\t\t}}\n\t\tKeyAttrFlags: *1 {{\n\t\t\ta: 4\n\t\t}}\n\t\tKeyAttrDataFloat: *4 {{\n\t\t\ta: 0,0,0,0\n\t\t}}\n\t\tKeyAttrRefCount: *1 {{\n\t\t\ta: 2\n\t\t}}\n\t}}\n",
            n = id - 1
        )
    };
    let text = format!(
        "; FBX 7.4.0 project file\nFBXHeaderExtension:  {{\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n}}\nObjects:  {{\n\tModel: 1, \"Model::early\", \"Null\" {{\n\t}}\n\tModel: 2, \"Model::late\", \"Null\" {{\n\t}}\n\tAnimationStack: 100, \"AnimStack::take\", \"\" {{\n\t}}\n\tAnimationLayer: 101, \"AnimLayer::base\", \"\" {{\n\t}}\n{}{}}}\nConnections:  {{\n\tC: \"OO\",1,0\n\tC: \"OO\",2,0\n\tC: \"OO\",101,100\n\tC: \"OO\",102,101\n\tC: \"OO\",104,101\n\tC: \"OP\",102,1, \"Lcl Translation\"\n\tC: \"OP\",103,102, \"d|X\"\n\tC: \"OP\",104,2, \"Lcl Translation\"\n\tC: \"OP\",105,104, \"d|X\"\n}}\n",
        curve(103, -SECOND, 0),
        curve(105, SECOND, 2 * SECOND)
    );
    let doc = read(text.as_bytes(), "two.fbx");
    let rig = doc.rig().unwrap();
    let a = &rig.animations[0];
    let start = |node: &str| {
        let n = rig.find(node).unwrap();
        let c = a.channels.iter().find(|c| c.node == n).unwrap();
        (c.times[0], *c.times.last().unwrap())
    };
    assert_eq!(start("early"), (0.0, 1.0));
    assert_eq!(start("late"), (2.0, 3.0));
    assert_eq!(doc.json()["animations"][0]["extras"]["timeOffset"].as_f64(), Some(-1.0));
}

#[test]
fn a_frame_rate_the_file_does_not_give_is_one_assumption() {
    // A TRC header with no rates, and an IQM-like guess, take the same assumed 30; one the file gives is declared.
    let trc = |rate: &str| {
        format!(
            "PathFileType\t4\t(X/Y/Z)\tt.trc\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n{rate}\t{rate}\t2\t1\tmm\t{rate}\t1\t2\nFrame#\tTime\tM1\t\t\n\t\tX1\tY1\tZ1\n1\t\t0\t0\t0\n2\t\t1\t0\t0\n"
        )
    };
    assert_eq!(
        read(trc("").as_bytes(), "t.trc").source_info().frame_rate,
        Fact::Assumed(30.0)
    );
    assert_eq!(
        read(trc("50").as_bytes(), "t.trc").source_info().frame_rate,
        Fact::Declared(50.0)
    );
    // An SMD has no rate: the compilers' default, which is the same 30.
    assert_eq!(
        read(smd().as_bytes(), "t.smd").source_info().frame_rate,
        Fact::Assumed(30.0)
    );
}

#[test]
fn a_vertex_no_joint_moves_stays_with_its_mesh_and_not_with_the_first_joint() {
    // The strip with its top row weighted to nothing. The first joint, root, is the one that turns here: the top
    // row must stay where it is, as the node drawing the mesh does, in every format.
    let dae = DAE
        .replace("<vcount>1 1 2 2 1 1</vcount>", "<vcount>1 1 2 2 0 0</vcount>")
        .replace(
            "<v>0 0  0 0  0 1 1 1  0 1 1 1  1 0  1 0</v>",
            "<v>0 0  0 0  0 1 1 1  0 1 1 1</v>",
        )
        .replace("target=\"tip/rz.ANGLE\"", "target=\"root/rz.ANGLE\"")
        .replace(
            "<node id=\"root\" name=\"root\" sid=\"root\" type=\"JOINT\">",
            "<node id=\"root\" name=\"root\" sid=\"root\" type=\"JOINT\"><rotate sid=\"rz\">0 0 1 0</rotate>",
        );
    let ogex = OGEX
        .replace("{ 1, 1, 2, 2, 1, 1 }", "{ 1, 1, 2, 2, 0, 0 }")
        .replace("{ 0, 0, 0, 1, 0, 1, 1, 1 }", "{ 0, 0, 0, 1, 0, 1 }")
        .replace("{ 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1 }", "{ 1, 1, 0.5, 0.5, 0.5, 0.5 }")
        .replace(
            "  Name { string { \"root\" } }\n",
            "  Name { string { \"root\" } }\n  Rotation %rr (kind = \"z\") { float { 0 } }\n  Animation { Track (target = %rr) { Time { Key { float { 0, 1 } } } Value { Key { float { 0, 1.5707963267948966 } } } } }\n",
        )
        .replace("    Animation { Track (target = %rz)", "    Animation (clip = 1) { Track (target = %rz)");
    for (name, text) in [("strip.dae", dae), ("strip.ogex", ogex)] {
        let doc = read(text.as_bytes(), name);
        let rig = doc.rig().unwrap();
        let joints: Vec<_> = rig.skins[0]
            .joints
            .iter()
            .map(|&j| rig.nodes[j].name.clone().unwrap_or_default())
            .collect();
        // The file's joints, then the node that draws the mesh.
        assert_eq!(joints, ["root", "tip", "body"], "{name}");
        let (_, end) = rig.animations[0].span();
        let mesh = doc
            .flatten(Pose::Animation { index: 0, time: end })
            .unwrap()
            .expect("a mesh");
        let all: Vec<[f32; 3]> = mesh.prims.iter().flat_map(|p| p.positions.clone()).collect();
        for still in [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]] {
            assert!(all.iter().any(|p| near(*p, still)), "{name}: {still:?} moved: {all:?}");
        }
        // The bottom right corner, root's, has turned a quarter about the origin.
        assert!(all.iter().any(|p| near(*p, [0.0, 1.0, 0.0])), "{name}: {all:?}");
    }
}

// --- Textures ----------------------------------------------------------------------------------------------

const PNG: &[u8] = include_bytes!("corpus/md5/skinned_quad/skin_d.png");

/// A 3DS file: one textured triangle whose material's texture map names `texture`.
fn tds(texture: &str) -> Vec<u8> {
    fn chunk(id: u16, body: &[u8]) -> Vec<u8> {
        let mut out = id.to_le_bytes().to_vec();
        out.extend(u32::try_from(body.len() + 6).unwrap().to_le_bytes());
        out.extend_from_slice(body);
        out
    }
    let cstr = |s: &str| [s.as_bytes(), &[0]].concat();
    let map = chunk(0xA200, &chunk(0xA300, &cstr(texture)));
    let material = chunk(0xAFFF, &[chunk(0xA000, &cstr("m")), map].concat());
    let mut vertices = 3u16.to_le_bytes().to_vec();
    for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        vertices.extend(v.iter().flat_map(|c| c.to_le_bytes()));
    }
    let mut uvs = 3u16.to_le_bytes().to_vec();
    for v in [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
        uvs.extend(v.iter().flat_map(|c| c.to_le_bytes()));
    }
    let group = chunk(
        0x4130,
        &[cstr("m"), 1u16.to_le_bytes().to_vec(), 0u16.to_le_bytes().to_vec()].concat(),
    );
    let mut faces = 1u16.to_le_bytes().to_vec();
    for i in [0u16, 1, 2, 0] {
        faces.extend(i.to_le_bytes());
    }
    faces.extend(group);
    let mesh = chunk(
        0x4100,
        &[chunk(0x4110, &vertices), chunk(0x4140, &uvs), chunk(0x4120, &faces)].concat(),
    );
    let object = chunk(0x4000, &[cstr("o"), mesh].concat());
    chunk(0x4D4D, &chunk(0x3D3D, &[material, object].concat()))
}

#[test]
fn a_texture_named_another_machines_way_is_found_beside_the_model_in_every_format() {
    // The file is `tex.png`, beside the model. The model names it by another machine's absolute path, by a
    // `file:` URI with a space escaped, and in another case.
    let references = [r"C:\Old\Path\Tex.PNG", "file:///C:/Old%20Path/tex.png", "TEX.png"];
    for reference in references {
        // JSON and XML want the backslashes escaped or left alone, as each does.
        let json = reference.replace('\\', "\\\\");
        let gltf = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"TEXCOORD_0":1}},"material":0}}]}}],
"materials":[{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}],
"textures":[{{"source":0}}],"images":[{{"uri":"{json}"}}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
{{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}}],
"bufferViews":[{{"buffer":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":24}}],
"buffers":[{{"byteLength":60,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/"}}]}}"#
        );
        let obj = "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl m\nf 1/1 2/2 3/3\n";
        let mtl = format!("newmtl m\nKd 1 1 1\nmap_Kd {reference}\n");
        let dae = format!(
            r##"<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<library_images><image id="img"><init_from>{reference}</init_from></image></library_images>
<library_effects><effect id="fx"><profile_COMMON>
  <newparam sid="surf"><surface type="2D"><init_from>img</init_from></surface></newparam>
  <newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>
  <technique sid="t"><lambert><diffuse><texture texture="samp" texcoord="UV"/></diffuse></lambert></technique>
</profile_COMMON></effect></library_effects>
<library_materials><material id="mat"><instance_effect url="#fx"/></material></library_materials>
<library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array id="pa" count="9">0 0 0 1 0 0 0 1 0</float_array><technique_common><accessor source="#pa" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
  <source id="u"><float_array id="ua" count="6">0 0 1 0 0 1</float_array><technique_common><accessor source="#ua" count="3" stride="2"><param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="1" material="m"><input semantic="VERTEX" source="#v" offset="0"/><input semantic="TEXCOORD" source="#u" offset="0" set="0"/><p>0 1 2</p></triangles>
</mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"><bind_material><technique_common>
  <instance_material symbol="m" target="#mat"><bind_vertex_input semantic="UV" input_semantic="TEXCOORD" input_set="0"/></instance_material>
</technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#s"/></scene></COLLADA>"##
        );
        let fbx = format!(
            "; FBX 7.4.0 project file\nFBXHeaderExtension:  {{\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n}}\nObjects:  {{\n\tModel: 1, \"Model::m\", \"Mesh\" {{\n\t}}\n\tGeometry: 10, \"Geometry::g\", \"Mesh\" {{\n\t\tVertices: *9 {{\n\t\t\ta: 0,0,0,1,0,0,0,1,0\n\t\t}}\n\t\tPolygonVertexIndex: *3 {{\n\t\t\ta: 0,1,-3\n\t\t}}\n\t\tLayerElementUV: 0 {{\n\t\t\tMappingInformationType: \"ByPolygonVertex\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tUV: *6 {{\n\t\t\t\ta: 0,0,1,0,0,1\n\t\t\t}}\n\t\t}}\n\t}}\n\tMaterial: 20, \"Material::mat\", \"\" {{\n\t}}\n\tTexture: 30, \"Texture::t\", \"\" {{\n\t\tFileName: \"{reference}\"\n\t\tRelativeFilename: \"{reference}\"\n\t}}\n}}\nConnections:  {{\n\tC: \"OO\",1,0\n\tC: \"OO\",10,1\n\tC: \"OO\",20,1\n\tC: \"OP\",30,20, \"DiffuseColor\"\n}}\n"
        );
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("pack/model.gltf", gltf.into_bytes()),
            ("pack/model.obj", obj.into()),
            ("pack/model.dae", dae.into_bytes()),
            ("pack/model.fbx", fbx.into_bytes()),
            ("pack/model.3ds", tds(reference)),
        ];
        for (name, bytes) in cases {
            let assets = Memory::new()
                .with("pack/tex.png", PNG.to_vec())
                .with("pack/m.mtl", mtl.clone().into_bytes());
            let doc = Loader::new()
                .assets(assets)
                .read(bytes, name)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let json = doc.json();
            let embedded = json
                .get("images")
                .and_then(Value::as_array)
                .is_some_and(|i| i.len() == 1 && i[0].get("bufferView").is_some());
            assert!(embedded, "{name} naming {reference:?}: {:?}", json.get("images"));
            let textured = json["materials"].as_array().is_some_and(|m| {
                m.iter()
                    .any(|m| m.pointer("/pbrMetallicRoughness/baseColorTexture").is_some())
            });
            assert!(textured, "{name} naming {reference:?}");
            let source = &json["asset"]["extras"]["source"];
            assert!(source.get("droppedTextures").is_none(), "{name}: {source}");
            assert!(source.get("missingFiles").is_none(), "{name}: {source}");
        }
    }
}

#[test]
fn a_texture_that_is_not_there_is_counted_and_named_the_same_way_by_every_reader() {
    let obj = "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl m\nf 1/1 2/2 3/3\n";
    let cases: Vec<(&str, Vec<u8>)> = vec![("pack/model.obj", obj.into()), ("pack/model.3ds", tds("gone.png"))];
    for (name, bytes) in cases {
        let assets = Memory::new().with("pack/m.mtl", "newmtl m\nKd 1 1 1\nmap_Kd gone.png\n");
        let doc = Loader::new().assets(assets).read(bytes, name).unwrap();
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["droppedTextures"], 1, "{name}: {source}");
        assert_eq!(
            source["missingFiles"],
            serde_json::json!(["gone.png"]),
            "{name}: {source}"
        );
    }
}
