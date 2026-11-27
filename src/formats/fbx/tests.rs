use super::animation::{Animated, Blend, Curve, FLAT, Interp, Layered, THIRD, handle_lengths, timeline};
use super::curves::whole;
use super::deform::split_weight;
use super::walker::rotate;
use super::*;
use crate::document::Pose;
use crate::mesh::{Mesh, srgb_to_linear};
use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

/// The file at `p`, with assets under `dir`, flattened in its rest pose.
fn flat(p: &Path, dir: &Path) -> Mesh {
    let doc = load(
        crate::formats::Input::file(p),
        &crate::assets::Folder::new(dir),
        Keep::ALL,
    )
    .unwrap();
    doc.flatten(Pose::Rest).unwrap().unwrap()
}

fn png(rgb: [u8; 3]) -> Vec<u8> {
    let mut b = Vec::new();
    image::RgbImage::from_pixel(2, 2, image::Rgb(rgb))
        .write_to(&mut std::io::Cursor::new(&mut b), image::ImageFormat::Png)
        .unwrap();
    b
}

/// A Z-up, centimeter FBX (3ds Max style): one textured quad on the ground, its model raised 100 cm.
pub(crate) fn z_up_fbx() -> Vec<u8> {
    let mut w = Writer::new(std::io::Cursor::new(Vec::new()), fbxcel::low::FbxVersion::V7_4).unwrap();
    let p = |w: &mut Writer<std::io::Cursor<Vec<u8>>>, name: &str, vals: &[A]| {
        let mut a = w.new_node("P").unwrap();
        for s in [name, name, "", "A"] {
            a.append_string_direct(s).unwrap();
        }
        for v in vals {
            match v {
                A::F64(f) => a.append_f64(*f).unwrap(),
                A::I32(i) => a.append_i32(*i).unwrap(),
                _ => unreachable!(),
            }
        }
        w.close_node().unwrap();
    };
    let object = |w: &mut Writer<std::io::Cursor<Vec<u8>>>, kind: &str, id: i64, name: &str, sub: &str| {
        let mut a = w.new_node(kind).unwrap();
        a.append_i64(id).unwrap();
        a.append_string_direct(&format!("{name}\0\x01{kind}")).unwrap();
        a.append_string_direct(sub).unwrap();
    };
    // Global settings: Z up, -Y front, X right, centimeters.
    w.new_node("GlobalSettings").unwrap();
    w.new_node("Properties70").unwrap();
    for (k, v) in [
        ("UpAxis", 2),
        ("UpAxisSign", 1),
        ("FrontAxis", 1),
        ("FrontAxisSign", -1),
        ("CoordAxis", 0),
        ("CoordAxisSign", 1),
    ] {
        p(&mut w, k, &[A::I32(v)]);
    }
    p(&mut w, "UnitScaleFactor", &[A::F64(1.0)]);
    w.close_node().unwrap();
    w.close_node().unwrap();

    w.new_node("Objects").unwrap();
    object(&mut w, "Geometry", 10, "Quad", "Mesh");
    {
        let mut a = w.new_node("Vertices").unwrap();
        a.append_arr_f64_from_iter(None, [0., 0., 0., 100., 0., 0., 100., 100., 0., 0., 100., 0.])
            .unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("PolygonVertexIndex").unwrap();
        a.append_arr_i32_from_iter(None, [0, 1, 2, !3]).unwrap();
    }
    w.close_node().unwrap();
    {
        let mut a = w.new_node("LayerElementUV").unwrap();
        a.append_i32(0).unwrap();
    }
    for (name, s) in [
        ("MappingInformationType", "ByPolygonVertex"),
        ("ReferenceInformationType", "IndexToDirect"),
    ] {
        w.new_node(name).unwrap().append_string_direct(s).unwrap();
        w.close_node().unwrap();
    }
    w.new_node("UV")
        .unwrap()
        .append_arr_f64_from_iter(None, [0., 0., 1., 0., 1., 1., 0., 1.])
        .unwrap();
    w.close_node().unwrap();
    w.new_node("UVIndex")
        .unwrap()
        .append_arr_i32_from_iter(None, [0, 1, 2, 3])
        .unwrap();
    w.close_node().unwrap();
    w.close_node().unwrap(); // LayerElementUV
    w.close_node().unwrap(); // Geometry

    object(&mut w, "Model", 20, "Floor", "Mesh");
    w.new_node("Properties70").unwrap();
    p(&mut w, "Lcl Translation", &[A::F64(0.0), A::F64(0.0), A::F64(100.0)]);
    w.close_node().unwrap();
    w.close_node().unwrap();

    object(&mut w, "Material", 30, "Stone", "");
    w.new_node("Properties70").unwrap();
    p(&mut w, "DiffuseColor", &[A::F64(0.5), A::F64(0.4), A::F64(0.3)]);
    w.close_node().unwrap();
    w.close_node().unwrap();

    object(&mut w, "Texture", 40, "StoneTex", "");
    w.new_node("RelativeFilename")
        .unwrap()
        .append_string_direct("C:\\art\\stone.png")
        .unwrap();
    w.close_node().unwrap();
    w.close_node().unwrap();

    object(&mut w, "Video", 50, "StoneVideo", "Clip");
    w.new_node("Content")
        .unwrap()
        .append_binary_direct(&png([10, 200, 30]))
        .unwrap();
    w.close_node().unwrap();
    w.close_node().unwrap();
    w.close_node().unwrap(); // Objects

    w.new_node("Connections").unwrap();
    for (kind, child, parent, prop) in [
        ("OO", 10, 20, ""),
        ("OO", 20, 0, ""),
        ("OO", 30, 20, ""),
        ("OP", 40, 30, "DiffuseColor"),
        ("OO", 50, 40, ""),
    ] {
        let mut a = w.new_node("C").unwrap();
        a.append_string_direct(kind).unwrap();
        a.append_i64(child).unwrap();
        a.append_i64(parent).unwrap();
        if !prop.is_empty() {
            a.append_string_direct(prop).unwrap();
        }
        w.close_node().unwrap();
    }
    w.close_node().unwrap();
    w.finalize_and_flush(&FbxFooter::default()).unwrap().into_inner()
}

#[test]
fn z_up_centimeters_become_y_up_meters_with_embedded_texture() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("floor.fbx");
    std::fs::write(&p, z_up_fbx()).unwrap();
    let mesh = flat(&p, dir.path());
    assert_eq!(mesh.triangles(), 2);
    let prim = &mesh.prims[0];
    for v in &prim.positions {
        assert!((v[1] - 1.0).abs() < 1e-5, "raised 100 cm = 1 m up: {v:?}");
    }
    let xs: Vec<f32> = prim.positions.iter().map(|v| v[0]).collect();
    let zs: Vec<f32> = prim.positions.iter().map(|v| v[2]).collect();
    assert!(
        xs.iter().any(|&x| (x - 1.0).abs() < 1e-5) && zs.iter().any(|&z| (z + 1.0).abs() < 1e-5),
        "{:?}",
        prim.positions
    );
    // Ground-facing quad in Z-up faces +Y once converted.
    assert!(prim.normals.iter().all(|n| n[1] > 0.99), "{:?}", prim.normals);
    assert_eq!(prim.uvs.len(), prim.positions.len());
    assert!(
        prim.uvs.iter().any(|t| (t[1] - 1.0).abs() < 1e-6),
        "V flipped to glTF's origin"
    );
    let mat = &mesh.materials[prim.material];
    assert_eq!(mat.name, "Stone");
    assert_eq!(mat.color_image, Some(0));
    assert_eq!(mesh.images.len(), 1);
    let glb = mesh.to_glb().unwrap();
    let doc: serde_json::Value = {
        let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        serde_json::from_slice(&glb[20..20 + len]).unwrap()
    };
    assert_eq!(
        doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"],
        0
    );
    assert!(doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"].is_number());
}

/// `z_up_fbx` as ASCII FBX, the way Asset Forge writes it.
fn z_up_ascii_fbx() -> String {
    use base64::Engine as _;
    let png = base64::engine::general_purpose::STANDARD.encode(png([10, 200, 30]));
    let (head, tail) = png.split_at(png.len() / 2);
    format!(
        r#"; FBX 7.3.0 project file
FBXHeaderExtension:  {{
	FBXHeaderVersion: 1003
	FBXVersion: 7300
}}
GlobalSettings:  {{
	Version: 1000
	Properties70:  {{
		P: "UpAxis", "int", "Integer", "",2
		P: "UpAxisSign", "int", "Integer", "",1
		P: "FrontAxis", "int", "Integer", "",1
		P: "FrontAxisSign", "int", "Integer", "",-1
		P: "CoordAxis", "int", "Integer", "",0
		P: "CoordAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",1
	}}
}}
Objects:  {{
	Geometry: 10, "Geometry::Quad", "Mesh" {{
		Vertices: *12 {{
			a: 0,0,0,100,0,0,100,100,0,0,100,0
		}}
		PolygonVertexIndex: *4 {{
			a: 0,1,2,-4
		}}
		LayerElementUV: 0 {{
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "IndexToDirect"
			UV: *8 {{
				a: 0,0,1,0,1,1,0,1
			}}
			UVIndex: *4 {{
				a: 0,1,2,3
			}}
		}}
	}}
	Model: 20, "Model::Floor", "Mesh" {{
		Properties70:  {{
			P: "Lcl Translation", "Lcl Translation", "", "A",0,0,100
		}}
	}}
	Material: 30, "Material::Stone", "" {{
		Properties70:  {{
			P: "DiffuseColor", "Color", "", "A",0.5,0.4,0.3
		}}
	}}
	Texture: 40, "Texture::StoneTex", "" {{
		RelativeFilename: "C:\art\stone.png"
	}}
	Video: 50, "Video::StoneVideo", "Clip" {{
		Content: , "{head}",
			"{tail}"
	}}
}}
Connections:  {{
	;Geometry::Quad, Model::Floor
	C: "OO",10,20
	C: "OO",20,0
	C: "OO",30,20
	C: "OP",40,30, "DiffuseColor"
	C: "OO",50,40
}}
"#
    )
}

#[test]
fn ascii_fbx_reads_like_binary() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (dir.path().join("ascii.fbx"), dir.path().join("binary.fbx"));
    std::fs::write(&a, z_up_ascii_fbx()).unwrap();
    std::fs::write(&b, z_up_fbx()).unwrap();
    let (a, b) = (flat(&a, dir.path()), flat(&b, dir.path()));
    assert_eq!(a.triangles(), 2);
    assert_eq!(a.prims[0].positions, b.prims[0].positions);
    assert_eq!(a.prims[0].normals, b.prims[0].normals);
    assert_eq!(a.prims[0].uvs, b.prims[0].uvs);
    assert_eq!(a.materials[a.prims[0].material].name, "Stone");
    assert_eq!(a.materials[a.prims[0].material].color_image, Some(0));
    assert_eq!(a.images.len(), 1);
}

#[test]
fn every_uv_and_color_layer_and_the_tangents_are_read() {
    let layers = r#"		LayerElementUV: 1 {
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "Direct"
			UV: *8 {
				a: 0,0,2,0,2,2,0,2
			}
		}
		LayerElementNormal: 0 {
			MappingInformationType: "ByControlPoint"
			ReferenceInformationType: "Direct"
			Normals: *12 {
				a: 0,0,1,0,0,1,0,0,1,0,0,1
			}
		}
		LayerElementTangent: 0 {
			MappingInformationType: "ByControlPoint"
			ReferenceInformationType: "Direct"
			Tangents: *12 {
				a: 1,0,0.5,1,0,0.5,1,0,0.5,1,0,0.5
			}
		}
		LayerElementBinormal: 0 {
			MappingInformationType: "ByControlPoint"
			ReferenceInformationType: "Direct"
			Binormals: *12 {
				a: 0,-1,0,0,-1,0,0,-1,0,0,-1,0
			}
		}
		LayerElementColor: 0 {
			MappingInformationType: "AllSame"
			ReferenceInformationType: "Direct"
			Colors: *4 {
				a: 1,0,0,1
			}
		}
		LayerElementColor: 1 {
			MappingInformationType: "AllSame"
			ReferenceInformationType: "Direct"
			Colors: *4 {
				a: 0,0,1,0.5
			}
		}
		LayerElementUV: 0 {"#;
    let text = z_up_ascii_fbx().replacen("\t\tLayerElementUV: 0 {", layers, 1);
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("layers.fbx");
    std::fs::write(&p, text).unwrap();
    let mesh = flat(&p, dir.path());
    let prim = &mesh.prims[0];
    let n = prim.positions.len();
    assert_eq!(prim.uvs.len(), n);
    assert_eq!(prim.uv_sets.len(), 2, "slot 0 is `uvs`, slot 1 the second layer");
    assert!(prim.uv_sets[0].is_empty(), "slot 0 is `uvs`");
    assert_eq!(prim.uv_sets[1].len(), n);
    for (a, b) in prim.uvs.iter().zip(&prim.uv_sets[1]) {
        assert_eq!(
            [a[0] * 2.0, 1.0 - (1.0 - a[1]) * 2.0],
            *b,
            "V flipped, like the first set"
        );
    }
    assert_eq!(prim.colors, vec![[1.0, 0.0, 0.0, 1.0]; n]);
    assert_eq!(prim.color_sets.len(), 2);
    assert!(prim.color_sets[0].is_empty(), "slot 0 is `colors`");
    assert_eq!(prim.color_sets[1], vec![[0.0, 0.0, 1.0, 0.5]; n]);
    // The tangent is made perpendicular to the normal (+Z in the file, +Y here), and the
    // binormal on the far side of cross(normal, tangent) makes it left-handed.
    assert_eq!(prim.tangents.len(), n);
    for (t, nrm) in prim.tangents.iter().zip(&prim.normals) {
        assert!(
            (t[0] - 1.0).abs() < 1e-6 && t[1].abs() < 1e-6 && t[2].abs() < 1e-6,
            "{t:?}"
        );
        assert_eq!(t[3], -1.0);
        assert!(nrm[1] > 0.99);
    }
    // Written out, every set survives, numbered from 0 without gaps.
    let glb = mesh.to_glb().unwrap();
    let memory = crate::assets::Memory::new();
    let doc = crate::formats::load_bytes(&glb, "x.glb", "glb", &memory, Keep::ALL).unwrap();
    let again = doc.flatten(Pose::Rest).unwrap().unwrap();
    let back = &again.prims[0];
    assert_eq!(back.uvs, prim.uvs);
    assert_eq!(back.uv_sets[1], prim.uv_sets[1]);
    assert_eq!(back.color_sets[1], prim.color_sets[1]);
    assert_eq!(back.tangents, prim.tangents);
}

#[test]
fn object_ids_past_2_pow_53_stay_distinct() {
    // As doubles these are one number, so the model and its material would share a key.
    let (model, material) = ("9007199254740992", "9007199254740993");
    let text = z_up_ascii_fbx()
        .replace("Model: 20,", &format!("Model: {model},"))
        .replace("Material: 30,", &format!("Material: {material},"))
        .replace("C: \"OO\",10,20", &format!("C: \"OO\",10,{model}"))
        .replace("C: \"OO\",20,0", &format!("C: \"OO\",{model},0"))
        .replace("C: \"OO\",30,20", &format!("C: \"OO\",{material},{model}"))
        .replace("C: \"OP\",40,30,", &format!("C: \"OP\",40,{material},"));
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("big_ids.fbx");
    std::fs::write(&p, text).unwrap();
    let mesh = flat(&p, dir.path());
    assert_eq!(mesh.triangles(), 2);
    assert!(
        mesh.prims[0].positions.iter().all(|v| (v[1] - 1.0).abs() < 1e-5),
        "placed by its model"
    );
    assert_eq!(mesh.materials[mesh.prims[0].material].name, "Stone");
}

fn curve(values: &[f64]) -> Curve {
    #[allow(clippy::cast_precision_loss, reason = "a few keys")]
    let times = (0..values.len()).map(|i| i as f64 / 30.0).collect();
    Curve::linear(times, values.to_vec())
}

#[test]
fn a_turn_past_180_baked_as_its_other_side_goes_the_short_way() {
    // A sway through 180 (a model turned round), written as -179 after 179, and back.
    let mut c = curve(&[175.0, 177.0, 179.0, -179.0, -177.0, -179.0, 179.0, 177.0]);
    c.unwrap_degrees();
    assert_eq!(c.values, [175.0, 177.0, 179.0, 181.0, 183.0, 181.0, 179.0, 177.0]);
    // Past 180 twice over, on a curve that steps a degree at a time.
    let mut c = curve(&[0.0, 1.0, 2.0, 361.0, 362.0, 1.0, 0.0]);
    c.unwrap_degrees();
    assert_eq!(c.values, [0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0]);
}

#[test]
fn a_limb_raised_past_straight_up_keeps_turning_the_short_way() {
    // An arm raised from 70° through straight up to 110° about Y (XYZ order), baked past 90 as the
    // other set of three: X and Z half a turn on, Y mirrored (Kenney's kicks are written so).
    let keys = [
        [10.0, 70.0, 20.0],
        [10.0, 85.0, 20.0],
        [-170.0, 80.0, -160.0],
        [-170.0, 70.0, -160.0],
    ];
    let mut a = Animated {
        curves: std::array::from_fn(|k| Some(curve(&keys.map(|v| v[k])))),
        fixed: [0.0; 3],
    };
    a.unwrap_rotation(0);
    let got: Vec<[f64; 3]> = (0..4)
        .map(|i| std::array::from_fn(|k| a.curves[k].as_ref().unwrap().values[i]))
        .collect();
    let want = [
        [10.0, 70.0, 20.0],
        [10.0, 85.0, 20.0],
        [10.0, 100.0, 20.0],
        [10.0, 110.0, 20.0],
    ];
    for (g, w) in got.iter().zip(&want) {
        assert!((0..3).all(|k| (g[k] - w[k]).abs() < 1e-9), "{got:?}");
    }
}

#[test]
fn a_whole_turn_keyed_as_one_still_spins() {
    // A coin keyed at 0, 360 and 720: each step is the curve's usual one.
    let mut c = curve(&[0.0, 360.0, 720.0]);
    c.unwrap_degrees();
    assert_eq!(c.values, [0.0, 360.0, 720.0]);
    let mut c = curve(&[0.0, 360.0]);
    c.unwrap_degrees();
    assert_eq!(c.values, [0.0, 360.0]);
    let mut a = rotation(&[[0.0, 0.0, 0.0], [0.0, 360.0, 0.0], [0.0, 720.0, 0.0]]);
    a.unwrap_rotation(0);
    assert_eq!(keys(&a), [[0.0, 0.0, 0.0], [0.0, 360.0, 0.0], [0.0, 720.0, 0.0]]);
}

/// An animated rotation with a curve on every axis, one key per row (a thirtieth of a second apart).
fn rotation(rows: &[[f64; 3]]) -> Animated {
    Animated {
        curves: std::array::from_fn(|k| Some(curve(&rows.iter().map(|v| v[k]).collect::<Vec<_>>()))),
        fixed: [0.0; 3],
    }
}

/// The angles at each key: an axis without a curve holds its fixed value.
fn keys(a: &Animated) -> Vec<[f64; 3]> {
    let n = a.curves.iter().flatten().map(|c| c.values.len()).max().unwrap_or(0);
    (0..n)
        .map(|i| std::array::from_fn(|k| a.curves[k].as_ref().map_or(a.fixed[k], |c| c.values[i])))
        .collect()
}

fn same_rotation(a: [f64; 3], b: [f64; 3], order: i32) -> bool {
    let (m, n) = (rotate(a, order), rotate(b, order));
    (0..3).all(|r| (0..3).all(|c| (m[r][c] - n[r][c]).abs() < 1e-6))
}

fn near(got: &[[f64; 3]], want: &[[f64; 3]]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| (0..3).all(|k| (g[k] - w[k]).abs() < 1e-6))
}

/// An angle as an exporter writes it: in (-180, 180].
fn baked(a: f64) -> f64 {
    let w = (a + 180.0).rem_euclid(360.0) - 180.0;
    if w <= -180.0 { w + 360.0 } else { w }
}

/// The other angles making the same rotation: the order's middle axis mirrored, the others half a turn on.
fn flipped(v: [f64; 3], order: i32) -> [f64; 3] {
    let middle = middle_axis(order);
    std::array::from_fn(|k| if k == middle { 180.0 - v[k] } else { v[k] + 180.0 })
}

/// The axis applied second: XYZ and ZYX turn about Y in the middle, XZY and YZX about Z, YXZ and ZXY about X.
fn middle_axis(order: i32) -> usize {
    match order {
        1 | 2 => 2,
        3 | 4 => 0,
        _ => 1,
    }
}

#[test]
fn a_limb_raised_past_straight_up_turns_the_short_way_in_every_rotation_order() {
    for order in 0..=5 {
        let middle = middle_axis(order);
        // The middle axis swings from 60° through straight up to 120°, the others hold; past 90 the
        // exporter writes the other set of three.
        let truth: Vec<[f64; 3]> = [60.0, 70.0, 80.0, 95.0, 105.0, 120.0, 105.0, 85.0]
            .iter()
            .map(|&m| std::array::from_fn(|k| if k == middle { m } else { [10.0, -30.0, 20.0][k] }))
            .collect();
        let written: Vec<[f64; 3]> = truth
            .iter()
            .map(|v| {
                if v[middle] > 90.0 {
                    flipped(*v, order).map(baked)
                } else {
                    *v
                }
            })
            .collect();
        for (t, w) in truth.iter().zip(&written) {
            assert!(
                same_rotation(*t, *w, order),
                "order {order}: the test's flip makes the same rotation"
            );
        }
        let mut a = rotation(&written);
        a.unwrap_rotation(order);
        assert!(near(&keys(&a), &truth), "order {order}: {:?}", keys(&a));
    }
}

#[test]
fn a_steady_spin_baked_into_half_turns_either_side_keeps_turning_one_way() {
    for step in [30.0, -30.0, 45.0, -7.5] {
        let truth: Vec<[f64; 3]> = (0..40).map(|i| [0.0, 0.0, 15.0 + step * f64::from(i)]).collect();
        let written: Vec<[f64; 3]> = truth.iter().map(|v| v.map(baked)).collect();
        // The spin about Z alone, with no curves on X and Y.
        let mut a = Animated {
            curves: [
                None,
                None,
                Some(curve(&written.iter().map(|v| v[2]).collect::<Vec<_>>())),
            ],
            fixed: [0.0; 3],
        };
        a.unwrap_rotation(0);
        assert!(near(&keys(&a), &truth), "{step}° a key: {:?}", keys(&a));
        assert!(
            a.curves[0].is_none() && a.curves[1].is_none(),
            "axes that keep their angle get no curve"
        );
        // Each curve on its own turns the short way too.
        let mut c = curve(&written.iter().map(|v| v[2]).collect::<Vec<_>>());
        c.unwrap_degrees();
        assert!(
            c.values.iter().zip(&truth).all(|(g, w)| (g - w[2]).abs() < 1e-9),
            "{step}°: {:?}",
            c.values
        );
    }
}

#[test]
fn an_axis_without_a_curve_gets_one_when_a_flip_turns_it() {
    // Y holds 80 with no curve of its own while X and Z, turning a little each frame, flip half a turn:
    // the rotation the keys make is Y going on to 100, so Y needs keys.
    let written = [
        [10.0, 80.0, 20.0],
        [12.0, 80.0, 22.0],
        [-166.0, 80.0, -156.0],
        [-164.0, 80.0, -154.0],
    ];
    let mut a = Animated {
        curves: [
            Some(curve(&written.map(|v| v[0]))),
            None,
            Some(curve(&written.map(|v| v[2]))),
        ],
        fixed: [0.0, 80.0, 0.0],
    };
    a.unwrap_rotation(0);
    let want = [
        [10.0, 80.0, 20.0],
        [12.0, 80.0, 22.0],
        [14.0, 100.0, 24.0],
        [16.0, 100.0, 26.0],
    ];
    assert!(near(&keys(&a), &want), "{:?}", keys(&a));
    let y = a.curves[1].as_ref().expect("Y has keys now");
    assert_eq!(y.times, a.curves[0].as_ref().unwrap().times);
}

#[test]
fn a_cut_or_a_fast_spin_is_kept_as_keyed() {
    // A cut: a degree a frame, then a jump to somewhere else entirely. The nearest angles making that
    // rotation are the ones written.
    let cut = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [150.0, 0.0, 0.0],
        [151.0, 0.0, 0.0],
    ];
    let mut a = rotation(&cut);
    a.unwrap_rotation(0);
    assert_eq!(keys(&a), cut);
    // A propeller keyed every third of a turn: every step is its usual one.
    let spin: Vec<[f64; 3]> = (0..6).map(|i| [0.0, 120.0 * f64::from(i), 0.0]).collect();
    let mut a = rotation(&spin);
    a.unwrap_rotation(0);
    assert_eq!(keys(&a), spin);
    let mut c = curve(&spin.iter().map(|v| v[1]).collect::<Vec<_>>());
    c.unwrap_degrees();
    assert_eq!(c.values, spin.iter().map(|v| v[1]).collect::<Vec<_>>());
}

#[test]
fn keys_a_hair_apart_become_one_time() {
    // The same frame written to two curves through two float paths, 6e-8 s apart.
    let t = timeline(vec![
        0.0,
        0.666_666_746,
        0.666_666_687,
        1.0,
        f64::NAN,
        1.0 + 4e-7,
        1.0 + 8e-7,
    ]);
    assert_eq!(t, [0.0, 0.666_666_746, 1.0 + 8e-7]);
    // Keys a frame apart stay apart, even at 1000 frames a second.
    assert_eq!(timeline(vec![0.002, 0.001, 0.0]), [0.0, 0.001, 0.002]);
}

#[test]
fn curves_keyed_at_different_times_each_turn_the_short_way() {
    let mut x = curve(&[30.0, 30.0]);
    x.times = vec![0.0, 0.5];
    let z = curve(&[150.0, 165.0, 180.0, -165.0, -150.0]);
    let mut a = Animated {
        curves: [Some(x), None, Some(z)],
        fixed: [0.0; 3],
    };
    a.unwrap_rotation(0);
    assert_eq!(a.curves[0].as_ref().unwrap().values, [30.0, 30.0]);
    assert_eq!(
        a.curves[2].as_ref().unwrap().values,
        [150.0, 165.0, 180.0, 195.0, 210.0]
    );
}

#[test]
fn odd_rotation_curves_are_left_alone() {
    let cases: [&[f64]; 6] = [
        &[42.0],
        &[5.0, 5.0, 5.0],
        &[0.0, f64::NAN, 10.0],
        &[0.0, f64::INFINITY, 0.0],
        &[f64::NEG_INFINITY, f64::INFINITY],
        &[1e300, -1e300, 1e300],
    ];
    for values in cases {
        let mut c = curve(values);
        c.unwrap_degrees();
        assert_eq!(format!("{:?}", c.values), format!("{values:?}"));
        let rows: Vec<[f64; 3]> = values.iter().map(|&v| [v, 0.0, 0.0]).collect();
        for order in [0, 3, 6, -1, 99] {
            let mut a = rotation(&rows);
            a.unwrap_rotation(order);
            assert_eq!(format!("{:?}", keys(&a)), format!("{rows:?}"), "order {order}");
        }
    }
    let mut none = Animated {
        curves: [None, None, None],
        fixed: [1.0, 2.0, 3.0],
    };
    none.unwrap_rotation(0);
    assert!(none.curves.iter().all(Option::is_none));
}

#[test]
fn turning_keys_the_short_way_never_moves_a_key() {
    // Whatever angles the filter picks, each key makes the rotation it was written as: random keys in any
    // order, some wild, some with axes missing or keyed at other times.
    let mut rng = crate::test_rng::Pcg32::new(11);
    let angle = |rng: &mut crate::test_rng::Pcg32| f64::from(rng.f32()) * 1440.0 - 720.0;
    for trial in 0..500 {
        let order = [0, 1, 2, 3, 4, 5, 6, 42][rng.below(8) as usize];
        let n = 1 + rng.below(24) as usize;
        let mut a = Animated {
            curves: [None, None, None],
            fixed: std::array::from_fn(|_| angle(&mut rng)),
        };
        let mut last = [0.0; 3];
        for (k, last) in last.iter_mut().enumerate() {
            if rng.below(4) == 0 {
                continue;
            }
            let values = (0..n)
                .map(|_| {
                    if rng.below(3) == 0 {
                        return angle(&mut rng);
                    }
                    *last += f64::from(rng.f32()) * 20.0 - 10.0;
                    if rng.below(5) == 0 { baked(*last) } else { *last }
                })
                .collect::<Vec<_>>();
            let mut c = curve(&values);
            if rng.below(6) == 0 {
                c.times.iter_mut().for_each(|t| *t *= 2.0);
            }
            a.curves[k] = Some(c);
        }
        let before = keys(&a);
        let shared = a.curves.iter().flatten().all(|c| {
            let first = a.curves.iter().flatten().next().unwrap();
            c.times == first.times
        });
        a.unwrap_rotation(order);
        let after = keys(&a);
        if shared {
            for (i, (b, g)) in before.iter().zip(&after).enumerate() {
                assert!(
                    same_rotation(*b, *g, order),
                    "trial {trial}, key {i}: {b:?} became {g:?}"
                );
            }
        } else {
            for (b, g) in before.iter().flatten().zip(after.iter().flatten()) {
                let turns = (g - b) / 360.0;
                assert!((turns - turns.round()).abs() < 1e-9, "trial {trial}: {b} became {g}");
            }
        }
    }
}

#[test]
fn a_baked_turn_comes_back_as_smooth_as_it_was() {
    // A limb wandering a few degrees a frame for four seconds, written as an exporter bakes it: each angle
    // in (-180, 180], and now and then as the other set of three. The filter gives back a path as smooth
    // as the one wandered, making the same rotation at every key.
    let mut rng = crate::test_rng::Pcg32::new(3);
    for trial in 0..200 {
        let order = rng.below(6).cast_signed();
        let middle = middle_axis(order);
        let mut v: [f64; 3] = std::array::from_fn(|_| f64::from(rng.f32()) * 360.0 - 180.0);
        v[middle] = f64::from(rng.f32()) * 120.0 - 60.0;
        let mut truth = Vec::new();
        for _ in 0..120 {
            truth.push(v);
            for a in &mut v {
                *a += f64::from(rng.f32()) * 8.0 - 4.0;
            }
            v[middle] = v[middle].clamp(-60.0, 60.0);
        }
        let written: Vec<[f64; 3]> = truth
            .iter()
            .map(|t| if rng.below(10) == 0 { flipped(*t, order) } else { *t }.map(baked))
            .collect();
        let mut a = rotation(&written);
        a.unwrap_rotation(order);
        let got = keys(&a);
        for (i, w) in got.windows(2).enumerate() {
            let step = (0..3).map(|k| (w[1][k] - w[0][k]).abs()).fold(0.0, f64::max);
            assert!(
                step <= 4.0 + 1e-6,
                "trial {trial} (order {order}), key {}: {:?} to {:?}",
                i + 1,
                w[0],
                w[1]
            );
        }
        for (t, g) in truth.iter().zip(&got) {
            assert!(same_rotation(*t, *g, order), "trial {trial}: {t:?} became {g:?}");
        }
    }
}

// --- Files built here as ASCII FBX, read from memory -------------------------------------------------------

/// An ASCII FBX file, Y up in meters, with these objects and connections.
fn ascii(objects: &str, connections: &str) -> String {
    format!(
        r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {{
	FBXHeaderVersion: 1003
	FBXVersion: 7400
}}
GlobalSettings:  {{
	Version: 1000
	Properties70:  {{
		P: "UnitScaleFactor", "double", "Number", "",100
	}}
}}
Objects:  {{
{objects}
}}
Connections:  {{
{connections}
}}
"#
    )
}

/// A model with these `Properties70` lines.
fn model(id: i64, name: &str, props: &str) -> String {
    format!("\tModel: {id}, \"Model::{name}\", \"Mesh\" {{\n\t\tProperties70:  {{\n{props}\n\t\t}}\n\t}}\n")
}

/// A property line.
fn prop(name: &str, values: &str) -> String {
    format!("\t\t\tP: \"{name}\", \"{name}\", \"\", \"A\",{values}\n")
}

/// Geometry 10: a unit quad facing +Z, with `layers` inside.
fn quad(layers: &str) -> String {
    format!(
        "\tGeometry: 10, \"Geometry::Quad\", \"Mesh\" {{\n\t\tVertices: *12 {{\n\t\t\ta: 0,0,0,1,0,0,1,1,0,0,1,0\n\t\t}}\n\t\tPolygonVertexIndex: *4 {{\n\t\t\ta: 0,1,2,-4\n\t\t}}\n{layers}\t}}\n"
    )
}

/// A video with an embedded PNG.
fn video(id: i64) -> String {
    use base64::Engine as _;
    let png = base64::engine::general_purpose::STANDARD.encode(png([10, 200, 30]));
    format!("\tVideo: {id}, \"Video::V{id}\", \"Clip\" {{\n\t\tContent: , \"{png}\"\n\t}}\n")
}

/// Connection lines: `(child, parent)` object links, and `(child, parent, property)` property links.
fn links(oo: &[(i64, i64)], op: &[(i64, i64, &str)]) -> String {
    let oo = oo.iter().map(|(c, p)| format!("\tC: \"OO\",{c},{p}\n"));
    let op = op
        .iter()
        .map(|(c, p, name)| format!("\tC: \"OP\",{c},{p}, \"{name}\"\n"));
    oo.chain(op).collect()
}

fn read(text: &str) -> Document {
    let memory = crate::assets::Memory::new();
    crate::formats::load_bytes(text.as_bytes(), "t.fbx", "fbx", &memory, Keep::ALL).unwrap()
}

fn index(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}

/// A float accessor's numbers.
fn accessor(doc: &Document, i: &Value) -> Vec<f32> {
    let a = &doc.json()["accessors"][index(i)];
    let view = &doc.json()["bufferViews"][index(&a["bufferView"])];
    let bytes = doc.buffer(index(&view["buffer"])).unwrap();
    let offset = |v: &Value| usize::try_from(v.as_u64().unwrap_or(0)).unwrap();
    let at = offset(&view["byteOffset"]) + offset(&a["byteOffset"]);
    let width = match a["type"].as_str().unwrap() {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        _ => 16,
    };
    bytes[at..][..index(&a["count"]) * width * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The node named `name`.
fn node<'d>(doc: &'d Document, name: &str) -> &'d Value {
    doc.json()["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["name"] == name)
        .unwrap_or_else(|| panic!("no node {name}"))
}

fn vector(v: &Value, default: &[f64]) -> Vec<f64> {
    v.as_array()
        .map_or_else(|| default.to_vec(), |a| a.iter().map(|x| x.as_f64().unwrap()).collect())
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

fn visible(n: &Value) -> bool {
    n["extensions"]["KHR_node_visibility"]["visible"] != false
}

#[test]
fn hidden_models_and_coarser_levels_of_detail_are_kept_hidden() {
    let objects = [
        quad(""),
        model(20, "Shown", ""),
        model(21, "Gone", &prop("Visibility", "0")),
        model(22, "Lods", ""),
        "\tNodeAttribute: 60, \"NodeAttribute::\", \"LodGroup\" {\n\t}\n".to_owned(),
        model(23, "Fine", ""),
        model(24, "Coarse", ""),
        model(25, "UnderGone", ""),
    ]
    .concat();
    let doc = read(&ascii(
        &objects,
        &links(
            &[
                (10, 20),
                (10, 23),
                (10, 24),
                (20, 0),
                (21, 0),
                (22, 0),
                (60, 22),
                (23, 22),
                (24, 22),
                (25, 21),
            ],
            &[],
        ),
    ));
    assert!(visible(node(&doc, "Shown")));
    assert!(!visible(node(&doc, "Gone")));
    assert!(visible(node(&doc, "UnderGone")), "hidden by its parent, not itself");
    assert!(visible(node(&doc, "Fine")), "the finest level shows");
    assert!(!visible(node(&doc, "Coarse")));
    assert!(
        doc.json()["extensionsUsed"]
            .as_array()
            .unwrap()
            .contains(&json!("KHR_node_visibility"))
    );
}

#[test]
fn rotation_active_off_ignores_pre_rotation_and_order() {
    let pre = prop("PreRotation", "90,0,0");
    let off = pre.clone() + &prop("RotationActive", "0");
    let objects = [model(20, "On", &pre), model(21, "Off", &off)].concat();
    let doc = read(&ascii(&objects, &links(&[(20, 0), (21, 0)], &[])));
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert!(close(
        &vector(&node(&doc, "On")["rotation"], &[0.0, 0.0, 0.0, 1.0]),
        &[h, 0.0, 0.0, h]
    ));
    assert!(close(
        &vector(&node(&doc, "Off")["rotation"], &[0.0, 0.0, 0.0, 1.0]),
        &[0.0, 0.0, 0.0, 1.0]
    ));
}

#[test]
fn a_child_that_ignores_its_parents_scale_is_not_scaled_by_it() {
    let objects = [
        model(20, "Parent", &prop("Lcl Scaling", "2,2,2")),
        model(
            21,
            "Child",
            &(prop("InheritType", "2") + &prop("Lcl Translation", "1,0,0")),
        ),
    ]
    .concat();
    let doc = read(&ascii(&objects, &links(&[(20, 0), (21, 20)], &[])));
    let child = node(&doc, "Child");
    assert!(close(&vector(&child["scale"], &[1.0; 3]), &[0.5; 3]), "{child}");
    assert!(
        close(&vector(&child["translation"], &[0.0; 3]), &[1.0, 0.0, 0.0]),
        "{child}"
    );
}

#[test]
fn properties_a_model_leaves_out_come_from_its_template() {
    let definitions = "Definitions:  {\n\tObjectType: \"Model\" {\n\t\tPropertyTemplate: \"FbxNode\" {\n\t\t\tProperties70:  {\n\t\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,5,0\n\t\t\t}\n\t\t}\n\t}\n}\nObjects:  {";
    let objects = [
        model(20, "Templated", ""),
        model(21, "Own", &prop("Lcl Translation", "1,0,0")),
    ]
    .concat();
    let text = ascii(&objects, &links(&[(20, 0), (21, 0)], &[])).replacen("Objects:  {", definitions, 1);
    let doc = read(&text);
    assert!(close(
        &vector(&node(&doc, "Templated")["translation"], &[0.0; 3]),
        &[0.0, 5.0, 0.0]
    ));
    assert!(close(
        &vector(&node(&doc, "Own")["translation"], &[0.0; 3]),
        &[1.0, 0.0, 0.0]
    ));
}

#[test]
fn transforms_that_overflow_or_are_not_numbers_never_write_null() {
    let objects = [
        model(20, "Infinite", &prop("Lcl Translation", "1e400,0,0")),
        model(
            21,
            "Overflowing",
            &(prop("Lcl Scaling", "1e200,1e200,1e200") + &prop("ScalingPivot", "1e200,1e200,1e200")),
        ),
    ]
    .concat();
    let doc = read(&ascii(&objects, &links(&[(20, 0), (21, 0)], &[])));
    let text = serde_json::to_string(&doc.json()).unwrap();
    assert!(!text.contains("null"), "{text}");
}

#[test]
fn a_deep_chain_listed_leaf_first_loads_quickly() {
    const N: i64 = 20_000;
    use std::fmt::Write as _;
    let (mut objects, mut connections) = (String::new(), String::new());
    for i in 0..N {
        let parent = if i + 1 == N { 0 } else { 101 + i };
        writeln!(objects, "\tModel: {}, \"Model::M{i}\", \"Null\" {{\n\t}}", 100 + i).unwrap();
        writeln!(connections, "\tC: \"OO\",{},{parent}", 100 + i).unwrap();
    }
    let text = ascii(&objects, &connections);
    let start = std::time::Instant::now();
    let doc = read(&text);
    let took = start.elapsed();
    // The models, and the root node over them.
    assert_eq!(doc.json()["nodes"].as_array().unwrap().len(), 20_001);
    assert!(took < std::time::Duration::from_secs(10), "took {took:?}");
}

/// The flattened mesh's normals at the corners at `p`.
fn normals_at(mesh: &Mesh, p: [f32; 3]) -> Vec<[f32; 3]> {
    let prim = &mesh.prims[0];
    prim.positions
        .iter()
        .zip(&prim.normals)
        .filter(|(q, _)| (0..3).all(|k| (q[k] - p[k]).abs() < 1e-5))
        .map(|(_, n)| *n)
        .collect()
}

/// A wall (facing +Z) and a floor (facing +Y) meeting along the edge from the origin to +X, with `layers`.
fn fold(layers: &str) -> Mesh {
    let geometry = format!(
        "\tGeometry: 10, \"Geometry::Fold\", \"Mesh\" {{\n\t\tVertices: *18 {{\n\t\t\ta: 0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0,0,1\n\t\t}}\n\t\tPolygonVertexIndex: *8 {{\n\t\t\ta: 0,1,2,-4,1,0,5,-5\n\t\t}}\n\t\tEdges: *7 {{\n\t\t\ta: 0,1,2,3,5,6,7\n\t\t}}\n{layers}\t}}\n"
    );
    let text = ascii(&(geometry + &model(20, "Fold", "")), &links(&[(10, 20), (20, 0)], &[]));
    read(&text).flatten(Pose::Rest).unwrap().unwrap()
}

fn smoothing_layer(mapping: &str, values: &str) -> String {
    let n = values.split(',').count();
    format!(
        "\t\tLayerElementSmoothing: 0 {{\n\t\t\tMappingInformationType: \"{mapping}\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tSmoothing: *{n} {{\n\t\t\t\ta: {values}\n\t\t\t}}\n\t\t}}\n"
    )
}

fn all_near(normals: &[[f32; 3]], want: [f32; 3]) -> bool {
    !normals.is_empty() && normals.iter().all(|n| (0..3).all(|k| (n[k] - want[k]).abs() < 1e-4))
}

#[test]
fn smoothing_groups_by_polygon_make_the_normals() {
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let shared = fold(&smoothing_layer("ByPolygon", "1,1"));
    assert!(
        all_near(&normals_at(&shared, [0.0; 3]), [0.0, h, h]),
        "{:?}",
        shared.prims[0].normals
    );
    assert!(
        all_near(&normals_at(&shared, [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]),
        "a corner of the wall alone"
    );
    for apart in ["1,2", "0,0"] {
        let mesh = fold(&smoothing_layer("ByPolygon", apart));
        let at = normals_at(&mesh, [0.0; 3]);
        assert_eq!(at.len(), 2, "{apart}: {at:?}");
        assert!(
            at.iter().any(|n| n[2] > 0.99) && at.iter().any(|n| n[1] > 0.99),
            "{apart}: {at:?}"
        );
    }
}

#[test]
fn smoothing_by_edge_splits_the_normals_at_hard_edges() {
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let soft = fold(&smoothing_layer("ByEdge", "1,1,1,1,1,1,1"));
    assert!(
        all_near(&normals_at(&soft, [1.0, 0.0, 0.0]), [0.0, h, h]),
        "{:?}",
        soft.prims[0].normals
    );
    let hard = fold(&smoothing_layer("ByEdge", "0,1,1,1,1,1,1"));
    let at = normals_at(&hard, [1.0, 0.0, 0.0]);
    assert!(
        at.iter().any(|n| n[2] > 0.99) && at.iter().any(|n| n[1] > 0.99),
        "{at:?}"
    );
}

#[test]
fn holes_are_left_out() {
    let hole = "\t\tLayerElementHole: 0 {\n\t\t\tMappingInformationType: \"ByPolygon\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tHole: *2 {\n\t\t\t\ta: 0,1\n\t\t\t}\n\t\t}\n";
    let mesh = fold(hole);
    assert_eq!(mesh.triangles(), 2);
    assert!(
        mesh.prims[0].positions.iter().all(|p| p[2].abs() < 1e-6),
        "only the wall is left"
    );
}

#[test]
fn textures_read_their_uv_set_by_name() {
    let uv = |n: u32, name: &str, values: &str| {
        format!(
            "\t\tLayerElementUV: {n} {{\n\t\t\tName: \"{name}\"\n\t\t\tMappingInformationType: \"ByPolygonVertex\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tUV: *8 {{\n\t\t\t\ta: {values}\n\t\t\t}}\n\t\t}}\n"
        )
    };
    let layers = uv(0, "map1", "0,0,1,0,1,1,0,1") + &uv(1, "detail", "0,0,2,0,2,2,0,2");
    let texture = |id: i64, set: &str| {
        format!(
            "\tTexture: {id}, \"Texture::T{id}\", \"\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"UVSet\", \"KString\", \"\", \"\", \"{set}\"\n\t\t}}\n\t}}\n"
        )
    };
    let objects = [
        quad(&layers),
        model(20, "Quad", ""),
        "\tMaterial: 30, \"Material::M\", \"\" {\n\t}\n".to_owned(),
        texture(40, "detail"),
        texture(41, "map1"),
        video(50),
    ]
    .concat();
    let doc = read(&ascii(
        &objects,
        &links(
            &[(10, 20), (20, 0), (30, 20), (50, 40), (50, 41)],
            &[(40, 30, "DiffuseColor"), (41, 30, "NormalMap")],
        ),
    ));
    let mat = &doc.json()["materials"][0];
    assert_eq!(
        mat["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"]
            .as_u64()
            .unwrap_or(0),
        0
    );
    assert_eq!(mat["normalTexture"]["texCoord"], 1, "{mat}");
    let attributes = &doc.json()["meshes"][0]["primitives"][0]["attributes"];
    let max_u = |set: &str| {
        accessor(&doc, &attributes[set])
            .chunks_exact(2)
            .map(|t| t[0])
            .fold(0.0, f32::max)
    };
    assert!(
        (max_u("TEXCOORD_0") - 2.0).abs() < 1e-6,
        "the color texture's set comes first"
    );
    assert!((max_u("TEXCOORD_1") - 1.0).abs() < 1e-6);
}

/// The glTF material made from a material with these properties and `(texture id, property)` links, each
/// texture showing video 50.
fn material(props: &str, textures: &[(i64, &str)]) -> Value {
    let mut objects = [
        quad(""),
        model(20, "Quad", ""),
        format!("\tMaterial: 30, \"Material::M\", \"\" {{\n\t\tProperties70:  {{\n{props}\n\t\t}}\n\t}}\n"),
        video(50),
    ]
    .concat();
    let mut oo = vec![(10, 20), (20, 0), (30, 20)];
    for &(id, _) in textures {
        objects.extend(format!("\tTexture: {id}, \"Texture::T{id}\", \"\" {{\n\t}}\n").chars());
        oo.push((50, id));
    }
    let op: Vec<(i64, i64, &str)> = textures.iter().map(|&(id, p)| (id, 30, p)).collect();
    let doc = read(&ascii(&objects, &links(&oo, &op)));
    doc.json()["materials"][0].clone()
}

fn number(v: &Value, default: f64) -> f64 {
    v.as_f64().unwrap_or(default)
}

#[test]
fn a_standard_surface_reads_as_metal_rough() {
    let props = [
        prop("Maya|baseColor", "1,1,1"),
        prop("Maya|base", "0.5"),
        prop("Maya|metalness", "1"),
        prop("Maya|specularRoughness", "0.25"),
    ]
    .concat();
    let m = material(&props, &[]);
    let pbr = &m["pbrMetallicRoughness"];
    assert!(
        close(&vector(&pbr["baseColorFactor"], &[1.0; 4])[..3], &[0.5; 3]),
        "{m}"
    );
    assert!((number(&pbr["metallicFactor"], 1.0) - 1.0).abs() < 1e-6, "{m}");
    assert!((number(&pbr["roughnessFactor"], 1.0) - 0.25).abs() < 1e-6, "{m}");
}

#[test]
fn a_pbr_material_with_inverted_roughness_reads_as_metal_rough() {
    let props = [
        prop("3dsMax|main|basecolor", "1,0,0"),
        prop("3dsMax|main|metalness", "0"),
        prop("3dsMax|main|roughness", "0.2"),
        prop("3dsMax|main|roughness_inv", "1"),
    ]
    .concat();
    let m = material(&props, &[]);
    let pbr = &m["pbrMetallicRoughness"];
    assert!(
        close(&vector(&pbr["baseColorFactor"], &[1.0; 4]), &[1.0, 0.0, 0.0, 1.0]),
        "{m}"
    );
    assert!(number(&pbr["metallicFactor"], 1.0).abs() < 1e-6, "{m}");
    assert!((number(&pbr["roughnessFactor"], 1.0) - 0.8).abs() < 1e-6, "{m}");
}

#[test]
fn a_packed_occlusion_roughness_metal_texture_feeds_both() {
    let props = prop("3dsMax|Parameters|base_color", "1,1,1");
    let m = material(&props, &[(40, "3dsMax|Parameters|ao_roughness_metalness_map")]);
    let mr = &m["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"];
    assert!(mr.is_number(), "{m}");
    assert_eq!(&m["occlusionTexture"]["index"], mr, "{m}");
}

#[test]
fn a_layered_texture_shows_its_first_layer() {
    let layered =
        "\tLayeredTexture: 45, \"LayeredTexture::L\", \"\" {\n\t}\n\tTexture: 46, \"Texture::T46\", \"\" {\n\t}\n";
    // The layered texture holds texture 46, which shows video 50.
    let objects = [
        quad(""),
        model(20, "Quad", ""),
        "\tMaterial: 30, \"Material::M\", \"\" {\n\t}\n".to_owned(),
        video(50),
        layered.to_owned(),
    ]
    .concat();
    let doc = read(&ascii(
        &objects,
        &links(
            &[(10, 20), (20, 0), (30, 20), (46, 45), (50, 46)],
            &[(45, 30, "DiffuseColor")],
        ),
    ));
    assert!(doc.json()["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"].is_number());
}

#[test]
fn in_between_shapes_and_their_normals_become_targets() {
    let normals = "\t\tLayerElementNormal: 0 {\n\t\t\tMappingInformationType: \"ByControlPoint\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tNormals: *12 {\n\t\t\t\ta: 0,0,1,0,0,1,0,0,1,0,0,1\n\t\t\t}\n\t\t}\n";
    let shape = |id: i64, z: f64| {
        format!(
            "\tGeometry: {id}, \"Geometry::S{id}\", \"Shape\" {{\n\t\tIndexes: *1 {{\n\t\t\ta: 2\n\t\t}}\n\t\tVertices: *3 {{\n\t\t\ta: 0,0,{z}\n\t\t}}\n\t\tNormals: *3 {{\n\t\t\ta: 0,0.5,0\n\t\t}}\n\t}}\n"
        )
    };
    let objects = [
        quad(normals),
        model(20, "Quad", ""),
        "\tDeformer: 70, \"Deformer::B\", \"BlendShape\" {\n\t}\n".to_owned(),
        "\tDeformer: 71, \"SubDeformer::Bulge\", \"BlendShapeChannel\" {\n\t\tDeformPercent: 25\n\t\tFullWeights: *2 {\n\t\t\ta: 50,100\n\t\t}\n\t\tProperties70:  {\n\t\t\tP: \"DeformPercent\", \"Number\", \"\", \"A\",25\n\t\t}\n\t}\n".to_owned(),
        shape(80, 1.0),
        shape(81, 4.0),
    ]
    .concat();
    let doc = read(&ascii(
        &objects,
        &links(&[(10, 20), (20, 0), (70, 10), (71, 70), (80, 71), (81, 71)], &[]),
    ));
    let mesh = &doc.json()["meshes"][0];
    let targets = mesh["primitives"][0]["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 2, "{mesh}");
    let reach = |t: &Value, key: &str, k: usize| {
        accessor(&doc, &t[key])
            .chunks_exact(3)
            .map(|v| v[k])
            .fold(0.0, f32::max)
    };
    assert!(
        (reach(&targets[0], "POSITION", 2) - 1.0).abs() < 1e-6,
        "the in-between first"
    );
    assert!((reach(&targets[1], "POSITION", 2) - 4.0).abs() < 1e-6);
    assert!(reach(&targets[0], "NORMAL", 1) > 0.1, "{mesh}");
    // A quarter of the way is half the in-between.
    assert!(close(&vector(&mesh["weights"], &[]), &[0.5, 0.0]), "{mesh}");
}

#[test]
fn a_channel_weight_is_shared_among_its_shapes_and_not_clamped() {
    let mut out = [0.0; 2];
    for (w, want) in [
        (0.25, [0.5, 0.0]),
        (0.75, [0.5, 0.5]),
        (1.0, [0.0, 1.0]),
        (1.5, [-1.0, 2.0]),
        (-0.2, [-0.4, 0.0]),
    ] {
        split_weight(w, &[0.5, 1.0], &mut out);
        assert!(close(&out, &want), "{w}: {out:?}");
    }
    let mut one = [0.0];
    split_weight(1.5, &[1.0], &mut one);
    assert!(close(&one, &[1.5]));
    split_weight(f64::NAN, &[1.0], &mut one);
    assert!(close(&one, &[0.0]));
}

fn cubic(handles: [f64; 4]) -> Curve {
    let mut c = Curve::linear(vec![0.0, 1.0], vec![0.0, 1.0]);
    c.interp[0] = Interp::Cubic;
    c.handles[0] = handles;
    c
}

#[test]
fn cubic_keys_follow_their_slopes_and_handle_lengths() {
    let flat = cubic(FLAT);
    assert!((flat.at(0.5) - 0.5).abs() < 1e-9);
    assert!((flat.at(0.25) - 0.156_25).abs() < 1e-9);
    // Leaving at a slope of two: the first handle at (1/3, 2/3).
    let steep = cubic([2.0, THIRD, 0.0, THIRD]);
    assert!((steep.at(0.5) - 0.75).abs() < 1e-9);
    // Longer handles, the same either side: still half way at half time, but slower to leave.
    let long = cubic([0.0, 0.5, 0.0, 0.5]);
    assert!((long.at(0.5) - 0.5).abs() < 1e-9);
    assert!(long.at(0.25) < flat.at(0.25));
    assert!(long.at(0.0).abs() < 1e-12 && (long.at(1.0) - 1.0).abs() < 1e-12);
    // Steps: held until the next key, or jumping to it at once.
    let mut step = Curve::linear(vec![0.0, 1.0], vec![0.0, 1.0]);
    step.interp[0] = Interp::Constant;
    assert!(step.at(0.99).abs() < 1e-12 && (step.at(1.0) - 1.0).abs() < 1e-12);
    step.interp[0] = Interp::ConstantNext;
    assert!((step.at(0.01) - 1.0).abs() < 1e-12 && step.at(0.0).abs() < 1e-12);
}

#[test]
fn tcb_keys_make_catmull_rom_slopes() {
    let mut c = Curve::linear(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]);
    c.interp.fill(Interp::Cubic);
    c.tcb(&[Some([0.0; 3]); 3]);
    // Out of the first key at its segment's slope (1), into the peak flat.
    assert!((c.at(0.5) - 0.625).abs() < 1e-9, "{}", c.at(0.5));
    assert!((c.at(1.5) - 0.625).abs() < 1e-9, "{}", c.at(1.5));
}

#[test]
fn packed_handle_lengths_read_from_text_and_binary() {
    // 3333 out, 5000 in.
    let packed = f64::from(0x1388_0D05_u32);
    let (out, next_in) = handle_lengths(packed);
    assert!((out - THIRD).abs() < 1e-12 && (next_in - 5000.0 / 9999.0).abs() < 1e-12);
    // 2500 out, 3333 in.
    let bits = f64::from(f32::from_bits(0x0D05_09C4_u32));
    let (out, next_in) = handle_lengths(bits);
    assert!((out - 2500.0 / 9999.0).abs() < 1e-12 && (next_in - THIRD).abs() < 1e-12);
}

#[test]
fn cubic_curves_in_a_file_are_sampled_along_their_shape() {
    // Lcl Translation X eases from 0 to 1 over a second.
    let curve = "\tAnimationStack: 100, \"AnimStack::Take\", \"\" {\n\t}\n\tAnimationLayer: 101, \"AnimLayer::Base\", \"\" {\n\t}\n\tAnimationCurveNode: 102, \"AnimCurveNode::T\", \"\" {\n\t\tProperties70:  {\n\t\t\tP: \"d|X\", \"Number\", \"\", \"A\",0\n\t\t}\n\t}\n\tAnimationCurve: 103, \"AnimCurve::\", \"\" {\n\t\tKeyTime: *2 {\n\t\t\ta: 0,46186158000\n\t\t}\n\t\tKeyValueFloat: *2 {\n\t\t\ta: 0,1\n\t\t}\n\t\tKeyAttrFlags: *1 {\n\t\t\ta: 8\n\t\t}\n\t\tKeyAttrDataFloat: *4 {\n\t\t\ta: 0,0,218434821,0\n\t\t}\n\t\tKeyAttrRefCount: *1 {\n\t\t\ta: 2\n\t\t}\n\t}\n";
    let objects = model(20, "Mover", "") + curve;
    let doc = read(&ascii(
        &objects,
        &links(
            &[(20, 0), (101, 100), (102, 101)],
            &[(102, 20, "Lcl Translation"), (103, 102, "d|X")],
        ),
    ));
    let animation = &doc.json()["animations"][0];
    let sampler = animation["channels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["target"]["path"] == "translation")
        .map(|c| &animation["samplers"][index(&c["sampler"])])
        .unwrap();
    let times = accessor(&doc, &sampler["input"]);
    let values = accessor(&doc, &sampler["output"]);
    assert!(times.len() > 4, "{times:?}");
    let flat = cubic(FLAT);
    for (t, v) in times.iter().zip(values.chunks_exact(3)) {
        assert!((f64::from(v[0]) - flat.at(f64::from(*t))).abs() < 1e-5, "{t}: {v:?}");
    }
}

#[test]
fn a_component_without_a_curve_or_a_value_stays_at_rest() {
    // Only X is animated, and the curve node gives no d|Y or d|Z: Y and Z hold the model's own.
    let curve = "\tAnimationStack: 100, \"AnimStack::Take\", \"\" {\n\t}\n\tAnimationLayer: 101, \"AnimLayer::Base\", \"\" {\n\t}\n\tAnimationCurveNode: 102, \"AnimCurveNode::T\", \"\" {\n\t}\n\tAnimationCurve: 103, \"AnimCurve::\", \"\" {\n\t\tKeyTime: *2 {\n\t\t\ta: 0,46186158000\n\t\t}\n\t\tKeyValueFloat: *2 {\n\t\t\ta: 0,1\n\t\t}\n\t\tKeyAttrFlags: *1 {\n\t\t\ta: 4\n\t\t}\n\t\tKeyAttrDataFloat: *4 {\n\t\t\ta: 0,0,0,0\n\t\t}\n\t\tKeyAttrRefCount: *1 {\n\t\t\ta: 2\n\t\t}\n\t}\n";
    let objects = model(20, "Mover", &prop("Lcl Translation", "5,6,7")) + curve;
    let doc = read(&ascii(
        &objects,
        &links(
            &[(20, 0), (101, 100), (102, 101)],
            &[(102, 20, "Lcl Translation"), (103, 102, "d|X")],
        ),
    ));
    let rest = vector(&node(&doc, "Mover")["translation"], &[0.0; 3]);
    assert!(rest[1].abs() > 1e-3 && rest[2].abs() > 1e-3, "{rest:?}");
    let animation = &doc.json()["animations"][0];
    let sampler = animation["channels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["target"]["path"] == "translation")
        .map(|c| &animation["samplers"][index(&c["sampler"])])
        .unwrap();
    let values = accessor(&doc, &sampler["output"]);
    assert!(values.len() >= 6);
    for v in values.chunks_exact(3) {
        assert!(
            (f64::from(v[1]) - rest[1]).abs() < 1e-6 && (f64::from(v[2]) - rest[2]).abs() < 1e-6,
            "{v:?} {rest:?}"
        );
    }
    assert!((values[0] - values[values.len() - 3]).abs() > 1e-6, "X moves");
}

#[test]
fn transparency_is_the_factor_times_the_color() {
    let alpha = |props: &str| number(&material(props, &[])["pbrMetallicRoughness"]["baseColorFactor"][3], 1.0);
    let p = |name: &str, v: &str| prop(name, v);
    // A factor of 1 over a black (or no) transparent color is opaque: the color is what is transparent.
    assert_eq!(alpha(&p("TransparencyFactor", "1")), 1.0);
    assert_eq!(
        alpha(&(p("TransparencyFactor", "1") + &p("TransparentColor", "0,0,0"))),
        1.0
    );
    assert_eq!(
        alpha(&(p("TransparencyFactor", "0") + &p("TransparentColor", "1,1,1"))),
        1.0
    );
    // The color scaled by the factor.
    let a = alpha(&(p("TransparencyFactor", "1") + &p("TransparentColor", "0.25,0.25,0.25")));
    assert!((a - 0.75).abs() < 1e-6, "{a}");
    let a = alpha(&(p("TransparencyFactor", "0.5") + &p("TransparentColor", "1,1,1")));
    assert!((a - 0.5).abs() < 1e-6, "{a}");
    // A factor between the ends alone is the transparency, as the tools that write only it mean.
    let a = alpha(&p("TransparencyFactor", "0.3"));
    assert!((a - 0.7).abs() < 1e-6, "{a}");
    // Opacity, where the material gives it, is the answer.
    let a = alpha(&(p("Opacity", "0.4") + &p("TransparencyFactor", "1") + &p("TransparentColor", "1,1,1")));
    assert!((a - 0.4).abs() < 1e-6, "{a}");
}

#[test]
fn ascii_fbx_6_without_its_version_comment_is_refused_as_too_old() {
    let old = ascii("", "")
        .replace("; FBX 7.4.0 project file\n", "")
        .replace("FBXVersion: 7400", "FBXVersion: 6100");
    let e = crate::formats::load_bytes(old.as_bytes(), "t.fbx", "fbx", &crate::assets::Memory::new(), Keep::ALL)
        .unwrap_err();
    assert_eq!(e.kind(), crate::ErrorKind::Unsupported, "{e}");
    let with = format!("; FBX 6.1.0 project file\n{old}");
    let e = crate::formats::load_bytes(
        with.as_bytes(),
        "t.fbx",
        "fbx",
        &crate::assets::Memory::new(),
        Keep::ALL,
    )
    .unwrap_err();
    assert_eq!(e.kind(), crate::ErrorKind::Unsupported, "{e}");
}

#[test]
fn layers_blend_by_their_weight_and_mode() {
    let steady = |v: f64| Animated {
        curves: [Some(Curve::linear(vec![0.0, 1.0], vec![v, v])), None, None],
        fixed: [0.0; 3],
    };
    let layered = Layered {
        rest: [0.0; 3],
        layers: vec![
            (Blend::Override, 1.0, steady(1.0)),
            (Blend::Add, 0.5, steady(2.0)),
            (Blend::Override, 0.25, steady(10.0)),
        ],
    };
    // 1, plus half of 2, then a quarter of the way to 10.
    assert!((layered.at(0.5)[0] - 4.0).abs() < 1e-9);
    let scaled = Layered {
        rest: [1.0; 3],
        layers: vec![(Blend::Override, 1.0, steady(2.0)), (Blend::Multiply, 0.5, steady(4.0))],
    };
    assert!((scaled.at(0.5)[0] - 4.0).abs() < 1e-9, "2 times the square root of 4");
}

#[test]
fn area_and_volume_lights_are_kept() {
    let light = |id: i64, kind: u32| {
        format!(
            "\tNodeAttribute: {id}, \"NodeAttribute::L\", \"Light\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"LightType\", \"enum\", \"\", \"\",{kind}\n\t\t\tP: \"DecayType\", \"enum\", \"\", \"\",1\n\t\t}}\n\t}}\n"
        )
    };
    let objects = [
        model(20, "Area", ""),
        model(21, "Volume", ""),
        light(60, 3),
        light(61, 4),
    ]
    .concat();
    let doc = read(&ascii(&objects, &links(&[(20, 0), (21, 0), (60, 20), (61, 21)], &[])));
    let lights = doc.json()["extensions"]["KHR_lights_punctual"]["lights"]
        .as_array()
        .unwrap();
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0]["type"], "spot");
    assert_eq!(lights[0]["extras"]["lightType"], "area");
    assert_eq!(lights[0]["extras"]["decayType"], "linear");
    assert_eq!(lights[1]["type"], "point");
    assert_eq!(lights[1]["extras"]["lightType"], "volume");
}

// --- Lines ---------------------------------------------------------------------------------------------------

/// Geometry 10: a `Line` through `points` (x, y, z each) by `index`, with these `Properties70` lines.
fn line(points: &str, index: &str, props: &str) -> String {
    let count = |a: &str| a.split(',').count();
    format!(
        "\tGeometry: 10, \"Geometry::Wire\", \"Line\" {{\n\t\tProperties70:  {{\n{props}\t\t}}\n\t\tPoints: *{} {{\n\t\t\ta: {points}\n\t\t}}\n\t\tPointsIndex: *{} {{\n\t\t\ta: {index}\n\t\t}}\n\t}}\n",
        count(points),
        count(index)
    )
}

/// The corners of a unit square, then a point off it.
const SQUARE: &str = "0,0,0,1,0,0,1,1,0,0,1,0,5,5,5";

#[test]
fn line_geometry_draws_its_polylines_unlit_in_its_color() {
    // Round three sides of the square to its last corner, written negative, then a polyline from that corner
    // back to the first. The point off the square isn't used.
    let objects = [
        line(SQUARE, "0,1,2,-4,3,-1", &prop("Color", "1,0.5,0")),
        model(20, "Wire", &prop("Lcl Translation", "10,0,0")),
    ]
    .concat();
    let doc = read(&ascii(&objects, &links(&[(10, 20), (20, 0)], &[])));
    let mesh = &doc.json()["meshes"][0];
    assert_eq!(mesh["name"], "Wire");
    assert_eq!(mesh["primitives"].as_array().unwrap().len(), 1);
    assert_eq!(mesh["primitives"][0]["mode"], 1);
    assert_eq!(node(&doc, "Wire")["mesh"], 0);
    let material = &doc.json()["materials"][index(&mesh["primitives"][0]["material"])];
    assert!(material["extensions"]["KHR_materials_unlit"].is_object());
    let color = vector(&material["pbrMetallicRoughness"]["baseColorFactor"], &[]);
    assert!(
        close(&color, &[1.0, f64::from(srgb_to_linear(0.5)), 0.0, 1.0]),
        "{color:?}"
    );
    let m = doc.flatten(Pose::Rest).unwrap().unwrap();
    assert_eq!((m.triangles(), m.lines()), (0, 4));
    assert_eq!(m.prims[0].positions.len(), 4);
    assert!(
        m.prims[0]
            .positions
            .iter()
            .all(|p| (10.0..=11.0).contains(&p[0]) && (0.0..=1.0).contains(&p[1]) && p[2] == 0.0),
        "{:?}",
        m.prims[0].positions
    );
}

#[test]
fn line_indices_that_are_not_points_break_the_polyline() {
    // The sixth point is too far out for a float.
    let points = [SQUARE, "1e300,0,0"].join(",");
    for (index, lines) in [
        // 9 isn't a point: 0 to 1, then 2 to 3 to the fifth point, written negative.
        ("0,1,9,2,3,-5", 3),
        // Negative, past every point: it ends nothing more.
        ("0,1,-2147483648,2,3", 2),
        ("0,5,1,2", 1),
        ("-1,-2,-3", 0),
        ("0,0,1", 1),
    ] {
        let objects = [line(&points, index, ""), model(20, "Wire", "")].concat();
        let doc = read(&ascii(&objects, &links(&[(10, 20), (20, 0)], &[])));
        let drawn = doc.flatten(Pose::Rest).unwrap().map_or(0, |m| m.lines());
        assert_eq!(drawn, lines, "{index}");
        assert_eq!(doc.json().get("meshes").is_some(), lines > 0, "{index}");
    }
    // Indexes written as floats: whole numbers are indexes, other numbers no point's.
    assert_eq!((whole(3.0), whole(-1.0)), (3, -1));
    for f in [0.5, f64::NAN, f64::INFINITY, -1e300] {
        assert_eq!(whole(f), i64::MAX, "{f}");
    }
}

#[test]
fn a_line_takes_its_models_first_material() {
    let material = |id: i64, name: &str| format!("\tMaterial: {id}, \"Material::{name}\", \"\" {{\n\t}}\n");
    let objects = [
        line(SQUARE, "0,1,-3", &prop("Color", "1,0,0")),
        model(20, "Wire", ""),
        material(30, "First"),
        material(31, "Second"),
    ]
    .concat();
    let doc = read(&ascii(&objects, &links(&[(10, 20), (20, 0), (30, 20), (31, 20)], &[])));
    let materials = doc.json()["materials"].as_array().unwrap();
    let used = &materials[index(&doc.json()["meshes"][0]["primitives"][0]["material"])];
    assert_eq!(used["name"], "First");
    assert!(materials.iter().all(|m| m.get("extensions").is_none()));
}

#[test]
fn a_line_no_model_holds_is_drawn_in_its_templates_color_and_other_curves_are_counted() {
    let template = |name: &str, color: &str| {
        format!(
            "\t\tPropertyTemplate: \"{name}\" {{\n\t\t\tProperties70:  {{\n\t\t\t\tP: \"Color\", \"ColorRGB\", \"Color\", \"\",{color}\n\t\t\t}}\n\t\t}}\n"
        )
    };
    let curves = "\tGeometry: 11, \"Geometry::C\", \"NurbsCurve\" {\n\t}\n\tGeometry: 12, \"Geometry::S\", \"NurbsSurface\" {\n\t}\n\tGeometry: 13, \"Geometry::D\", \"NurbsCurve\" {\n\t}\n";
    // A mesh's template isn't a line's, even the only one there is.
    for (templates, want) in [
        (
            template("FbxMesh", "0,0,1") + &template("FbxLine", "0,1,0"),
            [0.0, 1.0, 0.0, 1.0],
        ),
        (template("FbxMesh", "0,0,1"), [1.0; 4]),
    ] {
        let definitions = format!("Definitions:  {{\n\tObjectType: \"Geometry\" {{\n{templates}\t}}\n}}\nObjects:  {{");
        let objects = [line(SQUARE, "0,1,2,-4", ""), curves.to_owned()].concat();
        let doc = read(&ascii(&objects, "").replacen("Objects:  {", &definitions, 1));
        assert_eq!(doc.json()["nodes"][0]["mesh"], 0);
        let material = &doc.json()["materials"][index(&doc.json()["meshes"][0]["primitives"][0]["material"])];
        let color = vector(&material["pbrMetallicRoughness"]["baseColorFactor"], &[]);
        assert!(close(&color, &want), "{color:?}");
        assert_eq!(
            doc.json()["asset"]["extras"]["source"]["skipped"],
            json!({ "NurbsCurve": 2, "NurbsSurface": 1 })
        );
        assert_eq!(doc.flatten(Pose::Rest).unwrap().unwrap().lines(), 3);
    }
}

#[test]
fn what_the_file_says_of_itself_reads_from_ascii_and_binary() {
    use crate::source::{Axis, Fact, Handedness};
    let header = r#"FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7400
	CreationTimeStamp:  {
		Version: 1000
		Year: 2024
		Month: 3
		Day: 9
		Hour: 14
		Minute: 5
		Second: 7
		Millisecond: 0
	}
	Creator: "FBX SDK/FBX Plugins version 2020.3"
	SceneInfo: "SceneInfo::GlobalInfo", "UserData" {
		Type: "UserData"
		Version: 100
		MetaData:  {
			Version: 100
			Title: "Crate"
			Subject: "A wooden crate"
			Author: "Someone"
			Keywords: "prop, wood"
			Revision: "rev. 2"
			Comment: "first pass"
		}
		Properties70:  {
			P: "Original|ApplicationName", "KString", "", "", "Maya"
			P: "Original|ApplicationVersion", "KString", "", "", "2020"
			P: "Original|DateTime_GMT", "DateTime", "", "", "09/03/2024 14:05:07.000"
			P: "LastSaved|ApplicationName", "KString", "", "", "Maya"
			P: "LastSaved|ApplicationVersion", "KString", "", "", "2022"
			P: "LastSaved|DateTime_GMT", "DateTime", "", "", "10/03/2024 08:00:00.250"
		}
	}
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",2
		P: "UpAxisSign", "int", "Integer", "",1
		P: "FrontAxis", "int", "Integer", "",1
		P: "FrontAxisSign", "int", "Integer", "",-1
		P: "CoordAxis", "int", "Integer", "",0
		P: "CoordAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",2.54
		P: "TimeMode", "enum", "", "",11
	}
}
"#;
    let text = ascii(
        &[quad(""), model(20, "Box", "")].concat(),
        &links(&[(10, 20), (20, 0)], &[]),
    );
    let start = text.find("Objects:").unwrap();
    let text = format!("; FBX 7.4.0 project file\n{header}{}", &text[start..]);
    let binary = crate::formats::fbx_ascii::to_binary(text.as_bytes()).unwrap();
    let memory = crate::assets::Memory::new();
    for data in [text.as_bytes(), &binary] {
        let doc = crate::formats::load_bytes(data, "t.fbx", "fbx", &memory, Keep::ALL).unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("FBX"));
        assert_eq!(info.meters_per_unit, Fact::Declared(0.0254));
        assert_eq!(info.unit.as_deref(), Some("inch"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
        assert_eq!(info.forward_axis, Fact::Declared(Axis::NegY));
        assert_eq!(info.handedness, Fact::Declared(Handedness::Right));
        assert_eq!(info.frame_rate, Fact::Declared(24.0));
        assert!(info.converted.units && info.converted.up_axis && !info.converted.handedness);
        assert_eq!(info.tool.as_deref(), Some("Maya 2022"));
        assert_eq!(info.title.as_deref(), Some("Crate"));
        assert_eq!(info.author.as_deref(), Some("Someone"));
        assert_eq!(info.description.as_deref(), Some("A wooden crate"));
        assert_eq!(info.keywords, ["prop", "wood"]);
        assert_eq!(info.comments, ["first pass"]);
        assert_eq!(info.created.as_deref(), Some("2024-03-09T14:05:07"));
        assert_eq!(info.modified.as_deref(), Some("2024-03-10T08:00:00.250Z"));
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["creator"], "FBX SDK/FBX Plugins version 2020.3");
        assert_eq!(source["originalApplication"], "Maya 2020");
        assert_eq!(source["revision"], "rev. 2");
        assert_eq!(source["coordAxis"], "+X");
    }

    // Settings that say nothing: FBX's centimeters and Y up, assumed; the creator is the tool.
    let doc = read(
        &ascii(
            &[quad(""), model(20, "Box", "")].concat(),
            &links(&[(10, 20), (20, 0)], &[]),
        )
        .replace("\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",100\n", ""),
    );
    let info = doc.source_info();
    assert_eq!(info.meters_per_unit, Fact::Assumed(0.01));
    assert_eq!(info.unit.as_deref(), Some("centimeter"));
    assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
    assert_eq!(info.handedness, Fact::Assumed(Handedness::Right));
    assert_eq!(info.frame_rate, Fact::Unknown);
    assert!(info.converted.units && !info.converted.up_axis);
    assert_eq!(info.tool, None);
}

#[test]
fn damaged_line_geometry_never_panics() {
    let objects = [
        line(SQUARE, "0,1,2,-4,3,-1", &prop("Color", "1,0,0")),
        model(20, "Wire", ""),
        "\tGeometry: 11, \"Geometry::C\", \"NurbsCurve\" {\n\t}\n".to_owned(),
    ]
    .concat();
    let text = ascii(&objects, &links(&[(10, 20), (20, 0), (11, 20)], &[]));
    let binary = crate::formats::fbx_ascii::to_binary(text.as_bytes()).unwrap();
    let memory = crate::assets::Memory::new();
    let load = |data: &[u8]| {
        if let Ok(doc) = crate::formats::load_bytes(data, "t.fbx", "fbx", &memory, Keep::ALL) {
            let _flat = doc.flatten(Pose::Rest);
        }
    };
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        usize::try_from(seed >> 40).unwrap()
    };
    for data in [text.into_bytes(), binary] {
        for len in (0..data.len()).step_by(7) {
            load(&data[..len]);
        }
        for _ in 0..300 {
            let mut bad = data.clone();
            for _ in 0..=rand() % 8 {
                let at = rand() % bad.len();
                bad[at] = u8::try_from(rand() & 0xff).unwrap();
            }
            load(&bad);
        }
    }
}

#[test]
fn a_left_handed_file_is_mirrored_as_it_is_read_and_never_by_a_negative_scale() {
    // Y up, -Z front, X right: left-handed. A quad facing +Z, a unit along Z from its parent, which is turned a
    // quarter about Y and moved: the scene the file describes, with Z the other way.
    let settings = "\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",1\n\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",2\n\
                    \t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",-1\n\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n";
    let normals = "\t\tLayerElementNormal: 0 {\n\t\t\tMappingInformationType: \"ByControlPoint\"\n\
                   \t\t\tReferenceInformationType: \"Direct\"\n\t\t\tNormals: *12 {\n\t\t\t\ta: 0,0,1,0,0,1,0,0,1,0,0,1\n\t\t\t}\n\t\t}\n";
    let objects = quad(normals)
        + &model(
            20,
            "Parent",
            &(prop("Lcl Translation", "1,2,3") + &prop("Lcl Rotation", "0,90,0")),
        )
        + &model(21, "Child", &prop("Lcl Translation", "0,0,1"));
    let text = ascii(&objects, &links(&[(10, 21), (21, 20), (20, 0)], &[])).replacen(
        "\tProperties70:  {\n",
        &format!("\tProperties70:  {{\n{settings}"),
        1,
    );
    let doc = read(&text);
    let info = doc.source_info();
    assert_eq!(
        info.handedness,
        crate::source::Fact::Declared(crate::source::Handedness::Left)
    );
    assert!(
        info.converted.handedness && !info.converted.up_axis,
        "{:?}",
        info.converted
    );
    // No node mirrors: the root node has nothing to turn or scale here, and no scale is negative anywhere.
    let nodes = doc.json()["nodes"].as_array().unwrap();
    let top = nodes.last().unwrap();
    assert_eq!(top["name"], "RootNode");
    assert!(top.get("rotation").is_none() && top.get("scale").is_none(), "{top}");
    for n in nodes {
        let negative = n["scale"]
            .as_array()
            .is_some_and(|s| s.iter().any(|c| c.as_f64().unwrap() < 0.0));
        assert!(!negative && n.get("matrix").is_none(), "{n}");
    }
    // In the file's space the child is a unit along the parent's Z, which the quarter turn about Y points along
    // +X: its quad's corners are (2, 2..3, 3 - 0..1). Mirrored, Z is negated.
    let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
    let p = &mesh.prims[0];
    let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in &p.positions {
        for k in 0..3 {
            min[k] = min[k].min(v[k]);
            max[k] = max[k].max(v[k]);
        }
    }
    let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-5);
    assert!(
        near(min, [2.0, 2.0, -3.0]) && near(max, [2.0, 3.0, -2.0]),
        "{min:?}..{max:?}"
    );
    // The quad's normal, +Z in its own space, is +X in the world, mirrored or not; and its triangles still wind
    // counter-clockwise about it.
    for t in p.indices.chunks_exact(3) {
        let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
        let (u, v) = ([0, 1, 2].map(|k| b[k] - a[k]), [0, 1, 2].map(|k| c[k] - a[k]));
        let face = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        assert!(face[0] > 0.5, "{face:?}");
        assert!(
            near(p.normals[t[0] as usize], [1.0, 0.0, 0.0]),
            "{:?}",
            p.normals[t[0] as usize]
        );
    }
    // The same file said to be right-handed is the same scene unmirrored: the mirror is all that differs.
    let right = read(&text.replace(
        "\"FrontAxisSign\", \"int\", \"Integer\", \"\",-1",
        "\"FrontAxisSign\", \"int\", \"Integer\", \"\",1",
    ));
    assert!(!right.source_info().converted.handedness);
    let unmirrored = right.flatten(Pose::Rest).unwrap().unwrap();
    let mut a: Vec<[f32; 3]> = unmirrored.prims[0]
        .positions
        .iter()
        .map(|&[x, y, z]| [x, y, -z])
        .collect();
    let mut b = p.positions.clone();
    #[allow(clippy::cast_possible_truncation, reason = "small test values")]
    let key = |v: &[f32; 3]| v.map(|c| (c * 1000.0).round() as i64);
    a.sort_by_key(key);
    b.sort_by_key(key);
    assert!(a.iter().zip(&b).all(|(a, b)| near(*a, *b)), "{a:?} vs {b:?}");
}
