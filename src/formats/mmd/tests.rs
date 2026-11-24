use super::*;
use crate::assets::Memory;

/// Builds a file by hand.
#[derive(Default)]
struct W {
    out: Vec<u8>,
    utf8: bool,
}

impl W {
    fn u8(&mut self, v: u8) -> &mut Self {
        self.out.push(v);
        self
    }
    fn u16(&mut self, v: u16) -> &mut Self {
        self.out.extend(v.to_le_bytes());
        self
    }
    fn i32(&mut self, v: i32) -> &mut Self {
        self.out.extend(v.to_le_bytes());
        self
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.out.extend(v.to_le_bytes());
        self
    }
    fn f(&mut self, v: &[f32]) -> &mut Self {
        for x in v {
            self.out.extend(x.to_le_bytes());
        }
        self
    }
    fn text(&mut self, s: &str) -> &mut Self {
        let b: Vec<u8> = if self.utf8 {
            s.as_bytes().to_vec()
        } else {
            s.encode_utf16().flat_map(u16::to_le_bytes).collect()
        };
        self.i32(i32::try_from(b.len()).unwrap());
        self.out.extend(b);
        self
    }
    fn fixed(&mut self, b: &[u8], n: usize) -> &mut Self {
        let mut b = b.to_vec();
        b.resize(n, 0);
        self.out.extend(b);
        self
    }
}

/// A PMX 2.0 model: two bones, a quad of two materials (one textured), a vertex morph and a group morph over it,
/// a bone morph, and every deform. Index sizes are all `size`.
fn pmx_sample(utf8: bool, size: u8) -> Vec<u8> {
    let mut w = W { utf8, ..W::default() };
    w.out.extend(b"PMX ");
    w.f(&[2.0]).u8(8).u8(u8::from(utf8)).u8(1);
    for _ in 0..6 {
        w.u8(size);
    }
    w.text("モデル").text("model").text("").text("");
    let idx = |w: &mut W, v: i32| match size {
        1 => w.out.push(i8::try_from(v).unwrap().to_le_bytes()[0]),
        2 => w.out.extend(i16::try_from(v).unwrap().to_le_bytes()),
        _ => w.out.extend(v.to_le_bytes()),
    };
    // Vertices: position, normal (facing -Z, toward a viewer in MMD), uv, one additional UV, deform, edge.
    let corners = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]];
    w.i32(4);
    for (i, c) in corners.iter().enumerate() {
        w.f(c).f(&[0.0, 0.0, -1.0]).f(&[0.0, 0.0]).f(&[0.5, 0.25, 0.0, 0.0]);
        match i {
            0 => {
                w.u8(0);
                idx(&mut w, 0);
            }
            1 => {
                w.u8(1);
                idx(&mut w, 0);
                idx(&mut w, 1);
                w.f(&[0.25]);
            }
            2 => {
                w.u8(2);
                for b in [0, 1, -1, -1] {
                    idx(&mut w, b);
                }
                w.f(&[0.5, 0.5, 0.0, 0.0]);
            }
            _ => {
                w.u8(3);
                idx(&mut w, 1);
                idx(&mut w, 1);
                w.f(&[1.0]).f(&[0.0; 9]);
            }
        }
        w.f(&[1.0]);
    }
    // Faces: clockwise seen from -Z, as MMD's front faces are.
    w.i32(6);
    for v in [0i32, 2, 1, 0, 3, 2] {
        match size {
            1 => w.out.push(u8::try_from(v).unwrap()),
            2 => w.out.extend(u16::try_from(v).unwrap().to_le_bytes()),
            _ => w.out.extend(v.to_le_bytes()),
        }
    }
    w.i32(1).text("tex\\skin.png");
    w.i32(2);
    for (name, tex, alpha, flags) in [("face", 0, 1.0, 0x01u8), ("veil", -1, 0.5, 0)] {
        w.text(name)
            .text("")
            .f(&[1.0, 0.5, 0.25, alpha])
            .f(&[0.5, 0.5, 0.5])
            .f(&[30.0])
            .f(&[0.2; 3]);
        w.u8(flags).f(&[0.0, 0.0, 0.0, 1.0]).f(&[1.0]);
        idx(&mut w, tex);
        idx(&mut w, -1);
        w.u8(0).u8(1).u8(2).text("").i32(3);
    }
    // Bones: a root at (0, 10, 0), a child at (0, 15, 5) with IK and an inherited rotation.
    w.i32(2);
    w.text("センター").text("center").f(&[0.0, 10.0, 0.0]);
    idx(&mut w, -1);
    w.i32(0).u16(0x001E).f(&[0.0, 1.0, 0.0]);
    w.text("腕").text("arm").f(&[0.0, 15.0, 5.0]);
    idx(&mut w, 0);
    w.i32(0).u16(0x0001 | 0x0020 | 0x0100);
    idx(&mut w, 0);
    idx(&mut w, 0);
    w.f(&[0.5]);
    idx(&mut w, 0);
    w.i32(40).f(&[2.0]).i32(1);
    idx(&mut w, 0);
    w.u8(1).f(&[-1.0, 0.0, 0.0]).f(&[0.5, 0.0, 0.0]);
    // Morphs: vertex, group of it at half, bone.
    w.i32(3);
    w.text("smile").text("").u8(3).u8(1).i32(1);
    match size {
        1 => w.out.push(2),
        2 => w.out.extend(2u16.to_le_bytes()),
        _ => w.out.extend(2i32.to_le_bytes()),
    }
    w.f(&[0.0, 0.0, 2.0]);
    w.text("half").text("").u8(4).u8(0).i32(1);
    idx(&mut w, 0);
    w.f(&[0.5]);
    w.text("lift").text("").u8(4).u8(2).i32(1);
    idx(&mut w, 1);
    w.f(&[0.0, 1.0, 0.0]).f(&[0.0, 0.0, 0.0, 1.0]);
    // Display frames, rigid bodies, joints: none.
    w.i32(0).i32(0).i32(0);
    w.out
}

fn read(bytes: &[u8], name: &str) -> Result<Document> {
    let assets = Memory::new().with("tex/skin.png", PNG.to_vec());
    let ext = name.rsplit('.').next().unwrap();
    load(super::super::Input::memory(bytes, name), ext, &assets, Keep::ALL)
}

/// A 1x1 RGBA PNG.
const PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49,
    0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99, 0x3D,
    0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

#[test]
fn a_pmx_model_in_every_encoding_and_index_size() {
    for utf8 in [false, true] {
        for size in [1, 2, 4] {
            let doc = read(&pmx_sample(utf8, size), "m.pmx").unwrap();
            let j = &doc.json;
            assert_eq!(doc.facts["triangles"], json!(2), "{utf8} {size}");
            assert_eq!(doc.facts["bones"], json!(2));
            assert_eq!(doc.facts["sdef_vertices"], json!(1));
            assert_eq!(doc.source_info().title.as_deref(), Some("モデル"));
            assert_eq!(j["meshes"][0]["primitives"].as_array().unwrap().len(), 2);
            let names = &j["meshes"][0]["extras"]["targetNames"];
            assert_eq!(names, &json!(["smile", "half"]));
            let rig = doc.rig().unwrap();
            let arm = rig.find("腕").unwrap();
            assert_eq!(rig.nodes[arm].parent, rig.find("センター"));
        }
    }
}

#[test]
fn pmx_is_mirrored_and_scaled() {
    let doc = read(&pmx_sample(true, 2), "m.pmx").unwrap();
    let rig = doc.rig().unwrap();
    let arm = rig.find("腕").unwrap();
    // (0, 5, 5) from its parent, mirrored and in meters.
    let t = rig.nodes[arm].rest.translation;
    assert!((t[1] - 0.4).abs() < 1e-6 && (t[2] + 0.4).abs() < 1e-6, "{t:?}");
    let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
    let p = &mesh.prims[0];
    // Faces wind counter-clockwise about their normals once mirrored: the normal now faces +Z.
    let [a, b, c] = [0, 1, 2].map(|k| p.positions[p.indices[k] as usize]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let z = u[0] * v[1] - u[1] * v[0];
    assert!(z > 0.0 && p.normals[0][2] > 0.99, "{z} {:?}", p.normals[0]);
    assert!(p.positions.iter().any(|q| (q[0] - 0.8).abs() < 1e-6));
}

#[test]
fn pmx_materials_and_extras() {
    let doc = read(&pmx_sample(false, 4), "m.pmx").unwrap();
    let j = &doc.json;
    let face = &j["materials"][0];
    assert_eq!(face["name"], "face");
    assert_eq!(face["doubleSided"], true);
    assert_eq!(face["alphaMode"], "MASK");
    assert!(face["pbrMetallicRoughness"]["baseColorTexture"].is_object());
    assert_eq!(j["materials"][1]["alphaMode"], "BLEND");
    let src = &j["asset"]["extras"]["source"];
    assert_eq!(src["materials"][0]["toon"], "toon03.bmp");
    assert_eq!(src["metersPerUnit"], json!(0.08));
    let info = doc.source_info();
    assert_eq!(info.format.as_deref(), Some("PMX"));
    assert_eq!(info.version.as_deref(), Some("2.0"));
    assert_eq!(info.meters_per_unit, Fact::Assumed(0.08));
    assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
    assert_eq!(info.handedness, Fact::Declared(Handedness::Left));
    assert!(info.converted.units && info.converted.handedness && !info.converted.up_axis);
    let morphs = j["meshes"][0]["extras"]["morphs"].as_array().unwrap();
    assert_eq!(morphs.len(), 2);
    assert_eq!(morphs[1]["type"], "bone");
    let arm = &j["nodes"][1]["extras"];
    assert_eq!(arm["ik"]["loops"], 40);
    assert_eq!(arm["inherit"]["rotation"], true);
    // Limits about X turn the other way once mirrored.
    assert_eq!(arm["ik"]["links"][0]["min"], json!([-0.5, -0.0, 0.0]));
}

#[test]
fn pmx_morph_targets_and_skin() {
    let doc = read(&pmx_sample(true, 1), "m.pmx").unwrap();
    let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
    let w = mesh.prims[0].skin_weights.iter().find(|w| (w[0] - 0.75).abs() < 1e-6);
    assert!(w.is_some(), "BDEF2 at 0.25 for the second bone");
    let rig = doc.rig().unwrap();
    assert_eq!(rig.skins[0].joints.len(), 2);
}

#[test]
fn reading_less_drops_morphs_and_skins() {
    let bytes = pmx_sample(true, 2);
    let keep = Keep::ALL.with_morph_targets(false).with_skins(false);
    let doc = load(
        super::super::Input::memory(&bytes, "m.pmx"),
        "pmx",
        &Memory::new(),
        keep,
    )
    .unwrap();
    assert!(doc.json["meshes"][0]["primitives"][0].get("targets").is_none());
    assert!(doc.json.get("skins").is_none());
}

/// A PMD model: a triangle, a material with a texture and a sphere map, two bones, IK, a base and an eye morph,
/// English names and toons.
fn pmd_sample() -> Vec<u8> {
    let mut w = W::default();
    w.out.extend(b"Pmd");
    w.f(&[1.0])
        .fixed(&[0x83, 0x82, 0x83, 0x66, 0x83, 0x8B], 20)
        .fixed(b"comment", 256);
    w.u32(3);
    for (p, b0, b1, wt) in [
        ([0.0, 0.0, 0.0], 0u16, 0u16, 100u8),
        ([10.0, 0.0, 0.0], 0, 1, 50),
        ([0.0, 10.0, 0.0], 1, 1, 100),
    ] {
        w.f(&p).f(&[0.0, 0.0, -1.0]).f(&[0.0, 1.0]).u16(b0).u16(b1).u8(wt).u8(0);
    }
    w.u32(3).u16(0).u16(2).u16(1);
    w.u32(1);
    w.f(&[0.8, 0.8, 0.8])
        .f(&[0.5])
        .f(&[5.0])
        .f(&[0.1; 3])
        .f(&[0.4; 3])
        .u8(0)
        .u8(1)
        .u32(3);
    w.fixed(b"c.bmp*s.sph", 20);
    w.u16(2);
    w.fixed(b"root", 20).u16(0xFFFF).u16(1).u8(1).u16(0).f(&[0.0, 0.0, 0.0]);
    w.fixed(b"tip", 20).u16(0).u16(0).u8(0).u16(0).f(&[0.0, 10.0, 0.0]);
    w.u16(1).u16(0).u16(1).u8(1).u16(15).f(&[0.5]).u16(1);
    w.u16(2);
    w.fixed(b"base", 20).u32(1).u8(0).u32(2).f(&[0.0, 10.0, 0.0]);
    w.fixed(b"eye", 20).u32(1).u8(1).u32(0).f(&[0.0, 0.0, 3.0]);
    w.u8(0).u8(0).u32(0);
    w.u8(1)
        .fixed(b"Model", 20)
        .fixed(b"", 256)
        .fixed(b"Root", 20)
        .fixed(b"Tip", 20)
        .fixed(b"Eye", 20);
    for _ in 0..10 {
        w.fixed(b"toon.bmp", 100);
    }
    w.u32(0).u32(0);
    w.out
}

#[test]
fn a_pmd_model() {
    let doc = read(&pmd_sample(), "m.pmd").unwrap();
    let j = &doc.json;
    let src = &j["asset"]["extras"]["source"];
    assert_eq!(doc.source_info().title.as_deref(), Some("モデル"));
    assert_eq!(doc.source_info().version.as_deref(), Some("1.0"));
    assert_eq!(src["englishName"], "Model");
    assert_eq!(src["materials"][0]["sphereMode"], "multiply");
    assert_eq!(j["materials"][0]["doubleSided"], true);
    assert_eq!(j["meshes"][0]["extras"]["targetNames"], json!(["eye"]));
    assert_eq!(j["nodes"][0]["extras"]["ik"]["controlWeight"], json!(0.5));
    assert_eq!(j["nodes"][1]["extras"]["englishName"], "Tip");
    assert_eq!(src["toonTextures"].as_array().unwrap().len(), 10);
}

/// A VMD motion: a bone with a linear span and a bent one, a morph key, a camera key.
fn vmd_sample() -> Vec<u8> {
    let mut w = W::default();
    w.fixed(b"Vocaloid Motion Data 0002", 30).fixed(b"model", 20);
    let linear: Vec<u8> = (0..16)
        .map(|i| [20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107][i])
        .collect();
    let mut bent = linear.clone();
    bent[3] = 127;
    bent[7] = 0;
    w.u32(3);
    for (frame, x, curve) in [(10u32, 0.0f32, &linear), (0, 0.0, &linear), (20, 10.0, &bent)] {
        w.fixed(b"bone", 15)
            .u32(frame)
            .f(&[x, 0.0, 0.0])
            .f(&[0.0, 0.0, 0.0, 1.0]);
        let mut c = curve.clone();
        c.resize(64, 0);
        w.out.extend(c);
    }
    w.u32(1).fixed(b"smile", 15).u32(30).f(&[1.0]);
    w.u32(1).fixed(&[0; 61], 61);
    w.out
}

#[test]
fn a_vmd_motion() {
    let doc = read(&vmd_sample(), "m.vmd").unwrap();
    let j = &doc.json;
    assert_eq!(doc.facts["bone_keys"], json!(3));
    assert_eq!(doc.facts["camera_keys"], json!(1));
    let src = &j["asset"]["extras"]["source"];
    assert_eq!(src["morphKeys"]["smile"]["times"], json!([1.0]));
    let info = doc.source_info();
    assert_eq!(info.format.as_deref(), Some("VMD"));
    assert_eq!(info.frame_rate, Fact::Declared(30.0));
    assert_eq!(info.handedness, Fact::Declared(Handedness::Left));
    let rig = doc.rig().unwrap();
    let a = &rig.animations[0];
    // The translation's span from 10 to 20 is straight; the rotation's is bent and sampled every frame.
    let bone = rig.find("bone").unwrap();
    let rot = a
        .channels
        .iter()
        .find(|c| c.node == bone && c.path == crate::rig::Path::Rotation)
        .unwrap();
    assert_eq!(rot.times.len(), 2 + 10);
    let tr = a
        .channels
        .iter()
        .find(|c| c.node == bone && c.path == crate::rig::Path::Translation)
        .unwrap();
    assert_eq!(tr.times.len(), 3);
}

#[test]
fn bezier_curves() {
    assert!((bezier([20, 20, 107, 107], 0.3) - 0.3).abs() < 1e-6);
    assert!(bezier([127, 0, 127, 0], 0.5) < 0.2);
    assert!((bezier([0, 127, 0, 127], 0.5) - 1.0).abs() < 0.2);
}

#[test]
fn cut_short_and_broken_files() {
    for (good, name) in [
        (pmx_sample(false, 2), "m.pmx"),
        (pmd_sample(), "m.pmd"),
        (vmd_sample(), "m.vmd"),
    ] {
        for k in 0..good.len() {
            let _ = read(&good[..k], name);
        }
        let mut flipped = good.clone();
        for i in (0..flipped.len()).step_by(7) {
            flipped[i] ^= 0xA5;
            let _ = read(&flipped, name);
        }
    }
    assert!(
        read(b"nothing here", "m.pmx")
            .unwrap_err()
            .to_string()
            .contains("isn't")
    );
    let mut v = pmx_sample(true, 2);
    v[4..8].copy_from_slice(&3.0f32.to_le_bytes());
    assert!(read(&v, "m.pmx").unwrap_err().to_string().contains("version"));
    let mut v = pmx_sample(true, 2);
    v[9] = 7;
    assert!(read(&v, "m.pmx").unwrap_err().to_string().contains("encoding"));
}

#[test]
fn huge_counts_are_refused_before_allocating() {
    let good = pmx_sample(true, 2);
    // The vertex count follows the header and four texts.
    let mut w = W {
        utf8: true,
        ..W::default()
    };
    w.text("モデル").text("model").text("").text("");
    let at = 17 + w.out.len();
    let mut v = good.clone();
    v[at..at + 4].copy_from_slice(&i32::MAX.to_le_bytes());
    assert!(
        read(&v, "m.pmx")
            .unwrap_err()
            .to_string()
            .contains("more than it holds")
    );
    v[at..at + 4].copy_from_slice(&(-1i32).to_le_bytes());
    assert!(read(&v, "m.pmx").unwrap_err().to_string().contains("negative"));
}

#[test]
fn a_group_morph_that_contains_itself_does_not_hang() {
    let morphs = vec![Morph {
        name: "loop".into(),
        english: String::new(),
        panel: 0,
        kind: 0,
        data: MorphData::Group(vec![(0, 1.0); 64]),
    }];
    let mut acc = HashMap::new();
    let mut budget = 1 << 20;
    expand(&morphs, 0, 1.0, 0, &mut acc, &mut budget);
    assert!(acc.is_empty());
}
