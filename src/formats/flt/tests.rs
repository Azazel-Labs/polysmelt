use super::*;
use crate::assets::Memory;
use crate::document::Pose;
use crate::formats::Input;
use crate::mesh::Mesh;
use crate::test_rng::Pcg32;

/// A vertex: its position, texture coordinates and, when it has one, normal.
type Vertex = ([f64; 3], [f32; 2], Option<[f32; 3]>);

/// An `OpenFlight` file being written, record by record.
struct W {
    b: Vec<u8>,
    /// The vertex palette's offset in the file, once written.
    palette: usize,
}

impl W {
    /// A file with a header of `version`, in `units`.
    fn new(version: i32, units: u8) -> W {
        let mut h = vec![0u8; 320];
        h[0..4].copy_from_slice(b"db\0\0");
        h[8..12].copy_from_slice(&version.to_be_bytes());
        h[58] = units;
        let mut w = W {
            b: Vec::new(),
            palette: 0,
        };
        w.rec(1, &h);
        w
    }

    /// A record of `op` with `body` after its header.
    fn rec(&mut self, op: u16, body: &[u8]) -> &mut W {
        let len = u16::try_from(body.len() + 4).unwrap();
        self.b.extend_from_slice(&op.to_be_bytes());
        self.b.extend_from_slice(&len.to_be_bytes());
        self.b.extend_from_slice(body);
        self
    }

    fn push(&mut self) -> &mut W {
        self.rec(10, &[])
    }

    fn pop(&mut self) -> &mut W {
        self.rec(11, &[])
    }

    /// A named node record of `op` with `size` bytes after its header, `fill` writing its
    /// fields (offsets counted from the record's start).
    fn node(&mut self, op: u16, size: usize, name: &str, fill: impl FnOnce(&mut Vec<u8>)) -> &mut W {
        let mut r = vec![0u8; size + 4];
        let n = name.as_bytes();
        r[4..4 + n.len().min(7)].copy_from_slice(&n[..n.len().min(7)]);
        fill(&mut r);
        self.rec(op, &r[4..])
    }

    fn group(&mut self, name: &str) -> &mut W {
        self.node(2, 40, name, |_| {})
    }

    /// A group showing its children in turn (forward), over `duration` seconds.
    fn sequence(&mut self, name: &str, duration: f32) -> &mut W {
        self.node(2, 40, name, |r| {
            put(r, 16, &(0x4000_0000u32).to_be_bytes());
            put(r, 36, &duration.to_be_bytes());
        })
    }

    /// An animated group with its record's `flags`, loop `count`, loop `duration` and `last` frame duration.
    fn animated(&mut self, name: &str, flags: u32, count: i32, duration: f32, last: f32) -> &mut W {
        self.node(2, 40, name, |r| {
            put(r, 16, &flags.to_be_bytes());
            put(r, 32, &count.to_be_bytes());
            put(r, 36, &duration.to_be_bytes());
            put(r, 40, &last.to_be_bytes());
        })
    }

    fn object(&mut self, name: &str, transparency: u16) -> &mut W {
        self.node(4, 24, name, |r| put(r, 18, &transparency.to_be_bytes()))
    }

    fn lod(&mut self, name: &str, switch_in: f64, switch_out: f64) -> &mut W {
        self.node(73, 76, name, |r| {
            put(r, 16, &switch_in.to_be_bytes());
            put(r, 24, &switch_out.to_be_bytes());
        })
    }

    fn switch(&mut self, name: &str, current: u32, masks: &[u32]) -> &mut W {
        self.node(96, 24 + masks.len() * 4, name, |r| {
            put(r, 16, &current.to_be_bytes());
            put(r, 20, &u32_of(masks.len()).to_be_bytes());
            put(r, 24, &1u32.to_be_bytes());
            for (k, m) in masks.iter().enumerate() {
                put(r, 28 + k * 4, &m.to_be_bytes());
            }
        })
    }

    fn matrix(&mut self, m: [f32; 16]) -> &mut W {
        let body: Vec<u8> = m.iter().flat_map(|v| v.to_be_bytes()).collect();
        self.rec(49, &body)
    }

    fn long_id(&mut self, name: &str) -> &mut W {
        let mut body = name.as_bytes().to_vec();
        body.push(0);
        self.rec(33, &body)
    }

    fn instance(&mut self, op: u16, number: u16) -> &mut W {
        let mut body = vec![0u8; 4];
        body[2..4].copy_from_slice(&number.to_be_bytes());
        self.rec(op, &body)
    }

    fn external(&mut self, file: &str, flags: u32) -> &mut W {
        let mut r = vec![0u8; 216];
        put(&mut r, 4, file.as_bytes());
        put(&mut r, 208, &flags.to_be_bytes());
        self.rec(63, &r[4..])
    }

    /// A color palette with `colors` first.
    fn colors(&mut self, colors: &[[u8; 3]]) -> &mut W {
        let mut r = vec![0u8; 132 + 1024 * 4];
        for (k, c) in colors.iter().enumerate() {
            put(&mut r, 132 + k * 4, &[255, c[2], c[1], c[0]]);
        }
        self.rec(32, &r[4..])
    }

    fn material(&mut self, index: i32, name: &str, diffuse: [f32; 3], alpha: f32) -> &mut W {
        let mut r = vec![0u8; 84];
        put(&mut r, 4, &index.to_be_bytes());
        put(&mut r, 8, name.as_bytes());
        for (k, v) in diffuse.iter().enumerate() {
            put(&mut r, 36 + k * 4, &v.to_be_bytes());
        }
        put(&mut r, 76, &alpha.to_be_bytes());
        self.rec(113, &r[4..])
    }

    fn texture(&mut self, index: i32, file: &str) -> &mut W {
        let mut r = vec![0u8; 216];
        put(&mut r, 4, file.as_bytes());
        put(&mut r, 204, &index.to_be_bytes());
        self.rec(64, &r[4..])
    }

    /// A vertex palette of vertices with positions and coordinates, and normals when given;
    /// returns each one's offset.
    fn vertices(&mut self, v: &[Vertex]) -> Vec<u32> {
        self.palette = self.b.len();
        let total = 8 + v.len() * 64;
        self.rec(67, &u32_of(total).to_be_bytes());
        let mut at = Vec::new();
        for (p, uv, n) in v {
            at.push(u32_of(self.b.len() - self.palette));
            let mut r = vec![0u8; 64];
            put(&mut r, 6, &0x2000u16.to_be_bytes());
            for (k, c) in p.iter().enumerate() {
                put(&mut r, 8 + k * 8, &c.to_be_bytes());
            }
            if let Some(n) = n {
                for (k, c) in n.iter().enumerate() {
                    put(&mut r, 32 + k * 4, &c.to_be_bytes());
                }
                put(&mut r, 44, &uv[0].to_be_bytes());
                put(&mut r, 48, &uv[1].to_be_bytes());
                self.rec(70, &r[4..]);
            } else {
                put(&mut r, 32, &uv[0].to_be_bytes());
                put(&mut r, 36, &uv[1].to_be_bytes());
                self.rec(71, &r[4..60]);
            }
        }
        at
    }

    /// A face of `corners` (vertex palette offsets), `fill` writing its fields.
    fn face(&mut self, name: &str, corners: &[u32], fill: impl FnOnce(&mut Vec<u8>)) -> &mut W {
        self.node(5, 76, name, |r| {
            r[18] = 0;
            put(r, 28, &(-1i16).to_be_bytes());
            put(r, 30, &(-1i16).to_be_bytes());
            put(r, 68, &u32::MAX.to_be_bytes());
            put(r, 44, &F_NO_COLOR.to_be_bytes());
            fill(r);
        });
        self.push();
        let body: Vec<u8> = corners.iter().flat_map(|c| c.to_be_bytes()).collect();
        self.rec(72, &body);
        self.pop()
    }

    fn bytes(&self) -> Vec<u8> {
        self.b.clone()
    }
}

fn put(r: &mut [u8], at: usize, b: &[u8]) {
    r[at..at + b.len()].copy_from_slice(b);
}

/// The unit square's corners in the XY plane, facing +Z.
const SQUARE: [Vertex; 4] = [
    ([0.0, 0.0, 0.0], [0.0, 0.0], None),
    ([1.0, 0.0, 0.0], [1.0, 0.0], None),
    ([1.0, 1.0, 0.0], [1.0, 1.0], None),
    ([0.0, 1.0, 0.0], [0.0, 1.0], None),
];

fn read(b: &[u8], assets: &Memory) -> Result<Document> {
    load(Input::memory(b, "t.flt"), assets, Keep::ALL)
}

fn read_named(b: &[u8], name: &str, assets: &Memory) -> Result<Document> {
    load(Input::memory(b, name), assets, Keep::ALL)
}

fn flat(doc: &Document) -> Mesh {
    doc.flatten(Pose::Rest).unwrap().unwrap()
}

fn close(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

fn corners(m: &Mesh) -> Vec<[f32; 3]> {
    m.prims
        .iter()
        .flat_map(|p| p.indices.iter().map(|&i| p.positions[i as usize]))
        .collect()
}

/// A file of one square face in an object in a group, the face's fields set by `fill`.
fn one_square(version: i32, units: u8, fill: impl FnOnce(&mut Vec<u8>)) -> W {
    let mut w = W::new(version, units);
    w.colors(&[[255, 0, 0], [0, 255, 0]]);
    w.material(0, "paint", [0.5, 0.25, 1.0], 1.0);
    let v = w.vertices(&SQUARE);
    w.push().group("g1").push().object("o1", 0).push();
    w.face("f1", &v, fill);
    w.pop().pop().pop();
    w
}

#[test]
fn a_face_in_meters_and_y_up() {
    let w = one_square(1640, 4, |_| {});
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let src = &doc.json()["asset"]["extras"]["source"];
    assert_eq!(src["format"], "OpenFlight");
    assert_eq!(src["version"], "16.4");
    assert_eq!(src["unit"], "foot");
    let info = doc.source_info();
    assert_eq!(info.meters_per_unit, Fact::Declared(0.3048));
    assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
    assert!(info.converted.units && info.converted.up_axis);
    assert_eq!(info.created, None);
    assert_eq!(info.georeference, None);
    let nodes = doc.json()["nodes"].as_array().unwrap();
    assert_eq!(nodes[0]["name"], "db");
    assert_eq!(nodes[1]["name"], "g1");
    assert_eq!(nodes[2]["name"], "o1");
    assert_eq!(nodes[2]["extras"]["openflight"]["type"], "object");
    let m = flat(&doc);
    assert_eq!(m.triangles(), 2);
    // (1, 1, 0) feet, Z up, lands at (0.3048, 0, -0.3048) meters, Y up.
    let c = corners(&m);
    assert!(c.iter().any(|&p| close(p, [0.3048, 0.0, -0.3048])), "{c:?}");
    // The face looked down on from +Z now faces +Y.
    assert!(
        close(m.prims[0].normals[0], [0.0, 1.0, 0.0]),
        "{:?}",
        m.prims[0].normals[0]
    );
    // Draw type 0 culls its back; no material, no color: white.
    assert!(!m.materials[0].double_sided);
    assert_eq!(m.prims[0].colors.len(), 0);
    assert!(doc.to_glb().is_ok());
}

#[test]
fn materials_colors_and_draw_types() {
    // Two sided, material 0, palette color 1 (green) at full intensity, lit with face colors.
    let w = one_square(1640, 0, |r| {
        r[18] = 1;
        put(r, 30, &0i16.to_be_bytes());
        put(r, 44, &0u32.to_be_bytes());
        put(r, 68, &((1u32 << 7) | 127).to_be_bytes());
        r[48] = 2;
    });
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let m = flat(&doc);
    let mat = &m.materials[0];
    assert_eq!(mat.name, "paint");
    assert!(mat.double_sided);
    assert!(!mat.unlit);
    assert!(close(
        [mat.color[0], mat.color[1], mat.color[2]],
        [0.5, 0.25, 1.0].map(srgb_to_linear)
    ));
    assert_eq!(m.prims[0].colors[0], [0.0, 1.0, 0.0, 1.0]);

    // Packed color at half transparency: blended, and the alpha is in the colors.
    let w = one_square(1640, 0, |r| {
        put(r, 44, &F_PACKED.to_be_bytes());
        put(r, 56, &[255, 0, 0, 255]);
        put(r, 40, &32768u16.to_be_bytes());
    });
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let mf = &doc.json()["materials"][0];
    assert_eq!(mf["alphaMode"], "BLEND");
    let m = flat(&doc);
    let c = m.prims[0].colors[0];
    assert!(
        close([c[0], c[1], c[2]], [1.0, 0.0, 0.0]) && (c[3] - 0.5).abs() < 0.01,
        "{c:?}"
    );

    // A wireframe loop: four lines, no triangles; points: four points.
    for (draw, lines, points) in [(2u8, 4, 0), (3, 3, 0), (8, 0, 4)] {
        let w = one_square(1640, 0, |r| r[18] = draw);
        let doc = read(&w.bytes(), &Memory::new()).unwrap();
        let c = doc.counts();
        assert_eq!((c.triangles, c.lines, c.points), (0, lines, points), "draw {draw}");
    }

    // A hidden face is left out, and with nothing else there's no geometry.
    let w = one_square(1640, 0, |r| put(r, 44, &F_HIDDEN.to_be_bytes()));
    assert!(read(&w.bytes(), &Memory::new()).is_err());
}

#[test]
fn levels_of_detail_switches_and_matrices() {
    let mut w = W::new(1640, 0);
    let v = w.vertices(&SQUARE);
    w.push();
    // Of two levels of detail, the nearer (switch out 100) is kept.
    w.lod("far", 1000.0, 100.0).push().face("farface", &v, |_| {}).pop();
    w.lod("near", 100.0, 0.0).push().face("nearface", &v, |_| {}).pop();
    // A switch showing its second child only.
    w.switch("sw", 0, &[0b10]).push();
    w.group("off").push().face("a", &v, |_| {}).pop();
    w.group("on").push().face("b", &v, |_| {}).pop();
    w.pop();
    // A group moved 5 along X.
    w.group("moved");
    let mut m = [0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m[12] = 5.0;
    w.matrix(m).long_id("a long name for the moved group");
    w.push().face("c", &v, |_| {}).pop();
    w.pop();
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let nodes = doc.json()["nodes"].as_array().unwrap();
    let names: Vec<&str> = nodes.iter().map(|n| n["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"near") && !names.contains(&"far"), "{names:?}");
    assert!(names.contains(&"on") && !names.contains(&"off"), "{names:?}");
    let moved = nodes
        .iter()
        .find(|n| n["name"] == "a long name for the moved group")
        .unwrap();
    assert_eq!(moved["translation"], json!([5.0, 0.0, 0.0]));
    let near = nodes.iter().find(|n| n["name"] == "near").unwrap();
    assert_eq!(near["extras"]["openflight"]["switchIn"], json!(100.0));
    let src = &doc.json()["asset"]["extras"]["source"];
    assert_eq!(src["droppedLevelsOfDetail"], 1);
    assert_eq!(src["switchedOff"], 1);
    assert_eq!(flat(&doc).triangles(), 6);
}

#[test]
fn instances_replicas_and_cycles() {
    let mut w = W::new(1640, 0);
    let v = w.vertices(&SQUARE);
    w.push();
    // Instance 7: one face.
    w.instance(62, 7).push().face("shared", &v, |_| {}).pop();
    w.group("a").push().instance(61, 7).pop();
    w.group("b").push().instance(61, 7).pop();
    // Replicated three times more, each moved 2 along Y.
    w.group("rep");
    let mut m = [0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m[13] = 2.0;
    w.matrix(m).rec(60, &[0, 3, 0, 0]);
    w.push().face("r", &v, |_| {}).pop();
    // An instance of itself.
    w.instance(62, 9).push().instance(61, 9).pop();
    w.group("loop").push().instance(61, 9).pop();
    w.pop();
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let m = flat(&doc);
    // Two instances and four replicas.
    assert_eq!(m.triangles(), 12);
    // The instances share their mesh.
    let meshes = doc.json()["meshes"].as_array().unwrap();
    assert_eq!(meshes.len(), 2);
    let c = corners(&m);
    // The last replica, three moves along Y (−Z in glTF).
    assert!(c.iter().any(|&p| close(p, [1.0, 0.0, -7.0])), "{c:?}");
    assert_eq!(doc.json()["asset"]["extras"]["source"]["cycles"], 1);
}

#[test]
fn external_references_with_units_and_palettes() {
    // The child, in feet, with its own material 0 (red).
    let mut child = W::new(1640, 4);
    child.material(0, "childred", [1.0, 0.0, 0.0], 1.0);
    let v = child.vertices(&SQUARE);
    child.push().face("cf", &v, |r| put(r, 30, &0i16.to_be_bytes()));
    // And a reference back to the parent.
    child.external("parent.flt", u32::MAX).pop();
    let mut parent = W::new(1640, 0);
    parent.material(0, "parentblue", [0.0, 0.0, 1.0], 1.0);
    parent.push();
    // Once with its own palettes, once with the parent's materials.
    parent.external("c:\\somewhere\\else\\child.flt", u32::MAX);
    parent.external("child.flt", !(0x8000_0000u32 >> 1));
    parent.external("missing.flt", 0);
    parent.pop();
    let assets = Memory::new()
        .with("child.flt", child.bytes())
        .with("parent.flt", parent.bytes());
    let doc = read_named(&parent.bytes(), "parent.flt", &assets).unwrap();
    let m = flat(&doc);
    assert_eq!(m.triangles(), 4);
    let names: Vec<&str> = m.materials.iter().map(|m| m.name.as_str()).collect();
    assert!(
        names.contains(&"childred") && names.contains(&"parentblue"),
        "{names:?}"
    );
    // The child's feet become meters.
    let c = corners(&m);
    assert!(c.iter().any(|&p| close(p, [0.3048, 0.0, -0.3048])), "{c:?}");
    let src = &doc.json()["asset"]["extras"]["source"];
    assert_eq!(src["missingFiles"].as_array().map(Vec::len), Some(1), "{src}");
    assert_eq!(src["cycles"], 2);
    assert!(doc.to_glb().is_ok());
}

/// A 2×2 RGBA SGI image, stored, its top-left pixel half transparent.
fn sgi_image() -> Vec<u8> {
    let mut b = vec![0u8; 512];
    b[0] = 0x01;
    b[1] = 0xDA;
    b[2] = 0;
    b[3] = 1;
    b[4..6].copy_from_slice(&3u16.to_be_bytes());
    b[6..8].copy_from_slice(&2u16.to_be_bytes());
    b[8..10].copy_from_slice(&2u16.to_be_bytes());
    b[10..12].copy_from_slice(&4u16.to_be_bytes());
    // Channels, each bottom row first.
    b.extend_from_slice(&[255, 0, 255, 0]);
    b.extend_from_slice(&[0, 255, 0, 255]);
    b.extend_from_slice(&[0, 0, 0, 0]);
    b.extend_from_slice(&[255, 255, 128, 255]);
    b
}

#[test]
fn sgi_textures_are_embedded() {
    let mut w = W::new(1640, 0);
    w.texture(3, "/old/disk/maps/skin.rgba");
    let v = w.vertices(&SQUARE);
    w.push().face("tf", &v, |r| put(r, 28, &3i16.to_be_bytes())).pop();
    let assets = Memory::new().with("skin.rgba", sgi_image());
    let doc = read(&w.bytes(), &assets).unwrap();
    let m = flat(&doc);
    assert_eq!(m.images.len(), 1);
    assert_eq!(m.images[0].mime, "image/png");
    assert_eq!(m.materials[0].color_image, Some(0));
    // The transparent pixel blends the material.
    assert_eq!(doc.json()["materials"][0]["alphaMode"], "BLEND");
    // Coordinates flip v: the file's (1, 1) corner is glTF's (1, 0).
    let p = &m.prims[0];
    let k = p.positions.iter().position(|&q| close(q, [1.0, 0.0, -1.0])).unwrap();
    assert_eq!(p.uvs[k], [1.0, 0.0]);
    // Missing, the texture is counted and the face kept.
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    assert_eq!(doc.json()["asset"]["extras"]["source"]["droppedTextures"], 1);
}

#[test]
fn a_texture_many_palette_entries_and_faces_name_is_counted_once_when_it_is_not_there() {
    // Two palette entries name the one file (`second` as written, or by another path to it); three faces use them.
    let model = |second: &str| {
        let mut w = W::new(1640, 0);
        w.texture(3, "/old/disk/maps/wood.rgb");
        w.texture(4, second);
        let v = w.vertices(&SQUARE);
        w.push();
        for (name, index) in [("a", 3i16), ("b", 4), ("c", 3)] {
            w.face(name, &v, |r| put(r, 28, &index.to_be_bytes()));
        }
        w.pop();
        w.bytes()
    };
    let doc = read(&model("/old/disk/maps/wood.rgb"), &Memory::new()).unwrap();
    let src = &doc.json()["asset"]["extras"]["source"];
    assert_eq!(src["missingFiles"], json!(["/old/disk/maps/wood.rgb"]), "{src}");
    assert_eq!(src["droppedTextures"], 1, "{src}");
    // Found but not an image, names that find the one file are one texture.
    let doc = read(
        &model("wood.rgb"),
        &Memory::new().with("wood.rgb", b"not an image".to_vec()),
    )
    .unwrap();
    let src = &doc.json()["asset"]["extras"]["source"];
    assert_eq!(src["droppedTextures"], 1, "{src}");
    assert!(src.get("missingFiles").is_none(), "{src}");
}

#[test]
fn light_sources_become_lights() {
    let mut w = W::new(1640, 0);
    let mut lp = vec![0u8; 240];
    put(&mut lp, 4, &2i32.to_be_bytes());
    put(&mut lp, 16, b"lamp");
    for (k, v) in [1.0f32, 0.5, 0.25, 1.0].iter().enumerate() {
        put(&mut lp, 56 + k * 4, &v.to_be_bytes());
    }
    put(&mut lp, 88, &2i32.to_be_bytes());
    put(&mut lp, 136, &30f32.to_be_bytes());
    w.rec(102, &lp[4..]);
    let v = w.vertices(&SQUARE);
    w.push().face("f", &v, |_| {});
    w.node(101, 60, "light", |r| {
        put(r, 16, &2i32.to_be_bytes());
        put(r, 24, &0x8000_0000u32.to_be_bytes());
        put(r, 32, &3f64.to_be_bytes());
        // Pitched straight down.
        put(r, 60, &(-90f32).to_be_bytes());
    });
    w.pop();
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let lights = &doc.json()["extensions"]["KHR_lights_punctual"]["lights"];
    assert_eq!(lights[0]["type"], "spot");
    assert_eq!(lights[0]["name"], "lamp");
    let outer = lights[0]["spot"]["outerConeAngle"].as_f64().unwrap();
    assert!((outer - 30f64.to_radians()).abs() < 1e-6);
    let node = doc.json()["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["name"] == "light")
        .unwrap();
    assert_eq!(node["extensions"]["KHR_lights_punctual"]["light"], 0);
    assert_eq!(node["translation"], json!([3.0, 0.0, 0.0]));
    // Its -Z turned to -Z in the file (down): no turn at all.
    assert!(node.get("rotation").is_none(), "{node}");
    assert!(doc.to_glb().is_ok());
}

#[test]
fn meshes_with_strips() {
    let mut w = W::new(1640, 0);
    w.push();
    // A mesh: its fields four bytes later than a face's.
    w.node(84, 80, "mesh", |r| {
        put(r, 32, &(-1i16).to_be_bytes());
        put(r, 34, &(-1i16).to_be_bytes());
    });
    w.push();
    // A local vertex pool of four positions and normals.
    let mut pool = vec![0u8; 12];
    put(&mut pool, 4, &4u32.to_be_bytes());
    put(&mut pool, 8, &(0x8000_0000u32 | (0x8000_0000 >> 3)).to_be_bytes());
    for (p, ..) in SQUARE {
        for c in p {
            pool.extend_from_slice(&c.to_be_bytes());
        }
        for c in [0f32, 0.0, 1.0] {
            pool.extend_from_slice(&c.to_be_bytes());
        }
    }
    w.rec(85, &pool[4..]);
    // A strip of 0 1 3 2: two triangles.
    let mut strip = vec![0u8; 12];
    put(&mut strip, 4, &1i16.to_be_bytes());
    put(&mut strip, 6, &2u16.to_be_bytes());
    put(&mut strip, 8, &4u32.to_be_bytes());
    for i in [0u16, 1, 3, 2] {
        strip.extend_from_slice(&i.to_be_bytes());
    }
    w.rec(86, &strip[4..]);
    w.pop().pop();
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let m = flat(&doc);
    assert_eq!(m.triangles(), 2);
    // Both triangles face +Z in the file, +Y here.
    let p = &m.prims[0];
    for t in p.indices.chunks(3) {
        let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
        let n = [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ];
        assert!(n[1] > 0.0, "{n:?}");
    }
}

#[test]
fn animated_groups_take_turns() {
    let mut w = W::new(1640, 0);
    let v = w.vertices(&SQUARE);
    w.push().sequence("seq", 3.0).push();
    for name in ["one", "two", "three"] {
        w.group(name).push().face(name, &v, |_| {}).pop();
    }
    w.pop().pop();
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let anim = &doc.json()["animations"][0];
    assert_eq!(anim["channels"].as_array().unwrap().len(), 3);
    assert_eq!(anim["samplers"][0]["interpolation"], "STEP");
    // At rest, only the first shows.
    assert_eq!(flat(&doc).triangles(), 6);
    let nodes = doc.json()["nodes"].as_array().unwrap();
    let frames: Vec<&Value> = (0..3)
        .map(|k| nodes.iter().find(|n| n["name"] == format!("frame{k}")).unwrap())
        .collect();
    assert!(frames[0].get("scale").is_none());
    assert_eq!(frames[1]["scale"], json!([0.0, 0.0, 0.0]));
    assert!(doc.to_glb().is_ok());
}

#[test]
fn old_files_with_inline_vertices() {
    // Version 12: faces followed by integer vertices, an old color palette.
    let mut w = W::new(12, 0);
    let mut pal = vec![0u8; 4 + 88 * 6];
    // Variable color 2: blue.
    put(&mut pal, 4 + 2 * 6, &[0, 0, 0, 0, 0, 255]);
    w.rec(32, &pal[4..]);
    w.push();
    w.node(5, 76, "old", |r| {
        put(r, 20, &((2u16 << 7) | 127).to_be_bytes());
        put(r, 28, &(-1i16).to_be_bytes());
        put(r, 30, &(-1i16).to_be_bytes());
    });
    for (p, ..) in SQUARE {
        let mut r = vec![0u8; 16];
        for (k, c) in p.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, reason = "small whole numbers")]
            put(&mut r, 4 + k * 4, &((*c as i32) * 10).to_be_bytes());
        }
        w.rec(6, &r[4..]);
    }
    w.pop();
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let m = flat(&doc);
    assert_eq!(m.triangles(), 2);
    assert!(corners(&m).iter().any(|&p| close(p, [10.0, 0.0, -10.0])));
    assert_eq!(m.prims[0].colors[0], [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn hostile_files_never_panic() {
    let mut w = one_square(1640, 0, |r| r[18] = 1);
    w.push().lod("l", 10.0, 0.0).instance(61, 1).external("x.flt", 0).pop();
    let good = w.bytes();
    let assets = Memory::new().with("x.flt", good.clone());
    // Every truncation.
    for n in 0..good.len() {
        let _ = read(&good[..n], &assets);
    }
    // Random corruption.
    let mut rng = Pcg32::new(7);
    for _ in 0..3000 {
        let mut b = good.clone();
        for _ in 0..=rng.below(8) {
            let at = rng.below(u32_of(b.len())) as usize;
            b[at] = u8::try_from(rng.next_u32() & 0xFF).unwrap();
        }
        if let Ok(doc) = read(&b, &assets) {
            let _ = doc.to_glb();
        }
    }
    // A record too short, and endless pushes.
    assert!(read(&[0, 1, 0, 2], &assets).is_err());
    let mut deep = W::new(1640, 0);
    let v = deep.vertices(&SQUARE);
    for _ in 0..10_000 {
        deep.push().group("g");
    }
    deep.push().face("f", &v, |_| {});
    let doc = read(&deep.bytes(), &assets).unwrap();
    assert_eq!(flat(&doc).triangles(), 2);
    // Instances doubling 20 times stop at the node limit.
    let mut wide = W::new(1640, 0);
    let v = wide.vertices(&SQUARE);
    wide.push();
    wide.instance(62, 0).push().face("f", &v, |_| {}).pop();
    for k in 1..21u16 {
        wide.instance(62, k)
            .push()
            .instance(61, k - 1)
            .instance(61, k - 1)
            .pop();
    }
    wide.instance(61, 20).pop();
    let doc = read(&wide.bytes(), &assets).unwrap();
    assert!(
        doc.json()["asset"]["extras"]["source"]["droppedNodes"]
            .as_u64()
            .unwrap()
            > 0
    );
    // Nested past the depth limit, they stop there.
    let mut tall = W::new(1640, 0);
    let v = tall.vertices(&SQUARE);
    tall.push().face("f", &v, |_| {});
    tall.instance(62, 0).push().face("f", &v, |_| {}).pop();
    for k in 1..40u16 {
        tall.instance(62, k).push().instance(61, k - 1).pop();
    }
    tall.instance(61, 39).pop();
    let doc = read(&tall.bytes(), &assets).unwrap();
    assert_eq!(doc.json()["asset"]["extras"]["source"]["tooDeep"], 1);
    assert_eq!(flat(&doc).triangles(), 2);
    // A replicate of 32767 copies, each a replicate of 32767.
    let mut rep = W::new(1640, 0);
    let v = rep.vertices(&SQUARE);
    let mut m = [0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    rep.push().group("a").matrix(m).rec(60, &[0x7F, 0xFF, 0, 0]).push();
    rep.group("b").matrix(m).rec(60, &[0x7F, 0xFF, 0, 0]).push();
    rep.face("f", &v, |_| {}).pop().pop().pop();
    let doc = read(&rep.bytes(), &assets).unwrap();
    assert!(
        doc.json()["asset"]["extras"]["source"]["droppedNodes"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[test]
fn the_header_says_when_and_where() {
    let mut w = one_square(1640, 0, |_| {});
    // The header record's body follows its 4 bytes of opcode and length.
    let body = |at: usize| at - 4;
    let h = &mut w.b[4..];
    put(h, body(20), b"Tue Mar 05 14:30:00 2024");
    put(h, body(92), &4i32.to_be_bytes());
    put(
        h,
        body(188),
        &[47.0f64.to_be_bytes(), (-123.0f64).to_be_bytes()].concat(),
    );
    put(
        h,
        body(204),
        &[48.0f64.to_be_bytes(), (-122.0f64).to_be_bytes()].concat(),
    );
    put(
        h,
        body(220),
        &[47.5f64.to_be_bytes(), (-122.5f64).to_be_bytes()].concat(),
    );
    put(h, body(276), &10i16.to_be_bytes());
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    let info = doc.source_info();
    assert_eq!(info.format.as_deref(), Some("OpenFlight"));
    assert_eq!(info.unit.as_deref(), Some("meter"));
    assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
    assert!(!info.converted.units && info.converted.up_axis);
    assert_eq!(info.created.as_deref(), Some("2024-03-05T14:30:00"));
    let g = info.georeference.unwrap();
    assert_eq!(g["latitude"], 47.5);
    assert_eq!(g["longitude"], -122.5);
    assert_eq!(g["southwest"], json!([47.0, -123.0]));
    assert_eq!(g["projection"], "UTM");
    assert_eq!(g["ellipsoid"], "WGS 84");
    assert_eq!(g["utmZone"], 10);
    assert!(doc.json()["asset"]["extras"]["source"].get("date").is_none());
}

#[test]
fn a_face_with_a_corner_that_isnt_there_is_dropped_whole() {
    let mut w = W::new(1640, 0);
    let v = w.vertices(&SQUARE);
    w.push().group("g1").push().object("o1", 0).push();
    w.face("good", &v, |_| {});
    // Its last corner names no vertex of the palette: not drawn as a triangle of the other three.
    w.face("bad", &[v[0], v[1], v[2], 9999], |_| {});
    w.pop().pop().pop();
    let doc = read(&w.bytes(), &Memory::new()).unwrap();
    assert_eq!(flat(&doc).triangles(), 2);
    assert_eq!(doc.json()["asset"]["extras"]["source"]["droppedGeometry"], 1);
}

/// Group flags: forward animation, swing, backward animation.
const FORWARD: u32 = 0x8000_0000 >> 1;
const SWING: u32 = 0x8000_0000 >> 2;
const BACKWARD: u32 = 0x8000_0000 >> 6;

/// A file of one animated group of `n` squares, child k `k + 1` wide, with the group record's values.
fn sequence_file(n: usize, flags: u32, count: i32, duration: f32, last: f32) -> Vec<u8> {
    let mut w = W::new(1640, 0);
    let mut squares = Vec::new();
    for k in 0..n {
        #[allow(clippy::cast_precision_loss, reason = "a few squares")]
        let s = (k + 1) as f64;
        squares.extend(SQUARE.iter().map(|&(p, uv, nrm)| (p.map(|c| c * s), uv, nrm)));
    }
    let v = w.vertices(&squares);
    w.push().animated("seq", flags, count, duration, last).push();
    for k in 0..n {
        let name = format!("c{k}");
        w.group(&name).push().face(&name, &v[4 * k..4 * k + 4], |_| {}).pop();
    }
    w.pop().pop();
    w.bytes()
}

/// The order of the children shown in animation 0, at each of its keys, and the key times.
fn shown_in_turn(doc: &Document) -> (Vec<usize>, Vec<f32>) {
    let rig = doc.rig().unwrap();
    let a = &rig.animations[0];
    let times = a.channels[0].times.clone();
    let frames: Vec<usize> = (0..).map_while(|k| rig.find(&format!("frame{k}"))).collect();
    let order = times
        .iter()
        .map(|&t| {
            let pose = rig.pose(Some(0), t);
            frames.iter().position(|&f| pose[f].scale[0] > 0.5).unwrap()
        })
        .collect();
    (order, times)
}

fn close_times(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5)
}

#[test]
fn animated_groups_play_their_loops_as_the_record_says() {
    let read_one = |b: &[u8]| read(b, &Memory::new()).unwrap();
    let loops_played = |doc: &Document| {
        let nodes = doc.json()["nodes"].as_array().unwrap();
        let seq = nodes.iter().find(|n| n["name"] == "seq").unwrap();
        seq["extras"]["openflight"]["loopsPlayed"].as_u64()
    };

    // Two loops of three, a second a frame, the last of the last loop 1.5 seconds longer.
    let doc = read_one(&sequence_file(3, FORWARD, 2, 3.0, 1.5));
    let (order, times) = shown_in_turn(&doc);
    assert_eq!(order, [0, 1, 2, 0, 1, 2, 2]);
    assert!(close_times(&times, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5]), "{times:?}");
    assert_eq!(loops_played(&doc), Some(2));
    assert_eq!(doc.source_info().frame_rate, Fact::Unknown);

    // Forever: one loop, the last frame duration not applicable.
    let doc = read_one(&sequence_file(3, FORWARD, 0, 3.0, 2.0));
    let (order, times) = shown_in_turn(&doc);
    assert_eq!(order, [0, 1, 2, 2]);
    assert!(close_times(&times, &[0.0, 1.0, 2.0, 3.0]), "{times:?}");
    assert_eq!(loops_played(&doc), Some(1));
    assert_eq!(
        doc.json()["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["name"] == "seq")
            .unwrap()["extras"]["openflight"]["loopCount"],
        0
    );

    // A forward swing twice: 2N - 1 frames sharing the first loop's 4 seconds, then 2N - 2 sharing the second's.
    let doc = read_one(&sequence_file(3, FORWARD | SWING, 2, 4.0, 0.0));
    let (order, times) = shown_in_turn(&doc);
    assert_eq!(order, [0, 1, 2, 1, 0, 1, 2, 1, 0, 0]);
    assert!(
        close_times(&times, &[0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 5.0, 6.0, 7.0, 8.0]),
        "{times:?}"
    );

    // A backward swing forever: the last child first, and at rest.
    let doc = read_one(&sequence_file(3, BACKWARD | SWING, 0, 5.0, 0.0));
    let (order, _) = shown_in_turn(&doc);
    assert_eq!(order, [2, 1, 0, 1, 2, 2]);
    let rest = flat(&doc);
    assert!(
        corners(&rest).iter().any(|&p| close(p, [3.0, 0.0, -3.0])),
        "{:?}",
        corners(&rest)
    );

    // Backward alone, and forward with backward, which is undefined: forward.
    let (order, _) = shown_in_turn(&read_one(&sequence_file(3, BACKWARD, 1, 3.0, 0.0)));
    assert_eq!(order, [2, 1, 0, 0]);
    let (order, _) = shown_in_turn(&read_one(&sequence_file(3, FORWARD | BACKWARD, 1, 3.0, 0.0)));
    assert_eq!(order, [0, 1, 2, 2]);

    // As fast as possible: the shared 30 frames a second, assumed; the last frame half a second longer.
    let doc = read_one(&sequence_file(2, FORWARD, 2, 0.0, 0.5));
    let (_, times) = shown_in_turn(&doc);
    assert!(
        close_times(&times, &[0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0, 4.0 / 30.0 + 0.5]),
        "{times:?}"
    );
    assert_eq!(doc.source_info().frame_rate, Fact::Assumed(30.0));

    // A loop count past what's kept plays as many loops as fit, and says so; the last frame isn't held longer, its
    // loop not reached.
    let doc = read_one(&sequence_file(2, FORWARD, i32::MAX, 2.0, 5.0));
    let played = loops_played(&doc).unwrap();
    assert!((2..u64::from(u32::MAX)).contains(&played), "{played}");
    let (_, times) = shown_in_turn(&doc);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "a whole number of seconds"
    )]
    let end = (2 * played) as f32;
    assert!(
        (times[times.len() - 1] - end).abs() < 1e-3 * end,
        "{} {end}",
        times[times.len() - 1]
    );
}
