use super::*;
use crate::assets::Memory;
use crate::document::Pose;
use crate::mesh::Mesh;

const TEXT_CUBE: &[u8] = include_bytes!("../../../tests/corpus/x/test_cube_text/test_cube_text.x");
const BINARY_CUBE: &[u8] = include_bytes!("../../../tests/corpus/x/test_cube_binary/test_cube_binary.x");
const COMPRESSED_CUBE: &[u8] = include_bytes!("../../../tests/corpus/x/test_cube_compressed/test_cube_compressed.x");
const GEN_TEXT: &[u8] = include_bytes!("../../../tests/corpus/x/text_cube/cube.x");
const GEN_TZIP: &[u8] = include_bytes!("../../../tests/corpus/x/compressed_text/cube.x");
const GEN_STORED: &[u8] = include_bytes!("../../../tests/corpus/x/compressed_stored/cube.x");
const GEN_DOUBLE: &[u8] = include_bytes!("../../../tests/corpus/x/binary_double/cube.x");
const GEN_FIXED: &[u8] = include_bytes!("../../../tests/corpus/x/compressed_fixed/cube.x");
const GEN_ANIM: &[u8] = include_bytes!("../../../tests/corpus/x/animation/anim.x");
const GEN_SKIN: &[u8] = include_bytes!("../../../tests/corpus/x/skin/cube.x");
const GEN_DECL: &[u8] = include_bytes!("../../../tests/corpus/x/decl_data/cube.x");
const RED_PNG: &[u8] = include_bytes!("../../../tests/corpus/x/texture/tex/red.png");

fn read_bytes(bytes: &[u8], max: usize) -> Result<Document<'static>> {
    let keep = Keep {
        max_triangles: max,
        ..Keep::ALL
    };
    super::super::load_bytes(bytes, "a.x", "x", &Memory::new(), keep).map(Document::into_owned)
}

fn read(text: &str) -> Document<'static> {
    read_bytes(text.as_bytes(), usize::MAX).unwrap()
}

fn error(bytes: &[u8]) -> String {
    match read_bytes(bytes, usize::MAX) {
        Ok(_) => panic!("read without an error"),
        Err(e) => e.to_string(),
    }
}

fn mesh(doc: &Document<'_>) -> Mesh {
    doc.flatten(Pose::Rest).unwrap().unwrap()
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

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Every triangle faces away from the mesh's center.
fn faces_outward(m: &Mesh) -> bool {
    let (lo, hi) = bounds(m);
    let center = [0, 1, 2].map(|k| f32::midpoint(lo[k], hi[k]));
    m.prims.iter().all(|p| {
        p.indices.chunks(3).all(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
            let n = cross(sub(b, a), sub(c, a));
            let mid = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0);
            dot(n, sub(mid, center)) > 0.0
        })
    })
}

/// A unit cube's eight corners in the file's (left-handed) space, Z negated, and its six faces, clockwise seen from
/// outside as Direct3D draws them.
const CUBE: &str = "8;
 0;0;0;, 1;0;0;, 0;1;0;, 1;1;0;, 0;0;-1;, 1;0;-1;, 0;1;-1;, 1;1;-1;;
 6;
 4;1,3,2,0;, 4;6,7,5,4;, 4;4,5,1,0;, 4;3,7,6,2;, 4;2,6,4,0;, 4;5,7,3,1;;";

fn cube_file(extra: &str) -> String {
    format!("xof 0303txt 0032\nFrame Box {{\n Mesh Cube {{\n {CUBE}\n {extra}\n }}\n}}\n")
}

// --- Header and encodings ------------------------------------------------------------------------------------------

#[test]
fn reads_the_header() {
    let doc = read_bytes(TEXT_CUBE, usize::MAX).unwrap();
    let source = &doc.json["asset"]["extras"]["source"];
    assert_eq!(source["format"], "DirectX");
    assert_eq!(source["version"], "3.3");
    assert_eq!(source["encoding"], "text");
    assert_eq!(source["floatSize"], 32);
    assert_eq!(doc.facts["triangles"], 12);
    let info = doc.source_info();
    assert_eq!(info.version.as_deref(), Some("3.3"));
    assert_eq!(info.frame_rate, crate::source::Fact::Declared(24.0));
    assert_eq!(info.up_axis, crate::source::Fact::Assumed(crate::source::Axis::PosY));
    assert_eq!(
        info.handedness,
        crate::source::Fact::Assumed(crate::source::Handedness::Left)
    );
    assert!(info.converted.handedness && !info.converted.up_axis && !info.converted.units);
    assert_eq!(info.meters_per_unit, crate::source::Fact::Unknown);
    assert_eq!(doc.facts["meshes"], 1);
}

#[test]
fn three_encodings_read_alike() {
    let text = mesh(&read_bytes(TEXT_CUBE, usize::MAX).unwrap());
    for (bytes, encoding) in [(BINARY_CUBE, "binary"), (COMPRESSED_CUBE, "compressed binary")] {
        let doc = read_bytes(bytes, usize::MAX).unwrap();
        assert_eq!(doc.json["asset"]["extras"]["source"]["encoding"], encoding);
        let m = mesh(&doc);
        assert_eq!(m.triangles(), text.triangles());
        let (a, b) = (bounds(&m), bounds(&text));
        assert!(close(&a.0, &b.0) && close(&a.1, &b.1), "{a:?} != {b:?}");
    }
}

#[test]
fn doubles_read_like_floats() {
    let a = mesh(&read_bytes(GEN_TEXT, usize::MAX).unwrap());
    let b = mesh(&read_bytes(GEN_DOUBLE, usize::MAX).unwrap());
    assert_eq!(a.prims[0].positions, b.prims[0].positions);
    assert_eq!(a.prims[0].uvs, b.prims[0].uvs);
}

#[test]
fn refuses_what_isnt_a_x_file() {
    assert!(error(b"").contains("not a DirectX"));
    assert!(error(b"xof 0303").contains("not a DirectX"));
    assert!(error(b"xif 0303txt 0032 Mesh {}").contains("not a DirectX"));
    assert!(error(b"xof 0303zip 0032 Mesh {}").contains("unknown .x encoding"));
    assert!(error(b"xof 0303txt 0032\n").contains("empty"));
}

// --- MSZIP -----------------------------------------------------------------------------------------------------------

#[test]
fn inflates_a_stored_block() {
    let mut out = Vec::new();
    inflate(&[1, 5, 0, !5, !0, b'h', b'e', b'l', b'l', b'o'], &mut out, 5).unwrap();
    assert_eq!(out, b"hello");
    // The length's complement must match.
    assert!(inflate(&[1, 5, 0, 0, 0, b'h', b'e', b'l', b'l', b'o'], &mut Vec::new(), 5).is_err());
    // No more than the limit.
    assert!(inflate(&[1, 5, 0, !5, !0, b'h', b'e', b'l', b'l', b'o'], &mut Vec::new(), 4).is_err());
}

#[test]
fn inflates_a_fixed_block() {
    let data = [203, 72, 205, 201, 201, 215, 81, 200, 64, 162, 20, 1];
    let mut out = Vec::new();
    inflate(&data, &mut out, 100).unwrap();
    assert_eq!(out, b"hello, hello, hello!");
    // Cut short anywhere, it fails without panicking.
    for n in 0..data.len() {
        let _ = inflate(&data[..n], &mut Vec::new(), 100);
    }
}

#[test]
fn a_match_reaches_into_what_came_before() {
    // `abcdefghabcdefgh` compressed against `abcdefgh` as its dictionary: two matches back into it.
    let data = [75, 68, 163, 1];
    let mut out = b"abcdefgh".to_vec();
    inflate(&data, &mut out, 16).unwrap();
    assert_eq!(out, b"abcdefghabcdefghabcdefgh");
    // Without what came before, it reaches back too far.
    let e = inflate(&data, &mut Vec::new(), 16).unwrap_err().to_string();
    assert!(e.contains("too far"), "{e}");
    // A match may not inflate past the limit.
    assert!(inflate(&data, &mut b"abcdefgh".to_vec(), 15).is_err());
}

#[test]
fn inflates_mszip_files() {
    // Dynamic blocks whose matches reach into the block before, stored blocks, and fixed blocks of doubles.
    assert_eq!(mszip(&GEN_TZIP[16..]).unwrap(), &GEN_TEXT[16..]);
    assert_eq!(mszip(&GEN_STORED[16..]).unwrap(), &GEN_TEXT[16..]);
    assert_eq!(mszip(&GEN_FIXED[16..]).unwrap(), &GEN_DOUBLE[16..]);
    let a = mesh(&read_bytes(GEN_TZIP, usize::MAX).unwrap());
    let b = mesh(&read_bytes(GEN_TEXT, usize::MAX).unwrap());
    assert_eq!(a.prims[0].positions, b.prims[0].positions);
}

#[test]
fn mszip_refuses_a_bad_block() {
    let mut bad = GEN_TZIP.to_vec();
    bad[24] = b'X';
    assert!(error(&bad).contains("no CK"));
    // A claimed size past the limit is refused before anything is allocated for it.
    let mut huge = b"xof 0303tzip0032".to_vec();
    huge.extend_from_slice(&u32::MAX.to_le_bytes());
    if usize::BITS > 32 {
        assert!(error(&huge).contains("too big"));
    }
}

// --- Text and binary syntax ------------------------------------------------------------------------------------------

#[test]
fn reads_comments_templates_and_nan() {
    let doc = read(&format!(
        "xof 0303txt 0032\n# comment\n// another\ntemplate Mesh {{ <3D82AB44-62DA-11cf-AB39-0020AF71E433> \
         DWORD nVertices; [...] }}\nFrame Box {{ Mesh {{ {CUBE}\n MeshTextureCoords {{ 8; -1.#IND00;1.#QNAN0;, \
         1;0;, 0;1;, 1;1;, 0;0;, 1;0;, 0;1;, 1;1;; }} }} }}"
    ));
    let m = mesh(&doc);
    assert_eq!(m.triangles(), 12);
    assert!(m.prims[0].uvs.iter().all(|uv| uv.iter().all(|c| c.is_finite())));
    assert!(m.prims[0].uvs.contains(&[0.0, 0.0]));
}

#[test]
fn a_number_never_starts_an_object() {
    // The face list's last index comes right before `MeshNormals {`: it's a value, not the object's template.
    let doc = read(&cube_file(
        "MeshNormals { 6; 0;0;1;, 0;0;-1;, 0;-1;0;, 0;1;0;, -1;0;0;, 1;0;0;; \
         6; 4;0,0,0,0;, 4;1,1,1,1;, 4;2,2,2,2;, 4;3,3,3,3;, 4;4,4,4,4;, 4;5,5,5,5;; }",
    ));
    assert_eq!(mesh(&doc).triangles(), 12);
    assert!(numeric(b"23") && numeric(b"-1.5") && numeric(b".5") && numeric(b"+2"));
    assert!(!numeric(b"Mesh") && !numeric(b"Frame1"));
}

#[test]
fn binary_tokens() {
    // A binary mesh with one triangle: name, integer list and float list tokens, and a template skipped.
    let mut f = b"xof 0303bin 0032".to_vec();
    let tok = |f: &mut Vec<u8>, t: u16| f.extend_from_slice(&t.to_le_bytes());
    let name = |f: &mut Vec<u8>, s: &str| {
        f.extend_from_slice(&1u16.to_le_bytes());
        f.extend_from_slice(&u32::try_from(s.len()).unwrap().to_le_bytes());
        f.extend_from_slice(s.as_bytes());
    };
    let ints = |f: &mut Vec<u8>, v: &[u32]| {
        f.extend_from_slice(&6u16.to_le_bytes());
        f.extend_from_slice(&u32::try_from(v.len()).unwrap().to_le_bytes());
        for x in v {
            f.extend_from_slice(&x.to_le_bytes());
        }
    };
    tok(&mut f, 31);
    name(&mut f, "Vector");
    tok(&mut f, 10);
    tok(&mut f, 5);
    f.extend_from_slice(&[0; 16]);
    tok(&mut f, 42);
    name(&mut f, "x");
    tok(&mut f, 20);
    tok(&mut f, 11);
    name(&mut f, "Mesh");
    tok(&mut f, 10);
    ints(&mut f, &[3]);
    f.extend_from_slice(&7u16.to_le_bytes());
    f.extend_from_slice(&9u32.to_le_bytes());
    for x in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        f.extend_from_slice(&x.to_le_bytes());
    }
    ints(&mut f, &[1, 3, 0, 2, 1]);
    tok(&mut f, 11);
    let m = mesh(&read_bytes(&f, usize::MAX).unwrap());
    assert_eq!(m.triangles(), 1);
    // An unknown token is an error.
    let mut bad = f[..16].to_vec();
    tok(&mut bad, 99);
    assert!(error(&bad).contains("unknown token"));
}

// --- Geometry --------------------------------------------------------------------------------------------------------

#[test]
fn left_handed_faces_turn_out_facing_out() {
    let m = mesh(&read(&cube_file("")));
    assert_eq!(m.triangles(), 12);
    let (lo, hi) = bounds(&m);
    assert!(close(&lo, &[0.0; 3]) && close(&hi, &[1.0; 3]), "{lo:?} {hi:?}");
    assert!(faces_outward(&m));
}

#[test]
fn faces_become_fans() {
    let doc = read("xof 0303txt 0032\nMesh { 5; 0;0;0;, 2;0;0;, 2;1;0;, 1;2;0;, 0;1;0;; 1; 5;4,3,2,1,0;; }");
    let m = mesh(&doc);
    assert_eq!(m.triangles(), 3);
    let p = &m.prims[0];
    // Each triangle of the fan shares the face's first corner.
    let first: Vec<u32> = p.indices.chunks(3).map(|t| t[0]).collect();
    let corner = p.positions[first[0] as usize];
    assert!(first.iter().all(|&i| close(&p.positions[i as usize], &corner)));
}

#[test]
fn bad_faces_are_skipped() {
    let doc = read(&cube_file("").replace("6;\n 4;1,3,2,0;", "8;\n 2;0,1;, 3;0,1,99;, 4;1,3,2,0;"));
    assert_eq!(mesh(&doc).triangles(), 12);
}

#[test]
fn normals_have_their_own_indices() {
    // One normal per face, through MeshNormals' own face list; each mirrors to glTF's right-handed axes.
    let doc = read(&cube_file(
        "MeshNormals { 6; 0;0;1;, 0;0;-1;, 0;-1;0;, 0;1;0;, -1;0;0;, 1;0;0;; \
         6; 4;0,0,0,0;, 4;1,1,1,1;, 4;2,2,2,2;, 4;3,3,3,3;, 4;4,4,4,4;, 4;5,5,5,5;; }",
    ));
    let m = mesh(&doc);
    let p = &m.prims[0];
    assert_eq!(p.normals.len(), p.positions.len());
    for t in p.indices.chunks(3) {
        let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
        let n = cross(sub(b, a), sub(c, a));
        assert!(dot(n, p.normals[t[0] as usize]) > 0.0, "a normal against its winding");
    }
}

#[test]
fn vertex_colors_and_texture_coordinates() {
    let doc = read(&cube_file(
        "MeshTextureCoords { 8; 0;0;, 1;0;, 0;1;, 1;1;, 0;0;, 1;0;, 0;1;, 0.5;0.25;; } \
         MeshVertexColors { 8; 0;1;0;0;1;;, 1;0;1;0;1;;, 2;0;0;1;1;;, 3;1;1;1;0.5;;, \
         4;0;0;0;1;;, 5;0;0;0;1;;, 6;0;0;0;1;;, 7;0;0;0;1;;; }",
    ));
    let p = &mesh(&doc).prims[0];
    assert!(p.uvs.contains(&[0.5, 0.25]));
    assert!(p.colors.contains(&[1.0, 0.0, 0.0, 1.0]));
    assert!(p.colors.contains(&[1.0, 1.0, 1.0, 0.5]));
}

#[test]
fn decl_values() {
    assert_eq!(
        decl_value(2, &[1f32.to_bits(), 2f32.to_bits(), 3f32.to_bits()]),
        Some([1.0, 2.0, 3.0, 1.0])
    );
    assert_eq!(decl_value(4, &[0x80FF_0000]), Some([1.0, 0.0, 0.0, 128.0 / 255.0]));
    assert_eq!(decl_value(6, &[0xFFFF_0002]), Some([2.0, -1.0, 0.0, 1.0]));
    assert_eq!(decl_value(9, &[0x8000_7FFF]), Some([1.0, -1.0, 0.0, 1.0]));
    assert_eq!(decl_value(15, &[0x3C00_3800]), Some([0.5, 1.0, 0.0, 1.0]));
    assert_eq!(decl_value(13, &[0]), None);
    assert_eq!(half(0xC000), -2.0);
    assert_eq!(decl_size(17), None);
}

#[test]
fn materials_and_their_list() {
    let doc = read(&format!(
        "xof 0303txt 0032\nMaterial Red {{ 1;0;0;1;; 250; 0;0;0;; 0.5;0;0;; }}\nFrame Box {{ Mesh {{ {CUBE}\n \
         MeshMaterialList {{ 2; 3; 0,1,1;; Material {{ 0.5;0.5;0.5;1;; 0; 0;0;0;; 0;0;0;; }} {{ Red }} }} }} }}"
    ));
    let m = mesh(&doc);
    assert_eq!(m.prims.len(), 2);
    assert_eq!(m.triangles(), 12);
    let red = doc.json["materials"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == "Red")
        .unwrap();
    let rough = red["pbrMetallicRoughness"]["roughnessFactor"].as_f64().unwrap();
    assert!((rough - 0.5).abs() < 1e-3, "{rough}");
    assert!(red["emissiveFactor"][0].as_f64().unwrap() > 0.2);
    // A reference to a material that isn't there leaves the faces without one, and nothing breaks.
    let doc = read(&cube_file("MeshMaterialList { 1; 6; 0,0,0,0,0,0;; { Nowhere } }"));
    assert_eq!(mesh(&doc).triangles(), 12);
}

#[test]
fn finds_a_texture() {
    let mut files = Memory::new();
    files.insert("tex/red.png", RED_PNG.to_vec());
    let text = format!(
        "xof 0303txt 0032\nFrame Box {{ Mesh {{ {CUBE}\n MeshMaterialList {{ 1; 1; 0;; Material {{ 1;1;1;1;; 0; \
         0;0;0;; 0;0;0;; TextureFilename {{ \"tex\\\\red.png\"; }} }} }} }} }}"
    );
    let doc = super::super::load_bytes(text.as_bytes(), "a.x", "x", &files, Keep::ALL).unwrap();
    assert_eq!(doc.json["images"].as_array().map_or(0, Vec::len), 1);
    assert!(doc.json["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"].is_object());
}

// --- Frames, skins and animation -----------------------------------------------------------------------------------

#[test]
fn frames_nest_and_meshes_are_shared() {
    let doc = read(&format!(
        "xof 0303txt 0032\nMesh Shared {{ {CUBE} }}\nFrame A {{ {{ Shared }} }}\nFrame B {{ \
         FrameTransformMatrix {{ 1,0,0,0, 0,1,0,0, 0,0,1,0, 2,0,0,1;; }} {{ Shared }} }}"
    ));
    assert_eq!(doc.facts["triangles"], 24);
    let m = mesh(&doc);
    assert_eq!(m.triangles(), 24);
    let (lo, hi) = bounds(&m);
    assert!(close(&lo, &[0.0; 3]) && close(&hi, &[3.0, 1.0, 1.0]), "{lo:?} {hi:?}");
}

#[test]
fn frame_matrices_mirror_to_right_handed() {
    // A frame moved (1, 2, 3) in the file's space sits at (1, 2, -3) in glTF's.
    let doc = read(&format!(
        "xof 0303txt 0032\nFrame Moved {{ FrameTransformMatrix {{ 1,0,0,0, 0,1,0,0, 0,0,1,0, 1,2,3,1;; }} \
         Mesh {{ {CUBE} }} }}"
    ));
    let rig = doc.rig().unwrap();
    let node = rig.find("Moved").unwrap();
    assert!(close(&rig.nodes[node].rest.translation, &[1.0, 2.0, -3.0]));
}

#[test]
fn rotation_keys_are_conjugated() {
    // Stored (w, x, y, z) is the conjugate of the frame matrix's quaternion, so a key equal to the frame's own
    // rotation (90 degrees about Y in glTF) leaves it where it is.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    // glTF's rotation (0, h, 0, h) mirrors to (0, -h, 0, h) in the file's space; its matrix, rows for row vectors:
    let doc = read(&format!(
        "xof 0303txt 0032\nFrame Spin {{ FrameTransformMatrix {{ 0,0,1,0, 0,1,0,0, -1,0,0,0, 0,0,0,1;; }} \
         Mesh {{ {CUBE} }} }}\nAnimationSet Bind {{ Animation {{ {{ Spin }} AnimationKey {{ 0; 1; \
         0;4;{h},0,{h},0;;; }} }} }}"
    ));
    let rig = doc.rig().unwrap();
    let node = rig.find("Spin").unwrap();
    let rest = rig.nodes[node].rest.rotation;
    let posed = rig.pose(Some(0), 0.0)[node].rotation;
    let d: f32 = (0..4).map(|k| rest[k] * posed[k]).sum();
    assert!(d.abs() > 0.9999, "{rest:?} {posed:?}");
    assert!(close(&rest.map(f32::abs), &[0.0, h, 0.0, h]), "{rest:?}");
    assert_eq!(rotation_key([1.0, 2.0, 3.0, 4.0]), [-2.0, -3.0, -4.0, 1.0]);
}

#[test]
fn animation_keys_and_ticks() {
    let doc = read(&format!(
        "xof 0303txt 0032\nAnimTicksPerSecond {{ 24; }}\nFrame F {{ Mesh {{ {CUBE} }} }}\n\
         AnimationSet Go {{ Animation {{ {{ F }} AnimationOptions {{ 1; 0; }} \
         AnimationKey {{ 2; 2; 0;3;0,0,0;;, 24;3;1,0,2;;; }} \
         AnimationKey {{ 1; 2; 0;3;1,1,1;;, 48;3;2,2,2;;; }} }} }}"
    ));
    assert_eq!(doc.json["asset"]["extras"]["source"]["frameRate"], 24.0);
    assert_eq!(doc.json["animations"][0]["name"], "Go");
    assert_eq!(
        doc.json["animations"][0]["extras"]["animationOptions"][0]["closed"],
        true
    );
    let rig = doc.rig().unwrap();
    let node = rig.find("F").unwrap();
    let at = rig.pose(Some(0), 1.0)[node];
    assert!(close(&at.translation, &[1.0, 0.0, -2.0]), "{:?}", at.translation);
    assert!(close(&at.scale, &[1.5, 1.5, 1.5]), "{:?}", at.scale);
    // Without AnimTicksPerSecond, 4800 ticks are a second.
    let doc = read(&format!(
        "xof 0303txt 0032\nFrame F {{ Mesh {{ {CUBE} }} }}\nAnimation {{ {{ F }} \
         AnimationKey {{ 2; 2; 0;3;0,0,0;;, 4800;3;1,0,0;;; }} }}"
    ));
    let rig = doc.rig().unwrap();
    let span = rig.animations[0].span();
    assert!((span.1 - 1.0).abs() < 1e-6, "{span:?}");
}

#[test]
fn skin_weights_become_a_skin() {
    let doc = read(&format!(
        "xof 0303txt 0032\nFrame Body {{ Mesh {{ {CUBE}\n \
         SkinWeights {{ \"Lower\"; 4; 0,1,4,5; 1;1;1;1; 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1;; }} \
         SkinWeights {{ \"Upper\"; 4; 2,3,6,7; 1;1;1;1; 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,-1,0,1;; }} }} }}\n\
         Frame Lower {{ Frame Upper {{ FrameTransformMatrix {{ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,1,0,1;; }} }} }}"
    ));
    assert_eq!(doc.json["skins"].as_array().unwrap().len(), 1);
    assert_eq!(doc.json["skins"][0]["joints"].as_array().unwrap().len(), 2);
    assert_eq!(doc.facts["skinned_meshes"], 1);
    // The bind pose puts the cube where it was.
    let (lo, hi) = bounds(&mesh(&doc));
    assert!(close(&lo, &[0.0; 3]) && close(&hi, &[1.0; 3]), "{lo:?} {hi:?}");
    // Weights naming vertices past the mesh's are skipped.
    let doc = read(&cube_file(
        "SkinWeights { \"Ghost\"; 2; 0,99; 1;1; 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1;; }",
    ));
    assert_eq!(mesh(&doc).triangles(), 12);
}

// --- Hostile input ---------------------------------------------------------------------------------------------------

#[test]
fn refuses_counts_past_the_file() {
    let e = error(b"xof 0303txt 0032\nMesh { 4294967295; 0;0;0;; 0; }");
    assert!(e.contains("more than it holds"), "{e}");
    let e = error(b"xof 0303txt 0032\nMesh { 3; 0;0;0;; 0; }");
    assert!(e.contains("more than it holds"), "{e}");
    let e = error(b"xof 0303txt 0032\nMesh { 3; 0;0;0;, 1;0;0;, 0;1;0;; 1; 4000000000;0,1,2;; }");
    assert!(e.contains("more than it holds"), "{e}");
}

#[test]
fn refuses_deep_nesting() {
    let deep = format!("xof 0303txt 0032\n{}{}", "Frame {\n".repeat(300), "}\n".repeat(300));
    assert!(error(deep.as_bytes()).contains("nest too deeply"));
    // Deep but within the limit, with a mesh at the bottom.
    let ok = format!(
        "xof 0303txt 0032\n{}Mesh {{ {CUBE} }}{}",
        "Frame {\n".repeat(200),
        "}\n".repeat(200)
    );
    assert_eq!(mesh(&read(&ok)).triangles(), 12);
}

#[test]
fn keeps_to_the_triangle_limit() {
    assert!(read_bytes(GEN_TEXT, 11).is_err());
    assert!(read_bytes(GEN_TEXT, 12).is_ok());
}

/// Reading `bytes` must not panic, whatever it holds.
fn survives(bytes: &[u8]) {
    let _ = read_bytes(bytes, 1 << 20).map(|d| d.flatten(Pose::Rest));
}

#[test]
fn every_prefix_reads_or_fails() {
    for file in [BINARY_CUBE, COMPRESSED_CUBE, GEN_FIXED] {
        for n in 0..file.len() {
            survives(&file[..n]);
        }
    }
    for n in (0..TEXT_CUBE.len()).step_by(7) {
        survives(&TEXT_CUBE[..n]);
    }
}

#[test]
fn corrupted_bytes_read_or_fail() {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for file in [
        TEXT_CUBE,
        BINARY_CUBE,
        COMPRESSED_CUBE,
        GEN_TZIP,
        GEN_DOUBLE,
        GEN_ANIM,
        GEN_SKIN,
        GEN_DECL,
    ] {
        for _ in 0..300 {
            let mut bad = file.to_vec();
            for _ in 0..=next() % 4 {
                let at = 16 + usize::try_from(next() % (bad.len() as u64 - 16)).unwrap();
                bad[at] = u8::try_from(next() & 0xff).unwrap();
            }
            survives(&bad);
        }
    }
}
