use super::*;
use crate::assets::Memory;

/// Loads every prefix of `bytes`, and `bytes` with each byte overwritten by a few values, through `load`; any
/// result but a panic passes.
fn sweep(bytes: &[u8], load: impl Fn(&[u8])) {
    for end in 0..bytes.len() {
        load(&bytes[..end]);
    }
    let mut copy = bytes.to_vec();
    for at in 0..bytes.len() {
        for value in [0x00, 0x01, 0x7F, 0x80, 0xFF] {
            copy[at] = value;
            load(&copy);
        }
        copy[at] = bytes[at];
    }
}

fn load_as(bytes: &[u8], name: &str, assets: &Memory) -> Result<Document> {
    load(Input::memory(bytes, name), assets, Keep::ALL)
}

/// A hostile copy of `bytes` must never panic, whatever the assets beside it.
fn hostile(bytes: &[u8], name: &str, assets: &Memory) {
    assert!(load_as(bytes, name, assets).is_ok(), "{name} reads");
    sweep(bytes, |b| {
        let _ = load_as(b, name, assets);
    });
}

const QUAD: &[u8] = include_bytes!("../../../tests/corpus/ogre/quad_v1100/quad.mesh");

#[test]
fn hostile_binary_meshes_never_panic() {
    let material = include_bytes!("../../../tests/corpus/ogre/twin_binary/twin.material");
    hostile(
        include_bytes!("../../../tests/corpus/ogre/twin_binary/twin.mesh"),
        "twin.mesh",
        &Memory::new().with("twin.material", &material[..]),
    );
    hostile(
        include_bytes!("../../../tests/corpus/ogre/quad_big_endian/quad.mesh"),
        "quad.mesh",
        &Memory::new(),
    );
    hostile(
        include_bytes!("../../../tests/corpus/ogre/v120_big_endian/old.mesh"),
        "old.mesh",
        &Memory::new(),
    );
    hostile(
        include_bytes!("../../../tests/corpus/ogre/element_types/types.mesh"),
        "types.mesh",
        &Memory::new(),
    );
    hostile(
        include_bytes!("../../../tests/corpus/ogre/poses_v18/face.mesh"),
        "face.mesh",
        &Memory::new(),
    );
    hostile(
        include_bytes!("../../../tests/corpus/ogre/morph_big_endian/grow.mesh"),
        "grow.mesh",
        &Memory::new(),
    );
}

#[test]
fn hostile_skeletons_never_panic() {
    let skeleton = include_bytes!("../../../tests/corpus/ogre/skinned/rig.skeleton");
    let mesh = include_bytes!("../../../tests/corpus/ogre/skinned/rig.mesh");
    hostile(skeleton, "rig.skeleton", &Memory::new());
    hostile(
        include_bytes!("../../../tests/corpus/ogre/skeleton_only/walk.skeleton"),
        "walk.skeleton",
        &Memory::new(),
    );
    // A sound mesh linked to a broken skeleton.
    sweep(skeleton, |b| {
        let _ = load_as(mesh, "rig.mesh", &Memory::new().with("rig.skeleton", b.to_vec()));
    });
}

#[test]
fn hostile_xml_never_panics() {
    let extra = include_bytes!("../../../tests/corpus/ogre/skeleton_xml/extra.skeleton");
    let skeleton = include_bytes!("../../../tests/corpus/ogre/skeleton_xml/robot.skeleton.xml");
    hostile(
        include_bytes!("../../../tests/corpus/ogre/twin_xml/twin.mesh.xml"),
        "twin.mesh.xml",
        &Memory::new(),
    );
    hostile(
        skeleton,
        "robot.skeleton.xml",
        &Memory::new().with("extra.skeleton", &extra[..]),
    );
}

#[test]
fn hostile_material_scripts_never_panic() {
    let mesh = include_bytes!("../../../tests/corpus/ogre/materials/mats.mesh");
    let script = include_bytes!("../../../tests/corpus/ogre/materials/mats.material");
    let base = include_bytes!("../../../tests/corpus/ogre/materials/lib/base.material");
    sweep(script, |b| {
        let assets = Memory::new()
            .with("mats.material", b.to_vec())
            .with("lib/base.material", &base[..]);
        assert!(load_as(mesh, "mats.mesh", &assets).is_ok());
    });
}

#[test]
fn huge_counts_fail_without_allocating() {
    // Every four bytes of the quad in turn made enormous: counts, lengths and offsets among them.
    for at in 0..QUAD.len() - 4 {
        let mut copy = QUAD.to_vec();
        copy[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let _ = load_as(&copy, "quad.mesh", &Memory::new());
    }
    // An XML vertex count far past its vertices.
    let xml = "<mesh><sharedgeometry vertexcount=\"4000000000\"><vertexbuffer positions=\"true\">\
               <vertex><position x=\"0\" y=\"0\" z=\"0\"/></vertex></vertexbuffer></sharedgeometry>\
               <submeshes><submesh material=\"m\" usesharedvertices=\"true\"><faces count=\"4000000000\">\
               <face v1=\"0\" v2=\"0\" v3=\"0\"/></faces></submesh></submeshes></mesh>";
    let _ = load_as(xml.as_bytes(), "big.mesh.xml", &Memory::new());
}

#[test]
fn skeleton_parent_cycles_are_broken() {
    let xml = "<skeleton><bones><bone id=\"0\" name=\"a\"/><bone id=\"1\" name=\"b\"/><bone id=\"2\" name=\"c\"/>\
               </bones><bonehierarchy><boneparent bone=\"a\" parent=\"b\"/><boneparent bone=\"b\" parent=\"c\"/>\
               <boneparent bone=\"c\" parent=\"a\"/></bonehierarchy></skeleton>";
    let doc = load_as(xml.as_bytes(), "loop.skeleton.xml", &Memory::new()).unwrap();
    let nodes = doc.json()["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 3);
    let roots = doc.json()["scenes"][0]["nodes"].as_array().unwrap();
    assert!(!roots.is_empty(), "a cycle leaves a root");
}

fn sub(operation: u16, indices: &[u32]) -> SubMesh {
    SubMesh {
        operation,
        indices: Some(indices.to_vec()),
        ..SubMesh::default()
    }
}

#[test]
fn operations_become_lists() {
    // A strip alternates winding and drops degenerate triangles.
    let (strip, points, lines) = lists(&sub(5, &[0, 1, 2, 3, 3, 4]), 5);
    assert!(!points && !lines);
    assert_eq!(strip, [0, 1, 2, 2, 1, 3]);
    let (fan, ..) = lists(&sub(6, &[0, 1, 2, 3]), 4);
    assert_eq!(fan, [0, 1, 2, 0, 2, 3]);
    let (strip, _, lines) = lists(&sub(3, &[0, 1, 2]), 3);
    assert!(lines);
    assert_eq!(strip, [0, 1, 1, 2]);
    assert_eq!(lists(&sub(2, &[0, 1, 2]), 3).0, [0, 1]);
    assert_eq!(lists(&sub(4, &[0, 1, 2, 3]), 4).0, [0, 1, 2]);
    assert!(lists(&sub(1, &[2]), 3).1);
    // Too few indices draw nothing, and no indices draw the vertices in order.
    assert_eq!(lists(&sub(5, &[0, 1]), 2).0, [0u32; 0]);
    assert_eq!(lists(&sub(6, &[]), 0).0, [0u32; 0]);
    let none = SubMesh {
        operation: 4,
        ..SubMesh::default()
    };
    assert_eq!(lists(&none, 3).0, [0, 1, 2]);
}

#[test]
fn influences_are_every_assignment_to_a_known_bone() {
    let bones = Bones {
        nodes: (0..6).collect(),
        by_handle: (0..6u16).map(|h| (h + 10, usize::from(h))).collect(),
    };
    let a = |vertex, bone, weight| Assign { vertex, bone, weight };
    let assigns = [
        a(0, 10, 1.0),
        a(0, 11, 5.0),
        a(0, 12, 2.0),
        a(0, 13, 4.0),
        a(0, 14, 3.0),
        // The same bone twice adds up.
        a(1, 15, 1.0),
        a(1, 15, 1.0),
        a(1, 10, 2.0),
        // Zero, negative and NaN weights count for nothing; unknown bones and vertices are counted.
        a(1, 11, 0.0),
        a(1, 12, -1.0),
        a(1, 13, f32::NAN),
        a(1, 99, 1.0),
        a(7, 10, 1.0),
    ];
    let mut unknown = 0;
    let out = influences(&assigns, 2, &bones, &mut unknown);
    assert_eq!(unknown, 2);
    // Every assignment, as listed: the shared writer merges, sorts and scales them.
    assert_eq!(out[0], [(0, 1.0), (1, 5.0), (2, 2.0), (3, 4.0), (4, 3.0)]);
    assert_eq!(out[1], [(5, 1.0), (5, 1.0), (0, 2.0)]);
}

#[test]
fn between_keys() {
    let t = [0.0, 1.0, 3.0];
    assert_eq!(between(&t, -1.0), (0, 0, 0.0));
    assert_eq!(between(&t, 0.5), (0, 1, 0.5));
    assert_eq!(between(&t, 1.0), (1, 2, 0.0));
    assert_eq!(between(&t, 2.0), (1, 2, 0.5));
    assert_eq!(between(&t, 9.0), (2, 2, 0.0));
    assert_eq!(between(&[2.0], 5.0), (0, 0, 0.0));
}

#[test]
fn every_byte_order_and_form_agree() {
    // The same quad read little-endian, big-endian and through its corpus XML twin draws the same.
    let le = load_as(QUAD, "quad.mesh", &Memory::new()).unwrap();
    let be = load_as(
        include_bytes!("../../../tests/corpus/ogre/quad_big_endian/quad.mesh"),
        "quad.mesh",
        &Memory::new(),
    )
    .unwrap();
    assert_eq!(le.json()["meshes"], be.json()["meshes"]);
    assert_eq!(le.json()["accessors"], be.json()["accessors"]);
    let bin = load_as(
        include_bytes!("../../../tests/corpus/ogre/twin_binary/twin.mesh"),
        "twin.mesh",
        &Memory::new(),
    )
    .unwrap();
    let xml = load_as(
        include_bytes!("../../../tests/corpus/ogre/twin_xml/twin.mesh.xml"),
        "twin.mesh.xml",
        &Memory::new(),
    )
    .unwrap();
    assert_eq!(bin.json()["meshes"], xml.json()["meshes"]);
    assert_eq!(bin.json()["accessors"], xml.json()["accessors"]);
}

#[test]
fn source_info_is_the_engine_convention() {
    let doc = load_as(QUAD, "quad.mesh", &Memory::new()).unwrap();
    let info = doc.source_info();
    assert_eq!(info.format.as_deref(), Some("OGRE"));
    assert_eq!(info.version.as_deref(), Some("MeshSerializer_v1.100"));
    assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
    assert_eq!(info.handedness, Fact::Assumed(Handedness::Right));
    assert_eq!(info.meters_per_unit, Fact::Unknown);
    assert_eq!(info.converted, crate::source::Converted::default());
}
