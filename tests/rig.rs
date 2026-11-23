//! The typed rig views over real exporter output: Blender's two-bone armature as FBX and USD.
#![cfg(feature = "fs")]

use polysmelt::document::Keep;
use polysmelt::formats;

/// Blender's armature, the same from either exporter.
#[cfg(any(feature = "fbx", feature = "usd"))]
mod armature {
    use std::path::PathBuf;

    use polysmelt::assets::Folder;
    use polysmelt::rig::{Path, Rig, Trs};

    use super::{Keep, formats};

    fn load(rel: &str, ext: &str) -> Rig {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus").join(rel);
        let assets = Folder::new(path.parent().unwrap());
        formats::load(&path, ext, &assets, Keep::ALL).unwrap().rig().unwrap()
    }

    fn mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
        let mut o = [0.0; 16];
        for c in 0..4 {
            for r in 0..4 {
                o[4 * c + r] = (0..4).map(|k| a[4 * k + r] * b[4 * c + k]).sum();
            }
        }
        o
    }

    fn matrix(t: &Trs) -> [f32; 16] {
        let [x, y, z, w] = t.rotation;
        let [sx, sy, sz] = t.scale;
        let [tx, ty, tz] = t.translation;
        [
            (1.0 - 2.0 * (y * y + z * z)) * sx,
            2.0 * (x * y + z * w) * sx,
            2.0 * (x * z - y * w) * sx,
            0.0,
            2.0 * (x * y - z * w) * sy,
            (1.0 - 2.0 * (x * x + z * z)) * sy,
            2.0 * (y * z + x * w) * sy,
            0.0,
            2.0 * (x * z + y * w) * sz,
            2.0 * (y * z - x * w) * sz,
            (1.0 - 2.0 * (x * x + y * y)) * sz,
            0.0,
            tx,
            ty,
            tz,
            1.0,
        ]
    }

    fn world(rig: &Rig, pose: &[Trs]) -> Vec<[f32; 16]> {
        let mut out = vec![[0.0; 16]; rig.nodes.len()];
        for i in rig.parents_first() {
            let local = matrix(&pose[i]);
            out[i] = match rig.nodes[i].parent {
                Some(p) => mul(&out[p], &local),
                None => local,
            };
        }
        out
    }

    fn check(rig: &Rig) {
        assert_eq!(rig.skins.len(), 1);
        let skin = &rig.skins[0];
        let names: Vec<_> = skin
            .joints
            .iter()
            .map(|&j| rig.nodes[j].name.clone().unwrap_or_default())
            .collect();
        assert!(names.iter().any(|n| n.contains("Lower")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("Upper")), "{names:?}");
        let lower = skin.joints[names.iter().position(|n| n.contains("Lower")).unwrap()];
        let upper = skin.joints[names.iter().position(|n| n.contains("Upper")).unwrap()];
        assert_eq!(rig.nodes[upper].parent, Some(lower), "Upper hangs under Lower");
        let order = rig.parents_first();
        assert_eq!(order.len(), rig.nodes.len());
        let at = |n: usize| order.iter().position(|&x| x == n).unwrap();
        assert!(at(lower) < at(upper));

        // At rest a joint's world matrix times its inverse bind matrix is the transform the mesh
        // was bound in, the same for every joint of a mesh bound where it stands.
        let rest = world(rig, &rig.pose(None, 0.0));
        let bound: Vec<[f32; 16]> = skin
            .joints
            .iter()
            .zip(&skin.inverse_bind)
            .map(|(j, ib)| mul(&rest[*j], ib))
            .collect();
        for m in &bound {
            assert!(m.iter().zip(&bound[0]).all(|(a, b)| (a - b).abs() < 1e-4), "{bound:?}");
        }

        assert_eq!(rig.animations.len(), 1);
        let a = &rig.animations[0];
        let (first, last) = a.span();
        assert!(last > first, "{first}..{last}");
        assert!(
            a.channels.iter().any(|c| c.node == upper && c.path == Path::Rotation),
            "Upper turns"
        );
        let posed = rig.pose(Some(0), last);
        assert_ne!(
            posed[upper].rotation, rig.nodes[upper].rest.rotation,
            "the pose moves Upper"
        );
        for p in &posed {
            let len: f32 = p.rotation.iter().map(|c| c * c).sum::<f32>().sqrt();
            assert!((len - 1.0).abs() < 1e-4);
        }
        assert!(a.key_times().windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    #[cfg(feature = "fbx")]
    fn fbx_armature() {
        check(&load("fbx/blender_armature_animated/armature_animated.fbx", "fbx"));
    }

    /// A pose handed in draws the mesh as the same pose played from the file does.
    #[test]
    #[cfg(feature = "fbx")]
    fn a_pose_handed_in_draws_as_the_animation_does() {
        use polysmelt::document::Pose;
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/corpus/fbx/blender_armature_animated/armature_animated.fbx");
        let doc = formats::load(&path, "fbx", &Folder::new(path.parent().unwrap()), Keep::ALL).unwrap();
        let rig = doc.rig().unwrap();
        let positions =
            |m: polysmelt::mesh::Mesh| -> Vec<[f32; 3]> { m.prims.into_iter().flat_map(|p| p.positions).collect() };
        let rest: Vec<Trs> = rig.nodes.iter().map(|n| n.rest).collect();
        let (a, b) = (
            positions(doc.flatten(Pose::Rest).unwrap().unwrap()),
            positions(doc.flatten_nodes(&rest).unwrap().unwrap()),
        );
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!(
                x.iter().zip(y).all(|(x, y)| (x - y).abs() < 1e-4),
                "{x:?} at rest, {y:?} handed in"
            );
        }
        let times = rig.animations[0].key_times();
        let mut moved = false;
        for t in [times[0], times[times.len() / 2], times[times.len() - 1]] {
            let played = positions(doc.flatten(Pose::Animation { index: 0, time: t }).unwrap().unwrap());
            let handed = positions(doc.flatten_nodes(&rig.pose(Some(0), t)).unwrap().unwrap());
            assert_eq!(played.len(), handed.len());
            for (x, y) in played.iter().zip(&handed) {
                assert!(
                    x.iter().zip(y).all(|(x, y)| (x - y).abs() < 1e-4),
                    "{x:?} played, {y:?} handed in at {t}"
                );
            }
            moved |= played
                .iter()
                .zip(&a)
                .any(|(x, y)| x.iter().zip(y).any(|(x, y)| (x - y).abs() > 1e-3));
        }
        assert!(moved, "the animation never moved the mesh");
    }

    #[test]
    #[cfg(feature = "fbx")]
    fn each_skinned_vertex_names_its_joint() {
        use polysmelt::document::Pose;
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/corpus/fbx/blender_armature_animated/armature_animated.fbx");
        let doc = formats::load(&path, "fbx", &Folder::new(path.parent().unwrap()), Keep::ALL).unwrap();
        let rig = doc.rig().unwrap();
        let joints: Vec<usize> = rig.skins.iter().flat_map(|s| s.joints.iter().copied()).collect();
        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for p in &mesh.prims {
            assert_eq!(p.joints.len(), p.positions.len());
            for &j in &p.joints {
                assert!(joints.contains(&(j as usize)), "vertex on node {j}, not a joint");
                seen.insert(j);
            }
        }
        assert!(seen.len() > 1, "every vertex on one joint: {seen:?}");
    }

    #[test]
    #[cfg(feature = "usd")]
    fn usd_armature() {
        check(&load("usd/blender_armature_animated/armature_animated.usdc", "usdc"));
    }
}

/// An accessor that starts partway into its buffer view and runs to the buffer's end reads, as glTF allows.
#[test]
#[cfg(feature = "gltf")]
fn accessor_offset_to_the_buffer_end() {
    // Key times 0 and 1, then two translations, in one view filling the buffer.
    let floats = [0.0f32, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bin: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let json = r#"{"asset":{"version":"2.0"},"nodes":[{"name":"a"}],
        "buffers":[{"byteLength":32}],"bufferViews":[{"buffer":0,"byteLength":32}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR","min":[0],"max":[1]},
                     {"bufferView":0,"byteOffset":8,"componentType":5126,"count":2,"type":"VEC3"}],
        "animations":[{"samplers":[{"input":0,"output":1}],
                       "channels":[{"sampler":0,"target":{"node":0,"path":"translation"}}]}]}"#;
    let mut json = json.as_bytes().to_vec();
    json.resize(json.len().next_multiple_of(4), b' ');
    let word = |n: usize| u32::try_from(n).expect("a small test file").to_le_bytes();
    let mut glb = Vec::new();
    glb.extend(b"glTF");
    glb.extend(2u32.to_le_bytes());
    glb.extend(word(12 + 8 + json.len() + 8 + bin.len()));
    glb.extend(word(json.len()));
    glb.extend(b"JSON");
    glb.extend(&json);
    glb.extend(word(bin.len()));
    glb.extend(b"BIN\0");
    glb.extend(&bin);
    let assets = polysmelt::assets::Memory::new();
    let rig = formats::load_bytes(&glb, "a.glb", "glb", &assets, Keep::ALL)
        .unwrap()
        .rig()
        .unwrap();
    assert_eq!(rig.pose(Some(0), 1.0)[0].translation, [4.0, 5.0, 6.0]);
}
