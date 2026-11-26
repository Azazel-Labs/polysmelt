use crate::assets::Folder;
use crate::document::{Keep, Pose};
use crate::mesh::{Material, Mesh, Prim};

/// A bumpy grid with normals, UVs and colors, in two materials.
fn grid(n: u32) -> Mesh {
    let mut prims = Vec::new();
    for m in 0..2u32 {
        let mut p = Prim {
            material: m as usize,
            ..Prim::default()
        };
        for y in 0..=n {
            for x in 0..=n {
                let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
                p.positions.push([u + m as f32, (u * 7.0).sin() * 0.1, v]);
                p.normals.push([0.0, 1.0, 0.0]);
                p.uvs.push([u, v]);
                p.colors.push([u, v, 0.5, 1.0]);
            }
        }
        for y in 0..n {
            for x in 0..n {
                let i = y * (n + 1) + x;
                p.indices
                    .extend_from_slice(&[i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2]);
            }
        }
        prims.push(p);
    }
    let mut red = Material::plain("Red", [1.0, 0.0, 0.0, 1.0]);
    red.metallic = 0.5;
    red.roughness = 0.25;
    red.emissive = [0.1, 0.2, 0.3];
    red.double_sided = true;
    Mesh {
        prims,
        materials: vec![red, Material::plain("Blue", [0.0, 0.0, 1.0, 0.5])],
        images: Vec::new(),
        skeleton: Vec::new(),
    }
}

/// Each triangle's corners, rotated to start at its smallest, so triangles that begin at
/// another corner (as meshopt's triangle codec may write them) compare equal.
pub(super) fn triangles(p: &Prim) -> Vec<[[u32; 12]; 3]> {
    let key = |i: u32| {
        let i = i as usize;
        let f = |v: &[f32]| v.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
        let mut k = [0u32; 12];
        let all = [f(&p.positions[i]), f(&p.normals[i]), f(&p.uvs[i]), f(&p.colors[i])].concat();
        k.copy_from_slice(&all);
        k
    };
    let mut out: Vec<_> = p
        .indices
        .chunks(3)
        .map(|t| {
            let c = [key(t[0]), key(t[1]), key(t[2])];
            let r = (0..3).min_by_key(|&s| c[s]).unwrap();
            [c[r], c[(r + 1) % 3], c[(r + 2) % 3]]
        })
        .collect();
    out.sort_unstable();
    out
}

fn load(dir: &std::path::Path, name: &str, bytes: &[u8]) -> Mesh {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    let ext = name.rsplit('.').next().unwrap();
    let doc = crate::formats::load(&path, ext, &Folder::new(dir), Keep::ALL).unwrap();
    // From memory, with the folder's files as the assets, it reads the same.
    let mut memory = crate::assets::Memory::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        memory.insert(&p.file_name().unwrap().to_string_lossy(), std::fs::read(&p).unwrap());
    }
    let in_memory = crate::formats::load_bytes(bytes, name, ext, &memory, Keep::ALL).unwrap();
    assert!(
        doc.to_glb().unwrap() == in_memory.to_glb().unwrap(),
        "{name} reads differently from memory"
    );
    let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
    assert!(
        mesh.to_glb().unwrap() == in_memory.flatten(Pose::Rest).unwrap().unwrap().to_glb().unwrap(),
        "{name} flattens differently from memory"
    );
    mesh
}

fn same(a: &Mesh, b: &Mesh) {
    assert_eq!(a.prims.len(), b.prims.len());
    for (pa, pb) in a.prims.iter().zip(&b.prims) {
        assert_eq!(triangles(pa), triangles(pb));
        assert_eq!(pa.material, pb.material);
    }
    for (ma, mb) in a.materials.iter().zip(&b.materials) {
        assert_eq!(
            (
                &ma.name,
                ma.color,
                ma.emissive,
                ma.metallic,
                ma.roughness,
                ma.double_sided
            ),
            (
                &mb.name,
                mb.color,
                mb.emissive,
                mb.metallic,
                mb.roughness,
                mb.double_sided
            )
        );
    }
}

#[test]
fn tangents_and_every_set_are_read_and_follow_the_node() {
    // One triangle with a tangent, three UV sets and a second color set (but no first), its
    // color texture reading set 1 at twice the scale, on a node mirrored in x.
    let floats: Vec<f32> = [
        &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..],         // POSITION
        &[0., 0., 1., 0., 0., 1., 0., 0., 1.],             // NORMAL
        &[1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.], // TANGENT
        &[0., 0., 1., 0., 0., 1.],                         // TEXCOORD_0
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],                   // TEXCOORD_1
        &[7., 8., 7., 8., 7., 8.],                         // TEXCOORD_2
        &[1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1.], // COLOR_1
    ]
    .concat();
    let bin: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let parts = [
        ("VEC3", 3),
        ("VEC3", 3),
        ("VEC4", 4),
        ("VEC2", 2),
        ("VEC2", 2),
        ("VEC2", 2),
        ("VEC4", 4),
    ];
    let mut offset = 0;
    let mut accessors = Vec::new();
    let mut views = Vec::new();
    for (i, (kind, width)) in parts.iter().enumerate() {
        views.push(serde_json::json!({ "buffer": 0, "byteOffset": offset, "byteLength": 12 * width }));
        let mut a = serde_json::json!({ "bufferView": i, "componentType": 5126, "count": 3, "type": kind });
        if i == 0 {
            a["min"] = serde_json::json!([0, 0, 0]);
            a["max"] = serde_json::json!([1, 1, 0]);
        }
        accessors.push(a);
        offset += 12 * width;
    }
    let gltf = serde_json::json!({
        "asset": { "version": "2.0" },
        "buffers": [{ "uri": "t.bin", "byteLength": bin.len() }],
        "bufferViews": views,
        "accessors": accessors,
        "materials": [{ "pbrMetallicRoughness": { "baseColorTexture": {
            "index": 0, "texCoord": 0,
            "extensions": { "KHR_texture_transform": { "texCoord": 1, "scale": [2, 2] } }
        } } }],
        "meshes": [{ "primitives": [{ "material": 0, "attributes": {
            "POSITION": 0, "NORMAL": 1, "TANGENT": 2,
            "TEXCOORD_0": 3, "TEXCOORD_1": 4, "TEXCOORD_2": 5, "COLOR_1": 6, "TEXCOORD_99999999999": 5
        } }] }],
        "nodes": [{ "mesh": 0, "scale": [-1, 1, 1] }],
        "scenes": [{ "nodes": [0] }],
    });
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("t.bin"), &bin).unwrap();
    let mesh = load(dir.path(), "t.gltf", gltf.to_string().as_bytes());
    let p = &mesh.prims[0];
    let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6;
    assert!(
        near(p.uvs[0], [0.2, 0.4]) && near(p.uvs[2], [1.0, 1.2]),
        "set 1, transformed: {:?}",
        p.uvs
    );
    assert_eq!(
        p.uv_sets.len(),
        3,
        "set 0, set 1 (in `uvs`), set 2; a set number past any real one is ignored"
    );
    assert_eq!(
        p.uv_sets[0],
        [[0., 0.], [1., 0.], [0., 1.]],
        "the other sets as the file has them"
    );
    assert_eq!(p.uv_sets[1], Vec::<[f32; 2]>::new());
    assert_eq!(p.uv_sets[2], [[7., 8.]; 3]);
    assert_eq!(p.colors, Vec::<[f32; 4]>::new());
    assert_eq!(p.color_sets.len(), 2);
    assert_eq!(p.color_sets[1][2], [0., 0., 1., 1.]);
    // Mirrored, the tangent turns with the node and its handedness flips, so the bitangent,
    // cross(normal, tangent) * w, still points up +y as the node moves it.
    assert_eq!(p.tangents, [[-1., 0., 0., -1.]; 3]);
    assert_eq!(p.normals, [[0., 0., 1.]; 3]);

    // Written out, the sets are renumbered from 0 with no gaps, the color texture's first.
    let glb = mesh.to_glb().unwrap();
    let back = load(dir.path(), "back.glb", &glb);
    let q = &back.prims[0];
    assert_eq!(q.uvs, p.uvs);
    assert_eq!(q.uv_sets, [vec![], p.uv_sets[0].clone(), p.uv_sets[2].clone()]);
    assert_eq!(q.colors, p.color_sets[1]);
    assert_eq!(q.tangents, p.tangents);
}

#[test]
fn a_cut_out_unlit_material_keeps_its_textures_through_a_round_trip() {
    let floats: Vec<f32> = [
        &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..], // POSITION
        &[0., 0., 1., 0., 0., 1.],                 // TEXCOORD_0
        &[0., 0., 1., 0., 0., 1.],                 // TEXCOORD_1
    ]
    .concat();
    let bin: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let gltf = serde_json::json!({
        "asset": { "version": "2.0" },
        "buffers": [{ "uri": "m.bin", "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 24 },
            { "buffer": 0, "byteOffset": 60, "byteLength": 24 },
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
              "min": [0, 0, 0], "max": [1, 1, 0] },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2" },
            { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" },
        ],
        "images": [{ "uri": "a.png" }, { "uri": "b.png" }, { "uri": "c.png" }],
        "textures": [{ "source": 0 }, { "source": 1 }, { "source": 2 }],
        "materials": [{
            "alphaMode": "MASK", "alphaCutoff": 0.3,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1, 1, 1, 0.8],
                "baseColorTexture": { "index": 0 },
                "metallicRoughnessTexture": { "index": 1 },
            },
            "occlusionTexture": { "index": 1 },
            // Read through the other set: kept with the set it reads.
            "emissiveTexture": { "index": 2, "texCoord": 1 },
            "extensions": { "KHR_materials_unlit": {} },
        }],
        "meshes": [{ "primitives": [{ "material": 0,
            "attributes": { "POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2 } }] }],
        "nodes": [{ "mesh": 0 }],
        "scenes": [{ "nodes": [0] }],
    });
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("m.bin"), &bin).unwrap();
    for (i, name) in ["a.png", "b.png", "c.png"].into_iter().enumerate() {
        let png = [
            &b"\x89PNG\r\n\x1a\n"[..],
            &[0, 0, 0, 13, b'I', b'H', b'D', b'R', u8::try_from(i).unwrap()],
        ]
        .concat();
        std::fs::write(dir.path().join(name), png).unwrap();
    }
    let mesh = load(dir.path(), "m.gltf", gltf.to_string().as_bytes());
    let check = |mesh: &Mesh| {
        let m = &mesh.materials[mesh.prims[0].material];
        assert_eq!(m.alpha_cutoff, Some(0.3));
        assert!((m.color[3] - 0.8).abs() < 1e-6, "a cut-out material keeps its alpha");
        assert!(m.unlit);
        let (Some(c), Some(o)) = (m.color_image, m.occlusion_image) else {
            panic!("{m:?}")
        };
        assert_ne!(mesh.images[c], mesh.images[o]);
        assert_eq!(m.metallic_roughness_image, Some(o), "one image, two textures");
        let e = m.emissive_image.expect("the emissive texture on the other set");
        assert!(mesh.images[e] != mesh.images[c] && mesh.images[e] != mesh.images[o]);
        let set = m.sampling.emissive.uv_set.expect("its own set");
        assert_eq!(mesh.prims[0].uv_sets[set], [[0., 0.], [1., 0.], [0., 1.]]);
    };
    check(&mesh);
    let back = load(dir.path(), "back.glb", &mesh.to_glb().unwrap());
    check(&back);
}

#[test]
fn an_animation_pointer_to_a_node_moves_it() {
    let floats: Vec<f32> = [
        &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..], // POSITION
        &[0., 1.],                                 // times
        &[0., 0., 0., 2., 0., 0.],                 // translations
    ]
    .concat();
    let bin: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let pointer = |p: &str| {
        serde_json::json!({ "sampler": 0, "target": { "path": "pointer",
        "extensions": { "KHR_animation_pointer": { "pointer": p } } } })
    };
    let gltf = serde_json::json!({
        "asset": { "version": "2.0" },
        "extensionsUsed": ["KHR_animation_pointer"],
        "extensionsRequired": ["KHR_animation_pointer"],
        "buffers": [{ "uri": "a.bin", "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 8 },
            { "buffer": 0, "byteOffset": 44, "byteLength": 24 },
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
              "min": [0, 0, 0], "max": [1, 1, 0] },
            { "bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1] },
            { "bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3" },
        ],
        "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 } }] }],
        "nodes": [{ "mesh": 0 }],
        "scenes": [{ "nodes": [0] }],
        "animations": [{
            "samplers": [{ "input": 1, "output": 2 }],
            // The first is the one that counts; the others aren't a node's, or aren't
            // indices as JSON pointers write them.
            "channels": [pointer("/nodes/0/translation"), pointer("/materials/0/emissiveFactor"),
                         pointer("/nodes/00/translation"), pointer("/nodes/0/extras")],
        }],
    });
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.bin"), &bin).unwrap();
    std::fs::write(dir.path().join("a.gltf"), gltf.to_string()).unwrap();
    let doc = crate::formats::load(&dir.path().join("a.gltf"), "gltf", &Folder::new(dir.path()), Keep::ALL).unwrap();
    let mesh = doc.flatten(Pose::Animation { index: 0, time: 0.5 }).unwrap().unwrap();
    assert_eq!(mesh.prims[0].positions[0], [1., 0., 0.]);
    let rig = doc.rig().unwrap();
    let ch = &rig.animations[0].channels;
    assert_eq!(ch.len(), 1);
    assert_eq!((ch[0].node, ch[0].property), (0, crate::rig::Property::Translation));
}

#[test]
fn an_animation_pointer_shows_and_hides_nodes() {
    let floats: Vec<f32> = [
        &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..], // POSITION
        &[0., 1.],                                 // times
    ]
    .concat();
    let mut bin: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    // Node 0 is shown at the second key, when node 1 is hidden.
    bin.extend_from_slice(&[0, 1, 1, 0]);
    let visible = |node: usize, sampler: usize| {
        let pointer = format!("/nodes/{node}/extensions/KHR_node_visibility/visible");
        serde_json::json!({ "sampler": sampler, "target": { "path": "pointer",
        "extensions": { "KHR_animation_pointer": { "pointer": pointer } } } })
    };
    let gltf = serde_json::json!({
        "asset": { "version": "2.0" },
        "extensionsUsed": ["KHR_animation_pointer", "KHR_node_visibility"],
        "buffers": [{ "uri": "a.bin", "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 8 },
            { "buffer": 0, "byteOffset": 44, "byteLength": 4 },
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
              "min": [0, 0, 0], "max": [1, 1, 0] },
            { "bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1] },
            { "bufferView": 2, "componentType": 5121, "count": 2, "type": "SCALAR" },
            { "bufferView": 2, "byteOffset": 2, "componentType": 5121, "count": 2, "type": "SCALAR" },
        ],
        "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 } }] }],
        "nodes": [
            { "mesh": 0, "extensions": { "KHR_node_visibility": { "visible": false } } },
            { "mesh": 0, "translation": [5, 0, 0] },
        ],
        "scenes": [{ "nodes": [0, 1] }],
        "animations": [{
            // Linear by default, but a flag holds each key until the next.
            "samplers": [{ "input": 1, "output": 2 }, { "input": 1, "output": 3, "interpolation": "STEP" }],
            "channels": [visible(0, 0), visible(1, 1)],
        }],
    });
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.bin"), &bin).unwrap();
    std::fs::write(dir.path().join("a.gltf"), gltf.to_string()).unwrap();
    let doc = crate::formats::load(&dir.path().join("a.gltf"), "gltf", &Folder::new(dir.path()), Keep::ALL).unwrap();
    let firsts = |pose: Pose| -> Vec<[f32; 3]> {
        let mesh = doc.flatten(pose).unwrap().unwrap();
        mesh.prims
            .iter()
            .flat_map(|p| p.positions.iter().step_by(3).copied())
            .collect()
    };
    let at = |time: f32| firsts(Pose::Animation { index: 0, time });
    assert_eq!(firsts(Pose::Rest), [[5., 0., 0.]], "node 0 is hidden at rest");
    assert_eq!(at(0.0), [[5., 0., 0.]]);
    assert_eq!(at(0.75), [[5., 0., 0.]], "nothing changes between the keys");
    assert_eq!(at(1.0), [[0., 0., 0.]], "node 0 is shown and node 1 hidden");
    // The rig has no channel for them, and the animation still spans its keys.
    assert_eq!(doc.rig().unwrap().animations[0].channels.len(), 0);
    assert_eq!(doc.animation_bounds(0, 3).unwrap().unwrap().max, [6., 1., 0.]);
}

#[test]
fn reads_back_what_to_glb_writes() {
    let dir = tempfile::tempdir().unwrap();
    let mesh = grid(30);
    let glb = mesh.to_glb().unwrap();
    same(&mesh, &load(dir.path(), "plain.glb", &glb));
}

#[test]
fn streamed_glb_cut_short_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let glb = grid(60).to_glb().unwrap();
    assert!(glb.len() > 2 * 64 * 1024);
    let path = dir.path().join("cut.glb");
    std::fs::write(&path, &glb[..glb.len() - 1000]).unwrap();
    let err = crate::formats::load(&path, "glb", &Folder::new(dir.path()), Keep::ALL).unwrap_err();
    assert!(err.to_string().contains("declares"), "{err}");
}

#[test]
fn reads_meshopt_compressed_glb() {
    let dir = tempfile::tempdir().unwrap();
    let mesh = grid(30);
    let packed = crate::gltf::compress(&mesh.to_glb().unwrap()).unwrap();
    same(&mesh, &load(dir.path(), "packed.glb", &packed));
}

#[test]
fn decode_leaves_plain_gltf() {
    let dir = tempfile::tempdir().unwrap();
    let mesh = grid(30);
    let packed = crate::gltf::compress(&mesh.to_glb().unwrap()).unwrap();
    let path = dir.path().join("packed.glb");
    std::fs::write(&path, &packed).unwrap();
    let mut doc = crate::formats::load(&path, "glb", &Folder::new(dir.path()), Keep::ALL).unwrap();
    assert_eq!(doc.json()["buffers"].as_array().unwrap().len(), 2, "data and fallback");
    let before = doc.flatten(Pose::Rest).unwrap().unwrap();
    doc.decode().unwrap();
    let text = serde_json::to_string(&doc.json()).unwrap();
    assert!(!text.contains("meshopt"), "{text}");
    // The compressed data and the fallback are gone: only the decoded buffer is left.
    assert_eq!(doc.json()["buffers"].as_array().unwrap().len(), 1);
    let after = doc.flatten(Pose::Rest).unwrap().unwrap();
    assert_eq!(before.to_glb().unwrap(), after.to_glb().unwrap());
    same(&mesh, &load(dir.path(), "decoded.glb", &doc.to_glb().unwrap()));
    // Decoding plain glTF changes nothing.
    let glb = doc.to_glb().unwrap();
    doc.decode().unwrap();
    assert_eq!(glb, doc.to_glb().unwrap());
}

#[test]
fn reads_accessors_matrices_padded() {
    // MAT2 of bytes: each column 2 bytes, padded to 4. MAT3 of normalized shorts: each
    // column 6 bytes, padded to 8. A sparse VEC3 of floats with one element replaced.
    let mut bin = vec![1, 2, 0xAA, 0xAA, 3, 4, 0xAA, 0xAA];
    for c in 0..3i16 {
        for r in 0..3i16 {
            bin.extend_from_slice(&(c * 3 + r - 4).saturating_mul(8191).to_le_bytes());
        }
        bin.extend_from_slice(&[0xAA, 0xAA]);
    }
    let floats = bin.len();
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 8.0, 7.0] {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    let sparse = bin.len();
    bin.extend_from_slice(&[1, 0, 0, 0]);
    let json = serde_json::json!({
        "asset": {"version": "2.0"},
        "bufferViews": [
            {"buffer": 0, "byteLength": 8},
            {"buffer": 0, "byteOffset": 8, "byteLength": 24},
            {"buffer": 0, "byteOffset": floats, "byteLength": 24},
            {"buffer": 0, "byteOffset": floats + 24, "byteLength": 12},
            {"buffer": 0, "byteOffset": sparse, "byteLength": 4},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5121, "count": 1, "type": "MAT2"},
            {"bufferView": 1, "componentType": 5122, "normalized": true, "count": 1, "type": "MAT3"},
            {"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3",
             "sparse": {"count": 1, "indices": {"bufferView": 4, "componentType": 5121},
                        "values": {"bufferView": 3}}},
            {"bufferView": 4, "componentType": 5121, "count": 1, "type": "SCALAR"},
        ],
    });
    let mut doc = crate::document::Document::empty(serde_json::Map::new());
    *doc.json_mut() = json.as_object().unwrap().clone();
    doc.push_buffer(bin);
    assert_eq!(doc.read_floats(0).unwrap(), (4, vec![1.0, 2.0, 3.0, 4.0]));
    let (width, m3) = doc.read_floats(1).unwrap();
    assert_eq!(width, 9);
    let want: Vec<f32> = (0..9).map(|k| ((k - 4) * 8191) as f32 / 32767.0).collect();
    assert!(m3.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6), "{m3:?}");
    assert_eq!(doc.read_floats(2).unwrap(), (3, vec![1.0, 2.0, 3.0, 9.0, 8.0, 7.0]));
    assert_eq!(doc.read_indices(3).unwrap(), vec![1]);
    assert!(doc.read_indices(0).is_err(), "a matrix isn't indices");
    assert!(doc.read_floats(9).is_err());
}

#[test]
fn decode_keeps_required_extensions_it_cant_draw() {
    let dir = tempfile::tempdir().unwrap();
    let packed = crate::gltf::compress(&grid(10).to_glb().unwrap()).unwrap();
    let path = dir.path().join("packed.glb");
    std::fs::write(&path, &packed).unwrap();
    let mut doc = crate::formats::load(&path, "glb", &Folder::new(dir.path()), Keep::ALL).unwrap();
    for key in ["extensionsUsed", "extensionsRequired"] {
        let list = doc.json_mut().entry(key).or_insert_with(|| serde_json::json!([]));
        list.as_array_mut().unwrap().push("EXT_made_up".into());
    }
    assert!(
        doc.flatten(Pose::Rest).is_err(),
        "drawing needs every required extension"
    );
    doc.decode().unwrap();
    assert_eq!(doc.json()["extensionsRequired"], serde_json::json!(["EXT_made_up"]));
    assert!(!serde_json::to_string(&doc.json()).unwrap().contains("meshopt"));
}

#[test]
fn decode_widens_indices_holding_their_types_largest_value() {
    // 256 vertices on a line, u8 indices naming the last one: 255, a restart in WebGL 2.
    let dir = tempfile::tempdir().unwrap();
    let mut bin = Vec::new();
    for i in 0..256u16 {
        for v in [f32::from(i), f32::from(i % 2), 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
    }
    let index_offset = bin.len();
    bin.extend_from_slice(&[0, 1, 2, 253, 254, 255, 0, 0]);
    std::fs::write(dir.path().join("line.bin"), &bin).unwrap();
    let json = serde_json::json!({
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "line.bin", "byteLength": bin.len()}],
        "bufferViews": [
            {"buffer": 0, "byteLength": index_offset},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": 6},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 256, "type": "VEC3",
             "min": [0.0, 0.0, 0.0], "max": [255.0, 1.0, 0.0]},
            {"bufferView": 1, "componentType": 5121, "count": 6, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
    });
    let path = dir.path().join("line.gltf");
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let mut doc = crate::formats::load(&path, "gltf", &Folder::new(dir.path()), Keep::ALL).unwrap();
    let before = doc.flatten(Pose::Rest).unwrap().unwrap();
    doc.decode().unwrap();
    assert_eq!(doc.json()["accessors"][1]["componentType"], 5123);
    let after = doc.flatten(Pose::Rest).unwrap().unwrap();
    assert_eq!(before.to_glb().unwrap(), after.to_glb().unwrap());
}

#[test]
fn reads_meshopt_compressed_gltf_with_external_buffers() {
    // A compressed GLB split into a .gltf and its buffer files.
    let dir = tempfile::tempdir().unwrap();
    let mesh = grid(10);
    let packed = crate::gltf::compress(&mesh.to_glb().unwrap()).unwrap();
    let json_len = u32::from_le_bytes(packed[12..16].try_into().unwrap()) as usize;
    let mut json: serde_json::Value = serde_json::from_slice(&packed[20..20 + json_len]).unwrap();
    let bin = &packed[20 + json_len + 8..];
    json["buffers"][0]["uri"] = "packed.bin".into();
    std::fs::write(dir.path().join("packed.bin"), bin).unwrap();
    same(
        &mesh,
        &load(dir.path(), "packed.gltf", &serde_json::to_vec(&json).unwrap()),
    );
}

#[test]
fn emissive_brighter_than_one_round_trips_as_emissive_strength() {
    let mut mesh = grid(2);
    mesh.materials[0].emissive = [4.0, 2.0, 0.0];
    let doc = crate::document::Document::from_mesh(mesh, serde_json::Map::new());
    let m = &doc.json()["materials"][0];
    assert_eq!(m["emissiveFactor"], serde_json::json!([1.0, 0.5, 0.0]));
    assert_eq!(
        m["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"],
        4.0
    );
    assert_eq!(
        doc.json()["extensionsUsed"],
        serde_json::json!(["KHR_materials_emissive_strength"])
    );
    let back = doc.flatten(Pose::Rest).unwrap().unwrap();
    assert_eq!(back.materials[0].emissive, [4.0, 2.0, 0.0]);
}

#[test]
fn a_required_extension_it_cant_draw_is_an_error() {
    let mut doc = crate::document::Document::from_mesh(grid(2), serde_json::Map::new());
    doc.json_mut().insert(
        "extensionsRequired".into(),
        serde_json::json!(["KHR_materials_unlit", "KHR_techniques_webgl"]),
    );
    assert!(
        doc.flatten(Pose::Rest).unwrap().is_some(),
        "a look is drawn as best it can be"
    );
    // Interaction, behavior, metadata and promises about the file change nothing drawn.
    doc.json_mut().insert(
        "extensionsRequired".into(),
        serde_json::json!([
            "KHR_node_selectability",
            "KHR_node_hoverability",
            "KHR_interactivity",
            "EXT_structural_metadata",
            "EXT_mesh_features",
            "GODOT_single_root",
            "CESIUM_primitive_outline",
            "EXT_mesh_manifold"
        ]),
    );
    let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
    assert_eq!(mesh.triangles(), grid(2).triangles());
    doc.json_mut().insert(
        "extensionsRequired".into(),
        serde_json::json!(["EXT_some_future_geometry"]),
    );
    let err = doc.flatten(Pose::Rest).unwrap_err().to_string();
    assert!(err.contains("EXT_some_future_geometry"), "{err}");
}

/// A document of a unit square's four corners (accessor 0), with index accessors 1
/// (`[3, 2, 1, 0]`), 2 (two triangles) and 3 (`[0, 0, 1, 2, 3]`), drawn by `meshes` on
/// `nodes`.
fn corners(meshes: serde_json::Value, nodes: serde_json::Value) -> crate::document::Document {
    let mut bin: Vec<u8> = [[0f32, 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    bin.extend([3u16, 2, 1, 0].iter().flat_map(|i| i.to_le_bytes()));
    bin.extend([0u8, 1, 2, 0, 2, 3, 0, 0, 1, 2, 3, 0]);
    let n = (0..nodes.as_array().unwrap().len()).collect::<Vec<_>>();
    let mut json = serde_json::json!({
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteLength": 48 },
            { "buffer": 0, "byteOffset": 48, "byteLength": 8 },
            { "buffer": 0, "byteOffset": 56, "byteLength": 6 },
            { "buffer": 0, "byteOffset": 62, "byteLength": 5 },
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
              "min": [0, 0, 0], "max": [1, 1, 0] },
            { "bufferView": 1, "componentType": 5123, "count": 4, "type": "SCALAR" },
            { "bufferView": 2, "componentType": 5121, "count": 6, "type": "SCALAR" },
            { "bufferView": 3, "componentType": 5121, "count": 5, "type": "SCALAR" },
        ],
        "scenes": [{ "nodes": n }],
    });
    json["meshes"] = meshes;
    json["nodes"] = nodes;
    let mut doc = crate::document::Document::empty(serde_json::Map::new());
    let serde_json::Value::Object(json) = json else {
        unreachable!()
    };
    *doc.json_mut() = json;
    doc.push_buffer(bin);
    doc
}

#[test]
fn point_only_mesh_flattens_with_points() {
    let meshes = serde_json::json!([{ "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 0 }] }]);
    let mut doc = corners(meshes, serde_json::json!([{ "mesh": 0 }]));
    let mesh = doc.flatten(Pose::Rest).unwrap().expect("points are drawn");
    assert_eq!(mesh.prims.len(), 1);
    let p = &mesh.prims[0];
    assert_eq!(p.points, [0, 1, 2, 3]);
    assert!(p.indices.is_empty() && p.lines.is_empty());
    assert_eq!(p.positions.len(), 4);
    assert!(p.normals.is_empty(), "no normals are made up for points");
    assert_eq!((mesh.triangles(), mesh.lines(), mesh.points()), (0, 0, 4));

    doc.set_max_points(3);
    let err = doc.flatten(Pose::Rest).unwrap_err().to_string();
    assert!(err.contains("too many points"), "{err}");
    doc.set_lines_and_points(false);
    assert!(doc.flatten(Pose::Rest).unwrap().is_none(), "left out, nothing is drawn");
}

#[test]
fn strips_and_loops_expand_to_segments() {
    let meshes = serde_json::json!([
        { "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 3, "indices": 1 }] },
        { "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 2 }] },
        { "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 1, "indices": 3 }] },
        { "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 1 }] },
    ]);
    let nodes = serde_json::json!([{ "mesh": 0 }, { "mesh": 1 }, { "mesh": 2 }, { "mesh": 3 }]);
    let mesh = corners(meshes, nodes).flatten(Pose::Rest).unwrap().unwrap();
    let lines: Vec<&[u32]> = mesh.prims.iter().map(|p| p.lines.as_slice()).collect();
    assert_eq!(lines[0], [3, 2, 2, 1, 1, 0], "a strip: each index to the next");
    assert_eq!(lines[1], [0, 1, 1, 2, 2, 3, 3, 0], "a loop: closed back to the first");
    assert_eq!(lines[2], [1, 2], "a segment to itself and an odd index out are dropped");
    assert_eq!(
        lines[3],
        [0, 1, 2, 3],
        "plain lines, without indices, pair the vertices"
    );
    assert_eq!(mesh.lines(), 3 + 4 + 1 + 2);
    assert_eq!(mesh.triangles(), 0);
}

#[test]
fn node_transforms_move_lines_and_points() {
    let meshes = serde_json::json!([{ "primitives": [
        { "attributes": { "POSITION": 0 }, "indices": 2 },
        { "attributes": { "POSITION": 0 }, "mode": 3, "indices": 1 },
        { "attributes": { "POSITION": 0 }, "mode": 0 },
    ] }]);
    let nodes = serde_json::json!([
        { "mesh": 0, "translation": [10, 0, 0] },
        { "mesh": 0, "scale": [-1, 1, 1] },
    ]);
    let mesh = corners(meshes, nodes).flatten(Pose::Rest).unwrap().unwrap();
    // Both nodes draw the mesh: one primitive, the vertices once in each place.
    assert_eq!(
        mesh.prims.len(),
        1,
        "lines and points on a primitive's vertices join it"
    );
    let p = &mesh.prims[0];
    assert_eq!(p.positions[2], [11.0, 1.0, 0.0], "moved");
    assert_eq!(p.positions[5], [-1.0, 0.0, 0.0], "mirrored");
    assert_eq!(
        p.indices,
        [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6],
        "a mirror flips triangles back"
    );
    assert_eq!(p.lines, [3, 2, 2, 1, 1, 0, 7, 6, 6, 5, 5, 4], "but not lines");
    assert_eq!(p.points, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!((mesh.triangles(), mesh.lines(), mesh.points()), (4, 6, 8));
}

#[test]
fn instances_repeat_lines_and_points() {
    let meshes = serde_json::json!([{ "primitives": [
        { "attributes": { "POSITION": 0 }, "mode": 1, "indices": 1 },
        { "attributes": { "POSITION": 0 }, "mode": 0 },
    ] }]);
    let mut doc = corners(meshes, serde_json::json!([{ "mesh": 0 }]));
    // Two instances, 5 apart along x.
    let bin: Vec<u8> = [0f32, 0., 0., 5., 0., 0.]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let b = doc.push_buffer(bin);
    let json = doc.json_mut();
    json["buffers"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({ "byteLength": 24 }));
    json["bufferViews"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({ "buffer": b, "byteLength": 24 }));
    json["accessors"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({ "bufferView": 4, "componentType": 5126, "count": 2, "type": "VEC3" }));
    json["nodes"][0]["extensions"] =
        serde_json::json!({ "EXT_mesh_gpu_instancing": { "attributes": { "TRANSLATION": 4 } } });
    json.insert("extensionsUsed".into(), serde_json::json!(["EXT_mesh_gpu_instancing"]));
    let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
    assert_eq!(mesh.prims.len(), 1);
    let p = &mesh.prims[0];
    assert_eq!(p.positions.len(), 8);
    assert_eq!(p.positions[5], [6.0, 0.0, 0.0]);
    assert_eq!(
        p.lines,
        [3, 2, 1, 0, 7, 6, 5, 4],
        "each instance's segments on its own vertices"
    );
    assert_eq!(p.points, [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn glb_round_trip_keeps_lines_and_points() {
    let mut mesh = grid(2);
    mesh.prims[0].lines = vec![0, 1, 1, 2, 2, 5];
    mesh.prims[0].points = vec![4, 8];
    mesh.prims[1].indices.clear();
    mesh.prims[1].normals.clear();
    mesh.prims[1].points = (0..9).collect();
    let glb = mesh.to_glb().unwrap();
    let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let json: serde_json::Value = serde_json::from_slice(&glb[20..20 + len]).unwrap();
    let prims = json["meshes"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["primitives"].as_array().unwrap().clone())
        .collect::<Vec<_>>();
    let modes: Vec<u64> = prims.iter().map(|p| p["mode"].as_u64().unwrap_or(4)).collect();
    assert_eq!(modes, [4, 1, 0, 0]);
    assert_eq!(prims[0]["attributes"], prims[1]["attributes"], "one set of vertices");
    assert_eq!(prims[0]["attributes"], prims[2]["attributes"]);
    assert_eq!(prims[1]["material"], prims[0]["material"]);
    let lines = &json["accessors"][usize::try_from(prims[1]["indices"].as_u64().unwrap()).unwrap()];
    assert_eq!(
        (lines["componentType"].as_u64(), lines["count"].as_u64()),
        (Some(5123), Some(6))
    );
    assert!(prims[3].get("indices").is_none(), "a point cloud needs no indices");

    let none = crate::assets::Memory::new();
    let back = crate::formats::load_bytes(&glb, "m.glb", "glb", &none, Keep::ALL)
        .unwrap()
        .flatten(Pose::Rest)
        .unwrap()
        .unwrap();
    assert_eq!(
        back.prims.len(),
        2,
        "lines and points join the triangles they share vertices with"
    );
    assert_eq!(back.prims[0].indices, mesh.prims[0].indices);
    assert_eq!(back.prims[0].lines, mesh.prims[0].lines);
    assert_eq!(back.prims[0].points, mesh.prims[0].points);
    assert_eq!(back.prims[0].positions, mesh.prims[0].positions);
    assert_eq!(back.prims[1].points, mesh.prims[1].points);
    assert!(back.prims[1].normals.is_empty(), "no normals are made up for points");
    assert!(back.to_glb().unwrap() == glb, "written again, the same bytes");
}

/// The box around a flattened mesh's positions.
fn box_of(mesh: Option<&Mesh>) -> Option<crate::document::Bounds> {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for v in mesh.into_iter().flat_map(|m| &m.prims).flat_map(|p| &p.positions) {
        for c in 0..3 {
            lo[c] = lo[c].min(v[c]);
            hi[c] = hi[c].max(v[c]);
        }
    }
    (lo[0] <= hi[0]).then_some(crate::document::Bounds { min: lo, max: hi })
}

#[cfg(all(feature = "gltf", feature = "fs"))]
#[test]
fn bounds_hold_what_flatten_draws_in_every_pose() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf");
    let cases = [
        "skin_animated/skin.glb",
        "skin_rest_pose",
        "skin_eight_influences",
        "blender_shape_key/shape_key.glb",
        "morph_default_weights",
        "gpu_instancing/inst.glb",
        "mirror_x/cube.glb",
        "node_hierarchy",
        "node_matrix",
        "points_and_lines",
        "quantized_normalized_i16",
        "sparse",
        "blender_draco/draco.glb",
    ];
    for case in cases {
        let mut path = root.join(case);
        if path.is_dir() {
            path = std::fs::read_dir(&path)
                .unwrap()
                .map(|e| e.unwrap().path())
                .find(|p| p.extension().is_some_and(|e| e == "glb" || e == "gltf"))
                .unwrap();
        }
        let doc = crate::open(&path).unwrap();
        let mut poses = vec![Pose::Rest, Pose::Bind];
        let animations = super::list(doc.json(), "animations").len();
        poses.extend((0..animations).map(|index| Pose::Animation { index, time: 0.4 }));
        for pose in poses {
            let Ok(mesh) = doc.flatten(pose) else { continue };
            let want = box_of(mesh.as_ref());
            assert_eq!(doc.bounds(pose).unwrap(), want, "{case} in {pose:?}");
            assert!(want.is_some(), "{case} draws something");
        }
    }
}

#[cfg(all(feature = "gltf", feature = "fs"))]
#[test]
fn animation_bounds_hold_every_sampled_pose() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf/skin_animated/skin.glb");
    let doc = crate::open(&path).unwrap();
    let (first, last) = doc.rig().unwrap().animations[0].span();
    assert!(first < last);
    let all = doc.animation_bounds(0, 5).unwrap().unwrap();
    for k in 0..5u8 {
        let time = first + (last - first) * f32::from(k) / 4.0;
        let b = doc.bounds(Pose::Animation { index: 0, time }).unwrap().unwrap();
        assert!(
            (0..3).all(|c| all.min[c] <= b.min[c] && b.max[c] <= all.max[c]),
            "{b:?} in {all:?}"
        );
    }
    let start = doc.bounds(Pose::Animation { index: 0, time: first }).unwrap();
    assert_eq!(doc.animation_bounds(0, 1).unwrap(), start);
    assert_eq!(doc.animation_bounds(0, 0).unwrap(), start);
    assert!(doc.animation_bounds(1, 5).is_err());
}

#[test]
fn bounds_and_vertices_count_every_instance() {
    let meshes = serde_json::json!([{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 2 }] }]);
    let doc = corners(
        meshes,
        serde_json::json!([{ "mesh": 0 }, { "mesh": 0, "translation": [5, 0, -2], "rotation": [0, 0, 0.707_106_77, 0.707_106_77] }]),
    );
    let b = doc.bounds(Pose::Rest).unwrap().unwrap();
    assert_eq!(b, box_of(doc.flatten(Pose::Rest).unwrap().as_ref()).unwrap());
    let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
    assert!(near(b.min, [0.0, 0.0, -2.0]) && near(b.max, [5.0, 1.0, 0.0]), "{b:?}");
    assert!((b.center()[0] - 2.5).abs() < 1e-6 && (b.size()[2] - 2.0).abs() < 1e-6);
    let c = doc.counts();
    assert_eq!((c.vertices, c.triangles), (8, 4));
    let empty = corners(serde_json::json!([]), serde_json::json!([]));
    assert_eq!(empty.bounds(Pose::Rest).unwrap(), None);
}

#[cfg(feature = "draco")]
#[test]
fn draco_compressed_accessors_read_as_their_stream_decodes() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf/blender_draco/draco.glb");
    let dir = path.parent().unwrap();
    let doc = crate::formats::load(&path, "glb", &Folder::new(dir), Keep::ALL).unwrap();
    let mut decoded = crate::formats::load(&path, "glb", &Folder::new(dir), Keep::ALL).unwrap();
    decoded.decode().unwrap();
    let prim = &doc.json()["meshes"][0]["primitives"][0];
    let position = usize::try_from(prim["attributes"]["POSITION"].as_u64().unwrap()).unwrap();
    let indices = usize::try_from(prim["indices"].as_u64().unwrap()).unwrap();
    assert!(
        doc.json()["accessors"][position].get("bufferView").is_none(),
        "the data is in the stream"
    );
    let (n, floats) = doc.read_floats(position).unwrap();
    assert_eq!((n, &floats), (3, &decoded.read_floats(position).unwrap().1));
    assert!(floats.iter().any(|&f| f != 0.0), "not zeros");
    assert_eq!(
        doc.read_indices(indices).unwrap(),
        decoded.read_indices(indices).unwrap()
    );
}

#[test]
fn a_volumes_thickness_scales_with_its_node() {
    let prim =
        |material: u32| serde_json::json!({ "attributes": { "POSITION": 0 }, "indices": 2, "material": material });
    let meshes = serde_json::json!([{ "primitives": [prim(0), prim(1)] }]);
    let h = std::f64::consts::FRAC_1_SQRT_2;
    // As written, a quarter the size twice (once turned, under a parent), stretched by 1, 2 and 4, and mirrored.
    let nodes = serde_json::json!([
        { "mesh": 0 },
        { "mesh": 0, "scale": [0.25, 0.25, 0.25] },
        { "children": [3], "scale": [0.5, 0.5, 0.5] },
        { "mesh": 0, "rotation": [0.0, h, 0.0, h], "scale": [0.5, 0.5, 0.5] },
        { "mesh": 0, "scale": [1.0, 2.0, 4.0] },
        { "mesh": 0, "scale": [-1.0, 1.0, 1.0] },
    ]);
    let mut doc = corners(meshes, nodes);
    doc.json_mut()["scenes"] = serde_json::json!([{ "nodes": [0, 1, 2, 4, 5] }]);
    let volume = |thickness: f64| serde_json::json!({ "thicknessFactor": thickness, "attenuationDistance": 3.0 });
    doc.json_mut().insert(
        "materials".into(),
        serde_json::json!([
            { "name": "thick", "extensions": { "KHR_materials_volume": volume(2.0) } },
            { "name": "thin", "extensions": { "KHR_materials_volume": volume(0.0) } },
        ]),
    );
    let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
    let volumes: Vec<(&str, f32, Option<f32>)> = mesh
        .materials
        .iter()
        .map(|m| {
            let v = m.extensions.as_ref().unwrap().volume.as_ref().unwrap();
            (m.name.as_str(), v.thickness.factor, v.attenuation_distance)
        })
        .collect();
    // The thick material once for each scale, the thin one once; the attenuation distance is the world's.
    let d = Some(3.0);
    assert_eq!(
        volumes,
        [
            ("thick", 2.0, d),
            ("thin", 0.0, d),
            ("thick", 0.5, d),
            ("thick", 4.0, d)
        ]
    );
    // The mesh is read once for each scale, and drawn at every node of that scale.
    let drawn: Vec<(usize, usize)> = mesh.prims.iter().map(|p| (p.material, p.indices.len() / 3)).collect();
    assert_eq!(drawn, [(0, 4), (1, 4), (2, 4), (1, 4), (3, 2), (1, 2)]);
    assert_eq!(mesh.triangles(), doc.counts().triangles);
}
