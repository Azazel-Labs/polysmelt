#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    reason = "small test values, compared exactly where they pass through unchanged"
)]

use serde_json::{Value, json};

use super::super::{detect, load_bytes};
use super::{i3dm, oct_decode, tileset};
use crate::assets::Memory;
use crate::document::{Document, Keep, Pose};
use crate::{ErrorKind, Format, Result};

/// `v` padded to a multiple of eight bytes from `lead`.
fn pad(mut v: Vec<u8>, fill: u8, lead: usize) -> Vec<u8> {
    while !(lead + v.len()).is_multiple_of(8) {
        v.push(fill);
    }
    v
}

/// A tile: `format` is an i3dm's `gltfFormat`.
fn tile(magic: &[u8; 4], ft: &Value, fbin: &[u8], bt: Option<&Value>, bbin: &[u8], body: &[u8]) -> Vec<u8> {
    tile_with(magic, ft, fbin, bt, bbin, body, 1)
}

fn tile_with(
    magic: &[u8; 4],
    ft: &Value,
    fbin: &[u8],
    bt: Option<&Value>,
    bbin: &[u8],
    body: &[u8],
    format: u32,
) -> Vec<u8> {
    let head = if magic == b"i3dm" { 32 } else { 28 };
    let fj = if ft.is_null() {
        Vec::new()
    } else {
        pad(ft.to_string().into_bytes(), b' ', head)
    };
    let fbin = pad(fbin.to_vec(), 0, 0);
    let bj = bt.map_or(Vec::new(), |b| pad(b.to_string().into_bytes(), b' ', 0));
    let bbin = pad(bbin.to_vec(), 0, 0);
    let total = head + fj.len() + fbin.len() + bj.len() + bbin.len() + body.len();
    let mut out = magic.to_vec();
    for n in [1, total, fj.len(), fbin.len(), bj.len(), bbin.len()] {
        out.extend((n as u32).to_le_bytes());
    }
    if magic == b"i3dm" {
        out.extend(format.to_le_bytes());
    }
    for part in [&fj[..], &fbin, &bj, &bbin, body] {
        out.extend(part);
    }
    out
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A binary glTF of one triangle, (0,0,0), (1,0,0), (0,2,0), with `more` merged into its JSON.
fn glb_with(more: &Value) -> Vec<u8> {
    let mut g = json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 } }] }],
        "accessors": [{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                        "min": [0, 0, 0], "max": [1, 2, 0] }],
        "bufferViews": [{ "buffer": 0, "byteLength": 36 }],
        "buffers": [{ "byteLength": 36 }],
    });
    for (k, v) in more.as_object().into_iter().flatten() {
        g[k] = v.clone();
    }
    let mut text = g.to_string().into_bytes();
    while !text.len().is_multiple_of(4) {
        text.push(b' ');
    }
    let bin = floats(&[0., 0., 0., 1., 0., 0., 0., 2., 0.]);
    let mut out = b"glTF".to_vec();
    out.extend(2u32.to_le_bytes());
    out.extend(((28 + text.len() + bin.len()) as u32).to_le_bytes());
    out.extend((text.len() as u32).to_le_bytes());
    out.extend(b"JSON");
    out.extend(&text);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(b"BIN\0");
    out.extend(&bin);
    out
}

fn glb() -> Vec<u8> {
    glb_with(&Value::Null)
}

fn cmpt(tiles: &[Vec<u8>]) -> Vec<u8> {
    let len: usize = 16 + tiles.iter().map(Vec::len).sum::<usize>();
    let mut out = b"cmpt".to_vec();
    for n in [1, len, tiles.len()] {
        out.extend((n as u32).to_le_bytes());
    }
    for t in tiles {
        out.extend(t);
    }
    out
}

fn load_with(bytes: &[u8], name: &str, assets: &Memory, keep: Keep) -> Result<Document> {
    load_bytes(bytes, name, name.rsplit('.').next().unwrap(), assets, keep)
}

fn load(bytes: &[u8], name: &str) -> Result<Document> {
    load_with(bytes, name, &Memory::new(), Keep::ALL)
}

fn kind(bytes: &[u8], name: &str) -> ErrorKind {
    load(bytes, name).expect_err("an error").kind()
}

/// A point cloud's attribute: its width and values.
fn attribute(doc: &Document, name: &str) -> (usize, Vec<f32>) {
    let a = &doc.json()["meshes"][0]["primitives"][0]["attributes"][name];
    doc.read_floats(usize::try_from(a.as_u64().unwrap_or_else(|| panic!("no {name}"))).unwrap())
        .unwrap()
}

fn bounds(doc: &Document) -> ([f32; 3], [f32; 3]) {
    let b = doc.bounds(Pose::Rest).unwrap().unwrap();
    (b.min, b.max)
}

/// A point cloud of `n` points whose positions are `positions`, with `more` in its feature table and
/// `fbin` after the positions.
fn cloud(positions: &[f32], more: &Value, fbin: &[u8], bt: Option<&Value>, bbin: &[u8]) -> Vec<u8> {
    let mut ft = json!({ "POINTS_LENGTH": positions.len() / 3, "POSITION": { "byteOffset": 0 } });
    for (k, v) in more.as_object().into_iter().flatten() {
        ft[k] = v.clone();
    }
    let mut body = floats(positions);
    body.extend(fbin);
    tile(b"pnts", &ft, &body, bt, bbin, &[])
}

// --- headers -----------------------------------------------------------------------------------

#[test]
fn a_b3dm_reads_the_same_under_each_header_layout() {
    let table = json!({ "id": [5, 6], "name": ["a", "b"] });
    let current = tile(b"b3dm", &json!({ "BATCH_LENGTH": 2 }), &[], Some(&table), &[], &glb());
    let bj = pad(table.to_string().into_bytes(), b' ', 0);
    let legacy = |fields: &[usize]| {
        let total = 12 + fields.len() * 4 + bj.len() + glb().len();
        let mut out = b"b3dm".to_vec();
        for n in [1, total].iter().chain(fields) {
            out.extend((*n as u32).to_le_bytes());
        }
        out.extend(&bj);
        out.extend(glb());
        out
    };
    // [batchLength][batchTableByteLength], then [batchTableJsonByteLength][batchTableBinaryByteLength]
    // [batchLength].
    let first = legacy(&[2, bj.len()]);
    let second = legacy(&[bj.len(), 0, 2]);
    let want = load(&current, "t.b3dm").unwrap();
    let node = &want.json()["nodes"][1];
    assert_eq!(node["extras"]["batchLength"], 2);
    assert_eq!(node["extras"]["batchTable"], table);
    assert_eq!(node["extras"]["tile"], "b3dm");
    for other in [first, second] {
        let got = load(&other, "t.b3dm").unwrap();
        assert_eq!(got.json()["nodes"], want.json()["nodes"]);
        assert_eq!(got.facts, want.facts);
    }
    assert_eq!(want.facts["features"], 2);
    assert_eq!(want.facts["triangles"], 1);
}

#[test]
fn a_header_that_lies_is_an_error() {
    let good = tile(b"b3dm", &json!({ "BATCH_LENGTH": 0 }), &[], None, &[], &glb());
    // Each length in turn larger than the file, and as large as a length gets.
    for field in 0..4 {
        for value in [good.len() as u32, 0x21FF_FFFF, u32::MAX] {
            let mut bad = good.clone();
            bad[12 + field * 4..16 + field * 4].copy_from_slice(&value.to_le_bytes());
            assert!(load(&bad, "t.b3dm").is_err(), "length {field} set to {value}");
        }
    }
    let mut bad = good.clone();
    bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(kind(&bad, "t.b3dm"), ErrorKind::Truncated);
    bad[8..12].copy_from_slice(&4u32.to_le_bytes());
    assert_eq!(kind(&bad, "t.b3dm"), ErrorKind::Corrupt);
    let mut bad = good.clone();
    bad[4] = 2;
    assert_eq!(kind(&bad, "t.b3dm"), ErrorKind::Unsupported);
    let mut bad = good.clone();
    bad[28] = b'[';
    assert_eq!(kind(&bad, "t.b3dm"), ErrorKind::Corrupt);
    // The tables alone, with no glTF after them.
    let bare = tile(b"b3dm", &json!({ "BATCH_LENGTH": 0 }), &[], None, &[], &[]);
    assert_eq!(kind(&bare, "t.b3dm"), ErrorKind::Truncated);
    assert_eq!(kind(b"b3dm", "t.b3dm"), ErrorKind::Truncated);
    assert_eq!(kind(b"nothing like a tile", "t.b3dm"), ErrorKind::Corrupt);
    assert_eq!(kind(b"vctr\x01\0\0\0", "t.b3dm"), ErrorKind::Unsupported);
}

#[test]
fn damaged_tiles_fail_without_a_panic() {
    let points = cloud(
        &[0., 0., 0., 1., 2., 3.],
        &json!({ "RGB": { "byteOffset": 24 }, "BATCH_ID": { "byteOffset": 30 }, "BATCH_LENGTH": 2 }),
        &[1, 2, 3, 4, 5, 6, 0, 0, 1, 0],
        Some(&json!({ "h": { "byteOffset": 0, "componentType": "FLOAT", "type": "SCALAR" } })),
        &floats(&[1.5, 2.5]),
    );
    let instanced = tile(
        b"i3dm",
        &json!({ "INSTANCES_LENGTH": 2, "POSITION": { "byteOffset": 0 }, "SCALE": { "byteOffset": 24 } }),
        &floats(&[0., 0., 0., 5., 0., 0., 1., 2.]),
        Some(&json!({ "n": [1, 2] })),
        &[],
        &glb(),
    );
    let batched = tile(
        b"b3dm",
        &json!({ "BATCH_LENGTH": 1 }),
        &[],
        Some(&json!({ "n": [1] })),
        &[],
        &glb(),
    );
    let composite = cmpt(&[points.clone(), instanced.clone(), batched.clone()]);
    let mut rng = crate::test_rng::Pcg32::new(7);
    for (name, good) in [
        ("t.pnts", points),
        ("t.i3dm", instanced),
        ("t.b3dm", batched),
        ("t.cmpt", composite),
    ] {
        load(&good, name).unwrap();
        // Cut short anywhere: an error, since the header says how long the tile is.
        for len in 0..good.len() {
            assert!(load(&good[..len], name).is_err(), "{name} cut to {len} bytes");
        }
        for _ in 0..400 {
            let mut bad = good.clone();
            let k = 1 + rng.below(4) as usize;
            crate::formats::points::corrupt(&mut bad, &mut rng, k);
            let _ = load(&bad, name).map(|doc| doc.flatten(Pose::Rest));
        }
    }
}

// --- tables ------------------------------------------------------------------------------------

#[test]
fn counts_and_offsets_the_file_cannot_hold_are_errors() {
    let p = [0.0f32; 3];
    let with = |ft: Value| tile(b"pnts", &ft, &floats(&p), None, &[], &[]);
    let position = json!({ "byteOffset": 0 });
    assert_eq!(
        kind(&with(json!({ "POINTS_LENGTH": 2, "POSITION": position })), "t.pnts"),
        ErrorKind::Truncated
    );
    assert_eq!(
        kind(
            &with(json!({ "POINTS_LENGTH": 4_000_000_000_u64, "POSITION": position })),
            "t.pnts"
        ),
        ErrorKind::LimitExceeded
    );
    for offset in [json!(8), json!(u64::MAX), json!(-4), json!("0")] {
        let ft = json!({ "POINTS_LENGTH": 1, "POSITION": { "byteOffset": offset } });
        assert!(load(&with(ft), "t.pnts").is_err(), "offset {offset}");
    }
    for count in [
        json!(-1),
        json!(1.5),
        json!("1"),
        json!([1, 2]),
        json!(5_000_000_000_u64),
    ] {
        let ft = json!({ "POINTS_LENGTH": count, "POSITION": position });
        assert_eq!(kind(&with(ft), "t.pnts"), ErrorKind::Corrupt, "count {count}");
    }
    for center in [json!([1, 2]), json!([1, 2, "3"]), json!(7), json!({ "byteOffset": 8 })] {
        let ft = json!({ "POINTS_LENGTH": 1, "POSITION": position, "RTC_CENTER": center });
        assert!(load(&with(ft), "t.pnts").is_err(), "center {center}");
    }
    assert_eq!(
        kind(&with(json!({ "POSITION": position })), "t.pnts"),
        ErrorKind::Corrupt
    );
    assert_eq!(kind(&with(json!({ "POINTS_LENGTH": 1 })), "t.pnts"), ErrorKind::Corrupt);
    // More points than the load allows are refused before anything is read.
    let two = cloud(&[0.; 6], &Value::Null, &[], None, &[]);
    let few = Keep::ALL.with_max_points(1);
    let e = load_with(&two, "t.pnts", &Memory::new(), few).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LimitExceeded);
}

#[test]
fn batch_table_properties_of_every_component_type_are_decoded() {
    let mut bbin = Vec::new();
    bbin.extend([0x80u8, 0x7F]); // BYTE at 0
    bbin.extend([0xFFu8, 1]); // UNSIGNED_BYTE at 2
    bbin.extend((-2i16).to_le_bytes()); // SHORT at 4
    bbin.extend(3i16.to_le_bytes());
    bbin.extend(65535u16.to_le_bytes()); // UNSIGNED_SHORT at 8
    bbin.extend(4u16.to_le_bytes());
    bbin.extend((-70000i32).to_le_bytes()); // INT at 12
    bbin.extend(5i32.to_le_bytes());
    bbin.extend(4_000_000_000_u32.to_le_bytes()); // UNSIGNED_INT at 20
    bbin.extend(6u32.to_le_bytes());
    bbin.extend(floats(&[0.5, -0.25])); // FLOAT at 28
    bbin.extend([0u8; 4]);
    bbin.extend(1e100f64.to_le_bytes()); // DOUBLE at 40
    bbin.extend((-7.0f64).to_le_bytes());
    bbin.extend(floats(&[1., 2., 3., 4., 5., 6.])); // FLOAT VEC3 at 56
    let property =
        |at: usize, component: &str, kind: &str| json!({ "byteOffset": at, "componentType": component, "type": kind });
    let table = json!({
        "i8": property(0, "BYTE", "SCALAR"),
        "u8": property(2, "UNSIGNED_BYTE", "SCALAR"),
        "i16": property(4, "SHORT", "SCALAR"),
        "u16": property(8, "UNSIGNED_SHORT", "SCALAR"),
        "i32": property(12, "INT", "SCALAR"),
        "u32": property(20, "UNSIGNED_INT", "SCALAR"),
        "f32": property(28, "FLOAT", "SCALAR"),
        "f64": property(40, "DOUBLE", "SCALAR"),
        "v3": property(56, "FLOAT", "VEC3"),
        "pair": property(0, "UNSIGNED_BYTE", "VEC2"),
        "listed": ["x", null],
        "extras": { "note": 1 },
    });
    let b3dm = tile(b"b3dm", &json!({ "BATCH_LENGTH": 2 }), &[], Some(&table), &bbin, &glb());
    let doc = load(&b3dm, "t.b3dm").unwrap();
    let got = &doc.json()["nodes"][1]["extras"]["batchTable"];
    assert_eq!(
        *got,
        json!({
            "i8": [-128.0, 127.0], "u8": [255.0, 1.0], "i16": [-2.0, 3.0], "u16": [65535.0, 4.0],
            "i32": [-70000.0, 5.0], "u32": [4_000_000_000.0, 6.0], "f32": [0.5, -0.25], "f64": [1e100, -7.0],
            "v3": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "pair": [[128.0, 127.0], [255.0, 1.0]],
            "listed": ["x", null], "extras": { "note": 1 },
        })
    );
    // Without extras nothing is decoded or kept, and nothing needs a node over the glTF.
    let bare = load_with(&b3dm, "t.b3dm", &Memory::new(), Keep::ALL.with_extras(false)).unwrap();
    assert_eq!(bare.json()["nodes"].as_array().unwrap().len(), 1);

    // A type or component type that isn't one, and a property past the body.
    for bad in [
        property(0, "HALF", "SCALAR"),
        property(0, "FLOAT", "MAT4"),
        property(72, "DOUBLE", "SCALAR"),
        json!({ "byteOffset": 0 }),
    ] {
        let t = tile(
            b"b3dm",
            &json!({ "BATCH_LENGTH": 2 }),
            &[],
            Some(&json!({ "p": bad })),
            &bbin,
            &glb(),
        );
        assert!(load(&t, "t.b3dm").is_err(), "{bad}");
    }
}

#[test]
fn a_batch_table_too_large_for_the_extras_is_left_out_by_name() {
    let n = super::table::MAX_TABLE_VALUES / 2;
    let table = json!({
        "big": { "byteOffset": 0, "componentType": "UNSIGNED_BYTE", "type": "SCALAR" },
        "small": { "byteOffset": 0, "componentType": "UNSIGNED_BYTE", "type": "VEC4" },
    });
    let t = tile(
        b"b3dm",
        &json!({ "BATCH_LENGTH": n }),
        &[],
        Some(&table),
        &vec![3u8; n * 4],
        &glb(),
    );
    let doc = load(&t, "t.b3dm").unwrap();
    let got = &doc.json()["nodes"][1]["extras"]["batchTable"];
    // The first fits; with it, the second is over the budget.
    assert!(got["big"].is_array());
    assert_eq!(got["batchTableDropped"], json!(["small"]));
    assert!(got.get("small").is_none());
}

// --- point clouds ------------------------------------------------------------------------------

#[test]
fn points_keep_the_files_values_and_the_node_carries_the_center() {
    let center = [1_215_012.882_887_673_8, -4_736_313.051_199_594, 4_081_605.221_260_42];
    let t = cloud(
        &[0.5, 1.5, 2.5, -1., -2., -3.],
        &json!({ "RTC_CENTER": center }),
        &[],
        None,
        &[],
    );
    let doc = load(&t, "t.pnts").unwrap();
    // Z up to Y up: (x, y, z) to (x, z, -y), in double precision for the center.
    assert_eq!(
        doc.json()["nodes"][0]["translation"],
        json!([center[0], center[2], -center[1]])
    );
    assert_eq!(attribute(&doc, "POSITION").1, [0.5, 2.5, -1.5, -1., -3., 2.]);
    assert_eq!(doc.facts["points"], 2);
    let info = doc.source_info();
    assert_eq!(info.format.as_deref(), Some("3D Tiles"));
    assert_eq!(info.up_axis, crate::source::Fact::Declared(crate::source::Axis::PosZ));
    assert!(info.converted.up_axis && !info.converted.units);

    // Far from the origin with no center: measured from a whole number near the middle, exactly.
    let far = [6_378_137.0f32, 100.0, -2_000_000.5];
    let t = cloud(
        &[far[0], far[1], far[2], far[0] + 10.0, far[1], far[2]],
        &Value::Null,
        &[],
        None,
        &[],
    );
    let doc = load(&t, "t.pnts").unwrap();
    let origin: Vec<f64> = doc.json()["nodes"][0]["translation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(origin, [6_378_142.0, -2_000_001.0, 0.0]);
    let local = attribute(&doc, "POSITION").1;
    let back: Vec<f32> = local
        .iter()
        .enumerate()
        .map(|(i, &p)| (origin[i % 3] + f64::from(p)) as f32)
        .collect();
    assert_eq!(back, [far[0], far[2], -far[1], far[0] + 10.0, far[2], -far[1]]);
}

#[test]
fn quantized_points_are_scaled_through_their_volume() {
    let q: Vec<u8> = [0u16, 65535, 13107, 65535, 0, 32768]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ft = json!({
        "POINTS_LENGTH": 2, "POSITION_QUANTIZED": { "byteOffset": 0 },
        "QUANTIZED_VOLUME_OFFSET": [100.0, 200.0, 300.0], "QUANTIZED_VOLUME_SCALE": [10.0, 20.0, 5.0],
        "RTC_CENTER": [1000.0, 0.0, 0.0],
    });
    let doc = load(&tile(b"pnts", &ft, &q, None, &[], &[]), "t.pnts").unwrap();
    // The offset and the center are the node's; the positions are the scaled steps alone.
    assert_eq!(doc.json()["nodes"][0]["translation"], json!([1100.0, 300.0, -200.0]));
    let want = [
        0.0,
        (13107.0f64 * 5.0 / 65535.0) as f32,
        -20.0,
        10.0,
        (32768.0f64 * 5.0 / 65535.0) as f32,
        -0.0,
    ];
    assert_eq!(attribute(&doc, "POSITION").1, want);
    let mut bare = ft.clone();
    bare.as_object_mut().unwrap().remove("QUANTIZED_VOLUME_SCALE");
    assert_eq!(
        kind(&tile(b"pnts", &bare, &q, None, &[], &[]), "t.pnts"),
        ErrorKind::Corrupt
    );
}

#[test]
fn point_colors_in_each_encoding_come_out_linear() {
    let p = [0., 0., 0., 1., 0., 0.];
    let color = |more: Value, bytes: &[u8]| {
        let doc = load(&cloud(&p, &more, bytes, None, &[]), "t.pnts").unwrap();
        let (width, values) = attribute(&doc, "COLOR_0");
        assert_eq!(width, 4);
        values
    };
    let near = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-5);
    // sRGB 128 is 0.21586 linear.
    let half = 0.215_86;
    let rgb = color(json!({ "RGB": { "byteOffset": 24 } }), &[255, 0, 128, 0, 255, 0]);
    assert!(near(&rgb, &[1., 0., half, 1., 0., 1., 0., 1.]), "{rgb:?}");
    let rgba = color(
        json!({ "RGBA": { "byteOffset": 24 } }),
        &[255, 0, 128, 51, 0, 0, 0, 255],
    );
    assert!(near(&rgba, &[1., 0., half, 0.2, 0., 0., 0., 1.]), "{rgba:?}");
    // 5, 6 and 5 bits: all of red, half of green, none of blue; then all of blue.
    let packed: Vec<u8> = [0xF800u16 | (32 << 5), 0x001F]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let rgb565 = color(json!({ "RGB565": { "byteOffset": 24 } }), &packed);
    let green = crate::mesh::srgb_to_linear(32.0 / 63.0);
    assert!(near(&rgb565, &[1., green, 0., 1., 0., 0., 1., 1.]), "{rgb565:?}");
    // The first of several wins, as the specification orders them.
    let both = color(
        json!({ "RGBA": { "byteOffset": 24 }, "RGB": { "byteOffset": 24 }, "CONSTANT_RGBA": [9, 9, 9, 9] }),
        &[255, 0, 0, 255, 0, 0, 255, 255],
    );
    assert!(near(&both, &[1., 0., 0., 1., 0., 0., 1., 1.]), "{both:?}");
    // One color for every point is the mesh's, not an attribute.
    let doc = load(
        &cloud(&p, &json!({ "CONSTANT_RGBA": [255, 255, 0, 51] }), &[], None, &[]),
        "t.pnts",
    )
    .unwrap();
    let mesh = &doc.json()["meshes"][0];
    assert!(mesh["primitives"][0]["attributes"].get("COLOR_0").is_none());
    let constant: Vec<f32> = mesh["extras"]["constant"]["COLOR_0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert!(near(&constant, &[1., 1., 0., 0.2]), "{constant:?}");
}

#[test]
fn point_normals_turn_y_up_and_decode_from_two_bytes() {
    let p = [0., 0., 0., 1., 0., 0.];
    let doc = load(
        &cloud(
            &p,
            &json!({ "NORMAL": { "byteOffset": 24 } }),
            &floats(&[0., 0., 2., 0., 1., 0.]),
            None,
            &[],
        ),
        "t.pnts",
    )
    .unwrap();
    // Up in the tile is +Z, in the document +Y; the tile's +Y is the document's -Z. Unit length.
    assert_eq!(attribute(&doc, "NORMAL").1, [0., 1., 0., 0., 0., -1.]);
    let doc = load(
        &cloud(
            &p,
            &json!({ "NORMAL_OCT16P": { "byteOffset": 24 } }),
            &[255, 127, 0, 0],
            None,
            &[],
        ),
        "t.pnts",
    )
    .unwrap();
    let n = attribute(&doc, "NORMAL").1;
    // (255, 127) is just off +X; (0, 0) folds over to -Z, which is the document's -Y.
    assert!(
        (n[0] - 1.0).abs() < 1e-4 && n[1].abs() < 0.01 && n[2].abs() < 0.01,
        "{n:?}"
    );
    assert_eq!(&n[3..], [0., -1., 0.]);
}

#[test]
fn oct_encoded_vectors_decode_to_unit_length() {
    let near = |a: [f64; 3], b: [f64; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-12);
    assert!(near(oct_decode(127.5, 127.5, 255.0), [0., 0., 1.]));
    assert!(near(oct_decode(255.0, 127.5, 255.0), [1., 0., 0.]));
    assert!(near(oct_decode(127.5, 0.0, 255.0), [0., -1., 0.]));
    assert!(near(oct_decode(0.0, 0.0, 255.0), [0., 0., -1.]));
    assert!(near(oct_decode(65535.0, 65535.0, 65535.0), [0., 0., -1.]));
    // Out of range reads as the nearest end, not as something else.
    assert!(near(oct_decode(-5.0, 127.5, 255.0), [-1., 0., 0.]));
    for x in (0..=255).step_by(15) {
        for y in (0..=255).step_by(15) {
            let v = oct_decode(f64::from(x), f64::from(y), 255.0);
            let len = v.iter().map(|c| c * c).sum::<f64>().sqrt();
            assert!((len - 1.0).abs() < 1e-12, "({x}, {y}) decodes to {v:?}");
        }
    }
}

#[test]
fn batch_ids_and_per_point_properties_become_attributes() {
    let p = [0., 0., 0., 1., 0., 0., 2., 0., 0.];
    // Unsigned shorts unless the property says otherwise.
    let ids: Vec<u8> = [2u16, 0, 300].iter().flat_map(|v| v.to_le_bytes()).collect();
    let table = json!({ "name": ["a", "b", "c"] });
    let t = cloud(
        &p,
        &json!({ "BATCH_ID": { "byteOffset": 36 }, "BATCH_LENGTH": 3 }),
        &ids,
        Some(&table),
        &[],
    );
    let doc = load(&t, "t.pnts").unwrap();
    assert_eq!(attribute(&doc, "_BATCHID").1, [2., 0., 300.]);
    assert_eq!(
        doc.json()["nodes"][0]["extras"],
        json!({ "tile": "pnts", "batchLength": 3, "batchTable": table })
    );
    assert_eq!(doc.facts["features"], 3);
    for (component, bytes, want) in [
        ("UNSIGNED_BYTE", vec![7u8, 8, 9], [7., 8., 9.]),
        (
            "UNSIGNED_INT",
            [70000u32, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
            [70000., 1., 2.],
        ),
    ] {
        let ft = json!({ "BATCH_ID": { "byteOffset": 36, "componentType": component }, "BATCH_LENGTH": 9 });
        let doc = load(&cloud(&p, &ft, &bytes, None, &[]), "t.pnts").unwrap();
        assert_eq!(attribute(&doc, "_BATCHID").1, want, "{component}");
    }
    let ft = json!({ "BATCH_ID": { "byteOffset": 36, "componentType": "FLOAT64" } });
    assert_eq!(kind(&cloud(&p, &ft, &ids, None, &[]), "t.pnts"), ErrorKind::Corrupt);

    // Without batch ids the table has a value for each point.
    let mut bbin = floats(&[20.5, 21.5, 22.5]);
    bbin.extend([1u8, 2, 3, 4, 5, 6]);
    let table = json!({
        "temperature \u{2103}": { "byteOffset": 0, "componentType": "FLOAT", "type": "SCALAR" },
        "pair": { "byteOffset": 12, "componentType": "UNSIGNED_BYTE", "type": "VEC2" },
        "rank": [3, 1, 2],
        "same": [4, 4, 4],
        "label": ["x", "y", "z"],
        "short": [1, 2],
    });
    let doc = load(&cloud(&p, &Value::Null, &[], Some(&table), &bbin), "t.pnts").unwrap();
    assert_eq!(attribute(&doc, "_TEMPERATURE__").1, [20.5, 21.5, 22.5]);
    assert_eq!(attribute(&doc, "_PAIR_0").1, [1., 3., 5.]);
    assert_eq!(attribute(&doc, "_PAIR_1").1, [2., 4., 6.]);
    assert_eq!(attribute(&doc, "_RANK").1, [3., 1., 2.]);
    assert_eq!(doc.json()["meshes"][0]["extras"]["constant"]["_SAME"], 4.0);
    // What isn't a number for each point stays in the extras.
    assert_eq!(
        doc.json()["nodes"][0]["extras"]["batchTable"],
        json!({ "label": ["x", "y", "z"], "short": [1, 2] })
    );
}

#[test]
fn draco_points_need_the_feature_and_sound_data() {
    let ft = json!({
        "POINTS_LENGTH": 1, "POSITION": { "byteOffset": 0 },
        "extensions": { "3DTILES_draco_point_compression": {
            "properties": { "POSITION": 0 }, "byteOffset": 0, "byteLength": 12 } },
    });
    let t = tile(b"pnts", &ft, b"DRACO, not!!", None, &[], &[]);
    let want = if cfg!(feature = "draco") {
        ErrorKind::Corrupt
    } else {
        ErrorKind::FeatureDisabled
    };
    assert_eq!(kind(&t, "t.pnts"), want);
    #[cfg(feature = "draco")]
    {
        let mut past = ft.clone();
        past["extensions"]["3DTILES_draco_point_compression"]["byteLength"] = json!(4096);
        assert_eq!(
            kind(&tile(b"pnts", &past, b"DRACO, not!!", None, &[], &[]), "t.pnts"),
            ErrorKind::Truncated
        );
    }
}

// --- instances ---------------------------------------------------------------------------------

/// What a document's instanced node holds: each instance's translation, rotation and scale.
fn instances(doc: &Document) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let a = &doc.json()["nodes"][0]["extensions"]["EXT_mesh_gpu_instancing"]["attributes"];
    let column = |name: &str| {
        a.get(name).map(|i| {
            doc.read_floats(usize::try_from(i.as_u64().unwrap()).unwrap())
                .unwrap()
                .1
        })
    };
    (
        column("TRANSLATION").expect("translations"),
        column("ROTATION"),
        column("SCALE"),
    )
}

fn instanced(more: &Value, fbin: &[u8]) -> Vec<u8> {
    let mut ft = json!({ "INSTANCES_LENGTH": 2, "POSITION": { "byteOffset": 0 } });
    for (k, v) in more.as_object().into_iter().flatten() {
        ft[k] = v.clone();
    }
    let mut body = floats(&[1., 2., 3., -4., 0., 0.]);
    body.extend(fbin);
    tile(b"i3dm", &ft, &body, None, &[], &glb())
}

#[test]
fn instances_are_placed_turned_and_scaled_y_up() {
    // Positions alone: Z up to Y up, and nothing else.
    let doc = load(&instanced(&Value::Null, &[]), "t.i3dm").unwrap();
    let (t, r, s) = instances(&doc);
    assert_eq!(t, [1., 3., -2., -4., 0., 0.]);
    assert!(r.is_none() && s.is_none());
    assert_eq!(doc.json()["extensionsUsed"], json!(["EXT_mesh_gpu_instancing"]));
    assert_eq!(doc.facts["instances"], 2);
    assert_eq!(doc.facts["triangles"], 2);
    // The triangle (0,0,0), (1,0,0), (0,2,0), at each instance.
    assert_eq!(bounds(&doc), ([-4., 0., -2.], [2., 5., 0.]));

    // Both scales multiply; the non-uniform one is along the tile's axes, where the glTF's Y is Z.
    let more = json!({ "SCALE": { "byteOffset": 24 }, "SCALE_NON_UNIFORM": { "byteOffset": 32 } });
    let doc = load(
        &instanced(&more, &floats(&[2., 1., 1., 10., 100., 1., 1., 1.])),
        "t.i3dm",
    )
    .unwrap();
    let (_, r, s) = instances(&doc);
    assert!(r.is_none());
    assert_eq!(s.unwrap(), [2., 200., 20., 1., 1., 1.]);

    // Up along the tile's X and right along its Y: the model's Y goes to X and its X to Y.
    let more = json!({ "NORMAL_UP": { "byteOffset": 24 }, "NORMAL_RIGHT": { "byteOffset": 48 } });
    let frames = floats(&[1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.]);
    let doc = load(&instanced(&more, &frames), "t.i3dm").unwrap();
    // Instance 0: the glTF's (1,0,0) is the tile's (1,0,0), which goes to right, the tile's +Y, the
    // document's -Z; the glTF's (0,2,0) is the tile's (0,0,2), which goes to forward, right cross up, -Z
    // in the tile and -Y in the document.
    let b = bounds(&doc);
    assert_eq!((b.0[1], b.1[1]), (0., 3.), "{b:?}");
    let (_, r, _) = instances(&doc);
    let r = r.unwrap();
    // The second instance's up is +Z and its right +X: the model's Y, already the tile's Z, is taken
    // on to -Y, a quarter turn about X.
    let h = 0.5f32.sqrt();
    assert!(
        r[4..].iter().zip([h, 0., 0., h]).all(|(a, b)| (a - b).abs() < 1e-6),
        "{r:?}"
    );

    // One of the pair alone is no orientation.
    let doc = load(
        &instanced(&json!({ "NORMAL_UP": { "byteOffset": 24 } }), &frames),
        "t.i3dm",
    )
    .unwrap();
    assert!(instances(&doc).1.is_none());
}

#[test]
fn east_north_up_stands_on_the_ellipsoid() {
    let near = |a: [f64; 3], b: [f64; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-9);
    let [east, north, up] = i3dm::east_north_up_at([6_378_137.0, 0.0, 0.0]);
    assert!(near(east, [0., 1., 0.]) && near(north, [0., 0., 1.]) && near(up, [1., 0., 0.]));
    // A point on the surface at 60 degrees north, 90 east: up is the geodetic normal, not the
    // direction from the center.
    let (lat, a, b) = (60f64.to_radians(), 6_378_137.0f64, 6_356_752.314_245_179_f64);
    let e2 = 1.0 - b * b / (a * a);
    let n = a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    let [east, north, up] = i3dm::east_north_up_at([0.0, n * lat.cos(), n * (1.0 - e2) * lat.sin()]);
    assert!(near(up, [0., lat.cos(), lat.sin()]), "{up:?}");
    assert!(near(east, [-1., 0., 0.]) && near(north, [0., -lat.sin(), lat.cos()]));
    // The poles and the center have no east: the frames CesiumJS gives them.
    assert_eq!(
        i3dm::east_north_up_at([0., 0., 9.]),
        [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]
    );
    assert_eq!(
        i3dm::east_north_up_at([0., 0., -9.]),
        [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]
    );
    assert_eq!(
        i3dm::east_north_up_at([0.; 3]),
        [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]
    );
    assert_eq!(
        i3dm::east_north_up_at([f64::NAN; 3]),
        [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]
    );

    // In a tile: the frame is taken at the position with the center added.
    let more = json!({ "EAST_NORTH_UP": true, "RTC_CENTER": [6_378_137.0, 0.0, 0.0], "INSTANCES_LENGTH": 1 });
    let t = tile(
        b"i3dm",
        &{
            let mut ft = json!({ "POSITION": { "byteOffset": 0 } });
            for (k, v) in more.as_object().unwrap() {
                ft[k] = v.clone();
            }
            ft
        },
        &floats(&[0., 0., 0.]),
        None,
        &[],
        &glb(),
    );
    let doc = load(&t, "t.i3dm").unwrap();
    assert_eq!(doc.json()["nodes"][1]["translation"], json!([6_378_137.0, 0.0, -0.0]));
    // The glTF's (1,0,0) is the tile's X, which goes east, the tile's +Y, the document's -Z; its (0,2,0) is
    // the tile's Z, which goes up, the tile's +X.
    let (min, max) = bounds(&doc);
    assert!(
        (max[0] - min[0] - 2.0).abs() < 1.0 && (max[2] - min[2] - 1.0).abs() < 1.0,
        "{min:?} {max:?}"
    );
}

#[test]
fn right_and_up_make_a_frame_at_right_angles() {
    assert_eq!(
        i3dm::frame([1., 0., 0.], [0., 1., 0.]),
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    );
    // Longer than one, and leaning: right keeps its direction, up stays in the plane of the two.
    let f = i3dm::frame([2., 0., 0.], [1., 1., 0.]);
    assert_eq!(f, [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
    // No frame in them: the model's own axes.
    for (right, up) in [
        ([0.; 3], [0., 1., 0.]),
        ([1., 0., 0.], [3., 0., 0.]),
        ([f64::NAN; 3], [0., 1., 0.]),
    ] {
        assert_eq!(i3dm::frame(right, up), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
    }
}

#[test]
fn an_instanced_gltf_is_found_by_its_uri() {
    let ft = json!({ "INSTANCES_LENGTH": 1, "POSITION": { "byteOffset": 0 } });
    let named = |uri: &[u8]| tile_with(b"i3dm", &ft, &floats(&[0., 0., 7.]), None, &[], uri, 0);
    let assets = Memory::new().with("tiles/models/a b.glb", glb());
    let doc = load_with(&named(b"models/a%20b.glb\0\0\0"), "tiles/t.i3dm", &assets, Keep::ALL).unwrap();
    assert_eq!(bounds(&doc), ([0., 7., 0.], [1., 9., 0.]));
    let e = load_with(&named(b"models/gone.glb"), "tiles/t.i3dm", &assets, Keep::ALL).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::MissingAsset);
    for uri in [
        &b"https://example.com/a.glb"[..],
        b"/etc/a.glb",
        b"C:/a.glb",
        b"",
        b"\0",
    ] {
        let e = load_with(&named(uri), "tiles/t.i3dm", &assets, Keep::ALL).unwrap_err();
        assert!(
            matches!(e.kind(), ErrorKind::Unsupported | ErrorKind::Corrupt),
            "{uri:?}: {e}"
        );
    }
    let odd = tile_with(b"i3dm", &ft, &floats(&[0., 0., 7.]), None, &[], &glb(), 2);
    assert_eq!(kind(&odd, "t.i3dm"), ErrorKind::Corrupt);
    // More instances of more meshes than the load allows.
    let e = load_with(
        &instanced(&Value::Null, &[]),
        "t.i3dm",
        &Memory::new(),
        Keep::ALL.with_max_triangles(1),
    )
    .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LimitExceeded);
}

// --- glTF content ------------------------------------------------------------------------------

#[test]
fn centers_move_a_node_not_the_vertices() {
    let center = [1_215_012.5, -4_736_313.25, 4_081_604.75];
    let t = tile(
        b"b3dm",
        &json!({ "BATCH_LENGTH": 0, "RTC_CENTER": center }),
        &[],
        None,
        &[],
        &glb(),
    );
    let doc = load(&t, "t.b3dm").unwrap();
    let nodes = doc.json()["nodes"].as_array().unwrap();
    assert_eq!(
        nodes[1],
        json!({ "name": "b3dm", "children": [0], "translation": [center[0], center[2], -center[1]] })
    );
    assert_eq!(doc.json()["scenes"][0]["nodes"], json!([1]));
    // The glTF's own center, from the extension, is in tile space too, and the extension is spent.
    let own = glb_with(&json!({
        "extensions": { "CESIUM_RTC": { "center": [10.0, 20.0, 30.0] } },
        "extensionsUsed": ["CESIUM_RTC"],
        "extensionsRequired": ["CESIUM_RTC"],
    }));
    let t = tile(
        b"b3dm",
        &json!({ "BATCH_LENGTH": 0, "RTC_CENTER": [1.0, 2.0, 3.0] }),
        &[],
        None,
        &[],
        &own,
    );
    let doc = load(&t, "t.b3dm").unwrap();
    assert_eq!(doc.json()["nodes"][1]["translation"], json!([11.0, 33.0, -22.0]));
    for key in ["extensions", "extensionsUsed", "extensionsRequired"] {
        assert!(doc.json().get(key).is_none(), "{key} is left");
    }
    let bad = glb_with(&json!({ "extensions": { "CESIUM_RTC": { "center": [1, 2] } } }));
    let t = tile(b"b3dm", &json!({ "BATCH_LENGTH": 0 }), &[], None, &[], &bad);
    assert_eq!(kind(&t, "t.b3dm"), ErrorKind::Corrupt);
    // Nothing to hold: the glTF as it is.
    let t = tile(b"b3dm", &Value::Null, &[], None, &[], &glb());
    assert_eq!(load(&t, "t.b3dm").unwrap().json()["nodes"], json!([{ "mesh": 0 }]));
}

#[test]
fn an_old_batch_id_attribute_takes_its_name() {
    let old = glb_with(&json!({ "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0, "BATCHID": 0 } }] }] }));
    let t = tile(b"b3dm", &json!({ "BATCH_LENGTH": 1 }), &[], None, &[], &old);
    let doc = load(&t, "t.b3dm").unwrap();
    let attributes = &doc.json()["meshes"][0]["primitives"][0]["attributes"];
    assert_eq!(*attributes, json!({ "POSITION": 0, "_BATCHID": 0 }));
}

// --- composites and tilesets -------------------------------------------------------------------

#[test]
fn composites_join_their_tiles_and_stop_nesting() {
    let b = tile(
        b"b3dm",
        &json!({ "BATCH_LENGTH": 0, "RTC_CENTER": [0.0, 0.0, 10.0] }),
        &[],
        None,
        &[],
        &glb(),
    );
    let p = cloud(&[5., 0., 0.], &Value::Null, &[], None, &[]);
    let doc = load(&cmpt(&[b.clone(), cmpt(&[p])]), "t.cmpt").unwrap();
    assert_eq!(doc.facts["tiles"], 2);
    assert_eq!(
        (doc.facts["triangles"].clone(), doc.facts["points"].clone()),
        (json!(1), json!(1))
    );
    assert_eq!(bounds(&doc), ([0., 0., 0.], [5., 12., 0.]));
    assert_eq!(
        doc.json()["asset"]["extras"]["source"]["content"],
        json!(["b3dm", "pnts", "cmpt"])
    );
    // A count past the tiles there are, and a tile shorter than its own header.
    let mut more = cmpt(std::slice::from_ref(&b));
    more[12] = 2;
    assert_eq!(kind(&more, "t.cmpt"), ErrorKind::Truncated);
    let mut short = cmpt(std::slice::from_ref(&b));
    short[24..28].copy_from_slice(&4u32.to_le_bytes());
    assert_eq!(kind(&short, "t.cmpt"), ErrorKind::Truncated);
    let mut deep = b.clone();
    for _ in 0..=super::MAX_DEPTH {
        deep = cmpt(&[deep]);
    }
    assert_eq!(kind(&deep, "t.cmpt"), ErrorKind::LimitExceeded);
    // Too many triangles in all.
    let e = load_with(
        &cmpt(&[b.clone(), b]),
        "t.cmpt",
        &Memory::new(),
        Keep::ALL.with_max_triangles(1),
    )
    .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LimitExceeded);
}

fn moved(x: f64, y: f64, z: f64) -> Value {
    json!([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1])
}

fn tileset_of(root: &Value) -> Vec<u8> {
    json!({ "asset": { "version": "1.0" }, "geometricError": 100, "root": root })
        .to_string()
        .into_bytes()
}

#[test]
fn a_tileset_is_read_at_its_finest() {
    let b = tile(b"b3dm", &json!({ "BATCH_LENGTH": 0 }), &[], None, &[], &glb());
    let assets = Memory::new()
        .with("set/a.b3dm", b.clone())
        .with("set/deep/b.b3dm", b.clone())
        .with("set/model.glb", glb());
    let leaf = |uri: &str, at: Value| json!({ "geometricError": 0, "transform": at, "content": { "uri": uri } });
    // The root gives way to its children; the child that adds is drawn with its own.
    let root = json!({
        "geometricError": 50, "refine": "REPLACE", "content": { "uri": "a.b3dm" }, "transform": moved(100., 0., 0.),
        "children": [
            leaf("deep/b.b3dm", moved(0., 0., 10.)),
            { "geometricError": 10, "refine": "ADD", "content": { "uri": "model.glb" }, "transform": moved(0., 20., 0.),
              "children": [leaf("a.b3dm?v=2#top", moved(0., 0., 5.))] },
        ],
    });
    let doc = load_with(&tileset_of(&root), "set/tileset.json", &assets, Keep::ALL).unwrap();
    assert_eq!(doc.facts["tiles"], 3);
    assert_eq!(doc.facts["triangles"], 3);
    assert!(doc.facts.get("unrefined_tiles").is_none() && doc.facts.get("missing_tiles").is_none());
    // In the tileset's own order, each under a node named for its URI with the transforms multiplied.
    let nodes = doc.json()["nodes"].as_array().unwrap();
    let placed: Vec<(&str, &Value)> = nodes
        .iter()
        .filter(|n| n.get("translation").is_some())
        .map(|n| (n["name"].as_str().unwrap(), &n["translation"]))
        .collect();
    assert_eq!(
        placed,
        [
            ("deep/b.b3dm", &json!([100.0, 10.0, -0.0])),
            ("model.glb", &json!([100.0, 0.0, -20.0])),
            ("a.b3dm?v=2#top", &json!([100.0, 5.0, -20.0])),
        ]
    );
    assert_eq!(bounds(&doc), ([100., 0., -20.], [101., 12., 0.]));
    let source = &doc.json()["asset"]["extras"]["source"];
    assert_eq!(
        (&source["format"], &source["version"]),
        (&json!("3D Tiles"), &json!("1.0"))
    );

    // Read by its content, whatever it's called.
    assert_eq!(detect::sniff(&tileset_of(&root), None), Some("cmpt"));
    assert_eq!(Format::detect(&tileset_of(&root)), Some(Format::Tiles));
    let doc = load_bytes(&tileset_of(&root), "set/anything.json", "json", &assets, Keep::ALL).unwrap();
    assert_eq!(doc.facts["tiles"], 3);
}

#[test]
fn a_tilesets_gltf_may_be_another_way_up() {
    let b = tile(b"b3dm", &json!({ "BATCH_LENGTH": 0 }), &[], None, &[], &glb());
    let assets = Memory::new().with("a.b3dm", b);
    let set = |axis: &str| {
        json!({
            "asset": { "version": "0.0", "gltfUpAxis": axis }, "geometricError": 1,
            "root": { "geometricError": 0, "refine": "ADD", "content": { "url": "a.b3dm" } },
        })
        .to_string()
        .into_bytes()
    };
    // To within the turn's rounding.
    let b = |axis: &str| {
        let (min, max) = bounds(&load_with(&set(axis), "tileset.json", &assets, Keep::ALL).unwrap());
        let whole = |v: [f32; 3]| v.map(|c| (c * 1024.0).round() / 1024.0 + 0.0);
        (whole(min), whole(max))
    };
    // The triangle spans x 0..1 and y 0..2 in the glTF. Y up, it is as it is; Z up, its y is the
    // tile's y, the document's -z; X up, its x is the tile's z and its y the tile's y.
    assert_eq!(b("Y"), ([0., 0., 0.], [1., 2., 0.]));
    assert_eq!(b("z"), ([0., 0., -2.], [1., 0., 0.]));
    assert_eq!(b("X"), ([0., 0., -2.], [0., 1., 0.]));
}

#[test]
fn a_tileset_past_its_cap_is_drawn_unrefined() {
    let b = tile(b"b3dm", &json!({ "BATCH_LENGTH": 0 }), &[], None, &[], &glb());
    let assets = Memory::new().with("a.b3dm", b);
    let leaf = json!({ "geometricError": 0, "content": { "uri": "a.b3dm" } });
    let wide = |n: usize| {
        json!({
            "geometricError": 9, "refine": "REPLACE", "content": { "uri": "a.b3dm" },
            "children": vec![leaf.clone(); n],
        })
    };
    let read = |root: &Value| load_with(&tileset_of(root), "tileset.json", &assets, Keep::ALL).unwrap();
    let doc = read(&wide(8));
    assert_eq!(doc.facts["tiles"], 8);
    // More children than tiles are drawn: the root stays, and the facts say so.
    let doc = read(&wide(tileset::MAX_TILES + 1));
    assert_eq!(doc.facts["tiles"], 1);
    assert_eq!(doc.facts["unrefined_tiles"], true);
    // Two branches, one within the cap and one past it: the coarser is refined first.
    let root = json!({
        "geometricError": 9, "refine": "REPLACE",
        "children": [
            { "geometricError": 5, "content": { "uri": "a.b3dm" }, "children": vec![leaf.clone(); 4] },
            { "geometricError": 1, "content": { "uri": "a.b3dm" }, "children": vec![leaf.clone(); tileset::MAX_TILES] },
        ],
    });
    let doc = read(&root);
    assert_eq!(doc.facts["tiles"], 5);
    assert_eq!(doc.facts["unrefined_tiles"], true);
}

#[test]
fn a_tileset_that_cannot_be_read_says_why() {
    let b = tile(b"b3dm", &json!({ "BATCH_LENGTH": 0 }), &[], None, &[], &glb());
    let assets = Memory::new().with("a.b3dm", b);
    let read = |root: &Value| load_with(&tileset_of(root), "tileset.json", &assets, Keep::ALL);
    let content = |uri: &str| json!({ "geometricError": 0, "content": { "uri": uri } });
    // Some content missing: left out and listed. All of it: an error.
    let root = json!({ "geometricError": 1, "refine": "ADD", "content": { "uri": "a.b3dm" },
                       "children": [content("gone.b3dm"), content("https://example.com/a.b3dm")] });
    let doc = read(&root).unwrap();
    assert_eq!(doc.facts["missing_tiles"], 2);
    assert_eq!(
        doc.json()["asset"]["extras"]["source"]["missingTiles"],
        json!(["gone.b3dm", "https://example.com/a.b3dm"])
    );
    assert_eq!(read(&content("gone.b3dm")).unwrap_err().kind(), ErrorKind::MissingAsset);
    // A tileset that is its own content never ends: the nesting is capped.
    let own = Memory::new().with(
        "tileset.json",
        tileset_of(&json!({
        "geometricError": 1, "refine": "ADD", "content": { "uri": "tileset.json" } })),
    );
    let e = load_with(
        &own.get("tileset.json").unwrap().clone(),
        "tileset.json",
        &own,
        Keep::ALL,
    )
    .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LimitExceeded);
    // Implicit tiling, on a tile or as the 1.0 extension.
    let implicit = json!({ "geometricError": 1, "content": { "uri": "{level}.glb" },
                           "implicitTiling": { "subdivisionScheme": "QUADTREE" } });
    assert_eq!(read(&implicit).unwrap_err().kind(), ErrorKind::Unsupported);
    let extension = json!({ "geometricError": 1, "extensions": { "3DTILES_implicit_tiling": {} } });
    assert_eq!(read(&extension).unwrap_err().kind(), ErrorKind::Unsupported);
    for transform in [json!([1, 2, 3]), json!("identity"), json!(vec![0; 17])] {
        let mut root = content("a.b3dm");
        root["transform"] = transform.clone();
        assert_eq!(read(&root).unwrap_err().kind(), ErrorKind::Corrupt, "{transform}");
    }
    assert_eq!(
        kind(br#"{"asset":{"version":"1.0"},"geometricError":1}"#, "tileset.json"),
        ErrorKind::Corrupt
    );
    assert_eq!(
        kind(br#"{"geometricError":1,"root":7}"#, "tileset.json"),
        ErrorKind::Corrupt
    );
    assert_eq!(
        kind(br#"{"geometricError":1,"root":{"#, "tileset.json"),
        ErrorKind::Corrupt
    );
    // A tileset with nothing to draw draws nothing.
    let empty = read(&json!({ "geometricError": 1 })).unwrap();
    assert_eq!(empty.facts["tiles"], 0);
    assert!(empty.flatten(Pose::Rest).unwrap().is_none());
}

// --- detection ---------------------------------------------------------------------------------

#[test]
fn tiles_are_told_by_their_first_bytes() {
    for magic in ["b3dm", "i3dm", "pnts", "cmpt"] {
        let head = [magic.as_bytes(), &[1, 0, 0, 0, 28, 0, 0, 0]].concat();
        assert_eq!(detect::sniff(&head, None), Some(magic));
        assert_eq!(Format::detect(&head), Some(Format::Tiles));
        assert_eq!(Format::from_extension(magic), Some(Format::Tiles));
    }
    // A glTF's JSON has an asset too, and stays a glTF.
    assert_eq!(
        detect::sniff(br#"{"asset":{"version":"2.0"},"scenes":[]}"#, None),
        Some("gltf")
    );
    assert_eq!(
        detect::sniff(
            br#"{ "asset": {"version": "1.1"}, "geometricError": 4, "root": {} }"#,
            None
        ),
        Some("cmpt")
    );
    // A tile under another tile's extension, or none, reads as what it is.
    let p = cloud(&[1., 2., 3.], &Value::Null, &[], None, &[]);
    for name in ["t.b3dm", "t.cmpt", "t.bin", "t"] {
        let ext = name.rsplit_once('.').map_or("", |(_, e)| e);
        let doc = load_bytes(&p, name, ext, &Memory::new(), Keep::ALL).unwrap();
        assert_eq!(doc.facts["points"], 1, "{name}");
    }
}
