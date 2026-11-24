use std::fmt::Write as _;

use super::*;
use crate::assets::Memory;
use crate::document::Pose;
use crate::keep::Keep;
use crate::mesh::Mesh;

fn read_bytes(bytes: &[u8], max: usize) -> Result<Document<'static>> {
    let keep = Keep {
        max_triangles: max,
        ..Keep::ALL
    };
    super::super::load_bytes(bytes, "a.dxf", "dxf", &Memory::new(), keep).map(Document::into_owned)
}

fn read(text: &str) -> Document<'static> {
    read_bytes(text.as_bytes(), usize::MAX).unwrap()
}

fn mesh(doc: &Document<'_>) -> Mesh {
    doc.flatten(Pose::Rest).unwrap().unwrap()
}

fn triangles(doc: &Document<'_>) -> u64 {
    doc.facts["triangles"].as_u64().unwrap()
}

fn source<'a>(doc: &'a Document<'_>) -> &'a Value {
    &doc.json["asset"]["extras"]["source"]
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

fn near(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5)
}

/// Groups written as space-separated codes and values.
fn groups(s: &str) -> Vec<(i32, String)> {
    let words: Vec<&str> = s.split_whitespace().collect();
    words
        .chunks(2)
        .map(|w| (w[0].parse().unwrap(), w.get(1).copied().unwrap_or("").to_owned()))
        .collect()
}

/// A drawing's groups: a header giving its units, then tables, blocks and entities.
fn drawing(units: i64, tables: &str, blocks: &str, entities: &str) -> Vec<(i32, String)> {
    let mut s = format!("0 SECTION 2 HEADER 9 $ACADVER 1 AC1027 9 $INSUNITS 70 {units} 0 ENDSEC ");
    if !tables.is_empty() {
        write!(s, "0 SECTION 2 TABLES 0 TABLE 2 LAYER {tables} 0 ENDTAB 0 ENDSEC ").unwrap();
    }
    if !blocks.is_empty() {
        write!(s, "0 SECTION 2 BLOCKS {blocks} 0 ENDSEC ").unwrap();
    }
    write!(s, "0 SECTION 2 ENTITIES {entities} 0 ENDSEC 0 EOF").unwrap();
    groups(&s)
}

fn text(gs: &[(i32, String)], eol: &str) -> String {
    gs.iter().fold(String::new(), |mut s, (c, v)| {
        write!(s, "{c:>3}{eol}{v}{eol}").unwrap();
        s
    })
}

/// A binary file of the groups.
fn binary(gs: &[(i32, String)]) -> Vec<u8> {
    let mut out = SENTINEL.to_vec();
    for (c, v) in gs {
        out.extend(i16::try_from(*c).unwrap().to_le_bytes());
        match kind(*c) {
            Kind::Real => out.extend(v.parse::<f64>().unwrap().to_le_bytes()),
            Kind::I16 => out.extend(v.parse::<i16>().unwrap().to_le_bytes()),
            Kind::I32 => out.extend(v.parse::<i32>().unwrap().to_le_bytes()),
            Kind::I64 => out.extend(v.parse::<i64>().unwrap().to_le_bytes()),
            Kind::Byte => out.push(v.parse().unwrap()),
            Kind::Chunk => {
                out.push(u8::try_from(v.len()).unwrap());
                out.extend(v.as_bytes());
            }
            Kind::Text => {
                out.extend(v.as_bytes());
                out.push(0);
            }
        }
    }
    out
}

const CUBE: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];
const QUADS: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 2, 6, 5],
    [0, 4, 7, 3],
];

/// A `3DFACE` with extra groups.
fn face(extra: &str, pts: &[[f64; 3]]) -> String {
    let mut s = format!("0 3DFACE {extra} ");
    for (k, p) in pts.iter().enumerate() {
        write!(s, "1{k} {} 2{k} {} 3{k} {} ", p[0], p[1], p[2]).unwrap();
    }
    s
}

fn cube_faces(extra: &str) -> String {
    QUADS.iter().map(|q| face(extra, &q.map(|k| CUBE[k]))).collect()
}

#[test]
fn faces_turn_y_up_in_meters() {
    let doc = read(&text(&drawing(4, "", "", &cube_faces("8 Walls")), "\n"));
    assert_eq!(triangles(&doc), 12);
    let src = source(&doc);
    assert_eq!(src["format"], "DXF");
    assert_eq!(src["encoding"], "ASCII");
    assert_eq!(src["version"], "AC1027");
    assert_eq!(src["units"], "millimeters");
    assert!(src.get("truncated").is_none());
    assert_eq!(doc.json["nodes"][0]["name"], "RootNode");
    let m = mesh(&doc);
    assert_eq!(m.triangles(), 12);
    let (lo, hi) = bounds(&m);
    assert!(near(lo, [0.0, 0.0, -0.001]), "{lo:?}");
    assert!(near(hi, [0.001, 0.001, 0.0]), "{hi:?}");
    assert_eq!(m.materials.len(), 1);
    assert_eq!(m.materials[0].name, "Walls");
    assert!(m.materials[0].double_sided);
    // Shared corners are one vertex.
    assert_eq!(m.prims[0].positions.len(), 8);
}

#[test]
fn a_three_corner_face_and_degenerate_ones() {
    let tri = face("8 0", &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let line = face(
        "8 0",
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    );
    let doc = read(&text(&drawing(0, "", "", &format!("{tri}{line}")), "\n"));
    assert_eq!(triangles(&doc), 1);
    // A unitless drawing isn't scaled.
    assert!(source(&doc).get("units").is_none());
    assert!(doc.json["nodes"][0].get("scale").is_none());
}

fn polyface_cube(extra: &str) -> String {
    let mut s = format!("0 POLYLINE 8 Box 66 1 70 64 71 8 72 6 {extra} ");
    for p in CUBE {
        write!(s, "0 VERTEX 8 Box 10 {} 20 {} 30 {} 70 192 ", p[0], p[1], p[2]).unwrap();
    }
    for (i, q) in QUADS.iter().enumerate() {
        // Negative numbers hide an edge; they're corners all the same.
        let sign = if i % 2 == 0 { "-" } else { "" };
        write!(
            s,
            "0 VERTEX 8 Box 10 0 20 0 30 0 70 128 71 {sign}{} 72 {} 73 {} 74 {} ",
            q[0] + 1,
            q[1] + 1,
            q[2] + 1,
            q[3] + 1
        )
        .unwrap();
    }
    s + "0 SEQEND "
}

#[test]
fn polyface_mesh() {
    let doc = read(&text(&drawing(6, "", "", &polyface_cube("")), "\n"));
    assert_eq!(triangles(&doc), 12);
    let m = mesh(&doc);
    assert_eq!(m.prims[0].positions.len(), 8);
    let (lo, hi) = bounds(&m);
    assert!(near(lo, [0.0, 0.0, -1.0]) && near(hi, [1.0, 1.0, 0.0]), "{lo:?} {hi:?}");
    // A face with its own color goes in its own primitive.
    let colored = polyface_cube("").replacen("70 128 71", "62 1 70 128 71", 1);
    let doc = read(&text(&drawing(6, "", "", &colored), "\n"));
    assert_eq!(triangles(&doc), 12);
    assert_eq!(mesh(&doc).prims.len(), 2);
}

/// A polygon mesh of `m` rows (heights) of `n` points around a square.
fn grid(m: u32, n: u32, flags: i64) -> String {
    let mut s = format!("0 POLYLINE 8 Grid 66 1 70 {} 71 {m} 72 {n} ", 16 | flags);
    for i in 0..m {
        for j in 0..n {
            let a = std::f64::consts::TAU * f64::from(j) / f64::from(n);
            write!(s, "0 VERTEX 8 Grid 10 {} 20 {} 30 {i} 70 64 ", a.cos(), a.sin()).unwrap();
        }
    }
    s + "0 SEQEND "
}

#[test]
fn polygon_meshes_open_and_closed() {
    let open = read(&text(&drawing(0, "", "", &grid(3, 4, 0)), "\n"));
    assert_eq!(triangles(&open), 2 * 3 * 2);
    let closed_n = read(&text(&drawing(0, "", "", &grid(3, 4, 32)), "\n"));
    assert_eq!(triangles(&closed_n), 2 * 4 * 2);
    let closed_both = read(&text(&drawing(0, "", "", &grid(3, 4, 33)), "\n"));
    assert_eq!(triangles(&closed_both), 3 * 4 * 2);
    // A fitted surface replaces its control points: 2 by 3 of them here.
    let mut fitted = String::from("0 POLYLINE 8 0 66 1 70 20 71 2 72 2 73 2 74 3 75 6 ");
    for (x, y) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        write!(fitted, "0 VERTEX 10 {x} 20 {y} 30 5 70 80 ").unwrap();
    }
    for x in 0..2 {
        for y in 0..3 {
            write!(fitted, "0 VERTEX 10 {x} 20 {y} 30 0 70 72 ").unwrap();
        }
    }
    fitted += "0 SEQEND ";
    let doc = read(&text(&drawing(0, "", "", &fitted), "\n"));
    assert_eq!(triangles(&doc), 4);
    let (lo, hi) = bounds(&mesh(&doc));
    assert!(near(lo, [0.0, 0.0, -2.0]) && near(hi, [1.0, 0.0, 0.0]), "{lo:?} {hi:?}");
}

#[test]
fn a_mesh_entity() {
    let mut s = String::from("0 MESH 8 Solid 100 AcDbSubDMesh 71 2 72 0 91 1 92 8 ");
    for p in CUBE {
        write!(s, "10 {} 20 {} 30 {} ", p[0], p[1], p[2]).unwrap();
    }
    write!(s, "93 30 ").unwrap();
    for q in QUADS {
        write!(s, "90 4 90 {} 90 {} 90 {} 90 {} ", q[0], q[1], q[2], q[3]).unwrap();
    }
    s += "94 12 90 0 90 1 95 0 140 1 ";
    let doc = read(&text(&drawing(0, "", "", &s), "\n"));
    assert_eq!(triangles(&doc), 12);
    assert_eq!(mesh(&doc).prims[0].positions.len(), 8);
    let notes = source(&doc)["notes"].to_string();
    assert!(notes.contains("control mesh"), "{notes}");
}

const TRI_BLOCK: &str = "0 BLOCK 8 0 2 Tri 70 0 10 1 20 0 30 0 3 Tri \
                         0 3DFACE 8 0 10 1 20 0 30 0 11 2 21 0 31 0 12 1 22 1 32 0 0 ENDBLK 8 0 ";

#[test]
fn an_insert_by_base_scale_and_rotation() {
    let insert = "0 INSERT 8 0 2 tri 10 10 20 0 30 0 41 2 42 2 43 2 50 90 ";
    let doc = read(&text(&drawing(0, "", TRI_BLOCK, insert), "\n"));
    assert_eq!(triangles(&doc), 1);
    let node = &doc.json["nodes"][0];
    assert_eq!(node["name"], "Tri");
    assert_eq!(node["mesh"], 0);
    assert_eq!(doc.json["meshes"][0]["name"], "Tri");
    // The block's (2,0,0) and (1,1,0) are 1 and (0,1) from its base, doubled and turned a
    // quarter: (10,2,0) and (8,0,0); its base point lands on (10,0,0). Y-up, that's z = -y.
    let (lo, hi) = bounds(&mesh(&doc));
    assert!(
        near(lo, [8.0, 0.0, -2.0]) && near(hi, [10.0, 0.0, 0.0]),
        "{lo:?} {hi:?}"
    );
}

#[test]
fn an_insert_array_shares_one_mesh() {
    let insert = "0 INSERT 8 0 2 Tri 10 0 20 0 30 0 70 2 71 3 44 5 45 7 ";
    let doc = read(&text(&drawing(0, "", TRI_BLOCK, insert), "\n"));
    assert_eq!(triangles(&doc), 6);
    assert_eq!(doc.json["meshes"].as_array().unwrap().len(), 1);
    assert_eq!(doc.json["nodes"].as_array().unwrap().len(), 7);
    let m = mesh(&doc);
    assert_eq!(m.triangles(), 6);
    let (lo, hi) = bounds(&m);
    assert!(
        near(lo, [0.0, 0.0, -15.0]) && near(hi, [6.0, 0.0, 0.0]),
        "{lo:?} {hi:?}"
    );
}

#[test]
fn an_insert_extruded_down_mirrors_x() {
    let block = "0 BLOCK 2 Flat 10 0 20 0 30 0 0 3DFACE 10 0 20 0 30 0 11 1 21 0 31 0 12 0 22 1 32 0 0 ENDBLK ";
    let insert = "0 INSERT 2 Flat 10 5 20 0 30 0 210 0 220 0 230 -1 ";
    let doc = read(&text(&drawing(0, "", block, insert), "\n"));
    // The object's x axis is the world's -x, so its point (5,0,0) is the world's (-5,0,0).
    let (lo, hi) = bounds(&mesh(&doc));
    assert!(
        near(lo, [-6.0, 0.0, -1.0]) && near(hi, [-5.0, 0.0, 0.0]),
        "{lo:?} {hi:?}"
    );
    // The other axes, by the arbitrary axis algorithm.
    assert_eq!(
        axes([0.0, 0.0, -1.0]),
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    );
    let [x, y, z] = axes([1.0, 0.0, 0.0]);
    assert_eq!((x, y, z), ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]));
}

#[test]
fn nested_blocks_cycles_and_depth() {
    let blocks = format!(
        "{TRI_BLOCK} 0 BLOCK 2 Outer 10 0 20 0 30 0 0 INSERT 2 Tri 10 0 20 0 30 5 0 INSERT 2 Tri 10 0 20 0 30 9 0 ENDBLK \
         0 BLOCK 2 Loop 10 0 20 0 30 0 0 INSERT 2 Tri 0 INSERT 2 Loop 0 ENDBLK"
    );
    let entities = "0 INSERT 2 Outer 10 0 20 0 30 0 0 INSERT 2 Loop 0 INSERT 2 Missing";
    let doc = read(&text(&drawing(0, "", &blocks, entities), "\n"));
    assert_eq!(triangles(&doc), 3);
    assert_eq!(doc.json["meshes"].as_array().unwrap().len(), 1);
    let notes = source(&doc)["notes"].to_string();
    assert!(
        notes.contains("contains itself") && notes.contains("isn't there"),
        "{notes}"
    );
    assert_eq!(mesh(&doc).triangles(), 3);
    // A chain deeper than the most nested.
    let mut chain = String::from(TRI_BLOCK);
    for k in 0..40 {
        let inner = if k == 0 {
            "Tri".to_owned()
        } else {
            format!("C{}", k - 1)
        };
        write!(chain, "0 BLOCK 2 C{k} 0 INSERT 2 {inner} 0 ENDBLK ").unwrap();
    }
    let doc = read(&text(&drawing(0, "", &chain, "0 INSERT 2 C39"), "\n"));
    assert_eq!(triangles(&doc), 0);
    assert!(source(&doc)["notes"].to_string().contains("too deep"));
}

#[test]
fn colors_by_layer_index_true_color_and_block() {
    let tables = "0 LAYER 2 Red 70 0 62 1 0 LAYER 2 Off 70 0 62 -3 0 LAYER 2 Frozen 70 1 62 3 \
                  0 LAYER 2 Deep 70 0 62 7 420 1193046";
    let block = "0 BLOCK 2 Part 10 0 20 0 30 0 \
                 0 3DFACE 8 0 62 0 10 0 20 0 30 0 11 1 21 0 31 0 12 0 22 1 32 0 \
                 0 3DFACE 8 0 10 0 20 0 30 1 11 1 21 0 31 1 12 0 22 1 32 1 0 ENDBLK";
    let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let entities = [
        face("8 Red", &tri),
        face("8 red 62 5", &tri),
        face("8 Red 62 5 420 65408", &tri),
        face("8 Off", &tri),
        face("8 Frozen", &tri),
        face("8 Red 62 0", &tri),
        "0 INSERT 8 Deep 62 2 2 Part 10 9 20 0 30 0 ".to_owned(),
        "0 INSERT 8 Off 2 Part ".to_owned(),
    ]
    .concat();
    let doc = read(&text(&drawing(0, tables, block, &entities), "\n"));
    assert_eq!(triangles(&doc), 6);
    let m = mesh(&doc);
    let lin = |rgb: u32| {
        let c = |s: u32| srgb_to_linear(f32::from(u8::try_from((rgb >> s) & 0xFF).unwrap()) / 255.0);
        [c(16), c(8), c(0), 1.0].map(f32::to_bits)
    };
    let mut looks: Vec<(String, [u32; 4])> = m
        .materials
        .iter()
        .map(|x| (x.name.clone(), x.color.map(f32::to_bits)))
        .collect();
    looks.sort();
    let mut want = vec![
        ("Red".to_owned(), lin(0xFF_0000)),
        ("Red".to_owned(), lin(0x00_00FF)),
        ("Red".to_owned(), lin(0x00_FF80)),
        // A block's color at the top is white.
        ("Red".to_owned(), lin(0xFF_FFFF)),
        // In the block, layer 0 is the reference's layer, `BYBLOCK` its color.
        ("Deep".to_owned(), lin(0xFF_FF00)),
        ("Deep".to_owned(), lin(0x12_3456)),
    ];
    want.sort();
    assert_eq!(looks, want);
}

#[test]
fn color_indices() {
    for (c, rgb) in [
        (1, 0xFF_0000),
        (5, 0x00_00FF),
        (7, 0xFF_FFFF),
        (8, 0x80_8080),
        (10, 0xFF_0000),
        (11, 0xFF_7F7F),
        (12, 0xA5_0000),
        (30, 0xFF_7F00),
        (140, 0x00_BFFF),
        (250, 0x33_3333),
        (255, 0xFF_FFFF),
    ] {
        assert_eq!(aci(c), rgb, "{c}");
    }
}

#[test]
fn binary_files() {
    let gs = drawing(
        5,
        "0 LAYER 2 A 62 1",
        TRI_BLOCK,
        &(cube_faces("8 A") + "0 INSERT 2 Tri 41 3"),
    );
    let doc = read_bytes(&binary(&gs), usize::MAX).unwrap();
    assert_eq!(triangles(&doc), 13);
    let src = source(&doc);
    assert_eq!(src["encoding"], "binary");
    assert_eq!(src["units"], "centimeters");
    let ascii = read(&text(&gs, "\n"));
    assert_eq!(mesh(&doc).prims.len(), mesh(&ascii).prims.len());
    assert_eq!(bounds(&mesh(&doc)), bounds(&mesh(&ascii)));
}

#[test]
fn line_ends_bom_and_spacing() {
    let gs = drawing(6, "", "", &cube_faces("8 0"));
    for eol in ["\n", "\r\n", "\r", " \t\r\n"] {
        let mut bytes = b"\xEF\xBB\xBF".to_vec();
        bytes.extend(text(&gs, eol).bytes());
        let doc = read_bytes(&bytes, usize::MAX).unwrap();
        assert_eq!(triangles(&doc), 12, "{eol:?}");
    }
}

#[test]
fn triangle_limit() {
    let faces = text(&drawing(0, "", "", &cube_faces("8 0")), "\n");
    assert!(read_bytes(faces.as_bytes(), 12).is_ok());
    let err = read_bytes(faces.as_bytes(), 11).err().unwrap().to_string();
    assert!(err.contains("triangles"), "{err}");
    // Instances count each time.
    let array = text(&drawing(0, "", TRI_BLOCK, "0 INSERT 2 Tri 70 4 71 4"), "\n");
    assert!(read_bytes(array.as_bytes(), 16).is_ok());
    let err = read_bytes(array.as_bytes(), 15).err().unwrap().to_string();
    assert!(err.contains("triangles"), "{err}");
}

#[test]
fn skipped_and_left_out() {
    let entities = "0 DIMENSION 2 *D1 0 MTEXT 1 x 0 3DSOLID 1 abc 0 TEXT 1 hi 0 HATCH 2 SOLID ".to_owned()
        + &face("8 0 67 1", &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        + "0 LINE 8 0 67 1 10 0 20 0 30 0 11 1 21 1 31 1 0 CIRCLE 67 1 40 1 \
           0 SPLINE 67 1 71 1 10 0 20 0 30 0 10 1 20 0 30 0";
    let doc = read(&text(&drawing(0, "", "", &entities), "\n"));
    assert_eq!(triangles(&doc), 0);
    let src = source(&doc);
    assert_eq!(
        src["skipped"],
        json!({ "3DSOLID": 1, "DIMENSION": 1, "HATCH": 1, "MTEXT": 1, "TEXT": 1 })
    );
    assert!(src["notes"].to_string().contains("paper space"));
    assert!(doc.json["scenes"][0].get("nodes").is_none());
}

#[test]
fn truncated_and_not_dxf() {
    let full = text(&drawing(0, "", "", &cube_faces("8 0")), "\n");
    let cut = &full[..full.len() / 2];
    let doc = read(cut);
    assert_eq!(source(&doc)["truncated"], true);
    for bad in ["", "hello world", "0\nLINE\n", "999\ncomment\n"] {
        assert!(read_bytes(bad.as_bytes(), usize::MAX).is_err(), "{bad:?}");
    }
    let err = read_bytes(b" \t\r\n", usize::MAX).err().unwrap().to_string();
    assert!(err.contains("empty"), "{err}");
}

#[test]
fn damaged_files_never_panic() {
    let tables = "0 LAYER 2 A 62 1 0 LAYER 2 B 62 -2";
    let blocks = format!("{TRI_BLOCK} 0 BLOCK 2 Loop 0 INSERT 2 Tri 0 INSERT 2 Loop 70 2 0 ENDBLK");
    let entities = format!(
        "{}{}{}0 LWPOLYLINE 70 1 10 0 20 0 42 1 10 1 20 0 42 1e300 10 2 20 2 \
         0 INSERT 8 B 2 Loop 70 2 71 2 44 1 45 1 50 30 210 1 220 1 230 1 0 LINE 10 0 20 0 \
         0 CIRCLE 10 1 20 1 40 2 210 0 220 0 230 -1 0 ARC 40 1 50 350 51 10 \
         0 ELLIPSE 11 2 21 0 31 0 40 0.5 41 0 42 3 \
         0 SPLINE 70 4 71 2 40 0 40 0 40 0 40 1 40 1 40 1 41 1 41 0.7 41 1 \
         10 0 20 0 30 0 10 1 20 1 30 0 10 2 20 0 30 0 0 SPLINE 71 3 11 0 21 0 11 1 21 1",
        face(
            "8 A 62 3",
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        ),
        polyface_cube("62 4").replace("Box", "A"),
        grid(2, 3, 32),
    );
    let gs = drawing(4, tables, &blocks, &entities);
    let check = |bytes: &[u8]| {
        if let Ok(doc) = read_bytes(bytes, 64) {
            let _ = doc.flatten(Pose::Rest);
            let _ = doc.to_glb();
        }
    };
    for sample in [text(&gs, "\n").into_bytes(), binary(&gs)] {
        for n in 0..=sample.len() {
            check(&sample[..n]);
        }
        let mut bytes = sample.clone();
        for i in 0..bytes.len() {
            for flip in [0x01, 0x20, 0x80, 0xff] {
                bytes[i] ^= flip;
                check(&bytes);
                bytes[i] ^= flip;
            }
        }
    }
}

#[test]
fn lines_points_and_polylines() {
    // A LINE, a POINT, an open 2D POLYLINE at elevation 2, a closed 3D one and a closed
    // LWPOLYLINE square.
    let entities = "0 LINE 10 0 20 0 30 0 11 1 21 2 31 3 0 POINT 10 5 20 0 30 0 \
                    0 POLYLINE 66 1 10 0 20 0 30 2 70 0 0 VERTEX 10 0 20 0 30 0 0 VERTEX 10 1 20 0 30 0 \
                    0 VERTEX 10 1 20 1 30 0 0 SEQEND \
                    0 POLYLINE 66 1 70 9 0 VERTEX 10 0 20 0 30 0 70 32 0 VERTEX 10 0 20 0 30 1 70 32 \
                    0 VERTEX 10 0 20 1 30 1 70 32 0 SEQEND \
                    0 LWPOLYLINE 90 4 70 1 10 0 20 0 10 1 20 0 10 1 20 1 10 0 20 1 ";
    let doc = read(&text(&drawing(6, "", "", entities), "\n"));
    assert_eq!(triangles(&doc), 0);
    let m = mesh(&doc);
    assert_eq!((m.lines(), m.points()), (1 + 2 + 3 + 4, 1));
    let (lo, hi) = bounds(&m);
    assert!(near(lo, [0.0, 0.0, -2.0]) && near(hi, [5.0, 3.0, 0.0]), "{lo:?} {hi:?}");
    // Left out with lines and points.
    let keep = Keep {
        lines_and_points: false,
        ..Keep::ALL
    };
    let text = text(&drawing(6, "", "", entities), "\n");
    let doc = super::super::load_bytes(text.as_bytes(), "a.dxf", "dxf", &Memory::new(), keep)
        .map(Document::into_owned)
        .unwrap();
    let m = doc.flatten(Pose::Rest).unwrap();
    assert_eq!(m.map_or(0, |m| m.lines() + m.points()), 0);
}

#[test]
fn bulges_are_arcs() {
    // A bulge of 1 is a half turn counterclockwise: 16 segments, so 15 points between the ends,
    // on the circle through both. A negative bulge turns the other way.
    for (bulge, side) in [(1.0, -1.0), (-1.0, 1.0)] {
        let mut out = Vec::new();
        bulge_arc([0.0, 0.0], [2.0, 0.0], bulge, &mut out);
        assert_eq!(out.len(), 15);
        assert!(out.iter().all(|p| ((p[0] - 1.0).hypot(p[1]) - 1.0).abs() < 1e-9));
        assert!(
            (out[7][0] - 1.0).abs() < 1e-9 && (out[7][1] - side).abs() < 1e-9,
            "{:?}",
            out[7]
        );
    }
    // A quarter turn is 8 segments; no bulge, or no chord, is no arc.
    let mut out = Vec::new();
    bulge_arc([0.0, 0.0], [1.0, 1.0], (std::f64::consts::PI / 8.0).tan(), &mut out);
    assert_eq!(out.len(), 7);
    for (a, b, bulge) in [([0.0, 0.0], [1.0, 0.0], 0.0), ([1.0, 1.0], [1.0, 1.0], 1.0)] {
        out.clear();
        bulge_arc(a, b, bulge, &mut out);
        assert_eq!(out, Vec::<[f64; 2]>::new());
    }
    // An LWPOLYLINE's arc in a drawing: out to x = 3 at elevation 1.
    let lw = "0 LWPOLYLINE 90 4 70 1 38 1 10 0 20 0 10 2 20 0 42 1 10 2 20 2 10 0 20 2 ";
    let m = mesh(&read(&text(&drawing(6, "", "", lw), "\n")));
    assert_eq!(m.lines(), 19);
    let (lo, hi) = bounds(&m);
    assert!(near(lo, [0.0, 1.0, -2.0]) && near(hi, [3.0, 1.0, 0.0]), "{lo:?} {hi:?}");
}

#[test]
fn circles_and_arcs() {
    // A circle is 32 segments round its center in the object coordinate system, at the center's
    // height; an arc a segment for each sixteenth of a half turn from its start counterclockwise.
    let one = |e: &str| mesh(&read(&text(&drawing(6, "", "", e), "\n")));
    let m = one("0 CIRCLE 10 1 20 2 30 3 40 2");
    assert_eq!(m.lines(), 32);
    let (lo, hi) = bounds(&m);
    assert!(
        near(lo, [-1.0, 3.0, -4.0]) && near(hi, [3.0, 3.0, 0.0]),
        "{lo:?} {hi:?}"
    );
    for (from, to, lines) in [(0, 90, 8), (350, 10, 2), (90, 0, 24), (0, 360, 32), (45, 45, 32)] {
        let m = one(&format!("0 ARC 40 1 50 {from} 51 {to}"));
        assert_eq!(m.lines(), lines, "{from} to {to}");
    }
    // A quarter turn from 0 is in the first quadrant: x and y (-z here) from 0 to 1.
    let (lo, hi) = bounds(&one("0 ARC 40 1 50 0 51 90"));
    assert!(near(lo, [0.0, 0.0, -1.0]) && near(hi, [1.0, 0.0, 0.0]), "{lo:?} {hi:?}");
    // Extruded down, the object coordinate system's x is the world's -x: the center at x 1 is at
    // x -1, and its elevation 2 is at z -2.
    let (lo, hi) = bounds(&one("0 CIRCLE 10 1 20 0 30 2 40 0.5 210 0 220 0 230 -1"));
    assert!(
        near(lo, [-1.5, -2.0, -0.5]) && near(hi, [-0.5, -2.0, 0.5]),
        "{lo:?} {hi:?}"
    );
    // No radius, or one that isn't a number, draws nothing.
    let doc = read(&text(&drawing(6, "", "", "0 CIRCLE 40 0 0 ARC 40 nan 51 90"), "\n"));
    assert!(doc.flatten(Pose::Rest).unwrap().is_none_or(|m| m.lines() == 0));
}

#[test]
fn ellipses_whole_and_parts() {
    let one = |e: &str| mesh(&read(&text(&drawing(6, "", "", e), "\n")));
    // A whole ellipse: its major axis along x, half as wide along y (-z here).
    let m = one("0 ELLIPSE 10 1 20 0 30 0 11 2 21 0 31 0 40 0.5 41 0 42 6.283185307179586");
    assert_eq!(m.lines(), 32);
    let (lo, hi) = bounds(&m);
    assert!(
        near(lo, [-1.0, 0.0, -1.0]) && near(hi, [3.0, 0.0, 1.0]),
        "{lo:?} {hi:?}"
    );
    // Half of one, from parameter 0 to a half turn: counterclockwise, through +y.
    let m = one("0 ELLIPSE 11 2 21 0 31 0 40 0.5 41 0 42 3.141592653589793");
    assert_eq!(m.lines(), 16);
    let (lo, hi) = bounds(&m);
    assert!(
        near(lo, [-2.0, 0.0, -1.0]) && near(hi, [2.0, 0.0, 0.0]),
        "{lo:?} {hi:?}"
    );
    // Extruded down, counterclockwise about -z is through -y.
    let (lo, hi) = bounds(&one(
        "0 ELLIPSE 11 2 21 0 31 0 40 0.5 41 0 42 3.141592653589793 210 0 220 0 230 -1",
    ));
    assert!(near(lo, [-2.0, 0.0, 0.0]) && near(hi, [2.0, 0.0, 1.0]), "{lo:?} {hi:?}");
    // Its axis needn't be along x: a whole one along the line y = z, in the plane x = 0, its
    // minor axis along y = -z.
    let m = one("0 ELLIPSE 11 0 21 1 31 1 40 0.5 210 1 220 0 230 0");
    assert_eq!(m.lines(), 32);
    let far = 1.25f32.sqrt() + 1e-5;
    assert!(
        m.prims[0]
            .positions
            .iter()
            .all(|p| p[0].abs() < 1e-6 && p[1].abs() <= far && p[2].abs() <= far),
        "{:?}",
        m.prims[0].positions
    );
    assert!(m.prims[0].positions.iter().any(|p| near(*p, [0.0, 1.0, -1.0])));
}

#[test]
fn splines_by_knots_weights_and_fit_points() {
    let one = |e: &str| mesh(&read(&text(&drawing(6, "", "", e), "\n")));
    // A quadratic with its knots clamped is a Bezier curve: its middle at (1, 0.5), its control
    // points turning a quarter turn, so 8 segments.
    let quad = "0 SPLINE 70 8 71 2 72 6 73 3 40 0 40 0 40 0 40 1 40 1 40 1 \
                10 0 20 0 30 0 10 1 20 1 30 0 10 2 20 0 30 0";
    let m = one(quad);
    assert_eq!(m.lines(), 8);
    let (lo, hi) = bounds(&m);
    assert!(near(lo, [0.0, 0.0, -0.5]) && near(hi, [2.0, 0.0, 0.0]), "{lo:?} {hi:?}");
    assert!(m.prims[0].positions.iter().any(|p| near(*p, [1.0, 0.0, -0.5])));
    // No knots are uniform and clamped: the same curve.
    let m = one(&quad.replace("40 0 40 0 40 0 40 1 40 1 40 1", ""));
    assert_eq!(bounds(&m), (lo, hi));
    // Knots that don't fit draw the control points, and note it.
    let doc = read(&text(
        &drawing(6, "", "", &quad.replace("40 1 40 1 40 1", "40 1 40 1")),
        "\n",
    ));
    assert_eq!(mesh(&doc).lines(), 2);
    assert!(source(&doc)["notes"].to_string().contains("knots don't fit"));
    // Only fit points: drawn through them, closed by flag 1.
    let m = one("0 SPLINE 70 1 71 3 74 3 11 0 21 0 31 0 11 1 21 1 31 0 11 2 21 0 31 0");
    assert_eq!(m.lines(), 3);
    // A spline in a block is drawn where the block is inserted.
    let block = "0 BLOCK 2 S 10 0 20 0 30 0 0 SPLINE 71 1 10 0 20 0 30 0 10 1 20 0 30 0 0 ENDBLK";
    let doc = read(&text(&drawing(6, "", block, "0 INSERT 2 S 10 5 20 0 30 0"), "\n"));
    let (lo, hi) = bounds(&mesh(&doc));
    assert!(near(lo, [5.0, 0.0, 0.0]) && near(hi, [6.0, 0.0, 0.0]), "{lo:?} {hi:?}");
}

#[test]
fn nurbs_curves() {
    // A quarter circle, as a rational quadratic: every point is on the circle.
    let ctrl = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    let knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let weights = [1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0];
    let mut out = Vec::new();
    assert_eq!(nurbs(2, &ctrl, &weights, &knots, usize::MAX, &mut out), Ok(9));
    assert_eq!(out.len(), 9);
    assert!(out.iter().all(|p| (p[0].hypot(p[1]) - 1.0).abs() < 1e-12), "{out:?}");
    assert_eq!((out[0], out[8]), (ctrl[0], ctrl[2]));
    // Weights that aren't one for each control point, or aren't all above zero, aren't used.
    for weights in [&weights[..2], &[1.0, 0.0, 1.0], &[1.0, f64::NAN, 1.0]] {
        nurbs(2, &ctrl, weights, &knots, usize::MAX, &mut out).unwrap();
        assert!(out.iter().any(|p| (p[0].hypot(p[1]) - 1.0).abs() > 1e-3));
    }
    // A cubic with unclamped knots is drawn over its knots from the fourth to the fourth last.
    let line: Vec<[f64; 3]> = (0..6).map(|i| [f64::from(i), 0.0, 0.0]).collect();
    let uniform: Vec<f64> = (0..10).map(f64::from).collect();
    nurbs(3, &line, &[], &uniform, usize::MAX, &mut out).unwrap();
    let (first, last) = (out[0][0], out[out.len() - 1][0]);
    assert!((first - 1.0).abs() < 1e-12 && (last - 4.0).abs() < 1e-12, "{out:?}");
    // Degrees that don't fit, knots that go back, aren't numbers or leave no span; or more points
    // than the budget.
    let bad = Err("splines whose degree or knots don't fit were drawn through their control points");
    assert_eq!(nurbs(0, &ctrl, &[], &knots, usize::MAX, &mut out), bad);
    assert_eq!(nurbs(3, &ctrl, &[], &[0.0; 7], usize::MAX, &mut out), bad);
    assert_eq!(nurbs(12, &[[0.0; 3]; 13], &[], &[], usize::MAX, &mut out), bad);
    assert_eq!(nurbs(i64::MIN, &ctrl, &[], &knots, usize::MAX, &mut out), bad);
    for knots in [
        [0.0, 0.0, 1.0, 0.5, 1.0, 1.0],
        [0.0, 0.0, 0.0, f64::NAN, 1.0, 1.0],
        [0.0; 6],
    ] {
        assert_eq!(nurbs(2, &ctrl, &[], &knots, usize::MAX, &mut out), bad, "{knots:?}");
    }
    assert_eq!(
        nurbs(2, &ctrl, &weights, &knots, 8, &mut out),
        Err("curves past the most points were drawn coarsely")
    );
}

#[test]
fn curves_past_the_most_points_are_coarse() {
    // A circle past the budget is a segment for each quarter turn.
    let mut m = Model {
        arc_points: MAX_ARC_POINTS - 1,
        spaces: vec![Space::default()],
        ..Model::default()
    };
    let bytes = text(&groups("40 1 0 EOF"), "\n");
    let mut p = Pairs {
        d: bytes.as_bytes(),
        at: 0,
        binary: false,
        wide: false,
        back: None,
    };
    m.conic(b"CIRCLE", &mut p, 0);
    assert_eq!(m.spaces[0].buckets[0].lines.len(), 4);
    assert!(m.notes.contains("curves past the most points were drawn coarsely"));
    assert_eq!(m.arc_points, MAX_ARC_POINTS - 1);
}
