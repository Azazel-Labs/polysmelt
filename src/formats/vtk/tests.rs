use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::Value;

use super::super::load_bytes;
use crate::assets::Memory;
use crate::document::{Document, Keep, Pose};
use crate::{ErrorKind, Format, Result};

fn read_with(bytes: &[u8], name: &str, max: usize) -> Result<Document> {
    let ext = name.rsplit('.').next().unwrap();
    load_bytes(bytes, name, ext, &Memory::new(), Keep::ALL.with_max_triangles(max))
}

fn read(bytes: &[u8], name: &str) -> Document {
    read_with(bytes, name, usize::MAX).unwrap()
}

/// What a document of one piece draws.
#[derive(Debug, Default, PartialEq)]
struct Got {
    positions: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    lines: Vec<[u32; 2]>,
    points: Vec<u32>,
    attributes: BTreeMap<String, Vec<f32>>,
}

fn got(doc: &Document) -> Got {
    let mut out = Got::default();
    let prims = doc.json()["meshes"][0]["primitives"].as_array().unwrap().clone();
    let index = |v: &Value| usize::try_from(v.as_u64().unwrap()).unwrap();
    for (k, p) in prims.iter().enumerate() {
        assert_eq!(p["attributes"], prims[0]["attributes"], "one piece shares its vertices");
        if k == 0 {
            for (name, accessor) in p["attributes"].as_object().unwrap() {
                let (_, values) = doc.read_floats(index(accessor)).unwrap();
                if name == "POSITION" {
                    out.positions = values.as_chunks::<3>().0.to_vec();
                } else {
                    out.attributes.insert(name.clone(), values);
                }
            }
        }
        let indices = match p.get("indices") {
            Some(a) => doc.read_indices(index(a)).unwrap(),
            None => (0..u32::try_from(out.positions.len()).unwrap()).collect(),
        };
        match p.get("mode").and_then(Value::as_u64).unwrap_or(4) {
            4 => out.tris = indices.as_chunks::<3>().0.to_vec(),
            1 => out.lines = indices.as_chunks::<2>().0.to_vec(),
            _ => out.points = indices,
        }
    }
    out
}

fn source(doc: &Document) -> Value {
    doc.json()["asset"]["extras"]["source"].clone()
}

// Two tetrahedra sharing the face of points 1, 2 and 3, with a scalar at each point.
const POINTS: [f32; 15] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
];
const NODES: [i64; 8] = [0, 1, 2, 3, 1, 2, 3, 4];
const SCALARS: [f32; 5] = [0.5, 1.5, 2.5, 3.5, 4.5];

const ASCII: &str = "# vtk DataFile Version 4.2
two tetrahedra
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 5 float
0 0 0 1 0 0 0 1 0
0 0 1 1 1 1
CELLS 2 10
4 0 1 2 3
4 1 2 3 4
CELL_TYPES 2
10
10
POINT_DATA 5
SCALARS t float 1
LOOKUP_TABLE default
0.5 1.5 2.5 3.5 4.5
";

fn be<const N: usize>(values: impl IntoIterator<Item = [u8; N]>) -> Vec<u8> {
    values.into_iter().flatten().collect()
}

/// The two tetrahedra as a binary legacy file, in the old cell layout or version 5's.
fn binary(v5: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let version = if v5 { "5.1" } else { "4.2" };
    write!(
        out,
        "# vtk DataFile Version {version}\ntwo tetrahedra\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
    )
    .unwrap();
    out.extend(b"POINTS 5 float\n");
    out.extend(be(POINTS.map(f32::to_be_bytes)));
    if v5 {
        out.extend(b"\nCELLS 3 8\nOFFSETS vtktypeint64\n");
        out.extend(be([0i64, 4, 8].map(i64::to_be_bytes)));
        out.extend(b"\nCONNECTIVITY vtktypeint64\n");
        out.extend(be(NODES.map(i64::to_be_bytes)));
    } else {
        out.extend(b"\nCELLS 2 10\n");
        out.extend(be([4i32, 0, 1, 2, 3, 4, 1, 2, 3, 4].map(i32::to_be_bytes)));
    }
    out.extend(b"\nCELL_TYPES 2\n");
    out.extend(be([10i32, 10].map(i32::to_be_bytes)));
    out.extend(b"\nPOINT_DATA 5\nSCALARS t float 1\nLOOKUP_TABLE default\n");
    out.extend(be(SCALARS.map(f32::to_be_bytes)));
    out.push(b'\n');
    out
}

/// The bytes of a test file's compressed blocks, so an array takes several.
const SIZE: usize = 16;

/// How an XML file holds its arrays.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Text,
    /// Base64 between the tags, a block's header and data encoded apart.
    Inline,
    /// The same, header and data encoded as one, as meshio writes it.
    InlineJoined,
    AppendedRaw,
    AppendedBase64,
}

/// Writes the two tetrahedra as a `.vtu` in any of the ways one can be written.
struct Vtu {
    mode: Mode,
    zlib: bool,
    wide: bool,
    big: bool,
    /// The compressor the file names.
    compressor: &'static str,
}

impl Vtu {
    fn new(mode: Mode) -> Self {
        Self {
            mode,
            zlib: false,
            wide: false,
            big: false,
            compressor: "vtkZLibDataCompressor",
        }
    }

    fn word(&self, n: usize) -> Vec<u8> {
        match (self.wide, self.big) {
            (true, true) => (n as u64).to_be_bytes().to_vec(),
            (true, false) => (n as u64).to_le_bytes().to_vec(),
            (false, true) => u32::try_from(n).unwrap().to_be_bytes().to_vec(),
            (false, false) => u32::try_from(n).unwrap().to_le_bytes().to_vec(),
        }
    }

    /// A block's header and its data. Compressed blocks are 16 bytes each, so an array takes several.
    fn block(&self, data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        if !self.zlib {
            return (self.word(data.len()), data.to_vec());
        }
        let chunks: Vec<Vec<u8>> = data
            .chunks(SIZE)
            .map(|c| {
                let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                z.write_all(c).unwrap();
                z.finish().unwrap()
            })
            .collect();
        let mut header = self.word(chunks.len());
        header.extend(self.word(SIZE));
        header.extend(self.word(data.len() % SIZE));
        for c in &chunks {
            header.extend(self.word(c.len()));
        }
        (header, chunks.concat())
    }

    fn array(&self, kind: &str, attrs: &str, text: &str, data: &[u8], appended: &mut Vec<u8>) -> String {
        let (header, body) = self.block(data);
        let split = || STANDARD.encode(&header) + &STANDARD.encode(&body);
        match self.mode {
            Mode::Text => format!("<DataArray type=\"{kind}\" {attrs} format=\"ascii\">{text}</DataArray>\n"),
            Mode::Inline => format!(
                "<DataArray type=\"{kind}\" {attrs} format=\"binary\">\n  {}\n</DataArray>\n",
                split()
            ),
            Mode::InlineJoined => format!(
                "<DataArray type=\"{kind}\" {attrs} format=\"binary\">{}</DataArray>\n",
                STANDARD.encode([header, body].concat())
            ),
            Mode::AppendedRaw | Mode::AppendedBase64 => {
                let offset = appended.len();
                if self.mode == Mode::AppendedRaw {
                    appended.extend(header);
                    appended.extend(body);
                } else {
                    appended.extend(split().into_bytes());
                }
                format!("<DataArray type=\"{kind}\" {attrs} format=\"appended\" offset=\"{offset}\"/>\n")
            }
        }
    }

    fn write(&self) -> Vec<u8> {
        let big = self.big;
        let f32s = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| if big { x.to_be_bytes() } else { x.to_le_bytes() })
                .collect()
        };
        let i64s = |v: &[i64]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| if big { x.to_be_bytes() } else { x.to_le_bytes() })
                .collect()
        };
        let mut appended = Vec::new();
        let mut out = String::from("<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\"");
        out += if big {
            " byte_order=\"BigEndian\""
        } else {
            " byte_order=\"LittleEndian\""
        };
        out += if self.wide {
            " header_type=\"UInt64\""
        } else {
            " header_type=\"UInt32\""
        };
        if self.zlib {
            write!(out, " compressor=\"{}\"", self.compressor).unwrap();
        }
        out += ">\n<UnstructuredGrid>\n<Piece NumberOfPoints=\"5\" NumberOfCells=\"2\">\n<PointData Scalars=\"t\">\n";
        out += &self.array(
            "Float32",
            "Name=\"t\"",
            "0.5 1.5 2.5 3.5 4.5",
            &f32s(&SCALARS),
            &mut appended,
        );
        out += "</PointData>\n<Points>\n";
        out += &self.array(
            "Float32",
            "NumberOfComponents=\"3\"",
            "\n0 0 0 1 0 0 0 1 0\n0 0 1 1 1 1\n",
            &f32s(&POINTS),
            &mut appended,
        );
        out += "</Points>\n<Cells>\n";
        out += &self.array(
            "Int64",
            "Name=\"connectivity\"",
            "0 1 2 3 1 2 3 4",
            &i64s(&NODES),
            &mut appended,
        );
        out += &self.array("Int64", "Name=\"offsets\"", "4 8", &i64s(&[4, 8]), &mut appended);
        out += &self.array("UInt8", "Name=\"types\"", "10 10", &[10, 10], &mut appended);
        out += "</Cells>\n</Piece>\n</UnstructuredGrid>\n";
        let mut out = out.into_bytes();
        if matches!(self.mode, Mode::AppendedRaw | Mode::AppendedBase64) {
            let encoding = if self.mode == Mode::AppendedRaw {
                "raw"
            } else {
                "base64"
            };
            out.extend(format!("<AppendedData encoding=\"{encoding}\">\n  _").into_bytes());
            out.extend(appended);
            out.extend(b"\n</AppendedData>\n");
        }
        out.extend(b"</VTKFile>\n");
        out
    }
}

#[test]
fn tetrahedra_draw_the_faces_they_dont_share() {
    let g = got(&read(ASCII.as_bytes(), "t.vtk"));
    assert_eq!(g.positions.as_flattened(), POINTS);
    assert_eq!(g.attributes["_T"], SCALARS);
    // Each tetrahedron's four faces but the one between them, wound as VTK winds them.
    assert_eq!(
        g.tris,
        [[0, 1, 3], [2, 0, 3], [0, 2, 1], [1, 2, 4], [2, 3, 4], [3, 1, 4]]
    );
    assert!(g.lines.is_empty() && g.points.is_empty());
}

#[test]
fn binary_legacy_files_read_as_the_text_does() {
    let text = got(&read(ASCII.as_bytes(), "t.vtk"));
    for v5 in [false, true] {
        let doc = read(&binary(v5), "t.vtk");
        assert_eq!(got(&doc), text, "version 5 layout: {v5}");
        assert_eq!(source(&doc)["encoding"], "binary");
    }
    let doc = read(ASCII.as_bytes(), "t.vtk");
    let info = doc.source_info();
    assert_eq!(info.format.as_deref(), Some("VTK"));
    assert_eq!(info.version.as_deref(), Some("4.2"));
    assert_eq!(info.title.as_deref(), Some("two tetrahedra"));
    assert_eq!(info.up_axis, crate::source::Fact::Unknown);
    assert_eq!(source(&doc)["cells"]["tetra"], 2);
    assert_eq!(doc.facts["triangles"], 6);
}

#[test]
fn text_cells_in_the_version_5_layout() {
    let v5 = ASCII.replace(
        "CELLS 2 10\n4 0 1 2 3\n4 1 2 3 4\n",
        "CELLS 3 8\nOFFSETS vtktypeint64\n0 4 8\nCONNECTIVITY vtktypeint64\n0 1 2 3 1 2 3 4\n",
    );
    assert_ne!(v5, ASCII);
    assert_eq!(
        got(&read(v5.as_bytes(), "t.vtk")),
        got(&read(ASCII.as_bytes(), "t.vtk"))
    );
}

#[test]
fn xml_reads_the_same_however_its_arrays_are_written() {
    let want = got(&read(ASCII.as_bytes(), "t.vtk"));
    let mut seen = 0;
    for mode in [
        Mode::Text,
        Mode::Inline,
        Mode::InlineJoined,
        Mode::AppendedRaw,
        Mode::AppendedBase64,
    ] {
        for zlib in [false, true] {
            for wide in [false, true] {
                for big in [false, true] {
                    if zlib && mode == Mode::InlineJoined {
                        continue;
                    }
                    let file = Vtu {
                        zlib,
                        wide,
                        big,
                        ..Vtu::new(mode)
                    }
                    .write();
                    let doc = read_with(&file, "t.vtu", usize::MAX)
                        .unwrap_or_else(|e| panic!("{mode:?} zlib {zlib} wide {wide} big {big}: {e}"));
                    assert_eq!(got(&doc), want, "{mode:?} zlib {zlib} wide {wide} big {big}");
                    seen += 1;
                }
            }
        }
    }
    assert_eq!(seen, 36);
}

#[test]
fn base64_may_be_broken_across_lines() {
    let file = String::from_utf8(Vtu::new(Mode::Inline).write()).unwrap();
    // A line break every few characters of each array.
    let mut broken = String::new();
    for line in file.lines() {
        if line.starts_with("  ") && !line.contains('<') {
            for chunk in line.trim().as_bytes().chunks(7) {
                broken += std::str::from_utf8(chunk).unwrap();
                broken += "\r\n \t";
            }
        } else {
            broken += line;
        }
        broken.push('\n');
    }
    assert_ne!(broken, file);
    assert_eq!(
        got(&read(broken.as_bytes(), "t.vtu")),
        got(&read(file.as_bytes(), "t.vtu"))
    );
}

#[test]
fn other_compressors_are_refused_by_name() {
    for name in ["vtkLZ4DataCompressor", "vtkLZMADataCompressor"] {
        for mode in [Mode::Inline, Mode::AppendedRaw, Mode::AppendedBase64] {
            let file = Vtu {
                zlib: true,
                compressor: name,
                ..Vtu::new(mode)
            }
            .write();
            let err = read_with(&file, "t.vtu", usize::MAX).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::Unsupported, "{name}");
            assert!(err.to_string().contains(name), "{err}");
        }
        // Text arrays aren't compressed, whatever the file names.
        let file = Vtu {
            zlib: true,
            compressor: name,
            ..Vtu::new(Mode::Text)
        }
        .write();
        assert_eq!(got(&read(&file, "t.vtu")).tris.len(), 6);
    }
}

#[test]
fn a_block_of_hexahedra_is_its_outside() {
    // Three points a side: eight cells, 24 quads outside, and the middle point unused.
    let mut text =
        String::from("# vtk DataFile Version 3.0\nblock\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 27 float\n");
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                writeln!(text, "{x} {y} {z}").unwrap();
            }
        }
    }
    text += "CELLS 8 72\n";
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let id = |dx: i32, dy: i32, dz: i32| (x + dx) + 3 * ((y + dy) + 3 * (z + dz));
                let c = [
                    id(0, 0, 0),
                    id(1, 0, 0),
                    id(1, 1, 0),
                    id(0, 1, 0),
                    id(0, 0, 1),
                    id(1, 0, 1),
                    id(1, 1, 1),
                    id(0, 1, 1),
                ];
                writeln!(text, "8 {}", c.map(|i| i.to_string()).join(" ")).unwrap();
            }
        }
    }
    text += "CELL_TYPES 8\n12 12 12 12 12 12 12 12\n";
    let doc = read(text.as_bytes(), "block.vtk");
    let g = got(&doc);
    assert_eq!(g.tris.len(), 48);
    assert_eq!(g.positions.len(), 26);
    assert!(!g.positions.contains(&[1.0, 1.0, 1.0]));
    // Every triangle faces out of the block.
    let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
    for p in &mesh.prims {
        for t in p.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| p.positions[i as usize]);
            let n = crate::mesh::face_normal(a, b, c);
            let out = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0 - 1.0);
            assert!(n[0] * out[0] + n[1] * out[1] + n[2] * out[2] > 0.0);
        }
    }
    // The same block as an image: its cells are implied.
    let image = "# vtk DataFile Version 3.0\nblock\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 3 3 3\n\
                 ORIGIN 0 0 0\nSPACING 1 1 1\n";
    let g = got(&read(image.as_bytes(), "image.vtk"));
    assert_eq!((g.tris.len(), g.positions.len()), (48, 26));
    assert!(!g.positions.contains(&[1.0, 1.0, 1.0]));
}

#[test]
fn structured_sheets_rows_and_points() {
    let file = |dims: &str| {
        format!(
            "# vtk DataFile Version 3.0\ngrid\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS {dims}\nSPACING 2 1 1\n"
        )
    };
    let g = got(&read(file("4 3 1").as_bytes(), "a.vtk"));
    assert_eq!((g.tris.len(), g.positions.len()), (12, 12));
    let g = got(&read(file("1 1 5").as_bytes(), "a.vtk"));
    assert_eq!(g.lines, [[0, 1], [1, 2], [2, 3], [3, 4]]);
    assert_eq!(g.positions[4], [0.0, 0.0, 4.0]);
    let g = got(&read(file("1 1 1").as_bytes(), "a.vtk"));
    assert_eq!((g.points.len(), g.positions.len()), (1, 1));
    let rect = "# vtk DataFile Version 3.0\ngrid\nASCII\nDATASET RECTILINEAR_GRID\nDIMENSIONS 2 2 1\n\
                X_COORDINATES 2 float\n0 4\nY_COORDINATES 2 float\n1 3\nZ_COORDINATES 1 float\n7\n";
    let g = got(&read(rect.as_bytes(), "a.vtk"));
    assert_eq!(
        g.positions,
        [[0.0, 1.0, 7.0], [4.0, 1.0, 7.0], [0.0, 3.0, 7.0], [4.0, 3.0, 7.0]]
    );
    assert_eq!(g.tris, [[0, 1, 3], [0, 3, 2]]);
}

const QUADS: &str = "# vtk DataFile Version 3.0
two quads
ASCII
DATASET POLYDATA
POINTS 6 float
0 0 0 1 0 0 2 0 0 0 1 0 1 1 0 2 1 0
POLYGONS 2 10
4 0 1 4 3
4 1 2 5 4
";

#[test]
fn cell_data_splits_vertices_only_where_cells_differ() {
    let same = format!("{QUADS}CELL_DATA 2\nSCALARS id int 1\nLOOKUP_TABLE default\n7 7\n");
    let g = got(&read(same.as_bytes(), "q.vtk"));
    assert_eq!(g.positions.len(), 6);
    assert_eq!(g.attributes["_ID"], [7.0; 6]);
    let differ = format!("{QUADS}CELL_DATA 2\nSCALARS id int 1\nLOOKUP_TABLE default\n7 9\n");
    let g = got(&read(differ.as_bytes(), "q.vtk"));
    assert_eq!(g.positions.len(), 8);
    assert_eq!(g.tris.len(), 4);
    for (k, t) in g.tris.iter().enumerate() {
        let want: f32 = if k < 2 { 7.0 } else { 9.0 };
        assert!(
            t.iter()
                .all(|&i| g.attributes["_ID"][i as usize].to_bits() == want.to_bits())
        );
    }
}

#[test]
fn colors_normals_and_texture_coordinates() {
    let file = format!(
        "{QUADS}POINT_DATA 6\nCOLOR_SCALARS c 3\n1 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0.5 0.5 0.5\n\
         NORMALS n float\n0 0 2 0 0 2 0 0 2 0 0 2 0 0 2 0 0 0\n\
         TEXTURE_COORDINATES uv 2 float\n0 0 0.5 0 1 0 0 1 0.5 1 1 1\n"
    );
    let g = got(&read(file.as_bytes(), "q.vtk"));
    let gray = crate::mesh::srgb_to_linear(0.5);
    assert_eq!(g.attributes["COLOR_0"][..8], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    assert_eq!(g.attributes["COLOR_0"][20..], [gray, gray, gray, 1.0]);
    assert_eq!(g.attributes["NORMAL"][..3], [0.0, 0.0, 1.0]);
    assert_eq!(
        g.attributes["NORMAL"][15..],
        [0.0, 1.0, 0.0],
        "a normal with no direction"
    );
    assert_eq!(g.attributes["TEXCOORD_0"][..4], [0.0, 1.0, 0.5, 1.0]);
    // Bytes of three components are colors too.
    let file = format!("{QUADS}POINT_DATA 6\nSCALARS rgb unsigned_char 3\n255 0 0 0 255 0 0 0 255 0 0 0 0 0 0 0 0 0\n");
    let g = got(&read(file.as_bytes(), "q.vtk"));
    assert_eq!(g.attributes["COLOR_0"][..8], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn scalars_take_their_lookup_tables_colors() {
    let file = format!(
        "{QUADS}POINT_DATA 6\nSCALARS level float\nLOOKUP_TABLE ramp\n0 1 2 3 4 5\n\
         LOOKUP_TABLE ramp 2\n1 0 0 1\n0 0 1 0.25\n"
    );
    let g = got(&read(file.as_bytes(), "q.vtk"));
    assert_eq!(g.attributes["_LEVEL"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    // The lower half of the range takes the first color, the upper the second.
    let (red, blue) = ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.25]);
    assert_eq!(g.attributes["COLOR_0"], [red, red, red, blue, blue, blue].concat());
}

#[test]
fn surface_line_and_vertex_cells() {
    let file = "# vtk DataFile Version 3.0\ncells\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 7 double\n\
                0 0 0 1 0 0 0 1 0 1 1 0 2 0 0 2 1 0 9 9 9\n\
                CELLS 5 19\n4 0 1 2 3\n4 1 3 4 5\n3 0 1 4\n1 5\n2 2 3\nCELL_TYPES 5\n8 6 4 1 2\n";
    let g = got(&read(file.as_bytes(), "c.vtk"));
    // The point no cell uses is left out.
    assert_eq!(g.positions.len(), 6);
    // A pixel's nodes run along x and then y; a strip's second triangle is turned.
    assert_eq!(g.tris, [[0, 1, 3], [0, 3, 2], [1, 3, 4], [4, 3, 5]]);
    assert_eq!(g.lines, [[0, 1], [1, 4]]);
    assert_eq!(g.points, [5, 2, 3]);
}

#[test]
fn a_collapsed_hexahedron_shares_its_triangle_with_a_wedge() {
    // A wedge written as a hexahedron with two pairs of nodes repeated, and a tetrahedron on its end.
    let file = "# vtk DataFile Version 3.0\ncells\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 7 float\n\
                0 0 0 1 0 0 0 1 0 0 0 1 1 0 1 0 1 1 0.3 0.3 2\n\
                CELLS 2 14\n8 0 1 2 2 3 4 5 5\n4 3 4 5 6\nCELL_TYPES 2\n12 10\n";
    let g = got(&read(file.as_bytes(), "c.vtk"));
    // The wedge's bottom triangle and three quads, and the tetrahedron's three other faces.
    assert_eq!(g.tris.len(), 1 + 6 + 3);
}

#[test]
fn reading_stops_at_a_keyword_it_doesnt_know() {
    let file =
        format!("{QUADS}POINT_DATA 6\nSCALARS a float\nLOOKUP_TABLE default\n1 2 3 4 5 6\nSOMETHING_NEW 3\n1 2 3\n");
    let doc = read(file.as_bytes(), "q.vtk");
    assert_eq!(source(&doc)["unread"], "SOMETHING_NEW");
    assert_eq!(got(&doc).attributes["_A"].len(), 6);
}

#[test]
fn a_binary_long_is_as_wide_as_the_file_makes_it() {
    for wide in [true, false] {
        let mut file = b"# vtk DataFile Version 3.0\nlongs\nBINARY\nDATASET POLYDATA\nPOINTS 3 float\n".to_vec();
        file.extend(be([0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].map(f32::to_be_bytes)));
        file.extend(b"\nPOLYGONS 1 4\n");
        file.extend(be([3i32, 0, 1, 2].map(i32::to_be_bytes)));
        file.extend(b"\nPOINT_DATA 3\nSCALARS n long 1\nLOOKUP_TABLE default\n");
        if wide {
            file.extend(be([5i64, 6, 7].map(i64::to_be_bytes)));
        } else {
            file.extend(be([5i32, 6, 7].map(i32::to_be_bytes)));
        }
        file.extend(b"\nNORMALS up float\n");
        file.extend(be([0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0].map(f32::to_be_bytes)));
        let g = got(&read(&file, "l.vtk"));
        assert_eq!(g.attributes["_N"], [5.0, 6.0, 7.0], "eight bytes: {wide}");
        assert_eq!(g.attributes["NORMAL"].len(), 9);
    }
}

#[test]
fn types_by_their_sized_names() {
    // meshio writes 32-bit integers as `vtktypeint32`, which VTK's own reader refuses.
    for (name, bytes) in [
        ("vtktypeint32", be([5i32, 6, 7].map(i32::to_be_bytes))),
        ("vtktypeuint16", be([5u16, 6, 7].map(u16::to_be_bytes))),
    ] {
        let mut file = b"# vtk DataFile Version 5.1
sized
BINARY
DATASET POLYDATA
POINTS 3 vtktypefloat32
"
        .to_vec();
        file.extend(be([0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].map(f32::to_be_bytes)));
        file.extend(format!("\nPOINT_DATA 3\nSCALARS n {name} 1\nLOOKUP_TABLE default\n").into_bytes());
        file.extend(bytes);
        let g = got(&read(&file, "s.vtk"));
        assert_eq!(g.attributes["_N"], [5.0, 6.0, 7.0], "{name}");
        assert_eq!(g.points, [0, 1, 2]);
    }
}

#[test]
fn files_are_told_by_their_first_bytes() {
    assert_eq!(Format::detect(ASCII.as_bytes()), Some(Format::Vtk));
    assert_eq!(Format::detect(&binary(true)), Some(Format::Vtk));
    assert_eq!(Format::detect(&Vtu::new(Mode::AppendedRaw).write()), Some(Format::Vtk));
    assert_eq!(Format::from_extension("VTP"), Some(Format::Vtk));
    // An XML file named .vtk, and a legacy one named .vtu, are read as what they are.
    let xml = Vtu::new(Mode::Text).write();
    assert_eq!(got(&read(&xml, "t.vtk")), got(&read(ASCII.as_bytes(), "t.vtu")));
}

/// Counts no file of this size could hold are errors, found before anything is allocated for them.
#[test]
fn counts_the_file_cant_hold_are_refused() {
    let head = "# vtk DataFile Version 3.0\nhuge\n";
    let cut = |text: String, name: &str| {
        let err = read_with(text.as_bytes(), name, usize::MAX).unwrap_err();
        assert!(
            matches!(
                err.kind(),
                ErrorKind::Truncated | ErrorKind::LimitExceeded | ErrorKind::Corrupt
            ),
            "{text}: {err}"
        );
    };
    for encoding in ["ASCII", "BINARY"] {
        let poly = format!("{head}{encoding}\nDATASET POLYDATA\n");
        // One point, as text or as twelve bytes.
        let point = if encoding == "ASCII" {
            "0 0 0"
        } else {
            "\0\0\0\0\0\0\0\0\0\0\0\0"
        };
        let one = format!("{poly}POINTS 1 float\n{point}\n");
        cut(format!("{poly}POINTS 4000000000 float\n0 0 0\n"), "a.vtk");
        cut(format!("{poly}POINTS 6148914691236517205 double\n"), "a.vtk");
        cut(format!("{one}POLYGONS 4000000000 4000000000\n"), "a.vtk");
        cut(format!("{one}POLYGONS 4000000000 8\nOFFSETS vtktypeint64\n"), "a.vtk");
        cut(format!("{one}POINT_DATA 4000000000\nSCALARS s double 4\n"), "a.vtk");
        cut(
            format!("{one}POINT_DATA 1\nFIELD f 1\nbig 4000000000 4000000000 bit\n"),
            "a.vtk",
        );
        cut(format!("{poly}LOOKUP_TABLE t 4000000000\n"), "a.vtk");
        // An image's points aren't in the file: what it would draw is held to the triangle limit.
        let image = format!("{head}{encoding}\nDATASET STRUCTURED_POINTS\n");
        cut(format!("{image}DIMENSIONS 60000 60000 1\n"), "a.vtk");
        cut(format!("{image}DIMENSIONS 4000000000 4000000000 4000000000\n"), "a.vtk");
        cut(format!("{image}DIMENSIONS 0 5 5\n"), "a.vtk");
    }
    let image = format!("{head}ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 4000000000 1 1\n");
    cut(image, "a.vtk");
    // XML: a piece's point count, a block's length, and a compressed block's counts and sizes.
    let text = String::from_utf8(Vtu::new(Mode::Text).write()).unwrap();
    cut(
        text.replace("NumberOfPoints=\"5\"", "NumberOfPoints=\"4000000000\""),
        "a.vtu",
    );
    let words = |w: &[u32]| STANDARD.encode(be(w.iter().map(|x| x.to_le_bytes())));
    let with = |attrs: &str, data: String| {
        format!(
            "<VTKFile type=\"UnstructuredGrid\" byte_order=\"LittleEndian\"{attrs}><UnstructuredGrid>\
             <Piece NumberOfPoints=\"1\" NumberOfCells=\"0\"><Points>\
             <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"binary\">{data}</DataArray>\
             </Points></Piece></UnstructuredGrid></VTKFile>"
        )
    };
    let zlib = " compressor=\"vtkZLibDataCompressor\"";
    cut(with("", words(&[4_000_000_000])), "a.vtu");
    cut(with(zlib, words(&[4_000_000_000, 32768, 0])), "a.vtu");
    cut(with(zlib, words(&[1, 2_000_000_000, 0, 20])), "a.vtu");
    cut(with(zlib, words(&[60000, 2_000_000_000, 0, 20])), "a.vtu");
    // A real zlib stream that claims to inflate to more than it does.
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(&[0; 12]).unwrap();
    let packed = z.finish().unwrap();
    let block = |size: u32| words(&[1, size, 0, u32::try_from(packed.len()).unwrap()]) + &STANDARD.encode(&packed);
    cut(with(zlib, block(4096)), "a.vtu");
    cut(with(zlib, block(8)), "a.vtu");
    assert_eq!(got(&read(with(zlib, block(12)).as_bytes(), "a.vtu")).points, [0]);
    // An appended array's offset past the data.
    let raw = crate::bytes::text(&Vtu::new(Mode::AppendedBase64).write()).replace("offset=\"0\"", "offset=\"999999\"");
    cut(raw, "a.vtu");
}

/// Reading any prefix of a file, or the file with a byte changed, never panics, and keeps to the triangle
/// limit.
fn survives(sample: &[u8], name: &str) {
    let check = |bytes: &[u8]| {
        if let Ok(doc) = read_with(bytes, name, 64) {
            if let Ok(Some(m)) = doc.flatten(Pose::Rest) {
                assert!(m.triangles() <= 64);
            }
            let _ = doc.to_glb();
        }
    };
    for n in 0..=sample.len() {
        check(&sample[..n]);
    }
    let mut bytes = sample.to_vec();
    for i in 0..bytes.len() {
        for flip in [0x01, 0x20, 0x80, 0xFF] {
            bytes[i] ^= flip;
            check(&bytes);
            bytes[i] ^= flip;
        }
    }
}

#[test]
fn damaged_files_never_panic() {
    survives(ASCII.as_bytes(), "t.vtk");
    survives(&binary(false), "t.vtk");
    survives(&binary(true), "t.vtk");
    for mode in [Mode::Text, Mode::Inline, Mode::AppendedRaw, Mode::AppendedBase64] {
        survives(&Vtu::new(mode).write(), "t.vtu");
        let zlib = Vtu {
            zlib: true,
            wide: mode == Mode::Inline,
            ..Vtu::new(mode)
        };
        survives(&zlib.write(), "t.vtu");
    }
    survives(
        format!("{QUADS}CELL_DATA 2\nSCALARS id int 1\nLOOKUP_TABLE t\n7 9\nLOOKUP_TABLE t 2\n1 0 0 1 0 1 0 1\n")
            .as_bytes(),
        "q.vtk",
    );
    survives(
        b"# vtk DataFile Version 3.0\ngrid\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 3 2 2\nORIGIN 0 0 0\n\
          SPACING 1 1 1\nPOINT_DATA 12\nSCALARS s float\nLOOKUP_TABLE default\n1 2 3 4 5 6 7 8 9 10 11 12\n\
          CELL_DATA 2\nSCALARS c float\nLOOKUP_TABLE default\n1 2\n",
        "g.vtk",
    );
    survives(
        b"# vtk DataFile Version 3.0\ngrid\nASCII\nDATASET RECTILINEAR_GRID\nDIMENSIONS 2 2 2\n\
          X_COORDINATES 2 float\n0 4\nY_COORDINATES 2 float\n1 3\nZ_COORDINATES 2 float\n7 8\n",
        "g.vtk",
    );
}

#[test]
fn files_cut_short_say_so() {
    let whole = binary(false);
    let cut = &whole[..whole.windows(6).position(|w| w == b"POINTS").unwrap() + 30];
    assert_eq!(
        read_with(cut, "t.vtk", usize::MAX).unwrap_err().kind(),
        ErrorKind::Truncated
    );
    let whole = Vtu::new(Mode::AppendedRaw).write();
    let cut = &whole[..whole.windows(13).position(|w| w == b"<AppendedData").unwrap() + 50];
    assert_eq!(
        read_with(cut, "t.vtu", usize::MAX).unwrap_err().kind(),
        ErrorKind::Truncated
    );
    assert_eq!(
        read_with(b"", "t.vtk", usize::MAX).unwrap_err().kind(),
        ErrorKind::Empty
    );
    assert_eq!(
        read_with(b"solid nothing", "t.vtk", usize::MAX).unwrap_err().kind(),
        ErrorKind::UnknownFormat
    );
}

#[test]
fn the_triangle_limit_holds() {
    let err = read_with(ASCII.as_bytes(), "t.vtk", 5).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::LimitExceeded);
    assert!(read_with(ASCII.as_bytes(), "t.vtk", 6).is_ok());
}
