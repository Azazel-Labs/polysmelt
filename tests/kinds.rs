//! The same failure is the same `ErrorKind` in every reader. Under every extension that is read:
//! a file with no bytes is `Empty`, a line of plain text is `UnknownFormat`, and the first half
//! of a valid file is either `Truncated` or a partial model whose `asset.extras.source.truncated`
//! is `true`. A well-formed file with nothing in it is `Empty`.
#![cfg(all(feature = "fs", feature = "default"))]

use std::path::Path;

use polysmelt::assets::Folder;
use polysmelt::{ErrorKind, Loader};

/// What reading the first half of a valid file gives.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Half {
    /// A `Truncated` error.
    Truncated,
    /// What came before the cut, flagged: `asset.extras.source.truncated` is `true`.
    Partial,
    /// A model, unflagged. Only for formats with no framing at all, where half a file is a
    /// valid, shorter file and nothing can tell: the reason is beside each.
    Whole,
    /// Not yet the same as the others: what the reader gives is beside each. A reader fixed
    /// to say `Truncated` or to flag its partial model fails this test until its row says so.
    Gap(Result<bool, ErrorKind>),
}
use Half::{Gap, Partial, Truncated, Whole};

/// Every extension read, a valid corpus file of it, and what its first half reads as.
const CASES: &[(&str, &str, Half)] = &[
    ("stl", "stl/ascii_crlf_tabs_numbers/cube.stl", Partial),
    ("stl", "stl/binary_cube/cube.stl", Truncated),
    ("ply", "ply/all_types_ascii/cube.ply", Truncated),
    ("ply", "ply/binary_le_cube/cube.ply", Truncated),
    ("vox", "vox/adjacent_faces_culled/two.vox", Truncated),
    // OBJ is statements with no count and no end mark: half of one is a whole, smaller OBJ.
    ("obj", "obj/blender_concave_star/concave_star.obj", Whole),
    ("fbx", "fbx/anim_constant_keys/anim_constant_keys.fbx", Truncated),
    ("3ds", "3ds/bump_gray_image_named_normal/quad.3ds", Truncated),
    ("3mf", "3mf/bambu_filaments_parts_paint/plate.3mf", Truncated),
    ("usd", "usd/cube_usd_crate/cube_usd_crate.usd", Truncated),
    ("usda", "usd/blender_cube_usda/cube_usda.usda", Truncated),
    ("usdc", "usd/blender_animated/animated.usdc", Truncated),
    ("usdz", "usd/blender_textured_usdz/textured_usdz.usdz", Truncated),
    ("gltf", "gltf/gltf_data_uri/cube.gltf", Truncated),
    ("glb", "gltf/animation_short_sampler/anim.glb", Truncated),
    ("skp", "skp/annotation_layers_vff/annotation_layers_vff.skp", Truncated),
    ("bvh", "bvh/basic_two_joint/basic_two_joint.bvh", Truncated),
    ("dae", "dae/accessor_stride_offset_unnamed/cube.dae", Partial),
    ("zae", "dae/zae_manifest/model.zae", Truncated),
    ("kmz", "dae/kmz_model/place.kmz", Truncated),
    ("off", "off/bad_faces_dropped/cube.off", Partial),
    ("vrm", "vrm/vrm0_spec_version_missing/avatar.vrm", Truncated),
    ("vrma", "vrm/vrma_animation/walk.vrma", Truncated),
    ("x3d", "x3d/x3d_2d_lines_and_points/flat.x3d", Partial),
    ("x3dv", "x3d/x3dv_cube/cube.x3dv", Truncated),
    ("x3dz", "x3d/x3dz_gzipped_classic/cube.x3dz", Truncated),
    ("wrl", "x3d/wrl_2d_filled/flat.wrl", Partial),
    ("wrz", "x3d/wrl_gzipped_as_wrz/cube.wrz", Truncated),
    ("dxf", "dxf/binary/cube.dxf", Partial),
    // A `.splat` is 32-byte records and nothing else: half the records are a whole, smaller file.
    ("splat", "splat/broken_position/scene.splat", Whole),
    ("spz", "spz/coordinates_luf/scene.spz", Truncated),
    ("sog", "sog/meta_first_in_zip/scene.sog", Truncated),
    ("las", "las/colors_stored_as_8_bit/points.las", Truncated),
    ("laz", "las/pdal_simple_laz/simple.laz", Truncated),
    ("pcd", "pcd/crlf/cloud.pcd", Truncated),
    // A point per line, with no count: half of the lines is a whole, smaller cloud.
    ("xyz", "xyz/xyz_commas_tabs_crlf/cloud.xyz", Whole),
    ("pts", "xyz/pts_leica/cloud.pts", Whole),
    ("asc", "xyz/asc_points/cloud.asc", Whole),
    ("ptx", "xyz/ptx_colors/scan.ptx", Partial),
    ("e57", "e57/cartesian_float_rgb/tinyCartesianFloatRgb.e57", Truncated),
    ("amf", "amf/assimp_constellation/test8.amf", Truncated),
    ("lwo", "lwo/assimp_lwo2_boxuv/LWO_LWO2__boxuv.lwo", Truncated),
    ("ac", "ac/assimp_closed_line/closedLine.ac", Truncated),
    ("pmx", "mmd/pmx21_deforms_colors/pmx21_deforms_colors.pmx", Truncated),
    ("pmd", "mmd/pmd_model/pmd_model.pmd", Partial),
    ("vmd", "mmd/vmd_bezier_keys/vmd_bezier_keys.vmd", Truncated),
    ("x", "x/animation/anim.x", Truncated),
    ("md2", "md2/loop_back_to_first_frame/loop.md2", Truncated),
    ("md3", "md3/shader_without_extension/wall.md3", Truncated),
    ("md5mesh", "md5/ref_assimp_simple_cube/SimpleCube.md5mesh", Truncated),
    ("md5anim", "md5/anim_alone/wave.md5anim", Truncated),
    ("iqm", "iqm/skinned_animated/rig.iqm", Truncated),
    // SMD stops at the line the cut left unfinished and calls it corrupt.
    (
        "smd",
        "smd/animation_only/animation_only.smd",
        Gap(Err(ErrorKind::Corrupt)),
    ),
    (
        "vta",
        "smd/vertex_animation/vertex_animation.vta",
        Gap(Err(ErrorKind::Corrupt)),
    ),
    (
        "ms3d",
        "ms3d/comments_then_weights/comments_then_weights.ms3d",
        Truncated,
    ),
    ("b3d", "b3d/brushes_and_colors/brushes_and_colors.b3d", Truncated),
    ("asf", "asf/bom_crlf_comments/crlf.asf", Truncated),
    // A motion is frames one after another with no count: half of them is a shorter motion.
    ("amc", "asf/bad_motion_lines/bad.amc", Whole),
    ("c3d", "c3d/analog_and_events/analog_and_events.c3d", Partial),
    ("trc", "trc/basic/basic.trc", Partial),
    ("htr", "htr/basic/basic.htr", Partial),
    ("mesh", "ogre/element_types/types.mesh", Truncated),
    // A binary skeleton is chunks with no count of them: the reader keeps the whole ones and stops.
    ("skeleton", "ogre/skeleton_only/walk.skeleton", Gap(Ok(false))),
    ("xml", "ogre/twin_xml/twin.mesh.xml", Truncated),
    ("flt", "flt/animated_group/anim.flt", Truncated),
    ("ogex", "ogex/assimp_camera/camera.ogex", Truncated),
    ("irrmesh", "irr/mesh_2tcoords_lightmap/box.irrmesh", Truncated),
    ("irr", "irr/scene_cube/cube.irr", Truncated),
    ("xgl", "xgl/assimp_sample_official/sample_official.xgl", Truncated),
    ("zgl", "xgl/assimp_cubes_with_alpha/cubes_with_alpha.zgl", Truncated),
    ("vtk", "vtk/cell_data_legacy/block.vtk", Truncated),
    ("vtp", "vtk/real_vtk_cow/cow.vtp", Truncated),
    ("vtu", "vtk/cell_data_vtu/block.vtu", Truncated),
    ("b3dm", "tiles/b3dm_binary_globals/wedge.b3dm", Truncated),
    (
        "i3dm",
        "tiles/cesium_instanced_animated/instancedAnimated.i3dm",
        Truncated,
    ),
    ("pnts", "tiles/cesium_points_batched/pointCloudBatched.pnts", Truncated),
    ("cmpt", "tiles/cesium_composite/composite.cmpt", Truncated),
    ("msh", "msh/cube_v21_ascii/cube.msh", Truncated),
    ("msh", "msh/cube_all_v22_binary/cube.msh", Truncated),
    ("mesh", "medit/cube_meshio_text/cube.mesh", Truncated),
    ("meshb", "medit/cube_binary_v2/cube.meshb", Truncated),
];

/// Formats where a line of text reads as a file of the format: none. Even `.splat`, which has no
/// signature, is only ever whole 32-byte records, and 61 bytes of text aren't.
const TEXT_READS: &[&str] = &[];

fn corpus() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

/// What reading `bytes` under `ext` gives: the error's kind, or whether the model is flagged
/// as truncated.
fn read(bytes: &[u8], ext: &str, dir: &Path) -> Result<bool, ErrorKind> {
    match Loader::new()
        .assets(Folder::new(dir))
        .read(bytes.to_vec(), &format!("model.{ext}"))
    {
        Ok(doc) => Ok(doc
            .json()
            .get("asset")
            .and_then(|a| a.pointer("/extras/source/truncated"))
            == Some(&serde_json::Value::Bool(true))),
        Err(e) => Err(e.kind()),
    }
}

#[test]
fn every_extension_has_a_case() {
    // `Format`'s extensions, which `formats::EXTENSIONS` lists: every one is in the table.
    for format in polysmelt::Format::all() {
        for ext in format.extensions() {
            assert!(CASES.iter().any(|c| c.0 == *ext), "no case for .{ext}");
        }
    }
}

#[test]
fn an_empty_file_is_empty_whatever_the_extension() {
    let dir = corpus();
    let mut wrong = Vec::new();
    for &(ext, _, _) in CASES {
        let got = read(b"", ext, &dir);
        if got != Err(ErrorKind::Empty) {
            wrong.push(format!(".{ext}: {got:?}"));
        }
    }
    // With no extension to go by, too.
    let got = Loader::new()
        .read(Vec::new(), "model")
        .map(|_| ())
        .map_err(|e| e.kind());
    assert_eq!(got, Err(ErrorKind::Empty));
    assert!(wrong.is_empty(), "an empty file isn't Empty:\n{}", wrong.join("\n"));
}

#[test]
fn plain_text_is_not_the_format() {
    let dir = corpus();
    let text = b"hello this is not a model file at all, just some words 1 2 3\n";
    let mut wrong = Vec::new();
    for &(ext, file, _) in CASES {
        // A motion finds its skeleton beside it, as the real file does.
        let dir = if ext == "amc" {
            dir.join(file).parent().unwrap().to_path_buf()
        } else {
            dir.clone()
        };
        let name = if ext == "amc" {
            Path::new(file).file_stem().unwrap().to_str().unwrap()
        } else {
            "model"
        };
        let got = Loader::new()
            .assets(Folder::new(&dir))
            .read(text.to_vec(), &format!("{name}.{ext}"))
            .map(|_| ())
            .map_err(|e| e.kind());
        let want = if TEXT_READS.contains(&ext) {
            Ok(())
        } else {
            Err(ErrorKind::UnknownFormat)
        };
        if got != want {
            wrong.push(format!(".{ext}: {got:?}, expected {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "plain text isn't UnknownFormat:\n{}",
        wrong.join("\n")
    );
}

#[test]
fn half_a_file_is_truncated_or_says_it_was_cut() {
    let root = corpus();
    let mut wrong = Vec::new();
    for &(_, file, want) in CASES {
        let path = root.join(file);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{file}: {e}"));
        let dir = path.parent().unwrap();
        // The whole file reads, unflagged: the case is a valid file.
        let name = path.file_name().unwrap().to_str().unwrap();
        let whole = Loader::new().assets(Folder::new(dir)).read(bytes.clone(), name);
        match whole {
            Ok(doc) => {
                let flagged = doc.json()["asset"].pointer("/extras/source/truncated").is_some();
                if flagged {
                    wrong.push(format!("{file}: the whole file is flagged as truncated"));
                }
            }
            Err(e) => wrong.push(format!("{file}: the whole file doesn't read: {e}")),
        }
        let got = Loader::new()
            .assets(Folder::new(dir))
            .read(bytes[..bytes.len() / 2].to_vec(), name)
            .map(|doc| doc.json()["asset"].pointer("/extras/source/truncated") == Some(&serde_json::Value::Bool(true)))
            .map_err(|e| e.kind());
        let ok = match want {
            Truncated => got == Err(ErrorKind::Truncated),
            Partial => got == Ok(true),
            Whole => got == Ok(false),
            Gap(now) => got == now,
        };
        if !ok {
            wrong.push(format!("{file}: half of it gives {got:?}, expected {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "half a file:\n{}", wrong.join("\n"));
}

#[test]
fn a_well_formed_file_with_nothing_in_it_is_empty() {
    let dir = corpus();
    let dae = br#"<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><asset/></COLLADA>"#;
    let cases: &[(&str, &[u8])] = &[
        ("obj", b"# only a comment\n"),
        ("stl", b"solid nothing\nendsolid nothing\n"),
        ("ply", b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n"),
        ("off", b"OFF\n0 0 0\n"),
        ("gltf", br#"{"asset":{"version":"2.0"}}"#),
        ("gltf", br#"{"asset":{"version":"2.0"},"materials":[{"name":"only a material"}]}"#),
        ("dae", dae),
        ("usda", b"#usda 1.0\n"),
        ("x3d", br#"<X3D version="3.3"><Scene/></X3D>"#),
        ("wrl", b"#VRML V2.0 utf8\n"),
        ("dxf", b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"),
        ("xyz", b"# no points\n"),
        ("amf", br#"<?xml version="1.0"?><amf unit="millimeter"></amf>"#),
        ("xgl", b"<WORLD></WORLD>"),
        ("irrmesh", br#"<?xml version="1.0"?><mesh xmlns="http://irrlicht.sourceforge.net/IRRMESH_09_2007" version="1.0"></mesh>"#),
        ("vtk", b"# vtk DataFile Version 3.0\nnothing\nASCII\nDATASET POLYDATA\nPOINTS 0 float\n"),
        ("pcd", b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 0\nHEIGHT 1\nPOINTS 0\nDATA ascii\n"),
    ];
    let mut wrong = Vec::new();
    for (ext, bytes) in cases {
        let got = read(bytes, ext, &dir);
        if got != Err(ErrorKind::Empty) {
            wrong.push(format!(".{ext}: {got:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "nothing to convert isn't Empty:\n{}",
        wrong.join("\n")
    );
}
