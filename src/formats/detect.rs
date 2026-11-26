//! Telling a model's format from its first bytes, for files whose extension is missing, wrong
//! or shared by several formats (`.xml`).

/// How many bytes from a model's start [`sniff`] looks at.
pub(crate) const HEAD: usize = 4096;

/// The format of the model whose first bytes are `head` (of `len` bytes in all, when known),
/// named by the extension it's read under; `None` when nothing in them
/// says.
pub(crate) fn sniff(head: &[u8], len: Option<u64>) -> Option<&'static str> {
    binary(head, len).or_else(|| text(&narrow(head)))
}

/// The XML formats among those [`sniff`] names, which `.xml` files are read as.
pub(crate) fn xml(head: &[u8]) -> Option<&'static str> {
    root(&narrow(head))
}

/// Formats with a binary signature.
fn binary(head: &[u8], len: Option<u64>) -> Option<&'static str> {
    let starts = |magic: &[u8]| head.starts_with(magic);
    let at = |i: usize, magic: &[u8]| head.get(i..i + magic.len()) == Some(magic);
    let found = if starts(b"glTF") {
        if contains(head, b"VRMC_vrm_animation") {
            "vrma"
        } else if contains(head, b"\"VRM\"") || contains(head, b"\"VRMC_vrm\"") {
            "vrm"
        } else {
            "glb"
        }
    } else if starts(b"b3dm") {
        "b3dm"
    } else if starts(b"i3dm") {
        "i3dm"
    } else if starts(b"pnts") {
        "pnts"
    } else if starts(b"cmpt") {
        "cmpt"
    } else if starts(b"Kaydara FBX Binary") {
        "fbx"
    } else if starts(b"PXR-USDC") {
        "usdc"
    } else if starts(b"PK\x03\x04") {
        return zip(head);
    } else if starts(b"VOX ") {
        "vox"
    } else if starts(b"IDP2") {
        "md2"
    } else if starts(b"IDP3") {
        "md3"
    } else if starts(b"INTERQUAKEMODEL\0") {
        "iqm"
    } else if starts(b"MS3D000000") {
        "ms3d"
    } else if starts(b"BB3D") {
        "b3d"
    } else if starts(b"Pmx ") {
        "pmx"
    } else if starts(b"Pmd") {
        "pmd"
    } else if starts(b"Vocaloid Motion Data") {
        "vmd"
    } else if starts(b"LASF") {
        "las"
    } else if starts(b"ASTM-E57") {
        "e57"
    } else if starts(b"AutoCAD Binary DXF") {
        "dxf"
    } else if starts(b"FORM") && (at(8, b"LWO2") || at(8, b"LWOB") || at(8, b"LWLO") || at(8, b"LWO3")) {
        "lwo"
    } else if starts(b"\xFF\xFE\xFF\x0E") && contains(head, &utf16(b"SketchUp Model")) {
        "skp"
    } else if starts(b"MM") && (at(6, &[0x02, 0x00]) || at(6, &[0x3D, 0x3D])) {
        "3ds"
    } else if contains(&head[..head.len().min(64)], b"[MeshSerializer_") {
        "mesh"
    } else if contains(&head[..head.len().min(64)], b"[Serializer_") {
        "skeleton"
    } else if openflight(head) {
        "flt"
    } else if zgl(head) {
        "zgl"
    } else if c3d(head) {
        "c3d"
    } else if stl(head, len) {
        "stl"
    } else {
        return None;
    };
    Some(found)
}

/// A zip archive's format, from the names of its first entries: 3MF's content types and model
/// part, a USDZ's layer, a KMZ's `doc.kml`, a ZAE's manifest, a SOG's `meta.json` or the images
/// of positions that come before it.
fn zip(head: &[u8]) -> Option<&'static str> {
    let mut at = 0;
    let mut names = Vec::new();
    while head.get(at..at + 4) == Some(b"PK\x03\x04") && names.len() < 4 {
        let u16_at = |i: usize| {
            head.get(at + i..at + i + 2)
                .map(|b| usize::from(u16::from_le_bytes([b[0], b[1]])))
        };
        let u32_at = |i: usize| {
            head.get(at + i..at + i + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        };
        let (Some(name_len), Some(extra), Some(size)) = (u16_at(26), u16_at(28), u32_at(18)) else {
            break;
        };
        let Some(name) = head.get(at + 30..at + 30 + name_len) else {
            break;
        };
        names.push(String::from_utf8_lossy(name).to_ascii_lowercase());
        at += 30 + name_len + extra + size;
    }
    names.iter().find_map(|n| {
        let ends = |e: &str| n.ends_with(e);
        Some(
            if n == "[content_types].xml" || n.starts_with("3d/") || n == "_rels/.rels" {
                "3mf"
            } else if ends(".usdc") || ends(".usda") || ends(".usd") {
                "usdz"
            } else if ends(".kml") {
                "kmz"
            } else if n == "manifest.xml" || ends(".dae") {
                "zae"
            } else if ends(".x3d") {
                "x3dz"
            } else if ends(".amf") {
                "amf"
            } else if n == "meta.json" || n.starts_with("means_") && ends(".webp") {
                "sog"
            } else {
                return None;
            },
        )
    })
}

/// Whether `head` starts with an `OpenFlight` header record: opcode 1, a record long enough to
/// be one, and a format revision it has had.
fn openflight(head: &[u8]) -> bool {
    let be16 = |i: usize| head.get(i..i + 2).map(|b| u16::from_be_bytes([b[0], b[1]]));
    let be32 = |i: usize| head.get(i..i + 4).map(|b| i32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    be16(0) == Some(1)
        && be16(2).is_some_and(|n| n >= 128)
        && be32(12).is_some_and(|r| (11..=16).contains(&r) || (1400..=1700).contains(&r))
}

/// Whether `head` starts like a C3D file: the parameter section's block number, the C3D key,
/// and a parameter section naming a processor type.
fn c3d(head: &[u8]) -> bool {
    let (Some(&block), Some(&0x50)) = (head.first(), head.get(1)) else {
        return false;
    };
    block >= 2
        && head
            .get((usize::from(block) - 1) * 512 + 3)
            .is_some_and(|p| (0x54..=0x56).contains(p))
}

/// Whether the model is a binary STL: an 80-byte header, a triangle count, and exactly that
/// many 50-byte triangles. Without the whole length it can't be told.
fn stl(head: &[u8], len: Option<u64>) -> bool {
    let (Some(len), Some(n)) = (len, head.get(80..84)) else {
        return false;
    };
    let n = u64::from(u32::from_le_bytes([n[0], n[1], n[2], n[3]]));
    n > 0 && len == 84 + 50 * n
}

/// Formats written as text, from their first words.
fn text(text: &str) -> Option<&'static str> {
    let t = text.trim_start();
    let starts = |s: &str| t.starts_with(s);
    let found = if starts("<") {
        return root(t);
    } else if starts("#usda") {
        "usda"
    } else if starts("; FBX") {
        "fbx"
    } else if t
        .get(..22)
        .is_some_and(|s| s.eq_ignore_ascii_case("# vtk DataFile Version"))
    {
        "vtk"
    } else if starts("#VRML") {
        "wrl"
    } else if starts("#X3D") {
        "x3dv"
    } else if starts("ply") && t[3..].starts_with(['\n', '\r']) {
        "ply"
    } else if starts("MD5Version") {
        if t.contains("numFrames") { "md5anim" } else { "md5mesh" }
    } else if starts("xof ") {
        "x"
    } else if starts("AC3D") {
        "ac"
    } else if starts("HIERARCHY") {
        "bvh"
    } else if starts("PathFileType") {
        "trc"
    } else if starts("# .PCD") || starts("VERSION .7") || starts("VERSION 0.7") {
        "pcd"
    } else if starts("solid") && t.contains("facet") {
        "stl"
    } else if ["OFF", "COFF", "NOFF", "CNOFF", "STOFF", "4OFF", "nOFF"]
        .iter()
        .any(|m| t.starts_with(m) && t[m.len()..].starts_with(|c: char| c.is_ascii_whitespace()))
    {
        "off"
    } else if t.starts_with('0') && t.get(..64).unwrap_or(t).contains("SECTION") {
        "dxf"
    } else if starts("version 1") {
        if t.contains("vertexanimation") { "vta" } else { "smd" }
    } else if t.contains("[Header]") && t.contains("htr") {
        "htr"
    } else if starts(":FULLY-SPECIFIED") || starts(":DEGREES") {
        "amc"
    } else if starts(":version") || t.contains("\n:root") {
        "asf"
    } else if starts("{") && ["\"means\"", "\"mins\"", "\"maxs\""].iter().all(|k| t.contains(k)) {
        // A SOG's `meta.json`, which may name its generator under `asset` as glTF does.
        "sog"
    } else if starts("{")
        && ["\"geometricError\"", "\"boundingVolume\"", "\"tilesetVersion\""]
            .iter()
            .any(|k| t.contains(k))
    {
        // A 3D Tiles tileset, which has an `asset` as glTF does. The tile reader goes by the
        // file's first bytes, not the extension, so any of its extensions reads it.
        "cmpt"
    } else if starts("{") && t.contains("\"asset\"") {
        "gltf"
    } else if t.contains("Metric") && (t.contains("GeometryNode") || t.contains("GeometryObject")) {
        "ogex"
    } else if obj(t) {
        "obj"
    } else {
        return None;
    };
    Some(found)
}

/// Whether the text's first statement (comments aside) is one only OBJ starts with.
fn obj(t: &str) -> bool {
    t.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .is_some_and(|l| {
            let word = l.split_ascii_whitespace().next().unwrap_or("");
            matches!(word, "v" | "vn" | "vt" | "mtllib" | "o" | "g" | "usemtl" | "f" | "s")
                && (word != "v" || l.split_ascii_whitespace().count() >= 4)
        })
}

/// The format named by an XML document's root element.
fn root(t: &str) -> Option<&'static str> {
    let mut rest = t.trim_start_matches('\u{feff}').trim_start();
    // The XML declaration, processing instructions, comments and a doctype come before the root.
    loop {
        if let Some(r) = rest.strip_prefix("<?") {
            rest = r.split_once("?>")?.1.trim_start();
        } else if let Some(r) = rest.strip_prefix("<!--") {
            rest = r.split_once("-->")?.1.trim_start();
        } else if let Some(r) = rest.strip_prefix("<!") {
            rest = r.split_once('>')?.1.trim_start();
        } else {
            break;
        }
    }
    let after = rest.strip_prefix('<')?;
    let end = after
        .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
        .unwrap_or(after.len());
    let name = &after[..end];
    let name = name.rsplit(':').next().unwrap_or(name);
    Some(match name {
        "COLLADA" => "dae",
        "X3D" => "x3d",
        "amf" => "amf",
        "mesh" if irrmesh(&after[end..]) => "irrmesh",
        "mesh" => "mesh",
        "skeleton" => "skeleton",
        "irr_scene" => "irr",
        n if n.eq_ignore_ascii_case("world") => "xgl",
        // Poly data or a grid: the reader goes by the root's type, not the extension.
        "VTKFile" => "vtu",
        _ => return None,
    })
}

/// Whether a `<mesh>` element, from just past its name, is an Irrlicht mesh rather than an OGRE one: its namespace
/// says so, or its first child is a `<buffer>`.
fn irrmesh(t: &str) -> bool {
    let tag = &t[..t.find('>').unwrap_or(t.len())];
    if tag.contains("irrlicht") || tag.contains("IRRMESH") {
        return true;
    }
    let body = t[tag.len()..].trim_start_matches('>');
    let mut rest = body.trim_start();
    while let Some(r) = rest.strip_prefix("<!--") {
        let Some((_, r)) = r.split_once("-->") else {
            return false;
        };
        rest = r.trim_start();
    }
    rest.starts_with("<buffer")
}

/// Whether `head` starts a zlib stream (RFC 1950) that inflates to an XGL document.
fn zgl(head: &[u8]) -> bool {
    let [cmf, flg, ..] = *head else {
        return false;
    };
    if cmf & 0x0F != 8 || cmf >> 4 > 7 || (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 || flg & 0x20 != 0 {
        return false;
    }
    #[cfg(feature = "xgl")]
    {
        let mut out = Vec::new();
        // The head is cut short: what inflates before the cut is enough to see the root.
        let _ = super::inflate::inflate(&head[2..], &mut out, 1024);
        root(&narrow(&out)) == Some("xgl")
    }
    #[cfg(not(feature = "xgl"))]
    false
}

/// The head as text: UTF-16 (told as XML tells it, by a byte order mark or the zero bytes of its first `<`)
/// narrowed to its ASCII, anything else read as UTF-8, lossily.
fn narrow(head: &[u8]) -> String {
    let Some((le, bom)) = super::xml::utf16(head) else {
        return String::from_utf8_lossy(head).into_owned();
    };
    let first = usize::from(!le);
    head[bom..]
        .chunks_exact(2)
        .map(|c| {
            if c[1 - first] == 0 && c[first].is_ascii() {
                char::from(c[first])
            } else {
                '?'
            }
        })
        .collect()
}

/// `ascii` in UTF-16LE.
fn utf16(ascii: &[u8]) -> Vec<u8> {
    ascii.iter().flat_map(|&b| [b, 0]).collect()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "xgl")]
    fn zipped_xgl_is_told_by_what_it_inflates_to() {
        let zlib = |text: &[u8]| {
            let n = u16::try_from(text.len()).unwrap();
            let mut z = vec![0x78, 0x01, 0x01];
            z.extend(n.to_le_bytes());
            z.extend((!n).to_le_bytes());
            z.extend(text);
            z
        };
        assert_eq!(
            sniff(
                &zlib(
                    b"<WORLD>
<MESH ID=\"0\">"
                ),
                None
            ),
            Some("zgl")
        );
        assert_eq!(sniff(&zlib(b"<COLLADA>")[..9], None), None, "cut short, and not XGL");
        assert_eq!(sniff(&zlib(b"<scene>"), None), None);
    }

    #[test]
    fn binary_signatures_name_their_formats() {
        assert_eq!(sniff(b"glTF\x02\0\0\0", None), Some("glb"));
        assert_eq!(
            sniff(b"glTF\x02\0\0\0....{\"extensionsUsed\":[\"VRMC_vrm\"]", None),
            Some("vrm")
        );
        assert_eq!(sniff(b"Kaydara FBX Binary  \0", None), Some("fbx"));
        assert_eq!(sniff(b"PXR-USDC", None), Some("usdc"));
        assert_eq!(sniff(b"IDP2\x08\0\0\0", None), Some("md2"));
        assert_eq!(sniff(b"INTERQUAKEMODEL\0", None), Some("iqm"));
        assert_eq!(sniff(b"FORM\0\0\0\x10LWO2", None), Some("lwo"));
        assert_eq!(sniff(b"MM\x10\0\0\0\x02\0", None), Some("3ds"));
        let mut stl = vec![0u8; 84 + 50];
        stl[80] = 1;
        assert_eq!(sniff(&stl, Some(134)), Some("stl"));
        assert_eq!(sniff(&stl, Some(135)), None, "a length that isn't whole triangles");
        assert_eq!(sniff(&stl, None), None, "unknowable without the length");
    }

    #[test]
    fn zip_archives_are_told_apart_by_their_entries() {
        let entry = |name: &str| {
            let mut e = b"PK\x03\x04".to_vec();
            e.extend([0; 14]);
            e.extend(0u32.to_le_bytes()); // compressed size
            e.extend(0u32.to_le_bytes());
            e.extend(u16::try_from(name.len()).unwrap().to_le_bytes());
            e.extend(0u16.to_le_bytes());
            e.extend(name.as_bytes());
            e
        };
        assert_eq!(sniff(&entry("[Content_Types].xml"), None), Some("3mf"));
        assert_eq!(sniff(&entry("meta.json"), None), Some("sog"));
        assert_eq!(sniff(&entry("means_l.webp"), None), Some("sog"));
        assert_eq!(sniff(&entry("Model.usdc"), None), Some("usdz"));
        assert_eq!(sniff(&entry("doc.kml"), None), Some("kmz"));
        let mut two = entry("readme.txt");
        two.extend(entry("scene.dae"));
        assert_eq!(sniff(&two, None), Some("zae"), "a later entry decides");
        assert_eq!(sniff(&entry("readme.txt"), None), None);
    }

    #[test]
    fn text_formats_are_known_by_their_first_words() {
        for (head, ext) in [
            ("#usda 1.0\n", "usda"),
            ("ply\nformat ascii 1.0\n", "ply"),
            ("solid cube\n facet normal 0 0 1\n", "stl"),
            ("OFF\n8 6 0\n", "off"),
            ("  0\nSECTION\n  2\nHEADER\n", "dxf"),
            ("HIERARCHY\nROOT Hips\n", "bvh"),
            ("MD5Version 10\ncommandline \"\"\nnumJoints 3\nnumMeshes 1\n", "md5mesh"),
            ("MD5Version 10\ncommandline \"\"\nnumFrames 3\n", "md5anim"),
            ("version 1\nnodes\n", "smd"),
            ("xof 0302txt 0032\n", "x"),
            ("{\n  \"asset\": {\"version\": \"2.0\"}\n}", "gltf"),
            (
                "{\"version\":2,\"asset\":{\"generator\":\"x\"},\"count\":1,\"means\":{\"mins\":[0,0,0],\"maxs\":[1,1,1]",
                "sog",
            ),
            ("# made by hand\nmtllib a.mtl\nv 0 0 0\n", "obj"),
            ("#VRML V2.0 utf8\n", "wrl"),
        ] {
            assert_eq!(sniff(head.as_bytes(), None), Some(ext), "{head:?}");
        }
        assert_eq!(
            sniff(b"1.0 2.0 3.0\n4 5 6\n", None),
            None,
            "bare numbers could be any point cloud"
        );
    }

    #[test]
    fn xml_is_read_by_its_root_element() {
        let collada = "<?xml version=\"1.0\"?>\n<!-- exported -->\n<COLLADA xmlns=\"x\" version=\"1.4.1\">";
        assert_eq!(xml(collada.as_bytes()), Some("dae"));
        assert_eq!(xml(b"<mesh><submeshes>"), Some("mesh"));
        assert_eq!(
            xml(b"<?xml version='1.0'?><!DOCTYPE X3D><X3D profile='Immersive'>"),
            Some("x3d")
        );
        assert_eq!(xml(b"<amf unit=\"millimeter\">"), Some("amf"));
        assert_eq!(xml(b"<irr_scene>"), Some("irr"));
        assert_eq!(
            xml(b"<WORLD>
<MESH ID=\"0\">"),
            Some("xgl")
        );
        assert_eq!(xml(b"<world>"), Some("xgl"));
        assert_eq!(
            xml(b"<?xml version=\"1.0\"?>
<mesh xmlns=\"http://irrlicht.sourceforge.net/IRRMESH_09_2007\" version=\"1.0\">"),
            Some("irrmesh")
        );
        assert_eq!(
            xml(b"<mesh>
  <!-- made by hand -->
  <buffer>"),
            Some("irrmesh")
        );
        assert_eq!(
            xml(b"<mesh>
  <sharedgeometry>"),
            Some("mesh")
        );
        let mut bare = Vec::new();
        bare.extend(utf16(b"<?xml version=\"1.0\"?><irr_scene>"));
        assert_eq!(xml(&bare), Some("irr"), "UTF-16 without a byte order mark");
        let be: Vec<u8> = b"<?xml version=\"1.0\"?><WORLD>".iter().flat_map(|&b| [0, b]).collect();
        assert_eq!(xml(&be), Some("xgl"), "big-endian UTF-16 without a byte order mark");
        let mut wide = vec![0xFF, 0xFE];
        wide.extend(utf16(b"<?xml version=\"1.0\" encoding=\"UTF-16\"?><COLLADA>"));
        assert_eq!(xml(&wide), Some("dae"), "UTF-16 with a byte order mark");
    }
}
