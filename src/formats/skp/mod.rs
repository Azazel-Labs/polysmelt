//! `SketchUp` `.skp`, read without the `SketchUp` SDK and converted to glTF.
//!
//! Ported from openskp (<https://github.com/iamahsanmehmood/openskp>, MIT, see `LICENSE-openskp`
//! beside this file): its C++ read path, face triangulation and glTF building, without its writer
//! or other exporters.
//!
//! - Files from `SketchUp` 3 to 2020 are an MFC object stream ([`legacy`]); 2021 and later are a
//!   ZIP holding tagged records ([`vff`]).
//! - Component instances become glTF nodes with their transforms; each definition's faces become
//!   one mesh per inherited material, shared by every placement.
//! - Hidden entities and entities on hidden tags (layers) are kept as nodes marked invisible with
//!   `KHR_node_visibility`, when the triangle budget leaves room for them.
//! - Normals are averaged across smooth edges and split at hard ones. A face painted differently on
//!   its back gets single-sided front and back faces; otherwise it is one double-sided face.
//! - BMP and TGA textures are re-encoded as PNG; GIF and TIFF textures are dropped. A texture with
//!   an alpha channel is cut out at a half, as `SketchUp` draws it.
//! - Unpainted faces take the default material every reader gives faces with none.
//! - The model stays in its own inches and Z up: the document's `RootNode`, over the model's node, turns it to
//!   glTF's Y up and scales it to meters.
//!
//! The asset's `extras.source` records the header's version, `SketchUp` and its version as the
//! tool, and the inches and Z up the file is in, both converted.

mod glb;
mod image;
mod legacy;
mod model;
mod reader;
mod vff;
mod zip;

use serde_json::{Map, json};

use super::Input;
use super::frame::{self, Frame};
use crate::assets::Memory;
use crate::document::{Document, Keep};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Result, Within as _};

/// Reads the `SketchUp` model `input`, with what it holds in a glTF document.
pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document> {
    let bytes = input.read()?;
    let (glb, major) = convert(&bytes, input.limit)?;
    // An empty model, or one of only edges, guides and text, has nothing to draw.
    let mut doc = match glb {
        Some(glb) => {
            let glb = crate::Bytes::from(glb);
            super::gltf::load(Input::shared(&glb, input.path), &Memory::new(), keep)?
        }
        None => Document::empty(Map::new()),
    };
    let mut source = Map::new();
    source_info(&bytes, major).write(&mut source);
    doc.base_mut()
        .entry("asset")
        .or_insert_with(|| json!({ "version": "2.0" }))["extras"] = json!({ "source": source });
    // The model is kept in its own inches and Z up: the root node turns it Y up and scales it to meters.
    frame::root(&mut doc, Frame::z_up().scaled(glb::INCH));
    Ok(doc)
}

/// What the file says of itself: only its header's version, and the inches and Z up every
/// `SketchUp` model is in.
fn source_info(bytes: &[u8], major: u32) -> SourceInfo {
    SourceInfo {
        format: Some("SketchUp".into()),
        version: header_version(bytes),
        meters_per_unit: Fact::Declared(0.0254),
        unit: Some("inch".into()),
        up_axis: Fact::Declared(Axis::PosZ),
        handedness: Fact::Declared(Handedness::Right),
        tool: Some(format!("SketchUp {}", version_name(major))),
        ..SourceInfo::default()
    }
}

/// Every `SketchUp` file opens with a string record (`FF FE FF`) holding the 14 characters `SketchUp Model`.
const MAGIC: [u8; 4] = [0xff, 0xfe, 0xff, 0x0e];
/// The last major version written as an MFC object stream; 2021 (21) moved to the ZIP container.
const LAST_LEGACY: u32 = 20;

/// Whether `bytes` look like a `SketchUp` file, from its header alone.
fn is_skp(bytes: &[u8]) -> bool {
    bytes.starts_with(&MAGIC)
}

/// Converts a `SketchUp` file to a GLB file in meters with Y up (`None` when nothing in it is
/// visible), with the file's major version; `limit` is the most triangles it may draw.
///
/// # Errors
/// When the bytes are not a `SketchUp` file, come from a version this reader doesn't know, or
/// are damaged.
fn convert(bytes: &[u8], limit: usize) -> Result<(Option<Vec<u8>>, u32)> {
    if !is_skp(bytes) {
        bail!(UnknownFormat, "not a SketchUp file");
    }
    let Some(major) = major_version(bytes) else {
        bail!(Corrupt, "SketchUp file has no version in its header")
    };
    let reading = || format!("reading a SketchUp {} file", version_name(major));
    let model = if major <= LAST_LEGACY {
        if major < 3 {
            bail!(
                Unsupported,
                "SketchUp {major} files are not supported (SketchUp 3 is the oldest this reader knows)"
            );
        }
        legacy::parse(bytes, major).with_within(reading)?
    } else {
        vff::parse(bytes).with_within(reading)?
    };
    Ok((glb::write(&model, limit)?, major))
}

/// `SketchUp`'s marketing name for a major version: 3 to 8 kept the number, 13 to 2020 became years.
fn version_name(major: u32) -> String {
    if major >= 13 {
        (2000 + major).to_string()
    } else {
        major.to_string()
    }
}

/// The header's `{major.minor.build}` string, which follows `SketchUp Model` in UTF-16, without
/// its braces.
fn header_version(bytes: &[u8]) -> Option<String> {
    let ascii: Vec<u8> = bytes.iter().take(96).copied().filter(|&b| b != 0).collect();
    let open = ascii.iter().position(|&b| b == b'{')?;
    let inside: Vec<u8> = ascii[open + 1..]
        .iter()
        .copied()
        .take_while(|&b| b.is_ascii_digit() || b == b'.')
        .collect();
    (ascii.get(open + 1 + inside.len()) == Some(&b'}')).then(|| crate::bytes::text(&inside).into_owned())
}

/// The major version from the header's version string.
fn major_version(bytes: &[u8]) -> Option<u32> {
    let version = header_version(bytes)?;
    let (major, _) = version.split_once('.')?;
    major.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MAX_TRIANGLES;

    fn header(version: &str) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend("SketchUp Model".encode_utf16().flat_map(u16::to_le_bytes));
        out.extend([0xff, 0xfe, 0xff, 12]);
        out.extend(version.encode_utf16().flat_map(u16::to_le_bytes));
        out
    }

    #[test]
    fn reads_the_header_version() {
        assert!(is_skp(&header("{15.0.9350}")));
        assert!(!is_skp(b"PK\x03\x04"));
        assert_eq!(major_version(&header("{15.0.9350}")), Some(15));
        assert_eq!(major_version(&header("{23.1.340}")), Some(23));
        assert_eq!(major_version(&header("{x}")), None);
        assert_eq!(header_version(&header("{23.1.340}")).as_deref(), Some("23.1.340"));
    }

    #[test]
    fn the_source_is_in_inches_and_z_up() {
        let info = source_info(&header("{23.1.340}"), 23);
        assert_eq!(info.format.as_deref(), Some("SketchUp"));
        assert_eq!(info.version.as_deref(), Some("23.1.340"));
        assert_eq!(info.tool.as_deref(), Some("SketchUp 2023"));
        assert_eq!(info.meters_per_unit, Fact::Declared(0.0254));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
        // What is converted is said when the root node is made, on a load.
        assert!(!info.converted.units && !info.converted.up_axis);
    }

    #[test]
    fn rejects_what_it_cannot_read() {
        assert!(convert(b"hello", MAX_TRIANGLES).is_err());
        assert!(convert(&header("{2.0.1}"), MAX_TRIANGLES).is_err());
        assert!(convert(&header("{15.0.9350}"), MAX_TRIANGLES).is_err());
        assert!(convert(&header("{23.0.1}"), MAX_TRIANGLES).is_err());
        assert!(convert(&MAGIC, MAX_TRIANGLES).is_err());
    }

    #[test]
    fn truncated_files_never_panic() {
        // A crate built from crates.io has no corpus to read.
        let Ok(data) = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/corpus/skp/legacy_v5_textured/legacy_v5_textured.skp"
        )) else {
            return;
        };
        let vff = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/corpus/skp/nesting_2021/nesting_2021.skp"
        ))
        .unwrap();
        for len in (0..data.len()).step_by(7) {
            let _ = convert(&data[..len], MAX_TRIANGLES);
        }
        for len in (0..vff.len()).step_by(7) {
            let _ = convert(&vff[..len], MAX_TRIANGLES);
        }
        // Tags with line styles, in the 2019 layout.
        let styles = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/corpus/skp/sdk_saved_tag_line_styles_2019/sdk_saved_tag_line_styles_2019.skp"
        ))
        .unwrap();
        assert!(convert(&styles, MAX_TRIANGLES).is_ok());
        for len in (0..styles.len()).step_by(7) {
            let _ = convert(&styles[..len], MAX_TRIANGLES);
        }
    }
}
