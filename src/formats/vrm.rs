//! VRM avatars (`.vrm`) and VRM animations (`.vrma`): GLB files whose extensions (`VRM` for
//! 0.x, `VRMC_vrm`, `VRMC_springBone`, `VRMC_materials_mtoon` and the rest for 1.0,
//! `VRMC_vrm_animation` for animations) describe the humanoid, its expressions and its toon
//! materials. The glTF underneath draws as it is: every VRM material has a metallic-roughness
//! fallback, so the extensions are kept but not needed, and a file that marks one required
//! (which VRM never asks for) is read anyway.
//!
//! A VRM 0.x avatar faces -Z, where glTF's front (and VRM 1.0's) is +Z, so it is turned half
//! about Y under a `RootNode` to face the viewer as other models do.
//!
//! The asset's `extras.source` says which version the file is and whether it was turned, with
//! glTF's frame declared (a 0.x avatar's forward is -Z) and the avatar's meta: its name or
//! title, authors, copyright and license (the license URL in 1.0; in 0.x the license name, or
//! the other license's URL), with its own version, contact and references kept as details. The
//! rest of the glTF is passed through as it is.

use serde_json::{Map, Value, json};

use super::Input;
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::source::{Axis, Fact, Handedness, SourceInfo};

/// Whether an extension is VRM's own.
fn vrm_extension(name: &str) -> bool {
    name == "VRM" || name.starts_with("VRMC_")
}

pub(crate) fn load<'a>(input: Input<'a>, assets: &'a dyn Assets, keep: Keep) -> Result<Document> {
    let mut doc = super::gltf::load(input, assets, keep)?;
    let json = doc.json_mut();
    if let Some(Value::Array(required)) = json.get_mut("extensionsRequired") {
        required.retain(|e| !e.as_str().is_some_and(vrm_extension));
        if required.is_empty() {
            json.remove("extensionsRequired");
        }
    }
    let ext = json.get("extensions");
    let text = |v: Option<&Value>| v.and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_owned);
    let one = ext.and_then(|e| e.get("VRMC_vrm"));
    let zero = ext.and_then(|e| e.get("VRM"));
    let animation = ext.and_then(|e| e.get("VRMC_vrm_animation"));
    let mut info = SourceInfo {
        format: Some(if animation.is_some() { "VRMA" } else { "VRM" }.into()),
        // glTF's frame, which VRM keeps.
        meters_per_unit: Fact::Declared(1.0),
        unit: Some("meter".into()),
        up_axis: Fact::Declared(Axis::PosY),
        forward_axis: Fact::Declared(Axis::PosZ),
        handedness: Fact::Declared(Handedness::Right),
        tool: text(json.get("asset").and_then(|a| a.get("generator"))),
        ..SourceInfo::default()
    };
    let mut source = Map::new();
    let mut detail = |k: &str, v: Option<String>| {
        if let Some(v) = v {
            source.insert(k.into(), json!(v));
        }
    };
    let mut turn = false;
    if let Some(v) = one {
        info.version = Some(text(v.get("specVersion")).unwrap_or_else(|| "1.0".into()));
        let meta = |k: &str| text(v.get("meta").and_then(|m| m.get(k)));
        info.title = meta("name");
        let authors: Vec<String> = v
            .pointer("/meta/authors")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|a| text(Some(a)))
            .collect();
        info.author = (!authors.is_empty()).then(|| authors.join("; "));
        info.copyright = meta("copyrightInformation");
        info.license = meta("licenseUrl");
        detail("modelVersion", meta("version"));
        detail("contactInformation", meta("contactInformation"));
        detail("otherLicenseUrl", meta("otherLicenseUrl"));
        if let Some(r) = v
            .pointer("/meta/references")
            .filter(|r| r.as_array().is_some_and(|r| !r.is_empty()))
        {
            source.insert("references".into(), r.clone());
        }
    } else if let Some(v) = zero {
        info.version = Some(text(v.get("specVersion")).unwrap_or_else(|| "0.0".into()));
        let meta = |k: &str| text(v.get("meta").and_then(|m| m.get(k)));
        info.title = meta("title");
        info.author = meta("author");
        // 0.x names a license, or says "Other" and points at one.
        info.license = match (meta("licenseName"), meta("otherLicenseUrl")) {
            (Some(n), Some(url)) if n == "Other" => Some(url),
            (Some(n), _) => Some(n),
            (None, url) => url,
        };
        // A 0.x avatar faces -Z.
        info.forward_axis = Fact::Declared(Axis::NegZ);
        detail("modelVersion", meta("version"));
        detail("contactInformation", meta("contactInformation"));
        detail("reference", meta("reference"));
        detail(
            "turned",
            Some("VRM 0.x faces -Z: turned half about Y to face +Z".into()),
        );
        turn = true;
    } else if let Some(v) = animation {
        info.version = Some(text(v.get("specVersion")).unwrap_or_else(|| "1.0".into()));
    }
    info.write(&mut source);
    if turn {
        turn_half(json);
    }
    let asset = json.entry("asset").or_insert_with(|| json!({ "version": "2.0" }));
    if !asset["extras"].is_object() {
        asset["extras"] = json!({});
    }
    match asset["extras"].get_mut("source") {
        Some(Value::Object(old)) => old.extend(source),
        _ => asset["extras"]["source"] = Value::Object(source),
    }
    Ok(doc)
}

/// Puts each scene's roots under a `RootNode` turned half about Y.
fn turn_half(json: &mut serde_json::Map<String, Value>) {
    let scenes = json.get("scenes").and_then(Value::as_array).map_or(0, Vec::len);
    for s in 0..scenes {
        let roots = json["scenes"][s]["nodes"].take();
        if roots.as_array().is_none_or(Vec::is_empty) {
            json["scenes"][s]["nodes"] = roots;
            continue;
        }
        let nodes = json.entry("nodes").or_insert_with(|| json!([]));
        let Some(nodes) = nodes.as_array_mut() else {
            json["scenes"][s]["nodes"] = roots;
            continue;
        };
        nodes.push(json!({ "name": "RootNode", "children": roots, "rotation": [0.0, 1.0, 0.0, 0.0] }));
        let root = nodes.len() - 1;
        json["scenes"][s]["nodes"] = json!([root]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use crate::document::Pose;

    /// A GLB of one triangle in front of the origin (+Z when `z` is positive), with `json`'s
    /// keys merged in.
    pub(super) fn glb(z: f32, extra: &Value) -> Vec<u8> {
        let mut bin = Vec::new();
        for v in [0f32, 0., z, 1., 0., z, 0., 1., z] {
            bin.extend(v.to_le_bytes());
        }
        for i in [0u16, 1, 2, 0] {
            bin.extend(i.to_le_bytes());
        }
        let mut j = json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }] }],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0.0, 0.0, z], "max": [1.0, 1.0, z] },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "buffers": [{ "byteLength": 44 }]
        });
        for (k, v) in extra.as_object().unwrap() {
            j[k] = v.clone();
        }
        let mut text = serde_json::to_vec(&j).unwrap();
        while !text.len().is_multiple_of(4) {
            text.push(b' ');
        }
        let mut out = b"glTF".to_vec();
        out.extend(2u32.to_le_bytes());
        out.extend(
            u32::try_from(12 + 8 + text.len() + 8 + bin.len())
                .unwrap()
                .to_le_bytes(),
        );
        out.extend(u32::try_from(text.len()).unwrap().to_le_bytes());
        out.extend(b"JSON");
        out.extend(text);
        out.extend(u32::try_from(bin.len()).unwrap().to_le_bytes());
        out.extend(b"BIN\0");
        out.extend(bin);
        out
    }

    fn read(bytes: &[u8], ext: &str) -> Document {
        super::super::load_bytes(bytes, "a.vrm", ext, &Memory::new(), Keep::ALL).unwrap()
    }

    #[test]
    fn vrm_1_draws_even_with_its_extensions_required() {
        let extra = json!({
            "extensionsUsed": ["VRMC_vrm", "VRMC_springBone"],
            "extensionsRequired": ["VRMC_vrm"],
            "extensions": { "VRMC_vrm": { "specVersion": "1.0", "meta": { "name": "Avatar" } } }
        });
        let doc = read(&glb(1.0, &extra), "vrm");
        assert!(doc.json().get("extensionsRequired").is_none());
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["format"], "VRM");
        assert_eq!(source["version"], "1.0");
        assert_eq!(source["title"], "Avatar");
        assert_eq!(doc.source_info().forward_axis, Fact::Declared(Axis::PosZ));
        assert_eq!(doc.facts["triangles"], 1);
        let m = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert!(m.prims[0].positions.iter().all(|p| p[2] > 0.0), "VRM 1.0 isn't turned");
    }

    #[test]
    fn vrm_0_is_turned_to_face_forward() {
        let extra = json!({
            "extensionsUsed": ["VRM", "KHR_materials_unlit"],
            "extensions": { "VRM": { "specVersion": "0.0", "meta": { "title": "Old" } } }
        });
        let doc = read(&glb(-1.0, &extra), "vrm");
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["version"], "0.0");
        assert_eq!(source["title"], "Old");
        let m = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert!(
            m.prims[0].positions.iter().all(|p| p[2] > 0.99),
            "{:?}",
            m.prims[0].positions
        );
        assert!(doc.to_glb().is_ok());
    }

    #[test]
    fn the_avatar_says_who_made_it() {
        let extra = json!({
            "extensionsUsed": ["VRMC_vrm"],
            "extensions": { "VRMC_vrm": { "specVersion": "1.0", "meta": {
                "name": "Avatar", "version": "3", "authors": ["Ann", "Bo"],
                "copyrightInformation": "(c) Ann", "licenseUrl": "https://vrm.dev/licenses/1.0/",
                "references": ["https://example.com/a"],
            } } }
        });
        let doc = read(&glb(1.0, &extra), "vrm");
        let info = doc.source_info();
        assert_eq!(info.title.as_deref(), Some("Avatar"));
        assert_eq!(info.author.as_deref(), Some("Ann; Bo"));
        assert_eq!(info.copyright.as_deref(), Some("(c) Ann"));
        assert_eq!(info.license.as_deref(), Some("https://vrm.dev/licenses/1.0/"));
        assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosY));
        assert_eq!(info.converted, crate::source::Converted::default());
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["modelVersion"], "3");
        assert_eq!(source["references"], json!(["https://example.com/a"]));

        let extra = json!({
            "extensionsUsed": ["VRM"],
            "extensions": { "VRM": { "specVersion": "0.0", "meta": {
                "title": "Old", "author": "Cy", "version": "1.2", "contactInformation": "cy@example.com",
                "licenseName": "Other", "otherLicenseUrl": "https://example.com/license",
            } } }
        });
        let doc = read(&glb(-1.0, &extra), "vrm");
        let info = doc.source_info();
        assert_eq!(info.title.as_deref(), Some("Old"));
        assert_eq!(info.author.as_deref(), Some("Cy"));
        assert_eq!(info.license.as_deref(), Some("https://example.com/license"));
        assert_eq!(info.forward_axis, Fact::Declared(Axis::NegZ));
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["contactInformation"], "cy@example.com");
        assert_eq!(source["modelVersion"], "1.2");
        // The glTF itself is as it was, but for the turn.
        assert!(doc.json()["asset"].get("copyright").is_none());
    }

    #[test]
    fn vrma_reads_as_glb() {
        let extra = json!({
            "extensionsUsed": ["VRMC_vrm_animation"],
            "extensions": { "VRMC_vrm_animation": { "specVersion": "1.0" } }
        });
        let doc = read(&glb(1.0, &extra), "vrma");
        assert_eq!(doc.json()["asset"]["extras"]["source"]["format"], "VRMA");
    }

    #[test]
    fn damaged_files_never_panic() {
        let extra = json!({
            "extensionsUsed": ["VRM"],
            "extensionsRequired": ["VRM"],
            "extensions": { "VRM": { "specVersion": "0.0" } }
        });
        let sample = glb(-1.0, &extra);
        let check = |bytes: &[u8]| {
            if let Ok(doc) = super::super::load_bytes(bytes, "a.vrm", "vrm", &Memory::new(), Keep::ALL) {
                let _ = doc.flatten(Pose::Rest);
                let _ = doc.to_glb();
            }
        };
        for n in 0..=sample.len() {
            check(&sample[..n]);
        }
        let mut bytes = sample.clone();
        for i in 0..bytes.len() {
            for flip in [0x01, 0x80] {
                bytes[i] ^= flip;
                check(&bytes);
                bytes[i] ^= flip;
            }
        }
    }
}
