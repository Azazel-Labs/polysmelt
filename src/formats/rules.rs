//! The reader rules (`docs/consistency.md`) for a glTF document that a reader converts rather than passes through:
//! glTF 1.0 upgraded to 2.0 ([`super::gltf`]), a VRM avatar or animation ([`super::vrm`]), and what a 3D Tiles
//! tile or tileset places ([`super::tiles`]). A glTF 2.0 file read as itself never comes here: it is kept as it
//! is, compression and all.
//!
//! What the file already has right is left as it is; only what breaks a rule is rewritten:
//!
//! - **Animations start at 0** and their keys are clean ([`super::keys::conform`]), as every reader's are.
//! - **Skins follow the one skinning rule** ([`crate::skin::conform`]): at most sixteen influences, each joint
//!   once, weights summing to 1, a vertex no joint moves following the node that draws it, and `skin.skeleton`
//!   the joints' common root.
//! - **No empty names**: a `name` of `""` is taken away, as no reader writes one.
//!
//! The root node is the caller's, since where it goes differs: glTF 1.0 and VRM hang from a `RootNode` ([`super::frame`]),
//! and 3D Tiles converts its tile space in place.

use serde_json::Value;

use crate::document::{Document, Keep};

/// Holds `doc` to the rules above, as far as `keep` keeps what they're about.
pub(crate) fn conform(doc: &mut Document, keep: Keep) {
    if keep.animations {
        super::keys::conform(doc, keep.extras);
    }
    if keep.skins {
        crate::skin::conform(doc);
    }
    unnamed(doc);
}

/// Takes the empty `name` off everything in the document's lists: something with no name has no `name` key.
fn unnamed(doc: &mut Document) {
    let empty = |v: &Value| v.get("name").and_then(Value::as_str) == Some("");
    let any = doc
        .json()
        .values()
        .filter_map(Value::as_array)
        .any(|list| list.iter().any(empty));
    if !any {
        return;
    }
    for list in doc.json_mut().values_mut().filter_map(Value::as_array_mut) {
        for item in list.iter_mut().filter(|v| empty(v)) {
            if let Some(o) = item.as_object_mut() {
                o.remove("name");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};

    use super::*;

    fn doc(v: Value) -> Document {
        let Value::Object(json) = v else { unreachable!() };
        Document::new(json, Vec::new(), Map::new(), crate::images::Refs::new())
    }

    #[test]
    fn an_empty_name_is_no_name() {
        let mut d = doc(json!({
            "asset": { "version": "2.0" },
            "nodes": [{ "name": "" }, { "name": "kept" }],
            "materials": [{ "name": "" }],
        }));
        unnamed(&mut d);
        assert_eq!(d.json()["nodes"], json!([{}, { "name": "kept" }]));
        assert_eq!(d.json()["materials"], json!([{}]));
    }
}
