//! One glTF document grafted into another under a node: every index the grafted one holds moved past what the
//! other already has.
#![allow(
    clippy::cast_possible_truncation,
    reason = "indices into a glTF document in memory fit in usize"
)]
#![cfg_attr(
    not(feature = "irr"),
    allow(dead_code, reason = "only Irrlicht scenes place a graft a second time")
)]

use serde_json::{Map, Value, json};

use crate::document::Document;

/// Takes into `doc`'s report what `part`, a document read as part of it (a mesh file a scene places, a tile's
/// content), reported in its `asset.extras.source`: the files it didn't find, each once, and its dropped
/// textures, a texture both name the same way counted once ([`crate::images::Missed::add`]). Its dropped
/// textures count as `doc`'s own, which a prune of `doc`'s textures takes away with the files among them that
/// are textures.
pub(super) fn report(doc: &mut Document, part: &Document) {
    let source = |d: &Document| {
        d.json()
            .get("asset")
            .and_then(|a| a.pointer("/extras/source"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    };
    let theirs = source(part);
    let dropped = theirs.get("droppedTextures").and_then(Value::as_u64).unwrap_or(0) as usize;
    let files: Vec<String> = theirs
        .get("missingFiles")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|f| f.as_str().map(str::to_owned))
        .collect();
    if dropped == 0 && files.is_empty() {
        return;
    }
    // What the part counted, whichever way it did; the names it knows its textures by, to count each once.
    let mut more = part.missed.as_ref().map(|m| m.0.clone()).unwrap_or_default();
    more.dropped = dropped;
    more.files.retain(|f| files.contains(f));
    more.names.truncate(dropped);
    let mine = source(doc);
    let (mut missed, _) = doc.missed.take().map(|m| *m).unwrap_or_default();
    missed.dropped = mine.get("droppedTextures").and_then(Value::as_u64).unwrap_or(0) as usize;
    missed.names.truncate(missed.dropped);
    missed.add(more);
    let json = doc.base_mut();
    let asset = json.entry("asset").or_insert_with(|| json!({ "version": "2.0" }));
    if !asset.get("extras").is_some_and(Value::is_object) {
        asset["extras"] = json!({});
    }
    if !asset["extras"].get("source").is_some_and(Value::is_object) {
        asset["extras"]["source"] = json!({});
    }
    let Some(out) = asset["extras"]["source"].as_object_mut() else {
        return;
    };
    if missed.dropped > 0 {
        out.insert("droppedTextures".into(), json!(missed.dropped));
    }
    if !files.is_empty() {
        let list = out.entry("missingFiles").or_insert_with(|| json!([]));
        if let Some(list) = list.as_array_mut() {
            for f in files {
                if !list.iter().any(|v| v.as_str() == Some(f.as_str())) {
                    list.push(Value::from(f));
                }
            }
        }
    }
    doc.missed = Some(Box::new((missed, true)));
}

/// Where a graft's nodes went, so a static one can be placed again sharing its meshes.
pub(super) struct Placed {
    first: usize,
    count: usize,
    roots: Vec<usize>,
    /// Whether it has no skins or animations, which name its nodes and so can't be shared.
    pub shareable: bool,
}

/// How many entries `json[key]` has.
fn len(json: &Map<String, Value>, key: &str) -> usize {
    json.get(key).and_then(Value::as_array).map_or(0, Vec::len)
}

/// How many punctual lights the document has.
fn lights(json: &Map<String, Value>) -> usize {
    json.get("extensions")
        .and_then(|e| e.get("KHR_lights_punctual"))
        .and_then(|l| l.get("lights"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

/// The index counts of a document: where the grafted one's entries start.
#[derive(Clone, Copy, Default)]
struct Offsets {
    nodes: usize,
    meshes: usize,
    materials: usize,
    textures: usize,
    images: usize,
    samplers: usize,
    accessors: usize,
    views: usize,
    buffers: usize,
    skins: usize,
    cameras: usize,
    lights: usize,
    variants: usize,
}

impl Offsets {
    fn of(json: &Map<String, Value>) -> Self {
        Self {
            nodes: len(json, "nodes"),
            meshes: len(json, "meshes"),
            materials: len(json, "materials"),
            textures: len(json, "textures"),
            images: len(json, "images"),
            samplers: len(json, "samplers"),
            accessors: len(json, "accessors"),
            views: len(json, "bufferViews"),
            buffers: len(json, "buffers"),
            skins: len(json, "skins"),
            cameras: len(json, "cameras"),
            lights: lights(json),
            variants: json
                .get("extensions")
                .and_then(|e| e.get("KHR_materials_variants"))
                .and_then(|v| v.get("variants"))
                .and_then(Value::as_array)
                .map_or(0, Vec::len),
        }
    }
}

/// Adds `by` to the index at `v[key]`, when there is one.
fn bump(v: &mut Value, key: &str, by: usize) {
    if let Some(x) = v.get_mut(key)
        && let Some(i) = x.as_u64()
    {
        *x = json!(i as usize + by);
    }
}

/// Adds `by` to every index in the list at `v[key]`.
fn bump_all(v: &mut Value, key: &str, by: usize) {
    if let Some(Value::Array(list)) = v.get_mut(key) {
        for x in list {
            if let Some(i) = x.as_u64() {
                *x = json!(i as usize + by);
            }
        }
    }
}

/// Adds `by` to every value of the object at `v[key]` (a primitive's attributes, a morph target).
fn bump_values(v: &mut Value, by: usize) {
    if let Some(o) = v.as_object_mut() {
        for x in o.values_mut() {
            if let Some(i) = x.as_u64() {
                *x = json!(i as usize + by);
            }
        }
    }
}

/// Moves every texture reference (`{"index": n}` under a key ending in `Texture`) in a material.
fn bump_textures(v: &mut Value, by: usize) {
    match v {
        Value::Object(o) => {
            for (k, x) in o.iter_mut() {
                if k.ends_with("Texture") && x.is_object() {
                    bump(x, "index", by);
                }
                bump_textures(x, by);
            }
        }
        Value::Array(list) => list.iter_mut().for_each(|x| bump_textures(x, by)),
        _ => {}
    }
}

/// An animation pointer with the index it names moved.
fn pointer(p: &str, o: &Offsets) -> String {
    let rules: [(&str, usize); 6] = [
        ("/nodes/", o.nodes),
        ("/materials/", o.materials),
        ("/meshes/", o.meshes),
        ("/cameras/", o.cameras),
        ("/textures/", o.textures),
        ("/extensions/KHR_lights_punctual/lights/", o.lights),
    ];
    for (prefix, by) in rules {
        if let Some(rest) = p.strip_prefix(prefix) {
            let (num, tail) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
            if let Ok(i) = num.parse::<usize>() {
                return format!("{prefix}{}{tail}", i + by);
            }
        }
    }
    p.to_owned()
}

/// The entries of `src[key]`, each changed by `f`, appended to `dst[key]`.
fn append(dst: &mut Map<String, Value>, src: &mut Map<String, Value>, key: &str, mut f: impl FnMut(&mut Value)) {
    let Some(Value::Array(mut list)) = src.remove(key) else {
        return;
    };
    if list.is_empty() {
        return;
    }
    list.iter_mut().for_each(&mut f);
    if let Value::Array(d) = dst.entry(key).or_insert_with(|| json!([])) {
        d.extend(list);
    }
}

/// Adds `child` to node `parent`'s children.
fn adopt(json: &mut Map<String, Value>, parent: usize, children: &[usize]) {
    if children.is_empty() {
        return;
    }
    if let Some(Value::Array(nodes)) = json.get_mut("nodes")
        && let Some(Value::Object(p)) = nodes.get_mut(parent)
        && let Value::Array(c) = p.entry("children").or_insert_with(|| json!([]))
    {
        c.extend(children.iter().map(|&i| json!(i)));
    }
}

/// Grafts `src` into `dst` under node `parent`. With `materials`, a list as long as the grafted document's
/// materials, its primitives use those of `dst` in their place. What `src` reported missing is reported by `dst`
/// (see [`report`]).
pub(super) fn graft(dst: &mut Document, mut src: Document, parent: usize, materials: Option<&[usize]>) -> Placed {
    report(dst, &src);
    let o = Offsets::of(dst.json());
    let mut s = std::mem::take(src.json_mut());
    let Document {
        buffers,
        refs,
        lines_and_points,
        ..
    } = src;
    let materials = materials.filter(|m| m.len() == len(&s, "materials"));
    let map_material = |i: usize| materials.map_or(i + o.materials, |m| m[i]);
    let shareable = len(&s, "skins") == 0 && len(&s, "animations") == 0;
    let scene = s.get("scene").and_then(Value::as_u64).unwrap_or(0) as usize;
    let roots: Vec<usize> = s
        .get("scenes")
        .and_then(|v| v.get(scene))
        .and_then(|v| v.get("nodes"))
        .and_then(Value::as_array)
        .map(|r| {
            r.iter()
                .filter_map(Value::as_u64)
                .map(|i| i as usize + o.nodes)
                .collect()
        })
        .unwrap_or_default();
    let count = len(&s, "nodes");
    let d = dst.json_mut();

    append(d, &mut s, "nodes", |n| {
        bump_all(n, "children", o.nodes);
        bump(n, "mesh", o.meshes);
        bump(n, "skin", o.skins);
        bump(n, "camera", o.cameras);
        if let Some(e) = n.get_mut("extensions") {
            if let Some(l) = e.get_mut("KHR_lights_punctual") {
                bump(l, "light", o.lights);
            }
            if let Some(a) = e
                .get_mut("EXT_mesh_gpu_instancing")
                .and_then(|i| i.get_mut("attributes"))
            {
                bump_values(a, o.accessors);
            }
        }
    });
    append(d, &mut s, "meshes", |m| {
        let Some(Value::Array(prims)) = m.get_mut("primitives") else {
            return;
        };
        for p in prims {
            if let Some(a) = p.get_mut("attributes") {
                bump_values(a, o.accessors);
            }
            bump(p, "indices", o.accessors);
            if let Some(x) = p.get_mut("material")
                && let Some(i) = x.as_u64()
            {
                *x = json!(map_material(i as usize));
            }
            if let Some(Value::Array(targets)) = p.get_mut("targets") {
                for t in targets.iter_mut() {
                    bump_values(t, o.accessors);
                }
            }
            if let Some(e) = p.get_mut("extensions") {
                if let Some(draco) = e.get_mut("KHR_draco_mesh_compression") {
                    bump(draco, "bufferView", o.views);
                }
                if let Some(Value::Array(maps)) =
                    e.get_mut("KHR_materials_variants").and_then(|v| v.get_mut("mappings"))
                {
                    for map in maps {
                        if let Some(x) = map.get_mut("material")
                            && let Some(i) = x.as_u64()
                        {
                            *x = json!(map_material(i as usize));
                        }
                        bump_all(map, "variants", o.variants);
                    }
                }
            }
        }
    });
    if materials.is_some() {
        s.remove("materials");
    } else {
        append(d, &mut s, "materials", |m| bump_textures(m, o.textures));
    }
    append(d, &mut s, "textures", |t| {
        bump(t, "source", o.images);
        bump(t, "sampler", o.samplers);
        if let Some(Value::Object(e)) = t.get_mut("extensions") {
            e.values_mut().for_each(|x| bump(x, "source", o.images));
        }
    });
    append(d, &mut s, "images", |i| bump(i, "bufferView", o.views));
    append(d, &mut s, "samplers", |_| {});
    append(d, &mut s, "accessors", |a| {
        bump(a, "bufferView", o.views);
        if let Some(sp) = a.get_mut("sparse") {
            for part in ["indices", "values"] {
                if let Some(x) = sp.get_mut(part) {
                    bump(x, "bufferView", o.views);
                }
            }
        }
    });
    append(d, &mut s, "bufferViews", |v| {
        bump(v, "buffer", o.buffers);
        if let Some(Value::Object(e)) = v.get_mut("extensions") {
            e.values_mut().for_each(|x| bump(x, "buffer", o.buffers));
        }
    });
    append(d, &mut s, "buffers", |_| {});
    append(d, &mut s, "skins", |k| {
        bump(k, "inverseBindMatrices", o.accessors);
        bump(k, "skeleton", o.nodes);
        bump_all(k, "joints", o.nodes);
    });
    append(d, &mut s, "cameras", |_| {});
    append(d, &mut s, "animations", |a| {
        if let Some(Value::Array(samplers)) = a.get_mut("samplers") {
            for x in samplers {
                bump(x, "input", o.accessors);
                bump(x, "output", o.accessors);
            }
        }
        if let Some(Value::Array(channels)) = a.get_mut("channels") {
            for c in channels {
                let Some(t) = c.get_mut("target") else { continue };
                bump(t, "node", o.nodes);
                if let Some(Value::String(p)) = t.pointer_mut("/extensions/KHR_animation_pointer/pointer") {
                    *p = pointer(p, &o);
                }
            }
        }
    });

    // Root extensions: the lights and variants are lists to join; the names used, a set.
    if let Some(Value::Object(mut ext)) = s.remove("extensions") {
        for (name, list) in [
            ("KHR_lights_punctual", "lights"),
            ("KHR_materials_variants", "variants"),
        ] {
            let Some(Value::Array(items)) = ext.remove(name).and_then(|mut e| e.get_mut(list).map(Value::take)) else {
                continue;
            };
            let de = d.entry("extensions").or_insert_with(|| json!({}));
            if let Some(de) = de.as_object_mut() {
                let entry = de
                    .entry(name)
                    .or_insert_with(|| Value::Object(Map::from_iter([(list.to_owned(), json!([]))])));
                if let Some(Value::Array(dl)) = entry.get_mut(list) {
                    dl.extend(items);
                }
            }
        }
    }
    for key in ["extensionsUsed", "extensionsRequired"] {
        if let Some(Value::Array(names)) = s.remove(key)
            && let Value::Array(dl) = d.entry(key).or_insert_with(|| json!([]))
        {
            for n in names {
                if !dl.contains(&n) {
                    dl.push(n);
                }
            }
        }
    }
    adopt(d, parent, &roots);
    dst.buffers.extend(buffers);
    dst.refs.extend(refs.into_iter().map(|(i, r)| (i + o.images, r)));
    dst.lines_and_points |= lines_and_points;
    Placed {
        first: o.nodes,
        count,
        roots,
        shareable,
    }
}

/// Places a static graft again under node `parent`: its nodes copied, sharing its meshes.
pub(super) fn repeat(dst: &mut Document, placed: &Placed, parent: usize) {
    let at = len(dst.json(), "nodes");
    let Some(Value::Array(nodes)) = dst.base_mut().get_mut("nodes") else {
        return;
    };
    let Some(copies) = nodes
        .get(placed.first..placed.first + placed.count)
        .map(<[Value]>::to_vec)
    else {
        return;
    };
    for mut n in copies {
        if let Some(Value::Array(c)) = n.get_mut("children") {
            for x in c {
                if let Some(i) = x.as_u64() {
                    *x = json!(i as usize - placed.first + at);
                }
            }
        }
        nodes.push(n);
    }
    let roots: Vec<usize> = placed.roots.iter().map(|r| r - placed.first + at).collect();
    adopt(dst.base_mut(), parent, &roots);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_pointers() {
        let o = Offsets {
            nodes: 3,
            materials: 2,
            lights: 1,
            ..Offsets::default()
        };
        assert_eq!(pointer("/nodes/0/translation", &o), "/nodes/3/translation");
        assert_eq!(
            pointer("/materials/1/pbrMetallicRoughness/baseColorFactor", &o),
            "/materials/3/pbrMetallicRoughness/baseColorFactor"
        );
        assert_eq!(
            pointer("/extensions/KHR_lights_punctual/lights/0/color", &o),
            "/extensions/KHR_lights_punctual/lights/1/color"
        );
        let mut m = json!({ "normalTexture": { "index": 0 }, "extensions": { "KHR_materials_clearcoat": { "clearcoatTexture": { "index": 1 } } } });
        bump_textures(&mut m, 5);
        assert_eq!(m["normalTexture"]["index"], 5);
        assert_eq!(
            m["extensions"]["KHR_materials_clearcoat"]["clearcoatTexture"]["index"],
            6
        );
    }
}
