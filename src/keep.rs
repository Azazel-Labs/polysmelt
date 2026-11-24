//! What a load keeps: [`Keep`], applied by [`Document::prune`].

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::document::Document;
use crate::mesh::{MAX_POINTS, MAX_TRIANGLES, TRIANGLE_CEILING};

/// Which parts of a model a load keeps. Meshes and the node hierarchy are always kept; each
/// switch here drops one kind of data on top of that. [`Keep::default`] keeps everything.
/// [`Keep::max_triangles`] caps how big a model a load takes at all, and [`Keep::max_points`]
/// how many points and lines [`Document::flatten`] draws.
///
/// Readers skip reading what isn't kept, so a narrow `Keep` also makes a load faster: no
/// texture files, embedded images or (without materials) material libraries and shaders are
/// read, nor animations, skins, morph targets, cameras or lights where the format keeps them
/// apart. What a reader did read and [`Document::prune`] then dropped leaves its buffer views
/// and accessors behind, unreferenced.
///
/// Without textures, a material's factors are its own: a reader that would have found a color
/// texture doesn't turn the base color white for it.
///
/// A `Keep` is built from [`Keep::ALL`] (or [`Keep::MESHES`], [`Keep::LIGHT`]) with a `with_`
/// setter for each switch, which chain and work in `const`s:
///
/// ```
/// use polysmelt::Keep;
/// // Static meshes for an engine that animates nothing, but still skins them.
/// let keep = Keep::ALL.with_animations(false);
/// # assert!(keep.skins);
/// // A 100-million-triangle scan, past the default limit.
/// let keep = Keep::ALL.with_max_triangles(120_000_000);
/// // A 40-million-point lidar scan, drawn as points, read on every core.
/// const SCAN: Keep = Keep::ALL.with_max_points(40_000_000).with_threads(0);
/// // Only the surfaces: no points or lines in the flattened mesh.
/// let keep = Keep::ALL.with_lines_and_points(false);
/// ```
///
/// New switches may come in later versions, each keeping by default what it switches, so a
/// `Keep` can't be written out field by field.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::struct_excessive_bools, reason = "independent switches")]
pub struct Keep {
    /// Animations.
    pub animations: bool,
    /// Skins: skeletons, their inverse bind matrices, and the joints and weights on vertices.
    /// The joint nodes stay either way, as part of the hierarchy.
    pub skins: bool,
    /// Morph targets and their weights (and animation channels that set them).
    pub morph_targets: bool,
    /// Materials; without them every primitive takes the default material, and textures go
    /// too.
    pub materials: bool,
    /// Textures, their images and samplers, and where materials use them.
    pub textures: bool,
    /// The images' bytes. Without them, kept textures' images are found but not read or
    /// converted: each is a glTF image with a `uri`, and [`Document::images`] has a handle on it
    /// that says where it is (its file, if it's in one) and reads it on request.
    /// [`Document::embed_images`] reads them all. [`Document::prune`] leaves images as they
    /// are: once read, they stay read. Images a reader must look into to make a texture (an OBJ's
    /// roughness and metalness maps packed into one) are read all the same, so the textures match.
    pub image_bytes: bool,
    /// Cameras.
    pub cameras: bool,
    /// Lights (`KHR_lights_punctual`).
    pub lights: bool,
    /// `extras` everywhere: custom properties and each format's own details. What the file says
    /// about itself in the keys [`SourceInfo`](crate::source::SourceInfo) reads (units, up axis,
    /// author...) stays in `asset.extras.source`, and the document's facts stay, either way.
    pub extras: bool,
    /// The most triangles a load accepts, against hostile or runaway files: a model that draws
    /// more is refused (and [`Document::flatten`] refuses to draw more). Defaults to
    /// [`MAX_TRIANGLES`]; raise it for big scans. Values past [`TRIANGLE_CEILING`] are clamped
    /// to it.
    pub max_triangles: usize,
    /// Points and lines (glTF's `POINTS`, `LINES`, `LINE_LOOP` and `LINE_STRIP` primitives) in
    /// what [`Document::flatten`] draws: point clouds, wireframes, curves. Without them it draws
    /// only triangles, and a model of points and lines alone flattens to nothing. The document
    /// keeps them either way.
    pub lines_and_points: bool,
    /// The most points and line segments [`Document::flatten`] draws, against huge point clouds:
    /// a model that draws more is refused (unless [`Keep::lines_and_points`] leaves them out).
    /// Counted apart from triangles. Defaults to [`MAX_POINTS`]; raise it for big scans. Values
    /// past [`TRIANGLE_CEILING`] are clamped to it.
    pub max_points: usize,
    /// Threads a reader may split its work over, where it can: OBJ text and SPZ, so far. 1, the
    /// default, reads on the calling thread; 0 takes one per core. The model read is the same
    /// either way. Where threads can't be had (WebAssembly without shared memory, or a
    /// platform that refuses to start one), it reads on the calling thread.
    pub threads: usize,
}

impl Default for Keep {
    fn default() -> Self {
        Self::ALL
    }
}

impl Keep {
    /// Everything.
    pub const ALL: Keep = Keep {
        animations: true,
        skins: true,
        morph_targets: true,
        materials: true,
        textures: true,
        image_bytes: true,
        cameras: true,
        lights: true,
        extras: true,
        max_triangles: MAX_TRIANGLES,
        lines_and_points: true,
        max_points: MAX_POINTS,
        threads: 1,
    };

    /// The meshes and what shapes or shades them (skins, morph targets, materials, textures,
    /// extras), without animations, cameras or lights.
    pub const MESHES: Keep = Keep {
        animations: false,
        cameras: false,
        lights: false,
        ..Keep::ALL
    };

    /// [`Keep::MESHES`] without reading the images: textures stay, as references to their
    /// images (see [`Keep::image_bytes`]).
    pub const LIGHT: Keep = Keep {
        image_bytes: false,
        ..Keep::MESHES
    };

    /// With [`Keep::animations`] set to `animations`.
    #[must_use]
    pub const fn with_animations(mut self, animations: bool) -> Self {
        self.animations = animations;
        self
    }

    /// With [`Keep::skins`] set to `skins`.
    #[must_use]
    pub const fn with_skins(mut self, skins: bool) -> Self {
        self.skins = skins;
        self
    }

    /// With [`Keep::morph_targets`] set to `morph_targets`.
    #[must_use]
    pub const fn with_morph_targets(mut self, morph_targets: bool) -> Self {
        self.morph_targets = morph_targets;
        self
    }

    /// With [`Keep::materials`] set to `materials`.
    #[must_use]
    pub const fn with_materials(mut self, materials: bool) -> Self {
        self.materials = materials;
        self
    }

    /// With [`Keep::textures`] set to `textures`.
    #[must_use]
    pub const fn with_textures(mut self, textures: bool) -> Self {
        self.textures = textures;
        self
    }

    /// With [`Keep::image_bytes`] set to `image_bytes`.
    #[must_use]
    pub const fn with_image_bytes(mut self, image_bytes: bool) -> Self {
        self.image_bytes = image_bytes;
        self
    }

    /// With [`Keep::cameras`] set to `cameras`.
    #[must_use]
    pub const fn with_cameras(mut self, cameras: bool) -> Self {
        self.cameras = cameras;
        self
    }

    /// With [`Keep::lights`] set to `lights`.
    #[must_use]
    pub const fn with_lights(mut self, lights: bool) -> Self {
        self.lights = lights;
        self
    }

    /// With [`Keep::extras`] set to `extras`.
    #[must_use]
    pub const fn with_extras(mut self, extras: bool) -> Self {
        self.extras = extras;
        self
    }

    /// With [`Keep::max_triangles`] set to `max_triangles`.
    #[must_use]
    pub const fn with_max_triangles(mut self, max_triangles: usize) -> Self {
        self.max_triangles = max_triangles;
        self
    }

    /// With [`Keep::lines_and_points`] set to `lines_and_points`.
    #[must_use]
    pub const fn with_lines_and_points(mut self, lines_and_points: bool) -> Self {
        self.lines_and_points = lines_and_points;
        self
    }

    /// With [`Keep::max_points`] set to `max_points`.
    #[must_use]
    pub const fn with_max_points(mut self, max_points: usize) -> Self {
        self.max_points = max_points;
        self
    }

    /// With [`Keep::threads`] set to `threads`.
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// [`Keep::max_triangles`], clamped to [`TRIANGLE_CEILING`].
    pub(crate) fn triangles(self) -> usize {
        self.max_triangles.min(TRIANGLE_CEILING)
    }

    /// [`Keep::max_points`], clamped to [`TRIANGLE_CEILING`].
    pub(crate) fn points(self) -> usize {
        self.max_points.min(TRIANGLE_CEILING)
    }

    /// [`Keep::threads`]: how many to use here, at least 1. WebAssembly without shared memory
    /// has no threads; elsewhere, a reader whose threads can't be started reads on one.
    #[cfg_attr(
        not(any(
            feature = "obj",
            feature = "spz",
            feature = "las",
            feature = "pcd",
            feature = "xyz",
            feature = "e57",
            feature = "ac3d"
        )),
        allow(dead_code, reason = "only the OBJ and SPZ readers take threads")
    )]
    pub(crate) fn thread_count(self) -> usize {
        if cfg!(all(target_family = "wasm", not(target_feature = "atomics"))) {
            1
        } else if self.threads == 0 {
            std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
        } else {
            self.threads
        }
    }
}

#[cfg(test)]
thread_local! {
    /// For tests: thread starts fail, as on a platform without threads.
    pub(crate) static NO_THREADS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `f` started on a thread of `s`; false, and `f` dropped, where no thread can be started.
#[cfg_attr(
    not(any(
        feature = "obj",
        feature = "spz",
        feature = "las",
        feature = "pcd",
        feature = "xyz",
        feature = "e57",
        feature = "ac3d"
    )),
    allow(dead_code, reason = "only the OBJ and SPZ readers take threads")
)]
pub(crate) fn spawn<'s>(s: &'s std::thread::Scope<'s, '_>, f: impl FnOnce() + Send + 's) -> bool {
    #[cfg(test)]
    if NO_THREADS.get() {
        return false;
    }
    std::thread::Builder::new().spawn_scoped(s, f).is_ok()
}

/// Calls `f` on each item of `obj[key]` that's an object.
fn each(obj: &mut Map<String, Value>, key: &str, mut f: impl FnMut(&mut Map<String, Value>)) {
    let items = obj.get_mut(key).and_then(Value::as_array_mut);
    for item in items.into_iter().flatten().filter_map(Value::as_object_mut) {
        f(item);
    }
}

/// Calls `f` on every primitive.
fn primitives(json: &mut Map<String, Value>, mut f: impl FnMut(&mut Map<String, Value>)) {
    each(json, "meshes", |m| each(m, "primitives", &mut f));
}

/// Removes extension `name` from `obj`, and its `extensions` if that leaves it empty.
fn remove_extension(obj: &mut Map<String, Value>, name: &str) {
    if let Some(exts) = obj.get_mut("extensions").and_then(Value::as_object_mut) {
        exts.remove(name);
        if exts.is_empty() {
            obj.remove("extensions");
        }
    }
}

/// Removes extension `name` wherever it is.
fn remove_extension_everywhere(v: &mut Value, name: &str) {
    match v {
        Value::Object(obj) => {
            remove_extension(obj, name);
            obj.values_mut().for_each(|v| remove_extension_everywhere(v, name));
        }
        Value::Array(a) => a.iter_mut().for_each(|v| remove_extension_everywhere(v, name)),
        _ => {}
    }
}

/// Removes every `key` member, at any depth.
fn remove_key_everywhere(v: &mut Value, key: &str) {
    match v {
        Value::Object(obj) => {
            obj.remove(key);
            obj.values_mut().for_each(|v| remove_key_everywhere(v, key));
        }
        Value::Array(a) => a.iter_mut().for_each(|v| remove_key_everywhere(v, key)),
        _ => {}
    }
}

/// Removes the texture references (`...Texture` objects) in a material, its extensions'
/// included.
fn remove_textures(obj: &mut Map<String, Value>) {
    obj.retain(|k, v| !(k.ends_with("Texture") && v.is_object()));
    for v in obj.values_mut() {
        if let Value::Object(o) = v {
            remove_textures(o);
        }
    }
}

/// The names of every extension the JSON uses an object for.
fn extensions_present(v: &Value, out: &mut BTreeSet<String>) {
    match v {
        Value::Object(obj) => {
            if let Some(exts) = obj.get("extensions").and_then(Value::as_object) {
                out.extend(exts.keys().cloned());
            }
            obj.values().for_each(|v| extensions_present(v, out));
        }
        Value::Array(a) => a.iter().for_each(|v| extensions_present(v, out)),
        _ => {}
    }
}

impl Document {
    /// Drops what `keep` doesn't keep. The JSON stays valid glTF: references to what went are
    /// removed with it, and extensions no longer used are unlisted. Extensions Polysmelt doesn't
    /// know that refer to dropped materials, textures or cameras may be left pointing at them.
    /// [`Keep::max_triangles`] and [`Keep::threads`] play no part here. Points and lines stay in
    /// the JSON: [`Keep::lines_and_points`] and [`Keep::max_points`] are recorded for
    /// [`Document::flatten`] (see [`Document::set_lines_and_points`] and
    /// [`Document::set_max_points`]).
    pub fn prune(&mut self, keep: Keep) {
        self.lines_and_points = keep.lines_and_points;
        self.max_points = keep.points();
        if (Keep {
            image_bytes: true,
            max_triangles: Keep::ALL.max_triangles,
            lines_and_points: true,
            max_points: Keep::ALL.max_points,
            threads: Keep::ALL.threads,
            ..keep
        }) == Keep::ALL
        {
            return;
        }
        let mut before = BTreeSet::new();
        let mut root = Value::Object(std::mem::take(&mut self.json));
        extensions_present(&root, &mut before);
        let Value::Object(json) = &mut root else { unreachable!() };
        // Animation pointers (`KHR_animation_pointer`) into what goes, go with it.
        let mut dead: Vec<&str> = Vec::new();
        if !keep.animations {
            json.remove("animations");
        }
        if !keep.cameras {
            json.remove("cameras");
            each(json, "nodes", |n| {
                n.remove("camera");
            });
            dead.push("/cameras/");
        }
        if !keep.lights {
            remove_extension_everywhere(&mut root, "KHR_lights_punctual");
            dead.push("/extensions/KHR_lights_punctual/");
        }
        let Value::Object(json) = &mut root else { unreachable!() };
        if !keep.skins {
            json.remove("skins");
            each(json, "nodes", |n| {
                n.remove("skin");
            });
            primitives(json, |p| {
                let skinning = |k: &String| k.starts_with("JOINTS_") || k.starts_with("WEIGHTS_");
                if let Some(a) = p.get_mut("attributes").and_then(Value::as_object_mut) {
                    a.retain(|k, _| !skinning(k));
                }
                let draco = p
                    .get_mut("extensions")
                    .and_then(|e| e.get_mut("KHR_draco_mesh_compression"))
                    .and_then(|d| d.get_mut("attributes"))
                    .and_then(Value::as_object_mut);
                if let Some(a) = draco {
                    a.retain(|k, _| !skinning(k));
                }
            });
        }
        if !keep.morph_targets {
            primitives(json, |p| {
                p.remove("targets");
            });
            each(json, "meshes", |m| {
                m.remove("weights");
                if let Some(x) = m.get_mut("extras").and_then(Value::as_object_mut) {
                    x.remove("targetNames");
                }
            });
            each(json, "nodes", |n| {
                n.remove("weights");
            });
            each(json, "animations", |a| {
                if let Some(ch) = a.get_mut("channels").and_then(Value::as_array_mut) {
                    ch.retain(|c| c.pointer("/target/path").and_then(Value::as_str) != Some("weights"));
                }
            });
            dead.extend(["/meshes/*/weights", "/nodes/*/weights"]);
        }
        let textures = keep.textures && keep.materials;
        if !keep.materials {
            json.remove("materials");
            primitives(json, |p| {
                p.remove("material");
            });
            dead.push("/materials/");
        }
        if !keep.materials {
            remove_extension_everywhere(&mut root, "KHR_materials_variants");
        }
        let Value::Object(json) = &mut root else { unreachable!() };
        if !textures {
            for key in ["textures", "images", "samplers"] {
                json.remove(key);
            }
            each(json, "materials", remove_textures);
            dead.push("/textures/");
            self.refs.clear();
        }
        if !keep.extras {
            // What the file says about itself isn't a custom property: its normalized keys stay.
            let info: Option<Map<String, Value>> = root
                .pointer("/asset/extras/source")
                .and_then(Value::as_object)
                .map(|s| {
                    s.iter()
                        .filter(|(k, _)| crate::source::KEYS.contains(&k.as_str()))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                })
                .filter(|m: &Map<String, Value>| !m.is_empty());
            remove_key_everywhere(&mut root, "extras");
            if let (Some(info), Some(asset)) = (info, root.get_mut("asset").and_then(Value::as_object_mut)) {
                asset.insert("extras".into(), serde_json::json!({ "source": info }));
            }
        }
        let Value::Object(json) = &mut root else { unreachable!() };
        if !dead.is_empty() {
            each(json, "animations", |a| {
                if let Some(ch) = a.get_mut("channels").and_then(Value::as_array_mut) {
                    ch.retain(|c| {
                        let pointer = c
                            .pointer("/target/extensions/KHR_animation_pointer/pointer")
                            .and_then(Value::as_str);
                        !pointer.is_some_and(|p| dead.iter().any(|d| matches_pointer(p, d)))
                    });
                }
            });
        }
        // An animation needs a channel.
        if let Some(anims) = json.get_mut("animations").and_then(Value::as_array_mut) {
            anims.retain(|a| {
                a.get("channels")
                    .and_then(Value::as_array)
                    .is_some_and(|c| !c.is_empty())
            });
            if anims.is_empty() {
                json.remove("animations");
            }
        }
        let mut after = BTreeSet::new();
        extensions_present(&root, &mut after);
        let Value::Object(json) = &mut root else { unreachable!() };
        for key in ["extensionsUsed", "extensionsRequired"] {
            if let Some(list) = json.get_mut(key).and_then(Value::as_array_mut) {
                list.retain(|e| e.as_str().is_none_or(|e| !before.contains(e) || after.contains(e)));
                if list.is_empty() {
                    json.remove(key);
                }
            }
        }
        let Value::Object(json) = root else { unreachable!() };
        self.json = json;
    }
}

/// Whether JSON pointer `p` starts with `prefix`, where `*` in the prefix stands for one
/// segment.
fn matches_pointer(p: &str, prefix: &str) -> bool {
    let mut segs = p.split('/');
    let pre: Vec<&str> = prefix.trim_end_matches('/').split('/').collect();
    pre.iter()
        .all(|want| segs.next().is_some_and(|s| *want == "*" || s == *want))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn doc(v: Value) -> Document {
        let Value::Object(json) = v else { unreachable!() };
        let mut d = Document::empty(Map::new());
        d.json = json;
        d
    }

    #[test]
    fn keeps_everything_by_default() {
        let v = json!({ "asset": { "version": "2.0" }, "animations": [{ "channels": [{}] }], "extras": { "a": 1 } });
        let mut d = doc(v.clone());
        d.prune(Keep::default());
        assert_eq!(Value::Object(d.json), v);
    }

    #[test]
    fn a_triangle_limit_alone_prunes_nothing() {
        // Pruning drops animations without channels, whatever it keeps: a limit alone mustn't start it.
        let v = json!({ "asset": { "version": "2.0" }, "animations": [{ "channels": [] }] });
        let mut d = doc(v.clone());
        d.prune(Keep {
            max_triangles: 10,
            image_bytes: false,
            ..Keep::ALL
        });
        assert_eq!(Value::Object(d.json), v);
        let mut d = doc(v);
        d.prune(Keep {
            extras: false,
            ..Keep::ALL
        });
        assert!(!d.json.contains_key("animations"));
    }

    #[test]
    fn point_settings_prune_nothing_but_reach_flatten() {
        let v = json!({ "asset": { "version": "2.0" }, "animations": [{ "channels": [] }] });
        let mut d = doc(v.clone());
        d.prune(Keep {
            lines_and_points: false,
            max_points: usize::MAX,
            ..Keep::ALL
        });
        assert_eq!(Value::Object(d.json.clone()), v);
        assert!(!d.lines_and_points());
        assert_eq!(d.max_points(), TRIANGLE_CEILING);
        d.prune(Keep::LIGHT);
        assert!(d.lines_and_points());
        assert_eq!(d.max_points(), MAX_POINTS);
    }

    #[test]
    fn the_triangle_limit_is_clamped() {
        assert_eq!(Keep::ALL.triangles(), MAX_TRIANGLES);
        assert_eq!(Keep::LIGHT.max_triangles, MAX_TRIANGLES);
        let keep = |max_triangles| Keep {
            max_triangles,
            ..Keep::ALL
        };
        assert_eq!(keep(12_000_000).triangles(), 12_000_000);
        assert_eq!(keep(usize::MAX).triangles(), TRIANGLE_CEILING);
    }

    #[test]
    fn drops_animations_but_keeps_skins() {
        let mut d = doc(json!({
            "asset": { "version": "2.0" },
            "nodes": [{ "mesh": 0, "skin": 0 }, { "camera": 0 }],
            "skins": [{ "joints": [1] }],
            "cameras": [{ "type": "perspective" }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2 } }] }],
            "animations": [{ "channels": [{ "sampler": 0, "target": { "node": 1, "path": "rotation" } }] }],
            "extensions": { "KHR_lights_punctual": { "lights": [{ "type": "point" }] } },
            "extensionsUsed": ["KHR_lights_punctual", "KHR_mesh_quantization"],
        }));
        d.prune(Keep::MESHES);
        assert_eq!(
            Value::Object(d.json),
            json!({
                "asset": { "version": "2.0" },
                "nodes": [{ "mesh": 0, "skin": 0 }, {}],
                "skins": [{ "joints": [1] }],
                "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2 } }] }],
                "extensionsUsed": ["KHR_mesh_quantization"],
            })
        );
    }

    #[test]
    fn drops_skins_morphs_and_textures() {
        let mut d = doc(json!({
            "asset": { "version": "2.0" },
            "nodes": [{ "mesh": 0, "skin": 0, "weights": [0.5] }],
            "skins": [{ "joints": [0] }],
            "meshes": [{
                "weights": [0.0],
                "extras": { "targetNames": ["smile"], "keep": true },
                "primitives": [{ "attributes": { "POSITION": 0, "JOINTS_0": 1 }, "targets": [{ "POSITION": 2 }], "material": 0 }],
            }],
            "materials": [{
                "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 }, "baseColorFactor": [1, 1, 1, 1] },
                "extensions": { "KHR_materials_clearcoat": { "clearcoatTexture": { "index": 0 }, "clearcoatFactor": 1 } },
            }],
            "textures": [{ "source": 0 }],
            "images": [{ "uri": "a.png" }],
            "animations": [
                { "channels": [{ "sampler": 0, "target": { "node": 0, "path": "weights" } }] },
                { "channels": [{ "sampler": 0, "target": { "node": 0, "path": "translation" } }] },
            ],
            "extensionsUsed": ["KHR_materials_clearcoat"],
        }));
        d.prune(Keep {
            skins: false,
            morph_targets: false,
            textures: false,
            ..Keep::ALL
        });
        assert_eq!(
            Value::Object(d.json),
            json!({
                "asset": { "version": "2.0" },
                "nodes": [{ "mesh": 0 }],
                "meshes": [{ "extras": { "keep": true }, "primitives": [{ "attributes": { "POSITION": 0 }, "material": 0 }] }],
                "materials": [{
                    "pbrMetallicRoughness": { "baseColorFactor": [1, 1, 1, 1] },
                    "extensions": { "KHR_materials_clearcoat": { "clearcoatFactor": 1 } },
                }],
                "animations": [{ "channels": [{ "sampler": 0, "target": { "node": 0, "path": "translation" } }] }],
                "extensionsUsed": ["KHR_materials_clearcoat"],
            })
        );
    }

    #[test]
    fn drops_extras_and_pointer_channels() {
        let mut d = doc(json!({
            "asset": { "version": "2.0", "extras": 1 },
            "materials": [{ "extras": { "x": 1 } }],
            "animations": [{ "channels": [
                { "sampler": 0, "target": { "path": "pointer", "extensions": { "KHR_animation_pointer": { "pointer": "/materials/0/emissiveFactor" } } } },
                { "sampler": 0, "target": { "path": "pointer", "extensions": { "KHR_animation_pointer": { "pointer": "/nodes/0/translation" } } } },
            ] }],
        }));
        d.prune(Keep {
            materials: false,
            extras: false,
            ..Keep::ALL
        });
        assert_eq!(
            Value::Object(d.json),
            json!({
                "asset": { "version": "2.0" },
                "animations": [{ "channels": [
                    { "sampler": 0, "target": { "path": "pointer", "extensions": { "KHR_animation_pointer": { "pointer": "/nodes/0/translation" } } } },
                ] }],
            })
        );
    }

    #[test]
    fn pointer_prefixes() {
        assert!(matches_pointer("/meshes/3/weights", "/meshes/*/weights"));
        assert!(matches_pointer("/meshes/3/weights/1", "/meshes/*/weights"));
        assert!(!matches_pointer("/meshes/3/primitives", "/meshes/*/weights"));
        assert!(matches_pointer("/cameras/0/perspective/yfov", "/cameras/"));
        assert!(!matches_pointer("/camerasx/0", "/cameras/"));
    }
}
