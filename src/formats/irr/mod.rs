//! Irrlicht's XML formats: meshes (`.irrmesh`) and scenes (`.irr`), in UTF-8 or UTF-16.
//!
//! A mesh is a list of `<buffer>`s, each a material, vertices (`standard`: position, normal, an ARGB color and a
//! texture coordinate; `2tcoords` with a second coordinate set; `tangents` with a tangent and binormal) and
//! triangle indices. Each buffer becomes a primitive with its own material, kept indexed; a triangle with an index
//! past the vertices, or a last one cut short, is dropped and counted.
//!
//! A scene is a tree of `<node>`s, each with a name, position, rotation (degrees about X, then Y, then Z) and
//! scale, optionally materials, animators and children. Mesh and animated mesh nodes (`mesh`, `octTree`,
//! `animatedMesh`) load the file they name: an Irrlicht mesh directly, any other format through its own reader,
//! the result grafted under the node, a file used again sharing the first one's meshes when it has no skins or
//! animations. The node's materials replace the file's when there are as many of them and the node doesn't mark
//! them read-only. Cubes, spheres, sky boxes (six quads, inside out) and billboards (a quad facing -Z, since glTF
//! can't turn one to the camera) are generated; lights become `KHR_lights_punctual` lights (a spot's cone angles
//! read as half-angles) and cameras glTF cameras, on a child node aimed at the camera's target with its up vector.
//! Animators become one animation per node, sampled in Irrlicht's own formulas: `rotation` (degrees per 10 ms,
//! several adding up), and the last of `flyCircle`, `flyStraight` and `followSpline` (a Hermite spline with the
//! animator's tightness) for the position; channels that repeat run for the shortest span after which they all
//! do, within 1%, or else for the longest repeat. A camera keeps aiming where it did at rest. Terrain, particle
//! systems, water, text and other node and animator types are counted as unsupported; hidden nodes are left out
//! with their children.
//!
//! Materials: the diffuse color (replaced by the vertices' color when they all share one that isn't white, or set
//! to white when they differ and the colors are kept per vertex), emissive, the specular color and shininess as
//! the highlight (none at a shininess of 0), the first
//! texture as the color image with its wrap modes and filters, the second as a normal map (normal map and parallax
//! types) or a lightmap (lightmap types; written as the occlusion texture, on the second coordinate set when there
//! is one), lighting off as unlit, no back-face culling as double-sided. Vertex alpha types blend by the vertices'
//! alpha, alpha channel types by the texture's (`trans_alphach_ref` as a mask); other types glTF can't show are
//! named in the material's extras.
//!
//! Irrlicht is Y up and left-handed, so everything is mirrored through the XY plane: in the source info the
//! handedness is declared left and converted, the up axis an assumed `+Y`, the units' length unknown.

mod material;
mod mesh;
mod world;

use std::collections::HashMap;
use std::path::Path;

use serde_json::json;

use super::scene::{Node, Scene, SceneMesh};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::source::{Axis, Fact, Handedness};
use material::{IrrMaterial, Patch, Textures};
use mesh::Buffer;

pub(crate) fn load(input: super::Input<'_>, ext: &str, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let bytes = input.read()?;
    let text = crate::formats::xml::text(&bytes);
    if ext == "irr" || (ext != "irrmesh" && text.contains("<irr_scene")) {
        world::load(&text, input.dir(), input.limit, assets, keep)
    } else {
        irrmesh(&text, &input, assets, keep)
    }
}

/// A mesh file on its own: one node, one mesh, a primitive per buffer.
fn irrmesh(text: &str, input: &super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let mesh = mesh::parse(text, input.limit)?;
    ensure!(!mesh.buffers.is_empty(), Empty, "this Irrlicht mesh has no buffers");
    let dir = input.dir();
    let mut ctx = Ctx::new(assets, keep, dir);
    let stem = input
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let m = ctx.mesh(&stem, &mesh.buffers, None, dir);
    let scene = &mut ctx.scene;
    scene.meshes.push(m);
    let mut node = Node::named(stem, None);
    node.mesh = Some(0);
    scene.add(node);
    scene.dropped = mesh.dropped;
    scene.facts.insert("buffers".into(), json!(mesh.buffers.len()));
    scene.info.format = Some("Irrlicht mesh".into());
    scene.info.version = mesh.version;
    ctx.finish()
}

/// What both readers build into: the scene, its materials' patches, and the textures read so far.
struct Ctx<'a> {
    scene: Scene,
    patches: Vec<Patch>,
    tex: Textures<'a>,
    /// Materials already made, by everything that makes them.
    made: HashMap<String, usize>,
    keep: Keep,
}

impl<'a> Ctx<'a> {
    fn new(assets: &'a dyn Assets, keep: Keep, root: &Path) -> Self {
        Self {
            scene: Scene::default(),
            patches: Vec::new(),
            tex: Textures::new(assets, keep, root),
            made: HashMap::new(),
            keep,
        }
    }

    /// The material for `m`, its textures found from `dir`, on vertices with `sets` coordinate sets that all
    /// have color `uniform` (or `None` when they differ).
    fn material(&mut self, m: &IrrMaterial, uniform: Option<u32>, sets: usize, dir: &Path) -> usize {
        let key = format!("{m:?}{uniform:?}{sets}{}", dir.display());
        if let Some(&i) = self.made.get(&key) {
            return i;
        }
        let index = self.scene.materials.len();
        let name = Path::new(&m.textures[0])
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("material{index}"));
        // A buffer with no material takes the default one, white under its vertices' colors.
        let (mut out, mut patch) = if m.missing {
            (
                crate::legacy_material::default_material(true),
                material::Patch::default(),
            )
        } else {
            material::convert(m, &name, &mut self.tex, dir, sets)
        };
        match uniform {
            // Irrlicht lights and draws with the vertices' color in place of the diffuse; all white reads as
            // no color given.
            Some(c) if c & 0x00FF_FFFF != 0x00FF_FFFF => {
                let l = material::linear(c);
                out.color[..3].copy_from_slice(&l[..3]);
            }
            Some(_) if m.missing => out.color = crate::mesh::GRAY,
            Some(_) => {}
            None => out.color[..3].fill(1.0),
        }
        if m.vertex_alpha() {
            match uniform {
                Some(c) => out.color[3] = material::linear(c)[3],
                None => patch.blend = true,
            }
        }
        self.scene.materials.push(out);
        self.patches.push(patch);
        self.made.insert(key, index);
        index
    }

    /// A mesh of `buffers`, with `overrides` in place of their own materials when given.
    fn mesh(&mut self, name: &str, buffers: &[Buffer], overrides: Option<&[IrrMaterial]>, dir: &Path) -> SceneMesh {
        let overrides = overrides.filter(|o| o.len() == buffers.len());
        let mut out = SceneMesh {
            name: name.to_owned(),
            ..SceneMesh::default()
        };
        for (i, b) in buffers.iter().enumerate() {
            if b.prim.indices.is_empty() {
                continue;
            }
            let m = overrides.map_or(&b.material, |o| &o[i]);
            let mut prim = b.prim.clone();
            if !m.vertex_alpha()
                && let Some(c) = prim.colors.first_mut()
            {
                for c in c.iter_mut() {
                    c[3] = 1.0;
                }
            }
            prim.material = Some(self.material(m, b.uniform, b.sets, dir));
            out.prims.push(prim);
        }
        out
    }

    /// Mirrors the scene into glTF's coordinates and writes it, with the materials' patches.
    fn finish(mut self) -> Result<Document> {
        let mut scene = std::mem::take(&mut self.scene);
        scene.images = std::mem::take(&mut self.tex.images.list);
        scene.info.up_axis = Fact::Assumed(Axis::PosY);
        scene.info.handedness = Fact::Declared(Handedness::Left);
        scene.mirror_z();
        let mut doc = scene.finish(self.keep)?;
        material::patch(doc.base_mut(), &self.patches, self.keep.extras);
        Ok(doc)
    }
}

#[cfg(test)]
mod tests;
