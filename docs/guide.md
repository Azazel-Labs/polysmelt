# A tour of the API

The full reference is on [docs.rs](https://docs.rs/polysmelt). This page shows how the pieces
fit together.

## Loading

`polysmelt::open(path)` reads a file and `polysmelt::read(bytes, name)` reads a model already in
memory. Both pick the format by the extension, or else by the first bytes, and return a
`Document`.

A `Loader` sets what those two decide for you:

```rust ignore
use polysmelt::{Format, Keep, Loader, assets::Folder};

let doc = Loader::new()
    .keep(Keep::ALL.with_animations(false)) // what to keep
    .assets(Folder::new("textures"))         // where referenced files are found
    .format(Format::Obj)                     // the format, if the name doesn't say
    .open("scan.txt")?;
```

### What to keep

`Keep` says what a load keeps: everything by default (`Keep::ALL`), or less. Readers skip what
isn't kept rather than reading it and throwing it away, so a lighter `Keep` is a faster load.

- `Keep { animations: false, ..Keep::ALL }` for static meshes that still carry their skins.
- `Keep { lines_and_points: false, ..Keep::ALL }` leaves out point clouds, wireframes and curves.
- With `image_bytes` off (as in `Keep::LIGHT`), textures are found but not read.
  `Document::images` gives a handle on each image that says where it is (a file, an archive
  entry, inside the model) and loads it on request; `Document::embed_images` reads them all.
- `Keep::with_crease_angle(degrees)` sets the angle used to make normals for meshes that have
  none (30 by default; see [consistency.md](consistency.md#geometry)).
- `Keep::threads` reads large OBJ files on several threads (`0` for one per core) and inflates
  SPZ files on one thread while unpacking them on another, with the same result as one thread.
  It is 1 by default.

### Limits

Against hostile or runaway files, a load refuses a model of more than `Keep::max_triangles`
triangles (64 million by default, `mesh::MAX_TRIANGLES`) or `Keep::max_points` points and line
segments (16 million, `mesh::MAX_POINTS`), and `Document::flatten` draws no more. Raise them for
big scans: `Keep { max_triangles: 120_000_000, ..Keep::ALL }`.

### Finding referenced files

Readers find the textures, material libraries, buffers and layers a model names through the
`Assets` trait. `assets::Folder` searches a folder tree (it needs the `fs` feature);
`assets::Memory` holds files by path, for models loaded from memory or in the browser. Override
`Assets::image` to convert texture formats polysmelt doesn't decode.

## The document

A `Document` is a glTF 2.0 document: its JSON (`json()`), its buffers, and `facts` about the
source. glTF input passes through untouched, nothing decoded; every other format is converted
into it following the rules in [consistency.md](consistency.md).

- `Document::source_info()` says what the file was: format, version, the tool that wrote it,
  its units and axes, and for each whether the file declared it or it was assumed.
- `Document::to_glb()` and `write_glb()` write it as GLB with images embedded. glTF input goes
  back out as it came, compression included.
- `Document::to_fbx()` and `write_fbx()` write binary FBX 7.4.
- `Document::decode()` rewrites it as plain glTF any reader takes: meshopt views and Draco
  primitives decoded, quantized attributes turned into floats, everything else untouched.
- `Document::read_floats` and `Document::read_indices` read any accessor as floats or exact
  integers, sparse, normalized, interleaved or meshopt-compressed alike.
- `Document::counts()` counts triangles, points and lines from the JSON alone, without decoding
  anything.

## Flat meshes

`Document::flatten(Pose)` bakes the document into a `mesh::Mesh`: transforms, instances, skins
and morph targets applied, ready to draw.

- `Pose::Rest` (the default) skins vertices in the rest pose, blending every `JOINTS_n` and
  `WEIGHTS_n` set. `Pose::Animation` poses them at a moment of an animation. `Pose::Bind`
  leaves skinned vertices as the file stores them, for a renderer that skins on the GPU, and is
  quicker.
- A primitive (`mesh::Prim`) has triangles, line segments (strips and loops opened into pairs)
  and points over one set of vertices. Vertices carry positions, normals, tangents, the UVs the
  color texture reads (with `KHR_texture_transform` applied), every other UV set
  (`Prim::uv_sets`, for lightmaps, masks and decals), vertex colors and their other sets, and for
  skinned meshes each vertex's four strongest joints and weights.
- A `mesh::Material` has the full glTF metallic-roughness model (base color, metallic-roughness,
  normal, occlusion and emissive textures and factors), alpha mode, double-sidedness, `unlit`,
  and the other `KHR_materials_*` extensions (ior, specular, transmission, clearcoat, sheen,
  volume, anisotropy, iridescence, dispersion) with their textures. Each texture's `Sampling`
  says which UVs it reads and its sampler's wrap and filters.
- `mesh::to_glb` writes a `Mesh` back out as a plain GLB, skins included.

`Document::flatten_nodes` draws the mesh in a pose you hand in (one `Trs` per node, from a
retarget, say) instead of one played from the file.

## Rigs and animation

`Document::rig()` gives typed views of the node tree, skins and animations, for code that plays
or retargets motion rather than drawing triangles.

- A `Rig` has nodes with names, parents and rest `Trs`, skins with their joints and inverse bind
  matrices, and animations as keyed channels (step, linear or cubic spline).
- `Channel::sample` and `Rig::pose(animation, time)` evaluate them. `Rig::poser` poses one
  animation time after time, stepping through each channel's keys instead of searching them.

An FBX armature, a `UsdSkel` skeleton, a Doom 3 skeleton and a glTF skin all come out the same
way.

## Examples

- `cargo run --example convert -- in.fbx out.glb` converts a model (to FBX for an `.fbx`
  output).
- `cargo run --example inspect -- in.fbx` prints what a model holds.
- `cargo run --example meshopt -- sample out.glb` writes a small meshopt-compressed test file.
