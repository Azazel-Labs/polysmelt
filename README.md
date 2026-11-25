# polysmelt

<p align="center"><a href="https://azazel-labs.github.io/polysmelt/"><img src="https://raw.githubusercontent.com/Azazel-Labs/polysmelt/main/media/hero.png" alt="polysmelt: 60+ 3D formats in, clean glTF out" width="100%"></a></p>

<p align="center"><b><a href="https://azazel-labs.github.io/polysmelt/">Live demo</a></b>: drop a model or a folder in your browser to see polysmelt read it, light it, play its animations and hand it back as GLB.</p>

[![crates.io](https://img.shields.io/crates/v/polysmelt.svg)](https://crates.io/crates/polysmelt)
[![docs.rs](https://img.shields.io/docsrs/polysmelt)](https://docs.rs/polysmelt)
[![CI](https://github.com/Azazel-Labs/polysmelt/actions/workflows/ci.yml/badge.svg)](https://github.com/Azazel-Labs/polysmelt/actions/workflows/ci.yml)

Pure-Rust 3D asset interchange, with glTF as the hub. Read what other tools write, keep what
you don't understand, and write compact or plain files back out.

## Usage

```rust no_run
use polysmelt::document::Pose;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Any format polysmelt reads: told by the extension, or else by the file's first bytes.
    // The textures and libraries the model names are found beside it.
    let doc = polysmelt::open("models/robot.fbx")?;

    // What the file says about itself, and what it draws.
    let info = doc.source_info();
    println!("{:?}, up {:?}: {} triangles", info.format, info.up_axis, doc.triangles());

    // Out as a GLB, everything it holds kept, or as flat, posed triangles.
    doc.write_glb(std::io::BufWriter::new(std::fs::File::create("robot.glb")?))?;
    let mesh = doc.flatten(Pose::Rest)?;
    Ok(())
}
```

A `Loader` chooses what `open` and `read` (for a model in memory) don't: what to keep, where
the files a model names are found, and its format.

```rust no_run
use polysmelt::{Format, Keep, Loader, assets::Folder};

fn main() -> polysmelt::Result<()> {
    let doc = Loader::new()
        .keep(Keep::ALL.with_animations(false))
        .assets(Folder::new("textures"))
        .format(Format::Obj)
        .open("scan.txt")?;
    let glb = doc.to_glb()?;
    Ok(())
}
```

`cargo run --example convert -- in.fbx out.glb` converts a model (to FBX for an `.fbx` output), and
`cargo run --example inspect -- in.fbx` prints what it holds.

## Status

Early. What exists today:

- **`meshopt`**: the `EXT_meshopt_compression` and `KHR_meshopt_compression` bitstreams,
  ported from meshoptimizer and byte for byte its output on its own test vectors: the
  attribute encoder (codec versions 0 and 1, levels 0 to 3), the triangle and index-sequence
  encoders, their size bounds, the filter encoders (octahedral, quaternion, exponential in all
  four exponent modes, color), and decoders for every mode and filter.
- **`gltf`**: `decompress` turns a meshopt-compressed GLB into a plain one (quantization is
  kept); `compress` does the reverse, losslessly, with `EXT_meshopt_compression`, and
  `compress_with` writes `KHR_meshopt_compression` instead (the version 1 attribute codec at a
  chosen level, and optionally the `COLOR` filter for 8- and 16-bit vertex colors, within 1
  per channel); `pack_glb` and `pack_glb_decompressed` turn a `.gltf` and its files into one
  GLB, with `files_in` as a fetch that stays inside the model's folder.
  `cargo run --example meshopt -- sample out.glb` writes a small test file.
- **`formats`**: readers for STL (binary and ASCII), PLY, MagicaVoxel `.vox`, OBJ with MTL
  materials and textures, FBX (binary and ASCII, line geometry and NURBS curves drawn as lines), 3DS, 3MF, USD (`.usda`, `.usdc`,
  `.usdz`, with basis, NURBS and Hermite curves and points), SketchUp `.skp` (SketchUp 3 to the latest, without the SketchUp SDK) and glTF 2.0
  (`.gltf`, `.glb`, meshopt- or Draco-compressed or not), and glTF 1.0 (with
  `KHR_binary_glTF` and `KHR_materials_common`), upgraded to 2.0 as it's read, BVH
  motion capture (the skeleton as nodes, the motion as one animation, no mesh), COLLADA
  (`.dae`, and the `.zae` and `.kmz` archives that hold one, lines and line strips
  included), VRM avatars and `.vrma` animations (VRM 0.x and 1.0, through the glTF reader),
  OFF (text and binary), X3D and VRML97 (XML and classic, gzipped or not, line and point
  sets, Extrusion, the 2D nodes and PROTO and EXTERNPROTO instances included) and DXF meshes, lines and points (3D faces, polyface and polygon meshes,
  MESH entities, lines, points, 2D and 3D polylines with their arcs, circles, arcs, ellipses
  and splines, and block inserts, text or binary), and gaussian splats (3D gaussian splatting's
  PLY files, `.splat` and gzipped `.spz`, versions 1 to 3) as `KHR_gaussian_splatting`, which
  the glTF reader also reads: `to_glb` keeps every splat's spherical harmonics, scale and
  rotation, and `flatten` draws the splats as points in their base color. Point clouds:
  LAS lidar (1.0 to 1.4, point formats 0 to 10) and LAZ, PCD (ascii, binary and
  binary_compressed), E57 scans (each scan placed by its pose) and text clouds (`.xyz`, `.pts`,
  `.ptx` with its scanner transforms, `.asc`, ESRI ASCII grids), as point primitives with their colors, the other
  fields (intensity, classification, return numbers, GPS time...) as vertex attributes, and the
  coordinate system in `extras`. AMF (plain or zipped: objects, volumes and their materials,
  vertex and triangle colors, textures, constellations; curved triangles stay flat), LightWave objects (`.lwo`: LWOB,
  LWO2 and LWO3, with layers and pivots, surfaces and their image maps, UV and color maps,
  smoothing angles and morph targets), AC3D (`.ac`: the object tree, materials, textures,
  crease angles, lines and triangle strips), OpenFlight (`.flt`: the node tree with levels of
  detail, switches, instances, replication and references to other files, faces and meshes,
  color, material and texture palettes with SGI images, light sources and animated groups),
  OpenGEX (`.ogex`: nodes and their transforms, meshes, materials and textures, skins, morph
  targets, lights, cameras and animation tracks), Irrlicht meshes and scenes (`.irrmesh`: buffers of
  each vertex type with their materials; `.irr`: the node tree, cubes, spheres, sky boxes and
  billboards, mesh nodes naming other files, lights, cameras and animators as animations) and XGL
  (`.xgl`, zipped `.zgl`: objects and their transforms, meshes placed by reference, materials,
  lines and points, a directional light). The XML formats read UTF-16 files too.
  Game and animation formats too: MikuMikuDance models (`.pmx` 2.0 and 2.1, `.pmd`, skinned,
  with morph targets, IK and materials) and `.vmd` motions, DirectX `.x` (text, binary and
  compressed, with frames, skins and animation sets), Quake II `.md2` and Quake III `.md3`
  (frames as morph targets, MD3 tags as nodes), Doom 3 `.md5mesh` and `.md5anim`, Inter-Quake
  Model `.iqm`, Valve `.smd` and `.vta`, MilkShape 3D `.ms3d` and Blitz3D `.b3d` (skinned and
  animated), OGRE `.mesh` and `.skeleton` (binary in either byte order and XML, with poses and
  morphs as morph targets and `.material` scripts as materials), and motion capture: Acclaim `.asf` skeletons with `.amc` motions, HTR, and C3D
  and TRC marker tracks (each marker an animated node). Left-handed and Z-up sources are turned
  to glTF's axes, and keys are kept as the file gives them.
  `open` reads a file and `read` a model already in memory; both go by the extension (or else
  the first bytes) and return a `Document`. `Format` names each format and its extensions.
- **`document`**: the common form, a glTF 2.0 document (its JSON and its buffers' bytes).
  glTF passes through untouched: nothing is decoded, and `Document::to_glb` writes the
  buffers back as they are, compression included, with images embedded. The other formats
  are converted into it. `Document::flatten(Pose)` derives a flat, posed triangle
  `mesh::Mesh` (transforms, instances, skins and morphs baked in). Skinned vertices are
  skinned by default (`Pose::Rest`, or `Pose::Animation` at a time), blending every
  `JOINTS_n`/`WEIGHTS_n` set; `Pose::Bind` leaves them as the file stores them, for a
  renderer that skins on the GPU, and is quicker. `Document::read_floats` and
  `Document::read_indices` read any accessor as floats or exact integers (sparse,
  normalized, interleaved or meshopt-compressed alike), and `Document::counts`
  counts triangles, points and lines from the JSON alone. `Document::decode` rewrites it
  as plain glTF any reader takes: meshopt views and Draco primitives decoded, quantized
  attributes turned into floats, the scene graph, animation and materials untouched. A
  `Keep` passed to the load says what to keep: everything by default, or for instance
  `Keep { animations: false, ..Keep::ALL }` for static meshes that still carry their skins.
  With `Keep::image_bytes` off (as in `Keep::LIGHT`), textures are found but not read:
  `Document::images` gives a handle on each image that says where it is (a file, an archive
  entry, inside the model) and loads it on request, and `Document::embed_images` reads
  them all (a 3DS bump map's image is read regardless, to tell a normal map from a height
  map). Against hostile or runaway files, a load refuses a model of more than
  `Keep::max_triangles` triangles (64 million by default, `mesh::MAX_TRIANGLES`), and
  `Document::flatten` draws no more; raise it for big scans, as in
  `Keep { max_triangles: 120_000_000, ..Keep::ALL }`. Points and lines (point clouds, wireframes,
  curves) are drawn too, up to `Keep::max_points` of them (16 million by default,
  `mesh::MAX_POINTS`); `Keep { lines_and_points: false, ..Keep::ALL }` leaves them out.
  `Keep::threads` reads large OBJ files on several threads (`0` for one per core) and inflates
  SPZ files on one thread while unpacking them on another, with the same result as one thread;
  it is `1`, one thread, by default.
- **FBX out** (`fbx`): `Document::to_fbx` and `Document::write_fbx` write binary FBX 7.4
  (7.5 when a file passes 4 GiB), the other format written in full besides GLB. The node tree
  goes out with its names, transforms and instancing (a mesh is one geometry, however many
  models draw it); meshes with every UV and color set, normals and tangents, per-polygon
  materials, and lines; PBR materials with their textures embedded, wrap modes, UV sets and
  texture transforms; skins as clusters with a bind pose; morph targets as blend shapes; each
  animation as its own take, rotations as Euler curves keyed closely enough to follow the
  quaternions; cameras, `KHR_lights_punctual` lights, and extras as user properties.
  Draco, meshopt, sparse and quantized data are decoded on the way. Points have no FBX form
  and are left out. The output is the same bytes every time.
- **Conformance corpus**: 1155 cases under `tests/corpus`, each a file and the GLB it must
  become. They cover every format above and the variants exporters really write: OBJ's face
  forms and MTL dialects, both PLY byte orders, every FBX version and axis system, and USD's
  three encodings and its composition, glTF's accessor types, node hierarchy, skins and extensions, BVH's rotation orders and
  damaged motion rows, and files SketchUp itself saved, from SketchUp 3 to 2026. Most are generated against the specifications; the rest
  are Blender 5.2 exports and SketchUp's own files. See [tests/corpus/README.md](tests/corpus/README.md). Every
  GLB the conformance tests write passes the Khronos glTF validator (`tools/validate`), which
  CI runs. The validator predates `KHR_gaussian_splatting` and calls that extension's attribute
  names invalid; `validate.mjs` lets that one error through and no other.
- **Benchmarks**: `bench/` races Polysmelt's readers against tobj, stl_io, ply-rs and gltf on load
  time and peak heap. Run it on a clean GitHub runner with `gh workflow run bench.yml` (or
  from the Actions tab); the tables land in the run summary.
- **`rig`**: typed views of a document's node tree, skins and animations, for code that
  plays or retargets motion rather than drawing triangles. `Document::rig` gives a `Rig`:
  nodes with names, parents and rest `Trs`, skins with their joints and inverse bind
  matrices, and animations as keyed channels (step, linear or cubic spline) that
  `Channel::sample` and `Rig::pose(animation, time)` evaluate; `Rig::poser` poses one
  animation time after time, stepping through each channel's keys rather than searching
  them. An FBX armature, a `UsdSkel` skeleton and a glTF skin all come out the same way.
  `Document::flatten_nodes` draws the mesh in a pose handed in (one `Trs` per node, from a
  retarget, say) rather than one played from the file.
- **`mesh`**: the flat model `flatten` produces (primitives, materials, images), written out
  as a plain GLB by `mesh::to_glb`. A primitive has triangles (`Prim::indices`), line
  segments (`Prim::lines`, strips and loops opened into pairs) and points (`Prim::points`)
  over one set of vertices, which carry positions,
  normals, tangents (`Prim::tangents`, xyz and the bitangent sign), the UVs its color texture
  reads (`Prim::uvs`, with `KHR_texture_transform` applied) and every other UV set the file
  has (`Prim::uv_sets`, by `TEXCOORD` number, for lightmaps, masks and decals), and vertex
  colors likewise (`Prim::colors` and `Prim::color_sets`); `to_glb` writes them all back. A
  `Material` has base color, metallic-roughness, normal, occlusion and emissive textures and
  factors, the normal map's scale and occlusion's strength, double-sidedness, alpha (blend
  from the color's alpha, or cut out at `alpha_cutoff`), `unlit`, and the other
  `KHR_materials_*` extensions (`Material::extensions`: ior, specular, transmission,
  clearcoat, sheen, volume, anisotropy, iridescence, dispersion) with their textures. Each
  texture's `Sampling` says which UVs it reads (`Prim::uvs`, or a set in `Prim::uv_sets`),
  the transform on them left after the color texture's, and its sampler's wrap and filters;
  only a transform that would shear relative to the color texture's is left off rather than
  drawn wrong. A skinned
  primitive keeps each
  vertex's most-weighted joint (`Prim::joints`) and its four strongest joints and weights
  (`Prim::skin_joints`, `Prim::skin_weights`), over the mesh's `skeleton` of bones; `to_glb`
  writes them back as a glTF skin, so a skinned model survives the round trip.
- **`assets`**: the `Assets` trait readers use to find and read referenced files (MTL
  libraries, textures, glTF buffers, USD layers) and turn texture bytes into embeddable
  images. `Folder` searches a pack's folder tree; `Memory` holds files by path, for models
  loaded from memory. PNG and JPEG embed as they are; WebP, TGA, BMP, GIF, TIFF, PSD and DDS
  textures are re-encoded as PNG, each behind its feature. Override `Assets::image` to convert
  other formats.

### Cargo features

| Feature | Format | Extra dependencies |
| --- | --- | --- |
| `stl` | STL | none |
| `ply` | PLY, 3D gaussian splatting's files included | none |
| `vox` | MagicaVoxel | none |
| `obj` | OBJ + MTL | none |
| `3ds` | 3DS | none |
| `gltf` | glTF 2.0 + GLB, glTF 1.0, and VRM `.vrm`/`.vrma` | none |
| `bvh` | BVH motion capture | none |
| `off` | OFF | none |
| `dxf` | DXF meshes, lines and points | none |
| `splat` | Gaussian splats in the 32-byte `.splat` layout | none |
| `spz` | Gaussian splats in SPZ, versions 1 to 3 (gzip) | `flate2` |
| `mmd` | MikuMikuDance `.pmx` (2.0 and 2.1) and `.pmd` models, `.vmd` motions | none |
| `directx` | DirectX `.x` (text, binary, MSZIP-compressed; 32- and 64-bit floats) | none |
| `md2` | Quake II `.md2`, frames as morph targets | none |
| `md3` | Quake III `.md3`, frames as morph targets, tags as nodes | none |
| `md5` | Doom 3 `.md5mesh` and `.md5anim` | none |
| `iqm` | Inter-Quake Model `.iqm` | none |
| `smd` | Valve `.smd` (reference and animation) and `.vta` | none |
| `ms3d` | MilkShape 3D `.ms3d` | none |
| `b3d` | Blitz3D `.b3d` | none |
| `asf` | Acclaim `.asf` skeletons and `.amc` motions | none |
| `c3d` | C3D motion capture markers | none |
| `trc` | TRC motion capture markers | none |
| `htr` | HTR motion capture | none |
| `fbx` | FBX | `flate2` |
| `3mf` | 3MF | `zip`, `quick-xml` |
| `usd` | USD | `openusd`, `openusd-schemas`, `zip` |
| `dae` | COLLADA `.dae`, `.zae` and `.kmz` | `zip`, `quick-xml` |
| `x3d` | X3D and VRML97 (`.x3d`, `.x3dv`, `.wrl`, gzipped `.x3dz`, `.wrz`) | `quick-xml`, `flate2` |
| `skp` | SketchUp `.skp` (through glTF, so it turns on `gltf`) | `flate2` |
| `draco` | Draco-compressed glTF (KHR_draco_mesh_compression) | `draco-core` |
| `webp` | WebP textures (EXT_texture_webp, `.webp` files), re-encoded as PNG for the flat mesh | `image-webp`, `flate2` |
| `tga` | TGA textures (color-mapped, true-color and gray, raw or run-length), re-encoded as PNG | `flate2` |
| `bmp` | BMP textures (1 to 32 bits, RLE4 and RLE8, bit-field masks), re-encoded as PNG | `flate2` |
| `gif` | GIF textures (the first frame), re-encoded as PNG | `flate2` |
| `tiff` | TIFF textures (strips or tiles; uncompressed, LZW, Deflate or PackBits; 1 to 16 bits), re-encoded as PNG | `flate2` |
| `psd` | Photoshop `.psd` and `.psb` textures (the merged image, raw or RLE), re-encoded as PNG | `flate2` |
| `dds` | DDS textures (BC1 to BC5 and uncompressed), re-encoded as PNG | `flate2` |
| `las` | LAS lidar point clouds, 1.0 to 1.4 | none |
| `laz` | LAZ, compressed LAS (turns on `las`) | `laz` |
| `pcd` | PCD point clouds | none |
| `xyz` | Text point clouds: `.xyz`, `.pts`, `.ptx`, `.asc` | none |
| `e57` | E57 scans | `e57` |
| `amf` | AMF, plain or zipped | `quick-xml`, `zip`, `flate2` |
| `lwo` | LightWave objects (`.lwo`) | none |
| `ac3d` | AC3D (`.ac`) | none |
| `ogre` | OGRE `.mesh` and `.skeleton`, binary (v1.10 to v1.100) and XML, with `.material` scripts | none |
| `flt` | OpenFlight (`.flt`), with SGI image textures | none |
| `ogex` | OpenGEX (`.ogex`) | none |
| `irr` | Irrlicht meshes (`.irrmesh`) and scenes (`.irr`) | none |
| `xgl` | XGL (`.xgl`) and zipped XGL (`.zgl`) | none |
| `fs` | Reading files: `open`, `Loader::open`, `assets::Folder`, `gltf::files_in` | none |

Every reader is on by default, and so are `fs`, `webp` and the six texture decoders. `minimal`
is the thirty-three readers with no dependencies, plus `fs`, for
`default-features = false, features = ["minimal"]`. Asking
`open` for a format whose feature is off returns an error that names the feature.

Without `fs` the file APIs don't exist, so a build that has no file system (such as
`wasm32-unknown-unknown`) fails to compile rather than failing when it runs. Models load from
memory instead:

```toml
polysmelt = { version = "0.1", default-features = false, features = ["stl", "ply", "obj", "gltf", "usd"] }
```

```rust no_run
use polysmelt::{Loader, assets::Memory, document::Pose};

fn show(gltf: Vec<u8>, bin: Vec<u8>, png: Vec<u8>) -> polysmelt::Result<()> {
    // The bytes are shared, not copied: the document keeps what it needs of them.
    let assets = Memory::new().with("scene.bin", bin).with("albedo.png", png);
    let mut doc = Loader::new().assets(assets).read(gltf, "scene.gltf")?;
    let glb = doc.to_glb()?; // the scene, losslessly
    let mesh = doc.flatten(Pose::Rest)?; // or flat triangles
    doc.decode()?; // or plain glTF, for a reader without Draco, meshopt or quantization
    Ok(())
}
```

### Minimum Rust

Rust 1.96, which `openusd` (the `usd` feature, on by default) needs. CI builds everything on
1.96.

## Planned scope

- **Lossless conversion**: every format is converted into the document with its hierarchy,
  animations, cameras and lights, and its source's units, axes and raw values in extras.
  Materials are written in full: the core model plus `KHR_materials_*` (ior, specular,
  transmission, volume, clearcoat, sheen, anisotropy, emissive strength, unlit) and
  `KHR_texture_transform`, from each format's own values. Readers skip what the `Keep`
  drops: texture files, material libraries and shaders, animation, skins and morphs; and
  with `image_bytes` off, they find textures without reading them.
- **Typed views** over the document's meshes and accessors (nodes, skins and animations
  have theirs in `rig`).
- **Compression**: writing `KHR_mesh_quantization`, with the filtered streams it enables
  (octahedral normals, quaternion rotations, exponential floats).
- **Scene**: animations read and written, and instancing kept rather than baked.
- **Writer**: a builder with plain output (any viewer, Blender) or compact output
  (quantized and meshopt-compressed).
- **Validation** of glTF documents.
- **Texture slots** (`KHR_texture_basisu`/KTX2), so a cook step can swap textures without
  touching geometry.

Out of scope: rendering, material semantics, and conversion through external tools such as
Blender.

## License

MIT-0. See [LICENSE](LICENSE). The SketchUp reader (`src/formats/skp`) is ported from
[openskp](https://github.com/iamahsanmehmood/openskp) and is MIT: see
[its license](src/formats/skp/LICENSE-openskp).
