# Changelog

Breaking changes are allowed in minor releases until 1.0.

## 0.19.0

- SuperSplat's compressed PLY files (`.compressed.ply`) are read as gaussian splats, with
  their spherical harmonics when the file has them. They come out as
  `KHR_gaussian_splatting`, the same as a plain 3D gaussian splatting PLY. This is part of
  the `ply` feature.
- New reader: PlayCanvas's SOG gaussian splats, versions 1 to 3, behind the new `sog`
  feature (on by default; it uses `zip` and `image-webp`). A `.sog` file opens directly, and
  an unzipped one opens by its `meta.json`, with the WebP images read from beside it.
  Spherical harmonics are kept up to degree 3; version 3's fourth band is dropped, its
  opacities above 1 are held to 1, and its blind data become custom attributes.
  `Format::Sog` names the format. A streamed SOG's `lod-meta.json` isn't read, but each of
  its chunks is.
- Format detection tells a SOG's `meta.json` and a zipped SOG from their content, so they
  open without the `.sog` extension.
- glTF: a node hidden or shown by an animation through `KHR_animation_pointer`
  (`/nodes/N/extensions/KHR_node_visibility/visible`) is drawn or left out at the time asked
  for. The flag holds between keys.
- glTF: a texture whose images are all in a format Polysmelt doesn't decode (KTX2 through
  `KHR_texture_basisu`, AVIF through `EXT_texture_avif`) is still not drawn, but is now
  counted in `asset.extras.source.droppedTextures`.
- glTF: files that require `KHR_node_selectability`, `KHR_node_hoverability`,
  `KHR_interactivity`, `EXT_structural_metadata`, `EXT_mesh_features`, `GODOT_single_root`,
  `CESIUM_primitive_outline` or `EXT_mesh_manifold` flatten instead of failing: none changes
  the triangles drawn (outlines aren't drawn).
- glTF 1.0: `WEB3D_quantized_attributes` accessors are decoded by their `decodeMatrix`, and
  `CESIUM_RTC`'s center is added to the root nodes. Both drew in the wrong place before.
- The glTF corpus gains 18 Khronos sample models: Draco, quantized and meshopt variants are
  checked triangle for triangle against their uncompressed ones.
- LightWave objects read without textures (`Keep` with `textures` or `materials` off) keep the
  texture coordinates an image map projects onto them, as a full read does. They were left
  out, though the mesh's other coordinates were kept.
- DDS textures: a BC3 alpha, BC4 or BC5 value between a block's two ends is the nearest
  value, as Direct3D defines it, where it was rounded down. Such texels come out up to
  1/255 brighter.
- VTK files are read, behind the new `vtk` feature (on by default): legacy `.vtk` (versions
  1.0 to 5.1, ASCII and binary) and XML `.vtp` and `.vtu` (text, base64 and appended data,
  either byte order, zlib-compressed or not, in any number of pieces). Poly data, unstructured
  grids, structured grids, images and rectilinear grids are drawn as triangles, lines and
  points. Volume cells are drawn as the mesh's boundary faces, and quadratic cells by their
  corner nodes; a structured dataset is its outer faces.
- VTK normals, texture coordinates and colors become `NORMAL`, `TEXCOORD_0` and `COLOR_0`.
  Every other point or cell array is kept as a custom attribute (`_PRESSURE`, `_VELOCITY`),
  and scalars with a lookup table in the file are colored through it.
- LZ4 and LZMA compressed VTK files, parallel files (`.pvtu`, `.pvtp`), `.vtm` and the XML
  image and grid files (`.vti`, `.vtr`, `.vts`) are refused with an error that says why.

## 0.18.4

- FBX export allocates less: key arrays are sized up front and the rotation curve fitting
  reuses one work list, about 1.1 times faster on animated characters. The bytes written are
  unchanged.
- AMF and 3MF read units with the same names as the other readers: 3MF takes spellings such
  as `mm`, `millimeters` or `metre` beside the specification's own, and both now know
  decimeters, kilometers, yards and the other lengths the rest of the crate reads.
- A 1-bit or 4-bit BMP whose pixel data is cut short is refused when padding it out would
  make a large image from a few bytes, as 24-bit ones already were.
- IQM files whose frames animate no channels are refused past 65,536 poses (frames times
  joints): such frames take no room in the file, so a tiny file could ask for 160 MB of them.
- Readers that look up textures by name (B3D, DirectX, FBX, IQM, Irrlicht, LightWave, MD2,
  MD3, MD5, MilkShape, OBJ, Ogre, PLY, SMD, 3DS, USD and X3D) share one lookup, so a texture
  named by many materials is found and read once. An OBJ texture that is also a material's
  gray roughness map is now read once rather than twice.
- The FBX reader and the glTF flattening are split into modules. No behavior changes.

## 0.18.3

- OFF files with face colors no longer take time that grows with the square of their size to
  read: a 490k-vertex colored mesh took 140 s and now takes 0.3 s. The output is unchanged.

## 0.18.2

- FBX export is faster: rotation curves are fit in one pass over the keys instead of searching
  them for every sample, and big arrays are deflated after the tree is built, reusing one
  compressor: about 1.3 times faster on the calling thread.
- `Document::to_fbx_with` and `Document::write_fbx_with` take `FbxOptions`, whose `threads`
  shares the rotation curves and the deflating across threads (0 for one per core): 2.4 to 4.7
  times faster than 0.18.1 on 22 threads for the animated characters measured. The bytes
  written are the same however many threads there are. `to_fbx` and `write_fbx` stay on the
  calling thread.
- FBX export samples rotations in double precision, so the Euler curves are a little closer
  to the quaternions.

## 0.18.1

- FBX export: a skinned mesh read from FBX is written back on its own model rather than on a
  second model beside it, so FBX read and written again keeps its nodes and joints instead of
  gaining one each time. When the mesh's geometry is offset from its model, the writer keeps the
  offset as a geometric transform.
- The XML readers (COLLADA, X3D, AMF, 3MF and SketchUp 2021+ materials) share one entity and character
  reference decoder. It's faster than any of the copies it replaces. AMF and 3MF now decode
  references in attribute values, 3MF metadata keeps its text around references and CDATA, and
  X3D and COLLADA no longer decode some text twice (`&amp;lt;` is `&lt;`). Numeric references
  with a sign, such as `&#+65;`, are left as written, as XML requires.

## 0.18.0

- FBX export: `Document::to_fbx` and `Document::write_fbx` (the `fbx` feature) write binary
  FBX 7.4, or 7.5 for files over 4 GiB, as the second fully supported output beside GLB.
  - Nodes: names, transforms and hierarchy; instanced meshes share one geometry, and each
    `EXT_mesh_gpu_instancing` instance becomes its own model.
  - Meshes: normals, tangents and binormals, every UV and color set, per-polygon materials,
    and line geometry. Points are left out, as FBX has no point geometry. Draco, meshopt,
    sparse and quantized data are decoded first.
  - Materials: Phong properties with the PBR values, alpha mode, double-sidedness and
    `KHR_materials_*` settings beside them; specular-glossiness materials as their metal/rough
    equivalent. Textures are embedded with their wrap modes, UV set and texture transform.
  - Skins become a skin deformer with a cluster per joint and a bind pose; morph targets become
    blend shapes with their weights.
  - Each animation is its own stack, layer and take. Stepped, linear and cubic keys stay as
    they are, and rotations are resampled as continuous Euler curves within 0.05 degrees of
    the quaternions.
  - Perspective and orthographic cameras, `KHR_lights_punctual` lights, and extras as user
    properties.
  - The output is deterministic. Every corpus model that loads round-trips through the reader,
    apart from three that differ only between the reader's own samples or by a material default.
- The FBX reader reads back what the writer adds beyond FBX's own properties: PBR values,
  alpha mode and the glTF extensions.
- FBX animation: key times less than a microsecond apart (one frame written to two curves
  through different float paths) are merged into one time, so no near-zero step reaches
  per-frame velocities.
- `examples/convert.rs` writes FBX when the output ends in `.fbx`.

## 0.17.8

- Polygons with holes (COLLADA `<ph>`, SketchUp inner loops) are triangulated by one shared
  routine. A hole touching the outline or another hole is bridged where it touches, and holes
  sharing a corner no longer drop most of the face, which fixes faces that came out overlapping
  or partly missing. COLLADA polygons are triangulated in their own plane, so tilted faces with
  holes are exact. Large faces with many holes triangulate up to twice as fast.

## 0.17.7

- Number parsing: every text reader (OBJ, OFF, COLLADA, 3MF, SMD, AC3D, DXF, AMF, MD5, `.x`,
  ASCII FBX, HTR, ASF, TRC, WKT) goes through one fast float and integer parser. Numbers written
  as `1.#INF`, `1.#QNAN` or `-1.#IND`, or with a decimal comma, are read instead of rejected.
  COLLADA single-precision lists round once, correctly; its index lists read about 20% faster.
- Internal: one quaternion module replaces the readers' own rotation math (normalizing,
  slerp, Euler and axis-angle conversion, matrix to rotation), and glTF matrices are converted
  to and from column order in one place. About 230 fewer lines. Output is unchanged.

## 0.17.6

- USDZ: a texture that inflates implausibly far (over 1 GiB, or 200 times its compressed size)
  is left out instead of being read into memory, as SketchUp archive entries already were.
- glTF 1.0: an accessor whose count runs past its buffer is an error before anything is
  allocated, and one with a count but no `bufferView` is reported as corrupt.
- A missing meshopt fallback buffer fails before room is made for the length it declares.
- One pixel limit (16384 × 16384) for every texture decoder; PNG, SketchUp and SGI images
  were capped at a quarter of that. A large SGI image decodes only when the file holds its data.
- Internal: the curve point budget, rotation resampling step, texture coordinate set and
  joint influence limits, and the inflate caps are each defined once. Output is unchanged.

## 0.17.5

- 3ds: a spotlight whose hotspot or falloff angle is NaN no longer panics; it takes the widest
  cone. Chunk lengths, counts and offsets past the end of the file take only what is there.
- LAS: offsets near the end of the address range no longer overflow; data that runs past the end
  of the file is reported as cut short.
- OGRE binary meshes: a vertex element shorter than its declared type reads as zero instead of
  indexing past it.
- Internal: one bounds-checked binary cursor replaces the readers' own copies (3ds, b3d,
  DirectX, IQM, LAS, MD2, MD3, MMD, MS3D, OFF, OGRE, OpenFlight, SketchUp). A count larger than
  the file can hold fails before anything is allocated. Output is unchanged.

## 0.17.4

- Writing a GLB is faster: accessors, buffer views and animations stay typed until the document
  is written, instead of being built as JSON and read back. Small animated models (IQM, B3D,
  `.x`, SMD, MD5, BVH) convert 6–29% faster, and the GLB step alone is up to 2.4× faster.
  Output is unchanged.
- `Document::json()` builds the full JSON on first use and keeps it.

## 0.17.3

- DirectX `.x` skins merge a joint listed twice for a vertex before keeping the strongest
  influences, so its weight isn't lost. Every reader that trims influences now keeps the same
  ones the glTF writer would.
- Base64 in AMF textures and ASCII FBX `Content` decodes as leniently as glTF data URIs: either
  alphabet, whitespace anywhere, and optional padding. COLLADA `data:` images that aren't base64
  keep their raw bytes.
- `base64` is a regular dependency rather than part of the `fbx` and `amf` features.
- Internal: one set of vector, polygon normal and sRGB helpers replaces the readers' own
  copies. Output is unchanged.

## 0.17.2

- `Loader::open_with` and `Loader::read_with` borrow an `&dyn Assets` for one read, so assets
  that aren't `Send`, `Sync` or `'static` work, and can be asked afterwards what was read.

## 0.17.1

- `Error::new(kind, message)` and `Error::caused(kind, message, source)` are public, so an
  `Assets` source or a `pack_glb` fetch of your own can return any `ErrorKind`.

## 0.17.0

This release reshapes the public API ahead of 1.0. Most code needs only the small changes
listed under Migration.

### Added

- `polysmelt::open(path)` and `polysmelt::read(bytes, name)` read any supported model into a
  `Document`.
- `Loader` sets the `Keep` options, the `Assets` source and the `Format` before a read, as in
  `Loader::new().keep(keep).assets(assets).open(path)`.
- `Format` lists the supported formats, with `from_extension`, `detect`, `extensions` and
  `name`.
- `Error` has `kind()`, `path()` and `format()`, and `source()` returns the underlying zip,
  XML, JSON or decoder error. `ErrorKind` has `UnknownFormat`, `FeatureDisabled`,
  `Unsupported`, `Corrupt`, `Truncated`, `Empty`, `MissingAsset`, `LimitExceeded`, `Io` and
  `Invalid`.
- `Document::to_glb_with(&CompressOptions)`, `Document::write_glb(writer)`,
  `Document::animation_bounds`, `Document::json()` and `json_mut()`, plus `Clone`, `Default`
  and `From<Mesh>`.
- `rig::Channel::sample_into` samples into a reused buffer without allocating.
- `CompressOptions` setters that chain: `CompressOptions::KHR.level(3).color_filter(true)`.
- `Keep` setters that chain: `Keep::ALL.with_animations(false)`.
- `meshopt::Filter`, `source::ParseAxisError`, and `FromStr`/`Display` for `Axis`.
- `PartialEq` on `Mesh`, `Prim` and `Material`, and `Default` on `UvTransform`.
- `examples/convert.rs` and `examples/inspect.rs`.

### Changed

- Rust 1.96 is now the minimum, because the `usd` feature (on by default) needs it.
- A `Document` owns its data and has no lifetime. Reads are eager, so the source file is not
  held open until the GLB is written.
- Error messages name the file that failed. Truncated data reads "is cut short" or "runs past
  the end of" in every reader.
- A file that is well formed but has nothing to convert is `ErrorKind::Empty`, not an error
  about corrupt data.
- `Rig::world` no longer panics on a bad parent index.

### Removed

- The `formats` module is private. Use `open`, `read`, `Loader` and `Format`.
- `Assets::reads_files`.
- The `From` impls on `Error` for `ParseIntError`, `ParseFloatError`, `TryFromIntError`,
  `TryFromSliceError`, `Utf8Error`, `serde_json::Error`, `zip::result::ZipError` and
  `quick_xml::Error`. `From<std::io::Error>` stays.
- The `thiserror` dependency.

### Migration

| 0.16 | 0.17 |
|---|---|
| `formats::open(p, keep)` | `polysmelt::open(p)`, or `Loader::new().keep(keep).open(p)` |
| `formats::load(path, ext, &assets, keep)` | `Loader::new().assets(assets).keep(keep).open(path)`, adding `.format(Format::X)` when the extension is wrong |
| `formats::load_bytes(bytes, name, ext, &assets, keep)` | `Loader::new().assets(assets).keep(keep).read(bytes, name)` |
| `formats::detect(b)` | `Format::detect(b)` |
| `Document<'a>`, `.into_owned()` | `Document` |
| `doc.json` | `doc.json()` / `doc.json_mut()` |
| `Document::empty(facts)` / `Document::from_mesh(m, facts)` | `Document::default()` / `Document::from(m)` |
| `Keep { animations: false, ..Keep::ALL }` | `Keep::ALL.with_animations(false)` |
| `Memory::get` returns `Option<&[u8]>` | returns `Option<&Bytes>`; `insert` and `with` take `impl Into<Bytes>` |
| `match err { Error::InvalidInput(msg) => .. }` | `match err.kind() { ErrorKind::Corrupt => .., _ => .. }`, with `err.to_string()` for the text |
| `Error::Io(e)` (a short read or invalid data is now `Truncated` or `Corrupt`) | `err.kind() == ErrorKind::Io`, and `err.source()` for the `io::Error` |
| `rig::Path`, `channel.path` | `rig::Property`, `channel.property` |
| `Axis::parse(s)` | `s.parse::<Axis>().ok()` |
| `MeshoptExtension::name()` | `as_str()` or `to_string()` |
| `mesh::to_glb(&m)` | `m.to_glb()` |
| `Prim::skinned()` | `Prim::is_skinned()` |
| `meshopt::apply_filter("OCTAHEDRAL", ..)` | `apply_filter(Filter::Octahedral, ..)` |
| `CompressOptions { level: 3, ..CompressOptions::KHR }` | `CompressOptions::KHR.level(3)` |

Outside the crate, a struct literal no longer compiles for `SourceInfo`, `Converted`,
`Extensions`, `Material` and `CompressOptions`. Start from `default()`, a constant or a
constructor such as `Material::plain`, then assign the fields. A `match` on `ErrorKind`,
`Property`, `Interpolation`, `MeshoptExtension`, `ExpMode`, `Filter` or `Location` needs a
`_ =>` arm.
