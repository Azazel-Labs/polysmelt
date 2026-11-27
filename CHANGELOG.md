# Changelog

Breaking changes are allowed in minor releases until 1.0.

## Unreleased

- OFF: in a text file whose colors are 0 to 255, a vertex with no color is white. It was
  scaled as if its white were one of those colors, to near black with an alpha of 0.004.
- PLY: a color channel of a 32-bit whole-number type (`int`, `uint`) holds 0 to 255, as the
  writers that use the type write it, so 128 is mid-gray where it was read as full intensity.
- PLY gaussian splats, plain and compressed, say what the reader does to them, as the other
  splat readers do: `upAxis` `-Y` and right-handed, both assumed, and `converted` `upAxis`.
- Point clouds (LAS, PCD, text clouds, E57, `.pnts`): a cloud whose points all have the same
  color keeps `COLOR_0`. The color was moved to the mesh's `extras.constant`, where nothing
  drew it.
- LAS: `converted` lists `units` only when the cloud's unit across is known. With a vertical
  unit alone (heights in feet under degrees, say) the heights are still made meters, but
  `output_meters_per_unit` no longer says the whole cloud is in meters.
- `.splat`: a file that isn't a whole number of 32-byte splats long is refused as
  `UnknownFormat`. Any bytes at all used to load, as no splats or with `trailingBytes` counted.
- X3D and VRML: a file must start as the formats say, with `#VRML V2.0` or `#X3D V3` (or
  later) for the classic encoding or an `<X3D>` root element for XML; anything else is
  `UnknownFormat`. Any text or XML used to load as an empty scene.
- X3D and VRML: a `SpotLight` that leaves its cone to the defaults gets its version's. VRML97
  and X3D up to 3.2 have a cut-off of Ï€/4 with a beam that fills it, X3D 3.3 a beam of Ï€/4
  in a cut-off of Ï€/2, and X3D 4 a beam of 3Ï€/16; every version got 3.3's.
- C3D and HTR: `converted` lists `upAxis` only when the file is turned. A file that is Y up
  already said it too.
- DirectX text: a value written as `nan`, `inf` or `infinity` (signed or not, `nan(ind)`
  too) is a number, read as 0 like `1.#QNAN`. It was taken for a string, so every number
  after it moved up one place and the mesh came out wrong. A signed infinity is 0 too.
- glTF 1.0: looking for a skin's joint ends when a node is listed among its own descendants,
  where it used to go round forever.
- Blitz3D: a skinned mesh whose node is moved, turned or scaled keeps that transform: the
  node stays where it is for the bones under it, and the mesh and skin hang from an unnamed
  child of it. The transform was dropped, bones and all.
- FBX: an animated component with no curve and no value on its curve node holds the node's
  rest value (or adds nothing on an additive layer). It was animated to 0.
- FBX: a material's transparency is `TransparentColor` scaled by `TransparencyFactor`, as
  the FBX SDK defines it, when it gives no `Opacity`. A factor of 1 with a black or absent
  color is opaque, where it made the material invisible; a factor below 1 without a color
  is the transparency, as before.
- FBX: an ASCII file older than FBX 7 that leaves out its first-line comment is `Unsupported`
  by its header's `FBXVersion`, as with the comment; it was `Empty`.
- 3DS: a keyframe whose value isn't a number is left out. It put `null` in the node's
  translation, rotation or scale.
- OpenGEX: a camera's `fov` is its horizontal field of view, as the format says. Given
  alone it is turned into glTF's vertical one across an assumed 16:9 frame, with that
  `aspectRatio` written beside it; it was written as `yfov` unchanged.
- Irrlicht scenes: a mesh file the scene shows that is past `Keep::max_triangles` fails the
  load as `LimitExceeded`. It was listed among the missing files and left out.
- MilkShape 3D: a position key moves its joint in the joint's rest frame, turned by the rest
  rotation before it is added to the rest position, as MilkShape plays it.
- VOX: glass is `KHR_materials_transmission` alone, opaque to alpha. It was also blended by
  an opacity from the same transparency, so a renderer with both thinned it twice.
- The same failure is the same `ErrorKind` in every reader:
  - A file with no bytes is `Empty`, whatever its extension. It was `UnknownFormat`, `Corrupt`,
    `Truncated`, `MissingAsset` or an empty model, by format.
  - A file that isn't the format it is read as is `UnknownFormat`: one without the format's
    signature (its magic bytes, or the first words of its text or XML). USD, E57, SOG, OGRE,
    Irrlicht, XGL, OpenGEX, PTX and 3D Tiles called such a file `Corrupt`, X3D and VRML read it as
    an empty model, and OBJ and the text point clouds called it `Empty`.
  - A file cut short is `Truncated`, or, where the reader keeps what came before the cut, a
    model whose `asset.extras.source.truncated` is `true`. A binary STL whose header counts more
    triangles than the file holds is `Truncated` (it was `UnknownFormat`); so are a zip archive
    with no end record under any zipped format (3MF, USDZ, ZAE, KMZ, SOG, AMF), glTF JSON that
    ends early, a PLY header with no `end_header`, a LAS or PCD header that stops, an E57 file
    shorter than its header says, an OpenGEX file that ends inside a structure, and XML that
    ends with an element open (Irrlicht, OGRE, XGL, VTK). 3DS, ASCII STL, COLLADA, X3D and VRML,
    PTX, C3D, BVH, HTR, TRC and PMD now set `truncated` on the partial model they return, and
    3DS, AMF, LightWave, AC3D, OpenFlight, COLLADA and X3D are `Truncated` rather than `Empty`
    when nothing whole came before the cut. OBJ, `.xyz`, `.pts` and `.asc` have no count and no
    end mark, so half of one still reads as a smaller model.
  - A well-formed file with nothing to convert is `Empty` in every reader: a document that
    draws nothing and has no node that is named, has children, or carries a camera, a light or
    a skin. glTF, COLLADA, USD, SketchUp, DXF, X3D, the point cloud and splat readers and
    3D Tiles used to return an empty document for one; a skeleton, a motion, a scene of only
    cameras or lights, or a library of meshes no node instances still loads.
  - glTF JSON that starts as JSON and doesn't parse is `Corrupt`; it was `UnknownFormat`.
- Every reader reports the same facts, spelled the same way:
  - The glTF 2.0 reader writes `asset.extras.source` as every other reader does: `format`,
    `version`, glTF's meters, +Y up, +Z forward and right-handedness (declared), the generator
    as `tool` and the copyright. A file that carries a source already (a model converted from
    another format, read back) keeps it.
  - `Document::facts` has `triangles`, and `points` and `lines` when there are any, from every
    reader, counted by `Document::counts`. VOX and BVH had none, and motion capture files now
    say `triangles: 0`.
  - `Fact::Declared` means the file states the value or its format fixes it for every file;
    `Fact::Assumed` means a default the file didn't override, or a guess. Now declared: the up
    axis and handedness of 3DS, MD2, MD3, MD5, IQM, VOX, AC3D, MilkShape 3D and glTF 1.0 (MD2,
    MD3 and MD5 reported no handedness), the up axis of AMF, E57, DXF, X3D and MikuMikuDance,
    and DirectX's handedness.
  - `unit` is always the unit's usual name by its length (`"millimeter"`), and absent when the
    length is unknown or no common unit's. The file's own word, where it differs, is
    `unitAsWritten`: COLLADA, C3D, TRC and HTR wrote the file's word as `unit` (`"mm"`). An AMF
    with a unit the format doesn't have reports the millimeters it was read as, assumed, where
    it reported `unit: "furlong"` beside `metersPerUnit: 0.001`.
  - An assumed `frameRate` is written only for a model with animation. USD and VOX wrote one
    for static files.
  - `encoding` is `"ascii"` or `"binary"` (STL, OFF, DXF, VTK, Gmsh and Medit wrote `"ASCII"`, X3D
    `"XML"`, DirectX `"text"`). PLY's byte order is `byteOrder` (`"little"`, `"big"`) beside
    `encoding: "binary"`, as OGRE's, VTK's, Gmsh's and Medit's are; DirectX's compression is `compressed`;
    MikuMikuDance's text encoding is `textEncoding`.
  - Textures a file names that the document doesn't have are counted as
    `source.droppedTextures` (OpenGEX wrote `missingTextures`), and the files a model refers to
    that weren't found are listed as `source.missingFiles` (OpenFlight wrote a count, Irrlicht a
    `missing_files` fact). OGRE's `missing_materials` and `missing_textures` facts stay facts:
    a read that keeps no materials can't know them.
  - LightWave reports `format: "LWO"` with the variant (`LWOB`, `LWO2`, `LWO3`) as `version`.
    USD's `truncated` is `true`, the reason beside it as `truncatedBy`.
  - MD2, MD3 and MD5 refuse versions other than 8, 15 and 10 as `Unsupported`, as their docs
    said they read only those.
- Names and text decode the same way in every reader: UTF-8 as it is (a byte order mark
  dropped, UTF-16 with a byte order mark decoded), anything else as Windows-1252, unless the
  format or the file states an encoding (an XML declaration naming ISO-8859-1 or Windows-1252
  is honored). Names that aren't UTF-8 used to come out as Latin-1 (OBJ, LightWave, AC3D,
  OpenFlight, DirectX, the game formats), with U+FFFD in them (3DS, FBX, STL, PLY, DXF, X3D,
  AMF, 3MF) or empty (COLLADA, VOX). Padding after a fixed-size name is dropped everywhere.
- A thing with no name has no `name` key: unnamed 3DS objects and unnamed materials (X3D,
  COLLADA) had `"name": ""`. Names a reader makes up for things a format numbers are the kind
  and the number together, in every reader: `material3`, `object1`, `frame2`, `model0` (they
  were `material 3`, `material_3`, `object_1`, `frame 2`, `model_0` by format).

## 0.20.0

- `Document::flatten` scales a `KHR_materials_volume` thickness by the transform it bakes
  into the vertices, as the extension has a node scale it: a thickness of 2.27 under a node
  scaled by 0.25 is 0.5675 in the flat mesh, where it used to stay 2.27. The scale is the
  cube root of the transform's volume change (the scale itself when uniform, the mean of the
  three otherwise). A material drawn at several scales becomes one material for each, and a
  skinned mesh takes the mean scale of its skin's joints. The attenuation distance is in
  world space and is left alone.
- 3MF: a translucent display property's thickness is written in the model's unit, the
  mesh's space, so it is as long as the attenuation distance once the unit's node scales
  it: 500 in a millimeter file for a distance of 0.5 m, where it was 0.5. The attenuation
  is per meter, as the materials extension gives it, whatever the model's unit.
- 3DS: a file saved on a frame after the first holds that frame's vertices and mesh matrix.
  Its mesh is now placed back through the mesh matrix, as 3D Studio places it, when the
  matrix is where the object stands at any of its keys, not only the first. Before, such a
  mesh kept the later frame's turn in every pose.
- New reader: OGC 3D Tiles, behind the new `tiles` feature (on by default, and part of
  `minimal`; it needs `gltf` and adds no dependency). `Format::Tiles` names the format. A
  tile is told by its first bytes, whatever its extension.
- `.b3dm` (Batched 3D Model): the embedded glTF, 2.0 or 1.0, is the document, with either
  legacy header layout read too. The batch table is kept in the extras of a node over the
  glTF: `batchLength`, and `batchTable` with a list of one value for each feature under each
  property's name, binary properties decoded. A `BATCHID` attribute takes the name
  `_BATCHID`.
- `.i3dm` (Instanced 3D Model): the instances become `EXT_mesh_gpu_instancing` on each node
  that draws a mesh, from positions (plain or quantized), `NORMAL_UP` and `NORMAL_RIGHT`
  (plain or oct-encoded), `SCALE`, `SCALE_NON_UNIFORM` and `EAST_NORTH_UP` on the WGS84
  ellipsoid. The glTF is embedded or found by its URI through the `Assets`. `BATCH_ID` is the
  `_BATCHID` instance attribute. Instances that would shear a turned node become nodes with
  matrices, and an instanced glTF's animations are dropped.
- `.pnts` (Point Cloud): positions (plain or quantized), `RGB`, `RGBA`, `RGB565` and
  `CONSTANT_RGBA` colors, `NORMAL` and `NORMAL_OCT16P`, `BATCH_ID` as `_BATCHID`, and
  per-point batch table properties as custom attributes. `3DTILES_draco_point_compression`
  is decoded with the `draco` feature, and refused with an error that names it without.
- `.cmpt` (Composite): its tiles under one node, composites inside composites too.
- `tileset.json`: the tileset at its finest level of detail (`REPLACE` and `ADD` refinement,
  external tilesets, several `contents`, glTF and GLB content as 3D Tiles 1.1 has it), each
  tile under a node with its transform. At most 1024 tiles are drawn: past that the coarsest
  are refined first and the `unrefined_tiles` fact is set. Content that isn't among the
  assets is left out and listed in `asset.extras.source.missingTiles`. Implicit tiling is
  refused with an error that says so.
- 3D Tiles is Z up and its glTF Y up: everything in tile space is turned Y up, and the source
  info says `+Z`, converted. An `RTC_CENTER`, a glTF's `CESIUM_RTC` center and a quantized
  volume's offset are never added to single-precision positions: they are node translations
  in double precision, so tiles on the globe keep their detail.
- Format detection tells the four tile formats by their magic and a `tileset.json` by its
  content, before taking it for a glTF.
- 3DS: an object whose parent in the keyframer is a camera, a light or the target one aims
  at is now under that node. A camera's or spotlight's node stands where its position track
  has it and faces its target's node, rolled as its roll track says, so its children are in
  the frame it aims, at rest and as the camera, its target or its roll moves (the turn is
  sampled on every frame). The camera or light is on that node. Before, the keyframer's
  hierarchy was its object nodes alone, and such an object landed under the root with its
  keys read as if in the scene's space. A camera or light nothing hangs under stays where
  the editor has it, as before. Nodes of every kind now count toward the ids of a file that
  has none.
- Gmsh meshes (`.msh`) are read, behind the new `msh` feature (on by default, no
  dependencies, part of `minimal`): versions 1.0, 2.0 to 2.2, 4.0 and 4.1, ASCII and binary in
  either byte order, with a 4-byte or 8-byte `size_t`. Every element type is drawn by its
  corner nodes: points, lines, triangles and quadrangles as they are, and tetrahedra,
  hexahedra, prisms and pyramids as the faces only one of them has. `Format::Gmsh` names the
  format.
- Each Gmsh physical group is a mesh and a node of its own, named by `$PhysicalNames` (or
  `Physical Surface 3` without a name), with its tag and dimension in the mesh's `extras`; a
  partitioned file's groups are read from `$PartitionedEntities`. Elements in no group are
  one mesh with no name. `$NodeData`, `$ElementData` and `$ElementNodeData` views become
  custom attributes (`_TEMPERATURE`), a view's first time step only.
- Medit (INRIA) meshes are read, behind the new `medit` feature (on by default, no
  dependencies, part of `minimal`): `.mesh` as text and `.meshb` as binary, versions 1 to 4 in
  either byte order. Vertices, edges, triangles, quadrilaterals, tetrahedra, hexahedra, prisms
  and pyramids are drawn, their higher-order forms by their corner nodes, and the elements of
  each reference are a mesh of their own (`Surface 3`, `Volume 1`). Normals become `NORMAL`
  when every vertex has one, and tangents a `_TANGENT` attribute. `Format::Medit` names the
  format.
- A `.mesh` file is read as a Medit mesh when it starts as one (`MeshVersionFormatted`, or the
  binary form's header) and as an OGRE mesh otherwise, as before. Format detection tells Gmsh
  and Medit files from their first bytes.

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
