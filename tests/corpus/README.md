# Conformance corpus

Every case here is a model file, whatever it refers to, and an `expect.json`. The
`conformance` test (`tests/conformance/`) loads each case with `formats::load` and writes the
result as a GLB. It validates that GLB as glTF 2.0, decodes it, and checks it against the
case's expectations.

```
cargo test --test conformance          # every format
cargo test --test conformance usd      # one format
```

A failing run lists every failed case with what was wrong, not just the first one. A build
without some readers (`--no-default-features --features minimal`) skips the formats it can't
read.

## Layout

```
tests/corpus/<format>/<case>/
    <model file>            e.g. cube.usdz, model.obj + model.mtl, tex/red.png
    expect.json
```

`corpus_covers_every_extension` checks that every extension in `formats::EXTENSIONS` has
cases.

## Where cases come from

| Format | Cases | Generated | Blender 5.2 exports |
| --- | --- | --- | --- |
| STL | 29 | 27 | 2 |
| PLY | 47 | 42 | 4 |
| VOX | 31 | 27 | none |
| OBJ | 67 | 58 | 9 |
| 3DS | 58 | 51 | none |
| 3MF | 63 | 55 | none |
| FBX | 136 | 118 | 18 |
| USD | 141 | 121 | 20 |
| glTF | 104 | 71 | 15 |
| SketchUp | 31 | 7 | none |
| BVH | 42 | 41 | none |
| COLLADA | 92 | 79 | none |
| OFF | 26 | 24 | none |
| VRM | 9 | 9 | none |
| X3D | 55 | 55 | none |
| DXF | 38 | 34 | none |
| Gaussian splats (`.splat`) | 8 | 8 | none |
| SPZ | 19 | 19 | none |
| SOG | 29 | 28 | none |
| MikuMikuDance (`mmd/`: PMX, PMD, VMD) | 23 | 23 | none |
| DirectX (`x/`) | 54 | 43 | none |
| MD2 | 14 | 13 | none |
| MD3 | 15 | 15 | none |
| MD5 | 14 | 13 | none |
| IQM | 16 | 16 | none |
| SMD | 11 | 9 | none |
| MilkShape 3D | 15 | 13 | none |
| Blitz3D | 17 | 16 | none |
| ASF/AMC (`asf/`) | 26 | 26 | none |
| C3D | 29 | 26 | none |
| TRC | 20 | 20 | none |
| HTR | 31 | 31 | none |
| LAS and LAZ | 17 | 15 | none |
| PCD | 25 | 23 | none |
| Text point clouds (`xyz/`) | 21 | 21 | none |
| E57 | 3 | 1 | none |
| AMF | 26 | 21 | none |
| LightWave (`lwo/`) | 34 | 22 | none |
| AC3D (`ac/`) | 18 | 13 | none |
| OGRE | 32 | 31 | none |
| OpenFlight | 21 | 21 | none |
| OpenGEX | 30 | 24 | none |
| Irrlicht (`irr/`: IRRMesh and scenes) | 30 | 27 | none |
| XGL (`xgl/`: XGL and ZGL) | 26 | 19 | none |
| VTK (`vtk/`: legacy and XML) | 86 | 72 | none |
| 3D Tiles (`tiles/`: b3dm, i3dm, pnts, cmpt, tileset.json) | 81 | 39 | none |
| Gmsh (`msh/`) | 88 | 86 | none |
| Medit (`medit/`: `.mesh` and `.meshb`) | 60 | 57 | none |

- **Generated** cases are written byte by byte by `tools/corpus/<format>.py` (3DS is
  `tds.py`, 3MF is `threemf.py`, COLLADA is `dae.py`, PMX, PMD and VMD are `mmd.py`, DirectX
  is `x.py`, ASF/AMC is `asf.py`). Each one targets one clause of the format's specification,
  or one variant that real exporters write. The expectations come from what the generator
  wrote, never from reading the file back with Polysmelt, so a case tests the reader against
  the standard rather than against itself.
- **Classic materials** (Phong, Blinn, Lambert) are expected by one rule, `common.py`'s `phong`:
  roughness `(2 / (n + 2))^(1/4)` for an exponent `n`, and the specular color as
  `KHR_materials_specular`. It is worked out there from the rule, apart from the readers. Ten
  formats each have a `same_phong` case holding the same material (`common.py`'s `SAME_PHONG`),
  and `tests/materials.rs` checks that all ten read as the same glTF material.
- **USD** expectations are computed with Pixar's `usd-core` (the `pxr` module). It composes
  the stage and measures it: triangles, bounds, visibility, purpose and transforms. It draws
  neither curves nor points, so their expectations are worked out in `usd.py`.
- **BVH** expectations come from numpy: each joint's rotation is the product of 3×3 axis
  rotation matrices in the order its channels list them, converted to a quaternion. The
  reader's own quaternion product isn't used.
- **SketchUp**'s format is closed, so its other 24 cases are files SketchUp saved, taken from
  openskp's test fixtures (MIT, `skp/LICENSE-openskp`). `tools/corpus/skp.py` copies them and
  writes their expectations from what openskp's tests assert about each file and what was
  drawn in it, and the application fact from each file's header.
- **DirectX**'s other 6 cases are real files: three cubes (text, binary and compressed
  binary) and two malformed files from the Open Asset Import Library's test models
  (BSD-3-Clause, `x/LICENSE-assimp`), and a cube kW Xport exported (MIT, its notice in
  `x/kwxport_cube/LICENSE.txt`). `tools/corpus/x.py` copies them and writes their
  expectations from what the files hold, read by hand.
- **ASF/AMC, HTR, C3D, SMD, MilkShape 3D, Blitz3D, IQM, MD3 and MD5** poses come from 3×3 or
  4×4 rotation matrices multiplied with numpy (or plain matrix code) in the order each format
  defines, converted to quaternions; the readers' own quaternion products aren't used.
  **MikuMikuDance** motions sampled along their Bézier curves are worked out in `mmd.py` by
  bisection, apart from the reader's solver.
- **Compressed PLY and SOG** cases are written by `tools/corpus/packed_splats.py`, WebP
  images included, from PlayCanvas's SOG specification and, for what it leaves out, its
  splat-transform source. It returns each splat as its quantized form decodes by the format's
  own rule, and the expectations are those. The real SOG's come from its images as ffmpeg (or
  Pillow) decodes them.
- **VTK** cases are written by `tools/corpus/vtkfile.py` (`vtk.py` would hide the module it
  imports) with VTK's own writers and meshio's, in every encoding each has, and a few by hand. It
  needs `pip install vtk meshio`. The expectations come from VTK: each file is read back with
  VTK's readers, and what's drawn is worked out from VTK's cells and their faces, the boundary
  being the faces only one cell has; that count is held against `vtkDataSetSurfaceFilter`'s.
  Its other 14 cases are real files from VTK's test data (BSD, `vtk/LICENSE-VTK`) and meshio's
  test meshes (MIT, `vtk/LICENSE-meshio`), measured the same way. Each case also has a
  `reference.json` of hashes of the exact points, triangles and attributes VTK reads, which
  `tests/vtk.rs` compares with Polysmelt's.
- **3D Tiles**' other 42 cases (`cesium_*` folders) are real tiles from CesiumJS's test data
  (CesiumGS/cesium, `Specs/Data/Cesium3DTiles`, at the commit `tools/corpus/tiles.py` names;
  Apache-2.0, `tiles/LICENSE-cesium`), unchanged. The generator is a reference decoder written
  from the specification: it reads each tile's header, tables and glTF, places the instances,
  points and tiles, and writes the expectations from that, for the real tiles and the ones it
  writes alike. Each case also has a `reference.json` of what `tests/tiles.rs` compares
  exactly: hashes of the positions as the file's single-precision values, of the colors as the
  levels the file stores and of each attribute, the node translation the file's center gives,
  each instance's transform, the batch table decoded, and each tile's placement in a tileset.
  Draco data isn't decoded there: the three Draco tiles hold the same clouds as uncompressed
  tiles beside them, and are held to those, point for point.
- **Gmsh** cases are written by `tools/corpus/msh.py` (`gmsh.py` would hide the module it imports)
  with Gmsh's own writer in every version and encoding it has, with meshio's, and with small
  writers of its own for the rest (big-endian files, a 4-byte `size_t`, binary 4.0, versions 2.0
  and 2.1, parametric nodes, sparse tags). It needs `pip install gmsh meshio vtk`. The
  expectations come from Gmsh: each file is read back with `gmsh.open`, which gives the nodes,
  the elements by entity, each entity's physical groups and the views; where meshio reads the
  file too, its points give the nodes' order and its cells must agree. A file the generator wrote
  itself is expected to hold what was written, and Gmsh must agree where it reads the file. What's
  drawn is worked out with VTK in `tools/corpus/cellmesh.py`: each group's boundary is the faces
  only one of its elements has, a count held against `vtkDataSetSurfaceFilter`'s. Its other 2
  cases are real files from meshio's test meshes (MIT, `msh/LICENSE-meshio`). Each case also has
  a `reference.json` of hashes of the exact points and of every triangle, line and point drawn
  with its attributes, mesh by mesh, which `tests/fem.rs` compares with Polysmelt's.
- **Medit** cases are written by `tools/corpus/medit.py` with Gmsh (text), meshio (text, and
  binary versions 3 and 4) and writers of its own (binary versions 1 and 2, big-endian files,
  two dimensions, the higher-order keywords, normals and tangents). The expectations come from
  meshio reading each file back, or from what the generator wrote where meshio doesn't read
  it, drawn with VTK the same way and compared by `tests/fem.rs` through each case's
  `reference.json`. Its other 3 cases are real files from meshio's test meshes (MIT,
  `medit/LICENSE-meshio`).
- **Blender** cases (`blender_*` folders) are real exporter output, from
  `tools/corpus/blender_export.py`.
- **Khronos samples** (glTF's other 18 cases, `khronos_*` folders) are models from
  KhronosGroup/glTF-Sample-Assets at a pinned commit, CC0 or CC-BY 4.0 only
  (`gltf/LICENSE-khronos-samples.md` lists each one's license, credit and what was changed:
  textures over 64 pixels are scaled down). `tools/corpus/khronos.py` copies them and writes
  their expectations from the uncompressed variant, measured by its own small reader. A
  compressed or quantized variant is `same_as` the plain one. Samples too large for the
  corpus are read by the `khronos_large_samples` test from `target/khronos-samples` (or
  `POLYSMELT_KHRONOS_SAMPLES`) when `python tools/corpus/khronos.py --large` has fetched them
  there, and passed over otherwise.
- **Real files** from other projects' test data sit in their case folders, with the
  project's license beside them: PDAL's `simple.las` and `simple.laz` (BSD,
  `las/LICENSE-PDAL`), PCL's `bun0.pcd` and `car6.pcd` (BSD, `pcd/LICENSE-PCL`), two scans
  from the e57 crate's test data (MIT, `e57/LICENSE-e57`), and assimp's AMF, LWO, AC,
  OpenGEX, IRRMesh, XGL and 3DS test models (BSD, `LICENSE-assimp` in each), and two gaussian
  splat scenes from the PlayCanvas engine's examples: the first two chunks of
  `guitar.compressed.ply` (MIT, `ply/LICENSE-playcanvas`) and a level of detail of PLAYBOT by
  Stephane Agullo as an unzipped SOG (CC-BY-4.0, `sog/LICENSE-playbot.txt`). The generator reads each file with its own small
  parser, written from the specification, and writes the expectations from that, so these
  too are checked against the standard rather than against Polysmelt. A generator keeps
  these files when it empties its folder (or takes them from the folder given as its first
  argument).

- **Reference cases** (`ref_*` folders, 75 of them across 21 formats) are files from other
  projects, measured by other tools: `tools/corpus/thirdparty.py` fetches each at a pinned
  commit and writes its expectations from an independent reader, never from Polysmelt and
  never from a parser of this project's. assimp 6.0.5 (through `assimp-py`) measures its own
  test models (BSD, `LICENSE-assimp` in each format's folder; a model with terms of its own
  has them beside it) for 3DS, COLLADA, DirectX, LightWave, AC3D, AMF, XGL, IRRMesh,
  OpenGEX, OGRE, OFF, MD2, MD5, SMD, MilkShape 3D, Blitz3D, DXF and BVH. lib3mf measures the
  3MF Consortium's samples (BSD, `3mf/LICENSE-3mf-samples`), py-vox-io MagicaVoxel's
  author's models (MIT, `vox/LICENSE-voxel-model`), and ezdxf two of assimp's DXF files that
  hold lines and points. C3D has no sample files with clear terms, so its three are written
  by ezc3d and read back with it. Each case compares what's drawn (triangles, bounds, area and
  the set of positions) and, where the file has them, node names, where each bone is at rest
  and at a key, and the animations' spans. The generator states how each tool's space maps to
  glTF's, and a case's `source` names what the tool gets wrong about the file, which is then
  left out. The table's Cases column counts them; its Generated column doesn't.

Regenerate a format from the repository root:

```
python tools/corpus/obj.py
"<blender.exe>" -b --factory-startup --python-exit-code 1 --python tools/corpus/blender_export.py -- obj
```

`usd.py` needs `pip install usd-core`. A generator replaces its own cases and leaves the
`blender_*` folders alone, and the reverse is also true. Each one finds the modules it shares
with the others (`common.py`, `packed_splats.py`) beside itself, so `python -I` runs it too.

Running a generator must write the corpus that is checked in, byte for byte.
`tools/corpus/check.py` checks that: it runs each generator on a copy of the corpus in a
scratch folder (never on `tests/corpus`) and lists every file that comes out different,
missing or new.

```
python tools/corpus/check.py                  # every generator
python tools/corpus/check.py obj gltf         # only these
python tools/corpus/check.py --path <dir>     # with packages from `pip install --target <dir>`
python tools/corpus/check.py --blender "<blender.exe>" --network
```

It runs the generators as `python -I`, which leaves out the environment and the user's site
packages; `--path` (repeatable) names folders of packages for them. A generator whose needs
aren't met is passed over, with the reason: a Python package that isn't installed (numpy,
Pillow, usd-core, vtk, meshio), Blender (`blender_export.py`, unless `--blender` names one or
`blender` is on the path), or the network (`khronos.py`, unless `--network` is given). The
exit code is 1 when a generator fails or writes something else. When an expectation changes
(a reader was fixed), change the generator and regenerate, so the two stay in step. Blender
writes what its version writes, so `blender_export.py` can differ under another one.

## What the cases cover

- **STL**: binary and ASCII. Covers headers that start with `solid`, truncation, trailing
  bytes, VisCAM and Magics colors, NaN and degenerate facets, and normals that are wrong or
  zero. Also several solids in one file, CRLF line endings and non-UTF-8 names, all turned
  from Z up to Y up.
- **PLY**: ASCII and binary in both byte orders, and every property type. Covers vertex
  colors in each encoding, the various UV property names, n-gons and triangle strips, point
  clouds and gaussian splats (read as `KHR_gaussian_splatting`, with spherical harmonics of
  degree 1). Also unknown elements, broken headers, and data shorter than the header says,
  which keeps the whole faces it holds. SuperSplat's compressed PLY: chunks with and without
  color bounds, a part chunk, an `sh` element of 1 and 3 bands, and a real file's first two
  chunks. Errors there: truncation, too few chunks, a count past `Keep::max_points`.
- **VOX**: palettes (default and off-by-one), materials, and the scene graph: transforms,
  rotations, mirrors, groups, instances and hidden layers or nodes. Also version 150 packs
  and unknown chunks.
- **OBJ**: every face-corner form, negative indices and n-gons. Covers MTL colors, PBR
  extensions, texture maps and their options, Windows paths and missing libraries, and TGA,
  BMP, GIF, TIFF, PSD and DDS textures re-encoded as PNG. Roughness, metalness and occlusion
  maps glTF can't read as they are get packed into its channels: `map_RMA`, separate `map_Pr`
  and `map_Pm` (with `-imfchan`), `map_Pm` alone, and maps laid out apart, which can't be. A
  gray `map_Pr` alone serves as it is, and `anisor` rotates the anisotropy. Also
  vertex colors and syntax variants: line continuation, `w` components, number forms and
  non-UTF-8 text.
- **3DS**: the master scale (1, which scales nothing, and 12), smoothing groups, materials (24-bit and float colors, transparency, shininess,
  self-illumination), textures and two-sided faces. The bump map is the normal map when its
  image is one (by its pixels, a gray image's header, or its name), scaled by its amount; a
  height map, the shininess map and the reflection map stay in `extras`. Also cameras, point and spot lights, and
  the keyframer: its hierarchy, pivots, dummies, instances and tracks; a scene of only a
  camera and a light. A real file has a box whose parent is a camera's node, placed in the
  frame the camera aims: its mesh matrix says where that is. Also truncated and
  malformed chunks.
- **3MF**: units, build items and components with transforms (including mirrored ones), the
  production extension, and colors from base materials, color groups and gradients.
  Textures, plus the files Bambu and Prusa write. Beam lattices: butt, sphere and
  hemisphere caps, tapered beams, balls, beam sets, colors blended along a beam, clipping
  inside and outside a mesh, and beams beside triangles. A build of only a volumetric level
  set fails with its own error. Also packaging variants and broken packages.
- **FBX**: ASCII 7300–7500 and binary 7400–7700 (zlib-compressed arrays too); FBX 6.1, in either
  encoding, is refused as unsupported. Covers
  axis systems and units, every rotation order, pivots, offsets and geometric transforms, and
  animation, including Euler keys baked past ±180° or flipped past straight up.
  Also every normal, UV and color mapping, materials, textures (embedded, external and
  missing), and instancing. Line geometry is drawn as lines (polylines through `PointsIndex`,
  broken where an index isn't a point, in the model's material or the line's color, its own
  or its template's). NURBS curves are drawn too: open, closed, rational and periodic ones
  along the curve, and one whose knots don't fit through its control points. NURBS surfaces,
  and curves with no order, are counted as skipped.
- **USD**: `.usda`, `.usdc`, `.usd` in either encoding, and `.usdz`. Covers up axis and
  `metersPerUnit`, every xformOp and rotation order, holes and invalid faces,
  `leftHanded` orientation with authored and computed normals (mirrored too), and
  `primvars:normals` versus `normals`. Also UV primvar names, display color and opacity, and
  composition: references, payloads, sublayers, variants, instancing, classes and overs.
  Also visibility and purpose, UsdPreviewSurface, textures (beside the file, in a subfolder,
  inside a `.usdz`, missing), GeomSubsets, gprims, point instancers and time samples.
  BasisCurves are drawn as lines (linear, periodic, and cubic Bezier, B-spline and pinned
  Catmull-Rom sampled along the curve, with colors per curve and per segment end), as are
  NurbsCurves (rational, cut to their ranges) and HermiteCurves (along their tangents, colors
  per point), and either drawn through its points when its knots or tangents don't fit it;
  Points as points with colors per point.
- **glTF**: GLB, `.gltf` with an external buffer (in a subfolder, percent-encoded) or a data
  URI, and a byte order mark. Covers u8, u16 and u32 indices, missing indices and normals,
  interleaved views, sparse accessors (over a view and over zeros), `KHR_mesh_quantization`
  (integer and normalized types), strips and fans, points and lines. Also the node hierarchy
  (TRS and matrices, mirrors, the default scene, no scenes, no nodes), skins and morph
  targets at rest, skins posed by an animation, eight influences over two
  `JOINTS_n`/`WEIGHTS_n` sets and weights only in the second set, an animation sampler
  shorter than its keys, `EXT_mesh_gpu_instancing`, metallic-roughness and specular-glossiness
  materials, and textures (embedded, external, data URI, missing, WebP fallback, a second UV
  set, `KHR_texture_transform` with a sheared transform). Also every texture slot with its
  own sampler, scale and strength, material extensions carried to the flat mesh, an empty
  buffer with no URI, and a GLB whose buffer is an external file. glTF 1.0, upgraded as it's read: objects keyed by ID,
  binary glTF, `KHR_materials_common` and technique materials, accessor strides, a node's
  several meshes, skins found by joint name with a bind shape matrix, animation parameters,
  cameras and lights. Errors: Draco, truncation, buffers named by absolute path, and
  accessors that overrun their views. Textures only in KTX2 or AVIF, counted as dropped, and
  one only in DDS (`MSFT_texture_dds`). Khronos samples: Avocado and Lantern plain, quantized
  and Draco-compressed, WaterBottle plain and Draco-compressed, MeshoptCubeTest against its
  fallback buffer, SimpleInstancing, NodeVisibilityTest, CubeVisibility (a node an animation
  pointer shows and hides), TextureTransformTest, TextureTransformMultiTest, UnlitTest and
  SpecGlossVsMetalRough.
- **SketchUp**: every layout from SketchUp 3 to 2026: the legacy object stream (3 to 6,
  2013 to 2020) and the ZIP container of 2021 on. Covers faces bounded by arcs beside a loose
  arc with no face, textured materials stored inline, groups and components nested inside
  each other, an instance scaled through its transform's last value, tags carrying line
  styles (2019 and 2020 layouts), and files with nothing to draw (empty, guides only, materials only), which are `Empty` errors. Also annotations, scenes
  and an SDK-saved file beside what is drawn, and truncated, foreign and too-old files.
- **BVH**: a skeleton and its motion, with no mesh. Covers six channels on every joint with
  positions and rotations interleaved, the ZXY, XYZ, YXZ and ZYX rotation orders, position
  channels on some axes only, scale channels, joints with no channels, unknown channel names,
  channel names in other spellings (`Xposition`, `XTRANSLATION`, `Xscaling`) and several
  ROOTs. Also End Site spellings, CRLF, bare CR, a byte order mark, tabs, blank lines,
  scientific notation and `-0`, and names with spaces and non-ASCII letters. Motion edge
  cases: no MOTION section, `Frames: 0`, a header in either order and any case, `=` in
  place of `:`, no `Frames` line (every row read), no `Frame Time` line (30 per second), and
  more or fewer rows than declared. Rows can be too long, too short (dropped, the others keeping
  their times) or cut off by the end of the file. A 25-joint humanoid has 480 frames,
  sampled between keys too. Errors: a missing brace, a JOINT outside any ROOT, a short
  OFFSET, NaN and junk values, a zero frame time, a fractional frame count, and files that
  aren't BVH.
- **COLLADA** (`.dae`, `.zae`, `.kmz`): 1.4.1 and 1.5.0. Covers every triangle-drawing
  primitive (`<triangles>`, `<polylist>` with a concave n-gon, `<polygons>` with one and two
  holes, `<trifans>`, `<tristrips>`), `<lines>` and `<linestrips>` (beside triangles, with vertex colors), inputs sharing an offset, accessors with a stride, an
  offset and an unnamed param, vertex colors and a second texture coordinate set chosen by
  `<bind_vertex_input>`, and positions that come straight from a source or from no input
  called POSITION. The scene: every transform element, `<instance_node>` (shared, in a cycle
  and straight under the scene), nested nodes, Z up in centimeters and X up in inches, a
  node's own `<asset>` changing the unit and up axis below it, and documents with no
  `<scene>`. Other documents: geometry, nodes and lights instanced from other `.dae` files in
  other units, with their textures, and one that's missing. Materials: phong, lambert and an
  unlit constant, the A_ONE, RGB_ZERO and A_ZERO transparency modes, double-sided extras on
  the effect and on the geometry, textures through a 1.4 surface and sampler, a 1.5
  `<instance_image>`, `<setparam>` overriding an effect's colors and textures, an image named
  straight, percent-encoded and `file://` paths, embedded hex, a missing file, a bump map, a
  height map, texture placement extras, and specular color and index of refraction. Skins
  with six and twenty influences, a bind shape matrix and IDREF joints, NORMALIZED and
  RELATIVE morphs and animated morph weights, morph targets' normals, animations of whole matrices, components,
  angles and matrix elements with LINEAR, STEP, 2D Bezier, BSPLINE and CARDINAL curves,
  nested animations and clips, cameras and lights in centimeters, and `<extra>` kept as
  extras. Also `.zae` with and without a manifest, a `.kmz`, byte order marks, UTF-16, a
  cut-off file, bad indices, counts far beyond the data, dangling URLs, and files that aren't
  COLLADA.
- **OFF**: text and `OFF BINARY`, and every header prefix (`ST`, `C`, `N`, `4`, `n`): texture
  coordinates, vertex colors as integers or floats, normals, homogeneous coordinates and a
  dimension of 2. Covers n-gons and concave faces, face colors (integer, float or a color map
  index), counts on the header line, comments, CRLF, bare CR and a byte order mark. Also faces
  out of range, truncation, and files that are empty, have no faces or aren't OFF.
- **VRM**: VRM 1.0 avatars drawn by their MToon materials' fallback, with their humanoid and
  spring bones kept, and `extensionsRequired` that names VRM's extensions. VRM 0.x avatars,
  turned half about Y to face +Z, with or without a `specVersion`. A `.vrma` animation posing
  a humanoid, and a GLB cut short.
- **X3D**: the XML encoding (`.x3d`, UTF-8 or UTF-16), the classic encoding (`.x3dv`) and
  VRML97 (`.wrl`), each gzipped too (`.x3dz`, `.wrz`, or a gzipped `.wrl`). Covers transforms
  (with a center), DEF and USE, `containerField`, Inline (several URLs), Switch and LOD,
  IndexedFaceSet (`ccw`, `solid`, per-face colors, separate indices), the triangle and quad
  sets, ElevationGrid and the primitives, and IndexedLineSet, LineSet and PointSet (colors per
  vertex, per polyline and by `colorIndex`, unlit in their emissive color). Extrusion (its
  defaults, a straight spine, coincident spine points with scale and orientation, a closed
  spine and section without caps, textured and uncapped), the filled 2D nodes (Rectangle2D,
  Disk2D, ArcClose2D, TriangleSet2D, double-sided unless `solid`) and the line and point ones
  (Circle2D, Arc2D, Polyline2D, Polypoint2D). Prototypes: PROTO instances with fields given or
  defaulted through IS, an MFNode field and a prototype in another's body, EXTERNPROTO from
  another file trying each URL, and the XML ProtoDeclare, ProtoInstance (with a node-valued
  `fieldValue`) and ExternProtoDeclare. Also Material, PhysicalMaterial, transparency and
  textures (in a subfolder, missing), UNIT and unit statements, viewpoints and lights, and
  nodes that draw nothing (an unused prototype, sensors, interpolators, scripts, routes).
  Errors: VRML 1.0, empty files and broken gzip.
- **DXF**: text and binary. Covers 3DFACE, POLYLINE polyface and polygon meshes (open and
  closed), MESH at its control mesh, LINE and POINT, LWPOLYLINE with bulges (arcs) at an
  elevation, 2D and 3D POLYLINEs, lines and points in blocks, and INSERT by base point, scale, rotation, extrusion and
  arrays, nested, cyclic and missing blocks. Also CIRCLE and ARC (in an extrusion's object
  coordinate system too), whole and partial ELLIPSE, and SPLINE as a NURBS curve (rational
  too), by fit points in a block, and with knots that don't fit it, drawn through its control
  points; text and binary. Also colors by layer, index, true color and
  block, layers turned off or frozen, paper space, `$INSUNITS`, CRLF, bare CR and a byte order
  mark, skipped entities and truncation. Errors: empty files and files that aren't DXF.
- **Gaussian splats** (`splat/`): the 32-byte `.splat` layout, its axes flipped to glTF's
  and its base color and opacity drawn as point colors, and a splat at an infinite position.
  Errors: files that aren't a whole number of splats long (trailing bytes, an empty file,
  text). Also glTF files that already use `KHR_gaussian_splatting`:
  drawn in their base spherical harmonics' color in either color space, or in a COLOR_0
  they give.
- **SPZ**: versions 1 to 3 (half-float and fixed-point positions, three-byte and
  smallest-three rotations), spherical harmonics of degrees 1 to 3 and degree 4 dropped, the
  antialiased flag, and the coordinate system record (axes flipped, or turned on the node)
  among other records or without the flag that says records follow. Errors: truncation, a
  count past `Keep::max_points`, version 4, an unknown version, an unzipped stream.
- **SOG**: versions 1 to 3, zipped and as a `meta.json` beside its images. Covers bounds and
  codebooks, the smallest-three and exponential-map rotations, each opacity encoding
  (version 3's above 1 held to 1), harmonics of 1 to 3 bands in a palette (a fourth band
  dropped, and bands told by the palette image's width where the meta doesn't state them),
  version 3's blind data as custom attributes, and its model, extras and comments; one real
  scene of splat-transform's. Errors: missing images, images that aren't WebP or are cut
  short, a count past the images' pixels or `Keep::max_points`, a short codebook, a palette
  image of the wrong width, an unknown version, a broken `meta.json`, a zip without one and a
  zip cut short.
- **MikuMikuDance** (`mmd/`): PMX 2.0 with UTF-16 and UTF-8 text and index sizes 1, 2 and 4,
  and PMX 2.1 with every deform (BDEF1, BDEF2, BDEF4, SDEF, QDEF), additional UVs, vertex
  colors and a translucent material. Covers bones with IK and inherited rotation, vertex,
  group and bone morphs (named targets), UV, material, flip and impulse morphs kept as data,
  and PMD with its sphere maps, toon list, English
  names and controlled IK. VMD motions with keys out of order, straight and bent Bézier spans,
  two keys on one frame, morph and camera keys, and the old header. Errors: wrong magic,
  version, encoding and index size, huge and negative counts, and truncation.
- **DirectX** (`x/`): text, binary and MSZIP-compressed files with 32- and 64-bit floats.
  Covers the frame hierarchy (mirrored frames too), meshes with n-gons, several meshes in a
  frame and instances, normals per face corner and per vertex, two UV sets, vertex colors,
  `DeclData` and `FVFData`, materials and textures (missing ones too), skins (partial, and a
  bone with no frame), and animation sets with rotation, scale, translation and matrix keys.
  Errors: unknown encodings, broken tokens, deep nesting, huge counts and truncation.
- **MD2** and **MD3**: frames as morph targets with their weights animations, frame names
  grouped into actions, unnamed frames, a single frame, skins and shaders found by name (and
  MD3's `_default.skin`), and MD3 tags as animated nodes. Errors: offsets and indices out of
  range, negative and huge counts, missing signatures and truncation.
- **MD5**: a skinned `.md5mesh` with weights in joint space and its texture by the shader's
  name, and `.md5anim` motions alone, without a frame rate and with short base frames.
  Errors: indices out of range, too few components, numbers that aren't, and truncation.
- **IQM**: static and skinned meshes with several number formats (half floats, doubles,
  integer fractions), two UV and color sets, joints and animated poses.
  Errors: version, names outside the text, joints out of order, mismatched poses and frames,
  and truncation.
- **SMD**: a skinned reference file, an animation-only file, links with weights, and `.vta`
  vertex animation. Errors: bad vertex lines, cut-short triangles and files that aren't SMD.
- **MilkShape 3D** and **Blitz3D**: groups and materials, textures, skeletons with keys, and
  MilkShape's extra weights and comments; Blitz3D's brushes, vertex colors and node hierarchy.
  Errors: indices out of range, chunks too big or nested too deep, and truncation.
- **ASF/AMC**: skeletons in degrees and radians, root orders and orientations, translation
  and length dofs, limits in their spellings, unattached bones, and motions found by their
  `#!OML:ASF` line, their name or their subject number, with `:SAMPLES-PER-SECOND`. Also bad
  and missing motion lines, a byte order mark, CRLF and comments. Errors: no skeleton, bad
  units, axes and directions.
- **C3D**: Intel, DEC and MIPS files with integer and float data, labels continued, missing
  markers, screen axes along each axis, units (and none: nothing is scaled), frame counts from the parameters, and analog
  data and events (counted). Errors: unknown processors and truncation.
- **TRC** and **HTR**: units (and a TRC with none: nothing is scaled), missing cells, dropped and cut-short rows, files without frame
  rates, times or frame lines; HTR's Euler orders, gravity and bone length axes, unit
  fractions, the global form and scale factors.

- **LAS and LAZ**: versions 1.0, 1.2, 1.3 and 1.4 and point formats 0, 1, 2, 3, 6, 7 and 8,
  with colors (16-bit, and 8-bit as some writers store them), near infrared, intensity,
  returns, classification, GPS time and point source as attributes. Also scale and offset per
  axis far from the origin, GeoKeyDirectory units (feet, and meters across with feet up),
  truncation, an unknown point format and files that aren't LAS; a real LAS file and its LAZ
  copy.
- **PCD**: `ascii`, `binary` and `binary_compressed` (LZF), positions only, packed `rgb` and
  `rgba`, normals and other fields, multi-count and padding fields, double and integer
  positions, organized clouds with NaN holes, version 0.5 without a viewpoint, a viewpoint
  kept in `extras`, and CRLF; every cloud turned from Z up as the text formats' are. Errors: truncation, no x y z, unknown data and files that
  aren't PCD; two real PCL clouds.
- **Text point clouds**: `.xyz` with spaces, commas, tabs or semicolons (with decimal commas),
  comments and blank lines, 8- and 16-bit colors, normals, intensity, named columns and
  coordinates far from the origin; Leica PTS (with several count blocks) and PTX (missing
  returns, several scans placed by their matrices, colors, a broken header); `.asc` point
  lists, and Esri ASCII grids from their corner or center with NODATA cells. Also files with
  no points.
- **E57**: two real files, a Cartesian scan of single-precision floats with 8-bit colors and a
  spherical scan of doubles with invalid points left out; a file that isn't E57.
- **AMF**: units (millimeters when unsaid, the standard's others and an unknown one), vertex
  and triangle colors, volume, object and material colors in order of precedence, composite
  materials, constellations (nested, turned about x, y and z, an instance of a missing
  object), zipped files, metadata, curved edges kept flat, textures, bad indices, truncation
  and files that aren't AMF or have no triangles; two real assimp models.
- **LightWave**: LWO2 and LWOB cubes (left-handed points and clockwise polygons turned to
  glTF's), z mirrored, surfaces (diffuse, luminosity, transparency, two sides), two surfaces,
  UV maps with an image map, vertex color maps, layers with pivots and parents, lines and
  points, concave polygons, morph maps and descriptions. Also an image sequence clip used
  through a reference clip, cylindrical and spherical image maps, an LWOB planar texture with a
  size and a center, and LWO3 surfaces
  made of nodes (a Principled BSDF with an image, a Standard material). Errors: other IFF files, no
  polygons, indices past the points; three real assimp models.
- **AC3D**: a cube of shaded quads, CRLF and a byte order mark, materials (color, emission,
  transparency), two-sided surfaces, `loc` and `rot` in nested objects, textures with `texrep`
  and `texoff`, closed and open lines, triangle strips and concave polygons. Errors: files
  that aren't AC3D or have no geometry; four real assimp models.
- **OGRE**: binary meshes of every serializer version (v1.10 to v1.100) in both byte orders,
  each version's own geometry, level of detail and edge list layouts, vertex elements of every
  type (half floats, doubles, packed colors, normalized and 10-10-10-2 integers), all six
  operations, poses and pose and morph animations, a skinned mesh with its binary skeleton, a
  skeleton alone, an XML mesh and skeleton linking to a binary one, a binary and XML twin with
  the same expectations, a morph animation in XML, and material scripts (imports, inheritance, blending, alpha
  rejection, normal maps, texture aliases, schemes). Poses are worked out in `ogre.py` with
  plain matrices. Errors: versions, the later v2 layout, unknown element and track types,
  buffers and offsets out of range, huge counts, broken XML and truncation.
- **OpenFlight**: a cube in feet, face colors (palette entries with intensities, packed,
  transparent, unlit, hidden), vertex colors and normals, material palette entries, SGI
  textures (stored and run-length encoded) and a PNG found by name, a missing texture,
  multitexture layers with a UV list (a bump map as the normal texture), levels of detail, a
  switch, matrices and a long ID, instances and replication, external references with their
  own units and either file's materials, wireframes and light points, a mesh's vertex pool and
  strips, fans, quadrilateral strips and polygons, light sources and an animated group. Also
  subfaces, light point and indexed light point records in a light point system, a binary
  separating plane, extension and attribute blocks skipped whole, and version 14.2's material
  table. Errors: files that aren't OpenFlight or have no geometry.
- **OpenGEX**: cubes Z up and Y up in other units, index arrays of each integer size with
  their own materials, triangle strips and clockwise fronts, lines and points, texture
  coordinates and colors, a textured material with a texture transform and a missing texture,
  node transforms of every kind, object-only transforms, shared geometry, levels of detail,
  morph targets, a skin with an animated bone, linear, constant, Bezier and rotation tracks,
  clips, lights and cameras, a file with objects but no nodes, one with nodes but no objects,
  and OpenDDL syntax (comments, string concatenation, hex and
  binary floats). Errors: an unclosed structure, no geometry; four real assimp models.
- **Irrlicht**: IRRMesh buffers of each vertex type (standard, two texture coordinates with
  the lightmap as the occlusion texture, tangents with the normal map), uniform and differing
  vertex colors and vertex alpha, UTF-8 and UTF-16 in both byte orders, and indices out of
  range. Scenes: node transforms nested and mirrored from Irrlicht's left-handed space, cube,
  sphere, sky box and billboard nodes, mesh nodes naming IRRMesh and OBJ files (one placed
  twice, one missing), a node's own materials over its file's, point and spot lights, a
  camera, rotation, fly-circle and fly-straight animators sampled into keys, hidden and
  unsupported nodes. Errors: files that aren't Irrlicht's, a scene with no nodes; one real
  assimp scene.
- **XGL**: meshes of referenced and inline vertices, lower-case tags, UTF-16, ZGL with and
  without the zlib header, a `.xml` told by its root, materials (diffuse, emissive,
  shininess, alpha) by reference, a face's own and a patch's, texture coordinates, lines and
  points, object transforms nested, a mesh placed by two references, a directional light,
  the ambient light and the background, unsupported tags, and broken references. Errors:
  files that aren't XGL, no geometry; three real assimp models.

- **VTK**: a block of hexahedra in every encoding (legacy ASCII and binary in the 4.2 and 5.1
  layouts, a 2.0 header, CRLF, doubles; XML as text, inline base64 and appended raw and base64
  data, with 32-bit and 64-bit headers, either byte order and zlib; meshio's own spellings of
  each), cells of every drawn kind sharing faces (tetrahedron, voxel, hexahedron, wedge,
  pyramid, prisms, their quadratic forms, surface, line and vertex cells), cell data splitting
  vertices, poly data with vertices, lines, polygons and strips, normals, texture coordinates
  and byte colors, `COLOR_SCALARS`, lookup tables, field arrays, a concave polygon, points
  alone, two pieces, and structured points, rectilinear and structured grids as blocks, sheets
  and rows. Errors: LZ4 and LZMA compression, files cut short, a count the file can't hold,
  image data, a parallel file, polyhedra alone, no points, and a file that isn't VTK.

- **3D Tiles**: batched models with a JSON and a binary batch table, without one, with no
  batch ids, with a small `RTC_CENTER` and none, under both legacy headers, with both centers
  at once (the glTF's own `CESIUM_RTC`, in glTF 2.0 and binary glTF 1.0), with globals in the
  binary body and with no tables at all; instanced models with each orientation (plain,
  oct-encoded, `EAST_NORTH_UP`, at the poles too, none), quantized positions, `RTC_CENTER`,
  both scales, batch ids of each width, a glTF by URI (escaped, in a folder), several and
  turned nodes, instances that shear, an animated glTF and no instances; point clouds with
  each color encoding, plain and oct-encoded normals, quantized positions, batches, per-point
  properties of every type, positions on the globe without a center, Draco data whole, in
  part and batched, and no points; composites of each tile, of composites and of nothing;
  tilesets of tilesets, with transforms, with `REPLACE` and `ADD` mixed and inherited, with
  `gltfUpAxis`, with several contents and glTF content, and with content missing. Errors:
  tiles cut short, lengths past the file, a version that isn't 1, tables that aren't JSON,
  counts and properties the body can't hold, quantized positions without their volume, no
  positions, a missing or absolute glTF URI, a bad `gltfFormat`, vector tiles, a file that
  isn't a tile, and tilesets with implicit tiling, no root, a bad transform or nothing there.

- **Gmsh**: a cube of tetrahedra with a physical volume and a physical surface in every version
  and encoding (4.1, 4.0, 2.2, 2.1, 2.0 and 1.0, ASCII and binary, either byte order, a 4-byte
  and an 8-byte `size_t`, meshio's spellings, CRLF, a `$Comments` section, partition tags,
  parametric nodes, sparse and gapped tags, every entity saved), node, element and element-node
  data as scalars, vectors and tensors, every shape (hexahedron, pyramid, prism, tetrahedron,
  triangle, quadrangle, line, point) at first, second (complete and incomplete) and third order,
  one of each volume element facing outward, two volumes sharing a face, no groups at all,
  partitioned meshes, and nodes alone. Errors: files cut short, counts the file can't hold,
  version 3, a bad data size or byte-order mark, a binary polyhedron, a post-processing file, no
  nodes, and a file that isn't Gmsh's.

- **Medit**: a cube of tetrahedra with its faces and edges by reference, as Gmsh and meshio
  write it and in binary versions 1 to 4 in either byte order, CRLF, comments, exponents
  (Fortran's too), no `End`, no keyword positions, and keywords that aren't read; every element
  keyword and the higher-order ones; one of each volume element facing outward; a
  two-dimensional mesh; normals for every vertex, assigned out of order, one each, and for some
  vertices only; tangents; and vertices alone. Errors: files cut short, counts the file can't
  hold, dimension 4, a binary version 5, vertices before the dimension, no vertices, and a file
  that isn't Medit's.

## `expect.json`

`spec` (what standard or variant the case exercises) and `source` (where the file came from)
are required. Every other key is optional and is checked only when present.

| Key | Meaning |
| --- | --- |
| `file` | The model file to load. Required when the folder holds more than one model file. |
| `needs` | A cargo feature the case needs beyond its format's own, such as `draco`, or `png` for the PNG encoder any image re-encoding feature brings. Builds without it skip the case. |
| `error` | Loading must fail with an error whose message (after the path it names) contains this text (case-insensitive). |
| `error_kind` | Required with `error`: the `ErrorKind` the error has, as Debug prints it (`Truncated`, `Corrupt`, `Empty`, ...). An error from loading must also name a file in the case's folder. |
| `no_mesh` | `true`: the file loads but has nothing to draw: no triangles, lines or points. |
| `facts` | An object; each entry must match the loader's `facts`. |
| `triangles`, `vertices`, `primitives`, `material_count`, `image_count` | Exact counts in the GLB. |
| `lines`, `points` | Exact counts of line segments and points in the GLB. A primitive's lines and points share its vertices, so they add no primitives or vertices of their own. |
| `bounds` | `{"min": [x,y,z], "max": [x,y,z]}`, glTF space (Y up, meters). |
| `area` | Total surface area. Catches polygons triangulated outside their outline. |
| `position_set` | A file beside the model of little-endian `f32` triples, glTF space: the distinct positions another tool read. Every position drawn must be one of them and each of them must be drawn, in any order and however either side welds. |
| `animation_spans` | A list of `[first, last]` key times in seconds, one for each animation. |
| `uvs`, `colors` | Whether any primitive has `TEXCOORD_0` / `COLOR_0`. |
| `positions_include`, `uvs_include`, `colors_include` | Lists of values that must each appear somewhere. Colors are linear RGBA. |
| `normals_in` | Every normal must be one of these directions. |
| `default_normals` | `true`: the file has no normals and nothing to say how it is shaded, so its normals are the default ones, made with a crease angle of 30 degrees. No vertex's normal may be further than that from the normal of a triangle that uses it. |
| `winding_matches_normals` | Every triangle winds counter-clockwise when seen from the side its normals point to. |
| `faces_outward` | For convex models: every triangle faces away from the center of the bounds. Catches mirrors that turn faces inside out. |
| `tolerance` | Relative tolerance for the numeric checks (default `1e-4`). |
| `same_as` | `{"case": "<another case of the format>"}`: the flat mesh has the same materials and images as that case's, and the same triangles in each material, in any order, whatever vertices they share. `positions`, `normals`, `uvs` and `colors` beside `case` give the largest difference allowed in any component (default `1e-5`). |
| `poses` | A list of `{"animation", "time", "bounds"}`: the bounds of what's drawn in that animation at that time (seconds). |
| `materials` | A list of material checks (below). |
| `document` | Counts in the document's glTF JSON (`nodes`, `skins`, `animations`, `cameras`, `lights`, `joints`, `morph_targets`, `channels`), plus `node_names` and primitive `attributes` that must be there. |
| `json` | Values in the document's glTF JSON, keyed by JSON pointer (`/materials/0/extensions/KHR_materials_ior/ior`). Numbers match to within `1e-4`, and objects match on the members given. `null` means the pointer must lead nowhere. |
| `flat_json` | The same, in the glTF JSON of the flat mesh written as GLB, so what reaches the mesh (its materials, textures and samplers) can be checked. |
| `rig` | A list of checks on what `Document::rig` makes of a node, found by `node` name. Each may give its `parent` (a name, or `null` for a root) and its `translation`, `rotation` (either sign) and `scale`. These are in the rest pose, or with `time` (seconds), posed by animation `animation` (default 0) at that time. `world` is where the node's origin is in the scene, its parents' transforms applied. `keys` maps a path (`translation`, `rotation`, `scale`) to the channel's key count, its key times, or `null` for no channel. |

Each `materials` entry is matched by `name`, or by position when it has no name:

| Key | Meaning |
| --- | --- |
| `color` | Base color factor, linear (3 or 4 components). `tolerance` overrides the default `2e-3`. |
| `emissive`, `metallic`, `roughness` | glTF factors. |
| `textured`, `normal_map` | Whether a base color texture or normal texture is present. |
| `double_sided` | The glTF `doubleSided` flag. |
| `alpha_mode` | `OPAQUE`, `MASK` or `BLEND`. |
| `pixel` | The base color texture's top-left pixel, as 8-bit RGB. |
| `metal_rough_pixel` | The same, of the metallic-roughness texture. |
| `occlusion_pixel` | The same, of the occlusion texture. |

## What every format's cases expect alike

Whatever the format, the same thing is expected the same way (`tests/kinds.rs` and
`tests/uniform.rs` hold every reader to it, extension by extension):

- **Errors.** A file with no bytes is `Empty`. A file that isn't the format it's read as (no
  signature, no first words of its text) is `UnknownFormat`. A file cut short is `Truncated`, or,
  where the reader keeps what came before the cut, loads with `document.source.truncated: true`.
  A well-formed file with nothing to convert (nothing drawn, and no named node, hierarchy,
  camera, light or skin) is `Empty`.
- **Facts.** `facts.triangles` is always there, and `facts.points` and `facts.lines` when there
  are any; they are what the document draws, counted one way for every reader.
- **The source** (`document.source`, or `json` pointers under `/asset/extras/source`). The keys
  are listed in `src/source.rs`. `unit` is the unit's usual name by its length (`millimeter`,
  never `mm`), with the file's own word as `unitAsWritten` when it differs
  (`common.unit_source` writes both). `assumed` lists only what the file could have said and
  didn't, or what is guessed; what a format fixes for every file is declared. An assumed
  `frameRate` is there only when the model is animated. `encoding` is `ascii` or `binary` (or a
  format's own other way, in lower case), `byteOrder` is `little` or `big`, a text encoding is
  `textEncoding`. Textures named and not in the document are counted as `droppedTextures`,
  triangles, segments and points left out for a vertex that isn't there or isn't a finite point as
  `droppedGeometry`, and files named and not found are listed as `missingFiles`.
- **Names.** Text that isn't UTF-8 is Windows-1252. A thing with no name has no `name`, and a
  name the reader makes up for something the format numbers is the kind and the number
  together: `material3`, `object1`, `frame2`.

## Adding a case

Add the case to the format's generator rather than hand-editing a folder, so regenerating
keeps it. Name the case after what it tests (`texture_subfolder`, `rotation_order_zyx`).
Expect what the standard says, not what Polysmelt currently does. If the reader is wrong, fix
the reader.
