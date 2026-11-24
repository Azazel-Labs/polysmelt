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
| PLY | 37 | 33 | 4 |
| VOX | 27 | 27 | none |
| OBJ | 58 | 49 | 9 |
| 3DS | 48 | 48 | none |
| 3MF | 43 | 43 | none |
| FBX | 130 | 112 | 18 |
| USD | 141 | 121 | 20 |
| glTF | 75 | 60 | 15 |
| SketchUp | 31 | 7 | none |
| BVH | 41 | 41 | none |
| COLLADA | 76 | 76 | none |
| OFF | 23 | 23 | none |
| VRM | 7 | 7 | none |
| X3D | 53 | 53 | none |
| DXF | 34 | 34 | none |
| Gaussian splats (`.splat`) | 7 | 7 | none |
| SPZ | 19 | 19 | none |

- **Generated** cases are written byte by byte by `tools/corpus/<format>.py` (3DS is
  `tds.py`, 3MF is `threemf.py`, COLLADA is `dae.py`). Each one targets one clause of the format's specification,
  or one variant that real exporters write. The expectations come from what the generator
  wrote, never from reading the file back with Polysmelt, so a case tests the reader against
  the standard rather than against itself.
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
- **Blender** cases (`blender_*` folders) are real exporter output, from
  `tools/corpus/blender_export.py`.

Regenerate a format from the repository root:

```
python tools/corpus/obj.py
"<blender.exe>" -b --factory-startup --python-exit-code 1 --python tools/corpus/blender_export.py -- obj
```

`usd.py` needs `pip install usd-core`. A generator replaces its own cases and leaves the
`blender_*` folders alone, and the reverse is also true.

## What the cases cover

- **STL**: binary and ASCII. Covers headers that start with `solid`, truncation, trailing
  bytes, VisCAM and Magics colors, NaN and degenerate facets, and normals that are wrong or
  zero. Also several solids in one file, CRLF line endings and non-UTF-8 names, all turned
  from Z up to Y up.
- **PLY**: ASCII and binary in both byte orders, and every property type. Covers vertex
  colors in each encoding, the various UV property names, n-gons and triangle strips, point
  clouds and gaussian splats (read as `KHR_gaussian_splatting`, with spherical harmonics of
  degree 1). Also unknown elements and broken headers.
- **VOX**: palettes (default and off-by-one), materials, and the scene graph: transforms,
  rotations, mirrors, groups, instances and hidden layers or nodes. Also version 150 packs
  and unknown chunks.
- **OBJ**: every face-corner form, negative indices and n-gons. Covers MTL colors, PBR
  extensions, texture maps and their options, Windows paths and missing libraries, and TGA,
  BMP, GIF, TIFF, PSD and DDS textures re-encoded as PNG. Also
  vertex colors and syntax variants: line continuation, `w` components, number forms and
  non-UTF-8 text.
- **3DS**: smoothing groups, materials (24-bit and float colors, transparency, shininess,
  self-illumination), textures and two-sided faces. The bump map is the normal map when its
  image is one (by its pixels, a gray image's header, or its name), scaled by its amount; a
  height map, the shininess map and the reflection map stay in `extras`. Also cameras, point and spot lights, and
  the keyframer: its hierarchy, pivots, dummies, instances and tracks. Also truncated and
  malformed chunks.
- **3MF**: units, build items and components with transforms (including mirrored ones), the
  production extension, and colors from base materials, color groups and gradients.
  Textures, plus the files Bambu and Prusa write. Also packaging variants and broken
  packages.
- **FBX**: ASCII 6100 and 7300–7500, binary 7400–7700 (zlib-compressed arrays too). Covers
  axis systems and units, every rotation order, pivots, offsets and geometric transforms, and
  animation, including Euler keys baked past ±180° or flipped past straight up.
  Also every normal, UV and color mapping, materials, textures (embedded, external and
  missing), and instancing. Line geometry is drawn as lines (polylines through `PointsIndex`,
  broken where an index isn't a point, in the model's material or the line's color, its own
  or its template's), and NURBS curves and surfaces are counted as skipped.
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
  set, `KHR_texture_transform`). glTF 1.0, upgraded as it's read: objects keyed by ID,
  binary glTF, `KHR_materials_common` and technique materials, accessor strides, a node's
  several meshes, skins found by joint name with a bind shape matrix, animation parameters,
  cameras and lights. Errors: Draco, truncation, buffers outside the model's folder, and
  accessors that overrun their views.
- **SketchUp**: every layout from SketchUp 3 to 2026: the legacy object stream (3 to 6,
  2013 to 2020) and the ZIP container of 2021 on. Covers faces bounded by arcs beside a loose
  arc with no face, textured materials stored inline, groups and components nested inside
  each other, an instance scaled through its transform's last value, tags carrying line
  styles (2019 and 2020 layouts), and files with nothing to draw (empty, guides only, materials only). Also annotations, scenes
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
  RELATIVE morphs and animated morph weights, animations of whole matrices, components,
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
  and its base color and opacity drawn as point colors; trailing bytes, an empty file and a
  splat at an infinite position. Also glTF files that already use `KHR_gaussian_splatting`:
  drawn in their base spherical harmonics' color in either color space, or in a COLOR_0
  they give.
- **SPZ**: versions 1 to 3 (half-float and fixed-point positions, three-byte and
  smallest-three rotations), spherical harmonics of degrees 1 to 3 and degree 4 dropped, the
  antialiased flag, and the coordinate system record (axes flipped, or turned on the node)
  among other records or without the flag that says records follow. Errors: truncation, a
  count past `Keep::max_points`, version 4, an unknown version, an unzipped stream.

## `expect.json`

`spec` (what standard or variant the case exercises) and `source` (where the file came from)
are required. Every other key is optional and is checked only when present.

| Key | Meaning |
| --- | --- |
| `file` | The model file to load. Required when the folder holds more than one model file. |
| `needs` | A cargo feature the case needs beyond its format's own, such as `draco`. Builds without it skip the case. |
| `error` | Loading must fail with an error containing this text (case-insensitive). |
| `no_mesh` | `true`: the file loads but has nothing to draw: no triangles, lines or points. |
| `facts` | An object; each entry must match the loader's `facts`. |
| `triangles`, `vertices`, `primitives`, `material_count`, `image_count` | Exact counts in the GLB. |
| `lines`, `points` | Exact counts of line segments and points in the GLB. A primitive's lines and points share its vertices, so they add no primitives or vertices of their own. |
| `bounds` | `{"min": [x,y,z], "max": [x,y,z]}`, glTF space (Y up, meters). |
| `area` | Total surface area. Catches polygons triangulated outside their outline. |
| `uvs`, `colors` | Whether any primitive has `TEXCOORD_0` / `COLOR_0`. |
| `positions_include`, `uvs_include`, `colors_include` | Lists of values that must each appear somewhere. Colors are linear RGBA. |
| `normals_in` | Every normal must be one of these directions. |
| `winding_matches_normals` | Every triangle winds counter-clockwise when seen from the side its normals point to. |
| `faces_outward` | For convex models: every triangle faces away from the center of the bounds. Catches mirrors that turn faces inside out. |
| `tolerance` | Relative tolerance for the numeric checks (default `1e-4`). |
| `materials` | A list of material checks (below). |
| `document` | Counts in the document's glTF JSON (`nodes`, `skins`, `animations`, `cameras`, `lights`, `joints`, `morph_targets`, `channels`), plus `node_names` and primitive `attributes` that must be there. |
| `json` | Values in the document's glTF JSON, keyed by JSON pointer (`/materials/0/extensions/KHR_materials_ior/ior`). Numbers match to within `1e-4`, and objects match on the members given. `null` means the pointer must lead nowhere. |
| `rig` | A list of checks on what `Document::rig` makes of a node, found by `node` name. Each may give its `parent` (a name, or `null` for a root) and its `translation`, `rotation` (either sign) and `scale`. These are in the rest pose, or with `time` (seconds), posed by animation `animation` (default 0) at that time. `keys` maps a path (`translation`, `rotation`, `scale`) to the channel's key count, its key times, or `null` for no channel. |

Each `materials` entry is matched by `name`, or by position when it has no name:

| Key | Meaning |
| --- | --- |
| `color` | Base color factor, linear (3 or 4 components). `tolerance` overrides the default `2e-3`. |
| `emissive`, `metallic`, `roughness` | glTF factors. |
| `textured`, `normal_map` | Whether a base color texture or normal texture is present. |
| `double_sided` | The glTF `doubleSided` flag. |
| `alpha_mode` | `OPAQUE`, `MASK` or `BLEND`. |
| `pixel` | The base color texture's top-left pixel, as 8-bit RGB. |

## Adding a case

Add the case to the format's generator rather than hand-editing a folder, so regenerating
keeps it. Name the case after what it tests (`texture_subfolder`, `rotation_order_zyx`).
Expect what the standard says, not what Polysmelt currently does. If the reader is wrong, fix
the reader.
