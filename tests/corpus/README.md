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
| PLY | 36 | 32 | 4 |
| VOX | 27 | 27 | none |
| OBJ | 52 | 43 | 9 |
| 3DS | 43 | 43 | none |
| 3MF | 43 | 43 | none |
| FBX | 124 | 106 | 18 |
| USD | 134 | 114 | 20 |
| glTF | 70 | 57 | 13 |
| SketchUp | 28 | 7 | none |

- **Generated** cases are written byte by byte by `tools/corpus/<format>.py` (3DS is
  `tds.py`, 3MF is `threemf.py`). Each one targets one clause of the format's specification,
  or one variant that real exporters write. The expectations come from what the generator
  wrote, never from reading the file back with Polysmelt, so a case tests the reader against
  the standard rather than against itself.
- **USD** expectations are computed with Pixar's `usd-core` (the `pxr` module). It composes
  the stage and measures it: triangles, bounds, visibility, purpose and transforms.
- **SketchUp**'s format is closed, so its other 21 cases are files SketchUp saved, taken from
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
  clouds and Gaussian splats. Also unknown elements and broken headers.
- **VOX**: palettes (default and off-by-one), materials, and the scene graph: transforms,
  rotations, mirrors, groups, instances and hidden layers or nodes. Also version 150 packs
  and unknown chunks.
- **OBJ**: every face-corner form, negative indices and n-gons. Covers MTL colors, PBR
  extensions, texture maps and their options, Windows paths and missing libraries. Also
  vertex colors and syntax variants: line continuation, `w` components, number forms and
  non-UTF-8 text.
- **3DS**: smoothing groups, materials (24-bit and float colors, transparency, shininess,
  self-illumination), textures and two-sided faces. Also cameras, point and spot lights, and
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
  missing), and instancing.
- **USD**: `.usda`, `.usdc`, `.usd` in either encoding, and `.usdz`. Covers up axis and
  `metersPerUnit`, every xformOp and rotation order, holes and invalid faces,
  `leftHanded` orientation with authored and computed normals (mirrored too), and
  `primvars:normals` versus `normals`. Also UV primvar names, display color and opacity, and
  composition: references, payloads, sublayers, variants, instancing, classes and overs.
  Also visibility and purpose, UsdPreviewSurface, textures (beside the file, in a subfolder,
  inside a `.usdz`, missing), GeomSubsets, gprims, point instancers and time samples.
- **glTF**: GLB, `.gltf` with an external buffer (in a subfolder, percent-encoded) or a data
  URI, and a byte order mark. Covers u8, u16 and u32 indices, missing indices and normals,
  interleaved views, sparse accessors (over a view and over zeros), `KHR_mesh_quantization`
  (integer and normalized types), strips and fans, points and lines. Also the node hierarchy
  (TRS and matrices, mirrors, the default scene, no scenes, no nodes), skins and morph
  targets at rest, `EXT_mesh_gpu_instancing`, metallic-roughness and specular-glossiness
  materials, and textures (embedded, external, data URI, missing, WebP fallback, a second UV
  set, `KHR_texture_transform`). glTF 1.0, upgraded as it's read: objects keyed by ID,
  binary glTF, `KHR_materials_common` and technique materials, accessor strides, a node's
  several meshes, skins found by joint name with a bind shape matrix, animation parameters,
  cameras and lights. Errors: Draco, truncation, buffers outside the model's folder, and
  accessors that overrun their views.
- **SketchUp**: every layout from SketchUp 3 to 2026: the legacy object stream (3 to 6,
  2013 to 2020) and the ZIP container of 2021 on. Covers faces bounded by arcs beside a loose
  arc with no face, textured materials stored inline, groups and components nested inside
  each other, an instance scaled through its transform's last value, and
  files with nothing to draw (empty, guides only, materials only). Also annotations, scenes
  and an SDK-saved file beside what is drawn, and truncated, foreign and too-old files.

## `expect.json`

`spec` (what standard or variant the case exercises) and `source` (where the file came from)
are required. Every other key is optional and is checked only when present.

| Key | Meaning |
| --- | --- |
| `file` | The model file to load. Required when the folder holds more than one model file. |
| `needs` | A cargo feature the case needs beyond its format's own, such as `draco`. Builds without it skip the case. |
| `error` | Loading must fail with an error containing this text (case-insensitive). |
| `no_mesh` | `true`: the file loads but has nothing to draw. |
| `facts` | An object; each entry must match the loader's `facts`. |
| `triangles`, `vertices`, `primitives`, `material_count`, `image_count` | Exact counts in the GLB. |
| `bounds` | `{"min": [x,y,z], "max": [x,y,z]}`, glTF space (Y up, metres). |
| `area` | Total surface area. Catches polygons triangulated outside their outline. |
| `uvs`, `colors` | Whether any primitive has `TEXCOORD_0` / `COLOR_0`. |
| `positions_include`, `uvs_include`, `colors_include` | Lists of values that must each appear somewhere. Colors are linear RGBA. |
| `normals_in` | Every normal must be one of these directions. |
| `winding_matches_normals` | Every triangle winds counter-clockwise when seen from the side its normals point to. |
| `faces_outward` | For convex models: every triangle faces away from the centre of the bounds. Catches mirrors that turn faces inside out. |
| `tolerance` | Relative tolerance for the numeric checks (default `1e-4`). |
| `materials` | A list of material checks (below). |
| `document` | Counts in the document's glTF JSON (`nodes`, `skins`, `animations`, `cameras`, `lights`, `joints`, `morph_targets`, `channels`), plus `node_names` and primitive `attributes` that must be there. |
| `json` | Values in the document's glTF JSON, keyed by JSON pointer (`/materials/0/extensions/KHR_materials_ior/ior`). Numbers match to within `1e-4`, and objects match on the members given. `null` means the pointer must lead nowhere. |

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
