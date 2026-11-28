# Changelog

Breaking changes are allowed in minor releases until 1.0.

## 0.23.0

- OpenFlight animated groups play as the group record says. The last frame duration was taken as
  the last frame's whole length; it is how much longer than the others the last frame of the last
  loop is shown, and only when the loop count is finite. The loop count was ignored: a finite one
  is now played out, every loop in the animation, and a loop count of 0 (forever) is played once.
  A swing loop's frames now share its loop duration (2N - 1 frames in the first loop, 2N - 2 after,
  not N), a swing repeating forever is one pass there and back, and a backward group shows its
  last child at rest, where it starts. A group's node says in `extras.openflight.loopsPlayed` how
  many loops its animation holds.
- IQM: an animation whose `framerate` is 0, in a file whose animations' rates differ, says its
  `extras.frameRate` of 30 is assumed (`extras.assumed: ["frameRate"]`), as the source info does.
  This is the convention for any animation's own assumed rate (see docs/consistency.md).
- USD conformance cases for the frame-rate rule (`timeCodesPerSecond`, then `framesPerSecond`,
  then 24): each alone, both, a `timeCodesPerSecond` of 0, and neither.
- SketchUp decodes GIF and TIFF textures through the shared image path (with the `gif` and `tiff`
  features, on by default), as every other reader does, instead of dropping them silently. A
  texture it still can't make into an image is counted in `source.droppedTextures`, and one a
  2021+ file names but doesn't hold is counted and listed in `missingFiles`, by the rule in
  docs/consistency.md.
- 3MF counts a texture whose part isn't in the package in `droppedTextures` and lists the path, as
  the model writes it, in `missingFiles`; a part that isn't an image is counted too. Two textures
  of one part (tiled two ways) are one texture. An image reference's name (with
  `Keep::image_bytes` off) is the path as written rather than lowercased.
- FBX, COLLADA and USD count an embedded or packaged texture that isn't an image once, after
  looking for the file it names in its place, where they counted it only when that file was
  missing, or not at all (COLLADA's `<hex>` and `data:` images, a `.usdz`'s entries). Only names
  looked up and not found are listed in `missingFiles`. A UDIM texture none of whose tiles is
  found is counted once and listed by its pattern. With `Keep::image_bytes` off, FBX and USD try
  embedded bytes that don't start like a PNG or JPEG, so the file they fall back to is the one a
  full read draws.
- Irrlicht scenes and 3D Tiles tilesets report the `droppedTextures` and `missingFiles` of the
  files they place as their own, a name counted and listed once across all of them; before, a
  scene kept only its own and a tileset only its missing tiles.

## 0.22.0

- A flattened `mesh::Prim` keeps up to 16 joints and weights a vertex, as documents do, instead
  of only the 4 strongest: so `Pose::Bind` meshes skinned on the GPU, and meshes written back out
  with `mesh::to_glb`, keep every influence a file gives. `skin_joints` and `skin_weights` are
  still the strongest four (`JOINTS_0`/`WEIGHTS_0`); the new `skin_joint_sets` and
  `skin_weight_sets` hold influences 5 to 16 four at a time (`JOINTS_n`/`WEIGHTS_n`, slot 0 left
  empty, as `uv_sets` and `color_sets` do). Weights now sum to 1 over every set, as glTF's do, so
  `skin_weights` alone sums to less where a vertex has more than four; the new
  `Prim::renormalized_skin_weights()` gives the four scaled to sum to 1, for a renderer that
  blends four, and `Prim::skin_sets()` walks every set. `mesh::to_glb` writes each set, and
  `mesh::MAX_INFLUENCES` is public. Breaking: a `Prim` built with a struct literal needs the two
  new fields (or `..Prim::default()`).
- Every reader counts `source.droppedTextures` the same way: each distinct texture once, however
  many materials use it, with the names not found in `missingFiles`, each once (see
  docs/consistency.md). AC3D, OpenFlight, LightWave, OpenGEX, AMF and glTF counted their own way:
  - OpenFlight counted a texture once per palette entry and file naming it, and LightWave once
    per surface; LightWave now also counts a missing normal map. Names that find the same file
    that isn't an image are one texture in AC3D and OpenFlight too, and OpenGEX counts such a file.
  - AC3D, OpenFlight, LightWave and OpenGEX now list missing textures in `missingFiles`. They
    counted missing textures even in a read that keeps no textures or materials, which no other
    reader does; such a read now reports none.
  - AMF counted each set of channels that made no image, and each unreadable texture whether or not
    a triangle uses it; now it counts each texture triangles name that no image has.
  - glTF counted only textures in KTX2 or AVIF, once per texture object. Now an image file that
    isn't found counts too, and is listed in `missingFiles`, and each image is counted once however
    many textures use it. Only that report changes: the document's images and textures are as before.
  - An Irrlicht scene that refers to a mesh file that isn't there kept the textures its materials
    didn't find out of `missingFiles`; they're listed, and each missing file once.
- USD: a `timeCodesPerSecond` that isn't above 0 is passed over for the stage's
  `framesPerSecond`, as USD does when it isn't authored, then for USD's 24, which is now
  reported as assumed. It was reported as a declared 24, and a `framesPerSecond` beside it was
  ignored for timing keys.
- glTF 1.0, VRM and 3D Tiles, which convert their glTF rather than pass it through, follow
  the animation, skin and name rules every other reader does. glTF 2.0 read as itself is
  unchanged, as before. What already follows the rules is left as the file has it:
  - An animation whose keys don't start at 0 starts at 0, what was taken off kept as its
    `extras.timeOffset`, and its keys are cleaned (increasing times, unit rotations that turn
    the short way).
  - A skinned primitive with more than 16 influences on a vertex, a joint named twice, weights
    that don't sum to 1 or a vertex no joint weights has its joints and weights rewritten: the
    16 strongest kept, duplicates merged, weights normalized, and an unweighted vertex bound to
    the node that draws the mesh (added to the skin as a joint, its inverse bind matrix the
    identity) rather than to the skin's first joint. Every skin's `skeleton` is its joints'
    common root.
  - A `name` of `""` is taken away.
- A glTF 1.0 file loads under a `RootNode` with no transform, as every format with a scene
  does. The node is the last in the list, so the file's nodes keep their indices. (A glTF 1.0
  inside a 3D Tiles tile has none, as 3D Tiles has no `RootNode`.)
- **Breaking:** a tileset lists the tiles it couldn't find in
  `asset.extras.source.missingFiles`, the key every reader uses, rather than `missingTiles`.
  `Document::facts` still counts them as `missing_tiles`.

## 0.21.0

- Every reader brings a file's axes and units to glTF's (+Y up, right-handed, meters) the
  same way, and gives the same shape of document, so nothing depends on the format, or on
  which axes and unit a file happens to use. Before there were four ways: a `RootNode` that
  was always there (STL, 3DS, 3MF, AMF, DXF, OpenGEX), one only when the file needed
  converting (FBX, COLLADA, USD, X3D, VRM), a rotation on each of the file's roots (MD2, MD3,
  MD5, IQM, SMD, HTR, C3D) and conversion written into the data. Now:
  - A format with a scene has exactly one top-level node, always there and always named
    `RootNode`, with the turn to +Y up and the scale to meters (no transform when the file
    needs none). Every root of the file is its child: node paths gain that parent in every
    format but glTF, and `scenes[0].nodes` is one node. This is every format except the
    sampled ones below and glTF, so OBJ, PLY and OFF meshes, VTK, Gmsh, Medit, LightWave,
    AC3D, XGL, Irrlicht, DirectX, the game and motion capture formats and SketchUp have the
    node too. OpenFlight's header node and DXF's model space are children of it, not it.
  - Skinned meshes are under `RootNode` in FBX, COLLADA and USD (they were beside it), and
    the game formats no longer add a `Skeleton` node over a skin's several roots.
  - Units are the root's scale everywhere: MikuMikuDance (0.08), ASF, C3D, TRC, HTR and
    SketchUp keep their positions, translations, keys and bind matrices in the file's unit
    and axes, as FBX and COLLADA always did. What is drawn is where it was. `Pose::Bind`
    gives a skinned mesh as the file stores it, so in the file's units for these formats too.
  - A left-handed FBX file is mirrored as it is read (vertices, normals, tangents, morph
    targets, node transforms, bind matrices and keys, triangles wound the other way) as
    every other left-handed format is; it was left as a negative scale on the root.
  - Point clouds, gaussian splats and voxel grids are converted in their data, as before, and
    have no `RootNode`. A PCD cloud is kept as stored, with no up axis assumed: PCD has no
    convention (PCL's own samples are Y up, robotics clouds Z up, camera clouds Y down), and
    point cloud tools show it as stored.
  - 3DS: the master scale is applied. It is the file's "master scale factor for all
    geometry coordinates", so the root scales by it, and camera clip planes and light
    ranges with it. A file whose master scale is 1, as nearly all are, is the size it was.
    3DS names no unit, so the unit stays unknown and `converted` doesn't list `units`.
  - A file that doesn't say its unit, in a format that defines none, has an unknown unit and
    is not scaled: C3D without `POINT:UNITS` and TRC without `Units` were taken to be
    millimeters, and now are as HTR and BVH are.
  - `converted` in the source info says what the root does: `upAxis` when it turns, `units`
    when it scales, neither for a file that needed nothing. C3D claimed a turn for Y-up
    files, and VRM 0.x claimed none for its half turn.
  - glTF 2.0 and 1.0 and 3D Tiles are kept as they are. A `.vrm` or `.vrma` has the
    `RootNode` whatever its version.
  - The root's turn is written as the same exact numbers by every reader (a quarter turn
    about X is `[-0.7071067811865476, 0, 0, 0.7071067811865476]`), where some wrote it in
    single precision and some from a matrix.
  - The FBX writer doesn't write a `RootNode` that converts nothing as a model, so a file
    read and written again keeps the nodes it had. 3DS keeps the node of a camera or light
    that isn't kept (`Keep` without cameras or lights), as pruning one does.

- Every reader now treats geometry that can't be drawn the same way. A triangle, line segment
  or point that names a vertex that isn't there, or one whose position isn't finite, is
  dropped: never moved to the origin, and never an error for the whole file. Each is counted
  under one key, `asset.extras.source.droppedGeometry`, which replaces `droppedFaces` (OFF),
  `droppedCells` (VTK, Gmsh, Medit), `droppedSurfaces` (AC3D), `badPrimitives` (OpenGEX), `droppedCorners`
  (OpenFlight), the bad polygons in LightWave's `droppedPolygons` (which keeps the polygon
  kinds that aren't drawn), and the facts `dropped_triangles` (MilkShape 3D, Blitz3D,
  MikuMikuDance, SMD, Irrlicht) and `dropped_primitives` (XGL). OBJ, PLY, 3MF, DXF, 3DS,
  DirectX, OGRE, USD, FBX, COLLADA, X3D and STL, which dropped such faces without a word, count
  them there too. What changes in the documents:
  - A position that isn't finite no longer stays in `POSITION` (OBJ, PLY, 3MF, 3DS, LightWave,
    OpenGEX, USD) and is no longer drawn at the origin (COLLADA, AMF, XGL, OpenFlight, and the
    game and character formats): what was drawn on it is dropped. No document read from these
    formats holds a position that is NaN or infinite, or an index past its vertices.
  - MD2, MD3, MD5 and IQM: a triangle naming a vertex past its list (MD2: or a texture
    coordinate) is dropped and counted, where the file was refused as corrupt.
  - OpenFlight: a face with a corner that names no vertex is dropped whole, where it was drawn
    with the corners left; a mesh strip's or fan's triangle on such a corner is dropped alone.
  - DXF: a polyface mesh's face naming a vertex past the list is dropped, where it was drawn
    without that corner.
  - ASCII FBX: `1.#INF`, `1.#QNAN` and `-1.#IND` are the infinity or NaN they stand for, as a
    binary file holds them, so a polygon on such a point is dropped in both; it read as 0.
  - `Document::to_glb`, `Document::repair` and `Document::flatten` follow the same rule for a
    glTF file, which is read as it is: what is drawn on a position that isn't finite is
    dropped (also from a primitive without indices), where `repair` moved the vertex to the
    origin and `flatten` kept the NaN. The three now agree.
- Triangles with no area, or with a corner twice, are kept as the file has them, in every
  format: they hold a mesh's topology together, and other tools count them. COLLADA no longer
  drops a triangle with two corners at one position, and a polygon keeps a corner it repeats.
  DXF keeps a face with a repeated corner (four corners whose last repeats the third are still
  the triangle DXF means by that), and one whose corners are all in a line. The one exception
  is a triangle a reader makes itself by unrolling a strip or a fan: one that names a vertex
  twice restarts the strip and draws nothing. That was so in AC3D, OpenGEX and OGRE strips, and
  is now so in OGRE fans and in PLY, VTK, X3D, COLLADA and OpenFlight strips and fans.
- Every triangle primitive read from a file has unit normals (`NORMAL`); points and lines get
  none, and a glTF file without normals keeps none, as glTF defines flat shading for it. A
  file's own normals are used, made unit length (AMF's were passed on as written). One that
  is zero or not finite takes the area-weighted normal of the triangles around its vertex, and
  `[0, 1, 0]` when those have no area: one rule, where readers put `[0, 1, 0]`, `[0, 0, 1]` or
  a face's normal. Where a file has no normals, what its format says about smoothing is
  followed: smoothing groups (OBJ's `s`, 3DS, FBX), X3D's `creaseAngle` (0 by default, so a
  face set without normals is flat, as the standard has it, where it had none), LightWave's
  smoothing angle (none is flat), MD5's rule, SketchUp's soft edges.
- A file with neither normals nor smoothing data gets the same normals whatever its format:
  made with a crease angle of 30 degrees. Each corner takes the normals of the triangles
  around its point that turn less than that from its own, weighted by their areas, and a
  vertex is split where its corners' normals differ. So a cube comes out hard-edged, with 24
  vertices, and a finely cut curved surface smooth, with its vertices shared, from STL, OBJ
  (with no `vn` and no `s`), PLY, OFF, 3MF, AMF, VTK, COLLADA, DXF, DirectX, Blitz3D, IQM,
  OpenGEX, XGL, Gmsh, Medit and FBX alike. The triangles around a point are those of the primitive with a
  corner at that position, so a vertex split for its texture coordinates shows no seam. Before,
  STL, COLLADA, AMF, 3MF, DXF, VTK, DirectX, Blitz3D, IQM, OpenGEX and XGL had no normals, and
  OBJ, PLY, OFF and FBX smoothed across every shared vertex, hard edges too. The vertices added
  come after the file's own, each with everything its original holds (skin weights, morph
  targets, other attributes).
- `Keep::with_crease_angle` sets that angle, in degrees, and `Keep::crease_angle` reads it:
  30 by default, 0 for every face flat with normals of its own, 180 for smooth across every
  point the faces share. Values past either end are that end, and one that isn't a number
  leaves the default. A file's own normals and smoothing data always win, and glTF input is
  untouched.
- `mesh::smooth_normals` weights each triangle's normal by its area; it averaged the faces
  unweighted.
- X3D's crease angle is now worked out by the code LightWave and AC3D use, and a point with
  very many faces around it (the tip of a cone, the middle of a fan) is smoothed like any
  other: past 256 faces X3D averaged them all, whatever the angle, and past 512 LightWave and
  AC3D left the point flat.
- DirectX and OpenGEX: a face of four or more corners (OpenGEX: a quad) that isn't convex is
  cut into triangles inside its outline, as every other reader cuts one; it was fanned from
  its first corner.
- Classic materials (Phong, Blinn, Lambert) are read one way by every reader, so the same
  material is the same glTF material whatever format held it. Before, ten formulas turned a
  highlight's exponent into roughness: an exponent of 20 was 0.30 from COLLADA and OGRE,
  0.55 from FBX and AC3D, 0.86 from OBJ and DirectX. Now, for OBJ, COLLADA, 3DS, FBX's
  classic materials, DirectX, X3D and VRML97, glTF 1.0, LightWave, AC3D, OpenFlight,
  OpenGEX, Irrlicht, XGL, OGRE, MikuMikuDance, MilkShape 3D and Blitz3D:
  - Roughness is `(2 / (n + 2))^(1/4)` for a Blinn-Phong exponent `n` (a lobe of exponent
    `n` is nearest a GGX lobe of alpha squared `2 / (n + 2)`, and roughness is the root of
    alpha): 0.549 at 20. Each reader brings its own scale to the exponent: `Ns` as it is in
    MTL, `2^(10 g)` for a 3DS shininess percentage, `2^(10 g + 2)` for a LightWave
    glossiness, shininess times 128 in X3D and Blitz3D, the value itself elsewhere.
    Metalness is 0.
  - The specular color, scaled by its strength and made linear, is written as
    `KHR_materials_specular`: its largest channel is `specularFactor` and the color over
    it `specularColorFactor`. A full white specular is glTF's own highlight and writes no
    extension; 0.5 gray leaves 0.214 of it. FBX's classic materials, DirectX, LightWave,
    OpenFlight, Irrlicht, AC3D, XGL, X3D, MikuMikuDance, MilkShape 3D, Blitz3D, OGRE and
    glTF 1.0 dropped the color before. OBJ's `Ks` is now made linear, and no longer doubled
    (a file Blender wrote still has its specular level there, where 0.5 is the whole
    highlight); COLLADA's is no longer doubled either.
  - No highlight (a Lambert material, a black specular color, a strength of zero, MTL's
    `illum` 0 or 1, an Irrlicht shininess of 0) is a roughness of 1 and a `specularFactor`
    of 0. It was 1 in some readers and 0.85 in others. A format's own default counts: an
    X3D `Material`, an OGRE pass and a `KHR_materials_common` material have a black
    specular color unless they say otherwise, so they have no highlight.
  - A material that says nothing of its highlight keeps a roughness of 0.85 (SketchUp's
    were 0.9).
  - OBJ and FBX files Blender wrote hold a roughness in the exponent's place, and are read
    back as Blender wrote them, to the same roughness as before.
- Faces with no material take one default material in every reader but glTF's: named
  `default`, gray (0.72, 0.72, 0.74), not metallic, roughness 0.85, and white under vertex
  colors or a texture, which show as they are. COLLADA, OpenGEX, DirectX, Blitz3D, MilkShape
  3D and MikuMikuDance left such faces with no material; OBJ gave them a 0.604 gray, XGL
  and AC3D white, SketchUp a white `Default`, LightWave a 0.78 gray `Default`, OpenFlight
  `material-1`, X3D an unnamed gray, Irrlicht a white `material N`. A material that is
  named but not defined (an OBJ library that isn't there, a LightWave tag with no surface,
  a COLLADA material with no effect) has the default's look under its own name. OpenGEX's
  `materials` fact still counts the file's materials.
- One rule decides how a material's transparency is drawn. It is cut out (`MASK`) where the
  format asks for an alpha test, blended (`BLEND`) where the material, its color texture's
  alpha or its vertices' colors are less than opaque, and opaque otherwise.
  - Vertex colors with an alpha below 1 make their material blend in every reader. 3MF,
    VTK, PLY, OFF and the others left it opaque, which hid the alpha; AMF blended already.
    A `Mesh` turned into a document (`Document::from`, `Mesh::to_glb`) keeps the alpha
    modes its own materials give.
  - Blitz3D: a texture's flags are honored. Masked (4) cuts out at 0.5, alpha (2) blends;
    they were read and dropped.
  - MikuMikuDance: whether a material is cut out is decided by its texture's name alone.
    It used to need the file found, so the same model read differently with and without
    its textures beside it.
  - OBJ (a `map_d` naming the `map_Kd` file) and SketchUp (a PNG texture with an alpha
    channel) still cut out at 0.5, as before: the cutout is a hint each reader hands the
    rule, as MikuMikuDance's is.
- LightWave: a surface's diffuse level scales its color in the values the file stores, as
  its specular level scales the highlight: a 0.5 gray at a level of 0.8 is sRGB 0.4 (0.133
  linear), where it was 0.8 of 0.5's linear value (0.171). A node's linear color is scaled
  as it was.
- COLLADA: a camera with only `<xfov>` (no `<yfov>` or `<aspect_ratio>`) wrote its
  horizontal angle as glTF's vertical `yfov`. A 16:9 frame is now assumed and written as
  `aspectRatio` with the `yfov` that gives the file's angle across, as for OpenGEX.
- COLLADA and glTF 1.0 light colors are made linear, as the materials' colors are: a
  light of (1, 0.5, 0) is (1, 0.214, 0).
- `Document::to_fbx` writes `Shininess` as the exponent this rule reads back,
  `2 / roughness^4 - 2`, where it wrote `((1 - roughness) * 10)^2`. The `glTF|roughness`
  property that holds the roughness itself is unchanged.
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
- Animations are written one way by every reader but glTF's (a glTF, glTF 1.0, VRM or 3D Tiles
  file's are kept as the file has them). FBX, COLLADA, USD, 3DS, OpenGEX, BVH, MagicaVoxel and
  OpenFlight wrote their own until now, each a little differently:
  - **An animation starts at time 0.** The time of its earliest key is taken off every key of
    every channel, the same amount for all of them, and kept as `timeOffset` (seconds) in the
    animation's `extras`: a time in the file is the animation's time plus `timeOffset`. FBX
    moved a negative start to 0 one node at a time, so two nodes keyed from different times
    came out of step; it also left a take that starts late where it was, as USD, COLLADA
    (clips too) and BVH did, and 3DS kept negative times, which glTF doesn't allow. SMD and
    TRC already started at 0 and now say what they took off. An FBX stack's `start` and
    `stop` in `extras` are in the animation's time too. A COLLADA clip has keys at its
    `start` and `end`, whether or not the curves have keys there.
  - **Keys are finite, their times strictly increasing, and rotations unit quaternions on one
    side**, by one piece of code (`formats::keys`): OpenGEX left out a track whose times
    repeated, COLLADA dropped the later key.
  - **Names.** An animation the file names keeps its name; one it doesn't name has none. The
    made-up `motion` (BVH, AMC, C3D, TRC, HTR), `keyframer` (3DS), `frames` (MagicaVoxel, and
    MD2 frames with no name), `sequences` (OpenFlight), `anim0` (IQM) and the model's or
    node's name on a MilkShape 3D, Blitz3D, MD3 or Irrlicht animation are gone. A file that
    is one clip (BVH, AMC, C3D, TRC, HTR, VMD, `.md5anim`, SMD) names it as its file is named,
    without the extension: `walk.bvh` holds `walk`.
  - **Frame rates.** Where a file gives no rate and its format's own tools document no
    default, the rate is one assumed 30 a second: TRC took 60, IQM 24, MD3 and MagicaVoxel 10,
    an OpenFlight sequence with no duration a tenth of a second a frame (it now reports the
    assumed `frameRate`). Formats with a documented default keep it, still as assumed: MD2's
    10 (the engine's 100 ms step), MD5's 24 (the engine's), Blitz3D's 60 (its specification),
    DirectX's 4800 ticks (D3DX), AMC's 120 (the CMU database's capture rate), USD's 24.
  - All of a document's keys are in one buffer view, and channels with the same times share
    them, as the game formats' already were and did.
- Skins are written one way by every reader but glTF's:
  - **Up to sixteen joints move a vertex** everywhere: OpenGEX and OGRE kept four. A joint
    listed twice is one, the strongest are kept, strongest first, and the weights sum to 1.
  - **A vertex no joint moves stays with the node that draws its mesh**, in every format: that
    node joins the skin as a joint after the file's, bound where it stands, and only in a
    skin with such a vertex. COLLADA, USD, MD5, IQM, SMD, OGRE and OpenGEX bound such a vertex
    to the skin's first joint, so it followed that joint's animation. FBX added its model
    node to every skin, needed or not: an FBX skin now has one joint fewer unless a vertex
    needs it.
  - **`skin.skeleton` is always the joints' common root** (the nearest node every joint is or
    is under). Only COLLADA and USD set it; USD set the skeleton's first joint.
  - Joints are bytes whenever the skin has no more than 256 of them (FBX and USD always
    wrote shorts).
- Textures a model names are found one way by every reader:
  - **glTF** images named by URI go through the same search as every other format's when
    the file isn't where the URI says: by file name beside the model or in the pack, in any
    case, or in another image format. Another machine's absolute path
    (`C:\Old\Path\Tex.PNG`) or a `file:` URI was refused before.
  - **`file:` URIs and percent-escapes** (`file:///C:/Old%20Path/tex.png`, `my%20tex.png`) are
    read in every format, where only COLLADA did; a name is tried as written first.
  - **LightWave** no longer looks in the folder above the model's, outside what the caller
    gave it; what the assets hold is still found by name.
  - IQM, MD3 and MD5 find a texture named without an extension the same way: as written (a
    file of that name in any image format), then by the format's own ending.
  - **A texture that isn't there is reported by every reader**: `droppedTextures` counts the
    textures a model names that its document doesn't have, and `missingFiles` lists the
    names of those not found. OBJ, 3DS, FBX, COLLADA, USD, X3D, PLY, MikuMikuDance, DirectX,
    MD2, MD3, MD5, IQM, SMD, MilkShape 3D, Blitz3D and Irrlicht said nothing; OGRE's
    `missing_textures` fact is `missingFiles` now.
  - **One default sampler**: repeating, linear, and linear between linear mipmaps, unless the
    file says otherwise. SketchUp wrote no filters, and an Irrlicht material that states
    none took `LINEAR_MIPMAP_NEAREST`.

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
