# What every reader guarantees

polysmelt reads more than sixty formats written over four decades by tools that agreed on almost
nothing. Its readers don't pass those differences on to you. Every reader follows the rules
below, so code that works on a document from one format works on a document from any of them.

glTF 2.0 is the one exception: it is already the target, so a glTF 2.0 file (`.gltf` or `.glb`)
is passed through as it is, with no `RootNode`, its compression kept and its animations, skins
and names as the file has them.

The readers that convert into glTF are not passed through, and follow these rules like any
other: glTF 1.0, upgraded to 2.0 as it's read, and VRM avatars and animations each load under a
`RootNode`, and have their animations, skins and names held to the rules below. 3D Tiles is held
to the same animation, skin and name rules but has no `RootNode`: a tileset is Z up and centered
on the earth while each tile's glTF is Y up, so there's no one frame of the file to keep under a
root, and each tile is placed by the node over it. Tiles it can't find are listed in
`missingFiles`, as every reader lists the files it can't find. In all three, only what breaks a
rule is rewritten: a skin or an animation that already follows them is left as the file has it.

## Axes and units

The document is in glTF's frame: +Y up, right-handed, meters.

- **One root node.** A format with a scene (nodes, meshes, skeletons, animation, cameras or
  lights) loads under exactly one top-level node, always named `RootNode`. Its rotation turns
  the file's up axis to +Y and its scale turns the file's unit into meters. Everything under it
  stays in the file's own axes and units, so nothing is lost and the numbers match what the
  authoring tool shows. A file that needs no conversion still gets a `RootNode`, with no
  transform, so code walking the tree sees one shape whatever the format.
- **Mirrors are baked.** A left-handed file is mirrored as it is read (vertices, normals,
  transforms, bind matrices and keys, with triangles wound the other way), never left as a
  negative scale.
- **Sampled data is converted in place.** Point clouds, gaussian splats and voxel grids have no
  scene to hang a root node on, so they are turned and scaled in their data.
- **Nothing is guessed silently.** A file that doesn't state its unit, in a format that defines
  none, is not scaled and its unit is unknown. A file that doesn't state its up axis gets its
  format's convention where there is one (lidar text clouds are Z up), and is left alone where
  there isn't (PLY, OFF, VTK and PCD).
- **It's all on record.** `Document::source_info()` tells what the file declared
  (`Fact::Declared`), what was assumed (`Fact::Assumed`) and what was converted
  (`converted.up_axis`, `converted.units`).

## Geometry

- **What can't be drawn is dropped and counted.** A triangle, segment or point that names a
  vertex that isn't there, or whose position isn't a finite number, is dropped: never moved to
  the origin, and never an error for the whole file. The count is in
  `asset.extras.source.droppedGeometry`. No document holds a NaN position or an index past its
  vertices.
- **Degenerate triangles are kept.** Triangles with no area or a repeated corner hold a mesh's
  topology together, and other tools count them, so they stay as the file has them.
- **Every triangle mesh has normals.** A file's own normals are used, made unit length. Where a
  file has none, its format's smoothing data is followed (OBJ and 3DS smoothing groups, FBX
  smoothing, X3D's `creaseAngle`, LightWave's smoothing angle, SketchUp's soft edges). Where it
  has neither, normals are made with a 30° crease angle: a box keeps its hard edges and a curved
  surface comes out smooth, whatever format it came in. Change the angle with
  `Keep::with_crease_angle(degrees)`: 0 for flat faces, 180 for smooth everywhere.

## Materials

- **One formula for classic materials.** Phong, Blinn and Lambert materials become glTF PBR the
  same way in every reader. A Blinn-Phong exponent `n` becomes a roughness of
  `(2 / (n + 2))^(1/4)`, after each format's own scale is applied to the exponent. The specular
  color becomes `KHR_materials_specular`. Before this, the same exponent of 20 came out as
  anything from 0.30 to 0.86 depending on the reader.
- **One default material.** Faces with no material get one material named `default`: gray
  (0.72, 0.72, 0.74), not metallic, roughness 0.85, white under vertex colors or a texture.
- **One transparency rule.** A material is cut out (`MASK`) where its format asks for an alpha
  test, blended (`BLEND`) where its color, its color texture's alpha or its vertex colors are
  less than opaque, and opaque otherwise.

## Animation and skins

- **Animations start at 0.** The earliest key is subtracted from every channel, and what was
  taken off is kept as the animation's `extras.timeOffset`, in seconds.
- **Keys are clean.** Times are finite and increasing, rotations are unit quaternions that turn
  the short way, and the file's curve is kept wherever glTF can hold it (step, linear, cubic
  spline). Curves glTF can't hold are sampled into linear keys.
- **One frame-rate default.** A format that stores frames without saying how fast to play them
  is played at 30 frames a second, reported as an assumed `frameRate`. A rate the file gives
  always wins, declared. Where a format has a convention of its own, that stands in instead,
  still assumed, and its reader cites where it comes from: MD2 10 (Quake II's game tick), MD5
  24 (Doom 3's starting value), Blitz3D 60 (its specification), DirectX 4800 ticks (a
  convention: Microsoft documents no default), AMC 120 (the CMU database's capture rate) and
  USD 24 time codes (its specification). Where a format gives each animation a rate of its own
  (IQM), and the rates differ, each animation's is its `extras.frameRate`, and one that was
  assumed is listed in its `extras.assumed` (`["frameRate"]`), as the source info lists it.
- **One pass.** A glTF animation plays once. A sequence the file plays a set number of times is
  played out in full; one it repeats forever is played once, and what the file says about
  repeating stays in `extras` (IQM's `loop`, OpenFlight's `loopCount`, beside `loopsPlayed`).
- **Names are the file's.** An animation keeps the name the file gives it. A one-clip file
  (BVH, AMC, C3D, TRC, HTR, VMD, `.md5anim`, SMD) is named after the file. Nothing gets a
  made-up name like `motion`.
- **One skinning rule.** Up to 16 influences per vertex, duplicates merged, weights normalized.
  A vertex no bone weights follows the node that draws the mesh, not whichever joint happens to
  be first. `skin.skeleton` is always set to the joints' common root.

## Textures

- **One lookup.** Every reader finds referenced files the same way: as written, then
  percent-decoded, with `file:` URIs understood, through the `Assets` you give the loader.
- **Missing files are reported, not fatal.** Files a model refers to that weren't found are
  listed in `asset.extras.source.missingFiles`, each once, by the name the model gives. Textures
  the document doesn't have are counted in `droppedTextures`: each distinct texture once, however
  many materials, objects or palette entries use it. A texture isn't there when no file is found
  for its name, or when the file found isn't an image that can be embedded (known when images
  are read, with `Keep::image_bytes`). The same name is the same texture, and so are names that
  find the same file. In glTF, where materials share images through textures, each image is a
  texture: an image a texture can't be drawn from (not found, or in a format Polysmelt doesn't
  decode, such as KTX2) is counted once however many textures use it, and only when the texture
  has no other image to draw. A read without textures or materials reports neither, as it
  doesn't look.

## Errors

The same failure is the same `ErrorKind` from every reader:

| Kind | When |
| --- | --- |
| `Empty` | The file has no bytes, or is well formed but holds nothing to convert. |
| `UnknownFormat` | The file isn't the format it's read as: its magic bytes or first words are wrong. |
| `Truncated` | The file is cut short. Where a reader can keep what came before the cut, it returns that instead, with `asset.extras.source.truncated` set to `true`. |
| `Corrupt` | The file is the right format but its contents contradict themselves. |
| `Unsupported` | A version or feature the reader says it doesn't handle, such as FBX before 7. |
| `LimitExceeded` | More than `Keep::max_triangles` or `Keep::max_points`, against runaway or hostile files. |

Every error names the file it came from.

## Facts and names

- **One vocabulary.** `asset.extras.source` uses the same keys in every reader: `format`,
  `version`, `tool`, `unit` (always the unit's usual name, such as `"millimeter"`, with the
  file's own spelling in `unitAsWritten`), `encoding` (`"ascii"` or `"binary"`), `byteOrder`
  (`"little"` or `"big"`), `droppedGeometry`, `droppedTextures`, `missingFiles`, `truncated`.
  Every reader also fills `Document::facts` with its counts of triangles, and of lines and
  points where there are any.
- **One text decoder.** Names are read as UTF-8 (a byte order mark dropped, UTF-16 with one
  decoded) and otherwise as Windows-1252, unless the file or its format states an encoding.
- **No empty names.** Something without a name has no `name` key, never `""`. Names a reader
  makes up for things a format only numbers are the kind and the number: `material3`,
  `object1`.
