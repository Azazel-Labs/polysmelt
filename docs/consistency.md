# What every reader guarantees

polysmelt reads more than sixty formats written over four decades by tools that agreed on almost
nothing. Its readers don't pass those differences on to you. Every reader follows the rules
below, so code that works on a document from one format works on a document from any of them.

glTF is the one exception: it is already the target, so a glTF file is passed through as it is.

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
  is played at 30 frames a second, reported as an assumed `frameRate`.
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
  listed in `asset.extras.source.missingFiles`, and textures the document doesn't have are
  counted in `droppedTextures`.

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
