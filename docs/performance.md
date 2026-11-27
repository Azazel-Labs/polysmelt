# Performance

`bench/` races polysmelt's readers against the leading Rust crate for each format, on load time
and peak heap. Each loader reads the same generated file (a torus of quads with normals and
texture coordinates) from disk into its own in-memory form. Loaders take turns in a rotating
order, so noise on a shared machine hits all of them alike, and a counting allocator measures
memory exactly.

Run it on a clean GitHub runner with `gh workflow run bench.yml` (or from the Actions tab); the
tables land in the run's summary. Times from separate runs aren't comparable, since each run
may get different hardware, but the ratios within one run are.

## Results

From [run 46010058100](https://github.com/Azazel-Labs/polysmelt/actions/runs/46010058100)
(November 27, 2026), at 2 million triangles. "Their time" is the other crate's time over
polysmelt's: above 1, polysmelt is faster.

| File | Size | polysmelt | Other crate | Their time | polysmelt peak | Their peak |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| OBJ | 154 MB | 344 ms | tobj, 1,054 ms | 3.1× | 120 MB | 315 MB |
| PLY, binary | 47 MB | 101 ms | ply-rs (typed), 377 ms | 3.7× | 54 MB | 71 MB |
| PLY, ASCII | 100 MB | 317 ms | ply-rs (typed), 1,238 ms | 3.9× | 54 MB | 71 MB |
| STL, ASCII | 460 MB | 1,240 ms | stl_io, 3,506 ms | 2.8× | 246 MB | 167 MB |
| STL, binary | 95 MB | 545 ms | stl_io, 391 ms | 0.72× | 244 MB | 167 MB |
| GLB | 54 MB | 12 ms | gltf, 41 ms | 3.3× | 107 MB | 107 MB |
| GLB, skinned | 73 MB | 33 ms | gltf, 52 ms | 1.6× | 191 MB | 149 MB |

The small (10,000 triangles) and medium (200,000) sizes give the same picture: polysmelt is
faster everywhere but binary STL, by 2 to 4 times, and by 1.2 to 1.6 times on skinned GLB.

## What the numbers include

The loaders don't all do the same work. polysmelt's result is a document any of its formats would
produce, so it does more than a plain parser:

- **STL.** An STL file stores every triangle separately, with a face normal. stl_io returns those
  triangles as they are. polysmelt joins them into shared vertices and builds smooth normals with
  a 30° crease angle (see [consistency.md](consistency.md#geometry)), so a model comes out ready
  to render with its hard edges kept. That costs time and memory, and on binary STL, where parsing
  itself is nearly free, it makes polysmelt the slower of the two. Making this pass cheaper is open
  work.
- **Skinned GLB.** polysmelt's default load poses skinned vertices in the rest pose. With
  `Pose::Bind`, which leaves skinning to the GPU, it takes 25 ms instead of 33.
- **OBJ.** polysmelt reads materials, groups and smoothing groups and builds normals where the
  file has none; tobj is asked for the same triangulated, single-index output.
