//! A tileset: a tree of tiles, each with a transform, a geometric error, content and children
//! that refine it.
//!
//! What's read is the tileset at its finest: a tile that refines by `REPLACE` gives way to its
//! children, one that refines by `ADD` is drawn with them, and so on down to the leaves. A
//! tile's `refine` is its parent's when it has none, and the root's is `REPLACE`. Each content
//! read sits under a node named for its URI, with the product of the transforms from the root
//! down to its tile (turned Y up) as its matrix, in double precision.
//!
//! - Content is found by its `uri` (or 1.0's `url`), relative to the tileset, through the
//!   [`Assets`](crate::assets::Assets): any tile format, a glTF or GLB (3D Tiles 1.1), or
//!   another tileset, read the same way under its tile. A 1.1 tile's several `contents` are
//!   all read.
//! - At most [`MAX_TILES`] tiles are drawn. Past that the tiles with the largest geometric
//!   error are refined first and the rest are drawn unrefined, which the `unrefined_tiles`
//!   fact says.
//! - Content that isn't among the assets (or is named by an absolute URL) is left out and
//!   listed in `asset.extras.source.missingTiles`; when none of it is there, that's an error.
//! - `asset.gltfUpAxis` (`X`, `Y` or `Z`, from before 1.0) says which way is up in the glTF
//!   of the tileset's tiles.
//!
//! Not read: implicit tiling (`implicitTiling`, or the `3DTILES_implicit_tiling` extension),
//! which is refused; bounding volumes, viewer request volumes, metadata, groups and
//! `properties`, which nothing here needs.

use serde_json::{Map, Value};

use super::xform::{self, IDENTITY, M4};
use super::{Ctx, MAX_DEPTH, Merged, Stats, Up, Y_TO_Z, Z_TO_Y, content, fetch, reference};
use crate::document::Document;
use crate::formats::Input;
use crate::{ErrorKind, Result};

/// The most tiles of a tileset that are drawn (see the [module](self) docs), tilesets inside
/// it counted.
pub(super) const MAX_TILES: usize = 1024;

/// A tile on its way to being drawn or refined.
struct Tile {
    json: Value,
    /// The tile's space to the tileset's: every transform from the root down to its own.
    transform: M4,
    /// Whether its children are drawn with it (`ADD`) or in its place.
    add: bool,
    error: f64,
    /// Where it is in the tree, as the index of each tile on the way down among its parent's
    /// children: tiles are drawn in the tileset's own order.
    path: Vec<usize>,
}

impl Tile {
    fn new(mut json: Value, parent: &M4, parent_adds: bool, path: Vec<usize>) -> Result<Self> {
        ensure!(
            json.get("implicitTiling").is_none() && json.pointer("/extensions/3DTILES_implicit_tiling").is_none(),
            Unsupported,
            "3D Tiles implicit tiling isn't read"
        );
        let transform = match json.get("transform") {
            None | Some(Value::Null) => *parent,
            Some(t) => {
                let m: Vec<f64> = t
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_f64)
                    .filter(|x| x.is_finite())
                    .collect();
                ensure!(
                    m.len() == 16 && t.as_array().is_some_and(|t| t.len() == 16),
                    Corrupt,
                    "3D Tiles tile transform isn't sixteen numbers"
                );
                let mut flat = [0.0; 16];
                flat.copy_from_slice(&m);
                xform::mul(parent, &xform::from_column_major(&flat))
            }
        };
        let add = match json.get("refine").and_then(Value::as_str) {
            Some(r) if r.eq_ignore_ascii_case("ADD") => true,
            Some(r) if r.eq_ignore_ascii_case("REPLACE") => false,
            _ => parent_adds,
        };
        let error = json
            .get("geometricError")
            .and_then(Value::as_f64)
            .filter(|e| e.is_finite())
            .unwrap_or(0.0);
        // The children are taken out as tiles of their own, so a tile's JSON is its own alone.
        let _ = json.get_mut("children").filter(|c| !c.is_array()).map(Value::take);
        Ok(Self {
            json,
            transform,
            add,
            error,
            path,
        })
    }

    fn children(&mut self) -> Vec<Value> {
        match self.json.get_mut("children").map(Value::take) {
            Some(Value::Array(list)) => list,
            _ => Vec::new(),
        }
    }

    fn child_count(&self) -> usize {
        self.json.get("children").and_then(Value::as_array).map_or(0, Vec::len)
    }

    /// The URIs of the tile's content.
    fn uris(&self) -> Vec<&str> {
        fn uri(c: &Value) -> Option<&str> {
            c.get("uri").or_else(|| c.get("url"))?.as_str()
        }
        let one = self.json.get("content").and_then(uri);
        let more = self
            .json
            .get("contents")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(uri);
        one.into_iter().chain(more).collect()
    }
}

pub(super) fn read(ctx: &Ctx<'_>, json: &Map<String, Value>, stats: &mut Stats) -> Result<Document> {
    ensure!(
        ctx.depth < MAX_DEPTH,
        LimitExceeded,
        "3D Tiles tilesets nest more than {MAX_DEPTH} deep"
    );
    let asset = json.get("asset");
    if stats.version.is_none() {
        stats.version = asset
            .and_then(|a| a.get("version"))
            .and_then(Value::as_str)
            .map(str::to_owned);
    }
    let up = match asset.and_then(|a| a.get("gltfUpAxis")).and_then(Value::as_str) {
        Some(a) if a.eq_ignore_ascii_case("X") => Up::X,
        Some(a) if a.eq_ignore_ascii_case("Z") => Up::Z,
        Some(a) if a.eq_ignore_ascii_case("Y") => Up::Y,
        _ => ctx.up,
    };
    ensure!(
        json.get("extensions")
            .and_then(|e| e.get("3DTILES_implicit_tiling"))
            .is_none(),
        Unsupported,
        "3D Tiles implicit tiling isn't read"
    );
    let Some(root) = json.get("root").filter(|r| r.is_object()) else {
        bail!(Corrupt, "3D Tiles tileset has no root tile");
    };

    // The tiles to draw: refine the coarsest tile left while the count stays under the cap.
    let budget = MAX_TILES.saturating_sub(stats.tiles).max(1);
    let mut open = vec![Tile::new(root.clone(), &IDENTITY, false, Vec::new())?];
    let mut drawn: Vec<Tile> = Vec::new();
    while !open.is_empty() {
        let coarsest = (0..open.len())
            .max_by(|&a, &b| open[a].error.total_cmp(&open[b].error))
            .unwrap_or(0);
        let mut tile = open.swap_remove(coarsest);
        let kids = tile.child_count();
        if kids == 0 {
            drawn.push(tile);
            continue;
        }
        if drawn.len() + open.len() + kids + usize::from(tile.add) > budget {
            stats.capped = true;
            drawn.push(tile);
            continue;
        }
        for (i, child) in tile.children().into_iter().enumerate() {
            let mut path = tile.path.clone();
            path.push(i);
            open.push(Tile::new(child, &tile.transform, tile.add, path)?);
        }
        if tile.add {
            drawn.push(tile);
        }
    }
    drawn.sort_by(|a, b| a.path.cmp(&b.path));

    let mut out = Merged::new("tileset");
    let mut read = 0;
    let before = stats.missing.len();
    for tile in &drawn {
        let place = xform::mul(&Z_TO_Y, &xform::mul(&tile.transform, &Y_TO_Z));
        for uri in tile.uris() {
            let found = reference(ctx, uri).and_then(|path| Ok((fetch(ctx, &path, uri)?, path)));
            let (bytes, path) = match found {
                Ok(found) => found,
                Err(e) if matches!(e.kind(), ErrorKind::MissingAsset | ErrorKind::Unsupported) => {
                    stats.missing.push(uri.to_owned());
                    continue;
                }
                Err(e) => return Err(e),
            };
            let sub = Ctx {
                input: Input {
                    limit: ctx.input.limit,
                    ..Input::shared(&bytes, &path)
                },
                up,
                depth: ctx.depth + 1,
                ..*ctx
            };
            let listed = stats.missing.len();
            let doc = match content(&sub, &bytes, stats) {
                Ok(doc) => doc,
                // A tile whose own glTF, or a tileset all of whose content, isn't there.
                Err(e) if e.kind() == ErrorKind::MissingAsset => {
                    if stats.missing.len() == listed {
                        stats.missing.push(uri.to_owned());
                    }
                    continue;
                }
                Err(e) => return Err(e.at(&path)),
            };
            out.add(doc, &place, Some(uri), ctx.input.limit)?;
            read += 1;
        }
    }
    if read == 0 && stats.missing.len() > before {
        let missing = &stats.missing[before..];
        return Err(crate::Error::missing(format!(
            "none of the 3D Tiles tileset's content is among the assets: {:?}{}",
            missing[0],
            if missing.len() > 1 {
                format!(" and {} more", missing.len() - 1)
            } else {
                String::new()
            }
        )));
    }
    stats.kind("tileset");
    Ok(out.doc)
}
