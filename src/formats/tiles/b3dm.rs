//! Batched 3D Model: a binary glTF whose vertices carry a batch id, and a batch table of
//! properties for each id.
//!
//! The glTF is the document, untouched but for its `BATCHID` attribute's name. A node is put
//! over what it draws when there's something for it to hold: the `RTC_CENTER` (with the glTF's
//! own `CESIUM_RTC` center) as its translation, turned Y up, and in its extras `tile`
//! (`"b3dm"`), `batchLength` and the `batchTable`.

use super::xform::IDENTITY;
use super::{Ctx, Stats, batch_extras, model, place, table, wrap};
use crate::document::Document;
use crate::num::usize_of;
use crate::{Result, Within as _};

pub(super) fn read(ctx: &Ctx<'_>, data: &[u8], stats: &mut Stats) -> Result<Document> {
    let tile = table::parse(data, "b3dm")?;
    let length = match tile.batch_length {
        Some(n) => n,
        None => tile.feature.count("BATCH_LENGTH")?.unwrap_or(0),
    };
    let rtc = tile.feature.vec3("RTC_CENTER")?.unwrap_or([0.0; 3]);
    ensure!(
        !tile.body.is_empty(),
        Truncated,
        "b3dm tile has no glTF after its tables"
    );
    let (mut doc, center) = model(ctx, tile.body).within("reading a b3dm tile's glTF")?;
    let center = [rtc[0] + center[0], rtc[1] + center[1], rtc[2] + center[2]];
    let extras = if ctx.keep.extras && (length > 0 || !tile.batch.json.is_empty()) {
        Some(batch_extras("b3dm", &tile.batch, length, usize_of(length))?)
    } else {
        None
    };
    // The node over the glTF, when there's something for it to hold.
    let m = place(ctx.up, center);
    if m != IDENTITY || extras.is_some() {
        wrap(&mut doc, &m, "b3dm", extras);
    }
    stats.kind("b3dm");
    stats.tiles += 1;
    stats.features += length;
    Ok(doc)
}
