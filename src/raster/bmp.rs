//! Windows and OS/2 bitmaps: 1 to 32 bits a pixel, palette or bit-field masks, raw or RLE.

use std::borrow::Cow;

use super::{FREE, Raster, fill, flip, get, le16, le32, masked, may_grow, opaque_if_clear, pixels};
use crate::num::usize_of;

/// The PNG or JPEG a bitmap wraps (compression 4 or 5), which embeds as it is.
pub(super) fn wrapped(b: &[u8]) -> Option<&[u8]> {
    let header = le32(b, 14)?;
    if !(40..64).contains(&header) || !matches!(le32(b, 30)?, 4 | 5) {
        return None;
    }
    let data = b.get(usize_of(u64::from(le32(b, 10)?))..)?;
    let size = usize_of(u64::from(le32(b, 34)?));
    Some(if size > 0 {
        data.get(..size).unwrap_or(data)
    } else {
        data
    })
}

pub(super) fn decode(b: &[u8]) -> Option<Raster> {
    let offset = usize_of(u64::from(le32(b, 10)?));
    let header = usize_of(u64::from(le32(b, 14)?));
    let core = header == 12;
    let (w, h, bpp, compression, used) = if core {
        (u32::from(le16(b, 18)?), i64::from(le16(b, 20)?), le16(b, 24)?, 0, 0)
    } else if header >= 16 {
        let w = u32::try_from(le32(b, 18)?.cast_signed()).ok()?;
        let h = i64::from(le32(b, 22)?.cast_signed());
        let compression = if header >= 20 { le32(b, 30)? } else { 0 };
        let used = if header >= 36 { le32(b, 46)? } else { 0 };
        (w, h, le16(b, 28)?, compression, used)
    } else {
        return None;
    };
    // An OS/2 2.x header's compression 3 is Huffman coding, not bit fields.
    if header == 64 && compression == 3 {
        return None;
    }
    let top_down = h < 0;
    let h = u32::try_from(h.unsigned_abs()).ok()?;
    let n = pixels(w, h)?;
    let (wu, hu) = (usize_of(u64::from(w)), usize_of(u64::from(h)));
    let data = b.get(offset..).unwrap_or_default();

    // Indices into a palette, or colors through masks.
    let mut px = Vec::new();
    if matches!(bpp, 1 | 2 | 4 | 8) {
        let entry = if core { 3 } else { 4 };
        let mut at = 14 + header;
        if header == 40 {
            at += match compression {
                3 => 12,
                6 => 16,
                _ => 0,
            };
        }
        let count = if used > 0 {
            usize_of(u64::from(used)).min(256)
        } else {
            1 << bpp
        };
        let palette: Vec<[u8; 4]> = b
            .get(at..)
            .unwrap_or_default()
            .chunks_exact(entry)
            .take(count)
            .map(|p| [p[2], p[1], p[0], 255])
            .collect();
        let indices = match (compression, bpp) {
            (0, _) => rows(data, wu, hu, bpp)?,
            (1, 8) | (2, 4) => rle(data, wu, hu, bpp == 4)?,
            _ => return None,
        };
        px.reserve(n * 4);
        for i in indices {
            // A pixel RLE skipped over is transparent.
            let c = if i == u16::MAX {
                [0; 4]
            } else {
                palette.get(usize::from(i)).copied().unwrap_or([0, 0, 0, 255])
            };
            px.extend_from_slice(&c);
        }
    } else {
        let bytes = match bpp {
            16 => 2,
            24 => 3,
            32 => 4,
            _ => return None,
        };
        let (masks, alpha) = match (compression, bpp) {
            (0, 16) => ([0x7C00, 0x3E0, 0x1F], 0),
            (0, 24 | 32) => ([0xFF_0000, 0xFF00, 0xFF], if bpp == 32 { 0xFF00_0000 } else { 0 }),
            (3 | 6, 16 | 32) => {
                let m = [le32(b, 54)?, le32(b, 58)?, le32(b, 62)?];
                let alpha = if header >= 56 || compression == 6 {
                    le32(b, 66).unwrap_or(0)
                } else {
                    0
                };
                (m, alpha)
            }
            _ => return None,
        };
        let stride = (wu * usize::from(bpp)).div_ceil(32) * 4;
        // The rows in place when they're all there; a copy padded out when they fall short.
        let raw: Cow<'_, [u8]> = if let Some(all) = data.get(..stride * hu) {
            Cow::Borrowed(all)
        } else {
            let mut raw = data.to_vec();
            fill(&mut raw, stride * hu)?;
            Cow::Owned(raw)
        };
        let channels = if alpha == 0 { 3 } else { 4 };
        px.reserve(n * channels);
        // Rows top to bottom as they're read, so nothing is flipped after.
        let rows = raw.chunks_exact(stride);
        let rows: Box<dyn Iterator<Item = &[u8]>> = if top_down { Box::new(rows) } else { Box::new(rows.rev()) };
        let plain = masks == [0xFF_0000, 0xFF00, 0xFF] && matches!(alpha, 0 | 0xFF00_0000);
        for row in rows {
            let row = &row[..wu * bytes];
            match (plain, bytes) {
                // Blue, green, red (and alpha) bytes, as most bitmaps are.
                (true, 3) => {
                    for p in row.chunks_exact(3) {
                        px.extend_from_slice(&[p[2], p[1], p[0]]);
                    }
                }
                (true, 4) if alpha == 0 => {
                    for p in row.chunks_exact(4) {
                        px.extend_from_slice(&[p[2], p[1], p[0]]);
                    }
                }
                (true, 4) => {
                    for p in row.chunks_exact(4) {
                        px.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
                    }
                }
                _ => {
                    for p in row.chunks_exact(bytes) {
                        let mut v = [0u8; 4];
                        v[..bytes].copy_from_slice(p);
                        let v = u32::from_le_bytes(v);
                        px.extend_from_slice(&[masked(v, masks[0]), masked(v, masks[1]), masked(v, masks[2])]);
                        if alpha != 0 {
                            px.push(masked(v, alpha));
                        }
                    }
                }
            }
        }
        if channels == 4 {
            // 32-bit pixels' fourth byte is alpha unless it's zero throughout (it's padding then).
            opaque_if_clear(&mut px, 4);
        }
        return Some(Raster {
            w,
            h,
            channels,
            data: px,
        });
    }
    if !top_down {
        flip(&mut px, wu * 4);
    }
    Some(Raster {
        w,
        h,
        channels: 4,
        data: px,
    })
}

/// Uncompressed rows of `bpp`-bit palette indices, each padded to 4 bytes, as one index a pixel.
fn rows(data: &[u8], w: usize, h: usize, bpp: u16) -> Option<Vec<u16>> {
    let stride = (w * usize::from(bpp)).div_ceil(32) * 4;
    let need = stride * h;
    // Short data is judged by the pixels padding it makes, not by its packed bytes: at 1 bit a pixel those are an
    // eighth as many, and a few bytes would make hundreds of megabytes.
    if data.len() < need / 2 && w * h > FREE {
        return None;
    }
    let mut raw = data[..data.len().min(need)].to_vec();
    fill(&mut raw, need)?;
    let per = 8 / usize::from(bpp);
    let mask = u8::MAX >> (8 - bpp);
    let mut out = Vec::with_capacity(w * h);
    for row in raw.chunks_exact(stride) {
        out.extend((0..w).map(|x| {
            let shift = 8 - usize::from(bpp) * (x % per + 1);
            u16::from((row[x / per] >> shift) & mask)
        }));
    }
    Some(out)
}

/// RLE8 or RLE4 palette indices, as one a pixel; `u16::MAX` where the coding skipped a pixel.
fn rle(data: &[u8], w: usize, h: usize, four: bool) -> Option<Vec<u16>> {
    let total = w * h;
    let mut out: Vec<u16> = Vec::new();
    let (mut x, mut y, mut real, mut i) = (0usize, 0usize, 0usize, 0usize);
    // Pixels set at (x, y) and on: clipped at the row's end.
    let put = |x: &mut usize, y: usize, v: u16, out: &mut Vec<u16>, real: &mut usize| -> Option<()> {
        if *x < w && y < h {
            let at = y * w + *x;
            if at >= out.len() {
                if !may_grow(at + 1, *real + 1) {
                    return None;
                }
                out.resize(at + 1, u16::MAX);
            }
            out[at] = v;
            *real += 1;
        }
        *x += 1;
        Some(())
    };
    while y < h && i + 1 < data.len() {
        let (count, value) = (data[i], data[i + 1]);
        i += 2;
        if count > 0 {
            for k in 0..usize::from(count) {
                let v = if !four {
                    value
                } else if k.is_multiple_of(2) {
                    value >> 4
                } else {
                    value & 15
                };
                if x >= w {
                    break;
                }
                put(&mut x, y, u16::from(v), &mut out, &mut real)?;
            }
            continue;
        }
        match value {
            0 => (x, y) = (0, y + 1),
            1 => break,
            2 => {
                let (Some(&dx), Some(&dy)) = (data.get(i), data.get(i + 1)) else {
                    break;
                };
                i += 2;
                x += usize::from(dx);
                y += usize::from(dy);
            }
            n => {
                // n pixels as they are, padded to a 2-byte boundary.
                let n = usize::from(n);
                let len = if four { n.div_ceil(2) } else { n };
                let Some(src) = get(data, i, len) else { break };
                for k in 0..n {
                    let v = if !four {
                        src[k]
                    } else if k.is_multiple_of(2) {
                        src[k / 2] >> 4
                    } else {
                        src[k / 2] & 15
                    };
                    put(&mut x, y, u16::from(v), &mut out, &mut real)?;
                }
                i += len + len % 2;
            }
        }
    }
    if out.len() < total {
        if !may_grow(total, real.max(FREE / 2)) {
            return None;
        }
        out.resize(total, u16::MAX);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{decoded, survives_damage};

    /// A bitmap: a 40-byte (or `header`-byte) info header, then `palette`, `extra` and the pixels.
    fn bmp(header: u32, w: i32, h: i32, bpp: u16, compression: u32, extra: &[u8], data: &[u8]) -> Vec<u8> {
        let mut b = b"BM".to_vec();
        b.extend([0; 12]);
        b.extend(header.to_le_bytes());
        b.extend(w.to_le_bytes());
        b.extend(h.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(bpp.to_le_bytes());
        b.extend(compression.to_le_bytes());
        b.resize(14 + usize::try_from(header).unwrap(), 0);
        b.extend(extra);
        let offset = u32::try_from(b.len()).unwrap();
        b[10..14].copy_from_slice(&offset.to_le_bytes());
        b.extend(data);
        b
    }

    #[test]
    fn true_color_rows_are_padded_and_bottom_up() {
        // 1 × 2 at 24 bits: each row 3 bytes and a pad byte; bottom row blue.
        let b = bmp(40, 1, 2, 24, 0, &[], &[255, 0, 0, 0, 0, 0, 255, 0]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [255, 0, 0, 255, 0, 0, 255, 255]);
        // Top-down, as a negative height.
        let b = bmp(40, 1, -2, 24, 0, &[], &[255, 0, 0, 0, 0, 0, 255, 0]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [0, 0, 255, 255, 255, 0, 0, 255]);
        // 16 bits: 5-5-5 by default.
        let b = bmp(40, 2, 1, 16, 0, &[], &[0x00, 0x7C, 0x1F, 0x00]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [255, 0, 0, 255, 0, 0, 255, 255]);
        // 32 bits: the fourth byte is alpha, unless it's zero throughout.
        let b = bmp(40, 2, 1, 32, 0, &[], &[1, 2, 3, 0, 4, 5, 6, 0]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [3, 2, 1, 255, 6, 5, 4, 255]);
        let b = bmp(40, 2, 1, 32, 0, &[], &[1, 2, 3, 10, 4, 5, 6, 255]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [3, 2, 1, 10, 6, 5, 4, 255]);
        // Rows the file cuts short are padded with black.
        let b = bmp(40, 1, -2, 24, 0, &[], &[255, 0, 0, 0, 0]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [0, 0, 255, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn masks_and_alpha() {
        // 5-6-5 through bit fields after a 40-byte header.
        let masks: Vec<u8> = [0xF800u32, 0x07E0, 0x001F]
            .iter()
            .flat_map(|m| m.to_le_bytes())
            .collect();
        let b = bmp(40, 1, 1, 16, 3, &masks, &[0xE0, 0x07, 0, 0]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [0, 255, 0, 255]);
        // A version 5 header's alpha mask; 32-bit pixels.
        let mut v5 = vec![0u8; 0];
        for m in [0xFF_0000u32, 0xFF00, 0xFF, 0xFF00_0000] {
            v5.extend(m.to_le_bytes());
        }
        let mut b = bmp(124, 1, 1, 32, 3, &[], &[1, 2, 3, 128]);
        b[54..70].copy_from_slice(&v5);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [3, 2, 1, 128]);
        // Plain 32-bit pixels with the fourth byte zero are opaque.
        let b = bmp(40, 1, 1, 32, 0, &[], &[1, 2, 3, 0]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [3, 2, 1, 255]);
    }

    #[test]
    fn palettes() {
        // 1 bit, 9 pixels wide: two bytes and padding a row.
        let pal = [0, 0, 0, 0, 255, 255, 255, 0];
        let b = bmp(40, 9, 1, 1, 0, &pal, &[0b1000_0001, 0b1000_0000, 0, 0]);
        let px = decoded(&b, "a.bmp").unwrap().2;
        let white: Vec<bool> = px.chunks(4).map(|p| p[0] == 255).collect();
        assert_eq!(white, [true, false, false, false, false, false, false, true, true]);
        // 4 bits through an OS/2 core header, whose entries take 3 bytes.
        let mut b = b"BM".to_vec();
        b.extend([0; 12]);
        b.extend(12u32.to_le_bytes());
        b.extend(2u16.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(4u16.to_le_bytes());
        b.extend([0, 0, 200, 0, 100, 0]);
        b[10] = u8::try_from(b.len()).unwrap();
        b.extend([0x10, 0, 0, 0]);
        assert_eq!(decoded(&b, "a.bmp").unwrap().2, [0, 100, 0, 255, 200, 0, 0, 255]);
    }

    #[test]
    fn run_lengths() {
        let pal = [0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0];
        // RLE8, 4 × 2: a run of 3 red, end of line; a delta past one pixel, 3 literal pixels and a pad byte, end.
        let data = [3, 0, 0, 0, 0, 2, 1, 0, 0, 3, 1, 2, 0, 0, 0, 1];
        let b = bmp(40, 4, 2, 8, 1, &pal, &data);
        let px = decoded(&b, "a.bmp").unwrap().2;
        // Top row (the second coded) first: skipped, then green, blue and red.
        assert_eq!(px[..16], [0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 255]);
        assert_eq!(px[16..], [255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 0]);
        // RLE4: a run alternating two indices.
        let b = bmp(40, 3, 1, 4, 2, &pal, &[3, 0x12, 0, 1]);
        let px = decoded(&b, "a.bmp").unwrap().2;
        assert_eq!(px, [0, 255, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255]);
        // A delta far out over a huge image makes nothing.
        let b = bmp(40, 16000, 16000, 8, 1, &pal, &[0, 2, 255, 255, 1, 0]);
        assert!(decoded(&b, "a.bmp").is_none());
    }

    #[test]
    fn wrapped_png_and_damage() {
        let png = crate::png::encode(1, 1, 3, &[1, 2, 3]).unwrap();
        let mut b = bmp(40, 1, 1, 0, 5, &[], &png);
        b[34..38].copy_from_slice(&u32::try_from(png.len()).unwrap().to_le_bytes());
        let img = super::super::decode(&b, std::path::Path::new("a.bmp")).unwrap();
        assert_eq!(img.bytes, png);
        let huge = bmp(40, 16000, 16000, 24, 0, &[], &[1; 100]);
        assert!(decoded(&huge, "a.bmp").is_none());
        // At 1 bit a pixel 16000 x 2000 packs into 4 MB, which alone would pass as small.
        let pal = [0u8, 0, 0, 0, 255, 255, 255, 0];
        assert!(decoded(&bmp(40, 16000, 2000, 1, 0, &pal, &[0xAA; 8]), "a.bmp").is_none());
        // A small one still pads.
        assert!(decoded(&bmp(40, 64, 64, 1, 0, &pal, &[0xAA; 8]), "a.bmp").is_some());
        let pal = [9u8; 64];
        survives_damage(
            &bmp(
                40,
                7,
                5,
                8,
                1,
                &pal,
                &[3, 1, 0, 2, 1, 1, 0, 4, 1, 2, 3, 4, 0, 0, 5, 9, 0, 1],
            ),
            "a.bmp",
        );
        survives_damage(&bmp(40, 5, 3, 4, 0, &pal, &[0x12; 12]), "a.bmp");
        let masks: Vec<u8> = [0xF800u32, 0x07E0, 0x001F]
            .iter()
            .flat_map(|m| m.to_le_bytes())
            .collect();
        survives_damage(&bmp(40, 3, 3, 16, 3, &masks, &[0x55; 24]), "a.bmp");
    }
}
