//! GIF's first frame, transparency included.

use super::lzw::{self, Codes};
use super::{FREE, Raster, fill, get, le16, pixels};

pub(super) fn decode(b: &[u8]) -> Option<Raster> {
    let (sw, sh, flags) = (le16(b, 6)?, le16(b, 8)?, *b.get(10)?);
    let mut at = 13;
    let mut global: &[u8] = &[];
    if flags & 0x80 != 0 {
        let len = 3 * (2 << (flags & 7));
        global = b.get(at..).unwrap_or_default();
        global = &global[..global.len().min(len)];
        at += len;
    }
    let mut transparent = None;
    loop {
        match *b.get(at)? {
            0x21 => {
                let label = *b.get(at + 1)?;
                at += 2;
                // A graphic control extension's transparent color.
                if label == 0xF9 && b.get(at).is_some_and(|&n| n >= 4) {
                    let block = get(b, at + 1, 4)?;
                    transparent = (block[0] & 1 != 0).then_some(block[3]);
                }
                at = skip(b, at)?;
            }
            0x2C => break,
            _ => return None,
        }
    }
    let d = get(b, at, 10)?;
    let (x, y, fw, fh, fflags) = (le16(d, 1)?, le16(d, 3)?, le16(d, 5)?, le16(d, 7)?, d[9]);
    at += 10;
    let mut table = global;
    if fflags & 0x80 != 0 {
        let len = 3 * (2 << (fflags & 7));
        table = b.get(at..).unwrap_or_default();
        table = &table[..table.len().min(len)];
        at += len;
    }
    let frame = pixels(u32::from(fw), u32::from(fh))?;
    let min = u32::from(*b.get(at)?);
    at += 1;
    // The coded data, gathered out of its sub-blocks.
    let mut coded = Vec::new();
    while let Some(&n) = b.get(at) {
        if n == 0 {
            break;
        }
        let end = b.len().min(at + 1 + usize::from(n));
        coded.extend_from_slice(&b[at + 1..end]);
        at = end;
    }
    let mut idx = Vec::new();
    lzw::decode(
        &coded,
        Codes {
            min,
            msb: false,
            early: false,
        },
        &mut idx,
        frame,
    );
    fill(&mut idx, frame)?;
    let (fw, fh) = (usize::from(fw), usize::from(fh));
    if fflags & 0x40 != 0 {
        idx = deinterlace(&idx, fw, fh);
    }

    // The frame on the logical screen when that's not much larger; the frame alone otherwise.
    let screen = pixels(u32::from(sw), u32::from(sh)).filter(|&s| s <= (2 * frame).max(FREE / 4));
    let (w, h, ox, oy) = match screen {
        Some(_) => (usize::from(sw), usize::from(sh), usize::from(x), usize::from(y)),
        None => (fw, fh, 0, 0),
    };
    let mut px = vec![0u8; w * h * 4];
    for (r, row) in idx.chunks_exact(fw).enumerate() {
        let ty = oy + r;
        if ty >= h {
            break;
        }
        for (c, &i) in row.iter().enumerate() {
            let tx = ox + c;
            if tx >= w {
                break;
            }
            if transparent == Some(i) {
                continue;
            }
            let rgb = get(table, usize::from(i) * 3, 3).unwrap_or(&[0, 0, 0]);
            let o = (ty * w + tx) * 4;
            px[o..o + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }
    Some(Raster {
        w: u32::try_from(w).ok()?,
        h: u32::try_from(h).ok()?,
        channels: 4,
        data: px,
    })
}

/// Past the sub-blocks starting at `at`.
fn skip(b: &[u8], mut at: usize) -> Option<usize> {
    loop {
        let n = usize::from(*b.get(at)?);
        at += 1 + n;
        if n == 0 {
            return Some(at);
        }
    }
}

/// Interlaced rows (every 8th from 0, every 8th from 4, every 4th from 2, every 2nd from 1) put in order.
fn deinterlace(idx: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; idx.len()];
    let order = [(0, 8), (4, 8), (2, 4), (1, 2)]
        .into_iter()
        .flat_map(|(s, step)| (s..h).step_by(step));
    for (src, dst) in order.enumerate() {
        out[dst * w..(dst + 1) * w].copy_from_slice(&idx[src * w..(src + 1) * w]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::lzw::tests::encode;
    use super::super::tests::{decoded, survives_damage};
    use super::*;

    /// A GIF of one frame at (`x`, `y`) on a `screen`, with a 4-color global table (black, red, green, blue).
    fn gif(
        screen: (u16, u16),
        frame: (u16, u16, u16, u16),
        interlaced: bool,
        transparent: Option<u8>,
        idx: &[u8],
    ) -> Vec<u8> {
        let mut b = b"GIF89a".to_vec();
        b.extend(screen.0.to_le_bytes());
        b.extend(screen.1.to_le_bytes());
        b.extend([0x81, 0, 0]);
        b.extend([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
        if let Some(t) = transparent {
            b.extend([0x21, 0xF9, 4, 1, 0, 0, t, 0]);
        }
        // A comment, skipped.
        b.extend([0x21, 0xFE, 3, b'h', b'i', b'!', 0]);
        b.push(0x2C);
        for v in [frame.0, frame.1, frame.2, frame.3] {
            b.extend(v.to_le_bytes());
        }
        b.push(if interlaced { 0x40 } else { 0 });
        b.push(2);
        let coded = encode(
            idx,
            Codes {
                min: 2,
                msb: false,
                early: false,
            },
        );
        for chunk in coded.chunks(255) {
            b.push(u8::try_from(chunk.len()).unwrap());
            b.extend(chunk);
        }
        b.extend([0, 0x3B]);
        b
    }

    #[test]
    fn first_frame_with_transparency() {
        let (w, h, px) = decoded(&gif((2, 2), (0, 0, 2, 2), false, Some(0), &[1, 2, 3, 0]), "a.gif").unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(px, [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0]);
        // A frame smaller than the screen sits at its place, the rest clear.
        let px = decoded(&gif((3, 1), (2, 0, 1, 1), false, None, &[2]), "a.gif")
            .unwrap()
            .2;
        assert_eq!(px, [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255]);
        // A huge screen around a small frame is the frame alone.
        let (w, h, _) = decoded(&gif((16000, 16000), (5, 5, 2, 1), false, None, &[1, 1]), "a.gif").unwrap();
        assert_eq!((w, h), (2, 1));
    }

    #[test]
    fn interlaced_rows() {
        // Rows 0..5 coded in the order 0, 4, 2, 1, 3: each row's index names its place.
        let idx = [0, 0, 1, 1, 2, 2, 3, 3, 1, 1];
        let px = decoded(&gif((2, 5), (0, 0, 2, 5), true, None, &idx), "a.gif")
            .unwrap()
            .2;
        let rows: Vec<u8> = px.chunks(8).map(|r| r[0]).collect();
        // Row 0 black, 4 red, 2 green, 1 blue, 3 red: the red channel of each row.
        assert_eq!(rows, [0, 0, 0, 255, 255]);
        assert_eq!(px[8..12], [0, 0, 255, 255]);
    }

    #[test]
    fn damage() {
        let idx: Vec<u8> = (0..400u32).map(|i| u8::try_from(i * 7 % 4).unwrap()).collect();
        let b = gif((20, 20), (0, 0, 20, 20), true, Some(3), &idx);
        survives_damage(&b, "a.gif");
        // Cut short, the rest is padded.
        assert!(decoded(&b[..b.len() - 8], "a.gif").is_some());
        assert!(decoded(b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x3B", "a.gif").is_none());
    }
}
