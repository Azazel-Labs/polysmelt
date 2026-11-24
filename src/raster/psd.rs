//! A Photoshop document's merged image (PSD, or large-document PSB): bitmap, gray, duotone, indexed, RGB or CMYK,
//! 8 or 16 bits, raw or RLE.

use super::{FREE, Raster, be16, be32, fill, get, packbits, pixels};
use crate::num::usize_of;

pub(super) fn decode(b: &[u8]) -> Option<Raster> {
    let version = be16(b, 4)?;
    let large = match version {
        1 => false,
        2 => true,
        _ => return None,
    };
    let channels = usize::from(be16(b, 12)?);
    let (h, w) = (be32(b, 14)?, be32(b, 18)?);
    let (depth, mode) = (be16(b, 22)?, be16(b, 24)?);
    let n = pixels(w, h)?;
    let base = match (mode, depth) {
        (0, 1) | (1 | 2 | 8, 8) | (1 | 8, 16) => 1,
        (3, 8 | 16) => 3,
        (4, 8 | 16) => 4,
        _ => return None,
    };
    if channels < base {
        return None;
    }
    // Sections: the color mode data (an indexed image's palette), resources, layers; then the merged image.
    let mut at = 26;
    let cmap_len = usize_of(u64::from(be32(b, at)?));
    let palette = get(b, at + 4, 768).filter(|_| mode == 2 && cmap_len >= 768);
    at = at.checked_add(4 + cmap_len)?;
    at = at.checked_add(4 + usize_of(u64::from(be32(b, at)?)))?;
    at = if large {
        let len = u64::from(be32(b, at)?) << 32 | u64::from(be32(b, at + 4)?);
        at.checked_add(8)?.checked_add(usize::try_from(len).ok()?)?
    } else {
        at.checked_add(4 + usize_of(u64::from(be32(b, at)?)))?
    };
    let compression = be16(b, at)?;
    at += 2;
    // Only channels it shows are decoded: the colors, and an alpha channel after them.
    let alpha = channels > base && mode != 0 && mode != 2;
    let used = base + usize::from(alpha);
    let (wu, hu) = (usize_of(u64::from(w)), usize_of(u64::from(h)));
    let row = (wu * usize::from(depth)).div_ceil(8);
    let need = row.checked_mul(hu)?.checked_mul(used)?;
    let mut planes = Vec::new();
    match compression {
        0 => {
            let data = b.get(at..).unwrap_or_default();
            planes.extend_from_slice(&data[..data.len().min(need)]);
        }
        1 => {
            // Each row's coded length, every channel's rows, then the rows.
            let size = if large { 4 } else { 2 };
            let table = at;
            let mut src = table.checked_add(channels.checked_mul(hu)?.checked_mul(size)?)?;
            for i in 0..used * hu {
                let len = if large {
                    be32(b, table + i * 4)
                } else {
                    be16(b, table + i * 2).map(u32::from)
                };
                let Some(len) = len.map(|l| usize_of(u64::from(l))) else {
                    break;
                };
                let coded = b.get(src..).unwrap_or_default();
                let start = planes.len();
                packbits(&coded[..coded.len().min(len)], &mut planes, start + row);
                // A short row is padded, keeping the rows after it in place, while data lasts.
                if coded.is_empty() {
                    break;
                }
                planes.resize(start + row, 0);
                src = src.saturating_add(len);
            }
        }
        _ => return None,
    }
    if planes.len() < need / 2 && need > FREE {
        return None;
    }
    fill(&mut planes, need)?;
    let plane = row * hu;
    let sample = |c: usize, x: usize, y: usize| -> u8 {
        let p = &planes[c * plane + y * row..];
        if depth == 16 { p[x * 2] } else { p[x] }
    };
    let channels_out = if base == 1 && mode != 2 { 1 } else { 3 } + usize::from(alpha);
    let mut px = Vec::with_capacity(n * channels_out);
    for y in 0..hu {
        for x in 0..wu {
            let mut c = [0u8; 4];
            match mode {
                0 => {
                    c[0] = if planes[y * row + x / 8] & (0x80 >> (x % 8)) != 0 {
                        0
                    } else {
                        255
                    }
                }
                2 => {
                    let i = usize::from(sample(0, x, y));
                    if let Some(p) = palette {
                        c[..3].copy_from_slice(&[p[i], p[256 + i], p[512 + i]]);
                    }
                }
                4 => {
                    // Stored inverted: 255 is no ink.
                    let k = u32::from(sample(3, x, y));
                    for (ch, v) in c[..3].iter_mut().enumerate() {
                        *v = u8::try_from(u32::from(sample(ch, x, y)) * k / 255).unwrap_or(u8::MAX);
                    }
                }
                _ => {
                    for (ch, v) in c[..base].iter_mut().enumerate() {
                        *v = sample(ch, x, y);
                    }
                }
            }
            let colors = channels_out - usize::from(alpha);
            if alpha {
                // The merged image's colors are blended over white: taken back off.
                let a = sample(base, x, y);
                if a > 0 && a < 255 {
                    for v in &mut c[..colors] {
                        let white = 255 * (255 - u32::from(a));
                        *v =
                            u8::try_from((u32::from(*v) * 255).saturating_sub(white) / u32::from(a)).unwrap_or(u8::MAX);
                    }
                }
                c[colors] = a;
            }
            px.extend_from_slice(&c[..channels_out]);
        }
    }
    Some(Raster {
        w,
        h,
        channels: channels_out,
        data: px,
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::{decoded, survives_damage};

    /// A document of `channels` planes, `w` × `h`, of `depth` bits in color `mode`, with the merged image `data`
    /// coded as `compression` says.
    fn psd(
        channels: u16,
        w: u32,
        h: u32,
        (depth, mode): (u16, u16),
        cmap: &[u8],
        compression: u16,
        data: &[u8],
    ) -> Vec<u8> {
        let mut b = b"8BPS".to_vec();
        b.extend(1u16.to_be_bytes());
        b.extend([0; 6]);
        b.extend(channels.to_be_bytes());
        b.extend(h.to_be_bytes());
        b.extend(w.to_be_bytes());
        b.extend(depth.to_be_bytes());
        b.extend(mode.to_be_bytes());
        b.extend(u32::try_from(cmap.len()).unwrap().to_be_bytes());
        b.extend(cmap);
        // Resources and layers, skipped.
        b.extend(4u32.to_be_bytes());
        b.extend(b"8BIM");
        b.extend(2u32.to_be_bytes());
        b.extend([0, 0]);
        b.extend(compression.to_be_bytes());
        b.extend(data);
        b
    }

    #[test]
    fn rgb_and_alpha() {
        // 2 × 1 RGB planes, raw.
        let b = psd(3, 2, 1, (8, 3), &[], 0, &[10, 20, 30, 40, 50, 60]);
        assert_eq!(decoded(&b, "a.psd").unwrap().2, [10, 30, 50, 255, 20, 40, 60, 255]);
        // With an alpha plane: colors blended over white come back.
        let b = psd(4, 1, 1, (8, 3), &[], 0, &[255, 127, 191, 128]);
        assert_eq!(decoded(&b, "a.psd").unwrap().2, [255, 0, 127, 128]);
        // 16 bits keep the high byte.
        let b = psd(1, 2, 1, (16, 1), &[], 0, &[0x12, 0x34, 0xAB, 0xCD]);
        assert_eq!(
            decoded(&b, "a.psd").unwrap().2,
            [0x12, 0x12, 0x12, 255, 0xAB, 0xAB, 0xAB, 255]
        );
    }

    #[test]
    fn rle_cmyk_indexed_and_bitmap() {
        // RLE, 3 × 2 gray: row lengths, then each row a run or literals.
        let mut data = Vec::new();
        for len in [2u16, 4] {
            data.extend(len.to_be_bytes());
        }
        data.extend([0xFE, 7, 2, 1, 2, 3]);
        let b = psd(1, 3, 2, (8, 1), &[], 1, &data);
        let gray: Vec<u8> = decoded(&b, "a.psd").unwrap().2.chunks(4).map(|p| p[0]).collect();
        assert_eq!(gray, [7, 7, 7, 1, 2, 3]);
        // CMYK, stored inverted: full cyan and no other ink.
        let b = psd(4, 1, 1, (8, 4), &[], 0, &[0, 255, 255, 255]);
        assert_eq!(decoded(&b, "a.psd").unwrap().2, [0, 255, 255, 255]);
        // Indexed: 256 reds, then greens, then blues.
        let mut pal = vec![0u8; 768];
        (pal[3], pal[256 + 3], pal[512 + 3]) = (10, 20, 30);
        let b = psd(1, 1, 1, (8, 2), &pal, 0, &[3]);
        assert_eq!(decoded(&b, "a.psd").unwrap().2, [10, 20, 30, 255]);
        // Bitmap: a set bit is black.
        let b = psd(1, 3, 1, (1, 0), &[], 0, &[0b0100_0000]);
        let gray: Vec<u8> = decoded(&b, "a.psd").unwrap().2.chunks(4).map(|p| p[0]).collect();
        assert_eq!(gray, [255, 0, 255]);
    }

    #[test]
    fn unsupported_and_damaged() {
        assert!(decoded(&psd(3, 1, 1, (32, 3), &[], 0, &[0; 12]), "a.psd").is_none());
        assert!(decoded(&psd(3, 1, 1, (8, 3), &[], 2, &[0; 12]), "a.psd").is_none());
        assert!(decoded(&psd(3, 1, 1, (8, 9), &[], 0, &[0; 12]), "a.psd").is_none());
        assert!(decoded(&psd(3, 16000, 16000, (8, 3), &[], 0, &[0; 100]), "a.psd").is_none());
        let mut data = Vec::new();
        for _ in 0..4 * 6 {
            data.extend(3u16.to_be_bytes());
        }
        for _ in 0..4 * 6 {
            data.extend([0xFB, 9, 0x00]);
        }
        survives_damage(&psd(4, 6, 6, (8, 3), &[], 1, &data), "a.psd");
    }
}
