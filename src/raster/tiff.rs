//! Baseline TIFF's first image: strips or tiles; uncompressed, LZW, `PackBits` or Deflate; gray, palette, RGB or
//! CMYK, with alpha.

use std::io::Read;

use super::lzw::{self, Codes};
use super::{FREE, Raster, be16, be32, get, le16, le32, packbits, pixels, scale};
use crate::num::usize_of;

/// The first image's tags this reads, by number.
#[derive(Default)]
struct Tags {
    values: Vec<(u16, Vec<u64>)>,
}

impl Tags {
    fn all(&self, tag: u16) -> Option<&[u64]> {
        self.values.iter().find(|(t, _)| *t == tag).map(|(_, v)| &v[..])
    }

    fn one(&self, tag: u16) -> Option<u64> {
        self.all(tag)?.first().copied()
    }
}

/// Tags read: sizes, bits, compression, photometric, strips, samples, rows per strip, strip sizes, planar
/// layout, predictor, color map, tiles, extra samples and sample format.
const WANTED: [u16; 18] = [
    256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 317, 320, 322, 323, 324, 325, 338, 339,
];

pub(super) fn decode(b: &[u8]) -> Option<Raster> {
    let le = b[0] == b'I';
    let u16_at = |at| if le { le16(b, at) } else { be16(b, at) };
    let u32_at = |at| if le { le32(b, at) } else { be32(b, at) };
    let ifd = usize_of(u64::from(u32_at(4)?));
    let count = usize::from(u16_at(ifd)?);
    let mut tags = Tags::default();
    for e in 0..count {
        let at = ifd + 2 + e * 12;
        let Some(tag) = u16_at(at) else { break };
        if !WANTED.contains(&tag) {
            continue;
        }
        let (Some(kind), Some(n)) = (u16_at(at + 2), u32_at(at + 4)) else {
            break;
        };
        let size = match kind {
            1 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 => 4,
            _ => continue,
        };
        let n = usize_of(u64::from(n));
        let Some(len) = n.checked_mul(size) else { continue };
        let from = if len <= 4 {
            at + 8
        } else {
            usize_of(u64::from(u32_at(at + 8)?))
        };
        // Only values the file holds are read.
        let Some(bytes) = get(b, from, len) else { continue };
        let values = bytes
            .chunks_exact(size)
            .map(|v| match size {
                1 => u64::from(v[0]),
                2 => u64::from(if le {
                    u16::from_le_bytes([v[0], v[1]])
                } else {
                    u16::from_be_bytes([v[0], v[1]])
                }),
                _ => u64::from(if le { le32(v, 0) } else { be32(v, 0) }.unwrap_or(0)),
            })
            .collect();
        tags.values.push((tag, values));
    }

    let small = |tag, default: u64| tags.one(tag).unwrap_or(default);
    let w = u32::try_from(tags.one(256)?).ok()?;
    let h = u32::try_from(tags.one(257)?).ok()?;
    let n = pixels(w, h)?;
    let spp = usize_of(small(277, 1));
    let bits = tags.all(258).unwrap_or(&[1]);
    let bits = *bits.first()?;
    if tags.all(258).is_some_and(|all| all.iter().any(|&b| b != bits))
        || tags.all(339).is_some_and(|f| f.iter().any(|&f| f != 1))
    {
        return None;
    }
    let photometric = tags.one(262).unwrap_or(if spp >= 3 { 2 } else { 1 });
    let base = match photometric {
        0 | 1 | 3 => 1,
        2 => 3,
        5 => 4,
        _ => return None,
    };
    let ok_bits = match bits {
        1 | 2 | 4 => base == 1 && spp == 1,
        8 => true,
        16 => photometric != 3,
        _ => false,
    };
    if !ok_bits || spp < base || spp > 16 {
        return None;
    }
    let bits = u32::try_from(bits).ok()?;
    let compression = small(259, 1);
    let planar = small(284, 1) == 2 && spp > 1;
    let predictor = small(317, 1);
    if predictor > 2 || (predictor == 2 && bits < 8) {
        return None;
    }

    // The chunks: tiles, or strips as wide as the image. Each is a whole number of rows (`cw` pixels wide) of one
    // plane, or of all samples together.
    let (wu, hu) = (usize_of(u64::from(w)), usize_of(u64::from(h)));
    let (cw, ch, offsets, sizes) = match (tags.one(322), tags.one(323)) {
        (Some(tw), Some(th)) => (usize_of(tw), usize_of(th), tags.all(324)?, tags.all(325)),
        _ => (
            wu,
            usize_of(small(278, u64::from(h))).min(hu),
            tags.all(273)?,
            tags.all(279),
        ),
    };
    if cw == 0 || ch == 0 || cw > wu.max(1 << 14) || ch > hu.max(1 << 14) {
        return None;
    }
    let (across, down) = (wu.div_ceil(cw), hu.div_ceil(ch));
    let planes = if planar { spp } else { 1 };
    let per_pixel = if planar { 1 } else { spp };
    let row_bytes = (cw * per_pixel * usize_of(u64::from(bits))).div_ceil(8);
    let chunk_bytes = row_bytes.checked_mul(ch)?;
    let chunks = across * down * planes;
    let need = chunk_bytes.checked_mul(chunks)?;

    // Decode every chunk, then check enough of the image came out before padding.
    let mut decoded = Vec::with_capacity(chunks.min(offsets.len()));
    let mut real = 0usize;
    for (i, &off) in offsets.iter().enumerate().take(chunks) {
        let off = usize_of(off);
        let len = sizes
            .and_then(|s| s.get(i))
            .map_or(b.len().saturating_sub(off), |&s| usize_of(s));
        let src = b.get(off..).unwrap_or_default();
        let src = &src[..src.len().min(len)];
        let mut out = Vec::new();
        match compression {
            1 => out.extend_from_slice(&src[..src.len().min(chunk_bytes)]),
            5 => {
                // Old-style LZW codes from each byte's low bits and widens late.
                let old = src.len() > 1 && src[0] == 0 && src[1] & 1 != 0;
                let codes = Codes {
                    min: 8,
                    msb: !old,
                    early: !old,
                };
                lzw::decode(src, codes, &mut out, chunk_bytes);
            }
            32773 => packbits(src, &mut out, chunk_bytes),
            8 | 32946 => {
                let mut z = flate2::read::ZlibDecoder::new(src).take(u64::try_from(chunk_bytes).ok()?);
                let _ = z.read_to_end(&mut out);
            }
            _ => return None,
        }
        real += out.len();
        decoded.push(out);
    }
    if real < need / 2 && need > FREE {
        return None;
    }
    let sixteen = bits == 16;
    let mut samples = vec![0u8; n * spp];
    for (i, mut chunk) in decoded.into_iter().enumerate() {
        chunk.resize(chunk_bytes, 0);
        if predictor == 2 {
            undo_predictor(&mut chunk, row_bytes, per_pixel, sixteen, le);
        }
        let plane = i / (across * down);
        let (cy, cx) = ((i % (across * down)) / across, i % across);
        for r in 0..ch {
            let y = cy * ch + r;
            if y >= hu {
                break;
            }
            let row = &chunk[r * row_bytes..(r + 1) * row_bytes];
            for c in 0..cw {
                let x = cx * cw + c;
                if x >= wu {
                    break;
                }
                for s in 0..per_pixel {
                    let k = c * per_pixel + s;
                    let v = match bits {
                        16 => row[k * 2 + usize::from(le)],
                        8 => row[k],
                        _ => {
                            let bit = k * usize_of(u64::from(bits));
                            let shift = 8 - usize_of(u64::from(bits)) - bit % 8;
                            (row[bit / 8] >> shift) & (u8::MAX >> (8 - bits))
                        }
                    };
                    samples[(y * wu + x) * spp + plane + s] = v;
                }
            }
        }
    }

    // Samples to pixels.
    // The first extra sample is alpha when it says so: associated (premultiplied) or not.
    let extra = tags.one(338);
    let alpha = spp > base && matches!(extra, Some(1 | 2));
    let associated = extra == Some(1);
    let channels = if base == 1 && photometric != 3 { 1 } else { 3 } + usize::from(alpha);
    let low = if sixteen { 8 } else { bits };
    let colormap = tags.all(320).unwrap_or_default();
    let entries = 1usize << bits.min(8);
    let mut px = Vec::with_capacity(n * channels);
    for s in samples.chunks_exact(spp) {
        let mut c = [0u8; 4];
        let color = &mut c[..channels - usize::from(alpha)];
        match photometric {
            0 => color[0] = 255 - scale(u32::from(s[0]), low),
            1 => color[0] = scale(u32::from(s[0]), low),
            2 => color.copy_from_slice(&s[..3]),
            3 => {
                let i = usize::from(s[0]);
                for (k, v) in color.iter_mut().enumerate() {
                    let at = k * entries + i;
                    *v = colormap.get(at).map_or(0, |&m| u8::try_from(m >> 8).unwrap_or(u8::MAX));
                }
            }
            _ => {
                let k = u32::from(255 - s[3]);
                for (v, &ink) in color.iter_mut().zip(s) {
                    *v = u8::try_from(u32::from(255 - ink) * k / 255).unwrap_or(0);
                }
            }
        }
        if alpha {
            let a = s[base];
            if associated && a < 255 {
                for v in color.iter_mut() {
                    *v = if a == 0 {
                        0
                    } else {
                        u8::try_from((u32::from(*v) * 255 / u32::from(a)).min(255)).unwrap_or(255)
                    };
                }
            }
            c[channels - 1] = a;
        }
        px.extend_from_slice(&c[..channels]);
    }
    Some(Raster {
        w,
        h,
        channels,
        data: px,
    })
}

/// Horizontal differencing undone: each sample is the difference from the one `per_pixel` samples before it.
fn undo_predictor(chunk: &mut [u8], row_bytes: usize, per_pixel: usize, sixteen: bool, le: bool) {
    for row in chunk.chunks_exact_mut(row_bytes) {
        if sixteen {
            let read = |r: &[u8], i: usize| {
                if le {
                    u16::from_le_bytes([r[i * 2], r[i * 2 + 1]])
                } else {
                    u16::from_be_bytes([r[i * 2], r[i * 2 + 1]])
                }
            };
            for i in per_pixel..row.len() / 2 {
                let v = read(row, i).wrapping_add(read(row, i - per_pixel));
                let bytes = if le { v.to_le_bytes() } else { v.to_be_bytes() };
                row[i * 2..i * 2 + 2].copy_from_slice(&bytes);
            }
        } else {
            for i in per_pixel..row.len() {
                row[i] = row[i].wrapping_add(row[i - per_pixel]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::lzw::tests::encode;
    use super::super::tests::{decoded, survives_damage};
    use super::*;

    /// A TIFF: `entries` (tag, type, values) in an IFD, values too long for an entry stored after it. Little-endian
    /// unless `big`. Strip data goes at the offsets `data` places, given as tag 273's values of 0, 1, ...
    fn tiff(big: bool, entries: &[(u16, u16, Vec<u32>)], strips: &[Vec<u8>]) -> Vec<u8> {
        let u16b = |v: u16| if big { v.to_be_bytes() } else { v.to_le_bytes() };
        let u32b = |v: u32| if big { v.to_be_bytes() } else { v.to_le_bytes() };
        let mut b = if big { b"MM\0*".to_vec() } else { b"II*\0".to_vec() };
        b.extend(u32b(8));
        let mut entries = entries.to_vec();
        entries.push((273, 4, vec![0; strips.len()]));
        entries.push((279, 4, strips.iter().map(|s| u32::try_from(s.len()).unwrap()).collect()));
        entries.sort_by_key(|e| e.0);
        let ifd_end = 8 + 2 + entries.len() * 12 + 4;
        let mut extra = Vec::new();
        let mut strip_at = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for s in strips {
            strip_at.push(u32::try_from(ifd_end + 4096 + data.len()).unwrap());
            data.extend(s);
        }
        b.extend(u16b(u16::try_from(entries.len()).unwrap()));
        for (tag, kind, values) in &entries {
            let values = if *tag == 273 { &strip_at } else { values };
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&v| {
                    if *kind == 3 {
                        u16b(u16::try_from(v).unwrap()).to_vec()
                    } else {
                        u32b(v).to_vec()
                    }
                })
                .collect();
            b.extend(u16b(*tag));
            b.extend(u16b(*kind));
            b.extend(u32b(u32::try_from(values.len()).unwrap()));
            if bytes.len() <= 4 {
                let mut v = bytes.clone();
                v.resize(4, 0);
                b.extend(v);
            } else {
                b.extend(u32b(u32::try_from(ifd_end + extra.len()).unwrap()));
                extra.extend(bytes);
            }
        }
        b.extend([0; 4]);
        assert!(extra.len() <= 4096);
        extra.resize(4096, 0);
        b.extend(extra);
        b.extend(data);
        b
    }

    fn rgb_tags(w: u32, h: u32, compression: u32) -> Vec<(u16, u16, Vec<u32>)> {
        vec![
            (256, 4, vec![w]),
            (257, 4, vec![h]),
            (258, 3, vec![8, 8, 8]),
            (259, 3, vec![compression]),
            (262, 3, vec![2]),
            (277, 3, vec![3]),
        ]
    }

    #[test]
    fn rgb_strips_every_compression() {
        let px: Vec<u8> = (0..4 * 3 * 3).map(|i| u8::try_from(i * 7).unwrap()).collect();
        let lzw = encode(
            &px,
            Codes {
                min: 8,
                msb: true,
                early: true,
            },
        );
        let mut zlib = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut zlib, &px).unwrap();
        let zlib = zlib.finish().unwrap();
        let pack: Vec<u8> = std::iter::once(35).chain(px.iter().copied()).collect();
        for (compression, data) in [(1, px.clone()), (5, lzw), (8, zlib), (32773, pack)] {
            for big in [false, true] {
                let b = tiff(big, &rgb_tags(4, 3, compression), std::slice::from_ref(&data));
                let (w, h, out) = decoded(&b, "a.tif").unwrap();
                assert_eq!((w, h), (4, 3));
                let rgb: Vec<u8> = out.chunks(4).flat_map(|p| p[..3].to_vec()).collect();
                assert_eq!(rgb, px, "{compression} {big}");
            }
        }
        // Two strips of rows, the second cut short and padded.
        let mut tags = rgb_tags(4, 3, 1);
        tags.push((278, 3, vec![2]));
        let b = tiff(false, &tags, &[px[..24].to_vec(), px[24..30].to_vec()]);
        let out = decoded(&b, "a.tif").unwrap().2;
        assert_eq!(out[32..40], [px[24], px[25], px[26], 255, px[27], px[28], px[29], 255]);
        assert_eq!(out[40..], [0, 0, 0, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn gray_palette_and_bits() {
        // 1-bit, white is zero.
        let tags = vec![(256, 3, vec![10]), (257, 3, vec![1]), (262, 3, vec![0])];
        let out = decoded(&tiff(false, &tags, &[vec![0b1000_0000, 0b0100_0000]]), "a.tif")
            .unwrap()
            .2;
        let gray: Vec<u8> = out.chunks(4).map(|p| p[0]).collect();
        assert_eq!(gray, [0, 255, 255, 255, 255, 255, 255, 255, 255, 0]);
        // 4-bit palette: 16 entries a channel, 16 bits each.
        let mut map = vec![0u32; 48];
        (map[1], map[16 + 2], map[32 + 3]) = (0xFFFF, 0x8000, 0x4000);
        let tags = vec![
            (256, 3, vec![3]),
            (257, 3, vec![1]),
            (258, 3, vec![4]),
            (262, 3, vec![3]),
            (320, 3, map),
        ];
        let out = decoded(&tiff(true, &tags, &[vec![0x12, 0x30]]), "a.tif").unwrap().2;
        assert_eq!(out, [255, 0, 0, 255, 0, 128, 0, 255, 0, 0, 64, 255]);
        // 16-bit gray keeps the high byte, in either order; a predictor undone.
        let tags = vec![
            (256, 3, vec![2]),
            (257, 3, vec![1]),
            (258, 3, vec![16]),
            (262, 3, vec![1]),
            (317, 3, vec![2]),
        ];
        let out = decoded(&tiff(false, &tags, &[vec![0x00, 0x40, 0x00, 0x20]]), "a.tif")
            .unwrap()
            .2;
        assert_eq!([out[0], out[4]], [0x40, 0x60]);
        let out = decoded(&tiff(true, &tags, &[vec![0x40, 0x00, 0x20, 0x00]]), "a.tif")
            .unwrap()
            .2;
        assert_eq!([out[0], out[4]], [0x40, 0x60]);
    }

    #[test]
    fn alpha_cmyk_planes_and_tiles() {
        // Associated alpha is unpremultiplied.
        let mut tags = rgb_tags(1, 1, 1);
        tags[5].2 = vec![4];
        tags[2].2 = vec![8; 4];
        tags.push((338, 3, vec![1]));
        assert_eq!(
            decoded(&tiff(false, &tags, &[vec![50, 100, 0, 128]]), "a.tif")
                .unwrap()
                .2,
            [99, 199, 0, 128]
        );
        // An extra sample of unspecified meaning isn't alpha.
        tags.last_mut().unwrap().2 = vec![0];
        let out = decoded(&tiff(false, &tags, &[vec![50, 100, 0, 128]]), "a.tif")
            .unwrap()
            .2;
        assert_eq!(out, [50, 100, 0, 255]);
        // CMYK.
        let tags = vec![
            (256, 3, vec![1]),
            (257, 3, vec![1]),
            (258, 3, vec![8; 4]),
            (262, 3, vec![5]),
            (277, 3, vec![4]),
        ];
        assert_eq!(
            decoded(&tiff(false, &tags, &[vec![255, 0, 0, 0]]), "a.tif").unwrap().2,
            [0, 255, 255, 255]
        );
        // Planes apart: all red, then all green, then all blue.
        let mut tags = rgb_tags(2, 1, 1);
        tags.push((284, 3, vec![2]));
        let out = decoded(&tiff(false, &tags, &[vec![1, 2], vec![3, 4], vec![5, 6]]), "a.tif")
            .unwrap()
            .2;
        assert_eq!(out, [1, 3, 5, 255, 2, 4, 6, 255]);
        // 3 × 3 in 2 × 2 tiles, which run past the image's edges.
        let tags = vec![
            (256, 3, vec![3]),
            (257, 3, vec![3]),
            (258, 3, vec![8]),
            (262, 3, vec![1]),
            (322, 3, vec![2]),
            (323, 3, vec![2]),
        ];
        let tiles = vec![vec![1, 2, 4, 5], vec![3, 0, 6, 0], vec![7, 8, 0, 0], vec![9, 0, 0, 0]];
        let mut b = tiff(false, &tags, &tiles);
        // The helper wrote strip tags; make them tile tags.
        for i in (10..b.len() - 12).step_by(12).take(9) {
            if b[i..i + 2] == 273u16.to_le_bytes() {
                b[i..i + 2].copy_from_slice(&324u16.to_le_bytes());
            } else if b[i..i + 2] == 279u16.to_le_bytes() {
                b[i..i + 2].copy_from_slice(&325u16.to_le_bytes());
            }
        }
        let out = decoded(&b, "a.tif").unwrap().2;
        let gray: Vec<u8> = out.chunks(4).map(|p| p[0]).collect();
        assert_eq!(gray, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn unsupported_and_damaged() {
        let mut tags = rgb_tags(2, 2, 7);
        assert!(decoded(&tiff(false, &tags, &[vec![0; 12]]), "a.tif").is_none());
        tags[3].2 = vec![1];
        tags.push((339, 3, vec![3, 3, 3]));
        assert!(decoded(&tiff(false, &tags, &[vec![0; 12]]), "a.tif").is_none());
        let huge = tiff(false, &rgb_tags(16000, 16000, 1), &[vec![0; 100]]);
        assert!(decoded(&huge, "a.tif").is_none());
        let px: Vec<u8> = (0..300u32).map(|i| u8::try_from(i * 13 % 251).unwrap()).collect();
        let mut tags = rgb_tags(10, 10, 5);
        tags.push((278, 3, vec![4]));
        let strips: Vec<Vec<u8>> = px
            .chunks(120)
            .map(|s| {
                encode(
                    s,
                    Codes {
                        min: 8,
                        msb: true,
                        early: true,
                    },
                )
            })
            .collect();
        survives_damage(&tiff(true, &tags, &strips), "a.tif");
    }
}
