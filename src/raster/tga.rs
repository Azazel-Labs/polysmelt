//! Truevision TGA: color-mapped, true-color and gray images, raw or run-length coded.

use std::path::Path;

use super::{Raster, fill, flip, get, le16, mirror, opaque_if_clear, pixels, scale};

/// The end of a version 2 file.
const FOOTER: &[u8] = b"TRUEVISION-XFILE.\0";

/// Whether `bytes`, named `name`, are TGA: by the extension, or the version 2 footer.
pub(super) fn is_tga(bytes: &[u8], name: &Path) -> bool {
    let ext = name.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    matches!(ext.as_deref(), Some("tga" | "icb" | "vda" | "vst")) || bytes.ends_with(FOOTER)
}

pub(super) fn decode(b: &[u8]) -> Option<Raster> {
    let head = get(b, 0, 18)?;
    let id_len = usize::from(head[0]);
    let (cmap_type, kind) = (head[1], head[2]);
    let (cmap_first, cmap_len, cmap_bits) = (usize::from(le16(head, 3)?), usize::from(le16(head, 5)?), head[7]);
    let (w, h) = (le16(head, 12)?, le16(head, 14)?);
    let (depth, descriptor) = (head[16], head[17]);
    let n = pixels(u32::from(w), u32::from(h))?;
    let alpha_bits = descriptor & 0x0F;
    let rle = kind & 8 != 0;
    // Bytes per stored pixel, and whether that's a color-map index.
    let indexed = match (kind & !8, depth) {
        (1, 8 | 16) if cmap_type == 1 => true,
        (2, 15 | 16 | 24 | 32) | (3, 8 | 16) => false,
        _ => return None,
    };
    let bpp = usize::from(depth.div_ceil(8));
    // A color map comes after the image ID, whether the pixels use it or not.
    let cmap_at = 18 + id_len;
    let entry = usize::from(cmap_bits.div_ceil(8));
    let cmap_size = if cmap_type == 1 { cmap_len * entry } else { 0 };
    let palette: Vec<[u8; 4]> = if indexed {
        if !matches!(cmap_bits, 15 | 16 | 24 | 32) {
            return None;
        }
        // Entries past the file's end are left out.
        let have = b.get(cmap_at..).unwrap_or_default();
        have.chunks_exact(entry)
            .take(cmap_len)
            .map(|p| color(p, cmap_bits, alpha_bits))
            .collect()
    } else {
        Vec::new()
    };
    let data = b.get(cmap_at + cmap_size..).unwrap_or_default();
    let need = n * bpp;
    let mut raw = if rle {
        unpack(data, bpp, need)
    } else {
        data[..data.len().min(need)].to_vec()
    };
    fill(&mut raw, need)?;

    let (channels, mut px) = if indexed {
        let mut px = Vec::with_capacity(n * 4);
        for p in raw.chunks_exact(bpp) {
            let i = if bpp == 2 {
                usize::from(u16::from_le_bytes([p[0], p[1]]))
            } else {
                usize::from(p[0])
            };
            let c = i
                .checked_sub(cmap_first)
                .and_then(|i| palette.get(i))
                .copied()
                .unwrap_or([0, 0, 0, 255]);
            px.extend_from_slice(&c);
        }
        (4, px)
    } else if kind & !8 == 3 {
        // Gray, or gray and alpha.
        (bpp, raw)
    } else if depth == 24 {
        (3, raw.chunks_exact(3).flat_map(|p| [p[2], p[1], p[0]]).collect())
    } else {
        (
            4,
            raw.chunks_exact(bpp)
                .flat_map(|p| color(p, depth, alpha_bits))
                .collect(),
        )
    };
    if channels.is_multiple_of(2) {
        opaque_if_clear(&mut px, channels);
    }
    let row = usize::from(w) * channels;
    if descriptor & 0x20 == 0 {
        flip(&mut px, row);
    }
    if descriptor & 0x10 != 0 {
        mirror(&mut px, row, channels);
    }
    Some(Raster {
        w: u32::from(w),
        h: u32::from(h),
        channels,
        data: px,
    })
}

/// A stored BGR(A) color of `bits` bits as RGBA; a 16-bit color's top bit is alpha only when the image says it has
/// alpha.
fn color(p: &[u8], bits: u8, alpha_bits: u8) -> [u8; 4] {
    match bits {
        15 | 16 => {
            let v = u32::from(u16::from_le_bytes([p[0], p[1]]));
            let a = if bits == 16 && alpha_bits > 0 && v & 0x8000 == 0 {
                0
            } else {
                255
            };
            [scale((v >> 10) & 31, 5), scale((v >> 5) & 31, 5), scale(v & 31, 5), a]
        }
        24 => [p[2], p[1], p[0], 255],
        _ => [p[2], p[1], p[0], p[3]],
    }
}

/// Run-length packets decoded toward `need` bytes of `bpp`-byte pixels. Packets may run across rows.
fn unpack(data: &[u8], bpp: usize, need: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while out.len() < need {
        let Some(&head) = data.get(i) else { break };
        i += 1;
        let count = usize::from(head & 0x7F) + 1;
        if head & 0x80 != 0 {
            let Some(p) = get(data, i, bpp) else { break };
            i += bpp;
            for _ in 0..count.min((need - out.len()).div_ceil(bpp)) {
                out.extend_from_slice(p);
            }
        } else {
            let end = data.len().min(i + count * bpp);
            out.extend_from_slice(&data[i..end]);
            i = end;
        }
    }
    out.truncate(need);
    out
}

#[cfg(test)]
mod tests {
    use super::super::tests::{decoded, survives_damage};
    use super::*;

    fn header(kind: u8, w: u16, h: u16, depth: u8, descriptor: u8, cmap: Option<(u16, u16, u8)>) -> Vec<u8> {
        let mut b = vec![0u8; 18];
        b[2] = kind;
        if let Some((first, len, bits)) = cmap {
            b[1] = 1;
            b[3..5].copy_from_slice(&first.to_le_bytes());
            b[5..7].copy_from_slice(&len.to_le_bytes());
            b[7] = bits;
        }
        b[12..14].copy_from_slice(&w.to_le_bytes());
        b[14..16].copy_from_slice(&h.to_le_bytes());
        b[16] = depth;
        b[17] = descriptor;
        b
    }

    #[test]
    fn true_color_bottom_up() {
        // 2 × 2, bottom row first: blue, green; then red, white.
        let mut b = header(2, 2, 2, 24, 0, None);
        b.extend([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
        let (w, h, px) = decoded(&b, "a.tga").unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(px, [255, 0, 0, 255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 255, 0, 255]);
        // Without the extension it isn't taken for TGA, unless the footer says so.
        assert!(decoded(&b, "a.bin").is_none());
        b.extend([0; 8]);
        b.extend(FOOTER);
        assert_eq!(decoded(&b, "a.bin").unwrap().2[..4], [255, 0, 0, 255]);
    }

    #[test]
    fn alpha_rle_and_right_to_left() {
        // 32-bit, top-down, right to left; one run of 3 and a literal.
        let mut b = header(10, 2, 2, 32, 0x38, None);
        b.extend([0x82, 10, 20, 30, 128, 0x00, 1, 2, 3, 4]);
        let (_, _, px) = decoded(&b, "a.TGA").unwrap();
        assert_eq!(px[..8], [30, 20, 10, 128, 30, 20, 10, 128]);
        assert_eq!(px[8..], [3, 2, 1, 4, 30, 20, 10, 128]);
        // Alpha that's zero everywhere is a writer that didn't fill it: opaque.
        let mut b = header(2, 1, 1, 32, 0x20, None);
        b.extend([1, 2, 3, 0]);
        assert_eq!(decoded(&b, "a.tga").unwrap().2, [3, 2, 1, 255]);
    }

    #[test]
    fn sixteen_bit_gray_and_color_mapped() {
        // 16-bit: alpha only with attribute bits.
        let mut b = header(2, 2, 1, 16, 0x21, None);
        b.extend((0x7C00u16 | 0x8000).to_le_bytes());
        b.extend(0x001Fu16.to_le_bytes());
        assert_eq!(decoded(&b, "a.tga").unwrap().2, [255, 0, 0, 255, 0, 0, 255, 0]);
        b[17] = 0x20;
        assert_eq!(decoded(&b, "a.tga").unwrap().2[4..], [0, 0, 255, 255]);
        // Gray and alpha.
        let mut b = header(3, 1, 1, 16, 0x28, None);
        b.extend([90, 200]);
        assert_eq!(decoded(&b, "a.tga").unwrap().2, [90, 90, 90, 200]);
        // A color map starting at entry 4, with an ID before it; an index outside it is black.
        let mut b = header(1, 3, 1, 8, 0x20, Some((4, 2, 24)));
        b[0] = 3;
        b.extend(b"abc");
        b.extend([0, 0, 200, 0, 100, 0]);
        b.extend([4, 5, 9]);
        assert_eq!(
            decoded(&b, "a.tga").unwrap().2,
            [200, 0, 0, 255, 0, 100, 0, 255, 0, 0, 0, 255]
        );
    }

    #[test]
    fn short_and_damaged_files() {
        // A huge image over a few bytes makes nothing; a small one pads.
        let mut b = header(2, 16000, 16000, 24, 0, None);
        b.extend([1; 64]);
        assert!(decoded(&b, "a.tga").is_none());
        let mut b = header(2, 4, 4, 24, 0x20, None);
        b.extend([7; 3]);
        assert_eq!(decoded(&b, "a.tga").unwrap().2[..8], [7, 7, 7, 255, 0, 0, 0, 255]);
        for bad in [
            header(2, 0, 4, 24, 0, None),
            header(2, 4, 4, 12, 0, None),
            header(1, 4, 4, 8, 0, None),
        ] {
            assert!(decoded(&bad, "a.tga").is_none());
        }
        let mut b = header(9, 8, 8, 16, 0, Some((0, 3, 32)));
        b.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        b.extend([0x83, 1, 0, 2, 0, 0x7F, 2, 0, 0x01]);
        survives_damage(&b, "a.tga");
    }
}
