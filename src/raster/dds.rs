//! DDS textures: the top mipmap level of the first image, block-compressed (BC1 to BC5) or uncompressed.

use super::{Raster, fill, get, le32, masked, pixels};

/// How the pixels are stored.
#[derive(Debug, Clone, Copy)]
enum Format {
    /// 4 × 4 blocks: BC1 to BC5.
    Block(u8),
    /// Pixels of `bits` bits, channels picked out by masks; gray from the red mask.
    Masks { bits: u32, masks: [u32; 4], gray: bool },
}

const RGBA: Format = Format::Masks {
    bits: 32,
    masks: [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000],
    gray: false,
};
const BGRA: Format = Format::Masks {
    bits: 32,
    masks: [0xFF_0000, 0xFF00, 0xFF, 0xFF00_0000],
    gray: false,
};
const BGRX: Format = Format::Masks {
    bits: 32,
    masks: [0xFF_0000, 0xFF00, 0xFF, 0],
    gray: false,
};
const R8: Format = Format::Masks {
    bits: 8,
    masks: [0xFF, 0, 0, 0],
    gray: true,
};

pub(super) fn decode(b: &[u8]) -> Option<Raster> {
    let (h, w) = (le32(b, 12)?, le32(b, 16)?);
    let n = pixels(w, h)?;
    let flags = le32(b, 80)?;
    let mut at = 128;
    let format = if flags & 0x4 != 0 {
        match get(b, 84, 4)? {
            b"DXT1" => Format::Block(1),
            b"DXT2" | b"DXT3" => Format::Block(2),
            b"DXT4" | b"DXT5" => Format::Block(3),
            b"ATI1" | b"BC4U" => Format::Block(4),
            b"ATI2" | b"BC5U" => Format::Block(5),
            b"DX10" => {
                at = 148;
                match le32(b, 128)? {
                    70..=72 => Format::Block(1),
                    73..=75 => Format::Block(2),
                    76..=78 => Format::Block(3),
                    79 | 80 => Format::Block(4),
                    82 | 83 => Format::Block(5),
                    27..=29 => RGBA,
                    87 | 90 | 91 => BGRA,
                    88 | 92 | 93 => BGRX,
                    61 => R8,
                    _ => return None,
                }
            }
            _ => return None,
        }
    } else if flags & (0x40 | 0x2_0000 | 0x2) != 0 {
        // RGB, luminance or alpha alone; alpha pixels with flag 1 (or 2).
        let (bits, gray, has_alpha) = (le32(b, 88)?, flags & 0x40 == 0, flags & 0x3 != 0);
        let alpha = if has_alpha { le32(b, 104)? } else { 0 };
        let mut masks = [le32(b, 92)?, le32(b, 96)?, le32(b, 100)?, alpha];
        // Masks reaching past the pixel's bits are a writer's mistake: the usual layout for its size instead.
        if bits < 32 && masks.iter().any(|m| m >> bits != 0) {
            let a = |m: u32| if has_alpha { m } else { 0 };
            masks = match (gray, bits) {
                (true, _) => [0xFF, 0, 0, a(0xFF00)],
                (false, 16) => [0xF800, 0x07E0, 0x1F, 0],
                (false, _) => [0xFF_0000, 0xFF00, 0xFF, a(0xFF00_0000)],
            };
        }
        Format::Masks { bits, masks, gray }
    } else {
        return None;
    };
    let data = b.get(at..).unwrap_or_default();
    let (wu, hu) = (crate::num::usize_of(u64::from(w)), crate::num::usize_of(u64::from(h)));
    match format {
        Format::Block(kind) => {
            let size = if matches!(kind, 1 | 4) { 8 } else { 16 };
            let (across, down) = (wu.div_ceil(4), hu.div_ceil(4));
            let need = across * down * size;
            let mut raw = data[..data.len().min(need)].to_vec();
            fill(&mut raw, need)?;
            let channels = match kind {
                4 => 1,
                5 => 3,
                _ => 4,
            };
            let mut px = vec![0u8; n * channels];
            for (i, block) in raw.chunks_exact(size).enumerate() {
                let texels = texels(kind, block);
                let (bx, by) = (i % across * 4, i / across * 4);
                for (t, texel) in texels.iter().enumerate() {
                    let (x, y) = (bx + t % 4, by + t / 4);
                    if x < wu && y < hu {
                        let o = (y * wu + x) * channels;
                        px[o..o + channels].copy_from_slice(&texel[..channels]);
                    }
                }
            }
            Some(Raster {
                w,
                h,
                channels,
                data: px,
            })
        }
        Format::Masks { bits, masks, gray } => {
            let bytes = match bits {
                8 | 16 | 24 | 32 => crate::num::usize_of(u64::from(bits / 8)),
                _ => return None,
            };
            let need = n * bytes;
            let mut raw = data[..data.len().min(need)].to_vec();
            fill(&mut raw, need)?;
            let alpha = masks[3] != 0;
            let channels = if gray { 1 } else { 3 } + usize::from(alpha);
            let mut px = Vec::with_capacity(n * channels);
            for p in raw.chunks_exact(bytes) {
                let mut v = [0u8; 4];
                v[..bytes].copy_from_slice(p);
                let v = u32::from_le_bytes(v);
                let colors = if gray { 1 } else { 3 };
                px.extend(masks[..colors].iter().map(|&m| masked(v, m)));
                if alpha {
                    px.push(masked(v, masks[3]));
                }
            }
            Some(Raster {
                w,
                h,
                channels,
                data: px,
            })
        }
    }
}

/// A block's 16 texels, rows top to bottom, as RGBA (BC4's gray, and BC5's red and green, come first).
fn texels(kind: u8, block: &[u8]) -> [[u8; 4]; 16] {
    let mut out = [[0, 0, 0, 255]; 16];
    match kind {
        1 => color(block, false, &mut out),
        2 => {
            color(&block[8..], true, &mut out);
            let bits = u64::from_le_bytes(block[..8].try_into().unwrap_or_default());
            for (i, t) in out.iter_mut().enumerate() {
                t[3] = u8::try_from((bits >> (4 * i)) & 15).unwrap_or(0) * 17;
            }
        }
        3 => {
            color(&block[8..], true, &mut out);
            for (t, a) in out.iter_mut().zip(ramp(block)) {
                t[3] = a;
            }
        }
        4 => {
            for (t, g) in out.iter_mut().zip(ramp(block)) {
                *t = [g, g, g, 255];
            }
        }
        _ => {
            for ((t, x), y) in out.iter_mut().zip(ramp(block)).zip(ramp(&block[8..])) {
                // A normal map's two stored axes; the third rebuilt, pointing out of the surface.
                let (nx, ny) = (2 * i32::from(x) - 255, 2 * i32::from(y) - 255);
                let zz = (255 * 255 - nx * nx - ny * ny).max(0).cast_unsigned();
                let z = u8::try_from(zz.isqrt().midpoint(255)).unwrap_or(u8::MAX);
                *t = [x, y, z, 255];
            }
        }
    }
    out
}

/// A BC1 color block's texels; `four` for the four-color coding BC2 and BC3 always use.
fn color(block: &[u8], four: bool, out: &mut [[u8; 4]; 16]) {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let rgb = |c: u16| {
        let c = u32::from(c);
        [
            super::scale(c >> 11, 5),
            super::scale((c >> 5) & 63, 6),
            super::scale(c & 31, 5),
        ]
        .map(u32::from)
    };
    let (a, b) = (rgb(c0), rgb(c1));
    let mix = |wa: u32, wb: u32| -> [u8; 4] {
        let m = |i: usize| u8::try_from((a[i] * wa + b[i] * wb) / (wa + wb)).unwrap_or(u8::MAX);
        [m(0), m(1), m(2), 255]
    };
    let palette = if four || c0 > c1 {
        [mix(1, 0), mix(0, 1), mix(2, 1), mix(1, 2)]
    } else {
        [mix(1, 0), mix(0, 1), mix(1, 1), [0, 0, 0, 0]]
    };
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for (i, t) in out.iter_mut().enumerate() {
        *t = palette[usize::try_from((bits >> (2 * i)) & 3).unwrap_or(0)];
    }
}

/// A BC4 block's 16 values: two ends and six or eight steps between them.
fn ramp(block: &[u8]) -> [u8; 16] {
    let (a, b) = (u32::from(block[0]), u32::from(block[1]));
    let mut values = [0u8; 8];
    // Direct3D blends the two ends as fractions, so a step is the nearest value, not the one below it.
    let mix = |wa: u32, wb: u32, sum: u32| u8::try_from((2 * (a * wa + b * wb) + sum) / (2 * sum)).unwrap_or(u8::MAX);
    values[0] = mix(1, 0, 1);
    values[1] = mix(0, 1, 1);
    if a > b {
        for k in 1..7 {
            values[usize::try_from(k + 1).unwrap_or(0)] = mix(7 - k, k, 7);
        }
    } else {
        for k in 1..5 {
            values[usize::try_from(k + 1).unwrap_or(0)] = mix(5 - k, k, 5);
        }
        values[7] = 255;
    }
    let mut bits = [0u8; 8];
    bits[..6].copy_from_slice(&block[2..8]);
    let bits = u64::from_le_bytes(bits);
    std::array::from_fn(|i| values[usize::try_from((bits >> (3 * i)) & 7).unwrap_or(0)])
}

#[cfg(test)]
mod tests {
    use super::super::tests::{decoded, survives_damage};

    /// A surface's header: `fourcc` for a compressed one, or masks with `flags`.
    fn dds(w: u32, h: u32, flags: u32, fourcc: [u8; 4], bits: u32, masks: [u32; 4]) -> Vec<u8> {
        let mut b = b"DDS ".to_vec();
        b.extend(124u32.to_le_bytes());
        b.resize(128, 0);
        b[12..16].copy_from_slice(&h.to_le_bytes());
        b[16..20].copy_from_slice(&w.to_le_bytes());
        b[76..80].copy_from_slice(&32u32.to_le_bytes());
        b[80..84].copy_from_slice(&flags.to_le_bytes());
        b[84..88].copy_from_slice(&fourcc);
        b[88..92].copy_from_slice(&bits.to_le_bytes());
        for (k, m) in masks.iter().enumerate() {
            b[92 + k * 4..96 + k * 4].copy_from_slice(&m.to_le_bytes());
        }
        b
    }

    #[test]
    fn bc1_bc3_and_bc5() {
        // BC1: blue and red ends, three-color mode (blue < red as numbers): blue, red, then their mix and clear.
        let mut b = dds(2, 2, 4, *b"DXT1", 0, [0; 4]);
        b.extend(0x001Fu16.to_le_bytes());
        b.extend(0xF800u16.to_le_bytes());
        b.extend([0b0100, 0b1110, 0, 0]);
        let px = decoded(&b, "a.dds").unwrap().2;
        assert_eq!(px, [0, 0, 255, 255, 255, 0, 0, 255, 127, 0, 127, 255, 0, 0, 0, 0]);
        // BC3: an alpha ramp from 255 to 0 over a white block. Six sevenths of 255 is 218.57, five sevenths 182.14.
        let mut b = dds(4, 1, 4, *b"DXT5", 0, [0; 4]);
        b.extend([255, 0, 0b1000_1000, 0b0000_0110, 0, 0, 0, 0]);
        b.extend([0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]);
        let alpha: Vec<u8> = decoded(&b, "a.dds").unwrap().2.chunks(4).map(|p| p[3]).collect();
        assert_eq!(alpha, [255, 0, 219, 182]);
        // BC5 by DX10 format 83: a flat normal's z is all but full.
        let mut b = dds(1, 1, 4, *b"DX10", 0, [0; 4]);
        b.extend(83u32.to_le_bytes());
        b.extend([0; 16]);
        b.extend([128, 128, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 0, 0]);
        assert_eq!(decoded(&b, "a.dds").unwrap().2, [128, 128, 254, 255]);
    }

    #[test]
    fn uncompressed() {
        let mut b = dds(2, 1, 0x41, *b"\0\0\0\0", 32, [0xFF_0000, 0xFF00, 0xFF, 0xFF00_0000]);
        b.extend([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(decoded(&b, "a.dds").unwrap().2, [3, 2, 1, 4, 7, 6, 5, 8]);
        // 16-bit luminance with alpha, 8 bits each.
        let mut b = dds(1, 1, 0x2_0001, *b"\0\0\0\0", 16, [0xFF, 0, 0, 0xFF00]);
        b.extend([90, 30]);
        assert_eq!(decoded(&b, "a.dds").unwrap().2, [90, 90, 90, 30]);
        // Masks outside the pixel's bits: the usual layout.
        let mut b = dds(1, 1, 0x2_0001, *b"    ", 16, [0xFF, 0xFF, 0xFF, 0xFF00_0000]);
        b.extend([90, 30]);
        assert_eq!(decoded(&b, "a.dds").unwrap().2, [90, 90, 90, 30]);
        // 5-6-5.
        let mut b = dds(1, 1, 0x40, *b"\0\0\0\0", 16, [0xF800, 0x07E0, 0x1F, 0]);
        b.extend(0x07E0u16.to_le_bytes());
        assert_eq!(decoded(&b, "a.dds").unwrap().2, [0, 255, 0, 255]);
    }

    #[test]
    fn unsupported_and_damaged() {
        let mut bc7 = dds(4, 4, 4, *b"DX10", 0, [0; 4]);
        bc7.extend(98u32.to_le_bytes());
        bc7.extend([0; 32]);
        assert!(decoded(&bc7, "a.dds").is_none());
        assert!(decoded(&dds(4, 4, 4, *b"DXT9", 0, [0; 4]), "a.dds").is_none());
        assert!(decoded(&dds(16000, 16000, 4, *b"DXT1", 0, [0; 4]), "a.dds").is_none());
        for fourcc in [b"DXT1", b"DXT3", b"DXT5", b"ATI1", b"ATI2"] {
            let mut b = dds(9, 7, 4, *fourcc, 0, [0; 4]);
            b.extend((0..16 * 9).map(|i: u32| u8::try_from(i * 37 % 256).unwrap()));
            survives_damage(&b, "a.dds");
        }
    }
}
