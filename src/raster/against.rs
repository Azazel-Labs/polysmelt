//! The decoders against independent ones. Files built here, one for each variant a decoder claims, are decoded by
//! this module and by another crate, and the pixels must agree: the `image` crate for TGA, BMP, GIF and TIFF, `psd`
//! for Photoshop documents, and `image_dds` for DDS.
//!
//! Where a format leaves a choice open and the two decoders make it differently, the case carries a tolerance, or
//! compares only the channels both make the same way, and says why. Variants the other decoder doesn't read at all
//! are listed in each test and stay with the decoder's own tests.
//!
//! Built only when all six decoders are, as they are by default.

#![allow(
    clippy::cast_possible_truncation,
    reason = "the builders write small numbers into the widths their fields have"
)]

use super::tests::decoded;
use crate::test_rng::Pcg32;

/// Width, height and RGBA pixels.
type Pixels = (u32, u32, Vec<u8>);

/// One file to decode both ways.
struct Case {
    name: String,
    bytes: Vec<u8>,
    /// How far apart a channel's two values may be.
    tolerance: u8,
    /// How many of a pixel's four channels are compared, from red.
    channels: usize,
}

fn case(name: &str, bytes: Vec<u8>) -> Case {
    within(name, 0, bytes)
}

fn within(name: &str, tolerance: u8, bytes: Vec<u8>) -> Case {
    Case {
        name: name.into(),
        bytes,
        tolerance,
        channels: 4,
    }
}

/// Decodes every case as `file` and with `reference`, and fails with every case that differs.
fn agree(file: &str, cases: &[Case], reference: impl Fn(&[u8]) -> Result<Pixels, String>) {
    let mut wrong = Vec::new();
    for c in cases {
        let ours = decoded(&c.bytes, file);
        // A reference that panics on a file says no more than one that refuses it.
        let theirs = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reference(&c.bytes)))
            .unwrap_or_else(|_| Err("it panicked".into()));
        match (ours, theirs) {
            (None, _) => wrong.push(format!("{}: not decoded", c.name)),
            (_, Err(e)) => wrong.push(format!("{}: the reference can't read it: {e}", c.name)),
            (Some(a), Ok(b)) => {
                if (a.0, a.1) != (b.0, b.1) {
                    wrong.push(format!("{}: {}x{}, the reference {}x{}", c.name, a.0, a.1, b.0, b.1));
                    continue;
                }
                // A fully transparent pixel's color is nobody's to see.
                let differs = |(p, q): (&[u8], &[u8])| {
                    (p[3] != 0 || q[3] != 0) && (0..c.channels).any(|k| p[k].abs_diff(q[k]) > c.tolerance)
                };
                if let Some(i) = a.2.chunks_exact(4).zip(b.2.chunks_exact(4)).position(differs) {
                    let (p, q) = (&a.2[i * 4..i * 4 + 4], &b.2[i * 4..i * 4 + 4]);
                    let count =
                        a.2.chunks_exact(4)
                            .zip(b.2.chunks_exact(4))
                            .filter(|&x| differs(x))
                            .count();
                    wrong.push(format!(
                        "{}: {count} pixels differ, first pixel {i}: {p:?}, the reference {q:?}",
                        c.name
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// The `image` crate's decoder for `format`.
fn image_crate(format: image::ImageFormat) -> impl Fn(&[u8]) -> Result<Pixels, String> {
    move |bytes| {
        let img = image::load_from_memory_with_format(bytes, format).map_err(|e| e.to_string())?;
        Ok((img.width(), img.height(), img.to_rgba8().into_raw()))
    }
}

/// `n` bytes of noise.
fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Pcg32::new(seed);
    (0..n).map(|_| rng.next_u32().to_le_bytes()[0]).collect()
}

/// `n` pixels of `size` bytes of noise where about half repeat the pixel before them, so run-length coders find
/// runs to code.
fn runs(seed: u64, n: usize, size: usize) -> Vec<u8> {
    let mut rng = Pcg32::new(seed);
    let mut out: Vec<u8> = Vec::with_capacity(n * size);
    for i in 0..n {
        if i > 0 && rng.below(2) == 0 {
            out.extend_from_within((i - 1) * size..i * size);
        } else {
            out.extend((0..size).map(|_| rng.next_u32().to_le_bytes()[0]));
        }
    }
    out
}

/// `n` numbers below `below`.
fn indices(seed: u64, n: usize, below: u32) -> Vec<u8> {
    let mut rng = Pcg32::new(seed);
    (0..n).map(|_| rng.below(below) as u8).collect()
}

/// Indices of `bits` bits each packed into bytes, the first in the high bits; a row ends on a byte.
fn pack(values: &[u8], bits: usize) -> Vec<u8> {
    let mut out = vec![0u8; (values.len() * bits).div_ceil(8)];
    for (i, &v) in values.iter().enumerate() {
        let bit = i * bits;
        out[bit / 8] |= v << (8 - bits - bit % 8);
    }
    out
}

/// `PackBits`-coded `row`: runs of three or more as runs, the rest as literals.
fn packbits(row: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < row.len() {
        let run = row[i..].iter().take(128).take_while(|&&b| b == row[i]).count();
        if run >= 3 {
            out.extend([(257 - run) as u8, row[i]]);
            i += run;
            continue;
        }
        // Literals up to the next run of three.
        let mut end = i + 1;
        while end < row.len()
            && end - i < 128
            && !(end + 2 < row.len() && row[end] == row[end + 1] && row[end] == row[end + 2])
        {
            end += 1;
        }
        out.push((end - i - 1) as u8);
        out.extend_from_slice(&row[i..end]);
        i = end;
    }
    out
}

mod tga {
    use super::*;

    /// A TGA file: `kind` 1 to 3 or 9 to 11, a color map of `bits`-bit entries starting at an index, and the
    /// pixel data as stored.
    fn tga(
        kind: u8,
        (w, h): (u16, u16),
        depth: u8,
        descriptor: u8,
        map: Option<(u16, u8, &[u8])>,
        data: &[u8],
    ) -> Vec<u8> {
        let mut b = vec![0u8; 18];
        b[2] = kind;
        if let Some((first, bits, entries)) = map {
            let count = (entries.len() / usize::from(bits.div_ceil(8))) as u16;
            b[1] = 1;
            b[3..5].copy_from_slice(&first.to_le_bytes());
            b[5..7].copy_from_slice(&count.to_le_bytes());
            b[7] = bits;
        }
        b[12..14].copy_from_slice(&w.to_le_bytes());
        b[14..16].copy_from_slice(&h.to_le_bytes());
        b[16] = depth;
        b[17] = descriptor;
        if let Some((_, _, entries)) = map {
            b.extend_from_slice(entries);
        }
        b.extend_from_slice(data);
        b
    }

    /// Run-length packets for rows of `w` pixels of `size` bytes: a run packet where a pixel repeats, raw packets
    /// between. No packet crosses a row, as the specification asks.
    fn rle(pixels: &[u8], size: usize, w: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for row in pixels.chunks_exact(w * size) {
            let px: Vec<&[u8]> = row.chunks_exact(size).collect();
            let mut i = 0;
            while i < px.len() {
                let run = px[i..].iter().take(128).take_while(|p| **p == px[i]).count();
                if run >= 2 {
                    out.push(0x80 | (run - 1) as u8);
                    out.extend_from_slice(px[i]);
                    i += run;
                    continue;
                }
                let mut end = i + 1;
                while end < px.len() && end - i < 128 && !(end + 1 < px.len() && px[end] == px[end + 1]) {
                    end += 1;
                }
                out.push((end - i - 1) as u8);
                out.extend(px[i..end].iter().flat_map(|p| p.iter().copied()));
                i = end;
            }
        }
        out
    }

    /// Every image kind, depth, color map entry size and orientation, raw and run-length coded.
    ///
    /// Not compared: a 16-bit pixel's top bit as alpha when the header gives one attribute bit, which `image`
    /// never reads as alpha.
    #[test]
    fn tga_agrees_with_the_image_crate() {
        let size = (7u16, 5u16);
        let n = 35;
        let mut cases = Vec::new();
        // True color, in each of the four orientations.
        for (name, descriptor) in [
            ("bottom up", 0x00),
            ("top down", 0x20),
            ("bottom up, right to left", 0x10),
            ("top down, right to left", 0x30),
        ] {
            let data = noise(1, n * 3);
            cases.push(case(
                &format!("24-bit {name}"),
                tga(2, size, 24, descriptor, None, &data),
            ));
            let data = noise(2, n * 4);
            cases.push(case(
                &format!("32-bit {name}"),
                tga(2, size, 32, descriptor | 8, None, &data),
            ));
        }
        let data = runs(3, n, 3);
        cases.push(case("24-bit RLE", tga(10, size, 24, 0, None, &rle(&data, 3, 7))));
        let data = runs(4, n, 4);
        cases.push(case("32-bit RLE", tga(10, size, 32, 0x28, None, &rle(&data, 4, 7))));
        // Five bits a channel, the top bit unused.
        let data = noise(5, n * 2);
        cases.push(case("16-bit", tga(2, size, 16, 0x20, None, &data)));
        cases.push(case("15-bit", tga(2, size, 15, 0x20, None, &data)));
        let data = runs(6, n, 2);
        cases.push(case("16-bit RLE", tga(10, size, 16, 0, None, &rle(&data, 2, 7))));
        // Gray, and gray with alpha.
        let data = noise(7, n);
        cases.push(case("gray", tga(3, size, 8, 0, None, &data)));
        let data = runs(8, n, 1);
        cases.push(case("gray RLE", tga(11, size, 8, 0x20, None, &rle(&data, 1, 7))));
        let data = noise(9, n * 2);
        cases.push(case("gray and alpha", tga(3, size, 16, 0x28, None, &data)));
        let data = runs(10, n, 2);
        cases.push(case(
            "gray and alpha RLE",
            tga(11, size, 16, 0x08, None, &rle(&data, 2, 7)),
        ));
        // Color maps of each entry size, indexed by one byte or two.
        let index = indices(11, n, 6);
        for (bits, alpha) in [(24u8, 0u8), (32, 8), (16, 0), (15, 0)] {
            let map = noise(12 + u64::from(bits), 6 * usize::from(bits.div_ceil(8)));
            let file = tga(1, size, 8, 0x20 | alpha, Some((0, bits, &map)), &index);
            cases.push(case(&format!("color map of {bits}-bit entries"), file));
        }
        let map = noise(20, 6 * 3);
        let coded = rle(&runs(21, n, 1).iter().map(|b| b % 6).collect::<Vec<u8>>(), 1, 7);
        cases.push(case("color map, RLE", tga(9, size, 8, 0, Some((0, 24, &map)), &coded)));
        let wide: Vec<u8> = index.iter().flat_map(|&i| [i, 0]).collect();
        cases.push(case(
            "color map, 16-bit indices",
            tga(1, size, 16, 0, Some((0, 24, &map)), &wide),
        ));
        agree("a.tga", &cases, image_crate(image::ImageFormat::Tga));
    }
}

mod bmp {
    use super::*;

    /// A bitmap's parts.
    struct Bmp<'a> {
        /// The info header's size: 12 (OS/2), 40, or 108 (version 4).
        header: u32,
        w: u32,
        /// Negative for rows stored top to bottom.
        h: i32,
        bpp: u16,
        compression: u32,
        /// Red, green, blue and alpha masks, for compression 3.
        masks: Option<[u32; 4]>,
        palette: &'a [[u8; 3]],
        /// The pixel data as stored.
        data: &'a [u8],
    }

    impl Default for Bmp<'_> {
        fn default() -> Self {
            Self {
                header: 40,
                w: 5,
                h: 3,
                bpp: 24,
                compression: 0,
                masks: None,
                palette: &[],
                data: &[],
            }
        }
    }

    fn bmp(f: &Bmp) -> Vec<u8> {
        let mut info = Vec::new();
        info.extend(f.header.to_le_bytes());
        if f.header == 12 {
            info.extend((f.w as u16).to_le_bytes());
            info.extend(u16::try_from(f.h).unwrap().to_le_bytes());
            info.extend(1u16.to_le_bytes());
            info.extend(f.bpp.to_le_bytes());
        } else {
            info.extend(f.w.to_le_bytes());
            info.extend(f.h.to_le_bytes());
            info.extend(1u16.to_le_bytes());
            info.extend(f.bpp.to_le_bytes());
            info.extend(f.compression.to_le_bytes());
            info.extend((f.data.len() as u32).to_le_bytes());
            info.extend([0u8; 8]);
            info.extend((f.palette.len() as u32).to_le_bytes());
            info.extend(0u32.to_le_bytes());
        }
        if let Some(masks) = f.masks {
            // After a 40-byte header come the three color masks; a version 4 header holds all four.
            let count = if f.header == 40 { 3 } else { 4 };
            info.extend(masks[..count].iter().flat_map(|m| m.to_le_bytes()));
        }
        if f.header > 40 {
            info.resize(f.header as usize, 0);
        }
        for c in f.palette {
            info.extend([c[2], c[1], c[0]]);
            if f.header != 12 {
                info.push(0);
            }
        }
        let offset = 14 + info.len();
        let mut b = b"BM".to_vec();
        b.extend(((offset + f.data.len()) as u32).to_le_bytes());
        b.extend([0u8; 4]);
        b.extend((offset as u32).to_le_bytes());
        b.extend(info);
        b.extend_from_slice(f.data);
        b
    }

    /// Rows of `row` bytes given top to bottom, each padded to four bytes, stored bottom to top unless `top_down`.
    fn rows(pixels: &[u8], row: usize, top_down: bool) -> Vec<u8> {
        let mut rows: Vec<Vec<u8>> = pixels
            .chunks_exact(row)
            .map(|r| {
                let mut r = r.to_vec();
                r.resize(row.div_ceil(4) * 4, 0);
                r
            })
            .collect();
        if !top_down {
            rows.reverse();
        }
        rows.concat()
    }

    /// RLE8 (or RLE4) data for rows of `w` palette indices, stored bottom to top: runs as encoded runs, three or
    /// more differing pixels in absolute mode, an end-of-line mark after each row and an end-of-bitmap mark.
    fn rle(idx: &[u8], w: usize, four: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for row in idx.chunks_exact(w).rev() {
            let mut i = 0;
            while i < row.len() {
                let run = row[i..].iter().take(255).take_while(|&&v| v == row[i]).count();
                let mut end = i + 1;
                while end < row.len() && end - i < 255 && !(end + 1 < row.len() && row[end] == row[end + 1]) {
                    end += 1;
                }
                if run >= 2 || end - i < 3 {
                    let v = row[i];
                    out.extend([run as u8, if four { v << 4 | v } else { v }]);
                    i += run;
                    continue;
                }
                let literal = &row[i..end];
                out.extend([0, literal.len() as u8]);
                let packed = if four { pack(literal, 4) } else { literal.to_vec() };
                out.extend_from_slice(&packed);
                if packed.len() % 2 == 1 {
                    out.push(0);
                }
                i = end;
            }
            out.extend([0, 0]);
        }
        out.extend([0, 1]);
        out
    }

    fn palette(seed: u64, n: usize) -> Vec<[u8; 3]> {
        noise(seed, n * 3).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    }

    /// Every depth, header, mask layout and compression.
    ///
    /// Not compared: a 32-bit bitmap's fourth byte when no mask says what it is. The format calls it unused and
    /// `image` drops it; this decoder reads it as alpha unless it's zero throughout, since writers do put alpha
    /// there.
    #[test]
    fn bmp_agrees_with_the_image_crate() {
        let (w, h) = (5usize, 3usize);
        let n = w * h;
        let mut cases = Vec::new();
        for (name, top_down) in [("bottom up", false), ("top down", true)] {
            let f = Bmp {
                h: if top_down { -3 } else { 3 },
                data: &rows(&noise(1, n * 3), w * 3, top_down),
                ..Bmp::default()
            };
            cases.push(case(&format!("24-bit {name}"), bmp(&f)));
        }
        // 32 bits with nothing in the fourth byte: three channels either way.
        let bgrx: Vec<u8> = noise(2, n * 3)
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 0])
            .collect();
        let f = Bmp {
            bpp: 32,
            data: &rows(&bgrx, w * 4, false),
            ..Bmp::default()
        };
        cases.push(case("32-bit", bmp(&f)));
        // 16 bits: five bits a channel by default, or as the masks say.
        let f = Bmp {
            bpp: 16,
            data: &rows(&noise(3, n * 2), w * 2, false),
            ..Bmp::default()
        };
        cases.push(case("16-bit 5-5-5", bmp(&f)));
        let f = Bmp {
            bpp: 16,
            compression: 3,
            masks: Some([0xF800, 0x07E0, 0x001F, 0]),
            data: &rows(&noise(4, n * 2), w * 2, false),
            ..Bmp::default()
        };
        cases.push(case("16-bit 5-6-5 masks", bmp(&f)));
        let f = Bmp {
            bpp: 16,
            compression: 3,
            header: 108,
            masks: Some([0x0F00, 0x00F0, 0x000F, 0xF000]),
            data: &rows(&noise(5, n * 2), w * 2, false),
            ..Bmp::default()
        };
        cases.push(case("16-bit 4-4-4-4 masks", bmp(&f)));
        let f = Bmp {
            bpp: 32,
            compression: 3,
            masks: Some([0x0000_00FF, 0x0000_FF00, 0x00FF_0000, 0]),
            data: &rows(&noise(6, n * 4), w * 4, true),
            h: -3,
            ..Bmp::default()
        };
        cases.push(case("32-bit masks, red first", bmp(&f)));
        let f = Bmp {
            bpp: 32,
            compression: 3,
            header: 108,
            masks: Some([0x00FF_0000, 0x0000_FF00, 0x0000_00FF, 0xFF00_0000]),
            data: &rows(&noise(7, n * 4), w * 4, false),
            ..Bmp::default()
        };
        cases.push(case("32-bit masks with alpha", bmp(&f)));
        let f = Bmp {
            bpp: 32,
            compression: 3,
            masks: Some([0x3FF0_0000, 0x000F_FC00, 0x0000_03FF, 0]),
            data: &rows(&noise(8, n * 4), w * 4, false),
            ..Bmp::default()
        };
        cases.push(case("32-bit 10-10-10 masks", bmp(&f)));
        // Palettes at each depth: full, and shorter than the depth allows.
        for (bpp, colors) in [(8u16, 256usize), (8, 7), (4, 16), (4, 5), (2, 4), (1, 2)] {
            let (w, h) = (9usize, 4usize);
            let idx = indices(9 + u64::from(bpp), w * h, colors as u32);
            let packed: Vec<u8> = idx.chunks_exact(w).flat_map(|r| pack(r, usize::from(bpp))).collect();
            let f = Bmp {
                w: 9,
                h: 4,
                bpp,
                palette: &palette(20, colors),
                data: &rows(&packed, (w * usize::from(bpp)).div_ceil(8), false),
                ..Bmp::default()
            };
            cases.push(case(&format!("{bpp}-bit palette of {colors}"), bmp(&f)));
        }
        // Run-length coding, with runs and absolute stretches in every row.
        for (bpp, compression) in [(8u16, 1u32), (4, 2)] {
            let (w, h) = (23usize, 6usize);
            let idx: Vec<u8> = runs(30 + u64::from(bpp), w * h, 1).iter().map(|v| v % 16).collect();
            let f = Bmp {
                w: 23,
                h: 6,
                bpp,
                compression,
                palette: &palette(31, 16),
                data: &rle(&idx, w, bpp == 4),
                ..Bmp::default()
            };
            cases.push(case(&format!("RLE{bpp}"), bmp(&f)));
        }
        // OS/2 headers: 16-bit sizes and three-byte palette entries.
        let f = Bmp {
            header: 12,
            data: &rows(&noise(40, n * 3), w * 3, false),
            ..Bmp::default()
        };
        cases.push(case("OS/2 24-bit", bmp(&f)));
        let f = Bmp {
            header: 12,
            bpp: 8,
            palette: &palette(41, 256),
            data: &rows(&noise(42, n), w, false),
            ..Bmp::default()
        };
        cases.push(case("OS/2 8-bit palette", bmp(&f)));
        agree("a.bmp", &cases, image_crate(image::ImageFormat::Bmp));
    }
}

mod gif {
    use super::*;

    /// A GIF's parts: one frame on a logical screen.
    struct Gif<'a> {
        version: &'a [u8],
        screen: (u16, u16),
        /// Left, top, width and height.
        frame: (u16, u16, u16, u16),
        /// Color tables: a power of two entries each, from 2 to 256.
        global: Option<&'a [u8]>,
        local: Option<&'a [u8]>,
        transparent: Option<u8>,
        interlaced: bool,
        /// The frame's color indices, rows top to bottom.
        idx: &'a [u8],
    }

    fn gif(f: &Gif) -> Vec<u8> {
        // A table of 2^(n + 1) colors is flagged with n.
        let size = |table: &[u8]| (table.len() / 3).trailing_zeros() as u8 - 1;
        let mut b = f.version.to_vec();
        b.extend(f.screen.0.to_le_bytes());
        b.extend(f.screen.1.to_le_bytes());
        b.extend([f.global.map_or(0, |t| 0x80 | size(t)), 0, 0]);
        b.extend_from_slice(f.global.unwrap_or_default());
        if let Some(t) = f.transparent {
            b.extend([0x21, 0xF9, 4, 1, 0, 0, t, 0]);
        }
        b.push(0x2C);
        for v in [f.frame.0, f.frame.1, f.frame.2, f.frame.3] {
            b.extend(v.to_le_bytes());
        }
        b.push(f.local.map_or(0, |t| 0x80 | size(t)) | if f.interlaced { 0x40 } else { 0 });
        b.extend_from_slice(f.local.unwrap_or_default());
        let (w, h) = (usize::from(f.frame.2), usize::from(f.frame.3));
        let stored: Vec<u8> = if f.interlaced {
            // Four passes: every 8th row from 0, every 8th from 4, every 4th from 2, every 2nd from 1.
            [(0, 8), (4, 8), (2, 4), (1, 2)]
                .into_iter()
                .flat_map(|(start, step)| (start..h).step_by(step))
                .flat_map(|y| f.idx[y * w..(y + 1) * w].iter().copied())
                .collect()
        } else {
            f.idx.to_vec()
        };
        let colors = f.local.or(f.global).map_or(2, |t| t.len() / 3);
        let min = colors.trailing_zeros().max(2);
        b.push(min as u8);
        let coded = weezl::encode::Encoder::new(weezl::BitOrder::Lsb, min as u8)
            .encode(&stored)
            .unwrap();
        for block in coded.chunks(255) {
            b.push(block.len() as u8);
            b.extend_from_slice(block);
        }
        b.extend([0, 0x3B]);
        b
    }

    /// Color tables of every size, global and local, transparency, interlacing, and a frame set inside its screen.
    #[test]
    fn gif_agrees_with_the_image_crate() {
        let mut cases = Vec::new();
        let base = Gif {
            version: b"GIF89a",
            screen: (13, 11),
            frame: (0, 0, 13, 11),
            global: None,
            local: None,
            transparent: None,
            interlaced: false,
            idx: &[],
        };
        let n = 13 * 11;
        for colors in [2usize, 4, 16, 128, 256] {
            let table = noise(colors as u64, colors * 3);
            let f = Gif {
                global: Some(&table),
                idx: &indices(1, n, colors as u32),
                ..base
            };
            cases.push(case(&format!("{colors} colors"), gif(&f)));
        }
        let table = noise(2, 16 * 3);
        let idx = indices(3, n, 16);
        let f = Gif {
            version: b"GIF87a",
            global: Some(&table),
            idx: &idx,
            ..base
        };
        cases.push(case("version 87a", gif(&f)));
        let f = Gif {
            global: Some(&table),
            transparent: Some(5),
            idx: &idx,
            ..base
        };
        cases.push(case("a transparent color", gif(&f)));
        let f = Gif {
            global: Some(&table),
            interlaced: true,
            idx: &idx,
            ..base
        };
        cases.push(case("interlaced", gif(&f)));
        let local = noise(4, 8 * 3);
        let f = Gif {
            global: Some(&table),
            local: Some(&local),
            idx: &indices(5, n, 8),
            ..base
        };
        cases.push(case("a local color table", gif(&f)));
        let f = Gif {
            local: Some(&local),
            idx: &indices(6, n, 8),
            ..base
        };
        cases.push(case("only a local color table", gif(&f)));
        // The frame covers part of the screen; the rest is transparent.
        let f = Gif {
            screen: (16, 14),
            frame: (2, 1, 13, 11),
            global: Some(&table),
            idx: &idx,
            ..base
        };
        cases.push(case("a frame inside its screen", gif(&f)));
        // Enough pixels that the codes widen to 12 bits and the table is cleared.
        let table = noise(7, 256 * 3);
        let f = Gif {
            screen: (128, 96),
            frame: (0, 0, 128, 96),
            global: Some(&table),
            interlaced: true,
            idx: &noise(8, 128 * 96),
            ..base
        };
        cases.push(case("a full code table", gif(&f)));
        agree("a.gif", &cases, image_crate(image::ImageFormat::Gif));
    }
}

mod tiff {
    use std::io::Write as _;

    use super::*;

    /// A TIFF's parts: one image.
    #[derive(Clone)]
    struct Tiff {
        little_endian: bool,
        w: usize,
        h: usize,
        /// Samples per pixel and bits per sample.
        spp: usize,
        bits: usize,
        photometric: u16,
        /// 1 (none), 5 (LZW), 8 (Deflate) or 32773 (`PackBits`).
        compression: u16,
        predictor: bool,
        planar: bool,
        rows_per_strip: usize,
        /// Tile width and height, instead of strips.
        tile: Option<(usize, usize)>,
        /// What the sample after the colors is: 1 (associated alpha) or 2 (unassociated).
        extra: Option<u16>,
        /// Red, green and blue values for each index.
        colormap: Vec<u16>,
        /// `w * h * spp` sample values.
        samples: Vec<u16>,
    }

    impl Tiff {
        fn new(w: usize, h: usize, spp: usize, bits: usize, photometric: u16, seed: u64) -> Self {
            let mut rng = Pcg32::new(seed);
            Self {
                little_endian: true,
                w,
                h,
                spp,
                bits,
                photometric,
                compression: 1,
                predictor: false,
                planar: false,
                rows_per_strip: h,
                tile: None,
                extra: None,
                colormap: Vec::new(),
                samples: (0..w * h * spp).map(|_| rng.below(1 << bits) as u16).collect(),
            }
        }

        fn u16(&self, v: u16) -> [u8; 2] {
            if self.little_endian {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        }

        fn u32(&self, v: u32) -> [u8; 4] {
            if self.little_endian {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        }

        /// The chunk `cw` × `ch` at (`x0`, `y0`), of one plane or of every sample: packed rows, differenced when
        /// the predictor is on, then compressed. Pixels past the image's edge (a tile's padding) are zero.
        fn chunk(&self, (x0, y0): (usize, usize), (cw, ch): (usize, usize), plane: Option<usize>) -> Vec<u8> {
            let per_pixel = if plane.is_some() { 1 } else { self.spp };
            let mut rows = Vec::new();
            for y in y0..y0 + ch {
                let mut row: Vec<u16> = Vec::with_capacity(cw * per_pixel);
                for x in x0..x0 + cw {
                    for s in 0..per_pixel {
                        let at = (y * self.w + x) * self.spp + plane.unwrap_or(0) + s;
                        row.push(if x < self.w && y < self.h { self.samples[at] } else { 0 });
                    }
                }
                if self.predictor {
                    // Each sample less the same sample of the pixel before, from the row's end back.
                    for i in (per_pixel..row.len()).rev() {
                        row[i] = row[i].wrapping_sub(row[i - per_pixel]) & (u32::MAX >> (32 - self.bits)) as u16;
                    }
                }
                let bytes: Vec<u8> = match self.bits {
                    16 => row.iter().flat_map(|&v| self.u16(v)).collect(),
                    8 => row.iter().map(|&v| v as u8).collect(),
                    bits => pack(&row.iter().map(|&v| v as u8).collect::<Vec<u8>>(), bits),
                };
                rows.push(bytes);
            }
            match self.compression {
                1 => rows.concat(),
                5 => weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
                    .encode(&rows.concat())
                    .unwrap(),
                8 => {
                    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                    z.write_all(&rows.concat()).unwrap();
                    z.finish().unwrap()
                }
                // Each row is packed by itself.
                _ => rows.iter().flat_map(|r| packbits(r)).collect(),
            }
        }

        fn file(&self) -> Vec<u8> {
            let mut b = if self.little_endian {
                b"II".to_vec()
            } else {
                b"MM".to_vec()
            };
            b.extend(self.u16(42));
            b.extend([0u8; 4]);
            // The chunks, plane by plane, rows of tiles (or strips) top to bottom.
            let (cw, ch) = self.tile.unwrap_or((self.w, self.rows_per_strip));
            let planes: Vec<Option<usize>> = if self.planar {
                (0..self.spp).map(Some).collect()
            } else {
                vec![None]
            };
            let (mut offsets, mut counts) = (Vec::new(), Vec::new());
            for plane in planes {
                for y0 in (0..self.h).step_by(ch) {
                    for x0 in (0..self.w).step_by(cw) {
                        // A strip holds only the rows the image has; a tile is always whole.
                        let rows = if self.tile.is_some() { ch } else { ch.min(self.h - y0) };
                        let data = self.chunk((x0, y0), (cw, rows), plane);
                        offsets.push(b.len() as u32);
                        counts.push(data.len() as u32);
                        b.extend(data);
                        if b.len() % 2 == 1 {
                            b.push(0);
                        }
                    }
                }
            }
            // The tags, in order: (tag, type (3 is a short, 4 a long), values).
            let short = |v: usize| (3u16, vec![v as u32]);
            let mut tags: Vec<(u16, (u16, Vec<u32>))> = vec![
                (256, short(self.w)),
                (257, short(self.h)),
                (258, (3, vec![self.bits as u32; self.spp])),
                (259, short(usize::from(self.compression))),
                (262, short(usize::from(self.photometric))),
                (277, short(self.spp)),
                (284, short(if self.planar { 2 } else { 1 })),
            ];
            if self.tile.is_some() {
                tags.extend([
                    (322, short(cw)),
                    (323, short(ch)),
                    (324, (4, offsets)),
                    (325, (4, counts)),
                ]);
            } else {
                tags.extend([(273, (4, offsets)), (278, short(ch)), (279, (4, counts))]);
            }
            if self.predictor {
                tags.push((317, short(2)));
            }
            if !self.colormap.is_empty() {
                tags.push((320, (3, self.colormap.iter().map(|&v| u32::from(v)).collect())));
            }
            if let Some(extra) = self.extra {
                tags.push((338, short(usize::from(extra))));
            }
            tags.sort_by_key(|t| t.0);
            // Values longer than four bytes go before the directory, which points at them.
            let mut entries = Vec::new();
            for (tag, (kind, values)) in tags {
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|&v| {
                        if kind == 3 {
                            self.u16(v as u16).to_vec()
                        } else {
                            self.u32(v).to_vec()
                        }
                    })
                    .collect();
                let mut field = [0u8; 4];
                if bytes.len() <= 4 {
                    field[..bytes.len()].copy_from_slice(&bytes);
                } else {
                    field = self.u32(b.len() as u32);
                    b.extend(&bytes);
                    if b.len() % 2 == 1 {
                        b.push(0);
                    }
                }
                entries.extend(self.u16(tag));
                entries.extend(self.u16(kind));
                entries.extend(self.u32(values.len() as u32));
                entries.extend(field);
            }
            let ifd = self.u32(b.len() as u32);
            b[4..8].copy_from_slice(&ifd);
            b.extend(self.u16((entries.len() / 12) as u16));
            b.extend(entries);
            b.extend([0u8; 4]);
            b
        }
    }

    /// Gray, RGB and CMYK images at each depth, in strips and tiles, each compression, both byte orders, separate
    /// planes, the predictor, and alpha.
    ///
    /// Not compared, because `image` doesn't read them: palette images, gray of 2 and 4 bits, gray with alpha,
    /// tiles of separate planes (it panics), and LZW in tiles (it finds no end code in what its own encoder
    /// wrote).
    #[test]
    fn tiff_agrees_with_the_image_crate() {
        let mut cases = Vec::new();
        let rgb = Tiff::new(21, 18, 3, 8, 2, 1);
        cases.push(case("RGB", rgb.file()));
        let mut t = rgb.clone();
        t.little_endian = false;
        cases.push(case("RGB, big-endian", t.file()));
        let mut t = rgb.clone();
        t.rows_per_strip = 4;
        cases.push(case("RGB in strips of 4 rows", t.file()));
        let mut t = rgb.clone();
        t.tile = Some((16, 16));
        cases.push(case("RGB in tiles", t.file()));
        let mut t = rgb.clone();
        t.planar = true;
        t.rows_per_strip = 5;
        cases.push(case("RGB in separate planes", t.file()));
        // Each compression over samples with runs in them, and over noise.
        for (name, compression) in [("LZW", 5u16), ("Deflate", 8), ("PackBits", 32773)] {
            let mut t = rgb.clone();
            t.compression = compression;
            t.rows_per_strip = 7;
            cases.push(case(&format!("RGB, {name}"), t.file()));
            for s in &mut t.samples {
                *s /= 64;
            }
            cases.push(case(&format!("RGB with runs, {name}"), t.file()));
            t.predictor = true;
            cases.push(case(&format!("RGB with runs, {name}, predictor"), t.file()));
            if compression != 5 {
                t.tile = Some((16, 16));
                cases.push(case(&format!("RGB in tiles, {name}, predictor"), t.file()));
            }
        }
        // Enough LZW codes to fill the table.
        let mut t = Tiff::new(96, 80, 3, 8, 2, 2);
        t.compression = 5;
        cases.push(case("RGB, LZW, a full code table", t.file()));
        // Gray at each depth, either way up the scale.
        for bits in [1usize, 8] {
            for (name, photometric) in [("black is zero", 1u16), ("white is zero", 0)] {
                let t = Tiff::new(21, 6, 1, bits, photometric, 3);
                cases.push(case(&format!("{bits}-bit gray, {name}"), t.file()));
            }
        }
        // 16 bits to 8: this decoder keeps the high byte, `image` rounds.
        for little_endian in [true, false] {
            let mut t = Tiff::new(21, 6, 1, 16, 1, 4);
            t.little_endian = little_endian;
            cases.push(within(
                &format!("16-bit gray, little-endian {little_endian}"),
                1,
                t.file(),
            ));
            let mut t = Tiff::new(21, 6, 3, 16, 2, 5);
            t.little_endian = little_endian;
            t.compression = 5;
            t.predictor = true;
            let name = format!("16-bit RGB, LZW, predictor, little-endian {little_endian}");
            cases.push(within(&name, 1, t.file()));
        }
        // Alpha that isn't multiplied in, and alpha that is: fully clear or opaque, where the two agree.
        let mut t = Tiff::new(21, 6, 4, 8, 2, 6);
        t.extra = Some(2);
        cases.push(case("RGBA", t.file()));
        t.extra = Some(1);
        for p in t.samples.chunks_exact_mut(4) {
            p[3] = if p[3] < 128 { 0 } else { 255 };
            if p[3] == 0 {
                p[..3].fill(0);
            }
        }
        cases.push(case("RGBA, associated alpha", t.file()));
        let mut t = Tiff::new(21, 6, 4, 16, 2, 8);
        t.extra = Some(2);
        cases.push(within("16-bit RGBA", 1, t.file()));
        // CMYK: each ink times black, rounded differently by the two.
        let t = Tiff::new(21, 6, 4, 8, 5, 11);
        cases.push(within("CMYK", 1, t.file()));
        agree("a.tif", &cases, image_crate(image::ImageFormat::Tiff));
    }
}

mod psd {
    use super::*;

    /// A Photoshop document with only its merged image: `channels` planes of `depth`-bit samples, raw or
    /// run-length coded.
    fn psd((w, h): (u32, u32), channels: u16, depth: u16, mode: u16, rle: bool, planes: &[u8]) -> Vec<u8> {
        let mut b = b"8BPS".to_vec();
        b.extend(1u16.to_be_bytes());
        b.extend([0u8; 6]);
        b.extend(channels.to_be_bytes());
        b.extend(h.to_be_bytes());
        b.extend(w.to_be_bytes());
        b.extend(depth.to_be_bytes());
        b.extend(mode.to_be_bytes());
        // No color mode data, no resources, no layers.
        b.extend([0u8; 12]);
        b.extend(u16::from(rle).to_be_bytes());
        if rle {
            // Every row's coded length, then the rows.
            let row = w as usize * usize::from(depth / 8);
            let coded: Vec<Vec<u8>> = planes.chunks_exact(row).map(packbits).collect();
            b.extend(coded.iter().flat_map(|r| (r.len() as u16).to_be_bytes()));
            b.extend(coded.concat());
        } else {
            b.extend_from_slice(planes);
        }
        b
    }

    /// Alpha made fully clear or opaque. Between the two this decoder takes the white Photoshop blended the
    /// colors over back off, and the reference doesn't.
    fn hard_alpha(planes: &mut [u8], n: usize, sample: usize) {
        for a in &mut planes[3 * n * sample..] {
            *a = if *a < 128 { 0 } else { 255 };
        }
    }

    /// RGB and gray merged images, raw and run-length coded, with and without alpha.
    ///
    /// Not compared: 16-bit samples and bitmap, indexed, duotone and CMYK documents, which the `psd` crate
    /// doesn't read, and alpha between clear and opaque.
    #[test]
    fn psd_agrees_with_the_psd_crate() {
        let reference = |bytes: &[u8]| {
            let doc = ::psd::Psd::from_bytes(bytes).map_err(|e| e.to_string())?;
            Ok((doc.width(), doc.height(), doc.rgba()))
        };
        let size = (9u32, 5u32);
        let n = 45;
        let mut cases = Vec::new();
        for rle in [false, true] {
            let how = if rle { "RLE" } else { "raw" };
            cases.push(case(
                &format!("RGB, {how}"),
                psd(size, 3, 8, 3, rle, &runs(1, n * 3, 1)),
            ));
            let mut planes = runs(2, n * 4, 1);
            hard_alpha(&mut planes, n, 1);
            cases.push(case(&format!("RGB and alpha, {how}"), psd(size, 4, 8, 3, rle, &planes)));
            cases.push(case(&format!("gray, {how}"), psd(size, 1, 8, 1, rle, &runs(3, n, 1))));
        }
        agree("a.psd", &cases, reference);
    }
}

mod dds {
    use super::*;

    /// A DDS file of one level: pixel format flags, a four-character code or a bit count with masks, and a
    /// DXGI format for a `DX10` header.
    fn dds(
        (w, h): (u32, u32),
        flags: u32,
        code: [u8; 4],
        bits: u32,
        masks: [u32; 4],
        dxgi: Option<u32>,
        data: &[u8],
    ) -> Vec<u8> {
        let mut b = b"DDS ".to_vec();
        // Size, flags (caps, height, width, pixel format), height, width, pitch, depth, mipmaps, 11 reserved.
        for v in [124, 0x1007, h, w, 0, 0, 1] {
            b.extend(u32::to_le_bytes(v));
        }
        b.extend([0u8; 44]);
        b.extend(32u32.to_le_bytes());
        b.extend(flags.to_le_bytes());
        b.extend(code);
        b.extend(bits.to_le_bytes());
        b.extend(masks.iter().flat_map(|m| m.to_le_bytes()));
        // Caps (a texture), three more caps, one reserved.
        b.extend(0x1000u32.to_le_bytes());
        b.extend([0u8; 16]);
        if let Some(format) = dxgi {
            // Format, dimension (2D), flags, array size, more flags.
            for v in [format, 3, 0, 1, 0] {
                b.extend(u32::to_le_bytes(v));
            }
        }
        b.extend_from_slice(data);
        b
    }

    /// Every block compression by both its names, and uncompressed pixels by DXGI format and by masks. The
    /// blocks are noise, so each coding within a block comes up; the size isn't a multiple of four.
    ///
    /// A value between a block's two ends is rounded differently by the two decoders: Direct3D allows a color
    /// either way, and a BC3 alpha or a BC4 step is the nearest value here and often the one below it there. So
    /// the blocks compare within one step. BC4 and BC5 compare the channels they store: this decoder shows BC4
    /// as gray and rebuilds a normal's third axis from BC5.
    ///
    /// Not compared, because `image_dds` doesn't read them: BGRX, 5-6-5 and luminance pixels.
    #[test]
    fn dds_agrees_with_image_dds() {
        let reference = |bytes: &[u8]| {
            let dds = image_dds::ddsfile::Dds::read(bytes).map_err(|e| e.to_string())?;
            let surface = image_dds::Surface::from_dds(&dds).map_err(|e| e.to_string())?;
            let image = surface.decode_rgba8().map_err(|e| e.to_string())?;
            Ok((image.width, image.height, image.data))
        };
        let size = (10u32, 7u32);
        let blocks = 3 * 2;
        let n = 70;
        let mut cases = Vec::new();
        let no = [0u32; 4];
        for (name, code, dxgi, block, tolerance, channels) in [
            ("BC1", b"DXT1", 71u32, 8usize, 1u8, 4usize),
            ("BC2", b"DXT3", 74, 16, 1, 4),
            ("BC3", b"DXT5", 77, 16, 1, 4),
            ("BC4", b"ATI1", 80, 8, 1, 1),
            ("BC4", b"BC4U", 80, 8, 1, 1),
            ("BC5", b"ATI2", 83, 16, 1, 2),
            ("BC5", b"BC5U", 83, 16, 1, 2),
        ] {
            let data = noise(u64::from(dxgi), blocks * block);
            let by_code = dds(size, 4, *code, 0, no, None, &data);
            let by_format = dds(size, 4, *b"DX10", 0, no, Some(dxgi), &data);
            for (how, bytes) in [
                (String::from_utf8_lossy(code).into_owned(), by_code),
                (format!("DXGI {dxgi}"), by_format),
            ] {
                cases.push(Case {
                    name: format!("{name} as {how}"),
                    bytes,
                    tolerance,
                    channels,
                });
            }
        }
        // Uncompressed, by DXGI format: RGBA, BGRA and one red channel shown as gray.
        for (name, dxgi, bytes, channels) in [("RGBA8", 28u32, 4usize, 4usize), ("BGRA8", 87, 4, 4), ("R8", 61, 1, 1)] {
            let file = dds(size, 4, *b"DX10", 0, no, Some(dxgi), &noise(u64::from(dxgi), n * bytes));
            cases.push(Case {
                name: name.into(),
                bytes: file,
                tolerance: 0,
                channels,
            });
        }
        // Uncompressed, by masks: flags 0x40 is RGB, 0x1 alpha pixels.
        for (name, flags, bits, masks) in [
            (
                "BGRA masks",
                0x41u32,
                32u32,
                [0x00FF_0000u32, 0xFF00, 0xFF, 0xFF00_0000],
            ),
            ("RGBA masks", 0x41, 32, [0xFF, 0xFF00, 0x00FF_0000, 0xFF00_0000]),
            ("BGR masks", 0x40, 24, [0x00FF_0000, 0xFF00, 0xFF, 0]),
            ("5-5-5-1 masks", 0x41, 16, [0x7C00, 0x03E0, 0x1F, 0x8000]),
            ("4-4-4-4 masks", 0x41, 16, [0x0F00, 0x00F0, 0x0F, 0xF000]),
        ] {
            let data = noise(u64::from(bits) + u64::from(flags), n * (bits / 8) as usize);
            cases.push(case(name, dds(size, flags, [0; 4], bits, masks, None, &data)));
        }
        agree("a.dds", &cases, reference);
    }
}
