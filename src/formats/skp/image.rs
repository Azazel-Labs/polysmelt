//! Texture images in a form glTF can embed. `SketchUp` keeps each texture as the file it was made from: PNG and JPEG
//! pass through as they are, uncompressed BMP and TGA (plain or run-length) are decoded and written out as PNG, and
//! anything else goes to the [`Assets`], which turn GIF, TIFF and the other formats with features of their names
//! into PNG as they do every reader's textures. What none of these makes an image of is dropped, and the face keeps
//! its material's color.

use std::path::Path;

use super::glb::image_mime;
use crate::assets::Assets;
use crate::bytes::{u16_at, u32_at};
use crate::images::MAX_PIXELS;

/// The widest or tallest image decoded. Real textures are a few thousand pixels across.
const MAX_SIDE: usize = 16_384;

/// `bytes` as a PNG or JPEG file: unchanged when it already is one, re-encoded as PNG when it is a BMP or TGA this
/// can decode, or as `assets` turn the texture file `name` into one ([`Assets::image`]). `None` otherwise.
pub(crate) fn embeddable(bytes: &[u8], name: &Path, assets: &dyn Assets) -> Option<Vec<u8>> {
    if image_mime(bytes).is_some() {
        return Some(bytes.to_vec());
    }
    if let Some(png) = bmp(bytes).or_else(|| tga(bytes)).and_then(|image| image.png()) {
        return Some(png);
    }
    let image = assets.image(bytes, name)?;
    image_mime(&image.bytes).map(|_| image.bytes.to_vec())
}

/// Decoded pixels, RGBA, top row first.
struct Rgba {
    w: usize,
    h: usize,
    pixels: Vec<u8>,
}

impl Rgba {
    /// A blank image, or `None` when its size is out of bounds or more than `budget` pixels.
    fn new(w: usize, h: usize, budget: usize) -> Option<Self> {
        let n = w.checked_mul(h)?;
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || n > MAX_PIXELS.min(budget) {
            return None;
        }
        Some(Self {
            w,
            h,
            pixels: vec![0; n * 4],
        })
    }

    /// Makes the image opaque when every alpha value is zero: writers that keep a fourth byte but no alpha
    /// leave it at zero, and nobody means a texture that is entirely invisible.
    fn fix_unused_alpha(&mut self) {
        if self.pixels.chunks_exact(4).all(|p| p[3] == 0) {
            for p in self.pixels.chunks_exact_mut(4) {
                p[3] = 255;
            }
        }
    }

    /// The image as a PNG, RGB when it is fully opaque.
    fn png(&self) -> Option<Vec<u8>> {
        let opaque = self.pixels.chunks_exact(4).all(|p| p[3] == 255);
        let (w, h) = (u32::try_from(self.w).ok()?, u32::try_from(self.h).ok()?);
        if !opaque {
            return crate::png::encode(w, h, 4, &self.pixels);
        }
        let rgb: Vec<u8> = self.pixels.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        crate::png::encode(w, h, 3, &rgb)
    }
}

fn usize_at(d: &[u8], at: usize) -> Option<usize> {
    usize::try_from(u32_at(d, at)?).ok()
}

/// One channel of a packed pixel: the bits `mask` selects, scaled to 0-255. `None` when the mask is empty.
fn channel(px: u32, mask: u32) -> Option<u8> {
    if mask == 0 {
        return None;
    }
    let v = u64::from((px & mask) >> mask.trailing_zeros());
    let max = u64::from(mask >> mask.trailing_zeros());
    u8::try_from(v * 255 / max).ok()
}

/// A BMP bitmap, with its 14-byte file header (`BM`) or without it (a bare DIB, as some `SketchUp` versions
/// store textures). Uncompressed 1, 4, 8, 16, 24 and 32 bits a pixel, with bit-field masks or without.
fn bmp(d: &[u8]) -> Option<Rgba> {
    let (dib, pixel_offset) = if d.starts_with(b"BM") {
        (14, Some(usize_at(d, 10)?))
    } else {
        (0, None)
    };
    let header = usize_at(d, dib)?;
    if !matches!(header, 40 | 52 | 56 | 108 | 124) {
        return None;
    }
    let field = |at: usize| u32_at(d, dib + at);
    let width = i32::from_le_bytes(field(4)?.to_le_bytes());
    let height = i32::from_le_bytes(field(8)?.to_le_bytes());
    let bpp = usize::from(u16_at(d, dib + 14)?);
    let compression = field(16)?;
    if u16_at(d, dib + 12)? != 1 || width <= 0 || height == 0 {
        return None;
    }
    let w = usize::try_from(width).ok()?;
    let h = usize::try_from(height.unsigned_abs()).ok()?;
    let top_down = height < 0;
    // masks follow a 40-byte header when it says bit fields (3, or 6 with alpha), and sit inside the longer ones
    let (masks, after_header) = match (compression, header) {
        (0, _) => (None, header),
        (3, 40) => (Some([field(40)?, field(44)?, field(48)?, 0]), 52),
        (6, 40) => (Some([field(40)?, field(44)?, field(48)?, field(52)?]), 56),
        (3 | 6, _) => {
            let alpha = if header >= 56 { field(52)? } else { 0 };
            (Some([field(40)?, field(44)?, field(48)?, alpha]), header)
        }
        _ => return None,
    };
    let palette_at = dib.checked_add(after_header)?;
    let colors = match usize_at(d, dib + 32)? {
        0 if bpp <= 8 => 1 << bpp,
        n => n.min(256),
    };
    let palette_len = if bpp <= 8 { colors * 4 } else { 0 };
    let pixels_at = match pixel_offset {
        Some(p) => p,
        None => palette_at.checked_add(palette_len)?,
    };
    let stride = w.checked_mul(bpp)?.checked_add(31)? / 32 * 4;
    let data = d.get(pixels_at..pixels_at.checked_add(stride.checked_mul(h)?)?)?;
    let palette = if bpp <= 8 {
        d.get(palette_at..palette_at.checked_add(palette_len)?)?
    } else {
        &[]
    };
    let masks = match (bpp, masks) {
        (16 | 32, Some(m)) => m,
        (16, None) => [0x7c00, 0x03e0, 0x001f, 0],
        (32, None) => [0x00ff_0000, 0x0000_ff00, 0x0000_00ff, 0xff00_0000],
        (1 | 4 | 8 | 24, None) => [0; 4],
        _ => return None,
    };
    let mut out = Rgba::new(w, h, usize::MAX)?;
    for (y, row) in data.chunks_exact(stride).enumerate() {
        let dst_y = if top_down { y } else { h - 1 - y };
        let dst = &mut out.pixels[dst_y * w * 4..(dst_y + 1) * w * 4];
        for (x, px) in dst.chunks_exact_mut(4).enumerate() {
            match bpp {
                1 | 4 | 8 => {
                    let bit = x * bpp;
                    let byte = row[bit / 8];
                    let index = usize::from(byte >> (8 - bpp - bit % 8)) & ((1 << bpp) - 1);
                    let c = palette.get(index * 4..index * 4 + 3).unwrap_or(&[0, 0, 0]);
                    px.copy_from_slice(&[c[2], c[1], c[0], 255]);
                }
                24 => {
                    let c = &row[x * 3..x * 3 + 3];
                    px.copy_from_slice(&[c[2], c[1], c[0], 255]);
                }
                _ => {
                    let v = if bpp == 16 {
                        u32::from(u16_at(row, x * 2)?)
                    } else {
                        u32_at(row, x * 4)?
                    };
                    for (o, &m) in px.iter_mut().zip(&masks) {
                        *o = channel(v, m).unwrap_or(255);
                    }
                }
            }
        }
    }
    if bpp == 32 {
        out.fix_unused_alpha();
    }
    Some(out)
}

/// A TGA image: color-mapped, truecolor or grayscale, plain or run-length encoded. TGA has no signature, so
/// the header has to make sense throughout before it is believed.
fn tga(d: &[u8]) -> Option<Rgba> {
    let &[id_len, map_type, kind] = d.get(..3)? else {
        return None;
    };
    let map_first = usize::from(u16_at(d, 3)?);
    let map_len = usize::from(u16_at(d, 5)?);
    let map_bits = *d.get(7)?;
    let w = usize::from(u16_at(d, 12)?);
    let h = usize::from(u16_at(d, 14)?);
    let depth = *d.get(16)?;
    let descriptor = *d.get(17)?;
    let mapped = matches!(kind, 1 | 9);
    let plausible = match kind & !8 {
        1 => map_type == 1 && matches!(depth, 8 | 16),
        2 => matches!(depth, 15 | 16 | 24 | 32),
        3 => matches!(depth, 8 | 16),
        _ => false,
    };
    if !plausible || map_type > 1 || (map_type == 1 && !matches!(map_bits, 15 | 16 | 24 | 32)) || descriptor & 0xc0 != 0
    {
        return None;
    }
    let map_at = 18 + usize::from(id_len);
    let map_entry = usize::from(map_bits).div_ceil(8);
    let map_size = if map_type == 1 { map_len * map_entry } else { 0 };
    let map = d.get(map_at..map_at + map_size)?;
    let data = d.get(map_at + map_size..)?;
    let bytes = usize::from(depth).div_ceil(8);
    // a run-length packet of one pixel covers at most 128, which caps what a small hostile file can ask for
    let budget = if kind & 8 != 0 {
        data.len().saturating_mul(128)
    } else {
        data.len() / bytes
    };
    let mut out = Rgba::new(w, h, budget)?;
    let alpha_bits = descriptor & 0x0f;
    let color = |p: &[u8], bits: u8, gray: bool| -> [u8; 4] {
        match (bits, gray) {
            (8, true) => [p[0], p[0], p[0], 255],
            (16, true) => [p[0], p[0], p[0], p[1]],
            (15 | 16, false) => {
                let v = u32::from(u16::from_le_bytes([p[0], p[1]]));
                let a = if bits == 16 && alpha_bits > 0 {
                    channel(v, 0x8000)
                } else {
                    None
                };
                let c = |m| channel(v, m).unwrap_or(0);
                [c(0x7c00), c(0x03e0), c(0x001f), a.unwrap_or(255)]
            }
            (24, _) => [p[2], p[1], p[0], 255],
            (_, _) => [p[2], p[1], p[0], p[3]],
        }
    };
    let gray = kind & !8 == 3;
    let pixel = |p: &[u8]| -> Option<[u8; 4]> {
        if mapped {
            let index = if depth == 8 {
                usize::from(p[0])
            } else {
                usize::from(u16::from_le_bytes([p[0], p[1]]))
            };
            let at = index.checked_sub(map_first)? * map_entry;
            Some(color(map.get(at..at + map_entry)?, map_bits, false))
        } else {
            Some(color(p, depth, gray))
        }
    };
    let n = w * h;
    let mut written = 0;
    let mut at = 0;
    while written < n {
        let (count, run) = if kind & 8 != 0 {
            let head = *data.get(at)?;
            at += 1;
            (usize::from(head & 0x7f) + 1, head & 0x80 != 0)
        } else {
            (n, false)
        };
        let count = count.min(n - written);
        if run {
            let c = pixel(data.get(at..at + bytes)?).unwrap_or([0, 0, 0, 255]);
            at += bytes;
            for px in out.pixels[written * 4..(written + count) * 4].chunks_exact_mut(4) {
                px.copy_from_slice(&c);
            }
        } else {
            let src = data.get(at..at + count * bytes)?;
            at += count * bytes;
            for (px, p) in out.pixels[written * 4..(written + count) * 4]
                .chunks_exact_mut(4)
                .zip(src.chunks_exact(bytes))
            {
                px.copy_from_slice(&pixel(p).unwrap_or([0, 0, 0, 255]));
            }
        }
        written += count;
    }
    // rows run bottom to top unless descriptor bit 5 says otherwise, and left to right unless bit 4 does
    let row = w * 4;
    if descriptor & 0x20 == 0 {
        for y in 0..h / 2 {
            let (top, bottom) = out.pixels.split_at_mut((h - 1 - y) * row);
            top[y * row..(y + 1) * row].swap_with_slice(&mut bottom[..row]);
        }
    }
    if descriptor & 0x10 != 0 {
        for line in out.pixels.chunks_exact_mut(row) {
            for x in 0..w / 2 {
                for c in 0..4 {
                    line.swap(x * 4 + c, (w - 1 - x) * 4 + c);
                }
            }
        }
    }
    if depth == 32 && !mapped {
        out.fix_unused_alpha();
    }
    Some(out)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, reason = "test images are tiny")]
mod tests {
    use super::*;

    fn embed(bytes: &[u8]) -> Option<Vec<u8>> {
        embeddable(bytes, Path::new(""), &crate::assets::Memory::new())
    }

    /// What this module's own decoders make of `bytes`, without the shared image path behind them.
    fn own(bytes: &[u8]) -> Option<Vec<u8>> {
        bmp(bytes).or_else(|| tga(bytes)).and_then(|image| image.png())
    }

    /// The size, color type and pixels of a PNG this module wrote.
    fn unpng(png: &[u8]) -> (u32, u32, u8, Vec<u8>) {
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        let kind = png[25];
        let img = image::load_from_memory_with_format(png, image::ImageFormat::Png).unwrap();
        let (w, h) = (img.width(), img.height());
        let out = if kind == 6 {
            img.to_rgba8().into_raw()
        } else {
            img.to_rgb8().into_raw()
        };
        (w, h, kind, out)
    }

    fn bmp_file(bpp: u16, height: i32, palette: &[u8], rows: &[u8], file_header: bool) -> Vec<u8> {
        let mut dib = Vec::new();
        dib.extend(40u32.to_le_bytes());
        dib.extend(2i32.to_le_bytes());
        dib.extend(height.to_le_bytes());
        dib.extend(1u16.to_le_bytes());
        dib.extend(bpp.to_le_bytes());
        dib.extend([0; 24]);
        dib.extend(palette);
        dib.extend(rows);
        if !file_header {
            return dib;
        }
        let mut out = b"BM".to_vec();
        out.extend(((14 + dib.len()) as u32).to_le_bytes());
        out.extend([0; 4]);
        out.extend(((14 + 40 + palette.len()) as u32).to_le_bytes());
        out.extend(dib);
        out
    }

    #[test]
    fn passes_png_and_jpeg_through() {
        let png = b"\x89PNG\r\n\x1a\nrest".to_vec();
        assert_eq!(embed(&png), Some(png));
        assert_eq!(embed(&[0xff, 0xd8, 0xff, 0xe0]), Some(vec![0xff, 0xd8, 0xff, 0xe0]));
        assert_eq!(embed(b"GIF89a"), None);
        assert_eq!(embed(b""), None);
    }

    #[test]
    fn decodes_bottom_up_24_bit_bitmaps() {
        // two rows of two pixels, bottom row first, each row padded to four bytes: blue, green / red, white
        let rows = [255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0];
        for file_header in [true, false] {
            let png = embed(&bmp_file(24, 2, &[], &rows, file_header)).unwrap();
            let (w, h, kind, px) = unpng(&png);
            assert_eq!((w, h, kind), (2, 2, 2));
            assert_eq!(px, [255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0]);
        }
    }

    #[test]
    fn decodes_palette_bitmaps_top_down() {
        let palette = [0, 0, 0, 0, 0, 0, 255, 0];
        // 1 bit a pixel: top row black, red; bottom row red, black
        let rows = [0b0100_0000, 0, 0, 0, 0b1000_0000, 0, 0, 0];
        let (w, h, _, px) = unpng(&embed(&bmp_file(1, -2, &palette, &rows, true)).unwrap());
        assert_eq!((w, h), (2, 2));
        assert_eq!(px, [0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn keeps_bitmap_alpha_only_when_used() {
        let opaque = [1, 2, 3, 0, 4, 5, 6, 0];
        let (_, _, kind, px) = unpng(&embed(&bmp_file(32, 1, &[], &opaque, true)).unwrap());
        assert_eq!((kind, px), (2, vec![3, 2, 1, 6, 5, 4]));
        let cutout = [1, 2, 3, 255, 4, 5, 6, 0];
        let (_, _, kind, px) = unpng(&embed(&bmp_file(32, 1, &[], &cutout, true)).unwrap());
        assert_eq!((kind, px), (6, vec![3, 2, 1, 255, 6, 5, 4, 0]));
    }

    fn tga_file(kind: u8, depth: u8, descriptor: u8, w: u16, h: u16, body: &[u8]) -> Vec<u8> {
        let mut out = vec![0, 0, kind, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        out.extend(w.to_le_bytes());
        out.extend(h.to_le_bytes());
        out.extend([depth, descriptor]);
        out.extend(body);
        out
    }

    #[test]
    fn decodes_truecolor_targas() {
        // bottom-left origin: the first row stored is the bottom one
        let body = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let (w, h, kind, px) = unpng(&embed(&tga_file(2, 24, 0, 2, 2, &body)).unwrap());
        assert_eq!((w, h, kind), (2, 2, 2));
        assert_eq!(px, [255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0]);
        let top_left = tga_file(2, 24, 0x20, 2, 2, &body);
        let (_, _, _, px) = unpng(&embed(&top_left).unwrap());
        assert_eq!(px, [0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn decodes_run_length_targas_with_alpha() {
        // a run of three translucent red pixels, then one raw opaque blue
        let body = [0x82, 0, 0, 255, 128, 0x00, 255, 0, 0, 255];
        let (w, h, kind, px) = unpng(&embed(&tga_file(10, 32, 0x28, 4, 1, &body)).unwrap());
        assert_eq!((w, h, kind), (4, 1, 6));
        assert_eq!(px, [255, 0, 0, 128, 255, 0, 0, 128, 255, 0, 0, 128, 0, 0, 255, 255]);
    }

    #[test]
    fn rejects_implausible_and_oversized_images() {
        // a run-length header claiming a huge image from a few bytes
        assert!(own(&tga_file(10, 24, 0, 16_000, 16_000, &[0xff, 1, 2, 3])).is_none());
        assert!(own(&tga_file(7, 24, 0, 2, 2, &[0; 12])).is_none());
        assert!(own(&tga_file(2, 24, 0, 2, 2, &[0; 11])).is_none());
        assert!(own(&bmp_file(24, 2, &[], &[0; 15], true)).is_none());
        assert!(own(&bmp_file(24, 0, &[], &[], true)).is_none());
        // and the shared image path behind them makes no image of a huge one either
        assert!(embed(&tga_file(10, 24, 0, 16_000, 16_000, &[0xff, 1, 2, 3])).is_none());
    }

    #[test]
    fn damaged_images_never_panic() {
        let files = [
            bmp_file(
                1,
                -2,
                &[0, 0, 0, 0, 0, 0, 255, 0],
                &[0x40, 0, 0, 0, 0x80, 0, 0, 0],
                true,
            ),
            bmp_file(32, 1, &[], &[1, 2, 3, 255, 4, 5, 6, 0], false),
            tga_file(10, 32, 0x28, 4, 1, &[0x82, 0, 0, 255, 128, 0x00, 255, 0, 0, 255]),
            tga_file(2, 16, 0x31, 2, 2, &[1, 2, 3, 4, 5, 6, 7, 8]),
        ];
        for file in files {
            for len in 0..file.len() {
                let _ = embed(&file[..len]);
            }
            let mut file = file;
            for i in 0..file.len() {
                for x in [0x01, 0x80, 0xff] {
                    file[i] ^= x;
                    let _ = embed(&file);
                    file[i] ^= x;
                }
            }
        }
    }
}
