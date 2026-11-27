//! Texture formats glTF can't embed, decoded and re-encoded as PNG for the mesh writer: TGA, BMP, GIF, TIFF, PSD
//! and DDS, each behind a feature of its name.
//!
//! Each decoder reads what texture files hold in practice and leaves the rest out (it returns no image): TGA in
//! all its color-mapped, true-color and gray kinds, raw or run-length coded; BMP from 1 to 32 bits a pixel, with
//! bit-field masks and RLE, or the PNG or JPEG one wraps; a GIF's first frame; baseline TIFF's first image (strips
//! or tiles, uncompressed, LZW, `PackBits` or Deflate, 1- to 16-bit gray, palette, RGB and CMYK, with alpha); a PSD's
//! merged image; and a DDS's top mipmap level (BC1 to BC5, or uncompressed). 16-bit samples keep their high byte.
//!
//! Nothing is allocated from a size a file states: pixels take room as they decode, and an image whose data falls
//! short is padded only when at least half of it is there (or it's small), so a header claiming a huge image over a
//! few bytes of data makes nothing. Images are capped at [`MAX_PIXELS`].

use std::path::Path;

use crate::images::{FREE, MAX_PIXELS};
use crate::mesh::Image;
use crate::num::usize_of;

#[cfg(feature = "bmp")]
mod bmp;
#[cfg(feature = "dds")]
mod dds;
#[cfg(feature = "gif")]
mod gif;
#[cfg(any(feature = "gif", feature = "tiff"))]
mod lzw;
#[cfg(feature = "psd")]
mod psd;
#[cfg(feature = "tga")]
mod tga;
#[cfg(feature = "tiff")]
mod tiff;

/// `bytes`, a texture file named `name`, as a PNG (or the PNG or JPEG a BMP wraps), when it's a format a feature
/// decodes. The format comes from the bytes; TGA, which has no signature, from its name (or its version 2 footer).
pub(crate) fn decode(bytes: &[u8], name: &Path) -> Option<Image> {
    let raster = match bytes {
        #[cfg(feature = "bmp")]
        [b'B', b'M', ..] => {
            if let Some(inner) = bmp::wrapped(bytes) {
                return Image::png_or_jpeg(inner);
            }
            bmp::decode(bytes)
        }
        #[cfg(feature = "gif")]
        [b'G', b'I', b'F', b'8', b'7' | b'9', b'a', ..] => gif::decode(bytes),
        #[cfg(feature = "tiff")]
        [b'I', b'I', 42, 0, ..] | [b'M', b'M', 0, 42, ..] => tiff::decode(bytes),
        #[cfg(feature = "psd")]
        [b'8', b'B', b'P', b'S', ..] => psd::decode(bytes),
        #[cfg(feature = "dds")]
        [b'D', b'D', b'S', b' ', ..] => dds::decode(bytes),
        #[cfg(feature = "tga")]
        _ if tga::is_tga(bytes, name) => tga::decode(bytes),
        _ => None,
    }?;
    let _ = name;
    Image::from_png_or_jpeg(raster.png()?)
}

/// Decoded 8-bit pixels, rows top to bottom: gray, gray and alpha, RGB or RGBA (`channels` 1 to 4).
struct Raster {
    w: u32,
    h: u32,
    channels: usize,
    data: Vec<u8>,
}

impl Raster {
    /// As a PNG; an alpha channel that's opaque everywhere is left out.
    fn png(mut self) -> Option<Vec<u8>> {
        if self.channels.is_multiple_of(2)
            && self
                .data
                .chunks_exact(self.channels)
                .all(|p| p[self.channels - 1] == 255)
        {
            let c = self.channels;
            self.data = self
                .data
                .chunks_exact(c)
                .flat_map(|p| p[..c - 1].iter().copied())
                .collect();
            self.channels -= 1;
        }
        crate::png::encode(self.w, self.h, self.channels, &self.data)
    }
}

/// The pixels of a `w` × `h` image, when it may be made: not empty and not past [`MAX_PIXELS`].
fn pixels(w: u32, h: u32) -> Option<usize> {
    let n = usize_of(u64::from(w)).checked_mul(usize_of(u64::from(h)))?;
    (n > 0 && n <= MAX_PIXELS).then_some(n)
}

/// `data`, decoded toward `need` bytes, made exactly that long: cut, or padded with zeros when at least half of it
/// is there (or it's small). `None` when too little is there to pad.
#[cfg(any(feature = "bmp", feature = "dds", feature = "gif", feature = "psd", feature = "tga"))]
fn fill(data: &mut Vec<u8>, need: usize) -> Option<()> {
    if data.len() < need && data.len() < need / 2 && need > FREE {
        return None;
    }
    data.resize(need, 0);
    Some(())
}

/// Whether a buffer holding `real` bytes of decoded data may be stretched to `to` bytes (to skip pixels a format
/// leaves out): no further than twice the data, or [`FREE`].
#[cfg(feature = "bmp")]
fn may_grow(to: usize, real: usize) -> bool {
    to <= real.saturating_mul(2).max(FREE)
}

/// `n` bytes at `at`.
fn get(b: &[u8], at: usize, n: usize) -> Option<&[u8]> {
    b.get(at..at.checked_add(n)?)
}

#[cfg(any(feature = "bmp", feature = "gif", feature = "tga", feature = "tiff"))]
fn le16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(get(b, at, 2)?.try_into().ok()?))
}

#[cfg(any(feature = "bmp", feature = "dds", feature = "tiff"))]
fn le32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(get(b, at, 4)?.try_into().ok()?))
}

#[cfg(any(feature = "tiff", feature = "psd"))]
fn be16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(get(b, at, 2)?.try_into().ok()?))
}

#[cfg(any(feature = "tiff", feature = "psd"))]
fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(get(b, at, 4)?.try_into().ok()?))
}

/// `PackBits`-coded `src` decoded onto `out`, up to `limit` bytes in all.
#[cfg(any(feature = "tiff", feature = "psd"))]
fn packbits(src: &[u8], out: &mut Vec<u8>, limit: usize) {
    let mut i = 0;
    while i < src.len() && out.len() < limit {
        let n = src[i].cast_signed();
        i += 1;
        let room = limit - out.len();
        if n >= 0 {
            // n + 1 bytes as they are.
            let end = src.len().min(i + usize::from(n.unsigned_abs()) + 1);
            let take = (end - i).min(room);
            out.extend_from_slice(&src[i..i + take]);
            i = end;
        } else if n != -128 {
            // The next byte, 1 - n times.
            let Some(&b) = src.get(i) else { break };
            i += 1;
            let count = usize::from(n.unsigned_abs()) + 1;
            out.resize(out.len() + count.min(room), b);
        }
    }
}

/// A `bits`-wide channel value scaled to 8 bits.
#[cfg(any(feature = "bmp", feature = "tga", feature = "dds", feature = "tiff"))]
fn scale(v: u32, bits: u32) -> u8 {
    match bits {
        0 => 0,
        8.. => u8::try_from(v >> (bits - 8)).unwrap_or(u8::MAX),
        _ => {
            let max = (1u32 << bits) - 1;
            u8::try_from((v.min(max) * 255 + max / 2) / max).unwrap_or(u8::MAX)
        }
    }
}

/// `rows` of `row` bytes each, in the opposite order.
#[cfg(any(feature = "bmp", feature = "tga"))]
fn flip(data: &mut [u8], row: usize) {
    let rows = data.len() / row.max(1);
    for y in 0..rows / 2 {
        let (top, bottom) = data.split_at_mut((rows - 1 - y) * row);
        top[y * row..(y + 1) * row].swap_with_slice(&mut bottom[..row]);
    }
}

/// Each `row`-byte row's pixels of `channels` bytes, in the opposite order.
#[cfg(feature = "tga")]
fn mirror(data: &mut [u8], row: usize, channels: usize) {
    for r in data.chunks_exact_mut(row.max(1)) {
        let n = r.len() / channels;
        for x in 0..n / 2 {
            for c in 0..channels {
                r.swap(x * channels + c, (n - 1 - x) * channels + c);
            }
        }
    }
}

/// An alpha channel (the last of `channels`) that's zero everywhere made opaque: a writer that left it unset.
#[cfg(any(feature = "bmp", feature = "tga"))]
fn opaque_if_clear(data: &mut [u8], channels: usize) {
    if data.chunks_exact(channels).all(|p| p[channels - 1] == 0) {
        for p in data.chunks_exact_mut(channels) {
            p[channels - 1] = 255;
        }
    }
}

/// The channel `mask` picks out of `v`, scaled to 8 bits.
#[cfg(any(feature = "bmp", feature = "dds"))]
fn masked(v: u32, mask: u32) -> u8 {
    if mask == 0 {
        return 0;
    }
    scale((v & mask) >> mask.trailing_zeros(), mask.count_ones())
}

#[cfg(all(
    test,
    feature = "tga",
    feature = "bmp",
    feature = "gif",
    feature = "tiff",
    feature = "psd",
    feature = "dds"
))]
mod against;

#[cfg(test)]
mod tests {
    use super::*;

    /// A PNG's pixels as RGBA.
    pub(super) fn rgba(png: &[u8]) -> (u32, u32, Vec<u8>) {
        let img = image::load_from_memory_with_format(png, image::ImageFormat::Png).unwrap();
        (img.width(), img.height(), img.to_rgba8().into_raw())
    }

    /// An image's pixels as RGBA, decoded through [`decode`].
    pub(super) fn decoded(bytes: &[u8], name: &str) -> Option<(u32, u32, Vec<u8>)> {
        let img = decode(bytes, Path::new(name))?;
        assert_eq!(img.mime, "image/png");
        Some(rgba(&img.bytes))
    }

    #[test]
    fn opaque_alpha_is_left_out() {
        let r = Raster {
            w: 2,
            h: 1,
            channels: 4,
            data: vec![1, 2, 3, 255, 4, 5, 6, 255],
        };
        let png = r.png().unwrap();
        // Color type 2: RGB.
        assert_eq!(png[25], 2);
        let r = Raster {
            w: 1,
            h: 1,
            channels: 2,
            data: vec![9, 128],
        };
        assert_eq!(r.png().unwrap()[25], 4);
        let r = Raster {
            w: 1,
            h: 1,
            channels: 1,
            data: vec![9],
        };
        assert_eq!(rgba(&r.png().unwrap()).2, [9, 9, 9, 255]);
    }

    #[cfg(any(feature = "bmp", feature = "dds", feature = "gif", feature = "psd", feature = "tga"))]
    #[test]
    fn short_data_is_padded_only_when_half_is_there() {
        let mut v = vec![1; 10];
        fill(&mut v, 4).unwrap();
        assert_eq!(v, [1; 4]);
        let mut v = vec![1; FREE];
        fill(&mut v, FREE * 2).unwrap();
        assert_eq!(v.len(), FREE * 2);
        let mut v = vec![1; FREE];
        assert!(fill(&mut v, FREE * 2 + 2).is_none());
        // Small images pad from nothing.
        let mut v = Vec::new();
        fill(&mut v, 64).unwrap();
        assert!(pixels(0, 5).is_none() && pixels(1 << 15, 1 << 14).is_none());
        assert_eq!(pixels(1 << 14, 1 << 14), Some(MAX_PIXELS));
    }

    #[cfg(any(feature = "tiff", feature = "psd"))]
    #[test]
    fn packbits_runs_and_literals() {
        let mut out = Vec::new();
        // 3 literals, a run of 4, a no-op, and a literal cut short by the end.
        packbits(&[2, b'a', b'b', b'c', 0xFD, b'z', 0x80, 5, b'q'], &mut out, 100);
        assert_eq!(out, b"abczzzzq");
        let mut out = Vec::new();
        packbits(&[0xFD, b'z', 2, b'a', b'b', b'c'], &mut out, 6);
        assert_eq!(out, b"zzzzab");
    }

    #[test]
    fn nothing_else_decodes() {
        for bytes in [
            &b""[..],
            b"BM",
            b"GIF89a",
            b"II*\0",
            b"8BPS",
            b"DDS ",
            b"\x89PNG\r\n\x1a\n",
        ] {
            assert!(decode(bytes, Path::new("x.png")).is_none());
            assert!(decode(bytes, Path::new("x.tga")).is_none());
        }
    }

    /// Every prefix and some bit flips of `bytes` decode or don't, without a panic.
    pub(super) fn survives_damage(bytes: &[u8], name: &str) {
        let name = Path::new(name);
        let step = (bytes.len() / 200).max(1);
        for cut in (0..bytes.len()).step_by(step) {
            let _ = decode(&bytes[..cut], name);
        }
        let mut rng = crate::test_rng::Pcg32::new(11);
        for _ in 0..300 {
            let mut b = bytes.to_vec();
            for _ in 0..4 {
                let i = usize_of(u64::from(rng.next_u32())) % b.len();
                b[i] ^= 1 << (rng.next_u32() % 8);
            }
            let _ = decode(&b, name);
        }
    }
}
