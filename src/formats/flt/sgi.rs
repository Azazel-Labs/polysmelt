//! SGI images (`.rgb`, `.rgba`, `.int`, `.inta`, `.bw`, `.sgi`), the textures `OpenFlight`
//! databases usually use, decoded and written as PNG so glTF can embed them.
//!
//! An SGI image is a 512-byte big-endian header, then each channel's rows, bottom row first:
//! stored as they are, or run-length encoded with a table of where each row starts.

/// The most pixels an image may have.
const MAX_PIXELS: usize = 1 << 26;

/// Whether `name` is an SGI image by its extension.
pub(super) fn is_sgi_name(name: &str) -> bool {
    let ext = name.rsplit('.').next().unwrap_or_default().to_ascii_lowercase();
    matches!(ext.as_str(), "rgb" | "rgba" | "int" | "inta" | "bw" | "sgi")
}

/// Whether `b` starts as an SGI image does.
pub(super) fn is_sgi(b: &[u8]) -> bool {
    b.len() >= 512 && b[0] == 0x01 && b[1] == 0xDA && b[2] <= 1 && matches!(b[3], 1 | 2)
}

/// An SGI image's width, height, channels (1 to 4) and 8-bit pixels, top row first.
pub(super) fn decode(b: &[u8]) -> Option<(u32, u32, usize, Vec<u8>)> {
    if !is_sgi(b) {
        return None;
    }
    let rle = b[2] == 1;
    let bpc = usize::from(b[3]);
    let be16 = |at: usize| usize::from(u16::from_be_bytes([b[at], b[at + 1]]));
    let dimension = be16(4);
    let w = be16(6);
    let mut h = be16(8);
    let mut zsize = be16(10);
    if dimension == 1 {
        h = 1;
    }
    if dimension <= 2 {
        zsize = 1;
    }
    let channels = zsize.min(4);
    if w == 0 || h == 0 || channels == 0 || w * h > MAX_PIXELS {
        return None;
    }
    let mut out = vec![0u8; w * h * channels];
    // Gray and alpha without the alpha would be gray; RGB without blue, black: fill the
    // channels a short image lacks as it would be read.
    let row_into = |out: &mut [u8], y: usize, c: usize, src: &mut dyn Iterator<Item = u8>| {
        // Bottom row first: row `y` of the file is row `h - 1 - y` of the image.
        let base = (h - 1 - y) * w * channels + c;
        for x in 0..w {
            out[base + x * channels] = src.next().unwrap_or(0);
        }
    };
    if rle {
        let rows = h.checked_mul(zsize)?;
        let table = 512usize.checked_add(rows.checked_mul(8)?)?;
        if table > b.len() {
            return None;
        }
        let be32 = |at: usize| u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;
        let mut buf: Vec<u8> = Vec::with_capacity(w);
        for c in 0..channels {
            for y in 0..h {
                let k = c * h + y;
                let start = be32(512 + k * 4);
                let len = be32(512 + rows * 4 + k * 4);
                let src = b.get(start..start.checked_add(len)?.min(b.len())).unwrap_or_default();
                buf.clear();
                unpack(src, bpc, w, &mut buf);
                row_into(&mut out, y, c, &mut buf.iter().copied());
            }
        }
    } else {
        let plane = w * h * bpc;
        for c in 0..channels {
            for y in 0..h {
                let at = 512 + c * plane + y * w * bpc;
                let row = b.get(at..(at + w * bpc).min(b.len())).unwrap_or_default();
                // 16-bit samples keep their high byte.
                row_into(&mut out, y, c, &mut row.iter().step_by(bpc).copied());
            }
        }
    }
    let w = u32::try_from(w).ok()?;
    let h = u32::try_from(h).ok()?;
    Some((w, h, channels, out))
}

/// One run-length encoded row into `out`, at most `w` samples, 8-bit.
fn unpack(src: &[u8], bpc: usize, w: usize, out: &mut Vec<u8>) {
    let mut at = 0;
    let sample = |at: usize| -> Option<u8> { src.get(at).copied() };
    while out.len() < w {
        let Some(head) = sample(at + bpc - 1) else { break };
        at += bpc;
        let count = usize::from(head & 0x7F);
        if count == 0 {
            break;
        }
        let count = count.min(w - out.len());
        if head & 0x80 != 0 {
            // A run of literal samples.
            for _ in 0..count {
                let Some(v) = sample(at) else { return };
                out.push(v);
                at += bpc;
            }
        } else {
            // One sample repeated.
            let Some(v) = sample(at) else { return };
            at += bpc;
            out.resize(out.len() + count, v);
        }
    }
}

/// 8-bit pixels as a PNG, compressed when the crate has a compressor.
#[cfg(any(
    feature = "webp",
    feature = "skp",
    feature = "tga",
    feature = "bmp",
    feature = "gif",
    feature = "tiff",
    feature = "psd",
    feature = "dds",
    feature = "amf"
))]
pub(super) fn png(w: u32, h: u32, channels: usize, pixels: &[u8]) -> Option<Vec<u8>> {
    crate::png::encode(w, h, channels, pixels)
}

/// 8-bit pixels as a PNG, its data in stored (uncompressed) deflate blocks.
#[cfg(not(any(
    feature = "webp",
    feature = "skp",
    feature = "tga",
    feature = "bmp",
    feature = "gif",
    feature = "tiff",
    feature = "psd",
    feature = "dds",
    feature = "amf"
)))]
pub(super) fn png(w: u32, h: u32, channels: usize, pixels: &[u8]) -> Option<Vec<u8>> {
    stored_png(w, h, channels, pixels)
}

/// 8-bit gray, gray and alpha, RGB or RGBA pixels as a PNG with stored deflate blocks.
#[cfg_attr(
    any(
        feature = "webp",
        feature = "skp",
        feature = "tga",
        feature = "bmp",
        feature = "gif",
        feature = "tiff",
        feature = "psd",
        feature = "dds",
        feature = "amf"
    ),
    allow(dead_code, reason = "the compressing encoder is used instead")
)]
pub(super) fn stored_png(w: u32, h: u32, channels: usize, pixels: &[u8]) -> Option<Vec<u8>> {
    let row = (w as usize).checked_mul(channels)?;
    if !(1..=4).contains(&channels) || row == 0 || pixels.len() != row.checked_mul(h as usize)? {
        return None;
    }
    // Each row behind its filter byte (none).
    let mut raw = Vec::with_capacity(pixels.len() + h as usize);
    for line in pixels.chunks_exact(row) {
        raw.push(0);
        raw.extend_from_slice(line);
    }
    let mut z = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 16);
    z.extend_from_slice(&[0x78, 0x01]);
    let mut blocks = raw.chunks(65535).peekable();
    if blocks.peek().is_none() {
        z.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
    }
    while let Some(block) = blocks.next() {
        let len = u16::try_from(block.len()).ok()?;
        z.push(u8::from(blocks.peek().is_none()));
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let color = [0u8, 4, 2, 6][channels - 1];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, color, 0, 0, 0]);
    let mut out = Vec::with_capacity(z.len() + 64);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    chunk(&mut out, *b"IHDR", &ihdr)?;
    chunk(&mut out, *b"IDAT", &z)?;
    chunk(&mut out, *b"IEND", &[])?;
    Some(out)
}

fn chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) -> Option<()> {
    out.extend_from_slice(&u32::try_from(data.len()).ok()?.to_be_bytes());
    let from = out.len();
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[from..]);
    out.extend_from_slice(&crc.to_be_bytes());
    Some(())
}

fn crc32(b: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (n, e) in table.iter_mut().enumerate() {
        let mut c = u32::try_from(n).unwrap_or(0);
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *e = c;
    }
    let mut c = !0u32;
    for &x in b {
        c = table[((c ^ u32::from(x)) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

fn adler32(b: &[u8]) -> u32 {
    let (mut a, mut s) = (1u32, 0u32);
    for chunk in b.chunks(5552) {
        for &x in chunk {
            a += u32::from(x);
            s += a;
        }
        a %= 65521;
        s %= 65521;
    }
    (s << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2 RGB image, stored or run-length encoded.
    fn image(rle: bool) -> Vec<u8> {
        let mut b = vec![0u8; 512];
        b[0..2].copy_from_slice(&474u16.to_be_bytes());
        b[2] = u8::from(rle);
        b[3] = 1;
        b[4..6].copy_from_slice(&3u16.to_be_bytes());
        b[6..8].copy_from_slice(&2u16.to_be_bytes());
        b[8..10].copy_from_slice(&2u16.to_be_bytes());
        b[10..12].copy_from_slice(&3u16.to_be_bytes());
        // Channel c, row y (bottom first): [10c + y, 10c + y + 100].
        let rows: Vec<[u8; 2]> = (0..3u8)
            .flat_map(|c| (0..2u8).map(move |y| [10 * c + y, 10 * c + y + 100]))
            .collect();
        if rle {
            let mut data = Vec::new();
            let mut starts = Vec::new();
            for r in &rows {
                starts.push(u32::try_from(512 + 48 + data.len()).unwrap());
                data.extend_from_slice(&[0x82, r[0], r[1], 0]);
            }
            for s in &starts {
                b.extend_from_slice(&s.to_be_bytes());
            }
            for _ in &starts {
                b.extend_from_slice(&4u32.to_be_bytes());
            }
            b.extend_from_slice(&data);
        } else {
            for r in &rows {
                b.extend_from_slice(r);
            }
        }
        b
    }

    #[test]
    fn stored_and_encoded_images_decode_alike_top_row_first() {
        for rle in [false, true] {
            let (w, h, c, px) = decode(&image(rle)).unwrap();
            assert_eq!((w, h, c), (2, 2, 3));
            // The top row is the file's second (y = 1).
            assert_eq!(&px[0..6], &[1, 11, 21, 101, 111, 121], "rle {rle}");
            assert_eq!(&px[6..12], &[0, 10, 20, 100, 110, 120], "rle {rle}");
            let (w, h, channels, px) = decode(&image(rle)).unwrap();
            let png = png(w, h, channels, &px).unwrap();
            assert!(png.starts_with(b"\x89PNG"));
        }
    }

    #[test]
    fn broken_images_never_panic() {
        let good = image(true);
        let mut rng = crate::test_rng::Pcg32::new(7);
        for n in 0..good.len() {
            let _ = decode(&good[..n]);
        }
        for _ in 0..2000 {
            let mut b = good.clone();
            let at = rng.below(u32::try_from(b.len()).unwrap()) as usize;
            b[at] = u8::try_from(rng.below(256)).unwrap();
            let _ = decode(&b);
        }
    }

    #[test]
    fn the_stored_png_is_a_valid_png() {
        let px: Vec<u8> = (0..=255u8).cycle().take(300 * 300 * 3).collect();
        let png = stored_png(300, 300, 3, &px).unwrap();
        assert!(crate::mesh::Image::png_or_jpeg(&png).is_some());
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }
}
