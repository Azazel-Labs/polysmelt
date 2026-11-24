//! WebP images re-encoded as PNG, for the mesh writer, which embeds only PNG and JPEG.

use std::io::Cursor;

/// `bytes`, a WebP image (an animation's first frame), as PNG bytes.
pub(crate) fn to_png(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut d = image_webp::WebPDecoder::new(Cursor::new(bytes)).ok()?;
    let (w, h) = d.dimensions();
    let channels = if d.has_alpha() { 4 } else { 3 };
    let mut pixels = vec![0; d.output_buffer_size()?];
    d.read_image(&mut pixels).ok()?;
    crate::png::encode(w, h, channels, &pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pixels a filter must predict: a gradient, noise and flat runs.
    fn pixels(w: u32, h: u32, channels: usize) -> Vec<u8> {
        let mut rng = crate::test_rng::Pcg32::new(7);
        let mut p = Vec::new();
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels {
                    p.push(match (x / 4 + y) % 3 {
                        0 => u8::try_from((x * 7 + y * 3 + u32::try_from(c).unwrap() * 50) % 256).unwrap(),
                        1 => rng.next_u32().to_le_bytes()[0],
                        _ => 200,
                    });
                }
            }
        }
        p
    }

    #[test]
    fn lossless_webp_comes_back_as_the_same_pixels() {
        for (channels, color) in [(4, image_webp::ColorType::Rgba8), (3, image_webp::ColorType::Rgb8)] {
            let (w, h) = (37, 23);
            let p = pixels(w, h, channels);
            let mut webp = Vec::new();
            image_webp::WebPEncoder::new(&mut webp).encode(&p, w, h, color).unwrap();
            let png = to_png(&webp).unwrap();
            let back = image::load_from_memory_with_format(&png, image::ImageFormat::Png).unwrap();
            assert_eq!((back.width(), back.height()), (w, h));
            let got = if channels == 4 {
                back.to_rgba8().into_raw()
            } else {
                back.to_rgb8().into_raw()
            };
            assert_eq!(got, p, "{channels} channels");
        }
    }

    #[test]
    fn not_webp_is_none() {
        assert!(to_png(b"\x89PNG\r\n\x1a\n").is_none());
        assert!(to_png(b"RIFF\0\0\0\0WEBP").is_none());
        assert!(crate::png::encode(0, 1, 3, &[]).is_none());
        assert!(crate::png::encode(2, 2, 3, &[0; 3]).is_none());
    }
}
