//! A PNG encoder for 8-bit gray, gray and alpha, RGB and RGBA pixels, for images decoded from formats glTF can't embed,
//! and a sampler that reads a few of a PNG's pixels without keeping the image.

use std::io::Write as _;

use flate2::{Compression, Crc, write::ZlibEncoder};
#[cfg(any(feature = "3ds", feature = "obj"))]
use flate2::{Decompress, FlushDecompress, Status};

use crate::num::usize_of;

/// 8-bit gray (`channels` 1), gray and alpha (2), RGB (3) or RGBA (4) pixels as a PNG, each
/// row filtered the way that leaves the smallest residuals.
pub(crate) fn encode(w: u32, h: u32, channels: usize, pixels: &[u8]) -> Option<Vec<u8>> {
    let row = usize_of(u64::from(w)).checked_mul(channels)?;
    if !(1..=4).contains(&channels) || row == 0 || pixels.len() != row.checked_mul(usize_of(u64::from(h)))? {
        return None;
    }
    let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
    let (mut prev, mut best, mut cand) = (vec![0; row], vec![0; row], vec![0; row]);
    for line in pixels.chunks_exact(row) {
        let mut best_kind = 0;
        let mut best_score = u64::MAX;
        for kind in 0..5 {
            filter(kind, channels, line, &prev, &mut cand);
            let score = score(&cand, best_score);
            if score < best_score {
                (best_kind, best_score) = (kind, score);
                std::mem::swap(&mut best, &mut cand);
            }
        }
        z.write_all(&[best_kind]).ok()?;
        z.write_all(&best).ok()?;
        prev.copy_from_slice(line);
    }
    let data = z.finish().ok()?;

    let mut out = Vec::with_capacity(data.len() + 64);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = [0; 13];
    ihdr[..4].copy_from_slice(&w.to_be_bytes());
    ihdr[4..8].copy_from_slice(&h.to_be_bytes());
    // 8 bits a channel, gray or truecolor, with or without alpha; deflate, adaptive filtering,
    // no interlacing.
    ihdr[8] = 8;
    ihdr[9] = [0, 4, 2, 6][channels - 1];
    chunk(&mut out, *b"IHDR", &ihdr)?;
    chunk(&mut out, *b"IDAT", &data)?;
    chunk(&mut out, *b"IEND", &[])?;
    Some(out)
}

/// A filtered row's bytes summed as signed magnitudes, the smallest sum being the best guess at
/// what compresses best. Counting stops at `limit`, the best so far, which this row can't beat.
fn score(row: &[u8], limit: u64) -> u64 {
    let mut total = 0;
    // Sums of 256 bytes fit in 16 bits, which vectorize twice as wide as 32.
    for c in row.chunks(256) {
        total += u64::from(
            c.iter()
                .map(|&b| u16::from(b.cast_signed().unsigned_abs()))
                .sum::<u16>(),
        );
        if total >= limit {
            break;
        }
    }
    total
}

/// One row through PNG filter `kind` (none, sub, up, average, Paeth) into `out`. A loop for
/// each filter, the first pixel apart: one loop testing both per byte cost most of an encode.
fn filter(kind: u8, bpp: usize, line: &[u8], prev: &[u8], out: &mut [u8]) {
    let k = bpp.min(line.len());
    let (head, tail) = out.split_at_mut(k);
    // The first pixel has no left neighbors: they count as 0.
    let rest = line[k..].iter().zip(&line[..line.len() - k]);
    match kind {
        0 => out.copy_from_slice(line),
        1 => {
            head.copy_from_slice(&line[..k]);
            for (o, (&x, &a)) in tail.iter_mut().zip(rest) {
                *o = x.wrapping_sub(a);
            }
        }
        2 => {
            for (o, (&x, &b)) in out.iter_mut().zip(line.iter().zip(prev)) {
                *o = x.wrapping_sub(b);
            }
        }
        3 => {
            for (o, (&x, &b)) in head.iter_mut().zip(line.iter().zip(prev)) {
                *o = x.wrapping_sub(b / 2);
            }
            for ((o, (&x, &a)), &b) in tail.iter_mut().zip(rest).zip(&prev[k..]) {
                *o = x.wrapping_sub(a.midpoint(b));
            }
        }
        _ => {
            for (o, (&x, &b)) in head.iter_mut().zip(line.iter().zip(prev)) {
                *o = x.wrapping_sub(paeth(0, b, 0));
            }
            for (((o, (&x, &a)), &b), &c) in tail.iter_mut().zip(rest).zip(&prev[k..]).zip(prev) {
                *o = x.wrapping_sub(paeth(a, b, c));
            }
        }
    }
}

/// Whichever of left, up and up-left is nearest `left + up - up_left`.
#[inline]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (i16::from(a), i16::from(b), i16::from(c));
    // `p - a` is `b - c`, and so on: one subtraction each, nothing to branch on until the pick.
    let (pa, pb, pc) = ((ib - ic).abs(), (ia - ic).abs(), (ia + ib - 2 * ic).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Samples a row and rows sampled: enough to tell what an image shows.
#[cfg(feature = "3ds")]
const SAMPLES: u32 = 64;
/// The longest row [`sample`] and [`decode`] read, in bytes: 65536 RGBA pixels of 16 bits.
#[cfg(any(feature = "3ds", feature = "obj"))]
const MAX_ROW: usize = 1 << 19;
/// How many bytes of rows [`sample`] decodes before it stops: the top of a huge image stands for all of it.
#[cfg(feature = "3ds")]
const SAMPLE_BYTES: usize = 1 << 24;
/// The most pixels [`decode`] makes: 8192 by 8192.
#[cfg(feature = "obj")]
const MAX_PIXELS: usize = 1 << 26;

/// A PNG's header and its compressed rows: gray, RGB, palette and alpha kinds of 8 or 16 bits, not interlaced.
#[cfg(any(feature = "3ds", feature = "obj"))]
struct Png<'a> {
    w: u32,
    rows: u32,
    kind: u8,
    channels: usize,
    /// Bytes a sample takes: 1 or 2.
    size: usize,
    palette: &'a [u8],
    data: Vec<u8>,
}

#[cfg(any(feature = "3ds", feature = "obj"))]
impl<'a> Png<'a> {
    /// The PNG in `bytes`, when it's a kind read here; `None` for the rest.
    fn parse(bytes: &'a [u8]) -> Option<Self> {
        let mut rest = bytes.strip_prefix(b"\x89PNG\r\n\x1a\n")?;
        let (mut ihdr, mut palette, mut data) = (None, &[][..], Vec::new());
        while let Some(head) = rest.get(..8) {
            let len = usize_of(u64::from(u32::from_be_bytes([head[0], head[1], head[2], head[3]])));
            // A chunk cut short gives what it has.
            let body = rest.get(8..).unwrap_or_default();
            let body = &body[..len.min(body.len())];
            match &head[4..8] {
                b"IHDR" => ihdr = Some(body),
                b"PLTE" => palette = body,
                b"IDAT" => data.extend_from_slice(body),
                b"IEND" => break,
                _ => {}
            }
            rest = rest.get(len.saturating_add(12)..).unwrap_or_default();
        }
        let h = ihdr.filter(|h| h.len() >= 13)?;
        let (w, rows) = (
            u32::from_be_bytes([h[0], h[1], h[2], h[3]]),
            u32::from_be_bytes([h[4], h[5], h[6], h[7]]),
        );
        let (depth, kind) = (h[8], h[9]);
        let channels = match kind {
            0 | 3 => 1,
            4 => 2,
            2 => 3,
            6 => 4,
            _ => return None,
        };
        if h[12] != 0 || !(depth == 8 || depth == 16 && kind != 3) || w == 0 || rows == 0 {
            return None;
        }
        Some(Self {
            w,
            rows,
            kind,
            channels,
            size: usize::from(depth / 8),
            palette,
            data,
        })
    }

    /// The bytes a row's pixels take, when that's no more than [`MAX_ROW`].
    fn row(&self) -> Option<usize> {
        usize_of(u64::from(self.w))
            .checked_mul(self.channels * self.size)
            .filter(|&r| r <= MAX_ROW)
    }

    /// Decodes the first `rows` rows one at a time, handing each to `f` with its number: only one is held. It stops
    /// early where the data ends; `None` when a row has a filter that isn't one or the data isn't zlib.
    fn each_row(&self, rows: usize, mut f: impl FnMut(usize, &[u8])) -> Option<()> {
        let row = self.row()?;
        let bpp = self.channels * self.size;
        let (mut line, mut prev) = (vec![0; row + 1], vec![0; row]);
        let mut z = Decompress::new(true);
        let (mut input, mut filled, mut y) = (&self.data[..], 0, 0);
        while y < rows {
            let (before_in, before_out) = (z.total_in(), z.total_out());
            let status = z.decompress(input, &mut line[filled..], FlushDecompress::None).ok()?;
            let used = usize_of(z.total_in() - before_in);
            let made = usize_of(z.total_out() - before_out);
            input = input.get(used..).unwrap_or_default();
            filled += made;
            if filled == line.len() {
                let (filter, pixels) = line.split_first_mut()?;
                unfilter(*filter, bpp, pixels, &prev)?;
                f(y, pixels);
                prev.copy_from_slice(pixels);
                filled = 0;
                y += 1;
            } else if status == Status::StreamEnd || used == 0 && made == 0 {
                break;
            }
        }
        Some(())
    }

    /// A pixel's bytes as RGB: gray in every channel, a palette index looked up (black past the palette), 16-bit
    /// samples by their high bytes, alpha left out.
    fn rgb(&self, p: &[u8]) -> [u8; 3] {
        let c = |i: usize| p[i * self.size];
        match self.channels {
            1 if self.kind == 3 => {
                let i = usize::from(c(0)) * 3;
                self.palette
                    .get(i..i + 3)
                    .map_or([0; 3], |rgb| [rgb[0], rgb[1], rgb[2]])
            }
            1 | 2 => [c(0); 3],
            _ => [c(0), c(1), c(2)],
        }
    }
}

/// Pixels spread over a PNG, as RGB: up to [`SAMPLES`] from each of up to as many rows. Gray, RGB, palette and alpha
/// kinds of 8 or 16 bits (high bytes kept), not interlaced; `None` for the rest, or when no row decodes. The rows
/// decode one at a time, so only one is held, and past [`SAMPLE_BYTES`] of them the rest is left.
#[cfg(feature = "3ds")]
pub(crate) fn sample(bytes: &[u8]) -> Option<Vec<[u8; 3]>> {
    let png = Png::parse(bytes)?;
    let row = png.row()?;
    let rows = usize_of(u64::from(png.rows)).min((SAMPLE_BYTES / row).max(1));
    let (y_step, x_step) = (
        rows.div_ceil(SAMPLES as usize),
        usize_of(u64::from(png.w.div_ceil(SAMPLES))),
    );
    let bpp = png.channels * png.size;
    let mut out = Vec::new();
    png.each_row(rows, |y, pixels| {
        if y.is_multiple_of(y_step) {
            out.extend(pixels.chunks_exact(bpp).step_by(x_step).map(|p| png.rgb(p)));
        }
    })?;
    (!out.is_empty()).then_some(out)
}

/// A value for each pixel of a PNG, rows top to bottom, from each pixel's RGB by `pick`: the image's width, height
/// and values. The kinds [`sample`] reads; `None` for the rest, for an image of more than [`MAX_PIXELS`], or when its
/// rows don't all decode. Room is taken as rows decode, not from the size the header states.
#[cfg(feature = "obj")]
pub(crate) fn decode<T>(bytes: &[u8], pick: impl Fn([u8; 3]) -> T) -> Option<(u32, u32, Vec<T>)> {
    let png = Png::parse(bytes)?;
    let (w, h) = (usize_of(u64::from(png.w)), usize_of(u64::from(png.rows)));
    if w.checked_mul(h)? > MAX_PIXELS {
        return None;
    }
    let bpp = png.channels * png.size;
    let mut out = Vec::new();
    png.each_row(h, |_, pixels| {
        out.extend(pixels.chunks_exact(bpp).map(|p| pick(png.rgb(p))));
    })?;
    (out.len() == w * h).then_some((png.w, png.rows, out))
}

/// A row through the inverse of PNG filter `kind`, in place, given the row above.
#[cfg(any(feature = "3ds", feature = "obj"))]
fn unfilter(kind: u8, bpp: usize, line: &mut [u8], prev: &[u8]) -> Option<()> {
    if kind > 4 {
        return None;
    }
    for x in 0..line.len() {
        let a = if x >= bpp { line[x - bpp] } else { 0 };
        let b = prev[x];
        let c = if x >= bpp { prev[x - bpp] } else { 0 };
        let predict = match kind {
            0 => 0,
            1 => a,
            2 => b,
            3 => a.midpoint(b),
            _ => paeth(a, b, c),
        };
        line[x] = line[x].wrapping_add(predict);
    }
    Some(())
}

fn chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) -> Option<()> {
    out.extend_from_slice(&u32::try_from(data.len()).ok()?.to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    let mut crc = Crc::new();
    crc.update(&kind);
    crc.update(data);
    out.extend_from_slice(&crc.sum().to_be_bytes());
    Some(())
}

#[cfg(all(test, feature = "3ds"))]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::cast_possible_truncation, reason = "wrapping test patterns")]
    fn sample_reads_back_what_encode_wrote() {
        let (w, h) = (100u32, 70u32);
        let px = |x: u32, y: u32| [x as u8, y as u8, (x * 3 + y * 7) as u8];
        for channels in [1, 2, 3, 4] {
            let mut data = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y);
                    data.extend(match channels {
                        1 => vec![p[0]],
                        2 => vec![p[0], 9],
                        3 => p.to_vec(),
                        _ => vec![p[0], p[1], p[2], 9],
                    });
                }
            }
            let png = encode(w, h, channels, &data).unwrap();
            let got = sample(&png).unwrap();
            // Every other row and column: 35 rows of 50.
            assert_eq!(got.len(), 35 * 50);
            for (i, s) in got.iter().enumerate() {
                let (x, y) = (u32::try_from(i % 50).unwrap() * 2, u32::try_from(i / 50).unwrap() * 2);
                let p = px(x, y);
                assert_eq!(
                    *s,
                    if channels < 3 { [p[0]; 3] } else { p },
                    "{channels} channels at {x}, {y}"
                );
            }
            // Cut short or damaged: whatever rows decode, never a panic.
            for len in 0..png.len() {
                let _ = sample(&png[..len]);
            }
            let mut bad = png.clone();
            for i in (8..bad.len()).step_by(3) {
                bad[i] = bad[i].wrapping_add(77);
            }
            let _ = sample(&bad);
        }
    }

    #[test]
    fn sample_reads_16_bits_and_palettes_and_leaves_interlacing() {
        let mut ihdr = [0; 13];
        ihdr[..4].copy_from_slice(&2u32.to_be_bytes());
        ihdr[4..8].copy_from_slice(&1u32.to_be_bytes());
        let png = |ihdr: &[u8], plte: &[u8], row: &[u8]| {
            let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
            z.write_all(row).unwrap();
            let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
            chunk(&mut out, *b"IHDR", ihdr).unwrap();
            if !plte.is_empty() {
                chunk(&mut out, *b"PLTE", plte).unwrap();
            }
            chunk(&mut out, *b"IDAT", &z.finish().unwrap()).unwrap();
            chunk(&mut out, *b"IEND", &[]).unwrap();
            out
        };
        // 16-bit RGB: the high bytes.
        ihdr[8] = 16;
        ihdr[9] = 2;
        let wide = png(&ihdr, &[], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(sample(&wide), Some(vec![[1, 3, 5], [7, 9, 11]]));
        // A palette, and an index past it as black.
        ihdr[8] = 8;
        ihdr[9] = 3;
        let indexed = png(&ihdr, &[10, 20, 30, 40, 50, 60], &[0, 1, 7]);
        assert_eq!(sample(&indexed), Some(vec![[40, 50, 60], [0, 0, 0]]));
        // Interlaced, 16-bit palettes and unknown filters: none.
        ihdr[12] = 1;
        assert_eq!(sample(&png(&ihdr, &[0; 6], &[0, 0, 0])), None);
        ihdr[12] = 0;
        ihdr[8] = 16;
        assert_eq!(sample(&png(&ihdr, &[0; 6], &[0, 0, 0, 0, 0])), None);
        ihdr[8] = 8;
        assert_eq!(sample(&png(&ihdr, &[0; 6], &[5, 0, 0])), None);
        // A row past the limit, whatever the data: none, and nothing allocated for it.
        ihdr[..4].copy_from_slice(&u32::MAX.to_be_bytes());
        ihdr[9] = 6;
        assert_eq!(sample(&png(&ihdr, &[], &[0; 9])), None);
    }
}

#[cfg(all(test, feature = "obj"))]
mod decode_tests {
    use super::*;

    #[test]
    #[allow(clippy::cast_possible_truncation, reason = "wrapping test patterns")]
    fn decode_reads_back_every_pixel_encode_wrote() {
        let (w, h) = (37u32, 21u32);
        let px = |x: u32, y: u32| [x as u8, (y * 5) as u8, (x * 3 + y * 7) as u8];
        for channels in [1, 2, 3, 4] {
            let mut data = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y);
                    data.extend(match channels {
                        1 => vec![p[0]],
                        2 => vec![p[0], 9],
                        3 => p.to_vec(),
                        _ => vec![p[0], p[1], p[2], 9],
                    });
                }
            }
            let png = encode(w, h, channels, &data).unwrap();
            let (gw, gh, got) = decode(&png, |rgb| rgb).unwrap();
            assert_eq!((gw, gh, got.len()), (w, h, (w * h) as usize));
            for (i, s) in got.iter().enumerate() {
                let (x, y) = (u32::try_from(i).unwrap() % w, u32::try_from(i).unwrap() / w);
                let p = px(x, y);
                assert_eq!(
                    *s,
                    if channels < 3 { [p[0]; 3] } else { p },
                    "{channels} channels at {x}, {y}"
                );
            }
            // One channel picked.
            let (_, _, blue) = decode(&png, |rgb| rgb[2]).unwrap();
            assert_eq!(blue[w as usize + 1], if channels < 3 { 1 } else { px(1, 1)[2] });
            // Cut short: none once rows are missing, never a panic.
            for len in 0..png.len() {
                let _ = decode(&png[..len], |rgb| rgb);
            }
            assert_eq!(decode(&png[..png.len() / 2], |rgb| rgb), None);
        }
    }
}
