//! Raw deflate streams (RFC 1951), inflated without a dependency: the compressed `.x` files' MSZIP blocks and
//! zipped XGL (`.zgl`).
//!
//! Codes are decoded a bit at a time, canonically, which keeps the reader small; it checks every length,
//! distance and code, so damaged data is an error rather than a panic.

use crate::{Context, Result};

/// Reads a deflate stream's bits, least significant first.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
    buf: usize,
    count: u32,
}

impl Bits<'_> {
    /// The next `n` bits (at most 16).
    fn get(&mut self, n: u32) -> Result<usize> {
        while self.count < n {
            let byte = *self.data.get(self.at).context("compressed data cut short")?;
            self.at += 1;
            self.buf |= usize::from(byte) << self.count;
            self.count += 8;
        }
        let v = self.buf & ((1usize << n) - 1);
        self.buf >>= n;
        self.count -= n;
        Ok(v)
    }

    /// The next bit, as an integer.
    fn bit(&mut self) -> Result<i32> {
        Ok(i32::from(self.get(1)? != 0))
    }
}

/// A canonical Huffman code: how many codes there are of each length, and the symbols in code order.
struct Huff {
    count: [u16; 16],
    symbols: Vec<u16>,
}

impl Huff {
    /// The code for these code lengths (each below 16); an over-subscribed set is an error.
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[usize::from(l)] += 1;
        }
        let mut left = 1i32;
        for &c in &count[1..] {
            left = (left << 1) - i32::from(c);
            ensure!(left >= 0, "bad Huffman code in compressed data");
        }
        let mut offsets = [0u16; 16];
        for len in 1..15 {
            offsets[len + 1] = offsets[len] + count[len];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (s, &l) in (0u16..).zip(lengths) {
            if l != 0 {
                let o = &mut offsets[usize::from(l)];
                symbols[usize::from(*o)] = s;
                *o += 1;
            }
        }
        Ok(Self { count, symbols })
    }

    /// Decodes one symbol.
    fn decode(&self, bits: &mut Bits<'_>) -> Result<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for &c in &self.count[1..] {
            code |= bits.bit()?;
            let count = i32::from(c);
            if code - count < first {
                let i = usize::try_from(index + code - first).ok();
                return i
                    .and_then(|i| self.symbols.get(i))
                    .copied()
                    .context("bad Huffman code in compressed data");
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        bail!("bad Huffman code in compressed data")
    }
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];
/// The order code length code lengths come in.
const CODE_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// Inflates the raw deflate stream `data` onto `out`, adding at most `limit` bytes; what `out` already holds is the
/// window a match may reach back into.
pub(crate) fn inflate(data: &[u8], out: &mut Vec<u8>, limit: usize) -> Result<()> {
    let end = out.len().saturating_add(limit);
    let mut bits = Bits {
        data,
        at: 0,
        buf: 0,
        count: 0,
    };
    loop {
        let last = bits.get(1)? == 1;
        match bits.get(2)? {
            0 => {
                // Stored: aligned to a byte, a length and its complement, then the bytes.
                bits.buf = 0;
                bits.count = 0;
                let at = bits.at;
                let head = data.get(at..at + 4).context("compressed data cut short")?;
                let len = u16::from_le_bytes([head[0], head[1]]);
                let not = u16::from_le_bytes([head[2], head[3]]);
                ensure!(len == !not, "bad stored block in compressed data");
                let len = usize::from(len);
                ensure!(len <= end - out.len(), "compressed data inflates past its size");
                out.extend_from_slice(data.get(at + 4..at + 4 + len).context("compressed data cut short")?);
                bits.at = at + 4 + len;
            }
            1 => {
                let mut lengths = [0u8; 288];
                lengths[..144].fill(8);
                lengths[144..256].fill(9);
                lengths[256..280].fill(7);
                lengths[280..].fill(8);
                let lit = Huff::new(&lengths)?;
                let dist = Huff::new(&[5; 30])?;
                codes(&mut bits, out, end, &lit, &dist)?;
            }
            2 => {
                let (lit, dist) = dynamic(&mut bits)?;
                codes(&mut bits, out, end, &lit, &dist)?;
            }
            _ => bail!("bad block type in compressed data"),
        }
        if last || (out.len() >= end && bits.at >= data.len()) {
            return Ok(());
        }
    }
}

/// Reads a dynamic block's literal and distance codes.
fn dynamic(bits: &mut Bits<'_>) -> Result<(Huff, Huff)> {
    let nlen = bits.get(5)? + 257;
    let ndist = bits.get(5)? + 1;
    let ncode = bits.get(4)? + 4;
    ensure!(nlen <= 286 && ndist <= 30, "bad dynamic block in compressed data");
    let mut lengths = [0u8; 320];
    for &k in &CODE_ORDER[..ncode] {
        lengths[k] = u8::try_from(bits.get(3)?).unwrap_or(0);
    }
    let lencode = Huff::new(&lengths[..19])?;
    lengths = [0u8; 320];
    let mut at = 0;
    while at < nlen + ndist {
        let sym = lencode.decode(bits)?;
        if sym < 16 {
            lengths[at] = u8::try_from(sym).unwrap_or(0);
            at += 1;
            continue;
        }
        let (value, repeat) = match sym {
            16 => {
                ensure!(at > 0, "bad dynamic block in compressed data");
                (lengths[at - 1], 3 + bits.get(2)?)
            }
            17 => (0, 3 + bits.get(3)?),
            _ => (0, 11 + bits.get(7)?),
        };
        ensure!(at + repeat <= nlen + ndist, "bad dynamic block in compressed data");
        lengths[at..at + repeat].fill(value);
        at += repeat;
    }
    ensure!(lengths[256] != 0, "bad dynamic block in compressed data (no end code)");
    Ok((Huff::new(&lengths[..nlen])?, Huff::new(&lengths[nlen..nlen + ndist])?))
}

/// Decodes a compressed block's literals and matches until its end code, up to `end` bytes in all.
fn codes(bits: &mut Bits<'_>, out: &mut Vec<u8>, end: usize, lit: &Huff, dist: &Huff) -> Result<()> {
    loop {
        let sym = lit.decode(bits)?;
        if let Ok(byte) = u8::try_from(sym) {
            ensure!(out.len() < end, "compressed data inflates past its size");
            out.push(byte);
            continue;
        }
        if sym == 256 {
            return Ok(());
        }
        let i = usize::from(sym - 257);
        ensure!(i < 29, "bad length code in compressed data");
        let len = usize::from(LENGTH_BASE[i]) + bits.get(u32::from(LENGTH_EXTRA[i]))?;
        let d = usize::from(dist.decode(bits)?);
        ensure!(d < 30, "bad distance code in compressed data");
        let back = usize::from(DIST_BASE[d]) + bits.get(u32::from(DIST_EXTRA[d]))?;
        ensure!(back <= out.len(), "compressed data reaches back too far");
        ensure!(len <= end - out.len(), "compressed data inflates past its size");
        let from = out.len() - back;
        for k in 0..len {
            let byte = out[from + k];
            out.push(byte);
        }
    }
}
