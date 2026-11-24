//! Reading a model a block at a time, so a reader never holds the whole file, and parsing
//! numbers out of its text.

use std::io::{self, Read};

/// How much of the file is in memory at once, unless a single line or token is longer.
#[cfg(feature = "fs")]
const BLOCK: usize = 256 * 1024;

/// A model read through a buffer of one block, or bytes already in memory read in place.
/// `fill` tops the buffer up and returns the bytes not yet consumed; `consume` marks some of
/// them done.
pub(crate) struct Blocks<'a> {
    /// The stream; `None` when the bytes are all in `slice`.
    source: Option<Box<dyn Read + 'a>>,
    /// The whole model, when it was in memory to begin with.
    slice: &'a [u8],
    buf: Vec<u8>,
    /// Consumed bytes at the front of the view.
    pos: usize,
    /// Filled bytes in the view.
    len: usize,
    eof: bool,
    /// The model's size, as far as is known when it was opened.
    size: u64,
}

impl<'a> Blocks<'a> {
    /// Reads `source`, whose size is about `size` bytes.
    #[cfg(feature = "fs")]
    pub fn reader(source: impl Read + 'a, size: u64) -> Self {
        let cap = usize::try_from(size).unwrap_or(BLOCK).clamp(64, BLOCK);
        Self {
            source: Some(Box::new(source)),
            slice: &[],
            buf: vec![0; cap],
            pos: 0,
            len: 0,
            eof: false,
            size,
        }
    }

    /// Bytes already in memory, read where they are.
    pub fn slice(bytes: &'a [u8]) -> Self {
        Self {
            source: None,
            slice: bytes,
            buf: Vec::new(),
            pos: 0,
            len: bytes.len(),
            eof: true,
            size: bytes.len() as u64,
        }
    }

    /// The buffered bytes, consumed or not.
    #[inline]
    fn view(&self) -> &[u8] {
        if self.source.is_some() { &self.buf } else { self.slice }
    }

    /// The unconsumed bytes, after reading as much as fits. Empty only at the end of the file.
    pub fn fill(&mut self) -> io::Result<&[u8]> {
        if let Some(source) = &mut self.source {
            if self.pos > 0 {
                self.buf.copy_within(self.pos..self.len, 0);
                self.len -= self.pos;
                self.pos = 0;
            }
            while self.len < self.buf.len() && !self.eof {
                match source.read(&mut self.buf[self.len..]) {
                    Ok(0) => self.eof = true,
                    Ok(n) => self.len += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(self.data())
    }

    /// The file's size, for estimating what it holds.
    pub fn size_hint(&self) -> usize {
        usize::try_from(self.size).unwrap_or(usize::MAX)
    }

    /// The unconsumed bytes, as they are.
    pub fn data(&self) -> &[u8] {
        &self.view()[self.pos..self.len]
    }

    pub fn consume(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.len);
    }

    /// Whether the whole file has been read into the buffer (the unconsumed bytes are all that
    /// is left).
    pub fn at_end(&self) -> bool {
        self.eof
    }

    /// Room for more than a block: for a line or record longer than the buffer.
    fn grow(&mut self) {
        let n = self.buf.len() * 2;
        self.buf.resize(n, 0);
    }

    /// At least `n` unconsumed bytes, or fewer at the end of the file.
    pub fn need(&mut self, n: usize) -> io::Result<&[u8]> {
        if self.len - self.pos < n && !self.eof {
            while self.buf.len() < n {
                self.grow();
            }
            self.fill()?;
        }
        Ok(self.data())
    }

    /// The next stretch of text, cut where `cut` says (the length of a prefix that ends on a
    /// line or token boundary), and consumed. The rest of the file once it has all been read;
    /// `None` after that.
    pub fn text(&mut self, cut: impl Fn(&[u8]) -> Option<usize>) -> io::Result<Option<&[u8]>> {
        loop {
            self.fill()?;
            let (start, len) = (self.pos, self.len);
            if len == start {
                return Ok(None);
            }
            let n = if self.eof {
                len - start
            } else if let Some(n) = cut(&self.view()[start..len]).filter(|&n| n > 0) {
                n
            } else {
                // One line fills the buffer: make room for more of it.
                self.grow();
                continue;
            };
            self.pos = start + n;
            return Ok(Some(&self.view()[start..start + n]));
        }
    }
}

/// Text up to and including the last line break.
#[cfg(any(feature = "bvh", all(test, feature = "fs")))]
pub(crate) fn lines(data: &[u8]) -> Option<usize> {
    data.iter().rposition(|&b| b == b'\n').map(|i| i + 1)
}

/// Text up to and including the last byte that separates tokens.
pub(crate) fn words(data: &[u8]) -> Option<usize> {
    data.iter().rposition(|&b| space(b)).map(|i| i + 1)
}

/// Whether `b` separates tokens: ASCII whitespace, or any other control character.
#[inline]
pub(crate) fn space(b: u8) -> bool {
    b <= b' '
}

/// A plain decimal number, `[+-]digits[.digits][(e|E)[+-]digits]`, as its sign, its digits as
/// an integer and the power of ten that scales them. `None` for anything else, or more digits
/// than a `u64` holds.
#[inline]
fn decimal(s: &[u8]) -> Option<(bool, u64, i32)> {
    let (neg, m, e, len) = decimal_lead(s)?;
    (len == s.len()).then_some((neg, m, e))
}

/// The run of decimal digits `s` starts with: its value (meaningless past 19 digits) and its
/// length. Eight bytes are read at once where there are eight.
#[inline]
fn leading_digits(s: &[u8]) -> (u64, usize) {
    let (mut v, mut n) = s.first_chunk::<8>().map_or((0, 0), |w| eight_digits(*w));
    if n < 8 && s.len() >= 8 {
        return (v, n);
    }
    while let Some(&c) = s.get(n) {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            break;
        }
        v = v.wrapping_mul(10).wrapping_add(u64::from(d));
        n += 1;
    }
    (v, n)
}

/// The run of decimal digits `s` starts with, a digit at a time: quicker for runs of a few, like
/// a number's whole part or a face's index.
#[inline]
pub(crate) fn few_digits(s: &[u8]) -> (u64, usize) {
    let mut v = 0u64;
    let mut n = 0;
    while let Some(&c) = s.get(n) {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            break;
        }
        v = v.wrapping_mul(10).wrapping_add(u64::from(d));
        n += 1;
    }
    (v, n)
}

/// The value of the digits that start `w` and how many there are, from all eight bytes at once:
/// each byte's distance from `'0'` marks the first that isn't a digit, and three multiplies sum
/// the digits before it.
#[inline]
fn eight_digits(w: [u8; 8]) -> (u64, usize) {
    const LOW: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let a = u64::from_le_bytes(w) ^ 0x3030_3030_3030_3030;
    // A byte's top bit is set unless it was a digit: 10 or more (or top bit already set).
    let over = (((a & LOW) + 0x7676_7676_7676_7676) | a) & HIGH;
    let n = (over.trailing_zeros() / 8) as usize;
    if n == 0 {
        return (0, 0);
    }
    // The digits to the top, after zeros as leading digits; then pairs, fours and all eight.
    let v = a << (8 * (8 - n));
    let v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let pairs = 0x0000_00FF_0000_00FF;
    let v = (v & pairs)
        .wrapping_mul(100 + (1_000_000 << 32))
        .wrapping_add(((v >> 16) & pairs).wrapping_mul(1 + (10_000 << 32)))
        >> 32;
    (v, n)
}

const POW10_U64: [u64; 20] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
    100_000_000_000_000,
    1_000_000_000_000_000,
    10_000_000_000_000_000,
    100_000_000_000_000_000,
    1_000_000_000_000_000_000,
    10_000_000_000_000_000_000,
];

/// The plain decimal number `s` starts with, like [`decimal`], and how many bytes it takes.
#[inline]
fn decimal_lead(s: &[u8]) -> Option<(bool, u64, i32, usize)> {
    let mut i = 0;
    let neg = match s.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    // Nineteen digits always fit in a u64; more are left to the standard parser. The whole part
    // is usually short and the fraction long.
    let (mut m, mut digits) = few_digits(&s[i..]);
    i += digits;
    let mut exp: i32 = 0;
    if s.get(i) == Some(&b'.') {
        i += 1;
        let (f, n) = leading_digits(&s[i..]);
        if n > 0 {
            if digits + n > 19 {
                return None;
            }
            m = m * POW10_U64[n] + f;
            digits += n;
            i += n;
            exp = -i32::try_from(n).ok()?;
        }
    }
    if digits == 0 || digits > 19 {
        return None;
    }
    if let Some(b'e' | b'E') = s.get(i) {
        i += 1;
        let eneg = match s.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let start = i;
        let mut e: i32 = 0;
        while let Some(&c) = s.get(i) {
            let d = c.wrapping_sub(b'0');
            if d > 9 || i - start >= 6 {
                break;
            }
            e = e * 10 + i32::from(d);
            i += 1;
        }
        if i == start {
            return None;
        }
        exp += if eneg { -e } else { e };
    }
    Some((neg, m, exp, i))
}

const POW10_F32: [f32; 11] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10];
const POW10_F64: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20,
    1e21, 1e22,
];

/// A decimal's value as an `f32`, when it can be had exactly rounded without the standard
/// parser.
#[inline]
fn f32_quick(neg: bool, m: u64, e: i32) -> Option<f32> {
    // Both the digits and the power of ten are exact in an f32, so one multiply or divide
    // rounds correctly (Clinger's fast path).
    if m <= 1 << 24 && (-10..=10).contains(&e) {
        #[allow(clippy::cast_precision_loss, reason = "checked to be at most 2^24, exact in an f32")]
        let v = m as f32;
        let p = POW10_F32[e.unsigned_abs() as usize];
        let v = if e < 0 { v / p } else { v * p };
        return Some(if neg { -v } else { v });
    }
    // Longer digits (nine, as `%.9g` writes): the same in an f64, then rounded to an f32. That
    // second rounding goes the same way as rounding the exact value, since every point halfway
    // between two f32s is an f64 too, unless the f64 landed on one; those go to the standard
    // parser. The values here (1e-22 to 1e38) are all normal f32s.
    if m <= 1 << 53 && (-22..=22).contains(&e) {
        #[allow(clippy::cast_precision_loss, reason = "checked to be at most 2^53, exact in an f64")]
        let v = m as f64;
        let p = POW10_F64[e.unsigned_abs() as usize];
        let v = if e < 0 { v / p } else { v * p };
        if v.to_bits() & 0x1FFF_FFFF != 0x1000_0000 {
            #[allow(clippy::cast_possible_truncation, reason = "rounded to the nearest f32 on purpose")]
            let v = v as f32;
            return Some(if neg { -v } else { v });
        }
    }
    None
}

/// `s` as an `f32`, rounded exactly as `str::parse` rounds it. Decimals of up to 15 digits or so
/// with small exponents (what exporters write) take a fast path; everything else goes to the
/// standard parser.
#[inline]
pub(crate) fn f32_from(s: &[u8]) -> Option<f32> {
    if let Some((neg, m, e)) = decimal(s)
        && let Some(v) = f32_quick(neg, m, e)
    {
        return Some(v);
    }
    std::str::from_utf8(s).ok()?.parse().ok()
}

/// The number `s` starts with, as an `f32` like [`f32_from`], and how many bytes it takes; `None`
/// unless it ends at a space, a control character or the end of `s`.
#[inline]
pub(crate) fn f32_lead(s: &[u8]) -> Option<(f32, usize)> {
    let (neg, m, e, len) = decimal_lead(s)?;
    if s.get(len).is_some_and(|&b| !space(b)) {
        return None;
    }
    if let Some(v) = f32_quick(neg, m, e) {
        return Some((v, len));
    }
    Some((std::str::from_utf8(&s[..len]).ok()?.parse().ok()?, len))
}

/// `s` as an `f64`, like [`f32_from`].
#[inline]
pub(crate) fn f64_from(s: &[u8]) -> Option<f64> {
    if let Some((neg, m, e)) = decimal(s)
        && m <= 1 << 53
        && (-22..=22).contains(&e)
    {
        #[allow(clippy::cast_precision_loss, reason = "checked to be at most 2^53, exact in an f64")]
        let v = m as f64;
        let p = POW10_F64[e.unsigned_abs() as usize];
        let v = if e < 0 { v / p } else { v * p };
        return Some(if neg { -v } else { v });
    }
    std::str::from_utf8(s).ok()?.parse().ok()
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;

    #[test]
    fn numbers_round_like_the_standard_parser() {
        let cases = [
            "0",
            "-0",
            "+1",
            "1.",
            ".5",
            "-.5",
            "1e5",
            "1E-5",
            "1.5e+3",
            "-1.234567e-01",
            "0.1",
            "0.2",
            "0.3",
            "3.4028235e38",
            "1e-45",
            "1e39",
            "16777217",
            "123.456789",
            "0.000001",
            "9.999999",
            "1.17549435e-38",
            "inf",
            "-infinity",
            "NaN",
            "1e",
            "e5",
            "--1",
            "1.2.3",
            "",
            ".",
            "+",
            "0x10",
            "1_0",
            "99999999999999999999999",
            "7e-11",
            "1234567.0",
            "0.12345678",
            "-12.345678",
        ];
        for s in cases {
            let want32 = s.parse::<f32>().ok();
            let got32 = f32_from(s.as_bytes());
            assert_eq!(
                got32.map(f32::to_bits),
                want32.map(f32::to_bits),
                "f32 {s:?}: {got32:?} vs {want32:?}"
            );
            let want64 = s.parse::<f64>().ok();
            let got64 = f64_from(s.as_bytes());
            assert_eq!(got64.map(f64::to_bits), want64.map(f64::to_bits), "f64 {s:?}");
        }
        // Every short decimal the fast path takes, against the standard parser.
        let mut x: u32 = 1;
        for _ in 0..200_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let m = x % 16_777_216;
            let e = (x >> 24) % 21;
            let s = format!("{m}e-{e}");
            assert_eq!(
                f32_from(s.as_bytes()).map(f32::to_bits),
                s.parse::<f32>().ok().map(f32::to_bits),
                "{s}"
            );
            let s = format!("-{m}.{e}e{e}");
            assert_eq!(
                f32_from(s.as_bytes()).map(f32::to_bits),
                s.parse::<f32>().ok().map(f32::to_bits),
                "{s}"
            );
        }
    }

    #[test]
    fn digit_runs_read_eight_at_a_time_like_one_at_a_time() {
        let slow = |s: &[u8]| {
            let n = s.iter().take_while(|c| c.is_ascii_digit()).count();
            let v = s[..n]
                .iter()
                .fold(0u64, |v, &c| v.wrapping_mul(10).wrapping_add(u64::from(c - b'0')));
            (v, n)
        };
        let mut rng = crate::test_rng::Pcg32::new(11);
        let stops = b" /\t\n\r.e-+:\x00\xff\x2f\x3a";
        for _ in 0..200_000 {
            let len = rng.below(26) as usize;
            let mut s: Vec<u8> = (0..len).map(|_| b"0123456789"[rng.below(10) as usize]).collect();
            if rng.below(4) > 0 {
                s.push(stops[rng.below(u32::try_from(stops.len()).unwrap()) as usize]);
            }
            s.extend(b"123456789");
            let cut = rng.below(u32::try_from(s.len()).unwrap() + 1) as usize;
            for t in [&s[..], &s[..cut]] {
                let (v, n) = leading_digits(t);
                let (w, m) = slow(t);
                assert_eq!(n, m, "{:?}", String::from_utf8_lossy(t));
                if n <= 19 {
                    assert_eq!(v, w, "{:?}", String::from_utf8_lossy(t));
                }
            }
        }
    }

    #[test]
    fn long_decimals_round_like_the_standard_parser() {
        let check = |s: &str| {
            let want = s.parse::<f32>().ok().map(f32::to_bits);
            assert_eq!(f32_from(s.as_bytes()).map(f32::to_bits), want, "{s}");
            // Read from the front of a line, ended by a space, a line break or nothing; anything
            // but a plain decimal (`inf`, more digits than a u64 holds) is left to the line's own
            // reader.
            let plain = decimal(s.as_bytes()).is_some();
            for end in ["", " ", "\t", "\r\n", "\n"] {
                let t = format!("{s}{end}");
                let got = f32_lead(t.as_bytes());
                if !plain && got.is_none() {
                    continue;
                }
                assert_eq!(got.map(|g| g.0.to_bits()), want, "{t:?}");
                if let Some((_, len)) = got {
                    assert_eq!(len, s.len(), "{t:?}");
                }
            }
        };
        for s in [
            "-2.19910035e-10",
            "1.23456789",
            "-0",
            "-0.0e5",
            "0e-30",
            // Exactly halfway between two f32s: ties go to even.
            "16777217",
            "16777219",
            "33554434",
            "0.5000000298023223876953125",
            "1.00000005960464477539062500",
            "3.40282356779733661637539395458142568448e38",
            "3.4028236e38",
            "1e22",
            "1e-22",
            "123456789012345e-22",
            "9007199254740993",
            "9007199254740992e22",
            "1e38",
            "1e39",
            "1e-38",
            "1e-46",
            "inf",
            "-nan",
            "1.5x",
        ] {
            check(s);
        }
        // Nine and more significant digits over the exponents the fast path takes, and past it.
        let mut x: u64 = 7;
        for _ in 0..300_000 {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let digits = 1 + (x >> 60) as u32;
            let m = (x >> 8) % 10u64.pow(digits.min(16));
            let e = i64::from(((x >> 3) % 60) as u32) - 34;
            check(&format!("{m}e{e}"));
            let m = (x >> 20) % 1_000_000_000;
            check(&format!("-{}.{:08}e{}", m / 100_000_000, m % 100_000_000, e / 3));
        }
        // Lines the line reader shouldn't take a number from.
        for s in ["1.5\\", "2#", "nan", "", "-", "1e", "1.5/2"] {
            assert!(f32_lead(s.as_bytes()).is_none(), "{s:?}");
        }
    }

    #[test]
    fn blocks_cut_on_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.txt");
        let line = "abc def\n".repeat(100_000);
        std::fs::write(&p, &line).unwrap();
        let open = |p: &std::path::Path| {
            let f = std::fs::File::open(p).unwrap();
            let n = f.metadata().unwrap().len();
            Blocks::reader(f, n)
        };
        let mut b = open(&p);
        let mut all = Vec::new();
        while let Some(t) = b.text(lines).unwrap() {
            assert!(t.ends_with(b"\n"));
            all.extend_from_slice(t);
        }
        assert_eq!(all, line.as_bytes());
        // In memory: the same text, read in place.
        let mut b = Blocks::slice(line.as_bytes());
        assert_eq!(b.text(lines).unwrap(), Some(line.as_bytes()));
        assert_eq!(b.text(lines).unwrap(), None);
        // One line longer than a block.
        let long = format!("{}\nx", "y".repeat(BLOCK * 3));
        std::fs::write(&p, &long).unwrap();
        let mut b = open(&p);
        let mut n = 0;
        while let Some(t) = b.text(lines).unwrap() {
            assert!(t.ends_with(b"\n") || t.ends_with(b"x"));
            n += t.len();
        }
        assert_eq!(n, long.len());
    }
}
