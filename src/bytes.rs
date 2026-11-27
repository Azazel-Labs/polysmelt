//! Reading binary files: [`Bin`], a cursor that checks every read against the end, and reads at
//! offsets that give `None` past it.
//!
//! The byte order is a type: [`Le`] and [`Be`] cost nothing at run time, and [`Endian`] holds
//! one a file chooses for itself. Bulk reads ([`Bin::f32s`], [`Bin::u32s`], [`Bin::vec3s`]) check
//! the whole run once, before allocating, so a huge count in a small file is an error rather
//! than a huge allocation, and decode without a bounds check per value.

use crate::Result;

/// A byte order.
pub(crate) trait Order: Copy {
    fn u16(self, b: [u8; 2]) -> u16;
    fn u32(self, b: [u8; 4]) -> u32;
    fn u64(self, b: [u8; 8]) -> u64;
}

/// Little-endian, the order most formats use.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Le;

/// Big-endian.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Be;

/// The order a file says it uses: big-endian when `big`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Endian {
    pub big: bool,
}

impl Order for Le {
    #[inline]
    fn u16(self, b: [u8; 2]) -> u16 {
        u16::from_le_bytes(b)
    }
    #[inline]
    fn u32(self, b: [u8; 4]) -> u32 {
        u32::from_le_bytes(b)
    }
    #[inline]
    fn u64(self, b: [u8; 8]) -> u64 {
        u64::from_le_bytes(b)
    }
}

impl Order for Be {
    #[inline]
    fn u16(self, b: [u8; 2]) -> u16 {
        u16::from_be_bytes(b)
    }
    #[inline]
    fn u32(self, b: [u8; 4]) -> u32 {
        u32::from_be_bytes(b)
    }
    #[inline]
    fn u64(self, b: [u8; 8]) -> u64 {
        u64::from_be_bytes(b)
    }
}

impl Order for Endian {
    #[inline]
    fn u16(self, b: [u8; 2]) -> u16 {
        if self.big { Be.u16(b) } else { Le.u16(b) }
    }
    #[inline]
    fn u32(self, b: [u8; 4]) -> u32 {
        if self.big { Be.u32(b) } else { Le.u32(b) }
    }
    #[inline]
    fn u64(self, b: [u8; 8]) -> u64 {
        if self.big { Be.u64(b) } else { Le.u64(b) }
    }
}

/// The `N` bytes at `at`, or `None` where they'd run past the end.
#[inline]
pub(crate) fn array_at<const N: usize>(d: &[u8], at: usize) -> Option<[u8; N]> {
    d.get(at..at.checked_add(N)?)?.try_into().ok()
}

/// The little-endian `u16` at `at`, or `None` past the end.
#[inline]
pub(crate) fn u16_at(d: &[u8], at: usize) -> Option<u16> {
    array_at(d, at).map(u16::from_le_bytes)
}

/// The little-endian `u32` at `at`, or `None` past the end.
#[inline]
pub(crate) fn u32_at(d: &[u8], at: usize) -> Option<u32> {
    array_at(d, at).map(u32::from_le_bytes)
}

/// The little-endian `u64` at `at`, or `None` past the end.
#[inline]
pub(crate) fn u64_at(d: &[u8], at: usize) -> Option<u64> {
    array_at(d, at).map(u64::from_le_bytes)
}

/// The little-endian `f32` at `at`, or `None` past the end.
#[inline]
pub(crate) fn f32_at(d: &[u8], at: usize) -> Option<f32> {
    array_at(d, at).map(f32::from_le_bytes)
}

/// The little-endian `f64` at `at`, or `None` past the end.
#[inline]
pub(crate) fn f64_at(d: &[u8], at: usize) -> Option<f64> {
    array_at(d, at).map(f64::from_le_bytes)
}

/// Up to `n` whole `N`-byte records from `at` on: as many as there are bytes for.
#[inline]
pub(crate) fn records<const N: usize>(d: &[u8], at: usize, n: usize) -> &[[u8; N]] {
    let all = d.get(at..).unwrap_or_default().as_chunks::<N>().0;
    all.get(..n).unwrap_or(all)
}

/// The `f32`s in `b`, in order `o`.
#[inline]
pub(crate) fn floats<O: Order, const B: usize, const N: usize>(o: O, b: [u8; B]) -> [f32; N] {
    let (words, _) = b.as_chunks::<4>();
    std::array::from_fn(|k| f32::from_bits(o.u32(words[k])))
}

/// The bytes before the first NUL, or all of them.
#[inline]
pub(crate) fn until_nul(b: &[u8]) -> &[u8] {
    b.iter().position(|&c| c == 0).map_or(b, |end| &b[..end])
}

/// Windows-1252's characters for the bytes 0x80 to 0x9F, where it differs from Latin-1. The five bytes it leaves
/// undefined stay the control characters Latin-1 has there, as browsers read them.
const CP1252_HIGH: [char; 32] = [
    '\u{20AC}', '\u{81}', '\u{201A}', '\u{192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}', '\u{2C6}', '\u{2030}',
    '\u{160}', '\u{2039}', '\u{152}', '\u{8D}', '\u{17D}', '\u{8F}', '\u{90}', '\u{2018}', '\u{2019}', '\u{201C}',
    '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}', '\u{2DC}', '\u{2122}', '\u{161}', '\u{203A}', '\u{153}', '\u{9D}',
    '\u{17E}', '\u{178}',
];

/// The character a byte is in Windows-1252: Latin-1's, but for 0x80 to 0x9F.
#[inline]
pub(crate) fn windows_1252_char(b: u8) -> char {
    match b {
        0x80..=0x9F => CP1252_HIGH[usize::from(b - 0x80)],
        _ => char::from(b),
    }
}

/// Bytes read as Windows-1252, each one a character.
pub(crate) fn windows_1252(b: &[u8]) -> String {
    b.iter().map(|&c| windows_1252_char(c)).collect()
}

/// Text from a file as a string, the one way every reader decodes text whose encoding its format doesn't state.
///
/// UTF-8 is taken as it is, without a byte order mark. UTF-16 that starts with a byte order mark is decoded (an
/// unpaired surrogate becomes U+FFFD). Anything else is read as Windows-1252, which older tools wrote and which
/// gives every byte a character: text that has bytes never comes out empty or as U+FFFD. Formats that state
/// their encoding (`MikuMikuDance`'s Shift-JIS and UTF-16, an XML declaration) decode by it instead.
pub(crate) fn text(b: &[u8]) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;
    let b = b.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(b);
    let wide = match b {
        [0xFF, 0xFE, rest @ ..] if rest.len() % 2 == 0 => Some((true, rest)),
        [0xFE, 0xFF, rest @ ..] if rest.len() % 2 == 0 => Some((false, rest)),
        _ => None,
    };
    if let Some((le, rest)) = wide {
        let units = rest.as_chunks::<2>().0.iter().map(|&c| {
            if le {
                u16::from_le_bytes(c)
            } else {
                u16::from_be_bytes(c)
            }
        });
        return Cow::Owned(
            char::decode_utf16(units)
                .map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER))
                .collect(),
        );
    }
    match std::str::from_utf8(b) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(windows_1252(b)),
    }
}

/// [`text`] from bytes a reader owns.
pub(crate) fn text_owned(b: Vec<u8>) -> String {
    match String::from_utf8(b) {
        Ok(s) if !s.starts_with('\u{FEFF}') => s,
        Ok(s) => s.trim_start_matches('\u{FEFF}').to_owned(),
        Err(e) => text(e.as_bytes()).into_owned(),
    }
}

/// A name from a binary file's text field, the same for every reader: the bytes before the first NUL, as
/// [`text`] reads them, without the spaces (or other whitespace) that pad its end.
pub(crate) fn name(b: &[u8]) -> String {
    text(until_nul(b)).trim_end().to_owned()
}

/// A reader over a file's bytes that checks every read against the end. A read past it fails
/// with a [`Truncated`](crate::ErrorKind::Truncated) error that names the format and leaves the
/// cursor where it was.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bin<'a, O: Order = Le> {
    pub data: &'a [u8],
    /// Where the next read starts; never past the end of `data`.
    pub at: usize,
    /// The format's name, for errors.
    pub what: &'static str,
    pub order: O,
}

impl<'a> Bin<'a, Le> {
    /// A little-endian reader from the start of `data`.
    pub fn new(data: &'a [u8], what: &'static str) -> Self {
        Self::with_order(data, what, Le)
    }
}

impl<'a> Bin<'a, Be> {
    /// A big-endian reader from the start of `data`.
    pub fn be(data: &'a [u8], what: &'static str) -> Self {
        Self::with_order(data, what, Be)
    }
}

/// `n` values of `size` bytes each, read in bulk: one check for the run, then no bounds check
/// per value.
macro_rules! bulk {
    ($(#[$doc:meta] $name:ident -> $t:ty, $size:literal, |$o:ident, $c:ident| $conv:expr;)*) => {$(
        #[$doc]
        #[inline]
        pub fn $name(&mut self, n: usize) -> Result<Vec<$t>> {
            let $o = self.order;
            let b = self.run(n, $size)?;
            Ok(b.as_chunks::<$size>().0.iter().map(|&$c| $conv).collect())
        }
    )*};
}

impl<'a, O: Order> Bin<'a, O> {
    /// A reader from the start of `data` in byte order `order`.
    pub fn with_order(data: &'a [u8], what: &'static str, order: O) -> Self {
        Self {
            data,
            at: 0,
            what,
            order,
        }
    }

    /// Where the next read starts.
    #[inline]
    pub fn pos(&self) -> usize {
        self.at
    }

    /// Bytes left.
    #[inline]
    pub fn left(&self) -> usize {
        self.data.len().saturating_sub(self.at)
    }

    /// The bytes left, without moving past them.
    #[inline]
    pub fn rest(&self) -> &'a [u8] {
        self.data.get(self.at..).unwrap_or_default()
    }

    /// The next `n` bytes.
    #[inline]
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        match self.at.checked_add(n).filter(|&end| end <= self.data.len()) {
            Some(end) => {
                let s = &self.data[self.at..end];
                self.at = end;
                Ok(s)
            }
            None => Err(self.cut_short()),
        }
    }

    #[cold]
    #[inline(never)]
    fn cut_short(&self) -> crate::Error {
        err!(Truncated, "{} file is cut short", self.what)
    }

    /// Moves past `n` bytes.
    #[inline]
    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }

    /// Moves to `off` from the start.
    pub fn seek(&mut self, off: usize) -> Result<()> {
        ensure!(
            off <= self.data.len(),
            Truncated,
            "{} file has an offset that runs past the end of the file",
            self.what
        );
        self.at = off;
        Ok(())
    }

    /// A reader over the next `n` bytes, which this one moves past: a chunk's body, whose reads
    /// can't run into what follows it.
    #[inline]
    pub fn sub(&mut self, n: usize) -> Result<Self> {
        let data = self.bytes(n)?;
        Ok(Self { data, at: 0, ..*self })
    }

    /// The next `N` bytes as an array.
    #[inline]
    pub fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.bytes(N)?);
        Ok(out)
    }

    #[inline]
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    #[inline]
    pub fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()?.cast_signed())
    }

    /// A byte, true unless 0.
    #[inline]
    pub fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    #[inline]
    pub fn u16(&mut self) -> Result<u16> {
        let b = self.array()?;
        Ok(self.order.u16(b))
    }

    #[inline]
    pub fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()?.cast_signed())
    }

    #[inline]
    pub fn u32(&mut self) -> Result<u32> {
        let b = self.array()?;
        Ok(self.order.u32(b))
    }

    #[inline]
    pub fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()?.cast_signed())
    }

    #[inline]
    pub fn u64(&mut self) -> Result<u64> {
        let b = self.array()?;
        Ok(self.order.u64(b))
    }

    #[inline]
    pub fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()?.cast_signed())
    }

    #[inline]
    pub fn f32(&mut self) -> Result<f32> {
        self.u32().map(f32::from_bits)
    }

    #[inline]
    pub fn f64(&mut self) -> Result<f64> {
        self.u64().map(f64::from_bits)
    }

    #[inline]
    pub fn vec2(&mut self) -> Result<[f32; 2]> {
        let b = self.array::<8>()?;
        Ok(floats(self.order, b))
    }

    #[inline]
    pub fn vec3(&mut self) -> Result<[f32; 3]> {
        let b = self.array::<12>()?;
        Ok(floats(self.order, b))
    }

    #[inline]
    pub fn vec4(&mut self) -> Result<[f32; 4]> {
        let b = self.array::<16>()?;
        Ok(floats(self.order, b))
    }

    /// A count of items of `size` bytes each that must fit in what's left: refused otherwise,
    /// so nothing is allocated for a count the file can't hold.
    pub fn count(&self, n: u64, size: usize) -> Result<usize> {
        self.fits(crate::num::usize_of(n), size)
    }

    /// `n`, when `n` items of `size` bytes each fit in what's left.
    #[inline]
    fn fits(&self, n: usize, size: usize) -> Result<usize> {
        match n.checked_mul(size.max(1)) {
            Some(b) if b <= self.left() => Ok(n),
            _ => bail!(
                Truncated,
                "{} file is cut short: it claims {n} items, more than it holds",
                self.what
            ),
        }
    }

    /// The bytes of `n` items of `size` bytes each, checked as [`count`](Self::count) checks
    /// them.
    #[inline]
    pub fn run(&mut self, n: usize, size: usize) -> Result<&'a [u8]> {
        let n = self.fits(n, size)?;
        self.bytes(n * size)
    }

    bulk! {
        /// `n` bytes as signed values.
        i8s -> i8, 1, |_o, c| c[0].cast_signed();
        /// `n` 16-bit values.
        u16s -> u16, 2, |o, c| o.u16(c);
        /// `n` signed 16-bit values.
        i16s -> i16, 2, |o, c| o.u16(c).cast_signed();
        /// `n` 32-bit values.
        u32s -> u32, 4, |o, c| o.u32(c);
        /// `n` signed 32-bit values.
        i32s -> i32, 4, |o, c| o.u32(c).cast_signed();
        /// `n` floats.
        f32s -> f32, 4, |o, c| f32::from_bits(o.u32(c));
        /// `n` doubles.
        f64s -> f64, 8, |o, c| f64::from_bits(o.u64(c));
        /// `n` pairs of floats.
        vec2s -> [f32; 2], 8, |o, c| {
            let (w, _) = c.as_chunks::<4>();
            [0, 1].map(|k| f32::from_bits(o.u32(w[k])))
        };
        /// `n` vectors of three floats.
        vec3s -> [f32; 3], 12, |o, c| {
            let (w, _) = c.as_chunks::<4>();
            [0, 1, 2].map(|k| f32::from_bits(o.u32(w[k])))
        };
        /// `n` vectors of four floats.
        vec4s -> [f32; 4], 16, |o, c| {
            let (w, _) = c.as_chunks::<4>();
            [0, 1, 2, 3].map(|k| f32::from_bits(o.u32(w[k])))
        };
    }

    /// `n` indices of `size` bytes each (1, 2 or 4), widened to 32 bits.
    pub fn indices(&mut self, n: usize, size: usize) -> Result<Vec<u32>> {
        Ok(match size {
            1 => self.run(n, 1)?.iter().map(|&b| u32::from(b)).collect(),
            2 => {
                let o = self.order;
                let b = self.run(n, 2)?;
                b.as_chunks::<2>().0.iter().map(|&c| u32::from(o.u16(c))).collect()
            }
            _ => self.u32s(n)?,
        })
    }

    /// A fixed-size string field: up to its first NUL.
    #[inline]
    pub fn fixed_str(&mut self, n: usize) -> Result<&'a [u8]> {
        self.bytes(n).map(until_nul)
    }

    /// A NUL-terminated string, without its NUL; one the file ends first is refused.
    pub fn cstr(&mut self) -> Result<&'a [u8]> {
        let rest = self.rest();
        match rest.iter().position(|&c| c == 0) {
            Some(end) => {
                self.at += end + 1;
                Ok(&rest[..end])
            }
            None => Err(self.cut_short()),
        }
    }

    /// The bytes up to `stop` (or the end), moving past `stop` too.
    pub fn until(&mut self, stop: u8) -> &'a [u8] {
        let rest = self.rest();
        let end = rest.iter().position(|&c| c == stop);
        self.at += end.map_or(rest.len(), |e| e + 1);
        &rest[..end.unwrap_or(rest.len())]
    }

    /// The `u16` at `at` from the start, in this reader's order, without moving; `None` past
    /// the end.
    #[inline]
    pub fn u16_at(&self, at: usize) -> Option<u16> {
        array_at(self.data, at).map(|b| self.order.u16(b))
    }

    /// The `u32` at `at` from the start, in this reader's order, without moving; `None` past
    /// the end.
    #[inline]
    pub fn u32_at(&self, at: usize) -> Option<u32> {
        array_at(self.data, at).map(|b| self.order.u32(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn reads_in_either_order() {
        let d = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut le = Bin::new(&d, "test");
        assert_eq!(le.u16().unwrap(), 0x0201);
        assert_eq!(le.u32().unwrap(), 0x0605_0403);
        let mut be = Bin::be(&d, "test");
        assert_eq!(be.u16().unwrap(), 0x0102);
        assert_eq!(be.u32().unwrap(), 0x0304_0506);
        assert_eq!(be.u16().unwrap(), 0x0708);
        let mut e = Bin::with_order(&d, "test", Endian { big: true });
        assert_eq!(e.u64().unwrap(), 0x0102_0304_0506_0708);
        let mut e = Bin::with_order(&d, "test", Endian { big: false });
        assert_eq!(e.u64().unwrap(), 0x0807_0605_0403_0201);
        assert_eq!(Bin::be(&d, "t").u32s(2).unwrap(), [0x0102_0304, 0x0506_0708]);
        assert_eq!(Bin::new(&d, "t").u16s(4).unwrap(), [0x0201, 0x0403, 0x0605, 0x0807]);
        assert_eq!(Bin::new(&[0xff, 0x80], "t").i8s(2).unwrap(), [-1, -128]);
    }

    #[test]
    fn floats_and_vectors() {
        let mut d = Vec::new();
        for v in [1.0f32, -2.0, 0.5, 8.0] {
            d.extend(v.to_be_bytes());
        }
        assert_eq!(Bin::be(&d, "t").vec4().unwrap(), [1.0, -2.0, 0.5, 8.0]);
        assert_eq!(Bin::be(&d, "t").vec2s(2).unwrap(), [[1.0, -2.0], [0.5, 8.0]]);
        let mut b = Bin::be(&d, "t");
        assert_eq!(b.vec3s(1).unwrap(), [[1.0, -2.0, 0.5]]);
        assert_eq!(b.f32().unwrap(), 8.0);
        let le: Vec<u8> = [3.0f64, -1.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(Bin::new(&le, "t").f64s(2).unwrap(), [3.0, -1.5]);
        assert_eq!(f64_at(&le, 8), Some(-1.5));
        assert_eq!(f64_at(&le, 9), None);
    }

    #[test]
    fn a_short_read_is_truncated_and_moves_nothing() {
        let mut b = Bin::new(&[1, 2, 3], "XYZ");
        assert_eq!(b.u16().unwrap(), 0x0201);
        let e = b.u32().unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Truncated);
        assert_eq!(e.to_string(), "XYZ file is cut short");
        assert_eq!(b.pos(), 2);
        assert_eq!(b.u8().unwrap(), 3);
        assert_eq!(b.left(), 0);
        assert!(b.u8().is_err());
        assert!(b.skip(usize::MAX).is_err());
        assert_eq!(b.seek(4).unwrap_err().kind(), ErrorKind::Truncated);
        b.seek(1).unwrap();
        assert_eq!(b.rest(), [2, 3]);
    }

    #[test]
    fn a_huge_count_in_a_tiny_file_fails_before_allocating() {
        let d = [0u8; 16];
        for n in [9, 1 << 40, usize::MAX / 2, usize::MAX] {
            let mut b = Bin::new(&d, "T");
            assert_eq!(b.f32s(n).unwrap_err().kind(), ErrorKind::Truncated);
            assert_eq!(b.vec3s(n).unwrap_err().kind(), ErrorKind::Truncated);
            assert_eq!(b.u32s(n).unwrap_err().kind(), ErrorKind::Truncated);
            assert_eq!(b.indices(n, 2).unwrap_err().kind(), ErrorKind::Truncated);
            assert_eq!(b.pos(), 0);
        }
        assert_eq!(
            Bin::new(&d, "T").count(u64::MAX, 4).unwrap_err().kind(),
            ErrorKind::Truncated
        );
        assert_eq!(Bin::new(&d, "T").count(4, 4).unwrap(), 4);
        assert_eq!(Bin::new(&d, "T").f32s(4).unwrap().len(), 4);
        assert_eq!(Bin::new(&d, "T").f32s(0).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn sub_readers_end_with_their_chunk() {
        let d = [1, 0, 2, 0, 3, 0];
        let mut b = Bin::new(&d, "T");
        let mut c = b.sub(4).unwrap();
        assert_eq!(b.pos(), 4);
        assert_eq!(c.u16().unwrap(), 1);
        assert_eq!(c.u16().unwrap(), 2);
        assert!(c.u16().is_err());
        assert_eq!(b.u16().unwrap(), 3);
        assert!(b.sub(1).is_err());
    }

    #[test]
    fn strings() {
        let d = b"ab\0cd\0\0\0ef";
        let mut b = Bin::new(d, "T");
        assert_eq!(b.cstr().unwrap(), b"ab");
        assert_eq!(b.fixed_str(5).unwrap(), b"cd");
        assert_eq!(b.until(b'\n'), b"ef");
        assert_eq!(b.left(), 0);
        assert!(Bin::new(b"no end", "T").cstr().is_err());
        assert_eq!(until_nul(b"x\0y"), b"x");
    }

    #[test]
    fn text_is_utf8_then_utf16_by_its_mark_then_windows_1252() {
        assert_eq!(text(b"plain"), "plain");
        assert_eq!(text("caf\u{e9} \u{65e5}".as_bytes()), "caf\u{e9} \u{65e5}");
        assert_eq!(
            text(b"\xEF\xBB\xBFmarked"),
            "marked",
            "a UTF-8 byte order mark is dropped"
        );
        assert_eq!(text(b"\xFF\xFEh\0i\0"), "hi");
        assert_eq!(text(b"\xFE\xFF\0h\0i"), "hi");
        // Not UTF-8: Windows-1252, where 0x80 to 0x9F aren't Latin-1's control characters.
        assert_eq!(text(b"caf\xE9"), "caf\u{e9}");
        assert_eq!(
            text(b"\x80 5 \x93quoted\x94 \x99"),
            "\u{20ac} 5 \u{201c}quoted\u{201d} \u{2122}"
        );
        assert_eq!(text(b"\x81\x8D\x8F\x90\x9D"), "\u{81}\u{8d}\u{8f}\u{90}\u{9d}");
        // Every byte has a character: nothing comes out empty or replaced.
        let all: Vec<u8> = (0x80..=0xFF).collect();
        let s = text(&all);
        assert_eq!(s.chars().count(), 128);
        assert!(!s.contains('\u{FFFD}'));
        assert_eq!(text_owned(b"caf\xE9".to_vec()), "caf\u{e9}");
        assert_eq!(text_owned(b"\xEF\xBB\xBFok".to_vec()), "ok");
    }

    #[test]
    fn names_stop_at_a_nul_and_lose_their_padding() {
        assert_eq!(name(b"bone\0junk"), "bone");
        assert_eq!(name(b"padded   \0\0"), "padded");
        assert_eq!(name(b"  lead kept "), "  lead kept");
        assert_eq!(name(b"t\xEAte\0"), "t\u{ea}te");
        assert_eq!(name(b"\0"), "");
    }

    #[test]
    fn reads_at_offsets_stop_at_the_end() {
        let d = [1, 0, 0, 0];
        assert_eq!(u32_at(&d, 0), Some(1));
        assert_eq!(u16_at(&d, 2), Some(0));
        assert_eq!(u16_at(&d, 3), None);
        assert_eq!(u32_at(&d, usize::MAX), None);
        assert_eq!(u64_at(&d, 0), None);
        assert_eq!(Bin::be(&d, "T").u16_at(0), Some(0x0100));
        assert_eq!(Bin::be(&d, "T").u32_at(1), None);
    }
}
