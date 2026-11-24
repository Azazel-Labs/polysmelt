//! LZW decoding, as GIF and TIFF code their pixels.

/// Codes a table holds: 12-bit codes.
const MAX: usize = 4096;

/// How a stream packs its codes.
#[derive(Debug, Clone, Copy)]
pub(super) struct Codes {
    /// The literals' code size: a GIF's own (2 to 8, mostly), 8 for TIFF.
    pub min: u32,
    /// Codes start at each byte's high bits (TIFF) rather than its low ones (GIF).
    pub msb: bool,
    /// Codes widen one code early, as TIFF writers do.
    pub early: bool,
}

/// LZW-coded `data` decoded onto `out`, up to `limit` bytes in all. A damaged stream stops where it goes wrong,
/// keeping what came before.
pub(super) fn decode(data: &[u8], codes: Codes, out: &mut Vec<u8>, limit: usize) {
    let Codes { min, msb, early } = codes;
    if !(1..=11).contains(&min) {
        return;
    }
    let clear = 1usize << min;
    let end = clear + 1;
    // Each code's string: the code before it, its last byte, its first byte and its length.
    let mut prefix = vec![0u16; MAX];
    let mut suffix = vec![0u8; MAX];
    let mut first = vec![0u8; MAX];
    let mut len = vec![0u16; MAX];
    for i in 0..clear {
        let b = u8::try_from(i & 0xFF).unwrap_or(0);
        (suffix[i], first[i], len[i]) = (b, b, 1);
    }
    let mut size = min + 1;
    let mut next = clear + 2;
    let mut prev: Option<usize> = None;
    let (mut acc, mut bits, mut at) = (0u32, 0u32, 0usize);
    let start = out.len();
    while out.len() < limit {
        while bits < size {
            let Some(&byte) = data.get(at) else {
                out.truncate(limit.max(start));
                return;
            };
            at += 1;
            if msb {
                acc = (acc << 8) | u32::from(byte);
            } else {
                acc |= u32::from(byte) << bits;
            }
            bits += 8;
        }
        let mask = (1u32 << size) - 1;
        let code = if msb {
            let c = (acc >> (bits - size)) & mask;
            acc &= (1u32 << (bits - size)) - 1;
            c
        } else {
            let c = acc & mask;
            acc >>= size;
            c
        } as usize;
        bits -= size;
        if code == clear {
            size = min + 1;
            next = clear + 2;
            prev = None;
            continue;
        }
        if code == end {
            break;
        }
        let Some(p) = prev else {
            // The first code after a clear is a literal.
            if code >= clear {
                break;
            }
            out.push(first[code]);
            prev = Some(code);
            continue;
        };
        if code < next && code != clear && code != end {
            emit(code, &prefix, &suffix, &len, out);
            if next < MAX {
                (prefix[next], suffix[next], first[next], len[next]) = (
                    u16::try_from(p).unwrap_or(0),
                    first[code],
                    first[p],
                    len[p].saturating_add(1),
                );
                next += 1;
            }
        } else if code == next && next < MAX {
            // The string being defined: the last one and its own first byte.
            (prefix[next], suffix[next], first[next], len[next]) = (
                u16::try_from(p).unwrap_or(0),
                first[p],
                first[p],
                len[p].saturating_add(1),
            );
            next += 1;
            emit(code, &prefix, &suffix, &len, out);
        } else {
            break;
        }
        prev = Some(code);
        if next + usize::from(early) >= 1 << size && size < 12 {
            size += 1;
        }
    }
    out.truncate(limit.max(start));
}

/// Code `code`'s string onto `out`, written back to front.
fn emit(code: usize, prefix: &[u16], suffix: &[u8], len: &[u16], out: &mut Vec<u8>) {
    let n = usize::from(len[code]);
    let at = out.len();
    out.resize(at + n, 0);
    let mut c = code;
    for o in out[at..].iter_mut().rev() {
        *o = suffix[c];
        c = usize::from(prefix[c]);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// `data` LZW-coded: a plain encoder, clearing when the table fills.
    pub(in crate::raster) fn encode(data: &[u8], codes: Codes) -> Vec<u8> {
        let Codes { min, msb, early } = codes;
        let clear = 1u32 << min;
        let mut out = Vec::new();
        let (mut acc, mut bits) = (0u64, 0u32);
        let mut put = |code: u32, size: u32, out: &mut Vec<u8>| {
            if msb {
                acc = (acc << size) | u64::from(code);
                bits += size;
                while bits >= 8 {
                    out.push(u8::try_from((acc >> (bits - 8)) & 0xFF).unwrap());
                    bits -= 8;
                }
            } else {
                acc |= u64::from(code) << bits;
                bits += size;
                while bits >= 8 {
                    out.push(u8::try_from(acc & 0xFF).unwrap());
                    acc >>= 8;
                    bits -= 8;
                }
            }
        };
        let mut table = std::collections::HashMap::new();
        let mut size = min + 1;
        let mut next = clear + 2;
        put(clear, size, &mut out);
        let mut cur: Option<u32> = None;
        for &b in data {
            let Some(c) = cur else {
                cur = Some(u32::from(b));
                continue;
            };
            if let Some(&n) = table.get(&(c, b)) {
                cur = Some(n);
                continue;
            }
            put(c, size, &mut out);
            table.insert((c, b), next);
            next += 1;
            if next + u32::from(early) > 1 << size {
                if size == 12 {
                    put(clear, size, &mut out);
                    table.clear();
                    size = min + 1;
                    next = clear + 2;
                } else {
                    size += 1;
                }
            }
            cur = Some(u32::from(b));
        }
        if let Some(c) = cur {
            put(c, size, &mut out);
            next += 1;
            if next + u32::from(early) > 1 << size && size < 12 {
                size += 1;
            }
        }
        put(clear + 1, size, &mut out);
        put(0, 7, &mut out);
        out
    }

    #[test]
    fn round_trips() {
        let mut rng = crate::test_rng::Pcg32::new(3);
        let mut data: Vec<u8> = (0..20_000u32).map(|i| u8::try_from((i / 7) % 4).unwrap()).collect();
        data.extend((0..20_000).map(|_| u8::try_from(rng.next_u32() % 4).unwrap()));
        data.extend(std::iter::repeat_n(3, 30_000));
        for codes in [
            Codes {
                min: 2,
                msb: false,
                early: false,
            },
            Codes {
                min: 8,
                msb: true,
                early: true,
            },
            Codes {
                min: 8,
                msb: false,
                early: false,
            },
        ] {
            let coded = encode(&data, codes);
            let mut out = Vec::new();
            decode(&coded, codes, &mut out, usize::MAX);
            assert_eq!(out.len(), data.len(), "{codes:?}");
            assert!(out == data, "{codes:?}");
            // A limit stops it, and damage only stops it early.
            let mut out = Vec::new();
            decode(&coded, codes, &mut out, 1000);
            assert_eq!(out[..], data[..1000]);
            let mut out = Vec::new();
            decode(&coded[..coded.len() / 2], codes, &mut out, usize::MAX);
            assert!(data.starts_with(&out));
        }
    }

    #[test]
    fn bad_streams_stop() {
        let mut out = Vec::new();
        decode(
            &[0xFF; 64],
            Codes {
                min: 2,
                msb: false,
                early: false,
            },
            &mut out,
            1 << 20,
        );
        decode(
            &[0xFF; 64],
            Codes {
                min: 0,
                msb: false,
                early: false,
            },
            &mut out,
            1 << 20,
        );
        decode(
            &[0xFF; 64],
            Codes {
                min: 12,
                msb: true,
                early: true,
            },
            &mut out,
            1 << 20,
        );
        assert!(out.len() < 1 << 20);
    }
}
