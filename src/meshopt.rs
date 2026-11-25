//! The `EXT_meshopt_compression` and `KHR_meshopt_compression` bitstreams, ported from
//! meshoptimizer: the attribute encoder (codec versions 0 and 1, levels 0 to 3), the triangle and
//! index sequence encoders, their bounds, the octahedral, quaternion, exponential and color filter
//! encoders, and decoders for every mode and filter.
//!
//! Casts here are the codec's own: bytes and nibbles packed into and out of wider integers,
//! zigzag and two's-complement reinterpretation, and quantizing floats to fixed point, all as
//! meshoptimizer does them, bit for bit (the tests round-trip each).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "bit-exact port of meshoptimizer's codec"
)]

use crate::{Error, Result};

/// Elements per delta group.
const GROUP: usize = 16;

/// The triangle codec's `codeaux` table, meshoptimizer's (built from symbol frequencies).
const CODEAUX: [u8; 16] = [
    0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0, 0,
];

/// Rotations that put a matched edge first.
const ORDER: [[usize; 3]; 3] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];

fn bad(what: &str) -> Error {
    Error::corrupt(format!("EXT_meshopt_compression: {what}"))
}

fn cut(what: &str) -> Error {
    Error::truncated(format!("EXT_meshopt_compression: {what} is cut short"))
}

fn zigzag8(v: u8) -> u8 {
    if v & 0x80 != 0 { !(v << 1) } else { v << 1 }
}

fn unzigzag8(v: u8) -> u8 {
    if v & 1 != 0 { !(v >> 1) } else { v >> 1 }
}

/// Elements per attribute block.
fn block_elements(stride: usize) -> usize {
    ((8192 / stride) & !15).min(256)
}

/// Mode 0 (`ATTRIBUTES`) at attribute codec version 0, the version `EXT_meshopt_compression`
/// allows: `data` holds elements of `stride` bytes (a multiple of 4, at most 256), each byte
/// position delta coded against the previous element in groups of 16. The same bytes as
/// [`encode_vertex_buffer_level`] at level 2, version 0. (meshoptimizer's own
/// `meshopt_encodeVertexBuffer` writes version 1 since its 1.0 release.)
///
/// # Panics
///
/// When `stride` isn't a multiple of 4 from 4 to 256, or `data` isn't whole elements.
#[must_use]
pub fn encode_vertex_buffer(data: &[u8], stride: usize) -> Vec<u8> {
    encode_vertex_buffer_level(data, stride, 2, 0)
}

/// Mode 0 (`ATTRIBUTES`) at a compression `level` and attribute codec `version`, byte for byte
/// meshoptimizer's `meshopt_encodeVertexBufferLevel`. `data` holds elements of `stride` bytes (a
/// multiple of 4, at most 256).
///
/// Version 0 (header `0xa0`) is what `EXT_meshopt_compression` allows; it ignores `level`.
/// Version 1 (header `0xa1`, `KHR_meshopt_compression` only) adds per-block control modes and
/// per-channel delta modes, and `level` picks how hard the encoder looks for them: 0 is fastest,
/// 1 picks each byte plane's control mode, 2 (meshoptimizer's default) also picks byte or 16-bit
/// deltas per 4-byte channel, and 3 also tries rotated 32-bit XOR deltas, which suit floats.
/// Levels above 3, up to 9, encode as 3.
///
/// # Panics
///
/// When `stride` isn't a multiple of 4 from 4 to 256, `data` isn't whole elements, `level` is
/// above 9 or `version` above 1.
#[must_use]
pub fn encode_vertex_buffer_level(data: &[u8], stride: usize, level: u8, version: u8) -> Vec<u8> {
    assert!(
        stride > 0 && stride <= 256 && stride.is_multiple_of(4),
        "attribute stride"
    );
    assert!(data.len().is_multiple_of(stride), "attribute data isn't whole elements");
    assert!(level <= 9, "attribute codec level above 9");
    assert!(version <= 1, "attribute codec version above 1");
    let count = data.len() / stride;
    let v1 = version == 1;
    let block = block_elements(stride);
    let mut first = [0u8; 256];
    if count > 0 {
        first[..stride].copy_from_slice(&data[..stride]);
    }
    // Delta mode per 4-byte channel: 0 bytes, 1 16-bit halves, 2 | rotation << 4 a 32-bit XOR.
    let mut channels = [0u8; 64];
    if v1 && level > 1 && count > 1 {
        for k in (0..stride).step_by(4) {
            let rot = if level >= 3 {
                estimate_rotate(data, stride, k)
            } else {
                0
            };
            channels[k / 4] = estimate_channel(data, stride, k, block, if level >= 3 { 3 } else { 2 }, rot);
        }
    }
    let mut out = Vec::with_capacity(encode_vertex_buffer_bound(count, stride));
    out.push(0xa0 | version);
    let mut last = first;
    for rows in data.chunks(block * stride) {
        encode_vertex_block(&mut out, rows, stride, &mut last, &channels, v1, level);
    }
    // Tail: the baseline element (and in version 1 the channel modes), padded in front.
    let tail = stride + if v1 { stride / 4 } else { 0 };
    out.resize(out.len() + (if v1 { 24usize } else { 32 }).saturating_sub(tail), 0);
    out.extend_from_slice(&first[..stride]);
    if v1 {
        out.extend_from_slice(&channels[..stride / 4]);
    }
    out
}

/// The most bytes [`encode_vertex_buffer_level`] writes for `count` elements of `stride` bytes,
/// at any level and version: meshoptimizer's `meshopt_encodeVertexBufferBound`.
///
/// # Panics
///
/// When `stride` isn't a multiple of 4 from 4 to 256.
#[must_use]
pub fn encode_vertex_buffer_bound(count: usize, stride: usize) -> usize {
    assert!(
        stride > 0 && stride <= 256 && stride.is_multiple_of(4),
        "attribute stride"
    );
    let block = block_elements(stride);
    let header = (block / GROUP).div_ceil(4);
    1 + count.div_ceil(block) * (stride / 4 + stride * (header + block)) + (stride + stride / 4).max(32)
}

/// One block of up to [`block_elements`] elements (`rows`), each byte plane of each channel
/// delta coded against the block before (`last`, which then becomes this block's last element).
fn encode_vertex_block(
    out: &mut Vec<u8>,
    rows: &[u8],
    stride: usize,
    last: &mut [u8; 256],
    channels: &[u8; 64],
    v1: bool,
    level: u8,
) {
    let n = rows.len() / stride;
    let padded = n.next_multiple_of(GROUP);
    let control = out.len();
    if v1 {
        out.resize(control + stride / 4, 0);
    }
    // Deltas past `n`, up to the whole group, stay zero.
    let mut planes = [[0u8; 256]; 4];
    for k in (0..stride).step_by(4) {
        let channel = if v1 { channels[k / 4] } else { 0 };
        encode_deltas(&mut planes, rows, stride, &last[k..k + 4], k, channel);
        for (j, plane) in planes.iter().enumerate() {
            let plane = &plane[..padded];
            let ctrl = if v1 { estimate_control(plane, n, level) } else { 0 };
            if v1 {
                out[control + k / 4] |= ctrl << (j * 2);
            }
            match ctrl {
                // Literal bytes, unpadded.
                3 => out.extend_from_slice(&plane[..n]),
                // All zero: nothing to write.
                2 => {}
                c => encode_bytes(out, plane, if v1 { BITS_V1[usize::from(c)] } else { BITS_V0 }),
            }
        }
    }
    last[..stride].copy_from_slice(&rows[rows.len() - stride..]);
}

/// The deltas of channel `k..k + 4` of `rows` against `prev` and then each previous element, one
/// byte plane per byte, in `channel`'s mode: zigzag bytes, zigzag 16-bit halves (little endian),
/// or a 32-bit XOR rotated left by `channel >> 4`.
fn encode_deltas(planes: &mut [[u8; 256]; 4], rows: &[u8], stride: usize, prev: &[u8], k: usize, channel: u8) {
    let mut p: [u8; 4] = prev.try_into().unwrap();
    for (i, row) in rows.chunks_exact(stride).enumerate() {
        let v: [u8; 4] = row[k..k + 4].try_into().unwrap();
        let d = match channel & 3 {
            0 => [0, 1, 2, 3].map(|j| zigzag8(v[j].wrapping_sub(p[j]))),
            1 => {
                let half = |h: usize| {
                    let d = u16::from_le_bytes([v[2 * h], v[2 * h + 1]])
                        .wrapping_sub(u16::from_le_bytes([p[2 * h], p[2 * h + 1]]));
                    (d << 1) ^ (d >> 15).wrapping_neg()
                };
                let [a, b] = [half(0).to_le_bytes(), half(1).to_le_bytes()];
                [a[0], a[1], b[0], b[1]]
            }
            _ => (u32::from_le_bytes(v) ^ u32::from_le_bytes(p))
                .rotate_left(u32::from(channel >> 4))
                .to_le_bytes(),
        };
        for (plane, d) in planes.iter_mut().zip(d) {
            plane[i] = d;
        }
        p = v;
    }
}

/// Bytes one group of 16 takes packed `bits` wide (0, 1, 2, 4 or 8), out-of-range values
/// included; `usize::MAX` when 0 bits can't hold it.
fn group_size(d: &[u8], bits: u8) -> usize {
    match bits {
        0 if d.iter().all(|&x| x == 0) => 0,
        0 => usize::MAX,
        8 => GROUP,
        b => {
            let sentinel = (1u8 << b) - 1;
            GROUP * usize::from(b) / 8 + d.iter().filter(|&&x| x >= sentinel).count()
        }
    }
}

/// One byte plane (`deltas`, whole groups): 2 header bits per group, then each group in the
/// smallest of the 4 widths `bits` offers. Ties go to the previous group's width, unless the
/// smallest so far is the 8-bit literal, then to the widest.
fn encode_bytes(out: &mut Vec<u8>, deltas: &[u8], bits: [u8; 4]) {
    let header = out.len();
    out.resize(header + (deltas.len() / GROUP).div_ceil(4), 0);
    let mut last_bits = None;
    for (g, d) in deltas.chunks_exact(GROUP).enumerate() {
        let mut best = 3;
        let mut best_size = group_size(d, bits[3]);
        for k in 0..3 {
            let size = group_size(d, bits[k]);
            if size < best_size || (size == best_size && last_bits == Some(bits[k]) && bits[best] != 8) {
                best = k;
                best_size = size;
            }
        }
        out[header + g / 4] |= (best as u8) << ((g % 4) * 2);
        last_bits = Some(bits[best]);
        match bits[best] {
            0 => {}
            8 => out.extend_from_slice(d),
            b => {
                let sentinel = (1u8 << b) - 1;
                for chunk in d.chunks_exact(usize::from(8 / b)) {
                    let mut byte = 0u8;
                    for &x in chunk {
                        byte = (byte << b) | x.min(sentinel);
                    }
                    // 1-bit groups hold their first element in the lowest bit.
                    out.push(if b == 1 { byte.reverse_bits() } else { byte });
                }
                out.extend(d.iter().filter(|&&x| x >= sentinel));
            }
        }
    }
}

/// A version 1 byte plane's control mode (`plane` padded to whole groups, `n` elements): 2 when
/// all zero, 3 (literal) when neither packed mode beats `n` bytes, else 0 or 1, whichever packs
/// smaller by estimate. Level 0 skips the estimate and takes 1.
fn estimate_control(plane: &[u8], n: usize, level: u8) -> u8 {
    if plane.iter().all(|&x| x == 0) {
        return 2;
    }
    if level == 0 {
        return 1;
    }
    let header = (plane.len() / GROUP).div_ceil(4);
    let (mut est0, mut est1) = (header, header);
    for d in plane.chunks_exact(GROUP) {
        let size124 = group_size(d, 1).min(group_size(d, 2)).min(group_size(d, 4));
        est0 += size124.min(group_size(d, 0));
        est1 += size124.min(group_size(d, 8));
    }
    if est0 < n || est1 < n {
        u8::from(est0 >= est1)
    } else {
        3
    }
}

/// Packed bits a byte of XOR deltas costs, by estimate.
fn estimate_bits(v: u8) -> usize {
    match v {
        0 => 0,
        1..=3 => 2,
        4..=15 => 4,
        _ => 8,
    }
}

/// The left rotation (0 to 7) that packs channel `k..k + 4`'s XOR deltas smallest by estimate,
/// judged on the bits that change within each group of 16 elements.
fn estimate_rotate(data: &[u8], stride: usize, k: usize) -> u8 {
    let word = |row: &[u8]| u32::from_le_bytes(row[k..k + 4].try_into().unwrap());
    let mut last = word(data);
    let mut sizes = [0usize; 8];
    for group in data.chunks(stride * GROUP) {
        let mut changed = 0;
        for row in group.chunks_exact(stride) {
            let v = word(row);
            changed |= v ^ last;
            last = v;
        }
        for (rot, size) in (0u32..).zip(&mut sizes) {
            *size += changed
                .rotate_left(rot)
                .to_le_bytes()
                .map(estimate_bits)
                .iter()
                .sum::<usize>();
        }
    }
    let mut best = 0;
    for (rot, &size) in sizes.iter().enumerate().skip(1) {
        if size < sizes[best] {
            best = rot;
        }
    }
    best as u8
}

/// The delta mode for channel `k..k + 4`, of the first `max_channel` (bytes, 16-bit halves, XOR
/// rotated by `rot`), that packs every third block of `block` elements smallest by estimate.
fn estimate_channel(data: &[u8], stride: usize, k: usize, block: usize, max_channel: u8, rot: u8) -> u8 {
    let count = data.len() / stride;
    let mut planes = [[0u8; 256]; 4];
    let mut sizes = [0usize; 3];
    for start in (0..count).step_by(block * 3) {
        let n = block.min(count - start);
        let padded = n.next_multiple_of(GROUP);
        let prev = &data[start.saturating_sub(1) * stride + k..][..4];
        for plane in &mut planes {
            plane[n..padded].fill(0);
        }
        let rows = &data[start * stride..(start + n) * stride];
        for (channel, size) in (0..max_channel).zip(&mut sizes) {
            encode_deltas(&mut planes, rows, stride, prev, k, channel | rot << 4);
            for plane in &planes {
                for d in plane[..padded].chunks_exact(GROUP) {
                    *size += [1, 2, 4, 8].map(|b| group_size(d, b)).into_iter().min().unwrap();
                }
            }
        }
    }
    let mut best = 0;
    for channel in 1..usize::from(max_channel) {
        if sizes[channel] < sizes[best] {
            best = channel;
        }
    }
    if best == 2 { 2 | rot << 4 } else { best as u8 }
}

/// Group bit widths per header value: version 0, then version 1 under control modes 0 and 1.
const BITS_V0: [u8; 4] = [0, 2, 4, 8];
const BITS_V1: [[u8; 4]; 2] = [[0, 1, 2, 4], [1, 2, 4, 8]];

/// Decodes mode 0 into `count` elements of `stride` bytes. Reads both attribute codec versions:
/// 0 (`EXT_meshopt_compression`, header `0xa0`) and 1 (`KHR_meshopt_compression`, `0xa1`).
///
/// # Errors
///
/// When `src` is truncated or malformed, or `stride` isn't a multiple of 4 from 4 to 256.
pub fn decode_vertex_buffer(src: &[u8], count: usize, stride: usize) -> Result<Vec<u8>> {
    let size = count.checked_mul(stride).ok_or_else(|| bad("attribute count"))?;
    // A block of at most 256 elements costs at least a quarter byte per element byte, so a
    // stream can't hold more than 1024 times its size: refuse before allocating.
    if size / 1024 > src.len() {
        return Err(cut("attribute data"));
    }
    let mut out = vec![0u8; size];
    decode_vertex_buffer_into(src, &mut out, stride)?;
    Ok(out)
}

/// Decodes mode 0 into `out`, whole elements of `stride` bytes, without allocating.
///
/// # Errors
///
/// When `src` is truncated or malformed, `stride` isn't a multiple of 4 from 4 to 256, or
/// `out` isn't whole elements.
pub fn decode_vertex_buffer_into(src: &[u8], out: &mut [u8], stride: usize) -> Result<()> {
    if stride == 0 || stride > 256 || !stride.is_multiple_of(4) || !out.len().is_multiple_of(stride) {
        return Err(bad("attribute stride"));
    }
    let count = out.len() / stride;
    let header = *src.first().ok_or_else(|| bad("attribute header"))?;
    if header & 0xf0 != 0xa0 {
        return Err(bad("attribute header"));
    }
    let v1 = match header & 15 {
        0 => false,
        1 => true,
        v => {
            return Err(Error::unsupported(format!(
                "EXT_meshopt_compression: attribute codec version {v} is not supported"
            )));
        }
    };
    let channels_len = if v1 { stride / 4 } else { 0 };
    let tail = stride + channels_len;
    let tail_pad = tail.max(if v1 { 24 } else { 32 });
    if src.len() < 1 + tail_pad {
        return Err(cut("attribute data"));
    }
    // Block data must end exactly where the (padded) tail starts.
    let end = src.len() - tail_pad;
    let tail_at = src.len() - tail;
    let mut last = [0u8; 256];
    last[..stride].copy_from_slice(&src[tail_at..tail_at + stride]);
    let channels = &src[tail_at + stride..];
    if channels.iter().any(|&c| c & 3 == 3) {
        return Err(bad("attribute channel mode"));
    }
    let mut pos = 1;
    // One byte plane per byte of a 4-byte channel, padded to whole groups.
    let mut planes = [[0u8; 256]; 4];
    let mut start = 0;
    while start < count {
        let n = (count - start).min(block_elements(stride));
        let groups = n.div_ceil(GROUP);
        let control = pos;
        pos += channels_len;
        if pos > end {
            return Err(cut("attribute data"));
        }
        for k in (0..stride).step_by(4) {
            let ctrl_byte = if v1 { src[control + k / 4] } else { 0 };
            for (j, plane) in planes.iter_mut().enumerate() {
                let ctrl = (ctrl_byte >> (j * 2)) & 3;
                match ctrl {
                    // Literal bytes, unpadded.
                    3 => {
                        if pos + n > end {
                            return Err(cut("attribute data"));
                        }
                        plane[..n].copy_from_slice(&src[pos..pos + n]);
                        pos += n;
                    }
                    2 => plane[..n].fill(0),
                    _ => {
                        let bits = if v1 { &BITS_V1[ctrl as usize] } else { &BITS_V0 };
                        let header = pos;
                        pos += groups.div_ceil(4);
                        if pos > end {
                            return Err(cut("attribute data"));
                        }
                        for g in 0..groups {
                            let h = (src[header + g / 4] >> ((g % 4) * 2)) & 3;
                            let d = &mut plane[g * GROUP..(g + 1) * GROUP];
                            pos = decode_group(src, pos, end, bits[h as usize], d)?;
                        }
                    }
                }
            }
            let channel = if v1 { channels[k / 4] } else { 0 };
            let prev = &mut last[k..k + 4];
            let rows = out[start * stride..].chunks_exact_mut(stride).take(n);
            match channel & 3 {
                // Byte deltas.
                0 => {
                    for (i, row) in rows.enumerate() {
                        for j in 0..4 {
                            prev[j] = prev[j].wrapping_add(unzigzag8(planes[j][i]));
                        }
                        row[k..k + 4].copy_from_slice(prev);
                    }
                }
                // Two 16-bit deltas.
                1 => {
                    let mut p = [
                        u16::from_le_bytes([prev[0], prev[1]]),
                        u16::from_le_bytes([prev[2], prev[3]]),
                    ];
                    for (i, row) in rows.enumerate() {
                        for h in 0..2 {
                            let v = u16::from_le_bytes([planes[2 * h][i], planes[2 * h + 1][i]]);
                            let d = if v & 1 != 0 { !(v >> 1) } else { v >> 1 };
                            p[h] = p[h].wrapping_add(d);
                            row[k + 2 * h..k + 2 * h + 2].copy_from_slice(&p[h].to_le_bytes());
                        }
                    }
                    prev[..2].copy_from_slice(&p[0].to_le_bytes());
                    prev[2..].copy_from_slice(&p[1].to_le_bytes());
                }
                // A 32-bit XOR delta, rotated.
                _ => {
                    let rot = u32::from(channel >> 4);
                    let mut p = u32::from_le_bytes([prev[0], prev[1], prev[2], prev[3]]);
                    for (i, row) in rows.enumerate() {
                        let v = u32::from_le_bytes([planes[0][i], planes[1][i], planes[2][i], planes[3][i]]);
                        p ^= v.rotate_right(rot);
                        row[k..k + 4].copy_from_slice(&p.to_le_bytes());
                    }
                    prev.copy_from_slice(&p.to_le_bytes());
                }
            }
        }
        start += n;
    }
    if pos != end {
        return Err(bad("attribute data has trailing bytes"));
    }
    Ok(())
}

/// Decodes one group of 16 deltas packed `bits` wide (0, 1, 2, 4 or 8) into `d`.
fn decode_group(src: &[u8], mut pos: usize, end: usize, bits: u8, d: &mut [u8]) -> Result<usize> {
    let truncated = || cut("attribute data");
    match bits {
        0 => d.fill(0),
        8 => {
            if pos + GROUP > end {
                return Err(truncated());
            }
            d.copy_from_slice(&src[pos..pos + GROUP]);
            pos += GROUP;
        }
        _ => {
            let bits = usize::from(bits);
            let sentinel = (1u8 << bits) - 1;
            let packed = GROUP * bits / 8;
            if pos + packed > end {
                return Err(truncated());
            }
            let per = 8 / bits;
            for (i, x) in d.iter_mut().enumerate() {
                let byte = src[pos + i / per];
                // 1-bit groups run least significant bit first, wider ones most significant.
                let shift = if bits == 1 { i % 8 } else { 8 - bits * (i % per + 1) };
                *x = (byte >> shift) & sentinel;
            }
            pos += packed;
            for x in d.iter_mut().filter(|x| **x == sentinel) {
                if pos >= end {
                    return Err(truncated());
                }
                *x = src[pos];
                pos += 1;
            }
        }
    }
    Ok(pos)
}

/// The triangle codec's recent edges and vertices.
struct Fifos {
    edges: [[u32; 2]; 16],
    edge_at: usize,
    verts: [u32; 16],
    vert_at: usize,
}

impl Fifos {
    fn new() -> Self {
        Fifos {
            edges: [[u32::MAX; 2]; 16],
            edge_at: 0,
            verts: [u32::MAX; 16],
            vert_at: 0,
        }
    }
    /// The `i`th most recent edge.
    fn edge_at(&self, i: usize) -> [u32; 2] {
        self.edges[self.edge_at.wrapping_sub(1 + i) & 15]
    }
    /// The `i`th most recent vertex.
    fn vert(&self, i: usize) -> u32 {
        self.verts[self.vert_at.wrapping_sub(1 + i) & 15]
    }
    /// The most recent edge of triangle (a, b, c), and which rotation of it starts there.
    fn find_edge(&self, a: u32, b: u32, c: u32) -> Option<(usize, usize)> {
        (0..16).find_map(|i| match self.edge_at(i) {
            e if e == [a, b] => Some((i, 0)),
            e if e == [b, c] => Some((i, 1)),
            e if e == [c, a] => Some((i, 2)),
            _ => None,
        })
    }
    fn find_vert(&self, v: u32) -> Option<usize> {
        (0..16).find(|&i| self.vert(i) == v)
    }
    fn push_edge(&mut self, a: u32, b: u32) {
        self.edges[self.edge_at] = [a, b];
        self.edge_at = (self.edge_at + 1) & 15;
    }
    fn push_vert(&mut self, v: u32) {
        self.verts[self.vert_at] = v;
        self.vert_at = (self.vert_at + 1) & 15;
    }
}

fn write_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn read_varint(src: &[u8], pos: &mut usize, end: usize) -> Result<u32> {
    let mut v = 0u32;
    for shift in (0..35).step_by(7) {
        let byte = *src.get(*pos).filter(|_| *pos < end).ok_or_else(|| cut("index data"))?;
        *pos += 1;
        v |= u32::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok(v);
        }
    }
    Err(bad("index varint too long"))
}

/// `index` as a zigzag delta from `last`.
fn write_index(out: &mut Vec<u8>, index: u32, last: u32) {
    let d = index.wrapping_sub(last);
    write_varint(out, (d << 1) ^ ((d as i32 >> 31) as u32));
}

fn read_index(src: &[u8], pos: &mut usize, end: usize, last: &mut u32) -> Result<u32> {
    let v = read_varint(src, pos, end)?;
    let delta = if v & 1 != 0 { !(v >> 1) } else { v >> 1 };
    *last = last.wrapping_add(delta);
    Ok(*last)
}

/// Mode 1 (`TRIANGLES`): a triangle list, most triangles in one byte when the list is ordered
/// for the vertex cache and its vertices by first use.
///
/// # Panics
///
/// When `indices` isn't whole triangles.
#[must_use]
pub fn encode_index_buffer(indices: &[u32]) -> Vec<u8> {
    assert!(indices.len().is_multiple_of(3));
    let mut codes = Vec::with_capacity(indices.len() / 3);
    let mut data = Vec::new();
    let mut f = Fifos::new();
    let (mut next, mut last) = (0u32, 0u32);
    for t in indices.chunks_exact(3) {
        if let Some((fe, rot)) = f.find_edge(t[0], t[1], t[2]).filter(|&(fe, _)| fe < 15) {
            // A recent edge and a third vertex: recent, next, last ± 1 or explicit.
            let [a, b, c] = ORDER[rot].map(|k| t[k]);
            let mut fec = match f.find_vert(c) {
                Some(i) if (1..13).contains(&i) => i,
                _ if c == next => {
                    next += 1;
                    0
                }
                _ => 15,
            };
            if fec == 15 {
                if c.wrapping_add(1) == last {
                    fec = 13;
                    last = c;
                }
                if c == last.wrapping_add(1) {
                    fec = 14;
                    last = c;
                }
            }
            codes.push((fe << 4 | fec) as u8);
            if fec == 15 {
                write_index(&mut data, c, last);
                last = c;
            }
            if fec == 0 || fec >= 13 {
                f.push_vert(c);
            }
            f.push_edge(c, b);
            f.push_edge(a, c);
        } else {
            // No recent edge: three vertices, each recent, next or explicit.
            let rot = if t[1] == next {
                1
            } else if t[2] == next {
                2
            } else {
                0
            };
            let [a, b, c] = ORDER[rot].map(|k| t[k]);
            // 0, 1, 2 after other vertices restarts `next` (concatenated streams).
            let reset = [a, b, c] == [0, 1, 2] && next > 0;
            if reset {
                next = 0;
                f.verts = [u32::MAX; 16];
            }
            let (fb, fc) = (f.find_vert(b), f.find_vert(c));
            let mut take = |v: u32, fifo: Option<usize>| match fifo {
                Some(i) if i < 14 => i + 1,
                _ if v == next => {
                    next += 1;
                    0
                }
                _ => 15,
            };
            let fea = take(a, None);
            let feb = take(b, fb);
            let fec = take(c, fc);
            let aux = (feb << 4 | fec) as u8;
            match CODEAUX.iter().position(|&x| x == aux) {
                Some(i) if fea == 0 && i < 14 && !reset => codes.push(0xf0 | i as u8),
                _ => {
                    codes.push(if fea == 0 { 0xfe } else { 0xff });
                    data.push(aux);
                }
            }
            for (v, fe) in [(a, fea), (b, feb), (c, fec)] {
                if fe == 15 {
                    write_index(&mut data, v, last);
                    last = v;
                }
            }
            for (v, fe) in [(a, fea), (b, feb), (c, fec)] {
                if fe == 0 || fe == 15 {
                    f.push_vert(v);
                }
            }
            f.push_edge(b, a);
            f.push_edge(c, b);
            f.push_edge(a, c);
        }
    }
    let mut out = Vec::with_capacity(1 + codes.len() + data.len() + 16);
    out.push(0xe1);
    out.extend_from_slice(&codes);
    out.extend_from_slice(&data);
    out.extend_from_slice(&CODEAUX);
    out
}

/// Decodes mode 1 into `count` indices.
///
/// # Errors
///
/// When `src` is truncated or malformed.
pub fn decode_index_buffer(src: &[u8], count: usize) -> Result<Vec<u32>> {
    let tris = count / 3;
    if !count.is_multiple_of(3) || src.len() < 1 + tris + 16 || src[0] & 0xf0 != 0xe0 {
        return Err(bad("triangle header"));
    }
    if src[0] & 15 != 1 {
        return Err(Error::unsupported(
            "EXT_meshopt_compression: this triangle codec version is not supported",
        ));
    }
    let end = src.len() - 16;
    let table = &src[end..];
    let mut pos = 1 + tris;
    let mut f = Fifos::new();
    let (mut next, mut last) = (0u32, 0u32);
    let mut out = Vec::with_capacity(count);
    for &code in &src[1..=tris] {
        let (hi, lo) = ((code >> 4) as usize, (code & 15) as usize);
        let tri = if hi < 15 {
            let [a, b] = f.edge_at(hi);
            let c = match lo {
                0 => {
                    next += 1;
                    next - 1
                }
                1..=12 => f.vert(lo),
                13 => {
                    last = last.wrapping_sub(1);
                    last
                }
                14 => {
                    last = last.wrapping_add(1);
                    last
                }
                _ => read_index(src, &mut pos, end, &mut last)?,
            };
            if lo == 0 || lo >= 13 {
                f.push_vert(c);
            }
            f.push_edge(c, b);
            f.push_edge(a, c);
            [a, b, c]
        } else {
            let aux = if lo < 14 {
                table[lo]
            } else {
                let x = *src.get(pos).filter(|_| pos < end).ok_or_else(|| cut("index data"))?;
                pos += 1;
                if x == 0 {
                    next = 0;
                }
                x
            };
            let (feb, fec) = ((aux >> 4) as usize, (aux & 15) as usize);
            let fea = if lo == 15 { 15 } else { 0 };
            // FIFO reads and `next` increments happen in a, b, c order, before any pushes.
            let mut pick = |fe: usize| -> Option<u32> {
                match fe {
                    0 => {
                        next += 1;
                        Some(next - 1)
                    }
                    15 => None,
                    i => Some(f.vert(i - 1)),
                }
            };
            let (a, b, c) = (pick(fea), pick(feb), pick(fec));
            let mut explicit = |v: Option<u32>| match v {
                Some(v) => Ok(v),
                None => read_index(src, &mut pos, end, &mut last),
            };
            let (a, b, c) = (explicit(a)?, explicit(b)?, explicit(c)?);
            f.push_vert(a);
            if feb == 0 || feb == 15 {
                f.push_vert(b);
            }
            if fec == 0 || fec == 15 {
                f.push_vert(c);
            }
            f.push_edge(b, a);
            f.push_edge(c, b);
            f.push_edge(a, c);
            [a, b, c]
        };
        out.extend_from_slice(&tri);
    }
    Ok(out)
}

/// Decodes mode 2 (`INDICES`, any index sequence) into `count` indices.
///
/// # Errors
///
/// When `src` is truncated or malformed.
pub fn decode_index_sequence(src: &[u8], count: usize) -> Result<Vec<u32>> {
    if src.len() < 1 + count + 4 || src[0] != 0xd1 {
        return Err(bad("index sequence header"));
    }
    let end = src.len() - 4;
    let mut pos = 1;
    let mut last = [0u32; 2];
    (0..count)
        .map(|_| {
            let v = read_varint(src, &mut pos, end)?;
            let base = (v & 1) as usize;
            let delta = if v & 2 != 0 { !(v >> 2) } else { v >> 2 };
            last[base] = last[base].wrapping_add(delta);
            Ok(last[base])
        })
        .collect()
}

/// Mode 2 (`INDICES`): any index sequence (strips, lines, points, sparse accessor indices),
/// byte for byte meshoptimizer's `meshopt_encodeIndexSequence`. Each index is a delta from one
/// of two baselines (the previous index coded against each), switching baselines whenever the
/// delta from the current one reaches 30.
///
/// # Errors
///
/// When a delta reaches 2^30 either way, which the format can't hold.
pub fn encode_index_sequence(indices: &[u32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(indices.len() + 5);
    out.push(0xd1);
    let mut last = [0u32; 2];
    let mut base = 0;
    for &i in indices {
        if (i.wrapping_sub(last[base]) as i32).unsigned_abs() >= 30 {
            base ^= 1;
        }
        let d = i.wrapping_sub(last[base]) as i32;
        if !(-(1 << 30)..1 << 30).contains(&d) {
            return Err(bad("index sequence delta out of range"));
        }
        let zz = ((d << 1) ^ (d >> 31)) as u32;
        write_varint(&mut out, (zz << 1) | base as u32);
        last[base] = i;
    }
    out.extend_from_slice(&[0; 4]);
    Ok(out)
}

/// Bits an index below `vertex_count` takes, at least 1: the index bounds' measure.
fn index_bits(vertex_count: usize) -> usize {
    let mut bits = 1;
    while bits < 32 && vertex_count > 1 << bits {
        bits += 1;
    }
    bits
}

/// The most bytes [`encode_index_buffer`] writes for `index_count` indices (whole triangles)
/// below `vertex_count`: meshoptimizer's `meshopt_encodeIndexBufferBound`.
///
/// # Panics
///
/// When `index_count` isn't a multiple of 3.
#[must_use]
pub fn encode_index_buffer_bound(index_count: usize, vertex_count: usize) -> usize {
    assert!(index_count.is_multiple_of(3), "index count isn't whole triangles");
    // At worst 2 code bytes and 3 varint indices per triangle.
    1 + index_count / 3 * (2 + 3 * (index_bits(vertex_count) + 1).div_ceil(7)) + 16
}

/// The most bytes [`encode_index_sequence`] writes for `index_count` indices below
/// `vertex_count`: meshoptimizer's `meshopt_encodeIndexSequenceBound`.
#[must_use]
pub fn encode_index_sequence_bound(index_count: usize, vertex_count: usize) -> usize {
    // At worst one varint per index: the zigzag delta and the baseline bit.
    1 + index_count * (index_bits(vertex_count) + 2).div_ceil(7) + 4
}

/// `v` in -1 to 1 as a `bits`-wide signed normalized integer, rounded half away from zero:
/// meshoptimizer's `meshopt_quantizeSnorm`.
fn snorm(v: f32, bits: u32) -> i32 {
    let scale = ((1 << (bits - 1)) - 1) as f32;
    let round = if v >= 0.0 { 0.5 } else { -0.5 };
    let v = if v >= -1.0 { v } else { -1.0 };
    let v = if v <= 1.0 { v } else { 1.0 };
    (v * scale + round) as i32
}

/// `v` in 0 to 1 as a `bits`-wide unsigned normalized integer, rounded half up:
/// meshoptimizer's `meshopt_quantizeUnorm`.
fn unorm(v: f32, bits: u32) -> i32 {
    let scale = ((1 << bits) - 1) as f32;
    let v = if v >= 0.0 { v } else { 0.0 };
    let v = if v <= 1.0 { v } else { 1.0 };
    (v * scale + 0.5) as i32
}

/// Four integer components as a filter element of `stride` bytes: bytes when 4, little-endian
/// 16-bit values when 8.
fn push_components(out: &mut Vec<u8>, c: [i32; 4], stride: usize) {
    for v in c {
        if stride == 4 {
            out.push(v as u8);
        } else {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
    }
}

/// The octahedral filter's 8-bit input for unit vector `n` and fourth component `w`
/// (passed through; ±1 for a tangent's handedness). The same bytes as
/// [`encode_filter_oct`] at stride 4, 8 bits.
#[must_use]
pub fn encode_oct8(n: [f32; 3], w: f32) -> [u8; 4] {
    let e = encode_filter_oct(&[[n[0], n[1], n[2], w]], 4, 8);
    [e[0], e[1], e[2], e[3]]
}

/// The `OCTAHEDRAL` filter's input for unit vectors `data` (x, y, z and a fourth component
/// passed through, ±1 for a tangent's handedness), byte for byte meshoptimizer's
/// `meshopt_encodeFilterOct`: elements of `stride` bytes (4: signed bytes, 8: signed 16-bit),
/// the octahedral coordinates `bits` wide. Vectors needn't be unit length; a zero vector
/// encodes as +Z.
///
/// # Panics
///
/// When `stride` isn't 4 or 8, or `bits` isn't from 2 to 8 (stride 4) or 16 (stride 8).
#[must_use]
pub fn encode_filter_oct(data: &[[f32; 4]], stride: usize, bits: u32) -> Vec<u8> {
    assert!(stride == 4 || stride == 8, "octahedral filter stride");
    assert!(
        (2..=16).contains(&bits) && (stride == 8 || bits <= 8),
        "octahedral filter bits"
    );
    let mut out = Vec::with_capacity(data.len() * stride);
    for &[x, y, z, w] in data {
        let l = x.abs() + y.abs() + z.abs();
        let s = if l == 0.0 { 0.0 } else { 1.0 / l };
        let (x, y) = (x * s, y * s);
        let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
        let u = if z >= 0.0 { x } else { (1.0 - y.abs()) * sign(x) };
        let v = if z >= 0.0 { y } else { (1.0 - x.abs()) * sign(y) };
        let c = [
            snorm(u, bits),
            snorm(v, bits),
            snorm(1.0, bits),
            snorm(w, stride as u32 * 2),
        ];
        push_components(&mut out, c, stride);
    }
    out
}

/// The `QUATERNION` filter's input for unit quaternions `data` (x, y, z, w), byte for byte
/// meshoptimizer's `meshopt_encodeFilterQuat`: elements of 8 bytes, the three smallest
/// components `bits` wide and the largest one's index in the fourth. A quaternion and its
/// negation encode alike.
///
/// # Panics
///
/// When `bits` isn't from 4 to 16.
#[must_use]
pub fn encode_filter_quat(data: &[[f32; 4]], bits: u32) -> Vec<u8> {
    assert!((4..=16).contains(&bits), "quaternion filter bits");
    let mut out = Vec::with_capacity(data.len() * 8);
    for q in data {
        let mut qc = 0;
        for i in 1..4 {
            if q[i].abs() > q[qc].abs() {
                qc = i;
            }
        }
        // Double cover: flip so the largest component is positive.
        let scale = std::f32::consts::SQRT_2 * if q[qc] < 0.0 { -1.0 } else { 1.0 };
        let c = |i: usize| snorm(q[(qc + i) & 3] * scale, bits);
        push_components(&mut out, [c(1), c(2), c(3), (snorm(1.0, bits) & !3) | qc as i32], 8);
    }
    out
}

/// How [`encode_filter_exp`] picks exponents, meshoptimizer's `meshopt_EncodeExpMode`.
///
/// Non-exhaustive: meshoptimizer may add modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExpMode {
    /// Each component its own exponent, so the most precision per value. A zero keeps the
    /// exponent the same component had in the element before, which compresses better.
    Separate,
    /// One exponent per element (the largest its components need), so every component of an
    /// element shares the same absolute error.
    SharedVector,
    /// One exponent per component position across all elements (the largest any needs), so the
    /// stream compresses best.
    SharedComponent,
    /// Like `Separate`, but no exponent below 0, so values under 1 lose precision to an
    /// absolute error of 2^-(bits-1): for values whose small magnitudes needn't be exact.
    Clamped,
}

/// The `EXPONENTIAL` filter's input for floats `data` (elements of `stride` bytes, `stride / 4`
/// components each), byte for byte meshoptimizer's `meshopt_encodeFilterExp`: each component a
/// little-endian 32-bit value holding an 8-bit exponent and a 24-bit signed mantissa, `bits` of
/// which carry precision, the exponents picked per `mode`.
///
/// # Panics
///
/// When `stride` isn't a multiple of 4 from 4 to 256, `data` isn't whole elements, or `bits`
/// isn't from 1 to 24.
#[must_use]
pub fn encode_filter_exp(data: &[f32], stride: usize, bits: u32, mode: ExpMode) -> Vec<u8> {
    /// The lowest exponent picked: smaller magnitudes round to zero.
    const MIN_EXP: i32 = -100;
    assert!(
        stride > 0 && stride <= 256 && stride.is_multiple_of(4),
        "exponential filter stride"
    );
    assert!((1..=24).contains(&bits), "exponential filter bits");
    let width = stride / 4;
    assert!(
        data.len().is_multiple_of(width),
        "exponential filter data isn't whole elements"
    );
    // The exponent that puts `v` in [0.5, 1); zeros and subnormals land low and get clamped.
    let log2 = |v: f32| ((v.to_bits() >> 23) & 0xff) as i32 - 126;
    let exp2 = |e: i32| f32::from_bits(((e + 127) as u32) << 23);
    let mut component = [if mode == ExpMode::Separate { 0 } else { MIN_EXP }; 64];
    if mode == ExpMode::SharedComponent {
        for v in data.chunks_exact(width) {
            for (c, &x) in component.iter_mut().zip(v) {
                *c = (*c).max(log2(x));
            }
        }
    }
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data.chunks_exact(width) {
        let shared = v.iter().fold(MIN_EXP, |e, &x| e.max(log2(x)));
        for (c, &x) in component.iter_mut().zip(v) {
            match mode {
                ExpMode::Separate if x != 0.0 => *c = log2(x).max(MIN_EXP),
                ExpMode::Clamped => *c = log2(x).max(0),
                _ => {}
            }
            let e = if mode == ExpMode::SharedVector { shared } else { *c } - (bits as i32 - 1);
            let m = ((x * exp2(-e) + if x >= 0.0 { 0.5 } else { -0.5 }) as i32).min(0x7f_ffff);
            out.extend_from_slice(&((m as u32 & 0xff_ffff) | (e as u32) << 24).to_le_bytes());
        }
    }
    out
}

/// The `COLOR` filter's input (`KHR_meshopt_compression` only) for RGBA colors `data` in 0 to 1,
/// byte for byte meshoptimizer's `meshopt_encodeFilterColor`: elements of `stride` bytes (4:
/// bytes, 8: little-endian 16-bit), each color quantized to `bits` per channel and stored as
/// `YCoCg` with Co and Cg truncated (so it decodes with integers, each RGB channel within one
/// quantization step) and alpha `bits - 1` wide with the top bit marking the width. That costs
/// up to a step and a half of `bits` per channel, but compresses better than RGBA.
///
/// # Panics
///
/// When `stride` isn't 4 or 8, or `bits` isn't from 2 to 8 (stride 4) or 16 (stride 8).
#[must_use]
pub fn encode_filter_color(data: &[[f32; 4]], stride: usize, bits: u32) -> Vec<u8> {
    assert!(stride == 4 || stride == 8, "color filter stride");
    assert!(
        (2..=16).contains(&bits) && (stride == 8 || bits <= 8),
        "color filter bits"
    );
    let mut out = Vec::with_capacity(data.len() * stride);
    for &[r, g, b, a] in data {
        let [r, g, b] = [r, g, b].map(|v| unorm(v, bits));
        // YCoCg with Co and Cg truncated (not the lossless YCoCg-R), so the decoder needs only integers.
        let co = (r - b) / 2;
        let t = b + co;
        let cg = (g - t) / 2;
        let alpha = (unorm(a, bits) >> 1) | (1 << (bits - 1));
        push_components(&mut out, [t + cg, co, cg, alpha], stride);
    }
    out
}

/// A meshopt filter: how decoded attribute data was transformed before it was encoded, undone
/// by [`apply_filter`]. Parses from, and displays as, its name in the glTF extensions
/// (`"OCTAHEDRAL"`).
///
/// Non-exhaustive: the meshopt extensions may add filters.
///
/// ```
/// use polysmelt::meshopt::Filter;
///
/// let filter: Filter = "QUATERNION".parse()?;
/// assert_eq!(filter, Filter::Quaternion);
/// assert_eq!(Filter::Color.to_string(), "COLOR");
/// assert!("SPLINE".parse::<Filter>().is_err());
/// # Ok::<(), polysmelt::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Filter {
    /// `NONE`: the data as it is.
    #[default]
    None,
    /// `OCTAHEDRAL`: unit vectors, two components each (4 or 8 byte elements).
    Octahedral,
    /// `QUATERNION`: unit quaternions, three components and an index (8 byte elements).
    Quaternion,
    /// `EXPONENTIAL`: floats as an exponent and mantissa each (elements a multiple of 4 bytes).
    Exponential,
    /// `COLOR` (`KHR_meshopt_compression` only): RGBA colors as `YCoCg` (4 or 8 byte elements).
    Color,
}

impl Filter {
    /// The filter's name in the glTF extensions: `"OCTAHEDRAL"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Filter::None => "NONE",
            Filter::Octahedral => "OCTAHEDRAL",
            Filter::Quaternion => "QUATERNION",
            Filter::Exponential => "EXPONENTIAL",
            Filter::Color => "COLOR",
        }
    }
}

impl std::fmt::Display for Filter {
    /// [`Filter::as_str`].
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Filter {
    type Err = Error;

    /// A filter by its name in the glTF extensions, exactly (`"OCTAHEDRAL"`).
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "NONE" => Filter::None,
            "OCTAHEDRAL" => Filter::Octahedral,
            "QUATERNION" => Filter::Quaternion,
            "EXPONENTIAL" => Filter::Exponential,
            "COLOR" => Filter::Color,
            _ => return Err(bad(&format!("unknown filter {}", crate::shown(s.as_bytes())))),
        })
    }
}

/// Undoes `filter` on decoded attribute data in place, elements of `stride` bytes.
///
/// # Errors
///
/// When `stride` doesn't suit the filter.
pub fn apply_filter(filter: Filter, data: &mut [u8], stride: usize) -> Result<()> {
    match filter {
        Filter::None => {}
        Filter::Octahedral if stride == 4 => {
            for e in data.chunks_exact_mut(4) {
                let [x, y, z] = octahedral(
                    [f32::from(e[0] as i8), f32::from(e[1] as i8), f32::from(e[2] as i8)],
                    127.0,
                );
                e[..3].copy_from_slice(&[x as i8 as u8, y as i8 as u8, z as i8 as u8]);
            }
        }
        Filter::Octahedral if stride == 8 => {
            for e in data.chunks_exact_mut(8) {
                let c = |i: usize| f32::from(i16::from_le_bytes([e[2 * i], e[2 * i + 1]]));
                let out = octahedral([c(0), c(1), c(2)], 32767.0);
                for (i, v) in out.into_iter().enumerate() {
                    e[2 * i..2 * i + 2].copy_from_slice(&(v as i16).to_le_bytes());
                }
            }
        }
        Filter::Quaternion if stride == 8 => {
            for e in data.chunks_exact_mut(8) {
                let c = |i: usize| i16::from_le_bytes([e[2 * i], e[2 * i + 1]]);
                let range = std::f32::consts::FRAC_1_SQRT_2;
                let one = f32::from(c(3) | 3);
                let x = f32::from(c(0)) / one * range;
                let y = f32::from(c(1)) / one * range;
                let z = f32::from(c(2)) / one * range;
                let w = (1.0 - x * x - y * y - z * z).max(0.0).sqrt();
                let max = (c(3) & 3) as usize;
                let mut out = [0i16; 4];
                for (k, v) in [(1, x), (2, y), (3, z), (0, w)] {
                    out[(max + k) % 4] = (v * 32767.0).round() as i16;
                }
                for (i, v) in out.into_iter().enumerate() {
                    e[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        Filter::Exponential if stride.is_multiple_of(4) => {
            for e in data.chunks_exact_mut(4) {
                let v = i32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                let m = (v << 8) >> 8;
                let f = (m as f32) * 2f32.powi(v >> 24);
                e.copy_from_slice(&f.to_le_bytes());
            }
        }
        Filter::Color if stride == 4 => {
            for e in data.chunks_exact_mut(4) {
                let rgba = ycocg(
                    [e[0].into(), i32::from(e[1] as i8), i32::from(e[2] as i8), e[3].into()],
                    255.0,
                );
                for (o, v) in e.iter_mut().zip(rgba) {
                    *o = v as u8;
                }
            }
        }
        Filter::Color if stride == 8 => {
            for e in data.chunks_exact_mut(8) {
                let c = |i: usize| u16::from_le_bytes([e[2 * i], e[2 * i + 1]]);
                let rgba = ycocg(
                    [c(0).into(), i32::from(c(1) as i16), i32::from(c(2) as i16), c(3).into()],
                    65535.0,
                );
                for (i, v) in rgba.into_iter().enumerate() {
                    e[2 * i..2 * i + 2].copy_from_slice(&(v as u16).to_le_bytes());
                }
            }
        }
        _ => return Err(bad(&format!("filter {filter} with stride {stride}"))),
    }
    Ok(())
}

/// The color filter (`KHR_meshopt_compression`): Y, Co, Cg and a marked alpha, `bits` wide, to
/// RGBA in units of the component's maximum (255 or 65535, by the element size).
fn ycocg([y, co, cg, alpha]: [i32; 4], max: f32) -> [i32; 4] {
    // The alpha's top set bit marks its width; `as` is all ones up to it.
    let mut scale = alpha;
    scale |= scale >> 1;
    scale |= scale >> 2;
    scale |= scale >> 4;
    scale |= scale >> 8;
    let a = ((alpha << 1) & scale) | (alpha & 1);
    let ss = if scale == 0 { 0.0 } else { max / scale as f32 };
    [y + co - cg, y + cg, y - co - cg, a].map(|v| (v as f32 * ss + 0.5) as i32)
}

/// The octahedral filter: components in units of `one` (the third input) to a unit vector in
/// units of `max`.
fn octahedral(c: [f32; 3], max: f32) -> [f32; 3] {
    let (mut x, mut y) = (c[0] / c[2], c[1] / c[2]);
    let z = 1.0 - x.abs() - y.abs();
    let t = z.min(0.0);
    x += if x >= 0.0 { t } else { -t };
    y += if y >= 0.0 { t } else { -t };
    let len = (x * x + y * y + z * z).sqrt();
    [x, y, z].map(|v| (v / len * max).round())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::unit as normalize;

    use crate::test_rng::Pcg32;
    use std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn attributes_roundtrip() {
        let mut r = Pcg32::new(3);
        for (count, stride) in [(0, 4), (1, 4), (17, 8), (300, 12), (1000, 16), (40, 256), (700, 64)] {
            // Smooth data with occasional jumps exercises every group mode.
            let mut data = vec![0u8; count * stride];
            for i in 1..count {
                for k in 0..stride {
                    let prev = data[(i - 1) * stride + k];
                    let step = match r.below(10) {
                        0 => r.below(256) as u8,
                        1..=3 => r.below(9) as u8,
                        _ => 0,
                    };
                    data[i * stride + k] = prev.wrapping_add(step);
                }
            }
            let enc = encode_vertex_buffer(&data, stride);
            assert_eq!(
                decode_vertex_buffer(&enc, count, stride).unwrap(),
                data,
                "{count}x{stride}"
            );
        }
    }

    #[test]
    fn triangles_roundtrip() {
        // A grid in cache order, a shuffled one, two streams concatenated (reset) and
        // strip-like and far-apart explicit indices.
        let n = 30u32;
        let mut grid = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                grid.extend_from_slice(&[a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        let mut r = Pcg32::new(9);
        let mut shuffled: Vec<[u32; 3]> = grid.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, r.below(i as u32 + 1) as usize);
        }
        let shuffled = shuffled.concat();
        let concat = [&grid[..], &grid[..]].concat();
        let wild: Vec<u32> = (0..300).map(|_| r.next_u32() >> r.below(32)).collect();
        let strip: Vec<u32> = (0..100u32).flat_map(|i| [i + 2, i + 1, i]).collect();
        for idx in [&grid, &shuffled, &concat, &wild, &strip, &vec![]] {
            // Triangles keep their order and winding, but may start at another vertex.
            let canon = |v: &[u32]| -> Vec<[u32; 3]> {
                v.chunks_exact(3)
                    .map(|t| {
                        let r = (0..3).min_by_key(|&k| t[k]).unwrap();
                        [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
                    })
                    .collect()
            };
            let enc = encode_index_buffer(idx);
            assert_eq!(canon(&decode_index_buffer(&enc, idx.len()).unwrap()), canon(idx));
        }
        // Ordered input costs about a byte per triangle.
        assert!(encode_index_buffer(&grid).len() < grid.len() / 3 * 3 / 2);
    }

    /// The worked example in the extension's specification (4-bit sentinel group).
    #[test]
    fn spec_group_example() {
        let src = [0x17, 0x5f, 0xf0, 0xbc, 0x77, 0xa9, 0x21, 0x00, 0x34, 0xb5];
        let mut d = [0u8; 16];
        assert_eq!(decode_group(&src, 0, src.len(), 4, &mut d).unwrap(), src.len());
        let got: Vec<i8> = d.iter().map(|&x| unzigzag8(x) as i8).collect();
        assert_eq!(got, [-1, -4, -3, 26, -91, 0, -6, 6, -4, -4, 5, -5, 1, -1, 0, 0]);
    }

    #[test]
    fn index_sequence_decodes() {
        // 5, 3 (baseline 0), 100 (baseline 1), 4 (baseline 0).
        let mut src = vec![0xd1];
        for (delta, base) in [(5i32, 0u32), (-2, 0), (100, 1), (1, 0)] {
            let z = ((delta << 1) ^ (delta >> 31)) as u32;
            write_varint(&mut src, z << 1 | base);
        }
        src.extend_from_slice(&[0; 4]);
        assert_eq!(decode_index_sequence(&src, 4).unwrap(), vec![5, 3, 100, 4]);
    }

    /// meshoptimizer's `PV` test vertices (u16 x3, u8 x2, u16 x2) as bytes.
    fn pv_bytes() -> Vec<u8> {
        let pv: [([u16; 3], [u8; 2], [u16; 2]); 4] = [
            ([0, 0, 0], [0, 0], [0, 0]),
            ([300, 0, 0], [0, 0], [500, 0]),
            ([0, 300, 0], [0, 0], [0, 500]),
            ([300, 300, 0], [0, 0], [500, 500]),
        ];
        let mut out = Vec::new();
        for (p, n, t) in pv {
            out.extend(p.iter().flat_map(|v| v.to_le_bytes()));
            out.extend(n);
            out.extend(t.iter().flat_map(|v| v.to_le_bytes()));
        }
        out
    }

    /// meshoptimizer's own encodings (`demo/tests.cpp`): version 0, version 1, and a version 1
    /// stream from a custom encoder that uses every control and channel mode.
    #[test]
    fn reference_vertex_streams() {
        let v0: &[u8] = &[
            0xa0, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x58, 0x57, 0x58, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00,
            0x00, 0x58, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x17, 0x18,
            0x17, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00, 0x00, 0x17, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let v1: &[u8] = &[
            0xa1, 0xee, 0xaa, 0xee, 0x00, 0x4b, 0x4b, 0x4b, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x7d, 0x7d, 0x7d, 0x00, 0x00,
            0x7d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x62, 0x00, 0x62,
        ];
        let v1_custom: &[u8] = &[
            0xa1, 0xd4, 0x94, 0xd4, 0x01, 0x0e, 0x00, 0x58, 0x57, 0x58, 0x02, 0x02, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x03, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x7d, 0x7d, 0x7d, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x7d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x62,
        ];
        let want = pv_bytes();
        // meshoptimizer's encoder wrote these two: version 0 (any level) and version 1 at level 3.
        for level in 0..=3 {
            assert_eq!(encode_vertex_buffer_level(&want, 12, level, 0), v0, "v0 level {level}");
        }
        assert_eq!(encode_vertex_buffer(&want, 12), v0);
        assert_eq!(encode_vertex_buffer_level(&want, 12, 3, 1), v1);
        for (name, src) in [("v0", v0), ("v1", v1), ("v1 custom", v1_custom)] {
            assert_eq!(decode_vertex_buffer(src, 4, 12).unwrap(), want, "{name}");
            // One byte more or less is a malformed stream.
            let mut long = src.to_vec();
            long.push(0);
            assert!(decode_vertex_buffer(&long, 4, 12).is_err(), "{name} long");
            assert!(
                decode_vertex_buffer(&src[..src.len() - 1], 4, 12).is_err(),
                "{name} short"
            );
        }
    }

    /// meshoptimizer's 16-bit delta stream (`decodeVertexV1Deltas`).
    #[test]
    fn reference_vertex_deltas() {
        let want: [u16; 64] = [
            248, 248, 240, 240, 249, 250, 243, 244, 250, 252, 246, 248, 251, 254, 249, 252, 252, 256, 252, 256, 253,
            258, 255, 260, 254, 260, 258, 264, 255, 262, 261, 268, 256, 264, 264, 272, 257, 262, 267, 268, 258, 260,
            270, 264, 259, 258, 273, 260, 260, 256, 276, 256, 261, 254, 279, 252, 262, 252, 282, 248, 263, 250, 285,
            244,
        ];
        let src: &[u8] = &[
            0xa1, 0x99, 0x99, 0x01, 0x2a, 0xaa, 0xaa, 0xaa, 0x02, 0x04, 0x44, 0x44, 0x44, 0x43, 0x33, 0x33, 0x33, 0x02,
            0x06, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x02, 0x08, 0x88, 0x88, 0x88, 0x87, 0x77, 0x77, 0x77, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x00, 0xf8, 0x00, 0xf0,
            0x00, 0xf0, 0x00, 0x01, 0x01,
        ];
        let got = decode_vertex_buffer(src, 16, 8).unwrap();
        // meshoptimizer's level 2 encoding, which level 3 matches.
        assert_eq!(encode_vertex_buffer_level(&got, 8, 2, 1), src);
        assert_eq!(encode_vertex_buffer_level(&got, 8, 3, 1), src);
        let got: Vec<u16> = got.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(got, want);
    }

    /// meshoptimizer's color filter vectors (`encodeFilterColor8`/`12`), decoded.
    #[test]
    fn color_filter() {
        let want = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.25],
            [0.4, 0.4, 0.4, 0.75],
        ];
        let mut c8 = vec![
            0x40, 0x7f, 0xc1, 0xff, 0x7f, 0x00, 0x7f, 0xc0, 0x40, 0x81, 0xc0, 0xa0, 0x66, 0x00, 0x00, 0xdf,
        ];
        apply_filter(Filter::Color, &mut c8, 4).unwrap();
        for (i, &v) in c8.iter().enumerate() {
            assert!((f32::from(v) / 255.0 - want[i / 4][i % 4]).abs() < 1e-2, "{i}: {v}");
        }
        assert!(c8[12] == c8[13] && c8[13] == c8[14]);
        let c12: [u16; 16] = [
            0x0400, 0x07ff, 0xfc01, 0x0fff, 0x07ff, 0x0000, 0x07ff, 0x0c00, 0x0400, 0xf801, 0xfc00, 0x0a00, 0x0666,
            0x0000, 0x0000, 0x0dff,
        ];
        let mut b: Vec<u8> = c12.iter().flat_map(|v| v.to_le_bytes()).collect();
        apply_filter(Filter::Color, &mut b, 8).unwrap();
        for (i, c) in b.chunks_exact(2).enumerate() {
            let v = f32::from(u16::from_le_bytes([c[0], c[1]])) / 65535.0;
            assert!((v - want[i / 4][i % 4]).abs() < 1e-3, "{i}: {v}");
        }
    }

    #[test]
    fn index_sequence_roundtrip() {
        // meshoptimizer's reference encoding decodes, and ours round-trips.
        let reference = [
            0xd1, 0x00, 0x04, 0xcd, 0x01, 0x04, 0x07, 0x98, 0x1f, 0x00, 0x00, 0x00, 0x00,
        ];
        let seq = [0, 1, 51, 2, 49, 1000];
        assert_eq!(decode_index_sequence(&reference, 6).unwrap(), seq);
        assert_eq!(encode_index_sequence(&seq).unwrap(), reference);
        let mut r = Pcg32::new(5);
        let wild: Vec<u32> = (0..500).map(|_| r.next_u32() >> 2).collect();
        let strip: Vec<u32> = (0..300).collect();
        for idx in [&seq[..], &wild, &strip, &[]] {
            let enc = encode_index_sequence(idx).unwrap();
            assert_eq!(decode_index_sequence(&enc, idx.len()).unwrap(), idx);
        }
        assert!(encode_index_sequence(&[0, 1 << 31]).is_err());
    }

    #[test]
    fn octahedral_roundtrip() {
        let mut r = Pcg32::new(1);
        for _ in 0..2000 {
            let v = [r.f32() - 0.5, r.f32() - 0.5, r.f32() - 0.5];
            let n = normalize(v).unwrap_or([0.0, 0.0, 1.0]);
            let mut e = encode_oct8(n, -1.0).to_vec();
            apply_filter(Filter::Octahedral, &mut e, 4).unwrap();
            let d = normalize([f32::from(e[0] as i8), f32::from(e[1] as i8), f32::from(e[2] as i8)]).unwrap();
            let dot: f32 = (0..3).map(|i| d[i] * n[i]).sum();
            assert!(dot > 0.999, "{n:?} -> {d:?}");
            assert_eq!(e[3] as i8, -127);
        }
    }

    /// Streams from meshoptimizer's encoder (its JavaScript package's tests) re-encode byte for
    /// byte from their decoded data, at the levels it used.
    #[test]
    fn encoder_reproduces_reference_streams() {
        let more: &[u8] = &[
            0xa0, 0x00, 0x01, 0x2a, 0xaa, 0xaa, 0xaa, 0x02, 0x04, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x03, 0x00,
            0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let mode2: &[u8] = &[
            0xa0, 0x02, 0x08, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x02, 0x0a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
            0xaa, 0x02, 0x0c, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0x02, 0x0e, 0xee, 0xee, 0xee, 0xee, 0xee, 0xee,
            0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        // Version 1 at level 3 with 32-bit XOR channels, rotated.
        let xor: &[u8] = &[
            0xa1, 0xab, 0xab, 0xfa, 0xff, 0x00, 0x02, 0x02, 0x02, 0x00, 0xa5, 0xc2, 0x3a, 0x00, 0xab, 0xf9, 0x57, 0x00,
            0x95, 0x42, 0x85, 0x00, 0xdd, 0xca, 0x6d, 0x00, 0xac, 0x1a, 0x50, 0x00, 0x5c, 0x99, 0xfe, 0x00, 0x4d, 0x0c,
            0x39, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x00, 0x80, 0x07, 0x0c, 0x22,
            0x26, 0x3b, 0x86, 0x00, 0x00, 0x12, 0x00,
        ];
        let xor_data: [u32; 16] = [
            0,
            112,
            201_818_112,
            2_252_023_330,
            1,
            29,
            1_188_167_680,
            1_600_748_723,
            2,
            126,
            1_739_489_280,
            1_696_368_920,
            3,
            155,
            621_084_672,
            1_218_163_169,
        ];
        let xor_bytes: Vec<u8> = xor_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(decode_vertex_buffer(xor, 4, 16).unwrap(), xor_bytes);
        assert_eq!(encode_vertex_buffer_level(&xor_bytes, 16, 3, 1), xor);
        for (name, src) in [("more", more), ("mode2", mode2)] {
            let data = decode_vertex_buffer(src, 16, 4).unwrap();
            assert_eq!(encode_vertex_buffer(&data, 4), src, "{name}");
        }
        // meshoptimizer's triangle encodings (`kIndexDataV1` and the package's `More` case).
        let tricky = [0, 1, 2, 2, 1, 3, 0, 1, 2, 2, 1, 5, 2, 1, 4];
        let tricky_enc: &[u8] = &[
            0xe1, 0xf0, 0x10, 0xfe, 0x1f, 0x3d, 0x00, 0x0a, 0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89,
            0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
        ];
        assert_eq!(encode_index_buffer(&tricky), tricky_enc);
        let more_ib = [0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9];
        let more_ib_enc: &[u8] = &[
            0xe1, 0xf0, 0x10, 0xfe, 0xff, 0xf0, 0x0c, 0xff, 0x02, 0x02, 0x02, 0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9,
            0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
        ];
        assert_eq!(encode_index_buffer(&more_ib), more_ib_enc);
    }

    /// Attribute data of assorted shapes, every level and version: decodes back exactly and fits
    /// the bound.
    #[test]
    fn attribute_levels_roundtrip() {
        let mut r = Pcg32::new(11);
        for stride in [4, 8, 12, 16, 20, 32, 64, 68, 128, 252, 256] {
            for kind in 0..6 {
                for _ in 0..2 {
                    let count = [0, 1, 2, 15, 16, 17, 100, 257, 600][r.below(9) as usize].min(40_000 / stride);
                    let mut data = vec![0u8; count * stride];
                    let constant: Vec<u8> = (0..stride).map(|_| r.below(256) as u8).collect();
                    for (w, c) in data.chunks_exact_mut(4).enumerate() {
                        let v: u32 = match kind {
                            // Random bytes.
                            0 => r.next_u32(),
                            // Smooth floats with noise.
                            1 => ((w as f32 * 0.01).sin() * 10.0 + r.f32() * 1e-3).to_bits(),
                            // Mostly zero.
                            2 => u32::from(r.below(13) == 0) * r.next_u32(),
                            // The same element over and over.
                            3 => u32::from_le_bytes(constant[w * 4 % stride..][..4].try_into().unwrap()),
                            // Small increments.
                            4 => w as u32 * 3 + r.below(4),
                            // A few bits set high up.
                            _ => r.below(16) << r.below(29),
                        };
                        c.copy_from_slice(&v.to_le_bytes());
                    }
                    let bound = encode_vertex_buffer_bound(count, stride);
                    for version in 0..=1 {
                        for level in 0..=3 {
                            let enc = encode_vertex_buffer_level(&data, stride, level, version);
                            assert_eq!(enc[0], 0xa0 | version);
                            assert!(enc.len() <= bound, "{count}x{stride} kind {kind} v{version} l{level}");
                            assert_eq!(
                                decode_vertex_buffer(&enc, count, stride).unwrap(),
                                data,
                                "{count}x{stride} kind {kind} v{version} l{level}"
                            );
                        }
                        assert_eq!(
                            encode_vertex_buffer_level(&data, stride, 9, version),
                            encode_vertex_buffer_level(&data, stride, 3, version)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn index_bounds_hold() {
        let mut r = Pcg32::new(4);
        for vertex_count in [1usize, 2, 3, 100, 128, 129, 70_000, 1 << 24, 1 << 31] {
            for _ in 0..20 {
                let n = r.below(200) as usize * 3;
                let idx: Vec<u32> = (0..n).map(|_| r.below(vertex_count as u32)).collect();
                let enc = encode_index_buffer(&idx);
                assert!(
                    enc.len() <= encode_index_buffer_bound(n, vertex_count),
                    "{vertex_count}"
                );
                if let Ok(enc) = encode_index_sequence(&idx) {
                    assert!(
                        enc.len() <= encode_index_sequence_bound(n, vertex_count),
                        "{vertex_count}"
                    );
                }
            }
        }
        assert_eq!(encode_index_buffer_bound(0, 0), 17);
        assert_eq!(encode_index_sequence_bound(0, 0), 5);
        // 2 bits (3 with the edge bit) fit one varint byte; 2 code bytes per triangle.
        assert_eq!(encode_index_buffer_bound(3, 4), 1 + 5 + 16);
        assert_eq!(encode_vertex_buffer_bound(0, 4), 33);
    }

    /// Floats as little-endian 32-bit words.
    fn words(b: &[u8]) -> Vec<u32> {
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Little-endian 16-bit values.
    fn halves(b: &[u8]) -> Vec<u16> {
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    /// meshoptimizer's filter encoder vectors (`demo/tests.cpp`).
    #[test]
    fn reference_filter_encodings() {
        let oct = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [FRAC_1_SQRT_2, 0.0, 0.707_168, 1.0],
            [-FRAC_1_SQRT_2, 0.0, -0.707_168, 1.0],
        ];
        assert_eq!(
            encode_filter_oct(&oct, 4, 8),
            [
                0x7f, 0, 0x7f, 0, 0, 0x81, 0x7f, 0, 0x3f, 0, 0x7f, 0x7f, 0x81, 0x40, 0x7f, 0x7f
            ]
        );
        assert_eq!(
            halves(&encode_filter_oct(&oct, 8, 12)),
            [
                0x7ff, 0, 0x7ff, 0, 0, 0xf801, 0x7ff, 0, 0x3ff, 0, 0x7ff, 0x7fff, 0xf801, 0x400, 0x7ff, 0x7fff
            ]
        );
        let quat = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [FRAC_1_SQRT_2, 0.0, 0.0, 0.707_168],
            [-FRAC_1_SQRT_2, 0.0, 0.0, -0.707_168],
        ];
        assert_eq!(
            halves(&encode_filter_quat(&quat, 12)),
            [0, 0, 0, 0x7fc, 0, 0, 0, 0x7fd, 0x7ff, 0, 0, 0x7ff, 0x7ff, 0, 0, 0x7ff]
        );
        let exp = [1.0, -23.4, -0.1, 11.0];
        for (mode, want) in [
            (ExpMode::Separate, [0xf300_2000, 0xf7ff_d133, 0xefff_cccd, 0xf600_2c00]),
            (
                ExpMode::SharedVector,
                [0xf700_0200, 0xf7ff_d133, 0xf6ff_ff9a, 0xf600_2c00],
            ),
            (
                ExpMode::SharedComponent,
                [0xf300_2000, 0xf7ff_d133, 0xf3ff_fccd, 0xf700_1600],
            ),
            (ExpMode::Clamped, [0xf300_2000, 0xf7ff_d133, 0xf2ff_f99a, 0xf600_2c00]),
        ] {
            let mut enc = encode_filter_exp(&exp, 8, 15, mode);
            assert_eq!(words(&enc), want, "{mode:?}");
            apply_filter(Filter::Exponential, &mut enc, 8).unwrap();
            for (d, x) in words(&enc).into_iter().map(f32::from_bits).zip(exp) {
                assert!((d - x).abs() < 1e-3, "{mode:?}: {d} {x}");
            }
        }
        let zero = [0.0, -0.0, 1.175_494_4e-38, -1.175_494_4e-38];
        let mut enc = encode_filter_exp(&zero, 4, 15, ExpMode::Separate);
        assert_eq!(words(&enc), [0xf200_0000, 0xf200_0000, 0x8e00_0000, 0x8e00_0000]);
        apply_filter(Filter::Exponential, &mut enc, 4).unwrap();
        assert!(words(&enc).into_iter().all(|w| f32::from_bits(w) == 0.0));
        // At 24 bits an all-ones mantissa rounds up to 2^23, which clamps.
        let overflow = [0.999_999_94, -0.999_999_94, 255.999_98, 1.0];
        let mut enc = encode_filter_exp(&overflow, 4, 24, ExpMode::Separate);
        assert_eq!(words(&enc), [0xe97f_ffff, 0xe980_0000, 0xf17f_ffff, 0xea40_0000]);
        apply_filter(Filter::Exponential, &mut enc, 4).unwrap();
        let got: Vec<f32> = words(&enc).into_iter().map(f32::from_bits).collect();
        assert_eq!(got, [0.999_999_9, -1.0, 255.999_97, 1.0]);
        let shared = [0.0, 0.1, -0.025, -0.0];
        for (mode, want) in [
            (
                ExpMode::SharedVector,
                [0xef00_0000, 0xef00_3333, 0xedff_cccd, 0xed00_0000],
            ),
            (
                ExpMode::SharedComponent,
                [0xed00_0000, 0xef00_3333, 0xedff_cccd, 0xef00_0000],
            ),
        ] {
            let mut enc = encode_filter_exp(&shared, 8, 15, mode);
            assert_eq!(words(&enc), want, "{mode:?}");
            apply_filter(Filter::Exponential, &mut enc, 8).unwrap();
            let got: Vec<f32> = words(&enc).into_iter().map(f32::from_bits).collect();
            assert!(got[0] == 0.0 && got[3] == 0.0, "{mode:?}");
        }
        let color = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.25],
            [0.4, 0.4, 0.4, 0.75],
        ];
        assert_eq!(
            encode_filter_color(&color, 4, 8),
            [
                0x40, 0x7f, 0xc1, 0xff, 0x7f, 0x00, 0x7f, 0xc0, 0x40, 0x81, 0xc0, 0xa0, 0x66, 0x00, 0x00, 0xdf
            ]
        );
        assert_eq!(
            halves(&encode_filter_color(&color, 8, 12)),
            [
                0x0400, 0x07ff, 0xfc01, 0x0fff, 0x07ff, 0x0000, 0x07ff, 0x0c00, 0x0400, 0xf801, 0xfc00, 0x0a00, 0x0666,
                0x0000, 0x0000, 0x0dff
            ]
        );
        assert_eq!(encode_oct8([0.0, -1.0, 0.0], 0.0), [0, 0x81, 0x7f, 0]);
    }

    /// A component of a decoded filter element, as -1 to 1 (signed) or 0 to 1.
    fn component(e: &[u8], i: usize, stride: usize, signed: bool) -> f32 {
        match (stride, signed) {
            (4, true) => f32::from(e[i] as i8) / 127.0,
            (4, false) => f32::from(e[i]) / 255.0,
            (_, true) => f32::from(i16::from_le_bytes([e[2 * i], e[2 * i + 1]])) / 32767.0,
            (_, false) => f32::from(u16::from_le_bytes([e[2 * i], e[2 * i + 1]])) / 65535.0,
        }
    }

    #[test]
    fn filters_roundtrip() {
        let mut r = Pcg32::new(21);
        let mut unit = |n: usize| -> [f32; 4] {
            loop {
                let v = [0; 4].map(|_| r.f32() * 2.0 - 1.0);
                let len = v[..n].iter().map(|c| c * c).sum::<f32>().sqrt();
                if len > 0.1 && len <= 1.0 {
                    return v.map(|c| c / len);
                }
            }
        };
        let vectors: Vec<[f32; 4]> = (0..500).map(|_| unit(3)).collect();
        for (stride, bits, min_dot) in [(4, 8, 0.999), (8, 10, 0.9999), (8, 12, 0.999_99), (8, 16, 0.999_999)] {
            let data: Vec<[f32; 4]> = vectors.iter().map(|v| [v[0], v[1], v[2], -1.0]).collect();
            let mut enc = encode_filter_oct(&data, stride, bits);
            apply_filter(Filter::Octahedral, &mut enc, stride).unwrap();
            for (e, v) in enc.chunks_exact(stride).zip(&data) {
                let d = normalize([0, 1, 2].map(|i| component(e, i, stride, true))).unwrap();
                let dot = (0..3).map(|i| d[i] * v[i]).sum::<f32>();
                assert!(dot > min_dot, "oct {stride}/{bits}: {v:?} -> {d:?}");
                // The fourth component passes through, quantized to the full width.
                assert_eq!(component(e, 3, stride, true), -1.0);
            }
        }
        let quats: Vec<[f32; 4]> = (0..500).map(|_| unit(4)).collect();
        for (bits, min_dot) in [(8, 0.998), (12, 0.9999), (16, 0.9999)] {
            let mut enc = encode_filter_quat(&quats, bits);
            apply_filter(Filter::Quaternion, &mut enc, 8).unwrap();
            for (e, q) in enc.chunks_exact(8).zip(&quats) {
                let d = [0, 1, 2, 3].map(|i| component(e, i, 8, true));
                let dot = (0..4).map(|i| d[i] * q[i]).sum::<f32>().abs();
                assert!(dot > min_dot, "quat {bits}: {q:?} -> {d:?}");
            }
        }
        for mode in [
            ExpMode::Separate,
            ExpMode::SharedVector,
            ExpMode::SharedComponent,
            ExpMode::Clamped,
        ] {
            for (stride, bits) in [(4, 10), (12, 15), (16, 24), (8, 1)] {
                let width = stride / 4;
                let data: Vec<f32> = (0..width * 200)
                    .map(|_| match r.below(8) {
                        0 => 0.0,
                        1 => (r.f32() - 0.5) * 1e6,
                        2 => (r.f32() - 0.5) * 1e-4,
                        _ => (r.f32() - 0.5) * 20.0,
                    })
                    .collect();
                let mut enc = encode_filter_exp(&data, stride, bits, mode);
                apply_filter(Filter::Exponential, &mut enc, stride).unwrap();
                let got: Vec<f32> = words(&enc).into_iter().map(f32::from_bits).collect();
                // The error is half a step of the exponent the value shared, in units of the
                // largest value sharing it.
                let column = |k: usize| data.iter().skip(k).step_by(width).fold(0f32, |m, x| m.max(x.abs()));
                for (i, (&d, &x)) in got.iter().zip(&data).enumerate() {
                    let scale = match mode {
                        ExpMode::Separate => x.abs(),
                        ExpMode::Clamped => x.abs().max(1.0),
                        ExpMode::SharedVector => data[i / width * width..][..width]
                            .iter()
                            .fold(0f32, |m, x| m.max(x.abs())),
                        ExpMode::SharedComponent => column(i % width),
                    };
                    let step = scale * 2f32.powi(2 - bits as i32);
                    assert!(
                        (d - x).abs() <= step * 0.5 + 1e-30,
                        "{mode:?} {stride}/{bits}: {x} -> {d}"
                    );
                }
            }
        }
        let colors: Vec<[f32; 4]> = (0..500).map(|_| [0; 4].map(|_| r.f32())).collect();
        for (stride, bits) in [(4, 8), (4, 5), (8, 10), (8, 12), (8, 16)] {
            let mut enc = encode_filter_color(&colors, stride, bits);
            apply_filter(Filter::Color, &mut enc, stride).unwrap();
            let step = 1.0 / ((1u32 << bits) - 1) as f32;
            let output = if stride == 4 { 0.5 / 255.0 } else { 0.5 / 65535.0 };
            for (e, c) in enc.chunks_exact(stride).zip(&colors) {
                for i in 0..4 {
                    // Half a step rounding, a step truncated in YCoCg or alpha's dropped bit, and
                    // rounding to the output width.
                    let tolerance = 1.5 * step + output + 1e-6;
                    let d = component(e, i, stride, false);
                    assert!(
                        (d - c[i]).abs() <= tolerance,
                        "color {stride}/{bits}: {c:?} channel {i} -> {d}"
                    );
                }
            }
        }
    }
}
