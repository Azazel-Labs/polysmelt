//! Binary FBX with its compressed arrays inflated ahead of the tree reader, through zlib-rs, which inflates them more
//! than twice as fast as the reader's own decoder: vertices, indices, normals and animation keys are most of a big
//! file, and nearly all of them compressed. The file is written again with every array stored and every node's end
//! offset moved to match. Anything not laid out as binary FBX 7 should be is left for the reader as it was, so it says
//! what's wrong.

use flate2::{Decompress, FlushDecompress, Status};

/// The file with its arrays stored rather than compressed, or `None` if it has none, isn't binary FBX 7, or isn't laid
/// out as one should be.
pub(super) fn inflated(data: &[u8]) -> Option<Vec<u8>> {
    if !data.starts_with(b"Kaydara FBX Binary  \0") {
        return None;
    }
    let version = u32::from_le_bytes(data.get(23..27)?.try_into().ok()?);
    if version < 7000 {
        return None;
    }
    let mut w = Writer {
        data,
        wide: version >= 7500,
        out: Vec::with_capacity(data.len().saturating_mul(2)),
        any: false,
        zlib: Decompress::new(true),
    };
    w.out.extend_from_slice(&data[..27]);
    let end = w.records(27, data.len())?;
    // The footer, as it was: the reader only warns if its padding no longer lines up.
    w.out.extend_from_slice(&data[end..]);
    w.any.then_some(w.out)
}

struct Writer<'a> {
    data: &'a [u8],
    /// Whether offsets and counts in node headers are 64-bit (FBX 7.5 on).
    wide: bool,
    out: Vec<u8>,
    /// Whether any array was inflated.
    any: bool,
    /// One decoder for every array, reset between them.
    zlib: Decompress,
}

impl Writer<'_> {
    fn word(&self, at: usize, wide: bool) -> Option<usize> {
        let d = self.data;
        if wide {
            usize::try_from(u64::from_le_bytes(d.get(at..at.checked_add(8)?)?.try_into().ok()?)).ok()
        } else {
            usize::try_from(u32::from_le_bytes(d.get(at..at.checked_add(4)?)?.try_into().ok()?)).ok()
        }
    }

    /// Writes `v` over the header word at `at` in the output.
    fn patch(&mut self, at: usize, v: usize) -> Option<()> {
        if self.wide {
            self.out[at..at + 8].copy_from_slice(&u64::try_from(v).ok()?.to_le_bytes());
        } else {
            self.out[at..at + 4].copy_from_slice(&u32::try_from(v).ok()?.to_le_bytes());
        }
        Some(())
    }

    fn copy(&mut self, from: usize, to: usize) -> Option<()> {
        self.out.extend_from_slice(self.data.get(from..to)?);
        Some(())
    }

    /// Writes the records from `at`, up to and including the null record that closes them, which must come before
    /// `end`: where they stop.
    fn records(&mut self, mut at: usize, end: usize) -> Option<usize> {
        let w = if self.wide { 8 } else { 4 };
        let header = 3 * w + 1;
        loop {
            let head = self.data.get(at..at.checked_add(header)?.min(end))?;
            if head.len() < header {
                return None;
            }
            let node_end = self.word(at, self.wide)?;
            if node_end == 0 {
                if head.iter().any(|&b| b != 0) {
                    return None;
                }
                self.copy(at, at + header)?;
                return Some(at + header);
            }
            let count = self.word(at + w, self.wide)?;
            let list = self.word(at + 2 * w, self.wide)?;
            let props = at + header + usize::from(self.data[at + 3 * w]);
            let props_end = props.checked_add(list)?;
            if node_end <= at || node_end > end || props_end > node_end {
                return None;
            }
            let start = self.out.len();
            self.copy(at, props)?;
            let written = self.out.len();
            self.properties(props, props_end, count)?;
            let list = self.out.len() - written;
            if props_end < node_end && self.records(props_end, node_end)? != node_end {
                return None;
            }
            let node_end_now = self.out.len();
            self.patch(start, node_end_now)?;
            self.patch(start + 2 * w, list)?;
            at = node_end;
        }
    }

    /// Writes `count` properties from `at`, which must fill it up to `end`, compressed arrays inflated.
    fn properties(&mut self, mut at: usize, end: usize, count: usize) -> Option<()> {
        for _ in 0..count {
            let kind = *self.data.get(at)?;
            let size = match kind {
                b'C' => 1,
                b'Y' => 2,
                b'I' | b'F' => 4,
                b'D' | b'L' => 8,
                b'S' | b'R' => self.word(at + 1, false)?.checked_add(4)?,
                b'b' | b'i' | b'f' | b'l' | b'd' => {
                    let n = self.word(at + 1, false)?;
                    let encoding = self.word(at + 5, false)?;
                    let packed = self.word(at + 9, false)?;
                    let body = at + 13;
                    let body_end = body.checked_add(packed)?;
                    if body_end > end {
                        return None;
                    }
                    if encoding == 1 {
                        let each = match kind {
                            b'b' => 1,
                            b'i' | b'f' => 4,
                            _ => 8,
                        };
                        let size = n.checked_mul(each)?;
                        let size32 = u32::try_from(size).ok()?;
                        self.out.push(kind);
                        self.out.extend_from_slice(&u32::try_from(n).ok()?.to_le_bytes());
                        self.out.extend_from_slice(&0u32.to_le_bytes());
                        self.out.extend_from_slice(&size32.to_le_bytes());
                        let before = self.out.len();
                        // Room for a byte more than the array says it holds, to tell a stream that inflates to more.
                        self.out.reserve(size + 1);
                        self.zlib.reset(true);
                        let status = self
                            .zlib
                            .decompress_vec(&self.data[body..body_end], &mut self.out, FlushDecompress::Finish)
                            .ok()?;
                        if status != Status::StreamEnd || self.out.len() - before != size {
                            return None;
                        }
                        self.any = true;
                    } else {
                        self.copy(at, body_end)?;
                    }
                    at = body_end;
                    continue;
                }
                _ => return None,
            };
            let next = at.checked_add(1 + size)?;
            if next > end {
                return None;
            }
            self.copy(at, next)?;
            at = next;
        }
        (at == end).then_some(())
    }
}

#[cfg(test)]
mod tests {
    use fbxcel::tree::any::AnyTree;
    use fbxcel::tree::v7400::NodeHandle;

    use super::inflated;

    fn same(a: NodeHandle<'_>, b: NodeHandle<'_>) {
        assert_eq!(a.name(), b.name());
        assert_eq!(a.attributes(), b.attributes(), "in {}", a.name());
        let (ac, bc): (Vec<_>, Vec<_>) = (a.children().collect(), b.children().collect());
        assert_eq!(ac.len(), bc.len(), "in {}", a.name());
        for (x, y) in ac.into_iter().zip(bc) {
            same(x, y);
        }
    }

    fn tree(data: &[u8]) -> fbxcel::tree::v7400::Tree {
        let AnyTree::V7400(_, tree, _) = AnyTree::from_seekable_reader(std::io::Cursor::new(data)).unwrap() else {
            panic!("not FBX 7.4")
        };
        tree
    }

    #[test]
    fn every_corpus_file_reads_the_same_inflated() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/fbx");
        let mut inflated_any = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            for file in std::fs::read_dir(entry.unwrap().path()).into_iter().flatten() {
                let path = file.unwrap().path();
                if path.extension().is_none_or(|e| !e.eq_ignore_ascii_case("fbx")) {
                    continue;
                }
                let data = std::fs::read(&path).unwrap();
                let Some(stored) = inflated(&data) else { continue };
                inflated_any += 1;
                assert!(inflated(&stored).is_none(), "{path:?} still has compressed arrays");
                let (a, b) = (tree(&data), tree(&stored));
                same(a.root(), b.root());
            }
        }
        assert!(inflated_any > 0, "no corpus file had compressed arrays");
    }

    #[test]
    fn leaves_what_it_does_not_follow() {
        assert_eq!(inflated(b"not fbx"), None);
        let mut data = b"Kaydara FBX Binary  \0\x1a\0".to_vec();
        data.extend_from_slice(&7400u32.to_le_bytes());
        assert_eq!(inflated(&data), None, "no records");
        // A node claiming to end past the file.
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&[0; 9]);
        assert_eq!(inflated(&data), None);
    }
}
