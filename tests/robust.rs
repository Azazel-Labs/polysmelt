//! Robustness: every corpus case's model file, damaged in fixed ways, must load or fail with an error. It must
//! never panic, never allocate without bound, and never hang.
//!
//! Each case's file is read from memory with its folder's files as [`Memory`] assets, the way the `conformance`
//! test reads it, but cut short at many lengths, with bytes changed, and with ranges moved about. A file that still
//! loads is also flattened and written as GLB, since a reader's mistake often shows only there. The damage is
//! seeded by the case's name, so a run is the same every time.
//!
//! A counting allocator stands in for the system's: one load that holds more than [`MEMORY_LIMIT`] at once fails
//! the test. (A request the system refuses outright aborts the run with "memory allocation of N bytes failed",
//! which is the same finding; `ROBUST_ONLY` narrows it down.) A watchdog stops the run when one load takes longer
//! than [`HANG_LIMIT`].
//!
//! `ROBUST_ONLY=<text>` runs only the cases whose name holds the text, and `ROBUST_DUMP=<folder>` writes each
//! failing input there.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

use peak_alloc::PeakAlloc;
use polysmelt::Loader;
use polysmelt::assets::Memory;
use polysmelt::document::Pose;
use serde_json::Value;

/// The most one load may hold at once: far above what any corpus file needs, far below what a count read from
/// damaged bytes asks for when nothing checks it.
const MEMORY_LIMIT: usize = 1 << 30;
/// How long one load may take before the watchdog stops the run. Loads here take microseconds.
const HANG_LIMIT: Duration = Duration::from_secs(60);
/// Files shorter than this are cut at every length.
const SMALL: usize = 4096;
/// A longer file is cut at every length this near its start and its end, and at about this many lengths between.
const EDGE: usize = 384;
const STRIDES: usize = 384;
/// A debug build runs several times slower, so it tries one length in this many and a quarter of the rest. A
/// release build tries everything.
const THIN: usize = if cfg!(debug_assertions) { 8 } else { 1 };
/// Changed bytes and moved ranges tried per case.
const FLIPS: usize = 192 / if THIN > 1 { 4 } else { 1 };
const SPLICES: usize = 64 / if THIN > 1 { 4 } else { 1 };

/// The system allocator, counting what the whole test holds: the library forbids `unsafe`, and so does this test,
/// so the counting is a crate's.
#[global_allocator]
static ALLOCATOR: PeakAlloc = PeakAlloc;

/// Loads run under a read lock. One that may have gone over the limit runs again alone, under the write lock, where
/// the allocator's peak is that load's and no other's.
static ALONE: RwLock<()> = RwLock::new(());

/// Every model extension a full build reads, as the `conformance` test lists them.
const ALL_EXTENSIONS: &[&str] = &[
    "stl", "ply", "vox", "obj", "3ds", "fbx", "3mf", "usd", "usda", "usdc", "usdz", "gltf", "glb", "skp", "bvh", "dae",
    "zae", "kmz", "off", "vrm", "vrma", "x3d", "x3dv", "x3dz", "wrl", "wrz", "dxf", "splat", "spz", "las", "laz",
    "pcd", "xyz", "pts", "ptx", "asc", "e57", "amf", "lwo", "ac", "pmx", "pmd", "vmd", "x", "md2", "md3", "md5mesh",
    "md5anim", "iqm", "smd", "vta", "ms3d", "b3d", "asf", "amc", "c3d", "trc", "htr", "mesh", "skeleton", "xml", "flt",
    "ogex", "irrmesh", "irr", "xgl", "zgl",
];

/// A case: its name (`obj/quads`), its model file's name within its folder and bytes, and its folder's files.
struct Case {
    name: String,
    file: String,
    bytes: Vec<u8>,
    assets: Memory,
}

fn relative(p: &Path, root: &Path) -> String {
    p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/")
}

/// Every corpus case, in order.
fn cases() -> Vec<Case> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
    let only = std::env::var("ROBUST_ONLY").ok();
    let sorted = |dir: &Path| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
        v.sort();
        v
    };
    let mut out = Vec::new();
    for format in sorted(&root).into_iter().filter(|p| p.is_dir()) {
        for dir in sorted(&format).into_iter().filter(|p| p.is_dir()) {
            let name = relative(&dir, &root);
            if only.as_ref().is_some_and(|o| !name.contains(o.as_str())) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(dir.join("expect.json")) else {
                continue;
            };
            let expect: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: expect.json: {e}"));
            let file = if let Some(f) = expect.get("file").and_then(Value::as_str) {
                dir.join(f)
            } else {
                let model = |p: &PathBuf| {
                    let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase());
                    p.is_file() && ext.is_some_and(|e| ALL_EXTENSIONS.contains(&e.as_str()))
                };
                let models: Vec<PathBuf> = sorted(&dir).into_iter().filter(model).collect();
                assert!(models.len() == 1, "{name}: expected one model file, found {models:?}");
                models.into_iter().next().unwrap()
            };
            let mut assets = Memory::new();
            let mut dirs = vec![dir.clone()];
            while let Some(d) = dirs.pop() {
                for p in sorted(&d) {
                    if p.is_dir() {
                        dirs.push(p);
                    } else {
                        assets.insert(&relative(&p, &dir), std::fs::read(&p).unwrap());
                    }
                }
            }
            out.push(Case {
                name,
                file: relative(&file, &dir),
                bytes: std::fs::read(&file).unwrap(),
                assets,
            });
        }
    }
    out
}

/// A small seeded generator (SplitMix64): the same damage on every run and every machine.
struct Rng(u64);

impl Rng {
    fn named(name: &str) -> Self {
        // FNV-1a over the case's name.
        Self(name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        }))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number below `n`, which isn't zero.
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap()
    }
}

/// One way of damaging a file.
#[derive(Debug, Clone, Copy)]
enum Damage {
    /// Only the first `len` bytes.
    Truncate { len: usize },
    /// One bit flipped.
    Bit { at: usize, bit: u8 },
    /// One byte replaced.
    Byte { at: usize, with: u8 },
    /// Four bytes replaced: where a count or an offset was, an extreme one.
    Word { at: usize, with: [u8; 4] },
    /// `len` bytes at `at` removed.
    Delete { at: usize, len: usize },
    /// `len` bytes at `from` inserted again at `to`.
    Repeat { from: usize, len: usize, to: usize },
    /// `len` bytes at `from` copied over those at `to`.
    Overwrite { from: usize, len: usize, to: usize },
}

impl Damage {
    fn apply(self, bytes: &[u8]) -> Vec<u8> {
        let mut out = bytes.to_vec();
        match self {
            Self::Truncate { len } => out.truncate(len),
            Self::Bit { at, bit } => out[at] ^= 1 << bit,
            Self::Byte { at, with } => out[at] = with,
            Self::Word { at, with } => {
                for (o, w) in out[at..].iter_mut().zip(with) {
                    *o = w;
                }
            }
            Self::Delete { at, len } => {
                out.drain(at..at + len);
            }
            Self::Repeat { from, len, to } => {
                out.splice(to..to, bytes[from..from + len].iter().copied());
            }
            Self::Overwrite { from, len, to } => {
                for (o, &b) in out[to..].iter_mut().zip(&bytes[from..from + len]) {
                    *o = b;
                }
            }
        }
        out
    }
}

/// The damage tried on a case, the same list every run.
fn damage(case: &Case) -> Vec<Damage> {
    let len = case.bytes.len();
    let mut out = Vec::new();
    if len < SMALL {
        out.extend((0..len).step_by(THIN).map(|len| Damage::Truncate { len }));
    } else {
        // An odd stride, so the cuts don't all fall on the same alignment.
        let stride = ((len - 2 * EDGE) / STRIDES) | 1;
        out.extend((0..EDGE).step_by(THIN).map(|len| Damage::Truncate { len }));
        out.extend(
            (EDGE..len - EDGE)
                .step_by(stride * THIN)
                .map(|len| Damage::Truncate { len }),
        );
        out.extend((len - EDGE..len).step_by(THIN).map(|len| Damage::Truncate { len }));
    }
    if len == 0 {
        return out;
    }
    let mut rng = Rng::named(&case.name);
    for _ in 0..FLIPS {
        let at = rng.below(len);
        out.push(match rng.below(4) {
            0 => Damage::Bit {
                at,
                bit: u8::try_from(rng.below(8)).unwrap(),
            },
            1 => Damage::Byte {
                at,
                with: rng.next().to_le_bytes()[0],
            },
            2 => Damage::Byte {
                at,
                with: [0x00, 0xff, 0x7f, 0x80][rng.below(4)],
            },
            _ => Damage::Word {
                at,
                with: [[0xff; 4], [0; 4], [0xff, 0xff, 0xff, 0x7f], [0x7f, 0xff, 0xff, 0xff]][rng.below(4)],
            },
        });
    }
    for _ in 0..SPLICES {
        let from = rng.below(len);
        let len_of = 1 + rng.below((len - from).min(256));
        let to = rng.below(len);
        out.push(match rng.below(3) {
            0 => Damage::Delete { at: from, len: len_of },
            1 => Damage::Repeat { from, len: len_of, to },
            _ => Damage::Overwrite { from, len: len_of, to },
        });
    }
    out
}

thread_local! {
    /// The message of this thread's last panic, with where it was raised.
    static PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Loads the damaged `bytes`, and flattens and writes what loads. `Err` is a panic's message.
fn load(case: &Case, bytes: Vec<u8>) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        if let Ok(doc) = Loader::new().read_with(bytes, &case.file, &case.assets) {
            let _ = doc.flatten(Pose::Rest);
            let _ = doc.to_glb();
        }
    }))
    .map_err(|_| PANIC.take().unwrap_or_else(|| "a panic on another thread".into()))
}

/// What went wrong with one load, or nothing.
fn attempt(case: &Case, bytes: Vec<u8>) -> Option<String> {
    let start = Instant::now();
    let shared = ALONE.read().unwrap();
    let again = bytes.clone();
    if let Err(message) = load(case, bytes) {
        return Some(format!("panic: {message}"));
    }
    // The peak is every thread's together, so a high one may be another load's: this load again, alone.
    let over = |held: usize| ALLOCATOR.peak_usage().saturating_sub(held) > MEMORY_LIMIT;
    if over(0) {
        drop(shared);
        let _alone = ALONE.write().unwrap();
        let held = ALLOCATOR.current_usage();
        ALLOCATOR.reset_peak_usage();
        let _ = load(case, again);
        let peak = ALLOCATOR.peak_usage().saturating_sub(held);
        ALLOCATOR.reset_peak_usage();
        if peak > MEMORY_LIMIT {
            return Some(format!("allocated {} MiB", peak >> 20));
        }
    }
    // Far under the watchdog's limit, and still thousands of times what a load of a file this size takes.
    (start.elapsed() > HANG_LIMIT / 4).then(|| format!("took {:?}", start.elapsed()))
}

#[test]
fn damaged_files_load_or_fail_cleanly() {
    let cases = cases();
    assert!(!cases.is_empty(), "no corpus cases");
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        let payload = info.payload();
        let text = payload
            .downcast_ref::<&str>()
            .map(ToString::to_string)
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        let at = info.location().map(|l| format!("{}:{}", l.file(), l.line()));
        PANIC.set(Some(format!("{} at {}", text, at.unwrap_or_default())));
    }));

    let threads = std::thread::available_parallelism()
        .map_or(4, usize::from)
        .min(cases.len());
    let next = AtomicUsize::new(0);
    // Per worker: the case it's on (plus one) and when its current load began, in milliseconds since `epoch`.
    let epoch = Instant::now();
    let millis = || u64::try_from(epoch.elapsed().as_millis()).unwrap();
    let busy: Vec<(AtomicUsize, AtomicU64)> = (0..threads).map(|_| Default::default()).collect();
    let failures: Mutex<Vec<(String, Damage, String)>> = Mutex::new(Vec::new());
    let loads = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let dump = std::env::var_os("ROBUST_DUMP").map(PathBuf::from);

    std::thread::scope(|scope| {
        for slot in &busy {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(case) = cases.get(i) else { break };
                    slot.0.store(i + 1, Ordering::Relaxed);
                    // A case this build has no reader for fails the same way however it's damaged.
                    slot.1.store(millis(), Ordering::Relaxed);
                    let whole = Loader::new().read_with(case.bytes.clone(), &case.file, &case.assets);
                    let skip = whole.as_ref().is_err_and(|e| {
                        matches!(
                            e.kind(),
                            polysmelt::ErrorKind::FeatureDisabled | polysmelt::ErrorKind::UnknownFormat
                        )
                    });
                    drop(whole);
                    if skip {
                        continue;
                    }
                    for damage in damage(case) {
                        slot.1.store(millis(), Ordering::Relaxed);
                        let bytes = damage.apply(&case.bytes);
                        let kept = dump.as_ref().map(|_| bytes.clone());
                        loads.fetch_add(1, Ordering::Relaxed);
                        if let Some(what) = attempt(case, bytes) {
                            let mut failures = failures.lock().unwrap();
                            if let (Some(dir), Some(kept)) = (&dump, kept) {
                                let name = format!("{}-{}-{}", failures.len(), case.name.replace('/', "-"), case.file);
                                std::fs::create_dir_all(dir).unwrap();
                                std::fs::write(dir.join(name.replace('/', "-")), kept).unwrap();
                            }
                            failures.push((case.name.clone(), damage, what));
                        }
                    }
                }
                slot.0.store(0, Ordering::Relaxed);
                done.fetch_add(1, Ordering::Relaxed);
            });
        }
        // The watchdog: a load that never returns would otherwise hang the run with nothing to say which it was.
        while done.load(Ordering::Relaxed) < threads {
            std::thread::sleep(Duration::from_millis(50));
            for slot in &busy {
                let (case, since) = (slot.0.load(Ordering::Relaxed), slot.1.load(Ordering::Relaxed));
                if case != 0 && millis().saturating_sub(since) > u64::try_from(HANG_LIMIT.as_millis()).unwrap() {
                    eprintln!("{}: a load has run for over {HANG_LIMIT:?}", cases[case - 1].name);
                    std::process::exit(1);
                }
            }
        }
    });
    std::panic::set_hook(default_hook);

    let mut failures = failures.into_inner().unwrap();
    failures.sort_by(|a, b| a.0.cmp(&b.0));
    // One line per fault, with how many inputs reach it and the first few.
    let mut by_fault: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (case, damage, what) in &failures {
        by_fault.entry(what).or_default().push(format!("{case} {damage:?}"));
    }
    for (what, inputs) in &by_fault {
        eprintln!("{what}: {} inputs", inputs.len());
        for input in inputs.iter().take(4) {
            eprintln!("    {input}");
        }
    }
    println!(
        "{} loads of {} cases in {:?}",
        loads.load(Ordering::Relaxed),
        cases.len(),
        epoch.elapsed()
    );
    assert!(
        failures.is_empty(),
        "{} damaged inputs in {} ways panicked, hung or allocated without bound",
        failures.len(),
        by_fault.len()
    );
}
