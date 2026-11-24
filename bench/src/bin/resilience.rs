//! Resilience: real files truncated at many offsets and with bytes flipped (a fixed seed per file),
//! each damaged file put through Polysmelt (load, flatten, both GLBs, the rig) and through the
//! competitor crate for its format (see `loaders.rs`). Counts what each does: reads it, returns an
//! error, panics, aborts the process (a stack overflow, or an allocation past the ceiling), or
//! hangs past the deadline.
//!
//! Cases run in a worker process, one at a time; a panic is caught there, anything that kills it
//! or runs past the deadline is recorded and a new worker takes the next case. Every loader runs on
//! a thread with an 8 MiB stack, and allocations past `--limit-mb` fail.
//!
//! ```text
//! cargo run --release --features evidence --bin resilience -- <file>... [--root <dir>] [--json out.json]
//!     [--repro <dir>] [--timeout 10] [--flips 60] [--limit-mb 3072]
//! cargo run --release --features evidence --bin resilience -- --case <loader> <file> <mutation>
//! ```
//!
//! `--case` replays one case in this process, where a panic shows its message and backtrace.

#[path = "../counting.rs"]
mod counting;
#[path = "../loaders.rs"]
mod loaders;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use loaders::{Run, Src};
use serde_json::json;

#[global_allocator]
static ALLOC: counting::Counting = counting::Counting;

const POLYSMELT: &str = "polysmelt";

fn ext_of(p: &Path) -> String {
    p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

/// The loaders a file goes through: Polysmelt's whole conversion, and its format's competitor.
fn loaders_for(path: &Path) -> Vec<(&'static str, Run)> {
    let mut out: Vec<(&'static str, Run)> = vec![(POLYSMELT, loaders::polysmelt_convert)];
    // `gltf (import)` does all the no-image variant does, and decodes images too.
    out.extend(loaders::competitors(&ext_of(path)).into_iter().filter(|l| l.name != "gltf (no image decode)").map(|l| (l.name, l.run)));
    out
}

// --- Mutations ------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// The damaged files made from one of `len` bytes named `name`.
fn mutations(name: &str, len: usize, flips: usize) -> Vec<String> {
    let mut cuts: Vec<usize> = [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 84, 100, 128, 256, 512, 1024, 4096]
        .into_iter()
        .collect();
    cuts.extend((1..32).map(|k| len * k / 32));
    cuts.extend([len.saturating_sub(1), len.saturating_sub(2), len.saturating_sub(4), len.saturating_sub(16)]);
    cuts.retain(|&c| c < len);
    cuts.sort_unstable();
    cuts.dedup();
    let mut out = vec!["orig".to_owned()];
    out.extend(cuts.into_iter().map(|c| format!("trunc:{c}")));
    let seed = fnv(name);
    for k in 0..flips {
        let count = [1, 1, 2, 4, 8, 32][k % 6];
        // A third of them within the first KiB, where headers and lengths are.
        let region = if k % 3 == 0 { len.min(1024) } else { len };
        out.push(format!("flip:{:x}:{count}:{region}", seed ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1));
    }
    out
}

fn mutate(bytes: &[u8], spec: &str) -> Vec<u8> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts[0] {
        "orig" => bytes.to_vec(),
        "trunc" => bytes[..parts[1].parse::<usize>().unwrap().min(bytes.len())].to_vec(),
        "flip" => {
            let mut rng = Rng(u64::from_str_radix(parts[1], 16).unwrap());
            let count: usize = parts[2].parse().unwrap();
            let region: usize = parts[3].parse::<usize>().unwrap().min(bytes.len());
            let mut out = bytes.to_vec();
            if region > 0 {
                for _ in 0..count {
                    let at = (rng.next() % region as u64) as usize;
                    let x = (rng.next() % 255 + 1) as u8;
                    out[at] ^= x;
                }
            }
            out
        }
        other => panic!("unknown mutation {other}"),
    }
}

// --- Worker ---------------------------------------------------------------------------------

static PANIC: Mutex<Option<String>> = Mutex::new(None);

/// Runs one case on a thread of its own: `ok`, `err` or `panic`, and what was said.
fn run_case(run: Run, path: &Path, bytes: Vec<u8>) -> (String, String) {
    let path = path.to_path_buf();
    let h = std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&Src::Bytes(&bytes, &path)).map(|l| {
                    drop(l.result);
                    l.units
                })
            }))
        })
        .unwrap();
    match h.join() {
        Ok(Ok(Ok(n))) => ("ok".into(), n.to_string()),
        Ok(Ok(Err(e))) => ("err".into(), e),
        Ok(Err(_)) | Err(_) => ("panic".into(), PANIC.lock().unwrap().take().unwrap_or_default()),
    }
}

fn worker(limit: usize) {
    counting::LIMIT.store(limit, Relaxed);
    // The counting allocator sees only Rust's heap; an address-space ceiling holds C++ (assimp)
    // to about the same budget, with room for the 8 MiB stacks and the code.
    #[cfg(target_os = "linux")]
    unsafe {
        let cap = (limit + (1 << 30)) as libc::rlim_t;
        let r = libc::rlimit { rlim_cur: cap, rlim_max: cap };
        libc::setrlimit(libc::RLIMIT_AS, &raw const r);
    }
    std::panic::set_hook(Box::new(|info| {
        *PANIC.lock().unwrap() = Some(info.to_string());
    }));
    let mut cache: Option<(PathBuf, Vec<u8>)> = None;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let f: Vec<&str> = line.splitn(4, '\t').collect();
        let (id, loader, path, spec) = (f[0], f[1], PathBuf::from(f[2]), f[3]);
        if cache.as_ref().is_none_or(|(p, _)| *p != path) {
            cache = Some((path.clone(), std::fs::read(&path).unwrap()));
        }
        let bytes = mutate(&cache.as_ref().unwrap().1, spec);
        let run = loaders_for(&path).into_iter().find(|(n, _)| *n == loader).expect("loader").1;
        let (status, detail) = run_case(run, &path, bytes);
        let detail: String = detail.replace(['\t', '\n', '\r'], " ").chars().take(300).collect();
        writeln!(stdout, "{id}\t{status}\t{detail}").unwrap();
        stdout.flush().unwrap();
    }
}

// --- Parent ---------------------------------------------------------------------------------

struct Worker {
    child: Child,
    stdin: ChildStdin,
    replies: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
}

impl Worker {
    fn spawn(limit_mb: usize) -> Worker {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--worker", "--limit-mb", &limit_mb.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let out = child.stdout.take().unwrap();
        let err = child.stderr.take().unwrap();
        let (tx, replies) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let log = stderr.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                let mut l = log.lock().unwrap();
                l.push(line);
                if l.len() > 20 {
                    l.remove(0);
                }
            }
        });
        Worker { child, stdin, replies, stderr }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut files = Vec::new();
    let mut opts: BTreeMap<String, String> = BTreeMap::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--worker" => {
                opts.insert("worker".into(), String::new());
            }
            "--case" => {
                let (l, f, m) = (it.next().unwrap(), PathBuf::from(it.next().unwrap()), it.next().unwrap());
                let bytes = mutate(&std::fs::read(&f).unwrap(), m);
                let run = loaders_for(&f).into_iter().find(|(n, _)| n == l).expect("no such loader for this file").1;
                match run(&Src::Bytes(&bytes, &f)) {
                    Ok(l) => println!("ok: {}", l.units),
                    Err(e) => println!("error: {e}"),
                }
                return;
            }
            s if s.starts_with("--") => {
                opts.insert(s[2..].to_owned(), it.next().cloned().unwrap_or_default());
            }
            _ => files.push(PathBuf::from(a)),
        }
    }
    let limit_mb: usize = opts.get("limit-mb").map_or(3072, |v| v.parse().unwrap());
    if opts.contains_key("worker") {
        worker(limit_mb << 20);
        return;
    }
    let timeout = Duration::from_secs_f64(opts.get("timeout").map_or(10.0, |v| v.parse().unwrap()));
    let flips: usize = opts.get("flips").map_or(60, |v| v.parse().unwrap());
    let root = opts.get("root").map(PathBuf::from);
    let repro = opts.get("repro").map(PathBuf::from);

    let mut w = Worker::spawn(limit_mb);
    let mut id = 0u64;
    // (format, loader) -> status -> count
    let mut table: BTreeMap<(String, String), BTreeMap<String, usize>> = BTreeMap::new();
    let mut per_file = Vec::new();
    let mut notable = Vec::new();
    let started = Instant::now();
    for path in &files {
        let name = root
            .as_ref()
            .and_then(|r| path.strip_prefix(r).ok())
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = std::fs::read(path).unwrap();
        let specs = mutations(&name, bytes.len(), flips);
        let ext = ext_of(path);
        let format = if ext == "glb" { "gltf".to_owned() } else { ext.clone() };
        for (loader, _) in loaders_for(path) {
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            let t = Instant::now();
            for spec in &specs {
                id += 1;
                writeln!(w.stdin, "{id}\t{loader}\t{}\t{spec}", path.display()).unwrap();
                w.stdin.flush().unwrap();
                let (status, detail) = loop {
                    match w.replies.recv_timeout(timeout) {
                        Ok(line) => {
                            let f: Vec<&str> = line.splitn(3, '\t').collect();
                            if f[0] == id.to_string() {
                                break (f[1].to_owned(), f.get(2).unwrap_or(&"").to_string());
                            }
                        }
                        Err(RecvTimeoutError::Timeout) => {
                            let _ = w.child.kill();
                            let _ = w.child.wait();
                            w = Worker::spawn(limit_mb);
                            break ("hang".into(), format!("no answer in {timeout:?}"));
                        }
                        Err(RecvTimeoutError::Disconnected) => {
                            let code = w.child.wait().ok().and_then(|s| s.code());
                            std::thread::sleep(Duration::from_millis(50));
                            let err = w.stderr.lock().unwrap().join(" | ");
                            let status = if err.contains("memory allocation of") { "abort-alloc" } else { "crash" };
                            let detail = format!("exit {code:?}: {}", err.chars().rev().take(300).collect::<String>().chars().rev().collect::<String>());
                            w = Worker::spawn(limit_mb);
                            break (status.into(), detail);
                        }
                    }
                };
                // A file that doesn't load undamaged says nothing about resilience: note it.
                let status = if spec == "orig" && status != "ok" { format!("orig-{status}") } else { status };
                *counts.entry(status.clone()).or_default() += 1;
                if !matches!(status.as_str(), "ok" | "err") {
                    eprintln!("  {loader} {name} {spec}: {status} {detail}");
                    let mut repro_path = None;
                    if let Some(dir) = &repro {
                        let safe: String = loader.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
                        // The last folders and the file name: short enough for Windows paths.
                        let parts: Vec<&str> = name.split('/').collect();
                        let stem = parts[parts.len().saturating_sub(3)..].join("__");
                        let p = dir.join(safe).join(format!("{stem}__{}.{ext}", spec.replace(':', "_")));
                        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                        std::fs::write(&p, mutate(&bytes, spec)).unwrap();
                        repro_path = Some(p.to_string_lossy().replace('\\', "/"));
                    }
                    notable.push(json!({
                        "file": name, "loader": loader, "mutation": spec, "status": status, "detail": detail,
                        "repro": repro_path,
                    }));
                }
            }
            eprintln!("{name} [{loader}] {:?} in {:.1}s", counts, t.elapsed().as_secs_f64());
            for (s, n) in &counts {
                *table.entry((format.clone(), loader.to_owned())).or_default().entry(s.clone()).or_default() += n;
            }
            per_file.push(json!({ "file": name, "loader": loader, "cases": specs.len(), "counts": counts }));
        }
    }
    let _ = w.child.kill();
    let mut md = String::from(
        "| Format | Loader | Cases | Read | Error | Panic | Abort (alloc) | Crash | Hang | Undamaged file failed |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for ((format, loader), c) in &table {
        let g = |k: &str| c.get(k).copied().unwrap_or(0);
        let orig: usize = c.iter().filter(|(k, _)| k.starts_with("orig-")).map(|(_, v)| v).sum();
        let total: usize = c.values().sum();
        md.push_str(&format!(
            "| {format} | {loader} | {total} | {} | {} | {} | {} | {} | {} | {orig} |\n",
            g("ok"),
            g("err"),
            g("panic"),
            g("abort-alloc"),
            g("crash"),
            g("hang"),
        ));
    }
    println!("{md}");
    eprintln!("{} cases in {:.0}s", id, started.elapsed().as_secs_f64());
    if let Some(p) = opts.get("json") {
        let table: Vec<_> = table.iter().map(|((f, l), c)| json!({ "format": f, "loader": l, "counts": c })).collect();
        let doc = json!({ "table": table, "files": per_file, "notable": notable, "flips": flips,
            "timeout_s": timeout.as_secs_f64(), "limit_mb": limit_mb });
        std::fs::write(p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    }
    if let Some(p) = opts.get("markdown") {
        std::fs::write(p, &md).unwrap();
    }
}
