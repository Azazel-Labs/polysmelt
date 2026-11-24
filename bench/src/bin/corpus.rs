//! Whole corpora through every loader: which files each reads, what it finds in them, how long it
//! takes and how much memory it needs (see `loaders.rs` for the loaders).
//!
//! ```text
//! corpus sweep --corpus <name>=<dir>... --out sweep.jsonl [--jobs 4] [--timeout 60] [--limit-mb 6144]
//!     [--glb <dir>] [--loaders a,b]
//! corpus one <loader> <file> [--glb-out <file.glb>] [--limit-mb 6144] [--repeat 0]
//! corpus time --sweep sweep.jsonl [--budget 0.25] [--loaders a,b] [--ext obj,ply] [--min-bytes N]
//!     [--json out.json]
//! ```
//!
//! `sweep` runs each (file, loader) pair in a process of its own (`one`), so a crash, a hang or a
//! runaway allocation costs only that pair: it's recorded (`crash`, `timeout`, `oom`) and the sweep
//! goes on. Each process reports its own figures: the time, the peak resident memory the load
//! added (any language's heap, so assimp's C++ heap counts too), the peak Rust heap, the threads
//! alive after it, and what the result holds. With `--glb`, the conversions write their GLBs under
//! that folder for the validator.
//!
//! `time` takes the pairs that read cleanly in a sweep and times them in this one process, the
//! loaders for each file taking turns in a rotating order, so drift on a shared machine hits them
//! all alike: the median and the fastest of the rounds.

#[path = "../counting.rs"]
mod counting;
#[path = "../loaders.rs"]
mod loaders;

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use counting::{CURRENT, PEAK};
use loaders::{Loader, Src};
use serde_json::{Value, json};

#[global_allocator]
static ALLOC: counting::Counting = counting::Counting;

/// Extensions never a model: images, and the side files models refer to.
const NOT_MODELS: &[&str] = &[
    "png", "jpg", "jpeg", "bmp", "tga", "gif", "dds", "ktx", "ktx2", "hdr", "exr", "webp", "tif", "tiff", "psd", "bin",
    "mtl", "txt", "md", "json", "js", "html", "py", "zip",
];

fn args_of(args: &[String]) -> (Vec<String>, BTreeMap<String, Vec<String>>) {
    let mut pos = Vec::new();
    let mut opts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(k) = a.strip_prefix("--") {
            opts.entry(k.to_owned()).or_default().push(it.next().cloned().unwrap_or_default());
        } else {
            pos.push(a.clone());
        }
    }
    (pos, opts)
}

fn opt<'a>(opts: &'a BTreeMap<String, Vec<String>>, k: &str) -> Option<&'a str> {
    opts.get(k).and_then(|v| v.last()).map(String::as_str)
}

/// A comma-separated list; a comma inside parentheses belongs to the item, as in the loader
/// name `polysmelt (threads: 0, all cores)`.
fn list_opt(opts: &BTreeMap<String, Vec<String>>, k: &str) -> Option<Vec<String>> {
    opt(opts, k).map(|v| {
        let mut depth = 0i32;
        v.split(|c: char| {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            c == ',' && depth == 0
        })
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        eprintln!("usage: corpus sweep|one|time ...");
        std::process::exit(2);
    };
    let (pos, opts) = args_of(&args[1..]);
    match cmd.as_str() {
        "sweep" => sweep(&opts),
        "one" => one(&pos, &opts),
        "time" => time(&opts),
        "time-file" => time_one(&pos, &opts),
        "info" => println!(
            "{}",
            json!({ "assimp": loaders::assimp_version(), "polysmelt_extensions": polysmelt::formats::EXTENSIONS })
        ),
        _ => {
            eprintln!("unknown command {cmd}");
            std::process::exit(2);
        }
    }
}

// --- Memory ---------------------------------------------------------------------------------

/// `VmRSS`, `VmHWM` (peak) and `Threads` from /proc/self/status, in KiB (threads as a count).
fn proc_status() -> Option<(u64, u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let f = |k: &str| {
        s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok())
    };
    Some((f("VmRSS:")?, f("VmHWM:")?, f("Threads:")?))
}

/// Resets the peak resident set to what's resident now (Linux).
fn reset_peak() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

fn set_limits(limit_mb: usize) {
    counting::LIMIT.store(limit_mb << 20, Relaxed);
    #[cfg(target_os = "linux")]
    unsafe {
        // Room above the Rust heap's ceiling for code, stacks and allocator slack; C++ heaps
        // (assimp) meet this one alone.
        let cap = ((limit_mb + 1024) << 20) as libc::rlim_t;
        let r = libc::rlimit { rlim_cur: cap, rlim_max: cap };
        libc::setrlimit(libc::RLIMIT_AS, &raw const r);
    }
}

// --- one ------------------------------------------------------------------------------------

fn one(pos: &[String], opts: &BTreeMap<String, Vec<String>>) {
    let (name, file) = (&pos[0], PathBuf::from(&pos[1]));
    let limit_mb: usize = opt(opts, "limit-mb").map_or(6144, |v| v.parse().unwrap());
    let glb_out = opt(opts, "glb-out").map(PathBuf::from);
    // Runs before the measured one (for `perf record` of a short load).
    let repeat: usize = opt(opts, "repeat").map_or(0, |v| v.parse().unwrap());
    set_limits(limit_mb);
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string().replace('\n', " ");
        println!("{}", json!({ "status": "panic", "error": msg.chars().take(500).collect::<String>() }));
        let _ = std::io::stdout().flush();
    }));
    let ext = loaders::ext_of(&file);
    let Some(loader) = loaders::by_name(name, &ext) else {
        println!("{}", json!({ "status": "unsupported" }));
        return;
    };
    // Warm the page cache, so the first loader of a file isn't charged for the disk.
    let _ = std::fs::read(&file);
    for _ in 0..repeat {
        let _ = (loader.run)(&Src::File(&file));
    }
    reset_peak();
    let before = proc_status();
    let heap0 = CURRENT.load(Relaxed);
    PEAK.store(heap0, Relaxed);
    let start = Instant::now();
    let out = std::panic::catch_unwind(|| run_one(&loader, &file, glb_out.as_deref()));
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let heap_peak = PEAK.load(Relaxed) - heap0;
    let after = proc_status();
    let Ok(out) = out else {
        // The hook has printed the panic.
        std::process::exit(101);
    };
    let mut line = json!({ "ms": ms, "heap_peak": heap_peak });
    if let (Some((rss0, _, _)), Some((_, hwm, threads))) = (before, after) {
        line["rss_kb"] = json!(hwm.saturating_sub(rss0));
        line["hwm_kb"] = json!(hwm);
        line["threads"] = json!(threads);
    }
    match out {
        Ok((l, glb_bytes)) => {
            line["status"] = json!("ok");
            line["units"] = json!(l.units);
            line["stats"] = json!((l.stats)(l.result.as_ref()));
            if let Some(b) = glb_bytes {
                line["glb_bytes"] = json!(b);
            }
            // Don't spend the time freeing a big result: the process ends here.
            std::mem::forget(l);
        }
        Err(e) => {
            line["status"] = json!(if e.contains("bad_alloc") { "oom" } else { "err" });
            line["error"] = json!(e.replace('\n', " ").chars().take(500).collect::<String>());
        }
    }
    println!("{line}");
    let _ = std::io::stdout().flush();
}

/// Runs the loader; for a conversion with somewhere to write it, writes the GLB too (inside the
/// timing: a conversion's whole job).
fn run_one(loader: &Loader, file: &Path, glb_out: Option<&Path>) -> Result<(loaders::Loaded, Option<u64>), String> {
    let src = Src::File(file);
    let Some(out) = glb_out else { return (loader.run)(&src).map(|l| (l, None)) };
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    match loader.name {
        loaders::POLYSMELT_GLB => {
            let l = (loader.run)(&src)?;
            type Doc = (polysmelt::document::Document<'static>, Vec<u8>);
            let glb = &l.result.downcast_ref::<Doc>().ok_or("unexpected result")?.1;
            std::fs::write(out, glb).map_err(|e| e.to_string())?;
            let k = glb.len() as u64;
            Ok((l, Some(k)))
        }
        #[cfg(feature = "assimp")]
        loaders::ASSIMP_GLB => {
            let s = loaders::assimp::read(&src)?;
            let k = loaders::assimp::export(&s, Some(out))?;
            // Textures it leaves outside the GLB resolve from the source's folder.
            let _ = std::fs::write(out.with_extension("glb.src"), file.parent().unwrap_or(Path::new(".")).to_string_lossy().as_bytes());
            let stats = loaders::assimp::scene_stats(&s);
            fn from_map(r: &dyn std::any::Any) -> loaders::Stats {
                r.downcast_ref::<loaders::Stats>().cloned().unwrap_or_default()
            }
            drop(s);
            Ok((loaders::Loaded { result: Box::new(stats), units: k as usize, stats: from_map }, Some(k as u64)))
        }
        _ => (loader.run)(&src).map(|l| (l, None)),
    }
}

// --- sweep ----------------------------------------------------------------------------------

struct Task {
    corpus: String,
    rel: String,
    path: PathBuf,
    ext: String,
    bytes: u64,
    loader: &'static str,
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.filter_map(Result::ok).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        match e.file_type() {
            Ok(t) if t.is_dir() => walk(&p, out),
            Ok(t) if t.is_file() => out.push(p),
            _ => {}
        }
    }
}

/// Every loader that reads `ext`, Polysmelt's first.
fn loaders_for(ext: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    if loaders::polysmelt_reads(ext) {
        out.extend([loaders::POLYSMELT, loaders::POLYSMELT_GLB]);
    }
    out.extend(loaders::competitors(ext).iter().map(|l| l.name));
    out.extend(loaders::assimp_glb(ext).map(|l| l.name));
    out
}

fn slug(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect::<String>()
}

fn sweep(opts: &BTreeMap<String, Vec<String>>) {
    let out_path = PathBuf::from(opt(opts, "out").expect("--out"));
    let jobs: usize = opt(opts, "jobs").map_or(4, |v| v.parse().unwrap());
    let timeout = Duration::from_secs_f64(opt(opts, "timeout").map_or(60.0, |v| v.parse().unwrap()));
    let limit_mb: usize = opt(opts, "limit-mb").map_or(6144, |v| v.parse().unwrap());
    let glb_dir = opt(opts, "glb").map(PathBuf::from);
    let only = list_opt(opts, "loaders");
    let exe = std::env::current_exe().unwrap();

    let mut tasks = Vec::new();
    let mut unsupported = Vec::new();
    for c in opts.get("corpus").into_iter().flatten() {
        let (name, dir) = c.split_once('=').expect("--corpus name=dir");
        let dir = PathBuf::from(dir);
        let mut files = Vec::new();
        walk(&dir, &mut files);
        for path in files {
            let ext = loaders::ext_of(&path);
            if ext.is_empty() || NOT_MODELS.contains(&ext.as_str()) {
                continue;
            }
            let ls = loaders_for(&ext);
            if ls.is_empty() {
                continue;
            }
            let rel = path.strip_prefix(&dir).unwrap_or(&path).to_string_lossy().replace('\\', "/");
            let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
            if !loaders::polysmelt_reads(&ext) {
                unsupported.push(json!({ "corpus": name, "file": rel, "ext": ext, "bytes": bytes,
                    "loader": loaders::POLYSMELT, "status": "unsupported" }));
            }
            for l in ls {
                if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == l)) {
                    continue;
                }
                tasks.push(Task { corpus: name.to_owned(), rel: rel.clone(), path: path.clone(), ext: ext.clone(), bytes, loader: l });
            }
        }
    }
    // The biggest files first, a few at a time, so two giants rarely share the machine's memory;
    // then the rest in parallel.
    tasks.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.rel.cmp(&b.rel)));
    let big: VecDeque<Task> = tasks.iter().take_while(|t| t.bytes >= 48 << 20).map(clone_task).collect();
    let rest: VecDeque<Task> = tasks.into_iter().skip(big.len()).collect();
    let total = big.len() + rest.len();
    eprintln!("{total} runs ({} of files over 48 MB), {} files Polysmelt doesn't read", big.len(), unsupported.len());

    let out = Arc::new(Mutex::new(std::fs::File::create(&out_path).unwrap()));
    for u in &unsupported {
        writeln!(out.lock().unwrap(), "{u}").unwrap();
    }
    let done = Arc::new(Mutex::new(0usize));
    let started = Instant::now();
    for (queue, workers) in [(big, jobs.min(2)), (rest, jobs)] {
        let queue = Arc::new(Mutex::new(queue));
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let (queue, out, done, exe, glb_dir) = (queue.clone(), out.clone(), done.clone(), exe.clone(), glb_dir.clone());
                std::thread::spawn(move || {
                    loop {
                        let Some(t) = queue.lock().unwrap().pop_front() else { break };
                        let glb = glb_dir.as_ref().filter(|_| t.loader.contains("GLB")).map(|d| {
                            d.join(slug(t.loader.split(' ').next().unwrap())).join(&t.corpus).join(format!("{}.glb", t.rel))
                        });
                        let mut r = run_child(&exe, &t, glb.as_deref(), timeout, limit_mb);
                        r["corpus"] = json!(t.corpus);
                        r["file"] = json!(t.rel);
                        r["ext"] = json!(t.ext);
                        r["bytes"] = json!(t.bytes);
                        r["loader"] = json!(t.loader);
                        if let Some(g) = &glb {
                            if r["status"] == "ok" {
                                r["glb"] = json!(g.to_string_lossy().replace('\\', "/"));
                            }
                        }
                        writeln!(out.lock().unwrap(), "{r}").unwrap();
                        let mut d = done.lock().unwrap();
                        *d += 1;
                        if *d % 250 == 0 || r["status"] != "ok" && r["status"] != "err" {
                            eprintln!(
                                "[{}/{total} {:.0}s] {} {} {}: {}",
                                *d,
                                started.elapsed().as_secs_f64(),
                                t.corpus,
                                t.rel,
                                t.loader,
                                r["status"]
                            );
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
    eprintln!("{total} runs in {:.0}s", started.elapsed().as_secs_f64());
}

fn clone_task(t: &Task) -> Task {
    Task { corpus: t.corpus.clone(), rel: t.rel.clone(), path: t.path.clone(), ext: t.ext.clone(), bytes: t.bytes, loader: t.loader }
}

fn run_child(exe: &Path, t: &Task, glb: Option<&Path>, timeout: Duration, limit_mb: usize) -> Value {
    let mut cmd = Command::new(exe);
    cmd.arg("one").arg(t.loader).arg(&t.path).arg("--limit-mb").arg(limit_mb.to_string());
    if let Some(g) = glb {
        cmd.arg("--glb-out").arg(g);
    }
    let mut child = match cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return json!({ "status": "crash", "error": format!("spawn: {e}") }),
    };
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let ho = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = so.read_to_string(&mut s);
        s
    });
    let he = std::thread::spawn(move || {
        let mut s = Vec::new();
        let _ = se.read_to_end(&mut s);
        String::from_utf8_lossy(&s).into_owned()
    });
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(_) => break None,
        }
    };
    let stdout = ho.join().unwrap_or_default();
    let stderr = he.join().unwrap_or_default();
    let tail: String = {
        let s = stderr.trim();
        let k = s.char_indices().rev().nth(400).map_or(0, |(i, _)| i);
        s[k..].replace('\n', " | ")
    };
    let Some(status) = status else {
        return json!({ "status": "timeout", "error": format!("no result in {:.0}s", timeout.as_secs_f64()), "ms": start.elapsed().as_secs_f64() * 1e3 });
    };
    let mut v: Value = stdout.lines().rev().find_map(|l| serde_json::from_str(l).ok()).unwrap_or(Value::Null);
    if v.get("status").and_then(Value::as_str) == Some("panic") {
        return v;
    }
    if status.success() && v.get("status").is_some() {
        return v;
    }
    let signal = {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            status.signal()
        }
        #[cfg(not(unix))]
        {
            None::<i32>
        }
    };
    let kind = if stderr.contains("memory allocation of") || stderr.contains("bad_alloc") || signal == Some(9) {
        "oom"
    } else {
        "crash"
    };
    if !v.is_object() {
        v = json!({});
    }
    v["status"] = json!(kind);
    v["error"] = json!(format!("{} {tail}", signal.map_or(format!("exit {:?}", status.code()), |s| format!("signal {s}"))));
    v
}

// --- time -----------------------------------------------------------------------------------

fn time(opts: &BTreeMap<String, Vec<String>>) {
    let sweep = std::fs::read_to_string(opt(opts, "sweep").expect("--sweep")).unwrap();
    let budget = Duration::from_secs_f64(opt(opts, "budget").map_or(0.25, |v| v.parse().unwrap()));
    let only = list_opt(opts, "loaders");
    let exts = list_opt(opts, "ext");
    let min_bytes: u64 = opt(opts, "min-bytes").map_or(0, |v| v.parse().unwrap());
    let roots: BTreeMap<String, PathBuf> = opts
        .get("corpus")
        .into_iter()
        .flatten()
        .filter_map(|c| c.split_once('=').map(|(n, d)| (n.to_owned(), PathBuf::from(d))))
        .collect();

    // (corpus, file) -> loaders that read it cleanly.
    let mut ok: BTreeMap<(String, String), (u64, Vec<String>)> = BTreeMap::new();
    for line in sweep.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
        let e = ok.entry((s("corpus"), s("file"))).or_insert((v["bytes"].as_u64().unwrap_or(0), Vec::new()));
        if s("status") == "ok" {
            e.1.push(s("loader"));
        }
    }
    let mut results = Vec::new();
    let mut threads = 0;
    let started = Instant::now();
    let files = ok.len();
    for (i, ((corpus, rel), (bytes, good))) in ok.into_iter().enumerate() {
        let Some(root) = roots.get(&corpus) else { continue };
        let path = root.join(&rel);
        let ext = loaders::ext_of(&path);
        if bytes < min_bytes || exts.as_ref().is_some_and(|x| !x.contains(&ext)) {
            continue;
        }
        // The loaders asked for, of those that read this file cleanly; a Polysmelt variant runs
        // where Polysmelt itself did.
        let wanted: Vec<String> = match &only {
            Some(o) => o.clone(),
            None => {
                let mut w: Vec<String> = loaders_for(&ext).into_iter().map(str::to_owned).collect();
                // The gltf crate's no-image loader reads no image files, so it's timed against
                // Polysmelt reading none either, as well as against Polysmelt's full load.
                if ext == "gltf" || ext == "glb" {
                    w.push(loaders::POLYSMELT_NO_IMAGES.to_owned());
                }
                // assimp reads these formats' first frame alone: timed against Polysmelt doing the
                // same, as well as against its full load.
                if loaders::first_frame_only(&ext) {
                    w.push(loaders::POLYSMELT_FIRST_FRAME.to_owned());
                }
                w
            }
        };
        let ls: Vec<Loader> = wanted
            .iter()
            .filter(|w| good.iter().any(|g| g == *w) || (w.starts_with("polysmelt (") && good.iter().any(|g| g == loaders::POLYSMELT)))
            .filter_map(|w| loaders::by_name(w, &ext))
            .collect();
        if ls.len() < 2 {
            continue;
        }
        // Each file in a process of its own (pinned like this one), all its loaders in that process
        // in rotating turns: a loader that crashes when it runs again in one process costs that
        // file's timings, not the run's.
        let names: Vec<&str> = ls.iter().map(|l| l.name).collect();
        let run = |names: &[&str]| -> Result<Value, String> {
            let out = Command::new(std::env::current_exe().unwrap())
                .arg("time-file")
                .arg(&path)
                .args(["--ext", &ext, "--loaders", &names.join(","), "--budget", &budget.as_secs_f64().to_string()])
                .stderr(Stdio::inherit())
                .output()
                .map_err(|e| format!("timing process didn't start: {e}"))?;
            if !out.status.success() {
                return Err(format!("timing process failed: {}", out.status));
            }
            serde_json::from_slice::<Value>(&out.stdout).map_err(|e| format!("timing process wrote no result: {e}"))
        };
        let r = match run(&names) {
            Ok(v) => {
                threads = threads.max(v["threads"].as_u64().unwrap_or(0));
                v["loaders"].clone()
            }
            // Each loader again in a process of its own, to find the one that crashes and keep the
            // others' timings (no longer interleaved with each other, so a little less even).
            Err(why) => {
                eprintln!("{corpus}/{rel}: {why}; timing each loader alone");
                let mut each = Vec::new();
                for n in &names {
                    match run(&[n]) {
                        Ok(v) => {
                            threads = threads.max(v["threads"].as_u64().unwrap_or(0));
                            each.extend(v["loaders"].as_array().cloned().unwrap_or_default());
                        }
                        Err(why) => {
                            eprintln!("{corpus}/{rel}: {n} alone: {why}");
                            each.push(json!({ "loader": n, "error": why }));
                        }
                    }
                }
                Value::from(each)
            }
        };
        if (i + 1) % 100 == 0 {
            eprintln!("[{}/{files} {:.0}s]", i + 1, started.elapsed().as_secs_f64());
        }
        results.push(json!({ "corpus": corpus, "file": rel, "ext": ext, "bytes": bytes, "loaders": r }));
    }
    let doc = json!({
        "budget_s": budget.as_secs_f64(),
        "threads_after": threads,
        "cpus_allowed": allowed_cpus(),
        "files": results,
    });
    eprintln!("timed {} files in {:.0}s", doc["files"].as_array().unwrap().len(), started.elapsed().as_secs_f64());
    if let Some(p) = opt(opts, "json") {
        std::fs::write(p, serde_json::to_string(&doc).unwrap()).unwrap();
    }
}

fn allowed_cpus() -> Option<String> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines().find(|l| l.starts_with("Cpus_allowed_list:")).map(|l| l.split_whitespace().nth(1).unwrap_or("").to_owned())
}

/// `time-file <file> --ext <ext> --loaders a,b --budget <s>`: one file's timings, as JSON on stdout.
fn time_one(pos: &[String], opts: &BTreeMap<String, Vec<String>>) {
    let path = Path::new(pos.first().expect("file"));
    let ext = opt(opts, "ext").map_or_else(|| loaders::ext_of(path), str::to_owned);
    let budget = Duration::from_secs_f64(opt(opts, "budget").map_or(0.25, |v| v.parse().unwrap()));
    let ls: Vec<Loader> =
        list_opt(opts, "loaders").unwrap_or_default().iter().filter_map(|w| loaders::by_name(w, &ext)).collect();
    let r = time_file(path, &ls, budget);
    let threads = proc_status().map_or(0, |s| s.2);
    println!("{}", json!({ "loaders": r, "threads": threads }));
}

fn time_file(path: &Path, ls: &[Loader], budget: Duration) -> Vec<Value> {
    let src = Src::File(path);
    let _ = std::fs::read(path);
    let measure = |l: &Loader| -> Result<(Duration, usize, usize), String> {
        let base = CURRENT.load(Relaxed);
        PEAK.store(base, Relaxed);
        let start = Instant::now();
        let out = (l.run)(&src)?;
        let t = start.elapsed();
        let peak = PEAK.load(Relaxed) - base;
        drop(std::hint::black_box(out.result));
        Ok((t, peak, out.units))
    };
    // A warm-up round, which also gives each loader's peak Rust heap.
    let mut first = Vec::new();
    for l in ls {
        first.push(measure(l));
    }
    let round: Duration = first.iter().filter_map(|r| r.as_ref().ok()).map(|r| r.0).sum();
    let min_rounds = if round > Duration::from_secs(20) {
        1
    } else if round > Duration::from_secs(4) {
        3
    } else {
        5
    };
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); ls.len()];
    let start = Instant::now();
    let mut r = 0;
    while r < min_rounds || (start.elapsed() < budget && r < 100) {
        for k in 0..ls.len() {
            let i = (k + r) % ls.len();
            if first[i].is_ok() {
                if let Ok((t, _, _)) = measure(&ls[i]) {
                    times[i].push(t.as_secs_f64() * 1e3);
                }
            }
        }
        r += 1;
    }
    ls.iter()
        .zip(first)
        .zip(times)
        .map(|((l, f), mut t)| {
            t.sort_by(f64::total_cmp);
            match f {
                Ok((_, peak, units)) => json!({
                    "loader": l.name,
                    "median_ms": t.get(t.len() / 2).copied(),
                    "min_ms": t.first().copied(),
                    "rounds": t.len(),
                    "heap_peak": peak,
                    "units": units,
                }),
                Err(e) => json!({ "loader": l.name, "error": e }),
            }
        })
        .collect()
}

