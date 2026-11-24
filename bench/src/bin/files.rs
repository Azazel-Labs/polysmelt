//! Load speed and peak memory on real files: Polysmelt against the competitor crate for each
//! format (see `loaders.rs`), each reading the file from disk into its own in-memory result.
//!
//! Loaders take turns in a rotating order, so drift on a shared machine hits all of them alike;
//! the table gives the median and the fastest round. A counting allocator gives the peak heap
//! during a load and the heap the result keeps, exact however noisy the machine.
//!
//! ```text
//! cargo run --release --features evidence --bin files -- <file>... [--budget 3] [--json out.json] [--markdown out.md]
//! ```

#[path = "../counting.rs"]
mod counting;
#[path = "../loaders.rs"]
mod loaders;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

use counting::{CURRENT, PEAK};
use loaders::{Loader, Run, Src};
use serde_json::json;

#[global_allocator]
static ALLOC: counting::Counting = counting::Counting;

struct Sample {
    time: Duration,
    peak: usize,
    kept: usize,
    units: usize,
}

fn measure(run: Run, path: &Path) -> Result<Sample, String> {
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    let start = Instant::now();
    let loaders::Loaded { result, units, .. } = run(&Src::File(path))?;
    let time = start.elapsed();
    let (peak, kept) = (PEAK.load(Relaxed) - base, CURRENT.load(Relaxed).saturating_sub(base));
    drop(std::hint::black_box(result));
    Ok(Sample { time, peak, kept, units })
}

fn loaders_for(path: &Path) -> Vec<Loader> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let mut out = vec![Loader { name: "polysmelt", run: loaders::polysmelt }];
    // Only the OBJ reader splits its work over threads; elsewhere both settings run the same code.
    if ext == "obj" {
        out.push(Loader { name: "polysmelt (threads: 0, all cores)", run: loaders::polysmelt_all_cores });
    }
    out.extend(loaders::competitors(&ext));
    out
}

fn mb(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut files = Vec::new();
    let mut opts = std::collections::HashMap::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(k) = a.strip_prefix("--") {
            opts.insert(k.to_owned(), it.next().cloned().unwrap_or_default());
        } else {
            files.push(PathBuf::from(a));
        }
    }
    let budget = Duration::from_secs_f64(opts.get("budget").map_or(3.0, |b| b.parse().expect("--budget seconds")));
    let names_from = opts.get("root").map(PathBuf::from);
    let mut md = String::new();
    let mut results = Vec::new();
    writeln!(
        md,
        "Median and fastest of interleaved rounds ({budget:?} budget per file); peak heap during the load and heap the \
         result keeps. `×` is relative to Polysmelt: above 1 means Polysmelt wins. Units: triangles (voxels for VOX, \
         channels × frames for BVH).\n"
    )
    .unwrap();
    for path in &files {
        let name = names_from
            .as_ref()
            .and_then(|r| path.strip_prefix(r).ok())
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);
        eprintln!("{name} ({:.1} MB)", mb(bytes));
        let loaders = loaders_for(path);
        // Warm-up round: memory figures, and which loaders read the file at all.
        let first: Vec<Result<Sample, String>> = loaders.iter().map(|l| measure(l.run, path)).collect();
        let mut times: Vec<Vec<Duration>> = loaders.iter().map(|_| Vec::new()).collect();
        let start = Instant::now();
        let mut round = 0;
        while round < 5 || (start.elapsed() < budget && round < 200) {
            for k in 0..loaders.len() {
                let i = (k + round) % loaders.len();
                if first[i].is_ok() {
                    times[i].push(measure(loaders[i].run, path).unwrap().time);
                }
            }
            round += 1;
        }
        writeln!(
            md,
            "### {name} — {:.1} MB\n\n| Loader | Median ms | Fastest ms | × time | Peak MB | × peak | Kept MB | Units | Rounds |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            mb(bytes)
        )
        .unwrap();
        let mine = times[0].clone();
        let my_median = if mine.is_empty() { None } else {
            let mut t = mine;
            t.sort();
            Some(t[t.len() / 2])
        };
        let my_peak = first[0].as_ref().ok().map(|s| s.peak);
        for ((l, s), mut t) in loaders.iter().zip(&first).zip(times) {
            match s {
                Ok(s) => {
                    t.sort();
                    let (median, best) = (t[t.len() / 2], t[0]);
                    let xt = my_median.map_or(f64::NAN, |m| median.as_secs_f64() / m.as_secs_f64());
                    let xp = my_peak.map_or(f64::NAN, |m| s.peak as f64 / m as f64);
                    writeln!(
                        md,
                        "| {} | {:.2} | {:.2} | {:.2} | {:.1} | {:.2} | {:.1} | {} | {} |",
                        if l.name == "polysmelt" { "**polysmelt**" } else { l.name },
                        median.as_secs_f64() * 1e3,
                        best.as_secs_f64() * 1e3,
                        xt,
                        mb(s.peak),
                        xp,
                        mb(s.kept),
                        s.units,
                        t.len()
                    )
                    .unwrap();
                    results.push(json!({
                        "file": name, "bytes": bytes, "loader": l.name, "median_ns": median.as_nanos() as u64,
                        "fastest_ns": best.as_nanos() as u64, "rounds": t.len(), "peak_bytes": s.peak,
                        "kept_bytes": s.kept, "units": s.units,
                    }));
                }
                Err(e) => {
                    let e: String = e.chars().take(120).collect();
                    writeln!(md, "| {} | failed: {} | | | | | | | |", l.name, e.replace('|', "/")).unwrap();
                    results.push(json!({ "file": name, "bytes": bytes, "loader": l.name, "error": e }));
                }
            }
        }
        md.push('\n');
    }
    println!("{md}");
    if let Some(p) = opts.get("markdown") {
        std::fs::write(p, &md).unwrap();
    }
    if let Some(p) = opts.get("json") {
        let doc = json!({ "os": std::env::consts::OS, "arch": std::env::consts::ARCH, "results": results });
        std::fs::write(p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    }
}
