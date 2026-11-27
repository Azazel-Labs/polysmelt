//! Prints what a model holds: its format, what the file says about itself, what it draws and
//! how big it is, and its animations.
//!
//! ```text
//! cargo run --release --example inspect -- model.fbx
//! ```

use polysmelt::document::Pose;
use polysmelt::serde_json::Value;
use polysmelt::source::Fact;
use polysmelt::{Format, open};

fn fact<T: std::fmt::Debug>(f: &Fact<T>) -> String {
    match f {
        Fact::Declared(v) => format!("{v:?}"),
        Fact::Assumed(v) => format!("{v:?} (assumed)"),
        Fact::Unknown => "unknown".into(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: inspect MODEL");
        std::process::exit(2);
    };
    let by_extension = std::path::Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .and_then(Format::from_extension);
    let doc = open(&path)?;

    let info = doc.source_info();
    let unknown = || "unknown".to_owned();
    println!("{path}");
    println!(
        "  format     {} {}",
        info.format
            .clone()
            .or(by_extension.map(|f| f.name().to_owned()))
            .unwrap_or_else(unknown),
        info.version.as_deref().unwrap_or_default()
    );
    if let Some(tool) = &info.tool {
        println!("  written by {tool}");
    }
    println!("  unit       {} m", fact(&info.meters_per_unit));
    println!("  up         {}", fact(&info.up_axis));
    println!("  handedness {}", fact(&info.handedness));

    let counts = doc.counts();
    println!(
        "  draws      {} triangles, {} lines, {} points, {} vertices",
        counts.triangles, counts.lines, counts.points, counts.vertices
    );
    let json = doc.json();
    let len = |key: &str| json.get(key).and_then(Value::as_array).map_or(0, Vec::len);
    println!(
        "  holds      {} nodes, {} meshes, {} materials, {} images, {} skins",
        len("nodes"),
        len("meshes"),
        len("materials"),
        len("images"),
        len("skins")
    );
    if let Some(b) = doc.bounds(Pose::Rest)? {
        let [x, y, z] = b.size();
        println!("  size       {x} x {y} x {z} (in its output units)");
    }

    let animations = json.get("animations").and_then(Value::as_array);
    for (i, a) in animations.into_iter().flatten().enumerate() {
        let name = a.get("name").and_then(Value::as_str).unwrap_or("(unnamed)");
        let reach = match doc.animation_bounds(i, 16)? {
            Some(b) => format!("{:?} to {:?}", b.min, b.max),
            None => "nothing drawn".into(),
        };
        println!("  animation  {i} {name}: reaches {reach}");
    }
    Ok(())
}
