//! Converts any model Polysmelt reads to a GLB.
//!
//! ```text
//! cargo run --release --example convert -- in.fbx out.glb
//! cargo run --release --example convert -- in.fbx out.glb --meshopt
//! cargo run --release --example convert -- in.dae out.glb --meshes
//! ```
//!
//! The textures and libraries the model names are read from beside it and embedded.
//! `--meshopt` compresses the GLB with `KHR_meshopt_compression`; `--meshes` keeps only meshes,
//! materials and textures (no animations, skins, morph targets, cameras or lights).

use std::io::BufWriter;

use polysmelt::gltf::CompressOptions;
use polysmelt::{Keep, Loader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (paths, flags): (Vec<&String>, Vec<&String>) = args.iter().partition(|a| !a.starts_with("--"));
    let [input, output] = paths.as_slice() else {
        eprintln!("usage: convert IN OUT.glb [--meshopt] [--meshes]");
        std::process::exit(2);
    };
    let flag = |name: &str| flags.iter().any(|f| *f == name);
    let keep = if flag("--meshes") { Keep::MESHES } else { Keep::ALL };
    let doc = Loader::new().keep(keep).open(input)?;
    if flag("--meshopt") {
        std::fs::write(output, doc.to_glb_with(&CompressOptions::KHR)?)?;
    } else {
        doc.write_glb(BufWriter::new(std::fs::File::create(output)?))?;
    }
    let counts = doc.counts();
    println!(
        "{input} -> {output}: {} triangles, {} vertices",
        counts.triangles, counts.vertices
    );
    Ok(())
}
