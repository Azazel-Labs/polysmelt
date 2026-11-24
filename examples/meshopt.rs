//! Meshopt-compresses or decompresses a GLB.
//!
//! ```text
//! cargo run --example meshopt -- compress     in.glb out.glb
//! cargo run --example meshopt -- compress-khr in.glb out.glb
//! cargo run --example meshopt -- decompress   in.glb out.glb
//! cargo run --example meshopt -- pack         in.gltf out.glb
//! cargo run --example meshopt -- sample       out.glb
//! ```
//!
//! `compress` writes `EXT_meshopt_compression`, losslessly; `compress-khr` writes
//! `KHR_meshopt_compression` at level 3 with vertex colors `COLOR`-filtered (gltfpack's `-cz`
//! codec settings). `sample` writes a small compressed test file (a textured, wavy grid) without
//! any input.

use std::path::Path;

use polysmelt::gltf;
use polysmelt::mesh::{Material, Mesh, Prim};

#[allow(clippy::cast_precision_loss, reason = "a 25 by 25 grid")]
fn sample() -> polysmelt::Result<Vec<u8>> {
    let n = 24u32;
    let mut prim = Prim::default();
    for y in 0..=n {
        for x in 0..=n {
            let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
            prim.positions
                .push([u - 0.5, (u * 9.0).sin() * (v * 5.0).cos() * 0.05, v - 0.5]);
            prim.normals.push([0.0, 1.0, 0.0]);
            prim.uvs.push([u, v]);
        }
    }
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            prim.indices.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
        }
    }
    let mesh = Mesh {
        prims: vec![prim],
        materials: vec![Material::plain("grid", [0.8, 0.6, 0.3, 1.0])],
        images: Vec::new(),
        skeleton: Vec::new(),
    };
    gltf::compress(&mesh.to_glb()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (out, glb) = match args.as_slice() {
        [cmd, input, out] => {
            let data = std::fs::read(input)?;
            let glb = match cmd.as_str() {
                "compress" => gltf::compress(&data)?,
                "compress-khr" => {
                    let options = gltf::CompressOptions::KHR.level(3).color_filter(true);
                    gltf::compress_with(&data, options)?
                }
                "decompress" => gltf::decompress(&data)?,
                "pack" => {
                    let dir = Path::new(input).parent().unwrap_or(Path::new("."));
                    gltf::pack_glb(&data, gltf::files_in(dir))?
                }
                c => return Err(format!("unknown command {c}").into()),
            };
            (out, glb)
        }
        [cmd, out] if cmd == "sample" => (out, sample()?),
        _ => {
            return Err("usage: meshopt compress|compress-khr|decompress|pack <in> <out> | sample <out>".into());
        }
    };
    std::fs::write(out, &glb)?;
    println!("wrote {} bytes to {out}", glb.len());
    Ok(())
}
