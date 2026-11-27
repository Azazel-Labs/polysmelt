//! One classic material, written in every format that can hold it, reads as one glTF material: the same base
//! color, metalness, roughness, highlight and alpha mode whatever file it came from.
//!
//! The files are the corpus's `same_phong` cases, each written by its format's generator from
//! `tools/corpus/common.py`'s `SAME_PHONG`: a diffuse color of sRGB (0.8, 0.2, 0.1), a specular color of 0.5 gray,
//! a Blinn-Phong exponent of 20, opaque. Each format stores them its own way (a glossiness of log2(20) / 10 in
//! 3DS, of (log2(20) - 2) / 10 and a specular level of 0.5 in `LightWave`, a shininess of 20 / 128 in X3D), and
//! the expected numbers here are worked out by hand from the rule, not read back from the reader. `Blitz3D` is
//! left out: a brush has one number for its highlight's strength and size, so it can't hold this material.

#![cfg(feature = "fs")]

use std::path::PathBuf;

use serde_json::Value;

/// (whether the format's reader is built, the case's file).
const FILES: &[(bool, &str)] = &[
    (cfg!(feature = "obj"), "obj/same_phong/model.obj"),
    (cfg!(feature = "dae"), "dae/same_phong/model.dae"),
    (cfg!(feature = "3ds"), "3ds/same_phong/model.3ds"),
    (cfg!(feature = "directx"), "x/same_phong/model.x"),
    (cfg!(feature = "fbx"), "fbx/same_phong/same_phong.fbx"),
    (cfg!(feature = "x3d"), "x3d/same_phong/model.x3d"),
    (cfg!(feature = "ac3d"), "ac/same_phong/model.ac"),
    (cfg!(feature = "ms3d"), "ms3d/same_phong/model.ms3d"),
    (cfg!(feature = "lwo"), "lwo/same_phong/model.lwo"),
];

/// sRGB 0.8, 0.2 and 0.1 as linear values: ((c + 0.055) / 1.055)^2.4.
const BASE: [f64; 3] = [0.603_827, 0.033_105, 0.010_023];
/// (2 / (20 + 2))^(1/4).
const ROUGHNESS: f64 = 0.549_100;
/// sRGB 0.5 as a linear value: how much of glTF's highlight a 0.5 gray specular color leaves.
const SPECULAR: f64 = 0.214_041;

/// What a material shows: base color, metalness, roughness, specular factor, specular color and alpha mode.
#[derive(Debug)]
struct Look {
    base: [f64; 4],
    metallic: f64,
    roughness: f64,
    specular: f64,
    specular_color: [f64; 3],
    alpha_mode: String,
}

fn numbers<const N: usize>(v: Option<&Value>, default: [f64; N]) -> [f64; N] {
    let Some(a) = v.and_then(Value::as_array) else {
        return default;
    };
    std::array::from_fn(|k| a.get(k).and_then(Value::as_f64).unwrap_or(default[k]))
}

/// The first material of the document `rel` loads as, with glTF's defaults for what it leaves out.
fn look(rel: &str) -> Look {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus").join(rel);
    let doc = polysmelt::open(&path).unwrap_or_else(|e| panic!("{rel}: {e}"));
    let materials = doc.json().get("materials").and_then(Value::as_array);
    let m = materials
        .and_then(|m| m.first())
        .unwrap_or_else(|| panic!("{rel}: no materials"));
    assert_eq!(materials.map(Vec::len), Some(1), "{rel}: one material");
    let pbr = &m["pbrMetallicRoughness"];
    let spec = m.pointer("/extensions/KHR_materials_specular");
    Look {
        base: numbers(pbr.get("baseColorFactor"), [1.0; 4]),
        metallic: pbr.get("metallicFactor").and_then(Value::as_f64).unwrap_or(1.0),
        roughness: pbr.get("roughnessFactor").and_then(Value::as_f64).unwrap_or(1.0),
        specular: spec
            .and_then(|s| s.get("specularFactor"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0),
        specular_color: numbers(spec.and_then(|s| s.get("specularColorFactor")), [1.0; 3]),
        alpha_mode: m
            .get("alphaMode")
            .and_then(Value::as_str)
            .unwrap_or("OPAQUE")
            .to_owned(),
    }
}

#[test]
fn the_same_phong_material_reads_the_same_from_every_format() {
    // A format's own number of digits (a float, a percentage, a byte) moves the values a little.
    const TOLERANCE: f64 = 2e-3;
    let close = |a: f64, b: f64| (a - b).abs() <= TOLERANCE;
    let mut failures = Vec::new();
    let mut read = 0;
    for &(built, rel) in FILES {
        if !built {
            continue;
        }
        read += 1;
        let l = look(rel);
        let same = l.base[..3].iter().zip(BASE).all(|(a, b)| close(*a, b))
            && close(l.base[3], 1.0)
            && close(l.metallic, 0.0)
            && close(l.roughness, ROUGHNESS)
            && close(l.specular, SPECULAR)
            && l.specular_color.iter().all(|c| close(*c, 1.0))
            && l.alpha_mode == "OPAQUE";
        if !same {
            failures.push(format!("{rel}: {l:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "expected base {BASE:?}, metallic 0, roughness {ROUGHNESS}, specular {SPECULAR} white, OPAQUE; got\n{}",
        failures.join("\n")
    );
    assert!(read > 0 || FILES.iter().all(|(built, _)| !built));
}
