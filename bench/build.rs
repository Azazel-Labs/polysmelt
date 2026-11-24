//! With the `assimp` feature, compiles the C++ shim the evidence binaries call assimp through and
//! links assimp, found under `ASSIMP_DIR` (an install prefix: `include/` and `lib/`). Without it,
//! nothing.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "assimp")]
    {
        println!("cargo:rerun-if-changed=shim/assimp_shim.cpp");
        println!("cargo:rerun-if-env-changed=ASSIMP_DIR");
        let dir = std::env::var("ASSIMP_DIR").expect("set ASSIMP_DIR to assimp's install prefix");
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("shim/assimp_shim.cpp")
            .include(format!("{dir}/include"))
            .warnings(false)
            .compile("assimp_shim");
        println!("cargo:rustc-link-search=native={dir}/lib");
        println!("cargo:rustc-link-lib=dylib=assimp");
        // The binaries find libassimp beside themselves (the workflow ships it there), or where it was built.
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN:{dir}/lib");
    }
}
