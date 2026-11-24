# polysmelt web demo

The page at <https://azazel-labs.github.io/polysmelt/>: drop a model (one file, several, or a folder) and polysmelt
reads it in WebAssembly, the page draws it with wgpu (WebGPU, or WebGL 2 where there's none), plays its animations, and
gives it back as GLB.

- `src/model.rs`: picking the model out of what was dropped, loading it with `formats::load_bytes` from an
  `assets::Memory`, posing it (skins on the GPU when the rig allows, otherwise `flatten` each frame), and the facts the
  panel shows.
- `src/render.rs` and the `.wgsl` files: the stage: PBR-ish shading under a sky and ground light, a shadowed floor
  grid, points and lines, GPU skinning.
- `src/web.rs`: the `Viewer` the page drives through wasm-bindgen.
- `web/`: the page itself (`index.html`, `style.css`, `app.js`).
- `src/bin/media.rs`: the same renderer, native and offscreen, for the README and social images (see
  `../media/compose.sh`).

Every reader that builds for wasm is in; USD isn't, since OpenUSD doesn't build for the web yet.

## Building

Needs the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` at the version in `Cargo.lock`; `wasm-opt` is used
when it's on `PATH`.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version "$(cargo pkgid wasm-bindgen | sed 's/.*@//')" --locked
scripts/build.sh                 # into dist/
python -m http.server -d dist    # then open http://localhost:8000
```

`.github/workflows/demo.yml` does the same on every push to main that touches the demo or the crate, and deploys
`dist/` to GitHub Pages.

The checks CI runs:

```sh
cargo fmt --check
cargo clippy --target wasm32-unknown-unknown -- -D warnings
cargo test
```
