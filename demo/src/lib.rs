//! The polysmelt web demo: drop a model in any format the crate reads and see it lit, staged and
//! animated, then take it away as GLB.
//!
//! JavaScript does the page (files, drag and drop, folders, pointer input, the frame loop); this
//! crate reads the model with polysmelt and draws it with wgpu, on WebGPU or WebGL 2.

pub mod camera;
pub mod model;
pub mod render;

#[cfg(target_arch = "wasm32")]
mod web;
