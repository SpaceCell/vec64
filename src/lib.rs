//! # Vec64
//!
//! High-performance Rust vector type with automatic 64-byte SIMD alignment.
//!
//! ## Summary
//! `Vec64<T>` is a drop-in replacement for `Vec<T>` that ensures the starting pointer is aligned to a 64-byte boundary.
//! This alignment is useful for optimal performance with SIMD instruction extensions like AVX-512, and helps avoid split loads/stores across cache lines.
//!
//! Benefits will vary based on one's target architecture.
//!
//! ## WASM Compatibility
//!
//! Enable the `wasm` feature for Web Worker-based parallelism:
//!
//! ```toml
//! vec64 = { version = "0.3", features = ["wasm"] }
//! ```
//!
//! Required build configuration (`.cargo/config.toml`):
//!
//! ```toml
//! [target.wasm32-unknown-unknown]
//! rustflags = ["-C", "target-feature=+atomics,+bulk-memory"]
//!
//! [unstable]
//! build-std = ["panic_abort", "std"]
//! ```
//!
//! Build with `wasm-pack`:
//!
//! ```bash
//! wasm-pack build --target web
//! ```
//!
//! Initialise the thread pool from JavaScript before using parallel methods:
//!
//! ```javascript
//! import init, { initThreadPool } from './pkg/your_crate.js';
//! await init();
//! await initThreadPool(navigator.hardwareConcurrency);
//! ```
//!
//! Note: Requires nightly Rust with `rust-src` component for `build-std`. The browser
//! needs SharedArrayBuffer support (cross-origin isolation via COOP/COEP headers).
//! See `wasm-test/` for a complete working example.

#![feature(allocator_api)]
#![feature(slice_ptr_get)]

pub mod alloc64;
pub mod vec64;

pub use vec64::Vec64;
pub use alloc64::Alloc64;

#[cfg(feature = "wasm")]
pub use wasm_bindgen_rayon::init_thread_pool;
