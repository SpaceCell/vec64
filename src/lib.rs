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
#[cfg(feature = "append_only_vec")]
pub mod append_only_vec;
#[cfg(feature = "global")]
pub mod global;
#[cfg(all(feature = "mmap", target_os = "linux"))]
pub mod mmap_alloc;
pub mod page_aligned;
pub mod vec64;

pub use page_aligned::{PageAligned, page_align_bitmask_step, page_align_t_step};
pub use vec64::Vec64;
pub use alloc64::Alloc64;
#[cfg(feature = "append_only_vec")]
pub use append_only_vec::AppendOnlyVec;
#[cfg(all(feature = "mmap", target_os = "linux"))]
pub use mmap_alloc::MAllocPg64;
#[cfg(feature = "global")]
pub use global::Alloc64Global;
#[cfg(all(feature = "global", feature = "mmap", target_os = "linux"))]
pub use global::MAllocPg64Global;

// The mmap-backed allocator depends on Linux-only syscalls (mmap, mremap,
// madvise). When the `mmap` feature is enabled on a non-Linux target we
// silently fall back to the standard `Alloc64` allocator so that downstream
// crates can ship a single feature set portably; the mmap optimisation
// activates only where the kernel supports it.

/// Allocator type backing Vec64, determined by feature flags.
#[cfg(all(feature = "mmap", target_os = "linux"))]
pub type Vec64Alloc = mmap_alloc::MAllocPg64;

/// Allocator type backing Vec64, determined by feature flags.
///
/// On non-Linux targets the `mmap` feature has no effect and the standard
/// `Alloc64` allocator is used instead.
#[cfg(any(not(feature = "mmap"), all(feature = "mmap", not(target_os = "linux"))))]
pub type Vec64Alloc = alloc64::Alloc64;

#[cfg(feature = "wasm")]
pub use wasm_bindgen_rayon::init_thread_pool;
