# Vec64

High-performance Rust vector type with automatic 64-byte SIMD alignment.

Requires nightly Rust.

## Overview

`Vec64<T>` is a drop-in replacement for `Vec<T>` that guarantees the starting pointer is aligned to a 64-byte boundary, enabling optimal performance with SIMD instruction extensions like AVX-512.

Vec64 is the foundational buffer type behind [Minarrow](https://github.com/pbow/minarrow) - a typed, high-performance columnar data library for Rust. If you're building data-intensive applications and want aligned buffers, automatic padding, parallel processing, and Arrow-compatible columnar structures out of the box, Minarrow has it for you.

## Quick Start

```toml
[dependencies]
vec64 = "0.4"
```

```rust
use vec64::{Vec64, vec64};

let mut v = Vec64::new();
v.push(42);

let v = vec64![1, 2, 3, 4, 5];

let v = Vec64::from_slice(&[1, 2, 3]);
```

All standard `Vec` operations work as expected.

## License

Apache-2.0

## Backed by SpaceCell

Vec64 is backed by SpaceCell, ensuring ongoing support for the project.

For an edge in high-performance data computing, visit [spacecell.com](http://spacecell.com).
