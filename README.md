# Vec64

A Rust vector with a 64-byte-aligned allocation, designed for SIMD and high-performance data processing.

> **Requires nightly Rust.**

## Overview

`Vec64<T>` provides a familiar, `Vec`-like interface while guaranteeing that the allocation begins on a 64-byte boundary. This supports cache-line alignment and wide SIMD operations, including AVX-512.

Alignment applies to the start of the allocation. Actual SIMD performance still depends on element layout, access patterns and the generated instructions.

`Vec64` is the foundational buffer type used by [Minarrow](https://github.com/pbow/minarrow), a typed, Arrow-compatible columnar data library for Rust.

## Installation

```toml
[dependencies]
vec64 = "0.4"
```

## Usage

```rust
use vec64::{vec64, Vec64};

let mut values = Vec64::new();
values.push(42);

let values = vec64![1, 2, 3, 4, 5];

let values = Vec64::from_slice(&[1, 2, 3]);
```

`Vec64` supports the standard vector operations needed for allocation, mutation, iteration and indexed access.

## License

Copyright Peter Garfield Bower 2025-2026.

Licensed under the Apache License 2.0.

## SpaceCell

`Vec64` is maintained by [SpaceCell](https://spacecell.com) and forms part of its open-source foundation for high-performance data computing.
