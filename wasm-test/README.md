# WASM Test

Minimal browser test for Vec64's WASM parallel processing support.

## Prerequisites

- Rust nightly with `rust-src` component: `rustup component add rust-src`
- `wasm-pack`: `cargo install wasm-pack`
- Python 3 (for the test server)

## Build

```bash
wasm-pack build --target web
```

## Run

```bash
python serve.py
```

Open http://localhost:8080 in a browser. The page will show PASS or FAIL.

## What it tests

- WASM module loads correctly
- Thread pool initialises via `initThreadPool()`
- `par_iter().sum()` returns the correct result

## Notes

- The server sets COOP/COEP headers required for SharedArrayBuffer
- Requires nightly toolchain for `build-std` support
