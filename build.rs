use std::env;
use std::fs;
use std::path::Path;

const DEFAULT_HUGE_PAGE_BYTES: usize = 256 * 1024;
const MIN_HUGE_PAGE_BYTES: usize = 4096;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VEC64_HUGE_PAGE_BYTES");

    // Only generate the constant when the mmap feature is enabled;
    // otherwise the mmap_alloc module is not compiled and the file
    // would not be included.
    if env::var_os("CARGO_FEATURE_MMAP").is_none() {
        return;
    }

    let value: usize = match env::var("VEC64_HUGE_PAGE_BYTES") {
        Ok(raw) => raw
            .parse()
            .unwrap_or_else(|_| panic!("VEC64_HUGE_PAGE_BYTES must be a positive integer; got {raw:?}")),
        Err(_) => DEFAULT_HUGE_PAGE_BYTES,
    };

    if value < MIN_HUGE_PAGE_BYTES {
        panic!(
            "VEC64_HUGE_PAGE_BYTES must be at least {MIN_HUGE_PAGE_BYTES}; got {value}"
        );
    }
    if !value.is_power_of_two() {
        panic!("VEC64_HUGE_PAGE_BYTES must be a power of two; got {value}");
    }

    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR not set");
    let path = Path::new(&out_dir).join("huge_page.rs");
    fs::write(
        &path,
        format!("pub(crate) const HUGE_PAGE: usize = {value};\n"),
    )
    .expect("failed to write huge_page.rs");
}
