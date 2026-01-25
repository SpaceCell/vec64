use wasm_bindgen::prelude::*;
use rayon::prelude::*;
use vec64::Vec64;

pub use vec64::init_thread_pool;

/// Test parallel sum on Vec64
#[wasm_bindgen]
pub fn par_sum(len: usize) -> u64 {
    let v: Vec64<u64> = (0..len as u64).collect();
    v.par_iter().sum()
}

/// Expected result for verification
#[wasm_bindgen]
pub fn expected_sum(len: usize) -> u64 {
    let n = len as u64;
    n * (n - 1) / 2
}
