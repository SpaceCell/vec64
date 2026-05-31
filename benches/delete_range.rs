//! Benchmark: middle-range delete on a large mmap-backed Vec64<u64>.
//!
//! Three paths compared:
//!   1. `Vec64::delete_range`   - splice via mmap + 2x mremap (this crate's feature).
//!   2. `Vec::drain(start..end)` - in-place memmove of the tail leftward.
//!   3. realloc-and-copy        - allocate fresh Vec64, copy head + copy tail,
//!                                drop old. Mirrors what a naive caller does
//!                                without an in-place delete primitive.
//!
//! Construction is excluded from every timed window.

use std::time::{Duration, Instant};

use vec64::Vec64;

const ELEM_BYTES: usize = std::mem::size_of::<u64>();

/// Allocate a fresh Vec64<u64> of length `n`, contents `0..n as u64`.
/// Called outside the timed window.
fn prep(n: usize) -> Vec64<u64> {
    let mut v: Vec64<u64> = Vec64::with_capacity(n);
    for i in 0..n as u64 {
        v.push(i);
    }
    v
}

/// Verify the post-delete contents match `0..start ++ end..n`.
fn assert_correct(v: &Vec64<u64>, n: usize, start: usize, end: usize) {
    let expected_len = n - (end - start);
    assert_eq!(v.len(), expected_len);
    assert_eq!(v[0], 0);
    if start > 0 {
        assert_eq!(v[start - 1], (start - 1) as u64);
    }
    assert_eq!(v[start], end as u64);
    assert_eq!(v[v.len() - 1], (n - 1) as u64);
}

/// Median of `samples` durations.
fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn bench_path<F>(name: &str, n: usize, start: usize, end: usize, iters: usize, mut path: F)
where
    F: FnMut(Vec64<u64>) -> Vec64<u64>,
{
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let v = prep(n);                    // untimed
        let t0 = Instant::now();
        let v = path(v);                    // timed
        let elapsed = t0.elapsed();
        assert_correct(&v, n, start, end);  // untimed
        drop(v);                            // untimed
        times.push(elapsed);
    }
    let med = median(&mut times);
    let bytes_moved = (n - end) * ELEM_BYTES;
    let throughput_gibs = if med.as_secs_f64() > 0.0 {
        (bytes_moved as f64) / med.as_secs_f64() / (1024.0 * 1024.0 * 1024.0)
    } else {
        f64::INFINITY
    };
    println!(
        "  {name:<22} median {:>10.3?}  (tail bytes moved: {:>10} MiB, ~{:>7.2} GiB/s)",
        med,
        bytes_moved / (1024 * 1024),
        throughput_gibs,
    );
}

fn run_size(label: &str, n: usize, delete_frac_lo: f64, delete_frac_hi: f64, iters: usize) {
    let start_unaligned = (n as f64 * delete_frac_lo) as usize;
    let end_unaligned = (n as f64 * delete_frac_hi) as usize;
    // Snap to page boundaries (4096 byte = 512 u64).
    let elems_per_page = 4096 / ELEM_BYTES;
    let start = (start_unaligned / elems_per_page) * elems_per_page;
    let end = (end_unaligned / elems_per_page) * elems_per_page;
    let delete_bytes = (end - start) * ELEM_BYTES;

    println!(
        "\n[{label}] n = {} elems ({} MiB), delete [{start}, {end}) = {} MiB, iters = {iters}",
        n,
        (n * ELEM_BYTES) / (1024 * 1024),
        delete_bytes / (1024 * 1024),
    );

    bench_path("splice (delete_range)", n, start, end, iters, |mut v| {
        v.delete_range(start, end);
        v
    });

    bench_path("Vec::drain (memmove)", n, start, end, iters, |mut v| {
        v.0.drain(start..end);
        v
    });

    bench_path("realloc + copy", n, start, end, iters, |v| {
        let new_len = n - (end - start);
        let mut out: Vec64<u64> = Vec64::with_capacity(new_len);
        // SAFETY: `out` has capacity for new_len elements and we write exactly
        // that many; the two spans live in distinct allocations so
        // copy_nonoverlapping is sound.
        unsafe {
            let src = v.as_ptr();
            let dst = out.as_mut_ptr();
            std::ptr::copy_nonoverlapping(src, dst, start);
            std::ptr::copy_nonoverlapping(src.add(end), dst.add(start), n - end);
            out.set_len(new_len);
        }
        drop(v);
        out
    });
}

fn main() {
    // Modest warmup to settle the allocator and page faults.
    let _warm = prep(1 << 20);
    drop(_warm);

    //  16 MiB:  2 M elems  (mmap-backed)
    //  64 MiB:  8 M elems
    // 256 MiB: 32 M elems  (point where splice wins clearly)
    run_size("16 MiB", 2 * 1024 * 1024,      0.25, 0.75, 9);
    run_size("64 MiB", 8 * 1024 * 1024,      0.25, 0.75, 9);
    run_size("256 MiB", 32 * 1024 * 1024,    0.25, 0.75, 5);

    // Asymmetric deletes: small head, big tail. Splice avoids moving the
    // entire tail; drain has to memmove all of it leftward.
    run_size("64 MiB, head-heavy delete", 8 * 1024 * 1024,  0.02, 0.10, 9);
    run_size("64 MiB, tail-heavy delete", 8 * 1024 * 1024,  0.85, 0.95, 9);
}
