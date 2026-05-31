//! Benchmark: `Vec64::from_chunks` vs `with_capacity` + per-chunk copy.

use std::time::{Duration, Instant};

use vec64::Vec64;

const ELEM_BYTES: usize = std::mem::size_of::<u64>();
const MIB: usize = 1024 * 1024;

fn prep_chunks(num_chunks: usize, rows_per_chunk: usize) -> Vec<Vec64<u64>> {
    (0..num_chunks)
        .map(|chunk_id| {
            let mut v: Vec64<u64> = Vec64::with_capacity(rows_per_chunk);
            let base = (chunk_id * rows_per_chunk) as u64;
            for i in 0..rows_per_chunk as u64 {
                v.push(base + i);
            }
            v
        })
        .collect()
}

fn assert_correct(out: &Vec64<u64>, num_chunks: usize, rows_per_chunk: usize) {
    let total = num_chunks * rows_per_chunk;
    assert_eq!(out.len(), total);
    assert_eq!(out[0], 0);
    assert_eq!(out[total - 1], (total - 1) as u64);
    let mid = total / 2;
    assert_eq!(out[mid], mid as u64);
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn bench_path<F>(name: &str, num_chunks: usize, rows_per_chunk: usize, iters: usize, mut path: F)
where
    F: FnMut(Vec<Vec64<u64>>) -> Vec64<u64>,
{
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let chunks = prep_chunks(num_chunks, rows_per_chunk);
        let t0 = Instant::now();
        let out = path(chunks);
        let elapsed = t0.elapsed();
        assert_correct(&out, num_chunks, rows_per_chunk);
        drop(out);
        times.push(elapsed);
    }
    let med = median(&mut times);
    let bytes = num_chunks * rows_per_chunk * ELEM_BYTES;
    let throughput_gibs = if med.as_secs_f64() > 0.0 {
        (bytes as f64) / med.as_secs_f64() / (1024.0 * 1024.0 * 1024.0)
    } else {
        f64::INFINITY
    };
    println!(
        "  {name:<26} median {:>10.3?}  ({:>8} MiB, ~{:>7.2} GiB/s)",
        med,
        bytes / MIB,
        throughput_gibs,
    );
}

fn run_case(num_chunks: usize, rows_per_chunk: usize, iters: usize) {
    let chunk_bytes = rows_per_chunk * ELEM_BYTES;
    println!(
        "\n{num_chunks} x {rows_per_chunk} rows  ({} KiB / chunk, {} MiB total, iters = {iters})",
        chunk_bytes / 1024,
        (num_chunks * chunk_bytes) / MIB,
    );

    bench_path("Vec64::from_chunks", num_chunks, rows_per_chunk, iters, |chunks| {
        Vec64::from_chunks(chunks)
    });

    bench_path("with_cap + copy", num_chunks, rows_per_chunk, iters, |chunks| {
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut out: Vec64<u64> = Vec64::with_capacity(total);
        for chunk in chunks {
            let n = chunk.len();
            let current = out.len();
            unsafe {
                let dst = out.as_mut_ptr().add(current);
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, n);
                out.set_len(current + n);
            }
            drop(chunk);
        }
        out
    });
}

fn main() {
    let warm: Vec64<u64> = (0..1 << 20).collect();
    drop(warm);

    let rows_2mib = (2 * MIB) / ELEM_BYTES;

    run_case(4,   rows_2mib,         9);
    run_case(8,   rows_2mib,         9);
    run_case(16,  rows_2mib * 2,     5);
    run_case(32,  rows_2mib * 4,     5);

    run_case(64,  32 * 1024,         9);
    run_case(128, 8 * 1024,          9);
    run_case(16,  rows_2mib + rows_2mib / 2, 5);
}
