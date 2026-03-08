#![feature(allocator_api)]

//! Criterion benchmarks comparing jemalloc vs MAllocPg64 under concurrent workloads.
//!
//! jemalloc is set as the global allocator, so:
//! - Vec<u8> uses jemalloc's arena-per-thread allocation
//! - Vec<u8, Alloc64> uses jemalloc + 64-byte alignment wrapper
//! - Vec<u8, MAllocPg64> uses mmap/mremap, bypassing jemalloc entirely
//!
//! These benchmarks test the scenarios where jemalloc is expected to
//! have advantages: thread contention, concurrent allocation storms,
//! and mixed-size workloads across threads.
//!
//! Run with:
//!     cargo bench --features mmap --bench jemalloc_concurrent

use std::hint::black_box;
use std::thread;
use std::vec::Vec;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tikv_jemallocator::Jemalloc;
use vec64::alloc64::Alloc64;
use vec64::mmap_alloc::MAllocPg64;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const MB: usize = 1024 * 1024;
const NUM_THREADS: usize = 8;

/// Concurrent column build: N threads each grow a buffer from empty to target
/// via 4KB extend_from_slice. Measures total wall-clock time for all threads.
///
/// This simulates building multiple Arrow columns in parallel, where each
/// column is constructed by streaming data in small chunks.
fn bench_concurrent_column_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_column_build");
    let chunk = vec![0xCDu8; 4096];

    for &target_mb in &[4, 16, 64] {
        let target_bytes = target_mb * MB;

        group.bench_with_input(
            BenchmarkId::new("jemalloc_Vec", format!("{target_mb}MB_x{NUM_THREADS}")),
            &target_bytes,
            |b, &target| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut v: Vec<u8> = Vec::new();
                                while v.len() < target {
                                    v.extend_from_slice(&chunk);
                                }
                                black_box(&v);
                            });
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Alloc64_via_jemalloc", format!("{target_mb}MB_x{NUM_THREADS}")),
            &target_bytes,
            |b, &target| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut v: Vec<u8, Alloc64> = Vec::new_in(Alloc64::default());
                                while v.len() < target {
                                    v.extend_from_slice(&chunk);
                                }
                                black_box(&v);
                            });
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("MAllocPg64_mremap", format!("{target_mb}MB_x{NUM_THREADS}")),
            &target_bytes,
            |b, &target| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut v: Vec<u8, MAllocPg64> = Vec::new_in(MAllocPg64);
                                while v.len() < target {
                                    v.extend_from_slice(&chunk);
                                }
                                black_box(&v);
                            });
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Concurrent growth: N threads each pre-fill a buffer then double it.
/// Measures the growth/reallocation cost under thread contention.
///
/// This is where jemalloc's arena-per-thread design should reduce lock
/// contention on realloc, while MAllocPg64 issues concurrent mremap
/// syscalls to the kernel.
///
/// Each thread builds its own buffer in setup (timed), then grows it.
/// The benchmark measures the full allocate-fill-grow cycle concurrently.
fn bench_concurrent_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_growth");

    for &data_mb in &[4, 16, 64] {
        let data_bytes = data_mb * MB;
        let target_bytes = data_bytes * 2;

        group.bench_with_input(
            BenchmarkId::new("jemalloc_Vec", format!("{data_mb}MB_x{NUM_THREADS}")),
            &data_bytes,
            |b, &size| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut v: Vec<u8> = Vec::with_capacity(size);
                                v.resize(size, 0xAB);
                                v.reserve(target_bytes - v.len());
                                black_box(&v);
                            });
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Alloc64_via_jemalloc", format!("{data_mb}MB_x{NUM_THREADS}")),
            &data_bytes,
            |b, &size| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut v: Vec<u8, Alloc64> =
                                    Vec::with_capacity_in(size, Alloc64::default());
                                v.resize(size, 0xAB);
                                v.reserve(target_bytes - v.len());
                                black_box(&v);
                            });
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("MAllocPg64_mremap", format!("{data_mb}MB_x{NUM_THREADS}")),
            &data_bytes,
            |b, &size| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                let mut v: Vec<u8, MAllocPg64> =
                                    Vec::with_capacity_in(size, MAllocPg64);
                                v.resize(size, 0xAB);
                                v.reserve(target_bytes - v.len());
                                black_box(&v);
                            });
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Alloc/dealloc storm: N threads each do rapid alloc-fill-drop cycles.
/// This is jemalloc's home turf - arena-per-thread avoids cross-thread
/// lock contention on the allocator.
///
/// MAllocPg64 issues mmap/munmap syscalls per cycle, which are kernel
/// calls with higher per-call overhead than jemalloc's userspace pools.
fn bench_alloc_dealloc_storm(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_alloc_dealloc_storm");

    for &size in &[4096, 65536, MB] {
        let label = if size >= MB {
            format!("{}MB", size / MB)
        } else {
            format!("{}KB", size / 1024)
        };
        let iterations = 50;

        group.bench_with_input(
            BenchmarkId::new("jemalloc_Vec", format!("{label}_x{NUM_THREADS}")),
            &size,
            |b, &sz| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                for _ in 0..iterations {
                                    let mut v: Vec<u8> = Vec::with_capacity(sz);
                                    v.resize(sz, 0xAB);
                                    black_box(&v);
                                }
                            });
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Alloc64_via_jemalloc", format!("{label}_x{NUM_THREADS}")),
            &size,
            |b, &sz| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                for _ in 0..iterations {
                                    let mut v: Vec<u8, Alloc64> =
                                        Vec::new_in(Alloc64::default());
                                    v.reserve(sz);
                                    v.resize(sz, 0xAB);
                                    black_box(&v);
                                }
                            });
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("MAllocPg64_mremap", format!("{label}_x{NUM_THREADS}")),
            &size,
            |b, &sz| {
                b.iter(|| {
                    thread::scope(|s| {
                        for _ in 0..NUM_THREADS {
                            s.spawn(|| {
                                for _ in 0..iterations {
                                    let mut v: Vec<u8, MAllocPg64> =
                                        Vec::new_in(MAllocPg64);
                                    v.reserve(sz);
                                    v.resize(sz, 0xAB);
                                    black_box(&v);
                                }
                            });
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Mixed-size concurrent allocation: threads allocate buffers of varying
/// sizes in interleaved patterns, simulating a server processing different
/// column types and row group sizes concurrently.
///
/// Each thread cycles through small, medium, and large allocations,
/// keeping several live at once to create memory fragmentation pressure.
fn bench_mixed_size_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_mixed_size");
    let sizes = [256, 4096, 65536, MB, 4 * MB];
    let rounds = 20;

    group.bench_function(
        BenchmarkId::new("jemalloc_Vec", format!("x{NUM_THREADS}")),
        |b| {
            b.iter(|| {
                thread::scope(|s| {
                    for tid in 0..NUM_THREADS {
                        s.spawn(move || {
                            let mut live: Vec<Vec<u8>> = Vec::new();
                            for r in 0..rounds {
                                let size = sizes[(tid + r) % sizes.len()];
                                let mut v: Vec<u8> = Vec::with_capacity(size);
                                v.resize(size, 0xCD);
                                live.push(v);
                                if live.len() > 3 {
                                    black_box(live.remove(0));
                                }
                            }
                            black_box(&live);
                        });
                    }
                });
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("Alloc64_via_jemalloc", format!("x{NUM_THREADS}")),
        |b| {
            b.iter(|| {
                thread::scope(|s| {
                    for tid in 0..NUM_THREADS {
                        s.spawn(move || {
                            let mut live: Vec<Vec<u8, Alloc64>> = Vec::new();
                            for r in 0..rounds {
                                let size = sizes[(tid + r) % sizes.len()];
                                let mut v: Vec<u8, Alloc64> =
                                    Vec::with_capacity_in(size, Alloc64::default());
                                v.resize(size, 0xCD);
                                live.push(v);
                                if live.len() > 3 {
                                    black_box(live.remove(0));
                                }
                            }
                            black_box(&live);
                        });
                    }
                });
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("MAllocPg64_mremap", format!("x{NUM_THREADS}")),
        |b| {
            b.iter(|| {
                thread::scope(|s| {
                    for tid in 0..NUM_THREADS {
                        s.spawn(move || {
                            let mut live: Vec<Vec<u8, MAllocPg64>> = Vec::new();
                            for r in 0..rounds {
                                let size = sizes[(tid + r) % sizes.len()];
                                let mut v: Vec<u8, MAllocPg64> =
                                    Vec::with_capacity_in(size, MAllocPg64);
                                v.resize(size, 0xCD);
                                live.push(v);
                                if live.len() > 3 {
                                    black_box(live.remove(0));
                                }
                            }
                            black_box(&live);
                        });
                    }
                });
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_concurrent_column_build,
    bench_concurrent_growth,
    bench_alloc_dealloc_storm,
    bench_mixed_size_concurrent,
);
criterion_main!(benches);
