#![feature(allocator_api)]

//! Criterion benchmarks comparing jemalloc against Alloc64 and MAllocPg64.
//!
//! jemalloc is set as the global allocator, so:
//! - Vec<u8> uses jemalloc realloc directly
//! - Vec<u8, Alloc64> uses jemalloc + 64-byte alignment wrapper
//! - Vec<u8, MAllocPg64> uses mmap/mremap, bypassing jemalloc entirely
//!
//! Run with:
//!     cargo bench --features mmap --bench jemalloc_comparison

use std::hint::black_box;
use std::vec::Vec;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tikv_jemallocator::Jemalloc;
use vec64::alloc64::Alloc64;
use vec64::mmap_alloc::MAllocPg64;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const MB: usize = 1024 * 1024;

/// Single growth: pre-fill then double the buffer.
fn bench_single_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("jemalloc_single_growth");

    for &data_mb in &[4, 16, 64] {
        let data_bytes = data_mb * MB;
        let target_bytes = data_bytes * 2;

        group.bench_with_input(
            BenchmarkId::new("jemalloc_Vec", format!("{data_mb}MB")),
            &data_bytes,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut v: Vec<u8> = Vec::with_capacity(size);
                        v.resize(size, 0xAB);
                        v
                    },
                    |mut v| {
                        v.reserve(target_bytes - v.len());
                        black_box(&v);
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Alloc64_via_jemalloc", format!("{data_mb}MB")),
            &data_bytes,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut v: Vec<u8, Alloc64> =
                            Vec::with_capacity_in(size, Alloc64::default());
                        v.resize(size, 0xAB);
                        v
                    },
                    |mut v| {
                        v.reserve(target_bytes - v.len());
                        black_box(&v);
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("MAllocPg64_mremap", format!("{data_mb}MB")),
            &data_bytes,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut v: Vec<u8, MAllocPg64> =
                            Vec::with_capacity_in(size, MAllocPg64);
                        v.resize(size, 0xAB);
                        v
                    },
                    |mut v| {
                        v.reserve(target_bytes - v.len());
                        black_box(&v);
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

/// Incremental build: extend in 4KB chunks from empty to target.
fn bench_incremental_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("jemalloc_incremental_build");
    let chunk = vec![0xCDu8; 4096];

    for &target_mb in &[4, 16, 64] {
        let target_bytes = target_mb * MB;

        group.bench_with_input(
            BenchmarkId::new("jemalloc_Vec", format!("{target_mb}MB")),
            &target_bytes,
            |b, &target| {
                b.iter(|| {
                    let mut v: Vec<u8> = Vec::new();
                    while v.len() < target {
                        v.extend_from_slice(&chunk);
                    }
                    black_box(&v);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Alloc64_via_jemalloc", format!("{target_mb}MB")),
            &target_bytes,
            |b, &target| {
                b.iter(|| {
                    let mut v: Vec<u8, Alloc64> = Vec::new_in(Alloc64::default());
                    while v.len() < target {
                        v.extend_from_slice(&chunk);
                    }
                    black_box(&v);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("MAllocPg64_mremap", format!("{target_mb}MB")),
            &target_bytes,
            |b, &target| {
                b.iter(|| {
                    let mut v: Vec<u8, MAllocPg64> = Vec::new_in(MAllocPg64);
                    while v.len() < target {
                        v.extend_from_slice(&chunk);
                    }
                    black_box(&v);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_single_growth, bench_incremental_build);
criterion_main!(benches);
