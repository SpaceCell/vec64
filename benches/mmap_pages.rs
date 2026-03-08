#![feature(allocator_api)]

//! Criterion benchmarks for mmap page-level behaviour: allocation
//! latency, sequential scan throughput, and huge page verification.
//!
//! Run with:
//!     cargo bench --features mmap --bench mmap_pages

use std::alloc::{Allocator, Layout};
use std::hint::black_box;
use std::vec::Vec;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vec64::mmap_alloc::MAllocPg64;

const MB: usize = 1024 * 1024;
const HUGE_PAGE: usize = 2 * MB;

/// Measure raw mmap allocation + munmap latency at various sizes.
/// Shows overhead of the mmap syscall path vs heap.
fn bench_alloc_dealloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmap_alloc_dealloc");

    for &size_mb in &[2, 8, 32, 128] {
        let size = size_mb * MB;
        let layout = Layout::from_size_align(size, 64).unwrap();

        group.bench_with_input(
            BenchmarkId::new("MAllocPg64", format!("{size_mb}MB")),
            &layout,
            |b, layout| {
                let a = MAllocPg64;
                b.iter(|| {
                    let ptr = a.allocate(*layout).expect("allocate failed");
                    black_box(ptr.cast::<u8>());
                    unsafe { a.deallocate(ptr.cast::<u8>(), *layout) };
                });
            },
        );
    }

    group.finish();
}

/// Sequential scan over mmap'd memory. Exercises the TLB - with huge
/// pages backing the allocation, this should show fewer TLB misses
/// and higher throughput than 4KB-paged memory.
fn bench_sequential_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_scan");
    group.throughput(criterion::Throughput::Bytes((64 * MB) as u64));

    // MAllocPg64 path - 2MB page-rounded allocation with THP hints
    group.bench_function("MAllocPg64_64MB", |b| {
        let mut v: Vec<u64, MAllocPg64> = Vec::with_capacity_in(8 * MB, MAllocPg64);
        v.resize(8 * MB, 1u64);
        b.iter(|| {
            let sum: u64 = v.iter().sum();
            black_box(sum);
        });
    });

    // Standard heap path for comparison
    group.bench_function("heap_64MB", |b| {
        let mut v: Vec<u64> = Vec::with_capacity(8 * MB);
        v.resize(8 * MB, 1u64);
        b.iter(|| {
            let sum: u64 = v.iter().sum();
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark mremap growth across huge page boundaries.
/// Each iteration grows from one huge page to the target,
/// measuring the mremap syscall path.
fn bench_mremap_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("mremap_growth");

    for &target_pages in &[2, 4, 16, 64] {
        let target_size = target_pages * HUGE_PAGE;

        group.bench_with_input(
            BenchmarkId::new("grow", format!("{target_pages}_pages")),
            &target_size,
            |b, &target| {
                b.iter_batched(
                    || {
                        let mut v: Vec<u8, MAllocPg64> =
                            Vec::with_capacity_in(HUGE_PAGE, MAllocPg64);
                        v.resize(HUGE_PAGE, 0xAA);
                        v
                    },
                    |mut v| {
                        v.reserve(target - v.len());
                        black_box(&v);
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

/// Verify mmap allocations are 64-byte aligned and return
/// the full mapped size. This is a correctness check that
/// runs as a benchmark to confirm no regression under load.
fn bench_alignment_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment_check");

    for &size in &[1, 64, 4096, HUGE_PAGE, HUGE_PAGE + 1] {
        let layout = Layout::from_size_align(size, 1).unwrap();

        group.bench_with_input(
            BenchmarkId::new("alloc_check", size),
            &layout,
            |b, layout| {
                let a = MAllocPg64;
                b.iter(|| {
                    let ptr = a.allocate(*layout).expect("allocate failed");
                    let addr = ptr.cast::<u8>().as_ptr() as usize;
                    assert_eq!(addr % 64, 0, "not 64-byte aligned");
                    assert!(ptr.len() >= HUGE_PAGE, "returned size below huge page");
                    black_box(addr);
                    unsafe { a.deallocate(ptr.cast::<u8>(), *layout) };
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_alloc_dealloc,
    bench_sequential_scan,
    bench_mremap_growth,
    bench_alignment_check,
);
criterion_main!(benches);
