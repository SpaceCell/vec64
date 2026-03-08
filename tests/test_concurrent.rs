#![cfg_attr(feature = "mmap", feature(allocator_api))]

//! Concurrent and multi-threaded tests for Vec64 and its allocators.
//!
//! Simulates Arrow buffer-style workloads: parallel column construction,
//! contended allocation/deallocation, concurrent buffer growth, and
//! parallel in-place transforms.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use vec64::Vec64;

const NUM_THREADS: usize = 8;

/// Verify 64-byte alignment of a Vec64's backing pointer.
fn assert_aligned<T>(v: &Vec64<T>, context: &str) {
    if v.capacity() > 0 {
        let addr = v.as_ptr() as usize;
        assert_eq!(addr % 64, 0, "Alignment violated: {context}");
    }
}

// ---------------------------------------------------------------------------
// Parallel column construction - each thread builds its own column buffer
// ---------------------------------------------------------------------------

/// Simulates building N columns in parallel, like constructing a RecordBatch
/// with one thread per column. Each thread allocates a Vec64, fills it with
/// a deterministic pattern, and returns it for verification.
#[test]
fn concurrent_column_build() {
    let rows = 100_000;

    let columns: Vec<Vec64<f64>> = (0..NUM_THREADS)
        .map(|col_id| {
            thread::scope(|_| {
                let mut column = Vec64::with_capacity(rows);
                for row in 0..rows {
                    column.push((col_id * rows + row) as f64);
                }
                column
            })
        })
        .collect();

    for (col_id, column) in columns.iter().enumerate() {
        assert_eq!(column.len(), rows);
        assert_aligned(column, &format!("column {col_id}"));
        for row in 0..rows {
            let expected = (col_id * rows + row) as f64;
            assert_eq!(column[row], expected, "column {col_id} row {row}");
        }
    }
}

/// Same as above but all threads run truly concurrently via thread::scope.
#[test]
fn concurrent_column_build_scoped() {
    let rows = 100_000;

    let columns: Vec<Vec64<f64>> = thread::scope(|s| {
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|col_id| {
                s.spawn(move || {
                    let mut column = Vec64::with_capacity(rows);
                    for row in 0..rows {
                        column.push((col_id * rows + row) as f64);
                    }
                    column
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });

    for (col_id, column) in columns.iter().enumerate() {
        assert_eq!(column.len(), rows);
        assert_aligned(column, &format!("scoped column {col_id}"));
        // Spot check first and last
        assert_eq!(column[0], (col_id * rows) as f64);
        assert_eq!(column[rows - 1], (col_id * rows + rows - 1) as f64);
    }
}

// ---------------------------------------------------------------------------
// Contended allocation/deallocation - threads hammer the allocator
// ---------------------------------------------------------------------------

/// Multiple threads in tight loops allocating, filling, and dropping Vec64
/// buffers. Exercises allocator thread safety and checks that alignment
/// holds under contention. Each thread does many small alloc/fill/drop
/// cycles to stress-test the allocation path.
#[test]
fn contended_alloc_dealloc() {
    let iterations = 500;
    let sizes = [64, 256, 4096, 16384, 65536];

    thread::scope(|s| {
        for thread_id in 0..NUM_THREADS {
            s.spawn(move || {
                for i in 0..iterations {
                    let size = sizes[i % sizes.len()];
                    let mut v: Vec64<u8> = Vec64::with_capacity(size);
                    assert_aligned(&v, &format!("thread {thread_id} iter {i} size {size}"));

                    // Fill with thread-specific pattern
                    let pattern = (thread_id as u8).wrapping_add(i as u8);
                    v.resize(size, pattern);

                    // Verify no corruption from other threads
                    assert!(
                        v.iter().all(|&b| b == pattern),
                        "Data corruption in thread {thread_id} iter {i}"
                    );
                    // v drops here, exercising deallocate under contention
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Concurrent buffer growth - threads grow their buffers simultaneously
// ---------------------------------------------------------------------------

/// Each thread starts with a small Vec64 and grows it incrementally via
/// extend_from_slice, forcing many reallocations. Verifies alignment is
/// maintained through every growth step and data integrity is preserved.
#[test]
fn concurrent_growth_alignment() {
    let chunk_size = 4096;
    let num_chunks = 256; // 1MB total per thread

    thread::scope(|s| {
        for thread_id in 0..NUM_THREADS {
            s.spawn(move || {
                let pattern = (thread_id as u8).wrapping_mul(31);
                let chunk = vec![pattern; chunk_size];
                let mut v: Vec64<u8> = Vec64::new();

                for chunk_idx in 0..num_chunks {
                    v.extend_from_slice(&chunk);
                    assert_aligned(
                        &v,
                        &format!("thread {thread_id} after chunk {chunk_idx}"),
                    );
                }

                // Verify all data is correct after all growth
                assert_eq!(v.len(), chunk_size * num_chunks);
                assert!(
                    v.iter().all(|&b| b == pattern),
                    "Data corruption after growth in thread {thread_id}"
                );
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Parallel segment writes via shared buffer
// ---------------------------------------------------------------------------

/// Pre-allocates a large Vec64, then multiple threads write to
/// non-overlapping segments concurrently. This mirrors how Arrow
/// column encoders might parallelise encoding across row groups.
///
/// Uses raw pointer arithmetic for concurrent segment writes, which
/// is safe because segments are non-overlapping and all writes
/// complete before reads.
#[test]
fn parallel_segment_write() {
    let total_elements = NUM_THREADS * 100_000;
    let segment_size = total_elements / NUM_THREADS;

    // Pre-allocate and zero-fill
    let mut buffer: Vec64<u64> = Vec64::with_capacity(total_elements);
    buffer.resize(total_elements, 0);
    assert_aligned(&buffer, "pre-allocated buffer");

    // Use usize to carry the pointer across thread boundaries.
    // SAFETY: each thread writes to a non-overlapping segment,
    // and all threads join before we read.
    let base_addr = buffer.as_mut_ptr() as usize;

    thread::scope(|s| {
        for thread_id in 0..NUM_THREADS {
            let start = thread_id * segment_size;
            s.spawn(move || {
                let base = base_addr as *mut u64;
                for i in 0..segment_size {
                    unsafe {
                        *base.add(start + i) = (thread_id * segment_size + i) as u64;
                    }
                }
            });
        }
    });

    // Verify all segments
    for thread_id in 0..NUM_THREADS {
        let start = thread_id * segment_size;
        for i in 0..segment_size {
            let expected = (thread_id * segment_size + i) as u64;
            assert_eq!(
                buffer[start + i], expected,
                "segment {thread_id} offset {i}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Shared atomic counter with per-thread Vec64 accumulation
// ---------------------------------------------------------------------------

/// Multiple threads each build a Vec64<u64> of partial results, then
/// accumulate into a shared atomic. Tests that Vec64 works correctly
/// alongside standard concurrency primitives.
#[test]
fn concurrent_accumulate() {
    let elements_per_thread = 50_000;
    let total = Arc::new(AtomicU64::new(0));

    thread::scope(|s| {
        for _ in 0..NUM_THREADS {
            let total = Arc::clone(&total);
            s.spawn(move || {
                let mut buffer: Vec64<u64> = Vec64::with_capacity(elements_per_thread);
                for i in 0..elements_per_thread {
                    buffer.push(i as u64);
                }
                let local_sum: u64 = buffer.iter().sum();
                total.fetch_add(local_sum, Ordering::Relaxed);
            });
        }
    });

    // Each thread sums 0..49999 = 49999 * 50000 / 2 = 1_249_975_000
    let expected = (elements_per_thread as u64 - 1) * elements_per_thread as u64 / 2
        * NUM_THREADS as u64;
    assert_eq!(total.load(Ordering::Relaxed), expected);
}

// ---------------------------------------------------------------------------
// Mixed-size allocation storm
// ---------------------------------------------------------------------------

/// Threads allocate buffers of wildly varying sizes, from tiny to large,
/// interleaved. Exercises the allocator's ability to handle concurrent
/// mixed-size requests without corruption.
#[test]
fn mixed_size_allocation_storm() {
    let sizes: Vec<usize> = (0..12).map(|i| 1 << i).collect(); // 1 to 4096 bytes

    thread::scope(|s| {
        for thread_id in 0..NUM_THREADS {
            let sizes = &sizes;
            s.spawn(move || {
                let mut live_buffers: Vec<Vec64<u8>> = Vec::new();

                for round in 0..100 {
                    let size = sizes[(thread_id + round) % sizes.len()];
                    let mut v: Vec64<u8> = Vec64::with_capacity(size);
                    let pattern = ((thread_id * 100 + round) % 256) as u8;
                    v.resize(size, pattern);
                    assert_aligned(&v, &format!("thread {thread_id} round {round}"));

                    live_buffers.push(v);

                    // Periodically drop old buffers to create fragmentation
                    if live_buffers.len() > 5 {
                        let removed = live_buffers.remove(0);
                        // Verify data was still intact before drop
                        let first = removed[0];
                        assert!(
                            removed.iter().all(|&b| b == first),
                            "Corruption in thread {thread_id} round {round}"
                        );
                    }
                }

                // Verify remaining buffers
                for buf in &live_buffers {
                    let first = buf[0];
                    assert!(buf.iter().all(|&b| b == first));
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Concurrent clone and drop
// ---------------------------------------------------------------------------

/// Multiple threads clone and drop a shared source buffer concurrently.
/// Exercises the allocator's handling of many simultaneous allocations
/// from cloned data, and many simultaneous deallocations from drops.
#[test]
fn concurrent_clone_and_drop() {
    let source: Vec64<u64> = (0..10_000).collect();
    let source = Arc::new(source);

    thread::scope(|s| {
        for _ in 0..NUM_THREADS {
            let source = Arc::clone(&source);
            s.spawn(move || {
                for _ in 0..50 {
                    let clone = (*source).clone();
                    assert_aligned(&clone, "cloned buffer");
                    assert_eq!(clone.len(), 10_000);

                    // Verify data integrity
                    for (i, &val) in clone.iter().enumerate() {
                        assert_eq!(val, i as u64);
                    }
                    // clone drops here
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Parallel column construction with type conversion
// ---------------------------------------------------------------------------

/// Simulates building typed columns: one thread builds i32 values,
/// another builds f64, another builds u8 bitmasks - all concurrently.
/// Verifies type safety and alignment across different element sizes.
#[test]
fn concurrent_heterogeneous_columns() {
    let rows = 50_000;

    thread::scope(|s| {
        let h_i32 = s.spawn(move || {
            let mut col: Vec64<i32> = Vec64::with_capacity(rows);
            for i in 0..rows {
                col.push(i as i32);
            }
            assert_aligned(&col, "i32 column");
            col
        });

        let h_f64 = s.spawn(move || {
            let mut col: Vec64<f64> = Vec64::with_capacity(rows);
            for i in 0..rows {
                col.push(i as f64 * 1.5);
            }
            assert_aligned(&col, "f64 column");
            col
        });

        let h_u8 = s.spawn(move || {
            let mut col: Vec64<u8> = Vec64::with_capacity(rows);
            for i in 0..rows {
                col.push((i % 256) as u8);
            }
            assert_aligned(&col, "u8 column");
            col
        });

        let h_u64 = s.spawn(move || {
            let mut col: Vec64<u64> = Vec64::with_capacity(rows);
            for i in 0..rows {
                col.push(i as u64 * 7);
            }
            assert_aligned(&col, "u64 column");
            col
        });

        let col_i32 = h_i32.join().unwrap();
        let col_f64 = h_f64.join().unwrap();
        let col_u8 = h_u8.join().unwrap();
        let col_u64 = h_u64.join().unwrap();

        assert_eq!(col_i32.len(), rows);
        assert_eq!(col_f64.len(), rows);
        assert_eq!(col_u8.len(), rows);
        assert_eq!(col_u64.len(), rows);

        // Spot checks
        assert_eq!(col_i32[rows - 1], (rows - 1) as i32);
        assert_eq!(col_f64[rows - 1], (rows - 1) as f64 * 1.5);
        assert_eq!(col_u8[rows - 1], ((rows - 1) % 256) as u8);
        assert_eq!(col_u64[rows - 1], (rows - 1) as u64 * 7);
    });
}

// ---------------------------------------------------------------------------
// Parallel extend from multiple sources
// ---------------------------------------------------------------------------

/// Each thread produces a chunk of data, then all chunks are sequentially
/// extended into a single Vec64. Tests that extend_from_slice works
/// correctly after concurrent production.
#[test]
fn concurrent_produce_sequential_extend() {
    let chunk_size = 50_000;

    let chunks: Vec<Vec64<u32>> = thread::scope(|s| {
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                s.spawn(move || {
                    let mut chunk = Vec64::with_capacity(chunk_size);
                    for i in 0..chunk_size {
                        chunk.push((thread_id * chunk_size + i) as u32);
                    }
                    chunk
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });

    // Sequentially merge into one buffer
    let mut combined: Vec64<u32> = Vec64::with_capacity(chunk_size * NUM_THREADS);
    for chunk in &chunks {
        combined.extend_from_slice(chunk);
    }

    assert_eq!(combined.len(), chunk_size * NUM_THREADS);
    assert_aligned(&combined, "combined buffer");

    // Verify ordering
    for thread_id in 0..NUM_THREADS {
        let offset = thread_id * chunk_size;
        for i in 0..chunk_size {
            assert_eq!(
                combined[offset + i],
                (thread_id * chunk_size + i) as u32
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel iterator tests (requires parallel_proc)
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel_proc")]
mod parallel_iter_tests {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    use super::*;

    /// Parallel in-place transform: multiply every element by 2.
    /// Simulates a parallel column transformation (e.g. unit conversion).
    #[test]
    fn par_iter_mut_transform() {
        let n = 500_000;
        let mut buffer: Vec64<i64> = (0..n as i64).collect();
        assert_aligned(&buffer, "before transform");

        buffer.par_iter_mut().for_each(|x| *x *= 2);

        assert_aligned(&buffer, "after transform");
        for i in 0..n {
            assert_eq!(buffer[i], i as i64 * 2);
        }
    }

    /// Parallel filter: count elements matching a predicate.
    /// Simulates computing filter masks on a column.
    #[test]
    fn par_iter_filter_count() {
        let n = 1_000_000;
        let buffer: Vec64<u64> = (0..n as u64).collect();

        let count = buffer.par_iter().filter(|&&x| x % 3 == 0).count();

        // 0, 3, 6, ..., 999999: count = ceil(1_000_000 / 3) = 333_334
        let expected = (n + 2) / 3;
        assert_eq!(count, expected);
    }

    /// Parallel reduction: sum across a large buffer.
    /// Simulates aggregation queries on a column.
    #[test]
    fn par_iter_reduction() {
        let n = 1_000_000u64;
        let buffer: Vec64<u64> = (0..n).collect();

        let sum: u64 = buffer.par_iter().sum();
        let expected = n * (n - 1) / 2;
        assert_eq!(sum, expected);
    }

    /// Parallel zip and compute: element-wise addition of two columns.
    /// Simulates vectorised arithmetic across columns.
    #[test]
    fn par_iter_zip_columns() {
        let n = 500_000;
        let col_a: Vec64<f64> = (0..n).map(|i| i as f64).collect();
        let col_b: Vec64<f64> = (0..n).map(|i| i as f64 * 0.5).collect();

        // Compute in parallel, collect into std Vec, then convert
        let result_vec: std::vec::Vec<f64> = col_a
            .par_iter()
            .zip(col_b.par_iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let result: Vec64<f64> = Vec64::from(result_vec);

        assert_aligned(&result, "zip result");
        assert_eq!(result.len(), n);
        for i in 0..n {
            let expected = i as f64 + i as f64 * 0.5;
            assert_eq!(result[i], expected, "zip mismatch at {i}");
        }
    }
}

// ---------------------------------------------------------------------------
//  MAllocPg64 concurrent tests (requires mmap)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "mmap", target_os = "linux"))]
mod mmap_concurrent_tests {
    use std::vec::Vec;

    use super::*;
    use vec64::mmap_alloc::MAllocPg64;

    /// Multiple threads allocate and grow MAllocPg64-backed vectors
    /// concurrently. mmap/mremap syscalls are process-global, so this
    /// verifies the kernel handles concurrent mapping operations correctly
    /// with our allocator.
    #[test]
    fn concurrent_mmap_column_build() {
        let rows = 100_000;

        let columns: std::vec::Vec<Vec<u64, MAllocPg64>> = thread::scope(|s| {
            let handles: std::vec::Vec<_> = (0..NUM_THREADS)
                .map(|col_id| {
                    s.spawn(move || {
                        let mut column: Vec<u64, MAllocPg64> =
                            Vec::with_capacity_in(rows, MAllocPg64);
                        for row in 0..rows {
                            column.push((col_id * rows + row) as u64);
                        }
                        column
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });

        for (col_id, column) in columns.iter().enumerate() {
            assert_eq!(column.len(), rows);
            let addr = column.as_ptr() as usize;
            assert_eq!(addr % 64, 0, "MAllocPg64 column {col_id} not aligned");

            // Spot check
            assert_eq!(column[0], (col_id * rows) as u64);
            assert_eq!(column[rows - 1], (col_id * rows + rows - 1) as u64);
        }
    }

    /// Concurrent mmap alloc/dealloc storm. Exercises munmap concurrency -
    /// each thread rapidly allocates and drops MAllocPg64 vectors, creating
    /// many concurrent mmap/munmap syscalls.
    #[test]
    fn concurrent_mmap_alloc_storm() {
        let iterations = 100;

        thread::scope(|s| {
            for thread_id in 0..NUM_THREADS {
                s.spawn(move || {
                    for i in 0..iterations {
                        // Each allocation is at least 2MB due to huge page rounding
                        let size = 4096 * (1 + (i % 10));
                        let mut v: Vec<u8, MAllocPg64> =
                            Vec::with_capacity_in(size, MAllocPg64);

                        let pattern = (thread_id as u8).wrapping_add(i as u8);
                        v.resize(size, pattern);

                        let addr = v.as_ptr() as usize;
                        assert_eq!(
                            addr % 64, 0,
                            "Unaligned mmap alloc: thread {thread_id} iter {i}"
                        );
                        assert!(
                            v.iter().all(|&b| b == pattern),
                            "Corruption: thread {thread_id} iter {i}"
                        );
                    }
                });
            }
        });
    }

    /// Concurrent mremap growth. Threads grow their MAllocPg64 buffers
    /// past the 2MB huge page boundary, forcing mremap syscalls. Verifies
    /// data preservation across concurrent mremap operations.
    #[test]
    fn concurrent_mremap_growth() {
        let chunk_size = 65536; // 64KB chunks
        let target_size = 3 * 1024 * 1024; // 3MB, forces at least one mremap
        let num_chunks = target_size / chunk_size;

        thread::scope(|s| {
            for thread_id in 0..NUM_THREADS {
                s.spawn(move || {
                    let pattern = (thread_id as u8).wrapping_mul(37);
                    let chunk = vec![pattern; chunk_size];
                    let mut v: Vec<u8, MAllocPg64> = Vec::new_in(MAllocPg64);

                    for _ in 0..num_chunks {
                        v.extend_from_slice(&chunk);
                    }

                    assert_eq!(v.len(), target_size);
                    let addr = v.as_ptr() as usize;
                    assert_eq!(
                        addr % 64, 0,
                        "Unaligned after mremap growth: thread {thread_id}"
                    );
                    assert!(
                        v.iter().all(|&b| b == pattern),
                        "Data corruption after mremap: thread {thread_id}"
                    );
                });
            }
        });
    }
}
