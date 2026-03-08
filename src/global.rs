//! # **Global Allocator** - *Process-Wide 64-Byte Aligned Allocation*
//!
//! Provides `Alloc64Global`, a `#[global_allocator]`-compatible type
//! that enforces 64-byte alignment on every allocation in the process.
//!
//! Delegates to `std::alloc::System` directly rather than `std::alloc::alloc`
//! to avoid infinite recursion through `__rust_alloc` when installed as the
//! global allocator.
use std::alloc::{GlobalAlloc, Layout, System};

use crate::alloc64::align_layout;

/// # Alloc64Global
///
/// Zero-sized global allocator that enforces 64-byte alignment on all allocations.
///
/// ## Usage
///
/// ```rust,ignore
/// use vec64::Alloc64Global;
///
/// #[global_allocator]
/// static GLOBAL: Alloc64Global = Alloc64Global;
/// ```
///
/// Once installed, every allocation in the process including those from
/// third-party crates uses 64-byte alignment. This is useful when
/// zero-copy interop with SIMD or Arrow-aligned buffers is required
/// across the entire application, for example when network transport
/// libraries allocate receive buffers that are later consumed directly
/// as Arrow data-formatted columns.
///
/// ## Trade-offs
///
/// Every allocation is rounded up to a 64-byte boundary, so small
/// allocations consume more virtual memory than with the default
/// allocator. This is negligible in practice when working with large
/// buffers. Columnar libraries like Minarrow handle this inherently
/// through design.
///
/// In embedded environments, or workloads that allocate many small
/// independent buffers, consider whether the overhead is acceptable.
/// The primary use case is zero-copy alignment for scenarios where
/// third-party libraries e.g. network transports would otherwise
/// allocate unaligned receive buffers.
#[derive(Copy, Clone, Default, Debug)]
pub struct Alloc64Global;

unsafe impl GlobalAlloc for Alloc64Global {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { System.alloc(align_layout(layout)) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, align_layout(layout)) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { System.alloc_zeroed(align_layout(layout)) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe { System.realloc(ptr, align_layout(layout), new_size) }
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout};

    use super::*;

    #[test]
    fn test_global_alloc_alignment() {
        let g = Alloc64Global;
        for size in [1, 7, 32, 64, 256, 4096] {
            let layout = Layout::from_size_align(size, 1).unwrap();
            let ptr = unsafe { g.alloc(layout) };
            assert!(!ptr.is_null(), "alloc returned null for size {}", size);
            assert_eq!(
                ptr as usize % 64,
                0,
                "Pointer {:#x} not 64-byte aligned for size {}",
                ptr as usize,
                size
            );
            unsafe { g.dealloc(ptr, layout) };
        }
    }

    #[test]
    fn test_global_alloc_zeroed() {
        let g = Alloc64Global;
        let layout = Layout::from_size_align(128, 1).unwrap();
        let ptr = unsafe { g.alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 64, 0);
        let data = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
        assert!(data.iter().all(|&b| b == 0));
        unsafe { g.dealloc(ptr, layout) };
    }

    #[test]
    fn test_global_realloc_alignment() {
        let g = Alloc64Global;
        let layout = Layout::from_size_align(64, 1).unwrap();
        let ptr = unsafe { g.alloc(layout) };
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 64, 0);

        // Write a marker byte to verify data is preserved
        unsafe { *ptr = 0xAB };

        let new_ptr = unsafe { g.realloc(ptr, layout, 256) };
        assert!(!new_ptr.is_null());
        assert_eq!(new_ptr as usize % 64, 0);
        assert_eq!(unsafe { *new_ptr }, 0xAB, "data not preserved after realloc");

        let shrunk = unsafe {
            g.realloc(
                new_ptr,
                Layout::from_size_align(256, 1).unwrap(),
                32,
            )
        };
        assert!(!shrunk.is_null());
        assert_eq!(shrunk as usize % 64, 0);
        assert_eq!(unsafe { *shrunk }, 0xAB, "data not preserved after shrink");

        unsafe { g.dealloc(shrunk, Layout::from_size_align(32, 1).unwrap()) };
    }

    #[test]
    fn test_global_alignment_override() {
        // Even with a lower alignment request, pointer should be 64-byte aligned
        let g = Alloc64Global;
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let ptr = unsafe { g.alloc(layout) };
        assert!(!ptr.is_null());
        assert_eq!(
            ptr as usize % 64,
            0,
            "Even with align=8 request, pointer should be 64-byte aligned"
        );
        unsafe { g.dealloc(ptr, layout) };
    }
}
