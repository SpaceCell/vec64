//! # **Global Allocator** - *Process-Wide 64-Byte Aligned Allocation*
//!
//! Provides two `#[global_allocator]`-compatible types:
//!
//! - [`Alloc64Global`] - delegates to `std::alloc::System` with 64-byte
//!   alignment. Works on all platforms.
//!
//! - [`MAllocPg64Global`] - delegates to [`MAllocPg64`](crate::MAllocPg64),
//!   using mmap for large allocations with mremap-based zero-copy growth.
//!   Linux only, requires the `mmap` feature.
//!
//! Both types delegate to `std::alloc::System` directly rather than
//! `std::alloc::alloc` to avoid infinite recursion through `__rust_alloc`
//! when installed as the global allocator.
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

/// Global allocator backed by [`MAllocPg64`](crate::MAllocPg64).
///
/// ## Usage
///
/// ```rust,ignore
/// use vec64::MAllocPg64Global;
///
/// #[global_allocator]
/// static GLOBAL: MAllocPg64Global = MAllocPg64Global;
/// ```
///
/// Uses mmap for allocations above the huge-page threshold with
/// mremap-based zero-copy growth. Smaller allocations fall back
/// to the system allocator with 64-byte alignment.
#[cfg(all(feature = "mmap", target_os = "linux"))]
#[derive(Copy, Clone, Default, Debug)]
pub struct MAllocPg64Global;

#[cfg(all(feature = "mmap", target_os = "linux"))]
unsafe impl GlobalAlloc for MAllocPg64Global {
    /// Small allocations delegate to `System` with 64-byte alignment.
    /// Large allocations at or above 2MB use mmap with huge page support.
    ///
    /// Delegates to `System` directly rather than `std::alloc::alloc` to
    /// avoid infinite recursion through `__rust_alloc` when installed as
    /// the global allocator.
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        use crate::mmap_alloc::{do_mmap, hint_thp, mapped_size, uses_mmap};
        let size = layout.size();
        if !uses_mmap(size) {
            return unsafe { System.alloc(align_layout(layout)) };
        }
        let mapped = mapped_size(size);
        match do_mmap(size, mapped) {
            Ok(ptr) => {
                hint_thp(ptr, size, mapped);
                ptr.as_ptr()
            }
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        use crate::mmap_alloc::{mapped_size, uses_mmap};
        let size = layout.size();
        if !uses_mmap(size) {
            return unsafe { System.dealloc(ptr, align_layout(layout)) };
        }
        let mapped = mapped_size(size);
        unsafe { libc::munmap(ptr as *mut libc::c_void, mapped) };
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        use crate::mmap_alloc::{do_mmap, hint_thp, mapped_size, uses_mmap};
        let size = layout.size();
        if !uses_mmap(size) {
            return unsafe { System.alloc_zeroed(align_layout(layout)) };
        }
        // mmap anonymous pages are zero-filled by the kernel
        let mapped = mapped_size(size);
        match do_mmap(size, mapped) {
            Ok(ptr) => {
                hint_thp(ptr, size, mapped);
                ptr.as_ptr()
            }
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        use crate::mmap_alloc::{
            do_mmap, hint_thp, mapped_size, uses_giant_pages, uses_mmap,
        };
        let old_size = layout.size();
        let old_mmap = uses_mmap(old_size);
        let new_mmap = uses_mmap(new_size);

        // Both below threshold, delegate to System
        if !old_mmap && !new_mmap {
            return unsafe { System.realloc(ptr, align_layout(layout), new_size) };
        }

        // Transitioning from heap to mmap, allocate mmap then copy and free heap
        if !old_mmap && new_mmap {
            let new_mapped = mapped_size(new_size);
            let Ok(new_ptr) = do_mmap(new_size, new_mapped) else {
                return std::ptr::null_mut();
            };
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, new_ptr.as_ptr(), old_size);
                System.dealloc(ptr, align_layout(layout));
            }
            hint_thp(new_ptr, new_size, new_mapped);
            return new_ptr.as_ptr();
        }

        // Transitioning from mmap to heap, allocate heap then copy and munmap
        if old_mmap && !new_mmap {
            let new_layout = align_layout(
                unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) }
            );
            let new_ptr = unsafe { System.alloc(new_layout) };
            if new_ptr.is_null() {
                return std::ptr::null_mut();
            }
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, new_ptr, new_size);
                let old_mapped = mapped_size(old_size);
                libc::munmap(ptr as *mut libc::c_void, old_mapped);
            }
            return new_ptr;
        }

        // Both in mmap territory, use mremap for zero-copy growth
        let old_mapped = mapped_size(old_size);
        let new_mapped = mapped_size(new_size);

        // Same mapped size means no work needed
        if old_mapped == new_mapped {
            return ptr;
        }

        // Shrinking across gigantic to non-gigantic page boundary cannot
        // partially unmap 1GB pages. Fall back to fresh allocation with copy.
        if new_size < old_size && uses_giant_pages(old_size) && !uses_giant_pages(new_size) {
            let Ok(new_ptr) = do_mmap(new_size, new_mapped) else {
                return std::ptr::null_mut();
            };
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, new_ptr.as_ptr(), new_size);
                libc::munmap(ptr as *mut libc::c_void, old_mapped);
            }
            hint_thp(new_ptr, new_size, new_mapped);
            return new_ptr.as_ptr();
        }

        // Shrinking within mmap, release tail pages
        if new_size < old_size {
            unsafe {
                libc::munmap(
                    (ptr as usize + new_mapped) as *mut libc::c_void,
                    old_mapped - new_mapped,
                );
            }
            return ptr;
        }

        // Growing within mmap via mremap
        let raw = unsafe {
            libc::mremap(
                ptr as *mut libc::c_void,
                old_mapped,
                new_mapped,
                libc::MREMAP_MAYMOVE,
            )
        };
        if raw == libc::MAP_FAILED {
            return std::ptr::null_mut();
        }
        let nn = match std::ptr::NonNull::new(raw as *mut u8) {
            Some(nn) => nn,
            None => return std::ptr::null_mut(),
        };
        hint_thp(nn, new_size, new_mapped);
        nn.as_ptr()
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
