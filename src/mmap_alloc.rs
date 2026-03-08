//! # MAllocPg64 - mmap-based 64-byte Aligned Allocator
//!
//! Linux-specific allocator using `mmap` for page-aligned allocations,
//! `mremap` for zero-copy growth, and transparent huge pages via 2MB
//! page rounding with `madvise(MADV_HUGEPAGE)`.
//!
//! ## Allocation strategy
//!
//! All allocations are rounded up to 2MB huge page boundaries,
//! ensuring both page alignment and 64-byte SIMD alignment.
//! Growth uses `mremap(MREMAP_MAYMOVE)` for zero-copy virtual
//! address remapping - the kernel adjusts page tables without
//! copying physical memory.
//!
//!
//! With the `giant_pages` feature, initial allocations >= 1GB
//! use `MAP_HUGETLB | MAP_HUGE_1GB` for 1GB huge pages. Gigantic
//! pages are only used for the initial allocation; growing into
//! 1GB territory from a smaller mapping stays on 2MB THP.
//!
//! ## Trade-offs
//!
//! Every allocation consumes at least 2MB of virtual address space.
//! This is by design for columnar and SIMD workloads with large
//! buffers. Not suitable for many small independent allocations.
//!
//! ## Platform
//!
//! Linux only. Requires mmap, mremap, munmap, madvise.
use core::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use std::ptr::slice_from_raw_parts_mut;

const HUGE_PAGE: usize = 2 * 1024 * 1024;

#[cfg(feature = "giant_pages")]
const GIGANTIC_PAGE: usize = 1024 * 1024 * 1024;

/// # MAllocPg64
///
/// Zero-sized, mmap-based allocator with inherent 64-byte alignment.
///
/// Uses anonymous mmap for all allocations, rounded to 2MB huge page
/// boundaries. Growth uses mremap for zero-copy virtual address
/// remapping. Transparent huge pages are hinted via madvise.
///
/// mmap returns page-aligned memory, so 64-byte alignment is
/// inherently satisfied without explicit alignment enforcement.
#[derive(Copy, Clone, Default, Debug)]
pub struct MAllocPg64;

/// Round `size` up to the appropriate page boundary.
///
/// With `giant_pages` enabled, sizes >= 1GB round to 1GB.
/// All other sizes round to 2MB. Zero-sized requests get one
/// 2MB page.
#[inline]
fn mapped_size(size: usize) -> usize {
    #[cfg(feature = "giant_pages")]
    if size >= GIGANTIC_PAGE {
        return (size + GIGANTIC_PAGE - 1) & !(GIGANTIC_PAGE - 1);
    }
    let rounded = (size + HUGE_PAGE - 1) & !(HUGE_PAGE - 1);
    if rounded == 0 { HUGE_PAGE } else { rounded }
}

/// Whether this allocation size qualifies for gigantic 1GB pages.
#[inline]
fn uses_giant_pages(_size: usize) -> bool {
    #[cfg(feature = "giant_pages")]
    { _size >= GIGANTIC_PAGE }
    #[cfg(not(feature = "giant_pages"))]
    { false }
}

/// Perform the mmap syscall with appropriate flags.
fn do_mmap(original_size: usize, mapped: usize) -> Result<NonNull<u8>, AllocError> {
    #[cfg(feature = "giant_pages")]
    let flags = if original_size >= GIGANTIC_PAGE {
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_HUGE_1GB
    } else {
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS
    };

    #[cfg(not(feature = "giant_pages"))]
    let flags = {
        let _ = original_size;
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS
    };

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            mapped,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };

    if ptr == libc::MAP_FAILED {
        return Err(AllocError);
    }

    NonNull::new(ptr as *mut u8).ok_or(AllocError)
}

/// Hint transparent huge pages on a mapping.
/// Skipped for gigantic page allocations which already use explicit huge pages.
#[inline]
fn hint_thp(ptr: NonNull<u8>, original_size: usize, mapped: usize) {
    if uses_giant_pages(original_size) {
        return;
    }
    unsafe {
        libc::madvise(ptr.as_ptr() as *mut libc::c_void, mapped, libc::MADV_HUGEPAGE);
    }
}

/// Wrap a raw pointer and mapped size into the fat NonNull<[u8]> the Allocator trait expects.
///
/// # Safety
/// `ptr` must be non-null and valid for `mapped` bytes.
#[inline]
unsafe fn fat_ptr(ptr: NonNull<u8>, mapped: usize) -> NonNull<[u8]> {
    unsafe { NonNull::new_unchecked(slice_from_raw_parts_mut(ptr.as_ptr(), mapped)) }
}

unsafe impl Allocator for MAllocPg64 {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        let mapped = mapped_size(size);
        let ptr = do_mmap(size, mapped)?;
        hint_thp(ptr, size, mapped);
        Ok(unsafe { fat_ptr(ptr, mapped) })
    }

    /// mmap anonymous pages are zero-filled by the kernel.
    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let mapped = mapped_size(layout.size());
        unsafe {
            libc::munmap(ptr.as_ptr() as *mut libc::c_void, mapped);
        }
    }

    /// Grows the mapping using mremap for zero-copy virtual address remapping.
    ///
    /// If the new size fits within the current mapped region, returns
    /// the existing pointer without a syscall.
    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_mapped = mapped_size(old.size());
        let new_mapped = mapped_size(new.size());

        if old_mapped == new_mapped {
            return Ok(unsafe { fat_ptr(ptr, new_mapped) });
        }

        let raw = unsafe {
            libc::mremap(
                ptr.as_ptr() as *mut libc::c_void,
                old_mapped,
                new_mapped,
                libc::MREMAP_MAYMOVE,
            )
        };

        if raw == libc::MAP_FAILED {
            return Err(AllocError);
        }

        let nn = NonNull::new(raw as *mut u8).ok_or(AllocError)?;
        hint_thp(nn, new.size(), new_mapped);
        Ok(unsafe { fat_ptr(nn, new_mapped) })
    }

    /// mremap extends with anonymous pages which are zero-filled by the kernel.
    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.grow(ptr, old, new) }
    }

    /// Shrinks the mapping by releasing tail pages via munmap.
    ///
    /// If the new size still fits within the same mapped region,
    /// returns the existing pointer without a syscall.
    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_mapped = mapped_size(old.size());
        let new_mapped = mapped_size(new.size());

        if old_mapped == new_mapped {
            return Ok(unsafe { fat_ptr(ptr, new_mapped) });
        }

        // Crossing from gigantic to non-gigantic page sizes cannot
        // partially unmap 1GB pages at non-1GB boundaries. Fall back
        // to a fresh allocation with copy.
        if uses_giant_pages(old.size()) && !uses_giant_pages(new.size()) {
            let new_ptr = do_mmap(new.size(), new_mapped)?;
            unsafe {
                core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), new.size());
                libc::munmap(ptr.as_ptr() as *mut libc::c_void, old_mapped);
            }
            hint_thp(new_ptr, new.size(), new_mapped);
            return Ok(unsafe { fat_ptr(new_ptr, new_mapped) });
        }

        // Release tail pages
        unsafe {
            libc::munmap(
                (ptr.as_ptr() as usize + new_mapped) as *mut libc::c_void,
                old_mapped - new_mapped,
            );
        }

        Ok(unsafe { fat_ptr(ptr, new_mapped) })
    }

    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{Allocator, Layout};

    #[test]
    fn test_allocate_alignment() {
        let a = MAllocPg64;
        for size in [1, 64, 4096, HUGE_PAGE, HUGE_PAGE + 1] {
            let layout = Layout::from_size_align(size, 1).unwrap();
            let ptr = a.allocate(layout).expect("allocate failed");
            let addr = ptr.as_non_null_ptr().as_ptr() as usize;
            assert_eq!(
                addr % 64, 0,
                "Pointer {:#x} not 64-byte aligned for size {}", addr, size
            );
            unsafe { a.deallocate(ptr.as_non_null_ptr(), layout) };
        }
    }

    #[test]
    fn test_allocate_returns_mapped_size() {
        let a = MAllocPg64;
        let layout = Layout::from_size_align(100, 1).unwrap();
        let ptr = a.allocate(layout).expect("allocate failed");
        // Returned slice length should be at least HUGE_PAGE
        assert!(ptr.len() >= HUGE_PAGE);
        unsafe { a.deallocate(ptr.as_non_null_ptr(), layout) };
    }

    #[test]
    fn test_allocate_zeroed() {
        let a = MAllocPg64;
        let layout = Layout::from_size_align(4096, 1).unwrap();
        let ptr = a.allocate_zeroed(layout).expect("allocate_zeroed failed");
        let data = unsafe {
            std::slice::from_raw_parts(ptr.as_non_null_ptr().as_ptr(), 4096)
        };
        assert!(data.iter().all(|&b| b == 0));
        unsafe { a.deallocate(ptr.as_non_null_ptr(), layout) };
    }

    #[test]
    fn test_grow_preserves_data() {
        let a = MAllocPg64;
        let small = Layout::from_size_align(1024, 1).unwrap();
        let ptr = a.allocate(small).expect("allocate failed");

        // Write a pattern
        let data = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Grow beyond one huge page to force mremap
        let big = Layout::from_size_align(HUGE_PAGE + 4096, 1).unwrap();
        let grown = unsafe {
            a.grow(ptr.as_non_null_ptr(), small, big).expect("grow failed")
        };

        // Verify data preserved
        let addr = grown.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0, "Grown pointer not 64-byte aligned");
        let check = unsafe {
            std::slice::from_raw_parts(grown.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, &byte) in check.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data corrupted at offset {}", i);
        }

        unsafe { a.deallocate(grown.as_non_null_ptr(), big) };
    }

    #[test]
    fn test_grow_within_mapping_noop() {
        let a = MAllocPg64;
        // Allocate 100 bytes - gets rounded to 2MB
        let small = Layout::from_size_align(100, 1).unwrap();
        let ptr = a.allocate(small).expect("allocate failed");
        let original_addr = ptr.as_non_null_ptr().as_ptr() as usize;

        // Grow to 1MB - still within the same 2MB mapping
        let medium = Layout::from_size_align(1024 * 1024, 1).unwrap();
        let grown = unsafe {
            a.grow(ptr.as_non_null_ptr(), small, medium).expect("grow failed")
        };
        let grown_addr = grown.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(original_addr, grown_addr, "Should reuse same mapping");

        unsafe { a.deallocate(grown.as_non_null_ptr(), medium) };
    }

    #[test]
    fn test_shrink() {
        let a = MAllocPg64;
        // Allocate across two huge pages
        let big = Layout::from_size_align(HUGE_PAGE * 3, 1).unwrap();
        let ptr = a.allocate(big).expect("allocate failed");

        // Write pattern
        let data = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Shrink to one huge page
        let small = Layout::from_size_align(HUGE_PAGE, 1).unwrap();
        let shrunk = unsafe {
            a.shrink(ptr.as_non_null_ptr(), big, small).expect("shrink failed")
        };

        let addr = shrunk.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0, "Shrunk pointer not 64-byte aligned");

        // Verify data preserved
        let check = unsafe {
            std::slice::from_raw_parts(shrunk.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, &byte) in check.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data corrupted at offset {}", i);
        }

        unsafe { a.deallocate(shrunk.as_non_null_ptr(), small) };
    }

    #[test]
    fn test_grow_zeroed_new_region_is_zero() {
        let a = MAllocPg64;
        let small = Layout::from_size_align(1024, 1).unwrap();
        let ptr = a.allocate(small).expect("allocate failed");

        // Grow beyond current mapping to force mremap
        let big = Layout::from_size_align(HUGE_PAGE + 4096, 1).unwrap();
        let grown = unsafe {
            a.grow_zeroed(ptr.as_non_null_ptr(), small, big).expect("grow_zeroed failed")
        };

        // Check that the new region past the original huge page is zeroed
        let new_start = HUGE_PAGE;
        let check = unsafe {
            std::slice::from_raw_parts(
                grown.as_non_null_ptr().as_ptr().add(new_start),
                4096,
            )
        };
        assert!(check.iter().all(|&b| b == 0), "New region not zeroed after grow_zeroed");

        unsafe { a.deallocate(grown.as_non_null_ptr(), big) };
    }

    #[test]
    fn test_mapped_size_rounding() {
        assert_eq!(mapped_size(0), HUGE_PAGE);
        assert_eq!(mapped_size(1), HUGE_PAGE);
        assert_eq!(mapped_size(HUGE_PAGE), HUGE_PAGE);
        assert_eq!(mapped_size(HUGE_PAGE + 1), HUGE_PAGE * 2);
        assert_eq!(mapped_size(HUGE_PAGE * 3), HUGE_PAGE * 3);
    }

    #[test]
    fn test_by_ref() {
        let a = MAllocPg64;
        let b = a.by_ref();
        assert!(std::ptr::eq(&a, b));
    }
}
