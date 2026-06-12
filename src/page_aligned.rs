//! `PageAligned` trait.

use crate::Vec64;

/// Reports page-alignment of a buffer's length.
pub trait PageAligned {
    /// Returns true when the buffer is page-aligned.
    fn is_page_aligned(&self) -> bool;

    /// Smallest aligned length `>=` the current length. `None` when no
    /// row-count answer applies.
    fn next_page_aligned_len(&self) -> Option<usize>;

    /// Largest aligned length `<=` the current length. `None` when no
    /// row-count answer applies.
    fn prev_page_aligned_len(&self) -> Option<usize>;

    /// Elements required to reach the next alignment. `Some(0)` when
    /// already aligned.
    fn aligned_len_step(&self) -> Option<usize>;
}

#[cfg(all(feature = "mmap", target_os = "linux"))]
impl<T> PageAligned for Vec64<T> {
    fn is_page_aligned(&self) -> bool {
        let elem = std::mem::size_of::<T>();
        if elem == 0 {
            return false;
        }
        let len = self.len();
        let cap = self.capacity();
        if cap != len {
            return false;
        }
        let cap_bytes = cap * elem;
        if !crate::mmap_alloc::uses_mmap(cap_bytes) {
            return false;
        }
        let len_bytes = len * elem;
        len_bytes & (crate::mmap_alloc::HUGE_PAGE - 1) == 0
    }

    fn next_page_aligned_len(&self) -> Option<usize> {
        let step = page_align_t_step::<T>()?;
        let len = self.len();
        let rem = len % step;
        Some(if rem == 0 { len } else { len + (step - rem) })
    }

    fn prev_page_aligned_len(&self) -> Option<usize> {
        let step = page_align_t_step::<T>()?;
        Some((self.len() / step) * step)
    }

    fn aligned_len_step(&self) -> Option<usize> {
        self.next_page_aligned_len().map(|n| n - self.len())
    }
}

#[cfg(not(all(feature = "mmap", target_os = "linux")))]
impl<T> PageAligned for Vec64<T> {
    fn is_page_aligned(&self) -> bool {
        false
    }
    fn next_page_aligned_len(&self) -> Option<usize> {
        None
    }
    fn prev_page_aligned_len(&self) -> Option<usize> {
        None
    }
    fn aligned_len_step(&self) -> Option<usize> {
        None
    }
}

/// Row-count step for the page-alignment of a `Vec64<T>`. Returns the
/// smallest non-zero length at which a buffer of element type `T`
/// becomes page-aligned. `None` for ZSTs or builds where the mmap path
/// is unavailable.
pub fn page_align_t_step<T>() -> Option<usize> {
    step_rows_inner(std::mem::size_of::<T>())
}

/// Row-count step for a bit-packed mask. Returns the smallest non-zero
/// bit count at which the underlying `u8` buffer is page-aligned. `None`
/// for builds where the mmap path is unavailable.
pub fn page_align_bitmask_step() -> Option<usize> {
    step_rows_inner(1).map(|n| n * 8)
}

#[cfg(all(feature = "mmap", target_os = "linux"))]
fn step_rows_inner(elem: usize) -> Option<usize> {
    if elem == 0 {
        return None;
    }
    let g = gcd(elem, crate::mmap_alloc::HUGE_PAGE);
    Some(crate::mmap_alloc::HUGE_PAGE / g)
}

#[cfg(not(all(feature = "mmap", target_os = "linux")))]
fn step_rows_inner(_elem: usize) -> Option<usize> {
    None
}

#[cfg(all(feature = "mmap", target_os = "linux"))]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vec64_is_not_aligned() {
        let v: Vec64<u64> = Vec64::new();
        assert!(!v.is_page_aligned());
    }

    #[test]
    fn heap_backed_vec64_is_not_aligned() {
        let v: Vec64<u64> = (0..100u64).collect();
        assert!(!v.is_page_aligned());
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn mmap_backed_unaligned_is_not_aligned() {
        let n = 200_000usize;
        let mut v: Vec64<u64> = Vec64::with_capacity(n);
        for i in 0..n as u64 {
            v.push(i);
        }
        assert!(!v.is_page_aligned());
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn mmap_backed_exact_huge_page_multiple_is_aligned() {
        let n = crate::mmap_alloc::HUGE_PAGE / 8;
        let mut v: Vec64<u64> = Vec64::with_capacity(n);
        for i in 0..n as u64 {
            v.push(i);
        }
        assert_eq!(v.len(), n);
        assert_eq!(v.capacity(), n);
        assert!(v.is_page_aligned());
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn mmap_backed_loose_capacity_is_not_aligned() {
        let n = crate::mmap_alloc::HUGE_PAGE / 8;
        let mut v: Vec64<u64> = Vec64::with_capacity(n + 10);
        for i in 0..n as u64 {
            v.push(i);
        }
        // len matches the page-multiple target, but cap was rounded up by the allocator.
        assert!(!v.is_page_aligned());
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn next_page_aligned_len_for_u64() {
        let v: Vec64<u64> = Vec64::new();
        assert_eq!(v.next_page_aligned_len(), Some(0));
        let step = crate::mmap_alloc::HUGE_PAGE / 8;

        let mut v: Vec64<u64> = Vec64::with_capacity(step);
        for _ in 0..1u64 {
            v.push(0);
        }
        assert_eq!(v.next_page_aligned_len(), Some(step));

        for _ in 1..step as u64 - 1 {
            v.push(0);
        }
        // len = step - 1, next is step
        assert_eq!(v.next_page_aligned_len(), Some(step));

        v.push(0);
        assert_eq!(v.len(), step);
        assert_eq!(v.next_page_aligned_len(), Some(step));
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn prev_page_aligned_len_for_u64() {
        let step = crate::mmap_alloc::HUGE_PAGE / 8;
        let mut v: Vec64<u64> = Vec64::with_capacity(step);
        for _ in 0..step as u64 / 2 {
            v.push(0);
        }
        assert_eq!(v.prev_page_aligned_len(), Some(0));

        for _ in 0..step as u64 / 2 {
            v.push(0);
        }
        assert_eq!(v.len(), step);
        assert_eq!(v.prev_page_aligned_len(), Some(step));

        v.push(0);
        assert_eq!(v.prev_page_aligned_len(), Some(step));
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn aligned_len_step_basic() {
        let step = crate::mmap_alloc::HUGE_PAGE / 8;
        let mut v: Vec64<u64> = Vec64::with_capacity(step);
        for _ in 0..(step / 4) as u64 {
            v.push(0);
        }
        assert_eq!(v.aligned_len_step(), Some(step - step / 4));

        while v.len() < step {
            v.push(0);
        }
        assert_eq!(v.aligned_len_step(), Some(0));
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn step_size_depends_on_t() {
        // u8 (1 byte): step = HUGE_PAGE
        // u32 (4 bytes): step = HUGE_PAGE / 4
        // u64 (8 bytes): step = HUGE_PAGE / 8
        // u16 (2 bytes): step = HUGE_PAGE / 2
        let v_u8: Vec64<u8> = Vec64::new();
        assert_eq!(v_u8.next_page_aligned_len(), Some(0));
        let v_u8: Vec64<u8> = (0..1u8).collect();
        assert_eq!(v_u8.next_page_aligned_len(), Some(crate::mmap_alloc::HUGE_PAGE));

        let v_u32: Vec64<u32> = (0..1u32).collect();
        assert_eq!(v_u32.next_page_aligned_len(), Some(crate::mmap_alloc::HUGE_PAGE / 4));

        let v_u64: Vec64<u64> = (0..1u64).collect();
        assert_eq!(v_u64.next_page_aligned_len(), Some(crate::mmap_alloc::HUGE_PAGE / 8));
    }

    /// Integration check: when the source satisfies `is_page_aligned()`
    /// and the destination has page-aligned write position with adequate
    /// spare capacity, `extend_from_vec64` produces a correct contiguous
    /// result. The strict assertion that the mremap path executed lives
    /// in `vec64.rs::test_try_mremap_append_succeeds_under_valid_conditions`.
    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn aligned_source_extends_correctly() {
        let n = crate::mmap_alloc::HUGE_PAGE / 8;

        let mut src: Vec64<u64> = Vec64::with_capacity(n);
        for i in 0..n as u64 {
            src.push(i);
        }
        assert!(src.is_page_aligned(),
            "source built to spec must report is_page_aligned() == true");

        let mut dst: Vec64<u64> = Vec64::with_capacity(n * 2);
        for i in 0..n as u64 {
            dst.push(i * 1000);
        }
        // dst write position (len * sizeof(T)) is a HUGE_PAGE multiple.
        // Capacity exceeds length, so is_page_aligned() does not apply here
        // (it is a source-side predicate); dst-side validity is the write
        // position being page-aligned plus having spare capacity.
        assert_eq!((dst.len() * std::mem::size_of::<u64>()) % crate::mmap_alloc::HUGE_PAGE, 0);
        assert!(dst.capacity() - dst.len() >= n);

        dst.extend_from_vec64(src);

        assert_eq!(dst.len(), n * 2);
        for i in 0..n {
            assert_eq!(dst[i], i as u64 * 1000, "head corruption at {i}");
        }
        for i in 0..n {
            assert_eq!(dst[n + i], i as u64, "tail corruption at {i}");
        }
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn zst_returns_none() {
        let v: Vec64<()> = Vec64::new();
        assert!(!v.is_page_aligned());
        assert_eq!(v.next_page_aligned_len(), None);
        assert_eq!(v.prev_page_aligned_len(), None);
        assert_eq!(v.aligned_len_step(), None);
    }
}
