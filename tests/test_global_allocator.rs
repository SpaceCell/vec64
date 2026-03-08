//! Integration test that installs `Alloc64Global` as the process-wide
//! allocator and verifies standard `Vec` allocations are 64-byte aligned.

#[cfg(feature = "global")]
mod global_allocator {
    #![feature(allocator_api)]
    use vec64::Alloc64Global;

    #[global_allocator]
    static GLOBAL: Alloc64Global = Alloc64Global;

    #[test]
    fn std_vec_is_64_byte_aligned() {
        let v: Vec<u8> = vec![1u8; 128];
        let addr = v.as_ptr() as usize;
        assert_eq!(
            addr % 64,
            0,
            "Standard Vec pointer {:#x} is not 64-byte aligned",
            addr,
        );
    }

    #[test]
    fn std_vec_stays_aligned_after_grow() {
        let mut v: Vec<u64> = Vec::with_capacity(4);
        v.extend(0u64..1000);
        let addr = v.as_ptr() as usize;
        assert_eq!(
            addr % 64,
            0,
            "Vec pointer {:#x} lost 64-byte alignment after growth",
            addr,
        );
    }

    #[test]
    fn std_string_is_64_byte_aligned() {
        let s =
            String::from("hello world, this is a long enough string to force a heap allocation");
        let addr = s.as_ptr() as usize;
        assert_eq!(
            addr % 64,
            0,
            "String pointer {:#x} is not 64-byte aligned",
            addr,
        );
    }

    #[test]
    fn boxed_slice_is_64_byte_aligned() {
        let b: Box<[u8]> = vec![0u8; 256].into_boxed_slice();
        let addr = b.as_ptr() as usize;
        assert_eq!(
            addr % 64,
            0,
            "Boxed slice pointer {:#x} is not 64-byte aligned",
            addr,
        );
    }
}
