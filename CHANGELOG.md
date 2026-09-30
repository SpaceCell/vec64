# Changelog

## [0.5.3] - 2026-10-01

Nightly compatibility release.

The 2026-09-30 Rust nightly stabilises the core of `allocator_api` and moves
the remaining unstable allocator surface, including `Vec::drain` on custom
allocators, to the new `allocator_ext` feature. The crate now enables
`allocator_ext` in place of `allocator_api` and requires the 2026-09-30
nightly or later. There is no public API change.

## [0.5.2] - 2026-09-16

Added `Vec64::zeroed`, an unsafe constructor that allocates zero-initialised
elements through the allocator's zeroed path, avoiding a separate fill pass.

## [0.5.1] - 2026-08-28

Nightly compatibility release.

The unstable `allocator_api` raw-parts methods were renamed for consistency
with `Box::into_raw_with_allocator`, so `Vec::into_raw_parts_with_alloc` is
now `Vec::into_raw_parts_with_allocator`. This only affects the `mmap`
feature's internal `try_mremap_append` path, so there is no public API
change and this is a patch version bump.

## [0.5.0] - 2026-08-15

Nightly compatibility release.

As of the 2026-08-14 Rust nightly the unstable `allocator_api` `Allocator`
trait no longer provides a `by_ref` method, so the `by_ref` overrides on
`Alloc64` and `MAllocPg64` stopped compiling. Both overrides are removed.
Where code previously called `alloc.by_ref()`, take a reference to the
allocator with `&alloc`, which acts as an allocator through the standard
`impl Allocator for &A` blanket. Removing the methods is a breaking change,
so this is a minor version bump.
