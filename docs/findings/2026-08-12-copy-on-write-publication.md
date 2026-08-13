# Copy-on-write index publication

## Decision

Use macOS `clonefile(2)` when publishing the mutable root mirrors of an
immutable index generation. Keep the existing byte-copy path on unsupported
filesystems and non-macOS hosts.

## Diagnosis

Atomic publication intentionally keeps two views of each artifact: an
immutable generation for recovery and a root-level mirror for existing
readers. On both a public n=80 index and the preserved LLVM-scale index, the
two views had distinct inodes and occupied separate physical blocks. The LLVM
generation used 2,275,016,704 allocated bytes and its root mirror used another
2,275,016,704 bytes, so publication doubled the artifact allocation before
later writes.

Hard links are not safe here: SQLite and FAISS consumers may mutate the root
mirror, which would also mutate the supposedly immutable recovery generation.
An APFS clone begins with distinct inodes and shared blocks, then separates
only the blocks that a writer changes.

## Measurement

A controlled replay used the same 282,106,413-byte `code.index` source:

| Publication method | Newly allocated blocks |
|---|---:|
| Ordinary copy | 282,017,792 bytes |
| APFS clone | 446,464 bytes |

The clone therefore avoided 99.84% of the initial physical allocation in this
single-artifact replay while preserving byte equality and independent writes.
The production test also verifies that every published root artifact has a
different inode from its immutable-generation counterpart.

## Boundary

This reduces initial publication storage on clone-capable macOS filesystems;
it does not change logical file sizes, compact existing indexes retroactively,
or promise the same saving after extensive root-mirror mutation. Unsupported
clone errors (`ENOTSUP`, `EXDEV`, and `EINVAL`) fall back to the prior portable
copy. Unexpected clone failures remain fail-closed so publication cannot hide
an I/O or permission error.
