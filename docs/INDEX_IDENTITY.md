# Index identity contract

Code Search records the exact checkout state used to build an index. The
envelope is designed to compare cleanly with other indexing engines without
sharing local storage:

```json
{
  "schema_version": 1,
  "repository_id": "<sha256>",
  "checkout_id": "<sha256>",
  "source_revision": "<git-head-or-unborn>",
  "dirty_fingerprint": "<clean-or-sha256>",
  "index_generation": "<sha256>",
  "captured_at": "<utc-rfc3339>"
}
```

## Repository and checkout identifiers

`repository_id` is SHA-256 over the UTF-8 bytes of:

```text
remote:<normalized-origin>
```

Origin normalization trims surrounding whitespace and trailing `/` and
`.git`, translates SCP-style remotes to `https://host/path`, lowercases the
URL scheme and host, and removes credentials, query parameters, and
fragments. If no origin is available, the seed is the resolved POSIX checkout
path:

```text
path:<resolved-root>
```

`checkout_id` is always SHA-256 over that path seed. Thus clones of the same
remote share a repository ID but have distinct checkout IDs.

## Source and dirty state

`source_revision` is the full Git `HEAD` object ID, or the literal `unborn`
for an initialized repository without a commit. Every Git subprocess inherits
the caller's environment with `LC_ALL=C`.

A clean porcelain status produces the literal `dirty_fingerprint` value
`clean`. A dirty checkout produces SHA-256 over the following frames, in this
exact order:

1. `STATUS`
2. `WORKTREE_DIFF`
3. `CACHED_DIFF`
4. one `UNTRACKED` frame per raw path, sorted lexicographically

Every frame is:

```text
ASCII label || NUL || uint64 big-endian payload length || raw payload
```

The first three payloads are the raw stdout bytes from:

```text
git status --porcelain=v1 -z --untracked-files=all
git diff --binary --no-ext-diff <base> --
git diff --binary --no-ext-diff --cached <base> --
```

`<base>` is the captured source-revision object ID for a committed repository
and Git's empty-tree object
`4b825dc642cb6eb9a060e54bf8d69288fbee4904` for an unborn repository.

Each `UNTRACKED` payload is:

```text
raw path || NUL || kind || NUL || raw 32-byte SHA-256 digest
```

`kind` is `file`, `symlink`, or `missing`. File contents and the raw symlink
target are hashed. A path that disappears during capture uses `missing` and
32 zero bytes. Other ambiguous reads fail capture rather than guessing.

`index_generation` is SHA-256 over:

```text
repository_id || NUL || source_revision || NUL || dirty_fingerprint
```

All three fields are encoded as UTF-8.

## Publication and readiness

The server captures identity synchronously before starting the background
worker and immediately invalidates any previously ready identity. After the
underlying index reports success, it captures again. It publishes `ready`
only when the two `index_generation` values are equal.

`index_ready` is the stable orchestration field:

- `true` means a completed index has a persisted envelope and a fresh live
  capture still matches its generation.
- `false` covers indexing, failure, cancellation, legacy indexes, capture
  errors, and stale source.

Detailed state is carried by `index_identity_status`:

- `indexing`: a new run has invalidated the previous envelope.
- `ready`: start and end captures matched.
- `source_changed_during_index`: the source changed while indexing; the job
  fails and no envelope is published.
- `stale_source`: the persisted index was coherent when built, but a live
  status capture now differs. The persisted envelope remains attached so
  consumers can identify what was indexed.
- `legacy_missing`: an older index has no identity envelope.
- `error` or `cancelled`: the run cannot be treated as cross-engine ready.

Non-Git indexing remains available for legacy callers, but identity capture
reports an error and `index_ready` remains false. Re-run `index_directory`
against a stable Git checkout to recover any non-ready state.
