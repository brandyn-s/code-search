# Index format versioning

Every published index records `index_format_version` in its
`project_info.json`. The constant lives in `search/index_format.py`:

| Constant | Value | Meaning |
|---|---|---|
| `INDEX_FORMAT_VERSION` | 1 | Format this build writes |
| `MIN_SUPPORTED_INDEX_FORMAT` | 1 | Oldest format this build still reads |

Indexes written before the field existed carry no version and are treated as
format 1.

## What a reader does

| Recorded version | `get_index_status` | `index_directory(incremental=True)` |
|---|---|---|
| within the supported range | normal status | normal incremental run |
| newer than this build | `index_identity_status: index_format_newer`, `index_ready: false`, message says to upgrade code-search or rebuild | fails with the same message; the index is not touched |
| older than the minimum | `index_identity_status: index_format_unsupported`, message says a reindex is required | forced to a full rebuild |
| not an integer | treated as unsupported | forced to a full rebuild |

`index_directory(incremental=False)` always rebuilds and writes the current
version, so it is the universal recovery path.

## When to bump

Bump `INDEX_FORMAT_VERSION` when the layout under `index/` changes in a way an
older reader cannot handle: the FAISS file, `chunk_ids.pkl`, the SQLite
schemas of `fts5.db` or `metadata.db`, the epoch manifest, or the
`project_info.json` fields a reader depends on. Adding an optional field does
not need a bump.

Raise `MIN_SUPPORTED_INDEX_FORMAT` only when this build drops the code that
reads an older layout. That is a breaking change for users with existing
indexes and belongs in a major version with a CHANGELOG entry.

## Fixture

`tests/fixtures/index-format-v1/` is a real index of the frozen fixture corpus
built at format 1 with the deterministic test embedder. `tests/unit/test_index_format.py`
installs it under a temporary storage root and asserts that the current build
opens it, that a higher version is refused with upgrade guidance, and that a
lower version asks for a reindex. When you bump the format, build a new
fixture directory for the new version and keep the old one so the
compatibility claim stays tested.
