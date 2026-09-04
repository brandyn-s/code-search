# Epoch-Manifest Design

How code-search publishes an index generation atomically and how readers fall
back to the last verified generation. Implemented in `search/epoch_manifest.py`;
write paths in `search/indexer.py` and `mcp_server/code_search_server.py` commit
through it, and `verify_index_integrity` reports its status.

## Why

An early incident truncated `chunk_ids.pkl` while other artifacts kept their
full length, leaving an index that loaded but returned wrong results. The root
cause was that index writes had no atomic-or-fail discipline; per-file guards
only patched individual symptoms.

The structural primitive that closes the entire incident class is an
epoch-manifest: a single source of truth for what constitutes a "committed"
index state, with atomic-rename commit semantics and reader fallback to the
prior epoch on checksum failure.


## Manifest contents

A manifest is a JSON file describing one consistent epoch of an index:

```json
{
  "version": 1,
  "epoch_id": "2026-05-05T22-31-08-abcdef12",
  "created_at": "2026-05-05T22:31:08.123456+00:00",
  "provider": "voyage",
  "model": "voyage-4-large",
  "vector_dim": 1024,
  "quantization": "int8",
  "pipeline_version": "f9c4...",
  "artifacts": {
    "chunk_ids.pkl": {
      "path": "index/chunk_ids.pkl",
      "sha256": "abcd...",
      "bytes": 456789,
      "count": 10093
    },
    "metadata.db": {
      "path": "index/metadata.db",
      "sha256": "efgh...",
      "bytes": 12345678,
      "count": 10093
    },
    "fts5.db": {
      "path": "index/fts5.db",
      "sha256": "ijkl...",
      "bytes": 9876543,
      "count": 10093
    },
    "code.index": {
      "path": "index/code.index",
      "sha256": "mnop...",
      "bytes": 23456789,
      "count": 10093
    },
    "stats.json": {
      "path": "index/stats.json",
      "sha256": "qrst...",
      "bytes": 4321,
      "count": null
    }
  },
  "consistency": {
    "all_artifacts_share_count": true,
    "expected_count": 10093
  }
}
```

The `count` field across `chunk_ids.pkl` / `metadata.db` / `fts5.db` /
`code.index` MUST match. `consistency.all_artifacts_share_count` is a
post-write validation that prevents the truncation regression — if an
artifact write succeeded for some chunks but not all, the count mismatch
fails validation BEFORE the manifest gets committed.

## File layout

Per project (under `~/.claude_code_search/projects/<hash>/`):

```
projects/<hash>/
  index/
    chunk_ids.pkl
    metadata.db
    fts5.db
    code.index
    stats.json
  manifest/
    current.json    # the committed epoch readers consume
    prior.json      # the previous epoch (kept for downgrade tolerance, E3)
    candidate.json  # in-progress writer staging (transient)
```

## Write protocol (commit-last)

```
WRITE_EPOCH(new_artifacts):
  1. write artifacts to index/ (existing paths)
  2. compute manifest dict:
       - epoch_id from timestamp + random suffix
       - artifact sha256 + byte count + record count for each
       - consistency check: all_artifacts_share_count
  3. fail loud if consistency check fails:
       - log + raise ManifestConsistencyError
       - do NOT commit
  4. atomic write candidate:
       open(manifest/candidate.json, "w", encoding="utf-8")
       json.dump(manifest, f); flush; os.fsync
       close
  5. promote prior:
       if manifest/current.json exists:
         os.replace(manifest/current.json, manifest/prior.json)
   6. atomic rename:
       os.replace(manifest/candidate.json, manifest/current.json)
       # os.replace is atomic on Windows + POSIX (Python 3.3+ guarantee)
```

Failure modes and outcomes:

| Crash point | candidate.json | current.json | prior.json | Reader sees |
|-------------|---------------|-------------|-----------|-------------|
| Before step 4 | absent | unchanged | unchanged | prior epoch (clean) |
| Mid-step-4 (partial write) | partial | unchanged | unchanged | prior epoch (clean); recovery: delete candidate |
| Step 4 done, step 5 not started | complete | unchanged | unchanged | prior epoch; recovery: delete candidate |
| Step 5 done, step 6 not started | complete | absent | old current | E3 reader fallback to prior; new write has clean state |
| Step 6 done | absent | new | old current | new epoch (committed) |

The key property: **at no point can a reader see a partially-committed
state**. Either the new epoch has fully replaced the old one, or the old
one is still in place.

## Read protocol

```
READ_EPOCH():
  1. read manifest/current.json
     - if missing: raise ManifestMissing (no committed state)
  2. verify all artifact sha256 against manifest hashes
  3. on verification success: return current
  4. on verification failure:
       - log warning
       - read manifest/prior.json (if exists)
       - verify prior's artifacts
       - on prior verify success: return prior with freshness="stale_using_prior_epoch"
       - on prior verify failure: raise ManifestCorrupt (escalate to operator,
         pointing at verify_index_integrity)
```

## Cleanup of stale candidate.json

If a writer crashes between steps 3 and 4 (or partway through 4), a
stale `candidate.json` may persist. The cleanup is idempotent:

- Next writer overwrites it (step 4 opens with `mode="w"`).
- Or call `cleanup_stale_candidate()` explicitly (offered by this module).

`prior.json` is intentionally NOT cleaned up automatically — it's the
fallback for E3's reader downgrade tolerance.

## Why this design works on Windows

The plan flagged Windows rename atomicity as the hard part. Python's
`os.replace()` is documented to be atomic on Windows since Python 3.3
(unlike `os.rename()` which fails when the target exists on Windows).
We use `os.replace` exclusively for the commit step.

`CodeIndexManager` also holds a per-index advisory writer lock while it
performs startup recovery and across the complete mutation/publication
transaction. The lock uses `fcntl.flock` on POSIX and `msvcrt.locking` on
Windows. It is process-reentrant and the operating system releases it if a
writer exits unexpectedly.

`os.replace()` remains the atomic manifest commit primitive. The writer lock
solves the separate multi-process ordering problem: without it, one process
could prune a generation after another process selected it but before that
process committed `current.json`.

## What this design does NOT do

- **Not a transaction log**: each manifest is a snapshot, not a delta.
  Recovery is "read prior", not "replay journal".
- **Not multi-version**: only `current` and `prior` are kept. Older
  epochs are gone after promotion. This is acceptable because the FAISS
  index can be rebuilt from source if both fail.
- **Not concurrent multi-writer**: writers for one local index are serialized
  by the advisory lock; they do not merge concurrent working sets.
- **Not consensus / distributed**: single-machine local index. No leader
  election, no Paxos, no Raft.

## Test coverage

- Clean write + read roundtrip
- Write-crash mid-step-4 (partial candidate.json)
- Write-crash between steps 4 and 5 (candidate complete, current unchanged)
- Write-crash between steps 5 and 6 (prior promoted, current absent)
- Consistency check fails when artifact counts disagree
- Concurrent overwrite of candidate.json (last writer wins, no corruption)
- Verify checksums detects corruption
- prior.json preserved across promotions

## Out of scope

- Manifest hash signing for tamper detection.
- Compression of historical manifests; current/prior is sufficient.
