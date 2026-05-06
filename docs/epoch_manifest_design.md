# Epoch-Manifest Design (Plan-2 E1)

**Status**: Reference implementation shipped (this PR). Migrating production
write paths to use it is Plan-2 E2 (separate PRs); reader path with
downgrade tolerance is E3.

**Source**: `~/Documents/knowledge-base/plans/2026-05-05-codesearch-recommendations.md` Phase E1.

## Why

The chunk_ids.pkl truncation incident (PR #97-#103, 2026-05-04/05) was the
visible symptom of a deeper class: code-search has no atomic-or-fail
discipline across its index write paths. PR #98's load-before-modify, #99's
count guard, and #103's cleanup tool are tactical guards.

The structural primitive that closes the entire incident class is an
epoch-manifest: a single source of truth for what constitutes a "committed"
index state, with atomic-rename commit semantics and reader fallback to the
prior epoch on checksum failure.

The roundtable (2026-05-05) flagged this as the keystone — 3-of-3 LLMs
converged on it as the highest-leverage structural fix.

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

## Read protocol (E3 — separate PR)

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

We do NOT use file locks (`fcntl.flock` / `msvcrt.locking`) because:
1. `os.replace` is atomic; concurrent writers either lose or win the race
   without leaving a half-committed state.
2. Cross-platform file locking adds complexity without solving the
   real problem (which is crash recovery, not write contention).

The single-writer assumption is documented (one MCP server per project at
a time). If multi-writer is ever needed, an advisory lockfile next to
`current.json` can be added.

## What this design does NOT do

- **Not a transaction log**: each manifest is a snapshot, not a delta.
  Recovery is "read prior", not "replay journal".
- **Not multi-version**: only `current` and `prior` are kept. Older
  epochs are gone after promotion. This is acceptable because the FAISS
  index can be rebuilt from source if both fail.
- **Not multi-writer**: single writer per project. Documented; not
  enforced by code.
- **Not consensus / distributed**: single-machine local index. No leader
  election, no Paxos, no Raft.

## Migration path (E2)

E2 will wire production write paths into this primitive:

1. `search/indexer.py::add_embeddings`: after current write completes,
   compute manifest + commit.
2. `mcp_server/code_search_server.py::index_directory`: same hook.
3. `scripts/cleanup_index_orphans.py --apply-stats`: re-write stats.json
   under manifest commit instead of direct write.
4. Existing PR #98 `load-before-modify` pattern in `add_embeddings`
   becomes redundant once manifest commit-last semantics are in place;
   explicitly retire it with a note in the diff.

E3 will then add the reader downgrade tolerance.

## Test coverage in this PR

- Clean write + read roundtrip
- Write-crash mid-step-4 (partial candidate.json)
- Write-crash between steps 4 and 5 (candidate complete, current unchanged)
- Write-crash between steps 5 and 6 (prior promoted, current absent)
- Consistency check fails when artifact counts disagree
- Concurrent overwrite of candidate.json (last writer wins, no corruption)
- Verify checksums detects corruption
- prior.json preserved across promotions

## Out of scope for E1

- Migrating production write paths (E2)
- Reader downgrade tolerance using prior.json (E3)
- Manifest hash signing for tamper detection (deferred — single-user prototype)
- Compression of historical manifests (deferred — current/prior is sufficient)
