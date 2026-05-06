# Index Write Paths Audit (Plan-2 E2)

**Status**: Audit complete. Per-path migration decisions recorded.
**Date**: 2026-05-05
**Plan**: Plan-2 Phase E2 (`~/Documents/knowledge-base/plans/2026-05-05-codesearch-recommendations.md`).

The epoch-manifest primitive shipped in PRs #111 (E1) + #114 (E3) defines a structural commit-or-fail discipline for the index. This audit lists every code path that mutates index artifacts, and decides per path whether to (a) migrate to manifest-commit, (b) keep direct write because it's not part of the index proper, or (c) deprecate.

Each migration is a future PR. Order matters: paths that produce inconsistent state if interleaved should migrate together.

## Inventory

### Write paths in `search/indexer.py`

| # | Path | What it writes | Lines | Decision |
|---|------|----------------|-------|----------|
| 1 | `add_embeddings()` first batch | chunk_ids.pkl + metadata.db + fts5.db + code.index + stats.json | 365-460 | **migrate** |
| 2 | `add_embeddings()` incremental | same artifacts (append) | 365-460 | **migrate** (same path) |
| 3 | `save_index()` direct save | chunk_ids.pkl + code.index | 667-757 | **migrate** |
| 4 | `save_index(force=True)` | same | 667-757 | **migrate** (same path) |
| 5 | `_persist_stats()` | stats.json | 832-852 | **migrate** (combine with #3) |
| 6 | `remove_file_chunks()` | chunk_ids.pkl + metadata.db + fts5.db + code.index | 611-665 | **migrate** |
| 7 | `clear_index()` | deletes ALL artifacts | 862+ | **deprecate** (use delete_project workflow instead) |

### Write paths in `mcp_server/code_search_server.py`

| # | Path | What it writes | Lines | Decision |
|---|------|----------------|-------|----------|
| 8 | `index_directory()` | calls indexer.add_embeddings via incremental_indexer; full corpus | 700+ | **inherits #1/#2** |
| 9 | `delete_project()` | deletes project directory entirely | 1555+ | **keep direct** — atomic-or-fail of deletion is filesystem-level; manifest doesn't help |
| 10 | `cancel_indexing()` | sets cancel flag; doesn't write artifacts directly | 1664+ | **keep direct** — not an artifact write |
| 11 | `index_directory()` writes `project_info.json` | project_info.json | 191 | **keep direct** — not part of the index epoch (per-project metadata) |
| 12 | `index_directory()` writes `project_info.pipeline_version` | project_info.json | 785 | **keep direct** — same |
| 13 | Search-time auto-reindex (search_code) | calls #8 path | (via search_code) | **inherits #1** |
| 14 | F2 background reindex (PR #112) | calls #8 path | (via _dispatch_background_reindex) | **inherits #1** |

### Write paths in `scripts/`

| # | Path | What it writes | Decision |
|---|------|----------------|----------|
| 15 | `cleanup_index_orphans.py --apply-fts5` | deletes orphan rows in fts5.db | **keep direct (recovery)** — runs against quiesced index; producing a manifest mid-recovery would conflate "good state" with "post-cleanup state" |
| 16 | `cleanup_index_orphans.py --apply-metadata` | deletes orphan keys in metadata.db | **keep direct (recovery)** — same |
| 17 | `cleanup_index_orphans.py --apply-stats` | rewrites stats.json from authoritative pkl | **migrate** — the script could compute a fresh manifest after rewriting stats.json so the post-recovery state is committed |

### Write paths in `merkle/`

| # | Path | What it writes | Decision |
|---|------|----------------|----------|
| 18 | `snapshot_manager.save_snapshot()` | snapshot.json + metadata.json | **keep direct** — these are change-detection inputs, NOT index artifacts. The merkle DAG is computed from on-disk source files, not from the index. |

## Migration ordering

Migrations should ship in this order to keep each PR independently revertable:

### PR E2-1: `save_index()` + `_persist_stats()` (paths #3, #4, #5)

**Why first**: this is the simplest call site. It touches a defined set of artifacts in one place. If the manifest commit goes wrong, only the stats/chunk_ids state is at risk.

Migration shape:
1. After existing writes complete, build manifest from current on-disk state.
2. Call `commit_manifest()`.
3. Existing readers continue to consume artifacts directly (no read-side change yet).

PR E3 (#114) already lays the read-side groundwork (`read_with_fallback`); this PR only writes the manifest, doesn't make it authoritative.

### PR E2-2: `add_embeddings()` (paths #1, #2)

**Why second**: highest-traffic write path; depends on E2-1's pattern being proven on a simpler path. Same shape: writes complete first, manifest committed last.

### PR E2-3: `remove_file_chunks()` (path #6)

**Why third**: similar shape but smaller blast radius. Migration verifies the manifest framework handles "shrinking" the index correctly (record counts decrease).

### PR E2-4: `cleanup_index_orphans.py --apply-stats` (path #17)

**Why last**: this is the recovery tool. Migrating it last means the rest of the system already produces manifests; recovery just brings stats.json back into manifest-consistent state.

### NOT migrating (decisions)

- **path #7** `clear_index()`: deprecate. The codebase's clear-state-and-rebuild workflow goes through `delete_project()` (path #9). `clear_index()` is dead-code-by-policy; future PR can remove it.
- **paths #9, #10, #11, #12, #18**: keep direct. Reasons documented inline above.
- **paths #15, #16**: keep direct. Recovery tooling shouldn't produce manifests during the recovery itself; it produces them on the final stats.json rewrite (path #17 migration).

## What gets retired after E2 ships

- **PR #98 load-before-modify pattern** in `add_embeddings`: subsumed by E2-2's manifest commit. The pattern was a tactical guard that prevented `chunk_ids.pkl` from being clobbered when in-memory state was empty. Manifest commit-last semantics make it impossible to commit a manifest with fewer artifacts than the prior committed state — so the load-before-modify check becomes redundant. Explicit retirement note in E2-2's PR description.

- **PR #99 count-based truncation guard**: same logic. The manifest's `consistency.all_artifacts_share_count` check is the same invariant, enforced at a higher level.

- **PR #103 `cleanup_index_orphans.py`**: stays as a recovery tool for manifests created PRE-migration. Once all paths are migrated, orphan production becomes structurally impossible (consistency check fails BEFORE commit), but the tool stays useful for migrating legacy on-disk state.

## What changes in `verify_index_integrity` MCP tool (A3, PR #105)

Currently scans pkl/metadata/fts5/stats consistency directly. After E2 ships, extend to ALSO verify:
- `manifest/current.json` exists and parses
- artifact sha256s match the manifest
- `manifest/prior.json` exists if a second commit has happened
- `cleanup_stale_candidate()` would be a no-op (no leftover candidate.json)

This extension is its own follow-up PR (call it E2-5).

## Read-side migration (separate from this audit)

E3 (PR #114) shipped `read_with_fallback()`. Production read paths that consume manifests (vs reading artifacts directly) are a SEPARATE migration:

- `search_code` would consume `_metadata.freshness` from `read_with_fallback().freshness` instead of computing it from env vars (current F2 behavior).
- `verify_index_integrity` would call `read_with_fallback()` to surface ReadResult.freshness in its output.

Tracked as future work; not in this audit's scope.

## Cross-references

- E1 (PR #111) — primitive: `commit_manifest`, `verify_manifest`, `cleanup_stale_candidate`
- E3 (PR #114) — reader: `read_with_fallback`, `ReadResult`
- A3 (PR #105) — `verify_index_integrity` MCP tool (extension target)
- F2 (PR #112) — `_metadata.freshness` in search responses (downstream consumer)
- PR #98/99/103 — tactical guards retired after E2-2
