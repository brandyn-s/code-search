# Controlled minimal repro for the FAISS chunk truncation regression

> **Status**: runbook, not a script. Programmatic repro tried first
> (`repro_chunk_truncation.py`, deleted) but the IndexManager + embedder
> + chunker constructors require server-side orchestration to wire up
> correctly. The cleaner path is to exercise the full MCP code path
> against a controlled tiny project and read the new `[CHUNK_ID_DIAG]`
> log lines added to `search/indexer.py` (2026-05-05).

## Hypothesis (under test)

Post-MCP-restart, `CodeIndexManager._load_index` correctly loads the
FAISS index file but the lazy-loaded `_chunk_ids` ends up shorter than
`faiss.ntotal` because `chunk_ids.pkl` is missing/empty/truncated, AND
`_maybe_rebuild_chunk_ids` doesn't fire (it only triggers when
`faiss_n != chunk_n` in a specific direction). When the next
`incremental_index` saves, `pickle.dump(self._chunk_ids, f)` overwrites
the on-disk pkl with the truncated in-memory list, and `save_index`
also rewrites `code.index` from the (now-mutated) `self._index`.

## Prereqs

- Code-search source has the `[CHUNK_ID_DIAG]` log lines added on
  2026-05-05 (`search/indexer.py` `_load_index` and `save_index`).
- MCP server has been restarted since those changes were made (so the
  log lines actually fire — confirmed by `grep CHUNK_ID_DIAG
  ~/.claude/logs/code-search-mcp.log` returning at least one entry).
- A clean session where the .claude project is healthy (chunks > 5000).

## Steps

### Step 1: tiny project setup

```bash
mkdir -p ~/code-search-repro/tiny
for i in $(seq 0 29); do
  printf '"""mod %d."""\ndef fn_%d():\n    return %d\n' "$i" "$i" "$i" > ~/code-search-repro/tiny/mod_$(printf %03d $i).py
done
```

### Step 2: full index via MCP

Call `mcp__code-search__index_directory(directory_path="~/code-search-repro/tiny", incremental=false)`.

Wait for `mcp__code-search__get_indexing_progress` to report `phase=done`.

Snapshot the on-disk state:

```bash
find ~/.claude_code_search/projects/tiny_* -type f -exec stat -c "%n %s %Y" {} \;
python -c "
import pickle
with open('~/.claude_code_search/projects/tiny_<HASH>/index/chunk_ids.pkl', 'rb') as f:
    ids = pickle.load(f)
print('chunk_ids count:', len(ids))
"
```

Expected: 30 files, ~30-90 chunks, `code.index` ~10-50 KB,
`chunk_ids.pkl` ~1-3 KB.

### Step 3: simulate >5min staleness

The auto-reindex fires when `snapshot_age > 5 * 60` seconds. Age the
snapshot file:

```bash
HASH=$(ls ~/.claude_code_search/projects/ | grep '^tiny_' | sed 's/tiny_//')
SNAP="$HOME/.claude_code_search/merkle/${HASH}_snapshot.json"
# Backdate by 10 minutes
touch -t $(date -u -d '10 minutes ago' +%Y%m%d%H%M.%S) "$SNAP"
```

### Step 4: restart MCP

Restart Claude Code (closes and re-opens the MCP server). This forces
a fresh `CodeIndexManager` to be constructed on the next call.

### Step 5: trigger auto-reindex

After restart, modify ONE file:

```bash
echo "# touched" >> ~/code-search-repro/tiny/mod_000.py
```

Then trigger any `mcp__code-search__search_code(query="anything")` call
against the tiny project. The MCP's `auto_reindex_if_needed` fires (per
the >5min stale check), runs `incremental_index` which detects 1 file
changed, removes its old chunks, adds new chunks, and saves.

### Step 6: read the diagnostic log

```bash
grep CHUNK_ID_DIAG ~/.claude/logs/code-search-mcp.log | tail -20
```

Look for these signatures:

**Hypothesis CONFIRMED** — load saw mismatch, save dumped truncated:

```
[CHUNK_ID_DIAG] _load_index post-load: faiss.ntotal=87 chunk_ids_len=0 chunk_id_pkl_size=0
[CHUNK_ID_DIAG] save_index pre-save: in_memory_chunk_ids_len=3 faiss.ntotal=3 on_disk_pkl_size=2123
[CHUNK_ID_DIAG] save_index post-save: chunk_ids_len=3 new_pkl_size=87
```

The pre-save `on_disk_pkl_size=2123` (>0, healthy state) followed by
`in_memory_chunk_ids_len=3` (way less than the faiss.ntotal=87 we
loaded) is the smoking gun. Save then dumps the truncated 3-entry list
over the 2123-byte pkl.

**Hypothesis REFUTED** — load was healthy, save was correct:

```
[CHUNK_ID_DIAG] _load_index post-load: faiss.ntotal=87 chunk_ids_len=87 chunk_id_pkl_size=2123
[CHUNK_ID_DIAG] save_index pre-save: in_memory_chunk_ids_len=87 faiss.ntotal=87 on_disk_pkl_size=2123
[CHUNK_ID_DIAG] save_index post-save: chunk_ids_len=87 new_pkl_size=2123
```

If chunks remain healthy, the regression mechanism is something else
(concurrent ops, MCP-server-specific state corruption, pipeline_version
trigger paths, etc.) and we need a different probe.

### Step 7: inspect on-disk delta

```bash
ls -la ~/.claude_code_search/projects/tiny_*/index/
cat ~/.claude_code_search/projects/tiny_*/index/stats.json
```

If `total_chunks` dropped from ~30 to 1-3, the regression reproduced.
If it's still ~30, the regression didn't fire in this run.

## What to do with the result

- **CONFIRMED**: design fix for `_load_index`. Two candidates:
  1. Make `_chunk_ids` load eager (pickle + check) and fail loudly if
     `faiss.ntotal > 0 && chunk_ids_len == 0` — never proceed to save
     in that state.
  2. Extend `_maybe_rebuild_chunk_ids` to also fire when
     `chunk_ids_len == 0 && faiss.ntotal > 0` (currently it only fires
     when they mismatch in a specific way).
- **REFUTED**: the load-side hypothesis is wrong. Look elsewhere —
  candidates:
  - Snapshot becomes corrupted such that change-detector reports
    massive removals
  - `_remove_old_chunks(changes)` is over-aggressive on certain change
    shapes
  - A delete_project on a sibling path corrupts shared state
  - pipeline_version mismatch between binary and on-disk state forces
    full-clear-then-incremental-only-changes
- **MIXED / can't reproduce**: the field-only nature of the regression
  suggests a race or a state-shape we can't replicate in isolation.
  Move to permanent diagnostic logging in production + wait for the
  next field occurrence with full instrumentation in place.

## Cross-references

- Storage-layer evidence from 2026-05-05 in-session reproduction:
  `~/Documents/roundtable-runs/2026-05-05-codegraph-codesearch-blindspots/hypothesis_matrix.md`
- The `[CHUNK_ID_DIAG]` log additions: `search/indexer.py` `_load_index`
  and `save_index` (commits on 2026-05-05).
- The chunk-drop guard hook (already shipped, alert-only):
  `~/.claude/hooks/code-search-chunk-drop-guard.py`
