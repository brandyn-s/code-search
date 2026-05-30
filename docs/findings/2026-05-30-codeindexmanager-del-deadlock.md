# CodeIndexManager.__del__ → SqliteDict.close() can deadlock (latent server-hang)

**Date**: 2026-05-30
**Found by**: the new keyless test battery (`tests/unit/test_battery.py`)
**Status**: mechanism MEASURED (repro + stack); production impact BLOCKED ON MEASUREMENT.
**Severity**: MED-HIGH — a GC-triggered `__del__` can hang the calling thread (and
thus the long-lived MCP server) on project switch.

## What

`CodeIndexManager.__del__` (`search/indexer.py:1185-1190`) unconditionally calls
`self._metadata_db.close()` (a `SqliteDict`). `SqliteDict.close()` can block
forever on its background writer thread's queue. Because `__del__` runs during
garbage collection / interpreter shutdown, the **calling thread** (not the
daemon writer) is the one that deadlocks.

## Repro (this session)

A minimal driver — full index → edit a file → incremental re-index — completes
the indexing correctly (`chunks_added=1, chunks_removed=1`, prints `DONE`), then
hangs at process teardown. `faulthandler` stack at the hang:

```
File ".../threading.py", line 327 in wait
File ".../queue.py", line 171 in get
File ".../sqlitedict.py", line 653 in select
File ".../sqlitedict.py", line 662 in select_one
File ".../sqlitedict.py", line 689 in close
File ".../sqlitedict.py", line 391 in close
File "search/indexer.py", line 1188 in __del__   <-- here
```

The SqliteDict writer thread is created `daemon=True` (`sqlitedict.py:442`), so
leaking it at exit is harmless — but `close()` blocking the **caller** is not.

## Trigger conditions

- A **full-only** index (the determinism battery test) closes cleanly — no hang.
- The **incremental** path (which does more `_metadata_db` writes: remove + add
  before close) reliably triggers it. So it correlates with the metadata-DB
  write/flush state at close time.

## Production risk

The MCP server is normally long-lived with one index manager, which masks this.
But **project switching** (`switch_project`) constructs a new `CodeIndexManager`
and drops the old one; when the old one is GC'd, `__del__` → `close()` can
deadlock the server thread. That path is the one to measure.

## Recommended fix (not applied here — needs maintainer eval of the resource path)

Make the close path idempotent and non-deadlocking. Options, cheapest first:

1. **Guard double-close** with a `_closed` flag; only the first `close()` runs.
2. **Don't block in `__del__`** — the SqliteDict thread is daemon, so the safe
   pattern is: explicit `close()` on the lifecycle path (project switch / clear)
   with a bounded watchdog, and a `__del__` that no-ops if not already closed
   (or closes in a short-timeout daemon thread).
3. Audit `clear_index` / re-init for **orphaned SqliteDicts** (a new
   `_metadata_db` assigned without closing the old) whose later GC repeats this.

## Regression test

`tests/unit/test_battery.py::test_battery_incremental_equals_full` is written and
correct (it asserts incremental==full convergence) but is **skipped** pending
this fix, because its teardown deadlocks the process. Unskip it when the close
path is made non-deadlocking — it then doubles as the regression test for this
bug. `test_battery_index_determinism` (full-only) runs and passes today.
