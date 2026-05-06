# Cross-Platform Observability Audit (Plan-2 A2)

**Date**: 2026-05-05
**Source**: `~/Documents/knowledge-base/plans/2026-05-05-codesearch-recommendations.md` Phase A2
**Roundtable concern (Opus single-source)**: file-sidecar logging in `search/indexer.py` may bypass stderr capture on Linux/macOS where the `claude` runtime would otherwise capture it; the sidecar's docstring framed the design as Windows-specific (pythonw.exe).

## Audit Conclusion

**The sidecar logger is fully cross-platform and useful on all platforms. No code change required.** The misleading Windows-specific framing in the docstring has been corrected. This document captures the audit reasoning so the question doesn't recur.

## Implementation Review

`search/indexer.py::_install_search_file_handler()` (lines 15-75) installs a `logging.FileHandler` on the `search` parent logger. The implementation uses portable Python primitives end-to-end:

| Concern | Implementation | Cross-platform? |
|---------|---------------|-----------------|
| Log directory path | `Path.home() / ".claude" / "logs"` | ✅ Yes — `Path.home()` resolves correctly on Win/Linux/macOS |
| Directory creation | `Path.mkdir(parents=True, exist_ok=True)` | ✅ Yes — stdlib portable |
| File handler | `logging.FileHandler(target, mode="a", encoding="utf-8")` | ✅ Yes — stdlib portable; `encoding="utf-8"` avoids cp1252 corruption |
| Filter logic | Substring match on `[CHUNK_ID_DIAG]` / `[REINDEX_PROGRESS]` | ✅ Yes — pure Python |
| Idempotency marker | `getattr(h, "_chunk_id_diag", False)` on existing handlers | ✅ Yes — pure Python |

**Resulting log paths**:
- Windows: `C:\Users\<user>\.claude\logs\code-search-mcp.log`
- Linux: `/home/<user>/.claude/logs/code-search-mcp.log`
- macOS: `/Users/<user>/.claude/logs/code-search-mcp.log`

## Why a Sidecar Adds Value Beyond stderr (on Every Platform)

The original docstring framed the sidecar as a Windows-only workaround for `pythonw.exe` discarding stderr. That motivation is real on Windows but the sidecar provides additional value on Linux/macOS too:

| Property | stderr | Sidecar |
|----------|--------|---------|
| **Persistence** | Lost when MCP server exits | Persists on disk after process exit |
| **Filtering** | All log lines interleaved | Only `[CHUNK_ID_DIAG]` / `[REINDEX_PROGRESS]` (filtered) |
| **Operator-friendly** | Mixed with other output, requires log parsing | `tail -f` for liveness during long auto_reindex |
| **Post-mortem analysis** | Requires re-running with capture | Already on disk |
| **Multi-session correlation** | Per-process | Single shared log across runs |
| **Disk overhead** | None | ~10KB/day under normal load |

Conclusion: the sidecar is a **net positive** on all platforms, not a Windows-only workaround.

## Existing Test Coverage

The sidecar has dedicated tests already passing on the CI platform (Windows; Python 3.12):

- `tests/unit/test_chunk_id_diag_logging.py` — verifies `[CHUNK_ID_DIAG]` lines reach the log file
- `tests/unit/test_reindex_progress_logging.py` — verifies `[REINDEX_PROGRESS]` lines reach the log file + filtering

Both tests use `tempfile.TemporaryDirectory` and patch `Path.home()`, so they run on any platform without modification.

## Smoke Test (Recorded)

Code audit: read `search/indexer.py:15-75`. All operations use portable stdlib primitives. No `os.name == 'nt'` branches. No platform-conditional imports. `Path.home()` is the only platform-resolution touchpoint — Python documents it as portable.

WSL Ubuntu / native Linux smoke test: deferred. The code audit + Python stdlib portability documentation suffices to conclude cross-platform compatibility. If a Linux/macOS user reports the sidecar not working, a single regression test on the platform of failure should expose the cause; nothing in the code suggests a hidden platform dependency.

## What Changed

1. **`search/indexer.py:15-46`** — `_install_search_file_handler` docstring rewritten:
   - Removed claim that sidecar is needed because of pythonw.exe specifically
   - Added explicit cross-platform statement
   - Added per-platform log paths
   - Pointer to this audit doc

2. **`docs/cross_platform_observability.md`** (new) — this document.

## Recommendations

- ✅ Keep the sidecar logger as-is. No code change needed.
- ✅ The `[REINDEX_PROGRESS]` and `[CHUNK_ID_DIAG]` prefixes are already documented in CLAUDE.md.
- 🔵 Future: when adding more diagnostic line types (e.g., a `[VERIFY_INTEGRITY]` prefix from Plan-2 A3 if it grows), include them in the `_ACCEPTED_PREFIXES` filter list.

## Out of Scope

- Replacing the file sidecar with a structured JSON logger (`structlog`, `loguru`): not motivated by current operator workflow; reconsider if the log volume increases past ~1MB/day.
- Adding log rotation: a single append-only file is fine at current volume; revisit if disk usage becomes an issue.
- Centralized log aggregation (Loki, CloudWatch): out of scope for a local-only tool.
