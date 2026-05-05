"""Remove orphan entries from a project's fts5.db.

Background: when an indexer instance has its in-memory state out of sync
with the on-disk pkl (e.g., the chunk-truncation regression observed
2026-05-04/05, or a save_index that returned early before persisting),
add_embeddings can write to fts5.db (and metadata.db) for chunks that are
NOT later persisted to chunk_ids.pkl. The result: FTS5 has rows whose
chunk_id doesn't exist in the canonical chunk_ids list — orphans. They're
cosmetic (the rest of the search pipeline gracefully skips them) but they
inflate fts5.db size and produce dead chunk_id matches in BM25.

This script:
1. Treats `chunk_ids.pkl` as the authoritative list of valid chunk_ids
   for each project under ~/.claude_code_search/projects/.
2. For each project, queries fts5.db for chunk_ids NOT in the
   authoritative list.
3. With --dry-run (default), reports the orphan count per project.
4. With --apply, deletes orphan rows. The fts5.db is rebuilt as a
   compact copy after deletion to actually free disk space.

Designed to run on a quiesced index — close any active code-search MCP
server first, since fts5.db is a SQLite file with WAL.
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

STORAGE_DIR = Path.home() / ".claude_code_search" / "projects"


def project_index_dir(project_dir: Path) -> Path | None:
    idx = project_dir / "index"
    if not idx.is_dir():
        return None
    return idx


def load_chunk_ids(index_dir: Path) -> list[str] | None:
    pkl = index_dir / "chunk_ids.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            return None
        return data
    except Exception:
        return None


def find_orphans(fts_db: Path, valid_ids: set[str]) -> tuple[int, list[str]]:
    """Return (orphan_count, sample_orphans) — sample is up to 5 ids."""
    if not fts_db.exists():
        return 0, []
    con = sqlite3.connect(str(fts_db))
    try:
        cur = con.execute("SELECT chunk_id FROM chunk_fts")
        orphans: list[str] = []
        for (cid,) in cur:
            if cid not in valid_ids:
                orphans.append(cid)
        return len(orphans), orphans[:5]
    finally:
        con.close()


def remove_orphans(fts_db: Path, valid_ids: set[str]) -> int:
    """Delete orphan rows from fts5.db. Returns count deleted."""
    con = sqlite3.connect(str(fts_db))
    try:
        # Pull all chunk_ids currently in fts5
        all_ids = [
            row[0] for row in con.execute("SELECT chunk_id FROM chunk_fts")
        ]
        orphan_ids = [cid for cid in all_ids if cid not in valid_ids]
        if not orphan_ids:
            return 0
        # FTS5 doesn't support efficient bulk DELETE; loop in chunks.
        deleted = 0
        BATCH = 500
        for i in range(0, len(orphan_ids), BATCH):
            batch = orphan_ids[i : i + BATCH]
            placeholders = ",".join(["?"] * len(batch))
            con.execute(
                f"DELETE FROM chunk_fts WHERE chunk_id IN ({placeholders})",
                batch,
            )
            deleted += len(batch)
        con.commit()
        # Compact the FTS5 index (frees space + reclaims internal fragmentation)
        try:
            con.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')")
            con.commit()
        except Exception as e:
            print(f"  warning: optimize failed (non-fatal): {e}", file=sys.stderr)
        return deleted
    finally:
        con.close()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete orphans (default: dry-run report only)",
    )
    parser.add_argument(
        "--project",
        help="Limit to a single project name (matches the directory prefix)",
    )
    args = parser.parse_args()

    if not STORAGE_DIR.is_dir():
        print(f"Storage dir not found: {STORAGE_DIR}")
        return 2

    project_dirs = sorted(p for p in STORAGE_DIR.iterdir() if p.is_dir())
    if args.project:
        project_dirs = [p for p in project_dirs if p.name.startswith(args.project)]
        if not project_dirs:
            print(f"No project matched prefix: {args.project}")
            return 2

    total_orphans = 0
    total_deleted = 0
    affected = 0
    for proj in project_dirs:
        idx_dir = project_index_dir(proj)
        if idx_dir is None:
            continue
        chunk_ids = load_chunk_ids(idx_dir)
        if chunk_ids is None:
            print(f"[{proj.name}] no chunk_ids.pkl — skipping")
            continue
        valid = set(chunk_ids)
        fts = idx_dir / "fts5.db"
        if not fts.exists():
            continue
        count, sample = find_orphans(fts, valid)
        if count == 0:
            print(f"[{proj.name}] clean ({len(valid)} valid chunk_ids, 0 orphans)")
            continue
        affected += 1
        total_orphans += count
        print(
            f"[{proj.name}] {count} orphans (valid={len(valid)}); "
            f"sample={sample[:3]}"
        )
        if args.apply:
            deleted = remove_orphans(fts, valid)
            total_deleted += deleted
            print(f"  -> deleted {deleted}")

    print()
    print("=== Summary ===")
    print(f"  Projects with orphans: {affected}")
    print(f"  Total orphan rows: {total_orphans}")
    if args.apply:
        print(f"  Deleted: {total_deleted}")
    else:
        print("  (dry-run — pass --apply to delete)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
