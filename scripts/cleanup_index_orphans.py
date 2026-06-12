"""Clean up orphan/inconsistent state across a project's index files.

Background: when an indexer instance has its in-memory state out of sync
with the on-disk pkl (e.g., the chunk-truncation regression observed
2026-05-04/05, or a save_index that returned early via the count-based
guard before persisting), `add_embeddings` writes to fts5.db AND
metadata.db AND stats.json for chunks that don't make it into the
canonical chunk_ids.pkl. The result:

  - fts5.db: rows whose chunk_id isn't in the pkl (BM25 dead matches)
  - metadata.db: SqliteDict keys that are orphans relative to the pkl
  - stats.json: total_chunks reflects the in-memory state at the failed
    save, NOT the actually-persisted on-disk pkl

All three are cosmetic in the sense that the search pipeline gracefully
skips chunks that don't resolve, but they:
  - inflate file sizes
  - produce dead BM25 matches in keyword search
  - mislead operators staring at stats.json or `list_projects` output

This script is the canonical admin tool for restoring consistency across
all three. It treats `chunk_ids.pkl` as the single source of truth and
brings every other artifact in line with it.

Modes:
  --dry-run                  (default) report inconsistencies; no writes
  --apply-fts5               delete orphan fts5 rows + optimize
  --apply-metadata           delete orphan metadata.db keys + commit
  --apply-stats              rewrite stats.json from authoritative state
  --apply-all                all three

  --project <prefix>         limit to one project (matches dir name prefix)

Plan-2 E2-4 (PR #123): after any --apply-* mode actually changes disk,
the script ALSO commits a fresh epoch manifest so verify_index_integrity
reports `manifest_status: fresh` instead of `stale_using_prior_epoch`
(or `corrupt`, if the prior manifest's recorded SHAs no longer match the
post-cleanup artifact bytes). Stale `manifest/candidate.json` files
(crash residue from a prior interrupted commit) are removed before the
new commit. Skipped on dry-run and on projects where no cleanup was
needed.

Designed to run on a quiesced index — close any active code-search MCP
server first, since fts5.db / metadata.db are SQLite files with WAL.
Honors CODE_SEARCH_STORAGE env var (matches the MCP server's behavior).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def _storage_dir() -> Path:
    base = os.environ.get("CODE_SEARCH_STORAGE")
    if base:
        return Path(os.path.expanduser(base)) / "projects"
    return Path.home() / ".claude_code_search" / "projects"


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


# ───────────────────────── fts5 ─────────────────────────


def find_fts5_orphans(fts_db: Path, valid_ids: set[str]) -> tuple[int, list[str]]:
    if not fts_db.exists():
        return 0, []
    con = sqlite3.connect(str(fts_db))
    try:
        cur = con.execute("SELECT chunk_id FROM chunk_fts")
        orphans = [cid for (cid,) in cur if cid not in valid_ids]
        return len(orphans), orphans[:5]
    finally:
        con.close()


def remove_fts5_orphans(fts_db: Path, valid_ids: set[str]) -> int:
    con = sqlite3.connect(str(fts_db))
    try:
        all_ids = [row[0] for row in con.execute("SELECT chunk_id FROM chunk_fts")]
        orphan_ids = [cid for cid in all_ids if cid not in valid_ids]
        if not orphan_ids:
            return 0
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
        try:
            con.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')")
            con.commit()
        except Exception as e:
            print(f"  warning: fts5 optimize failed (non-fatal): {e}", file=sys.stderr)
        return deleted
    finally:
        con.close()


# ──────────────────────── metadata ───────────────────────


def find_metadata_orphans(meta_db: Path, valid_ids: set[str]) -> tuple[int, list[str]]:
    if not meta_db.exists():
        return 0, []
    con = sqlite3.connect(str(meta_db))
    try:
        try:
            cur = con.execute("SELECT key FROM unnamed")
        except sqlite3.OperationalError:
            return 0, []
        orphans = [k for (k,) in cur if k not in valid_ids]
        return len(orphans), orphans[:5]
    finally:
        con.close()


def remove_metadata_orphans(meta_db: Path, valid_ids: set[str]) -> int:
    con = sqlite3.connect(str(meta_db))
    try:
        try:
            all_keys = [row[0] for row in con.execute("SELECT key FROM unnamed")]
        except sqlite3.OperationalError:
            return 0
        orphan_keys = [k for k in all_keys if k not in valid_ids]
        if not orphan_keys:
            return 0
        BATCH = 500
        for i in range(0, len(orphan_keys), BATCH):
            batch = orphan_keys[i : i + BATCH]
            placeholders = ",".join(["?"] * len(batch))
            con.execute(
                f"DELETE FROM unnamed WHERE key IN ({placeholders})",
                batch,
            )
        con.commit()
        return len(orphan_keys)
    finally:
        con.close()


# ───────────────────────── stats ─────────────────────────


def _path_top_folder(p: str) -> str:
    """First path segment, e.g. `skills/foo/...` -> `skills`."""
    p = p.replace("\\", "/")
    parts = p.split("/", 1)
    return parts[0] if parts and parts[0] else ""


def _detect_lang_tag(file_path: str) -> str | None:
    ext_map = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".rs": "rust",
        ".go": "go",
        ".md": "markdown",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".tf": "hcl",
        ".hcl": "hcl",
        ".nix": "nix",
        ".c": "c",
        ".h": "c",
    }
    p = file_path.lower()
    for ext, tag in ext_map.items():
        if p.endswith(ext):
            return tag
    return None


def compute_stats_from_truth(
    index_dir: Path, valid_ids: list[str]
) -> dict | None:
    """Recompute stats.json content from the authoritative pkl + metadata.db
    + code.index. Returns the stats dict or None if data is unrecoverable.
    """
    meta_db = index_dir / "metadata.db"
    code_index = index_dir / "code.index"
    if not meta_db.exists() or not code_index.exists():
        return None

    # FAISS state
    try:
        import faiss
        idx = faiss.read_index(str(code_index))
        embedding_dim = idx.d
        index_size = idx.ntotal
    except Exception:
        return None

    # Walk metadata for valid chunk_ids only.
    file_paths_seen: set[str] = set()
    chunk_types: Counter[str] = Counter()
    top_folders: Counter[str] = Counter()
    top_tags: Counter[str] = Counter()

    con = sqlite3.connect(str(meta_db))
    try:
        for cid in valid_ids:
            row = con.execute(
                "SELECT value FROM unnamed WHERE key = ?", (cid,)
            ).fetchone()
            if not row:
                continue
            try:
                outer = pickle.loads(row[0])
                meta = outer.get("metadata", outer) if isinstance(outer, dict) else {}
            except Exception:
                continue
            file_path = (
                meta.get("relative_path", "") or meta.get("file_path", "")
            )
            if file_path:
                file_paths_seen.add(file_path)
                top = _path_top_folder(file_path)
                if top:
                    top_folders[top] += 1
                tag = _detect_lang_tag(file_path)
                if tag:
                    top_tags[tag] += 1
            ct = meta.get("chunk_type", "")
            if ct:
                chunk_types[ct] += 1
    finally:
        con.close()

    # Detect index_type / quantization from FAISS object class name.
    cls_name = type(idx).__name__
    if "ScalarQuantizer" in cls_name:
        index_type = "IndexScalarQuantizer"
        quantization = "int8"
    elif "Binary" in cls_name:
        index_type = "IndexBinaryFlat"
        quantization = "binary"
    else:
        index_type = "IndexFlatIP"
        quantization = "float32"

    return {
        "total_chunks": len(valid_ids),
        "index_size": index_size,
        "embedding_dimension": embedding_dim,
        "index_type": index_type,
        "quantization": quantization,
        "files_indexed": len(file_paths_seen),
        "top_folders": dict(top_folders.most_common(10)),
        "chunk_types": dict(chunk_types),
        "top_tags": dict(top_tags.most_common(20)),
    }


def stats_drift(stats_path: Path, valid_count: int) -> int:
    """Return signed drift between stats.json total_chunks and pkl count.

    Positive = stats.json claims more chunks than pkl (the documented
    failure-mode shape: 10114 stats vs 10093 pkl).
    """
    if not stats_path.exists():
        return 0
    try:
        s = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return int(s.get("total_chunks", 0)) - valid_count


# ──────────────────── post-cleanup manifest ────────────────────
# Plan-2 E2-4 (PR #123): after cleanup mutates disk, the prior manifest's
# recorded SHAs no longer match the artifact bytes. Without a fresh
# commit, verify_index_integrity reports `manifest_corrupt` until the
# next regular save_index. This helper closes that gap by re-publishing
# a manifest covering the post-recovery state, so cleanup leaves the
# project in a consistent + verified state.


def commit_post_cleanup_manifest(
    index_dir: Path, valid_chunk_ids: list[str]
) -> str | None:
    """Build + commit a fresh epoch manifest for the post-cleanup artifacts.

    Called after any --apply-* mode that actually mutated disk. The
    manifest covers chunk_ids.pkl, code.index, metadata.db, fts5.db, and
    stats.json — whichever of those exist on disk now. Returns the
    epoch_id on success, None if no manifest could be committed (e.g.,
    no artifacts present, consistency check failed).

    Stale `candidate.json` files from a prior interrupted commit are
    removed before the new commit (cleanup_stale_candidate is idempotent).
    """
    # Lazy import: keep the module importable in test contexts that don't
    # need the manifest layer (and avoid pulling FAISS at import time).
    from search.epoch_manifest import (
        ArtifactSpec,
        ManifestConsistencyError,
        build_manifest,
        cleanup_stale_candidate,
        commit_manifest,
        count_fts5_db,
        count_metadata_db,
    )

    cleanup_stale_candidate(index_dir)

    chunk_count = len(valid_chunk_ids)
    artifacts: list[ArtifactSpec] = []

    chunk_id_path = index_dir / "chunk_ids.pkl"
    if chunk_id_path.exists():
        artifacts.append(ArtifactSpec(
            name="chunk_ids.pkl", path=chunk_id_path, count=chunk_count,
        ))

    code_index_path = index_dir / "code.index"
    if code_index_path.exists():
        # Read FAISS ntotal directly to confirm consistency at commit time.
        try:
            import faiss
            idx = faiss.read_index(str(code_index_path))
            ntotal = int(idx.ntotal)
        except Exception:
            ntotal = chunk_count  # best-effort
        artifacts.append(ArtifactSpec(
            name="code.index", path=code_index_path, count=ntotal,
        ))

    meta_path = index_dir / "metadata.db"
    if meta_path.exists():
        artifacts.append(ArtifactSpec(
            name="metadata.db", path=meta_path,
            count=count_metadata_db(meta_path),
        ))

    fts_path = index_dir / "fts5.db"
    if fts_path.exists():
        artifacts.append(ArtifactSpec(
            name="fts5.db", path=fts_path,
            count=count_fts5_db(fts_path),
        ))

    stats_path = index_dir / "stats.json"
    if stats_path.exists():
        artifacts.append(ArtifactSpec(
            name="stats.json", path=stats_path, count=None,
        ))

    if not artifacts:
        return None

    try:
        manifest = build_manifest(index_dir, artifacts)
    except ManifestConsistencyError as exc:
        print(
            f"  !! manifest commit skipped: cross-artifact consistency check "
            f"failed: {exc}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        print(
            f"  !! manifest commit skipped (unexpected error): {exc}",
            file=sys.stderr,
        )
        return None

    try:
        commit_manifest(index_dir, manifest)
    except Exception as exc:
        print(
            f"  !! manifest commit-rename failed: {exc}",
            file=sys.stderr,
        )
        return None

    return manifest.get("epoch_id")


# ───────────────────────── main ─────────────────────────


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply-fts5", action="store_true",
        help="Delete orphan fts5 rows + optimize",
    )
    parser.add_argument(
        "--apply-metadata", action="store_true",
        help="Delete orphan metadata.db keys",
    )
    parser.add_argument(
        "--apply-stats", action="store_true",
        help="Rewrite stats.json from authoritative chunk_ids.pkl + metadata.db",
    )
    parser.add_argument(
        "--apply-all", action="store_true",
        help="Apply all three cleanups (fts5 + metadata + stats)",
    )
    parser.add_argument(
        "--project",
        help="Limit to one project (matches the directory name prefix)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report inconsistencies without writing. Already the default "
             "when no --apply-* flag is given (the docstring documents this "
             "flag, so it must parse); with --apply-* flags present it wins "
             "and forces report-only.",
    )
    args = parser.parse_args()
    if args.apply_all:
        args.apply_fts5 = True
        args.apply_metadata = True
        args.apply_stats = True
    if args.dry_run and any([args.apply_fts5, args.apply_metadata, args.apply_stats]):
        print("--dry-run given — ignoring --apply-* flags (report only)")
    if args.dry_run:
        args.apply_fts5 = args.apply_metadata = args.apply_stats = False

    storage = _storage_dir()
    if not storage.is_dir():
        print(f"Storage dir not found: {storage}")
        return 2

    project_dirs = sorted(p for p in storage.iterdir() if p.is_dir())
    if args.project:
        project_dirs = [p for p in project_dirs if p.name.startswith(args.project)]
        if not project_dirs:
            print(f"No project matched prefix: {args.project}")
            return 2

    affected = 0
    totals = {
        "fts5_orphans": 0, "fts5_deleted": 0,
        "meta_orphans": 0, "meta_deleted": 0,
        "stats_drift": 0, "stats_rewritten": 0,
        "manifests_committed": 0,
    }
    for proj in project_dirs:
        idx_dir = project_index_dir(proj)
        if idx_dir is None:
            continue
        chunk_ids = load_chunk_ids(idx_dir)
        if chunk_ids is None:
            print(f"[{proj.name}] no chunk_ids.pkl — skipping")
            continue
        valid_set = set(chunk_ids)
        valid_count = len(chunk_ids)

        fts_count, fts_sample = find_fts5_orphans(idx_dir / "fts5.db", valid_set)
        meta_count, meta_sample = find_metadata_orphans(
            idx_dir / "metadata.db", valid_set
        )
        drift = stats_drift(idx_dir / "stats.json", valid_count)
        anything_off = fts_count or meta_count or drift
        if not anything_off:
            print(f"[{proj.name}] clean (valid={valid_count})")
            continue

        affected += 1
        totals["fts5_orphans"] += fts_count
        totals["meta_orphans"] += meta_count
        if drift:
            totals["stats_drift"] += abs(drift)
        print(f"[{proj.name}] valid_chunk_ids={valid_count}")
        if fts_count:
            print(f"  fts5 orphans:    {fts_count} (sample: {fts_sample[:2]})")
        if meta_count:
            print(f"  metadata orphans:{meta_count} (sample: {meta_sample[:2]})")
        if drift:
            print(
                f"  stats.json drift:{drift:+d}"
                f" (stats.total_chunks={valid_count + drift}, pkl={valid_count})"
            )

        # Track whether this project was actually mutated; gate manifest
        # commit on real disk changes (skip on dry-run).
        mutated = False
        if args.apply_fts5 and fts_count:
            n = remove_fts5_orphans(idx_dir / "fts5.db", valid_set)
            totals["fts5_deleted"] += n
            print(f"  -> fts5 deleted: {n}")
            mutated = True
        if args.apply_metadata and meta_count:
            n = remove_metadata_orphans(idx_dir / "metadata.db", valid_set)
            totals["meta_deleted"] += n
            print(f"  -> metadata deleted: {n}")
            mutated = True
        if args.apply_stats and drift:
            new_stats = compute_stats_from_truth(idx_dir, chunk_ids)
            if new_stats is None:
                print(f"  !! stats recompute failed (missing FAISS/metadata)")
            else:
                (idx_dir / "stats.json").write_text(
                    json.dumps(new_stats, indent=2),
                    encoding="utf-8",
                )
                totals["stats_rewritten"] += 1
                print(
                    f"  -> stats.json rewritten "
                    f"(total_chunks {valid_count + drift} -> {valid_count})"
                )
                mutated = True

        # Plan-2 E2-4 (PR #123): commit a fresh manifest after cleanup so
        # the post-recovery state is the published epoch. Without this,
        # the prior manifest's recorded SHAs would no longer match the
        # artifact bytes and verify_index_integrity would report
        # `manifest_corrupt` until the next regular save_index call.
        if mutated:
            epoch_id = commit_post_cleanup_manifest(idx_dir, chunk_ids)
            if epoch_id:
                totals["manifests_committed"] += 1
                print(f"  -> manifest committed: epoch_id={epoch_id}")
            else:
                print("  -> manifest commit skipped (see stderr)")

    print()
    print("=== Summary ===")
    print(f"  Projects with inconsistencies: {affected}")
    if any([args.apply_fts5, args.apply_metadata, args.apply_stats]):
        print(
            f"  fts5 orphans deleted:      {totals['fts5_deleted']}/"
            f"{totals['fts5_orphans']}"
        )
        print(
            f"  metadata orphans deleted:  {totals['meta_deleted']}/"
            f"{totals['meta_orphans']}"
        )
        print(
            f"  stats.json files rewritten:{totals['stats_rewritten']}"
        )
        print(
            f"  manifests committed:       {totals['manifests_committed']}"
        )
    else:
        print(
            f"  fts5 orphans:     {totals['fts5_orphans']}"
        )
        print(
            f"  metadata orphans: {totals['meta_orphans']}"
        )
        print(
            f"  stats drift sum:  {totals['stats_drift']}"
        )
        print("  (dry-run — pass --apply-all or --apply-{fts5,metadata,stats})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
