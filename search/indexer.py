"""Vector index management with FAISS and metadata storage."""

import fnmatch
import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from search.metadata_store import JsonSqliteKV, LegacyMetadataFormatError
from embeddings.embedder import EmbeddingResult


def _install_search_file_handler() -> None:
    """Attach a FileHandler that captures structured diagnostic lines to disk.

    Cross-platform sidecar logger (Plan-2 A2 audit, 2026-05-05): works on
    Windows, Linux, and macOS via portable `Path.home()` resolution. See
    docs/cross_platform_observability.md for the full audit.

    Why a sidecar (not just stderr):
      Windows: the MCP server runs under pythonw.exe with no console; stderr
        is discarded entirely.
      Linux/macOS: stderr IS captured by the parent (Claude Code or another
        MCP transport), but it's interleaved with the rest of the process
        output and is ephemeral. The sidecar gives operators a persistent,
        filtered, `tail -f`-able log file regardless of platform.

    Captured prefixes (filter accepts any line containing one of these):
      [CHUNK_ID_DIAG]      — load/save state diagnostics in indexer
      [REINDEX_PROGRESS]   — incremental_index progress milestones
      [ANTHROPIC_DIAG]     — per-call Sonnet rerank latency (Plan D1-Pass-2 A.1)

    Attaches to the `search` parent logger so children
    (`search.indexer`, `search.incremental_indexer`) propagate up and
    share the handler. Idempotent: a marker attribute on the handler
    prevents stacking on re-import.

    Output (all platforms): ~/.claude/logs/code-search-mcp.log
      Windows: C:\\Users\\<user>\\.claude\\logs\\code-search-mcp.log
      Linux:   /home/<user>/.claude/logs/code-search-mcp.log
      macOS:   /Users/<user>/.claude/logs/code-search-mcp.log

    Degrades silently if the log directory cannot be created (we never
    want logging setup to break the indexer).
    """
    try:
        log_dir = Path.home() / ".claude" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "code-search-mcp.log"
    except Exception:
        return

    # Attach to the package logger so both indexer and incremental_indexer
    # log lines flow to the same sidecar.
    logger = logging.getLogger("search")
    target = str(log_path.resolve())
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "_chunk_id_diag", False):
            return
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target:
            return

    try:
        handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    except Exception:
        return
    handler._chunk_id_diag = True  # type: ignore[attr-defined]
    handler.setLevel(logging.DEBUG)

    _ACCEPTED_PREFIXES = (
        "[CHUNK_ID_DIAG]",
        "[REINDEX_PROGRESS]",
        "[ANTHROPIC_DIAG]",
        # Phase A1 (2026-05-10): per-cohort override-trigger records,
        # emitted by _effective_threshold in sonnet_reranker.py when
        # SONNET_RERANKER_LOG_OVERRIDE_TRIGGERS=1. Used by
        # paired_bootstrap_per_subproject.py to count spillover.
        "[PATH_OVERRIDE_TRIGGER]",
        # Phase A1 (2026-05-11): per-cohort reranker outcome records,
        # emitted by _rerank_async in sonnet_reranker.py for non-OK
        # outcomes (hybrid_prior_fallback, timeout, too_many_failures).
        # Promoted from LOG.debug to LOG.info to close the silent-fallback
        # observability gap surfaced in the 2026-05-10 Phase B audit.
        "[RERANK_REASON]",
        # Arc A (2026-05-11): per-search PPR diagnostic emitted by
        # search/ppr_scorer.py — db-not-found / insufficient-subgraph /
        # computed t_ms records. Same observability pattern as
        # RERANK_REASON / ANTHROPIC_DIAG.
        "[PPR_DIAG]",
    )

    class _SearchDiagFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return False
            return any(p in msg for p in _ACCEPTED_PREFIXES)

    handler.addFilter(_SearchDiagFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.WARNING:
        logger.setLevel(logging.WARNING)

    # [ANTHROPIC_DIAG] is INFO-level (per Plan D1-Pass-2 A.1). The parent
    # `search` logger above stays at WARNING so unrelated INFO chatter
    # doesn't fill the sidecar; we elevate ONLY the reranker child logger
    # to INFO so its [ANTHROPIC_DIAG] records can reach the sidecar handler.
    # The filter still gates on the prefix, so other reranker INFO logs
    # (judge prompts, etc.) are dropped.
    sonnet_logger = logging.getLogger("search.sonnet_reranker")
    if sonnet_logger.level == logging.NOTSET or sonnet_logger.level > logging.INFO:
        sonnet_logger.setLevel(logging.INFO)


# Public alias for backward compat with tests that imported the v1 name.
_install_chunk_id_diag_file_handler = _install_search_file_handler

_install_search_file_handler()


class CodeIndexManager:
    """Manages FAISS vector index and metadata storage for code chunks."""

    # P5 (2026-06-10 roadmap): stale-vector compaction thresholds. FAISS rows
    # are never removed in place (removal is "rebuild on demand"), so
    # modify/delete churn accumulates stale vectors. ADVISORY surfaces in
    # search `_metadata` and verify_index_integrity; COMPACTION escalates an
    # incremental index run to a full reindex (which clears the index and
    # resets the ratio to 0 — self-limiting). Hard-coded, not env knobs.
    STALE_ADVISORY_RATIO = 0.25
    STALE_COMPACTION_RATIO = 0.5

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.storage_dir / "code.index"
        self.metadata_path = self.storage_dir / "metadata.db" 
        self.chunk_id_path = self.storage_dir / "chunk_ids.pkl"
        self.stats_path = self.storage_dir / "stats.json"
        
        # Initialize components
        self._index = None
        self._metadata_db = None
        self._chunk_ids = []
        self._logger = logging.getLogger(__name__)
        self._on_gpu = False

        # Observability: status of the most recent _commit_epoch_manifest call.
        # Stable string vocabulary: "ok", "skipped_empty", "consistency_error",
        # "build_error", "commit_error", or None (no commit attempted yet).
        # Callers (incremental_indexer, MCP layer) can surface this in
        # `_metadata` to distinguish silent-success-with-stale-manifest from
        # true success.
        self.last_manifest_commit_status: Optional[str] = None

        # Initialize FTS5
        self._init_fts5()
        
    def _init_fts5(self):
        """Initialize FTS5 full-text search table.

        Corruption-hardened (2026-06-10 torn-write fuzz): a truncated or
        garbage fts5.db raised sqlite3.DatabaseError HERE — in the
        constructor — making the manager unconstructable until manual
        cleanup. FTS5 is derived data (rebuilt by reindex), so corruption
        quarantines the bad file and recreates an empty table: BM25 leg
        degrades until the next reindex, search keeps working.
        """
        import sqlite3
        self._fts_db_path = self.storage_dir / "fts5.db"
        for attempt in (1, 2):
            try:
                self._fts_conn = sqlite3.connect(
                    str(self._fts_db_path),
                    check_same_thread=False,
                )
                self._fts_conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                        chunk_id,
                        content,
                        file_path,
                        name,
                        tokenize='porter unicode61'
                    )
                """)
                self._fts_conn.commit()
                return
            except sqlite3.DatabaseError as e:
                try:
                    self._fts_conn.close()
                except Exception:
                    pass
                self._fts_conn = None
                if attempt == 2:
                    self._logger.error(
                        "fts5.db unusable after quarantine (%s); BM25 leg "
                        "disabled until reindex", e,
                    )
                    return
                import time
                quarantine = self._fts_db_path.with_suffix(
                    f".db.corrupt.{time.strftime('%Y%m%dT%H%M%S')}"
                )
                try:
                    self._fts_db_path.rename(quarantine)
                except OSError:
                    try:
                        self._fts_db_path.unlink()
                    except OSError:
                        return
                self._logger.error(
                    "fts5.db is corrupt (%s) — quarantined to %s and "
                    "recreated empty. BM25 results degrade until a full "
                    "reindex (index_directory(incremental=false)).",
                    e, quarantine.name,
                )

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize a natural-language query for FTS5 MATCH syntax.

        Strips FTS5 operators and special chars, quotes each token,
        and joins with OR so any keyword match counts.
        """
        import re
        # Remove characters that are FTS5 operators or cause syntax errors.
        # C0 control chars (esp. NUL) are included: a NUL inside a quoted
        # token terminates the SQL string early and raises "unterminated
        # string", which silently emptied the BM25 leg for that query
        # (found by fuzzing, 2026-06-10).
        cleaned = re.sub(r'[?"*/\\(){}^~:+\-\x00-\x1f]', ' ', query)
        tokens = [t for t in cleaned.split() if t and len(t) > 1]
        if not tokens:
            return ""
        # Quote each token to prevent column-name interpretation
        return " OR ".join(f'"{t}"' for t in tokens)

    def search_bm25(
        self,
        query: str,
        k: int = 50,
        name_weight: float = 5.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using BM25 full-text search. Returns (chunk_id, rank, metadata).

        When `filters` is provided, applies the same `_matches_filters`
        check used by the vector search path. Without this, the BM25 half
        of hybrid search returned chunks the user explicitly filtered out
        via `file_pattern` / `chunk_type` (regression fixed 2026-05-07).

        Over-fetches by 3x when filters are present so a useful k is
        retained after filter rejection. Caps at the original `k` after
        filtering.
        """
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            return []

        fts_query = self._sanitize_fts5_query(query)
        if not fts_query:
            return []

        # Over-fetch when filters will reduce the result set
        fetch_k = k * 3 if filters else k

        try:
            cursor = self._fts_conn.execute(
                f"SELECT chunk_id, bm25(chunk_fts, 0.0, 1.0, 0.5, {float(name_weight)}) as rank "
                "FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, fetch_k),
            )
            results = []
            seen = set()
            for chunk_id, rank in cursor.fetchall():
                # Dedupe: legacy indexes built before FTS rows were cleaned
                # on remove/re-add can hold the same chunk_id several times.
                # Rows arrive best-rank-first, so keep the first occurrence.
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                metadata_entry = self.metadata_db.get(chunk_id)
                if not metadata_entry:
                    continue
                metadata = metadata_entry["metadata"]
                if filters and not self._matches_filters(metadata, filters):
                    continue
                results.append((chunk_id, float(rank), metadata))
                if len(results) >= k:
                    break
            return results
        except Exception as e:
            self._logger.warning(f"FTS5 search failed for '{fts_query}': {e}")
            return []

    @property
    def index(self):
        """Lazy loading of FAISS index."""
        if self._index is None:
            self._load_index()
        return self._index
    
    @property
    def metadata_db(self):
        """Lazy loading of metadata database.

        Metadata is NOT recoverable from the other artifacts, so a corrupt
        metadata.db raises an ACTIONABLE error instead of a raw storage-layer
        traceback (2026-06-10 torn-write fuzz).
        """
        if self._metadata_db is None:
            try:
                self._metadata_db = JsonSqliteKV(str(self.metadata_path))
            except LegacyMetadataFormatError:
                # Pre-2026-06-11 sqlitedict format: actionable on its own.
                raise
            except Exception as e:
                raise RuntimeError(
                    f"metadata.db at {self.metadata_path} is corrupt or "
                    f"unreadable ({type(e).__name__}: {e}). Metadata cannot "
                    "be rebuilt from other artifacts — run "
                    "index_directory(incremental=false) to reindex this "
                    "project."
                ) from e
        return self._metadata_db
    
    def _load_index(self):
        """Load existing FAISS index or create new one."""
        self._is_binary = False
        float_store_path = self.storage_dir / "float_store.npy"

        # CHUNK_ID DIAGNOSTIC (2026-05-05): tracks the load-side state so we
        # can spot when chunk_ids.pkl is empty or out-of-sync with FAISS.
        # The hypothesis under investigation: post-MCP-restart, the lazy
        # load sees an empty/short chunk_ids.pkl, then a subsequent
        # incremental save dumps the truncated list, overwriting prior
        # healthy state. Logging at every load + save lets us catch the
        # transition.
        self._logger.warning(
            "[CHUNK_ID_DIAG] _load_index pre-load: index_path=%s exists=%s "
            "chunk_id_path=%s exists=%s",
            self.index_path, self.index_path.exists(),
            self.chunk_id_path, self.chunk_id_path.exists(),
        )

        if self.index_path.exists():
            # Corruption-hardened (2026-06-10 torn-write fuzz): a truncated/
            # garbage code.index raised a raw faiss RuntimeError from every
            # search. The FAISS index is rebuilt by reindex, so corruption
            # degrades to vector-leg-disabled (BM25 keeps working) with a
            # loud actionable log instead of crashing the read path.
            try:
                # Detect binary mode: float_store.npy exists alongside the index
                if float_store_path.exists():
                    self._logger.info(f"Loading binary index from {self.index_path}")
                    self._index = faiss.read_index_binary(str(self.index_path))
                    self._float_store = np.load(str(float_store_path))
                    self._is_binary = True
                else:
                    self._logger.info(f"Loading index from {self.index_path}")
                    self._index = faiss.read_index(str(self.index_path))
                    if not self._is_binary:
                        self._maybe_move_index_to_gpu()
            except Exception as e:
                self._logger.error(
                    "FAISS index at %s is corrupt or unreadable (%s: %s). "
                    "Vector search disabled (BM25 still serves) until a "
                    "full reindex (index_directory(incremental=false)).",
                    self.index_path, type(e).__name__, str(e)[:200],
                )
                self._index = None
                self._chunk_ids = []
                self._is_binary = False
                return

            # Load chunk IDs. A corrupt pickle is recoverable: fall through
            # with an empty list and let _maybe_rebuild_chunk_ids reconstruct
            # the FAISS-position mapping losslessly from metadata.db
            # (pre-fix this raised UnpicklingError from every search even
            # though the rebuild machinery existed one call away).
            if self.chunk_id_path.exists():
                try:
                    with open(self.chunk_id_path, 'rb') as f:
                        loaded = pickle.load(f)
                    self._chunk_ids = loaded if isinstance(loaded, list) else []
                except Exception as e:
                    self._logger.error(
                        "chunk_ids.pkl is corrupt (%s: %s) — attempting "
                        "rebuild from metadata.db",
                        type(e).__name__, str(e)[:120],
                    )
                    self._chunk_ids = []

            # CHUNK_ID DIAGNOSTIC: log the state right after load.
            self._logger.warning(
                "[CHUNK_ID_DIAG] _load_index post-load: faiss.ntotal=%s "
                "chunk_ids_len=%s chunk_id_pkl_size=%s",
                self._index.ntotal if self._index else None,
                len(self._chunk_ids),
                self.chunk_id_path.stat().st_size if self.chunk_id_path.exists() else 0,
            )

            # Detect and repair chunk_ids.pkl corruption: if FAISS has vectors
            # but chunk_ids is missing/empty/shorter than expected, rebuild
            # from metadata.db. Each metadata value is a dict with 'index_id'
            # giving its FAISS position; we reconstruct the ordered list.
            self._maybe_rebuild_chunk_ids()

            # CHUNK_ID DIAGNOSTIC: log post-repair state, in case rebuild
            # fired and changed chunk_ids_len.
            self._logger.warning(
                "[CHUNK_ID_DIAG] _load_index post-repair: faiss.ntotal=%s "
                "chunk_ids_len=%s",
                self._index.ntotal if self._index else None,
                len(self._chunk_ids),
            )
        else:
            self._logger.warning(
                "[CHUNK_ID_DIAG] _load_index: no existing index, starting fresh"
            )
            self._index = None
            self._chunk_ids = []

    def _maybe_rebuild_chunk_ids(self):
        """Rebuild chunk_ids.pkl from metadata.db if it's missing or out of sync.

        Guards against the failure mode where chunk_ids.pkl gets truncated to
        an empty list (5 bytes: empty pickle) by a failed load path, causing
        every subsequent search to raise `list index out of range`. The FAISS
        index and metadata database are still intact; only the parallel
        chunk-id list is lost. Recovery is lossless as long as metadata.db
        still holds an `index_id` for every row.
        """
        if self._index is None:
            return
        faiss_n = self._index.ntotal
        chunk_n = len(self._chunk_ids)
        if faiss_n == 0:
            return
        if chunk_n == faiss_n:
            return
        if not self.metadata_path.exists():
            self._logger.warning(
                "chunk_ids mismatch (faiss=%d, chunk_ids=%d) but metadata.db missing — "
                "cannot auto-rebuild; reindex required",
                faiss_n, chunk_n,
            )
            return
        self._logger.warning(
            "chunk_ids out of sync with FAISS (faiss=%d, chunk_ids=%d) — rebuilding from metadata.db",
            faiss_n, chunk_n,
        )
        rebuilt = [None] * faiss_n
        filled = 0
        for chunk_id, entry in self.metadata_db.items():
            idx = entry.get("index_id") if isinstance(entry, dict) else None
            if not isinstance(idx, int) or idx < 0 or idx >= faiss_n:
                continue
            if rebuilt[idx] is None:
                rebuilt[idx] = chunk_id
                filled += 1
        missing = faiss_n - filled
        if missing > 0:
            self._logger.error(
                "chunk_ids rebuild incomplete: %d of %d slots still missing — reindex recommended",
                missing, faiss_n,
            )
            # Leave self._chunk_ids as-is rather than shipping a half-rebuilt list
            # that would mismatch FAISS positions.
            return
        # Back up the corrupted pkl (if present) before overwriting.
        if self.chunk_id_path.exists() and chunk_n != faiss_n:
            import time
            bak = self.chunk_id_path.with_suffix(
                f".pkl.bak.{time.strftime('%Y%m%dT%H%M%S')}"
            )
            try:
                bak.write_bytes(self.chunk_id_path.read_bytes())
            except OSError as exc:
                self._logger.warning("could not back up corrupted chunk_ids.pkl: %s", exc)
        self._chunk_ids = rebuilt
        with open(self.chunk_id_path, "wb") as f:
            pickle.dump(self._chunk_ids, f)
        self._logger.info("chunk_ids rebuilt and persisted (%d entries)", faiss_n)
    
    def create_index(self, embedding_dimension: int, index_type: str = "flat"):
        """Create a new FAISS index.

        Quantization controlled by QUANTIZATION env var:
        - "int8" (default): ScalarQuantizer with QT_8bit_direct — 4x smaller, <0.1% quality loss
        - "float32": IndexFlatIP — original full-precision
        - "binary": IndexBinaryFlat + float store — 32x smaller, needs rescore (opt-in for 100K+ chunks)
        """
        quantization = os.environ.get("QUANTIZATION", "int8").lower()
        self._is_binary = False

        if quantization == "binary":
            self._index = faiss.IndexBinaryFlat(embedding_dimension)
            self._float_store = np.empty((0, embedding_dimension), dtype=np.float32)
            self._is_binary = True
            self._logger.info(f"Created binary index with dimension {embedding_dimension} (32x compression, requires rescore)")
        elif quantization == "int8" and index_type == "flat":
            # QT_8bit (trained) learns the value range from data, then linearly maps to [0,255].
            # QT_8bit_direct was wrong — it interprets float bytes as raw ints, producing
            # all-zero similarities on normalized [-1,1] vectors. (Confirmed 2026-04-05:
            # isolated FAISS test showed QT_8bit_direct returns 0.0 for all queries.)
            self._index = faiss.IndexScalarQuantizer(
                embedding_dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
            )
            self._logger.info(f"Created int8 quantized index with dimension {embedding_dimension} (4x compression, requires training)")
        elif index_type == "flat":
            self._index = faiss.IndexFlatIP(embedding_dimension)
            self._logger.info(f"Created float32 flat index with dimension {embedding_dimension}")
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dimension)
            n_centroids = min(100, max(10, embedding_dimension // 8))
            self._index = faiss.IndexIVFFlat(quantizer, embedding_dimension, n_centroids)
            self._logger.info(f"Created IVF index with dimension {embedding_dimension}")
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        if not self._is_binary:
            self._maybe_move_index_to_gpu()
    
    def add_embeddings(self, embedding_results: List[EmbeddingResult]) -> None:
        """Add embeddings to the index and metadata to the database."""
        if not embedding_results:
            return

        # Load existing on-disk index BEFORE deciding to create a new one.
        # Without this, a fresh CodeIndexManager (e.g., after switch_project)
        # whose `_index` is still None will fall through to create_index()
        # and start an empty FAISS while the on-disk index already holds
        # 30+ vectors. The next save_index then dumps that empty in-memory
        # state over the healthy on-disk pkl/index — the chunk-truncation
        # regression observed 2026-05-04/05.
        if self._index is None and self.index_path.exists():
            self._load_index()

        # Initialize index if needed
        if self._index is None:
            embedding_dim = embedding_results[0].embedding.shape[0]
            # Always use flat index - IVF breaks reconstruct() needed by get_similar_chunks
            # Flat handles 20K+ vectors fine for our use case
            index_type = "flat"
            self.create_index(embedding_dim, index_type)
        
        # Prepare embeddings and metadata
        embeddings = np.array([result.embedding for result in embedding_results])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Train quantized/IVF index if needed
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            # The quantizer trains ONCE, on whatever the first batch is.
            # Full reindexes pass all chunks in one add_embeddings call, so
            # training data is representative. An index born from a small
            # incremental batch learns its value range from few vectors —
            # later additions outside that range clip. Warn so operators
            # know a full reindex would improve int8 fidelity.
            if len(embeddings) < 256:
                self._logger.warning(
                    "Training quantizer on only %d vectors; value ranges may "
                    "be unrepresentative. A full reindex (force=true) trains "
                    "on the complete corpus.",
                    len(embeddings),
                )
            self._logger.info("Training index...")
            self._index.train(embeddings)

        # Add to FAISS index (binary mode packs bits separately)
        if getattr(self, '_is_binary', False):
            # Binary mode: pack sign bits and store float originals for rescoring
            codes = np.packbits((embeddings > 0).astype(np.uint8), axis=1)
            self._index.add(codes)
            self._float_store = np.concatenate([self._float_store, embeddings], axis=0)
        else:
            self._index.add(embeddings)
        start_id = len(self._chunk_ids)
        
        # Store metadata and update chunk IDs
        for i, result in enumerate(embedding_results):
            chunk_id = result.chunk_id
            self._chunk_ids.append(chunk_id)
            
            # Store in metadata database
            self.metadata_db[chunk_id] = {
                'index_id': start_id + i,
                'metadata': result.metadata
            }
        
        self._logger.info(f"Added {len(embedding_results)} embeddings to index")
        
        # Commit metadata in a single transaction for performance
        try:
            self.metadata_db.commit()
        except Exception:
            # If commit is unavailable for some reason, continue without failing
            pass

        # Add to FTS5 index (re-init if connection was lost)
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            self._init_fts5()
        # Idempotency: drop any existing FTS rows for the incoming chunk_ids
        # first. chunk_fts has no uniqueness constraint, so re-adding a
        # chunk_id (modified file whose chunk boundaries didn't move) would
        # otherwise duplicate it in BM25 results.
        incoming_ids = [r.chunk_id for r in embedding_results]
        try:
            for i in range(0, len(incoming_ids), 500):
                batch = incoming_ids[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                self._fts_conn.execute(
                    f"DELETE FROM chunk_fts WHERE chunk_id IN ({placeholders})",
                    batch,
                )
        except Exception as e:
            self._logger.warning(f"FTS5 pre-insert cleanup failed: {e}")
        for result in embedding_results:
            content = result.metadata.get("full_content", result.metadata.get("content_preview", ""))
            file_path = result.metadata.get("relative_path", result.metadata.get("file_path", ""))
            name = result.metadata.get("name", "") or ""

            # Contextual BM25: prepend metadata header so BM25 can match on
            # file path, type, and name even when the code body doesn't contain
            # the query terms. Evidence: +0.128 MRR on TypeScript when combined
            # with query rewriting (A/B eval 2026-04-07, 102 queries).
            chunk_type = result.metadata.get("chunk_type", "")
            parent = result.metadata.get("parent_name", "")
            header_parts = []
            if file_path:
                header_parts.append(f"# From {file_path}")
            if parent and name:
                header_parts.append(f"- {chunk_type} {parent}.{name}")
            elif name:
                header_parts.append(f"- {chunk_type} {name}")
            elif chunk_type:
                header_parts.append(f"- {chunk_type}")
            if header_parts:
                content = " ".join(header_parts) + "\n" + content

            self._fts_conn.execute(
                "INSERT INTO chunk_fts (chunk_id, content, file_path, name) VALUES (?, ?, ?, ?)",
                (result.chunk_id, content, file_path, name),
            )
        self._fts_conn.commit()

        # Update statistics
        self._update_stats()

    def _gpu_is_available(self) -> bool:
        """Check if GPU FAISS support is available and GPUs are present."""
        try:
            if not hasattr(faiss, 'StandardGpuResources'):
                return False
            get_num_gpus = getattr(faiss, 'get_num_gpus', None)
            if get_num_gpus is None:
                return False
            return get_num_gpus() > 0
        except Exception:
            return False

    def _maybe_move_index_to_gpu(self) -> None:
        """Move the current index to GPU if supported. No-op if already on GPU or unsupported."""
        if self._index is None or self._on_gpu:
            return
        if not self._gpu_is_available():
            return
        try:
            # Move index to all GPUs for faster add/search
            self._index = faiss.index_cpu_to_all_gpus(self._index)
            self._on_gpu = True
            self._logger.info("FAISS index moved to GPU(s)")
        except Exception as e:
            self._logger.warning(f"Failed to move FAISS index to GPU, continuing on CPU: {e}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar code chunks."""
        import logging
        logger = logging.getLogger(__name__)

        # R2: reject k <= 0 with a clean ValueError instead of letting it
        # hit the FAISS bindings (which raise an AssertionError with no
        # context). The MCP surface accepts a k arg from external callers
        # who can trivially pass 0 or a negative; an explicit error here
        # is the right boundary.
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                f"k must be a positive integer, got {k!r} (type={type(k).__name__})"
            )

        logger.info(f"Index manager search called with k={k}, filters={filters}")

        # Use property to trigger lazy loading
        index = self.index
        if index is None or index.ntotal == 0:
            logger.warning(f"Index is empty or None. Index: {index}, ntotal: {index.ntotal if index else 'N/A'}")
            return []

        logger.info(f"Index has {index.ntotal} total vectors")

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Binary mode: hamming search → float rescore
        if getattr(self, '_is_binary', False) and hasattr(self, '_float_store'):
            search_k = min(k * 20, index.ntotal)
            q_codes = np.packbits((query_embedding[0] > 0).astype(np.uint8)).reshape(1, -1)
            _distances, bin_indices = index.search(q_codes, search_k)
            # Rescore with float dot product
            candidate_ids = bin_indices[0][bin_indices[0] >= 0]
            if len(candidate_ids) == 0:
                return []
            candidate_vecs = self._float_store[candidate_ids]
            scores = candidate_vecs @ query_embedding[0]
            top_order = np.argsort(-scores)
            indices = np.array([candidate_ids[top_order]])
            similarities = np.array([scores[top_order]])
        else:
            # Standard search (float32 or int8)
            search_k = min(k * 3, index.ntotal)
            similarities, indices = index.search(query_embedding, search_k)
        
        results = []
        seen = set()
        for i, (similarity, index_id) in enumerate(zip(similarities[0], indices[0])):
            if index_id == -1:  # No more results
                break

            # Defensive bounds check: a truncated chunk_ids list (pre-repair)
            # must degrade to fewer results, not IndexError.
            if index_id >= len(self._chunk_ids):
                continue

            chunk_id = self._chunk_ids[index_id]
            if chunk_id is None:
                continue
            # Dedupe: after a modify→re-add cycle the same chunk_id exists at
            # two FAISS positions (the stale vector is never removed). FAISS
            # returns results sorted by similarity, so the first occurrence
            # is the best-scoring one; later duplicates would otherwise
            # occupy extra result slots AND get double-counted by RRF, which
            # sums contributions per appearance.
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            metadata_entry = self.metadata_db.get(chunk_id)

            if metadata_entry is None:
                continue

            metadata = metadata_entry['metadata']

            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue

            results.append((chunk_id, float(similarity), metadata))

            if len(results) >= k:
                break

        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters.

        R1: filters with `None` values are treated as "filter absent" rather
        than "match nothing". Pre-fix, `chunk_type=None` would compare
        `metadata['chunk_type'] != None` which is True for every indexed
        chunk → filter rejects all results silently. Pre-fix,
        `file_pattern=None` would crash with TypeError on the for-loop.
        Both are operator-facing failure shapes (the MCP `search_code`
        tool accepts these args from external callers) so we normalize
        to a no-op filter here rather than at every call site.
        """
        for key, value in filters.items():
            # R1: a None value means "this filter is not provided" — skip it.
            # If a caller wants to filter for chunks where chunk_type IS
            # literally None (it's not, but for symmetry), they must pass
            # the explicit string the indexer stores.
            if value is None:
                continue

            if key == 'file_pattern':
                # Glob matching for file paths. fnmatch does shell-style:
                # `*.rs` matches `foo.rs`, `internal/x/foo.rs`, etc. (against
                # full path segments). Previously this was substring match,
                # which meant `*.rs` was never a substring of any real path
                # and silently filtered out all results — except it didn't,
                # because the BM25 path bypassed filtering entirely (see
                # search_bm25). Both bugs fixed together in this change.
                relative_path = metadata.get('relative_path', '') or ''
                # Match against both full path AND basename so users can
                # write `*.rs` (basename pattern) or `internal/**/*.rs`
                # (path pattern) interchangeably.
                basename = relative_path.split('/')[-1].split('\\')[-1]
                # Accept both a single pattern string and a list of patterns.
                # Pre-R1 the for-pattern-in-value path crashed on a single
                # string (it'd iterate chars); normalize first.
                patterns = value if isinstance(value, (list, tuple)) else [value]
                if not any(
                    fnmatch.fnmatch(relative_path, pattern)
                    or fnmatch.fnmatch(basename, pattern)
                    for pattern in patterns
                ):
                    return False
            elif key == 'chunk_type':
                # Exact match for chunk type
                if metadata.get('chunk_type') != value:
                    return False
            elif key == 'tags':
                # Tag intersection
                chunk_tags = set(metadata.get('tags', []))
                required_tags = set(value if isinstance(value, list) else [value])
                if not required_tags.intersection(chunk_tags):
                    return False
            elif key == 'folder_structure':
                # Check if any of the required folders are in the path
                chunk_folders = set(metadata.get('folder_structure', []))
                required_folders = set(value if isinstance(value, list) else [value])
                if not required_folders.intersection(chunk_folders):
                    return False
            elif key in metadata:
                # Direct metadata comparison
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata by ID."""
        metadata_entry = self.metadata_db.get(chunk_id)
        return metadata_entry['metadata'] if metadata_entry else None

    def count_chunks_in_file(self, relative_path: str) -> int:
        """Count the live chunks indexed for a specific file.

        Uses the FTS5 table's file_path column (plain equality scan, no
        MATCH). FTS rows are now deleted on remove_file_chunks, so this
        reflects live chunks only.
        """
        if not relative_path:
            return 0
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            return 0
        try:
            cursor = self._fts_conn.execute(
                "SELECT COUNT(*) FROM chunk_fts WHERE file_path = ?",
                (relative_path,),
            )
            return int(cursor.fetchone()[0])
        except Exception:
            return 0
    
    def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find chunks similar to a given chunk."""
        metadata_entry = self.metadata_db.get(chunk_id)
        if not metadata_entry:
            return []
        
        index_id = metadata_entry['index_id']
        if self._index is None or index_id >= self._index.ntotal:
            return []

        # Get the embedding for this chunk. Binary indexes reconstruct to
        # packed uint8 codes, not floats — pull the original vector from the
        # float store instead so the downstream float search path works.
        if getattr(self, '_is_binary', False) and hasattr(self, '_float_store'):
            if index_id >= len(self._float_store):
                return []
            embedding = self._float_store[index_id].copy()
        else:
            embedding = self._index.reconstruct(index_id)

        # Search for similar chunks (excluding the original)
        results = self.search(embedding, k + 1)
        
        # Filter out the original chunk
        return [(cid, sim, meta) for cid, sim, meta in results if cid != chunk_id][:k]
    
    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize a path for comparison: forward slashes, no trailing slash."""
        return str(path).replace("\\", "/").rstrip("/")

    @staticmethod
    def _is_absolute_norm(path: str) -> bool:
        """Absolute-path check on a _normalize_path'd string (POSIX or drive)."""
        return path.startswith("/") or (
            len(path) >= 3 and path[1] == ":" and path[2] == "/"
        )

    @classmethod
    def _paths_refer_to_same_file(cls, target: str, chunk_rel: str, chunk_abs: str) -> bool:
        """True if `target` (relative or absolute) identifies the chunk's file.

        Matching rules (the previous implementation used bare substring
        containment — `file_path in chunk_file or chunk_file in file_path` —
        which made removing `test.py` also delete `conftest.py`'s chunks;
        the un-modified file then silently vanished from the index until
        its next edit or a full reindex):

        - exact equality against the stored relative or absolute path;
        - an ABSOLUTE target matches a stored relative path when it ends
          with "/<relative path>" (path-segment boundary);
        - a RELATIVE target matches only the stored relative path exactly.
          It is NOT suffix-matched against the absolute path unless no
          relative path is stored — otherwise removing a root-level
          `util.py` would also match `src/util.py` (the absolute path ends
          with "/util.py").
        """
        target = cls._normalize_path(target)
        chunk_rel = cls._normalize_path(chunk_rel) if chunk_rel else ""
        chunk_abs = cls._normalize_path(chunk_abs) if chunk_abs else ""
        if not target:
            return False
        if target in (chunk_rel, chunk_abs):
            return True
        if cls._is_absolute_norm(target):
            # Absolute target vs relative metadata.
            return bool(chunk_rel) and target.endswith("/" + chunk_rel)
        # Relative target: only fall back to an absolute-suffix match when
        # the chunk stored no relative path at all.
        if not chunk_rel and chunk_abs:
            return chunk_abs.endswith("/" + target)
        return False

    def remove_file_chunks(self, file_path: str, project_name: Optional[str] = None) -> int:
        """Remove all chunks from a specific file.

        Removes the metadata rows AND the FTS5 rows. Pre-fix only metadata
        was deleted, so every modified file left its old FTS5 rows behind:
        re-adding the same chunk_id duplicated it in BM25 results (and RRF
        sums per-appearance, inflating fused scores), while shifted
        chunk_ids left dead rows that consumed the BM25 LIMIT quota.

        Args:
            file_path: Path to the file (relative or absolute)
            project_name: Optional project name filter

        Returns:
            Number of chunks removed
        """
        # Load existing on-disk state BEFORE iterating _chunk_ids. Without
        # this, a fresh CodeIndexManager (e.g., after switch_project) sees
        # an empty in-memory _chunk_ids and silently removes nothing —
        # the file's old chunks become orphans the next save will not
        # preserve. See add_embeddings for the symmetric fix.
        if self._index is None and self.index_path.exists():
            self._load_index()

        chunks_to_remove = []
        seen = set()

        # Find chunks to remove. _chunk_ids can contain the same chunk_id at
        # multiple FAISS positions after a modify→re-add cycle; dedupe so the
        # metadata delete below doesn't KeyError on the second occurrence.
        for chunk_id in self._chunk_ids:
            if chunk_id is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            metadata_entry = self.metadata_db.get(chunk_id)
            if not metadata_entry:
                continue

            metadata = metadata_entry['metadata']

            chunk_rel = metadata.get('relative_path') or ''
            chunk_abs = metadata.get('file_path') or ''
            if not (chunk_rel or chunk_abs):
                continue

            if self._paths_refer_to_same_file(file_path, chunk_rel, chunk_abs):
                # Check project name if provided
                if project_name and metadata.get('project_name') != project_name:
                    continue
                chunks_to_remove.append(chunk_id)

        # Remove chunks from metadata
        for chunk_id in chunks_to_remove:
            try:
                del self.metadata_db[chunk_id]
            except KeyError:
                pass

        # Remove the corresponding FTS5 rows (batched under SQLite's
        # parameter limit).
        if chunks_to_remove:
            if not hasattr(self, "_fts_conn") or self._fts_conn is None:
                self._init_fts5()
            try:
                for i in range(0, len(chunks_to_remove), 500):
                    batch = chunks_to_remove[i:i + 500]
                    placeholders = ",".join("?" * len(batch))
                    self._fts_conn.execute(
                        f"DELETE FROM chunk_fts WHERE chunk_id IN ({placeholders})",
                        batch,
                    )
                self._fts_conn.commit()
            except Exception as e:
                self._logger.warning(
                    f"FTS5 cleanup failed for {file_path}: {e}"
                )

        # Note: We don't remove from FAISS index directly as it's complex
        # Instead, we'll rebuild the index periodically or on demand

        self._logger.info(f"Removed {len(chunks_to_remove)} chunks from {file_path}")

        # Commit removals in batch
        try:
            self.metadata_db.commit()
        except Exception:
            pass
        return len(chunks_to_remove)
    
    def save_index(self, force: bool = False):
        """Save the FAISS index and chunk IDs to disk.

        Args:
            force: Bypass the chunk-truncation guard. Set True only by
                callers that legitimately shrink the index (clear_index,
                full reindex with deletions, explicit user reset). Default
                False so accidental truncation aborts loudly.
        """
        # CHUNK_ID DIAGNOSTIC (2026-05-05): log pre-save state to catch the
        # hypothesized failure mode where save_index dumps a truncated
        # _chunk_ids over a previously healthy on-disk pkl. If the on-disk
        # pkl was 10K entries and we're about to save 12, that's the bug.
        try:
            existing_pkl_size = (
                self.chunk_id_path.stat().st_size
                if self.chunk_id_path.exists()
                else 0
            )
        except Exception:
            existing_pkl_size = -1
        self._logger.warning(
            "[CHUNK_ID_DIAG] save_index pre-save: in_memory_chunk_ids_len=%s "
            "faiss.ntotal=%s on_disk_pkl_size=%s caller_path=%s",
            len(self._chunk_ids),
            self._index.ntotal if self._index else None,
            existing_pkl_size,
            self.chunk_id_path,
        )

        # Defense-in-depth: refuse to clobber a healthy on-disk pkl with a
        # dramatically smaller in-memory list unless the caller explicitly
        # opted in via force=True. The 2026-05-04/05 chunk-truncation
        # regression dumped 1 entry over a 966-byte (30-entry) pkl because
        # add_embeddings created a fresh empty FAISS instead of loading the
        # existing one. The lazy-load fix in add_embeddings/remove_file_chunks
        # closes that path; this guard catches any future variant.
        #
        # Threshold by COUNT, not bytes: load the existing pkl and compare
        # entry counts. Bytes-per-entry varies widely with chunk_id length
        # (~30 bytes for short paths, 150+ for nested ones), so a fixed
        # bytes-per-entry constant produced false positives on real data
        # (.claude with 10093 entries / 1.46MB = ~144 bytes/entry, not 32 —
        # a healthy 10114-entry save was rejected because 10114 * 32 < pkl
        # bytes / 2). Counting entries is robust and the load cost (~50ms
        # for a 10K-entry pkl) is trivial relative to a save_index call.
        in_memory_len = len(self._chunk_ids)
        existing_count = -1  # unknown
        if self.chunk_id_path.exists():
            try:
                with open(self.chunk_id_path, "rb") as f:
                    existing_chunk_ids = pickle.load(f)
                if isinstance(existing_chunk_ids, list):
                    existing_count = len(existing_chunk_ids)
            except Exception:
                # Corrupt or partial pkl — let the save proceed; the
                # rebuild paths in _load_index handle recovery.
                existing_count = -1

        TRUNCATION_GUARD_MIN_COUNT = 5  # only guard when on-disk has >= 5 entries
        TRUNCATION_GUARD_RATIO = 0.5  # refuse if in-memory < 50% of on-disk
        if (
            not force
            and existing_count >= TRUNCATION_GUARD_MIN_COUNT
            and in_memory_len < existing_count * TRUNCATION_GUARD_RATIO
        ):
            self._logger.error(
                "[CHUNK_ID_DIAG] save_index REFUSED: in_memory_chunk_ids_len=%s "
                "would clobber healthy on_disk_chunk_ids_count=%s "
                "(on_disk_pkl_size=%s bytes). This is the "
                "chunk-truncation regression shape. Pass force=True to "
                "override (e.g., after clear_index or intentional reset).",
                in_memory_len, existing_count, existing_pkl_size,
            )
            return

        if self._index is not None:
            try:
                if getattr(self, '_is_binary', False):
                    # Binary mode: save binary index + float store
                    faiss.write_index_binary(self._index, str(self.index_path))
                    float_path = self.storage_dir / "float_store.npy"
                    np.save(str(float_path), self._float_store)
                    self._logger.info(f"Saved binary index + float store to {self.storage_dir}")
                else:
                    index_to_write = self._index
                    if self._on_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                        index_to_write = faiss.index_gpu_to_cpu(self._index)
                    faiss.write_index(index_to_write, str(self.index_path))
                    self._logger.info(f"Saved index to {self.index_path}")
            except Exception as e:
                self._logger.warning(f"Failed to save index: {e}")
                if not getattr(self, '_is_binary', False):
                    try:
                        cpu_index = faiss.index_gpu_to_cpu(self._index)
                        faiss.write_index(cpu_index, str(self.index_path))
                    except Exception as e2:
                        self._logger.error(f"Failed to save FAISS index: {e2}")
        
        # Save chunk IDs
        with open(self.chunk_id_path, 'wb') as f:
            pickle.dump(self._chunk_ids, f)

        # CHUNK_ID DIAGNOSTIC: log post-save state.
        try:
            new_pkl_size = self.chunk_id_path.stat().st_size
        except Exception:
            new_pkl_size = -1
        self._logger.warning(
            "[CHUNK_ID_DIAG] save_index post-save: chunk_ids_len=%s "
            "new_pkl_size=%s",
            len(self._chunk_ids),
            new_pkl_size,
        )

        self._update_stats()

        # Plan-2 E2: commit an epoch-manifest for the artifacts just written.
        # The cross-artifact consistency check at build_manifest time
        # structurally prevents the chunk-truncation regression class —
        # if FAISS ntotal disagrees with len(chunk_ids), commit fails loudly.
        self._commit_epoch_manifest()

    def _commit_epoch_manifest(self) -> None:
        """Build + commit an epoch manifest covering the just-written artifacts.

        Called from save_index after FAISS + chunk_ids.pkl + stats are
        written.

        Failure modes and their handling:
        - ManifestConsistencyError (cross-artifact record counts disagree):
          re-raised. This is a structural-invariant violation — on-disk
          artifacts are demonstrably inconsistent (e.g., FAISS ntotal !=
          len(chunk_ids)). Silently swallowing this masks the chunk-truncation
          regression class. Caller learns about it via the propagated
          exception AND `self.last_manifest_commit_status == "consistency_error"`.
        - Other build_manifest errors (e.g., sha256 IO failure): logged and
          swallowed. Transient; readers safely fall back to prior.json.
        - commit_manifest errors (rename/fsync failure): logged and swallowed.
          Same rationale — prior epoch's manifest stays current; readers OK.

        Always sets `self.last_manifest_commit_status` to a stable string
        ("ok", "skipped_empty", "consistency_error", "build_error",
        "commit_error") for observability.
        """
        from search.epoch_manifest import (
            ArtifactSpec,
            ManifestConsistencyError,
            build_manifest,
            commit_manifest,
        )

        # WAL checkpoint guard: metadata.db is opened with journal_mode=WAL
        # (see the metadata_db property). Without an explicit checkpoint
        # here, pending writes sit in the .db-wal sidecar and the main
        # metadata.db file still reflects an earlier (often empty) state.
        # sha256(metadata.db) computed below would then capture that stale
        # state, and any later auto-checkpoint (SQLite default ≥1000 pages)
        # merges the WAL into the main file, breaking sha verification
        # permanently. Observed 2026-05-22: 16/24 projects had identical
        # manifest sha for metadata.db (42f67cde...) because all were
        # captured at the empty-schema state. TRUNCATE merges WAL → main
        # db and reclaims the .wal file space.
        if self._metadata_db is not None:
            try:
                self._metadata_db.commit()
            except Exception as exc:
                self._logger.warning(
                    "[EPOCH_MANIFEST] metadata_db.commit() failed before "
                    "checkpoint: %s", exc,
                )
        if self.metadata_path.exists():
            try:
                import sqlite3
                _con = sqlite3.connect(str(self.metadata_path))
                try:
                    _con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    _con.commit()
                finally:
                    _con.close()
            except Exception as exc:
                self._logger.warning(
                    "[EPOCH_MANIFEST] metadata.db WAL checkpoint failed "
                    "(manifest may capture stale sha): %s", exc,
                )

        artifacts: list[ArtifactSpec] = []
        chunk_count = len(self._chunk_ids)

        # Authoritative chunk_ids count drives consistency check.
        if self.chunk_id_path.exists():
            artifacts.append(ArtifactSpec(
                name="chunk_ids.pkl",
                path=self.chunk_id_path,
                count=chunk_count,
            ))

        # FAISS index — count via in-memory _index.ntotal (same value as on
        # disk since we just wrote it). Keep optional in case _index is None.
        if self.index_path.exists() and self._index is not None:
            try:
                ntotal = int(self._index.ntotal)
            except Exception:
                ntotal = chunk_count  # best-effort: assume consistent
            artifacts.append(ArtifactSpec(
                name="code.index",
                path=self.index_path,
                count=ntotal,
            ))

        # Sidecar SQLite stores. Pass count=None so they participate in
        # SHA verification but NOT in the cross-artifact record-count
        # consistency check.
        #
        # Why exclude them: remove_file_chunks updates metadata.db + fts5.db
        # (the rows are deleted) but intentionally leaves chunk_ids.pkl and
        # FAISS at their previous size — FAISS row removal is "rebuild on
        # demand" (see test_save_index_allows_small_legitimate_shrinks).
        # Treating sidecars as strict-count artifacts would fire a false
        # consistency error on every incremental run with deletions. The
        # invariant that DOES matter (the 2026-05-04 chunk-truncation
        # regression signature: chunk_ids.pkl row count != FAISS ntotal)
        # is preserved by chunk_ids + code.index above.
        if self.metadata_path.exists():
            artifacts.append(ArtifactSpec(
                name="metadata.db",
                path=self.metadata_path,
                count=None,
            ))
        if self._fts_db_path.exists():
            artifacts.append(ArtifactSpec(
                name="fts5.db",
                path=self._fts_db_path,
                count=None,
            ))
        # stats.json is metadata, not a record-bearing artifact (count=None).
        if self.stats_path.exists():
            artifacts.append(ArtifactSpec(
                name="stats.json",
                path=self.stats_path,
                count=None,
            ))

        if not artifacts:
            self._logger.info(
                "[EPOCH_MANIFEST] no artifacts to commit (empty index?); skipping"
            )
            self.last_manifest_commit_status = "skipped_empty"
            return

        try:
            manifest = build_manifest(
                project_dir=self.storage_dir,
                artifacts=artifacts,
                provider=getattr(self, "_embedder_provider", "") or "",
                model=getattr(self, "_embedder_model", "") or "",
                vector_dim=int(getattr(self._index, "d", 0) or 0),
                quantization="binary" if getattr(self, "_is_binary", False) else "int8",
                pipeline_version=getattr(self, "_pipeline_version", "") or "",
            )
        except ManifestConsistencyError as exc:
            # Cross-artifact consistency failure: on-disk artifacts have
            # mismatched record counts (e.g., FAISS ntotal != len(chunk_ids)).
            # This is a structural-invariant violation, not a transient
            # error — silently swallowing it masks the chunk-truncation
            # regression class. Re-raise so the caller (IncrementalIndexer)
            # learns the operation produced inconsistent state.
            self._logger.error(
                "[EPOCH_MANIFEST] consistency check failed; refusing to commit: %s",
                exc,
            )
            self.last_manifest_commit_status = "consistency_error"
            raise
        except Exception as exc:
            # Unexpected error in manifest construction — log + swallow.
            # Transient (e.g., sha256 IO failure on a slow disk); readers
            # safely fall back to prior.json. Operators can detect via
            # verify_index_integrity or last_manifest_commit_status.
            self._logger.warning(
                "[EPOCH_MANIFEST] build failed (non-blocking): %s", exc,
            )
            self.last_manifest_commit_status = "build_error"
            return

        try:
            committed = commit_manifest(self.storage_dir, manifest)
            self._logger.info(
                "[EPOCH_MANIFEST] committed epoch=%s artifacts=%d at %s",
                manifest["epoch_id"], len(artifacts), committed,
            )
            self.last_manifest_commit_status = "ok"
        except Exception as exc:
            # Commit-time error (rename failure, fsync failure). Log; the
            # artifacts on disk are unchanged — readers see the previous
            # epoch's manifest until the next successful commit.
            self._logger.warning(
                "[EPOCH_MANIFEST] commit failed (non-blocking): %s", exc,
            )
            self.last_manifest_commit_status = "commit_error"
    
    def _update_stats(self):
        """Update index statistics."""
        # Detect quantization type for reporting
        if getattr(self, '_is_binary', False):
            quant = "binary"
            idx_dim = self._float_store.shape[1] if hasattr(self, '_float_store') and len(self._float_store) > 0 else 0
        elif self._index and "ScalarQuantizer" in type(self._index).__name__:
            quant = "int8"
            idx_dim = self._index.d if self._index else 0
        else:
            quant = "float32"
            idx_dim = self._index.d if self._index else 0

        # Live vs stale accounting: FAISS rows are never removed in place
        # (removal is "rebuild on demand"), so after modify/delete churn
        # ntotal exceeds the live metadata row count. stale_vectors is the
        # operator signal for "a full reindex would compact this index".
        ntotal = self._index.ntotal if self._index else 0
        try:
            live_chunks = len(self.metadata_db)
        except Exception:
            live_chunks = None

        stats = {
            'total_chunks': len(self._chunk_ids),
            'index_size': ntotal,
            'embedding_dimension': idx_dim,
            'index_type': type(self._index).__name__ if self._index else 'None',
            'quantization': quant,
            'live_chunks': live_chunks,
            'stale_vectors': (
                max(0, ntotal - live_chunks) if live_chunks is not None else None
            ),
        }
        
        # Add file and folder statistics
        file_counts = {}
        folder_counts = {}
        chunk_type_counts = {}
        tag_counts = {}
        
        for chunk_id in self._chunk_ids:
            metadata_entry = self.metadata_db.get(chunk_id)
            if not metadata_entry:
                continue
            
            metadata = metadata_entry['metadata']
            
            # Count by file
            file_path = metadata.get('relative_path', 'unknown')
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
            
            # Count by folder
            for folder in metadata.get('folder_structure', []):
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            # Count by chunk type
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            
            # Count by tags
            for tag in metadata.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats.update({
            'files_indexed': len(file_counts),
            'top_folders': dict(sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'chunk_types': chunk_type_counts,
            'top_tags': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        })
        
        # Save stats
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        stats.json is derived (rewritten on every save); corruption returns
        the empty defaults with a warning instead of raising
        JSONDecodeError from the read path (2026-06-10 torn-write fuzz).
        """
        if self.stats_path.exists():
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (OSError, ValueError, UnicodeDecodeError) as e:
                self._logger.warning(
                    "stats.json is corrupt (%s) — returning defaults; it is "
                    "regenerated on the next index save", e,
                )
        return {
            'total_chunks': 0,
            'index_size': 0,
            'embedding_dimension': 0,
            'files_indexed': 0
        }
    
    def get_index_size(self) -> int:
        """Get the number of chunks in the index."""
        return len(self._chunk_ids)

    def stale_ratio(self) -> Optional[float]:
        """stale_vectors / live_chunks for the current on-disk index.

        Returns None when unknown (no index, empty index, or metadata
        unreadable). A ratio above STALE_COMPACTION_RATIO means the FAISS
        index holds more dead rows than live chunks and a full reindex is
        strictly better. Computed from live state (FAISS ntotal vs
        metadata.db row count), not stats.json, so it reflects churn that
        happened since the last save.
        """
        if self._index is None and self.index_path.exists():
            self._load_index()
        ntotal = int(self._index.ntotal) if self._index is not None else 0
        if ntotal == 0:
            return None
        try:
            live = len(self.metadata_db)
        except Exception:
            return None
        return max(0, ntotal - live) / max(live, 1)
    
    def clear_index(self):
        """Clear the entire index and metadata."""
        # Close database connection
        if self._metadata_db is not None:
            self._metadata_db.close()
            self._metadata_db = None

        # Close and remove FTS5 database
        if hasattr(self, "_fts_conn") and self._fts_conn is not None:
            self._fts_conn.close()
            self._fts_conn = None
        fts_path = self.storage_dir / "fts5.db"
        if fts_path.exists():
            fts_path.unlink()

        # Remove files
        for file_path in [self.index_path, self.metadata_path, self.chunk_id_path, self.stats_path]:
            if file_path.exists():
                file_path.unlink()

        # Reset in-memory state
        self._index = None
        self._chunk_ids = []

        # Re-initialize FTS5 for new inserts
        self._init_fts5()

        self._logger.info("Index cleared")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._metadata_db is not None:
            self._metadata_db.close()
        if hasattr(self, "_fts_conn") and self._fts_conn is not None:
            self._fts_conn.close()
