"""Incremental indexing using Merkle tree change detection."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from merkle.change_detector import ChangeDetector, FileChanges
from merkle.merkle_dag import MerkleDAG
from merkle.snapshot_manager import SnapshotManager
from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import CodeEmbedder
from search.indexer import CodeIndexManager as Indexer
from search.path_validation import refuse_as_project_root_reason
from search.env import env_get

logger = logging.getLogger(__name__)

REINDEX_DISPOSITION_COMPLETED = "completed"
REINDEX_DISPOSITION_SKIPPED_FRESH = "skipped_fresh"
REINDEX_DISPOSITION_SKIPPED_DISABLED = "skipped_disabled"
REINDEX_DISPOSITION_REFUSED = "refused"


@dataclass
class ChunkingDiagnostics:
    """Per-run summary of chunking-loop outcomes.

    Surfaces the silent failure modes (parse errors, empty results) that
    `MultiLanguageChunker.chunk_file` masks by returning `[]`. Counts here
    are coarse (per-file outcome categories); per-file detail lives in
    structured `[CHUNKING_DIAG_FILE]` log lines.

    Fields:
        files_attempted: Files passed to the chunker (all supported types).
        files_with_chunks: Files that produced ≥1 chunk.
        files_zero_chunks: Files that produced 0 chunks. Causes: parse
            error (chunker raised; see CHUNKING_DIAG_FILE log line),
            encoding error (UnicodeDecodeError), or genuinely empty file.
            All three look the same from the caller's perspective; check
            the log file to disambiguate per-file.
        chunks_extracted: Total CodeChunk count across all files.
    """

    files_attempted: int = 0
    files_with_chunks: int = 0
    files_zero_chunks: int = 0
    chunks_extracted: int = 0

    @property
    def zero_chunk_rate(self) -> float:
        """Fraction of attempted files that produced no chunks. Useful
        signal for `should I be worried?` — sustained >5% probably means
        parse errors or encoding issues warranting investigation."""
        if self.files_attempted == 0:
            return 0.0
        return self.files_zero_chunks / self.files_attempted

    def to_dict(self) -> Dict:
        return {
            "files_attempted": self.files_attempted,
            "files_with_chunks": self.files_with_chunks,
            "files_zero_chunks": self.files_zero_chunks,
            "chunks_extracted": self.chunks_extracted,
            "zero_chunk_rate": round(self.zero_chunk_rate, 4),
        }


@dataclass
class IncrementalIndexResult:
    """Result of incremental indexing operation."""

    files_added: int
    files_removed: int
    files_modified: int
    chunks_added: int
    chunks_removed: int
    time_taken: float
    success: bool
    error: Optional[str] = None
    chunking_diagnostics: Optional[ChunkingDiagnostics] = None
    reindex_disposition: str = REINDEX_DISPOSITION_COMPLETED

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        out = {
            "files_added": self.files_added,
            "files_removed": self.files_removed,
            "files_modified": self.files_modified,
            "chunks_added": self.chunks_added,
            "chunks_removed": self.chunks_removed,
            "time_taken": self.time_taken,
            "success": self.success,
            "error": self.error,
            "reindex_disposition": self.reindex_disposition,
        }
        if self.chunking_diagnostics is not None:
            out["chunking_diagnostics"] = self.chunking_diagnostics.to_dict()
        return out


class IncrementalIndexer:
    """Handles incremental indexing of code changes."""

    def __init__(
        self,
        indexer: Optional[Indexer] = None,
        embedder: Optional[CodeEmbedder] = None,
        chunker: Optional[MultiLanguageChunker] = None,
        snapshot_manager: Optional[SnapshotManager] = None,
        progress_fn=None,
        cancel_check=None,
    ):
        """Initialize incremental indexer.

        Args:
            indexer: Indexer instance
            embedder: Embedder instance
            chunker: Code chunker instance
            snapshot_manager: Snapshot manager instance
            progress_fn: Optional callback(phase, current, total) for progress reporting
            cancel_check: Optional callable returning True if the
                indexing run should abort. Polled by MerkleDAG at every
                directory boundary during the file-walk phase. Without
                this, cancel_indexing only propagates after chunking
                begins — useless when the walk itself is the slow phase
                (e.g., $HOME indexing wedge resolved by Phase A2 at the
                server-level guard, but A3 closes it at the indexer level
                for any other long legitimate walk).
        """
        self.indexer = indexer or Indexer()
        self.embedder = embedder or CodeEmbedder()
        self.chunker = chunker or MultiLanguageChunker()
        self.snapshot_manager = snapshot_manager or SnapshotManager()
        self.change_detector = ChangeDetector(self.snapshot_manager)
        self._progress_fn = progress_fn or (lambda phase, current, total: None)
        self._cancel_check = cancel_check
        # Latest chunking diagnostics from _add_new_chunks (incremental path)
        # or _full_index (full path). Attached to IncrementalIndexResult on
        # the next return so callers can inspect zero-chunk rate.
        self._last_chunking_diag: Optional[ChunkingDiagnostics] = None

    @staticmethod
    def _log_chunking_diag(
        diag: "ChunkingDiagnostics", scope: str, project_name: str
    ) -> None:
        """Emit a single structured summary line to the sidecar log.

        Operator visibility: `tail -f ~/.claude/logs/code-search-mcp.log
        | grep CHUNKING_DIAG` shows zero-chunk rates across runs without
        scrolling through per-file noise. The threshold for "should I
        investigate?" is roughly zero_chunk_rate > 0.05.
        """
        logger.warning(
            "[CHUNKING_DIAG] %s project=%s files_attempted=%d "
            "files_with_chunks=%d files_zero_chunks=%d chunks_extracted=%d "
            "zero_chunk_rate=%.4f",
            scope, project_name,
            diag.files_attempted, diag.files_with_chunks,
            diag.files_zero_chunks, diag.chunks_extracted,
            diag.zero_chunk_rate,
        )

    def detect_changes(self, project_path: str) -> Tuple[FileChanges, MerkleDAG]:
        """Detect changes in project since last snapshot.

        Args:
            project_path: Path to project

        Returns:
            Tuple of (FileChanges, current MerkleDAG)
        """
        return self.change_detector.detect_changes_from_snapshot(project_path)

    def incremental_index(
        self,
        project_path: str,
        project_name: Optional[str] = None,
        force_full: bool = False,
    ) -> IncrementalIndexResult:
        """Perform incremental indexing of a project.

        Args:
            project_path: Path to project
            project_name: Optional project name
            force_full: Force full reindex even if snapshot exists

        Returns:
            IncrementalIndexResult with statistics
        """
        start_time = time.time()
        project_path = str(Path(project_path).resolve())

        if not project_name:
            project_name = Path(project_path).name

        # Reset chunking diag for this run so a subsequent "no changes" call
        # doesn't return a stale diag from a previous indexing pass.
        self._last_chunking_diag = None

        # REINDEX PROGRESS: structured milestones land in the file-sidecar
        # logger (~/.claude/logs/code-search-mcp.log) so an operator can
        # `tail -f` and see liveness during long auto_reindex calls. The
        # MCP server's pythonw.exe discards stderr, so without this the
        # call is invisibly slow.
        logger.warning(
            "[REINDEX_PROGRESS] incremental_index: starting project=%s force_full=%s",
            project_name, force_full,
        )

        try:
            # Check if we should do full index
            if force_full or not self.snapshot_manager.has_snapshot(project_path):
                logger.warning(
                    "[REINDEX_PROGRESS] incremental_index: dispatching to _full_index "
                    "project=%s",
                    project_name,
                )
                return self._full_index(project_path, project_name, start_time)

            # Snapshot-without-artifacts guard (2026-06-12): a merkle
            # snapshot can outlive the index artifacts — observed live when
            # the storage projects/ dir was deleted while merkle/ survived.
            # With an unchanged source tree, change detection against the
            # stale snapshot reports "no changes" and the call no-ops
            # "successfully" against a void index. An existing snapshot only
            # licenses the incremental path when the index actually holds
            # chunks. Probe failure (e.g. an indexer double without
            # get_stats) keeps the legacy behavior — never force a full
            # reindex on observability errors.
            try:
                indexed_chunks = int(
                    (self.indexer.get_stats() or {}).get("total_chunks", 0)
                )
            except Exception:
                indexed_chunks = -1
            try:
                snapshot_metadata = (
                    self.snapshot_manager.load_metadata(project_path) or {}
                )
                snapshot_declares_empty_corpus = (
                    int(snapshot_metadata.get("supported_files", -1)) == 0
                    and int(snapshot_metadata.get("chunks_indexed", -1)) == 0
                )
            except Exception:
                snapshot_declares_empty_corpus = False
            if indexed_chunks == 0 and not snapshot_declares_empty_corpus:
                logger.warning(
                    "[REINDEX_PROGRESS] incremental_index: snapshot present "
                    "but index artifacts hold 0 chunks — dispatching to "
                    "_full_index project=%s",
                    project_name,
                )
                return self._full_index(project_path, project_name, start_time)

            # P5 (2026-06-10 roadmap): stale-vector auto-compaction. FAISS
            # rows are never removed in place, so modify/delete churn
            # accumulates dead vectors that waste search-quota and memory.
            # When the index holds more garbage than live data, escalate to
            # a full reindex (clears + rebuilds; ratio resets to 0, so this
            # is self-limiting). Ratio None (empty/unknown) never escalates.
            try:
                ratio = self.indexer.stale_ratio()
            except Exception:
                ratio = None
            if ratio is not None and ratio > self.indexer.STALE_COMPACTION_RATIO:
                logger.warning(
                    "[REINDEX_PROGRESS] compaction: stale_ratio=%.2f exceeds "
                    "%.2f — escalating to full reindex project=%s",
                    ratio, self.indexer.STALE_COMPACTION_RATIO, project_name,
                )
                return self._full_index(project_path, project_name, start_time)

            # Detect changes — Merkle-hashing every file in the tree;
            # this is the slowest single step on large projects.
            t_detect = time.time()
            logger.warning(
                "[REINDEX_PROGRESS] detect_changes: starting project=%s",
                project_name,
            )
            self._progress_fn("detecting_changes", 0, 0)
            changes, current_dag = self.detect_changes(project_path)
            logger.warning(
                "[REINDEX_PROGRESS] detect_changes: done in %.1fs project=%s "
                "added=%d removed=%d modified=%d",
                time.time() - t_detect, project_name,
                len(changes.added), len(changes.removed), len(changes.modified),
            )

            if not changes.has_changes():
                logger.warning(
                    "[REINDEX_PROGRESS] incremental_index: no changes "
                    "project=%s elapsed=%.1fs",
                    project_name, time.time() - start_time,
                )
                return IncrementalIndexResult(
                    files_added=0,
                    files_removed=0,
                    files_modified=0,
                    chunks_added=0,
                    chunks_removed=0,
                    time_taken=time.time() - start_time,
                    success=True,
                )

            # CHUNK + EMBED FIRST. Embedding can fail transiently (API 5xx,
            # timeout). _chunk_and_embed_changes raises on embed failure rather
            # than swallowing it, so the `except` below returns success=False
            # WITHOUT having removed the modified files' old chunks or advanced
            # the snapshot. Sequencing embed-before-remove is what prevents a
            # transient embed error from permanently dropping a modified file
            # from the index (the pre-2026-05 order removed old chunks first,
            # then swallowed the embed error and saved the snapshot — so the
            # file was lost AND marked up-to-date, never to be re-indexed).
            t_embed = time.time()
            logger.warning(
                "[REINDEX_PROGRESS] _chunk_and_embed_changes: starting "
                "files_to_index=%d project=%s",
                len(changes.added) + len(changes.modified), project_name,
            )
            embedding_results = self._chunk_and_embed_changes(
                changes, project_path, project_name
            )
            logger.warning(
                "[REINDEX_PROGRESS] _chunk_and_embed_changes: done in %.1fs embedded=%d",
                time.time() - t_embed, len(embedding_results),
            )

            # Hold one storage transaction from the first root-store mutation
            # through publication. A second manager must not interpret this
            # live writer's marker as evidence of a crashed publication.
            with self.indexer.publication_transaction():
                # Only after embedding SUCCEEDS do we mutate the index: drop
                # stale chunks, then add the freshly-embedded ones.
                t_remove = time.time()
                logger.warning(
                    "[REINDEX_PROGRESS] _remove_old_chunks: starting "
                    "files_to_remove=%d project=%s",
                    len(changes.removed) + len(changes.modified), project_name,
                )
                chunks_removed = self._remove_old_chunks(
                    changes, project_name
                )
                logger.warning(
                    "[REINDEX_PROGRESS] _remove_old_chunks: done in %.1fs "
                    "removed=%d",
                    time.time() - t_remove, chunks_removed,
                )

                if embedding_results:
                    self._progress_fn("saving", 0, 0)
                    self.indexer.add_embeddings(embedding_results)
                chunks_added = len(embedding_results)

                # Persist and VERIFY the index BEFORE advancing the snapshot.
                # save_index() validates chunk IDs against FAISS and raises on
                # divergence or disk error.
                self.indexer.save_index()

            # Snapshot LAST: only once the index is durably and consistently
            # saved do we record the new content hashes.
            self.snapshot_manager.save_snapshot(
                current_dag,
                {
                    "project_name": project_name,
                    "incremental_update": True,
                    "files_added": len(changes.added),
                    "files_removed": len(changes.removed),
                    "files_modified": len(changes.modified),
                },
            )
            logger.warning(
                "[REINDEX_PROGRESS] incremental_index: done in %.1fs "
                "project=%s chunks_added=%d chunks_removed=%d",
                time.time() - start_time, project_name,
                chunks_added, chunks_removed,
            )

            return IncrementalIndexResult(
                files_added=len(changes.added),
                files_removed=len(changes.removed),
                files_modified=len(changes.modified),
                chunks_added=chunks_added,
                chunks_removed=chunks_removed,
                time_taken=time.time() - start_time,
                success=True,
                chunking_diagnostics=self._last_chunking_diag,
            )

        except InterruptedError:
            # Cancellation (progress checkpoint or Merkle walk) is not a
            # failure: restore the last-good index and let the caller record
            # the job as cancelled.
            logger.info("Incremental indexing cancelled; restoring last-good index")
            try:
                self.indexer.rollback_unpublished_changes()
            except Exception as rollback_error:
                logger.error(
                    "Failed to restore the last-good index after incremental "
                    "indexing was cancelled: %s",
                    rollback_error,
                )
            raise
        except Exception as e:
            logger.error(f"Incremental indexing failed: {e}")
            try:
                self.indexer.rollback_unpublished_changes()
            except Exception as rollback_error:
                logger.error(
                    "Failed to restore the last-good index after incremental "
                    "indexing aborted: %s",
                    rollback_error,
                )
            return IncrementalIndexResult(
                files_added=0,
                files_removed=0,
                files_modified=0,
                chunks_added=0,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def _full_index(
        self, project_path: str, project_name: str, start_time: float
    ) -> IncrementalIndexResult:
        """Perform full indexing of a project.

        Args:
            project_path: Path to project
            project_name: Project name
            start_time: Start time for timing

        Returns:
            IncrementalIndexResult
        """
        try:
            # Dimension mismatch guard: detect if the embedding dimension changed
            # between providers (e.g., voyage-code-3 1024 vs local 384). Log a
            # warning since we're about to clear_index() anyway in full reindex.
            configuration = getattr(self.embedder, "configuration", None)
            current_dim = getattr(configuration, "output_dimension", None)
            if current_dim is None:
                # Compatibility for lightweight third-party/test embedders
                # that predate the effective-configuration contract.
                current_dim = self.embedder._model.get_embedding_dimension()
            stats = self.indexer.get_stats()
            stored_dim = stats.get("embedding_dimension", 0)
            if stored_dim and stored_dim != current_dim:
                logger.warning(
                    f"Embedding dimension changed ({stored_dim} -> {current_dim}), "
                    f"clearing index to prevent mixed embeddings"
                )

            # Build DAG for all files. Phase A3 (2026-05-08): pass the
            # cancel_check callable so an in-flight cancel_indexing call
            # propagates within seconds during the merkle walk, not just
            # after chunking begins.
            dag = MerkleDAG(project_path, cancel_check=self._cancel_check)
            dag.build()
            all_files = dag.get_all_files()

            # Filter supported files
            supported_files = [f for f in all_files if self.chunker.is_supported(f)]

            # Collect all chunks first, then embed in a single pass for efficiency
            all_chunks = []
            diag = ChunkingDiagnostics()
            self._progress_fn("chunking", 0, len(supported_files))
            for idx, file_path in enumerate(supported_files):
                full_path = Path(project_path) / file_path
                diag.files_attempted += 1
                try:
                    chunks = self.chunker.chunk_file(str(full_path))
                except Exception as e:
                    # chunk_file's own try/except returns [] on most failures,
                    # so this is a belt-and-suspenders catch for unexpected
                    # propagation. Either way, count as zero-chunk.
                    logger.warning(f"Failed to chunk {file_path}: {e}")
                    chunks = []
                if chunks:
                    all_chunks.extend(chunks)
                    diag.files_with_chunks += 1
                    diag.chunks_extracted += len(chunks)
                else:
                    diag.files_zero_chunks += 1
                self._progress_fn("chunking", idx + 1, len(supported_files))
            self._log_chunking_diag(diag, scope="_full_index", project_name=project_name)
            self._last_chunking_diag = diag

            # Embed chunks — use Batch API for large full reindexes if enabled
            all_embedding_results = []
            # Batch ceiling matches Voyage's hard API limits (1000 inputs,
            # 120K tokens). Provider-specific embedders (voyage-context,
            # openai_embedder) enforce their own token-aware sub-batching, so
            # this is an upper bound not a target size. Prior value of 64 was
            # ~16x below even the inner voyage-context cap, causing Voyage
            # API's fixed ~20s per-call overhead to dominate throughput on
            # large repos (2026-04-17 incident: 10,993 chunks took hours at
            # 64 instead of ~20 min at 1000).
            embed_batch_size = 1000
            self._progress_fn("embedding", 0, len(all_chunks))

            use_batch = (
                env_get("VOYAGE_BATCH_API", "off") == "on"
                and len(all_chunks) >= int(env_get("VOYAGE_BATCH_THRESHOLD", "1000"))
                and hasattr(self.embedder._model, '_model_name')
                and self.embedder._model._model_name.startswith("voyage-")
            )

            embed_batch_failures = 0
            if use_batch and all_chunks:
                logger.info(f"Using Voyage Batch API for {len(all_chunks)} chunks (33% cheaper)")
                try:
                    from embeddings.voyage_batch_embedder import VoyageBatchEmbedder
                    # Route through the same chunk_id/metadata builder as
                    # every other embed path. The previous inline metadata
                    # block omitted `full_content`, so Batch-API-built
                    # indexes silently fed only the 200-char preview to the
                    # FTS5/BM25 side (add_embeddings reads `full_content`).
                    from embeddings.embedder import _make_embedding_result

                    all_contents = [self.embedder.create_embedding_content(c) for c in all_chunks]
                    batch_emb = VoyageBatchEmbedder(model=self.embedder._model._model_name)
                    embeddings_array = batch_emb.embed_all(all_contents, input_type="document")
                    batch_emb.close()

                    if embeddings_array is not None and len(embeddings_array) == len(all_chunks):
                        for chunk, emb_vec in zip(all_chunks, embeddings_array, strict=False):
                            embedding_result = _make_embedding_result(chunk, emb_vec)
                            embedding_result.metadata["project_name"] = project_name
                            all_embedding_results.append(embedding_result)
                        self._progress_fn("embedding", len(all_chunks), len(all_chunks))
                        logger.info(f"Batch API: embedded {len(all_embedding_results)} chunks")
                    else:
                        logger.warning("Batch API returned incomplete results, falling back to real-time")
                        use_batch = False  # Fall through to real-time loop below
                except Exception as e:
                    logger.warning(f"Batch API failed ({e}), falling back to real-time")
                    use_batch = False

            if not use_batch and all_chunks:
                for batch_start in range(0, len(all_chunks), embed_batch_size):
                    batch = all_chunks[batch_start : batch_start + embed_batch_size]
                    try:
                        batch_results = self.embedder.embed_chunks_grouped(
                            batch, batch_size=len(batch)
                        )
                        for chunk, embedding_result in zip(batch, batch_results, strict=False):
                            embedding_result.metadata["project_name"] = project_name
                        all_embedding_results.extend(batch_results)
                        self._progress_fn(
                            "embedding", len(all_embedding_results), len(all_chunks)
                        )
                        logger.info(
                            f"Embedded {batch_start + len(batch)}/{len(all_chunks)} chunks"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Embedding batch {batch_start}-{batch_start + len(batch)} failed: {e}"
                        )
                        # Continue with next batch instead of losing everything,
                        # but record the failure: a partial index must NOT be
                        # recorded as a current snapshot (see below).
                        embed_batch_failures += 1

            if embed_batch_failures > 0:
                error_msg = (
                    f"{embed_batch_failures} embedding batch(es) failed; "
                    f"prepared {len(all_embedding_results)}/{len(all_chunks)} "
                    "chunks. Last-good index preserved and snapshot NOT "
                    "advanced."
                )
                logger.error(
                    "[REINDEX_PROGRESS] _full_index: %s",
                    error_msg,
                )
                return IncrementalIndexResult(
                    files_added=len(supported_files),
                    files_removed=0,
                    files_modified=0,
                    chunks_added=0,
                    chunks_removed=0,
                    time_taken=time.time() - start_time,
                    success=False,
                    error=error_msg,
                    chunking_diagnostics=self._last_chunking_diag,
                )

            if not all_chunks and supported_files:
                error_msg = (
                    "Full rebuild produced no chunks; last-good index "
                    "preserved and snapshot NOT advanced."
                )
                logger.error(
                    "[REINDEX_PROGRESS] _full_index: %s",
                    error_msg,
                )
                return IncrementalIndexResult(
                    files_added=len(supported_files),
                    files_removed=0,
                    files_modified=0,
                    chunks_added=0,
                    chunks_removed=0,
                    time_taken=time.time() - start_time,
                    success=False,
                    error=error_msg,
                    chunking_diagnostics=self._last_chunking_diag,
                )

            if len(all_embedding_results) != len(all_chunks):
                error_msg = (
                    "Full rebuild embedding output was incomplete "
                    f"({len(all_embedding_results)}/{len(all_chunks)}); "
                    "last-good index preserved and snapshot NOT advanced."
                )
                logger.error(
                    "[REINDEX_PROGRESS] _full_index: %s",
                    error_msg,
                )
                return IncrementalIndexResult(
                    files_added=len(supported_files),
                    files_removed=0,
                    files_modified=0,
                    chunks_added=0,
                    chunks_removed=0,
                    time_taken=time.time() - start_time,
                    success=False,
                    error=error_msg,
                    chunking_diagnostics=self._last_chunking_diag,
                )

            # All fallible chunking and embedding work completed before
            # replacing the mutable root mirrors. The committed generation
            # remains available for rollback until save_index succeeds.
            self._progress_fn("saving", 0, 0)
            with self.indexer.publication_transaction():
                self.indexer.begin_rebuild()
                if all_embedding_results:
                    self.indexer.add_embeddings(all_embedding_results)
                else:
                    # An actually empty corpus is a valid new state. Publish
                    # an empty FAISS generation so deleted source trees do
                    # not keep returning stale results forever.
                    self.indexer.create_index(int(current_dim))

                chunks_added = len(all_embedding_results)

                # Save and VERIFY the index BEFORE advancing the snapshot
                # (same ordering invariant as the incremental path).
                self.indexer.save_index(force=True)

            # Post-indexing smoke test: verify vector search returns non-zero
            # similarities. Catches silent FAISS quantizer bugs (QT_8bit_direct
            # returned 0.0 for all queries — discovered 2026-04-05).
            if all_embedding_results and self.indexer.index is not None:
                try:
                    test_embedding = all_embedding_results[0].embedding
                    test_results = self.indexer.search(test_embedding, k=3)
                    if test_results:
                        top_sim = test_results[0][1]
                        if top_sim == 0.0:
                            logger.error(
                                "SMOKE TEST FAILED: Vector search returns 0.0 similarity. "
                                "FAISS quantizer may be misconfigured (QT_8bit_direct?). "
                                "Search will fall back to BM25-only."
                            )
                        elif len(test_results) >= 2 and len(set(r[1] for r in test_results)) == 1:
                            logger.warning(
                                "SMOKE TEST WARNING: All similarities identical. "
                                "Vector component may not be discriminating."
                            )
                        else:
                            logger.info(f"Smoke test passed: top similarity={top_sim:.4f}")
                except Exception as e:
                    logger.warning(f"Smoke test skipped: {e}")

            # Snapshot LAST — only after the index is durably saved and
            # smoke-tested. Snapshot-last is the invariant that lets a failed
            # save leave the prior snapshot in place so the next run retries.
            self.snapshot_manager.save_snapshot(
                dag,
                {
                    "project_name": project_name,
                    "full_index": True,
                    "total_files": len(all_files),
                    "supported_files": len(supported_files),
                    "chunks_indexed": chunks_added,
                },
            )

            return IncrementalIndexResult(
                files_added=len(supported_files),
                files_removed=0,
                files_modified=0,
                chunks_added=chunks_added,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=True,
                chunking_diagnostics=self._last_chunking_diag,
            )

        except InterruptedError:
            # Cancellation (progress checkpoint or Merkle walk) is not a
            # failure: restore the last-good index and let the caller record
            # the job as cancelled.
            logger.info("Full indexing cancelled; restoring last-good index")
            try:
                self.indexer.rollback_unpublished_changes()
            except Exception as rollback_error:
                logger.error(
                    "Failed to restore the last-good index after full "
                    "indexing was cancelled: %s",
                    rollback_error,
                )
            raise
        except Exception as e:
            logger.error(f"Full indexing failed: {e}")
            try:
                self.indexer.rollback_unpublished_changes()
            except Exception as rollback_error:
                logger.error(
                    "Failed to restore the last-good index after full "
                    "indexing aborted: %s",
                    rollback_error,
                )
            return IncrementalIndexResult(
                files_added=0,
                files_removed=0,
                files_modified=0,
                chunks_added=0,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def _remove_old_chunks(self, changes: FileChanges, project_name: str) -> int:
        """Remove chunks for deleted and modified files.

        Args:
            changes: File changes
            project_name: Project name

        Returns:
            Number of chunks removed
        """
        files_to_remove = self.change_detector.get_files_to_remove(changes)
        chunks_removed = 0

        total = len(files_to_remove)
        self._progress_fn("removing", 0, total)
        for idx, file_path in enumerate(files_to_remove):
            # Remove from metadata
            removed = self.indexer.remove_file_chunks(file_path, project_name)
            chunks_removed += removed
            logger.debug(f"Removed {removed} chunks from {file_path}")
            self._progress_fn("removing", idx + 1, total)

        return chunks_removed

    def _chunk_and_embed_changes(
        self, changes: FileChanges, project_path: str, project_name: str
    ) -> list:
        """Chunk and embed new/modified files; return the embedding results.

        This is the data-loss-safe half of the old `_add_new_chunks`: it does
        NOT touch the index. `incremental_index` calls it BEFORE removing the
        stale chunks, so a failure here aborts the run while the old chunks are
        still intact.

        Per-file chunking failures stay graceful (a parse error on one file is
        counted as a zero-chunk file, not fatal). But an embedding failure
        deliberately PROPAGATES — the previous code swallowed it and returned
        0, which, after `_remove_old_chunks` had already deleted the modified
        file's chunks and the snapshot was saved, dropped the file from the
        index permanently. Raising lets the caller abort before any of that.

        Args:
            changes: File changes
            project_path: Project root path
            project_name: Project name

        Returns:
            List of EmbeddingResult (metadata stamped with project_name +
            content), ready to hand to indexer.add_embeddings.
        """
        files_to_index = self.change_detector.get_files_to_reindex(changes)

        # Filter supported files
        supported_files = [f for f in files_to_index if self.chunker.is_supported(f)]

        # Collect all chunks first, then embed in a single pass
        chunks_to_embed = []
        diag = ChunkingDiagnostics()
        total_files = len(supported_files)
        self._progress_fn("chunking", 0, total_files)
        for idx, file_path in enumerate(supported_files):
            full_path = Path(project_path) / file_path
            diag.files_attempted += 1
            try:
                chunks = self.chunker.chunk_file(str(full_path))
            except Exception as e:
                logger.warning(f"Failed to chunk {file_path}: {e}")
                chunks = []
            if chunks:
                chunks_to_embed.extend(chunks)
                diag.files_with_chunks += 1
                diag.chunks_extracted += len(chunks)
            else:
                diag.files_zero_chunks += 1
            self._progress_fn("chunking", idx + 1, total_files)
        self._log_chunking_diag(diag, scope="_add_new_chunks", project_name=project_name)
        self._last_chunking_diag = diag

        all_embedding_results = []
        if chunks_to_embed:
            self._progress_fn("embedding", 0, len(chunks_to_embed))
            # NOT wrapped in try/except: a transient embed failure must
            # propagate so incremental_index aborts BEFORE removing old chunks
            # or advancing the snapshot. (Data-loss fix, 2026-05.)
            all_embedding_results = self.embedder.embed_chunks_grouped(
                chunks_to_embed
            )
            # Update metadata. (project_name only — the chunk text already
            # lives in metadata as `full_content`; the old `content` stamp
            # duplicated the full chunk body in every metadata row, doubling
            # metadata.db size for no consumer benefit.)
            for chunk, embedding_result in zip(
                chunks_to_embed, all_embedding_results, strict=False
            ):
                embedding_result.metadata["project_name"] = project_name
            self._progress_fn(
                "embedding", len(chunks_to_embed), len(chunks_to_embed)
            )

        return all_embedding_results

    def get_indexing_stats(self, project_path: str) -> Optional[Dict]:
        """Get indexing statistics for a project.

        Args:
            project_path: Path to project

        Returns:
            Dictionary with statistics or None
        """
        metadata = self.snapshot_manager.load_metadata(project_path)
        if not metadata:
            return None

        # Add current index stats - prefer in-memory count, fall back to
        # snapshot metadata when index hasn't been loaded (e.g. no-changes path)
        in_memory_size = self.indexer.get_index_size()
        metadata["current_chunks"] = in_memory_size if in_memory_size > 0 else metadata.get("chunks_indexed", 0)
        metadata["snapshot_age"] = self.snapshot_manager.get_snapshot_age(project_path)

        return metadata

    def needs_reindex(self, project_path: str, max_age_minutes: float = 5) -> bool:
        """Check if a project needs reindexing.

        Args:
            project_path: Path to project
            max_age_minutes: Maximum age of snapshot in minutes (default 5)

        Returns:
            True if reindex is needed
        """
        # No snapshot means needs index
        if not self.snapshot_manager.has_snapshot(project_path):
            return True

        # Check snapshot age (convert minutes to seconds)
        age = self.snapshot_manager.get_snapshot_age(project_path)
        if age and age > max_age_minutes * 60:
            return True

        # Quick check for changes
        return self.change_detector.quick_check(project_path)

    def auto_reindex_if_needed(
        self,
        project_path: str,
        project_name: Optional[str] = None,
        max_age_minutes: float = 5,
    ) -> IncrementalIndexResult:
        """Automatically reindex if the index is stale.

        Args:
            project_path: Path to project
            project_name: Optional project name
            max_age_minutes: Maximum age before auto-reindex (default 5 minutes)

        Returns:
            IncrementalIndexResult with statistics

        Env escape hatch:
            CODE_SEARCH_DISABLE_AUTO_REINDEX=1 makes this a no-op. Useful
            when auto_reindex is hitting a large project (~10K+ chunks)
            whose detect_changes pass is multi-minute and the caller
            doesn't want to block the search call. The MCP user can run
            `mcp__code-search__index_directory(incremental=false)` later
            to refresh deliberately.
        """
        import time

        start_time = time.time()

        # U2 (2026-05-13): refuse to reindex home/root/workspace paths.
        # The Phase A refuse-check in `ensure_project_indexed` (caller side)
        # prevents new orphan entries from being created, but pre-existing
        # orphan entries in `~/.claude_code_search/projects/` from older
        # server versions would otherwise still trigger a full-index walk
        # on every 5-min cron tick. Defense-in-depth: skip them here too.
        refuse = refuse_as_project_root_reason(project_path)
        if refuse:
            logger.warning(
                "[REINDEX_PROGRESS] auto_reindex_if_needed: REFUSED "
                "project=%s reason=%s. Delete the orphan entry via "
                "clear_index/delete_project or filesystem rm.",
                project_name, refuse,
            )
            return IncrementalIndexResult(
                files_added=0,
                files_removed=0,
                files_modified=0,
                chunks_added=0,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=True,
                reindex_disposition=REINDEX_DISPOSITION_REFUSED,
            )

        if env_get("CODE_SEARCH_DISABLE_AUTO_REINDEX", "").lower() in (
            "1", "true", "yes", "on"
        ):
            logger.warning(
                "[REINDEX_PROGRESS] auto_reindex_if_needed: SKIPPED via "
                "CODE_SEARCH_DISABLE_AUTO_REINDEX project=%s",
                project_name,
            )
            return IncrementalIndexResult(
                files_added=0,
                files_removed=0,
                files_modified=0,
                chunks_added=0,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=True,
                reindex_disposition=REINDEX_DISPOSITION_SKIPPED_DISABLED,
            )

        if self.needs_reindex(project_path, max_age_minutes):
            logger.warning(
                "[REINDEX_PROGRESS] auto_reindex_if_needed: needs_reindex "
                "project=%s max_age_minutes=%s -> dispatching to incremental_index",
                project_name, max_age_minutes,
            )
            return self.incremental_index(project_path, project_name)
        else:
            logger.debug(f"Index for {project_path} is fresh, skipping reindex")
            return IncrementalIndexResult(
                files_added=0,
                files_removed=0,
                files_modified=0,
                chunks_added=0,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=True,
                reindex_disposition=REINDEX_DISPOSITION_SKIPPED_FRESH,
            )
