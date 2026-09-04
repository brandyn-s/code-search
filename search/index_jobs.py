"""Background indexing job state, extracted from the MCP server class.

Two small state machines live here:

* :class:`IndexingJobState` owns the single foreground ``index_directory`` job:
  its lock, the job record, cancellation, progress callbacks, the
  "already indexing" conflict response, and the ``get_indexing_progress``
  payload.
* :class:`BackgroundReindexGuard` owns the flag, start timestamp, and
  watchdog for the search-time background reindex thread
  (``CODE_SEARCH_NONBLOCKING_SEARCH``).

Neither class starts threads or touches the index; the server composes them.
Responses are returned as dicts; the server serialises them to JSON so the
wire format is unchanged.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = ("completed", "failed", "cancelled")


class IndexingJobState:
    """Process-global foreground indexing job (at most one at a time)."""

    def __init__(self, lock: Optional[Any] = None) -> None:
        self._lock = lock if lock is not None else threading.RLock()
        self.job: Optional[Dict[str, Any]] = None

    # --- primitives -------------------------------------------------------

    @property
    def lock(self) -> Any:
        return self._lock

    @lock.setter
    def lock(self, value: Any) -> None:
        self._lock = value

    def snapshot(self) -> Optional[Dict[str, Any]]:
        """Return one coherent copy of the current job, or ``None``."""
        with self._lock:
            if self.job is None:
                return None
            return dict(self.job)

    def update(self, job_id: str, **updates: Any) -> bool:
        """Atomically update only the job owned by ``job_id``."""
        with self._lock:
            if not self.job or self.job.get("job_id") != job_id:
                return False
            self.job.update(updates)
            return True

    def is_active(self) -> bool:
        with self._lock:
            return bool(self.job and self.job.get("status") == "indexing")

    def request_cancel(self) -> Optional[str]:
        """Mark the active job for cancellation; returns its id or ``None``."""
        with self._lock:
            job = self.job
            if not job or job.get("status") != "indexing":
                return None
            job["cancel_requested"] = True
            return job["job_id"]

    def cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self.job
            return bool(job and job.get("job_id") == job_id and job.get("cancel_requested"))

    def progress_callback(self, job_id: str) -> Callable[[str, int, int], None]:
        """Build the ``progress_fn`` handed to the incremental indexer.

        Records phase/current/total for ``job_id`` and raises
        ``InterruptedError`` when cancellation was requested, which is how the
        indexing thread learns to stop at the next checkpoint.
        """

        def _progress(phase: str, current: int, total: int) -> None:
            with self._lock:
                job = self.job
                if not job or job.get("job_id") != job_id:
                    return
                job.update({"phase": phase, "current": current, "total": total})
                cancel = bool(job.get("cancel_requested"))
            if cancel:
                raise InterruptedError("Indexing cancelled by user")

        return _progress

    # --- responses --------------------------------------------------------

    def active_conflict_response(
        self,
        directory_path: str,
        provider: Optional[str],
        plan_storage_target: Callable[[str, Optional[str]], Path],
    ) -> Optional[Dict[str, Any]]:
        """Describe the active job when a new request would collide with it.

        ``plan_storage_target(directory, provider)`` resolves where an index
        for that request would live, so two requests for the same checkout
        and provider are recognised as the same job.
        """
        active_job = self.job
        if not active_job or active_job["status"] != "indexing":
            return None

        requested_directory = str(Path(directory_path).resolve())
        active_directory = str(active_job.get("directory", ""))
        active_provider_value = active_job.get("provider")
        active_provider = (
            active_provider_value.strip().lower()
            if isinstance(active_provider_value, str) and active_provider_value.strip()
            else None
        )
        active_storage_target = str(
            Path(
                active_job.get("storage_target")
                or plan_storage_target(active_directory, active_provider)
            ).resolve()
        )
        requested_storage_target = str(
            plan_storage_target(requested_directory, provider).resolve()
        )
        directory_conflict = requested_directory != active_directory
        provider_conflict = provider != active_provider
        storage_target_conflict = requested_storage_target != active_storage_target
        indexing_conflict = directory_conflict or provider_conflict or storage_target_conflict
        active_project = str(
            active_job.get("project_name") or Path(active_directory).name or "unknown"
        )
        requested_project = Path(requested_directory).name or "unknown"
        if indexing_conflict:
            message = (
                f"Indexing job {active_job['job_id']} is already active "
                f"for {active_project}; request for {requested_project} "
                "did not start another job"
            )
        else:
            message = (
                f"Indexing already in progress for {active_project}; "
                "reusing the active job"
            )
        response: Dict[str, Any] = {
            "status": "indexing",
            "index_ready": False,
            "message": message,
            "job_id": active_job["job_id"],
            "phase": active_job.get("phase", "unknown"),
            "chunks_done": active_job.get("current", 0),
            "chunks_total": active_job.get("total", 0),
            "directory": active_directory,
            "project_name": active_project,
            "provider": active_provider,
            "storage_target": active_storage_target,
            "requested_directory": requested_directory,
            "requested_provider": provider,
            "requested_storage_target": requested_storage_target,
            "indexing_conflict": indexing_conflict,
        }
        if directory_conflict:
            response["conflict_reason"] = "different_project_indexing"
        elif provider_conflict:
            response["conflict_reason"] = "different_provider_indexing"
        elif storage_target_conflict:
            response["conflict_reason"] = "different_storage_target_indexing"
        return response

    def progress_response(self, background_reindex_active: bool) -> Dict[str, Any]:
        """Payload for ``get_indexing_progress``.

        Status values: ``idle``, ``indexing``, ``completed``/``failed``/
        ``cancelled`` (with ``result``), or ``background_reindex_active`` when
        only the search-time reindex thread is running.
        """
        job = self.snapshot()
        if not job:
            if background_reindex_active:
                return {
                    "status": "background_reindex_active",
                    "index_ready": False,
                    "background_reindex_active": True,
                    "message": (
                        "Search-time background reindex in progress "
                        "(CODE_SEARCH_NONBLOCKING_SEARCH=1). search_code "
                        "calls return last-good-index results with "
                        "_metadata.freshness=stale_reindex_in_progress "
                        "until this completes."
                    ),
                }
            return {
                "status": "idle",
                "index_ready": False,
                "background_reindex_active": False,
                "message": "No indexing job running",
            }

        response: Dict[str, Any] = {
            "job_id": job["job_id"],
            "status": job["status"],
            "phase": job["phase"],
            "directory": job.get("directory", ""),
            "project_name": job.get("project_name", ""),
            "provider": job.get("provider"),
            "storage_target": job.get("storage_target"),
            "background_reindex_active": background_reindex_active,
            "index_ready": bool(job.get("index_ready", False)),
        }
        if job["total"] > 0:
            response["chunks_done"] = job["current"]
            response["chunks_total"] = job["total"]
            response["percent"] = round(100 * job["current"] / job["total"], 1)
            # chunks_done/chunks_total count FILES during chunking and
            # removing and CHUNKS during embedding and saving; label the unit
            # so the total does not look like it jumps mid-job.
            phase = job.get("phase", "")
            if phase in ("chunking", "removing"):
                response["unit"] = "files"
            elif phase in ("embedding", "saving"):
                response["unit"] = "chunks"

        if job["status"] in TERMINAL_STATUSES and job.get("result"):
            terminal_result = job["result"]
            if isinstance(terminal_result, dict):
                terminal_result = {
                    **terminal_result,
                    **{key: job[key] for key in ("provider", "storage_target") if key in job},
                }
            response["result"] = terminal_result
        return response


class BackgroundReindexGuard:
    """Check-and-set flag for the search-time background reindex thread.

    ``try_acquire`` returns ``False`` while a reindex is in flight and younger
    than the watchdog deadline. A flag older than the deadline is assumed to
    belong to a stuck thread (crashed before ``finally``, hung walk) and is
    reclaimed so a fresh dispatch can proceed.
    """

    def __init__(self, lock: Optional[Any] = None) -> None:
        self.lock = lock if lock is not None else threading.Lock()
        self.active = False
        self.started_at: Optional[float] = None
        self.thread: Optional[Any] = None

    def try_acquire(self, now: float, watchdog_seconds: float) -> bool:
        with self.lock:
            if self.active:
                started = self.started_at or now
                age = now - started
                if age <= watchdog_seconds:
                    return False
                logger.warning(
                    "[F2-bg] watchdog: prior reindex 'active' for %.1fs (>%.0fs deadline); "
                    "releasing flag and dispatching fresh reindex. Stuck thread name=%s",
                    age,
                    watchdog_seconds,
                    getattr(self.thread, "name", "?"),
                )
            self.active = True
            self.started_at = now
            return True

    def release(self) -> None:
        with self.lock:
            self.active = False
            self.started_at = None
