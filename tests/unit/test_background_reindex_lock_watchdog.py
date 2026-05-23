"""Tests for R4 + R5: background-reindex lock + watchdog.

R4 closes the TOCTOU window at the `_background_reindex_active` check-and-set
in CodeSearchServer._dispatch_background_reindex. Pre-fix, two concurrent
search_code calls observing `active=False` could both enter and dispatch
duplicate reindex threads.

R5 adds a watchdog so a stuck reindex (crashed thread between line ~596's
`finally` and process restart, hung Merkle walk, API stall) doesn't pin the
flag at True forever — every subsequent search would otherwise report
`stale_reindex_in_progress` until the process restarted.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from mcp_server.code_search_server import CodeSearchServer


@pytest.fixture
def server():
    """A bare CodeSearchServer instance — we don't index anything; we just
    exercise the dispatch path's locking + watchdog logic."""
    s = CodeSearchServer()
    # Set a project so the dispatch doesn't short-circuit on None.
    s._current_project = "/tmp/fake_project_for_dispatch_tests"
    yield s


# ---------------------------------------------------------------------------
# R4 — lock closes the TOCTOU window
# ---------------------------------------------------------------------------

class TestDispatchLockClosesTOCTOU:
    """Two concurrent dispatch calls must NOT both start a reindex thread."""

    def test_only_one_dispatch_wins_under_contention(self, server, monkeypatch):
        """Stub the actual reindex work so the lock-window is the only thing
        being exercised. Spawn N threads racing to dispatch; assert exactly
        one wins."""
        dispatches_won = []
        worker_started = threading.Event()
        worker_can_finish = threading.Event()

        def fake_run():
            worker_started.set()
            # Block until the test releases us — this maximizes contention:
            # other threads attempting dispatch must see active=True and
            # be rejected by the lock.
            worker_can_finish.wait(timeout=5)

        # Replace _run inside _dispatch_background_reindex by patching the
        # IncrementalIndexer construction. Simpler: patch threading.Thread
        # so we control when work happens, and capture how many threads
        # were created.
        threads_created = []
        real_thread_init = threading.Thread.__init__

        def thread_init(self, *args, **kwargs):
            real_thread_init(self, *args, **kwargs)
            # Replace target with our blocking stub.
            if kwargs.get("name") == "bg-reindex":
                self._target = fake_run
                threads_created.append(self)

        monkeypatch.setattr(threading.Thread, "__init__", thread_init)

        # Race N threads.
        N = 10
        results: list[bool] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(N)

        def caller():
            barrier.wait()  # synchronize start
            won = server._dispatch_background_reindex(
                server._current_project, max_age_minutes=5,
            )
            with results_lock:
                results.append(won)

        callers = [threading.Thread(target=caller) for _ in range(N)]
        for t in callers:
            t.start()
        for t in callers:
            t.join(timeout=5)

        # Let the worker (if any started) finish.
        worker_can_finish.set()

        assert sum(results) == 1, (
            f"expected exactly one winning dispatch under N={N} contention, "
            f"got {sum(results)}. Results: {results}"
        )
        # Pre-fix would create N threads (TOCTOU); post-fix creates exactly 1.
        assert len(threads_created) == 1, (
            f"expected exactly one bg-reindex thread created, got "
            f"{len(threads_created)}"
        )


# ---------------------------------------------------------------------------
# R5 — watchdog releases a stuck flag
# ---------------------------------------------------------------------------

class TestWatchdogReleasesStuckFlag:
    """A reindex that hangs past BG_REINDEX_WATCHDOG_SECONDS must not pin
    `_background_reindex_active` forever. The next dispatch attempt detects
    the deadline overrun and forcibly releases the flag, then proceeds."""

    def test_watchdog_lets_a_new_dispatch_through(self, server, monkeypatch):
        """Set the watchdog to a very short value, simulate a stuck thread,
        wait past the deadline, dispatch again. The second dispatch must
        succeed and emit the watchdog warning."""
        # Tighten the watchdog so we don't have to wait 30 min.
        monkeypatch.setattr(server, "BG_REINDEX_WATCHDOG_SECONDS", 0.05)

        # Simulate a stuck thread: set the active flag manually, with a
        # start time in the past. No actual thread runs.
        with server._background_reindex_lock:
            server._background_reindex_active = True
            server._background_reindex_started_at = time.monotonic() - 1.0

        # First sanity check: a dispatch attempt RIGHT NOW (before watchdog
        # fires? — actually with started_at=now-1.0 and watchdog=0.05, the
        # watchdog SHOULD fire immediately. Confirm.).
        threads_created = []

        def stub_target():
            pass  # do nothing

        # Stub the real reindex work so the dispatch returns quickly.
        real_thread_init = threading.Thread.__init__

        def thread_init(self, *args, **kwargs):
            real_thread_init(self, *args, **kwargs)
            if kwargs.get("name") == "bg-reindex":
                self._target = stub_target
                threads_created.append(self)

        monkeypatch.setattr(threading.Thread, "__init__", thread_init)

        # The previous "thread" is past the watchdog deadline → next
        # dispatch should release the flag and start a fresh thread.
        result = server._dispatch_background_reindex(
            server._current_project, max_age_minutes=5,
        )
        assert result is True, (
            "watchdog must release a stuck flag and let a new dispatch through"
        )
        # And a real thread was created.
        assert len(threads_created) == 1

    def test_watchdog_does_not_release_a_fresh_reindex(self, server, monkeypatch):
        """A reindex that has only been active for a moment (well within
        the watchdog deadline) must NOT be killed off — the watchdog
        only fires past the deadline."""
        # Generous watchdog; the dispatch we're simulating is "fresh".
        monkeypatch.setattr(server, "BG_REINDEX_WATCHDOG_SECONDS", 60.0)

        with server._background_reindex_lock:
            server._background_reindex_active = True
            server._background_reindex_started_at = time.monotonic()

        threads_created = []
        real_thread_init = threading.Thread.__init__

        def thread_init(self, *args, **kwargs):
            real_thread_init(self, *args, **kwargs)
            if kwargs.get("name") == "bg-reindex":
                threads_created.append(self)

        monkeypatch.setattr(threading.Thread, "__init__", thread_init)

        # Within deadline → dispatch should return False (busy).
        result = server._dispatch_background_reindex(
            server._current_project, max_age_minutes=5,
        )
        assert result is False, (
            "dispatch within watchdog deadline must return False; "
            "watchdog must not kill off a fresh reindex"
        )
        assert len(threads_created) == 0


# ---------------------------------------------------------------------------
# Finally-clause clears the flag even on exception
# ---------------------------------------------------------------------------

class TestFinallyClearsFlagOnException:
    """If the reindex worker raises, the `finally` block must still release
    `_background_reindex_active` so subsequent dispatches aren't blocked."""

    def test_exception_in_worker_releases_flag(self, server, monkeypatch):
        # Force the worker to raise immediately.
        from search import incremental_indexer

        class BoomIndexer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("boom — simulated worker crash")

        monkeypatch.setattr(
            incremental_indexer, "IncrementalIndexer", BoomIndexer,
        )

        # Stub embedder/get_index_manager so we don't blow up before the
        # IncrementalIndexer line.
        monkeypatch.setattr(server, "get_index_manager", lambda *a, **kw: object())
        monkeypatch.setattr(server, "embedder", lambda *a, **kw: object())

        # Patch MultiLanguageChunker the same way to be safe.
        from mcp_server import code_search_server as csm
        monkeypatch.setattr(csm, "MultiLanguageChunker", lambda *a, **kw: object())

        result = server._dispatch_background_reindex(
            server._current_project, max_age_minutes=5,
        )
        assert result is True

        # The worker raises immediately; wait briefly for the daemon to
        # complete its `finally` block.
        t = server._background_reindex_thread
        t.join(timeout=5)
        assert not t.is_alive(), "background thread should have exited"

        # Flag must be cleared.
        assert server._background_reindex_active is False, (
            "finally clause must release _background_reindex_active even "
            "when the worker raises"
        )
        assert server._background_reindex_started_at is None
