"""Job terminal-state mapping: a failed index result must not read "completed".

Regression guard for the 2026-06-12 P1 arm-2 incident: a network outage
dropped 11 embedding batches, IncrementalIndexResult returned success=False,
but the background job still set status="completed" — a downstream eval
polled "completed", ran against the half-index, and measured a phantom
retrieval collapse. Pollers key on the status string; it must honor the
result.
"""
from mcp_server.code_search_server import _job_terminal_state


def test_success_maps_to_completed_done():
    assert _job_terminal_state(True) == ("completed", "done")


def test_failure_maps_to_failed_error():
    # The load-bearing case: partial-index runs (failed embedding batches,
    # snapshot held back) end with success=False and MUST surface as the
    # existing "failed" status every poller already handles — never as
    # "completed", and never as a new enum value pollers would spin on.
    assert _job_terminal_state(False) == ("failed", "error")
