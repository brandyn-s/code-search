"""Tests for the Anthropic pre-flight throughput probe in sweep_rrf_weights.

Phase B4 of plan #462. Verifies the probe correctly detects slow Anthropic
calls and returns ok=False, so the sweep aborts before committing to ~900
slow calls.
"""
from __future__ import annotations

import time
from unittest import mock


def test_probe_returns_ok_when_fast():
    """Probe under threshold returns ok=True."""
    from bench.research.sweep_rrf_weights import _probe_anthropic_latency

    def fake_rerank(*args, **kwargs):
        return kwargs.get("candidates", [])

    with mock.patch("search.sonnet_reranker.rerank_with_sonnet",
                    side_effect=fake_rerank):
        elapsed, ok = _probe_anthropic_latency(threshold_sec=5.0)
    assert ok is True
    assert elapsed < 5.0


def test_probe_returns_not_ok_when_slow():
    """Probe over threshold returns ok=False."""
    from bench.research.sweep_rrf_weights import _probe_anthropic_latency

    def fake_rerank_slow(*args, **kwargs):
        time.sleep(6.0)
        return kwargs.get("candidates", [])

    with mock.patch("search.sonnet_reranker.rerank_with_sonnet",
                    side_effect=fake_rerank_slow):
        elapsed, ok = _probe_anthropic_latency(threshold_sec=5.0)
    assert ok is False
    assert elapsed >= 5.0


def test_probe_returns_not_ok_on_exception():
    """Probe that raises returns ok=False with elapsed time captured."""
    from bench.research.sweep_rrf_weights import _probe_anthropic_latency

    def fake_rerank_raise(*args, **kwargs):
        raise RuntimeError("anthropic api down")

    with mock.patch("search.sonnet_reranker.rerank_with_sonnet",
                    side_effect=fake_rerank_raise):
        elapsed, ok = _probe_anthropic_latency(threshold_sec=5.0)
    assert ok is False
    assert elapsed >= 0.0


def test_probe_threshold_is_respected():
    """Custom threshold is honored — 3s threshold rejects 4s call."""
    from bench.research.sweep_rrf_weights import _probe_anthropic_latency

    def fake_rerank_4s(*args, **kwargs):
        time.sleep(4.0)
        return kwargs.get("candidates", [])

    with mock.patch("search.sonnet_reranker.rerank_with_sonnet",
                    side_effect=fake_rerank_4s):
        elapsed, ok = _probe_anthropic_latency(threshold_sec=3.0)
    assert ok is False
    assert elapsed >= 3.0
