"""Unit tests for listwise Sonnet reranker.

All tests mock the Anthropic API via the _client_factory test seam; no
network calls. Validates the always-on / never-raises contract and the
strict schema validation that prevents the reranker from emitting
ill-formed output.
"""
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from search.listwise_sonnet_reranker import (
    REASON_API_KEY_MISSING,
    REASON_EMPTY_INPUT,
    REASON_ID_MISMATCH,
    REASON_OK,
    REASON_PARSE_FAILED,
    REASON_RATE_LIMIT,
    REASON_TIMEOUT,
    REASON_UNEXPECTED_ERROR,
    _build_candidates_block,
    _validate_response,
    listwise_rerank_with_sonnet,
)


# ---- Helpers -------------------------------------------------------------

def make_candidates(n: int = 5) -> list[dict]:
    """Build n test candidates with rich metadata."""
    return [
        {
            "chunk_id": f"chunk_{i}",
            "file_path": f"src/module_{i}/file_{i}.py",
            "name": f"function_{i}",
            "parent_name": "TestClass" if i % 2 == 0 else None,
            "chunk_type": "function",
            "start_line": 10 * i + 1,
            "end_line": 10 * i + 8,
            "content_preview": f"def function_{i}():\n    return {i}\n",
            "similarity_score": 0.9 - 0.1 * i,
        }
        for i in range(n)
    ]


def mock_anthropic_client(json_payload: dict | str, raise_exc: Exception | None = None):
    """Build a mock Anthropic client returning the given JSON payload.

    If raise_exc is set, messages.create raises that exception instead.
    """
    if isinstance(json_payload, dict):
        text = json.dumps(json_payload)
    else:
        text = json_payload

    def factory():
        client = MagicMock()
        if raise_exc is not None:
            client.messages.create.side_effect = raise_exc
        else:
            resp = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
            client.messages.create.return_value = resp
        return client
    return factory


# ---- Test cases ---------------------------------------------------------

def test_empty_input_returns_empty():
    out, meta = listwise_rerank_with_sonnet(
        "q", [], top_k=10, return_metadata=True,
        _client_factory=mock_anthropic_client({}),
    )
    assert out == []
    assert meta["applied"] is False
    assert meta["reason"] == REASON_EMPTY_INPUT


def test_missing_api_key_falls_back_to_baseline_order(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cands = make_candidates(5)
    # Pass _client_factory=None to force the env-var check path
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        _client_factory=None,
    )
    assert [c["chunk_id"] for c in out] == ["chunk_0", "chunk_1", "chunk_2"]
    assert meta["applied"] is False
    assert meta["reason"] == REASON_API_KEY_MISSING


def test_happy_path_reorders_candidates(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    # Model presents candidates as C01,C02,C03 (NOT shuffled) and returns reversed
    payload = {
        "ranked_ids": ["C03", "C02", "C01"],
        "scores": {"C01": 2, "C02": 5, "C03": 9},
    }
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        shuffle_seed=None,  # no shuffle so C01..C03 maps to cands[0..2]
        _client_factory=mock_anthropic_client(payload),
    )
    assert meta["applied"] is True
    assert meta["reason"] == REASON_OK
    # Reversed order: cands[2], cands[1], cands[0]
    assert [c["chunk_id"] for c in out] == ["chunk_2", "chunk_1", "chunk_0"]


def test_listwise_respects_shuffle_seed(monkeypatch):
    """Same shuffle_seed must yield same C-id assignment across calls.

    Position-bias defense: the model sees a permuted order, but the
    output must map back to original candidate identities correctly.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(5)
    # Two calls with same seed produce the same id_to_orig mapping
    _, mapping_a = _build_candidates_block(cands, shuffle_seed=42)
    _, mapping_b = _build_candidates_block(cands, shuffle_seed=42)
    assert mapping_a == mapping_b
    # Two calls with different seeds produce different mappings
    _, mapping_c = _build_candidates_block(cands, shuffle_seed=99)
    assert mapping_a != mapping_c
    # No-shuffle yields identity mapping
    _, mapping_id = _build_candidates_block(cands, shuffle_seed=None)
    assert [orig for _, orig in mapping_id] == list(range(5))


def test_shuffled_response_maps_back_to_original_candidates(monkeypatch):
    """The C-id -> original-index mapping must work under shuffle."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    # With shuffle_seed=1: rebuild the mapping to know what C01..C03 map to
    _, mapping = _build_candidates_block(cands, shuffle_seed=1)
    # mapping is [("C01", orig_a), ("C02", orig_b), ("C03", orig_c)]
    # Model picks C03 first (whichever original that is)
    chosen_first_cid = mapping[2][0]  # "C03"
    chosen_first_orig = mapping[2][1]
    payload = {
        "ranked_ids": [chosen_first_cid, mapping[0][0], mapping[1][0]],
        "scores": {mapping[i][0]: 5 for i in range(3)},
    }
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        shuffle_seed=1,
        _client_factory=mock_anthropic_client(payload),
    )
    assert meta["applied"] is True
    # First result must be the original candidate at chosen_first_orig
    assert out[0]["chunk_id"] == cands[chosen_first_orig]["chunk_id"]


def test_invalid_json_falls_back_with_parse_failed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        _client_factory=mock_anthropic_client("not json at all"),
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_PARSE_FAILED
    # Baseline order preserved
    assert [c["chunk_id"] for c in out] == ["chunk_0", "chunk_1", "chunk_2"]


def test_missing_id_falls_back_with_id_mismatch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    # Schema says all 3 IDs must appear; model returns only 2
    payload = {"ranked_ids": ["C01", "C02"], "scores": {"C01": 3, "C02": 7}}
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        shuffle_seed=None,
        _client_factory=mock_anthropic_client(payload),
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_ID_MISMATCH


def test_duplicate_id_falls_back_with_id_mismatch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    payload = {
        "ranked_ids": ["C01", "C01", "C02"],
        "scores": {"C01": 5, "C02": 7, "C03": 1},
    }
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        shuffle_seed=None,
        _client_factory=mock_anthropic_client(payload),
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_ID_MISMATCH


def test_unknown_id_falls_back_with_id_mismatch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    # C99 is not in the input
    payload = {
        "ranked_ids": ["C01", "C99", "C03"],
        "scores": {"C01": 5, "C99": 9, "C03": 3},
    }
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        shuffle_seed=None,
        _client_factory=mock_anthropic_client(payload),
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_ID_MISMATCH


def test_rate_limit_exception_falls_back_with_rate_limit_reason(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    exc = Exception("rate_limit_error: too many requests")
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        _client_factory=mock_anthropic_client({}, raise_exc=exc),
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_RATE_LIMIT


def test_timeout_exception_falls_back_with_timeout_reason(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    exc = Exception("request timed out after 12.0s")
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        _client_factory=mock_anthropic_client({}, raise_exc=exc),
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_TIMEOUT


def test_arbitrary_exception_falls_back_with_unexpected_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    exc = ValueError("something inexplicable")
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        _client_factory=mock_anthropic_client({}, raise_exc=exc),
    )
    assert meta["applied"] is False
    assert meta["reason"] == REASON_UNEXPECTED_ERROR


def test_code_fence_wrapped_json_still_parses(monkeypatch):
    """Sonnet sometimes wraps JSON in ```json ... ``` despite system prompt.

    The validator strips code fences before parsing.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    wrapped = '```json\n{"ranked_ids":["C03","C02","C01"],"scores":{"C01":2,"C02":5,"C03":9}}\n```'
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        shuffle_seed=None,
        _client_factory=mock_anthropic_client(wrapped),
    )
    assert meta["applied"] is True
    assert [c["chunk_id"] for c in out] == ["chunk_2", "chunk_1", "chunk_0"]


def test_top_k_truncates_output():
    out = listwise_rerank_with_sonnet(
        "q", make_candidates(5), top_k=2,
        _client_factory=mock_anthropic_client(
            {"ranked_ids": ["C01", "C02", "C03", "C04", "C05"],
             "scores": {f"C0{i}": 5 for i in range(1, 6)}}
        ),
    )
    # Note: no env var set, may fall through to API_KEY_MISSING and return cands[:2]
    # Either way, length must be top_k
    assert len(out) == 2


def test_never_raises_under_pathological_input():
    """The contract: this module never raises.

    Throw the worst-case input at it and verify return is well-formed.
    """
    pathological = [{"file_path": None, "content_preview": ""}, {}, {"name": None}]
    # No API key -> immediate fallback path. Must not raise.
    out = listwise_rerank_with_sonnet("", pathological, top_k=10)
    assert isinstance(out, list)
    assert len(out) == len(pathological)


def test_validate_response_rejects_non_dict():
    """_validate_response: top-level must be dict."""
    cands = make_candidates(2)
    _, id_to_orig = _build_candidates_block(cands, shuffle_seed=None)
    ordered, err = _validate_response('["just", "an", "array"]', id_to_orig)
    assert ordered is None
    assert err == REASON_PARSE_FAILED


def test_validate_response_rejects_non_list_ranked_ids():
    """ranked_ids must be a list of strings."""
    cands = make_candidates(2)
    _, id_to_orig = _build_candidates_block(cands, shuffle_seed=None)
    ordered, err = _validate_response('{"ranked_ids": "not a list", "scores": {}}', id_to_orig)
    assert ordered is None
    assert err == REASON_PARSE_FAILED


def test_validate_response_happy_path():
    """Well-formed response returns ordered original indices."""
    cands = make_candidates(3)
    _, id_to_orig = _build_candidates_block(cands, shuffle_seed=None)
    payload = json.dumps({
        "ranked_ids": ["C02", "C03", "C01"],
        "scores": {"C01": 1, "C02": 9, "C03": 5},
    })
    ordered, err = _validate_response(payload, id_to_orig)
    assert err is None
    # C01 -> orig 0, C02 -> orig 1, C03 -> orig 2 (no-shuffle mapping)
    # Response order: C02, C03, C01 -> ordered_origs = [1, 2, 0]
    assert ordered == [1, 2, 0]


def test_never_raises_meta_envelope_shape(monkeypatch):
    """return_metadata=True must always return (list, dict) with applied/reason/latency_ms."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cands = make_candidates(3)
    out, meta = listwise_rerank_with_sonnet(
        "q", cands, top_k=3, return_metadata=True,
        _client_factory=mock_anthropic_client("garbage"),
    )
    assert isinstance(out, list)
    assert set(meta.keys()) == {"applied", "reason", "latency_ms"}
    assert isinstance(meta["applied"], bool)
    assert isinstance(meta["reason"], str)
    assert isinstance(meta["latency_ms"], int)
