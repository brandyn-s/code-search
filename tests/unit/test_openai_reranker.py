"""RERANKER=openai: pointwise reranking through any OpenAI-compatible chat model."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from search import openai_reranker as oai
from search.llm_judge import JUDGE_PROMPT, build_judge_prompt, parse_score


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "RERANKER", "RERANKER_LLM_MODEL", "RERANKER_LLM_BASE_URL", "RERANKER_LLM_API_KEY",
        "RERANKER_LLM_TIMEOUT_S", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_AUTH_HEADER",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "SONNET_RERANKER_POOL_SIZE",
        "SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)
    from search.config import get_search_config

    get_search_config.cache_clear()
    yield
    get_search_config.cache_clear()


def _candidates(n: int = 4) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "file_path": f"src/mod{i}.py", "full_content": f"def f{i}(): pass"}
        for i in range(n)
    ]


class FakeChatServer:
    """Stands in for POST /chat/completions; scores come from ``scores_by_file``."""

    def __init__(self, scores_by_file: dict[str, object], *, status: int = 200):
        self.scores_by_file = scores_by_file
        self.status = status
        self.requests: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None, **kwargs):
        self.requests.append({"url": url, "headers": headers, "body": json, "timeout": timeout})
        prompt = json["messages"][0]["content"]
        file_path = next(fp for fp in self.scores_by_file if f"(file: {fp})" in prompt)
        payload = self.scores_by_file[file_path]
        if isinstance(payload, Exception):
            raise payload
        response = MagicMock()
        response.status_code = self.status
        if self.status >= 400:
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "err", request=MagicMock(), response=response
            )
        else:
            response.raise_for_status.return_value = None
        text = payload if isinstance(payload, str) else __import__("json").dumps({"score": payload, "reasoning": "x"})
        response.json.return_value = {"choices": [{"message": {"role": "assistant", "content": text}}]}
        return response


def _serve(server: FakeChatServer):
    return patch("httpx.Client.post", new=server.post)


# ---------------------------------------------------------------- shared judge

def test_shared_prompt_is_the_historical_sonnet_prompt():
    from search import sonnet_reranker

    assert sonnet_reranker.JUDGE_PROMPT is JUDGE_PROMPT
    rendered = build_judge_prompt("auth flow", "a.py", "x" * 5000)
    assert "Query: auth flow" in rendered and "(file: a.py)" in rendered
    assert "x" * 4000 in rendered and "x" * 4001 not in rendered


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"score": 7, "reasoning": "ok"}', 7),
        ('```json\n{"score": 12}\n```', 10),
        ('{"score": -3}', 0),
        ("not json", None),
        ('{"reasoning": "no score"}', 0),
        ('{"score": "nine"}', None),
    ],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


# -------------------------------------------------------------- happy path

def test_scores_are_parsed_and_ordering_applied(monkeypatch):
    monkeypatch.setenv("RERANKER_LLM_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setenv("RERANKER_LLM_BASE_URL", "http://localhost:11434/v1/")
    server = FakeChatServer({"src/mod0.py": 3, "src/mod1.py": 9, "src/mod2.py": 6, "src/mod3.py": 9})

    with _serve(server):
        out, meta = oai.rerank_with_openai("query", _candidates(), top_k=3, return_metadata=True)

    assert [c["chunk_id"] for c in out] == ["c1", "c3", "c2"]  # stable on ties
    assert meta["applied"] is True and meta["reason"] == "ok"
    request = server.requests[0]
    assert request["url"] == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in request["headers"]  # self-hosted, no key
    assert request["body"]["model"] == "qwen2.5-coder:7b"
    assert request["body"]["temperature"] == 0
    assert request["body"]["max_tokens"] == oai.MAX_TOKENS
    assert request["timeout"] == pytest.approx(min(12.0, 8.0))


def test_bearer_key_and_base_url_fallbacks(monkeypatch):
    monkeypatch.setenv("RERANKER_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or")
    server = FakeChatServer({"src/mod0.py": 8, "src/mod1.py": 2})

    with _serve(server):
        out, meta = oai.rerank_with_openai("q", _candidates(2), top_k=1, return_metadata=True)

    assert meta["applied"] is True and out[0]["chunk_id"] == "c0"
    assert server.requests[0]["url"].startswith("https://openrouter.ai/api/v1/")
    assert server.requests[0]["headers"]["Authorization"] == "Bearer sk-or"


# ------------------------------------------------------------ failure paths

def test_malformed_json_preserves_hybrid_order_with_reason(monkeypatch):
    monkeypatch.setenv("RERANKER_LLM_MODEL", "m")
    monkeypatch.setenv("RERANKER_LLM_BASE_URL", "http://localhost:8000/v1")
    server = FakeChatServer({fp: "I think it is relevant" for fp in (f"src/mod{i}.py" for i in range(4))})

    with _serve(server):
        out, meta = oai.rerank_with_openai("q", _candidates(), top_k=4, return_metadata=True)

    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2", "c3"]
    assert meta["applied"] is False and meta["reason"] == "too_many_failures"


def test_timeouts_report_timeout_reason(monkeypatch):
    monkeypatch.setenv("RERANKER_LLM_MODEL", "m")
    monkeypatch.setenv("RERANKER_LLM_BASE_URL", "http://localhost:8000/v1")
    server = FakeChatServer({fp: httpx.ReadTimeout("slow") for fp in (f"src/mod{i}.py" for i in range(4))})

    with _serve(server):
        out, meta = oai.rerank_with_openai("q", _candidates(), top_k=4, return_metadata=True)

    assert meta["applied"] is False and meta["reason"] == "timeout"
    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2", "c3"]


def test_rate_limit_reason_and_partial_failures_tolerated(monkeypatch):
    monkeypatch.setenv("RERANKER_LLM_MODEL", "m")
    monkeypatch.setenv("RERANKER_LLM_BASE_URL", "http://localhost:8000/v1")
    # One of four fails (25% < 30% tolerance): rerank still applies, failure sinks.
    response = MagicMock()
    response.status_code = 429
    err = httpx.HTTPStatusError("429", request=MagicMock(), response=response)
    server = FakeChatServer({"src/mod0.py": err, "src/mod1.py": 9, "src/mod2.py": 7, "src/mod3.py": 8})
    with _serve(server):
        out, meta = oai.rerank_with_openai("q", _candidates(), top_k=4, return_metadata=True)
    assert meta["reason"] == "ok" and [c["chunk_id"] for c in out] == ["c1", "c3", "c2", "c0"]

    # All four rate-limited: reason is rate_limit.
    server = FakeChatServer({fp: err for fp in (f"src/mod{i}.py" for i in range(4))})
    with _serve(server):
        _, meta = oai.rerank_with_openai("q", _candidates(), top_k=4, return_metadata=True)
    assert meta["reason"] == "rate_limit"


def test_hybrid_prior_fallback_applies_to_openai_engine(monkeypatch):
    monkeypatch.setenv("RERANKER_LLM_MODEL", "m")
    monkeypatch.setenv("RERANKER_LLM_BASE_URL", "http://localhost:8000/v1")
    server = FakeChatServer({fp: 2 for fp in (f"src/mod{i}.py" for i in range(4))})
    with _serve(server):
        _, meta = oai.rerank_with_openai("q", _candidates(), top_k=4, return_metadata=True)
    assert meta["reason"] == "hybrid_prior_fallback"


def test_missing_model_returns_model_not_configured():
    out, meta = oai.rerank_with_openai("q", _candidates(), top_k=2, return_metadata=True)
    assert meta["reason"] == oai.REASON_MODEL_NOT_CONFIGURED and meta["applied"] is False
    assert [c["chunk_id"] for c in out] == ["c0", "c1"]
    assert "RERANKER_LLM_MODEL is not set" in oai.preflight_problems()


def test_openai_host_without_key_is_api_key_missing(monkeypatch):
    monkeypatch.setenv("RERANKER_LLM_MODEL", "gpt-4o-mini")
    _, meta = oai.rerank_with_openai("q", _candidates(), top_k=2, return_metadata=True)
    assert meta["reason"] == "api_key_missing"
    assert any("RERANKER_LLM_API_KEY" in p for p in oai.preflight_problems())


# ------------------------------------------------------- config and dispatch

def test_openai_is_a_valid_mode_but_auto_never_selects_it(monkeypatch):
    from search.config import RERANKER_MODES, get_search_config

    assert "openai" in RERANKER_MODES
    monkeypatch.setenv("RERANKER", "openai")
    assert get_search_config().reranker_mode == "openai"

    get_search_config.cache_clear()
    monkeypatch.delenv("RERANKER")
    monkeypatch.setenv("RERANKER_LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert get_search_config().reranker_mode == "off"  # auto stays sonnet-or-off


def test_registry_dispatch_uses_openai_engine(monkeypatch):
    from search.reranker_registry import get_reranker
    from search.result_models import SearchResult

    calls = {}

    def fake_rerank(query, candidates, top_k, return_metadata):
        calls["n"] = len(candidates)
        return list(reversed(candidates))[:top_k], {"applied": True, "reason": "ok", "latency_ms": 1}

    monkeypatch.setattr("search.openai_reranker.rerank_with_openai", fake_rerank)
    results = []
    for i in range(3):
        r = MagicMock(spec=SearchResult)
        r.chunk_id = f"c{i}"
        r.relative_path = f"f{i}.py"
        r.content_preview = "x"
        r.similarity_score = 1.0 - i / 10
        results.append(r)
    searcher = MagicMock()
    config = MagicMock(sonnet_skip_threshold=None)
    out = get_reranker("openai")(searcher, "q", k=2, config=config, candidates=results, metadata_lookup={})
    assert calls["n"] == 3 and [r.chunk_id for r in out] == ["c2", "c1"]
    assert searcher.last_reranker_metadata["reason"] == "ok"


def test_startup_preflight_warns_when_model_missing(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("RERANKER", "openai")
    monkeypatch.setenv("RERANKER_LLM_BASE_URL", "http://localhost:8000/v1")
    from common_utils import get_storage_dir
    from mcp_server.code_search_server import CodeSearchServer

    get_storage_dir.cache_clear()
    with caplog.at_level(logging.WARNING, logger="mcp_server.code_search_server"):
        CodeSearchServer()
    get_storage_dir.cache_clear()
    assert any("RERANKER=openai" in r.getMessage() and "RERANKER_LLM_MODEL" in r.getMessage() for r in caplog.records)


def test_startup_line_names_openai_model(monkeypatch, capsys):
    monkeypatch.setenv("RERANKER", "openai")
    monkeypatch.setenv("RERANKER_LLM_MODEL", "qwen2.5-coder:7b")
    from mcp_server.server import _log_startup_mode

    _log_startup_mode()
    assert "reranker=openai(qwen2.5-coder:7b)" in capsys.readouterr().err


# ------------------------------------------------ Anthropic model override

def test_sonnet_pointwise_model_is_configurable(monkeypatch):
    from search import sonnet_reranker as sr

    assert sr._resolve_model() == sr.DEFAULT_MODEL == "claude-sonnet-4-6"
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    assert sr._resolve_model() == "claude-haiku-4-5-20251001"

    class FakeMessages:
        def __init__(self):
            self.models = []

        async def create(self, **kwargs):
            self.models.append(kwargs["model"])
            block = MagicMock()
            block.type = "text"
            block.text = json.dumps({"score": 8})
            return MagicMock(content=[block])

    client = MagicMock()
    client.messages = FakeMessages()
    import asyncio

    score = asyncio.run(sr._score_one(client, "q", "a.py", "content"))
    assert score == 8 and client.messages.models == ["claude-haiku-4-5-20251001"]
