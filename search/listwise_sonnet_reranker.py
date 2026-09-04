"""Listwise Sonnet 4.6 reranker — single comparative call replacing pointwise.

Replaces the pointwise reranker (15 isolated `_rerank_async` calls in
`sonnet_reranker.py`) with ONE listwise call that compares all candidates
simultaneously and returns ordered IDs + scores.

Hypothesis (per GPT-5.5-pro session pass 2, D section): listwise closes
billing regression (CI [-0.16, -0.01] vs hybrid), webapp over-rotation
(CI [+0.02, +0.33]), slowest-of-15 latency tail, and arbitrary-tie behavior
— in one architectural change. Retires
`SONNET_RERANKER_HYBRID_PRIOR_THRESHOLD_PATH_OVERRIDES` if it ships.

ALWAYS-ON CONTRACT: this module never raises. Any failure (missing API key,
timeout, HTTP error, invalid JSON, schema mismatch, duplicate/missing IDs,
exception) returns the input candidates in baseline order.

PoC stage: not yet production-default. Activated via
`RERANKER=listwise` env var. Mirrors `sonnet_reranker.rerank_with_sonnet`
signature so the dispatcher in `searcher.py` can swap modules.

Cost: ~$0.04-0.05/query at 15 candidates (~10K input tokens × $1.50/M).
Comparable to pointwise's 15×$0.005, slightly more due to larger prompt.

Critical constraint (per GPT pass 2 F.5): listwise MUST be a REPLACEMENT
arm, not additive. The dispatcher must choose between pointwise OR
listwise, never both.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Sequence

from search.logging_privacy import (
    format_query_exception_for_log,
    query_text_logging_enabled,
)
from search.env import env_get

LOG = logging.getLogger(__name__)

# Reuse the same reason vocabulary as the pointwise reranker for
# _metadata.reranker observability continuity. Plus 2 listwise-specific
# reasons for schema-failure paths.
REASON_OK = "ok"
REASON_EMPTY_INPUT = "empty_input"
REASON_API_KEY_MISSING = "api_key_missing"
REASON_PACKAGE_NOT_INSTALLED = "package_not_installed"
REASON_TIMEOUT = "timeout"
REASON_RATE_LIMIT = "rate_limit"
REASON_PARSE_FAILED = "parse_failed"          # listwise-specific: JSON / schema violation
REASON_ID_MISMATCH = "id_mismatch"            # listwise-specific: missing / duplicate / extra IDs
REASON_UNEXPECTED_ERROR = "unexpected_error"

DEFAULT_TIMEOUT_S = 12.0
DEFAULT_MODEL = "claude-sonnet-4-6"
SNIPPET_MAX_LINES = 10  # cap each candidate snippet preview to keep prompt compact

# JSON Schema for the listwise output. Strict validation: any deviation -> fallback.
# Format: {"ranked_ids": ["C03","C01","C02",...], "scores": {"C01":4,"C02":7,...}}

SYSTEM_PROMPT = (
    "You are a code-search reranker. Given a user query and candidate code "
    "snippets from one repository, rank the candidates by how likely each is "
    "to be the file/region the user should open to answer or edit the code. "
    "Use only the provided candidates. Do not answer the query. Do not invent "
    "candidate IDs. Return strict JSON only — no prose, no markdown."
)

RANKING_RUBRIC = """
RANKING RUBRIC:
- Prefer candidates that directly implement the queried behavior.
- Exact symbol, function, class, method, file, or error-string matches are strong evidence.
- Implementation code usually ranks above call sites, tests, docs, or examples,
  unless the query explicitly asks for those.
- For Nix/NixOS/configuration queries, treat `option` and `binding` chunks as primary
  definitions. If the query asks about `mkOption`, `services.<name>`, enable /
  configuration / module setup, systemd service configuration, hardware configuration,
  or NixOS modules, prefer the Nix module or option declaration over daemon
  implementation code, tests, call sites, or docs. For service setup queries,
  `nix/modules/<service>.nix` is usually the implementation of the user-visible
  configuration surface.
- Prefer specific local evidence over broad topical similarity.
- If two candidates are equally relevant, preserve their lower baseline_rank.
- Every candidate ID MUST appear exactly once in ranked_ids.
"""

OUTPUT_SCHEMA_DESC = """
OUTPUT JSON SCHEMA (return this and nothing else):
{
  "ranked_ids": ["C03", "C01", "C02", ...],   // every input ID, exactly once
  "scores":     { "C01": 4, "C02": 7, ... }   // integer 0-10 per ID
}
"""


def _candidate_id(i: int) -> str:
    """Stable candidate ID format. C01..C99."""
    return f"C{i+1:02d}"


def _extract_snippet(cand: dict) -> str:
    """Pull snippet preview from candidate; cap to SNIPPET_MAX_LINES lines.

    Accepts dataclass-shape (content_preview) and MCP-JSON-shape (snippet).
    """
    body = (
        cand.get("content_preview")
        or cand.get("snippet")
        or cand.get("content")
        or cand.get("full_content")
        or ""
    )
    lines = body.splitlines()
    if len(lines) > SNIPPET_MAX_LINES:
        lines = lines[:SNIPPET_MAX_LINES] + [f"...(truncated, {len(body)} chars total)"]
    return "\n".join(lines)


def _extract_path(cand: dict) -> str:
    return (
        cand.get("file_path")
        or cand.get("file")
        or cand.get("relative_path")
        or "(unknown path)"
    )


def _extract_symbol(cand: dict) -> str:
    """Accepts dataclass-shape (chunk_type) and MCP-JSON-shape (kind)."""
    name = cand.get("name") or ""
    parent = cand.get("parent_name") or ""
    chunk_type = cand.get("chunk_type") or cand.get("kind") or ""
    if parent and name:
        return f"{chunk_type} {parent}.{name}".strip()
    if name:
        return f"{chunk_type} {name}".strip()
    return f"({chunk_type or 'chunk'})"


def _extract_lines(cand: dict) -> str:
    """Accepts both (start_line, end_line) tuple-shape and string "N-M" shape."""
    s = cand.get("start_line")
    e = cand.get("end_line")
    if s is not None and e is not None:
        return f"{s}-{e}"
    lines = cand.get("lines")
    if isinstance(lines, str) and lines:
        return lines
    return ""


def _build_candidates_block(
    candidates: Sequence[dict], shuffle_seed: int | None,
) -> tuple[str, list[tuple[str, int]]]:
    """Build the candidate-listing block. Returns (text, [(id, original_index), ...]).

    The original_index tracking lets us map ranked_ids back to original candidate
    positions after the model returns.

    shuffle_seed: if not None, shuffle candidate presentation order using this seed.
    The candidate IDs (C01..) follow PRESENTATION order, not original-rank order,
    so the model cannot infer baseline_rank from ID. baseline_rank is given
    explicitly in the prompt.
    """
    indices = list(range(len(candidates)))
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(indices)

    blocks: list[str] = []
    id_to_orig: list[tuple[str, int]] = []
    for presentation_pos, orig_idx in enumerate(indices):
        cid = _candidate_id(presentation_pos)
        cand = candidates[orig_idx]
        baseline_rank = orig_idx + 1  # 1-indexed original rank
        snippet = _extract_snippet(cand)
        path = _extract_path(cand)
        symbol = _extract_symbol(cand)
        lines = _extract_lines(cand)
        block = f"""{cid}
baseline_rank: {baseline_rank}
path: {path}
symbol: {symbol}
lines: {lines}
snippet:
```
{snippet}
```"""
        blocks.append(block)
        id_to_orig.append((cid, orig_idx))
    return "\n\n".join(blocks), id_to_orig


def _build_user_message(query: str, candidates_block: str) -> str:
    return f"""QUERY:
{query}

{RANKING_RUBRIC}

CANDIDATES:
{candidates_block}

{OUTPUT_SCHEMA_DESC}"""


def _extract_json_block(text: str) -> str | None:
    """Find the first balanced {...} JSON block in `text`, returning the
    block string or None if no balanced block is found.

    Handles chain-of-thought-prefixed responses (e.g. Sonnet emits
    "Looking at the query 'X'... {ranked_ids: [...]}") by scanning for
    the first '{' and tracking brace depth, respecting string literals
    and escape sequences. Returns the substring spanning the matching
    outer braces.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def _validate_response(
    raw_text: str, id_to_orig: list[tuple[str, int]],
) -> tuple[list[int] | None, str | None]:
    """Parse and validate the model response.

    Tolerates chain-of-thought-prefixed responses by extracting the first
    balanced {...} JSON block (Sonnet 4.6 doesn't support assistant-prefill
    via the standard messages API; this is the alternative recovery path
    per the 2026-05-16 Phase C v2 attempt).

    Returns (ordered_original_indices, error_reason). On success,
    error_reason is None and ordered_original_indices is a permutation of
    range(len(id_to_orig)). On failure, ordered_original_indices is None
    and error_reason is one of REASON_PARSE_FAILED / REASON_ID_MISMATCH.
    """
    text = raw_text.strip()
    # Strip code-fence if model wrapped it
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline > 0:
            text = text[first_newline+1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse first (cleanest case)
    obj = None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: extract first balanced {...} block (handles CoT prefix)
        block = _extract_json_block(text)
        if block is None:
            if query_text_logging_enabled():
                LOG.warning(
                    "[LISTWISE] no JSON block found; head=%r",
                    raw_text[:200],
                )
            else:
                LOG.warning(
                    "[LISTWISE] no JSON block found; response text omitted"
                )
            return None, REASON_PARSE_FAILED
        try:
            obj = json.loads(block)
        except (json.JSONDecodeError, ValueError) as e:
            if query_text_logging_enabled():
                LOG.warning(
                    "[LISTWISE] JSON parse failed even after block "
                    "extraction: %s; head=%r",
                    e,
                    raw_text[:200],
                )
            else:
                LOG.warning(
                    "[LISTWISE] JSON parse failed even after block "
                    "extraction: %s; response text omitted",
                    format_query_exception_for_log(e),
                )
            return None, REASON_PARSE_FAILED

    if not isinstance(obj, dict):
        return None, REASON_PARSE_FAILED
    ranked_ids = obj.get("ranked_ids")
    if not isinstance(ranked_ids, list) or not all(isinstance(x, str) for x in ranked_ids):
        return None, REASON_PARSE_FAILED

    expected_ids = {cid for cid, _ in id_to_orig}
    seen: set[str] = set()
    ordered_origs: list[int] = []
    id_map = {cid: orig for cid, orig in id_to_orig}
    for cid in ranked_ids:
        if cid in seen:
            LOG.warning("[LISTWISE] duplicate id %r in response", cid)
            return None, REASON_ID_MISMATCH
        if cid not in id_map:
            LOG.warning("[LISTWISE] unknown id %r in response", cid)
            return None, REASON_ID_MISMATCH
        seen.add(cid)
        ordered_origs.append(id_map[cid])
    missing = expected_ids - seen
    if missing:
        LOG.warning("[LISTWISE] missing ids in response: %s", sorted(missing))
        return None, REASON_ID_MISMATCH

    return ordered_origs, None


def listwise_rerank_with_sonnet(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    *,
    timeout: float | None = None,
    return_metadata: bool = False,
    shuffle_seed: int | None = 0,  # default deterministic shuffle per call
    _client_factory=None,  # test seam: () -> anthropic.Anthropic-like client
):
    """Listwise rerank candidates via one Sonnet 4.6 call. Returns top-k.

    Args:
        query: original search query string
        candidates: list of dicts, each with at least path + snippet keys.
        top_k: number of results to return after reranking
        timeout: per-call budget in seconds (default DEFAULT_TIMEOUT_S=12.0,
            override via SONNET_LISTWISE_TIMEOUT env)
        return_metadata: when True, returns (list, dict) with applied/reason/latency_ms
        shuffle_seed: if not None, shuffle candidate presentation order with this seed.
            Set None to preserve order (useful for debugging).
        _client_factory: test seam — function returning an Anthropic client-like
            object exposing `messages.create(...)`.

    Returns: reranked candidates[:top_k]. On any failure, returns
        candidates[:top_k] in input order.

    Never raises.
    """
    t_start = time.monotonic()

    def _emit(out_list: list[dict], applied: bool, reason: str):
        latency_ms = int((time.monotonic() - t_start) * 1000)
        if return_metadata:
            return out_list, {"applied": applied, "reason": reason, "latency_ms": latency_ms}
        return out_list

    if not candidates:
        return _emit([], False, REASON_EMPTY_INPUT)
    if not env_get("ANTHROPIC_API_KEY") and _client_factory is None:
        return _emit(candidates[:top_k], False, REASON_API_KEY_MISSING)
    if timeout is None:
        try:
            timeout = float(env_get("SONNET_LISTWISE_TIMEOUT", DEFAULT_TIMEOUT_S))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_S

    # Build prompt
    candidates_block, id_to_orig = _build_candidates_block(candidates, shuffle_seed)
    user_msg = _build_user_message(query, candidates_block)

    # Get Anthropic client (test seam or real)
    if _client_factory is not None:
        client = _client_factory()
    else:
        try:
            import anthropic  # type: ignore
        except ImportError:
            return _emit(candidates[:top_k], False, REASON_PACKAGE_NOT_INSTALLED)
        client = anthropic.Anthropic(
            max_retries=int(env_get("ANTHROPIC_MAX_RETRIES", "1")),
            timeout=timeout,
        )

    # Single Sonnet call.
    # NOTE: Anthropic Sonnet 4.6 does NOT support assistant-message prefill
    # ("This model does not support assistant message prefill. The conversation
    # must end with a user message.", 400 error). Tried 2026-05-16 Phase C v2,
    # reverted. Chain-of-thought-before-JSON leak is handled in the validator
    # via brace-balanced extraction instead.
    try:
        resp = client.messages.create(
            model=env_get("ANTHROPIC_MODEL", DEFAULT_MODEL),
            max_tokens=2000,  # output schema is small; ranked_ids + scores
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0,
        )
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg and "limit" in msg:
            LOG.warning(
                "[LISTWISE] rate-limited: %s; using hybrid order",
                format_query_exception_for_log(e),
                exc_info=query_text_logging_enabled(),
            )
            return _emit(candidates[:top_k], False, REASON_RATE_LIMIT)
        if "timeout" in msg or "timed out" in msg:
            LOG.warning(
                "[LISTWISE] timeout: %s; using hybrid order",
                format_query_exception_for_log(e),
                exc_info=query_text_logging_enabled(),
            )
            return _emit(candidates[:top_k], False, REASON_TIMEOUT)
        LOG.warning(
            "[LISTWISE] unexpected error: %s; using hybrid order",
            format_query_exception_for_log(e),
            exc_info=query_text_logging_enabled(),
        )
        return _emit(candidates[:top_k], False, REASON_UNEXPECTED_ERROR)

    # Extract text from response (Anthropic API shape).
    # NOTE: the assistant pre-fill ("{") is NOT echoed in resp.content — Anthropic
    # returns only the continuation. We must prepend "{" before JSON parsing so
    # the validator sees a well-formed object.
    try:
        # resp.content is a list of content blocks; pull text from first text block
        blocks = getattr(resp, "content", None) or []
        text_parts = []
        for block in blocks:
            # block could be a TextBlock with .text, or a dict
            t = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            if t:
                text_parts.append(t)
        raw_text = "".join(text_parts)
    except Exception as e:
        LOG.warning(
            "[LISTWISE] response shape error: %s",
            format_query_exception_for_log(e),
            exc_info=query_text_logging_enabled(),
        )
        return _emit(candidates[:top_k], False, REASON_UNEXPECTED_ERROR)

    if not raw_text:
        return _emit(candidates[:top_k], False, REASON_PARSE_FAILED)

    ordered_origs, err = _validate_response(raw_text, id_to_orig)
    if err is not None or ordered_origs is None:
        return _emit(candidates[:top_k], False, err or REASON_PARSE_FAILED)

    # Reorder candidates per ordered_origs
    reordered = [candidates[i] for i in ordered_origs]
    LOG.info(
        "[LISTWISE_REASON] %s n_candidates=%d top_k=%d latency_ms=%d",
        REASON_OK, len(candidates), top_k,
        int((time.monotonic() - t_start) * 1000),
    )
    return _emit(reordered[:top_k], True, REASON_OK)
