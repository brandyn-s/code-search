"""Shared pointwise relevance judge used by every LLM reranker.

Both the Anthropic engine (``search.sonnet_reranker``) and the
OpenAI-compatible engine (``search.openai_reranker``) render the same prompt
and parse the same ``{"score": <0-10>}`` reply, so their scores are directly
comparable and a model swap changes only the transport.

The prompt text is byte-identical to the historical ``sonnet_reranker``
template; ``sonnet_reranker`` re-exports the names below for callers that
imported them from there.
"""

from __future__ import annotations

import json
from typing import Optional

MAX_CONTENT_CHARS = 4000

# The {extra_clauses} slot is filled per-candidate at scoring time when
# SONNET_RERANKER_PROMPT_CLAUSE_OVERRIDES injects path-prefix-matched clauses.
# When the env var is unset (default), {extra_clauses} resolves to "" and the
# prompt is byte-identical to the baseline rubric.
JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a code chunk is relevant to a developer search query.

Query: {query}

Code chunk (file: {file_path}):
```
{content}
```

Rate the relevance on a scale of 0-10:
- 10 = This chunk IS exactly what the user is searching for
- 7-9 = Highly relevant; clearly matches the user's intent
- 4-6 = Partially relevant; related but not the primary target
- 1-3 = Tangentially related
- 0 = Not relevant at all

Domain notes:
- For Nix/NixOS/configuration queries, treat `option` and `binding` chunks
  as primary definitions. If the query asks about `mkOption`,
  `services.<name>`, enable / configuration / module setup, systemd
  service configuration, hardware configuration, or NixOS modules, prefer
  the Nix module or option declaration over daemon implementation code,
  tests, call sites, or docs. For service setup queries,
  `nix/modules/<service>.nix` is usually the implementation of the
  user-visible configuration surface.{extra_clauses}

Respond with ONLY valid JSON:
{{"score": <int 0-10>, "reasoning": "<one sentence>"}}"""

# Backward-compat alias: JUDGE_PROMPT == JUDGE_PROMPT_TEMPLATE with extra_clauses="".
JUDGE_PROMPT = JUDGE_PROMPT_TEMPLATE.replace("{extra_clauses}", "")


def build_judge_prompt(
    query: str,
    file_path: str,
    content: str,
    extra_clauses: str = "",
) -> str:
    """Render the judge prompt for one (query, chunk) pair."""
    truncated = content[:MAX_CONTENT_CHARS] if len(content) > MAX_CONTENT_CHARS else content
    return JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        file_path=file_path or "(unknown)",
        content=truncated or "(empty content)",
        extra_clauses=extra_clauses,
    )


def strip_code_fences(text: str) -> str:
    """Drop ``` fence lines that some models wrap JSON in."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```"))
    return text


def parse_score(text: str) -> Optional[int]:
    """Parse ``{"score": n}`` from a model reply; ``None`` when unparseable.

    The score is clamped to [0, 10]. Non-JSON text, missing keys, and
    non-integer scores all return ``None`` so callers can count the failure.
    """
    try:
        obj = json.loads(strip_code_fences(text))
        score = int(obj.get("score", 0))
    except (ValueError, TypeError, AttributeError):
        return None
    return max(0, min(10, score))
