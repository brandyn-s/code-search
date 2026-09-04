"""Voyage AI reranker using /v1/rerank endpoint."""

import logging
import time
from typing import List, Tuple

import httpx
from search.env import env_get

logger = logging.getLogger(__name__)


def voyage_rerank(
    query: str,
    documents: List[str],
    model: str = "rerank-2.5",
    top_k: int = 5,
    api_key: str = "",
) -> List[Tuple[int, float]]:
    """Rerank documents using Voyage AI cross-encoder reranker.

    Returns list of (original_index, relevance_score) sorted by descending relevance.
    """
    key = api_key or env_get("VOYAGE_API_KEY", "")
    if not key:
        raise ValueError("VOYAGE_API_KEY required for Voyage reranker")
    if not documents:
        return []

    client = httpx.Client(timeout=60.0)
    try:
        for attempt in range(3):
            try:
                response = client.post(
                    "https://api.voyageai.com/v1/rerank",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "documents": documents,
                        "model": model,
                        "top_k": min(top_k, len(documents)),
                    },
                )
                response.raise_for_status()
                data = response.json()
                results = [(item["index"], item["relevance_score"]) for item in data.get("data", [])]
                return sorted(results, key=lambda x: x[1], reverse=True)
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", 0)
                if attempt < 2 and status in (429, 500, 502, 503):
                    wait = (15 * (attempt + 1)) if status == 429 else 2 ** attempt
                    logger.warning(f"Voyage rerank error {status}, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise
    finally:
        client.close()
    return []
