"""Voyage AI contextualized chunk embedder using /v1/contextualizedembeddings."""

import os
import logging
import time
from typing import List, Dict, Any
import numpy as np
import httpx

from embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


# --- Voyage batch token management ---
# API limits (verified from docs.voyageai.com/reference/contextualized-embeddings-api):
#   - 1,000 inputs per request
#   - 120K tokens per request
#   - 16K chunks across inputs
# Headroom: tokens at 100K, inputs at 500 (half of 1000). The previous cap of
# 4 was unjustified and throttled throughput by ~100x on repos with many small
# files (2026-04-17 incident: 10,993 chunks took 3+ hours at the 4-group cap
# while GHES with larger files completed in 15 min because groups hit the
# token cap before the group-count cap).
_VOYAGE_MAX_TOKENS_PER_DOC = 30_000   # API limit 32K, leave headroom
_VOYAGE_MAX_TOKENS_PER_BATCH = 100_000  # API limit 120K, leave headroom
_VOYAGE_MAX_INPUTS_PER_BATCH = 500  # API limit 1000, leave headroom


def _estimate_tokens(texts: list[str]) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return sum(len(t) for t in texts) // 4


def _split_oversized_group(group: list[str], max_tokens: int = _VOYAGE_MAX_TOKENS_PER_DOC) -> list[list[str]]:
    """Split a document group that exceeds per-document token limit."""
    sub_groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in group:
        text_tokens = len(text) // 4
        if current and current_tokens + text_tokens > max_tokens:
            sub_groups.append(current)
            current = [text]
            current_tokens = text_tokens
        else:
            current.append(text)
            current_tokens += text_tokens
    if current:
        sub_groups.append(current)
    return sub_groups if sub_groups else [group]


def _prepare_voyage_batches(grouped_texts: list[list[str]]) -> list[list[list[str]]]:
    """Pre-split oversized groups and build token-aware batches.

    Returns a list of batches, where each batch is a list of groups
    that fit within Voyage API limits.
    """
    # Step 1: Split oversized groups
    split_groups: list[list[str]] = []
    for group in grouped_texts:
        if _estimate_tokens(group) > _VOYAGE_MAX_TOKENS_PER_DOC:
            split_groups.extend(_split_oversized_group(group))
        else:
            split_groups.append(group)

    # Step 2: Build batches respecting total token limit
    batches: list[list[list[str]]] = []
    current_batch: list[list[str]] = []
    current_tokens = 0
    for group in split_groups:
        group_tokens = _estimate_tokens(group)
        if current_batch and (current_tokens + group_tokens > _VOYAGE_MAX_TOKENS_PER_BATCH
                              or len(current_batch) >= _VOYAGE_MAX_INPUTS_PER_BATCH):
            batches.append(current_batch)
            current_batch = [group]
            current_tokens = group_tokens
        else:
            current_batch.append(group)
            current_tokens += group_tokens
    if current_batch:
        batches.append(current_batch)
    return batches


class VoyageContextEmbedder(EmbeddingModel):
    """Voyage AI contextualized chunk embedding model.

    Uses /v1/contextualizedembeddings which produces per-chunk embeddings
    that capture the full document context. Chunks from the same file are
    grouped together so the model can see cross-section relationships.
    """

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "voyage-context-3",
        batch_delay: float = 1.0,
        **kwargs,
    ):
        self._device = "api"
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "VOYAGE_API_KEY is required. Set it as an environment variable "
                "or pass api_key to the constructor."
            )
        self._model_name = model_name
        self._base_url = "https://api.voyageai.com/v1"
        self._client = httpx.Client(timeout=300.0)
        self._dimension = 1024
        self._batch_delay = batch_delay

        logger.info(
            f"Voyage context embedder initialized: model={model_name}, dim={self._dimension}"
        )

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts as individual items (for queries).

        Wraps each text in its own document group, since queries
        don't have multi-chunk document context.
        """
        input_type = kwargs.get("input_type", "query")
        grouped = [[text] for text in texts]
        return self._call_api(grouped, input_type)

    def encode_grouped(
        self,
        grouped_texts: List[List[str]],
        input_type: str = "document",
    ) -> np.ndarray:
        """Encode chunks grouped by source document.

        Args:
            grouped_texts: List of documents, each a list of chunk texts.
                Example: [["chunk1_file1", "chunk2_file1"], ["chunk1_file2"]]
            input_type: "document" for indexing, "query" for searching.

        Returns:
            Flat array of embeddings, one per chunk across all documents.
        """
        return self._call_api(grouped_texts, input_type)

    def _call_api(
        self,
        grouped_texts: List[List[str]],
        input_type: str,
    ) -> np.ndarray:
        """Call the contextualized embeddings API with retry."""
        all_embeddings = []
        batches = _prepare_voyage_batches(grouped_texts)

        for batch_idx, batch in enumerate(batches):
            if batch_idx > 0 and self._batch_delay > 0:
                time.sleep(self._batch_delay)

            for attempt in range(4):
                try:
                    response = self._client.post(
                        f"{self._base_url}/contextualizedembeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "inputs": batch,
                            "model": self._model_name,
                            "input_type": input_type,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    for doc_result in data["data"]:
                        for chunk_emb in doc_result["data"]:
                            all_embeddings.append(chunk_emb["embedding"])
                    break
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", 0)
                    if attempt < 3 and status in (429, 500, 502, 503, 529):
                        wait = (15 * (attempt + 1)) if status == 429 else 2**attempt
                        logger.warning(
                            f"Context embed batch {batch_idx} error {status}, "
                            f"retrying in {wait}s (attempt {attempt + 1}/3)..."
                        )
                        time.sleep(wait)
                        continue
                    raise

        return np.array(all_embeddings, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "embedding_dimension": self._dimension,
            "provider": "voyage-context",
            "device": "api",
            "status": "loaded",
        }

    def cleanup(self):
        if hasattr(self, "_client"):
            self._client.close()
