"""Voyage AI contextualized chunk embedder using /v1/contextualizedembeddings."""

import os
import logging
import time
from typing import List, Dict, Any
import numpy as np
import httpx

from embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


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

        # Process in batches of documents (not chunks) to respect rate limits
        batch_size = 4  # 4 documents per request
        for i in range(0, len(grouped_texts), batch_size):
            batch = grouped_texts[i : i + batch_size]

            if i > 0 and self._batch_delay > 0:
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

                    # Flatten: each item in data has "embeddings" (list of chunk embeddings)
                    for doc_result in data["data"]:
                        all_embeddings.extend(doc_result["embeddings"])
                    break
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", 0)
                    if attempt < 3 and status in (429, 500, 502, 503, 529):
                        wait = (15 * (attempt + 1)) if status == 429 else 2**attempt
                        logger.warning(
                            f"Context embed batch {i} error {status}, "
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
