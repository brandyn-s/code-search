"""OpenAI embedding model implementation."""

import os
import logging
from typing import List, Dict, Any
import numpy as np
import httpx

from embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

# Known dimensions for OpenAI models
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "voyage-code-3": 1024,
    "voyage-3-large": 1024,
    "voyage-3-lite": 512,
}


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI API embedding model."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 0,
        batch_delay: float = 0.0,
        **kwargs,
    ):
        # Skip device resolution - not needed for API model
        self._device = "api"
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Set it as an environment variable "
                "or pass api_key to the constructor."
            )
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=300.0)
        self._dimension = MODEL_DIMENSIONS.get(model_name, 1536)

        # Voyage has stricter rate limits (6M TPM) - use smaller batches with delay
        is_voyage = model_name.startswith("voyage-")
        self._batch_size = batch_size or (16 if is_voyage else 256)
        self._batch_delay = batch_delay or (1.0 if is_voyage else 0.0)

        logger.info(
            f"OpenAI embedder initialized: model={model_name}, dim={self._dimension}, "
            f"batch_size={self._batch_size}, batch_delay={self._batch_delay}s"
        )

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts via embeddings API with retry on server/rate-limit errors."""
        import time

        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            # Inter-batch delay to stay within TPM rate limits
            if i > 0 and self._batch_delay > 0:
                time.sleep(self._batch_delay)

            for attempt in range(4):
                try:
                    response = self._client.post(
                        f"{self._base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"input": batch, "model": self._model_name},
                    )
                    response.raise_for_status()
                    data = response.json()
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", 0)
                    if attempt < 3 and status in (429, 500, 502, 503, 529):
                        # 429: back off longer to let TPM window reset
                        wait = (15 * (attempt + 1)) if status == 429 else 2**attempt
                        logger.warning(
                            f"Embedding batch {i}-{i + len(batch)} error {status}, "
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
            "provider": "openai",
            "device": "api",
            "status": "loaded",
        }

    def cleanup(self):
        if hasattr(self, "_client"):
            self._client.close()
