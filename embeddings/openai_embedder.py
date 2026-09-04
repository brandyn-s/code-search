"""OpenAI embedding model implementation."""

import logging
from typing import List, Dict, Any
import numpy as np
import httpx

from embeddings.embedding_model import EmbeddingModel
from search.env import env_get

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
KEYED_HOSTS = ("api.openai.com", "api.voyageai.com")
AUTH_HEADER_STYLES = ("bearer", "api-key")


def _host_of(url: str) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").lower()


def resolve_openai_base_url(explicit: str = "") -> str:
    """Resolve the embeddings endpoint root.

    ``OPENAI_BASE_URL`` (the standard OpenAI SDK name) points the ``openai``
    provider at any OpenAI-compatible server: Ollama, vLLM, LM Studio, Azure
    OpenAI, OpenRouter, or a gateway in front of Gemini or Bedrock. The value
    must include the API version path (for example ``http://localhost:11434/v1``);
    trailing slashes are removed. ``explicit`` wins over the environment.
    """
    raw = explicit or env_get("OPENAI_BASE_URL", "") or DEFAULT_OPENAI_BASE_URL
    return raw.strip().rstrip("/")


def requires_api_key(base_url: str) -> bool:
    """OpenAI and Voyage always need a key; self-hosted servers often do not."""
    return _host_of(base_url) in KEYED_HOSTS


def resolve_auth_header_style() -> str:
    """``OPENAI_AUTH_HEADER``: ``bearer`` (default) or ``api-key`` (Azure OpenAI keys)."""
    style = (env_get("OPENAI_AUTH_HEADER", "bearer") or "bearer").strip().lower()
    if style not in AUTH_HEADER_STYLES:
        raise ValueError(
            f"OPENAI_AUTH_HEADER must be one of {AUTH_HEADER_STYLES}, got {style!r}"
        )
    return style


def build_auth_headers(api_key: str, style: str = "bearer") -> Dict[str, str]:
    """Authorization headers for one request; empty when there is no key."""
    if not api_key:
        return {}
    if style == "api-key":
        return {"api-key": api_key}
    return {"Authorization": f"Bearer {api_key}"}


# Known dimensions for OpenAI models
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "voyage-code-3": 1024,
    "voyage-3-large": 1024,
    "voyage-3-lite": 512,
    "voyage-4-large": 1024,
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
}


# Voyage API token limits (leave headroom below the 120K hard limit)
_VOYAGE_MAX_TOKENS_PER_REQUEST = 100_000


def _env_dimension() -> int:
    """``EMBEDDING_DIMENSION`` as an int, or 0 when unset/invalid (validated elsewhere)."""
    raw = (env_get("EMBEDDING_DIMENSION", "") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for code/English text."""
    return len(text) // 4


def _split_batch_by_tokens(texts: list, max_tokens: int = _VOYAGE_MAX_TOKENS_PER_REQUEST) -> list:
    """Split a batch into sub-batches that fit within token limits."""
    sub_batches = []
    current = []
    current_tokens = 0
    for text in texts:
        t = _estimate_tokens(text)
        if current and current_tokens + t > max_tokens:
            sub_batches.append(current)
            current = [text]
            current_tokens = t
        else:
            current.append(text)
            current_tokens += t
    if current:
        sub_batches.append(current)
    return sub_batches


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI API embedding model."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "text-embedding-3-small",
        base_url: str = "",
        batch_size: int = 0,
        batch_delay: float = 0.0,
        auth_header: str = "",
        **kwargs,
    ):
        # Skip device resolution - not needed for API model
        self._device = "api"
        self._base_url = resolve_openai_base_url(base_url)
        self._api_key = api_key or env_get("OPENAI_API_KEY", "")
        if not self._api_key and requires_api_key(self._base_url):
            key_name = "VOYAGE_API_KEY" if "voyageai" in _host_of(self._base_url) else "OPENAI_API_KEY"
            raise ValueError(
                f"{key_name} is required for {_host_of(self._base_url)}. Set it as an "
                "environment variable, pass api_key to the constructor, or point "
                "OPENAI_BASE_URL at a self-hosted OpenAI-compatible server "
                "(Ollama, vLLM, LM Studio) that needs no key."
            )
        self._auth_header = auth_header or resolve_auth_header_style()
        self._model_name = model_name
        self._client = httpx.Client(timeout=300.0)
        self._dimension = MODEL_DIMENSIONS.get(model_name) or _env_dimension() or 1536

        # Voyage has stricter rate limits (6M TPM) - use smaller batches with delay
        is_voyage = model_name.startswith("voyage-")
        self._batch_size = batch_size or (16 if is_voyage else 256)
        self._batch_delay = batch_delay or (1.0 if is_voyage else 0.0)

        logger.info(
            f"OpenAI embedder initialized: model={model_name}, dim={self._dimension}, "
            f"base_url={self._base_url}, auth={'none' if not self._api_key else self._auth_header}, "
            f"batch_size={self._batch_size}, batch_delay={self._batch_delay}s"
        )

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts via embeddings API with retry on server/rate-limit errors.

        Kwargs:
            input_type: Optional "query" or "document" for Voyage models.
                        Adds retrieval-optimized prompts to the embeddings.
        """
        import time

        input_type = kwargs.get("input_type")
        all_embeddings = []

        is_voyage = self._model_name.startswith("voyage-")

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            # Token-aware sub-batching: split oversized batches to avoid 400 errors
            sub_batches = _split_batch_by_tokens(batch) if is_voyage else [batch]

            for sub_batch in sub_batches:
                # Inter-batch delay to stay within TPM rate limits
                if all_embeddings and self._batch_delay > 0:
                    time.sleep(self._batch_delay)

                for attempt in range(4):
                    try:
                        payload = {"input": sub_batch, "model": self._model_name}
                        # Voyage models support input_type for retrieval optimization
                        if input_type and is_voyage:
                            payload["input_type"] = input_type
                        headers = {"Content-Type": "application/json"}
                        headers.update(build_auth_headers(self._api_key, self._auth_header))
                        response = self._client.post(
                            f"{self._base_url}/embeddings",
                            headers=headers,
                            json=payload,
                        )
                        response.raise_for_status()
                        data = response.json()
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        all_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        status = getattr(getattr(e, "response", None), "status_code", 0)
                        if attempt < 3 and status in (429, 500, 502, 503, 529):
                            wait = (15 * (attempt + 1)) if status == 429 else 2**attempt
                            logger.warning(
                                f"Embedding sub-batch error {status}, "
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
            "base_url": self._base_url,
            "status": "loaded",
        }

    def cleanup(self):
        if hasattr(self, "_client"):
            self._client.close()
