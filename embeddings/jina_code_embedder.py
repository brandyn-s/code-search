"""Jina Code Embeddings model for local code retrieval.

Uses jinaai/jina-code-embeddings (0.5b or 1.5b) via sentence-transformers.
These models achieve near-parity with voyage-code-3 while running fully local
— no API calls, no data leaves the machine.

Design decisions:
- Last-token pooling (not mean pooling) — the models are decoder-only (Qwen2.5-Coder)
  and were trained with last-token pooling. sentence-transformers handles this automatically
  when loading the model config from HuggingFace.
- Left padding required — decoder models need left-aligned padding for correct last-token
  extraction. Set via tokenizer_kwargs.
- Task-specific prompt prefixes — the models use nl2code_query/nl2code_document prompts
  for code retrieval. We map the generic prompt names used by CodeEmbedder to these.
- BFloat16 on CUDA, float32 on CPU — bfloat16 halves memory but not all CPUs support it.
- Matryoshka dimensions supported — embeddings can be truncated to smaller dims without
  recomputing (0.5b: 64-896, 1.5b: 128-1536).

Benchmarks (MTEB Code, 25 benchmarks):
  jina-code-embeddings-0.5b: 78.41% (494M params, 896-dim)
  jina-code-embeddings-1.5b: 79.04% (1.54B params, 1536-dim)
  voyage-code-3:             79.23% (proprietary, 1024-dim)
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import cached_property
import numpy as np

from embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

# Default model — 0.5b is the best balance of quality vs resource usage.
# 1.5b is higher quality but needs ~3GB RAM and is slower on CPU.
DEFAULT_MODEL = "jinaai/jina-code-embeddings-0.5b"

# Map CodeEmbedder's generic prompt names to Jina code-specific prompts.
# CodeEmbedder passes "Retrieval-document" for indexing and "InstructionRetrieval"
# for queries. Jina code models use nl2code_query/nl2code_document for the primary
# natural-language-to-code retrieval task.
_PROMPT_MAP = {
    "Retrieval-document": "nl2code_document",
    "InstructionRetrieval": "nl2code_query",
    "document": "nl2code_document",
    "query": "nl2code_query",
}


class JinaCodeEmbedder(EmbeddingModel):
    """Jina Code Embeddings — local code retrieval model.

    Runs entirely on-device via sentence-transformers. No API calls.
    """

    def __init__(
        self,
        model_name: str = "",
        cache_dir: Optional[str] = None,
        device: str = "auto",
        truncate_dim: Optional[int] = None,
    ):
        """Initialize JinaCodeEmbedder.

        Args:
            model_name: HuggingFace model name. Defaults to jina-code-embeddings-0.5b.
            cache_dir: Directory to cache downloaded model weights.
            device: Device for inference ("auto", "cuda", "cpu", "mps").
            truncate_dim: Truncate embeddings to this dimension (Matryoshka).
                0.5b supports: 64, 128, 256, 512, 896.
                1.5b supports: 128, 256, 512, 1024, 1536.
                None uses the model's native dimension.
        """
        super().__init__(device=device)
        self.model_name = model_name or DEFAULT_MODEL
        self.cache_dir = cache_dir
        self._truncate_dim = truncate_dim

    @cached_property
    def model(self):
        """Load the SentenceTransformer model with Jina-specific config."""
        import torch
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading Jina code embedder: {self.model_name} on {self._device}")

        model_kwargs = {}
        # bfloat16 halves memory and speeds up inference on GPU.
        # CPU: some support bfloat16 (AVX-512) but float32 is safer default.
        if self._device in ("cuda", "mps"):
            model_kwargs["torch_dtype"] = torch.bfloat16

        st_kwargs = {
            "model_kwargs": model_kwargs,
            "tokenizer_kwargs": {"padding_side": "left"},
            "device": self._device,
        }
        if self.cache_dir:
            st_kwargs["cache_folder"] = self.cache_dir
        if self._truncate_dim:
            st_kwargs["truncate_dim"] = self._truncate_dim

        # Enable offline mode if model is already cached
        if self.cache_dir and self._is_model_cached():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            logger.info("Model cache detected — loading offline")

        model = SentenceTransformer(self.model_name, **st_kwargs)

        # CRITICAL: The model defaults to max_seq_length=32768. The decoder pads
        # every input to this length, making CPU inference ~5x slower than needed.
        # Our chunks are capped at 6000 chars (~1500 tokens) by create_embedding_content.
        # 512 tokens covers 95%+ of chunks; longer ones get truncated with minimal
        # quality loss (Matryoshka training makes the model robust to truncation).
        model.max_seq_length = 512

        logger.info(
            f"Jina code embedder loaded: dim={model.get_sentence_embedding_dimension()}, "
            f"max_seq={model.max_seq_length}, device={model.device}"
        )
        return model

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings.

        Maps generic prompt names to Jina code-specific task prompts.
        """
        prompt_name = kwargs.get("prompt_name")
        if prompt_name is not None:
            mapped = _PROMPT_MAP.get(prompt_name)
            if mapped:
                model_prompts = getattr(self.model, "prompts", {})
                if mapped in model_prompts:
                    kwargs["prompt_name"] = mapped
                else:
                    # Model doesn't have this prompt — drop it
                    kwargs.pop("prompt_name", None)
            else:
                # Unknown prompt name — check if model has it directly
                model_prompts = getattr(self.model, "prompts", {})
                if prompt_name not in model_prompts:
                    kwargs.pop("prompt_name", None)

        return self.model.encode(texts, **kwargs)

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension (respects Matryoshka truncation)."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "model_name": self.model_name,
            "provider": "jina-code",
            "device": self._device,
            "status": "loaded" if "model" in self.__dict__ else "not_loaded",
        }
        if "model" in self.__dict__:
            info["embedding_dimension"] = self.get_embedding_dimension()
            info["max_seq_length"] = getattr(self.model, "max_seq_length", "unknown")
        if self._truncate_dim:
            info["truncate_dim"] = self._truncate_dim
        return info

    def cleanup(self):
        """Clean up model resources."""
        if "model" not in self.__dict__:
            return
        try:
            import torch

            model = self.model
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del model
            logger.info("Jina code embedder cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def _is_model_cached(self) -> bool:
        """Check if model weights are already downloaded."""
        if not self.cache_dir:
            return False
        try:
            model_key = self.model_name.split("/")[-1].lower()
            cache_root = Path(self.cache_dir)
            if not cache_root.exists():
                return False
            for d in cache_root.rglob("config.json"):
                if model_key in str(d.parent).lower():
                    return True
        except Exception:
            pass
        return False
