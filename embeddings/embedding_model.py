"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, device: str):
        """Initialize with device resolution."""
        self._device = self._resolve_device(device)

    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            **kwargs: Additional model-specific arguments

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up model resources."""
        pass

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass

    def _resolve_device(self, requested: Optional[str]) -> str:
        """Resolve device string.

        torch is imported lazily: API-only providers (OpenAI, Voyage) set
        their device directly and never call this, so a deployment that
        only uses API embeddings must not require torch to be installed.
        Before this, the module-level `import torch` made every embedder —
        including the pure-httpx ones — transitively depend on a ~2GB
        package they never use.
        """
        req = (requested or "auto").lower()
        try:
            import torch
        except ImportError:
            return "cpu"
        if req in ("auto", "none", ""):
            if torch.cuda.is_available():
                return "cuda"
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        if req.startswith("cuda"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "mps":
            try:
                return "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
            except Exception:
                return "cpu"
        return "cpu"
