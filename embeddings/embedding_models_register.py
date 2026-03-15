"""Embedding models registry. Lazy imports to avoid loading torch at module load time."""

# Only GemmaEmbeddingModel is registered here for legacy conftest patching.
# Actual provider selection happens in embedder.py via conditional imports.
AVAILIABLE_MODELS = {}


def _load_models():
    """Load model classes on demand."""
    if not AVAILIABLE_MODELS:
        from embeddings.gemma import GemmaEmbeddingModel
        AVAILIABLE_MODELS["google/embeddinggemma-300m"] = GemmaEmbeddingModel
