"""Embedding models registry."""
from embeddings.gemma import GemmaEmbeddingModel
from embeddings.sentence_transformer import SentenceTransformerModel
from embeddings.openai_embedder import OpenAIEmbeddingModel

AVAILIABLE_MODELS = {
    "google/embeddinggemma-300m": GemmaEmbeddingModel,
}

# Provider -> (model_class, default_model_name, requires_api_key)
PROVIDERS = {
    "openai": (OpenAIEmbeddingModel, "text-embedding-3-small", True),
    "local": (SentenceTransformerModel, "sentence-transformers/all-MiniLM-L6-v2", False),
    "gemma": (GemmaEmbeddingModel, "google/embeddinggemma-300m", False),
}
