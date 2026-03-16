"""Code embedding wrapper using EmbeddingGemma model."""

import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from chunking.code_chunk import CodeChunk
from common_utils import get_storage_dir


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embedding: np.ndarray
    chunk_id: str
    metadata: Dict[str, Any]


class CodeEmbedder:
    """Wrapper for embedding code chunks."""

    def __init__(
        self,
        model_name: str = "",
        cache_dir: Optional[str] = None,
        device: str = "auto",
    ):
        if not cache_dir:
            cache_dir = str(get_storage_dir() / "models")
        self.device = device
        self._logger = logging.getLogger(__name__)

        # Determine provider from env
        provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
        if not provider:
            # Default: openai if key exists, else local
            provider = "openai" if os.environ.get("OPENAI_API_KEY") else "local"

        if provider == "openai":
            from embeddings.openai_embedder import OpenAIEmbeddingModel

            model_name = model_name or os.environ.get(
                "EMBEDDING_MODEL", "text-embedding-3-small"
            )
            self._model = OpenAIEmbeddingModel(model_name=model_name)
        elif provider == "local":
            from embeddings.sentence_transformer import SentenceTransformerModel

            model_name = model_name or os.environ.get(
                "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self._model = SentenceTransformerModel(
                model_name=model_name, cache_dir=cache_dir, device=device
            )
        elif provider == "gemma":
            from embeddings.gemma import GemmaEmbeddingModel

            model_name = model_name or "google/embeddinggemma-300m"
            self._model = GemmaEmbeddingModel(cache_dir=cache_dir, device=device)
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER: {provider}. Use 'openai', 'local', or 'gemma'."
            )

        self._logger.info(f"Embedding provider: {provider}, model: {model_name}")

    @property
    def model(self):
        """Get the underlying embedding model."""
        return self._model.model

    def create_embedding_content(self, chunk: CodeChunk, max_chars: int = 6000) -> str:
        """Create clean content for embedding generation.

        Args:
            chunk: Code chunk to create content for
            max_chars: Maximum characters to include

        Returns:
            Content string for embedding
        """
        import os

        content_parts = []

        # Contextual header: prepend file path + type + name for better embeddings
        # (Anthropic research: +20-49% retrieval improvement)
        if os.environ.get("CONTEXTUAL_HEADERS", "on") == "on":
            header_parts = [f"# From {chunk.relative_path}"]
            if chunk.parent_name and chunk.name:
                header_parts.append(
                    f"- {chunk.chunk_type} {chunk.parent_name}.{chunk.name}"
                )
            elif chunk.name:
                header_parts.append(f"- {chunk.chunk_type} {chunk.name}")
            else:
                header_parts.append(f"- {chunk.chunk_type}")
            content_parts.append(" ".join(header_parts))

        # Add docstring if available
        docstring_budget = 300
        if chunk.docstring:
            docstring = (
                chunk.docstring[:docstring_budget] + "..."
                if len(chunk.docstring) > docstring_budget
                else chunk.docstring
            )
            content_parts.append(f'"""{docstring}"""')

        # Calculate remaining budget for code content
        docstring_len = len(content_parts[0]) if content_parts else 0
        remaining_budget = max_chars - docstring_len - 10

        # Add code content with smart truncation
        if len(chunk.content) <= remaining_budget:
            content_parts.append(chunk.content)
        else:
            lines = chunk.content.split("\n")
            if len(lines) > 3:
                head_lines = []
                tail_lines = []
                current_length = docstring_len

                # Add head lines
                for line in lines[: min(len(lines) // 2, 20)]:
                    if current_length + len(line) + 1 > remaining_budget * 0.7:
                        break
                    head_lines.append(line)
                    current_length += len(line) + 1

                # Add tail lines
                remaining_space = remaining_budget - current_length - 20
                for line in reversed(lines[-min(len(lines) // 3, 10) :]):
                    if len("\n".join(tail_lines)) + len(line) + 1 > remaining_space:
                        break
                    tail_lines.insert(0, line)

                if tail_lines:
                    truncated_content = (
                        "\n".join(head_lines)
                        + "\n    # ... (truncated) ...\n"
                        + "\n".join(tail_lines)
                    )
                else:
                    truncated_content = (
                        "\n".join(head_lines) + "\n    # ... (truncated) ..."
                    )
                content_parts.append(truncated_content)
            else:
                content_parts.append(
                    chunk.content[:remaining_budget] + "..."
                    if len(chunk.content) > remaining_budget
                    else chunk.content
                )

        return "\n".join(content_parts)

    def embed_chunk(self, chunk: CodeChunk) -> EmbeddingResult:
        """Generate embedding for a single code chunk.

        Args:
            chunk: Code chunk to embed

        Returns:
            EmbeddingResult with embedding and metadata
        """
        content = self.create_embedding_content(chunk)

        # Encode using model with proper prompt
        embedding = self._model.encode(
            [content], prompt_name="Retrieval-document", show_progress_bar=False
        )[0]

        # Create chunk ID
        chunk_id = f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}:{chunk.chunk_type}"
        if chunk.name:
            chunk_id += f":{chunk.name}"

        # Prepare metadata
        metadata = {
            "file_path": chunk.file_path,
            "relative_path": chunk.relative_path,
            "folder_structure": chunk.folder_structure,
            "chunk_type": chunk.chunk_type,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "name": chunk.name,
            "parent_name": chunk.parent_name,
            "docstring": chunk.docstring,
            "decorators": chunk.decorators,
            "imports": chunk.imports,
            "complexity_score": chunk.complexity_score,
            "tags": chunk.tags,
            "content_preview": chunk.content[:200] + "..."
            if len(chunk.content) > 200
            else chunk.content,
            "full_content": chunk.content,
        }

        return EmbeddingResult(
            embedding=embedding, chunk_id=chunk_id, metadata=metadata
        )

    def embed_chunks(
        self, chunks: List[CodeChunk], batch_size: int = 32
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple chunks with batching.

        Args:
            chunks: List of code chunks to embed
            batch_size: Batch size for processing

        Returns:
            List of EmbeddingResults
        """
        results = []

        self._logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_contents = [self.create_embedding_content(chunk) for chunk in batch]

            # Generate embeddings for batch
            batch_embeddings = self._model.encode(
                batch_contents,
                prompt_name="Retrieval-document",
                show_progress_bar=False,
            )

            # Create results
            for chunk, embedding in zip(batch, batch_embeddings):
                chunk_id = f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}:{chunk.chunk_type}"
                if chunk.name:
                    chunk_id += f":{chunk.name}"

                metadata = {
                    "file_path": chunk.file_path,
                    "relative_path": chunk.relative_path,
                    "folder_structure": chunk.folder_structure,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "name": chunk.name,
                    "parent_name": chunk.parent_name,
                    "docstring": chunk.docstring,
                    "decorators": chunk.decorators,
                    "imports": chunk.imports,
                    "complexity_score": chunk.complexity_score,
                    "tags": chunk.tags,
                    "content_preview": chunk.content[:200] + "..."
                    if len(chunk.content) > 200
                    else chunk.content,
                    "full_content": chunk.content,
                }

                results.append(
                    EmbeddingResult(
                        embedding=embedding, chunk_id=chunk_id, metadata=metadata
                    )
                )

            if i + batch_size < len(chunks):
                self._logger.info(f"Processed {i + batch_size}/{len(chunks)} chunks")

        self._logger.info("Embedding generation completed")
        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        embedding = self._model.encode(
            [query], prompt_name="InstructionRetrieval", show_progress_bar=False
        )[0]
        return embedding

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model.

        Returns:
            Dictionary with model information
        """
        return self._model.get_model_info()

    def cleanup(self):
        """Clean up model resources."""
        self._model.cleanup()

    def __del__(self):
        """Ensure cleanup on object destruction."""
        try:
            self.cleanup()
        except Exception:
            pass
