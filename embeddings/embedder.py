"""Code embedding wrapper using EmbeddingGemma model."""

import os
import logging
from collections import OrderedDict
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
            # Default: voyage-4-large if Voyage key exists (+0.053 weighted avg MRR
            # over voyage-context-3 across 4 languages, 102 queries, 2026-04-08).
            # Uses standard /embeddings endpoint (not /contextualizedembeddings).
            if os.environ.get("VOYAGE_API_KEY"):
                provider = "voyage"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "local"

        if provider == "openai":
            from embeddings.openai_embedder import OpenAIEmbeddingModel

            model_name = model_name or os.environ.get(
                "EMBEDDING_MODEL", "text-embedding-3-small"
            )
            self._model = OpenAIEmbeddingModel(model_name=model_name)
        elif provider == "voyage":
            from embeddings.openai_embedder import OpenAIEmbeddingModel

            model_name = model_name or os.environ.get(
                "EMBEDDING_MODEL", "voyage-4-large"
            )
            api_key = os.environ.get("VOYAGE_API_KEY", "")
            self._model = OpenAIEmbeddingModel(
                api_key=api_key,
                model_name=model_name,
                base_url="https://api.voyageai.com/v1",
            )
        elif provider == "voyage-context":
            from embeddings.voyage_context_embedder import VoyageContextEmbedder

            model_name = model_name or os.environ.get(
                "EMBEDDING_MODEL", "voyage-context-3"
            )
            api_key = os.environ.get("VOYAGE_API_KEY", "")
            self._model = VoyageContextEmbedder(
                api_key=api_key,
                model_name=model_name,
            )
        elif provider in ("jina", "jina-code"):
            from embeddings.jina_code_embedder import JinaCodeEmbedder

            model_name = model_name or os.environ.get(
                "LOCAL_EMBEDDING_MODEL", "jinaai/jina-code-embeddings-0.5b"
            )
            truncate_dim_str = os.environ.get("JINA_TRUNCATE_DIM", "")
            truncate_dim = int(truncate_dim_str) if truncate_dim_str else None
            self._model = JinaCodeEmbedder(
                model_name=model_name,
                cache_dir=cache_dir,
                device=device,
                truncate_dim=truncate_dim,
            )
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
                f"Unknown EMBEDDING_PROVIDER: {provider}. "
                f"Use 'voyage-context', 'voyage', 'openai', 'jina', 'local', or 'gemma'."
            )

        self._logger.info(f"Embedding provider: {provider}, model: {model_name}")

    @property
    def model(self):
        """Get the underlying embedding model."""
        return self._model.model

    # Sibling context: populated by embed_chunks/embed_chunks_grouped before
    # calling create_embedding_content. Maps relative_path -> list of sibling
    # chunk names in the same file. Approximates Voyage's contextualized
    # embeddings for non-contextual models (Jina, local).
    _sibling_context: Dict[str, list] = {}

    def set_sibling_context(self, chunks: list) -> None:
        """Build sibling name index from a batch of chunks grouped by file."""
        from collections import defaultdict
        groups = defaultdict(list)
        for c in chunks:
            if c.name:
                groups[c.relative_path].append(c.name)
        self._sibling_context = dict(groups)

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

            # Enriched context: add sibling chunk names from the same file.
            # Approximates Voyage's contextualized embeddings for models that
            # embed each chunk independently (Jina, local).
            # Default: "on" for non-contextualized providers, "off" for voyage-context
            # (which already has file-grouped context via the API).
            # Eval result: +9.6% MRR on Nix, closing 40% of the gap to voyage-context-3.
            enriched_default = "off" if os.environ.get("EMBEDDING_PROVIDER", "") == "voyage-context" else "on"
            if os.environ.get("ENRICHED_CONTEXT", enriched_default) == "on":
                siblings = self._sibling_context.get(chunk.relative_path, [])
                # Exclude self, limit to 15 siblings to stay within token budget
                other_names = [n for n in siblings if n != chunk.name][:15]
                if other_names:
                    content_parts.append(f"# File also contains: {', '.join(other_names)}")

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

        # Build sibling context for enriched headers
        self.set_sibling_context(chunks)

        self._logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_contents = [self.create_embedding_content(chunk) for chunk in batch]

            # Generate embeddings for batch
            encode_kwargs = {
                "prompt_name": "Retrieval-document",
                "show_progress_bar": False,
            }
            # Voyage input_type optimization: "document" for indexing
            if os.environ.get("VOYAGE_INPUT_TYPE", "off") == "on":
                encode_kwargs["input_type"] = "document"
            batch_embeddings = self._model.encode(
                batch_contents,
                **encode_kwargs,
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

    def embed_chunks_grouped(
        self, chunks: List[CodeChunk], batch_size: int = 32
    ) -> List[EmbeddingResult]:
        """Generate embeddings with chunks grouped by source file.

        Uses encode_grouped() if the model supports it (voyage-context-3),
        otherwise falls back to flat embed_chunks().
        """
        if not hasattr(self._model, "encode_grouped"):
            return self.embed_chunks(chunks, batch_size)

        from collections import defaultdict

        # Group chunks by source file
        file_groups: Dict[str, List[CodeChunk]] = defaultdict(list)
        for chunk in chunks:
            file_groups[chunk.relative_path].append(chunk)

        self._logger.info(
            f"Grouped {len(chunks)} chunks into {len(file_groups)} files for contextualized embedding"
        )

        results = []
        file_items = list(file_groups.items())

        for batch_start in range(0, len(file_items), batch_size):
            batch_files = file_items[batch_start : batch_start + batch_size]

            # Build grouped texts and track chunk ordering
            grouped_texts = []
            batch_chunks = []
            for _file_path, file_chunks in batch_files:
                texts = [self.create_embedding_content(c) for c in file_chunks]
                grouped_texts.append(texts)
                batch_chunks.extend(file_chunks)

            # Get grouped embeddings (flattened)
            batch_embeddings = self._model.encode_grouped(
                grouped_texts, input_type="document"
            )

            # Create results
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
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

            if batch_start + batch_size < len(file_items):
                self._logger.info(
                    f"Processed {batch_start + batch_size}/{len(file_items)} files"
                )

        self._logger.info("Grouped embedding generation completed")
        return results

    # LRU query embedding cache (in-memory) + optional SQLite persistent cache.
    # In-memory cache: fast, dies with process. SQLite: survives restarts,
    # eliminates cold-start latency for Jina on CPU (~5s per query → instant).
    _query_cache: OrderedDict = OrderedDict()
    _QUERY_CACHE_MAX: int = 256
    _disk_cache_db = None

    def _get_disk_cache(self):
        """Lazy-init SQLite persistent query cache."""
        if self._disk_cache_db is not None:
            return self._disk_cache_db
        try:
            import sqlite3
            cache_path = get_storage_dir() / "query_cache.db"
            db = sqlite3.connect(str(cache_path), check_same_thread=False)
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS query_embeddings (
                    query_key TEXT PRIMARY KEY,
                    embedding BLOB,
                    provider TEXT,
                    dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db.commit()
            self._disk_cache_db = db
            return db
        except Exception as e:
            self._logger.debug(f"Disk cache unavailable: {e}")
            return None

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query. Cached via LRU + SQLite.

        Cache layers:
        1. In-memory LRU (instant, per-session)
        2. SQLite on disk (survives restarts, eliminates Jina cold-start)
        3. Model encode (slowest, only on cache miss)

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        cache_key = query.strip().lower()

        # Layer 1: in-memory LRU
        if cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key]

        # Layer 2: SQLite persistent cache
        provider = os.environ.get("EMBEDDING_PROVIDER", "")
        db = self._get_disk_cache()
        if db is not None:
            try:
                row = db.execute(
                    "SELECT embedding, dim FROM query_embeddings WHERE query_key = ? AND provider = ?",
                    (cache_key, provider),
                ).fetchone()
                if row:
                    embedding = np.frombuffer(row[0], dtype=np.float32).copy()
                    if embedding.shape[0] == row[1]:
                        self._query_cache[cache_key] = embedding
                        return embedding
            except Exception:
                pass

        # Layer 3: model encode
        encode_kwargs = {
            "prompt_name": "InstructionRetrieval",
            "show_progress_bar": False,
        }
        # Voyage input_type optimization: "query" for search
        if os.environ.get("VOYAGE_INPUT_TYPE", "off") == "on":
            encode_kwargs["input_type"] = "query"
        embedding = self._model.encode(
            [query], **encode_kwargs
        )[0]

        # Store in both caches
        self._query_cache[cache_key] = embedding
        if len(self._query_cache) > self._QUERY_CACHE_MAX:
            self._query_cache.popitem(last=False)

        if db is not None:
            try:
                db.execute(
                    "INSERT OR REPLACE INTO query_embeddings (query_key, embedding, provider, dim) VALUES (?, ?, ?, ?)",
                    (cache_key, embedding.tobytes(), provider, embedding.shape[0]),
                )
                db.commit()
            except Exception:
                pass

        return embedding

    def clear_query_cache(self):
        """Clear both in-memory and persistent query caches (call after reindex)."""
        self._query_cache.clear()
        db = self._get_disk_cache()
        if db is not None:
            try:
                db.execute("DELETE FROM query_embeddings")
                db.commit()
            except Exception:
                pass

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
