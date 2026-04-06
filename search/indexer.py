"""Vector index management with FAISS and metadata storage."""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sqlitedict import SqliteDict
from embeddings.embedder import EmbeddingResult


class CodeIndexManager:
    """Manages FAISS vector index and metadata storage for code chunks."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.storage_dir / "code.index"
        self.metadata_path = self.storage_dir / "metadata.db" 
        self.chunk_id_path = self.storage_dir / "chunk_ids.pkl"
        self.stats_path = self.storage_dir / "stats.json"
        
        # Initialize components
        self._index = None
        self._metadata_db = None
        self._chunk_ids = []
        self._logger = logging.getLogger(__name__)
        self._on_gpu = False

        # Initialize FTS5
        self._init_fts5()
        
    def _init_fts5(self):
        """Initialize FTS5 full-text search table."""
        import sqlite3
        self._fts_db_path = self.storage_dir / "fts5.db"
        self._fts_conn = sqlite3.connect(
            str(self._fts_db_path),
            check_same_thread=False,
        )
        self._fts_conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id,
                content,
                file_path,
                name,
                tokenize='porter unicode61'
            )
        """)
        self._fts_conn.commit()

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize a natural-language query for FTS5 MATCH syntax.

        Strips FTS5 operators and special chars, quotes each token,
        and joins with OR so any keyword match counts.
        """
        import re
        # Remove characters that are FTS5 operators or cause syntax errors
        cleaned = re.sub(r'[?"*/\\(){}^~:+\-]', ' ', query)
        tokens = [t for t in cleaned.split() if t and len(t) > 1]
        if not tokens:
            return ""
        # Quote each token to prevent column-name interpretation
        return " OR ".join(f'"{t}"' for t in tokens)

    def search_bm25(self, query: str, k: int = 50, name_weight: float = 5.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using BM25 full-text search. Returns (chunk_id, rank, metadata)."""
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            return []

        fts_query = self._sanitize_fts5_query(query)
        if not fts_query:
            return []

        try:
            cursor = self._fts_conn.execute(
                f"SELECT chunk_id, bm25(chunk_fts, 0.0, 1.0, 0.5, {float(name_weight)}) as rank "
                "FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, k),
            )
            results = []
            for chunk_id, rank in cursor.fetchall():
                metadata_entry = self.metadata_db.get(chunk_id)
                if metadata_entry:
                    results.append((chunk_id, float(rank), metadata_entry["metadata"]))
            return results
        except Exception as e:
            self._logger.warning(f"FTS5 search failed for '{fts_query}': {e}")
            return []

    @property
    def index(self):
        """Lazy loading of FAISS index."""
        if self._index is None:
            self._load_index()
        return self._index
    
    @property
    def metadata_db(self):
        """Lazy loading of metadata database."""
        if self._metadata_db is None:
            self._metadata_db = SqliteDict(
                str(self.metadata_path), 
                autocommit=False,
                journal_mode="WAL"
            )
        return self._metadata_db
    
    def _load_index(self):
        """Load existing FAISS index or create new one."""
        self._is_binary = False
        float_store_path = self.storage_dir / "float_store.npy"

        if self.index_path.exists():
            # Detect binary mode: float_store.npy exists alongside the index
            if float_store_path.exists():
                self._logger.info(f"Loading binary index from {self.index_path}")
                self._index = faiss.read_index_binary(str(self.index_path))
                self._float_store = np.load(str(float_store_path))
                self._is_binary = True
            else:
                self._logger.info(f"Loading index from {self.index_path}")
                self._index = faiss.read_index(str(self.index_path))
                if not self._is_binary:
                    self._maybe_move_index_to_gpu()

            # Load chunk IDs
            if self.chunk_id_path.exists():
                with open(self.chunk_id_path, 'rb') as f:
                    self._chunk_ids = pickle.load(f)
        else:
            self._logger.info("Creating new index")
            self._index = None
            self._chunk_ids = []
    
    def create_index(self, embedding_dimension: int, index_type: str = "flat"):
        """Create a new FAISS index.

        Quantization controlled by QUANTIZATION env var:
        - "int8" (default): ScalarQuantizer with QT_8bit_direct — 4x smaller, <0.1% quality loss
        - "float32": IndexFlatIP — original full-precision
        - "binary": IndexBinaryFlat + float store — 32x smaller, needs rescore (opt-in for 100K+ chunks)
        """
        quantization = os.environ.get("QUANTIZATION", "int8").lower()
        self._is_binary = False

        if quantization == "binary":
            self._index = faiss.IndexBinaryFlat(embedding_dimension)
            self._float_store = np.empty((0, embedding_dimension), dtype=np.float32)
            self._is_binary = True
            self._logger.info(f"Created binary index with dimension {embedding_dimension} (32x compression, requires rescore)")
        elif quantization == "int8" and index_type == "flat":
            # QT_8bit (trained) learns the value range from data, then linearly maps to [0,255].
            # QT_8bit_direct was wrong — it interprets float bytes as raw ints, producing
            # all-zero similarities on normalized [-1,1] vectors. (Confirmed 2026-04-05:
            # isolated FAISS test showed QT_8bit_direct returns 0.0 for all queries.)
            self._index = faiss.IndexScalarQuantizer(
                embedding_dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
            )
            self._logger.info(f"Created int8 quantized index with dimension {embedding_dimension} (4x compression, requires training)")
        elif index_type == "flat":
            self._index = faiss.IndexFlatIP(embedding_dimension)
            self._logger.info(f"Created float32 flat index with dimension {embedding_dimension}")
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dimension)
            n_centroids = min(100, max(10, embedding_dimension // 8))
            self._index = faiss.IndexIVFFlat(quantizer, embedding_dimension, n_centroids)
            self._logger.info(f"Created IVF index with dimension {embedding_dimension}")
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        if not self._is_binary:
            self._maybe_move_index_to_gpu()
    
    def add_embeddings(self, embedding_results: List[EmbeddingResult]) -> None:
        """Add embeddings to the index and metadata to the database."""
        if not embedding_results:
            return
        
        # Initialize index if needed
        if self._index is None:
            embedding_dim = embedding_results[0].embedding.shape[0]
            # Always use flat index - IVF breaks reconstruct() needed by get_similar_chunks
            # Flat handles 20K+ vectors fine for our use case
            index_type = "flat"
            self.create_index(embedding_dim, index_type)
        
        # Prepare embeddings and metadata
        embeddings = np.array([result.embedding for result in embedding_results])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Train quantized/IVF index if needed
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            self._logger.info("Training index...")
            self._index.train(embeddings)

        # Add to FAISS index (binary mode packs bits separately)
        if getattr(self, '_is_binary', False):
            # Binary mode: pack sign bits and store float originals for rescoring
            codes = np.packbits((embeddings > 0).astype(np.uint8), axis=1)
            self._index.add(codes)
            self._float_store = np.concatenate([self._float_store, embeddings], axis=0)
        else:
            self._index.add(embeddings)
        start_id = len(self._chunk_ids)
        
        # Store metadata and update chunk IDs
        for i, result in enumerate(embedding_results):
            chunk_id = result.chunk_id
            self._chunk_ids.append(chunk_id)
            
            # Store in metadata database
            self.metadata_db[chunk_id] = {
                'index_id': start_id + i,
                'metadata': result.metadata
            }
        
        self._logger.info(f"Added {len(embedding_results)} embeddings to index")
        
        # Commit metadata in a single transaction for performance
        try:
            self.metadata_db.commit()
        except Exception:
            # If commit is unavailable for some reason, continue without failing
            pass

        # Add to FTS5 index (re-init if connection was lost)
        if not hasattr(self, "_fts_conn") or self._fts_conn is None:
            self._init_fts5()
        for result in embedding_results:
            content = result.metadata.get("full_content", result.metadata.get("content_preview", ""))
            file_path = result.metadata.get("relative_path", result.metadata.get("file_path", ""))
            name = result.metadata.get("name", "") or ""
            self._fts_conn.execute(
                "INSERT INTO chunk_fts (chunk_id, content, file_path, name) VALUES (?, ?, ?, ?)",
                (result.chunk_id, content, file_path, name),
            )
        self._fts_conn.commit()

        # Update statistics
        self._update_stats()

    def _gpu_is_available(self) -> bool:
        """Check if GPU FAISS support is available and GPUs are present."""
        try:
            if not hasattr(faiss, 'StandardGpuResources'):
                return False
            get_num_gpus = getattr(faiss, 'get_num_gpus', None)
            if get_num_gpus is None:
                return False
            return get_num_gpus() > 0
        except Exception:
            return False

    def _maybe_move_index_to_gpu(self) -> None:
        """Move the current index to GPU if supported. No-op if already on GPU or unsupported."""
        if self._index is None or self._on_gpu:
            return
        if not self._gpu_is_available():
            return
        try:
            # Move index to all GPUs for faster add/search
            self._index = faiss.index_cpu_to_all_gpus(self._index)
            self._on_gpu = True
            self._logger.info("FAISS index moved to GPU(s)")
        except Exception as e:
            self._logger.warning(f"Failed to move FAISS index to GPU, continuing on CPU: {e}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar code chunks."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Index manager search called with k={k}, filters={filters}")
        
        # Use property to trigger lazy loading
        index = self.index
        if index is None or index.ntotal == 0:
            logger.warning(f"Index is empty or None. Index: {index}, ntotal: {index.ntotal if index else 'N/A'}")
            return []

        logger.info(f"Index has {index.ntotal} total vectors")

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Binary mode: hamming search → float rescore
        if getattr(self, '_is_binary', False) and hasattr(self, '_float_store'):
            search_k = min(k * 20, index.ntotal)
            q_codes = np.packbits((query_embedding[0] > 0).astype(np.uint8)).reshape(1, -1)
            _distances, bin_indices = index.search(q_codes, search_k)
            # Rescore with float dot product
            candidate_ids = bin_indices[0][bin_indices[0] >= 0]
            if len(candidate_ids) == 0:
                return []
            candidate_vecs = self._float_store[candidate_ids]
            scores = candidate_vecs @ query_embedding[0]
            top_order = np.argsort(-scores)
            indices = np.array([candidate_ids[top_order]])
            similarities = np.array([scores[top_order]])
        else:
            # Standard search (float32 or int8)
            search_k = min(k * 3, index.ntotal)
            similarities, indices = index.search(query_embedding, search_k)
        
        results = []
        for i, (similarity, index_id) in enumerate(zip(similarities[0], indices[0])):
            if index_id == -1:  # No more results
                break
            
            chunk_id = self._chunk_ids[index_id]
            metadata_entry = self.metadata_db.get(chunk_id)
            
            if metadata_entry is None:
                continue
            
            metadata = metadata_entry['metadata']
            
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            results.append((chunk_id, float(similarity), metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if key == 'file_pattern':
                # Pattern matching for file paths
                if not any(pattern in metadata.get('relative_path', '') for pattern in value):
                    return False
            elif key == 'chunk_type':
                # Exact match for chunk type
                if metadata.get('chunk_type') != value:
                    return False
            elif key == 'tags':
                # Tag intersection
                chunk_tags = set(metadata.get('tags', []))
                required_tags = set(value if isinstance(value, list) else [value])
                if not required_tags.intersection(chunk_tags):
                    return False
            elif key == 'folder_structure':
                # Check if any of the required folders are in the path
                chunk_folders = set(metadata.get('folder_structure', []))
                required_folders = set(value if isinstance(value, list) else [value])
                if not required_folders.intersection(chunk_folders):
                    return False
            elif key in metadata:
                # Direct metadata comparison
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata by ID."""
        metadata_entry = self.metadata_db.get(chunk_id)
        return metadata_entry['metadata'] if metadata_entry else None
    
    def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find chunks similar to a given chunk."""
        metadata_entry = self.metadata_db.get(chunk_id)
        if not metadata_entry:
            return []
        
        index_id = metadata_entry['index_id']
        if self._index is None or index_id >= self._index.ntotal:
            return []
        
        # Get the embedding for this chunk
        embedding = self._index.reconstruct(index_id)
        
        # Search for similar chunks (excluding the original)
        results = self.search(embedding, k + 1)
        
        # Filter out the original chunk
        return [(cid, sim, meta) for cid, sim, meta in results if cid != chunk_id][:k]
    
    def remove_file_chunks(self, file_path: str, project_name: Optional[str] = None) -> int:
        """Remove all chunks from a specific file.
        
        Args:
            file_path: Path to the file (relative or absolute)
            project_name: Optional project name filter
            
        Returns:
            Number of chunks removed
        """
        chunks_to_remove = []
        
        # Find chunks to remove
        for chunk_id in self._chunk_ids:
            metadata_entry = self.metadata_db.get(chunk_id)
            if not metadata_entry:
                continue
            
            metadata = metadata_entry['metadata']
            
            # Check if this chunk belongs to the file
            chunk_file = metadata.get('file_path') or metadata.get('relative_path')
            if not chunk_file:
                continue
            
            # Check if paths match (handle both relative and absolute)
            if file_path in chunk_file or chunk_file in file_path:
                # Check project name if provided
                if project_name and metadata.get('project_name') != project_name:
                    continue
                chunks_to_remove.append(chunk_id)
        
        # Remove chunks from metadata
        for chunk_id in chunks_to_remove:
            del self.metadata_db[chunk_id]
        
        # Note: We don't remove from FAISS index directly as it's complex
        # Instead, we'll rebuild the index periodically or on demand
        
        self._logger.info(f"Removed {len(chunks_to_remove)} chunks from {file_path}")
        
        # Commit removals in batch
        try:
            self.metadata_db.commit()
        except Exception:
            pass
        return len(chunks_to_remove)
    
    def save_index(self):
        """Save the FAISS index and chunk IDs to disk."""
        if self._index is not None:
            try:
                if getattr(self, '_is_binary', False):
                    # Binary mode: save binary index + float store
                    faiss.write_index_binary(self._index, str(self.index_path))
                    float_path = self.storage_dir / "float_store.npy"
                    np.save(str(float_path), self._float_store)
                    self._logger.info(f"Saved binary index + float store to {self.storage_dir}")
                else:
                    index_to_write = self._index
                    if self._on_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                        index_to_write = faiss.index_gpu_to_cpu(self._index)
                    faiss.write_index(index_to_write, str(self.index_path))
                    self._logger.info(f"Saved index to {self.index_path}")
            except Exception as e:
                self._logger.warning(f"Failed to save index: {e}")
                if not getattr(self, '_is_binary', False):
                    try:
                        cpu_index = faiss.index_gpu_to_cpu(self._index)
                        faiss.write_index(cpu_index, str(self.index_path))
                    except Exception as e2:
                        self._logger.error(f"Failed to save FAISS index: {e2}")
        
        # Save chunk IDs
        with open(self.chunk_id_path, 'wb') as f:
            pickle.dump(self._chunk_ids, f)
        
        self._update_stats()
    
    def _update_stats(self):
        """Update index statistics."""
        # Detect quantization type for reporting
        if getattr(self, '_is_binary', False):
            quant = "binary"
            idx_dim = self._float_store.shape[1] if hasattr(self, '_float_store') and len(self._float_store) > 0 else 0
        elif self._index and "ScalarQuantizer" in type(self._index).__name__:
            quant = "int8"
            idx_dim = self._index.d if self._index else 0
        else:
            quant = "float32"
            idx_dim = self._index.d if self._index else 0

        stats = {
            'total_chunks': len(self._chunk_ids),
            'index_size': self._index.ntotal if self._index else 0,
            'embedding_dimension': idx_dim,
            'index_type': type(self._index).__name__ if self._index else 'None',
            'quantization': quant,
        }
        
        # Add file and folder statistics
        file_counts = {}
        folder_counts = {}
        chunk_type_counts = {}
        tag_counts = {}
        
        for chunk_id in self._chunk_ids:
            metadata_entry = self.metadata_db.get(chunk_id)
            if not metadata_entry:
                continue
            
            metadata = metadata_entry['metadata']
            
            # Count by file
            file_path = metadata.get('relative_path', 'unknown')
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
            
            # Count by folder
            for folder in metadata.get('folder_structure', []):
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            # Count by chunk type
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            
            # Count by tags
            for tag in metadata.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats.update({
            'files_indexed': len(file_counts),
            'top_folders': dict(sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'chunk_types': chunk_type_counts,
            'top_tags': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        })
        
        # Save stats
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.stats_path.exists():
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'total_chunks': 0,
                'index_size': 0,
                'embedding_dimension': 0,
                'files_indexed': 0
            }
    
    def get_index_size(self) -> int:
        """Get the number of chunks in the index."""
        return len(self._chunk_ids)
    
    def clear_index(self):
        """Clear the entire index and metadata."""
        # Close database connection
        if self._metadata_db is not None:
            self._metadata_db.close()
            self._metadata_db = None

        # Close and remove FTS5 database
        if hasattr(self, "_fts_conn") and self._fts_conn is not None:
            self._fts_conn.close()
            self._fts_conn = None
        fts_path = self.storage_dir / "fts5.db"
        if fts_path.exists():
            fts_path.unlink()

        # Remove files
        for file_path in [self.index_path, self.metadata_path, self.chunk_id_path, self.stats_path]:
            if file_path.exists():
                file_path.unlink()

        # Reset in-memory state
        self._index = None
        self._chunk_ids = []

        # Re-initialize FTS5 for new inserts
        self._init_fts5()

        self._logger.info("Index cleared")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._metadata_db is not None:
            self._metadata_db.close()
        if hasattr(self, "_fts_conn") and self._fts_conn is not None:
            self._fts_conn.close()
