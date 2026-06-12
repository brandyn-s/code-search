#!/usr/bin/env python3
"""Command-line tool for indexing a Python codebase."""

import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import CodeEmbedder
from search.indexer import CodeIndexManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Index a Python codebase for semantic search"
    )
    parser.add_argument(
        "directory",
        help="Directory containing Python files to index"
    )
    parser.add_argument(
        "--storage-dir",
        default=str(Path.home() / ".claude_code_search"),
        help="Directory to store index and embeddings (default: ~/.claude_code_search)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding generation (default: 8)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before indexing"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate directory
    directory_path = Path(args.directory).resolve()
    if not directory_path.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        sys.exit(1)
    
    if not directory_path.is_dir():
        logger.error(f"Path is not a directory: {directory_path}")
        sys.exit(1)
    
    # Setup storage
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Indexing directory: {directory_path}")
        logger.info(f"Storage directory: {storage_dir}")
        
        # Initialize components
        logger.info("Initializing components...")
        chunker = MultiLanguageChunker(str(directory_path))
        
        # Initialize embedder with cache in storage directory
        models_dir = storage_dir / "models"
        models_dir.mkdir(exist_ok=True)
        embedder = CodeEmbedder(cache_dir=str(models_dir))
        
        # Initialize index manager
        index_dir = storage_dir / "index"
        index_manager = CodeIndexManager(str(index_dir))
        
        # Clear existing index if requested
        if args.clear:
            logger.info("Clearing existing index...")
            index_manager.clear_index()
        
        # Chunk the codebase. chunk_directory requires the directory path —
        # the chunker constructor binds the project root for support checks,
        # but the walk target is an explicit argument (API change this CLI
        # missed; caught 2026-06-11 by the SweRank pilot's Arm C smoke).
        logger.info("Parsing and chunking Python files...")
        chunks = chunker.chunk_directory(str(directory_path))
        
        if not chunks:
            logger.error("No Python files found or no chunks extracted")
            sys.exit(1)
        
        logger.info(f"Generated {len(chunks)} chunks from Python files")
        
        # Display some statistics
        chunk_types = {}
        file_count = {}
        
        for chunk in chunks:
            # Count chunk types
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            
            # Count files
            file_count[chunk.relative_path] = file_count.get(chunk.relative_path, 0) + 1
        
        logger.info(f"Chunk types: {dict(chunk_types)}")
        logger.info(f"Files processed: {len(file_count)}")
        
        # Generate embeddings
        logger.info("Generating embeddings (this may take a while)...")
        embedding_results = embedder.embed_chunks(chunks, batch_size=args.batch_size)
        
        logger.info(f"Generated {len(embedding_results)} embeddings")
        
        # Add to index
        logger.info("Building search index...")
        index_manager.add_embeddings(embedding_results)
        
        # Save index
        logger.info("Saving index to disk...")
        index_manager.save_index()
        
        # Display final statistics
        stats = index_manager.get_stats()
        model_info = embedder.get_model_info()
        
        logger.info("=" * 50)
        logger.info("INDEXING COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Total chunks indexed: {stats['total_chunks']}")
        logger.info(f"Files processed: {stats['files_indexed']}")
        logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"Index type: {stats['index_type']}")
        logger.info(f"Model: {model_info['model_name']}")
        
        if stats.get('chunk_types'):
            logger.info("\nChunk type distribution:")
            for chunk_type, count in stats['chunk_types'].items():
                logger.info(f"  {chunk_type}: {count}")
        
        if stats.get('top_tags'):
            logger.info("\nTop semantic tags:")
            for tag, count in list(stats['top_tags'].items())[:10]:
                logger.info(f"  {tag}: {count}")
        
        logger.info(f"\nStorage location: {storage_dir}")
        logger.info("\nYou can now use the MCP server for Claude Code integration:")
        logger.info(f"  python {Path(__file__).parent.parent / 'mcp_server' / 'server.py'}")
        
    except KeyboardInterrupt:
        logger.info("\nIndexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
