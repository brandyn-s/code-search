#!/usr/bin/env python3
"""Standalone model download script that doesn't depend on our modules."""

import os
import sys
import logging
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Install with: pip install 'code-search-mcp[local]'")
    sys.exit(1)


def download_model(model_name: str = "google/embeddinggemma-300m", storage_dir: str = None):
    """Download the embedding model."""
    if storage_dir is None:
        storage_dir = os.path.expanduser("~/.claude_code_search")
    
    # Create storage directory
    storage_path = Path(storage_dir)
    models_dir = storage_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model: {model_name}")
    print(f"Storage directory: {models_dir}")
    
    try:
        # Download and cache the model
        model = SentenceTransformer(
            model_name,
            cache_folder=str(models_dir),
            device="cpu"  # Use CPU to avoid GPU issues
        )
        
        print("Testing model...")
        test_text = "def hello_world():\n    return 'Hello, World!'"
        embedding = model.encode([test_text])
        
        print(f"Model downloaded successfully!")
        print(f"Embedding dimension: {embedding.shape[1]}")
        print(f"Model cached in: {models_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download embedding model for testing")
    parser.add_argument(
        "--model", 
        default="google/embeddinggemma-300m",
        help="Model name to download"
    )
    parser.add_argument(
        "--storage-dir",
        help="Storage directory (default: ~/.claude_code_search)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    success = download_model(args.model, args.storage_dir)
    sys.exit(0 if success else 1)