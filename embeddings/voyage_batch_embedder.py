"""Voyage AI Batch API embedder for cost-efficient full reindexing.

Uses the async Batch API (/v1/batches) which is 33% cheaper than real-time
and handles retries automatically. 12-hour completion window.

Only used for full reindex (1000+ chunks). Incremental updates use real-time API.
Opt-in via VOYAGE_BATCH_API=on env var.
"""

import json
import logging
import os
import tempfile
import time
from typing import List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

_BASE = "https://api.voyageai.com/v1"


class VoyageBatchEmbedder:
    """Embeds texts via Voyage Batch API for full reindexing."""

    def __init__(self, api_key: str = "", model: str = "voyage-code-3"):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        self.model = model
        self.client = httpx.Client(timeout=120.0)
        self._headers = {"Authorization": f"Bearer {self.api_key}"}
        self._json_headers = {**self._headers, "Content-Type": "application/json"}

    def embed_all(
        self,
        texts: List[str],
        input_type: str = "document",
        poll_interval: int = 10,
        timeout_minutes: int = 30,
    ) -> Optional[np.ndarray]:
        """Embed all texts via batch API. Returns (N, dim) float32 array or None on failure.

        Args:
            texts: All texts to embed.
            input_type: "document" for indexing, "query" for search.
            poll_interval: Seconds between status polls.
            timeout_minutes: Max wait time before giving up.

        Returns:
            numpy array of shape (len(texts), embedding_dim) or None.
        """
        if not self.api_key:
            logger.warning("VOYAGE_API_KEY not set, cannot use batch API")
            return None

        # Step 1: Create JSONL input file
        # Batch API: each line is {"custom_id": "...", "body": {"input": [...]}}
        # Max 1000 inputs per line, max 100K lines per batch
        jsonl_path = self._create_jsonl(texts)
        if not jsonl_path:
            return None

        try:
            # Step 2: Upload file
            file_id = self._upload_file(jsonl_path)
            if not file_id:
                return None

            # Step 3: Create batch job
            batch_id = self._create_batch(file_id, input_type)
            if not batch_id:
                return None

            # Step 4: Poll for completion
            output_file_id = self._poll_batch(batch_id, poll_interval, timeout_minutes)
            if not output_file_id:
                return None

            # Step 5: Download and parse results
            return self._download_results(output_file_id, len(texts))

        finally:
            try:
                os.unlink(jsonl_path)
            except OSError:
                pass

    def _create_jsonl(self, texts: List[str]) -> Optional[str]:
        """Create JSONL input file. Batches texts into groups of 128 per request."""
        try:
            batch_size = 128
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as f:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    line = json.dumps({
                        "custom_id": f"batch_{i}",
                        "body": {"input": batch},
                    })
                    f.write(line + "\n")
                return f.name
        except Exception as e:
            logger.error(f"Failed to create JSONL: {e}")
            return None

    def _upload_file(self, path: str) -> Optional[str]:
        """Upload JSONL to Voyage Files API."""
        try:
            with open(path, "rb") as f:
                resp = self.client.post(
                    f"{_BASE}/files",
                    headers=self._headers,
                    files={"file": ("batch_embed.jsonl", f, "application/jsonl")},
                    data={"purpose": "batch"},
                )
            if resp.status_code != 200:
                logger.error(f"File upload failed: {resp.status_code} {resp.text[:200]}")
                return None
            file_id = resp.json()["id"]
            logger.info(f"Uploaded batch file: {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"File upload error: {e}")
            return None

    def _create_batch(self, file_id: str, input_type: str) -> Optional[str]:
        """Create a batch embedding job."""
        try:
            resp = self.client.post(
                f"{_BASE}/batches",
                headers=self._json_headers,
                json={
                    "endpoint": "/v1/embeddings",
                    "completion_window": "12h",
                    "request_params": {
                        "model": self.model,
                        "input_type": input_type,
                    },
                    "input_file_id": file_id,
                },
            )
            if resp.status_code != 200:
                logger.error(f"Batch creation failed: {resp.status_code} {resp.text[:200]}")
                return None
            batch_id = resp.json()["id"]
            logger.info(f"Batch job created: {batch_id}")
            return batch_id
        except Exception as e:
            logger.error(f"Batch creation error: {e}")
            return None

    def _poll_batch(self, batch_id: str, interval: int, timeout_min: int) -> Optional[str]:
        """Poll batch job until completion. Returns output_file_id or None."""
        deadline = time.time() + timeout_min * 60
        while time.time() < deadline:
            try:
                resp = self.client.get(f"{_BASE}/batches/{batch_id}", headers=self._headers)
                info = resp.json()
                status = info.get("status", "unknown")
                counts = info.get("request_counts", {})

                if status == "completed":
                    logger.info(
                        f"Batch completed: {counts.get('completed', 0)}/{counts.get('total', 0)} requests"
                    )
                    return info.get("output_file_id")
                elif status == "failed":
                    logger.error(f"Batch failed: {info.get('errors', 'unknown')}")
                    return None
                # Still processing
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                time.sleep(interval)

        logger.error(f"Batch timed out after {timeout_min} minutes")
        return None

    def _download_results(self, file_id: str, expected_count: int) -> Optional[np.ndarray]:
        """Download batch results and reconstruct embedding array."""
        try:
            resp = self.client.get(f"{_BASE}/files/{file_id}/content", headers=self._headers)
            if resp.status_code != 200:
                logger.error(f"Result download failed: {resp.status_code}")
                return None

            # Parse JSONL results — each line has embeddings for one batch
            # Results may be out of order, keyed by custom_id
            all_embeddings = {}
            for line in resp.text.strip().split("\n"):
                result = json.loads(line)
                custom_id = result.get("custom_id", "")
                response_body = result.get("response", {}).get("body", {})
                embeddings = [item["embedding"] for item in response_body.get("data", [])]
                # Extract batch index from custom_id (e.g., "batch_128" → 128)
                try:
                    batch_start = int(custom_id.split("_")[1])
                except (IndexError, ValueError):
                    batch_start = 0
                for j, emb in enumerate(embeddings):
                    all_embeddings[batch_start + j] = emb

            # Reconstruct ordered array
            if len(all_embeddings) != expected_count:
                logger.warning(
                    f"Expected {expected_count} embeddings, got {len(all_embeddings)}"
                )

            ordered = [all_embeddings[i] for i in range(len(all_embeddings))]
            return np.array(ordered, dtype=np.float32)

        except Exception as e:
            logger.error(f"Result parsing error: {e}")
            return None

    def close(self):
        self.client.close()
