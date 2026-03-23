"""Code Search Server - manages code search state and business logic."""

import os
import sys
import json
import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from functools import lru_cache

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import get_storage_dir
from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import CodeEmbedder
from search.indexer import CodeIndexManager
from search.searcher import IntelligentSearcher

# Configure logging
logger = logging.getLogger(__name__)


class CodeSearchServer:
    """Server that manages code search state and implements business logic."""

    def __init__(self):
        """Initialize the code search server."""
        # State management
        self._index_manager: Optional[CodeIndexManager] = None
        self._searcher: Optional[IntelligentSearcher] = None
        self._current_project: Optional[str] = None
        self._indexing_job = (
            None  # {job_id, status, phase, current, total, errors, result}
        )
        self._indexing_thread = None

        # Query logging (ported from memory-search)
        self._query_log_db = self._init_query_log()

    def _init_query_log(self) -> Optional[sqlite3.Connection]:
        """Initialize query log database."""
        try:
            log_path = get_storage_dir() / "query_log.db"
            db = sqlite3.connect(str(log_path), check_same_thread=False)
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    project TEXT DEFAULT '',
                    search_mode TEXT DEFAULT 'auto',
                    result_count INTEGER DEFAULT 0,
                    top_score REAL DEFAULT 0.0,
                    latency_ms REAL DEFAULT 0.0,
                    cache_hit INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL
                )
            """)
            db.commit()
            return db
        except Exception:
            return None

    def _log_query(self, query: str, project: str, mode: str,
                   result_count: int, top_score: float, latency_ms: float,
                   cache_hit: bool):
        """Log a search query for offline analysis."""
        if self._query_log_db is None:
            return
        try:
            self._query_log_db.execute(
                "INSERT INTO query_log (query, project, search_mode, result_count, "
                "top_score, latency_ms, cache_hit, timestamp) VALUES (?,?,?,?,?,?,?,?)",
                (query, project or "", mode, result_count, top_score,
                 latency_ms, 1 if cache_hit else 0, time.time()),
            )
            self._query_log_db.commit()
        except Exception:
            pass

    def get_project_storage_dir(self, project_path: str) -> Path:
        """Get or create project-specific storage directory."""
        base_dir = get_storage_dir()
        project_path_obj = Path(project_path).resolve()
        project_name = project_path_obj.name
        import hashlib

        project_hash = hashlib.md5(str(project_path_obj).encode()).hexdigest()[:8]

        # Use common utils for base directory
        project_dir = base_dir / "projects" / f"{project_name}_{project_hash}"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Store project info
        project_info_file = project_dir / "project_info.json"
        if not project_info_file.exists():
            # Auto-select embedding provider from CONTENT_MODE if not explicitly set
            content_mode = os.environ.get("CONTENT_MODE", "code").lower()
            default_provider = "voyage-context" if content_mode == "docs" else "voyage"
            project_info = {
                "project_name": project_name,
                "project_path": str(project_path_obj),
                "project_hash": project_hash,
                "created_at": datetime.now().isoformat(),
                "embedding_provider": os.environ.get(
                    "EMBEDDING_PROVIDER", default_provider
                ),
                "embedding_model": os.environ.get("EMBEDDING_MODEL", ""),
                "content_mode": content_mode,
            }
            with open(project_info_file, "w") as f:
                json.dump(project_info, f, indent=2)

        return project_dir

    def ensure_project_indexed(self, project_path: str) -> bool:
        """Check if project is indexed, auto-index if needed."""
        try:
            project_dir = self.get_project_storage_dir(project_path)
            index_dir = project_dir / "index"

            if index_dir.exists() and (index_dir / "code.index").exists():
                return True

            project_path_obj = Path(project_path)
            if project_path_obj == Path.cwd() and list(
                project_path_obj.glob("**/*.py")
            ):
                logger.info(f"Auto-indexing current directory: {project_path}")
                result = self.index_directory(project_path)
                result_data = json.loads(result)
                return "error" not in result_data

            return False
        except Exception as e:
            logger.warning(f"Failed to check/auto-index project {project_path}: {e}")
            return False

    def embedder(self, project_path: str = None) -> CodeEmbedder:
        """Get embedder for a project, using its stored model config if available."""
        cache_dir = get_storage_dir() / "models"
        cache_dir.mkdir(exist_ok=True)

        # Read project's stored embedding config if project_path given
        provider = ""
        model_name = ""
        if project_path:
            project_dir = self.get_project_storage_dir(project_path)
            info_file = project_dir / "project_info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r") as f:
                        info = json.load(f)
                    provider = info.get("embedding_provider", "")
                    model_name = info.get("embedding_model", "")
                except Exception:
                    pass

        # Override env vars temporarily if project has stored config
        env_overrides = {}
        if provider:
            env_overrides["EMBEDDING_PROVIDER"] = provider
        if model_name:
            env_overrides["EMBEDDING_MODEL"] = model_name

        old_env = {}
        for k, v in env_overrides.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            embedder = CodeEmbedder(cache_dir=str(cache_dir))
            logger.info(
                f"Embedder initialized: provider={provider or 'default'}, model={model_name or 'default'}"
            )
            return embedder
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @lru_cache(maxsize=1)
    def _maybe_start_model_preload(self) -> None:
        """Preload the embedding model in the background (local models only)."""
        # OpenAI provider uses httpx - no heavy model to preload
        if os.environ.get("EMBEDDING_PROVIDER", "openai") == "openai":
            return None

        async def _preload():
            try:
                logger.info("Starting background model preload")
                _ = self.embedder().model
                logger.info("Background model preload completed")
            except Exception as e:
                logger.warning(f"Background model preload failed: {e}")

        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_preload())
            except RuntimeError:
                asyncio.run(_preload())
        except Exception as e:
            logger.debug(f"Model preload scheduling skipped: {e}")
        return None

    def get_index_manager(self, project_path: str = None) -> CodeIndexManager:
        """Get index manager for specific or current project."""
        if project_path is None:
            if self._current_project is None:
                project_path = os.getcwd()
                logger.info(f"No active project. Using cwd: {project_path}")
                # Skip auto-indexing - let the user explicitly index via index_directory
            else:
                project_path = self._current_project

        if self._current_project != project_path:
            self._index_manager = None
            self._current_project = project_path

        if self._index_manager is None:
            project_dir = self.get_project_storage_dir(project_path)
            index_dir = project_dir / "index"
            index_dir.mkdir(exist_ok=True)
            self._index_manager = CodeIndexManager(str(index_dir))
            logger.info(f"Index manager initialized for: {Path(project_path).name}")

        return self._index_manager

    def get_searcher(self, project_path: str = None) -> IntelligentSearcher:
        """Get searcher for specific or current project."""
        if project_path is None and self._current_project is None:
            project_path = os.getcwd()
            logger.info(f"No active project. Using cwd: {project_path}")
            self.ensure_project_indexed(project_path)

        if self._current_project != project_path or self._searcher is None:
            self._searcher = IntelligentSearcher(
                self.get_index_manager(project_path),
                self.embedder(self._current_project),
            )
            logger.info(
                f"Searcher initialized for: {Path(self._current_project).name if self._current_project else 'unknown'}"
            )

        return self._searcher

    def search_code(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "auto",
        file_pattern: str = None,
        chunk_type: str = None,
        include_context: bool = True,
        auto_reindex: bool = True,
        max_age_minutes: float = 5,
    ) -> str:
        """Implementation of search_code tool."""
        t_start = time.time()
        try:
            logger.info(
                f"🔍 MCP REQUEST: search_code(query='{query}', k={k}, mode='{search_mode}', file_pattern={file_pattern}, chunk_type={chunk_type})"
            )

            # If indexing is in progress, report that instead of returning empty
            if self._indexing_job and self._indexing_job["status"] == "indexing":
                job = self._indexing_job
                pct = (
                    round(100 * job["current"] / job["total"], 1)
                    if job["total"] > 0
                    else 0
                )
                return json.dumps(
                    {
                        "query": query,
                        "results": [],
                        "indexing_in_progress": True,
                        "message": f"Indexing in progress ({job['phase']}: {pct}% - {job['current']}/{job['total']} chunks). Results will be available when complete.",
                    }
                )

            if auto_reindex and self._current_project:
                from search.incremental_indexer import IncrementalIndexer

                logger.info(
                    f"Checking if index needs refresh (max age: {max_age_minutes} minutes)"
                )

                index_manager = self.get_index_manager(self._current_project)
                embedder = self.embedder(self._current_project)
                chunker = MultiLanguageChunker(self._current_project)

                incremental_indexer = IncrementalIndexer(
                    indexer=index_manager, embedder=embedder, chunker=chunker
                )

                reindex_result = incremental_indexer.auto_reindex_if_needed(
                    self._current_project, max_age_minutes=max_age_minutes
                )

                if reindex_result.files_modified > 0 or reindex_result.files_added > 0:
                    logger.info(
                        f"Auto-reindexed: {reindex_result.files_added} added, {reindex_result.files_modified} modified, took {reindex_result.time_taken:.2f}s"
                    )
                    # Clear query embedding cache before resetting searcher
                    if self._searcher:
                        self._searcher.clear_cache()
                    self._searcher = None  # Reset to force reload

            searcher = self.get_searcher()
            logger.info(f"Current project: {self._current_project}")

            index_stats = searcher.index_manager.get_stats()
            logger.info(f"Index contains {index_stats.get('total_chunks', 0)} chunks")

            filters = {}
            if file_pattern:
                filters["file_pattern"] = [file_pattern]
            if chunk_type:
                filters["chunk_type"] = chunk_type

            logger.info(f"Search filters: {filters}")

            context_depth = 1 if include_context else 0
            logger.info(
                f"Calling searcher.search with query='{query}', k={k}, mode={search_mode}"
            )
            results = searcher.search(
                query=query,
                k=k,
                search_mode=search_mode,
                context_depth=context_depth,
                filters=filters if filters else None,
            )
            logger.info(f"Search returned {len(results)} results")

            def make_snippet(preview: Optional[str]) -> str:
                if not preview:
                    return ""
                for line in preview.split("\n"):
                    s = line.strip()
                    if s:
                        snippet = " ".join(s.split())
                        return (
                            (snippet[:157] + "...") if len(snippet) > 160 else snippet
                        )
                return ""

            formatted_results = []
            for result in results:
                item = {
                    "file": result.relative_path,
                    "lines": f"{result.start_line}-{result.end_line}",
                    "kind": result.chunk_type,
                    "score": round(result.similarity_score, 2),
                    "chunk_id": result.chunk_id,
                }
                if result.name:
                    item["name"] = result.name
                snippet = make_snippet(result.content_preview)
                if snippet:
                    item["snippet"] = snippet
                formatted_results.append(item)

            response = {"query": query, "results": formatted_results}

            # Log query for offline analysis
            latency_ms = (time.time() - t_start) * 1000
            top_score = formatted_results[0]["score"] if formatted_results else 0.0
            self._log_query(
                query=query,
                project=self._current_project or "",
                mode=search_mode,
                result_count=len(formatted_results),
                top_score=top_score,
                latency_ms=latency_ms,
                cache_hit=query.strip().lower() in searcher._query_embedding_cache if hasattr(searcher, '_query_embedding_cache') else False,
            )

            return json.dumps(response, separators=(",", ":"))
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})

    def index_directory(
        self,
        directory_path: str,
        project_name: str = None,
        file_patterns: List[str] = None,
        incremental: bool = True,
    ) -> str:
        """Start indexing a directory. Returns immediately with job status."""
        import threading
        import uuid

        # If already indexing, return current status
        if self._indexing_job and self._indexing_job["status"] == "indexing":
            return json.dumps(
                {
                    "status": "indexing",
                    "message": "Indexing already in progress",
                    "job_id": self._indexing_job["job_id"],
                    "phase": self._indexing_job.get("phase", "unknown"),
                    "chunks_done": self._indexing_job.get("current", 0),
                    "chunks_total": self._indexing_job.get("total", 0),
                }
            )

        directory_path_obj = Path(directory_path).resolve()
        if not directory_path_obj.exists():
            return json.dumps(
                {"error": f"Directory does not exist: {directory_path_obj}"}
            )
        if not directory_path_obj.is_dir():
            return json.dumps(
                {"error": f"Path is not a directory: {directory_path_obj}"}
            )

        project_name = project_name or directory_path_obj.name
        job_id = uuid.uuid4().hex[:8]

        self._indexing_job = {
            "job_id": job_id,
            "status": "indexing",
            "phase": "starting",
            "current": 0,
            "total": 0,
            "errors": [],
            "directory": str(directory_path_obj),
            "project_name": project_name,
            "result": None,
        }

        def _progress_callback(phase, current, total):
            if self._indexing_job and self._indexing_job["job_id"] == job_id:
                self._indexing_job["phase"] = phase
                self._indexing_job["current"] = current
                self._indexing_job["total"] = total

        def _run_indexing():
            try:
                from search.incremental_indexer import IncrementalIndexer

                self._maybe_start_model_preload()

                index_manager = self.get_index_manager(str(directory_path_obj))
                embedder = self.embedder(str(directory_path_obj))
                chunker = MultiLanguageChunker(str(directory_path_obj))

                incremental_indexer = IncrementalIndexer(
                    indexer=index_manager,
                    embedder=embedder,
                    chunker=chunker,
                    progress_fn=_progress_callback,
                )

                result = incremental_indexer.incremental_index(
                    str(directory_path_obj), project_name, force_full=not incremental
                )

                stats = incremental_indexer.get_indexing_stats(str(directory_path_obj))

                self._indexing_job["status"] = "completed"
                self._indexing_job["phase"] = "done"
                self._indexing_job["result"] = {
                    "success": result.success,
                    "directory": str(directory_path_obj),
                    "project_name": project_name,
                    "files_added": result.files_added,
                    "files_removed": result.files_removed,
                    "files_modified": result.files_modified,
                    "chunks_added": result.chunks_added,
                    "chunks_removed": result.chunks_removed,
                    "time_taken": round(result.time_taken, 2),
                    "index_stats": stats,
                    "error": result.error,
                }
                logger.info(
                    f"Indexing completed. Added: {result.files_added}, Modified: {result.files_modified}, Time: {result.time_taken:.2f}s"
                )
                # Clear query embedding cache after reindex
                if self._searcher:
                    self._searcher.clear_cache()
            except Exception as e:
                logger.error(f"Background indexing failed: {e}", exc_info=True)
                self._indexing_job["status"] = "failed"
                self._indexing_job["phase"] = "error"
                self._indexing_job["result"] = {"error": str(e)}

        logger.info(
            f"Starting background indexing: {directory_path_obj} (incremental={incremental})"
        )
        self._indexing_thread = threading.Thread(target=_run_indexing, daemon=True)
        self._indexing_thread.start()

        return json.dumps(
            {
                "status": "indexing",
                "job_id": job_id,
                "directory": str(directory_path_obj),
                "project_name": project_name,
                "message": "Indexing started in background. Use get_indexing_progress to check status.",
            }
        )

    def get_indexing_progress(self) -> str:
        """Get current indexing job progress."""
        if not self._indexing_job:
            return json.dumps({"status": "idle", "message": "No indexing job running"})

        job = self._indexing_job
        response = {
            "job_id": job["job_id"],
            "status": job["status"],
            "phase": job["phase"],
            "directory": job.get("directory", ""),
            "project_name": job.get("project_name", ""),
        }

        if job["total"] > 0:
            response["chunks_done"] = job["current"]
            response["chunks_total"] = job["total"]
            response["percent"] = round(100 * job["current"] / job["total"], 1)

        if job["status"] in ("completed", "failed") and job.get("result"):
            response["result"] = job["result"]

        return json.dumps(response)

    def find_similar_code(self, chunk_id: str, k: int = 5) -> str:
        """Implementation of find_similar_code tool."""
        try:
            searcher = self.get_searcher()
            results = searcher.find_similar_to_chunk(chunk_id, k=k)

            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "file_path": result.relative_path,
                        "lines": f"{result.start_line}-{result.end_line}",
                        "chunk_type": result.chunk_type,
                        "name": result.name,
                        "similarity_score": round(result.similarity_score, 3),
                        "content_preview": result.content_preview,
                        "tags": result.tags,
                    }
                )

            response = {
                "reference_chunk": chunk_id,
                "similar_chunks": formatted_results,
            }

            return json.dumps(response, indent=2)
        except Exception as e:
            error_msg = f"Similar code search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})

    def get_index_status(self) -> str:
        """Implementation of get_index_status tool."""
        try:
            index_manager = self.get_index_manager()
            stats = index_manager.get_stats()

            # Return model info without triggering API calls or heavy imports
            provider = os.environ.get("EMBEDDING_PROVIDER", "openai")
            model_info = {
                "provider": provider,
                "model_name": os.environ.get(
                    "EMBEDDING_MODEL", "text-embedding-3-small"
                )
                if provider == "openai"
                else os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                "status": "configured",
            }

            response = {
                "index_statistics": stats,
                "model_information": model_info,
                "storage_directory": str(get_storage_dir()),
            }

            return json.dumps(response, indent=2)
        except Exception as e:
            error_msg = f"Status check failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})

    def list_projects(self) -> str:
        """Implementation of list_projects tool."""
        try:
            base_dir = get_storage_dir()
            projects_dir = base_dir / "projects"

            if not projects_dir.exists():
                return json.dumps(
                    {"projects": [], "count": 0, "message": "No projects indexed yet"}
                )

            projects = []
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    info_file = project_dir / "project_info.json"
                    if info_file.exists():
                        with open(info_file) as f:
                            project_info = json.load(f)

                        stats_file = project_dir / "index" / "stats.json"
                        if stats_file.exists():
                            with open(stats_file) as f:
                                stats = json.load(f)
                            project_info["index_stats"] = stats

                        projects.append(project_info)

            return json.dumps(
                {
                    "projects": projects,
                    "count": len(projects),
                    "current_project": self._current_project,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return json.dumps({"error": str(e)})

    def switch_project(self, project_path: str) -> str:
        """Implementation of switch_project tool."""
        try:
            project_path = Path(project_path).resolve()
            if not project_path.exists():
                return json.dumps(
                    {"error": f"Project path does not exist: {project_path}"}
                )

            project_dir = self.get_project_storage_dir(str(project_path))
            index_dir = project_dir / "index"

            if not index_dir.exists() or not (index_dir / "code.index").exists():
                return json.dumps(
                    {
                        "error": f"Project not indexed: {project_path}",
                        "suggestion": f"Run index_directory('{project_path}') first",
                    }
                )

            self._current_project = str(project_path)
            self._index_manager = None
            self._searcher = None

            info_file = project_dir / "project_info.json"
            project_info = {}
            if info_file.exists():
                with open(info_file) as f:
                    project_info = json.load(f)

            logger.info(f"Switched to project: {project_path.name}")

            return json.dumps(
                {
                    "success": True,
                    "message": f"Switched to project: {project_path.name}",
                    "project_info": project_info,
                }
            )
        except Exception as e:
            logger.error(f"Error switching project: {e}")
            return json.dumps({"error": str(e)})

    def index_test_project(self) -> str:
        """Implementation of index_test_project tool."""
        try:
            logger.info("Indexing built-in test project")

            server_dir = Path(__file__).parent
            test_project_path = (
                server_dir.parent / "tests" / "test_data" / "python_project"
            )

            if not test_project_path.exists():
                return json.dumps(
                    {
                        "success": False,
                        "error": "Test project not found. The sample project may not be available.",
                    }
                )

            result = self.index_directory(str(test_project_path))
            result_data = json.loads(result)

            if "error" not in result_data:
                result_data["demo_info"] = {
                    "project_type": "Sample Python Project",
                    "includes": [
                        "Authentication module (user login, password hashing)",
                        "Database module (connections, queries, transactions)",
                        "API module (HTTP handlers, request validation)",
                        "Utilities (helpers, validation, configuration)",
                    ],
                    "sample_searches": [
                        "user authentication functions",
                        "database connection code",
                        "HTTP API handlers",
                        "input validation",
                        "error handling patterns",
                    ],
                }

            return json.dumps(result_data, indent=2)
        except Exception as e:
            logger.error(f"Error indexing test project: {e}")
            return json.dumps({"success": False, "error": str(e)})

    def clear_index(self) -> str:
        """Implementation of clear_index tool."""
        try:
            if self._current_project is None:
                return json.dumps(
                    {
                        "error": "No project is currently active. Use index_directory() to index a project first."
                    }
                )

            index_manager = self.get_index_manager()
            index_manager.clear_index()

            response = {"success": True, "message": "Search index cleared successfully"}

            logger.info("Search index cleared")
            return json.dumps(response, indent=2)
        except Exception as e:
            error_msg = f"Clear index failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})
