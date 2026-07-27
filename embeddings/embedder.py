"""Code embedding wrapper using EmbeddingGemma model."""

import os
import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional
from dataclasses import dataclass, replace
import numpy as np

from chunking.code_chunk import CodeChunk
from common_utils import get_storage_dir


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embedding: np.ndarray
    chunk_id: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EffectiveEmbeddingConfig:
    """Provider, model, and content mode used by one embedding pipeline."""

    provider: str
    model_name: str
    content_mode: str
    output_dimension: Optional[int] = None
    input_type_enabled: bool = False


_DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "voyage": "voyage-4-large",
    "voyage-code-3": "voyage-code-3",
    "voyage-context": "voyage-context-3",
    "jina": "jinaai/jina-code-embeddings-0.5b",
    "jina-code": "jinaai/jina-code-embeddings-0.5b",
    "local": "sentence-transformers/all-MiniLM-L6-v2",
    "gemma": "google/embeddinggemma-300m",
}
_LOCAL_MODEL_PROVIDERS = frozenset(("gemma", "jina", "jina-code", "local"))
_VOYAGE_INPUT_TYPE_PROVIDERS = frozenset(
    ("voyage", "voyage-code-3", "voyage-context")
)
_KNOWN_OUTPUT_DIMENSIONS = {
    "google/embeddinggemma-300m": 768,
    "jinaai/jina-code-embeddings-0.5b": 896,
    "jinaai/jina-code-embeddings-1.5b": 1536,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "voyage-3-large": 1024,
    "voyage-3-lite": 512,
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "voyage-4-lite": 1024,
    "voyage-code-3": 1024,
    "voyage-context-3": 1024,
}
_JINA_MATRYOSHKA_DIMENSIONS = {
    "jinaai/jina-code-embeddings-0.5b": frozenset(
        (64, 128, 256, 512, 896)
    ),
    "jinaai/jina-code-embeddings-1.5b": frozenset(
        (128, 256, 512, 1024, 1536)
    ),
}
_BUILTIN_PROVIDER_MODELS = {
    "gemma": frozenset(("google/embeddinggemma-300m",)),
    "jina": frozenset(_JINA_MATRYOSHKA_DIMENSIONS),
    "jina-code": frozenset(_JINA_MATRYOSHKA_DIMENSIONS),
    "openai": frozenset(
        (
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        )
    ),
    "voyage": frozenset(
        (
            "voyage-3-large",
            "voyage-3-lite",
            "voyage-4",
            "voyage-4-large",
            "voyage-4-lite",
            "voyage-code-3",
        )
    ),
    "voyage-code-3": frozenset(("voyage-code-3",)),
    "voyage-context": frozenset(("voyage-context-3",)),
}
_CUSTOM_REMOTE_MODEL_PROVIDERS = frozenset(
    ("openai", "voyage", "voyage-code-3", "voyage-context")
)
_KNOWN_MODEL_PROVIDERS = {
    model_name: frozenset(
        provider
        for provider, provider_models in _BUILTIN_PROVIDER_MODELS.items()
        if model_name in provider_models
    )
    for model_name in {
        model
        for models in _BUILTIN_PROVIDER_MODELS.values()
        for model in models
    }
}


def _parse_output_dimension(value: Any, *, source: str) -> Optional[int]:
    """Parse a positive embedding dimension from config or persisted JSON."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{source} output dimension must be a positive integer")
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} output dimension must be a positive integer"
        ) from exc
    if dimension <= 0 or (
        isinstance(value, float) and not value.is_integer()
    ):
        raise ValueError(f"{source} output dimension must be a positive integer")
    return dimension


def _configured_output_dimension(
    configuration: EffectiveEmbeddingConfig,
) -> Optional[int]:
    if configuration.output_dimension is not None:
        return _parse_output_dimension(
            configuration.output_dimension,
            source="configured",
        )
    if configuration.provider in {"jina", "jina-code"}:
        truncate_dimension = os.environ.get("JINA_TRUNCATE_DIM", "").strip()
        if truncate_dimension:
            return _parse_output_dimension(
                truncate_dimension,
                source="JINA_TRUNCATE_DIM",
            )
    return _KNOWN_OUTPUT_DIMENSIONS.get(configuration.model_name)


def resolve_embedding_config(
    *,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    content_mode: Optional[str] = None,
    output_dimension: Optional[int] = None,
    input_type_enabled: Optional[bool] = None,
    stored: Optional[Mapping[str, Any]] = None,
) -> EffectiveEmbeddingConfig:
    """Resolve explicit, stored, and ambient configuration by precedence."""
    stored = stored or {}
    stored_provider = str(stored.get("embedding_provider") or "").strip().lower()
    ambient_provider = (
        os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
    )
    effective_content_mode = (
        content_mode
        or stored.get("content_mode")
        or os.environ.get("CONTENT_MODE")
        or "code"
    )
    effective_content_mode = str(effective_content_mode).strip().lower() or "code"

    explicit_provider = (provider or "").strip().lower()
    provider_is_explicit = bool(explicit_provider)
    effective_provider = explicit_provider
    if not provider_is_explicit:
        effective_provider = stored_provider
    if not effective_provider:
        effective_provider = ambient_provider
    if not effective_provider:
        if os.environ.get("VOYAGE_API_KEY"):
            effective_provider = (
                "voyage-context"
                if effective_content_mode == "docs"
                else "voyage"
            )
        else:
            effective_provider = "local"

    model_environment = (
        "LOCAL_EMBEDDING_MODEL"
        if effective_provider in _LOCAL_MODEL_PROVIDERS
        else "EMBEDDING_MODEL"
    )
    stored_model = str(stored.get("embedding_model") or "").strip()
    explicit_model_name = (model_name or "").strip()
    model_is_explicit = bool(explicit_model_name)
    effective_model_name = explicit_model_name
    if (
        not model_is_explicit
        and stored_model
        and effective_provider == stored_provider
    ):
        effective_model_name = stored_model
    ambient_provider_is_compatible = (
        not ambient_provider or ambient_provider == effective_provider
    )
    ambient_model_name = ""
    if ambient_provider_is_compatible:
        ambient_model_name = os.environ.get(model_environment, "").strip()
    effective_model_name = (
        effective_model_name
        or ambient_model_name
        or _DEFAULT_EMBEDDING_MODELS.get(
            effective_provider,
            "(provider-default)",
        )
    )
    ambient_model_is_compatible = (
        not ambient_model_name or ambient_model_name == effective_model_name
    )
    dimension_is_explicit = output_dimension is not None
    dimension_source = "explicit"
    dimension_value: Any = output_dimension
    if not dimension_is_explicit and (
        effective_provider == stored_provider
        and effective_model_name == stored_model
        and "embedding_dimension" in stored
    ):
        dimension_source = "stored"
        dimension_value = stored.get("embedding_dimension")
    if (
        dimension_value is None
        and ambient_provider_is_compatible
        and ambient_model_is_compatible
        and os.environ.get("EMBEDDING_DIMENSION", "").strip()
    ):
        dimension_source = "EMBEDDING_DIMENSION"
        dimension_value = os.environ["EMBEDDING_DIMENSION"]
    requested_dimension = _parse_output_dimension(
        dimension_value,
        source=dimension_source,
    )
    if input_type_enabled is not None and not isinstance(
        input_type_enabled,
        bool,
    ):
        raise ValueError("input_type_enabled must be a boolean")
    stored_input_type = stored.get("embedding_input_type_enabled")
    stored_identity_matches = (
        effective_provider == stored_provider
        and effective_model_name == stored_model
    )
    if input_type_enabled is not None:
        effective_input_type_enabled = input_type_enabled
    elif stored_identity_matches and stored_input_type is not None:
        if not isinstance(stored_input_type, bool):
            raise ValueError(
                "stored embedding_input_type_enabled must be a boolean"
            )
        effective_input_type_enabled = stored_input_type
    else:
        effective_input_type_enabled = (
            os.environ.get("VOYAGE_INPUT_TYPE", "off").strip().lower()
            == "on"
        )
    if effective_provider == "voyage-context":
        # The contextualized endpoint always receives document/query modes:
        # encode_grouped sends "document" and encode defaults to "query".
        # A false identity would describe behavior this provider cannot have.
        effective_input_type_enabled = True
    elif effective_provider not in _VOYAGE_INPUT_TYPE_PROVIDERS:
        effective_input_type_enabled = False

    available_providers = sorted(_PROVIDER_REGISTRY)
    allowed_models = _BUILTIN_PROVIDER_MODELS.get(effective_provider)
    known_model_providers = _KNOWN_MODEL_PROVIDERS.get(effective_model_name)
    provider_model_is_invalid = bool(
        known_model_providers
        and effective_provider not in known_model_providers
    )
    custom_remote_model_without_contract = bool(
        allowed_models is not None
        and effective_model_name not in allowed_models
        and not known_model_providers
        and (
            effective_provider not in _CUSTOM_REMOTE_MODEL_PROVIDERS
            or requested_dimension is None
        )
    )
    if (
        effective_provider not in _PROVIDER_REGISTRY
        or provider_model_is_invalid
        or custom_remote_model_without_contract
    ):
        raise ValueError(
            "Unsupported provider/model configuration: "
            f"provider={effective_provider!r}, model={effective_model_name!r}. "
            f"Available providers: {available_providers}"
        )
    known_dimension = _KNOWN_OUTPUT_DIMENSIONS.get(effective_model_name)
    if requested_dimension is not None:
        matryoshka_dimensions = (
            _JINA_MATRYOSHKA_DIMENSIONS.get(effective_model_name)
            if effective_provider in {"jina", "jina-code"}
            else None
        )
        if (
            matryoshka_dimensions is not None
            and requested_dimension not in matryoshka_dimensions
        ):
            supported = ", ".join(
                str(value) for value in sorted(matryoshka_dimensions)
            )
            raise ValueError(
                f"Configured output dimension {requested_dimension} is not a "
                f"supported Matryoshka dimension for "
                f"{effective_model_name!r}; choose one of: {supported}"
            )
        if (
            matryoshka_dimensions is None
            and known_dimension is not None
            and requested_dimension != known_dimension
        ):
            raise ValueError(
                f"Configured output dimension {requested_dimension} "
                f"conflicts with the known output dimension "
                f"{known_dimension} for {effective_model_name!r}"
            )
    configuration = EffectiveEmbeddingConfig(
        provider=effective_provider,
        model_name=effective_model_name,
        content_mode=effective_content_mode,
        output_dimension=requested_dimension,
        input_type_enabled=effective_input_type_enabled,
    )
    return replace(
        configuration,
        output_dimension=_configured_output_dimension(configuration),
    )


# ---------------------------------------------------------------------------
# Provider registry (R12)
#
# Pre-R12: CodeEmbedder.__init__ was a 6-branch if/elif over EMBEDDING_PROVIDER.
# Each branch handled its own model_name default, env-var reads, and
# constructor signature. Adding a new provider meant editing __init__.
# The registry pattern below decouples provider implementations from the
# wrapper, letting each provider self-register via @register_provider and
# letting the wrapper dispatch by string name.
# ---------------------------------------------------------------------------

# A factory takes (model_name, cache_dir, device) and returns the embedding
# model instance. Each provider's factory owns its own EMBEDDING_MODEL /
# LOCAL_EMBEDDING_MODEL / API_KEY defaults.
ProviderFactory = Callable[..., Any]
_PROVIDER_REGISTRY: Dict[str, ProviderFactory] = {}


def register_provider(*names: str):
    """Register a factory under one or more provider names.

    Multiple names support aliases — e.g., `@register_provider("jina",
    "jina-code")` exposes the same factory under both strings.

    Idempotent re-registration overwrites; this is intentional so tests
    can swap factories without monkey-patching the dict directly.
    """
    def decorator(fn: ProviderFactory) -> ProviderFactory:
        for name in names:
            _PROVIDER_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def list_providers() -> list[str]:
    """Return the registered provider names, sorted. Used in error
    messages and exposed for inspection."""
    return sorted(_PROVIDER_REGISTRY)


# Each provider's factory is registered below. The functions are kept tiny
# and self-contained — they own only the provider-specific setup that used
# to live inline in the giant if/elif.

@register_provider("openai")
def _factory_openai(model_name: str, cache_dir: str, device: str) -> Any:
    from embeddings.openai_embedder import OpenAIEmbeddingModel
    model_name = model_name or os.environ.get(
        "EMBEDDING_MODEL", "text-embedding-3-small"
    )
    return OpenAIEmbeddingModel(model_name=model_name)


@register_provider("voyage")
def _factory_voyage(model_name: str, cache_dir: str, device: str) -> Any:
    """voyage-4-large via the standard /embeddings endpoint.

    +0.053 weighted-avg MRR over voyage-context-3 across 4 languages,
    102 queries (2026-04-08).
    """
    from embeddings.openai_embedder import OpenAIEmbeddingModel
    model_name = model_name or os.environ.get(
        "EMBEDDING_MODEL", "voyage-4-large"
    )
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    return OpenAIEmbeddingModel(
        api_key=api_key,
        model_name=model_name,
        base_url="https://api.voyageai.com/v1",
    )


@register_provider("voyage-code-3")
def _factory_voyage_code3(model_name: str, cache_dir: str, device: str) -> Any:
    """voyage-code-3 via the standard /v1/embeddings endpoint (non-default).

    Uses the same OpenAI-compatible client path as the "voyage" provider.
    Reads VOYAGE_API_KEY + EMBEDDING_MODEL (defaults to "voyage-code-3").

    A/B vs voyage-4-large (PSM-full, n=102 golden + 183 harvested, rerank=off,
    2026-05-15): aggregate CI includes zero; per-subproject CI excludes zero on
    mithrandir TypeScript (+0.119 MRR) and nix declarative config (-0.091 MRR).
    Production default stays voyage-4-large. Enable with
    EMBEDDING_PROVIDER=voyage-code-3 for TypeScript-heavy corpora.
    See docs/findings/2026-05-15-voyage-code-3-ab-finding.md.
    """
    from embeddings.openai_embedder import OpenAIEmbeddingModel
    model_name = model_name or os.environ.get(
        "EMBEDDING_MODEL", "voyage-code-3"
    )
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    return OpenAIEmbeddingModel(
        api_key=api_key,
        model_name=model_name,
        base_url="https://api.voyageai.com/v1",
    )


@register_provider("voyage-context")
def _factory_voyage_context(model_name: str, cache_dir: str, device: str) -> Any:
    from embeddings.voyage_context_embedder import VoyageContextEmbedder
    model_name = model_name or os.environ.get(
        "EMBEDDING_MODEL", "voyage-context-3"
    )
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    return VoyageContextEmbedder(api_key=api_key, model_name=model_name)


@register_provider("jina", "jina-code")
def _factory_jina(
    model_name: str,
    cache_dir: str,
    device: str,
    *,
    output_dimension: Optional[int] = None,
) -> Any:
    from embeddings.jina_code_embedder import JinaCodeEmbedder
    model_name = model_name or os.environ.get(
        "LOCAL_EMBEDDING_MODEL", "jinaai/jina-code-embeddings-0.5b"
    )
    truncate_dim = output_dimension
    if truncate_dim is None:
        truncate_dim_str = os.environ.get("JINA_TRUNCATE_DIM", "")
        truncate_dim = int(truncate_dim_str) if truncate_dim_str else None
    return JinaCodeEmbedder(
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        truncate_dim=truncate_dim,
    )


@register_provider("local")
def _factory_local(model_name: str, cache_dir: str, device: str) -> Any:
    from embeddings.sentence_transformer import SentenceTransformerModel
    model_name = model_name or os.environ.get(
        "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return SentenceTransformerModel(
        model_name=model_name, cache_dir=cache_dir, device=device,
    )


@register_provider("gemma")
def _factory_gemma(model_name: str, cache_dir: str, device: str) -> Any:
    from embeddings.gemma import GemmaEmbeddingModel
    # Gemma constructor doesn't accept model_name; keep the original
    # behavior of ignoring it.
    return GemmaEmbeddingModel(cache_dir=cache_dir, device=device)


def _resolve_provider_name() -> str:
    """Resolve the active provider name from env + sensible fallbacks.

    Same precedence as pre-R12: explicit EMBEDDING_PROVIDER wins; otherwise
    auto-detect Voyage when its documented key is present; otherwise use the
    credential-free local provider. OpenAI remains an explicit opt-in because
    an ambient OPENAI_API_KEY must not silently enable source-code egress.
    """
    return resolve_embedding_config().provider


# ---------------------------------------------------------------------------
# Metadata dedup helper (R12)
#
# Pre-R12: three byte-for-byte-identical chunk_id + metadata + EmbeddingResult
# construction blocks at embed_chunk:276-302, embed_chunks:344-370, and
# embed_chunks_grouped:423-449. Adding a new metadata field meant editing
# three places.
# ---------------------------------------------------------------------------


def _make_embedding_result(chunk: CodeChunk, embedding: Any) -> EmbeddingResult:
    """Build an EmbeddingResult from a chunk + its computed embedding.

    Single source of truth for chunk_id format + metadata shape. All three
    embedder code paths route through this so the schema stays consistent.
    """
    chunk_id = (
        f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}:"
        f"{chunk.chunk_type}"
    )
    if chunk.name:
        chunk_id += f":{chunk.name}"

    content_preview = (
        chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
    )
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
        "content_preview": content_preview,
        "full_content": chunk.content,
    }
    return EmbeddingResult(
        embedding=embedding, chunk_id=chunk_id, metadata=metadata,
    )


class CodeEmbedder:
    """Wrapper for embedding code chunks."""

    def __init__(
        self,
        model_name: str = "",
        cache_dir: Optional[str] = None,
        device: str = "auto",
        configuration: Optional[EffectiveEmbeddingConfig] = None,
    ):
        if not cache_dir:
            cache_dir = str(get_storage_dir() / "models")
        self.device = device
        self._logger = logging.getLogger(__name__)

        # R12: provider dispatch via registry instead of 6-branch if/elif.
        # Each factory owns its own model_name + API-key defaults.
        effective_config = configuration or resolve_embedding_config(
            model_name=model_name or None,
        )
        if effective_config.provider == "voyage-context":
            effective_config = replace(
                effective_config,
                input_type_enabled=True,
            )
        elif effective_config.provider not in _VOYAGE_INPUT_TYPE_PROVIDERS:
            effective_config = replace(
                effective_config,
                input_type_enabled=False,
            )
        provider = effective_config.provider
        factory = _PROVIDER_REGISTRY.get(provider)
        if factory is None:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER: {provider!r}. "
                f"Available: {list_providers()}"
            )
        if provider in {"jina", "jina-code"}:
            self._model = factory(
                effective_config.model_name,
                cache_dir,
                device,
                output_dimension=effective_config.output_dimension,
            )
        else:
            self._model = factory(
                effective_config.model_name,
                cache_dir,
                device,
            )
        # Provider name is recorded for observability; factories own their
        # model_name resolution and the model itself knows its name.
        self._provider = provider
        resolved_model_name = getattr(self._model, "model_name", None) or getattr(
            self._model, "_model_name", None,
        ) or effective_config.model_name or "(unknown)"
        output_dimension = _configured_output_dimension(effective_config)
        if output_dimension is None:
            output_dimension = int(self._model.get_embedding_dimension())
        self._configuration = replace(
            effective_config,
            model_name=resolved_model_name,
            output_dimension=output_dimension,
        )
        # Resolved (provider, model) pair namespaces the query-embedding
        # caches. The server constructs one CodeEmbedder per project and
    # projects can use different providers/models/input-type modes
    # ("dual-model workflows"); a cache keyed on query text alone returned
    # another model's vector after a project switch — a loud dim-mismatch
    # when dimensions differ, silently wrong similarities when they match
    # (voyage-4-large and voyage-code-3 are both 1024-d).
        self._resolved_model_name = resolved_model_name
        self._logger.info(
            f"Embedding provider: {provider}, model: {resolved_model_name}"
        )

    @property
    def configuration(self) -> EffectiveEmbeddingConfig:
        """Exact provider/model/content-mode/dimension used by this instance."""
        return self._configuration

    @property
    def model(self):
        """Get the underlying embedding model."""
        return self._model.model

    # Sibling context: populated by embed_chunks/embed_chunks_grouped before
    # calling create_embedding_content. Maps relative_path -> list of sibling
    # chunk names in the same file. Approximates Voyage's contextualized
    # embeddings for non-contextual models (Jina, local).
    _sibling_context: Dict[str, list] = {}

    # Tier 2C (2026-05-24): LLM-generated context map, lazy-loaded from
    # the JSON file at LLM_CONTEXT_PATH. Keyed by chunk_id; value is the
    # context paragraph to prepend in place of the simple header. Cached
    # at the class level (shared across embedder instances) because the
    # file is read-only and identical across embedders for one index run.
    # Sentinel `None` means "not yet attempted"; `{}` means "load failed
    # or file empty" (don't retry).
    _llm_context_map: Optional[Dict[str, str]] = None
    _llm_context_map_path: Optional[str] = None

    @classmethod
    def _load_llm_context_map(cls, path: str) -> Dict[str, str]:
        """Lazy-load the LLM context paragraphs JSON.

        Memoizes by path so multiple embedder instances share one load.
        Returns an empty dict on any load failure (file missing, invalid
        JSON, type mismatch) so the caller's fallback path fires
        naturally.
        """
        if cls._llm_context_map is not None and cls._llm_context_map_path == path:
            return cls._llm_context_map
        import json
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                cls._llm_context_map = {}
            else:
                cls._llm_context_map = {
                    str(k): str(v)
                    for k, v in data.items()
                    if isinstance(v, str) and v
                }
        except (OSError, ValueError) as e:
            logger = logging.getLogger(__name__)
            logger.warning(
                "[LLM_CONTEXT_LOAD] failed path=%s err=%r; falling back to simple header",
                path,
                e,
            )
            cls._llm_context_map = {}
        cls._llm_context_map_path = path
        return cls._llm_context_map

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
        #
        # Tier 2C (2026-05-24): LLM_CONTEXT_PATH env var lets operators
        # supply a pre-computed JSON map of {chunk_id: context_paragraph}
        # produced by `bench/research/generate_llm_contexts.py`. When set
        # AND the JSON contains the current chunk's id, the paragraph
        # replaces the simple "# From <path> - <type> <name>" header.
        # Falls back to the simple header on cache miss or JSON load
        # failure — graceful degradation.
        #
        # Status per ship-discipline rule 10: BLOCKED ON MEASUREMENT
        # until the A/B vs the simple-header baseline completes.
        # On SHIP -> remove the LLM_CONTEXT_PATH gate and bake the
        # LLM-context substrate into the default indexing pipeline.
        # On REVERT -> remove the env var path entirely + the
        # generate_llm_contexts.py helper.
        llm_context_replaced = False
        llm_context_path = os.environ.get("LLM_CONTEXT_PATH", "")
        if llm_context_path:
            llm_map = self._load_llm_context_map(llm_context_path)
            # Reconstruct chunk_id using the same format as
            # build_embedding_result() (single source of truth in this
            # module). Keep in sync if that format ever changes.
            cid = (
                f"{chunk.relative_path}:{chunk.start_line}-"
                f"{chunk.end_line}:{chunk.chunk_type}"
            )
            if chunk.name:
                cid += f":{chunk.name}"
            if cid in llm_map:
                content_parts.append(llm_map[cid])
                llm_context_replaced = True

        if not llm_context_replaced and os.environ.get("CONTEXTUAL_HEADERS", "on") == "on":
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

            # D1 (Plan: 2026-05-09 PSM follow-up, item 7): prop-interface
            # parent-component link. When an interface chunk's name ends
            # in "Props", append "(props for X)" to the contextual header
            # where X is the inferred parent component name. The signal
            # is that mithrandir TSX has 153 such interfaces, all sibling-
            # paired with a same-file component declaration.
            #
            # Implementation note: the suffix-strip heuristic is approximate
            # (no sibling lookup); D1's purpose is to validate whether the
            # extra identifier-token in the header moves retrieval at the
            # embedding level. A4 (post-Phase-E baseline) showed Voyage-4-
            # large doesn't weight identifier tokens enough to change
            # rankings on the chunker name-extraction change. D1 is the
            # falsifier check: if the more-targeted prop-interface signal
            # also doesn't move retrieval, the embedding ceiling — not the
            # chunker design — is the constraint.
            if (chunk.chunk_type == "interface"
                    and chunk.name
                    and chunk.name.endswith("Props")
                    and len(chunk.name) > 5):
                parent_component = chunk.name[:-5]
                content_parts[-1] = content_parts[-1] + f" (props for {parent_component})"

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

    # P4 (2026-06-10 roadmap): document-embedding cache controls. Row cap is
    # hard-coded (no env knob); at 1024-dim float32 ≈ 4KB/row, 200K rows ≈
    # 800MB ceiling, evicted oldest-first.
    _DOC_CACHE_MAX_ROWS: int = 200_000

    def _doc_encode_kwargs(self) -> Dict[str, Any]:
        """Shared encode kwargs for the document side (single + batch paths).

        Pre-P4 the single-chunk path skipped `input_type` while the batch
        path honored VOYAGE_INPUT_TYPE — the same content could embed
        differently depending on which path indexed it. Unified here.
        """
        encode_kwargs: Dict[str, Any] = {
            "prompt_name": "Retrieval-document",
            "show_progress_bar": False,
        }
        if self._configuration.input_type_enabled:
            encode_kwargs["input_type"] = "document"
        return encode_kwargs

    def _doc_cache_input_mode(self) -> str:
        """Cache-key component mirroring _doc_encode_kwargs' input_type."""
        return (
            "document+it"
            if self._configuration.input_type_enabled
            else "document"
        )

    def _maybe_evict_doc_cache(self, db) -> None:
        """Oldest-first eviction once the row cap is exceeded."""
        try:
            (count,) = db.execute(
                "SELECT COUNT(*) FROM doc_embedding_cache"
            ).fetchone()
            if count > self._DOC_CACHE_MAX_ROWS:
                db.execute(
                    "DELETE FROM doc_embedding_cache WHERE rowid IN ("
                    "SELECT rowid FROM doc_embedding_cache "
                    "ORDER BY created_at ASC LIMIT ?)",
                    (count - self._DOC_CACHE_MAX_ROWS,),
                )
                db.commit()
        except Exception:
            pass

    def _validate_output_dimension(
        self,
        embeddings: Any,
        *,
        operation: str,
    ) -> None:
        """Reject provider output that violates the published dimension."""
        expected = self._configuration.output_dimension
        if expected is None:
            raise ValueError(
                "Embedding output dimension is unresolved; refusing to "
                f"publish {operation} embeddings"
            )
        try:
            array = np.asarray(embeddings)
            received = int(array.shape[-1]) if array.ndim else 0
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{operation} embeddings do not have a rectangular output "
                f"with the expected {expected} dimensions"
            ) from exc
        if received != expected:
            raise ValueError(
                f"{operation} embeddings expected {expected} dimensions "
                f"but received {received}"
            )

    def _embed_documents_cached(self, contents: List[str]) -> List[np.ndarray]:
        """Encode document texts with a content-hash cache in front of the model.

        Key = (sha256(content), provider, model, input_mode) — the
        provider/model keying discipline from the PR #224 query-cache fix.
        Grouped/contextualized providers (voyage-context) never reach this
        path: embed_chunks_grouped short-circuits to encode_grouped first,
        and those vectors are document-context-dependent so caching them by
        content alone would be wrong. Both document paths use the same
        prompt_name constant, so it is not part of the key.

        Cache failures degrade to a plain encode; encode failures propagate
        (same contract as before).
        """
        import hashlib

        encode_kwargs = self._doc_encode_kwargs()
        db = self._get_disk_cache()
        if db is None:
            embeddings = self._model.encode(contents, **encode_kwargs)
            self._validate_output_dimension(
                embeddings,
                operation="document",
            )
            return list(embeddings)

        mode = self._doc_cache_input_mode()
        shas = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in contents]
        by_sha: Dict[str, np.ndarray] = {}

        try:
            unique = list(dict.fromkeys(shas))
            for j in range(0, len(unique), 500):
                keys = unique[j:j + 500]
                placeholders = ",".join("?" * len(keys))
                rows = db.execute(
                    "SELECT content_sha, embedding, dim FROM doc_embedding_cache "
                    "WHERE provider = ? AND model = ? AND input_mode = ? "
                    f"AND content_sha IN ({placeholders})",
                    (self._provider, self._resolved_model_name, mode, *keys),
                ).fetchall()
                for sha, blob, dim in rows:
                    vec = np.frombuffer(blob, dtype=np.float32).copy()
                    if (
                        vec.shape[0] == dim
                        and dim == self._configuration.output_dimension
                    ):
                        by_sha[sha] = vec
        except Exception:
            by_sha = {}

        # Encode only the misses, deduped within the batch (identical content
        # at two positions encodes once).
        miss_shas: List[str] = []
        miss_contents: List[str] = []
        seen_miss = set()
        for sha, content in zip(shas, contents):
            if sha not in by_sha and sha not in seen_miss:
                seen_miss.add(sha)
                miss_shas.append(sha)
                miss_contents.append(content)

        if miss_contents:
            fresh = self._model.encode(miss_contents, **encode_kwargs)
            self._validate_output_dimension(
                fresh,
                operation="document",
            )
            for sha, vec in zip(miss_shas, fresh):
                by_sha[sha] = np.asarray(vec, dtype=np.float32)
            try:
                db.executemany(
                    "INSERT OR REPLACE INTO doc_embedding_cache "
                    "(content_sha, provider, model, input_mode, embedding, dim) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (sha, self._provider, self._resolved_model_name, mode,
                         by_sha[sha].tobytes(), int(by_sha[sha].shape[0]))
                        for sha in miss_shas
                    ],
                )
                db.commit()
                self._maybe_evict_doc_cache(db)
            except Exception:
                pass

        hits = len(contents) - len(miss_contents)
        if hits:
            self._logger.info(
                "doc-embedding cache: %d/%d texts served from cache",
                hits, len(contents),
            )
        return [by_sha[sha] for sha in shas]

    def clear_document_cache(self):
        """Clear the persistent document-embedding cache.

        Not wired to clear_index: the cache is content-addressed and shared
        across projects by design (same content, same vector), so clearing
        one project's index must not evict other projects' entries.
        """
        db = self._get_disk_cache()
        if db is not None:
            try:
                db.execute("DELETE FROM doc_embedding_cache")
                db.commit()
            except Exception:
                pass

    def embed_chunk(self, chunk: CodeChunk) -> EmbeddingResult:
        """Generate embedding for a single code chunk.

        Args:
            chunk: Code chunk to embed

        Returns:
            EmbeddingResult with embedding and metadata
        """
        content = self.create_embedding_content(chunk)

        # P4: cached document encode (shared with the batch path).
        embedding = self._embed_documents_cached([content])[0]
        # R12: route through _make_embedding_result for consistent
        # chunk_id format + metadata shape across all three embed paths.
        return _make_embedding_result(chunk, embedding)

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

            # P4: cached document encode — only cache misses hit the model/API.
            batch_embeddings = self._embed_documents_cached(batch_contents)

            # Create results — R12 dedup
            for chunk, embedding in zip(batch, batch_embeddings):
                results.append(_make_embedding_result(chunk, embedding))

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
            self._validate_output_dimension(
                batch_embeddings,
                operation="grouped document",
            )

            # voyage-context API can return fewer embeddings than input chunks
            # when individual chunks are rejected (oversized, malformed, etc.).
            # Bare zip() would silently truncate, dropping the surviving chunks
            # at the tail without any error signal. Raise so the indexer's
            # batch-failure handler (search/incremental_indexer.py) sees the
            # mismatch as an exception instead of producing a silent partial
            # index. (Knowledge-base 2026-05-26: voyage-context dropped 5,886
            # of 7,886 chunks via this silent truncation path.)
            if len(batch_embeddings) != len(batch_chunks):
                raise ValueError(
                    f"voyage-context returned {len(batch_embeddings)} "
                    f"embeddings for {len(batch_chunks)} input chunks "
                    f"(batch starting at file index {batch_start}); the API "
                    "likely rejected one or more chunks. Surface as error "
                    "instead of silently truncating."
                )

            # Create results — R12 dedup
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                results.append(_make_embedding_result(chunk, embedding))

            if batch_start + batch_size < len(file_items):
                self._logger.info(
                    f"Processed {batch_start + batch_size}/{len(file_items)} files"
                )

        self._logger.info("Grouped embedding generation completed")
        return results

    # LRU query embedding cache (in-memory) + optional SQLite persistent cache.
    # In-memory cache: fast, dies with process. SQLite: survives restarts,
    # eliminates cold-start latency for Jina on CPU (~5s per query → instant).
    #
    # The in-memory cache is deliberately class-level so it survives the
    # per-project CodeEmbedder reconstruction the MCP server does on every
    # switch_project. Sharing is safe ONLY because keys are namespaced by
    # the instance's resolved (provider, model, input type) — see
    # _cache_namespace.
    # Pre-fix, the key was the bare query text, so switching between
    # projects with different providers returned the previous model's
    # embedding (cross-provider cache poisoning).
    _query_cache: OrderedDict = OrderedDict()
    _QUERY_CACHE_MAX: int = 256
    _disk_cache_db = None

    def _cache_namespace(self) -> str:
        """Provider, model, and input-type namespace for query cache keys.

        Uses the instance's resolved identity, NOT os.environ at call time:
        the server's env-var override during construction is restored before
        the first query, so the env no longer reflects this instance.
        """
        input_mode = (
            "query+it"
            if self._configuration.input_type_enabled
            else "query"
        )
        return (
            f"{self._provider}::{self._resolved_model_name}::{input_mode}"
        )

    def _query_cache_input_mode(self) -> str:
        return (
            "query+it"
            if self._configuration.input_type_enabled
            else "query"
        )

    def _get_disk_cache(self):
        """Lazy-init SQLite persistent query cache.

        Schema note: query_embeddings_v3 keys on (query_key, provider,
        model, input_mode). Earlier tables omitted at least input mode, so
        the same-dimension case could return a vector produced with different
        provider semantics. Obsolete tables are dropped on first open; this
        is a cache, so the only cost is re-warming.
        """
        if self._disk_cache_db is not None:
            return self._disk_cache_db
        try:
            import sqlite3
            cache_path = get_storage_dir() / "query_cache.db"
            db = sqlite3.connect(str(cache_path), check_same_thread=False)
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS query_embeddings_v3 (
                    query_key TEXT,
                    provider TEXT,
                    model TEXT,
                    input_mode TEXT,
                    embedding BLOB,
                    dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (query_key, provider, model, input_mode)
                )
            """)
            # P4 (2026-06-10 roadmap): content-hash-keyed DOCUMENT embedding
            # cache. Re-indexes and branch switches re-embed mostly-unchanged
            # content; identical (content, provider, model, input_mode) must
            # not hit the API twice. input_mode captures VOYAGE_INPUT_TYPE
            # ("document" vs "document+it") because the same text embeds
            # differently with input_type set. Cross-provider keying follows
            # the query-cache discipline (PR #224).
            db.execute("""
                CREATE TABLE IF NOT EXISTS doc_embedding_cache (
                    content_sha TEXT,
                    provider TEXT,
                    model TEXT,
                    input_mode TEXT,
                    embedding BLOB,
                    dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (content_sha, provider, model, input_mode)
                )
            """)
            db.execute("DROP TABLE IF EXISTS query_embeddings")
            db.execute("DROP TABLE IF EXISTS query_embeddings_v2")
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

        Both cache layers are keyed by
        (provider, model, input type, normalized query) so per-project
        provider switches never cross-contaminate.

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        query_key = query.strip().lower()
        namespace = self._cache_namespace()
        cache_key = (namespace, query_key)

        # Layer 1: in-memory LRU
        if cache_key in self._query_cache:
            cached_embedding = self._query_cache[cache_key]
            if (
                cached_embedding.shape[0]
                == self._configuration.output_dimension
            ):
                self._query_cache.move_to_end(cache_key)
                return cached_embedding
            del self._query_cache[cache_key]

        # Layer 2: SQLite persistent cache
        db = self._get_disk_cache()
        if db is not None:
            try:
                row = db.execute(
                    "SELECT embedding, dim FROM query_embeddings_v3 "
                    "WHERE query_key = ? AND provider = ? AND model = ? "
                    "AND input_mode = ?",
                    (
                        query_key,
                        self._provider,
                        self._resolved_model_name,
                        self._query_cache_input_mode(),
                    ),
                ).fetchone()
                if row:
                    embedding = np.frombuffer(row[0], dtype=np.float32).copy()
                    if (
                        embedding.shape[0] == row[1]
                        and row[1] == self._configuration.output_dimension
                    ):
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
        if self._configuration.input_type_enabled:
            encode_kwargs["input_type"] = "query"
        embedding = self._model.encode(
            [query], **encode_kwargs
        )[0]
        self._validate_output_dimension(
            embedding,
            operation="query",
        )

        # Store in both caches
        self._query_cache[cache_key] = embedding
        if len(self._query_cache) > self._QUERY_CACHE_MAX:
            self._query_cache.popitem(last=False)

        if db is not None:
            try:
                db.execute(
                    "INSERT OR REPLACE INTO query_embeddings_v3 "
                    "(query_key, provider, model, input_mode, embedding, dim) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        query_key,
                        self._provider,
                        self._resolved_model_name,
                        self._query_cache_input_mode(),
                        embedding.tobytes(),
                        embedding.shape[0],
                    ),
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
                db.execute("DELETE FROM query_embeddings_v3")
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
