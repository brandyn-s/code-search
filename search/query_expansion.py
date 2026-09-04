"""Static code-domain query expansion and per-deployment overlays."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SynonymProfile:
    """A packaged synonym policy with a stable, versioned identity."""

    name: str
    version: Optional[int]
    id: str
    synonyms: Dict[str, List[str]]


_PROFILE_FILES = {
    "generic": "generic-v1.json",
}


@lru_cache(maxsize=len(_PROFILE_FILES) + 1)
def load_synonym_profile(name: str) -> SynonymProfile:
    """Load and validate one immutable packaged profile identity."""
    if name == "off":
        return SynonymProfile(name="off", version=None, id="off", synonyms={})
    try:
        filename = _PROFILE_FILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown synonym profile: {name!r}") from exc

    profile_path = resources.files("search").joinpath("profiles", filename)
    with profile_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    expected_id = filename.removesuffix(".json")
    if (
        not isinstance(payload, dict)
        or payload.get("name") != name
        or payload.get("id") != expected_id
        or payload.get("version") != 1
        or not isinstance(payload.get("synonyms"), dict)
    ):
        raise ValueError(f"Invalid packaged synonym profile: {filename}")

    synonyms: Dict[str, List[str]] = {}
    for key, values in payload["synonyms"].items():
        if (
            not isinstance(key, str)
            or not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError(f"Invalid synonyms in packaged profile: {filename}")
        synonyms[key] = list(values)

    return SynonymProfile(
        name=name,
        version=payload["version"],
        id=payload["id"],
        synonyms=synonyms,
    )


def get_active_synonym_profile() -> SynonymProfile:
    """Return the process-static profile selected by SearchConfig."""
    from search.config import get_search_config

    return load_synonym_profile(get_search_config().synonym_profile)


def get_active_synonym_profile_metadata() -> Dict[str, object]:
    """Return the active profile identity in a JSON-serializable shape."""
    profile = get_active_synonym_profile()
    metadata: Dict[str, object] = {
        "name": profile.name,
        "version": profile.version,
        "id": profile.id,
    }
    if profile.name != "off" and os.environ.get("CODE_SYNONYMS_PATH", ""):
        effective_synonyms = json.dumps(
            _active_synonyms(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        metadata["effective_synonyms_sha256"] = hashlib.sha256(
            effective_synonyms
        ).hexdigest()
    return metadata


# Backward-compatible constant: the packaged generic-v1 mapping. Query
# expansion reads the active profile (see get_active_synonym_profile), not
# this constant.
CODE_SYNONYMS = load_synonym_profile("generic").synonyms


# Order matters: longest suffix first so "navigations" -> strip "s" (not "tion").
# Each token has AT MOST one suffix stripped — we only need stem-equivalence to a
# CODE_SYNONYMS key, not full English morphology. `str.rstrip(<chars>)` is a
# character-class strip and was the prior implementation's bug
# (e.g. "navigation".rstrip("ing") removes trailing n, producing "navigatio";
# subsequent rstrip("tion") strips o/i/t producing "naviga" — never matches "navigation").
_QUERY_STEM_SUFFIXES = ("tion", "ing", "ed", "s")


def _query_stem(token: str) -> str:
    for suffix in _QUERY_STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


# Per-deployment synonym overlay. Lists replace a selected profile's key and
# null removes it. The profile id belongs in the cache key so test/tooling
# cache clears cannot accidentally reuse an overlay against another profile.
_SYNONYMS_OVERLAY_CACHE: Optional[
    Tuple[Tuple[str, str], Dict[str, List[str]]]
] = None


def clear_synonym_cache() -> None:
    """Clear process-static overlay state (primarily for tests/tooling)."""
    global _SYNONYMS_OVERLAY_CACHE
    _SYNONYMS_OVERLAY_CACHE = None


def _active_synonyms() -> Dict[str, List[str]]:
    """Return the synonym map, applying the CODE_SYNONYMS_PATH overlay."""
    global _SYNONYMS_OVERLAY_CACHE
    profile = get_active_synonym_profile()
    if profile.name == "off":
        return {}

    base_synonyms = profile.synonyms
    path = os.environ.get("CODE_SYNONYMS_PATH", "")
    if not path:
        return base_synonyms

    cache_key = (profile.id, path)
    if (
        _SYNONYMS_OVERLAY_CACHE is not None
        and _SYNONYMS_OVERLAY_CACHE[0] == cache_key
    ):
        return _SYNONYMS_OVERLAY_CACHE[1]
    merged: Dict[str, List[str]] = dict(base_synonyms)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key, value in data.items():
                norm_key = str(key).lower()
                if value is None:
                    merged.pop(norm_key, None)
                elif isinstance(value, list):
                    merged[norm_key] = [str(s) for s in value]
    except (OSError, ValueError) as e:
        logging.getLogger(__name__).warning(
            "CODE_SYNONYMS_PATH=%s load failed (%s); using %s profile",
            path,
            e,
            profile.id,
        )
        merged = base_synonyms
    _SYNONYMS_OVERLAY_CACHE = (cache_key, merged)
    return merged


def expand_code_query(query: str) -> str:
    """Expand a query with code-domain synonyms for better BM25 recall."""
    tokens = query.lower().split()
    expanded_tokens = list(tokens)

    synonyms_map = _active_synonyms()
    for token in tokens:
        stem = _query_stem(token)
        for key, synonyms in synonyms_map.items():
            if token == key or stem == key or token in synonyms:
                for syn in synonyms:
                    if syn.lower() not in [t.lower() for t in expanded_tokens]:
                        expanded_tokens.append(syn)
                break

    if expanded_tokens == tokens:
        return query  # No expansion happened, return original

    return " ".join(expanded_tokens)
