"""Contracts for packaged, versioned synonym profiles."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from importlib import resources
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _clear_process_static_profile_state_after_test():
    yield

    from search.config import get_search_config
    from search.query_expansion import clear_synonym_cache

    get_search_config.cache_clear()
    clear_synonym_cache()


def test_default_profile_loads_packaged_generic_v1(monkeypatch):
    monkeypatch.delenv("CODE_SYNONYM_PROFILE", raising=False)
    monkeypatch.delenv("CODE_SYNONYMS_PATH", raising=False)

    from search.config import get_search_config
    from search.query_expansion import (
        clear_synonym_cache,
        expand_code_query,
        get_active_synonym_profile,
        get_active_synonym_profile_metadata,
    )

    get_search_config.cache_clear()
    clear_synonym_cache()

    profile = get_active_synonym_profile()
    assert profile.name == "generic"
    assert profile.version == 1
    assert profile.id == "generic-v1"
    assert get_active_synonym_profile_metadata() == {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }
    assert resources.files("search").joinpath(
        "profiles", "generic-v1.json"
    ).is_file()
    assert expand_code_query("auth retry") == 'auth retry authentication oauth jwt token credential login entra backoff retryable retry_delay 429 529'


def test_generic_profile_selected_explicitly(monkeypatch):
    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "generic")
    monkeypatch.delenv("CODE_SYNONYMS_PATH", raising=False)

    from search.config import get_search_config
    from search.query_expansion import (
        clear_synonym_cache,
        expand_code_query,
        get_active_synonym_profile,
        get_active_synonym_profile_metadata,
    )

    get_search_config.cache_clear()
    clear_synonym_cache()

    profile = get_active_synonym_profile()
    assert profile.name == "generic"
    assert profile.version == 1
    assert profile.id == "generic-v1"
    assert get_active_synonym_profile_metadata() == {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }
    assert resources.files("search").joinpath(
        "profiles", "generic-v1.json"
    ).is_file()
    assert "navigation" not in profile.synonyms
    assert "vlan" in profile.synonyms["network"]
    assert "oauth" in expand_code_query("auth").split()
    assert "nixpkgs" in profile.synonyms["nix"]
    assert expand_code_query("navigation") == "navigation"


def test_off_profile_disables_packaged_and_overlay_expansion(tmp_path, monkeypatch):
    overlay = tmp_path / "synonyms.json"
    overlay.write_text('{"auth": ["custom-auth"]}', encoding="utf-8")
    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "off")
    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(overlay))

    from search.config import get_search_config
    from search.query_expansion import (
        clear_synonym_cache,
        expand_code_query,
        get_active_synonym_profile,
        get_active_synonym_profile_metadata,
    )

    get_search_config.cache_clear()
    clear_synonym_cache()

    profile = get_active_synonym_profile()
    assert profile.name == "off"
    assert profile.version is None
    assert profile.id == "off"
    assert profile.synonyms == {}
    assert get_active_synonym_profile_metadata() == {
        "name": "off",
        "version": None,
        "id": "off",
    }
    assert expand_code_query("Auth  retry") == "Auth  retry"


def test_overlay_applies_after_selected_generic_profile(tmp_path, monkeypatch):
    overlay = tmp_path / "synonyms.json"
    overlay.write_text(
        json.dumps(
            {
                "auth": ["custom-auth"],
                "network": None,
                "telemetry": ["metrics"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "generic")
    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(overlay))

    from search.config import get_search_config
    from search.query_expansion import (
        _active_synonyms,
        clear_synonym_cache,
        expand_code_query,
        get_active_synonym_profile,
    )

    get_search_config.cache_clear()
    clear_synonym_cache()

    profile = get_active_synonym_profile()
    merged = _active_synonyms()
    assert profile.id == "generic-v1"
    assert merged["auth"] == ["custom-auth"]
    assert "network" not in merged
    assert merged["telemetry"] == ["metrics"]
    assert merged["error"] == profile.synonyms["error"]
    assert "navigation" not in merged
    assert expand_code_query("auth telemetry network") == (
        "auth telemetry network custom-auth metrics"
    )


def test_overlay_metadata_fingerprints_effective_synonyms_without_leaking_input(
    tmp_path,
    monkeypatch,
) -> None:
    first_overlay = tmp_path / "first-private-overlay.json"
    second_overlay = tmp_path / "second-private-overlay.json"
    first_term = "first-private-synonym"
    second_term = "second-private-synonym"
    first_overlay.write_text(
        json.dumps({"auth": [first_term]}),
        encoding="utf-8",
    )
    second_overlay.write_text(
        json.dumps({"auth": [second_term]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "generic")

    from search.config import get_search_config
    from search.query_expansion import (
        clear_synonym_cache,
        get_active_synonym_profile_metadata,
    )

    get_search_config.cache_clear()

    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(first_overlay))
    clear_synonym_cache()
    first_metadata = get_active_synonym_profile_metadata()

    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(second_overlay))
    clear_synonym_cache()
    second_metadata = get_active_synonym_profile_metadata()

    first_digest = first_metadata["effective_synonyms_sha256"]
    second_digest = second_metadata["effective_synonyms_sha256"]
    assert first_metadata["id"] == second_metadata["id"] == "generic-v1"
    assert len(first_digest) == len(second_digest) == 64
    assert all(character in "0123456789abcdef" for character in first_digest)
    assert all(character in "0123456789abcdef" for character in second_digest)
    assert first_digest != second_digest

    serialized_metadata = json.dumps([first_metadata, second_metadata])
    assert str(first_overlay) not in serialized_metadata
    assert str(second_overlay) not in serialized_metadata
    assert first_term not in serialized_metadata
    assert second_term not in serialized_metadata


def test_effective_synonym_digest_is_stable_across_mapping_order(
    tmp_path,
    monkeypatch,
) -> None:
    first_overlay = tmp_path / "first.json"
    second_overlay = tmp_path / "second.json"
    first_overlay.write_text(
        '{"alpha": ["one"], "beta": ["two"]}',
        encoding="utf-8",
    )
    second_overlay.write_text(
        '{"beta": ["two"], "alpha": ["one"]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "generic")

    from search.config import get_search_config
    from search.query_expansion import (
        clear_synonym_cache,
        get_active_synonym_profile_metadata,
    )

    get_search_config.cache_clear()

    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(first_overlay))
    clear_synonym_cache()
    first_digest = get_active_synonym_profile_metadata()[
        "effective_synonyms_sha256"
    ]

    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(second_overlay))
    clear_synonym_cache()
    second_digest = get_active_synonym_profile_metadata()[
        "effective_synonyms_sha256"
    ]

    assert first_digest == second_digest


@pytest.mark.parametrize("failure_mode", ["malformed", "missing"])
def test_failed_overlay_metadata_identifies_base_fallback_without_leaking_input(
    tmp_path,
    monkeypatch,
    failure_mode,
) -> None:
    empty_overlay = tmp_path / "empty.json"
    empty_overlay.write_text("{}", encoding="utf-8")
    failed_overlay = tmp_path / f"private-{failure_mode}-overlay.json"
    private_content = "{private-sentinel-overlay-content"
    if failure_mode == "malformed":
        failed_overlay.write_text(private_content, encoding="utf-8")

    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "generic")

    from search.config import get_search_config
    from search.query_expansion import (
        clear_synonym_cache,
        get_active_synonym_profile_metadata,
    )

    get_search_config.cache_clear()

    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(empty_overlay))
    clear_synonym_cache()
    base_policy_metadata = get_active_synonym_profile_metadata()

    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(failed_overlay))
    clear_synonym_cache()
    fallback_metadata = get_active_synonym_profile_metadata()

    assert fallback_metadata == base_policy_metadata
    serialized_metadata = json.dumps(fallback_metadata)
    assert str(failed_overlay) not in serialized_metadata
    assert private_content not in serialized_metadata


def test_invalid_profile_warns_and_falls_back_to_generic(monkeypatch, caplog):
    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "typo")

    from search.config import get_search_config
    from search.query_expansion import get_active_synonym_profile

    get_search_config.cache_clear()

    with caplog.at_level("WARNING", logger="search.config"):
        config = get_search_config()

    assert config.synonym_profile == "generic"
    assert get_active_synonym_profile().id == "generic-v1"
    assert "CODE_SYNONYM_PROFILE" in caplog.text
    assert "using default 'generic'" in caplog.text


def test_profile_selection_is_process_static_until_config_cache_clear(monkeypatch):
    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "off")

    from search.config import get_search_config
    from search.query_expansion import get_active_synonym_profile

    get_search_config.cache_clear()
    assert get_active_synonym_profile().id == "off"

    monkeypatch.setenv("CODE_SYNONYM_PROFILE", "generic")
    assert get_active_synonym_profile().id == "off"

    get_search_config.cache_clear()
    assert get_active_synonym_profile().id == "generic-v1"


def test_profile_identity_fields_are_immutable():
    from search.query_expansion import CODE_SYNONYMS, load_synonym_profile

    profile = load_synonym_profile("generic")

    with pytest.raises(FrozenInstanceError):
        profile.id = "generic-v2"
    with pytest.raises(FrozenInstanceError):
        profile.version = 2

    load_synonym_profile("off")
    assert load_synonym_profile("generic").synonyms is CODE_SYNONYMS


def test_search_response_reports_active_synonym_profile(
    tmp_path,
    monkeypatch,
) -> None:
    from common_utils import get_storage_dir
    import mcp_server.code_search_server as server_module
    from mcp_server.code_search_server import CodeSearchServer

    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    monkeypatch.setenv("CODE_SEARCH_QUERY_HISTORY", "off")
    get_storage_dir.cache_clear()
    monkeypatch.setattr(
        server_module,
        "_active_synonym_profile_metadata",
        lambda: {
            "name": "generic",
            "version": 1,
            "id": "generic-v1",
        },
    )
    server = CodeSearchServer()
    searcher = MagicMock()
    searcher.search.return_value = []
    searcher.index_manager.get_stats.return_value = {"total_chunks": 0}
    searcher._query_embedding_cache = {}
    searcher.last_reranker_metadata = {
        "applied": False,
        "reason": "not_invoked_no_candidates",
        "latency_ms": 0,
    }
    server.get_searcher = MagicMock(return_value=searcher)
    server._current_project = None

    response = json.loads(
        server.search_code(
            query="find authentication",
            auto_reindex=False,
        )
    )

    assert response["_metadata"]["synonym_profile"] == {
        "name": "generic",
        "version": 1,
        "id": "generic-v1",
    }
