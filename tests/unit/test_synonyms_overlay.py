"""Tests for the CODE_SYNONYMS_PATH per-deployment overlay.

The built-in CODE_SYNONYMS map carries Corsair-specific daemon names; the
overlay lets a deployment extend or disable entries without a source
change. Default behavior (env unset) must stay byte-identical to the
built-in map — changing defaults is a measured retrieval change.
"""
from __future__ import annotations

import json

import search.searcher as searcher_module
from search.searcher import CODE_SYNONYMS, _active_synonyms, expand_code_query


def _reset_overlay_cache():
    searcher_module._SYNONYMS_OVERLAY_CACHE = None


def test_unset_env_returns_builtin_map(monkeypatch):
    monkeypatch.delenv("CODE_SYNONYMS_PATH", raising=False)
    _reset_overlay_cache()
    assert _active_synonyms() is CODE_SYNONYMS


def test_overlay_extends_and_overrides(tmp_path, monkeypatch):
    overlay = tmp_path / "synonyms.json"
    overlay.write_text(json.dumps({
        "telemetry": ["metricsd", "statsd"],          # new key
        "auth": ["sso", "saml"],                       # overrides built-in
    }))
    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(overlay))
    _reset_overlay_cache()

    merged = _active_synonyms()
    assert merged["telemetry"] == ["metricsd", "statsd"]
    assert merged["auth"] == ["sso", "saml"]
    # Untouched built-in keys pass through.
    assert merged["error"] == CODE_SYNONYMS["error"]

    expanded = expand_code_query("telemetry pipeline")
    assert "metricsd" in expanded.split()
    _reset_overlay_cache()


def test_null_disables_builtin_key(tmp_path, monkeypatch):
    """A non-Corsair deployment can switch off the daemon expansions."""
    overlay = tmp_path / "synonyms.json"
    overlay.write_text(json.dumps({"power": None, "camera": None}))
    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(overlay))
    _reset_overlay_cache()

    merged = _active_synonyms()
    assert "power" not in merged
    assert "camera" not in merged

    expanded = expand_code_query("power management")
    assert "internal-svc-18" not in expanded.split(), (
        "disabled key still expanded with Corsair daemon names"
    )
    _reset_overlay_cache()


def test_malformed_overlay_falls_back_to_builtins(tmp_path, monkeypatch):
    overlay = tmp_path / "synonyms.json"
    overlay.write_text("{not json")
    monkeypatch.setenv("CODE_SYNONYMS_PATH", str(overlay))
    _reset_overlay_cache()

    assert _active_synonyms() == CODE_SYNONYMS
    _reset_overlay_cache()


def test_default_expansion_behavior_unchanged(monkeypatch):
    """Ship-discipline guard: with the env unset, expansion output matches
    the built-in map's behavior exactly."""
    monkeypatch.delenv("CODE_SYNONYMS_PATH", raising=False)
    _reset_overlay_cache()
    expanded = expand_code_query("auth retry")
    for syn in CODE_SYNONYMS["auth"]:
        assert syn.lower() in expanded.lower().split()
