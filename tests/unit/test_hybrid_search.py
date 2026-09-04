"""Tests for RRF fusion and hybrid search."""

import pytest


def test_rrf_fusion_boosts_documents_in_both_lists():
    """Documents appearing in both vector and BM25 results should rank higher."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8), ("doc_c", 0.7)]
    bm25_results = [("doc_b", -1.0), ("doc_d", -2.0), ("doc_a", -3.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    fused_ids = [chunk_id for chunk_id, score in fused]

    # doc_a and doc_b appear in both lists, should rank top
    assert fused_ids[0] in ("doc_a", "doc_b")
    assert fused_ids[1] in ("doc_a", "doc_b")
    # doc_c and doc_d appear in only one list, should rank lower
    assert "doc_c" in fused_ids
    assert "doc_d" in fused_ids


def test_rrf_fusion_handles_no_overlap():
    """Fusion with zero overlap should interleave by rank."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8)]
    bm25_results = [("doc_c", -1.0), ("doc_d", -2.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    assert len(fused) == 4


def test_rrf_fusion_handles_empty_bm25():
    """If BM25 returns nothing, fusion should return vector results only."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9), ("doc_b", 0.8)]
    bm25_results = []

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    assert len(fused) == 2
    assert fused[0][0] == "doc_a"


def test_weighted_rrf_bm25_heavy():
    """Higher BM25 weight should boost BM25-only docs over vector-only docs."""
    from search.searcher import reciprocal_rank_fusion

    # doc_a only in vector, doc_b only in bm25, both at rank 0
    vector_results = [("doc_a", 0.9)]
    bm25_results = [("doc_b", -1.0)]

    fused = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        k=60,
        vector_weight=0.3,
        bm25_weight=0.7,
    )
    fused_ids = [chunk_id for chunk_id, score in fused]

    # With bm25 weight 0.7 vs vector 0.3, doc_b should rank first
    assert fused_ids[0] == "doc_b"
    assert fused_ids[1] == "doc_a"


def test_weighted_rrf_vector_heavy():
    """Higher vector weight should boost vector-only docs over BM25-only docs."""
    from search.searcher import reciprocal_rank_fusion

    vector_results = [("doc_a", 0.9)]
    bm25_results = [("doc_b", -1.0)]

    fused = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        k=60,
        vector_weight=0.7,
        bm25_weight=0.3,
    )
    fused_ids = [chunk_id for chunk_id, score in fused]

    assert fused_ids[0] == "doc_a"
    assert fused_ids[1] == "doc_b"


def test_chunk_type_boosts_code_mode():
    """In code mode, function chunks should rank above section chunks at same RRF score."""
    from search.searcher import CHUNK_TYPE_BOOSTS

    boosts = CHUNK_TYPE_BOOSTS["code"]
    assert boosts["function"] > boosts["section"]
    assert boosts["method"] > boosts["section"]
    assert boosts["class"] > boosts["section"]
    assert boosts["decorated_definition"] > boosts["section"]


def test_chunk_type_boosts_docs_mode():
    """In docs mode, section chunks should rank above function chunks."""
    from search.searcher import CHUNK_TYPE_BOOSTS

    boosts = CHUNK_TYPE_BOOSTS["docs"]
    assert boosts["section"] > boosts["function"]
    assert boosts["section"] > boosts["method"]


def test_expand_query_adds_synonyms():
    """Query expansion should add known code-domain synonyms."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("authentication logic")
    assert "auth" in expanded.lower()
    assert "oauth" in expanded.lower() or "jwt" in expanded.lower()


def test_expand_query_passthrough_unknown():
    """Queries with no known synonyms should pass through unchanged."""
    from search.searcher import expand_code_query

    result = expand_code_query("foobar baz")
    assert result == "foobar baz"


def test_expand_query_stem_strips_whole_suffix_not_char_class():
    """Stem must strip a whole suffix, not a character class.

    Regression: prior implementation chained `token.rstrip("s").rstrip("ing")
    .rstrip("tion").rstrip("ed")`. Each rstrip(<chars>) removes a SET of
    trailing chars, not a suffix. So "navigations" stemmed to "naviga"
    (s -> ing-chars -> tion-chars -> ed-chars), never matching the
    synonym key "service". This test exercises the stem-equivalence
    path that the existing tests don't (they pass the key directly).
    """
    from search.searcher import expand_code_query, _query_stem

    # Direct stemmer assertions
    assert _query_stem("navigations") == "navigation"
    assert _query_stem("sensors") == "sensor"
    assert _query_stem("logging") == "logg"  # strips "ing" suffix cleanly
    assert _query_stem("powered") == "power"
    assert _query_stem("xy") == "xy"  # too short to stem
    # Stem-via-expansion: "services" should hit the "service" key and expand
    expanded = expand_code_query("services list")
    assert "systemd" in expanded.lower(), expanded


def _use_profile(name):
    """Select a synonym profile for one test and restore the default afterwards."""
    import os
    from search.config import get_search_config
    from search.query_expansion import clear_synonym_cache

    os.environ["CODE_SYNONYM_PROFILE"] = name
    get_search_config.cache_clear()
    clear_synonym_cache()


@pytest.fixture(autouse=True)
def _restore_synonym_profile(monkeypatch):
    from search.config import get_search_config
    from search.query_expansion import clear_synonym_cache

    monkeypatch.delenv("CODE_SYNONYM_PROFILE", raising=False)
    yield
    monkeypatch.delenv("CODE_SYNONYM_PROFILE", raising=False)
    get_search_config.cache_clear()
    clear_synonym_cache()


def test_expand_query_nix_domain():
    """Nix-domain synonyms should expand network/service queries."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("network configuration")
    assert "networking" in expanded.lower()
    assert "firewall" in expanded.lower() or "interface" in expanded.lower()

    expanded2 = expand_code_query("service daemon")
    assert "systemd" in expanded2.lower()



def test_expand_query_case_insensitive():
    """Query expansion should work regardless of case."""
    from search.searcher import expand_code_query

    expanded_lower = expand_code_query("authentication")
    expanded_upper = expand_code_query("Authentication")
    assert "oauth" in expanded_lower.lower()
    assert "oauth" in expanded_upper.lower()


def test_expand_query_stemming():
    """Query expansion should match stemmed forms."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("errors handling")
    assert "exception" in expanded.lower() or "raise" in expanded.lower()


def test_expand_query_no_double_expansion():
    """Expanded synonyms should not trigger further expansion chains."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("auth")
    tokens = expanded.lower().split()
    assert tokens.count("auth") <= 1


def test_expand_query_nix_specific_terms():
    """Nix-specific terms should expand to NixOS ecosystem synonyms."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("nix module")
    lower = expanded.lower()
    assert "nixos" in lower or "nixpkgs" in lower
    assert "imports" in lower or "options" in lower


def test_expand_query_nixos_enable():
    """'enable service' should expand to mkEnableOption and systemd terms.

    'enable' is a synonym of 'service', so it matches the service key first.
    mkEnableOption is added via the service synonym list.
    """
    from search.searcher import expand_code_query

    expanded = expand_code_query("enable service")
    lower = expanded.lower()
    assert "mkenableoption" in lower
    assert "systemd" in lower
    assert "serviceconfig" in lower


def test_expand_query_nix_derivation():
    """'derivation' should expand to build-related Nix terms.

    'derivation' is a synonym of 'package', so it matches the package key.
    stdenv and mkDerivation are added via the package synonym list.
    """
    from search.searcher import expand_code_query

    expanded = expand_code_query("derivation build")
    lower = expanded.lower()
    assert "stdenv" in lower or "mkderivation" in lower
    assert "buildinputs" in lower or "nativebuildinputs" in lower


def test_expand_query_nix_firewall():
    """'firewall' should expand to nftables and port terms."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("firewall rules")
    lower = expanded.lower()
    assert "nftables" in lower
    assert "allowedtcpports" in lower or "allowedudpports" in lower


def test_expand_query_nix_boot():
    """'boot' should expand to bootloader terms."""
    from search.searcher import expand_code_query

    expanded = expand_code_query("boot configuration")
    lower = expanded.lower()
    assert "bootloader" in lower or "grub" in lower or "systemd-boot" in lower


def test_chunk_type_boosts_nix_types():
    """Nix-specific chunk types (option, service_config) should get boosted in code mode."""
    from search.searcher import CHUNK_TYPE_BOOSTS

    boosts = CHUNK_TYPE_BOOSTS["code"]
    assert "option" in boosts, "option chunk type missing from code boosts"
    assert "service_config" in boosts, "service_config chunk type missing from code boosts"
    assert boosts["option"] > boosts["section"]
    assert boosts["service_config"] > boosts["section"]
