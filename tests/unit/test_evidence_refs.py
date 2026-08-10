from types import SimpleNamespace

from search.evidence import EvidenceRef, SymbolRef, symbol_ref_from_search_result


def test_symbol_ref_is_path_normalized_and_deterministic():
    first = SymbolRef(
        "a" * 64,
        "b" * 40,
        "./src\\auth.py",
        "Method",
        "Auth.verify",
        10,
        20,
    )
    second = SymbolRef(
        "a" * 64,
        "b" * 40,
        "src/auth.py",
        "method",
        "Auth.verify",
        10,
        20,
    )
    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["id"] == (
        "sym:v1:228a411de5eb1f52b61bd1abb05f9b7b4d20680cd35163f3e8934a47ce6952a0"
    )


def test_evidence_is_bound_to_index_generation():
    symbol = SymbolRef(
        "a" * 64,
        "b" * 40,
        "src/auth.py",
        "method",
        "Auth.verify",
        10,
        20,
    )
    first = EvidenceRef(
        "a" * 64,
        "b" * 40,
        "c" * 64,
        "src/auth.py",
        10,
        20,
        "semantic_match",
        symbol,
    )
    second = EvidenceRef(
        "a" * 64,
        "b" * 40,
        "d" * 64,
        "src/auth.py",
        10,
        20,
        "semantic_match",
        symbol,
    )
    assert first.to_dict()["id"] != second.to_dict()["id"]


def test_search_result_symbol_ref_requires_canonical_qualified_name():
    result = SimpleNamespace(
        name="verify",
        parent_name="Auth",
        qualified_name="repo.src.auth.Auth.verify",
        relative_path="src/auth.py",
        file_path="",
        chunk_type="method",
        start_line=10,
        end_line=20,
    )
    ref = symbol_ref_from_search_result(
        result,
        repository_id="a" * 64,
        source_revision="b" * 40,
    )
    assert ref is not None
    assert ref.to_dict()["qualified_name"] == "repo.src.auth.Auth.verify"


def test_search_result_short_name_does_not_create_false_join():
    result = SimpleNamespace(
        name="verify",
        parent_name="Auth",
        relative_path="src/auth.py",
        file_path="",
        chunk_type="method",
        start_line=10,
        end_line=20,
    )
    assert (
        symbol_ref_from_search_result(
            result,
            repository_id="a" * 64,
            source_revision="b" * 40,
        )
        is None
    )
