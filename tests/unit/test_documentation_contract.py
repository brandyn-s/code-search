"""Executable contracts preventing user-facing documentation drift."""

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
CLAUDE = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
ENV_REFERENCE = (REPO_ROOT / "docs" / "ENV_REFERENCE.md").read_text(
    encoding="utf-8"
)
DOCUMENT_PATHS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "docs" / "ENV_REFERENCE.md",
)


def test_readme_has_one_pipeline_diagram_and_one_copy_of_each_terminal_section() -> None:
    assert README.count("```mermaid") == 1
    assert len(re.findall(r"^## Troubleshooting$", README, flags=re.MULTILINE)) == 1
    assert (
        len(
            re.findall(
                r"^## Comparison to Alternatives$",
                README,
                flags=re.MULTILINE,
            )
        )
        == 1
    )


def test_chunk_merging_budget_is_documented_as_2500_nws_characters() -> None:
    assert "2,500 non-whitespace character budget" in README
    assert "400-2500 NWS budget" in README
    assert "1,500 non-whitespace character budget" not in README
    assert "400-1500 NWS budget" not in README

    assert "2500 NWS char budget" in CLAUDE
    assert "1500 NWS char budget" not in CLAUDE


def test_rrf_weights_match_each_content_mode() -> None:
    expected = "code 65/35, docs 70/30, all 50/50 (vector/BM25)"
    assert expected in README
    assert expected in CLAUDE
    assert expected in ENV_REFERENCE

    assert "50/50 for code" not in README
    assert "weighted RRF (50/50)" not in CLAUDE
    assert "Weighted RRF fusion (50/50)" not in ENV_REFERENCE


def test_historical_mrr_is_not_misrepresented_as_current_top_rank_accuracy() -> None:
    assert (
        "MRR aggregates reciprocal rank across queries; it does not by itself "
        "determine top-result accuracy, a typical rank, or the probability that "
        "any one query succeeds."
    ) in README
    assert "These are historical evaluation results, not current production guarantees." in README
    assert "frozen balanced public LocBench n=80 endpoint" in README
    assert "This establishes narrow superiority for this" in README
    assert "frozen file-localization endpoint, not general platform superiority." in README
    assert "Current-stack retrieval quality is **BLOCKED ON MEASUREMENT**" not in README

    forbidden_claims = (
        "A score of 0.828 means the correct answer is almost always the #1 result.",
        "right answer is typically at position #3-5",
        "Voyage-4-large scores 0.828 — position #1 almost every time.",
        "The correct file is typically the #1 result.",
        "gets the right answer 83% of the time.",
        "The correct file is typically at position #3-5.",
        "benchmarks/eval_v4/run_psm-full-voyage-multitarget/summary.json",
    )
    for claim in forbidden_claims:
        assert claim not in README


def test_architecture_lists_extracted_search_policy_modules() -> None:
    for module in (
        "fusion.py",
        "query_expansion.py",
        "result_models.py",
        "retrieval.py",
        "pipeline.py",
    ):
        assert module in README
        assert f"`search/{module}`" in CLAUDE


def test_process_static_environment_contracts_and_defaults_are_documented() -> None:
    defaults = {
        "CODE_SYNONYM_PROFILE": "corsair",
        "CODE_SYNONYMS_PATH": "unset",
        "CODE_SEARCH_LOG_LEVEL": "INFO",
        "CODE_SEARCH_LOG_QUERY_TEXT": "off",
        "CODE_SEARCH_QUERY_HISTORY": "metadata",
        "CODE_SEARCH_QUERY_RETENTION_DAYS": "30",
    }
    for document in (README, CLAUDE, ENV_REFERENCE):
        for variable, default in defaults.items():
            row_prefix = f"| `{variable}` | `{default}` |"
            assert row_prefix in document
        assert (
            "These settings are process-static: they are read once when the MCP "
            "server starts. Restart the MCP server after changing them."
        ) in document
        assert "`off`, `metadata`, or `full`" in document

    for document in (README, CLAUDE, ENV_REFERENCE):
        assert "`corsair`, `generic`, or `off`" in document
    assert (
        "Changing the default away from `corsair` is **BLOCKED ON MEASUREMENT**"
        in ENV_REFERENCE
    )


def test_custom_remote_embedding_dimension_contract_is_documented() -> None:
    for document in (README, CLAUDE, ENV_REFERENCE):
        assert "| `EMBEDDING_DIMENSION` | `unset` |" in document
        assert "custom remote embedding models" in document
        assert "positive output-dimension contract" in document


def _heading_anchors(markdown: str) -> set[str]:
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", markdown, flags=re.MULTILINE):
        plain = re.sub(r"[^\w\s-]", "", heading.casefold())
        anchors.add(re.sub(r"[\s-]+", "-", plain).strip("-"))
    return anchors


def test_all_local_markdown_links_resolve() -> None:
    failures: list[str] = []
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

    for document_path in DOCUMENT_PATHS:
        markdown = document_path.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(markdown):
            target = raw_target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme in {"http", "https", "mailto"}:
                continue

            relative_path = unquote(parsed.path)
            resolved = (
                document_path
                if relative_path == ""
                else (document_path.parent / relative_path).resolve()
            )
            if not resolved.exists():
                failures.append(
                    f"{document_path.relative_to(REPO_ROOT)} -> {target}: missing target"
                )
                continue

            if parsed.fragment and resolved.suffix.lower() == ".md":
                anchors = _heading_anchors(resolved.read_text(encoding="utf-8"))
                if parsed.fragment.casefold() not in anchors:
                    failures.append(
                        f"{document_path.relative_to(REPO_ROOT)} -> {target}: "
                        "missing heading"
                    )

    assert failures == []
