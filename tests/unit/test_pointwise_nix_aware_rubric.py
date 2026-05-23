"""R9: pin the Nix-aware rubric clause in the pointwise reranker prompt.

Before R9, the pointwise reranker prompt (sonnet_reranker.JUDGE_PROMPT)
had no domain awareness for Nix/NixOS — the dominant language in the
production corpus. The listwise reranker (RERANKER=listwise) already
had a dedicated Nix-aware rubric clause (PR #179), but pointwise is
the production default until listwise graduates per
docs/LISTWISE_CANARY.md.

This file pins the Nix-aware clause text so it can't be accidentally
removed without an explicit test update. The actual quality impact
needs paired-bootstrap CI on the harvested holdout before merge — this
test ONLY verifies the clause is wired into the prompt template.
"""
from __future__ import annotations

from search.sonnet_reranker import JUDGE_PROMPT


class TestPointwisePromptHasNixAwareness:
    """Pre-R9: zero Nix references in JUDGE_PROMPT. Post-R9: explicit
    clause matching the listwise reranker's domain note."""

    def test_prompt_mentions_nix(self):
        # Lowercased substring check so future rephrasings of the clause
        # (Nix vs NixOS, mkOption vs mk_option) don't break the test.
        assert "nix" in JUDGE_PROMPT.lower(), (
            "JUDGE_PROMPT must include a Nix domain note. Pre-R9 the "
            "pointwise reranker prompt was Nix-blind; listwise already "
            "had the clause via PR #179. Removing the clause without an "
            "explicit test update should be impossible."
        )

    def test_prompt_mentions_mkoption(self):
        """The clause should specifically flag the mkOption shape, since
        that's the primary signal for whether a Nix chunk is a definition
        vs a use site."""
        assert "mkoption" in JUDGE_PROMPT.lower()

    def test_prompt_mentions_option_or_binding_chunks(self):
        """The chunk taxonomy distinction is the load-bearing signal:
        `option` and `binding` chunks are primary, daemon impl is
        secondary. Reusing the listwise rubric's wording."""
        text = JUDGE_PROMPT.lower()
        assert "option" in text and "binding" in text

    def test_prompt_still_carries_baseline_scoring_rubric(self):
        """Regression: the 0-10 scoring scale must still be present.
        R9 added a domain note; it must not have replaced the core
        rubric."""
        assert "0-10" in JUDGE_PROMPT or "scale of 0-10" in JUDGE_PROMPT
        # Strong-anchor scoring exemplars.
        assert "10 =" in JUDGE_PROMPT
        assert "0 =" in JUDGE_PROMPT

    def test_prompt_format_placeholders_intact(self):
        """Regression: the format() call site uses {query}, {file_path},
        {content}. The added clause must not have collided with any of
        those placeholders or introduced spurious braces."""
        for placeholder in ("{query}", "{file_path}", "{content}"):
            assert placeholder in JUDGE_PROMPT, (
                f"{placeholder} placeholder missing from JUDGE_PROMPT"
            )
        # And the JSON example at the end uses escaped braces. Make sure
        # str.format() on it doesn't blow up — the escaped {{ }} should
        # render as literal { } in the output.
        rendered = JUDGE_PROMPT.format(
            query="x", file_path="x.py", content="def x(): pass",
        )
        assert '{"score"' in rendered  # escaped {{ becomes literal {
