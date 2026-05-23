"""Tests pinning the rule-10 closing-phrase detector contract.

The detector lives at `.claude/lib/rule10_check.py` and is consumed by
the Stop hook at `.claude/hooks/check_rule10_closing.sh`. It also has a
prompt-level analog in `.claude/skills/goal-disciplined.md` — these tests
ensure the structural mechanism agrees with the prompt's expectations.

Pre-rule-10 closings (the PR #199 "the ship is defensible on latency
grounds" pattern) MUST be flagged as non-compliant. Compliant closings
that say DONE / DECIDE / BLOCKED ON MEASUREMENT verbatim with attached
measurements MUST pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The detector lives outside the normal Python import path (it's an
# operator artifact, not a production module). Add it explicitly.
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / ".claude" / "lib"))

from rule10_check import (  # noqa: E402
    MEASUREMENT_ANCHORS,
    SMELL_WORDS,
    STATUS_PHRASES,
    check_closing,
    find_status_phrase,
    find_unsupported_smell_words,
)


# ---------------------------------------------------------------------------
# Status-phrase detection
# ---------------------------------------------------------------------------


class TestFindStatusPhrase:
    """The three rule-10 closing phrases must be detectable in realistic
    closing-summary positions: at the start of a paragraph, in a header,
    or inside a status-line sentence."""

    def test_done_at_start_of_summary(self):
        text = "DONE — measurement on current state shows +0.05 MRR (CI excludes 0)."
        result = find_status_phrase(text)
        assert result.found
        assert result.phrase == "DONE"

    def test_decide_in_status_line(self):
        text = """Wrapping up. DECIDE — measurement on current state shows X does
        not improve Y. Recommend reverting."""
        result = find_status_phrase(text)
        assert result.found
        assert result.phrase == "DECIDE"

    def test_blocked_on_measurement_multi_word_phrase(self):
        """BLOCKED ON MEASUREMENT is multi-word; detector must prefer it
        over a bare BLOCKED match."""
        text = "Status: BLOCKED ON MEASUREMENT — outcome unmeasured."
        result = find_status_phrase(text)
        assert result.found
        assert result.phrase == "BLOCKED ON MEASUREMENT"

    def test_blocked_on_measurement_inside_markdown_emphasis(self):
        text = "Verdict: **BLOCKED ON MEASUREMENT** — eval not run."
        result = find_status_phrase(text)
        assert result.found
        assert result.phrase == "BLOCKED ON MEASUREMENT"

    def test_no_status_phrase_returns_not_found(self):
        text = "The change is shipped and the tests pass. Things look good."
        result = find_status_phrase(text)
        assert not result.found
        assert result.phrase is None

    def test_done_inside_another_word_does_not_match(self):
        """Word-boundary check: 'DONE' must not match e.g. 'DONEE' or
        'OVERDONE'."""
        text = "The work is overdone but no formal close was declared."
        result = find_status_phrase(text)
        assert not result.found


# ---------------------------------------------------------------------------
# Smell-word detection
# ---------------------------------------------------------------------------


class TestFindUnsupportedSmellWords:
    """Smell-words flagged ONLY when no measurement anchor is in the
    same sentence. A smell-word paired with evidence is accepted; alone
    it's the rule-10 violation pattern."""

    def test_defensible_without_anchor_is_flagged(self):
        text = "The ship is defensible on latency grounds."
        findings = find_unsupported_smell_words(text)
        assert len(findings) == 1
        assert findings[0].word == "defensible"
        assert "defensible" in findings[0].sentence.lower()

    def test_defensible_with_ci_anchor_is_accepted(self):
        text = "The change is defensible given the +0.05 MRR CI [+0.012, +0.090] result."
        findings = find_unsupported_smell_words(text)
        assert findings == []

    def test_probably_without_anchor_is_flagged(self):
        text = "Listwise probably has lower p99 than pointwise."
        findings = find_unsupported_smell_words(text)
        assert len(findings) == 1
        assert findings[0].word == "probably"

    def test_probably_with_numeric_delta_is_accepted(self):
        text = "Listwise has +12ms lower p99 — probably attributable to the single-call pattern."
        findings = find_unsupported_smell_words(text)
        assert findings == []

    def test_should_be_without_anchor_is_flagged(self):
        text = "This should be a net improvement."
        findings = find_unsupported_smell_words(text)
        assert len(findings) == 1
        assert findings[0].word == "should be"

    def test_multiple_smell_words_in_same_sentence_each_flagged(self):
        text = "It's reasonable to assume probably this works."
        findings = find_unsupported_smell_words(text)
        # Both "reasonable" and "probably" flagged.
        words = {f.word for f in findings}
        assert "reasonable" in words
        assert "probably" in words

    def test_smell_words_across_sentences_with_one_anchored(self):
        """Anchor in sentence A doesn't help smell-word in sentence B."""
        text = (
            "Latency is probably improved per the +12ms p99 measurement. "
            "Quality is defensible without re-eval."
        )
        findings = find_unsupported_smell_words(text)
        # 'probably' is paired with the +12ms anchor; 'defensible' is not.
        assert len(findings) == 1
        assert findings[0].word == "defensible"

    def test_no_smell_words_returns_empty(self):
        text = "DONE — measurement on current state shows +0.05 MRR CI excludes zero."
        findings = find_unsupported_smell_words(text)
        assert findings == []

    def test_measurement_anchor_finding_doc_reference(self):
        text = "Should be net positive per the finding doc at docs/findings/..."
        findings = find_unsupported_smell_words(text)
        assert findings == []  # "finding doc" is an anchor

    def test_measurement_anchor_bootstrap_reference(self):
        text = "Probably accurate per paired-bootstrap CI."
        findings = find_unsupported_smell_words(text)
        assert findings == []


# ---------------------------------------------------------------------------
# Composite check
# ---------------------------------------------------------------------------


class TestCheckClosing:
    """check_closing combines status-phrase + smell-word findings into a
    pass/fail verdict appropriate for the Stop hook."""

    def test_compliant_done_with_measurement(self):
        text = (
            "DONE — measurement on current state shows +0.0495 MRR "
            "CI [+0.0112, +0.0900] (PR #X, finding doc Y)."
        )
        result = check_closing(text, goal_is_outcome_shaped=True)
        assert result.is_compliant
        assert result.status.found
        assert result.status.phrase == "DONE"
        assert result.unsupported_smell_words == []

    def test_compliant_blocked_on_measurement(self):
        text = (
            "BLOCKED ON MEASUREMENT — outcome unmeasured under current state; "
            "Phase C v2 cited but stale per rule 9. Re-run paired-bootstrap "
            "CI before closing."
        )
        result = check_closing(text, goal_is_outcome_shaped=True)
        assert result.is_compliant

    def test_non_compliant_pr199_pattern(self):
        """The exact failure mode rule 10 is designed to catch: smell-words
        carrying the ship justification without a measurement."""
        text = (
            "The default is flipped and the plumbing is green. "
            "The ship is defensible on latency-dominator + "
            "graceful-fallback grounds; the headline MRR lift is "
            "overstated relative to current main."
        )
        result = check_closing(text, goal_is_outcome_shaped=True)
        assert not result.is_compliant
        # Two failure modes simultaneously:
        # 1. No status phrase
        assert not result.status.found
        # 2. Defensible without anchor
        assert any(
            f.word == "defensible" for f in result.unsupported_smell_words
        )

    def test_non_compliant_status_present_but_smell_words(self):
        text = (
            "DONE — change merged. Latency is probably faster and the "
            "tradeoff is reasonable."
        )
        result = check_closing(text, goal_is_outcome_shaped=True)
        # Status phrase IS present but unsupported smell-words also present.
        assert not result.is_compliant
        assert result.status.found
        assert len(result.unsupported_smell_words) >= 1

    def test_contract_shaped_goal_is_always_compliant(self):
        """Contract-shaped goals don't require rule-10 phrasing. Smell
        words and missing status phrases are both irrelevant."""
        text = "The ship is defensible. Tests pass. Refactor complete."
        result = check_closing(text, goal_is_outcome_shaped=False)
        assert result.is_compliant
        assert any("contract-shaped" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class TestRender:
    """The hook's stderr output must be clear enough that a model reading
    it knows what to fix."""

    def test_compliant_render_mentions_ok(self):
        text = "DONE — +0.05 MRR CI [+0.01, +0.09]."
        result = check_closing(text, goal_is_outcome_shaped=True)
        rendered = result.render()
        assert "COMPLIANT" in rendered
        assert "DONE" in rendered

    def test_non_compliant_render_lists_problems(self):
        text = "The ship is defensible. It probably works."
        result = check_closing(text, goal_is_outcome_shaped=True)
        rendered = result.render()
        assert "NON-COMPLIANT" in rendered
        assert "MISSING" in rendered  # status phrase missing
        assert "defensible" in rendered  # smell-word listed
        # Long snippets get truncated, not silently dropped.
        long_text = "The ship is defensible because " + "x" * 500
        long_result = check_closing(long_text, goal_is_outcome_shaped=True)
        long_rendered = long_result.render()
        assert "..." in long_rendered or len(long_rendered) < len(long_text)


# ---------------------------------------------------------------------------
# Vocabulary stability
# ---------------------------------------------------------------------------


class TestVocabularyStability:
    """The vocabulary is part of the public contract — the skill markdown
    and docs/SHIP_DISCIPLINE.md both reference these specific strings.
    A test pins the set so a careless edit can't drift them silently."""

    def test_status_phrases_are_canonical(self):
        assert set(STATUS_PHRASES) == {
            "DONE", "DECIDE", "BLOCKED ON MEASUREMENT",
        }

    def test_smell_words_include_pr199_failure_pattern(self):
        # These are the words from the PR #199 closing that rule 10 was
        # designed to catch.
        assert "defensible" in SMELL_WORDS
        assert "probably" in SMELL_WORDS
        assert "reasonable" in SMELL_WORDS

    def test_measurement_anchors_include_ci_pattern(self):
        anchors_lower = {a.lower() for a in MEASUREMENT_ANCHORS}
        assert "ci [" in anchors_lower
        # Paired-bootstrap is the canonical eval shape.
        assert any("bootstrap" in a for a in anchors_lower)
