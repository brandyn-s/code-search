"""Rule-10 closing-phrase detector.

Used by `.claude/hooks/check_rule10_closing.sh` to gate goal-disciplined
sessions from closing without a compliant outcome statement. Defined
independently of any hook framework so it's testable in isolation and
reusable from other scripts (PR description linters, CI policy checks).

The detector answers two questions about a closing summary:

1. **Does the closing contain one of the rule-10 status phrases?**
   `DONE`, `DECIDE`, or `BLOCKED ON MEASUREMENT` — verbatim, in
   reasonable positioning (header, first/last paragraph, status line).

2. **Does the closing lean on smell-words without an attached
   measurement?**
   "defensible", "reasonable", "plausibly", "probably", "should be" —
   when these appear, the detector looks for a nearby measurement
   anchor (numeric value with units, CI brackets, finding-doc reference)
   and flags the sentence as suspect if none is present.

The detector is intentionally lightweight: it's a structural smoke check,
not an NLI model. False negatives (a non-compliant closing that happens
to contain the phrase verbatim) are acceptable; false positives (rejecting
a compliant closing because the phrasing is unusual) are not.

Source of truth for the rule is `docs/SHIP_DISCIPLINE.md`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Vocabulary — kept in sync with docs/SHIP_DISCIPLINE.md
# ---------------------------------------------------------------------------

STATUS_PHRASES: Sequence[str] = (
    "DONE",
    "DECIDE",
    "BLOCKED ON MEASUREMENT",
)

# Smell-words from rule 10's watch list. The detector flags these only
# when they appear WITHOUT a nearby measurement anchor.
SMELL_WORDS: Sequence[str] = (
    "defensible",
    "reasonable",
    "plausibly",
    "probably",
    "should be",
)

# Measurement anchors — substrings that, when present in the same sentence
# as a smell-word, indicate the speaker has attached real evidence. The
# list is conservative: it catches obvious measurement shapes, not every
# possible citation.
#
# Bare metric names (MRR, nDCG, HR@) are intentionally NOT here. A
# sentence saying "the MRR lift is overstated" mentions a metric but
# is not itself a measurement — it's a reference TO a measurement.
# The _NUMERIC_DELTA regex below catches the real-measurement pattern
# ("+0.05 MRR") via the optional unit group; bare metric names fall
# through, which is correct.
MEASUREMENT_ANCHORS: Sequence[str] = (
    # CI brackets like [+0.012, +0.089] — unambiguous measurement signal.
    "CI [",
    "ci [",
    # Citation patterns — point at an external artifact carrying numbers.
    "finding doc",
    "summary.json",
    # "bootstrap" alone is too loose ("bootstrap the system"). Require
    # the CI pairing.
    "bootstrap CI",
    "paired-bootstrap",
    # Latency anchors that include the value
    "p99=",
    "p50=",
    # Test-result anchors (for contract-shaped goals that happen to use
    # smell-words while citing test outcomes)
    "tests pass",
    "all tests",
    "unit tests",
)

# Pattern: a signed numeric delta with optional %, MRR, ms, etc. unit.
# Examples: "+0.0495", "-12%", "+3.2ms", "+0.1330 MRR"
#
# The leading sign + digit is load-bearing — it distinguishes a real
# measurement ("+0.05 MRR") from a stray "+" in arbitrary prose
# ("latency-dominator + graceful-fallback").
_NUMERIC_DELTA = re.compile(
    r"[+\-]\s?\d+(?:[.,]\d+)?\s?(?:%|ms|s|MRR|HR|nDCG)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StatusPhraseResult:
    """Outcome of looking for a status phrase in the closing summary."""

    found: bool
    phrase: Optional[str] = None
    location_hint: Optional[str] = None  # rough position (e.g., "header", "body")


@dataclass
class SmellWordFinding:
    """One unsupported smell-word occurrence."""

    word: str
    sentence: str
    has_anchor: bool


@dataclass
class CheckResult:
    """Composite result of running rule-10 detection on a closing summary."""

    is_compliant: bool
    status: StatusPhraseResult
    unsupported_smell_words: List[SmellWordFinding]
    notes: List[str]

    def render(self) -> str:
        """Render a one-screen summary suitable for a hook's stderr output."""
        lines = []
        if self.status.found:
            lines.append(
                f"[rule10] status phrase: {self.status.phrase!r} OK"
            )
        else:
            lines.append(
                f"[rule10] status phrase: MISSING (expected one of "
                f"{', '.join(STATUS_PHRASES)})"
            )
        if self.unsupported_smell_words:
            lines.append(
                f"[rule10] unsupported smell-words: "
                f"{len(self.unsupported_smell_words)}"
            )
            for f in self.unsupported_smell_words:
                snippet = f.sentence.strip()
                if len(snippet) > 100:
                    snippet = snippet[:97] + "..."
                lines.append(f"  - {f.word!r}: {snippet}")
        for note in self.notes:
            lines.append(f"[rule10] note: {note}")
        verdict = "COMPLIANT" if self.is_compliant else "NON-COMPLIANT"
        lines.append(f"[rule10] verdict: {verdict}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------


def find_status_phrase(text: str) -> StatusPhraseResult:
    """Look for one of DONE / DECIDE / BLOCKED ON MEASUREMENT in `text`.

    Prefers longest match first (so "BLOCKED ON MEASUREMENT" wins over a
    bare "BLOCKED") and returns the highest-priority hit.
    """
    # Order matters: try multi-word phrases before single-word ones.
    candidates = sorted(STATUS_PHRASES, key=len, reverse=True)
    for phrase in candidates:
        # Word-boundary match. Surrounded by punctuation, whitespace, or
        # markdown emphasis chars. Avoids matching "DONE" inside an
        # unrelated word.
        pattern = (
            r"(?:^|[\s\*\`\"'>\(])" + re.escape(phrase) + r"(?:[\s\*\`\"'>\),:.;!]|$)"
        )
        if re.search(pattern, text):
            # Rough location hint: first 200 chars vs middle vs last 200
            idx = text.find(phrase)
            if idx < 200:
                loc = "header/early"
            elif idx > max(0, len(text) - 200):
                loc = "footer/late"
            else:
                loc = "body"
            return StatusPhraseResult(found=True, phrase=phrase, location_hint=loc)
    return StatusPhraseResult(found=False)


def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter — good enough for the rule-10 use case.

    Splits on `.`, `!`, `?` followed by whitespace. Doesn't try to handle
    abbreviations or quoted text; the worst-case is a slightly oversized
    'sentence' which doesn't affect smell-word/anchor pairing meaningfully.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _sentence_has_measurement_anchor(sentence: str) -> bool:
    """True if `sentence` contains evidence of an actual measurement."""
    lowered = sentence.lower()
    for anchor in MEASUREMENT_ANCHORS:
        if anchor.lower() in lowered:
            return True
    if _NUMERIC_DELTA.search(sentence):
        return True
    return False


def find_unsupported_smell_words(text: str) -> List[SmellWordFinding]:
    """Return any smell-word occurrences NOT paired with a measurement anchor.

    Pre-rule-10 closings often use phrasings like 'the ship is defensible
    on latency grounds' — this finds exactly that pattern and reports it
    as suspect. A sentence using a smell-word alongside a measurement
    ('latency probably improved by +12% per p99=42ms benchmark') is
    accepted: the smell-word is mitigated by the attached evidence.
    """
    findings: List[SmellWordFinding] = []
    for sentence in _split_sentences(text):
        has_anchor = _sentence_has_measurement_anchor(sentence)
        for smell in SMELL_WORDS:
            # Smell-word match: word-boundary, case-insensitive
            pattern = r"\b" + re.escape(smell) + r"\b"
            if re.search(pattern, sentence, re.IGNORECASE):
                findings.append(
                    SmellWordFinding(
                        word=smell, sentence=sentence, has_anchor=has_anchor,
                    )
                )
    # Filter out the ones that ARE supported by an anchor.
    return [f for f in findings if not f.has_anchor]


# ---------------------------------------------------------------------------
# Composite check
# ---------------------------------------------------------------------------


def check_closing(
    text: str,
    *,
    goal_is_outcome_shaped: bool = True,
) -> CheckResult:
    """Run rule-10 detection on a closing summary.

    `goal_is_outcome_shaped=False` makes the check a no-op — contract-shaped
    goals don't require rule-10 phrasing. The flag is preserved in the
    returned `CheckResult.notes` so the caller's log shows why a check
    passed trivially.
    """
    if not goal_is_outcome_shaped:
        return CheckResult(
            is_compliant=True,
            status=StatusPhraseResult(found=False),
            unsupported_smell_words=[],
            notes=["goal classified as contract-shaped; rule-10 not required"],
        )

    status = find_status_phrase(text)
    smells = find_unsupported_smell_words(text)
    # Compliance requires: status phrase present AND no unsupported smell-words
    # in the closing summary.
    is_compliant = status.found and len(smells) == 0
    return CheckResult(
        is_compliant=is_compliant,
        status=status,
        unsupported_smell_words=smells,
        notes=[],
    )


# ---------------------------------------------------------------------------
# CLI entry point — invoked by the Stop hook
# ---------------------------------------------------------------------------


def _read_stdin_or_path(args: Iterable[str]) -> tuple[str, bool]:
    """Read closing text from a file path or stdin.

    Returns (text, goal_is_outcome_shaped). The shape flag is read from
    a marker file at ~/.claude/state/rule10_<session>.flag if present,
    defaulting to True (safer to over-check).
    """
    import os
    import sys

    args = list(args)
    if args and args[0] == "--contract-shaped":
        outcome = False
        args = args[1:]
    elif args and args[0] == "--outcome-shaped":
        outcome = True
        args = args[1:]
    else:
        # Check for the marker file. The skill writes this when it
        # classifies the goal at session start.
        marker = os.path.expanduser("~/.claude/state/rule10_active.flag")
        outcome = os.path.exists(marker)

    if args:
        with open(args[0], "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    return text, outcome


if __name__ == "__main__":
    import sys

    text, outcome_shaped = _read_stdin_or_path(sys.argv[1:])
    result = check_closing(text, goal_is_outcome_shaped=outcome_shaped)
    print(result.render(), file=sys.stderr)
    # Exit non-zero on non-compliance — the Stop hook treats this as
    # "block stopping, surface the failure to the model".
    sys.exit(0 if result.is_compliant else 1)
