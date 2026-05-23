# Claude Code Artifacts — `.claude/`

Operator-facing tooling that lives alongside the codebase. None of this
is on the Python import path; these are skill / hook / lib files that
Claude Code (the CLI / web / IDE clients) reads when working in this
repo.

## Contents

```
.claude/
├── README.md                         (this file)
├── skills/
│   └── goal-disciplined.md           Wrapped /goal that enforces rules 9 + 10
├── hooks/
│   └── check_rule10_closing.sh       Stop-hook gate for rule 10
└── lib/
    └── rule10_check.py               Detector (importable + CLI)
```

The rules being enforced are defined in `docs/SHIP_DISCIPLINE.md`. This
directory provides the **structural enforcement** — the skill + hook
make rule 10 mechanically harder to violate than doc-only policy
allows.

## `goal-disciplined` skill

A wrapper around the user's existing `/goal` skill that:

1. **Classifies the goal** upfront as outcome-shaped (measurable claim:
   quality, latency, reliability) or contract-shaped (test/structural
   property).
2. **Runs rule-9 staleness checks** when the goal references prior eval
   evidence (PR numbers, finding docs).
3. **Gates the closing phrase** for outcome-shaped goals: the closing
   summary must contain `DONE` / `DECIDE` / `BLOCKED ON MEASUREMENT`
   verbatim, and must not lean on smell-words (defensible, reasonable,
   plausibly, probably, should be) without an attached measurement.

### Installing globally

Claude Code reads project-local skills from `<repo>/.claude/skills/`
automatically, so this skill is available as `/goal-disciplined` when
you're in this repo. To make it available across all repos, copy it
into your global skills directory:

```bash
# Linux / macOS
cp .claude/skills/goal-disciplined.md ~/.claude/skills/

# Windows
copy .claude\skills\goal-disciplined.md %USERPROFILE%\.claude\skills\
```

### Usage

```
/goal-disciplined <goal text>
```

The skill will:
- Read `docs/SHIP_DISCIPLINE.md` (in this repo) or the equivalent in
  whichever repo it's invoked from.
- Ask which shape the goal is, if ambiguous.
- Execute the goal.
- Apply the rule-10 self-check before closing.

## `check_rule10_closing.sh` Stop hook

Defense-in-depth check that runs at every Stop event. If the goal was
classified as outcome-shaped (marker file at
`~/.claude/state/rule10_active.flag`) AND the final assistant turn
fails the rule-10 detector, the hook exits non-zero — which blocks the
stop and surfaces the failure to the model.

### Installing the hook

In Claude Code's settings (`~/.claude/settings.json`), add a Stop hook
entry:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/to/code-search/.claude/hooks/check_rule10_closing.sh"
          }
        ]
      }
    ]
  }
}
```

Path notes:
- Use the absolute path to the hook so it works regardless of which
  directory the Claude Code session was launched from.
- The hook fails open on missing infrastructure (transcript not
  provided, Python not on PATH, lib script missing). It will never
  block a stop because of its own setup problems — only because the
  closing summary itself violates rule 10.

### Marker file convention

The `goal-disciplined` skill writes
`~/.claude/state/rule10_active.flag` when it classifies a goal as
outcome-shaped, and removes it when the session closes. The hook
reads this file to decide whether to apply the rule-10 check:

- Marker present → outcome-shaped → run rule-10 check
- Marker absent → contract-shaped (or `/goal-disciplined` not used) → pass

If you forget to remove the marker between sessions (or the skill
errors before cleanup), runs on subsequent goals will continue to
apply rule 10. That's the conservative failure mode — checking too
much is annoying but safe; checking too little is the bug rule 10
exists to prevent.

## `rule10_check.py` detector

Importable Python module that classifies a closing summary as
compliant / non-compliant per rule 10. Used by the Stop hook but also
reusable from CI scripts, PR description linters, etc.

```python
from rule10_check import check_closing

result = check_closing(
    closing_text,
    goal_is_outcome_shaped=True,
)
if not result.is_compliant:
    print(result.render(), file=sys.stderr)
    sys.exit(1)
```

Test coverage at `tests/unit/test_rule10_closing_detector.py`. The
detector's contract is intentionally tight: false positives (rejecting
compliant closings) are not acceptable; false negatives (accepting
non-compliant ones that happen to contain the phrase verbatim) are
acceptable since the prompt-level enforcement in the skill is the
primary mechanism.

## Trade-offs and what this does NOT do

- **Not a full /goal replacement.** Use the ordinary `/goal` for
  contract-shaped work (mechanical refactors, doc edits) where outcome
  reduces to "tests pass". The wrapper adds friction; that friction is
  the point only when the friction prevents a real failure mode.
- **Prompt-level enforcement is primary, hook is secondary.** A
  well-written closing per the skill prompt won't trigger the hook.
  The hook catches cases where the model ignored or forgot the prompt
  instruction — which can happen, but shouldn't be the common path.
- **Classification is interpretive.** "Is this an outcome-shaped goal?"
  is a judgment call. The skill prompt explicitly asks the user when
  ambiguous; don't expect the wrapper to silently classify every goal
  correctly.
- **The detector is regex, not NLI.** A closing summary worded in an
  unusual way that's logically compliant but doesn't match the
  pattern can be flagged. Mitigation: the prompt teaches the model
  the exact phrasing the detector expects.

## Updating the policy

`docs/SHIP_DISCIPLINE.md` is the source of truth for the rule text.
This directory's artifacts implement the enforcement. If you change
the rules:

1. Update `docs/SHIP_DISCIPLINE.md` first.
2. Update `.claude/skills/goal-disciplined.md` to match the new text.
3. Update `.claude/lib/rule10_check.py` vocabulary if the status
   phrases or smell-words change.
4. Re-run `pytest tests/unit/test_rule10_closing_detector.py` — the
   vocabulary tests fail if the canonical strings drift.
