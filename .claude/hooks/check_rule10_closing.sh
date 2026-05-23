#!/usr/bin/env bash
# Stop-hook gate for ship-discipline rule 10.
#
# Wires into a Claude Code Stop hook in settings.json:
#
#   {
#     "hooks": {
#       "Stop": [
#         {
#           "matcher": "",
#           "hooks": [
#             {
#               "type": "command",
#               "command": "/path/to/.claude/hooks/check_rule10_closing.sh"
#             }
#           ]
#         }
#       ]
#     }
#   }
#
# Behavior:
#   - Reads the conversation transcript from $CLAUDE_TRANSCRIPT_PATH (set
#     by Claude Code when invoking the hook).
#   - Extracts the final assistant turn (the closing summary).
#   - Pipes it to .claude/lib/rule10_check.py which classifies whether
#     the closing satisfies rule 10.
#   - If the goal is outcome-shaped (marker file present) AND the closing
#     is non-compliant, exits 1 with stderr explaining what's missing.
#     Claude Code surfaces stderr to the model on a blocked stop, so the
#     model can self-correct.
#
# Exit codes:
#   0 — compliant, or contract-shaped (no rule-10 ceremony required)
#   1 — non-compliant; closing must be revised before stopping
#
# Failure modes:
#   - Missing transcript: exit 0 (don't block on hook infrastructure
#     problems; the prompt-level enforcement is the primary mechanism).
#   - Missing rule10_check.py: exit 0 (same reasoning).
#   - Python not on PATH: exit 0.

set -u

# Locate the lib directory relative to this script.
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="$HOOK_DIR/../lib/rule10_check.py"

# If the detector script isn't present, fail open. The hook is defense-in-
# depth; the prompt-level enforcement in the skill is the primary gate.
if [ ! -f "$LIB" ]; then
    exit 0
fi

# Locate Python — prefer the project's venv if it exists, otherwise use
# the first python3 on PATH.
PYTHON=""
if [ -x "$HOOK_DIR/../../.venv/Scripts/python.exe" ]; then
    PYTHON="$HOOK_DIR/../../.venv/Scripts/python.exe"
elif [ -x "$HOOK_DIR/../../.venv/bin/python" ]; then
    PYTHON="$HOOK_DIR/../../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    # No python: fail open.
    exit 0
fi

# Transcript path: Claude Code sets this when invoking the hook. If
# unset, the hook was invoked in a context that doesn't provide
# transcript access — fail open.
if [ -z "${CLAUDE_TRANSCRIPT_PATH:-}" ] || [ ! -f "${CLAUDE_TRANSCRIPT_PATH}" ]; then
    exit 0
fi

# Extract the final assistant turn from the transcript.
#
# Claude Code transcripts are JSONL with one message per line. The last
# message with role="assistant" is the closing summary. We grep + parse
# rather than depend on jq (which isn't guaranteed to be installed).
LAST_ASSISTANT_TEXT="$(
    "$PYTHON" - "$CLAUDE_TRANSCRIPT_PATH" <<'PYEOF'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

# Find the last entry with role=assistant.
for line in reversed(lines):
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    if entry.get("role") == "assistant" or entry.get("type") == "assistant":
        # Different transcript formats store the text in different places.
        text = entry.get("content") or entry.get("text") or ""
        if isinstance(text, list):
            # Anthropic-style content blocks
            text = "\n\n".join(
                block.get("text", "")
                for block in text
                if isinstance(block, dict) and block.get("type") == "text"
            )
        print(text)
        sys.exit(0)
sys.exit(0)
PYEOF
)"

# Empty extraction: fail open. We can't check what we can't see.
if [ -z "$LAST_ASSISTANT_TEXT" ]; then
    exit 0
fi

# Run the detector. It returns 1 if non-compliant for an outcome-shaped
# goal; 0 otherwise. The marker file at ~/.claude/state/rule10_active.flag
# (set by the goal-disciplined skill at start-of-session) is what the
# detector uses to decide outcome-vs-contract shape.
echo "$LAST_ASSISTANT_TEXT" | "$PYTHON" "$LIB"
