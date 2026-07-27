#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Install code-search from the current source checkout.

Usage:
  ./scripts/install.sh
  ./scripts/install.sh --help

Environment:
  PYTHON            Python interpreter to use (default: python3, then python)
  CODE_SEARCH_VENV  Virtual environment path (default: <checkout>/.venv)

The installer does not clone, update, or delete repositories. It creates or
reuses a virtual environment and installs the current source checkout into it.
EOF
}

case "${1:-}" in
  "")
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    printf 'Unknown argument: %s\n\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac

if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  printf 'Python 3.12 or newer is required.\n' >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c \
  'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'
then
  printf 'Python 3.12 or newer is required (found %s).\n' \
    "$("$PYTHON_BIN" --version 2>&1)" >&2
  exit 1
fi

VENV_DIR="${CODE_SEARCH_VENV:-$REPO_ROOT/.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  printf 'Creating virtual environment at %s\n' "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

printf 'Installing current source checkout from %s\n' "$REPO_ROOT"
"$VENV_DIR/bin/python" -m pip install "$REPO_ROOT"

cat <<EOF

Installation complete.

Python:
  $VENV_DIR/bin/python

Example Claude Code registration:
  claude mcp add code-search --scope user -- \
    "$VENV_DIR/bin/python" -m mcp_server.server
EOF
