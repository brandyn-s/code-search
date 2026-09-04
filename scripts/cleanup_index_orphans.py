"""Compatibility CLI for the installable index-integrity implementation."""

if __name__ == "__main__":
    # Running a file puts scripts/ (not the repository root) on sys.path.
    # Re-exec the installable module through Python's module entry point so
    # the documented source-checkout command works without path mutation.
    import os
    import sys

    os.execv(
        sys.executable,
        [sys.executable, "-m", "search.integrity_audit", *sys.argv[1:]],
    )

from search.integrity_audit import *  # noqa: F401,F403,E402 - legacy imports
from search.integrity_audit import main  # noqa: E402,F401 - script entry point
