#!/usr/bin/env python3
"""Build the deterministic offline BoW model used by frozen retrieval CI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


VOCABULARY = (
    "audit",
    "backoff",
    "bearer",
    "calculate",
    "claims",
    "configuration",
    "discount",
    "environment",
    "exponential",
    "feature",
    "flag",
    "immutable",
    "invoice",
    "load",
    "network",
    "persist",
    "record",
    "retry",
    "signature",
    "storage",
    "tax",
    "timeout",
    "token",
    "total",
    "validate",
)
REQUIRED_FILES = (
    Path("modules.json"),
    Path("config_sentence_transformers.json"),
    Path("0_BoW/config.json"),
)


def build(output: Path) -> None:
    if os.path.lexists(output):
        raise RuntimeError(f"model output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import BoW, Normalize

    model = SentenceTransformer(
        modules=[
            BoW(
                vocab=list(VOCABULARY),
                word_weights={},
                unknown_word_weight=1,
                cumulative_term_frequency=True,
            ),
            Normalize(),
        ]
    )
    model.save(str(output))
    missing = [
        str(relative)
        for relative in REQUIRED_FILES
        if not (output / relative).is_file()
    ]
    if missing:
        raise RuntimeError("model build omitted files: " + ", ".join(missing))

    loaded = SentenceTransformer(str(output), local_files_only=True)
    vectors = loaded.encode(
        [
            "validate bearer token signature claims",
            "exponential retry backoff network timeout",
        ],
        convert_to_numpy=True,
    )
    if vectors.shape != (2, len(VOCABULARY)):
        raise RuntimeError(f"unexpected frozen model shape: {vectors.shape!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic local frozen retrieval model"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        build(args.output.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Frozen model build FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Frozen retrieval model written to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
