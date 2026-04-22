"""
Tests for Markdown body-only hashing in merkle/merkle_dag.py.

Regression target: api-docs markdown files carry YAML frontmatter
(``reviewed: 2026-04-22``, ``tags:``, ``status:``, etc.) that rotates
frequently without changing the actual documented API. Before this change,
any such metadata edit invalidated the cache and forced a full Voyage
re-embed of the file's chunks — wasting money and time.

After this change:
- Metadata-only edits to the YAML block produce an identical SHA256.
- Body edits produce a different SHA256 as expected.
- Files without frontmatter hash exactly as before (no behavior change).
- Non-markdown extensions are unaffected (no behavior change).
"""
import hashlib
from pathlib import Path

from merkle.merkle_dag import MerkleDAG, _body_content, _MD_EXTENSIONS


def _write(p: Path, content: str) -> Path:
    # Write raw bytes to avoid the Windows newline-translation gotcha that
    # Path.write_text triggers (LF -> CRLF), which would make the test
    # SHA256 assertions non-portable across platforms.
    p.write_bytes(content.encode("utf-8"))
    return p


def test_body_content_strips_frontmatter():
    raw = b"---\nreviewed: 2026-04-22\ntags: [a, b]\n---\n\n# Body\n\ntext"
    # The trailing newline on the ``---`` closing line is retained as part
    # of the body content separator.
    assert _body_content(raw) == b"\n\n# Body\n\ntext"


def test_body_content_no_frontmatter_passthrough():
    raw = b"# Just a heading\n\nwith no frontmatter\n"
    assert _body_content(raw) == raw


def test_body_content_unterminated_frontmatter_passthrough():
    # File starts with --- but never closes: treat as content, don't truncate.
    raw = b"---\nreviewed: 2026-04-22\n\n# Body without closing delimiter\n"
    assert _body_content(raw) == raw


def test_body_content_empty_body():
    raw = b"---\nreviewed: 2026-04-22\n---\n"
    # After ``\n---`` (4 chars past the closing-delimiter newline position)
    # we land on the trailing newline that separated frontmatter from body.
    # Body is just that newline. Not empty, but nearly so — and critically,
    # stable so two metadata-only edits hash the same.
    assert _body_content(raw) == b"\n"


def test_body_content_binary_safe():
    # Bytes that aren't valid utf-8 must not raise.
    raw = b"---\n\xff\xfe\n---\nbody\n"
    # Just assert it does not raise and returns some bytes.
    result = _body_content(raw)
    assert isinstance(result, bytes)


def test_md_extensions_scope():
    # We only want .md / .mdx / .markdown — nothing else should get trimmed.
    assert ".md" in _MD_EXTENSIONS
    assert ".mdx" in _MD_EXTENSIONS
    assert ".markdown" in _MD_EXTENSIONS
    assert ".py" not in _MD_EXTENSIONS
    assert ".txt" not in _MD_EXTENSIONS


def test_hash_file_metadata_only_edit_produces_same_hash(tmp_path: Path):
    dag = MerkleDAG(Path(tmp_path))
    f = tmp_path / "doc.md"

    _write(f,
        "---\n"
        "reviewed: 2026-04-22\n"
        "tags: [a, b]\n"
        "---\n\n"
        "# API Reference\n\n"
        "The canonical documented content.\n"
    )
    h1, _ = dag.hash_file(f)

    # Change ONLY frontmatter — body byte-identical.
    _write(f,
        "---\n"
        "reviewed: 2026-05-01\n"
        "tags: [a, b, c]\n"
        "status: approved\n"
        "---\n\n"
        "# API Reference\n\n"
        "The canonical documented content.\n"
    )
    h2, _ = dag.hash_file(f)

    assert h1 == h2, (
        f"expected metadata-only edit to produce identical hash; got "
        f"{h1[:12]} -> {h2[:12]}. This is the headline regression this "
        f"change exists to prevent."
    )


def test_hash_file_body_edit_changes_hash(tmp_path: Path):
    dag = MerkleDAG(Path(tmp_path))
    f = tmp_path / "doc.md"

    _write(f, "---\nreviewed: 2026-04-22\n---\n\n# Original title\n\nOriginal body.\n")
    h1, _ = dag.hash_file(f)

    _write(f, "---\nreviewed: 2026-04-22\n---\n\n# New title\n\nReworded body.\n")
    h2, _ = dag.hash_file(f)

    assert h1 != h2, "body edits MUST change the hash — otherwise incremental indexing breaks"


def test_hash_file_no_frontmatter_baseline(tmp_path: Path):
    """A markdown file with no frontmatter hashes exactly as if we hashed the raw bytes."""
    dag = MerkleDAG(Path(tmp_path))
    f = tmp_path / "nofrontmatter.md"

    content = "# Title\n\nNo frontmatter here.\n"
    _write(f, content)

    h_dag, _ = dag.hash_file(f)
    h_raw = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert h_dag == h_raw


def test_hash_file_python_unchanged(tmp_path: Path):
    """Non-markdown files must keep the original hash behavior (streamed raw bytes)."""
    dag = MerkleDAG(Path(tmp_path))
    f = tmp_path / "script.py"

    content = "---\nlooks: like-yaml\n---\nactually: a python file\n"
    _write(f, content)

    h_dag, size = dag.hash_file(f)
    h_raw = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert h_dag == h_raw
    assert size == len(content.encode("utf-8"))


def test_hash_file_mdx_and_markdown_extensions(tmp_path: Path):
    """Verify .mdx and .markdown extensions also get the stripping."""
    dag = MerkleDAG(Path(tmp_path))

    body = "# Body\n\ntext\n"
    for ext in (".mdx", ".markdown"):
        f = tmp_path / f"doc{ext}"
        _write(f, f"---\nkey: v1\n---\n\n{body}")
        h1, _ = dag.hash_file(f)
        _write(f, f"---\nkey: v2\n---\n\n{body}")
        h2, _ = dag.hash_file(f)
        assert h1 == h2, f"frontmatter-only edit should not change hash for {ext}"
