"""Stable repository and checkout identity shared by indexing engines.

The public envelope deliberately separates repository identity (portable
across clones), checkout identity (local path), source revision, and index
generation.  Callers persist an envelope only after a successful index and
only when captures from the start and end of that run agree on every source
identity field.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
from urllib.parse import urlsplit, urlunsplit


class IdentityCaptureError(RuntimeError):
    """The source tree could not be captured without ambiguity."""


@dataclass(frozen=True)
class IndexIdentity:
    schema_version: int
    repository_id: str
    checkout_id: str
    source_revision: str
    dirty_fingerprint: str
    index_generation: str
    captured_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SCP_REMOTE = re.compile(r"^(?:[^@/\s]+@)?([^:/\s]+):(.+)$")
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_COHERENCE_FIELDS = (
    "repository_id",
    "checkout_id",
    "source_revision",
    "dirty_fingerprint",
    "index_generation",
)


def validate_index_identity_dict(value: object) -> IndexIdentity:
    """Parse and validate an untrusted persisted identity envelope."""
    if not isinstance(value, Mapping):
        raise ValueError("index_identity must be an object")
    if (
        isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
    ):
        raise ValueError("schema_version must equal 1")

    string_fields = (
        "repository_id",
        "checkout_id",
        "source_revision",
        "dirty_fingerprint",
        "index_generation",
        "captured_at",
    )
    parsed: dict[str, str] = {}
    for field in string_fields:
        field_value = value.get(field)
        if not isinstance(field_value, str) or not field_value:
            raise ValueError(f"{field} must be a nonempty string")
        parsed[field] = field_value

    for field in ("repository_id", "checkout_id", "index_generation"):
        if _LOWER_SHA256.fullmatch(parsed[field]) is None:
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if (
        parsed["source_revision"] != "unborn"
        and _LOWER_GIT_OBJECT_ID.fullmatch(parsed["source_revision"]) is None
    ):
        raise ValueError(
            "source_revision must be unborn or a lowercase Git object ID"
        )
    if (
        parsed["dirty_fingerprint"] != "clean"
        and _LOWER_SHA256.fullmatch(parsed["dirty_fingerprint"]) is None
    ):
        raise ValueError(
            "dirty_fingerprint must be clean or a lowercase SHA-256 digest"
        )

    timestamp_text = parsed["captured_at"]
    try:
        timestamp = datetime.fromisoformat(
            timestamp_text[:-1] + "+00:00"
            if timestamp_text.endswith("Z")
            else timestamp_text
        )
    except ValueError as exc:
        raise ValueError("captured_at must be an RFC3339 timestamp") from exc
    if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError("captured_at must be UTC")

    expected_generation = derive_index_generation(
        repository_id=parsed["repository_id"],
        source_revision=parsed["source_revision"],
        dirty_fingerprint=parsed["dirty_fingerprint"],
    )
    if parsed["index_generation"] != expected_generation:
        raise ValueError(
            "index_generation does not match repository/source/dirty fields"
        )

    return IndexIdentity(schema_version=1, **parsed)


def identity_mismatch_fields(
    first: IndexIdentity,
    second: IndexIdentity,
) -> list[str]:
    """Return source fields that differ, ignoring capture time."""
    return [
        field
        for field in _COHERENCE_FIELDS
        if getattr(first, field) != getattr(second, field)
    ]


def describe_identity_mismatches(
    first: IndexIdentity,
    second: IndexIdentity,
) -> str:
    """Describe identity mismatches using only safe digests and sentinels."""
    return ", ".join(
        f"{field}: {getattr(first, field)} -> {getattr(second, field)}"
        for field in identity_mismatch_fields(first, second)
    )


def normalize_remote_url(remote: str) -> str:
    """Normalize a Git origin without retaining credentials."""
    value = remote.strip()
    scp_match = _SCP_REMOTE.fullmatch(value)
    if (
        scp_match
        and "://" not in value
        and not _WINDOWS_DRIVE_PATH.match(value)
    ):
        value = f"https://{scp_match.group(1)}/{scp_match.group(2)}"

    parsed = urlsplit(value)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        value = urlunsplit(
            (
                parsed.scheme.lower(),
                host,
                parsed.path,
                "",
                "",
            )
        )

    value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/")


def _git(
    root: Path,
    *args: str,
    input_data: bytes | None = None,
) -> bytes:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env=environment,
            input=input_data,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IdentityCaptureError(
            f"git {' '.join(args)} failed for the indexed checkout"
        ) from exc
    return completed.stdout


def _git_optional(root: Path, *args: str) -> bytes | None:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            env=environment,
        )
    except OSError as exc:
        raise IdentityCaptureError(
            f"git {' '.join(args)} could not run for the indexed checkout"
        ) from exc
    if completed.returncode != 0:
        return None
    return completed.stdout


def _captured_at(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat().replace("+00:00", "Z")


def _frame(label: str, payload: bytes) -> bytes:
    """Encode one unambiguous dirty-state frame.

    Cross-engine format: ASCII label, NUL, unsigned 64-bit big-endian payload
    length, then the raw payload bytes.
    """
    return (
        label.encode("ascii")
        + b"\0"
        + len(payload).to_bytes(8, byteorder="big", signed=False)
        + payload
    )


def _untracked_paths(status: bytes) -> list[bytes]:
    return sorted(
        record[3:]
        for record in status.split(b"\0")
        if record.startswith(b"?? ") and len(record) > 3
    )


def _hash_file(path: bytes) -> bytes:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise IdentityCaptureError(
            "an untracked file changed while index identity was captured"
        ) from exc
    return digest.digest()


def _untracked_payload(root: Path, relative_path: bytes) -> bytes:
    absolute_path = os.path.join(os.fsencode(root), relative_path)
    try:
        mode = os.lstat(absolute_path).st_mode
    except FileNotFoundError:
        kind = b"missing"
        content_digest = bytes(hashlib.sha256().digest_size)
    except OSError as exc:
        raise IdentityCaptureError(
            "an untracked path could not be inspected"
        ) from exc
    else:
        if stat.S_ISLNK(mode):
            kind = b"symlink"
            try:
                target = os.readlink(absolute_path)
            except OSError as exc:
                raise IdentityCaptureError(
                    "an untracked symlink changed during identity capture"
                ) from exc
            if isinstance(target, str):
                target = os.fsencode(target)
            content_digest = hashlib.sha256(target).digest()
        else:
            kind = b"file"
            content_digest = _hash_file(absolute_path)
    return relative_path + b"\0" + kind + b"\0" + content_digest


def _gitlinks(root: Path) -> list[tuple[bytes, bytes]]:
    entries = _git(
        root,
        "ls-files",
        "--stage",
        "--full-name",
        "-z",
    )
    gitlinks: list[tuple[bytes, bytes]] = []
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, separator, relative_path = entry.partition(b"\t")
        fields = metadata.split()
        if separator != b"\t" or len(fields) != 3:
            raise IdentityCaptureError("Git index entry is malformed")
        mode, object_id, stage = fields
        if mode != b"160000" or stage != b"0":
            continue
        if re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})", object_id) is None:
            raise IdentityCaptureError("Git submodule object ID is invalid")
        gitlinks.append((relative_path, object_id))
    return sorted(gitlinks)


def _submodule_payload(
    relative_path: bytes,
    expected_object_id: bytes,
    current_object_id: bytes,
    nested_fingerprint: str,
) -> bytes:
    return (
        relative_path
        + b"\0"
        + expected_object_id
        + b"\0"
        + current_object_id
        + b"\0"
        + nested_fingerprint.encode("ascii")
    )


def _submodule_payloads(root: Path) -> list[bytes]:
    top_level = _git(
        root,
        "rev-parse",
        "--show-toplevel",
    ).removesuffix(b"\n")
    if not top_level:
        raise IdentityCaptureError("Git repository root is empty")

    payloads: list[bytes] = []
    for relative_path, expected_object_id in _gitlinks(root):
        absolute_path = os.path.join(top_level, relative_path)
        if not os.path.lexists(os.path.join(absolute_path, b".git")):
            continue
        submodule_root = Path(os.fsdecode(absolute_path))
        current_object_id = _git(
            submodule_root,
            "rev-parse",
            "--verify",
            "HEAD",
        ).strip()
        if (
            re.fullmatch(
                rb"(?:[0-9a-f]{40}|[0-9a-f]{64})",
                current_object_id,
            )
            is None
        ):
            raise IdentityCaptureError(
                "Git submodule current object ID is invalid"
            )
        status = _git(
            submodule_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        nested_fingerprint = _dirty_fingerprint(
            submodule_root,
            status,
            current_object_id.decode("ascii"),
        )
        if (
            current_object_id == expected_object_id
            and nested_fingerprint == "clean"
        ):
            continue
        payloads.append(
            _submodule_payload(
                relative_path,
                expected_object_id,
                current_object_id,
                nested_fingerprint,
            )
        )
    return payloads


def _dirty_fingerprint(
    root: Path,
    status: bytes,
    diff_base: str,
) -> str:
    submodule_payloads = _submodule_payloads(root)
    if not status and not submodule_payloads:
        return "clean"

    worktree_diff = _git(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        diff_base,
        "--",
    )
    cached_diff = _git(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--cached",
        diff_base,
        "--",
    )
    digest = hashlib.sha256()
    digest.update(_frame("STATUS", status))
    digest.update(_frame("WORKTREE_DIFF", worktree_diff))
    digest.update(_frame("CACHED_DIFF", cached_diff))
    untracked_paths = _untracked_paths(status)
    untracked_root = root
    if untracked_paths:
        try:
            top_level = _git(
                root,
                "rev-parse",
                "--show-toplevel",
            ).removesuffix(b"\n").decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise IdentityCaptureError(
                "Git repository root is not valid UTF-8"
            ) from exc
        if not top_level:
            raise IdentityCaptureError("Git repository root is empty")
        untracked_root = Path(top_level).resolve()
    for relative_path in untracked_paths:
        digest.update(
            _frame(
                "UNTRACKED",
                _untracked_payload(untracked_root, relative_path),
            )
        )
    for payload in submodule_payloads:
        digest.update(_frame("SUBMODULE", payload))
    return digest.hexdigest()


def derive_index_generation(
    *,
    repository_id: str,
    source_revision: str,
    dirty_fingerprint: str,
) -> str:
    """Derive the shared generation identifier from three UTF-8 fields."""
    return hashlib.sha256(
        (
            repository_id
            + "\0"
            + source_revision
            + "\0"
            + dirty_fingerprint
        ).encode("utf-8")
    ).hexdigest()


def capture_index_identity(
    root: Path | str,
    *,
    captured_at: datetime | None = None,
) -> IndexIdentity:
    """Capture a Git checkout as a versioned index identity."""
    resolved_root = Path(root).resolve()
    _git(resolved_root, "rev-parse", "--git-dir")
    remote_bytes = _git_optional(
        resolved_root,
        "remote",
        "get-url",
        "origin",
    )
    if remote_bytes is None:
        repository_source = f"path:{resolved_root.as_posix()}"
    else:
        try:
            remote = remote_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise IdentityCaptureError(
                "origin URL is not valid UTF-8"
            ) from exc
        normalized_remote = normalize_remote_url(remote)
        repository_source = (
            f"remote:{normalized_remote}"
            if normalized_remote
            else f"path:{resolved_root.as_posix()}"
        )
    repository_id = hashlib.sha256(
        repository_source.encode("utf-8")
    ).hexdigest()
    checkout_id = hashlib.sha256(
        f"path:{resolved_root.as_posix()}".encode("utf-8")
    ).hexdigest()
    revision_bytes = _git_optional(
        resolved_root,
        "rev-parse",
        "--verify",
        "HEAD",
    )
    if revision_bytes is None:
        source_revision = "unborn"
        try:
            diff_base = _git(
                resolved_root,
                "hash-object",
                "-t",
                "tree",
                "--stdin",
                input_data=b"",
            ).decode("ascii", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise IdentityCaptureError(
                "Git empty tree object ID is not valid ASCII"
            ) from exc
        if _LOWER_GIT_OBJECT_ID.fullmatch(diff_base) is None:
            raise IdentityCaptureError(
                "Git empty tree object ID is invalid"
            )
    else:
        source_revision = revision_bytes.decode(
            "ascii", errors="strict"
        ).strip()
        diff_base = source_revision
    status = _git(
        resolved_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    dirty_fingerprint = _dirty_fingerprint(
        resolved_root,
        status,
        diff_base,
    )
    index_generation = derive_index_generation(
        repository_id=repository_id,
        source_revision=source_revision,
        dirty_fingerprint=dirty_fingerprint,
    )
    return IndexIdentity(
        schema_version=1,
        repository_id=repository_id,
        checkout_id=checkout_id,
        source_revision=source_revision,
        dirty_fingerprint=dirty_fingerprint,
        index_generation=index_generation,
        captured_at=_captured_at(captured_at),
    )
