"""Strict integrity binding for dispatcher-pinned skill trees."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from pathlib import Path
from typing import Mapping

PINNED_SKILL_DIGESTS_ENV = "HERMES_KANBAN_PINNED_SKILL_DIGESTS"
_DURABLE_REVIEW_SKILL = "immutable-change-reviews"
_SKILL_DIGEST_CONTRACT = b"turpi-skill-snapshot-v1\0"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_skill_tree_snapshot(skill_dir: Path) -> dict[str, bytes] | None:
    """Read one symlink/hardlink-free tree snapshot into immutable bytes."""
    try:
        root_stat = skill_dir.lstat()
    except OSError:
        return None
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        return None

    files: dict[str, bytes] = {}
    seen_inodes: set[tuple[int, int]] = set()
    try:
        paths = list(skill_dir.rglob("*"))
    except OSError:
        return None
    for path in paths:
        try:
            entry_stat = path.lstat()
        except OSError:
            return None
        if stat.S_ISLNK(entry_stat.st_mode):
            return None
        if stat.S_ISDIR(entry_stat.st_mode):
            continue
        if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
            return None
        inode = (int(entry_stat.st_dev), int(entry_stat.st_ino))
        if inode in seen_inodes:
            return None
        seen_inodes.add(inode)
        try:
            relative = path.relative_to(skill_dir)
            rel = relative.as_posix()
            if rel.startswith("../") or rel == ".." or "\x00" in rel:
                return None
            rel.encode("utf-8")
            with path.open("rb") as handle:
                opened_stat = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(opened_stat.st_mode)
                    or opened_stat.st_nlink != 1
                    or int(opened_stat.st_dev) != int(entry_stat.st_dev)
                    or int(opened_stat.st_ino) != int(entry_stat.st_ino)
                ):
                    return None
                data = handle.read()
                closed_stat = os.fstat(handle.fileno())
            if (
                int(closed_stat.st_dev),
                int(closed_stat.st_ino),
                int(closed_stat.st_size),
                int(closed_stat.st_mtime_ns),
                int(closed_stat.st_ctime_ns),
            ) != (
                int(opened_stat.st_dev),
                int(opened_stat.st_ino),
                int(opened_stat.st_size),
                int(opened_stat.st_mtime_ns),
                int(opened_stat.st_ctime_ns),
            ):
                return None
        except (OSError, UnicodeError, ValueError):
            return None
        files[rel] = data

    if "SKILL.md" not in files:
        return None
    return files


def _compute_snapshot_digest(files: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(_SKILL_DIGEST_CONTRACT)
    for rel_text in sorted(files):
        rel = rel_text.encode("utf-8")
        data = files[rel_text]
        digest.update(struct.pack(">Q", len(rel)))
        digest.update(rel)
        digest.update(struct.pack(">Q", len(data)))
        digest.update(data)
    return digest.hexdigest()


def compute_skill_tree_digest(skill_dir: Path) -> str | None:
    """Return the canonical digest for a symlink-free regular-file skill tree."""
    snapshot = _read_skill_tree_snapshot(skill_dir)
    return _compute_snapshot_digest(snapshot) if snapshot is not None else None


def pinned_skill_digests_from_env() -> dict[str, str]:
    """Parse the dispatcher-owned digest map, failing closed on malformed data."""
    raw = os.environ.get(PINNED_SKILL_DIGESTS_ENV)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {"*": "invalid"}
    if not isinstance(payload, Mapping):
        return {"*": "invalid"}
    result: dict[str, str] = {}
    for name, digest in payload.items():
        normalized_name = str(name or "").strip()
        normalized_digest = str(digest or "").strip()
        if not normalized_name or not _DIGEST_RE.fullmatch(normalized_digest):
            return {"*": "invalid"}
        result[normalized_name] = normalized_digest
    return result


def verify_pinned_skill_tree(
    name: str,
    skill_dir: Path | None,
    *,
    skills_root: Path,
) -> tuple[bool, str | None]:
    """Verify a pinned profile-local tree; ordinary unpinned skills pass."""
    pinned = pinned_skill_digests_from_env()
    if "*" in pinned:
        return False, "malformed dispatcher-pinned skill digest map"
    if (
        name == _DURABLE_REVIEW_SKILL
        and (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
        and (os.environ.get("HERMES_KANBAN_ROLE") or "").strip().lower()
        == "reviewer"
        and name not in pinned
    ):
        return False, "missing dispatcher-pinned reviewer skill digest"

    pinned_name = name if name in pinned else None
    resolved_skill_dir: Path | None = None
    if skill_dir is not None:
        try:
            resolved_skill_dir = skill_dir.resolve(strict=True)
        except OSError:
            resolved_skill_dir = None
    if pinned_name is None and resolved_skill_dir is not None:
        for candidate in pinned:
            try:
                expected_dir = (skills_root / candidate).resolve(strict=True)
            except OSError:
                continue
            if resolved_skill_dir == expected_dir:
                pinned_name = candidate
                break
    if pinned_name is None:
        return True, None

    try:
        expected_dir = (skills_root / pinned_name).resolve(strict=True)
    except OSError:
        expected_dir = None
    if resolved_skill_dir is None or resolved_skill_dir != expected_dir:
        return False, f"pinned skill digest mismatch for {pinned_name}"
    expected = pinned[pinned_name]
    actual = compute_skill_tree_digest(resolved_skill_dir)
    if actual != expected:
        return False, f"pinned skill digest mismatch for {pinned_name}"
    return True, None


def read_verified_pinned_skill_snapshot(
    name: str,
    skill_dir: Path | None,
    *,
    skills_root: Path,
) -> tuple[dict[str, bytes] | None, str | None]:
    """Return the exact verified bytes for a pinned tree, or ``None`` if unpinned."""
    pinned = pinned_skill_digests_from_env()
    if "*" in pinned:
        return None, "malformed dispatcher-pinned skill digest map"
    if (
        name == _DURABLE_REVIEW_SKILL
        and (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
        and (os.environ.get("HERMES_KANBAN_ROLE") or "").strip().lower()
        == "reviewer"
        and name not in pinned
    ):
        return None, "missing dispatcher-pinned reviewer skill digest"

    resolved_skill_dir: Path | None = None
    if skill_dir is not None:
        try:
            resolved_skill_dir = skill_dir.resolve(strict=True)
        except OSError:
            pass

    pinned_name = name if name in pinned else None
    if pinned_name is None and resolved_skill_dir is not None:
        for candidate in pinned:
            try:
                expected_dir = (skills_root / candidate).resolve(strict=True)
            except OSError:
                continue
            if resolved_skill_dir == expected_dir:
                pinned_name = candidate
                break
    if pinned_name is None:
        return None, None

    try:
        expected_dir = (skills_root / pinned_name).resolve(strict=True)
    except OSError:
        expected_dir = None
    if resolved_skill_dir is None or resolved_skill_dir != expected_dir:
        return None, f"pinned skill digest mismatch for {pinned_name}"

    snapshot = _read_skill_tree_snapshot(resolved_skill_dir)
    if snapshot is None or _compute_snapshot_digest(snapshot) != pinned[pinned_name]:
        return None, f"pinned skill digest mismatch for {pinned_name}"
    return snapshot, None
