"""Portable JSONL manifest helpers for Drive2Gauss datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PATH_FIELDS = (
    "path",
    "clip_pt",
    "source_rgbd_latent_path",
    "generated_latent_source",
    "generated_rgb_root",
    "generated_flow_rgb_root",
)


def resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve a manifest path while preserving legacy absolute paths."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def resolve_row_paths(row: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Resolve known path fields relative to a manifest or explicit row base.

    New portable manifests may set ``path_base`` to a directory relative to the
    manifest.  When absent, relative paths are interpreted from the manifest's
    parent.  Absolute paths from historical manifests remain unchanged.
    """
    manifest_root = manifest_path.expanduser().resolve().parent
    path_base = row.get("path_base", ".")
    root = resolve_path(path_base, manifest_root)
    resolved = dict(row)
    for field in PATH_FIELDS:
        value = resolved.get(field)
        if value is not None:
            resolved[field] = str(resolve_path(value, root))
    return resolved


def read_jsonl(path: Path, *, add_manifest_index: bool = True) -> list[dict[str, Any]]:
    """Read a manifest and make every data path immediately usable."""
    manifest_path = path.expanduser().resolve()
    rows = []
    with manifest_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = resolve_row_paths(json.loads(line), manifest_path)
            if add_manifest_index:
                row["manifest_index"] = len(rows)
            rows.append(row)
    return rows


def relative_row(row: dict[str, Any], root: Path, fields: Iterable[str] = PATH_FIELDS) -> dict[str, Any]:
    """Return a copy with selected paths made relative to ``root``.

    Raises when a path lies outside the release root, preventing publication of
    a manifest that silently retains machine-specific paths.
    """
    release_root = root.expanduser().resolve()
    portable = dict(row)
    portable["path_base"] = "."
    for field in fields:
        value = portable.get(field)
        if value is None:
            continue
        try:
            portable[field] = Path(value).expanduser().resolve().relative_to(release_root).as_posix()
        except ValueError as error:
            raise ValueError(f"{field}={value!r} is outside release root {release_root}") from error
    return portable
