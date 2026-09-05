"""Filesystem path guards shared by the API and the Streamlit dashboard.

Both surfaces accept caller-supplied dataset paths. Every such path must
resolve (symlinks included) to a regular file strictly inside the configured
data directory, otherwise it is rejected. Centralising the rule here keeps
the two surfaces from drifting apart.
"""

from __future__ import annotations

import os
from typing import Optional

MAX_PATH_CHARS = 1_024
_ALLOWED_SUFFIXES = (".csv", ".parquet")


class UnsafePathError(ValueError):
    """Caller-supplied path escapes the data directory or is not a data file."""


def data_dir() -> str:
    """Resolved (symlink-free) data root. Read at call time so tests can monkeypatch the env."""
    return os.path.realpath(os.getenv("ROSEGOLD_DATA_DIR", "data"))


def resolve_data_file(user_path: Optional[str], root: Optional[str] = None) -> str:
    """Return the real path of ``user_path`` or raise :class:`UnsafePathError`.

    Rules:
    * length-bounded, no NUL bytes;
    * resolves to a *regular file* (directories, devices, missing files rejected);
    * the resolved path is strictly inside ``root`` — ``root + "_evil"`` prefix
      collisions, ``..`` traversal and symlink escapes are all rejected;
    * extension must be ``.csv`` or ``.parquet`` (the only formats the loader reads).
    """
    if user_path is None or not str(user_path).strip():
        raise UnsafePathError("Empty path.")
    text = str(user_path)
    if len(text) > MAX_PATH_CHARS or "\x00" in text:
        raise UnsafePathError("Invalid path.")
    base = os.path.realpath(root) if root else data_dir()
    candidate = os.path.realpath(text)
    try:
        inside = os.path.commonpath([base, candidate]) == base
    except ValueError:
        inside = False
    if not inside or candidate == base:
        raise UnsafePathError("Path must be inside the configured data directory.")
    if not os.path.isfile(candidate):
        raise UnsafePathError("Path is not a regular file.")
    if not candidate.lower().endswith(_ALLOWED_SUFFIXES):
        raise UnsafePathError("Only .csv and .parquet files are accepted.")
    return candidate


def is_safe_data_file(user_path: Optional[str], root: Optional[str] = None) -> bool:
    try:
        resolve_data_file(user_path, root)
        return True
    except UnsafePathError:
        return False
