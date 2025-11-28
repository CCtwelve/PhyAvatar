from __future__ import annotations

import os
from collections.abc import Sequence


def normalize_process_subdirs(process_subdirs: Sequence[str] | str | None) -> list[str] | None:
    """
    Normalize user-provided subdirectory specifiers.

    Args:
        process_subdirs: Sequence of strings/ints or a comma-separated string.

    Returns:
        A list of zero-padded strings (e.g., \"00\") or None if the input is empty.
    """

    if process_subdirs is None:
        return None

    if isinstance(process_subdirs, str):
        raw_items = [item.strip() for item in process_subdirs.split(",")]
    else:
        raw_items = [str(item).strip() for item in process_subdirs]

    normalized: list[str] = []
    for item in raw_items:
        if not item:
            continue
        if item.isdigit():
            normalized.append(f"{int(item):02d}")
        else:
            normalized.append(item)

    return normalized or None


def list_selected_image_paths(
    root_dir: str,
    process_subdirs: list[str] | None,
    pattern: str,
) -> list[str]:
    """
    Collect file paths from the specified subdirectories only.
    """

    search_roots: list[str]
    if process_subdirs is None:
        search_roots = [root_dir]
    else:
        search_roots = [os.path.join(root_dir, subdir) for subdir in process_subdirs]

    paths: list[str] = []
    for sub_root in search_roots:
        if not os.path.isdir(sub_root):
            continue
        paths.extend(sorted(glob_recursive(sub_root, pattern)))

    return paths


def glob_recursive(root: str, pattern: str) -> list[str]:
    """
    Lightweight glob wrapper to avoid importing glob in every call site.
    """

    from glob import glob

    return glob(os.path.join(root, pattern), recursive=True)

