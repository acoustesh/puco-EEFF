"""Complexity map loaders for refactor index integration."""

from __future__ import annotations

from pathlib import Path

from tests.similarity.constants import EXTRACTOR_DIR, PUCO_EEFF_DIR, PUCO_EEFF_SUBDIRS


def _get_all_function_complexities(file_path: Path) -> list[tuple[str, int, int]]:
    """Get cyclomatic complexity for all functions in a file."""
    from radon.complexity import cc_visit

    source = file_path.read_text(encoding="utf-8")
    try:
        blocks = cc_visit(source)
    except SyntaxError:
        return []

    return [(block.name, block.lineno, block.complexity) for block in blocks]


def _get_all_cognitive_complexities(file_path: Path) -> list[tuple[str, int, int]]:
    """Get cognitive complexity for all functions in a file using complexipy."""
    from complexipy import file_complexity

    try:
        result = file_complexity(str(file_path))
    except Exception:
        return []

    return [(func.name, func.line_start, func.complexity) for func in result.functions]


def _load_dir_complexity(
    directory: Path,
    prefix: str = "",
) -> tuple[dict[str, int], dict[str, int]]:
    """Load complexity maps for a single directory."""
    cc_map: dict[str, int] = {}
    cog_map: dict[str, int] = {}

    for py_file in directory.glob("*.py"):
        key_prefix = f"{prefix}{py_file.name}" if prefix else py_file.name
        for func_name, _, cc in _get_all_function_complexities(py_file):
            cc_map[f"{key_prefix}:{func_name}"] = cc
        for func_name, _, cog in _get_all_cognitive_complexities(py_file):
            cog_map[f"{key_prefix}:{func_name}"] = cog

    return cc_map, cog_map


def _load_complexity_maps(directory: Path | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """Load CC and COG complexity maps for functions in a directory."""
    if directory is None:
        directory = EXTRACTOR_DIR
    return _load_dir_complexity(directory)


def _load_all_complexity_maps() -> tuple[dict[str, int], dict[str, int]]:
    """Load CC and COG complexity maps for ALL puco_eeff functions."""
    cc_map: dict[str, int] = {}
    cog_map: dict[str, int] = {}

    for subdir in PUCO_EEFF_SUBDIRS:
        dir_path = PUCO_EEFF_DIR / subdir
        if dir_path.exists():
            sub_cc, sub_cog = _load_dir_complexity(dir_path, prefix=f"{subdir}/")
            cc_map.update(sub_cc)
            cog_map.update(sub_cog)

    # Root-level files
    root_cc, root_cog = _load_dir_complexity(PUCO_EEFF_DIR)
    cc_map.update(root_cc)
    cog_map.update(root_cog)

    return cc_map, cog_map
