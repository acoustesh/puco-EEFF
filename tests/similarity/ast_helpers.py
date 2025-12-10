"""AST helpers for extracting function information from Python files."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from tests.similarity.constants import EXTRACTOR_DIR, PUCO_EEFF_DIR, PUCO_EEFF_SUBDIRS
from tests.similarity.types import FunctionInfo


def get_function_text(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
) -> str:
    """Extract the full text of a function or class from source code."""
    lines = source.splitlines()
    start = node.lineno - 1
    end = node.end_lineno or start + 1
    return "\n".join(lines[start:end])


def normalize_ast_tokens(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Produce deterministic token sequence from function or class AST."""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def compute_content_hash(normalized_tokens: str) -> str:
    """Compute SHA256 hash of normalized token sequence."""
    return hashlib.sha256(normalized_tokens.encode("utf-8")).hexdigest()


def _extract_function_from_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
    file_name: str,
    min_loc: int,
) -> FunctionInfo | None:
    """Extract FunctionInfo from an AST node if it meets LOC threshold."""
    start = node.lineno
    end = node.end_lineno or start
    loc = end - start + 1

    if loc < min_loc:
        return None

    text = get_function_text(node, source)
    normalized = normalize_ast_tokens(node)
    content_hash = compute_content_hash(normalized)

    return FunctionInfo(
        name=node.name,
        file=file_name,
        start_line=start,
        end_line=end,
        loc=loc,
        hash=content_hash,
        text=text,
    )


def _extract_functions_from_source(
    source: str,
    file_path: Path,
    min_loc: int,
    file_prefix: str = "",
) -> list[FunctionInfo]:
    """Extract all functions/classes from source code meeting LOC threshold."""
    functions = []
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return functions

    file_name = f"{file_prefix}{file_path.name}" if file_prefix else file_path.name

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            func_info = _extract_function_from_node(node, source, file_name, min_loc)
            if func_info is not None:
                functions.append(func_info)

    return functions


def extract_function_infos(
    min_loc: int = 15,
    directory: Path | None = None,
) -> list[FunctionInfo]:
    """Extract all functions and classes from a directory meeting LOC threshold."""
    if directory is None:
        directory = EXTRACTOR_DIR

    functions = []
    for py_file in directory.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        functions.extend(_extract_functions_from_source(source, py_file, min_loc))

    return functions


def extract_all_function_infos(min_loc: int = 15) -> list[FunctionInfo]:
    """Extract all functions and classes from ALL puco_eeff directories."""
    all_functions = []

    for subdir in PUCO_EEFF_SUBDIRS:
        dir_path = PUCO_EEFF_DIR / subdir
        if not dir_path.exists():
            continue
        for py_file in dir_path.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            funcs = _extract_functions_from_source(source, py_file, min_loc, f"{subdir}/")
            all_functions.extend(funcs)

    # Root-level files
    for py_file in PUCO_EEFF_DIR.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        all_functions.extend(_extract_functions_from_source(source, py_file, min_loc))

    return all_functions


def extract_function_infos_from_file(file_path: Path, min_loc: int = 1) -> list[FunctionInfo]:
    """Extract all functions from a single file meeting LOC threshold."""
    source = file_path.read_text(encoding="utf-8")
    return _extract_functions_from_source(source, file_path, min_loc)
