"""AST helpers for extracting function information from Python files."""

from __future__ import annotations

import ast
import hashlib
import io
import tokenize
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


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_ast_text(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Get human-readable AST representation via ast.unparse().

    Returns the code reconstructed from AST, which excludes comments
    but preserves structure, docstrings, and type hints.
    """
    return ast.unparse(node)


def extract_text_for_embedding(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
) -> str:
    """Extract signature + docstring + comment lines for text embedding.

    This extracts:
    - Decorators and function/class signature
    - Docstring (if present)
    - All lines containing # comments (with their context)

    Args:
        node: AST node for the function/class
        source: Full source code of the file

    Returns
    -------
        Text suitable for embedding (signature + docstring + comments)
    """
    lines = source.splitlines()
    start_line = node.lineno  # 1-indexed
    end_line = node.end_lineno or start_line

    # Get decorator lines (before function definition)
    decorator_lines = []
    for decorator in getattr(node, "decorator_list", []):
        if decorator.lineno < start_line:
            for ln in range(decorator.lineno, start_line):
                decorator_lines.append(lines[ln - 1])

    # Get signature line(s) - from node.lineno to first body statement or end
    signature_end = start_line
    if node.body:
        first_body = node.body[0]
        signature_end = first_body.lineno - 1
    signature_lines = [lines[ln - 1] for ln in range(start_line, signature_end + 1)]

    # Get docstring
    docstring = ast.get_docstring(node) or ""
    docstring_text = f'"""{docstring}"""' if docstring else ""

    # Extract comment lines within function range
    comment_lines = []
    try:
        func_source = "\n".join(lines[start_line - 1 : end_line])
        tokens = tokenize.generate_tokens(io.StringIO(func_source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                # tok.start[0] is 1-indexed line within func_source
                actual_line_idx = start_line - 1 + tok.start[0] - 1
                if actual_line_idx < len(lines):
                    comment_lines.append(lines[actual_line_idx])
    except tokenize.TokenizeError:  # type: ignore[attr-defined]
        # Fall back to simple # detection
        for ln in range(start_line - 1, end_line):
            if ln < len(lines) and "#" in lines[ln]:
                comment_lines.append(lines[ln])

    # Combine all parts
    parts = []
    if decorator_lines:
        parts.extend(decorator_lines)
    parts.extend(signature_lines)
    if docstring_text:
        parts.append(docstring_text)
    if comment_lines:
        # Deduplicate (signature lines might contain comments)
        seen = set(parts)
        for cl in comment_lines:
            if cl not in seen:
                parts.append(cl)
                seen.add(cl)

    return "\n".join(parts)


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

    # Full raw source text
    text = get_function_text(node, source)

    # AST-based hash (ignores comments)
    normalized = normalize_ast_tokens(node)
    ast_hash = compute_content_hash(normalized)

    # AST text for AST embeddings
    ast_text = get_ast_text(node)

    # Text for text embeddings (signature + docstring + comments)
    text_for_embedding = extract_text_for_embedding(node, source)
    text_hash = compute_content_hash(text_for_embedding)

    return FunctionInfo(
        name=node.name,
        file=file_name,
        start_line=start,
        end_line=end,
        loc=loc,
        hash=ast_hash,
        text=text,
        text_for_embedding=text_for_embedding,
        text_hash=text_hash,
        ast_text=ast_text,
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
