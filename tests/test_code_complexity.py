"""Code Complexity Metrics Tests for puco_eeff/.

This module enforces code quality standards via CI:
- Docstring coverage (warnings for missing docstrings)
- File line count limits (prevents code bloat) - extractor only
- Cyclomatic complexity caps (keeps functions maintainable) - extractor only
- Cognitive complexity caps - extractor only
- Maintainability Index - all directories

Baseline Management:
    Run `pytest --update-baselines` to update metrics baselines.
    Review the diff in tests/baselines/complexity_baselines.json and commit via PR.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from radon.complexity import cc_visit

if TYPE_CHECKING:
    from collections.abc import Iterator

# =============================================================================
# Constants
# =============================================================================

# Base directories
PUCO_EEFF_DIR = Path(__file__).parent.parent / "puco_eeff"
EXTRACTOR_DIR = PUCO_EEFF_DIR / "extractor"

# All subdirectories under puco_eeff (for tests that span all)
PUCO_EEFF_SUBDIRS = ["extractor", "scraper", "sheets", "transformer", "writer"]

# Baseline file
BASELINES_FILE = Path(__file__).parent / "baselines" / "complexity_baselines.json"


# =============================================================================
# Baseline Management
# =============================================================================


def load_baselines() -> dict:
    """Load baselines from JSON file with fallback to empty defaults."""
    baselines: dict = {
        "line_baselines": {},
        "cc_baselines": {},
        "cog_baselines": {},
        "mi_baselines": {},
        "mi_baselines_by_dir": {},
        "docstring_baselines": {},
        "config": {
            "cc_threshold": 15,
            "cog_threshold": 15,
            "mi_threshold": 13,
            "total_lines_baseline": None,
            "max_missing_docstrings": 0,
        },
    }

    if BASELINES_FILE.exists():
        with Path(BASELINES_FILE).open(encoding="utf-8") as f:
            baselines.update(json.load(f))

    return baselines


def save_baselines(baselines: dict) -> None:
    """Save baselines atomically (write to temp, then rename)."""
    BASELINES_FILE.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="baselines_", dir=BASELINES_FILE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(baselines, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(BASELINES_FILE)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


# =============================================================================
# AST Helpers - Function Extraction
# =============================================================================


def extract_functions_from_file(
    file_path: Path,
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]]:
    """Extract all function definitions from a Python file.

    Yields:
        Tuples of (function_node, source_code)

    """
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node, source


def extract_items_with_docstrings(file_path: Path) -> list[tuple[str, int, bool, str]]:
    """Extract functions, methods, and classes with their docstring status.

    Args:
        file_path: Path to the Python file

    Returns:
        List of (name, line_number, has_docstring, kind) tuples
        where kind is "function", "method", or "class"

    """
    results = []
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return results

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            has_docstring = ast.get_docstring(node) is not None
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            results.append((node.name, node.lineno, has_docstring, kind))
    return results


# =============================================================================
# Line Count Helpers
# =============================================================================


def get_line_count(file_path: Path) -> int:
    """Get line count consistent with `wc -l`.

    Uses total lines count (len(file.readlines())).
    """
    with Path(file_path).open(encoding="utf-8") as f:
        return len(f.readlines())


# =============================================================================
# Complexity Helpers
# =============================================================================


def get_all_function_complexities(file_path: Path) -> list[tuple[str, int, int]]:
    """Get cyclomatic complexity for all functions in a file.

    Returns:
        List of (function_name, line_number, complexity) tuples

    """
    source = file_path.read_text(encoding="utf-8")
    try:
        blocks = cc_visit(source)
    except SyntaxError:
        return []

    return [(block.name, block.lineno, block.complexity) for block in blocks]


def get_all_cognitive_complexities(file_path: Path) -> list[tuple[str, int, int]]:
    """Get cognitive complexity for all functions in a file using complexipy.

    Returns:
        List of (function_name, line_number, complexity) tuples

    """
    from complexipy import file_complexity

    try:
        result = file_complexity(str(file_path))
    except Exception:
        return []

    return [(func.name, func.line_start, func.complexity) for func in result.functions]


def get_maintainability_index(file_path: Path) -> float:
    """Get the Maintainability Index for a file using radon.

    Returns:
        MI score (0-100, higher is better)

    """
    from radon.metrics import mi_visit

    source = file_path.read_text(encoding="utf-8")
    try:
        return mi_visit(source, multi=False)
    except Exception:
        return 0.0


# =============================================================================
# Refactor Priority Integration
# =============================================================================


def _get_refactor_priority_suffix() -> str:
    """Get the refactoring priority message to append to assertion errors.

    Loads embeddings and complexity data to compute refactor indices.
    Uses ALL puco_eeff directories for comprehensive analysis.
    Returns empty string if no functions meet the threshold or similarity module unavailable.
    """
    try:
        # Import from similarity module if available
        from tests.test_code_similarity import get_refactor_priority_message_for_complexity

        return get_refactor_priority_message_for_complexity()
    except ImportError:
        return ""
    except Exception:
        # Don't fail the test due to refactor index computation errors
        return ""


# =============================================================================
# Tests
# =============================================================================


class TestDocstringCoverage:
    """Tests for docstring presence in all puco_eeff modules."""

    def test_docstring_coverage_all_directories(self) -> None:
        """Check that all functions, methods, and classes have docstrings.

        Tests all puco_eeff directories. Emits warnings for missing docstrings.
        Fails if any directory exceeds its baseline (initially 0 for all).
        """
        baselines = load_baselines()
        docstring_baselines = baselines.get("docstring_baselines", {})
        max_missing_default = baselines.get("config", {}).get("max_missing_docstrings", 0)

        all_missing: dict[str, list[str]] = {}
        violations: list[str] = []

        # Check all subdirectories
        for subdir in PUCO_EEFF_SUBDIRS:
            dir_path = PUCO_EEFF_DIR / subdir
            if not dir_path.exists():
                continue

            missing_in_dir: list[str] = []
            for py_file in dir_path.glob("*.py"):
                items = extract_items_with_docstrings(py_file)
                for name, line_no, has_docstring, kind in items:
                    if not has_docstring:
                        location = f"{py_file.name}:{line_no}"
                        missing_in_dir.append(f"  - {kind} {name} ({location})")

            if missing_in_dir:
                all_missing[subdir] = missing_in_dir
                # Check against per-directory baseline
                baseline = docstring_baselines.get(subdir, max_missing_default)
                if len(missing_in_dir) > baseline:
                    violations.append(
                        f"{subdir}/: {len(missing_in_dir)} missing docstrings, baseline allows {baseline}",
                    )

        # Also check root-level files in puco_eeff/
        root_missing: list[str] = []
        for py_file in PUCO_EEFF_DIR.glob("*.py"):
            items = extract_items_with_docstrings(py_file)
            for name, line_no, has_docstring, kind in items:
                if not has_docstring:
                    location = f"{py_file.name}:{line_no}"
                    root_missing.append(f"  - {kind} {name} ({location})")

        if root_missing:
            all_missing["root"] = root_missing
            baseline = docstring_baselines.get("root", max_missing_default)
            if len(root_missing) > baseline:
                violations.append(
                    f"root/: {len(root_missing)} missing docstrings, baseline allows {baseline}"
                )

        # Emit warnings for all missing docstrings
        if all_missing:
            total = sum(len(v) for v in all_missing.values())
            warning_lines = [f"Found {total} functions/classes without docstrings:"]
            for dir_name, items in sorted(all_missing.items()):
                warning_lines.append(f"\n{dir_name}/ ({len(items)}):")
                warning_lines.extend(items)
            warnings.warn("\n".join(warning_lines), UserWarning, stacklevel=2)

        if violations:
            error_msg = "Docstring coverage regression:\n" + "\n".join(violations)
            # Show details of missing docstrings
            for dir_name, items in sorted(all_missing.items()):
                error_msg += f"\n\n{dir_name}/:\n" + "\n".join(items)
            error_msg += _get_refactor_priority_suffix()
            pytest.fail(error_msg)


class TestLineCount:
    """Tests for file line count limits."""

    def test_file_line_counts(self, update_baselines: bool) -> None:
        """Check that files don't exceed their line count baselines.

        Prevents code bloat by failing if any file grows beyond baseline.
        """
        baselines = load_baselines()
        line_baselines = baselines.get("line_baselines", {})

        violations: list[str] = []
        current_counts: dict[str, int] = {}

        for py_file in EXTRACTOR_DIR.glob("*.py"):
            count = get_line_count(py_file)
            current_counts[py_file.name] = count

            baseline = line_baselines.get(py_file.name)
            if baseline is not None and count > baseline:
                violations.append(
                    f"File {py_file.name} has {count} lines, exceeds baseline of {baseline}"
                )

        if update_baselines:
            baselines["line_baselines"] = current_counts
            save_baselines(baselines)
            pytest.skip("Updated line count baselines")

        if violations:
            error_msg = "Line count violations:\n" + "\n".join(violations)
            error_msg += _get_refactor_priority_suffix()
            pytest.fail(error_msg)

    def test_total_line_count(self, update_baselines: bool) -> None:
        """Check that total lines across all extractor files don't exceed baseline.

        Prevents overall code bloat in the extractor module.
        """
        baselines = load_baselines()
        config = baselines.get("config", {})

        total_lines = sum(get_line_count(py_file) for py_file in EXTRACTOR_DIR.glob("*.py"))
        baseline = config.get("total_lines_baseline")

        if update_baselines:
            baselines.setdefault("config", {})["total_lines_baseline"] = total_lines
            save_baselines(baselines)
            pytest.skip(f"Updated total lines baseline to {total_lines}")

        if baseline is not None and total_lines > baseline:
            error_msg = (
                f"Total line count violation: {total_lines} lines across all extractor files, "
                f"exceeds baseline of {baseline}"
            )
            error_msg += _get_refactor_priority_suffix()
            pytest.fail(error_msg)


class TestCyclomaticComplexity:
    """Tests for cyclomatic complexity limits."""

    def test_cyclomatic_complexity(self, update_baselines: bool) -> None:
        """Check that no function exceeds the complexity threshold.

        Uses radon for CC analysis. Threshold is configurable in baselines.
        Existing complex functions can be grandfathered in cc_baselines.
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        threshold = config.get("cc_threshold", 15)
        cc_baselines = baselines.get("cc_baselines", {})

        violations: list[str] = []
        current_cc: dict[str, int] = {}

        for py_file in EXTRACTOR_DIR.glob("*.py"):
            complexities = get_all_function_complexities(py_file)
            for func_name, line_no, cc in complexities:
                key = f"{py_file.name}:{func_name}"
                current_cc[key] = cc

                # Check against function-specific baseline or global threshold
                func_baseline = cc_baselines.get(key, threshold)
                if cc > func_baseline:
                    violations.append(
                        f"{py_file.name}:{line_no} - {func_name}() has CC={cc}, "
                        f"exceeds {'baseline' if key in cc_baselines else 'threshold'} of {func_baseline}",
                    )

        if update_baselines:
            # Update cc_baselines for functions exceeding threshold
            for key, cc in current_cc.items():
                if cc > threshold:
                    baselines.setdefault("cc_baselines", {})[key] = cc
            save_baselines(baselines)
            pytest.skip("Updated CC baselines")

        if violations:
            error_msg = f"Cyclomatic complexity violations (threshold: {threshold}):\n" + "\n".join(
                violations
            )
            error_msg += _get_refactor_priority_suffix()
            pytest.fail(error_msg)


class TestCognitiveComplexity:
    """Tests for cognitive complexity limits using complexipy."""

    def test_cognitive_complexity(self, update_baselines: bool) -> None:
        """Check that no function exceeds the cognitive complexity threshold.

        Uses complexipy for cognitive complexity analysis. Threshold defaults to 15.
        Existing complex functions can be grandfathered in cog_baselines.
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        threshold = config.get("cog_threshold", 15)
        cog_baselines = baselines.get("cog_baselines", {})

        violations: list[str] = []
        current_cog: dict[str, int] = {}

        for py_file in EXTRACTOR_DIR.glob("*.py"):
            complexities = get_all_cognitive_complexities(py_file)
            for func_name, line_no, cog in complexities:
                key = f"{py_file.name}:{func_name}"
                current_cog[key] = cog

                # Check against function-specific baseline or global threshold
                func_baseline = cog_baselines.get(key, threshold)
                if cog > func_baseline:
                    violations.append(
                        f"{py_file.name}:{line_no} - {func_name}() has cognitive complexity={cog}, "
                        f"exceeds {'baseline' if key in cog_baselines else 'threshold'} of {func_baseline}",
                    )

        if update_baselines:
            # Update cog_baselines for functions exceeding threshold
            for key, cog in current_cog.items():
                if cog > threshold:
                    baselines.setdefault("cog_baselines", {})[key] = cog
            save_baselines(baselines)
            pytest.skip("Updated cognitive complexity baselines")

        if violations:
            error_msg = f"Cognitive complexity violations (threshold: {threshold}):\n" + "\n".join(
                violations
            )
            error_msg += _get_refactor_priority_suffix()
            pytest.fail(error_msg)


class TestMaintainabilityIndex:
    """Tests for maintainability index limits using radon."""

    def test_maintainability_index_all_directories(self, update_baselines: bool) -> None:
        """Check that no file in any puco_eeff directory has MI below threshold.

        Uses radon for MI analysis. Threshold defaults to 13 (very low).
        MI scale: 0-100, higher is better. 20+ is generally acceptable.
        Baselines are stored per-directory in mi_baselines_by_dir.
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        threshold = config.get("mi_threshold", 13)
        mi_baselines_by_dir = baselines.get("mi_baselines_by_dir", {})

        violations: list[str] = []
        current_mi_by_dir: dict[str, dict[str, float]] = {}

        # Check all subdirectories
        for subdir in PUCO_EEFF_SUBDIRS:
            dir_path = PUCO_EEFF_DIR / subdir
            if not dir_path.exists():
                continue

            dir_mi_baselines = mi_baselines_by_dir.get(subdir, {})
            current_mi: dict[str, float] = {}

            for py_file in dir_path.glob("*.py"):
                mi = get_maintainability_index(py_file)
                current_mi[py_file.name] = round(mi, 2)

                # Check against file-specific baseline or global threshold
                file_baseline = dir_mi_baselines.get(py_file.name, threshold)
                if mi < file_baseline:
                    violations.append(
                        f"{subdir}/{py_file.name} has MI={mi:.2f}, "
                        f"below {'baseline' if py_file.name in dir_mi_baselines else 'threshold'} of {file_baseline}",
                    )

            if current_mi:
                current_mi_by_dir[subdir] = current_mi

        # Also check root-level files
        root_mi_baselines = mi_baselines_by_dir.get("root", {})
        root_mi: dict[str, float] = {}

        for py_file in PUCO_EEFF_DIR.glob("*.py"):
            mi = get_maintainability_index(py_file)
            root_mi[py_file.name] = round(mi, 2)

            file_baseline = root_mi_baselines.get(py_file.name, threshold)
            if mi < file_baseline:
                violations.append(
                    f"{py_file.name} has MI={mi:.2f}, "
                    f"below {'baseline' if py_file.name in root_mi_baselines else 'threshold'} of {file_baseline}",
                )

        if root_mi:
            current_mi_by_dir["root"] = root_mi

        if update_baselines:
            # Update mi_baselines_by_dir for files below threshold
            for subdir, mi_dict in current_mi_by_dir.items():
                for filename, mi in mi_dict.items():
                    if mi < threshold:
                        baselines.setdefault("mi_baselines_by_dir", {}).setdefault(subdir, {})[
                            filename
                        ] = mi
            save_baselines(baselines)
            pytest.skip("Updated MI baselines")

        if violations:
            error_msg = f"Maintainability Index violations (threshold: {threshold}):\n" + "\n".join(
                violations
            )
            error_msg += _get_refactor_priority_suffix()
            pytest.fail(error_msg)

    def test_maintainability_index_extractor(self, update_baselines: bool) -> None:
        """Check that no file in extractor/ has MI below threshold.

        Legacy test for backward compatibility. Uses mi_baselines from baselines.
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        threshold = config.get("mi_threshold", 13)
        mi_baselines = baselines.get("mi_baselines", {})

        violations: list[str] = []
        current_mi: dict[str, float] = {}

        for py_file in EXTRACTOR_DIR.glob("*.py"):
            mi = get_maintainability_index(py_file)
            current_mi[py_file.name] = round(mi, 2)

            # Check against file-specific baseline or global threshold
            file_baseline = mi_baselines.get(py_file.name, threshold)
            if mi < file_baseline:
                violations.append(
                    f"{py_file.name} has MI={mi:.2f}, "
                    f"below {'baseline' if py_file.name in mi_baselines else 'threshold'} of {file_baseline}",
                )

        if update_baselines:
            # Update mi_baselines for files below threshold
            for filename, mi in current_mi.items():
                if mi < threshold:
                    baselines.setdefault("mi_baselines", {})[filename] = mi
            save_baselines(baselines)
            pytest.skip("Updated MI baselines")

        if violations:
            error_msg = f"Maintainability Index violations (threshold: {threshold}):\n" + "\n".join(
                violations
            )
            error_msg += _get_refactor_priority_suffix()
            pytest.fail(error_msg)
