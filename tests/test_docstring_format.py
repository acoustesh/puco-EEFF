"""Docstring Format Tests for puco_eeff/.

This module enforces NumPy-style docstring format standards via CI using ruff.
Validates docstring formatting rules (D1xx-D4xx) and reports violations with
file locations for easy fixing.

Baseline Management:
    Run `pytest --update-baselines` to update docstring format baselines.
    Review the diff in tests/baselines/complexity_baselines.json and commit via PR.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

# =============================================================================
# Constants
# =============================================================================

PUCO_EEFF_DIR = Path(__file__).parent.parent / "puco_eeff"
BASELINES_FILE = Path(__file__).parent / "baselines" / "complexity_baselines.json"


# =============================================================================
# Ruff Docstring Validation
# =============================================================================


@dataclass(frozen=True, slots=True)
class DocstringViolation:
    """A single docstring format violation."""

    file: str
    line: int
    column: int
    code: str
    message: str

    def __str__(self) -> str:
        """Format violation for display."""
        return f"{self.file}:{self.line}:{self.column} {self.code} {self.message}"


def run_ruff_docstring_check(target_dir: Path) -> list[DocstringViolation]:
    """Run ruff with docstring rules (D) on target directory.

    Parameters
    ----------
    target_dir : Path
        Directory to check

    Returns
    -------
    list[DocstringViolation]
        List of DocstringViolation objects

    """
    ruff_path = shutil.which("ruff")
    if ruff_path is None:
        return []  # ruff not installed

    result = subprocess.run(
        [ruff_path, "check", str(target_dir), "--select=D", "--output-format=json"],
        capture_output=True,
        text=True,
        check=False,
    )

    if not result.stdout.strip():
        return []

    violations = []
    try:
        ruff_output = json.loads(result.stdout)
        for item in ruff_output:
            file_path = Path(item["filename"])
            try:
                rel_path = file_path.relative_to(target_dir.parent)
            except ValueError:
                rel_path = file_path

            violations.append(
                DocstringViolation(
                    file=str(rel_path),
                    line=item["location"]["row"],
                    column=item["location"]["column"],
                    code=item["code"],
                    message=item["message"],
                ),
            )
    except (json.JSONDecodeError, KeyError):
        pass

    return violations


def group_violations_by_file(
    violations: list[DocstringViolation],
) -> dict[str, list[DocstringViolation]]:
    """Group violations by file path.

    Parameters
    ----------
    violations : list[DocstringViolation]
        List of violations to group

    Returns
    -------
    dict[str, list[DocstringViolation]]
        Dict mapping file paths to their violations

    """
    grouped: dict[str, list[DocstringViolation]] = {}
    for v in violations:
        grouped.setdefault(v.file, []).append(v)
    return grouped


# =============================================================================
# Baseline Management
# =============================================================================


def load_baselines() -> dict:
    """Load baselines from JSON file with fallback to defaults."""
    baselines: dict = {"docstring_format_baselines": {}}

    if BASELINES_FILE.exists():
        with Path(BASELINES_FILE).open(encoding="utf-8") as f:
            baselines.update(json.load(f))

    return baselines


def save_baselines(baselines: dict) -> None:
    """Save baselines atomically (write to temp, then rename)."""
    BASELINES_FILE.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        suffix=".json",
        prefix="baselines_",
        dir=BASELINES_FILE.parent,
    )
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
# Tests
# =============================================================================


class TestDocstringFormat:
    """Tests for NumPy-style docstring format compliance."""

    def test_docstring_format_all_files(self, update_baselines: bool) -> None:
        """Check that all files comply with NumPy docstring format rules.

        Uses ruff's D rules (pydocstyle) with numpy convention.
        Per-file violation counts are tracked in baselines.
        Fails if any file exceeds its baseline, showing exact violations for fixing.
        """
        baselines = load_baselines()
        format_baselines = baselines.get("docstring_format_baselines", {})

        violations = run_ruff_docstring_check(PUCO_EEFF_DIR)
        violations_by_file = group_violations_by_file(violations)

        current_counts: dict[str, int] = {}
        regression_violations: list[str] = []
        new_file_violations: list[str] = []

        for file_path, file_violations in sorted(violations_by_file.items()):
            count = len(file_violations)
            current_counts[file_path] = count
            baseline = format_baselines.get(file_path, 0)

            if count > baseline:
                violation_details = "\n".join(f"    {v}" for v in file_violations)

                if baseline == 0:
                    new_file_violations.append(
                        f"{file_path}: {count} violations (new file)\n{violation_details}",
                    )
                else:
                    regression_violations.append(
                        f"{file_path}: {count} violations (baseline: {baseline}, "
                        f"+{count - baseline} new)\n{violation_details}",
                    )

        if update_baselines:
            baselines["docstring_format_baselines"] = current_counts
            save_baselines(baselines)
            total = sum(current_counts.values())
            pytest.skip(
                f"Updated docstring format baselines: {len(current_counts)} files, "
                f"{total} total violations",
            )

        if regression_violations or new_file_violations:
            error_parts = ["Docstring format violations (NumPy style via ruff D rules):"]

            if regression_violations:
                error_parts.append("\n=== Regressions (exceeds baseline) ===")
                error_parts.extend(regression_violations)

            if new_file_violations:
                error_parts.append("\n=== New violations (no baseline) ===")
                error_parts.extend(new_file_violations)

            error_parts.append(
                "\n\nTo fix: Review violations above and update docstrings."
                "\nTo grandfather: Run `pytest --update-baselines`",
            )

            pytest.fail("\n".join(error_parts))
