"""Comment Density Tests for puco_eeff/.

This module enforces comment density standards via CI:
- Minimum comment density (ensures code is documented)
- Maximum comment density (prevents over-commenting / commented-out code)

Comment density is measured as: comments / (sloc + comments) * 100

Baseline Management:
    Run `pytest --update-baselines` to update comment density baselines.
    Review the diff in tests/baselines/complexity_baselines.json and commit via PR.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from radon.raw import analyze  # type: ignore[import-untyped]

# =============================================================================
# Constants
# =============================================================================

PUCO_EEFF_DIR = Path(__file__).parent.parent / "puco_eeff"
PUCO_EEFF_SUBDIRS = ["extractor", "scraper", "sheets", "transformer", "writer"]
BASELINES_FILE = Path(__file__).parent / "baselines" / "complexity_baselines.json"

# Default thresholds
DEFAULT_MIN_COMMENT_DENSITY = 5.0  # Minimum 5% comments
DEFAULT_MAX_COMMENT_DENSITY = 40.0  # Maximum 40% comments


# =============================================================================
# Comment Density Analysis
# =============================================================================


@dataclass(frozen=True, slots=True)
class CommentStats:
    """Statistics about comments in source code."""

    sloc: int
    comments: int
    density_pct: float


def comment_density_from_source(source: str) -> CommentStats:
    """Calculate comment density from source code.

    Args:
        source: Python source code as a string

    Returns
    -------
        CommentStats with sloc, comments count, and density percentage

    """
    m = analyze(source)
    denom = m.sloc + m.comments
    density = 0.0 if denom == 0 else 100.0 * (m.comments / denom)
    return CommentStats(sloc=m.sloc, comments=m.comments, density_pct=density)


def get_file_comment_stats(file_path: Path) -> CommentStats:
    """Get comment statistics for a Python file.

    Args:
        file_path: Path to the Python file

    Returns
    -------
        CommentStats for the file

    """
    source = file_path.read_text(encoding="utf-8")
    return comment_density_from_source(source)


# =============================================================================
# Baseline Management
# =============================================================================


def load_baselines() -> dict:
    """Load baselines from JSON file with fallback to defaults."""
    baselines: dict = {
        "comment_density_baselines": {},
        "comment_density_config": {
            "min_density": DEFAULT_MIN_COMMENT_DENSITY,
            "max_density": DEFAULT_MAX_COMMENT_DENSITY,
        },
    }

    if BASELINES_FILE.exists():
        with Path(BASELINES_FILE).open(encoding="utf-8") as f:
            stored = json.load(f)
            baselines.update(stored)

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
# Tests
# =============================================================================


class TestCommentDensity:
    """Tests for comment density in puco_eeff modules."""

    def test_comment_density_all_directories(self, update_baselines: bool) -> None:
        """Check that all files have comment density within acceptable range.

        Tests all puco_eeff directories. Fails if any file has:
        - Too few comments (below min_density threshold)
        - Too many comments (above max_density threshold or file-specific baseline)

        File-specific baselines can override global thresholds.
        """
        baselines = load_baselines()
        config = baselines.get("comment_density_config", {})
        min_density = config.get("min_density", DEFAULT_MIN_COMMENT_DENSITY)
        max_density = config.get("max_density", DEFAULT_MAX_COMMENT_DENSITY)
        density_baselines = baselines.get("comment_density_baselines", {})

        violations: list[str] = []
        current_densities: dict[str, dict[str, float]] = {}

        # Check all subdirectories
        for subdir in PUCO_EEFF_SUBDIRS:
            dir_path = PUCO_EEFF_DIR / subdir
            if not dir_path.exists():
                continue

            dir_baselines = density_baselines.get(subdir, {})
            current_dir: dict[str, float] = {}

            for py_file in sorted(dir_path.glob("*.py")):
                stats = get_file_comment_stats(py_file)
                density_rounded = round(stats.density_pct, 2)
                current_dir[py_file.name] = density_rounded

                # Get file-specific bounds or use global thresholds
                file_baseline = dir_baselines.get(py_file.name, {})
                file_min = file_baseline.get("min", min_density)
                file_max = file_baseline.get("max", max_density)

                if density_rounded < file_min:
                    violations.append(
                        f"{subdir}/{py_file.name}: density={density_rounded}% "
                        f"(sloc={stats.sloc}, comments={stats.comments}), "
                        f"below minimum of {file_min}%",
                    )
                elif density_rounded > file_max:
                    violations.append(
                        f"{subdir}/{py_file.name}: density={density_rounded}% "
                        f"(sloc={stats.sloc}, comments={stats.comments}), "
                        f"above maximum of {file_max}%",
                    )

            if current_dir:
                current_densities[subdir] = current_dir

        # Also check root-level files
        root_baselines = density_baselines.get("root", {})
        root_densities: dict[str, float] = {}

        for py_file in sorted(PUCO_EEFF_DIR.glob("*.py")):
            stats = get_file_comment_stats(py_file)
            density_rounded = round(stats.density_pct, 2)
            root_densities[py_file.name] = density_rounded

            file_baseline = root_baselines.get(py_file.name, {})
            file_min = file_baseline.get("min", min_density)
            file_max = file_baseline.get("max", max_density)

            if density_rounded < file_min:
                violations.append(
                    f"{py_file.name}: density={density_rounded}% "
                    f"(sloc={stats.sloc}, comments={stats.comments}), "
                    f"below minimum of {file_min}%",
                )
            elif density_rounded > file_max:
                violations.append(
                    f"{py_file.name}: density={density_rounded}% "
                    f"(sloc={stats.sloc}, comments={stats.comments}), "
                    f"above maximum of {file_max}%",
                )

        if root_densities:
            current_densities["root"] = root_densities

        if update_baselines:
            # Update baselines for files outside thresholds
            for subdir, densities in current_densities.items():
                for filename, density in densities.items():
                    if density < min_density or density > max_density:
                        baselines.setdefault("comment_density_baselines", {}).setdefault(
                            subdir,
                            {},
                        )[filename] = {
                            "min": min(density, min_density),
                            "max": max(density, max_density),
                            "current": density,
                        }
            save_baselines(baselines)
            pytest.skip("Updated comment density baselines")

        if violations:
            error_msg = (
                f"Comment density violations (min: {min_density}%, max: {max_density}%):\n"
                + "\n".join(violations)
            )
            pytest.fail(error_msg)

    def test_aggregate_comment_density(self, update_baselines: bool) -> None:
        """Check aggregate comment density across all puco_eeff files.

        Ensures the overall codebase maintains a healthy comment ratio.
        """
        baselines = load_baselines()
        config = baselines.get("comment_density_config", {})
        min_density = config.get("aggregate_min_density", DEFAULT_MIN_COMMENT_DENSITY)
        max_density = config.get("aggregate_max_density", DEFAULT_MAX_COMMENT_DENSITY)

        total_sloc = 0
        total_comments = 0

        # Aggregate across all directories
        for subdir in PUCO_EEFF_SUBDIRS:
            dir_path = PUCO_EEFF_DIR / subdir
            if not dir_path.exists():
                continue

            for py_file in dir_path.glob("*.py"):
                stats = get_file_comment_stats(py_file)
                total_sloc += stats.sloc
                total_comments += stats.comments

        # Also include root-level files
        for py_file in PUCO_EEFF_DIR.glob("*.py"):
            stats = get_file_comment_stats(py_file)
            total_sloc += stats.sloc
            total_comments += stats.comments

        denom = total_sloc + total_comments
        aggregate_density = 0.0 if denom == 0 else 100.0 * (total_comments / denom)

        if update_baselines:
            baselines.setdefault("comment_density_config", {})["aggregate_current"] = round(
                aggregate_density,
                2,
            )
            save_baselines(baselines)
            pytest.skip(f"Updated aggregate comment density: {aggregate_density:.2f}%")

        violations: list[str] = []

        if aggregate_density < min_density:
            violations.append(
                f"Aggregate comment density {aggregate_density:.1f}% "
                f"(sloc={total_sloc}, comments={total_comments}) "
                f"is below minimum of {min_density}%",
            )

        if aggregate_density > max_density:
            violations.append(
                f"Aggregate comment density {aggregate_density:.1f}% "
                f"(sloc={total_sloc}, comments={total_comments}) "
                f"is above maximum of {max_density}%",
            )

        if violations:
            pytest.fail("\n".join(violations))
