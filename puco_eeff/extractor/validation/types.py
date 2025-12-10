"""Validation dataclasses and type definitions.

This module contains pure data structures with no business logic dependencies,
ensuring they can be imported without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from puco_eeff.extractor.extraction import SectionBreakdown

__all__ = [
    "ComparisonResult",
    "CrossValidationResult",
    "ExtractionResult",
    "PDFXBRLComparisonResult",
    "ReferenceValidationResult",
    "SumValidationResult",
    "ValidationReport",
    "ValidationResult",
]


@dataclass
class ComparisonResult:
    """Unified validation result for comparing two values from different sources.

    This class handles both PDF/XBRL cross-validation and reference baseline
    validation through a single interface. The `comparison_type` field indicates
    which kind of validation was performed.

    Comparison Types:
        - "pdf_xbrl": Cross-validation between PDF extraction and XBRL data
        - "reference": Validation against ground-truth reference baseline

    Attributes
    ----------
        field_name: Name of the validated field (e.g., "total_costo_venta")
        value_a: First value (PDF value or expected reference)
        value_b: Second value (XBRL value or actual extracted)
        match: True if values agree within tolerance
        comparison_type: Type of comparison ("pdf_xbrl" or "reference")
        source: For pdf_xbrl: origin indicator ("pdf_only", "xbrl_only", "both")
                For reference: None (not applicable)
        difference: Absolute difference when mismatch (None if match/missing)
    """

    field_name: str
    value_a: int | None  # PDF value or expected reference
    value_b: int | None  # XBRL value or actual extracted
    match: bool
    comparison_type: str = "pdf_xbrl"  # "pdf_xbrl" or "reference"
    source: str | None = None  # Only used for pdf_xbrl comparison
    difference: int | None = None

    @property
    def status(self) -> str:
        """Format validation status with context-appropriate messaging."""
        # Handle pdf_xbrl source-specific messages
        if self.comparison_type == "pdf_xbrl":
            source_warnings = {
                "pdf_only": "⚠ PDF only (no XBRL)",
                "xbrl_only": "⚠ XBRL only (PDF extraction failed)",
            }
            if warning := source_warnings.get(self.source or ""):
                return warning

        # Handle reference baseline availability
        if self.comparison_type == "reference":
            if self.value_a is None:
                return "⚠ Reference baseline unavailable"
            if self.value_b is None:
                return "⚠ Extraction returned empty"

        # Common match/mismatch formatting
        if self.match:
            return "✓ Match"
        diff_display = (
            f" (diff: {self.difference:,})"
            if self.difference
            else (" (diff: n/a)" if self.comparison_type == "pdf_xbrl" else "")
        )
        return f"✗ Mismatch{diff_display}"

    @property
    def field(self) -> str:
        """Field name alias for backward compatibility."""
        return self.field_name


# Backward-compatible aliases
PDFXBRLComparisonResult = ComparisonResult
ValidationResult = ComparisonResult
ReferenceValidationResult = ComparisonResult


@dataclass
class ExtractionResult:
    """Complete extraction result with optional validation."""

    year: int
    quarter: int
    sections: dict[str, SectionBreakdown] = field(default_factory=dict)
    xbrl_available: bool = False
    xbrl_totals: dict[str, int | None] = field(default_factory=dict)
    validations: list[ValidationResult] = field(default_factory=list)
    validation_report: ValidationReport | None = None
    source: str = "cmf"
    pdf_path: Path | None = None
    xbrl_path: Path | None = None

    def get_section(self, section_id: str) -> SectionBreakdown | None:
        """Get a section by its ID."""
        return self.sections.get(section_id)

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        if not self.validations:
            return len(self.sections) > 0 and all(s is not None for s in self.sections.values())
        return all(v.match for v in self.validations)


@dataclass
class SumValidationResult:
    """Result of line-item sum validation."""

    description: str
    total_field: str
    expected_total: int | None
    calculated_sum: int
    match: bool
    difference: int
    tolerance: int


@dataclass
class CrossValidationResult:
    """Result of cross-validation formula check."""

    description: str
    formula: str
    expected_value: int | None
    calculated_value: int | None
    match: bool
    difference: int | None
    tolerance: int
    missing_facts: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated validation results for unified reporting."""

    sum_validations: list[SumValidationResult] = field(default_factory=list)
    cross_validations: list[CrossValidationResult] = field(default_factory=list)
    pdf_xbrl_validations: list[ValidationResult] = field(default_factory=list)
    reference_issues: list[str] | None = None

    def has_failures(self, category: str | None = None) -> bool:
        """Check for validation failures, optionally filtered by category.

        Parameters
        ----------
        category
            Optional filter: ``None`` (all), ``"sum"``, ``"cross"``,
            ``"pdf_xbrl"``, or ``"reference"``.

        Returns
        -------
        bool
            ``True`` when any validation in the selected category failed.
        """
        if category == "reference":
            return bool(self.reference_issues)

        checks_by_category = {
            "sum": self.sum_validations,
            "cross": self.cross_validations,
            "pdf_xbrl": self.pdf_xbrl_validations,
        }

        if category is not None:
            validations = checks_by_category.get(category, [])
            return any(not v.match for v in validations)

        # Check all categories when no filter specified
        return (
            any(not v.match for v in self.sum_validations)
            or any(not v.match for v in self.cross_validations)
            or any(not v.match for v in self.pdf_xbrl_validations)
            or bool(self.reference_issues)
        )
