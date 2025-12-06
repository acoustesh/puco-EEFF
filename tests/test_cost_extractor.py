"""Tests for the cost extractor module.

Tests cover:
1. Number parsing (Chilean format)
2. ExtractionResult and CostBreakdown dataclasses
3. Validation logic between PDF and XBRL
4. Full extraction scenarios (mock-based)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from puco_eeff.extractor.cost_extractor import (
    CostBreakdown,
    ExtractionResult,
    LineItem,
    ValidationResult,
    parse_chilean_number,
    validate_extraction,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Tests for parse_chilean_number
# =============================================================================


class TestParseChileanNumber:
    """Tests for Chilean number format parsing."""

    def test_positive_number_with_period(self) -> None:
        """Period is thousands separator: 30.294 = 30294."""
        assert parse_chilean_number("30.294") == 30294

    def test_negative_with_parentheses(self) -> None:
        """Parentheses indicate negative: (30.294) = -30294."""
        assert parse_chilean_number("(30.294)") == -30294

    def test_large_number_multiple_periods(self) -> None:
        """Multiple periods as thousand separators: 1.234.567 = 1234567."""
        assert parse_chilean_number("1.234.567") == 1234567

    def test_negative_large_number(self) -> None:
        """Large negative: (1.234.567) = -1234567."""
        assert parse_chilean_number("(1.234.567)") == -1234567

    def test_number_without_separator(self) -> None:
        """Plain number without separators."""
        assert parse_chilean_number("12345") == 12345

    def test_negative_without_separator(self) -> None:
        """Negative without separators: (12345) = -12345."""
        assert parse_chilean_number("(12345)") == -12345

    def test_zero(self) -> None:
        """Zero value."""
        assert parse_chilean_number("0") == 0

    def test_none_input(self) -> None:
        """None input returns None."""
        assert parse_chilean_number(None) is None

    def test_empty_string(self) -> None:
        """Empty string returns None."""
        assert parse_chilean_number("") is None

    def test_whitespace_only(self) -> None:
        """Whitespace only returns None."""
        assert parse_chilean_number("   ") is None

    def test_non_numeric(self) -> None:
        """Non-numeric string returns None."""
        assert parse_chilean_number("abc") is None

    def test_with_currency_symbol(self) -> None:
        """Should strip currency symbols."""
        assert parse_chilean_number("MUS$ 30.294") == 30294

    def test_with_spaces(self) -> None:
        """Should handle extra spaces."""
        assert parse_chilean_number("  30.294  ") == 30294

    def test_minus_sign_only(self) -> None:
        """Single minus sign returns None."""
        assert parse_chilean_number("-") is None

    def test_period_only(self) -> None:
        """Single period returns None."""
        assert parse_chilean_number(".") is None


# =============================================================================
# Tests for LineItem dataclass
# =============================================================================


class TestLineItem:
    """Tests for LineItem dataclass."""

    def test_create_with_all_values(self) -> None:
        """Create LineItem with all values."""
        item = LineItem(
            concepto="Gastos en personal",
            ytd_actual=-30294,
            ytd_anterior=-28150,
            quarter_actual=-10098,
            quarter_anterior=-9383,
        )
        assert item.concepto == "Gastos en personal"
        assert item.ytd_actual == -30294
        assert item.ytd_anterior == -28150
        assert item.quarter_actual == -10098
        assert item.quarter_anterior == -9383

    def test_create_with_none_values(self) -> None:
        """Create LineItem with None values (default)."""
        item = LineItem(concepto="Test Item")
        assert item.concepto == "Test Item"
        assert item.ytd_actual is None
        assert item.ytd_anterior is None


# =============================================================================
# Tests for CostBreakdown dataclass
# =============================================================================


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    @pytest.fixture
    def sample_breakdown(self) -> CostBreakdown:
        """Create a sample CostBreakdown for testing."""
        breakdown = CostBreakdown(
            nota_number=21,
            nota_title="Costo de Venta",
            page_number=71,
        )
        breakdown.items = [
            LineItem("Gastos en personal", ytd_actual=-30294),
            LineItem("Materiales y repuestos", ytd_actual=-37269),
            LineItem("Energía eléctrica", ytd_actual=-14710),
        ]
        breakdown.total_ytd_actual = -82273  # Sum of items
        return breakdown

    def test_sum_items_ytd_actual(self, sample_breakdown: CostBreakdown) -> None:
        """Sum of YTD actual should match total."""
        expected_sum = -30294 + -37269 + -14710
        assert sample_breakdown.sum_items_ytd_actual() == expected_sum

    def test_is_valid_when_sum_matches(self, sample_breakdown: CostBreakdown) -> None:
        """Should be valid when sum matches total."""
        assert sample_breakdown.is_valid() is True

    def test_is_valid_when_sum_mismatch(self, sample_breakdown: CostBreakdown) -> None:
        """Should be invalid when sum doesn't match total."""
        sample_breakdown.total_ytd_actual = -999999  # Wrong total
        assert sample_breakdown.is_valid() is False

    def test_is_valid_with_none_total(self) -> None:
        """Should be invalid when total is None."""
        breakdown = CostBreakdown(
            nota_number=21,
            nota_title="Test",
        )
        assert breakdown.is_valid() is False


# =============================================================================
# Tests for ValidationResult dataclass
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_status_match(self) -> None:
        """Status should show match."""
        result = ValidationResult(
            field_name="Test Field",
            pdf_value=100,
            xbrl_value=100,
            match=True,
            source="both",
        )
        assert result.status == "✓ Match"

    def test_status_pdf_only(self) -> None:
        """Status should indicate PDF only."""
        result = ValidationResult(
            field_name="Test Field",
            pdf_value=100,
            xbrl_value=None,
            match=True,
            source="pdf_only",
        )
        assert result.status == "⚠ PDF only (no XBRL)"

    def test_status_xbrl_only(self) -> None:
        """Status should indicate XBRL only."""
        result = ValidationResult(
            field_name="Test Field",
            pdf_value=None,
            xbrl_value=100,
            match=False,
            source="xbrl_only",
        )
        assert result.status == "⚠ XBRL only (PDF extraction failed)"

    def test_status_mismatch(self) -> None:
        """Status should show mismatch with difference."""
        result = ValidationResult(
            field_name="Test Field",
            pdf_value=100,
            xbrl_value=110,
            match=False,
            source="both",
            difference=10,
        )
        assert "Mismatch" in result.status
        assert "10" in result.status


# =============================================================================
# Tests for ExtractionResult dataclass
# =============================================================================


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_is_valid_with_both_notas(self) -> None:
        """Valid when both notas extracted and no validations."""
        result = ExtractionResult(
            year=2024,
            quarter=1,
            nota_21=CostBreakdown(21, "Costo de Venta"),
            nota_22=CostBreakdown(22, "Gastos Admin"),
        )
        assert result.is_valid() is True

    def test_is_valid_missing_nota(self) -> None:
        """Invalid when missing a nota and no validations."""
        result = ExtractionResult(
            year=2024,
            quarter=1,
            nota_21=CostBreakdown(21, "Costo de Venta"),
            nota_22=None,
        )
        assert result.is_valid() is False

    def test_is_valid_with_all_validations_pass(self) -> None:
        """Valid when all validations pass."""
        result = ExtractionResult(
            year=2024,
            quarter=2,
            validations=[
                ValidationResult("Field1", 100, 100, True, "both"),
                ValidationResult("Field2", 200, 200, True, "both"),
            ],
        )
        assert result.is_valid() is True

    def test_is_valid_with_validation_fail(self) -> None:
        """Invalid when any validation fails."""
        result = ExtractionResult(
            year=2024,
            quarter=2,
            validations=[
                ValidationResult("Field1", 100, 100, True, "both"),
                ValidationResult("Field2", 200, 300, False, "both"),
            ],
        )
        assert result.is_valid() is False


# =============================================================================
# Tests for validate_extraction function
# =============================================================================


class TestValidateExtraction:
    """Tests for the validate_extraction function."""

    @pytest.fixture
    def nota_21(self) -> CostBreakdown:
        """Sample Nota 21 breakdown."""
        breakdown = CostBreakdown(21, "Costo de Venta")
        breakdown.total_ytd_actual = -170862
        return breakdown

    @pytest.fixture
    def nota_22(self) -> CostBreakdown:
        """Sample Nota 22 breakdown."""
        breakdown = CostBreakdown(22, "Gastos Admin")
        breakdown.total_ytd_actual = -17363
        return breakdown

    def test_both_sources_match(self, nota_21: CostBreakdown, nota_22: CostBreakdown) -> None:
        """Both PDF and XBRL values match."""
        xbrl_totals = {
            "cost_of_sales": -170862,
            "admin_expense": -17363,
        }

        validations = validate_extraction(nota_21, nota_22, xbrl_totals)

        assert len(validations) == 2
        assert all(v.match for v in validations)
        assert all(v.source == "both" for v in validations)

    def test_sign_difference_should_match(self, nota_21: CostBreakdown, nota_22: CostBreakdown) -> None:
        """Absolute values should match even with sign difference."""
        xbrl_totals = {
            "cost_of_sales": 170862,  # Positive vs PDF negative
            "admin_expense": 17363,
        }

        validations = validate_extraction(nota_21, nota_22, xbrl_totals)

        # Should match because we compare absolute values
        assert all(v.match for v in validations)

    def test_pdf_only_no_xbrl(self, nota_21: CostBreakdown, nota_22: CostBreakdown) -> None:
        """PDF-only extraction when no XBRL available."""
        validations = validate_extraction(nota_21, nota_22, None)

        assert len(validations) == 2
        assert all(v.source == "pdf_only" for v in validations)
        assert all(v.match for v in validations)  # Can't fail if no XBRL

    def test_xbrl_only_pdf_failed(self) -> None:
        """XBRL-only when PDF extraction failed."""
        xbrl_totals = {
            "cost_of_sales": -170862,
            "admin_expense": -17363,
        }

        validations = validate_extraction(None, None, xbrl_totals)

        assert len(validations) == 2
        assert all(v.source == "xbrl_only" for v in validations)
        assert all(not v.match for v in validations)

    def test_mismatch_shows_difference(self, nota_21: CostBreakdown) -> None:
        """Mismatch should include difference value."""
        nota_21.total_ytd_actual = -100000
        xbrl_totals = {
            "cost_of_sales": -110000,
            "admin_expense": None,
        }

        validations = validate_extraction(nota_21, None, xbrl_totals)

        cost_val = next(v for v in validations if "Costo" in v.field_name)
        assert cost_val.match is False
        assert cost_val.difference == -10000  # 100000 - 110000


# =============================================================================
# Integration-style tests (with mocks)
# =============================================================================


class TestExtractDetailedCostsMocked:
    """Integration tests using mocks for file I/O."""

    @pytest.fixture
    def mock_paths(self, tmp_path: Path) -> dict[str, Path]:
        """Create mock paths structure."""
        raw_pdf = tmp_path / "raw" / "pdf"
        raw_pdf.mkdir(parents=True)

        processed = tmp_path / "processed"
        processed.mkdir(parents=True)

        return {
            "raw_pdf": raw_pdf,
            "raw_xbrl": tmp_path / "raw" / "xbrl",
            "processed": processed,
            "audit": tmp_path / "audit",
        }

    @patch("puco_eeff.extractor.cost_extractor.get_period_paths")
    @patch("puco_eeff.extractor.cost_extractor.extract_nota_21")
    @patch("puco_eeff.extractor.cost_extractor.extract_nota_22")
    def test_extraction_without_xbrl(
        self,
        mock_nota_22: MagicMock,
        mock_nota_21: MagicMock,
        mock_paths_fn: MagicMock,
        mock_paths: dict[str, Path],
    ) -> None:
        """Test extraction when no XBRL is available (Pucobre.cl case)."""
        from puco_eeff.extractor.cost_extractor import extract_detailed_costs

        # Setup mocks
        mock_paths_fn.return_value = mock_paths

        # Create fake PDF file
        pdf_path = mock_paths["raw_pdf"] / "estados_financieros_2024_Q1.pdf"
        pdf_path.write_text("fake pdf content")

        # Create marker for Pucobre source
        combined_path = mock_paths["raw_pdf"] / "pucobre_combined_2024_Q1.pdf"
        combined_path.write_text("combined pdf")

        # Mock extraction results
        nota_21_mock = CostBreakdown(21, "Costo de Venta")
        nota_21_mock.total_ytd_actual = -54000
        mock_nota_21.return_value = nota_21_mock

        nota_22_mock = CostBreakdown(22, "Gastos Admin")
        nota_22_mock.total_ytd_actual = -12000
        mock_nota_22.return_value = nota_22_mock

        # Run extraction
        result = extract_detailed_costs(2024, 1)

        # Verify
        assert result.source == "pucobre.cl"
        assert result.xbrl_available is False
        assert result.nota_21 is not None
        assert result.nota_22 is not None
        assert all(v.source == "pdf_only" for v in result.validations)

    @patch("puco_eeff.extractor.cost_extractor.get_period_paths")
    @patch("puco_eeff.extractor.cost_extractor.extract_nota_21")
    @patch("puco_eeff.extractor.cost_extractor.extract_nota_22")
    @patch("puco_eeff.extractor.cost_extractor.extract_xbrl_totals")
    def test_extraction_with_xbrl_validation(
        self,
        mock_xbrl: MagicMock,
        mock_nota_22: MagicMock,
        mock_nota_21: MagicMock,
        mock_paths_fn: MagicMock,
        mock_paths: dict[str, Path],
    ) -> None:
        """Test extraction with XBRL validation (CMF case)."""
        from puco_eeff.extractor.cost_extractor import extract_detailed_costs

        # Setup mocks
        mock_paths_fn.return_value = mock_paths

        # Create fake PDF and XBRL files
        pdf_path = mock_paths["raw_pdf"] / "estados_financieros_2024_Q2.pdf"
        pdf_path.write_text("fake pdf content")

        xbrl_path = mock_paths["raw_pdf"] / "estados_financieros_2024_Q2.xbrl"
        xbrl_path.write_text("fake xbrl content")

        # Mock extraction results
        nota_21_mock = CostBreakdown(21, "Costo de Venta")
        nota_21_mock.total_ytd_actual = -170862
        mock_nota_21.return_value = nota_21_mock

        nota_22_mock = CostBreakdown(22, "Gastos Admin")
        nota_22_mock.total_ytd_actual = -17363
        mock_nota_22.return_value = nota_22_mock

        mock_xbrl.return_value = {
            "cost_of_sales": -170862,
            "admin_expense": -17363,
        }

        # Run extraction
        result = extract_detailed_costs(2024, 2)

        # Verify
        assert result.source == "cmf"
        assert result.xbrl_available is True
        assert result.is_valid() is True
        assert all(v.match for v in result.validations)


# =============================================================================
# Test save_extraction_result
# =============================================================================


class TestSaveExtractionResult:
    """Tests for saving extraction results."""

    def test_save_creates_json(self, tmp_path: Path) -> None:
        """Should create a JSON file with extraction data."""
        from puco_eeff.extractor.cost_extractor import save_extraction_result

        result = ExtractionResult(
            year=2024,
            quarter=1,
            source="pucobre.cl",
            xbrl_available=False,
        )
        result.nota_21 = CostBreakdown(21, "Costo de Venta")
        result.nota_21.total_ytd_actual = -50000
        result.nota_21.items = [LineItem("Test", ytd_actual=-50000)]

        output_path = save_extraction_result(result, tmp_path)

        assert output_path.exists()
        assert output_path.name == "detailed_costs.json"

        import json

        with open(output_path) as f:
            data = json.load(f)

        assert data["period"] == "2024_Q1"
        assert data["source"] == "pucobre.cl"
        assert data["xbrl_available"] is False
        assert data["nota_21"]["total_ytd_actual"] == -50000
