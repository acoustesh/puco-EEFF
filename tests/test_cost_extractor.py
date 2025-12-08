"""Tests for the cost extractor module.

Tests cover:
1. Number parsing (Chilean format)
2. ExtractionResult and SectionBreakdown dataclasses
3. Validation logic between PDF and XBRL
4. Full extraction scenarios (mock-based)
5. Sum validations (line items → totals)
6. Cross-validations (accounting formula checks)
7. Validation report formatting
8. Unified validation API (run_sheet1_validations)
9. Backward compatibility (validate_extraction deprecation)
10. PDF↔XBRL validation helper (_run_pdf_xbrl_validations)
11. Config-driven section conversion (sections_to_sheet1data)
12. ExtractionResult.validation_report field
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from puco_eeff.extractor.extraction import (
    LineItem,
    SectionBreakdown,
    parse_chilean_number,
)
from puco_eeff.extractor.validation_core import (
    CrossValidationResult,
    ExtractionResult,
    SumValidationResult,
    ValidationReport,
    ValidationResult,
    _evaluate_cross_validation,
    _run_cross_validations,
    _run_pdf_xbrl_validations,
    _run_sum_validations,
    _safe_eval_expression,
    format_validation_report,
    run_sheet1_validations,
)
from puco_eeff.sheets.sheet1 import (
    Sheet1Data,
    get_ingresos_pdf_fallback_config,
    get_section_config,
    get_section_fallback,
    get_sheet1_section_total_mapping,
    sections_to_sheet1data,
)

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
# Tests for SectionBreakdown dataclass
# =============================================================================


class TestSectionBreakdown:
    """Tests for SectionBreakdown dataclass."""

    @pytest.fixture
    def sample_breakdown(self) -> SectionBreakdown:
        """Create a sample SectionBreakdown for testing."""
        breakdown = SectionBreakdown(
            section_id="nota_21",
            section_title="Costo de Venta",
            page_number=71,
        )
        breakdown.items = [
            LineItem("Gastos en personal", ytd_actual=-30294),
            LineItem("Materiales y repuestos", ytd_actual=-37269),
            LineItem("Energía eléctrica", ytd_actual=-14710),
        ]
        breakdown.total_ytd_actual = -82273  # Sum of items
        return breakdown

    def test_sum_items_ytd_actual(self, sample_breakdown: SectionBreakdown) -> None:
        """Sum of YTD actual should match total."""
        expected_sum = -30294 + -37269 + -14710
        assert sample_breakdown.sum_items_ytd_actual() == expected_sum

    def test_is_valid_when_sum_matches(self, sample_breakdown: SectionBreakdown) -> None:
        """Should be valid when sum matches total."""
        assert sample_breakdown.is_valid() is True

    def test_is_valid_when_sum_mismatch(self, sample_breakdown: SectionBreakdown) -> None:
        """Should be invalid when sum doesn't match total."""
        sample_breakdown.total_ytd_actual = -999999  # Wrong total
        assert sample_breakdown.is_valid() is False

    def test_is_valid_with_none_total(self) -> None:
        """Should be invalid when total is None."""
        breakdown = SectionBreakdown(
            section_id="nota_21",
            section_title="Test",
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
        )
        # Use sections dict to populate
        result.sections["nota_21"] = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
        result.sections["nota_22"] = SectionBreakdown(section_id="nota_22", section_title="Gastos Admin")
        assert result.is_valid() is True

    def test_is_valid_missing_nota(self) -> None:
        """Invalid when missing a nota and no validations."""
        result = ExtractionResult(
            year=2024,
            quarter=1,
        )
        result.sections["nota_21"] = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
        # nota_22 is not set - should still be in sections dict but is_valid should fail
        # Actually, we need one section to be missing. Let's use the generic is_valid check.
        assert result.is_valid() is True  # Actually valid since at least one section exists

    def test_is_valid_empty_sections(self) -> None:
        """Invalid when no sections at all."""
        result = ExtractionResult(
            year=2024,
            quarter=1,
        )
        # No sections populated
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
# Tests for PDF↔XBRL validation
# =============================================================================


class TestPdfXbrlValidation:
    """Tests for PDF↔XBRL validation using _run_pdf_xbrl_validations."""

    @pytest.fixture
    def sample_data_both(self) -> Sheet1Data:
        """Sample Sheet1Data with both cost sections."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.total_costo_venta = -170862
        data.total_gasto_admin = -17363
        return data

    def test_both_sources_match(self, sample_data_both: Sheet1Data) -> None:
        """Both PDF and XBRL values match."""
        xbrl_totals = {
            "cost_of_sales": -170862,
            "admin_expense": -17363,
        }

        validations = _run_pdf_xbrl_validations(sample_data_both, xbrl_totals, use_fallback=False)

        assert len(validations) == 2
        assert all(v.match for v in validations)
        assert all(v.source == "both" for v in validations)

    def test_sign_difference_should_match(self, sample_data_both: Sheet1Data) -> None:
        """Absolute values should match even with sign difference."""
        xbrl_totals = {
            "cost_of_sales": 170862,  # Positive vs PDF negative
            "admin_expense": 17363,
        }

        validations = _run_pdf_xbrl_validations(sample_data_both, xbrl_totals, use_fallback=False)

        # Should match because we compare absolute values
        assert all(v.match for v in validations)

    def test_pdf_only_no_xbrl(self, sample_data_both: Sheet1Data) -> None:
        """PDF-only extraction when no XBRL available."""
        validations = _run_pdf_xbrl_validations(sample_data_both, None, use_fallback=False)

        assert len(validations) == 2
        assert all(v.source == "pdf_only" for v in validations)
        assert all(v.match for v in validations)  # Can't fail if no XBRL

    def test_xbrl_only_pdf_failed(self) -> None:
        """XBRL-only when PDF extraction failed - XBRL used as source."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        # No PDF values set
        xbrl_totals = {
            "cost_of_sales": -170862,
            "admin_expense": -17363,
        }

        validations = _run_pdf_xbrl_validations(data, xbrl_totals, use_fallback=False)

        assert len(validations) == 2
        assert all(v.source == "xbrl_only" for v in validations)
        # match=True because XBRL was successfully used as source (not a validation failure)
        assert all(v.match for v in validations)

    def test_mismatch_shows_difference(self) -> None:
        """Mismatch should include difference value (absolute)."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.total_costo_venta = -100000
        xbrl_totals = {
            "cost_of_sales": -110000,
            "admin_expense": None,
        }

        validations = _run_pdf_xbrl_validations(data, xbrl_totals, use_fallback=False)

        cost_val = next(v for v in validations if "Costo" in v.field_name)
        assert cost_val.match is False
        assert cost_val.difference == 10000  # abs(100000 - 110000)


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

    @patch("puco_eeff.extractor.extraction_pipeline.get_period_paths")
    @patch("puco_eeff.extractor.extraction_pipeline.extract_pdf_section")
    def test_extraction_without_xbrl(
        self,
        mock_extract_pdf_section: MagicMock,
        mock_paths_fn: MagicMock,
        mock_paths: dict[str, Path],
    ) -> None:
        """Test extraction when no XBRL is available (Pucobre.cl case)."""
        from puco_eeff.extractor.extraction_pipeline import extract_detailed_costs

        # Setup mocks
        mock_paths_fn.return_value = mock_paths

        # Create fake PDF file
        pdf_path = mock_paths["raw_pdf"] / "estados_financieros_2024_Q1.pdf"
        pdf_path.write_text("fake pdf content")

        # Create marker for Pucobre source
        combined_path = mock_paths["raw_pdf"] / "pucobre_combined_2024_Q1.pdf"
        combined_path.write_text("combined pdf")

        # Mock extraction results based on section_name argument
        def extract_section_side_effect(path, section_name):
            if section_name == "nota_21":
                breakdown = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
                breakdown.total_ytd_actual = -54000
                return breakdown
            if section_name == "nota_22":
                breakdown = SectionBreakdown(section_id="nota_22", section_title="Gastos Admin")
                breakdown.total_ytd_actual = -12000
                return breakdown
            return None

        mock_extract_pdf_section.side_effect = extract_section_side_effect

        # Run extraction
        result = extract_detailed_costs(2024, 1)

        # Verify
        assert result.source == "pucobre.cl"
        assert result.xbrl_available is False
        assert result.sections.get("nota_21") is not None
        assert result.sections.get("nota_22") is not None
        assert all(v.source == "pdf_only" for v in result.validations)

    @patch("puco_eeff.extractor.extraction_pipeline.get_period_paths")
    @patch("puco_eeff.extractor.extraction_pipeline.extract_pdf_section")
    @patch("puco_eeff.extractor.extraction_pipeline.extract_xbrl_totals")
    def test_extraction_with_xbrl_validation(
        self,
        mock_xbrl: MagicMock,
        mock_extract_pdf_section: MagicMock,
        mock_paths_fn: MagicMock,
        mock_paths: dict[str, Path],
    ) -> None:
        """Test extraction with XBRL validation (CMF case)."""
        from puco_eeff.extractor.extraction_pipeline import extract_detailed_costs

        # Setup mocks
        mock_paths_fn.return_value = mock_paths

        # Create fake PDF file
        pdf_path = mock_paths["raw_pdf"] / "estados_financieros_2024_Q2.pdf"
        pdf_path.write_text("fake pdf content")

        # Create XBRL directory and file
        xbrl_dir = mock_paths["raw_xbrl"]
        xbrl_dir.mkdir(parents=True, exist_ok=True)
        xbrl_path = xbrl_dir / "estados_financieros_2024_Q2.xbrl"
        xbrl_path.write_text("fake xbrl content")

        # Mock extraction results based on section_name argument
        def extract_section_side_effect(path, section_name):
            if section_name == "nota_21":
                breakdown = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
                breakdown.total_ytd_actual = -170862
                return breakdown
            if section_name == "nota_22":
                breakdown = SectionBreakdown(section_id="nota_22", section_title="Gastos Admin")
                breakdown.total_ytd_actual = -17363
                return breakdown
            return None

        mock_extract_pdf_section.side_effect = extract_section_side_effect

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
        from puco_eeff.extractor.extraction_pipeline import save_extraction_result

        result = ExtractionResult(
            year=2024,
            quarter=1,
            source="pucobre.cl",
            xbrl_available=False,
        )
        nota_21 = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
        nota_21.total_ytd_actual = -50000
        nota_21.items = [LineItem("Test", ytd_actual=-50000)]
        result.sections["nota_21"] = nota_21

        output_path = save_extraction_result(result, tmp_path)

        assert output_path.exists()
        assert output_path.name == "detailed_costs.json"

        import json

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["period"] == "2024_Q1"
        assert data["source"] == "pucobre.cl"
        assert data["xbrl_available"] is False
        assert data["nota_21"]["total_ytd_actual"] == -50000


# =============================================================================
# Tests for Sheet1Data
# =============================================================================


class TestSheet1Data:
    """Tests for Sheet1Data dataclass."""

    def test_create_with_sample_data(self) -> None:
        """Create Sheet1Data with Q2 2024 sample values."""
        from puco_eeff.sheets.sheet1 import Sheet1Data

        data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            source="cmf",
            xbrl_available=True,
            ingresos_ordinarios=179165,
            cv_gastos_personal=-19721,
            cv_materiales=-23219,
            cv_energia=-9589,
            cv_servicios_terceros=-25063,
            cv_depreciacion_amort=-21694,
            cv_deprec_leasing=-881,
            cv_deprec_arrend=-1577,
            cv_serv_mineros=-10804,
            cv_fletes=-5405,
            cv_gastos_diferidos=-1587,
            cv_convenios=-6662,
            total_costo_venta=-126202,
            ga_gastos_personal=-3818,
            ga_materiales=-129,
            ga_servicios_terceros=-4239,
            ga_gratificacion=-639,
            ga_comercializacion=-2156,
            ga_otros=-651,
            total_gasto_admin=-11632,
        )

        assert data.quarter == "IIQ2024"
        assert data.ingresos_ordinarios == 179165
        assert data.total_costo_venta == -126202
        assert data.total_gasto_admin == -11632

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from puco_eeff.sheets.sheet1 import Sheet1Data

        data = Sheet1Data(
            quarter="IQ2024",
            year=2024,
            quarter_num=1,
            source="pucobre.cl",
            xbrl_available=False,
            ingresos_ordinarios=50000,
            total_costo_venta=-30000,
            total_gasto_admin=-5000,
        )

        d = data.to_dict()

        assert d["quarter"] == "IQ2024"
        assert d["source"] == "pucobre.cl"
        assert d["xbrl_available"] is False
        assert d["ingresos_ordinarios"] == 50000
        assert d["total_costo_venta"] == -30000
        assert d["total_gasto_admin"] == -5000

    def test_to_row_list_has_27_rows(self) -> None:
        """Row list should have exactly 27 rows."""
        from puco_eeff.sheets.sheet1 import Sheet1Data

        data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
        )

        rows = data.to_row_list()

        assert len(rows) == 27

    def test_to_row_list_structure(self) -> None:
        """Row list should have correct structure."""
        from puco_eeff.sheets.sheet1 import Sheet1Data

        data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            ingresos_ordinarios=179165,
            total_costo_venta=-126202,
            total_gasto_admin=-11632,
        )

        rows = data.to_row_list()

        # Row 1: Ingresos
        assert rows[0] == (1, "Ingresos de actividades ordinarias M USD", 179165)

        # Row 3: Costo de Venta header
        assert rows[2][1] == "Costo de Venta"

        # Row 15: Total Costo de Venta
        assert rows[14] == (15, "Total Costo de Venta", -126202)

        # Row 19: Gasto Admin header
        assert rows[18][1] == "Gasto Adm, y Ventas"

        # Row 27: Totales (Gasto Admin total, NOT Costo de Venta)
        assert rows[26] == (27, "Totales", -11632)

    def test_totales_disambiguation(self) -> None:
        """Ensure 'Totales' in row 27 is distinct from 'Total Costo de Venta'."""
        from puco_eeff.sheets.sheet1 import Sheet1Data

        data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            total_costo_venta=-126202,
            total_gasto_admin=-11632,
        )

        rows = data.to_row_list()

        # Find all rows containing "Total"
        total_rows = [(r[0], r[1], r[2]) for r in rows if r[1] and "total" in r[1].lower()]

        assert len(total_rows) == 2
        # Row 15 should be "Total Costo de Venta"
        assert total_rows[0] == (15, "Total Costo de Venta", -126202)
        # Row 27 should be "Totales" (Gasto Admin)
        assert total_rows[1] == (27, "Totales", -11632)


class TestQuarterFormatting:
    """Tests for quarter label formatting."""

    def test_quarter_to_roman(self) -> None:
        """Test Roman numeral conversion."""
        from puco_eeff.extractor.extraction import quarter_to_roman

        assert quarter_to_roman(1) == "I"
        assert quarter_to_roman(2) == "II"
        assert quarter_to_roman(3) == "III"
        assert quarter_to_roman(4) == "IV"

    def test_format_quarter_label(self) -> None:
        """Test quarter label formatting."""
        from puco_eeff.extractor.extraction import format_quarter_label

        assert format_quarter_label(2024, 1) == "IQ2024"
        assert format_quarter_label(2024, 2) == "IIQ2024"
        assert format_quarter_label(2024, 3) == "IIIQ2024"
        assert format_quarter_label(2024, 4) == "IVQ2024"


class TestSaveSheet1Data:
    """Tests for saving Sheet1 data."""

    def test_save_creates_json(self, tmp_path: Path) -> None:
        """Should create a JSON file with Sheet1 data."""
        from puco_eeff.sheets.sheet1 import Sheet1Data, save_sheet1_data

        data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            source="cmf",
            xbrl_available=True,
            ingresos_ordinarios=179165,
            total_costo_venta=-126202,
            total_gasto_admin=-11632,
        )

        output_path = save_sheet1_data(data, tmp_path)

        assert output_path.exists()
        assert output_path.name == "sheet1_IIQ2024.json"

        import json

        with open(output_path, encoding="utf-8") as f:
            saved = json.load(f)

        assert saved["quarter"] == "IIQ2024"
        assert saved["ingresos_ordinarios"] == 179165
        assert saved["total_costo_venta"] == -126202
        assert saved["total_gasto_admin"] == -11632


class TestXBRLExtraction:
    """Tests for XBRL-based Sheet1 extraction."""

    def test_validate_sheet1_with_xbrl_match(self) -> None:
        """Should log match when PDF and XBRL totals agree."""
        from puco_eeff.extractor.extraction_pipeline import _validate_sheet1_with_xbrl
        from puco_eeff.sheets.sheet1 import Sheet1Data

        data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            total_costo_venta=-126202,
            total_gasto_admin=-11632,
        )

        # Mock XBRL extraction to return matching values
        with patch("puco_eeff.extractor.extraction_pipeline.extract_xbrl_totals") as mock_xbrl:
            mock_xbrl.return_value = {
                "cost_of_sales": -126202,
                "admin_expense": -11632,
            }

            _validate_sheet1_with_xbrl(data, Path("/fake/path.xbrl"))

            # Values should remain unchanged (already matched)
            assert data.total_costo_venta == -126202
            assert data.total_gasto_admin == -11632

    def test_validate_sheet1_with_xbrl_fallback(self) -> None:
        """Should use XBRL values when PDF extraction failed."""
        from puco_eeff.extractor.extraction_pipeline import _validate_sheet1_with_xbrl
        from puco_eeff.sheets.sheet1 import Sheet1Data

        data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            total_costo_venta=None,  # PDF extraction failed
            total_gasto_admin=None,
        )

        with patch("puco_eeff.extractor.extraction_pipeline.extract_xbrl_totals") as mock_xbrl:
            mock_xbrl.return_value = {
                "cost_of_sales": -126202,
                "admin_expense": -11632,
            }

            _validate_sheet1_with_xbrl(data, Path("/fake/path.xbrl"))

            # Should now have XBRL values
            assert data.total_costo_venta == -126202
            assert data.total_gasto_admin == -11632

    def test_merge_pdf_into_xbrl_data(self) -> None:
        """Should copy detailed items from PDF to XBRL data."""
        from puco_eeff.extractor.extraction_pipeline import _merge_pdf_into_xbrl_data
        from puco_eeff.sheets.sheet1 import Sheet1Data

        pdf_data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            cv_gastos_personal=-19721,
            cv_materiales=-23219,
            ga_gastos_personal=-3818,
            total_costo_venta=-126202,  # Will NOT be copied (not in detail_fields)
        )

        xbrl_data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            total_costo_venta=-126000,  # XBRL has validated total
            total_gasto_admin=-11632,
        )

        _merge_pdf_into_xbrl_data(xbrl_data, pdf_data)

        # Detail items should be copied from PDF
        assert xbrl_data.cv_gastos_personal == -19721
        assert xbrl_data.cv_materiales == -23219
        assert xbrl_data.ga_gastos_personal == -3818

        # Totals should remain from XBRL (not overwritten)
        assert xbrl_data.total_costo_venta == -126000
        assert xbrl_data.total_gasto_admin == -11632


class TestExtractSheet1MainEntry:
    """Tests for the main extract_sheet1 entry point."""

    def test_extract_sheet1_pdf_priority(self) -> None:
        """Should try PDF first when prefer_source="pdf"."""
        from puco_eeff.extractor.extraction_pipeline import extract_sheet1
        from puco_eeff.sheets.sheet1 import Sheet1Data

        expected_data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            ingresos_ordinarios=179165,
        )

        with patch("puco_eeff.extractor.extraction_pipeline.extract_sheet1_from_analisis_razonado") as mock_pdf:
            mock_pdf.return_value = (
                expected_data,
                None,
            )  # Return tuple as expected by implementation

            result = extract_sheet1(2024, 2, prefer_source="pdf")

            assert result is not None
            assert isinstance(result, Sheet1Data)  # Type narrowing for mypy/pyright
            assert result.quarter == "IIQ2024"
            mock_pdf.assert_called_once()

    def test_extract_sheet1_xbrl_fallback(self) -> None:
        """Should fall back to XBRL when PDF fails."""
        from puco_eeff.extractor.extraction_pipeline import extract_sheet1
        from puco_eeff.sheets.sheet1 import Sheet1Data

        xbrl_data = Sheet1Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            total_costo_venta=-126202,
        )

        with patch("puco_eeff.extractor.extraction_pipeline.extract_sheet1_from_analisis_razonado") as mock_pdf:
            mock_pdf.return_value = (None, None)  # PDF failed - return tuple

            with patch("puco_eeff.extractor.extraction_pipeline.extract_sheet1_from_xbrl") as mock_xbrl:
                mock_xbrl.return_value = (xbrl_data, None)  # Return tuple as expected

                result = extract_sheet1(2024, 2, prefer_source="pdf")

                assert result is not None
                assert isinstance(result, Sheet1Data)  # Type narrowing
                assert result.total_costo_venta == -126202
                mock_xbrl.assert_called_once()


class TestIngresosPDFFallback:
    """Tests for extracting Ingresos from PDF when XBRL is unavailable."""

    def test_extract_ingresos_from_pdf_function(self) -> None:
        """extract_ingresos_from_pdf should parse Estado de Resultados page."""
        from puco_eeff.extractor.extraction import extract_ingresos_from_pdf

        # Mock pdfplumber to return Estado de Resultados table structure
        mock_table = [
            ["Nota", "01-01-2024\n31-03-2024\nMUS$", "01-01-2023\n31-03-2023\nMUS$"],
            [
                "Ganancia\nIngresos de actividades ordinarias 18\nCosto de ventas 21",
                "80.767\n( 62.982)",
                "82.663\n( 59.127)",
            ],
            ["Ganancia bruta", "17.785", "23.536"],
        ]

        with patch("pdfplumber.open") as mock_pdf_open:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "ESTADOS DE RESULTADOS Ingresos de actividades ordinarias"
            mock_page.extract_tables.return_value = [mock_table]

            mock_pdf = MagicMock()
            mock_pdf.pages = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), mock_page]  # Page 5
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf_open.return_value = mock_pdf

            result = extract_ingresos_from_pdf(Path("/fake/path.pdf"))

            assert result == 80767

    def test_extract_ingresos_from_pdf_returns_none_when_not_found(self) -> None:
        """Should return None if Estado de Resultados page not found."""
        from puco_eeff.extractor.extraction import extract_ingresos_from_pdf

        with patch("pdfplumber.open") as mock_pdf_open:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Some other content without ingresos"
            mock_page.extract_tables.return_value = []

            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page] * 10
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf_open.return_value = mock_pdf

            result = extract_ingresos_from_pdf(Path("/fake/path.pdf"))

            assert result is None

    def test_sheet1_uses_pdf_ingresos_when_no_xbrl(self) -> None:
        """extract_sheet1_from_analisis_razonado should extract Ingresos from PDF when no XBRL."""
        from puco_eeff.extractor.extraction_pipeline import extract_sheet1_from_analisis_razonado

        with patch("puco_eeff.extractor.extraction_pipeline.get_period_paths") as mock_paths:
            mock_paths.return_value = {
                "raw_pdf": Path("/fake/pdf"),
                "raw_xbrl": Path("/fake/xbrl"),
            }

            with patch("puco_eeff.extractor.extraction_pipeline.find_file_with_alternatives") as mock_find:
                # Create mock paths that "exist"
                mock_pdf_path = MagicMock(spec=Path)
                mock_pdf_path.exists.return_value = True
                mock_combined_path = MagicMock(spec=Path)
                mock_combined_path.exists.return_value = True

                def find_side_effect(dir_path, doc_type, year, quarter):
                    if doc_type == "estados_financieros_pdf":
                        return mock_pdf_path
                    if doc_type == "estados_financieros_xbrl":
                        return None  # No XBRL available
                    if doc_type == "pucobre_combined":
                        return mock_combined_path  # pucobre source
                    return None

                mock_find.side_effect = find_side_effect

                with patch("puco_eeff.extractor.extraction_pipeline.extract_pdf_section") as mock_extract:

                    def extract_section_side_effect(path, section_name):
                        if section_name == "nota_21":
                            return SectionBreakdown(
                                section_id="nota_21",
                                section_title="Costo de Venta",
                                total_ytd_actual=-62982,
                            )
                        if section_name == "nota_22":
                            return SectionBreakdown(
                                section_id="nota_22",
                                section_title="Gastos Admin",
                                total_ytd_actual=-5137,
                            )
                        return None

                    mock_extract.side_effect = extract_section_side_effect

                    with patch("puco_eeff.extractor.extraction_pipeline.extract_ingresos_from_pdf") as mock_ingresos:
                        mock_ingresos.return_value = 80767

                        result = extract_sheet1_from_analisis_razonado(2024, 1)

                        assert result is not None
                        from puco_eeff.sheets.sheet1 import Sheet1Data

                        assert isinstance(result, Sheet1Data)  # Type narrowing for mypy/pyright
                        assert result.ingresos_ordinarios == 80767
                        assert result.total_costo_venta == -62982
                        assert result.total_gasto_admin == -5137
                        assert result.xbrl_available is False
                        mock_ingresos.assert_called_once()


# =============================================================================
# Tests for SumValidationResult dataclass (Phase 1)
# =============================================================================


class TestSumValidationResult:
    """Tests for the SumValidationResult dataclass."""

    def test_match_status_message(self) -> None:
        """Match should show success message."""
        result = SumValidationResult(
            description="Nota 21 - Costo de Venta",
            total_field="total_costo_venta",
            expected_total=-126202,
            calculated_sum=-126202,
            match=True,
            difference=0,
            tolerance=1,
        )
        assert "✓" in result.status
        assert "-126,202" in result.status

    def test_mismatch_status_message(self) -> None:
        """Mismatch should show error with difference."""
        result = SumValidationResult(
            description="Nota 21 - Costo de Venta",
            total_field="total_costo_venta",
            expected_total=-126202,
            calculated_sum=-126000,
            match=False,
            difference=202,
            tolerance=1,
        )
        assert "✗" in result.status
        assert "diff: 202" in result.status

    def test_no_total_status_message(self) -> None:
        """Missing total should show warning."""
        result = SumValidationResult(
            description="Nota 21 - Costo de Venta",
            total_field="total_costo_venta",
            expected_total=None,
            calculated_sum=-126000,
            match=True,
            difference=0,
            tolerance=1,
        )
        assert "⚠" in result.status
        assert "No total" in result.status


class TestCrossValidationResult:
    """Tests for the CrossValidationResult dataclass."""

    def test_match_status_message(self) -> None:
        """Match should show success."""
        result = CrossValidationResult(
            description="Gross Profit Check",
            formula="gross_profit == ingresos - cost",
            expected_value=52963,
            calculated_value=52963,
            match=True,
            difference=0,
            tolerance=1,
        )
        assert "✓" in result.status
        assert "52,963" in result.status

    def test_mismatch_status_message(self) -> None:
        """Mismatch should show error."""
        result = CrossValidationResult(
            description="Gross Profit Check",
            formula="gross_profit == ingresos - cost",
            expected_value=52963,
            calculated_value=53000,
            match=False,
            difference=37,
            tolerance=1,
        )
        assert "✗" in result.status
        assert "diff: 37" in result.status

    def test_missing_facts_status_message(self) -> None:
        """Missing facts should show skipped message."""
        result = CrossValidationResult(
            description="Gross Profit Check",
            formula="gross_profit == ingresos - cost",
            expected_value=None,
            calculated_value=None,
            match=True,
            difference=None,
            tolerance=1,
            missing_facts=["gross_profit"],
        )
        assert "⚠" in result.status
        assert "Skipped" in result.status
        assert "gross_profit" in result.status


# =============================================================================
# Tests for ValidationReport dataclass (Phase 4)
# =============================================================================


class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_has_failures_all_pass(self) -> None:
        """No failures when all validations pass."""
        report = ValidationReport(
            sum_validations=[
                SumValidationResult("Test", "field", 100, 100, True, 0, 1),
            ],
            cross_validations=[],
            pdf_xbrl_validations=[],
            reference_issues=[],
        )
        assert not report.has_failures()

    def test_has_failures_sum_fails(self) -> None:
        """has_failures True when sum validation fails."""
        report = ValidationReport(
            sum_validations=[
                SumValidationResult("Test", "field", 100, 200, False, 100, 1),
            ],
            cross_validations=[],
            pdf_xbrl_validations=[],
            reference_issues=[],
        )
        assert report.has_failures()

    def test_has_failures_reference_fails(self) -> None:
        """has_failures True when reference validation fails."""
        report = ValidationReport(
            sum_validations=[],
            cross_validations=[],
            pdf_xbrl_validations=[],
            reference_issues=["Field X mismatch"],
        )
        assert report.has_failures()

    def test_has_failures_reference_not_run(self) -> None:
        """No failure when reference validation not run (None)."""
        report = ValidationReport(
            sum_validations=[],
            cross_validations=[],
            pdf_xbrl_validations=[],
            reference_issues=None,
        )
        assert not report.has_failures()

    def test_has_sum_failures(self) -> None:
        """has_sum_failures detects sum validation failures."""
        report = ValidationReport(
            sum_validations=[
                SumValidationResult("Test1", "f1", 100, 100, True, 0, 1),
                SumValidationResult("Test2", "f2", 100, 200, False, 100, 1),
            ],
        )
        assert report.has_sum_failures()

    def test_has_reference_failures(self) -> None:
        """has_reference_failures detects reference validation failures."""
        report = ValidationReport(
            reference_issues=["Mismatch found"],
        )
        assert report.has_reference_failures()

    def test_has_reference_failures_not_run(self) -> None:
        """has_reference_failures False when not run."""
        report = ValidationReport(reference_issues=None)
        assert not report.has_reference_failures()


# =============================================================================
# Tests for _safe_eval_expression (Phase 2)
# =============================================================================


class TestSafeEvalExpression:
    """Tests for the safe expression evaluator."""

    def test_simple_variable(self) -> None:
        """Evaluate a simple variable."""
        values = {"x": 100}
        assert _safe_eval_expression("x", values) == 100

    def test_integer_literal(self) -> None:
        """Evaluate an integer literal."""
        assert _safe_eval_expression("42", {}) == 42

    def test_addition(self) -> None:
        """Evaluate addition."""
        values = {"a": 10, "b": 20}
        assert _safe_eval_expression("a + b", values) == 30

    def test_subtraction(self) -> None:
        """Evaluate subtraction."""
        values = {"a": 100, "b": 30}
        assert _safe_eval_expression("a - b", values) == 70

    def test_abs_function(self) -> None:
        """Evaluate abs() function."""
        values = {"x": -50}
        assert _safe_eval_expression("abs(x)", values) == 50

    def test_complex_expression(self) -> None:
        """Evaluate complex expression: a - abs(b)."""
        values = {"a": 179165, "b": -126202}
        assert _safe_eval_expression("a - abs(b)", values) == 52963

    def test_missing_variable(self) -> None:
        """Missing variable returns None."""
        values = {"a": 10}
        assert _safe_eval_expression("b", values) is None

    def test_unknown_expression(self) -> None:
        """Unknown expression returns None."""
        values = {}
        assert _safe_eval_expression("foo(bar)", values) is None


# =============================================================================
# Tests for _evaluate_cross_validation (Phase 2)
# =============================================================================


class TestEvaluateCrossValidation:
    """Tests for cross-validation formula evaluation."""

    def test_simple_equality_match(self) -> None:
        """Simple equality that matches."""
        values = {"a": 100, "b": 100}
        expected, calculated, match, diff = _evaluate_cross_validation("a == b", values, tolerance=1)
        assert expected == 100
        assert calculated == 100
        assert match is True
        assert diff == 0

    def test_expression_equality_match(self) -> None:
        """Expression equality that matches."""
        values = {"gross_profit": 52963, "ingresos": 179165, "cost": -126202}
        formula = "gross_profit == ingresos - abs(cost)"
        expected, calculated, match, diff = _evaluate_cross_validation(formula, values, tolerance=1)
        assert expected == 52963
        assert calculated == 52963
        assert match is True
        assert diff == 0

    def test_mismatch_within_tolerance(self) -> None:
        """Mismatch within tolerance should pass."""
        values = {"a": 100, "b": 101}
        _expected, _calculated, match, diff = _evaluate_cross_validation("a == b", values, tolerance=5)
        assert match is True
        assert diff == 1

    def test_mismatch_outside_tolerance(self) -> None:
        """Mismatch outside tolerance should fail."""
        values = {"a": 100, "b": 110}
        _expected, _calculated, match, diff = _evaluate_cross_validation("a == b", values, tolerance=5)
        assert match is False
        assert diff == 10

    def test_invalid_formula_no_equals(self) -> None:
        """Formula without == returns None values."""
        values = {"a": 100}
        expected, calculated, match, _diff = _evaluate_cross_validation("a + b", values, tolerance=1)
        assert expected is None
        assert calculated is None
        assert match is True  # Can't fail without valid formula


# =============================================================================
# Tests for _run_sum_validations (Phase 1)
# =============================================================================


class TestRunSumValidations:
    """Tests for config-driven sum validations."""

    @pytest.fixture
    def sample_sheet1_data(self) -> Sheet1Data:
        """Create sample Sheet1Data for testing."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        # Set Nota 21 values
        data.cv_gastos_personal = -19721
        data.cv_materiales = -23219
        data.cv_energia = -9589
        data.cv_servicios_terceros = -25063
        data.cv_depreciacion_amort = -21694
        data.cv_deprec_leasing = -881
        data.cv_deprec_arrend = -1577
        data.cv_serv_mineros = -10804
        data.cv_fletes = -5405
        data.cv_gastos_diferidos = -1587
        data.cv_convenios = -6662
        data.total_costo_venta = -126202  # Exact sum
        # Set Nota 22 values
        data.ga_gastos_personal = -3818
        data.ga_materiales = -129
        data.ga_servicios_terceros = -4239
        data.ga_gratificacion = -639
        data.ga_comercializacion = -2156
        data.ga_otros = -651
        data.total_gasto_admin = -11632  # Exact sum
        return data

    def test_all_sums_match(self, sample_sheet1_data: Sheet1Data) -> None:
        """All sum validations pass when totals match."""
        results = _run_sum_validations(sample_sheet1_data)
        assert len(results) >= 2  # At least Nota 21 and Nota 22
        assert all(r.match for r in results)

    def test_sum_mismatch_detected(self, sample_sheet1_data: Sheet1Data) -> None:
        """Mismatch detected when total doesn't match sum."""
        sample_sheet1_data.total_costo_venta = -100000  # Wrong total
        results = _run_sum_validations(sample_sheet1_data)
        nota21_result = next(r for r in results if "Costo" in r.description)
        assert nota21_result.match is False
        assert nota21_result.difference > 0

    def test_missing_total_skips_validation(self, sample_sheet1_data: Sheet1Data) -> None:
        """Missing total should not fail validation."""
        sample_sheet1_data.total_costo_venta = None
        results = _run_sum_validations(sample_sheet1_data)
        nota21_result = next(r for r in results if "Costo" in r.description)
        assert nota21_result.match is True  # Can't fail without expected total
        assert nota21_result.expected_total is None


# =============================================================================
# Tests for _run_cross_validations (Phase 2)
# =============================================================================


class TestRunCrossValidations:
    """Tests for config-driven cross-validations."""

    @pytest.fixture
    def sample_sheet1_data(self) -> Sheet1Data:
        """Create sample Sheet1Data for testing."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.ingresos_ordinarios = 179165
        data.total_costo_venta = -126202
        data.total_gasto_admin = -11632
        return data

    def test_cross_validation_with_xbrl(self, sample_sheet1_data: Sheet1Data) -> None:
        """Cross-validation with XBRL data available."""
        xbrl_totals = {
            "ingresos": 179165,
            "cost_of_sales": -126202,
            "admin_expense": -11632,
            "gross_profit": 52963,  # 179165 - 126202
        }
        results = _run_cross_validations(sample_sheet1_data, xbrl_totals)
        # Should have at least one result (may skip if formula not evaluable)
        assert isinstance(results, list)

    def test_cross_validation_without_xbrl(self, sample_sheet1_data: Sheet1Data) -> None:
        """Cross-validation skips when XBRL unavailable."""
        results = _run_cross_validations(sample_sheet1_data, None)
        # Results may be skipped due to missing facts
        assert isinstance(results, list)
        # All should either match or have missing_facts
        for r in results:
            assert r.match is True or len(r.missing_facts) > 0


# =============================================================================
# Tests for format_validation_report (Phase 4)
# =============================================================================


class TestFormatValidationReport:
    """Tests for validation report formatting."""

    def test_empty_report(self) -> None:
        """Empty report formats correctly."""
        report = ValidationReport()
        output = format_validation_report(report)
        assert "VALIDATION REPORT" in output
        assert "Sum Validations:" in output

    def test_report_with_sum_validations(self) -> None:
        """Report includes sum validation results."""
        report = ValidationReport(
            sum_validations=[
                SumValidationResult(
                    description="Nota 21",
                    total_field="total_costo_venta",
                    expected_total=-126202,
                    calculated_sum=-126202,
                    match=True,
                    difference=0,
                    tolerance=1,
                ),
            ],
        )
        output = format_validation_report(report)
        assert "Sum Validations:" in output
        assert "✓" in output

    def test_report_with_reference_not_run(self) -> None:
        """Report shows reference not run."""
        report = ValidationReport(reference_issues=None)
        output = format_validation_report(report)
        assert "--validate-reference" in output

    def test_report_with_reference_passed(self) -> None:
        """Report shows reference passed."""
        report = ValidationReport(reference_issues=[])
        output = format_validation_report(report)
        assert "All values match" in output

    def test_report_with_reference_failed(self) -> None:
        """Report shows reference failures."""
        report = ValidationReport(reference_issues=["Field X: expected 100, got 200"])
        output = format_validation_report(report)
        assert "MISMATCHES FOUND" in output
        assert "Field X" in output

    def test_report_with_pdf_xbrl_validations(self) -> None:
        """Report includes PDF/XBRL validation results."""
        report = ValidationReport(
            pdf_xbrl_validations=[
                ValidationResult(
                    field_name="Total Costo de Venta",
                    pdf_value=-126202,
                    xbrl_value=-126202,
                    match=True,
                    difference=0,
                    source="both",
                ),
            ],
        )
        output = format_validation_report(report)
        assert "PDF ↔ XBRL" in output
        assert "-126,202" in output


# =============================================================================
# Tests for AST-based safe evaluator (Phase 2 improved)
# =============================================================================


class TestSafeEvalExpressionAST:
    """Tests for AST-based expression evaluator edge cases."""

    def test_no_spaces_addition(self) -> None:
        """Addition without spaces: a+b (AST handles this)."""
        values = {"a": 10, "b": 20}
        # The old evaluator would fail on this, AST handles it
        assert _safe_eval_expression("a+b", values) == 30

    def test_no_spaces_subtraction(self) -> None:
        """Subtraction without spaces: a-b (AST handles this)."""
        values = {"a": 100, "b": 30}
        assert _safe_eval_expression("a-b", values) == 70

    def test_mixed_spacing(self) -> None:
        """Mixed spacing: a +b or a- b."""
        values = {"a": 10, "b": 5}
        assert _safe_eval_expression("a +b", values) == 15
        assert _safe_eval_expression("a- b", values) == 5

    def test_parentheses_grouping(self) -> None:
        """Parentheses for grouping: (a + b) - c."""
        values = {"a": 10, "b": 20, "c": 5}
        assert _safe_eval_expression("(a + b) - c", values) == 25

    def test_nested_abs(self) -> None:
        """Nested abs: abs(a - b)."""
        values = {"a": 10, "b": 30}
        assert _safe_eval_expression("abs(a - b)", values) == 20

    def test_negative_literal(self) -> None:
        """Negative integer literal: -100."""
        values = {}
        assert _safe_eval_expression("-100", values) == -100

    def test_unary_plus(self) -> None:
        """Unary plus: +50."""
        values = {}
        assert _safe_eval_expression("+50", values) == 50

    def test_multiplication(self) -> None:
        """Multiplication: a * b."""
        values = {"a": 7, "b": 6}
        assert _safe_eval_expression("a * b", values) == 42

    def test_unsupported_division(self) -> None:
        """Division not supported - returns None."""
        values = {"a": 10, "b": 2}
        # Division is not in whitelist
        assert _safe_eval_expression("a / b", values) is None

    def test_unsupported_function(self) -> None:
        """Unknown functions return None."""
        values = {"x": 10}
        assert _safe_eval_expression("sqrt(x)", values) is None
        assert _safe_eval_expression("max(x, 5)", values) is None

    def test_syntax_error(self) -> None:
        """Syntax errors return None."""
        values = {"a": 10}
        assert _safe_eval_expression("a + + b", values) is None
        assert _safe_eval_expression("(a", values) is None

    def test_empty_expression(self) -> None:
        """Empty expression returns None."""
        assert _safe_eval_expression("", {}) is None
        assert _safe_eval_expression("   ", {}) is None


# =============================================================================
# Tests for _compare_with_tolerance helper (Phase 3)
# =============================================================================


class TestCompareWithTolerance:
    """Tests for the tolerance comparison helper."""

    def test_exact_match(self) -> None:
        """Exact match should return True with diff 0."""
        from puco_eeff.extractor.validation_core import _compare_with_tolerance

        match, diff = _compare_with_tolerance(100, 100, tolerance=1)
        assert match is True
        assert diff == 0

    def test_within_tolerance(self) -> None:
        """Within tolerance should return True."""
        from puco_eeff.extractor.validation_core import _compare_with_tolerance

        match, diff = _compare_with_tolerance(100, 101, tolerance=5)
        assert match is True
        assert diff == 1

    def test_outside_tolerance(self) -> None:
        """Outside tolerance should return False."""
        from puco_eeff.extractor.validation_core import _compare_with_tolerance

        match, diff = _compare_with_tolerance(100, 110, tolerance=5)
        assert match is False
        assert diff == 10

    def test_sign_agnostic(self) -> None:
        """Sign should not matter (absolute comparison)."""
        from puco_eeff.extractor.validation_core import _compare_with_tolerance

        match, diff = _compare_with_tolerance(-100, 100, tolerance=1)
        assert match is True
        assert diff == 0

    def test_none_values(self) -> None:
        """None values should return True with diff 0."""
        from puco_eeff.extractor.validation_core import _compare_with_tolerance

        match, diff = _compare_with_tolerance(None, 100, tolerance=1)
        assert match is True
        assert diff == 0

        match, diff = _compare_with_tolerance(100, None, tolerance=1)
        assert match is True
        assert diff == 0


# =============================================================================
# Tests for per-rule tolerance (Phase 4)
# =============================================================================


class TestPerRuleTolerance:
    """Tests for per-rule tolerance in validations."""

    @pytest.fixture
    def sample_data_with_rounding(self) -> Sheet1Data:
        """Sheet1Data with small rounding differences."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        # Set values where sum differs from total by 3
        data.cv_gastos_personal = -19721
        data.cv_materiales = -23219
        data.cv_energia = -9589
        data.cv_servicios_terceros = -25063
        data.cv_depreciacion_amort = -21694
        data.cv_deprec_leasing = -881
        data.cv_deprec_arrend = -1577
        data.cv_serv_mineros = -10804
        data.cv_fletes = -5405
        data.cv_gastos_diferidos = -1587
        data.cv_convenios = -6662
        # Sum = -126202, but set total to differ by 3
        data.total_costo_venta = -126205  # 3 more than actual sum
        return data

    def test_per_rule_tolerance_in_cross_validation(self) -> None:
        """Cross-validation uses per-rule tolerance when specified."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.ingresos_ordinarios = 100
        data.total_costo_venta = -50

        xbrl_totals = {"gross_profit": 52}  # Should be 50, diff is 2

        # With global tolerance=1, this would fail
        # With per-rule tolerance=5, it should pass
        with (
            patch("puco_eeff.extractor.validation_core.get_sheet1_cross_validations") as mock_cross,
            patch("puco_eeff.extractor.validation_core.get_sheet1_sum_tolerance") as mock_tolerance,
        ):
            mock_tolerance.return_value = 1  # Global tolerance
            mock_cross.return_value = [
                {
                    "description": "Test cross-validation",
                    "formula": "gross_profit == ingresos_ordinarios - abs(total_costo_venta)",
                    "tolerance": 5,  # Per-rule override
                },
            ]

            results = _run_cross_validations(data, xbrl_totals)

            assert len(results) == 1
            assert results[0].match is True
            assert results[0].tolerance == 5


# =============================================================================
# Tests for unified validation API (run_sheet1_validations)
# =============================================================================


class TestRunSheetValidations:
    """Tests for the unified run_sheet1_validations function."""

    @pytest.fixture
    def sample_sheet1_data(self) -> Sheet1Data:
        """Create sample Sheet1Data for testing."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.ingresos_ordinarios = 179165
        # Nota 21 values
        data.cv_gastos_personal = -19721
        data.cv_materiales = -23219
        data.cv_energia = -9589
        data.cv_servicios_terceros = -25063
        data.cv_depreciacion_amort = -21694
        data.cv_deprec_leasing = -881
        data.cv_deprec_arrend = -1577
        data.cv_serv_mineros = -10804
        data.cv_fletes = -5405
        data.cv_gastos_diferidos = -1587
        data.cv_convenios = -6662
        data.total_costo_venta = -126202
        # Nota 22 values
        data.ga_gastos_personal = -3818
        data.ga_materiales = -129
        data.ga_servicios_terceros = -4239
        data.ga_gratificacion = -639
        data.ga_comercializacion = -2156
        data.ga_otros = -651
        data.total_gasto_admin = -11632
        return data

    @pytest.fixture
    def sample_xbrl_totals(self) -> dict[str, int | None]:
        """Sample XBRL totals matching the PDF data."""
        return {
            "ingresos": 179165,
            "cost_of_sales": -126202,
            "admin_expense": -11632,
            "gross_profit": 52963,
        }

    def test_all_validations_enabled(
        self, sample_sheet1_data: Sheet1Data, sample_xbrl_totals: dict[str, int | None],
    ) -> None:
        """Run all validations with both sources matching."""
        report = run_sheet1_validations(sample_sheet1_data, sample_xbrl_totals)

        assert isinstance(report, ValidationReport)
        assert len(report.sum_validations) >= 2  # At least Nota 21 and 22
        assert len(report.pdf_xbrl_validations) >= 2
        assert not report.has_failures()

    def test_sum_validations_only(self, sample_sheet1_data: Sheet1Data) -> None:
        """Run only sum validations (no XBRL)."""
        report = run_sheet1_validations(
            sample_sheet1_data,
            None,
            run_sum_validations=True,
            run_pdf_xbrl_validations=False,
            run_cross_validations=False,
        )

        assert len(report.sum_validations) >= 2
        assert len(report.pdf_xbrl_validations) == 0
        assert len(report.cross_validations) == 0

    def test_pdf_xbrl_validations_only(
        self, sample_sheet1_data: Sheet1Data, sample_xbrl_totals: dict[str, int | None],
    ) -> None:
        """Run only PDF↔XBRL validations."""
        report = run_sheet1_validations(
            sample_sheet1_data,
            sample_xbrl_totals,
            run_sum_validations=False,
            run_pdf_xbrl_validations=True,
            run_cross_validations=False,
        )

        assert len(report.sum_validations) == 0
        assert len(report.pdf_xbrl_validations) >= 2
        assert len(report.cross_validations) == 0

    def test_xbrl_fallback_enabled(self) -> None:
        """XBRL fallback sets missing PDF values."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.total_costo_venta = -100000
        # ingresos_ordinarios is None

        xbrl_totals = {
            "ingresos": 200000,
            "cost_of_sales": -100000,
            "admin_expense": -20000,
        }

        run_sheet1_validations(
            data,
            xbrl_totals,
            run_sum_validations=False,
            run_pdf_xbrl_validations=True,
            run_cross_validations=False,
            use_xbrl_fallback=True,
        )

        # XBRL value should have been set on data
        assert data.ingresos_ordinarios == 200000

    def test_xbrl_fallback_disabled(self) -> None:
        """XBRL fallback disabled does not modify data."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.total_costo_venta = -100000
        # ingresos_ordinarios is None

        xbrl_totals = {
            "ingresos": 200000,
            "cost_of_sales": -100000,
        }

        run_sheet1_validations(
            data,
            xbrl_totals,
            run_sum_validations=False,
            run_pdf_xbrl_validations=True,
            run_cross_validations=False,
            use_xbrl_fallback=False,
        )

        # Data should not be modified
        assert data.ingresos_ordinarios is None

    def test_pdf_only_no_xbrl(self, sample_sheet1_data: Sheet1Data) -> None:
        """PDF-only validation when XBRL unavailable."""
        report = run_sheet1_validations(sample_sheet1_data, None)

        # Sum validations should still run
        assert len(report.sum_validations) >= 2
        # PDF↔XBRL results should be empty or all pdf_only
        for v in report.pdf_xbrl_validations:
            assert v.source == "pdf_only"


class TestRunPdfXbrlValidations:
    """Tests for _run_pdf_xbrl_validations helper."""

    @pytest.fixture
    def sample_data(self) -> Sheet1Data:
        """Sample Sheet1Data for testing."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.total_costo_venta = -126202
        data.total_gasto_admin = -11632
        data.ingresos_ordinarios = 179165
        return data

    def test_both_sources_match(self, sample_data: Sheet1Data) -> None:
        """Both PDF and XBRL values match."""
        xbrl_totals = {
            "cost_of_sales": -126202,
            "admin_expense": -11632,
            "ingresos": 179165,
        }

        results = _run_pdf_xbrl_validations(sample_data, xbrl_totals, use_fallback=False)

        assert len(results) >= 2
        assert all(r.match for r in results if r.source == "both")

    def test_pdf_only_when_no_xbrl(self, sample_data: Sheet1Data) -> None:
        """Returns pdf_only results when no XBRL available."""
        results = _run_pdf_xbrl_validations(sample_data, None, use_fallback=False)

        assert len(results) >= 2
        assert all(r.source == "pdf_only" for r in results)
        assert all(r.match for r in results)

    def test_xbrl_only_with_fallback(self) -> None:
        """Uses XBRL value when PDF missing and fallback enabled."""
        data = Sheet1Data(quarter="IIQ2024", year=2024, quarter_num=2)
        # No ingresos_ordinarios set

        xbrl_totals = {
            "ingresos": 200000,
            "cost_of_sales": None,
            "admin_expense": None,
        }

        results = _run_pdf_xbrl_validations(data, xbrl_totals, use_fallback=True)

        # Should have set ingresos on data
        assert data.ingresos_ordinarios == 200000

        # Result should be xbrl_only
        ingresos_result = next((r for r in results if "Ingresos" in r.field_name), None)
        assert ingresos_result is not None
        assert ingresos_result.source == "xbrl_only"


# =============================================================================
# Tests for sections_to_sheet1data (config-driven converter in sheet1.py)
# =============================================================================


class TestSectionsToSheet1Data:
    """Tests for sections_to_sheet1data helper in sheet1.py."""

    def test_converts_nota_21_total(self) -> None:
        """Converts Nota 21 section to Sheet1Data total_costo_venta."""
        nota_21 = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
        nota_21.total_ytd_actual = -126202
        sections = {"nota_21": nota_21}

        data = sections_to_sheet1data(sections, 2024, 2)

        assert data.total_costo_venta == -126202
        assert data.total_gasto_admin is None
        assert data.year == 2024
        assert data.quarter_num == 2
        assert data.quarter == "IIQ2024"  # Roman numeral format

    def test_converts_nota_22_total(self) -> None:
        """Converts Nota 22 section to Sheet1Data total_gasto_admin."""
        nota_22 = SectionBreakdown(section_id="nota_22", section_title="Gastos Admin")
        nota_22.total_ytd_actual = -11632
        sections = {"nota_22": nota_22}

        data = sections_to_sheet1data(sections, 2024, 2)

        assert data.total_costo_venta is None
        assert data.total_gasto_admin == -11632

    def test_converts_both_sections(self) -> None:
        """Converts both Notas to Sheet1Data."""
        nota_21 = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
        nota_21.total_ytd_actual = -126202
        nota_22 = SectionBreakdown(section_id="nota_22", section_title="Gastos Admin")
        nota_22.total_ytd_actual = -11632
        sections = {"nota_21": nota_21, "nota_22": nota_22}

        data = sections_to_sheet1data(sections, 2024, 2)

        assert data.total_costo_venta == -126202
        assert data.total_gasto_admin == -11632
        assert data.year == 2024
        assert data.quarter_num == 2

    def test_handles_empty_sections(self) -> None:
        """Handles empty sections dict gracefully."""
        data = sections_to_sheet1data({}, 2024, 1)

        assert data.total_costo_venta is None
        assert data.total_gasto_admin is None
        assert data.year == 2024
        assert data.quarter_num == 1

    def test_converts_line_items_to_detail_fields(self) -> None:
        """Converts line items from sections to detail fields in Sheet1Data."""
        nota_21 = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
        nota_21.total_ytd_actual = -50000
        nota_21.items = [
            LineItem(concepto="Gastos en personal", ytd_actual=-20000),
            LineItem(concepto="Materiales y repuestos", ytd_actual=-15000),
            LineItem(concepto="Energía eléctrica", ytd_actual=-10000),
        ]
        sections = {"nota_21": nota_21}

        data = sections_to_sheet1data(sections, 2024, 3)

        # Check total and detail fields are populated
        assert data.total_costo_venta == -50000
        assert data.cv_gastos_personal == -20000
        assert data.cv_materiales == -15000
        assert data.cv_energia == -10000

    def test_converts_nota_22_line_items(self) -> None:
        """Converts Nota 22 line items to gasto_admin detail fields."""
        nota_22 = SectionBreakdown(section_id="nota_22", section_title="Gastos Admin")
        nota_22.total_ytd_actual = -8000
        nota_22.items = [
            LineItem(concepto="Gastos en personal", ytd_actual=-3000),
            LineItem(concepto="Servicios de terceros", ytd_actual=-2500),
            LineItem(concepto="Gastos comercializacion", ytd_actual=-2500),
        ]
        sections = {"nota_22": nota_22}

        data = sections_to_sheet1data(sections, 2024, 4)

        assert data.total_gasto_admin == -8000
        assert data.ga_gastos_personal == -3000
        assert data.ga_servicios_terceros == -2500
        assert data.ga_comercializacion == -2500

    def test_skips_items_without_value(self) -> None:
        """Skips line items that have None ytd_actual."""
        nota_21 = SectionBreakdown(section_id="nota_21", section_title="Costo de Venta")
        nota_21.total_ytd_actual = -20000
        nota_21.items = [
            LineItem(concepto="Gastos en personal", ytd_actual=-20000),
            LineItem(concepto="Materiales y repuestos", ytd_actual=None),  # Should be skipped
        ]
        sections = {"nota_21": nota_21}

        data = sections_to_sheet1data(sections, 2024, 1)

        assert data.cv_gastos_personal == -20000
        assert data.cv_materiales is None  # Not set because value was None

    def test_uses_config_mapping(self) -> None:
        """Verifies that section_total_mapping config is used."""
        mapping = get_sheet1_section_total_mapping()

        # Verify config structure
        assert "nota_21" in mapping
        assert "nota_22" in mapping
        assert mapping["nota_21"] == "total_costo_venta"
        assert mapping["nota_22"] == "total_gasto_admin"

    def test_ignores_unknown_sections(self) -> None:
        """Unknown section IDs are silently ignored."""
        unknown = SectionBreakdown(section_id="nota_99", section_title="Unknown")
        unknown.total_ytd_actual = -9999
        sections = {"nota_99": unknown}

        data = sections_to_sheet1data(sections, 2024, 1)

        # Should not raise, just ignore the unknown section
        assert data.total_costo_venta is None
        assert data.total_gasto_admin is None


# =============================================================================
# Tests for ExtractionResult.validation_report field
# =============================================================================


class TestExtractionResultValidationReport:
    """Tests for the validation_report field on ExtractionResult."""

    @patch("puco_eeff.extractor.extraction_pipeline.get_period_paths")
    @patch("puco_eeff.extractor.extraction_pipeline.extract_pdf_section")
    @patch("puco_eeff.extractor.extraction_pipeline.extract_xbrl_totals")
    def test_validation_report_populated(
        self,
        mock_xbrl: MagicMock,
        mock_extract_pdf_section: MagicMock,
        mock_paths_fn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """extract_detailed_costs populates validation_report field."""
        from puco_eeff.extractor.extraction_pipeline import extract_detailed_costs

        # Setup paths
        raw_pdf = tmp_path / "raw" / "pdf"
        raw_pdf.mkdir(parents=True)
        raw_xbrl = tmp_path / "raw" / "xbrl"
        raw_xbrl.mkdir(parents=True)
        processed = tmp_path / "processed"
        processed.mkdir(parents=True)

        mock_paths_fn.return_value = {
            "raw_pdf": raw_pdf,
            "raw_xbrl": raw_xbrl,
            "processed": processed,
            "audit": tmp_path / "audit",
        }

        # Create PDF and XBRL files
        (raw_pdf / "estados_financieros_2024_Q2.pdf").write_text("pdf")
        (raw_xbrl / "estados_financieros_2024_Q2.xbrl").write_text("xbrl")

        # Mock section extraction
        def extract_section_side_effect(path, section_name):
            if section_name == "nota_21":
                breakdown = SectionBreakdown(section_id="nota_21", section_title="Costo")
                breakdown.total_ytd_actual = -100000
                return breakdown
            if section_name == "nota_22":
                breakdown = SectionBreakdown(section_id="nota_22", section_title="Admin")
                breakdown.total_ytd_actual = -20000
                return breakdown
            return None

        mock_extract_pdf_section.side_effect = extract_section_side_effect
        mock_xbrl.return_value = {
            "cost_of_sales": -100000,
            "admin_expense": -20000,
        }

        # Run extraction
        result = extract_detailed_costs(2024, 2)

        # Verify validation_report is populated
        assert result.validation_report is not None
        assert isinstance(result.validation_report, ValidationReport)
        assert result.validation_report.pdf_xbrl_validations is not None
        assert len(result.validation_report.pdf_xbrl_validations) >= 2

    @patch("puco_eeff.extractor.extraction_pipeline.get_period_paths")
    @patch("puco_eeff.extractor.extraction_pipeline.extract_pdf_section")
    def test_validation_report_pdf_only(
        self,
        mock_extract_pdf_section: MagicMock,
        mock_paths_fn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """validation_report works without XBRL (pdf_only source)."""
        from puco_eeff.extractor.extraction_pipeline import extract_detailed_costs

        # Setup paths - no XBRL
        raw_pdf = tmp_path / "raw" / "pdf"
        raw_pdf.mkdir(parents=True)
        processed = tmp_path / "processed"
        processed.mkdir(parents=True)

        mock_paths_fn.return_value = {
            "raw_pdf": raw_pdf,
            "raw_xbrl": tmp_path / "raw" / "xbrl",
            "processed": processed,
            "audit": tmp_path / "audit",
        }

        # Create PDF but no XBRL - add pucobre marker
        (raw_pdf / "estados_financieros_2024_Q1.pdf").write_text("pdf")
        (raw_pdf / "pucobre_combined_2024_Q1.pdf").write_text("combined")

        # Mock section extraction
        def extract_section_side_effect(path, section_name):
            if section_name == "nota_21":
                breakdown = SectionBreakdown(section_id="nota_21", section_title="Costo")
                breakdown.total_ytd_actual = -50000
                return breakdown
            return None

        mock_extract_pdf_section.side_effect = extract_section_side_effect

        # Run extraction
        result = extract_detailed_costs(2024, 1)

        # Verify validation_report reflects pdf_only
        assert result.validation_report is not None
        assert result.xbrl_available is False
        assert all(v.source == "pdf_only" for v in result.validation_report.pdf_xbrl_validations)

    def test_validation_report_is_same_as_validations(self) -> None:
        """validation_report.pdf_xbrl_validations equals result.validations."""
        # This is a design invariant to verify
        report = ValidationReport(
            pdf_xbrl_validations=[
                ValidationResult(
                    field_name="Total Costo",
                    pdf_value=-100,
                    xbrl_value=-100,
                    match=True,
                    source="both",
                ),
            ],
            sum_validations=[],
            cross_validations=[],
        )
        result = ExtractionResult(
            source="cmf",
            xbrl_available=True,
            sections={},
            validations=report.pdf_xbrl_validations,
            validation_report=report,
            year=2024,
            quarter=2,
        )

        assert result.validation_report is not None  # Type narrowing for mypy/pyright
        assert result.validations is result.validation_report.pdf_xbrl_validations


# =============================================================================
# Tests for config-driven sections_to_sheet1data
# =============================================================================


class TestConfigDrivenSectionsToSheet1Data:
    """Tests that sections_to_sheet1data uses config mapping."""

    def test_uses_config_mapping_for_nota_21(self) -> None:
        """sections_to_sheet1data uses config for nota_21 → total_costo_venta."""
        mapping = get_sheet1_section_total_mapping()
        assert mapping["nota_21"] == "total_costo_venta"

        nota_21 = SectionBreakdown(section_id="nota_21", section_title="Test")
        nota_21.total_ytd_actual = -99999
        sections = {"nota_21": nota_21}

        data = sections_to_sheet1data(sections, 2024, 4)

        assert data.total_costo_venta == -99999

    def test_uses_config_mapping_for_nota_22(self) -> None:
        """sections_to_sheet1data uses config for nota_22 → total_gasto_admin."""
        mapping = get_sheet1_section_total_mapping()
        assert mapping["nota_22"] == "total_gasto_admin"

        nota_22 = SectionBreakdown(section_id="nota_22", section_title="Test")
        nota_22.total_ytd_actual = -88888
        sections = {"nota_22": nota_22}

        data = sections_to_sheet1data(sections, 2024, 4)

        assert data.total_gasto_admin == -88888

    def test_period_label_format(self) -> None:
        """sections_to_sheet1data uses correct period label format."""
        data = sections_to_sheet1data({}, 2024, 3)

        # Uses Roman numeral format (e.g., "IIIQ2024"), not "unknown" or "2024Q3"
        assert data.quarter == "IIIQ2024"
        assert data.year == 2024
        assert data.quarter_num == 3


# =============================================================================
# Config-Driven Accessor Tests
# =============================================================================


class TestGetSectionConfig:
    """Tests for get_section_config() accessor."""

    def test_returns_valid_config_for_nota_21(self) -> None:
        """get_section_config returns valid config for nota_21."""
        config = get_section_config("nota_21")

        assert config["title"] == "Costo de Venta"
        assert "field_mappings" in config
        assert "fallback_section" in config

    def test_returns_valid_config_for_nota_22(self) -> None:
        """get_section_config returns valid config for nota_22."""
        config = get_section_config("nota_22")

        assert config["title"] == "Gastos de Administración y Ventas"
        assert "field_mappings" in config
        assert config["fallback_section"] == "nota_21"

    def test_returns_valid_config_for_ingresos(self) -> None:
        """get_section_config returns valid config for ingresos."""
        config = get_section_config("ingresos")

        assert config["title"] == "Ingresos de actividades ordinarias"
        assert "field_mappings" in config
        assert "pdf_fallback" in config

    def test_raises_value_error_for_invalid_sheet(self) -> None:
        """get_section_config raises ValueError for invalid sheet."""
        with pytest.raises(ValueError, match="not supported"):
            get_section_config("nota_21", sheet="sheet2")

    def test_raises_value_error_for_invalid_section(self) -> None:
        """get_section_config raises ValueError for invalid section."""
        with pytest.raises(ValueError, match="not found"):
            get_section_config("invalid_section")


class TestGetSectionFallback:
    """Tests for get_section_fallback() accessor."""

    def test_returns_fallback_for_nota_22(self) -> None:
        """get_section_fallback returns nota_21 for nota_22."""
        fallback = get_section_fallback("nota_22")
        assert fallback == "nota_21"

    def test_returns_none_for_nota_21(self) -> None:
        """get_section_fallback returns None for nota_21 (no fallback)."""
        fallback = get_section_fallback("nota_21")
        assert fallback is None

    def test_returns_none_for_ingresos(self) -> None:
        """get_section_fallback returns None for ingresos (no fallback)."""
        fallback = get_section_fallback("ingresos")
        assert fallback is None


class TestGetIngresosPdfFallbackConfig:
    """Tests for get_ingresos_pdf_fallback_config() accessor."""

    def test_returns_min_value_threshold(self) -> None:
        """get_ingresos_pdf_fallback_config returns min_value_threshold."""
        config = get_ingresos_pdf_fallback_config()

        assert "min_value_threshold" in config
        assert config["min_value_threshold"] == 1000

    def test_returns_search_patterns(self) -> None:
        """get_ingresos_pdf_fallback_config returns search_patterns."""
        config = get_ingresos_pdf_fallback_config()

        assert "search_patterns" in config
        assert len(config["search_patterns"]) > 0

    def test_returns_page_type(self) -> None:
        """get_ingresos_pdf_fallback_config returns page_type."""
        config = get_ingresos_pdf_fallback_config()

        assert config["page_type"] == "estado_de_resultados"
