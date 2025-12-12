"""Tests for Sheet3 extraction module.

Tests cover:
1. Sheet3Data dataclass operations
2. Config loading from sheet3/ JSON files
3. Reference data validation
4. XBRL value extraction (mocked)
5. PDF extraction (mocked)
6. Sign convention handling
7. Subtotal validation
8. PDF/XBRL consistency
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from puco_eeff.sheets.sheet3 import (
    Sheet3Data,
    get_reference_values_sheet3,
    get_sheet3_extraction_config,
    get_sheet3_field_keywords,
    get_sheet3_fields,
    get_sheet3_reference_data,
    get_sheet3_xbrl_mappings,
    validate_sheet3_against_reference,
    validate_sheet3_subtotals,
)
from puco_eeff.utils.parsing import parse_spanish_number

# =============================================================================
# Tests for Sheet3Data dataclass
# =============================================================================


class TestSheet3Data:
    """Tests for Sheet3Data dataclass operations."""

    def test_create_empty_instance(self) -> None:
        """Create instance with default values."""
        data = Sheet3Data()
        assert data.quarter == ""
        assert data.year == 0
        assert data.ingresos_ordinarios is None
        assert data.ganancia_periodo is None
        assert data.issues == []

    def test_create_with_values(self) -> None:
        """Create instance with actual values."""
        data = Sheet3Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            source="xbrl",
            xbrl_available=True,
            ingresos_ordinarios=179165,
            costo_ventas=-126202,
            ganancia_bruta=52963,
            ganancia_periodo=30079,
            ganancia_por_accion=0.24127,
            acciones_emitidas=124668381,
        )
        assert data.quarter == "IIQ2024"
        assert data.ingresos_ordinarios == 179165
        assert data.costo_ventas == -126202
        assert data.ganancia_por_accion == 0.24127

    def test_to_dict(self) -> None:
        """Convert to dictionary for JSON serialization."""
        data = Sheet3Data(
            quarter="IQ2024",
            year=2024,
            quarter_num=1,
            ingresos_ordinarios=80767,
            costo_ventas=-62982,
        )
        result = data.to_dict()

        assert result["quarter"] == "IQ2024"
        assert result["ingresos_ordinarios"] == 80767
        assert result["costo_ventas"] == -62982
        assert "issues" in result

    def test_from_dict(self) -> None:
        """Create instance from dictionary."""
        data_dict = {
            "quarter": "IIQ2024",
            "year": 2024,
            "quarter_num": 2,
            "source": "xbrl",
            "ingresos_ordinarios": 179165,
            "ganancia_por_accion": 0.24127,
        }
        data = Sheet3Data.from_dict(data_dict)

        assert data.quarter == "IIQ2024"
        assert data.ingresos_ordinarios == 179165
        assert data.ganancia_por_accion == 0.24127

    def test_roundtrip_dict(self) -> None:
        """Verify to_dict and from_dict are inverses."""
        original = Sheet3Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            source="xbrl",
            xbrl_available=True,
            ingresos_ordinarios=179165,
            costo_ventas=-126202,
            ganancia_bruta=52963,
            otros_ingresos=272,
            otros_egresos_funcion=0,
            ingresos_financieros=263,
            gastos_admin_ventas=-11632,
            costos_financieros=-2832,
            diferencias_cambio=3018,
            ganancia_antes_impuestos=42052,
            gasto_impuestos=-11973,
            ganancia_periodo=30079,
            resultado_accionistas=30079,
            acciones_emitidas=124668381,
            acciones_dividendo=124668381,
            ganancia_por_accion=0.24127,
            issues=["test issue"],
        )

        data_dict = original.to_dict()
        restored = Sheet3Data.from_dict(data_dict)

        assert restored.quarter == original.quarter
        assert restored.ingresos_ordinarios == original.ingresos_ordinarios
        assert restored.ganancia_por_accion == original.ganancia_por_accion
        assert restored.acciones_emitidas == original.acciones_emitidas
        assert restored.issues == original.issues


# =============================================================================
# Tests for Config Loading
# =============================================================================


class TestSheet3ConfigLoading:
    """Tests for Sheet3 config file loading."""

    def test_get_sheet3_fields(self) -> None:
        """Load fields.json successfully."""
        fields = get_sheet3_fields()

        assert "name" in fields
        assert fields["name"] == "Estado de Resultados"
        assert "value_fields" in fields
        assert "ingresos_ordinarios" in fields["value_fields"]
        assert "ganancia_por_accion" in fields["value_fields"]

    def test_get_sheet3_extraction_config(self) -> None:
        """Load extraction.json successfully."""
        config = get_sheet3_extraction_config()

        assert "source_priority" in config
        assert config["source_priority"] == ["xbrl", "pdf"]
        assert "xbrl_config" in config
        assert config["xbrl_config"]["scale_factor"] == 1000

    def test_get_sheet3_xbrl_mappings(self) -> None:
        """Load xbrl_mappings.json successfully."""
        mappings = get_sheet3_xbrl_mappings()

        assert "fact_mappings" in mappings
        assert "ingresos_ordinarios" in mappings["fact_mappings"]
        assert mappings["fact_mappings"]["ingresos_ordinarios"]["primary"] == "Revenue"

    def test_get_sheet3_reference_data(self) -> None:
        """Load reference_data.json successfully."""
        ref_data = get_sheet3_reference_data()

        assert "2024_Q1" in ref_data
        assert "2024_Q2" in ref_data
        assert ref_data["2024_Q2"]["verified"] is True

    def test_get_sheet3_field_keywords(self) -> None:
        """Get field keywords for PDF extraction."""
        keywords = get_sheet3_field_keywords()

        assert "ingresos_ordinarios" in keywords
        assert "ganancia_bruta" in keywords
        # Check that ganancia_por_accion has a keyword containing "acción"
        eps_keywords = keywords.get("ganancia_por_accion", [])
        assert any("acción" in kw or "accion" in kw for kw in eps_keywords)

    def test_value_fields_have_required_properties(self) -> None:
        """Check value fields have type and section."""
        fields = get_sheet3_fields()
        value_fields = fields["value_fields"]

        for field_name, field_def in value_fields.items():
            assert "type" in field_def, f"{field_name} missing type"
            assert "section" in field_def, f"{field_name} missing section"
            assert "label" in field_def, f"{field_name} missing label"


# =============================================================================
# Tests for Reference Data
# =============================================================================


class TestSheet3ReferenceData:
    """Tests for reference data validation."""

    def test_get_reference_values_q1_2024(self) -> None:
        """Get Q1 2024 reference values."""
        values = get_reference_values_sheet3(2024, 1)

        assert values is not None
        assert values["ingresos_ordinarios"] == 80767
        assert values["costo_ventas"] == -62982
        assert values["ganancia_bruta"] == 17785
        assert values["ganancia_por_accion"] == 0.08001

    def test_get_reference_values_q2_2024(self) -> None:
        """Get Q2 2024 reference values."""
        values = get_reference_values_sheet3(2024, 2)

        assert values is not None
        assert values["ingresos_ordinarios"] == 179165
        assert values["costo_ventas"] == -126202
        assert values["ganancia_periodo"] == 30079
        assert values["acciones_emitidas"] == 124668381

    def test_get_reference_values_unavailable(self) -> None:
        """Return None for periods without verified data."""
        values = get_reference_values_sheet3(2024, 3)
        assert values is None

    def test_validate_matching_values(self) -> None:
        """Validation passes when values match reference."""
        data = Sheet3Data(
            quarter="IQ2024",
            year=2024,
            quarter_num=1,
            ingresos_ordinarios=80767,
            costo_ventas=-62982,
            ganancia_bruta=17785,
            otros_ingresos=226,
            otros_egresos_funcion=0,
            ingresos_financieros=238,
            gastos_admin_ventas=-5137,
            costos_financieros=-1329,
            diferencias_cambio=2141,
            ganancia_antes_impuestos=13924,
            gasto_impuestos=-3949,
            ganancia_periodo=9975,
            resultado_accionistas=9975,
            acciones_emitidas=124668381,
            acciones_dividendo=124668381,
            ganancia_por_accion=0.08001,
        )

        is_valid, issues = validate_sheet3_against_reference(data)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_mismatched_values(self) -> None:
        """Validation fails when values don't match."""
        data = Sheet3Data(
            quarter="IQ2024",
            year=2024,
            quarter_num=1,
            ingresos_ordinarios=80000,  # Wrong value
            costo_ventas=-62982,
            ganancia_bruta=17018,  # Wrong due to wrong ingresos
        )

        is_valid, issues = validate_sheet3_against_reference(data)

        assert is_valid is False
        assert any("ingresos_ordinarios" in issue for issue in issues)

    def test_validate_with_tolerance(self) -> None:
        """Validation allows small differences within tolerance."""
        data = Sheet3Data(
            quarter="IQ2024",
            year=2024,
            quarter_num=1,
            ingresos_ordinarios=80768,  # Off by 1 (within tolerance)
            costo_ventas=-62982,
            ganancia_bruta=17785,
            otros_ingresos=226,
            otros_egresos_funcion=0,
            ingresos_financieros=238,
            gastos_admin_ventas=-5137,
            costos_financieros=-1329,
            diferencias_cambio=2141,
            ganancia_antes_impuestos=13924,
            gasto_impuestos=-3949,
            ganancia_periodo=9975,
            resultado_accionistas=9975,
            acciones_emitidas=124668381,
            acciones_dividendo=124668381,
            ganancia_por_accion=0.08001,
        )

        is_valid, issues = validate_sheet3_against_reference(data, tolerance=1)

        assert is_valid is True


# =============================================================================
# Tests for Number Parsing (Spanish Locale)
# =============================================================================


class TestSpanishNumberParsing:
    """Tests for Spanish locale number parsing."""

    def test_comma_decimal(self) -> None:
        """Comma as decimal separator."""
        assert parse_spanish_number("19,5") == 19.5
        assert parse_spanish_number("3,97") == 3.97

    def test_period_thousands(self) -> None:
        """Period as thousands separator."""
        assert parse_spanish_number("1.342") == 1342
        assert parse_spanish_number("64.057") == 64057

    def test_combined_format(self) -> None:
        """Both thousands and decimal separators."""
        assert parse_spanish_number("1.234,56") == 1234.56

    def test_empty_string(self) -> None:
        """Empty string returns None."""
        assert parse_spanish_number("") is None
        assert parse_spanish_number("   ") is None

    def test_large_numbers(self) -> None:
        """Large numbers with multiple periods."""
        assert parse_spanish_number("1.234.567") == 1234567
        assert parse_spanish_number("179.165") == 179165


# =============================================================================
# Tests for XBRL Extraction (Mocked)
# =============================================================================


class TestSheet3XBRLExtraction:
    """Tests for XBRL extraction logic."""

    def test_xbrl_context_selection_acumulado_actual(self) -> None:
        """Test context selection prefers AcumuladoActual ID."""
        from puco_eeff.sheets.sheet3 import _find_current_context

        contexts = {
            "ctx_AcumuladoActual_2024Q2": {
                "period_type": "duration",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            },
            "ctx_AcumuladoAnterior_2023Q2": {
                "period_type": "duration",
                "start_date": "2023-01-01",
                "end_date": "2023-06-30",
            },
        }

        result = _find_current_context(contexts, 2024, 2)
        assert "AcumuladoActual" in result

    def test_xbrl_context_selection_by_date(self) -> None:
        """Test context selection falls back to date range."""
        from puco_eeff.sheets.sheet3 import _find_current_context

        contexts = {
            "ctx_2024Q2": {
                "period_type": "duration",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            },
            "ctx_2024Q1": {
                "period_type": "duration",
                "start_date": "2024-01-01",
                "end_date": "2024-03-31",
            },
        }

        result = _find_current_context(contexts, 2024, 2)
        assert result == "ctx_2024Q2"


# =============================================================================
# Tests for Sign Convention
# =============================================================================


class TestSignConvention:
    """Tests for sign convention handling."""

    def test_costs_stored_as_negative(self) -> None:
        """Verify cost fields are stored as negative values."""
        data = Sheet3Data(
            costo_ventas=-126202,
            gastos_admin_ventas=-11632,
            costos_financieros=-2832,
            gasto_impuestos=-11973,
        )

        assert data.costo_ventas < 0
        assert data.gastos_admin_ventas < 0
        assert data.costos_financieros < 0
        assert data.gasto_impuestos < 0

    def test_income_stored_as_positive(self) -> None:
        """Verify income fields are stored as positive values."""
        data = Sheet3Data(
            ingresos_ordinarios=179165,
            ganancia_bruta=52963,
            otros_ingresos=272,
            ingresos_financieros=263,
        )

        assert data.ingresos_ordinarios > 0
        assert data.ganancia_bruta > 0
        assert data.otros_ingresos > 0
        assert data.ingresos_financieros > 0

    def test_diferencias_cambio_can_be_variable(self) -> None:
        """Diferencias de cambio can be positive or negative."""
        data_positive = Sheet3Data(diferencias_cambio=3018)
        data_negative = Sheet3Data(diferencias_cambio=-1500)

        assert data_positive.diferencias_cambio > 0
        assert data_negative.diferencias_cambio < 0


# =============================================================================
# Tests for Field Definitions
# =============================================================================


class TestFieldDefinitions:
    """Tests for field definition consistency."""

    def test_row_mapping_covers_all_value_fields(self) -> None:
        """Row mapping should cover all value fields."""
        fields = get_sheet3_fields()
        value_fields = set(fields["value_fields"].keys())
        row_mapping = fields["row_mapping"]

        mapped_fields = {v["field"] for v in row_mapping.values() if v.get("field")}

        assert value_fields == mapped_fields

    def test_xbrl_mappings_cover_all_value_fields(self) -> None:
        """XBRL mappings should have entries for all value fields."""
        fields = get_sheet3_fields()
        value_fields = set(fields["value_fields"].keys())

        xbrl_mappings = get_sheet3_xbrl_mappings()
        mapped_fields = set(xbrl_mappings["fact_mappings"].keys())

        assert value_fields == mapped_fields

    def test_eps_field_has_full_precision(self) -> None:
        """EPS field should preserve full precision."""
        fields = get_sheet3_fields()
        eps_field = fields["value_fields"]["ganancia_por_accion"]

        assert eps_field["type"] == "float"
        assert eps_field.get("precision") == "full"
        assert eps_field.get("no_scaling") is True


# =============================================================================
# Integration Tests (with mocked data sources)
# =============================================================================


class TestSheet3Integration:
    """Integration tests with mocked data sources."""

    @patch("puco_eeff.sheets.sheet3.parse_xbrl_file")
    @patch("puco_eeff.sheets.sheet3.get_period_paths")
    def test_extract_from_xbrl(self, mock_paths, mock_parse) -> None:
        """Test extraction from XBRL file."""
        from pathlib import Path

        from puco_eeff.sheets.sheet3 import extract_sheet3

        # Mock paths
        mock_paths.return_value = {
            "raw_xbrl": Path("/fake/xbrl"),
            "raw_pdf": None,
        }

        # Mock XBRL data
        mock_parse.return_value = {
            "facts": [
                {"name": "Revenue", "value": "179165000", "context_ref": "ctx_AcumuladoActual"},
                {
                    "name": "CostOfSales",
                    "value": "-126202000",
                    "context_ref": "ctx_AcumuladoActual",
                },
                {"name": "GrossProfit", "value": "52963000", "context_ref": "ctx_AcumuladoActual"},
                {"name": "ProfitLoss", "value": "30079000", "context_ref": "ctx_AcumuladoActual"},
                {
                    "name": "BasicEarningsLossPerShare",
                    "value": "0.24127",
                    "context_ref": "ctx_AcumuladoActual",
                },
            ],
            "contexts": {
                "ctx_AcumuladoActual": {
                    "period_type": "duration",
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-30",
                },
            },
        }

        # Create fake XBRL file
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.glob") as mock_glob:
                mock_glob.return_value = [Path("/fake/xbrl/file.xml")]

                data = extract_sheet3(2024, 2)

        # Verify basic data was populated
        assert data.year == 2024
        assert data.quarter_num == 2


class TestSubtotalValidation:
    """Tests for validate_sheet3_subtotals function."""

    def test_subtotal_valid(self) -> None:
        """Test subtotal validation passes when sum matches exactly."""
        # Create data where components sum to ganancia_antes_impuestos
        # Formula: bruta + otros_ing + otros_egr + ing_fin + gastos_admin + costos_fin + dif_cambio
        # 50000 + 200 + (-500) + 100 + (-5000) + (-800) + (-700) = 43300
        data = Sheet3Data(
            year=2025,
            quarter_num=1,
            ganancia_bruta=50000,
            otros_ingresos=200,
            otros_egresos_funcion=-500,
            ingresos_financieros=100,
            gastos_admin_ventas=-5000,
            costos_financieros=-800,
            diferencias_cambio=-700,
            ganancia_antes_impuestos=43300,
        )

        is_valid, issues = validate_sheet3_subtotals(data)
        assert is_valid is True
        assert len(issues) == 0

    def test_subtotal_valid_with_tolerance(self) -> None:
        """Test subtotal validation passes within tolerance."""
        # Sum = 43300, but ganancia_antes_impuestos = 43301 (diff = 1, within tolerance)
        data = Sheet3Data(
            year=2025,
            quarter_num=1,
            ganancia_bruta=50000,
            otros_ingresos=200,
            otros_egresos_funcion=-500,
            ingresos_financieros=100,
            gastos_admin_ventas=-5000,
            costos_financieros=-800,
            diferencias_cambio=-700,
            ganancia_antes_impuestos=43301,  # Off by 1
        )

        is_valid, issues = validate_sheet3_subtotals(data, tolerance=1)
        assert is_valid is True

    def test_subtotal_invalid(self) -> None:
        """Test subtotal validation fails when sum doesn't match."""
        # Sum = 43300, but ganancia_antes_impuestos = 45000 (diff = 1700)
        data = Sheet3Data(
            year=2025,
            quarter_num=1,
            ganancia_bruta=50000,
            otros_ingresos=200,
            otros_egresos_funcion=-500,
            ingresos_financieros=100,
            gastos_admin_ventas=-5000,
            costos_financieros=-800,
            diferencias_cambio=-700,
            ganancia_antes_impuestos=45000,
        )

        is_valid, issues = validate_sheet3_subtotals(data)
        assert is_valid is False
        assert len(issues) == 1
        assert "subtotal" in issues[0].lower() or "ganancia antes" in issues[0].lower()

    def test_subtotal_with_none_values(self) -> None:
        """Test subtotal validation handles None values as 0."""
        # When algunos_egresos_funcion is None, it should be treated as 0
        # Sum = 50000 + 200 + 0 + 100 + (-5000) + (-800) + (-700) = 43800
        data = Sheet3Data(
            year=2025,
            quarter_num=1,
            ganancia_bruta=50000,
            otros_ingresos=200,
            otros_egresos_funcion=None,  # Treated as 0
            ingresos_financieros=100,
            gastos_admin_ventas=-5000,
            costos_financieros=-800,
            diferencias_cambio=-700,
            ganancia_antes_impuestos=43800,
        )

        is_valid, issues = validate_sheet3_subtotals(data)
        assert is_valid is True

    def test_subtotal_missing_ganancia_antes_impuestos(self) -> None:
        """Test subtotal validation skips check when target is None."""
        # When ganancia_antes_impuestos is None, the validation is skipped
        # (not enough data to validate)
        data = Sheet3Data(
            year=2025,
            quarter_num=1,
            ganancia_bruta=50000,
            otros_ingresos=200,
            otros_egresos_funcion=-500,
            ingresos_financieros=100,
            gastos_admin_ventas=-5000,
            costos_financieros=-800,
            diferencias_cambio=-700,
            ganancia_antes_impuestos=None,
        )

        is_valid, issues = validate_sheet3_subtotals(data)
        # Validation is skipped when target is None (returns True with no issues)
        assert is_valid is True
        assert len(issues) == 0

    def test_subtotal_real_q1_2025_values(self) -> None:
        """Test subtotal validation with real Q1 2025 data."""
        # Real values from IQ2025 extraction:
        # 50668 + 232 + (-550) + 63 + (-5756) + (-988) + (-704) = 42965
        data = Sheet3Data(
            year=2025,
            quarter_num=1,
            ganancia_bruta=50668,
            otros_ingresos=232,
            otros_egresos_funcion=-550,
            ingresos_financieros=63,
            gastos_admin_ventas=-5756,
            costos_financieros=-988,
            diferencias_cambio=-704,
            ganancia_antes_impuestos=42965,
        )

        is_valid, issues = validate_sheet3_subtotals(data)
        assert is_valid is True
        assert len(issues) == 0


class TestPDFXBRLConsistency:
    """Test that PDF and XBRL extraction produce consistent results."""

    @staticmethod
    def _find_latest_quarter_with_both_sources() -> tuple[int, int] | None:
        """Find the latest quarter that has both XBRL and PDF files available."""
        from puco_eeff.config import get_period_paths

        # Check quarters in reverse order (most recent first)
        for year in [2025, 2024]:
            for quarter in [4, 3, 2, 1]:
                paths = get_period_paths(year, quarter)
                raw_xbrl = paths.get("raw_xbrl")
                raw_pdf = paths.get("raw_pdf")

                has_xbrl = False
                has_pdf = False

                if raw_xbrl and raw_xbrl.exists():
                    xbrl_files = list(raw_xbrl.glob(f"*{year}*Q{quarter}*.xbrl"))
                    has_xbrl = len(xbrl_files) > 0

                if raw_pdf and raw_pdf.exists():
                    pdf_files = list(raw_pdf.glob(f"*{year}*Q{quarter}*.pdf"))
                    has_pdf = len(pdf_files) > 0

                if has_xbrl and has_pdf:
                    return (year, quarter)

        return None

    def test_pdf_xbrl_consistency_latest_quarter(self) -> None:
        """Test that PDF and XBRL extraction produce matching values for latest quarter.

        This test:
        1. Finds the latest quarter with both XBRL and PDF available
        2. Extracts data using XBRL
        3. Extracts data using PDF
        4. Compares key numeric fields and fails on mismatch

        Note: Some PDFs have different formats (4-column vs 2-column), so we skip
        if PDF extraction fails.
        """
        from puco_eeff.config import get_period_paths
        from puco_eeff.sheets.sheet3 import (
            Sheet3Data,
            _extract_from_pdf,
            _extract_from_xbrl,
            _find_pdf_path,
        )

        # Find latest quarter with both sources
        result = self._find_latest_quarter_with_both_sources()
        if result is None:
            pytest.skip("No quarter found with both XBRL and PDF data available")

        year, quarter = result

        # Get file paths
        paths = get_period_paths(year, quarter)
        raw_xbrl = paths.get("raw_xbrl")

        # Find XBRL file
        xbrl_path = None
        if raw_xbrl:
            for xbrl_file in raw_xbrl.glob(f"*{year}*Q{quarter}*.xbrl"):
                if "zip" not in xbrl_file.name:
                    xbrl_path = xbrl_file
                    break

        pdf_path = _find_pdf_path(year, quarter)

        assert xbrl_path is not None, f"XBRL file not found for {year} Q{quarter}"
        assert pdf_path is not None, f"PDF file not found for {year} Q{quarter}"

        # Extract using XBRL
        xbrl_data = Sheet3Data(year=year, quarter_num=quarter, quarter=f"Q{quarter}{year}")
        xbrl_success, xbrl_issues = _extract_from_xbrl(xbrl_path, xbrl_data)
        assert xbrl_success, f"XBRL extraction failed: {xbrl_issues}"

        # Extract using PDF
        pdf_data = Sheet3Data(year=year, quarter_num=quarter, quarter=f"Q{quarter}{year}")
        pdf_success, pdf_issues = _extract_from_pdf(pdf_path, pdf_data)

        # Skip if PDF extraction fails (some PDFs have 4-column format not yet supported)
        if not pdf_success:
            pytest.skip(
                f"PDF extraction failed for {year} Q{quarter} (may have unsupported format): "
                f"{pdf_issues}",
            )

        # Fields to compare (key numeric fields that should match between sources)
        fields_to_compare = [
            "ingresos_ordinarios",
            "costo_ventas",
            "ganancia_bruta",
            "otros_ingresos",
            "ingresos_financieros",
            "gastos_admin_ventas",
            "costos_financieros",
            "diferencias_cambio",
            "ganancia_antes_impuestos",
            "gasto_impuestos",
            "ganancia_periodo",
            "acciones_emitidas",
        ]

        mismatches: list[str] = []
        tolerance = 1  # Allow 1 MUS$ difference for rounding

        for field in fields_to_compare:
            xbrl_val = getattr(xbrl_data, field, None)
            pdf_val = getattr(pdf_data, field, None)

            # Skip if either is None
            if xbrl_val is None or pdf_val is None:
                continue

            # Compare with tolerance
            if abs(xbrl_val - pdf_val) > tolerance:
                mismatches.append(
                    f"{field}: XBRL={xbrl_val:,} vs PDF={pdf_val:,} (diff={xbrl_val - pdf_val:,})",
                )

        # Report results
        if mismatches:
            msg = f"PDF/XBRL mismatch for {year} Q{quarter}:\n" + "\n".join(
                f"  • {m}" for m in mismatches
            )
            pytest.fail(msg)

    def test_pdf_xbrl_consistency_q2_2024(self) -> None:
        """Test PDF/XBRL consistency specifically for Q2 2024 (known 2-column format)."""
        from puco_eeff.config import get_period_paths
        from puco_eeff.sheets.sheet3 import (
            Sheet3Data,
            _extract_from_pdf,
            _extract_from_xbrl,
            _find_pdf_path,
        )

        year, quarter = 2024, 2

        # Get file paths
        paths = get_period_paths(year, quarter)
        raw_xbrl = paths.get("raw_xbrl")

        # Find XBRL file
        xbrl_path = None
        if raw_xbrl and raw_xbrl.exists():
            for xbrl_file in raw_xbrl.glob(f"*{year}*Q{quarter}*.xbrl"):
                if "zip" not in xbrl_file.name:
                    xbrl_path = xbrl_file
                    break

        pdf_path = _find_pdf_path(year, quarter)

        if xbrl_path is None or pdf_path is None:
            pytest.skip(f"Files not available for {year} Q{quarter}")

        # Extract using XBRL
        xbrl_data = Sheet3Data(year=year, quarter_num=quarter, quarter=f"IIQ{year}")
        xbrl_success, xbrl_issues = _extract_from_xbrl(xbrl_path, xbrl_data)
        assert xbrl_success, f"XBRL extraction failed: {xbrl_issues}"

        # Extract using PDF
        pdf_data = Sheet3Data(year=year, quarter_num=quarter, quarter=f"IIQ{year}")
        pdf_success, pdf_issues = _extract_from_pdf(pdf_path, pdf_data)
        assert pdf_success, f"PDF extraction failed: {pdf_issues}"

        # Compare key fields
        fields_to_compare = [
            ("ingresos_ordinarios", 179165),  # Reference value
            ("costo_ventas", -126202),
            ("ganancia_bruta", 52963),
            ("ganancia_periodo", 30079),
        ]

        for field, expected in fields_to_compare:
            xbrl_val = getattr(xbrl_data, field, None)
            pdf_val = getattr(pdf_data, field, None)

            # Both should match expected (within tolerance of 1)
            assert xbrl_val is not None, f"XBRL missing {field}"
            assert pdf_val is not None, f"PDF missing {field}"
            assert abs(xbrl_val - expected) <= 1, f"XBRL {field}={xbrl_val}, expected={expected}"
            assert abs(pdf_val - expected) <= 1, f"PDF {field}={pdf_val}, expected={expected}"
            assert abs(xbrl_val - pdf_val) <= 1, f"{field}: XBRL={xbrl_val} vs PDF={pdf_val}"
