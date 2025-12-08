"""Tests for formatter module with mock config."""

import pytest

from puco_eeff.transformer.formatter import (
    ValidationResult,
    format_validation_report,
    get_field_labels,
    get_standard_structure,
    map_to_structure,
    validate_against_reference,
    validate_balance_sheet,
    validate_costo_venta_total,
    validate_gasto_admin_total,
)


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return {
        "sheets": {
            "sheet1": {
                "row_mapping": {
                    "1": {"field": "ingresos_ordinarios", "label": "Ingresos", "section": None},
                    "2": {"field": None, "label": "", "section": None},
                    "3": {"field": "cv_gastos_personal", "label": "Gastos en personal", "section": "costo_venta"},
                    "4": {"field": "cv_materiales", "label": "Materiales", "section": "costo_venta"},
                    "5": {"field": "total_costo_venta", "label": "Total CV", "section": "costo_venta_total"},
                },
                "extraction_labels": {
                    "field_labels": {
                        "ingresos_ordinarios": "Ingresos de actividades ordinarias",
                        "cv_gastos_personal": "Gastos en personal",
                        "cv_materiales": "Materiales y repuestos",
                        "total_costo_venta": "Total Costo de Venta",
                    },
                },
                "data": {
                    "IIQ2024": {
                        "ingresos_ordinarios": 179165,
                        "cv_gastos_personal": -19721,
                        "cv_materiales": -23219,
                        "cv_energia": -9589,
                        "cv_servicios_terceros": -25063,
                        "cv_depreciacion_amort": -21694,
                        "cv_deprec_leasing": -881,
                        "cv_deprec_arrend": -1577,
                        "cv_serv_mineros": -10804,
                        "cv_fletes": -5405,
                        "cv_gastos_diferidos": -1587,
                        "cv_convenios": -6662,
                        "total_costo_venta": -126202,
                        "ga_gastos_personal": -3818,
                        "ga_materiales": -129,
                        "ga_servicios_terceros": -4239,
                        "ga_gratificacion": -639,
                        "ga_comercializacion": -2156,
                        "ga_otros": -651,
                        "total_gasto_admin": -11632,
                        "source": "cmf",
                        "xbrl_available": True,
                    },
                },
            },
        },
    }


class TestGetStandardStructure:
    """Tests for get_standard_structure."""

    def test_returns_ordered_list(self, mock_config) -> None:
        """Structure should be ordered by row number."""
        structure = get_standard_structure("sheet1", mock_config)

        assert len(structure) == 5
        assert structure[0]["row"] == 1
        assert structure[1]["row"] == 2
        assert structure[4]["row"] == 5

    def test_includes_field_and_label(self, mock_config) -> None:
        """Each row should have field, label, section."""
        structure = get_standard_structure("sheet1", mock_config)

        assert structure[0]["field"] == "ingresos_ordinarios"
        assert structure[0]["label"] == "Ingresos"
        assert structure[2]["section"] == "costo_venta"


class TestGetFieldLabels:
    """Tests for get_field_labels."""

    def test_returns_labels_dict(self, mock_config) -> None:
        """Should return field to label mapping."""
        labels = get_field_labels("sheet1", mock_config)

        assert labels["ingresos_ordinarios"] == "Ingresos de actividades ordinarias"
        assert labels["total_costo_venta"] == "Total Costo de Venta"


class TestMapToStructure:
    """Tests for map_to_structure."""

    def test_maps_data_to_rows(self, mock_config) -> None:
        """Should map data values to row structure."""
        data = {
            "ingresos_ordinarios": 100000,
            "cv_gastos_personal": -5000,
        }

        rows = map_to_structure(data, "sheet1", mock_config)

        assert len(rows) == 5
        assert rows[0]["valor"] == 100000
        assert rows[0]["concepto"] == "Ingresos"
        assert rows[2]["valor"] == -5000

    def test_missing_values_are_none(self, mock_config) -> None:
        """Missing data should result in None values."""
        data = {"ingresos_ordinarios": 100000}

        rows = map_to_structure(data, "sheet1", mock_config)

        assert rows[2]["valor"] is None  # cv_gastos_personal not provided


class TestValidateCostoVentaTotal:
    """Tests for validate_costo_venta_total."""

    def test_valid_total(self) -> None:
        """Sum should match total."""
        data = {
            "cv_gastos_personal": -1000,
            "cv_materiales": -2000,
            "cv_energia": -500,
            "cv_servicios_terceros": -500,
            "cv_depreciacion_amort": -0,
            "cv_deprec_leasing": -0,
            "cv_deprec_arrend": -0,
            "cv_serv_mineros": -0,
            "cv_fletes": -0,
            "cv_gastos_diferidos": -0,
            "cv_convenios": -0,
            "total_costo_venta": -4000,
        }

        result = validate_costo_venta_total(data)

        assert result.match is True
        assert result.actual == -4000
        assert result.expected == -4000

    def test_invalid_total(self) -> None:
        """Mismatch should be detected."""
        data = {
            "cv_gastos_personal": -1000,
            "cv_materiales": -2000,
            "total_costo_venta": -5000,  # Wrong - should be -3000
        }

        result = validate_costo_venta_total(data)

        assert result.match is False
        assert result.actual == -3000
        assert result.expected == -5000
        assert result.difference == -2000


class TestValidateGastoAdminTotal:
    """Tests for validate_gasto_admin_total."""

    def test_valid_total(self) -> None:
        """Sum should match total."""
        data = {
            "ga_gastos_personal": -1000,
            "ga_materiales": -100,
            "ga_servicios_terceros": -500,
            "ga_gratificacion": -200,
            "ga_comercializacion": -100,
            "ga_otros": -100,
            "total_gasto_admin": -2000,
        }

        result = validate_gasto_admin_total(data)

        assert result.match is True


class TestValidateBalanceSheet:
    """Tests for validate_balance_sheet."""

    def test_returns_multiple_results(self) -> None:
        """Should validate both totals."""
        data = {
            "cv_gastos_personal": -1000,
            "total_costo_venta": -1000,
            "ga_gastos_personal": -500,
            "total_gasto_admin": -500,
        }

        results = validate_balance_sheet(data)

        assert len(results) == 2
        assert all(r.match for r in results)


class TestValidateAgainstReference:
    """Tests for validate_against_reference."""

    def test_all_match(self, mock_config) -> None:
        """Exact match should pass all validations."""
        extracted = {
            "ingresos_ordinarios": 179165,
            "cv_gastos_personal": -19721,
            "cv_materiales": -23219,
            "total_costo_venta": -126202,
        }

        results = validate_against_reference(extracted, "IIQ2024", mock_config)

        matching = [r for r in results if r.match]
        assert len(matching) == 4

    def test_mismatch_detected(self, mock_config) -> None:
        """Mismatched values should be flagged."""
        extracted = {
            "ingresos_ordinarios": 179165,
            "cv_gastos_personal": -20000,  # Wrong - should be -19721
        }

        results = validate_against_reference(extracted, "IIQ2024", mock_config)

        cv_result = next(r for r in results if r.field == "cv_gastos_personal")
        assert cv_result.match is False
        assert cv_result.expected == -19721
        assert cv_result.actual == -20000
        assert cv_result.difference == -279

    def test_missing_reference_period(self, mock_config) -> None:
        """Unknown period should return empty list."""
        results = validate_against_reference({}, "IQ2030", mock_config)

        assert results == []

    def test_ignores_metadata_fields(self, mock_config) -> None:
        """Should not validate 'source' and 'xbrl_available'."""
        extracted = {"source": "cmf", "xbrl_available": True}

        results = validate_against_reference(extracted, "IIQ2024", mock_config)

        field_names = [r.field for r in results]
        assert "source" not in field_names
        assert "xbrl_available" not in field_names

    def test_missing_extracted_value(self, mock_config) -> None:
        """Missing extracted values should be flagged."""
        extracted = {}  # No values extracted

        results = validate_against_reference(extracted, "IIQ2024", mock_config)

        # All results should have actual=None and match=False
        assert all(r.actual is None for r in results)
        assert all(r.match is False for r in results)


class TestFormatValidationReport:
    """Tests for format_validation_report."""

    def test_empty_results(self) -> None:
        """Empty results should return simple message."""
        report = format_validation_report([])

        assert "No validations performed" in report

    def test_includes_summary(self) -> None:
        """Report should include pass/fail counts."""
        results = [
            ValidationResult("field1", 100, 100, True),
            ValidationResult("field2", 200, 300, False, 100),
        ]

        report = format_validation_report(results)

        assert "Passed: 1" in report
        assert "Failed: 1" in report
        assert "Total: 2" in report

    def test_includes_mismatch_details(self) -> None:
        """Failed validations should show expected vs actual."""
        results = [
            ValidationResult("test_field", 1000, 1500, False, 500),
        ]

        report = format_validation_report(results)

        assert "Expected: 1,000" in report
        assert "Actual:   1,500" in report


class TestValidationResultStatus:
    """Tests for ValidationResult.status property."""

    def test_match_status(self) -> None:
        """Matching result should show checkmark."""
        result = ValidationResult("field", 100, 100, True)
        assert "✓" in result.status

    def test_mismatch_status(self) -> None:
        """Mismatched result should show X and difference."""
        result = ValidationResult("field", 100, 150, False, 50)
        assert "✗" in result.status
        assert "50" in result.status

    def test_no_reference_status(self) -> None:
        """Missing reference should show warning."""
        result = ValidationResult("field", None, 100, False)
        assert "⚠" in result.status

    def test_no_actual_status(self) -> None:
        """Missing actual should show warning."""
        result = ValidationResult("field", 100, None, False)
        assert "⚠" in result.status
