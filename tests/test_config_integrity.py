"""Tests for JSON configuration integrity and cost_extractor compatibility.

Tests cover:
1. JSON file syntax and structure validity
2. General config (config.json, extraction_specs.json, xbrl_specs.json) schema validation
3. Sheet1-specific config (fields.json, extraction.json, etc.) schema validation
4. Cross-file consistency (field names match across configs)
5. Compatibility with cost_extractor functions
"""

from __future__ import annotations

import pytest

from puco_eeff.config import (
    CONFIG_DIR,
    get_config,
    get_extraction_specs,
    get_xbrl_specs,
)
from puco_eeff.extractor.extraction import (
    _get_extraction_labels,
    _get_table_identifiers,
)
from puco_eeff.sheets.sheet1 import (
    SHEET1_CONFIG_DIR,
    Sheet1Data,
    get_sheet1_extraction_config,
    get_sheet1_fields,
    get_sheet1_reference_data,
    get_sheet1_reference_values,
    get_sheet1_section_spec,
    get_sheet1_xbrl_mappings,
)

# =============================================================================
# JSON Syntax and Loading Tests
# =============================================================================


class TestJsonSyntax:
    """Tests that all JSON config files are syntactically valid."""

    def test_config_json_loads(self) -> None:
        """config.json should be valid JSON and loadable."""
        config = get_config()
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_extraction_specs_json_loads(self) -> None:
        """extraction_specs.json should be valid JSON and loadable."""
        specs = get_extraction_specs()
        assert isinstance(specs, dict)
        assert len(specs) > 0

    def test_xbrl_specs_json_loads(self) -> None:
        """xbrl_specs.json should be valid JSON and loadable."""
        xbrl_specs = get_xbrl_specs()
        assert isinstance(xbrl_specs, dict)
        assert len(xbrl_specs) > 0

    def test_sheet1_reference_data_loads(self) -> None:
        """sheet1/reference_data.json should be valid JSON and loadable."""
        ref_data = get_sheet1_reference_data()
        assert isinstance(ref_data, dict)
        assert len(ref_data) > 0

    def test_all_root_json_files_exist(self) -> None:
        """All expected root JSON config files should exist."""
        expected_files = ["config.json", "extraction_specs.json", "xbrl_specs.json"]
        for filename in expected_files:
            filepath = CONFIG_DIR / filename
            assert filepath.exists(), f"Missing config file: {filename}"

    def test_all_sheet1_json_files_exist(self) -> None:
        """All expected sheet1 JSON config files should exist."""
        expected_files = [
            "fields.json",
            "extraction.json",
            "xbrl_mappings.json",
            "reference_data.json",
        ]
        for filename in expected_files:
            filepath = SHEET1_CONFIG_DIR / filename
            assert filepath.exists(), f"Missing sheet1 config file: {filename}"


# =============================================================================
# Sheet1 Extraction Config Tests
# =============================================================================


class TestSheet1ExtractionConfig:
    """Tests for sheet1/extraction.json structure and required fields."""

    @pytest.fixture
    def extraction_config(self) -> dict:
        """Load sheet1 extraction config."""
        return get_sheet1_extraction_config()

    def test_has_sections(self, extraction_config: dict) -> None:
        """Should have nota_21, nota_22, and ingresos sections."""
        assert "sections" in extraction_config
        sections = extraction_config["sections"]
        assert "nota_21" in sections
        assert "nota_22" in sections
        assert "ingresos" in sections

    def test_nota_21_has_required_fields(self, extraction_config: dict) -> None:
        """nota_21 section should have all required fields."""
        nota_21 = extraction_config["sections"]["nota_21"]

        required_fields = ["title", "search_patterns", "table_identifiers", "field_mappings", "fallback_section"]
        for field in required_fields:
            assert field in nota_21, f"nota_21 missing required field: {field}"

    def test_nota_22_has_required_fields(self, extraction_config: dict) -> None:
        """nota_22 section should have all required fields."""
        nota_22 = extraction_config["sections"]["nota_22"]

        required_fields = ["title", "search_patterns", "table_identifiers", "field_mappings", "fallback_section"]
        for field in required_fields:
            assert field in nota_22, f"nota_22 missing required field: {field}"

    def test_ingresos_has_required_fields(self, extraction_config: dict) -> None:
        """ingresos section should have all required fields including pdf_fallback."""
        ingresos = extraction_config["sections"]["ingresos"]

        required_fields = ["title", "fallback_section", "pdf_fallback", "field_mappings"]
        for field in required_fields:
            assert field in ingresos, f"ingresos missing required field: {field}"

        # pdf_fallback should have min_value_threshold
        pdf_fallback = ingresos["pdf_fallback"]
        assert "min_value_threshold" in pdf_fallback, "ingresos.pdf_fallback missing min_value_threshold"
        assert isinstance(pdf_fallback["min_value_threshold"], int)

    def test_fallback_section_values(self, extraction_config: dict) -> None:
        """fallback_section should have correct values for each section."""
        sections = extraction_config["sections"]

        # nota_21 has no fallback
        assert sections["nota_21"]["fallback_section"] is None

        # nota_22 falls back to nota_21
        assert sections["nota_22"]["fallback_section"] == "nota_21"

        # ingresos has no fallback
        assert sections["ingresos"]["fallback_section"] is None

    def test_table_identifiers_structure(self, extraction_config: dict) -> None:
        """table_identifiers should have unique_items and exclude_items."""
        for section_name in ["nota_21", "nota_22"]:
            section = extraction_config["sections"][section_name]
            identifiers = section["table_identifiers"]

            assert "unique_items" in identifiers, f"{section_name} missing unique_items"
            assert "exclude_items" in identifiers, f"{section_name} missing exclude_items"
            assert isinstance(identifiers["unique_items"], list)
            assert isinstance(identifiers["exclude_items"], list)
            assert len(identifiers["unique_items"]) > 0, f"{section_name} has empty unique_items"


# =============================================================================
# Sheet1 Fields Config Tests
# =============================================================================


class TestSheet1FieldsConfig:
    """Tests for sheet1/fields.json structure."""

    @pytest.fixture
    def fields_config(self) -> dict:
        """Load sheet1 fields config."""
        return get_sheet1_fields()

    def test_has_value_fields(self, fields_config: dict) -> None:
        """Should have value_fields section."""
        assert "value_fields" in fields_config
        assert isinstance(fields_config["value_fields"], dict)
        assert len(fields_config["value_fields"]) > 0

    def test_has_row_mapping(self, fields_config: dict) -> None:
        """Should have row_mapping with 27 rows."""
        assert "row_mapping" in fields_config
        row_mapping = fields_config["row_mapping"]

        # Should have entries for rows 1-27
        for i in range(1, 28):
            assert str(i) in row_mapping, f"row_mapping missing row {i}"


# =============================================================================
# Sheet1 Reference Data Tests
# =============================================================================


class TestSheet1ReferenceData:
    """Tests for sheet1/reference_data.json structure."""

    @pytest.fixture
    def ref_data(self) -> dict:
        """Load sheet1 reference data."""
        return get_sheet1_reference_data()

    def test_has_verified_period(self, ref_data: dict) -> None:
        """Should have at least one verified period."""
        has_verified = any(entry.get("verified", False) for key, entry in ref_data.items() if not key.startswith("_"))
        assert has_verified, "Should have at least one verified period"

    def test_get_reference_values_returns_dict(self) -> None:
        """get_sheet1_reference_values should return dict for verified periods."""
        values = get_sheet1_reference_values(2024, 2)
        assert values is not None, "2024 Q2 should have reference values"
        assert isinstance(values, dict)

    def test_get_reference_values_returns_none_for_unverified(self) -> None:
        """get_sheet1_reference_values should return None for unverified periods."""
        values = get_sheet1_reference_values(2030, 4)
        assert values is None, "Future period should not have verified reference values"


# =============================================================================
# Sheet1 XBRL Mappings Tests
# =============================================================================


class TestSheet1XbrlMappings:
    """Tests for sheet1/xbrl_mappings.json structure."""

    @pytest.fixture
    def xbrl_mappings(self) -> dict:
        """Load sheet1 XBRL mappings."""
        return get_sheet1_xbrl_mappings()

    def test_has_fact_mappings(self, xbrl_mappings: dict) -> None:
        """Should have fact_mappings section."""
        assert "fact_mappings" in xbrl_mappings
        assert isinstance(xbrl_mappings["fact_mappings"], dict)

    def test_has_validation_rules(self, xbrl_mappings: dict) -> None:
        """Should have validation_rules section."""
        assert "validation_rules" in xbrl_mappings

    def test_totals_have_fact_mappings(self, xbrl_mappings: dict) -> None:
        """Total fields should have XBRL fact mappings."""
        fact_mappings = xbrl_mappings["fact_mappings"]
        for field_name in ["total_costo_venta", "total_gasto_admin", "ingresos_ordinarios"]:
            assert field_name in fact_mappings, f"{field_name} should have XBRL fact mapping"
            assert "primary" in fact_mappings[field_name], f"{field_name} should have primary fact name"


# =============================================================================
# cost_extractor Compatibility Tests
# =============================================================================


class TestCostExtractorCompatibility:
    """Tests that config files work correctly with cost_extractor functions."""

    def test_get_extraction_labels_returns_valid_lists(self) -> None:
        """_get_extraction_labels should return valid lists from specs."""
        costo_venta_items, gasto_admin_items, field_labels = _get_extraction_labels()

        # Should return non-empty lists
        assert len(costo_venta_items) >= 11, "Should have at least 11 costo_venta items"
        assert len(gasto_admin_items) >= 6, "Should have at least 6 gasto_admin items"
        assert len(field_labels) > 0, "Should have field labels"

    def test_get_section_spec_returns_valid_dict(self) -> None:
        """get_sheet1_section_spec should return valid section specs."""
        for section_name in ["nota_21", "nota_22", "ingresos"]:
            spec = get_sheet1_section_spec(section_name)
            assert isinstance(spec, dict), f"{section_name} spec should be dict"
            assert "field_mappings" in spec, f"{section_name} should have field_mappings"

    def test_get_table_identifiers_returns_tuples(self) -> None:
        """_get_table_identifiers should return (unique, exclude) tuple."""
        for section_name in ["nota_21", "nota_22"]:
            spec = get_sheet1_section_spec(section_name)
            unique_items, exclude_items = _get_table_identifiers(spec)

            assert isinstance(unique_items, list)
            assert isinstance(exclude_items, list)

    def test_nota_21_unique_items_distinguish_from_nota_22(self) -> None:
        """Nota 21 unique items should not overlap with Nota 22 unique items."""
        nota_21_spec = get_sheet1_section_spec("nota_21")
        nota_22_spec = get_sheet1_section_spec("nota_22")

        nota_21_unique, _ = _get_table_identifiers(nota_21_spec)
        nota_22_unique, _ = _get_table_identifiers(nota_22_spec)

        # Normalize for comparison
        nota_21_normalized = {item.lower() for item in nota_21_unique}
        nota_22_normalized = {item.lower() for item in nota_22_unique}

        overlap = nota_21_normalized & nota_22_normalized
        assert not overlap, f"Nota 21 and Nota 22 unique items should not overlap: {overlap}"


# =============================================================================
# Cross-File Consistency Tests
# =============================================================================


class TestCrossFileConsistency:
    """Tests that field names are consistent across config files."""

    def test_extraction_fields_match_fields_config(self) -> None:
        """Field names in extraction.json should match fields.json."""
        extraction = get_sheet1_extraction_config()
        fields = get_sheet1_fields()

        # Get all field names from extraction config
        extraction_fields = set()
        for section_name in ["nota_21", "nota_22", "ingresos"]:
            section = extraction["sections"].get(section_name, {})
            for field_name in section.get("field_mappings", {}):
                extraction_fields.add(field_name)

        # Get value fields from fields.json
        value_fields = set(fields.get("value_fields", {}).keys())

        # All extraction fields should be defined in value_fields
        missing = extraction_fields - value_fields
        assert not missing, f"Extraction fields missing in fields.json: {missing}"

    def test_extraction_fields_match_sheet1_data(self) -> None:
        """Field names in extraction.json should match Sheet1Data attributes."""
        extraction = get_sheet1_extraction_config()

        # Get all field names from extraction config
        extraction_fields = set()
        for section_name in ["nota_21", "nota_22", "ingresos"]:
            section = extraction["sections"].get(section_name, {})
            for field_name in section.get("field_mappings", {}):
                extraction_fields.add(field_name)

        # Get Sheet1Data attributes
        sheet1_sample = Sheet1Data(quarter="test", year=2024, quarter_num=2)
        sheet1_fields = set(sheet1_sample.to_dict().keys())

        # Remove metadata fields from Sheet1Data
        metadata_fields = {
            "quarter",
            "year",
            "quarter_num",
            "period_type",
            "source",
            "xbrl_available",
        }
        sheet1_value_fields = sheet1_fields - metadata_fields

        # All extraction fields should exist in Sheet1Data
        missing_in_sheet1 = extraction_fields - sheet1_value_fields
        assert not missing_in_sheet1, f"Extraction fields missing in Sheet1Data: {missing_in_sheet1}"


# =============================================================================
# Reference Value Validation Tests
# =============================================================================


class TestReferenceValueValidation:
    """Tests that validate reference values are internally consistent."""

    def test_nota_21_items_sum_to_total(self) -> None:
        """Sum of Nota 21 items should equal total_costo_venta."""
        values = get_sheet1_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")

        nota_21_items = [
            "cv_gastos_personal",
            "cv_materiales",
            "cv_energia",
            "cv_servicios_terceros",
            "cv_depreciacion_amort",
            "cv_deprec_leasing",
            "cv_deprec_arrend",
            "cv_serv_mineros",
            "cv_fletes",
            "cv_gastos_diferidos",
            "cv_convenios",
        ]

        calculated_sum = sum(values[item] for item in nota_21_items)
        expected_total = values["total_costo_venta"]

        assert calculated_sum == expected_total, (
            f"Nota 21 items sum ({calculated_sum}) != total_costo_venta ({expected_total})"
        )

    def test_nota_22_items_sum_to_total(self) -> None:
        """Sum of Nota 22 items should equal total_gasto_admin."""
        values = get_sheet1_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")

        nota_22_items = [
            "ga_gastos_personal",
            "ga_materiales",
            "ga_servicios_terceros",
            "ga_gratificacion",
            "ga_comercializacion",
            "ga_otros",
        ]

        calculated_sum = sum(values[item] for item in nota_22_items)
        expected_total = values["total_gasto_admin"]

        assert calculated_sum == expected_total, (
            f"Nota 22 items sum ({calculated_sum}) != total_gasto_admin ({expected_total})"
        )

    def test_all_cost_values_are_negative(self) -> None:
        """All cost breakdown values should be negative (expenses)."""
        values = get_sheet1_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")

        cost_fields = [
            "cv_gastos_personal",
            "cv_materiales",
            "cv_energia",
            "cv_servicios_terceros",
            "cv_depreciacion_amort",
            "cv_deprec_leasing",
            "cv_deprec_arrend",
            "cv_serv_mineros",
            "cv_fletes",
            "cv_gastos_diferidos",
            "cv_convenios",
            "total_costo_venta",
            "ga_gastos_personal",
            "ga_materiales",
            "ga_servicios_terceros",
            "ga_gratificacion",
            "ga_comercializacion",
            "ga_otros",
            "total_gasto_admin",
        ]

        for field in cost_fields:
            assert values[field] < 0, f"{field} should be negative (it's a cost)"

    def test_ingresos_is_positive(self) -> None:
        """Ingresos (revenue) should be positive."""
        values = get_sheet1_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")
        assert values["ingresos_ordinarios"] > 0, "ingresos_ordinarios should be positive"
