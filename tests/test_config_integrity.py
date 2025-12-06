"""Tests for JSON configuration integrity and cost_extractor compatibility.

Tests cover:
1. JSON file syntax and structure validity
2. extraction_specs.json schema validation
3. reference_data.json schema validation
4. config.json schema validation
5. xbrl_specs.json schema validation
6. Cross-file consistency (field names match across configs)
7. Compatibility with cost_extractor functions
8. Default + deviation merge logic
"""

from __future__ import annotations

import pytest

from puco_eeff.config import (
    CONFIG_DIR,
    get_config,
    get_extraction_specs,
    get_period_specs,
    get_reference_data,
    get_reference_values,
    get_xbrl_specs,
)
from puco_eeff.extractor.cost_extractor import (
    Sheet1Data,
    _get_extraction_labels,
    _get_section_spec,
    _get_table_identifiers,
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

    def test_reference_data_json_loads(self) -> None:
        """reference_data.json should be valid JSON and loadable."""
        ref_data = get_reference_data()
        assert isinstance(ref_data, dict)
        assert len(ref_data) > 0

    def test_all_json_files_exist(self) -> None:
        """All expected JSON config files should exist."""
        expected_files = ["config.json", "extraction_specs.json", "reference_data.json"]
        for filename in expected_files:
            filepath = CONFIG_DIR / filename
            assert filepath.exists(), f"Missing config file: {filename}"


# =============================================================================
# extraction_specs.json Schema Validation
# =============================================================================


class TestExtractionSpecsSchema:
    """Tests for extraction_specs.json structure and required fields."""

    @pytest.fixture
    def specs(self) -> dict:
        """Load extraction specs."""
        return get_extraction_specs()

    def test_has_default_section(self, specs: dict) -> None:
        """Should have a 'default' section with template."""
        assert "default" in specs
        assert isinstance(specs["default"], dict)

    def test_default_has_sections(self, specs: dict) -> None:
        """Default should have nota_21, nota_22, and ingresos sections."""
        default = specs["default"]
        assert "sections" in default
        sections = default["sections"]
        assert "nota_21" in sections
        assert "nota_22" in sections
        assert "ingresos" in sections

    def test_nota_21_has_required_fields(self, specs: dict) -> None:
        """nota_21 section should have all required fields."""
        nota_21 = specs["default"]["sections"]["nota_21"]

        required_fields = ["title", "search_patterns", "table_identifiers", "field_mappings"]
        for field in required_fields:
            assert field in nota_21, f"nota_21 missing required field: {field}"

    def test_nota_22_has_required_fields(self, specs: dict) -> None:
        """nota_22 section should have all required fields."""
        nota_22 = specs["default"]["sections"]["nota_22"]

        required_fields = ["title", "search_patterns", "table_identifiers", "field_mappings"]
        for field in required_fields:
            assert field in nota_22, f"nota_22 missing required field: {field}"

    def test_table_identifiers_structure(self, specs: dict) -> None:
        """table_identifiers should have unique_items and exclude_items."""
        for section_name in ["nota_21", "nota_22"]:
            section = specs["default"]["sections"][section_name]
            identifiers = section["table_identifiers"]

            assert "unique_items" in identifiers, f"{section_name} missing unique_items"
            assert "exclude_items" in identifiers, f"{section_name} missing exclude_items"
            assert isinstance(identifiers["unique_items"], list)
            assert isinstance(identifiers["exclude_items"], list)
            assert len(identifiers["unique_items"]) > 0, f"{section_name} has empty unique_items"

    def test_field_mappings_have_pdf_labels(self, specs: dict) -> None:
        """Each field mapping should have at least one pdf_label."""
        for section_name in ["nota_21", "nota_22"]:
            section = specs["default"]["sections"][section_name]
            field_mappings = section["field_mappings"]

            for field_name, field_spec in field_mappings.items():
                assert "pdf_labels" in field_spec, f"{section_name}.{field_name} missing pdf_labels"
                assert isinstance(field_spec["pdf_labels"], list)
                assert len(field_spec["pdf_labels"]) > 0, f"{section_name}.{field_name} has empty pdf_labels"

    def test_search_patterns_are_lowercase(self, specs: dict) -> None:
        """Search patterns should be lowercase for case-insensitive matching."""
        for section_name in ["nota_21", "nota_22"]:
            section = specs["default"]["sections"][section_name]
            patterns = section["search_patterns"]

            for pattern in patterns:
                assert pattern == pattern.lower(), f"Search pattern should be lowercase: {pattern}"

    def test_period_keys_format(self, specs: dict) -> None:
        """Period keys should follow YYYY_QN format or be valid special keys."""
        import re

        period_pattern = re.compile(r"^\d{4}_Q[1-4]$")
        # Special keys that are not period keys
        special_keys = ["default", "_description", "sheet1_fields"]

        for key in specs:
            if key not in special_keys:
                assert period_pattern.match(key), f"Invalid period key format: {key}"

    def test_nota_21_field_mappings_complete(self, specs: dict) -> None:
        """nota_21 should have all 11 cost items + total."""
        expected_fields = [
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
        ]
        field_mappings = specs["default"]["sections"]["nota_21"]["field_mappings"]

        for field in expected_fields:
            assert field in field_mappings, f"nota_21 missing field mapping: {field}"

    def test_nota_22_field_mappings_complete(self, specs: dict) -> None:
        """nota_22 should have all 6 admin items + total."""
        expected_fields = [
            "ga_gastos_personal",
            "ga_materiales",
            "ga_servicios_terceros",
            "ga_gratificacion",
            "ga_comercializacion",
            "ga_otros",
            "total_gasto_admin",
        ]
        field_mappings = specs["default"]["sections"]["nota_22"]["field_mappings"]

        for field in expected_fields:
            assert field in field_mappings, f"nota_22 missing field mapping: {field}"


# =============================================================================
# reference_data.json Schema Validation
# =============================================================================


class TestReferenceDataSchema:
    """Tests for reference_data.json structure."""

    @pytest.fixture
    def ref_data(self) -> dict:
        """Load reference data."""
        return get_reference_data()

    def test_has_description(self, ref_data: dict) -> None:
        """Should have a description field."""
        assert "_description" in ref_data

    def test_period_entries_have_verified_flag(self, ref_data: dict) -> None:
        """Each period entry should have verified flag."""
        for key, entry in ref_data.items():
            if key.startswith("_"):
                continue
            assert "verified" in entry, f"Period {key} missing 'verified' flag"
            assert isinstance(entry["verified"], bool)

    def test_verified_entries_have_values(self, ref_data: dict) -> None:
        """Verified entries should have values dict."""
        for key, entry in ref_data.items():
            if key.startswith("_"):
                continue
            if entry.get("verified"):
                assert "values" in entry, f"Verified period {key} missing 'values'"
                assert isinstance(entry["values"], dict)
                assert len(entry["values"]) > 0

    def test_2024_q2_has_all_sheet1_fields(self, ref_data: dict) -> None:
        """2024_Q2 (verified) should have all Sheet1 value fields."""
        expected_fields = [
            "ingresos_ordinarios",
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

        values = ref_data["2024_Q2"]["values"]
        for field in expected_fields:
            assert field in values, f"2024_Q2 missing reference value: {field}"
            assert isinstance(values[field], int), f"2024_Q2.{field} should be int"

    def test_reference_values_are_integers(self, ref_data: dict) -> None:
        """All reference values should be integers."""
        for key, entry in ref_data.items():
            if key.startswith("_"):
                continue
            if entry.get("values"):
                for field_name, value in entry["values"].items():
                    assert isinstance(value, int), f"{key}.{field_name} should be int, got {type(value)}"


# =============================================================================
# config.json Schema Validation
# =============================================================================


class TestConfigJsonSchema:
    """Tests for main config.json structure."""

    @pytest.fixture
    def config(self) -> dict:
        """Load main config."""
        return get_config()

    def test_has_sources_section(self, config: dict) -> None:
        """Should have data sources defined."""
        assert "sources" in config
        assert "cmf_chile" in config["sources"]

    def test_has_sheets_section(self, config: dict) -> None:
        """Should have sheets section with sheet1."""
        assert "sheets" in config
        assert "sheet1" in config["sheets"]

    def test_sheet1_has_row_mapping(self, config: dict) -> None:
        """sheet1 should have row_mapping with 27 rows."""
        sheet1 = config["sheets"]["sheet1"]
        assert "row_mapping" in sheet1

        row_mapping = sheet1["row_mapping"]
        # Should have entries for rows 1-27
        for i in range(1, 28):
            assert str(i) in row_mapping, f"row_mapping missing row {i}"

    def test_has_xbrl_specs_file(self) -> None:
        """XBRL config should be in separate xbrl_specs.json file."""
        from puco_eeff.config import get_xbrl_specs

        xbrl_specs = get_xbrl_specs()
        assert "namespaces" in xbrl_specs
        assert "fact_mappings" in xbrl_specs
        assert "validation_rules" in xbrl_specs

    def test_has_ocr_config(self, config: dict) -> None:
        """Should have OCR configuration."""
        assert "ocr" in config
        assert "primary" in config["ocr"]


# =============================================================================
# Cross-File Consistency Tests
# =============================================================================


class TestCrossFileConsistency:
    """Tests that field names are consistent across config files."""

    def test_extraction_specs_fields_match_reference_data(self) -> None:
        """Field names in extraction_specs should match reference_data."""
        specs = get_extraction_specs()
        ref_data = get_reference_data()

        # Get all field names from extraction_specs
        spec_fields = set()
        for section_name in ["nota_21", "nota_22", "ingresos"]:
            section = specs["default"]["sections"].get(section_name, {})
            for field_name in section.get("field_mappings", {}):
                spec_fields.add(field_name)

        # Get all field names from reference_data (2024_Q2)
        ref_fields = set(ref_data["2024_Q2"]["values"].keys())

        # Check that all reference fields exist in specs
        missing_in_specs = ref_fields - spec_fields
        assert not missing_in_specs, f"Reference fields missing in extraction_specs: {missing_in_specs}"

    def test_extraction_specs_fields_match_sheet1_data(self) -> None:
        """Field names in extraction_specs should match Sheet1Data attributes."""
        specs = get_extraction_specs()

        # Get all field names from extraction_specs
        spec_fields = set()
        for section_name in ["nota_21", "nota_22", "ingresos"]:
            section = specs["default"]["sections"].get(section_name, {})
            for field_name in section.get("field_mappings", {}):
                spec_fields.add(field_name)

        # Get Sheet1Data attributes
        sheet1_sample = Sheet1Data(quarter="test", year=2024, quarter_num=2)
        sheet1_fields = set(sheet1_sample.to_dict().keys())

        # Remove metadata fields from Sheet1Data
        metadata_fields = {"quarter", "year", "quarter_num", "source", "xbrl_available"}
        sheet1_value_fields = sheet1_fields - metadata_fields

        # All spec fields should exist in Sheet1Data
        missing_in_sheet1 = spec_fields - sheet1_value_fields
        assert not missing_in_sheet1, f"Spec fields missing in Sheet1Data: {missing_in_sheet1}"

    def test_config_row_mapping_fields_match_sheet1_data(self) -> None:
        """row_mapping field names should match Sheet1Data attributes."""
        config = get_config()
        row_mapping = config["sheets"]["sheet1"]["row_mapping"]

        # Get field names from row_mapping (exclude None and headers)
        row_mapping_fields = set()
        for _row_num, row_spec in row_mapping.items():
            field = row_spec.get("field")
            if field and not field.endswith("_header"):
                row_mapping_fields.add(field)

        # Get Sheet1Data attributes
        sheet1_sample = Sheet1Data(quarter="test", year=2024, quarter_num=2)
        sheet1_fields = set(sheet1_sample.to_dict().keys())

        # All row_mapping fields should exist in Sheet1Data
        missing_in_sheet1 = row_mapping_fields - sheet1_fields
        assert not missing_in_sheet1, f"row_mapping fields missing in Sheet1Data: {missing_in_sheet1}"


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

        # All items should be strings
        for item in costo_venta_items:
            assert isinstance(item, str)
        for item in gasto_admin_items:
            assert isinstance(item, str)

    def test_get_section_spec_returns_valid_dict(self) -> None:
        """_get_section_spec should return valid section specs."""
        for section_name in ["nota_21", "nota_22", "ingresos"]:
            spec = _get_section_spec(section_name, 2024, 2)
            assert isinstance(spec, dict), f"{section_name} spec should be dict"

    def test_get_section_spec_with_different_periods(self) -> None:
        """_get_section_spec should work for different periods."""
        # Q2 2024 - verified
        spec_q2 = _get_section_spec("nota_21", 2024, 2)
        assert "field_mappings" in spec_q2

        # Q3 2024 - placeholder
        spec_q3 = _get_section_spec("nota_21", 2024, 3)
        assert "field_mappings" in spec_q3

        # Q1 2024 - pucobre source
        spec_q1 = _get_section_spec("nota_21", 2024, 1)
        assert "field_mappings" in spec_q1

    def test_get_table_identifiers_returns_tuples(self) -> None:
        """_get_table_identifiers should return (unique, exclude) tuple."""
        for section_name in ["nota_21", "nota_22"]:
            spec = _get_section_spec(section_name, 2024, 2)
            unique_items, exclude_items = _get_table_identifiers(spec)

            assert isinstance(unique_items, list)
            assert isinstance(exclude_items, list)

    def test_nota_21_unique_items_distinguish_from_nota_22(self) -> None:
        """Nota 21 unique items should not overlap with Nota 22 unique items."""
        nota_21_spec = _get_section_spec("nota_21", 2024, 2)
        nota_22_spec = _get_section_spec("nota_22", 2024, 2)

        nota_21_unique, _ = _get_table_identifiers(nota_21_spec)
        nota_22_unique, _ = _get_table_identifiers(nota_22_spec)

        # Normalize for comparison
        nota_21_normalized = {item.lower() for item in nota_21_unique}
        nota_22_normalized = {item.lower() for item in nota_22_unique}

        overlap = nota_21_normalized & nota_22_normalized
        assert not overlap, f"Nota 21 and Nota 22 unique items should not overlap: {overlap}"

    def test_get_reference_values_returns_dict_for_verified(self) -> None:
        """get_reference_values should return dict for verified periods."""
        values = get_reference_values(2024, 2)
        assert values is not None, "2024 Q2 should have reference values"
        assert isinstance(values, dict)
        assert "ingresos_ordinarios" in values
        assert "total_costo_venta" in values
        assert "total_gasto_admin" in values

    def test_get_reference_values_returns_none_for_unverified(self) -> None:
        """get_reference_values should return None for unverified periods."""
        values = get_reference_values(2024, 3)
        assert values is None, "2024 Q3 should not have verified reference values"

    def test_get_period_specs_merges_defaults(self) -> None:
        """get_period_specs should merge defaults with period-specific deviations."""
        specs = get_period_specs(2024, 2)

        # Should have sections from default
        assert "sections" in specs
        assert "nota_21" in specs["sections"]

        # Should have merged metadata
        assert "_period" in specs
        assert specs["_period"] == "2024_Q2"
        assert "_verified" in specs


# =============================================================================
# Reference Value Validation Tests
# =============================================================================


class TestReferenceValueValidation:
    """Tests that validate reference values are internally consistent."""

    def test_nota_21_items_sum_to_total(self) -> None:
        """Sum of Nota 21 items should equal total_costo_venta."""
        values = get_reference_values(2024, 2)
        assert values is not None

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
        values = get_reference_values(2024, 2)
        assert values is not None

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
        values = get_reference_values(2024, 2)
        assert values is not None

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
        values = get_reference_values(2024, 2)
        assert values is not None
        assert values["ingresos_ordinarios"] > 0, "ingresos_ordinarios should be positive"


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_period_returns_defaults(self) -> None:
        """Requesting non-existent period should return defaults."""
        # A far future period that doesn't exist
        specs = get_period_specs(2030, 4)

        # Should still have sections from defaults
        assert "sections" in specs
        assert "nota_21" in specs["sections"]

    def test_xbrl_specs_has_fact_mappings(self) -> None:
        """XBRL specs should have fact mappings for totals and ingresos."""
        from puco_eeff.config import get_xbrl_fact_mapping

        # Check ingresos has XBRL mapping
        ingresos_mapping = get_xbrl_fact_mapping("ingresos_ordinarios")
        assert ingresos_mapping is not None, "ingresos should have XBRL fact mapping"
        assert "primary" in ingresos_mapping, "ingresos should have primary fact name"

    def test_totals_have_xbrl_fact_mappings(self) -> None:
        """Total fields should have XBRL fact mappings for validation."""
        from puco_eeff.config import get_xbrl_fact_mapping

        for field_name in ["total_costo_venta", "total_gasto_admin"]:
            mapping = get_xbrl_fact_mapping(field_name)
            assert mapping is not None, f"{field_name} should have XBRL fact mapping"
            assert "primary" in mapping, f"{field_name} should have primary fact name"
