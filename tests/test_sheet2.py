"""Tests for Sheet2 JSON configuration integrity and validation.

Tests cover:
1. JSON file syntax and structure validity
2. Sheet2-specific config (fields.json, extraction.json, etc.) schema validation
3. Cross-file consistency (field names match across configs)
4. Reference data validation (sums, types)
5. Sheet2Data dataclass compatibility
"""

from __future__ import annotations

import pytest

from puco_eeff.sheets.sheet2 import (
    SHEET2_CONFIG_DIR,
    Sheet2Data,
    get_sheet2_extraction_config,
    get_sheet2_fields,
    get_sheet2_reference_data,
    get_sheet2_reference_values,
    get_sheet2_xbrl_mappings,
    parse_spanish_number,
    validate_sheet2_sums,
)

# Type alias for reference values
ReferenceValues = dict[str, int | float | None]


# =============================================================================
# JSON Syntax and Loading Tests
# =============================================================================


class TestSheet2JsonSyntax:
    """Tests that all Sheet2 JSON config files are syntactically valid."""

    def test_all_sheet2_json_files_exist(self) -> None:
        """All expected sheet2 JSON config files should exist."""
        expected_files = [
            "fields.json",
            "extraction.json",
            "xbrl_mappings.json",
            "reference_data.json",
        ]
        for filename in expected_files:
            filepath = SHEET2_CONFIG_DIR / filename
            assert filepath.exists(), f"Missing sheet2 config file: {filename}"

    def test_sheet2_fields_loads(self) -> None:
        """sheet2/fields.json should be valid JSON and loadable."""
        fields = get_sheet2_fields()
        assert isinstance(fields, dict)
        assert len(fields) > 0

    def test_sheet2_extraction_config_loads(self) -> None:
        """sheet2/extraction.json should be valid JSON and loadable."""
        extraction = get_sheet2_extraction_config()
        assert isinstance(extraction, dict)
        assert len(extraction) > 0

    def test_sheet2_xbrl_mappings_loads(self) -> None:
        """sheet2/xbrl_mappings.json should be valid JSON and loadable."""
        xbrl = get_sheet2_xbrl_mappings()
        assert isinstance(xbrl, dict)
        assert len(xbrl) > 0

    def test_sheet2_reference_data_loads(self) -> None:
        """sheet2/reference_data.json should be valid JSON and loadable."""
        ref_data = get_sheet2_reference_data()
        assert isinstance(ref_data, dict)
        assert len(ref_data) > 0


# =============================================================================
# Sheet2 Fields Config Tests
# =============================================================================


class TestSheet2FieldsConfig:
    """Tests for sheet2/fields.json structure."""

    @pytest.fixture
    def fields_config(self) -> dict:
        """Load sheet2 fields config."""
        return get_sheet2_fields()

    def test_has_value_fields(self, fields_config: dict) -> None:
        """Should have value_fields section with expected KPI fields."""
        assert "value_fields" in fields_config
        value_fields = fields_config["value_fields"]
        assert isinstance(value_fields, dict)

        expected_fields = [
            "cobre_concentrados",
            "cobre_catodos",
            "oro_subproducto",
            "plata_subproducto",
            "total_ingresos",
            "ebitda",
            "libras_vendidas",
            "cobre_fino",
            "precio_efectivo",
            "cash_cost",
            "costo_unitario_total",
            "non_cash_cost",
            "toneladas_procesadas",
            "oro_onzas",
        ]

        for field in expected_fields:
            assert field in value_fields, f"Missing field: {field}"

    def test_has_row_mapping(self, fields_config: dict) -> None:
        """Should have row_mapping with 15 rows."""
        assert "row_mapping" in fields_config
        row_mapping = fields_config["row_mapping"]

        # Should have entries for rows 1-15
        for i in range(1, 16):
            assert str(i) in row_mapping, f"row_mapping missing row {i}"

    def test_value_fields_have_required_properties(self, fields_config: dict) -> None:
        """Each value field should have label, unit, and type."""
        value_fields = fields_config["value_fields"]

        for field_name, field_def in value_fields.items():
            assert "label" in field_def, f"{field_name} missing label"
            assert "unit" in field_def, f"{field_name} missing unit"
            assert "type" in field_def, f"{field_name} missing type"
            assert field_def["type"] in ("int", "float"), f"{field_name} has invalid type"

    def test_monetary_fields_are_int(self, fields_config: dict) -> None:
        """Monetary fields (MUS$) should be int type."""
        value_fields = fields_config["value_fields"]

        monetary_fields = [
            "cobre_concentrados",
            "cobre_catodos",
            "oro_subproducto",
            "plata_subproducto",
            "total_ingresos",
            "ebitda",
        ]

        for field in monetary_fields:
            assert value_fields[field]["type"] == "int", f"{field} should be int type"
            assert "MUS$" in value_fields[field]["unit"], f"{field} should have MUS$ unit"

    def test_metric_fields_are_float(self, fields_config: dict) -> None:
        """Operational metric fields should be float type."""
        value_fields = fields_config["value_fields"]

        float_fields = [
            "libras_vendidas",
            "cobre_fino",
            "precio_efectivo",
            "cash_cost",
            "costo_unitario_total",
            "non_cash_cost",
            "toneladas_procesadas",
            "oro_onzas",
        ]

        for field in float_fields:
            assert value_fields[field]["type"] == "float", f"{field} should be float type"


# =============================================================================
# Sheet2 Extraction Config Tests
# =============================================================================


class TestSheet2ExtractionConfig:
    """Tests for sheet2/extraction.json structure."""

    @pytest.fixture
    def extraction_config(self) -> dict:
        """Load sheet2 extraction config."""
        return get_sheet2_extraction_config()

    def test_has_field_keywords(self, extraction_config: dict) -> None:
        """Should have field_keywords section with all fields."""
        assert "field_keywords" in extraction_config
        field_keywords = extraction_config["field_keywords"]
        assert isinstance(field_keywords, dict)
        assert len(field_keywords) > 0

        # Should have all Sheet2 fields
        expected = [
            "cobre_concentrados",
            "cobre_catodos",
            "oro_subproducto",
            "plata_subproducto",
            "total_ingresos",
            "ebitda",
            "libras_vendidas",
            "cobre_fino",
            "precio_efectivo",
            "cash_cost",
            "costo_unitario_total",
            "non_cash_cost",
            "toneladas_procesadas",
            "oro_onzas",
        ]
        for field in expected:
            assert field in field_keywords, f"Missing field: {field}"

    def test_field_keywords_have_keyword_and_type(self, extraction_config: dict) -> None:
        """Each field should have keyword and type."""
        field_keywords = extraction_config["field_keywords"]
        for field_name, config in field_keywords.items():
            assert "keyword" in config, f"{field_name} missing keyword"
            assert "type" in config, f"{field_name} missing type"
            assert config["type"] in ("int", "float"), f"{field_name} invalid type"


# =============================================================================
# Sheet2 XBRL Mappings Tests
# =============================================================================


class TestSheet2XbrlMappings:
    """Tests for sheet2/xbrl_mappings.json structure."""

    @pytest.fixture
    def xbrl_mappings(self) -> dict:
        """Load sheet2 XBRL mappings."""
        return get_sheet2_xbrl_mappings()

    def test_has_fact_mappings(self, xbrl_mappings: dict) -> None:
        """Should have fact_mappings section."""
        assert "fact_mappings" in xbrl_mappings
        assert isinstance(xbrl_mappings["fact_mappings"], dict)

    def test_total_ingresos_mapped(self, xbrl_mappings: dict) -> None:
        """total_ingresos should have XBRL mapping for cross-validation."""
        fact_mappings = xbrl_mappings["fact_mappings"]
        assert "total_ingresos" in fact_mappings
        assert "primary" in fact_mappings["total_ingresos"]


# =============================================================================
# Sheet2 Reference Data Tests
# =============================================================================


class TestSheet2ReferenceData:
    """Tests for sheet2/reference_data.json structure."""

    @pytest.fixture
    def ref_data(self) -> dict:
        """Load sheet2 reference data."""
        return get_sheet2_reference_data()

    def test_has_verified_periods(self, ref_data: dict) -> None:
        """Should have verified periods."""
        verified_count = sum(
            1
            for key, entry in ref_data.items()
            if isinstance(entry, dict) and entry.get("verified", False)
        )
        assert verified_count >= 1, "Should have at least one verified period"

    def test_get_reference_values_returns_dict(self) -> None:
        """get_sheet2_reference_values should return dict for verified periods."""
        values = get_sheet2_reference_values(2024, 2)
        assert values is not None, "2024 Q2 should have reference values"
        assert isinstance(values, dict)

    def test_get_reference_values_returns_none_for_unverified(self) -> None:
        """get_sheet2_reference_values should return None for unverified periods."""
        values = get_sheet2_reference_values(2030, 4)
        assert values is None, "Future period should not have verified reference values"

    def test_reference_has_all_fields(self) -> None:
        """Reference values should include all KPI fields."""
        values = get_sheet2_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")

        expected_fields = [
            "cobre_concentrados",
            "cobre_catodos",
            "oro_subproducto",
            "plata_subproducto",
            "total_ingresos",
            "ebitda",
            "libras_vendidas",
            "cobre_fino",
            "precio_efectivo",
            "cash_cost",
            "costo_unitario_total",
            "non_cash_cost",
            "toneladas_procesadas",
            "oro_onzas",
        ]

        for field in expected_fields:
            assert field in values, f"Reference missing field: {field}"


# =============================================================================
# Reference Value Validation Tests
# =============================================================================


class TestReferenceValueValidation:
    """Tests that validate reference values are internally consistent."""

    def test_revenue_sum_validation(self) -> None:
        """Sum of revenue components should equal total_ingresos."""
        values = get_sheet2_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")

        revenue_items = [
            "cobre_concentrados",
            "cobre_catodos",
            "oro_subproducto",
            "plata_subproducto",
        ]

        # Filter out None values and sum
        item_values = [values[item] for item in revenue_items if values.get(item) is not None]
        calculated_sum = sum(v for v in item_values if v is not None)
        expected_total = values.get("total_ingresos")

        assert expected_total is not None, "total_ingresos should be present"
        # Allow ±1 tolerance for rounding
        assert abs(calculated_sum - expected_total) <= 1, (
            f"Revenue items sum ({calculated_sum}) != total_ingresos ({expected_total})"
        )

    def test_monetary_values_are_positive(self) -> None:
        """Revenue values should be positive."""
        values = get_sheet2_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")

        positive_fields = [
            "cobre_concentrados",
            "cobre_catodos",
            "oro_subproducto",
            "plata_subproducto",
            "total_ingresos",
            "ebitda",
        ]

        for field in positive_fields:
            val = values.get(field)
            assert val is not None, f"{field} should be present"
            assert val > 0, f"{field} should be positive"

    def test_cost_values_are_positive(self) -> None:
        """Cost metrics ($/lb) should be positive."""
        values = get_sheet2_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")

        cost_fields = ["cash_cost", "costo_unitario_total", "non_cash_cost"]

        for field in cost_fields:
            val = values.get(field)
            assert val is not None, f"{field} should be present"
            assert val > 0, f"{field} should be positive"

    def test_production_values_are_positive(self) -> None:
        """Production metrics should be positive."""
        values = get_sheet2_reference_values(2024, 2)
        if values is None:
            pytest.skip("No reference values for 2024 Q2")

        production_fields = [
            "libras_vendidas",
            "cobre_fino",
            "toneladas_procesadas",
            "oro_onzas",
        ]

        for field in production_fields:
            val = values.get(field)
            assert val is not None, f"{field} should be present"
            assert val > 0, f"{field} should be positive"


# =============================================================================
# Cross-File Consistency Tests
# =============================================================================


class TestCrossFileConsistency:
    """Tests that field names are consistent across config files."""

    def test_extraction_fields_match_fields_config(self) -> None:
        """Field names in extraction.json should match fields.json."""
        extraction = get_sheet2_extraction_config()
        fields = get_sheet2_fields()

        # Get all field names from extraction config
        extraction_fields = set(extraction.get("field_keywords", {}).keys())

        # Get value fields from fields.json
        value_fields = set(fields.get("value_fields", {}).keys())

        # All extraction fields should be defined in value_fields
        missing = extraction_fields - value_fields
        assert not missing, f"Extraction fields missing in fields.json: {missing}"

    def test_extraction_fields_match_sheet2_data(self) -> None:
        """Field names in extraction.json should match Sheet2Data attributes."""
        extraction = get_sheet2_extraction_config()

        # Get all field names from extraction config
        extraction_fields = set(extraction.get("field_keywords", {}).keys())

        # Get Sheet2Data attributes
        sheet2_sample = Sheet2Data(quarter="test", year=2024, quarter_num=2)
        sheet2_fields = set(sheet2_sample.to_dict().keys())

        # Remove metadata fields from Sheet2Data
        metadata_fields = {
            "quarter",
            "year",
            "quarter_num",
            "source",
            "extraction_timestamp",
            "validation_issues",
        }
        sheet2_value_fields = sheet2_fields - metadata_fields

        # All extraction fields should exist in Sheet2Data
        missing_in_sheet2 = extraction_fields - sheet2_value_fields
        assert not missing_in_sheet2, (
            f"Extraction fields missing in Sheet2Data: {missing_in_sheet2}"
        )


# =============================================================================
# Sheet2Data Dataclass Tests
# =============================================================================


class TestSheet2DataClass:
    """Tests for Sheet2Data dataclass functionality."""

    def test_create_empty_instance(self) -> None:
        """Should create Sheet2Data with default None values."""
        data = Sheet2Data(quarter="IIQ2024", year=2024, quarter_num=2)
        assert data.quarter == "IIQ2024"
        assert data.year == 2024
        assert data.quarter_num == 2
        assert data.cobre_concentrados is None

    def test_create_with_values(self) -> None:
        """Should create Sheet2Data with provided values."""
        data = Sheet2Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            cobre_concentrados=142391,
            cobre_catodos=15022,
            total_ingresos=179165,
        )
        assert data.cobre_concentrados == 142391
        assert data.cobre_catodos == 15022
        assert data.total_ingresos == 179165

    def test_to_dict(self) -> None:
        """to_dict should return all fields as dict."""
        data = Sheet2Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            ebitda=65483,
        )
        d = data.to_dict()
        assert isinstance(d, dict)
        assert d["quarter"] == "IIQ2024"
        assert d["ebitda"] == 65483

    def test_get_value(self) -> None:
        """get_value should retrieve field value."""
        data = Sheet2Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            cash_cost=2.59,
        )
        assert data.get_value("cash_cost") == 2.59
        assert data.get_value("nonexistent") is None

    def test_set_value(self) -> None:
        """set_value should update field value."""
        data = Sheet2Data(quarter="IIQ2024", year=2024, quarter_num=2)
        data.set_value("libras_vendidas", 38.4)
        assert data.libras_vendidas == 38.4

    def test_to_row_list(self) -> None:
        """to_row_list should return list of (row, label, value) tuples."""
        data = Sheet2Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            cobre_concentrados=142391,
        )
        rows = data.to_row_list()
        assert isinstance(rows, list)
        assert len(rows) == 15  # 15 rows in fields.json

        # First row should be cobre_concentrados
        row_num, _label, value = rows[0]
        assert row_num == 1
        assert value == 142391


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions in sheet2.py."""

    def test_parse_spanish_number_comma_decimal(self) -> None:
        """Should parse Spanish format with comma as decimal separator."""
        assert parse_spanish_number("19,5") == 19.5
        assert parse_spanish_number("3,97") == 3.97
        assert parse_spanish_number("0,98") == 0.98

    def test_parse_spanish_number_period_thousands(self) -> None:
        """Should parse Spanish format with period as thousands separator."""
        # Period as thousands separator (no decimal part)
        assert parse_spanish_number("64.057") == 64057
        assert parse_spanish_number("1.342") == 1342
        assert parse_spanish_number("142.391") == 142391

    def test_parse_spanish_number_mixed(self) -> None:
        """Should parse mixed format (thousands and decimal)."""
        # Thousands with comma decimal
        assert parse_spanish_number("1.234,56") == 1234.56

    def test_parse_spanish_number_plain_int(self) -> None:
        """Should parse plain integers."""
        assert parse_spanish_number("123") == 123
        assert parse_spanish_number("0") == 0

    def test_parse_spanish_number_negative(self) -> None:
        """Should handle negative numbers."""
        assert parse_spanish_number("-19,5") == -19.5
        assert parse_spanish_number("-1.234") == -1234

    def test_validate_sheet2_sums_valid(self) -> None:
        """Should return empty list for valid sums."""
        data = Sheet2Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            cobre_concentrados=100,
            cobre_catodos=20,
            oro_subproducto=15,
            plata_subproducto=5,
            total_ingresos=140,
        )
        issues = validate_sheet2_sums(data)
        assert issues == []

    def test_validate_sheet2_sums_mismatch(self) -> None:
        """Should return issue list for mismatched sums."""
        data = Sheet2Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            cobre_concentrados=100,
            cobre_catodos=20,
            oro_subproducto=15,
            plata_subproducto=5,
            total_ingresos=200,  # Wrong - should be 140
        )
        issues = validate_sheet2_sums(data)
        assert len(issues) > 0
        assert any("Sum validation" in issue or "diff" in issue for issue in issues)

    def test_validate_sheet2_sums_missing_values(self) -> None:
        """Should handle missing values gracefully."""
        data = Sheet2Data(
            quarter="IIQ2024",
            year=2024,
            quarter_num=2,
            cobre_concentrados=100,
            # Missing other revenue components
            total_ingresos=100,
        )
        # Should not raise, may or may not return issues depending on implementation
        issues = validate_sheet2_sums(data)
        assert isinstance(issues, list)


# =============================================================================
# Extraction and XBRL Integration Tests
# =============================================================================


class TestSheet2Extraction:
    """Tests for sheet2 extraction functionality."""

    def test_extract_returns_reference_data_for_known_periods(self) -> None:
        """extract_sheet2 should return data for periods with reference values."""
        from puco_eeff.sheets.sheet2 import extract_sheet2

        # Test all verified periods
        verified_periods = [(2024, 1), (2024, 2), (2024, 3), (2025, 1), (2025, 2), (2025, 3)]

        for year, quarter in verified_periods:
            data, issues = extract_sheet2(year, quarter)
            assert data is not None, f"Extraction failed for {year} Q{quarter}: {issues}"
            assert data.year == year
            assert data.quarter_num == quarter

    def test_extracted_data_matches_reference(self) -> None:
        """Extracted data should match reference values exactly."""
        from puco_eeff.sheets.sheet2 import extract_sheet2

        # Test Q2 2024 - a well-verified period
        data, _issues = extract_sheet2(2024, 2)
        ref = get_sheet2_reference_values(2024, 2)

        assert data is not None
        assert ref is not None

        # Check all value fields match reference
        fields_to_check = [
            "cobre_concentrados",
            "cobre_catodos",
            "oro_subproducto",
            "plata_subproducto",
            "total_ingresos",
            "ebitda",
            "libras_vendidas",
            "cobre_fino",
            "precio_efectivo",
            "cash_cost",
            "costo_unitario_total",
            "non_cash_cost",
            "toneladas_procesadas",
            "oro_onzas",
        ]

        for field in fields_to_check:
            extracted = data.get_value(field)
            expected = ref.get(field)
            assert extracted == expected, (
                f"Field {field} mismatch for 2024 Q2: extracted={extracted}, expected={expected}"
            )


class TestXbrlCrossValidation:
    """Tests for XBRL cross-validation of sheet2 total_ingresos."""

    def test_xbrl_revenue_within_tolerance(self) -> None:
        """XBRL Revenue should be within tolerance of reference total_ingresos."""
        from pathlib import Path

        from puco_eeff.extractor.xbrl_parser import parse_xbrl_file

        xbrl_dir = Path("data/raw/xbrl")
        if not xbrl_dir.exists():
            pytest.skip("XBRL directory not found")

        # Test periods that have both XBRL and reference data
        test_periods = [(2025, 2), (2025, 3)]
        tolerance_percent = 1.0  # 1% tolerance

        for year, quarter in test_periods:
            xbrl_path = xbrl_dir / f"estados_financieros_{year}_Q{quarter}.xbrl"
            if not xbrl_path.exists():
                continue

            ref = get_sheet2_reference_values(year, quarter)
            if ref is None or ref.get("total_ingresos") is None:
                continue

            # Parse XBRL and get YTD Revenue
            data = parse_xbrl_file(xbrl_path)
            contexts = data.get("contexts", {})

            # Find YTD revenue for current period (start of year to end of quarter)
            ytd_revenue = None
            for fact in data["facts"]:
                if fact.get("name") == "Revenue":
                    ctx = contexts.get(fact["context_ref"], {})
                    if ctx.get("start_date", "").startswith(f"{year}-01-01"):
                        try:
                            val = int(float(fact["value"])) // 1000
                            if ytd_revenue is None or val > ytd_revenue:
                                ytd_revenue = val
                        except (ValueError, TypeError):
                            pass

            if ytd_revenue is None:
                continue

            ref_total = ref.get("total_ingresos")
            if ref_total is None:
                continue

            # Type narrowing - ref_total is now int | float
            ref_total_num: int | float = ref_total
            diff_percent = abs(ytd_revenue - ref_total_num) / ref_total_num * 100

            assert diff_percent <= tolerance_percent, (
                f"{year} Q{quarter}: XBRL Revenue ({ytd_revenue:,}) differs from "
                f"reference total_ingresos ({ref_total:,}) by {diff_percent:.2f}%"
            )


class TestReferenceDataRegression:
    """Regression tests to ensure reference data doesn't accidentally change."""

    def test_q2_2024_reference_values(self) -> None:
        """Q2 2024 reference values should remain constant."""
        ref = get_sheet2_reference_values(2024, 2)
        assert ref is not None

        # These are the verified values from the CSV
        expected = {
            "cobre_concentrados": 142391,
            "cobre_catodos": 15022,
            "oro_subproducto": 20223,
            "plata_subproducto": 1529,
            "total_ingresos": 179165,
            "ebitda": 65483,
            "libras_vendidas": 38.4,
            "cobre_fino": 38.0,
            "precio_efectivo": 4.3,
            "cash_cost": 2.59,
            "costo_unitario_total": 3.57,
            "non_cash_cost": 0.98,
            "toneladas_procesadas": 2683.0,
            "oro_onzas": 8.2,
        }

        for field, expected_val in expected.items():
            actual_val = ref.get(field)
            assert actual_val == expected_val, (
                f"Q2 2024 {field}: expected {expected_val}, got {actual_val}"
            )

    def test_q3_2025_reference_values(self) -> None:
        """Q3 2025 reference values should remain constant (revenue fields only)."""
        ref = get_sheet2_reference_values(2025, 3)
        assert ref is not None

        # Key revenue values from the CSV (operational metrics may be null)
        expected_key_values = {
            "cobre_concentrados": 278551,
            "cobre_catodos": 27549,
            "oro_subproducto": 43959,
            "plata_subproducto": 2741,
            "total_ingresos": 352800,
        }

        for field, expected_val in expected_key_values.items():
            actual_val = ref.get(field)
            assert actual_val == expected_val, (
                f"Q3 2025 {field}: expected {expected_val}, got {actual_val}"
            )
