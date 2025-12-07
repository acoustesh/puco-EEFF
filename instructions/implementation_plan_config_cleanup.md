# Implementation Plan: Config-Driven Cleanup for cost_extractor.py

## Overview

This plan addresses the remaining hard-coded values and DRY violations identified in the gap analysis. The goal is to move all magic values to configuration files while maintaining backward compatibility.

**Scope:** This is a **sheet1-only** refactor. All changes are namespaced under sheet1 config and APIs.

---

## Phase 1: Config Schema Updates

### 1.1 Update extraction.json

Add new keys to support runtime configuration:

```json
// In config/sheet1/extraction.json sections:
{
  "sections": {
    "nota_21": {
      // existing fields...
      "fallback_section": null  // No fallback needed - explicit null
    },
    "nota_22": {
      // existing fields...
      "fallback_section": "nota_21"  // Falls back to nota_21's page
    },
    "ingresos": {
      // existing fields...
      "pdf_fallback": {
        // existing fields...
        "min_value_threshold": 1000  // NEW: Replaces hard-coded > 1000
        // search_keywords already defined via field_mappings.ingresos_ordinarios.match_keywords
      }
    }
  }
}
```

**Config validation:** Code must fail fast with clear message if required keys missing.

---

## Phase 2: New Config Accessor in sheet1.py

### 2.1 Add `get_section_config()` - The Canonical Accessor

```python
def get_section_config(section_name: str, *, sheet: str = "sheet1") -> dict[str, Any]:
    """Get full section config with validation.

    This is the canonical accessor for section configuration. Validates that
    required keys exist and raises clear errors if config is malformed.

    Args:
        section_name: Section key (e.g., "nota_21", "nota_22", "ingresos")
        sheet: Sheet name (currently only "sheet1" supported)

    Returns:
        Full section configuration dictionary.

    Raises:
        ValueError: If section not found or sheet not supported.
        KeyError: If required config keys are missing.
    """
    if sheet != "sheet1":
        raise ValueError(f"Sheet '{sheet}' not supported. Only 'sheet1' is implemented.")

    section = get_sheet1_section_spec(section_name)

    # Validate required keys exist
    required_keys = ["title", "field_mappings"]
    missing = [k for k in required_keys if k not in section]
    if missing:
        raise KeyError(f"Section '{section_name}' missing required keys: {missing}")

    return section
```

### 2.2 Add Convenience Helpers

```python
def get_section_fallback(section_name: str) -> str | None:
    """Get fallback section for page lookup.

    Args:
        section_name: Section key (e.g., "nota_22")

    Returns:
        Fallback section name, or None if no fallback configured.

    Raises:
        KeyError: If fallback_section key is missing from config.
    """
    section = get_section_config(section_name)
    if "fallback_section" not in section:
        raise KeyError(
            f"Section '{section_name}' missing 'fallback_section' key. "
            f"Add it to config/sheet1/extraction.json (use null for no fallback)."
        )
    return section.get("fallback_section")


def get_ingresos_pdf_fallback_config() -> dict[str, Any]:
    """Get ingresos PDF fallback extraction configuration.

    Returns:
        Dictionary with min_value_threshold, search_patterns, etc.

    Raises:
        KeyError: If required keys missing from config.
    """
    section = get_section_config("ingresos")
    pdf_fallback = section.get("pdf_fallback", {})

    # Validate required keys
    if "min_value_threshold" not in pdf_fallback:
        raise KeyError(
            "ingresos.pdf_fallback missing 'min_value_threshold' key. "
            "Add it to config/sheet1/extraction.json."
        )

    return pdf_fallback
```

### 2.3 Deprecate Redundant Getters

These getters add no logic beyond dict access - mark as deprecated but keep for compat:

```python
# In sheet1.py - add deprecation notices

def get_sheet1_extraction_sections() -> list[str]:
    """Get list of all extraction section keys for Sheet1.

    .. deprecated:: 0.x
        Use ``list(get_sheet1_extraction_config()["sections"].keys())`` directly.
    """
    # ... existing implementation
```

**Getters to deprecate** (they just do `config.get("key", default)`):
- `get_sheet1_extraction_sections()` - simple dict.keys()
- Keep getters that add validation logic or transform data.

---

## Phase 3: Refactor cost_extractor.py

### 3.1 Replace Hard-coded Fallback Map (line ~1017)

**Location:** `extract_section_breakdown()` function

**Before:**
```python
fallback_sections = {
    "nota_22": "nota_21",  # Nota 22 often on same page as Nota 21
}
if section_name in fallback_sections:
    fallback = fallback_sections[section_name]
```

**After:**
```python
from puco_eeff.sheets.sheet1 import get_section_fallback

fallback = get_section_fallback(section_name)
if fallback:
    page_idx = find_section_page(pdf_path, fallback, year, quarter)
```

### 3.2 Replace Hard-coded Threshold (lines ~1138, 1145)

**Location:** `extract_ingresos_from_pdf()` function

**Before:**
```python
if value is not None and value > 1000:
```

**After:**
```python
from puco_eeff.sheets.sheet1 import get_ingresos_pdf_fallback_config

pdf_config = get_ingresos_pdf_fallback_config()
min_threshold = pdf_config["min_value_threshold"]  # No default - must be in config
if value is not None and value > min_threshold:
```

### 3.3 Hard-coded XBRL Keys (lines ~1516-1517)

**Location:** `extract_detailed_costs()` function

**Before:**
```python
result.xbrl_totals["cost_of_sales"] = xbrl_totals.get("cost_of_sales")
result.xbrl_totals["admin_expense"] = xbrl_totals.get("admin_expense")
```

**After:**
```python
from puco_eeff.sheets.sheet1 import get_sheet1_result_key_mapping

# Copy all XBRL totals using config-driven keys
result_key_mapping = get_sheet1_result_key_mapping()
for field_name, xbrl_key in result_key_mapping.items():
    if xbrl_key in xbrl_totals:
        result.xbrl_totals[xbrl_key] = xbrl_totals.get(xbrl_key)
```

### 3.4 Period Label Consistency (line ~1385)

**Location:** `_section_breakdowns_to_sheet1data()` function

**Before:**
```python
quarter_label = format_quarter_label(year, quarter) if year and quarter else f"{year}Q{quarter}"
```

**After:**
```python
# Always use format_quarter_label for consistency
quarter_label = format_quarter_label(year, quarter)
```

---

## Phase 4: Deprecate Duplicate Functions

### 4.1 `_section_breakdowns_to_sheet1data()` (line ~1361)

This duplicates `sections_to_sheet1data()`. Add deprecation:

```python
def _section_breakdowns_to_sheet1data(...) -> Sheet1Data:
    """Convert SectionBreakdown objects to Sheet1Data for validation.

    .. deprecated:: 0.x
        Use :func:`sections_to_sheet1data` from puco_eeff.sheets.sheet1 instead
        for full config-driven conversion with detail fields.
    """
    import warnings
    warnings.warn(
        "_section_breakdowns_to_sheet1data() is deprecated. "
        "Use sections_to_sheet1data() from puco_eeff.sheets.sheet1 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Keep minimal implementation for backward compat with validate_extraction()
    ...
```

### 4.2 `_map_nota21_item_to_sheet1()` and `_map_nota22_item_to_sheet1()` (lines ~1881, 1891)

These are thin wrappers around `_map_nota_item_to_sheet1()`. Add deprecation notices:

```python
def _map_nota21_item_to_sheet1(item: LineItem, data: Sheet1Data) -> None:
    """Map a Nota 21 line item to Sheet1Data fields.

    .. deprecated:: 0.x
        Use :func:`_map_nota_item_to_sheet1` with section_name="nota_21" directly,
        or use :func:`sections_to_sheet1data` for complete conversion.
    """
    _map_nota_item_to_sheet1(item, data, "nota_21")
```

---

## Phase 5: Validation Namespace

### 5.1 Keep Validation Under sheet1 Namespace

Validation functions stay in cost_extractor.py but are accessed via sheet1 module:

**In `puco_eeff/sheets/sheet1.py`:**
```python
# Re-export validation functions for convenience
from puco_eeff.extractor.cost_extractor import (
    run_sheet1_validations,
    ValidationReport,
    ValidationResult,
    SumValidationResult,
)
```

### 5.2 Deprecated Re-exports in cost_extractor.py

For functions moved to sheet1.py, add thin re-exports with deprecation:

```python
# At module level in cost_extractor.py
def validate_extraction(...):
    """Deprecated - use run_sheet1_validations() instead."""
    import warnings
    warnings.warn(
        "validate_extraction() is deprecated. Use run_sheet1_validations() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation
```

---

## Phase 6: Documentation & Tests

### 6.1 Update Config Documentation

**In `instructions/data_mapping.md`:**
- Document new `fallback_section` key
- Document `min_value_threshold` key
- Explain config validation requirements

### 6.2 Add Tests

| Test | Description |
|------|-------------|
| `test_get_section_config_valid` | Valid section returns config |
| `test_get_section_config_invalid_sheet` | Non-sheet1 raises ValueError |
| `test_get_section_config_missing_keys` | Missing required keys raises KeyError |
| `test_get_section_fallback_nota22` | Returns "nota_21" for nota_22 |
| `test_get_section_fallback_nota21` | Returns None for nota_21 |
| `test_get_section_fallback_missing_key` | Raises KeyError if key missing |
| `test_get_ingresos_pdf_fallback_config` | Returns threshold config |
| `test_extract_section_uses_config_fallback` | Integration test |
| `test_ingresos_uses_config_threshold` | Integration test |

### 6.3 Update Docstrings

All deprecated functions must have:
- `.. deprecated::` directive in docstring
- `warnings.warn()` call in implementation
- Reference to replacement function

---

## Implementation Order

1. **✅ Config changes** - Added `fallback_section` and `min_value_threshold` to extraction.json
2. **✅ New accessor** - Added `get_section_config()` to sheet1.py
3. **✅ Convenience helpers** - Added `get_section_fallback()`, `get_ingresos_pdf_fallback_config()`
4. **✅ Refactor cost_extractor.py** - Replaced 4 hard-coded locations
5. **✅ Add deprecation warnings** - Marked `_section_breakdowns_to_sheet1data`, `_map_nota21_item_to_sheet1`, `_map_nota22_item_to_sheet1`
6. **✅ Re-exports** - Added validation re-exports in sheet1.py (lazy import to avoid circular deps)
7. **✅ Tests** - Added 11 tests for new config-driven behavior (208 total)
8. **✅ Documentation** - Updated data_mapping.md with new config keys

---

## Files to Modify

| File | Changes |
|------|---------|
| `config/sheet1/extraction.json` | Add `fallback_section` to all sections, `min_value_threshold` to ingresos |
| `puco_eeff/sheets/sheet1.py` | Add `get_section_config()`, `get_section_fallback()`, `get_ingresos_pdf_fallback_config()`, validation re-exports |
| `puco_eeff/extractor/cost_extractor.py` | Replace 4 hard-coded locations, add deprecation warnings |
| `tests/test_cost_extractor.py` | Add ~8 tests for new config-driven behavior |
| `instructions/data_mapping.md` | Document new config keys |

---

## Backward Compatibility

- All deprecated functions emit `DeprecationWarning`
- Existing function signatures unchanged
- **No defaults for new config keys** - fail fast if missing
- No breaking changes to public API
- Re-exports maintain old import paths
