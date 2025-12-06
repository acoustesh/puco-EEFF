"""Configuration management for puco-EEFF.

Handles environment variables, API clients, and config file loading.
Supports split configuration:
- config.json: Shared project config (sources, file patterns, period types)
- extraction_specs.json: PDF extraction rules (search patterns, field mappings)
- xbrl_specs.json: XBRL-specific config (fact names, validation, scaling)
- reference_data.json: Known-good values for validation
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
AUDIT_DIR = Path(os.getenv("AUDIT_DIR", PROJECT_ROOT / "audit"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", PROJECT_ROOT / "temp"))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def get_config() -> dict[str, Any]:
    """Load configuration from config.json."""
    config_path = CONFIG_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def get_period_paths(year: int, quarter: int) -> dict[str, Path]:
    """Get paths for a specific period (year/quarter).

    Args:
        year: The year (e.g., 2024)
        quarter: The quarter (1-4)

    Returns:
        Dictionary with paths for raw, processed, output, and audit directories.
    """
    period_str = f"{year}_Q{quarter}"

    return {
        "raw_pdf": DATA_DIR / "raw" / "pdf",
        "raw_xbrl": DATA_DIR / "raw" / "xbrl",
        "processed": DATA_DIR / "processed",
        "output": DATA_DIR / "output",
        "audit": AUDIT_DIR / period_str,
    }


def setup_logging(name: str = "puco_eeff") -> logging.Logger:
    """Set up logging for a run.

    Args:
        name: Logger name

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        log_filename = f"{datetime.now().strftime('%Y-%m-%d')}_run.log"
        file_handler = logging.FileHandler(LOGS_DIR / log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def validate_api_keys() -> dict[str, bool]:
    """Check which API keys are configured.

    Returns:
        Dictionary mapping API name to whether it's configured.
    """
    return {
        "mistral": bool(MISTRAL_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY),
        "openai": bool(OPENAI_API_KEY),
    }


def get_mistral_client() -> Any:
    """Get Mistral AI client.

    Returns:
        Configured Mistral client.

    Raises:
        ValueError: If MISTRAL_API_KEY is not set.
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set")

    from mistralai import Mistral

    return Mistral(api_key=MISTRAL_API_KEY)


def get_openrouter_client() -> Any:
    """Get OpenRouter client (uses OpenAI SDK).

    Returns:
        Configured OpenAI client pointing to OpenRouter.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set")

    from openai import OpenAI

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


# =============================================================================
# Split Configuration Loaders
# =============================================================================


def get_extraction_specs() -> dict[str, Any]:
    """Load extraction specifications from extraction_specs.json.

    Returns:
        Dictionary with default template and per-quarter specs.
    """
    specs_path = CONFIG_DIR / "extraction_specs.json"
    if not specs_path.exists():
        raise FileNotFoundError(f"Extraction specs not found: {specs_path}")

    with open(specs_path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def get_reference_data() -> dict[str, Any]:
    """Load reference data from reference_data.json.

    Returns:
        Dictionary with reference values for validation.
    """
    ref_path = CONFIG_DIR / "reference_data.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference data not found: {ref_path}")

    with open(ref_path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def get_period_specs(year: int, quarter: int) -> dict[str, Any]:
    """Get extraction specs for a specific period, merging default with deviations.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)

    Returns:
        Merged extraction specs (default + period-specific deviations)
    """
    specs = get_extraction_specs()
    default = specs.get("default", {})

    # Look for period-specific specs
    period_key = f"{year}_Q{quarter}"
    period_specs = specs.get(period_key, {})

    # Deep merge: start with default, overlay period-specific deviations
    merged = _deep_merge(default.copy(), period_specs.get("deviations", {}))

    # Add period metadata
    merged["_period"] = period_key
    merged["_verified"] = period_specs.get("verified", False)
    merged["_page_numbers"] = period_specs.get("page_numbers", {})

    return merged


def get_reference_values(year: int, quarter: int) -> dict[str, int] | None:
    """Get reference values for a specific period.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)

    Returns:
        Dictionary of reference values, or None if not available.
    """
    ref_data = get_reference_data()
    period_key = f"{year}_Q{quarter}"

    period_data = ref_data.get(period_key, {})
    if period_data.get("verified") and period_data.get("values"):
        return period_data["values"]

    return None


# =============================================================================
# XBRL Configuration Loader
# =============================================================================


def get_xbrl_specs() -> dict[str, Any]:
    """Load XBRL specifications from xbrl_specs.json.

    Returns:
        Dictionary with XBRL fact mappings, validation rules, and namespaces.
    """
    xbrl_path = CONFIG_DIR / "xbrl_specs.json"
    if not xbrl_path.exists():
        raise FileNotFoundError(f"XBRL specs not found: {xbrl_path}")

    with open(xbrl_path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def get_xbrl_fact_mapping(field_name: str) -> dict[str, Any] | None:
    """Get XBRL fact mapping for a specific field.

    Args:
        field_name: Field name (e.g., "ingresos_ordinarios", "total_costo_venta")

    Returns:
        Dictionary with primary, fallbacks, and other XBRL info, or None.
    """
    xbrl_specs = get_xbrl_specs()
    return xbrl_specs.get("fact_mappings", {}).get(field_name)


def get_xbrl_scaling_factor() -> int:
    """Get XBRL scaling factor from config.

    Returns:
        Scaling factor (default 1000 for MUS$ conversion).
    """
    xbrl_specs = get_xbrl_specs()
    return xbrl_specs.get("scaling_factor", 1000)


def get_validation_rules() -> dict[str, Any]:
    """Get validation rules from XBRL specs.

    Returns:
        Dictionary with sum_tolerance, total_validations, cross_validations.
    """
    xbrl_specs = get_xbrl_specs()
    return xbrl_specs.get("validation_rules", {})


def get_sum_tolerance() -> int:
    """Get sum tolerance for validation (allows for rounding differences).

    Returns:
        Tolerance value (default 1).
    """
    rules = get_validation_rules()
    return rules.get("sum_tolerance", 1)


# =============================================================================
# Period Formatting Functions
# =============================================================================


def get_period_type_config(period_type: str = "quarterly") -> dict[str, Any]:
    """Get period type configuration.

    Args:
        period_type: One of "quarterly", "monthly", "yearly"

    Returns:
        Period type configuration dictionary.
    """
    config = get_config()
    period_types = config.get("period_types", {})
    return period_types.get(period_type, period_types.get("quarterly", {}))


def format_period_key(
    year: int,
    period: int,
    period_type: str = "quarterly",
) -> str:
    """Format a period key string (e.g., "2024_Q2", "2024_M06", "2024_FY").

    Args:
        year: Year
        period: Period number (quarter 1-4, month 1-12, or 1 for yearly)
        period_type: One of "quarterly", "monthly", "yearly"

    Returns:
        Formatted period key string.
    """
    if period_type == "quarterly":
        return f"{year}_Q{period}"
    elif period_type == "monthly":
        return f"{year}_M{period:02d}"
    elif period_type == "yearly":
        return f"{year}_FY"
    else:
        # Default to quarterly format
        return f"{year}_Q{period}"


def format_period_display(
    year: int,
    period: int,
    period_type: str = "quarterly",
) -> str:
    """Format a period for display (e.g., "IIQ2024", "06-2024", "FY2024").

    Args:
        year: Year
        period: Period number
        period_type: One of "quarterly", "monthly", "yearly"

    Returns:
        Formatted display string.
    """
    type_config = get_period_type_config(period_type)

    if period_type == "quarterly":
        roman_map = type_config.get("roman_numerals", {"1": "I", "2": "II", "3": "III", "4": "IV"})
        roman = roman_map.get(str(period), str(period))
        return f"{roman}Q{year}"
    elif period_type == "monthly":
        return f"{period:02d}-{year}"
    elif period_type == "yearly":
        return f"FY{year}"
    else:
        return f"{period}_{year}"


def parse_period_key(period_key: str) -> tuple[int, int, str]:
    """Parse a period key into its components.

    Args:
        period_key: Period key string (e.g., "2024_Q2", "2024_M06", "2024_FY")

    Returns:
        Tuple of (year, period, period_type)

    Raises:
        ValueError: If period key format is not recognized.
    """
    # Quarterly: 2024_Q2
    quarterly_match = re.match(r"(\d{4})_Q(\d)", period_key)
    if quarterly_match:
        return int(quarterly_match.group(1)), int(quarterly_match.group(2)), "quarterly"

    # Monthly: 2024_M06
    monthly_match = re.match(r"(\d{4})_M(\d{2})", period_key)
    if monthly_match:
        return int(monthly_match.group(1)), int(monthly_match.group(2)), "monthly"

    # Yearly: 2024_FY
    yearly_match = re.match(r"(\d{4})_FY", period_key)
    if yearly_match:
        return int(yearly_match.group(1)), 1, "yearly"

    raise ValueError(f"Unrecognized period key format: {period_key}")


# =============================================================================
# File Pattern Functions
# =============================================================================


def get_file_pattern(file_type: str) -> str:
    """Get file pattern for a specific file type.

    Args:
        file_type: One of "analisis_razonado", "estados_financieros_pdf",
                   "estados_financieros_xbrl", "xbrl_zip", "pucobre_combined"

    Returns:
        File pattern string with placeholders.
    """
    config = get_config()
    patterns = config.get("file_patterns", {})
    file_config = patterns.get(file_type, {})
    return file_config.get("pattern", f"{file_type}_{{year}}_Q{{quarter}}")


def get_file_pattern_alternatives(file_type: str) -> list[str]:
    """Get alternative file patterns for a file type.

    Args:
        file_type: File type key

    Returns:
        List of alternative patterns (may be empty).
    """
    config = get_config()
    patterns = config.get("file_patterns", {})
    file_config = patterns.get(file_type, {})
    return file_config.get("alt_patterns", [])


def format_filename(
    file_type: str,
    year: int,
    quarter: int | None = None,
    period: int | None = None,
    period_type: str = "quarterly",
) -> str:
    """Format a filename using the configured pattern.

    Args:
        file_type: File type key
        year: Year
        quarter: Quarter number (for backward compatibility)
        period: Period number (alternative to quarter)
        period_type: Period type (for non-quarterly patterns)

    Returns:
        Formatted filename.
    """
    pattern = get_file_pattern(file_type)

    # Use quarter if provided, otherwise use period
    period_num = quarter if quarter is not None else (period if period is not None else 1)

    return pattern.format(
        year=year,
        quarter=period_num,
        period=period_num,
    )


def find_file_with_alternatives(
    base_dir: Path,
    file_type: str,
    year: int,
    quarter: int,
) -> Path | None:
    """Find a file using primary and alternative patterns.

    Args:
        base_dir: Directory to search in
        file_type: File type key
        year: Year
        quarter: Quarter number

    Returns:
        Path to found file, or None if not found.
    """
    # Try primary pattern
    primary_name = format_filename(file_type, year, quarter)
    primary_path = base_dir / primary_name
    if primary_path.exists():
        return primary_path

    # Try alternative patterns
    for alt_pattern in get_file_pattern_alternatives(file_type):
        alt_name = alt_pattern.format(year=year, quarter=quarter)
        alt_path = base_dir / alt_name
        if alt_path.exists():
            return alt_path

    return None


# =============================================================================
# Sheet1 Field Configuration
# =============================================================================


def get_sheet1_fields() -> dict[str, Any]:
    """Get Sheet1 field definitions from extraction_specs.json.

    Returns:
        Dictionary with metadata_fields and value_fields.
    """
    specs = get_extraction_specs()
    return specs.get("sheet1_fields", {})


def get_sheet1_value_fields() -> dict[str, dict[str, Any]]:
    """Get Sheet1 value field definitions.

    Returns:
        Dictionary mapping field names to their definitions.
    """
    fields = get_sheet1_fields()
    return fields.get("value_fields", {})


def get_sheet1_metadata_fields() -> list[str]:
    """Get Sheet1 metadata field names.

    Returns:
        List of metadata field names.

    Raises:
        ValueError: If metadata_fields not found in config.
    """
    fields = get_sheet1_fields()
    metadata = fields.get("metadata_fields")
    if metadata is None:
        raise ValueError("metadata_fields not found in extraction_specs.json sheet1_fields")
    return metadata


def get_sheet1_row_mapping() -> dict[str, dict[str, Any]]:
    """Get Sheet1 row mapping from config.json.

    Returns the row_mapping from config.json sheets.sheet1.row_mapping,
    which defines all 27 rows with field names, labels, and sections.

    Returns:
        Dictionary mapping row numbers (as strings) to row definitions.

    Raises:
        ValueError: If row_mapping not found in config.
    """
    config = get_config()
    sheets = config.get("sheets", {})
    sheet1 = sheets.get("sheet1", {})
    row_mapping = sheet1.get("row_mapping")
    if row_mapping is None:
        raise ValueError("row_mapping not found in config.json sheets.sheet1")
    return row_mapping


def get_fields_for_section(section_name: str) -> list[str]:
    """Get field names belonging to a specific section.

    Args:
        section_name: Section name ("nota_21", "nota_22", "ingresos")

    Returns:
        List of field names in that section.
    """
    value_fields = get_sheet1_value_fields()
    return [field_name for field_name, field_def in value_fields.items() if field_def.get("section") == section_name]


def get_section_field_mappings(section_name: str) -> dict[str, dict[str, Any]]:
    """Get field mappings for a specific section from extraction_specs.json.

    Args:
        section_name: Section name ("nota_21", "nota_22", "ingresos")

    Returns:
        Dictionary of field mappings with match_keywords, exclude_keywords, etc.
    """
    specs = get_extraction_specs()
    default = specs.get("default", {})
    sections = default.get("sections", {})
    section = sections.get(section_name, {})
    return section.get("field_mappings", {})


def match_concepto_to_field(
    concepto: str,
    section_name: str,
) -> str | None:
    """Match a concepto string to a Sheet1 field using config-driven keywords.

    Uses match_keywords (all must match) and exclude_keywords (none can match)
    from extraction_specs.json to determine which field a concepto belongs to.

    Args:
        concepto: The concepto string from the PDF (will be lowercased)
        section_name: Section name ("nota_21", "nota_22", "ingresos")

    Returns:
        Field name if matched, None otherwise.
    """
    concepto_lower = concepto.lower()
    field_mappings = get_section_field_mappings(section_name)

    for field_name, mapping in field_mappings.items():
        match_keywords = mapping.get("match_keywords", [])
        exclude_keywords = mapping.get("exclude_keywords", [])

        # Check exclude keywords first - if any match, skip this field
        if exclude_keywords and any(kw.lower() in concepto_lower for kw in exclude_keywords):
            continue

        # Check match keywords - at least one must match
        # (for compound keywords like "materiales, repuestos", ANY of them matching is sufficient)
        if match_keywords and any(kw.lower() in concepto_lower for kw in match_keywords):
            return mapping.get("sheet1_field", field_name)

    return None


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge two dictionaries, with overlay values taking precedence.

    Args:
        base: Base dictionary
        overlay: Dictionary to overlay on base

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# PDF Page Extraction Utility
# =============================================================================


def extract_pdf_page_to_temp(
    pdf_path: Path,
    page_number: int,
    prefix: str = "page_review_",
) -> Path:
    """Extract a single page from PDF to a temporary file for user review.

    This is used when automatic extraction fails and user needs to
    manually review a specific page to help debug the issue.

    Args:
        pdf_path: Path to the source PDF
        page_number: 1-indexed page number to extract
        prefix: Prefix for the temp file name

    Returns:
        Path to the temporary PDF file containing just that page

    Raises:
        ImportError: If pypdf is not installed
        ValueError: If page number is out of range
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        raise ImportError("pypdf is required for page extraction. Install with: pip install pypdf")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if page_number < 1 or page_number > total_pages:
        raise ValueError(f"Page {page_number} out of range (1-{total_pages})")

    # Create writer with just the requested page (0-indexed internally)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])

    # Create temp file in TEMP_DIR
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = TEMP_DIR / f"{prefix}{pdf_path.stem}_p{page_number}.pdf"

    with open(temp_path, "wb") as f:
        writer.write(f)

    return temp_path


def cleanup_temp_files(prefix: str = "page_review_") -> int:
    """Clean up temporary files created for page review.

    Args:
        prefix: Prefix of files to clean up

    Returns:
        Number of files deleted
    """
    count = 0
    if TEMP_DIR.exists():
        for f in TEMP_DIR.glob(f"{prefix}*.pdf"):
            f.unlink()
            count += 1
    return count
