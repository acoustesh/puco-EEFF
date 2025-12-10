"""Configuration management for puco-EEFF.

This module centralizes file-system paths, environment variables, and split
configuration loaders used by the scraping and extraction pipeline.

Split configuration files
-------------------------
* ``config.json``: shared project config (sources, file patterns, period types)
* ``extraction_specs.json``: PDF extraction rules (search patterns, field mappings)
* ``xbrl_specs.json``: XBRL-specific config (fact names, validation, scaling)
* ``reference_data.json``: Known-good values for validation

Environment variables
---------------------
``DATA_DIR``, ``AUDIT_DIR``, ``LOGS_DIR``, and ``TEMP_DIR`` override default
directories; OCR providers rely on ``MISTRAL_API_KEY`` and ``OPENROUTER_API_KEY``
while OpenAI-compatible fallbacks use ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY``.
Directories are created eagerly on import so downstream callers can rely on
their existence.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
    """Load the primary project configuration.

    Returns
    -------
    dict[str, Any]
        Parsed contents of ``config/config.json`` including period types and
        file pattern templates.

    Raises
    ------
    FileNotFoundError
        If ``config/config.json`` is missing.
    json.JSONDecodeError
        If the file exists but is not valid JSON.
    """
    config_path = CONFIG_DIR / "config.json"
    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with Path(config_path).open(encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def get_period_paths(year: int, quarter: int) -> dict[str, Path]:
    """Return per-period paths for raw, processed, output, and audit data.

    Parameters
    ----------
    year : int
        Four-digit fiscal year (e.g., 2024).
    quarter : int
        Quarter number in ``{1, 2, 3, 4}``.

    Returns
    -------
    dict[str, Path]
        Mapping with keys ``raw_pdf``, ``raw_xbrl``, ``processed``, ``output``,
        and ``audit`` (the latter names a subdirectory under ``AUDIT_DIR``).
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
    """Configure a console+file logger if not already present.

    Parameters
    ----------
    name : str, optional
        Logger namespace; reused to avoid duplicate handlers.

    Returns
    -------
    logging.Logger
        Logger with INFO-level console handler and DEBUG-level rotating file
        handler under ``LOGS_DIR``.
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
        log_filename = f"{datetime.now(UTC).strftime('%Y-%m-%d')}_run.log"
        file_handler = logging.FileHandler(LOGS_DIR / log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def validate_api_keys() -> dict[str, bool]:
    """Report availability of optional third-party API keys.

    Returns
    -------
    dict[str, bool]
        Flags for ``mistral``, ``anthropic``, ``openrouter``, and ``openai``
        indicating whether corresponding environment variables are set.
    """
    return {
        "mistral": bool(MISTRAL_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY),
        "openai": bool(OPENAI_API_KEY),
    }


def get_mistral_client() -> Any:
    """Instantiate the synchronous Mistral SDK client.

    Returns
    -------
    mistralai.Mistral
        Client configured with ``MISTRAL_API_KEY`` for chat/completions.

    Raises
    ------
    ValueError
        If ``MISTRAL_API_KEY`` is absent.
    """
    if not MISTRAL_API_KEY:
        msg = "MISTRAL_API_KEY is not set"
        raise ValueError(msg)

    from mistralai import Mistral

    return Mistral(api_key=MISTRAL_API_KEY)


def get_openrouter_client() -> Any:
    """Instantiate an OpenAI-compatible client against OpenRouter.

    Returns
    -------
    openai.OpenAI
        Client configured with ``OPENROUTER_API_KEY`` and OpenRouter base URL.

    Raises
    ------
    ValueError
        If ``OPENROUTER_API_KEY`` is absent.
    """
    if not OPENROUTER_API_KEY:
        msg = "OPENROUTER_API_KEY is not set"
        raise ValueError(msg)

    from openai import OpenAI

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


# =============================================================================
# Split Configuration Loaders
# =============================================================================


def get_extraction_specs() -> dict[str, Any]:
    """Load PDF extraction specifications from ``extraction_specs.json``.

    Returns
    -------
    dict[str, Any]
        Parsed extraction spec containing number formats, search strategy, and
        document structure hints.

    Raises
    ------
    FileNotFoundError
        If the specs file is missing.
    json.JSONDecodeError
        If the specs file cannot be parsed.
    """
    specs_path = CONFIG_DIR / "extraction_specs.json"
    if not specs_path.exists():
        msg = f"Extraction specs not found: {specs_path}"
        raise FileNotFoundError(msg)

    with Path(specs_path).open(encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


# =============================================================================
# XBRL Configuration Loader
# =============================================================================


def get_xbrl_specs() -> dict[str, Any]:
    """Load XBRL configuration from ``xbrl_specs.json``.

    Returns
    -------
    dict[str, Any]
        Scaling factor, namespaces, and period filters. Sheet-specific mappings
        live under ``config/<sheet>/xbrl_mappings.json``.

    Raises
    ------
    FileNotFoundError
        If the specs file is missing.
    json.JSONDecodeError
        If the specs file cannot be parsed.
    """
    xbrl_path = CONFIG_DIR / "xbrl_specs.json"
    if not xbrl_path.exists():
        msg = f"XBRL specs not found: {xbrl_path}"
        raise FileNotFoundError(msg)

    with Path(xbrl_path).open(encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def get_xbrl_scaling_factor() -> int:
    """Return the numeric scaling factor applied to XBRL facts.

    Returns
    -------
    int
        Factor used to down-scale values (default ``1000`` for MUS$ conversion).
    """
    xbrl_specs = get_xbrl_specs()
    return cast("int", xbrl_specs.get("scaling_factor", 1000))


def get_xbrl_namespaces() -> dict[str, str]:
    """Return namespace prefixes expected in XBRL documents.

    Returns
    -------
    dict[str, str]
        Mapping from prefix to namespace URI.
    """
    xbrl_specs = get_xbrl_specs()
    return cast("dict[str, str]", xbrl_specs.get("namespaces", {}))


# =============================================================================
# Period Formatting Functions
# =============================================================================


def get_period_type_config(period_type: str = "quarterly") -> dict[str, Any]:
    """Retrieve configuration for the requested period granularity.

    Parameters
    ----------
    period_type : str, optional
        One of ``"quarterly"``, ``"monthly"``, or ``"yearly"``.

    Returns
    -------
    dict[str, Any]
        Period-type configuration falling back to ``quarterly`` if missing.
    """
    config = get_config()
    period_types = config.get("period_types", {})
    return cast("dict[str, Any]", period_types.get(period_type, period_types.get("quarterly", {})))


def _get_roman_map(period_type: str = "quarterly") -> dict[str, str]:
    """Return the Roman numeral mapping for the configured period type."""
    type_config = get_period_type_config(period_type)
    return type_config.get("roman_numerals", {"1": "I", "2": "II", "3": "III", "4": "IV"})


def quarter_to_roman(quarter: int) -> str:
    """Convert a quarter number to its configured Roman numeral.

    Parameters
    ----------
    quarter : int
        Quarter index in ``{1, 2, 3, 4}``.

    Returns
    -------
    str
        Roman numeral label (defaults to ``I``–``IV`` unless overridden in
        config).

    Raises
    ------
    ValueError
        If ``quarter`` is outside the 1–4 range.
    """
    if quarter not in {1, 2, 3, 4}:
        msg = f"Invalid quarter: {quarter}. Must be 1-4."
        raise ValueError(msg)
    return cast("str", _get_roman_map()[str(quarter)])


def format_period(
    year: int,
    period: int,
    period_type: str = "quarterly",
    style: str = "key",
) -> str:
    """Format period identifiers for filenames or display labels.

    Parameters
    ----------
    year : int
        Fiscal year component.
    period : int
        Quarter (1–4), month (1–12), or ``1`` for yearly periods.
    period_type : str, optional
        One of ``"quarterly"``, ``"monthly"``, or ``"yearly"``.
    style : str, optional
        ``"key"`` for machine-readable keys or ``"display"`` for user-facing labels.

    Returns
    -------
    str
        Formatted period string such as ``"2024_QII"`` or ``"IIQ2024"``.

    Examples
    --------
    >>> format_period(2024, 2, "quarterly", "key")
    '2024_QII'
    >>> format_period(2024, 2, "quarterly", "display")
    'IIQ2024'
    >>> format_period(2024, 6, "monthly", "key")
    '2024_M06'
    """
    use_key_style = style == "key"

    if period_type == "quarterly":
        roman_numeral = _get_roman_map()[str(period)]
        return f"{year}_Q{roman_numeral}" if use_key_style else f"{roman_numeral}Q{year}"

    if period_type == "monthly":
        return f"{year}_M{period:02d}" if use_key_style else f"{period:02d}-{year}"

    if period_type == "yearly":
        return f"{year}_FY" if use_key_style else f"FY{year}"

    # Default fallback for unrecognized period types
    return f"{year}_Q{period}" if use_key_style else f"{period}_{year}"


def format_quarter_label(year: int, quarter: int) -> str:
    """Format a quarter for display (e.g., ``"IIQ2024"``)."""
    return format_period(year, quarter, "quarterly", "display")


def parse_period_key(period_key: str) -> tuple[int, int, str]:
    """Parse a period key into year, period number, and period type.

    Parameters
    ----------
    period_key : str
        Key formatted by :func:`format_period` (e.g., ``"2024_Q2"`` or ``"2024_M06"``).

    Returns
    -------
    tuple[int, int, str]
        Year, period number, and period type ``("quarterly"|"monthly"|"yearly")``.

    Raises
    ------
    ValueError
        If the key does not match any known pattern.
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

    msg = f"Unrecognized period key format: {period_key}"
    raise ValueError(msg)


# =============================================================================
# File Pattern Functions
# =============================================================================


def get_file_pattern(file_type: str) -> str:
    """Return the configured filename pattern for a document type.

    Parameters
    ----------
    file_type : str
        One of ``analisis_razonado``, ``estados_financieros_pdf``,
        ``estados_financieros_xbrl``, ``xbrl_zip``, or ``pucobre_combined``.

    Returns
    -------
    str
        Pattern string containing ``{year}`` and ``{quarter}`` placeholders.
    """
    config = get_config()
    patterns = config.get("file_patterns", {})
    file_config = patterns.get(file_type, {})
    return cast("str", file_config.get("pattern", f"{file_type}_{{year}}_Q{{quarter}}"))


def get_file_pattern_alternatives(file_type: str) -> list[str]:
    """Return alternative filename patterns for a document type.

    Parameters
    ----------
    file_type : str
        File type key as used in ``config.json``.

    Returns
    -------
    list[str]
        Optional pattern overrides, possibly empty.
    """
    config = get_config()
    patterns = config.get("file_patterns", {})
    file_config = patterns.get(file_type, {})
    return cast("list[str]", file_config.get("alt_patterns", []))


def format_filename(
    file_type: str,
    year: int,
    quarter: int | None = None,
    period: int | None = None,
    period_type: str = "quarterly",
) -> str:
    """Render a filename from configured patterns and period metadata.

    Parameters
    ----------
    file_type : str
        File type key (see :func:`get_file_pattern`).
    year : int
        Four-digit year.
    quarter : int, optional
        Quarter number used by legacy callers.
    period : int, optional
        Alternative period number for non-quarterly files.
    period_type : str, optional
        Period type label stored in the pattern config.

    Returns
    -------
    str
        Filename with placeholders substituted.
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
    """Locate a document using primary and alternative naming patterns.

    Parameters
    ----------
    base_dir : Path
        Directory to search.
    file_type : str
        File type key.
    year : int
        Four-digit year.
    quarter : int
        Quarter number.

    Returns
    -------
    Path | None
        First matching path if present, otherwise ``None``.
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


def get_total_row_markers() -> list[str]:
    """Return label markers that denote total rows in PDF tables.

    Returns
    -------
    list[str]
        Markers pulled from extraction specs, defaulting to ``["Totales", "Total"]``.
    """
    specs = get_extraction_specs()
    doc_structure = specs.get("document_structure", {})
    markers = doc_structure.get("section_markers", {})
    return cast("list[str]", markers.get("table_end_markers", ["Totales", "Total"]))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries, allowing overrides in ``overlay``.

    Parameters
    ----------
    base : dict[str, Any]
        Original mapping.
    overlay : dict[str, Any]
        Values that override or extend ``base``.

    Returns
    -------
    dict[str, Any]
        New merged mapping.
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
    """Write a single PDF page to ``TEMP_DIR`` for manual inspection.

    Parameters
    ----------
    pdf_path : Path
        Source PDF containing financial statements.
    page_number : int
        One-based page index to extract.
    prefix : str, optional
        Prefix used for the generated temporary filename.

    Returns
    -------
    Path
        Path to the temporary one-page PDF.

    Raises
    ------
    ImportError
        If ``pypdf`` is not installed (required for page extraction).
    ValueError
        If ``page_number`` falls outside the PDF page range.
    """
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore[import-not-found]
    except ImportError as err:
        msg = "pypdf is required for page extraction. Install with: pip install pypdf"
        raise ImportError(msg) from err

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if page_number < 1 or page_number > total_pages:
        msg = f"Page {page_number} out of range (1-{total_pages})"
        raise ValueError(msg)

    # Create writer with just the requested page (0-indexed internally)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])

    # Create temp file in TEMP_DIR
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = TEMP_DIR / f"{prefix}{pdf_path.stem}_p{page_number}.pdf"

    with Path(temp_path).open("wb") as f:
        writer.write(f)

    return temp_path


def cleanup_temp_files(prefix: str = "page_review_") -> int:
    """Remove temporary page review PDFs created by :func:`extract_pdf_page_to_temp`.

    Parameters
    ----------
    prefix : str, optional
        Filename prefix to match under ``TEMP_DIR``.

    Returns
    -------
    int
        Count of deleted files.
    """
    count = 0
    if TEMP_DIR.exists():
        for f in TEMP_DIR.glob(f"{prefix}*.pdf"):
            f.unlink()
            count += 1
    return count
