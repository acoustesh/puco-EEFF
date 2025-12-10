"""puco-EEFF: financial statement extraction for Pucobre.

The package downloads quarterly filings, parses PDF and XBRL data, runs
validations, and emits sheet-specific JSON plus combined Excel workbooks.

Architecture
------------
* ``scraper``: Playwright/httpx downloaders for PDFs (pucobre.cl) and XBRL (cmfchile.cl).
* ``extractor``: PDF table parsing (pdfplumber) and XBRL parsing (lxml) with OCR fallback.
* ``sheets``: Sheet-specific mappings and validation rules (currently Sheet1).
* ``transformer``: Normalization and source provenance tracking for auditability.
* ``writer``: JSON/CSV outputs and multi-period Excel workbook assembly.

Configuration and credentials
-----------------------------
Paths default to the ``data/`` and ``audit/`` trees but respect ``DATA_DIR``,
``AUDIT_DIR``, ``LOGS_DIR``, and ``TEMP_DIR`` overrides. OCR backends use
``MISTRAL_API_KEY`` and ``OPENROUTER_API_KEY`` (OpenAI-compatible), with
optional ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` for fallbacks.

Entrypoints
-----------
See ``instructions/*.md`` for operator workflows. The primary runnable module is
:mod:`puco_eeff.main_sheet1`, which orchestrates download → extract → validate →
persist for Sheet1.

Examples
--------
Run Sheet1 extraction for 2024 Q2:

    >>> python -m puco_eeff.main_sheet1 --year 2024 --quarter 2

Skip download and use existing files:

    >>> python -m puco_eeff.main_sheet1 -y 2024 -q 2 --skip-download
"""

from puco_eeff.config import quarter_to_roman

__version__ = "0.1.0"
__all__ = ["__version__", "quarter_to_roman"]

# Public helper for introspection tools.
def get_version() -> str:
    """Return the current package version string.

    Returns
    -------
    str
        Semantic version identifier (e.g., ``"0.1.0"``).
    """
    return __version__


__all__.append("get_version")
