"""PDF parser using pdfplumber for text and table extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pdfplumber

from puco_eeff.config import setup_logging

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)


def extract_text_from_pdf(
    file_path: Path,
    pages: list[int] | None = None,
) -> dict[int, str]:
    """Extract text content from PDF pages.

    Args:
        file_path: Path to the PDF file
        pages: List of page numbers to extract (1-indexed). If None, extract all.

    Returns:
        Dictionary mapping page numbers to extracted text

    """
    logger.info("Extracting text from PDF: %s", file_path)

    if not file_path.exists():
        msg = f"PDF file not found: {file_path}"
        raise FileNotFoundError(msg)

    result: dict[int, str] = {}

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        logger.debug("PDF has %s pages", total_pages)

        # Determine which pages to process
        page_indices = (
            range(total_pages) if pages is None else [p - 1 for p in pages if 0 < p <= total_pages]
        )

        for idx in page_indices:
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            result[idx + 1] = text  # 1-indexed page numbers
            logger.debug(f"Page {idx + 1}: {len(text)} characters")

    logger.info(f"Extracted text from {len(result)} pages")
    return result


def extract_tables_from_pdf(
    file_path: Path,
    pages: list[int] | None = None,
    table_settings: dict[str, Any] | None = None,
) -> dict[int, list[list[list[str | None]]]]:
    """Extract tables from PDF pages.

    Args:
        file_path: Path to the PDF file
        pages: List of page numbers to extract (1-indexed). If None, extract all.
        table_settings: pdfplumber table extraction settings

    Returns:
        Dictionary mapping page numbers to list of tables (each table is a list of rows)

    """
    logger.info("Extracting tables from PDF: %s", file_path)

    if not file_path.exists():
        msg = f"PDF file not found: {file_path}"
        raise FileNotFoundError(msg)

    result: dict[int, list[list[list[str | None]]]] = {}

    # Default table settings optimized for financial statements
    settings = table_settings or {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
    }

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

        page_indices = (
            range(total_pages) if pages is None else [p - 1 for p in pages if 0 < p <= total_pages]
        )

        for idx in page_indices:
            page = pdf.pages[idx]
            tables = page.extract_tables(table_settings=settings)
            if tables:
                result[idx + 1] = tables
                logger.debug(f"Page {idx + 1}: {len(tables)} tables found")

    total_tables = sum(len(t) for t in result.values())
    logger.info(f"Extracted {total_tables} tables from {len(result)} pages")
    return result


def find_section_in_pdf(
    file_path: Path,
    section_pattern: str,
) -> list[dict[str, Any]]:
    """Find sections in PDF matching a pattern.

    Args:
        file_path: Path to the PDF file
        section_pattern: Section pattern (e.g., "note 20.b", "section 5")

    Returns:
        List of dictionaries with page number and context

    """
    logger.info("Searching for section: %s", section_pattern)

    # Parse the section pattern
    pattern_lower = section_pattern.lower()
    matches: list[dict[str, Any]] = []

    text_by_page = extract_text_from_pdf(file_path)

    for page_num, text in text_by_page.items():
        text_lower = text.lower()

        # Simple pattern matching - can be enhanced
        if pattern_lower in text_lower:
            # Find the context around the match
            idx = text_lower.find(pattern_lower)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(pattern_lower) + 200)
            context = text[start:end]

            matches.append(
                {
                    "page": page_num,
                    "pattern": section_pattern,
                    "context": context,
                },
            )
            logger.debug("Found match on page %s", page_num)

    logger.info(f"Found {len(matches)} matches for pattern '{section_pattern}'")
    return matches


def get_pdf_info(file_path: Path) -> dict[str, Any]:
    """Get basic information about a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary with PDF metadata

    """
    if not file_path.exists():
        msg = f"PDF file not found: {file_path}"
        raise FileNotFoundError(msg)

    with pdfplumber.open(file_path) as pdf:
        return {
            "path": str(file_path),
            "num_pages": len(pdf.pages),
            "metadata": pdf.metadata,
        }
