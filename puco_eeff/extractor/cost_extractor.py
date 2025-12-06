"""Extract detailed cost breakdowns from PDF and validate against XBRL.

This module extracts Nota 21 (Costo de Venta) and Nota 22 (Gastos de Administración)
from the Estados Financieros PDF. When XBRL is available, it cross-validates
the totals between the two sources.

Works with both CMF Chile downloads (separate files) and Pucobre.cl fallback
(combined PDF with no XBRL available).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from puco_eeff.config import get_period_paths, setup_logging
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file

logger = setup_logging(__name__)

# Expected line items for Nota 21 - Costo de Venta
NOTA_21_ITEMS = [
    "Gastos en personal",
    "Materiales y repuestos",
    "Energía eléctrica",
    "Servicios de terceros",
    "Depreciación y amort. del periodo",
    "Depreciación Activos en leasing",
    "Depreciación Arrendamientos",
    "Servicios mineros de terceros",
    "Fletes y otros gastos operacionales",
    "Gastos Diferidos, ajustes existencias y otros",
    "Obligaciones por convenios colectivos",
]

# Expected line items for Nota 22 - Gastos de Administración y Ventas
NOTA_22_ITEMS = [
    "Gastos en personal",
    "Materiales y repuestos",
    "Servicios de terceros",
    "Provisión gratificación legal y otros",
    "Gastos comercialización",
    "Otros gastos",
]


@dataclass
class LineItem:
    """A single line item from the cost breakdown."""

    concepto: str
    ytd_actual: int | None = None
    ytd_anterior: int | None = None
    quarter_actual: int | None = None
    quarter_anterior: int | None = None


@dataclass
class CostBreakdown:
    """Cost breakdown from a Nota section."""

    nota_number: int
    nota_title: str
    items: list[LineItem] = field(default_factory=list)
    total_ytd_actual: int | None = None
    total_ytd_anterior: int | None = None
    total_quarter_actual: int | None = None
    total_quarter_anterior: int | None = None
    page_number: int | None = None

    def sum_items_ytd_actual(self) -> int:
        """Sum all YTD actual values (excluding total row)."""
        return sum(item.ytd_actual or 0 for item in self.items if "total" not in item.concepto.lower())

    def is_valid(self) -> bool:
        """Check if the sum of items equals the total."""
        if self.total_ytd_actual is None:
            return False
        calculated_sum = self.sum_items_ytd_actual()
        return calculated_sum == self.total_ytd_actual


@dataclass
class ValidationResult:
    """Result of cross-validation between PDF and XBRL."""

    field_name: str
    pdf_value: int | None
    xbrl_value: int | None
    match: bool
    source: str  # "pdf_only", "xbrl_only", "both"
    difference: int | None = None

    @property
    def status(self) -> str:
        """Return validation status string."""
        if self.source == "pdf_only":
            return "⚠ PDF only (no XBRL)"
        elif self.source == "xbrl_only":
            return "⚠ XBRL only (PDF extraction failed)"
        elif self.match:
            return "✓ Match"
        else:
            return f"✗ Mismatch (diff: {self.difference:,})"


@dataclass
class ExtractionResult:
    """Complete extraction result with optional validation."""

    year: int
    quarter: int
    nota_21: CostBreakdown | None = None
    nota_22: CostBreakdown | None = None
    xbrl_available: bool = False
    xbrl_cost_of_sales: int | None = None
    xbrl_admin_expense: int | None = None
    validations: list[ValidationResult] = field(default_factory=list)
    source: str = "cmf"  # "cmf" or "pucobre.cl"
    pdf_path: Path | None = None
    xbrl_path: Path | None = None

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        if not self.validations:
            # No validations performed (PDF-only extraction)
            return self.nota_21 is not None and self.nota_22 is not None
        return all(v.match for v in self.validations)


def parse_chilean_number(value: str | None) -> int | None:
    """Parse a Chilean-formatted number.

    Chilean format uses:
    - Period as thousands separator: 30.294 = 30,294
    - Parentheses for negatives: (30.294) = -30,294
    - Comma for decimals (rare in financial statements)

    Args:
        value: String value to parse

    Returns:
        Integer value or None if parsing fails
    """
    if not value:
        return None

    # Clean the string
    value = str(value).strip()

    # Check for negative (parentheses)
    is_negative = "(" in value and ")" in value

    # Remove non-numeric characters except period and minus
    value = re.sub(r"[^\d.\-]", "", value)

    if not value or value in (".", "-"):
        return None

    try:
        # Remove period (thousands separator) and convert
        value = value.replace(".", "")
        result = int(value)
        return -abs(result) if is_negative else result
    except ValueError:
        return None


def find_nota_page(pdf_path: Path, nota_number: int) -> int | None:
    """Find the page number where a Nota section starts.

    Args:
        pdf_path: Path to the PDF file
        nota_number: Nota number to find (21 or 22)

    Returns:
        0-indexed page number or None if not found
    """
    search_patterns = [
        f"{nota_number}. COSTO",
        f"{nota_number}. GASTOS",
        f"NOTA {nota_number}",
        f"{nota_number}.",
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").upper()

            for pattern in search_patterns:
                if pattern.upper() in text:
                    # Verify this is the right section
                    if nota_number == 21 and "COSTO" in text and "VENTA" in text:
                        logger.info(f"Found Nota 21 on page {page_idx + 1}")
                        return page_idx
                    elif nota_number == 22 and "GASTOS" in text and "ADMINISTRACI" in text:
                        logger.info(f"Found Nota 22 on page {page_idx + 1}")
                        return page_idx

    return None


def extract_table_from_page(
    pdf_path: Path,
    page_index: int,
    expected_items: list[str],
) -> list[dict[str, Any]]:
    """Extract cost table data from a specific page.

    Args:
        pdf_path: Path to the PDF file
        page_index: 0-indexed page number
        expected_items: List of expected line item names

    Returns:
        List of dictionaries with concepto and value columns
    """
    with pdfplumber.open(pdf_path) as pdf:
        if page_index >= len(pdf.pages):
            return []

        page = pdf.pages[page_index]

        # Try to extract tables
        tables = page.extract_tables()

        if not tables:
            logger.warning(f"No tables found on page {page_index + 1}")
            return []

        # Find the table containing our expected items
        best_table = None
        best_match_count = 0

        for table in tables:
            if not table:
                continue

            table_text = str(table).lower()
            match_count = sum(1 for item in expected_items if item.lower() in table_text)

            if match_count > best_match_count:
                best_match_count = match_count
                best_table = table

        if best_table is None or best_match_count < 3:
            logger.warning(f"Could not find expected cost table on page {page_index + 1}")
            return []

        # Parse the table rows
        return _parse_cost_table(best_table, expected_items)


def _parse_cost_table(table: list[list[str | None]], expected_items: list[str]) -> list[dict[str, Any]]:
    """Parse a cost breakdown table.

    Args:
        table: Raw table data from pdfplumber
        expected_items: List of expected line item names

    Returns:
        List of parsed row dictionaries
    """
    parsed_rows = []

    for row in table:
        if not row or not any(row):
            continue

        # First cell usually contains the concept name
        # May be merged with values or split across multiple lines
        row_text = str(row[0] or "")

        # Check if any expected item is in this row
        matched_item = None
        for expected in expected_items:
            if expected.lower() in row_text.lower():
                matched_item = expected
                break

        # Also check for "Totales" or "Total"
        if not matched_item and ("total" in row_text.lower()):
            matched_item = "Totales"

        if matched_item:
            # Extract numeric values from remaining columns
            values = []
            for cell in row[1:]:
                if cell:
                    parsed = parse_chilean_number(str(cell))
                    if parsed is not None:
                        values.append(parsed)

            parsed_rows.append({
                "concepto": matched_item,
                "values": values,
            })

    return parsed_rows


def extract_nota_21(pdf_path: Path) -> CostBreakdown | None:
    """Extract Nota 21 - Costo de Venta from PDF.

    Args:
        pdf_path: Path to Estados Financieros PDF

    Returns:
        CostBreakdown object or None if extraction fails
    """
    page_idx = find_nota_page(pdf_path, 21)
    if page_idx is None:
        logger.error("Could not find Nota 21 in PDF")
        return None

    rows = extract_table_from_page(pdf_path, page_idx, NOTA_21_ITEMS)

    if not rows:
        # Try next page (table might span pages)
        rows = extract_table_from_page(pdf_path, page_idx + 1, NOTA_21_ITEMS)

    if not rows:
        logger.error("Could not extract Nota 21 table")
        return None

    breakdown = CostBreakdown(
        nota_number=21,
        nota_title="Costo de Venta",
        page_number=page_idx + 1,
    )

    for row in rows:
        concepto = row["concepto"]
        values = row.get("values", [])

        if concepto.lower() == "totales":
            breakdown.total_ytd_actual = values[0] if len(values) > 0 else None
            breakdown.total_ytd_anterior = values[1] if len(values) > 1 else None
            breakdown.total_quarter_actual = values[2] if len(values) > 2 else None
            breakdown.total_quarter_anterior = values[3] if len(values) > 3 else None
        else:
            item = LineItem(
                concepto=concepto,
                ytd_actual=values[0] if len(values) > 0 else None,
                ytd_anterior=values[1] if len(values) > 1 else None,
                quarter_actual=values[2] if len(values) > 2 else None,
                quarter_anterior=values[3] if len(values) > 3 else None,
            )
            breakdown.items.append(item)

    logger.info(f"Extracted {len(breakdown.items)} items from Nota 21")
    return breakdown


def extract_nota_22(pdf_path: Path) -> CostBreakdown | None:
    """Extract Nota 22 - Gastos de Administración y Ventas from PDF.

    Args:
        pdf_path: Path to Estados Financieros PDF

    Returns:
        CostBreakdown object or None if extraction fails
    """
    page_idx = find_nota_page(pdf_path, 22)
    if page_idx is None:
        # Nota 22 is often on the same page as Nota 21
        page_idx = find_nota_page(pdf_path, 21)
        if page_idx is None:
            logger.error("Could not find Nota 22 in PDF")
            return None

    rows = extract_table_from_page(pdf_path, page_idx, NOTA_22_ITEMS)

    if not rows:
        # Try next page
        rows = extract_table_from_page(pdf_path, page_idx + 1, NOTA_22_ITEMS)

    if not rows:
        logger.error("Could not extract Nota 22 table")
        return None

    breakdown = CostBreakdown(
        nota_number=22,
        nota_title="Gastos de Administración y Ventas",
        page_number=page_idx + 1,
    )

    for row in rows:
        concepto = row["concepto"]
        values = row.get("values", [])

        if concepto.lower() == "totales":
            breakdown.total_ytd_actual = values[0] if len(values) > 0 else None
            breakdown.total_ytd_anterior = values[1] if len(values) > 1 else None
            breakdown.total_quarter_actual = values[2] if len(values) > 2 else None
            breakdown.total_quarter_anterior = values[3] if len(values) > 3 else None
        else:
            item = LineItem(
                concepto=concepto,
                ytd_actual=values[0] if len(values) > 0 else None,
                ytd_anterior=values[1] if len(values) > 1 else None,
                quarter_actual=values[2] if len(values) > 2 else None,
                quarter_anterior=values[3] if len(values) > 3 else None,
            )
            breakdown.items.append(item)

    logger.info(f"Extracted {len(breakdown.items)} items from Nota 22")
    return breakdown


def extract_xbrl_totals(xbrl_path: Path) -> dict[str, int | None]:
    """Extract relevant totals from XBRL file.

    Args:
        xbrl_path: Path to XBRL file

    Returns:
        Dictionary with cost_of_sales and admin_expense totals
    """
    try:
        data = parse_xbrl_file(xbrl_path)
    except Exception as e:
        logger.error(f"Failed to parse XBRL: {e}")
        return {"cost_of_sales": None, "admin_expense": None}

    result: dict[str, int | None] = {
        "cost_of_sales": None,
        "admin_expense": None,
    }

    # Look for Cost of Sales
    cost_facts = get_facts_by_name(data, "CostOfSales")
    if not cost_facts:
        cost_facts = get_facts_by_name(data, "CostoDeVentas")

    for fact in cost_facts:
        if fact.get("value"):
            try:
                result["cost_of_sales"] = int(float(fact["value"]))
                break
            except (ValueError, TypeError):
                continue

    # Look for Administrative Expense
    admin_facts = get_facts_by_name(data, "AdministrativeExpense")
    if not admin_facts:
        admin_facts = get_facts_by_name(data, "GastosDeAdministracion")

    for fact in admin_facts:
        if fact.get("value"):
            try:
                result["admin_expense"] = int(float(fact["value"]))
                break
            except (ValueError, TypeError):
                continue

    return result


def validate_extraction(
    pdf_nota_21: CostBreakdown | None,
    pdf_nota_22: CostBreakdown | None,
    xbrl_totals: dict[str, int | None] | None,
) -> list[ValidationResult]:
    """Cross-validate PDF extraction against XBRL totals.

    Args:
        pdf_nota_21: Extracted Nota 21 from PDF
        pdf_nota_22: Extracted Nota 22 from PDF
        xbrl_totals: Totals extracted from XBRL (or None if no XBRL)

    Returns:
        List of validation results
    """
    validations = []

    # Validate Cost of Sales (Nota 21)
    pdf_cost = pdf_nota_21.total_ytd_actual if pdf_nota_21 else None
    xbrl_cost = xbrl_totals.get("cost_of_sales") if xbrl_totals else None

    if pdf_cost is not None and xbrl_cost is not None:
        # Both sources available - compare
        # Use absolute values since signs may differ
        match = abs(pdf_cost) == abs(xbrl_cost)
        validations.append(
            ValidationResult(
                field_name="Costo de Venta (Nota 21)",
                pdf_value=pdf_cost,
                xbrl_value=xbrl_cost,
                match=match,
                source="both",
                difference=abs(pdf_cost) - abs(xbrl_cost) if not match else None,
            )
        )
    elif pdf_cost is not None:
        # PDF only
        validations.append(
            ValidationResult(
                field_name="Costo de Venta (Nota 21)",
                pdf_value=pdf_cost,
                xbrl_value=None,
                match=True,  # Can't validate, but extraction succeeded
                source="pdf_only",
            )
        )
    elif xbrl_cost is not None:
        # XBRL only
        validations.append(
            ValidationResult(
                field_name="Costo de Venta (Nota 21)",
                pdf_value=None,
                xbrl_value=xbrl_cost,
                match=False,
                source="xbrl_only",
            )
        )

    # Validate Administrative Expense (Nota 22)
    pdf_admin = pdf_nota_22.total_ytd_actual if pdf_nota_22 else None
    xbrl_admin = xbrl_totals.get("admin_expense") if xbrl_totals else None

    if pdf_admin is not None and xbrl_admin is not None:
        match = abs(pdf_admin) == abs(xbrl_admin)
        validations.append(
            ValidationResult(
                field_name="Gastos Administración (Nota 22)",
                pdf_value=pdf_admin,
                xbrl_value=xbrl_admin,
                match=match,
                source="both",
                difference=abs(pdf_admin) - abs(xbrl_admin) if not match else None,
            )
        )
    elif pdf_admin is not None:
        validations.append(
            ValidationResult(
                field_name="Gastos Administración (Nota 22)",
                pdf_value=pdf_admin,
                xbrl_value=None,
                match=True,
                source="pdf_only",
            )
        )
    elif xbrl_admin is not None:
        validations.append(
            ValidationResult(
                field_name="Gastos Administración (Nota 22)",
                pdf_value=None,
                xbrl_value=xbrl_admin,
                match=False,
                source="xbrl_only",
            )
        )

    return validations


def extract_detailed_costs(
    year: int,
    quarter: int,
    validate: bool = True,
) -> ExtractionResult:
    """Extract detailed cost breakdowns for a period.

    This function:
    1. Locates the Estados Financieros PDF
    2. Extracts Nota 21 (Costo de Venta) and Nota 22 (Gastos Admin)
    3. If XBRL is available, extracts totals for validation
    4. Cross-validates PDF totals against XBRL

    Works with both CMF Chile (with XBRL) and Pucobre.cl fallback (PDF only).

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        validate: If True, validate against XBRL when available

    Returns:
        ExtractionResult with breakdowns and validation status
    """
    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]

    # Find the PDF file
    pdf_path = raw_dir / f"estados_financieros_{year}_Q{quarter}.pdf"
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return ExtractionResult(
            year=year,
            quarter=quarter,
            xbrl_available=False,
        )

    # Determine source (check for pucobre_combined file)
    combined_path = raw_dir / f"pucobre_combined_{year}_Q{quarter}.pdf"
    source = "pucobre.cl" if combined_path.exists() else "cmf"

    result = ExtractionResult(
        year=year,
        quarter=quarter,
        source=source,
        pdf_path=pdf_path,
    )

    # Extract from PDF
    result.nota_21 = extract_nota_21(pdf_path)
    result.nota_22 = extract_nota_22(pdf_path)

    # Check for XBRL
    xbrl_path = raw_dir / f"estados_financieros_{year}_Q{quarter}.xbrl"
    if not xbrl_path.exists():
        xbrl_path = raw_dir / f"estados_financieros_{year}_Q{quarter}.xml"

    if xbrl_path.exists():
        result.xbrl_available = True
        result.xbrl_path = xbrl_path

        if validate:
            xbrl_totals = extract_xbrl_totals(xbrl_path)
            result.xbrl_cost_of_sales = xbrl_totals.get("cost_of_sales")
            result.xbrl_admin_expense = xbrl_totals.get("admin_expense")

            result.validations = validate_extraction(
                result.nota_21,
                result.nota_22,
                xbrl_totals,
            )
    else:
        logger.info(f"No XBRL available for {year} Q{quarter} - using PDF only")
        result.xbrl_available = False

        if validate:
            # Still perform PDF-only validation
            result.validations = validate_extraction(
                result.nota_21,
                result.nota_22,
                None,
            )

    return result


def save_extraction_result(result: ExtractionResult, output_dir: Path | None = None) -> Path:
    """Save extraction result to JSON file.

    Args:
        result: ExtractionResult to save
        output_dir: Output directory (defaults to processed dir)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        paths = get_period_paths(result.year, result.quarter)
        output_dir = paths["processed"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detailed_costs.json"

    # Convert to serializable dict
    data = {
        "period": f"{result.year}_Q{result.quarter}",
        "source": result.source,
        "pdf_path": str(result.pdf_path) if result.pdf_path else None,
        "xbrl_path": str(result.xbrl_path) if result.xbrl_path else None,
        "xbrl_available": result.xbrl_available,
        "nota_21": _breakdown_to_dict(result.nota_21) if result.nota_21 else None,
        "nota_22": _breakdown_to_dict(result.nota_22) if result.nota_22 else None,
        "validations": [
            {
                "field": v.field_name,
                "pdf_value": v.pdf_value,
                "xbrl_value": v.xbrl_value,
                "match": v.match,
                "source": v.source,
                "status": v.status,
            }
            for v in result.validations
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved extraction result to: {output_path}")
    return output_path


def _breakdown_to_dict(breakdown: CostBreakdown) -> dict[str, Any]:
    """Convert CostBreakdown to dictionary."""
    return {
        "nota_number": breakdown.nota_number,
        "nota_title": breakdown.nota_title,
        "page_number": breakdown.page_number,
        "items": [
            {
                "concepto": item.concepto,
                "ytd_actual": item.ytd_actual,
                "ytd_anterior": item.ytd_anterior,
                "quarter_actual": item.quarter_actual,
                "quarter_anterior": item.quarter_anterior,
            }
            for item in breakdown.items
        ],
        "total_ytd_actual": breakdown.total_ytd_actual,
        "total_ytd_anterior": breakdown.total_ytd_anterior,
        "total_quarter_actual": breakdown.total_quarter_actual,
        "total_quarter_anterior": breakdown.total_quarter_anterior,
    }


def print_extraction_report(result: ExtractionResult) -> None:
    """Print a formatted extraction report.

    Args:
        result: ExtractionResult to report
    """
    print(f"\n{'=' * 60}")
    print(f"Cost Extraction Report: {result.year} Q{result.quarter}")
    print(f"{'=' * 60}")
    print(f"Source: {result.source}")
    print(f"XBRL Available: {'Yes' if result.xbrl_available else 'No'}")

    if result.nota_21:
        print(f"\n--- Nota 21: {result.nota_21.nota_title} ---")
        print(f"Page: {result.nota_21.page_number}")
        print(f"Items extracted: {len(result.nota_21.items)}")
        for item in result.nota_21.items:
            val = f"{item.ytd_actual:,}" if item.ytd_actual else "N/A"
            print(f"  {item.concepto}: {val}")
        total = f"{result.nota_21.total_ytd_actual:,}" if result.nota_21.total_ytd_actual else "N/A"
        print(f"  TOTAL: {total}")
    else:
        print("\n--- Nota 21: EXTRACTION FAILED ---")

    if result.nota_22:
        print(f"\n--- Nota 22: {result.nota_22.nota_title} ---")
        print(f"Page: {result.nota_22.page_number}")
        print(f"Items extracted: {len(result.nota_22.items)}")
        for item in result.nota_22.items:
            val = f"{item.ytd_actual:,}" if item.ytd_actual else "N/A"
            print(f"  {item.concepto}: {val}")
        total = f"{result.nota_22.total_ytd_actual:,}" if result.nota_22.total_ytd_actual else "N/A"
        print(f"  TOTAL: {total}")
    else:
        print("\n--- Nota 22: EXTRACTION FAILED ---")

    if result.validations:
        print("\n--- Validation Results ---")
        for v in result.validations:
            print(f"  {v.field_name}:")
            if v.pdf_value is not None:
                print(f"    PDF:  {v.pdf_value:,}")
            if v.xbrl_value is not None:
                print(f"    XBRL: {v.xbrl_value:,}")
            print(f"    {v.status}")

    print(f"\n{'=' * 60}\n")
