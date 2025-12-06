"""Extract detailed cost breakdowns from PDF and validate against XBRL.

This module extracts the "Cuadro Resumen de Costos" from Análisis Razonado,
which contains:
- Ingresos de actividades ordinarias
- Costo de Venta breakdown (11 line items + total)
- Gastos de Administración y Ventas breakdown (6 line items + total)

The data structure follows config/config.json sheet1 specification with 27 rows.
When XBRL is available, it cross-validates totals between sources.

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

from puco_eeff.config import get_config, get_period_paths, setup_logging
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file

logger = setup_logging(__name__)


def load_sheet1_config() -> dict[str, Any]:
    """Load sheet1 configuration from config.json."""
    config = get_config()
    return config.get("sheets", {}).get("sheet1", {})


# Sheet1 row structure from config - Costo de Venta items (rows 4-14)
COSTO_VENTA_ITEMS = [
    "Gastos en personal",
    "Materiales y repuestos",
    "Energía eléctrica",
    "Servicios de terceros",
    "Depreciación y amort del periodo",
    "Depreciación Activos en leasing",
    "Depreciación Arrendamientos",
    "Servicios mineros de terceros",
    "Fletes y otros gastos operacionales",
    "Gastos Diferidos, ajustes existencias y otros",
    "Obligaciones por convenios colectivos",
]

# Sheet1 row structure - Gasto Admin y Ventas items (rows 20-25)
GASTO_ADMIN_ITEMS = [
    "Gastos en personal",
    "Materiales y repuestos",
    "Servicios de terceros",
    "Provision gratificacion legal y otros",
    "Gastos comercializacion",
    "Otros gastos",
]

# Field names matching config row_mapping
FIELD_MAPPING = {
    "ingresos_ordinarios": "Ingresos de actividades ordinarias",
    "cv_gastos_personal": "Gastos en personal",
    "cv_materiales": "Materiales y repuestos",
    "cv_energia": "Energía eléctrica",
    "cv_servicios_terceros": "Servicios de terceros",
    "cv_depreciacion_amort": "Depreciación y amort del periodo",
    "cv_deprec_leasing": "Depreciación Activos en leasing",
    "cv_deprec_arrend": "Depreciación Arrendamientos",
    "cv_serv_mineros": "Servicios mineros de terceros",
    "cv_fletes": "Fletes y otros gastos operacionales",
    "cv_gastos_diferidos": "Gastos Diferidos, ajustes existencias y otros",
    "cv_convenios": "Obligaciones por convenios colectivos",
    "total_costo_venta": "Total Costo de Venta",
    "ga_gastos_personal": "Gastos en personal",
    "ga_materiales": "Materiales y repuestos",
    "ga_servicios_terceros": "Servicios de terceros",
    "ga_gratificacion": "Provision gratificacion legal y otros",
    "ga_comercializacion": "Gastos comercializacion",
    "ga_otros": "Otros gastos",
    "total_gasto_admin": "Totales",  # Row 27 - specifically Gasto Admin total
}


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

    rows = extract_table_from_page(pdf_path, page_idx, COSTO_VENTA_ITEMS)

    if not rows:
        # Try next page (table might span pages)
        rows = extract_table_from_page(pdf_path, page_idx + 1, COSTO_VENTA_ITEMS)

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

    rows = extract_table_from_page(pdf_path, page_idx, GASTO_ADMIN_ITEMS)

    if not rows:
        # Try next page
        rows = extract_table_from_page(pdf_path, page_idx + 1, GASTO_ADMIN_ITEMS)

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


# =============================================================================
# Sheet1 Data Structure (27 rows as per config.json)
# =============================================================================


@dataclass
class Sheet1Data:
    """Data structure for Sheet1 - Ingresos y Costos.

    This follows the 27-row structure defined in config.json:
    - Row 1: Ingresos de actividades ordinarias
    - Rows 3-15: Costo de Venta section (header + 11 items + total)
    - Rows 19-27: Gasto Admin section (header + 6 items + Totales)

    The "Totales" in row 27 specifically refers to Gasto Admin total,
    NOT to be confused with "Total Costo de Venta" in row 15.
    """

    quarter: str  # e.g., "IIQ2024"
    year: int
    quarter_num: int
    source: str = "cmf"  # "cmf" or "pucobre.cl"
    xbrl_available: bool = False

    # Row 1: Ingresos
    ingresos_ordinarios: int | None = None

    # Rows 4-14: Costo de Venta breakdown
    cv_gastos_personal: int | None = None
    cv_materiales: int | None = None
    cv_energia: int | None = None
    cv_servicios_terceros: int | None = None
    cv_depreciacion_amort: int | None = None
    cv_deprec_leasing: int | None = None
    cv_deprec_arrend: int | None = None
    cv_serv_mineros: int | None = None
    cv_fletes: int | None = None
    cv_gastos_diferidos: int | None = None
    cv_convenios: int | None = None

    # Row 15: Total Costo de Venta
    total_costo_venta: int | None = None

    # Rows 20-25: Gasto Admin breakdown
    ga_gastos_personal: int | None = None
    ga_materiales: int | None = None
    ga_servicios_terceros: int | None = None
    ga_gratificacion: int | None = None
    ga_comercializacion: int | None = None
    ga_otros: int | None = None

    # Row 27: Totales (specifically Gasto Admin total, NOT Costo de Venta)
    total_gasto_admin: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching config row_mapping."""
        return {
            "quarter": self.quarter,
            "year": self.year,
            "quarter_num": self.quarter_num,
            "source": self.source,
            "xbrl_available": self.xbrl_available,
            "ingresos_ordinarios": self.ingresos_ordinarios,
            "cv_gastos_personal": self.cv_gastos_personal,
            "cv_materiales": self.cv_materiales,
            "cv_energia": self.cv_energia,
            "cv_servicios_terceros": self.cv_servicios_terceros,
            "cv_depreciacion_amort": self.cv_depreciacion_amort,
            "cv_deprec_leasing": self.cv_deprec_leasing,
            "cv_deprec_arrend": self.cv_deprec_arrend,
            "cv_serv_mineros": self.cv_serv_mineros,
            "cv_fletes": self.cv_fletes,
            "cv_gastos_diferidos": self.cv_gastos_diferidos,
            "cv_convenios": self.cv_convenios,
            "total_costo_venta": self.total_costo_venta,
            "ga_gastos_personal": self.ga_gastos_personal,
            "ga_materiales": self.ga_materiales,
            "ga_servicios_terceros": self.ga_servicios_terceros,
            "ga_gratificacion": self.ga_gratificacion,
            "ga_comercializacion": self.ga_comercializacion,
            "ga_otros": self.ga_otros,
            "total_gasto_admin": self.total_gasto_admin,
        }

    def to_row_list(self) -> list[tuple[int, str, int | None]]:
        """Convert to list of (row_number, label, value) tuples.

        Returns the 27-row structure as defined in config.
        """
        return [
            (1, "Ingresos de actividades ordinarias M USD", self.ingresos_ordinarios),
            (2, "", None),
            (3, "Costo de Venta", None),  # Header
            (4, "Gastos en personal", self.cv_gastos_personal),
            (5, "Materiales y repuestos", self.cv_materiales),
            (6, "Energía eléctrica", self.cv_energia),
            (7, "Servicios de terceros", self.cv_servicios_terceros),
            (8, "Depreciación y amort del periodo", self.cv_depreciacion_amort),
            (9, "Depreciación Activos en leasing -Nota 20", self.cv_deprec_leasing),
            (10, "Depreciación Arrendamientos -Nota 20", self.cv_deprec_arrend),
            (11, "Servicios mineros de terceros", self.cv_serv_mineros),
            (12, "Fletes y otros gastos operacionales", self.cv_fletes),
            (13, "Gastos Diferidos, ajustes existencias y otros", self.cv_gastos_diferidos),
            (14, "Obligaciones por convenios colectivos", self.cv_convenios),
            (15, "Total Costo de Venta", self.total_costo_venta),
            (16, "", None),
            (17, "", None),
            (18, "", None),
            (19, "Gasto Adm, y Ventas", None),  # Header
            (20, "Gastos en personal", self.ga_gastos_personal),
            (21, "Materiales y repuestos", self.ga_materiales),
            (22, "Servicios de terceros", self.ga_servicios_terceros),
            (23, "Provision gratificacion legal y otros", self.ga_gratificacion),
            (24, "Gastos comercializacion", self.ga_comercializacion),
            (25, "Otros gastos", self.ga_otros),
            (26, "", None),
            (27, "Totales", self.total_gasto_admin),  # Gasto Admin total ONLY
        ]


def quarter_to_roman(quarter: int) -> str:
    """Convert quarter number to Roman numeral format.

    Args:
        quarter: Quarter number (1-4)

    Returns:
        Roman numeral string (I, II, III, IV)
    """
    roman = {1: "I", 2: "II", 3: "III", 4: "IV"}
    return roman.get(quarter, str(quarter))


def format_quarter_label(year: int, quarter: int) -> str:
    """Format quarter label as used in Sheet1 headers.

    Args:
        year: Year (e.g., 2024)
        quarter: Quarter number (1-4)

    Returns:
        Formatted string like "IIQ2024"
    """
    return f"{quarter_to_roman(quarter)}Q{year}"


def find_cuadro_resumen_page(pdf_path: Path) -> int | None:
    """Find the page containing 'Cuadro Resumen de Costos' in Análisis Razonado.

    Args:
        pdf_path: Path to Análisis Razonado PDF

    Returns:
        0-indexed page number or None if not found
    """
    search_patterns = [
        "CUADRO RESUMEN DE COSTOS",
        "RESUMEN DE COSTOS",
        "Cuadro Resumen de Costos",
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").upper()

            for pattern in search_patterns:
                if pattern.upper() in text:
                    logger.info(f"Found 'Cuadro Resumen de Costos' on page {page_idx + 1}")
                    return page_idx

    return None


def extract_sheet1_from_analisis_razonado(
    year: int,
    quarter: int,
    validate_with_xbrl: bool = True,
) -> Sheet1Data | None:
    """Extract Sheet1 data from Análisis Razonado PDF.

    This extracts the "Cuadro Resumen de Costos" table which contains:
    - Ingresos de actividades ordinarias
    - Costo de Venta breakdown (11 items)
    - Total Costo de Venta
    - Gasto Admin breakdown (6 items)
    - Totales (specifically Gasto Admin total in row 27)

    When XBRL is available, it validates and supplements the PDF extraction.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        validate_with_xbrl: If True, validate PDF data against XBRL totals

    Returns:
        Sheet1Data object or None if extraction fails
    """
    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]

    # Find Análisis Razonado PDF
    ar_path = raw_dir / f"analisis_razonado_{year}_Q{quarter}.pdf"
    if not ar_path.exists():
        logger.warning(f"Análisis Razonado not found: {ar_path}")
        return None

    # Determine source
    combined_path = raw_dir / f"pucobre_combined_{year}_Q{quarter}.pdf"
    source = "pucobre.cl" if combined_path.exists() else "cmf"

    # Check XBRL availability
    xbrl_path = raw_dir / f"estados_financieros_{year}_Q{quarter}.xbrl"
    if not xbrl_path.exists():
        xbrl_path = raw_dir / f"estados_financieros_{year}_Q{quarter}.xml"
    xbrl_available = xbrl_path.exists()

    # Find the Cuadro Resumen page
    page_idx = find_cuadro_resumen_page(ar_path)
    if page_idx is None:
        logger.error("Could not find 'Cuadro Resumen de Costos' in Análisis Razonado")
        return None

    # Extract data from PDF
    data = Sheet1Data(
        quarter=format_quarter_label(year, quarter),
        year=year,
        quarter_num=quarter,
        source=source,
        xbrl_available=xbrl_available,
    )

    # Extract the table from PDF
    with pdfplumber.open(ar_path) as pdf:
        page = pdf.pages[page_idx]
        tables = page.extract_tables()

        if not tables:
            logger.error(f"No tables found on page {page_idx + 1}")
            return None

        # Find the cost summary table
        cost_table = _find_cost_summary_table(tables)
        if cost_table is None:
            logger.error("Could not identify cost summary table")
            return None

        # Parse the table into Sheet1Data
        _parse_cost_summary_table(cost_table, data)

    # Validate/supplement with XBRL if available
    if xbrl_available and validate_with_xbrl:
        _validate_sheet1_with_xbrl(data, xbrl_path)

    return data


def _validate_sheet1_with_xbrl(data: Sheet1Data, xbrl_path: Path) -> None:
    """Validate and supplement Sheet1 data with XBRL totals.

    This cross-validates PDF extraction against XBRL to:
    1. Log validation results (match/mismatch)
    2. Use XBRL values if PDF extraction failed for totals

    Args:
        data: Sheet1Data populated from PDF
        xbrl_path: Path to XBRL file
    """
    xbrl_totals = extract_xbrl_totals(xbrl_path)

    # Validate Total Costo de Venta
    xbrl_cost = xbrl_totals.get("cost_of_sales")
    if xbrl_cost is not None:
        if data.total_costo_venta is not None:
            # Both available - compare (using absolute values due to sign differences)
            if abs(data.total_costo_venta) == abs(xbrl_cost):
                logger.info(f"✓ Total Costo de Venta matches XBRL: {data.total_costo_venta:,}")
            else:
                logger.warning(
                    f"✗ Total Costo de Venta mismatch - PDF: {data.total_costo_venta:,}, XBRL: {xbrl_cost:,}"
                )
        else:
            # PDF extraction failed - use XBRL value
            logger.info(f"Using XBRL value for Total Costo de Venta: {xbrl_cost:,}")
            data.total_costo_venta = xbrl_cost

    # Validate Total Gasto Admin (row 27 "Totales")
    xbrl_admin = xbrl_totals.get("admin_expense")
    if xbrl_admin is not None:
        if data.total_gasto_admin is not None:
            if abs(data.total_gasto_admin) == abs(xbrl_admin):
                logger.info(f"✓ Total Gasto Admin matches XBRL: {data.total_gasto_admin:,}")
            else:
                logger.warning(f"✗ Total Gasto Admin mismatch - PDF: {data.total_gasto_admin:,}, XBRL: {xbrl_admin:,}")
        else:
            # PDF extraction failed - use XBRL value
            logger.info(f"Using XBRL value for Total Gasto Admin: {xbrl_admin:,}")
            data.total_gasto_admin = xbrl_admin

    # Try to extract Ingresos from XBRL if PDF failed
    if data.ingresos_ordinarios is None:
        try:
            xbrl_data = parse_xbrl_file(xbrl_path)
            revenue_facts = get_facts_by_name(xbrl_data, "RevenueFromContractsWithCustomers")
            if not revenue_facts:
                revenue_facts = get_facts_by_name(xbrl_data, "Revenue")
            if not revenue_facts:
                revenue_facts = get_facts_by_name(xbrl_data, "Ingresos")

            for fact in revenue_facts:
                if fact.get("value"):
                    data.ingresos_ordinarios = int(float(fact["value"]))
                    logger.info(f"Using XBRL value for Ingresos: {data.ingresos_ordinarios:,}")
                    break
        except Exception as e:
            logger.debug(f"Could not extract Ingresos from XBRL: {e}")


def _find_cost_summary_table(tables: list[list[list[str | None]]]) -> list[list[str | None]] | None:
    """Find the cost summary table among extracted tables.

    The cost summary table should contain:
    - "Ingresos" or "actividades ordinarias"
    - "Costo de Venta"
    - "Gasto" and "Admin" or "Ventas"

    Args:
        tables: List of tables extracted from page

    Returns:
        The cost summary table or None
    """
    for table in tables:
        if not table:
            continue

        table_text = str(table).lower()

        has_ingresos = "ingresos" in table_text or "actividades ordinarias" in table_text
        has_costo = "costo" in table_text and "venta" in table_text
        has_gasto = "gasto" in table_text and ("admin" in table_text or "ventas" in table_text)

        if has_ingresos and has_costo and has_gasto:
            return table

    return None


def _parse_cost_summary_table(table: list[list[str | None]], data: Sheet1Data) -> None:
    """Parse cost summary table and populate Sheet1Data.

    The table has sections:
    1. Ingresos row
    2. Costo de Venta section (header + 11 items + total)
    3. Gasto Admin section (header + 6 items + Totales)

    Important: "Totales" at the end is specifically Gasto Admin total (row 27),
    not to be confused with "Total Costo de Venta".

    Args:
        table: Raw table data
        data: Sheet1Data object to populate
    """
    current_section = None  # None, "costo_venta", "gasto_admin"

    for row in table:
        if not row or not any(row):
            continue

        row_text = str(row[0] or "").strip().lower()

        # Detect section headers
        if "costo" in row_text and "venta" in row_text and "total" not in row_text:
            current_section = "costo_venta"
            continue
        elif "gasto" in row_text and ("admin" in row_text or "ventas" in row_text):
            current_section = "gasto_admin"
            continue

        # Extract first numeric value from row
        value = None
        for cell in row[1:]:
            if cell:
                parsed = parse_chilean_number(str(cell))
                if parsed is not None:
                    value = parsed
                    break

        # Map row to data field based on content and current section
        _map_row_to_field(row_text, value, current_section, data)


def _map_row_to_field(
    row_text: str,
    value: int | None,
    section: str | None,
    data: Sheet1Data,
) -> None:
    """Map a table row to the appropriate Sheet1Data field.

    Uses section context to disambiguate items that appear in both sections
    (e.g., "Gastos en personal" appears in both Costo de Venta and Gasto Admin).

    Args:
        row_text: Lowercase text from the row's first cell
        value: Parsed numeric value
        section: Current section ("costo_venta", "gasto_admin", or None)
        data: Sheet1Data object to update
    """
    # Ingresos (no section context needed)
    if "ingresos" in row_text and "ordinarias" in row_text:
        data.ingresos_ordinarios = value
        return

    # Total Costo de Venta (explicit match to avoid confusion with "Totales")
    if "total" in row_text and "costo" in row_text and "venta" in row_text:
        data.total_costo_venta = value
        return

    # Totales at end = Gasto Admin total (row 27)
    # This is the ONLY "Totales" without "Costo" or "Venta" qualification
    if row_text == "totales" or (row_text.startswith("totales") and "costo" not in row_text):
        if section == "gasto_admin":
            data.total_gasto_admin = value
        return

    # Section-specific items
    if section == "costo_venta":
        if "gastos en personal" in row_text or row_text == "gastos en personal":
            data.cv_gastos_personal = value
        elif "materiales" in row_text and "repuestos" in row_text:
            data.cv_materiales = value
        elif "energía" in row_text or "energia" in row_text:
            data.cv_energia = value
        elif "servicios de terceros" in row_text:
            data.cv_servicios_terceros = value
        elif "depreciación" in row_text or "depreciacion" in row_text:
            if "leasing" in row_text:
                data.cv_deprec_leasing = value
            elif "arrendamiento" in row_text:
                data.cv_deprec_arrend = value
            else:
                data.cv_depreciacion_amort = value
        elif "servicios mineros" in row_text:
            data.cv_serv_mineros = value
        elif "fletes" in row_text:
            data.cv_fletes = value
        elif "gastos diferidos" in row_text or "ajustes existencias" in row_text:
            data.cv_gastos_diferidos = value
        elif "convenios" in row_text or "obligaciones" in row_text:
            data.cv_convenios = value

    elif section == "gasto_admin":
        if "gastos en personal" in row_text or row_text == "gastos en personal":
            data.ga_gastos_personal = value
        elif "materiales" in row_text and "repuestos" in row_text:
            data.ga_materiales = value
        elif "servicios de terceros" in row_text:
            data.ga_servicios_terceros = value
        elif "gratificacion" in row_text or "gratificación" in row_text:
            data.ga_gratificacion = value
        elif "comercializacion" in row_text or "comercialización" in row_text:
            data.ga_comercializacion = value
        elif "otros gastos" in row_text:
            data.ga_otros = value


def save_sheet1_data(data: Sheet1Data, output_dir: Path | None = None) -> Path:
    """Save Sheet1 data to JSON file.

    Args:
        data: Sheet1Data to save
        output_dir: Output directory (defaults to processed dir)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        paths = get_period_paths(data.year, data.quarter_num)
        output_dir = paths["processed"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sheet1_{data.quarter}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved Sheet1 data to: {output_path}")
    return output_path


def extract_sheet1_from_xbrl(year: int, quarter: int) -> Sheet1Data | None:
    """Extract Sheet1 totals directly from XBRL file.

    This extracts only the totals available in XBRL:
    - Ingresos (Revenue)
    - Total Costo de Venta (CostOfSales)
    - Total Gasto Admin (AdministrativeExpense)

    Note: XBRL does not contain the detailed line items (cv_*, ga_*),
    only totals. Use PDF extraction for full breakdown.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)

    Returns:
        Sheet1Data with totals only, or None if extraction fails
    """
    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]

    # Find XBRL file
    xbrl_path = raw_dir / f"estados_financieros_{year}_Q{quarter}.xbrl"
    if not xbrl_path.exists():
        xbrl_path = raw_dir / f"estados_financieros_{year}_Q{quarter}.xml"

    if not xbrl_path.exists():
        logger.warning(f"XBRL file not found for {year} Q{quarter}")
        return None

    try:
        xbrl_data = parse_xbrl_file(xbrl_path)
    except Exception as e:
        logger.error(f"Failed to parse XBRL: {e}")
        return None

    data = Sheet1Data(
        quarter=format_quarter_label(year, quarter),
        year=year,
        quarter_num=quarter,
        source="cmf",
        xbrl_available=True,
    )

    # Extract Revenue (Ingresos)
    revenue_facts = get_facts_by_name(xbrl_data, "RevenueFromContractsWithCustomers")
    if not revenue_facts:
        revenue_facts = get_facts_by_name(xbrl_data, "Revenue")
    if not revenue_facts:
        revenue_facts = get_facts_by_name(xbrl_data, "IngresosPorActividadesOrdinarias")

    for fact in revenue_facts:
        if fact.get("value"):
            try:
                data.ingresos_ordinarios = int(float(fact["value"]))
                logger.info(f"XBRL Ingresos: {data.ingresos_ordinarios:,}")
                break
            except (ValueError, TypeError):
                continue

    # Extract Cost of Sales (Total Costo de Venta)
    cost_facts = get_facts_by_name(xbrl_data, "CostOfSales")
    if not cost_facts:
        cost_facts = get_facts_by_name(xbrl_data, "CostoDeVentas")

    for fact in cost_facts:
        if fact.get("value"):
            try:
                data.total_costo_venta = int(float(fact["value"]))
                logger.info(f"XBRL Total Costo de Venta: {data.total_costo_venta:,}")
                break
            except (ValueError, TypeError):
                continue

    # Extract Administrative Expense (Total Gasto Admin - row 27 "Totales")
    admin_facts = get_facts_by_name(xbrl_data, "AdministrativeExpense")
    if not admin_facts:
        admin_facts = get_facts_by_name(xbrl_data, "GastosDeAdministracion")
    if not admin_facts:
        admin_facts = get_facts_by_name(xbrl_data, "GastosDeAdministracionYVentas")

    for fact in admin_facts:
        if fact.get("value"):
            try:
                data.total_gasto_admin = int(float(fact["value"]))
                logger.info(f"XBRL Total Gasto Admin (Totales row 27): {data.total_gasto_admin:,}")
                break
            except (ValueError, TypeError):
                continue

    # Check if we got at least one value
    if all(v is None for v in [data.ingresos_ordinarios, data.total_costo_venta, data.total_gasto_admin]):
        logger.warning(f"No Sheet1 data found in XBRL for {year} Q{quarter}")
        return None

    return data


def extract_sheet1(
    year: int,
    quarter: int,
    prefer_pdf: bool = True,
    validate: bool = True,
) -> Sheet1Data | None:
    """Extract Sheet1 data from available sources (PDF and/or XBRL).

    This is the main entry point for Sheet1 extraction. It:
    1. First tries PDF extraction from Análisis Razonado (full breakdown)
    2. Validates/supplements with XBRL if available
    3. Falls back to XBRL-only if PDF extraction fails

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        prefer_pdf: If True, prefer PDF for detailed breakdown
        validate: If True, validate PDF data against XBRL

    Returns:
        Sheet1Data object or None if extraction fails from all sources
    """
    data = None

    if prefer_pdf:
        # Try PDF first (has detailed breakdown)
        data = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=validate)

        if data is None:
            # Fall back to XBRL (totals only)
            logger.info(f"PDF extraction failed, trying XBRL for {year} Q{quarter}")
            data = extract_sheet1_from_xbrl(year, quarter)
    else:
        # Try XBRL first
        data = extract_sheet1_from_xbrl(year, quarter)

        if data is None or validate:
            # Try PDF for detailed breakdown or validation
            pdf_data = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=False)
            if pdf_data is not None:
                if data is None:
                    data = pdf_data
                else:
                    # Merge: use PDF detailed items, XBRL totals
                    _merge_pdf_into_xbrl_data(pdf_data, data)

    if data is None:
        logger.error(f"Failed to extract Sheet1 from any source for {year} Q{quarter}")

    return data


def _merge_pdf_into_xbrl_data(pdf_data: Sheet1Data, xbrl_data: Sheet1Data) -> None:
    """Merge PDF detailed items into XBRL data.

    PDF has detailed line items (cv_*, ga_*), XBRL has validated totals.
    This combines the best of both sources.

    Args:
        pdf_data: Sheet1Data from PDF (detailed breakdown)
        xbrl_data: Sheet1Data from XBRL (totals) - modified in place
    """
    # Copy detailed line items from PDF to XBRL data
    detail_fields = [
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
        "ga_gastos_personal",
        "ga_materiales",
        "ga_servicios_terceros",
        "ga_gratificacion",
        "ga_comercializacion",
        "ga_otros",
    ]

    for field_name in detail_fields:
        pdf_value = getattr(pdf_data, field_name, None)
        if pdf_value is not None:
            setattr(xbrl_data, field_name, pdf_value)


def print_sheet1_report(data: Sheet1Data) -> None:
    """Print a formatted Sheet1 report.

    Args:
        data: Sheet1Data to report
    """
    print(f"\n{'=' * 60}")
    print(f"Sheet1 Report: {data.quarter}")
    print(f"{'=' * 60}")
    print(f"Source: {data.source}")
    print(f"XBRL Available: {'Yes' if data.xbrl_available else 'No'}")

    print(f"\n{'Row':<4} {'Label':<45} {'Value':>12}")
    print("-" * 65)

    for row_num, label, value in data.to_row_list():
        if value is not None:
            val_str = f"{value:,}"
        elif label:
            val_str = ""
        else:
            val_str = ""
        print(f"{row_num:<4} {label:<45} {val_str:>12}")

    print(f"\n{'=' * 60}\n")
