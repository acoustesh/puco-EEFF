# 03 - Extract Detailed Cost Breakdown from PDF

## Objective

Extract the **detailed line-item breakdowns** from the Estados Financieros PDF that are NOT available in XBRL:
- **Nota 21**: Costo de Venta (11 line items)
- **Nota 22**: Gastos de Administración y Ventas (6 line items)

**Key Features:**
- Works with both CMF Chile (PDF + XBRL) and Pucobre.cl fallback (PDF only)
- Automatically validates against XBRL when available
- Handles cases when no XBRL exists (Q1 2024 from Pucobre.cl)

## Data Sources

### CMF Chile (Primary)
- **PDF**: `estados_financieros_YYYY_QN.pdf`
- **XBRL**: `estados_financieros_YYYY_QN.xbrl` (for validation)
- **XBRL Contains**: Aggregated totals (CostOfSales, AdministrativeExpense)
- **PDF Contains**: Detailed line-item breakdowns not in XBRL

### Pucobre.cl (Fallback)
- **PDF**: `estados_financieros_YYYY_QN.pdf` (split from combined)
- **XBRL**: NOT AVAILABLE
- **Use Case**: Q1 of any year (not on CMF Chile)

## Prerequisites

- Documents downloaded via `01_download_all.md`
- For CMF source: XBRL aggregates extracted via `02_parse_xbrl.md` (optional but recommended)

## Quick Start (Using the Module)

\`\`\`python
from puco_eeff.extractor import (
    extract_detailed_costs,
    print_extraction_report,
    save_extraction_result,
)

# Extract with automatic validation
result = extract_detailed_costs(year=2024, quarter=2)

# Print formatted report
print_extraction_report(result)

# Save to JSON
save_extraction_result(result)

# Check results
print(f"Source: {result.source}")
print(f"XBRL Available: {result.xbrl_available}")
print(f"Extraction Valid: {result.is_valid()}")
\`\`\`

## Target Data Structure

### Nota 21 - Costo de Venta (Cost of Sales Breakdown)

| Concepto | XBRL Field | Typical Value (Q3 2024) |
|----------|-----------|-------------------------|
| Gastos en personal | N/A | (30,294) |
| Materiales y repuestos | N/A | (37,269) |
| Energía eléctrica | N/A | (14,710) |
| Servicios de terceros | N/A | (39,213) |
| Depreciación y amort. del periodo | N/A | (33,178) |
| Depreciación Activos en leasing | N/A | (1,354) |
| Depreciación Arrendamientos | N/A | (2,168) |
| Servicios mineros de terceros | N/A | (19,577) |
| Fletes y otros gastos operacionales | N/A | (7,405) |
| Gastos Diferidos, ajustes existencias y otros | N/A | 20,968 |
| Obligaciones por convenios colectivos | N/A | (6,662) |
| **Total (debe = XBRL CostOfSales)** | CostOfSales | **(170,862)** |

### Nota 22 - Gastos de Administración y Ventas

| Concepto | XBRL Field | Typical Value (Q3 2024) |
|----------|-----------|-------------------------|
| Gastos en personal | N/A | (5,727) |
| Materiales y repuestos | N/A | (226) |
| Servicios de terceros | N/A | (4,474) |
| Provisión gratificación legal y otros | N/A | (2,766) |
| Gastos comercialización | N/A | (3,506) |
| Otros gastos | N/A | (664) |
| **Total (debe = XBRL AdminExpense)** | AdministrativeExpense | **(17,363)** |

---

## Extraction Scenarios

### Scenario 1: CMF Chile Source (With XBRL Validation)

When data is downloaded from CMF Chile, both PDF and XBRL are available:

\`\`\`python
from puco_eeff.extractor import extract_detailed_costs, print_extraction_report

# Q2-Q4 typically from CMF Chile
result = extract_detailed_costs(year=2024, quarter=2, validate=True)

print_extraction_report(result)

# Example output:
# ============================================================
# Cost Extraction Report: 2024 Q2
# ============================================================
# Source: cmf
# XBRL Available: Yes
#
# --- Validation Results ---
#   Costo de Venta (Nota 21):
#     PDF:  170,862
#     XBRL: 170,862
#     ✓ Match
#   Gastos Administración (Nota 22):
#     PDF:  17,363
#     XBRL: 17,363
#     ✓ Match
\`\`\`

### Scenario 2: Pucobre.cl Fallback (PDF Only, No XBRL)

When Q1 data comes from Pucobre.cl, only PDF is available:

\`\`\`python
# Q1 typically from Pucobre.cl fallback
result = extract_detailed_costs(year=2024, quarter=1, validate=True)

print_extraction_report(result)

# Example output:
# ============================================================
# Cost Extraction Report: 2024 Q1
# ============================================================
# Source: pucobre.cl
# XBRL Available: No
#
# --- Nota 21: Costo de Venta ---
# Page: 68
# Items extracted: 11
#   Gastos en personal: 9,542
#   Materiales y repuestos: 12,103
#   ...
#   TOTAL: 54,287
#
# --- Validation Results ---
#   Costo de Venta (Nota 21):
#     PDF:  54,287
#     ⚠ PDF only (no XBRL)
\`\`\`

### Scenario 3: Batch Extraction Across Quarters

\`\`\`python
from puco_eeff.extractor import extract_detailed_costs, save_extraction_result

results = {}
for quarter in [1, 2, 3, 4]:
    result = extract_detailed_costs(year=2024, quarter=quarter)
    results[f"Q{quarter}"] = result
    save_extraction_result(result)
    print(f"Q{quarter}: Source={result.source}, Valid={result.is_valid()}")

# Q1: Source=pucobre.cl, Valid=True (PDF only)
# Q2: Source=cmf, Valid=True (validated against XBRL)
# Q3: Source=cmf, Valid=True (validated against XBRL)
# Q4: Source=cmf, Valid=True (validated against XBRL)
\`\`\`

---

## Manual Extraction Steps (If Needed)

### 1. Load PDF and Find Nota 21

\`\`\`python
import pdfplumber
from puco_eeff.config import get_period_paths

year, quarter = 2024, 3  # Adjust as needed
paths = get_period_paths(year, quarter)

pdf_path = paths["raw_pdf"] / f"estados_financieros_{year}_Q{quarter}.pdf"

# Find page with Nota 21
nota_21_page = None
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if "21. COSTO DE VENTA" in text.upper() or "NOTA 21" in text.upper():
            nota_21_page = i
            print(f"✓ Found Nota 21 on page {i + 1}")
            break

if nota_21_page is None:
    # Fallback: search for "COSTO DE VENTA" as header
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if "21. COSTO DE VENTA" in text:
                nota_21_page = i
                break

print(f"Nota 21 is on page index {nota_21_page} (display: {nota_21_page + 1})")
\`\`\`

### 2. Extract Tables from Page

\`\`\`python
def extract_cost_tables(pdf_path, page_index):
    """Extract Nota 21 and Nota 22 tables from PDF."""

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]

        # Extract all tables from this page
        tables = page.extract_tables()

        results = {
            "nota_21": None,  # Costo de Venta
            "nota_22": None,  # Gastos Admin
        }

        for table in tables:
            if not table or len(table) < 3:
                continue

            # Check if this is Nota 21 (Costo de Venta)
            text_content = str(table).lower()

            if "gastos en personal" in text_content and "energía eléctrica" in text_content:
                results["nota_21"] = table
                print("✓ Found Nota 21 (Costo de Venta) table")

            elif "gastos en personal" in text_content and "gastos comercialización" in text_content:
                results["nota_22"] = table
                print("✓ Found Nota 22 (Gastos Admin) table")

        return results

# Extract tables
tables = extract_cost_tables(pdf_path, nota_21_page)
\`\`\`

### 3. Parse Chilean Number Format

\`\`\`python
import re

def parse_chilean_number(value):
    """Parse Chilean-formatted numbers.

    - Period as thousands separator: 30.294 = 30,294
    - Parentheses for negatives: (30.294) = -30,294
    """
    if not value:
        return None

    value = str(value).strip()
    is_negative = "(" in value and ")" in value

    # Remove non-numeric except period and minus
    value = re.sub(r"[^\d.\-]", "", value)
    if not value or value in (".", "-"):
        return None

    try:
        value = value.replace(".", "")  # Remove thousands sep
        result = int(value)
        return -abs(result) if is_negative else result
    except ValueError:
        return None
\`\`\`

### 4. Validate Against XBRL (When Available)

\`\`\`python
from puco_eeff.extractor import parse_xbrl_file, get_facts_by_name

# Check if XBRL exists
xbrl_path = paths["raw_xbrl"] / f"estados_financieros_{year}_Q{quarter}.xbrl"

if xbrl_path.exists():
    print("✓ XBRL available - performing validation")

    data = parse_xbrl_file(xbrl_path)

    # Get CostOfSales from XBRL
    cost_facts = get_facts_by_name(data, "CostOfSales")
    xbrl_cost = int(float(cost_facts[0]["value"])) if cost_facts else None

    # Get AdminExpense from XBRL
    admin_facts = get_facts_by_name(data, "AdministrativeExpense")
    xbrl_admin = int(float(admin_facts[0]["value"])) if admin_facts else None

    # Compare with PDF totals
    print(f"\nCosto de Venta:")
    print(f"  PDF:  {pdf_cost_total:,}")
    print(f"  XBRL: {xbrl_cost:,}")
    print(f"  Match: {'✓' if abs(pdf_cost_total) == abs(xbrl_cost) else '✗ MISMATCH!'}")

else:
    print("⚠ No XBRL available - using PDF-only extraction")
    print("  Source is likely Pucobre.cl fallback (Q1 data)")
\`\`\`

### 5. OCR Fallback (If Tables Cannot Be Extracted)

If pdfplumber fails to extract tables (image-based PDF), use OCR:

\`\`\`python
from puco_eeff.extractor import ocr_with_fallback

# OCR prompt for structured extraction
ocr_prompt = """
Extract the following tables from Nota 21 and Nota 22 of this financial statement:

NOTA 21 - COSTO DE VENTA:
Extract all line items with their values for the YTD column.
Expected items: Gastos en personal, Materiales y repuestos, Energía eléctrica, etc.

NOTA 22 - GASTOS DE ADMINISTRACION Y VENTAS:
Extract all line items with their values for the YTD column.

Return as JSON format:
{
    "nota_21": [{"concepto": "...", "valor": number}, ...],
    "nota_22": [{"concepto": "...", "valor": number}, ...]
}

Numbers in parentheses like (30.294) represent negative values: -30294
Values are in thousands of US dollars (MUS$).
"""

if not tables["nota_21"] or not tables["nota_22"]:
    print("Using OCR fallback for table extraction...")

    audit_dir = paths["audit"]
    audit_dir.mkdir(parents=True, exist_ok=True)

    ocr_result = ocr_with_fallback(
        pdf_path=pdf_path,
        prompt=ocr_prompt,
        pages=[nota_21_page + 1],  # 1-indexed for OCR
        save_all_responses=True,
        audit_dir=audit_dir
    )

    if ocr_result["success"]:
        import json
        try:
            ocr_data = json.loads(ocr_result["content"])
            print("✓ OCR extraction successful")
        except json.JSONDecodeError:
            print("⚠ OCR returned non-JSON, needs manual parsing")
\`\`\`

---

## Output Files

| File | Description |
|------|-------------|
| \`data/processed/detailed_costs.json\` | Extracted line items + validation |
| \`data/processed/xbrl_aggregates.json\` | XBRL totals (when available) |
| \`audit/YYYY_QN/ocr_*.json\` | OCR responses (if used) |

### Example Output (detailed_costs.json)

\`\`\`json
{
  "period": "2024_Q2",
  "source": "cmf",
  "pdf_path": "data/raw/pdf/estados_financieros_2024_Q2.pdf",
  "xbrl_path": "data/raw/xbrl/estados_financieros_2024_Q2.xbrl",
  "xbrl_available": true,
  "nota_21": {
    "nota_number": 21,
    "nota_title": "Costo de Venta",
    "page_number": 71,
    "items": [
      {"concepto": "Gastos en personal", "ytd_actual": -30294, "ytd_anterior": -28150},
      {"concepto": "Materiales y repuestos", "ytd_actual": -37269, "ytd_anterior": -34521}
    ],
    "total_ytd_actual": -170862
  },
  "nota_22": {
    "nota_number": 22,
    "nota_title": "Gastos de Administración y Ventas",
    "page_number": 71,
    "items": [...],
    "total_ytd_actual": -17363
  },
  "validations": [
    {
      "field": "Costo de Venta (Nota 21)",
      "pdf_value": -170862,
      "xbrl_value": -170862,
      "match": true,
      "source": "both",
      "status": "✓ Match"
    }
  ]
}
\`\`\`

---

## Number Format Notes

Chilean financial documents may use:
- **Period separators**: \`30.294\` means 30,294 (thousands separator)
- **Parentheses**: \`(30.294)\` means negative -30,294
- **Currency**: MUS$ = Miles de dólares estadounidenses (thousands of USD)

Always parse parentheses as negative values.

---

## Validation Checklist

### When XBRL is Available (CMF Source)
- [ ] Sum of Nota 21 line items = Total Costo de Venta
- [ ] Total Costo de Venta = XBRL CostOfSales ✓
- [ ] Sum of Nota 22 line items = Total Gastos Admin
- [ ] Total Gastos Admin = XBRL AdministrativeExpense ✓
- [ ] No line items with unexpected NULL values

### When XBRL is NOT Available (Pucobre.cl Source)
- [ ] Sum of Nota 21 line items = Total Costo de Venta (internal check)
- [ ] Sum of Nota 22 line items = Total Gastos Admin (internal check)
- [ ] All expected line items extracted (11 for Nota 21, 6 for Nota 22)
- [ ] Values appear reasonable compared to other quarters
- [ ] No line items with unexpected NULL values

---

## Troubleshooting

### Issue: "Could not find Nota 21 in PDF"
- The page number may vary. Try searching for different patterns:
  - "COSTO DE VENTA"
  - "NOTA 21"
  - "21."
- For Pucobre.cl combined PDFs, make sure the PDF was split correctly

### Issue: "Tables not extracted by pdfplumber"
- Some PDFs may have tables as images. Use OCR fallback.
- Check if PDF is scanned vs native text

### Issue: "XBRL values don't match PDF"
- Sign differences: XBRL may report positive, PDF shows parentheses (negative)
- Scale differences: Both should be in thousands (MUS\$)
- Period mismatch: Ensure you're comparing YTD values

### Issue: "Q1 data not available on CMF"
- This is expected! Q1 is typically only on Pucobre.cl
- Use the fallback downloader: \`download_from_pucobre(year, 1)\`
- Extraction will work in PDF-only mode

---

## Next Steps

1. Continue to \`04_extract_main_statements.md\` for Income Statement and Balance Sheet
2. Format data for Excel output in \`05_format_sheets.md\`
