# 03 - Extract Detailed Cost Breakdown from PDF

## Objective

Extract the **detailed line-item breakdowns** from the Estados Financieros PDF that are NOT available in XBRL:
- **Nota 21**: Costo de Venta (11 line items)
- **Nota 22**: Gastos de Administración y Ventas (6 line items)

## Data Location

The detailed breakdowns are in:
- **PDF**: `estados_financieros_YYYY_QN.pdf`
- **Page**: 71 (may vary by period - search for "NOTA 21" or "COSTO DE VENTA")

## Prerequisites

- Documents downloaded via `01_download_all.md`
- XBRL aggregates extracted via `02_parse_xbrl.md`

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

## Steps

### 1. Load PDF and Find Nota 21

```python
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
```

### 2. Extract Tables from Page 71 (or found page)

```python
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
            # Look for "Gastos en personal" in first column
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
```

### 3. Parse Nota 21 - Costo de Venta

```python
import pandas as pd
import re

def parse_nota_21(table_data):
    """
    Parse Nota 21 table into structured data.

    Expected columns:
    - Concepto (may be merged with values)
    - 01-01-YYYY to 30-09-YYYY (YTD current)
    - 01-01-YYYY to 30-09-YYYY (YTD prior)
    - 01-07-YYYY to 30-09-YYYY (Q3 current)
    - 01-07-YYYY to 30-09-YYYY (Q3 prior)
    """

    # Expected line items in order
    expected_items = [
        "Gastos en personal",
        "Materiales y repuestos",
        "Energía eléctrica",
        "Servicios de terceros",
        "Depreciación y amort. del periodo",
        "Depreciación Activos en leasing",
        "Depreciación Arrendamientos",
        "Servicios mineros de terceros",
        "Fletes y otros gastos operacionales",
        "Gastos Diferidos",  # May include more text
        "Obligaciones por convenios colectivos",
        "Totales",
    ]

    # Parse table rows
    parsed = []
    for row in table_data:
        if not row:
            continue

        # Get the text content (may be in first cell or merged)
        row_text = str(row[0]) if row[0] else ""

        # Split newline-merged content
        lines = row_text.split("\n")

        # Extract numbers from remaining columns
        values = []
        for cell in row[1:]:
            if cell:
                # Clean and parse number
                num_str = str(cell).replace(",", "").replace(".", "")
                num_str = num_str.replace("(", "-").replace(")", "")
                num_str = re.sub(r"[^\d\-]", "", num_str)
                if num_str:
                    try:
                        values.append(int(num_str))
                    except:
                        values.append(None)

        # Match to expected items
        for line in lines:
            for expected in expected_items:
                if expected.lower() in line.lower():
                    parsed.append({
                        "concepto": expected,
                        "ytd_actual": values[0] if len(values) > 0 else None,
                        "ytd_anterior": values[1] if len(values) > 1 else None,
                    })
                    break

    return pd.DataFrame(parsed)

# Parse the table
if tables["nota_21"]:
    nota_21_df = parse_nota_21(tables["nota_21"])
    print("\n=== Nota 21 - Costo de Venta ===")
    print(nota_21_df.to_string())
```

### 4. Parse Nota 22 - Gastos Administración

```python
def parse_nota_22(table_data):
    """Parse Nota 22 table into structured data."""

    expected_items = [
        "Gastos en personal",
        "Materiales y repuestos",
        "Servicios de terceros",
        "Provisión gratificación legal",
        "Gastos comercialización",
        "Otros gastos",
        "Totales",
    ]

    parsed = []
    for row in table_data:
        if not row:
            continue

        row_text = str(row[0]) if row[0] else ""
        lines = row_text.split("\n")

        values = []
        for cell in row[1:]:
            if cell:
                num_str = str(cell).replace(",", "").replace(".", "")
                num_str = num_str.replace("(", "-").replace(")", "")
                num_str = re.sub(r"[^\d\-]", "", num_str)
                if num_str:
                    try:
                        values.append(int(num_str))
                    except:
                        values.append(None)

        for line in lines:
            for expected in expected_items:
                if expected.lower() in line.lower():
                    parsed.append({
                        "concepto": expected,
                        "ytd_actual": values[0] if len(values) > 0 else None,
                        "ytd_anterior": values[1] if len(values) > 1 else None,
                    })
                    break

    return pd.DataFrame(parsed)

if tables["nota_22"]:
    nota_22_df = parse_nota_22(tables["nota_22"])
    print("\n=== Nota 22 - Gastos Admin y Ventas ===")
    print(nota_22_df.to_string())
```

### 5. Validate Against XBRL Totals

```python
import json

# Load XBRL aggregates
xbrl_path = paths["processed"] / "xbrl_aggregates.json"
if xbrl_path.exists():
    with open(xbrl_path) as f:
        xbrl = json.load(f)

    # Validate Nota 21 total matches XBRL CostOfSales
    pdf_costo_total = nota_21_df[nota_21_df["concepto"] == "Totales"]["ytd_actual"].values[0]
    xbrl_costo = int(xbrl.get("costo_ventas_total", 0))

    print(f"\n=== Validation ===")
    print(f"PDF Costo de Venta Total: {pdf_costo_total:,}")
    print(f"XBRL CostOfSales:         {xbrl_costo:,}")
    print(f"Match: {'✓' if abs(pdf_costo_total) == abs(xbrl_costo) else '✗ MISMATCH!'}")

    # Validate Nota 22 total matches XBRL AdminExpense
    pdf_admin_total = nota_22_df[nota_22_df["concepto"] == "Totales"]["ytd_actual"].values[0]
    xbrl_admin = int(xbrl.get("gastos_admin_ventas_total", 0))

    print(f"\nPDF Gastos Admin Total: {pdf_admin_total:,}")
    print(f"XBRL AdminExpense:      {xbrl_admin:,}")
    print(f"Match: {'✓' if abs(pdf_admin_total) == abs(xbrl_admin) else '✗ MISMATCH!'}")
```

### 6. OCR Fallback (If Tables Cannot Be Extracted)

If pdfplumber fails to extract tables (image-based PDF), use OCR:

```python
from puco_eeff.extractor.ocr_fallback import ocr_with_fallback

# OCR prompt for structured extraction
ocr_prompt = """
Extract the following tables from Nota 21 and Nota 22 of this financial statement:

NOTA 21 - COSTO DE VENTA:
Extract all line items with their values for the column "01-01-2024 30-09-2024".
Expected items:
- Gastos en personal
- Materiales y repuestos
- Energía eléctrica
- Servicios de terceros
- Depreciación y amort. del periodo
- Depreciación Activos en leasing
- Depreciación Arrendamientos
- Servicios mineros de terceros
- Fletes y otros gastos operacionales
- Gastos Diferidos, ajustes existencias y otros
- Obligaciones por convenios colectivos
- Totales

NOTA 22 - GASTOS DE ADMINISTRACION Y VENTAS:
Extract all line items with their values for the column "01-01-2024 30-09-2024".
Expected items:
- Gastos en personal
- Materiales y repuestos
- Servicios de terceros
- Provisión gratificación legal y otros
- Gastos comercialización
- Otros gastos
- Totales

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
        # Parse JSON from OCR response
        import json
        try:
            ocr_data = json.loads(ocr_result["content"])
            nota_21_df = pd.DataFrame(ocr_data["nota_21"])
            nota_22_df = pd.DataFrame(ocr_data["nota_22"])
            print("✓ OCR extraction successful")
        except json.JSONDecodeError:
            print("⚠ OCR returned non-JSON, needs manual parsing")
```

### 7. Save Extracted Data

```python
import json

# Prepare output structure
extracted_data = {
    "period": f"{year}_Q{quarter}",
    "source": str(pdf_path),
    "page": nota_21_page + 1,
    "nota_21_costo_venta": nota_21_df.to_dict(orient="records"),
    "nota_22_gastos_admin": nota_22_df.to_dict(orient="records"),
}

# Save to processed directory
output_path = paths["processed"] / "detailed_costs.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved extracted data to: {output_path}")
```

## Output Files

- `data/processed/YYYY_QN/detailed_costs.json` - Extracted line items
- `data/processed/YYYY_QN/xbrl_aggregates.json` - XBRL totals for validation
- `audit/YYYY_QN/ocr_*.json` - OCR responses (if used)

## Number Format Notes

Chilean financial documents may use:
- **Period separators**: `30.294` means 30,294 (thousands separator)
- **Parentheses**: `(30.294)` means negative -30,294
- **Currency**: MUS$ = Miles de dólares estadounidenses (thousands of USD)

Always parse parentheses as negative values.

## Validation Checklist

- [ ] Sum of Nota 21 line items = Total Costo de Venta
- [ ] Total Costo de Venta = XBRL CostOfSales
- [ ] Sum of Nota 22 line items = Total Gastos Admin
- [ ] Total Gastos Admin = XBRL AdministrativeExpense
- [ ] No line items with unexpected NULL values

## Next Steps

1. Continue to `04_extract_main_statements.md` for Income Statement and Balance Sheet
2. Format data for Excel output in `05_format_sheets.md`
