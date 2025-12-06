# 07 - Extract Data for Sheet 2

## Objective

Extract Estado de Resultados (Income Statement) data following the same pattern as Sheet 1.

## Steps

### 1. Extract from XBRL

```python
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file, extract_by_xpath
from puco_eeff.config import get_config, get_period_paths

year, quarter = 2024, 3
paths = get_period_paths(year, quarter)
config = get_config()

xml_path = paths["raw_xbrl"] / f"EEFF_{year}_Q{quarter}.xml"
sheet2_config = config["sheets"]["sheet2"]

extracted_data = {}
for xpath in sheet2_config.get("xml_paths", []):
    values = extract_by_xpath(xml_path, xpath)
    field_name = xpath.split(":")[-1]
    extracted_data[field_name] = values
    print(f"{field_name}: {values}")
```

### 2. Extract from PDF (if needed)

```python
from puco_eeff.extractor.pdf_parser import extract_tables_from_pdf, find_section_in_pdf

pdf_path = paths["raw_pdf"] / f"EEFF_{year}_Q{quarter}.pdf"
sections = find_section_in_pdf(pdf_path, "Estado de Resultados")

if sections:
    pages = [s['page'] for s in sections]
    tables = extract_tables_from_pdf(pdf_path, pages=pages)
```

### 3. OCR Fallback

```python
from puco_eeff.extractor.ocr_fallback import ocr_with_fallback

if not extracted_data or all(not v for v in extracted_data.values()):
    ocr_result = ocr_with_fallback(
        pdf_path=pdf_path,
        prompt="""Extract the Estado de Resultados (Income Statement) table.
Return as markdown table with:
- Concepto
- Período Actual
- Período Anterior

Include: Ingresos, Costos, Gastos, Resultado operacional, Resultado neto.""",
        audit_dir=paths["audit"]
    )
```

### 4. Structure and Save

```python
import pandas as pd
from puco_eeff.writer.sheet_writer import save_sheet_data
from puco_eeff.transformer.source_tracker import SourceTracker

# Create DataFrame with extracted data
income_statement = pd.DataFrame([
    {"concepto": "Ingresos de actividades ordinarias", "valor": extracted_data.get("Revenue", [None])[0]},
    {"concepto": "Costo de ventas", "valor": extracted_data.get("CostOfSales", [None])[0]},
    {"concepto": "Ganancia bruta", "valor": extracted_data.get("GrossProfit", [None])[0]},
    {"concepto": "Gastos de administración", "valor": extracted_data.get("AdministrativeExpense", [None])[0]},
    {"concepto": "Costos financieros", "valor": extracted_data.get("FinanceCosts", [None])[0]},
    {"concepto": "Ganancia antes de impuesto", "valor": extracted_data.get("ProfitLossBeforeTax", [None])[0]},
    {"concepto": "Gasto por impuesto", "valor": extracted_data.get("IncomeTaxExpenseContinuingOperations", [None])[0]},
    {"concepto": "Ganancia del período", "valor": extracted_data.get("ProfitLoss", [None])[0]},
])

# Save
save_sheet_data("sheet2", income_statement, f"{year}_Q{quarter}")

# Track sources
tracker = SourceTracker(period=f"{year}_Q{quarter}")
for field, values in extracted_data.items():
    if values:
        tracker.add_source(field, "xml", str(xml_path), f"//ifrs-full:{field}", "xbrl_parser")
tracker.save()
```

## Next Steps

Proceed to `08_format_sheet2.md`
