# 06 - Locate Data Sources for Sheet 2

## Objective

Identify and document data sources for Sheet 2 (Estado de Resultados / Income Statement).

## Context

Sheet 2 typically contains the Estado de Resultados (Income Statement), including:
- Ingresos (Revenue)
- Costos (Cost of Sales)
- Gastos (Expenses)
- Resultado (Profit/Loss)

## Steps

Follow the same pattern as `03_locate_sheet1.md`:

### 1. Explore XBRL for Income Statement Facts

```python
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file
from puco_eeff.config import get_period_paths

year, quarter = 2024, 3
paths = get_period_paths(year, quarter)
xml_path = paths["raw_xbrl"] / f"EEFF_{year}_Q{quarter}.xml"

data = parse_xbrl_file(xml_path)
fact_names = set(f["name"] for f in data["facts"])

# Look for income statement facts
income_keywords = ["Revenue", "Profit", "Loss", "Income", "Expense", "Cost", 
                   "Ingreso", "Gasto", "Costo", "Resultado", "Utilidad"]
income_facts = [n for n in fact_names if any(k.lower() in n.lower() for k in income_keywords)]

print(f"Income statement related facts ({len(income_facts)}):")
for name in sorted(income_facts)[:25]:
    print(f"  - {name}")
```

### 2. Find PDF Sections

```python
from puco_eeff.extractor.pdf_parser import find_section_in_pdf

pdf_path = paths["raw_pdf"] / f"EEFF_{year}_Q{quarter}.pdf"

sections = find_section_in_pdf(pdf_path, "Estado de Resultados")
print(f"Found on pages: {[s['page'] for s in sections]}")

# Also search for alternative names
alt_sections = find_section_in_pdf(pdf_path, "Estado de Resultados Integrales")
print(f"Alternative found on: {[s['page'] for s in alt_sections]}")
```

### 3. Document Mappings

Update `config/config.json`:

```json
{
  "sheets": {
    "sheet2": {
      "name": "Estado de Resultados",
      "description": "Income Statement / Statement of Profit or Loss",
      "xml_paths": [
        "//ifrs-full:Revenue",
        "//ifrs-full:CostOfSales",
        "//ifrs-full:GrossProfit",
        "//ifrs-full:AdministrativeExpense",
        "//ifrs-full:SellingExpense",
        "//ifrs-full:FinanceCosts",
        "//ifrs-full:ProfitLossBeforeTax",
        "//ifrs-full:IncomeTaxExpenseContinuingOperations",
        "//ifrs-full:ProfitLoss"
      ],
      "pdf_sections": [
        "Estado de Resultados",
        "Estado de Resultados Integrales"
      ]
    }
  }
}
```

## Common IFRS Income Statement Elements

| Field | Spanish | IFRS Element |
|-------|---------|--------------|
| Revenue | Ingresos | Revenue |
| Cost of Sales | Costo de Ventas | CostOfSales |
| Gross Profit | Margen Bruto | GrossProfit |
| Admin Expenses | Gastos de Administración | AdministrativeExpense |
| Selling Expenses | Gastos de Distribución | SellingExpense |
| Finance Costs | Costos Financieros | FinanceCosts |
| Profit Before Tax | Resultado Antes de Impuesto | ProfitLossBeforeTax |
| Income Tax | Impuesto a la Renta | IncomeTaxExpenseContinuingOperations |
| Net Profit | Ganancia/Pérdida Neta | ProfitLoss |

## Next Steps

Proceed to `07_extract_sheet2.md`
