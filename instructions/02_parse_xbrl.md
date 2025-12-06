# 02 - Parse XBRL and Extract Aggregate Data

## Objective

Parse the downloaded XBRL file to extract **aggregate financial data**. The XBRL contains high-level totals but NOT the detailed line-item breakdowns (those come from PDF - see `03_extract_sheet1.md`).

## What XBRL Contains

The XBRL provides **aggregate totals** only:
- Ingresos de actividades ordinarias (Revenue): ✓
- Costo de ventas total: ✓
- Ganancia bruta: ✓
- Gastos de administración y ventas total: ✓
- Ganancia (pérdida) del periodo: ✓

**NOT in XBRL** (requires PDF extraction):
- Detailed Costo de Venta breakdown (11 line items)
- Detailed Gastos Adm y Ventas breakdown (6 line items)

## Prerequisites

- Documents downloaded via `01_download_all.md`
- Files present:
  - `data/raw/pdf/estados_financieros_YYYY_QN_xbrl.zip`
  - `data/raw/pdf/estados_financieros_YYYY_QN.xml` (extracted)

## Context

XBRL (eXtensible Business Reporting Language) provides structured financial data that can be directly parsed without OCR. This is the **preferred data source** when available.

The XBRL file contains:
- **Contexts**: Define the reporting period and entity
- **Facts**: Individual financial data points with values
- **Units**: Currencies and measurement units

## Steps

### 1. Load the Downloaded XBRL

```python
from pathlib import Path
from puco_eeff.config import get_period_paths, DATA_DIR

year, quarter = 2024, 3  # Adjust as needed
paths = get_period_paths(year, quarter)

# XBRL is extracted to the raw_pdf directory (same as other downloads)
xml_path = paths["raw_pdf"] / f"estados_financieros_{year}_Q{quarter}.xml"

if xml_path.exists():
    print(f"✓ XBRL file found: {xml_path}")
    print(f"  Size: {xml_path.stat().st_size:,} bytes")
else:
    print("✗ XBRL file not found. Run 01_download_all.md first.")
```

### 2. Parse XBRL Structure

```python
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file

# Parse the XBRL file
data = parse_xbrl_file(xml_path)

print(f"\n=== XBRL Summary ===")
print(f"Contexts: {len(data['contexts'])}")
print(f"Facts: {len(data['facts'])}")
print(f"Units: {len(data['units'])}")

# Show sample contexts
print("\n=== Sample Contexts ===")
for ctx_id, ctx_info in list(data['contexts'].items())[:5]:
    print(f"  {ctx_id}: {ctx_info}")
```

### 3. Explore Available Facts

```python
# Get unique fact names
fact_names = set(f["name"] for f in data["facts"])
print(f"\nFound {len(fact_names)} unique fact types")

# Group facts by namespace/category
fact_categories = {}
for name in fact_names:
    # Extract category from fact name
    parts = name.split("_") if "_" in name else [name]
    category = parts[0] if len(parts) > 1 else "Other"
    fact_categories.setdefault(category, []).append(name)

print("\n=== Fact Categories ===")
for cat, names in sorted(fact_categories.items()):
    print(f"  {cat}: {len(names)} facts")
```

### 4. Extract Aggregate Financial Data (CONFIRMED FACT NAMES)

These fact names have been verified against actual Pucobre XBRL data:

```python
from puco_eeff.extractor.xbrl_parser import get_facts_by_name

# === INCOME STATEMENT AGGREGATES (Estado de Resultados) ===
# These are the exact fact names from Pucobre's XBRL taxonomy

aggregate_facts = {
    # Revenue and Cost
    "Revenue": "Ingresos de actividades ordinarias",
    "CostOfSales": "Costo de ventas (total)",
    "GrossProfit": "Ganancia bruta",
    
    # Operating expenses
    "AdministrativeExpense": "Gastos de administración",
    "SellingExpense": "Gastos de ventas (if separate)",
    
    # Financial items
    "FinanceIncome": "Ingresos financieros",
    "FinanceCosts": "Costos financieros",
    
    # Results
    "ProfitLossBeforeTax": "Ganancia antes de impuestos",
    "IncomeTaxExpenseContinuingOperations": "Gasto por impuesto a las ganancias",
    "ProfitLoss": "Ganancia (pérdida) del periodo",
}

# Extract and display each
print("=== XBRL AGGREGATE DATA ===\n")
for fact_name, spanish_name in aggregate_facts.items():
    facts = get_facts_by_name(data, fact_name)
    if facts:
        # Get the YTD value (9-month accumulation for Q3)
        for f in facts:
            ctx = f.get("context", "")
            value = f.get("value", "N/A")
            # Format large numbers
            if value and value.isdigit():
                value = f"{int(value):,}"
            print(f"{spanish_name}:")
            print(f"  Fact: {fact_name}")
            print(f"  Value: {value}")
            print(f"  Context: {ctx}")
            print()
```

### 5. Q3 2024 Verified Values

From actual Q3 2024 XBRL extraction:

| XBRL Fact Name | Spanish Name | Value (MUS$) |
|----------------|--------------|--------------|
| `Revenue` | Ingresos de actividades ordinarias | 231,472 |
| `CostOfSales` | Costo de ventas | 170,862 |
| `GrossProfit` | Ganancia bruta | 60,610 |
| `AdministrativeExpense` | Gastos de administración y ventas | 17,363 |
| `ProfitLoss` | Ganancia del periodo | 27,882 |

### 6. Balance Sheet Facts (Estado de Situación)

```python
# Balance sheet keywords
balance_facts = [
    "Assets",
    "CurrentAssets", 
    "NoncurrentAssets",
    "Liabilities",
    "CurrentLiabilities",
    "NoncurrentLiabilities",
    "Equity",
    "CashAndCashEquivalents",
    "Inventories",
    "TradeAndOtherCurrentReceivables",
    "PropertyPlantAndEquipment",
    "IntangibleAssetsOtherThanGoodwill",
    "Goodwill",
]

print("\n=== Balance Sheet Data ===")
for fact_name in balance_facts:
    facts = get_facts_by_name(data, fact_name)
    if facts:
        for f in facts[:1]:  # Just show first match
            value = f.get("value", "N/A")
            if value and value.lstrip("-").isdigit():
                value = f"{int(value):,}"
            print(f"{fact_name}: {value}")
```

### 7. Cash Flow Facts (Estado de Flujos de Efectivo)

```python
cashflow_facts = [
    "CashFlowsFromUsedInOperatingActivities",
    "CashFlowsFromUsedInInvestingActivities",
    "CashFlowsFromUsedInFinancingActivities",
    "IncreaseDecreaseInCashAndCashEquivalents",
    "CashAndCashEquivalentsAtBeginningOfPeriod",
    "CashAndCashEquivalentsAtEndOfPeriod",
]

print("\n=== Cash Flow Data ===")
for fact_name in cashflow_facts:
    facts = get_facts_by_name(data, fact_name)
    if facts:
        for f in facts[:1]:
            value = f.get("value", "N/A")
            if value and value.lstrip("-").isdigit():
                value = f"{int(value):,}"
            print(f"{fact_name}: {value}")
```

### 8. Export XBRL Data for Excel Sheet 1

```python
import json

# Aggregate data from XBRL for Excel Sheet 1
xbrl_aggregates = {
    "ingresos_ordinarios": get_facts_by_name(data, "Revenue")[0]["value"] if get_facts_by_name(data, "Revenue") else None,
    "costo_ventas_total": get_facts_by_name(data, "CostOfSales")[0]["value"] if get_facts_by_name(data, "CostOfSales") else None,
    "ganancia_bruta": get_facts_by_name(data, "GrossProfit")[0]["value"] if get_facts_by_name(data, "GrossProfit") else None,
    "gastos_admin_ventas_total": get_facts_by_name(data, "AdministrativeExpense")[0]["value"] if get_facts_by_name(data, "AdministrativeExpense") else None,
    "ganancia_periodo": get_facts_by_name(data, "ProfitLoss")[0]["value"] if get_facts_by_name(data, "ProfitLoss") else None,
}

# Save for later use
output_path = paths["processed"] / "xbrl_aggregates.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(xbrl_aggregates, f, indent=2)

print(f"\n✓ XBRL aggregates saved to: {output_path}")
print("\nValues extracted:")
for key, value in xbrl_aggregates.items():
    if value:
        print(f"  {key}: {int(value):,}")
```

## What XBRL Does NOT Contain

The detailed line-item breakdowns are NOT in XBRL. You must extract from PDF:

### Costo de Venta - Detailed Breakdown (PDF Nota 21, Page 71)
| Line Item | Q3 2024 Value |
|-----------|---------------|
| Gastos en personal | (30,294) |
| Materiales y repuestos | (37,269) |
| Energía eléctrica | (14,710) |
| Servicios de terceros | (39,213) |
| Depreciación y amort. del periodo | (33,178) |
| Depreciación Activos en leasing | (1,354) |
| Depreciación Arrendamientos | (2,168) |
| Servicios mineros de terceros | (19,577) |
| Fletes y otros gastos operacionales | (7,405) |
| Gastos Diferidos, ajustes existencias y otros | 20,968 |
| Obligaciones por convenios colectivos | (6,662) |
| **Total** | **(170,862)** |

### Gastos Adm y Ventas - Detailed Breakdown (PDF Nota 22, Page 71)
| Line Item | Q3 2024 Value |
|-----------|---------------|
| Gastos en personal | (5,727) |
| Materiales y repuestos | (226) |
| Servicios de terceros | (4,474) |
| Provisión gratificación legal y otros | (2,766) |
| Gastos comercialización | (3,506) |
| Otros gastos | (664) |
| **Total** | **(17,363)** |

## Output

- XBRL aggregate data saved to `data/processed/YYYY_QN/xbrl_aggregates.json`
- Understanding that detailed breakdown requires PDF extraction

## Next Steps

Since detailed line items require PDF extraction:
1. Use aggregates from XBRL as validation totals
2. Proceed to `03_extract_sheet1.md` for PDF extraction of Nota 21 and Nota 22
3. Cross-validate: Sum of PDF line items should match XBRL totals

## Troubleshooting

### XML parsing errors
- Check if the file is valid XML: `xmllint --noout file.xml`
- Ensure proper encoding (UTF-8)

### Facts not found
- Try different namespace prefixes
- Search for partial matches
- Check if data is in a different period context

### Empty values
- Some facts may only have values for specific contexts
- Filter by context period to get the right value
