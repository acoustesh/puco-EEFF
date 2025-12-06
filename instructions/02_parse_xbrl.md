# 02 - Parse XBRL and Extract Aggregate Data

## Objective

Parse the downloaded XBRL file to extract **aggregate financial data**. The XBRL contains high-level totals but NOT the detailed line-item breakdowns (those come from PDF - see `03_extract_detailed_costs.md`).

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
- Files present in `data/raw/xbrl/`:
  - `estados_financieros_YYYY_QN.xbrl` or `.xml`

## Steps

### 1. Extract Aggregates (Recommended)

Use the wrapper function for a simple one-liner extraction:

```python
from puco_eeff.config import get_period_paths
from puco_eeff.extractor.xbrl_parser import extract_xbrl_aggregates, save_xbrl_aggregates

year, quarter = 2024, 2
paths = get_period_paths(year, quarter)

# Find the XBRL file
xml_path = paths["raw_xbrl"] / f"estados_financieros_{year}_Q{quarter}.xml"
if not xml_path.exists():
    xml_path = paths["raw_xbrl"] / f"estados_financieros_{year}_Q{quarter}.xbrl"

# Extract all aggregates in one call
aggregates = extract_xbrl_aggregates(xml_path)

# Display results
print(f"Period: {aggregates['period']['start']} to {aggregates['period']['end']}")
print("\n=== Aggregate Values ===")
for name, value in aggregates["aggregates"].items():
    if value is not None:
        print(f"  {name}: {value:,}")

# Save to JSON
output_path = paths["processed"] / "xbrl_aggregates.json"
save_xbrl_aggregates(aggregates, output_path)
print(f"\n✓ Saved to: {output_path}")
```

### 2. Verify XBRL File Exists

```python
from puco_eeff.config import get_period_paths

year, quarter = 2024, 2
paths = get_period_paths(year, quarter)

xml_path = paths["raw_xbrl"] / f"estados_financieros_{year}_Q{quarter}.xml"

if xml_path.exists():
    print(f"✓ XBRL file found: {xml_path}")
    print(f"  Size: {xml_path.stat().st_size:,} bytes")
else:
    print("✗ XBRL file not found. Run 01_download_all.md first.")
```

### 3. Explore XBRL Structure (Optional)

For debugging or exploring the XBRL structure:

```python
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file, summarize_facts

data = parse_xbrl_file(xml_path)

print(f"Contexts: {len(data['contexts'])}")
print(f"Facts: {len(data['facts'])}")

# Summarize by category
summary = summarize_facts(data)
print("\n=== Fact Categories ===")
for cat, count in list(summary.items())[:10]:
    print(f"  {cat}: {count}")
```

### 4. Get Specific Facts (Optional)

```python
from puco_eeff.extractor.xbrl_parser import get_facts_by_name

# Search for specific fact
revenue_facts = get_facts_by_name(data, "Revenue", exact=True)
for fact in revenue_facts:
    print(f"Revenue: {fact['value']} (context: {fact['context_ref']})")
```

## Q2 2024 Reference Values (IIQ2024)

From actual XBRL extraction:

| XBRL Fact Name | Spanish Name | Value (MUS$) |
|----------------|--------------|--------------|
| `RevenueFromContractsWithCustomers` | Ingresos de actividades ordinarias | 179,165 |
| `CostOfSales` | Costo de ventas | -126,202 |
| `GrossProfit` | Ganancia bruta | 52,963 |
| `AdministrativeExpense` | Gastos de administración y ventas | -11,632 |
| `ProfitLoss` | Ganancia del periodo | TBD |

## What XBRL Does NOT Contain

The detailed line-item breakdowns are NOT in XBRL. You must extract from PDF:

### Costo de Venta - Detailed Breakdown (PDF Nota 21)
- 11 line items (Gastos en personal, Materiales, Energía, etc.)
- See `03_extract_detailed_costs.md`

### Gastos Adm y Ventas - Detailed Breakdown (PDF Nota 22)
- 6 line items
- See `03_extract_detailed_costs.md`

## Output

- XBRL aggregate data saved to `data/processed/xbrl_aggregates.json`
- Use these as validation totals when extracting PDF details

## Next Steps

1. Use XBRL aggregates as validation totals
2. Proceed to `03_extract_detailed_costs.md` for PDF extraction
3. Cross-validate: Sum of PDF line items should match XBRL totals

## Troubleshooting

### XML parsing errors
- Check encoding (UTF-8 or ISO-8859-1)
- The parser handles both automatically

### Facts not found
- Try `get_facts_by_name(data, "Revenue", exact=False)` for partial matches
- Check `summarize_facts(data)` to see what categories exist

### Wrong period values
- `extract_xbrl_aggregates()` automatically filters to the latest period
- For specific periods, filter by context manually
