# 04 - Format Sheet 1 for Excel

## Objective

Format the extracted cost breakdown data into a clean structure ready for the final Excel workbook, and validate against reference values.

## Prerequisites

- Sheet 1 data extracted: `data/processed/sheet1_YYYY_QN.json`

## Steps

### 1. Load and Validate Extracted Data

```python
from puco_eeff.config import get_period_paths
from puco_eeff.writer.sheet_writer import load_sheet_data
from puco_eeff.transformer import (
    get_standard_structure,
    map_to_structure,
    validate_balance_sheet,
    validate_against_reference,
    format_validation_report,
)

year, quarter = 2024, 2
period = f"{year}_Q{quarter}"

# Load extracted data
paths = get_period_paths(year, quarter)
sheet1_data = load_sheet_data("sheet1", period)
print(f"Loaded data for {period}")
```

### 2. Validate Totals

```python
# Validate that line items sum to totals
validation_results = validate_balance_sheet(sheet1_data)

print(format_validation_report(validation_results))
```

### 3. Validate Against Reference (IIQ2024)

For Q2 2024, we have known-good reference values in `config/reference_data.json`:

```python
from puco_eeff.transformer import validate_against_reference, format_validation_report

# Format period label for config lookup (IIQ2024 format)
quarter_roman = {1: "I", 2: "II", 3: "III", 4: "IV"}
period_label = f"{quarter_roman[quarter]}Q{year}"

# Compare against reference values
reference_results = validate_against_reference(sheet1_data, period_label)

if reference_results:
    print(f"\n=== Validation vs Reference ({period_label}) ===")
    print(format_validation_report(reference_results))
else:
    print(f"No reference data available for {period_label}")
```

### 4. Map to Standard Structure

```python
# Get the standard 27-row structure from config
structure = get_standard_structure("sheet1")
print(f"Standard structure has {len(structure)} rows")

# Map extracted data to structure
rows = map_to_structure(sheet1_data, "sheet1")

# Display formatted rows
for row in rows:
    if row.get("valor") is not None:
        print(f"Row {row['row']}: {row['concepto']} = {row['valor']:,}")
```

### 5. Save Formatted Data

```python
from puco_eeff.writer.sheet_writer import save_sheet_data

# Save with standardized naming
save_sheet_data(
    sheet_name="sheet1",
    data=sheet1_data,
    period=period
)

print(f"✓ Sheet 1 formatted and saved")
```

## Output

- Formatted JSON: `data/processed/sheet1_YYYY_QN.json`
- Validation report showing pass/fail for each field

## Sheet 1 Structure (27 rows)

| Row | Field | Label |
|-----|-------|-------|
| 1 | ingresos_ordinarios | Ingresos de actividades ordinarias M USD |
| 2 | | (blank) |
| 3 | header | Costo de Venta |
| 4-14 | cv_* | 11 Costo de Venta line items |
| 15 | total_costo_venta | Total Costo de Venta |
| 16-18 | | (blank) |
| 19 | header | Gasto Adm. y Ventas |
| 20-25 | ga_* | 6 Gasto Admin line items |
| 26 | | (blank) |
| 27 | total_gasto_admin | Totales |

## Validation Checks

1. **Internal consistency**: Sum of cv_* items == total_costo_venta
2. **Internal consistency**: Sum of ga_* items == total_gasto_admin
3. **Reference validation**: Compare against known-good IIQ2024 values

## Next Steps

1. Review validation results
2. Proceed to `05_locate_sheet2.md` for the next sheet
3. Or proceed to `NN_combine_workbook.md` if all sheets ready
