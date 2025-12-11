# 07 - Format Sheet 2 for Excel

## Objective

Format Cuadro Resumen KPIs data for the final Excel workbook.

## Output Structure

Sheet2 has a 15-row structure defined in `config/sheet2/fields.json`:

| Row | Field | Label | Unit |
|-----|-------|-------|------|
| 1 | `cobre_concentrados` | Cobre en concentrados M USD | MUS$ |
| 2 | `cobre_catodos` | Cobre en Cátodos | MUS$ |
| 3 | `oro_subproducto` | Oro subproducto | MUS$ |
| 4 | `plata_subproducto` | Plata subproducto | MUS$ |
| 5 | `total_ingresos` | **Total** | MUS$ |
| 6 | - | (blank) | - |
| 7 | `ebitda` | Ebitda del periodo | MUS$ |
| 8 | `libras_vendidas` | Libras de cobre vendido | MM lbs |
| 9 | `cobre_fino` | Cobre Fino Obtenido | MM lbs |
| 10 | `precio_efectivo` | Precio efectivo de venta | US$/lb |
| 11 | `cash_cost` | Cash Cost | US$/lb |
| 12 | `costo_unitario_total` | Costo unitario Total | US$/lb |
| 13 | `non_cash_cost` | Non Cash cost | US$/lb |
| 14 | `toneladas_procesadas` | Total Toneladas Procesadas | miles |
| 15 | `oro_onzas` | Oro en Onzas | miles oz |

## Convert to Row List

```python
from puco_eeff.sheets.sheet2 import extract_sheet2

data, _ = extract_sheet2(2024, 2)
rows = data.to_row_list()

for row_num, label, value in rows:
    if value is not None:
        print(f"Row {row_num:2d}: {label:40s} = {value}")
```

## Validation Rules

### Sum Validation

```
total_ingresos = cobre_concentrados + cobre_catodos + oro_subproducto + plata_subproducto
```

Tolerance: ±1 (for rounding)

### Cross-Sheet Validation

Sheet2 `total_ingresos` should match Sheet1 `ingresos_ordinarios` (within tolerance).

## Excel Formatting

The generic `sheet_writer.py` handles Excel output:

```python
from puco_eeff.writer.sheet_writer import save_sheet_data

# Save as JSON (used by workbook combiner)
output_path = save_sheet_data(
    sheet_name="sheet2",
    data=data.to_dict(),
    year=2024,
    quarter=2
)
```

## Combine with Workbook

After extracting all quarters, combine into final workbook:

```python
from puco_eeff.writer.workbook_combiner import combine_quarterly_data

# Combines sheet1 and sheet2 data for all quarters
combine_quarterly_data(
    year=2024,
    quarters=[1, 2, 3, 4],
    sheets=["sheet1", "sheet2"]
)
```

## Reference Values

Verified values are stored in `config/sheet2/reference_data.json` for periods:
- 2024_Q1, 2024_Q2, 2024_Q3 (verified)
- 2024_Q4 (incomplete data)
- 2025_Q1, 2025_Q2, 2025_Q3 (verified)

## Next Steps

Continue with additional sheets or proceed to `NN_combine_workbook.md` to generate the final Excel file.
