# NN - Combine Sheets into Final Workbook

## Objective

Combine all extracted and formatted sheets into the final Excel workbook: `EEFF_YYYY_QN.xlsx`

## Prerequisites

All sheets should be processed and saved in `data/processed/`:
- `YYYY_QN_estado_situacion_financiera.json`
- `YYYY_QN_estado_resultados.json`
- (Additional sheets as applicable)

## Steps

### 1. Check Available Sheets

```python
from puco_eeff.writer.sheet_writer import list_available_sheets
from puco_eeff.config import get_period_paths

year, quarter = 2024, 3
period = f"{year}_Q{quarter}"

available = list_available_sheets(period)
print(f"Available sheets for {period}:")
for sheet in available:
    print(f"  - {sheet}")
```

### 2. Define Sheet Order

```python
# Define the order sheets should appear in the workbook
sheet_order = [
    "estado_situacion_financiera",  # Balance General
    "estado_resultados",            # Income Statement
    "estado_flujos_efectivo",       # Cash Flow (if available)
    "notas",                        # Notes (if available)
]

# Filter to only available sheets
sheets_to_include = [s for s in sheet_order if s in available]
# Add any additional sheets not in predefined order
for s in available:
    if s not in sheets_to_include:
        sheets_to_include.append(s)

print(f"Sheets to include: {sheets_to_include}")
```

### 3. Combine into Workbook

```python
from puco_eeff.writer.workbook_combiner import combine_sheets_to_workbook

output_path = combine_sheets_to_workbook(
    period=period,
    sheet_order=sheets_to_include
)

print(f"✓ Workbook created: {output_path}")
```

### 4. Verify Output

```python
import pandas as pd

# Read back and verify
excel_file = pd.ExcelFile(output_path)
print(f"\nWorkbook contains {len(excel_file.sheet_names)} sheets:")
for sheet_name in excel_file.sheet_names:
    df = pd.read_excel(output_path, sheet_name=sheet_name)
    print(f"  - {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
```

### 5. Final Checklist

```python
from pathlib import Path

# Check all expected outputs exist
paths = get_period_paths(year, quarter)
expected_files = [
    paths["output"] / f"EEFF_{year}_Q{quarter}.xlsx",
    paths["audit"] / "source_mapping.json",
]

print("\nOutput verification:")
for filepath in expected_files:
    exists = filepath.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {filepath}")
```

## Output

- Final workbook: `data/output/EEFF_YYYY_QN.xlsx`

## Sheet Name Formatting

| Internal Name | Display Name |
|--------------|--------------|
| estado_situacion_financiera | Estado Situación Financiera |
| estado_resultados | Estado Resultados |
| estado_flujos_efectivo | Estado Flujos Efectivo |
| notas | Notas |

## Re-running for Specific Sheets

To re-run extraction for a specific sheet and regenerate the workbook:

```python
# 1. Re-run the extract step for that sheet (e.g., 04_extract_sheet1.md)
# 2. Re-run the format step (e.g., 05_format_sheet1.md)
# 3. Run this combine step again

# The workbook will be regenerated with the updated sheet
```

## Archive Previous Versions

```python
import shutil
from datetime import datetime

# If you want to keep previous versions
workbook_path = paths["output"] / f"EEFF_{year}_Q{quarter}.xlsx"
if workbook_path.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"EEFF_{year}_Q{quarter}_{timestamp}.xlsx"
    archive_path = paths["output"] / "archive" / archive_name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(workbook_path, archive_path)
    print(f"Previous version archived: {archive_path}")
```
