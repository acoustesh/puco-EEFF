# Troubleshooting Guide

## Common Issues and Solutions

### Browser / Playwright Issues

#### Browser not installed
```
Error: Executable doesn't exist at ...
```

**Solution:**
```bash
./setup_browser.sh
# or manually:
poetry run playwright install chromium
poetry run playwright install-deps chromium
```

#### Page doesn't load (timeout)
```
Error: Timeout 30000ms exceeded
```

**Solutions:**
1. Increase timeout:
   ```python
   page.goto(url, timeout=60000)
   ```
2. Check network connectivity
3. Try with `headless=False` to see what's happening
4. The site might be blocking automated browsers - check if a CAPTCHA appears

#### Element not found
```
Error: Element not found
```

**Solutions:**
1. Wait for element:
   ```python
   page.wait_for_selector("selector", timeout=10000)
   ```
2. The page structure might have changed - inspect manually
3. Use `headless=False` to debug

---

### PDF Extraction Issues

#### No tables extracted
**Possible causes:**
- PDF contains images instead of text tables
- Table structure is non-standard

**Solutions:**
1. Use OCR:
   ```python
   from puco_eeff.extractor.ocr_fallback import ocr_with_fallback
   result = ocr_with_fallback(pdf_path=pdf_path)
   ```
2. Adjust table settings:
   ```python
   tables = extract_tables_from_pdf(
       pdf_path,
       table_settings={
           "vertical_strategy": "text",
           "horizontal_strategy": "text"
       }
   )
   ```

#### Garbled text extraction
**Cause:** PDF uses non-standard fonts or encoding

**Solutions:**
1. Use OCR instead of direct text extraction
2. Check PDF properties for font embedding issues

---

### XBRL Parsing Issues

#### Namespace errors
```
Error: XPath evaluation failed
```

**Solutions:**
1. Check actual namespaces in the document:
   ```python
   from lxml import etree
   tree = etree.parse(str(xml_path))
   print(tree.getroot().nsmap)
   ```
2. Update namespace mappings in `xbrl_parser.py`

#### Missing facts
**Cause:** XBRL taxonomy might differ from expected

**Solution:** Explore available facts:
```python
data = parse_xbrl_file(xml_path)
print("Available facts:")
for fact in sorted(set(f["name"] for f in data["facts"])):
    print(f"  {fact}")
```

---

### OCR Issues

#### API key errors
```
Error: Invalid API key
```

**Solutions:**
1. Check `.env` file has correct keys
2. Verify keys are not expired
3. Check API quotas/limits

#### Poor OCR quality
**Solutions:**
1. Try different fallback providers (Anthropic might work better than OpenAI or vice versa)
2. Improve the prompt with more specific instructions
3. Process single pages instead of full PDF

#### Rate limiting
```
Error: Rate limit exceeded
```

**Solutions:**
1. The exponential retry should handle this, but you can increase delays:
   ```python
   # In config.json
   "retry": {
       "base_delay_seconds": 2,
       "max_delay_seconds": 60
   }
   ```

---

### Data Issues

#### Mismatched totals
```
Balance check failed: Assets != Liabilities + Equity
```

**Possible causes:**
1. Rounding differences (usually <1, acceptable)
2. Missing line items
3. Wrong period data

**Solutions:**
1. Check source documents manually
2. Verify all line items are captured
3. Check that values are from the same period

#### Empty or None values
**Solutions:**
1. Check if field exists in source:
   ```python
   # For XBRL
   values = extract_by_xpath(xml_path, "//ifrs-full:FieldName")
   print(f"Found: {values}")
   ```
2. Check PDF section mapping
3. May need to use OCR for that specific field

---

### File/Path Issues

#### File not found
```
FileNotFoundError: ...
```

**Solutions:**
1. Check the file actually exists:
   ```python
   from pathlib import Path
   print(Path(filepath).exists())
   ```
2. Verify period/year/quarter are correct
3. Check if download step completed successfully

#### Permission denied
**Solutions:**
1. Check file permissions: `ls -la`
2. Ensure you own the directory
3. Close any programs that might have the file open

---

### Environment Issues

#### Import errors
```
ModuleNotFoundError: No module named 'xxx'
```

**Solutions:**
1. Install missing package:
   ```bash
   poetry install
   ```
2. Activate virtual environment:
   ```bash
   poetry shell
   ```

#### Wrong Python version
**Solution:**
```bash
python --version  # Should be 3.12+
poetry env use python3.12
poetry install
```

---

## Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Updating Config After Successful Recovery

**IMPORTANT**: After successfully recovering from an extraction failure, always update the configuration files to prevent the same issue in the future.

### What to Update

#### 1. extraction_specs.json - Period-Specific Entry

Add or update the period entry with what worked:

```json
{
  "2024_Q3": {
    "_comment": "Describe what was different about this period",
    "verified": true,
    "verified_date": "2024-12-06",
    "deviations": {
      "sections": {
        "nota_21": {
          "search_patterns": ["pattern that worked"],
          "_deviation_reason": "Why this was needed"
        }
      }
    },
    "page_numbers": {
      "nota_21": 74,
      "nota_22": 74
    },
    "recovery_notes": "Brief description of recovery method used"
  }
}
```

#### 2. extraction_specs.json - Default Updates

If the same deviation appears in multiple periods, add to defaults:

```json
{
  "default": {
    "sections": {
      "nota_21": {
        "search_patterns": [
          "existing pattern",
          "new pattern that worked"  // Add here
        ],
        "field_mappings": {
          "cv_energia": {
            "pdf_labels": [
              "Energía eléctrica",
              "Energia electrica"  // Add variant if encountered
            ]
          }
        }
      }
    }
  }
}
```

#### 3. reference_data.json - Verified Values

After confirming extracted values are correct:

```json
{
  "2024_Q3": {
    "verified": true,
    "verified_date": "2024-12-06",
    "values": {
      "ingresos_ordinarios": 231472,
      "cv_gastos_personal": -30294,
      // ... all 20 values
    },
    "internal_checks": {
      "nota_21_sum_matches_total": true,
      "nota_22_sum_matches_total": true
    }
  }
}
```

### Decision Guide: When to Update Defaults

| Scenario | Action |
|----------|--------|
| New label variant (e.g., accent difference) | Add to default's pdf_labels |
| New search pattern works for 2+ periods | Add to default's search_patterns |
| Page numbers vary significantly | Keep in period-specific, not default |
| OCR consistently needed for a section | Note in default's extraction_method |
| Same unique_item missing | Add to default's unique_items |

### Validation After Updates

```bash
# Always run after config changes
poetry run pytest tests/test_config_integrity.py -v
```

This catches:
- JSON syntax errors
- Missing required fields
- Field name mismatches
- Invalid reference values

---

## Getting Help

1. Check the logs in `logs/YYYY-MM-DD_run.log`
2. Review audit files in `audit/YYYY_QN/`
3. Run with `headless=False` to see browser actions
4. Test individual components in a notebook
