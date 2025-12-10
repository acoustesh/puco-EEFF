---
name: splitModuleByPurpose
description: Split one Python file into 2+ modules by shared purpose, using tests/test_code_complexity.py to identify and rank candidates. No wrappers; update all internal callers. End state must pass the complexity test.
argument-hint: Target file path (or package root) + “auto-pick worst offenders” + any hard boundaries (e.g., puco_eeff/sheets)
agent: agent
---

You are a refactor agent. Goal: restructure the codebase by splitting modules (and refactoring symbols) so that:
- /home/alvaro/aa-ml/puco-EEFF/tests/test_code_complexity.py passes.

Use that test file as the authoritative source for:
- which metrics are measured,
- thresholds,
- symbol discovery scope,
- failure reporting.

## Hard constraints
1. Candidate selection MUST be driven by the complexity test:
   - parse its thresholds and scope
   - use its reported failing symbols to prioritize work
2. End state MUST pass /home/alvaro/aa-ml/puco-EEFF/tests/test_code_complexity.py.
3. No wrappers, no compatibility shims, no re-export layers. Update all internal callers and tests to new import paths.
4. Do not “cheat” by relaxing the test (no threshold reductions, no adding exclusions) unless the test explicitly requires an allowlist update for moved paths; prefer fixing code.
5. If any module contains Sheet1/Sheet2 specifics, keep them only in puco_eeff/sheets (no leakage).

## Step 0 — the tests test_maintainability_index_all_directories,  tests/test_code_complexity.py::TestMaintainabilityIndex::test_maintainability_index_extractor
2. Write a short “rule sheet” (for yourself) used to drive decisions.

## Step 1 — Build a ranked refactor queue from the test
1. Run or replicate the test’s discovery to list all violations.
2. Rank candidates by “excess over threshold” (largest first).
3. Prefer candidates that:
   - combine multiple responsibilities in one file
   - have long if/elif ladders or deep nesting
   - duplicate orchestration + parsing + formatting in one unit

## Step 2 — Purpose-driven split plan (for each top offender)
1. Inventory symbols in the offender file and tag each with ONE purpose:
   - types, core, parse, format, validate, io, orchestrate, cli, util(only if reused)
2. Design new layout (package dir preferred):
   - <module_name>/__init__.py
   - types.py (no sibling imports)
   - core.py (no io imports)
   - parse.py / format.py / validate.py / io.py / orchestrate.py as needed
3. Move symbols in safe order:
   1) types
   2) shared pure helpers (only if truly shared)
   3) core logic
   4) parse/format/validate
   5) io/orchestrate/cli

## Step 3 — Reduce complexity inside moved symbols (not cosmetic)
Apply transformations that actually reduce complexity metrics:
- replace long if/elif with dispatch tables / dict lookups
- split multi-responsibility functions (normalize → transform → emit)
- early returns to reduce nesting
- extract pure helpers for repeated decision blocks
- prefer data-driven configs over branching

## Step 4 — Update all call sites (no wrappers)
1. Update every internal import and reference to new module paths.
2. Update tests that import the old module paths.
3. Ensure old module path no longer exists or is no longer referenced.

## Step 5 — Verification loop (must end green)
1. Run: pytest -q /home/alvaro/aa-ml/puco-EEFF/tests/test_code_complexity.py
2. If failures remain:
   - read which symbols still violate thresholds
   - repeat Steps 2–4 focused on the remaining worst offenders
3. Stop only when the test passes.

## Deliverables
1. Ranked list of initial complexity offenders from the test output.
2. New module tree for each split (paths).
3. Mapping: old symbol → new module (counts).
4. Confirmation: all /home/alvaro/aa-ml/puco-EEFF/tests/ pass.
