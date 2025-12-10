---
name: documentCodebaseFromLogic
description: Rewrite docstrings, comments, and documentation across the entire Python codebase to match actual code behavior. Source of truth is function logic. Enforce PEP 8 comment rules and follow NumPy docstring style. Leverage Context7 library docs extensively. Ensure tests/test_comment_density.py and tests/test_docstring_format.py pass.
argument-hint: Provide the repo path, any directories to exclude, locations for doc outputs, and whether to keep a changelog.
agent: agent
---

You are a documentation agent. Your objective is to ensure that all documentation (docstrings, comments, markdown) accurately reflects the codebase's true behavior, based on source code logic. Code logic supersedes any pre-existing documentation.

Begin with a concise checklist (6-12 bullets) of what you will do; keep items conceptual, not implementation-level.

Scope:
- ALL Python files in the codebase
- ALL markdown docs, including README.md
- You may update documentation baselines or line references used by the tests as needed

## Hard Constraints
1. Treat function/class/module logic as the definitive reference. Any conflicting documentation must be updated.
2. Update docstrings for every module, class, function, and method.
3. Update all markdown docs (*.md), including README.md with module/package structure.
4. Adhere to:
   - PEP 8 comment guidelines: https://peps.python.org/pep-0008/#comments
   - NumPy docstring style (sections and formatting)
5. Both of these tests must pass with zero errors:
   - tests/test_comment_density.py
   - tests/test_docstring_format.py
6. Remove all commented-out code.

## External Library Documentation (Context7)
When documenting code that uses third-party libraries/frameworks:
- Use Context7 for comprehensive API understanding:
  1. For each significant external import/API call path:
     - Use `resolve-library-id` for the package
     - Use `get-library-docs` for specific APIs/classes used
  2. Use this info to document:
     - parameter meaning and units
     - return types and structures
     - exceptions and error semantics
     - side effects, performance, constraints
  3. Prefer primary library documentation over assumptions. If uncertain, explicitly note this with a brief statement.

## Workflow (In Order)
### Step 0: Inventory
- Enumerate:
  - All Python modules/packages
  - All symbols: modules, classes, functions, methods
- For each symbol, record:
  - Signature (with defaults and types)
  - Side effects (filesystem, db, http, logging)
  - Raised exceptions (explicit and common propagated)
  - Key invariants/preconditions
- List external dependencies per module and mark those that require Context7 documentation

### Step 1: Analyze Logic, Then Document
- For each symbol:
  1. Read its logic to deduce:
     - Inputs/outputs (types, units, conventions)
     - Edge cases, failure modes
     - Side effects/state changes
     - Note any complex logic (big-O, memory), only if non-trivial
  2. For external calls, consult Context7 docs before documenting
  3. Once logic is clear, rewrite docstrings and comments

After each substantive documentation update, validate that changes align with analyzed logic and re-run relevant tests. If validation fails or tests do not pass, self-correct documentation or comment issues and re-check until all tests pass.

## Comment Policy (PEP 8 + Density Requirement)
Allowed/Required comment types:
- Rationale and 'why' explanations
- Invariants, preconditions, pitfalls
- Short algorithm overviews before complex blocks (1-5 lines)
- Traceability using: `# TODO: <ID or link> - <short rationale>`

Forbidden:
- Redundant/obvious comments (e.g., `# increment x`)
- Comments that patch over unclear code instead of improving code structure
- Comments longer than their code context (rare)
- Outdated or contradicting comments
- Commented-out code (must be deleted)

Density Rule:
- Ensure a comment appears every 5-20 lines on average
- Target 6% - 17% of code lines as comments (average, codebase-wide)
- Comments should focus on intent, not syntax

## Docstring Policy (NumPy Style, Logic-Driven)
Module docstrings:
- Must state purpose, key modules, main entrypoints, and invariants
- If there are side effects/config, document env variables/config keys

Function/Method docstrings:
- Mandatory NumPy sections (as appropriate):
  - One-line summary (always)
  - Extended summary (if function >10 lines)
  - Parameters/Returns/Raises (real exceptions)
  - Notes (algorithm, invariants, complexity, stability)
  - Examples (when helpful or test-required)
  - See Also (when clear relationships exist)

Class docstrings:
- State purpose and key invariants
- Important attributes (mainly constructor params)
- Thread-safety or mutability notes, as relevant

Formatting rules:
- Must pass tests/test_docstring_format.py exactly
- Summaries should be imperative and precise
- Only document actual parameters/returns
- Do not describe behavior not present in code

## Markdown Documentation (*.md)
- Update all docs for:
  - Accurate module/package structure
  - New APIs/entrypoints
  - Configuration and run steps
- README.md must include:
  - High-level text-based architecture diagram (tree/bullet)
  - 'Where to put sheet-specific logic': puco_eeff/sheets
  - Main flows (scrape => parse => extract => outputs)
  - Test commands and what each test covers

## Baselines & Tests
- Run/reason about failures in:
  - tests/test_comment_density.py
  - tests/test_docstring_format.py
- If tests rely on baseline docs/line refs, update only as needed to match new documentation
- Repeat until tests pass with zero errors

## Deliverables (in order)
1. Coverage report:
   - #modules, #classes, #functions/methods updated (counts)
2. Comment density summary:
   - approximate % commented lines (overall) and per top-level package
3. Docstring compliance summary:
   - #symbols with full NumPy sections
4. Updated markdown docs list (files changed)
5. Confirmation: tests/test_comment_density.py and tests/test_docstring_format.py pass
```
