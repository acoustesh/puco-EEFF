---
name: documentCodebaseFromLogic
description: Rewrite docstrings/comments/docs across the entire Python codebase based on actual behavior. Function logic is source of truth. Enforce PEP 8 comment rules + NumPy docstring style. Use Context7 library docs heavily. Ensure tests/test_comment_density.py and tests/test_docstring_format.py pass.
argument-hint: Repo path + any excluded directories (if any) + doc output locations + whether to keep changelog
agent: Plan
---

You are a documentation agent. Objective: update documentation so it matches what the code actually does. Code logic takes precedence over all existing docstrings/comments/markdown text.

Scope is mandatory: ALL Python files in the codebase + ALL markdown docs (including README.md). You may update any documentation baselines/line references used by tests if needed.

## Hard constraints
1. Source of truth is function/class/module logic. If docs conflict, docs must change.
2. Update docstrings in every module, class, function, and method.
3. Update all markdown documentation files (*.md), including README.md with the module/package structure.
4. Follow:
   - PEP 8 comment guidance: https://peps.python.org/pep-0008/#comments
   - NumPy docstring style (sections + formatting)
5. Make tests pass with 0 errors:
   - tests/test_comment_density.py
   - tests/test_docstring_format.py
6. No commented-out code anywhere; delete it.

## Mandatory external-library understanding (Context7)
To document behavior accurately when code calls external libraries/frameworks, you MUST use Context7 extensively:

1. For each non-trivial third-party import or API call path, do:
   - call `resolve-library-id` for the library/package
   - call `get-library-docs` for the specific functions/classes used
2. Use those docs to correctly document:
   - parameter meaning/units
   - return types/structures
   - raised exceptions/error semantics
   - side effects, performance characteristics, and constraints
3. Prefer primary library docs over assumptions; if uncertain, document uncertainty explicitly and narrowly (1 sentence).

## Workflow (mandatory order)
### Step 0 — Build an inventory
1. Enumerate:
   - all Python modules/packages
   - all symbols: modules, classes, functions, methods
2. For each symbol, capture:
   - signature (incl. defaults/types)
   - side effects (fs/db/http/logging)
   - exceptions raised (explicit + common propagated)
   - key invariants/preconditions
3. Enumerate external dependencies per module (imports), mark “needs Context7 docs”.

### Step 1 — Read logic first, then write docs
For every symbol:
1. Read the implementation and derive behavior:
   - inputs/outputs (types, units, conventions)
   - edge cases and failure modes
   - state changes / side effects
   - complexity notes only if non-trivial (big-O, memory)
2. For external calls, consult Context7 docs (resolve + get docs) before writing.
3. Only after understanding logic, rewrite docstrings/comments.

## Comment policy (PEP 8 + density requirement)
### Allowed/required comment types (keep/add)
- Rationale / “why” (trade-offs, constraints, non-obvious decisions)
- Invariants / preconditions / pitfalls
- Short algorithm overview before dense blocks (1–5 lines)
- Traceability (ticket/spec/benchmark) using:
  - `# TODO: <ID or link> - <short rationale>`

### Forbidden (remove/refactor)
- redundant “W.E.T.” obvious comments (e.g., `# increment x`)
- comments that explain unclear code instead of improving names/structure
- comments longer than the code they support (rarely allowed)
- stale/contradicting comments (must be corrected immediately)
- commented-out code (must be deleted)

### Density rule (mandatory)
- Add/update comments while revising logic so that:
  - a comment appears every 5–20 lines on average
  - target: 6%–17% of lines commented (average across the codebase)
- Comments must be intent-focused; never narrate syntax.

## Docstring policy (NumPy style, logic-first)
### Module docstrings
- Mandatory: purpose, key modules, main entrypoints, and invariants.
- If module has side effects/config: document environment variables/config keys.

### Function/method docstrings
- Mandatory NumPy sections, as applicable:
  - One-line summary (always)
  - Extended Summary (mandatory if function > 10 lines)
  - Parameters / Returns / Raises (include real exceptions)
  - Notes (algorithm/invariants/complexity/stability)
  - Examples (if useful or if tests require)
  - See Also (when clear relationships exist)

### Class docstrings
- Purpose + key invariants
- Important attributes (esp. constructor params)
- Thread-safety / mutability notes if relevant

### Formatting rules
- Must comply with tests/test_docstring_format.py exactly.
- Keep summaries imperative and specific; avoid vague wording.
- Do not document parameters that do not exist.
- Do not claim behavior not guaranteed by code.

## Markdown documentation updates (*.md)
1. Update all existing docs to reflect:
   - current module layout
   - new APIs/entrypoints
   - configuration and run instructions
2. README.md must include:
   - high-level architecture diagram in text (tree or bullet structure)
   - “Where to put sheet-specific logic”: puco_eeff/sheets
   - main flows (scrape → parse → extract → outputs)
   - testing commands and what each relevant test validates

## Baselines / tests integration
1. Run or reason about failures from:
   - tests/test_comment_density.py
   - tests/test_docstring_format.py
2. If tests rely on baseline line references for documentation checks:
   - update those baselines/line lists as needed, but ONLY to reflect the new, correct documentation.
3. Iterate until both tests pass with 0 errors.

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
