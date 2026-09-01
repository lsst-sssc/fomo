---
phase: quick-260722-hpw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/tests/test_import_campaign_csv.py
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
autonomous: true
requirements: [QUICK-260722-hpw]
must_haves:
  truths:
    - "A campaign CSV whose real header is preceded by leading free-text/blank rows imports successfully (no false CommandError)."
    - "A CSV with no header row containing all required columns within the scan cap still fails fast with a clear CommandError."
    - "Behavior is unchanged for the common case where the header is already on row 1 (all existing tests keep passing)."
  artifacts:
    - path: "solsys_code/management/commands/import_campaign_csv.py"
      provides: "Leading-row skip that locates the real header before building the DictReader"
      contains: "_MAX_HEADER_SCAN"
    - path: "solsys_code/tests/test_import_campaign_csv.py"
      provides: "Regression tests for leading-comment header discovery and no-header fast-fail"
    - path: "docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb"
      provides: "Demonstration cell exercising the leading-comment-row skip against a synthetic inline CSV"
  key_links:
    - from: "import_campaign_csv.py handle()"
      to: "csv.DictReader"
      via: "header row located via csv.reader scan, DictReader built from lines[header_idx:]"
      pattern: "csv\\.DictReader"
---

<objective>
Fix `import_campaign_csv`'s `handle()` so it skips leading comment/blank rows before the real
CSV header. The real 3I/ATLAS Google Sheet export starts with a free-text attribution row
(row 1) and an entirely blank row (row 2), with the true 14-column header on row 3.
`csv.DictReader` currently treats row 1 as the header, so `reader.fieldnames` never contains
`_REQUIRED_HEADERS` and the command raises a false "missing required column(s)" `CommandError`
on an otherwise well-formed file.

Purpose: Let the command consume the real-world sheet export unchanged, while preserving
fast-fail behavior (and the existing `CommandError` message shape) when no valid header exists.
Output: Header-discovery scan in the command, regression tests, and a notebook demonstration cell.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@solsys_code/management/commands/import_campaign_csv.py
@solsys_code/tests/test_import_campaign_csv.py
@docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Scan for the real header row before building the DictReader</name>
  <files>solsys_code/management/commands/import_campaign_csv.py</files>
  <behavior>
    - A CSV whose header is on row 1 (no leading rows) imports exactly as before — existing
      tests in test_import_campaign_csv.py keep passing, and the first data row is still logged
      as "Row 2".
    - A CSV with a leading free-text comment row plus a blank row before the real header
      locates the header and imports every data row.
    - A CSV with no row containing all of _REQUIRED_HEADERS within the first _MAX_HEADER_SCAN
      rows raises CommandError whose message includes the required column names.
    - A CSV whose valid header only appears AFTER the scan cap still fails fast with the same
      CommandError (does not scan the whole file).
    - An OSError opening the file still surfaces as CommandError('Cannot open campaign CSV ...').
  </behavior>
  <action>
    Add a module-level constant `_MAX_HEADER_SCAN = 10` near `_REQUIRED_HEADERS`, with a short
    comment explaining it caps the leading-row scan so a genuinely malformed/wrong file fails
    fast rather than scanning the whole file.

    In `handle()`, replace the current open-and-DictReader block. Keep the `try/except OSError ->
    CommandError('Cannot open campaign CSV ...')` wrapper. Inside the `with open(...)` block, read
    the file into a list of raw text lines (`f.readlines()` — the `newline=''` and `encoding='utf-8'`
    on the open call are preserved). Then find the header index by iterating
    `csv.reader(lines[:_MAX_HEADER_SCAN])` with `enumerate`, selecting the first parsed row where
    every header in `_REQUIRED_HEADERS` is present (`all(h in parsed_row for h in _REQUIRED_HEADERS)`).

    If no header index is found, raise `CommandError` whose message names the file, states that no
    header row containing all required column(s) `{_REQUIRED_HEADERS!r}` was found within the first
    `{_MAX_HEADER_SCAN}` rows. The message MUST include the `_REQUIRED_HEADERS` repr so the string
    contains 'Telescope / Instrument' (the existing test_missing_required_header_raises_command_error
    asserts that substring and now flows through this path).

    When found, build the DictReader from the sublist starting at the header line:
    `reader = csv.DictReader(lines[header_idx:])` (csv.DictReader accepts any iterable of strings, so
    this preserves DictReader's restval/restkey semantics for short/long data rows and quoted
    multi-line fields). Set `rows = list(reader)`. Drop the now-redundant `reader.fieldnames`
    missing-header check — the header scan already guarantees the required columns are present;
    do not keep the old `missing_headers` branch, since a matched header always contains them.

    Update the data-row enumeration so line numbers stay truthful: `enumerate(rows, start=header_idx + 2)`
    (header is at file line `header_idx + 1`, first data row at `header_idx + 2`). For the common
    no-leading-rows case `header_idx == 0`, this yields `start=2`, preserving the existing "Row 2"
    logging that current tests assert. Update the inline `# header is row 1` comment accordingly.

    Keep single quotes and 120-col formatting; add a brief comment on the scan explaining WHY
    (real sheet export prepends a free-text attribution row and a blank row before the header).
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_import_campaign_csv 2>&1 | tail -20</automated>
  </verify>
  <done>All existing import_campaign_csv tests pass; the command scans up to _MAX_HEADER_SCAN
  leading rows for the header and fails fast with a CommandError naming the required columns when
  none is found.</done>
</task>

<task type="auto">
  <name>Task 2: Add regression tests for header discovery and no-header fast-fail</name>
  <files>solsys_code/tests/test_import_campaign_csv.py</files>
  <action>
    Add tests to the `TestImportCampaignCsv` class. Do NOT reuse `_WriteCsvMixin._write_csv`
    (it always writes the header on row 1) — build the raw CSV files directly with
    `tempfile.TemporaryDirectory()` and `pathlib.Path`, matching the style already used in
    `test_missing_required_header_raises_command_error`. Use small inline synthetic content only;
    never the user's real sheet. No Target fixture is required here; if you add one for any reason,
    use `tom_targets.tests.factories.NonSiderealTargetFactory` (already imported), never
    SiderealTargetFactory (CLAUDE.md).

    Add `test_skips_leading_comment_and_blank_rows_before_header`: write a file whose first line is
    a single free-text comment cell followed by trailing commas (mirroring the real export's row 1,
    e.g. `"This spreadsheet is for coordination...",,,,,,,,,,,,,`), then an entirely blank line
    (`,,,,,,,,,,,,,`), then the real header line (join `_HEADERS` with commas), then one data line
    with a resolvable `Telescope / Instrument`, `Obs. Date`, and `UT Time Range` (seed an
    Observatory for the Site Code, or leave Site Code blank and accept site_needs_review). Prefer
    `csv.writer` for the header and data rows so quoting is correct; write the comment and blank
    lines as raw text. Call the command and assert exactly one `CampaignRun` is created and stdout
    reports `created: 1`.

    Add `test_no_header_row_within_scan_cap_raises_command_error`: write a file consisting only of
    several comment/blank rows and rows that do NOT contain all required columns (no real header
    anywhere). Assert `call_command` raises `CommandError`, that the exception message contains
    'Telescope / Instrument', and that `CampaignRun.objects.count() == 0`.

    Add `test_header_beyond_scan_cap_fails_fast`: write a file with more than `_MAX_HEADER_SCAN`
    leading comment/blank rows before the valid header. Assert `call_command` raises `CommandError`
    (the header is past the cap, so it is not found) and no `CampaignRun` rows are created. Import
    `_MAX_HEADER_SCAN` from the command module to compute the number of leading rows rather than
    hardcoding it.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_import_campaign_csv 2>&1 | tail -20</automated>
  </verify>
  <done>Three new tests pass alongside the existing suite: leading comment/blank rows are skipped
  and the row imports; a file with no valid header and a file whose header is beyond the scan cap
  both raise CommandError with zero CampaignRun rows created.</done>
</task>

<task type="auto">
  <name>Task 3: Add a notebook demonstration cell for the leading-comment-row skip</name>
  <files>docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb</files>
  <action>
    The notebook's stated premise is fidelity to "the same 14-column shape as the real 3I/ATLAS
    coordination sheet", but the real export actually prepends a free-text attribution row and a
    blank row before the header — the exact case this fix addresses. Add a short markdown+code cell
    pair (place them after the existing fixture/import cells, e.g. after the range/TBD demonstration
    cells, before the approval-lifecycle section) demonstrating that a CSV shaped like the real
    export imports cleanly.

    Do NOT modify the shared `fixtures/campaign_sample.csv` or any existing cell — that would
    perturb the notebook's established created/unchanged counts. Instead, in the new code cell build
    a self-contained inline CSV with `tempfile` (a leading comment row, a blank row, then the
    14-column header and one or two synthetic PII-free data rows using placeholder names and
    `@example.com` emails), import it under a distinct campaign name (e.g. `'3I/ATLAS leading-comment demo'`),
    and print the summary plus the resulting CampaignRun count to show it succeeded where the old
    behavior would have raised CommandError. The markdown cell should explain that the real sheet
    export leads with an attribution row and a blank row, and that the command now scans past them
    to find the header.

    Regenerate executed output and commit it (pre_executed notebooks are committed WITH output, per
    the pre-commit convention). Run nbconvert from `docs/notebooks/pre_executed/` (the notebook's
    Django-setup cell assumes that CWD via `Path.cwd().resolve().parents[2]`).
  </action>
  <verify>
    <automated>cd docs/notebooks/pre_executed && jupyter nbconvert --to notebook --execute --inplace import_campaign_csv_demo.ipynb 2>&1 | tail -5</automated>
  </verify>
  <done>The notebook has a new markdown+code cell pair demonstrating the leading-comment-row skip
  against an inline synthetic CSV, executes end-to-end with committed output, and does not modify
  the shared fixture or existing cells.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CSV file → command | The campaign CSV is an untrusted external export (a Google Sheet download); its leading rows and header shape are attacker/typo-controlled input crossing into the importer. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-hpw-01 | Denial of Service | `handle()` header scan over CSV lines | mitigate | Cap the header search at `_MAX_HEADER_SCAN` (10) rows so a malformed/wrong file with no header fails fast instead of scanning the whole file. |
| T-hpw-02 | Tampering | Renamed/missing required column in header | mitigate | If no row within the cap contains all `_REQUIRED_HEADERS`, raise `CommandError` naming the required columns — preserves the existing fast-fail contract; no rows are written. |
| T-hpw-SC | Tampering | npm/pip/cargo installs | accept | No new dependencies added; fix uses only stdlib `csv`. No install step. |
</threat_model>

<verification>
- `./manage.py test solsys_code.tests.test_import_campaign_csv` passes (existing + 3 new tests).
- `ruff check .` and `ruff format --check .` are clean.
- The demo notebook re-executes without error and contains the new leading-comment demonstration cell.
</verification>

<success_criteria>
- A CSV with leading free-text + blank rows before the real header imports every data row (no false CommandError).
- A CSV with no valid header within the scan cap (and one whose header is beyond the cap) still raises CommandError with zero CampaignRun rows created.
- The header-on-row-1 common case is unchanged: all pre-existing tests pass and the first data row still logs as "Row 2".
- Quality gates (`ruff check .`, `ruff format --check .`) stay clean.
</success_criteria>

<output>
Create `.planning/quick/260722-hpw-fix-import-campaign-csv-to-skip-leading-/260722-hpw-SUMMARY.md` when done
</output>
