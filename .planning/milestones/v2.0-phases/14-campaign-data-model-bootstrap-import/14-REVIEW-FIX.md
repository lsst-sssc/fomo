---
phase: 14-campaign-data-model-bootstrap-import
fixed_at: 2026-07-03T08:24:25Z
review_path: .planning/phases/14-campaign-data-model-bootstrap-import/14-REVIEW.md
iteration: 1
findings_in_scope: 11
fixed: 11
skipped: 0
status: all_fixed
---

# Phase 14: Code Review Fix Report

**Fixed at:** 2026-07-03T08:24:25Z
**Source review:** .planning/phases/14-campaign-data-model-bootstrap-import/14-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 11 (2 critical, 9 warning; `fix_scope: critical_warning` -- the 2 Info
  findings, IN-01/IN-02, were out of scope and not touched)
- Fixed: 11
- Skipped: 0

All 11 in-scope findings were fixed and committed atomically, one commit per finding.
Each fix was verified with Tier 1 (re-read modified section), Tier 2 (`python -c
"import ast; ast.parse(...)"` syntax check on every touched file), and the project's own
gates (`ruff check .`, `ruff format --check .` clean; `./manage.py test solsys_code`
green -- 242/242 tests passing after the full set of fixes, up from 26 at review time).

## Fixed Issues

### CR-01: PM/AM time-of-day markers are matched but never applied

**Files modified:** `solsys_code/campaign_utils.py`, `solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `70f6ef3`
**Status:** fixed: requires human verification (logic-correctness fix -- see note below)
**Applied fix:** Made the `am`/`pm` regex groups in `_HHMM_RANGE` and `_APPROX_HOUR`
capturing (they were previously non-capturing and discarded), and added a `_to_24h(hour,
meridiem)` helper that converts a parsed 1-12 hour + optional am/pm marker to a correct
24-hour value (with the 12am->0 / 12pm->12 edge cases handled). `parse_obs_window` now
applies this conversion on both the HH:MM-range and approximate-hour paths. Added
regression tests reproducing the review's exact examples: `parse_obs_window('2025-07-16',
'~7:00:00 PM')` now correctly returns hour 19 (was 7), and `'08:50 pm - 11:50 pm'` now
returns 20:50-23:50 (was 08:50-11:50); an AM-marker no-op case and the pre-existing
24h-format cases were also re-verified to be unaffected.
**Note:** per the logic-bug verification limitation, syntax/structure checks alone can't
prove conditional-logic correctness -- flagged for a human to eyeball the `_to_24h` edge
cases (noon/midnight) even though the reproduction cases from REVIEW.md now pass.

### CR-02: Natural key collides for distinct rows when UT Time Range is unparseable

**Files modified:** `solsys_code/campaign_utils.py`,
`solsys_code/management/commands/import_campaign_csv.py`,
`solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `dab0314`
**Status:** fixed: requires human verification (logic-correctness fix -- see note below)
**Applied fix:** `parse_obs_window` now returns a 4th value, `ut_needs_review`, mirroring
`resolve_site`'s `needs_review` flag -- `True` only when `ut_start` is the midnight-UTC
fallback (blank/unparseable UT Time Range) rather than an actually-parsed time.
`import_campaign_csv.py`'s row loop tracks how many times each `(campaign,
telescope_instrument, ut_start)` key has been seen among fallback-flagged rows in the
current batch (`seen_fallback_keys`); on a repeat, it offsets `ut_start` by N seconds
before calling `insert_or_create_campaign_run` so the two rows get distinct natural keys
instead of merging, and logs a `WARNING duplicate natural key ...` line to stderr so the
operator knows to check the source rows. The offset is deterministic (based on row order
within the file), so re-running the same CSV twice still produces the same offsets and
stays idempotent (D-04) -- verified by the pre-existing `test_idempotent_rerun_no_duplicates`
test, which still passes. Added a regression test reproducing the review's exact scenario:
two same-telescope/same-date rows with different unparseable `UT Time Range` text (`'redo
later'` and `'exact start time not logged'`) now produce 2 `CampaignRun` rows (was 1) with
a `WARNING` logged to stderr.
**Notebook impact:** checked `docs/notebooks/pre_executed/fixtures/campaign_sample.csv`
for PM times or duplicate same-telescope/same-date unparseable-time rows per the task
instructions -- found neither (the fixture's only am/pm marker is `'~1 am'`, and its one
blank-time row has no same-telescope/same-date sibling). The paired demo notebook's
committed output is therefore unaffected by CR-01 or CR-02 and was **not** regenerated.
**Note:** flagged for human verification for the same logic-bug-limitation reason as CR-01.

### WR-01: `resolve_site`'s Tier 2 MPC lookup has no timeout and doesn't catch network exceptions

**Files modified:** `solsys_code/campaign_utils.py`, `solsys_code/solsys_code_observatory/utils.py`,
`solsys_code/tests/test_import_campaign_csv.py`, `solsys_code/solsys_code_observatory/tests/test_utils.py`
**Commit:** `4633925`
**Applied fix:** Added an explicit `timeout: float = 10` parameter to
`MPCObscodeFetcher.query()`, passed through to `requests.get(...)`. `resolve_site` now
wraps the `fetcher.query(code, timeout=10)` call in `try/except
requests.exceptions.RequestException`, falling through to tier 3 on any network failure
instead of letting the exception propagate and crash the whole batch import. Added a
regression test mocking `requests.get` to raise `ConnectionError` and asserting
`resolve_site` still returns a flagged placeholder, plus two tests on `query()` itself
confirming the timeout is passed through (default and explicit).

### WR-02: IntegrityError handler in Tier 2 assumes any conflict is an obscode race

**Files modified:** `solsys_code/campaign_utils.py`, `solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `7b92278`
**Applied fix:** Wrapped the tier-2 IntegrityError handler's re-fetch
(`Observatory.objects.get(obscode=code)`) in `try/except Observatory.DoesNotExist`,
falling through to tier 3 rather than letting `DoesNotExist` propagate when the
IntegrityError was actually caused by a `name` collision on a *different* obscode (so no
Observatory exists for `code` to re-fetch). Added a regression test mocking
`to_observatory()` to raise `IntegrityError` with no pre-existing Observatory for the
target obscode, confirming `resolve_site` falls through to a tier-3 placeholder instead
of raising.

### WR-03: Tier 3 placeholder creation has no race protection, unlike Tier 2

**Files modified:** `solsys_code/campaign_utils.py`, `solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `5390661`
**Applied fix:** Wrapped `Observatory.objects.create(...)` in tier 3 with the same
`except IntegrityError: return Observatory.objects.get(obscode=code), True` pattern
already used in tier 2. Added a regression test simulating a concurrent-process race
(pre-existing Observatory row + `Observatory.objects.create` mocked to raise
`IntegrityError`), confirming `resolve_site` re-fetches instead of crashing.

### WR-04: `resolve_site` doesn't catch malformed-but-"ok" MPC API responses

**Files modified:** `solsys_code/campaign_utils.py`, `solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `2ee379e`
**Applied fix:** Broadened the tier-2 `except MissingDataException` clause to also catch
`(KeyError, ValueError, TypeError)` around the `fetcher.to_observatory()` call, falling
through to tier 3 on any malformed-but-200-OK MPC response instead of crashing. Added a
regression test reproducing the review's exact repro case (a `response.json()` mock
missing `short_name` etc.), confirming `resolve_site` now falls through to a flagged
placeholder instead of raising `KeyError`.

### WR-05: No DB-level uniqueness constraint backs the documented natural key

**Files modified:** `solsys_code/models.py`,
`solsys_code/migrations/0003_campaignrun_natural_key_unique_constraint.py`
**Commit:** `23e9ca1`
**Applied fix:** Added `models.UniqueConstraint(fields=['campaign', 'telescope_instrument',
'ut_start'], name='unique_campaign_run_natural_key')` to `CampaignRun.Meta.constraints`
(exactly as suggested in REVIEW.md) and generated the corresponding migration via
`./manage.py makemigrations`. Verified `makemigrations --check --dry-run` reports no
further pending model changes, and the full `solsys_code` test suite (238 tests at that
point, including the pre-existing idempotent-rerun and CampaignRun-creation tests) passes
with the constraint in place.

### WR-06: Skipped-row error log includes the full raw CSV row, including contact PII

**Files modified:** `solsys_code/management/commands/import_campaign_csv.py`,
`solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `1662885`
**Applied fix:** Replaced the `(row: {row!r})` full-row dump in the skipped-row stderr
log with only the natural-key-relevant fields (`Telescope/Instrument=...`, `Obs.
Date=...`), matching REVIEW.md's suggested fix exactly. Updated the existing
`test_natural_key_failure_skipped_and_logged` test to include Contact Person/Email in the
skipped row and assert the telescope name still appears in the log; added a new
`test_natural_key_failure_log_excludes_contact_pii` test asserting the contact name/email
no longer appear in stderr.

### WR-07: Re-importing always overwrites `target` to the auto-resolved value

**Files modified:** `solsys_code/management/commands/import_campaign_csv.py`
**Commit:** `47734fb`
**Applied fix:** Chose REVIEW.md's documentation-only option (lower risk than adding
target-provenance tracking, which would be a larger behavior change for a genuine design
tradeoff rather than a bug). Added an explicit `WARNING:` paragraph to the command's
`help` text and a corresponding paragraph in `handle()`'s docstring explaining that
`target` is unconditionally reset to the auto-resolved value on every run/re-import, plus
an inline comment at the `fields['target'] = auto_target` assignment site pointing back
to the docstring. No behavior change, so no new test was needed.

### WR-08: `map_observation_status` substring matching has no negation awareness

**Files modified:** `solsys_code/campaign_utils.py`, `solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `24e5511`
**Status:** fixed: requires human verification (logic-correctness fix -- see note below)
**Applied fix:** Added a `_NOT_OBSERVED_RE = re.compile(r'\bnot\s+observ', re.IGNORECASE)`
guard that's checked only when the loop reaches the `('observ', OBSERVED)` table entry
(i.e. no more-specific keyword like `'weather'`/`'complet'` already matched earlier in
the ordered table) -- if the guard matches, that entry is skipped and evaluation falls
through to `'upcoming'`/`'planned'`/the `REQUESTED` default, rather than misclassifying a
bare `"Not observed"` as `OBSERVED`. Preserves the existing correct behavior for
co-occurring cases like `"Not observed -- weather"` (still maps to
`WEATHER_TECH_FAILURE` via the earlier `'weather'` match, unaffected by the negation
guard). Added regression tests for both the bare-negation case and the
negation-with-co-occurring-keyword case.
**Note:** flagged for human verification since this changes classification logic (not
just exception handling), per the verification strategy's logic-bug limitation.

### WR-09: No upfront validation of the CSV's column shape

**Files modified:** `solsys_code/management/commands/import_campaign_csv.py`,
`solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `61f3075`
**Applied fix:** Added a `_REQUIRED_HEADERS = ('Telescope / Instrument', 'Obs. Date', 'UT
Time Range')` module constant (the D-05 natural-key columns) and a check against
`reader.fieldnames` immediately after opening the CSV, before any row processing. If any
required column is missing, raises `CommandError` with the missing column name(s) and the
full found-columns list, rather than letting every row silently skip one-by-one with only
a per-row stderr line. Added a regression test writing a CSV with `'Telescope /
Instrument'` renamed to `'Telescope'`, confirming `CommandError` is raised (mentioning the
missing column) and zero `CampaignRun` rows are created.

## Skipped Issues

None -- all 11 in-scope findings were fixed.

## Verification

- `python -c "import ast; ast.parse(...)"` syntax check: clean on every file touched, at
  every step.
- `ruff check .` / `ruff format --check .` (the project's authoritative gate, run from the
  repo root exactly as CLAUDE.md specifies): clean on every source/test file touched. The
  new migration `0003_campaignrun_natural_key_unique_constraint.py` shows D101/E501 only
  when linted via an explicit file path (which bypasses pyproject.toml's `[tool.ruff]
  exclude = ["solsys_code/**/migrations/*.py", ...]`) -- this is the same pre-existing
  pattern as the sibling migrations `0001_calendareventtelescopelabel.py` and
  `0002_campaignrun.py`, not a regression, and the file is correctly excluded (and the
  pre-commit `ruff` hook reported "Passed") under the repo-wide `ruff check .` form that's
  actually enforced.
- `./manage.py test solsys_code`: 242/242 tests passing after all 11 fixes (up from 26 at
  review time -- 4 new tests for CR-01, 3 for CR-02, 5 for WR-01, 1 for WR-02, 1 for
  WR-03, 1 for WR-04, 2 for WR-06, 2 for WR-08, 1 for WR-09; no test needed for WR-05
  beyond the pre-existing suite passing with the new constraint, or WR-07's
  documentation-only change).
- Every fix was committed individually via `git commit` with the pre-commit hook suite
  (template-version check, ruff lint, ruff format, Sphinx docs build, and the project's
  `pytest`-based "Run unit tests" hook) passing at commit time.

---

_Fixed: 2026-07-03T08:24:25Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
