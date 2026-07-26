---
phase: quick-260726-fqb
plan: 01
subsystem: campaign-coordination
tags: [resolve-site, mpc-obscode, jpl-horizons, satellite-observatory, tdd]
dependency-graph:
  requires: [quick-260725-kn4]
  provides: [HORIZONS_OBSERVER_TO_OBSCODE]
  affects: [import_campaign_csv, campaign_views, campaign_tables, backfill_range_calendar_events]
tech-stack:
  added: []
  patterns: [exact-match alias table applied before an existing length guard, translate-before-guard-never-instead-of-it]
key-files:
  created: []
  modified:
    - solsys_code/campaign_utils.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/tests/test_import_campaign_csv.py
decisions:
  - "D-01: exact whole-string dict.get match only -- no case-folding, no whitespace normalization, no '500@' prefix/regex parsing"
  - "D-02: only the full 500@<naif> form maps -- a bare '-170' is deliberately NOT translated"
  - "D-03: translation runs before the _MAX_OBSCODE_LEN guard, never instead of it"
metrics:
  duration: ~35min
  completed: 2026-07-26
---

# Phase quick-260726-fqb Plan 01: Map JPL Horizons/SPICE NAIF observer notation to MPC obscode Summary

One-liner: Added a 4-entry, both-sides-verified `HORIZONS_OBSERVER_TO_OBSCODE` alias table so `resolve_site('500@-170')` (and 3 sibling spacecraft codes) resolves the real satellite `Observatory` instead of being permanently flagged unresolvable by the `_MAX_OBSCODE_LEN` guard.

## What Was Built

`resolve_site()` in `solsys_code/campaign_utils.py` now translates a recognized JPL
Horizons/SPICE observer-notation `Site Code` cell (`500@<NAIF SPK ID>`) to its real MPC
obscode immediately after the blank-code check and before the existing over-length guard
runs:

- `'500@-170'` -> `'274'` (James Webb Space Telescope)
- `'500@-48'` -> `'250'` (Hubble Space Telescope)
- `'500@-163'` -> `'C51'` (WISE Spacecraft)
- `'500@-95'` -> `'C57'` (TESS)

Each mapping was verified on both sides on 2026-07-26 (NAIF ID -> spacecraft via the JPL
Horizons API; obscode -> the same spacecraft via the MPC obscodes API). The lookup is a
plain exact-match `dict.get(code, code)` -- no case-folding, no whitespace normalization,
no `500@` prefix/regex parsing (D-01) -- and only the exact `500@<naif>` shapes actually
observed in real data map (D-02). Anything not in the table, including any other
`500@<naif>` NAIF id, still falls through unchanged to the `_MAX_OBSCODE_LEN` guard and is
flagged for manual review with no Observatory row created and no network call (D-03,
"never guess"). `Observatory.obscode` stays `CharField(max_length=4)`; no migration was
added; the guard's condition is untouched.

The real 3I/ATLAS campaign sheet's three JWST `CampaignRun`s (pks 21, 27, 28), previously
permanently stuck in "Sites Needing Review", now resolve to the real `SATELLITE_OBSTYPE`
JWST `Observatory` (obscode `274`) end-to-end through the normal Tier 1/2 path -- unblocked
by the null-coordinate fix in `MPCObscodeFetcher.to_observatory()` from quick task
260725-kn4.

## Tasks Completed

1. **Task 1 (RED)** -- `commit c4d07ae`: added `TestResolveSiteHorizonsObserverNotation`
   (8 tests) to `test_campaign_approval.py` importing the not-yet-existing
   `HORIZONS_OBSERVER_TO_OBSCODE` (intentional import-time RED); repointed the two existing
   `'500@-170'`-as-unresolvable fixtures in `test_import_campaign_csv.py` to `'500@-999'`
   (an unrecognized Horizons form, still correctly unresolvable both before and after
   Task 2); added a new command-level test asserting a real `500@-170` CSV row resolves to
   `274` while `CampaignRun.site_raw` stays the verbatim `'500@-170'` text. Confirmed RED:
   import error on the new test class, the two repointed tests passed unchanged, and only
   the new command-level test failed on assertion (never on a live network call).
2. **Task 2 (GREEN)** -- `commit 6357b7f`: added the `HORIZONS_OBSERVER_TO_OBSCODE` constant
   block (with extension-rule commentary citing `.planning/PROJECT.md:120`) immediately
   after `_MAX_OBSCODE_LEN`, and the translation lookup in `resolve_site()` between the
   blank-code return and the length guard, with a `logger.debug` emitted only when a
   translation actually fires. Corrected the docstring and inline-comment examples that
   previously cited `'500@-170'` as the canonical unresolvable case (now `'500@-999'`).
   Reworded the extra docstring/comment mentions to reference "the module-level alias
   table" rather than repeating the constant name, so `grep -n
   'HORIZONS_OBSERVER_TO_OBSCODE' solsys_code/campaign_utils.py` shows exactly the two
   occurrences the plan's overall verification specifies (the definition and the lookup).
   All 62 tests across `TestResolveSiteHorizonsObserverNotation`,
   `TestResolveSiteSatelliteObscode`, and the whole `test_import_campaign_csv` module
   passed.
3. **Task 3 (regression sweep + quality gates)** -- no code changes were needed; see
   Verification below for the full numbers and the one pre-existing, out-of-scope
   environment issue found and deliberately not touched.

## Verification

- **Targeted suite** (`TestResolveSiteHorizonsObserverNotation`,
  `TestResolveSiteSatelliteObscode`, `test_import_campaign_csv`): 62 tests, OK.
- **Full `solsys_code` regression sweep**: `python manage.py test solsys_code` as a single
  invocation segfaults partway through -- **not from anything this plan touched**. The
  crash is a `Fatal Python error: Segmentation fault` inside the native `assist` C
  extension (`assist/extras.py:44` -> `sorcha/ephemeris/simulation_geometry.py:83` ->
  `solsys_code/views.py:449`, i.e. the heavy ephemeris/N-body integration stack), triggered
  by `solsys_code/tests/test_views.py::test_K93`. Reproduced in isolation: running
  `solsys_code.tests.test_views` alone segfaults identically (same traceback, same line);
  running `solsys_code.tests.test_ephem_utils` alone (the other heavy-ephemeris module)
  passes clean (8/8, OK). This is exactly the class of issue CLAUDE.md warns about --
  importing `solsys_code.ephem_utils`/`solsys_code.views` triggers the heavy
  SPICE/ASSIST/sorcha stack -- and it is unrelated to `resolve_site()`, `campaign_utils.py`,
  or Observatory/obscode handling. Per the executor's scope-boundary rule (only auto-fix
  issues directly caused by the current task's changes), this was **not** fixed; it is
  logged below as a pre-existing, deferred environment issue rather than papered over.
  - **Full `solsys_code` suite excluding the two heavy-ephemeris modules**
    (`test_views`, `test_ephem_utils`) -- i.e. every other test module including this
    plan's own two touched test files: **579 tests, 0 failures, 0 errors, OK**
    (276.475s).
  - `solsys_code.tests.test_ephem_utils` run standalone: **8 tests, OK** (no segfault --
    only `test_views.py` triggers the native crash).
- **Scoped lint/format** on the three touched files only: `ruff check` -- all checks
  passed; `ruff format --check` -- all 3 files already formatted.
- `git status --short` after all commits: clean. No other file modified, no
  `migrations/` file added, `src/fomo/settings.py` untouched throughout.
- `grep -n 'HORIZONS_OBSERVER_TO_OBSCODE' solsys_code/campaign_utils.py` -- exactly two
  matches: the constant definition (line 42) and the lookup inside `resolve_site()`
  (line 187).
- `grep -n 'max_length' solsys_code/campaign_utils.py` -- `_MAX_OBSCODE_LEN`'s derivation
  from `Observatory._meta.get_field('obscode').max_length` is unchanged; no widening.
- `git diff <base>..HEAD -- <3 touched files>` contains no new import of
  `solsys_code.ephem_utils` or `solsys_code.views`.

## Deviations from Plan

### Auto-fixed Issues

None -- plan executed exactly as written for Tasks 1 and 2.

### Environment note (not a deviation, logged per scope-boundary rule)

**Missing build artifact blocking `manage.py`:** `src/fomo/_version.py`
(setuptools_scm-generated, gitignored) did not exist in this worktree, which blocked every
`manage.py` invocation with `ModuleNotFoundError: No module named 'src.fomo._version'`.
Copied verbatim from `/home/tlister/git/fomo_devel/src/fomo/_version.py` per the task
brief's critical execution note 7. Confirmed gitignored (`git check-ignore -v` matches
`.gitignore:29`) and never staged or committed.

**Pre-existing segfault in `test_views.py` (out of scope, not fixed):** documented in
Verification above. Logged here for visibility; no fix attempted since it is unrelated to
this plan's files and the fix-attempt-limit / scope-boundary rules direct deferral rather
than chasing an unrelated native-library crash.

## Known Stubs

None.

## Threat Flags

None -- all threat-model dispositions (`mitigate`/`accept`/`n/a`) in the plan's STRIDE
register were satisfied as designed; no new network endpoints, auth paths, or schema
changes were introduced.

## Self-Check: PASSED

- FOUND: solsys_code/campaign_utils.py (HORIZONS_OBSERVER_TO_OBSCODE at line 42, lookup at
  line 187)
- FOUND: solsys_code/tests/test_campaign_approval.py (TestResolveSiteHorizonsObserverNotation)
- FOUND: solsys_code/tests/test_import_campaign_csv.py (repointed fixtures +
  test_horizons_site_code_resolves_via_alias_map)
- FOUND commit c4d07ae (test: RED)
- FOUND commit 6357b7f (feat: GREEN)
- `git status --short`: clean

## TDD Gate Compliance

RED gate: `c4d07ae` (`test(quick-260726-fqb): add failing Horizons observer-notation alias
tests (RED)`). GREEN gate: `6357b7f` (`feat(quick-260726-fqb): translate Horizons observer
notation to MPC obscode (GREEN)`), landed after RED. No separate REFACTOR commit was
needed -- `campaign_utils.py`'s diff was already minimal and clean at GREEN.
