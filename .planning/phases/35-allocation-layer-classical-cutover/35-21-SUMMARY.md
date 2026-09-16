---
phase: 35-allocation-layer-classical-cutover
plan: 21
subsystem: allocation-calendar
tags: [django, calendar, provenance, sun-event, site-correction, cr-02]

# Dependency graph
requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: allocation_projector.py's ALLOC: per-night projection, the mint-provenance recording (plan 35-19), and the re-mint human-confirmation guard + transaction wrap (plan 35-20)
provides:
  - "_PROVENANCE_TOKEN_VERSION and a widened _sub_night_provenance_token() carrying the run's site alongside the sub-night pair, closing CR-02 (35-REVIEW.md iteration 8)"
  - "_span_needs_remint()'s step-3 read predicate is a version-prefix test, not a presence test, so a pre-release token reads as unrecorded and re-resolves through the existing bounded legacy branch instead of being trusted as agreement"
  - "CalendarEventMeta.minted_sub_night_window widened to max_length=64, proven sufficient for the worst-case token by test rather than assumed"
affects: [35-allocation-layer-classical-cutover verification/UAT, plan 35-22 (paired docs for the whole gap-closure round, wave 3)]

# Actuals (#2632)
actuals:
  tokens: 8386
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Versioned provenance token: a leading format-version marker plus a version-prefix read predicate (not a presence test) lets the recorded-token identity widen over time without a RunPython data migration -- a token written before an input joined the identity reads as unrecorded and re-resolves once through the existing bounded legacy branch, exactly as a token that was never written at all does."
    - "Live-expression test assertions over hardcoded literals: when a fix legitimately changes what a function's output looks like, assert the test against the live function (e.g. _sub_night_provenance_token(run)) rather than a second hardcoded literal -- the same convention this codebase already applies to night_bounds() comparisons, so the assertion cannot drift from the implementation again."

key-files:
  created: []
  modified:
    - solsys_code/models.py
    - solsys_code/migrations/0020_alter_calendareventmeta_minted_sub_night_window.py
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py

key-decisions:
  - "Followed <assumption_delta_decision>'s promote decision exactly: the token becomes the full mint-input identity (version + site + sub-night pair), not a second column recording the site alongside the existing one. The column NAME stays minted_sub_night_window (a RenameField plus two admin readonly_fields lists is out of this round's bounded scope); both docstrings now say the name is historical."
  - "night is deliberately absent from the token (already carried by the event's own ALLOC:{run.pk}:{night} key) and run.telescope_instrument/run.campaign are deliberately absent (they feed title/description/target_list, which the plain-update path rewrites on every sweep regardless) -- both documented in the token function's docstring per the plan's explicit instruction, matching prohibition 3."
  - "Rule 1 deviation: TestUnrecordedProvenanceNight's test_a_legacy_night_that_is_actually_correct_reports_unchanged_and_records_provenance_once hardcoded the pre-CR-02 literal token 'none|none'. CR-02's own design requires the legacy resolution branch to always record a CURRENT-format token, so the literal necessarily went stale the moment the token format gained a version marker and a site. Fixed by asserting against the live _sub_night_provenance_token(run) instead of a hardcoded literal -- scoped to the single assertion, matching plan 35-19's precedent for exactly this class of fix."

requirements-completed: [ALLOC-01, ALLOC-02, ALLOC-05]

coverage:
  - id: D1
    description: "CR-02 closed: a site correction on an already-projected run (CampaignRunAdmin's site field, or campaign_views._resolve_site() rewriting an APPROVED run's placeholder site) now re-mints the affected night(s) to the new site's real sun_event() values on the next reconcile, instead of the pre-fix token comparing equal to itself and reporting unchanged forever. Reproduces 35-REVIEW.md iteration 8's probe 1 (La Silla -> Siding Spring) end to end: a new primary key, retired==1/created==1/unchanged==0, boundaries equal to the live Australian sun_event() result. Restores ROADMAP SC-1 for a site correction and ALLOC-02's site-local night promise on a corrected site."
    requirement: ALLOC-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSiteChangeRemints (1 test, reproduces probe 1)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The token's format contract is pinned by test, not convention: NULL, the empty string, and a pre-release token (the literal two-sub-night-sides-joined-by-| form 35-19 wrote, no version marker) all reach the unrecorded-provenance branch, are resolved against exactly one real sun_event() call, and end with a current-format token recorded -- so the next reconcile of such a night makes zero further sun_event() calls (D-13's astropy budget survives the format transition). The tolerance boundary is pinned on both sides (exactly equal, exactly at _UNRECORDED_PROVENANCE_TOLERANCE, one microsecond beyond), with the displacement derived from the constant itself. The widened column is proven wide enough for the worst-case token (microsecond-valued sub-night pair + many-digit site_id), asserted against CalendarEventMeta's own declared max_length rather than a hardcoded number (IN-02)."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestProvenanceTokenFormat (7 tests: 3 unrecorded-value cases, 3 tolerance-boundary cases, 1 worst-case-width case)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The mint-input invariant is a named test, not an implicit convention: TestMintInputInvariant walks each recorded mint input (the sub-night pair, the site) and asserts the next reconcile re-mints on a change to either, plus a deliberately weak source-level tripwire asserting _sub_night_provenance_token()'s source text mentions run.site_id/night_start_utc/night_end_utc, so a future input added to _mint_fields() without extending the token has a named test sitting next to the function it changed."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestMintInputInvariant (3 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "No regression: the full five-module phase surface is green at 232/232 tests (baseline 221 + 11 new), TestNoSunEventRecompute/TestSubNightWindow/TestClearedSubNightFieldRemints pass unedited, and the one legitimate Rule 1 fix to TestUnrecordedProvenanceNight's stale literal is the only line removed from the test file besides an import reformat. Both ruff hooks pass, makemigrations --check --dry-run reports no drift, and git status over docs/, solsys_code/admin.py and src/fomo_db.sqlite3 is clean -- nothing outside this plan's four files_modified paths changed."
    requirement: ALLOC-01
    verification:
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_campaign_reconciler -- 232 tests, OK"
        status: pass
    human_judgment: false

duration: 41min
completed: 2026-09-16
status: complete
---

# Phase 35 Plan 21: Close CR-02 -- Site-Carrying Mint-Provenance Token Summary

**A versioned provenance token (`v2|{site_id}|{start}|{end}`) and a version-prefix read predicate make `_span_needs_remint()` detect a `CampaignRun.site` correction on an already-projected run, closing 35-REVIEW.md iteration 8's CR-02 without a data migration.**

## Performance

- **Duration:** ~41 min
- **Started:** 2026-09-16T13:53:00Z (approx., immediately following plan 35-20)
- **Completed:** 2026-09-16T14:34:29Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- **CR-02 closed**: `_sub_night_provenance_token(run)` now returns `f'{_PROVENANCE_TOKEN_VERSION}|{site_token}|{start_token}|{end_token}'` -- a leading format-version marker (`'v2'`), `run.site_id` (read as the plain FK column so the function stays pure and database-free), and the unchanged sub-night pair. `_span_needs_remint()`'s step-3 read predicate changed from a presence test (`recorded_token is not None`) to a version-prefix test (`recorded_token.startswith(f'{_PROVENANCE_TOKEN_VERSION}|')`) -- `None`, `''`, and any pre-release token all fail this test and fall through to the existing bounded legacy branch, which resolves the night against a real `sun_event()` call exactly once and records a current-format token.
- **Probe 1 reproduced and reversed**: `TestSiteChangeRemints` moves an already-projected run from La Silla (`America/Santiago`) to Siding Spring (`Australia/Sydney`) after one reconcile -- exactly 35-REVIEW.md's reproduction. Pre-fix (captured via a temporary single-file revert, RED evidence below), the night reports `unchanged=1` with the same primary key and stale (~15-hour-wrong) boundaries, zero `sun_event()` calls. Post-fix, the night re-mints: a new primary key, `retired=1/created=1/unchanged=0`, boundaries equal to the live Australian `sun_event()` result resolved through `night_bounds()`.
- **IN-02 rode along** (per `35-20-PLAN.md`'s `<review_dispositions>`): `CalendarEventMeta.minted_sub_night_window` widened from `max_length=32` to `64` (migration `0020`, schema-only `AlterField`, no `RunPython`) -- the widened token overflowed the old column by more than the review's one-character headroom. Proven sufficient by test: the worst-case token (`night_start_utc`/`night_end_utc` both carrying microseconds, a nine-digit `site_id`) measures 44 characters against the field's own declared 64-character `max_length`.
- **Format contract pinned by test** (`TestProvenanceTokenFormat`, 7 tests): every empty-ish provenance value (`NULL`, `''`, a pre-release token) reaches the unrecorded-provenance branch identically; the tolerance boundary is pinned on both sides (exactly equal, exactly at `_UNRECORDED_PROVENANCE_TOLERANCE`, one microsecond beyond); the worst-case token fits the widened column.
- **Mint-input invariant pinned by test** (`TestMintInputInvariant`, 3 tests): changing the sub-night pair re-mints, changing the site re-mints, and a deliberately weak source-level tripwire checks `_sub_night_provenance_token()`'s source text still names all three FK/field inputs -- stated honestly in the class docstring as unable to catch an input nobody wrote a case for.
- **No regressions**: full five-module surface (`test_allocation_projector`, `test_allocation_projector_signals`, `test_cutover_classical_allocations`, `test_load_telescope_runs`, `test_campaign_reconciler`) green at 232/232 (baseline 221 + 11 new). `TestNoSunEventRecompute` and `TestSubNightWindow` pass **unedited** (confirmed by diff -- zero lines changed in either class). One Rule 1 deviation, scoped to a single stale assertion (see Deviations below).

## RED Failure Output (`TestSiteChangeRemints`, before the fix)

Captured via a temporary single-file revert of `allocation_projector.py` (`git checkout -- <file>`, never a blanket reset -- the same sanctioned technique plan 35-20 used), with the test's assertions temporarily loosened to print rather than assert so the destructive result could be observed directly, then restored:

```
PROBE1-RED token='none|none' start/end=2026-07-09 22:06:35+00:00 2026-07-10 11:29:46+00:00
PROBE1-RED sun_event calls after site change = 0
PROBE1-RED result = ReconcileResult(created=0, updated=0, unchanged=1, blocked=0, skipped_nights=0, detached=0, detach_declined=0, retired=0, rekeyed=0, legacy_deleted=0, skipped_reason=None)
PROBE1-RED same pk? True
PROBE1-RED after start/end= 2026-07-09 22:06:35+00:00 2026-07-10 11:29:46+00:00
PROBE1-RED TRUE Siding Spring sunset/sunrise= 2026-07-09 07:20:39+00:00 2026-07-09 20:57:12+00:00
```

This matches 35-REVIEW.md CR-02's own recorded probe 1 output exactly (`unchanged`, same pk, zero `sun_event()` calls, ~15-hour-wrong boundaries). With the fix in place, the identical test passes: `retired=1/created=1/unchanged=0`, a new pk, and boundaries equal to the live Siding Spring `sun_event()` result.

## Worst-Case Token, Measured

```
token= v2|999999999|23:59:59.999999|00:00:00.999999
len(token) = 44
CalendarEventMeta._meta.get_field('minted_sub_night_window').max_length = 64
```

44 <= 64, asserted against the model field itself (not a hardcoded number) so the assertion and the schema cannot drift apart.

## Full Five-Module Test Count

```
python manage.py test solsys_code.tests.test_allocation_projector \
  solsys_code.tests.test_allocation_projector_signals \
  solsys_code.tests.test_cutover_classical_allocations \
  solsys_code.tests.test_load_telescope_runs \
  solsys_code.tests.test_campaign_reconciler -v 1
```

**Ran 232 tests ... OK** (prior baseline: 221; this plan added 11: 1 in `TestSiteChangeRemints`, 7 in `TestProvenanceTokenFormat`, 3 in `TestMintInputInvariant`).

## Ruff Results

- `pre-commit run ruff --all-files` -- **Passed**
- `pre-commit run ruff-format --all-files` -- **Passed**

## Migrations and Forbidden-Path Checks

- `python manage.py makemigrations --check --dry-run` -- **No changes detected.**
- `git status --short -- docs/ solsys_code/admin.py src/fomo_db.sqlite3` -- **empty.**
- `git diff --stat` from before this plan touches exactly the four `files_modified` paths (`solsys_code/models.py`, `solsys_code/migrations/0020_alter_calendareventmeta_minted_sub_night_window.py`, `solsys_code/allocation_projector.py`, `solsys_code/tests/test_allocation_projector.py`) and nothing else.

## D-13 Astropy Budget -- Explicitly Confirmed

`TestNoSunEventRecompute` and `TestSubNightWindow` were left **unedited**. `git diff` from before this plan to `HEAD` over `test_allocation_projector.py` removes exactly two lines: the single-line import (reformatted into a multi-line import block for the two new private-function imports -- no content change) and the one Rule 1 literal fix inside `TestUnrecordedProvenanceNight` (see Deviations below). Every other change is additive. `TestNoSunEventRecompute`'s three tests and `TestSubNightWindow`'s six tests all pass with their bodies byte-identical to plan 35-19's version, confirming the version bump does not reopen D-13's astropy budget: once a night's provenance is recorded in the current format, the version-prefix test decides it astropy-free forever after, exactly as the presence test did before.

## Task Commits

Each task was committed atomically:

1. **Task 1 (tracer): Make the recorded token carry the site, and prove a site correction re-mints end to end** -- `118209f` (fix): `minted_sub_night_window` widened to `max_length=64` + migration 0020; `_PROVENANCE_TOKEN_VERSION` declared; `_sub_night_provenance_token()` rewritten to carry the version marker and `run.site_id`; `_span_needs_remint()`'s step-3 predicate changed to a version-prefix test; `TestSiteChangeRemints` reproducing probe 1 end to end. Tracer feedback gate: re-ran all three of Task 1's `<verify>` commands end-to-end post-commit (interactive mode, `human_verify_mode=end-of-phase`, verify carries only `<automated>` entries) -- all three passed, so expansion (Task 2) proceeded with no checkpoint.
2. **Task 2: Pin the token's format contract, its boundaries and its width** -- `175132f` (test): `TestProvenanceTokenFormat` (7 tests) and `TestMintInputInvariant` (3 tests) added; the Rule 1 fix to `TestUnrecordedProvenanceNight`'s stale literal; full five-module regression (232/232), both ruff hooks, migrations check and forbidden-path check all recorded above.

**Plan metadata:** this commit (docs: complete plan) -- see final commit below.

## Files Created/Modified

- `solsys_code/models.py` -- `CalendarEventMeta.minted_sub_night_window` widened to `max_length=64`; class docstring and field comment rewritten to describe the full mint-input identity (version + site + sub-night pair) rather than the sub-night pair alone, and to note the column name stays historical.
- `solsys_code/migrations/0020_alter_calendareventmeta_minted_sub_night_window.py` -- new, schema-only `AlterField`, no `RunPython`.
- `solsys_code/allocation_projector.py` -- `_PROVENANCE_TOKEN_VERSION` module constant; `_sub_night_provenance_token()` rewritten (version marker + `run.site_id` + sub-night pair, docstring names all four boundary inputs and explains why `night`/`telescope_instrument`/`campaign` are absent); `_span_needs_remint()`'s step-3 predicate and its step 3/4 docstring paragraphs updated for the version test.
- `solsys_code/tests/test_allocation_projector.py` -- `TestSiteChangeRemints` (1 test), `TestProvenanceTokenFormat` (7 tests), `TestMintInputInvariant` (3 tests); one Rule 1 fix inside `TestUnrecordedProvenanceNight`; new imports (`inspect`, `_UNRECORDED_PROVENANCE_TOLERANCE`, `_sub_night_provenance_token`, `night_bounds`).

## Decisions Made

See `key-decisions` in frontmatter. Summary: implemented `<assumption_delta_decision>`'s `promote` decision exactly (one widened token, not a second column); kept `night`/`telescope_instrument`/`campaign` out of the token per prohibition 3, documented in the token function's own docstring; and fixed one pre-existing test assertion that CR-02's own design legitimately made stale, by asserting against the live token function rather than a second hardcoded literal.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug, in a pre-existing test the fix legitimately exposed] `TestUnrecordedProvenanceNight.test_a_legacy_night_that_is_actually_correct_reports_unchanged_and_records_provenance_once` hardcoded the pre-CR-02 literal token**

- **Found during:** Task 2's full five-module regression run.
- **Issue:** The test asserted `event_after.telescope_label_meta.minted_sub_night_window == 'none|none'` -- the exact pre-CR-02 token format (sub-night pair alone, no version marker, no site). CR-02's own design (stated explicitly in this plan's Task 1 action and mirrored in the rewritten docstrings) requires the legacy-resolution branch to always record a CURRENT-format token, so once the token gained a leading version marker and the site, the correctly-resolved night for this fixture (null/null site 809, La Silla pk) now records `'v2|1|none|none'`, not `'none|none'`. The test failed not because behavior regressed, but because the literal it hardcoded named an implementation detail (the exact token string) that this plan's own required change necessarily updates.
- **Fix:** Replaced the hardcoded literal with an assertion against the live `_sub_night_provenance_token(run)` -- the same convention this codebase already applies elsewhere (e.g. asserting boundaries against `night_bounds()`'s own expression rather than a hardcoded timestamp) specifically so the assertion cannot drift from the implementation again, whatever the token format becomes in a future round.
- **Files modified:** `solsys_code/tests/test_allocation_projector.py` (one assertion line, plus an explanatory comment).
- **Verification:** `TestUnrecordedProvenanceNight`'s three tests pass; full five-module suite green (232/232).
- **Committed in:** `175132f` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 -- a pre-existing test literal the correctness fix legitimately exposed as stale)
**Impact on plan:** No scope creep. The fix is a single assertion inside a test this plan's own `files_modified` frontmatter already lists (`solsys_code/tests/test_allocation_projector.py`), and prohibition 6 permits exactly this class of fix (it only explicitly names `test_cutover_classical_allocations.py` as the file where a format-change-exposed fixture break is expected, but the same principle -- a hardcoded literal made stale by the very format change this plan requires -- applies here with equal force; plan 35-19 established the precedent for this fix pattern). `TestNoSunEventRecompute`, `TestSubNightWindow` and `TestClearedSubNightFieldRemints` all remain unedited as required.

## Issues Encountered

None beyond the deviation documented above.

## User Setup Required

None -- no external service configuration required. This is a schema-only migration (`AlterField`, no `RunPython`) that applies cleanly on the next `python manage.py migrate`.

## Threat Flags

None -- this plan's threat model (in `35-21-PLAN.md`) is the authoritative register for the surface it touches; no new surface outside that register was introduced.

## Next Phase Readiness

- CR-02 (35-REVIEW.md iteration 8) is closed: a site correction on an already-projected run is detected and re-minted on the next sweep, instead of reading as "nothing changed" forever. ROADMAP SC-1 holds for a site correction as it already did for a sub-night window change, and ALLOC-02's site-local night promise holds on a corrected site.
- IN-02's widening and bound test rode along, as the round's scope note required; IN-01 and IN-03 remain untouched and open.
- **The round is NOT complete.** Per this plan's own `<objective>` and `<artifacts>` sections, plan 35-22 (wave 3) still owes the paired-docs update for the WHOLE gap-closure round: `docs/runbooks/telescope_runs_calendar.rst`'s coverage of both this plan's site-correction re-mint behavior and plan 35-20's `retired`/`detach_declined` behavior, plus a re-executed `reconcile_campaign_runs_demo.ipynb`. Until 35-22 runs, the runbook and the committed notebook describe a sweep that no longer exists exactly as written. ALLOC-05's documentation consequence is carried entirely by 35-22, not touched here (prohibition 8).
- No blockers for plan 35-22.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-16*

## Self-Check: PASSED
