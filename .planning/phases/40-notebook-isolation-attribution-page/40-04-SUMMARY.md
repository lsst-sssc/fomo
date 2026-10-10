---
phase: 40-notebook-isolation-attribution-page
plan: 04
subsystem: ui
tags: [attribution, return-state, redirect, django-tables2, playwright, runbook, notebook]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-03's High-band marker classes and Bootstrap 5.3 class names on the attribution page; 40-02's notebook guard; 40-01's scratch-database helper"
provides:
  - "attribution_return_state(): one whitelist (band, dismissed-page, dismissed-sort, confirmed-page, confirmed-sort) shared by the GET render and the POST redirect"
  - "Every attribution action redirects through one _redirect() keeping the validated state; the two Undo actions add a constant section fragment"
  - "Undo on a Dismissed row returns to the same page with the section open, proven server-side from the real rendered form and in Chromium with 27 dismissals"
  - "One dismissed_section_open context boolean for the Dismissed section's class, heading class and aria-expanded (IN-15)"
  - "Runbook attribution-pass text and campaign_lifecycle_demo.ipynb cells for the kept place and the High-band bar; the ledger closes WR-15, WR-18, IN-12, IN-15"
affects: [phase 40 verification, 41 triage]

actuals:
  tokens: 17000
  tasks: 3
  commits: 4

plan_head_before: f13ff473e8088906f4e290087fa1394acd629f71
plan_head_after: bda3591e94bd46a19990a3d31bda712eb60a4e12

tech-stack:
  added: []
  patterns:
    - "Return state travels as hidden form inputs plus one whitelisted server-side redirect (no HTMX, JavaScript or session storage); the POSTed copy is re-validated, never trusted"
    - "Tables name their own return-state page key from a class prefix constant, so a table built with request=None still posts dismissed-page / confirmed-page"
    - "Redirect fragment comes from a constant map keyed by the validated action, never from input"

key-files:
  created:
    - src/templates/campaigns/partials/attribution_return_state.html
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/campaign_tables.py
    - src/templates/campaigns/attribution_queue.html
    - solsys_code/tests/test_campaign_attribution_views.py
    - solsys_code/tests/test_attribution_template.py
    - solsys_code/tests/test_bootstrap5_rendering.py
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - .planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md

key-decisions:
  - "The two table prefixes live in campaign_tables.py (DISMISSED_TABLE_PREFIX, CONFIRMED_TABLE_PREFIX) and campaign_views imports them as _DISMISSED_TABLE_PREFIX / _CONFIRMED_TABLE_PREFIX, so the tables never import the view module and the key names cannot drift"
  - "A page past the end is not clamped in the redirect: the valid posted page is kept and django-tables2 renders the last page that has rows"
  - "Worklist actions add no fragment; only the two Undos do (RESEARCH A4)"

patterns-established:
  - "POST-then-follow attribution tests read the real rendered Undo form (HTMLParser scoped to the section) and post exactly its hidden fields"

requirements-completed: [WARN-07, WARN-04]

coverage:
  - id: D1
    description: "Undo on a Dismissed row redirects to ?dismissed-page=N#attribution-dismissed-section and the section renders open (collapse show, aria-expanded true), proven by posting the rendered form"
    requirement: WARN-07
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestAttributionReturnState.test_undo_on_a_dismissed_row_reopens_the_dismissed_section_on_the_same_page"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestAttributionReturnState.test_undo_from_the_dismissed_table_always_sends_its_page_even_on_a_plain_load"
        status: pass
    human_judgment: false
  - id: D2
    description: "A Confirmed Undo lands on the Confirmed fragment keeping its page, and Dismissed is open afterwards only if it was open before"
    requirement: WARN-07
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestAttributionReturnState.test_undo_on_a_confirmed_row_lands_on_the_confirmed_section_and_leaves_dismissed_as_it_was"
        status: pass
    human_judgment: false
  - id: D3
    description: "All five actions keep the five validated keys through one redirect; the worklist and bulk forms carry the state through one include; every decide form keeps its CSRF token"
    requirement: WARN-07
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestAttributionReturnState.test_every_action_keeps_band_page_and_sort"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_attribution_template.py#AttributionReturnStateFormTests"
        status: pass
    human_judgment: false
  - id: D4
    description: "Last-page clamp (26 dismissals, page 2 of 2), the empty Dismissed section, tampered values and unknown keys dropped, key names equal to the tables' own field names"
    requirement: WARN-07
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestAttributionReturnState"
        status: pass
    human_judgment: false
  - id: D5
    description: "IN-15: one boolean opens Dismissed for its own page or sort key only; the Confirmed table's key or a malformed value leaves it folded"
    requirement: WARN-07
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_attribution_template.py#AttributionCollapseSectionRenderTests"
        status: pass
    human_judgment: false
  - id: D6
    description: "ROADMAP success criterion 5 in Chromium: 27 dismissals, Undo accepted on page 2, lands on page 2 of the open section with the other page-2 row listed"
    requirement: WARN-07
    verification:
      - kind: automated_ui
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestBootstrap5Rendering.test_undo_on_a_dismissed_row_keeps_the_section_open_on_the_same_page"
        status: pass
    human_judgment: false
  - id: D7
    description: "Paired docs: runbook attribution-pass text and campaign_lifecycle_demo.ipynb cells (marker classes, Undo redirect) executed once top to bottom on the scratch database; notebook guard green"
    requirement: WARN-04
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py"
        status: pass
    human_judgment: true
    rationale: "The runbook prose and the notebook narrative are documentation; tests pin the notebook's committed output and the runbook phrases, not whether the wording reads well"

duration: 11min
completed: 2026-10-09
status: complete
---

# Phase 40 Plan 04: Attribution return state, runbook and lifecycle notebook Summary

**Every attribution action keeps the band filter, pages and sorts through hidden form fields and one whitelisted redirect, so Undo on a Dismissed row lands on the same page of the open section (server-tested from the real rendered form, browser-tested with 27 dismissals), with the runbook and the re-executed lifecycle notebook documenting it**

## Performance

- **Duration:** about 11 min
- **Started:** 2026-10-10T00:09:53Z
- **Completed:** 2026-10-10T00:21Z
- **Tasks:** 3 (one tracer, two auto)
- **Files modified:** 10 (one created)

## Accomplishments

- `attribution_return_state()` in `campaign_views.py` is the single whitelist: `band` in the three literals, page keys `[1-9][0-9]{0,5}`, sort keys `-?[a-z_]{1,40}`, everything else dropped, returned in a fixed order. `AttributionDecisionView._redirect()` builds the URL from `reverse('campaigns:attribution')`, `urlencode` of that state read from the POST, and a fragment from a constant map keyed by the validated action (T-40-11).
- Both Undo forms post the validated state plus their own current page (1 when never paginated or `request=None`), so a Dismissed section opened by a click on a plain load still reopens. The worklist and bulk forms use one autoescaped include, `attribution_return_state.html`.
- `dismissed_section_open` replaces three copies of the `request.GET` test in the template (IN-15); it is computed from the validated state, so `?confirmed-page=1` or `?dismissed-page=abc` leaves Dismissed folded.
- The last-page case needs no clamping code: undoing the only dismissal on page 2 of 2 posts `dismissed-page=2`, and django-tables2 renders page 1 (the new last page) with the section open and status 200.
- Runbook "What a dismissal means" and "What the evidence columns mean" gained the kept-place, last-page landing, Confirmed-section landing and green-bar text; `campaign_lifecycle_demo.ipynb` gained a marker cell and an Undo cell and was re-executed once top to bottom (counts 1..N, no errors) as the phase's last notebook run.
- The 37.1 ledger now reads `fixed` for WR-13 to WR-18, IN-12 and IN-15 in both front matter and table.

## Task Commits

1. **Task 1 (tracer): Undo on a Dismissed row returns to the same page, section open** - `66c1285` (fix)
2. **Task 2: every other action keeps the state, clamp, tampering, IN-15 render tests** - `aeddf31` (fix)
3. **Task 3: browser test, runbook, lifecycle notebook** - `bfd8ec0` (docs)
4. **Ledger: WR-15, WR-18, IN-12, IN-15 fixed** - `bda3591` (docs, `.planning`-only, `SKIP=django-test`)

**Plan metadata:** the docs commit following this summary.

## TDD Gate Compliance

Both `tdd="true"` tasks ran RED before any implementation, but each task's RED tests were committed together with its GREEN code, not as a separate `test(40-04)` commit. The plan's own commit step prescribes one commit per task through the full `django-test` hook, and a commit holding only failing tests would fail that hook (CLAUDE.md reserves `SKIP=django-test` for a deliberately red work-in-progress commit). There is therefore no `test(40-04)` commit and no `feat(40-04)` commit (the code commits are `fix(40-04)`); this is flagged here as the plan requires.

- **Task 1 RED** (3 target tests, run by name before touching the view): `test_undo_on_a_dismissed_row_reopens_the_dismissed_section_on_the_same_page` failed on `'/campaigns/attribution/' != '/campaigns/attribution/?dismissed-page=1#attribution-dismissed-section'`; `test_undo_from_the_dismissed_table_always_sends_its_page_even_on_a_plain_load` failed on `('dismissed-page', '1') not found` among the form's hidden inputs; the Confirmed test failed (both subTests) on `('confirmed-page', '1') not found`. Semantic assessment: all three target tests executed and failed on the planned assertion for the planned reason (the form carried no page and the redirect was bare); no import, fixture or zero-test fault.
- **Task 2 RED:** `test_worklist_and_bulk_forms_carry_the_current_state` and `test_a_plain_load_adds_no_state_input_to_the_worklist_forms` failed (8 subTest failures: the forms emitted only the old single `band` input). The per-action, clamp, tamper, key-name and whitelist tests already passed after Task 1, which the plan predicted ("they pin the behavior"). Semantic assessment: the two failing tests failed on the planned assertion; the passing ones are regression pins, not an unexpected green.
- **GREEN:** the full attribution suites (90 tests after Task 1, 102 after Task 2) pass. **REFACTOR:** none.

## Deviations from Plan

### UI-SPEC-mandated additions (not in the plan text; user-adopted 2026-10-09)

**F1. Section divs clear the fixed navbar after a fragment landing**
- Added `style="scroll-margin-top: 5rem;"` to the `attribution-dismissed-section` and `attribution-confirmed-section` divs in `src/templates/campaigns/attribution_queue.html`. It changes no class the render tests pin and does not affect the Playwright visibility check.
- **Commit:** `66c1285`

**F3. Two inaccurate Undo-confirmation strings corrected**
- The Undo-confirmation `confirm()` prompt in `solsys_code/campaign_tables.py` now reads "Undo this confirmation? It will move this pair to the Dismissed section above." (was "below"), and the flash in `solsys_code/campaign_views.py` now reads "Confirmation undone — the pair is now in Dismissed." (was "back in the queue"). "Dismissal undone — back in the queue." is unchanged.
- The four existing assertions on the old flash in `test_campaign_attribution_views.py` were updated. The runbook and the lifecycle notebook never quoted either string, so neither needed a wording change.
- **Commit:** `66c1285`

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The tables name their page key from class prefix constants, not from `prefixed_page_field`**
- **Found during:** Task 1 GREEN (the plan's test `AttributionDismissedTable([], request=None).render_actions(...)` renders `name="dismissed-page" value="1"`)
- **Issue:** a table built without the view's `prefix=` has `prefixed_page_field == 'page'`, so using it would post `page`, not `dismissed-page`, for an unconfigured table.
- **Fix:** `DISMISSED_TABLE_PREFIX` / `CONFIRMED_TABLE_PREFIX` are defined in `campaign_tables.py`, each table class carries `return_state_prefix`, and `campaign_views.py` imports them as `_DISMISSED_TABLE_PREFIX` / `_CONFIRMED_TABLE_PREFIX` (the plan put the constants in the views module). The tables still never import the view module, and a test pins the whitelist's keys to the constructed tables' own field names.
- **Files modified:** `solsys_code/campaign_tables.py`, `solsys_code/campaign_views.py`
- **Commit:** `66c1285`

**2. [Rule 1 - Lint] `zip()` without `strict=`**
- **Found during:** Task 1 (`ruff` B905)
- **Fix:** `zip(..., strict=True)` in `attribution_return_state()` (both tuples have two entries).
- **Commit:** `66c1285`

### Other differences from the plan text

- Two render tests beyond the plan's list: `test_a_plain_load_adds_no_state_input_to_the_worklist_forms` (the plan's "plain GET carries no state input" half, as its own test) and `test_a_malformed_dismissed_page_does_not_open_the_section` (UI-SPEC's IN-15 consequence for a junk value).
- The runbook also got one sentence saying the Undo confirmation prompts are plain browser dialogs; cancelling one changes nothing.
- The lifecycle notebook was executed twice: the first run printed `pk=None` after deleting the demo night (cosmetic) and a runbook phrase had wrapped across a line; both were fixed and the notebook re-run, so the committed output is from one fresh top-to-bottom run.
- `src/fomo_db.sqlite3` showed a fresh modification time during the notebook run; that is the user's `run_unattended` cron job (confirmed with `lsof`). The notebook's own output reports `Resolved database: '/tmp/fomo-notebook-db-.../fomo_db.sqlite3' (fresh scratch database, migrated from empty)`.

**Total deviations:** 2 UI-SPEC additions, 2 auto-fixed (1 blocking, 1 lint). **Impact:** none on scope; the F1/F3 changes are additive and the constants placement keeps the plan's no-import-cycle intent.

## Issues Encountered

None open. The known-flaky `test_observatory_create_form_submits_to_observatory_url` did not fail in this run.

## Verification

- `python manage.py test --noinput` on `test_campaign_attribution_views`, `test_attribution_template`, `test_pre_executed_notebooks`, `test_notebook_helper`, `test_projector_demo_notebook`: 159 tests OK.
- `python manage.py test --noinput solsys_code.tests.test_bootstrap5_rendering --tag functional`: 13 tests OK (includes the new Chromium Undo test).
- Phase gate `python manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault --exclude-tag migration --parallel --noinput`: 2294 tests OK.
- Full local suite `python manage.py test solsys_code --exclude-tag=ephemeris_segfault --parallel --noinput`: 2325 tests OK (functional and migration tags included).
- The `django-test` pre-commit hook passed on all three code commits; the ledger commit used `SKIP=django-test` (`.planning`-only).

## Known Stubs

None.

## Threat Flags

None. The only new surface is the POSTed return state, which is covered by the plan's threat model (T-40-11 to T-40-15) and its tests.

## Next Phase Readiness

All four plans of phase 40 are complete; the phase is ready for verification (`/gsd-verify-work` / phase verifier) and, after it, `/gsd-secure-phase` for the T-40-11 redirect mitigation.

## Self-Check: PASSED

- Files found: `src/templates/campaigns/partials/attribution_return_state.html`, `solsys_code/campaign_views.py`, `solsys_code/campaign_tables.py`, `docs/runbooks/telescope_runs_calendar.rst`, `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`, the 37.1 ledger.
- Commits `66c1285`, `aeddf31`, `bfd8ec0`, `bda3591` are ancestors of HEAD on `issue37-telescope-runs-calendar`.
