---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
plan: 04
subsystem: api
tags: [django, htmx, live-search, gap-closure]

# Dependency graph
requires:
  - phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
    provides: SiteSearchView, build_site_candidates(), substring_or_fuzzy_match_candidates() (plan 22-01)
provides:
  - "SiteSearchView.get() resolves its search term from q, then site_raw, then site_selection"
  - "Regression tests proving the widgets' own name-keyed GET params (site_raw, site_selection) surface suggestions"
affects: [22-site-matching-at-submission-and-unmatched-site-resolution-wo]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GET param fallback chain (q -> site_raw -> site_selection) preferring first non-empty value, so a single view-side change covers multiple htmx hx-get widgets sharing one endpoint without any template/markup edits"

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_site_search.py

key-decisions:
  - "Fixed the shared root cause (view reads only `q`, but htmx hx-get sends the triggering input's own name-keyed param) at the single SiteSearchView.get() call site rather than touching either widget's markup, per the plan's explicit view-only scope."
  - "Used `or`-chained request.GET.get() lookups (q, then site_raw, then site_selection) so an explicitly-empty `q` still falls through, matching 'prefer the first non-empty value' from the plan."

patterns-established: []

requirements-completed: [D-03, D-09, D-10]

coverage:
  - id: D1
    description: "Public 'Submit an Observing Run' form's site_raw live-search widget now renders suggestions (restores D-09)"
    requirement: D-09
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_site_search.py#test_site_raw_param_without_q_returns_suggestions"
        status: pass
    human_judgment: false
  - id: D2
    description: "Approval-queue / Sites Needing Review site_selection live-search widgets now render suggestions (restores D-10)"
    requirement: D-10
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_site_search.py#test_site_selection_param_without_q_returns_suggestions"
        status: pass
    human_judgment: false
  - id: D3
    description: "Existing `?q=<term>` callers unaffected; q still takes precedence when both q and a widget param are present"
    requirement: D-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_site_search.py#test_q_takes_precedence_over_site_raw"
        status: pass
      - kind: unit
        ref: "solsys_code.tests.test_campaign_site_search (full module, 21 tests)"
        status: pass
    human_judgment: false

# Metrics
duration: 20min
completed: 2026-07-15
status: complete
---

# Phase 22 Plan 04: SiteSearchView widget-param-name fix Summary

**SiteSearchView.get() now resolves its search term from `q`, then `site_raw`, then `site_selection` — a single view-side fallback chain that restores live-search rendering on the public submission form and both approval-queue widgets, with zero widget/template changes.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-07-15
- **Completed:** 2026-07-15
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments
- Fixed the shared root cause behind UAT gaps 1 and 3: `SiteSearchView.get()` only ever read `request.GET.get('q', '')`, but htmx's `hx-get` serializes only the triggering input's own `name`-keyed value (never an enclosing form's other fields for GET), so the public form's `site_raw` widget and both approval-queue `site_selection` widgets never sent `q` and always tripped the empty-fragment gate.
- Added a fallback chain — `q` or `site_raw` or `site_selection`, preferring the first non-empty value — so `q` still wins whenever explicitly supplied, preserving every existing caller/test unchanged.
- Added 4 new regression tests exercising the exact widget-side request shapes (`site_raw` alone, `site_selection` + `input_id` alone, `q`-precedence over `site_raw`, and a 1-char `site_raw` gate test), all passing alongside the pre-existing 17 tests in the module (21 total).
- Confirmed no regressions across the full `solsys_code` Django test suite (475 tests, up from 471 — the 4 new tests, all green).

## Task Commits

Each task was committed atomically:

1. **Task 1: Resolve the SiteSearchView search term from the widgets' own param names** - `dba220d` (fix)
2. **Task 2: Regression tests for the widget param keys and q-precedence** - `b2ec811` (test)

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified
- `solsys_code/campaign_views.py` - `SiteSearchView.get()` now resolves `query` from `q` → `site_raw` → `site_selection` instead of `q` alone; throttle, `input_id` allowlist, min-length gate, matcher call, and copy contract are untouched and evaluated against the resolved term.
- `solsys_code/tests/test_campaign_site_search.py` - Added 4 regression tests: `test_site_raw_param_without_q_returns_suggestions`, `test_site_selection_param_without_q_returns_suggestions`, `test_q_takes_precedence_over_site_raw`, `test_one_char_site_raw_returns_empty_fragment_without_building_pool`.

## Decisions Made
- Kept the fix entirely in `SiteSearchView.get()` — no widget/template edits in `campaign_forms.py` or `campaign_tables.py` — per the plan's explicit scope, so every existing markup/escapejs/hx-trigger-grammar/throttle test stays green untouched.
- Chose `or`-chained `.get()` calls (rather than checking each param's presence key) so a request that includes `q=''` explicitly still falls through to `site_raw`/`site_selection`, matching "prefer the first non-empty value."

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. `ruff format` auto-collapsed the fallback-chain assignment onto a single line (under the 120-column limit) after the initial edit; re-ran `ruff format` and both `ruff check` / `ruff format --check` pass clean on both touched files.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- UAT gaps 1 and 3 (of the 3 found in Phase 22 human UAT) are now closed. Gap 2 is handled by a separate gap-closure plan (22-05 or 22-06, per the orchestrator's plan split) and is out of scope here.
- Full `solsys_code` suite (475 tests) and `ruff check`/`ruff format --check` on touched files both clean — no known blockers for the orchestrator to re-verify D-09/D-10 via UAT.

---
*Phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo*
*Completed: 2026-07-15*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_views.py
- FOUND: solsys_code/tests/test_campaign_site_search.py
- FOUND: dba220d (Task 1 commit)
- FOUND: b2ec811 (Task 2 commit)
