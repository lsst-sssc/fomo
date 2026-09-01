---
phase: 28-operator-assisted-attribution
plan: 03
subsystem: backend
tags: [django, views, staff-workflow, campaign-coordination, security]

# Dependency graph
requires:
  - phase: 28-01
    provides: CalendarEventDismissal / ObservationRecordDismissal dismissal models and
      CalendarEventMeta.confirmed_by/confirmed_at audit fields, both written by this plan's
      confirm/dismiss/undo actions
  - phase: 28-02
    provides: campaign_attribution.py -- the matcher module this plan's GET context assembly
      and every write action's server-side re-validation (is_offered_candidate()) call into
provides:
  - AttributionQueueView (StaffRequiredMixin, TemplateView) -- GET context assembly for the
    two orphan worklists, band filter, and the Dismissed/Confirmed section rows; template
    itself arrives in 28-04
  - AttributionDecisionView (StaffRequiredMixin, View) -- the five POST actions (confirm,
    confirm_selected, dismiss, undo_confirmation, undo_dismissal), each re-validated
    server-side against campaign_attribution.is_offered_candidate() before writing
  - campaigns:attribution / campaigns:attribution_decide routes
  - CampaignListView.get_context_data()'s attribution_count banner clause (D-02), reading the
    same campaign_attribution.orphans_needing_attribution_count() the page itself uses
  - solsys_code/tests/test_campaign_attribution_views.py -- the shared fixture (AttributionViewTestBase)
    and six test classes 28-04 extends with the page-rendering table/template tests
affects: [28-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Atomic conditional .filter(...).update(...) for a field-SET confirm (event side,
      keyed on run__isnull=True), vs. get_or_create() + IntegrityError-catch inside its own
      transaction.atomic() savepoint for a row-CREATE confirm (record side) -- two distinct
      race-safe idioms per target model, never a shared one"
    - "Server-side re-validation via is_offered_candidate() before every write that creates
      an association -- never trust a rendered checkbox/button; the bulk path additionally
      re-derives sole-High-band status per pair, never trusting the client-side gate"
    - "Per-pair loop inside one transaction.atomic() for bulk confirm, never a single
      combined .filter(pk__in=...).update(run=X) (each checked pair may target a different run)"
    - "django.contrib.messages read via get_messages(response.wsgi_request) on an un-followed
      POST response, when the redirect target's template does not exist yet -- avoids
      TemplateDoesNotExist from self.client.post(..., follow=True)"

key-files:
  created:
    - solsys_code/tests/test_campaign_attribution_views.py
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/campaign_urls.py

key-decisions:
  - "AttributionDecisionView's five actions dispatch off one literal-tuple-validated `action`
    POST param on a single view, mirroring CampaignRunDecisionView's shape, rather than five
    separate URL routes -- the decide route is deliberately NOT <int:pk>/-prefixed, since an
    attribution action names a PAIR (orphan of one of two kinds, plus a run), and both
    identifiers travel in the POST body where is_offered_candidate() re-validates them together"
  - "Task 1 seeded a minimal AttributionDecisionView POST-only stub (returns
    HttpResponseBadRequest unconditionally) so campaigns:attribution_decide could resolve and
    the routes commit atomically ahead of Task 2's full dispatch logic -- matches the plan's
    own instruction to import both view classes into campaign_urls.py from Task 1"
  - "_dismissed_attribution_rows()/_confirmed_attribution_rows() merge two structurally
    different models (dismissal pair / confirmed pair) into one Python list, sorted and capped
    in Python rather than via a queryset .union() -- mirrors ApprovalQueueView's decided_qs
    'cannot reorder a query once a slice has been taken' materialize-first discipline, extended
    to a cross-model merge"
  - "is_drained and attribution_count are always computed unfiltered (ignoring the band GET
    param) -- D-15's 'done' signal is about the whole backlog draining, not the currently
    band-filtered view"

requirements-completed: [ATTRIB-02, ATTRIB-03, ATTRIB-04]

# Metrics
duration: ~65min
completed: 2026-08-01
---

# Phase 28 Plan 03: Attribution Write Path Summary

**AttributionDecisionView's five POST actions (confirm / confirm_selected / dismiss / undo_confirmation / undo_dismissal), each re-deriving eligibility server-side via the Phase 28-02 matcher before writing, plus AttributionQueueView's GET context assembly and the campaign-list banner count -- proven by 793 passing tests (full solsys_code suite, excluding the pre-existing test_views.TestEphemeris ASSIST segfault).**

## Performance

- **Duration:** ~65 min
- **Started:** 2026-08-01 (session start; first commit 09:47:56 PDT)
- **Completed:** 2026-08-01T10:25:08-07:00
- **Tasks:** 3
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- `AttributionQueueView(StaffRequiredMixin, TemplateView)`: assembles the two paginated,
  band-filterable orphan worklists (`event_groups`/`record_groups`), the capped/materialized
  Dismissed/Confirmed section rows, and the D-02/D-15 `attribution_count`/`unattributable_count`/
  `is_drained` context keys the 28-04 template will render.
- `AttributionDecisionView(StaffRequiredMixin, View)`: the five write actions. Every confirm
  path re-derives eligibility via `campaign_attribution.is_offered_candidate()` before writing
  -- a tampered POST naming a cross-campaign-boundary run is refused, never trusted from the
  rendered form. Event-side confirm/undo use the atomic conditional `.update()` idiom
  (`run__isnull=True`); record-side confirm uses `get_or_create()` inside its own savepoint,
  catching the `unique_campaign_run_observation_record` race. `confirm_selected` additionally
  re-checks `BAND_HIGH` and sole-High-candidate status per pair, looping the write inside one
  `transaction.atomic()` block rather than a combined queryset update.
- `campaigns:attribution` / `campaigns:attribution_decide` routes; `CampaignListView`'s banner
  now reads `campaign_attribution.orphans_needing_attribution_count()`, the same definition the
  page itself uses (D-02).
- `solsys_code/tests/test_campaign_attribution_views.py` (new, 638 lines): a shared fixture
  (`AttributionViewTestBase`, matching the criterion-5 real-shaped case from 28-02) and six test
  classes -- `TestAttributionStaffGating`, `TestConfirmUndo`, `TestDismissAndUndoDismissal`,
  `TestBulkConfirmGate`, `TestConcurrencyAndTampering`, `TestAttributionQueueViewContext` -- 29
  tests covering every write action, its race, its double-submit, its tampered-pair rejection,
  and the staff gate.

## Task Commits

Each task was committed atomically:

1. **Task 1: The two routes, the GET context assembly, and the campaign-list banner count** -
   `33b2314` (feat)
2. **Task 2: AttributionDecisionView -- confirm, bulk confirm, dismiss, and both undos** -
   `b905c03` (feat)
3. **Task 3: POST action, race, tampering and staff-gating tests** - `a4fe0fc` (test)

## Files Created/Modified

- `solsys_code/campaign_views.py` - `AttributionQueueView`, `AttributionDecisionView`,
  `_dismissed_attribution_rows()`/`_confirmed_attribution_rows()`, `_is_sole_high_candidate()`,
  `CampaignListView.get_context_data()`'s new `attribution_count` clause
- `solsys_code/campaign_urls.py` - `attribution/` and `attribution/decide/` routes
- `solsys_code/tests/test_campaign_attribution_views.py` (new) - shared fixture + 6 test
  classes, 29 tests

## Decisions Made

See `key-decisions` in frontmatter above -- the dispatching-view shape, the Task 1 stub
sequencing, the cross-model row-merge discipline, and the always-unfiltered drained/count
computation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Restored a gitignored build artifact so Django settings could import**
- **Found during:** Task 1, before `python manage.py check` could run
- **Issue:** `src/fomo/_version.py` (setuptools_scm's `write_to` target, gitignored, not
  committed) did not exist in this worktree, so importing `src.fomo.settings` raised
  `ModuleNotFoundError: No module named 'src.fomo._version'`. Identical to 28-01/28-02's
  documented deviation.
- **Fix:** Copied the existing generated file from the main checkout into this worktree at the
  same path. Not a tracked-file change (gitignored in both locations) and not committed.
- **Files modified:** none tracked (local build artifact only)

**2. [Rule 1 - Plan self-consistency] Reworded docstring/comment prose colliding with this
plan's own literal-substring acceptance-criteria greps**
- **Found during:** Tasks 1 and 2, running the plan's own acceptance-criteria greps
- **Issue:** Several of the plan's own explanatory-comment instructions produced prose that,
  read literally by the plan's own `grep -c "<exact substring>"` acceptance checks, inflated
  the count above the target: (a) Task 1's `CampaignListView.get_context_data()` docstring
  spelled out `orphans_needing_attribution_count()` in a sentence, pushing that grep from the
  target 2 to 3; (b) Task 2's `AttributionDecisionView`/`_confirm_selected()`/`_dismiss()`
  docstrings spelled out `http_method_names = ['post']`, `pk__in=`, and `mark_safe`
  respectively, each colliding with its own acceptance grep.
- **Fix:** Reworded each passage to preserve the exact same documented meaning without the
  literal substring (e.g. "one shared attribution-backlog-count function" instead of naming
  the function inline with its call syntax; "declared POST-only below" instead of restating the
  attribute assignment; "a single combined queryset update naming every checked pk at once"
  instead of `pk__in=`; "never bypasses Django's template auto-escaping" instead of
  `mark_safe`). Three of the four (`orphans_needing_attribution_count()`, `pk__in=`,
  `mark_safe`) now match their target counts exactly (2, 0, 0).
- **Files modified:** `solsys_code/campaign_views.py`
- **Verification:** re-ran each affected grep after the reword; `python manage.py check` and
  `ruff check .`/`ruff format --check .` stayed clean throughout.
- **Committed in:** `33b2314` (Task 1), `b905c03` (Task 2)

**3. [Not fully resolvable -- documented, not silently accepted] `http_method_names = ['post']`
grep count stays 3, not the plan's literal target of 2**
- **Found during:** Task 2, re-running the plan's own acceptance criteria after the reword above
- **Issue:** `CampaignRunDecisionView`'s pre-existing docstring (unrelated to this plan, present
  since Phase 16) already restates `` ``http_method_names = ['post']`` `` in prose alongside its
  own real code declaration -- so the UNMODIFIED baseline file already contains 2 matches from
  that ONE view alone, before this plan adds anything. `AttributionDecisionView`'s own required
  `http_method_names = ['post']` class attribute (a functional necessity, not removable) is a
  third, unavoidable match. The acceptance criterion's "returns 2 (the pre-existing decision
  view plus this one)" appears to assume one match per view, which the codebase's own existing
  docstring convention already falsifies for the pre-existing view alone.
- **Fix:** Not resolvable without either deleting the required `http_method_names = ['post']`
  attribute (breaks the POST-only security control this exact criterion exists to protect) or
  editing `CampaignRunDecisionView`'s pre-existing, out-of-scope docstring. Left the count at 3
  and documented the reasoning here rather than silently claiming the literal target was met.
- **Files modified:** none (no further change applied)
- **Verification:** `grep -c "http_method_names = \['post'\]" solsys_code/campaign_views.py`
  returns 3; the underlying security property (both views declare `http_method_names = ['post']`
  as a real class attribute, verified functionally by `TestAttributionStaffGating.
  test_staff_get_decide_returns_405`) holds regardless of the grep count.

**4. [Rule 1 - Test-authoring pitfall] `cls.campaign_run` chosen over `cls.run` from the start,
then a second, distinct test-authoring bug found and fixed before any commit**
- **Found during:** Task 3, first full run of the new test module
- **Issue A:** The fixture was initially written with `cls.run = CampaignRun.objects.create(...)`
  (matching the plan's own prose, which calls the fixture field "one `CampaignRun` shaped like
  the real pk=1 row" without naming the attribute) -- this is the exact `unittest.TestCase.run()`
  collision 28-02-SUMMARY.md already documented and fixed once; it crashed the test runner with
  `TypeError: 'CampaignRun' object is not callable` and no per-test traceback.
  **Issue B:** Independently, several tests initially used `self.client.post(..., follow=True)`
  to inspect `response.context['messages']` after a confirm/dismiss/undo POST -- since the
  redirect target (`campaigns:attribution`) has no template until Plan 28-04, following it
  raised `django.template.exceptions.TemplateDoesNotExist: campaigns/attribution_queue.html`.
- **Fix A:** Renamed the attribute to `cls.campaign_run` throughout the file (`sed` project-wide
  within this one file), matching 28-02's precedent exactly.
  **Fix B:** Replaced every `follow=True` + `response.context['messages']` pattern with a shared
  `_message_strings(response)` helper reading `django.contrib.messages.get_messages(response.
  wsgi_request)` directly off the un-followed 302 response, and asserted `status_code == 302`
  instead of `200`.
- **Files modified:** `solsys_code/tests/test_campaign_attribution_views.py`
- **Verification:** `python manage.py test solsys_code.tests.test_campaign_attribution_views` --
  all 29 tests pass; the full `solsys_code` suite (793 tests) also passes.

---

**Total deviations:** 4 (1 blocking/environment with no tracked-file impact, 2 plan
self-consistency reword fixes -- one fully resolved, one left partially unresolved and
documented rather than silently claimed, 1 test-authoring-only fix covering two distinct
pitfalls).
**Impact on plan:** None of these touch the actual write-path behaviour, its security
guarantees, or its public contract (routes, action names, message copy all match the plan and
UI-SPEC exactly). Deviation 3 is a plan-acceptance-criteria imprecision, not a code defect --
flagged explicitly rather than glossed over.

## Issues Encountered

None beyond the deviations above. `ruff check .`/`ruff format --check .` at the whole-project
level report the same 4 pre-existing, unrelated issues 28-01/28-02-SUMMARY.md already documented
(a docstring gap in a demo notebook, formatting drift in `src/fomo/settings.py`, and two files
under `.planning/quick/260619-f7u.../`) -- confirmed untouched by any file this plan modifies;
`ruff check`/`ruff format --check` scoped to this plan's own three files both pass cleanly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `AttributionQueueView`'s context contract (`event_groups`, `record_groups`, `band`,
  `dismissed_rows`, `confirmed_rows`, `attribution_count`, `unattributable_count`,
  `is_drained`) is fixed and tested -- 28-04 builds `attribution_queue.html` and the
  `django-tables2` table classes directly against it, with no further view-layer changes needed.
- `AttributionDecisionView`'s five actions, their exact message copy (matching UI-SPEC's
  Copywriting Contract verbatim, including both em-dash undo strings), and their POST-parameter
  contract (`action`, `kind`, `orphan_pk`, `run_pk`, `reason`, `candidate_ids`) are all fixed and
  tested -- 28-04's forms/buttons need only target these exact names.
- `test_campaign_attribution_views.py`'s shared fixture (`AttributionViewTestBase`,
  `_make_event()`/`_make_record()`, `_message_strings()`) and the six existing test classes are
  ready for 28-04 to extend with `TestEvidenceColumns`, `TestQueueDrainsToEmpty`, and
  `TestBandFilterAndBanner`, per its own plan text.
- No blockers. The one open item (deviation 3, the `http_method_names` grep count) is
  documented above and does not affect functionality, security, or any downstream plan.

---
*Phase: 28-operator-assisted-attribution*
*Completed: 2026-08-01*

## Self-Check: PASSED

- FOUND: `solsys_code/campaign_views.py`
- FOUND: `solsys_code/campaign_urls.py`
- FOUND: `solsys_code/tests/test_campaign_attribution_views.py`
- FOUND: `.planning/phases/28-operator-assisted-attribution/28-03-SUMMARY.md`
- FOUND commit: `33b2314` (Task 1)
- FOUND commit: `b905c03` (Task 2)
- FOUND commit: `a4fe0fc` (Task 3)
