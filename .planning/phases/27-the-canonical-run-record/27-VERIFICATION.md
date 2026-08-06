---
phase: 27-the-canonical-run-record
verified: 2026-08-06T18:38:45Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 5/5
  gaps_closed:
    - "27-UAT.md Test 8: 'Sites Needing Review — action required' rendered as the THIRD section on /campaigns/approval-queue/, below Pending Review and Recently Decided, even though it is the only actionable table when pending_count is 0 — closed by 27-07 Task 1 (card moved to the top of {% block content %})"
    - "27-UAT.md Test 9: the calendar-event modal rendered nothing for an unlinked event (CalendarEventMeta.run unset), even when a HIGH-band attribution-queue candidate already existed for it, giving an operator no clue why it was unlinked or what to do — closed by 27-07 Task 2 (staff-only 'Possible campaign run match' hint + attribution_display_extras.py)"
  gaps_remaining: []
  regressions: []
---

# Phase 27: The Canonical Run Record Verification Report

**Phase Goal:** Make `CampaignRun` canonical in the schema — it records how it was created, distinguishes a class-wide allocation from an unresolved site, owns the calendar events that show it, and owns the observation records that realise it — with every existing row and all four companion-record consumers surviving the change.
**Verified:** 2026-08-06T18:38:45Z
**Status:** passed
**Re-verification:** Yes — this run supersedes the 2026-07-30 initial verification. Since then: (a) Phase 27.1 (inserted) closed the first 3 gaps found by 27-UAT.md's re-verification round; (b) plan 27-07 (Wave 6, commits `52610be`/`ec57e2c`/`5899895`/`b6cfae6`/`2f87f73`) closed the final 2 gaps (27-UAT.md Test 8 and Test 9). This phase is now 7/7 plans complete, and 27-UAT.md's frontmatter and all 5 of its `Gaps` entries are marked `resolved`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every `CampaignRun` records which ingest path created it; a non-web run is never left in the review queue; non-staff visibility unchanged for old and new rows | VERIFIED | Unchanged from initial verification: `models.py:82-102` `Source` TextChoices, `import_campaign_csv.py:195-200` sets `APPROVED`+`CSV_IMPORT` directly, `campaign_views.py:207,237-258` web submissions default `PENDING_REVIEW`+`WEB`. Additionally now confirmed: the "Sites Needing Review" queue (the operational surface staff use to actually process non-web/site-unresolved rows) renders as the FIRST section on `/campaigns/approval-queue/` (`approval_queue.html:8-17`), ahead of Pending Review and Recently Decided — so a class-wide or site-needing-review run is never buried below an informational table when it's the only actionable item (27-UAT.md Test 8, closed by 27-07 Task 1). |
| 2 | A class-wide telescope allocation is distinguishable from an unresolved site, and the two coexist without colliding | VERIFIED | Unchanged from initial verification: `models.py:104-130` `TelescopeClass` TextChoices, `derive_telescope_class()` gated on `site is None`, both `UniqueConstraint`s untouched. Dev DB: 3 rows carry a non-blank `telescope_class` independent of `site`. |
| 3 | Companion rows survive the rename with `is_verified` intact: dashed-border fallback, LCO sync writes labels, admin registers the model, calendar page loads labels via one prefetch | VERIFIED | Unchanged from initial verification: `models.py:10-49` `CalendarEventMeta` via `RenameModel` migration `0008`; `calendar.html:228,244`, `sync_lco_observation_calendar.py:369`, `admin.py:99-102,112`, `views.py:114`'s single `.prefetch_related('telescope_label_meta')` all confirmed present, byte-unchanged by 27-07 (git diff for plan 07 touches only `approval_queue.html`, `event_form.html`, one new templatetag module, two test files, and the runbook — none of these truth-3 artifacts). |
| 4 | A calendar event can link to its run; an ObservationRecord can link to the run it realises with confirmation metadata; deleting a run never deletes calendar events, companion rows, or observation records | VERIFIED | Unchanged from initial verification: `models.py:39-46` `CalendarEventMeta.run` `SET_NULL`; `models.py:272-333` `CampaignRunObservation` with `confirmed_by`/`confirmed_at`; `test_campaign_run_observation.py` (7 tests, still passing in this run's regression sweep) asserts both cascade directions. |
| 5 | A staff user can see a run's linked calendar events and observation records, and get from an event back to its run | VERIFIED | Unchanged base wiring: `admin.py:8-31,57,64-96` inlines + `save_formset` stamping; `event_form.html:117-152` renders the "Campaign run" link block gated on `run.is_publicly_visible`. Newly strengthened: for an event that is NOT yet linked (`CalendarEventMeta.run` unset or no companion row), `event_form.html:153-181`'s new `{% elif not run and request.user.is_staff %}` branch calls `attribution_display_extras.high_band_attribution_candidates` and renders a "Possible campaign run match" hint naming the candidate run (`{{ candidate.run }}`, includes pk/campaign/telescope/window/site) and a link to `campaigns:attribution?band=high` — so a staff user is never left with a silently empty modal when a good match already exists (27-UAT.md Test 9, closed by 27-07 Task 2). Confirmed staff-only (`request.user.is_staff`, server-derived) and confirmed the branch does not fire when `run` is truthy (already-linked events still show the plain Campaign-run block, not the hint) or when there are zero candidates. |

**Score:** 5/5 truths verified

### UAT Gap Closure Verification (27-UAT.md Test 8 and Test 9)

Both gaps closed by plan 27-07 were verified directly against the running code, not by trusting SUMMARY.md's narrative:

| UAT Test | Claimed fix | Verified in codebase | Verified by test |
|----------|-------------|------------------------|-------------------|
| Test 8 — Sites Needing Review buried below two other sections | Card moved to top of `{% block content %}` | `src/templates/campaigns/approval_queue.html:5-24` read directly: "Sites Needing Review — action required" card (lines 8-17) now precedes the "Pending Review" `<h5>` (line 19) and "Recently Decided" `<h5>` (line 22); `mt-4` correctly swapped for `mb-4` | `TestApprovalQueueSitesNeedingReviewGrouping.test_sites_needing_review_now_precedes_pending_and_decided` (asserts `review_index < pending_index` and `review_index < decided_index` against real rendered HTML) — PASS |
| Test 9 — unlinked event with a HIGH-band candidate shows no hint | New `attribution_display_extras.high_band_attribution_candidates` tag + `event_form.html` `{% elif not run and request.user.is_staff %}` branch | `solsys_code/templatetags/attribution_display_extras.py:22-39` read directly (thin filter over `campaign_attribution.candidates_for_event()` to `BAND_HIGH`); `event_form.html:153-181` read directly — branch fires only when `not run` (both "no companion row" and "companion row, run unset") AND `request.user.is_staff`; renders `<label>Possible campaign run match</label>`, one line per candidate with `{{ candidate.run }}` + score badge, and a link to `{% url 'campaigns:attribution' %}?band=high` | `EventModalAttributionHintTest`'s 5 methods, all present and passing: `test_staff_sees_high_band_hint_for_unlinked_event`, `test_anonymous_does_not_see_hint`, `test_no_candidate_event_shows_no_hint`, `test_linked_event_shows_run_block_not_hint`, `test_stale_wr03_comment_removed_from_template_source` — PASS |

Both UAT debug sessions are correctly closed out: `.planning/debug/resolved/approval-queue-section-order.md` and `.planning/debug/resolved/calendar-event-run-link-inconsistent.md` both carry `status: resolved` and are in the `resolved/` directory. `27-UAT.md`'s `Gaps` section shows all 5 entries (including the two from this round) marked `status: resolved`.

Live test run performed by this verifier (not taken from SUMMARY claims):

```
python manage.py test solsys_code.tests.test_campaign_approval.TestApprovalQueueSitesNeedingReviewGrouping solsys_code.tests.test_calendar_template -v 1
→ Ran 42 tests in 6.312s — OK
```

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/models.py` | `CalendarEventMeta`, `CampaignRun.Source`/`TelescopeClass`/`source`/`telescope_class`/`is_publicly_visible`, `CampaignRunObservation` | VERIFIED | Unchanged by 27-07; re-confirmed present |
| `solsys_code/calendar_utils.py` | `derive_telescope_class()` etc. | VERIFIED | Unchanged by 27-07 |
| `solsys_code/admin.py` | Two inlines, `save_formset`, list_display/list_filter additions | VERIFIED | Unchanged by 27-07 |
| `solsys_code/management/commands/repair_stale_campaign_run_sites.py` | One-time repair command | VERIFIED | Unchanged by 27-07 |
| `solsys_code/migrations/0008,0009,0010,0011` | RenameModel, AddField, CreateModel+AddField, RunPython backfill | VERIFIED | Unchanged by 27-07 |
| `src/templates/campaigns/approval_queue.html` | Sites Needing Review card first | VERIFIED (NEW, 27-07) | Read directly: card at lines 8-17, before Pending Review (19) and Recently Decided (22); `mt-4`→`mb-4` |
| `solsys_code/templatetags/attribution_display_extras.py` | `high_band_attribution_candidates` simple_tag | VERIFIED (NEW, 27-07) | Read directly: single `@register.simple_tag` function, thin filter over `campaign_attribution.candidates_for_event()` to `BAND_HIGH`, never raises |
| `src/templates/tom_calendar/partials/event_form.html` | Event→run link + staff-only candidate hint, refreshed WR-03 comment | VERIFIED (UPDATED, 27-07) | Read directly; `{% load bootstrap4 attribution_display_extras %}`; WR-03 comment (lines 104-115) no longer claims "no production code writes CalendarEventMeta.run yet" — now documents the Phase 29 reconciler's automatic write and the new hint's purpose |
| `solsys_code/tests/test_calendar_template.py` | `EventModalAttributionHintTest` (5 tests) | VERIFIED (NEW, 27-07) | Read directly at lines 567-673; all 5 tests present, matching plan spec exactly; PASS |
| `solsys_code/tests/test_campaign_approval.py` | Renamed/inverted order-lock test | VERIFIED (UPDATED, 27-07) | `TestApprovalQueueSitesNeedingReviewGrouping.test_sites_needing_review_now_precedes_pending_and_decided` present at line 1675, docstring updated (lines 1654-1663) to cite the 27-07 gap closure instead of claiming the old order is "preserved"; PASS |
| `docs/runbooks/telescope_runs_calendar.rst` | Updated prose for both behaviors | VERIFIED (UPDATED, 27-07) | Grep-confirmed: "Sites Needing Review now renders first on the page" (line 162-164) and a new "27-UAT.md Test 9 gap closure" paragraph (line 630) documenting the "Possible campaign run match" hint |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | Regenerated with source/telescope_class in real output | VERIFIED | Unchanged by 27-07 (not in its `files_modified`) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `approval_queue.html` | rendered HTML order | static block order | WIRED | Card index precedes both other section indices — confirmed by direct file read and by `test_sites_needing_review_now_precedes_pending_and_decided` |
| `event_form.html` | `attribution_display_extras.py` | `{% load attribution_display_extras %}` + `{% high_band_attribution_candidates event as ... %}` | WIRED | Confirmed load tag at line 19, call at line 164 |
| `event_form.html` | `campaigns:attribution` | hint link `href="...?band=high"` | WIRED | Confirmed at line 176; `?band` already validated server-side by pre-existing `AttributionQueueView.get_context_data` (falls back to "all bands" on any unrecognised value — no new server code) |
| `attribution_display_extras.py` | `campaign_attribution.candidates_for_event()` | direct function call, filtered to `BAND_HIGH` | WIRED | Confirmed at line 39; delegates entirely, never raises independently |
| All Wave 1-5 links (`sync_lco_observation_calendar.py`→`calendar_utils.py`, `views.py`→`models.py` prefetch, `calendar.html`→`models.py`, `migrations/0011`→`calendar_utils.py`, `import_campaign_csv.py`→`calendar_utils.py`, `admin.py`→`models.py`, `campaign_views.py`→`models.py`) | — | — | WIRED (unchanged) | Not touched by 27-07's diff; carried forward from initial verification, confirmed still present by grep during this pass |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| `event_form.html` hint block | `attribution_candidates` | `campaign_attribution.candidates_for_event(event)` filtered to `BAND_HIGH` | Yes — confirmed live in the debug session for the real dev-DB event pk=59: HIGH-band, score 0.8, against `CampaignRun` pk=1 (cited in `.planning/debug/resolved/calendar-event-run-link-inconsistent.md`); mechanism additionally proven by 5 passing tests using realistic fixtures (matching instrument strings, overlapping dates) | FLOWING |
| `approval_queue.html` `review_table` | `review_table` context var | `ApprovalQueueView.get_context_data` (unchanged by 27-07 — confirmed no Python file in 27-07's diff) | Yes — same queryset as initial verification, now just rendered earlier in the page | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 27-07's two new/updated test surfaces pass | `python manage.py test solsys_code.tests.test_campaign_approval.TestApprovalQueueSitesNeedingReviewGrouping solsys_code.tests.test_calendar_template -v 1` | Ran 42 tests in 6.312s — OK | PASS |
| Full targeted phase-27 regression sweep (all modules from initial verification, re-run by this verifier) | `python manage.py test solsys_code.tests.test_canonical_record_migration solsys_code.tests.test_campaign_run_observation solsys_code.tests.test_repair_stale_campaign_run_sites solsys_code.tests.test_admin solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_utils solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_submission solsys_code.tests.test_campaign_approval solsys_code.solsys_code_observatory.tests.test_timezone_backfill_migration solsys_code.tests.test_import_campaign_csv -v 0` | Ran 420 tests in 51.113s — OK | PASS (no regressions from 27-07) |
| Repo-wide lint clean (excluding pre-existing, documented-as-deferred findings) | `ruff check .` | 1 finding: `D103` in `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`, confirmed via `deferred-items.md` and `git diff` to be untouched by 27-07 | PASS (with documented pre-existing exception, not introduced by this phase) |
| Sphinx doc build succeeds (proves runbook RST edits are valid) | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` | "build succeeded, 10 warnings" — no errors | PASS |

### Probe Execution

Not applicable — this phase has no `scripts/*/tests/probe-*.sh` convention; verification relies on the Django test suite and direct dev-DB/code inspection, consistent with the initial verification.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|--------------|--------|----------|
| CANON-01 | 27-04, 27-05, 27-06 | `source` field, WEB-only approval gating | SATISFIED | Unchanged; `models.py:82-102,195-200`; `campaign_views.py:257`; `import_campaign_csv.py:195-200` |
| CANON-02 | 27-01, 27-02, 27-04, 27-06, 27-07 | `telescope_class` field, distinguishable from unresolved site | SATISFIED | Schema/backfill unchanged; 27-07 strengthens the operational surface staff use to act on site-review rows (approval-queue reordering) |
| CANON-03 | 27-03 | `CalendarEventMeta` rename + `run` link, 4 integration points preserved | SATISFIED | Unchanged; `models.py:10-49`; migrations `0008`/`0009`; all 4 consumers confirmed unbroken |
| CANON-04 | 27-04 | `CampaignRunObservation` link model with confirmation metadata | SATISFIED | Unchanged; `models.py:272-333`; `test_campaign_run_observation.py` (7 tests) |
| CANON-05 | 27-05, 27-07 | Staff sees/edits run's linked events and observations; event links back to run | SATISFIED | Base wiring unchanged (`admin.py`, `event_form.html` run block); 27-07 adds the staff-only attribution-candidate hint closing the "silently empty modal" gap |

REQUIREMENTS.md (lines 23-27, 100-104) marks all five CANON IDs `[x]` complete and maps them all to Phase 27. `27-07-PLAN.md`'s `requirements: [CANON-02, CANON-05]` frontmatter is consistent with REQUIREMENTS.md — no orphaned or newly-introduced requirement IDs.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/templatetags/attribution_display_extras.py` / `event_form.html:164` | — | New per-render DB queries for the attribution hint have no test asserting a query cap (27-REVIEW.md IN-01) | INFO | Single-event modal render, not a list-page N+1; explicitly out of v1 scope per the code review. Not a blocker. |
| `event_form.html:164-180` (calls pre-existing `campaign_attribution.py:474-487`) | — | `_eligible_runs_for_event` does not filter by `approval_status`, so a REJECTED or PENDING_REVIEW run could appear in the "Possible campaign run match" hint (27-REVIEW.md IN-02) | INFO | Pre-existing behavior of `campaign_attribution.py`, not modified by 27-07; this delta newly surfaces it to a staff-only modal where previously nothing showed. Explicitly marked "no action required for this delta" by the code review. |
| `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` | cell 6 | `D103` missing docstring (ruff) | INFO | Pre-existing, confirmed untouched by 27-07 via `git diff`; logged in `deferred-items.md`, not introduced by this phase |
| carried from initial verification (still present, unaffected by 27-07) | `event_form.html` | Anti-pattern table rows previously flagged as WR-04 (TBD-run "None–None" rendering) is now guarded — `event_form.html:146` gates on `{% if run.window_start %}`, so this is RESOLVED, not carried forward as a gap | — | Confirmed fixed: the visible-run block at line 146 now conditionally renders the window dates, matching the WR-04 comment at lines 140-145 |
| `solsys_code/management/commands/import_campaign_csv.py` | 177-207 | Every re-import recomputes `site`/`telescope_class` from the CSV cell unconditionally (27-REVIEW.md WR-01) | WARNING (carried, unchanged by 27-07 — out of this plan's scope) | Same as initial verification: a real, demonstrated operational risk, does not block any of the 5 goal-backward truths |
| `solsys_code/admin.py` | 62 | `source` freely editable in admin (27-REVIEW.md WR-03, superseded numbering — since widened by Phase 27.1-05 per ROADMAP.md line 111) | INFO (largely addressed by Phase 27.1, not re-litigated here) | Not part of 27-07's scope; ROADMAP.md records Phase 27.1 widened the `source` provenance lock to every `WEB` run at any approval status |

No `TBD`/`FIXME`/`XXX` debt markers found in any of 27-07's modified files — the few "TBD" occurrences in `test_calendar_template.py`, `test_campaign_approval.py`, `event_form.html`, and the runbook are all the domain term "TBD window" (a `CampaignRun` with no resolved `window_start`/`window_end`), not code-debt markers.

### Human Verification Required

None. All 5 observable truths, plus both 27-07 gap closures, are verifiable programmatically against the codebase, the dev-DB-derived debug sessions, and the passing automated test suite. The plan's own `<verification>` step 4 lists an *optional* manual smoke test, but automation already closes both gaps per the plan's own note ("automation above is sufficient to close both gaps"), and this verifier independently confirmed the same code paths and tests directly.

### Gaps Summary

No blocking gaps. All 5 of Phase 27's roadmap Success Criteria are verified true against the current codebase, and both minor UAT gaps discovered during human re-verification (Test 8: approval-queue section order; Test 9: unlinked-event modal gives no hint about an available HIGH-band attribution match) are now closed and independently re-verified by this run — not merely trusted from 27-07-SUMMARY.md's claims. The reordered `approval_queue.html`, the new `attribution_display_extras.py` template tag, and the updated `event_form.html` hint block were all read directly; the 5 new `EventModalAttributionHintTest` methods and the inverted order-lock test were confirmed present with the exact assertions the plan specified, and a live test run (42/42, then a 420-test regression sweep) confirms no regression. Both referenced debug sessions are correctly marked `resolved` and moved to `.planning/debug/resolved/`, and `27-UAT.md`'s frontmatter and all 5 `Gaps` entries are marked `resolved`.

Three INFO-level and one WARNING-level anti-pattern remain from the code review, none of which block any of the 5 goal-backward truths: (1) no query-count test on the new attribution hint (low risk, single-event render); (2) the hint can theoretically surface a REJECTED/PENDING run as a candidate (pre-existing scorer behavior, not a 27-07 regression, explicitly accepted by the code review); (3) one pre-existing, documented-as-deferred ruff `D103` finding unrelated to this phase; (4) `import_campaign_csv.py`'s re-import-can-revert-a-repair risk (WR-01, unchanged, out of 27-07's scope). The previously tracked WR-04 ("None–None" TBD-run rendering) is confirmed fixed by the `{% if run.window_start %}` guard already present in `event_form.html`.

Phase 27 is complete: 7/7 plans executed, all 5 CANON requirements satisfied, all 5 of 27-UAT.md's gaps resolved, no regressions in a 420-test targeted sweep, ruff and Sphinx clean (net of one pre-existing, unrelated, documented exception).

---

_Verified: 2026-08-06T18:38:45Z_
_Verifier: Claude (gsd-verifier)_
