---
phase: 27-the-canonical-run-record
reviewed: 2026-08-06T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/templates/campaigns/approval_queue.html
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/templatetags/attribution_display_extras.py
  - src/templates/tom_calendar/partials/event_form.html
  - solsys_code/tests/test_calendar_template.py
  - docs/runbooks/telescope_runs_calendar.rst
findings:
  critical: 0
  warning: 0
  info: 2
  total: 2
status: clean
---

# Phase 27 (27-07 gap closure): Code Review Report

**Reviewed:** 2026-08-06
**Depth:** standard
**Files Reviewed:** 6
**Status:** clean

## Summary

This is a scoped review of plan 27-07, which closes two minor UAT gaps found during
Phase 27 re-verification: (1) reordering the approval-queue sections so "Sites Needing
Review" renders first, and (2) adding a staff-only "Possible campaign run match" hint to
the calendar-event modal for unlinked events that have a HIGH-band attribution-queue
candidate.

The diff is small and self-contained: a pure template reorder
(`approval_queue.html`), a new one-function template-tag module
(`attribution_display_extras.py`) that thinly wraps an already-reviewed, well-tested
scoring function (`campaign_attribution.candidates_for_event`), an `elif` branch added to
`event_form.html`, and matching test/doc updates. I traced the new template logic against
its callees (`event.telescope_label_meta`'s `related_name`, `candidates_for_event`'s
`None`-safe scoring chain, `CampaignRun.__str__`'s deliberate PII exclusion) and did not
find a correctness or security defect.

Verification performed beyond static reading:
- `ruff check` and `ruff format --check` on the three Python-syntax files: clean.
- `python manage.py test solsys_code.tests.test_calendar_template
  solsys_code.tests.test_campaign_approval`: 164/164 pass.
- Manually traced the `{% elif not run and request.user.is_staff %}` branch against all
  four companion-row states (no row, row with `run=None`, row with a
  non-publicly-visible run, row with a publicly-visible run) and confirmed the rendered
  output matches what the tests assert in each case.
- Confirmed `CampaignRun.__str__` (used via `{{ candidate.run }}`) deliberately excludes
  `contact_person`/`contact_email`, so the new staff-only hint does not leak PII beyond
  what the runbook claims.
- Confirmed the hint's `?band=high` query string matches `campaign_attribution.BAND_HIGH
  = 'high'` and the `campaigns:attribution` URL exists.

No blockers or warnings found. Two minor info-level observations below, neither of which
blocks shipping this delta.

## Info

### IN-01: New per-render DB queries for the attribution hint have no test asserting a query cap

**File:** `solsys_code/templatetags/attribution_display_extras.py:27-39`, `src/templates/tom_calendar/partials/event_form.html:164`
**Issue:** `high_band_attribution_candidates` calls `campaign_attribution.candidates_for_event(event)` with `dismissed_run_ids=None`, so every staff render of an unlinked event's modal now issues (at minimum) one dismissal query plus one `CampaignRun` query per eligible campaign — on top of whatever `tom_calendar.views.update_event` already does. This is a single-event view (not a list), so it's not a list-page N+1 in the classic sense, and per the review's explicit out-of-scope note, performance is not gated in v1. Flagging only because the existing `test_display09_query_count_bounded` pattern in `test_calendar_template.py` shows this codebase does care about asserting query counts elsewhere, and this new path has no equivalent guard.
**Fix:** Optional follow-up: add a `CaptureQueriesContext` assertion around the staff-visible modal render (mirroring `test_display09_query_count_bounded`) if this hint is later exercised from a list/summary view rather than a single-event modal.

### IN-02: `_eligible_runs_for_event` does not filter by `approval_status`, so the staff hint can surface a REJECTED run as a "possible match"

**File:** `src/templates/tom_calendar/partials/event_form.html:164-180` (calls into pre-existing `solsys_code/campaign_attribution.py:474-487`, not itself part of this delta)
**Issue:** `candidates_for_event()` scores every `CampaignRun` in the same campaign as the orphan event, regardless of `approval_status`. A REJECTED or still-PENDING_REVIEW run can therefore appear in the new "Possible campaign run match" hint, which could read as endorsing a run staff already rejected. This is pre-existing behavior of `campaign_attribution.py` (not modified by this delta) and is consistent with how the rest of the attribution queue already works, so it is not a regression introduced by 27-07 — noting it here only because this delta is what newly surfaces rejected/pending runs to this particular staff-only surface (the modal), where the previous behavior was to show nothing at all.
**Fix:** No action required for this delta. If desired, a future change to `_eligible_runs_for_event`/`candidates_for_event` could exclude `REJECTED` runs; that's a `campaign_attribution.py` change outside this review's scope.

---

_Reviewed: 2026-08-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
