---
phase: 23-weather-storm-cancellation-handling-give-staff-a-way-to-mark
reviewed: 2026-07-16T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/campaign_views.py
  - solsys_code/campaign_tables.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/test_load_telescope_runs.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_calendar_display_extras.py
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 23: Code Review Report

**Reviewed:** 2026-07-16
**Depth:** standard
**Files Reviewed:** 7 (+ 1 lower-priority demo notebook, inspected but not exhaustively line-audited)
**Status:** issues_found (no blockers — two low-probability robustness warnings and two minor info items)

## Summary

Reviewed all three plans' deliverables for Phase 23: the classical-run `[CANCELLED]` title
prefix (`load_telescope_runs.py`), the new `CampaignRunDecisionView._set_run_status()`
staff-facing status-change action plus its Decided-table UI (`campaign_views.py`,
`campaign_tables.py`), the `[WEATHERED]` terminal-prefix addition
(`calendar_display_extras.py`), and the Gemini FT-115 test encoding
(`test_campaign_approval.py`). Verified against the phase's CONTEXT/RESEARCH/PLAN/SUMMARY
docs and traced the actual runtime behavior of `insert_or_create_calendar_event()`
(`calendar_utils.py`) and `_resolve_status()`/`KNOWN_STATUSES` (`telescope_runs.py`) that the
new code depends on.

This is a well-executed, heavily self-reviewed implementation — the plans themselves already
bake in three internal review passes' worth of fixes (REVIEW findings #1-#5 cited throughout
the plan text and now present in the shipped code: the `updated_count == 0` short-circuit
before `refresh_from_db()`, the `status_actions` independent flag, the corrected
`BULK_MPC_FIXTURE`-avoidance for `resolve_site('I11')`, the `Run status:` description line,
and the real `WEATHER_TECH_FAILURE → CANCELLED` transition test). I ran the full
`test_campaign_approval` / `test_calendar_display_extras` / `test_load_telescope_runs` suites
directly (151 tests, all green) and `ruff check`/traced line lengths manually — both clean, so
I did not re-litigate anything ruff or the test run itself would have caught.

Given the explicit focus on `_set_run_status()`'s guard/existence-check logic: the
business-logic guard, the staleness-safe conditional `.update()` with an `updated_count == 0`
short-circuit placed correctly before `refresh_from_db()`/the calendar-sync branch, and the
"only touch a `CalendarEvent` that already exists" guard are all correctly implemented and
match their test coverage. The findings below are two narrow, low-probability robustness gaps
in that same method plus two minor code-quality/wording nits — none rise to a blocker.

## Warnings

### WR-01: TOCTOU race between the calendar-event existence check and the actual write in `_set_run_status()`

**File:** `solsys_code/campaign_views.py:752-760`
**Issue:** The existence guard is a classic check-then-act:

```python
if CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists():
    prefix = _RUN_STATUS_CALENDAR_PREFIX[new_run_status]
    insert_or_create_calendar_event(
        {'url': f'CAMPAIGN:{run.pk}'},
        fields={...},
    )
```

The `.exists()` check and the subsequent `insert_or_create_calendar_event()` call are two
separate queries with no transaction/locking between them. If the matched `CalendarEvent` row
is deleted between the two (currently only reachable via Django admin — there is no in-app
delete path for `CalendarEvent` anywhere in `solsys_code/`), `insert_or_create_calendar_event()`
falls through to `CalendarEvent.objects.get_or_create(url=..., defaults=fields)`'s create-path,
which raises (non-nullable `start_time`/`end_time` are not in `fields`) — the exact crash
RESEARCH Pitfall 1 / T-23-06 documents and this guard was written to prevent, just via a
narrower trigger than "never had a projected event." This is the same category of gap the
method's own docstring and the STRIDE table are careful to close for the `run_status` write
itself (the `updated_count == 0` short-circuit) but leave open for the calendar-sync branch.
**Likelihood is low** (no delete UI exists for `CalendarEvent` in this codebase today) but the
failure mode if it does occur is an unhandled 500, not a graceful warning+redirect like every
other guarded branch in this method.
**Fix:** Either wrap the existence-check + update in `transaction.atomic()` with
`select_for_update()` on the matched `CalendarEvent`, or — simpler, matching this codebase's
established idiom — re-check inside a try/except around the `insert_or_create_calendar_event()`
call and degrade to `messages.warning(...)` (mirroring `_resolve_site()`'s own
non-reverting `except Exception` around its projection call at campaign_views.py:688-697)
instead of letting a `CalendarEvent.DoesNotExist`/`IntegrityError` propagate as a 500.

### WR-02: New `_set_run_status()` call site inherits `insert_or_create_calendar_event()`'s non-unique `url`-based `get_or_create()` fragility

**File:** `solsys_code/campaign_views.py:752-760`, `solsys_code/calendar_utils.py:375`
**Issue:** `tom_calendar.models.CalendarEvent.url` (site-packages) is a plain
`models.URLField(blank=True, default="")` with **no** `unique=True` / `UniqueConstraint` at the
DB level. `insert_or_create_calendar_event()`'s URL-keyed path calls
`CalendarEvent.objects.get_or_create(url=..., defaults=fields)`, which raises
`CalendarEvent.MultipleObjectsReturned` (another unhandled 500) if more than one row ever shares
that `url` value — a state the DB schema does not itself prevent, only application-level
discipline (every existing caller routing through this one helper) does. This is a pre-existing
structural characteristic of the shared helper, not something this phase introduced, but Phase
23 does add a brand-new, staff-triggerable call site (`_set_run_status()`) to it, so it's worth
flagging now rather than only when a `CAMPAIGN:{pk}` duplicate eventually occurs (e.g. from a
future concurrency bug in `_project_calendar_event()`/`resolve_site()`, which itself has no DB
constraint backing its own "one event per run" assumption).
**Fix:** Out of scope to fix within this phase (would mean adding a migration + a
`unique=True`/`UniqueConstraint` on `CalendarEvent.url`, or a `.filter(url=...).first()` +
explicit `MultipleObjectsReturned`-tolerant helper), but worth tracking as a follow-up hardening
item for `calendar_utils.insert_or_create_calendar_event()` given it now has four distinct
call sites (LCO/SOAR sync, Gemini sync, `_project_calendar_event()`, and now
`_set_run_status()`).

## Info

### IN-01: `_set_run_status()`'s guard-rejection message is imprecise for a REJECTED run

**File:** `solsys_code/campaign_views.py:731-733`
**Issue:**
```python
if run.approval_status != CampaignRun.ApprovalStatus.APPROVED:
    messages.warning(request, 'This run has not been approved yet.')
```
This guard correctly rejects any non-APPROVED `approval_status` (PENDING_REVIEW *or*
REJECTED), but the message text only makes sense for the PENDING_REVIEW case ("not... yet"
implies it's still pending). A REJECTED run hit via a direct/tampered POST (not reachable
through the rendered UI, since the Decided-table action only renders for APPROVED rows) would
show a mildly misleading message. Purely cosmetic — this code path requires deliberately
crafting a POST outside the UI to reach.
**Fix:** `'This run is not approved (currently {}).'.format(run.get_approval_status_display())`
or similar, if this is ever revisited; not worth a standalone change today.

### IN-02: No automated test exercises the "single-night run with an unresolved site" no-event branch directly

**File:** `solsys_code/tests/test_campaign_approval.py` (`TestRunStatusChange`)
**Issue:** The must-have backstop test (`test_mark_range_window_run_does_not_crash_and_creates_no_event`)
covers the range-window "never had a projected event" case, which exercises the same
`if CalendarEvent.objects.filter(...).exists():` guard as the unresolved-site case — both paths
converge on the identical `_project_calendar_event()` early-return (`not (... and run.site and
...)`), so this isn't a functional gap, but a reviewer skimming only the test names could
mistake the range-window test as the sole "no event" scenario covered, when D-05's must-have
truth explicitly lists "range/TBD/unresolved-site" as three distinct scenarios.
**Fix:** Optional — a fourth `subTest`-style case in the same test method (e.g. a single-night
run with `site=None`) would make the coverage claim self-evident without meaningfully changing
risk, since the code path is already provably identical.

---

_Reviewed: 2026-07-16_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
