---
created: 2026-09-01T17:08:10.247Z
title: Guard attribution dismiss action with is_offered_candidate
area: general
severity: minor
files:
  - solsys_code/campaign_views.py:1275
  - solsys_code/campaign_attribution.py:798
  - solsys_code/tests/test_attribution_dismissals.py
---

## Problem

Finding F4 from the 2026-09-01 review of the `issue37-code-only` branch (artifact:
https://claude.ai/code/artifact/98811990-b03d-44e7-b9eb-48f967942c5c).

Every confirm path in `AttributionDecisionView` re-validates server-side via
`campaign_attribution.is_offered_candidate()` before writing (T-28-12), but
`_dismiss()` (`campaign_views.py:1275`) does not. Consequences:

1. A stale resubmit or tampered staff POST can persist a
   `CalendarEventDismissal`/`ObservationRecordDismissal` row for a pair that was
   never offered. Because `candidates_for_event()`/`candidates_for_record()`
   exclude every dismissed run unconditionally, that pair is then silently
   suppressed if it ever becomes a legitimate candidate later — the same hazard
   28-REVIEW.md WR-01 closed for `_undo_confirmation()` by gating its dismissal
   write on `changed_count`.
2. A nonexistent orphan/run pk raises a FK `IntegrityError` that the savepoint
   handler swallows, and the user still sees the success message "Candidate
   dismissed." — a false success report.

Staff-only surface, so severity is minor, but it is an inconsistency with the
view's own stated discipline (its class docstring claims server-side
re-validation "on every action that creates an association"; a dismissal is not
an association but has the same permanence).

## Solution

In `_dismiss()`, after the reason check, call
`campaign_attribution.is_offered_candidate(kind, orphan_pk, run_pk)` and abort
with the existing "already confirmed or dismissed by someone else"-style warning
(or a more accurate "this candidate is not currently offered" message) when it
returns None — mirroring `_confirm()`'s shape. This also fixes consequence 2
for free (nonexistent pks are never offered).

Note: dismissing a pair whose orphan was *just* confirmed to another run will
now warn instead of silently recording a dismissal — that is the correct
behavior (the pair is no longer in the queue). Add regression tests in
`test_attribution_dismissals.py`: never-offered pair → no dismissal row +
warning; nonexistent pks → no false success message.
