---
created: 2026-09-01T17:08:10.247Z
title: Skip sun_event computation for already-existing reconciler nights
area: general
severity: minor
files:
  - solsys_code/campaign_reconciler.py:381
  - solsys_code/campaign_reconciler.py:406
  - solsys_code/tests/test_campaign_reconciler.py
resolves_phase: 35
---

## Problem

Finding F2 from the 2026-09-01 review of the `issue37-code-only` branch (artifact:
https://claude.ai/code/artifact/98811990-b03d-44e7-b9eb-48f967942c5c).

`_reconcile_classical_nights()` calls `sun_event(run.site, night, kind='sun')`
unconditionally at the top of its per-night loop (`campaign_reconciler.py:381`),
but the resulting `sunset`/`sunrise` values are only used when minting a NEW
event (`existing is None` branch, ~line 406). On an update of an existing or
adopted event only `title`/`description`/`target_list` (and, for adopts, `url`)
are written — `start_time`/`end_time` are deliberately never rewritten after
creation.

Consequence: an idempotent `reconcile_campaign_runs` sweep over already-
reconciled multi-night runs pays a 1441-point astropy coarse solar-altitude scan
plus ~10 bisection refinements per night, per run, for results that are thrown
away. Behavior is correct; this is pure wasted computation that grows with
calendar size and sweep frequency.

## Solution

Move the `sun_event()` call inside the `existing is None` branch of
`_reconcile_classical_nights()` (after the per-night event lookup and the
adopt fallback), so it only runs when a new event will actually be created.

Care point: the D-06 contract says `sun_event()`'s `ValueError` (e.g. blank
`Observatory.timezone`) propagates uncaught out of `reconcile_run()` so callers
apply their own handling. Moving the call means a run whose events all already
exist no longer raises for a blank timezone — confirm no test or caller depends
on the raise happening for the update-only path (check
`test_campaign_reconciler.py` and the approve/resolve_site call sites in
`campaign_views.py`). Add/adjust a regression test asserting an idempotent
re-reconcile of an existing multi-night run performs no sun-event computation
(e.g. mock `campaign_reconciler.sun_event` and assert not called).

## v2.4 routing note (2026-09-03)

v2.4 Phase 35 (Allocation Layer & Classical Cutover) rewrites `_reconcile_classical_nights()`'s per-night `RUN:{pk}:{date}` projection into allocation events (ALLOC-01/05). The no-churn early-exit this todo asks for should be built into the new allocation projector rather than patched into the code being retired.
