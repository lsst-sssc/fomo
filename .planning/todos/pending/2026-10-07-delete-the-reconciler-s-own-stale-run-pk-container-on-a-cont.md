---
created: 2026-10-07T02:59:35.000Z
title: "Delete the reconciler's own stale RUN:{pk} container on a container-to-per-night re-classification"
area: campaign-reconciler
severity: major
files:
  - solsys_code/campaign_reconciler.py:543
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
---

## Problem

Re-classifying a run from container back to per-night leaves its own reconciler-minted `RUN:{pk}` event on the calendar: `_stale_attributions()` only detaches a stale bare container, so the duplicate span stays and the attribution queue re-offers it to the same run at HIGH band (a confirm would make it permanent). Hit once live (run 1, event 418, deleted by hand 2026-10-02).

Source: `.planning/v2.4-INTENT-REVIEW.md`, F8 (v2.4 intent review walkthrough, 2026-10-06).

## Solution

In the convergence step, a stale bare container whose url is the run's OWN `run_container_url(run)` and whose companion row has no `confirmed_by` is deleted and counted under `legacy_deleted`, like the two per-night families; an adopted/hand-entered container keeps the detach rule. Test a per-night → container → per-night round trip; paired notebook cell; runbook `legacy_deleted` wording.
