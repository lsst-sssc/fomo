---
id: SEED-003
status: dormant
planted: 2026-09-01
planted_during: v2.2 complete / awaiting next milestone (post-v2.2 branch review session)
trigger_when: next milestone (/gsd-new-milestone) — this is the operator's stated scope for it
scope: large
---

# SEED-003: Automatic sync of robotically scheduled LCO/SOAR observations and their outcomes

## Why This Matters

Operator decision (2026-09-01, during the issue37-code-only branch review): **the
branch is not "feature complete" for PR #43 without this** — even acknowledging the
considerable scope creep since issue #37 was written. Without automatic sync of
SOAR/LCO's robotically scheduled observations (recorded through the TOM's
`ObservationRecord`s) and their outcomes, the calendar only reflects reality when an
operator remembers to run the right management command. This is the
feature-completeness bar gating the PR.

Verified against the branch (see the review artifact,
https://claude.ai/code/artifact/98811990-b03d-44e7-b9eb-48f967942c5c): nothing on the
branch runs by itself — no celery/huey/cron dependency, no periodic-task wiring, and
the operator runbook never documents a scheduled invocation. The decomposition into
three concrete gaps:

1. **No automation layer.** `sync_lco_observation_calendar` handles the full
   lifecycle correctly ([QUEUED] banner -> placed block -> terminal/outcome
   prefixes) but only when invoked. Something must run it: the TOM-conventional
   answer is cron invoking the management commands (zero new dependencies, matches
   how TOM Toolkit handles alert ingestion); a real task queue
   (django-celery-beat/huey) is likely overkill for a single-server SQLite
   deployment. **This is the genuinely new requirement — the v2.3 deferral list
   never mentions automation.**
2. **Discovery of robotically scheduled observations isn't unattended-capable.**
   `backfill_lco_observation_records` covers observations that never went through
   the TOM, but is pull-on-demand: it requires a proposal + name prefix per
   invocation. Unattended operation needs a configured set of watched proposals it
   sweeps without arguments. Idempotency is already in place, so this is mostly a
   configuration surface, not a rewrite.
3. **Outcomes reach the calendar but not the run.** An observation's outcome
   updates its CalendarEvent today, but nothing propagates it to
   `CampaignRun.run_status` (stage 4 of the v2.2 four-stage window pipeline at
   run level). This is exactly the deferred v2.3 adapter-rewiring work: adapters
   write CampaignRuns (`Source.LCO_QUEUE` etc., declared in v2.2 but produced by
   no code path yet), the reconciler projects events, attribution closes the loop.

## When to Surface

**Trigger:** `/gsd-new-milestone` — this seed IS the intended next milestone scope,
per the operator's 2026-09-01 statement. Combine with the already-deferred v2.3
items it overlaps (adapter rewiring, status-vocabulary unification,
provenance-blind coverage gap) and issue #37 Stage 4, which this completes.

Suggested core value framing: "robotically scheduled LCO/SOAR observations and
their outcomes appear and update on the calendar and their campaign runs without an
operator running anything."

## Scope Estimate

**Large** — a full milestone: an automation/scheduling layer (needs a design
decision: cron vs task queue), unattended multi-proposal discovery configuration,
and the v2.3 adapter-rewiring + run-level outcome propagation. Phase 26's
deliberately-open D-11 adopt-vs-gap-fill write strategy resolves inside this work.

## Breadcrumbs

- `solsys_code/management/commands/sync_lco_observation_calendar.py` — the lifecycle sync that needs scheduling
- `solsys_code/management/commands/backfill_lco_observation_records.py` — pull-on-demand discovery (proposal + name-prefix args)
- `solsys_code/campaign_reconciler.py` — `reconcile_run()`, the projection the adapters will feed
- `solsys_code/models.py` — `CampaignRun.Source` (LCO_QUEUE/GEMINI_QUEUE/CLASSICAL_FILE declared, unproduced) and `RunStatus` (stage-4 target vocabulary)
- `.planning/PROJECT.md` "Deliberately deferred to v2.3" bullet — adapter rewiring, status-vocabulary unification, provenance-blind gap analysis
- `.planning/STATE.md` Deferred Items — ESO-10/11, SCHED-06, SUBMIT-06/07 candidates to triage alongside
- Review artifact: https://claude.ai/code/artifact/98811990-b03d-44e7-b9eb-48f967942c5c (Gap analysis A, Stage 4 row)
- PR #43 (draft) — stays draft until this ships, per the operator's bar; note the diff-growth trade-off discussed 2026-09-01

## Notes

Related dormant seeds SEED-001/SEED-002 (ESO-triggered) may partially wake with
this milestone if ESO-10/11 are pulled in — the ESO sync would be a fourth adapter
through the same rewired path.
