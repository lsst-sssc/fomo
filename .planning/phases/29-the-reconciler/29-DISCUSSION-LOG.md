# Phase 29: The Reconciler - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-04
**Phase:** 29-the-reconciler
**Areas discussed:** Adopt vs. gap-fill for classical runs' existing events, Per-run
reconciliation trigger mechanics (RECON-08), Dry-run / failure reporting shape (RECON-06)

---

## Pending todo cross-reference

| Option | Description | Selected |
|--------|-------------|----------|
| Leave it out | Consistent with Phases 26-28's reasoning — no CANON/ATTRIB/RECON requirement behind it, Phase 29 already carries enough scope | ✓ |
| Fold it in | Do the `calendar_utils.py` private-helper rename as part of Phase 29 | |

**User's choice:** Leave it out.
**Notes:** `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` reviewed a fourth time, declined a fourth time, same reasoning as Phases 26/27/28.

---

## Adopt vs. gap-fill for classical runs' existing events

| Option | Description | Selected |
|--------|-------------|----------|
| Adopt | Find the existing attributed row via the companion FK and update/re-key it to `RUN:{run_pk}:{date}` in place | ✓ |
| Gap-fill | Leave the existing attributed row byte-untouched; only mint `RUN:{pk}:{date}` for genuinely uncovered nights | |
| Something else / let's discuss | — | |

**User's choice:** Adopt.
**Notes:** Unlike the already-settled queue-run verdict (which chose gap-fill to avoid a recurring churn loop with `sync_lco_observation_calendar`, a url-keyed writer), `load_telescope_runs` looks up by `(telescope, instrument, start_time)`, not `url` — so re-keying doesn't create the same churn risk. Adopting closes the `event.telescope_label_meta.run`-always-unset gap WR-03 documented.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — reconciler takes over, CAMPAIGN: keys retired | approve/resolve_site stop calling `_project_calendar_event()` and call the reconciler instead; zero live `CAMPAIGN:` events means nothing to migrate | ✓ |
| No — CAMPAIGN: stays a separate, coexisting mechanism | `_project_calendar_event()` keeps running for WEB-sourced runs; reconciler only handles non-WEB sources | |

**User's choice:** Yes — reconciler takes over, CAMPAIGN: keys retired.
**Notes:** Extends the adopt logic to `_project_calendar_event()`'s own `CAMPAIGN:{pk}[:date]`-keyed events (SPIKE-02's fourth adapter mapping). D-15 (`26-DECISION.md`) confirms zero live `CAMPAIGN:` events exist, so this is a clean cutover, not a migration.

---

## Per-run reconciliation trigger mechanics (RECON-08)

| Option | Description | Selected |
|--------|-------------|----------|
| Yes to both | Synchronous inline call; preserve approve()'s swallow / resolve_site()'s keep-retryable asymmetry | ✓ |
| Synchronous, but uniform failure handling | Still inline, but treat all four actions' failures the same way | |
| Something else / let's discuss | — | |

**User's choice:** Yes to both.
**Notes:** No Celery/async — already rejected in Phase 26. `mark_cancelled`/`mark_weather_failure` follow whichever of the two existing patterns they already resemble in `_set_run_status()`.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, single shared function | `reconcile_run(run)` in `campaign_reconciler.py` does all four stages for one run; command loops it, staff actions call it directly | ✓ |
| Something else / let's discuss | — | |

**User's choice:** Yes, single shared function.
**Notes:** Matches Phase 25's precedent (`backfill_range_calendar_events` delegating 100% of projection math to a single shared function).

---

## Dry-run / failure reporting shape (RECON-06)

| Option | Description | Selected |
|--------|-------------|----------|
| Same created/updated/unchanged/skipped shape | Matches `import_campaign_csv`/`backfill_range_calendar_events`; skipped carries a reason string | ✓ |
| Per-stage breakdown (stage 0-4 counts) | Additional breakdown by which pipeline stage each run landed in | |
| Something else / let's discuss | — | |

**User's choice:** Same created/updated/unchanged/skipped shape.

| Option | Description | Selected |
|--------|-------------|----------|
| Catch per-run, continue the batch, no atomic wrap | Batch loop catches per-run exceptions, records skipped-with-reason, continues; no `transaction.atomic()` within a run | ✓ |
| Wrap each run's reconcile in transaction.atomic() | All-or-nothing per run, batch still continues | |
| Something else / let's discuss | — | |

**User's choice:** Catch per-run, continue the batch, no atomic wrap.
**Notes:** Matches Phase 25's accepted partial-projection posture; safe because a re-run is idempotent.

---

## Claude's Discretion

- Exact name of the shared per-run function (`reconcile_run` used as placeholder).
- Commit sequencing for deleting `_project_calendar_event()`/`_calendar_event_title()`
  relative to wiring the staff actions to the reconciler.
- Whether to add a per-stage (0-4) breakdown to the summary shape, beyond the required
  created/updated/unchanged/skipped counts.
- Test organisation, and how `mark_cancelled`/`mark_weather_failure` map onto the shared
  reconcile function.

## Deferred Ideas

- Per-stage breakdown in command summary output — offered, not chosen, not rejected;
  left as discretion rather than locked either way.
- v2.3 items (adapter rewiring, provenance-blind gap analysis, status-vocabulary
  unification, unused-allocation display) — untouched, as already established by prior
  phases' context.
