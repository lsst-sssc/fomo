# Spike Wrap-Up Summary

**Date:** 2026-09-03
**Spikes processed:** 5 (4 numbered; 001 is an a/b comparison)
**Feature areas:** event trigger, observation projector, allocation handoff
**Skill output:** `./.claude/skills/spike-findings-fomo_devel/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001a | trigger-tom-hook | comparison | PARTIAL | event trigger |
| 001b | trigger-django-post-save | comparison | VALIDATED (winner) | event trigger |
| 002 | observation-projector | standard | VALIDATED | observation projector |
| 003 | allocation-night-retirement | standard | VALIDATED | allocation handoff |
| 004 | live-narrowing-updatestatus | standard | PARTIAL (baseline captured; close after real nights) | observation projector |

## Key Findings

- **Trigger:** `ObservationRecord.save()` calls TOM's `observation_change_state` hook only on a
  status change or creation, so it misses the scheduler placing a block (schedule-only save).
  A FOMO-owned Django `post_save` receiver catches every save, including the real
  `updatestatus` path; `queryset.update()` bypasses both, so a sweep remains the backstop.
  Install the receiver in `apps.ready()`, not in a script.
- **Projector:** one `tom_calendar.CalendarEvent` per record, keyed by the facility's observation
  URL, built from `record_time_window` (the existing stage rule) and
  `insert_or_create_calendar_event` (the existing no-churn contract). On 146 real records:
  146/146 stage-correct spans, all 16 terminal-negative nights marked, all 9 groups
  series-titled, second sweep all `unchanged`, 0.1–0.3 s per sweep, `RUN:` untouched. A
  schedule-only save re-projects the event with no sweep.
- **Handoff:** an allocation (`CampaignRun` with `campaign=None`, the 32-01 slice) projects
  sunset→sunrise nights under its own key; a night with a linked `CampaignRunObservation` has
  no allocation event; unlinking restores it. Attribution is `CalendarEventMeta.run` only —
  the base event stayed byte-identical.
- **Landmines for the build:** the reconciler's adopt/re-key/detach paths read
  `CalendarEventMeta.run` as ownership and must be retired or scoped; series identity needs a
  real field; nights must be keyed site-locally; titles need a compact month-cell form;
  `[SCHEDULED]` is a vocabulary gap.
- **Answered:** `tom_calendar` ships inside tomtoolkit 3.0.1 (TOM-org maintained) — the
  natural upstream target for SEED-004.
- **Routing recommendation:** open a new milestone (v2.4, observation-first calendar) rather
  than re-scope v2.3, whose core value and `ADAPT-*`/`OUTCOME-*` requirements describe the
  `CampaignRun` middleman the spikes removed. Resolve plan 32-01's checkpoint by re-scoping,
  not by choosing a/b/c; do not run 32-02..04.
