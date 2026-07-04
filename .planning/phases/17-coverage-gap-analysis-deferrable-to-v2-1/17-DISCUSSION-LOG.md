# Phase 17: Coverage-Gap Analysis (Deferrable to v2.1) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-04
**Phase:** 17-Coverage-Gap Analysis (Deferrable to v2.1)
**Areas discussed:** GAP-01 research-spike decision, "Claimed" definition, Trigger/caching & date
range, Target + site selection

---

## GAP-01 research-spike decision

| Question | Options | Selected |
|---|---|---|
| Lock dark-window-only now, or have researcher re-validate at plan time? | Lock now ✓ / Have researcher re-validate | Lock dark-window-only now |
| Does GAP-01 still need its own written decision doc? | Write a short decision doc ✓ / CONTEXT.md is sufficient | Write a short GAP-01 decision doc |
| Per-date `sun_event()` `ValueError` handling | Skip that date, continue ✓ / Abort whole request | Skip that date, mark unknown, continue |
| Minimum dark-window duration to count as "observable"? | Any non-zero window ✓ / Require a minimum (e.g. 1h) | Any non-zero dark window counts |

**User's choice:** Lock dark-window-only immediately based on the unanimous pre-milestone research
(`ARCHITECTURE.md`, `PITFALLS.md`, `SUMMARY.md`, `STACK.md`); still write a short decision doc
during execution to satisfy the literal success-criterion wording; per-date failures are
skip-and-continue; no minimum dark-window duration threshold.
**Notes:** Pre-milestone research was cited directly during discussion — all four sources already
agreed, so no new research question remained.

---

## "Claimed" definition

| Question | Options | Selected |
|---|---|---|
| Which status combination counts as "claimed"? | approved + non-terminal-failure run_status ✓ / approved only / any non-rejected | approved AND run_status not in {cancelled, not_awarded, weather_tech_failure} |
| Which field(s) determine the claimed date? | obs_date, else ut_start ✓ / ut_start/ut_end range only / obs_date only | obs_date if set, else derive from ut_start |
| Timezone for the ut_start fallback? | Site's local date (sun_event convention) ✓ / raw UTC date | Site's local date, same convention as sun_event() |
| Handling of runs with neither obs_date nor ut_start? | Ignore it ✓(recommended) / Flag as undated, needs review | Flag it separately as "undated, needs review" |

**User's choice:** Approved-and-not-later-failed runs claim a date; `obs_date` is authoritative,
falling back to `ut_start`'s site-local date; runs with no date info at all are surfaced as a
separate "undated, needs review" flag rather than silently ignored (user chose the non-default
option here).
**Notes:** User deliberately picked the non-recommended option on the last question — preferring
visibility of the data-quality issue over silent omission.

---

## Trigger, caching & date range

| Question | Options | Selected |
|---|---|---|
| What triggers computation? | Server-rendered button ✓ / htmx-loaded partial | Button loading a separate section/page via normal request |
| How are results cached? | Django low-level cache w/ TTL ✓ / dedicated model w/ invalidation | Django's cache framework, keyed by (campaign, target, site, range) |
| Cache TTL? | 1 hour ✓ / 24 hours / Other | 1 hour |
| Default/max date range? | 90 default / 180 max ✓ / 30 default / 90 max / explicit only | Default 90 days, max 180 days |

**User's choice:** Plain server-rendered button trigger, Django cache framework with a 1-hour TTL,
default 90-day window capped at 180 days.
**Notes:** Directly follows `PITFALLS.md`'s explicit recommendations (cap date range, cache with
TTL, show "last computed at").

---

## Target + site selection

| Question | Options | Selected |
|---|---|---|
| How is the target chosen? | Auto-select sole target / dropdown if multiple ✓ / always show dropdown | Auto-use sole Target if count==1, else dropdown |
| How is the site chosen? | Dropdown of sites used by this campaign ✓ / all Observatory (altitude>0) / SITES dict only | Dropdown of Observatory records already used by this campaign's CampaignRuns |
| Zero-target campaign handling? | Hide/disable button ✓ / show button, error on click | Hide/disable the gap-analysis button with an explanatory message |
| Zero-resolved-site campaign handling? | Hide/disable button ✓ / show button, error on click | Hide/disable the gap-analysis button with an explanatory message |

**User's choice:** Auto-select the sole target when unambiguous; site dropdown scoped to sites the
campaign has actually used; hide/disable the button (rather than error after the fact) when there's
no target or no resolved site to choose from.
**Notes:** Mid-discussion, the user raised (via free-text "Other") the idea of auto-computing
`ut_start`/`ut_end` on the Phase 16 submission form via JS — identified as out of Phase 17's scope
and captured as a deferred idea (see below) rather than pursued further here.

---

## Claude's Discretion

- Exact URL names/paths for the gap-analysis view/section.
- Exact wording of the "last computed at" display and the "undated, needs review" flag.
- Whether the gap-analysis result lives on its own page or as a section on the existing
  per-campaign table page.
- Internal structure/naming of the `17-GAP-01-DECISION.md` artifact (follow Phase 13's
  `13-DECISION.md` shape loosely).
- Exact cache key format (any stable, collision-free format is fine).

## Deferred Ideas

- Auto-calculate/suggest `ut_start`/`ut_end` on the Phase 16 public submission form via JS, based
  on entered site (MPC code) + `obs_date`, likely reusing `telescope_runs.sun_event()`. Raised
  during the Target + site selection discussion; out of Phase 17's scope (it's a Phase 16
  submission-form enhancement). Candidate for a future phase or quick task.

### Reviewed Todos (not folded)
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — keyword-overlap
  match only; already resolved per Phase 14/15/16 context. Third consecutive
  reviewed-not-folded outcome; not re-asked of the user this session.
- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — keyword-overlap
  match only; unrelated to coverage-gap analysis. Third consecutive reviewed-not-folded outcome;
  not re-asked of the user this session.
