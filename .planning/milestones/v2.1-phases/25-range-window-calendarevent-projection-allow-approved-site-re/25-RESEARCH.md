# Phase 25: Range-window CalendarEvent Projection - Research

**Researched:** 2026-07-17
**Domain:** Django view-layer bug fix / date-math extension (internal codebase only — no new external stack)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Guard fix (carried forward from the debug spec — locked, not re-discussed)**
- **D-01:** `_project_calendar_event()`'s guard changes from requiring
  `run.window_start == run.window_end` to requiring both `run.window_start` and
  `run.window_end` to be truthy (drop the equality clause, add a `window_end`
  truthiness check — the latter is REQUIRED, not optional, or a half-open window with
  `window_end = None` would reach the date-math with an unusable `None`). TBD runs
  (`window_start` is `None`), unresolved-site runs, and missing-`telescope_instrument`
  runs are unaffected — the existing truthiness checks for those already exclude them.
  Full rationale, provenance (Phase 19 D-06), and the confirmation that
  `CalendarEvent.start_time`/`end_time` support this (no model limitation) are in
  `.planning/debug/range-window-calendar-event.md`'s Resolution section — read that
  file in full before planning.

**Ground-branch date math — SUPERSEDES the debug spec's "Option A/B" recommendation**
- **D-02:** Ground-based range-window runs get **one `CalendarEvent` per night**, each
  with real dip-corrected sunset/sunrise from `sun_event()` — mirroring
  `load_telescope_runs`' existing `INGEST-01` precedent exactly (`E - S + 1` nightly
  events for a classical-schedule date range). The debug spec's two options (a single
  whole-day-span event, or a single event with dip-corrected first/last-night edges)
  are both **rejected** — the user explicitly chose per-night expansion instead,
  reasoning that the codebase already has this exact idiom for multi-night ranges and a
  single blunt-boundary span would be a regression from that existing physical-accuracy
  standard. **This is a bigger change than the debug spec scoped** (see D-03/D-04 for
  the consequences) — the researcher and planner must treat "one event per night" as
  the locked requirement, not a discretionary implementation choice.
- **D-03:** Because a `CampaignRun` can now own **multiple** `CalendarEvent`s instead of
  exactly one, the existing `CAMPAIGN:{pk}` single-event `url` key (used everywhere
  today: `_project_calendar_event()`'s create path, `_set_run_status()`'s
  existence-check-and-update path, and the `TestCalendarProjection`/`TestGeminiFtScenario`
  test assertions) needs a **new per-night key scheme**. No existing convention in this
  codebase keys multiple `CalendarEvent`s under one parent identity with a `CAMPAIGN:`
  prefix (the closest precedent, `load_telescope_runs`, doesn't use a `url` lookup key
  at all — it keys on `{telescope, instrument, start_time}`). Recommended default,
  left as **Claude's Discretion** for the researcher/planner to confirm or refine:
  `CAMPAIGN:{pk}:{date.isoformat()}` (e.g. `CAMPAIGN:34:2026-07-13`) — keeps the
  existing `CAMPAIGN:{pk}` prefix greppable/filterable while disambiguating nights.
  A single-night run (`window_start == window_end`) should very likely KEEP the
  existing bare `CAMPAIGN:{pk}` key unchanged (no need to touch working, already-tested
  single-night behavior) — confirm this during planning.
- **D-04:** `_set_run_status()` (Phase 23's `mark_weather_failure`/`mark_cancelled`
  handler) currently does a single existence-check
  (`CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists()`) then updates that
  ONE event's title/description via `insert_or_create_calendar_event()`. This must
  become "find and update EVERY `CalendarEvent` belonging to this run" (e.g.
  `.filter(url__startswith=f'CAMPAIGN:{run.pk}:')` for multi-night runs, in addition to
  the existing exact-match check for single-night runs) so the `[CANCELLED]`/
  `[WEATHERED]` prefix (Phase 23 D-03) lands on every night's event, not just one. The
  debug spec's claim "`mark_weather_failure`/`mark_cancelled` need NO change" is
  **superseded** by this decision — it needs a real rewrite of the lookup/update loop,
  though its guard logic (`approval_status == APPROVED`, conditional `.update()`,
  never-fabricate-for-a-run-with-no-events) is unaffected and must be preserved exactly.

**Satellite branch — unchanged (not re-discussed, confirmed as still correct)**
- **D-05:** The satellite branch's existing whole-day-span construction
  (`window_start` 00:00 UTC → `window_end` 23:59 UTC, ONE event) stays exactly as-is.
  Satellites have no physical "night"/dip-corrected-sunset concept the way ground sites
  do, so per-night expansion (D-02) does not apply there — this asymmetry is
  intentional, not an oversight. Confirm during research that the satellite branch
  is genuinely unreachable for the disputed real case (Gemini I11 is
  `OPTICAL_OBSTYPE`/ground, not satellite) so this stays a documentation note, not a
  functional change.

**Title/description format**
- **D-06:** Each per-night event's title gets a window-context suffix distinguishing it
  from an ordinary single-night classical run, e.g.
  `f'{campaign.name}: {telescope_instrument} (window {window_start}..{window_end})'`
  — applied identically to every night's event within the same run (not per-night
  numbering like "Night 2 of 4"). Rationale: unlike `load_telescope_runs`' classical
  nightly events (which are separately, individually confirmed nights and don't need
  this framing), each night here is part of one *awarded allocation* the calendar
  viewer should be able to recognize as connected — the suffix carries that meaning
  even though the events are otherwise visually indistinguishable from separate nights.
  Single-night runs (`window_start == window_end`) keep their existing unsuffixed title
  format unchanged.

**Backfill for already-approved runs**
- **D-07:** A **one-off management command** finds already-`APPROVED` `CampaignRun`s
  with a resolved, ground-based site and a range window (`window_start` !=
  `window_end`, both non-null) that have no existing `CAMPAIGN:{pk}*` `CalendarEvent`,
  and runs them through the same (now-fixed) projection logic. This is what makes the
  real GS-2026A-FT-115 row (`CampaignRun` pk=34 in this dev DB, already `APPROVED`
  under the OLD zero-event behavior) actually get its events once this phase ships —
  without this command, pk=34 would silently stay eventless forever, since projection
  only fires on the approve/`resolve_site` POST actions, never retroactively. Rejected
  alternatives: a Django data migration (side-effecting business logic like
  site-resolution-adjacent projection doesn't belong in a migration, and can't be
  safely re-run if something's missed); manual-only (leaves the user's actual
  motivating case, FT-115, unresolved after this phase ships — defeats the point).
  Command name/location/flags are Claude's Discretion.

### Claude's Discretion
- Exact per-night `CalendarEvent.url` key format (D-03) — `CAMPAIGN:{pk}:{date}` is the
  recommended default; confirm/refine during research.
- Whether single-night runs keep the bare `CAMPAIGN:{pk}` key unchanged, or also move
  to the new per-night scheme for consistency (D-03 leans toward "keep unchanged" to
  avoid touching working, already-tested behavior, but confirm during planning).
- Exact backfill management command name, location, and whether it supports a dry-run
  flag (D-07).
- The exact `_set_run_status()` rewrite shape (single combined queryset filter using
  `Q(url=f'CAMPAIGN:{pk}') | Q(url__startswith=f'CAMPAIGN:{pk}:')`, vs. two separate
  filter calls) (D-04).
- Which exact test assertions need which new values — the debug spec's list of "tests
  that must flip 0→1" now needs `count == 1` corrected to `count == (number of nights
  in the window)` for ground range-window cases; re-derive the full list from
  `test_campaign_approval.py`'s current state during planning rather than trusting the
  debug spec's exact counts verbatim (the debug spec assumed single-event ground
  projection, which D-02 supersedes).

### Deferred Ideas (OUT OF SCOPE)
- **Any new calendar visual/UI treatment beyond title text** (e.g. a distinct
  box-shadow ring or color for "part of a multi-night awarded window" vs. a genuinely
  separate classical night) — out of scope for this phase; D-06's title suffix is the
  full extent of the visual differentiation. A future phase could revisit this if the
  title suffix proves insufficient in practice.
- **Extending per-night expansion to the satellite branch** — explicitly rejected
  (D-05); satellites have no physical-night concept, so this isn't deferred as a future
  idea, just noted as intentionally out of scope permanently unless a future need
  emerges.
- Two previously reviewed pending todos (site/telescope-mapping extraction module,
  calendar_utils.py private-helper renaming) were re-checked during discuss-phase and
  remain not relevant to this phase's scope.
</user_constraints>

## Summary

This phase is a source-verified bug fix, not new-technology research. Every line number,
signature, and test assertion cited in `.planning/debug/range-window-calendar-event.md` and
`25-CONTEXT.md` was re-read directly against the live repository (not trusted from the debug
doc's possibly-stale line numbers), and all match almost exactly — the ground branch's `try`
block extends two lines further than the debug doc implied (through line 455, not 453), and one
test method is a few lines earlier than estimated. The real `CampaignRun` pk=34 row
(GS-2026A-FT-115) was queried directly against the dev DB (`src/fomo_db.sqlite3`) and confirmed:
`approval_status='approved'`, `site_id=17` (Gemini South, `observations_type=0`/OPTICAL,
`timezone='America/Santiago'`), `window_start=2026-07-13`, `window_end=2026-07-16` (4 nights),
`site_needs_review=0`, and zero existing `CalendarEvent` rows with `url LIKE 'CAMPAIGN:34%'` —
this row is exactly the D-07 backfill target described in CONTEXT.md.

The critical new finding beyond CONTEXT.md's own text: **`_set_run_status()`'s title
recomputation (line ~757) will silently drop the D-06 window-context suffix** on every
`mark_cancelled`/`mark_weather_failure` update unless it reconstructs the exact same title
format `_project_calendar_event()` used at creation time. Because
`insert_or_create_calendar_event()`'s no-churn diff is a plain field-value comparison (not a
prefix-aware merge), a naive `f'{prefix} {run.campaign.name}: {run.telescope_instrument}'`
recomputation is a *different string* than the range title
`f'{run.campaign.name}: {run.telescope_instrument} (window {ws}..{we})'` — the diff sees a
change, saves it, and the saved title has silently lost its window-context suffix. This must be
fixed by extracting one shared title-building helper both call sites use, not duplicating the
format string.

The other load-bearing finding: `f'CAMPAIGN:{run.pk}:'` (with a **trailing colon**) is the only
safe `url__startswith` prefix for D-04's combined queryset filter — without the trailing colon,
`CAMPAIGN:3` as a prefix would incorrectly match `CAMPAIGN:34:2026-07-13` (a real, easy-to-miss
substring collision between single- and double-digit pks).

**Primary recommendation:** Rewrite `_project_calendar_event()`'s ground branch as a single loop
over `date(run.window_start)..date(run.window_end)` inclusive (`n = (window_end -
window_start).days + 1` iterations, always ≥1) that, for `n == 1`, reproduces today's exact
bare-`CAMPAIGN:{pk}`-keyed single-event behavior, and for `n > 1`, creates one
`CAMPAIGN:{pk}:{date.isoformat()}`-keyed event per night. Leave the satellite branch's
whole-day-span construction and bare `CAMPAIGN:{pk}` key completely untouched (it always
produces exactly one event regardless of window shape, so it never needs the per-night key).
Extract a shared `_calendar_event_title(run)` helper used by both `_project_calendar_event()`
and `_set_run_status()` so the window-suffix and the `[CANCELLED]`/`[WEATHERED]` prefix compose
correctly. Four existing test assertions flip from `count == 0` to `count == n_nights`
(15, 15, 4, and 15 respectively, not the debug doc's assumed `count == 1`) with url-scheme and
message-text updates. Three tests genuinely stay `count == 0` unchanged. This phase touches only
`solsys_code/campaign_views.py` and `solsys_code/tests/test_campaign_approval.py`, plus one new
management command file — no paired demo notebook is required (confirmed: `campaign_views.py`
and `calendar_utils.py` are not in CLAUDE.md's four-module notebook-parity list).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Projection guard + ground/satellite date math (`_project_calendar_event`) | API/Backend (Django view module, `campaign_views.py`) | Database/Storage (`CalendarEvent` rows written) | Pure server-side business logic invoked from `CampaignRunDecisionView.post()`; no client-side or SSR involvement |
| Run-status → event title/description sync (`_set_run_status`) | API/Backend | Database/Storage | Same module, same request-response cycle; updates existing rows in place |
| Sun-event / dip-corrected crossing math (`sun_event`, `horizon_dip`) | API/Backend (`telescope_runs.py`, reused as-is) | — | Pure computation library, no I/O; already the ground branch's dependency today |
| No-churn create-or-update (`insert_or_create_calendar_event`) | API/Backend (`calendar_utils.py`, reused as-is) | Database/Storage | Shared helper across 4 call sites (LCO/Gemini sync, `load_telescope_runs`, this phase); this phase adds call *volume* (N calls per run instead of 1), not new logic |
| Backfill for already-approved runs (new management command) | API/Backend (CLI entry point, same business logic layer as the view) | Database/Storage | Not web-reachable; delegates to the same rewritten `_project_calendar_event()`, so it inherits the same tier assignment as the fix itself |

## Package Legitimacy Audit

Not applicable — this phase introduces zero new external packages. `astropy`/`erfa` (via
`telescope_runs.sun_event()`) and Django's own `django.db.models.Q`/`F` are already
dependencies in active use elsewhere in this codebase (`Q`/`F` need a new *import* into
`campaign_views.py`, not a new package).

## Standard Stack

Not applicable in the traditional sense — no new libraries. The relevant "stack" is the set of
already-verified existing internal functions this phase must call correctly:

| Function | Location | Signature (verified against live source) |
|----------|----------|--------------------------------------------|
| `sun_event(site, date, kind)` | `solsys_code/telescope_runs.py:251` | `(site: Observatory, date: date, kind: str) -> tuple[Time, Time]` — returns `(setting, rising)` astropy `Time` objects, UTC. Raises `ValueError` if `site.timezone` unset or crossings != 2. Already imported into `campaign_views.py` (`from .telescope_runs import sun_event`, line 52). |
| `insert_or_create_calendar_event(lookup, fields, *, start_time_tolerance=None)` | `solsys_code/calendar_utils.py:318` | `lookup` is an exact-match kwargs dict passed to `CalendarEvent.objects.get_or_create(**lookup, defaults=fields)`; `fields` is NOT merged into `lookup` automatically. Already imported (`from .calendar_utils import insert_or_create_calendar_event`, line 37). No `start_time_tolerance` needed here (this phase's `url` lookup is exact-string, unlike `load_telescope_runs`' float-drift-tolerant `start_time` lookup). |
| `_iter_run_nights` idiom (pattern to mirror, not import) | `solsys_code/management/commands/load_telescope_runs.py:54-89` | `n_nights = parsed.day2 - parsed.day1 + 1; [first_night + timedelta(days=i) for i in range(n_nights)]` — the exact `E - S + 1` inclusive-range idiom D-02 explicitly copies. |

**Installation:** none — no new packages.

**Version verification:** not applicable.

## Architecture Patterns

### System Architecture Diagram

```
CampaignRunDecisionView.post()  (approve / resolve_site / mark_cancelled / mark_weather_failure)
        │
        ├── approve ──────► conditional .update() (PENDING_REVIEW→APPROVED)
        │                        │
        │                        ├── site resolution (resolve_site(), unchanged)
        │                        └── _project_calendar_event(run)  ◄── FIX TARGET
        │                                 │
        │                                 ├── guard: telescope_instrument, site,
        │                                 │          window_start, window_end all truthy
        │                                 │          (D-01: drop window_start==window_end)
        │                                 │
        │                                 ├── site.observations_type == SATELLITE?
        │                                 │     └─ YES → ONE event, whole-day span,
        │                                 │              bare `CAMPAIGN:{pk}` key (D-05, unchanged)
        │                                 │
        │                                 └─ NO (ground) → loop window_start..window_end
        │                                        inclusive; per night:
        │                                          sun_event(site, night, 'sun')
        │                                          → dip-corrected sunset/sunrise
        │                                          → insert_or_create_calendar_event(
        │                                              lookup={'url': <bare key if n==1
        │                                                       else per-night key>},
        │                                              fields={...title w/ D-06 suffix...})
        │                                        (D-02/D-03, NEW)
        │
        ├── resolve_site ─────► same _project_calendar_event() call, retry-guarded
        │                        (unchanged control flow — only the callee's internal
        │                        behavior changes)
        │
        └── mark_cancelled/
            mark_weather_failure ──► _set_run_status()  ◄── FIX TARGET
                                          │
                                          ├── conditional .update(run_status=...)
                                          └── CalendarEvent.objects.filter(
                                                Q(url=f'CAMPAIGN:{pk}') |
                                                Q(url__startswith=f'CAMPAIGN:{pk}:')
                                              )  (D-04, NEW — was single .filter(url=...).exists())
                                                   │
                                                   └── for each matching event:
                                                       insert_or_create_calendar_event(
                                                         lookup={'url': event.url},
                                                         fields={'title': <prefix> + <SAME
                                                           title-building logic as creation,
                                                           i.e. window suffix preserved>})
```

### Recommended Project Structure

No new files/folders beyond one new management command:

```
solsys_code/
├── campaign_views.py                                 # _project_calendar_event() + _set_run_status() rewritten;
│                                                       #   new _calendar_event_title(run) helper extracted
├── management/commands/
│   └── backfill_range_calendar_events.py              # NEW — D-07's one-off backfill command
└── tests/
    └── test_campaign_approval.py                      # 4 existing tests revised, 2-3 new tests added
```

### Pattern 1: Per-night loop generalizes single-night and range uniformly

**What:** A loop `for night in (window_start + timedelta(days=i) for i in range(n_nights))`
where `n_nights = (window_end - window_start).days + 1` naturally covers *both* the existing
single-night case (`n_nights == 1`, loop body runs once) and the new range case
(`n_nights > 1`), if the `url` key and title suffix are chosen conditionally on `n_nights == 1`
vs `> 1` (or equivalently, on `window_start == window_end`) inside the loop body rather than
via a separate code path.

**When to use:** This is the recommended shape for the ground branch's rewrite — it means the
already-tested single-night dip-corrected-sunset/sunrise behavior (`sun_event()` called with
`run.window_start`) is reproduced *exactly* as today's code for `n_nights == 1`, with zero risk
of accidentally changing single-night math while adding range support.

**Example (illustrative — not a locked implementation, D-02/D-03 are locked, this exact code
shape is Claude's Discretion):**
```python
# Source: derived from load_telescope_runs.py's E - S + 1 idiom (management/commands/load_telescope_runs.py:78)
n_nights = (run.window_end - run.window_start).days + 1  # always >= 1; single night -> 1
is_range = n_nights > 1
created_any = False
for i in range(n_nights):
    night = run.window_start + timedelta(days=i)
    try:
        sunset, sunrise = sun_event(run.site, night, kind='sun')
    except ValueError:
        logger.debug(...)  # unchanged CR-01 re-raise behavior
        raise
    night_fields = dict(event_fields)  # shared title/description/telescope/target_list base
    night_fields['title'] = _calendar_event_title(run)  # see Pitfall below — must be shared
    night_fields['start_time'] = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    night_fields['end_time'] = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    url = f'CAMPAIGN:{run.pk}' if not is_range else f'CAMPAIGN:{run.pk}:{night.isoformat()}'
    insert_or_create_calendar_event({'url': url}, fields=night_fields)
    created_any = True
return created_any
```

### Pattern 2: Satellite branch stays completely untouched (D-05)

**What:** Satellite runs always produce exactly one `CalendarEvent` regardless of window shape
(single night or range), so they never need the per-night key scheme — the bare `CAMPAIGN:{pk}`
key stays correct and collision-free for satellites unconditionally.

**When to use:** No code change to lines 420-429 at all. The *only* reason a satellite range run
was ever blocked was the guard (D-01 fixes that); once the guard passes, the existing
`datetime.combine(window_start, 00:00 UTC)` / `datetime.combine(window_end, 23:59 UTC)`
construction is already range-correct (confirmed in the debug doc's Evidence section and
re-verified against live source at lines 423-424).

**Open question this phase must resolve during planning, not deferred:** should the satellite
branch's single event *also* get the D-06 window-context title suffix when it's a range
(`window_start != window_end`)? D-06's locked wording ("Each per-night event's title gets a
window-context suffix... distinguishing it from an ordinary single-night classical run") is
phrased around per-night events, but the underlying rationale (distinguishing an *awarded
allocation spanning multiple days* from an ordinary single night) applies equally to a satellite
range event. **Recommendation:** apply the suffix whenever `window_start != window_end`,
regardless of branch — this keeps `_calendar_event_title(run)` a single shared function with one
condition (`run.window_start != run.window_end`), rather than a ground-only special case.

### Anti-Patterns to Avoid

- **Duplicating the title-format string at both call sites:** `_project_calendar_event()` and
  `_set_run_status()` must not each independently f-string the title. See the Common Pitfalls
  section below — this is the single highest-risk mistake in this phase.
- **`url__startswith=f'CAMPAIGN:{run.pk}'` without the trailing colon:** creates a real
  cross-run collision for any pk that is a prefix of another pk's digits (e.g. pk=3 vs pk=34).
  Always use `f'CAMPAIGN:{run.pk}:'` (trailing colon) as the startswith prefix.
- **Re-querying `sun_event()` per-night results without handling the mid-window `ValueError`
  case:** if a `ValueError` occurs on, say, night 3 of a 5-night range (a plausible real
  scenario — DST transition edge cases, IERS data gaps), the current CR-01 contract re-raises
  immediately. This means the loop must decide (and this phase must test) whether nights 1-2's
  already-`insert_or_create_calendar_event()`-called events are left in place (partial
  projection) when night 3 raises. **Recommendation:** accept partial projection as the
  behavior (consistent with `insert_or_create_calendar_event()`'s own no-churn philosophy: each
  call is independently committed, there's no wrapping transaction today either) but the planner
  must decide explicitly rather than leave this implicit — flag as an Open Question below.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Create-or-update a `CalendarEvent` for the per-night events | A bespoke `get_or_create` call inline in `_project_calendar_event()` | `insert_or_create_calendar_event()` (`calendar_utils.py:318`) | Already the shared no-churn helper for all 4 producers of `CalendarEvent`s in this codebase (SYNC-04 idempotency); D-02/D-07 both explicitly route through it per CONTEXT.md |
| Inclusive date-range iteration | A manual `while` loop with date arithmetic | The `E - S + 1` / `timedelta(days=i)` idiom already used in `load_telescope_runs.py:_iter_run_nights()` (lines 54-89) | Established, tested precedent in the same codebase for exactly this problem shape |
| Combined single-key-or-prefix `CalendarEvent` lookup for `_set_run_status()` | Two separate `.filter()` calls unioned in Python | `CalendarEvent.objects.filter(Q(url=f'CAMPAIGN:{run.pk}') \| Q(url__startswith=f'CAMPAIGN:{run.pk}:'))` | One DB query, one queryset to iterate; `Q`/`F` are Django ORM primitives already partially imported (`F` is; `Q` needs adding) |

**Key insight:** this phase is deliberately small in *new* mechanism — the only genuinely new
code is the per-night loop and the multi-event lookup. Everything else (sun-event math, no-churn
create/update, staff gating, business-logic bypass guards) is explicitly preserved unchanged per
D-01/D-04/D-05's own wording ("its guard logic... is unaffected and must be preserved exactly").

## Common Pitfalls

### Pitfall 1: `_set_run_status()` silently strips the D-06 window suffix on status change

**What goes wrong:** `_set_run_status()` (campaign_views.py:754-760) currently recomputes the
title from scratch as `f'{prefix} {run.campaign.name}: {run.telescope_instrument}'`. Once
`_project_calendar_event()` starts writing titles with a window suffix for range runs
(`f'{run.campaign.name}: {run.telescope_instrument} (window {ws}..{we})'`), a
`mark_cancelled`/`mark_weather_failure` action recomputing the OLD (unsuffixed) format will
differ from the stored value, `insert_or_create_calendar_event()`'s diff will treat this as a
real change, save it, and the suffix is gone from every night's event — a silent regression of
the exact feature this phase adds.

**Why it happens:** the title format is currently duplicated (once at creation in
`_project_calendar_event()`, once at status-update in `_set_run_status()`) rather than shared,
and neither location currently needs to know about a window suffix, so this coupling isn't
visible until D-06 introduces it.

**How to avoid:** extract one function, e.g. `_calendar_event_title(run) -> str` (no prefix
logic — returns the base title including the conditional window suffix), called by
`_project_calendar_event()` directly and by `_set_run_status()` as
`f'{prefix} {_calendar_event_title(run)}'`. This makes the two sites structurally unable to
drift.

**Warning signs:** a test asserting `event.title == '[CANCELLED] Didymos 2026: Gemini-South
GMOS-S'` (no window text) after `mark_cancelled` on the Gemini FT-115 scenario would be evidence
this bug was not caught — the planner/checker should verify the `TestGeminiFtScenario` revision
explicitly asserts the window suffix survives the `[WEATHERED]`/`[CANCELLED]` transitions.

### Pitfall 2: `url__startswith` prefix collision between pk digit-substrings

**What goes wrong:** `CalendarEvent.objects.filter(url__startswith=f'CAMPAIGN:{run.pk}')`
(without a trailing colon) for `run.pk == 3` matches `'CAMPAIGN:34:2026-07-13'` too, because the
string `'CAMPAIGN:34:2026-07-13'` literally starts with the substring `'CAMPAIGN:3'`. This is a
real, not theoretical, risk in a dev DB where sequential pks are common (pk=3 and pk=34 both
exist as plausible fixture/test values).

**Why it happens:** `startswith` is a pure string operation with no awareness of the `:`
delimiter's semantic role as a field separator.

**How to avoid:** always build the prefix as `f'CAMPAIGN:{run.pk}:'` — the trailing colon is
what makes `'CAMPAIGN:3:'` NOT a prefix of `'CAMPAIGN:34:...'` (since `'CAMPAIGN:34...'`'s 11th
character is `'4'`, not `':'`).

**Warning signs:** a backfill command or `_set_run_status()` test using two-digit-vs-one-digit
pks in the same test DB (very easy to hit by accident since Django auto-increments pks) that
shows cross-run event mutation.

### Pitfall 3: Test fixture default window is single-night — three "must stay 0" tests need no code change but must be re-verified they still pass unmodified

**What goes wrong:** assuming all four debug-spec-flagged tests need editing risks accidentally
also touching `test_approve_tbd_run_creates_no_calendar_event` (356),
`test_approve_without_telescope_instrument_creates_no_calendar_event` (363), and
`test_sun_event_valueerror_skips_projection_without_reverting_approval` (370) — all three use
`_make_pending_run()`'s **default** window (`window_start=window_end=date(2026, 8, 1)`, i.e.
single-night, `n_nights == 1`), so the rewritten loop runs exactly once for them, identically to
today. They should require literally zero code changes and should be run explicitly (not just
assumed) as a regression check that the rewrite didn't perturb the single-night path.

**Why it happens:** the debug doc's "tests that must stay count==0" list is correct, but a
planner skimming quickly could mistakenly "fix" these too, since they're textually adjacent to
the ones that do need changing.

**How to avoid:** explicitly list these three in the plan's non-goals / "must still pass
byte-identical" verification step.

**Warning signs:** a diff touching lines 356-380 of `test_campaign_approval.py` beyond
docstring/comment updates.

### Pitfall 4: Stale docstrings/comments describing pre-fix behavior (multiple locations)

**What goes wrong:** several docstrings and inline comments explicitly assert the OLD
"range... never projects" contract as if it were still true, and will silently become
misleading (not broken, just wrong) if left unedited:
- `_project_calendar_event()`'s own docstring (392-407): "False when projection was skipped by
  design (range/TBD run, or missing telescope_instrument/site)" — "range" must be removed from
  this list.
- The guard's inline comment (408-411): "only project a single concrete night... A range, TBD
  run, or unresolved site simply doesn't get a CalendarEvent yet" — needs rewriting to reflect
  D-01/D-02.
- `_set_run_status()`'s docstring (720-725): "A run whose window/site never projected a
  CAMPAIGN:{pk} event (a range/TBD run, or one with an unresolved site)..." — "range" is no
  longer accurate; only TBD/unresolved-site/missing-telescope_instrument runs never project.
- Test module docstring (lines 1-16) and `TestCalendarProjection`/`TestRunStatusChange`/
  `TestGeminiFtScenario` class docstrings (301-306, 384-391, 2130-2137) all describe the
  pre-fix "range... skipped by design" contract.

**Why it happens:** these were accurate when written (Phase 19/23); this phase changes the
underlying behavior without an automatic doc-sync mechanism.

**How to avoid:** treat every one of the above as an in-scope edit for this phase's tasks — the
plan-checker should flag any plan that fixes the code but leaves these docstrings/comments
asserting the old contract, since a future reader (including a future GSD phase) would be
actively misled.

**Warning signs:** `grep -n "skipped by design\|range.*no.*event\|never project" solsys_code/campaign_views.py solsys_code/tests/test_campaign_approval.py` still returning matches after the phase ships.

## Runtime State Inventory

Not applicable — this is a bug-fix/feature-extension phase, not a rename/refactor/migration.
No stored-data keys, service configs, OS-registered state, or secrets are being renamed. (D-07's
backfill command *writes new* `CalendarEvent` rows for already-approved runs, but that is
ordinary application data creation through the normal code path, not a migration of existing
identifiers — covered under Code Examples/backfill design below, not this section.)

## Code Examples

### `_project_calendar_event()` current guard (verified verbatim, line 412)

```python
# Source: solsys_code/campaign_views.py:412 (read in full, live source, 2026-07-17)
if not (run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end):
    return False
```

### D-01's required guard (locked)

```python
# Per 25-CONTEXT.md D-01 (locked) -- drop the equality clause, ADD window_end truthiness
if not (run.telescope_instrument and run.site and run.window_start and run.window_end):
    return False
```

Note: the model already enforces `window_start`/`window_end` are either both NULL or both
non-NULL via `campaign_run_window_start_end_null_together` (`CheckConstraint`,
`solsys_code/models.py:154-160`), confirmed by direct read. So the `window_end` truthiness check
is defense-in-depth against a hypothetical partially-migrated row, not a scenario the DB schema
can currently produce for a saved row — D-01's own docstring is correct that it's still required
(a view could theoretically construct an in-memory `CampaignRun` instance without saving, though
no current code path does this).

### `_set_run_status()`'s current single-key existence check (verified verbatim, lines 752-760)

```python
# Source: solsys_code/campaign_views.py:752-760 (read in full, live source, 2026-07-17)
if CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists():
    prefix = _RUN_STATUS_CALENDAR_PREFIX[new_run_status]
    insert_or_create_calendar_event(
        {'url': f'CAMPAIGN:{run.pk}'},
        fields={
            'title': f'{prefix} {run.campaign.name}: {run.telescope_instrument}',
            'description': f'{run.observation_details}\nRun status: {run.get_run_status_display()}',
        },
    )
```

### D-04's required rewrite (imports needed: add `Q` to the existing `from django.db.models
import Case, CharField, Count, EmailField, F, Value, When` on line 24 — `F` is already imported,
`Q` is not)

```python
# Recommended shape (D-04 locked requirement: find-and-update EVERY matching event;
# exact filter construction is Claude's Discretion per CONTEXT.md)
matching_events = CalendarEvent.objects.filter(
    Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')
)
if matching_events.exists():
    prefix = _RUN_STATUS_CALENDAR_PREFIX[new_run_status]
    for event in matching_events:
        insert_or_create_calendar_event(
            {'url': event.url},
            fields={
                # _calendar_event_title() -- NEW shared helper, see Pitfall 1 --
                # must be used here too, not a re-derived f-string.
                'title': f'{prefix} {_calendar_event_title(run)}',
                'description': f'{run.observation_details}\nRun status: {run.get_run_status_display()}',
            },
        )
```

### Existing `E - S + 1` inclusive-range precedent (verified verbatim,
`load_telescope_runs.py:54-89`, the pattern D-02 mirrors)

```python
# Source: solsys_code/management/commands/load_telescope_runs.py:76-89
if parsed.day2 < parsed.day1:
    raise ValueError(f'Cross-month run ranges not yet supported in Phase 3: {parsed!r}')
n_nights = parsed.day2 - parsed.day1 + 1
...
first_night = date(parsed.year, parsed.month, parsed.day1)
return [first_night + timedelta(days=i) for i in range(n_nights)]
```
`CampaignRun.window_start`/`window_end` are already `date` objects (not the raw day-of-month
ints `load_telescope_runs.py`'s `ParsedRun` uses), so the equivalent for this phase is simpler:
`n_nights = (run.window_end - run.window_start).days + 1` needs no month-rollover handling at
all (Python `date` subtraction already spans month/year boundaries correctly) — this phase does
NOT need `load_telescope_runs.py`'s `ESO_NOON_TO_NOON_SITES` special-casing or cross-month
guard; those exist only because `ParsedRun` stores raw `day1`/`day2` integers within a single
parsed month/year, a different representation than `CampaignRun`'s `DateField` pair.

### Real dev-DB verification of the D-07 backfill target (queried directly, 2026-07-17)

```
$ sqlite3 src/fomo_db.sqlite3 "SELECT id, campaign_id, telescope_instrument, site_id, window_start, window_end, approval_status, run_status, site_needs_review FROM solsys_code_campaignrun WHERE id=34;"
34|1|Gemini South/GMOS-s|17|2026-07-13|2026-07-16|approved|requested|0

$ sqlite3 src/fomo_db.sqlite3 "SELECT id, name, obscode, observations_type, timezone FROM solsys_code_observatory_observatory WHERE id=17;"
17|Gemini South Observatory, Cerro Pachon|I11|0|America/Santiago

$ sqlite3 src/fomo_db.sqlite3 "SELECT id, url, title, start_time, end_time FROM tom_calendar_calendarevent WHERE url LIKE 'CAMPAIGN:34%';"
(no rows)
```
`observations_type=0` = `Observatory.OPTICAL_OBSTYPE` (verified against
`solsys_code_observatory/models.py:28-36`) — confirms D-05's assumption that the real Gemini
I11 case is unambiguously ground, not satellite. This row exactly matches D-07's stated
qualifying criteria (`APPROVED`, ground site resolved, `window_start != window_end`, no existing
`CAMPAIGN:34*` event) and will get exactly 4 per-night events once the backfill command runs.

### Recommended D-07 backfill command query shape

```python
# NEW FILE: solsys_code/management/commands/backfill_range_calendar_events.py
from django.db.models import F, Q
from tom_calendar.models import CalendarEvent
from solsys_code.campaign_views import _project_calendar_event  # cross-module underscore-
    # prefixed import -- an accepted existing pattern in this codebase (see the pending
    # 2026-07-02 todo about calendar_utils.py's underscore helpers already having 3 consumers)
from solsys_code.models import CampaignRun

candidates = CampaignRun.objects.filter(
    approval_status=CampaignRun.ApprovalStatus.APPROVED,
    site__isnull=False,
    window_start__isnull=False,
).exclude(window_start=F('window_end'))

for run in candidates:
    already_has_event = CalendarEvent.objects.filter(
        Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')
    ).exists()
    if already_has_event:
        continue
    # ... call _project_calendar_event(run), same try/except ValueError swallow pattern
    # approve() already uses, counters incremented, per this codebase's established
    # created/updated/unchanged/skipped stdout-summary convention (see
    # sync_gemini_observation_calendar.py's `counters` dict + final self.stdout.write()).
```
Note this pre-filter is a *summary-scoping* choice, not a correctness requirement —
`insert_or_create_calendar_event()`'s own no-churn idempotency means re-running
`_project_calendar_event()` on an already-populated run would just leave it unchanged anyway.
D-07 locks the pre-filter as the search criterion regardless.

**No `--dry-run` precedent exists anywhere in this codebase today** (checked: `grep -rn
"dry.run\|dry_run\|store_true" solsys_code/management/` returns nothing). Recommending one
anyway is `[ASSUMED]` — a broadly standard Django management-command convention for a
production-data-writing one-off script, not something this codebase has established before.
Flag for discuss/planning confirmation rather than treating as settled.

## State of the Art

Not applicable — no external library/ecosystem changes are involved in this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A `--dry-run` flag is a good addition to the new backfill command | Code Examples / D-07 | Low — CONTEXT.md already marks this "Claude's Discretion"; if the user prefers no dry-run flag, dropping it is a trivial follow-up edit with no data-safety consequence either way, since `insert_or_create_calendar_event()` is itself idempotent |
| A2 | The satellite branch's single range event should ALSO get the D-06 window-title suffix | Architecture Patterns / Pattern 2 | Medium — if wrong, a satellite range CalendarEvent's title would look identical to an ordinary single-night event, losing the "this is one awarded allocation" framing D-06 wants for ground ranges; low functional risk (cosmetic only), but should be confirmed explicitly during planning, not silently decided by the researcher |
| A3 | Partial projection (some nights' events created, then a later night's `sun_event()` ValueError aborts the rest) is acceptable behavior, matching the existing per-call (not per-run-transaction) commit model | Anti-Patterns to Avoid | Medium — if the user actually wants all-or-nothing per run, this needs a `transaction.atomic()` wrap the current codebase has never used for this helper; flagged as an Open Question, not assumed as final |

## Open Questions

1. **Should the satellite branch's title get the D-06 window suffix for a range window?**
   - What we know: D-06's exact locked wording is written around "per-night events"; the
     satellite branch never produces per-night events (D-05, unchanged).
   - What's unclear: whether the underlying intent (distinguish an awarded-allocation range
     from an ordinary single night) was meant to apply to satellite ranges too, or whether D-06
     is deliberately scoped to ground-only because satellite ranges are a much rarer/theoretical
     case in this codebase's real data (all currently-known real space-mission rows in the
     3I/ATLAS sheet are TBD/window-narrowing, not resolved ranges, per Phase 20's
     `pending_narrowing_runs` bucketing).
   - Recommendation: apply uniformly (Assumption A2 above) via one shared
     `_calendar_event_title(run)` helper — simpler code, and CONTEXT.md's Deferred Ideas
     section only excludes "new visual/UI treatment beyond title text," not this.

2. **Partial-projection behavior when `sun_event()` raises mid-loop for a range run.**
   - What we know: `_project_calendar_event()`'s existing CR-01 contract re-raises `ValueError`
     immediately (never swallowed internally); callers (`approve()`, `resolve_site()`) decide
     revert-vs-retry behavior. `insert_or_create_calendar_event()` calls are independently
     committed (no surrounding transaction today, for any of its 4 call sites).
   - What's unclear: whether a mid-range `ValueError` (e.g. a genuine high-latitude/no-crossing
     night partway through a long window) should leave already-created earlier nights' events in
     place, or whether the planner should wrap the per-night loop in `transaction.atomic()` to
     roll back the whole run's partial projection.
   - Recommendation: keep the existing no-transaction-wrap behavior (Assumption A3) for
     consistency with every other `insert_or_create_calendar_event()` call site in this
     codebase, but the plan MUST include an explicit test exercising this case (e.g. mock
     `sun_event` with a `side_effect` list that raises on the 3rd call) so the chosen behavior is
     locked in by a test, not left implicit.

3. **Exact final decision on the `--dry-run` flag for the backfill command (D-07 discretion item).**
   - What we know: no precedent exists in this codebase (checked directly, see Code Examples).
   - What's unclear: whether the user wants one at all, given it's genuinely novel here.
   - Recommendation: include it — low cost, standard practice for a one-off command that writes
     production `CalendarEvent` rows for real (not test) `CampaignRun`s including the real
     GS-2026A-FT-115 row — but flag explicitly for a quick confirm during planning rather than
     treating as locked.

## Environment Availability

Skipped — this phase introduces no new external dependencies. `astropy`/`erfa` (via the
already-imported `sun_event()`) and Django's `Q`/`F` ORM primitives are already present and in
active use elsewhere in this exact codebase; only a new *import statement* is needed
(`Q` into `campaign_views.py`), not a new installed package.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django's built-in test runner (`django.test.TestCase`), NOT pytest — `pyproject.toml`'s `testpaths = ["tests", "src", "docs"]` deliberately excludes `solsys_code/` (per CLAUDE.md "Testing" section) |
| Config file | none — Django settings module `src.fomo.settings`, invoked via `./manage.py test` |
| Quick run command | `./manage.py test solsys_code.tests.test_campaign_approval.TestCalendarProjection solsys_code.tests.test_campaign_approval.TestRunStatusChange solsys_code.tests.test_campaign_approval.TestGeminiFtScenario solsys_code.tests.test_campaign_approval.TestSitesNeedingReview -v 2` |
| Full suite command | `./manage.py test solsys_code` (per CLAUDE.md) |

### Phase Requirements → Test Map

This phase has no `REQUIREMENTS.md`-mapped IDs (gap-closure phase added directly to
STATE.md/ROADMAP.md from a `/gsd-debug` diagnosis). Using phase-local IDs instead:

| ID | Behavior | Test Type | Automated Command | File Exists? |
|----|----------|-----------|-------------------|-------------|
| FIX-01 | Guard admits ground range-window runs (drops `window_start == window_end`) | unit (Django TestCase) | `./manage.py test solsys_code.tests.test_campaign_approval.TestCalendarProjection.test_approve_range_run_creates_one_event_per_night` (revised from `..._creates_no_calendar_event`) | ✅ exists, needs revision |
| FIX-02 | Ground branch creates one dip-corrected event per night, `count == n_nights` (15 for the 8/1..8/15 fixture, 4 for GS-2026A-FT-115) | unit | Same test as FIX-01, plus `TestGeminiFtScenario.test_gemini_ft115_range_window_flows_through_same_mechanism_...` (revised) | ✅ exists, needs revision |
| FIX-03 | First/last night's start/end times match `sun_event()` for `window_start`/`window_end` exactly (guards the first-night-only regression the debug doc specifically warned about) | unit | New assertion inside the revised FIX-01 test, or a new dedicated test | ❌ Wave 0 (new assertion) |
| FIX-04 | Satellite branch stays a single whole-day-span event with the bare key, for both single-night and range windows | unit | `TestCalendarProjection.test_approve_single_night_space_run_creates_midnight_utc_placeholder_event` (existing, unchanged) + new `test_approve_range_run_space_site_creates_single_whole_day_span_event` | ❌ Wave 0 (new test) |
| FIX-05 | `_set_run_status()` updates EVERY matching night's event with the `[CANCELLED]`/`[WEATHERED]` prefix, preserving the window suffix | unit | `TestRunStatusChange.test_mark_range_window_run_...` (revised) + `TestGeminiFtScenario`'s mark_weather_failure/mark_cancelled assertions (revised) | ✅ exists, needs revision |
| FIX-06 | `resolve_site()`'s retroactive projection path also creates N events and reports the "added to the calendar" message for a range run | unit | `TestSitesNeedingReview.test_resolve_range_tbd_run_clears_flag_with_no_calendar_event` (revised — also needs renaming, it's not actually a TBD case) | ✅ exists, needs revision |
| FIX-07 | Three genuinely-unaffected guard-exclusion tests (TBD, missing telescope_instrument, `sun_event` ValueError) still pass byte-identical | regression | `TestCalendarProjection.test_approve_tbd_run_creates_no_calendar_event`, `test_approve_without_telescope_instrument_creates_no_calendar_event`, `test_sun_event_valueerror_skips_projection_without_reverting_approval` | ✅ exists, no change expected |
| FIX-08 | Backfill command finds CampaignRun pk=34-shaped candidates, creates N events, skips already-populated / non-qualifying runs | unit + a manual/smoke run against the real dev DB (D-07's stated purpose is fixing the real pk=34 row) | New test class + `./manage.py backfill_range_calendar_events [--dry-run]` run manually against `src/fomo_db.sqlite3` | ❌ Wave 0 (new command + new tests) |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_campaign_approval -v 2`
- **Per wave merge:** `./manage.py test solsys_code` (full app suite) + `ruff check .` +
  `ruff format --check .`
- **Phase gate:** Full suite green before `/gsd-verify-work`; additionally, per this project's
  "Project" constraints section, sunset/sunrise accuracy is not re-derived by this phase (it
  reuses `sun_event()` unchanged) so no new LCO-skycalc cross-check is needed.

### Wave 0 Gaps
- [ ] New assertions inside the revised `TestCalendarProjection` range test — first/last night
      span verification (FIX-03)
- [ ] New test: satellite range single-event regression guard (FIX-04)
- [ ] New management command `solsys_code/management/commands/backfill_range_calendar_events.py`
      + its test class (FIX-08)
- [ ] No new fixtures/conftest needed — `CampaignApprovalTestBase` and its subclasses' existing
      `ground_site`/`gemini_south` Observatory fixtures already cover every scenario this phase
      needs (all Tier-1-resolvable, no live MPC calls required)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | Unchanged — `StaffRequiredMixin` gating on `CampaignRunDecisionView` is untouched by this phase |
| V3 Session Management | No | Unchanged |
| V4 Access Control | Yes (verify unchanged, not modify) | `StaffRequiredMixin` + the existing business-logic-bypass guards (`run.approval_status != APPROVED` checks in `_set_run_status()`/`_resolve_site()`) — this phase must not weaken these; D-04 explicitly requires "guard logic... is unaffected and must be preserved exactly" |
| V5 Input Validation | Yes (verify unchanged, not modify) | `_ACTION_TO_RUN_STATUS`/`_RUN_STATUS_CALENDAR_PREFIX` fixed whitelists (never derived from raw POST text) — this phase adds no new user-controllable input paths; the new `url` key format is built entirely from `run.pk` (an internal integer, not user text) and `date.isoformat()` (a `DateField` value, not free text) |
| V6 Cryptography | No | Not applicable — no secrets/crypto in this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Business-logic bypass (staff-only action reachable without proper state) | Elevation of Privilege | Already mitigated by existing `approval_status`/`site_needs_review` server-side guards in `_set_run_status()`/`_resolve_site()`, unchanged by this phase |
| Cross-run `CalendarEvent` mutation via a malformed/colliding `url` key | Tampering | The new `f'CAMPAIGN:{pk}:{date.isoformat()}'` key is built entirely server-side from trusted model fields (`pk`, `window_start`/`window_end`), never from request input — no injection surface. The trailing-colon `url__startswith` prefix (Pitfall 2) is a *correctness* bug risk, not a security vulnerability, since both `pk` values in any collision scenario are still legitimate `CampaignRun` rows the requesting staff user is already authorized to modify via this same endpoint |
| Backfill command run against production data with unintended side effects | Tampering (accidental, not adversarial) | Mitigated by keeping the command idempotent (routes through `insert_or_create_calendar_event()`'s no-churn contract) and by the recommended `--dry-run` flag (Assumption A1) — this is an ops-safety concern, not a traditional STRIDE security threat, since the command is CLI-only and requires shell access already equivalent to full DB access |

## Sources

### Primary (HIGH confidence — direct source reads and live DB queries this session, 2026-07-17)
- `solsys_code/campaign_views.py` (full file, read via multiple targeted reads) — `_project_calendar_event()` (392-455), `_set_run_status()` (708-763), `_resolve_site()` (577-706), `CampaignRunDecisionView.post()` (472-575), imports (1-63)
- `solsys_code/calendar_utils.py:297-378` — `_update_or_unchanged()`, `insert_or_create_calendar_event()`
- `solsys_code/telescope_runs.py:251-299` — `sun_event()`
- `solsys_code/management/commands/load_telescope_runs.py` (full file) — `_iter_run_nights()`, `Command.handle()`
- `solsys_code/models.py:31-165` — `CampaignRun` model, constraints
- `solsys_code/solsys_code_observatory/models.py:28-36` — `Observatory.OBSTYPE_CHOICES`
- `solsys_code/tests/test_campaign_approval.py` (targeted reads: 1-146, 300-518, 815-1054, 2089-2208) — all 7 flagged test methods plus surrounding fixture/class context
- `~/venv/fomo_venv/lib/python3.12/site-packages/tom_calendar/models.py` — `CalendarEvent` field definitions (`url = URLField(blank=True, default="")`, non-nullable `start_time`/`end_time`)
- `src/fomo_db.sqlite3` (direct `sqlite3` queries) — live confirmation of `CampaignRun` pk=34's exact field values and its zero existing `CalendarEvent` rows
- `solsys_code/management/commands/import_campaign_csv.py`, `sync_gemini_observation_calendar.py` — management command argument/output-summary conventions (no `--dry-run` precedent found anywhere in the codebase, confirmed via `grep`)
- `CLAUDE.md` (live file, read in full at session start) — confirmed `campaign_views.py`/`calendar_utils.py` are NOT among the four demo-notebook-parity modules (`telescope_runs.py`, `load_telescope_runs.py`, `sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`)

### Secondary (MEDIUM confidence)
- `.planning/debug/range-window-calendar-event.md` — the `/gsd-debug` diagnosis; treated as authoritative for root cause (D-01) and superseded for date-math approach (per CONTEXT.md D-02-D-04), all line numbers independently re-verified this session (found accurate to within 1-2 lines)
- `.planning/phases/19-window-schema-migration/19-CONTEXT.md` — D-06 provenance, cited but not re-read in full this session (already summarized accurately in both CONTEXT.md and the debug doc)

### Tertiary (LOW confidence)
- None — no WebSearch/external-doc lookups were performed this session; all findings are direct codebase/DB verification (no new external library or API is involved in this phase)

## Metadata

**Confidence breakdown:**
- Root cause / guard fix (D-01): HIGH — line-verified verbatim against live source, matches debug doc exactly
- Ground-branch date math (D-02/D-03): HIGH for the *requirement* (locked in CONTEXT.md), MEDIUM for the *exact code shape* (Claude's Discretion — the loop-based unification is a recommendation, not verified against any existing identical precedent in this exact form)
- `_set_run_status()` rewrite (D-04) + the title-drift pitfall: HIGH — both the current code and the drift risk were derived from direct source reads, not inference
- Test revision counts (15/15/4/15, not `count==1`): HIGH — every window date range was read directly from the live test file and arithmetic re-derived (8/1..8/15 = 15 nights; 7/13..7/16 = 4 nights), correcting the debug doc's stale assumption
- Backfill command design (D-07): MEDIUM — query shape and cross-module import precedent are HIGH confidence (verified), but the `--dry-run` flag recommendation is explicitly `[ASSUMED]` (no in-repo precedent)
- CLAUDE.md demo-notebook-parity non-applicability: HIGH — read the live CLAUDE.md directly this session, independently confirmed CONTEXT.md's claim rather than trusting it

**Research date:** 2026-07-17
**Valid until:** Until this phase's code lands (this is a same-session line-number-anchored
research artifact for an internal bug fix, not a general library-currency estimate — if
`campaign_views.py` or `test_campaign_approval.py` changes again before this phase is planned/
executed, re-verify line numbers before writing `<action>` blocks).
