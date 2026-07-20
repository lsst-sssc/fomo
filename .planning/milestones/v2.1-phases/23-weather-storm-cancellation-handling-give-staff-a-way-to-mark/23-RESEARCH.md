# Phase 23: Weather/Storm Cancellation Handling - Research

**Researched:** 2026-07-16
**Domain:** Django app-internal feature work — CalendarEvent title-prefix conventions, django-tables2 row actions, staff-facing status-change workflow. No new external services or packages.
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Classical-schedule cancellation (load_telescope_runs)**
- **D-01:** Staff mark a classical run cancelled by editing the source schedule file to add
  the already-recognized `cancelled` status word/parenthetical (`_resolve_status()`'s
  `KNOWN_STATUSES`), then re-running `load_telescope_runs` against it. This is the existing
  idempotent create-or-update path (`insert_or_create_calendar_event`) — this phase does NOT
  add a new command or UI action for classical events, only wires the already-embedded
  `Status: cancelled` value into a visible treatment.
- **D-02:** Visual treatment is a title prefix, matching the LCO/SOAR sync's existing pattern
  (`WINDOW_EXPIRED`→`[EXPIRED]`, `CANCELED`→`[CANCELLED]`, etc. in
  `sync_lco_observation_calendar.py`). `KNOWN_STATUSES` currently has only one relevant word
  (`cancelled`) — classical events get a single `[CANCELLED]` prefix; there is no
  classical-schedule equivalent of `WEATHER_TECH_FAILURE` (see D-03 for why that distinction
  only applies to `CampaignRun`).

**CampaignRun.run_status staff UI + calendar sync**
- **D-03:** `CANCELLED` and `WEATHER_TECH_FAILURE` get two DISTINCT title prefixes on the
  linked `CalendarEvent` (e.g. `[CANCELLED]` vs `[WEATHERED]` — exact wording is Claude's
  discretion, but they must render differently), not one shared label.
- **D-04:** The status-change action lives on the approval queue's **Decided** table
  (already-approved runs). `ApprovalQueueTable`'s Decided table currently renders
  `show_actions=False` and `render_actions()` returns `''` unconditionally for it — this
  phase adds the first action any Decided row has ever had. Exact control shape (dropdown+
  submit vs per-status buttons) is Claude's discretion.
- **D-05:** Setting `run_status` to `CANCELLED`/`WEATHER_TECH_FAILURE` on an already-approved
  run **updates the existing `CAMPAIGN:{pk}` `CalendarEvent` in place** (title/description),
  reusing `insert_or_create_calendar_event()`'s no-churn update path — it does NOT delete the
  event.

**Gemini FT program visibility (informational only)**
- **D-06:** GS-2026A-FT-115 is represented as an informational `CampaignRun` row under the
  existing "Didymos 2026" `TargetList` (pk=1), using the `window_start`/`window_end` pair —
  NOT a real Gemini `ObservationRecord`/ODB API submission. Real ODB sync stays out of scope.
- **D-07:** This CampaignRun entry is subject to the SAME run_status mechanism as D-03/D-04/
  D-05 — no special-casing.

### Claude's Discretion
- Exact wording/format of the `[WEATHERED]`-style prefix for `WEATHER_TECH_FAILURE` (D-03).
- Exact UI control shape for the Decided-table status-change action (dropdown, per-status
  buttons, etc.) (D-04).
- Whether the new `[WEATHERED]`/`[CANCELLED]` prefixes need to be added to
  `calendar_display_extras.py`'s `_TERMINAL_PREFIXES` tuple to also pick up the existing
  status box-shadow ring treatment (Phase 8/9), or whether a title prefix alone satisfies
  "visually reflected" for this phase.
- Site resolution for the Gemini CampaignRun entry (D-06) — whether "Gemini-South" already
  resolves via the existing MPC candidate pool / `resolve_site()` or needs a new local
  `Observatory` row created first.

### Deferred Ideas (OUT OF SCOPE)
- Real Gemini `ObservationRecord`/ODB API sync for FT-115 (or any Gemini program) — explicitly
  out of scope per D-06. `sync_gemini_observation_calendar.py` already exists for genuinely-
  submitted Gemini programs; extending it to represent proposal-level "hours awarded, not yet
  scheduled" allocations is real new scope for a future phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

This phase has **no REQ-IDs mapped** in `.planning/REQUIREMENTS.md` — it was added organically
outside the v2.1 milestone scope (`SCHED`/`ASSET`/`IMPORT`/`SITE`/`VIEW` requirements, Phases
18–22), triggered by a real incoming storm. The planner should treat CONTEXT.md's D-01
through D-07 as the effective requirement set for this phase; there is no traceability row to
add to REQUIREMENTS.md unless the operator later decides to formalize this as a v2.2 item.
</phase_requirements>

## Summary

This phase closes a real visibility gap across two independent, previously-unconnected
subsystems, both of which already have 90% of the machinery needed — this is a wiring/UI
phase, not a new-subsystem phase.

For classical (Magellan/NTT/FTS) runs, `_resolve_status()` in `telescope_runs.py` already
parses a `cancelled` status word and `load_telescope_runs.py` already embeds
`Status: cancelled` into `CalendarEvent.description` — it just never touches `title`. The
fix is a 3-4 line conditional in `load_telescope_runs.py`'s `handle()` loop, before the
existing `title = f'{parsed.telescope} {parsed.instrument}'` line, mirroring
`sync_lco_observation_calendar.py`'s `_title_for()`/`_FAILURE_PREFIX_BY_STATUS` pattern
byte-for-byte (`[CANCELLED] {telescope} {instrument}`).

For `CampaignRun`, `run_status` already has `CANCELLED`/`WEATHER_TECH_FAILURE` choices, a
badge renderer (`RUN_STATUS_BADGE_CLASSES`), and is *already editable* in Django admin (it is
NOT in `CampaignRunAdmin.readonly_fields`, unlike `approval_status`). What's missing is a
staff-facing, non-admin entry point. The natural fit is a new branch on
`CampaignRunDecisionView.post()` (mirroring the existing `resolve_site` branch's
conditional-update + business-logic-bypass-guard pattern) plus a new small action rendered on
`ApprovalQueueTable`'s Decided table. The calendar-sync half reuses
`insert_or_create_calendar_event()`, but **only when a `CAMPAIGN:{pk}` `CalendarEvent` already
exists** — `_project_calendar_event()` only ever creates one for a resolved-site, single-night
(`window_start == window_end`) run, so a range/TBD/unresolved-site run (including the Gemini
FT-115 entry itself, whose window is a 4-day range) has **no event to update**. This is the
single most important pitfall this research surfaces (see Common Pitfalls).

For the display layer, the investigation resolved the `_TERMINAL_PREFIXES` discretion item
definitively: `'[CANCELLED]'` is **already** in that tuple (inherited from the LCO sync's own
vocabulary), so both D-01/D-02's classical `[CANCELLED]` prefix and D-03's `CampaignRun`
`[CANCELLED]` prefix get the existing status box-shadow ring **for free, no code change**.
Only the new `[WEATHERED]` prefix (for `WEATHER_TECH_FAILURE`) needs to be added to
`_TERMINAL_PREFIXES` to get the same ring.

For Gemini site resolution, MPC obscode `I11` ("Gemini South Observatory, Cerro Pachon") is
confirmed against the official MPC ObsCodes list — a real, resolvable, ground-based site.
`resolve_site('I11')` (or the approval queue's fuzzy-match widget, which already indexes the
full MPC bulk pool) will Tier-2-resolve it without any new local `Observatory` row needing to
be pre-created, and will pick up a real IANA timezone automatically via the existing
`MPCObscodeFetcher.to_observatory()` timezone backfill (quick task `260716-h8c`).

**Primary recommendation:** Extend the existing title-prefix idiom (already used by
`sync_lco_observation_calendar.py`, already partially wired for `[CANCELLED]` in
`_TERMINAL_PREFIXES`) into `load_telescope_runs.py` and a new `CampaignRunDecisionView.post()`
branch; guard the calendar-sync half so it only updates a `CAMPAIGN:{pk}` event that already
exists, never fabricates one for a run that never had a projected event to begin with.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Classical run `[CANCELLED]` title prefix | Backend (management command: `load_telescope_runs.py`) | — | Pure data-transform before the existing `insert_or_create_calendar_event()` call; no view/template involved |
| CampaignRun status-change action (D-04) | Backend (`CampaignRunDecisionView.post()`) | Presentation (`ApprovalQueueTable.render_actions()`) | Business-logic guard (only APPROVED rows) belongs server-side; the button/form is a template-layer concern that must match the existing per-row-form HTMX-free POST pattern |
| CAMPAIGN CalendarEvent in-place update (D-05) | Backend (`campaign_views.py`, reusing `calendar_utils.insert_or_create_calendar_event()`) | — | Same helper both sync commands and the approval flow already share; no new persistence layer |
| Status box-shadow ring (Claude's Discretion #1) | Presentation (`calendar_display_extras.py` templatetag + `calendar.html` partial) | — | Already-built Phase 8/9 CSS-in-Python constant lookup; this phase only needs a 1-line tuple edit |
| Gemini FT-115 CampaignRun row + site resolution (D-06) | Database/Storage (data row via existing `Observatory`/`CampaignRun` models) | Backend (`resolve_site()`/`build_site_candidates()`) | No new model fields or migrations; purely a data-entry + existing-resolver exercise |

## Standard Stack

No new libraries, packages, or services are required for this phase. Every capability is built
on already-installed, already-used-in-this-exact-code-path dependencies:

| Library | Version | Purpose | Why no change needed |
|---------|---------|---------|-----------------------|
| Django | (project-pinned, 2.1+ via TOM Toolkit) | Views, admin, ORM | `CampaignRunDecisionView`, `CampaignRun` model, `CalendarEvent` model all already exist |
| django-tables2 | (project-pinned) | `ApprovalQueueTable` row rendering | `render_actions()`/`render_site()` override pattern already established (Phase 16/21/22) |
| astropy | (project-pinned) | `sun_event()` (unaffected by this phase — no new calendar projections are added, only title/description edits on existing events) | `_project_calendar_event()` is not modified by this phase |

**Installation:** none required.

## Package Legitimacy Audit

**N/A — this phase installs no external packages.** All work is confined to existing modules
(`solsys_code/management/commands/load_telescope_runs.py`, `solsys_code/campaign_views.py`,
`solsys_code/campaign_tables.py`, `solsys_code/templatetags/calendar_display_extras.py`) using
already-installed dependencies.

## Architecture Patterns

### System Architecture Diagram

```
Classical-schedule path (D-01/D-02):
  Staff edits schedule file (adds "(cancelled)")
        │
        ▼
  `load_telescope_runs <file>` (management command)
        │  parse_run_line() → _resolve_status() → status='cancelled'  [already exists]
        ▼
  NEW: if status == 'cancelled': title = f'[CANCELLED] {telescope} {instrument}'
        │  else: title = f'{telescope} {instrument}'                  [existing else-branch]
        ▼
  insert_or_create_calendar_event(lookup={telescope,instrument,start_time},
                                   fields={title, description, end_time})  [already exists]
        ▼
  CalendarEvent row created/updated in place (no-churn)
        │
        ▼
  calendar.html partial → status_border_css(event.title) → box-shadow ring
        (works automatically: '[CANCELLED]' already in _TERMINAL_PREFIXES)


CampaignRun status-change path (D-03/D-04/D-05):
  Staff clicks "Mark Cancelled" / "Mark Weathered" on a Decided-table row
        │  POST campaigns:decide  action=mark_cancelled|mark_weather_failure
        ▼
  CampaignRunDecisionView.post()
        │  NEW branch, mirrors resolve_site()'s guard style:
        │  1. business-logic guard: run.approval_status must be APPROVED
        │  2. conditional .update(run_status=...) (staleness-safe, mirrors approve/reject)
        ▼
  NEW: look up existing CalendarEvent(url=f'CAMPAIGN:{pk}')
        │
        ├─ event exists ──► insert_or_create_calendar_event(
        │                     lookup={'url': f'CAMPAIGN:{pk}'},
        │                     fields={'title': f'{prefix} {title}', 'description': ...})
        │                   → updates in place (D-05), no delete
        │
        └─ no event exists ──► skip silently (range/TBD/unresolved-site run never had
                                 a projected event — nothing to update, matches
                                 _project_calendar_event()'s own skip-by-design semantics)
        ▼
  calendar.html partial → status_border_css() → box-shadow ring IF prefix in
        _TERMINAL_PREFIXES ('[CANCELLED]' already there; '[WEATHERED]' needs adding)
```

### Recommended Project Structure

No new files. Changes land in:
```
solsys_code/
├── management/commands/load_telescope_runs.py   # D-01/D-02: title-prefix branch
├── campaign_views.py                             # D-04/D-05: new decide-action branch
├── campaign_tables.py                             # D-04: new Decided-table action rendering
├── templatetags/calendar_display_extras.py        # add '[WEATHERED]' to _TERMINAL_PREFIXES
└── tests/
    ├── test_load_telescope_runs.py                # extend test_event_fields_set_from_parsed_run-style tests
    ├── test_campaign_approval.py                   # new TestRunStatusChange-style test class
    └── test_calendar_display_extras.py             # extend TERMINAL_BOX_SHADOW parametrized tests
docs/notebooks/pre_executed/
└── load_telescope_runs_demo.ipynb                 # MUST update — see CLAUDE.md Compliance below
```

### Pattern 1: Title-prefix vocabulary via a small constant dict (established idiom)

**What:** Map a status/enum value to a fixed `'[WORD]'` string via a plain dict, never derive
the bracket text from user input.
**When to use:** Any time a `CalendarEvent.title` needs a machine-readable-and-human-readable
terminal-state marker.
**Example (existing precedent, `sync_lco_observation_calendar.py`):**
```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py:28-49
_FAILURE_PREFIX_BY_STATUS = {
    'WINDOW_EXPIRED': '[EXPIRED]',
    'CANCELED': '[CANCELLED]',
    'FAILURE_LIMIT_REACHED': '[FAILED]',
    'NOT_ATTEMPTED': '[FAILED]',
}

def _title_for(record, telescope, instrument, facility, label_was_fallback) -> str:
    prefix = _failure_prefix(record.status, facility)
    if prefix is not None:
        return f'{prefix} {telescope} {instrument}'
    ...
    return f'{telescope} {instrument}'
```
**Recommended new code (`load_telescope_runs.py`, before the existing `title = ...` line):**
```python
_CLASSICAL_STATUS_PREFIX = {'cancelled': '[CANCELLED]'}  # D-02: only 'cancelled' has a prefix today

...
prefix = _CLASSICAL_STATUS_PREFIX.get(parsed.status)
title = f'{prefix} {parsed.telescope} {parsed.instrument}' if prefix else f'{parsed.telescope} {parsed.instrument}'
```

**Recommended new code (`campaign_views.py`, new helper near `_project_calendar_event`):**
```python
# D-03: two distinct prefixes, never a shared label.
_RUN_STATUS_CALENDAR_PREFIX = {
    CampaignRun.RunStatus.CANCELLED: '[CANCELLED]',
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE: '[WEATHERED]',
}
```

### Pattern 2: django-tables2 per-row mini-form action (established idiom)

**What:** `render_actions(self, record)` returns a `format_html()`-built `<form>` with named
submit buttons (`name="action" value="..."`), CSRF token minted via
`get_token(self.request)`, POSTing to the same `campaigns:decide` URL every other action uses.
**When to use:** Any new staff action on a `CampaignRun` row.
**Example (existing precedent, `campaign_tables.py` resolve-mode branch):**
```python
# Source: solsys_code/campaign_tables.py:308-319
if self.mode == 'resolve':
    form_id = f'resolve-form-{record.pk}'
    return format_html(
        '<form id="{0}" method="post" action="{1}">'
        '<input type="hidden" name="csrfmiddlewaretoken" value="{2}">'
        '<button type="submit" name="action" value="resolve_site" '
        'class="btn btn-sm btn-primary">Resolve</button>'
        '</form>',
        form_id, decide_url, csrf_token,
    )
```
**Recommended shape for D-04 (two small buttons, not a dropdown):** mirrors the existing
Approve/Reject `d-flex` button-pair markup almost exactly — lowest-friction, no new JS, no new
form-serialization logic. Gate rendering on `record.approval_status == APPROVED` (a REJECTED
Decided row has nothing to mark).
```python
def render_actions(self, record):
    if self.mode == 'resolve':
        ...  # unchanged
    if not self.show_actions:
        if self.status_actions and Accessor('approval_status').resolve(record, quiet=True) == CampaignRun.ApprovalStatus.APPROVED:
            decide_url = reverse('campaigns:decide', kwargs={'pk': record.pk})
            csrf_token = get_token(self.request) if self.request is not None else ''
            return format_html(
                '<form method="post" action="{0}">'
                '<input type="hidden" name="csrfmiddlewaretoken" value="{1}">'
                '<div class="d-flex" style="gap: 0.5rem;">'
                '<button type="submit" name="action" value="mark_cancelled" '
                'class="btn btn-sm btn-outline-secondary">Mark Cancelled</button>'
                '<button type="submit" name="action" value="mark_weather_failure" '
                'class="btn btn-sm btn-outline-secondary">Mark Weathered</button>'
                '</div></form>',
                decide_url, csrf_token,
            )
        return ''
    ...  # unchanged pending-mode branch
```

### Pattern 3: No-churn CalendarEvent update via `insert_or_create_calendar_event()`

**What:** The shared helper create-or-updates by lookup key, comparing fields and skipping
`.save()` entirely if nothing changed.
**When to use:** Any time an existing `CalendarEvent` needs its title/description changed
without touching `start_time`/`end_time`/`telescope`.
**Example:**
```python
# Source: solsys_code/calendar_utils.py:318-378
event, action = insert_or_create_calendar_event(
    {'url': f'CAMPAIGN:{run.pk}'},
    fields={'title': new_title, 'description': new_description},
)
```
**Critical guard (see Common Pitfalls):** only call this when an event with that `url` is
already known to exist — `get_or_create()`'s create-path requires `start_time`/`end_time`
(non-nullable on `CalendarEvent`), which this call deliberately omits.

### Anti-Patterns to Avoid
- **Calling `insert_or_create_calendar_event()` unconditionally from the new status-change
  branch:** for a run whose window is a range/TBD or whose site never resolved, no
  `CAMPAIGN:{pk}` event exists. `get_or_create()` would then attempt
  `CalendarEvent.objects.create(url=..., title=..., description=...)` with no `start_time`/
  `end_time` — an immediate `IntegrityError`/`TypeError` (fields are non-nullable). Always
  check `CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists()` first and skip
  silently (matching `_project_calendar_event()`'s own skip-by-design behavior) if none.
- **Reusing `'[FAILED]'` for `WEATHER_TECH_FAILURE`:** technically already in
  `_TERMINAL_PREFIXES` (gets the ring for free), but violates D-03's explicit "must render
  differently" requirement — `[FAILED]` already means something else (LCO
  `FAILURE_LIMIT_REACHED`/`NOT_ATTEMPTED`). Use a new, distinct `[WEATHERED]` string.
- **Deriving the run_status action's title prefix from free user input:** always look it up
  from a fixed dict keyed on the `RunStatus` enum member, never interpolate a raw string
  (mirrors the existing `_FAILURE_PREFIX_BY_STATUS` discipline and the codebase's
  `format_html`-only, never-`mark_safe` convention for anything touching submitted text).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Create-or-update-in-place `CalendarEvent` writes | A new save/update helper | `insert_or_create_calendar_event()` (`calendar_utils.py`) | Already handles no-churn comparison, `modified` field discipline, and is the one function every sync command routes through — a second hand-rolled writer would fragment that invariant |
| Status→prefix mapping | A conditional if/elif chain inline in `handle()`/`post()` | A small constant dict, `.get(status)` | Matches the existing `_FAILURE_PREFIX_BY_STATUS` pattern exactly; keeps the vocabulary auditable in one place |
| Site resolution for "Gemini-South" | A hardcoded `Observatory.objects.create(obscode='I11', ...)` fixture with hand-typed lat/lon | `resolve_site('I11')` (Tier 2 MPC lookup) or the existing site-search widget | The MPC API already has the authoritative lat/lon/altitude for I11, and `MPCObscodeFetcher.to_observatory()` already backfills a real IANA timezone (quick task `260716-h8c`) — hand-typing risks getting the (unusual, GMOS-S-relevant) timezone or altitude wrong |
| Business-logic-bypass guard on the new status-change action | Trusting the button was only rendered for APPROVED rows | Server-side `if run.approval_status != CampaignRun.ApprovalStatus.APPROVED: ... return` (mirrors `_resolve_site()`'s existing guard) | The Decided table is staff-only UI, but the endpoint itself has no server-side proof the POST came from that specific row's rendered button — never trust client-side gating alone (established codebase discipline, see `_resolve_site()` docstring) |

**Key insight:** every piece of machinery this phase needs (create-or-update helper,
prefix-dict pattern, per-row mini-form, 3-tier site resolver, staleness-safe conditional
update) already exists somewhere in this codebase for an almost-identical use case. The task
is wiring, not invention — any solution that introduces a new abstraction for one of these
five things should be treated as a red flag during plan review.

## Common Pitfalls

### Pitfall 1: Calling the calendar-sync helper for a run that never had a projected event
**What goes wrong:** `_project_calendar_event()` only ever creates a `CAMPAIGN:{pk}`
`CalendarEvent` for a resolved-site, single-night (`window_start == window_end`) run. The
Gemini FT-115 entry itself (D-06's own seed data: `window_start=2026-07-13`,
`window_end=2026-07-16`) is a 4-day *range*, so it will never have a projected event even
after approval. If the new status-change handler blindly calls
`insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields={...title/description
only...})`, `get_or_create()` falls into its create-path (`CalendarEvent.objects.create(url=...,
title=..., description=...)`), which crashes immediately — `start_time`/`end_time` are
non-nullable and not supplied.
**Why it happens:** D-05's "updates the existing... CalendarEvent in place" phrasing reads as
if an event always exists once a run is approved; it doesn't, for range/TBD/unresolved-site
runs.
**How to avoid:** `if not CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists(): #
skip silently, run_status still gets set` — check existence before calling the shared helper.
**Warning signs:** a test approving a *range*-window `CampaignRun` (e.g. the Gemini fixture
itself) then marking it cancelled should NOT create a new `CalendarEvent`; if it does, the
guard is missing.

### Pitfall 2: `'[WEATHERED]'` (or whatever prefix is chosen) not in `_TERMINAL_PREFIXES`
**What goes wrong:** The event's title correctly shows `[WEATHERED] ...`, but the calendar tile
renders with no visual differentiation at all (no box-shadow ring) — because
`status_border_css()` only checks `title.startswith(p) for p in _TERMINAL_PREFIXES`, and the
new prefix isn't a member.
**Why it happens:** `_TERMINAL_PREFIXES` and the title-prefix vocabulary
(`_FAILURE_PREFIX_BY_STATUS` / the new `_RUN_STATUS_CALENDAR_PREFIX`) are two independent
constants in two different files with no shared source of truth — it's easy to add a new
prefix to one and forget the other.
**How to avoid:** `_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[WEATHERED]')`
— a one-line addition in `calendar_display_extras.py`. `'[CANCELLED]'` is **already** present
(inherited from the LCO vocabulary), so D-01/D-02's classical prefix and D-03's `CampaignRun`
`CANCELLED` prefix need **no change here** — only the new `WEATHER_TECH_FAILURE` prefix does.
**Warning signs:** `test_calendar_display_extras.py`'s existing parametrized-style tests
(`test_cancelled_returns_terminal_box_shadow` etc.) should get a sibling
`test_weathered_returns_terminal_box_shadow` — if it's missing, the gap wasn't caught.

### Pitfall 3: Decided-table `render_site()` accidentally rendering the live-search widget
**What goes wrong:** `ApprovalQueueTable.render_site()`'s current guard is
`elif not self.show_actions: return super().render_site(record)` for an unresolved site. If
the planner naively flips `show_actions=True` for the Decided table (to make
`render_actions()`'s early-return `if not self.show_actions: return ''` fire the new buttons),
any Decided row whose `site` is still unresolved would suddenly fall through to the live
`_render_site_search_widget()` — an interactive input with no matching `form=` target, since
the new action form isn't `decide-form-{pk}`/`resolve-form-{pk}`.
**Why it happens:** `show_actions` currently gates BOTH the site-widget AND the actions column
together; this phase needs actions-only gating for the Decided table.
**How to avoid:** add an independent constructor flag (e.g. `status_actions=False`, default
off) used *only* inside `render_actions()`'s `if not self.show_actions:` branch; leave
`show_actions=False` unchanged for the Decided table's `ApprovalQueueTable(...)` construction
in `ApprovalQueueView.get_context_data()` so `render_site()`'s existing plain-text-fallback
path is completely untouched.
**Warning signs:** any Decided-table row rendering an `<input>`/`hx-get` widget instead of
plain text is the tell.

### Pitfall 4: Re-running `load_telescope_runs` after a status word is removed
**What goes wrong:** if staff later remove the `(cancelled)` parenthetical from the source
line (storm passed, run reinstated) and re-run the command, the title must revert to
un-prefixed. This works automatically today (`insert_or_create_calendar_event()` compares
every field in `fields` against the existing row and updates on any diff), but only if the
new prefix logic is computed fresh on every `handle()` invocation rather than cached/merged
with the existing title.
**Why it happens:** an implementation that does `title = event.title + ' [CANCELLED]'`
(append-style) instead of recomputing `title` from `parsed.status` each time would never
un-prefix on revert, and would double-prefix on a second cancelled re-ingest.
**How to avoid:** always compute `title` fresh from `parsed.telescope`/`parsed.instrument`/
`parsed.status` each run (as the existing code already does for the unprefixed case) — never
read or mutate the previous `event.title`.
**Warning signs:** a test that ingests a cancelled line, then re-ingests the same line without
the status word, and asserts the title lost its prefix.

## Code Examples

### Existing classical-run title construction (the exact insertion point for D-01/D-02)
```python
# Source: solsys_code/management/commands/load_telescope_runs.py:139-150
title = f'{parsed.telescope} {parsed.instrument}'
description = (
    f'Dark window (-15 deg, UTC): {dark_start_dt.isoformat()} to {dark_end_dt.isoformat()}\n'
    f'Status: {parsed.status}\n'
    f'Source line: {line.strip()}'
)

event, action = insert_or_create_calendar_event(
    {'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': start_time},
    {'end_time': end_time, 'title': title, 'description': description},
    start_time_tolerance=_START_TIME_MATCH_TOLERANCE,
)
```

### Existing `_resolve_status()` / `KNOWN_STATUSES` (confirms D-01/D-02's exact insertion point)
```python
# Source: solsys_code/telescope_runs.py:36
KNOWN_STATUSES = {'allocation', 'proposed', 'confirmed', 'cancelled', 'not confirmed'}

# Source: solsys_code/telescope_runs.py:361-393 (_resolve_status signature/return)
def _resolve_status(line: str) -> tuple[str, str]:
    """Returns (status, remainder); status defaults to 'allocation' if absent."""
```
`parsed.status` is one of these five lowercase strings. Only `'cancelled'` maps to a title
prefix per D-02 — the other four (`allocation`/`proposed`/`confirmed`/`not confirmed`) keep
the unprefixed title, unchanged.

### Existing `CampaignRunDecisionView.post()` action dispatch (exact branch-point for D-04)
```python
# Source: solsys_code/campaign_views.py:452-458
def post(self, request, pk):
    action = request.POST.get('action')
    if action not in ('approve', 'reject', 'resolve_site'):
        return HttpResponseBadRequest()
    if action == 'resolve_site':
        return self._resolve_site(request, pk)
    ...
```
Extend the tuple to `('approve', 'reject', 'resolve_site', 'mark_cancelled',
'mark_weather_failure')` and add a new `elif action in (...)` branch (or a new
`self._set_run_status(request, pk, action)` method mirroring `_resolve_site()`'s shape:
`get_object_or_404`, business-logic guard, conditional `.update()`, then the
calendar-sync-if-exists logic from Pitfall 1).

### Existing `RUN_STATUS_BADGE_CLASSES` (already correct — no change needed)
```python
# Source: solsys_code/campaign_tables.py:32-41
RUN_STATUS_BADGE_CLASSES = {
    ...
    CampaignRun.RunStatus.CANCELLED: 'badge-light',
    CampaignRun.RunStatus.NOT_AWARDED: 'badge-light',
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE: 'badge-light',
}
```
The per-campaign table's `run_status` badge already renders both target states distinctly by
label text (`Cancelled` / `Weather/Technical Failure`) — this phase does not need to touch
`render_run_status()` at all, only add the write-path action that sets the value.

### Gemini South MPC obscode (D-06 site resolution — verified against the authoritative source)
```
Code I11: longitude=291.8200, "Gemini South Observatory, Cerro Pachon"
```
Ground-based (real lat/lon, not the `null`-longitude satellite shape that broke Tier-2
resolution for 250/274/289 per Phase 18's finding) — `resolve_site('I11')` or the approval
queue's fuzzy-match widget (searching "Gemini South") will Tier-2-resolve it cleanly and
backfill a real `America/Santiago` timezone automatically.

## State of the Art

No frameworks or conventions have changed since the last phase touching this code (Phase 22,
2026-07-16). This phase is additive to a stable, recently-built pattern set.

**Deprecated/outdated:** none.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `'[WEATHERED]'` is a reasonable exact wording for `WEATHER_TECH_FAILURE`'s prefix (D-03 leaves exact wording to discretion) | Architecture Patterns / Code Examples | Low — cosmetic only; any distinct bracketed word satisfies D-03's "must render differently" requirement, and changing it later is a one-line dict edit with no migration |
| A2 | Two small buttons (not a dropdown) is the lowest-friction shape for D-04's Decided-table action | Architecture Patterns Pattern 2 | Low — both shapes route through the same `campaigns:decide` POST endpoint; a later UI swap doesn't touch `CampaignRunDecisionView.post()` at all |
| A3 | The Gemini FT-115 `CampaignRun` row is intended to reach `approval_status=APPROVED` (so it's eligible for the Decided-table action per D-07) via the *existing* admin/approval-queue paths, with no new creation UI needed | Don't Hand-Roll / Summary | Low-medium — if the operator instead wants a management command or fixture script to seed this row, that's a small addition, but D-06/D-07's wording ("subject to the same run_status mechanism") implies it flows through the same admin-create-then-approve or CSV-import path every other `CampaignRun` uses |

**If this table is empty:** N/A — see rows above. All entries are low-risk cosmetic/process
choices explicitly delegated to Claude's discretion by CONTEXT.md; none affect correctness of
the core wiring (title-prefix logic, calendar-sync guard, `_TERMINAL_PREFIXES` addition, site
resolution), all of which are `[VERIFIED]` against the live codebase or the authoritative MPC
ObsCodes list.

## Open Questions

1. **Should the Decided-table status-change buttons remain visible after a run is already
   `CANCELLED`/`WEATHER_TECH_FAILURE` (to allow reverting), or disappear once set?**
   - What we know: `RUN_STATUS_BADGE_CLASSES`/the badge renderer already handles displaying
     any terminal state; nothing in CONTEXT.md specifies revert behavior.
   - What's unclear: whether a storm-cancelled run that's later reinstated needs a UI path
     back to `REQUESTED`/`OBSERVED`, or whether that's Django-admin-only (consistent with
     every other non-approval `run_status` transition today).
   - Recommendation: keep it simple for this phase — show the two buttons for any
     `APPROVED` row regardless of current `run_status` (so a mis-click is correctable via the
     same UI, and re-clicking the same button is a harmless idempotent no-op given the
     conditional-update pattern). Don't add a third "revert" button; that's Django-admin's
     existing job (`run_status` is already editable there).

2. **Does the Gemini FT-115 `CampaignRun` row need `target` set to a real `Target`, or does it
   stay `target=None` like most multi-target-campaign rows?**
   - What we know: `CampaignRun.target` is nullable/optional; `TestCampaignRunOptionalTarget`
     confirms both states persist fine; Phase 17's `claimed_dates()` buckets `target=None` rows
     into `unattributed_runs` separately.
   - What's unclear: whether "Didymos 2026" (`TargetList` pk=1) has a specific Didymos
     `Target` row the operator wants this CampaignRun linked to for downstream gap-analysis
     purposes.
   - Recommendation: leave `target=None` unless the operator specifies otherwise at plan or
     discuss time — D-06 doesn't mention it, and it's not required for the run_status/calendar
     wiring this phase delivers. If set, use `NonSiderealTargetFactory`-equivalent real data
     (never a sidereal fixture), per CLAUDE.md.

## Environment Availability

No new external dependencies. `resolve_site('I11')`'s Tier-2 path depends on the MPC Obscodes
API being reachable at test/runtime — this is a pre-existing dependency of every site-resolution
code path in the app (Phase 18/21/22), not new to this phase.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| MPC Obscodes API (minorplanetcenter.net) | D-06 Gemini site Tier-2 resolution | ✓ (confirmed live during this research session) | — | `resolve_site()` already degrades to `(None, True)` on network failure (existing behavior, unaffected by this phase) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none new — existing `resolve_site()`/
`build_site_candidates()` fallback behavior is unchanged by this phase.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (`./manage.py test solsys_code`) — this phase's changes are all DB-dependent (CalendarEvent, CampaignRun, Observatory), so `python -m pytest` (which only collects `tests/`, `src/`, `docs/`) does not apply |
| Config file | none — Django test runner uses `DJANGO_SETTINGS_MODULE=src.fomo.settings` via `manage.py` |
| Quick run command | `./manage.py test solsys_code.tests.test_load_telescope_runs` / `./manage.py test solsys_code.tests.test_campaign_approval` / `./manage.py test solsys_code.tests.test_calendar_display_extras` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D-01/D-02 | Classical run with `(cancelled)` status word gets `[CANCELLED]` title prefix; a non-cancelled line stays unprefixed | unit (Django TestCase) | `./manage.py test solsys_code.tests.test_load_telescope_runs.TestLoadTelescopeRuns.test_event_fields_set_from_parsed_run` (extend with a cancelled-line sibling test) | ✅ file exists, add new test method |
| D-01/D-02 (revert) | Re-ingesting a line after the status word is removed reverts the title to unprefixed (Pitfall 4) | unit | new test method in `test_load_telescope_runs.py` | ❌ Wave 0 — new test method |
| D-03 | `CANCELLED`/`WEATHER_TECH_FAILURE` produce two distinct title prefixes on the `CAMPAIGN:{pk}` event | unit | new test method in `test_campaign_approval.py` | ❌ Wave 0 — new test method |
| D-04 | Decided-table row for an APPROVED run renders the new action buttons; a REJECTED row does not; a non-staff/anonymous POST is rejected (mirrors `TestStaffGating`) | unit + view/integration | new `TestRunStatusChangeAction`-style class in `test_campaign_approval.py` | ❌ Wave 0 — new test class |
| D-05 | Marking a single-night, resolved-site run cancelled updates its existing `CAMPAIGN:{pk}` event's title/description in place, no duplicate created (mirrors `TestCalendarProjection`/`TestCalendarNoChurn` style) | unit | new test method | ❌ Wave 0 — new test method |
| D-05 (Pitfall 1) | Marking a *range-window* run (no projected event) cancelled does NOT crash and does NOT create a new `CalendarEvent` | unit | new test method — this is the highest-value regression test in the whole phase | ❌ Wave 0 — new test method |
| Discretion #1 | `[WEATHERED]` added to `_TERMINAL_PREFIXES` gets the box-shadow ring; `[CANCELLED]` already did (regression check) | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras` (extend `TestStatusBorderCss`-style class) | ✅ file exists, add new test method |
| D-06 | `resolve_site('I11')` resolves to a ground-based, non-placeholder `Observatory` with a real timezone (integration, may need to mock the MPC API call per existing `BULK_MPC_FIXTURE`-style precedent, or add `I11` to that fixture) | unit | new test method in `test_campaign_approval.py` or `test_campaign_utils.py` (whichever exists — confirm during planning) | Confirm at plan time |

### Sampling Rate
- **Per task commit:** the single most-relevant `./manage.py test solsys_code.tests.test_X` module for the file just touched.
- **Per wave merge:** `./manage.py test solsys_code` (full Django suite — this phase touches no `tests/`/`src/`/`docs/`-rooted code, so `python -m pytest` is not required to re-run, but running it is harmless/fast and catches any accidental import-time regression).
- **Phase gate:** `./manage.py test solsys_code` green, plus `ruff check .` / `ruff format --check .` clean (per CLAUDE.md), before `/gsd-verify-work`.

### Wave 0 Gaps
- New test methods listed above (marked ❌) — no new test *files* are needed; every relevant
  test module already exists (`test_load_telescope_runs.py`, `test_campaign_approval.py`,
  `test_calendar_display_extras.py`).
- If D-06's `resolve_site('I11')` test needs a mocked MPC response, extend the existing
  `BULK_MPC_FIXTURE` dict in `test_campaign_approval.py` with an `'I11'` entry (real values:
  `name_utf8='Gemini South Observatory, Cerro Pachon'`, `longitude=291.8200`,
  `observations_type='fixed'`) rather than inventing a new fixture shape.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | no | Unchanged — `CampaignRunDecisionView` already requires an authenticated staff session via `StaffRequiredMixin` |
| V3 Session Management | no | Unchanged — standard Django session auth, no new session state |
| V4 Access Control | yes | `StaffRequiredMixin` (existing) gates the whole view; the NEW status-change branch additionally needs its own business-logic guard (`run.approval_status == APPROVED`), mirroring `_resolve_site()`'s existing pattern — never trust that the button was only rendered on an eligible row |
| V5 Input Validation | yes | `action` POST field validated against a fixed whitelist tuple (existing pattern, extend it); title-prefix text is always looked up from a fixed dict, never interpolated from request data |
| V6 Cryptography | no | Not applicable — no crypto in this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Business-logic bypass: staff POSTs `action=mark_cancelled` directly to `campaigns:decide` for a PENDING_REVIEW or REJECTED run (bypassing the Decided-table's row-gating, which only exists client-side in the rendered HTML) | Elevation of Privilege | Server-side guard: `if run.approval_status != CampaignRun.ApprovalStatus.APPROVED: messages.warning(...); return redirect(...)` — exact mirror of `_resolve_site()`'s existing `if run.approval_status != APPROVED or not run.site_needs_review:` guard |
| Reflected/stored XSS via `run.telescope_instrument`/`run.campaign.name` interpolated into the new `CalendarEvent.title`/`.description` | Tampering | Continue the codebase's existing discipline: build the new title/description via `f'{prefix} {title}'`-style Python string formatting is fine here because `CalendarEvent.title`/`.description` are rendered later through Django's auto-escaping template layer (`{{ event.title }}`) — never introduce `mark_safe()` or string-concatenate raw request data into a `format_html()` call for this phase's new code |
| CSRF on the new action buttons | Tampering | Already covered — reuses the exact same `get_token(self.request)` hidden-field pattern every other `ApprovalQueueTable` action form already uses |
| Double-submit / race on the conditional `run_status` update | Repudiation (inconsistent audit state) | Use a conditional `.filter(pk=pk, approval_status=APPROVED).update(run_status=...)` (staleness-safe, mirrors the existing `updated_count == 1` discipline used by approve/reject and `_resolve_site()`'s conditional site-claim) rather than a plain fetch-then-save |

## Sources

### Primary (HIGH confidence)
- `solsys_code/templatetags/calendar_display_extras.py` (read in full) — `_TERMINAL_PREFIXES`, `status_border_css()`
- `src/templates/tom_calendar/partials/calendar.html` (read in full) — confirms `status_border_css` is invoked per-event from `event.title` for both all-day and timed events
- `solsys_code/telescope_runs.py` (read in full) — `KNOWN_STATUSES`, `_resolve_status()`
- `solsys_code/management/commands/load_telescope_runs.py` (read in full) — exact title/description construction site
- `solsys_code/calendar_utils.py` (read in full) — `insert_or_create_calendar_event()` full contract
- `solsys_code/management/commands/sync_lco_observation_calendar.py` (relevant excerpt) — `_FAILURE_PREFIX_BY_STATUS`, `_title_for()`
- `solsys_code/campaign_tables.py` (read in full) — `ApprovalQueueTable.render_actions()`/`render_site()`, `RUN_STATUS_BADGE_CLASSES`
- `solsys_code/campaign_views.py` (read in full) — `CampaignRunDecisionView.post()`, `_project_calendar_event()`, `ApprovalQueueView.get_context_data()`
- `solsys_code/models.py` (relevant excerpt) — `CampaignRun.RunStatus`/`ApprovalStatus` choices, field nullability
- `solsys_code/admin.py` (read in full) — confirms `run_status` NOT in `readonly_fields`
- `solsys_code/campaign_utils.py` (relevant excerpts) — `resolve_site()`, `build_site_candidates()`, `is_placeholder_observatory()`
- `solsys_code/tests/test_campaign_approval.py` (relevant excerpts) — `CampaignApprovalTestBase`, `BULK_MPC_FIXTURE`, `TestApproval` fixture pattern, `ISOLATED_TEST_CACHES`
- `solsys_code/tests/test_load_telescope_runs.py` (relevant excerpts) — existing title-assertion test pattern
- `solsys_code/tests/test_campaign_models.py` (relevant excerpts) — confirms `NonSiderealTargetFactory` usage convention
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` (inspected) — confirms `event.title` is printed in an existing output cell
- `src/templates/campaigns/approval_queue.html` (read in full) — confirms Decided table's `{% render_table decided_table %}` structure
- https://www.minorplanetcenter.net/iau/lists/ObsCodes.html — fetched directly; confirmed `I11` = "Gemini South Observatory, Cerro Pachon", longitude 291.8200 `[VERIFIED: minorplanetcenter.net ObsCodes.html]`
- `.planning/STATE.md` — Phase 18/19/21/22 decision log, confirms the 250/274/289 satellite-obscode Tier-2 bug does NOT apply to I11 (ground-based, real longitude)

### Secondary (MEDIUM confidence)
- none — every claim above was either read directly from the live codebase or confirmed against the authoritative MPC ObsCodes source.

### Tertiary (LOW confidence)
- Initial WebSearch results suggesting `I11` for Gemini South (before the direct-fetch confirmation) — superseded by the primary-source fetch above; kept no residual claims at this tier.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all reused code read directly from the repository.
- Architecture: HIGH — every recommended pattern is a direct extension of an existing, already-shipped precedent in this exact codebase (LCO prefix dict, django-tables2 mini-form actions, no-churn calendar helper).
- Pitfalls: HIGH — Pitfall 1 (range-window Gemini entry has no CalendarEvent to update) and Pitfall 3 (render_site widget leak) were discovered by tracing actual code paths, not inferred; both are concrete, verifiable-by-test failure modes.
- Gemini site resolution (D-06): HIGH — `I11` confirmed against the live, authoritative MPC ObsCodes list via direct fetch in this session.

**Research date:** 2026-07-16
**Valid until:** 2026-08-15 (30 days — stable, internal-only feature work with no external API surface beyond the already-integrated MPC Obscodes API)
