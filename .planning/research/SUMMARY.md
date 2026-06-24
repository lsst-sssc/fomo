# Project Research Summary

**Project:** FOMO v1.4 "Calendar Visual Clarity" (DISPLAY-01 proposal-keyed color + status visual treatment; DISPLAY-02 fallback-vs-verified telescope label field)
**Domain:** Django calendar visual-treatment + cross-app model extension on a third-party (`tom_calendar`) model
**Researched:** 2026-06-24
**Confidence:** HIGH

## Executive Summary

This is a narrow, additive UI-polish milestone on an already-validated stack (Django 5.2.15,
`tomtoolkit==3.0.0a9`). Neither DISPLAY-01 (color the calendar by proposal, not by row `pk`, and give
schedule status a visual — not just textual — language) nor DISPLAY-02 (persist whether a telescope
label was live-resolved or fallback-guessed) needs a new dependency, app, or framework. Both are solved
with patterns already present in this exact codebase: a Django template tag library mirroring
`tom_calendar`'s own `target_list_color` precedent (DISPLAY-01), and a `OneToOneField` sidecar model in
the existing `solsys_code` app extending a third-party model FOMO doesn't own (DISPLAY-02). The two
features are architecturally independent — DISPLAY-01 is read-side/template-only (no migration, since
`proposal` already exists on `CalendarEvent`), DISPLAY-02 is write-side-first (new model, new migration,
one new call site in `sync_lco_observation_calendar.py`) — and should be planned as two sequential
phases.

The recommended approach: hash a *normalized* `proposal` string (`.strip().upper()` first — real data
may have inconsistent casing/whitespace) into a small, curated, colorblind-vetted palette (extending the
existing `BOOTSTRAP_COLORS` precedent, not raw hash-to-HSL); layer schedule-status as an orthogonal
structural CSS treatment (border-style is the research-favored option, with opacity as the cheapest
incremental fix) on top of — not instead of — that color; and add `CalendarEventTelescopeLabel` as a
`primary_key=True` `OneToOneField` sidecar model, written via `update_or_create` colocated with the
existing `get_or_create`/diff/`save()` block in `sync_lco_observation_calendar.py`, with no row at all
for classical-schedule events (`load_telescope_runs.py` needs zero code change).

Key risks, all concretely identified rather than speculative: (1) the existing `[QUEUED]` template
override already discards `event.color` with a flat grey fill — DISPLAY-01 must fix this in the same
task that adds proposal coloring, or the new signal is invisible for every queued event; (2) reading the
new sidecar's reverse `OneToOneField` accessor per-event inside the month-grid loop is a genuine N+1 (no
Python-level hook exists on the upstream `render_calendar()` view to fix it server-side — mitigation must
live in a batching template tag); and (3) un-normalized proposal-string hashing silently breaks the
entire premise of DISPLAY-01 (same proposal, different color) the moment real data has mixed casing.
None of these are exotic — they're scoped, testable, and each maps to a specific phase and task below.

## Key Findings

### Recommended Stack

No new runtime dependency for either feature. `hashlib.sha256(s.encode()).hexdigest()` (never Python's
built-in `hash()`, which is per-process salted and empirically confirmed non-deterministic across
restarts) provides the deterministic proposal→color mapping. A Django `@register.simple_tag` template
tag library mirrors `tom_calendar.templatetags.calendar_tags.target_list_color`. A `OneToOneField`
sidecar model in `solsys_code/models.py` is the standard Django pattern for adding a field to a
third-party model (`CalendarEvent`) that hardcodes `Meta.app_label = 'tom_calendar'` and cannot be
subclassed via abstract-base or safely via multi-table inheritance.

**Core technologies:**
- `hashlib` (stdlib) — deterministic proposal→color hash — process-stable, unlike `hash()`
- Django template tag library (`simple_tag`) — exposes computed color/status values to `calendar.html` — exact mechanism `tom_calendar` already uses for `target_list_color`
- Django `OneToOneField(primary_key=True)` sidecar model — attaches `is_verified` to `CalendarEvent` without touching its (vendored) migrations — the standard documented pattern for extending a model you don't own

### Expected Features

**Must have (table stakes):**
- Deterministic color per proposal (same proposal = same color, every render, every restart)
- Non-color redundant signal for status (already partially satisfied by existing `[QUEUED]`/`[UNVERIFIED]`/terminal title prefixes — WCAG 1.4.1)
- Legible text over an unpredictable-lightness palette (solved by sticking to a small curated palette rather than open-ended hash-to-hue)
- Stable color across htmx month-grid re-renders (must hash on `proposal` value, never queryset position/pk — the exact bug being fixed)

**Should have (competitive/differentiator):**
- Curated, colorblind-vetted fixed palette (extend `BOOTSTRAP_COLORS`, not raw HSL-from-hash)
- Status as a structural CSS treatment (border-style favored; opacity cheapest, but the current `[QUEUED]` opacity-equivalent already destroys color and must be fixed regardless of which option wins)
- Dedicated `is_verified`-style field for DISPLAY-02, replacing string-parsing of `[UNVERIFIED]` titles
- A shared visual vocabulary (e.g. dashed border) for "provisional" across both DISPLAY-01 status and DISPLAY-02 verification, decided explicitly in sketch rather than defaulted

**Defer (v2+):**
- WCAG contrast-ratio-aware text-color switching (only matters if palette grows)
- FK-backed `Telescope`/`Proposal` models (already explicitly out of scope per `PROJECT.md` and the YSE_PZ comparison doc)
- On-page color legend, hover tooltips surfacing verification detail

### Architecture Approach

Both features are additive within the existing `solsys_code` app — no new Django app, no `INSTALLED_APPS`
change. DISPLAY-01 adds a `solsys_code/templatetags/calendar_display_extras.py` package (new) exposing
`proposal_color` and a status/prefix-to-CSS-class tag, loaded into the overridden
`src/templates/tom_calendar/partials/calendar.html`. DISPLAY-02 adds `CalendarEventTelescopeLabel` to
`solsys_code/models.py` (this app's first real model and first real migration) and one `update_or_create`
call colocated with the existing `get_or_create` write in `sync_lco_observation_calendar.py`;
`load_telescope_runs.py` needs no change. The template is the only customization seam for the read side
— `tom_calendar.views.render_calendar()` is a plain function-based view with no `extra_context`/
`get_queryset()` hook, confirmed by direct read, so any N+1 mitigation for the sidecar's reverse-O2O
accessor must live in a batching template tag, not a view-level fix.

**Major components:**
1. `solsys_code/models.py` — new `CalendarEventTelescopeLabel` sidecar model (DISPLAY-02 persisted flag)
2. `solsys_code/templatetags/calendar_display_extras.py` (new package) — DISPLAY-01's color-hash and status-treatment tags
3. `sync_lco_observation_calendar.py` — one new `update_or_create` call site immediately after the existing `CalendarEvent.objects.get_or_create(...)` (DISPLAY-02 write)
4. `src/templates/tom_calendar/partials/calendar.html` (existing override) — wires both features into the all-day and timed-event render branches

### Critical Pitfalls

1. **Hashing the raw (un-normalized) `proposal` string** — same logical proposal renders different colors if casing/whitespace differs across records. Avoid by normalizing (`.strip().upper()`) as the first line of `proposal_color()`, before any palette-index logic, and spot-checking real DB values first.
2. **Shipping proposal color without fixing the existing `[QUEUED]` grey override** — `calendar.html:158-161` already replaces `event.color` with flat grey for queued events; left unfixed, the new color signal is invisible for the most common transient state. Must be fixed in the same task that adds the color tag.
3. **N+1 queries from the sidecar's reverse `OneToOneField` accessor** read per-event inside the month-grid template loop — no Python-level hook exists to fix `render_calendar()`'s queryset. Mitigate with a batching template tag (`filter(event_id__in=...)` once per render) and a query-count regression test (`assertNumQueries`).
4. **Conflating the sidecar write with `CalendarEvent`'s existing no-churn diff block** — keep `CalendarEventTelescopeLabel.objects.update_or_create(...)` as a separate statement, never folded into the `fields` dict/`changed` comparison.
5. **Template-fork drift on a future `tomtoolkit` upgrade** — `calendar.html` is a full copy-and-modify override, not block-based; any upstream change becomes invisible. Accepted cost of the only available customization seam, but add a fork-provenance comment and a `PROJECT.md` constraint note.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: DISPLAY-02 — Verified/fallback telescope label sidecar field
**Rationale:** The riskier, more novel piece (first-ever migration for `solsys_code`, first cross-app `OneToOneField` extension in this codebase) — sequencing it first isolates schema/migration surprises from DISPLAY-01's purely-additive template work, and the existing 36-test `test_sync_lco_observation_calendar.py` suite is the natural place to add sidecar-row assertions without simultaneous template churn. (Alternative ordering — DISPLAY-01 first — is preferable only if the N+1 mitigation's batching tag is explicitly in scope and should be built once and reused; flag this as an explicit scope decision before locking phase order.)
**Delivers:** `CalendarEventTelescopeLabel` model + migration; `update_or_create` write wired into `sync_lco_observation_calendar.py`; `load_telescope_runs.py` confirmed unchanged; template read (`{{ event.telescope_label_meta.is_verified }}`) with a documented no-sidecar-row → "verified" default; N+1 mitigation (batching tag) or explicit accept-as-is decision; no-churn regression test.
**Addresses:** Dedicated `is_verified` field (DISPLAY-02's literal ask), minimal visual cue for verified/fallback.
**Avoids:** Pitfalls 3, 4, 5 (sidecar/no-churn conflation, staleness-contract ambiguity, N+1).

### Phase 2: DISPLAY-01 — Proposal-keyed color + status visual treatment
**Rationale:** Read-side/template-only, no migration, no command change — `proposal` already exists on `CalendarEvent`. Building second means any shared `templatetags/` package scaffolding needed by DISPLAY-02's N+1 mitigation already exists, and the riskier schema work from Phase 1 is already validated.
**Delivers:** `solsys_code/templatetags/calendar_display_extras.py` (`proposal_color` tag, normalized-hash logic, curated colorblind-vetted palette); fix to the `[QUEUED]` grey override so proposal color survives under a status modifier; status visual treatment (border-style favored, decided in `/gsd:sketch`) wired into both the all-day and timed-event branches (today only all-day calls `event.color` at all); fork-provenance comment + `PROJECT.md` constraint note for future `tomtoolkit` upgrades.
**Uses:** `hashlib.sha256`, Django `simple_tag`, the existing `BOOTSTRAP_COLORS`/`target_list_color` precedent (literal-hex local copy, not an import, to survive the pending Bootstrap4→5 CSS-variable rename).
**Implements:** `calendar_display_extras.py` template tag module; extension of the title-prefix-branching pattern.

### Phase Ordering Rationale

- DISPLAY-02's write path (new model/migration) is the higher-novelty, higher-risk piece for this codebase (`solsys_code`'s first real migration) — isolate it first.
- Both phases touch `calendar.html` but at disjoint lines/branches; a merge conflict, if either phase branches in parallel, is trivial to resolve — not a reason to combine the phases.
- Neither phase blocks the other's correctness: DISPLAY-01-only renders fine with zero sidecar rows; DISPLAY-02-only renders fine with `event.color`'s existing pk-keyed behavior untouched.
- A single, explicit roadmap/requirements decision should be made up front about whether the N+1 batching-tag mitigation is in scope for this milestone (vs. accept-as-is) — that decision is the actual tie-breaker for phase order, not a technical constraint.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (DISPLAY-02):** the `ObjectDoesNotExist.silent_variable_failure` template-engine behavior for the missing-sidecar-row case is documented HIGH confidence but not re-verified against this project's installed Django version by direct package-source read — exercise with an actual template-rendering test, not assumption, during planning/execution.
- **Phase 2 (DISPLAY-01):** the specific status-visual-treatment mechanism (border-style vs. opacity vs. stripe) is explicitly deferred to a `/gsd:sketch` session per FEATURES.md's framing — plan-phase should treat this as an open design decision with options already pre-vetted by research, not a closed spec.

Phases with standard patterns (skip research-phase):
- **Phase 1's sidecar-model/migration mechanics** — `OneToOneField(primary_key=True)`, `update_or_create()` are confirmed-current, standard Django idioms with no project-specific novelty beyond the integration call site (already pinpointed by ARCHITECTURE.md to exact line numbers).
- **Phase 2's hashing/palette mechanics** — `hashlib`-based deterministic hashing into a curated palette is a fully specified, low-risk pattern; no further research needed beyond the sketch-session decision on status-treatment CSS.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified by direct read of installed `tom_calendar` package source (models, views, utils, templatetags) matching the exact pinned `tomtoolkit==3.0.0a9`; empirically reproduced the `hash()` non-determinism finding in-session rather than relying on memory. |
| Features | MEDIUM-HIGH | General calendar/accessibility conventions (WCAG, Outlook tentative-striping precedent) are well-documented and cross-checked across multiple sources; astronomy-domain-specific precedent (telescope schedulers) is thinner — most don't publish their color semantics, so the status-treatment recommendation is explicitly framed as sketch-session options, not a closed decision. |
| Architecture | HIGH | Grounded in direct reads of this repo's actual files (`sync_lco_observation_calendar.py` full 643 lines, `load_telescope_runs.py` full 126 lines, `calendar.html` full 209 lines, installed `tom_calendar/views.py`) with exact line-number citations for every integration point. |
| Pitfalls | HIGH | Consistent with and built directly on STACK/FEATURES/ARCHITECTURE findings from the same research day; grounded in this project's actual prior bug history (Phase 5/6 notebook-sync gaps, the `[QUEUED]` override bug found by this same research pass) rather than generic Django advice. |

**Overall confidence:** HIGH

### Gaps to Address

- **Status visual treatment mechanism (border vs. opacity vs. stripe) is intentionally undecided** — research surfaced options and a working recommendation (border-style to lead), but the milestone explicitly wants this decided in a `/gsd:sketch` session, not pre-committed by research. Roadmap/planning should treat this as an open task, not assume an answer.
- **Whether timed events (`day.events`) should get proposal coloring at all** — today only the all-day branch calls `event.color`; this asymmetry is flagged but not resolved by research. Confirm scope with the user/roadmap before Phase 2 planning locks file-level task scope.
- **Whether the N+1 mitigation (batching template tag) is in scope for this milestone** — STACK.md/ARCHITECTURE.md treat "accept as-is" as a legitimate option given current calendar-event volume; this is a scope call that also determines phase ordering (see Phase Ordering Rationale above) and should be made explicit during roadmap creation, not left implicit.
- **Whether classically-scheduled (`proposal=''`) events should land on a dedicated "no proposal" palette slot or share whatever the hash of the empty string produces** — flagged by FEATURES.md as a decision to make explicitly, not yet resolved.
- **`ObjectDoesNotExist.silent_variable_failure` template behavior** — documented Django contract, not re-verified against this project's exact installed version by direct test; flagged for verification during DISPLAY-02 execution, not before.

## Sources

### Primary (HIGH confidence)
- Direct read: installed `tom_calendar` package source (`models.py`, `views.py`, `utils.py`, `templatetags/calendar_tags.py`) under `tomtoolkit==3.0.0a9`
- Direct read: `solsys_code/management/commands/sync_lco_observation_calendar.py` (full, 643 lines) — exact integration line numbers
- Direct read: `solsys_code/management/commands/load_telescope_runs.py` (full, 126 lines) — confirms no DISPLAY-02 integration needed
- Direct read: `src/templates/tom_calendar/partials/calendar.html` (full, 209 lines) — confirms `[QUEUED]` grey-override bug and color-call asymmetry
- Empirical in-session verification: `hash()` per-process non-determinism vs. `hashlib`/`zlib.crc32` determinism
- [Django Model field reference (5.2)](https://docs.djangoproject.com/en/5.2/ref/models/fields/), [Django one-to-one relationships docs](https://docs.djangoproject.com/en/6.0/topics/db/examples/one_to_one/), [Django QuerySet API reference](https://docs.djangoproject.com/en/6.0/ref/models/querysets/)
- [Free/Busy slashed lines — Microsoft Support](https://support.microsoft.com/en-us/topic/free-busy-shows-slashed-lines-in-scheduling-assistant-da1383a8-54fa-4e89-a2d2-214ae7d82615) — official confirmation of the diagonal-stripe-for-tentative convention
- [Making data visualizations accessible — TPGi](https://www.tpgi.com/making-data-visualizations-accessible/) — WCAG 2.1 SC 1.4.1 citation

### Secondary (MEDIUM confidence)
- [eventColor / event-colors-demo — FullCalendar docs](https://fullcalendar.io/docs/eventColor), [How to Color Code Google Calendar](https://www.usecarly.com/blog/how-to-color-code-google-calendar/)
- [colorhash on PyPI](https://pypi.org/project/colorhash/) — basis for rejecting raw hash-to-HSL
- [Stripes in CSS — CSS-Tricks](https://css-tricks.com/stripes-css/) — `repeating-linear-gradient` feasibility
- GitHub `TOMToolkit/tom_base` commit-history summary for `tom_calendar/models.py` (web-fetched summary, not a diff read)

### Tertiary (LOW confidence)
- None flagged — all sources above were judged MEDIUM or higher by the research agents.

---
*Research completed: 2026-06-24*
*Ready for roadmap: yes*
