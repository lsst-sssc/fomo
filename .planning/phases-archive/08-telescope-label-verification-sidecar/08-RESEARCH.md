# Phase 8: Telescope Label Verification Sidecar - Research

**Researched:** 2026-06-24
**Domain:** Django sidecar model (`OneToOneField(primary_key=True)`) extending a third-party model + template-layer visual cue/tooltip
**Confidence:** HIGH

## Summary

This phase is fully covered by milestone-level research already on disk
(`.planning/research/STACK.md`, `ARCHITECTURE.md`, `FEATURES.md`, `PITFALLS.md`, `SUMMARY.md`, all
written 2026-06-24, HIGH confidence) plus the locked decisions in `08-CONTEXT.md`. This document
re-verifies that research against the current state of the code (all line numbers, model shapes, and
package versions below were re-confirmed by direct read/grep in this session, not assumed from the
prior research date) and packages it for the planner in the format the planner expects, with the
phase's specific scope (sidecar model, visual cue, tooltip — NOT proposal color, NOT status treatment,
those are Phase 9).

Everything needed is already validated and present in this exact codebase: Django 5.2.15,
`tomtoolkit==3.0.0a9` (confirmed installed), `solsys_code` app with an all-comment `models.py` and an
`__init__.py`-only `migrations/` folder (confirmed — this is genuinely this app's first model and first
migration), and an existing full-copy template override
(`src/templates/tom_calendar/partials/calendar.html`) with a `[QUEUED]` conditional-class precedent at
lines 158-162 to mirror for the dashed-border branch. The write-site integration point in
`sync_lco_observation_calendar.py` (`telescope_api_failed` computed at line 470, popped at line 604,
`CalendarEvent.objects.get_or_create` at line 615) was re-confirmed at those exact line numbers.

**Primary recommendation:** Add `CalendarEventTelescopeLabel(OneToOneField(CalendarEvent,
on_delete=CASCADE, primary_key=True, related_name='telescope_label_meta'), is_verified=BooleanField)`
to `solsys_code/models.py`; write it via a standalone `update_or_create` call immediately after the
existing `get_or_create` at line 615; never write a row for `load_telescope_runs.py`-created events;
read it in `calendar.html` via `{{ event.telescope_label_meta.is_verified|default:True }}` defensively
guarded against `DoesNotExist`; accept the per-event reverse-accessor N+1 as-is per the locked
discretion decision (DISPLAY-09 deferred to v2).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Verification-outcome persistence (`is_verified`) | Database / Storage (Django ORM, `solsys_code` app) | API/Backend (management command write site) | A structured, queryable fact about a record's resolution outcome belongs in a model field, not a string-parsed title prefix — the management command is the only producer that has the signal to write it |
| Sidecar write timing | API/Backend (management command, batch/cron-style invocation) | — | `sync_lco_observation_calendar.py` runs off the request path; the write is colocated with the existing `CalendarEvent` write in the same loop iteration |
| Visual cue rendering (dashed border) | Frontend Server (Django template, server-rendered HTML) | — | `tom_calendar`'s `render_calendar()` is a plain FBV with no client-side JS framework; all event styling is computed server-side into inline `style=`/conditional CSS classes at template-render time |
| Tooltip text | Frontend Server (Django template, `title` HTML attribute) | — | Per DISPLAY-03, a native `title` attribute hover tooltip — no JS tooltip library, no API/Backend involvement (the text is a fixed string, not computed from variable data per D-01/D-04) |
| N+1 mitigation (deferred) | Database / Storage (would be a batched query) | Frontend Server (would be a template tag) | Out of scope this phase per locked discretion decision — noted for completeness only |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|---------------|
| Django `OneToOneField(primary_key=True)` | Django 5.2.15 (installed, confirmed via `python -c "import django; print(django.VERSION)"` → `(5, 2, 15, 'final', 0)`) [VERIFIED: local environment] | Attaches `is_verified` to `tom_calendar.models.CalendarEvent` without touching that model's (vendored, third-party-owned) migrations | `CalendarEvent` hardcodes `class Meta: app_label = 'tom_calendar'`, is not `abstract`, and ships its own migrations inside the installed `tomtoolkit` package — confirmed by direct read of the installed source this session (`site-packages/tom_calendar/models.py`). A `OneToOneField(primary_key=True)` sidecar model in the project's own app is Django's documented pattern for extending a model you don't own. [CITED: docs.djangoproject.com/en/5.2/topics/db/examples/one_to_one/] |
| Django `update_or_create()` | Django 5.2.15 | Write/refresh the sidecar row on every sync run, flipping `is_verified` if the resolution outcome changed between runs | Standard QuerySet API method; semantics unchanged across Django 5.2/6.0. [CITED: docs.djangoproject.com/en/5.2/ref/models/querysets/#update-or-create] |

### Supporting

None — no new third-party package, no new Django app. This phase needs nothing beyond stdlib Django
ORM/template features already used elsewhere in this codebase.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `OneToOneField(primary_key=True)` sidecar model | `Meta.proxy = True` proxy model | Rejected — proxy models share the parent's table exactly and cannot add new persisted columns; `is_verified` needs a real column. |
| `OneToOneField(primary_key=True)` sidecar model | Multi-table inheritance (`class CalendarEventTelescopeLabel(CalendarEvent)`) | Rejected — would require Django to manage a migration creating a child table against a model whose migration history is owned by the installed `tomtoolkit` package, not this project; fragile across `tomtoolkit` upgrades. |
| Explicit `update_or_create` call colocated with the existing write | `post_save` signal on `CalendarEvent` | Rejected as primary mechanism — would also fire for the upstream `EventForm`/`create_event` view (confirmed present in installed `tom_calendar/views.py` this session), which has no `telescope_api_failed` concept at all and would need an invented default. |
| Boolean-only `is_verified` field (D-01, locked) | A `reason`/`detail` `CharField` distinguishing API-timeout vs. unmapped-code | Explicitly rejected by D-01 — Phase 07's Key Decision already funnels both failure modes into one shared `telescope_api_failed` signal; splitting them now would require new upstream plumbing this phase doesn't touch. |

**Installation:**
```bash
# No new packages required.
# After adding the sidecar model to solsys_code/models.py:
./manage.py makemigrations solsys_code
./manage.py migrate
```

**Version verification:** `python -c "import django; print(django.VERSION)"` → `(5, 2, 15, 'final', 0)`
[VERIFIED: local environment, this session]. `pip show tomtoolkit` → `Version: 3.0.0a9`
[VERIFIED: local environment, this session]. Both match the prior milestone research and
`pyproject.toml`'s pin exactly — no drift.

## Package Legitimacy Audit

**Not applicable — this phase installs no new packages.** Both required APIs
(`OneToOneField(primary_key=True)`, `update_or_create()`) are part of Django core, already a project
dependency. No `npm view`/`pip index versions`/package-legitimacy check is needed because nothing new
is being added to `pyproject.toml`.

## Architecture Patterns

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│           sync_lco_observation_calendar.py  handle()  (write path)       │
├──────────────────────────────────────────────────────────────────────────┤
│  _build_event_fields() computes telescope_api_failed   (line 470)        │
│              │                                                           │
│              ▼                                                           │
│  fields.pop('telescope_api_failed')                      (line 604)      │
│              │                                                           │
│              ▼                                                           │
│  CalendarEvent.objects.get_or_create(url=url, defaults=fields)           │
│                                                            (line 615)     │
│              │                                                           │
│              ▼  [NEW] immediately after, same loop iteration             │
│  CalendarEventTelescopeLabel.objects.update_or_create(                   │
│      event=event, defaults={'is_verified': not telescope_api_failed})    │
└──────────────────────────────────────────────────────────────────────────┘
               │
               ▼ persisted, batch/off-request-path
┌──────────────────────────────────────────────────────────────────────────┐
│  solsys_code.models.CalendarEventTelescopeLabel  (NEW, first real model) │
│    event: OneToOneField(CalendarEvent, CASCADE, primary_key=True,        │
│           related_name='telescope_label_meta')                          │
│    is_verified: BooleanField(default=True)                              │
└──────────────────────────────────────────────────────────────────────────┘
               │
               ▼ read path, per calendar page load (on request path)
┌──────────────────────────────────────────────────────────────────────────┐
│  tom_calendar.views.render_calendar()  (installed, UNMODIFIED)           │
│    events = CalendarEvent.objects.filter(...)   -- no select_related     │
│    events = list(events)                                                │
└──────────────────────────────────────────────────────────────────────────┘
               │
               ▼ context dict -> template
┌──────────────────────────────────────────────────────────────────────────┐
│  src/templates/tom_calendar/partials/calendar.html  (OVERRIDE, MODIFIED) │
│  all-day branch (day.all_day_events):                                    │
│    {% if event.title|slice:":9" == "[QUEUED] " %} ... existing ...      │
│    [NEW] dashed-border class/style when                                  │
│      NOT event.telescope_label_meta.is_verified|default:True             │
│    [NEW] title="<tooltip text>" attribute on the fallback branch         │
│  timed branch (day.events): same [NEW] dashed-border + tooltip logic     │
└──────────────────────────────────────────────────────────────────────────┘
```

A reader can trace the primary use case end to end: a synced record's API outcome
(`telescope_api_failed`) flows from computation → persistence (sidecar row) → calendar-page render →
visual cue + tooltip, with `load_telescope_runs.py` events never entering the write path and defaulting
to "verified" at the read step.

### Recommended Project Structure

```
solsys_code/
├── models.py                                  # MODIFIED: add CalendarEventTelescopeLabel (first real model)
├── migrations/
│   └── 0001_calendareventtelescopelabel.py     # NEW: first real migration for this app
├── management/commands/
│   └── sync_lco_observation_calendar.py        # MODIFIED: +1 update_or_create call, line ~615-616
└── tests/
    └── test_sync_lco_observation_calendar.py   # MODIFIED: add sidecar-row assertions (no-churn, verified/fallback, classical-no-row)

src/templates/tom_calendar/partials/
└── calendar.html                               # MODIFIED: dashed-border class + tooltip on both
                                                 #            cal-event-all-day and cal-event-timed branches

docs/notebooks/pre_executed/
└── sync_lco_observation_calendar_demo.ipynb    # MODIFIED (CLAUDE.md convention): regenerate a cell
                                                 #   demonstrating the sidecar row write — see Project
                                                 #   Constraints below
```

No new app, no new `templatetags/` package is required for this phase specifically — DISPLAY-02's
direct-attribute-read pattern (`{{ event.telescope_label_meta.is_verified }}`) needs no `simple_tag`.
(A `calendar_display_extras.py` templatetags module is Phase 9's concern for `proposal_color`; if this
phase's planner judges the dashed-border/tooltip logic benefits from a tiny filter to avoid template
`{% if %}` sprawl, creating that module here is also reasonable and would simply be reused by Phase 9 —
Claude's discretion per CONTEXT.md.)

### Pattern 1: Sidecar write colocated with the existing `get_or_create`/diff/`save()` block

**What:** Add `CalendarEventTelescopeLabel.objects.update_or_create(event=event, defaults={'is_verified':
not telescope_api_failed})` as a standalone statement immediately after line 615's
`event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)`, inside the same
`for record in records:` loop. `telescope_api_failed` and `event` are both already in scope at that
point — no new variables need threading through.

**When to use:** This is the only call site that needs it — `load_telescope_runs.py`'s own
`get_or_create`/`.save()` block (confirmed at line 91/109, re-checked this session) has no
`telescope_api_failed` concept anywhere in the file and must NOT be touched.

**Example:**
```python
# sync_lco_observation_calendar.py, handle(), confirmed current line numbers this session:
event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)  # line 615, existing
if created:
    counters[record.facility]['created'] += 1
else:
    changed = any(getattr(event, field_name) != value for field_name, value in fields.items())
    if changed:
        for field_name, value in fields.items():
            setattr(event, field_name, value)
        event.save()
        counters[record.facility]['updated'] += 1
    else:
        counters[record.facility]['unchanged'] += 1

# NEW (Phase 8 / DISPLAY-01): always reconcile the sidecar row to the current
# telescope_api_failed signal, regardless of whether CalendarEvent's own fields
# changed -- kept as a separate statement, never folded into `fields` or `changed`.
CalendarEventTelescopeLabel.objects.update_or_create(
    event=event, defaults={'is_verified': not telescope_api_failed}
)
```

### Pattern 2: Defensive read of the reverse `OneToOneField` accessor in the template

**What:** `{{ event.telescope_label_meta.is_verified|default:True }}` is a direct attribute read — no
template tag needed for the boolean itself. `event.telescope_label_meta` raises
`CalendarEventTelescopeLabel.DoesNotExist` when no sidecar row exists (every `load_telescope_runs.py`
event, by design). Django's template engine silences this by default because `ObjectDoesNotExist` sets
`silent_variable_failure = True` [CITED: docs.djangoproject.com/en/5.2/ref/templates/api/#variables-and-lookups],
so `{{ }}` degrades to empty string rather than raising a 500 — **but the prior milestone research
explicitly flagged this as documented-but-not-yet-exercised-by-a-test in this project.** Re-verify with
a real template-rendering test in this phase rather than relying purely on the documented contract,
since `|default:True` on an empty-string fallback and `|default:True` on a "the attribute access raised"
fallback are subtly different code paths in Django's template engine.

**When to use:** Whenever the dashed-border conditional and tooltip text need the verified/fallback
boolean inside `calendar.html`.

**Recommended template idiom (defensive, removes ambiguity):**
```django
{% if event.telescope_label_meta.is_verified == False %}
  {# fallback-labeled: dashed border + tooltip #}
{% endif %}
```
Using `== False` (not bare truthiness) is deliberate: it makes "missing row" (silently empty
string/`None` from the silenced `DoesNotExist`) and "explicit `is_verified=True`" both fall through to
the `{% else %}`/no-dashed-border path, matching the documented default ("missing row renders as
verified"), while only an explicit, persisted `is_verified=False` triggers the fallback treatment. This
sidesteps relying on `|default:True` filter semantics entirely and is easier for a reviewer to read
correctly on first pass. **A template-rendering test asserting both cases (sidecar row absent → renders
solid border; sidecar row present with `is_verified=False` → renders dashed border + tooltip) is the
concrete verification step for this pattern, not an assumption.**

### Anti-Patterns to Avoid

- **Folding `is_verified` into the `fields` dict or the `changed` comparison:** `fields` only ever
  describes `CalendarEvent`'s own columns; the sidecar write must remain a separate ORM statement
  (Pitfall 3 below).
- **Adding a sidecar-creation call to `load_telescope_runs.py` "for symmetry":** confirmed by direct
  read this session — `load_telescope_runs.py` has no API call, no fallback concept, `parsed.telescope`
  is a deterministically-resolved trusted token (raises `ValueError` on ambiguity, never guesses). Do
  not add any code there.
- **Creating a new Django app for one model + one write-site change:** `solsys_code` is already
  installed, already has `models.py` and `migrations/` — use them.
- **A `post_save` signal on `CalendarEvent` as the write mechanism:** would also fire for the upstream
  `EventForm`/`create_event` view, which has no `telescope_api_failed` concept — would force an invented
  default for an unrelated code path. Use the explicit `update_or_create` call instead.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|--------------|-----|
| Attaching a field to a model you don't own | A custom field-injection/monkeypatch on `tom_calendar.models.CalendarEvent` | `OneToOneField(primary_key=True)` sidecar model | This is Django's own documented, supported pattern; monkeypatching a third-party model's class is unsupported and breaks on any `tomtoolkit` upgrade. |
| "Did this record's verification outcome change since last run" check | A custom diff/changed-flag computation for the sidecar | `update_or_create()`'s built-in compare-then-save | Django's `update_or_create` already only writes if the row is new or a field differs — no extra code needed to avoid spurious writes. |
| Detecting "no sidecar row" in the template | A custom `try/except DoesNotExist` Python helper function | Direct template comparison (`{% if event.telescope_label_meta.is_verified == False %}`) | Django's template engine already silences `ObjectDoesNotExist` by design; no Python-side wrapper needed unless the team prefers explicit code over relying on the silenced-exception contract (acceptable alternative, not required). |

**Key insight:** Every piece of this phase has a standard, already-documented Django answer. The only
genuine novelty for this codebase is that it is `solsys_code`'s *first* model/migration — there is no
novelty in the underlying Django mechanics themselves.

## Runtime State Inventory

> This phase is additive (new model, new migration, new template branch) — not a rename, refactor, or
> migration of existing identifiers. The Runtime State Inventory protocol triggers on rename/refactor
> phases; this is neither, so the full 5-category audit is not required. The two relevant checks are
> covered below for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | No existing `CalendarEvent` rows need a synthetic/backfilled sidecar value — `is_verified` needs no backfill, since the recommended pattern is "no row at all for events the new code never re-touches." Existing `CalendarEvent` rows created by prior `sync_lco_observation_calendar` runs will simply lack a sidecar row until the next sync run includes them, and the template's documented default (no row → verified) is the correct interim rendering. | None — confirmed no data migration needed. |
| Build artifacts | New migration file (`0001_calendareventtelescopelabel.py`) must be generated and committed — `solsys_code/migrations/` is currently `__init__.py`-only. | `./manage.py makemigrations solsys_code` then `./manage.py migrate`; commit the generated file. |

**Nothing found in remaining categories** (live service config, OS-registered state, secrets/env vars) —
this phase touches no external service configuration, no OS-level registration, and no secret/env var
names.

## Common Pitfalls

### Pitfall 1: Sidecar write conflated with `CalendarEvent`'s existing no-churn diff block

**What goes wrong:** Someone "completes the pattern" by stuffing `is_verified` into the `fields` dict
passed to `get_or_create`/compared in the `changed = any(...)` expression, since that's the one
well-known "don't churn" idiom in this file.

**Why it happens:** Pattern-matching on "no-churn" without reading the specific sidecar guidance; the
sidecar model has no such column on `CalendarEvent`, so this either errors or is silently dropped.

**How to avoid:** Keep `CalendarEventTelescopeLabel.objects.update_or_create(...)` as its own statement,
never merged into `fields`/`changed`. `update_or_create()` already does its own no-churn compare
internally.

**Warning signs:** A code review where `is_verified` appears as a key inside `fields`, or is assigned
via `setattr(event, ...)`.

### Pitfall 2: Sidecar staleness contract misunderstood as a bug

**What goes wrong:** `is_verified` is only updated for records included in a given sync run (e.g. a run
filtered with `--proposal`). A record outside that run's queryset keeps its prior value — correct
behavior, not staleness, but easy to misreport as "the sidecar didn't update."

**Why it happens:** No background re-check job exists (and none should be built — out of scope); the
sidecar's only writer is the per-record loop in `sync_lco_observation_calendar.py`.

**How to avoid:** Document the contract precisely in a code comment next to the `update_or_create` call:
"`is_verified` reflects the outcome of the most recent sync run that included this record, not
real-time state." This mirrors the staleness semantic `CalendarEvent.telescope`/`instrument` already
have.

**Warning signs:** A bug report claiming a record's verified/fallback status is "wrong" — check whether
a later run actually re-included that record before treating it as a defect.

### Pitfall 3: N+1 queries from the reverse-accessor read in the month-grid loop (accepted, not fixed, this phase)

**What goes wrong:** `render_calendar()` (confirmed, re-read this session: plain FBV, `events =
CalendarEvent.objects.filter(...)` then `list(events)`, no `select_related`/`prefetch_related`) means
`event.telescope_label_meta` triggers one extra `SELECT` per event with a sidecar row, per render.

**Why it happens:** No Python-level hook exists on `render_calendar()` to inject
`select_related('telescope_label_meta')` — confirmed by direct read this session, matching the prior
research exactly.

**How to avoid (this phase):** Per the locked CONTEXT.md discretion decision (DISPLAY-09 deferred to
v2), **accept this as-is** for current calendar-event volume — do not build a batching template tag in
Phase 8. If volume grows, the documented mitigation (a `{% load_telescope_label_meta %}` tag doing
`CalendarEventTelescopeLabel.objects.filter(event_id__in=ids)` once per render) is fully specified in
`.planning/research/ARCHITECTURE.md`'s "N+1 Query Risk" section for a future phase to pick up.

**Warning signs (for future revisit):** Page load noticeably slower on a busy month, or a query-count
test showing 1:1 scaling with event count.

### Pitfall 4: Shipping the dashed-border template change without the tooltip, or vice versa

**What goes wrong:** DISPLAY-02 and DISPLAY-03 are two separate success criteria (visual cue,
tooltip) but touch the exact same conditional branch in `calendar.html` — easy to add the dashed
border CSS and forget the `title="..."` attribute (or add the tooltip text but forget it needs to be
on an element that's actually hoverable, not just inside a flex child that doesn't receive `:hover`).

**How to avoid:** Add both the dashed-border class/style and the `title="..."` attribute in the same
task, on the same wrapping element (the `.cal-event-all-day`/`.cal-event-timed` div, consistent with
where the existing `[QUEUED]` override and `truncatechars` title text already live), and verify both
visually (border renders) and interactively (hover shows tooltip) before marking the task complete.

### Pitfall 5: Demo notebook gap (CLAUDE.md convention — already bitten this project twice)

**What goes wrong:** `sync_lco_observation_calendar.py`'s behavior changes (new sidecar write), but the
paired `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` is not included in
`files_modified` and not regenerated.

**Why it happens:** The notebook lives in a different directory than the code change and is easy to
scope out of a plan's `files_modified` if not deliberately included up front.

**How to avoid:** Per CLAUDE.md's explicit convention (already triggered twice — Phase 5 quick task
`260619-f7u`, Phase 6 quick task `260620-v9x`), the planner MUST scope
`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` into a task's `files_modified`
up front, with a cell demonstrating the new sidecar row write, regenerated via
`jupyter nbconvert --to notebook --execute --inplace` and committed with output.

**Warning signs:** A plan that lists `sync_lco_observation_calendar.py` in `files_modified` but not its
paired notebook.

## Code Examples

### Sidecar model definition

```python
# solsys_code/models.py — this app's first real model
from django.db import models

from tom_calendar.models import CalendarEvent


class CalendarEventTelescopeLabel(models.Model):
    """Sidecar record of whether a CalendarEvent's telescope label was live-verified
    against the LCO API or fallback-guessed (TELESCOPE-03/04). One row per
    CalendarEvent at most; no row at all means "verified" by documented default
    (e.g. classically-scheduled events from load_telescope_runs, which never go
    through telescope-label resolution).
    """

    event = models.OneToOneField(
        CalendarEvent,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='telescope_label_meta',
        verbose_name='Calendar event',
    )
    is_verified = models.BooleanField(
        default=True, verbose_name='Whether the telescope label was live-verified against the LCO API'
    )

    def __str__(self):
        return f'{"Verified" if self.is_verified else "Fallback"} label for {self.event.title}'
```

(`verbose_name` on every field mirrors this app's existing convention in
`solsys_code/solsys_code_observatory/models.py`'s `Observatory` model, confirmed this session.)

### Template dashed-border + tooltip branch (all-day events)

```django
{# src/templates/tom_calendar/partials/calendar.html, inside the day.all_day_events loop #}
{% if event.title|slice:":9" == "[QUEUED] " %}
<div class="cal-event-all-day" style="background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);">
{% elif event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day" style="background-color: {{ event.color }}; border: 2px dashed rgba(0, 0, 0, 0.65);"
     title="Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0).">
{% else %}
<div class="cal-event-all-day" style="background-color: {{ event.color }};">
{% endif %}
```

The same `{% elif event.telescope_label_meta.is_verified == False %}` branch must be added to the
`day.events` (timed-event, `cal-event-timed`) loop as well — per D-02, both render branches need the
dashed-border treatment.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Verification outcome only visible via `[UNVERIFIED]` title-text prefix (TELESCOPE-04, shipped Phase 07/07.1) | A queryable, structured `is_verified` field plus a visual border cue and hover tooltip — title-text prefix remains as the accessible fallback channel, not replaced | This phase (Phase 8) | Operators can now scan the calendar grid for fallback events without reading truncated title text; the text prefix still satisfies WCAG 1.4.1 as a non-color-or-shape-only fallback. |

**Deprecated/outdated:** Nothing in this phase deprecates the existing `[UNVERIFIED]` title-prefix
convention — it remains the accessible/textual channel; the sidecar model and visual cue are additive.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|----------------|
| A1 | Django's `ObjectDoesNotExist.silent_variable_failure = True` behavior degrades `{{ event.telescope_label_meta.is_verified }}` to empty/falsy in template rendering rather than raising a 500, for this project's exact installed Django 5.2.15 | Architecture Patterns, Pattern 2 | If wrong, every `load_telescope_runs`-created event (no sidecar row) would 500 on calendar render — high severity, but the recommended `== False` comparison plus a dedicated template-rendering test in this phase's verification step directly exercises this rather than relying on the documented contract alone, so the risk is mitigated by the phase's own test plan regardless. |

**If this table looks short:** Every other claim in this research was re-verified directly in this
session (Django/tomtoolkit versions via `pip show`/`python -c`, model shapes via direct source read,
line numbers via `grep -n`, template content via direct read) — not carried forward unverified from the
prior research date.

## Open Questions

None outstanding for this phase's scope. The two items the prior milestone research flagged as open
(N+1 mitigation scope, status-treatment mechanism) are both resolved: N+1 mitigation is explicitly
deferred per CONTEXT.md's locked discretion decision (accept-as-is, DISPLAY-09 to v2); status-treatment
mechanism is Phase 9's concern, not this phase's (D-03 explicitly reserves dash-style for this phase's
verification signal only).

## Environment Availability

No external dependencies beyond what's already installed and confirmed (Django 5.2.15, `tomtoolkit`
3.0.0a9, SQLite local dev DB). No new tool, service, or runtime is introduced by this phase.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django test runner (`django.test.TestCase`), DB-dependent — per CLAUDE.md, this phase's tests belong in `solsys_code/tests/`, NOT the separate pytest suite (`tests/`, `src/`, `docs/`) |
| Config file | None dedicated — uses `manage.py test` against `DJANGO_SETTINGS_MODULE=src.fomo.settings` |
| Quick run command | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|---------------------|--------------|
| DISPLAY-01 | Sync writes a sidecar row matching `telescope_api_failed` outcome (verified/fallback) | unit/integration | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar -v2` | ❌ Wave 0 — new test method needed in existing file |
| DISPLAY-01 | No sidecar row created for `load_telescope_runs`-created events | unit/integration | `./manage.py test solsys_code.tests.test_load_telescope_runs -v2` | ❌ Wave 0 — new test method needed in existing file |
| DISPLAY-01 | Re-running sync on unchanged records does not create duplicate sidecar rows or churn `CalendarEvent.modified` | unit/integration (no-churn regression) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.<NewNoChurnTest> -v2` | ❌ Wave 0 — new test method, mirrors existing `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` pattern (confirmed at line 320) |
| DISPLAY-02 | Fallback-labeled event renders with dashed-border CSS (all-day and timed branches); verified event renders with solid/default border | template-rendering / view-level integration test | new test method asserting rendered HTML contains the dashed-border marker for a fallback event and not for a verified event | ❌ Wave 0 — new test file/method needed (no template-rendering test precedent exists yet in this codebase for `calendar.html`) |
| DISPLAY-02 | A `load_telescope_runs` event (no sidecar row) renders as verified, not as a template error | template-rendering / view-level integration test | same new test file, assert no exception and solid-border rendering for a no-sidecar-row event | ❌ Wave 0 — same new test file |
| DISPLAY-03 | Hovering a fallback-labeled event shows a tooltip with the verification detail | template-rendering test (assert `title="..."` attribute present in rendered HTML) | same new test file, assert the tooltip text substring appears in rendered HTML for a fallback event | ❌ Wave 0 — same new test file |

### Sampling Rate

- **Per task commit:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`
- **Per wave merge:** `./manage.py test solsys_code`
- **Phase gate:** `./manage.py test solsys_code` green, plus `ruff check .` / `ruff format --check .` clean (per `08-CONTEXT.md`'s phase boundary and the project's standing quality gates), before `/gsd-verify-work`.

### Wave 0 Gaps

- [ ] No template-rendering test exists yet for `src/templates/tom_calendar/partials/calendar.html` in this codebase — this phase needs to establish that pattern (e.g. `Client.get('/calendar/...')` and assert on `response.content`, or `django.template.loader.render_to_string` with a constructed context). Use `django.test.Client` with a real `CalendarEvent`/`CalendarEventTelescopeLabel` fixture, consistent with `solsys_code/solsys_code_observatory/tests/test_views.py`'s existing `Client`-based pattern (confirmed precedent exists in this codebase for view-level tests, just not for `calendar.html` specifically).
- [ ] No `assertNumQueries` precedent exists yet in this codebase (confirmed via grep this session) — not required for this phase (N+1 mitigation deferred), but if a future phase revisits DISPLAY-09, that phase will be the first to introduce this pattern here.
- [ ] First-ever migration for `solsys_code` — confirm `./manage.py makemigrations solsys_code` produces a clean migration and `./manage.py migrate` runs clean on a fresh DB as part of Wave 0/first task, not assumed.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|----------------|---------|---------------------|
| V2 Authentication | No | Phase touches no auth surface — `is_verified` is an internal computed flag, not user-supplied. |
| V3 Session Management | No | No session-related change. |
| V4 Access Control | No | No new permission boundary — the calendar view's existing access control (TOM Toolkit's standard `AUTH_STRATEGY='READ_ONLY'`, per CLAUDE.md) is unchanged; this phase adds no new view, only a model field and template branch read through the existing view. |
| V5 Input Validation | Yes (narrow) | `is_verified` is a `BooleanField` written only by trusted, internal code (`sync_lco_observation_calendar.py`'s own computed `telescope_api_failed`) — no external/untrusted input reaches this field directly. The only user-influenceable string anywhere near this phase's render path is `event.proposal`/`event.title`, both already-existing `CalendarEvent` fields rendered today via Django's autoescaping (`{{ event.title }}`) — this phase introduces no new raw-string interpolation into `style=`/`title=` attributes beyond a fixed, hardcoded tooltip sentence (D-04) and a fixed dashed-border CSS literal, neither derived from variable/untrusted data. |
| V6 Cryptography | No | Not applicable to this phase. |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Stored-XSS-adjacent risk via raw string interpolation into an HTML attribute (`style=`/`title=`) | Tampering | Already mitigated by construction: this phase's new template additions interpolate only a fixed tooltip sentence (D-04's locked text) and a fixed CSS border literal — never raw `event.proposal`/`event.title`/any DB-sourced free text — into the new `title=`/`style=` attributes. Django's template autoescaping continues to protect the existing `{{ event.title }}` text-content rendering elsewhere in the same block, unchanged by this phase. |
| Orphaned sidecar row after `CalendarEvent` deletion | Tampering / Repudiation (data-integrity adjacent, not a classic security threat) | `on_delete=models.CASCADE` on the `OneToOneField` ensures the sidecar row is deleted automatically if its parent `CalendarEvent` is ever deleted — no orphan risk, no manual cleanup code needed. |

## Sources

### Primary (HIGH confidence)
- Direct read this session: `/home/tlister/git/fomo_devel/solsys_code/management/commands/sync_lco_observation_calendar.py` (grep for line numbers — confirmed `telescope_api_failed` computed at line 470, popped at line 604, `get_or_create` at line 615, matching prior research exactly)
- Direct read this session: `/home/tlister/git/fomo_devel/solsys_code/management/commands/load_telescope_runs.py` (grep — confirmed no `telescope_api_failed`/API-call concept anywhere)
- Direct read this session: `/home/tlister/git/fomo_devel/src/templates/tom_calendar/partials/calendar.html` (full file — confirmed `[QUEUED]` branch at lines 158-162, `{{ event.color }}` only in all-day branch, `{% load tz calendar_tags %}` at line 101)
- Direct read this session: `/home/tlister/git/fomo_devel/solsys_code/models.py` (confirmed all-comment, no real model) and `solsys_code/migrations/` (confirmed `__init__.py`-only)
- Direct read this session: installed `tom_calendar` package (`models.py`, `views.py`) at
  `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_calendar/` — confirmed `CalendarEvent`
  field shape, hardcoded `Meta.app_label = 'tom_calendar'`, `render_calendar()`'s lack of a
  `select_related`/`extra_context` hook
- `python -c "import django; print(django.VERSION)"` → `(5, 2, 15, 'final', 0)` [VERIFIED: local environment, this session]
- `pip show tomtoolkit` → `Version: 3.0.0a9` [VERIFIED: local environment, this session]
- Direct read this session: `/home/tlister/git/fomo_devel/solsys_code/solsys_code_observatory/models.py` (confirmed `verbose_name` field convention)
- `.planning/research/STACK.md`, `ARCHITECTURE.md`, `FEATURES.md`, `PITFALLS.md`, `SUMMARY.md` (this milestone, written 2026-06-24, HIGH confidence — re-verified, not contradicted, by this session's direct reads)
- `.planning/phases/08-telescope-label-verification-sidecar/08-CONTEXT.md` (locked decisions D-01 through D-04, Claude's Discretion)
- `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `/home/tlister/git/fomo_devel/CLAUDE.md`

### Secondary (MEDIUM confidence)
- [Django one-to-one relationships example docs](https://docs.djangoproject.com/en/5.2/topics/db/examples/one_to_one/) — `OneToOneField` reverse-accessor `DoesNotExist` semantics
- [Django QuerySet API reference](https://docs.djangoproject.com/en/5.2/ref/models/querysets/#update-or-create) — `update_or_create()` signature
- [Django template variables and lookups](https://docs.djangoproject.com/en/5.2/ref/templates/api/#variables-and-lookups) — `silent_variable_failure` contract (flagged as not yet exercised by a project-specific test — see Assumptions Log A1)

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — re-verified by direct package/version inspection this session, matches prior milestone research exactly, no drift detected.
- Architecture: HIGH — exact line numbers re-confirmed by grep this session; the integration points (write site, read site, no-op for `load_telescope_runs.py`) are unchanged from the prior research.
- Pitfalls: HIGH — grounded in this project's actual code and prior bug history (Phase 5/6 notebook gaps, the `[QUEUED]` override precedent), re-confirmed this session.

**Research date:** 2026-06-24
**Valid until:** 30 days (stable Django/tomtoolkit stack, no fast-moving dependency in this phase's scope) — re-verify line numbers if any other phase touches `sync_lco_observation_calendar.py` or `calendar.html` before this phase executes.
