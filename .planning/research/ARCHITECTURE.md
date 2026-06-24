# Architecture Research

**Domain:** Django calendar visual-treatment + cross-app model extension (FOMO v1.4 — DISPLAY-01/DISPLAY-02)
**Researched:** 2026-06-24
**Confidence:** HIGH

This file was previously stale (dated 2026-06-18, about an unrelated multi-facility/multi-proposal
sync-generalization topic from the v1.3 milestone). It has been fully overwritten with v1.4-specific
findings below. Builds directly on `.planning/research/STACK.md` (written today, same milestone) —
this file does not re-derive the stack choices (hashlib, OneToOneField sidecar, template tag library);
it answers exactly where each piece lives, exactly which lines of existing code it touches, and how the
two features sequence as build phases.

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Management Commands (write path)                      │
├──────────────────────────────────────────────────────────────────────────┤
│  sync_lco_observation_calendar.py        load_telescope_runs.py          │
│  _build_event_fields() computes              (no telescope_api_failed    │
│  telescope_api_failed (line 470,              concept at all — classical │
│  popped at line 604) ──────┐                  schedule, not API-derived) │
│                             │                                            │
│                             ▼                                            │
│              CalendarEvent.objects.get_or_create(url=url, ...)           │
│                       (line 615, existing)                               │
│                             │                                            │
│                             ▼  [NEW] DISPLAY-02 write, same call site    │
│        CalendarEventTelescopeLabel.objects.update_or_create(             │
│            event=event, defaults={'is_verified': not telescope_api_failed})│
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  persisted
┌──────────────────────────────────────────────────────────────────────────┐
│                          Data Layer (Django ORM)                          │
├──────────────────────────────────────────────────────────────────────────┤
│  tom_calendar.models.CalendarEvent (installed, not owned)                │
│    title, description, start_time, end_time, url, telescope,             │
│    instrument, proposal, user, target_list FK, color (property)          │
│                             ▲ OneToOneField(primary_key=True)             │
│  [NEW] solsys_code.models.CalendarEventTelescopeLabel                    │
│    event (PK/FK), is_verified (bool)                                     │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  read path
┌──────────────────────────────────────────────────────────────────────────┐
│                  tom_calendar.views.render_calendar()                    │
│  (installed, FBV, no extra_context hook, no get_queryset() to override)  │
│  events = CalendarEvent.objects.filter(...) -- NO select_related/        │
│  prefetch_related today; materialized via list(events)                  │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  context dict -> template
┌──────────────────────────────────────────────────────────────────────────┐
│        src/templates/tom_calendar/partials/calendar.html (OVERRIDE)      │
│  {% load tz calendar_tags %}              [NEW] add calendar_display_extras│
│  all-day branch: {{ event.color }}   -> [NEW] {% proposal_color %} tag   │
│  timed branch: no color call today   -> [NEW] add {% proposal_color %}  │
│  [QUEUED] prefix branch (existing, string-slice on event.title)          │
│  [NEW] status visual treatment: extend same string-slice pattern for    │
│  [UNVERIFIED]/[EXPIRED]/[CANCELLED]/[FAILED] prefixes                   │
│  [NEW, DISPLAY-02 display]: {{ event.telescope_label_meta.is_verified }}│
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `solsys_code/models.py` | Owns the new `CalendarEventTelescopeLabel` sidecar model (DISPLAY-02 persisted flag) | `OneToOneField(CalendarEvent, on_delete=CASCADE, primary_key=True, related_name='telescope_label_meta')` + one `BooleanField` |
| `solsys_code/migrations/0001_*.py` (new) | Creates the sidecar table | Django-generated via `./manage.py makemigrations solsys_code` — first real migration in this app's currently-empty migrations folder |
| `solsys_code/templatetags/calendar_display_extras.py` (new module, new package) | DISPLAY-01's color-hash and status-treatment logic, exposed as template tags | `@register.simple_tag def proposal_color(proposal: str) -> str`, mirroring `tom_calendar.templatetags.calendar_tags.target_list_color` |
| `sync_lco_observation_calendar.py` (`_build_event_fields`, `handle()`) | Already computes `telescope_api_failed` (line 470) per record; `handle()` already pops it (line 604) | Add one `update_or_create` call into the existing `get_or_create` block (line 615) — modification, not a new module |
| `load_telescope_runs.py` | DISPLAY-02 **no-op** — never calls a live API, has no fallback concept | No change required; confirmed by direct read (see Integration Points below) |
| `src/templates/tom_calendar/partials/calendar.html` | Renders both DISPLAY-01 (color/status) and DISPLAY-02 (verified/fallback indicator) | Template-only changes: swap `{{ event.color }}` for `{% proposal_color event.proposal %}`, extend the existing `{% if event.title|slice:... %}` prefix-branch pattern, add `{{ event.telescope_label_meta.is_verified }}` read |

## Recommended Project Structure

```
solsys_code/
├── models.py                          # MODIFIED: add CalendarEventTelescopeLabel
├── migrations/
│   └── 0001_calendareventtelescopelabel.py   # NEW: first real migration for this app
├── templatetags/                      # NEW package (solsys_code has none today)
│   ├── __init__.py                    # NEW
│   └── calendar_display_extras.py     # NEW: proposal_color + status-treatment tag(s)
├── management/commands/
│   ├── sync_lco_observation_calendar.py   # MODIFIED: +1 call site (DISPLAY-02 write)
│   └── load_telescope_runs.py             # UNCHANGED (confirmed, see below)
└── tests/
    ├── test_sync_lco_observation_calendar.py  # MODIFIED: assert sidecar row written/updated
    └── test_calendar_display_extras.py        # NEW: unit tests for proposal_color/status tag

src/templates/tom_calendar/partials/
└── calendar.html                      # MODIFIED: load new tag lib, swap color call,
                                        #           extend prefix-branch, add verified read
```

### Structure Rationale

- **`solsys_code/models.py`, not a new app:** STACK.md already rejected a new Django app for this
  milestone (no new `INSTALLED_APPS` entry, no new `AppConfig`). `solsys_code` is already installed,
  already has a `models.py` (currently all-comment) and a `migrations/` folder (currently
  `__init__.py`-only) — the sidecar model is this app's first real model and first real migration.
- **`solsys_code/templatetags/` as a new package, not `src/templatetags/`:** the project has two
  existing template-tag modules at `src/templatetags/` (`fomo_extras.py`, `solsys_code_extras.py`),
  registered as project-level/`src`-rooted tags. But the tag this milestone needs is conceptually
  scoped to `solsys_code`'s own domain logic (the same module that owns `SITE_TELESCOPE_MAP`-style
  business rules and the sidecar model), and Django auto-discovers any installed app's
  `templatetags/` package without a settings change — `solsys_code` being in `INSTALLED_APPS` is
  sufficient. Either location works mechanically; this research recommends colocating the tag with the
  model it reads (`solsys_code/`) over the project-template-only `src/templatetags/`, since
  `calendar_display_extras.py` is pure presentation-of-domain-data logic, not a generic
  project/template utility like `fomo_extras.py`. **A `calendar_display.py` plain helper module (not
  a templatetags module) is not recommended as the primary location** — Django template tags must be
  registered inside a `templatetags/` package to be loadable via `{% load %}`; a bare helper module
  would need a thin templatetags wrapper around it anyway, so it is simpler to write `proposal_color`
  directly as a `@register.simple_tag` function in `calendar_display_extras.py` (matching STACK.md's
  own recommendation) rather than introduce an extra indirection layer.
- **One migration file, not a data migration:** `is_verified` needs no backfill — STACK.md's
  recommended pattern (a) is "no sidecar row at all for classically-scheduled events," so no
  existing `CalendarEvent` row needs a synthetic sidecar value at migration time.

## Architectural Patterns

### Pattern 1: Sidecar model write colocated with the existing `get_or_create`/diff/`save()` block

**What:** `CalendarEventTelescopeLabel.objects.update_or_create(event=event, defaults={'is_verified':
not telescope_api_failed})` is added as a new line immediately after the existing
`CalendarEvent.objects.get_or_create(url=url, defaults=fields)` call, inside the same `for record in
records:` loop, in `sync_lco_observation_calendar.py`'s `handle()` method.

**When to use:** Whenever a `CalendarEvent` write site in this codebase also needs to persist
DISPLAY-02's verified/fallback signal — currently exactly one call site qualifies
(`sync_lco_observation_calendar.py`). `load_telescope_runs.py`'s `get_or_create` call site (its own
`handle()`, lines 91-100) does **not** qualify — confirmed by direct read, see Integration Points.

**Trade-offs:** Pro — keeps both writes (`CalendarEvent` and its sidecar) inside the same loop
iteration and same exception-handling scope, so a record that fails extraction/raises before reaching
this point never produces an orphaned sidecar row, and a record that succeeds always gets both rows
written together (or both left unchanged on a no-op re-run). Con — two separate ORM calls per record
instead of one, but this is the unavoidable cost of not owning `CalendarEvent`'s table (no single
`save()` can write both tables at once without a custom manager/signal layer that STACK.md already
ruled out as unnecessary for this milestone's two known producers).

**Example:**
```python
# sync_lco_observation_calendar.py, handle(), immediately after the existing
# CalendarEvent.objects.get_or_create(...) call (today's line 615) and the
# created/updated/unchanged counter logic that follows it:

event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)
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

# NEW: DISPLAY-02 — always reconcile the sidecar row to the current
# telescope_api_failed signal, regardless of whether CalendarEvent itself changed
# (a record's CalendarEvent fields can be unchanged while its verification
# outcome flips between runs only in edge cases, but update_or_create is cheap
# and avoids ever needing a separate "did is_verified change" diff).
CalendarEventTelescopeLabel.objects.update_or_create(
    event=event, defaults={'is_verified': not telescope_api_failed}
)
```

### Pattern 2: Template-tag-mediated computed display value vs. direct attribute read

**What:** DISPLAY-01's color needs a `@register.simple_tag` (`proposal_color`) because it requires
*computation* (hash → palette index) that plain Django template syntax cannot express. DISPLAY-02's
verified/fallback flag needs **no tag at all** — `{{ event.telescope_label_meta.is_verified }}` is a
direct attribute read through the reverse `OneToOneField` accessor, which Django template syntax
resolves natively (attribute lookup, no parentheses needed in template syntax).

**When to use:** Default to a direct template attribute read whenever the value is already a stored
field reachable by FK/O2O traversal; reach for a `simple_tag` only when the template needs to *derive*
a value (hashing, conditional branching beyond what `{% if %}`/template filters comfortably express).

**Trade-offs:** A direct attribute read is simpler and needs no `{% load %}` addition, but
`event.telescope_label_meta` raises `CalendarEventTelescopeLabel.DoesNotExist` when no sidecar row
exists for that event (classically-scheduled events, per STACK.md's recommended option (a) of never
writing a row for them) — **this is a hard `{{ }}` rendering trap, not a soft `None`.** Django's
template engine silences attribute-access exceptions that are explicitly listed in
`silent_variable_failure = True` on the exception class; `ObjectDoesNotExist` (the base class
`DoesNotExist` inherits from) sets this to `True` by Django's own design specifically so
`{{ obj.reverse_o2o.field }}` degrades to the engine's empty-string behavior in templates instead of
raising a 500. **Verify this assumption directly against the installed Django version before relying
on it in the template** (HIGH-confidence per Django's documented contract for `ObjectDoesNotExist`,
but this project's exact behavior should be exercised by a template-rendering test in the phase that
implements DISPLAY-02's display side, not merely assumed from documentation). A defensive
`{% if event.telescope_label_meta.is_verified|default:True %}`-style read, or a tiny
`@register.simple_tag` wrapper that explicitly catches `DoesNotExist` and returns `True`, removes any
ambiguity and is the safer choice if the build phase wants zero reliance on Django's silent-failure
contract.

### Pattern 3: Reusing the existing title-prefix string-slice convention for status visual treatment

**What:** `calendar.html` already branches on `{% if event.title|slice:":9" == "[QUEUED] " %}` to
de-emphasize queued events (the v1.2 fix). DISPLAY-01's "status-driven visual treatment
(opacity/border/striping)" should extend this exact pattern rather than introduce a new data source —
the signal already exists as a title-prefix vocabulary written by `_title_for()` in
`sync_lco_observation_calendar.py` (`[QUEUED]`, `[UNVERIFIED]`, `[EXPIRED]`, `[CANCELLED]`,
`[FAILED]`, or no prefix for clean/placed/completed) and by `load_telescope_runs.py` (no prefix at
all — classical runs are always "placed," never queued/fallback/terminal).

**When to use:** Any time the template needs to branch on which of these five-or-six states an event
is in. Recommend consolidating the prefix-detection logic into the new
`calendar_display_extras.py` tag/filter module (e.g. a `status_css_class(title)` filter returning a
CSS class name) rather than letting `calendar.html` accumulate more inline `{% if %}` chains —
the existing single `[QUEUED]` check is fine inline, but DISPLAY-01 will roughly double the number of
prefixes to detect, which is the point at which inline template conditionals become harder to read
than a tested Python filter.

**Trade-offs:** Pro — zero new model fields, zero migration, reuses a vocabulary that is already
unit-tested in `test_sync_lco_observation_calendar.py` (`_title_for` tests). Con — couples the visual
treatment to a string convention (title-prefix text) rather than a structured field; if a future
milestone wants to query/filter calendar events by status programmatically (not just display them),
a real status field would be the better long-term answer — out of scope for this milestone per
PROJECT.md's existing framing of DISPLAY-01 as the deferred "status-aware coloring" todo.

## Data Flow

### Write Flow (DISPLAY-02)

```
sync_lco_observation_calendar.py handle()
    ↓
_build_event_fields() computes telescope_api_failed (existing, line 470)
    ↓
handle() pops telescope_api_failed from fields dict (existing, line 604)
    ↓
CalendarEvent.objects.get_or_create(url=url, defaults=fields)  (existing, line 615)
    ↓
[NEW] CalendarEventTelescopeLabel.objects.update_or_create(
          event=event, defaults={'is_verified': not telescope_api_failed})
    ↓
Sidecar row created/updated in solsys_code's own table
```

`load_telescope_runs.py` never enters this flow — confirmed below.

### Read Flow (DISPLAY-01 + DISPLAY-02, calendar page render)

```
User loads /calendar/ (htmx partial or full page)
    ↓
tom_calendar.views.render_calendar()  (installed, unmodified)
    ↓
events = CalendarEvent.objects.filter(start_time__date__lte=..., end_time__date__gte=...)
    (NO select_related/prefetch_related today)
    ↓
events = list(events)  (materialized once, then sliced into per-day all_day_events/events lists)
    ↓
context = {..., "weeks": weeks_with_events, ...}
    ↓
render(request, "tom_calendar/partials/calendar.html", context)
    ↓
[OVERRIDDEN] src/templates/tom_calendar/partials/calendar.html
    ↓
{% load tz calendar_tags calendar_display_extras %}  [NEW: add calendar_display_extras]
    ↓
For each event in day.all_day_events / day.events:
    {% proposal_color event.proposal %}        [NEW tag call, DISPLAY-01]
    {% if event.title|slice:... %}...{% endif %}  [extended, DISPLAY-01 status treatment]
    {{ event.telescope_label_meta.is_verified }}   [NEW direct read, DISPLAY-02]
```

### Key Data Flows

1. **DISPLAY-02 write-then-read round trip:** the management command writes the sidecar row once per
   sync run (batch, off the request path); the template reads it once per event per calendar-page
   render (on the request path, potentially many times per day if the calendar is viewed often). This
   split is the standard write-rarely/read-often shape and is exactly why the N+1 risk below matters —
   the write side is cheap (one extra query per record, in a batch job), but the read side runs inside
   a user-facing request.
2. **DISPLAY-01 has no persisted write at all** — `proposal` is already a `CalendarEvent` field
   (written by both management commands today), so DISPLAY-01 is a pure read-side/template-side
   feature. No migration, no command change, no new counter. This is the key asymmetry between the two
   features' build complexity.

## N+1 Query Risk (flagged per quality gate)

**Confirmed by direct read of the installed `tom_calendar/views.py`:** `render_calendar()` builds
`events = CalendarEvent.objects.filter(...)` with no `select_related`/`prefetch_related`, then calls
`list(events)` to materialize it once before slicing into per-day buckets. This means:

- **DISPLAY-01's `{% proposal_color event.proposal %}`:** `event.proposal` is a plain `CharField`
  already loaded by the base queryset — **no extra query per event.** Safe as-is, no N+1 risk.
- **DISPLAY-02's `{{ event.telescope_label_meta.is_verified }}`:** `telescope_label_meta` is a
  **reverse** `OneToOneField` accessor. Django does not (and cannot, without an explicit
  `select_related('telescope_label_meta')` or `prefetch_related('telescope_label_meta')` on the
  queryset) eagerly load reverse-O2O relations by default. **This is a genuine N+1 risk:** rendering a
  month view with, say, 60 events would issue up to 60 additional `SELECT` queries (one per event with
  a sidecar row), each a single-row PK lookup.
- **Why this can't be fixed by overriding `render_calendar()`'s queryset:** STACK.md already confirmed
  (and this research's own direct read of `views.py` confirms identically) `render_calendar` is a
  plain function-based view with a fixed context dict and **no `extra_context`/subclass/`get_queryset()`
  hook** — there is no Python-level seam to inject `.select_related('telescope_label_meta')` into the
  installed view's queryset construction without monkeypatching or fully reimplementing the view
  (out of scope; the project's established customization seam for this view is the template override
  only, per the existing `[QUEUED]` precedent and per STACK.md's "Version Compatibility" finding).
- **Practical severity:** LOW in absolute terms for this project's current scale — a single month's
  events are typically a few dozen, not thousands, and SQLite single-row PK lookups are sub-millisecond
  — but it is a real, identifiable inefficiency that a future scale-up (e.g. syncing `ALL` proposals
  across both LCO and SOAR, the SELECT-03 capability already shipped in v1.3) could make noticeable.
- **Mitigation options for the DISPLAY-02 build phase to consider (no action required at research
  time, this is a phase-planning input):**
  1. **Accept it** — simplest, matches this project's current calendar-event volume, consistent with
     "don't over-engineer for scale this project doesn't have" (STACK.md's general posture).
  2. **Bulk-prefetch in a custom template tag**, e.g. a `{% load_telescope_label_meta events %}`
     tag called once at the top of `calendar.html` that does
     `CalendarEventTelescopeLabel.objects.filter(event__in=events)` in a single query and attaches
     results onto each event object as a plain Python attribute before the per-day loop runs — avoids
     touching the installed view, stays within the template-override seam, and turns N+1 into 1+1.
     This is the most idiomatic fix *given* the constraint that `render_calendar()` itself can't be
     modified.
  3. **Denormalize `is_verified` onto a `CalendarEvent`-adjacent cache** — rejected; reintroduces the
     "can't add a column to a model you don't own" problem DISPLAY-02 already solved by going with a
     sidecar model in the first place.

  Recommend flagging option 2 as the concrete mitigation if/when DISPLAY-02's phase plan reaches
  the template-read side, since it is a small, self-contained addition (one more tag in the same new
  `calendar_display_extras.py` module) that closes the gap without expanding scope elsewhere.

## Anti-Patterns

### Anti-Pattern 1: Adding a new Django app for this milestone's two small additions

**What people do:** Create a `calendar_display` app with its own `models.py`, `apps.py`,
`migrations/`, `templatetags/`, and an `INSTALLED_APPS` entry, treating "new feature" as "new app."

**Why it's wrong:** STACK.md already explicitly rejects this (see its "What NOT to Use" table) — both
DISPLAY-01 and DISPLAY-02 are small, single-purpose additions that belong naturally alongside the two
management commands already living in `solsys_code`, which is already installed, already has the
folders needed (`models.py`, `migrations/`). A new app adds settings-file churn and a second migration
history to maintain for what is functionally one model and one template-tag module.

**Do this instead:** Add the sidecar model to `solsys_code/models.py`; add a new
`solsys_code/templatetags/` package (this app currently has none, but adding one needs no settings
change — Django auto-discovers any installed app's `templatetags/`).

### Anti-Pattern 2: Writing the DISPLAY-02 sidecar row via a `post_save` signal on `CalendarEvent`

**What people do:** Reach for `@receiver(post_save, sender=CalendarEvent)` to auto-create/update the
sidecar row whenever any `CalendarEvent` is saved, reasoning that this guarantees consistency
regardless of which code path created the event.

**Why it's wrong:** STACK.md already flags this as a reasonable *supplementary* mechanism but not the
*primary* path — a signal handler on `CalendarEvent.post_save` would fire for the
**upstream `EventForm`/`create_event` view** too (the manual "+ New Event" button in `calendar.html`,
confirmed present in the installed `tom_calendar/views.py`), and that code path has no
`telescope_api_failed` concept at all — a signal would need to invent a default (`is_verified=True`?
skip entirely?) for events it knows nothing about, adding implicit behavior at a write site this
milestone's two known producers (`sync_lco_observation_calendar.py`, `load_telescope_runs.py`) don't
need help from.

**Do this instead:** An explicit `update_or_create` call colocated with the existing
`get_or_create`/conditional-`save()` block in `sync_lco_observation_calendar.py`'s `handle()` — the
one and only producer of `telescope_api_failed`-bearing events. `load_telescope_runs.py` writes no
sidecar row at all (see Integration Points).

### Anti-Pattern 3: Letting `calendar.html`'s inline `{% if %}` prefix-detection chain grow unbounded

**What people do:** Keep adding `{% if event.title|slice:":N" == "[X] " %}...{% elif %}...{% endif %}`
branches directly in the template as each new status/prefix is added (the existing single `[QUEUED]`
check is fine; DISPLAY-01 roughly doubles the prefix vocabulary to detect — `[UNVERIFIED]`,
`[EXPIRED]`, `[CANCELLED]`, `[FAILED]`, plus the existing `[QUEUED]` and the no-prefix clean case).

**Why it's wrong:** Template-embedded string-slicing logic is untested by the existing
`test_sync_lco_observation_calendar.py` suite (which tests `_title_for()`'s *output strings*, not how
a template subsequently re-parses them) and becomes harder to keep in sync with
`_FAILURE_PREFIX_BY_STATUS`'s vocabulary as that dict grows (the file's own comment already flags this
dict as a "hand-typed snapshot... if `LCOFacility` ever adds a new failure state, update this dict
too").

**Do this instead:** Move the prefix-to-visual-treatment mapping into a tested Python filter/tag in
`calendar_display_extras.py` (e.g. `status_css_class`), so the single source of truth for "what
prefixes exist and what they mean visually" lives in one Python module with unit tests, not scattered
across template conditionals.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| None — DISPLAY-01/DISPLAY-02 touch no external API | n/a | Both features operate entirely on data already persisted by the two existing management commands; no new network call, no new credential. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `sync_lco_observation_calendar.py` ↔ `solsys_code.models.CalendarEventTelescopeLabel` | Direct ORM call (`update_or_create`) | **Confirmed exact write point by direct read:** `telescope_api_failed` is computed at line 470 inside `_build_event_fields()`'s return dict, popped from `fields` at line 604 in `handle()`, and the existing `CalendarEvent.objects.get_or_create(url=url, defaults=fields)` call is at line 615. The new sidecar write goes immediately after line 615's `event, created = ...` assignment (inside the same `for record in records:` loop, before the loop's counter-increment `if created:`/`else:` block, or immediately after it — either ordering works since the sidecar write only needs `event` and `telescope_api_failed`, both already in scope). |
| `load_telescope_runs.py` ↔ DISPLAY-02 | **None — confirmed by direct read, no integration needed** | Read in full: `handle()` builds `title`/`description`/`start_time`/`end_time` purely from `parse_run_line()`'s parsed tokens and `sun_event()`'s ephemeris output (`solsys_code/telescope_runs.py`) — there is no LCO/SOAR API call anywhere in this file, no `telescope_api_failed`-equivalent boolean, no fallback-vs-verified distinction of any kind. `parsed.telescope` is a trusted, deterministically-parsed token (resolved via exact/prefix match against `SITES.keys()` in `telescope_runs.py`, raising `ValueError` on ambiguity — never a "best guess"). STACK.md's recommended option (a) — never create a sidecar row for these events — requires zero code change to this file. This file should **not** appear in `files_modified` for the DISPLAY-02 phase plan. |
| `calendar.html` ↔ `calendar_display_extras.py` (new tag module) | `{% load tz calendar_tags calendar_display_extras %}` then `{% proposal_color event.proposal %}` | Mirrors the existing `{% load tz calendar_tags %}` / `{% target_list_color ... %}` precedent already in this exact template (line 101, confirmed by direct read). Apply the tag call in **both** the all-day branch (`day.all_day_events`, where `{{ event.color }}` is used today at line 161) **and** the timed branch (`day.events`, which calls no color logic at all today — confirmed by direct read, lines 170-183) — STACK.md already flags this asymmetry; this milestone should decide explicitly whether timed events get colored too rather than leaving the gap unaddressed. |
| `calendar.html` ↔ `CalendarEventTelescopeLabel` (reverse O2O) | `{{ event.telescope_label_meta.is_verified }}` — direct template attribute read, **no tag needed** | Confirmed: Django template syntax resolves `.` as attribute/dict/index lookup automatically; a reverse `OneToOneField` accessor is just an attribute. **However:** flagged N+1 risk above — `render_calendar()`'s queryset has no `select_related`/`prefetch_related` and cannot be modified at the Python level (no hook). If the DISPLAY-02 phase wants to close this gap, the fix must live in the template layer (a bulk-prefetch tag, see N+1 section above), not in `tom_calendar`'s view. |
| `CalendarEvent` ↔ `CalendarEventTelescopeLabel` | `OneToOneField(CalendarEvent, on_delete=CASCADE, primary_key=True)` | Confirmed via STACK.md's direct read of the installed `tom_calendar.models.CalendarEvent` — hardcoded `Meta.app_label = 'tom_calendar'`, not abstract, owns its own migrations. The sidecar model's migration lives entirely in `solsys_code/migrations/`, touching zero files inside the installed `tomtoolkit` package. |

## Suggested Phase Build Order

**DISPLAY-01 and DISPLAY-02 can be built as two independent phases — they share a touched file
(`calendar.html`) but not touched logic within it, and neither has a code dependency on the other.**
Recommend building them as **separate, sequential phases**, not a single combined phase, for these
reasons:

1. **No shared backend plumbing.** DISPLAY-01 is read-side only (no migration, no command change —
   `proposal` already exists on `CalendarEvent`). DISPLAY-02 is write-side-first (new model, new
   migration, one new line in `sync_lco_observation_calendar.py`) before it has anything to display.
   Their non-template code is fully disjoint.
2. **Shared file, disjoint edits.** Both phases edit `src/templates/tom_calendar/partials/
   calendar.html`, but at different, non-overlapping locations: DISPLAY-01 touches the `{{ event.color
   }}` call sites and the `{% if event.title|slice:... %}` prefix branch; DISPLAY-02 adds a new,
   independent `{{ event.telescope_label_meta.is_verified }}` read (e.g. a small icon/badge). A merge
   conflict is possible if both phases' template edits land in parallel branches, but is mechanically
   trivial to resolve (different lines/regions of the same `{% for event in ... %}` block) — this is a
   *sequencing convenience* concern, not an architectural dependency.
3. **Recommended order: DISPLAY-02 first, DISPLAY-01 second.**
   - DISPLAY-02's write-path change (`sync_lco_observation_calendar.py` + new model + migration) is
     the riskier, more novel piece (first-ever migration for `solsys_code`, first cross-app
     `OneToOneField` extension in this codebase) — sequencing it first means any schema/migration
     surprises are isolated from DISPLAY-01's purely-additive template work, and the existing
     `test_sync_lco_observation_calendar.py` suite (36 tests, already covering the
     `telescope_api_failed` computation path) is the natural place to add sidecar-row assertions
     without churn from a simultaneous template change.
   - DISPLAY-01's `calendar_display_extras.py` module (new templatetags package) is needed by both
     features in practice if the DISPLAY-02 N+1 mitigation (a bulk-prefetch tag) is taken — building
     the templatetags package in the DISPLAY-01 phase first would mean DISPLAY-02 either duplicates
     the package-creation step or has to land out of order. **If the N+1 mitigation is in scope for
     this milestone, build DISPLAY-01 first** (it creates the `templatetags/` package and module that
     DISPLAY-02's prefetch tag would then extend) **and DISPLAY-02 second.** If the N+1 mitigation is
     explicitly deferred (accept-as-is, see mitigation option 1), the order in the bullet above
     (DISPLAY-02 first) is preferable since it isolates the riskier migration work.
   - **Net recommendation for roadmap purposes: sequence by whether the N+1 mitigation is in scope.**
     Flag this as an explicit roadmap/requirements decision point rather than presupposing an answer
     here — both orderings are architecturally sound; the tie-breaker is a scope call (N+1 mitigation:
     yes/no/defer), not a technical constraint.
4. **Neither phase blocks the other's correctness.** A `calendar.html` shipped with only DISPLAY-01's
   changes renders correctly with no sidecar rows in the DB (no `CalendarEventTelescopeLabel` access
   attempted). A `calendar.html` shipped with only DISPLAY-02's changes renders correctly with
   `event.color`'s existing PK-keyed behavior still in place for color (unchanged, since DISPLAY-01
   hasn't touched that call site yet). This independence is what makes the "separate phases" framing
   safe rather than merely convenient.

## Sources

- Direct read: `/home/tlister/git/fomo_devel/solsys_code/management/commands/sync_lco_observation_calendar.py`
  (full file, 643 lines) — HIGH confidence. Confirms `telescope_api_failed` computed at line 470
  (`_build_event_fields` return dict), popped at line 604, `CalendarEvent.objects.get_or_create` at
  line 615 — the exact DISPLAY-02 write-site coordinates cited throughout this file.
- Direct read: `/home/tlister/git/fomo_devel/solsys_code/management/commands/load_telescope_runs.py`
  (full file, 126 lines) — HIGH confidence. Confirms no API call, no fallback concept, no DISPLAY-02
  integration needed.
- Direct read: `/home/tlister/git/fomo_devel/src/templates/tom_calendar/partials/calendar.html`
  (full file, 209 lines) — HIGH confidence. Confirms existing `{% load tz calendar_tags %}` precedent
  (line 101), the `[QUEUED]` prefix branch (lines 158-162), `{{ event.color }}` used only in the
  all-day branch (line 161), no color call in the timed branch (lines 170-183).
- Direct read: `/home/tlister/venv/fomo311_venv/lib64/python3.11/site-packages/tom_calendar/views.py`
  (full file) — HIGH confidence. Confirms `render_calendar()` is a plain FBV with no
  `extra_context`/`get_queryset()` hook, confirms `events = CalendarEvent.objects.filter(...)` has no
  `select_related`/`prefetch_related` (the N+1 risk basis), confirms `create_event`'s `EventForm`
  exists as a second `CalendarEvent` producer with no `telescope_api_failed` concept (Anti-Pattern 2
  basis).
- Direct read: `/home/tlister/venv/fomo311_venv/lib64/python3.11/site-packages/tom_calendar/templatetags/calendar_tags.py`
  (full file) — HIGH confidence. Confirms the `target_list_color` `simple_tag` precedent this
  research's `proposal_color` tag mirrors.
- Direct read: `/home/tlister/git/fomo_devel/solsys_code/models.py` — HIGH confidence. Confirms this
  app's `models.py` is currently all-comment (no real model), so the sidecar model is this app's first.
- `/home/tlister/git/fomo_devel/.planning/research/STACK.md` (sibling research, same milestone,
  written 2026-06-24) — HIGH confidence, built on directly throughout this file (sidecar-model choice,
  template-tag mechanism, `update_or_create` write pattern, `BOOTSTRAP_COLORS` palette caveat).
- `/home/tlister/git/fomo_devel/.planning/PROJECT.md` — HIGH confidence, project context, Key
  Decisions table, Out of Scope entry for "status-aware coloring" (the deferred predecessor of
  DISPLAY-01).
- Django documentation on `ObjectDoesNotExist.silent_variable_failure` — MEDIUM confidence (not
  re-verified by direct package source read in this session; flagged in Pattern 2 above as something
  the DISPLAY-02 build phase should exercise with an actual template-rendering test rather than rely
  on purely from documentation memory).

---
*Architecture research for: FOMO v1.4 Calendar Visual Clarity (DISPLAY-01 proposal-keyed color +
status visual treatment, DISPLAY-02 fallback-vs-verified telescope label field)*
*Researched: 2026-06-24*
