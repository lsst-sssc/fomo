# Stack Research

**Domain:** Django calendar visual-treatment + cross-app model extension (FOMO v1.4 — DISPLAY-01/02)
**Researched:** 2026-06-24
**Confidence:** HIGH

This is a narrow, additive milestone on an already-validated stack (Django 5.2.15, `tomtoolkit==3.0.0a9`,
`tom_calendar` bundled inside it). No new runtime dependency, package, or app is needed for either
DISPLAY-01 or DISPLAY-02 — both are solved with stdlib (`hashlib`) plus Django patterns already used
elsewhere in this exact codebase (template tag library, sidecar model via existing `solsys_code` app).

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `hashlib` (stdlib) | Python 3.10+ (project floor) | Deterministic string→int hash for DISPLAY-01's proposal-to-color mapping | `hashlib.sha256(proposal.encode()).hexdigest()` is stable across process restarts and across machines — required because the same proposal string must map to the same color on every calendar render, including after a server/worker restart. Verified empirically below: Python's built-in `hash()` is salted per-process and is NOT usable here (see Pitfall). |
| Django template tag library (`django.template.Library`, `@register.simple_tag`) | Django 5.2.15 (installed; confirmed via `pip show`) | Compute and expose the proposal color (and DISPLAY-02's verified/fallback flag) inside `src/templates/tom_calendar/partials/calendar.html` | This is the exact mechanism `tom_calendar` itself already uses for an equivalent need: `tom_calendar/templatetags/calendar_tags.py` registers `target_list_color` as a `simple_tag` calling `tom_calendar.utils.target_list_color()`. Mirroring this pattern in a project-owned tag library is the most idiomatic, lowest-friction option — no new abstraction introduced, consistent with code already shipped in the very file being overridden. |
| Django `OneToOneField` sidecar model (`solsys_code/models.py`) | Django 5.2.15 | DISPLAY-02's structured `is_verified` (or equivalent) flag attached to third-party `CalendarEvent` rows | `tom_calendar.models.CalendarEvent` hardcodes `class Meta: app_label = 'tom_calendar'` with no `abstract = True` and ships its own migrations inside the installed `tomtoolkit` package — it cannot be subclassed via Django's normal "extend an abstract base" pattern, and a multi-table-inheritance subclass is blocked by the same `app_label`/migration-ownership problem. A `OneToOneField(CalendarEvent, on_delete=models.CASCADE, primary_key=True)` sidecar model living in the already-installed `solsys_code` app is the standard, documented Django pattern for attaching new fields to a model you don't own (confirmed current for Django 5.2/6.0 — see Sources). |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `zlib.crc32` (stdlib) | Python 3.10+ | Alternative fast deterministic hash, also process-stable | Only if `hashlib.sha256` is judged "too heavy" for a per-render, per-event computation (it isn't — see Stack Patterns below); `crc32` is a 1-line drop-in alternative with the same determinism guarantee, no security properties needed since this isn't a security context. |
| None — no new third-party color library | n/a | n/a | Do not add `colorhash`, `randomcolor`, or similar PyPI packages. The target palette is a fixed 9-entry Bootstrap4 list already defined in `tom_calendar.utils.BOOTSTRAP_COLORS` (or this project's own literal-hex mirror per the migration-safety constraint below) — picking 1-of-9 deterministically needs nothing beyond `int(hash, 16) % 9`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `ruff check . --fix` / `ruff format .` | Lint/format the new template-tag module and sidecar model | Same gate already enforced project-wide; no new config needed. |
| `./manage.py makemigrations solsys_code` / `./manage.py migrate` | Generate/apply the migration for the new sidecar model | Runs against `solsys_code`'s existing (currently `__init__.py`-only) migrations folder — no new app registration, no `INSTALLED_APPS` change. |
| `./manage.py test solsys_code` | Test the sidecar model and its sync-into hooks | Same Django test runner already used for `test_sync_lco_observation_calendar.py`/`test_load_telescope_runs.py`; DB-dependent, consistent with the two-suite split in CLAUDE.md. |

## Installation

```bash
# No new packages required for either DISPLAY-01 or DISPLAY-02.
# hashlib, zlib are stdlib. OneToOneField, template tags are Django (already installed: 5.2.15).

# After adding the sidecar model to solsys_code/models.py:
./manage.py makemigrations solsys_code
./manage.py migrate
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| `hashlib.sha256(s.encode()).hexdigest()` then `int(..., 16) % 9` | `zlib.crc32(s.encode()) % 9` | If profiling ever shows the SHA-256 computation is a measurable cost at calendar-render scale (it will not be — a few dozen events per month view). `crc32` is marginally faster and equally deterministic; switching is a one-line change, not an architecture decision. |
| `hashlib.sha256` | Python's built-in `hash()` | Never, for this use case — see Pitfall below. `hash()` is appropriate only for same-process, ephemeral lookups (e.g. dict/set keys within one Django request), never for a value that must be stable across requests/restarts. |
| `OneToOneField` sidecar model in `solsys_code` | Django `Meta.proxy = True` proxy model | Proxy models change *behavior* (custom manager, different `Meta.ordering`, alternate `__str__`) on the *same* underlying table — they cannot add new *columns*, because a proxy model has no migration of its own (it shares the parent's table exactly). DISPLAY-02 needs a new persisted field, so a proxy model is structurally incapable of solving it. Don't use it here. |
| `OneToOneField` sidecar model | Multi-table inheritance (`class TelescopeLabelMeta(CalendarEvent): ...`) | Would work mechanically (Django supports MTI against any concrete model) but requires Django to manage a migration that creates a child table for `CalendarEvent`, a model whose `Meta.app_label` and migration history are *owned by the installed `tomtoolkit` package*, not this project. Mixing migration ownership across an installed third-party package and a project app this way is fragile (e.g. breaks if `tomtoolkit` ever ships its own subclass or adds a field with a colliding name) and is not how Django's docs frame "extend a model you don't own." Stick with `OneToOneField`, which only ever touches *this project's* migration history. |
| `OneToOneField` sidecar model | Django Signals (`post_save` on `CalendarEvent`) auto-creating the sidecar row | Signals are a reasonable *supplementary* mechanism (e.g. for code paths outside the two management commands that might create a `CalendarEvent`, such as the upstream `EventForm`/`create_event` view), but should not be the *primary* write path for the two commands this milestone cares about — see Stack Patterns below for why an explicit `update_or_create` call colocated with the existing `get_or_create`/conditional-`save()` block is preferable for the two known producers. |
| Literal hex/rgba values for the palette | `tom_calendar.utils.BOOTSTRAP_COLORS` (`var(--red)` etc.) imported directly | Only safe to import directly while pinned to `tomtoolkit==3.0.0a9`. The PROJECT.md constraint already flags that `3.0.0a10` renames these to `var(--bs-red)` (Bootstrap5 migration). Define a small literal-hex/rgba palette of 9 colors locally in the new template-tag module (or copy-pin the *values*, not the *names*) so DISPLAY-01 survives that future upgrade unchanged. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| Python's built-in `hash(proposal_string)` | **Confirmed pitfall, empirically verified during this research**: `hash()` on `str` is salted with a per-process random seed (`PYTHONHASHSEED`, randomized by default since Python 3.3 as a hash-DoS mitigation). Three separate subprocess invocations of `hash("LTP2025A-004")` returned three different large negative/positive integers in this exact verification run. Using `hash()` would make a proposal's calendar color change every time the Django dev server or a production worker restarts — directly violating "same proposal string always maps to the same color." | `hashlib.sha256(s.encode()).hexdigest()` (or `zlib.crc32`), both confirmed stable across repeated invocations in the same verification run. |
| A new Django app (e.g. `calendar_display`) just to hold the sidecar model or template tags | Both DISPLAY-01 and DISPLAY-02 are small, single-purpose additions that belong naturally alongside the two management commands already living in `solsys_code` (which is already in `INSTALLED_APPS`, already has a `migrations/` folder, already has `models.py`). Adding a new app means a new `INSTALLED_APPS` entry, a new `AppConfig`, and a new migrations history to maintain for what is functionally one model and one template-tag module. | Add the sidecar model to `solsys_code/models.py` and the template tag to a new module under `src/templatetags/` (mirroring `fomo_extras.py`/`solsys_code_extras.py`'s existing placement) or to `solsys_code`'s own `templatetags/` package if one is added — either is consistent with existing project layout. |
| Editing `tom_calendar`'s installed package files directly (`models.py`, `views.py`, `utils.py`, any bundled template) | These live under the venv's `site-packages/tomtoolkit` install and are explicitly called out as not-a-project-app in the milestone context. Any direct edit is silently lost on the next `pip install`/upgrade and is invisible to this repo's version control. | The project-level template override at `src/templates/tom_calendar/partials/calendar.html` (already exists and already demonstrates this pattern for the `[QUEUED]` de-emphasis treatment) plus a project-owned template tag library for computed values (color, verified-flag) that `event.color`/`event.<sidecar>.is_verified` alone cannot express in-template. |
| A new PyPI color-hashing library (`colorhash`, `randomcolor`, etc.) | Unnecessary dependency for a 9-bucket deterministic mapping that `hashlib` + `%` solves in two lines. Adding a dependency here has no payoff and is explicitly against this milestone's "what NOT to add" guidance. | `hashlib.sha256(...).hexdigest()` → `int(..., 16) % len(PALETTE)`. |
| Importing `BOOTSTRAP_COLORS` from `tom_calendar.utils` by reference for the new palette | Ties DISPLAY-01's correctness to the upstream `var(--red)`→`var(--bs-red)` rename landing in `tomtoolkit==3.0.0a10`+ (Bootstrap5 migration); this project is pinned below that. | Define a small project-local literal-hex (or `rgba()`) 9-color palette, independent of upstream's CSS-variable-name choices. |
| Relying on `event.color` (the existing `CalendarEvent.color` Python property) for DISPLAY-01 | It is purely `BOOTSTRAP_COLORS[self.pk % 9]` — PK-keyed, not proposal-keyed, and not overridable per-instance without monkeypatching the installed class (out of bounds per this milestone). The property is also only referenced by the *all-day* event branch of the upstream/overridden template (`day.all_day_events` → `cal-event-all-day` block with `style="background-color: {{ event.color }};"`); the *timed* event branch (`day.events` → `cal-event-timed`) does not call `event.color` at all today. | A template tag (e.g. `{% proposal_color event.proposal %}`) called explicitly in the template, replacing the `{{ event.color }}` reference — and added to the timed-event branch too, since DISPLAY-01's deterministic-by-proposal coloring should presumably apply to both event display modes (confirm scope with the user/roadmap, since today only one branch is colored at all). |

## Stack Patterns by Variant

**For DISPLAY-01 (proposal-keyed color + status treatment):**
- Use a project template tag library, e.g. `solsys_code/templatetags/calendar_display_extras.py` (new file, `solsys_code` app already exists so this just needs an `__init__.py` + the module — no settings change required since Django auto-discovers `templatetags/` packages of any `INSTALLED_APPS` app), registering:
  - `proposal_color(proposal: str) -> str` — `@register.simple_tag`, hashes `proposal` via `hashlib.sha256` and indexes into a literal 9-entry hex/rgba palette.
  - A second tag or filter for the status visual treatment (opacity/border/striping), driven by parsing the same title-prefix vocabulary the two management commands already write (`[QUEUED]`, `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`, `[UNVERIFIED]`) — this can be pure string-prefix logic in the tag, no new model field needed, since the signal already exists in `title`.
- Load it in `src/templates/tom_calendar/partials/calendar.html` (`{% load tz calendar_tags calendar_display_extras %}`) and call it in place of `{{ event.color }}` in both the all-day and timed-event branches.
- Because the empty proposal string (`CalendarEvent.proposal` defaults to `''`, not null — confirmed in the installed model) is itself a valid hashable value, classically-scheduled events from `load_telescope_runs` (which never sets `proposal`) will deterministically all land on one shared palette slot. Decide during roadmap/requirements whether that's acceptable (likely yes — they're a different, already-visually-distinct category) or whether they need an explicit "no proposal" palette entry.

**For DISPLAY-02 (verified vs. fallback telescope label field):**
- Add a sidecar model to `solsys_code/models.py`:
  ```python
  class CalendarEventTelescopeLabel(models.Model):
      event = models.OneToOneField(
          CalendarEvent, on_delete=models.CASCADE, primary_key=True, related_name='telescope_label_meta'
      )
      is_verified = models.BooleanField(default=True)
  ```
  Using `primary_key=True` on the `OneToOneField` (rather than a separate auto `id` plus a `unique=True` FK) is the standard "true 1:1 extension" idiom — it guarantees at most one sidecar row per `CalendarEvent`, mirrors the relationship Django's own docs use for the canonical one-to-one example, and means deleting the `CalendarEvent` row (e.g. a future cleanup command) cascades the sidecar away with no orphan risk.
- **Integration with `sync_lco_observation_calendar.py`:** the command already computes a `telescope_api_failed` boolean per record (the exact DISPLAY-02 signal — "fallback-resolved" already exists as a local variable, it just isn't persisted as a queryable field today) right before the existing `event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)` call (`solsys_code/management/commands/sync_lco_observation_calendar.py:604-615`). Add one line immediately after that block:
  ```python
  CalendarEventTelescopeLabel.objects.update_or_create(
      event=event, defaults={'is_verified': not telescope_api_failed}
  )
  ```
  `update_or_create` (not `get_or_create`) is correct here because a re-run with a *changed* resolution outcome (e.g. the API call failed last time, succeeds this time) must flip `is_verified` on the existing sidecar row, not silently keep the first-ever value — same no-stale-data principle the command already applies to `CalendarEvent`'s own fields via its existing diff-then-`save()` block.
- **Integration with `load_telescope_runs.py`:** classical-schedule runs never go through telescope-label *resolution* at all (no LCO API call, no fallback concept) — they are populated by direct text parsing in Stage 2, with `parsed.telescope` already a 1:1 trusted token. Either (a) skip creating a sidecar row entirely for these events (then `event.telescope_label_meta` raises `CalendarEventTelescopeLabel.DoesNotExist` — `OneToOneField`'s standard "the meta object doesn't exist" behavior, confirmed in Django's official docs), and the template tag treats "no sidecar row" as "verified" (the safe default, since these are deterministically-known, not fallback-guessed); or (b) explicitly create one row with `is_verified=True` for symmetry/queryability. Recommend (a) — fewer writes, and `OneToOneField`'s missing-related-object behavior is exactly the right semantic ("this event was never subject to fallback resolution at all").
- **Template/display access:** `{{ event.telescope_label_meta.is_verified|default:True }}` works directly in the overridden `calendar.html` without a template tag, since `OneToOneField`'s reverse accessor is a normal attribute lookup — no extra Python glue layer needed for *reading* DISPLAY-02's value (unlike DISPLAY-01's color, which needs real computation and therefore a tag).

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|------------------|-------|
| `django>=5.2,<5.3` (installed: 5.2.15, pulled in transitively by `tomtoolkit==3.0.0a9`) | `OneToOneField(primary_key=True)`, `update_or_create()`, `@register.simple_tag` | All three APIs confirmed current and unchanged for Django 5.2/6.0 per official docs (see Sources) — no deprecation, no signature change relevant to this milestone. |
| `tomtoolkit==3.0.0a9` (pinned, per `pyproject.toml`: `"tomtoolkit>=3.0.0a9"` opting into the 3.0 alpha) | This project's `tom_calendar.models.CalendarEvent` shape | Confirmed installed `CalendarEvent` has exactly: `title`, `description`, `start_time`, `end_time`, `url`, `target_list`, `user`, `proposal`, `telescope`, `instrument`, `created`, `modified`, plus the `color` property and `active_todos` property. `Meta.app_label = 'tom_calendar'` is hardcoded, no `abstract`. |
| `tomtoolkit==3.0.0a9` → future `3.0.0a10`+ | `BOOTSTRAP_COLORS` palette naming | Upstream renames `var(--red)` → `var(--bs-red)` per the existing PROJECT.md constraint. **This research's recommendation (literal hex/rgba palette, not an import of `BOOTSTRAP_COLORS`) is specifically chosen to be unaffected by that future rename.** Re-verify this note if/when the project actually upgrades past `3.0.0a9`. |
| `tom_calendar`'s `render_calendar()` view function | Custom per-request context injection | Confirmed (read installed `tom_calendar/views.py`) this is a plain function-based view with a fixed context dict and no `extra_context`/class-based-view subclass hook — the only stable customization seam for adding computed values to the calendar page remains the template override, consistent with the milestone context's existing finding. Do not look for a Python-level hook; there isn't one. |

## Sources

- Installed package inspection (`/home/tlister/venv/fomo311_venv/lib64/python3.11/site-packages/tom_calendar/{models,views,utils,templatetags/calendar_tags}.py`) — HIGH confidence, read directly, matches `tomtoolkit==3.0.0a9` exactly as pinned in this project's `pyproject.toml`.
- This project's existing override (`/home/tlister/git/fomo_devel/src/templates/tom_calendar/partials/calendar.html`) — HIGH confidence, demonstrates the established title-prefix-branching pattern already shipped for `[QUEUED]` de-emphasis.
- This project's existing management commands (`solsys_code/management/commands/sync_lco_observation_calendar.py:570-626`, `solsys_code/management/commands/load_telescope_runs.py:60-116`) — HIGH confidence, read directly, identifies exact `get_or_create`/diff/`save()` integration point for the DISPLAY-02 sidecar write.
- Empirical verification in this session: `hash()` non-determinism across 3 subprocess invocations vs. `hashlib.sha256`/`hashlib.md5`/`zlib.crc32` determinism — HIGH confidence, reproduced directly, not from training memory.
- [Django Model field reference (5.2)](https://docs.djangoproject.com/en/5.2/ref/models/fields/) — `OneToOneField` semantics, confirmed current.
- [Django one-to-one relationships example docs](https://docs.djangoproject.com/en/6.0/topics/db/examples/one_to_one/) — confirms reverse-accessor `DoesNotExist` behavior referenced above.
- [Django QuerySet API reference](https://docs.djangoproject.com/en/6.0/ref/models/querysets/) — `update_or_create()` signature and semantics, confirmed current.
- GitHub `TOMToolkit/tom_base` commit history for `tom_calendar/models.py` — MEDIUM confidence (web-fetched commit-message summary, not a diff read); confirms `tom_calendar` is an actively maintained, evolving module (instrument/proposal/telescope/user fields and the explicit-`app_label` change are all relatively recent), reinforcing that this project should not assume future tomtoolkit releases preserve today's exact model shape — re-verify `CalendarEvent`'s fields on any future tomtoolkit upgrade.

---
*Stack research for: FOMO v1.4 Calendar Visual Clarity (DISPLAY-01 proposal-keyed color, DISPLAY-02 verified/fallback telescope label field)*
*Researched: 2026-06-24*
