# Phase 33: Series Identity & Reconciler Inversion - Pattern Map

**Mapped:** 2026-09-03
**Files analyzed:** 13 (2 new/likely: migration 0017; unlink helper home in `campaign_utils.py`)
**Analogs found:** 13 / 13 (all in-repo; RESEARCH.md already quotes verbatim source for most)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/models.py` (`CalendarEventMeta.observation_record`, `.observation_group`) | model | CRUD | `solsys_code/models.py` `CalendarEventMeta.run` FK (same file, lines 26-42) and `CampaignRunObservation.observation_record` (lines 458-463) | exact |
| `solsys_code/migrations/0017_*.py` | migration | batch (schema) | `solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py`, `0015_campaignrun_nullable_campaign_and_source_identifier.py` | exact |
| `solsys_code/campaign_reconciler.py` (`_reconcile_classical_nights()`, deletion of `_adopted_event_for_night()`, `event_title()`, `_detach_stale_family_events()`) | service | CRUD/batch | same file — existing `_may_write()`, `_reconcile_container()` control flow | exact (self-modification) |
| `solsys_code/campaign_utils.py` (`unlink_event_from_run()`, NEW) | utility | CRUD | `solsys_code/campaign_utils.py` `adopt_event_into_run()` (its mirror-image twin) | exact |
| `solsys_code/campaign_views.py` (`_undo_confirmation()`, call site swap) | controller (Django view) | request-response | same file, existing `.update(run=None, confirmed_by=None, confirmed_at=None)` block (~line 1326-1328) | exact (self-modification) |
| `solsys_code/admin.py` (`CalendarEventMetaAdmin`, `CalendarEventMetaInline`) | config (ModelAdmin) | CRUD | same file — existing `readonly_fields`/`save_model` branch for `confirmed_by`/`confirmed_at` (~line 382-386) | exact |
| `solsys_code/templatetags/calendar_display_extras.py` (NEW decoration tags) | utility (template tag) | transform | same file — existing `proposal_color`/`status_border_css` simple_tag pattern; `attribution_display_extras.py` for a sibling library layout | exact |
| `src/templates/tom_calendar/partials/calendar.html` (month cell marker) | component (template) | request-response | same file, event loop ~lines 215-262 | exact |
| `src/templates/tom_calendar/partials/event_form.html` (modal block extension) | component (template) | request-response | same file, existing `{% with run=event.telescope_label_meta.run %}` block, lines 118-153 | exact |
| `solsys_code/views.py` (`fomo_render_calendar` prefetch chain) | controller | request-response | same file, lines 107-115 (`prefetch_related('telescope_label_meta')`) | exact — **do not import this module in tests/probes (SPICE side effect)** |
| `solsys_code/tests/test_campaign_reconciler.py` (`TestAdoptAndRekey` rewrite, new D-01 skip tests) | test | CRUD | same file — existing `TestRecordEventNonInterference` (lines 600-682) as the "leave alone" model | exact |
| `solsys_code/tests/test_calendar_template.py` (`EventModalCampaignRunLinkTest` extension) | test | request-response | same file, lines 403-472 | exact |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, `campaign_lifecycle_demo.ipynb`, `docs/runbooks/telescope_runs_calendar.rst` | doc/notebook | batch (demo) | themselves (prior versions); paired-docs rule in CLAUDE.md | exact |

## Pattern Assignments

### `solsys_code/models.py` — `CalendarEventMeta.observation_record`, `.observation_group` (model, CRUD)

**Analog:** same file, `CalendarEventMeta.run` (lines 26-42) and `CampaignRunObservation.observation_record` (lines 458-463)

**Existing `run` FK to copy the form of** (verbatim, `solsys_code/models.py:26-42`):
```python
    run = models.ForeignKey(
        'CampaignRun',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='calendar_event_metas',
        verbose_name='Owning campaign run',
    )
```
Per D-17 this field's `verbose_name` becomes `'Attributed campaign run'` in this phase — an `AlterField` alongside the two `AddField`s.

**Direct-import FK precedent** (verbatim, `solsys_code/models.py:6` + `458-463`):
```python
from tom_observations.models import ObservationRecord
...
    observation_record = models.ForeignKey(
        ObservationRecord,
        on_delete=models.CASCADE,
        related_name='campaign_run_links',
        verbose_name='Observation record',
    )
```
`ObservationGroup` is not yet imported in `models.py` — add it to the same import line.

**New fields skeleton** (per D-05/D-06/D-07, RESEARCH.md "Code Examples"):
```python
from tom_observations.models import ObservationGroup, ObservationRecord

class CalendarEventMeta(models.Model):
    ...
    observation_record = models.OneToOneField(
        ObservationRecord,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='calendar_event_meta',
        verbose_name='Observation record',
    )
    observation_group = models.ForeignKey(
        ObservationGroup,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='calendar_event_metas',
        verbose_name='Observation group',
    )
```
Note: `observation_record` is `OneToOneField` (D-05, DB-enforced one event per record); `observation_group` is plain `ForeignKey` (D-06).

---

### `solsys_code/migrations/0017_*.py` (migration, batch)

**Analog:** `solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py`, `solsys_code/migrations/0015_campaignrun_nullable_campaign_and_source_identifier.py`

Both are small, additive migrations on `CalendarEventMeta`/`CampaignRun` with a cross-app dependency declared explicitly. Two prior migrations needed a `tom_observations` dependency for FKs into that app (verbatim):
```python
# solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py:29
# solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py:22
dependencies = [
    ...
    ('tom_observations', '0016_alter_facility_options'),
]
```
**Generation approach:** run `python manage.py makemigrations solsys_code` rather than hand-writing — verify the generated `dependencies` includes a `tom_observations` entry at or after `0016_alter_facility_options` before committing (Pitfall 5). Next number is `0017`; whether the `AlterField` (verbose_name rename) and the two `AddField`s ship as one file or two is planner's discretion (CONTEXT.md D-discretion).

---

### `solsys_code/campaign_reconciler.py` — reconciler inversion (service, CRUD/batch)

**Analog:** itself — `_may_write()` (unchanged control point) and the retiring `_reconcile_classical_nights()` resolution order.

**`_may_write()` — unchanged, the ownership gate every write passes through** (verbatim, lines 223-239):
```python
def _may_write(event: CalendarEvent | None, run: CampaignRun) -> bool:
    """RECON-05's ownership rule -- the first condition checked in every write path.
    ...
    """
    if event is None:
        return True
    meta = CalendarEventMeta.objects.filter(event=event).first()
    if meta is not None and meta.run_id is not None:
        return meta.run_id == run.pk
    container_url = run_container_url(run)
    return event.url == container_url or event.url.startswith(f'{container_url}:')
```

**Docstring of the resolution order being retired** (verbatim, lines 356-362) — read this before writing the new docstring/behavior, do not paraphrase:
```python
    Per-night resolution order (D-02): (1) an existing event already keyed at
    ``run_night_url(run, night)`` -- the common idempotent-rerun case; (2) failing that,
    ``_adopted_event_for_night(...)`` -- a ``load_telescope_runs``-created event already
    attributed to this run for this night via Phase 28's confirmation queue, re-keyed in
    place; (3) failing both, mint a new event. Both (1) and (2) go through ``_may_write()``
    before any write -- the ownership rule stays the first condition checked (RECON-05
    defence in depth; see T-29-05).
```

**D-01 skip-check form** (adapted from the retiring `_adopted_event_for_night()` query, `campaign_reconciler.py:332-341`; RESEARCH.md "Code Examples"):
```python
def _night_already_attributed(run: CampaignRun, night, site_zone: ZoneInfo) -> bool:
    """D-01: True when a non-RUN: event is already attributed to this run for this
    site-local night -- the reconciler must skip minting/writing for that night."""
    candidates = (
        CalendarEventMeta.objects.filter(run_id=run.pk)
        .exclude(event__url__startswith=RUN_URL_NAMESPACE)
        .select_related('event')
    )
    return any(
        meta.event.start_time.astimezone(site_zone).date() == night for meta in candidates
    )
```
**Important:** drop `_adopted_event_for_night()`'s `event__url=''` restriction (Anti-Pattern) so this also matches a future URL-keyed Phase 34 observation event, per D-01's own stated scope, even though this phase's own fixtures only exercise the blank-url case (D-08).

**`_detach_stale_family_events()` — today's bulk `.update()`, must route through the new unlink helper and start clearing `confirmed_by`/`confirmed_at`** (verbatim, lines 470-471):
```python
stale = owned_events(run).exclude(url__in=active_urls)
CalendarEventMeta.objects.filter(event__in=stale, run=run).update(run=None)
```
This is Pitfall 2 site 2 — a genuine behavior change (today it does not clear `confirmed_by`/`confirmed_at`), not a pure refactor.

**`update_calendar_event_key_and_fields()` second caller (do not break):** `_reconcile_container()` calls it at line 285 (unrelated to D-01, must keep working); the D-01 retiring call site is at line 429. If the function is deleted (discretion item), replace only the line-285 call with the plain no-churn update call — never delete the container branch's own call.

---

### `solsys_code/campaign_utils.py` — `unlink_event_from_run()` (NEW) (utility, CRUD)

**Analog:** `solsys_code/campaign_utils.py` `adopt_event_into_run()` — its mirror-image twin (link-only attribution write already at "attributed to" semantics per CONTEXT.md's Reusable Assets note).

**Three existing "clear the link" call sites this helper must serve — different call signatures, not a drop-in rename (Pitfall 2):**

1. `campaign_views._undo_confirmation()` — conditional bulk `.update()` (verbatim, `campaign_views.py:1326-1328`):
```python
changed_count = CalendarEventMeta.objects.filter(event_id=orphan_pk, run_id=run_pk).update(
    run=None, confirmed_by=None, confirmed_at=None
)
```
Caller needs the `changed_count` return value.

2. `campaign_reconciler._detach_stale_family_events()` — bulk `.update()` over a queryset of many rows, today clearing only `run` (see above) — must gain `confirmed_by`/`confirmed_at` clearing via the shared helper.

3. `admin.py CalendarEventMetaAdmin.save_model()` branch 2 — single in-memory instance mutation before `super().save_model()` (verbatim, `admin.py:382-386`):
```python
elif obj.run_id is None and prior_run_id is not None:
    obj.confirmed_by = None
    obj.confirmed_at = None
```

**Design guidance (RESEARCH.md):** design `unlink_event_from_run()` to serve the strictest caller (site 1: take a queryset or `(event_id, run_id)` filter, return a changed count); sites 2 and 3 call it with narrower filters. Never touches `is_verified` or any `CalendarEvent` field (D-16).

---

### `solsys_code/admin.py` — `CalendarEventMetaAdmin` / inline read-only fields (config, CRUD)

**Analog:** same file, existing `readonly_fields` and `save_model` handling for `confirmed_by`/`confirmed_at` (~lines 382-386, quoted above under `campaign_utils.py`).

Add `observation_record`, `observation_group` to `readonly_fields` on both `CalendarEventMetaAdmin` and the `CampaignRunAdmin` inline (D-09) — same mechanism already used for `confirmed_by`/`confirmed_at`. `save_model` branch 2 should call the new `unlink_event_from_run()` helper instead of its inline mutation once the helper exists (D-16/D-17 label updates: "Owning" → "Attributed").

---

### `solsys_code/templatetags/calendar_display_extras.py` (NEW decoration tags) (utility/template tag, transform)

**Analog:** existing `simple_tag` functions in the same file (`proposal_color`, `status_border_css`) and the sibling library `attribution_display_extras.py` (`high_band_attribution_candidates`) for the modal-hint tag form.

**Pattern to copy:** `{% register.simple_tag %}` functions reading `event.telescope_label_meta.run` at render time, never writing anything. Two tags needed: a compact cell-marker (chip/icon, campaign name as tooltip) and a modal-block helper/extension. **Guard required (Pitfall 1, not in CONTEXT.md but required by RESEARCH.md):** any tag emitting the `campaigns:table` link must check `run.campaign_id is not None` before building the `{% url %}` call — `CampaignRun.campaign` has been nullable since migration `0015`, and `{% url 'campaigns:table' run.campaign_id %}` raises `NoReverseMatch` (hard template error) for `campaign_id=None`, not a silent gap. `is_publicly_visible` alone (verbatim, `solsys_code/models.py:293-304`: `return self.approval_status != self.ApprovalStatus.PENDING_REVIEW`) does not check campaign nullness.

---

### `src/templates/tom_calendar/partials/event_form.html` (modal decoration) (component, request-response)

**Analog:** itself — existing "Campaign run" block, extend don't rebuild.

**Existing block to extend** (verbatim, lines 118-153):
```html
{% with run=event.telescope_label_meta.run %}
{% if run.is_publicly_visible %}
    <div class="row">
      <div class="col">
        <label>
          Campaign run
          <small>
            <a href="{% url 'campaigns:table' run.campaign_id %}" target="_blank">View campaign ↗</a>
          </small>
        </label>
        <div>
          {{ run.telescope_instrument }}
          {% if run.window_start %}
          ({{ run.window_start }}&ndash;{{ run.window_end }})
          {% endif %}
          &mdash; {{ run.get_run_status_display }}
        </div>
      </div>
    </div>
{% elif not run and request.user.is_staff %}
    ...
{% endif %}
{% endwith %}
```
**Changes required:** (1) D-13's `#run-{pk}` anchor: `{% url 'campaigns:table' run.campaign_id %}#run-{{ run.pk }}`; (2) the campaign-nullness guard added to the `{% if %}`: `{% if run.is_publicly_visible and run.campaign_id %}`; (3) D-17 label rename "Campaign run" → "Attributed campaign run" (or similar "attributed to" wording).

---

### `src/templates/tom_calendar/partials/calendar.html` (month cell marker) (component, request-response)

**Analog:** itself, existing event loop (~lines 215-262) where the proposal-colour legend, status rings, and `is_verified == False` dashed border are already rendered per-event — the cell marker is a new sibling decoration in the same loop, must not consume the truncated title text (16/18-char budget per D-10).

---

### `solsys_code/views.py` — `fomo_render_calendar` prefetch chain (controller, request-response)

**Analog:** same file, existing DISPLAY-09 prefetch (verbatim, lines 107-115):
```python
    # DISPLAY-09: prefetch telescope_label_meta to eliminate OneToOneField N+1;
    # annotate active_todo_count to eliminate active_todos.count() N+1.
    events = (
        CalendarEvent.objects.filter(
            start_time__date__lte=weeks[-1][-1],
            end_time__date__gte=weeks[0][0],
        )
        .prefetch_related('telescope_label_meta')
        .annotate(active_todo_count=Count('todos', filter=Q(todos__is_completed=False)))
    )
```
**Change required:** extend to `Prefetch('telescope_label_meta', queryset=CalendarEventMeta.objects.select_related('run__campaign'))` (or equivalent) so the cell marker's `run.campaign.name` lookup is N+1-free.

**CRITICAL constraint (CLAUDE.md + phase context):** `solsys_code/views.py` triggers the ~1.6 GB SPICE kernel download on import. This edit is required (it's the only home of `fomo_render_calendar`), but no test, probe, or notebook cell written for this phase should `import solsys_code.views` or `solsys_code.ephem_utils` directly — the existing test suite already routes around this via Django's HTTP test client (`test_calendar_template.py`), not direct imports. Follow that pattern for any new probes.

---

### `solsys_code/tests/test_campaign_reconciler.py` (test, CRUD)

**Analog:** same file — `TestRecordEventNonInterference` (lines 600-682) as the "must NOT be over-rewritten" model, `TestAdoptAndRekey` (lines 381-476) as the class that MUST be rewritten.

**`TestAdoptAndRekey` — currently asserts the exact behavior D-01 retires** (verbatim assertion, lines 404-430):
```python
self.assertEqual(adopted_event.url, f'RUN:{run.pk}:{first_night.isoformat()}')
```
Fixture pattern: `CalendarEventMeta.objects.create(event=adopted_event, run=run)` on a blank-url event, then reconcile. Under D-01 rewrite each of the 3 tests to assert: `url` stays unchanged (never re-keyed); `CalendarEvent.objects.count()` for the run's window is 1 (attributed night) + 1 per un-attributed night; `ReconcileResult` reports the attributed night as skipped, not `updated`. Keep the site-local-night matching sub-test's comparison logic (still needed for the skip check).

**`TestRecordEventNonInterference` — likely unaffected, do not over-rewrite** (verbatim fixture detail, line 648): links via `CampaignRunObservation.objects.create(run=run, observation_record=record)` only, **no** `CalendarEventMeta` row created for that event — D-01's skip rule keys off `CalendarEventMeta.run` (D-14: no fallback to `CampaignRunObservation`), so this fixture's record-event stays invisible to the skip check and existing assertions (`CalendarEvent.objects.count() == 1 + n_nights`) should remain correct. Add a **new** test (or extend this fixture) that additionally creates `CalendarEventMeta(event=record_event, run=run)` to prove the skip rule fires for a URL-keyed attributed event — the "Phase 34 observation event later" case D-01 names but no existing test exercises (D-08 defers URL-keyed linking).

**Target fixtures:** any new fixture touching `tom_targets.Target` must use `tom_targets.tests.factories.NonSiderealTargetFactory` (CLAUDE.md convention), never `SiderealTargetFactory`.

---

### `solsys_code/tests/test_calendar_template.py` — `EventModalCampaignRunLinkTest` extension (test, request-response)

**Analog:** same file, lines 403-472 — every existing fixture there sets `campaign=cls.campaign`, so no current test exercises Pitfall 1's `campaign=None` guard. Add a dedicated fixture/test with a `CampaignRun` whose `campaign` is `None` and assert the modal renders without `NoReverseMatch` (no link emitted) rather than crashing.

---

## Shared Patterns

### Ownership/attribution gate — do not rebuild
**Source:** `solsys_code/campaign_reconciler.py:223-239` (`_may_write()`)
**Apply to:** all reconciler write paths — unchanged by this phase (D-02); only what gets checked against it changes (the adopt candidate disappears from `_reconcile_classical_nights()`).

### No-churn create/update contract
**Source:** `solsys_code/calendar_utils.py` `insert_or_create_calendar_event()` / `_update_or_unchanged()`
**Apply to:** every write path in `campaign_reconciler.py` — D-01's change means the classical-night loop calls this *less often* (skip = no call), not differently.

### Link-only attribution write (the write `unlink_event_from_run()` inverts)
**Source:** `solsys_code/campaign_utils.py` `adopt_event_into_run()` — "set `meta.run`, touch nothing else, refuse if attributed elsewhere."
**Apply to:** `unlink_event_from_run()` design, and confirms `_link_event_to_run()` in the reconciler is its twin (self-attribution on every `RUN:` event created/updated).

### Template-tag-driven display decoration
**Source:** `solsys_code/templatetags/calendar_display_extras.py` (`proposal_color`, `status_border_css`), `attribution_display_extras.py` (`high_band_attribution_candidates`)
**Apply to:** the new cell-marker and modal-extension tags — same `simple_tag`, read-only-at-render form; never a `.save()` call inside a template tag (Anti-Pattern).

### Migration cross-app dependency
**Source:** `solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py:29`, `0013_attribution_dismissals_and_calendar_event_meta_audit.py:22` — both declare `('tom_observations', '0016_alter_facility_options')`
**Apply to:** migration `0017_*.py` — generate via `python manage.py makemigrations solsys_code`, verify the `tom_observations` dependency is present rather than hand-writing it.

## No Analog Found

None — every file in this phase's scope has a direct, same-file or same-module precedent already quoted verbatim in RESEARCH.md; no cross-codebase search for an unrelated analog was needed.

## Metadata

**Analog search scope:** `solsys_code/` (models, campaign_reconciler, campaign_utils, campaign_views, admin, templatetags, tests), `src/templates/tom_calendar/partials/`, `solsys_code/migrations/`, `docs/notebooks/pre_executed/`, `docs/runbooks/`
**Files scanned:** 13 target files + `solsys_code/migrations/0010-0016` for dependency precedent
**Tracked-source verification:** all 13 target paths confirmed via `git ls-files` (no `.gsd/` mirror paths involved; migration `0017` does not yet exist, next in sequence after `0016_alter_campaignrun_source_soar_queue.py`)
**Pattern extraction date:** 2026-09-03
