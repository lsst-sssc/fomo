# Phase 33: Series Identity & Reconciler Inversion - Research

**Researched:** 2026-09-03
**Domain:** Django ORM schema evolution (new FKs on an existing companion model) +
inversion of an existing write-owning batch function into an annotate-only one +
display-time decoration via template tags. No new external dependencies.
**Confidence:** HIGH — every claim below is grounded in a file this session opened and
quoted, or a tool probe run in this session; no web research was needed because the
domain is entirely in-repo (Django model/admin/template conventions already established
by Phases 27-32).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Reconciler write paths (ANNOT-01)
- **D-01: Skip the night, never adopt.** For a classical night that already has a
  non-`RUN:` event attributed to this run (a `load_telescope_runs` event confirmed via the
  attribution queue today; a Phase 34 observation event later), `_reconcile_classical_nights()`
  writes nothing for that night — the attributed event *is* the night, decorated via the
  link. `_adopted_event_for_night()` and the re-key through
  `update_calendar_event_key_and_fields()` are retired; no duplicate night is minted
  alongside. This is the same rule Phase 35's allocation handoff will use ("a night with a
  linked observation has no allocation event"). — **Reversibility:** costly — reinstating
  adopt means re-introducing a write path outside `RUN:` that Phases 34–35 are built on
  never existing.
- **D-02: `_may_write()` keeps blocking on a foreign attribution.** A `RUN:{pk}` event whose
  companion row points at a *different* run (staff re-attributed it via Phase 28) stays
  blocked and is reported as `blocked`, as today. Namespace is ownership, but a human
  attribution elsewhere still outranks an automated writer (T-29-19's reasoning stands). The
  reconciler never resets `meta.run` to itself on such a row.
- **D-03: Past adopts stay in `RUN:`.** Events already re-keyed into `RUN:{pk}:{date}` by
  earlier sweeps (among the 74 `RUN:` events in the dev DB) are reconciler-owned by key
  and are left alone; no data migration un-keys them. Phase 35's allocation cutover converts
  every `RUN:{pk}:{date}` event in one place.
- **D-04: Proof of criterion 2 is a fixture test plus a real-DB diff.** A unit test with
  attributed blank-url and URL-keyed fixture events asserts `url` / `title` /
  `description` / `meta.run` are byte-identical after `reconcile_run()`; and
  `reconcile_campaign_runs_demo.ipynb` snapshots every non-`RUN:` event's
  `(url, title, meta.run)` before and after a full `reconcile_campaign_runs` sweep over the
  dev DB and shows the diff is empty.
- The reconciler still self-attributes its own events (`_link_event_to_run()` on every
  `RUN:` event it creates/updates) so the decoration path covers them, and
  `_detach_stale_family_events()` stays as-is — it already filters to `owned_events(run)`,
  i.e. the `RUN:` namespace.

#### Series-identity fields (PROJ-04)
- **D-05: `observation_record` is a nullable `OneToOneField`** to
  `tom_observations.ObservationRecord` — DB-enforced one event per record (layering note
  D1; the URL-keyed contract Phase 34 inherits from the LCO sync). NULLs do not collide, so
  the 85 existing companion rows migrate untouched. A duplicate projection becomes an
  `IntegrityError` rather than a silent second event. — **Reversibility:** costly — relaxing
  to a plain FK later is a migration plus a re-think of the one-event-per-record contract
  every Phase 34 writer assumes.
- **D-06: `observation_group` is a nullable `ForeignKey`** to `tom_observations.ObservationGroup`
  — a denormalised copy of series identity (TOM's group↔record relation is a M2M, so the
  projector chooses the group it records; how it picks for a record in several groups is a
  Phase 34 decision, not this phase's).
- **D-07: `on_delete=SET_NULL` on both**, mirroring `run`. Deleting a record or group clears
  the link and leaves the companion row (and its `is_verified` / `run` / audit history) and
  the calendar event in place; the event's own lifecycle on record deletion belongs to
  Phase 34's projector.
- **D-08: No backfill in this phase.** Phase 33 is schema + semantics only. The 156 existing
  URL-keyed LCO sync events are linked by Phase 34's projector/sweep on its first pass
  (same key namespace) — one writer per source. Tests and notebooks link a fixture event by
  hand to demonstrate decoration.
- **D-09: Read-only in the admin.** Both new fields appear on `CalendarEventMetaAdmin` and
  the `CampaignRunAdmin` inline as read-only; only code writes them. Staff keep hand-editing
  `run` exactly as today.

#### Decoration at display time (ANNOT-02)
- **D-10: Month-cell marker + modal block.** The month cell gets a compact marker (a small
  campaign chip/icon with the campaign name as tooltip) on every event whose
  `telescope_label_meta.run` is set and publicly visible, leaving the 16/18-char title
  budget to the event's own title. The modal's existing "Campaign run" block is kept and
  extended with the link (D-13). Both are template-tag driven from the link — nothing is
  written to `CalendarEvent` fields.
- **D-11: Decoration carries campaign name + run status.** Cell: campaign name. Modal:
  campaign name, the run's telescope/instrument + window, and `get_run_status_display`
  (what the block shows today, now on every linked event). Status *styling* (rings) for
  attributed events stays Phase 37's; contact fields and `source` are never rendered
  (existing PII/staff-only gates).
- **D-12: `event_title()` stops embedding the campaign name.** The reconciler's own
  `RUN:` titles drop the `"{campaign.name}: "` prefix so the decoration tag is the single
  campaign label for every linked event, `RUN:` or not. `RUN_STATUS_CALENDAR_PREFIX`
  (`[CANCELLED]` / `[WEATHERED]`) stays in titles because `status_border_css` matches on it.
  One-time title churn on the next sweep (74 events) is accepted; the notebook/runbook
  updates that shows are already in scope. — **Reversibility:** reversible — a one-line
  change in `event_title()` plus a sweep.
- **D-13: Link back to the run = campaign table, run row anchored.** The decoration's link
  targets `campaigns:table` for the run's campaign with a `#run-{pk}` anchor and the row
  highlighted. No new view (a run-detail page is Phase 37's surface).

#### Unlink & orphan surfaces (criterion 4)
- **D-14: `CalendarEventMeta.run` is the single source of attribution** for every event,
  observation-backed or not. Decoration reads only that field (prefetchable, one lookup).
  Phase 35's allocation projector keeps it in step with `CampaignRunObservation` (link →
  set, unlink → clear); the attribution queue and admin may also set it directly. No
  render-time fallback to the record's `CampaignRunObservation`.
- **D-15: Observation-backed events stay in the event-level attribution queue.**
  `orphan_calendar_events()` keeps offering every event with no run link, including those
  with `observation_record` set; Phase 28's surfaces are unchanged.
- **D-16: One shared unlink helper.** A single `unlink_event_from_run()` (name at planner's
  discretion; home in `campaign_utils.py` or the reconciler module, never
  `campaign_views.py`) clears `run`, `confirmed_by` and `confirmed_at` together and never
  touches `is_verified` or any `CalendarEvent` field. It is used by Phase 28's
  `_undo_confirmation`, the reconciler's detach step, and the admin save path, so a cleared
  link never leaves a stale "confirmed by X" behind. Unlinking never deletes an event.
- **D-17: "Owning" becomes "Attributed" everywhere.** The FK's `verbose_name` becomes
  "Attributed campaign run" (trivial `AlterField` migration), and the admin labels, the
  modal label, module/model docstrings, and the runbook's "Why doesn't the calendar pop-up
  show a 'Campaign run' block?" section all say "attributed to".

### Claude's Discretion
- Exact names/homes of the unlink helper and the decoration template tag(s) (a new tag in
  `solsys_code/templatetags/calendar_display_extras.py` or a sibling library is the
  natural fit; `attribution_display_extras.py` already exists for the modal hint).
- The visual form of the cell marker (chip vs icon vs coloured dot), provided it does not
  consume the truncated title text and coexists with the proposal-colour legend, status
  rings, and the `is_verified == False` dashed border.
- Whether `update_calendar_event_key_and_fields()` in `calendar_utils.py` is deleted
  outright (the container branch is its only other caller and never changes the url) or
  kept for same-url updates.
- Whether `ReconcileResult` gains a `skipped_nights` counter for D-01's skip rule, and how
  `--dry-run` reports it (D-05 of 29-CONTEXT fixes the created/updated/unchanged/skipped
  summary form; a supplementary count is optional).
- `related_name`s for the two new fields and the `__str__` of `CalendarEventMeta`.
- Whether the `verbose_name` rename and the two `AddField`s ship as one migration
  (`0017_…`) or two.

### Deferred Ideas (OUT OF SCOPE)
Reviewed todos not folded into this phase (each already routed elsewhere by its own
routing note):
- `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` — routed to
  Phase 35 (the allocation projector replaces `_reconcile_classical_nights()`).
- `2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` — resolved
  by v2.4's routing to Phase 34.
- `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` and
  `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` —
  attribution-UI items unrelated to link fields or reconciler writes.
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — style
  cleanup unrelated to this phase.

**Out of scope for this phase (from the phase description):** the projector, trigger and
sweep that *write* the new fields (Phase 34); backfilling `observation_record` for the 156
existing URL-keyed sync events (Phase 34's sweep does it); the allocation layer and the
`RUN:{pk}:{date}` → `ALLOC:` cutover (Phase 35); status vocabulary / status rings for
attributed events (Phase 37); a run-detail view (Phase 37's TALLY-01 surface); any
automatic `run_status` derivation.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PROJ-04 | Series identity for a record in an `ObservationGroup` is carried by real foreign keys on `CalendarEventMeta` (`observation_record`, `observation_group`) — a shared title stem and a link back to the group; spike 002's title-suffix stopgap is not the carrier | See "Standard Stack" (field definitions verified against `tom_observations.models`), "Code Examples" (migration skeleton), "Common Pitfalls" (related_name, `on_delete=SET_NULL`, migration dependency) |
| ANNOT-01 | `CalendarEventMeta.run` means "attributed to", not "owned by"; `reconcile_run()` no longer adopts, re-keys, or detaches an event attributed to a run — it only annotates — so the base layer and the campaign layer can run side by side without one stealing the other's events | See "Architecture Patterns" (exact bodies of `_reconcile_classical_nights()`, `_adopted_event_for_night()`, `_may_write()`, `_detach_stale_family_events()` quoted verbatim), "Common Pitfalls" (D-01 rewrite of `TestAdoptAndRekey`/`TestRecordEventNonInterference`) |
| ANNOT-02 | Campaign decoration of an observation-backed event (campaign prefix/label, link to its run) is rendered from the `CalendarEventMeta.run` link at display time, never written into the event's own fields, so base re-projection cannot erase it | See "Architecture Patterns" (`event_form.html`/`calendar.html` quoted), "Common Pitfalls" (the `campaigns:table` `NoReverseMatch` landmine for a campaign-less run) |
</phase_requirements>

## Summary

Phase 33 is pure in-repo Django work: two new nullable FKs on `CalendarEventMeta`
(`observation_record` → `tom_observations.ObservationRecord`, `observation_group` →
`tom_observations.ObservationGroup`), one migration, and an inversion of three methods in
`solsys_code/campaign_reconciler.py` (`_adopted_event_for_night()` deleted,
`_reconcile_classical_nights()` gains a skip branch, `event_title()` drops the campaign
prefix) plus one new shared unlink helper consumed from three existing call sites
(`campaign_views._undo_confirmation`, the reconciler's `_detach_stale_family_events`,
`admin.py`'s two save paths). Decoration is a new template tag pair (cell marker + modal
extension) reading `event.telescope_label_meta.run` — the modal half already exists in
`event_form.html` and only needs extending, not building from scratch.

No new external package is needed anywhere in this phase — every building block
(`OneToOneField`/`ForeignKey` with `on_delete=SET_NULL`, Django admin `readonly_fields`,
`{% simple_tag %}`) is already in use elsewhere in this codebase with a directly
copyable precedent. The highest-risk item this research surfaced that CONTEXT.md does not
explicitly address is a **live landmine in D-13**: `{% url 'campaigns:table' run.campaign_id %}`
resolves against `path('<int:pk>/', ...)` — `campaigns:table` requires a real campaign pk,
and `CampaignRun.campaign` has been nullable since Phase 32 (migration `0015`). A `RUN:`
event whose run has `campaign=None` (legal today, `_reconcile_container`/
`_reconcile_classical_nights` self-attribute unconditionally regardless of campaign
nullness) will hit `NoReverseMatch` the instant D-13's link tag runs against it, not a
silent rendering gap — this is a template `{% url %}` tag, not a Python
`reverse()` call wrapped in try/except anywhere in this codebase. The dev DB currently has
zero campaign-less `CampaignRun` rows (verified this session via direct sqlite3 query), so
no *existing* fixture or fresh dev-DB reconcile will trip it today, but the schema already
allows it and Phase 35's allocation layer will make it common — the decoration tag must
guard on `run.campaign_id is not None` before emitting the link, not merely
`run.is_publicly_visible`.

**Primary recommendation:** Extend `CalendarEventMeta` with two nullable FKs
(`OneToOneField`/`ForeignKey`, `on_delete=SET_NULL`, imported directly from
`tom_observations.models` — the same import style `CampaignRunObservation.observation_record`
already uses, no string reference needed), invert the reconciler's classical-night branch to
skip an attributed night instead of adopting it, write one `unlink_event_from_run()` helper
used by all three existing "clear the link" call sites, and add a decoration tag pair that
reads `event.telescope_label_meta.run` at render time — guarding the campaign-table link on
`run.campaign_id is not None`, which no existing decision or test currently covers.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `CalendarEventMeta` series-identity fields | Database / Storage | API / Backend | New FK columns are pure schema; the model class (`solsys_code/models.py`) is the ORM-facing surface but the constraint (`OneToOneField` uniqueness) is enforced at the DB layer |
| Reconciler write-path inversion | API / Backend | — | `campaign_reconciler.py` is a pure-logic module (no view/HTTP surface) invoked by the management command and staff-action views; the inversion is business logic, not presentation |
| Unlink helper | API / Backend | — | Shared write path consumed by three call sites (a view, the reconciler, an admin `ModelAdmin`) — belongs in a peer utility module (`campaign_utils.py`), not any one of its callers |
| Decoration (cell marker + modal block) | Frontend Server (SSR) | — | Django template tags rendering server-side HTML from a prefetched ORM relation; no client-side JS involved |
| Admin read-only exposure | API / Backend | — | `admin.py` `ModelAdmin`/`TabularInline` configuration, server-rendered Django admin |

## Standard Stack

### Core

No new libraries. This phase uses only what is already installed and imported elsewhere
in the codebase:

| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| `django.db.models.OneToOneField`/`ForeignKey` | Django 2.1+ (project floor, TOM Toolkit) [VERIFIED: solsys_code/models.py:26-42 — `run = models.ForeignKey('CampaignRun', on_delete=models.SET_NULL, null=True, blank=True, related_name='calendar_event_metas', verbose_name='Owning campaign run',)`] | New link fields on `CalendarEventMeta` | Identical form to the existing `run` FK on the same model — direct precedent in the file being edited |
| `tom_observations.models.ObservationRecord` / `ObservationGroup` | tomtoolkit (pinned via `tomtoolkit>=2.31.4` in `pyproject.toml`, installed at `/home/tlister/venv/devel_fomo311_venv/.../tom_observations/models.py`) [VERIFIED: /home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/tom_observations/models.py:11,95 — `class ObservationRecord(models.Model):` / `class ObservationGroup(models.Model):`] | FK targets for the two new fields | Already imported directly (not by string reference) in `solsys_code/models.py:6` — `from tom_observations.models import ObservationRecord`, used by `CampaignRunObservation.observation_record` |

### Supporting

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `{% register.simple_tag %}` (Django template tags) | Decoration cell-marker/modal tags | Same mechanism as `proposal_color`, `status_border_css`, `high_band_attribution_candidates` — already the established pattern in `calendar_display_extras.py`/`attribution_display_extras.py` |
| `ModelAdmin.readonly_fields` | D-09's read-only exposure of the two new fields | Same mechanism already used for `confirmed_by`/`confirmed_at` on both `CalendarEventMetaInline` and `CalendarEventMetaAdmin` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Direct import of `ObservationRecord`/`ObservationGroup` | Django string FK reference (`'tom_observations.ObservationRecord'`) | String references avoid import-order issues in genuinely circular cases, but `solsys_code/models.py` already imports `ObservationRecord` directly at module level with no circularity — matching that precedent is simpler and consistent with the existing `CampaignRunObservation` FK |
| A new `series.py` module for decoration tags | Adding to `calendar_display_extras.py` (CONTEXT.md's own suggestion) | CONTEXT.md leaves this to planner discretion; either is fine, no research-driven reason to deviate from the suggested home |

**Installation:** None — no new package required for this phase.

**Version verification:** N/A — no new package. Django/tomtoolkit versions are the
project's existing pinned versions (`pyproject.toml`); no registry lookup needed since
nothing new is installed.

## Package Legitimacy Audit

**Not applicable.** This phase installs no external packages — it is schema, business
logic, and template work entirely within the existing FOMO/TOM Toolkit dependency set. No
`pip install`/`npm install` step exists in any plan this research supports.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────┐
                         │   Reconciler write path      │
                         │  (campaign_reconciler.py)    │
                         │                              │
  CampaignRun ──────────▶│ reconcile_run()              │
  (approve/site/status)  │  ├─ _reconcile_container()   │───▶ RUN:{pk}           (unchanged)
                         │  └─ _reconcile_classical_    │
                         │      nights()                │
                         │        ├─ existing at        │
                         │        │  RUN:{pk}:{date}?   │───▶ RUN:{pk}:{date}    (unchanged)
                         │        ├─ [NEW D-01] night   │
                         │        │  already attributed │───▶ SKIP — write nothing
                         │        │  (non-RUN: event's  │      (the attributed event
                         │        │  meta.run == run)?  │       IS the night)
                         │        └─ else mint new      │───▶ RUN:{pk}:{date}    (unchanged)
                         │  └─ _detach_stale_family_    │
                         │      events()                │───▶ unlink_event_from_run()
                         └───────────────┬──────────────┘        (NEW shared helper)
                                         │ _link_event_to_run()
                                         ▼
                         ┌─────────────────────────────┐
                         │   CalendarEventMeta          │
                         │  run, is_verified,           │
                         │  confirmed_by, confirmed_at,  │
                         │  [NEW] observation_record,    │
                         │  [NEW] observation_group      │
                         └───────────────┬──────────────┘
                                         │ read at render time only
                                         ▼
        ┌────────────────────────────────────────────────────────┐
        │  Decoration (display-time, template tags)                │
        │  calendar.html month cell  ──▶ {% campaign_chip event %} │
        │  event_form.html modal     ──▶ existing "Campaign run"   │
        │                                  block, extended (D-13)  │
        │  GUARD: run.campaign_id is not None before emitting the  │
        │         campaigns:table link (NoReverseMatch otherwise)  │
        └────────────────────────────────────────────────────────┘

  Other unlink call sites (all converge on the same NEW helper, D-16):
  campaign_views._undo_confirmation()  ──┐
  admin.py CalendarEventMetaAdmin      ──┼──▶ unlink_event_from_run()
    .save_model() branch 2               │      clears run + confirmed_by + confirmed_at
  campaign_reconciler                  ──┘      NEVER touches is_verified or CalendarEvent
    ._detach_stale_family_events()             (today's bulk .update(run=None) must gain
                                                 the confirmed_by/confirmed_at clear too)
```

### Recommended Project Structure

No new files/directories — all changes land in existing modules:

```
solsys_code/
├── models.py                       # CalendarEventMeta: +2 FKs, verbose_name rename (D-17)
├── migrations/
│   └── 0017_...py                  # AddField x2 (+ AlterField verbose_name, maybe combined)
├── campaign_reconciler.py          # _adopted_event_for_night() deleted; _reconcile_
│                                    # classical_nights() gains skip branch; event_title()
│                                    # drops campaign prefix; _detach_stale_family_events()
│                                    # routes through the new unlink helper
├── campaign_utils.py                # NEW: unlink_event_from_run() (or reconciler module —
│                                    # planner's discretion, D-16)
├── campaign_views.py                # _undo_confirmation() calls the new helper instead of
│                                    # its own inline .update(run=None, confirmed_by=None, ...)
├── admin.py                        # CalendarEventMetaAdmin/CalendarEventMetaInline: new
│                                    # fields in readonly_fields; save_model branch 2 calls
│                                    # the new helper
├── templatetags/
│   └── calendar_display_extras.py  # NEW: decoration tag(s) (cell marker + modal extension)
└── tests/
    ├── test_campaign_reconciler.py       # TestAdoptAndRekey rewritten under D-01;
    │                                    # TestRecordEventNonInterference likely unaffected
    │                                    # (uses CampaignRunObservation, not meta.run — see
    │                                    # Common Pitfalls)
    ├── test_calendar_utils.py            # update_calendar_event_key_and_fields tests,
    │                                    # if the function is deleted per discretion item
    ├── test_calendar_template.py         # EventModalCampaignRunLinkTest extended for D-13
    │                                    # link + the campaign-less-run guard
    └── test_campaign_attribution_views.py # TestConfirmUndo asserts on the shared helper

src/templates/tom_calendar/partials/
├── calendar.html                   # month cell: add decoration marker inside the event loop
└── event_form.html                 # modal: extend existing {% with run=... %} block (D-13 link)

docs/notebooks/pre_executed/
├── reconcile_campaign_runs_demo.ipynb   # NEW cell: before/after snapshot diff (D-04)
└── campaign_lifecycle_demo.ipynb        # decoration wording/link updated (D-17)

docs/runbooks/
└── telescope_runs_calendar.rst     # 3 sections reworded: "How do I get every campaign run
                                    # onto the calendar?" (adopt language removed), "Why
                                    # doesn't the calendar pop-up show a 'Campaign run'
                                    # block?" (owning → attributed), "How do I attribute
                                    # existing calendar events..." (unaffected wording, but
                                    # check for "owning" cross-references)
```

### Pattern 1: The existing `run` FK is the exact template for the two new FKs

**What:** `CalendarEventMeta.run` (quoted verbatim below) is the field to copy the form
of, substituting `on_delete=models.SET_NULL, null=True, blank=True` and picking new
`related_name`s.

**Verbatim source** [VERIFIED: solsys_code/models.py:26-42]:
```python
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
    run = models.ForeignKey(
        'CampaignRun',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='calendar_event_metas',
        verbose_name='Owning campaign run',
    )
```

**When to use:** Directly for `observation_group` (plain `ForeignKey`, same form as
`run`). For `observation_record`, use `OneToOneField` instead of `ForeignKey` per D-05 (DB-
enforced one event per record) — same `on_delete=models.SET_NULL, null=True, blank=True`
kwargs, `OneToOneField` in place of `ForeignKey`.

**Import form** — direct import already established in this file, not a string reference
[VERIFIED: solsys_code/models.py:6, and solsys_code/models.py:458-463 quoted below]:
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
`ObservationGroup` is not currently imported in `models.py` and must be added to the same
`from tom_observations.models import ...` line.

### Pattern 2: The reconciler's `_may_write()` / ownership check stays; only the classical-night resolution order changes

**What:** `_may_write()` (quoted below, unchanged by this phase per D-02) is the ownership
gate every write already passes through.

**Verbatim source** [VERIFIED: solsys_code/campaign_reconciler.py:223-239]:
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

**What changes:** `_reconcile_classical_nights()`'s per-night resolution order today is
"(1) `RUN:{pk}:{date}` exists → use it; (2) else `_adopted_event_for_night()` finds a
blank-url event already attributed to this run for this night, and re-keys it; (3) else
mint" [VERIFIED: solsys_code/campaign_reconciler.py:356-362, quoted in full below]. Under
D-01 this becomes: "(1) `RUN:{pk}:{date}` exists → use it (unchanged); (2) else — instead
of calling `_adopted_event_for_night()` and re-keying — check whether *any* non-`RUN:`
event is already attributed to this run for this night (same underlying query
`_adopted_event_for_night()` used: `CalendarEventMeta.objects.filter(run_id=run.pk,
event__url='')` matched on site-local night, generalised per D-01's Phase-34 forward
reference to "a load_telescope_runs event... or a Phase 34 observation event" — i.e. not
restricted to `event__url=''` once Phase 34 lands, though this phase's own fixture only
needs the blank-url case since PROJ-04's link fields carry no data until Phase 34 per
D-08); if found, skip — write nothing; (3) else mint as today."

**Verbatim source of the retiring docstring** (documents exactly what D-01 removes)
[VERIFIED: solsys_code/campaign_reconciler.py:356-362]:
```python
    Per-night resolution order (D-02): (1) an existing event already keyed at
    ``run_night_url(run, night)`` -- the common idempotent-rerun case; (2) failing that,
    ``_adopted_event_for_night(...)`` -- a ``load_telescope_runs``-created event already
    attributed to this run for this night via Phase 28's confirmation queue, re-keyed in
    place; (3) failing both, mint a new event. Both (1) and (2) go through ``_may_write()``
    before any write -- the ownership rule stays the first condition checked (RECON-05
    defence in depth; see T-29-05).
```

**When to use:** This is the single most important behavior change in the phase — every
plan task touching `_reconcile_classical_nights()` must reference this exact diff, not a
paraphrase, since `TestAdoptAndRekey`'s three tests (quoted in Common Pitfalls below)
currently assert the *opposite* behavior and must be rewritten, not merely left passing.

### Pattern 3: The modal decoration block already exists — extend, don't rebuild

**What:** `event_form.html`'s existing `{% with run=event.telescope_label_meta.run %}`
block, quoted in full below, is the D-10/D-13 modal target.

**Verbatim source** [VERIFIED: src/templates/tom_calendar/partials/event_form.html:118-153]:
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

**When to use:** D-13 adds a `#run-{pk}` anchor to the existing `{% url 'campaigns:table'
run.campaign_id %}` link (`{% url 'campaigns:table' run.campaign_id %}#run-{{ run.pk }}`),
and D-17 renames the label from "Campaign run" to reflect "attributed to" wording. D-11
says the modal keeps showing exactly what it shows today (telescope/instrument, window,
`get_run_status_display`) for every linked event, not only observation-backed ones — no
new fields are needed in this block beyond the anchor. **The `run.campaign_id` guard
(see Common Pitfalls) must be added to the `{% if %}` condition** —
`{% if run.is_publicly_visible and run.campaign_id %}` — before this phase's decoration
work reaches any run created without a campaign (none exist in the dev DB today, but the
schema permits it and this is new code, not an existing tested path).

### Pattern 4: Month-cell prefetch — extend the existing `select_related`/`prefetch_related` chain

**What:** `fomo_render_calendar` already prefetches `telescope_label_meta` for the N+1 fix
(DISPLAY-09).

**Verbatim source** [VERIFIED: solsys_code/views.py:107-115]:
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

**When to use:** The cell marker needs `run` and `run.campaign` (campaign name for the
tooltip, D-11), so the prefetch chain needs `Prefetch('telescope_label_meta',
queryset=CalendarEventMeta.objects.select_related('run__campaign'))` or an equivalent
`select_related` addition — a bare `.prefetch_related('telescope_label_meta')` alone still
N+1s on `.run.campaign.name` per event. **`solsys_code/views.py` is the heavy-import
module** (CLAUDE.md: importing it triggers a ~1.6 GB SPICE kernel download) — this edit is
still required (it is the only place `fomo_render_calendar` lives), but no test or probe
run during planning/execution should import this module directly; the existing Django test
suite already routes around it via HTTP client calls in `test_calendar_template.py`, not
direct imports.

### Anti-Patterns to Avoid
- **Writing campaign text into `CalendarEvent.title`/`description` from the decoration
  path:** the whole point of D-12/ANNOT-02 is that decoration is read-only at render time;
  a `save()` call anywhere in the new template tags is a structural violation, not just a
  style issue — the next reconcile or Phase 34 projection would silently discard it.
- **Restricting the D-01 skip check to `event__url == ''`:** `_adopted_event_for_night()`'s
  existing filter is `event__url=''` because today only `load_telescope_runs` produces
  blank-url attributed events; D-01's own text says the skip rule must also cover "a Phase
  34 observation event later" (which will be URL-keyed, not blank). Hard-coding the blank-
  url filter into the new skip logic silently narrows the rule versus what D-01 states,
  even though this phase's own fixtures only exercise the blank-url case (D-08: no
  backfill yet, so no URL-keyed attributed event exists to test against in Phase 33).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ownership/attribution check on write | A new query pattern | `_may_write()` (unchanged, D-02) | Already correct and already the first condition on every write path; this phase changes *what* gets checked against it (removing the adopt candidate), not the check itself |
| Attribution-clearing write | Ad-hoc `.update(run=None)` at each call site | The new shared `unlink_event_from_run()` (D-16) | Three existing call sites (`_undo_confirmation`, `_detach_stale_family_events`, admin `save_model`) each clear the link slightly differently today (see Common Pitfalls) — consolidating removes the risk of one of them drifting out of sync with "never touches `is_verified`" |
| No-churn create/update | Hand-written `if changed: save()` | `insert_or_create_calendar_event()` / `_update_or_unchanged()` (existing, `calendar_utils.py`) | Already the shared contract every write path in this module uses; this phase's D-01 change means the classical-night loop calls it *less often* (skip = no call at all), not differently |

**Key insight:** almost nothing in this phase is new machinery — it is removing one write
path (`_adopted_event_for_night()` + its re-key call) and consolidating three duplicated
"clear the link" call sites into one. The risk in this phase is behavioral (get the skip
condition and the unlink semantics exactly right) rather than a hand-rolled-vs-library
question.

## Runtime State Inventory

Phase 33 is additive schema (nullable FKs) plus a reconciler behavior inversion — not a
rename/rebrand/refactor of an identifier string, so the rename/refactor trigger for this
section does not strictly apply. It is included regardless because the phase does invert
a field's *meaning* (`run`: "owns" → "attributed to") across running systems, which is the
same category of question ("what runtime state still assumes the old meaning after every
file is updated?").

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Dev DB (`src/fomo_db.sqlite3`, verified via direct sqlite3 query this session): 0 `CampaignRun` rows with `campaign_id IS NULL` today, so 0 rows currently at risk of the `campaigns:table` `NoReverseMatch` landmine. 74 `RUN:`-keyed events, 85 `CalendarEventMeta` companion rows (75 with `run` set, 1 of them on a non-`RUN:` event) — none of these rows' *data* needs migrating; only the code that *reads* `run`'s meaning changes. | Code edit only (reconciler + templates + docstrings); no data migration required for existing rows — D-03 explicitly keeps past adopts as-is |
| Live service config | None — no external service (n8n, Datadog, etc.) is involved in this phase; the reconciler and calendar are entirely in-repo Django. | None |
| OS-registered state | None — no cron/task-scheduler/pm2 registration touches this phase's code (that is Phase 36's SCHED-08 territory). | None |
| Secrets/env vars | None — no secret or env var references `CalendarEventMeta.run`, the reconciler, or the decoration path. | None |
| Build artifacts | None — no installed package or egg-info is affected; the only artifact is the new migration file (`0017_…`), which is a normal additive migration, not a stale build artifact. | Standard `python manage.py migrate` after the plan lands; no reinstall needed |

**Nothing found in categories 2-4** — verified by inspecting the phase's actual scope
(no cron/service/secret file was named in CONTEXT.md's canonical refs, and grep across
`solsys_code/` for `CalendarEventMeta.run`/`meta.run` usage this session surfaced only the
in-repo call sites already covered in Architecture Patterns and Common Pitfalls).

## Common Pitfalls

### Pitfall 1: The `campaigns:table` link breaks for a campaign-less `RUN:` event
**What goes wrong:** `{% url 'campaigns:table' run.campaign_id %}` raises `NoReverseMatch`
(a hard template error, not a silently-empty render) the first time this decoration code
runs against a `CampaignRun` whose `campaign` is `None`.
**Why it happens:** `campaigns:table`'s URL pattern is `path('<int:pk>/', ...)`
[VERIFIED: solsys_code/campaign_urls.py:42 — `path('<int:pk>/', CampaignRunTableView.as_view(), name='table'),`],
which requires an integer; `CampaignRun.campaign` has been nullable since Phase 32
(migration `0015_campaignrun_nullable_campaign_and_source_identifier.py`), and the
reconciler's `_link_event_to_run()` attributes every `RUN:` event it creates
*unconditionally*, with no campaign-nullness check [VERIFIED: solsys_code/campaign_reconciler.py:242-252].
The existing `is_publicly_visible` gate (quoted in Pattern 3) does **not** check campaign
nullness either [VERIFIED: solsys_code/models.py:293-304 — `return self.approval_status !=
self.ApprovalStatus.PENDING_REVIEW`].
**How to avoid:** Guard the decoration link (both the modal's D-13 extension and any new
cell-marker link) on `run.campaign_id is not None`, not merely `run.is_publicly_visible`.
CONTEXT.md's own decisions (D-10/D-11/D-13/D-14) never mention campaign-nullness — this is
a genuine gap this research found, not a restatement of a locked decision, so it should be
flagged to the planner as a required addition rather than assumed pre-approved.
**Warning signs:** A `500` on the calendar month view or the event modal the first time a
campaign-less `CampaignRun` gets reconciled and rendered (0 such rows exist in the dev DB
today, so this will not surface in a manual smoke test against the current dev DB — it
needs a dedicated fixture test, exactly the kind `EventModalCampaignRunLinkTest`
(`test_calendar_template.py:403-472`) does not currently have, since every fixture run
there sets `campaign=cls.campaign`).

### Pitfall 2: Three separate call sites clear the link with three different call signatures today — consolidating them is not a pure rename
**What goes wrong:** Assuming `unlink_event_from_run()` can be a drop-in replacement with
identical call signatures at all three sites risks silently changing behavior at whichever
site the planner doesn't look closely at.
**Why it happens:** The three existing writers differ in call signature:
1. `campaign_views._undo_confirmation()` — a **conditional bulk `.update()`**, filtered on
   both `event_id` and `run_id` for concurrency safety [VERIFIED: solsys_code/campaign_views.py:1326-1328 —
   `changed_count = CalendarEventMeta.objects.filter(event_id=orphan_pk, run_id=run_pk).update(run=None, confirmed_by=None, confirmed_at=None)`],
   and the caller needs the `changed_count` return value to decide whether to also write a
   dismissal row.
2. `campaign_reconciler._detach_stale_family_events()` — a **bulk `.update()` over a
   queryset of many rows**, clearing only `run`, not `confirmed_by`/`confirmed_at`
   [VERIFIED: solsys_code/campaign_reconciler.py:470-471 —
   `stale = owned_events(run).exclude(url__in=active_urls)` /
   `CalendarEventMeta.objects.filter(event__in=stale, run=run).update(run=None)`]. Per D-16
   this needs to *start* clearing `confirmed_by`/`confirmed_at` too — a genuine behavior
   change from today, not a refactor-preserving-behavior move.
3. `admin.py CalendarEventMetaAdmin.save_model()` branch 2 — a **single in-memory instance
   mutation** before `super().save_model()` [VERIFIED: solsys_code/admin.py:382-386 —
   `elif obj.run_id is None and prior_run_id is not None: obj.confirmed_by = None;
   obj.confirmed_at = None`], relying on the form having already set `obj.run_id = None`.
**How to avoid:** Design `unlink_event_from_run()` to serve the bulk-conditional case (take
a queryset or a `(event_id, run_id)` filter and return a changed count) since that is the
strictest caller (site 1); sites 2 and 3 can call it with a narrower queryset/single
instance. Verify the reconciler's detach step's *new* behavior (now clearing
`confirmed_by`/`confirmed_at`) is deliberately tested, since it changes observable admin
history for detached rows.
**Warning signs:** A test asserting `_detach_stale_family_events()` leaves
`confirmed_by`/`confirmed_at` unchanged (none currently exists, so this is a new test to
write, not an existing one to keep green) failing, or `_undo_confirmation`'s `changed_count`
becoming `0` unexpectedly because the shared helper doesn't preserve the `run_id=run_pk`
conditional.

### Pitfall 3: Rewriting `TestAdoptAndRekey` is not optional — it currently asserts the exact behavior D-01 retires
**What goes wrong:** Leaving `TestAdoptAndRekey` (`test_campaign_reconciler.py:381-476`)
unmodified after implementing D-01 leaves 3 failing tests, or worse, a planner "fixes" the
test to pass without checking it now asserts the *right* new behavior.
**Why it happens:** All three tests in this class construct a blank-url event with
`CalendarEventMeta.objects.create(event=adopted_event, run=run)` and then assert the event
gets **re-keyed** to `RUN:{run.pk}:{night}` [VERIFIED: solsys_code/tests/test_campaign_reconciler.py:404-430,
quoted assertion: `self.assertEqual(adopted_event.url, f'RUN:{run.pk}:{first_night.isoformat()}')`].
Under D-01 the correct new behavior is: the attributed event's `url` stays unchanged
(never re-keyed), `CalendarEvent.objects.count()` for the run's window is 1 (the attributed
event) + 1 per un-attributed night (not 2 for a 2-night window where one night is
attributed), and the run's `ReconcileResult` should report the attributed night as skipped
(via the optional `skipped_nights` counter, planner's discretion) rather than `updated`.
**How to avoid:** Rewrite (not delete) each of the three tests in this class to assert the
skip behavior; keep the `test_adopt_matches_on_site_local_night_not_naive_utc_date` test's
site-local-night matching logic (still needed — D-01's skip check uses the same
site-local-night comparison `_adopted_event_for_night()` used), but assert "no `RUN:` event
minted for that night" instead of "re-keyed."
**Warning signs:** CI green with `TestAdoptAndRekey` still asserting `RUN:{run.pk}:{date}`
url values after the reconciler no longer re-keys anything — a silent contradiction between
test and code that only a close read (not a test-run) will catch, since Python doesn't
error on an assertion that happens to still be checking the old (now-impossible-to-produce
via this path, but not otherwise-prevented) behavior unless the fixture setup itself changes.

### Pitfall 4: `TestRecordEventNonInterference` is very likely unaffected by D-01 — don't over-rewrite it
**What goes wrong:** Assuming every test in `test_campaign_reconciler.py` involving
`CampaignRunObservation` needs updating for D-01, and touching
`TestRecordEventNonInterference` (`test_campaign_reconciler.py:600-682`) unnecessarily.
**Why it happens:** This test's fixture links a record-derived event to a run **only**
via `CampaignRunObservation.objects.create(run=run, observation_record=record)`
[VERIFIED: solsys_code/tests/test_campaign_reconciler.py:648] — it never creates a
`CalendarEventMeta` row for that event at all. D-01's skip rule keys off
`CalendarEventMeta.run` (the attribution link), not `CampaignRunObservation` (D-14: "No
render-time fallback to the record's `CampaignRunObservation`" — the two are deliberately
separate mechanisms). Since this fixture's record-event has no `CalendarEventMeta` row,
the reconciler still cannot see it as "attributed," and the existing assertion (both
per-night `RUN:` events still get minted alongside the record event,
`CalendarEvent.objects.count() == 1 + n_nights`) should remain correct after D-01.
**How to avoid:** Leave this test's behavior expectations as-is; only add a *new* test
(or extend this fixture) that additionally creates the `CalendarEventMeta(event=record_event,
run=run)` link to prove D-01's skip rule actually fires for a URL-keyed (non-blank-url)
attributed event — this is the "Phase 34 observation event later" case D-01's own text
names, and no existing test exercises it since D-08 defers all URL-keyed linking to Phase
34.
**Warning signs:** A plan task that rewrites this whole class "to be safe" without first
confirming (via the `CalendarEventMeta` absence) that its current assertions are actually
unaffected — wasted work and a needless diff against a test that documents an important
non-interference contract that must survive verbatim.

### Pitfall 5: Migration dependency — `tom_observations` app dependency must be declared explicitly
**What goes wrong:** Django's migration autodetector will generate the new `AddField`
operations correctly, but the migration's `dependencies` list must include a
`('tom_observations', '...')` entry or a cross-app FK can produce a circular/ordering
migration error.
**Why it happens:** Two prior migrations in this codebase already needed this exact
dependency for FKs into `tom_observations` [VERIFIED: solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py:29
and solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py:22
— both declare `('tom_observations', '0016_alter_facility_options')`].
**How to avoid:** Run `python manage.py makemigrations solsys_code` (which will
auto-detect the dependency correctly from the installed `tom_observations` app) rather than
hand-writing the migration file from scratch; verify the generated `dependencies` list
includes a `tom_observations` entry matching (or newer than) `0016_alter_facility_options`
before committing it.
**Warning signs:** `django.db.utils.ProgrammingError` or a migration-ordering error at
`python manage.py migrate` time if the dependency is hand-omitted.

### Pitfall 6: `update_calendar_event_key_and_fields()` has a second, unrelated caller that must not be broken
**What goes wrong:** Deleting `update_calendar_event_key_and_fields()` (one of the
discretion items) without checking `_reconcile_container()`'s own call to it
(`solsys_code/campaign_reconciler.py:285`) breaks the container branch, which is
**unrelated to D-01** and must keep working exactly as today.
**Why it happens:** The function has exactly two call sites in `campaign_reconciler.py`
[VERIFIED: grep this session, solsys_code/campaign_reconciler.py:285 and :429] — line 285
is inside `_reconcile_container()` (the class-wide/satellite branch, untouched by D-01) and
line 429 is inside `_reconcile_classical_nights()`'s adopt-and-rekey step (the one D-01
retires). Only the second call site goes away; the first does not.
**How to avoid:** If choosing to delete the function (discretion item), the container
branch's call at line 285 must be replaced with a plain `_update_or_unchanged(existing,
fields)`-equivalent call (same url, so no re-key is actually happening there today either —
CONTEXT.md's own discretion note already observes "the container branch is its only other
caller and never changes the url"). If choosing to keep the function, no change is needed
at line 285 at all.
**Warning signs:** `test_calendar_utils.py`'s `update_calendar_event_key_and_fields` tests
(`solsys_code/tests/test_calendar_utils.py:469,482,486`) failing if the function is deleted
but a caller still references it, or an `ImportError` in `campaign_reconciler.py` if the
import line isn't updated to match whichever choice is made.

## Code Examples

### New FK fields on `CalendarEventMeta` (skeleton, following Pattern 1)

```python
# Source: pattern copied verbatim from solsys_code/models.py:26-42 (the existing `run` FK)
# and solsys_code/models.py:6,458-463 (the existing direct-import ObservationRecord FK)
from tom_observations.models import ObservationGroup, ObservationRecord  # add ObservationGroup

class CalendarEventMeta(models.Model):
    ...
    observation_record = models.OneToOneField(
        ObservationRecord,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='calendar_event_meta',  # discretion: pick a related_name with no
                                              # existing clash (none exists today per this
                                              # session's grep of tom_observations.models.py)
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

### D-01's skip check (illustrative form, not a full rewrite)

```python
# Source: adapted from the retiring _adopted_event_for_night() query
# (solsys_code/campaign_reconciler.py:332-341) -- the query form survives, the
# consequence (skip vs. re-key) is what D-01 changes.
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

Note this drops `_adopted_event_for_night()`'s `event__url=''` restriction (see Anti-
Patterns above) so it also matches a future URL-keyed Phase 34 observation event, per
D-01's own stated scope — while this phase's own tests only need the blank-url case (D-08).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `CalendarEventMeta.run` unset/set means "not owned"/"owned" (Phase 26-29's model) | `run` unset/set means "not attributed"/"attributed" — the reconciler no longer writes to or re-keys an event it doesn't itself create | This phase (33) | Every reader of `run` (templates, admin labels, docstrings) must switch wording; every writer of `run` outside the reconciler's own `RUN:` namespace (Phase 28's queue, admin) is unaffected in mechanism, only in what the field now *means* |
| Campaign name embedded in `event_title()`'s string (`"{campaign.name}: {telescope_instrument}"`) | Campaign name rendered only via the decoration tag, never in the title string | This phase (33), D-12 | One-time title churn on the next sweep for the 74 existing `RUN:` events — expected and already scoped into the paired-docs update |

**Deprecated/outdated:**
- `_adopted_event_for_night()`: retired outright by D-01, not merely deprecated — its
  docstring's own re-keying contract is exactly what D-01 says must stop happening.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The D-01 skip check should generalize `_adopted_event_for_night()`'s query to any non-`RUN:` attributed event (not just blank-url), per D-01's forward reference to "a Phase 34 observation event later," even though no test in this phase exercises the URL-keyed case | Architecture Patterns Pattern 2, Code Examples | Low — if the planner narrows the skip check to blank-url only, Phase 34 will need to revisit `_reconcile_classical_nights()` again when it starts writing URL-keyed attributed events; not a Phase-33 correctness bug, only a scope-boundary judgment call CONTEXT.md leaves implicit rather than stating as a locked decision |
| A2 | `unlink_event_from_run()` should accept a queryset/filter signature general enough to serve `_undo_confirmation`'s conditional bulk `.update()`, rather than only a single-instance signature | Common Pitfalls Pitfall 2 | Medium — if the planner designs a single-instance-only helper, `_undo_confirmation` either can't use it cleanly or loses its concurrency-safe conditional update; this is a design recommendation from this research, not a CONTEXT.md-locked API signature (D-16 leaves the exact signature to the planner) |
| A3 | The `campaigns:table` `NoReverseMatch` risk (Pitfall 1) needs an explicit guard/test even though CONTEXT.md's decisions never mention campaign-nullness in the decoration context | Common Pitfalls Pitfall 1, Summary | Medium — if unaddressed, this is a genuine template crash risk (not merely a display gap) the moment Phase 35's allocation layer creates campaign-less runs whose events get decorated; currently latent because the dev DB has 0 such rows today (verified via direct query) |

## Open Questions (RESOLVED)

Both questions were closed by the planner during plan revision (2026-09-03). No open
research question remains for this phase.

1. **Does the D-01 skip rule need a companion migration/backfill note for the 1 non-`RUN:`
   event in the dev DB that already has `run` set today (per CONTEXT.md's "85 companion
   rows, 75 with run set, 1 of them on a non-`RUN:` event")?**
   - What we know: CONTEXT.md's dev-DB baseline explicitly names this row.
   - What's unclear: whether that row's night is inside any currently-approved run's
     window (if so, D-01's skip behavior will visibly change that night's calendar state
     the next time `reconcile_campaign_runs` sweeps — expected per D-01, but worth a
     planner checkpoint to confirm the sweep's dry-run output is inspected before the real
     sweep runs against the dev DB, matching D-04's "real-DB diff" proof).
   - Recommendation: the planner should have the D-04 notebook's before/after diff cell
     (already required) double as this check — if the diff isn't empty for that one row in
     a surprising way, it's the same evidence either way.
   - **RESOLVED — no migration or backfill; a dry-run inspection precedes the real sweep.**
     D-08 forbids any data step in this phase, so nothing about that row is migrated or
     backfilled. The recommendation's inspection is now an explicit, ordered step rather
     than a by-product of the diff: plan 33-05 Task 1's notebook takes the "before"
     snapshot, then runs `reconcile_campaign_runs --dry-run` and prints its previewed
     actions and its per-run `skipped_nights` for every non-`RUN:` event it would touch,
     **and only then** runs the real sweep and takes the "after" snapshot. The dry-run cell
     asserts the preview names no non-`RUN:` url at all, which is the direct check on that
     one row — a surprise is caught before any write, not diagnosed after one. The
     post-sweep diff is kept as the independent confirmation. Plan 33-01 Task 2's
     URL-keyed fixture test covers the same rule at unit level, so the phase does not
     depend on the state of any particular developer-database row.

2. **Should the decoration cell marker be a new `{% simple_tag %}` returning a dict (for a
   template `{% include %}`) or an `{% inclusion_tag %}` rendering its own partial
   directly?**
   - What we know: every existing tag in `calendar_display_extras.py` is a `simple_tag`
     returning a scalar (color string, CSS string, count) consumed inline by the calling
     template — no `inclusion_tag` precedent exists in this codebase today.
   - What's unclear: whether the marker's HTML (chip/icon/dot, per CONTEXT.md's discretion
     item) is simple enough to stay inline in `calendar.html` via a `simple_tag`, or complex
     enough to warrant its own partial.
   - Recommendation: default to `simple_tag` for consistency with the established pattern
     unless the marker's HTML grows non-trivial; this is a planner-level implementation
     choice, not a research gap.
   - **RESOLVED — `simple_tag`, following the recommendation.** Plan 33-01 Task 1 adds a
     single `@register.simple_tag` `campaign_decoration(event)` in
     `calendar_display_extras.py` returning `None` or a fixed-key dict, and plan 33-02
     Task 1 consumes the same tag inline in `calendar.html`'s two month-cell event loops.
     No `inclusion_tag` and no new partial is introduced, so the codebase keeps one
     template-tag convention.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (built into Django, no pytest — confirmed in `CLAUDE.md` and this project's `.planning/config.json`) |
| Config file | none — `python manage.py test` uses Django's own test discovery, not a `pytest.ini`/`setup.cfg` |
| Quick run command | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_calendar_template solsys_code.tests.test_campaign_attribution_views` |
| Full suite command | `LABELS=$(ls solsys_code/tests/test_*.py solsys_code/solsys_code_observatory/tests/test_*.py | grep -v "tests/test_views\.py$" | sed "s|/|.|g; s|\.py\$||" | tr "\n" " "); python manage.py test $LABELS && python manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex solsys_code.tests.test_views.TestJPLSBDBQuery` [VERIFIED: .planning/config.json — `workflow.test_command`, quoted verbatim] |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PROJ-04 | `CalendarEventMeta.observation_record`/`observation_group` are nullable FKs; existing companion rows migrate with `run`/`is_verified`/`confirmed_by`/`confirmed_at` intact | unit (model/migration) | `python manage.py test solsys_code.tests.test_models` (or wherever a new `CalendarEventMeta` field test lands — no existing `test_models.py` found this session; likely a new file or an addition to an existing model-adjacent test) | ❌ Wave 0 — no dedicated `CalendarEventMeta` field test exists today |
| ANNOT-01 (D-01 skip) | An attributed night is skipped, not re-keyed/adopted | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestAdoptAndRekey` (rewritten) | ✅ exists, needs rewriting (Pitfall 3) |
| ANNOT-01 (D-04 proof) | A full `reconcile_campaign_runs` sweep leaves every non-`RUN:` event byte-identical | unit + notebook | `python manage.py test solsys_code.tests.test_campaign_reconciler` + `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | ❌ Wave 0 — new fixture test + new notebook cell both needed |
| ANNOT-02 (decoration) | Cell marker + modal block render from the link, survive title/description rewrite | unit (template) | `python manage.py test solsys_code.tests.test_calendar_template.EventModalCampaignRunLinkTest` (extended) | ✅ exists, needs extending |
| ANNOT-02 (campaign-less guard, Pitfall 1) | Decoration link is omitted (not a crash) for a `run.campaign_id is None` event | unit (template) | new test in `test_calendar_template.py` | ❌ Wave 0 — no existing fixture covers a campaign-less run in this test class |
| D-16 (unlink helper) | `unlink_event_from_run()` clears `run`+`confirmed_by`+`confirmed_at`, never `is_verified`, from all three call sites | unit | `python manage.py test solsys_code.tests.test_campaign_attribution_views.TestConfirmUndo` + a new reconciler-detach test + a new admin test | Partially ✅ (`TestConfirmUndo` exists) / ❌ (reconciler-detach clearing `confirmed_by`/`confirmed_at` is new behavior with no existing test) |
| D-09 (admin read-only) | New fields are read-only in both admin surfaces | unit | new admin test, or manual smoke via Django admin | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** the quick run command above (reconciler + calendar-template +
  attribution-view suites — the three files this phase's diff concentrates in)
- **Per wave merge:** the full suite command
- **Phase gate:** full suite green before `/gsd-verify-work`, plus both paired notebooks
  re-executed (`jupyter nbconvert --to notebook --execute --inplace`) per CLAUDE.md's
  paired-docs rule

### Wave 0 Gaps
- [ ] A `CalendarEventMeta` field-level test (new file or addition) covering the two new
      FKs' nullability, `on_delete=SET_NULL` behavior, and that existing rows migrate with
      `run`/`is_verified`/`confirmed_by`/`confirmed_at` untouched — covers PROJ-04's
      success criterion 1 directly
- [ ] A fixture test proving D-04's "byte-identical after reconcile" claim for both a
      blank-url and a URL-keyed attributed event (the URL-keyed case has no existing
      fixture anywhere in `test_campaign_reconciler.py` — every existing attributed-event
      fixture in that file uses either blank url or the run's own `RUN:` namespace)
- [ ] A `test_calendar_template.py` fixture with `campaign=None` on the linked
      `CampaignRun`, proving the decoration link is safely omitted rather than raising
      `NoReverseMatch` (Pitfall 1) — no existing fixture in `EventModalCampaignRunLinkTest`
      exercises a campaign-less run
- [ ] A test proving the reconciler's `_detach_stale_family_events()` clears
      `confirmed_by`/`confirmed_at` alongside `run` once routed through the new shared
      helper — this is new behavior, not currently tested since today's detach step
      deliberately does not touch those fields

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No new auth surface — all three unlink call sites already sit behind existing staff/auth gates (`request.user.is_staff` checks, Django admin's own auth) |
| V3 Session Management | no | Unaffected |
| V4 Access Control | yes | The new FK fields are `readonly_fields` in both admin surfaces (D-09) — this is itself the access-control control: only reconciler/projector code (not a staff form submission) may write `observation_record`/`observation_group`. The decoration's staff-only "Possible campaign run match" hint already gates on `request.user.is_staff` (existing pattern, unchanged) |
| V5 Input Validation | yes | No new user-facing input in this phase — the new FKs are written only by code (D-09), never bound from a form. The unlink helper narrows a write surface, it doesn't add one |
| V6 Cryptography | no | Not applicable |

### Known Threat Patterns for {stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| A staff member (or a stale/tampered POST, per the existing WR-01 finding pattern already documented in `_undo_confirmation`'s own docstring) clearing an attribution they didn't confirm | Tampering | `_undo_confirmation`'s existing conditional `.filter(event_id=orphan_pk, run_id=run_pk).update(...)` pattern (already in place, not newly introduced) must be preserved by the shared `unlink_event_from_run()` helper — Pitfall 2 above documents exactly this requirement |
| A campaign-less run's decoration crashing the calendar view for every visitor (a denial-of-service-style bug, not a deliberate attack, but same category of "one bad row takes down a shared public page") | Denial of Service (availability) | The `run.campaign_id is not None` guard (Pitfall 1) — this is a correctness fix with an availability angle: the calendar month view and event modal are public, unauthenticated surfaces, so an unhandled `NoReverseMatch` there is a full-page failure for every visitor, not a scoped error |
| Information disclosure via the new FK fields in the admin | Information Disclosure | Not a new risk — `observation_record`/`observation_group` point at data already visible to staff (the admin is a staff-only surface throughout this codebase); read-only exposure (D-09) does not add a new disclosure surface beyond what `run` already established for `CampaignRun` |

## Sources

### Primary (HIGH confidence — read this session)
- `solsys_code/campaign_reconciler.py` (full file) — `_adopted_event_for_night()`,
  `_reconcile_classical_nights()`, `_may_write()`, `writable_events()`, `owned_events()`,
  `_detach_stale_family_events()`, `_link_event_to_run()`, `event_title()`,
  `reconcile_run()`, `ReconcileResult` — all bodies quoted verbatim above
- `solsys_code/models.py` — `CalendarEventMeta` (lines 1-73), `CampaignRunObservation`
  (lines 432-491), `CampaignRun.is_publicly_visible` (lines 293-304)
- `solsys_code/calendar_utils.py` — `update_calendar_event_key_and_fields()`,
  `insert_or_create_calendar_event()`, `_update_or_unchanged()`, `record_time_window()`
- `solsys_code/campaign_views.py` — `_undo_confirmation()` (lines 1303-1357), the
  `_do_confirm_event()` method for context
- `solsys_code/admin.py` — `CalendarEventMetaInline`, `CalendarEventMetaAdmin.save_model()`,
  `CampaignRunAdmin.save_formset()`
- `solsys_code/campaign_utils.py` — `adopt_event_into_run()` (lines 860-895)
- `solsys_code/campaign_attribution.py` — `orphan_calendar_events()` (lines 453-476)
- `src/templates/tom_calendar/partials/event_form.html` — the existing "Campaign run"
  block, quoted in full
- `src/templates/tom_calendar/partials/calendar.html` — month-cell loop, `truncatechars:18`
  and `truncatechars:16` confirmed
- `solsys_code/views.py` — `fomo_render_calendar()` prefetch chain
- `solsys_code/templatetags/calendar_display_extras.py`,
  `solsys_code/templatetags/attribution_display_extras.py` — existing `simple_tag`
  inventory
- `solsys_code/tests/test_campaign_reconciler.py`,
  `solsys_code/tests/test_calendar_template.py`,
  `solsys_code/tests/test_campaign_attribution_views.py` — fixture and assertion conventions
- `/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/tom_observations/models.py` —
  `ObservationRecord`/`ObservationGroup` field definitions, confirming no existing
  `related_name` collision
- `solsys_code/migrations/0010_...py`, `0013_...py`, `0015_...py`, `0016_...py` — migration
  dependency and nullable-campaign precedent
- `docs/runbooks/telescope_runs_calendar.rst` — the three named sections (lines 294, 626,
  717-769), "owning"/"owns" occurrence grep
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`,
  `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` — cell/section inventory via
  direct JSON parse this session
- `.claude/skills/spike-findings-fomo_devel/references/allocation-handoff.md`,
  `.claude/skills/spike-findings-fomo_devel/references/observation-projector.md` —
  spike-validated requirements and patterns
- `.planning/phases/33-series-identity-reconciler-inversion/33-CONTEXT.md`,
  `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `.planning/config.json`

### Secondary (MEDIUM confidence)
None — no web research was performed; the entire domain was answerable from files already
in the repository.

### Tertiary (LOW confidence)
None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; every field/tag form has a direct in-repo
  precedent read this session
- Architecture: HIGH — every quoted code block was read from the actual file this session,
  not reconstructed from memory or CONTEXT.md's summary
- Pitfalls: HIGH for Pitfalls 1, 2, 5, 6 (each grounded in a specific quoted line range);
  MEDIUM for Pitfalls 3-4 (grounded in the current test file's assertions, but the "correct"
  new assertions are this research's own reasoned prediction of what D-01 implies, not a
  locked CONTEXT.md decision — flagged as Assumption A1/A2 where relevant)

**Research date:** 2026-09-03
**Valid until:** No external deadline — this is in-repo, version-pinned research (no
package versions to go stale). Re-verify only if a later phase (34/35) lands first and
changes `campaign_reconciler.py`/`models.py` out from under this phase's plan.
