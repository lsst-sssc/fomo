# Phase 33: Series Identity & Reconciler Inversion - Context

**Gathered:** 2026-09-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 33 gives `CalendarEventMeta` the two real link fields the observation projector
(Phase 34) will write — `observation_record` and `observation_group` — through a migration
that leaves every existing companion row's `run` / `is_verified` / `confirmed_by` /
`confirmed_at` history intact; and it turns the campaign reconciler from an owner into an
annotator: `reconcile_run()` keeps writing only its own `RUN:{pk}` / `RUN:{pk}:{date}`
events and never again adopts, re-keys, or detaches an event outside that namespace.
`CalendarEventMeta.run` changes meaning from "owns" to "attributed to", and the campaign
decoration a user sees on an attributed event (campaign label, run status, link back to
the run) is rendered from that link at display time — never written into the event's own
fields — so a base re-projection cannot erase it and clearing the link removes only the
decoration.

**In scope:** the two new link fields + migration + admin exposure; the reconciler
inversion (`_adopted_event_for_night()` / re-key path retired, skip-the-night rule,
`_may_write()` semantics restated); a single display-time decoration path (month-cell
marker + modal block + anchored link to the campaign table); `event_title()` no longer
embedding the campaign name; one shared unlink helper; the "attributed" wording change;
the paired notebooks (`reconcile_campaign_runs_demo.ipynb`, `campaign_lifecycle_demo.ipynb`)
and `docs/runbooks/telescope_runs_calendar.rst` (CLAUDE.md paired-docs rule).

**Out of scope:** the projector, trigger and sweep that *write* the new fields (Phase 34);
backfilling `observation_record` for the 156 existing URL-keyed sync events (Phase 34's
sweep does it); the allocation layer and the `RUN:{pk}:{date}` → `ALLOC:` cutover
(Phase 35); status vocabulary / status rings for attributed events (Phase 37); a run-detail
view (Phase 37's TALLY-01 surface); any automatic `run_status` derivation.

</domain>

<decisions>
## Implementation Decisions

### Reconciler write paths (ANNOT-01)

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
- **D-03: Past adopts stay in the RUN namespace.** Events already re-keyed into `RUN:{pk}:{date}` by
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

### Series-identity fields (PROJ-04)

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

### Decoration at display time (ANNOT-02)

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

### Unlink & orphan surfaces (criterion 4)

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
  summary shape; a supplementary count is optional).
- `related_name`s for the two new fields and the `__str__` of `CalendarEventMeta`.
- Whether the `verbose_name` rename and the two `AddField`s ship as one migration
  (`0017_…`) or two.

### Folded Todos

None — all five matched todos were reviewed and left where their own routing notes put
them (see Reviewed Todos below).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The decisions this phase executes
- `.planning/ROADMAP.md` §"Phase 33: Series Identity & Reconciler Inversion" — goal, the
  four success criteria, paired-docs list, and the milestone's locked constraints (the
  reconciler inversion comes first; base layer never writes `RUN:`).
- `.planning/REQUIREMENTS.md` — PROJ-04, ANNOT-01, ANNOT-02 (this phase); PROJ-05 and
  ANNOT-03 (Phase 34) and ALLOC-01..05 (Phase 35) for what this phase must not pre-empt.
- `.planning/PROJECT.md` §"Current Milestone: v2.4 Observation-First Calendar" — target
  features, landmines the spikes named, conventions carried.

### Spike findings (validated patterns and landmines — read before touching the reconciler)
- `.claude/skills/spike-findings-fomo_devel/SKILL.md` — the non-negotiable requirements
  list (attribution is a link, never a text write; series identity needs a real carrier;
  the adopt/re-key landmine).
- `.claude/skills/spike-findings-fomo_devel/references/allocation-handoff.md` — spike 003:
  the handoff rule D-01 mirrors, and "What to Avoid" (writing into a base event's
  title/description; letting the unchanged reconciler run against base events).
- `.claude/skills/spike-findings-fomo_devel/references/observation-projector.md` — spike
  002: item 7 ("do not create `CalendarEventMeta` rows from the base layer") and the
  measured constraint that `CalendarEventMeta` today has only `run` / `is_verified` /
  `confirmed_by/at`.
- `.planning/notes/observation-first-calendar-layering.md` — decisions D1–D5 (D1: one
  event per record, group contributes identity not geometry; D2: base owns, campaign
  annotates).

### The code and prior decisions this phase reverses or reuses
- `.planning/milestones/v2.2-phases/29-the-reconciler/29-CONTEXT.md` — D-02 (adopt classical
  nights) is the decision ANNOT-01 reverses; D-05/D-06 fix the command's summary shape and
  per-run failure posture, which stay.
- `.planning/milestones/v2.2-phases/28-operator-assisted-attribution/28-CONTEXT.md` — D-12
  (the `confirmed_by/at` audit fields D-16 now clears together with `run`) and D-15 (the
  attribution queue "done" state D-15 here keeps intact).
- `.planning/milestones/v2.3-phases/32-adapter-consolidation/32-CONTEXT.md` — why 32-01
  Tasks 1–2 (`adopt_event_into_run()`, nullable `campaign`, `source_identifier`) are kept
  and Task 3 / plans 32-02..04 are not executed.
- `solsys_code/campaign_reconciler.py` — `_adopted_event_for_night()`,
  `_reconcile_classical_nights()`, `_may_write()`, `writable_events()`,
  `_detach_stale_family_events()`, `event_title()`, `reconcile_run()`.
- `solsys_code/models.py` — `CalendarEventMeta` (lines ~11–74), `CampaignRunObservation`,
  the `pre_delete` cascade that calls `writable_events()`.
- `solsys_code/campaign_utils.py` `adopt_event_into_run()` — the link-only attribution
  bridge already written to the "attributed to" semantics.
- `solsys_code/campaign_attribution.py` `orphan_calendar_events()` /
  `candidates_for_event()`; `solsys_code/campaign_views.py` `_undo_confirmation()`
  (~line 1303) and the confirm path (~line 1175).
- `solsys_code/admin.py` — `CalendarEventMetaAdmin` and the `CampaignRunAdmin` inline
  (`save_formset` audit branches) that D-09/D-17 touch.
- `src/templates/tom_calendar/partials/calendar.html` (month cell, ~lines 215–262) and
  `src/templates/tom_calendar/partials/event_form.html` (~lines 95–175, the existing
  "Campaign run" block and the staff-only attribution hint).
- `solsys_code/templatetags/calendar_display_extras.py` and
  `solsys_code/templatetags/attribution_display_extras.py` — where display-time tags live.

### Paired docs (CLAUDE.md rule — part of the deliverable)
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`
- `docs/runbooks/telescope_runs_calendar.rst` — §"How do I attribute existing calendar
  events and observation records to a run?", §"How do I get every campaign run onto the
  calendar?", §"Why doesn't the calendar pop-up show a 'Campaign run' block?".

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `campaign_utils.adopt_event_into_run()` (32-01) already implements "set `meta.run`,
  touch nothing else, refuse if attributed elsewhere" — the exact attribution write D-14
  wants; `_link_event_to_run()` in the reconciler is its twin.
- `calendar_utils.insert_or_create_calendar_event()` / `preview_calendar_event_action()` —
  the no-churn contract the reconciler keeps using for its own `RUN:` events.
- `event_form.html`'s existing `{% with run=event.telescope_label_meta.run %}` block and
  its `is_publicly_visible` gate — the modal half of D-10 already exists; extend, don't
  rebuild.
- `views.py` `fomo_render_calendar` already prefetches `telescope_label_meta` (DISPLAY-09);
  add `telescope_label_meta__run__campaign` so the cell marker is N+1-free.
- `test_campaign_reconciler.py` (45 tests), `test_calendar_template.py`,
  `test_attribution_template.py`, `test_campaign_attribution_views.py` — the suites the
  inverted behaviour, the decoration tag and the unlink helper extend.

### Established Patterns
- Ownership by key namespace: `owned_events()` / `writable_events()` filter on the `RUN:`
  url prefix; everything outside it is never created, modified or deleted by the reconciler.
- Template-tag-driven display (`proposal_color`, `status_border_css`,
  `high_band_attribution_candidates`) — decoration follows the same shape.
- Migrations are additive and small (`0013_…calendar_event_meta_audit`,
  `0015_…nullable_campaign_and_source_identifier`); model docstrings carry the "why".
- Phase 29 D-05/D-06: `reconcile_campaign_runs` summary shape and per-run failure
  isolation are fixed; `sun_event()` `ValueError`s still propagate out of `reconcile_run()`.

### Integration Points
- New logic lives in peer modules under `solsys_code/` (`campaign_reconciler.py`,
  `campaign_utils.py`, `templatetags/`), never inside `campaign_views.py`, and never
  imports `solsys_code.views` / `solsys_code.ephem_utils` (SPICE kernel side effect;
  Phase 36 runs the reconciler unattended).
- Dev DB baseline for the notebook diff (2026-09-03): 240 events — 74 `RUN:`, 156
  `http…` URL-keyed, 9 blank-url, 0 `GEM:`; 85 companion rows, 75 with `run` set, 1 of
  them on a non-`RUN:` event; 159 `ObservationRecord`s, 9 `ObservationGroup`s.

</code_context>

<specifics>
## Specific Ideas

- The skip-the-night rule (D-01) is deliberately the same sentence as spike 003's handoff
  rule, so that when Phase 35 replaces `_reconcile_classical_nights()` with the allocation
  projector the behaviour a user sees for an attributed night does not change.
- The cell marker must survive re-projection of the event's title/description from scratch
  (success criterion 3) — the test for D-10 should rewrite `title`/`description` on a
  linked event and assert the marker and modal block are still rendered.

</specifics>

<deferred>
## Deferred Ideas

### Reviewed Todos (not folded)
- `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` — its own
  routing note sends it to Phase 35 (the allocation projector replaces
  `_reconcile_classical_nights()`); folding here would patch code Phase 35 retires.
- `2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` — resolved
  by v2.4's routing to Phase 34 (PROJ-01, ANNOT-03); `SOAR_QUEUE` already shipped in 32-01.
- `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` and
  `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` —
  attribution-UI items unrelated to link fields or reconciler writes.
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — style
  cleanup of `sync_lco_observation_calendar` helpers; that command is retired in Phase 34.

No scope-creep ideas surfaced — a run-detail view was offered as a link target and declined
as Phase 37's surface.

</deferred>

---

*Phase: 33-Series Identity & Reconciler Inversion*
*Context gathered: 2026-09-03*
