---
phase: 33-series-identity-reconciler-inversion
verified: 2026-09-04T18:09:09Z
status: human_needed
score: 49/50 must-haves verified
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 17
  total: 17
  not_honored: []
insufficient_spec_items:
  - truth: "No code added by this phase depends on the iteration order of the `observation_group` reverse manager: `CalendarEventMeta` gains no `Meta.ordering`, and no reader added here iterates `group.calendar_event_metas` expecting a stable order (PROJ-04 ordering edge)."
    reason: insufficient_spec
    tier: backstop
    observed: "CalendarEventMeta declares no `class Meta` at all (solsys_code/models.py:12-88), and repo-wide grep finds no production reader of `group.calendar_event_metas` — the only hit is a string assertion on an admin inline prefix in solsys_code/tests/test_admin.py:472."
    why_human: "Tagged `verification: backstop` by the edge probe — a non-inferable check. Absence-by-grep plus symbol presence is explicitly NOT sufficient evidence for a backstop truth; only a wired held-out/property-based test (e.g. one that shuffles insert order and asserts a stable outcome) or directly observed ordering behavior can confirm it. Abstaining rather than confidently false-passing."
human_verification:
  - test: "Open the month calendar (`/calendar/`) on a month containing at least one campaign-attributed all-day entry AND one attributed timed entry, across several different proposal fill colours. Look at the ⚑ campaign chip."
    expected: "The chip is legible against every proposal fill (it inherits the entry's own foreground via `color: currentColor`), does not compress or clip in the timed entry's flex row (`flex-shrink: 0`), and hovering it shows the campaign name as a tooltip."
    why_human: "Visual legibility and layout across dynamic, data-driven fill colours. Tests assert the CSS declarations and the tooltip attribute are present in the rendered HTML (solsys_code/tests/test_calendar_template.py MonthCellCampaignMarkerTest), but cannot judge whether the chip actually reads clearly on every fill."
  - test: "Click a campaign-attributed calendar entry to open its pop-up, then click the 'View campaign ↗' link in the 'Attributed campaign run' block."
    expected: "The campaign table page loads scrolled to that run's own row, and the row is visibly highlighted (the `tr:target` rule in src/templates/campaigns/campaignrun_table.html)."
    why_human: "The anchor `id=\"run-{pk}\"` and the `tr:target` CSS rule are both asserted present by tests (solsys_code/tests/test_campaign_views.py:641-661), but browser anchor-scroll plus `:target` highlight rendering is real-browser behavior no server-side test observes."
  - test: "Review the abstained backstop truth above: decide whether a held-out test pinning `observation_group` reverse-manager ordering independence is wanted before Phase 34's projector starts writing these links, or whether the absence evidence is accepted as-is."
    expected: "Either a held-out/property-based test is added (shuffle insertion order of several CalendarEventMeta rows sharing one ObservationGroup; assert the consuming code's outcome is unchanged), or the item is explicitly accepted."
    why_human: "`verification: backstop` — non-inferable by design; routing, not diagnosis. reason: insufficient_spec (NOT ordinary manual UAT)."
  - test: "Review the 11 judgment-tier prohibitions listed in the Prohibitions section of this report (LLM-judge verdicts are recorded there and are NON-AUTHORITATIVE), with particular attention to 33-05 P1 (notebook output shows `contact_person=''`/`contact_email=''` for demo runs in a pre-existing cell) and 33-05 P2 (the reconciler notebook writes to and deletes rows in the real developer database `src/fomo_db.sqlite3`)."
    expected: "Each prohibition is confirmed as still not violated, or the deviation is accepted."
    why_human: "unverified-prohibition — human review recommended. Judgment-tier prohibitions carry no wired enforcement test; a model verdict is never authoritative for a must-NOT."
---

# Phase 33: Series Identity & Reconciler Inversion — Verification Report

**Phase Goal:** `CalendarEventMeta` carries real series identity and attribution links, and the campaign reconciler annotates instead of owning — so the observation projector can land next phase without the campaign layer stealing its events.
**Verified:** 2026-09-04T18:09:09Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### ROADMAP Success Criteria (the contract)

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Calendar event linkable to `ObservationRecord`/`ObservationGroup` through real FKs on `CalendarEventMeta`; every existing companion row survives migration with `run`/`is_verified`/`confirmed_by` intact | ✓ VERIFIED | `solsys_code/models.py:54-68` — `observation_record = OneToOneField(ObservationRecord, SET_NULL, null, blank)`, `observation_group = ForeignKey(ObservationGroup, SET_NULL, null, blank)`. `solsys_code/migrations/0017_calendareventmeta_observation_links.py` — two `AddField` + one `AlterField`, cross-app dependency on `tom_observations.0016`. Migration proved non-destructive by a real `MigrationExecutor` round trip: `TestCalendarEventMetaObservationLinksMigration` seeds two pre-0017 rows (one with audit fields populated, one NULL), migrates to 0017, asserts all four pre-existing values byte-identical and both new columns NULL. Ran `python manage.py test solsys_code.tests.test_calendar_event_meta_links` → 8 tests OK. `makemigrations --check` → "No changes detected". |
| 2 | `reconcile_campaign_runs` over the existing dev DB no longer adopts, re-keys, or detaches any event outside the reconciler's own `RUN:` namespace | ✓ VERIFIED | Every write path is namespace-keyed: `_reconcile_container` and `_reconcile_classical_nights` look up `CalendarEvent.objects.filter(url=run_container_url/run_night_url)` only; `_detach_stale_family_events` operates on `owned_events(run)` (url `startswith` this run's `RUN:` prefix). The adopt/re-key helper `_adopted_event_for_night()` is deleted outright (`grep _adopted_event_for_night solsys_code/campaign_reconciler.py` → no hits). Fixture proof: `TestAttributedEventsSurviveReconcile` snapshots 11 fields (url/title/description/start/end/telescope/instrument + run_id/is_verified/confirmed_by/confirmed_at) and asserts byte-identity across `reconcile_run()` for blank-url and facility-URL-keyed attributed events, plus idempotency on a second call. Real-DB proof: `reconcile_campaign_runs_demo.ipynb` cells 15-16 (executed output committed) snapshot **166** non-`RUN:` events, preview via `reconcile_run(dry_run=True)` per run (touchable url set = 90, all inside `RUN:`), run the real sweep, and report `Differences found: 0`. |
| 3 | A user sees campaign decoration on an attributed event, and it survives a from-scratch rewrite of the event's title and description | ✓ VERIFIED | `campaign_decoration()` (`solsys_code/templatetags/calendar_display_extras.py:434-483`) is a read-only `simple_tag` reading `CalendarEventMeta.run` at request time; rendered by `event_form.html:136-163` ("Attributed campaign run" block + `#run-{pk}` anchored campaign link) and `calendar.html:253,282` (month-cell ⚑ chip). Behavioral test `test_decoration_survives_from_scratch_rewrite_of_title_and_description` rewrites `title` and `description`, re-GETs both the modal and the month view, and asserts the decoration is still rendered. Notebook cell 28 shows the same against a real request with committed output. Nothing about the decoration is stored on `CalendarEvent` (no `.save()`/`.update()`/`.create()` anywhere in `calendar_display_extras.py`). Visual appearance routed to human verification. |
| 4 | Clearing `CalendarEventMeta.run` removes only the decoration; the event is untouched and nothing is deleted | ✓ VERIFIED | `unlink_event_from_run()` (`solsys_code/campaign_utils.py:860-910`) performs one conditional `CalendarEventMeta.objects.filter(run_id=run_pk, **event_filter).update(run=None, confirmed_by=None, confirmed_at=None)` — no delete, no `CalendarEvent` write, no `is_verified`/`observation_*` write. `TestUnlinkEventFromRun` (9 tests) asserts `is_verified` untouched and both `CalendarEvent`/`CalendarEventMeta` counts unchanged. `test_detach_clears_audit_fields_leaves_event_and_verification_flag_untouched` snapshots 7 `CalendarEvent` fields across a real detach. Notebook cell 30: counts 256→256, title/description preserved, `Modal shows "Attributed campaign run" after unlink: False`. |

### Plan Must-Haves (merged; roadmap-duplicating truths folded into the table above)

**Plan 33-01 — reconciler inversion + display-time decoration (12 truths, all ✓ VERIFIED)**

| Truth | Evidence |
|-------|----------|
| Attributed non-`RUN:` night mints no `RUN:{pk}:{date}` event; reported in `skipped_nights` | `_reconcile_classical_nights:372-374` — `if existing is None and night in attributed_nights: totals['skipped_nights'] += 1; continue`. `TestAttributedNightSkip` (4 tests). Notebook cell 18 real output: `skipped_nights=1`, `'RUN:59:2026-09-02' present: False`. |
| Skip matches site-local observing night, never naive UTC | `_attributed_nights:328` — `meta.event.start_time.astimezone(site_zone).date()`. Test `test_skip_matches_on_site_local_night_not_naive_utc_date`. |
| Skip matches ANY url outside `RUN:` — blank and facility-URL alike | `.exclude(event__url__startswith=RUN_URL_NAMESPACE)` with no blank-url restriction. Tests `test_facility_url_keyed_attributed_event_skips_its_night`, `test_blank_url_attributed_event_survives_reconcile`. |
| A `RUN:{pk}` event whose companion row points at a DIFFERENT run stays blocked; `meta.run` never reset | `_may_write:238-244` — exact-match on `meta.run_id` when set. `test_reconcile_reports_blocked_for_a_night_attributed_to_a_different_run`. |
| Modal shows campaign name, telescope/instrument, window, run-status from `CalendarEventMeta.run` at request time | `event_form.html:136-163`; `EventModalCampaignRunLinkTest::test_approved_run_shows_attributed_label_and_anchored_campaign_link`. |
| Modal campaign link targets `campaigns:table` with `#run-{pk}` | `calendar_display_extras.py:473` — `f"{reverse('campaigns:table', args=[run.campaign_id])}#run-{run.pk}"`. Same test asserts the anchored href. |
| No companion row / unset run → no decoration, no raise | `try: meta = event.telescope_label_meta except ObjectDoesNotExist: return None`; `if run is None ... return None`. Tests `test_no_companion_row_at_all_renders_200_with_no_exception`, `test_null_run_companion_row_renders_200_with_no_run_block`. |
| Run with `campaign=None` → decoration with no table link, 200, never `NoReverseMatch` | `if run.campaign_id is not None:` guard before `reverse()`; `table_url=None` otherwise. Test `test_no_campaign_run_renders_200_with_telescope_instrument_and_no_campaign_link`. |
| Not-publicly-visible run → no decoration for staff or anonymous | `if run is None or not run.is_publicly_visible: return None`. Tests `test_pending_run_shows_no_run_block_to_anonymous_visitor` + `..._to_staff_visitor`. |
| `RUN:` titles no longer carry a campaign-name prefix; `RUN_STATUS_CALENDAR_PREFIX` still leads | `event_title():181-195` — base is `run.telescope_instrument` (+ window), status prefix prepended; no campaign branch. `TestEventTitleGuard`, `TestEventTitleNullCampaignGuard`, `StatusBorderCssTest` all pass. |
| Attributed-night set computed with ONE query before the per-night loop | `attributed_nights = _attributed_nights(run, site_zone)` at line 364, immediately before `for i in range(n_nights)` at 365; the helper issues a single `CalendarEventMeta` queryset with `select_related('event')`. |
| `skipped_nights` seeded to 0 in the `totals` dict literal | Line 361 — `totals = {'created': 0, 'updated': 0, 'unchanged': 0, 'blocked': 0, 'skipped_nights': 0}`. |

**Plan 33-02 — month-cell marker and campaign-table anchor (9 truths, all ✓ VERIFIED)**

| Truth | Evidence |
|-------|----------|
| Month-cell entry for a publicly-visible attributed run shows a compact marker with the campaign name as tooltip | `calendar.html:253` (all-day) and `:282` (timed) — `<span class="cal-campaign-chip" title="{{ campaign_deco.campaign_name }}">&#9873;</span>`. `test_month_view_shows_campaign_chip_and_name_tooltip`. |
| Marker consumes none of the truncation budget; `truncatechars:18`/`:16` still over the event's own title only; campaign name only in the escaped tooltip | `{{ event.title\|truncatechars:18 }}` / `:16` are unchanged and outside the chip span. `test_chip_does_not_consume_title_truncation_budget`. |
| `campaign=None` → marker with no table link, month view 200 | `test_no_campaign_run_renders_marker_and_no_table_href`. |
| Not-publicly-visible run → no marker for staff or anonymous | `test_pending_review_run_shows_no_marker_for_staff_and_anonymous`. |
| No per-event query for run/campaign — event queryset prefetches the companion row with run+campaign selected | `solsys_code/views.py:118-122` — `Prefetch('telescope_label_meta', queryset=CalendarEventMeta.objects.select_related('run__campaign'))`. `test_query_count_does_not_grow_with_number_of_attributed_events` (count-comparison form, 1 vs N). |
| Campaign-table link lands on the run's own row; every row carries `id="run-{pk}"` for staff and non-staff; targeted row highlighted | `campaign_tables.py:119` — `row_attrs = {'id': _campaign_run_row_id}` on `CampaignRunTable.Meta`, inherited by `ApprovalQueueTable.Meta`. `campaignrun_table.html:9` — `tr:target` highlight rule. `test_staff_get_contains_run_row_id`, `test_anonymous_get_contains_run_row_id` (dict-row branch). Visual highlight routed to human verification. |
| Month cell never renders `contact_person`, `contact_email`, or `CampaignRun.source` | `campaign_decoration()` returns an explicit 7-key dict with none of those; `test_pii_fields_never_render_on_month_view` asserts absence of both contact values and the source value. |
| Marker inherits `color: currentColor` and does not compress (`flex-shrink: 0`) | `calendar.html:113-114` inside the `.cal-campaign-chip` rule. Legibility routed to human verification. |
| A row whose pk cannot be resolved renders with NO `id` attribute (never `id="run-None"`) | `_campaign_run_row_id:78-80` returns `None` when `Accessor('pk').resolve(record, quiet=True)` is `None`. `test_..._returns_none(_campaign_run_row_id({}))`; both view tests assert `id="run-None"` is absent. |

**Plan 33-03 — real FK carrier fields (8 truths: 7 ✓ VERIFIED, 1 ⚠️ insufficient_spec)**

| Truth | Status | Evidence |
|-------|--------|----------|
| `observation_record` is one-to-one — a second row on the same record raises `IntegrityError`; any number may be NULL | ✓ VERIFIED | `test_second_row_pointing_at_same_observation_record_raises_integrity_error`, `test_two_rows_may_both_leave_observation_record_null`, `test_several_rows_may_point_at_the_same_observation_group`. |
| No code added depends on `observation_group` reverse-manager iteration order | ⚠️ insufficient_spec (abstained) | `verification: backstop`. Observed: no `class Meta` on `CalendarEventMeta`; no production reader of `group.calendar_event_metas`. Absence-by-grep is not explicit evidence for a backstop truth — routed to human (see frontmatter). |
| Deleting an `ObservationRecord`/`ObservationGroup` clears only that link; row, `run`/`is_verified`/`confirmed_by`/`confirmed_at` and the `CalendarEvent` survive | ✓ VERIFIED | `SET_NULL` on both FKs. `test_deleting_observation_record_clears_only_that_link` and `test_deleting_observation_group_clears_only_that_link` assert the sibling link, all four audit/attribution values, and `CalendarEvent` existence. |
| Both new fields read-only on `CalendarEventMetaAdmin` and the `CampaignRunAdmin` inline; a submitted value is not bound | ✓ VERIFIED | `admin.py:100` (inline) and `admin.py:306` (standalone) both list them in `readonly_fields`. `test_admin.py:1186-1230` — asserts they appear in `get_readonly_fields` for both surfaces, that `name="observation_record"`/`name="observation_group"` never appear in the rendered form, and that POSTing values leaves both `None`. |
| `CalendarEventMeta.run` presents as attribution, not ownership | ✓ VERIFIED | `models.py:45` — `verbose_name='Attributed campaign run'` (also restated in migration 0017's `AlterField`). Model docstring: "A row whose `run` is unset means 'not attributed to any campaign run' -- never 'do not touch'." Inline/admin docstrings restated (`admin.py:74-77`). |
| This phase writes no `observation_record`/`observation_group` value onto any existing row; additive schema only | ✓ VERIFIED | Migration 0017 contains only `AddField`×2 + `AlterField`×1 — no `RunPython`, no `RunSQL`. `test_both_new_link_columns_are_null_on_every_migrated_row`. |
| PROJ-04's "shared title stem" clause deferred to Phase 34 — this plan's production diff adds no line writing `.title` | ✓ VERIFIED | `git show fe4d1ca d8b5676 -- solsys_code/models.py solsys_code/admin.py \| grep '^+' \| grep '\.title\|title='` → no output. REQUIREMENTS.md:18 records the scope split. |
| Migration test restores the DB to migration head in `tearDown` regardless of assertion outcome | ✓ VERIFIED | `test_calendar_event_meta_links.py:224-228` — `tearDown` rebuilds the graph and migrates to `leaf_nodes()`. Subsequent 379 tests in the same runs passed with no stranded-schema failures. |

**Plan 33-04 — one shared unlink helper (8 truths, all ✓ VERIFIED)**

| Truth | Evidence |
|-------|----------|
| One shared helper `unlink_event_from_run()` in `campaign_utils.py` is the single writer that clears an attribution; clears `run`+`confirmed_by`+`confirmed_at` together and returns the changed count | `campaign_utils.py:908-910`. Repo-wide grep for any other `run=None`/`run_id=None` write outside comments: only this one line. Admin branch 2 keeps an in-memory nulling of the same three values (documented deviation — see Notes below), not a second bulk writer. |
| Clearing never touches `is_verified` | `test_verification_flag_is_never_touched`; `test_detach_clears_audit_fields_leaves_event_and_verification_flag_untouched` asserts `is_verified` still `False` after detach. |
| The undo view still clears only when the row currently points at the named run, and gates its dismissal write on the changed count | `campaign_views.py:1330` — `changed_count = unlink_event_from_run(orphan_pk, run_pk)`, followed by `if changed_count:` before the `CalendarEventDismissal.get_or_create`. `test_wrong_run_returns_zero_and_changes_nothing`. 113 attribution-view tests pass. |
| The reconciler's detach step now clears `confirmed_by`/`confirmed_at` alongside `run` | `_detach_stale_family_events:450-453` routes through the helper. `test_detach_clears_audit_fields_leaves_event_and_verification_flag_untouched` asserts all three are `None` after a window shrink. |
| Detach still only reaches events inside the reconciler's own `RUN:` namespace whose row points at the run being reconciled | `owned_events(run).exclude(url__in=active_urls)` + the helper's `run_id=run_pk` filter term. `test_detach_never_clears_a_foreign_attribution_in_the_same_namespace`, `test_reconcile_never_detaches_an_event_attributed_to_a_different_run`. |
| An event with `observation_record` set but no `run` is still offered by the event attribution queue (D-15) | New `TestOrphanQuerysets` test in `test_campaign_attribution.py` (commit `7e1d7d8`); notebook cell 32 real output: `orphan_event.pk=333 in orphan_calendar_events(): True` with `run=None` and both observation links set. |
| Called with no run (`None`, or a run whose pk is `None`) the helper returns 0 and writes nothing — never a `run_id=None` filter | `campaign_utils.py:892-899` — `run_pk = getattr(run, 'pk', run); if not run_pk: return 0`, placed before any `CalendarEventMeta.objects` reference. `test_null_run_returns_zero_and_never_touches_an_already_unlinked_rows_audit_fields`, `test_campaign_run_shaped_argument_with_no_pk_behaves_like_none`. |
| Clearing through the admin standalone form leaves the STORED row with `confirmed_by`/`confirmed_at` both `None` | `admin.py:394-411` — in-memory nulling before `super().save_model()` delegates to `obj.save()`. `test_clearing_the_run_clears_the_audit_fields` extended to re-fetch from the database and assert `is_verified` + object-count invariance. |

**Plan 33-05 — paired docs (9 truths, all ✓ VERIFIED)**

| Truth | Evidence |
|-------|----------|
| Reconciler notebook shows a real sweep leaving every non-`RUN:` event's url/title/attribution unchanged; the before/after diff is empty | Cell 16 committed output: `Compared 166 non-RUN:-namespaced CalendarEvent row(s)... Differences found: 0`, with a hard `assert not diff`. |
| The sweep is previewed first via `reconcile_run(run, dry_run=True)` for every run, showing the touchable url set holds nothing outside the namespace | Cell 15 committed output: per-run table for 53 runs, then `Touchable url set size: 90; every member starts with RUN_URL_NAMESPACE ('RUN:'): True`, and an explicit listing of attributed events whose url is outside the namespace with `Any of those urls in the touchable set: False`. |
| Per-run detail comes from `ReconcileResult`, not from parsing `reconcile_campaign_runs` stdout | Cell 15 prints the aggregate stdout line separately under an explicit note: "the per-run detail and the url-namespace assertions above come from reconcile_run()s returned ReconcileResult, not from parsing this line". |
| Notebook demonstrates the skip rule | Cell 18 committed output: attributed blank-url event url unchanged (`''`), `'RUN:59:2026-09-02' present: False`, `skipped_nights: 1`. |
| Lifecycle notebook shows the decoration rendered from the link on a real request, and still rendered after a from-scratch title/description rewrite | Cells 26 and 28 committed output — label, campaign name, telescope/instrument, `#run-63` anchor; after rewrite, modal block and `cal-campaign-chip` both still `True`. |
| Lifecycle notebook shows clearing the attribution removes the decoration and deletes nothing | Cell 30 committed output — counts 256→256, fields preserved, decoration absent. |
| Runbook describes `CalendarEventMeta.run` as an attribution throughout its three affected sections | `docs/runbooks/telescope_runs_calendar.rst:381-391`, `:460-468`, `:753-823` — "Attributing changes nothing about the entry itself", "Attributed campaign run" block section, detach section noting audit stamps cleared with the link. |
| Runbook tells operators what the reconciler now does with an already-attributed night, and that the one-time title change is expected | `:725-742` — "**A night that already has an entry attributed to this run gets no reconciler entry at all.**" and "**One-time title change.** ... that one-time title change is expected, not a fault." |
| Both notebooks committed with executed output | JSON inspection: reconciler demo 9/9 code cells carry outputs; lifecycle demo 18/18 code cells carry outputs. |

**Score:** 49/50 truths verified (0 present-but-behavior-unverified; 1 abstained `insufficient_spec`)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/campaign_reconciler.py` | Skip rule, `skipped_nights`, no adopt/re-key, detach via shared helper | ✓ VERIFIED | 498 lines; `_adopted_event_for_night` absent; imported by `campaign_utils`, `campaign_views`, management commands |
| `solsys_code/templatetags/calendar_display_extras.py` | `campaign_decoration()` read-only simple_tag | ✓ VERIFIED | 483 lines; no write call of any kind; used by both calendar templates |
| `src/templates/tom_calendar/partials/event_form.html` | "Attributed campaign run" block, anchored link | ✓ VERIFIED | 202 lines; renders `deco.*` only |
| `src/templates/tom_calendar/partials/calendar.html` | Month-cell chip + CSS | ✓ VERIFIED | 359 lines; chip in both all-day and timed branches |
| `solsys_code/views.py` | Prefetch with `select_related('run__campaign')` | ✓ VERIFIED | Wired into `fomo_render_calendar` |
| `solsys_code/campaign_tables.py` | `_campaign_run_row_id` + `row_attrs` | ✓ VERIFIED | Wired via `CampaignRunTable.Meta`, inherited by `ApprovalQueueTable` |
| `src/templates/campaigns/campaignrun_table.html` | `tr:target` highlight | ✓ VERIFIED | 64 lines |
| `solsys_code/models.py` | Two new FKs, attribution verbose_name | ✓ VERIFIED | 620 lines |
| `solsys_code/migrations/0017_calendareventmeta_observation_links.py` | Additive schema only | ✓ VERIFIED | 30 lines; no RunPython/RunSQL; `makemigrations --check` clean |
| `solsys_code/admin.py` | Read-only exposure of both links, clear branch | ✓ VERIFIED | 426 lines |
| `solsys_code/campaign_utils.py` | `unlink_event_from_run()` | ✓ VERIFIED | 1021 lines; called by `campaign_views` and `campaign_reconciler` |
| `solsys_code/campaign_views.py` | Undo view routed through helper | ✓ VERIFIED | Import at line 62, call at 1330 |
| `solsys_code/tests/*` (6 files) | Coverage per plan | ✓ VERIFIED | 1401/976/1230/1112/247 lines etc.; 379 targeted tests executed, all OK |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | Executed D-04 diff + skip rule | ✓ VERIFIED | 20 cells, all 9 code cells carry output |
| `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` | Executed criteria 3 & 4 + link fields | ✓ VERIFIED | 38 cells, all 18 code cells carry output |
| `docs/runbooks/telescope_runs_calendar.rst` | Attribution wording, skip rule, title change | ✓ VERIFIED | 991 lines |

CLAUDE.md paired-docs rule: satisfied. Both mapped notebooks and the `docs/runbooks/` page were updated in-phase (33-05 as planned work; 33-04 added the detach-behaviour sentence as a documented Rule-2 deviation).

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `CalendarEventMeta.run` | `_reconcile_classical_nights()` skip branch | `_attributed_nights()` | ✓ WIRED | Line 364 call, line 372 skip |
| `CalendarEventMeta.run` | `event_form.html` | `campaign_decoration()` | ✓ WIRED | `{% campaign_decoration event as deco %}` line 136 |
| `CalendarEventMeta.run` | `calendar.html` month cell | `campaign_decoration()` | ✓ WIRED | Lines 240 / 265 |
| `_may_write()` | every reconciler write path | first condition | ✓ WIRED | Container line 285, classical line 375 |
| `views.fomo_render_calendar` queryset | `campaign_decoration()` | `Prefetch(... select_related('run__campaign'))` | ✓ WIRED | views.py:118-122; N+1 test passes |
| `campaign_decoration().table_url` | `CampaignRunTable` row `id="run-{pk}"` | anchored href | ✓ WIRED | Both ends asserted by tests |
| `CalendarEventMeta.observation_record` | `tom_observations.ObservationRecord` | OneToOneField SET_NULL | ✓ WIRED | Model + migration + DB-level IntegrityError test |
| `CalendarEventMeta.observation_group` | `tom_observations.ObservationGroup` | ForeignKey SET_NULL | ✓ WIRED | Model + migration + multi-row test |
| Migration 0017 | `tom_observations` migration | `dependencies` entry | ✓ WIRED | `('tom_observations', '0016_alter_facility_options')` |
| `unlink_event_from_run()` | `campaign_views._undo_confirmation()` | direct call, changed-count gate | ✓ WIRED | campaign_views.py:1330 |
| `unlink_event_from_run()` | `campaign_reconciler._detach_stale_family_events()` | local import + call | ✓ WIRED | campaign_reconciler.py:450-453 |
| `unlink_event_from_run()` | `admin.CalendarEventMetaAdmin.save_model()` clear branch | **shared semantics, not a call** | ⚠️ PARTIAL (accepted) | The plan's key_link wording said the admin's in-memory mutation "becomes a call into the shared helper". The implementation deliberately keeps the in-memory nulling and only *documents* the helper as the shared definition — because `super().save_model()`'s `obj.save()` would re-persist stale values over a helper-only bulk clear (33-REVIEWS.md Agreed Concern 3). The plan's own *truth* for this case specifies exactly the implemented behavior ("the in-memory instance is nulled before `super().save_model()` delegates"), and it is proven by a re-fetch-from-database test. Not a gap — the key_link wording is superseded by its sibling truth. |

### Data-Flow Trace (Level 4)

| Artifact | Data value | Source | Produces real data | Status |
|----------|-----------|--------|--------------------|--------|
| `event_form.html` decoration block | `deco.campaign_name`, `telescope_instrument`, `window_*`, `run_status_display`, `table_url` | `campaign_decoration()` → `event.telescope_label_meta.run` → `CampaignRun` row | Yes | ✓ FLOWING |
| `calendar.html` chip tooltip | `campaign_deco.campaign_name` | same tag, prefetched queryset | Yes | ✓ FLOWING |
| `CampaignRunTable` row id | `Accessor('pk').resolve(record)` | model instance (staff) or `.values()` dict (anonymous) | Yes, both branches tested | ✓ FLOWING |
| `_attributed_nights()` | site-local night set | `CalendarEventMeta` queryset joined to `CalendarEvent.start_time` | Yes | ✓ FLOWING |
| `ReconcileResult.skipped_nights` | counter | incremented in the real skip branch | Yes — notebook shows `skipped_nights=1` against the dev DB | ✓ FLOWING |

No hardcoded literals, static fallbacks, or hollow props found on any decoration path.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Observation-link fields + migration 0017 non-destructive | `python manage.py test solsys_code.tests.test_calendar_event_meta_links` | Ran 8 tests, OK (1.6s) | ✓ PASS |
| Reconciler skip rule, decoration, unlink helper, admin read-only | `python manage.py test solsys_code.tests.test_campaign_reconciler test_calendar_template test_campaign_attribution_views test_admin test_calendar_display_extras` | Ran 266 tests, OK (108s) | ✓ PASS |
| Campaign-table anchors, attribution queue, title guards, reconcile command | `python manage.py test solsys_code.tests.test_campaign_views test_campaign_attribution test_null_campaign_guards test_write_and_reconcile test_reconcile_campaign_runs` | Ran 113 tests, OK (45s) | ✓ PASS |
| No missing migrations | `python manage.py makemigrations --check --dry-run` | "No changes detected" | ✓ PASS |
| Lint gate (D-07) | `pre-commit run ruff --all-files` | Passed | ✓ PASS |
| Format gate (D-07) | `pre-commit run ruff-format --all-files` | Passed | ✓ PASS |
| Notebooks carry executed output | JSON inspection of both `pre_executed/` notebooks | 9/9 and 18/18 code cells have outputs | ✓ PASS |
| Adopt/re-key helper actually deleted | `grep _adopted_event_for_night solsys_code/campaign_reconciler.py` | no hits | ✓ PASS |
| Chip legibility / anchor-scroll highlight in a browser | n/a | — | ? SKIP → human verification |

379 phase-relevant tests executed by this verification, all passing. (The orchestrator's 980 + 40 full-suite runs are consistent with these results but were not the basis for any verdict here.)

### Probe Execution

N/A — no `scripts/*/tests/probe-*.sh` exist in this repository and no plan declares a probe.

### Test Quality Audit

| Test file | Linked req | Active | Skipped | Circular | Assertion level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `test_calendar_event_meta_links.py` | PROJ-04 | 8 | 0 | No | Value (field-by-field equality across a real `MigrationExecutor` round trip) | ✓ Sufficient |
| `test_campaign_reconciler.py` | ANNOT-01 | 70 | 0 | No | Value/behavioral (11-field byte-identity snapshots, counter values) | ✓ Sufficient |
| `test_calendar_template.py` | ANNOT-02 | 60 | 0 | No | Behavioral (real HTTP GETs, rewrite-then-re-render, query-count comparison) | ✓ Sufficient |
| `test_campaign_attribution_views.py` | ANNOT-01 | 9 (unlink class) | 0 | No | Value (field-level + object-count invariance) | ✓ Sufficient |
| `test_admin.py` | PROJ-04 | 8 (links class) | 0 | No | Behavioral (POST-does-not-bind, re-fetch from DB) | ✓ Sufficient |
| `test_campaign_views.py` | ANNOT-02 | 3 (row-id class) | 0 | No | Value (exact `id="run-{pk}"` string, negative `id="run-None"`) | ✓ Sufficient |

**Disabled tests on requirements:** 0. **Circular patterns detected:** 0 (no fixture-generating script imports the system under test; expected values are hand-written literals or independently-derived snapshots). **Insufficient assertions:** 0.

One note on provenance: the roadmap-criterion-2 proof has two independent legs — hand-written fixture tests (expected values authored by hand) *and* a real-developer-database before/after diff whose "expected" is the pre-sweep state captured before any write. Neither leg derives its expected values from the reconciler's own output.

### Requirements Coverage

| Requirement | Source plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PROJ-04 | 33-03, 33-05 | Series identity carried by real FKs on `CalendarEventMeta` (`observation_record`, `observation_group`) | ✓ SATISFIED (carrier clause) | Both FKs, migration 0017, read-only admin, 8 passing tests. The "shared title stem" clause is an explicitly recorded split to Phase 34 (REQUIREMENTS.md:18), and this phase provably adds no `.title` write. |
| ANNOT-01 | 33-01, 33-04, 33-05 | `CalendarEventMeta.run` means "attributed to"; `reconcile_run()` no longer adopts, re-keys, or detaches an attributed event | ✓ SATISFIED | Skip rule, deleted adopt helper, namespace-scoped detach with `run=run` filter, byte-identity tests, real-DB empty diff, attribution wording across model/admin/reconciler/runbook. |
| ANNOT-02 | 33-01, 33-02, 33-05 | Campaign decoration rendered from the link at display time, never written into the event's fields | ✓ SATISFIED | `campaign_decoration()` read-only tag, modal block, month-cell chip, N+1-free prefetch, survives-rewrite test and notebook cell. |

**Orphaned requirements:** none. `grep "Phase 33" .planning/REQUIREMENTS.md` maps exactly PROJ-04, ANNOT-01, ANNOT-02 to this phase; all three are claimed by plan frontmatter.

### Decision Coverage

All trackable CONTEXT.md decisions are honored by shipped artifacts — **17/17 honored, 0 not honored** (`gsd-tools query check.decision-coverage-verify`). Non-blocking gate; recorded for drift tracking.

### Prohibitions (judgment tier — NON-AUTHORITATIVE LLM-judge verdicts, human review recommended)

All 11 prohibitions across the five plans are `verification: judgment` with `status: resolved`. None has a wired negative-enforcement test, so none can be green: each is recorded below as a model judgement and flagged for human confirmation.

| Plan | Prohibition (abbreviated) | LLM-judge verdict | Basis |
|------|---------------------------|-------------------|-------|
| 33-01 | `reconcile_run()` must not create/modify/delete any event outside the `RUN:` namespace | Holds | All writes keyed via `run_container_url`/`run_night_url`; detach via `owned_events()`; adopt helper deleted; real-DB diff over 166 events empty |
| 33-01 | Decoration must not be written into any `CalendarEvent` field; no tag may call a write method | Holds | No `.save(`/`.update(`/`.create(`/`get_or_create`/`.delete(` anywhere in `calendar_display_extras.py` |
| 33-01 | Decoration must not render contact fields or `CampaignRun.source` | Holds | Tag returns a fixed 7-key dict; none of the forbidden fields appear |
| 33-02 | Marker must not consume title-truncation budget nor come from writing campaign text into `title` | Holds | Chip span sits outside the `truncatechars` filter; `event_title()` has no campaign branch |
| 33-02 | Month cell must not render contact fields, `source`, or any field of a non-public run | Holds | `test_pii_fields_never_render_on_month_view`; `is_publicly_visible` gate in the tag |
| 33-03 | Migration 0017 must contain no data transformation | Holds | File contains only `AddField`×2 + `AlterField`×1 |
| 33-03 | The two link fields must not be writable through any staff form | Holds | `readonly_fields` on both admin surfaces; POST-does-not-bind tests |
| 33-04 | Unlinking must not delete an event or the companion row, nor modify any `CalendarEvent` field | Holds | Single `.update()` of three columns; count-invariance and 7-field snapshot tests |
| 33-04 | The helper must not write `is_verified`, `observation_record`, or `observation_group` | Holds | Those columns are absent from the `.update()` call |
| 33-04 | The helper must not clear a row pointing at a different run | Holds | `filter(run_id=run_pk, ...)`; two foreign-attribution tests |
| 33-05 | No notebook cell may print or store a run's `contact_person`, `contact_email`, or `source` into committed output | Holds, with a flag | The reconciler demo has zero occurrences. `campaign_lifecycle_demo.ipynb` cell 36 (pre-existing, not added by this phase) prints `contact_person=''  contact_email=''` for demo runs — empty strings demonstrating the PII gate, no value disclosed. Worth a human eyeball. |
| 33-05 | No notebook cell may leave demo rows behind outside its own demo-scoped reset, nor delete rows it did not create | Holds, with a flag | Cell 18 deletes the `RUN:59:2026-09-02` event its own earlier sweep created in the same notebook run (stated in the cell comment and in its output). These notebooks write to the real developer DB (`src/fomo_db.sqlite3`), so this is worth a human eyeball. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/views.py` | 298 | `# XXX Could replace this by a creation of the missing Observatory` | ℹ️ Info | Pre-existing since commit `a8613bc` (2025-07-23), in `Ephemeris`/observatory code untouched by this phase — this phase's only `views.py` change is the `Prefetch`. Not a phase-33 debt marker. |
| various | many | Literal `TBD` | ℹ️ Info | Domain vocabulary, not a debt marker: `TBD` is this project's term for a to-be-determined observing window (`_skip_reason()` returns `'TBD window'`; `CampaignRun.original_obs_date_raw` is "TBD rows only"). No `FIXME`, no `HACK`, no `PLACEHOLDER` comment, and no unreferenced debt marker introduced by this phase. |

No stubs, empty implementations, static returns, or hollow props found on any path this phase touched.

### Human Verification Required

#### 1. Month-cell campaign chip — visual legibility and layout

**Test:** Open `/calendar/` on a month with at least one attributed all-day entry and one attributed timed entry, across several proposal fill colours. Inspect the ⚑ chip.
**Expected:** Legible on every fill (inherits `color: currentColor`), no compression/clipping in the timed entry's flex row (`flex-shrink: 0`), campaign name shows as a tooltip on hover.
**Why human:** Visual legibility against dynamic, data-driven fill colours. Tests assert the CSS declarations and the tooltip attribute exist in the rendered HTML; they cannot judge whether it actually reads clearly.

#### 2. Campaign-table anchor landing and row highlight

**Test:** Open an attributed entry's pop-up, click "View campaign ↗".
**Expected:** The campaign table loads scrolled to that run's row, and the row is visibly highlighted.
**Why human:** Anchor scroll plus `tr:target` highlight is real-browser rendering; both ends (the `id="run-{pk}"` attribute and the CSS rule) are asserted present server-side, the effect is not.

#### 3. Backstop abstention — `observation_group` reverse-manager ordering (reason: `insufficient_spec`)

**Test:** Decide whether a held-out test pinning ordering independence is wanted before Phase 34's projector starts writing these links, or accept the item as-is.
**Expected:** Either a held-out/property-based test (shuffle insertion order of several rows sharing one group; assert the consuming outcome is unchanged) is added, or the item is explicitly accepted.
**Why human:** `verification: backstop` — non-inferable by design. Observed evidence (no `Meta.ordering`; no production reader of `group.calendar_event_metas`) is absence-by-grep, which the honest-verifier protocol explicitly excludes as sufficient. This is an evidence gap, not a UX step.

#### 4. Judgment-tier prohibition review (unverified-prohibition — human review recommended)

**Test:** Review the 11 prohibitions in the Prohibitions table above, especially the two flagged 33-05 items (notebook output showing empty contact fields; notebook writes/deletes against the real developer database).
**Expected:** Each confirmed still not violated, or the deviation accepted.
**Why human:** Judgment-tier prohibitions have no wired enforcement test; a model verdict on a must-NOT is never authoritative.

### Gaps Summary

**No gaps.** Every ROADMAP success criterion is observably true in the codebase, backed by executed tests and — for criterion 2 — by a committed real-developer-database before/after diff over 166 events with zero differences. All three requirement IDs (PROJ-04 carrier clause, ANNOT-01, ANNOT-02) are satisfied, with PROJ-04's title-stem clause explicitly and traceably split to Phase 34. All 17 CONTEXT decisions are honored. Lint, format and migration gates are clean. The paired-docs rule is satisfied for both notebooks and the runbook.

Three things keep this from `passed` rather than indicating broken work:

1. One truth in plan 33-03 is tagged `verification: backstop` (non-inferable). Its supporting evidence is an absence check, which the honest-verifier protocol does not accept as explicit evidence — so it is abstained (`insufficient_spec`), never silently passed.
2. The phase has a genuinely user-facing surface (the calendar decoration and month-cell chip, ANNOT-02 / criterion 3), whose visual legibility and browser anchor behavior no server-side test can observe. The infrastructure-phase auto-pass shortcut does not apply.
3. All 11 prohibitions are judgment tier with no wired enforcement, so they are flagged rather than green.

One documented, accepted deviation to be aware of when reading plan 33-04: the admin's clear branch keeps an in-memory nulling of the audit stamps instead of calling `unlink_event_from_run()`, because `super().save_model()`'s `obj.save()` would otherwise re-persist stale values over a helper-only bulk clear. This is what the plan's own truth for that case specifies, it is proven by a re-fetch-from-database test, and it leaves exactly one bulk writer of the cleared link in the codebase.

Phase 34's observation projector is unblocked: the carrier fields exist with the one-to-one constraint enforced at the database level, and the reconciler no longer has any path that can adopt, re-key, or detach an event the projector will own.

---

_Verified: 2026-09-04T18:09:09Z_
_Verifier: Claude (gsd-verifier)_
