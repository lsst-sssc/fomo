---
phase: 33
reviewers: [gemini, antigravity]
reviewed_at: 2026-09-04T05:20:08Z
plans_reviewed: [33-01-PLAN.md, 33-02-PLAN.md, 33-03-PLAN.md, 33-04-PLAN.md, 33-05-PLAN.md]
models:
  gemini: "unknown"
  antigravity: "unknown"
model_sources:
  gemini: "unknown"
  antigravity: "unknown"
lane_status:
  gemini: "failed — IneligibleTierError (free Gemini Code Assist tier no longer serves this CLI; re-auth required); no review produced"
  antigravity: "succeeded on retry — first attempt aborted on a headless-mode RunCommand denial; retried with a file-view/grep-only instruction"
---

# Cross-AI Plan Review — Phase 33

## Gemini Review

gemini review failed or returned empty output. stderr:
Error authenticating: IneligibleTierError: This client is no longer supported for Gemini Code Assist for individuals. To continue using Gemini, please migrate to the Antigravity suite of products: https://antigravity.google
    at throwIneligibleOrProjectIdError (file:///home/tlister/.nvm/versions/node/v22.23.2/lib/node_modules/@google/gemini-cli/bundle/chunk-MFLFXOVQ.js:310176:11)
    at _doSetupUser (file:///home/tlister/.nvm/versions/node/v22.23.2/lib/node_modules/@google/gemini-cli/bundle/chunk-MFLFXOVQ.js:310165:5)
    at process.processTicksAndRejections (node:internal/process/task_queues:103:5) {
  ineligibleTiers: [
    {
      reasonCode: 'UNSUPPORTED_CLIENT',
      reasonMessage: 'This client is no longer supported for Gemini Code Assist for individuals. To continue using Gemini, please migrate to the Antigravity suite of products: https://antigravity.google',
      tierId: 'free-tier',
      tierName: 'Gemini Code Assist for individuals'
    }
  ]
}
Ripgrep is not available. Falling back to GrepTool.
An unexpected critical error occurred:IneligibleTierError: This client is no longer supported for Gemini Code Assist for individuals. To continue using Gemini, please migrate to the Antigravity suite of products: https://antigravity.google
    at throwIneligibleOrProjectIdError (file:///home/tlister/.nvm/versions/node/v22.23.2/lib/node_modules/@google/gemini-cli/bundle/chunk-MFLFXOVQ.js:310176:11)
    at _doSetupUser (file:///home/tlister/.nvm/versions/node/v22.23.2/lib/node_modules/@google/gemini-cli/bundle/chunk-MFLFXOVQ.js:310165:5)
    at process.processTicksAndRejections (node:internal/process/task_queues:103:5)


---

## Antigravity Review

# Cross-AI Plan Review: Phase 33 — Series Identity & Reconciler Inversion

**Reviewed Repository:** `/home/tlister/git/fomo_devel`  
**Phase Under Review:** Phase 33 (`33-01` through `33-05`)  
**Target Milestone:** v2.4 Observation-First Calendar  
**Review Mode:** Headless verification against source code and schema (read-only)

---

## Executive Summary

Phase 33 addresses a foundational architectural pivot for FOMO's calendar and campaign infrastructure: inverting the campaign reconciler from an aggressive "owner" that adopts and re-keys existing calendar events into an "annotator" that decorates events while strictly constraining writes to its own `RUN:` key namespace. This decoupling is prerequisite to Phase 34's observation projector landing safely without having observation-derived events stolen or mutated. Concurrently, Phase 33 introduces real foreign keys on `CalendarEventMeta` (`observation_record` and `observation_group`) to carry series identity directly, retiring the title-suffix stopgap identified in Spike 002.

Overall, the plan suite (`33-01` through `33-05`) demonstrates outstanding technical rigor, deep familiarity with repository invariants (such as SPICE kernel load hazards in `solsys_code/views.py`, PII boundary gates, and django-tables2 dict-row nuances), and clear dependency sequencing across three execution waves. A small number of mechanical edge cases were identified during source verification—most notably potential dictionary key errors during loop execution in `campaign_reconciler.py`, in-memory mutation synchronization in `admin.py:save_model`, and helper parameter guarding in `unlink_event_from_run`—all of which are easily remedied with targeted adjustments.

---

## Detailed Plan Reviews

---

### Plan 33-01: Reconciler Inversion + Attributed-Event Decoration (Tracer)

#### 1. Summary
Plan 33-01 executes the core behavioral inversion: it deletes the reconciler's `_adopted_event_for_night()` adopt/re-key path, introduces `_night_already_attributed()` to skip minting for classical nights already covered by an attributed non-`RUN:` event, and implements `campaign_decoration(event)` in `solsys_code/templatetags/calendar_display_extras.py` so the modal displays attribution at render time. It also strips the campaign name prefix from reconciler-generated titles in `event_title()` (D-12) and updates all associated tests.

#### 2. Strengths
- **Clean Namespace Boundary Enforcement:** Completely eliminates `_adopted_event_for_night()` ([solsys_code/campaign_reconciler.py:290-342](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L290-L342)) and removes the re-keying call in `_reconcile_classical_nights()` ([solsys_code/campaign_reconciler.py:391-395,429](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L391-L395)), preventing the reconciler from modifying any event outside the `RUN:` URL namespace.
- **Defensive URL Construction in Python:** Generates `table_url` via `django.urls.reverse('campaigns:table', args=[run.campaign_id])` in Python within `campaign_decoration` rather than `{% url %}` in the template. This directly defuses the latent `NoReverseMatch` crash hazard when `run.campaign_id is None` on public pages ([solsys_code/models.py:82](file:///home/tlister/git/fomo_devel/solsys_code/models.py#L82), [solsys_code/campaign_urls.py:42](file:///home/tlister/git/fomo_devel/solsys_code/campaign_urls.py#L42)).
- **Preserves Critical Invariants:** Explicitly preserves the container branch's call to `update_calendar_event_key_and_fields()` ([solsys_code/campaign_reconciler.py:285](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L285)) and maintains the exact call location of `sun_event()` ([solsys_code/campaign_reconciler.py:387](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L387)) to retain existing error propagation for unconfigured timezones.
- **Accurate Test Impact Assessment:** Accurately identifies that `TestEventTitleGuard` ([solsys_code/tests/test_null_campaign_guards.py:81](file:///home/tlister/git/fomo_devel/solsys_code/tests/test_null_campaign_guards.py#L81)) and `TestEventTitleNullCampaignGuard` ([solsys_code/tests/test_write_and_reconcile.py:136](file:///home/tlister/git/fomo_devel/solsys_code/tests/test_write_and_reconcile.py#L136)) are the only two test cases in the suite asserting hardcoded campaign title prefixes.

#### 3. Concerns
- **`totals['skipped_nights']` KeyError Risk (Severity: MEDIUM):** In [solsys_code/campaign_reconciler.py:382](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L382), `totals` is initialized as `totals = {'created': 0, 'updated': 0, 'unchanged': 0, 'blocked': 0}`. Plan 33-01 specifies incrementing `totals['skipped_nights']` on line 1578 without explicitly mandating that `'skipped_nights': 0` be added to the initial dictionary literal. Attempting `totals['skipped_nights'] += 1` will raise a `KeyError` on the first skipped night.
- **N-Queries in Night Loop (Severity: LOW):** In `_reconcile_classical_nights()`, calling `_night_already_attributed(run, night, site_zone)` inside the `for i in range(n_nights):` loop ([solsys_code/campaign_reconciler.py:385-432](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L385-L432)) executes an ORM query for each night of the window. For longer windows (e.g., multi-week allocations), this generates unnecessary database round-trips.

#### 4. Suggestions
- Explicitly update the initialization dictionary in [solsys_code/campaign_reconciler.py:382](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L382) to `totals = {'created': 0, 'updated': 0, 'unchanged': 0, 'blocked': 0, 'skipped_nights': 0}`.
- Pre-calculate the set of attributed night dates for `run` once before entering the loop:
  ```python
  attributed_nights = {
      meta.event.start_time.astimezone(site_zone).date()
      for meta in CalendarEventMeta.objects.filter(run_id=run.pk)
      .exclude(event__url__startswith=RUN_URL_NAMESPACE)
      .select_related('event')
  }
  ```
  Then in the loop, simply evaluate `if existing is None and night in attributed_nights:`.

#### 5. Risk Assessment: LOW
The design is verified against existing methods and constraints. The primary risk is a simple dictionary initialization oversight that is caught immediately by the planned test suite.

---

### Plan 33-02: Month-Cell Campaign Marker, N+1-Free Prefetch, and Anchored Run Row

#### 1. Summary
Plan 33-02 extends the display-time campaign decoration from the event modal to the month calendar view (`calendar.html`). It adds a compact `.cal-campaign-chip` marker with the campaign name in the tooltip, updates `fomo_render_calendar` in `solsys_code/views.py` to prefetch `run__campaign` to prevent N+1 queries, adds anchor `id="run-{pk}"` to rows in `CampaignRunTable` via `row_attrs`, and adds target highlighting in `campaignrun_table.html`.

#### 2. Strengths
- **Title Budget Protection:** Preserves the month grid's strict truncation limits (`truncatechars:18` in [src/templates/tom_calendar/partials/calendar.html:234](file:///home/tlister/git/fomo_devel/src/templates/tom_calendar/partials/calendar.html#L234) and `truncatechars:16` in line 260) by positioning the marker outside the truncated title string.
- **N+1 Query Elimination:** Correctly identifies the exact prefetch chain in `solsys_code/views.py:114` ([solsys_code/views.py:107-116](file:///home/tlister/git/fomo_devel/solsys_code/views.py#L107-L116)) and augments it with `Prefetch('telescope_label_meta', queryset=CalendarEventMeta.objects.select_related('run__campaign'))`, ensuring zero additional queries per event.
- **Robust Table Row Anchor Resolution:** Accommodates django-tables2's dual row representation in [solsys_code/campaign_views.py:92-120](file:///home/tlister/git/fomo_devel/solsys_code/campaign_views.py#L92-L120) (model instances for staff, `.values()` dictionaries for non-staff) by utilizing `Accessor('pk').resolve(record, quiet=True)` within `CampaignRunTable.Meta.row_attrs`.
- **Pure CSS Target Highlighting:** Implements row selection via `tr:target` in [src/templates/campaigns/campaignrun_table.html](file:///home/tlister/git/fomo_devel/src/templates/campaigns/campaignrun_table.html#L53), eliminating client-side JavaScript complexity.

#### 3. Concerns
- **Color Contrast in All-Day Event Entries (Severity: LOW):** All-day calendar entries use dynamic background fills ([src/templates/tom_calendar/partials/calendar.html:219-233](file:///home/tlister/git/fomo_devel/src/templates/tom_calendar/partials/calendar.html#L219-L233)) with text color dynamically assigned via `{% text_color_for_bg bg_color as text_color %}` for WCAG AA compliance. If `.cal-campaign-chip` specifies a hardcoded dark glyph color, it will fail contrast on dark proposal background fills (such as `#005f9e`, `#5b2080`, or `#9e1c1c`).
- **Flex Shrinkage in Timed Entries (Severity: LOW):** In `.cal-event-timed` ([src/templates/tom_calendar/partials/calendar.html:50-54](file:///home/tlister/git/fomo_devel/src/templates/tom_calendar/partials/calendar.html#L50-L54)), elements sit in a flex container. If `.cal-campaign-chip` lacks `flex-shrink: 0`, narrow calendar cells might compress or truncate the glyph.

#### 4. Suggestions
- In the `.cal-campaign-chip` CSS rule, specify `color: currentColor;` and `flex-shrink: 0;`. This guarantees that in `.cal-event-all-day` the chip automatically inherits the WCAG-AA compliant `text_color`, and in `.cal-event-timed` it inherits the default text color without shrinking.
- Ensure the lambda in `row_attrs` safely handles potential null primary keys:
  ```python
  row_attrs = {
      'id': lambda record: f"run-{pk}" if (pk := Accessor('pk').resolve(record, quiet=True)) is not None else None
  }
  ```

#### 5. Risk Assessment: LOW
The plan builds upon established patterns in `calendar_display_extras.py` and `views.py`. Query count and PII regression tests provide robust automated validation.

---

### Plan 33-03: Series-Identity Link Fields (`observation_record`, `observation_group`), Migration 0017, Read-Only Admin

#### 1. Summary
Plan 33-03 delivers requirement PROJ-04 by adding nullable `observation_record` (`OneToOneField`) and `observation_group` (`ForeignKey`) fields to `CalendarEventMeta`, generating additive migration `0017`, updating the verbose name of `run` to `"Attributed campaign run"`, exposing the new fields as `readonly_fields` in `CalendarEventMetaAdmin` and `CalendarEventMetaInline`, and writing unit and migration tests via `MigrationExecutor`.

#### 2. Strengths
- **DB-Enforced Cardinality:** Implements `observation_record` as `models.OneToOneField(ObservationRecord, on_delete=models.SET_NULL, null=True, blank=True)` ([solsys_code/models.py:26-42](file:///home/tlister/git/fomo_devel/solsys_code/models.py#L26-L42)), enforcing the invariant of at most one calendar event per observation record at the schema level while allowing existing rows with `NULL` values to coexist without collision.
- **Explicit Migration Dependency Guarding:** Proactively identifies the necessity of including `('tom_observations', '0016_alter_facility_options')` in the migration dependencies ([solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py:22](file:///home/tlister/git/fomo_devel/solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py#L22)), preventing cross-app ordering anomalies.
- **Tamper Protection:** Marks both fields as read-only across both standalone admin and inline formsets ([solsys_code/admin.py:94,297](file:///home/tlister/git/fomo_devel/solsys_code/admin.py#L94)), ensuring only background projector code—never form binding—can populate provenance links.
- **Migration Verification Harness:** Utilizes the established `MigrationExecutor` + `TransactionTestCase` pattern from [solsys_code/tests/test_window_schema_migration.py:19-87](file:///home/tlister/git/fomo_devel/solsys_code/tests/test_window_schema_migration.py#L19-L87) to verify that migrating from `0016` to `0017` preserves all pre-existing audit and attribution data.

#### 3. Concerns
- **Scope Split Clarity for PROJ-04 (Severity: LOW):** PROJ-04 mentions both the link carrier fields and a "shared title stem". The plan clearly records that the shared title stem is deferred to Phase 34's projector. However, it must be verified that no code in Phase 33 attempts partial title formatting based on `observation_group`.
- **MigrationExecutor Teardown (Severity: LOW):** If a test using `MigrationExecutor` fails mid-execution, the shared test database might remain in an intermediate migration state unless restored in `tearDown()`.

#### 4. Suggestions
- Mirror the exact `tearDown()` implementation from [solsys_code/tests/test_window_schema_migration.py:82-87](file:///home/tlister/git/fomo_devel/solsys_code/tests/test_window_schema_migration.py#L82-L87) to guarantee that the database is migrated back to leaf nodes regardless of test assertion outcomes.
- Confirm that `related_name='calendar_event_meta'` on `observation_record` and `'calendar_event_metas'` on `observation_group` do not conflict with existing attributes in `tom_observations.models.ObservationRecord` and `ObservationGroup`. (Confirmed: verified clear in installed `tom_observations` package).

#### 5. Risk Assessment: LOW
Schema changes are purely additive with `null=True` and `on_delete=SET_NULL`. Existing rows in `src/fomo_db.sqlite3` are unaffected.

---

### Plan 33-04: Shared `unlink_event_from_run()` Helper Across Call Sites

#### 1. Summary
Plan 33-04 unifies three divergent "clear attribution" implementations across the codebase into a single shared helper: `unlink_event_from_run(events, run)` in `solsys_code/campaign_utils.py`. It updates `_undo_confirmation()` in `campaign_views.py`, `_detach_stale_family_events()` in `campaign_reconciler.py`, and `save_model()` in `admin.py`, guaranteeing that clearing an attribution clears `run`, `confirmed_by`, and `confirmed_at` together without deleting calendar events or modifying `is_verified`.

#### 2. Strengths
- **Audit Field Consistency:** Resolves the audit leakage in [solsys_code/campaign_reconciler.py:471](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L471), where detaching stale family events previously cleared only `run` while leaving stale `confirmed_by` and `confirmed_at` timestamps intact.
- **Preserves Concurrency and Tamper Guards:** Preserves the conditional check `run_id=run_pk` in [solsys_code/campaign_views.py:1326-1328](file:///home/tlister/git/fomo_devel/solsys_code/campaign_views.py#L1326-L1328), ensuring that a stale undo request cannot unlink an event that was re-attributed elsewhere, and returns `changed_count` to gate dismissal creation.
- **Strict Data Safety:** Enforces through automated checks that `unlink_event_from_run()` never deletes a `CalendarEvent` or `CalendarEventMeta` record and leaves `is_verified` untouched.

#### 3. Concerns
- **Missing Guard Against `run is None` in Helper (Severity: MEDIUM):** If `unlink_event_from_run(events, run)` is invoked with `run=None` (or `run_id=None`), a query such as `CalendarEventMeta.objects.filter(event_id=..., run_id=None)` would target companion rows that are *already* unlinked and erroneously clear their audit timestamps. The helper must explicitly validate that a valid run identifier is supplied.
- **In-Memory Instance State in Admin `save_model` (Severity: MEDIUM):** In [solsys_code/admin.py:382-386](file:///home/tlister/git/fomo_devel/solsys_code/admin.py#L382-L386):
  ```python
  elif obj.run_id is None and prior_run_id is not None:
      obj.confirmed_by = None
      obj.confirmed_at = None
  super().save_model(request, obj, form, change)
  ```
  If `unlink_event_from_run(obj.pk, prior_run_id)` is invoked directly inside `save_model()`, it performs a database `.update()`. If `obj.confirmed_by` and `obj.confirmed_at` are not also nulled on the in-memory `obj` instance, the subsequent `super().save_model()` call will execute `obj.save()`, overwriting the database row with stale in-memory values.

#### 4. Suggestions
- In `unlink_event_from_run()`, add an immediate guard:
  ```python
  run_pk = run.pk if hasattr(run, 'pk') else run
  if not run_pk:
      return 0
  ```
- For `admin.py:save_model()`, retain the in-memory attribute assignment `obj.confirmed_by = None; obj.confirmed_at = None` before calling `super().save_model()`, and add a reference comment indicating it adheres to the semantics of `unlink_event_from_run()`, avoiding conflicting double-writes.

#### 5. Risk Assessment: LOW-to-MEDIUM
The logic consolidation eliminates code duplication and prevents audit drift. Addressing the in-memory admin update hazard and `run_pk` validation makes the helper completely robust.

---

### Plan 33-05: Paired Docs: Demo Notebooks Re-Execution & Operator Runbook

#### 1. Summary
Plan 33-05 satisfies repository documentation mandates (CLAUDE.md paired-docs rule) and provides empirical verification for ROADMAP criterion 2. It re-executes `reconcile_campaign_runs_demo.ipynb` to demonstrate the classical-night skip rule and verify an empty diff across non-`RUN:` events, re-executes `campaign_lifecycle_demo.ipynb` to demonstrate decoration survival across title/description rewrites and unlinking, and updates the operator runbook (`docs/runbooks/telescope_runs_calendar.rst`) to replace ownership terminology with attribution concepts.

#### 2. Strengths
- **Empirical Real-DB Verification (D-04):** Captures snapshots of all non-`RUN:` events in `src/fomo_db.sqlite3` before and after a full reconciler sweep to prove byte-identical non-interference on production-like data.
- **Strict Isolation from Heavy Imports:** Adheres to the CLAUDE.md constraint forbidding direct imports of `solsys_code.views` or `solsys_code.ephem_utils` in notebooks and tests to prevent triggering 1.6 GB SPICE kernel downloads ([solsys_code/views.py](file:///home/tlister/git/fomo_devel/solsys_code/views.py#L27-L30)).
- **Thorough Runbook Overhaul:** Correctly pinpoints all three affected sections in [docs/runbooks/telescope_runs_calendar.rst](file:///home/tlister/git/fomo_devel/docs/runbooks/telescope_runs_calendar.rst#L294,L626,L719), specifically addressing the one-time title prefix churn, the skip rule, and the unlink behavior.

#### 3. Concerns
- **Misconception of `call_command` Output for Preview Details (Severity: LOW):** Plan 33-05 Task 1 states that the notebook will run `call_command('reconcile_campaign_runs', dry_run=True)` and print "the full previewed action list", "per-run `skipped_nights` totals", and "the set of urls the preview names". However, as verified in [solsys_code/management/commands/reconcile_campaign_runs.py:76-95](file:///home/tlister/git/fomo_devel/solsys_code/management/commands/reconcile_campaign_runs.py#L76-L95), the management command only prints aggregate summary counters to stdout and error lines to stderr; it does not output individual event URLs or per-run action listings.

#### 4. Suggestions
- Clarify in the notebook cell implementation that per-run preview inspection and URL collection should be performed by calling `reconcile_run(run, dry_run=True)` across runs programmatically in Python, rather than expecting `call_command`'s text output to contain per-URL breakdowns.
- Ensure that `pre-commit run --all-files` is executed after notebook execution to verify that Sphinx documentation builds and formatting checks pass cleanly.

#### 5. Risk Assessment: LOW
Task activities involve documentation and demonstration notebooks. Strict gating against committing PII fields ensures low operational risk.

---

## Cross-Plan Cohesion & Requirements Traceability

| Requirement | Description | Delivered in Plans | Status & Verification Mechanism |
|---|---|---|---|
| **PROJ-04** | Series identity carried by real FKs on `CalendarEventMeta` (`observation_record`, `observation_group`) | `33-03`, `33-05` | Verified. `33-03` adds schema, migration `0017`, and admin read-only controls; `33-05` demonstrates the links in the lifecycle notebook. Shared title stem is properly deferred to Phase 34. |
| **ANNOT-01** | `CalendarEventMeta.run` means attribution, not ownership; `reconcile_run()` skips attributed nights and never adopts, re-keys, or detaches non-`RUN:` events | `33-01`, `33-04`, `33-05` | Verified. `33-01` inverts the reconciler classical loop; `33-04` unifies unlinking and cleans up stale audit timestamps; `33-05` proves the empty diff on the live developer database. |
| **ANNOT-02** | Campaign decoration rendered from link at display time; never written into `CalendarEvent` fields; survives field rewrites | `33-01`, `33-02`, `33-05` | Verified. `33-01` delivers the modal block; `33-02` delivers the month-cell marker and prefetch; `33-05` provides end-to-end regression tests proving survival after from-scratch title/description overwrite. |

---

## Consolidation of Recommendations & Open Questions

### Actionable Adjustments for Implementation
1. **Initialize `skipped_nights` Counter in Reconciler Loop:**
   In [solsys_code/campaign_reconciler.py:382](file:///home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py#L382), include `'skipped_nights': 0` in `totals = {'created': 0, 'updated': 0, 'unchanged': 0, 'blocked': 0, 'skipped_nights': 0}` to avoid a `KeyError` when incrementing.
2. **Pre-Query Attributed Nights Outside Classical Night Loop:**
   Rather than querying `CalendarEventMeta` on every iteration of `range(n_nights)` in `_reconcile_classical_nights()`, evaluate the attributed dates once per run prior to loop entry.
3. **Guard `unlink_event_from_run` Against Null Runs:**
   In [solsys_code/campaign_utils.py](file:///home/tlister/git/fomo_devel/solsys_code/campaign_utils.py), ensure `unlink_event_from_run(events, run)` returns `0` immediately if `run` or `run.pk` is `None`, preventing accidental matching against existing `run_id IS NULL` rows.
4. **Preserve In-Memory Synchronization in Admin `save_model`:**
   In [solsys_code/admin.py:382-386](file:///home/tlister/git/fomo_devel/solsys_code/admin.py#L382-L386), retain explicit in-memory clearing of `obj.confirmed_by = None` and `obj.confirmed_at = None` so `super().save_model()` does not re-persist stale in-memory values to the database.
5. **Set `currentColor` on `.cal-campaign-chip`:**
   In [src/templates/tom_calendar/partials/calendar.html](file:///home/tlister/git/fomo_devel/src/templates/tom_calendar/partials/calendar.html), ensure the `.cal-campaign-chip` CSS class uses `color: currentColor;` and `flex-shrink: 0;` to guarantee WCAG-AA contrast across all dynamic proposal backgrounds and avoid flexbox shrinkage.
6. **Programmatic Inspection in Demo Notebook:**
   In `reconcile_campaign_runs_demo.ipynb`, inspect dry-run actions and per-run `skipped_nights` by iterating over `CampaignRun` records and calling `reconcile_run(run, dry_run=True)` directly in Python, rather than expecting `call_command('reconcile_campaign_runs')` stdout to supply per-URL listings.

### Open Questions
- None requiring external clarification. All domain and architectural questions are fully resolved by existing repository code and established test patterns.

---

## Final Risk Assessment

**Overall Phase Risk Level: LOW**

The five plans present a cohesive, well-engineered, and comprehensive roadmap for Phase 33. The division of labor across Waves 1, 2, and 3 is strictly parallel-safe with zero file overlap between concurrent plans. All potential failure modes identified in this review are minor mechanical details with clear, low-overhead fixes. Execution may proceed with high confidence.

---

## Consensus Summary

Only one of the two selected lanes produced a review. **Gemini failed before reviewing** (authentication tier error — see its stub above and `.review-diagnostics/gsd-review-gemini.err`), so there is no cross-model consensus: the sections below reflect a single source-grounded reviewer (Antigravity), which cited `file:line` evidence throughout and rated every plan LOW risk (33-04 LOW-to-MEDIUM), overall phase risk LOW. Treat "agreed" items as single-reviewer findings that the planner should still act on; re-run `/gsd-review 33 --gemini` once Gemini is re-authenticated if a second opinion is wanted before execution.

### Agreed Strengths
- The reconciler inversion is a clean namespace boundary: `_adopted_event_for_night()` and the re-key call are removed, `sun_event()` and the container-branch `update_calendar_event_key_and_fields()` call are kept exactly where they are (`solsys_code/campaign_reconciler.py:285, 290-342, 387, 391-395`).
- `campaign_decoration()` builds the run link with `reverse()` in Python and guards `campaign_id is None`, so the `NoReverseMatch` hazard on public pages is structurally impossible (`solsys_code/models.py:82`, `solsys_code/campaign_urls.py:42`).
- Schema work is purely additive (`OneToOneField`/`ForeignKey`, `null=True`, `on_delete=SET_NULL`), the `tom_observations` migration dependency is called out, both fields are read-only in admin and inline, and the `MigrationExecutor` proof reuses the established `test_window_schema_migration.py` pattern.
- The month-cell marker stays outside the `truncatechars:16/18` title budget; the prefetch chain (`Prefetch('telescope_label_meta', queryset=…select_related('run__campaign'))`) keeps the cell N+1-free; the `Accessor('pk').resolve(record, quiet=True)` row id handles django-tables2's dict rows for non-staff.
- The shared `unlink_event_from_run()` fixes a real audit leak in `_detach_stale_family_events()` (`campaign_reconciler.py:471` cleared `run` but not `confirmed_by/at`) while preserving `_undo_confirmation`'s concurrency-safe conditional update (`campaign_views.py:1326-1328`).
- Requirements traceability: PROJ-04, ANNOT-01, ANNOT-02 each map to named plans with a stated verification mechanism; the PROJ-04 title-stem deferral to Phase 34 is judged correct.

### Agreed Concerns
Ordered by severity; all are implementation-detail fixes, none change a CONTEXT.md decision.
1. **MEDIUM — `totals['skipped_nights']` KeyError (33-01).** `totals` at `campaign_reconciler.py:382` is initialised as `{'created','updated','unchanged','blocked'}`; the plan increments `totals['skipped_nights']` without mandating `'skipped_nights': 0` in the literal. Fix: add it to the initialiser (and to `ReconcileResult`'s default).
2. **MEDIUM — `unlink_event_from_run(events, run)` with `run=None` (33-04).** A `filter(..., run_id=None)` would match already-unlinked rows and wipe their audit timestamps. Fix: return 0 immediately when the run/pk is falsy.
3. **MEDIUM — in-memory state in `admin.save_model` (33-04).** If the helper does a DB `.update()` inside `save_model()` but `obj.confirmed_by/at` are not also nulled on the instance, the following `super().save_model()` re-persists stale values (`admin.py:382-386`). Fix: keep the in-memory clearing alongside the helper call.
4. **LOW — dry-run inspection mechanism (33-05 Task 1).** `reconcile_campaign_runs --dry-run` prints only aggregate counters (`management/commands/reconcile_campaign_runs.py:76-95`), not per-URL actions or per-run `skipped_nights`. The notebook's `PRE-SWEEP DRY-RUN INSPECTION` cell must call `reconcile_run(run, dry_run=True)` per run in Python to get the URL set it asserts on.
5. **LOW — per-night query in `_reconcile_classical_nights()` (33-01).** `_night_already_attributed()` inside `for i in range(n_nights)` is one ORM query per night; compute the attributed-night date set once per run before the loop.
6. **LOW — `.cal-campaign-chip` CSS (33-02).** Use `color: currentColor; flex-shrink: 0;` so the chip inherits the WCAG-AA `text_color` on dark proposal fills and does not compress in `.cal-event-timed` flex rows; guard the `row_attrs` lambda for a null pk.
7. **LOW — `MigrationExecutor` teardown (33-03).** Mirror `test_window_schema_migration.py:82-87`'s `tearDown()` so a failed assertion cannot leave the test DB at `0016`.

### Divergent Views
None — only one reviewer produced a verdict. The Gemini lane's absence is the divergence to note: a second grounded review has not been obtained.
