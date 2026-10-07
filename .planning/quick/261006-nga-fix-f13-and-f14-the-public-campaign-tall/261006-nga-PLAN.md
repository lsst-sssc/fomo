---
phase: 261006-nga
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/campaign_tally.py
  - solsys_code/tests/test_campaign_tally.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/campaign_tables.py
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
autonomous: true
requirements: [TALLY-01, TALLY-02, ALLOC-01]

estimate:
  tokens: 115000
  raw_tokens: 115000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "A run with no site (a class-wide queue allocation such as KEY2026B-004's runs 69-75) no longer reports `[O] 0 [S] 0 [X/F] 0` when it has linked records. Each claiming record's night is the noon-anchored site-local observing night in the timezone of the record's `parameters['observed_site']`, looked up in the tally-only map `_NIGHT_SITE_TIMEZONES` (developer decision 2026-10-06). Failing that, the run's own site timezone is used, and failing that, the UTC calendar date of the record's start."
    - "D-11 de-duplication is kept, keyed on the night DATE alone: a date counts once in its state's set however many records, or sites, fall on it. Records at cpt, lsc or tfn whose exposures fall on either side of midnight UTC on one site-local night count as one night."
    - "A run WITH a site keeps today's result whenever its records were observed at that site or carry no observed site. A record whose observed site is in the map is keyed in that site's timezone even on a run with a different site (the record first, then the run, then the UTC date)."
    - "`night_counts_for_run()` still never raises. It still fetches records with the same `.only('pk', 'status', 'facility', 'scheduled_start', 'scheduled_end', 'parameters')` columns; no column is added, because `parameters` already carries `observed_site`. It still issues exactly ONE query per run, with no Observatory lookup for record sites."
    - "`_NIGHT_SITE_TIMEZONES` lives in `campaign_tally.py`, covers every LCO portal site code in `calendar_utils.SITE_TELESCOPE_MAP` plus `tlv`, and is used ONLY for tally night keying. It is not a site/obscode mapping, `campaign_attribution.LCO_SITE_CODE_TO_OBSCODE` is unchanged, and `campaign_attribution.py`, `campaign_gap.py` and `proposal_allocation.py` never reference it (pinned by a test)."
    - "Both cache keys carry a counting-rule version segment (`campaign_tally:v2:...`, `campaign_rollup:v2:...`), so a zero tally cached by the old rule in the shared file cache is never served after the update. The roll-up (`campaign_rollup()`, `_rollup_runs()`) needs no logic change: it sums the per-run tallies."
    - "On the public campaign table every badge uses a Bootstrap 5 colour class. The telescope-class badge reads `1m0` as dark text on a light background with its grey border (the `1m0 class allocation` tooltip is unchanged), and the run-status, approval and TBD badges are visible. The Progress cell renders as two lines that never break internally: `N groups · N records`, then the four night segments."
    - "`campaign_lifecycle_demo.ipynb` was re-executed on its scratch copy. Its stored output shows the class-wide run going from all-zero nights to `[O] 2 [S] 1 [X/F] 1`, with two lsc records either side of midnight UTC counting as one night, the public row's readable badge and two-line Progress cell, and `PASS: F13/F14`. The runbook's tally section says how a record's night is keyed."
  artifacts:
    - "solsys_code/campaign_tally.py: module constants `TALLY_CACHE_KEY_VERSION = 'v2'` (used by `build_tally_cache_key()` and `build_rollup_cache_key()`) and `_NIGHT_SITE_TIMEZONES` (LCO portal site code -> IANA timezone name, tally night keying only); private helpers `_usable_zone(tz_name)` and `_record_site_zone(record)`; `night_counts_for_run()` rewritten to key each record's night by its observed site, then the run's site, then the UTC date."
    - "solsys_code/tests/test_campaign_tally.py: new class `TestNightCountsPerRecordSite` (11 tests), new class `TestTallyNightSiteTimezones` (2 tests), new `TestBuildTallyCacheKey.test_keys_carry_the_counting_rule_version`, and `test_site_unset_returns_all_zero_no_exception` renamed to `test_site_unset_counts_the_utc_date_and_never_raises` with its new expected counts."
    - "solsys_code/tests/test_campaign_views.py: new classes `TestProgressColumnOnClassWideRun` (1 end-to-end test, Task 1) and `TestCampaignTableBadgesAndProgressLayout` (4 tests, Task 2)."
    - "solsys_code/campaign_tables.py: the `APPROVAL_BADGE_CLASSES`/`RUN_STATUS_BADGE_CLASSES` values, both renderers' fallbacks, `render_telescope_class()` and the TBD badge in `render_window_start()` all use `text-bg-*`; `render_progress()` emits two `d-block text-nowrap` lines."
    - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb: one new markdown+code pair between cells `eb92e3b4` and `914891c4`, and one added paragraph at the end of `bbf027b0` (Summary). Re-executed with output."
    - "docs/runbooks/telescope_runs_calendar.rst: three edits in 'What does a run's or a campaign's public tally show?'."
  key_links:
    - "`night_counts_for_run()` -> `_record_site_zone(record)` reads `record.parameters['observed_site']` -> `_NIGHT_SITE_TIMEZONES` -> `_usable_zone()` -> `telescope_runs.observing_night(start, zone)`. When that gives no zone, it falls back to `_usable_zone(run.site.timezone)`, then `start.astimezone(UTC).date()`. Neither campaign_gap.py nor campaign_attribution.py is called, imported or edited."
    - "`tallies_for_runs()` (the table, via `CampaignRunTableView.get_table()`) and `get_or_compute_tally()` (the calendar pop-up tag) both call `night_counts_for_run()` on a cache miss and key the cache with `build_tally_cache_key()`. `campaign_rollup()` sums `tallies_for_runs()`, and `get_or_compute_rollup()` keys with `build_rollup_cache_key()`. The version segment in the two key builders is what retires stale zeros on every surface at once."
    - "`CampaignRunTable.render_progress()` reads `campaign_tally.tally_segments()`, and `render_telescope_class()` reads `TELESCOPE_CLASS_LABELS`. `ApprovalQueueTable` inherits the badge renderers, so the staff approval queue gets readable badges too. `src/templates/campaigns/campaignrun_table.html` renders the table with `{% render_table table %}` and is not edited."
---

<objective>
Fix intent-review findings F13 and F14 (`.planning/v2.4-INTENT-REVIEW.md`, "### F13 (2026-10-06)" and "### F14 (2026-10-06)").

F13: logged out on the `KEY2026B-004_targets` campaign page, all seven run rows (CampaignRun pks 69-75: `source=lco_queue`, `telescope_class='1m0'`, `site=None`) and the roll-up read `[O] 0 [S] 0 [X/F] 0`. Yet run 69 alone holds 84 linked records, 67 of them `COMPLETED`. `campaign_tally.night_counts_for_run()` returns three zeros whenever the run has no site, because Phase 37 D-11 keys nights on the RUN's site timezone. TALLY-01 is therefore unmet for every queue allocation, which is the main case.

F14: on the same page the `1m0` telescope-class badge is invisible until hovered, and the Progress column is very narrow and hard to read.

Purpose: the public tally must show what a class-wide run actually got, and the table must be readable. This is display and counting only; nothing is written.

Output:
- Task 1 (the tracer): per-record night keying through a tally-only site timezone map, plus the cache-key version.
- Task 2: readable Bootstrap 5 badges and a two-line Progress cell.
- Task 3: the paired notebook and runbook, plus the full-suite gate.

No migration is needed. `ObservationRecord.parameters` and `Observatory.timezone` already exist.

**Planning-time findings. Read these before starting; each one changes how a task is done.**

1. **The cron job runs from THIS checkout every 15 minutes, and the web server may load these modules at any time.**
   - Write the tests first; no runner imports test files.
   - Make each production edit so the module still imports, and still works when called, the moment it is saved. Add new constants and helpers before the code that uses them.
   - Run the import smoke check (each task's first `<automated>`) after every production edit.
2. **The site timezone map (developer decision, 2026-10-06; this supersedes the first draft's reuse of `campaign_gap.observation_site_obscode()`).** Add `_NIGHT_SITE_TIMEZONES`, a dict from LCO portal site code (the value the observation projector stores in `parameters['observed_site']`) to an IANA timezone name.
   - **Contents, all eight entries:** `coj`->`Australia/Sydney`, `cpt`->`Africa/Johannesburg`, `elp`->`America/Chicago`, `lsc`->`America/Santiago`, `ogg`->`Pacific/Honolulu`, `sor`->`America/Santiago`, `tfn`->`Atlantic/Canary`, `tlv`->`Asia/Jerusalem`. The developer named seven; `sor` (SOAR) is added because `calendar_utils.SITE_TELESCOPE_MAP` already names it, and a test requires every site code in that map to be covered.
   - **Home: `campaign_tally.py`, not `calendar_utils.py`.** `calendar_utils` is imported by `campaign_attribution`, `campaign_gap` and (through `campaign_attribution`) `proposal_allocation`, so a map there would sit one import away from exactly the modules it must stay out of. None of those three imports `campaign_tally`, and `proposal_allocation` cannot (`campaign_tally` imports it, so the reverse import would be a cycle). The leading underscore marks it as private to the module.
   - **Documented in its comment as used ONLY for tally night keying.** It is not a site/obscode mapping, it answers only "which clock does an observing night at this site run on", and it must not be imported by attribution, gap analysis or proposal allocation.
   - **Nothing else changes for those modules.** `campaign_attribution.LCO_SITE_CODE_TO_OBSCODE` and `OBSERVED_TELESCOPE_OBSCODES` are not touched. `campaign_tally` no longer needs `campaign_gap` or `Observatory` for record sites. Task 1 verify 4 pins `campaign_attribution.py`, `campaign_gap.py`, `proposal_allocation.py` and `calendar_utils.py` unchanged, and `TestTallyNightSiteTimezones` pins that the three consumers never reference the map.
   - **A record's site code is read defensively.** Use it only when it is a `str`; normalise it with `.strip().lower()`; look it up in the map. Never pass record data to `ZoneInfo`: only the map's own values reach it.
   - **v2.5 idea, NOT in scope.** The developer's preferred general fix is per-site obscode SETS in `LCO_SITE_CODE_TO_OBSCODE` (each LCO site mapped to the set of its MPC obscodes) with membership checks. That would let attribution, gap analysis and the tally share one verified site table. It must not be started here. Record it in the SUMMARY under "Ideas for v2.5" only.
3. **The night key, decided here: the date alone, not (site, date).** One date counts once in its state's set however many records or sites fall on it. This is D-11's "one night counts once" applied across sites, and it matches the operator's sanity figure (run 69: 67 completed records on 48 distinct dates). Under a (site, date) key, two sites observing on the same night would count twice, which is not what "nights observed" means for a campaign. With every LCO site now in the map, a night that straddles midnight UTC (cpt, lsc, tfn) counts once, which the new straddle test pins.
4. **Precedence and fallbacks.** The order is: the record's mapped observed site, then the run's site timezone, then the UTC date (the order F13 states).
   - The UTC date is `start_time.astimezone(UTC).date()`, with no noon anchor.
   - Records with no observed site use the fallbacks. In practice these are a placed-but-not-yet-observed block and an expired, cancelled or failed request, since the projector stores `observed_site` only once an observation completes. On a class-wide run they are therefore counted on their UTC date. That is the specified rule; leave it as it is.
   - The run fallback reads `run.site.timezone`, which every caller already loads (`_rollup_runs()`'s `.only()` names `site__timezone`, and `CampaignRunTableView.get_table()` uses `select_related('site')`). No new query is added.
5. **Cache-key decision: bump both keys.**
   - **Why a bump is needed.** `CACHES` is `FileBasedCache` at `tempfile.gettempdir()` (settings.py:214; `local_settings.py` does not override it), so entries survive a server restart. A run's key moves only when a linked record's `modified`, its link count or its newest link id moves. A class-wide run whose records are all completed would therefore keep serving its cached zeros for up to `TALLY_CACHE_TTL_SECONDS` (3600 s), and so would its campaign roll-up.
   - **The bump.** Add `TALLY_CACHE_KEY_VERSION = 'v2'` and put it in both key formats. Old entries become unreachable and expire on their own TTL; nothing needs deleting.
   - **The roll-up needs no logic change.** `campaign_rollup()` sums `tallies_for_runs()`, and `_rollup_runs()` already loads everything the new rule reads from a run.
   - **Test caches.** New DB test classes use `@override_settings(CACHES=TEST_CACHES)`, because `cache.clear()` against the real `FileBasedCache` would empty the shared temp-dir cache.
6. **F14's cause: Bootstrap 5.3.3 is loaded (`django_bootstrap5`'s CDN), and every badge colour class in `campaign_tables.py` is a Bootstrap 4 name.**
   - **Why it is invisible.** Bootstrap 5 dropped those names. Its `.badge` has white text and no background, so `badge badge-light` is white on white, and only the inline grey border shows ("invisible until hovered").
   - **The same defect elsewhere in this table.** The run-status (`badge-secondary`/`-info`/`-primary`/`-light`), approval (`-warning`/`-success`/`-danger`) and TBD (`badge-secondary`) badges in the SAME table have it too, with no border to give them away. They are fixed in the same commit; this is the same defect, not another finding.
   - **The fix.** Use Bootstrap 5's `text-bg-*` classes, the form `attribution_queue.html` and `campaign_list.html` already use. The telescope-class badge becomes `text-bg-light` and keeps its inline `border: 1px solid #6c757d;`. That is exactly what the Bootstrap 4 badge used to render: a light background, dark text and a grey border (the muted CANON-02/D-18 token).
   - **Out of scope.** Other Bootstrap 4 leftovers (`font-italic`, `mr-*`, `form-group`, the table's `bootstrap4-responsive.html` template name) belong to WR-15's class of work. Do not edit `src/templates/`.
7. **Progress width: two lines, no CSS file.**
   - **Today.** `render_progress()` emits one span that wraps at every space in a narrow column.
   - **The change.** Render the counts and the segments as two `d-block text-nowrap` spans inside the existing span that carries the `title`. The column can then never be narrower than its longer line, and the cell is always exactly two lines.
   - **Existing tests are safe.** Every token they check (`2 groups`, `3 records`, `[O] 1`, `[U] ≈3`, `Progress not available`) stays contiguous. Only the ` · ` between the record count and the first segment goes; the line break replaces it.
8. **Paired docs.**
   - **Notebook.** `campaign_tally.py` and `campaign_tables.py` have no explicit CLAUDE.md map entry. `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` is the notebook that demonstrates the tally, so it is the paired one. Its tally cells are `e4de8c1d` (pop-up tally), `eb92e3b4` (Progress cells and roll-up, including `class_wide_run`: `telescope_class='1m0'`, `site=None`, all zeros today) and `67f4c20a`.
   - **Where the new cells go.** A new markdown+code pair goes between `eb92e3b4` and `914891c4`. No later cell reads the class-wide run's tally or links a record to a run (checked).
   - **Setup and teardown cells.** The setup cell `219a2c08` copies the live database to a `fomo-notebook-db-` scratch file. Never edit it or the teardown cell `cef92204`.
   - **Runbook.** The section to edit is "What does a run's or a campaign's public tally show?".
   - **Nothing else goes stale.** No pre-executed notebook or doc contains the old badge class names (checked).
9. **Files that are not yours.**
   - The operator has uncommitted edits in `.planning/v2.4-INTENT-REVIEW.md`, `.planning/v2.4-MILESTONE-AUDIT.md`, `.planning/state.json` and `.planning/milestones/v1.1-phases/03-classical-calendar-ingest/03-VERIFICATION.md`.
   - There are untracked files: `.planning/agent-history.json`, `reqgroup_2682493.json`, `src/fomo_db_*.sqlite3`, and a `.gitkeep` under `.planning/phases/37.1-*`.
   - Do not edit, stage or commit any of them. The F13/F14 "Routed" and "Fix landed" notes are the orchestrator's job.
   - Run `git branch --show-current` before the first commit; it must print `issue37-telescope-runs-calendar`. Stage every commit by explicit path.
10. **Allowed commands.**
    - `python manage.py test ...` (never `./manage.py`), `pre-commit run ...`, `jupyter nbconvert ...` (Task 3 only, after an END banner), `git`, and read-only `python -c` checks. The import smoke check calls `django.setup()` but opens no database connection.
    - Nothing else: no other `python manage.py` command, no network or portal call, and nothing that opens `src/fomo_db.sqlite3`. The notebook setup cell's `shutil.copy2` is the only permitted access.
    - Run every RED test run WITHOUT `--parallel`. Under `--parallel`, a failing run crashes with "cannot pickle 'traceback'" instead of printing the failure (261006-lsf SUMMARY).
11. **Code reading.** Prefer Serena's symbolic tools (`get_symbols_overview`, `find_symbol`) where available; otherwise use Read/Grep with offsets. Line numbers here are from planning time (~).

**Source coverage audit.** Nothing is unplanned.
- **GOAL.** F13 (a class-wide run's tally counts its nights) and F14 (badge readable, Progress column readable) are Task 1 and Task 2. The paired docs are Task 3.
- **REQ.**
  - TALLY-01 (per-run public tally, nights by state, updating as the projector narrows): Task 1 (counting) and Task 2 (legibility).
  - TALLY-02 (campaign roll-up): Task 1, where the sum is confirmed and tested and the key is bumped.
  - ALLOC-01 (the class-wide allocation's badge): Task 2.
- **CONTEXT.**
  - The night-keying rule as amended by the developer's 2026-10-06 decision: Task 1, with findings 2-4.
  - The tests to pin: Task 1 (T1-T11, M1-M2, V1, the cache-key test and the renamed site-unset test) and Task 2 (B1-B4).
  - Cache key: finding 5 and Task 1.
  - Roll-up: finding 5, T11 and V1.
  - Paired docs: Task 3, with finding 8.
  - Constraints: findings 1, 9 and 10, plus each task's verify.
  - The obscode-sets approach is recorded as a v2.5 idea (finding 2), at the developer's direction, and is not planned.
- **RESEARCH.** None (no research phase).
- **Planner contributions.**
  - Security: see `<threat_model>`.
  - Schema gate: no Payload/Prisma/Drizzle/Supabase/TypeORM file is in scope, and no Django model or migration changes.
  - API coverage: see `<api_coverage_decision>` and `COVERAGE.md`.
  - Assumption delta: see `<assumption_delta_decision>`.
- **Out of scope.** F4, F6, F8, F10 and the other findings must not be touched.
</objective>

<api_coverage_decision>
No external API integration. This task changes how FOMO's own tally module counts ObservationRecord rows that are already stored: it keys each record's night by the observed site earlier code stored on it, through a fixed site-code-to-timezone table. It also changes how FOMO's own django-tables2 campaign table renders its badges and Progress cell, and updates the paired notebook and runbook. It calls, wraps or adds no external API, SDK or service.
</api_coverage_decision>

<assumption_delta_decision>
Detector result: skipped (`phase_unresolved`, because a quick-task id has no ROADMAP section), so the checkpoint did not fire. This record is kept voluntarily, because the change moves from one to many: the timezone a night is keyed by goes from one per run (the run's site) to one per record (where that record was observed).

- **Noun that is now primary:** the record's observed site.
- **Decision:** `promote`. The record's own site decides its night; the run's site becomes the first fallback and the UTC date the last. Nothing is added alongside the old rule.
- **Invariant test:** `TestNightCountsPerRecordSite.test_unmapped_observed_site_falls_back_to_the_runs_site` goes red if a future change puts the run's site back ahead of a record's mapped site, or drops the run-site fallback.
</assumption_delta_decision>

<execution_context>
@.claude/gsd-core/workflows/execute-plan.md
@.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@solsys_code/campaign_tally.py
@solsys_code/campaign_tables.py
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1 (tracer): a class-wide run's nights are counted at each record's own observing site, end to end into the public table row and the roll-up</name>
  <files>solsys_code/campaign_tally.py, solsys_code/tests/test_campaign_tally.py, solsys_code/tests/test_campaign_views.py</files>
  <read_first>
    - .planning/v2.4-INTENT-REVIEW.md, section "### F13 (2026-10-06)" only (read; never edit)
    - solsys_code/campaign_tally.py: imports (~16-32), `TALLY_CACHE_TTL_SECONDS`/`_NO_RECORDS_VERSION_TOKEN` (~36-53), `build_tally_cache_key()` (~70-110), `night_counts_for_run()` (~165-230), `_rollup_runs()` (~514-546, read to confirm no change), `build_rollup_cache_key()` (~722-743)
    - solsys_code/calendar_utils.py `SITE_TELESCOPE_MAP` (~56-71) and `OBSERVED_SITE_PARAMETER_KEYS` (~95); read only, never edit
    - solsys_code/campaign_attribution.py `LCO_SITE_CODE_TO_OBSCODE` and its comment (~38-76): read only, so you know what the new map must NOT be
    - solsys_code/telescope_runs.py `observing_night()` (~309-335); solsys_code/calendar_utils.py `record_time_window()` (~564)
    - solsys_code/tests/test_campaign_tally.py: imports and `TEST_CACHES` (~15-53), `CampaignTallyTestBase` (~56-121), `TestModuleImportGuard` (~123-141, the source-scan style), `TestBuildTallyCacheKey` (~144-162), `TestNightCountsForRun` (~213-345), `TestCampaignRollup.test_only_does_not_trigger_deferred_field_queries` (~743-765)
    - solsys_code/tests/test_campaign_views.py: imports (~13-38) and `CampaignTallyViewTestBase` (~861-927)
  </read_first>
  <behavior>
    All datetimes are UTC, and every Target comes from `NonSiderealTargetFactory` (CLAUDE.md). Night arithmetic was checked at planning time. In July 2026 Sydney is UTC+10, Honolulu UTC-10, Johannesburg UTC+2, Santiago UTC-4 and the Canaries UTC+1. `observing_night()` subtracts 12 h from local time.

    **In solsys_code/tests/test_campaign_tally.py, changes to existing classes:**
    - Add `test_keys_carry_the_counting_rule_version` to `TestBuildTallyCacheKey`. It asserts `build_tally_cache_key(7, None) == 'campaign_tally:v2:7:none:0:0'` and `build_rollup_cache_key(7, None) == 'campaign_rollup:v2:7:none'`.
    - Rename `TestNightCountsForRun.test_site_unset_returns_all_zero_no_exception` to `test_site_unset_counts_the_utc_date_and_never_raises`. The fixture is unchanged: a site-less run with one COMPLETED record, block 2026-07-10 04:00-10:00, `parameters={}`. It now expects `{'nights_observed': 1, 'nights_scheduled': 0, 'nights_failed': 0}`. This is the one existing test that pinned the F13 defect.

    **New class `TestNightCountsPerRecordSite(CampaignTallyTestBase)`, appended after `TestNightCountsForRun`:**
    - Decorated `@override_settings(CACHES=TEST_CACHES)`; `setUp` calls `cache.clear()`.
    - Class constants, one per site:
      - `COJ_1M = {'observed_site': 'coj', 'observed_telescope': '1m0a', 'observed_enclosure': 'doma'}`
      - `OGG_2M = {'observed_site': 'ogg', 'observed_telescope': '2m0a', 'observed_enclosure': 'clma'}`
      - `CPT_1M = {'observed_site': 'cpt', 'observed_telescope': '1m0a', 'observed_enclosure': 'domc'}`
      - `LSC_1M = {'observed_site': 'lsc', 'observed_telescope': '1m0a', 'observed_enclosure': 'domb'}`
      - `TFN_1M = {'observed_site': 'tfn', 'observed_telescope': '1m0a', 'observed_enclosure': 'doma'}`
      - `UNMAPPED = {'observed_site': 'xxx', 'observed_telescope': '1m0a', 'observed_enclosure': 'doma'}` (a site code the map does not name)
    - Helper `_class_wide_run(**overrides)` returns `self._make_run(site=None, site_raw='', source=CampaignRun.Source.LCO_QUEUE, telescope_class=CampaignRun.TelescopeClass.ONE_M0, telescope_instrument='LCO 1m0 / Sinistro', ...)`, with overrides winning.
    - Helper `_observed(run, start, site_parameters, minutes=30)` links a COMPLETED record whose block runs from `start` to `start + minutes`, with `parameters=dict(site_parameters)`.
    - The 11 tests:
      - **T1** `test_class_wide_run_counts_observed_nights_by_each_records_own_site`. coj at 07-10 10:00 is Sydney night 07-10; ogg at 07-10 12:00 is Honolulu night 07-09. Expect `{'nights_observed': 2, 'nights_scheduled': 0, 'nights_failed': 0}`. UTC keying would give 1.
      - **T2** `test_two_records_on_one_site_local_night_count_once`. coj at 07-10 10:00 and at 07-10 16:00 are both Sydney night 07-10. Expect `nights_observed == 1`.
      - **T3** `test_one_night_date_at_two_sites_counts_once`. coj at 07-10 10:00 is night 07-10; ogg at 07-11 08:00 is 22:00 Honolulu on 07-10, so also night 07-10. Expect `nights_observed == 1`. UTC keying, or a (site, date) key, would give 2.
      - **T4** `test_class_wide_run_counts_a_placed_block_as_scheduled_and_an_expired_request_as_failed`. Three records:
        - coj COMPLETED at 07-10 10:00;
        - a `PENDING` record with a block 07-20 10:00-10:30 and `parameters={}`;
        - a `WINDOW_EXPIRED` record with no block and `parameters={'start': '2026-07-21T00:00:00', 'end': '2026-07-23T00:00:00'}`.
        Expect `{'nights_observed': 1, 'nights_scheduled': 1, 'nights_failed': 1}`.
      - **T5** `test_record_without_observed_site_on_a_site_less_run_uses_the_utc_date`. Two COMPLETED records with `parameters={}`, at 07-10 23:30 and 07-11 00:30. Expect `nights_observed == 2`, because these are two UTC calendar dates; a noon anchor would give 1.
      - **T6** `test_unmapped_observed_site_falls_back_to_the_runs_site`. The run HAS `site=self.site` (F65, Honolulu). coj at 07-10 10:00 uses its own site: Sydney night 07-10. The `UNMAPPED` record at 07-11 08:00 falls back to the run's site: Honolulu night 07-10. Expect `nights_observed == 1`. Run-site-first would give 2 (coj would become 07-09), and a UTC fallback for the unmapped record would give 2.
      - **T7** `test_unusable_run_timezone_and_messy_observed_site_fall_back_to_the_utc_date`. Inside the test, create `Observatory(obscode='Z99', name='Bad Zone Site', short_name='BAD', lat=0.0, lon=0.0, altitude=10, timezone='Not/A_Zone', observations_type=Observatory.OPTICAL_OBSTYPE)` and use it as the run's site. Link two COMPLETED records: one with `parameters={'observed_site': None}` at 07-10 23:30, and one with `parameters={'observed_site': 42}` at 07-11 00:30. Expect `nights_observed == 2` with no exception.
      - **T8** `test_single_site_run_result_is_unchanged_when_records_were_observed_there`. The run has `site=self.site`. Link ogg at 07-10 12:00, a `parameters={}` COMPLETED record at 07-10 06:00, and ogg at 07-11 12:00. Expect `nights_observed == 2`. This passes before and after the change: it is a guard.
      - **T9** `test_night_counts_issue_one_query_for_the_whole_run`. A site-less run with coj at 07-10 10:00, coj at 07-11 10:00, ogg at 07-10 12:00 and lsc at 07-10 23:30. Assert with `self.assertNumQueries(1): night_counts_for_run(run)`: one records query, and no Observatory lookup for record sites.
      - **T10** `test_records_straddling_midnight_utc_at_cpt_lsc_or_tfn_count_one_night`. This is the developer's required test. For each of `CPT_1M`, `LSC_1M` and `TFN_1M`, in a `subTest`, build a fresh site-less run with three COMPLETED records on one site-local night:
        - a block at 07-10 23:30;
        - a block at 07-10 23:50 that runs 30 minutes, so its exposure itself crosses 00:00 UTC;
        - a block at 07-11 01:30.
        Expect `nights_observed == 1` for each site. All three are night 07-10 in Johannesburg, Santiago and the Canaries; UTC dates would give 2.
      - **T11** `test_rollup_of_class_wide_runs_is_the_sum_of_their_tallies`. Two class-wide runs with `campaign=self.campaign`: run A has coj at 07-10 10:00; run B has ogg at 07-10 12:00 and ogg at 07-11 12:00.
        - `campaign_rollup(self.campaign)['nights_observed'] == 3`, which equals the sum over `tallies_for_runs([run_a, run_b]).values()`.
        - `get_or_compute_rollup(self.campaign)`, called twice, returns `nights_observed == 3` both times.

    **New class `TestTallyNightSiteTimezones(TestCase)`, appended after it.** No DB or cache is needed. Read the map as `getattr(campaign_tally, '_NIGHT_SITE_TIMEZONES', {})`, so that during RED it fails an assertion rather than erroring.
    - **M1** `test_map_covers_every_lco_site_code_with_a_valid_zone`:
      - its keys include `{'coj', 'cpt', 'elp', 'lsc', 'tfn', 'ogg', 'tlv'}`;
      - its keys include every site code in `calendar_utils.SITE_TELESCOPE_MAP` (`{site for site, _ in SITE_TELESCOPE_MAP}`);
      - every key is lower case;
      - `ZoneInfo(value)` succeeds for every value;
      - `coj` maps to `Australia/Sydney`, `cpt` to `Africa/Johannesburg`, `elp` to `America/Chicago`, `lsc` to `America/Santiago`, `tfn` to `Atlantic/Canary`, `ogg` to `Pacific/Honolulu` and `tlv` to `Asia/Jerusalem`.
    - **M2** `test_map_is_used_only_by_the_tally`:
      - `'_NIGHT_SITE_TIMEZONES'` appears in `inspect.getsource(campaign_tally)`;
      - it appears in none of `inspect.getsource()` of `campaign_attribution`, `campaign_gap` and `proposal_allocation`;
      - `campaign_attribution.LCO_SITE_CODE_TO_OBSCODE == {'coj': 'E10'}` (unchanged).
      - Import those three modules at the top of the test module; `campaign_gap` and `proposal_allocation` are already imported there.

    **In solsys_code/tests/test_campaign_views.py:**
    - Add `override_settings` to the `django.test` import, and define a module-level `TEST_CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}`.
    - New class `TestProgressColumnOnClassWideRun(CampaignTallyViewTestBase)`, decorated `@override_settings(CACHES=TEST_CACHES)` and appended after `TestCampaignRollup`. `setUp` calls `cache.clear()`.
    - **V1** `test_class_wide_run_row_and_rollup_show_its_nights` is the tracer's end-to-end leg.
      - Create `run = self._make_run(site=None, site_raw='', source=CampaignRun.Source.LCO_QUEUE, telescope_class=CampaignRun.TelescopeClass.ONE_M0)`.
      - Link four records: coj COMPLETED at 07-10 10:00, ogg COMPLETED at 07-10 12:00 (both using the observed-site dicts as in T1), a PENDING block at 07-20 10:00, and a WINDOW_EXPIRED request with the T4 parameters.
      - Make an anonymous GET of `reverse('campaigns:table', kwargs={'pk': self.campaign.pk})`.
      - Take the row slice from `f'id="run-{run.pk}"'` to the next `</tr>` and collapse its whitespace. It must contain `[O] 2`, `[S] 1` and `[X/F] 1`.
      - `response.context['rollup']` must have `nights_observed == 2`, `nights_scheduled == 1` and `nights_failed == 1`.
  </behavior>
  <action>
    **Step 0.** Run `git branch --show-current`; it must print `issue37-telescope-runs-calendar`.

    **Step 1: RED (tests only).**
    - Write the tests in `<behavior>`, adding only the imports they need. `override_settings`, `Observatory`, `ZoneInfo`, `inspect`, `campaign_gap` and `proposal_allocation` are already imported in test_campaign_tally.py. Add `campaign_attribution` and `from solsys_code.calendar_utils import SITE_TELESCOPE_MAP` there. test_campaign_views.py needs `override_settings` added.
    - Update test_campaign_tally.py's module docstring to mention the tally-only site timezone map.
    - Run, SERIALLY (finding 10): `python manage.py test solsys_code.tests.test_campaign_tally solsys_code.tests.test_campaign_views.TestProgressColumnOnClassWideRun --noinput`.
    - Expected RED, each failing with an AssertionError: the cache-key test, the renamed site-unset test, T1-T7, T9, T10, T11, M1, M2 and V1. T8 passes as a guard, and every other pre-existing test passes.
    - If a failure is an ImportError, fixture error or IntegrityError, fix the test first. If the pass/fail split differs, work out why before writing production code.

    **Step 2: GREEN, solsys_code/campaign_tally.py.** Each edit must leave the module importable and working (finding 1); run the smoke check after each one.
    - **Edit A1 (import only).** Add `from datetime import timezone as dt_timezone`; the module already binds `timezone` to `django.utils.timezone`. Do NOT import `campaign_gap`, `campaign_attribution` or `Observatory`: they are not needed (finding 2).
    - **Edit A2 (cache-key version, per finding 5).** Below `_NO_RECORDS_VERSION_TOKEN`, add the module constant `TALLY_CACHE_KEY_VERSION = 'v2'`.
      - Its comment says three things: it is the counting-rule version carried in both cache keys; F13 (quick task 261006-nga, 2026-10-06) bumped it from the unversioned form, because a class-wide run's night counts changed from zero to real values and the shared file-based cache would otherwise keep serving the old zeros for up to `TALLY_CACHE_TTL_SECONDS`; and it must be bumped again whenever the cached counting rule changes.
      - `build_tally_cache_key()` returns `campaign_tally:{TALLY_CACHE_KEY_VERSION}:{run_pk}:{version_segment}:{records_count}:{link_version or 0}`.
      - `build_rollup_cache_key()` returns `campaign_rollup:{TALLY_CACHE_KEY_VERSION}:{campaign_pk}:{version_segment}`.
      - Add one sentence about the version segment to each docstring.
    - **Edit A3 (map and helpers, per finding 2).**
      - Below `_NIGHT_CLAIMING_STATES`, add `_NIGHT_SITE_TIMEZONES: dict[str, str]` holding the eight entries listed in finding 2.
      - Its block comment says:
        - it maps an LCO portal site code (the value the observation projector stores in `ObservationRecord.parameters['observed_site']`) to the IANA timezone that site's observing nights run on;
        - it is used ONLY by `night_counts_for_run()` to key a linked record's night (F13, quick task 261006-nga, developer decision 2026-10-06);
        - it is NOT a site or obscode mapping and says nothing about which telescope or Observatory a record used;
        - it must never be imported by attribution, gap analysis or proposal allocation, which keep using `campaign_attribution.LCO_SITE_CODE_TO_OBSCODE`, unchanged;
        - a site's timezone is the same for every telescope there, which is why one entry per site is safe here although it is not safe for that obscode table;
        - `TestTallyNightSiteTimezones` pins the coverage and the scope.
      - Then add two private helpers above `night_counts_for_run()`, each with a Google-style docstring:
        - `_usable_zone(tz_name: str | None) -> ZoneInfo | None` returns `ZoneInfo(tz_name)` for a non-blank, known IANA name. It returns `None` for a blank name, or when `ZoneInfo` raises `ZoneInfoNotFoundError`, `TypeError` or `ValueError`. It never raises.
        - `_record_site_zone(record) -> ZoneInfo | None` reads `(record.parameters or {}).get('observed_site')`. It uses the value only if it is a `str`, normalises it with `.strip().lower()`, looks it up in `_NIGHT_SITE_TIMEZONES`, and returns `_usable_zone()` of the mapped name, or `None` when the site is missing, not a string or not in the map. Record data never reaches `ZoneInfo`; only map values do. It never raises.
    - **Edit A4: rewrite `night_counts_for_run(run)`.** Keep its signature, its return shape and its "never raises" contract.
      - (a) `run_zone = _usable_zone(run.site.timezone) if run.site_id is not None else None`. There is no early return any more.
      - (b) The records query and its `.only(...)` columns are unchanged. Extend the comment above it: `parameters` is also where `_record_site_zone()` reads the record's observed site, so no column is added.
      - (c) In the loop, keep today's per-record steps exactly: `facility_for_or_none()`, then `classify_record()`, then the `_NIGHT_CLAIMING_STATES` filter, then `record_time_window()` with its `KeyError`/`ValueError` skip.
      - (d) Then set `zone = _record_site_zone(record) or run_zone`. The night is `observing_night(start_time, zone)` when `zone` is not `None`; otherwise it is `start_time.astimezone(dt_timezone.utc).date()`.
      - (e) Add the night to the observed, scheduled or failed set exactly as today. The key is the date alone (finding 3).
      - (f) Emit one `logger.debug` line per run giving how many records were keyed by their observed site, by the run's site and by the UTC date.
      - **Rewrite the docstring** to cover:
        - the per-record rule and its order;
        - the date-only de-duplication (D-11), including that a night straddling midnight UTC counts once;
        - that records with no observed site (a placed block, or an expired or failed request) fall back to the run's site, then the UTC date;
        - that the query cost is unchanged (one records query);
        - that a site-less run no longer reports zero (F13, quick task 261006-nga).
      - Do not touch `_rollup_runs()`, `campaign_rollup()`, `tallies_for_runs()` or `get_or_compute_tally()` (finding 5).

    **Step 3: confirm.** Run every `<automated>` command below.
    - If a PRE-EXISTING test fails, stop and report it.
    - The only exception is a test that pins the F13 defect itself, a site-less run reporting zero nights. The one known instance is renamed in Step 1. Name any other such test you update in the SUMMARY.
    - Record each module's `Ran N tests` count.

    **Step 4: commit** the three files by explicit path, as `fix(261006-nga): count a class-wide run's nights at each record's own observing site`.
  </action>
  <verify>
    <automated>python -c "import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings'); import django; django.setup(); import solsys_code.campaign_tally as t, solsys_code.campaign_tables as tb, solsys_code.templatetags.calendar_display_extras as d; from solsys_code.calendar_utils import SITE_TELESCOPE_MAP; assert t.build_tally_cache_key(7, None) == 'campaign_tally:v2:7:none:0:0', t.build_tally_cache_key(7, None); assert t.build_rollup_cache_key(7, None) == 'campaign_rollup:v2:7:none'; m = t._NIGHT_SITE_TIMEZONES; need = {'coj', 'cpt', 'elp', 'lsc', 'tfn', 'ogg', 'tlv'} | {s for s, _ in SITE_TELESCOPE_MAP}; assert need <= set(m), sorted(need - set(m)); print('importable:', t.night_counts_for_run.__name__, tb.CampaignRunTable.__name__, d.run_tally.__name__, sorted(m))"</automated>
    <automated>python manage.py test solsys_code.tests.test_campaign_tally solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_gap solsys_code.tests.test_calendar_display_extras --noinput --parallel 4</automated>
    <automated>pre-commit run ruff --files solsys_code/campaign_tally.py solsys_code/tests/test_campaign_tally.py solsys_code/tests/test_campaign_views.py && pre-commit run ruff-format --files solsys_code/campaign_tally.py solsys_code/tests/test_campaign_tally.py solsys_code/tests/test_campaign_views.py</automated>
    <automated>git diff --quiet HEAD -- solsys_code/campaign_attribution.py solsys_code/campaign_gap.py solsys_code/proposal_allocation.py solsys_code/calendar_utils.py src/templates/ && echo "OK: obscode tables, gap ladder, proposal allocation, shared helpers and templates untouched"</automated>
  </verify>
  <done>
    - The cache-key test, T1-T11, M1, M2, V1 and the renamed site-unset test pass.
    - Every pre-existing test in the four listed modules passes, including the unchanged roll-up query-count tests (6 and 3).
    - A class-wide run's public row and the roll-up show its observed, scheduled and expired nights, and a cpt/lsc/tfn night that straddles midnight UTC counts once.
    - Both keys carry `v2`. The map lives in campaign_tally.py only, and the obscode table and its consumers are untouched.
    - ruff and ruff-format are clean, and one commit holds exactly the three files.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: readable Bootstrap 5 badges and a two-line Progress cell on the campaign table</name>
  <files>solsys_code/campaign_tables.py, solsys_code/tests/test_campaign_views.py</files>
  <read_first>
    - .planning/v2.4-INTENT-REVIEW.md, section "### F14 (2026-10-06)" only (read; never edit)
    - solsys_code/campaign_tables.py: the two badge dicts and their comments (~21-48), `progress` column (~94-98), `render_progress()` (~175-217), `render_run_status()` (~219-234), `render_approval_status()` (~236-244), `render_telescope_class()` (~246-269), `render_window_start()` (~299-322)
    - src/templates/campaigns/attribution_queue.html ~78-89 and campaign_list.html ~54 (read only: the `badge text-bg-*` form already in use)
    - solsys_code/tests/test_campaign_views.py `TestWindowColumnRendering` (~175-245, the `CampaignRunTable([run]).rows[0].get_cell(...)` pattern) and `TestCampaignRunTableProgressColumn` (~929-1035, the tokens that must stay contiguous)
  </read_first>
  <behavior>
    **New class `TestCampaignTableBadgesAndProgressLayout(CampaignTallyViewTestBase)`**, appended at the end of solsys_code/tests/test_campaign_views.py.
    - Decorated `@override_settings(CACHES=TEST_CACHES)` (the local constant Task 1 added); `setUp` calls `cache.clear()`.
    - Imports added: `APPROVAL_BADGE_CLASSES` and `RUN_STATUS_BADGE_CLASSES` from `solsys_code.campaign_tables`.

    **The four tests:**
    - **B1** `test_telescope_class_badge_is_dark_text_on_a_light_background`:
      - Make a run with `telescope_class='1m0'`. Its `CampaignRunTable([run]).rows[0].get_cell('telescope_class')` contains `class="badge text-bg-light"`, `>1m0<`, `title="1m0 class allocation"` and `border: 1px solid #6c757d;`.
      - `CampaignRunTable([]).render_telescope_class({'telescope_class': '1m0'})` returns the same markup. This is the dict row, which is the anonymous path.
    - **B2** `test_every_badge_uses_a_bootstrap5_colour_class`:
      - Every value of both dicts starts with `text-bg-`.
      - For each `CampaignRun.RunStatus` value, `render_run_status({'run_status': value})` contains `f'class="badge {RUN_STATUS_BADGE_CLASSES[value]}"'`. The three dead-end statuses (CANCELLED, NOT_AWARDED, WEATHER_TECH_FAILURE) still carry the grey border style.
      - For each `CampaignRun.ApprovalStatus` value, `render_approval_status({'approval_status': value})` contains `f'class="badge {APPROVAL_BADGE_CLASSES[value]}"'`.
      - The `window_start` cell of a TBD run (no window) contains `class="badge text-bg-secondary"`.
      - None of these rendered strings matches the regex `\bbadge-[a-z]`.
    - **B3** `test_progress_cell_puts_counts_and_segments_on_two_unbreakable_lines`:
      - Build `table = CampaignRunTable([run])` and set `table.tallies = {run.pk: tally}`, with `tally = {'groups': 2, 'records': 3, 'nights_observed': 1, 'nights_scheduled': 1, 'nights_failed': 1, 'nights_unused': None, 'unused_is_estimate': True, 'unused_known': False}`.
      - The `progress` cell contains exactly two `class="d-block text-nowrap"`.
      - The first holds `2 groups · 3 records` and the second `[O] 1 [S] 1 [X/F] 1 [U] not yet known`, in that order.
      - The outer span keeps `title="..."`, equal to the labels of `campaign_tally.tally_segments(tally)` joined by `', '`.
    - **B4** `test_public_table_row_carries_the_readable_badge_and_two_line_progress_cell`:
      - Make a class-wide run (`site=None`, `site_raw=''`, `telescope_class='1m0'`) with nothing linked.
      - Make an anonymous GET of the campaign table. The run's row slice (taken as in V1) contains `class="badge text-bg-light"` and `>1m0<`, contains `d-block text-nowrap` exactly twice, and contains `[O] 0` after whitespace collapse.
  </behavior>
  <action>
    This is display only, per F14 and finding 6.

    **Step 1: RED.** Write B1-B4 and run them SERIALLY: `python manage.py test solsys_code.tests.test_campaign_views.TestCampaignTableBadgesAndProgressLayout --noinput`. All four should fail with an AssertionError: the Bootstrap 4 class names are still rendered, and the cell is still one span. If a failure is anything other than an AssertionError, fix the test first.

    **Step 2: GREEN, solsys_code/campaign_tables.py.** The web server may load this module at any time, so run the smoke check after each edit.
    - **(a) Dict values.**
      - `APPROVAL_BADGE_CLASSES` becomes `text-bg-warning`/`text-bg-success`/`text-bg-danger`.
      - `RUN_STATUS_BADGE_CLASSES` becomes `text-bg-secondary` (requested, planned), `text-bg-info` (observed, reduced), `text-bg-primary` (published) and `text-bg-light` (the three dead-end statuses).
      - Update the two comments above the dicts to name the Bootstrap 5 classes. Add one sentence saying the Bootstrap 4 badge colour names render white-on-white under the Bootstrap 5.3 this site loads (F14, quick task 261006-nga).
    - **(b) Fallbacks.** `render_run_status()`: the fallback becomes `text-bg-secondary`, and the dead-end border check compares against `text-bg-light`. `render_approval_status()`: the fallback becomes `text-bg-secondary`.
    - **(c) `render_telescope_class()`.** The span's classes become `badge text-bg-light`. Keep the inline grey border style, the `title` tooltip and the raw value. Add a docstring sentence: dark text on a light background with a grey border, the muted CANON-02/D-18 token, readable without hovering (F14).
    - **(d) `render_window_start()`.** Both TBD spans use `badge text-bg-secondary`.
    - **(e) `render_progress()`.**
      - Keep the not-available branch exactly as it is.
      - The rendered branch becomes an outer span carrying the same `title`, containing two `d-block text-nowrap` spans: the first holds `{groups} group{s} · {records} record{s}`, the second `segments_text`.
      - Values are still interpolated through `format_html`; nothing is concatenated as raw HTML.
      - Update the docstring: the counts and the four segments sit on two lines that never wrap internally, so the column is never narrower than its longer line and the cell is always two lines (F14).
    - Do not edit `src/templates/`, `Meta.template_name` or any other Bootstrap 4 leftover (finding 6).

    **Step 3.** Run every `<automated>` command below. If a pre-existing test fails, stop and report it; no pre-existing test is expected to pin the old class names or the single-span cell (checked at planning time). Then commit the two files by explicit path, as `fix(261006-nga): readable Bootstrap 5 badges and a two-line Progress cell on the campaign table`.
  </action>
  <verify>
    <automated>python -c "import os, re; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings'); import django; django.setup(); from solsys_code.campaign_tables import APPROVAL_BADGE_CLASSES as A, RUN_STATUS_BADGE_CLASSES as R, CampaignRunTable as T; vals = list(A.values()) + list(R.values()); assert all(v.startswith('text-bg-') for v in vals), vals; html = str(T([]).render_telescope_class({'telescope_class': '1m0'})); assert 'badge text-bg-light' in html and '>1m0<' in html and not re.search(r'\bbadge-[a-z]', html), html; print('OK: Bootstrap 5 badge classes;', html)"</automated>
    <automated>python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_approval --noinput --parallel 4</automated>
    <automated>pre-commit run ruff --files solsys_code/campaign_tables.py solsys_code/tests/test_campaign_views.py && pre-commit run ruff-format --files solsys_code/campaign_tables.py solsys_code/tests/test_campaign_views.py</automated>
    <automated>git diff --quiet HEAD -- src/templates/ solsys_code/campaign_tally.py && echo "OK: templates and the tally module untouched by this task"</automated>
  </verify>
  <done>
    - B1-B4 pass, and test_campaign_views and test_campaign_approval pass in full.
    - Every badge on the campaign table carries a Bootstrap 5 colour class, and so does the staff approval queue, which inherits the renderers. The telescope-class badge is readable without hovering.
    - The Progress cell is two unbreakable lines.
    - The templates are untouched, lint is clean, and one commit holds exactly the two files.
  </done>
</task>

<task type="auto">
  <name>Task 3: paired notebook and runbook show a class-wide run's per-site nights and the readable row; full-suite gate</name>
  <files>docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst</files>
  <read_first>
    - CLAUDE.md "Paired docs are part of the deliverable"
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb cells (by id; positions may shift): `219a2c08` (scratch setup: read, never edit), `d687bfab` and `bb1b424c` (where `class_wide_run`, `campaign` and `public_client` come from), `ebdd8815` (ObservationRecord creation style), `7993474e` and `eb92e3b4` (the tally/roll-up demo the new pair follows), `914891c4` (the next section), `bbf027b0` (Summary), `cef92204` (teardown: never edit)
    - docs/runbooks/telescope_runs_calendar.rst, section "What does a run's or a campaign's public tally show?" (~2699-2790)
  </read_first>
  <action>
    **Part A: notebook sources.** Edit the JSON cell sources. Keep every existing cell id; new cells get fresh 8-hex-digit ids.

    Insert a new markdown cell and a new code cell between `eb92e3b4` and `914891c4`.

    **The markdown cell.** Its heading is "## A class-wide run's nights, counted where each observation was made (F13/F14, quick task 261006-nga)". In plain English it covers:
    - Before the fix, the tally keyed every night on the RUN's site. A run with no site (a class-wide queue allocation, like `class_wide_run` above) therefore showed zero nights however much it had observed.
    - Now each linked record's night is taken in the timezone of the site where it was observed: the LCO site code FOMO stores on the record when the observation completes (for example `coj`, `lsc`). Failing that, the run's own site is used, and failing that, the UTC date.
    - One night counts once, however many records or sites fall on it, so the two `lsc` records below, either side of midnight UTC, are one night.
    - The placed block and the expired request carry no observed site, and this run has no site, so they are counted on their UTC dates.
    - The tally's cache key now carries a version, so old cached zeros are never shown.
    - The same cell shows F14 on the public row: the readable telescope-class badge and the two-line Progress cell.

    **The code cell.** Its first line is the comment `# F13/F14 class-wide tally demo` (the gate looks for it). Import `datetime` and `timezone as dt_timezone` explicitly, plus `re`, `ObservationRecord`, `CampaignRunObservation`, `NonSiderealTargetFactory` and `campaign_tally`. Then, in order:
    - Compute `before = campaign_tally.night_counts_for_run(class_wide_run)`, print it, and assert all three counts are 0.
    - Get or create the Target named `Campaign Lifecycle Demo F13 Target`, following the `ebdd8815` pattern with `NonSiderealTargetFactory`.
    - Create six records with `ObservationRecord.objects.update_or_create(facility='LCO', observation_id=f'campaign-lifecycle-demo-f13-{n}', defaults=...)`. Link each with `CampaignRunObservation.objects.get_or_create(run=class_wide_run, observation_record=record)`. The six records:
      1. COMPLETED at coj (`observed_site='coj'`, `observed_telescope='1m0a'`, `observed_enclosure='doma'`), block 2026-09-10 10:00-10:30 UTC.
      2. COMPLETED at coj, block 2026-09-10 16:00-16:30 UTC. That is 02:00 Sydney on 09-11, so the same site-local night as record 1.
      3. COMPLETED at lsc (`'lsc'`, `'1m0a'`, `'domb'`), block 2026-09-11 23:30-23:50 UTC.
      4. COMPLETED at lsc, block 2026-09-12 02:00-02:30 UTC. Records 3 and 4 are both Santiago night 09-11, on two UTC dates.
      5. PENDING with a placed block 2026-09-20 03:00-03:30 UTC and `parameters={}`.
      6. WINDOW_EXPIRED with no block and `parameters={'start': '2026-09-14T00:00:00', 'end': '2026-09-16T00:00:00'}`.
    - Print one line per record: observation_id, status, the stored observed site or `-`, and a note. The note is `keyed at its observed site` when the record carries an observed site; otherwise it is `no observed site, run has no site: keyed by UTC date`.
    - Compute `after = campaign_tally.night_counts_for_run(class_wide_run)` and `print(f'after:  {after}')`. Assert `after == {'nights_observed': 2, 'nights_scheduled': 1, 'nights_failed': 1}`. Also print the four UTC dates of the observed records (09-10, 09-10, 09-11, 09-12), to show that they span three UTC dates while the tally counts two nights. Print `campaign_tally.TALLY_CACHE_KEY_VERSION`.
    - GET `reverse('campaigns:table', args=[campaign.pk])` with `public_client` and assert 200.
      - Slice the class-wide run's row from `f'id="run-{class_wide_run.pk}"'` to the next `</tr>`.
      - Assert the whitespace-collapsed row contains `[O] 2`, `[S] 1` and `[X/F] 1`.
      - Assert the row contains `class="badge text-bg-light"` and `>1m0<`, and contains `d-block text-nowrap` exactly twice.
      - Print the badge span, and the Progress cell's two lines as plain text (tags stripped with `re.sub`).
    - Take `rollup = response.context['rollup']`. Assert `rollup['nights_observed'] == 2`, and that it equals the sum of `nights_observed` over `campaign_tally.tallies_for_runs(...)` for the campaign's runs that are not pending review. Print both.
    - Finish with a line starting `PASS: F13/F14` that summarises the checks.

    **The Summary cell `bbf027b0`.** Add one short paragraph at the end. It names the F13/F14 demo and its proofs: the class-wide run's nights counted at each record's own site (with a night that straddles midnight UTC counting once), the roll-up summing it, and the readable badge with the two-line Progress cell.

    Do not touch `219a2c08`, `cef92204` or any other cell.

    **Part B: lint, then execute.**
    - Run `pre-commit run ruff --files docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`, then `pre-commit run ruff-format --files docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`. Re-run the first if the second changed anything.
    - The setup cell copies the live database, so start only when the last banner from `tail -n 3 /var/log/fomo/unattended.log` is an `=== FOMO unattended run END` line. Ticks run at :00, :15, :30 and :45 and take about 60 s.
    - Run from the repo root, in the FOREGROUND (Bash timeout 600000 ms; never `run_in_background`, never `&`): `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`. The setup cell resolves the repo root as `Path.cwd().parents[2]`, so the kernel must start in `docs/notebooks/pre_executed/`; nbconvert does this by default by running the kernel in the notebook's own directory.
    - Never export `FOMO_DATABASE_PATH`, and never copy a database yourself.
    - If setup fails with "database disk image is malformed", re-run after the next END banner.
    - If the NEW cell fails, fix its source and re-execute the WHOLE notebook. Never hand-write outputs.
    - If a PRE-EXISTING cell's assertion fails, stop and report the cell id and message.

    **Part C: runbook.** Make three scoped edits in docs/runbooks/telescope_runs_calendar.rst, all inside "What does a run's or a campaign's public tally show?". Use plain English and the page's RST style, and keep each bold lead-in on one line.
    1. After the first paragraph (it ends "exactly like the campaign table and the calendar itself."), add the sentence: "In a run row's Progress cell the group and record counts are on the first line and the four night counts on the second."
    2. Replace the paragraph that begins "Nights are counted on the site-local observing night (the same" with one paragraph. Its bold lead-in is **Each record's night is taken at the site where it was observed.** It says:
       - A linked record's night is the site-local observing night (the noon-anchored rule ``telescope_runs.observing_night()`` uses everywhere else in FOMO). It is taken in the timezone of the LCO site FOMO stored on the record when its observation completed (``coj``, ``cpt``, ``elp``, ``lsc``, ``ogg``, ``sor``, ``tfn`` or ``tlv``), so a night that crosses midnight UTC is still counted once, on the night an observer at that site would call it.
       - A record with no stored site, such as a scheduled block or an expired request, uses the run's own site. When the run has no site either (a class-wide allocation), its UTC calendar date is used.
       - A night counts once, however many records or sites fall on it.
       - Before quick task 261006-nga (2026-10-06), a run with no site of its own (every class-wide queue allocation) showed zero nights however much it had observed.
    3. In the "**How fresh is the tally?**" paragraph, directly after the sentence that ends "a fresh computation immediately, for the same reason as before.", add: "The cache key also carries a version that changes whenever the counting rule itself changes (it did with quick task 261006-nga), so a figure cached under an older rule is never shown after an update."

    Change nothing else in the runbook.

    **Part D: full suite and final gates.**
    - Run the `<automated>` commands below in order.
    - The full suite is about 2130 tests (2113 before this task) and takes about 2-3 min with `--parallel 4` (Bash timeout 600000 ms). It uses Django's test database, never the live one. Quote the `Ran N tests` and `OK` lines in the SUMMARY.
    - If a pre-existing test fails, stop and report it.
    - Commit the notebook and runbook by explicit path, as `docs(261006-nga): demo a class-wide run's per-site night tally and the readable campaign row; runbook tally wording`.
    - Then check `git status --porcelain`: the four operator-owned `.planning/` files are still modified and unstaged, and the untracked files are still untracked (finding 9).
    - In the SUMMARY, add an "Ideas for v2.5" line recording the developer's preferred general fix: per-site obscode sets in `LCO_SITE_CODE_TO_OBSCODE` with membership checks (finding 2). Do not implement it.
  </action>
  <verify>
    <automated>python -c "
import json, re
nb = json.load(open('docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb'))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
counts = [c.get('execution_count') for c in code]
assert counts == list(range(1, len(code) + 1)), f'not one fresh top-to-bottom run: {counts}'
errors = [i for i, c in enumerate(code) if any(o.get('output_type') == 'error' for o in c.get('outputs', []))]
assert not errors, f'error outputs in code cells {errors}'
def out(c):
    return ''.join(''.join(o.get('text', '')) for o in c.get('outputs', []))
setup = [c for c in code if 'FOMO_DATABASE_PATH' in ''.join(c['source'])]
assert len(setup) == 1, len(setup)
m = re.search(r\"Resolved database: '([^']+)'\", out(setup[0]))
assert m and 'fomo-notebook-db-' in m.group(1), 'setup cell did not resolve to a scratch database'
ids = [c.get('id') for c in nb['cells']]
i, j = ids.index('eb92e3b4'), ids.index('914891c4')
assert j - i == 3, 'expected exactly one new markdown+code pair between the tally demo and the unused-night section'
demo = nb['cells'][i + 2]
src = ''.join(demo['source'])
assert demo['cell_type'] == 'code' and src.startswith('# F13/F14 class-wide tally demo'), 'F13/F14 demo cell not where expected'
text = out(demo)
for token in (\"'nights_observed': 2, 'nights_scheduled': 1, 'nights_failed': 1\", 'keyed at its observed site', 'keyed by UTC date', 'text-bg-light', 'v2', 'PASS: F13/F14'):
    assert token in text, f'F13/F14 demo output lacks {token!r}'
assert 'Removed scratch database directory' in out(code[-1]), 'last code cell is not the scratch teardown'
print(f'OK: {len(code)} code cells, one fresh run, scratch database {m.group(1)}')
"</automated>
    <automated>python -c "
t = open('docs/runbooks/telescope_runs_calendar.rst').read()
a = t.index(\"What does a run's or a campaign's public tally show?\")
b = t.index('How do I find nights that were observed but never claimed', a)
s = ' '.join(t[a:b].split())
for phrase in (\"Each record's night is taken at the site where it was observed.\", '\`\`tlv\`\`', '261006-nga', 'UTC calendar date', \"In a run row's Progress cell the group and record counts are on the first line\", 'a figure cached under an older rule is never shown after an update'):
    assert phrase in s, f'tally section lacks {phrase!r}'
print('OK: runbook tally section updated')
"</automated>
    <automated>pre-commit run ruff --files docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb && pre-commit run ruff-format --files docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb && pre-commit run sphinx-build --files docs/runbooks/telescope_runs_calendar.rst</automated>
    <automated>python manage.py test solsys_code --exclude-tag=ephemeris_segfault --noinput --parallel 4</automated>
    <automated>git status --porcelain .planning/v2.4-INTENT-REVIEW.md .planning/v2.4-MILESTONE-AUDIT.md .planning/state.json .planning/milestones/v1.1-phases/03-classical-calendar-ingest/03-VERIFICATION.md</automated>
    <human-check>Operator: once the web server is serving the new code (an auto-reloading dev server picks it up by itself; a long-running server needs a restart), open the `KEY2026B-004_targets` campaign page logged out and check:
- Rows 69-75 show non-zero nights.
- Run 69's `[O]` is close to 48. It may differ by one or two, because nights are now site-local rather than UTC dates.
- Run 71's `[S]` is at least 1.
- Run 73's `[O]` is close to 29.
- Runs 70/72/74/75 show `[O]` of at most 7/11/8/8.
- The roll-up strip equals the sum of the rows.
- The `1m0` badge reads without hovering, the run-status badges are visible, and each Progress cell is two lines.
Then tick F13 and F14 in `.planning/v2.4-INTENT-REVIEW.md`.</human-check>
  </verify>
  <done>
    - The lifecycle notebook was executed once, top to bottom, with no error output, on a `fomo-notebook-db-` scratch copy that its last cell removed.
    - The new F13/F14 cell sits between `eb92e3b4` and `914891c4`. Its stored output shows:
      - the class-wide run going from all zeros to `[O] 2 [S] 1 [X/F] 1`, with the two lsc records either side of midnight UTC counting once;
      - which records were keyed at their observed site and which by the UTC date;
      - the `v2` key version;
      - the readable badge and the two-line Progress cell;
      - `PASS: F13/F14`.
    - The runbook carries the three edits, Sphinx builds and the full suite passes.
    - One docs commit holds exactly these two files, the operator's `.planning/` files are still unstaged and uncommitted, and the SUMMARY records the v2.5 idea.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| working tree -> running cron and web server | `run_unattended` runs from this checkout every 15 min, and the web server imports `campaign_tally`/`campaign_tables` on each table, list or pop-up render, so a saved edit is live code |
| anonymous visitor -> campaign table/list | the tally is public (TALLY-01), so its cost is paid on every anonymous page load that misses the cache |
| stored record parameters -> tally | `observed_site` came from the LCO portal earlier; it now chooses which timezone a night is keyed in |
| notebook kernel -> Django `DATABASES` | a docs artifact runs real ORM writes on the production host |
| executor -> git history | commits are made in a working tree that holds the operator's uncommitted planning edits |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-nga-01 | Denial of service | a half-edited `campaign_tally.py`/`campaign_tables.py` imported by the next web request or tick | high | mitigate | Tests are written first. Constants and helpers are added before use, and each edit leaves the module importable and callable. The import smoke check runs after every production edit (Task 1 verify 1, Task 2 verify 1). |
| T-nga-02 | Denial of service | anonymous campaign table/list load: class-wide runs now iterate their linked records on a cache miss | low | mitigate | No query is added: T9 pins one query per run. Per-run results stay behind the unchanged TTL cache. The existing roll-up and list query-count tests (6, 3, and 3 marginal per campaign) still pass. `MAX_TABLE_PER_PAGE` still caps an anonymous `?per_page=`. |
| T-nga-03 | Tampering (integrity of a public figure) | zero tallies cached under the old rule in the shared `FileBasedCache` | medium | mitigate | `TALLY_CACHE_KEY_VERSION = 'v2'` sits in both key formats (finding 5), pinned by `test_keys_carry_the_counting_rule_version`. Old entries are unreachable and expire on their own TTL. |
| T-nga-04 | Tampering (XSS / input handling) | badge and Progress markup; stored `observed_site` values | low | mitigate | All markup still goes through `format_html` with positional arguments, and no `|safe`/`mark_safe` is added. A stored site string is only ever a lookup key into the fixed `_NIGHT_SITE_TIMEZONES` map. It is never rendered and never passed to `ZoneInfo`, so only the map's own values are. A non-string or unknown value falls back (T7). `src/templates/` stays unedited (Task 1 verify 4, Task 2 verify 4). |
| T-nga-05 | Information disclosure | per-record site read on a public page | low | accept | Only counts are rendered; no new field reaches the page. The records query keeps its explicit `.only()` columns, and the T-37-09 contact-field discipline is untouched. |
| T-nga-06 | Tampering | notebook re-execution writing to `src/fomo_db.sqlite3`, or a copy torn mid-tick | high | mitigate | The setup cell `219a2c08` and teardown cell `cef92204` are untouched. The Task 3 gate reads the resolved `fomo-notebook-db-` path from stored output. The executor starts only after an END banner, never exports `FOMO_DATABASE_PATH`, and runs no command that writes to the live database (finding 10). |
| T-nga-07 | Tampering (integrity) | the tally-only timezone map leaking into attribution, gap analysis or proposal allocation, or `LCO_SITE_CODE_TO_OBSCODE` being edited to widen coverage | medium | mitigate | The map is private to `campaign_tally.py` (finding 2). M2 pins that the three consumers never reference it and that the obscode table is unchanged. Task 1 verify 4 pins `campaign_attribution.py`, `campaign_gap.py`, `proposal_allocation.py` and `calendar_utils.py` unchanged. |
| T-nga-08 | Repudiation / Tampering | a task commit sweeping the operator's uncommitted `.planning/` edits or untracked files into history | medium | mitigate | Every commit stages by explicit path after `git branch --show-current`. Task 3's last gate shows the four operator files still unstaged. |
| T-nga-SC | Tampering | package installs | low | accept | This plan installs no npm, pip or cargo package; `nbconvert` is already installed. |
</threat_model>

<verification>
- **Task 1:** the cache-key test, T1-T11, M1, M2, V1 and the renamed site-unset test pass. test_campaign_tally, test_campaign_views, test_campaign_gap and test_calendar_display_extras pass. The obscode tables, the gap ladder, proposal allocation, the shared helpers and the templates are untouched.
- **Task 2:** B1-B4 pass, and test_campaign_views and test_campaign_approval pass. Every table badge uses a `text-bg-*` class, and the templates are untouched.
- **Task 3:** the lifecycle notebook was re-executed on a scratch copy and ends `PASS: F13/F14`. The runbook's tally section carries the three edits. Ruff, ruff-format and Sphinx are clean, and the full suite passes.
- **After landing (operator):** the live campaign page shows real night counts for runs 69-75, a readable badge and two-line Progress cells (Task 3 human-check).
</verification>

<success_criteria>
- **Night counting.** A class-wide run's tally counts its nights: each record is keyed by its observed site's timezone from the tally-only map, otherwise by the run's site, otherwise by the UTC date. Each night date counts once, and a night straddling midnight UTC at cpt, lsc or tfn counts once.
- **Unchanged behaviour.** A single-site run's result is unchanged when its records were observed there. The function never raises and still makes one query per run.
- **Cache and roll-up.** Both cache keys carry `v2`. The roll-up sums the corrected per-run tallies with no logic change.
- **Scope of the map.** The map lives only in `campaign_tally.py`. `LCO_SITE_CODE_TO_OBSCODE` and its consumers are unchanged, and the obscode-sets idea is recorded for v2.5 only.
- **Table.** Every badge on the campaign table is readable under Bootstrap 5.3, and the Progress cell is two unbreakable lines.
- **Delivery.** The paired notebook and runbook are updated in this plan's own commits, and the full suite passes. Only the plan's six files are committed, in three commits.
</success_criteria>

<output>
Create `.planning/quick/261006-nga-fix-f13-and-f14-the-public-campaign-tall/261006-nga-SUMMARY.md` when done
</output>
