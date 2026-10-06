---
phase: 261006-lsf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/campaign_reconciler.py
  - solsys_code/allocation_projector.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/test_calendar_display_extras.py
  - solsys_code/tests/test_calendar_template.py
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
autonomous: true
requirements: [ALLOC-01, PROJ-06, UNUSED-01]

estimate:
  tokens: 110000
  raw_tokens: 110000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "A per-target queue run shaped like the live KEY2026B-004 runs (source `lco_queue`, `telescope_class='1m0'`, no site, target `10P`, `telescope_instrument='LCO 1m0 / Sinistro — 10P'`, `proposal_code='KEY2026B-004'`, window 2026-08-01..2027-01-31) gets one `RUN:{pk}` container whose `proposal` is `KEY2026B-004` and whose title is exactly `10P — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)`. The target appears once, first. A cancelled or weathered run's status marker still comes before it. A run with no target keeps today's title form."
    - "Every `ALLOC:{pk}:{night}` night carries `run.proposal_code` on all four write paths: a new night, a re-minted night, a plain label refresh (including a declined retirement), and a legacy `RUN:{pk}:{night}` re-key. A blank code writes a blank proposal. Night titles stay `<telescope> <instrument>`; only the container title leads with the target."
    - "No churn. A second reconcile on unchanged input reports `unchanged`, writes nothing and leaves `modified` alone, and `--dry-run` agrees. When an existing event's proposal (or a container's title) is first filled, that reconcile reports one `updated` for it, and the next one reports `unchanged`."
    - "On the calendar, a container or night whose run has a proposal code is drawn in that proposal's colour and listed under that code in the colour legend. The grey empty-proposal legend entry reads `No proposal recorded`. The CSS classes (`.cal-event-classical` and the rest) and template tag names (`visible_classical_telescopes`, `neutral_slot_color`) stay as they are, and `src/templates/` is not edited."
    - "Each module still writes only the events it owns. `campaign_reconciler.py` writes only `RUN:` keys and `allocation_projector.py` only `ALLOC:` keys, both through the shared no-churn writers in `calendar_utils.py`. `observation_projector.py` is not touched."
    - "The paired notebook `reconcile_campaign_runs_demo.ipynb` was re-executed top to bottom on its scratch copy. Its stored output shows a container carrying its proposal under a target-first title, nights carrying their run's code, a no-churn second reconcile, the legend listing the code alongside `No proposal recorded`, and, on the copied real data, every container and night of a run with a code carrying that code. The runbook says the same."
  artifacts:
    - "solsys_code/campaign_reconciler.py: a module constant `TARGET_TITLE_SEPARATOR = ' — '` and a private helper `_container_label(run)` that `event_title()` now builds its base from. `_write_container_event()` writes `'proposal': run.proposal_code`. The module, `event_title()` and `_write_container_event()` docstrings are updated."
    - "solsys_code/allocation_projector.py: `_mint_fields()` and `_label_fields()` each write `'proposal': run.proposal_code`. The legacy re-key builds its fields with `_label_fields(run, dark_line)` instead of an inline dict. The docstrings that list the label fields now include `proposal`."
    - "solsys_code/templatetags/calendar_display_extras.py: `NO_PROPOSAL_LABEL = 'No proposal recorded'` replaces the old constant, with updated wording in its comment and in the docstrings/comments that describe what an empty proposal means."
    - "solsys_code/tests/test_campaign_reconciler.py: a new class `TestContainerProposalAndTargetTitle` with 10 tests. solsys_code/tests/test_allocation_projector.py: a new class `TestAllocationNightCarriesProposal` with 7 tests."
    - "solsys_code/tests/test_calendar_display_extras.py and solsys_code/tests/test_calendar_template.py: tests import and assert the new label; one new test pins its exact text."
    - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb: one new markdown+code pair between cells `db8b9701` and `a5619fed`, a `proposal=` print in `31a60753`, and prose in `8b703ea8`, `dbe67c97` and `d2adacf8`. Re-executed with output."
    - "docs/runbooks/telescope_runs_calendar.rst: three edits covering the bracketed-token paragraph, the colour legend sentence, and the entry-title paragraph with a new proposal-colour paragraph."
  key_links:
    - "`_write_container_event()` builds `fields` (now with `proposal`) and passes it to `calendar_utils.insert_or_create_calendar_event()` on create or `update_calendar_event_key_and_fields()` on update. Both end in `_update_or_unchanged()`, which compares every field, so the no-churn contract covers `proposal` with no change to `calendar_utils.py`. The dry-run path passes the SAME `fields` to `preview_calendar_event_action()`, so a preview cannot disagree with the real write."
    - "In `allocation_projector.project_allocation()`, a new night (~line 1520) and a re-mint (~line 1413) both write `_mint_fields()`. A plain update and a declined retirement both go through `_refresh_labels()`, which uses `_label_fields()` (~line 1084). The legacy re-key (~line 1338) uses `_label_fields()` once this plan lands. Those two builders are the only places `proposal` is set, so all four paths agree."
    - "`CalendarEvent.proposal` feeds `calendar_display_extras.proposal_color()`, `visible_proposals()` and `visible_classical_telescopes()`, which `src/templates/tom_calendar/partials/calendar.html` (~lines 244-350) uses for the fill colour, the `bg_color == neutral_color` check that adds `.cal-event-classical`, and both legends. Filling the field is the whole display fix; the template does not change."
    - "`status_vocabulary.state_for_title()` and `status_border_css()` read a title's status-marker PREFIX, so `event_title()` keeps the marker first, as in `[C] 10P — ...`."
---

<objective>
Fix intent-review finding F12 (`.planning/v2.4-INTENT-REVIEW.md`, "### F12 (2026-10-06)", developer decision: fix before the milestone close). On the live September 2026 month view, the whole-window `RUN:` containers of the seven per-target KEY2026B-004 queue runs (CampaignRun pks 69-75) are grey bars reading `LCO 1m0 / Sinistr…`. They cannot be told apart, and the legend lists them under a grey swatch labelled with the v1.4 D-06 classical-schedule text. The cause: `campaign_reconciler._write_container_event()` and the allocation projector's night writers never set `CalendarEvent.proposal`, even though `CampaignRun.proposal_code` has existed since Phase 37 D-07. The calendar treats every empty-proposal event as a classical schedule line, and the container title starts with the instrument, so the month cell cuts the target off the end.

Purpose: queue observations shown without their proposal, queue runs labelled classical, and targets you can only read by clicking are misleading on the main page. The fix is display-only and damages nothing.

Output: proposal codes written onto owned events and a target-first container title (Task 1, the tracer), the relabelled legend entry (Task 2), and the paired notebook and runbook plus the full-suite gate (Task 3). No migration: `CalendarEvent.proposal` (tom_calendar, `CharField(max_length=200, blank=True, default='')`) and `CampaignRun.proposal_code` (`max_length=100`) both already exist.

**Planning-time findings. Read these before starting; each one changes how a task is done.**

1. **The cron job runs from THIS checkout, so every saved production edit is live code at the next tick.** crontab: `*/15 * * * * /usr/bin/flock ... /home/tlister/git/fomo_devel/manage.py run_unattended`, whose reconcile step calls `reconcile_run()` for every run. Write the tests first (the runner never imports test files). Make each production edit so that the module still imports, and still works when called, the moment that edit is saved; run the import smoke check (Task 1's first `<automated>`) after every production edit. A read-only planning query (`file:src/fomo_db.sqlite3?mode=ro`) found that only runs 1 (`LCO2026A-003`, per-night, 10 `ALLOC:1:*` nights), 69-75 (`KEY2026B-004`, containers `RUN:69`..`RUN:75`) and 76 (`117.2A2N.001`, 4 `ALLOC:76:*` nights) have a code or a target, and that all 30 live `ALLOC:` events and all 22 live bare `RUN:` containers have `proposal = ''`. So the first tick after Task 1 should report about 21 one-time `updated` (7 containers + 10 + 4 nights) and nothing else. The live confirmation is the operator's (Task 3 human-check). Do not run anything against the live database yourself.
2. **The title rule, decided here (the orchestrator asked the plan to name it).** When a run has a target with a non-blank name, the container label is `{name}{TARGET_TITLE_SEPARATOR}{rest}`. `TARGET_TITLE_SEPARATOR` is `' — '` (space, U+2014 em dash, space). `rest` is `run.telescope_instrument`, trimmed, with exactly one trailing `' — {name}'` removed if present; the match is exact and case-sensitive. If removing it would leave nothing, `rest` stays unstripped. Why this rule: the planning query showed all seven live per-target runs were created as `LCO 1m0 / Sinistro — {target.name}` (targets 10P, 112P, 11P, 169P, 220P, 248370, 259P), so this exact suffix is the only duplicate that occurs. Matching on separator plus name means target `11P` never strips run 70's `— 112P`, and the em dash and space never eat part of a real instrument name. The two broader options were removing the name anywhere in the string, or accepting any dash; both were rejected because they can eat into a legitimate instrument string. A run with no target, or a blank target name, keeps exactly today's label. The `(window a..b)` suffix and the status-marker prefix are unchanged. The event's separate `telescope`/`instrument` fields (`split_telescope_instrument()`, so `'LCO 1m0'` / `'Sinistro — 10P'`) are NOT changed: they are outside F12's scope.
3. **Only the container title changes.** `allocation_night_title()` (allocation_projector.py ~197) stays `<telescope> <instrument>` (D-12: per-night titles carry no window, and the classical loader's own form is matched). A per-night run with a target, such as run 1 (target 65803), keeps its night titles.
4. **Proposal value: `run.proposal_code` verbatim**, the same way `observation_projector` writes the record's `proposal` parameter. A blank code stays blank. No truncation is needed (100 is within the 200-character limit). Do not strip it or upper-case it: the legend's `proposal_color()` already normalises for colour and grouping.
5. **Every owned write path gets the field, through the two existing builders.** `_mint_fields()` (create and re-mint) and `_label_fields()` (plain update, retirement decline, and the legacy re-key, which this plan switches from an identical inline dict to `_label_fields(run, dark_line)`) are the only builders in `allocation_projector.py`. `proposal` is a label, in the same class as `title`/`description`/`target_list`. It is refreshed on update and is NOT a boundary input, so `TestMintInputInvariant`, the provenance token and `_span_needs_remint()` are untouched. `campaign_reconciler._write_container_event()` is the only container builder.
6. **Legend wording only.** Rename the constant (it is referenced only by `calendar_display_extras.py` and `test_calendar_display_extras.py`; the template never names it). Do NOT rename `visible_classical_telescopes`, `neutral_slot_color` or any `.cal-event-classical*` CSS class, and do not edit `src/templates/`. Comments that name the `.cal-event-classical` CSS hook or the stripe palettes (calendar_display_extras.py ~96-140) stay as they are. Only wording that says what an EMPTY PROPOSAL means is updated.
7. **Paired docs and their scope.** CLAUDE.md maps `campaign_reconciler.py` and `allocation_projector.py` to `reconcile_campaign_runs_demo.ipynb`, which this plan updates and re-executes. `campaign_reconciler.py` is also part of the collective v2.2 surface mapped to `campaign_lifecycle_demo.ipynb`. That notebook is deliberately NOT re-executed, because none of its runs carries a `target` or a `proposal_code` (verified at planning time: its only `target=` is an `ObservationRecord`'s, its runs come from `client.post` web submissions and one `CampaignRun.objects.create`, and neither the web form nor `campaign_utils` sets a run target). Its stored titles and outputs are therefore exactly what the fixed code produces. Task 3 has a gate that proves this still holds. `docs/runbooks/telescope_runs_calendar.rst` is updated (Task 3). It never contained the old legend text, but it describes the entry title and the classical bracketed token, both of which this fix changes.
8. **The notebook never renders the calendar page.** `visible_proposals()` reads only `.proposal` on the objects it is given and calls no `reverse()`, so the notebook can call it directly. Never call `campaign_decoration()`, `render_calendar` or anything else that resolves URLs: `src/fomo/urls.py` imports `solsys_code.views`, which imports `ephem_utils` and starts the SPICE kernel download.
9. **Files that are not yours.** `.planning/v2.4-INTENT-REVIEW.md`, `.planning/v2.4-MILESTONE-AUDIT.md`, `.planning/state.json`, `.planning/milestones/v1.1-phases/03-classical-calendar-ingest/03-VERIFICATION.md` have the operator's uncommitted edits, and there are untracked files (`.planning/agent-history.json`, `reqgroup_2682493.json`, `src/fomo_db_*.sqlite3`, a `.gitkeep`). Do not edit, stage or commit any of them. The F12 "Fix landed" note is the orchestrator's job, not this plan's. Run `git branch --show-current` before the first commit; it must print `issue37-telescope-runs-calendar`. Stage every commit by explicit path.
10. **Allowed commands.** `python manage.py test ...` (never `./manage.py`), `pre-commit run ...`, `jupyter nbconvert ...`, `git`, read-only `python -c` checks, and the import smoke check (which calls `django.setup()` and imports modules but opens no database connection). No other `python manage.py` command, no network or LCO portal call, and nothing that opens `src/fomo_db.sqlite3` for writing. The notebook's setup cell reading it with `shutil.copy2` is the only permitted access.
11. **Code reading.** Prefer Serena's symbolic tools (`get_symbols_overview`, `find_symbol`) where available; otherwise use Read/Grep with offsets. Line numbers in this plan are from planning time (~).

Source coverage audit. GOAL (F12, from ROADMAP-close intent review): owned events carry their run's proposal; the container title leads with the target; the empty-proposal legend says what it means → Tasks 1, 1, 2; paired docs → Task 3. REQ: ALLOC-01 (allocation nights and whole-window containers, now with their proposal) → Task 1; PROJ-06 (a compact title that fits a month cell, so the target comes first) → Task 1; UNUSED-01 (the calendar legend, where the empty-proposal entry is relabelled) → Task 2. CONTEXT (orchestrator required changes 1-3, the tests to pin, paired docs, executor constraints) → change 1: Task 1 production edits A and B; change 2: Task 1 edit A, with the rule in finding 2; change 3: Task 2; tests to pin: Task 1 (R1-R10, A1-A7) and Task 2; paired docs: Task 3; constraints: findings 1, 9 and 10, and each task's verify. The old D-06 (v1.4) label is superseded on purpose by the developer's 2026-10-06 F12 decision. RESEARCH: none (no research phase). Planner contributions: security → `<threat_model>`; schema-gate → no Payload/Prisma/Drizzle/Supabase/TypeORM file is in scope, and no Django model or migration changes either; the API-coverage detector matched this sentence's own wording, so a reasoned "No external API integration" declaration is in `COVERAGE.md` next to this plan and in `<api_coverage_decision>`; the assumption-delta probe was skipped (`phase_unresolved`), see `<assumption_delta_decision>`. Nothing is unplanned. F6, F8, F10 and F11 are out of scope and must not be touched.
</objective>

<api_coverage_decision>
No external API integration: this task changes which fields FOMO's own reconciler and allocation projector write onto local `CalendarEvent` rows, plus one template-tag label and docs. It calls no external API, SDK or service, and adds none.
</api_coverage_decision>

<assumption_delta_decision>
Detector result: skipped (`phase_unresolved`, because a quick-task id has no ROADMAP section), so the checkpoint did not fire. This record is kept voluntarily because the change does give the empty-proposal slot a new meaning. Noun that is primary: `CalendarEvent.proposal`, the event's proposal identity. The empty value now means "no proposal code recorded", no longer "classical schedule line". Decision: `promote`. The empty-proposal slot's meaning moves from the old specific case (classical line) to the general one (no code recorded), and the legend label is renamed to match. Nothing is added alongside the old meaning. Invariant: `TestContainerProposalAndTargetTitle.test_container_is_legended_under_its_proposal` makes sure a run-owned event with a code never falls into the grey entry again.
</assumption_delta_decision>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@solsys_code/campaign_reconciler.py
@solsys_code/allocation_projector.py
@solsys_code/calendar_utils.py
@solsys_code/templatetags/calendar_display_extras.py
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1 (tracer): a run's proposal code and target reach its own RUN: container and ALLOC: nights, end to end into the legend</name>
  <files>solsys_code/campaign_reconciler.py, solsys_code/allocation_projector.py, solsys_code/tests/test_campaign_reconciler.py, solsys_code/tests/test_allocation_projector.py</files>
  <read_first>
    - .planning/v2.4-INTENT-REVIEW.md, section "### F12 (2026-10-06)" only (read; never edit)
    - solsys_code/campaign_reconciler.py: module docstring (~1-47), `RUN_URL_NAMESPACE` (~72), `split_telescope_instrument()` (~173), `event_title()` (~203-217), `_write_container_event()` (~344-376)
    - solsys_code/allocation_projector.py: `allocation_night_title()` (~197), the docstring near ~492 that lists label fields, `_mint_fields()` (~875-901), `_label_fields()` (~1020-1040), `_refresh_labels()` (~1043-1086), the `project_allocation()` docstring "Field authority" (~1089-1115), the legacy re-key block (~1325-1350), the comment near ~1424
    - solsys_code/calendar_utils.py `_update_or_unchanged()`, `insert_or_create_calendar_event()`, `update_calendar_event_key_and_fields()`, `preview_calendar_event_action()` (~613-747): the no-churn contract this plan relies on, unchanged
    - solsys_code/tests/test_campaign_reconciler.py `CampaignReconcilerTestBase` (~42-118) and `TestContainerIdempotency` (~474-518)
    - solsys_code/tests/test_allocation_projector.py `AllocationProjectorTestBase` (~39-115), `test_legacy_run_keyed_night_is_rekeyed_in_place` (~382-410), `TestMintInputInvariant.test_changing_the_site_remints` (~3315-3330)
  </read_first>
  <behavior>
    New class `TestContainerProposalAndTargetTitle(CampaignReconcilerTestBase)`, appended at the end of solsys_code/tests/test_campaign_reconciler.py. Its helper `_make_queue_run(**overrides)` calls `self._make_run()` with the live shape: `source=CampaignRun.Source.LCO_QUEUE`, `telescope_class=CampaignRun.TelescopeClass.ONE_M0`, `site=None`, `site_raw=''`, `campaign=None`, `telescope_instrument='LCO 1m0 / Sinistro — 10P'`, `window_start=date(2026, 8, 1)`, `window_end=date(2027, 1, 31)`, `proposal_code='KEY2026B-004'`, and `target=NonSiderealTargetFactory.create(name='10P')` (CLAUDE.md: never a sidereal factory). Overrides take precedence.
    - R1 `test_container_carries_run_proposal_code`: after `reconcile_run(run)`, the event at `RUN:{pk}` has `proposal == 'KEY2026B-004'`.
    - R2 `test_blank_proposal_code_writes_blank_proposal`: `proposal_code=''` gives `proposal == ''`.
    - R3 `test_title_leads_with_target_without_repeating_it`: the title equals `'10P — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)'`, `title.count('10P') == 1`, and `event_title(run)` returns the same string.
    - R4 `test_title_leads_with_target_when_instrument_has_no_target_suffix`: `telescope_instrument='FTN/MuSCAT3'` gives `'10P — FTN/MuSCAT3 (window 2026-08-01..2027-01-31)'`.
    - R5 `test_title_without_target_keeps_current_form`: `target=None`, `telescope_instrument='LCO 1m0 / Sinistro'` gives `'LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)'`.
    - R6 `test_status_marker_stays_first`: `run_status=CampaignRun.RunStatus.CANCELLED` gives `RUN_STATUS_MARKER[CampaignRun.RunStatus.CANCELLED] + ' 10P — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)'` (import `RUN_STATUS_MARKER` from `solsys_code.status_vocabulary`).
    - R7 `test_single_day_window_has_no_window_suffix`: `window_end=date(2026, 8, 1)` gives `'10P — LCO 1m0 / Sinistro'`.
    - R8 `test_second_reconcile_is_unchanged`: the first reconcile has `created == 1`. The second has `unchanged == 1` and `created == updated == 0`, and the event's `modified` is unchanged. `reconcile_run(run, dry_run=True)` reports `unchanged == 1`.
    - R9 `test_filling_a_blank_code_updates_once_then_is_unchanged`: create with `proposal_code=''` and reconcile. Then set `run.proposal_code = 'KEY2026B-004'` and `run.save(update_fields=['proposal_code'])`. A dry run reports `updated == 1` and the stored proposal is still `''`. The real reconcile reports `updated == 1`, the proposal is now set and the event pk is the same. A third reconcile reports `unchanged == 1`.
    - R10 `test_container_is_legended_under_its_proposal` (the end-to-end tracer leg): after reconcile, `visible_proposals([[SimpleNamespace(all_day_events=[event], events=[])]])` (import from `solsys_code.templatetags.calendar_display_extras`, along with `NEUTRAL_SLOT_COLOR`; `SimpleNamespace` from `types`) returns exactly one entry whose `codes == ['KEY2026B-004']` and whose `color != NEUTRAL_SLOT_COLOR`.
    New class `TestAllocationNightCarriesProposal(AllocationProjectorTestBase)`, appended at the end of solsys_code/tests/test_allocation_projector.py, uses the base `_make_run()` (classical_file, Chilean site `809`, `NTT/EFOSC2`, 3 nights 2026-07-09..11):
    - A1 `test_every_night_carries_run_proposal_code`: `proposal_code='117.2A2N.001'`; all 3 `allocation_events(run)` have that proposal.
    - A2 `test_blank_code_writes_blank_proposal`: every night's proposal is `''`.
    - A3 `test_second_reconcile_is_unchanged`: with a code, the first reconcile has `created == 3`. The second has `unchanged == 3`, `created == updated == 0`, and every night's `modified` is unchanged.
    - A4 `test_filling_a_blank_code_updates_each_night_once`: create with `''` and reconcile. Set the code and `save(update_fields=['proposal_code'])`. A dry run reports `updated == 3`. The real reconcile reports `updated == 3`, every night carries the code, and the set of night pks is unchanged (no re-mint). A third reconcile reports `unchanged == 3`.
    - A5 `test_rekeyed_legacy_night_carries_proposal`: copy the setup of `test_legacy_run_keyed_night_is_rekeyed_in_place` with a single night and `proposal_code='117.2A2N.001'`. The result has `rekeyed == 1`; the `ALLOC:` event keeps the legacy pk and has `proposal == '117.2A2N.001'`.
    - A6 `test_night_title_is_not_led_by_a_target`: `target=NonSiderealTargetFactory.create(name='65803')`; every night's title is `'NTT EFOSC2'`.
    - A7 `test_reminted_night_carries_proposal`: single night with a code; reconcile. Then follow `TestMintInputInvariant.test_changing_the_site_remints`: switch `site` to `self.australian_site` with `site_raw='E10'` and `save(update_fields=['site', 'site_raw'])`. The reconcile reports `retired == 1` and `created == 1`, and the new event's proposal is the code.
  </behavior>
  <action>
    **Step 0.** Run `git branch --show-current`; it must print `issue37-telescope-runs-calendar`.

    **Step 1: RED (tests only; the runner never imports test files).** Write the two new test classes from `<behavior>` at the ends of the two test modules, adding only the imports they need (`SimpleNamespace`, `RUN_STATUS_MARKER`, `visible_proposals`, `NEUTRAL_SLOT_COLOR` in the reconciler test module; nothing new should be needed in the allocation one). Run the second `<automated>` command restricted to the two new classes (labels `solsys_code.tests.test_campaign_reconciler.TestContainerProposalAndTargetTitle solsys_code.tests.test_allocation_projector.TestAllocationNightCarriesProposal`). Expected RED: R1, R3, R4, R6, R7, R9, R10, A1, A4, A5 and A7 fail with an AssertionError. R2, R5, R8, A2, A3 and A6 already pass; they guard the behaviour that must not change. If any failure is an ImportError, NameError, fixture error or IntegrityError instead, fix the test before going on. If the pass/fail split differs, work out why before writing production code.

    **Step 2: GREEN, campaign_reconciler.py (per finding 1, each Edit leaves the module importable and working; run the import smoke check after each).**
    - Edit A1: below `RUN_URL_NAMESPACE`, add the module constant `TARGET_TITLE_SEPARATOR = ' — '` with a one-line comment: it is the separator the per-target queue runs were created with and the one the container title writes (F12, quick task 261006-lsf). Also add the private helper `_container_label(run: CampaignRun) -> str` just above `event_title()`, implementing finding 2 exactly. Return `run.telescope_instrument` unchanged when `run.target_id is None` (checked first, so no query) or when the target's name stripped is empty. Otherwise strip trailing whitespace from the text, remove exactly one trailing `TARGET_TITLE_SEPARATOR + name` when present and something remains, and return `name + TARGET_TITLE_SEPARATOR + rest`. Give it a Google-style docstring stating the rule and why the duplicate exists: the live runs carry the target in `telescope_instrument` to keep `unique_campaign_run_resolved_window` distinct.
    - Edit A2: in `event_title()`, build `base` from `_container_label(run)` instead of `run.telescope_instrument`; the window suffix and marker logic stay as they are. Rewrite the docstring to say the title leads with the run's target when it has one (F12), so the month cell shows the target first, keeps the marker first for the status ring, and still carries no campaign label (D-12).
    - Edit A3: in `_write_container_event()`, add `'proposal': run.proposal_code` to `fields`. Add a sentence to the docstring: the container carries its run's proposal code, so it takes that proposal's colour and legend entry (F12); a blank code stays blank.
    - Edit A4 (docstring only): in the module docstring's field-authority paragraph (~39-46), the allocation projector's update refresh list becomes `title`/`description`/`target_list`/`proposal`.

    **Step 3: GREEN, allocation_projector.py (same discipline as Step 2).**
    - Edit B1: add `'proposal': run.proposal_code` to `_mint_fields()`'s return dict and to `_label_fields()`'s. In `_label_fields()`'s docstring, "three non-destructive label fields" becomes four, naming `proposal`, and the Returns line lists it.
    - Edit B2: in the legacy re-key block (~1338-1342), replace the inline `rekey_fields` dict with `rekey_fields: dict[str, Any] = _label_fields(run, dark_line)`. Its three keys are identical to `_label_fields()`'s today; this gives the re-key the fourth key and keeps one builder (finding 5).
    - Edit B3 (docstrings and comments only): every place that lists the label set as `title`/`description`/`target_list` gains `proposal`: the docstring near ~492, `_refresh_labels()`, `project_allocation()`'s "Field authority" text, and the comment near ~1424. `grep -n target_list solsys_code/allocation_projector.py` finds each one. Do NOT change `allocation_night_title()` (finding 3).

    **Step 4: confirm.** Run every `<automated>` command below. If a PRE-EXISTING test fails, stop and report it. The only exception is a test whose assertion pins the F12 defect itself: the old title form for a container whose run has a target, or an empty proposal on a `RUN:`/`ALLOC:` event whose run has a code. You may update such a test, and must name it in the SUMMARY. Record each module's `Ran N tests` count.

    **Step 5: commit** these four files by explicit path, as `fix(261006-lsf): write run.proposal_code onto RUN:/ALLOC: events and lead the container title with the target`.
  </action>
  <verify>
    <automated>python -c "import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings'); import django; django.setup(); import solsys_code.campaign_reconciler as r, solsys_code.allocation_projector as a, solsys_code.templatetags.calendar_display_extras as d; print('importable:', r.event_title.__name__, a._label_fields.__name__, d.visible_proposals.__name__)"</automated>
    <automated>python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_allocation_projector solsys_code.tests.test_write_and_reconcile solsys_code.tests.test_null_campaign_guards solsys_code.tests.test_campaign_approval solsys_code.tests.test_reconcile_campaign_runs --noinput --parallel 4</automated>
    <automated>pre-commit run ruff --files solsys_code/campaign_reconciler.py solsys_code/allocation_projector.py solsys_code/tests/test_campaign_reconciler.py solsys_code/tests/test_allocation_projector.py && pre-commit run ruff-format --files solsys_code/campaign_reconciler.py solsys_code/allocation_projector.py solsys_code/tests/test_campaign_reconciler.py solsys_code/tests/test_allocation_projector.py</automated>
    <automated>git diff --quiet HEAD -- solsys_code/observation_projector.py solsys_code/calendar_utils.py src/templates/ && echo "OK: observation projector, shared writer and templates untouched"</automated>
  </verify>
  <done>
    All 17 new tests pass and every pre-existing test in the six listed modules passes, with any F12-defect test updated and named. A KEY2026B-004-shaped run's container reads `10P — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)` with proposal `KEY2026B-004`, and `visible_proposals()` lists it under that code, not the grey entry. Nights carry the code on all four write paths. Re-runs are no-churn. ruff and ruff-format are clean, and one commit holds exactly the four files.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: the empty-proposal legend entry reads "No proposal recorded"</name>
  <files>solsys_code/templatetags/calendar_display_extras.py, solsys_code/tests/test_calendar_display_extras.py, solsys_code/tests/test_calendar_template.py</files>
  <read_first>
    - solsys_code/templatetags/calendar_display_extras.py: module docstring (~1-29), the D-05/D-06 constants (~144-150), `visible_proposals()` (~284-344), `neutral_slot_color()` (~347-357), `visible_classical_telescopes()` (~434-483)
    - solsys_code/tests/test_calendar_display_extras.py: imports (~15-40) and `VisibleProposalsTest` (~157-205)
    - solsys_code/tests/test_calendar_template.py: `test_display07_classical_schedule_label_present_when_empty_proposal_events_visible` (~271-275) and the empty-proposal fixtures it relies on (~150-160)
    - src/templates/tom_calendar/partials/calendar.html ~244-350 (read only; confirms the tag and class names that must not change)
  </read_first>
  <behavior>
    - `NO_PROPOSAL_LABEL == 'No proposal recorded'` (new test `test_no_proposal_label_text` in `VisibleProposalsTest`).
    - `visible_proposals()` on a single empty-proposal event returns one entry whose label is `NO_PROPOSAL_LABEL` and whose colour is `NEUTRAL_SLOT_COLOR`, ordered last after a coded entry. These are the existing tests, renamed and rewired to the new constant.
    - The rendered calendar page contains `No proposal recorded` and no longer contains the v1.4 classical-schedule legend text (the existing template test, renamed, with an added `assertNotIn`).
  </behavior>
  <action>
    Per the developer's F12 decision (2026-10-06), which supersedes the v1.4 D-06 label wording: the D-05 neutral slot and its ordering are unchanged.

    **Step 1: RED.** In solsys_code/tests/test_calendar_display_extras.py, import `NO_PROPOSAL_LABEL` instead of the old constant and use it in `test_groups_by_color_with_collision_handling`'s expected-label line. Rename `test_neutral_slot_label_is_classical_schedule` to `test_neutral_slot_label_is_no_proposal_recorded`, update its comment and the comment in `test_neutral_slot_ordered_last`, and add `test_no_proposal_label_text`, which asserts the exact string. In solsys_code/tests/test_calendar_template.py, rename `test_display07_classical_schedule_label_present_when_empty_proposal_events_visible` to `test_display07_no_proposal_label_present_when_empty_proposal_events_visible`, update its docstring, assert `'No proposal recorded'` is in the content, and add an `assertNotIn` for the old two-word label text. Run the second `<automated>` command. Expected RED: test_calendar_display_extras fails to import the new name, and the renamed template test fails its `assertIn`.

    **Step 2: GREEN (the module stays importable and callable after each Edit, because the web server may load it).** (a) Add `NO_PROPOSAL_LABEL = 'No proposal recorded'` directly below the old constant. Its comment says: D-06's empty-proposal legend label, relabelled by F12 (quick task 261006-lsf, 2026-10-06), because `RUN:` containers and `ALLOC:` nights now carry their run's proposal code, so an empty proposal means only that no code is recorded (a run with a blank code, a hand-entered event, a record with no proposal parameter). (b) Switch `visible_proposals()`'s label line to the new constant. (c) Delete the old constant and its comment. (d) Wording: in the module docstring's `visible_classical_telescopes` bullet, the `visible_proposals()` docstring (the "appear as ..." sentence and the Returns text), the `neutral_slot_color()` docstring, and the `visible_classical_telescopes()` docstring and its inline `continue` comment, describe empty-proposal events as having "no proposal recorded" rather than as classical-schedule lines. Keep the function names, CSS class names and the stripe/palette comments (finding 6). Do not edit the template.

    **Step 3.** Run every `<automated>` command below, then commit the three files by explicit path, as `fix(261006-lsf): relabel the empty-proposal calendar legend entry "No proposal recorded"`.
  </action>
  <verify>
    <automated>python -c "import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings'); import django; django.setup(); import solsys_code.templatetags.calendar_display_extras as d; assert d.NO_PROPOSAL_LABEL == 'No proposal recorded'; assert not hasattr(d, 'CLASSICAL_SCHEDULE_LABEL'); assert callable(d.visible_classical_telescopes) and callable(d.neutral_slot_color); print('OK: label renamed, template hooks intact')"</automated>
    <automated>python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template --noinput --parallel 4</automated>
    <automated>pre-commit run ruff --files solsys_code/templatetags/calendar_display_extras.py solsys_code/tests/test_calendar_display_extras.py solsys_code/tests/test_calendar_template.py && pre-commit run ruff-format --files solsys_code/templatetags/calendar_display_extras.py solsys_code/tests/test_calendar_display_extras.py solsys_code/tests/test_calendar_template.py</automated>
    <automated>git diff --quiet HEAD -- src/templates/ && grep -c "visible_classical_telescopes" src/templates/tom_calendar/partials/calendar.html</automated>
  </verify>
  <done>
    The legend's grey entry reads `No proposal recorded`. The new constant is the only label definition. Function, tag and CSS names are unchanged and the template is untouched. Both test modules pass, lint is clean, and one commit holds exactly the three files.
  </done>
</task>

<task type="auto">
  <name>Task 3: paired notebook and runbook show proposal-coloured, target-first entries; full-suite gate</name>
  <files>docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst</files>
  <read_first>
    - CLAUDE.md "Paired docs are part of the deliverable" (the map entries for `campaign_reconciler.py`/`allocation_projector.py`)
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb cells `8b703ea8` (intro bullets), `9084663a` (scratch setup: read, never edit), `8ed4f673` (seeds `ground_site`-based runs and `class_wide_run`), `dbe67c97` and `31a60753` (real-sweep print loop), `9a9bfc44` and `db8b9701` (the F5 pair, style to follow), `a5619fed`, `d2adacf8` (Summary), `d5248b35` (teardown: never edit). Use the cell ids; positions may shift.
    - docs/runbooks/telescope_runs_calendar.rst: "The optional bracketed proposal token" (~46-60), the status-legend paragraph ending "colour alone." (~248-256), and in "How do I get every campaign run onto the calendar?" the paragraph beginning "The run's free-text ``Telescope / Instrument`` value is split" (~1745-1756)
  </read_first>
  <action>
    **Part A: notebook sources** (edit the JSON cell sources; keep every existing cell id; new cells get fresh short hex ids).
    - `8b703ea8`: add a final bullet, "A run's own proposal code and target on its calendar entries (F12, quick task 261006-lsf)". It says that a per-target queue run's whole-window container leads with its target and carries the run's proposal code, as do a classical run's `ALLOC:` nights, so both take that proposal's colour and legend entry, and that an entry with no code recorded is listed under `No proposal recorded`.
    - `dbe67c97`: add one sentence saying each event now also prints its `proposal`, which is blank for all four seeded runs because none carries a code; the F12 demo further down sets one.
    - `31a60753`: add a `proposal=` print line (`ev.proposal!r`) directly after the title print. Change nothing else.
    - Insert a new markdown cell and a new code cell between `db8b9701` and `a5619fed`.
      - The markdown heading is "## A run's proposal code and target reach its own calendar entries (F12, quick task 261006-lsf)". In plain English, cover these points. Before the fix, neither the container writer nor the allocation night writer set the event's `proposal`, so every `RUN:` container and `ALLOC:` night was grey and listed under the empty-proposal legend entry, then labelled as a classical schedule, even for LCO queue runs. The container title also led with the instrument, so a month cell cut the target off. Now both writers copy `CampaignRun.proposal_code`, the container title leads with the target, and a trailing ` — <target>` already in the Telescope / Instrument text is not repeated (finding 2's rule). The legend entry for events with no code reads `No proposal recorded`. The cell calls `visible_proposals()` directly on the event rows, because rendering the page would resolve URLs and import the ephemeris views (finding 8).
      - The code cell's first line is the comment `# F12 proposal-and-target demo` (the gate looks for it). Part 1 is read-only on the real data in this scratch copy. For every `CampaignRun` with a non-blank `proposal_code`, ordered by pk, count the events in `owned_events(run)` and `allocation_events(run)` and how many carry `run.proposal_code`. Print one line per run (pk, code, target name or `-`, events, carrying), then assert every count matches. For each such run that has a target and a `RUN:{pk}` container, assert the container title contains the target name exactly once and that the name comes before the first `(window` (after any marker). Print one example title. If a real-data assertion fails, stop and report the run pk and message rather than loosening it: that is evidence. Part 2 is synthetic.
        - Create `f12_target = NonSiderealTargetFactory.create(name='F12 demo comet')`.
        - Create a container run `f12_queue_run` with `CampaignRun.objects.create`: `campaign=None`, `target=f12_target`, `telescope_instrument=f'LCO 1m0 / Sinistro — {f12_target.name}'`, `telescope_class=CampaignRun.TelescopeClass.ONE_M0`, `site=None`, `source=CampaignRun.Source.LCO_QUEUE`, window 2026-08-01..2027-01-31, `proposal_code='DEMO2026B-001'`, approved, and a short `observation_details`.
        - Create a per-night run `f12_classical_run`: `campaign=None`, `telescope_instrument='RDGS/EFOSC2 (F12 demo)'`, `site=ground_site`, `site_raw='X29'`, `source=CampaignRun.Source.CLASSICAL_FILE`, window 2026-09-20..2026-09-21, `proposal_code='DEMO.ESO-001'`, approved.
        - Call `reconcile_run()` on each and print the results.
        - Print the container's url, title, proposal, telescope and instrument, and each night's url, title and proposal.
        - Assert: the container title equals `f'{f12_target.name} — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)'`, contains the name once, and has proposal `DEMO2026B-001`; there are exactly 2 nights and both carry `DEMO.ESO-001`.
        - Reconcile both runs again, print the results, and assert `created == updated == 0` with `unchanged` equal to 1 and 2.
        - Get `class_wide_run`'s container through `run_container_url(class_wide_run)` and assert it exists and its proposal is `''`.
        - Print each entry of `visible_proposals([[SimpleNamespace(all_day_events=[container, *nights, blank_container], events=[])]])`, as label and colour.
        - Assert the last entry's label is `NO_PROPOSAL_LABEL` and its colour is `NEUTRAL_SLOT_COLOR`, and that `DEMO2026B-001` and `DEMO.ESO-001` each appear in some entry's `codes`.
        - Finish with a line starting `PASS: F12` that summarises the checks.
    - `d2adacf8`: add one short paragraph naming the F12 demo and its two proofs: the real-data check that every coded run's entries carry the code, and the synthetic target-first container, coded nights, no-churn re-run and legend.
    - Do not touch `9084663a` or `d5248b35`.

    **Part B: lint, then execute.**
    - Run `pre-commit run ruff --files docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, then `pre-commit run ruff-format --files docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, and re-run the first if the second changed anything.
    - To avoid copying the live database mid-tick, run `tail -n 3 /var/log/fomo/unattended.log` and start only when the last banner is an `=== FOMO unattended run END` line. Ticks run at :00, :15, :30 and :45 and take about 60 s.
    - From the repo root, in the FOREGROUND (Bash timeout 600000 ms; never `run_in_background`, never `&`), run `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`.
    - Never export `FOMO_DATABASE_PATH` or copy a database yourself; the setup cell does that.
    - If setup fails with "database disk image is malformed", the copy was torn mid-tick: re-run after the next END banner.
    - If the NEW cell fails, fix its source and re-execute the WHOLE notebook. Never hand-write outputs.
    - If a PRE-EXISTING cell's assertion fails, stop and report the cell id and message.
    - Expected and fine: the cutover sweep (`abb9338a`) now also reports one-time `updated` for the copy's coded or targeted runs' entries, unless a post-fix live tick already wrote them; the second sweep (`7d9dc6b8`) still reads `updated: 0`.

    **Part C: runbook** (three scoped Edits in docs/runbooks/telescope_runs_calendar.rst; plain English, matching the page's RST style).
    1. At the end of "The optional bracketed proposal token" (after the "then re-import." paragraph, before "The two summary lines a real run prints"), add a short paragraph. The token is stored as the run's **Proposal code**, and the run's calendar nights carry it: they are drawn in that proposal's colour and listed under that code in the calendar's colour legend. A line with no token gets grey nights listed under **No proposal recorded**.
    2. After the status-legend paragraph that ends "colour alone." (~256), add a sentence or two. Alongside that status legend, the calendar's colour legend lists each proposal code visible in the month in its colour, with entries that have no proposal code recorded grouped last under **No proposal recorded**. Point to the new paragraph in edit 3 by its bold lead-in.
    3. In the paragraph beginning "The run's free-text ``Telescope / Instrument`` value is split", replace the sentence "The entry's title still shows the full combined text either way." with text that says the following. A whole-window entry's title shows the combined text led by the run's target when the run has one, for example ``10P — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)``, so a month cell shows the target first. A trailing `` — <target>`` already at the end of the Telescope / Instrument text is not repeated. A run with no target keeps the combined text first, and a per-night entry's title is ``<telescope> <instrument>``. Directly after that paragraph, add a new paragraph with the bold lead-in **Every reconciler entry carries its run's proposal code.** It says:
       - Whole-window and per-night entries are written with the run's **Proposal code** (the classical loader's bracketed token, or a code a staff member enters in the admin), so they take that proposal's colour and legend entry, next to that proposal's own observation entries.
       - An entry with no code recorded (a run whose code is blank, or a hand-entered event) is drawn in the neutral grey and listed under **No proposal recorded**. Before quick task 261006-lsf (2026-10-06), every reconciler entry was grey and that legend entry was labelled as a classical schedule, even for an LCO queue run.
       - A code or target edited in the admin reaches the run's entries on the next reconcile (each unattended tick's reconcile step, ``reconcile_campaign_runs``, or a staff action on the run). That reconcile counts each affected entry once under ``updated``, and later reconciles count it as ``unchanged``.
       Change nothing else in the runbook.

    **Part D: full suite and final gates.**
    - Run the `<automated>` commands below in order. The full suite is about 2095 tests and takes about 2-3 min with `--parallel 4` (Bash timeout 600000 ms). It uses Django's test database and never the live one. Quote the `Ran N tests` / `OK` lines in the SUMMARY.
    - If any pre-existing test fails outside the F12-defect exception in Task 1 Step 4, stop and report it.
    - Commit the notebook and runbook by explicit path, as `docs(261006-lsf): demo proposal-coloured, target-first reconciler entries; runbook legend and title wording`. Check `git status --porcelain` afterwards: the four operator-owned `.planning/` files are still modified and unstaged, and the untracked files are still untracked (finding 9).
  </action>
  <verify>
    <automated>python -c "
import json, re
nb = json.load(open('docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb'))
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
i, j = ids.index('db8b9701'), ids.index('a5619fed')
assert j - i == 3, 'expected exactly one new markdown+code pair between the F5 demo and the site-correction section'
demo = nb['cells'][i + 2]
src = ''.join(demo['source'])
assert demo['cell_type'] == 'code' and src.startswith('# F12 proposal-and-target demo'), 'F12 demo cell not where expected'
assert 'campaign_decoration' not in src and 'reverse(' not in src, 'the notebook must not resolve URLs'
text = out(demo)
for token in ('F12 demo comet — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)', 'DEMO2026B-001', 'DEMO.ESO-001', 'No proposal recorded', 'PASS: F12'):
    assert token in text, f'F12 demo output lacks {token!r}'
sweep = [c for c in code if c.get('id') == '31a60753'][0]
assert 'proposal=' in out(sweep), 'real-sweep loop does not print proposal'
assert 'Removed scratch database directory' in out(code[-1]), 'last code cell is not the scratch teardown'
print(f'OK: {len(code)} code cells, one fresh run, scratch database {m.group(1)}')
"</automated>
    <automated>python -c "
t = open('docs/runbooks/telescope_runs_calendar.rst').read()
a = t.index('The optional bracketed proposal token')
b = t.index('The two summary lines a real run prints', a)
assert 'No proposal recorded' in t[a:b], 'bracketed-token paragraph lacks the legend wording'
k = t.index('Every reconciler entry carries its run')
assert '10P — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)' in t[k - 2500:k], 'target-first title example not just before the new paragraph'
assert '261006-lsf' in t[k:k + 2500] and 'updated' in t[k:k + 2500], 'new paragraph lacks the history or the one-time updated note'
assert t.count('No proposal recorded') >= 3, 'expected the legend wording in all three places'
print('OK: runbook edits in place')
"</automated>
    <automated>python -c "
import json
nb = json.load(open('docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
assert 'proposal_code' not in src, 'lifecycle notebook now sets a proposal code: its stored output may be stale (finding 7)'
assert src.count('target=') == 1 and 'target=observation_target' in src, 'lifecycle notebook now sets a run target: its stored output may be stale (finding 7)'
print('OK: campaign_lifecycle_demo exercises no run code or target, so its stored output stays accurate')
"</automated>
    <automated>pre-commit run ruff --files docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb && pre-commit run ruff-format --files docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb && pre-commit run sphinx-build --files docs/runbooks/telescope_runs_calendar.rst</automated>
    <automated>python manage.py test solsys_code --exclude-tag=ephemeris_segfault --noinput --parallel 4</automated>
    <automated>git status --porcelain .planning/v2.4-INTENT-REVIEW.md .planning/v2.4-MILESTONE-AUDIT.md .planning/state.json .planning/milestones/v1.1-phases/03-classical-calendar-ingest/03-VERIFICATION.md</automated>
    <human-check>Operator, after the first unattended tick that starts after Task 1's commit: in the September 2026 month view, the KEY2026B-004 containers read `10P — LCO 1m0 / Sinistro…` (and so on for each target) in KEY2026B-004's legend colour, not grey. Run 1's FTS nights and run 76's NTT nights are in their proposals' colours. The grey legend entry, if any grey entries are left, reads `No proposal recorded`. That tick's reconcile line should show about 21 one-time `updated`, and the next tick 0. Read-only cross-check: `sqlite3 'file:src/fomo_db.sqlite3?mode=ro' "select url, proposal, title from tom_calendar_calendarevent where url in ('RUN:69','RUN:70','RUN:71','RUN:72','RUN:73','RUN:74','RUN:75') or url like 'ALLOC:1:%' or url like 'ALLOC:76:%'"`: every row should carry its run's code, and each container title should start with its target. Then tick F12 in `.planning/v2.4-INTENT-REVIEW.md`.</human-check>
  </verify>
  <done>
    The notebook was executed once, top to bottom, with no error output, on a `fomo-notebook-db-` scratch copy that its last cell removed. The new F12 cell sits between `db8b9701` and `a5619fed`, and its stored output shows the target-first container title, both codes, the `No proposal recorded` legend entry and `PASS: F12`. The runbook carries the three edits and Sphinx builds. The full suite passes. One docs commit holds exactly these two files, and the operator's `.planning/` files are still unstaged and uncommitted.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| working tree -> running cron | `run_unattended` imports this checkout every 15 min, so a saved edit is live code |
| `CampaignRun` admin data -> public calendar | staff-entered `proposal_code` and target name are now rendered in event titles, fills and the legend |
| notebook kernel -> Django `DATABASES` | a docs artifact runs real ORM writes on the production host |
| executor -> git history | commits are made in a working tree that holds the operator's uncommitted planning edits |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-lsf-01 | Denial of service | a half-edited `campaign_reconciler.py`/`allocation_projector.py`/`calendar_display_extras.py` imported by the next cron tick or web request | high | mitigate | Tests are written first. Each production Edit leaves the module importable and working (the helper and constant are added before use; the new label is added before the old one is removed). The import smoke check runs after every production edit (Task 1 verify 1, Task 2 verify 1). |
| T-lsf-02 | Tampering | notebook re-execution writing to `src/fomo_db.sqlite3`, or a copy torn mid-tick | high | mitigate | Setup cell `9084663a` and teardown `d5248b35` are untouched. The Task 3 gate reads the resolved `fomo-notebook-db-` path from stored output. The executor starts only after an END banner and never exports `FOMO_DATABASE_PATH` or runs a writing command against the live database (finding 10). |
| T-lsf-03 | Tampering | the run's own event writes overwriting an event attributed to a different run, or a confirmed night | medium | mitigate | No new write path. `proposal` rides on the existing builders, which sit behind `_may_write()`/`writable_*()` and the CR-04/CR-05 confirmation guards, so the ownership rules are unchanged. `observation_projector.py` and `calendar_utils.py` are pinned unchanged (Task 1 verify 4). |
| T-lsf-04 | Tampering (XSS) | proposal code or target name echoed into the legend label, title and inline style | low | mitigate | The template is unchanged: labels and titles render through Django autoescaping, and the fill colour comes from `proposal_color()`'s fixed palette (the raw code is only a hash input, T-09-01). No `|safe` is added and `src/templates/` stays unedited (Task 2 verify 4). |
| T-lsf-05 | Information disclosure | proposal codes now visible on public `RUN:`/`ALLOC:` entries | low | accept | Codes are already shown on the same proposal's observation entries. `_skip_reason()` keeps unapproved runs off the calendar entirely, and `AUTH_STRATEGY='READ_ONLY'` applies. The values are the operator's own allocation identifiers, not secrets. |
| T-lsf-06 | Tampering (integrity) | a target-first title longer than `CalendarEvent.title`'s 200 characters | low | accept | This already existed: `telescope_instrument` allows 255. The live store is SQLite, which does not enforce the length. Live titles are under 70 characters. Truncation would be a separate change. |
| T-lsf-07 | Repudiation / Tampering | a task commit sweeping the operator's uncommitted `.planning/` edits or untracked files into history | medium | mitigate | Every commit stages by explicit path after `git branch --show-current`. Task 3's last gate shows the four operator files still unstaged. |
| T-lsf-SC | Tampering | package installs | low | accept | No npm/pip/cargo install in this plan; `nbconvert`/`nbformat` are already installed. |
</threat_model>

<verification>
- Task 1: 17 new tests (R1-R10, A1-A7) pass; the reconciler, allocation projector, approval, null-campaign, write-and-reconcile and command test modules pass; the shared writer, the observation projector and the templates are untouched.
- Task 2: `NO_PROPOSAL_LABEL == 'No proposal recorded'` drives the grey legend entry; template hooks and CSS names are unchanged.
- Task 3: the paired notebook was re-executed on a scratch copy and shows the F12 proofs with `PASS: F12`; the runbook has three edits; the lifecycle notebook is proven unaffected; the full suite passes; ruff, ruff-format and Sphinx are clean.
- After landing (operator): the live month view shows target-first containers in their proposal colours (Task 3 human-check).
</verification>

<success_criteria>
- `RUN:` containers and `ALLOC:` nights carry `run.proposal_code` (blank stays blank) through the existing no-churn writers. Re-runs report `unchanged`; a first fill reports one `updated`.
- The container title leads with the target, with no duplicate, following finding 2's rule; a run with no target keeps today's title; night titles are unchanged.
- The empty-proposal legend entry reads `No proposal recorded`; no CSS class or template hook is renamed.
- The paired notebook and runbook are updated in this plan's own commits; the full suite passes; only the plan's nine files are committed, in three commits.
</success_criteria>

<output>
Create `.planning/quick/261006-lsf-fix-f12-write-run-proposal-code-onto-run/261006-lsf-SUMMARY.md` when done
</output>
