---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
verified: 2026-09-19T17:05:00Z
status: human_needed
score: 5/5 must-haves verified
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-01-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-01-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-02-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-02-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-03-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-03-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-04-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-04-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-05-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-05-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-06-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-06-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-07-PLAN.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-07-SUMMARY.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-REVIEW-FIX.md"
  - ".planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-REVIEW.md"
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/admin.py"
  - "solsys_code/allocation_projector.py"
  - "solsys_code/calendar_utils.py"
  - "solsys_code/campaign_gap.py"
  - "solsys_code/campaign_reconciler.py"
  - "solsys_code/campaign_tables.py"
  - "solsys_code/campaign_tally.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/management/commands/load_telescope_runs.py"
  - "solsys_code/migrations/0023_proposal_time_allocation_and_campaignrun_proposal_code.py"
  - "solsys_code/models.py"
  - "solsys_code/observation_projector.py"
  - "solsys_code/proposal_allocation.py"
  - "solsys_code/status_vocabulary.py"
  - "solsys_code/templatetags/calendar_display_extras.py"
  - "solsys_code/unattended.py"
  - "src/templates/campaigns/campaign_list.html"
  - "src/templates/campaigns/campaignrun_gap_analysis.html"
  - "src/templates/campaigns/campaignrun_table.html"
  - "src/templates/tom_calendar/partials/calendar.html"
  - "src/templates/tom_calendar/partials/event_form.html"
covered_digest: "v1:sha256:e4dad695923c0d453460dcdfdb3c75fcfa153eaaaeec272faf39f22b7c64a0c3"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Open the calendar month view on a month containing an elapsed allocation night, a cancelled run night and an observed record. Confirm by eye: the unused night is visibly muted (opacity 0.55 + dashed border) and its label starts with `[U]`; the cancelled night shows `[C]` with its terminal ring; clicking the `[U]` legend entry isolates the unused nights and clicking again clears the filter. Then open one attributed entry's pop-up and confirm the tally line reads sensibly beside the campaign name."
    expected: "Unused nights are muted and `[U]`-prefixed; cancelled nights are `[C]` with a ring; the legend `[U]` swatch toggles a filter; the pop-up's attributed-run block shows `N groups · N records · [O] n [S] n [X/F] n [U] n`."
    why_human: "Visual distinctness, colour/opacity legibility and a JavaScript click-to-filter interaction cannot be verified by grep or by the Django test client. Deferred from execution by plan 37-07's own `<human-check>` block (37-07-PLAN.md:307)."
  - test: "Product decision on the campaign roll-up strip's unused figure (CR-02 residual). On a campaign runs page, set a run's `run_status` to CANCELLED (or let an awarded night elapse) without touching any linked observation record, then reload the page and compare the roll-up strip's `[U]` total against the sum of the `[U]` values in the Progress cells directly beneath it."
    expected: "Decide whether an up-to-1-hour disagreement between the strip and the rows on the unused figure alone is acceptable, or whether `campaign_rollup()`/`get_or_compute_rollup()` should be given the same live-unused split `get_or_compute_tally()`/`tallies_for_runs()` received — which requires relaxing `test_campaign_list_query_count_bound_with_three_campaigns`'s zero-marginal-query bound on the anonymous campaign list."
    why_human: "A deliberate, documented trade-off between public-page query amplification and tally freshness. The fixer flagged it for a human product decision rather than picking a side (37-REVIEW-FIX.md, CR-02 scope note). Neither choice is derivable from the codebase."
---

# Phase 37: Status Vocabulary, Public Tallies, Provenance-Blind Gaps — Verification Report

**Phase Goal:** The layered calendar reads correctly to everyone — one status vocabulary instead of three that agree by convention, an ongoing public tally of what each run and campaign actually got, unused awarded nights that look unused, and coverage gaps that count every observation.
**Verified:** 2026-09-19T17:05:00Z
**Status:** human_needed
**Re-verification:** No — initial verification (post code-review-fix codebase, `3c14424`..`3a19846`)

## Goal Achievement

### Observable Truths

| # | Truth (ROADMAP Success Criterion) | Status | Evidence |
|---|---|---|---|
| 1 | One status vocabulary drives every calendar title prefix and status ring — the three parallel prefix maps are gone, a placed-but-unobserved night has its own named state, and terminal-state detection goes through one facility-aware classifier instead of a hardcoded `status == 'COMPLETED'` | ✓ VERIFIED | `solsys_code/status_vocabulary.py` (277 lines) owns `MARKER`/`LABEL`/`LEGEND`/`STAGE_MARKER`/`FAILURE_MARKER_BY_STATUS`/`RUN_STATUS_MARKER`/`OCSState`/`DisplayState`. Grep for the three retired maps (`_STAGE_MARKER`, `_FAILURE_MARKER_BY_STATUS`, `RUN_STATUS_CALENDAR_PREFIX`, `_TERMINAL_PREFIXES`, `_OBSERVATION_STATUS_LEGEND`) returns only docstring/comment references — no live definition anywhere. Consumers import it: `observation_projector.py:49`, `campaign_reconciler.py:65`, `allocation_projector.py:59`, `calendar_utils.py:27`, `calendar_display_extras.py:42`, `campaign_tally.py:31`, `campaign_gap.py:34`. `DisplayState.SCHEDULED` → `'[S]'`, label `'Scheduled'` (D-03). `stage_for()` now reads `observed_states_for(facility)`/`failed_states_for(facility)` instead of the facility's own two methods; `OBSERVED_STATES_BY_FACILITY` maps `GEM`/`ESO` → `frozenset()`. `calendar_utils.py:337` compares against `OCSState.COMPLETED`, not a bare literal. Behavioral: `test_status_vocabulary` + `test_observation_projector` + `test_calendar_utils` green. |
| 2 | Any visitor — not only staff — sees on each run a live tally of linked observation groups and records and of nights observed / scheduled / expired-or-failed / unused so far, updating as the projector narrows, and the campaign page rolls the same tally up across its runs | ✓ VERIFIED | `campaign_tally.py` (647 lines) is the single computation home. `CampaignRunTable.progress` column (`campaign_tables.py:98`) is in `Meta.sequence` and not in any non-staff exclusion; `CampaignRunTableView.get_table_kwargs()` (`campaign_views.py:228`) feeds it one pre-computed `tallies_for_runs()` pass, page-sliced. Roll-up: `campaignrun_table.html` header strip (`rollup`/`rollup_segments` from `get_context_data`) and `campaign_list.html` badge (`campaign.rollup.nights_observed`). Liveness is a real state transition, proven by passing named tests, not by presence: `test_saving_a_linked_record_is_reflected_with_no_clock_advance`, `test_linking_an_older_untouched_record_is_reflected_with_no_clock_advance`, `test_removing_a_non_newest_link_is_reflected_with_no_clock_advance`, `test_unused_count_is_live_even_on_a_cache_hit` (both call sites), `test_saving_a_linked_record_moves_the_rollup_on_next_load_no_clock_advance_no_cache_clear`, `test_anonymous_and_staff_requests_render_the_same_segments`, `test_anonymous_get_of_runs_page_returns_200_with_rollup_content`. PII gate unchanged: `ALLOWED_FIELDS_FOR_NON_STAFF` not widened (`test_get_queryset_is_unchanged_and_no_field_added_for_the_tally`, `test_anonymous_context_rows_have_no_contact_fields`). See the ⚠️ roll-up residual below — bounded, does not defeat the criterion's narrowing clause. |
| 3 | A run's `run_status` never changes by itself: whatever its linked records did, it stays what a staff member set | ✓ VERIFIED | Exhaustive grep for `run_status =` / `run_status=` assignment across `solsys_code/` (excluding tests/migrations/filters) finds exactly three write paths, none record-derived: the staff decision view (`campaign_views.py:860`, `_ACTION_TO_RUN_STATUS[action]`) and the two staff-run schedule-file ingest paths (`load_telescope_runs.py:269`, `cutover_classical_allocations.py:480`, both `_CLASSICAL_RUN_STATUS[parsed.status]` from the file). `status_vocabulary.py` and `campaign_tally.py` both state the TALLY-03 invariant in words and hold no write. Behavioral guard tests pass: `test_run_status_unchanged_for_a_fully_observed_run`, `test_run_status_unchanged_for_a_run_with_no_linked_records`, `test_no_computation_path_module_assigns_to_run_status_attribute`, `test_module_states_tally_03_invariant_in_words`. |
| 4 | An awarded night that came and went with nothing scheduled or observed is visibly different on the calendar from a night that was actually observed | ✓ VERIFIED | `calendar_display_extras.unused_night_decoration()` returns `{token: '[U]', label, tooltip}` for an elapsed, still-standing `ALLOC:`-namespace night on a publicly visible run, gated through the single shared rule `campaign_tally.is_unused_allocation_night()` (the same function the table's count reads — D-15 agreement by construction). Two independent channels in `calendar.html`: the `.cal-event-unused` style rule (lines 185-188: `opacity: 0.55` + dashed border) and the `[U]` text token prepended to the title (lines 284, 317), plus `title="{{ unused.tooltip }}"`. No fourth ring colour added (`RING_*` sets exclude `UNUSED`). Legend click-to-filter wired on `entry.filterable` from `status_vocabulary.LEGEND`, not on a `'[U]'` string compare (calendar.html:356-362, JS 389-425). Behavioral: `test_elapsed_allocation_nights_render_unused_class_attribute_and_token`, `test_only_the_unused_entry_is_filterable`, `test_calendar_page_renders_every_legend_marker_and_label`, `test_past_night_but_cancelled_run_status_is_not_unused`, `test_past_night_but_weather_tech_failure_run_status_is_not_unused` — all green. |
| 5 | Coverage-gap analysis counts every observation on the campaign calendar, so classical and queue time is no longer reported as unclaimed | ✓ VERIFIED | `campaign_gap.observation_claimed_dates()` (lines 172-243) is a real second claim source, unioned into `claimed_dates()` (line 353) alongside the untouched approved-run-window source. Provenance-blind by construction: `Q(calendar_event_meta__run__campaign=campaign) | Q(target__in=campaign.targets.all())` — an unattributed classical/queue observation of a campaign target claims its night. Only `_CLAIMING_DISPLAY_STATES` (OBSERVED/SCHEDULED per `status_vocabulary.classify_record()`) claim; queued/expired/cancelled/failed/inconsistent claim nothing. Unresolvable site → `site_unknown_count`, surfaced on the gap page (`campaignrun_gap_analysis.html:66-70`), never silently dropped. Night derived via `telescope_runs.observing_night()`, never re-derived. Behavioral: 21 named tests in `TestObservationClaimedDates` / union / ordering / empty classes all green, including `test_observed_block_claims_its_site_local_night` (fixture record carries no run link — the exact provenance-blind case), `test_union_with_approved_run_window_produces_one_claimed_date`, `test_record_with_unresolvable_site_increments_unknown_count_not_claimed`. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `solsys_code/status_vocabulary.py` | Single marker/label/legend/classifier home | ✓ VERIFIED | 277 lines; 8 importers; no local marker table survives anywhere |
| `solsys_code/tests/test_status_vocabulary.py` | Invariant tests incl. retirement | ✓ VERIFIED | 13.8 KB; asserts `RETIRED_TITLE_PREFIXES` is gone |
| `solsys_code/campaign_tally.py` | Every public tally figure | ✓ VERIFIED | 647 lines; 68 tests green |
| `solsys_code/tests/test_campaign_tally.py` | Tally + TALLY-03 guard tests | ✓ VERIFIED | 40.7 KB; guard class present and green |
| `solsys_code/campaign_gap.py` | Provenance-blind claim source | ✓ VERIFIED | 442 lines; second source + site-unknown counter |
| `solsys_code/tests/test_campaign_gap.py` | Union/exclusion/site-ladder tests | ✓ VERIFIED | green |
| `solsys_code/proposal_allocation.py` | Portal fetch + stored estimate | ✓ VERIFIED | 289 lines; `_PROPOSAL_CODE_RE` validated sink; no key in logs/summaries |
| `solsys_code/migrations/0023_...py` | `ProposalTimeAllocation` + `CampaignRun.proposal_code` | ✓ VERIFIED | `makemigrations --check --dry-run` → "No changes detected" |
| `src/templates/campaigns/campaignrun_table.html` | Roll-up strip | ✓ VERIFIED | strip renders `rollup`/`rollup_segments` |
| `src/templates/campaigns/campaign_list.html` | Nights-observed badge | ✓ VERIFIED | `campaign.rollup.nights_observed` clause |
| `src/templates/tom_calendar/partials/event_form.html` | Pop-up tally line | ✓ VERIFIED | `{% run_tally event as tally %}` inside the existing visibility gate |
| `src/templates/tom_calendar/partials/calendar.html` | `[U]` chip + filterable legend | ✓ VERIFIED | two channels + `entry.filterable` swatch |
| `docs/runbooks/telescope_runs_calendar.rst` | 9 markers, 5 unattended steps, coverage-gap section | ✓ VERIFIED | marker table lines 149-180; "runs five steps" line 1466 incl. `proposal_allocation`; coverage-gap section line 2014ff |
| `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` | Tally, roll-up, unused night with executed output | ✓ VERIFIED | cells 29/30 (pop-up tally), 41/42 (public Progress + roll-up, anonymous==staff), 43/44 (unused nights, `[W]` precedence) all carry real stored output |

### Key Link Verification

| From | To | Via | Status |
|---|---|---|---|
| `observation_projector.title_for()` | `status_vocabulary.STAGE_MARKER` / `FAILURE_MARKER_BY_STATUS` / `MARKER` | import + call, no local table | ✓ WIRED |
| `campaign_reconciler.event_title()` / `allocation_projector.allocation_night_title()` | `status_vocabulary.RUN_STATUS_MARKER` | import | ✓ WIRED |
| `calendar_display_extras.status_border_css()` | `status_vocabulary.state_for_title()` + `RING_*` | line 185-189 | ✓ WIRED |
| `calendar_utils.resolve_placement_block()` | `status_vocabulary.OCSState.COMPLETED` | line 337 | ✓ WIRED |
| `CampaignRunTableView.get_table_kwargs()` | `campaign_tally.tallies_for_runs()` → `render_progress()` | `tallies` kwarg, dict lookup only | ✓ WIRED |
| `CampaignRunTableView.get_context_data()` / `CampaignListView` | `campaign_tally.get_or_compute_rollup()` | templates | ✓ WIRED |
| `calendar_display_extras.run_tally()` | `campaign_tally.get_or_compute_tally()` → `event_form.html` | simple_tag | ✓ WIRED |
| `calendar_display_extras.unused_night_decoration()` | `campaign_tally.is_unused_allocation_night()` | same function the table count uses | ✓ WIRED |
| `campaign_gap.claimed_dates()` | `observation_claimed_dates()` → `classify_record()` / `observing_night()` | line 353 | ✓ WIRED |
| `unattended.STEPS` | `step_proposal_allocation()` → `proposal_allocation.refresh_all()` | `STEPS` tuple, 5th entry | ✓ WIRED |
| `load_telescope_runs` | `CampaignRun.proposal_code` → `ProposalTimeAllocation.proposal_code` | model field + `proposal_codes_to_fetch()` | ✓ WIRED |

### Data-Flow Trace (Level 4)

| Rendered value | Source | Produces real data | Status |
|---|---|---|---|
| Progress cell group/record counts | `link_counts_for_runs()` — one `Count`/`Max` aggregate over `CampaignRunObservation` | Yes | ✓ FLOWING |
| Progress cell night segments | `night_counts_for_run()` → `classify_record()` + `observing_night()` over linked `ObservationRecord`s | Yes | ✓ FLOWING |
| Progress cell `[U]` figure | `unused_nights_for_run()` (live `ALLOC:` event query) else `proposal_allocation.estimated_unused_nights()` (stored rows) else `None` → "not yet known" | Yes; never falsely zero | ✓ FLOWING |
| Roll-up strip | `campaign_rollup()` → `tallies_for_runs()`, keyed on `campaign_records_version()` | Yes (unused sub-figure hour-bounded — see Warning) | ✓ FLOWING |
| Campaign list badge | `campaign.rollup.nights_observed` | Yes | ✓ FLOWING |
| Pop-up tally line | `run_tally()` → `get_or_compute_tally()` | Yes | ✓ FLOWING |
| `[U]` calendar chip | `unused_night_decoration()` → live `is_unused_allocation_night(event.end_time, run.run_status)` | Yes | ✓ FLOWING |
| Gap page claimed/site-unknown | `_compute_gap()` result dict | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Vocabulary, tally and gap modules behave as specified | `python manage.py test solsys_code.tests.test_status_vocabulary solsys_code.tests.test_campaign_tally solsys_code.tests.test_campaign_gap` | Ran 126 tests in 79.2s — OK | ✓ PASS |
| Calendar `[U]` decoration, legend, pop-up tally | `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` | Ran 158 tests in 27.0s — OK | ✓ PASS |
| Public Progress cell, roll-up, anonymity, portal fetch | `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_proposal_allocation` | Ran 111 tests in 143.9s — OK | ✓ PASS |
| Migration state clean | `python manage.py makemigrations --check --dry-run` | "No changes detected" | ✓ PASS |
| Lint gate | `pre-commit run ruff --all-files` | Passed | ✓ PASS |
| Title migration complete on the developer database | read-only `sqlite3` count of `[CANCELLED]%`/`[WEATHERED]%`/`[EXPIRED]%`/`[FAILED]%` titles in `src/fomo_db.sqlite3` | 0 of 233 events | ✓ PASS |
| Ephemeris segfault exclusion is a property of the invocation | `grep -n "ephemeris_segfault" solsys_code/tests/test_views.py` | `@tag('ephemeris_segfault')` at line 98, class still present | ✓ PASS |

Total independently re-run: 395 tests, all green. The orchestrator's full-suite run (1736 tests, OK) was not repeated.

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| — | — | No `scripts/*/tests/probe-*.sh` exist and no PLAN/SUMMARY declares a probe | ? SKIP |

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| STATUS-01 | 37-01, 37-07 | One status vocabulary replaces the three parallel prefix maps, incl. a placed-but-unobserved state | ✓ SATISFIED | Truth 1; `status_vocabulary.py`; three maps deleted; `[S]`/`Scheduled` present; runbook marker table |
| STATUS-02 | 37-01 | General terminal-state classifier replaces `status == 'COMPLETED'` | ✓ SATISFIED | Truth 1; `observed_states_for()`/`failed_states_for()`; `OBSERVED_STATES_BY_FACILITY` GEM/ESO → empty; `OCSState.COMPLETED` at `calendar_utils.py:337` |
| TALLY-01 | 37-02, 37-04, 37-05, 37-06, 37-07 | Public per-run tally, updating as the projector narrows | ✓ SATISFIED | Truth 2; Progress cell + pop-up block; `test_anonymous_and_staff_requests_render_the_same_segments` |
| TALLY-02 | 37-04, 37-05, 37-07 | Campaign page rolls the same tally up | ✓ SATISFIED | Truth 2; roll-up strip + list badge; `campaign_rollup()` sums through `tallies_for_runs()` |
| TALLY-03 | 37-04 | `run_status` never set automatically from linked records | ✓ SATISFIED | Truth 3; grep of every write path + three guard tests |
| UNUSED-01 | 37-04, 37-06, 37-07 | An unused awarded night is visually distinct from a realised one | ✓ SATISFIED | Truth 4; two channels; shared classifier |
| GAPB-01 | 37-03, 37-07 | `claimed_dates()` counts every observation on the campaign calendar | ✓ SATISFIED | Truth 5; `observation_claimed_dates()` union |

No orphaned requirements: REQUIREMENTS.md maps exactly these seven IDs to Phase 37, and every one appears in at least one plan's `requirements` field.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `solsys_code/campaign_tally.py` + `campaign_views.py` | `campaign_rollup()` / `get_or_compute_rollup()` | Roll-up strip caches its `unused_*` fields wholesale while the Progress cells beneath it recompute them live on every call | ⚠️ Warning | Two figures on one page can disagree for up to `TALLY_CACHE_TTL_SECONDS` (1 h) after a purely time-driven change — a night elapsing, or a staff `run_status` edit (which does not move `records_version`). Documented and deliberately deferred by the fix pass (37-REVIEW-FIX.md, CR-02 scope note); routed to a human product decision below. |
| `solsys_code/management/commands/backfill_lco_observations.py` | 303 | `if block.get('state') == 'COMPLETED':` — a bare OCS-state literal outside `status_vocabulary` | ℹ️ Info | Same concept `calendar_utils.resolve_placement_block()` converted to `OCSState.COMPLETED` (line 337). This file is in no Phase 37 plan's `files_modified`, so it is out of the phase's declared scope — a small residual of STATUS-01's "exactly one spelling" ambition, not a Phase 37 regression. |
| `solsys_code/telescope_runs.py` | `_resolve_proposal()` | Schedule-file proposal token not charset-validated (WR-07 partial) | ℹ️ Info | Not exploitable: `proposal_allocation.fetch_proposal_allocations()` validates against `_PROPOSAL_CODE_RE` and `quote(..., safe='')`-escapes before building the credentialed URL, and raises `PortalUnavailable('ValueError')` on failure. The request sink — the only boundary that matters for the URL-redirection risk — is closed. Deferring the ingestion grammar avoids breaking real schedule files with no fixture coverage. |
| various | — | `TBD` string occurrences | ℹ️ Info (not a debt marker) | Every hit is the domain term "TBD window" (a to-be-determined observing window) on `CampaignRun`, present since Phase 15. No `FIXME`/`XXX` anywhere in the phase's changed files. `PLACEHOLDER` hits are the tier-3 `PLACEHOLDER` Observatory concept. No debt-marker gate fires. |

### Paired-Docs Rule (CLAUDE.md) — Independent Finding

The orchestrator asked whether `project_observation_calendar_demo.ipynb` not being regenerated is a real paired-docs gap. **It is not.** Evidence:

- Phase 37's change to `observation_projector.py` is, for the facilities that notebook exercises, a **pure refactor**. `git diff 2adcd8c~1 HEAD -- solsys_code/observation_projector.py` shows the marker *values* are byte-identical (`'[Q]'`, `'[S]'`, `'[O]'`, `'[X]'`, `'[C]'`, `'[F]'`, `'[?]'` before and after); only their source moved to `status_vocabulary`.
- The one genuine behaviour change (STATUS-02: `observed_states_for()` returns `frozenset()` for `GEM`/`ESO`) cannot reach this notebook: `observation_projector.PROJECTED_FACILITIES = ('LCO', 'SOAR')`, and for LCO/SOAR `observed_states_for()` evaluates to exactly the pre-existing `set(terminal) - set(failed)` expression.
- The notebook's stored output carries only current-vocabulary markers (`[?] [C] [F] [O] [Q] [S] [X]`) and zero legacy bracket-word spellings.

CLAUDE.md's rule triggers on a plan that changes a module's *behavior* and explicitly carves out pure refactors. The three notebooks that did change (`campaign_lifecycle`, `reconcile_campaign_runs`, `load_telescope_runs`) are the three whose demonstrated behaviour actually moved. The discrepancy is confined to plan 37-07's commit message wording ("regenerate the four pre-executed notebooks" where three changed) — a ℹ️ Info-level documentation inaccuracy in the commit log, not a stale artifact and not a gap.

### Deferred Items

None. Phase 37 is the final phase of the v2.4 milestone (`roadmap.analyze` returns phases 33-37 only), so no later phase exists to defer to.

### Human Verification Required

#### 1. Calendar visual and interaction check (deferred from plan 37-07)

**Test:** Open the calendar month view on a month containing an elapsed allocation night, a cancelled run night and an observed record. Confirm by eye: the unused night is visibly muted and its label starts with `[U]`; the cancelled night shows `[C]` and its terminal ring; clicking the `[U]` legend entry isolates the unused nights and clicking it again clears the filter. Then open one attributed entry's pop-up and confirm the tally line reads sensibly beside the campaign name.
**Expected:** Unused nights are muted (opacity 0.55, dashed border) and `[U]`-prefixed; cancelled nights keep `[C]` plus their ring; the `[U]` legend swatch toggles a filter on and off; the pop-up shows `N groups · N records · [O] n [S] n [X/F] n [U] n`.
**Why human:** Visual distinctness, colour/opacity legibility and a JavaScript click-to-filter interaction are not observable to grep or the Django test client. The markup, CSS, tag and filter wiring are all verified present and unit-tested — only the "reads correctly by eye" half is outstanding.

#### 2. Product decision: roll-up strip unused-figure staleness (CR-02 residual)

**Test:** On a campaign runs page, change a run's `run_status` to CANCELLED (or let an awarded night elapse) without touching any linked observation record, reload, and compare the roll-up strip's `[U]` total against the sum of the `[U]` values in the Progress cells beneath it.
**Expected:** A decision, not a code state. Either (a) accept the hour-bounded disagreement on the unused sub-figure alone, or (b) give `campaign_rollup()`/`get_or_compute_rollup()` the same live-unused split `get_or_compute_tally()`/`tallies_for_runs()` received, which requires relaxing `test_campaign_list_query_count_bound_with_three_campaigns`'s zero-marginal-query bound on the anonymous campaign list page.
**Why human:** A deliberate, documented trade-off between anonymous-page query amplification and tally freshness. The fix pass explicitly declined to pick a side. Nothing in the codebase determines the right answer.

### Gaps Summary

No gaps. All five ROADMAP Success Criteria are satisfied by real, wired, data-flowing code with passing behavioral tests, and all seven requirement IDs are accounted for. The phase status is `human_needed` rather than `passed` solely because two items cannot be resolved programmatically: one visual/interaction check the planner deliberately deferred to end-of-phase, and one product decision the code-review fix pass deliberately escalated.

Notably, the two items the orchestrator carried forward for independent judgement resolved differently from how they were framed:

- **CR-02 residual is narrower than described.** `build_rollup_cache_key()` *does* fold in `campaign_records_version()`, so the roll-up strip is fully live to a projector narrowing — the roadmap's own trigger phrase. The residual is confined to the `unused_*` sub-figure under a purely time-driven or staff-`run_status`-driven change. It does not breach Success Criterion 2; it is a visible one-page inconsistency worth a decision.
- **The paired-docs concern is not a breach.** The un-regenerated notebook demonstrates a code path Phase 37 refactored without changing, on facilities the one real behaviour change cannot reach.

---

_Verified: 2026-09-19T17:05:00Z_
_Verifier: Claude (gsd-verifier)_
