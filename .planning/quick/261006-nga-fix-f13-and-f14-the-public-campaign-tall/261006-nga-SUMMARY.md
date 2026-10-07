---
phase: 261006-nga
plan: 01
subsystem: campaign-tally
tags: [tally, campaign, bootstrap5, F13, F14, notebook, runbook]
status: complete
requirements: [TALLY-01, TALLY-02, ALLOC-01]
commits: 3
plan_head_before: 8b481a48a42a7648245caf9270415b8e632f5828
plan_head_after: d7780ad359b6c7ee5cbbfc76eb488d943184e109
actuals:
  tokens: 60000
  tasks: 3
  commits: 3
key-files:
  modified:
    - solsys_code/campaign_tally.py
    - solsys_code/campaign_tables.py
    - solsys_code/tests/test_campaign_tally.py
    - solsys_code/tests/test_campaign_views.py
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst
decisions:
  - "Night key is the date alone; each record's night is taken in its observed site's timezone (tally-only map), then the run's site, then the UTC date"
  - "Both tally cache keys carry a counting-rule version segment (v2) so cached zeros are never served"
  - "Badges use Bootstrap 5 text-bg-* classes; Progress cell is two d-block text-nowrap lines"
---

# Phase 261006-nga Plan 01: F13/F14 class-wide tally and readable campaign table Summary

A class-wide run's public tally now counts its nights at each linked record's own observing site (tally-only site timezone map, then the run's site, then the UTC date), and the campaign table's badges use Bootstrap 5 classes with a two-line Progress cell.

## Commits

| Task | Commit | Description |
| ---- | ------ | ----------- |
| 1 (tracer) | `1b2cc14` | fix(261006-nga): count a class-wide run's nights at each record's own observing site |
| 2 | `48d81c6` | fix(261006-nga): readable Bootstrap 5 badges and a two-line Progress cell on the campaign table |
| 3 | `d7780ad` | docs(261006-nga): demo a class-wide run's per-site night tally and the readable campaign row; runbook tally wording |

`commits: 3` is measured (`git rev-list --count 8b481a4..HEAD`).

## What changed

- **`campaign_tally.py`**
  - `_NIGHT_SITE_TIMEZONES` (eight LCO site codes: coj, cpt, elp, lsc, ogg, sor, tfn, tlv), private to the module and used only for tally night keying.
  - Helpers `_usable_zone()` and `_record_site_zone()`. Record data is only ever a lookup key into the map; only the map's own values reach `ZoneInfo`.
  - `night_counts_for_run()` has no early zero return any more. It keys each record's night by its mapped observed site, then the run's site, then `start.astimezone(UTC).date()`. Date-only de-duplication, one query per run, same `.only()` columns, never raises.
  - `TALLY_CACHE_KEY_VERSION = 'v2'` is in `build_tally_cache_key()` and `build_rollup_cache_key()`. The roll-up needed no logic change.
- **`campaign_tables.py`**
  - Badge dict values, both renderer fallbacks, `render_telescope_class()` and the TBD badge all use `text-bg-*`. The `1m0` badge is `badge text-bg-light` with its grey border and tooltip kept.
  - `render_progress()` emits two `d-block text-nowrap` spans inside the span that carries the `title`.
- **Notebook** `campaign_lifecycle_demo.ipynb`: one new markdown+code pair between `eb92e3b4` and `914891c4`, one paragraph added to `bbf027b0`, re-executed on a `fomo-notebook-db-` scratch copy (25 code cells, one fresh run, teardown removed the copy). The stored output shows the class-wide run going from `{0, 0, 0}` to `{'nights_observed': 2, 'nights_scheduled': 1, 'nights_failed': 1}`, the four observed records on three UTC dates counting as two nights, key version `v2`, the readable badge, the two Progress lines, and `PASS: F13/F14`.
- **Runbook** `telescope_runs_calendar.rst`: the three scoped edits in "What does a run's or a campaign's public tally show?".

## Test counts

| Scope | Before | After |
| ----- | ------ | ----- |
| `test_campaign_tally` | 84 | 98 (+11 `TestNightCountsPerRecordSite`, +2 `TestTallyNightSiteTimezones`, +1 cache-key test) |
| `test_campaign_views` | 93 | 98 (+1 V1, +4 B1-B4) |
| Full suite (`solsys_code --exclude-tag=ephemeris_segfault --parallel 4`) | 2113 | 2132 |

Full-suite result of the final run: `Ran 2132 tests in 444.078s` / `OK`.

The renamed test `test_site_unset_counts_the_utc_date_and_never_raises` is the only existing test updated; it was the one pinning the F13 defect (site-less run reporting zero). No other test was changed.

RED phase matched the plan: the cache-key test, the renamed test, T1-T7, T9-T11, M1, M2 and V1 failed with AssertionError, T8 passed as a guard (Task 1); B1-B4 failed with AssertionError (Task 2).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] T11 fixture hit a unique constraint**
- **Found during:** Task 1 RED run
- **Issue:** two class-wide runs built from `_class_wide_run()` shared campaign, instrument and window, so `CampaignRun` raised `IntegrityError` (unique on campaign + telescope_instrument + window). The plan's RED rule says to fix a fixture error in the test first.
- **Fix:** T11 gives the two runs distinct `telescope_instrument` values (`... Sinistro A` / `... Sinistro B`).
- **Files modified:** `solsys_code/tests/test_campaign_tally.py`
- **Commit:** `1b2cc14`

### Other notes

- **One flaky full-suite run.** The first full-suite run under `--parallel 4` crashed with "cannot pickle 'traceback'" after `TestCampaignRunTableProgressColumn.test_page_query_count_grows_by_a_bounded_per_row_amount_not_unboundedly` failed with `2 != 3`. That test (pre-existing, not touched here) calls `cache.clear()` on the real shared `FileBasedCache`, which parallel workers and the cron `run_unattended` tick also use. It passed alone, passed within the Task 1 and 2 module runs, and the full suite passed on the immediate re-run (2132 OK). Treated as a pre-existing shared-cache race, not caused by this change. Worth a follow-up: put that class under `@override_settings(CACHES=TEST_CACHES)` like the new classes.
- Ruff's lint hook reports "no files to check" for the notebook (only `ruff-format` applies); format passes.
- The runbook's "a site's timezone being edited" sentence in "How fresh is the tally?" was left unchanged as instructed. It now applies to the run's site fallback only.

## Authentication gates

None.

## Known Stubs

None.

## Threat Flags

None. The only new input reaching a lookup is the stored `observed_site` string, used as a key into a fixed map and never rendered or passed to `ZoneInfo` (T-nga-04 mitigated and tested by T7).

## Ideas for v2.5

The developer's preferred general fix, recorded here and deliberately NOT implemented: per-site obscode SETS in `campaign_attribution.LCO_SITE_CODE_TO_OBSCODE` (each LCO site mapped to the set of its MPC obscodes) with membership checks. That would let attribution, gap analysis and the tally share one verified site table, and would retire the tally-only `_NIGHT_SITE_TIMEZONES` map.

## Operator follow-up (human check)

Once the web server serves the new code, open the `KEY2026B-004_targets` campaign page logged out: rows 69-75 should show non-zero nights (run 69 `[O]` near 48, run 71 `[S]` at least 1, run 73 `[O]` near 29, runs 70/72/74/75 `[O]` at most 7/11/8/8), the roll-up equals the sum of the rows, the `1m0` badge reads without hovering, and each Progress cell is two lines. Then tick F13 and F14 in `.planning/v2.4-INTENT-REVIEW.md`. Not done here (orchestrator/operator job).

## Self-Check: PASSED

- Files exist: all six modified files; commits `1b2cc14`, `48d81c6`, `d7780ad` are ancestors of HEAD.
- `campaign_attribution.py`, `campaign_gap.py`, `proposal_allocation.py`, `calendar_utils.py` and `src/templates/` are unchanged versus the plan base.
- The four operator-owned `.planning/` files are still modified and unstaged; untracked files are still untracked.
