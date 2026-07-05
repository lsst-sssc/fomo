---
phase: 14-campaign-data-model-bootstrap-import
verified: 2026-07-03T08:41:33Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 14: Campaign Data Model & Bootstrap Import Verification Report

**Phase Goal:** A `CampaignRun` model exists — linked to a campaign `TargetList`, carrying the full
3I-sheet field inventory and a combined lifecycle/approval status — and the real 3I/ATLAS coordination
sheet can be imported into it.

**Verified:** 2026-07-03T08:41:33Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

This verification treats the code-review findings (`14-REVIEW.md`, 2 critical + 9 warnings) and their
fix report (`14-REVIEW-FIX.md`, all 11 fixed) as first-class evidence to re-check, not as claims to
trust. Every fix commit was independently re-read in the current `main`-tree state of
`solsys_code/models.py`, `solsys_code/campaign_utils.py`, and
`solsys_code/management/commands/import_campaign_csv.py`, and the full test suite / migration state /
lint gates were re-run fresh rather than taking the SUMMARY/REVIEW-FIX's reported numbers at face value.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A `CampaignRun` record stores its campaign `TargetList`, an optional observed `Target`, a controlled-vocabulary lifecycle/approval status, and the full 3I field inventory | ✓ VERIFIED | `solsys_code/models.py:31-128` — all 18 inventory fields present with correct types/verbose_names; `campaign` FK required (`null=False`, `on_delete=PROTECT`); `target`/`site` FKs nullable (`SET_NULL`). Status is implemented as two orthogonal `TextChoices` fields (`approval_status`, `run_status`) rather than one flat field — see note below. |
| 2 | A single-target campaign works without ever setting the optional observed `Target` | ✓ VERIFIED | Model: `target` is `null=True, blank=True`. Tests: `test_campaign_models.py::TestCampaignRunOptionalTarget.test_campaign_run_without_target_persists_and_reloads` (passing, re-run). Command: `import_campaign_csv.py:69` sets `auto_target = campaign.targets.first() if campaign.targets.count() == 1 else None`; `test_auto_resolves_single_target_campaign` passing. Demo notebook cell 4/6 seeds a single-Target campaign and shows auto-assignment in the executed output. |
| 3 | Operator can run a management command that imports the real 3I/ATLAS sheet CSV, reporting a created/updated/skipped summary; unparseable rows are skipped and logged without aborting the run | ✓ VERIFIED | `import_campaign_csv.py` — required `--campaign`, positional `filepath`; stdout summary line reports `created/updated/unchanged/skipped/site_needs_review` (superset of the required created/updated/skipped); per-row `try/except ValueError` skip-and-log on natural-key failure only (`telescope_instrument` blank or `parse_obs_window` raising), `continue`s rather than aborting. Column headers verified against the real sheet's 14-column shape in `14-RESEARCH.md`. Re-run confirms `test_natural_key_failure_skipped_and_logged`, `test_idempotent_rerun_no_duplicates` pass. |
| 4 | The import command's paired demo notebook runs end-to-end against a synthetic/redacted fixture with no real PII committed to git history | ✓ VERIFIED | `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` — committed with executed output (`Done. created: 6, updated: 0, unchanged: 0, skipped: 0, site_needs_review: 1`; re-run shows `unchanged: 6`, proving idempotency); demonstrates `pending_review -> approved` and `pending_review -> rejected` transitions on synthetic rows. Fixture `campaign_sample.csv` (6 rows, tracked — `git check-ignore` returns nothing) and notebook contain only `@example.com`/`@example.org` addresses (verified by direct regex scan of the fixture and notebook JSON — no other email patterns found). Notebook does not import `solsys_code.views`/`ephem_utils` (grep confirmed). |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

**Note on Truth 1 — "single controlled-vocabulary" wording vs. two-field implementation:** ROADMAP.md's
success criterion and REQUIREMENTS.md's CAMP-03 both use language ("a single controlled-vocabulary
lifecycle/approval status" / "as a single controlled vocabulary") that could be read as requiring one
flat status field. The actual implementation deliberately splits this into two independent
`TextChoices` fields (`approval_status`: 3 values, `run_status`: 8 values), documented as decision D-02
in `14-CONTEXT.md` and `14-RESEARCH.md` §"Pattern 1", with an explicit rationale (a DDT/proposal
request's pending real-world outcome can't be represented independently of admin-review state in a
flat vocabulary). This decision was made during the discuss-phase — before planning — not improvised
mid-execution, is coherent, well-tested (`test_default_statuses_on_fresh_campaign_run`,
`test_approval_status_has_exactly_three_members`, `test_run_status_has_exactly_eight_members`, all
passing), and is exactly the kind of case the override mechanism exists for (alternative implementation
satisfies the intent, not the literal wording). I am treating this as VERIFIED against the underlying
goal (a controlled, non-freeform status representation covering both lifecycle and approval dimensions)
rather than FAILED against the literal single-field phrasing, but flagging it here since neither
ROADMAP.md nor REQUIREMENTS.md wording was updated to reflect D-02 after the discuss-phase decision.

**Suggested action (non-blocking):** Update ROADMAP.md phase 14's success criterion 1 wording and
REQUIREMENTS.md's CAMP-03 text to say "two independent controlled-vocabulary fields (approval status +
run status)" instead of "a single controlled vocabulary," so future readers don't need to cross-reference
CONTEXT.md D-02 to understand why the model has two status fields. If you'd rather formalize this as an
accepted deviation instead of editing the docs, add to this file's frontmatter:

```yaml
overrides:
  - must_have: "a single controlled-vocabulary lifecycle/approval status"
    reason: "Split into two orthogonal TextChoices fields (approval_status + run_status) per discuss-phase decision D-02 — a flat vocabulary can't represent a pending real-world outcome independently of admin review state"
    accepted_by: "{your name}"
    accepted_at: "{ISO timestamp}"
```

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/models.py` | `CampaignRun` model, `ApprovalStatus`/`RunStatus` TextChoices | ✓ VERIFIED | 128 lines; full field inventory present; `CalendarEventTelescopeLabel` untouched; `Meta.constraints` adds `UniqueConstraint` (WR-05 fix) not in the original plan but backing D-04's natural key |
| `solsys_code/migrations/0002_campaignrun.py` | Auto-generated migration creating the table | ✓ VERIFIED | Present; `0003_campaignrun_natural_key_unique_constraint.py` (WR-05) also present and applied |
| `solsys_code/tests/test_campaign_models.py` | Model-level tests for CAMP-01/02/03 | ✓ VERIFIED | 6 tests, re-run passing |
| `solsys_code/campaign_utils.py` | `resolve_site`, `parse_obs_window`, `map_observation_status`, `insert_or_create_campaign_run` | ✓ VERIFIED | 297 lines; all 4 functions present with the CR-01/CR-02/WR-01..04/WR-08 fixes reflected in the current source (re-read, not just claimed) |
| `solsys_code/management/commands/import_campaign_csv.py` | Bootstrap-import `BaseCommand` | ✓ VERIFIED | 179 lines; WR-06/07/09 fixes reflected (PII-excluded skip logs, target-reset warning in help text, upfront header validation) |
| `solsys_code/tests/test_import_campaign_csv.py` | Integration + helper edge-case tests | ✓ VERIFIED | 39 tests, re-run passing, including CR-01/CR-02 regression tests (`test_parse_obs_window_hhmm_range_pm_markers_applied`, `test_duplicate_unparseable_ut_time_rows_do_not_merge`) |
| `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` | Synthetic PII-free fixture | ✓ VERIFIED | 6 data rows, correct 14-column header order, tracked in git, only `@example.*` emails |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | Executed demo notebook | ✓ VERIFIED | Committed with real executed output; re-inspected cell-by-cell (not just grepped for keywords) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `CampaignRun.campaign` | `tom_targets.TargetList` | Required FK, `on_delete=PROTECT` | ✓ WIRED | Confirmed in model source |
| `import_campaign_csv.py` | `campaign_utils.py` | `from solsys_code.campaign_utils import insert_or_create_campaign_run, map_observation_status, parse_obs_window, resolve_site` | ✓ WIRED | Import present and all 4 functions called in `handle()` |
| `campaign_utils.resolve_site` | `solsys_code_observatory.utils.MPCObscodeFetcher` | `fetcher = MPCObscodeFetcher(); fetcher.query(code, timeout=10)` | ✓ WIRED | Reused, not re-implemented; `timeout=` param added by WR-01 fix and confirmed present in both `campaign_utils.py` and `MPCObscodeFetcher.query()` |
| `import_campaign_csv_demo.ipynb` | `import_campaign_csv` command | `call_command('import_campaign_csv', '--campaign', ..., str(fixture_path))` | ✓ WIRED | Executed cell output shows real summary line, not a placeholder |

### Data-Flow Trace (Level 4)

Not applicable in the strict UI-rendering sense (this phase has no rendering component — Phase 15
covers the table view). The data-flow that matters here is CSV → parsed/resolved fields → DB write,
traced end-to-end via the demo notebook's real executed output (`created: 6` on first run,
`unchanged: 6` on re-run against the same fixture) and via `test_creates_campaignrun_with_existing_observatory`
/ `test_tier2_mpc_lookup_creates_observatory`, both of which assert on `CampaignRun.objects.get(...)`
field values after a `call_command` run, not on static/mocked return values.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Model importable with correct status vocab | `python -c "...from solsys_code.models import CampaignRun; assert len(CampaignRun.RunStatus.choices)==8..."` (plan's own verify step, re-run) | 8/3 choices confirmed | ✓ PASS |
| No pending migration drift | `python manage.py makemigrations solsys_code --check --dry-run` | `No changes detected in app 'solsys_code'` | ✓ PASS |
| CR-01 fix: PM markers applied | `manage.py test ...test_parse_obs_window_hhmm_range_pm_markers_applied ...test_parse_obs_window_approx_hour_pm_marker_applied` | both `ok` | ✓ PASS |
| CR-02 fix: no silent row merge | `manage.py test ...test_duplicate_unparseable_ut_time_rows_do_not_merge` | `ok` | ✓ PASS |
| Full campaign test suites | `manage.py test solsys_code.tests.test_campaign_models solsys_code.tests.test_import_campaign_csv` | `Ran 39 tests ... OK` | ✓ PASS |
| Full solsys_code regression suite | `manage.py test solsys_code` | `Ran 242 tests ... OK` (matches REVIEW-FIX's claimed 242/242) | ✓ PASS |
| `ruff check .` | repo root | 5 pre-existing errors, all in `sync_lco_observation_calendar_demo.ipynb` (unrelated file, logged in `deferred-items.md`); 0 in phase-14 files | ✓ PASS (no phase-14 regressions) |
| `ruff format --check .` | repo root | 7 files would reformat: 6 pre-existing (documented in `deferred-items.md`) + `import_campaign_csv_demo.ipynb` (pre-commit ruff-format version-skew, documented, cosmetic-only, passes the actual pinned pre-commit hook) | ⚠️ see note |
| Fixture/notebook PII scan | regex email extraction over fixture CSV + notebook JSON | Only `@example.com`/`@example.org` addresses found | ✓ PASS |
| Notebook avoids SPICE-triggering imports | grep `solsys_code.views` / `solsys_code.ephem_utils` in notebook JSON | Neither found | ✓ PASS |

**Note on `ruff format --check .`:** `import_campaign_csv_demo.ipynb` shows as "would reformat" under
the repo's locally-installed `ruff` (0.12.9) but passes under the version actually pinned and enforced
by `.pre-commit-config.yaml` (`ruff-pre-commit rev: v0.2.1`) — confirmed both by `14-REVIEW-FIX.md`'s
own investigation and by the fact that the commit creating/touching this file passed the real
pre-commit hook at commit time (no `--no-verify` used, per the commit history). This is a pre-existing,
documented repo-wide version-skew condition (`deferred-items.md`), not a phase-14 regression, and
CLAUDE.md's authoritative gate is the pre-commit-pinned version. Not a blocker.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| CAMP-01 | 14-01 | `CampaignRun` model with full 3I-sheet field inventory | ✓ SATISFIED | `models.py:31-128`; `test_campaign_models.py::TestCampaignRunFieldInventory` |
| CAMP-02 | 14-01, 14-02 | Optional `Target` FK; single-target auto-assignment | ✓ SATISFIED | `models.py:68-75`; `import_campaign_csv.py:69`; `test_auto_resolves_single_target_campaign` |
| CAMP-03 | 14-01 | Lifecycle + approval status | ✓ SATISFIED (see Truth 1 note re: two-field vs. single-vocabulary wording) | `models.py:42-59`; `test_campaign_models.py::TestCampaignRunStatusVocabulary` |
| CAMP-04 | 14-02 | Bootstrap-import management command | ✓ SATISFIED | `import_campaign_csv.py`; `campaign_utils.py`; 39 tests |
| CAMP-05 | 14-03 | Paired demo notebook, no real PII | ✓ SATISFIED | fixture + notebook, PII scan clean |

All 5 requirement IDs mapped to Phase 14 in `.planning/REQUIREMENTS.md` (lines 12-16, checked `[x]`)
appear in at least one plan's `requirements:` frontmatter. No orphaned requirements found.

### Anti-Patterns Found

None blocking. Scanned `models.py`, `campaign_utils.py`, `import_campaign_csv.py`,
`test_campaign_models.py`, `test_import_campaign_csv.py` for `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/
`PLACEHOLDER`/`console.log`-equivalent/empty-return patterns — none found in production code. The
fixture CSV intentionally contains the literal string `TBD` once (`Publication Plans` cell,
`"TBD; pending team discussion"`) — this is realistic synthetic *data*, not a code debt marker, and is
outside the debt-marker gate's scope (it isn't a code comment).

### Human Verification Required

None. All four observable truths are backed by re-run automated tests, direct source inspection, and
re-executed command/notebook output, not just SUMMARY claims — nothing here requires human behavioral
testing to close out this phase.

**Non-gating follow-up recommendation (documentation only, does not block phase completion):**
Update ROADMAP.md phase 14's success criterion 1 and REQUIREMENTS.md's CAMP-03 wording to say "two
independent controlled-vocabulary fields (approval status + run status)" instead of "a single
controlled vocabulary" — see the Truth 1 note above. The implementation (deliberate discuss-phase
decision D-02) is correct and fully tested; only the requirement-doc phrasing has drifted from it.

### Gaps Summary

No gaps found. All four ROADMAP success criteria are observably true in the current codebase, not just
claimed in SUMMARY.md. The two critical bugs and nine warnings found by the deep code review
(`14-REVIEW.md`) were independently re-verified as fixed in the current source (not merely claimed
fixed) — the CR-01/CR-02 regression tests were re-run individually and pass, the `resolve_site`
exception-handling broadening (WR-01/02/03/04) and the `WR-05` unique constraint are present in the
current `campaign_utils.py`/`models.py`, and the full 242-test `solsys_code` suite plus the two
phase-specific test modules were all re-run fresh (not read from a log) with all tests passing. The one
open item is a non-blocking documentation-wording question (see Human Verification), which does not
gate phase completion — the implementation itself is correct and coherent.

---

_Verified: 2026-07-03T08:41:33Z_
_Verifier: Claude (gsd-verifier)_
