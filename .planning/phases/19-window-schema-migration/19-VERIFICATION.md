---
phase: 19-window-schema-migration
verified: 2026-07-10T06:46:25Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 19: Window-Schema Migration Verification Report

**Phase Goal:** Replace CampaignRun's single-night obs_date/ut_start/ut_end representation with a nullable window (window_start/window_end), migrating every existing row with no data loss.
**Verified:** 2026-07-10T06:46:25Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A CampaignRun can be saved with window_start == window_end (single classically-scheduled night) | ✓ VERIFIED | `models.py:89-90` defines both fields as nullable `DateField`s; `test_campaign_models.py::TestCampaignRunFieldInventory.test_full_field_inventory_persists_and_reloads` — ran directly, passes. |
| 2 | A CampaignRun can be saved in a "TBD" state (both window fields null), distinct from a resolved window | ✓ VERIFIED | `test_campaign_models.py::TestCampaignRunWindowSchema.test_tbd_run_saves_with_both_window_fields_null` — ran directly, passes. |
| 3 | Two distinct TBD rows for the same campaign + telescope neither silently merge nor silently duplicate under the DB constraint | ✓ VERIFIED | Two partial `UniqueConstraint`s in `models.py:124-139` (`unique_campaign_run_resolved_window`, `unique_campaign_run_tbd_natural_key`). TBD branch keys on `(campaign, telescope_instrument, contact_person)` per Phase 18's locked decision (`19-CONTEXT.md` D-07/D-08) — same-key TBD rows raise `IntegrityError` (`test_tbd_same_contact_person_collides`, ran directly, passes: raises `IntegrityError`, no row created, no silent merge); differing-`contact_person` TBD rows both persist as two genuinely distinct rows (`test_tbd_differing_contact_person_both_save`, passes). This is not a silent duplicate — the discriminator (contact_person) is a deliberate, Phase-18-locked design choice, not scope creep. |
| 4 | Every existing CampaignRun row survives the migration with window_start == window_end == former obs_date — no data loss | ✓ VERIFIED (see Gaps Summary caveat) | Migration 0004 actually applied to the real dev DB (`src/fomo_db.sqlite3`, confirmed via `showmigrations`: `[X] 0004_campaignrun_window_schema`); direct SQL query against the live DB found **zero** resolved-window duplicate groups and 14 surviving rows (16 pre-migration minus the 2 known TBD duplicate pks 17/18, matching both 19-01-SUMMARY.md's and 19-04-SUMMARY.md's reported outcome exactly). `dedupe_tbd_collisions` logs every deletion (D-08 audit trail). See Gaps Summary for a known, user-accepted robustness gap (CR-01) in the migration's dedup logic that did not manifest against the actual data but is not fully closed for other environments. |
| 5 | The existing per-campaign table, approval queue, and coverage-gap pages still render correctly against the new window fields | ✓ VERIFIED | `campaign_tables.py::render_window_start` (TBD badge / single date / `-&gt;` range) exercised by `test_campaign_views.py::TestWindowColumnRendering` (3 tests, ran directly, all pass); nulls-last sort exercised by `test_default_sort_is_window_start_desc_tbd_last` (passes); approval queue + D-06 calendar projection exercised by `test_campaign_approval.py::TestCalendarProjection` (5 tests referenced in 19-03-SUMMARY.md); coverage-gap `claimed_dates()` exercised by `test_campaign_gap.py::TestClaimedDates`/`TestClaimedDatesMultiTarget`. Full `./manage.py test solsys_code` suite (355 tests) run directly by this verifier: **OK, 0 failures**. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/models.py` | window_start/window_end DateFields + two partial UniqueConstraints | ✓ VERIFIED | Confirmed fields at lines 89-90, constraints at 124-139, no `obs_date`/`ut_start`/`ut_end` remain (grep confirmed). |
| `solsys_code/migrations/0004_campaignrun_window_schema.py` | Single combined migration: backfill + dedup before constraint swap | ✓ VERIFIED (wired, applied) | Operation order confirmed by direct read: AddField x2 → RunPython(backfill) → RunPython(dedup) → RemoveConstraint → RemoveField x3 → AddConstraint x2. `makemigrations --check --dry-run` exits 0 (no drift). Applied to the real dev DB. Known gap: dedup only covers the TBD branch, not a symmetric resolved-window collision case (CR-01, see Gaps Summary). |
| `solsys_code/campaign_gap.py` | claimed_dates() reads window fields; `_observing_night_date` removed | ✓ VERIFIED | `.only('pk', 'window_start', 'window_end')` at line 162; inclusive range-claim loop at 174-183; `_observing_night_date` absent (grep confirmed). |
| `solsys_code/campaign_tables.py` | render_window_start; single window_start column | ✓ VERIFIED | `render_window_start` at line 136, TBD/single/range branches confirmed by direct read. |
| `solsys_code/campaign_views.py` | nulls-last sort, PII allowlist swap, D-06 projection | ✓ VERIFIED | `ALLOWED_FIELDS_FOR_NON_STAFF` lists window_start/window_end (grep confirmed); D-06 hybrid ground/space gate present per SUMMARY and test evidence. |
| `solsys_code/campaign_forms.py` | Collapsed submission form (single date, no UT fields) | ✓ VERIFIED | `obs_date` DateField only, no `ut_start`/`ut_end`/`DateTimeField` (grep confirmed). |
| `solsys_code/management/commands/import_campaign_csv.py` | window_start natural-key lookup, log-and-skip collisions | ✓ VERIFIED | Lookup dict keys on `window_start` (grep confirmed line 171); `window_end` set in fields dict; no sub-second offset remains. Known gap: lookup omits `window_end` from the key (WR-01), currently unreachable since every write path this phase ships sets `window_end == window_start`. |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | Regenerated notebook, committed with executed output | ✓ VERIFIED | Contains `window_start`/`window_end` (2 occurrences), zero `obs_date`/`ut_start`/`ut_end` references (grep confirmed); regeneration commit `a295b3e` per SUMMARY. |
| `solsys_code/tests/test_campaign_models.py` | SCHED-02/03/04 assertions | ✓ VERIFIED | Ran directly: 10/10 tests pass, including all 4 constraint/TBD/resolved-window cases. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| Migration 0004 RunPython steps | AddConstraint operations | Operation ordering | ✓ WIRED | Dedup (op index 4) strictly precedes both AddConstraint ops (indices 9-10) — confirmed by reading the operations list directly. |
| `campaign_tables.render_window_start` | `CampaignRun.window_start`/`window_end` | `Accessor(...).resolve(record, quiet=True)` | ✓ WIRED | Dict-vs-model dual accessor confirmed at lines 143-149. |
| `CampaignRunTableView.get_queryset` | `get_table_kwargs` | `F('window_start').desc(nulls_last=True)` + `order_by: ()` suppression | ✓ WIRED | Confirmed present via SUMMARY + grep; test `test_default_sort_is_window_start_desc_tbd_last` passes. |
| `CampaignRunDecisionView.post` | `telescope_runs.sun_event` | D-06 ground-branch projection call | ✓ WIRED | Import + `Observatory.SATELLITE_OBSTYPE` branch confirmed via grep; `except ValueError` isolation confirmed distinct from the broad `except Exception` per SUMMARY and test `test_sun_event_valueerror_skips_projection_without_reverting_approval`. |
| `import_campaign_csv` lookup dict | `insert_or_create_campaign_run` | `window_start` key | ✓ WIRED | Confirmed at line 171: `{'campaign': ..., 'telescope_instrument': ..., 'window_start': obs_date}`. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Model/constraint tests (SCHED-02/03/04) | `python manage.py test solsys_code.tests.test_campaign_models` | 10/10 pass | ✓ PASS |
| `makemigrations` drift check | `python manage.py makemigrations solsys_code --check --dry-run` | exit 0, "No changes detected" | ✓ PASS |
| Full app suite (wave-merge gate) | `python manage.py test solsys_code` | 355/355 pass, 0 failures | ✓ PASS |
| Real dev DB migration state | `python manage.py showmigrations solsys_code` | `[X] 0004_campaignrun_window_schema` applied | ✓ PASS |
| Real dev DB resolved-window duplicate check | Direct SQL: `GROUP BY campaign_id, telescope_instrument, window_start, window_end HAVING COUNT(*) > 1` | `[]` (zero collision groups), 14 total rows | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SCHED-02 | 19-01, 19-02, 19-03, 19-04 | Single-night representation as window_start == window_end | ✓ SATISFIED | Model fields, tests, table/view/form rewrite, CSV importer all confirmed. |
| SCHED-03 | 19-01, 19-02, 19-03 | CampaignRun can exist fully TBD, distinct from resolved window | ✓ SATISFIED | Model test + claimed_dates() TBD bucketing + table TBD badge confirmed. |
| SCHED-04 | 19-01, 19-04 | Partial UniqueConstraint closes NULL-uniqueness gap | ✓ SATISFIED | Both partial constraints confirmed present and tested; CSV import collision handling confirmed. |
| SCHED-05 | 19-01 | Existing rows migrate with no data loss | ✓ SATISFIED | Migration applied to real dev DB, verified via direct SQL — see Gaps Summary caveat on CR-01. |

`REQUIREMENTS.md` maps exactly these 4 IDs to Phase 19 (lines 71-74); no orphaned requirement IDs found for this phase.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/migrations/0004_campaignrun_window_schema.py` | 17-47, 93-100 | CR-01 (from `19-REVIEW.md`, deep code review): dedup covers only the TBD branch, not the structurally identical resolved-window collision the backfill also creates | ⚠️ WARNING (deferred by explicit user decision, tracked via `/gsd-code-review 19 --fix`) | Did not manifest against the actual dev DB (verified: zero resolved-window collision groups after migration). Remains a latent risk for a fresh deploy/staging DB with a data shape not yet seen (two pre-migration rows sharing `obs_date` but distinct `ut_start`). Failure mode is a loud `IntegrityError` at migrate time (migration refuses to apply), not silent data corruption. |
| `solsys_code/management/commands/import_campaign_csv.py` | 170-173 | WR-01 (from `19-REVIEW.md`): natural-key lookup omits `window_end` | ℹ️ INFO | Currently unreachable — every write path this phase ships always sets `window_end == window_start`. Becomes live once Phase 20 introduces real ranges. |
| `solsys_code/models.py` / `campaign_gap.py:176-183` | — | WR-02 (from `19-REVIEW.md`): no CheckConstraint or defensive guard against a `window_start` set / `window_end` null mismatch | ℹ️ INFO | Currently unreachable via any shipped write path; would raise `TypeError` in `claimed_dates()` if it ever occurred. |
| `solsys_code/migrations/0004_campaignrun_window_schema.py` | 17-47 | WR-03 (from `19-REVIEW.md`): migration RunPython logic has zero automated regression coverage | ℹ️ INFO | Relies on a manual dry-run (documented in 19-01-SUMMARY.md and independently re-confirmed by this verifier against the live dev DB), not a repeatable test artifact. |

No `TBD`/`FIXME`/`XXX` debt markers found in phase-touched files (grep matches for "TBD" are all legitimate identifiers — e.g. `unique_campaign_run_tbd_natural_key` — not debt markers). No stub/placeholder rendering patterns found; `render_window_start` and the D-06 projection branches are fully implemented, not scaffolding.

### Human Verification Required

None. Every observable truth in this phase is verifiable via automated Django tests and direct inspection of rendered HTML strings (`render_window_start`'s exact `-&gt;`/TBD-badge output is asserted by unit tests, not left to visual judgment), and this verifier independently confirmed the live dev DB's post-migration state by direct SQL query rather than relying on SUMMARY.md's narrative.

### Gaps Summary

No blocking gaps. All 5 roadmap Success Criteria and all 4 requirement IDs (SCHED-02/03/04/05) are independently verified against the actual codebase (not SUMMARY.md claims): the model/migration/constraint layer, all four consumer surfaces (coverage-gap, table/view/form/calendar-projection, CSV import + demo notebook), and the real dev DB's actual post-migration state were each checked directly by this verifier — including running the full 355-test suite and a direct SQL scan of the live database for the exact collision shape CR-01 describes, which found none.

**On CR-01 specifically** (the one Critical finding from the phase's deep code review): this verifier independently confirmed it did not cause data loss against the data that actually existed — the live dev DB has zero resolved-window duplicate groups post-migration, and the migration's own failure mode for this gap is a loud `IntegrityError` (migration refuses to apply), never a silent row loss. The user's decision to defer the fix (tracked in `19-REVIEW.md`, actionable via `/gsd-code-review 19 --fix`) is respected and not re-litigated here. It remains a real latent robustness gap for any future environment with pre-migration rows sharing `obs_date` but differing `ut_start` (e.g. a fresh deploy re-importing richer historical CSV data before migrating) — worth prioritizing before Phase 19's migration is replayed against any database other than the one already verified here.

---

_Verified: 2026-07-10T06:46:25Z_
_Verifier: Claude (gsd-verifier)_
