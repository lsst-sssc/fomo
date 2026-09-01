---
phase: 27-the-canonical-run-record
plan: 02
subsystem: api
tags: [django, management-command, migration, data-repair, timezone, mpc]

# Dependency graph
requires:
  - phase: 26-canonical-record-spike
    provides: "D-15/D-16/D-16a/D-16b/D-17/D-22/D-23 locked verdicts on the site-repair scope, create_placeholder fail-safe, and the E10 timezone gap"
provides:
  - "repair_stale_campaign_run_sites management command -- one-time re-resolution of every approved, site-less CampaignRun through the real resolve_site() path, run once against the dev DB"
  - "Observatory.timezone backfilled from coordinates for every blank-timezone row with known lat/lon (migration 0003), closing the E10 (Siding Spring) gap Phase 29's reconciler needs"
affects: [27-04-window-schema-migration-and-telescope-class-backfill, 29-reconciler]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Read-only tier-1-only probe function mirrors a stateful helper's early steps (alias translation + length guard) without calling the stateful helper itself, so --dry-run can report intent without triggering the real function's side effects (Observatory creation on a tier-2 hit)"
    - "AST-based (not text-based) proof that a security-relevant keyword argument is a literal at every call site, immune to comment mentions or ruff line-wrapping"

key-files:
  created:
    - solsys_code/management/commands/repair_stale_campaign_run_sites.py
    - solsys_code/tests/test_repair_stale_campaign_run_sites.py
    - solsys_code/solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py
    - solsys_code/solsys_code_observatory/tests/test_timezone_backfill_migration.py
  modified: []

key-decisions:
  - "D-22's mutation proof (temporarily flip create_placeholder=False to True, confirm the network-failure test fails, revert, confirm green) was run manually during execution rather than left as a permanent CI step -- confirmed the assertion is load-bearing: under mutation, Observatory.objects.count() delta became 1 and site was non-None (a placeholder was fabricated); reverted, all 7 tests pass again, and the file diffed byte-identical to its pre-mutation state"
  - "The live repair (Task 2) produced no git diff by design -- src/fomo_db.sqlite3 is gitignored -- so Task 2 has no code commit; its evidence is the before/after table below plus the dry-run/idempotency proof, matching the plan's own framing (dev-DB-only, D-16a)"
  - "The MPC Obscodes API was reachable during the live run, so both HST rows (pk 8, 12) and the Swift row (pk 13) resolved through genuine tier-2 lookups, creating 2 new real (non-placeholder) Observatory rows -- the D-22 fail-safe path (network unreachable) was proven separately via the offline mocked test, not on the live DB"

patterns-established:
  - "A one-time repair/backfill command's --dry-run mode never calls the same function the real run calls when that function has a side effect (row creation) on success -- it implements a narrower, explicitly-limited read-only probe and documents the limitation in --help text"

requirements-completed: [CANON-02]

# Metrics
duration: ~25min
completed: 2026-07-30
---

# Phase 27 Plan 02: Site Repair & Timezone Backfill Summary

**One-time repair command re-resolved 6 of 9 approved site-less CampaignRuns (JWST offline via alias, HST/Swift via live MPC tier-2) with create_placeholder=False as the D-22 fail-safe, plus a coordinate-derived Observatory.timezone backfill migration closing the E10 (Siding Spring) gap.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-30T00:13:14Z
- **Completed:** 2026-07-30T00:26:53Z
- **Tasks:** 3 completed (2 code commits + 1 live-DB-only operation with no git diff)
- **Files modified:** 4 created, 0 modified

## Accomplishments

- Built `repair_stale_campaign_run_sites`, a one-off `BaseCommand` that re-resolves every `approval_status=APPROVED, site__isnull=True` `CampaignRun` through the real `resolve_site(..., create_placeholder=False)` path (D-16, D-22), applies the D-16b owner-supplied `site_raw='C52'` correction for Swift only when `site_raw` is blank, and never touches `approval_status`/`run_status`/window fields/`target`, and never projects a calendar event.
- 7 offline mocked tests cover every D-16 dev-DB row shape (HST, Swift, JWST, JUICE), the D-22 network-failure proof (zero new `Observatory` rows, row stays flagged), `--dry-run` (writes nothing), and the D-15 rejected-row-untouched case.
- Ran the D-22 mutation proof manually: flipped `create_placeholder=False` to `True`, re-ran the network-failure test, confirmed it **failed** (a placeholder `Observatory` for HST was fabricated: `site` became non-`None`, `Observatory.objects.count()` delta became 1), reverted, confirmed the module green again, and confirmed the reverted file was byte-identical to its pre-mutation state.
- Ran the live one-time repair against the real dev DB (backed up first). The MPC Obscodes API was reachable: HST (pk 8, 12) and Swift (pk 13, after the D-16b `site_raw='C52'` correction) resolved through genuine live tier-2 lookups; JWST (pk 21, 27, 28) resolved offline via the existing alias-to-Observatory-274 path with zero network calls. JUICE (pk 26) and the two class-wide rows (pk 29, 30) stayed site-less with no site code to resolve, exactly as D-16 predicted. The rejected `X05` row (pk 31) was never touched.
- Added migration `0003_backfill_observatory_timezone` to `solsys_code_observatory`: derives an IANA timezone name via `timezonefinder` (reusing the existing `_get_timezone_finder()` cache) for every `Observatory` with a blank `timezone` and known `lat`/`lon`. Applied against the real dev DB: `E10` (Siding Spring) is now `'Australia/Sydney'`, plus five other previously-blank rows (`F65` Haleakala -> `Pacific/Honolulu`, `253`, `X05`, `K92`, `K93`, `500`) picked up correctly by the same derived rule.
- 3 new `MigrationExecutor`-based tests prove the E10 backfill, the coordinate-less-row-stays-blank case, and the never-overwrite-an-existing-timezone case.

## Task Commits

Each code-producing task was committed atomically; Task 2 (the live DB repair) produced no git diff by design (`src/fomo_db.sqlite3` is gitignored) and has no commit of its own.

1. **Task 1: Build the repair_stale_campaign_run_sites command with offline mocked tests** - `ed11ce6` (feat)
2. **Task 2: Run the one-time live repair against the dev DB** - no commit (gitignored DB; see before/after table below)
3. **Task 3: Backfill Observatory.timezone from coordinates (D-23)** - `9c75484` (feat), committed separately from Tasks 1-2 per D-23

_No TDD tasks in this plan (autonomous, non-TDD execute plan)._

## Files Created/Modified

- `solsys_code/management/commands/repair_stale_campaign_run_sites.py` - New one-off repair command: `_OWNER_SUPPLIED_SITE_RAW` (D-16b), `_first_instrument_token()`, `_probe_site_resolution()` (the read-only `--dry-run` tier-1-only probe), `Command.handle()`
- `solsys_code/tests/test_repair_stale_campaign_run_sites.py` - 7 tests: HST tier-2, Swift D-16b+tier-2, JWST offline-alias (no network call), JUICE skipped (no network call), D-22 network-failure proof, `--dry-run` writes-nothing, D-15 rejected-row-untouched
- `solsys_code/solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py` - New data migration: `backfill_observatory_timezone()` RunPython step, `reverse_code=migrations.RunPython.noop`
- `solsys_code/solsys_code_observatory/tests/test_timezone_backfill_migration.py` - 3 `MigrationExecutor`-based tests seeding the pre-0003 schema

## Live Repair: Before/After (Task 2, dev DB)

No contact fields (`contact_person`/`contact_email`) are printed anywhere below, per T-27-09.

**Before** (`approval_status=approved, site__isnull=True`, 9 rows):

| pk | telescope_instrument | site_raw | site_id | site_needs_review |
|----|----------------------|----------|---------|--------------------|
| 8  | Hubble\nWFC3/UVIS    | 250      | None    | True |
| 12 | HST STIS/COS         | 250      | None    | True |
| 13 | Swift/UVOT           | (blank)  | None    | True |
| 21 | JWST                 | 500@-170 | None    | True |
| 26 | JUICE                | (blank)  | None    | True |
| 27 | JWST                 | 500@-170 | None    | True |
| 28 | JWST                 | 500@-170 | None    | True |
| 29 | LCO 1m               | (blank)  | None    | True |
| 30 | LCO 2m               | (blank)  | None    | True |

Observatory count before: 22.

**`--dry-run` output** (writes nothing; row states after dry-run identical to before):
- pk 8, 12: `site_raw='250': would query MPC (live tier-2 lookup needed)`
- pk 13: `site_raw='C52': would query MPC (live tier-2 lookup needed) [site_raw would be set via D-16b owner-supplied correction]`
- pk 21, 27, 28: `site_raw='500@-170': would resolve OFFLINE to existing Observatory '274'`
- pk 26, 29, 30: `skipped (no site code)`

**Live run (real, non-dry-run):**

| pk | telescope_instrument | site_raw (after) | site (after) | site_needs_review (after) |
|----|----------------------|-------------------|---------------|------------------------------|
| 8  | Hubble\nWFC3/UVIS    | 250               | 250 (new)     | False |
| 12 | HST STIS/COS         | 250               | 250 (same row)| False |
| 13 | Swift/UVOT           | C52 (set, D-16b)  | C52 (new)     | False |
| 21 | JWST                 | 500@-170          | 274 (existing)| False |
| 26 | JUICE                | (blank)           | None          | True (unchanged) |
| 27 | JWST                 | 500@-170          | 274 (existing)| False |
| 28 | JWST                 | 500@-170          | 274 (existing)| False |
| 29 | LCO 1m               | (blank)           | None          | True (unchanged) |
| 30 | LCO 2m               | (blank)           | None          | True (unchanged) |

Observatory count after: 24 (delta +2: one new HST row shared by pk 8/12, one new Swift row for pk 13 -- matches D-16's predicted "creates one HST Observatory row").

**Divergence from D-16's table:** none. Every row resolved exactly as D-16 predicted; the MPC Obscodes API was reachable throughout, so the D-22 fail-safe path (network unreachable) was not exercised on the live DB -- it is proven separately by the offline mocked network-failure test and its mutation proof.

**Idempotency check:** re-running `repair_stale_campaign_run_sites` a second time reported `resolved: 0` -- the 3 remaining site-less rows (pk 26, 29, 30) were skipped again as `no site code available`, with no field changes.

**Verification queries:**
- `CampaignRun.objects.filter(approval_status='approved', site__isnull=True, site_raw='500@-170').count()` -> `0`
- pk 31 (rejected, `site_raw='X05'`): `approval_status='rejected', site_raw='X05', site_id=None, site_needs_review=False` -- untouched, exactly as before (D-15).

**Backup:** the dev DB was copied to `/tmp/fomo_db.sqlite3.pre-27-02.bak` before the live run (946,176 bytes, matches the pre-repair file size).

## D-23 Timezone Backfill: Live Dev-DB Result

Applied migration `0003_backfill_observatory_timezone` against the real dev DB. Every previously-blank-timezone row with coordinates was backfilled:

| obscode | derived timezone |
|---------|-------------------|
| E10 (Siding Spring) | Australia/Sydney |
| F65 (Haleakala) | Pacific/Honolulu |
| 253 | America/Los_Angeles |
| X05 | America/Santiago |
| K92 | Africa/Johannesburg |
| K93 | Africa/Johannesburg |
| 500 | Etc/GMT |

`Observatory.objects.get(obscode='E10').timezone` == `'Australia/Sydney'` -- confirmed directly against the dev DB. `./manage.py makemigrations --check --dry-run` reports "No changes detected" (the data migration implies no model change).

## Decisions Made

- Task 2's live repair intentionally produced no git-visible change (the dev DB is gitignored); its evidence lives entirely in this Summary's before/after table rather than a commit diff, matching the plan's own framing that this run is manual-by-nature and not CI-gated (27-VALIDATION.md, D-16a).
- The D-22 mutation proof was executed manually (edit -> run failing test -> revert -> confirm green -> confirm byte-identical) rather than committed as a permanent CI mutation-testing step, per the plan's acceptance criterion 3 wording ("Record both outcomes ... in the summary").
- `_probe_site_resolution()` (the `--dry-run` helper) deliberately does not call `resolve_site()` at all, even for a tier-1 hit case, to guarantee zero MPC calls and zero `Observatory` writes under `--dry-run` -- it duplicates only the alias-translation and length-guard steps, which are pure and side-effect-free, and is documented as a known limitation in `--help` text (a row needing tier 2 is reported as "would query MPC", not resolved).

## Deviations from Plan

None - plan executed exactly as written. All three tasks' acceptance criteria were met, including the D-22 AST proof (`ast`-based check confirming every `resolve_site()` call site passes the literal `create_placeholder=False`), the D-22 mutation proof (confirmed load-bearing), the D-15 untouched-rejected-row test, and the D-23 `makemigrations --check --dry-run` clean result.

## Issues Encountered

None specific to this plan. The pre-existing `./manage.py test solsys_code` segfault in `test_views.py::test_K93` (SPICE/ASSIST native code, unrelated to this plan's files) is a known environment issue documented by Plan 27-01 -- verified instead via the plan's own quick-run command (307 tests), the two new test modules plus `solsys_code_observatory` (35 tests), and a combined run of every `solsys_code.tests.*` module except `test_views`/`test_ephem_utils` (602 tests, all pass).

## User Setup Required

None - no external service configuration required. The live repair (Task 2) required a reachable network to reach the real MPC Obscodes API, which was available during this execution.

## Next Phase Readiness

- The 3 approved `CampaignRun`s that remain site-less (pk 26 JUICE, pk 29/30 class-wide LCO 1m/2m) are the genuinely-unresolvable rows D-16 predicted -- they have no site code to resolve and are expected to feed Plan 27-04's `telescope_class` backfill as `SPACE`/`1m0`/`2m0` respectively rather than as site data.
- `Observatory.timezone` is now populated for every coordinate-bearing row that previously lacked one, including `E10`, which Plan 29's reconciler needs for site-local-night key derivation.
- No blockers. `ruff check .`/`ruff format --check .` report the same pre-existing, unrelated issues present before this plan (1 notebook docstring warning, 7 files needing reformat -- none touched by this plan; `src/fomo/settings.py` is the user's local dev config and was never staged or committed).

---
*Phase: 27-the-canonical-run-record*
*Completed: 2026-07-30*

## Self-Check: PASSED

All 4 created files confirmed present on disk; both task commit hashes (`ed11ce6`, `9c75484`) confirmed present in `git log --oneline --all`.
