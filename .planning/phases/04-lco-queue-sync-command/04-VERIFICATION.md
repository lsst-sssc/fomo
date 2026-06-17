---
phase: 04-lco-queue-sync-command
verified: 2026-06-17T22:05:53Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
---

# Phase 4: LCO Queue Sync Command Verification Report

**Phase Goal:** Users can run `sync_lco_observation_calendar --proposal <code>` to sync FTS/MuSCAT4 queue records to the FOMO calendar, with each ObservationRecord represented as a CalendarEvent that transitions from an unscheduled banner to a placed block as the LCO scheduler acts, and is marked with a status prefix on reaching a terminal state
**Verified:** 2026-06-17T22:05:53Z
**Status:** passed
**Re-verification:** No — initial verification

**Important context for this verification:** After the executor completed plan 04-01, code review (04-REVIEW.md) flagged a Critical bug (CR-01: inconsistent `scheduled_start`/`scheduled_end` could crash the whole sync run with an unhandled `IntegrityError`). The orchestrator applied a direct fix on top (commit `9d88a54`) that is **not reflected in 04-01-SUMMARY.md**. This verification reads the current state of the files on disk (which includes the fix) rather than trusting the SUMMARY, per the goal-backward mandate. The fix, the new regression test, and the two accompanying warning fixes (WR-01, WR-02) were independently confirmed against source and a live test run, not assumed from the prompt's description.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running `sync_lco_observation_calendar --proposal <code>` creates one CalendarEvent per matching LCO ObservationRecord and none for non-matching records (SELECT-01, SYNC-01) | VERIFIED | `solsys_code/management/commands/sync_lco_observation_calendar.py:194` filters `ObservationRecord.objects.filter(facility='LCO', parameters__proposal=proposal)`. `test_select_01_only_matching_proposal_creates_events` creates one matching + one non-matching record and asserts `CalendarEvent.objects.count() == 1`. Ran live: PASS. |
| 2 | `CalendarEvent.url` equals `LCOFacility().get_observation_url(observation_id)`, used as the find-or-create key (SYNC-01, D-01) | VERIFIED | Line 141: `url = facility.get_observation_url(record.observation_id)`; line 205: `get_or_create(url=url, defaults=fields)`. Confirmed live: `LCOFacility().get_observation_url('12345')` → `https://observe.lco.global/requests/12345` (no `requestgroups`, no trailing slash — matches D-01's correction of the literal ROADMAP wording). `grep -c requestgroups` on the source = 0. `test_sync_01_d01_url_uses_requests_path_not_requestgroups` PASS. |
| 3 | Unscheduled record (`scheduled_start is None`) yields times from `parameters['start']/['end']` and a `[QUEUED]`-prefixed title (SYNC-02, D-03) | VERIFIED | `_time_window` (lines 91-119) and `_title_for` (lines 70-88) implement this branch exactly. `test_sync_02_d03_unscheduled_uses_parameters_times_and_queued_title` asserts exact UTC datetimes and `title == '[QUEUED] FTS 2M0-SCICAM-MUSCAT'`. PASS. |
| 4 | Placed record (`scheduled_start` populated) yields times from `scheduled_start/scheduled_end` and a clean title (SYNC-03, D-03) | VERIFIED | Same functions, populated branch. `test_sync_03_d03_placed_uses_scheduled_times_and_clean_title` asserts times equal the scheduled values and title has no `[QUEUED]`. PASS. |
| 5 | Re-running after a reschedule updates the existing event in place with no duplicate and no modified-timestamp churn on unchanged records (SYNC-04) | VERIFIED | Lines 204-216: `get_or_create` then compares all 7 fields before `.save()`; unchanged records take the `unchanged_count` branch with no save call. `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` runs the command twice, reschedules one of two records, and asserts: count stays 2, rescheduled event's `modified` changes, unchanged event's `modified` is bit-for-bit identical, and stdout reports `updated: 1` / `unchanged: 1`. PASS. |
| 6 | `telescope`, `instrument`, `proposal` on CalendarEvent are populated from the record (SYNC-05) | VERIFIED | `_build_event_fields` (lines 122-158) sets all three from `SITE_TELESCOPE_MAP[parameters['site']]`, `parameters['instrument_type']`, `parameters['proposal']`. `test_sync_05_telescope_instrument_proposal_populated` asserts all three values. PASS. |
| 7 | Records in WINDOW_EXPIRED/CANCELED/FAILURE_LIMIT_REACHED/NOT_ATTEMPTED get `[EXPIRED]`/`[CANCELLED]`/`[FAILED]` title prefix and the event is retained; COMPLETED gets a clean title (TERM-01, D-04, D-06 research correction) | VERIFIED | `_failure_prefix` checks membership in `facility.get_failed_observing_states()` (confirmed live: returns exactly the 4 failure states, NOT including `COMPLETED`; `get_terminal_observing_states()` returns those 4 + `COMPLETED` = 5, confirming the D-06 premise). `grep -c get_terminal_observing_states` outside comments = 0 (prefix trigger correctly uses the 4-state failure list, not the 5-state terminal list). 4 dedicated TERM-01 tests + 1 D-06 COMPLETED-clean-title test all PASS, each also asserting `CalendarEvent.objects.count() == 1` (retained, not deleted). |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | `sync_lco_observation_calendar` command + `SITE_TELESCOPE_MAP`, `class Command`, ≥100 lines | VERIFIED | 225 lines. Contains `class Command(BaseCommand)`, `add_arguments` with required `--proposal`, full `handle()` implementation with per-record try/except skip path. |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` | Django TestCase covering SELECT-01, SYNC-01..05, TERM-01, ≥100 lines | VERIFIED | 365 lines, `class TestSyncLcoObservationCalendar(TestCase)`, 15 test methods (plan specified ≥10; SUMMARY claimed 14; actual file on disk has 15 — the extra one is the post-review CR-01 regression test `test_skip_path_inconsistent_scheduled_times_logged_and_skipped`, added by the orchestrator's fix commit and correctly not claimed by the SUMMARY since it predates the fix). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `sync_lco_observation_calendar.py` | `ObservationRecord.objects.filter` | `facility='LCO', parameters__proposal=code` | WIRED | Line 194, exact pattern present; confirmed by `test_select_01...` passing against real ORM JSONField lookup (SQLite). |
| `sync_lco_observation_calendar.py` | `LCOFacility().get_observation_url` | URL construction for the create-or-update key | WIRED | Line 141; library call confirmed live to return `/requests/<id>` format (no `requestgroups`). |
| `sync_lco_observation_calendar.py` | `CalendarEvent.objects.get_or_create` | find-by-url, then create or conditional update | WIRED | Line 205, `get_or_create(url=url, defaults=fields)` followed by a 7-field diff-then-save block (lines 208-216) — confirmed by the no-churn `modified`-timestamp test. |

### Data-Flow Trace (Level 4)

Not applicable in the React/UI sense — this is a server-side batch sync command, not a component rendering fetched state. The equivalent trace (ORM query → field derivation → model write) is covered by the Key Link table above and confirmed via live database reads in the test run (`CalendarEvent.objects.get()` assertions against real saved rows, not mocks).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `_failure_prefix` correctly distinguishes the 4 LCO failure states from the 5 terminal states (D-06 premise) | `LCOFacility().get_failed_observing_states()` vs `get_terminal_observing_states()` run live in a Django shell | `['WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED']` vs the same 4 + `['COMPLETED']` | PASS |
| `LCOFacility().get_observation_url` produces the real URL format used by D-01 | `LCOFacility().get_observation_url('12345')` run live | `https://observe.lco.global/requests/12345` | PASS |
| `CalendarEvent.start_time`/`end_time` are non-nullable at the model level (confirms why CR-01 was a real bug, not a hypothetical) | inspected `CalendarEvent._meta.get_fields()` live | `start_time False, end_time False` (i.e. `null=False`) confirms an unhandled `None` would raise `IntegrityError` pre-fix | PASS |
| New CR-01 regression test passes in isolation | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_skip_path_inconsistent_scheduled_times_logged_and_skipped` | `Ran 1 test ... OK` | PASS |
| Pre-fix code (commit `1b49957`) is confirmed to have lacked the inconsistent-state branch | `git show 1b49957:.../sync_lco_observation_calendar.py` inspected | Pre-fix `_time_window` only branches on `scheduled_start is None`, would produce `end_time=None` for the asymmetric case — confirms CR-01 was real, not a false positive in the review | PASS |
| Full sync command test module | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` | `Ran 15 tests ... OK` | PASS |
| Full `solsys_code` Django suite | `python manage.py test solsys_code` | `Ran 110 tests ... OK` | PASS (matches the prompt's claimed 110/110, independently reproduced) |
| Ruff lint/format on the two phase files | `ruff check ...; ruff format --check ...` | `All checks passed!`; `2 files already formatted` | PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` convention exists in this repository and none is declared in the PLAN/SUMMARY for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SELECT-01 | 04-01-PLAN.md | `--proposal` filters `ObservationRecord(facility='LCO')` by `parameters['proposal']` | SATISFIED | Truth #1 |
| SYNC-01 | 04-01-PLAN.md | One CalendarEvent per record, keyed on `url` | SATISFIED | Truth #2 |
| SYNC-02 | 04-01-PLAN.md | Unscheduled → `parameters['start']/['end']`, queue-status title | SATISFIED | Truth #3 |
| SYNC-03 | 04-01-PLAN.md | Placed → `scheduled_start/scheduled_end`, times updated in existing event | SATISFIED | Truth #4 |
| SYNC-04 | 04-01-PLAN.md | Re-run updates in place, no duplicate, no `modified` churn | SATISFIED | Truth #5 |
| SYNC-05 | 04-01-PLAN.md | `instrument`/`proposal`/`telescope` populated | SATISFIED | Truth #6 |
| TERM-01 | 04-01-PLAN.md | Terminal-failure states get title prefix, event retained | SATISFIED | Truth #7 |

No orphaned requirements: REQUIREMENTS.md's "Future Requirements" section (TARG-01, TARG-02) is explicitly out of scope for Phase 4 and not claimed by the plan. All 7 v1.2 requirement IDs are claimed by 04-01-PLAN.md's `requirements:` frontmatter and all 7 are satisfied in code.

**Documentation staleness note (non-blocking):** `.planning/REQUIREMENTS.md`'s checkboxes (`- [ ]`) and its Traceability table (`Status: Pending` for all 7 IDs) were not updated after Phase 4 completed, even though ROADMAP.md correctly shows Phase 4 as `[x]` complete and all 7 requirements are demonstrably satisfied in code and tests. This is a documentation-sync gap in `.planning/REQUIREMENTS.md`, not a code or goal-achievement gap — flagged for housekeeping, does not affect phase status.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found (`TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`/empty-implementation scans all clean) | — | — |

### Post-Review Fix Verification (CR-01, WR-01, WR-02)

The prompt notes that `9d88a54` is not reflected in 04-01-SUMMARY.md. Independent verification of this fix, confirmed against source and a live test run rather than the prompt's narrative:

- **CR-01 (Critical — fixed):** `_time_window` (lines 91-119) now requires both `scheduled_start` and `scheduled_end` to be either both `None` or both populated; the asymmetric case raises `ValueError`, caught by the existing `except (KeyError, ValueError)` in `handle()` (lines 197-202) — confirmed this routes to the skip path, not a crash, via the new regression test passing live.
- **WR-01 (Warning — fixed):** `_failure_prefix` (line 49) now uses `.get(status, '[FAILED]')` instead of `[status]`, so an unmapped future failure state degrades to a generic `[FAILED]` prefix instead of raising an uncaught `KeyError` from the dict lookup; the misleading "automatically" comment was corrected (lines 24-27).
- **WR-02 (Warning — fixed):** `_derive_telescope` (lines 64-67) now raises a descriptive `KeyError(f'Unmapped LCO site code {site_code!r}; add it to SITE_TELESCOPE_MAP')` instead of a bare `KeyError`.
- **IN-01 (Info — not fixed, not required):** Skip-path test coverage still only directly exercises `site` (pre-existing) and the new inconsistent-schedule case; missing `instrument_type`/`proposal`/malformed `start`/`end` parametrized cases remain untested. Info-severity, non-blocking.
- **IN-02 (Info — not fixed, not required):** No `transaction.atomic()` or DB-level unique constraint on `CalendarEvent.url` — concurrent runs could theoretically race. Info-severity, single-operator CLI tool per the plan's threat model (T-04-05 accepted), non-blocking.

### Human Verification Required

None. All must-haves are verifiable programmatically and were independently confirmed by running the actual test suite and inspecting library behavior live (not just reading SUMMARY.md claims). The plan's `<verification>` item 3 ("Manual spot-check ... optional, end-of-phase human verify") is explicitly marked optional in the plan itself, and the automated test suite already exercises the equivalent create/update/title/skip behaviors end-to-end against a real database, so it does not need to gate phase status.

### Gaps Summary

No gaps. All 7 observable truths tied to the 7 phase requirements (SELECT-01, SYNC-01..05, TERM-01) are verified against the current state of the source files on disk, including the post-review CR-01 fix that the SUMMARY.md does not mention. The full `solsys_code` suite passes at 110/110 (independently reproduced, matching the prompt's claim), ruff is clean on both phase files, and the one Critical review finding has a verified, working fix with a passing regression test. Two Info-level review findings (IN-01, IN-02) remain open by design (review classified them as non-blocking) and one documentation-sync issue (REQUIREMENTS.md checkboxes/traceability table not updated) is noted for housekeeping but does not block phase completion.

---

*Verified: 2026-06-17T22:05:53Z*
*Verifier: Claude (gsd-verifier)*
