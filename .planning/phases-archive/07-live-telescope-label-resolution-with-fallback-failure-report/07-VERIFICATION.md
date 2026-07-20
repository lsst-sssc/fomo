---
phase: 07-live-telescope-label-resolution-with-fallback-failure-report
verified: 2026-06-24T04:31:27Z
status: passed
score: 13/13 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 7: Live Telescope-Label Resolution with Fallback & Failure Report Verification Report

**Phase Goal:** Every synced record gets a telescope label — the verified, API-resolved one when possible, a clearly-marked coarse fallback when not — and a degraded API call never aborts the run, hides its own failure, or leaks credentials
**Verified:** 2026-06-24T04:31:27Z
**Status:** passed
**Re-verification:** No — initial verification

## Context

This is a retroactive backfill verification. Phase 7 was fully executed (2 plans), security-audited (`SECURITY.md`, 7/7 threats closed — one real gap found and fixed via quick task `260623-ocs`), Nyquist-validated (`07-VALIDATION.md`, `nyquist_compliant: true`, 35/35 tests green), and UAT-tested (`07-UAT.md`, status `complete`, 6/6 tests passed — one real production-data gap found during live UAT and fixed via quick task `260623-su3`). This verification independently re-derives and checks all must-haves directly against the current codebase rather than trusting any of those prior documents' claims.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Verified `SITE_TELESCOPE_MAP` covers all real LCO-network sites this codebase's installed facility settings confirm, keyed by `(site, aperture_class)` | ✓ VERIFIED | `sync_lco_observation_calendar.py:33-48` — 14 entries covering 7 sites (`ogg`, `elp`, `lsc`, `cpt`, `coj`, `tfn`, `sor`); `tlv` deliberately excluded per documented operator decision (ROADMAP.md "Scope correction (Wave 1)" note, `07-01-SUMMARY.md` Deviations). The 3 legacy entries (`coj`/`ogg`/`sor`) migrated to `SITECODE-CLASS` format; no `'FTS'`/`'FTN'` strings remain (`grep` returns zero matches). `test_telescope_01_verified_dict_covers_all_sites` and `test_telescope_01_coj_ogg_full_aperture_class_coverage` pass. |
| 2 | A placed record's site/enclosure/telescope is resolved via a single timeout-bounded LCO API call | ✓ VERIFIED | `_resolve_placement_block` (`sync_lco_observation_calendar.py:148-194`) issues exactly one `make_request(...)` call with `timeout=_API_TIMEOUT_SECONDS` (10s), no loop/retry/backoff. `test_sync_08_single_attempt_no_retry` asserts `assert_called_once()` against a `Timeout` side_effect — confirmed passing. |
| 3 | A failed/timed-out/malformed API call returns None and never raises (run never aborts) | ✓ VERIFIED | Single try/except catches `requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError, ValueError` and returns `None`; `block.get(...)` (not bracket indexing) used at the consumption point in `_build_event_fields`, closing the T-07-03 gap (quick task `260623-ocs`). `test_sync_07_api_failure_does_not_abort_run` drives a real two-record run where the first-processed record's API call raises `Timeout` — both records still produce `CalendarEvent` rows, no exception propagates out of `call_command`. Confirmed passing directly (ran the single named test). |
| 4 | No caught exception's str()/repr() is interpolated into any logged or returned message | ✓ VERIFIED | `_resolve_placement_block`'s except clause has no `as exc` binding. `Command.handle()`'s fallback stderr line interpolates only `record.observation_id!r`. `test_sync_09_log_line_is_fixed_generic_message` embeds a literal leak-marker token in a raised `ImproperCredentialsException` and asserts the marker is absent from captured stderr while `observation_id`/`'fallback'` are present — confirmed passing directly. `test_sync_09_no_credential_or_body_leak_in_logs` covers the same guarantee at the helper level for both exception types. |
| 5 | A placed record resolves its telescope label via the live API and gets a clean title on success | ✓ VERIFIED | `_build_event_fields`'s placed-record branch calls `_resolve_placement_block`, maps the result via `_derive_telescope`, and `_title_for` returns a clean (no-prefix) title when `label_was_fallback` is False. `test_telescope_02_placed_record_resolves_via_api` and `test_sync_03_d03_placed_uses_scheduled_times_and_clean_title` confirm this with a mocked successful API response. |
| 6 | A placed record whose API call fails/times out/returns an unmapped code still gets a `CalendarEvent` with a coarse fallback label, `[UNVERIFIED]` prefix, and a description noting the lookup failed | ✓ VERIFIED | `_build_event_fields` routes `block is None` OR `resolved is None` (Pitfall 4 — same bucket) to `telescope = coarse`, `label_was_fallback = True`; `_title_for` returns `[UNVERIFIED] {telescope} {instrument}`; description appends the fixed generic unverified note. `test_telescope_03_api_failure_falls_back_not_skipped` and `test_telescope_04_fallback_label_visibly_distinguishable` confirm this, including the visible flip back to a verified label on a subsequent successful run. |
| 7 | A banner-stage (not-yet-scheduled) record gets the coarse fallback label with NO API call and keeps `[QUEUED]` (never `[UNVERIFIED]`) | ✓ VERIFIED | `_build_event_fields`'s `record.scheduled_start is None` branch never calls `_resolve_placement_block`. `test_d01_banner_record_no_api_call_no_unverified_prefix` asserts `make_request.assert_not_called()` and the title is `[QUEUED]`-prefixed, never `[UNVERIFIED]` — confirmed passing directly. |
| 8 | `telescope_api_failed` counter increments only for placed records whose API call failed/unmapped, is distinct from `skipped`, and appears in the per-facility summary line | ✓ VERIFIED | Key present in both `counters['LCO']`/`counters['SOAR']` dict literals, the defensive `setdefault` default, incremented only inside the `telescope_api_failed` truthy branch of `handle()`, and appended to the summary f-string. `test_sync_06_fallback_counter_distinct_from_skipped` confirms `telescope_api_failed: 1, skipped: 0` in the printed summary. |
| 9 | A per-record API failure never aborts the run — subsequent records still sync | ✓ VERIFIED | Same evidence as Truth 3 — `test_sync_07_api_failure_does_not_abort_run` confirmed passing directly. |
| 10 | The failure log line is a fixed generic message that never contains the response body or API key | ✓ VERIFIED | Same evidence as Truth 4. |
| 11 | The verified `SITE_TELESCOPE_MAP` and resolution helpers are exercised by passing unit tests (Plan 01 scope) | ✓ VERIFIED | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` run directly: 35/35 tests pass. `ruff check`/`ruff format --check` clean on both changed files. |
| 12 | The fallback decision tree, `[UNVERIFIED]` prefix, and counter are exercised by passing integration tests (Plan 02 scope) | ✓ VERIFIED | Same full-suite run confirms all Plan 02 integration tests pass, including `test_telescope_04_fallback_label_visibly_distinguishable`'s success→fallback→success label-flip assertion. |
| 13 | The paired demo notebook stays in sync with the new behavior (CLAUDE.md demo-notebook-companion convention) | ✓ VERIFIED | `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` contains 12 occurrences of `UNVERIFIED` and 13 of `telescope_api_failed` in committed output cells; only the Django-setup import cell has empty output (expected — no printable result). |

**Score:** 13/13 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | `SITE_TELESCOPE_MAP`, `_resolve_placement_block`, `_aperture_class_from_telescope_code`, 2-arg `_derive_telescope`, `_coarse_telescope_label`, extended `_title_for`, reworked `_build_event_fields`, `telescope_api_failed` counter wiring | ✓ VERIFIED | All symbols present, substantive (not stubs), wired into `Command.handle()`'s per-record loop. Read in full — no debt markers, no placeholder patterns. |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` | Unit + integration tests for all of the above | ✓ VERIFIED | 35 test methods total; all claimed test names exist and pass (`./manage.py test` run directly, not inferred from SUMMARY). |
| `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` | Executed demo cells for API-success, API-failure/fallback, and counter paths | ✓ VERIFIED | Present, contains executed output with `UNVERIFIED`/`telescope_api_failed` evidence. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `sync_lco_observation_calendar.py` | `tom_observations.facilities.ocs.make_request` | `_resolve_placement_block` calls `make_request` with `timeout=_API_TIMEOUT_SECONDS` | ✓ WIRED | `grep -n "timeout=_API_TIMEOUT_SECONDS"` matches inside `_resolve_placement_block`; exactly one `make_request(` call in the function body. |
| `_derive_telescope` | `SITE_TELESCOPE_MAP` | dict lookup keyed on `(site, aperture_class)` | ✓ WIRED | `SITE_TELESCOPE_MAP.get((site, aperture_class))` at `sync_lco_observation_calendar.py:145`. |
| `_build_event_fields` | `_resolve_placement_block` | called only for placed records (`scheduled_start` populated, D-01) | ✓ WIRED | Call is inside the `else` branch of `if record.scheduled_start is None`; confirmed by `test_d01_banner_record_no_api_call_no_unverified_prefix`'s `assert_not_called()`. |
| `Command.handle` | `counters[...]['telescope_api_failed']` | incremented on placed+failed/unmapped condition, surfaced in summary line | ✓ WIRED | Confirmed by direct grep and by `test_sync_06_fallback_counter_distinct_from_skipped` passing. |
| `_title_for` | `[UNVERIFIED]` prefix | `label_was_fallback` branch between terminal-prefix and `[QUEUED]`/clean | ✓ WIRED | `[UNVERIFIED]` appears only inside `_title_for`, in the branch after the `[QUEUED]` check, before the clean return — matches D-09 priority order exactly. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SYNC-07 no-abort (behavior-dependent: cross-record continuation) | `python manage.py test ...test_sync_07_api_failure_does_not_abort_run` | `ok` — 2 CalendarEvents created, no exception | ✓ PASS |
| SYNC-09 no-leak (behavior-dependent: exception content must not reach output) | `python manage.py test ...test_sync_09_log_line_is_fixed_generic_message` | `ok` — leak marker absent, observation_id present | ✓ PASS |
| TELESCOPE-04 visible label flip (behavior-dependent: state transition fallback→verified) | `python manage.py test ...test_telescope_04_fallback_label_visibly_distinguishable` | `ok` | ✓ PASS |
| D-01 banner no-API-call (behavior-dependent: call-count invariant) | `python manage.py test ...test_d01_banner_record_no_api_call_no_unverified_prefix` | `ok` — `make_request.assert_not_called()` holds | ✓ PASS |
| Full telescope-resolution test file | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar -v 1` | `Ran 35 tests ... OK` | ✓ PASS |
| Full Django app suite (regression check) | `python manage.py test solsys_code` | `Ran 130 tests ... OK` | ✓ PASS |
| pytest suite (regression check) | `python -m pytest -q` | `1 passed` | ✓ PASS |
| Lint/format | `ruff check .../sync_lco_observation_calendar.py .../test_sync_lco_observation_calendar.py` + `ruff format --check` | `All checks passed!` / `2 files already formatted` | ✓ PASS |

All behavior-dependent truths (3, 6, 7, 9, 10) were upgraded from presence-only to VERIFIED by directly running the single named test that exercises each invariant — not inferred from SUMMARY/SECURITY/VALIDATION/UAT claims.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TELESCOPE-01 | 07-01 | Verified static site/telescope mapping dict covering all real LCO-network sites | ✓ SATISFIED | 7-site `(site, aperture_class)` dict; `tlv` exclusion is a documented, operator-approved scope correction (ROADMAP.md, `07-01-SUMMARY.md`), not a gap. REQUIREMENTS.md's literal "siteid-enclid-telid" wording predates the locked D-03 decision (collapse by site+aperture-class) — implementation matches the design decision, not the stale requirement prose. |
| TELESCOPE-02 | 07-01, 07-02 | Live API call resolves actual site/enclosure/telescope for placed records, mapped via verified dict | ✓ SATISFIED | `_resolve_placement_block` + `_derive_telescope`, wired into `_build_event_fields`'s placed-record branch. |
| TELESCOPE-03 | 07-02 (+ quick `260623-ocs`) | API failure/timeout/unmapped code falls back to coarse label instead of skipping | ✓ SATISFIED | Decision tree confirmed; T-07-03 malformed-block edge case (missing `'site'`/`'telescope'` keys) closed via `.get()` access, regression-tested. |
| TELESCOPE-04 | 07-02 | Fallback event visibly distinguishable; label flip is visible, not hidden | ✓ SATISFIED | `[UNVERIFIED]` prefix + coarse token + description note; flip-back tested. |
| SYNC-06 | 07-02 | Distinct fallback counter from `skipped`, reported in summary | ✓ SATISFIED | `telescope_api_failed` counter confirmed wired and tested. |
| SYNC-07 | 07-02 | API failure does not abort the run | ✓ SATISFIED | Confirmed by direct test execution. |
| SYNC-08 | 07-01 | Explicit timeout, single attempt, no retry | ✓ SATISFIED | Confirmed by direct test execution (`assert_called_once`). |
| SYNC-09 | 07-01, 07-02 | No credential/response-body leakage in logs | ✓ SATISFIED | Confirmed by direct test execution with embedded leak marker. |

No orphaned requirements — all 8 IDs declared in `07-01-PLAN.md`/`07-02-PLAN.md` frontmatter appear in REQUIREMENTS.md's "Telescope Label Resolution" / "Partial-Failure Handling & Reporting" sections and in the v1.3 traceability table, all marked `Complete`.

### Anti-Patterns Found

None. Scanned both modified production/test files for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER`, placeholder/not-yet-implemented language, and empty-implementation patterns — zero matches.

### Post-Execution Gap Closures (verified, not just claimed)

Two real gaps were found and fixed after initial execution, both independently re-verified here:

1. **T-07-03** (security): `_build_event_fields` originally read the resolved API block via bracket indexing, raising `KeyError` (misrouted to `skipped`) on a malformed/tampered block missing `'site'`/`'telescope'`. Fixed via quick task `260623-ocs` (commits `3fc6554`, `2fa0300`) — confirmed: `grep -n "block\['"` returns zero matches; `test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped` exists and passes.
2. **SITE_TELESCOPE_MAP completeness** (UAT): a real production record (`observation_id=4213127`) resolved to `('coj', '1m0')` via a genuine live API call, but the dict only had `('coj', '2m0')`, causing an incorrect fallback. Fixed via quick task `260623-su3` (commits `cd3b17e`, `5583400`, `2473aa8`) adding `('coj','1m0')`, `('coj','0m4')`, `('ogg','0m4')` — confirmed present in the current `SITE_TELESCOPE_MAP` and covered by `test_telescope_01_coj_ogg_full_aperture_class_coverage`.

Both commits exist in git history (`git cat-file -e` confirmed for all 11 commits cited across both SUMMARYs and both quick-task SUMMARYs).

### Human Verification Required

None. UAT (`07-UAT.md`) already performed live, real-API verification against production data (observation_id=4213127, including a genuine credential-failure run) for all 6 of its tests, with 6/6 passed after the two gap closures above. No further human verification items identified during this codebase-level check.

### Gaps Summary

No gaps. All 13 derived must-haves (roadmap success criteria + plan-level truths) are verified directly against the current codebase: the verified `SITE_TELESCOPE_MAP` exists with correct format and coverage (post both gap-closure quick tasks), the live API resolution path is single-attempt/timeout-bounded/never-raising, the fallback decision tree correctly routes placed-record failures and banner-stage records, the `[UNVERIFIED]` prefix and `telescope_api_failed` counter are wired end-to-end, no credential/response-body content reaches any log or returned string, and a per-record failure provably does not abort the run (the run-continuation invariant was independently re-run, not inferred from SUMMARY narrative). Full Django suite (130 tests) and pytest suite are green; ruff is clean. The paired demo notebook is in sync per CLAUDE.md's convention.

The one wording mismatch noted (REQUIREMENTS.md's TELESCOPE-01 description literally says "siteid-enclid-telid" while the shipped dict is keyed on `(site, aperture_class)`) is not a functional gap — it reflects a deliberate, documented design decision (D-03/D-04 in `07-CONTEXT.md`, reiterated in `07-RESEARCH.md`/`07-PATTERNS.md`) that was made during planning and is correctly reflected in ROADMAP.md's success criteria and ADR-level decision trail. REQUIREMENTS.md's requirement-line prose is stale relative to that decision but the requirement itself (a verified mapping dict covering real sites, replacing the 2-site `[ASSUMED]` dict) is satisfied.

---

*Verified: 2026-06-24T04:31:27Z*
*Verifier: Claude (gsd-verifier)*
