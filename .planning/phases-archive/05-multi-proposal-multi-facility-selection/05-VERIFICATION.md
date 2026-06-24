---
phase: 05-multi-proposal-multi-facility-selection
verified: 2026-06-19T18:30:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
---

# Phase 5: Multi-Proposal & Multi-Facility Selection Verification Report

**Phase Goal:** Generalize the existing `sync_lco_observation_calendar` management command (Phase 4: single-proposal, LCO-only) into a multi-proposal, multi-facility (LCO + SOAR) sync. One invocation must cover any combination of proposals (or the whole LCO-family network via `ALL`) across both `ObservationRecord(facility='LCO')` and `ObservationRecord(facility='SOAR')` rows, with each record processed through the facility instance matching its own `facility` value — never a single shared instance.
**Verified:** 2026-06-19T18:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `--proposal A,B,C` syncs records matching exactly A/B/C with no substring match on 'AB' (SELECT-02) | VERIFIED | `_parse_proposal_arg` uses comma-split/strip/dedupe (no regex/substring); queryset uses `parameters__proposal__in=codes` (exact-match `IN`, not `icontains`). Live test `test_select_02_comma_list_matches_any_no_substring_leakage` creates A/B/C + decoy 'AB' record, asserts 3 events created and the 'AB' decoy produces none. Ran via `python manage.py test ... test_select_02...` — PASS. |
| 2 | `--proposal ALL` (any casing) syncs every LCO + SOAR record regardless of proposal (SELECT-03) | VERIFIED | `_parse_proposal_arg('all'/'All'/'ALL')` returns `None` (verified live via `manage.py shell`: `_parse_proposal_arg('all')`, `_parse_proposal_arg('All')`, `_parse_proposal_arg('ALL')` all → `None`); `handle()` omits the `parameters__proposal__in` clause entirely when `codes is None`. Test `test_select_03_all_token_case_insensitive_syncs_everything` (lowercase 'all', mixed proposals, one SOAR record) — PASS. |
| 3 | A single run produces correct `CalendarEvent`s for both `facility='LCO'` and `facility='SOAR'` records together, without two invocations (SELECT-04) | VERIFIED | `records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])` is the single base queryset (no facility-specific branching at the query level). Test `test_select_04_single_run_covers_both_facilities` creates one LCO + one SOAR record sharing a proposal, single `call_command` invocation, asserts both produced events. Independently confirmed live (not mocked) via a manual `manage.py shell` script that created real LCO+SOAR `ObservationRecord`s and ran the actual command end-to-end: stdout was `LCO: created: 1, ... | SOAR: created: 1, ...` and 2 `CalendarEvent`s were created from one invocation. |
| 4 | A SOAR record is processed through a `SOARFacility` instance, never a reused `LCOFacility` instance, verified by a spy asserting which class method was called (SELECT-05) | VERIFIED | `handle()` builds `facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}` eagerly (both keys, unconditionally) and dispatches via `facilities.get(record.facility)` per record — never a single shared instance. Confirmed live that `SOARFacility` and `LCOFacility` are genuinely distinct classes (`SOARFacility.__bases__ == (LCOFacility,)`, `SOARFacility is not LCOFacility`) but inherit the *same* `get_observation_url` implementation (`SOARFacility.get_observation_url is LCOFacility.get_observation_url` → `True`) — confirming the plan's premise that a black-box URL-equality check cannot discriminate dispatch, so the discriminating spy test design is necessary, not gratuitous. Test `test_select_05_soar_record_uses_soar_facility_instance` patches `SOARFacility.get_observation_url` and `LCOFacility.get_observation_url` separately (via `patch.object(..., autospec=True, side_effect=real_method)`), asserts the SOAR spy was called once and the LCO spy was never called — PASS. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | `_parse_proposal_arg` helper, eager LCO/SOAR dispatch dict, conditional `facility__in`/`proposal__in` queryset filter, per-facility counters, per-facility summary line | VERIFIED | `_parse_proposal_arg` defined at line 162; `facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}` at line 224; `records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])` at line 233 with conditional `.filter(parameters__proposal__in=codes)` only when `codes is not None`; `counters` dict per-facility at lines 228-231; per-facility summary line built at lines 275-279 (`' \| '.join(...)`). 281 lines total, no stub patterns. |
| `src/fomo/settings.py` | `FACILITIES['SOAR']` entry so `SOARSettings('SOAR')` resolves a real `api_key` key (D-04/D-05) | VERIFIED | `'SOAR': {'portal_url': 'https://observe.lco.global', 'api_key': ''}` present (lines 223-226), directly after `'LCO'` and before `'GEM'`, exactly as planned. The committed `'LCO'` entry's literal `api_key` value is unchanged by this phase's commit `adc5a61` (confirmed via `git show adc5a61 -- src/fomo/settings.py`: diff shows only a new `'SOAR'` block added, `'LCO'` untouched). Note: a pre-existing *uncommitted, local-only* dev override of `LCO.api_key` exists in the working tree (unrelated `ALLOWED_HOSTS` + real-looking API key for local dev), confirmed via `git diff` and `git stash` to be outside this phase's diff and present before/after this phase's commits. |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` | `_create_record(facility='LCO')` signature extension + SELECT-02/03/04/05 tests | VERIFIED | `_create_record` signature includes `facility: str = 'LCO'` (line 58), passed to `ObservationRecord.objects.create(facility=facility, ...)`. All four named test methods present and pass: `test_select_02_comma_list_matches_any_no_substring_leakage`, `test_select_03_all_token_case_insensitive_syncs_everything`, `test_select_04_single_run_covers_both_facilities`, `test_select_05_soar_record_uses_soar_facility_instance`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `sync_lco_observation_calendar.py` | `tom_observations.facilities.soar.SOARFacility` | import + dispatch dict keyed by `facility` | WIRED | `from tom_observations.facilities.soar import SOARFacility` (line 8); used at `facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}` (line 224) and dispatched via `facilities.get(record.facility)` (line 239) inside the per-record loop, with the resolved `facility` instance passed into `_build_event_fields(record, facility)`. |
| `sync_lco_observation_calendar.py` | `ObservationRecord` queryset | `facility__in` plus conditional `parameters__proposal__in` | WIRED | `records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])` then conditionally `.filter(parameters__proposal__in=codes)` when `codes is not None` (lines 233-236). Confirmed both code paths exercised by tests (`test_select_02`/`test_select_03` for the conditional branch; `test_select_04` for the unconditional `facility__in` coverage). |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `_parse_proposal_arg` dedupe/case behavior (D-01/D-02/D-03) | `manage.py shell -c "_parse_proposal_arg('A,A,B')"` etc. | `['A','B']`, `['A','B']`, `None`/`None`/`None` (all 3 ALL casings), `['a','B']` (case preserved) | PASS |
| Full end-to-end live run across LCO+SOAR (SELECT-03/04/08) | Manual `manage.py shell` script creating real `ObservationRecord`s and invoking `call_command('sync_lco_observation_calendar', ...)` against the real (non-mocked) test DB | stdout: `Done. proposal: VERIFYCODE, LCO: created: 1, updated: 0, unchanged: 0, skipped: 0 \| SOAR: created: 1, updated: 0, unchanged: 0, skipped: 0`; 2 `CalendarEvent`s created | PASS |
| `SOARFacility`/`LCOFacility` class-distinctness check underlying SELECT-05's spy design | `manage.py shell -c "SOARFacility.get_observation_url is LCOFacility.get_observation_url"` | `True` (same inherited method) while `LCOFacility is not SOARFacility` (`SOARFacility.__bases__ == (LCOFacility,)`) | PASS — confirms the discriminating-spy test design is necessary (a url-equality check could not distinguish dispatch) |
| Full Django app test suite | `python manage.py test solsys_code` | `Ran 114 tests ... OK` | PASS |
| Targeted command test module | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar -v 2` | `Ran 19 tests ... OK` (includes all 4 SELECT tests) | PASS |
| Separate pytest suite | `python -m pytest -q` | `1 passed` | PASS |
| Lint | `ruff check <3 changed files>` | `All checks passed!` | PASS |
| Format | `ruff format --check <3 changed files>` | `settings.py` flagged; isolated via `git stash`/`stash pop` to confirm pre-existing on the *committed* state (unrelated to phase 05's diff, tracked in Phase 4's `deferred-items.md`) | PASS (pre-existing, documented, non-blocking) |
| Credential leakage | `grep -v '^#' sync_lco_observation_calendar.py \| grep -c api_key` | `0` | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SELECT-02 | 05-01-PLAN.md | `--proposal` accepts comma-separated list, syncs matching any code, no substring leakage | SATISFIED | `_parse_proposal_arg` + `parameters__proposal__in` exact-match `IN` lookup; `test_select_02_comma_list_matches_any_no_substring_leakage` passes |
| SELECT-03 | 05-01-PLAN.md | `--proposal ALL` syncs every LCO-family record regardless of proposal | SATISFIED | `_parse_proposal_arg('all'/'All'/'ALL') -> None`; `test_select_03_all_token_case_insensitive_syncs_everything` passes |
| SELECT-04 | 05-01-PLAN.md | Sync covers both `facility='LCO'` and `facility='SOAR'` records in a single run | SATISFIED | `facility__in=['LCO', 'SOAR']` single base queryset; `test_select_04_single_run_covers_both_facilities` passes; confirmed live end-to-end |
| SELECT-05 | 05-01-PLAN.md | Each record processed via the facility instance matching its own `facility` value, never a single shared instance | SATISFIED | Eager `facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dict + `facilities.get(record.facility)` dispatch; `test_select_05_soar_record_uses_soar_facility_instance` discriminating spy passes |

**Orphaned requirements check:** REQUIREMENTS.md maps exactly SELECT-02/03/04/05 to Phase 5 (lines 68-71). All four appear in the PLAN frontmatter `requirements:` field and all four are satisfied. No orphaned requirements for this phase.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers found in any of the 3 phase-modified files | — | None — clean |
| `sync_lco_observation_calendar.py` | 239-249 | Defensive `if facility is None:` branch is unreachable given `facility__in=['LCO','SOAR']` always seeding both dict keys (flagged as code-review Info IN-01) | Info | Not a blocker — intentional defense-in-depth per D-07, harmless dead-code-by-construction, not a goal-blocking gap |
| `src/fomo/settings.py` | 225 | `SOAR.api_key` hardcoded to `''` rather than sourced from the same value as `LCO.api_key` (code-review Warning WR-03) | Warning | Out of phase-05 scope per the plan's explicit "narrower reading" decision (D-04/D-05) — SOAR credential plumbing is not one of SELECT-02/03/04/05; tracked in 05-REVIEW.md, not a regression introduced silently |
| `sync_lco_observation_calendar.py` | 251-256 | `except (KeyError, ValueError)` does not catch `TypeError` if `record.parameters` is ever `None` (code-review Warning WR-01) | Warning | Pre-existing robustness gap inherited from Phase 4's `_build_event_fields`, not introduced or worsened by this phase's diff; does not affect SELECT-02/03/04/05 achievement |

None of the code-review warnings (WR-01/WR-02/WR-03) block any of the four phase-05 success criteria; they are legitimate follow-up items already documented in `05-REVIEW.md`, separate from this phase's required-truths scope.

### Human Verification Required

None. This phase is backend/CLI logic only (Django management command + settings + tests), with no UI surface, no visual rendering, no real-time behavior, and no external-service dependency that can't be verified by automated tests and direct invocation. All four success criteria were independently confirmed via (a) the phase's own automated test suite, (b) a live, non-mocked, end-to-end manual invocation against the real test database, and (c) direct interrogation of the `LCOFacility`/`SOARFacility` class hierarchy.

### Gaps Summary

No gaps. All four must-have truths (SELECT-02/03/04/05) are verified at all three levels (exists, substantive, wired) plus a live data-flow trace (Level 4: a real, non-mocked command invocation produced correct per-facility `CalendarEvent`s and a correct per-facility summary line). The full Django test suite (114 tests) and the separate pytest suite both pass. Lint is clean; the one `ruff format --check` flag on `settings.py` is confirmed pre-existing on the committed state (unrelated to this phase's diff, already tracked in Phase 4's deferred-items.md). No credential leakage. No debt markers. Code review surfaced 3 non-blocking warnings, all explicitly out of this phase's requirement scope and already documented.

---

*Verified: 2026-06-19T18:30:00Z*
*Verifier: Claude (gsd-verifier)*
