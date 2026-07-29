---
phase: quick-260623-ocs
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
autonomous: true
requirements: [T-07-03]
must_haves:
  truths:
    - "A COMPLETED/PENDING API block missing 'site' or 'telescope' produces a coarse-fallback CalendarEvent, not a skipped record."
    - "A block missing a key increments telescope_api_failed (not skipped) and gets the [UNVERIFIED] prefix."
    - "_derive_telescope never raises when site or telescope_code is None — it returns None (routes to the same Pitfall-4 fallback bucket as an unmapped pair)."
  artifacts:
    - path: "solsys_code/management/commands/sync_lco_observation_calendar.py"
      provides: "block.get('site')/block.get('telescope') field access in _build_event_fields plus None-safe aperture-class parsing"
      contains: "block.get('site')"
    - path: "solsys_code/tests/test_sync_lco_observation_calendar.py"
      provides: "Regression test for a COMPLETED block missing 'site'/'telescope'"
      contains: "telescope_api_failed: 1"
  key_links:
    - from: "solsys_code/management/commands/sync_lco_observation_calendar.py:_build_event_fields"
      to: "solsys_code/management/commands/sync_lco_observation_calendar.py:_derive_telescope"
      via: "passes block.get('site')/block.get('telescope') (possibly None) — _derive_telescope returns None on either being None/unparseable"
      pattern: "_derive_telescope\\(block.get"
---

<objective>
Fix security/spec gap T-07-03 in `sync_lco_observation_calendar.py`: a syntactically
valid JSON block with `state: 'COMPLETED'` (or `'PENDING'`) but missing the `'site'`
or `'telescope'` key currently raises `KeyError` at the point of consumption
(`_build_event_fields` line 400, bracket indexing), which is caught one layer up by the
generic `except (KeyError, ValueError)` and misrouted into the `skipped` counter — NOT
the declared "routes to fallback (returns None)" mitigation, and NOT the dedicated
`telescope_api_failed` (Pitfall-4) bucket the TELESCOPE-03/04/SYNC-06 design relies on.

Purpose: Make the declared T-07-03 mitigation true at the point of consumption — a
malformed/tampered block missing either field routes through the existing coarse-fallback
bucket (same counter, same `[UNVERIFIED]` prefix) exactly like an unmapped
`(site, aperture_class)` pair already does, instead of being silently dropped as skipped.

Output: One-line field-access change (plus a None-guard so the change cannot raise
`TypeError` on a missing `'telescope'`), and a regression test mirroring the existing
SYNC-06/TELESCOPE-03 fallback tests.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/07-live-telescope-label-resolution-with-fallback-failure-report/SECURITY.md
@solsys_code/management/commands/sync_lco_observation_calendar.py
@solsys_code/tests/test_sync_lco_observation_calendar.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Route a block missing 'site'/'telescope' to coarse fallback (None-safe)</name>
  <files>solsys_code/management/commands/sync_lco_observation_calendar.py</files>
  <behavior>
    - `_aperture_class_from_telescope_code(None)` returns None (does not raise TypeError on `len(None)`).
    - `_derive_telescope(None, '1m0a')` returns None (site None -> SITE_TELESCOPE_MAP miss -> coarse fallback).
    - `_derive_telescope('lsc', None)` returns None (telescope_code None -> aperture-class parse returns None -> coarse fallback).
    - In `_build_event_fields`, a placed record whose resolved block is missing 'site' or 'telescope' sets label_was_fallback=True and telescope=coarse (Pitfall-4 bucket), not a raised KeyError.
  </behavior>
  <action>
    Per T-07-03 (SECURITY.md "Open Threat Detail" / "Required fix"), make two None-safe edits:

    (1) In `_build_event_fields`, change the line that reads the resolved block's fields
    via bracket indexing to use `.get(...)` instead, consistent with
    `_resolve_placement_block`'s own `block.get('state')` convention. The call becomes
    `_derive_telescope(block.get('site'), block.get('telescope'))` (still guarded by the
    existing `if block is not None else None` ternary). Do not change any other line in
    `_build_event_fields` — the existing `if resolved is None:` branch already sets the
    coarse fallback label, telescope_api_failed counter, and `[UNVERIFIED]` prefix
    (Pitfall 4, same bucket as an unmapped pair).

    (2) Guard `_aperture_class_from_telescope_code` against a None argument so the new
    `.get('telescope')` (which can now legitimately be None for a malformed block)
    cannot raise TypeError on `len(None)`. Add an early guard returning None when
    telescope_code is falsy/None, BEFORE the existing length/slice check. This keeps
    `_derive_telescope`'s documented "Never raises." contract true now that a None
    telescope_code can reach it. Update `_derive_telescope`'s and/or
    `_aperture_class_from_telescope_code`'s docstring Args/Returns to note the parameter
    may be None and that None routes to the coarse fallback. (`site` being None needs no
    code guard — `SITE_TELESCOPE_MAP.get((None, aperture_class))` simply returns None.)

    Rationale to encode in a brief inline comment at the changed `.get(...)` line: a
    COMPLETED/PENDING block from a malformed/tampered API response may omit 'site'/
    'telescope' (only 'state' is validated upstream); .get keeps the missing-key case in
    the same coarse-fallback bucket as an unmapped pair instead of raising into skipped.
  </action>
  <verify>
    <automated>python -c "from solsys_code.management.commands.sync_lco_observation_calendar import _derive_telescope, _aperture_class_from_telescope_code; assert _aperture_class_from_telescope_code(None) is None; assert _derive_telescope(None, '1m0a') is None; assert _derive_telescope('lsc', None) is None; print('ok')" 2>/dev/null || echo 'requires DJANGO_SETTINGS_MODULE — covered by Task 2 management-test run'</automated>
  </verify>
  <done>
    `_build_event_fields` reads the resolved block via `block.get('site')`/
    `block.get('telescope')`; `_aperture_class_from_telescope_code` returns None for a
    None argument instead of raising; `_derive_telescope` returns None (not raises) for
    either a None site or a None telescope_code.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Regression test — block missing 'site'/'telescope' falls back, not skipped</name>
  <files>solsys_code/tests/test_sync_lco_observation_calendar.py</files>
  <behavior>
    - A placed (scheduled_start set) COMPLETED block missing 'site' (or 'telescope')
      produces exactly one CalendarEvent (record is NOT skipped).
    - That event's title starts with '[UNVERIFIED]' and telescope is the coarse label
      (e.g. '1m0' from instrument_type '1M0-SCICAM-SINISTRO').
    - Summary line shows 'telescope_api_failed: 1' and 'skipped: 0'.
  </behavior>
  <action>
    Add a regression test mirroring the existing `test_sync_06_fallback_counter_distinct_from_skipped`
    / `test_telescope_03_api_failure_falls_back_not_skipped` pattern (same file). Name it
    for the threat, e.g. `test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped`.

    The existing `_observations_block_response()` helper always populates all four keys
    (site/enclosure/telescope/state), so it cannot express a missing key. Build the mock
    response inline in the test instead: a `MagicMock()` whose `.json.return_value` is a
    one-element list containing a block dict with `'state': 'COMPLETED'` and `'telescope':
    '1m0a'` but NO `'site'` key (the malformed/tampered shape T-07-03 describes). Patch
    `solsys_code.management.commands.sync_lco_observation_calendar.make_request` with
    `return_value=` that mock (not side_effect — the call succeeds; the body is just
    missing a field).

    Create the record via `self._create_record(...)` with a `scheduled_start`/
    `scheduled_end` set (placed record, so the live-API path runs) and an
    `instrument_type='1M0-SCICAM-SINISTRO'` so the coarse fallback token is '1m0'. Use a
    distinct proposal code and observation_id from the other fallback tests.

    Assert: `CalendarEvent.objects.count() == 1`; the single event's `.telescope == '1m0'`;
    `event.title.startswith('[UNVERIFIED]')`; the captured stdout summary contains
    `'telescope_api_failed: 1'` and `'skipped: 0'`. (Optionally add a second assertion
    or a sibling test that omits 'telescope' instead of 'site' — the same fallback must
    hold — but a single missing-'site' case is the minimum regression for T-07-03.)
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_sync_lco_observation_calendar 2>&1 | tail -20</automated>
  </verify>
  <done>
    New test exists, exercises a COMPLETED block missing 'site' (and/or 'telescope'),
    asserts a single coarse-fallback CalendarEvent with '[UNVERIFIED]' title +
    'telescope_api_failed: 1' + 'skipped: 0', and the full
    `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` suite passes.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| sync command -> LCO Observation Portal API | Untrusted remote JSON block (may be malformed/tampered: valid JSON, `state` present, `site`/`telescope` absent) crosses into `_build_event_fields` via `_resolve_placement_block`. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-07-03 | Denial of Service / Tampering | `_build_event_fields` block field access (line 400) | mitigate | Read block fields via `block.get('site')`/`block.get('telescope')` (not `[]`) so a missing key yields None and routes to the existing coarse-fallback (Pitfall-4) bucket instead of raising KeyError into the generic `skipped` counter. None-guard `_aperture_class_from_telescope_code` so a None telescope_code cannot raise TypeError. Regression test (Task 2) asserts fallback, not skip. |
| T-07-01 | Information Disclosure | fallback log/description path | accept (unchanged) | This fix never references, stringifies, or logs the caught/missing value; the fixed-string fallback note and generic stderr line (verified CLOSED in SECURITY.md) are untouched, so no credential/body leakage is reintroduced. |
| T-07-SC | Tampering (supply chain) | npm/pip/cargo installs | accept | No new packages installed — only edits to two existing files. Per SECURITY.md Accepted Risks Log, all imports already present. |
</threat_model>

<verification>
- `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` — full suite passes (existing 19+ tests plus the new regression test).
- `ruff check solsys_code/management/commands/sync_lco_observation_calendar.py solsys_code/tests/test_sync_lco_observation_calendar.py` — clean.
- `ruff format --check solsys_code/management/commands/sync_lco_observation_calendar.py solsys_code/tests/test_sync_lco_observation_calendar.py` — clean.
</verification>

<success_criteria>
- `_build_event_fields` uses `block.get('site')`/`block.get('telescope')`; no bracket indexing of the resolved block remains.
- `_derive_telescope`/`_aperture_class_from_telescope_code` are None-safe (return None, never raise) for a None site or telescope_code.
- A COMPLETED/PENDING block missing 'site' or 'telescope' produces a coarse-fallback CalendarEvent counted under `telescope_api_failed` (not `skipped`), with the `[UNVERIFIED]` title prefix.
- New regression test passes; full `solsys_code.tests.test_sync_lco_observation_calendar` suite passes; ruff check/format clean on both changed files.
- Demo notebook NOT regenerated: this is a defensive edge-case hardening of an already-demonstrated function (success/fallback/counter paths are already shown in `sync_lco_observation_calendar_demo.ipynb`); the user-visible documented behavior paths are unchanged, so CLAUDE.md's demo-notebook-companion convention does not require a notebook update here.
</success_criteria>

<output>
Create `.planning/quick/260623-ocs-fix-t-07-03-security-spec-gap-in-sync-lc/260623-ocs-SUMMARY.md` when done
</output>
