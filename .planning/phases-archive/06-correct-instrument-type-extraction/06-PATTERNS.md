# Phase 6: Correct Instrument-Type Extraction - Pattern Map

**Mapped:** 2026-06-20
**Files analyzed:** 2 (both modified, not created)
**Analogs found:** 2 / 2 (in-file analogs — both target files already contain the closest analog to themselves)

## File Classification

| Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------|------|-----------|-----------------|----------------|
| `solsys_code/management/commands/sync_lco_observation_calendar.py` (new helper(s), e.g. `_extract_instrument`) | utility (private helper function in a management command module) | transform (pure dict-parsing, no I/O) | same file: `_derive_telescope` (lines 56-71), `_failure_prefix` (lines 40-53), `_title_for` (lines 74-92), `_time_window` (lines 95-123) | exact — these are the file's own established small-helper convention |
| `solsys_code/management/commands/sync_lco_observation_calendar.py` (`handle()` per-record loop edit for D-06 dedicated counter) | controller (command orchestration / per-record loop) | event-driven (per-record catch-log-continue) | same file: existing `except (KeyError, ValueError)` skip block (lines 254-259) and `counters.setdefault(...)` D-07 defensive pattern (lines 244-251) | exact |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` (`_parameters()` extension) | test (fixture helper) | transform (dict construction) | same file: `_parameters()` itself (lines 16-43) — extend in place | exact |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` (new test cases) | test | request-response (via `call_command` + DB assertions) | same file: `test_sync_05_telescope_instrument_proposal_populated` (lines 150-168), `test_skip_path_missing_site_logged_and_skipped` (lines 317-332), `test_select_04_single_run_covers_both_facilities` (lines 411-434) | exact |

No external/cross-file analogs were needed — both files already contain the strongest possible analog for their own extension (the file's established internal conventions). No `## No Analog Found` section is required.

## Pattern Assignments

### New extraction helper(s) in `sync_lco_observation_calendar.py`

**Analog:** the file's own existing small-helper style — `_derive_telescope`, `_failure_prefix`, `_title_for`, `_time_window` (lines 40-123)

**Docstring style** (copy exactly, Google-style with Args/Returns/Raises) — from `_derive_telescope` (lines 56-71):
```python
def _derive_telescope(site_code: str) -> str:
    """Map an LCO/SOAR site code to a telescope label.

    Args:
        site_code: LCO/SOAR 3-letter site code (e.g. 'coj').

    Returns:
        str: telescope label (e.g. 'FTS').

    Raises:
        KeyError: if site_code is not in SITE_TELESCOPE_MAP.
    """
    try:
        return SITE_TELESCOPE_MAP[site_code]
    except KeyError:
        raise KeyError(f'Unmapped LCO site code {site_code!r}; add it to SITE_TELESCOPE_MAP') from None
```
Apply the same shape to the new helper: short body, explicit `Raises:` section naming exactly what it raises (or, per the D-06 pitfall below, explicitly documenting the sentinel-return contract instead of raising).

**Module-level whitelist/lookup-table convention** — from `_FAILURE_PREFIX_BY_STATUS` (lines 32-37) and its consuming helper `_failure_prefix` (lines 40-53):
```python
_FAILURE_PREFIX_BY_STATUS = {
    'WINDOW_EXPIRED': '[EXPIRED]',
    'CANCELED': '[CANCELLED]',
    'FAILURE_LIMIT_REACHED': '[FAILED]',
    'NOT_ATTEMPTED': '[FAILED]',
}


def _failure_prefix(status: str, facility: LCOFacility) -> str | None:
    """Return the terminal-failure title prefix for a status, or None if not a failure state."""
    if status not in set(facility.get_failed_observing_states()):
        return None
    return _FAILURE_PREFIX_BY_STATUS.get(status, '[FAILED]')
```
This is the direct analog for D-01's `configuration_type` whitelist (research's "Don't Hand-Roll" recommendation): define a module-level set, e.g. `_SCIENCE_CONFIGURATION_TYPES = {'EXPOSE', 'REPEAT_EXPOSE', 'SPECTRUM', 'REPEAT_SPECTRUM', 'STANDARD'}`, right alongside `_FAILURE_PREFIX_BY_STATUS` and `SITE_TELESCOPE_MAP` (both near the top of the file, lines 18-37), then check membership via `in` inside the new helper — same pattern as `_failure_prefix`'s `status not in set(...)` check, returning `None`/a sentinel on no-match rather than raising, exactly like `_failure_prefix` returns `None` instead of raising when status isn't a failure state.

**Safe-lookup convention for scanning indexed keys** (`c_1..c_5`) — no existing helper in this file loops over a numeric range of keys, but `_time_window`'s defensive `.get`-free explicit-checks style (lines 95-123) and `_derive_telescope`'s try/except-around-dict-access (lines 68-71) both show the file's preference for **explicit, narrow exception handling** rather than blanket `try/except Exception`. For the new helper's `c_N_*` scan, use `.get(...)` (per RESEARCH.md Pitfall 3 — keys may be **entirely absent**, not just empty) in a loop `for n in range(1, 6):`, never direct `[...]` indexing on a `c_N_*` key.

**Replacing the line itself** — exact before-state to change, `_build_event_fields` (lines 142-143):
```python
telescope = _derive_telescope(record.parameters['site'])
instrument = record.parameters['instrument_type']
```
Only line 143 changes; line 142 (`_derive_telescope` call) is untouched (Phase 7 scope). Replace line 143 with a call to the new helper, e.g. `instrument = _extract_instrument(record.parameters)`.

**D-06 sentinel-vs-exception contract (critical, from RESEARCH.md Pitfall 1):** the new helper must NOT let extraction failure surface as a bare `KeyError`/`ValueError`, because `_build_event_fields`'s caller already has a blanket `except (KeyError, ValueError)` (lines 254-259, see Shared Patterns below) that would silently route it into the **existing** `'skipped'` counter, not the new dedicated D-06 counter. Two compliant options, both consistent with this file's existing style:
1. Return `None` from the new helper on total failure, and have `_build_event_fields` raise a **distinct** exception type (or have `handle()` check the return before calling `_build_event_fields`'s downstream consumers) — preferred, matches `_failure_prefix`'s existing "return `None` to signal non-match" convention.
2. Define and raise a small custom exception (e.g. `class InstrumentExtractionError(Exception): pass`) that `handle()` catches in a **second**, separate `except` clause before the generic one.

Given this file has zero existing custom exception classes (confirmed by reading the whole file), option 1 (sentinel `None` + explicit check) is the better stylistic fit — but either is explicitly Claude's discretion per CONTEXT.md.

---

### `_build_event_fields` docstring update

**Analog:** `_build_event_fields`'s own existing `Raises:` docstring section (lines 137-140):
```python
Raises:
    KeyError: if a required parameters key (site/instrument_type/proposal/
        start/end) is missing.
    ValueError: if parameters['start']/['end'] cannot be parsed as datetimes.
```
Update this section to reflect the new instrument-extraction failure mode (whatever sentinel/exception contract is chosen) — keep the same terse one-line-per-exception-type style.

---

### `handle()` D-06 dedicated-counter wiring

**Analog:** the existing `counters` dict-of-dicts pattern, three locations that must all change together (RESEARCH.md "Counter-naming clarification"):

**1. Eager dict-literal init** (lines 231-234):
```python
counters = {
    'LCO': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0},
    'SOAR': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0},
}
```
Add the new key (e.g. `'extraction_failed': 0`) to both dict literals here.

**2. Defensive `setdefault` default** (line 250):
```python
counters.setdefault(record.facility, {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0})
```
Add the same new key here too — RESEARCH.md flags this as the most likely place to forget it.

**3. Per-record catch-log-continue block** (lines 254-259) — the analog for the **new** D-06 skip path:
```python
try:
    fields = _build_event_fields(record, facility)
except (KeyError, ValueError) as exc:
    self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
    counters[record.facility]['skipped'] += 1
    continue
```
If using the sentinel-exception approach (option 2 above), add a **second** `except` clause here (or branch on the helper's return) that increments `counters[record.facility]['extraction_failed']` instead of `'skipped'`, using the same `self.stderr.write(f'Skipping observation_id={record.observation_id!r}: ...')` log-message phrasing for consistency (D-06's "Claude's Discretion" wording note).

**4. Summary line join** (lines 278-283):
```python
summary = ' | '.join(
    f'{facility_name}: created: {counts["created"]}, updated: {counts["updated"]}, '
    f'unchanged: {counts["unchanged"]}, skipped: {counts["skipped"]}'
    for facility_name, counts in counters.items()
)
```
Add `, extraction_failed: {counts["extraction_failed"]}` (or chosen name) to this f-string, same comma-joined phrasing as the other four counters.

---

### Test fixture extension: `_parameters()` in `test_sync_lco_observation_calendar.py`

**Analog:** the function's own current signature (lines 16-43) — extend additively, do not replace:
```python
def _parameters(
    proposal: str = 'TESTCODE123',
    start: str = '2026-07-01T00:00:00',
    end: str = '2026-07-02T00:00:00',
    instrument_type: str = '2M0-SCICAM-MUSCAT',
    site: str | None = 'coj',
) -> dict:
    """Build a parameters dict matching the real ObservationRecord.parameters shape.
    ...
    """
    params = {
        'proposal': proposal,
        'start': start,
        'end': end,
        'instrument_type': instrument_type,
    }
    if site is not None:
        params['site'] = site
    return params
```
RESEARCH.md's recommended extension (option 1, the better fit per this file's explicit-named-params convention): add a sixth parameter `extra_params: dict | None = None`, merged at the end via `params.update(extra_params or {})`, right before `return params`. This preserves every one of the 19 existing call sites unmodified (Pitfall 4) and lets new tests pass arbitrary `c_N_*`/MUSCAT-channel keys without touching `_parameters`'s five existing named params or `_create_record`'s `**parameter_overrides` passthrough (lines 52-71 — unchanged, already forwards any kwarg into `_parameters`).

---

### New test cases

**Analog 1 — happy-path single-config assertion style:** `test_sync_05_telescope_instrument_proposal_populated` (lines 150-168):
```python
def test_sync_05_telescope_instrument_proposal_populated(self):
    """SYNC-05: telescope/instrument/proposal populated from the record."""
    self._create_record(
        '666666',
        proposal='MATCHCODE',
        site='ogg',
        instrument_type='2M0-SCICAM-MUSCAT',
    )
    call_command(
        'sync_lco_observation_calendar',
        '--proposal',
        'MATCHCODE',
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    event = CalendarEvent.objects.get()
    self.assertEqual(event.telescope, 'FTN')
    self.assertEqual(event.instrument, '2M0-SCICAM-MUSCAT')
    self.assertEqual(event.proposal, 'MATCHCODE')
```
Use this exact shape for the new EXTRACT-02 SOAR-multi-config and MUSCAT-per-channel tests: create record(s) with `extra_params={...}`, call the command, then assert `event.instrument` equals the expected science-config value (never the calibration value).

**Analog 2 — coexistence pattern (good record + record exercising the new shape, never new-shape-in-isolation):** `test_select_04_single_run_covers_both_facilities` (lines 411-434) and `test_skip_path_missing_site_logged_and_skipped` (lines 317-332):
```python
def test_skip_path_missing_site_logged_and_skipped(self):
    """A matching record with no parameters['site'] is logged to stderr and skipped; others still sync."""
    self._create_record('990001', proposal='MATCHCODE', site=None)
    self._create_record('990002', proposal='MATCHCODE', site='coj')

    stderr_buf = io.StringIO()
    call_command(
        'sync_lco_observation_calendar',
        '--proposal',
        'MATCHCODE',
        stdout=io.StringIO(),
        stderr=stderr_buf,
    )
    self.assertEqual(CalendarEvent.objects.count(), 1)
    err = stderr_buf.getvalue()
    self.assertIn('990001', err)
```
Use this exact shape for the D-06 fully-malformed-record test: one record with `extra_params` containing no `configuration_type` and no exposure signal anywhere (e.g. via `site=None`-style explicit override pattern, but for instrument fields), plus one normal baseline record; assert `CalendarEvent.objects.count() == 1`, assert the failing `observation_id` appears in `stderr_buf.getvalue()`, and additionally assert the new dedicated counter's summary substring (e.g. `self.assertIn('extraction_failed: 1', stdout_buf.getvalue())`) following the counter-assertion style of `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` (lines 291-292: `self.assertIn('updated: 1', stdout2.getvalue())`).

**Analog 3 — SOAR-specific fixture convention (`site='sor'`, explicit `instrument_type` override):** `test_select_04_single_run_covers_both_facilities` (lines 414-416) and `test_select_03_all_token_case_insensitive_syncs_everything` (line 396):
```python
soar_record = self._create_record(
    '620002', proposal='SHARED', facility='SOAR', site='sor', instrument_type='SOAR_GHTS_REDCAM'
)
```
New SOAR multi-config tests must keep `facility='SOAR'` and `site='sor'` exactly like this, adding `extra_params={'c_1_configuration_type': 'SPECTRUM', 'c_1_instrument_type': 'SOAR_GHTS_REDCAM', 'c_2_configuration_type': 'ARC', 'c_2_instrument_type': 'SOAR_GHTS_REDCAM', 'c_3_configuration_type': 'LAMP_FLAT', 'c_3_instrument_type': 'SOAR_GHTS_REDCAM', ...}` (per EXTRACT-02 / RESEARCH.md Wave 0 gap) on top.

## Shared Patterns

### Per-record catch-log-continue (error handling)
**Source:** `solsys_code/management/commands/sync_lco_observation_calendar.py` lines 254-259 (and the D-07 defensive variant lines 244-251)
**Apply to:** the new extraction helper's failure path inside `handle()`'s loop
```python
try:
    fields = _build_event_fields(record, facility)
except (KeyError, ValueError) as exc:
    self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
    counters[record.facility]['skipped'] += 1
    continue
```
**Critical caveat:** do not let the new D-06 failure path fall through this *existing* except block unmodified — it would silently merge into `'skipped'` rather than the new dedicated counter (RESEARCH.md Pitfall 1). Add a sentinel check or a second except clause specifically for the new failure mode.

### Module-level whitelist/lookup table
**Source:** `SITE_TELESCOPE_MAP` (lines 18-22) and `_FAILURE_PREFIX_BY_STATUS` (lines 32-37)
**Apply to:** D-01's science-vs-calibration `configuration_type` whitelist — define as a module-level set/dict near these two existing tables, with the same inline-comment-citing-source style (both existing tables have multi-line comments above them citing their confirmation source — follow this for the new whitelist, citing CONTEXT.md D-01 / the installed `tom_observations` line numbers already gathered in RESEARCH.md).

### Google-style docstrings with Args/Returns/Raises
**Source:** every existing helper in this file (`_derive_telescope`, `_failure_prefix`, `_title_for`, `_time_window`, `_build_event_fields`)
**Apply to:** all new helper function(s)

### Additive, non-breaking fixture-helper extension
**Source:** `_parameters()` / `_create_record()` in the test file
**Apply to:** the new `extra_params: dict | None = None` parameter — must not change any of the 5 existing named defaults or break any of the 19 existing tests (RESEARCH.md Pitfall 4 — re-run the full test file after the fixture change to confirm).

## No Analog Found

None — both files contain strong in-file analogs for every new piece of code in this phase; no external pattern search was needed.

## Metadata

**Analog search scope:** `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/tests/test_sync_lco_observation_calendar.py` (both files read in full; no other files in the repo touch `instrument_type`/`c_N_*`/`configuration_type` per RESEARCH.md's repo-wide grep)
**Files scanned:** 2
**Pattern extraction date:** 2026-06-20
