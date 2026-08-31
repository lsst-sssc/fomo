# Phase 7: Live Telescope-Label Resolution with Fallback & Failure Reporting - Pattern Map

**Mapped:** 2026-06-21
**Files analyzed:** 3 (1 source module modified, 1 test file extended, 1 paired notebook updated)
**Analogs found:** 3 / 3 (all self-referential — this phase extends its own existing module, no new files created)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | command (route/controller equivalent for a batch job) | request-response (per-record HTTP call) + batch | itself (Phase 4-6 code in the same file) | exact — same file, same function set, extend in place |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` | test | request-response (mocked HTTP) + CRUD (DB fixtures) | itself (Phase 4-6 tests in the same file) | exact — same file, same `TestCase`/fixture-helper conventions |
| `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` | demo notebook (companion deliverable, not optional) | file-I/O / batch (executed cells exercising the command) | itself (existing notebook from Phase 4-6) | exact — same notebook, add cells for new behavior |

No genuinely new files are created in this phase — `RESEARCH.md`'s "Recommended Project Structure" confirms all changes live inside the existing single-file management command, its existing test file, and its existing paired notebook. Analogs below are therefore **intra-file**: existing functions in the same module/test file/notebook that the new code must structurally match.

## Pattern Assignments

### `solsys_code/management/commands/sync_lco_observation_calendar.py` (command, request-response + batch)

**Analog:** the file's own existing functions (`_derive_telescope`, `_failure_prefix`, `_title_for`, `_build_event_fields`, `Command.handle`)

**Imports pattern** (lines 1-9, current):
```python
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord
```
New imports needed for this phase (add to this same block, alphabetically grouped by existing convention — stdlib, then third-party):
```python
import requests
from django import forms
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.facilities.ocs import make_request
from urllib.parse import urljoin
```

**Existing module-level constant + docstring-comment convention** (lines 11-22, `SITE_TELESCOPE_MAP`):
```python
# Site code -> telescope label, mirroring solsys_code/telescope_runs.py:SITES naming
# convention (e.g. 'FTS'). The 'coj'/'ogg' LCO entries are [ASSUMED] per RESEARCH.md
# Assumptions Log A1/A2 (web-search only, not yet confirmed against real
# ObservationRecord.parameters data for this project's LCO proposal) — confirm against
# real records before relying on this mapping in production. The 'sor' SOAR entry is
# confirmed (not [ASSUMED]) against tom_observations.facilities.soar, which hardcodes
# 'sitecode': 'sor'.
SITE_TELESCOPE_MAP = {
    'coj': 'FTS',
    'ogg': 'FTN',
    'sor': 'SOAR',
}
```
**Copy this exact commenting convention** for the new `(site, aperture_class) -> label` dict (D-03/D-04/D-05): cite the confirmation source per entry, tag unconfirmed entries `[ASSUMED]` (D-05/Pitfall 5's `tlv` entry specifically), and keep the dict definition immediately following the imports, before any function definitions — same placement as the existing dict.

**Terminal-prefix-priority pattern to extend** (lines 24-37, 55-68 — `_FAILURE_PREFIX_BY_STATUS` + `_failure_prefix`):
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
**Apply this exact "small lookup dict + one pure function with Args/Returns docstring" shape** to the new `[UNVERIFIED]` prefix logic — do not inline the new prefix condition directly into `_title_for`; keep it as composable as `_failure_prefix` is.

**Core pattern to extend — `_derive_telescope`** (lines 71-86, current signature/behavior to change):
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
Per RESEARCH.md Code Examples, this becomes two functions — keep the same Google-style docstring shape (`Args:`/`Returns:`, no `Raises:` once the new version returns `None` instead of raising, per TELESCOPE-03's "never abort" requirement):
```python
def _aperture_class_from_telescope_code(telescope_code: str) -> str | None: ...
def _derive_telescope(site: str, telescope_code: str) -> str | None: ...
```
Note the **return-type change from "raise KeyError" to "return None"** is itself the pattern shift this phase introduces — every other new function in this phase (`_resolve_placement_block`, the new fallback-label helper) must follow this same "never raise, return None/sentinel on any failure path" convention, matching SYNC-07's "never abort the run."

**Core pattern to extend — `_title_for`** (lines 166-184, current):
```python
def _title_for(record: ObservationRecord, telescope: str, instrument: str, facility: LCOFacility) -> str:
    """Build the CalendarEvent title for a record (D-03/D-04/D-06).
    ...
    """
    prefix = _failure_prefix(record.status, facility)
    if prefix is not None:
        return f'{prefix} {telescope} {instrument}'
    if record.scheduled_start is None:
        return f'[QUEUED] {telescope} {instrument}'
    return f'{telescope} {instrument}'
```
RESEARCH.md's Pattern 3 gives the exact extended form (new `label_was_fallback: bool` parameter, `[UNVERIFIED]` inserted between the terminal-prefix branch and the `[QUEUED]` branch) — **copy this signature/branch-order exactly**, do not reorder the existing branches:
```python
def _title_for(record, telescope, instrument, facility, label_was_fallback: bool) -> str:
    prefix = _failure_prefix(record.status, facility)
    if prefix is not None:
        return f'{prefix} {telescope} {instrument}'
    if record.scheduled_start is None:
        return f'[QUEUED] {telescope} {instrument}'
    if label_was_fallback:
        return f'[UNVERIFIED] {telescope} {instrument}'
    return f'{telescope} {instrument}'
```

**Core pattern for the new API call — model on `OCSFacility.get_observation_status()`** (installed `tom_observations/facilities/ocs.py:1548-1575`, NOT this repo's code but the direct analog to copy structurally):
```python
def get_observation_status(self, observation_id):
    response = make_request(
        'GET',
        urljoin(self.facility_settings.get_setting('portal_url'), f'/api/requests/{observation_id}/observations/'),
        headers=self._portal_headers()
    )
    blocks = response.json()
    current_block = None
    for block in blocks:
        if block['state'] == 'COMPLETED':
            current_block = block
            break
        elif block['state'] == 'PENDING':
            current_block = block
    if current_block:
        scheduled_start = current_block['start']
        scheduled_end = current_block['end']
    else:
        scheduled_start, scheduled_end = None, None
    return {'state': state, 'scheduled_start': scheduled_start, 'scheduled_end': scheduled_end}
```
**Copy the block-selection loop verbatim** (`COMPLETED`-first, else `PENDING`) into the new `_resolve_placement_block()` function per RESEARCH.md Pattern 1/Pitfall 3 — do not reimplement this selection independently, to avoid picking a different block than the one timing data comes from. New function adds `timeout=_API_TIMEOUT_SECONDS` to the `make_request()` call (D-10) and extracts `site`/`enclosure`/`telescope` from the matched block instead of `start`/`end`.

**Error handling pattern — new, no existing in-repo precedent (first explicit-timeout call), model on RESEARCH.md Pattern 2 exactly:**
```python
_API_TIMEOUT_SECONDS = 10  # D-10: explicit timeout, single attempt, no retry loop.

def _resolve_placement_block(observation_id: str, facility) -> dict[str, Any] | None:
    try:
        response = make_request(
            'GET',
            urljoin(facility.facility_settings.get_setting('portal_url'),
                     f'/api/requests/{observation_id}/observations/'),
            headers=facility._portal_headers(),
            timeout=_API_TIMEOUT_SECONDS,
        )
    except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError):
        # SYNC-09: never log str(exc) -- both library exception types embed response.content.
        return None
    blocks = response.json()
    ...
    return current_block
```
**Caller logging discipline (SYNC-09) — must follow this exact "fixed generic message, never str(exc)" shape**, matching the existing `Command.handle()` skip-and-log convention at lines 354-357/370/374:
```python
self.stderr.write(
    f'Skipping observation_id={record.observation_id!r}: unrecognized facility {record.facility!r}'
)
```
i.e. the existing convention is already "interpolate only known-safe values (`observation_id`, a fixed message), never an exception's `str()`" — this phase's new catch site continues that same discipline, it doesn't introduce a new logging style.

**Existing `try/except`-per-record dispatch in `Command.handle()` to extend** (lines 365-376):
```python
try:
    fields = _build_event_fields(record, facility)
except InstrumentExtractionError as exc:
    self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
    counters[record.facility]['extraction_failed'] += 1
    continue
except (KeyError, ValueError) as exc:
    self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
    counters[record.facility]['skipped'] += 1
    continue
```
The new telescope-API-failure path does **not** raise an exception out of `_build_event_fields` (per the "never raise" convention above) — it's a `bool`/sentinel return consumed inside `_build_event_fields` itself, then surfaced to `handle()` via the returned `fields` dict (e.g. an extra `telescope_api_failed: bool` key, popped before constructing the `CalendarEvent` kwargs, same as `url` is already popped at line 378). This keeps the same "one exception type per counter" structure already established, without adding a new exception class for a non-aborting condition.

---

### `solsys_code/tests/test_sync_lco_observation_calendar.py` (test, request-response mocked + CRUD fixtures)

**Analog:** the file's own existing fixture helpers and mock-based tests (`_parameters`, `_create_record`, `test_select_05_soar_record_uses_soar_facility_instance`)

**Imports pattern** (lines 1-13, current):
```python
import io
from datetime import datetime
from datetime import timezone as dt_timezone
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory
```
New tests need `MagicMock`/`side_effect` support (already importable via the existing `from unittest.mock import patch` — add `MagicMock` to the same import line) and `requests.exceptions.Timeout`/`RequestException` for SYNC-08's timeout-test double.

**Fixture-helper pattern to extend** (lines 16-49, `_parameters`) and (lines 58-77, `_create_record`):
```python
def _parameters(
    proposal: str = 'TESTCODE123',
    start: str = '2026-07-01T00:00:00',
    end: str = '2026-07-02T00:00:00',
    instrument_type: str = '2M0-SCICAM-MUSCAT',
    site: str | None = 'coj',
    extra_params: dict | None = None,
) -> dict:
    ...

def _create_record(
    self,
    observation_id: str,
    status: str = 'PENDING',
    scheduled_start: datetime | None = None,
    scheduled_end: datetime | None = None,
    facility: str = 'LCO',
    **parameter_overrides,
) -> ObservationRecord:
    return ObservationRecord.objects.create(
        target=self.target,
        user=self.user,
        facility=facility,
        observation_id=observation_id,
        status=status,
        scheduled_start=scheduled_start,
        scheduled_end=scheduled_end,
        parameters=_parameters(**parameter_overrides),
    )
```
**Add a new shared mock-response-builder helper** (RESEARCH.md Wave 0 Gaps) in this exact style — a plain module-level function returning a dict/`MagicMock`, parameterized by keyword args with defaults, docstring with `Args:`/`Returns:` — e.g. `_observations_block_response(site='lsc', enclosure='doma', telescope='1m0a', state='COMPLETED')` returning a `MagicMock` whose `.json()` returns the block list, for `patch`-ing `make_request`'s return value in the new TELESCOPE-02 success-path test.

**Mocking pattern to copy exactly** (lines 442-479, `test_select_05_soar_record_uses_soar_facility_instance` — the file's only existing example of patching a method used internally by the command and asserting call behavior):
```python
with (
    patch.object(
        SOARFacility,
        'get_observation_url',
        autospec=True,
        side_effect=real_get_observation_url,
    ) as soar_spy,
    patch.object(
        LCOFacility,
        'get_observation_url',
        autospec=True,
        side_effect=real_get_observation_url,
    ) as lco_spy,
):
    call_command(...)
    soar_spy.assert_called_once()
    lco_spy.assert_not_called()
```
For the new SYNC-08 "single attempt, no retry" test, **copy this `with patch.object(...) as spy: ...; spy.assert_called_once()` shape** but patch `make_request` as imported into `sync_lco_observation_calendar` (i.e. `patch('solsys_code.management.commands.sync_lco_observation_calendar.make_request', side_effect=requests.exceptions.Timeout)`), and assert `mock.assert_called_once()` to prove no retry loop ran.

**Test-method docstring convention to follow** — every existing test method opens with one line citing the requirement ID(s) it covers, e.g.:
```python
def test_sync_01_d01_url_uses_requests_path_not_requestgroups(self):
    """SYNC-01/D-01: event.url equals LCOFacility().get_observation_url(observation_id)."""
```
**New test names must follow this exact `test_{req_id_lower}_{description}` naming + one-line-docstring convention**, matching RESEARCH.md's Phase Requirements → Test Map naming (`test_telescope_01_verified_dict_covers_all_sites`, `test_sync_06_fallback_counter_distinct_from_skipped`, etc.).

---

### `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` (demo notebook, file-I/O / batch)

**Analog:** itself — the existing notebook's cell structure from Phase 4-6 (not read in full this pass; structure inferred from CLAUDE.md's explicit convention and RESEARCH.md's "Recommended Project Structure")

**Required pattern (per CLAUDE.md, mandatory not optional):** add cells exercising (1) a placed record with successful mocked API resolution producing a verified `SITECODE-CLASS` label, (2) a placed record with a mocked API failure/timeout producing the `[UNVERIFIED]`-prefixed fallback, and (3) the new `telescope_api_failed` summary-line counter visible in the printed output. Regenerate via:
```bash
jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
```
and commit **with output** (this is the one notebook location pre-commit's output-clearing hook does not apply to, per the existing `pre_executed/` convention already in place for this and the other two paired notebooks).

## Shared Patterns

### Never-raise / sentinel-on-failure convention (new to this phase, but consistent with existing module style)
**Source:** `solsys_code/management/commands/sync_lco_observation_calendar.py` existing `_derive_telescope` (raises `KeyError`, to be changed) vs. the rest of the module's per-record dispatch (`InstrumentExtractionError`, `KeyError`, `ValueError` — all caught at the `Command.handle()` loop level, never propagated past one record)
**Apply to:** `_resolve_placement_block`, the new `_derive_telescope(site, telescope_code)`, and any new fallback-label helper — all must return `None`/a sentinel rather than raising, since SYNC-07 requires the new failure mode (unlike the old `_derive_telescope`'s `KeyError`) to never abort or even skip the record — it must still produce a `CalendarEvent`, just with a fallback label.

### Per-facility counters dict extension
**Source:** `solsys_code/management/commands/sync_lco_observation_calendar.py:339-342` (`counters` dict literal) and `:392-402` (summary f-string)
**Apply to:** add `'telescope_api_failed': 0` to both `counters['LCO']` and `counters['SOAR']` dict literals, and extend the summary f-string with `telescope_api_failed: {counts["telescope_api_failed"]}` in the exact same `key: N` comma-joined phrasing already used for `extraction_failed`.
```python
counters = {
    'LCO': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0, 'extraction_failed': 0},
    'SOAR': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0, 'extraction_failed': 0},
}
...
summary = ' | '.join(
    f'{facility_name}: created: {counts["created"]}, updated: {counts["updated"]}, '
    f'unchanged: {counts["unchanged"]}, skipped: {counts["skipped"]}, '
    f'extraction_failed: {counts["extraction_failed"]}'
    for facility_name, counts in counters.items()
)
```

### SYNC-09 no-leak logging discipline
**Source:** existing `Command.handle()` skip-and-log lines (`solsys_code/management/commands/sync_lco_observation_calendar.py:354-357, 370, 374`) — these already only interpolate `record.observation_id`/`record.facility`/a caught exception's `str()` for *non-credential-bearing* exception types (`InstrumentExtractionError`, `KeyError`, `ValueError` — all locally-raised, never library exceptions touching HTTP responses)
**Apply to:** the new telescope-API catch site must NOT follow the `f'...: {exc}'` pattern used for those local exceptions — `ImproperCredentialsException`/`forms.ValidationError` are categorically different (library-raised, response-body-embedding) and need the fixed-generic-message form from RESEARCH.md Pattern 2's Pitfall 1 instead.

### Google-style docstring with Args/Returns(/Raises) on every new function
**Source:** every existing function in the file (`_failure_prefix`, `_derive_telescope`, `_extract_instrument`, `_title_for`, `_time_window`, `_build_event_fields`) uses this exact docstring shape — one-line summary, blank line, `Args:` block, blank line, `Returns:` block, optional `Raises:` block.
**Apply to:** all new functions (`_resolve_placement_block`, `_aperture_class_from_telescope_code`, the new 2-arg `_derive_telescope`, the extended `_title_for`) — match this shape exactly, dropping `Raises:` for any function in this phase that follows the new never-raise convention.

## No Analog Found

None — this phase has no files without an analog. Every change is an in-place extension of an existing module/test file/notebook, and every new function has a structurally close existing function in the same file to model on (see Pattern Assignments above). The one external (non-FOMO) analog used — `OCSFacility.get_observation_status()` in the installed `tom_observations` library — is documented above as the structural model for the new `_resolve_placement_block()`, explicitly NOT to be extended in place (per RESEARCH.md's locked recommendation against widening its return contract).

## Metadata

**Analog search scope:** `solsys_code/management/commands/sync_lco_observation_calendar.py` (full file read), `solsys_code/tests/test_sync_lco_observation_calendar.py` (read in sections — fixture helpers + `test_select_05_*` mocking example), installed `tom_observations/facilities/ocs.py` (targeted grep + read of `make_request`, `get_observation_status`, `_build_location`)
**Files scanned:** 3 in-repo files (1 source, 1 test, 1 notebook referenced not opened — binary/large ipynb, structure taken from CLAUDE.md's explicit convention text) + 1 installed third-party library file
**Pattern extraction date:** 2026-06-21
