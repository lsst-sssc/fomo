# Phase 5: Multi-Proposal & Multi-Facility Selection - Pattern Map

**Mapped:** 2026-06-19
**Files analyzed:** 3 (all modified, no new files)
**Analogs found:** 3 / 3 (self-analogs — this phase generalizes its own Phase 4 predecessor code)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | controller (management command) | batch / CRUD (find-or-create + per-record dispatch) | itself (Phase 4 version, same file) | exact — in-place generalization, not a new role |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` | test | request-response (via `call_command`) / CRUD assertions | itself (Phase 4 version, same file) | exact |
| `src/fomo/settings.py` (`FACILITIES` dict) | config | — (static config, no data flow) | `FACILITIES['LCO']` entry in the same file | exact — same dict, new sibling key |

This phase has no cross-codebase analog search to do beyond the files themselves: Phase 4 already
established every pattern needed (single-facility instantiation, single-proposal filter, per-record
catch-log-continue, no-churn upsert, stdout summary). Phase 5 is a structured generalization of that
exact code, so the "analog" for each file is its own current (Phase 4) state.

## Pattern Assignments

### `solsys_code/management/commands/sync_lco_observation_calendar.py` (controller, batch/CRUD)

**Analog:** the file's own current contents (Phase 4), specifically `add_arguments`, `handle()`,
and the helper functions which already accept a `facility` parameter unchanged.

**Imports pattern** (current lines 1-8 — add `SOARFacility` alongside the existing `LCOFacility` import):
```python
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility  # NEW — Phase 5
from tom_observations.models import ObservationRecord
```
No path-alias or barrel-import conventions in this codebase — flat absolute imports, alphabetized
within each block, exactly as already present. Add the one new `SOARFacility` import directly below
`LCOFacility`'s line to keep the existing block's alphabetical/grouping convention.

**Helper functions need NO signature changes** — `_failure_prefix`, `_title_for`, `_build_event_fields`
already take `facility: LCOFacility` as a parameter (current lines 36, 70, 122) and only call
`facility.get_failed_observing_states()` / `facility.get_observation_url()` on it — both inherited
unchanged by `SOARFacility`. The type hint `facility: LCOFacility` can stay as-is (SOAR *is-a* LCO
facility via inheritance) or be loosened to a `Union`/no-hint; this is a discretion-level cosmetic
choice, not a functional one.

**`add_arguments` pattern to replace** (current lines 166-173):
```python
def add_arguments(self, parser: CommandParser) -> None:
    """Parse command line arguments."""
    parser.add_argument(
        '--proposal',
        type=str,
        required=True,
        help='LCO proposal code to filter ObservationRecords by',
    )
```
Keep the same `parser.add_argument` call shape and `help=` docstring convention; only the help text
needs updating to mention comma-list/`ALL` syntax (D-01/D-02). Argument stays `type=str`, `required=True`
— parsing into a list/`ALL` sentinel happens in `handle()` (per RESEARCH.md Pattern 1), not in
`add_arguments` itself, matching this file's existing separation (parsing/validation lives in `handle`,
e.g. the current `proposal = options['proposal']` at line 186).

**Core dispatch pattern to replace** (current lines 186-194 — the D-06 bug site):
```python
proposal = options['proposal']
facility = LCOFacility()
...
records = ObservationRecord.objects.filter(facility='LCO', parameters__proposal=proposal)
```
Replace with the eager dispatch dict + conditional filter (RESEARCH.md Patterns 2/3, both confirmed
live against this repo's DB):
```python
facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}
codes = _parse_proposal_arg(options['proposal'])
records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])
if codes is not None:
    records = records.filter(parameters__proposal__in=codes)
```

**Per-record loop pattern to replace** (current lines 196-216 — single shared `facility` used for every
record at line 198's `_build_event_fields(record, facility)` call):
```python
for record in records:
    try:
        fields = _build_event_fields(record, facility)
    except (KeyError, ValueError) as exc:
        self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
        skipped_count += 1
        continue
    url = fields.pop('url')
    event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)
    if created:
        created_count += 1
    else:
        changed = any(getattr(event, field_name) != value for field_name, value in fields.items())
        if changed:
            for field_name, value in fields.items():
                setattr(event, field_name, value)
            event.save()
            updated_count += 1
        else:
            unchanged_count += 1
```
Generalize to dispatch by `record.facility` (D-06/D-07) and track counters per facility (D-08):
```python
for record in records:
    facility = facilities.get(record.facility)
    if facility is None:
        self.stderr.write(
            f'Skipping observation_id={record.observation_id!r}: unrecognized facility {record.facility!r}'
        )
        counters[record.facility]['skipped'] += 1  # or a dedicated 'unknown' bucket — discretion
        continue
    try:
        fields = _build_event_fields(record, facility)
    except (KeyError, ValueError) as exc:
        self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
        counters[record.facility]['skipped'] += 1
        continue
    url = fields.pop('url')
    event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)
    if created:
        counters[record.facility]['created'] += 1
    else:
        changed = any(getattr(event, field_name) != value for field_name, value in fields.items())
        if changed:
            for field_name, value in fields.items():
                setattr(event, field_name, value)
            event.save()
            counters[record.facility]['updated'] += 1
        else:
            counters[record.facility]['unchanged'] += 1
```
The no-churn save-only-if-changed body is **byte-for-byte unchanged** from Phase 4 — only the counter
variable names change from flat ints to a `counters[facility_name][bucket]` dict, initialized eagerly
for both `'LCO'` and `'SOAR'` keys before the loop (mirrors D-06's "build both eagerly" philosophy
applied to counters too).

**Summary/error-handling pattern to replace** (current lines 218-224 — single aggregate line):
```python
self.stdout.write(
    f'Done. proposal: {proposal}, '
    f'created: {created_count}, '
    f'updated: {updated_count}, '
    f'unchanged: {unchanged_count}, '
    f'skipped: {skipped_count}'
)
```
Generalize to a per-facility breakdown (D-08, exact format is discretion):
```python
summary_parts = [
    f"{name}: {c['created']} created, {c['updated']} updated, "
    f"{c['unchanged']} unchanged, {c['skipped']} skipped"
    for name, c in counters.items()
]
self.stdout.write('Done. ' + ' | '.join(summary_parts))
```
Keep `self.stdout.write(...)` / `self.stderr.write(...)` — this file does NOT use `logger.*` calls
(unlike most of the rest of this codebase); CONTEXT.md/RESEARCH.md both confirm this management-command
convention should continue unchanged, not be replaced with `logging.getLogger(__name__)`.

**New helper function pattern** (`_parse_proposal_arg`, D-01/D-02/D-03 — same Google-style docstring
convention as every other helper already in this file, e.g. `_derive_telescope` at line 52):
```python
def _parse_proposal_arg(raw: str) -> list[str] | None:
    """Parse the --proposal argument into a code list, or None for ALL (D-01/D-02/D-03).

    Args:
        raw: the raw --proposal string as typed (e.g. 'A, B,B,' or 'all').

    Returns:
        list[str] | None: deduped, stripped, case-preserved list of codes, in first-seen
            order; or None if raw case-insensitively equals 'all' (sync everything).
    """
    if raw.strip().lower() == 'all':
        return None
    seen: dict[str, None] = {}
    for segment in raw.split(','):
        code = segment.strip()
        if code and code not in seen:
            seen[code] = None
    return list(seen)
```
Place this directly above `class Command` alongside the other module-level helpers (`_failure_prefix`,
`_derive_telescope`, `_title_for`, `_time_window`, `_build_event_fields`), matching the existing file's
flat-module-function-then-class-at-bottom structure.

---

### `solsys_code/tests/test_sync_lco_observation_calendar.py` (test, request-response via `call_command`)

**Analog:** the file's own current contents (Phase 4) — `_parameters()`, `_create_record()`, and every
existing `test_*` method's structure (create fixture record(s) -> `call_command(..., stdout=io.StringIO(),
stderr=io.StringIO())` -> assert on `CalendarEvent` rows / stdout / stderr content).

**Imports pattern** (current lines 1-11 — add `SOARFacility` alongside `LCOFacility`):
```python
import io
from datetime import datetime
from datetime import timezone as dt_timezone

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility  # NEW — Phase 5, for SELECT-05's spy/patch
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory
```
Per CLAUDE.md, `NonSiderealTargetFactory` is already used (line 11, line 48) — no change needed there,
just reuse the existing class-level `cls.target` fixture for all new `ObservationRecord` rows (no new
`Target` fixtures required this phase, confirmed in RESEARCH.md).

**`_create_record()` signature extension required** (RESEARCH.md Pitfall 4 — current lines 50-68
hardcode `facility='LCO'` with no override parameter):
```python
def _create_record(
    self,
    observation_id: str,
    status: str = 'PENDING',
    scheduled_start: datetime | None = None,
    scheduled_end: datetime | None = None,
    facility: str = 'LCO',  # NEW — Phase 5
    **parameter_overrides,
) -> ObservationRecord:
    """Create an ObservationRecord fixture sharing the class-level target/user."""
    return ObservationRecord.objects.create(
        target=self.target,
        user=self.user,
        facility=facility,  # was hardcoded 'LCO'
        observation_id=observation_id,
        status=status,
        scheduled_start=scheduled_start,
        scheduled_end=scheduled_end,
        parameters=_parameters(**parameter_overrides),
    )
```
`_parameters()` itself (current lines 14-41) needs **no changes** — `facility` is a field on
`ObservationRecord`, not part of `parameters`, so it's orthogonal to this helper.

**Core test-method pattern** (e.g. `test_select_01_only_matching_proposal_creates_events`, lines 70-81)
— the exact shape every new SELECT-02..05 test should follow:
```python
def test_select_01_only_matching_proposal_creates_events(self):
    """SELECT-01: only the matching-proposal record creates a CalendarEvent."""
    self._create_record('111111', proposal='MATCHCODE')
    self._create_record('222222', proposal='OTHERCODE')
    call_command(
        'sync_lco_observation_calendar',
        '--proposal',
        'MATCHCODE',
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    self.assertEqual(CalendarEvent.objects.count(), 1)
```
For SELECT-02 (comma-list, no substring leakage), extend with a decoy `proposal='AB'` record exactly
as RESEARCH.md's "Confirmed live this session" code example shows, and assert the decoy's
`observation_id` produced no event.

For SELECT-03 (`ALL` token, case-insensitive), pass `'--proposal', 'all'` (or `'All'`/`'ALL'`) and
assert every fixture record (regardless of proposal) produced an event.

For SELECT-04 (both facilities in one run), use the new `facility='SOAR'` parameter on `_create_record`
and assert `CalendarEvent` rows exist for both an LCO-origin and a SOAR-origin record in the same
`call_command` invocation.

**Discriminating test pattern required for SELECT-05** (RESEARCH.md Pitfall 3 — a black-box `event.url`
assertion is provably insufficient since `SOARFacility().get_observation_url()` ==
`LCOFacility().get_observation_url()` byte-for-byte). Use `unittest.mock.patch` to spy on construction
or method calls, e.g.:
```python
from unittest.mock import patch

def test_select_05_soar_record_uses_soar_facility_instance(self):
    """SELECT-05: a SOAR record is dispatched via SOARFacility, never a shared LCOFacility."""
    self._create_record('700010', proposal='MATCHCODE', facility='SOAR')
    with (
        patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.SOARFacility.get_observation_url',
            wraps=SOARFacility.get_observation_url,
        ) as soar_spy,
        patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.LCOFacility.get_observation_url',
            wraps=LCOFacility.get_observation_url,
        ) as lco_spy,
    ):
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
    soar_spy.assert_called_once()
    lco_spy.assert_not_called()
```
(Exact patch target path / mock style is implementation discretion — the load-bearing requirement per
RESEARCH.md is that the assertion discriminates which *class's* method was invoked, not just the
resulting URL string.)

**No mocking of facility construction itself elsewhere** — every other existing test calls
`LCOFacility()` directly inline (e.g. line 94, line 279-280) for computing expected URLs; continue this
unmocked-by-default convention for all non-SELECT-05 tests, since `LCOFacility()`/`SOARFacility()`
construction does no I/O (confirmed in RESEARCH.md).

---

### `src/fomo/settings.py` (config, FACILITIES dict)

**Analog:** the existing `FACILITIES['LCO']` entry in the same dict (current lines 215-218).

**Current pattern to mirror** (lines 214-218):
```python
FACILITIES = {
    'LCO': {
        'portal_url': 'https://observe.lco.global',
        'api_key': '',
    },
```

**New entry to add** (D-04/D-05, per RESEARCH.md Pitfall 1's recommended narrower reading — option (b),
mirror the LCO entry's literal value exactly rather than introducing a new env var this phase didn't
otherwise need):
```python
FACILITIES = {
    'LCO': {
        'portal_url': 'https://observe.lco.global',
        'api_key': '',
    },
    'SOAR': {
        # Reuses the same LCO Observation Portal credentials as 'LCO' above (D-05) —
        # SOARFacility authenticates against the same API, not a distinct SOAR-specific one.
        'portal_url': 'https://observe.lco.global',
        'api_key': '',
    },
    'GEM': {
        ...  # unchanged
```
Place the new `'SOAR'` key directly after `'LCO'` (before `'GEM'`) to group the two LCO-family
facilities together — purely cosmetic ordering, no functional requirement, but matches the dict's
existing implicit grouping (LCO-family first, then GEM).

**Open item flagged for the planner (not resolved here, per RESEARCH.md Open Question 1):** whether to
also migrate the existing `'LCO'` entry's `api_key` to `os.getenv('LCO_API_KEY', '')` as part of this
phase. RESEARCH.md recommends the narrower reading — leave `'LCO'`'s value untouched, give `'SOAR'`
the same literal `''` — to stay within this phase's stated "query/selection/dispatch scope only"
boundary. The planner should make this an explicit one-line decision in the PLAN.md, not silently
pick a default.

## Shared Patterns

### Per-record catch-log-continue error handling
**Source:** `solsys_code/management/commands/sync_lco_observation_calendar.py` lines 196-202 (current),
established convention restated from Phase 3's D-02.
**Apply to:** Both the existing `(KeyError, ValueError)` catch around `_build_event_fields` AND the new
D-07 "unrecognized facility" branch — same `self.stderr.write(f'Skipping observation_id={record.observation_id!r}: ...')`
message shape, same "increment a counter, `continue`, never abort the whole run" structure.

### No-churn idempotent save
**Source:** `solsys_code/management/commands/sync_lco_observation_calendar.py` lines 204-216 (current).
**Apply to:** Unchanged in this phase — only the counter variables threading through this block change
shape (flat int -> `counters[facility][bucket]`); the `get_or_create` + conditional `any(...)` /
`setattr` / `.save()` logic itself is copied verbatim.

### Google-style docstrings with Args/Returns/Raises
**Source:** every existing helper in `sync_lco_observation_calendar.py` (e.g. `_derive_telescope` lines
52-67, `_build_event_fields` lines 122-137).
**Apply to:** The new `_parse_proposal_arg` helper, and any new docstring/comment additions around the
`FACILITIES['SOAR']` settings.py entry (per CONTEXT.md's "Claude's Discretion" on D-04 comment wording).

### `self.stdout.write(...)` / `self.stderr.write(...)`, not `logger.*`
**Source:** entire `sync_lco_observation_calendar.py` file — no `logging.getLogger(__name__)` call
exists in this file, unlike the rest of the codebase's general logging convention.
**Apply to:** All new stdout/stderr output in this phase (D-08 summary line, D-07 skip message) —
continue this file-local exception to the general logging convention rather than introducing `logger.*`
calls inconsistent with the rest of the file.

## No Analog Found

None — every file in scope already exists with a Phase-4-established pattern that this phase
generalizes in place. No new role/data-flow combination is introduced.

## Metadata

**Analog search scope:** `solsys_code/management/commands/sync_lco_observation_calendar.py`,
`solsys_code/tests/test_sync_lco_observation_calendar.py`, `src/fomo/settings.py` (all three read in
full this session — each well under 2,000 lines, single-pass reads, no re-reads needed).
**Files scanned:** 3 (all in scope; no broader codebase search was needed since CONTEXT.md/RESEARCH.md
both explicitly point at these exact files' Phase 4 versions as the canonical pattern source).
**Pattern extraction date:** 2026-06-19
