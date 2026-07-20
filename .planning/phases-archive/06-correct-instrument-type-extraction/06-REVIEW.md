---
phase: 06-correct-instrument-type-extraction
reviewed: 2026-06-20T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
findings:
  critical: 0
  warning: 4
  info: 2
  total: 6
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-06-20T00:00:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed `sync_lco_observation_calendar.py` (the management command implementing
instrument-type extraction and CalendarEvent sync) and its test suite. All 22 Django
tests pass (`./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`),
and `ruff check`/`ruff format --check` are clean. The `_FAILURE_PREFIX_BY_STATUS` and
`_SCIENCE_CONFIGURATION_TYPES` mappings were cross-checked against the installed
`tom_observations` library (`ocs.py`, `lco.py`, `soar.py`) and match the library's
actual behavior (`get_failed_observing_states()` returns exactly the 4 statuses keyed
in the dict; `max_instrument_configs`/`max_configurations` are both 5).

No critical/security issues were found. The findings below are about an idempotency
gap in the `get_or_create` pattern, a silent-no-op UX gap for blank `--proposal`
values, one provably-unreachable defensive branch, a hardcoded instrument-config index
in MUSCAT detection, and two test-coverage gaps for already-written exception paths.

## Warnings

### WR-01: `get_or_create(url=...)` without a DB-level unique constraint on `url` risks duplicate CalendarEvents under concurrent runs

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:379`
**Issue:** The command's entire idempotency story (SYNC-04 "no churn on unchanged",
dedup keyed on `url`) depends on `url` uniquely identifying a record. `CalendarEvent.url`
(in the installed `tom_calendar` library) is declared `models.URLField(blank=True,
default="")` with no `unique=True` and no `Meta.constraints`/`unique_together`. Django's
`get_or_create` is documented to be unsafe against races without a DB-level unique
constraint: two overlapping invocations of this command (e.g. an overlapping cron
schedule, or a manual run racing a scheduled one — the intended real-world usage
pattern for a sync command) can both pass the `get()` check, both find nothing, and
both `create()`, producing two `CalendarEvent` rows for the same observation.
**Fix:** Add a `unique=True` constraint (or `UniqueConstraint` on `url`) via a migration
in `tom_calendar`, or wrap the `get_or_create` call in `transaction.atomic()` with
`select_for_update()`/`IntegrityError` handling if the upstream model can't be changed:
```python
from django.db import IntegrityError, transaction

try:
    with transaction.atomic():
        event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)
except IntegrityError:
    event = CalendarEvent.objects.get(url=url)
    created = False
```
(Requires a unique constraint on `url` to actually close the race; without one, the
`atomic()` wrapper alone does not prevent the duplicate.)

### WR-02: Blank or whitespace-only `--proposal` value silently matches zero records instead of erroring

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:271-293, 345-347`
**Issue:** `_parse_proposal_arg('')` and `_parse_proposal_arg('   ')` both return `[]`
(every segment is stripped to empty and dropped). Back in `handle()`, `codes = []` is
not `None`, so the code takes the `records.filter(parameters__proposal__in=[])` branch,
which Django evaluates to an empty queryset. The command then exits cleanly with
`Done. proposal: , LCO: created: 0, ... | SOAR: created: 0, ...` — no error, no warning,
just a confusing zero-record run. `argparse`'s `required=True` only checks the flag was
supplied, not that the value is non-empty, so `--proposal ""` or `--proposal "  "` slips
through silently.
**Fix:** Raise a `CommandError` when the parsed code list is empty (and not the `ALL`
sentinel):
```python
codes = _parse_proposal_arg(proposal)
if codes is not None and not codes:
    raise CommandError(f'--proposal {proposal!r} contained no usable proposal codes')
```

### WR-03: `_has_muscat_exposure_signal` only checks instrument-config index 1 (`ic_1`), missing `ic_2`-`ic_5` exposure signals

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:89-99`
**Issue:** The installed `tom_observations.facilities.lco.LCOMuscatImagingObservationForm`
supports up to `max_instrument_configs = 5` instrument-configs per configuration block
(`c_{j}_ic_{i}_*` for `i` in 1..5; confirmed at `lco.py:580-599`). `_has_muscat_exposure_signal`
hardcodes `ic_1` only: `parameters.get(f'c_{n}_ic_1_exposure_time_{suffix}')`. While
MUSCAT's *simple* submission form always normalizes to `c_1_ic_1_*` (per `lco.py:615`),
nothing in the docstring or the calling code restricts this function to only ever being
called against simple-form submissions — if a future/advanced MUSCAT submission (or a
different multi-instrument-config record) populates exposure signal at `ic_2` or higher,
`_find_exposure_signal_config` will silently treat that config as having no exposure
signal and fall through past it, potentially landing on the flat `instrument_type` key
or raising `InstrumentExtractionError` incorrectly.
**Fix:** Loop over instrument-config indices too, mirroring the configuration-index loop:
```python
def _has_muscat_exposure_signal(parameters: dict[str, Any], n: int) -> bool:
    return any(
        parameters.get(f'c_{n}_ic_{i}_exposure_time_{suffix}')
        for i in range(1, 6)
        for suffix in _MUSCAT_CHANNEL_SUFFIXES
    )
```

### WR-04: `_find_science_config`'s "scan past non-science configs" behavior is untested — every test fixture places the science config at index 1

**File:** `solsys_code/tests/test_sync_lco_observation_calendar.py:481-516`
**Issue:** `_find_science_config` loops `range(1, 6)` specifically so it can skip past
calibration configs (ARC/LAMP_FLAT) to find a science config that appears *later* in the
parameters. Every test fixture in the suite (`test_extract_02_soar_multi_config_picks_spectrum_not_calibration`
and both MUSCAT extraction tests) places the science `configuration_type` at `c_1`,
matching the SOAR "Simple Goodman" form's *default* layout — but the loop's actual
differentiating behavior (continuing past `c_1` when it is calibration, to find science
at `c_2`/`c_3`) is never exercised. A regression that broke the loop (e.g. an
off-by-one, or accidentally `break`ing on the first non-science config instead of
`continue`-ing) would not be caught by the current suite.
**Fix:** Add a test fixture with calibration configs preceding the science config, e.g.
`c_1_configuration_type='ARC'`, `c_2_configuration_type='SPECTRUM'`, and assert the
extracted instrument comes from `c_2_instrument_type`.

## Info

### IN-01: `facility is None` defensive branch is provably unreachable given the preceding queryset filter

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:344, 350-363`
**Issue:** `records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])`
guarantees every iterated `record.facility` is exactly `'LCO'` or `'SOAR'`, and
`facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}` has keys for precisely
those two values. `facilities.get(record.facility)` can therefore never return `None`
for any row produced by this loop — this isn't merely "shouldn't happen" as the comment
states, it's unreachable by construction as the code is currently written. This is
defensible as future-proofing against the filter changing later, but the comment
overstates it as a runtime possibility rather than a structural guarantee, and the
branch carries no test coverage (correctly, since it can't be hit).
**Fix:** Either leave as documented defensive code but soften the comment to make the
unreachability explicit (e.g. "currently unreachable given the filter above; kept in
case the filter is loosened"), or remove it and rely on a `KeyError` from a dict-index
lookup with the same message, which would at least be honest about being a hard failure
rather than dead skip-and-log logic.

### IN-02: No test covers `_derive_telescope`'s `KeyError` path for an unmapped site code

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:71-86`
**Issue:** `SITE_TELESCOPE_MAP` is explicitly called out in its own comment as an
`[ASSUMED]` mapping (RESEARCH.md Assumptions Log A1/A2) not yet confirmed against real
production data, and `_derive_telescope` raises a descriptive `KeyError` for any
unmapped site code so the run can skip-and-log rather than crash. This is exactly the
kind of "assumption might be wrong" path that benefits most from a regression test, but
no test in the suite supplies a `site` value outside `{'coj', 'ogg', 'sor'}` to verify
the record is skipped (not crashed on) and the observation_id appears in stderr.
**Fix:** Add a test analogous to `test_skip_path_missing_site_logged_and_skipped` using
e.g. `site='tfn'` (a real LCO site not yet in the map) and assert the record is skipped
and logged, while a sibling record with a mapped site still syncs.

---

_Reviewed: 2026-06-20T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
