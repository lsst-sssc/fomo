# Phase 4: LCO Queue Sync Command - Research

**Researched:** 2026-06-17
**Domain:** Django management command syncing third-party (`tom_observations`) ORM records into a sibling app's (`tom_calendar`) model, using TOM Toolkit facility-class helpers
**Confidence:** HIGH

## Summary

Phase 4 is a single Django management command (`sync_lco_observation_calendar`) that reads
`tom_observations.models.ObservationRecord` rows and upserts `tom_calendar.models.CalendarEvent`
rows. All the model/library facts CONTEXT.md flagged as open questions were verified live against
this repo's actual installed environment (`tomtoolkit==3.0.0a9`, Django `5.2.15`, SQLite `3.34.1`)
rather than from source-reading alone, by running real ORM queries and instantiating
`LCOFacility()` in `manage.py shell`. Every one of CONTEXT.md's open questions has a definitive,
empirically-confirmed answer below — there are no remaining "confirm during planning" gaps for
the facility/model API surface. One finding **corrects** CONTEXT.md/TERM-01's framing:
`get_terminal_observing_states()` returns **5** states, not 4 — it includes `'COMPLETED'` in
addition to the 4 failure states. The planner must decide how completed (successful) records are
titled, since TERM-01's 3-prefix table (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`) has no entry for it.

The closest analog, `solsys_code/management/commands/load_telescope_runs.py`, remains the right
structural template: `add_arguments` for `--proposal`, a `handle()` loop with per-record
try/except-skip, a conditional-save upsert, and an stdout summary line. The only structural
difference from that analog is the upsert key: Phase 3 keys on `(telescope, instrument,
start_time)`; Phase 4 must key on `CalendarEvent.url` (per D-01/SYNC-01), built from
`LCOFacility().get_observation_url(observation_id)`.

**Primary recommendation:** Build the command exactly on the `load_telescope_runs.py` skeleton,
but key the upsert on `url`; instantiate `LCOFacility()` once per command run (no network cost at
`__init__`); derive `telescope` via a static dict keyed on `parameters['site']` (LCO 3-letter site
code, e.g. `'coj'`), separate from `instrument` which is `parameters['instrument_type']` verbatim
(e.g. `'2M0-SCICAM-MUSCAT'`); and decide explicitly (flag to planner) what title prefix, if any,
applies to `COMPLETED` records, since it is terminal per the library but absent from TERM-01's
3-state prefix table.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Record selection (`--proposal` filter) | Database / ORM | — | `JSONField` key lookup (`parameters__proposal=`) executes as SQL via SQLite's `JSON_EXTRACT`, confirmed working in this repo |
| CalendarEvent upsert | API / Backend (management command) | Database / ORM | Business logic lives in the command's `handle()`; persistence is plain Django ORM `save()`/`get_or_create()` |
| Portal URL construction | API / Backend | — | `LCOFacility().get_observation_url()` is a pure string-building helper, no I/O |
| Terminal-state detection | API / Backend | — | `record.terminal` property / `LCOFacility().get_terminal_observing_states()` — in-process, no network call |
| Telescope/instrument labeling | API / Backend | — | Static lookup dict, mirrors `telescope_runs.py:SITES` pattern but keyed on LCO site codes, not MPC obscodes |
| Calendar display | Browser / Client (existing `tom_calendar` views/templates) | — | Out of scope for this phase — no template changes (Out of Scope table in REQUIREMENTS.md) |

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SELECT-01 | `--proposal <code>` filters `ObservationRecord(facility='LCO')` by `parameters['proposal']` | Confirmed: `ObservationRecord.objects.filter(facility='LCO', parameters__proposal=code)` works correctly on this repo's SQLite/Django via `JSON_EXTRACT` — see Code Examples |
| SYNC-01 | Create/update one `CalendarEvent` per record, keyed on `url` (real portal URL via `get_observation_url`, not the literal string in REQUIREMENTS.md — see D-01) | Confirmed: `LCOFacility().get_observation_url('123456')` → `'https://observe.lco.global/requests/123456'` |
| SYNC-02 | Unscheduled record (`scheduled_start is None`): times from `parameters['start']`/`['end']`; title shows queue status | Confirmed: these are stored as ISO-8601 **strings**, need `datetime.fromisoformat()` parsing before assignment to `DateTimeField` |
| SYNC-03 | Scheduled record: times from `scheduled_start`/`scheduled_end` (already `datetime`, from `DateTimeField`) | `DateTimeField` values come back as aware `datetime` objects already — no parsing needed, unlike SYNC-02's path |
| SYNC-04 | Re-run updates in place, no duplicate, no `modified` churn on unchanged records | Mirror `load_telescope_runs.py`'s conditional-save pattern: compare all target fields before calling `.save()` |
| SYNC-05 | `instrument`, `proposal`, `telescope` populated from record parameters | `parameters['instrument_type']`, `parameters['proposal']` direct; `telescope` via site-code lookup table (see Don't Hand-Roll / Pitfall 2) |
| TERM-01 | Terminal-state records get title prefix (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`); event retained | Confirmed 4 of the **5** states in `get_terminal_observing_states()` map to TERM-01's 3 prefixes; `'COMPLETED'` (the 5th, successful-terminal state) is unaddressed by TERM-01 — planner must resolve (see Common Pitfalls) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `tomtoolkit` (`tom_observations`, `tom_calendar`) | 3.0.0a9 (installed, confirmed via `pip show`) | Source/destination ORM models + facility helper classes | Already a project dependency; this phase only consumes existing models, no new package |
| Django | 5.2.15 (installed, confirmed) | ORM, management command framework, `JSONField` | Existing project framework |

No new packages are required for this phase — confirmed by reading `pyproject.toml`'s dependency
list (`tomtoolkit`, `tom_observations` is bundled inside `tomtoolkit`) and by the fact that all
needed classes (`ObservationRecord`, `CalendarEvent`, `LCOFacility`) are already importable in this
venv (`/home/tlister/venv/fomo311_venv`).

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `datetime` (stdlib) | builtin | Parse `parameters['start']`/`['end']` ISO strings into aware `datetime` | Always for SYNC-02's banner-time branch |
| `django.core.management.base.BaseCommand`/`CommandError`/`CommandParser` | bundled with Django | Command scaffolding | Same pattern as `load_telescope_runs.py` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ORM-level `parameters__proposal=code` filter | `ObservationRecord.objects.filter(facility='LCO')` + Python-side `[r for r in qs if r.parameters.get('proposal') == code]` | ORM-level filter is confirmed working and is the better choice — fewer rows pulled into Python, same correctness. Python-side filtering would only be needed if a future Django/SQLite combination broke `JSON_EXTRACT` support, which is not the case here |
| Static dict for site→telescope label | Live `LCOFacility.get_instruments()` (network call, instrument-level, no site mapping) | `get_instruments()` doesn't return a site→telescope mapping at all — it returns instrument metadata (filters, types) and requires either cached or live API access with credentials. Not usable for this phase's telescope-label need; a static dict is simpler, deterministic, and avoids a network dependency in a unit test |

**Installation:**
No new packages — uses only currently-installed `tomtoolkit`/Django.

**Version verification:**
```bash
pip show tomtoolkit   # Version: 3.0.0a9 (confirmed installed)
python -c "import django; print(django.VERSION)"   # (5, 2, 15, 'final', 0)
```

## Package Legitimacy Audit

Not applicable — this phase installs no new packages. All classes consumed
(`tom_observations.models.ObservationRecord`, `tom_observations.facilities.lco.LCOFacility`,
`tom_calendar.models.CalendarEvent`) come from `tomtoolkit`, already a locked, installed project
dependency confirmed present in `/home/tlister/venv/fomo311_venv`.

## Architecture Patterns

### System Architecture Diagram

```
$ ./manage.py sync_lco_observation_calendar --proposal PROPOSAL2025A-001
        |
        v
[Command.handle()]
        |
        |--> ObservationRecord.objects.filter(facility='LCO',
        |        parameters__proposal=<code>)              (SELECT-01)
        |
        |--> LCOFacility()  [instantiated once, no I/O]
        |        |--> .get_observation_url(observation_id)  -> url   (SYNC-01/D-01)
        |        `--> .get_terminal_observing_states()       -> 5 states (TERM-01)
        |
        v
   for each ObservationRecord:
        |
        |-- derive telescope  <- SITE_TELESCOPE_MAP[parameters['site']]      (SYNC-05/D-02)
        |-- derive instrument <- parameters['instrument_type']               (SYNC-05)
        |-- branch on scheduled_start:
        |       None      -> start/end = parse(parameters['start']/['end'])  (SYNC-02)
        |       not None  -> start/end = scheduled_start/scheduled_end       (SYNC-03)
        |-- branch on record.status:
        |       in terminal-failure set -> title prefix [EXPIRED|CANCELLED|FAILED]  (TERM-01)
        |       == 'COMPLETED'           -> *unresolved by TERM-01* (planner decision)
        |       else (queued/scheduled)  -> '[QUEUED] ...' or clean title           (D-03/D-04)
        |-- build description: proposal + status + active time window        (D-05)
        |
        v
   CalendarEvent.objects.get_or_create(url=url, defaults={...})
        |
        |-- created            -> created_count += 1
        |-- exists, changed    -> update fields, .save(), updated_count += 1
        |-- exists, unchanged  -> unchanged_count += 1   (no .save() call -> no modified churn)
        |
        v
   stdout summary: "created: N, updated: N, unchanged: N, skipped: N"
```

### Recommended Project Structure
```
solsys_code/
├── management/
│   └── commands/
│       └── sync_lco_observation_calendar.py   # new — mirrors load_telescope_runs.py
├── tests/
│   └── test_sync_lco_observation_calendar.py  # new — mirrors test_load_telescope_runs.py
```

No new module under `solsys_code/` is strictly required for the site→telescope map — it can live
as a module-level dict at the top of the command file (same scale as `telescope_runs.py:SITES`,
~4-6 entries). If the planner prefers separation of concerns matching `telescope_runs.py`'s
existing convention, a small constant could instead live in `solsys_code/telescope_runs.py`
alongside `SITES`, but there is no functional need to import `Observatory`/MPC-obscode logic for
this phase — LCO site codes (`coj`, `ogg`, ...) and MPC obscodes (`E10`, ...) are different
vocabularies for the same physical sites.

### Pattern 1: Upsert keyed on a single unique-ish field, with conditional save
**What:** `get_or_create(url=url, defaults={...})`, then for the non-created branch, compare each
field that legitimately changes over the record's lifecycle and only call `.save()` if something
differs.
**When to use:** Any sync command where re-running must not duplicate rows or touch `modified`
timestamps on unchanged data (SYNC-04).
**Example:**
```python
# Source: solsys_code/management/commands/load_telescope_runs.py (existing project pattern)
event, created = CalendarEvent.objects.get_or_create(
    url=url,
    defaults={
        'title': title,
        'description': description,
        'start_time': start_time,
        'end_time': end_time,
        'telescope': telescope,
        'instrument': instrument,
        'proposal': proposal,
    },
)
if created:
    created_count += 1
else:
    changed = (
        event.title != title
        or event.description != description
        or event.start_time != start_time
        or event.end_time != end_time
        or event.telescope != telescope
        or event.instrument != instrument
        or event.proposal != proposal
    )
    if changed:
        event.title = title
        event.description = description
        event.start_time = start_time
        event.end_time = end_time
        event.telescope = telescope
        event.instrument = instrument
        event.proposal = proposal
        event.save()
        updated_count += 1
    else:
        unchanged_count += 1
```

### Pattern 2: ORM-level JSONField key lookup (verified working on this stack)
**What:** Django's `__<key>` lookup on a `JSONField` compiles to `JSON_EXTRACT`/`JSON_TYPE` SQL on
SQLite.
**When to use:** SELECT-01's `--proposal` filter.
**Example:**
```python
# Source: live-verified in this repo via `python manage.py shell` (2026-06-17)
from tom_observations.models import ObservationRecord

records = ObservationRecord.objects.filter(
    facility='LCO',
    parameters__proposal=proposal_code,
)
# Confirmed SQL generated (Django 5.2.15 / SQLite 3.34.1):
#   ... WHERE ("facility" = LCO AND
#       (CASE WHEN JSON_TYPE("parameters", $."proposal") IN ('null','false','true')
#             THEN JSON_TYPE("parameters", $."proposal")
#             ELSE JSON_EXTRACT("parameters", $."proposal") END)
#       = JSON_EXTRACT("<code>", '$'))
```

### Pattern 3: Parsing JSONField string datetimes vs native DateTimeField datetimes
**What:** `parameters['start']`/`['end']` are plain strings inside the JSON blob (confirmed via
the official `tom_observations.tests.factories.ObservingRecordFactory` fixture data AND a live
round-trip create/refresh_from_db in this repo: `'2026-07-01T00:00:00'`, type `str`).
`scheduled_start`/`scheduled_end` are real `DateTimeField` columns and come back as native,
timezone-aware `datetime` objects.
**When to use:** Building `start_time`/`end_time` for `CalendarEvent` — the two source fields need
different handling.
**Example:**
```python
# Source: live-verified in this repo, ObservationRecord round trip (2026-06-17)
from datetime import datetime

if record.scheduled_start is None:
    start_time = datetime.fromisoformat(record.parameters['start'])
    end_time = datetime.fromisoformat(record.parameters['end'])
    # NOTE: confirm timezone-awareness — see Common Pitfalls. USE_TZ is True in this
    # project (Django default with TOM Toolkit); fromisoformat() on a naive string like
    # '2026-07-01T00:00:00' produces a naive datetime, which Django will reject/warn on
    # save into a DateTimeField when USE_TZ=True. Localize/attach UTC explicitly.
else:
    start_time = record.scheduled_start
    end_time = record.scheduled_end
```

### Anti-Patterns to Avoid
- **Re-deriving the portal URL by string formatting** (e.g. hardcoding
  `f'https://observe.lco.global/requestgroups/{id}/'`): this is the exact mistake CONTEXT.md's D-01
  already corrected — the real helper produces `/requests/<id>` (no trailing slash, not
  `/requestgroups/`). Always call `LCOFacility().get_observation_url(observation_id)`.
- **Treating `get_terminal_observing_states()` as a 4-element list**: it returns 5 elements
  (`COMPLETED` plus the 4 failure states). Filtering "is this terminal" and "which prefix to use"
  are two different questions — don't conflate them in one lookup table without an explicit
  branch for the successful-terminal case.
- **Calling `.save()` unconditionally in the upsert's update branch**: defeats SYNC-04's no-churn
  requirement and will fail the test analog to
  `test_unchanged_rerun_does_not_update_existing_rows` from Phase 3.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Portal URL construction | Custom f-string URL builder | `LCOFacility().get_observation_url(observation_id)` | Confirmed live: returns `urljoin(portal_url, f'/requests/{id}')`; tracks library's own format if it changes; the literal format in REQUIREMENTS.md SYNC-01 is stale/wrong (404s on real portal per D-01) |
| Terminal-state classification | Custom list of status strings | `LCOFacility().get_terminal_observing_states()` / `.get_failed_observing_states()` | Confirmed live; `get_failed_observing_states()` returns exactly the 4 TERM-01 states (`WINDOW_EXPIRED`, `CANCELED`, `FAILURE_LIMIT_REACHED`, `NOT_ATTEMPTED`) — use this method directly instead of re-typing the list, so a future library update to the failure-state set is picked up automatically |
| Proposal-code filtering | Manual Python loop over all `ObservationRecord(facility='LCO')` rows parsing JSON per-row | `ObservationRecord.objects.filter(facility='LCO', parameters__proposal=code)` | Confirmed working ORM-level JSON lookup on this exact SQLite/Django combination — no need for Python-side filtering fallback |

**Key insight:** Every "build vs. use library helper" decision in this phase has a verified,
working library method already available on the installed `tomtoolkit` version — there is no case
in this phase where hand-rolling is justified by a library gap.

## Common Pitfalls

### Pitfall 1: `get_terminal_observing_states()` returns 5 states, not 4 — `COMPLETED` is unhandled by TERM-01
**What goes wrong:** A naive 1:1 mapping from `get_terminal_observing_states()` to TERM-01's 3
prefixes will either crash (`KeyError`) or silently mislabel successfully-completed observations
when the status is `COMPLETED`.
**Why it happens:** `OCSSettings.get_terminal_observing_states()` is defined as
`get_successful_observing_states() + get_failed_observing_states()`, i.e.
`['COMPLETED'] + ['WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED']`.
TERM-01 in REQUIREMENTS.md and CONTEXT.md's D-04 only enumerate the 4 failure states with prefixes
— `COMPLETED` was apparently assumed out of scope for "terminal" by the discussion, but the library
disagrees.
**How to avoid:** The planner should either (a) use `LCOFacility().get_failed_observing_states()`
(verified to return exactly the 4 TERM-01 states) as the trigger for the prefix logic, leaving
`COMPLETED` records with a clean/no-prefix title (or some other explicit, documented choice), or
(b) explicitly add a `COMPLETED` branch (e.g. no prefix, or a `[DONE]` prefix) to D-04's mapping
table. Either is a one-line decision but it must be made explicitly — don't let it fall through
default/else logic unexamined.
**Warning signs:** A test asserting `event.title` for a `COMPLETED`-status fixture record will
either be missing from the test suite (silent gap) or will fail if the planner assumed only 4
terminal states exist.

### Pitfall 2: `instrument_type` does not contain `'FTS'` or any MPC site identifier — site code is a separate `parameters['site']` key
**What goes wrong:** CONTEXT.md's open question #1 asks whether `'MUSCAT'` appears in
`instrument_type` — confirmed yes (`'2M0-SCICAM-MUSCAT'`, used as the LCO form code for MuSCAT3
*and* MuSCAT4, since both telescopes use the identical 4-channel imager). But `instrument_type`
alone cannot disambiguate FTS (Siding Spring) from FTN (Haleakala) — both use the same
`instrument_type` string. The telescope identity comes from a *separate* `parameters['site']` key
(LCO's 3-letter site codes: `coj` = Siding Spring/FTS, `ogg` = Haleakala/FTN — `[ASSUMED]`, web
search only, not yet confirmed against a real or fixture record with a `'site'` key populated).
**Why it happens:** LCO's request-submission form (`LCOMuscatImagingObservationForm` /
`LCOFullObservationForm._build_location()`) stores `site` as its own `cleaned_data`/parameters
key, populated from a `ChoiceField`, independent of `instrument_type`.
**How to avoid:** Build the telescope-label map keyed on `parameters.get('site')`, not on substring
matching against `instrument_type`. Confirm the exact key name (`'site'`, confirmed present in
`LCOFullObservationForm._build_location()` source) and exact code values (`'coj'`/`'ogg'`, web-search
only) against a fixture or — if available before planning concludes — a real `ObservationRecord`
from this LCO proposal, before locking the map's values into a plan. This is exactly the
uncertainty CONTEXT.md flagged in open question #1; this research narrows it from "which field" (now
confirmed: `instrument_type` for instrument, `site` for telescope) to "confirm two specific code
string values" (`coj`, `ogg`) which remains `[ASSUMED]`.
**Warning signs:** If a real fixture record's `parameters` dict has no `'site'` key at all (e.g. if
this project's submission form never set it, or it lives under a different param name for the
specific form variant used), the map will need a different key — verify against either real
`ObservationRecord.parameters` data pulled from this project's LCO proposal or by re-reading the
exact submission-form class actually used (likely `LCOMuscatImagingObservationForm` if MuSCAT4 is
explicitly targeted, though SELECT-01's broader `--proposal`-only filter means any LCO instrument
type could appear).

### Pitfall 3: `parameters['start']`/`['end']` parse to naive datetimes; Django with `USE_TZ=True` requires aware datetimes
**What goes wrong:** `datetime.fromisoformat('2026-07-01T00:00:00')` produces a naive `datetime`
(no `tzinfo`). Assigning a naive datetime to a `DateTimeField` when `settings.USE_TZ = True`
triggers a `RuntimeWarning` (and, depending on Django version/strictness, can store the wrong
absolute time if the DB or Django assumes a different default timezone than intended).
**Why it happens:** ISO strings without a `Z`/offset suffix are inherently timezone-naive; the LCO
API/portal's convention (confirmed via the official factory fixture
`'start': '2020-01-01T00:00:00'`) does not embed a UTC offset in this field.
**How to avoid:** Explicitly attach UTC (LCO's request-submission times are documented/conventionally
UTC) using `datetime.fromisoformat(s).replace(tzinfo=dt_timezone.utc)` before assigning to
`start_time`/`end_time` — mirrors the existing `.replace(microsecond=0)` UTC-attachment precedent
already used in `load_telescope_runs.py` (`sunset.to_datetime(timezone=dt_timezone.utc)`).
**Warning signs:** Django test runner emitting `RuntimeWarning: DateTimeField ... received a naive
datetime` during `./manage.py test solsys_code` — treat any such warning as a correctness bug, not
noise, since it indicates `USE_TZ` is active and the value may be silently mis-localized.

### Pitfall 4: `JSONField.__key` ORM lookups are correct here, but don't assume this generalizes to all SQLite versions
**What goes wrong:** Relying on `parameters__proposal=code` without verifying it on the *actual*
target deployment SQLite version could silently break on older SQLite builds lacking JSON1
extension support.
**Why it happens:** Django's JSON lookups on SQLite depend on the SQLite library being compiled
with the JSON1 extension (standard in modern SQLite, but not universal on ancient builds).
**How to avoid:** Already verified empirically in this repo: SQLite `3.34.1` (well above the
JSON1-mandatory threshold) + Django `5.2.15` produces correct `JSON_EXTRACT`/`JSON_TYPE` SQL and a
correct, non-empty result set when matching records exist. No further mitigation needed for this
project's environment; just don't assume this is portable to an arbitrary SQLite without
re-verifying, if this command is ever deployed elsewhere.
**Warning signs:** None expected in this repo's CI (same SQLite/Django stack); flag if CI ever runs
on a different OS image with an older system SQLite.

## Code Examples

### Confirmed live in this repo (2026-06-17, `python manage.py shell`)
```python
# Source: live verification, not Context7/docs (no official docs page covers this combination)
from tom_observations.facilities.lco import LCOFacility

fac = LCOFacility()  # no network call, no credentials required at __init__
fac.get_observation_url('123456')
# -> 'https://observe.lco.global/requests/123456'

fac.get_terminal_observing_states()
# -> ['COMPLETED', 'WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED']

fac.get_failed_observing_states()
# -> ['WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED']
```

```python
# Source: live verification — full ObservationRecord round trip
from tom_observations.models import ObservationRecord

rec = ObservationRecord.objects.create(
    target=some_target, facility='LCO', observation_id='999999', status='PENDING',
    parameters={
        'proposal': 'TESTCODE123',
        'start': '2026-07-01T00:00:00',
        'end': '2026-07-02T00:00:00',
        'instrument_type': '2M0-SCICAM-MUSCAT',
        'site': 'coj',
    },
)
rec.refresh_from_db()
type(rec.parameters['start'])  # -> <class 'str'>  (NOT datetime)
rec.terminal                   # -> False  (status='PENDING')
rec.status = 'WINDOW_EXPIRED'; rec.save()
rec.terminal                   # -> True

ObservationRecord.objects.filter(
    facility='LCO', parameters__proposal='TESTCODE123'
).count()  # -> 1 (ORM-level JSON lookup confirmed working)
```

### Official factory fixture (confirms parameter shapes — installed `tomtoolkit` package source)
```python
# Source: /home/tlister/venv/fomo311_venv/.../tom_observations/tests/factories.py (installed package)
class ObservingRecordFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = ObservationRecord
    target = factory.RelatedFactory(SiderealTargetFactory)
    facility = 'LCO'
    observation_id = factory.Faker('pydecimal', right_digits=0, left_digits=7)
    status = 'PENDING'
    parameters = {
        'facility': 'LCO', 'target_id': 1, 'observation_type': 'IMAGING', 'name': 'With Perms',
        'ipp_value': 1.05, 'start': '2020-01-01T00:00:00', 'end': '2020-01-02T00:00:00',
        'exposure_count': 1, 'exposure_time': 2.0, 'max_airmass': 4.0,
        'observation_mode': 'NORMAL', 'proposal': 'LCOSchedulerTest', 'filter': 'I',
        'instrument_type': '1M0-SCICAM-SINISTRO',
    }
```
This is a useful test-fixture template (it doesn't use MuSCAT/FTS, but proves the field-shape
contract: flat dict, ISO string times, `instrument_type` as a flat key). Phase 4 test fixtures
should follow this same shape but with `'instrument_type': '2M0-SCICAM-MUSCAT'` and a `'site'` key
added (the official factory predates/doesn't need a `'site'` key since its instrument isn't
site-ambiguous).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Literal `https://observe.lco.global/requestgroups/<id>/` URL format (as written in REQUIREMENTS.md SYNC-01) | `LCOFacility().get_observation_url(id)` → `/requests/<id>` | Confirmed against installed `tomtoolkit==3.0.0a9`; superseded format predates this | Hardcoding the old format would 404 on the real LCO portal — D-01 already corrects this, research confirms the correction is accurate |
| `parameters` assumed `TextField` (STATE.md's stale "Key Technical Notes") | `parameters` is `JSONField` (Django native) | Confirmed via installed `tom_observations/models.py` source and live ORM query | Enables direct ORM-level `parameters__proposal=` lookups without any JSON parsing/Python-side filtering workaround |

**Deprecated/outdated:**
- STATE.md's Phase 4 "Key Technical Notes" claim that `parameters` is a `TextField` requiring
  Python-side JSON parsing is **incorrect** and superseded by CONTEXT.md's correction (and now this
  research's live verification). The planner must use CONTEXT.md/this RESEARCH.md as authoritative
  over STATE.md for this specific fact.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | LCO 3-letter site code `'coj'` = Siding Spring = FTS, `'ogg'` = Haleakala = FTN | Common Pitfalls (Pitfall 2), Architecture Patterns | If wrong, the static telescope-label map produces incorrect `telescope` values on `CalendarEvent` (SYNC-05 partially fails) — low blast radius (cosmetic, one field), easy to fix once a real fixture/record is inspected, but should be confirmed before locking the map into a plan |
| A2 | `parameters['site']` is the key actually populated for FTS/MuSCAT4 submissions in this project's real LCO proposal data (vs. some other key name, or absent entirely if a different submission-form variant was used) | Common Pitfalls (Pitfall 2) | If the key is absent or differently named, the telescope-label derivation needs a fallback (e.g. parse from `instrument_type` plus a documented "MUSCAT means FTS for this project, since only one MuSCAT site is used here" simplifying assumption) — moderate risk since SYNC-05 explicitly requires `telescope` to be populated |
| A3 | LCO request-submission `start`/`end` times are conventionally UTC even though the stored string carries no offset suffix | Common Pitfalls (Pitfall 3), Code Examples | If LCO actually means local-site time (unlikely per general LCO API convention, but unverified against official LCO API docs in this session), `CalendarEvent.start_time`/`end_time` for unscheduled records would be off by the site's UTC offset (similar magnitude of error to a timezone bug in Phase 1) |

**Confirmation path for A1/A2 before/during planning:** if any real `ObservationRecord` rows exist
in this project's database from prior LCO submissions for the target proposal, inspect
`ObservationRecord.objects.filter(facility='LCO').values_list('parameters', flat=True)` directly —
this resolves A1/A2 with HIGH confidence in seconds and should be the first thing the planner (or
an execute-phase task) does, before committing to the static map's values.

## Open Questions

1. **What title prefix (if any) applies to `COMPLETED` (successful terminal) records?**
   - What we know: `get_terminal_observing_states()` includes `COMPLETED` alongside the 4 failure
     states; `get_failed_observing_states()` cleanly isolates just the 4 TERM-01 states.
   - What's unclear: TERM-01/D-04 only specify prefixes for the failure states. No decision was
     made for `COMPLETED`.
   - Recommendation: planner should add an explicit branch — likely "no prefix, clean title" (since
     a successfully completed observation isn't a failure/audit-trail case the way the other 4 are)
     — and add a test case asserting this, rather than leaving it to fall through undefined
     `if`/`elif` logic.

2. **Exact key name and values for the site code in `parameters` for this project's real FTS/MuSCAT4 submissions.**
   - What we know: the LCO form-layer source confirms a `'site'` key exists in `cleaned_data`/
     `parameters` for site-selectable instruments, and `'coj'`/`'ogg'` are documented LCO 3-letter
     site codes for Siding Spring/Haleakala (web search, not official LCO API docs).
   - What's unclear: whether this project's actual stored `ObservationRecord.parameters` (from
     real or planned LCO submissions) populates `'site'` with exactly `'coj'`, or potentially
     omits it if a different/simpler submission form variant was used that doesn't expose site
     choice (e.g. if only one MuSCAT site is ever targeted and the form defaults it).
   - Recommendation: inspect real `ObservationRecord` rows in this project's DB if any exist for
     the target proposal before finalizing the map (see Assumptions Log confirmation path above);
     otherwise build test fixtures explicitly setting `'site': 'coj'` and document the assumption
     inline in the command's site-map constant with a comment pointing back to this research.

## Environment Availability

Skipped — this phase has no external service/tool dependencies. `LCOFacility()` instantiation,
`ObservationRecord`/`CalendarEvent` ORM access, and `manage.py test` all run fully in-process
against the local SQLite test database, with zero network calls (confirmed: `LCOFacility.__init__`
makes no HTTP request; `get_observation_url()` and `get_terminal_observing_states()` are pure
string/list operations).

## Validation Architecture

Skipped — `.planning/config.json` sets `workflow.nyquist_validation: false`.

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1` per `.planning/config.json`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | This is a `manage.py` CLI command, not a web-facing endpoint; runs with whatever OS/shell privileges invoke it, same trust boundary as any other management command in this codebase |
| V3 Session Management | No | No session/request context involved |
| V4 Access Control | No | No multi-user access boundary inside the command; if the command is later wrapped by a scheduled task or web trigger, access control belongs to that wrapper, not this phase |
| V5 Input Validation | Yes | `--proposal <code>` is a free-text CLI argument used directly in an ORM filter (`parameters__proposal=code`). Django's ORM parameterizes JSONField lookups (confirmed via the generated SQL in Code Examples — the value is bound, not string-interpolated), so this is **not** vulnerable to SQL injection via the ORM. No additional sanitization library needed; do not hand-roll string escaping for this value |
| V6 Cryptography | No | No secrets, tokens, or cryptographic operations in this phase's scope |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| SQL/JSON-path injection via `--proposal` argument | Tampering | Already mitigated by Django ORM parameterization of `JSONField` key lookups — confirmed the generated SQL binds the value rather than interpolating it (see Code Examples' `JSON_EXTRACT("<code>", '$')` pattern, where `<code>` is a bound parameter, not raw string concatenation). No additional validation library required beyond what Django already provides |
| Unbounded/malformed proposal code causing command to silently match zero rows and report false success | Repudiation (audit-trail ambiguity) | Mirror `load_telescope_runs.py`'s precedent: report a clear stdout summary count (`created: 0, updated: 0, ...`) so a no-op run is visible to the operator, rather than exiting silently with no output |

## Sources

### Primary (HIGH confidence)
- Installed `tomtoolkit==3.0.0a9` package source, read directly from
  `/home/tlister/venv/fomo311_venv/lib64/python3.11/site-packages/tom_observations/` and
  `/tom_calendar/` — `models.py`, `facilities/ocs.py`, `facilities/lco.py`, `facility.py`,
  `tests/factories.py`
- Live verification via `python manage.py shell` in this repo against the actual project SQLite DB
  (2026-06-17): `LCOFacility()` instantiation, `get_observation_url()`, `get_terminal_observing_states()`,
  `get_failed_observing_states()`, full `ObservationRecord` create/refresh_from_db round trip,
  `parameters__proposal=` ORM filter and generated SQL
- This repo's existing files: `solsys_code/management/commands/load_telescope_runs.py`,
  `solsys_code/tests/test_load_telescope_runs.py`, `solsys_code/telescope_runs.py`,
  `src/fomo/settings.py` (`FACILITIES['LCO']['portal_url']`), `.planning/config.json`

### Secondary (MEDIUM confidence)
- `docs/design/telescope_runs_calendar.rst` — original Stage 3/4 design notes (superseded in scope
  by REQUIREMENTS.md per CONTEXT.md's note, but useful background)

### Tertiary (LOW confidence)
- WebSearch: LCO 3-letter site codes (`coj` = Siding Spring/FTS, `ogg` = Haleakala/FTN) and
  `2M0-SCICAM-MUSCAT` as the shared MuSCAT3/MuSCAT4 `instrument_type` — not cross-checked against
  an official LCO API reference page or Context7 in this session; flagged `[ASSUMED]` (see
  Assumptions Log A1/A2)

## Metadata

**Confidence breakdown:**
- Standard stack / library API surface: HIGH — every fact about `ObservationRecord`,
  `LCOFacility`/`OCSFacility`, and `CalendarEvent` was verified by reading the actually-installed
  package source and/or running live code against this repo's real database, not by trusting
  training-data assumptions or the discussion notes alone
- Architecture/upsert pattern: HIGH — directly mirrors an existing, tested, merged pattern in this
  same codebase (`load_telescope_runs.py`)
- Site-code → telescope-label values (A1/A2): LOW — web-search only, not yet confirmed against
  real project data or official LCO docs; flagged explicitly for planner/execute-phase follow-up
- Pitfalls: HIGH — Pitfall 1 (5 vs 4 terminal states) and Pitfall 3 (naive datetime) are both
  reproduced/confirmed by direct code execution in this session, not inferred

**Research date:** 2026-06-17
**Valid until:** 30 days (stable third-party library already pinned in this project; the only
fast-moving unknowns are A1/A2, which depend on this project's own LCO proposal data, not external
library churn)
