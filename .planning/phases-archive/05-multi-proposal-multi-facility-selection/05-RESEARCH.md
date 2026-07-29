# Phase 5: Multi-Proposal & Multi-Facility Selection - Research

**Researched:** 2026-06-19
**Domain:** Django management command generalization (query filtering + per-record facility dispatch), `tom_observations` / TOM Toolkit OCS facility classes
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Multi-proposal / ALL syntax**
- **D-01:** `--proposal` accepts a comma-separated list (e.g. `--proposal A,B,C`). Each segment is
  whitespace-stripped but **not** case-normalized — proposal codes are case-sensitive (e.g.
  `LTP2025A-004`), so `' B '` becomes `'B'` but casing is preserved as typed.
- **D-02:** The special `ALL` token is matched **case-insensitively** (`all`/`All`/`ALL` all trigger
  sync-everything) — no real proposal code could collide with the word "all", and
  case-insensitivity is friendlier for command-line operators.
- **D-03:** Duplicate or empty entries in the list (`--proposal A,A,B` or a trailing comma
  `--proposal A,B,`) are **silently deduped and dropped** — split, strip, drop empty strings,
  dedupe — rather than erroring out.

**SOAR facility settings & credentials**
- **D-04:** This phase adds a `FACILITIES['SOAR']` entry to `src/fomo/settings.py` as part of the
  code change — SOAR sync cannot actually work end-to-end without it (today `SOARSettings`
  silently resolves to blank `portal_url`/`api_key` since the key is absent), and SELECT-04's "both
  facilities synced in one run" success criterion implicitly requires real, resolvable SOAR
  credentials.
- **D-05:** `FACILITIES['SOAR']` reuses the **same** `api_key`/`portal_url` env var as
  `FACILITIES['LCO']` (e.g. both read `os.getenv('LCO_API_KEY')`) rather than introducing a separate
  `SOAR_API_KEY` — matches `SOARFacility`'s documented behavior that it authenticates against the
  same LCO Observation Portal API, not a distinct one.

**Per-record facility dispatch & defensive behavior**
- **D-06:** Replace the current single shared `LCOFacility()` instance (created once at
  `sync_lco_observation_calendar.py:187`, reused for every record regardless of that record's own
  `facility` field — the exact bug SELECT-05 targets) with a dispatch dict `{'LCO': LCOFacility(),
  'SOAR': SOARFacility()}` built **once, eagerly**, before the queryset loop, then looked up per
  record via `record.facility`. Build both instances unconditionally regardless of which facilities
  actually appear in a given run's records — simplest, mirrors today's single-eager-instance
  pattern just extended to two keys.
- **D-07:** If a record's `facility` value is something other than `'LCO'` or `'SOAR'` (shouldn't
  occur since the queryset filters on exactly those two, but defensively): **skip + log, continue
  the run** — same per-record error-handling convention already used elsewhere in this command and
  established in Phase 3's D-02. A single unexpected row never aborts the whole sync.

**Run summary reporting**
- **D-08:** The end-of-run summary line reports a **per-facility breakdown** (e.g. `LCO: 3 created,
  1 updated, 0 skipped | SOAR: 2 created, 0 updated, 1 skipped`), not just aggregate counts — lets
  an operator see at a glance whether one facility (e.g. SOAR, if credentials are misconfigured) had
  problems, without re-running filtered to a single facility.

### Claude's Discretion
- Exact stdout formatting/line layout of the per-facility summary (D-08), as long as both
  facilities' created/updated/unchanged/skipped counts are each individually visible.
- Exact log message wording for the D-07 skip-on-unexpected-facility case and the D-04
  settings-key addition's surrounding comments/docstring.
- Whether the proposal-list parsing (D-01..D-03) is implemented as a small helper function or
  inlined in `add_arguments`/`handle()` — either is fine as long as the behavior matches D-01..D-03.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. Instrument-type extraction correctness (Phase 6) and
telescope-label resolution/fallback (Phase 7) were not raised as in-scope here; they're already
sequenced as later phases per PROJECT.md's Key Decisions ("Phase ordering follows research's
dependency chain").
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|---------------------|
| SELECT-02 | `--proposal` accepts a comma-separated list of proposal codes (e.g. `--proposal A,B,C`), syncing records matching any of them | Confirmed live this session: `parameters__proposal__in=[...]` ORM filter produces zero substring/partial-match leakage against this repo's real SQLite database. See Architecture Pattern 2 and Code Examples. |
| SELECT-03 | `--proposal ALL` syncs every LCO-family `ObservationRecord` regardless of proposal code | Conditional-filter pattern (Pattern 2): proposal clause is omitted entirely when the parsed value is the `ALL` sentinel, confirmed via live `.filter(facility__in=['LCO','SOAR'])` with no proposal clause matching all 5 fixture records this session. |
| SELECT-04 | Sync covers both `facility='LCO'` and `facility='SOAR'` records in a single run | `facility__in=['LCO','SOAR']` base filter (Pattern 2) + per-record dispatch dict (Pattern 3); requires extending `_create_record()` test helper per Pitfall 4. |
| SELECT-05 | Each record is processed using the facility instance/credentials matching its own `facility` value, never a single shared instance reused across both | Dispatch dict pattern (Pattern 3), grounded in confirmed `SOARFacility(LCOFacility)` inheritance (identical `get_observation_url`/`get_terminal_observing_states`/`get_failed_observing_states`). Pitfall 3 flags that a black-box URL/title assertion cannot discriminate this bug — the test must spy/assert on which instance was actually used. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

These directives apply to any planning/implementation work in this phase and must not be
contradicted by the plan:

- **GSD workflow enforcement:** No direct file edits outside a GSD command — this phase's
  implementation must go through `/gsd:execute-phase`, not ad-hoc `Edit`/`Write` calls.
- **Test factory:** Any new `Target` test fixture must use
  `tom_targets.tests.factories.NonSiderealTargetFactory`, never `SiderealTargetFactory` — already
  followed by the existing `setUpTestData()` in `test_sync_lco_observation_calendar.py`; new SOAR
  fixtures must continue this pattern (no new `Target` fixtures are actually needed this phase —
  the existing class-level `cls.target` is reused for all new `ObservationRecord` fixtures, per
  existing convention).
- **Plain-English planning-doc terminology:** Write "create or update" / "find-or-create" instead
  of "upsert" in CONTEXT.md/RESEARCH.md/PLAN.md/PATTERNS.md and other `.planning/` artifacts. This
  research document follows that convention; the planner must too.
- **Lint/format:** `ruff check .` and `ruff format --check .` must stay clean — single quotes,
  120-column line length. Both existing target files already pass cleanly (confirmed this session).
- **Docstrings:** Google-style, with `Args:`/`Returns:`/`Raises:` sections — the existing helper
  functions in `sync_lco_observation_calendar.py` already follow this; any new helper (e.g. a
  proposal-list-parsing function) must match.
- **Test location:** DB-dependent tests (this phase's tests all are, since they create
  `ObservationRecord`/`CalendarEvent` rows) belong in `solsys_code/tests/`, run via `./manage.py
  test solsys_code` — not the separate `pytest` suite (`tests/`, `src/`, `docs/`), which does not
  collect Django app tests.
- **Logging convention:** `logger = logging.getLogger(__name__)`, debug-level for expected
  failures, f-strings for messages — though note `sync_lco_observation_calendar.py` currently uses
  `self.stderr.write(...)` directly rather than a `logger`, matching Django management-command
  convention; this phase should continue that existing pattern rather than introducing a new
  `logger` call style inconsistent with the rest of the file.

## Summary

This phase generalizes the existing `sync_lco_observation_calendar` command (Phase 4) from a
single-proposal, LCO-only sync into a multi-proposal, multi-facility (LCO + SOAR) sync. All the
work is pure query/dispatch logic on top of code that already exists and already works — no new
models, no new external packages, no new I/O. Every locked decision in CONTEXT.md (D-01..D-08) has
now been grounded against the actual installed library source and live ORM probes run against this
repo's database during this research session, not against training-data assumptions.

The two load-bearing facts confirmed this session: (1) `SOARFacility` is a **direct, unmodified
subclass of `LCOFacility`** (`tom_observations/facilities/soar.py:234`) — `get_observation_url()`,
`get_terminal_observing_states()`, and `get_failed_observing_states()` are inherited unchanged from
`OCSFacility`, so every existing helper in `sync_lco_observation_calendar.py` works against a
`SOARFacility` instance with zero signature changes, exactly as D-06 assumes. (2)
`ObservationRecord.parameters` is a Django `JSONField` (not `TextField` — STATE.md's "Key Technical
Notes" is stale on this point, already corrected once during Phase 4), and `field__in=[...]`
combined with `parameters__proposal__in=[...]` was verified live against this repo's SQLite database
this session: it compiles to parameterized `JSON_EXTRACT`/`IN (...)` SQL and produces **zero
substring/partial-match leakage** (a record with `proposal='AB'` does not match a filter list of
`['A','B','C']`) — directly satisfying Success Criterion 1's explicit anti-requirement.

One nuance refines D-04/D-05: `SOARSettings` extends `LCOSettings`, whose `default_settings` dict
already hardcodes `portal_url: 'https://observe.lco.global'` as a *class-level* default (not from
`FACILITIES`). This means `SOARFacility().get_observation_url(...)` already returns the correct URL
**even without** a `FACILITIES['SOAR']` entry — confirmed live this session. The `FACILITIES['SOAR']`
entry D-04 adds is still necessary, but specifically for `api_key` (which has no class-level
non-blank default in any tier) — not for `portal_url`, which would silently work by accident either
way. This doesn't change D-04's outcome (the entry should still be added) but matters for how the
planner frames the verification task: test what changes (api_key has a real value) not what doesn't
(portal_url was never broken).

**Primary recommendation:** Replace the single hardcoded `LCOFacility()` instance and single-proposal
filter with (1) a small proposal-list-parsing helper (D-01..D-03), (2) a `field__in`-based queryset
filter that conditionally omits the proposal clause for `ALL` (D-02/D-03), and (3) an eagerly-built
`{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dispatch dict looked up per-record by
`record.facility` (D-06/D-07), threaded into the existing `_build_event_fields`/`_title_for`/
`_failure_prefix` helpers with no signature changes. Add `FACILITIES['SOAR']` to settings.py (D-04)
for `api_key` correctness, using the same env var the LCO entry should read from — noting that today
neither entry actually reads from an env var (see Pitfall 1).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `--proposal` argument parsing (comma-list/ALL/dedup) | API / Backend (mgmt command) | — | Pure CLI input parsing, no I/O; belongs in `add_arguments`/`handle()` exactly as today |
| Record selection (facility + proposal filter) | Database / ORM | — | `ObservationRecord.objects.filter(facility__in=..., parameters__proposal__in=...)` — verified ORM-level, no app-level Python filtering needed |
| Per-record facility instance dispatch | API / Backend (mgmt command) | — | In-process object dispatch (`dict[record.facility]`), no network call at `LCOFacility()`/`SOARFacility()` construction time |
| Facility credentials resolution | Database / Storage (settings.py / env) | — | `FACILITIES['SOAR']` lookup happens inside `OCSSettings.get_setting()`, reading Django settings — config tier, not code tier |
| CalendarEvent create/update (no-churn) | Database / ORM | — | Existing `get_or_create` + conditional `save()` pattern, untouched by this phase |
| Per-facility summary reporting | API / Backend (mgmt command) | — | stdout formatting, in-process counters only |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django ORM | 5.2.15 [VERIFIED: installed `python3 -c "import django; print(django.VERSION)"`] | `field__in` queryset filtering on `JSONField` keys | Already the project's ORM; no alternative needed |
| `tom_observations.facilities.lco.LCOFacility` | tomtoolkit 3.0.0a9 [VERIFIED: `pip show tomtoolkit`, already used in Phase 4] | Existing LCO facility wrapper; supplies `get_observation_url()`, `get_terminal_observing_states()`, `get_failed_observing_states()` | Already a project dependency, used unchanged since Phase 4 |
| `tom_observations.facilities.soar.SOARFacility` | tomtoolkit 3.0.0a9 [VERIFIED: read installed source at `tom_observations/facilities/soar.py:234`] | Direct `LCOFacility` subclass for SOAR records; this phase's only new import | Already listed in `TOM_FACILITY_CLASSES` (`src/fomo/settings.py`); no new package install required |

No new packages are installed by this phase — `tom_observations` (the package providing both
`LCOFacility` and `SOARFacility`) is already a transitive dependency of `tomtoolkit`, already
imported in the file this phase modifies. **The Package Legitimacy Audit section below is therefore
N/A** — no `pip install` is introduced.

### Supporting
None — no new libraries needed beyond what Phase 4 already imports.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `field__in=[...]` ORM filter (D-06's integration point, already specified) | Python-side `[r for r in qs if r.facility in {...} and r.parameters.get('proposal') in {...}]` | ORM-level filter is confirmed correct and pulls fewer rows from the DB; Python-side filtering was only ever a fallback for JSON-lookup-unsupported SQLite builds, which this repo's build does not need (per Phase 4 research, re-confirmed this session) |
| Eager `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dict (D-06, locked) | Lazy dict built via `defaultdict`/`functools.lru_cache` keyed on facility name as records are encountered | D-06 explicitly locks "build both eagerly, unconditionally" for simplicity; rejecting lazy construction is intentional, not an oversight — do not introduce it |

**Installation:** None required — no new packages.

## Package Legitimacy Audit

**Not applicable to this phase.** No new external packages are installed. `tom_observations` (the
module providing `SOARFacility`) is already an installed transitive dependency of `tomtoolkit`,
already imported elsewhere in this codebase (`solsys_code/management/commands/
sync_lco_observation_calendar.py:7`, `solsys_code/tests/test_sync_lco_observation_calendar.py:9`).
No `npm view`/`pip index versions`/registry legitimacy check is needed for an import path that
already exists in the working tree and is exercised by passing tests today.

## Architecture Patterns

### System Architecture Diagram

```text
CLI invocation
  ./manage.py sync_lco_observation_calendar --proposal A,B,C   (or --proposal ALL)
        │
        ▼
  add_arguments() / handle()
        │  parse --proposal into either:
        │    - a deduped, stripped, case-preserved list of codes   [D-01, D-03]
        │    - the ALL sentinel (matched case-insensitively)        [D-02]
        ▼
  Build facility dispatch dict (once, eager, before the loop)       [D-06]
        facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}
        ▼
  Build queryset filter                                             [D-01..D-03, SELECT-02/03]
        ObservationRecord.objects.filter(facility__in=['LCO','SOAR'])
          + .filter(parameters__proposal__in=[codes])  -- omitted entirely when ALL
        ▼
  For each record in queryset:
        │
        ├─ facility = facilities.get(record.facility)
        │     │
        │     ├─ found ('LCO' or 'SOAR') ─────────────┐
        │     └─ NOT found (unexpected value) ──┐     │
        │                                        ▼     ▼
        │                                  log + skip   _build_event_fields(record, facility)
        │                                  [D-07]              │  (unchanged helper signature,
        │                                  continue            │   just a different facility arg)
        │                                                       ▼
        │                                          get_or_create(url=...) / conditional save()
        │                                          (no-churn upsert, unchanged from Phase 4)
        │                                                       │
        │                                          increment per-FACILITY counters [D-08]
        ▼
  Print per-facility summary line                                   [D-08]
        'LCO: 3 created, 1 updated, 0 skipped | SOAR: 2 created, 0 updated, 1 skipped'
```

### Recommended Project Structure
No new files. All changes are in-place edits to two existing files:
```
solsys_code/
├── management/commands/
│   └── sync_lco_observation_calendar.py   # add_arguments, handle(), dispatch dict, per-facility counters
└── tests/
    └── test_sync_lco_observation_calendar.py   # extend _create_record()/_parameters() for SOAR + multi-proposal cases
```

### Pattern 1: Proposal-list parsing helper (D-01/D-02/D-03)
**What:** A small pure function that turns the raw `--proposal` string into either the `ALL`
sentinel or a deduped list of codes.
**When to use:** Called once in `handle()` before building the queryset.
**Example:**
```python
# Source: derived from this repo's existing conventions (no upstream library helper for this --
# argparse/Django give you the raw string, parsing is local logic)
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
Note: dict-as-ordered-set is used here for first-seen-order dedup; `set()` would also satisfy D-03
but loses insertion order, which is purely cosmetic for this use case (summary output doesn't depend
on order) — either is acceptable per CONTEXT.md's "Claude's Discretion" on this point.

### Pattern 2: Conditional queryset filter construction (D-01..D-03, SELECT-02/03)
**What:** Build the base facility filter unconditionally, then add the proposal clause only when a
code list (not `ALL`) was given.
**When to use:** Directly in `handle()`, replacing the current single `.filter(facility='LCO',
parameters__proposal=proposal)` call.
**Example:**
```python
# Source: verified live against this repo's installed Django 5.2.15 / SQLite this session —
# .query inspection confirmed JSON_EXTRACT/IN(...) compilation with no substring leakage.
codes = _parse_proposal_arg(options['proposal'])
records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])
if codes is not None:
    records = records.filter(parameters__proposal__in=codes)
```
**Verified this session:** with fixture records `proposal in {'A', 'AB', 'B', 'C', 'Z'}` across
mixed `facility` values, `.filter(facility__in=['LCO','SOAR'], parameters__proposal__in=['A','B','C'])`
matched exactly `{'A','B','C'}` — the `'AB'` record was correctly excluded, confirming Success
Criterion 1's "no partial/single-character substring matching" requirement holds at the ORM/SQL
level, not just in application code.

### Pattern 3: Per-facility dispatch dict (D-06/D-07)
**What:** Build both facility instances unconditionally before the loop; look each record up by its
own `facility` field; skip-and-log on an unexpected value.
**When to use:** Replaces the single `facility = LCOFacility()` line; threaded into every existing
call site (`_build_event_fields`, `_failure_prefix` via `_title_for`).
**Example:**
```python
# Source: SOARFacility confirmed (this session, reading installed
# tom_observations/facilities/soar.py:234) as `class SOARFacility(LCOFacility)`, name='SOAR',
# no override of get_observation_url/get_terminal_observing_states/get_failed_observing_states.
from tom_observations.facilities.soar import SOARFacility

facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}

for record in records:
    facility = facilities.get(record.facility)
    if facility is None:
        self.stderr.write(
            f'Skipping observation_id={record.observation_id!r}: unrecognized facility {record.facility!r}'
        )
        skipped_by_facility[record.facility] = skipped_by_facility.get(record.facility, 0) + 1
        continue
    try:
        fields = _build_event_fields(record, facility)
    except (KeyError, ValueError) as exc:
        self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
        counters[record.facility]['skipped'] += 1
        continue
    # ... existing get_or_create / conditional save, now incrementing counters[record.facility][...]
```
**Confirmed no instantiation cost:** Both `LCOFacility()` and `SOARFacility()` constructors run
synchronously with zero network calls — confirmed in Phase 4 research and re-confirmed this session
(`SOARFacility()` instantiated successfully with no credentials configured, no exception, no HTTP
call observed).

### Anti-Patterns to Avoid
- **Reusing one shared facility instance across both LCO and SOAR records:** This is the exact bug
  SELECT-05 targets (today's `sync_lco_observation_calendar.py:187,194` instantiates one
  `LCOFacility()` and applies it to every record regardless of `record.facility`). A SOAR record
  processed through `LCOFacility()` would still produce a syntactically valid URL (since
  `LCOSettings.default_settings['portal_url']` is the same string SOAR falls back to), masking the
  bug in casual testing — this is precisely why SELECT-05's acceptance criterion requires a test that
  asserts dispatch-by-`record.facility`, not just a test that the URL looks right.
- **Lazy/conditional facility instantiation ("only build SOARFacility() if a SOAR record is seen"):**
  Explicitly rejected by D-06 in favor of eager unconditional construction — do not introduce this
  even though it's marginally more "efficient"; it adds branching complexity for negligible cost
  (constructing a facility instance does no I/O).
- **String case-folding the proposal codes themselves:** D-01 explicitly locks proposal codes as
  case-sensitive (only the `ALL` sentinel is case-insensitive, per D-02) — do not `.upper()`/`.lower()`
  the codes when building the filter list.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Determining SOAR's terminal/failed observing states | A second hand-typed `_FAILURE_PREFIX_BY_STATUS`-style dict for SOAR | `facility.get_failed_observing_states()` / `facility.get_terminal_observing_states()` (already used in `_failure_prefix`) called on whichever facility instance the dispatch dict returns | Verified this session: `SOARFacility().get_failed_observing_states()` and `.get_terminal_observing_states()` return byte-identical lists to `LCOFacility()`'s (both inherited unchanged from `OCSSettings`/`OCSFacility`) — a parallel SOAR-specific dict would be redundant and a future maintenance trap if the two ever drift |
| Building the SOAR portal URL | A hardcoded `'https://observe.lco.global/requests/{id}'` string for SOAR records | `facility.get_observation_url(record.observation_id)` via the dispatch dict (same call already used for LCO) | `OCSFacility.get_observation_url()` is inherited unchanged by `SOARFacility`; confirmed this session it returns the identical URL shape LCO does (`SOARFacility().get_observation_url('12345')` == `'https://observe.lco.global/requests/12345'`) |
| Multi-value SQL filtering on a JSON key | A Python loop parsing `record.parameters` per row to check list membership | `parameters__proposal__in=[...]` ORM lookup | Confirmed this session: compiles to parameterized `JSON_EXTRACT(...)`/`IN (...)` SQL, executes correctly against this repo's actual SQLite database, with zero substring-match leakage — no reason to drop to Python-side filtering |

**Key insight:** Every "new" capability this phase needs (SOAR state lists, SOAR URL building,
multi-value filtering) already has a library- or ORM-level answer that was directly probed and
confirmed working against this exact codebase during this research session. There is no part of
this phase where hand-rolling is the only option.

## Common Pitfalls

### Pitfall 1: D-05's "same env var as LCO" doesn't exist yet — neither facility's `api_key` is env-var-driven today
**What goes wrong:** D-05 says `FACILITIES['SOAR']` should reuse "the same `api_key`/`portal_url` env
var as `FACILITIES['LCO']` (e.g. both read `os.getenv('LCO_API_KEY')`)". Reading the actual current
`src/fomo/settings.py:215-218` this session: `FACILITIES['LCO']['api_key']` is a **hardcoded literal
empty string `''`**, not `os.getenv('LCO_API_KEY')` — that env var name appears nowhere in this repo
(confirmed via grep across `src/` and `solsys_code/`), and `INTEGRATIONS.md` documents LCO auth only
as "configured in `FACILITIES` dict" with no env var listed.
**Why it happens:** Both `LCOFacility`'s and `SOARFacility`'s class **docstrings** (in the installed
`tom_observations` library, not this repo) show `os.getenv('LCO_API_KEY')` as an illustrative example
— D-05's phrasing borrows that example as if it were already this repo's pattern.
**How to avoid:** The planner has two honest options, both acceptable under D-05's intent ("reuse the
same value/source as LCO, don't introduce a separate SOAR-specific key"): (a) add `FACILITIES['SOAR']
= {'portal_url': 'https://observe.lco.global', 'api_key': os.getenv('LCO_API_KEY', '')}` AND
simultaneously change the existing LCO entry to also read `os.getenv('LCO_API_KEY', '')` instead of
the hardcoded `''` (makes both real and consistent, slightly expands phase scope by touching the LCO
entry too); or (b) mirror today's literal value exactly — `FACILITIES['SOAR'] = {'portal_url':
'https://observe.lco.global', 'api_key': ''}` (true to "same as LCO" without introducing a new env
var this phase didn't otherwise need). Given this phase's explicit non-goal of expanding scope beyond
query/selection/dispatch, **option (b) is the narrower, more defensible reading of D-05** — flag this
explicitly as a discussion point for the planner rather than silently picking one.
**Warning signs:** A test asserting `SOARSettings('SOAR').get_setting('api_key')` returns a non-empty
value will fail in CI/dev environments with no `LCO_API_KEY` env var set regardless of which option is
chosen — do not write such an assertion; test only that the **key exists** in `FACILITIES`, not that
it resolves to a populated secret.

### Pitfall 2: `portal_url` already "works" for SOAR without the settings entry — don't let that mask why `api_key` still needs it
**What goes wrong:** Confirmed live this session: `SOARFacility().get_observation_url('12345')`
already returns the correct `https://observe.lco.global/requests/12345` URL even with **no**
`FACILITIES['SOAR']` entry at all, because `SOARSettings` extends `LCOSettings`, whose
`default_settings['portal_url']` class attribute is already `'https://observe.lco.global'` (not
blank, unlike the base `OCSSettings.default_settings['portal_url'] = ''`). A planner/tester who
verifies "SOAR URLs build correctly" before adding `FACILITIES['SOAR']` will see it pass and might
conclude D-04 is unnecessary.
**Why it happens:** `default_settings['api_key']` is blank (`''`) at every tier (`OCSSettings`,
`LCOSettings`, no override in `SOARSettings`) — there is no class-level non-blank default for
`api_key` the way there is for `portal_url`. So the *URL-building* part of this command (the only
part actually exercised by `sync_lco_observation_calendar.py`, which never makes an authenticated API
call) doesn't need `FACILITIES['SOAR']` to be correct — but real end-to-end SOAR usage (submitting
observations, which this phase's command does NOT do) would need it.
**How to avoid:** Frame the D-04 verification task correctly: confirm `FACILITIES['SOAR']` key
*exists* (so `SOARSettings('SOAR').get_setting('api_key')` resolves to whatever value option (a) or
(b) above specifies, rather than silently falling through to the blank default with no entry at all)
— don't write a test asserting "the URL changes once I add this," because it won't; the URL was
already correct by accident.
**Warning signs:** A plan/test phrased as "before: SOAR URL is broken; after: SOAR URL works" will be
factually wrong and should be caught in review — the actual before/after delta is in `api_key`
resolution and in `get_unconfigured_settings()` (which would list `'SOAR'` as having an unconfigured
`api_key` either way, until/unless a real key is supplied).

### Pitfall 3: A SOAR-facility test fixture using `LCOFacility().get_observation_url()` to compute an expected URL will pass even if the dispatch dict bug (SELECT-05) is still present
**What goes wrong:** Because `SOARFacility().get_observation_url(id)` and `LCOFacility().get_observation_url(id)`
return byte-identical strings (both resolve `portal_url` to the same hardcoded LCO default), a test
that only checks `event.url == LCOFacility().get_observation_url(observation_id)` for a SOAR record
will pass whether or not the dispatch dict correctly used `SOARFacility()` for that record — the URL
field cannot distinguish the bug.
**Why it happens:** Same root cause as Pitfall 2 — `get_observation_url()` is identical across both
classes by inheritance, so it is not a discriminating assertion for SELECT-05.
**How to avoid:** Per the phase's own Success Criterion 4 wording ("verified by a test asserting the
per-facility instance dict dispatches by each record's own `facility` value"), the test must assert
something that *would* differ if the wrong instance were used — e.g. patch/spy on `SOARFacility` vs
`LCOFacility` construction (or their `get_observation_url`/`get_failed_observing_states` methods) and
assert the SOAR record's processing called the SOAR instance's method, not the LCO instance's. A
pure black-box assertion on `event.url` or `event.title` is insufficient for this specific
requirement, even though it's the natural style used by every other test in this file.

### Pitfall 4: Forgetting `_create_record()`'s hardcoded `facility='LCO'` when extending the test fixture
**What goes wrong:** `solsys_code/tests/test_sync_lco_observation_calendar.py:_create_record()`
(lines 50-68) hardcodes `facility='LCO'` in the `ObservationRecord.objects.create(...)` call with no
parameter to override it. A new SOAR test case naively calling `self._create_record(...)` will
silently create an LCO record, not a SOAR one, and the new test will pass for the wrong reason (or
fail confusingly).
**Why it happens:** `_create_record`'s signature is `(self, observation_id, status='PENDING',
scheduled_start=None, scheduled_end=None, **parameter_overrides)` — `facility` is not a parameter at
all; `**parameter_overrides` only forwards to `_parameters()` (which builds the `parameters` dict,
not the `ObservationRecord.facility` field).
**How to avoid:** The planner must specify adding a `facility: str = 'LCO'` parameter to
`_create_record()`'s signature (passed through to `ObservationRecord.objects.create(facility=facility,
...)`), not just adding new call sites that assume an override already exists. This is a concrete,
required signature change to the test helper, not optional.
**Warning signs:** A new SOAR test that asserts on `event.telescope`/`event.title` without first
asserting `ObservationRecord.objects.get(observation_id=...).facility == 'SOAR'` could mask this if
the SOAR-vs-LCO output happens to look identical (which, per Pitfall 3, the URL does).

## Code Examples

### Confirmed live this session: substring-safe multi-proposal + multi-facility filter
```python
# Verified against this repo's actual installed Django 5.2.15 / SQLite, using
# NonSiderealTargetFactory-created fixtures (per CLAUDE.md convention).
# Records created: ('1','LCO','A'), ('2','LCO','AB'), ('3','SOAR','B'), ('4','LCO','C'), ('5','LCO','Z')
qs = ObservationRecord.objects.filter(
    facility__in=['LCO', 'SOAR'],
    parameters__proposal__in=['A', 'B', 'C'],
)
# Result: matched observation_ids == ['1', '3', '4'] -- '2' (proposal='AB') correctly excluded,
# '5' (proposal='Z') correctly excluded. No substring/partial-match leakage.
```

### Confirmed live this session: SOARFacility inheritance and method parity
```python
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility

lco, soar = LCOFacility(), SOARFacility()
isinstance(soar, LCOFacility)                       # True
soar.name                                            # 'SOAR'
soar.get_failed_observing_states()                   # == lco.get_failed_observing_states()
soar.get_terminal_observing_states()                 # == lco.get_terminal_observing_states()
soar.get_observation_url('12345')                    # 'https://observe.lco.global/requests/12345'
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Single `--proposal <code>` (required, exact match), single hardcoded `LCOFacility()` | Comma-list/`ALL` `--proposal`, per-record `{'LCO':.., 'SOAR':..}` dispatch | This phase (Phase 5) | Enables one invocation to cover the whole LCO-family network across both facilities |

**Deprecated/outdated:** None — this is a straightforward additive generalization of Phase 4's
command, not a replacement of a deprecated pattern.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | D-05's intended `api_key` resolution strategy (reuse literal `''` vs. introduce `os.getenv('LCO_API_KEY', '')` for both LCO and SOAR) is ambiguous from CONTEXT.md's wording alone — Pitfall 1 flags both readings as plausible but recommends the narrower one (mirror literal `''`) | Common Pitfalls / Pitfall 1 | If the planner picks the broader reading (introducing a real env var) without flagging it, the phase silently expands scope beyond "query/selection/dispatch" into credentials plumbing, which CONTEXT.md's `<domain>` section explicitly scopes out |

**This is the only assumption requiring confirmation.** All other claims in this research (SOARFacility
class hierarchy, method inheritance, `FACILITIES` dict current contents, `parameters` field type,
`field__in` filter behavior, no-network-call instantiation, existing test helper signatures) were
directly verified this session by reading the installed library source or this repo's own files, or by
executing live probes against this repo's database — see Sources below.

## Open Questions

1. **Should the LCO `FACILITIES` entry's `api_key` also move to `os.getenv()` as part of this phase, or stay exactly as-is (hardcoded `''`)?**
   - What we know: Today it's a hardcoded literal `''`. D-05 references "the same env var" as if one
     already exists; it doesn't.
   - What's unclear: Whether fixing this pre-existing gap is in scope for Phase 5 or a separate concern.
   - Recommendation: Default to the narrower reading (don't touch the LCO entry's value, add
     `FACILITIES['SOAR']` with the same literal `''` or the same expression as LCO's current entry,
     whichever it is) — raise this as an explicit one-line confirmation in the plan rather than
     silently deciding it. This keeps the phase strictly to its stated "query/selection/dispatch scope
     only" boundary.

2. **Exact per-facility counter data structure for D-08's summary (dict keyed by facility name vs. two parallel counter sets)?**
   - What we know: D-08 requires created/updated/unchanged/skipped visible *per facility*; exact
     formatting is Claude's discretion per CONTEXT.md.
   - What's unclear: Nothing blocking — this is genuinely open to implementation choice.
   - Recommendation: A `dict[str, dict[str, int]]` (e.g. `counters = {'LCO': {'created': 0, ...},
     'SOAR': {'created': 0, ...}}`) initialized for both keys eagerly (mirroring D-06's eager
     dispatch-dict pattern) is the simplest structure; the planner should pick this unless a
     compelling reason emerges during implementation.

## Environment Availability

Skipped — this phase has no new external tool/service dependencies. `tom_observations` (providing
`SOARFacility`) is already installed and importable (confirmed this session via direct import and
live instantiation). No network calls are made by this command (confirmed: `LCOFacility()` and
`SOARFacility()` construction is synchronous and credential-independent; `get_observation_url()` is
pure string building). The local SQLite test database is already configured and used by the existing
test suite.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django test runner (`django.test.TestCase`), not pytest — confirmed `pyproject.toml` `testpaths = ["tests", "src", "docs"]` excludes `solsys_code/` |
| Config file | none — Django test discovery via `./manage.py test` |
| Quick run command | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SELECT-02 | `--proposal A,B,C` matches any of 3 codes, no substring leakage | unit (Django TestCase) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_02_comma_list_matches_any_no_substring_leakage` | ❌ Wave 0 — new test, new fixture (`proposal='AB'` decoy record) |
| SELECT-03 | `--proposal ALL` (case-insensitive) syncs every LCO-family record regardless of proposal | unit (Django TestCase) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_03_all_token_case_insensitive_syncs_everything` | ❌ Wave 0 |
| SELECT-04 | A single run produces correct events for both `facility='LCO'` and `facility='SOAR'` records together | unit (Django TestCase) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_04_single_run_covers_both_facilities` | ❌ Wave 0 — requires `_create_record(facility=...)` signature extension (Pitfall 4) |
| SELECT-05 | SOAR record dispatched via SOAR-credentialed instance, never reused `LCOFacility()` | unit (Django TestCase, with spy/patch — see Pitfall 3) | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_select_05_soar_record_uses_soar_facility_instance` | ❌ Wave 0 — needs a discriminating assertion (mock/spy), not a black-box URL check |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`
- **Per wave merge:** `./manage.py test solsys_code` (full Django app suite) + `ruff check .` + `ruff format --check .`
- **Phase gate:** Full Django suite green, plus `python -m pytest` (the separate pytest suite, unaffected by this phase but should remain green) before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_sync_lco_observation_calendar.py::_create_record` — add `facility: str = 'LCO'` parameter (Pitfall 4)
- [ ] New test cases for SELECT-02/03/04/05 listed above
- [ ] `src/fomo/settings.py` — add `FACILITIES['SOAR']` entry (D-04), resolving Open Question 1 for `api_key`'s exact value/expression
- [ ] No new pytest fixtures/conftest needed — existing `setUpTestData()` class-level `user`/`target` fixtures are reused as-is

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | This command does not authenticate end users; it runs as a trusted operator-invoked management command |
| V3 Session Management | No | No web session involved (CLI command) |
| V4 Access Control | No | No new access-control surface; management commands are already gated by server/shell access |
| V5 Input Validation | Yes | `--proposal` is a free-text CLI argument used in an ORM filter. Already confirmed (Phase 4 research, re-verified this session) that Django's `JSONField.__in` lookup parameterizes the bound values rather than string-interpolating them — no SQL/JSON-path injection risk via the ORM. No additional sanitization library needed for the comma-split/strip/dedup logic (D-01..D-03), which is pure string manipulation with no shell/SQL/HTML sink |
| V6 Cryptography | No | No cryptographic operations introduced; `api_key` is opaque credential storage via Django settings, not crypto logic this phase implements |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| SQL/JSON-path injection via `--proposal` comma-separated codes | Tampering | Already mitigated by Django ORM parameterization of `JSONField.__in` lookups (confirmed via live `.query` inspection this session: bound parameters, not raw interpolation) |
| Credential leakage via `api_key` value appearing in a log/stderr message | Information Disclosure | Do not log `facility.facility_settings.get_setting('api_key')` or any raw settings dict in the D-07 skip-and-log path or D-08 summary — only log `record.observation_id`, `record.facility`, and the exception message, exactly as the existing skip-path code already does (it has never logged credentials, and this phase must not introduce that) |

## Sources

### Primary (HIGH confidence — verified this session via direct tool execution)
- `/home/tlister/venv/fomo311_venv/lib64/python3.11/site-packages/tom_observations/facilities/soar.py` — read directly; confirmed `SOARFacility(LCOFacility)` class definition, `name = 'SOAR'`, no override of `get_observation_url`/`get_terminal_observing_states`/`get_failed_observing_states`, `SOARSettings(LCOSettings)` with `get_sites()` override only
- `/home/tlister/venv/fomo311_venv/lib64/python3.11/site-packages/tom_observations/facilities/lco.py` — read directly; confirmed `LCOFacility(OCSFacility)`, `LCOSettings(OCSSettings)` with `default_settings['portal_url']` overridden to a non-blank value
- `/home/tlister/venv/fomo311_venv/lib64/python3.11/site-packages/tom_observations/facilities/ocs.py` — read directly; confirmed `OCSFacility.get_observation_url()`/`get_terminal_observing_states()`/`get_failed_observing_states()` implementations, `OCSSettings.get_setting()` fallback-to-default behavior
- `/home/tlister/venv/fomo311_venv/lib64/python3.11/site-packages/tom_observations/models.py` — read directly; confirmed `ObservationRecord.parameters = models.JSONField()`
- Live ORM probe (this session, against this repo's own test database via Django test runner): `ObservationRecord.objects.filter(facility__in=[...], parameters__proposal__in=[...])` — confirmed correct SQL compilation and zero substring-match leakage with 5 fixture records spanning both facilities
- Live instantiation probe (this session): `SOARFacility()` constructs successfully with no `FACILITIES['SOAR']` entry present, no network call, `portal_url` resolves to the correct value by inherited default while `api_key` resolves to `''`
- `solsys_code/management/commands/sync_lco_observation_calendar.py` (this repo, read directly) — current `--proposal` arg, queryset filter, single `LCOFacility()` instantiation
- `solsys_code/tests/test_sync_lco_observation_calendar.py` (this repo, read directly) — `_create_record()`/`_parameters()` helper signatures
- `src/fomo/settings.py` (this repo, read directly) — confirmed current `FACILITIES` dict contents (`'LCO'`, `'GEM'` only; no `'SOAR'`), confirmed `SOARFacility` already in `TOM_FACILITY_CLASSES`

### Secondary (MEDIUM confidence)
- `.planning/phases/04-lco-queue-sync-command/04-RESEARCH.md` (this repo) — Phase 4's own verified findings on `JSONField`, `parameters__proposal=` ORM lookup correctness, no-network-call facility instantiation; cross-checked and re-confirmed live this session rather than taken on faith
- `.planning/codebase/INTEGRATIONS.md` (this repo) — documents LCO/SOAR/GEM facility configuration status; confirmed no env var is currently documented for LCO's `api_key` (informs Pitfall 1)

### Tertiary (LOW confidence)
None — every claim in this research was either read directly from installed library source, read
directly from this repo's own files, or confirmed via a live probe executed this session.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; existing imports confirmed by direct source read
- Architecture: HIGH — dispatch pattern and filter pattern both confirmed via live execution against this repo's actual database and installed library
- Pitfalls: HIGH — all four pitfalls derived from directly observed behavior this session (file contents, live probe output), not speculation

**Research date:** 2026-06-19
**Valid until:** 30 days (stable, no fast-moving dependencies; revisit if `tomtoolkit` is upgraded past `3.0.0a9` before this phase executes, since it's a pre-release version)
