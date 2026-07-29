# Phase 6: Correct Instrument-Type Extraction - Research

**Researched:** 2026-06-20
**Domain:** Parsing LCO/SOAR multi-configuration `ObservationRecord.parameters` shapes (pure Python, no I/O)
**Confidence:** HIGH

## Summary

CONTEXT.md for this phase already contains fully confirmed, source-cited decisions (D-01..D-06)
for the extraction algorithm itself — this research does not relitigate those. What it adds is
the precise current-state codebase context the planner needs to write tasks against: the exact
lines to change in `sync_lco_observation_calendar.py`, the exact shape of the existing test
fixture helpers (which do **not** currently support injecting `c_N_*` keys and must be extended),
confirmation that no other code in the repo touches `parameters['instrument_type']` or assumes
the flat single-config shape, and a clarification that CONTEXT.md's references to a
"`skipped_count`" counter describe a *concept* (the per-facility `counters[facility]['skipped']`
dict entry), not a literally-named variable — there is no `skipped_count` identifier anywhere in
the codebase to match against.

Independent re-verification against the installed `tom_observations` package on this machine
(`/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_observations`) confirms every
field-name and line-number claim in CONTEXT.md's `<canonical_refs>` section is accurate as of the
currently installed version. One additional nuance not previously surfaced: `ocs.py`'s
`_build_configuration` returns `None` (and the configuration is dropped from the submitted
payload entirely) when a configuration's `instrument_configs` list ends up empty — meaning a
real record can have a `c_N_*` slot **entirely absent** from `parameters`, not just present-but-
empty. This reinforces D-02's "key missing entirely" fallback case as a realistic scenario, not
just a legacy-data edge case.

**Primary recommendation:** Replace the single line `instrument = record.parameters['instrument_type']`
(`sync_lco_observation_calendar.py:143`) with a call to a new helper implementing D-01..D-06.
Extend (don't replace) `_parameters()`/`_create_record()` in the test file to accept arbitrary
`c_N_*` overrides, since their current fixed-keyword-argument signatures cannot express the new
multi-config/MUSCAT test shapes.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Instrument-type extraction from `parameters` | Business logic (management command helper) | — | Pure data transformation, no DB/API/template involvement; belongs alongside `_derive_telescope`/`_title_for` as a private helper in the same command module |
| Science vs. calibration disambiguation | Business logic | — | Domain rule (D-01/D-02), not a model or view concern |
| Skip/log/count on extraction failure | Business logic (per-record loop) | — | Reuses the existing per-record catch-log-continue convention already in `handle()` |

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXTRACT-01 | Instrument type is extracted by scanning `c_1_instrument_type`..`c_5_instrument_type` for the configuration with a populated exposure time, replacing the v1.2 flat-key assumption | Confirmed exact current code (`_build_event_fields` line 143) to replace; confirmed `c_{j+1}_instrument_type`/`c_{j+1}_configuration_type` field-name generation in installed `ocs.py:1025-1030`; D-02 fallback heuristic in CONTEXT.md covers this requirement's literal wording |
| EXTRACT-02 | Extraction verified against SOAR's multi-config shape (spectrum/arc/lamp-flat) and LCO MUSCAT's per-channel shape, never mistaking calibration for science | Confirmed `SOARSpectroscopyObservationForm.configuration_type_choices()` returns exactly `SPECTRUM`/`ARC`/`LAMP_FLAT` (`soar.py:103`); confirmed `LCOMuscatImagingObservationForm` defines `c_{j+1}_ic_{i+1}_exposure_time_g/_r/_i/_z` with no flat `exposure_time` field (`lco.py:583-596`), and that `_build_instrument_config` requires all 4 channels truthy or returns `None` (`lco.py:649-654`) |

## Project Constraints (from CLAUDE.md)

- Any new `Target` fixture in tests/notebooks **must** use
  `tom_targets.tests.factories.NonSiderealTargetFactory`, never `SiderealTargetFactory` — already
  followed by the existing test file's `setUpTestData` (`cls.target = NonSiderealTargetFactory.create()`);
  no new `Target` fixture should be introduced by this phase since the class-level `cls.target` is
  reused, but if the planner adds a per-test target, it must use the non-sidereal factory.
- Planning-doc terminology: write "create or update" / "find-or-create" instead of "upsert" in
  CONTEXT.md/RESEARCH.md/PLAN.md/PATTERNS.md and other `.planning/` artifacts.
- `ruff check . --fix` / `ruff format .` — single quotes, 120-col line length; both files currently
  pass `ruff check` cleanly (`All checks passed!` verified during this research) — any new code
  must keep that clean.
- DB-dependent tests (this phase's tests touch `ObservationRecord`/`Target`) belong in
  `solsys_code/tests/`, run via `./manage.py test solsys_code` (or
  `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` for just this file)
  — confirmed working, 19/19 passing as of this research.
- Google-style docstrings with `Args:`/`Returns:`/`Raises:` — match the existing helpers'
  (`_derive_telescope`, `_failure_prefix`, `_time_window`) docstring style exactly.

## Existing Code: Precise Before-State

### `sync_lco_observation_calendar.py` — the exact line and its context

The single line to replace is **line 143** inside `_build_event_fields` (lines 126-162):

```python
def _build_event_fields(record: ObservationRecord, facility: LCOFacility) -> dict[str, Any]:
    ...
    telescope = _derive_telescope(record.parameters['site'])      # line 142 — UNCHANGED (Phase 7 scope)
    instrument = record.parameters['instrument_type']              # line 143 — THIS LINE IS REPLACED
    proposal = record.parameters['proposal']                       # line 144 — UNCHANGED
    url = facility.get_observation_url(record.observation_id)
    start_time, end_time = _time_window(record)
    title = _title_for(record, telescope, instrument, facility)
    ...
```

`_build_event_fields` is called from inside the per-record loop in `handle()` (lines 254-259),
already wrapped in:

```python
try:
    fields = _build_event_fields(record, facility)
except (KeyError, ValueError) as exc:
    self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
    counters[record.facility]['skipped'] += 1
    continue
```

**Important implication for D-06:** today, a missing `parameters['instrument_type']` key raises
`KeyError`, which is already caught by this `except (KeyError, ValueError)` block and counted in
`counters[record.facility]['skipped']`. If the new extraction helper also raises `KeyError` (or a
similar built-in) on total extraction failure, it will be silently absorbed into the **existing**
generic skip path and counted as a plain `'skipped'` — NOT the new dedicated D-06 counter the
phase requires. **The planner must ensure the new helper either:**
1. raises a distinguishable exception type that `handle()`'s per-record loop catches separately
   and routes to the new dedicated counter, or
2. returns a sentinel (e.g. `None`) that the loop checks explicitly before incrementing the new
   counter and continuing,

rather than letting `KeyError` silently fall through to the pre-existing `'skipped'` bucket. This
is the most consequential planning detail in this research — D-06's "own dedicated counter…
distinct from the existing `skipped_count`" requirement will silently fail to be met if the new
helper just raises `KeyError` like the line it replaces does today.

**Counter-naming clarification:** CONTEXT.md (D-06) and REQUIREMENTS.md (SYNC-06, a *different*
phase's requirement reusing similar wording) both refer to "`skipped_count`" prose-style. There is
**no variable literally named `skipped_count`** anywhere in this codebase — the actual structure is
`counters: dict[str, dict[str, int]]` with `counters[facility_name]['skipped']` as the accumulator,
initialized per-facility in two places: the eager dict literal (lines 231-234) and the defensive
`counters.setdefault(...)` (line 250). The new dedicated extraction-failure counter must follow this
**same nested-dict-of-counters pattern**, e.g. adding an `'extraction_failed'` key (or similar — naming
is Claude's discretion per CONTEXT.md) alongside `'created'/'updated'/'unchanged'/'skipped'` in both
the dict-literal initializer and the `setdefault` defensive default, and must appear in the per-facility
summary line (lines 278-283) using the same `f'{key}: {count}'` phrasing as the other four.

### Confirmed: no other code reads `instrument_type` or `c_N_*` keys

Repo-wide search (`grep -rn "instrument_type\|c_1_\|c_2_\|c_3_\|c_4_\|c_5_\|configuration_type"` across
`solsys_code/`, excluding tests) returns matches **only** in
`sync_lco_observation_calendar.py` itself (the docstrings and the one line). A broader search for any
`parameters[...]` consumer (`grep -rn "parameters\["`) likewise returns matches only inside this same
file (`'start'`, `'end'`, `'site'`, `'instrument_type'`, `'proposal'` — all already-known keys for this
command). **Nothing else in the repo needs updating or flagging as out-of-scope** — this phase's blast
radius is confirmed to be exactly the one file CONTEXT.md identified, plus its test file.

Separately, `CalendarEvent.instrument` (the model field this command populates) is also written by
`solsys_code/management/commands/load_telescope_runs.py:93` — but that is the unrelated Stage 2
classical-run-ingest command, sourcing its own `parsed.instrument` from a different (non-`ObservationRecord`)
input entirely. No shared code path; no change needed there.

## Existing Code: Test Fixture Helpers (precise before-state)

`solsys_code/tests/test_sync_lco_observation_calendar.py` has two fixture helpers, both with
**fixed keyword-argument signatures** — neither currently accepts arbitrary extra parameter keys:

```python
def _parameters(
    proposal: str = 'TESTCODE123',
    start: str = '2026-07-01T00:00:00',
    end: str = '2026-07-02T00:00:00',
    instrument_type: str = '2M0-SCICAM-MUSCAT',
    site: str | None = 'coj',
) -> dict:
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

```python
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

`_create_record`'s `**parameter_overrides` is forwarded **directly** into `_parameters(**parameter_overrides)`
— so any keyword passed to `_create_record` that isn't one of `_parameters`'s five named params
(`proposal`/`start`/`end`/`instrument_type`/`site`) will raise `TypeError: _parameters() got an
unexpected keyword argument` today. **This means `_parameters()` itself must be extended** (its
signature, not just call sites) before any test can construct a SOAR multi-config or MUSCAT
per-channel fixture. Two concrete extension shapes the planner should choose between:

1. **Add an explicit `extra_params: dict | None = None` parameter** to `_parameters()`, merged into
   the returned dict via `params.update(extra_params or {})` — keeps the five existing named params
   untouched (no risk of breaking the 19 existing call sites) and lets new tests pass an arbitrary
   `c_N_*` dict without touching every existing call.
2. **Convert `_parameters()` to accept `**kwargs`** merged on top of the defaults dict — slightly
   less explicit but more flexible; risk is silently typo'd kwargs (e.g. `c_1_configuation_type`)
   passing through uncaught since there's no fixed parameter list to catch a `TypeError` on misspelling.

Given this codebase's existing convention of fixed, explicit, type-hinted parameters everywhere
(per CLAUDE.md conventions and the rest of this file), **option 1 (`extra_params: dict | None`) is
the better fit** — it's additive, preserves the explicit five-arg signature for the common case
(today's still-valid single-config shape), and avoids the typo-silently-swallowed risk of `**kwargs`.

Also note: today's default `instrument_type='2M0-SCICAM-MUSCAT'` and default `site='coj'` mean
**every existing test that doesn't override these still produces today's flat single-config shape**
(success criterion 1 in the phase goal — "unchanged from today's correct cases"). The new
extraction helper's D-02 fallback path must continue to handle this exact shape (no
`configuration_type` key in the legacy/flat parameters dict at all) — i.e. all 19 existing tests
are themselves a regression suite for D-02's fallback heuristic and must keep passing unmodified.

### Existing test-assertion conventions to follow

- **Status/event-state assertions:** `event = CalendarEvent.objects.get()` then assert individual
  field values (`event.title`, `event.instrument`, `event.telescope`) — see
  `test_sync_05_telescope_instrument_proposal_populated` for the closest existing analog to what a
  new "MUSCAT per-channel populates correctly" test should look like.
- **Log-output assertions:** capture `stderr_buf = io.StringIO()`, pass as `stderr=stderr_buf` to
  `call_command(...)`, then `self.assertIn('990001', stderr_buf.getvalue())` — see
  `test_skip_path_missing_site_logged_and_skipped` for the exact pattern the new D-06 skip-and-log
  test should mirror (assert the failing record's `observation_id` appears in stderr).
- **Counter/summary-line assertions:** capture `stdout_buf = io.StringIO()`, then
  `self.assertIn('created: 0', stdout_buf.getvalue())` or `self.assertIn('updated: 1', ...)` — the
  new dedicated D-06 counter's test should assert its own summary-line substring (e.g.
  `self.assertIn('extraction_failed: 1', stdout_buf.getvalue())`, using whatever name is chosen)
  the same way.
- **Multi-record coexistence pattern:** every skip-path test creates **two** records — one that
  should fail/skip and one that should succeed — then asserts `CalendarEvent.objects.count() == 1`
  and that the surviving event's url matches the *good* record. New SOAR-multi-config and MUSCAT
  tests should follow this same "one record exercising the new shape + one baseline record" pattern
  rather than testing the new shape in isolation, to also exercise non-regression.
- **`facility='SOAR'` fixtures:** existing SOAR tests always pass `site='sor'` explicitly (default
  `site='coj'` is an LCO site code) — e.g.
  `self._create_record('620002', proposal='SHARED', facility='SOAR', site='sor', instrument_type='SOAR_GHTS_REDCAM')`.
  New SOAR multi-config fixtures must keep `site='sor'` for the same reason (confirmed correct in
  `SITE_TELESCOPE_MAP`, not part of this phase's scope to change).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Recognizing science vs. calibration `configuration_type` values | A hardcoded if/elif chain duplicated per-form | A single whitelist set/tuple (`{'EXPOSE', 'REPEAT_EXPOSE', 'SPECTRUM', 'REPEAT_SPECTRUM', 'STANDARD'}`) checked via `in`, per D-01 | Matches the existing `_FAILURE_PREFIX_BY_STATUS`-style dict/set lookup convention already used in this same file (`_failure_prefix`); keeps the recognized-vocabulary list in one place, easy to extend for a future Blanco phase |
| Detecting MUSCAT channel population | Per-channel if/elif duplicated 4 times | A single `any(...)` over the 4 channel key names, per D-04 | Matches D-04's explicit "more lenient than the form's own all-4-required validation" design intent; a single `any()` call is the natural Pythonic expression of "any of the 4 truthy" |

**Key insight:** This phase's whole "don't hand-roll" risk is over-engineering the dispatch —
CONTEXT.md's D-05 explicitly notes `parameters['observation_type']` is available as a free
correctness cross-check but is **not required** as a dispatch mechanism. The planner should not
introduce a `observation_type`-keyed branch table; the `configuration_type` whitelist plus
exposure-signal fallback (D-01/D-02/D-04) is sufficient and was deliberately chosen as the simpler
of the two approaches during discussion.

## Common Pitfalls

### Pitfall 1: Letting `KeyError` from the new helper fall into the wrong counter
**What goes wrong:** The new extraction-failure path gets silently merged into the pre-existing
generic `'skipped'` counter instead of getting its own dedicated counter, because `handle()`'s
existing `except (KeyError, ValueError)` block is broad enough to swallow it.
**Why it happens:** The line being replaced (`record.parameters['instrument_type']`) already
relies on `KeyError` propagating up to that generic handler — habit/inertia could lead to writing
the replacement the same way.
**How to avoid:** Either raise a distinct exception subtype caught separately in `handle()`, or
have the new helper return `None`/a sentinel that `_build_event_fields` (or `handle()`) checks
explicitly before falling through to the generic except block.
**Warning signs:** A new test asserting on a dedicated counter name passes only because it
happens to match the *existing* `'skipped'` count, not because a new counter actually exists.

### Pitfall 2: Forgetting `_parameters()`'s current signature can't accept `c_N_*` keys
**What goes wrong:** A new test passes `self._create_record('x', **{'c_1_configuration_type': 'SPECTRUM', ...})`
and gets `TypeError: _parameters() got an unexpected keyword argument 'c_1_configuration_type'`
because `_create_record`'s `**parameter_overrides` forwards straight into `_parameters`'s five
fixed named params.
**Why it happens:** `_create_record`'s existing `**parameter_overrides` pass-through *looks* like
it already supports arbitrary keys, but it only supports whatever `_parameters()` itself declares.
**How to avoid:** Extend `_parameters()`'s own signature first (see "Existing Code: Test Fixture
Helpers" above) before writing any new-shape test.
**Warning signs:** `TypeError` at test collection/run time naming `_parameters`.

### Pitfall 3: Treating a missing `c_N_*` slot as a malformed record
**What goes wrong:** Assuming every config slot 1-5 is always present with at least empty/falsy
keys, when in reality (confirmed via `ocs.py`'s `_build_configuration` — see Summary) a configuration
with no populated `instrument_configs` is dropped from the submitted payload **entirely**, so its
`c_N_*` keys may be **completely absent** from `parameters`, not merely empty-stringed.
**Why it happens:** Test fixtures built by hand might always include all 5 slots with `None`/`''`
placeholders, which doesn't exercise the "key absent" path the real API actually produces.
**How to avoid:** The extraction helper must use `.get(...)` (or equivalent safe lookup) when
checking each `c_N_configuration_type`/`c_N_instrument_type`/exposure key — never assume the key
exists — and at least one new test should omit slots entirely (only include `c_1_*`/`c_2_*`, no
`c_3_*`/`c_4_*`/`c_5_*` keys at all) to prove this.
**Warning signs:** `KeyError` raised from inside the new helper itself (as opposed to the
deliberate D-06 "nothing found" signal) when scanning a record with fewer than 5 populated slots.

### Pitfall 4: Breaking the 19 existing passing tests
**What goes wrong:** Changing `_parameters()`'s default behavior (e.g. its five named defaults)
while extending it breaks all the existing tests that rely on `instrument_type='2M0-SCICAM-MUSCAT'`,
`site='coj'` defaults producing today's flat single-config shape.
**Why it happens:** Refactoring a widely-used fixture helper risks unintended default changes.
**How to avoid:** Add new capability purely additively (e.g. the `extra_params` param defaulting
to `None`/merging nothing) — confirmed via this research that all 19 existing tests pass today;
re-run `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` after any
fixture-helper change to confirm they still pass unmodified.
**Warning signs:** Any of the 19 existing tests in this file start failing after a fixture-helper edit.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (via `django.test`), run through the Django test runner (NOT pytest — `pyproject.toml` `testpaths` excludes `solsys_code/`) |
| Config file | none dedicated — Django settings module `src.fomo.settings`, test discovery via `./manage.py test` |
| Quick run command | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` (0.15s for 19 tests, confirmed during this research) |
| Full suite command | `python manage.py test solsys_code` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXTRACT-01 | Single populated config (today's real LCO shape) extracts unchanged | unit (Django TestCase) | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar.TestSyncLcoObservationCalendar.test_sync_05_telescope_instrument_proposal_populated` (existing, must keep passing) | Yes — existing test |
| EXTRACT-02 | SOAR multi-config (spectrum/arc/lamp-flat): extracted instrument is the science config, never `ARC`/`LAMP_FLAT` | unit (Django TestCase) | new test, e.g. `test_extract_02_soar_multi_config_picks_spectrum_not_calibration` | No — Wave 0 gap |
| EXTRACT-02 | MUSCAT per-channel: extraction reflects populated channel(s), no crash/empty result | unit (Django TestCase) | new test, e.g. `test_extract_02_muscat_per_channel_exposure_extracts_instrument` | No — Wave 0 gap |
| D-02 (CONTEXT.md) | Legacy/flat shape with no `configuration_type` key falls back to exposure-signal heuristic | unit (Django TestCase) | covered by existing flat-shape tests (regression) — confirm no new test needed, or add one explicit fallback-path test for clarity | Partial — existing tests cover happy path implicitly |
| D-06 (CONTEXT.md) | Fully malformed/empty record (no config found by either signal) is skipped, logged, counted in dedicated counter | unit (Django TestCase) | new test, e.g. `test_d06_no_extractable_config_logged_and_counted_separately` | No — Wave 0 gap |

### Sampling Rate
- **Per task commit:** `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar`
- **Per wave merge:** `python manage.py test solsys_code` (full app suite) plus `ruff check . && ruff format --check .`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] Extend `_parameters()` in `solsys_code/tests/test_sync_lco_observation_calendar.py` with an
  additive `extra_params: dict | None = None` parameter (or equivalent) — required before any new
  test in this phase can be written.
- [ ] New test: SOAR multi-config (`c_1_configuration_type='SPECTRUM'`, `c_2_configuration_type='ARC'`,
  `c_3_configuration_type='LAMP_FLAT'`, all with populated `c_N_instrument_type`/exposure fields) —
  asserts extracted instrument matches `c_1`'s instrument type, never `c_2`/`c_3`'s.
- [ ] New test: LCO MUSCAT per-channel (`c_1_ic_1_exposure_time_g/_r/_i/_z` populated, no flat
  `c_1_exposure_time`) — asserts extraction succeeds and reflects the populated channel config,
  per D-04's "any of the 4 truthy" leniency (test at least one case with all 4 populated, and
  per D-04's lenient design, the planner should consider one case with fewer than 4 populated to
  prove the leniency claim, even though REQUIREMENTS.md's literal wording only requires the
  populated-channel case to work).
- [ ] New test: D-06 fully malformed record (no `configuration_type` anywhere, no exposure signal
  anywhere) — asserts the record is skipped, logged (observation_id visible in stderr), and counted
  in the new dedicated counter (distinct from the existing per-facility `'skipped'` key) — visible
  in the stdout summary line.
- [ ] No new test framework/config needed — existing Django `TestCase` infrastructure in this file
  fully covers the new requirements once `_parameters()` is extended.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Not touched — this phase only changes parsing of already-fetched `ObservationRecord.parameters` |
| V3 Session Management | No | Not touched |
| V4 Access Control | No | Not touched — no new view/permission surface |
| V5 Input Validation | Yes | The new extraction helper must use safe dict lookups (`.get(...)`, membership checks against a fixed whitelist) rather than blind indexing, so a malformed/partial `parameters` dict (attacker-uncontrolled here — sourced from the LCO/SOAR API via prior sync, not user input — but still externally-sourced data) cannot raise an unhandled exception that aborts the whole sync run. This is exactly D-06's purpose. |
| V6 Cryptography | No | Not touched |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed/unexpected external API data shape causing unhandled exception and aborting a batch job | Denial of Service (availability) | Per-record catch-log-continue (already established convention in this file); D-06 extends it with a dedicated counter so operators can see how often it happens without it ever crashing the run |

No new external-facing attack surface is introduced by this phase — it is a pure internal data-shape
parsing change against data the command already had read access to (no new fields, no new queries,
no new network calls). The only security-relevant property is robustness against malformed input
shape, already addressed by D-02/D-06's fallback-then-skip design.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| (none) | — | — | All claims in this research were independently re-verified against the installed `tom_observations` package source on this machine, the current repo's `sync_lco_observation_calendar.py`/test file, and a passing test run — no `[ASSUMED]` claims were introduced beyond what CONTEXT.md already carries forward from its own discussion (which itself cites confirmed library source lines). |

**This table is empty:** All claims in this research were verified against the installed library
source or the local codebase directly during this research session — no user confirmation needed
beyond what CONTEXT.md's own decisions already represent (those are already locked).

## Open Questions

1. **Exact name for the new dedicated extraction-failure counter (D-06)**
   - What we know: CONTEXT.md leaves this to Claude's discretion; must be distinct from the
     existing `'skipped'` key and visible in the per-facility summary line.
   - What's unclear: No specific name is mandated. Candidates: `'extraction_failed'`,
     `'no_instrument'`, `'unrecognized'`.
   - Recommendation: `'extraction_failed'` reads clearly in the summary line
     (`extraction_failed: 1`) and matches the existing terse style of `'created'/'updated'/'unchanged'/'skipped'`.
     The planner should pick one and ensure it's added to the dict literal init (lines 231-234),
     the `setdefault` defensive default (line 250), and the summary join (lines 278-283) — all
     three locations, or the counter will silently default to 0/never display for some code paths.

2. **Whether the new extraction logic should live as one helper or several (Claude's discretion per CONTEXT.md)**
   - What we know: D-01..D-06 describe a single conceptual algorithm (scan whitelist, fall back to
     exposure-signal heuristic, fall back to skip-and-count).
   - What's unclear: Whether to split into e.g. `_extract_instrument()` (orchestrator) +
     `_find_science_config()` (D-01 whitelist scan) + `_find_exposure_signal_config()` (D-02/D-04
     fallback scan) as three small functions, vs. one larger function.
   - Recommendation: Given this file's existing convention of small, single-purpose private helpers
     (`_derive_telescope`, `_failure_prefix`, `_title_for`, `_time_window` are all under ~20 lines
     each with focused docstrings), splitting into 2-3 small helpers matching the file's existing
     style is preferable to one large function, but either is explicitly acceptable per CONTEXT.md.

## Sources

### Primary (HIGH confidence)
- Installed `tom_observations` package, version as currently pinned in this venv
  (`/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_observations`) —
  `facilities/ocs.py` lines 1024-1030 (`_add_config_fields`), 1192-1213 (`_build_instrument_configs`/
  `_build_configuration`); `facilities/soar.py` lines 95-125
  (`SOARSpectroscopyObservationForm`/`SOARSimpleGoodmanSpectroscopyObservationForm`);
  `facilities/lco.py` lines 565-660 (`LCOMuscatImagingObservationForm`/`_build_instrument_config`) —
  all read directly from disk during this research session, confirming every field-name/line-number
  claim already made in CONTEXT.md's `<canonical_refs>`.
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — read in full during this
  research session (current production code, 285 lines).
- `solsys_code/tests/test_sync_lco_observation_calendar.py` — read in full during this research
  session (current test suite, 474 lines, 19 tests, all confirmed passing via
  `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar`).
- Repo-wide `grep` for `instrument_type`/`c_1_`.. `c_5_`/`configuration_type`/`parameters[` across
  `solsys_code/` (excluding tests) — confirms no other consumer of these keys/shapes exists.

### Secondary (MEDIUM confidence)
- None — all claims this phase depends on were directly verifiable against installed source or
  the local repo; no documentation/web sources were needed (no external network dependency in this
  phase's scope, and `.planning/config.json` has all search providers disabled).

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: N/A — no new libraries; uses only stdlib dict access and existing CONTEXT.md-confirmed field names
- Architecture: HIGH — single-file, single-line change confirmed via direct code read + repo-wide grep for blast radius
- Pitfalls: HIGH — derived directly from tracing the existing exception-handling flow and fixture-helper signatures in the actual code, not speculation

**Research date:** 2026-06-20
**Valid until:** Stable until the installed `tom_observations` version changes (pinned dependency;
re-verify field names/line numbers if `tomtoolkit`/`tom_observations` is upgraded) or until Phase 7
changes adjacent code in the same file.
