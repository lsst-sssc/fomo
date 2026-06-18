# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Site/Ephemeris Helper

**Shipped:** 2026-06-14
**Phases:** 1 | **Plans:** 2 | **Sessions:** 1

### What Was Built
- `solsys_code/telescope_runs.py`: `SITES` registry, `get_site()`, `horizon_dip()`, and `sun_event()` returning dip-corrected UTC sunset/sunrise and -15° dark-window crossings via astropy `get_sun`/`AltAz` (coarse-scan + bisection root-finding), with no SPICE/`ephem_utils` dependency.
- `Observatory` model gained a `timezone` field and `to_earth_location()`; migration `0002` seeds the 4 telescope sites (Magellan-Clay/Baade, NTT, FTS).
- 12-test DB-dependent suite (`solsys_code/tests/test_telescope_runs.py`) covering site resolution, dip, sun/dark events, 4-date skycalc validation, -18° twilight crosscheck, and Santiago/Sydney DST resolution — all 9 v1 requirements verified 9/9.
- A pre-executed demo notebook (`docs/notebooks/pre_executed/telescope_runs_demo.ipynb`) plus a new "Demo Notebooks" convention in PROJECT.md for future phases.

### What Worked
- This was explicitly a trial of the GSD discuss→plan→execute→verify loop on this repo, and it completed end-to-end without the workflow stumbling — the secondary "experiment" goal of this milestone succeeded.
- Sourcing `SITES` coordinates from the existing `Observatory` model (by MPC obscode) instead of a standalone hardcoded dict avoided duplication and was checked against `tom_observations.facilities.lco` before deciding.
- Coarse-scan + bisection root-finding for solar-altitude threshold crossings, anchored at local noon with a forward 24h search window, produced results within ~10-55s of skycalc references — well inside the 2-minute tolerance.
- The executor's Rule-1 auto-fix path handled three real bugs inline (an import-shadowing collision, an incorrect crossing-search window, and an exception-type mismatch) without derailing the plan.

### What Was Inefficient
- A pre-existing environment issue (`tomtoolkit==3.0.0a9` no longer ships `tom_catalogs`, but `pyproject.toml`/`settings.py` are still 2.x-targeted) blocked `./manage.py migrate`/`./manage.py test` for both plans. All DB-dependent verification had to be reproduced via standalone `astropy`/`zoneinfo` scripts instead of the real Django test runner — extra executor/verifier effort, and the 12-test suite remains formally unconfirmed by `./manage.py test solsys_code`.
- Two follow-up quick tasks (demo notebook + PROJECT.md convention; ESO notebook data-dir cleanup) were needed after the phase to round out the Definition of Done — could have been folded into phase planning.

### Patterns Established
- Pure `astropy`/`zoneinfo` computation modules should avoid importing `solsys_code.ephem_utils` (triggers a ~1.6GB SPICE kernel download) unless genuinely needed.
- DB-dependent tests for `Observatory`-backed lookups go in `solsys_code/tests/` (Django suite); pure-math helpers can live in `tests/` (pytest).
- Each phase should ship a demo notebook under `docs/notebooks/pre_executed/` (or `docs/notebooks/` if dependency-free) per PROJECT.md's Demo Notebooks convention.

### Key Lessons
1. Resolve the `tom_catalogs`/`tomtoolkit==3.0.0a9` environment mismatch before starting v1.1 — it will block `./manage.py test` for every future phase until fixed.
2. When a milestone's goal includes "validate the workflow itself," call that out explicitly in verification — it materially changed how gaps (e.g. the environment blocker) were triaged (pre-existing vs. introduced).
3. Standalone-script re-verification of DB-dependent logic is a workable fallback when `./manage.py test` is blocked, but it's not a substitute for running the real test suite — track it as an open item, not a closed one.

### Cost Observations
- Sessions: 1
- Notable: Two plans (35min + 25min) completed in a single session with no replanning; both Rule-1 auto-fixes and the environment-blocker workaround were absorbed without escalation.

---

## Milestone: v1.1 — Classical Run Ingest

**Shipped:** 2026-06-16
**Phases:** 2 | **Plans:** 3 | **Sessions:** ~4

### What Was Built
- `ParsedRun` frozen dataclass + `parse_run_line()` in `telescope_runs.py` — handles 3 date-range orderings, hyphenated instruments, year defaulting, and telescope prefix-match with descriptive `ValueError` for ambiguous names (e.g. bare `'Magellan'`).
- `load_telescope_runs` Django management command — expands parsed run date ranges into idempotent nightly `CalendarEvent`s. Upsert via `get_or_create` keyed on `(telescope, instrument, start_time)` with conditional save; per-line `(ValueError, Observatory.DoesNotExist)` handler (log+skip, not abort).
- 16 test methods across `test_telescope_runs.py` (10 new) and `test_load_telescope_runs.py` (6 new); all 95 `./manage.py test solsys_code` tests pass — first milestone where DB-dependent tests ran under the real Django test runner from day 1.
- Demo notebook `load_telescope_runs_demo.ipynb` demonstrates full pipeline: Observatory seeding → schedule ingest → CalendarEvent display → idempotency re-run. Confirmed executable via `jupyter nbconvert --to notebook --execute`.

### What Worked
- Resolving the `tomtoolkit`/`tom_catalogs` environment blocker before v1.1 start paid off immediately — all DB-dependent tests ran under `./manage.py test` from plan 02-01 onward.
- TDD execution for 03-01 (RED commit `2d80e63` → GREEN commit `7134e10`) was clean: 6 tests collected/failed, then all 95 passing. Clear signal at each step.
- Code review (03-REVIEW.md) caught 4 real issues (OSError handling, TemporaryDirectory leak, assert guard, redundant return) that were fixed in a repair pass without reopening the plan.
- 6/6 UAT scenarios confirmed live on dev DB at milestone close — no surprises, behavior matched automated tests exactly.

### What Was Inefficient
- Microsecond accumulation in `get_or_create` key was a subtle bug: `astropy Time.to_datetime()` produces sub-second precision, producing different `start_time` values on each run and breaking idempotency. An explicit `assert start_time.microsecond == 0` assertion in the test at plan time would have caught this.
- UAT file (`03-UAT.md`) was generated but not run during the phase — required a separate UAT session at milestone close. Either run it inline during execution or fold the scenarios into `TestCase` methods.
- Context exhaustion at 79% on 2026-06-16 ended a session before the phase was fully closed out, requiring a fresh start for the final UAT/close steps.

### Patterns Established
- `get_or_create` with microsecond-stripped datetimes as key: `time.replace(microsecond=0)` before any DB key lookup involving `astropy Time.to_datetime()`.
- Management commands that process files line-by-line: catch per-line exceptions, write to `self.stderr` with line number + original text; report `skipped` count in summary. Don't abort on bad input.
- Code review repair pass: write a `03-REVIEW-FIX.md` with each finding's resolution before committing fixes; gives a traceable paper trail.

### Key Lessons
1. Write a test for the exact DB field precision you need. If `CalendarEvent.start_time` must be microsecond-free, assert `.microsecond == 0` in the test — don't assume `astropy`→Django timezone conversion is clean.
2. UAT generated by `/gsd-verify-work` is valuable only if it's actually run. Run it inline at phase completion or convert scenarios directly to `TestCase` methods instead of a separate UAT checklist.
3. Context exhaustion is predictable for phases with management command + test suite + demo notebook in 2 plans — split into 3 plans next time (command, tests, notebook as separate plans), or use `/gsd-pause-work` to create a clean handoff file.

### Cost Observations
- Sessions: ~4 across 4 days (2026-06-13 → 2026-06-16)
- Notable: TDD RED→GREEN on 03-01 was the cleanest plan execution of the milestone — clear success criteria + test-first discipline eliminated debugging cycles.

---

## Milestone: v1.2 — LCO Queue Calendar Sync

**Shipped:** 2026-06-18
**Phases:** 1 | **Plans:** 1 | **Sessions:** ~2

### What Was Built
- `sync_lco_observation_calendar` management command: `--proposal <code>` selection via `ObservationRecord.objects.filter(facility='LCO', parameters__proposal=code)`, one `CalendarEvent` per record keyed on `LCOFacility().get_observation_url(observation_id)` (the real `/requests/<id>` path, correcting the literal ROADMAP wording per D-01).
- Two time-source branches (`_time_window`): `parameters['start']`/`['end']` banner when `scheduled_start is None`, `scheduled_start`/`scheduled_end` placed block once the LCO scheduler acts — same record, same event, no duplicate.
- Terminal-state title prefixes (`_failure_prefix`) driven by `LCOFacility().get_failed_observing_states()` membership (4 states), deliberately excluding `COMPLETED` per D-06's research correction (`get_terminal_observing_states()` returns those 4 + `COMPLETED` = 5 — using the wrong helper would have wrongly prefixed completed observations).
- No-churn create-or-update: all 7 changeable fields compared before `.save()`, verified by a test asserting `modified` is untouched across two runs on an unchanged record.
- 14 new Django tests (110 total in `solsys_code` suite); demo notebook `sync_lco_observation_calendar_demo.ipynb`.
- Two follow-up quick tasks during the milestone window de-emphasized `[QUEUED]` event styling and then fixed a contrast regression introduced by the first fix — both outside the phase plan but inside the v1.2 close window.

### What Worked
- TDD RED→GREEN again was clean: Task 1 wrote 14 failing tests against a stub command, Task 2 made them all pass.
- The plan's locked D-06 decision (use `get_failed_observing_states()`, not `get_terminal_observing_states()`) was verified two independent ways — a live Django-shell call confirming the 4-vs-5 state sets, and a `grep -c get_terminal_observing_states` check on the source — leaving no ambiguity at verification time.
- Code review caught one real bug (CR-01: an inconsistent-scheduled-times case that would have raised `IntegrityError` via a `None` `end_time`) before milestone close, with a regression test added and independently reproduced by the verifier.
- Verification scored 7/7 truths and reproduced the full 110/110 test run and clean `ruff check`/`ruff format --check` independently rather than trusting the executor's self-report.

### What Was Inefficient
- The `[QUEUED]` calendar-event de-emphasis fix (quick task `260618-lw4`) shipped with a contrast bug (white text on near-transparent fill) that needed an immediate follow-up fix (`260618-mck`) — a quick visual check against both the white and `#f8f9fa` overflow-cell backgrounds before committing would have caught it in one pass.
- A demo notebook for Phase 4 was initially missed and had to be backfilled by a separate quick task (`260617-mlr`) after the phase closed, the same gap pattern noted in v1.0's retrospective — the "ship a demo notebook per phase" convention exists in PROJECT.md but phase plans don't always include it as an explicit task.
- The pre-close artifact audit flagged 4 completed quick tasks as `status: unknown` purely because their SUMMARY.md frontmatter doesn't set a field the audit tool checks — no actual gap, but it cost a manual cross-check against STATE.md's Quick Tasks Completed table to confirm.

### Patterns Established
- `_failure_prefix(status, facility)` pattern: derive prefix-trigger membership from a live facility helper call (`get_failed_observing_states()`) rather than hardcoding the same strings, so a future upstream change to the failure-state set is picked up automatically.
- Visual/style quick tasks (CSS, template overrides) should be checked against every background variant the element can appear on (e.g. both in-month white and `#f8f9fa` other-month overflow cells) before considering the task done.

### Key Lessons
1. When a plan's success criteria depend on a library helper returning a specific set (e.g. "4 failure states, not 5 terminal states"), verify the live return value during both planning and verification — don't trust documentation or memory of the API.
2. Add "ship the demo notebook" as an explicit task in the phase plan itself (not just a PROJECT.md convention) — it has now been missed and backfilled in two consecutive milestones (v1.0, v1.2).
3. A quick task that touches visual/CSS state should include a checklist of the backgrounds/contexts it must remain legible against, to catch contrast regressions before they ship.

### Cost Observations
- Sessions: ~2 (phase execution 2026-06-17; quick-task polish + milestone close 2026-06-18)
- Notable: single-plan phase with TDD RED→GREEN, one code-review bug found and fixed pre-close, zero replanning.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 1 | 1 | First GSD run on this repo; validated discuss→plan→execute→verify loop end-to-end |
| v1.1 | ~4 | 2 | First milestone with DB-dependent tests under real Django test runner from day 1; TDD RED→GREEN on management command |
| v1.2 | ~2 | 1 | Single-plan phase; code review caught a real pre-close bug (CR-01); two same-day quick-task follow-ups (style fix + contrast regression fix) |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | +12 (Django, unconfirmed by `./manage.py test` due to env blocker) | - | 0 |
| v1.1 | +16 (10 parser + 6 ingest; all 95 under `./manage.py test solsys_code`) | - | 0 |
| v1.2 | +14 (sync command; all 110 under `./manage.py test solsys_code`) | - | 0 |

### Top Lessons (Verified Across Milestones)

1. Fix environment/dependency blockers before they compound across milestones — the `tomtoolkit`/`tom_catalogs` mismatch surfaced in v1.0 and was resolved before v1.1. ✓ Validated.
2. Write a test for the exact DB field precision you need (astropy→Django datetime conversion produces microseconds; assert `.microsecond == 0` in tests that use `astropy Time.to_datetime()`).
3. UAT generated by `/gsd-verify-work` is only useful if it's run — fold scenarios into `TestCase` methods at plan time rather than generating a separate checklist.
4. "Ship a demo notebook per phase" needs to be an explicit phase-plan task, not just a PROJECT.md convention — missed and backfilled after the fact in both v1.0 and v1.2.
