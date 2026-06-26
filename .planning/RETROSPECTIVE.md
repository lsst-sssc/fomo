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

## Milestone: v1.3 — Full LCO Facility Sync

**Shipped:** 2026-06-24
**Phases:** 4 (5, 6, 7, 07.1) | **Plans:** 5 | **Sessions:** ~6

### What Was Built
- Generalized `sync_lco_observation_calendar` to accept a comma-list/`ALL` `--proposal` argument and dispatch LCO and SOAR `ObservationRecord`s through an eager `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dict, fixing the SELECT-05 single-shared-`LCOFacility()` dispatch bug.
- Replaced the flat `parameters['instrument_type']` read with a `c_1..c_5` multi-config scanner distinguishing SOAR's SPECTRUM science config from ARC/LAMP_FLAT calibration configs and LCO MUSCAT's per-channel exposure shape, plus a dedicated `extraction_failed` counter.
- Migrated `SITE_TELESCOPE_MAP` to a verified 7-site dict (`tlv` dropped — confirmed absent from installed library code) and added single-attempt, timeout-bounded, never-leaking live LCO Observation Portal API resolution with a coarse-label fallback, `[UNVERIFIED]` title prefix, and a `telescope_api_failed` counter distinct from `skipped`.
- Phase 07.1 closed a milestone-audit gap: `_coarse_telescope_label` made facility-aware so a SOAR record's API-failure fallback resolves to `'4m0'` instead of the raw `'SOAR_GHTS_REDCAM'` string (which had produced a doubled, non-coarse title).
- 21 new/changed tests across the milestone (110 → 131 total in `solsys_code` suite); demo notebook updated at every phase boundary (Phase 5, 6, 7, and 07.1 each added or updated cells with real executed output).

### What Worked
- The phase ordering locked in research (query generalization → instrument extraction → telescope-label API/fallback) meant each phase's fixture shapes were already correct by the time the next phase needed them — Phase 6's SOAR-shape extraction work, Phase 7's fallback labeling.
- Two real production-data gaps were caught by live UAT against real DB records (not just unit tests) and fixed same-day via quick tasks: missing `SITE_TELESCOPE_MAP` coverage for `coj`/`ogg` aperture classes (`260623-su3`) and a `.get()` vs bracket-indexing security gap in `_build_event_fields` (`260623-ocs`).
- Operator (LCO staff) checkpoints during Phase 7 planning resolved two real ambiguities cheaply — dropping `tlv` entirely and confirming the `elp`/`lsc`/`cpt`/`tfn` aperture-class inventory — rather than shipping `[ASSUMED]` guesses that would have needed a later fix.
- The milestone-level audit (`/gsd-audit-milestone`) did exactly what it's for: caught a real, shipped defect (SOAR fallback label) that was structurally invisible to all three phases' own LCO-centered verification, tests, and UAT. Phase 07.1 closed it same-day with a 4-line fix plus a regression test.

### What Was Inefficient
- Phase 7's `verify_phase_goal` step was skipped during original execution and had to be backfilled retroactively during the milestone audit (`07-VERIFICATION.md` is explicitly a retroactive backfill). This is exactly the kind of single-phase verification gap that let the SOAR fallback defect ship — a missed verification step on the highest-risk phase (new I/O, live API) compounded into a milestone-level gap.
- The pre-close artifact audit again flagged completed quick tasks as `status: unknown` — 7 this time (vs. 4 at v1.2 close), purely because their `SUMMARY.md` frontmatter omits a `status` field the audit tool checks. v1.2's retrospective already named this as a no-actual-gap friction point; it recurred and grew rather than getting fixed at the source.
- Every one of Phases 5, 6, and 7 needed its own SOAR- or production-data-shape gap closed via a same-day or near-term quick task (`260619-jpr` SOAR site mapping, `260620-v9x` notebook update, `260623-su3`/`260623-ocs`), suggesting SOAR fixture coverage should be a standing checklist item in phase planning for this command, not a per-phase discovery.

### Patterns Established
- `_coarse_telescope_label(instrument_type, facility_name)` — pass the facility *name* (string) into a labeling function that needs to branch by facility, not the credentialed facility instance in scope; keeps pure-labeling functions free of credential-bearing objects.
- Live external-API calls inside a sync command: single attempt, explicit timeout, generic fixed-message logging on failure (never raw response body or credentials), and a distinct failure counter separate from existing skip/error counters — established in Phase 7, reused as-is in Phase 07.1's fix.
- When a milestone-level audit finds an integration gap a phase's own LCO-only (or otherwise single-shape) verification couldn't see, scope the fix as an inserted decimal phase (07.1) rather than reopening the original phase — keeps the audit finding, the fix, and its dedicated regression test traceable as one unit.
- Quick-task `SUMMARY.md` frontmatter should always set `status: complete` — established as a fix-at-milestone-close convention twice now (v1.2, v1.3); worth promoting to the quick-task template itself so it stops recurring.

### Key Lessons
1. `verify_phase_goal` must run for every phase before milestone close, especially the highest-risk one (new I/O / external API) — skipping it on Phase 7 is exactly how the SOAR fallback defect reached a "35/35 tests green, UAT passed" state while still being broken for one real facility.
2. A phase whose tests, fixtures, and demo notebook are all scoped to one facility/shape (here: LCO-only) needs an explicit cross-facility/cross-shape check before being called done, not just before milestone audit — the milestone audit is reliable but later and more expensive than catching it in-phase.
3. Fix recurring metadata gaps at the template level once they're seen twice. The `status: complete` frontmatter gap cost manual cross-checking at both v1.2 and v1.3 close; the third occurrence should not happen.

### Cost Observations
- Sessions: ~6 across 6 days (2026-06-18 → 2026-06-24)
- Model mix: not tracked this milestone
- Notable: Phase 7 (highest-risk, new I/O) took the longest per-plan time (~50min/plan vs. 6-25min for the others) and was also the phase whose verification got skipped — the two are likely related; budget extra verification time, not less, on the riskiest phase.

---

## Milestone: v1.4 — Calendar Visual Clarity

**Shipped:** 2026-06-26
**Phases:** 2 (08, 09) | **Plans:** 4 | **Sessions:** ~2 (2026-06-24 → 2026-06-26)

### What Was Built
- Phase 8: `CalendarEventTelescopeLabel` OneToOneField sidecar model (solsys_code's first real model + migration `0001`); `sync_lco_observation_calendar` writes it via `update_or_create` after the existing `CalendarEvent` create-or-update; dashed-border + native-tooltip visual cue in `calendar.html` for fallback-labeled events; first `calendar.html` view-level rendering test.
- Phase 9: `solsys_code/templatetags/calendar_display_extras.py` with `proposal_color` (sha256 normalize → 8-color colorblind-vetted palette), `status_border_css` (locked CSS literals for queued/terminal box-shadow rings), and `visible_proposals` (collision-grouped legend aggregation); rewrote `calendar.html` event branches to use all three tags; fixed the `[QUEUED]` flat-grey override; added footer legend with click-to-filter JS IIFE surviving htmx month swaps.
- 40 new tests (131 → 171 total): 49-test sync suite, 23-test `calendar_display_extras` unit suite, 13-test calendar template integration suite. 3/3 human UAT items passed (CVD colorblind check deferred by user at close).

### What Worked
- Phase execution was fast: 9 min (09-01), 18 min (09-02), 24 min (08-01), 11 min (08-02) — no context overruns during plan execution.
- The `/gsd:sketch` session during Phase 9 planning resolved the status-treatment design decision (border vs. opacity vs. striping) without a dedicated research phase — the pre-vetted border-style recommendation from research translated cleanly into a visual spec, then into locked CSS literals in the tag library.
- Composing the Phase 8 dashed border and Phase 9 status box-shadow as orthogonal CSS properties (no collision, each on a separate property) was specified in 09-UI-SPEC.md before implementation and confirmed with a Pitfall 3 integration test — zero rework on the composition.
- Click-to-filter IIFE placement inside the htmx-swapped `#calendar-partial` fragment (before closing `</div>`) was identified as a pitfall in research (Pitfall 5) and baked into the implementation from the start — UAT confirmed it survived month swaps on first try.
- The `data-proposal` attribute keyed on the resolved color hex (not the raw proposal string) for legend ↔ event matching was a D-04 decision made during 09-01 planning, avoiding a separate deduplication pass in the JS that would have been needed if keyed on the string.

### What Was Inefficient
- The gsd-tools.cjs `summary-extract` tool picked up bug-report lines from a Phase 09 SUMMARY (code-review findings embedded in SUMMARY frontmatter as `one_liner` entries) and injected them into MILESTONES.md as accomplishments — required manual fix at close. The CLI can't distinguish "plan accomplishment" from "bug found during review" if both land in the same frontmatter field.
- Phase 09 ran into a context-exhaustion event at 76% through the verify step (2026-06-26), requiring a resume-and-finish cycle. The session had accumulated research, planning, execution, and verification context for both phases without a `/clear` between phases — a mid-milestone `/clear then /gsd:resume-work` after Phase 8 closed would have preserved more headroom.
- The STATE.md frontmatter still said `status: verifying` at session resume despite the last commit message saying "UAT passed, phase marked complete" — the state update and the commit description diverged during the context-exhausted close; resumption needed to read both to reconstruct what had actually happened.

### Patterns Established
- sha256-normalize-then-modulo palette-index for deterministic, cross-restart color assignment — never Python's built-in `hash()` (process-salted in CPython 3.3+). Enforce via `grep -c 'hash(' <templatetag_file>` = 0 in verification.
- Sidecar model using `OneToOneField(primary_key=True)` to extend a third-party model: no extra PK column, reverse accessor is a clean single-row attribute, third-party migrations untouched. Suitable when the parent model is owned by an installed app you can't edit.
- Template treats a missing sidecar row as the "default" state (here: verified, no dashed border) rather than requiring the sidecar to always be present — keeps classical-schedule events (no sidecar) safe without a special code path.
- JS IIFE placed inside the htmx-swapped fragment, not in the page `<head>`, for event-delegation state that needs to reset on each month swap — the IIFE re-executes, `activeProposal` resets to null, stale filter state is impossible.
- `/gsd:sketch` during phase planning (not as a separate pre-phase) is the right granularity for a single visual-design decision that would block plan writing — keeps the decision visible in the phase's CONTEXT/UI-SPEC history without adding a whole discuss-phase cycle.

### Key Lessons
1. Clear context between phases, not just between milestones. Accumulating research+planning+execution+verification for two phases in one session risks running out of context during the last verification step — a cheap `/clear then /gsd:resume-work` after Phase 8 close would have preserved Phase 9 headroom.
2. The gsd-tools.cjs `summary-extract` one-liner field captures the first "accomplishment"-style item it finds in a SUMMARY — if a plan's SUMMARY mixes deliverable descriptions and code-review bug entries in the same frontmatter namespace, the tool can grab a bug description as the one-liner. Fix at the SUMMARY authoring step: keep accomplishments and bug-findings in separate frontmatter keys.
3. When context exhaustion forces a mid-verification close, write the commit message AND the STATE.md `status` field to the same truth ("complete", not "verifying") — divergence between the two costs extra reconstruction time on the next resume.

### Cost Observations
- Sessions: ~2 (2026-06-24 → 2026-06-26, one context-exhaustion event mid-Phase 9 verification)
- Model mix: not tracked this milestone
- Notable: v1.4 was the fastest milestone per-plan in this project (avg ~15 min/plan vs. ~35 min/plan for v1.1 and ~50 min/plan for the highest-risk plans in v1.3) — the purely additive nature of the work (new sidecar model, new template tag library, template rewrite with no Django model changes beyond Phase 8) kept execution focused.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 1 | 1 | First GSD run on this repo; validated discuss→plan→execute→verify loop end-to-end |
| v1.1 | ~4 | 2 | First milestone with DB-dependent tests under real Django test runner from day 1; TDD RED→GREEN on management command |
| v1.2 | ~2 | 1 | Single-plan phase; code review caught a real pre-close bug (CR-01); two same-day quick-task follow-ups (style fix + contrast regression fix) |
| v1.3 | ~6 | 4 (incl. inserted 07.1) | First milestone with a dedicated milestone-level audit (`/gsd-audit-milestone`) finding a real shipped defect; first decimal gap-closure phase (07.1) inserted post-audit; first retroactive `verify_phase_goal` backfill (Phase 7) |
| v1.4 | ~2 | 2 | Fastest milestone per-plan (~15 min/plan avg); `/gsd:sketch` for single visual-design decision during planning; first Django template-tag library + first real solsys_code migration; context-exhaustion event mid-verification |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | +12 (Django, unconfirmed by `./manage.py test` due to env blocker) | - | 0 |
| v1.1 | +16 (10 parser + 6 ingest; all 95 under `./manage.py test solsys_code`) | - | 0 |
| v1.2 | +14 (sync command; all 110 under `./manage.py test solsys_code`) | - | 0 |
| v1.3 | +21 (multi-proposal/facility, multi-config extraction, telescope-label + fallback; all 131 under `./manage.py test solsys_code`) | - | 0 |
| v1.4 | +40 (sidecar write + rendering + template tag unit + integration; all 171 under `./manage.py test solsys_code`) | - | 0 |

### Top Lessons (Verified Across Milestones)

1. Fix environment/dependency blockers before they compound across milestones — the `tomtoolkit`/`tom_catalogs` mismatch surfaced in v1.0 and was resolved before v1.1. ✓ Validated.
2. Write a test for the exact DB field precision you need (astropy→Django datetime conversion produces microseconds; assert `.microsecond == 0` in tests that use `astropy Time.to_datetime()`).
3. UAT generated by `/gsd-verify-work` is only useful if it's run — fold scenarios into `TestCase` methods at plan time rather than generating a separate checklist.
4. "Ship a demo notebook per phase" needs to be an explicit phase-plan task, not just a PROJECT.md convention — missed and backfilled after the fact in both v1.0 and v1.2.
5. A phase's own verification is only as broad as its fixtures — a single-facility/single-shape test+UAT+notebook can all pass while a defect ships for an untested shape (v1.3 Phase 7's LCO-only verification missed a SOAR-only defect). Milestone-level audits exist precisely to catch this, but catching it in-phase is cheaper.
6. Quick-task `SUMMARY.md` frontmatter missing a `status` field has now cost manual cross-checking at two consecutive milestone closes (v1.2: 4 tasks, v1.3: 7 tasks) — fix at the template level, not at close time.
