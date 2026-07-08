---
status: verifying
trigger: "ESO/NTT classical run lines produce an off-by-one extra CalendarEvent when transcribed verbatim from Tatoo (LPO schedule tool). Tatoo displays date ranges like '2026-07-09 - 2026-07-13, 4.0 nights' where the end date is the noon-to-noon closing boundary of the LAST night, not itself an observing night. load_telescope_runs' _iter_run_nights() computes n_nights = day2 - day1 + 1 and treats both day1 and day2 as observing-night evening dates for ALL telescopes uniformly, silently over-counting ESO/NTT lines by one night."
created: 2026-07-08T00:00:00Z
updated: 2026-07-08T00:00:00Z
---

## Current Focus

hypothesis: The Magellan/Las Campanas "E - S + 1, both dates inclusive" night convention (docs/design/telescope_runs_calendar.rst:191-198), implemented uniformly in `_iter_run_nights()` (solsys_code/management/commands/load_telescope_runs.py:35-51), is correct for Magellan but was never validated against ESO/NTT/Tatoo semantics, and produces one spurious extra CalendarEvent night for NTT/La Silla run lines whose day2 was transcribed straight from Tatoo's displayed end date.
test: Trace parse_run_line() + _iter_run_nights() against the line "NTT EFOSC2 allocation 9-13 July" (from local file Didymos_runs, repo root, gitignored) and compare the generated night list to Tatoo's own stated night count for the same range.
expecting: Tatoo says 4.0 nights for 2026-07-09 - 2026-07-13 (real nights: Jul 9, 10, 11, 12; end date 07-13 is only the closing noon boundary of the night of the 12th). _iter_run_nights() should produce exactly those 4 dates for an NTT line, not 5.
next_action: Confirm the root cause already traced by the orchestrator (see Evidence) and implement a CODE-SIDE fix, per explicit user instruction ("fix it code-side"): make the night-count/expansion logic site-aware so ESO sites (NTT/La Silla) drop one night relative to the current day2-day1+1 formula (i.e. last observing night = day2 - 1 for ESO-sourced lines), while Magellan/Las Campanas keeps its existing, already-validated E-S+1 "both inclusive" behavior unchanged. Do not fall back to the input-transcription-convention workaround (documenting that users must type day2-1) -- the user explicitly asked for the code fix, not a documentation/process fix.

reasoning_checkpoint:
  hypothesis: "_iter_run_nights() applies the Magellan-validated both-inclusive formula n_nights = day2 - day1 + 1 uniformly, over-counting ESO/NTT lines by one because ESO's Tatoo end date is the noon-to-noon closing boundary of the last night, not itself an observing night."
  confirming_evidence:
    - "Direct trace: 'NTT EFOSC2 allocation 9-13 July' -> n_nights=5 [Jul 9,10,11,12,13]; Tatoo itself reports 4.0 nights for that range."
    - "docs/design/telescope_runs_calendar.rst:191-198 explicitly scopes the E-S+1 both-inclusive convention to Las Campanas (Magellan) only; ESO/Tatoo has no such validation."
    - "_iter_run_nights (load_telescope_runs.py:36-52) has zero site branching; parsed.telescope (resolved SITES key) is already available to branch on."
    - "SITES lives in telescope_runs.py as a flat module-level dict, alongside a module-level set KNOWN_STATUSES -- so a co-located module-level ESO-site set is consistent with existing code shape."
  falsification_test: "After the fix, parse_run_line('NTT EFOSC2 allocation 9-13 July') |> _iter_run_nights must yield exactly [Jul 9,10,11,12] (4 dates) AND 'Magellan-Baade IMACS 17-18 July' must still yield [Jul 17,18] (2 dates). If NTT still yields 5 or Magellan changes, the hypothesis/fix is wrong."
  fix_rationale: "Add a co-located per-site night-convention registry ESO_NOON_TO_NOON_SITES (frozenset, next to SITES in telescope_runs.py) and branch on membership in _iter_run_nights to drop the trailing Tatoo boundary night (n_nights -= 1) for ESO sites only. This targets the root cause (a site-specific night-count convention) rather than the symptom (one extra event), keeps Magellan's validated behavior byte-for-byte unchanged, and is extensible (add future ESO telescope keys to the frozenset)."
  blind_spots: "Degenerate ESO range day2 == day1 would drop to 0 nights -- guarded with a ValueError. Assumes NTT is the only current ESO/noon-to-noon site (true: SITES has Magellan-Clay, Magellan-Baade, NTT, FTS; FTS=Faulkes/LCO queue is both-inclusive like Magellan). Have not re-validated ESO La Silla twilight times themselves (out of scope -- only the night COUNT is wrong, not the per-night sun events)."
next_action_after_checkpoint: Implement fix in telescope_runs.py + load_telescope_runs.py, correct the 3 stale NTT==5 tests and add ESO/Magellan regression tests, update load_telescope_runs_demo.ipynb, run ruff + targeted Django tests, record Resolution.

## Symptoms

expected: For NTT (ESO La Silla) classical run lines whose date range was transcribed verbatim from ESO's Tatoo schedule tool, `./manage.py load_telescope_runs <file>` creates exactly the number of CalendarEvents Tatoo itself reports (e.g. 4 nights for "9-13 July" / Tatoo "4.0 nights").
actual: |
  `_iter_run_nights()` creates day2 - day1 + 1 = 5 nights for "NTT EFOSC2 allocation 9-13 July"
  (2026-07-09, 10, 11, 12, AND 13) -- one extra, spurious CalendarEvent for "night of July 13",
  which was never actually allocated. Confirmed by direct trace (not full manage.py run, to avoid
  mutating the dev DB):

  ```
  NTT EFOSC2 allocation 9-13 July
    day1/day2: 9 13 -> n_nights = 5
    nights: [2026-07-09, 2026-07-10, 2026-07-11, 2026-07-12, 2026-07-13]
  ```

  Magellan lines in the same file are unaffected (Las Campanas Start/End really are both
  inclusive observing nights per docs/design/telescope_runs_calendar.rst:191-198, confirmed
  against the actual Magellan schedule):
  ```
  Magellan-Baade IMACS 17-18 July -> 2 nights (17, 18)  [correct]
  Magellan-Clay Lightspeed 18-20 July BoN-0626 -> 3 nights (18, 19, 20)  [correct]
  ```
errors: None raised -- this is a silent over-count, not an exception. No test currently catches
  it: solsys_code/tests/test_load_telescope_runs.py and test_telescope_runs.py were written
  against the Magellan-derived E-S+1 convention and don't have an ESO/Tatoo-range case.
reproduction: |
  1. `cd /home/tlister/git/fomo_devel`
  2. In a Django-configured shell (DJANGO_SETTINGS_MODULE=src.fomo.settings):
     ```python
     from solsys_code.telescope_runs import parse_run_line
     from solsys_code.management.commands.load_telescope_runs import _iter_run_nights
     parsed = parse_run_line('NTT EFOSC2 allocation 9-13 July')
     _iter_run_nights(parsed)  # -> 5 dates, should be 4
     ```
  3. Equivalently: `./manage.py load_telescope_runs Didymos_runs` (repo-root gitignored file)
     would create 5 CalendarEvents for NTT/EFOSC2 instead of 4.
started: Present since the Magellan-derived night convention (docs/design/telescope_runs_calendar.rst
  "Night convention (confirmed from the Las Campanas telescope schedule)") was implemented and
  applied uniformly to all telescopes in Phase work on load_telescope_runs.py; only surfaced now
  while cross-checking a real Didymos (117.2A2N.001) ESO run against Tatoo's noon-to-noon semantics.

## Eliminated

(none yet -- root cause already strongly evidenced by direct code trace; session manager/debugger
should confirm and move straight to fix per user's "fix it code-side" instruction rather than
re-deriving from scratch.)

## Evidence

- timestamp: 2026-07-08T00:00:00Z
  checked: docs/design/telescope_runs_calendar.rst:191-198 ("Night convention" section)
  found: |
    "**Night convention (confirmed from the Las Campanas telescope schedule).** Run Start and End
    dates are both observing nights by evening date... Therefore a run from evening S to evening E
    yields E - S + 1 nights." This convention is explicitly scoped/validated to Las Campanas
    (Magellan) only -- ESO/NTT/Tatoo is never mentioned in this section, and the rest of the doc
    has no separate ESO night-count convention documented.
  implication: The E-S+1-both-inclusive formula is a Magellan-specific fact that got generalized
    to all telescopes in the implementation without a corresponding ESO validation step.

- timestamp: 2026-07-08T00:00:00Z
  checked: solsys_code/management/commands/load_telescope_runs.py:35-51 (_iter_run_nights)
  found: |
    ```python
    def _iter_run_nights(parsed: ParsedRun) -> list[date]:
        if parsed.day2 < parsed.day1:
            raise ValueError(...)
        first_night = date(parsed.year, parsed.month, parsed.day1)
        n_nights = parsed.day2 - parsed.day1 + 1
        return [first_night + timedelta(days=i) for i in range(n_nights)]
    ```
    No telescope/site-specific branching -- applies the same day2-day1+1 formula regardless of
    `parsed.telescope`.
  implication: This is the exact function to make site-aware for a code-side fix; `parsed.telescope`
    (resolved SITES key, e.g. 'NTT') is already available on ParsedRun and can be used to
    distinguish ESO sites from Magellan.

- timestamp: 2026-07-08T00:00:00Z
  checked: Direct trace via Django shell (DJANGO_SETTINGS_MODULE=src.fomo.settings), parsing the
    real Didymos_runs file (repo root, gitignored) line by line through parse_run_line() +
    _iter_run_nights(), no DB writes.
  found: |
    NTT EFOSC2 allocation 9-13 July -> day1=9 day2=13 -> n_nights=5 ->
      [2026-07-09, 2026-07-10, 2026-07-11, 2026-07-12, 2026-07-13]
    Magellan-Baade IMACS 17-18 July -> n_nights=2 -> [2026-07-17, 2026-07-18]  (correct, unaffected)
    Magellan-Clay Lightspeed 18-20 July BoN-0626 -> n_nights=3 -> [2026-07-18, 2026-07-19, 2026-07-20]
      (correct, unaffected)
  implication: Confirms the bug is isolated to ESO/NTT lines and does not affect Magellan lines --
    a site-aware fix in _iter_run_nights (or in ParsedRun/parse_run_line) that special-cases ESO
    sites will not regress the already-correct Magellan behavior.

- timestamp: 2026-07-08T00:00:00Z
  checked: solsys_code/telescope_runs.py SITES / get_site() (site registry used by parse_run_line
    via _resolve_telescope, and consumed by load_telescope_runs.py's Command.handle)
  found: |
    SITES maps telescope tokens (e.g. 'NTT', 'Magellan-Baade', 'Magellan-Clay') to Observatory-backed
    site records including timezone. This is the natural place to also carry an "ESO noon-to-noon"
    vs "Magellan both-inclusive" night-convention flag, or alternatively the fix can branch directly
    on `parsed.telescope == 'NTT'` (currently the only ESO site in SITES) in _iter_run_nights.
  implication: Two viable code-side implementation points: (a) add a per-site convention flag to
    SITES/get_site() and branch on it in _iter_run_nights (more extensible if more ESO
    telescopes/instruments are added later), or (b) a minimal telescope-name check in
    _iter_run_nights itself. Debugger should pick based on what's simplest/most consistent with
    existing SITES structure -- inspect get_site()/SITES definition before deciding.

## Resolution

root_cause: |
  load_telescope_runs._iter_run_nights() applied the Magellan/Las Campanas
  "both dates inclusive" night formula n_nights = day2 - day1 + 1 uniformly to
  every telescope. That formula was only ever validated for Las Campanas
  (docs/design/telescope_runs_calendar.rst "Night convention"). ESO sites
  (NTT / La Silla) have their date ranges transcribed verbatim from ESO's Tatoo
  scheduling tool, whose displayed END date is the noon-to-noon CLOSING boundary
  of the last night, not itself an observing night. Applying the both-inclusive
  formula to an ESO range therefore over-counts by exactly one night (e.g.
  "NTT EFOSC2 allocation 9-13 July", Tatoo "4.0 nights", produced 5 events for
  Jul 9-13 instead of 4 for Jul 9-12). Silent over-count, no exception; no
  existing test caught it because the suite encoded the Magellan convention as if
  universal (asserted 5 NTT events).
fix: |
  Made the night expansion site-aware. Added a co-located per-site night-convention
  registry ESO_NOON_TO_NOON_SITES = frozenset({'NTT'}) to solsys_code/telescope_runs.py
  (next to SITES / KNOWN_STATUSES, matching the existing module-level-collection
  code shape). In load_telescope_runs._iter_run_nights(), for a parsed.telescope in
  ESO_NOON_TO_NOON_SITES the trailing Tatoo boundary night is dropped
  (n_nights -= 1, i.e. last observing night = day2 - 1), with a ValueError guard
  for the degenerate day2 == day1 case. Magellan/Las Campanas keeps E - S + 1
  byte-for-byte unchanged. Future ESO telescopes are handled by adding their SITES
  key to the frozenset. Documented the ESO convention in the design doc.
verification: |
  - Django tests: `python manage.py test solsys_code.tests.test_load_telescope_runs
    solsys_code.tests.test_telescope_runs` -> 44 passed. Corrected 3 stale NTT==5
    assertions to ==4; added 4 regression tests: NTT "9-13 July" -> [Jul 9,10,11,12]
    (4 nights), Magellan-Baade "17-18 July" -> [Jul 17,18] (unchanged), ESO
    single-night "9-10 July" -> [Jul 9], and ESO degenerate "9-9 July" -> ValueError.
  - Quality gates: `ruff check .` clean, `ruff format --check .` clean on all changed
    source files.
  - Demo notebook regenerated (jupyter nbconvert --to notebook --execute --inplace):
    new deterministic convention cell prints NTT -> 4 nights (Jul 9-12) / Magellan ->
    2 nights; DB-dependent inspect cell now shows exactly 4 NTT events (Jul 9-12, no
    Jul 13) after clearing 8 stale/buggy NTT rows from the dev DB; summary table row
    updated 5 NTT -> 4 NTT. No execution errors.
files_changed:
  - solsys_code/telescope_runs.py (added ESO_NOON_TO_NOON_SITES registry)
  - solsys_code/management/commands/load_telescope_runs.py (site-aware _iter_run_nights)
  - solsys_code/tests/test_load_telescope_runs.py (corrected 3 tests, added 4 regression tests)
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb (convention demo cell + refreshed output)
  - docs/design/telescope_runs_calendar.rst (documented ESO noon-to-noon convention)
