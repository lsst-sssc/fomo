# Spike Conventions

Patterns and stack choices established across spike sessions. New spikes follow these unless the question requires otherwise.

## Stack

- Python 3.11 / Django (the project's own stack); no new packages. Spike code imports FOMO's
  existing helpers **read-only** (`solsys_code.calendar_utils`, the sync commands' vocabulary,
  `solsys_code.telescope_runs`) rather than re-implementing them.
- Real data over fixtures: the KEY2026B-004 records loaded by `backfill_lco_observations` are
  the standing dataset. Any Target fixture uses `tom_targets.tests.factories.NonSiderealTargetFactory`.

## Structure

- One directory per spike under `.planning/spikes/NNN-name/`; comparison spikes as
  `NNN-a-…` / `NNN-b-…`.
- Run a spike with Django set up, without adding it to the package:
  `python manage.py shell -c "import runpy; runpy.run_path('<spike>/spike.py', run_name='__main__')"`.
  A spike module that must be importable by dotted path (e.g. a TOM hook receiver) is a
  sibling file and the script inserts its own directory on `sys.path`.
- Spikes that reuse another spike's code do `sys.path.insert(0, '<other spike dir>')` and
  import it as a module (003/004 reuse 002's `projector`).
- Every spike writes a `forensic-log*.json` next to its README: ISO timestamps, per-scenario
  or per-record rows, and a `summary` dict of the booleans/counts the README's verdict cites.

## Patterns

- **Rollback by default.** Fact-finding spikes run inside `transaction.atomic()` and call
  `transaction.set_rollback(True)` at the end, then prove the real data is untouched
  (row counts, no spike-tagged rows). Only a spike whose *point* is something the user can
  look at (002's sweep) persists rows — and ships a `cleanup.py` that deletes exactly those
  rows, selected by its own key namespace plus a marker string in `description`.
- **Own key namespace.** Spike-written `CalendarEvent`s use a URL/key namespace the campaign
  reconciler never writes (`https://…/requests/{id}` for observations, `ALLOC:` for
  allocations), and the spike asserts the reconciler's `RUN:` count is unchanged before/after.
- **Never raise from a projector.** Anything that runs inside `post_save` returns an
  `unprojectable` action instead of propagating, so a bad record cannot break the save that
  triggered it.
- **Expectation tables.** Trigger/behaviour spikes tabulate `scenario / fired / expected / ok`
  and count `matched_expectation`, so a surprise shows up as ✗ rather than as prose.
- **Monkeypatch the portal, never call it.** Facility status calls are patched on the
  facility class inside a `try/finally`; the only network a spike may use is a real
  `updatestatus` run the user chooses to make (004).

## Tools & Libraries

- `insert_or_create_calendar_event(lookup, fields)` — the no-churn create/update/unchanged
  helper; use it, don't copy it.
- `record_time_window(record)` — the stage rule (window vs block); `extract_instrument`,
  `coarse_telescope_label` for labels; `_failure_prefix` from the LCO sync command for the
  `[EXPIRED]/[CANCELLED]/[FAILED]` vocabulary.
- `sun_event(site, night, kind='sun')` for sunset/sunrise nights; needs an `Observatory` with
  a timezone (K92 Sutherland, X05/268/269/809 Chile, E10 Siding Spring all have one).
- Django `post_save` (not TOM's `observation_change_state`) as the per-save trigger.
