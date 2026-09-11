---
status: testing
phase: 34-the-observation-projector-trigger
source: [34-01-SUMMARY.md, 34-02-SUMMARY.md, 34-03-SUMMARY.md, 34-04-SUMMARY.md]
started: 2026-09-11T04:44:59Z
updated: 2026-09-11T04:44:59Z
---

## SCHED-06: live narrowing over real observing nights

Spike 004 left SCHED-06 as a PARTIAL verdict -- the projector and sweep exist and are
tested, but nothing had yet proven a real, pending `KEY2026B-004` record narrow on the
calendar purely from `python manage.py updatestatus`, with no sweep run in between.
Plan 34-04 Task 1 captured the baseline this section tracks; the verdict closes only
when the re-check below is filled in.

### Baseline

- **Captured at:** 2026-09-11T04:44:59.526430+00:00 (UTC)
- **Proposal:** `KEY2026B-004`
- **Pending record count at baseline:** 74 (56 `queued`, 18 `placed`)
- **Baseline artifact:** `docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json`
  (per-record `observation_id`, target name, status, `scheduled_start`/`scheduled_end`,
  and the projected event's start/end/title, all as of the baseline capture)
- **Notebook cell:** `project_observation_calendar_demo.ipynb`, "SCHED-06 baseline: the
  pending `KEY2026B-004` records (D-20)"

### The rule between baseline and re-check

Over the coming nights, run **only**:

```console
$ python manage.py updatestatus
```

Nothing else -- and specifically **not** `python manage.py project_observation_calendar`
(the sweep). Running the sweep in between would let a bulk write path narrow the
records instead of the `post_save` receiver alone, which is the opposite of what
SCHED-06 needs proven. If the sweep is run for any other operational reason before the
re-check below happens, note it in the table and treat the verdict as still open rather
than closed by a mixed cause.

### Dated re-check table

Fill in one row per re-check. A re-check re-executes
`project_observation_calendar_demo.ipynb`
(`jupyter nbconvert --to notebook --execute --inplace
docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`) and diffs its
SCHED-06 section against the baseline JSON above -- committing the re-executed notebook
each time.

| Date | Nights watched (updatestatus only?) | Records narrowed queued->placed | Records narrowed placed->observed | Notes | Verdict |
|------|--------------------------------------|----------------------------------|-------------------------------------|-------|---------|
| _(pending)_ | | | | | PARTIAL (SCHED-06 still open) |

**Current verdict: PARTIAL.** SCHED-06 closes only once a re-check row above shows at
least one record narrowing with nothing but `updatestatus` run in between.

## Summary

total: 1
passed: 0
issues: 0
pending: 1
