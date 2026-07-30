---
created: 2026-07-27T00:00:00.000Z
title: Correct owned-nights framing in upstream planning docs
area: docs
files:
  - .planning/phases/26-canonical-record-spike/26-CONTEXT.md
---

## Problem

`.planning/phases/26-canonical-record-spike/26-CONTEXT.md`'s D-11 write-up describes
`CampaignRun` pk=1's real 15-night window as having "4 uncovered nights" that need
coverage — language that treats every night inside a run's window as a night the run
owns and should be observing on, with any night lacking an `ObservationRecord` framed as
a gap needing to be filled.

`CampaignRun` pk=1 (FTS/MuSCAT4) is a **queue-scheduled** run, not a classically
scheduled one. Per the domain correction recorded in `26-DECISION.md`'s `### Domain
correction — queue windows are not sets of owned nights` section (and settled further by
plan 26-05's `#### Queue-run projection — settled` verdict), a queue run's window is a
span of time during which an observation *could* happen, not a set of nights the run
owns outright. A night inside that window with no scheduled observation is the **normal,
correct state** for a queue run — not a gap that is "missing" or needs backfilling. The
"4 uncovered nights" framing in `26-CONTEXT.md` predates this correction and still
describes pk=1's unscheduled nights as coverage gaps, which is exactly the framing the
domain correction found to be wrong.

This matters because Phase 29 (the reconciler) reads `26-CONTEXT.md` as phase context
when it is planned, and would otherwise inherit language that contradicts the settled
verdict it is supposed to implement — the same kind of stale-upstream-wording risk
`26-VERIFICATION.md` already caught once for `ROADMAP.md`'s Phase 29 success criterion 2
(fixed by this same plan, 26-05).

`26-DECISION.md` and `docs/design/canonical_record_spike.rst` — the two documents this
spike phase actually produces and that Phases 27-29 are meant to consume — are **already
corrected**. This todo is about the archival planning document only:
`26-CONTEXT.md`, which was written before the domain correction and is a discussion
record of what was believed at the time, not a live deliverable.

## Solution

TBD. Options to weigh:

- Leave `26-CONTEXT.md` as an archival record of the discussion that took place before
  the domain correction, and instead add a short dated note at the top of its D-11
  section pointing forward to `26-DECISION.md`'s Domain-correction section and
  `#### Queue-run projection — settled` verdict as the current truth — preserving the
  historical record of what was believed at the time (useful context for why the
  correction was needed) without silently rewriting history.
- Or edit the "4 uncovered nights" language in place to describe the corrected framing
  directly, at the cost of losing the as-discussed historical record of the original
  (pre-correction) reasoning that led to the D-11 prototype's construction.
- Either way, the edit should point a reader at `26-DECISION.md`'s Domain-correction
  section and its `### SPIKE-03 gap closure` / `#### Queue-run projection — settled`
  Findings as the current, settled truth — not restate or re-derive the correction here.
- Note this todo does not itself change `26-DECISION.md` or `docs/design/
  canonical_record_spike.rst`, both of which are already corrected as of plan 26-05.

Once Phase 26's phase directory is archived to `.planning/phases-archive/` (per this
project's milestone-archival convention), this todo's `files` path should be updated to
match — check there first if the original path no longer resolves.
