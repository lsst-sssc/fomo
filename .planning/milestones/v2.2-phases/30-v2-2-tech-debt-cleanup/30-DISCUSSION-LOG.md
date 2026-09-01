# Phase 30: v2.2 Tech-Debt Cleanup - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-31
**Phase:** 30-v2-2-tech-debt-cleanup
**Areas discussed:** Area selection, Attribution filter semantics, Phase scope, Ruff disposition, WR-01 guard, Nyquist validation, Paired docs, Ruff root cause, Recording the closed items, Notebook staging, ROADMAP goal

---

## Pre-discussion scouting

Before generating gray areas, the three stated goal items were checked against the code. Two
were found already closed, which reshaped every question that followed:

| Item | Finding |
|---|---|
| Ruff drift | Repo clean under pinned ruff 0.2.1 (0 lint errors, 89/89 formatted). Installed dev ruff is 0.15.20, which produces 1 × D103 + 4 reformats. |
| WR-09 | Review's suggested text verbatim at `docs/runbooks/telescope_runs_calendar.rst:305-308`. |
| WR-10 | Covered by runbook `:347-357`, `admin.py` docstring, and `test_admin.py:946`. |
| IN-02 | Genuinely open; sibling `_eligible_runs_for_record` has the same gap, unmentioned by the audit. |

---

## Area selection

| Option | Description | Selected |
|--------|-------------|----------|
| Ruff version resolution | Bump the pin, pin the dev env, or document the mismatch | |
| Attribution filter semantics | Which statuses stay eligible; whether the record gate is included | ✓ |
| What replaces the closed items | WR-01, VALIDATION.md files, cosmetic bookkeeping, or narrow to IN-02 | ✓ |
| Paired-docs scope | Whether notebook + runbook are needed for the filter change | ✓ |

**User's choice:** Three of four areas. Ruff deselected as a discussion topic but later given
a disposition, then reopened once the root cause (CLAUDE.md's documented bare command) surfaced.

---

## Attribution filter semantics — statuses

| Option | Description | Selected |
|--------|-------------|----------|
| APPROVED only | Strictest read of IN-02; a run must clear review before being suggested | |
| APPROVED + PENDING_REVIEW | Exclude REJECTED only; a pending submission stays linkable | ✓ |
| APPROVED + flagged pending | Same eligibility, pending marked in queue and hint | |

**User's choice:** First selected "Other — need a short summary of what's behind these choices",
then chose APPROVED + PENDING_REVIEW after the background was provided.
**Notes:** The summary supplied the three-status vocabulary, 26-DECISION Criterion 1's dual
meaning of APPROVED, and a live count of the dev DB (48 runs: 46 approved, 2 rejected legacy,
0 pending_review) — establishing that the APPROVED-only distinction is inert on current data
and is purely forward-looking policy.

---

## Attribution filter semantics — gate scope

| Option | Description | Selected |
|--------|-------------|----------|
| Fix both gates | `_eligible_runs_for_event` and `_eligible_runs_for_record` | ✓ |
| Event gate only | Stay literally within the logged finding | |

**User's choice:** Fix both gates.

---

## Attribution filter semantics — where to filter

| Option | Description | Selected |
|--------|-------------|----------|
| Filter at the eligibility gate | Dashboard counts change consistently | ✓ |
| Filter at the suggestion surfaces only | Counts untouched, filter applied in several places | |

**User's choice:** Filter at the gate.
**Notes:** Accepted that `unattributable_orphan_count` and `event_attribution_backlog` shift as
a result. A related sub-question resolved from the code rather than asked:
`orphan_calendar_events()` returns only un-attributed events, so an already-confirmed link to a
later-rejected run is unaffected.

---

## Phase scope — what replaces the closed items

| Option | Description | Selected |
|--------|-------------|----------|
| WR-01 telescope_class re-import guard | Last WARNING-severity item in the audit | ✓ |
| Reconcile the 5 VALIDATION.md files | Nyquist coverage TODO | ✓ |
| Cosmetic bookkeeping | Reconciler docstrings + 26-DECISION.md header | ✓ |
| Nothing — narrow to IN-02 | Smallest honest phase | |

**User's choice:** All three additions.

---

## Ruff disposition (first pass)

| Option | Description | Selected |
|--------|-------------|----------|
| Record as closed, no code change | Document the environment artifact | |
| Record as closed + pin the dev environment | Add a ruff pin to dev extras | ✓ |
| Bump the pinned ruff instead | Turns phantom drift into real work | |

**User's choice:** Record as closed + pin the dev environment.

---

## WR-01 guard semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror the site guard | Preserve when the CSV cell didn't resolve, plus stderr reporting | ✓ |
| Never overwrite once set | Existing value always wins | |
| Overwrite but report it | Observability only, no behaviour change | |

**User's choice:** Mirror the site guard.
**Notes:** Grounded on the finding that `:371-378` already guards blanking; the residual is a
hand-corrected value being replaced by a different derived one.

---

## Nyquist validation handling

| Option | Description | Selected |
|--------|-------------|----------|
| Its own plan, run the 5 commands | Closes the coverage TODO for real | ✓ |
| Out of scope — do it after | Treat as a milestone-archival step | |
| Just 27.1 (the missing one) | Create only the absent file | |

**User's choice:** Its own plan, running `/gsd-validate-phase` for 26, 27, 27.1, 28, 29.

---

## Paired docs

| Option | Description | Selected |
|--------|-------------|----------|
| Runbook only | Notebook output unchanged since all demo runs are APPROVED | |
| Runbook + notebook cell | Demonstrate the exclusion with real executed output | ✓ |
| Neither — tests only | Risks tripping CLAUDE.md's paired-docs rule | |

**User's choice:** Runbook + notebook cell.

---

## Ruff root cause (reopened)

New evidence surfaced after the first ruff decision: CI runs no ruff at all, `pyproject.toml:42`
carries a bare `"ruff"`, and CLAUDE.md's own Commands section instructs agents to run
`ruff check . --fix` — the unpinned binary. That documented command is the plausible origin of
all three phantom deferrals.

| Option | Description | Selected |
|--------|-------------|----------|
| Pin dev extras + fix CLAUDE.md | Fix both the environment and the instruction | ✓ |
| Pin dev extras only | One-line change, leaves the instruction wrong | |
| Pin + fix CLAUDE.md + add CI job | Most durable, adds a third place to keep in sync | |

**User's choice:** Pin dev extras + fix CLAUDE.md. CI job declined (recorded as deferred).

---

## Recording the closed items

| Option | Description | Selected |
|--------|-------------|----------|
| Amend the milestone audit | Correct the file `/gsd-complete-milestone` reads | ✓ |
| Phase 30 artifacts only | Preserve the audit as a point-in-time snapshot | |
| Both, with a pointer | Keep original text, add "superseded by Phase 30" lines | |

**User's choice:** Amend the milestone audit in place.

---

## Notebook staging

| Option | Description | Selected |
|--------|-------------|----------|
| Add a fifth submission to reject | Leaves the existing four runs' output byte-identical | ✓ |
| Reject one of the existing four | Larger regenerated diff across several cells | |
| Show before/after on one run | Mutates and leaves state mid-notebook | |

**User's choice:** Add a fifth submission.

---

## ROADMAP goal

| Option | Description | Selected |
|--------|-------------|----------|
| Rewrite before planning | Stale goal is an active hazard for the planner | ✓ |
| Leave it — CONTEXT.md supersedes | Two documents disagree about the phase | |
| Rewrite it as a task inside the phase | Planner still reads the stale goal | |

**User's choice:** Rewrite before planning, via `/gsd-phase --edit 30`.

---

## Claude's Discretion

- Test placement and naming for the filter and the guard.
- Whether the two existing rejected legacy runs are used as fixtures or fresh factory rows built.
- The exact `.exclude()` vs `.filter()` formulation and whether a shared constant names the
  eligible statuses.
- Wave ordering and how the six in-scope items are grouped into plans.

## Deferred Ideas

- A ruff job in CI — offered and declined in favour of the two-place fix.
- Locking the relabel-to-web direction (27.1 WR-10's accepted residual).
- 28-REVIEW IN-01's rounded-vs-raw candidate score — inert at current weights.
- 27-REVIEW IN-01 — no query-cap test on the per-render attribution-hint queries.
- Todo reviewed but not folded: "Extract site/telescope mapping and instrument extraction into
  own module" (matched at 0.6) — a structural refactor, its own phase.
