# Phase 30: v2.2 Tech-Debt Cleanup - Context

**Gathered:** 2026-08-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Clear the deferred items that accumulated across the v2.2 milestone so it can be archived
with an accurate record.

**Scouting during discussion found the ROADMAP goal is materially out of date** — two of its
three stated items are already closed. The verified position, which supersedes the goal text:

| Stated item | Actual status |
|---|---|
| Repo-wide ruff/format drift | **Not dirty code.** Under the pinned pre-commit ruff (`v0.2.1`, `.pre-commit-config.yaml:50,58`) `ruff check .` reports 0 errors and `ruff format --check .` reports 89/89 already formatted. The findings logged three times by Phases 26/27/27.1 are an artifact of a dev environment running ruff 0.15.20. |
| WR-09 runbook ambiguous antecedent | **Already fixed.** The review's suggested replacement text is verbatim in place at `docs/runbooks/telescope_runs_calendar.rst:305-308`. |
| WR-10 residual-consequence documentation | **Already fixed.** Covered by the `.. warning::` block at `docs/runbooks/telescope_runs_calendar.rst:347-357`, the matching section in `solsys_code/admin.py`'s `get_readonly_fields` docstring, and the pinning test `solsys_code/tests/test_admin.py:946` (`test_relabel_to_web_locks_the_row_and_cannot_be_undone`). |
| 27-REVIEW IN-02 attribution filter | **Genuinely open.** `campaign_attribution.py:473-486` filters on campaign only. |

**In scope:**

1. The IN-02 `approval_status` filter on attribution eligibility.
2. WR-01's `telescope_class` re-import guard.
3. The ruff root-cause fix (a version pin and a corrected instruction — not a reformat).
4. Reconciling the five phase `VALIDATION.md` files.
5. Cosmetic bookkeeping (reconciler docstrings, `26-DECISION.md` header).
6. Amending `.planning/v2.2-MILESTONE-AUDIT.md` so the closed items stop being re-flagged.

**Out of scope:** any repo-wide reformat; re-doing WR-09/WR-10; new attribution scoring
behaviour; extracting the site/telescope mapping module (see Deferred).

</domain>

<decisions>
## Implementation Decisions

### Attribution eligibility (27-REVIEW IN-02)

- **D-01:** Exclude `REJECTED` only. `APPROVED` and `PENDING_REVIEW` both stay eligible as
  suggested run matches. Rationale from discussion: a rejected run being offered is the
  unambiguous defect; an orphan calendar event matching a *pending* web submission is useful
  evidence that the submission is genuine, so hiding pending runs would cost staff a
  two-step at review time. Grounding: the dev DB holds 48 runs — 46 `approved`, 2 `rejected`
  (both `legacy`), 0 `pending_review` — so the filter's only observable effect on today's
  data is dropping those 2 rejected rows.

- **D-02:** Apply the same filter to **both** eligibility gates —
  `campaign_attribution._eligible_runs_for_event` (`:473-486`) and
  `_eligible_runs_for_record` (`:489-508`). The audit names only the event gate; the record
  gate has the identical missing filter and was simply not looked at.

- **D-03:** Filter at the eligibility gate, not at the display surfaces. Both gates feed
  `candidates_for_event`/`candidates_for_record`, `event_attribution_backlog`,
  `unattributable_orphan_count` and `_sole_high_candidate_pk`, so the staff dashboard counts
  change with it. This is intended: an orphan whose only candidate was a rejected run
  genuinely has no valid candidate, and one filter in one place cannot drift out of sync the
  way four call-site filters could.
  — **Reversibility:** reversible — a single `.exclude()` per function; no schema change and
  no persisted state depends on it.

### CSV re-import guard (27-REVIEW WR-01)

- **D-04:** Mirror the existing `preserve_site` guard for `telescope_class`, including its
  reporting. `import_campaign_csv.py:371-378` already stops a re-import *blanking* a
  non-blank `telescope_class`, but a hand-corrected value can still be overwritten by a
  different derived one. Match the shape of the site guard at `:351-369`: preserve the
  existing value when the CSV row's own cell did not genuinely resolve, and emit a
  per-row `stderr` line naming what was kept and what was discarded — the site guard's
  "say so" behaviour, which exists precisely so a silently-dropped operator correction is
  visible in the command output.

### Ruff toolchain (root cause, not a cleanup)

- **D-05:** Do **not** run `ruff check . --fix && ruff format .`. The repo is already clean
  under the pinned version; formatting under 0.15.20 would commit a diff the pinned
  pre-commit hook never asked for.

- **D-06:** Pin the dev environment: change `pyproject.toml:42` from bare `"ruff"` to a pin
  matching `.pre-commit-config.yaml`'s `v0.2.1`. The bare dependency is how this environment
  reached 0.15.20.

- **D-07:** Fix the instruction that caused the phantom deferrals. `CLAUDE.md`'s Commands
  section documents `ruff check . --fix` / `ruff format .` as the quality gate; every
  executor followed it and got the unpinned binary. Change it to invoke the gate through
  pre-commit (e.g. `pre-commit run ruff --all-files`) so the documented command and the
  enforced version are the same thing.
  — **Reversibility:** reversible — documentation and a dependency specifier; no code path
  depends on either.

  Noted during discussion, not acted on: **CI does not run ruff at all** — no workflow under
  `.github/workflows/` references it, so pre-commit is the only enforcement. Adding a CI job
  was offered and declined in favour of the two-place fix; see Deferred.

### Nyquist validation coverage

- **D-08:** Reconcile all five phase `VALIDATION.md` files as its own plan — run
  `/gsd-validate-phase` for 26, 27, 27.1, 28 and 29 and commit the promoted files. Phases
  26/27/28/29 carry `status: draft` files never promoted by validate-phase; 27.1 has none at
  all. This is GSD process work touching `.planning/` only, and should be planned as a
  separate unit from the code changes so its cost is visible.

### Bookkeeping and the record

- **D-09:** Amend `.planning/v2.2-MILESTONE-AUDIT.md` in place with a resolution status for
  each closed item, citing where it was closed (`docs/runbooks/telescope_runs_calendar.rst`
  `:305-308` and `:347-357`, `solsys_code/admin.py` `get_readonly_fields` docstring,
  `solsys_code/tests/test_admin.py:946`, and the pinned-ruff evidence). This file is what
  `/gsd-complete-milestone` reads, so correcting it there is what actually stops the
  re-flag — Phase 30's own SUMMARY would not be consulted.

- **D-10:** Clean the cosmetic items: the docstring mentions of the deleted
  `_project_calendar_event` / `_calendar_event_title` at `solsys_code/campaign_reconciler.py`
  `:78, :100, :173, :195, :339`, and `26-DECISION.md`'s header preamble, which still says it
  was "built up across plans 26-01, 26-02 and 26-03" without 26-04/26-05.

- **D-11:** Rewrite ROADMAP.md's Phase 30 goal via `/gsd-phase --edit 30` **before**
  planning, restating it as the six in-scope items. The planner reads ROADMAP.md as required
  reading, so leaving a goal that describes already-completed work is an active hazard — it
  could plan work that is already done. Phase 30 is still `Pending`, so the edit needs no
  `--force`.

### Paired docs (CLAUDE.md rule)

- **D-12:** Both paired artifacts are in scope for the attribution filter and belong in
  `files_modified` from the start:
  - `docs/runbooks/telescope_runs_calendar.rst` — extend the attribution section at
    `:187-208`, which already explains what the queue does and does not show, with the fact
    that a rejected run is never offered as a match.
  - `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` — add a **fifth** public
    submission that gets `action=reject` instead of `approve`, used only by the attribution
    cell, and show it absent from the candidate list. A fifth run is chosen deliberately
    over rejecting one of the existing four: the four are asserted on by the approve-all,
    calendar-projection and public-table cells, so reusing one would churn a large amount of
    executed output for no extra demonstration. Regenerate with
    `jupyter nbconvert --to notebook --execute --inplace` and commit with output.

### Claude's Discretion

- Test placement and naming for the filter and the guard. Convention is settled:
  DB-dependent tests live in `solsys_code/tests/` and run under
  `python manage.py test solsys_code` (note `python manage.py`, not `./manage.py`).
- Whether the two pre-existing `rejected` legacy runs in the dev DB are used as fixtures or
  fresh factory rows are built. Per CLAUDE.md, any `Target` fixture must use
  `NonSiderealTargetFactory`.
- The exact `.exclude()` vs `.filter()` formulation, and whether a shared module-level
  constant names the eligible statuses.
- Wave ordering and how the six items are grouped into plans.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The source of scope

- `.planning/v2.2-MILESTONE-AUDIT.md` §"Tech Debt" — the origin of every item in this phase,
  and the file D-09 amends. Note its WR-09/WR-10/ruff entries are **stale**; this CONTEXT.md
  supersedes them.
- `.planning/phases/27.1-close-gap-staff-surfaces-and-data-integrity-risks-from-the-c/27.1-REVIEW.md`
  §WR-09 (`:185`), §WR-10 (`:215`) — the original findings, both verified already closed.
- `.planning/phases/27.1-close-gap-staff-surfaces-and-data-integrity-risks-from-the-c/27.1-VERIFICATION.md`
  `:148-149` — the deferral records for WR-09/WR-10.

### Attribution (IN-02)

- `solsys_code/campaign_attribution.py` `:435-458` (`orphan_calendar_events`), `:473-486`
  (`_eligible_runs_for_event`), `:489-508` (`_eligible_runs_for_record`), `:511-555`
  (`candidates_for_event`) — the gates being changed and the docstrings recording why each
  is deliberately permissive. `_eligible_runs_for_record`'s docstring carries a standing
  prohibition: it must **not** additionally require target-FK equality.
- `solsys_code/campaign_views.py` `:1101,1107` — the queue consumers.
- `solsys_code/models.py` `:83-120, :220-223` — the `ApprovalStatus` vocabulary and
  26-DECISION Criterion 1 (`APPROVED` + `source != WEB` means "no approval was required",
  a different fact from "a human approved this").

### CSV re-import (WR-01)

- `solsys_code/management/commands/import_campaign_csv.py` `:336-378` — the `preserve_site`
  guard D-04 mirrors, including its `stderr` reporting and the `telescope_class` pop that
  already handles the blanking half.

### Ruff toolchain

- `.pre-commit-config.yaml` `:48-62` — the pinned `v0.2.1` for both `ruff` and `ruff-format`.
  Note the lint hook is `types_or: [python, pyi]` while the format hook adds `jupyter`.
- `pyproject.toml` `:42` — the bare `"ruff"` dev dependency D-06 pins.
- `CLAUDE.md` §Commands — the `ruff check . --fix` instruction D-07 corrects.

### Paired docs

- `CLAUDE.md` §Conventions "Paired docs are part of the deliverable" — the rule that puts
  D-12's two artifacts in `files_modified` up front rather than as follow-ups.
- `docs/runbooks/telescope_runs_calendar.rst` `:187-208` (attribution section, to extend),
  `:292-357` (the source-lock section, already correct — do not re-edit).
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` cells 19-22 — the existing
  orphan-event and attribution-confirm cells the new cell sits beside.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`preserve_site` guard** (`import_campaign_csv.py:336-369`): a complete worked example of
  the exact behaviour D-04 wants — condition computed up front so a second derivation can
  gate on the same decision, fields popped as a unit, a counter, and a per-row `stderr`
  line. D-04 should follow its shape rather than invent a new one.
- **`SourceProvenanceLockTests` / `SourceProvenanceTwoStepBypassTests`**
  (`test_admin.py:837, :998`): the established pattern for pinning an admin-side invariant
  with a non-vacuous control test.
- **`test_campaign_attribution.py` `:257-265, :420`**: already asserts on candidate run pks
  for events and records — the natural home for the D-01/D-02 assertions.

### Established Patterns

- **Eligibility gates are deliberately permissive.** Both `_eligible_runs_*` docstrings state
  that downstream filters (dismissals, the zero-score drop, D-03's "must have at least one
  candidate") do the narrowing. D-03 adds `approval_status` as a *hard* gate, which is a
  deliberate departure — the plan should say so in the docstring, as the surrounding code
  does for every other such choice.
- **Findings are answered in the docstring, by ID.** `campaign_attribution.py` cites
  `28-REVIEW.md IN-01`, `D-11`, `D-03`; `admin.py` cites `WR-10`; `import_campaign_csv.py`
  cites `WR-01`, `CR-01`, `WR-04`. New code should cite `27-REVIEW IN-02` and `WR-01` the
  same way.
- **`telescope_class` is never cleared once set** (`models.py:207-219`) — a standing
  invariant, reaffirmed when the user rejected Phase 27 code-review finding CR-01. D-04 must
  not weaken it.

### Integration Points

- Changing the two gates propagates to the attribution queue view, the backlog builder, the
  unattributable count and the calendar-event modal's staff hint. Existing confirmed links
  are **not** affected: `orphan_calendar_events()` (`:435-458`) returns only un-attributed
  events, so a run rejected after its link was confirmed keeps that link.
- Test-suite gotcha carried forward: run `python manage.py test solsys_code`, and exclude
  `test_views.TestEphemeris` — it segfaults in native ASSIST.

</code_context>

<specifics>
## Specific Ideas

- The ruff evidence the plan should be able to reproduce: with the pinned binary at
  `~/.cache/pre-commit/repoz9dc1u5l/py_env-python3.11/bin/ruff` (0.2.1),
  `ruff check .` → no output, `ruff format --check .` → "89 files already formatted". With
  the environment's 0.15.20, `ruff check .` → 1 × `D103` in
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` cell 6, and
  `ruff format --check .` → 4 files would reformat. This contrast is the finding D-09
  records.
- The notebook's fifth run should be visibly labelled as existing for the rejection demo, so
  a reader does not mistake it for part of the four-run lifecycle narrative.

</specifics>

<deferred>
## Deferred Ideas

- **A ruff job in CI.** Offered during discussion and declined in favour of the pin +
  instruction fix. Worth revisiting: with no CI enforcement, the only thing stopping the next
  drift is that developers actually run pre-commit.
- **Locking the relabel-to-web direction** (27.1 WR-10's residual). Deliberately accepted and
  now documented and tested; re-opening it would re-open D-19 for the sources D-19
  legitimately covers.
- **28-REVIEW IN-01's rounded-vs-raw candidate score.** `candidates_for_event` drops on the
  rounded display score; inert at current weights, with a re-tuning trigger already recorded
  in the code comment. Not in this phase.
- **27-REVIEW IN-01**: no query-cap test on the per-render attribution-hint queries. Accepted
  in the audit as a single-event modal render, not a list-page N+1.

### Reviewed Todos (not folded)

- **"Extract site/telescope mapping and instrument extraction into own module"**
  (`.planning/todos/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`,
  matched at score 0.6) — a structural refactor, not a v2.2 deferred item. Phase 30 is scoped
  to closing the milestone's own audit; a refactor of this size belongs in its own phase.

</deferred>

---

*Phase: 30-v2-2-tech-debt-cleanup*
*Context gathered: 2026-08-31*
