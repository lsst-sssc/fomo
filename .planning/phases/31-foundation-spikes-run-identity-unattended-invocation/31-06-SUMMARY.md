---
phase: 31-foundation-spikes-run-identity-unattended-invocation
plan: 06
subsystem: investigation-spike (gap-closure, documentation-only)
tags: [gap-closure, docs, sphinx, uat-followup, facility-scope-correction]
requires:
  - 31-01 (SCHEMA-01/02 evidence)
  - 31-05 (published docs/design page and structural gates this plan re-runs)
  - .planning/debug/gemini-vs-soar-facility-scope.md (diagnosed root cause)
provides:
  - Corrected SCHEMA-02 evidence framing in 31-DECISION.md, qualified in place at all five
    Gemini touch points plus one consolidated correction section
  - The same distinction carried onto docs/design/run_identity_and_unattended_invocation_spike.rst
  - A durable, discoverable pending-todo hand-forward of the three Phase 32/33 consequences
affects:
  - Phase 32 (Adapter Consolidation) — ADAPT-03 and its third success criterion should
    re-target SOAR, per the new pending todo
  - Phase 33 (Outcome Propagation) — Gemini outcome propagation is structurally impossible;
    needs an explicit caveat before OUTCOME-01..04 planning
actuals:
  tokens: 7150
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - "Dated 'Corrected 2026-09-02' inline qualifications beside each original claim,
      matching the document's existing 'Correction, recorded during the code-review fix
      pass' idiom — original text never deleted"
    - "Pending-todo hand-forward with STATE.md surfacing, so a future phase planner meets
      a finding without re-opening a closed phase's UAT file"
key-files:
  created:
    - .planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md
  modified:
    - .planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md
    - docs/design/run_identity_and_unattended_invocation_spike.rst
    - .planning/STATE.md
key-decisions:
  - "Gemini's identity mechanism (source_identifier, the cited sync_gemini_observation_calendar.py:150
    line) is real and correct; only the surrounding premise (Gemini as a queue-visible
    facility) was wrong. Both artifacts now say this explicitly rather than implying the
    command itself is broken."
  - "SOAR_QUEUE's addition to CampaignRun.Source, ADAPT-03's re-target, and Phase 33's
    Gemini caveat are all handed forward as a pending todo rather than actioned here —
    Phase 31 stays investigation-only, per its own already-verified must-have."
requirements-completed: [SCHEMA-02]
coverage:
  - truth: "31-DECISION.md's SCHEMA-02 per-adapter tables distinguish a facility read-back path from a submission-echo path"
    human_judgment: false
    rationale: "Automated gate confirms >=4 'submission-echo' occurrences before the new correction section and >=6 total, at all five named touch points."
  - truth: "Every place in either committed artifact that offers the Gemini ingest path as evidence carries the read-back caveat at that place"
    human_judgment: false
    rationale: "Verified via targeted grep at each of the five 31-DECISION.md touch points and the four run_identity_and_unattended_invocation_spike.rst places named in the plan."
  - truth: "Both committed artifacts name SOAR as the facility with a real, API-backed queue read path, with source evidence"
    human_judgment: false
    rationale: "grep -q 'SOAR' / 'SOARFacility' / 'soar.py:240' / 'ocs.py:1548' all pass on both files."
  - truth: "All four verdicts (SCHEMA-01/02/03, SCHED-07) are unchanged"
    human_judgment: true
    rationale: "The plan's own structural gate (four ### SCHEMA/SCHED verdict subsections, all below ## Recommendation, unchanged count and order) passed automatically; that no verdict's substantive text itself was rewritten is a judgment confirmed during editing — every edit was additive, appended beside existing text, never replacing a verdict paragraph."
  - truth: "A Phase 32/33 planner who never opens 31-UAT.md still finds the facility-inventory consequence through the standard pending-todo surface"
    human_judgment: false
    rationale: "Pending todo file exists at the named path with SOAR_QUEUE/ADAPT-03/G-31-3/debug-session references; STATE.md's Pending Todos list carries a matching bullet."
  - truth: "Phase 31 remains investigation-only"
    human_judgment: false
    rationale: "git status --porcelain -- solsys_code src and -- solsys_code/migrations are both empty after all three tasks."
  - truth: "Sphinx build still succeeds; neither artifact carries a credential or contact value"
    human_judgment: false
    rationale: "sphinx-build printed 'build succeeded, 12 warnings' (same pre-existing warning count as 31-05's build, none new); the credential/hex/email regex scan over all three touched artifacts found nothing."
duration: ~35min
completed: 2026-09-02
status: complete
---

# Phase 31 Plan 06: Gap Closure — Gemini/SOAR Facility Framing (G-31-3) Summary

Closed UAT gap G-31-3: Phase 31's committed artifacts (`31-DECISION.md` and the published
`docs/design/run_identity_and_unattended_invocation_spike.rst`) presented
`sync_gemini_observation_calendar` as one of three facilities FOMO can read a queue from.
It is not — `GEMFacility` has no facility read-back at all, and SOAR is the real second
facility, already folded inside `sync_lco_observation_calendar`. Corrected both artifacts
at all five named touch points, appended a full-evidence correction section, and handed
forward the three consequences that belong to Phase 32/33 as a discoverable pending todo.

## Performance

- Duration: ~35 minutes
- Tasks completed: 3/3
- Files changed: 3 (`31-DECISION.md` corrected, `run_identity_and_unattended_invocation_spike.rst`
  corrected, new pending-todo file created); `STATE.md` gained one bullet

## Accomplishments

- **Task 1** — Corrected `31-DECISION.md` at all five named touch points (Block (E)'s
  per-ingest-path table Gemini row and its closing Tag paragraph; the SCHEMA-02 verdict's
  per-ingest-path table Gemini row, the Phase 32 invariant-test guidance bullet, and its
  closing Tag paragraph), each qualified in place with a dated "Corrected 2026-09-02"
  note carrying the submission-echo distinction — no original claim deleted. Appended a
  new `## Correction (2026-09-02): Gemini is submission-echo, not facility read-back —
  SOAR is the real second facility` section with five subsections (what is/isn't wrong,
  the `GEMFacility` stub evidence, the SOAR read-path evidence, why the probe couldn't
  catch it and how the finding was lost once before, and a three-item forward note for
  Phase 32/33), plus a short pointer added to the document's opening block above
  `## Findings`. All three of the task's automated `<verify>` gates passed on the first
  attempt (evidence citations present; correction section past line 900 with the required
  submission-echo density; the structural regression gate inherited from plan 31-05 still
  holds — 4 verdict subsections, 8 evidence subsections, correct ordering).
- **Task 2** — Applied the same correction to the published design page at all five named
  places: the opening paragraph and Background paragraph no longer pair LCO and Gemini as
  jointly queue-visible; the per-ingest-path table's Gemini key and cardinality are
  qualified in place with all three key values kept unchanged; the credential-handling row
  now names SOAR; a dated correction call-out (matching the page's existing
  `_notify_staff()`/`mail_admins()` correction shape) sits before the identity Decisions
  table, naming both `GEMFacility` and `SOARFacility` with no library line numbers; one
  Future scope bullet records the open facility-targeting question for Phase 32/33.
  `docs/design/design.rst` is untouched. Sphinx build printed `build succeeded, 12
  warnings` — the same warning count as plan 31-05's build (one pre-existing, unrelated
  `campaign_lifecycle_demo` cross-reference warning), confirming no new warning was
  introduced.
- **Task 3** — Created the pending todo
  `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`,
  carrying the debug session and gap G-31-3 pointer, the full facility-class evidence
  (file:line citations for `GEMFacility`'s stubs and `SOARFacility`'s inherited read path),
  the archived-and-lost v1.5 precedent, and three separable forward items (the missing
  `SOAR_QUEUE` source value, the ADAPT-03/Phase-32-success-criterion re-target with the
  internal STATE.md-vs-REQUIREMENTS.md contradiction named, and Phase 33's Gemini
  structural-impossibility caveat). Added exactly one bullet to `STATE.md`'s Pending Todos
  list pointing at it. Re-ran all four of the plan's phase-seal gates: `git status
  --porcelain -- solsys_code src` and `-- solsys_code/migrations` both empty;
  `ADAPT-03`/Phase 32's third success criterion wording both still present unchanged in
  `REQUIREMENTS.md`/`ROADMAP.md`; and the credential/contact regex scan over all three
  touched artifacts found nothing.
- Ran the plan's full end-of-plan `<verification>` section (all 7 items) after all three
  tasks: Sphinx build succeeded; source tree and migrations empty; `design.rst` untouched;
  `ADAPT-03`/roadmap wording present; `31-DECISION.md`'s structural gate passed; credential
  scan clean; the published page names `SOAR` (6 occurrences, up from 0 before this plan).

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Correct the facility framing in 31-DECISION.md | `6553526` | `31-DECISION.md` |
| 2 | Carry the same distinction onto the published design page | `d7dd012` | `docs/design/run_identity_and_unattended_invocation_spike.rst` |
| 3 | Record the Phase 32/33 consequences and re-run the phase-seal gates | `da50ed6` | new pending-todo file, `.planning/STATE.md` |

## Files Created/Modified

- `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
  (modified) — five in-place qualifications plus a new correction section and an opening
  pointer
- `docs/design/run_identity_and_unattended_invocation_spike.rst` (modified) — same
  distinction carried onto the published page, plus a dated correction call-out and one
  Future scope bullet
- `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`
  (new) — durable hand-forward of the three Phase 32/33 consequences
- `.planning/STATE.md` (modified) — one new Pending Todos bullet pointing at the todo file
  above. **Note:** this file also carried pre-existing uncommitted frontmatter/Current
  Position drift (`status`, `last_updated`, `state_head`, `total_plans`, `Plan: N of M`)
  from the orchestrator's shared-artifact bookkeeping ahead of this plan's own execution;
  that drift was already present in the working tree before this plan's Task 3 ran and is
  not this plan's own edit — see Deviations below.

## Decisions Made

- No new decisions were made in this plan. It corrects the evidence framing behind an
  already-settled SCHEMA-02 verdict; the verdict itself (nullable `campaign` FK,
  `source_identifier` field and constraint, classical-tolerance-insufficiency finding, and
  SCHED-07's cron+flock mechanism) is unchanged.

## Deviations from Plan

**Task 3's `git add .planning/STATE.md` staged pre-existing, non-task-related drift
alongside the plan-directed Pending Todos bullet.** `STATE.md`'s frontmatter
(`status`, `last_updated`, `last_activity`, `state_head`, `progress.total_plans`) and its
"Current Position" section were already modified in the working tree before this plan's
Task 3 began — visible in `git status --short` immediately after Task 1's commit, before
any Task 3 edit was made. This is the orchestrator's own shared-artifact bookkeeping
(advancing plan-count/position ahead of dispatching this plan), not content this plan
wrote. Per the task_commit_protocol, staging `.planning/STATE.md` for commit necessarily
includes whatever is currently unstaged in that file — there is no mechanism to stage only
one edit's lines within a single file. This is not a Rule 1-4 deviation: no code was
fixed, no functionality added, and no architectural change made. It is recorded here
because the plan's own acceptance criteria state "nothing else in STATE.md changes" from
this plan's perspective — the one bullet this plan wrote is exactly what Task 3's `<action>`
directed, and the accompanying drift originated outside this plan's edits, before Task 3
ran, and reflects real, accurate state (plan 06 exists, phase is executing).

Otherwise: none — all three tasks' automated `<verify>` gates and `<acceptance_criteria>`
passed on the first attempt; no Rule 1-4 deviation was needed for any task.

## Issues Encountered

None. The pre-commit hook's own Sphinx-build check (triggered by Task 2's commit) passed
independently of Task 2's own explicit `sphinx-build` invocation and the plan's
end-of-plan verification re-run, giving three independent confirmations of the same
`build succeeded` result.

## Known Stubs

None. This plan is documentation/planning-artifact-only; no UI component or data-flow stub
was introduced.

## Threat Flags

None beyond what the plan's own threat model already covers (T-31-20 through T-31-24,
T-31-SC) — all four `mitigate`-disposition threats were addressed by this plan's own
verify gates (credential/contact scan, source-tree emptiness gate, REQUIREMENTS/ROADMAP
wording-preservation gate, and the additive-not-deleting correction discipline), and no
new trust boundary or network/database surface was introduced by a documentation-only
gap-closure plan.

## Next Phase Readiness

Gap G-31-3 is closed at both artifacts it names. Phase 31's four verdicts
(SCHEMA-01/02/03, SCHED-07) are unchanged. The three consequences that belong to Phase 32
and Phase 33 — the missing `SOAR_QUEUE` source value, ADAPT-03's re-target, and Phase 33's
Gemini outcome-propagation caveat — are recorded on the pending-todo surface, not actioned
here; Phase 32 and Phase 33 planning should read
`.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`
before finalizing their own scope. Phase 31 remains investigation-only end to end: no file
under `solsys_code/` or `src/` was ever modified across all six of this phase's plans, and
no migration was added.

## Self-Check: PASSED

- FOUND: `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
  contains `GEMFacility`, `gemini.py:506`, `gemini.py:490`, `soar.py:240`, `ocs.py:1548`,
  `gemini-vs-soar-facility-scope`, `G-31-3`
- FOUND: `docs/design/run_identity_and_unattended_invocation_spike.rst` contains `SOAR`,
  `GEMFacility`, `SOARFacility`, and the submission-echo/read-back distinction
- FOUND: `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`
- FOUND: `.planning/STATE.md` contains `retarget-adapt-03-to-soar`
- FOUND commits `6553526`, `d7dd012`, `da50ed6` (`git log --oneline --all | grep` each)
- CONFIRMED: `sphinx-build` printed `build succeeded, 12 warnings` (matches plan 31-05's
  pre-existing warning count)
- CONFIRMED: `git status --porcelain -- solsys_code src` empty
- CONFIRMED: `git status --porcelain -- solsys_code/migrations` empty
- CONFIRMED: `git status --porcelain -- docs/design/design.rst` empty
- CONFIRMED: `31-DECISION.md`'s structural gate (4 verdict subsections below
  `## Recommendation`, 8 evidence subsections between `## Findings` and
  `## Recommendation`) still passes
- CONFIRMED: credential/email/hex scan over all three touched artifacts found nothing
- CONFIRMED: `ADAPT-03`/Phase 32 success-criterion-3 wording unchanged in
  `REQUIREMENTS.md`/`ROADMAP.md`
