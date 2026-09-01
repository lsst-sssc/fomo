---
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [CLAUDE.md]
autonomous: true
requirements: []

must_haves:
  truths:
    - "CLAUDE.md's first (hand-authored) Conventions section documents the demo-notebook-companion convention"
    - "The new bullet lists all three current module/notebook pairs and explains why the convention exists"
  artifacts:
    - path: "CLAUDE.md"
      provides: "Demo notebook companion convention bullet in first Conventions section"
      contains: "Demo notebook companions are part of the deliverable"
  key_links:
    - from: "CLAUDE.md first Conventions section"
      to: "Planning-doc terminology bullet"
      via: "new bullet immediately follows it, before <!-- GSD:project-start -->"
      pattern: "Planning-doc terminology.*Demo notebook companions"
---

<objective>
Add a new convention bullet to CLAUDE.md's first ("hand-authored") Conventions section,
documenting that the demo notebooks paired with `telescope_runs.py`,
`load_telescope_runs.py`, and `sync_lco_observation_calendar.py` are part of the
deliverable and must be scoped into `files_modified` and kept in sync whenever the
paired module's behavior changes.

Purpose: This was missed twice already (Phase 5, fixed via quick task 260619-f7u;
Phase 6, fixed via quick task 260620-v9x) because plans never scoped the paired
notebook into `files_modified` up front. Codifying the convention in CLAUDE.md makes
it visible to every GSD subagent (planner, plan-checker, executor, verifier) touching
these modules, closing the recurring gap at the source instead of catching it after
the fact each time.
Output: Updated CLAUDE.md with the new bullet in place.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add demo-notebook-companion convention bullet to CLAUDE.md</name>
  <files>CLAUDE.md</files>
  <action>
    Edit CLAUDE.md. In the FIRST "## Conventions" section (the hand-authored one that sits
    between the "## Testing" section and the "&lt;!-- GSD:project-start source:PROJECT.md --&gt;"
    marker comment — NOT the second, GSD-managed "## Conventions" section further down that is
    sourced from CONVENTIONS.md and gets regenerated), insert a new bullet immediately after the
    existing "**Planning-doc terminology:**" bullet (the last bullet in that section, ending
    "...they all read this file before producing planning docs.").

    The new bullet must follow the same Markdown bullet style as its neighbors (bold lead-in
    phrase followed by a colon, prose body, backtick-quoted file paths and identifiers) and must
    convey, in this repo's voice:

    1. A lead-in stating that demo notebook companions are part of the deliverable, not optional
       polish that can be added after the fact.
    2. That each of `solsys_code/telescope_runs.py`,
       `solsys_code/management/commands/load_telescope_runs.py`, and
       `solsys_code/management/commands/sync_lco_observation_calendar.py` has a paired demo
       notebook under `docs/notebooks/pre_executed/` — `telescope_runs_demo.ipynb`,
       `load_telescope_runs_demo.ipynb`, and `sync_lco_observation_calendar_demo.ipynb`
       respectively — that must stay in sync with the module's behavior.
    3. That any plan whose tasks change one of these modules' behavior (new extraction logic,
       new parameters, new fixture shapes — not pure refactors or typo fixes) must include its
       paired notebook in `files_modified` and add or update cells exercising the new behavior
       with real executed output, regenerated via
       `jupyter nbconvert --to notebook --execute --inplace` and committed (pre-commit clears
       notebook output everywhere else, but `pre_executed/` copies are committed with output,
       per the existing convention already noted earlier in this section).
    4. That when a new module gets its own demo notebook, this list should be extended.
    5. A note that this gap was hit twice already — Phase 5 (fixed after the fact via quick task
       `260619-f7u`) and Phase 6 (fixed via quick task `260620-v9x`) — both times because the
       plan's `files_modified` never scoped the notebook in.
    6. That this applies to every GSD subagent touching these modules: the planner (scope the
       paired notebook into `files_modified` and into a task up front, not as a follow-up); the
       plan-checker (treat this as part of CLAUDE.md Compliance — flag any plan that modifies one
       of the listed modules' behavior without its paired notebook in `files_modified`); the
       executor (update the notebook as part of plan execution, not as an afterthought); and the
       verifier (treat a missing or stale notebook update as a must-have gap, not a nice-to-have,
       whenever the plan touched one of these modules).

    Do not touch the second "## Conventions" section (the one inside the
    "&lt;!-- GSD:conventions-start source:CONVENTIONS.md --&gt;" / "&lt;!-- GSD:conventions-end --&gt;"
    block) — that block is regenerated by tooling and a hand-edit there would be lost. Do not
    modify any other section of CLAUDE.md.
  </action>
  <verify>
    <automated>grep -n 'Demo notebook companions are part of the deliverable' /home/tlister/git/fomo/CLAUDE.md | head -1</automated>
  </verify>
  <done>
    CLAUDE.md's first Conventions section contains a new bullet (immediately after the
    "Planning-doc terminology" bullet, before the "&lt;!-- GSD:project-start -->" marker) that
    names all three module/notebook pairs, states the files_modified + regeneration requirement,
    references both prior quick-task fixes (260619-f7u, 260620-v9x), and calls out
    planner/plan-checker/executor/verifier responsibilities. The second, GSD-managed Conventions
    section is unchanged. `ruff check .` is unaffected (no Python files touched).
  </done>
</task>

</tasks>

<verification>
Run `grep -n '## Conventions' CLAUDE.md` to confirm there are still exactly two
"## Conventions" headings, and that the new bullet text appears only once, between
the first heading and the "&lt;!-- GSD:project-start -->" marker (i.e. before the second
heading), not after it.
</verification>

<success_criteria>
- New bullet present in the first Conventions section, in the correct position
- Second (GSD-managed) Conventions section is byte-for-byte unchanged
- No other part of CLAUDE.md modified
</success_criteria>

<output>
Create `.planning/quick/260621-foo-add-a-claude-md-convention-requiring-dem/260621-foo-SUMMARY.md` when done
</output>
