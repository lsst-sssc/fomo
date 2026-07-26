---
phase: quick-260726-kdp
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/runbooks/telescope_runs_calendar.rst
  - CLAUDE.md
  - .planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/DEFERRED.md
autonomous: true
requirements: [DOC-01, DOC-02, DOC-03, DOC-04]

must_haves:
  truths:
    - "An operator reading the runbook learns that load_telescope_runs accepts an optional --campaign, and cannot confuse it with import_campaign_csv's required --campaign"
    - "An operator reading the runbook learns backfill_lco_observation_records exists, what its flags do, and where it sits relative to sync_lco_observation_calendar"
    - "The runbook no longer claims every unplaced LCO record becomes a [QUEUED] banner"
    - "CLAUDE.md's paired-deliverable rule covers any affected page under docs/runbooks/ by directory, not by filename list"
    - "sphinx-build (the same invocation pre-commit runs) emits no warning or error for docs/runbooks/telescope_runs_calendar.rst"
  artifacts:
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      provides: "Operator runbook, corrected and extended"
      contains: "backfill_lco_observation_records"
    - path: "CLAUDE.md"
      provides: "Directory-scoped paired-deliverable rule"
      contains: "docs/runbooks/"
    - path: ".planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/DEFERRED.md"
      provides: "Out-of-scope doc gaps recorded rather than silently dropped"
  key_links:
    - from: "docs/index.rst"
      to: "docs/runbooks/telescope_runs_calendar.rst"
      via: "toctree entry (line 24)"
      pattern: "runbooks/telescope_runs_calendar"
    - from: "docs/runbooks/telescope_runs_calendar.rst body sections"
      to: "command cheat-sheet list-table"
      via: "one table row per body section, in the same order"
      pattern: "list-table"
---

<objective>
Close documentation drift in the telescope-runs-calendar operator runbook, and amend
CLAUDE.md's paired-deliverable rule so `docs/runbooks/` is covered by directory rather
than by a filename enumeration that goes stale.

Purpose: The runbook shipped 2026-07-18. Since then `load_telescope_runs` gained an
optional `--campaign` flag (quick task `260723-02e`), `backfill_lco_observation_records`
was added and extended three times without ever being documented, and the `[QUEUED]`
title rule was narrowed. The runbook's own paired-deliverable rule in CLAUDE.md could
not have caught any of this, because that rule is a list of four notebooks written
before the runbook existed.

Output: A corrected and extended runbook, a directory-scoped CLAUDE.md rule, and a
DEFERRED.md recording the doc gaps deliberately left out of scope.

Documentation-only. No source code changes, no notebook regeneration.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@docs/runbooks/telescope_runs_calendar.rst
@docs/index.rst
@solsys_code/management/commands/load_telescope_runs.py
@solsys_code/management/commands/backfill_lco_observation_records.py
@solsys_code/management/commands/sync_lco_observation_calendar.py
</context>

<house_rules>
These apply to every task in this plan. Violating one fails the task.

1. **Match the runbook's existing voice exactly.** Task-oriented `How do I ...?`
   headings; `.. code-block:: console` blocks whose commands are prefixed `>> `;
   section underlines made of `-` for sections and `^` for subsections, and at least
   as long as the heading text. Do not restructure, re-voice, or reflow any existing
   section beyond the specific corrections named below.
2. **Body order and cheat-sheet order must stay in lockstep.** The `list-table` rows
   currently appear in exactly the same order as the body sections. Any inserted
   section must get its row inserted at the matching position.
3. **Do not touch `src/fomo/settings.py`.** It carries uncommitted user-local edits.
   Never stage, commit, revert, or reformat it.
4. **No repo-wide ruff.** No `.py` file is expected to change. If one somehow does,
   run ruff scoped to that single file path only. Never `ruff check .` or
   `ruff format .` — the tree has pre-existing unrelated violations that are not
   this task's business.
5. **Do not regenerate, re-execute, or open any notebook.** The notebook side of this
   is already correct. `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`
   already documents `--campaign` well; use it as a reference for tone if helpful,
   but do not modify it.
6. **No CLAUDE.md paired-notebook obligation is triggered by this plan.** The rule
   binds plans that change a listed *module's behavior*. This plan changes zero
   Python. Stated here so the plan-checker does not flag a missing notebook.
7. **Do not add `fetch_jplsbdb_objects` to this runbook.** It is a JPL target-ingest
   command, outside the telescope-runs-calendar operator story. It belongs in
   DEFERRED.md.
</house_rules>

<tasks>

<task type="auto">
  <name>Task 1: Correct load_telescope_runs --campaign and the [QUEUED] claim (DOC-01, DOC-03)</name>
  <files>docs/runbooks/telescope_runs_calendar.rst</files>
  <action>
Three surgical edits to the existing runbook. Verified against source before planning —
implement as described; do not re-derive.

**Edit A — "How do I load a classical telescope schedule?" (currently ~lines 16-28).**
Keep the existing two paragraphs intact. After them, add a short paragraph plus an
extra line in the existing `code-block:: console` (or a second code-block, whichever
reads cleaner) covering the optional `--campaign` flag. The behavior, read from
`load_telescope_runs.Command._resolve_campaign` and `handle`:

  - `--campaign NAME` is **optional**. It names a campaign (a `tom_targets.TargetList`)
    by exact name.
  - When given, every CalendarEvent the file creates or updates is associated with that
    campaign.
  - When omitted, no campaign association is set on any event — which is exactly the
    behavior this command had before the flag existed.
  - The name is resolved once, up front, before any schedule line is processed, so an
    unknown or ambiguous campaign name fails immediately rather than half-way through
    the file.

  Include a worked example line in the console block, matching the existing `>> ` prompt
  style, e.g. loading a schedule under a named campaign.

  **The disambiguation is the point of this edit.** `import_campaign_csv`, documented
  further down the same page, also has a `--campaign` — but there it is *required*.
  A reader who skims both will otherwise conflate them. Make the optionality here
  unmistakable in the prose (the word "optional" should appear, not merely be implied),
  and say explicitly what omitting it does. Do not edit the `import_campaign_csv`
  section itself.

**Edit B — cheat-sheet row for `load_telescope_runs` (currently ~lines 148-150).**
The Key flags cell currently reads ```<filepath>`` (positional)` only. Extend it to
also list ```--campaign <name>`` (optional)`. Mirror the formatting of the
`import_campaign_csv` row two rows below, which uses the `(required)` / `(positional)`
parenthetical idiom — but the parenthetical here must read `(optional)`, so the table
alone makes the required-vs-optional contrast visible without reading the body.
Preserve the `list-table` structure exactly: `:widths: 30 30 40`, one `* - ` per row,
`     - ` continuation cells.

**Edit C — the `[QUEUED]` sentence in "How do I sync LCO/SOAR queue observations?"
(currently ~lines 33-38).** The runbook says a record still awaiting placement becomes
a `[QUEUED]` banner. Since commit `1595619` that is no longer unconditionally true —
this is a genuine, user-visible mismatch, not merely terse prose.

  Actual rule, from `sync_lco_observation_calendar._title_for`: a record gets `[QUEUED]`
  only when it has no resolved `scheduled_start` **and** its status is not already a
  successful terminal state. A record whose status is a successful terminal state (for
  example `COMPLETED`) never gets `[QUEUED]`, even if no placement block was ever
  resolved for it — it would be misleading to banner an observation that has already
  happened as still queued.

  Add this as one or two sentences to the existing paragraph. Keep it tight. Do **not**
  expand this edit into general coverage of the other title prefixes
  (`[EXPIRED]` / `[FAILED]` / `[CANCELLED]` / `[UNVERIFIED]`) — those are an omission,
  not a contradiction, and are recorded in DEFERRED.md by Task 3.
  </action>
  <verify>
    <automated>
LOG=$(mktemp)
sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D "exclude_patterns=notebooks/*,_build" 2>&1 | tee "$LOG"
! grep -Ei 'runbooks/telescope_runs_calendar[^ ]*:.*(warning|error|severe)' "$LOG"
grep -c 'campaign' docs/runbooks/telescope_runs_calendar.rst
grep -q 'optional' docs/runbooks/telescope_runs_calendar.rst
grep -q 'load_telescope_runs --campaign\|load_telescope_runs .*--campaign' docs/runbooks/telescope_runs_calendar.rst
    </automated>
  </verify>
  <done>
The runbook's classical-schedule section documents `--campaign` as optional and states
what omitting it does; the cheat-sheet row for `load_telescope_runs` lists
``--campaign <name>`` (optional); the LCO sync section no longer claims every unplaced
record becomes `[QUEUED]`. The pre-commit Sphinx build emits no warning or error for
this file. No other section's prose was reflowed.
  </done>
</task>

<task type="auto">
  <name>Task 2: Document backfill_lco_observation_records (DOC-02)</name>
  <files>docs/runbooks/telescope_runs_calendar.rst</files>
  <action>
Add a new task-oriented body section and a matching cheat-sheet row for
`backfill_lco_observation_records`, which has never been documented (added 2026-07-19,
one day after the runbook shipped, then extended three times).

**Placement.** Insert the body section immediately *after* "How do I sync LCO/SOAR queue
observations?" and *before* "How do I sync Gemini queue observations?" — this keeps the
LCO material together and puts the record-creating command next to the command that
projects those records onto the calendar. Insert the cheat-sheet row at the matching
position: directly after the ``sync_lco_observation_calendar`` row (house rule 2).

Suggested heading, in the page's existing voice:
`How do I backfill ObservationRecords for LCO observations submitted outside FOMO?`

**Content**, read from `backfill_lco_observation_records.py` — this is the command's
real current behavior including all three later extensions, so write from this list
rather than from the extension commit messages:

  - *What it does.* Queries the LCO Observation Portal's "Get All RequestGroups" API for
    a proposal, keeps only RequestGroups whose name starts with `--name-prefix`, and
    creates one `ObservationRecord` per child request. A request that already has an
    ObservationRecord is skipped, so the command is safe to re-run.
  - *Why it exists / where it fits.* It creates the ObservationRecords for observations
    submitted directly at the LCO portal rather than through FOMO. Those records are
    what `sync_lco_observation_calendar` then projects onto the calendar — so run this
    first, then the sync. One sentence; cross-reference the LCO sync section above.
  - *Required flags.* `--proposal <code>` (exact match) and `--name-prefix <string>`.
  - *`--campaign <name>`.* Optional. Each request's target is matched **by name** against
    the Targets already belonging to this campaign; a request whose target is not a
    member is skipped and logged, never guessed at. If the flag is omitted the command
    prints the available campaigns and prompts for a selection interactively — note this
    is a different meaning of "omitted" from `load_telescope_runs --campaign`, where
    omitting it means "no campaign".
  - *`--create-missing-targets`.* Opt-in, default off. Changes the unmatched-target case
    from "skip" to: reuse an existing Target of that name if one exists anywhere in FOMO,
    otherwise build a new SIDEREAL field Target from the request's own RA/Dec — carrying
    across epoch, proper motion, and parallax when the request supplies them — then add
    it to the campaign and process the request normally. A *reused* Target is left
    untouched; only newly built ones get those fields populated. Say this explicitly:
    it is the single most surprising detail of the flag.
  - *`--username <user>`.* Optional. Attributes created records to that user; default is
    unattributed. An unknown username is a hard error.
  - *`--dry-run`.* Reports what would be created — including which field Targets would be
    created versus reused — without writing anything. Recommend running it first, in the
    same spirit as the `backfill_range_calendar_events` section already does.
  - *Status refresh.* Immediately after each new record is saved (non-dry-run only), the
    command makes one live best-effort status call to LCO so the record's status,
    `scheduled_start`, and `scheduled_end` are populated right away instead of staying
    unset until the next poll. If that call fails it is logged and counted, never fatal,
    and the already-created record is not rolled back.
  - *Summary line.* Reproduce the real final summary in a `console` block so an operator
    can recognize it, with its counters: created (or "would create" under `--dry-run`),
    already existed, unmatched target, no usable configuration, created field targets,
    status sync failed.

Include at least one `.. code-block:: console` with the `>> ` prompt, showing a dry-run
invocation and a real invocation, in the style of the existing sections.

**Cheat-sheet row.** Command ``backfill_lco_observation_records``; Key flags cell listing
``--proposal <code>`` and ``--name-prefix <str>`` as (required) and the rest as
(optional); a one-line description in the style of the existing rows.

**Do NOT add any Troubleshooting entries for this command.** That section is explicitly
scoped to failure modes actually observed against real data, and inventing speculative
ones for a newly documented command would degrade the section's value. The skip-and-log
and counter behavior belongs inline in the new body section above, as behavior
description, not as a troubleshooting claim. Leave the existing Troubleshooting section
byte-identical.
  </action>
  <verify>
    <automated>
LOG=$(mktemp)
sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D "exclude_patterns=notebooks/*,_build" 2>&1 | tee "$LOG"
! grep -Ei 'runbooks/telescope_runs_calendar[^ ]*:.*(warning|error|severe)' "$LOG"
test "$(grep -c 'backfill_lco_observation_records' docs/runbooks/telescope_runs_calendar.rst)" -ge 3
grep -q 'create-missing-targets' docs/runbooks/telescope_runs_calendar.rst
grep -q 'name-prefix' docs/runbooks/telescope_runs_calendar.rst
! grep -q 'fetch_jplsbdb_objects' docs/runbooks/telescope_runs_calendar.rst
test "$(grep -c '^   \* - ``' docs/runbooks/telescope_runs_calendar.rst)" -eq 6
    </automated>
  </verify>
  <done>
The runbook has a `How do I ...?` section for `backfill_lco_observation_records` sitting
between the LCO sync and Gemini sync sections, covering both required flags, all four
optional flags (including the reuse-vs-build distinction for
`--create-missing-targets`), the post-create status refresh, and the summary counters.
The cheat-sheet has 6 command rows (was 5), with the new row directly after
``sync_lco_observation_calendar``. The Troubleshooting section is unchanged. The
pre-commit Sphinx build emits no warning or error for this file.
  </done>
</task>

<task type="auto">
  <name>Task 3: Broaden the CLAUDE.md paired-deliverable rule to docs/runbooks/, record deferrals (DOC-04)</name>
  <files>CLAUDE.md, .planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/DEFERRED.md</files>
  <action>
**Part A — rewrite the CLAUDE.md rule (currently lines 106-129, the bullet beginning
"**Demo notebook companions are part of the deliverable**").**

The rule is currently an *enumeration*: four named modules, four named notebooks. It
could not have caught this task's drift, because `docs/runbooks/` did not exist when the
rule was written. Change the scope from a filename list to a directory-scoped,
behavioral rule. Retitle the bullet lead-in to something like
`**Paired docs are part of the deliverable**`.

The rewritten rule MUST still carry everything the current one gets right — do not lose
any of these:

  - The module-to-notebook pairing map (which notebook pairs with which module) — this
    is genuinely load-bearing information, keep it, just make clear it is a *pairing
    reference*, not the definition of the rule's scope.
  - The trigger condition: a plan whose tasks change a module's *behavior* (new
    extraction logic, new parameters, new fixture shapes — **not** pure refactors or
    typo fixes).
  - The "in `files_modified` up front, not as a follow-up" requirement.
  - The notebook regeneration mechanics: cells exercising the new behavior with real
    executed output, regenerated via
    `jupyter nbconvert --to notebook --execute --inplace` and committed, because
    pre-commit clears notebook output everywhere except `pre_executed/`.
  - The four named GSD subagent roles with their distinct duties: planner (scope it in
    up front), plan-checker (treat as CLAUDE.md Compliance and flag plans that miss it),
    executor (do it during execution, not after), verifier (treat missing/stale as a
    must-have gap, not a nice-to-have).
  - The breach history, as evidence the rule needs teeth.

What MUST change:

  - Scope becomes: the paired pre-executed demo notebook **and** any page under
    `docs/runbooks/` whose documented behavior the change affects.
  - The `docs/runbooks/` half must be phrased **by directory, not by filename**, so a
    future second runbook page is covered automatically with no list to keep in sync.
    Say that intent explicitly in one short clause — it is the whole point of the
    amendment. There is exactly one file there today
    (`docs/runbooks/telescope_runs_calendar.rst`, wired into the Sphinx toctree at
    `docs/index.rst:24`); mention it as the current instance, not as the scope.
  - Add this task to the breach history: the rule has now been breached a third way
    (quick task `260726-kdp`) — the operator runbook went stale because it was never on
    the list, alongside the existing Phase 5 (`260619-f7u`) and Phase 6 (`260620-v9x`)
    breaches.

**Keep it tight.** This is a bullet in a Conventions list, not a docs-policy essay. The
net line growth of CLAUDE.md must not exceed 15 lines (gated below). Match the
surrounding bullets' style: bolded lead-in, `- ` bullet with continuation lines indented
two spaces, ~120 column wrap, backticked paths.

Change nothing else in CLAUDE.md.

**Part B — write DEFERRED.md** at
`.planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/DEFERRED.md`.

Record the doc gaps found during the Finding 3 audit that were deliberately left out of
scope, each with a one-line rationale, so they are visible rather than silently dropped.
The audit rule applied was: correct the runbook only where existing prose is now
*false*; where the runbook is merely *silent*, defer. Items:

  1. `fetch_jplsbdb_objects` has no operator documentation anywhere. Excluded by the task
     brief — it is a JPL target-ingest command, not part of the telescope-runs-calendar
     operator story, so it does not belong in this runbook. Would need its own page.
  2. `7b1e873` (`260722-uyz`) — `sync_lco_observation_calendar` now sets each
     CalendarEvent's campaign association from the record's Target's campaign. The
     runbook is silent on this rather than wrong about it: omission, not contradiction.
  3. `83d024c` (`260722-hpw`) — `import_campaign_csv` now scans the first several leading
     rows for the real header before reading data, tolerating a title/blank row above the
     header, and fails fast with a clear error if no header is found. Omission, not
     contradiction.
  4. The LCO sync's other title prefixes — `[EXPIRED]`, `[FAILED]`, `[CANCELLED]` for
     terminal-failure statuses and `[UNVERIFIED]` for an unresolved telescope label — are
     undocumented in the runbook body (`[UNVERIFIED]` appears only in Troubleshooting).
     Omission, not contradiction; deliberately not expanded during the `[QUEUED]` fix in
     Task 1 to keep that correction surgical.

Keep it to a short markdown list. No frontmatter needed.
  </action>
  <verify>
    <automated>
grep -v '^#' CLAUDE.md | grep -c 'docs/runbooks/'
grep -v '^#' CLAUDE.md | grep -q 'plan-checker'
grep -v '^#' CLAUDE.md | grep -q 'jupyter nbconvert --to notebook --execute --inplace'
grep -v '^#' CLAUDE.md | grep -q 'pre_executed'
grep -v '^#' CLAUDE.md | grep -q '260726-kdp'
test "$(git diff --numstat -- CLAUDE.md | awk '{print $1-$2}')" -le 15
test -s .planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/DEFERRED.md
grep -q 'fetch_jplsbdb_objects' .planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/DEFERRED.md
git diff --name-only | grep -qv 'src/fomo/settings.py'
! git diff --cached --name-only | grep -q 'src/fomo/settings.py'
LOG=$(mktemp)
sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D "exclude_patterns=notebooks/*,_build" 2>&1 | tee "$LOG"
! grep -Ei 'runbooks/telescope_runs_calendar[^ ]*:.*(warning|error|severe)' "$LOG"
    </automated>
  </verify>
  <done>
CLAUDE.md's paired-deliverable rule covers `docs/runbooks/` by directory rather than by
filename, retains the pairing map, the behavior-change trigger, the `files_modified`
up-front requirement, the nbconvert regeneration mechanics, and all four subagent roles
with their duties, and records this task as the third breach — while growing the file by
no more than 15 net lines. DEFERRED.md records all four out-of-scope gaps.
`src/fomo/settings.py` is untouched and unstaged.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none introduced) | Documentation-only change. No new input parsing, no new endpoint, no new dependency, no executable code path. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-kdp-01 | Information disclosure | `docs/runbooks/telescope_runs_calendar.rst` | mitigate | The runbook states on-page that every example uses synthetic placeholder names, emails, and telescope/instrument strings. New examples added in Tasks 1-2 must use synthetic proposal codes, campaign names, and usernames only — no real LCO proposal codes, real observer usernames, or real contact details. |
| T-kdp-02 | Tampering | Working tree | mitigate | `src/fomo/settings.py` carries uncommitted user-local edits; house rule 3 forbids staging, committing, reverting, or reformatting it, and Task 3's verify gates assert it is neither modified beyond its current state nor staged. |
| T-kdp-SC | Tampering | npm/pip/cargo installs | accept | No package installs of any kind in this plan. Package Legitimacy Gate not applicable. |
</threat_model>

<verification>
Run from the repo root, after all three tasks:

1. **Sphinx build (the pre-commit gate).** A malformed `list-table` or a short heading
   underline renders silently wrong rather than failing loudly, so eyeballing the diff
   is not sufficient — this build must actually run:

   ```
   LOG=$(mktemp)
   sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees \
     -D "exclude_patterns=notebooks/*,_build" 2>&1 | tee "$LOG"
   ! grep -Ei 'runbooks/telescope_runs_calendar[^ ]*:.*(warning|error|severe)' "$LOG"
   ```

   Both `./_readthedocs/` and `./docs/_build/` are gitignored; do not stage them.

2. **Cheat-sheet integrity.** Exactly 6 command rows, body-section order matching table
   order.

3. **No source changes.** `git diff --name-only` lists only
   `docs/runbooks/telescope_runs_calendar.rst`, `CLAUDE.md`, and the pre-existing
   `src/fomo/settings.py`. No `.py` under `solsys_code/`, no `.ipynb` anywhere. Because
   no `.py` changes, ruff is a non-event — do not run it repo-wide (house rule 4).

4. **No Django/pytest run needed.** No code changed.
</verification>

<success_criteria>
- `load_telescope_runs --campaign` is documented as optional in both the body and the
  cheat-sheet, and cannot be confused with `import_campaign_csv`'s required `--campaign`.
- `backfill_lco_observation_records` has a body section and a cheat-sheet row, covering
  both required flags and all four optional flags, with the reuse-vs-build behavior of
  `--create-missing-targets` stated explicitly.
- The `[QUEUED]` claim reflects the successful-terminal-status guard.
- The Troubleshooting section is byte-identical; no speculative entries added.
- `fetch_jplsbdb_objects` appears nowhere in the runbook.
- CLAUDE.md's rule is directory-scoped on `docs/runbooks/`, retains all four subagent
  roles and the notebook mechanics, records the third breach, and grew by ≤ 15 net lines.
- DEFERRED.md records the four out-of-scope gaps.
- The pre-commit Sphinx build emits no warning or error for the runbook.
- `src/fomo/settings.py` untouched and unstaged.
</success_criteria>

<output>
Create `.planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/260726-kdp-SUMMARY.md` when done.
</output>
