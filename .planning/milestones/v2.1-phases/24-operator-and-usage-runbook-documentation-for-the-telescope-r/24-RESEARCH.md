# Phase 24: Operator and Usage Runbook Documentation - Research

**Researched:** 2026-07-18
**Domain:** Sphinx/RST technical documentation (docs-only phase, no source code changes)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** New Sphinx `.rst` page(s) -- not plain Markdown, not just expanded docstrings. Builds through the existing Sphinx pipeline (already run by pre-commit), gets a real URL on the hosted docs site.
- **D-02:** Lives in a new `docs/runbooks/` directory -- distinct from `docs/design/` (rationale docs). The new directory name signals "how to operate this" vs. "why it's built this way".
- **D-03:** Hand-written prose, not auto-embedded `--help` text. Do NOT add the `sphinx-django-command` plugin dependency -- full editorial control over voice, structure, and troubleshooting content that `--help` text can't carry is worth more than the DRY-sync benefit, especially since this project already carries heavy scientific dependencies and avoids adding new ones without strong justification.
- **D-04:** Must be linked into the existing `docs/index.rst` toctree (currently: Home / Installation / Design / API Reference / Notebooks) so it's actually discoverable, not an orphan page.
- **D-05:** One consolidated runbook page (single URL, easy top-to-bottom skim for a new operator) -- NOT one page per command.
- **D-06:** Within that single page, organize by task-oriented "how do I...?" framing (e.g. "How do I add a classical schedule?", "How do I sync LCO observations?") rather than a flat per-command reference dump. Combines the "one page" and "task-oriented" options the user was offered -- this was a deliberate combination, not a pick between them.
- **D-07:** The approval-queue status-change actions (mark_cancelled/mark_weather_failure) are folded into the calendar-sync documentation rather than getting their own dedicated section -- even though the interaction model differs (staff web-UI clicks vs. CLI command flags). Keep everything calendar-related together.
- **D-08:** The runbook itself assumes `manage.py`/venv/migration familiarity -- do NOT re-explain Django basics inline.
- **D-09:** General Django/`manage.py` orientation content (venv activation, running commands, migrations) goes in its own **separate section positioned immediately after the existing Installation doc** (`docs/installation.rst`), not duplicated into the runbook. The runbook cross-references that section for readers who need it, rather than re-explaining it. This resolves the apparent "assume familiarity" vs. "no background assumed" tension: no background is assumed of the *reader as a whole*, but the onboarding content lives in Installation-adjacent material, not inside the runbook proper.
- **D-10:** Include a quick-reference cheat-sheet (command + key flags, one line each) in addition to the narrative walkthrough -- for operators who already know the workflow and just need exact syntax.
- **D-11:** Happy-path "how to run" content PLUS a troubleshooting section covering failure modes already observed in production -- not a speculative/comprehensive enumeration of every code-level exception path, and not happy-path-only.
- **D-12:** Explicitly document the Observatory-missing-timezone gap as a known operational issue with a fix-it step (`Observatory.timezone` must be an IANA name, e.g. `"America/Santiago"`, before `sun_event()`/projection can succeed for that site). This is real, currently observed (Phase 25's `backfill_range_calendar_events --dry-run`/live run against the real dev DB: `Observatory 'FTN' (obscode=F65) has no timezone set`), and will keep tripping up any future backfill/projection run against that Observatory record until someone sets the field -- not a hypothetical.
- **D-13:** Other known failure modes worth documenting (from existing command behavior, not hypothetical): `load_telescope_runs`' per-line `(ValueError, Observatory.DoesNotExist)` skip-and-log; the LCO/SOAR per-record telescope-API timeout/fallback-label behavior; `import_campaign_csv`'s skip-and-log natural-key/site-resolution failures and `site_needs_review` flag; `backfill_range_calendar_events`' per-candidate `ValueError`-skip-and-continue behavior (never aborts the whole run).

### Claude's Discretion

- Exact page title and file name within `docs/runbooks/` (e.g. `telescope_runs_calendar.rst` matching the existing design doc's name, or a more explicit `operator_runbook.rst`).
- Exact section ordering and heading hierarchy within the task-oriented structure.
- Whether the Django-basics onboarding section is a wholly new `.rst` file or a new subsection appended to `docs/installation.rst` itself -- both satisfy D-09's "positioned right after Installation" placement; the planner/researcher should pick based on what reads better once outlined.

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope.

**Reviewed Todos (not folded into this phase):**
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` ("Extract site/telescope mapping and instrument extraction into own module") -- code refactor, not documentation. Out of this phase's scope; left pending for a future tech-debt phase.
- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` ("Rename calendar_utils.py private helpers to reflect shared-module API") -- code refactor, not documentation. Out of this phase's scope; left pending for a future tech-debt phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

No REQ-IDs are mapped to this phase in `.planning/REQUIREMENTS.md` (that file tracks v2.1's SCHED/ASSET/IMPORT/SITE/VIEW requirements, all already Complete/traced to Phases 18-21). Phase 24 was added directly to the roadmap on 2026-07-17 as a documentation gap found during PR #41/#43 split review (see STATE.md "Roadmap Evolution"), with no corresponding REQUIREMENTS.md entry. CONTEXT.md's D-01 through D-13 decisions are the authoritative scope for this phase; the planner should treat them as the requirement set (one plan task per D-decision cluster is a reasonable default, not a hard rule).

| ID | Description | Research Support |
|----|-------------|------------------|
| (none mapped) | Publish task-oriented operator runbook + Django-onboarding section + cheat-sheet + troubleshooting coverage per D-01..D-13 | See Architecture Patterns, Code Examples, and Common Pitfalls sections below for concrete per-command raw material |
</phase_requirements>

## Summary

This is a documentation-only phase: no source files change, only new Sphinx `.rst` content plus one toctree edit. The five commands in scope (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`, `import_campaign_csv`, and the not-yet-confirmed `backfill_range_calendar_events`) all already have `help` strings and docstrings — verified directly by running `--help` for each (Step 3 output below) and reading the source. None of that text should be copied verbatim (D-03); it's raw material for hand-written task-oriented prose. The house style to match is `docs/design/telescope_runs_calendar.rst` and `docs/design/design.rst` — both read and excerpted below, giving concrete RST patterns (list-table, code-block:: console, toctree, math directive) already in use in this repo.

The approval queue's `mark_cancelled` / `mark_weather_failure` staff actions (Phase 23, `CampaignRunDecisionView._set_run_status`) render as "Mark Cancelled" / "Mark Weathered" buttons on the Decided table when a row's `approval_status == APPROVED` (any run_status), confirmed by reading `campaign_tables.py`'s `render_actions`. These fold into the calendar-sync section per D-07 rather than getting a standalone section.

Three concrete, already-observed failure modes are confirmed from the codebase and from Phase 25's real dev-DB run (not hypothetical): (1) the Observatory-missing-timezone gap (D-12, `Observatory 'FTN' (obscode=F65) has no timezone set`, observed live in `25-UAT.md`); (2) each ingest/sync command's per-line/per-record skip-and-log behavior (never aborts the whole run) with specific counters in each command's final summary line; (3) `import_campaign_csv`'s `site_needs_review`/`window_needs_review` flags that surface unresolved rows in the approval queue's "Sites Needing Review" card rather than blocking import.

`backfill_range_calendar_events` (added in Phase 25, not in this phase's original scope text) has a full `help` string, class docstring, and a `--dry-run` flag, and its failure mode (per-candidate `ValueError`-skip, never aborting the run) is already documented in `25-UAT.md`'s real backfill run against `src/fomo_db.sqlite3`. It is the same command family (calendar-projection backfill) and has real, already-observed operational friction (the Observatory-timezone gap hit it directly) — the research recommends including it, but per CONTEXT.md this must be confirmed with the user rather than assumed (see Open Questions).

**Primary recommendation:** One new file, `docs/runbooks/telescope_runs_calendar.rst` (or `operator_runbook.rst` — planner's discretion), containing task-oriented "How do I...?" subsections for each command (with the approval-queue actions folded into the calendar-sync subsection), a quick-reference cheat-sheet table, and a troubleshooting section built from the three failure-mode families above — plus a new "Running FOMO Management Commands" subsection appended to `docs/installation.rst` (simplest way to satisfy D-09's "positioned immediately after Installation" placement without a second new file and a second toctree entry).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Runbook prose content (task-oriented how-to, cheat-sheet, troubleshooting) | Documentation source (`docs/runbooks/*.rst`) | — | New `.rst` file(s); this phase's actual deliverable |
| Django/manage.py onboarding content | Documentation source (`docs/installation.rst`) | — | D-09 locks this placement; existing file, new subsection |
| Toctree wiring / discoverability | Documentation source (`docs/index.rst`) | — | Single integration point (D-04); a page not listed here is an orphan |
| Doc build validation | Build tooling (Sphinx, invoked by pre-commit) | CI (`.github/workflows/`, ReadTheDocs) | `sphinx-build` hook (`always_run: true`) already runs on every commit; catches broken RST/toctree syntax before merge |
| Hosted delivery of the runbook | ReadTheDocs / static site hosting | — | Out of scope to configure — existing pipeline already publishes `docs/` |
| Command behavior being documented (`load_telescope_runs` etc.) | Application code (`solsys_code/management/commands/`) | — | Read-only source of truth for this phase; NOT modified |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Sphinx | 8.1.3 `[VERIFIED: pip show sphinx]` | RST -> HTML documentation build | Already the project's only doc toolchain (`docs/conf.py`, `docs/requirements.txt`) |
| sphinx-rtd-theme | (in `docs/requirements.txt`) `[CITED: docs/requirements.txt]` | HTML theme used for the hosted site | Already configured; no phase action needed |
| sphinx-autoapi | 3.6.0 `[VERIFIED: pip show sphinx-autoapi]` | Auto-generates the existing "API Reference" toctree entry | Unrelated to this phase's new content but confirms the toctree's 4th entry is machine-generated, not hand-written -- the new runbook entry is hand-added like Installation/Design |
| nbsphinx | (in `docs/requirements.txt`) `[VERIFIED: python3 -c "import nbsphinx"]` | Renders the `docs/notebooks/pre_executed/*.ipynb` files referenced from `docs/notebooks.rst` | Confirms the Notebooks toctree entry's mechanism; not touched by this phase |

### Supporting

None required. This phase adds zero new dependencies (see D-03: `sphinx-django-command` explicitly rejected).

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-written `.rst` prose (D-03, locked) | `sphinx-django-command` (auto-embeds `--help` into docs) | Rejected by the user: loses editorial control over troubleshooting/voice content that `--help` text structurally can't carry, and adds a new dependency this scientific-heavy project avoids without strong justification |
| Sphinx `.rst` (D-01, locked) | Plain Markdown / GitHub wiki | Rejected by the user: doesn't build through the existing Sphinx pipeline, no real hosted URL, breaks from the project's only existing documentation convention (`docs/design/*.rst`) |

**Installation:** None -- no new packages. `pip install -e .'[dev]'` (already covers Sphinx per `pyproject.toml`) is unchanged.

**Version verification:** Confirmed directly in this environment (not from training data):
```console
$ pip show sphinx
Name: Sphinx
Version: 8.1.3
$ pip show sphinx-autoapi
Name: sphinx-autoapi
Version: 3.6.0
```

## Package Legitimacy Audit

**Not applicable.** This phase installs zero external packages — D-03 explicitly rejects adding `sphinx-django-command`, and no other dependency is introduced. The Package Legitimacy Gate protocol is skipped per its own trigger condition ("Every phase that installs external packages").

## Architecture Patterns

### System (Documentation) Structure Diagram

```
Author writes prose  -->  docs/runbooks/<name>.rst  --\
                                                          >-- docs/index.rst toctree (D-04)
Author appends section --> docs/installation.rst  ------/        |
                                                                   v
                                                     pre-commit `sphinx-build` hook
                                                     (always_run: true, excludes
                                                      notebooks/* for speed)
                                                                   |
                                                                   v
                                                     ./_readthedocs (HTML build output)
                                                                   |
                                                                   v
                                                     ReadTheDocs / hosted docs site
                                                     (ultimate discoverability target
                                                      of D-04's toctree requirement)
```

A reader's actual path: hosted site homepage (`docs/index.rst`, "Welcome to fomo's documentation!") -> toctree link "Runbooks" (new) -> single consolidated page -> task-oriented subsection -> (optionally) cross-reference link to the Django-onboarding subsection in `docs/installation.rst` for readers needing venv/migrate background (D-09).

### Recommended Project Structure

```
docs/
├── index.rst                       # MODIFIED: add one new toctree line (D-04)
├── installation.rst                # MODIFIED: append "Running FOMO Management Commands"
│                                    #   subsection (D-09) -- satisfies "immediately after
│                                    #   Installation" without a second new file/toctree entry
├── design/
│   ├── design.rst                  # UNCHANGED -- structural model only (see Pattern 1)
│   └── telescope_runs_calendar.rst # UNCHANGED -- cross-referenced ("why"), not duplicated
└── runbooks/                       # NEW directory (D-02)
    └── telescope_runs_calendar.rst # NEW: the one consolidated runbook page (D-05)
                                     #   (exact filename is Claude's discretion)
```

### Pattern 1: Sub-toctree wiring (mirror `docs/design/design.rst`)

**What:** `docs/design/design.rst` is itself a landing page with prose, then a `.. toctree::` directive listing its child pages by bare module name (no directory prefix, since the toctree file lives inside the same directory as its children). `docs/index.rst` in turn lists `Design <design/design>` -- directory-prefixed, since `index.rst` lives one level up.

**When to use:** Exactly this phase's situation -- a new subdirectory (`docs/runbooks/`) needs both an entry in the top-level `docs/index.rst` toctree AND (if more than one page is ever added later) its own local toctree. Since D-05 locks this to a single consolidated page, a runbooks-local toctree file is unnecessary — the single page can be referenced directly from `docs/index.rst`, exactly as `Installation <installation>` is (a single file, no subdirectory landing page needed for a one-page section).

**Example (verbatim from `docs/index.rst`):**
```rst
.. toctree::
   :hidden:

   Home page <self>
   Installation and Getting Started <installation>
   Design <design/design>
   API Reference <autoapi/index>
   Notebooks <notebooks>
```
The new line to add (D-04), matching this exact `Label <path>` syntax and `:hidden:` toctree:
```rst
   Runbooks <runbooks/telescope_runs_calendar>
```
(adjust the path to whatever filename is chosen under `docs/runbooks/`).

### Pattern 2: `docs/design/design.rst`'s local toctree (structural model per CONTEXT.md canonical_refs)

**What:** A `maxdepth: 1` toctree listing sibling files by bare name.
**Example (verbatim from `docs/design/design.rst`):**
```rst
.. toctree::
   :maxdepth: 1

   telescope_runs_calendar
   tom_calendar_vs_yse_pz_calendar
   gsd_experiment
   eso_feasibility_spike
   uncertain_scheduling_spike
```
Only relevant if the planner decides the runbook needs more than one page later — not needed for this phase's single-page scope (D-05), but documents the pattern the codebase already uses if `docs/runbooks/` ever grows a second page.

### Pattern 3: House-style RST conventions (verified from `telescope_runs_calendar.rst`)

- **`list-table` with explicit `:header-rows:` and `:widths:`** for tabular data (site coordinates, field mappings) -- used twice in the design doc; the runbook's cheat-sheet (D-10) is naturally a `list-table` (columns: Command | Key Flags | One-line description).
- **`code-block:: console`** (not `bash` or plain ` :: `) for shell examples, using the `>>` prompt convention already established in `docs/installation.rst` (e.g. `>> python3 manage.py migrate`). The runbook should reuse this exact `>>` prompt style for consistency across the whole docs site.
- **Bold-lead-in sentence + explanatory prose**, not just a code block with no framing -- every design-doc section opens with a bold key term or short declarative sentence before diving into detail.
- **Cross-references via plain prose + backticked names**, not Sphinx `:func:`/`:class:` roles for management commands (those roles resolve only for autoapi-indexed Python objects; a `manage.py` command name is not one). Use double-backtick monospace (` ``load_telescope_runs`` `) for command/field names, matching the design doc's own usage (` ``CalendarEvent`` `, ` ``sun_event(site, date, kind)`` `).
- **`.. note::` admonitions** for asides that would clutter the main flow -- `docs/installation.rst` uses this for the pip/conda install note and the WSL2 Windows caveat; the runbook should use the same admonition for e.g. the Observatory-timezone gotcha (D-12) if it needs a callout separate from the troubleshooting section's prose.

### Anti-Patterns to Avoid

- **Auto-embedding `--help` text via a Sphinx extension:** explicitly rejected (D-03). Even though `sphinx-django-command`-style tooling would guarantee DRY sync between code and docs, the user traded that off deliberately for editorial control and to avoid a new dependency.
- **One page per command:** explicitly rejected (D-05). A `docs/runbooks/` directory with 5 separate command pages would fragment the "single top-to-bottom skim" experience the user wants.
- **A flat command-reference dump:** explicitly rejected (D-06) in favor of task-oriented "How do I...?" framing -- even though the underlying material (5 commands + 2 UI actions) maps naturally to a reference table, the primary organizing structure must be tasks, not commands (the cheat-sheet, D-10, is where the flat command-to-flag mapping belongs, as a secondary quick-reference, not the primary structure).
- **Duplicating design-doc rationale into the runbook:** the runbook should link to `docs/design/telescope_runs_calendar.rst` for "why" (e.g. why -15deg dark window, why dip correction) rather than re-explaining the astronomy. Per CONTEXT.md's canonical_refs framing.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Keeping docs in sync with `--help` text | A custom script or Sphinx extension to scrape `--help` output into RST | Hand-written prose (D-03) that references but doesn't duplicate `--help` text | Explicitly the user's locked decision; DRY-sync tooling was considered and rejected |
| Rendering the toctree / site navigation | A custom nav-building script | Sphinx's built-in `.. toctree::` directive (already used by every other doc) | Zero new tooling; matches the one existing convention exactly |

**Key insight:** This entire phase is about NOT hand-rolling a new documentation system -- everything must fit inside the Sphinx pipeline that already exists and already runs in pre-commit and CI. There is no "build vs. buy" decision left to make; the only work is content authoring plus one toctree edit.

## Common Pitfalls

### Pitfall 1: Orphan page (toctree omission)

**What goes wrong:** A new `.rst` file exists under `docs/runbooks/` but is never referenced from any `.. toctree::` directive.
**Why it happens:** Sphinx doesn't require every `.rst` file to be reachable from the root toctree to build successfully by default -- Sphinx normally emits an "orphan" warning (not a hard build failure) for an unreferenced document, so the phase could complete with the runbook technically live at its URL but never linked from `docs/index.rst`, silently violating D-04 with no build error to catch it.
**How to avoid:** The planner must include a task/verification step that explicitly confirms the new toctree line was added to `docs/index.rst` and that the built HTML nav (or at minimum, an `sphinx-build -W` strict run) shows no orphan warning for the new page.
**Warning signs:** `sphinx-build` output containing `WARNING: document isn't included in any toctree`.

### Pitfall 2: `sphinx-build` pre-commit hook always runs, but excludes notebooks

**What goes wrong:** Assuming the pre-commit `sphinx-build` hook validates everything including notebook rendering, when it deliberately passes `-D exclude_patterns=notebooks/*,_build` to skip notebook execution for speed (confirmed in `.pre-commit-config.yaml` lines 67-88). This phase doesn't touch notebooks, so this is a non-issue for its own content, but the planner should not assume the fast pre-commit build is the *complete* CI-equivalent check -- the full non-excluding build (used in CI/ReadTheDocs) is a separate, slower run.
**Why it happens:** The exclusion exists so ordinary commits (including this phase's) don't pay the cost of re-executing every demo notebook.
**How to avoid:** Rely on the pre-commit hook for the RST-syntax/toctree-completeness check (fast, `always_run: true`); it's sufficient for a docs-only phase since no notebook content is touched. No additional notebook re-execution needed for this phase per se (see next section on CLAUDE.md's notebook-pairing rule).
**Warning signs:** N/A for this phase -- documented so the planner doesn't over-scope a full CI-equivalent Sphinx run as a verification step when the fast pre-commit hook already covers what changed.

### Pitfall 3: Assuming CLAUDE.md's "demo notebook companion" rule applies here

**What goes wrong:** CLAUDE.md requires that any plan changing behavior in `telescope_runs.py`, `load_telescope_runs.py`, `sync_lco_observation_calendar.py`, or `sync_gemini_observation_calendar.py` also update that module's paired demo notebook. A planner skimming CLAUDE.md might over-apply this rule to a phase that merely *documents* (in Sphinx, not notebooks) these same modules.
**Why it happens:** The four modules named in CLAUDE.md's rule are exactly the four command modules this runbook describes, creating surface-level pattern-match risk.
**How to avoid:** This phase makes **zero changes** to any of those four modules' behavior (no new extraction logic, no new parameters, no new fixture shapes) -- it only adds new prose in `docs/runbooks/`. CLAUDE.md's rule is explicitly scoped to "new extraction logic, new parameters, new fixture shapes -- not pure refactors or typo fixes"; a docs-only phase with zero source diffs to those files doesn't trigger it at all. The planner should state this explicitly in the plan (one line: "no paired-notebook update required — no behavior change to the four listed modules") rather than silently omitting it, so the plan-checker doesn't need to re-derive the same reasoning.
**Warning signs:** A plan-checker flags a missing notebook update; the correct response is to point at this reasoning, not to add an unnecessary notebook task.

### Pitfall 4: Documentation examples leaking real PII

**What goes wrong:** The troubleshooting section (D-11/D-13) needs concrete examples of skip-and-log output and CSV row failures. `import_campaign_csv`'s own code deliberately avoids logging full rows because they "also carr[y] Contact Person/Email PII from the real 3I/ATLAS sheet" (verbatim comment in `import_campaign_csv.py`, WR-06). A runbook author copying a *real* log line or CSV row verbatim as an example risks embedding real submitter names/emails into a permanently public, hosted document.
**Why it happens:** The most natural way to write a concrete troubleshooting example is to paste real observed output, and the real observed output (from dev-DB testing) may include real contact fields.
**How to avoid:** Use synthetic/placeholder values (`Jane Doe`, `jane@example.com`, a fabricated telescope/instrument string) in every troubleshooting example, never real data pulled from `src/fomo_db.sqlite3` or the real 3I/ATLAS sheet. The one exception already safe to use verbatim is the Observatory-timezone message (`Observatory 'FTN' (obscode=F65) has no timezone set`) since an MPC obscode and telescope short-name are not personal data.
**Warning signs:** Any example containing a plausible-looking person's name + email pair together.

### Pitfall 5: Confusing `mark_cancelled`/`mark_weather_failure`'s "no server-side change" guarantee with "no user-visible effect"

**What goes wrong:** Documentation might describe these actions as low-stakes because `_set_run_status()`'s docstring notes it's idempotent and re-clickable ("Buttons render for ANY APPROVED row regardless of current run_status ... re-clicking is a harmless idempotent no-op"). But the action DOES have real user-visible effect: it prepends `[CANCELLED]`/`[WEATHERED]` to every `CalendarEvent` title belonging to that run (including all per-night events for a range-window run, per Phase 25) and this is visible on the public campaign calendar.
**Why it happens:** "Idempotent/no revert button" reads as "safe/low-consequence" but the state change itself (public-facing calendar title mutation) is real and immediate.
**How to avoid:** The runbook's description of these two actions should be accurate about consequence (calendar title changes immediately and publicly) even while noting it's safe to correct a mis-click by clicking the other button or re-clicking (idempotent).
**Warning signs:** N/A -- an editorial accuracy concern, not a build-time-detectable one.

## Code Examples

Verified patterns pulled directly from the actual codebase (not training data):

### `docs/index.rst`'s current toctree (the exact block to edit for D-04)
```rst
.. toctree::
   :hidden:

   Home page <self>
   Installation and Getting Started <installation>
   Design <design/design>
   API Reference <autoapi/index>
   Notebooks <notebooks>
```

### Each in-scope command's real `--help` output (run directly in this environment, 2026-07-18)

```console
$ python3 manage.py load_telescope_runs --help
usage: manage.py load_telescope_runs [-h] [--version] [-v {0,1,2,3}] ... filepath

Load classical telescope run lines from a file and create/update
CalendarEvents

positional arguments:
  filepath              Path to a text file of classical run lines (one per line)
```

```console
$ python3 manage.py sync_lco_observation_calendar --help
usage: manage.py sync_lco_observation_calendar [-h] --proposal PROPOSAL ...

Sync LCO queue ObservationRecords for a proposal to CalendarEvents

  --proposal PROPOSAL   LCO/SOAR proposal code(s) to filter ObservationRecords
                        by. Accepts a single code, a comma-separated list
                        (e.g. 'A,B,C'), or the case-insensitive token 'ALL' to
                        sync every record regardless of proposal.
```

```console
$ python3 manage.py sync_gemini_observation_calendar --help
usage: manage.py sync_gemini_observation_calendar [-h] ...

Sync Gemini queue ObservationRecords to CalendarEvents
```
(No command-specific flags -- it processes every `ObservationRecord(facility='GEM')` unconditionally, unlike LCO's `--proposal` filter. Worth calling out explicitly in the runbook since a reader coming from the LCO section might expect a similar filter flag.)

```console
$ python3 manage.py import_campaign_csv --help
usage: manage.py import_campaign_csv [-h] --campaign CAMPAIGN ... filepath

Bootstrap-import a campaign coordination CSV into CampaignRun rows (CAMP-04).
WARNING: re-running this command over the same campaign always resets each
row's `target` to the auto-resolved value (D-07) -- any manual correction a
staff user made to `target` after a previous import will be silently
overwritten on re-import (WR-07).

positional arguments:
  filepath              Path to the campaign coordination CSV file
options:
  --campaign CAMPAIGN   Campaign TargetList name (found-or-created, D-06)
```
This command's own `help` string already contains a load-bearing warning (target field gets reset on every re-import) -- this MUST make it into the runbook's prose/troubleshooting, not just the cheat-sheet, since it's a real "gotcha that will surprise an operator" (D-11 territory).

```console
$ python3 manage.py backfill_range_calendar_events --help
usage: manage.py backfill_range_calendar_events [-h] [--dry-run] ...

One-off backfill: project CalendarEvents for already-APPROVED, site-resolved
range-window CampaignRuns that were approved before per-night projection
existed.

  --dry-run             Report which runs would be backfilled without writing
                        any CalendarEvent rows.
```

### Approval-queue status-change actions (verbatim from `solsys_code/campaign_tables.py`, `render_actions`)

```python
# Buttons render for ANY APPROVED row regardless of current run_status
'<button type="submit" name="action" value="mark_cancelled" '
'class="btn btn-sm btn-outline-secondary">Mark Cancelled</button>'
'<button type="submit" name="action" value="mark_weather_failure" '
'class="btn btn-sm btn-outline-secondary">Mark Weathered</button>'
```
These appear on the Decided table's Actions column for any row with `approval_status == APPROVED`. Clicking either updates `run_status` and prepends `[CANCELLED]`/`[WEATHERED]` (per `_RUN_STATUS_CALENDAR_PREFIX`, `solsys_code/campaign_views.py:379-382`) to every matching `CalendarEvent` title -- including every per-night event for a range-window run (Phase 25).

### Real observed failure-mode text (safe to quote verbatim -- no PII)

```
Observatory 'FTN' (obscode=F65) has no timezone set
```
Source: `.planning/phases/25-range-window-calendarevent-projection-allow-approved-site-re/25-UAT.md`, an actual command run against the real dev DB (`src/fomo_db.sqlite3`). Fix-it step (D-12): set `Observatory.timezone` to a valid IANA name (e.g. `"America/Santiago"`) via the Django admin or `CreateObservatory` form before re-running the backfill/sync command for that site.

## State of the Art

Not applicable in the usual sense (no library-version drift risk here) -- but worth recording: **no runbook/how-to convention exists in this project yet.** `docs/design/*.rst` is the only existing narrative-documentation precedent (confirmed: `docs/design/` contains 5 design docs, zero how-to/operator docs). This phase establishes that convention from scratch, so there is no "old approach being replaced" -- only a new pattern being introduced alongside the existing one.

## Assumptions Log

No claims in this research are tagged `[ASSUMED]`. Every factual claim was either read directly from the project's own source files (`solsys_code/management/commands/*.py`, `solsys_code/campaign_views.py`, `solsys_code/campaign_tables.py`, `docs/*.rst`, `.pre-commit-config.yaml`) or confirmed by executing the actual `--help` commands and `pip show` in this environment. There is no third-party library research in this phase (no new packages), so there is no slopsquatting/hallucination-vector risk to flag.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| (none) | — | — | — |

## Open Questions

1. **Should `backfill_range_calendar_events` be included in this runbook?**
   - What we know: CONTEXT.md explicitly flags this as unresolved ("planner should confirm with the user whether to include it"). The command has a full `--help`/docstring, a real `--dry-run` flag, and a real observed failure mode (the same Observatory-timezone gap, hit directly during Phase 25's live backfill run against `src/fomo_db.sqlite3`). It's the same command family (calendar-projection) as the four in-scope commands.
   - What's unclear: whether the user considers a "one-off backfill" (per its own docstring: "One-off backfill... that were approved before per-night projection existed") worth permanent operator documentation, versus treating it as a historical migration script that doesn't need a discoverable how-to page.
   - Recommendation: include it as a short subsection (it reuses the same troubleshooting material -- the Observatory-timezone gap -- so the marginal documentation cost is low), but the planner MUST get explicit user confirmation before finalizing scope, per CONTEXT.md's directive. Default to "include" if the user has no preference, since its failure mode is already fully documented material from this research and omitting it risks the exact same silent-orphan-knowledge gap this whole phase exists to close.

2. **Django-onboarding section: new file vs. appended subsection to `docs/installation.rst`?**
   - What we know: CONTEXT.md marks this as Claude's discretion (D-09's placement constraint is satisfied either way).
   - What's unclear: nothing blocking -- purely an editorial call.
   - Recommendation: append as a new subsection to `docs/installation.rst` (no new file, no new toctree entry needed) -- `installation.rst` already ends with a "Starting up the webserver" section describing first-run web UI use; a new "Running FOMO Management Commands" subsection reads as a natural continuation of the same document's arc (install -> migrate -> createsuperuser -> runserver -> **run a management command**), and avoids adding a second orphan-page risk (Pitfall 1) for a small amount of content.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Sphinx | Doc build (pre-commit hook + CI/ReadTheDocs) | Yes | 8.1.3 | — |
| sphinx-autoapi | API Reference toctree entry (unrelated to this phase, but shares the build) | Yes | 3.6.0 | — |
| nbsphinx | Notebooks toctree entry (unrelated to this phase) | Yes | (import OK) | — |
| pandoc | Optional -- notebook rendering only, per `docs/installation.rst`'s own note | No | — | Not needed for this phase (no notebook changes); pre-commit's `sphinx-build` hook already excludes `notebooks/*` |
| `manage.py --help` (Django management command introspection) | Confirming exact current help text for the runbook | Yes | Django (project's pinned version) | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** `pandoc` -- not installed in this environment, but not needed since this phase makes no notebook changes (the pre-commit Sphinx hook already excludes notebook execution for speed).

## Validation Architecture

This phase produces no application code, so the standard pytest/Django-test mapping does not apply. Validation for a docs-only phase is build-correctness plus content-accuracy, not unit tests.

### "Test" Framework (docs build, not pytest)

| Property | Value |
|----------|-------|
| Framework | Sphinx 8.1.3, invoked via pre-commit's local `sphinx-build` hook |
| Config file | `docs/conf.py`; hook config in `.pre-commit-config.yaml` (lines 67-88) |
| Quick run command | `pre-commit run sphinx-build --all-files` (or just `git commit`, since `always_run: true`) |
| Full suite command | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees` (no `exclude_patterns` override -- the CI/ReadTheDocs-equivalent full build, matching quick task `260717-iae`'s stated verification method) |

### Phase Requirements -> Validation Map

| Req | Behavior | Validation Type | Command | File Exists? |
|-----|----------|-----------------|---------|-------------|
| D-01/D-04 | New `.rst` builds without error and is reachable from `docs/index.rst`'s toctree (no orphan warning) | build | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees` (check output for `WARNING: document isn't included in any toctree`) | N/A -- Sphinx itself, already present |
| D-02/D-05 | Single consolidated page exists under `docs/runbooks/` | manual/build | `ls docs/runbooks/` + toctree check above | N/A |
| D-10 | Cheat-sheet table present and renders correctly as an RST `list-table` | build + visual | Open `./_readthedocs/html/runbooks/<page>.html` after build | N/A |
| D-11/D-12/D-13 | Troubleshooting section content matches real observed failure text (accuracy, not automatable) | manual review | Cross-check each quoted error string against this RESEARCH.md's Code Examples section and the cited source files | N/A |

### Sampling Rate

- **Per task commit:** rely on pre-commit's `sphinx-build` hook (fast, `always_run: true`, excludes notebooks) -- runs automatically, no extra command needed.
- **Per wave merge / phase gate:** run the full non-excluding build once (`sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees`, no `-D exclude_patterns` override) to match CI/ReadTheDocs exactly, per the precedent set by quick task `260717-iae` ("verified with the full non-excluding sphinx-build (mirrors CI/ReadTheDocs)").

### Wave 0 Gaps

None -- the Sphinx build tooling that will validate this phase's output already exists and already runs in pre-commit and CI; no new test infrastructure is needed.

## Security Domain

`security_enforcement` is enabled in `.planning/config.json`, but this phase makes no application-code changes, has no auth/session/input-validation surface, and introduces no new attack surface. The one real security-adjacent consideration is **information disclosure through documentation content**, not a code vulnerability:

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface changes |
| V3 Session Management | No | No session surface changes |
| V4 Access Control | No | No access-control surface changes |
| V5 Input Validation | No | No new user input accepted |
| V6 Cryptography | No | No crypto surface |
| (informal) Information Disclosure | **Yes** | Documentation examples must use synthetic PII (Pitfall 4), never real contact_person/contact_email values pulled from the real 3I/ATLAS sheet or dev DB |

### Known Threat Patterns for this phase's content

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Real submitter PII (name/email) pasted into a permanently-hosted public doc as a "realistic" troubleshooting example | Information Disclosure | Use synthetic placeholder values in every example (Pitfall 4); the one safe-to-quote-verbatim real string is the Observatory-timezone error message (contains only an MPC obscode and telescope short-name, not personal data) |

## Sources

### Primary (HIGH confidence -- direct source read or tool execution, this session)

- `.planning/phases/24-operator-and-usage-runbook-documentation-for-the-telescope-r/24-CONTEXT.md` -- locked decisions D-01..D-13, discretion, canonical refs
- `.planning/REQUIREMENTS.md`, `.planning/STATE.md` -- confirmed no REQ-IDs mapped to Phase 24; roadmap-evolution note on why this phase was added
- `docs/index.rst`, `docs/installation.rst`, `docs/design/design.rst`, `docs/design/telescope_runs_calendar.rst`, `docs/notebooks.rst` -- read in full for house-style conventions and current toctree contents
- `solsys_code/management/commands/load_telescope_runs.py`, `sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`, `import_campaign_csv.py`, `backfill_range_calendar_events.py` -- read in full for `help` strings, docstrings, and failure-mode code paths
- `solsys_code/campaign_views.py` (`_project_calendar_event`, `CampaignRunDecisionView.post`/`_resolve_site`/`_set_run_status`) and `solsys_code/campaign_tables.py` (`render_actions`) -- read for the approval-queue status-change actions' real button labels and behavior
- `src/templates/campaigns/approval_queue.html` -- read in full (24 lines; the "Sites Needing Review" card wrapping `review_table`)
- `.planning/debug/range-window-calendar-event.md`, `.planning/phases/25-.../25-UAT.md` -- read for the real, live-observed Observatory-timezone failure and its exact error text
- `.pre-commit-config.yaml` -- read for the `sphinx-build` hook's exact args (confirms `always_run: true` and the notebook-exclusion flag)
- Live command execution in this environment, 2026-07-18: `python3 manage.py <cmd> --help` for all 5 commands; `pip show sphinx`, `pip show sphinx-autoapi`; `python3 -c "import nbsphinx"`; `command -v pandoc` (not found), `command -v sphinx-build` (found, 8.1.3)

### Secondary (MEDIUM confidence)

None -- no web search or third-party documentation lookup was needed for this phase; all material is internal to the repository.

### Tertiary (LOW confidence)

None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- confirmed by direct tool execution (`pip show`) in this environment, not training data
- Architecture: HIGH -- every toctree/RST pattern quoted verbatim from existing repo files
- Pitfalls: HIGH -- each pitfall is traced to a specific line/comment in the actual source (e.g. WR-06's PII comment, the pre-commit hook's exclude flag, CLAUDE.md's notebook-pairing rule text)

**Note on knowledge-graph freshness:** `.planning/graphs/graph.json` exists but is `commit_stale: true` (5 commits behind current HEAD as of this research). Graph queries for "runbook documentation" and "Observatory timezone" returned negligible/no useful cross-references beyond the phase's own roadmap entry, so the graph was not a material input to this research; all findings above come from direct file reads and command execution instead.

**Research date:** 2026-07-18
**Valid until:** Low decay risk (documentation-only phase, no library-version dependencies to go stale) -- but re-verify `--help` text against source if any of the 5 commands change before this phase is planned/executed, since the runbook's raw material is a point-in-time snapshot of current behavior.
