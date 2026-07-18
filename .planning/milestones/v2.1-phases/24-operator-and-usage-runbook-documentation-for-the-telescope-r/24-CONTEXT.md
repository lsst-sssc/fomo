# Phase 24: Operator and usage runbook documentation for the telescope-runs-calendar management commands and staff workflows - Context

**Gathered:** 2026-07-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Publish general, discoverable how-to-run documentation for the telescope-runs-calendar management commands (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`, `import_campaign_csv`) and the approval queue's staff status-change actions (`mark_cancelled`/`mark_weather_failure` from Phase 23) — beyond what already exists as design rationale (`docs/design/`) and in-code `--help` text/docstrings. No source code, no new features — this is a documentation-only phase.

</domain>

<decisions>
## Implementation Decisions

### Format & Location
- **D-01:** New Sphinx `.rst` page(s) — not plain Markdown, not just expanded docstrings. Builds through the existing Sphinx pipeline (already run by pre-commit), gets a real URL on the hosted docs site.
- **D-02:** Lives in a new `docs/runbooks/` directory — distinct from `docs/design/` (rationale docs). The new directory name signals "how to operate this" vs. "why it's built this way".
- **D-03:** Hand-written prose, not auto-embedded `--help` text. Do NOT add the `sphinx-django-command` plugin dependency — full editorial control over voice, structure, and troubleshooting content that `--help` text can't carry is worth more than the DRY-sync benefit, especially since this project already carries heavy scientific dependencies and avoids adding new ones without strong justification.
- **D-04:** Must be linked into the existing `docs/index.rst` toctree (currently: Home / Installation / Design / API Reference / Notebooks) so it's actually discoverable, not an orphan page.

### Structure
- **D-05:** One consolidated runbook page (single URL, easy top-to-bottom skim for a new operator) — NOT one page per command.
- **D-06:** Within that single page, organize by task-oriented "how do I...?" framing (e.g. "How do I add a classical schedule?", "How do I sync LCO observations?") rather than a flat per-command reference dump. Combines the "one page" and "task-oriented" options the user was offered — this was a deliberate combination, not a pick between them.
- **D-07:** The approval-queue status-change actions (mark_cancelled/mark_weather_failure) are folded into the calendar-sync documentation rather than getting their own dedicated section — even though the interaction model differs (staff web-UI clicks vs. CLI command flags). Keep everything calendar-related together.

### Audience & Depth
- **D-08:** The runbook itself assumes `manage.py`/venv/migration familiarity — do NOT re-explain Django basics inline.
- **D-09:** General Django/`manage.py` orientation content (venv activation, running commands, migrations) goes in its own **separate section positioned immediately after the existing Installation doc** (`docs/installation.rst`), not duplicated into the runbook. The runbook cross-references that section for readers who need it, rather than re-explaining it. This resolves the apparent "assume familiarity" vs. "no background assumed" tension: no background is assumed of the *reader as a whole*, but the onboarding content lives in Installation-adjacent material, not inside the runbook proper.
- **D-10:** Include a quick-reference cheat-sheet (command + key flags, one line each) in addition to the narrative walkthrough — for operators who already know the workflow and just need exact syntax.

### Failure/Troubleshooting Coverage
- **D-11:** Happy-path "how to run" content PLUS a troubleshooting section covering failure modes already observed in production — not a speculative/comprehensive enumeration of every code-level exception path, and not happy-path-only.
- **D-12:** Explicitly document the Observatory-missing-timezone gap as a known operational issue with a fix-it step (`Observatory.timezone` must be an IANA name, e.g. `"America/Santiago"`, before `sun_event()`/projection can succeed for that site). This is real, currently observed (Phase 25's `backfill_range_calendar_events --dry-run`/live run against the real dev DB: `Observatory 'FTN' (obscode=F65) has no timezone set`), and will keep tripping up any future backfill/projection run against that Observatory record until someone sets the field — not a hypothetical.
- **D-13:** Other known failure modes worth documenting (from existing command behavior, not hypothetical): `load_telescope_runs`' per-line `(ValueError, Observatory.DoesNotExist)` skip-and-log; the LCO/SOAR per-record telescope-API timeout/fallback-label behavior; `import_campaign_csv`'s skip-and-log natural-key/site-resolution failures and `site_needs_review` flag; `backfill_range_calendar_events`' per-candidate `ValueError`-skip-and-continue behavior (never aborts the whole run).

### Claude's Discretion
- Exact page title and file name within `docs/runbooks/` (e.g. `telescope_runs_calendar.rst` matching the existing design doc's name, or a more explicit `operator_runbook.rst`).
- Exact section ordering and heading hierarchy within the task-oriented structure.
- Whether the Django-basics onboarding section is a wholly new `.rst` file or a new subsection appended to `docs/installation.rst` itself — both satisfy D-09's "positioned right after Installation" placement; the planner/researcher should pick based on what reads better once outlined.

### Reviewed Todos (not folded)
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` ("Extract site/telescope mapping and instrument extraction into own module") — code refactor, not documentation. Out of this phase's scope; left pending for a future tech-debt phase.
- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` ("Rename calendar_utils.py private helpers to reflect shared-module API") — code refactor, not documentation. Out of this phase's scope; left pending for a future tech-debt phase.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing documentation conventions to match
- `docs/index.rst` — the toctree that MUST be updated to link the new runbook (currently: Home, Installation, Design, API Reference, Notebooks)
- `docs/installation.rst` — existing Installation doc; the new Django-basics onboarding section is positioned immediately after this (D-09)
- `docs/design/telescope_runs_calendar.rst` — the existing design-rationale doc for this same feature area (issue #37); the runbook should cross-reference this for "why", not duplicate it
- `docs/design/design.rst` — top-level design index, shows how design docs are cross-linked; use as a structural model for how `docs/runbooks/` should be wired into the toctree

### Commands and workflow to document
- `solsys_code/management/commands/load_telescope_runs.py` — classical-schedule ingest
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — LCO queue sync (multi-proposal/multi-facility)
- `solsys_code/management/commands/sync_gemini_observation_calendar.py` — Gemini ToO sync
- `solsys_code/management/commands/import_campaign_csv.py` — campaign CSV bootstrap import
- `solsys_code/management/commands/backfill_range_calendar_events.py` — one-off range-window backfill (new in Phase 25; not in the phase's original scope text but same command family — planner should confirm with the user whether to include it)
- `solsys_code/campaign_views.py` (`CampaignRunDecisionView._set_run_status()`) — the approval queue's mark_cancelled/mark_weather_failure staff actions (Phase 23)
- `src/templates/campaigns/approval_queue.html` — the approval-queue UI the mark_cancelled/mark_weather_failure docs will describe

### Known failure modes to document (D-11/D-12/D-13)
- `.planning/debug/range-window-calendar-event.md` — the debug investigation that produced Phase 25; background for the Observatory-timezone-missing failure mode
- `.planning/phases/25-range-window-calendarevent-projection-allow-approved-site-re/25-UAT.md` — live evidence of the Observatory-timezone-missing failure (`Observatory 'FTN' (obscode=F65) has no timezone set`)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Every management command already has a `help` string and class docstring (verified: all 4 original commands + the new backfill command) — these are a starting outline for the runbook's per-command sections, not something to duplicate verbatim (D-03).

### Established Patterns
- `docs/design/*.rst` files are the only existing precedent for narrative documentation in this project — no runbook/how-to convention exists yet. This phase establishes that convention.
- Sphinx build already runs in pre-commit (`sphinx-build` hook, confirmed fast: ~5-6s) — new `.rst` files under `docs/runbooks/` will be validated automatically on commit once added to the toctree.

### Integration Points
- `docs/index.rst`'s `toctree` directive is the single integration point that makes any new doc discoverable on the hosted site (D-04).

</code_context>

<specifics>
## Specific Ideas

No specific prose/wording requirements were given — open to standard technical-writing approaches for voice and structure within the decisions above.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

### Reviewed Todos (not folded)
See `<decisions>` → Reviewed Todos above.

</deferred>

---

*Phase: 24-operator-and-usage-runbook-documentation-for-the-telescope-r*
*Context gathered: 2026-07-18*
