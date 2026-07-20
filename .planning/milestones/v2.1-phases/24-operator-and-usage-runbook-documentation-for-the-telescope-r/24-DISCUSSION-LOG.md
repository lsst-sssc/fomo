# Phase 24: Operator and usage runbook documentation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-18
**Phase:** 24-operator-and-usage-runbook-documentation-for-the-telescope-r
**Areas discussed:** Format & location, Structure, Audience & depth, Failure/troubleshooting coverage

---

## Format & Location

| Option | Description | Selected |
|--------|-------------|----------|
| New Sphinx .rst page(s) | Matches docs/design/ convention, builds through existing Sphinx pipeline, real URL on hosted docs site | ✓ |
| Plain Markdown files | Simpler PR diffs, won't build into hosted Sphinx site without myst-parser | |
| Expand in-code docstrings/--help only | Zero new files, discoverable only via `manage.py help <command>` | |

**User's choice:** New Sphinx .rst page(s)

| Option | Description | Selected |
|--------|-------------|----------|
| New docs/runbooks/ directory | Distinct from docs/design/ (rationale) — new section signals "how to operate" vs "why built this way" | ✓ |
| Alongside docs/design/ | Reuses existing directory, relies on filename/title to distinguish | |

**User's choice:** New docs/runbooks/ directory

| Option | Description | Selected |
|--------|-------------|----------|
| Hand-written prose | No new dependency, full control over voice/structure/troubleshooting content | ✓ |
| Auto-embed via sphinx-django-command | New dependency, keeps --help and docs in sync automatically | |

**User's choice:** Hand-written prose
**Notes:** Research surfaced `sphinx-django-command` as a real plugin option (auto-embeds `--help` into Sphinx docs) but user chose not to add the dependency.

---

## Structure

| Option | Description | Selected |
|--------|-------------|----------|
| One consolidated operator runbook | Single page, one URL, easy top-to-bottom skim | ✓ (combined) |
| One page per command | More files, independently linkable/searchable | |
| Task-oriented sections cutting across commands | Organized by operator goal, most reader-friendly | ✓ (combined) |

**User's choice:** "Combination of first and third options to address more user entry points and needs" (free text) — interpreted and confirmed as: one consolidated page, organized internally with task-oriented "how do I...?" framing rather than a flat per-command reference.

| Option | Description | Selected |
|--------|-------------|----------|
| Own dedicated section | Approval-queue actions are a web-UI workflow, distinct from CLI commands | |
| Folded into calendar-sync docs | Keeps everything calendar-related in one place | ✓ |

**User's choice:** Folded into calendar-sync docs

---

## Audience & Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Assumes manage.py familiarity | Runbook focuses on FOMO-specific content, not general Django orientation | |
| No Django background assumed | Spells out venv/migrations/manage.py basics | ✓ (modified) |

**User's choice:** "The second one of assuming no Django background but put this in a separate section after Installation" (free text) — interpreted and confirmed as: Django-basics onboarding content lives in its own section positioned right after the existing Installation doc, not duplicated into the runbook proper; the runbook itself still assumes that baseline via cross-reference.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, include cheat-sheet | Quick reference table for readers who know the workflow | ✓ |
| No, narrative only | Simpler to write/maintain | |

**User's choice:** Yes, include cheat-sheet

---

## Failure/Troubleshooting Coverage

| Option | Description | Selected |
|--------|-------------|----------|
| Happy-path + known real failure modes | How-to-run plus troubleshooting for failures already seen in production | ✓ |
| Happy-path only | Relies on existing error messages/logging | |
| Comprehensive troubleshooting guide | Enumerate every documented error path, including untriggered ones | |

**User's choice:** Happy-path + known real failure modes

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, document the FTN timezone gap | Real, currently unfixed data gap that will keep recurring until fixed | ✓ |
| No, out of scope | Leave as implicit error-message consequence | |

**User's choice:** Yes, document the FTN timezone gap

---

## Claude's Discretion

- Exact page title/filename within `docs/runbooks/`
- Exact section ordering and heading hierarchy within the task-oriented structure
- Whether the Django-basics onboarding content is a new `.rst` file or a new subsection appended to `docs/installation.rst`

## Deferred Ideas

None raised during discussion — stayed within phase scope. Two pending todos were reviewed against phase scope (site/telescope mapping extraction, calendar_utils rename) and found to be code-refactor work, not documentation — left un-folded for a future tech-debt phase.
