---
phase: 24-operator-and-usage-runbook-documentation-for-the-telescope-r
reviewed: 2026-07-18T00:00:00Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - docs/index.rst
  - docs/installation.rst
  - docs/runbooks/telescope_runs_calendar.rst
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: issues_found
---

# Phase 24: Code Review Report

**Reviewed:** 2026-07-18T00:00:00Z
**Depth:** deep
**Files Reviewed:** 3
**Status:** issues_found

## Summary

This is a docs-only phase adding a new operator runbook
(`docs/runbooks/telescope_runs_calendar.rst`) and a new "Running FOMO
Management Commands" subsection of `docs/installation.rst`, plus the
`docs/index.rst` toctree entry that wires the runbook in.

Verification performed at deep depth:

- **Sphinx build check.** Ran `python -m sphinx -b dummy` against `docs/`
  (autodoc/autoapi disabled to avoid the SPICE-kernel-downloading import
  side effect noted in `CLAUDE.md`). Zero warnings were emitted for any of
  the three reviewed files — no broken `:ref:`/`:doc:` targets, no
  duplicate/undefined labels, no orphaned-document warnings, no title
  underline-length warnings. The only warnings the build produced are
  pre-existing and unrelated to this diff (`autoapi/index` and the
  `notebooks/pre_executed/*` entries in `docs/notebooks.rst`, which are only
  populated by a full build with autoapi/nbsphinx enabled).
- **Label/cross-reference wiring.** `docs/installation.rst`'s
  `.. _running-management-commands:` label and the runbook's
  `.. _command-cheat-sheet:` label both resolve correctly to the runbook's
  two `:ref:` uses (`running-management-commands` at line 14, and the two
  `command-cheat-sheet` uses at lines 136/246). No typos, no case mismatches.
- **PII check (T-24-01).** No `contact_person`/`contact_email`/email-address
  content appears anywhere in the runbook; the troubleshooting section's own
  disclaimer ("no real contact information appears anywhere on this page")
  is accurate. The one real-world identifier used, `3I/ATLAS`, is a public
  comet designation already used as the project's standing example
  throughout `solsys_code/` (`models.py`, `campaign_utils.py`,
  `import_campaign_csv.py`), not PII.
- **Factual accuracy against the implementation (cross-file, deep-depth
  requirement).** Every specific behavioral claim in the runbook was traced
  back to the actual command/view source and confirmed correct, including:
  - `sync_lco_observation_calendar --proposal` exact-match/no-substring-leak
    semantics (`ObservationRecord.objects...filter(parameters__proposal__in=codes)`
    in `solsys_code/management/commands/sync_lco_observation_calendar.py`) and
    the case-insensitive `ALL` sentinel (`_parse_proposal_arg`).
  - The `[UNVERIFIED]` fallback and separate `telescope_api_failed` counter
    (`_build_event_fields`/`Command.handle` in the same file).
  - `sync_gemini_observation_calendar` genuinely takes no flags at all
    (`add_arguments` is a no-op `pass`).
  - The exact `Observatory '<code>' (obscode=<code>) has no timezone set`
    error text and `Line N: ... (line text: ...)` / `Done. lines processed:
    ...` summary format in `solsys_code/telescope_runs.py` and
    `solsys_code/management/commands/load_telescope_runs.py`.
  - `import_campaign_csv`'s `site_needs_review`/`window_needs_review`
    flag-and-continue behavior and the `target`-reset-on-reimport gotcha
    (`solsys_code/management/commands/import_campaign_csv.py`).
  - `backfill_range_calendar_events`'s `--dry-run` flag, its
    already-has-an-event skip check, and that it only targets true
    range-window runs (`.exclude(window_start=F('window_end'))`).
  - The Decided-table "Mark Cancelled"/"Mark Weathered" buttons rendering
    unconditionally for any `APPROVED` row, the `http_method_names =
    ['post']` (no GET-triggered state change), and the idempotent re-click
    behavior (`solsys_code/campaign_views.py`,
    `solsys_code/campaign_tables.py`).

  No factual discrepancies were found between the runbook's claims and the
  actual code behavior.

One real defect was found: a new cross-document pointer in
`installation.rst` that points at the runbook only by its title in plain
prose, rather than via the `:doc:`/`:ref:` mechanism the project already
uses elsewhere (including inside the runbook itself, and in
`docs/design/tom_calendar_vs_yse_pz_calendar.rst`) — see WR-01.

## Warnings

### WR-01: Forward reference to the new runbook is plain text, not a working hyperlink

**File:** `docs/installation.rst:148-150`
**Issue:** The new "Running FOMO Management Commands" section ends with:

```rst
For a task-oriented walkthrough of the specific commands and staff actions
that keep the telescope runs calendar and campaign coordination up to date,
see the Telescope Runs Calendar Operator Runbook.
```

"Telescope Runs Calendar Operator Runbook" is plain prose, not a
`:doc:`/`:ref:` role. It will not render as a clickable link in the built
HTML, and if the runbook is ever renamed or moved, this sentence goes
silently stale with **no Sphinx build warning** (unlike a real `:doc:`/`:ref:`
reference, which would fail the build). This is the one place in the new
content where the project's own established convention for cross-document
references — used correctly by the runbook itself (`:doc:`/design/telescope_runs_calendar``)
and by `docs/design/tom_calendar_vs_yse_pz_calendar.rst` (`:doc:`telescope_runs_calendar``)
— was not followed, undermining part of the wiring this phase set out to
establish.

**Fix:**
```rst
For a task-oriented walkthrough of the specific commands and staff actions
that keep the telescope runs calendar and campaign coordination up to date,
see the :doc:`Telescope Runs Calendar Operator Runbook
</runbooks/telescope_runs_calendar>`.
```

## Info

### IN-01: Same-page reference also uses plain text instead of a Sphinx label

**File:** `docs/installation.rst:144-146`
**Issue:** "see 'Initializing FOMO and the database' above" is a quoted
section title rather than a `:ref:` to an anchor. Lower severity than WR-01
since it's a same-page reference the reader can locate by scrolling, and
the section has no existing label to `:ref:` to (adding one would require
inserting a new anchor). Still a minor inconsistency with the project's
`:doc:`/`:ref:` convention used elsewhere in these same files.
**Fix:** Add a label above the `Initializing FOMO and the database` heading
(e.g. `.. _initializing-fomo:`) and reference it with
`:ref:`Initializing FOMO and the database <initializing-fomo>``.

### IN-02: Toctree wires the runbook as a single file under a plural "Runbooks" caption, with no index page

**File:** `docs/index.rst:24`
**Issue:** `Runbooks <runbooks/telescope_runs_calendar>` maps the plural
sidebar caption "Runbooks" directly to one file. This works today (there is
exactly one runbook), and the Sphinx dummy-build check confirms it resolves
cleanly with no orphan warning. But `docs/design/design.rst` shows the
project's established pattern for a *collection* of documents is a
dedicated index file (`design/design.rst`) with its own `toctree` fanning
out to each design doc — not a single top-level entry per document. Given
this project explicitly describes this runbook as "Stage 1 of the... issue
#37" feature with Stages 2-4 still to come (per the runbook's own intro and
`CLAUDE.md`'s project description), it's likely more runbooks will be added
later, at which point this direct single-file mapping will need to be
restructured into a `runbooks/index.rst`-style pattern anyway. Not a bug
today, but worth deciding now rather than as an unplanned rework later.
**Fix:** Either accept the single-file mapping as scoped to "this project has
exactly one runbook for now," or preemptively create
`docs/runbooks/index.rst` with its own toctree (mirroring
`docs/design/design.rst`) and point `docs/index.rst` at that index instead.

---

_Reviewed: 2026-07-18T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
