# Phase 24: Operator and Usage Runbook Documentation - Pattern Map

**Mapped:** 2026-07-18
**Files analyzed:** 3 (new/modified docs files)
**Analogs found:** 3 / 3

This is a documentation-only phase (Sphinx `.rst` content, no application source
code). "Role" below is repurposed for doc structure (landing page, content page,
toctree entry) instead of code roles; "Data flow" is repurposed for the doc's
authoring pattern (narrative prose, reference table, navigation wiring).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|---------------|
| `docs/runbooks/telescope_runs_calendar.rst` (new; exact filename Claude's discretion) | content page (task-oriented how-to + cheat-sheet + troubleshooting) | narrative prose + reference table | `docs/design/telescope_runs_calendar.rst` | exact (same feature area, same house style, narrative RST with list-tables) |
| `docs/installation.rst` (append "Running FOMO Management Commands" subsection) | content page (append new subsection to existing page) | narrative prose + code-block console | `docs/installation.rst` itself (self-consistent — new subsection should mirror the existing subsections' structure) | exact (editing in place, so the analog is the file's own established conventions) |
| `docs/index.rst` (add one toctree line) | navigation wiring (toctree entry) | structural / list wiring, no prose | `docs/index.rst`'s existing toctree block (lines 18-25) | exact (single-line addition to an existing directive, not a new pattern) |

## Pattern Assignments

### `docs/runbooks/telescope_runs_calendar.rst` (new content page)

**Analog:** `docs/design/telescope_runs_calendar.rst` (same feature area; only
existing narrative-doc precedent in the repo) and `docs/design/design.rst` (for
the toctree/section-heading conventions if a runbooks-local toctree is ever
needed — not required for this phase since D-05 locks it to one page).

**Title + underline convention** (`docs/design/telescope_runs_calendar.rst` lines 1-2):
```rst
Telescope Runs on the Calendar
==============================
```
Section-title underline character (`=`) matches page title length; subsection
headings use `-` (see lines 9-10, `Background\n----------`), sub-subsections use
`^` (see `docs/design/design.rst` lines 18-19, `System Context\n^^^^^^^^^^^^^^^`).

**list-table pattern for tabular data** (`docs/design/telescope_runs_calendar.rst` lines 50-53, 134-137):
```rst
.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - Field
     - Type
     - Where it shows
     - Use for runs
   * - ``title``
     - ``CharField(200)``
     - **Grid label** (truncated ~16 chars) and edit modal
     - Short label; the only place to surface status at a glance
```
The D-10 cheat-sheet (Command | Key Flags | One-line description) should use
this exact `list-table` + `:header-rows: 1` + `:widths:` structure — a 3-column
variant of the pattern above.

**Bold-lead-in + monospace command/field names** (`docs/design/telescope_runs_calendar.rst` lines 32-42, 246-266):
```rst
**The feature is feasible with no changes to** ``tom_calendar`` **and no
database migrations.**  The stock :class:`tom_calendar.models.CalendarEvent`
already carries ``title``, ``description``, ``start_time``, ``end_time``, ...
```
```rst
**Stage 1 — site / ephemeris helper.**  A small module (e.g.
``solsys_code/telescope_runs.py``) with:
```
Use bold lead-in sentences for each new subsection/task ("How do I add a
classical schedule?" etc.), and double-backtick monospace for command names
(` ``load_telescope_runs`` `) and field names — never Sphinx `:func:`/`:class:`
roles for management commands, since those roles only resolve for
autoapi-indexed Python objects (per RESEARCH.md Pattern 3).

**Bulleted "Observed examples::" literal block for input formats** (`docs/design/telescope_runs_calendar.rst` lines 172-177):
```rst
Runs are recorded as free text lines, ``telescope instrument [status] daterange
[(status)]``.  Observed examples::

    NTT EFOSC2 allocation 9-13 July
    Magellan IMACS 13-19 July (proposed)
    Magellan Proto-Lightspeed  Jul 8-12 (proposed)
```
Use this literal-block (`::` + indented lines) convention for any CSV-row or
log-line examples in the troubleshooting section — remembering Pitfall 4
(synthetic PII only, e.g. `Jane Doe` / `jane@example.com`, never real rows).

**Verbatim safe-to-quote failure text** (from RESEARCH.md Code Examples, confirmed real):
```
Observatory 'FTN' (obscode=F65) has no timezone set
```
Source: `.planning/phases/25-.../25-UAT.md`. Safe to quote verbatim (no PII —
only an MPC obscode/telescope short-name). Use it directly in the
troubleshooting subsection for D-12, framed with a fix-it step: set
`Observatory.timezone` to a valid IANA name (e.g. `"America/Santiago"`) via the
Django admin or `CreateObservatory` form.

**Cross-reference to design doc, not duplication** — the runbook should link to
`docs/design/telescope_runs_calendar.rst` (e.g. via a `:doc:` role or plain
prose pointer, matching how `docs/index.rst` links `Design <design/design>`)
for "why" content (dip correction, -15 deg window rationale) rather than
re-explaining the astronomy.

---

### `docs/installation.rst` (append "Running FOMO Management Commands" subsection)

**Analog:** the file's own existing subsections (self-consistent conventions).

**Subsection heading convention** (`docs/installation.rst` lines 92-93, 77-78):
```rst
Starting up the webserver
--------------------------------
```
```rst
Initializing FOMO and the database
-------------------------------------
```
New subsection should be appended after "Starting up the webserver" (the last
existing section, ending line 118), following the exact same `-` underline
convention, e.g.:
```rst
Running FOMO Management Commands
--------------------------------------
```

**`code-block:: console` with `>>` prompt convention** (`docs/installation.rst` lines 82-85, 97-99):
```rst
.. code-block:: console

   >> python3 manage.py migrate
   >> python3 manage.py createsuperuser
```
The new subsection's example invocations (e.g. `>> python3 manage.py
load_telescope_runs <filepath>`) must reuse this exact `code-block:: console`
directive + `>>` prompt style — not `bash` or a plain `::` literal block — for
consistency with every other shell example in this doc.

**`.. note::` admonition for asides** (`docs/installation.rst` lines 6-8, 38-39):
```rst
.. note::
   ``fomo`` is both pip both and conda/mamba installable (but the latter is
   less used/tested by the developers). We strongly recommend installing into a virtual environment in either case.
```
Use this admonition style if the new subsection needs a caveat callout (e.g.
noting that `manage.py`/venv/migration familiarity is assumed by the runbook
proper, per D-08/D-09).

**Warning-quoted-log-line convention** (`docs/installation.rst` lines 88-90):
```rst
The second command creates an admin user for FOMO ... You can ignore any warnings that look like::

   User <foo> is not logged in. Cannot re-encrypt sensitive data. Clearing all encrypted fields instead.
```
Model for how to present expected/benign warning output inline with prose —
directly reusable pattern for troubleshooting content in the runbook page too.

---

### `docs/index.rst` (toctree edit)

**Analog:** the existing toctree block itself (lines 18-25).

**Current toctree, verbatim** (`docs/index.rst` lines 18-25):
```rst
.. toctree::
   :hidden:

   Home page <self>
   Installation and Getting Started <installation>
   Design <design/design>
   API Reference <autoapi/index>
   Notebooks <notebooks>
```

**New line to add** (D-04; per RESEARCH.md, single `Label <path>` entry, no
directory-prefix needed if the runbook page is directly under `docs/runbooks/`):
```rst
   Runbooks <runbooks/telescope_runs_calendar>
```
Insert as a new line within the existing `:hidden:` toctree (position: after
"Design", before "API Reference" is a reasonable default matching the
install -> design -> runbooks -> reference reading order, but exact position is
Claude's discretion — RESEARCH.md doesn't lock it).

---

## Shared Patterns

### RST heading underline hierarchy
**Source:** `docs/design/telescope_runs_calendar.rst` (title `=`, section `-`,
sub-subsection `^`) and `docs/design/design.rst` (same hierarchy).
**Apply to:** all new/modified `.rst` content in this phase — keep heading
levels consistent with this existing convention so Sphinx's implicit heading
depth doesn't produce inconsistent nesting warnings.

### `code-block:: console` + `>>` prompt
**Source:** `docs/installation.rst` lines 41-44, 49-52, 58-62, 82-85, 97-99.
**Apply to:** every shell/command example across the new runbook page and the
new installation subsection — this is the one established convention for shell
examples in this codebase's docs; do not switch to `bash` or `$`-prompted
blocks.

### `list-table` with `:header-rows:` / `:widths:`
**Source:** `docs/design/telescope_runs_calendar.rst` lines 50-53, 134-137.
**Apply to:** the D-10 cheat-sheet table (command / key flags / description)
in the new runbook page.

### Double-backtick monospace for names, not Sphinx roles
**Source:** `docs/design/telescope_runs_calendar.rst` (` ``CalendarEvent`` `,
` ``sun_event(site, date, kind)`` `, ` ``load_telescope_runs`` ` throughout).
**Apply to:** every mention of a management command name, model field, or
CLI flag in both new/modified files — plain double-backtick, never `:func:`/
`:class:` roles (those only resolve for autoapi-indexed Python objects).

### `.. note::` admonition for asides
**Source:** `docs/installation.rst` lines 6-8, 38-39.
**Apply to:** the Observatory-timezone gotcha (D-12) if it needs a standalone
callout separate from the main troubleshooting prose, and any other aside in
either file.

## No Analog Found

None. All three files/edits in scope have a strong, in-repo analog (the design
doc for the new runbook page's house style, the installation doc's own
conventions for its own new subsection, and the index toctree's existing block
for the one-line addition). No RESEARCH.md code-example fallback is needed.

## Metadata

**Analog search scope:** `docs/` (all `.rst` files: `index.rst`,
`installation.rst`, `design/design.rst`, `design/telescope_runs_calendar.rst`,
`notebooks.rst`).
**Files scanned:** 5 (`docs/index.rst`, `docs/installation.rst`,
`docs/design/design.rst`, `docs/design/telescope_runs_calendar.rst`, plus
CONTEXT.md/RESEARCH.md for canonical-ref confirmation).
**Pattern extraction date:** 2026-07-18
