# Phase 26: Canonical-Record Spike - Pattern Map

**Mapped:** 2026-07-27
**Files analyzed:** 6 (2 committed deliverables + 4 throwaway/investigation artifacts)
**Analogs found:** 6 / 6

**Framing reminder (from orchestrator instructions):** this is an investigation-only spike.
There is no application code, no migration, and no new module in the *committed* deliverable
set — only a decision doc and a durable `docs/design/` page survive phase close. The four
throwaway items below (scratch `local_settings.py`, hand-authored scratch migration,
`tmp/26_integrity_check.py`, `tmp/26_reconciler_prototype.py`, plus a dated DB snapshot) are
git-excluded investigation tooling, not shipped code — they are classified here only because
the planner needs concrete analogs to scope the investigation tasks correctly, not because they
belong in `files_modified` for the phase's real git history.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `.planning/phases/26-canonical-record-spike/26-DECISION.md` (committed) | doc / decision-record | batch (write-once findings doc) | `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` | exact |
| `docs/design/<name>.rst` (committed, e.g. `canonical_record_spike.rst`) | doc / durable design page | batch (rendered once, read many) | `docs/design/uncertain_scheduling_spike.rst` and `docs/design/eso_feasibility_spike.rst` | exact |
| `docs/design/design.rst` toctree entry (committed, 1-line edit) | config / doc index | batch | existing `.. toctree::` block in `docs/design/design.rst` | exact |
| scratch `local_settings.py` (throwaway, git-excluded) | config | file I/O (DB path override) | none tracked in repo — pattern is the project's own already-gitignored `local_settings.py` escape hatch referenced at the end of `src/fomo/settings.py` | role-match (mechanism exists in settings.py, no example file itself is tracked) |
| hand-authored scratch migration `solsys_code/migrations/0008_*.py` (throwaway, scratch-branch only) | migration | batch (schema DDL applied once to a DB copy) | `solsys_code/migrations/0007_campaignrun_contact_public_opt_in.py` (structure) and `solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py` (multi-op precedent) | exact (structure); no `RenameModel`-shaped analog exists in this migrations dir, see note below |
| `tmp/26_integrity_check.py` / `tmp/26_reconciler_prototype.py` (throwaway, git-excluded scripts) | utility / probe script | request-response (calls into `calendar_utils`/ORM, prints verbatim evidence) | Phase 13's `eso_p2_probe.py` precedent (**no longer present on disk — see note below**); the concrete call target is `solsys_code/calendar_utils.py:insert_or_create_calendar_event()` | role-match (precedent confirmed only via `.planning/` records, not a live file) |

## Pattern Assignments

### `.planning/phases/26-canonical-record-spike/26-DECISION.md`

**Analog:** `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`

**Section skeleton to copy** (verified structure of the real file, in order):
```
# Phase N: <Title> - Decision

**Investigated:** <date>
**Status:** Complete. <one-paragraph status incl. evidence-vehicle summary>

<one paragraph restating phase boundary + the throwaway-script name + what was NOT built>

## Findings

### <Requirement-ID> criterion N — <short title>

<prose finding, each factual claim followed by a source citation in the form
`[VERIFIED: <how>]` or, for constructed-not-real evidence, an explicit confidence note>

## Recommendation

<one subsection per criterion/decision, each headed "### Criterion N — <verdict>",
each **restating the locked CONTEXT.md decision** and then grounding it in the
Findings section above (not asserting it fresh)>

## Durable summary

See `docs/design/<page>.rst` for the durable, redaction-free summary of these decisions.
```

**The "confirmed against a real row" vs "confirmed via constructed input" discipline
(18-DECISION.md's D-09, which Phase 26's D-18 reapplies)** — copy this exact citation
convention verbatim into every finding that touches Gemini identity mapping (D-18 has zero
`GEM:` rows to check against):

```
line 143: | `500@-170` | Confirmed against real rows (the real JWST `Site Code` value, D-09) | ...
line 144: | `250`      | Confirmed against real rows (Jewitt's/Noonan's Hubble rows both use `250`) | ...
line 145: | `274`      | Constructed-input code-path check (no real row in the current snapshot types plain `274`) | ...
line 146: | `289`      | Constructed-input code-path check (no real row types plain `289`) | ...
```
The pattern: a two-word tag — `Confirmed against real rows (<why>)` or `Constructed-input
code-path check (<why>)` — placed as its own table column or parenthetical next to *every*
finding whose evidence quality differs. For Phase 26's D-18 (zero `GEM:`-namespaced events),
this becomes something like: "SPIKE-02's Gemini mapping: **Constructed-input code-path check**
(no `GEM:` events exist in the dev DB to confirm against; reasoned from `sync_gemini_
observation_calendar.py`'s own identity-key construction instead)."

**"Two-vehicle" evidence framing** (18-DECISION.md lines 8-16) — copy the pattern of naming
the throwaway script by filename and stating explicitly what was/wasn't persisted:
```
...built from a throwaway, git-excluded probe script (`fuzzy_match_probe.py`, never
staged, per D-08) run by the executor against the real CSV (D-01 path) and the local
`Observatory` DB (read-only; every `resolve_site()` call was wrapped in a rolled-back
`transaction.atomic()` block — `Observatory.objects.count()` was 8 at both the start and
the end of the run, confirming nothing was persisted).
```
Phase 26's own equivalent framing per D-01/D-04/RESEARCH.md's "Throwaway-Evidence Mechanics"
section is **not** the rollback pattern — RESEARCH.md explicitly calls this out as "a
genuinely different evidence posture from Phase 18's": Phase 26 writes for real against a
disposable DB copy (`tmp/26-spike-db-copy.sqlite3`) rather than rolling back against the live
DB. The decision doc should state this difference explicitly, mirroring how 18-DECISION.md
states its own rollback mechanic — do not silently reuse Phase 18's wording, adapt it to
name the DB-copy mechanism instead.

### `docs/design/<name>.rst` (new durable spike page)

**Analogs:** `docs/design/uncertain_scheduling_spike.rst` (full text above) and
`docs/design/eso_feasibility_spike.rst` (full text above) — both read in full; excerpts below.

**Title/underline convention** (RST section-title underline length must equal or exceed the
title's character count; both analogs use `=` for H1, `-` for H2, `^` for H3):
```rst
Uncertain-Scheduling Investigation Spike
========================================

Background
----------

Key finding
-----------

Decisions
---------

Future scope
------------
```
and, from the ESO page (an alternate section-name set for the same page shape):
```rst
ESO/VLT Calendar Sync — Feasibility Spike
==========================================

Investigation summary
----------------------
```
Both pages use the same four/five-section shape: **Background → Key finding → <a table
section, named "Decisions" or "Investigation summary"> → Future scope**. Phase 26's page
should follow this shape given its own decision count (D-01..D-20) — likely renaming the
table section to something like "Decisions" (uncertain-scheduling's naming, since Phase 26
is also settling several discrete named decisions, not one investigation-summary table).

**Opening paragraph pattern** — both pages open by (1) naming what was investigated, (2) the
date of the live investigation, (3) explicitly stating what was NOT built, and (4) pointing
at the full-detail companion doc with the archival-path caveat:
```rst
This document records the investigation spike that settled five open design
decisions for FOMO's ``CampaignRun`` scheduling model against the real
3I/ATLAS coordination sheet (2026-07-09 snapshot). ... No
``CampaignRun`` schema migration, no CSV importer change, and no fuzzy-match
UI code was built during this spike — the deliverable is this durable
summary and its full-detail companion, ``18-DECISION.md`` (originally at
``.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md``;
this project's milestone-archival workflow moves completed phase directories
to ``.planning/phases-archive/`` once their milestone closes, so check there
first if the original path no longer resolves).
```
Phase 26's page must use the equivalent phrase for its own committed decision doc filename
(`26-DECISION.md`) and its own current path
(`.planning/phases/26-canonical-record-spike/26-DECISION.md`), with the same archival caveat.

**`.. list-table::` directive usage** (both pages use `:header-rows: 1` and an explicit
`:widths:` list summing to 100, RST `*` bullet-of-bullets syntax for rows/cells):
```rst
.. list-table::
   :header-rows: 1
   :widths: 22 48 12

   * - SCHED-01 criterion
     - Decision
     - Phase
   * - Window field schema
     - Nullable ``window_start``/``window_end`` ``DateField`` pair, confirmed
       against real single-date, ranged, and TBD cell shapes.
     - 19
```
and the ESO page's variant (3 columns instead, `:widths: 22 20 58`):
```rst
.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Capability
     - Status
     - Notes
```
Phase 26's page has ~20 decisions (D-01..D-20) grouped into several thematic clusters —
consider one `.. list-table::` per cluster (evidence standard / stage-2 semantics / event
key & ownership / source vocabulary / measured findings) rather than one 20-row table, to
stay readable; this is Claude's Discretion per CONTEXT.md.

**Double-backtick inline-code convention**: both pages use RST double-backticks
(````CampaignRun````) for identifiers, never single backticks or Markdown-style code spans —
this must be followed exactly or the Sphinx build will render literally-backtick text instead
of `<code>` spans.

**Internal cross-reference convention**: both pages reference their own full-detail companion
by filename in double-backticks with a prose parenthetical, not a Sphinx `:doc:`/`:ref:` role
— e.g. ` ``18-DECISION.md`` (path note above)`. Follow this same plain-filename-reference
style rather than introducing a Sphinx cross-reference role neither analog uses.

**Sphinx build must stay clean (pre-commit runs it)** — both analogs avoid any directive not
already used elsewhere in `docs/design/` (`.. image::`, `.. toctree::`, `.. list-table::`);
stick to that same restricted directive set for the new page.

### `docs/design/design.rst` — toctree entry

**Analog:** the existing toctree block itself (`docs/design/design.rst` lines 36-44):
```rst
Design Notes
------------

.. toctree::
   :maxdepth: 1

   telescope_runs_calendar
   tom_calendar_vs_yse_pz_calendar
   gsd_experiment
   eso_feasibility_spike
   uncertain_scheduling_spike
```
**Important correction to the pattern-mapping brief's assumption:** the new page's toctree
entry belongs in **`docs/design/design.rst`** (the design-notes index this block already
lives in, alongside the two existing spike pages), **not** directly in the top-level
`docs/index.rst` — `docs/index.rst`'s own toctree (read in full above) already points at
`Design <design/design>` as a single entry and does not itself list individual design pages.
Add the new page's basename (no `.rst` extension, matching every existing entry's style,
e.g. `canonical_record_spike`) as one more line under the existing five, in the same
order-of-addition style (each analog spike was appended at the end of the list, not inserted
alphabetically — `eso_feasibility_spike` predates `uncertain_scheduling_spike` and sits above
it, matching creation order not alphabetical order).

## Shared Patterns

### Throwaway/git-excluded investigation-code pattern (Phase 13 precedent)

**Source:** `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-CONTEXT.md` /
`13-DECISION.md` / `13-01-PLAN.md` / `13-VERIFICATION.md` (all read this session).

**No trace of `eso_p2_probe.py` survives on disk** — confirmed this session: `find` across
the whole repo for any file matching `*eso_p2_probe*` returned nothing, and
`git log --all --oneline -- '*eso_p2_probe*'` returned nothing. This is exactly the intended
outcome of D-09's "throwaway, git-excluded, never committed" discipline — stating this
plainly per the pattern-mapping brief's instruction, rather than guessing at file contents
that no longer exist. What *is* recoverable is the mechanism, from the planning-doc trail:

```
13-VERIFICATION.md:56: | `eso_p2_probe.py` (intentional non-deliverable, D-09) | Git-excluded,
never committed, read-only only | VERIFIED | `git check-ignore -v` matches
`.git/info/exclude:18`; `git log --all -- eso_p2_probe.py` returns nothing (never
staged/committed); 0 write-style p2api call names present in the file on disk |
```

**The concrete mechanism to replicate for Phase 26's four throwaway artifacts**
(`tmp/26_integrity_check.py`, `tmp/26_reconciler_prototype.py`, scratch `local_settings.py`,
the scratch migration): register each path in **`.git/info/exclude`** (a local,
never-committed exclude file — distinct from the tracked, committed `.gitignore`), confirm
with `git check-ignore -v <path>`, and confirm with `git log --all -- <path>` that nothing
was ever staged. RESEARCH.md's own recommended layout already covers `tmp/` (already matched
by the tracked `.gitignore:145` entry) and `local_settings.py` (already matched by
`.gitignore:61`) — so for those two, the tracked `.gitignore` is sufficient and
`.git/info/exclude` is not strictly needed; `.git/info/exclude` becomes necessary only for
the scratch `solsys_code/models.py` edits and the new migration file, which sit inside an
already-tracked directory and must instead be isolated by **never merging the scratch git
branch** (RESEARCH.md's own recommended mechanism, step 5/8 of its procedure) rather than by
gitignore matching (a modification to an already-tracked file cannot be gitignored).

### Hand-authored migration structure

**Source:** `solsys_code/migrations/0007_campaignrun_contact_public_opt_in.py` (full file,
19 lines, reproduced above) and `solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py`
(multi-`AddField` precedent — same `dependencies`/`operations` shape, just two operations
instead of one).

**Apply to:** the throwaway scratch migration (`solsys_code/migrations/0008_*.py`,
scratch-branch only, never committed to the real phase-26 branch).

```python
# Generated by Django 5.2.15 on 2026-07-11 11:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('solsys_code', '0006_campaignrun_original_obs_date_raw_and_window_needs_review'),
    ]

    operations = [
        migrations.AddField(
            model_name='campaignrun',
            name='contact_public_opt_in',
            field=models.BooleanField(default=False, verbose_name='Show contact info publicly?'),
        ),
    ]
```

**No `RenameModel`-shaped migration exists yet in `solsys_code/migrations/`** — every
migration in the directory (`0001`..`0007`) is `AddField`/`CreateModel`/`AlterField`-only,
confirmed by directory listing this session (`0005_campaignrun_campaign_run_window_start_end_null_together.py`,
`0006_...`, `0007_...`, plus earlier numbers not individually inspected but none named with
`rename` in the filename). This means Phase 26's `RenameModel`-before-`AddField` shape (per
RESEARCH.md's "Pattern 2: Hand-authored migration, `RenameModel` before `AddField`" and its
full code skeleton, `solsys_code/migrations/0008_scratch_canonical_record_probe.py`, quoted
in full in RESEARCH.md lines ~710-737) has **no existing in-repo precedent to copy from
directly** — the executor should follow RESEARCH.md's own skeleton (already Django-API-correct
per that section's own verification against the installed Django 5.2.13), using this repo's
migrations only for the *file-header/dependencies/operations-list* boilerplate shape (as
shown above), not for the `RenameModel` operation itself.

**Note the next-free migration number**: `0007_campaignrun_contact_public_opt_in.py` is
current head; the scratch migration is `0008_*` per RESEARCH.md's own confirmed numbering.

### `insert_or_create_calendar_event()` — the one piece of application code the throwaway scripts call

**Source:** `solsys_code/calendar_utils.py:318-378` (`insert_or_create_calendar_event`, full
docstring + body read this session; `_update_or_unchanged` helper at lines 296-315 read in
the same pass).

**Apply to:** both `tmp/26_integrity_check.py` (SPIKE-01's coexistence proof, which does not
call this function directly — see the exact script body in RESEARCH.md's "IntegrityError
Coexistence Check" section, already reproduced there in full) and
`tmp/26_reconciler_prototype.py` (D-11's adopt-vs-gap-fill prototype, which *does* call this
function directly, per RESEARCH.md's "Prototyping Adopt-vs-Gap-Fill" section).

**Signature and caller-supplied `lookup` dict contract:**
```python
def insert_or_create_calendar_event(
    lookup: dict[str, Any],
    fields: dict[str, Any],
    *,
    start_time_tolerance: timedelta | None = None,
) -> tuple[CalendarEvent, str]:
```
- `lookup`: the exact keyword-argument mapping passed to
  `CalendarEvent.objects.get_or_create(**lookup, defaults=fields)` — this is the mechanism
  D-09 relies on: the reconciler prototype passes `{'url': f'RUN:{run_pk}:{date}'}` as its
  own namespaced key, with zero changes needed to this shared helper.
- `fields`: the value mapping applied on create or on update-if-changed; **not merged with
  `lookup`** — caller must ensure the combined key+fields set is complete.
- Returns `(event, action)` where `action` is `'created'` / `'updated'` / `'unchanged'` — the
  D-11 prototype script's per-copy summary (RESEARCH.md: "total `CalendarEvent` count... the
  url-key of each... whether the companion row's `run` FK was set") should tally these three
  action values directly rather than re-deriving change detection itself.

**The `start_time_tolerance` proximity path** (used by `load_telescope_runs`, not by the
Phase 26 reconciler prototype itself, but cited in RESEARCH.md/CONTEXT.md as evidence of the
"key on something other than an exact string" precedent D-19 draws on for classical events'
blank `url`):
```python
if start_time_tolerance is not None and 'start_time' in lookup:
    start_time = lookup['start_time']
    key = {k: v for k, v in lookup.items() if k != 'start_time'}
    window = (start_time - start_time_tolerance, start_time + start_time_tolerance)
    existing = CalendarEvent.objects.filter(**key, start_time__range=window).order_by('start_time').first()
    if existing is not None:
        return _update_or_unchanged(existing, fields)
    return CalendarEvent.objects.create(**lookup, **fields), 'created'

event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)
if created:
    return event, 'created'
return _update_or_unchanged(event, fields)
```
The Phase 26 reconciler's `RUN:{run_pk}:{date}` key (D-09) uses the plain `get_or_create`
exact-match branch (the URL-keyed callers' path, tolerance=`None`) — it does not need
`start_time_tolerance` itself, since its identity key is a synthetic string, not a computed
sun-event timestamp. This confirms D-19's framing: the tolerance path exists specifically
because `load_telescope_runs`' `start_time` drifts; the reconciler's synthetic `url` key has
no equivalent drift problem.

## No Analog Found

None — every file in this phase's file set has at least a role-match analog. The scratch
`local_settings.py` and the throwaway probe scripts have no *tracked file* to point at (by
design — they are git-excluded and, in Phase 13's case, no longer exist on disk), but the
project convention and mechanism they must follow is fully documented in tracked `.planning/`
records and `src/fomo/settings.py` itself, both cited above.

## Metadata

**Analog search scope:** `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/`,
`.planning/milestones/v1.7-phases/13-eso-feasibility-spike/`, `docs/design/`,
`solsys_code/migrations/`, `solsys_code/calendar_utils.py`, `src/fomo/settings.py`, repo-wide
`find`/`git log --all` for `eso_p2_probe.py`.
**Files scanned:** 6 read in full (18-DECISION.md, uncertain_scheduling_spike.rst,
eso_feasibility_spike.rst, docs/index.rst, docs/design/design.rst via directory listing +
targeted cat, 0007 migration), 2 targeted reads (calendar_utils.py lines 300-379,
13-*.md grep pass), 1 repo-wide existence check (eso_p2_probe.py).
**Pattern extraction date:** 2026-07-27
