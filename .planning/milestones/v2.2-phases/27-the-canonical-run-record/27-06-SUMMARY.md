---
phase: 27-the-canonical-run-record
plan: 06
subsystem: api
tags: [django, management-command, jupyter, sphinx, docs, campaign]

# Dependency graph
requires:
  - phase: 27-the-canonical-run-record (plan 01)
    provides: "calendar_utils.derive_telescope_class(site_raw, telescope_instrument) -- the shared, primitives-only derivation this plan's importer calls as D-20's second required call site"
  - phase: 27-the-canonical-run-record (plan 04)
    provides: "CampaignRun.Source/TelescopeClass TextChoices -- the enum members import_campaign_csv now writes"
  - phase: 27-the-canonical-run-record (plan 02)
    provides: "repair_stale_campaign_run_sites management command -- documented in this plan's runbook section"
provides:
  - "import_campaign_csv writes source=CSV_IMPORT and a derived telescope_class on every imported row (CANON-01/CANON-02 ingest-side completion)"
  - "Regenerated docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb with committed executed output showing both new fields, including the derived '1m0'"
  - "docs/runbooks/telescope_runs_calendar.rst documents the importer's new source/telescope_class behaviour and a full repair_stale_campaign_run_sites section + cheat-sheet row"
  - "Three folded planning-doc corrections landed: PROJECT.md's stale Phase 25 claim date-pinned, 26-CONTEXT.md's D-11 owned-nights framing gets a dated forward-pointer, docs/design/canonical_record_spike.rst's false 'space telescope has no ground site' premise corrected"
affects: [28-attribution-queue, 29-reconciler]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Second D-20 call site: import_campaign_csv gates derive_telescope_class() on `site is None`, mirroring the 0011 backfill migration's `site__isnull=True` filter, so a newly imported class-wide run gets the same value an existing one got"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/import_campaign_csv.py
    - solsys_code/tests/test_import_campaign_csv.py
    - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/design/canonical_record_spike.rst
    - .planning/PROJECT.md
    - .planning/phases/26-canonical-record-spike/26-CONTEXT.md

key-decisions:
  - "source and telescope_class both go in the shared `fields` dict, before the resolved-window/TBD branch split, and neither enters `lookup` -- matches 26-DECISION Criterion 1/D-14's rule that neither field participates in the natural key"
  - "approval_status is deliberately left unchanged (still APPROVED) -- CANON-01's real importer change is writing source, not approval gating, per 26-DECISION Criterion 1's derivation rule (APPROVED + source != WEB means no approval was required, not a human approved this)"
  - "PROJECT.md's Phase 25 claim is date-pinned to 2026-07-18 rather than deleted or silently updated -- the pk=34 occurrence count in the file is unchanged (still appears on the same 2 lines it did before this plan), satisfying T-27-24's audit-trail requirement"
  - "26-CONTEXT.md's D-11 'uncovered nights' discussion text is left in place verbatim as the archival record; only a dated forward-pointer to 26-DECISION.md's Domain correction section was added above it"
  - "docs/design/canonical_record_spike.rst's table row was renamed from 'Space-mission runs' to 'Genuinely site-less runs' with a dated correction note, rather than only tweaking the prose -- the old label was itself part of the false premise"

patterns-established: []

requirements-completed: [CANON-01, CANON-02]

# Metrics
duration: ~70min
completed: 2026-07-30
---

# Phase 27 Plan 06: import_campaign_csv source/telescope_class + Paired Docs + Planning Corrections Summary

**Every CSV-imported CampaignRun now records source=csv_import and a derived telescope_class through the one shared calendar_utils.derive_telescope_class() helper; the paired demo notebook and operator runbook were updated and regenerated to match; and the phase's three folded planning-doc corrections (a stale Phase 25 claim, a pre-domain-correction framing note, and a falsified 26-DECISION premise) landed as date-pinned/forward-pointing edits rather than silent rewrites.**

## Performance

- **Duration:** ~70 min
- **Started:** 2026-07-29T~21:55 (local; continuing from Wave 4)
- **Completed:** 2026-07-30T~22:35 (local)
- **Tasks:** 3 completed
- **Files modified:** 7 (0 created, 7 modified)

## Accomplishments

- `import_campaign_csv` now writes `source=CampaignRun.Source.CSV_IMPORT` on every imported row (CANON-01's real behaviour change; `approval_status` already wrote `APPROVED` and stays unchanged) and derives `telescope_class` via `calendar_utils.derive_telescope_class(site_raw=..., telescope_instrument=...)` — D-20's second required call site — only when `resolve_site()` returned `site=None`, mirroring the `0011` backfill migration's `site__isnull=True` gate. Neither field entered the importer's natural-key `lookup` on either branch.
- 4 new tests (57 total, up from 53): the site-less fixture-shaped row (`'Generic 1m robotic telescope'`) derives `telescope_class='1m0'`; a site-resolved row with instrument text naming a class (`'SOAR 4m Goodman'`) stays blank despite the `site is None` gate not firing; every imported row across a mixed batch records `source=csv_import`; and a re-import over the same campaign leaves both new fields stable (no churn).
- Regenerated `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` via `jupyter nbconvert --to notebook --execute --inplace`, committed with real executed output: the inspection cell's header/format string now includes `source`/`telescope_class`, a new markdown cell explains what the two columns mean (CANON-01 derivation-rule framing, CANON-02 distinction), and a new callout cell prints the one fixture row (`'Generic 1m robotic telescope'`) where the derivation fires, showing `telescope_class='1m0'` in real output. No real contact PII added — only the fixture's existing synthetic placeholders are printed.
- `docs/runbooks/telescope_runs_calendar.rst` gained: a note in the CSV-bootstrap-import section documenting the new `source`/`telescope_class` writes; a full new "How do I re-resolve campaign run sites that have gone stale?" section for `repair_stale_campaign_run_sites` (dry-run-first workflow, network-call/fail-safe behaviour, safe-to-re-run, never touches the calendar); a cheat-sheet row; and a troubleshooting cross-reference from the existing `import_campaign_csv` unresolved-rows subsection.
- Three folded planning-doc corrections landed, each following its own todo's preferred option (date-pin/forward-point, never delete or silently rewrite):
  - **A.** `.planning/PROJECT.md`'s Phase 25 paragraph — the "`CampaignRun` pk=34 ... now has its 4 per-night `CalendarEvent`s" claim and the "Observatory 'FTN' has no timezone set" claim are now explicitly dated to 2026-07-18, with a note that the dev DB has since been re-imported (26-DECISION.md D-16) and that Phase 27 has since backfilled `Observatory.timezone` (D-23). The `pk=34` occurrence count in the file is unchanged (still 2 matching lines, as before this plan).
  - **B.** `.planning/phases/26-canonical-record-spike/26-CONTEXT.md`'s D-11 section gained a dated (2026-07-27) forward-pointer to `26-DECISION.md`'s `### Domain correction — queue windows are not sets of owned nights` section and its `#### Queue-run projection — settled` verdict, immediately above the original pre-correction "uncovered nights" discussion text, which is left verbatim as the archival record.
  - **C.** `docs/design/canonical_record_spike.rst`'s pipeline-stage table row was renamed from "Space-mission runs" (describing "a run with no ground site at all (a space telescope)") to "Genuinely site-less runs", with a dated (2026-07-29) correction note recording that space observatories resolve to a real `Observatory` like any ground site (JWST=274, Roman=289, WISE=C51) and that the genuine exception is a space observatory with a Horizons code but no MPC obscode at all (JUICE=`500@-28`), and that Phase 27's `telescope_class` vocabulary is the narrower `2m0`/`1m0`/`0m4`/`SPACE` set, not 26-DECISION's originally recommended three-meaning scheme.
- Closed the three resolved todos via `git mv` into `.planning/todos/completed/`: the `calendar_utils.py` private-helper rename (closed by Plan 01), the owned-nights-framing correction (closed by correction B above), and the stale-Phase-25-claim correction (closed by correction A above). `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` (the `SITE_TELESCOPE_MAP` module split) stays in `pending/` per D-24 — it is the one folded todo explicitly dropped from this phase's scope.
- A grep sweep for the falsified "permanently site-less"/"no ground site at all"/"five space-mission rows" premise across `.planning/` and `docs/` found only the already-corrected `docs/design/canonical_record_spike.rst` occurrence as "live guidance" needing a fix; every other match was in a historical findings record (`26-DECISION.md` Findings, phase `*-CONTEXT.md`/`*-RESEARCH.md`/`*-PLAN.md` files, this very plan file) and was correctly left untouched per the plan's own instruction to leave historical records alone.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write source and telescope_class from import_campaign_csv** - `aadc57e` (feat)
2. **Task 2: Regenerate the paired demo notebook and update the operator runbook** - `b92807b` (docs)
3. **Task 3: Make the three folded planning-doc corrections and close the resolved todos** - `7a5589b` (docs)

_No TDD tasks in this plan (autonomous, non-TDD execute plan)._

## Files Created/Modified

- `solsys_code/management/commands/import_campaign_csv.py` - Imports `derive_telescope_class`; `fields` dict gains `source`/`telescope_class` keys with `# D-XX:`-style inline comments; neither key added to either `lookup` dict
- `solsys_code/tests/test_import_campaign_csv.py` - Extended `test_creates_campaignrun_with_existing_observatory` with `source`/`telescope_class` assertions; added `test_site_resolved_row_telescope_class_blank_despite_instrument_text`, `test_every_imported_row_records_source_csv_import`, `test_siteless_row_derives_telescope_class_from_instrument`, `test_reimport_keeps_source_and_telescope_class_stable`
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` - New markdown explainer cell, extended inspection-cell header/format string, new derivation-callout markdown+code cell; regenerated with real executed output via `nbconvert`
- `docs/runbooks/telescope_runs_calendar.rst` - New note under the CSV-import section; new "How do I re-resolve campaign run sites that have gone stale?" section; new cheat-sheet row; new troubleshooting cross-reference
- `docs/design/canonical_record_spike.rst` - "Space-mission runs" row renamed to "Genuinely site-less runs" with a dated correction note (JUICE/`500@-28`, JWST/HST/Swift MPC codes, narrower `telescope_class` vocabulary)
- `.planning/PROJECT.md` - Phase 25 paragraph date-pinned (2026-07-18) with a D-16/D-23 forward-pointer; `pk=34` occurrence count unchanged
- `.planning/phases/26-canonical-record-spike/26-CONTEXT.md` - Dated forward-pointer added above D-11's original discussion text
- `.planning/todos/pending/` → `.planning/todos/completed/` - 3 todos moved via `git mv`: `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md`, `2026-07-27-correct-owned-nights-framing-in-upstream-planning-docs.md`, `2026-07-27-correct-project-md-stale-phase-25-calendar-event-claim.md`

## Decisions Made

- `source`/`telescope_class` both live in the shared `fields` dict before the resolved-window/TBD branch split, so both apply on either branch; neither entered `lookup` (26-DECISION Criterion 1, D-14).
- `approval_status` left byte-for-byte unchanged — the importer already wrote `APPROVED` before this phase (D-17), and CANON-01's real behaviour change is `source`, not approval gating.
- Noted, per the plan's instruction, but deliberately did not "fix": `insert_or_create_campaign_run()`'s existing silent field-merge behaviour on a matching natural key (RESEARCH.md Pattern 1 Test D) now also applies to `telescope_class` on re-import — this is pre-existing, documented, intentional idempotent-re-import design, not a new bug this plan introduced.
- PROJECT.md's stale-claim correction preserves the `pk=34` mention count exactly (2 lines, unchanged) rather than removing it, satisfying the plan's own T-27-24 audit-trail acceptance criterion and the underlying todo's "preserve the verification record" preference.
- 26-CONTEXT.md's D-11 text was left as an untouched archival record with only a forward-pointing note added above it, per the todo's stated preference over rewriting the original discussion.
- `docs/design/canonical_record_spike.rst`'s row *label* itself ("Space-mission runs") was changed, not just its body text, since the old label encoded the same falsified premise the body corrected.

## Deviations from Plan

None — plan executed exactly as written. One self-correction during execution, not a deviation from the plan's intent: the notebook's first draft repeated the multi-line `pk=34`-style `CampaignRun.objects.get(...)` call across two lines in the new derivation-callout cell, which `ruff format` would have reformatted to one line; reformatted it to match ruff's expected style before the final `nbconvert --execute --inplace` pass, so the notebook needed no additional post-hoc `ruff format` fix and `ruff format --check .`'s diff for this file is now limited to the same single pre-existing (untouched) cell it had before this plan.

## Issues Encountered

None specific to this plan's own files. `./manage.py test solsys_code.tests.test_views`/`test_ephem_utils` continue to be excluded from all test runs per the documented pre-existing SPICE/ASSIST segfault (unrelated to this plan). `ruff check .` reports exactly the one documented pre-existing `D103` error in `sync_gemini_observation_calendar_demo.ipynb` (byte-identical to the pre-phase base commit, untouched by this plan); `ruff format --check .` reports the same 7 pre-existing offenders as before this plan (verified by diffing this plan's own edited notebook against its pre-plan git-HEAD version — the only content difference is this plan's own cells, and the file's one remaining pre-existing formatting diff, cell 3's `assert (...)` wrapping, already existed before this plan touched the file).

## User Setup Required

None — no external service configuration required. This plan's only live-DB step was executing the paired demo notebook (which creates `Observatory`/`TargetList`/`CampaignRun` rows against the real dev DB, as it always has). The dev DB was backed up first to `/tmp/fomo_db.sqlite3.pre-27-06.bak` (970,752 bytes) — does not overwrite the Plan 02/03/04 backups at `/tmp/fomo_db.sqlite3.pre-27-0{2,3,4}.bak`.

## Next Phase Readiness

- Phase 27 (the-canonical-run-record) is now complete — this was the final plan (Plan 06 of 6). Both CANON-01 (`source`) and CANON-02 (`telescope_class`) are fully shipped on the ingest side (backfill migration from Plan 04, `import_campaign_csv` from this plan).
- The three folded planning-doc corrections are landed; only the `SITE_TELESCOPE_MAP` extraction todo (D-24) remains pending, deliberately deferred to a future cleanup pass with no CANON requirement behind it.
- No blockers. Full regression gate: `./manage.py test <quick-run modules>` (322/322 pass), the phase-wide `./manage.py test <all fast solsys_code modules + solsys_code_observatory>` (636/636 pass, up from the documented 632-test Wave 4 baseline — exactly the 4 new tests this plan's Task 1 added), `python -m pytest` (1/1 pass, docs build clean), `python3 -m sphinx -b html docs docs/_build/html -q` (exit 0), `ruff check .` (1 pre-existing error, unrelated), `ruff format --check .` (7 pre-existing offenders, unrelated) — all green against the documented pre-phase baseline.

---
*Phase: 27-the-canonical-run-record*
*Completed: 2026-07-30*

## Self-Check: PASSED

All 10 created/modified files confirmed present on disk; all 3 task commit hashes (`aadc57e`, `b92807b`, `7a5589b`) confirmed present in `git log --oneline --all`.
