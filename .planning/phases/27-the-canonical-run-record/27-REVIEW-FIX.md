---
phase: 27-the-canonical-run-record
fixed_at: 2026-07-30T19:48:37Z
review_path: .planning/phases/27-the-canonical-run-record/27-REVIEW.md
iteration: 1
findings_in_scope: 18
fixed: 16
skipped: 2
status: partial
---

# Phase 27: Code Review Fix Report

**Fixed at:** 2026-07-30T19:48:37Z
**Source review:** `.planning/phases/27-the-canonical-run-record/27-REVIEW.md`
**Iteration:** 1

**Summary:**

- Findings in scope: 18 (fix scope: `all`)
- Fixed: 16
- Skipped: 2 (CR-01, WR-05 — both by explicit instruction; reasoning below)

> **Count note.** 27-REVIEW.md's frontmatter records `info: 6, total: 17`, but the body
> contains seven Info findings (IN-01 through IN-07). The true total is 18. This report
> counts the findings that actually exist in the body.

All work was done in an isolated git worktree on branch `gsd-reviewfix/27-2650971`,
fast-forwarded back onto `issue37-telescope-runs-calendar`. Every fix is one commit; no
commit used `--no-verify`, so each one passed the full pre-commit gate (ruff lint, ruff
format, Sphinx build, pytest suite).

**Verification actually run** (not assumed): 610 Django tests across every affected module
pass — `test_admin`, `test_calendar_utils`, `test_calendar_template`, `test_campaign_approval`,
`test_campaign_forms`, `test_campaign_gap`, `test_campaign_models`, `test_campaign_run_observation`,
`test_campaign_site_search`, `test_campaign_submission`, `test_campaign_views`,
`test_canonical_record_migration`, `test_import_campaign_csv`, `test_load_telescope_runs`,
`test_repair_stale_campaign_run_sites`, `test_sync_lco_observation_calendar`,
`test_sync_gemini_observation_calendar`, `test_backfill_lco_observation_records`,
`test_backfill_range_calendar_events`, `test_calendar_display_extras`,
`test_window_schema_migration`, and `solsys_code_observatory.tests`.
`test_views.TestEphemeris` was deliberately excluded (segfaults in native ASSIST).

## Fixed Issues

### CR-02: stale `sync_lco_observation_calendar_demo.ipynb`

**Files modified:** `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
**Commit:** `cbf44b0`
**Applied fix:** Renamed every stale reference (`CalendarEventTelescopeLabel` →
`CalendarEventMeta`, and `_extract_instrument` / `_resolve_placement_block` /
`_derive_telescope` / `_coarse_telescope_label` → their un-privatised names), added a short
note recording the Phase 27 rename and helper un-privatisation so the Phase 8 narrative
still reads correctly, and **re-executed the notebook end to end**:
`jupyter nbconvert --to notebook --execute --inplace`, against a freshly migrated database
in the worktree. Result: 25 PASS assertions, 0 errors, 0 FAILs. The previously stale
executed output in cell 48 (which literally printed the old class name) is now regenerated
and consistent with its code.

**Note on notebook execution:** re-execution WAS feasible here and was performed. It is the
one notebook where a clean-DB run is correct, because its fixtures are self-contained
(`demo-*` observation ids created and torn down within the notebook). See IN-03 for why the
opposite call was made for the other three notebooks.

### WR-01: CSV re-import erased WEB provenance

**Files modified:** `solsys_code/management/commands/import_campaign_csv.py`,
`solsys_code/tests/test_import_campaign_csv.py`
**Commit:** `d8653f9`
**Applied fix:** Guard before `insert_or_create_campaign_run()` that pops `source` and
`approval_status` from `fields` when the matched row already has `source == WEB`. Both keys
are popped together deliberately — preserving `source` while still forcing `APPROVED` would
publish the unreviewed submission anyway. Also documented in the handler docstring and the
command's `help` text. Two regression tests added: one proving a WEB row keeps both values
(while its other fields are still overwritten), one proving a LEGACY row is still stamped
`csv_import`/`approved`, so the guard cannot silently freeze every row.

### WR-02: `telescope_class` allowlisted but rendered by no column

**Files modified:** `solsys_code/campaign_tables.py`, `solsys_code/tests/test_campaign_views.py`
**Commit:** `43fa87f`
**Applied fix:** Took the "render the column" option (the one that delivers D-18) rather than
dropping the allowlist entry. Added `telescope_class` to `CampaignRunTable.Meta.fields`
immediately after `site`, plus a declared column and a `render_telescope_class` that resolves
the raw value via `Accessor` — necessary because model-instance rows (staff) would otherwise
get django-tables2's automatic `get_telescope_class_display()` verbose label while dict rows
(non-staff) get the raw code, i.e. two reader classes seeing different text for the same run.
Label lookup derived from the model's own `TextChoices` (`TELESCOPE_CLASS_LABELS`), used via
`.get()` so an unexpected stored value degrades rather than raising. Tests now assert against
`response.content` for both reader classes, per the review. Column order verified empirically
for both `CampaignRunTable` and `ApprovalQueueTable`.

### WR-03: `CalendarEventMeta.run` has no production writer

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`, `solsys_code/campaign_views.py`,
`src/templates/tom_calendar/partials/event_form.html`
**Commit:** `e22461d`
**Applied fix:** Took the **documentation** option, not the "populate the link" option. The
design spike genuinely defers the automatic writer to the Phase 29 reconciler
(`canonical_record_spike.rst`, "Ownership rule" row), so writing the link in
`_project_calendar_event()` would pre-empt a deferred design decision — and the reconciler is
also what will keep the link correct as runs are re-approved, re-sited or cancelled. Added a
runbook section "Why doesn't the calendar pop-up show a 'Campaign run' block?" covering the
manual admin-linking procedure, the WR-08 frozen-identity behaviour of the inline, and what an
absent link means, plus pointers to it from `_project_calendar_event()`'s docstring and the
template comment block.

### WR-04: placeholder tier-1 hit counted as "resolved"

**Files modified:** `solsys_code/management/commands/repair_stale_campaign_run_sites.py`
**Commit:** `bf7095b`
**Applied fix:** Summary now keys on `site is not None and not needs_review`, and the
still-flagged log line reports the site it landed on. Comment records why `site is not None`
alone is wrong and why this line is the command's last word on such a row (the candidate
filter is `site__isnull=True`, so it is excluded from every future run).

### WR-06 + IN-06: timezone backfill migration imported live app code

**Files modified:** `solsys_code/solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py`
**Commit:** `70d38f0` (single commit — one edit closes both findings, as IN-06 notes)
**Applied fix:** Constructs `TimezoneFinder` directly instead of importing
`solsys_code_observatory.utils._get_timezone_finder()` (whose module imports the live
`Observatory` model, `requests`, and `tom_dataservices` at module scope). Also returns early
on an empty queryset, so the boundary-polygon load no longer happens on every `migrate`,
test-database creation, and `TransactionTestCase` re-migration.

### WR-07: `derive_telescope_class`'s function-local import achieved nothing

**Files modified:** `solsys_code/observer_codes.py` (new), `solsys_code/calendar_utils.py`,
`solsys_code/campaign_utils.py`, `solsys_code/management/commands/repair_stale_campaign_run_sites.py`
**Commit:** `df0f0ba`
**Applied fix:** Moved `HORIZONS_OBSERVER_TO_OBSCODE` into a new model-free module
`solsys_code/observer_codes.py` and imported it at module scope from both `campaign_utils` and
`calendar_utils`, deleting the misleading comment. `campaign_utils.HORIZONS_OBSERVER_TO_OBSCODE`
remains a valid reference for existing readers (including `test_campaign_approval`). Also
corrected the docstring's D-20 claim to acknowledge that `calendar_utils` imports
`tom_calendar.models.CalendarEvent` at module scope anyway, so "no live models" was never
literally true of the whole module.

**New file created:** `solsys_code/observer_codes.py`. Justified — the finding's own fix
suggestion calls for a model-free module, and there was nowhere existing to put it.

**Verified empirically:** importing `solsys_code.calendar_utils` no longer loads
`solsys_code.models` or `solsys_code.campaign_utils`, and `derive_telescope_class` returns
identical values for `'500@-28'` and `'500@-170'`.

### WR-08: inline exposed the primary-key `event` field as editable

**Files modified:** `solsys_code/admin.py`, `solsys_code/tests/test_admin.py`
**Commit:** `22852e7`
**Applied fix:** **Adapted, not applied verbatim.** The review's suggested
`get_readonly_fields(request, obj)` does not work here: `obj` on an inline is the *parent*
`CampaignRun`, so making `event` readonly whenever the parent exists also strips the widget
from the blank "Add another" row, making it impossible to link a new event from an existing
run's change page. Worse, `disabled=True` alone is also insufficient — I confirmed by test
that `BaseModelFormSet._construct_form()` resolves each initial form's instance from the
*submitted* pk before any field is consulted, so a tampered pk still yields a fresh instance
and an INSERT.

Added `CalendarEventMetaInlineFormSet`, which (1) normalises each initial form's submitted pk
back to that row's real pk before form construction (mirroring the data-rewriting precedent in
Django's own `BaseInlineFormSet._construct_form()` `save_as_new` branch), and (2) disables the
widget on saved rows. Regression test asserts the hijack POST writes no duplicate, the saved
row's field is disabled, and the blank add row's field is not.

### WR-09: any unrecognised `500@…` labelled `SPACE`

**Files modified:** `solsys_code/calendar_utils.py`, `solsys_code/tests/test_calendar_utils.py`
**Commit:** `4df2c18`
**Applied fix:** Branch now requires a negative NAIF ID. Four new tests: natural bodies
(`500@399`/`500@10`/`500@301`) are not SPACE; malformed IDs (`500@`, `500@oops`, `500@-`) are
not SPACE; and an unaliased *negative* ID (`500@-999`) still is — a guard against someone
narrowing the branch further and breaking JUICE's `500@-28`.

### IN-01: six-key counter dict duplicated three times

**Files modified:** `solsys_code/management/commands/sync_lco_observation_calendar.py`
**Commit:** `8f12a2b`
**Applied fix:** `_COUNTER_KEYS` tuple plus `_new_counters()`, used in all three places.
`_new_counters()` returns a fresh dict per call (documented), so facilities never share one.

### IN-02: test module importing a private helper from another test module

**Files modified:** `solsys_code/tests/helpers.py` (new), `solsys_code/tests/test_calendar_utils.py`,
`solsys_code/tests/test_sync_lco_observation_calendar.py`
**Commit:** `6200b0c`
**Applied fix:** Moved the builder to `solsys_code/tests/helpers.py` and made it public
(`observations_block_response`) now that it is genuinely shared. The module is deliberately not
named `test_*.py`, so neither the Django runner's `test*.py` pattern nor pytest collects it.

**New file created:** `solsys_code/tests/helpers.py`, exactly as the finding's fix suggests.

### IN-03: `ruff format --check .` fails on the pre-executed notebooks

**Files modified:** all four affected notebooks under `docs/notebooks/pre_executed/`
**Commit:** `400a176`
**Applied fix:** **Adapted — the suggested fix does not work.** Running
`ruff format <notebook>` is silently undone on the next commit, because
`.pre-commit-config.yaml` pins **ruff v0.2.1** while the dev environment installs **ruff
0.15.20**, and the two disagree about exactly one construct: `assert (expr).exists(), (msg)`.
0.15 collapses it to one line; 0.2.1 splits it. Whichever version formats last wins. I hit
this directly — my first CR-02 commit attempt was rejected by the hook, which reformatted the
file back. That standoff is the actual root cause of all four notebook failures.

Rewrote the five affected asserts as explicit `if not ...: raise RuntimeError(...)`
preconditions, which both versions leave untouched. Verified against **both**: `ruff format
--check` (0.15) reports all five notebooks clean, and the pinned-0.2.1 pre-commit hook passes
with no modifications. `ruff format --check .` went from 7 failing files to 3.

**Deliberately a source-only edit, without re-executing.** Every replaced statement is a
precondition that produces no output and whose condition holds, so the committed outputs
remain valid. Re-executing would have been actively *wrong* here: I tried it, and because
these notebooks were originally executed against the maintainer's populated dev DB, a
clean-DB re-run rewrote unrelated narrative output (`load_telescope_runs_demo`'s Observatory
rows flipped from `updated/unchanged` to `created`). I reverted that and redid the edit
source-only.

### IN-04: docstring cited a symbol that never existed

**Files modified:** `solsys_code/campaign_utils.py`
**Commit:** `de9dd4f`
**Applied fix:** Pointer corrected to `derive_telescope`, with a parenthetical recording why
the old text was wrong (so nobody "restores" it).

### IN-05: demo notebook mixed `legacy` rows into a `csv_import` narrative

**Files modified:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`
**Commit:** `962c6eb`
**Applied fix:** Took the review's **prose** option rather than filtering the queryset, because
the prose cell is markdown with no output and therefore needs no re-execution, whereas changing
the code cell would have forced the clean-DB re-run problem described under IN-03. The note
names the two `Demo Telescope/*` rows as the notebook's own approval-lifecycle demo rows and
explains that `legacy` is the model default for a row created without an explicit `source`,
never something the importer writes.

### IN-07: runbook documented the `target` reset but not `source`/`approval_status`

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `f2e5938`
**Applied fix:** Extended the "re-import deliberately, not routinely" warning and the
"what the command now writes" note to name all three fields and to record the `source = web`
carve-out introduced by WR-01. The closing cross-reference in the troubleshooting section was
updated to match.

## Skipped Issues

### CR-01: `telescope_class` is never cleared when a run's site later resolves

**File:** `solsys_code/campaign_views.py:547-571`, `solsys_code/campaign_views.py:660-707`,
`solsys_code/management/commands/repair_stale_campaign_run_sites.py:189-199`,
`solsys_code/models.py:201-210`
**Reason:** skipped by explicit instruction — design-level invariant question, the user's call.

**Reasoning, for the decision that needs making:** the finding's *diagnosis* is sound and I
confirmed it in the code — all three site-resolution paths write `site`/`site_needs_review`
without touching `telescope_class`, so the resolve-after-derive sequence really does persist a
row with both a real `Observatory` and a class-wide allocation. What is genuinely undecided is
whether clearing is the right response:

- `models.py:201-203` says *"telescope_class is never inferred for a run whose site DID
  resolve"*. That is a statement about **inference at write time**, which both writers already
  honour. It does not by itself say a class-wide allocation that was correctly derived must be
  **destroyed** when a site is later resolved.
- There is a real case where clearing loses information: a run genuinely allocated to a
  telescope *class* whose site is later pinned down. Under CANON-02 the two facts are being
  treated as mutually exclusive, but that premise is exactly what D-11 already had to correct
  once (it falsified the spike's "space missions are permanently site-less" premise).
- The alternative reading — that `telescope_class` should survive and the mutual exclusivity
  should be relaxed — would require changing the model docstring and possibly the admin
  `list_filter` semantics instead of the three write paths.

Applying the suggested three-site edit would silently pick one of those readings and bake it
into a data-destroying write. That is a decision, not a fix. Whichever way it goes, the
regression test the review asks for (resolve-after-derive) should land with it.

### WR-05: hardcoded, database-specific `site_raw` correction

**File:** `solsys_code/management/commands/repair_stale_campaign_run_sites.py:52-56, 158-163, 196-199`
**Reason:** skipped by explicit instruction — standing guidance was to skip unless the fix is
purely to gate/parameterize rather than guess at which rows are correct.

**Reasoning:** the review's suggested fix (an `--apply-owner-site-raw-corrections` opt-in flag)
*is* purely a gating change and would be safe to apply mechanically. I did not apply it because
gating it changes the operational contract of a command whose documented purpose includes
applying that specific correction on the dev DB: after the flag lands, the existing recorded
one-time run procedure (`python manage.py repair_stale_campaign_run_sites`) silently stops
correcting the Swift row, and D-16b's phase-summary outcome no longer reproduces. Whether the
flag should default off (safe elsewhere, changes the documented dev-DB procedure) or the
correction should be dropped from this general-purpose command entirely and applied as a
one-off data edit is an owner call about intent, not a mechanical fix.

Note that the underlying concern is narrower than it first appears: `_OWNER_SUPPLIED_SITE_RAW`
only fires when a row's own `site_raw` is **blank** and the first `telescope_instrument` token
is exactly `swift`, and a non-blank `site_raw` is never overwritten. The exposure is a blank-site
Swift row on another database, not every Swift row.

## Notes and Deviations

- **Two findings share one commit.** WR-06 and IN-06 are the same edit to the same migration
  (IN-06's own text says "WR-05's snippet fixes both" — it means WR-06's snippet). Committed as
  `70d38f0` covering both rather than manufacturing an artificial split.

- **Three fixes were adapted rather than applied verbatim** (WR-08, IN-03, WR-03). In each case
  the review's suggested snippet would not have achieved its stated goal; the reasoning is
  recorded under each entry above and in the code comments.

- **Pre-existing issues left alone, as instructed / out of scope:**
  - `src/fomo/settings.py` — untouched; it carries an unrelated unstaged modification in the
    main working tree.
  - `ruff check .` still reports one D103 (`sync_gemini_observation_calendar_demo.ipynb`
    cell 6). The review itself records this as pre-existing and unrelated; it is a lint finding,
    not a format one, and is outside IN-03's scope.
  - `ruff format --check .` still reports 3 files: `src/fomo/settings.py` and two historical
    scratch scripts under `.planning/quick/260619-f7u-.../`. Down from 7.

- **Worktree hygiene:** all work was done in `/tmp/sv-27-reviewfix-23CKHM` on branch
  `gsd-reviewfix/27-2650971`, fast-forwarded onto `issue37-telescope-runs-calendar`; the
  worktree, temp branch, and recovery sentinel were cleaned up. A generated `src/fomo/_version.py`
  and a scratch `src/fomo_db.sqlite3` were created inside the worktree to run tests and execute
  the notebook — both are gitignored and neither was committed.

---

_Fixed: 2026-07-30T19:48:37Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
