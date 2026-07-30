---
phase: 27-the-canonical-run-record
reviewed: 2026-07-30T13:11:49Z
depth: deep
files_reviewed: 27
files_reviewed_list:
  - docs/design/canonical_record_spike.rst
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/admin.py
  - solsys_code/calendar_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/backfill_lco_observation_records.py
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/management/commands/repair_stale_campaign_run_sites.py
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/migrations/0008_rename_calendareventtelescopelabel_calendareventmeta.py
  - solsys_code/migrations/0009_calendareventmeta_run.py
  - solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py
  - solsys_code/migrations/0011_backfill_campaignrun_telescope_class.py
  - solsys_code/models.py
  - solsys_code/solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py
  - solsys_code/solsys_code_observatory/tests/test_timezone_backfill_migration.py
  - solsys_code/tests/test_admin.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_campaign_run_observation.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_canonical_record_migration.py
  - solsys_code/tests/test_import_campaign_csv.py
  - solsys_code/tests/test_load_telescope_runs.py
  - solsys_code/tests/test_repair_stale_campaign_run_sites.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
  - src/templates/tom_calendar/partials/event_form.html
findings:
  critical: 0
  warning: 4
  info: 2
  total: 6
status: issues_found
---

# Phase 27: Code Review Report

**Reviewed:** 2026-07-30T13:11:49Z
**Depth:** deep
**Files Reviewed:** 27 (28 listed, `sync_lco_observation_calendar.py` reviewed as diff+full for cross-reference)
**Status:** issues_found

## Summary

Phase 27 (`CalendarEventTelescopeLabel` -> `CalendarEventMeta` rename + `run` link,
`CampaignRun.source`/`telescope_class`, `CampaignRunObservation`, the
`repair_stale_campaign_run_sites` one-off command, and the Observatory timezone backfill)
is a well-executed, well-tested change. All 275 Django app tests pass
(`./manage.py test solsys_code ...` across every listed test module), `ruff check .` and
`ruff format --check .` are clean for every file in scope (migrations are correctly
excluded per `pyproject.toml`'s `[tool.ruff] exclude`), the hand-authored migrations
correctly use `RenameModel` (not `DeleteModel`/`CreateModel`) to preserve the 11 real
`CalendarEventMeta` rows, and the demo notebook was regenerated with real executed output
per CLAUDE.md's paired-docs rule. No hardcoded secrets, eval/exec, injection, or empty
`except` patterns were found anywhere in the reviewed files.

Deep cross-file tracing surfaced a real, demonstrated **data-reversion risk** between two
of this phase's own commands (`import_campaign_csv` and `repair_stale_campaign_run_sites`)
that isn't mentioned anywhere in the runbook's existing "re-import gotcha" note, plus a
latent invariant gap in `repair_stale_campaign_run_sites` around the new `telescope_class`
field it was never updated to account for. A separate finding flags that `CampaignRun.source`
— unlike `approval_status` — is left freely editable in the admin even though CANON-01's
approval-derivation rule depends on it being an honest, unaltered record of ingest
provenance. None of these rise to data loss or a crash; all are classified WARNING.

## Warnings

### WR-01: CSV re-import can silently revert a site just fixed by `repair_stale_campaign_run_sites`, undocumented

**File:** `solsys_code/management/commands/import_campaign_csv.py:177-207`
**Issue:** `import_campaign_csv` recomputes `site`/`site_raw`/`site_needs_review` (and, new
in this phase, `telescope_class`) from the CSV's own `Site Code` cell on **every** run,
including a re-import over an already-existing row (`site, needs_review =
resolve_site(site_raw)` at line 178, unconditionally included in the `fields` dict passed
to `insert_or_create_campaign_run`, which overwrites any differing field on update). This
phase adds a companion command, `repair_stale_campaign_run_sites`, whose entire purpose is
to re-resolve stale, site-less approved rows through a fixed `resolve_site()` path — and,
per its own D-16b (`_OWNER_SUPPLIED_SITE_RAW = {'swift': 'C52'}`), for the real dev-DB
Swift row (pk 13) it sets `site_raw` to a value (`'C52'`) that **does not come from the
CSV** (the CSV's own `Site Code` cell for that row is blank, confirmed by the Phase 27-02
live-repair before/after table in `27-02-SUMMARY.md`). If `import_campaign_csv` is
re-run over the same campaign after `repair_stale_campaign_run_sites` has fixed such a row,
`resolve_site('')` (the CSV's still-blank `Site Code`) returns `(None, True)` immediately,
silently reverting `site` back to `None` and `site_needs_review` back to `True` — undoing
the repair with no warning. The runbook's existing "Re-import gotcha" note
(`docs/runbooks/telescope_runs_calendar.rst:178-186`) documents only the `target`-field
reset; it says nothing about `site`/`site_raw`/`site_needs_review`/`telescope_class` also
being silently reset on every re-import, even though this phase is precisely what created
the operational scenario (repair a stale row, then later re-import the same sheet for new
rows) where that reset actively destroys real repair work.
**Fix:** Either (a) have `import_campaign_csv` skip re-deriving `site`/`site_raw` for a row
whose `site` is already resolved and not a placeholder (mirroring the "never re-resolve an
already-resolved site" guard `CampaignRunDecisionView` already uses elsewhere in this
codebase), or (b) at minimum, expand the runbook's "Re-import gotcha" note and the
command's own `help` text to explicitly call out that `site`, `site_raw`,
`site_needs_review`, and `telescope_class` are also silently reset to their CSV-derived
values on every re-import, and that `repair_stale_campaign_run_sites` fixes (in particular,
the D-16b Swift `site_raw` correction) can be undone by a subsequent CSV re-import over the
same campaign.

### WR-02: `repair_stale_campaign_run_sites` never clears `telescope_class`, so a resolved run can end up with both a real site and a stale `telescope_class` value

**File:** `solsys_code/management/commands/repair_stale_campaign_run_sites.py:193-199`
**Issue:** `run.telescope_class` is never referenced anywhere in this command. When it
successfully resolves a previously site-less run's site (`update_fields = ['site',
'site_needs_review']`, conditionally `+ ['site_raw']`), any `telescope_class` value the
row already carries (e.g. written earlier by migration `0011`'s backfill, or by
`import_campaign_csv`'s own `derive_telescope_class()` call) is left in place. This
violates the invariant `models.py` itself documents for `CampaignRun.telescope_class`:
"Blank is the normal value for a site-resolved run ... `telescope_class` is never inferred
for a run whose site DID resolve" (`solsys_code/models.py:201-203`), and no other code path
reconciles it afterward — `CampaignRunTableView` and the admin both render `site` and
`telescope_class` as independent columns, so a viewer would see a run simultaneously
claiming a real, specific site and a class-wide allocation. This exact interaction is
untested: `test_repair_stale_campaign_run_sites.py`'s `_make_run()` fixture never sets
`telescope_class`, so every seeded row starts blank and the test suite never exercises "a
row with a pre-existing `telescope_class` gets its site resolved." (In the one real
dev-DB run recorded in `27-02-SUMMARY.md`, this didn't manifest only because the repair
ran in Plan 27-02, before the `telescope_class` field existed at all in Plan 27-04 — the
ordering that avoided the bug this one time is not enforced or documented anywhere for a
future re-run.)
**Fix:** When this command successfully resolves `site` for a row, also clear
`telescope_class` back to `''` in the same `update_fields` write (and add a regression test
seeding a row with a non-blank `telescope_class` to prove it gets cleared on resolution).
Consider also adding a `CheckConstraint` or `clean()` validation on `CampaignRun` enforcing
that `telescope_class` is blank whenever `site` is set, so a future write path can't
silently reintroduce the same violation.

### WR-03: `CampaignRun.source` is freely editable in the admin, undermining the CANON-01 approval-derivation rule it backs

**File:** `solsys_code/admin.py:62`
**Issue:** `CampaignRunAdmin.readonly_fields = ['approval_status']` protects
`approval_status` from being hand-edited (its own comment explains why: its transition
triggers the calendar-projection side effect and the D-06 clobber guard, both of which
live in `CampaignRunDecisionView.post()`, not on the model). `source` gets no equivalent
protection, even though `models.py`'s own `Source` docstring defines a derivation rule that
depends on `source` being an honest record of provenance: "`approval_status == APPROVED`
together with `source != WEB` means no approval was required — a different fact from a
human having approved the run" (`solsys_code/models.py:90-94`). Any staff user with
`CampaignRun` change permission in the admin can freely edit `source` on an already-decided
row via the ordinary change form — e.g. relabeling an already-`APPROVED` `WEB` submission
(a run a human staff member genuinely reviewed and approved through the approval queue) to
`csv_import` or `legacy` silently erases the "a human approved this" signal the derivation
rule exists to preserve, with no admin log entry calling out the semantic significance of
that specific field. (The comment at `admin.py:60-61` — "`source` is deliberately NOT added
here (D-19) ... so the omission is a decision" — documents that the field is intentionally
left editable, but doesn't address this consequence.)
**Fix:** Either make `source` read-only after creation (e.g. `readonly_fields =
['approval_status', 'source']` for an existing instance, editable only via
`get_readonly_fields()`'s `obj is None` branch so it can still be set on manual creation),
or, if free-text correction is genuinely needed for data-entry fixes, at least surface a
`django.contrib.admin` `LogEntry`-visible change reason / confirmation step so a `source`
edit on an already-`APPROVED` row is auditable.

### WR-04: `event_form.html`'s Campaign-run block can render "(None–None)" for a linked TBD run

**File:** `src/templates/tom_calendar/partials/event_form.html:100-125`
**Issue:** The template unconditionally renders `({{ run.window_start }}&ndash;{{
run.window_end }})` once `event.telescope_label_meta.run.is_publicly_visible` is true
(lines 111-125). `CampaignRunAdmin`'s new `CalendarEventMetaInline`
(`solsys_code/admin.py:15-18`, `fk_name='run'`) lets a staff user attribute **any**
`CalendarEvent` to **any** `CampaignRun` from the admin's `event` dropdown, with no
validation tying the choice to whether the run actually has a resolved window — including a
TBD run, whose `window_start`/`window_end` are both `NULL` by the model's own
`CheckConstraint` (`campaign_run_window_start_end_null_together`,
`solsys_code/models.py:259-265`). If a staff member attributes an event to a TBD, publicly
visible (approved/rejected) run through this new inline, the public-facing event modal
renders the literal text `(None–None)` instead of anything readable. This exact scenario is
not covered by `EventModalCampaignRunLinkTest` in `test_calendar_template.py`, whose
`approved_run` fixture always has a concrete `window_start`/`window_end`.
**Fix:** Guard the window-range display, e.g. `{% if run.window_start %}({{
run.window_start }}&ndash;{{ run.window_end }}){% else %}(date TBD){% endif %}`, and add a
regression test seeding a TBD, approved `CampaignRun` linked via `CalendarEventMeta` to
prove the modal no longer prints `None`.

## Info

### IN-01: Placeholder-resolved sites are silently excluded from `telescope_class` derivation, same as a genuinely-resolved site

**File:** `solsys_code/calendar_utils.py:146-208`, `solsys_code/management/commands/import_campaign_csv.py:205-207`, `solsys_code/migrations/0011_backfill_campaignrun_telescope_class.py:45`
**Issue:** All three call sites gate `telescope_class` derivation on `site is None`
(`import_campaign_csv`'s ternary at line 205-207, migration 0011's `site__isnull=True`
filter at line 45). A tier-3 **placeholder** `Observatory` (created when `resolve_site(...,
create_placeholder=True)` — the CSV importer's default — can't resolve a code at all) still
sets `run.site` to a non-`None` (placeholder) row, so a placeholder-sited run is treated
identically to a genuinely-resolved one and never gets a `telescope_class`, even though
`is_placeholder_observatory()` elsewhere in this codebase (`campaign_views.py:547`,
`:660`) treats a placeholder as "not a genuine resolution" and re-enters resolution for it.
This is consistent across all three call sites (not a contradiction between them) and may
be a deliberate simplification, but it means a class-wide/space run whose site happened to
land on a placeholder never gets the `telescope_class` signal the model's own docstring
promises for "a genuinely site-less run." Worth a deliberate decision note if not already
covered by a later phase.

### IN-02: `save_formset`'s isinstance-gated stamping has no atomic wrapper

**File:** `solsys_code/admin.py:76-96`
**Issue:** `CampaignRunAdmin.save_formset()` iterates `formset.save(commit=False)` and
calls `instance.save()` per instance, then deletes `formset.deleted_objects`, all outside
any explicit `transaction.atomic()` block. Django wraps the overall admin `changeform_view`
POST in `transaction.atomic()` by default, so this is very likely safe in practice, but
it's worth confirming (or making explicit with a local `transaction.atomic()` block) given
the two inlines involved (`CalendarEventMetaInline`, `CampaignRunObservationInline`) can
both be submitted in the same POST — a partial failure between the two formsets could
otherwise leave one saved and the other not, on a backend where the outer atomic wrapper
isn't in effect (e.g. a future async admin path).

---

_Reviewed: 2026-07-30T13:11:49Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
