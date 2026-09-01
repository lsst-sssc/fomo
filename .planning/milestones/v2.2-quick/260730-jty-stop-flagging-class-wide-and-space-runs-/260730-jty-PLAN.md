---
phase: quick-260730-jty
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/repair_stale_campaign_run_sites.py
  - solsys_code/migrations/0012_unflag_class_wide_campaignrun_site_review.py
  - solsys_code/models.py
  - solsys_code/calendar_utils.py
  - solsys_code/tests/test_import_campaign_csv.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_canonical_record_migration.py
  - solsys_code/tests/test_repair_stale_campaign_run_sites.py
  - docs/notebooks/pre_executed/fixtures/campaign_sample.csv
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - .planning/phases/27-the-canonical-run-record/27-REVIEW-FIX.md
autonomous: true
requirements: [FLAG-01, FLAG-02, FLAG-03, DOC-01]

must_haves:
  truths:
    - "No writer flags site_needs_review=True on a CampaignRun that carries a telescope_class ('1m0'/'2m0'/'0m4'/'SPACE')"
    - "The four live class-carrying rows (pk 26 JUICE, 29 LCO 1m, 30 LCO 2m, 37 Generic 1m robotic telescope) no longer appear in the staff 'Sites Needing Review' queue"
    - "telescope_class survives site resolution unchanged on every path (approve, resolve, repair, import) — no code clears it"
    - "A genuinely unresolvable row with no class signal is still flagged and still surfaces in Sites Needing Review"
    - "import_campaign_csv's site_needs_review counter reports only rows that genuinely need staff action"
    - "models.py and calendar_utils.py state D-06's framing (telescope_class records WHY there is no site) instead of mutual exclusivity, and say explicitly that it is never cleared"
    - "The paired demo notebook and the operator runbook document the new flagging rule, the notebook with real executed output"
    - "27-REVIEW-FIX.md records CR-01 as REJECTED with its reasoning, not merely skipped"
  artifacts:
    - path: "solsys_code/migrations/0012_unflag_class_wide_campaignrun_site_review.py"
      provides: "One-way data migration unflagging every class-carrying row"
      contains: "apps.get_model"
    - path: "solsys_code/management/commands/import_campaign_csv.py"
      provides: "telescope_class derived before the review flag is computed"
    - path: "solsys_code/campaign_views.py"
      provides: "Approve-path review flag honours telescope_class"
    - path: "solsys_code/management/commands/repair_stale_campaign_run_sites.py"
      provides: "Class-carrying rows skipped with their own counter"
    - path: "docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb"
      provides: "Executed evidence of both branches (classed row unflagged, unresolvable row still flagged)"
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      provides: "Operator-facing statement of the corrected flagging rule"
  key_links:
    - from: "solsys_code/management/commands/import_campaign_csv.py"
      to: "solsys_code/calendar_utils.py derive_telescope_class"
      via: "derivation runs before the review flag is computed"
      pattern: "derive_telescope_class"
    - from: "solsys_code/migrations/0012_unflag_class_wide_campaignrun_site_review.py"
      to: "CampaignRun.site_needs_review"
      via: "historical model fetched with apps.get_model, filtered by non-blank telescope_class"
      pattern: "telescope_class"
    - from: "solsys_code/campaign_views.py review_qs"
      to: "approval_queue.html 'Sites Needing Review' card"
      via: "filter(approval_status=APPROVED, site_needs_review=True)"
      pattern: "site_needs_review=True"
---

<objective>
Stop flagging class-wide and space CampaignRuns as needing site review.

`CampaignRun.telescope_class` is a PERMANENT, CORRECT campaign-level fact, not a placeholder
to clear once a site is known. Per D-06 (`.planning/phases/26-canonical-record-spike/26-CONTEXT.md:94`)
the field is a "why is there no site" vocabulary, so a non-blank value is an ANSWER to that
question — not a resolution failure. Today all four classed rows sit in the staff "Sites
Needing Review" queue (`campaign_views.py:362`) where there is nothing to resolve; worse, a
staff member who "resolves" one would assert that a multi-site campaign (e.g. "LOOK Project
Comet Followup 2026B", following up many targets across the LCO 1m0 network) lives at a
single site. Per-site detail belongs on the linked ObservationRecords via
`CampaignRunObservation` (CANON-04), not on the run.

After this task, `site_needs_review` means only: the site did not resolve AND no
telescope_class explains why. This is consistent with D-13 and with
`derive_telescope_class`'s own docstring, which already says `''` is "the correct value for
a genuine site-resolution failure, since site_needs_review already carries 'unresolved'".

This also closes out phase-27 code-review finding CR-01. The user REJECTED CR-01: its
premise (telescope_class and a resolved site are mutually exclusive, so the field must be
cleared on resolution) is invalid. **`telescope_class` must NEVER be cleared when a site
resolves.** Do not apply CR-01's suggested three-site edit; record the rejection instead.

Purpose: remove noise and a data-corrupting trap from the staff work queue, and fix the
misleading docstrings that caused the whole detour.
Output: corrected writers, a data migration for the live rows, regression tests, refreshed
paired docs, and a recorded rejection.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md
@solsys_code/models.py
@solsys_code/calendar_utils.py
@solsys_code/management/commands/import_campaign_csv.py
@solsys_code/campaign_views.py
@solsys_code/management/commands/repair_stale_campaign_run_sites.py
@solsys_code/migrations/0011_backfill_campaignrun_telescope_class.py
@solsys_code/solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py
@solsys_code/tests/test_canonical_record_migration.py
@.planning/phases/27-the-canonical-run-record/27-REVIEW-FIX.md
</context>

<interfaces>
Facts confirmed by reading the code — do not re-derive them:

- `resolve_site(site_code_raw, *, create_placeholder=True) -> tuple[Observatory | None, bool]`
  (`solsys_code/campaign_utils.py:141`). It can return `(placeholder_observatory, True)`:
  a tier-1 hit on an existing `NEEDS REVIEW: ` placeholder, or a tier-3 fabrication when
  `create_placeholder` is True. `import_campaign_csv` calls it with the default (True);
  `campaign_views` and `repair_stale_campaign_run_sites` pass False.
- `derive_telescope_class(site_raw, telescope_instrument) -> str` (`calendar_utils.py:148`),
  primitives only, never raises, returns `'2m0'/'1m0'/'0m4'/'SPACE'/''`.
- Both existing derivation call sites gate on "no resolved site": migration 0011 filters
  `site__isnull=True`; the importer writes `''` when `site is not None`. That write-time
  inference gate is CORRECT and stays — the corrected premise is only about never CLEARING
  an already-derived value.
- `CampaignRunDecisionView._resolve_site()` (`campaign_views.py:635`) already refuses any run
  with `site_needs_review` False (guard at line 658) and its conditional claim filters
  `site_needs_review=True` (line 709), so classed rows become ineligible automatically once
  the flag is off. It must NOT gain a `telescope_class=''` write.
- Approval-queue and Sites-Needing-Review tests live in
  `solsys_code/tests/test_campaign_approval.py` (class `TestSitesNeedingReview` at line 915,
  `TestApprovalSiteResolution` at line 707), NOT in `test_campaign_views.py`.
- Migration tests use `TransactionTestCase` + `django.db.migrations.executor.MigrationExecutor`
  with `migrate_from`/`migrate_to` lists and a `tearDown` that migrates back to leaf nodes —
  see `TestSourceAndTelescopeClassBackfill` (`test_canonical_record_migration.py:107`).
- Latest migration is `0011_backfill_campaignrun_telescope_class`; the new one is `0012`.
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Make site_needs_review mean "no site AND no telescope_class" at every writer, and migrate the live rows</name>
  <files>solsys_code/management/commands/import_campaign_csv.py, solsys_code/campaign_views.py, solsys_code/management/commands/repair_stale_campaign_run_sites.py, solsys_code/migrations/0012_unflag_class_wide_campaignrun_site_review.py, solsys_code/models.py, solsys_code/calendar_utils.py</files>
  <action>
Five edits plus one new migration. Cite D-06 (26-CONTEXT.md:94) in each new comment so the
reasoning survives.

(a) `import_campaign_csv.py` (~lines 190-220). Today `site, needs_review = resolve_site(site_raw)`
runs at line 191, before telescope_class is derived inline inside the `fields` dict at line 218.
Reorder: keep the `resolve_site()` call but bind its second return value to a distinct name
(e.g. `site_resolution_failed`), then compute `telescope_class` on its own line immediately
after — keeping the existing `if site is None else ''` gate — and only then compute
`needs_review`. Compute the flag as `site_resolution_failed and not telescope_class`, NOT as
the literal `site is None and not telescope_class`: `resolve_site()` here runs with
`create_placeholder=True`, so it can return a placeholder Observatory with the flag True and
`site` not None, and that row genuinely still needs review. Because the derivation is gated on
`site is None`, `telescope_class` is only ever non-blank when the site is absent, so the two
forms agree wherever the class can fire while this one preserves the placeholder case. Use the
computed `telescope_class` and `needs_review` names in the `fields` dict instead of the inline
expressions. Keep `site_needs_review_count` incrementing on the FINAL `needs_review` so the
summary line counts only rows that genuinely need staff action. Replace the now-wrong comment
at lines 215-217 ("a site-resolved run never carries a telescope_class") with the D-06
framing: the class records why there is no site, is permanent, and a row that has one is not
a resolution failure.

(b) `campaign_views.py` approve branch (~lines 582-584). Keep the `resolve_site(...,
create_placeholder=False)` call; write the flag as `needs_review and not run.telescope_class`.
Leave `update_fields=['site', 'site_needs_review']` exactly as it is — deliberately never add
`telescope_class`, because it is never cleared (this is the rejected CR-01 edit; say so in the
comment so nobody re-applies it). Then add a short comment at the `_resolve_site()` eligibility
guard (~line 658) recording that a classed run can no longer carry the flag, so it is already
ineligible here, and that the conditional claim at ~line 709 must NOT gain a `telescope_class=''`
write. Make no other behavioural change in this file; the public submission path (~line 259)
still sets neither field and needs no edit.

(c) `repair_stale_campaign_run_sites.py`. A class-carrying row is permanently site-less by
design, so there is no site to repair: inside the candidate loop (before the `site_raw`
handling at ~line 156), skip any run whose `telescope_class` is non-blank, mirroring the
existing `skipped_no_site_code` pattern — its own counter (e.g. `skipped_class_wide`), a
`logger.info`, a `self.stdout.write` line naming the pk and the class, and `continue`. Add the
counter to BOTH final summary lines (dry-run and real). Update the module docstring and the
command's `help` text to state the skip.

(d) New migration `solsys_code/migrations/0012_unflag_class_wide_campaignrun_site_review.py`,
hand-authored in the style of 0011 and of
`solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py`: module docstring
explaining D-06 and naming the four live rows as the measured effect (pk 26 JUICE/SPACE,
29 LCO 1m, 30 LCO 2m, 37 Generic 1m robotic telescope — all with blank `site_raw`), one-way
(`reverse_code=migrations.RunPython.noop`), dependency on
`0011_backfill_campaignrun_telescope_class`. Fetch the model with
`apps.get_model('solsys_code', 'CampaignRun')` and import NO live application code at all
(WR-06) — this migration needs no helper, so the derived rule is a plain queryset: rows with
`site_needs_review=True` and a non-blank `telescope_class`. Log the affected pks and class
values at INFO before writing, then clear the flag with a single queryset
`.update(site_needs_review=False)`. Derived rule only — never a hand-enumerated pk list.

(e) Docstring corrections, the misleading wording that caused this detour.
`models.py:201-203`: the comment above `telescope_class` currently reads as an invariant that
a site-resolved run never carries a class. Reframe per D-06 — the field records WHY there is
no site; a class-wide campaign (many targets across the LCO 1m0 network) legitimately keeps
`site=None` permanently, and its per-site detail lives on the linked ObservationRecords via
`CampaignRunObservation` (CANON-04); the run-level `site` field is for single-site runs only.
State explicitly that the value is NEVER cleared when a site resolves, and that inference
still only happens at write time for site-less rows. Also add a comment above the
`site_needs_review` field (~line 157) giving its corrected meaning: the site did not resolve
AND no telescope_class explains why. Do NOT change that field's `verbose_name` — a
`verbose_name` change would force an extra AlterField migration for no behavioural gain.
`calendar_utils.py:148` `derive_telescope_class`: rewrite the paragraph ending "A site-resolved
run must never carry a telescope_class; that contract lives with the callers, not here"
(~lines 155-160) the same way — callers still gate derivation on "no resolved site", but
nothing clears an already-derived class, and a non-blank return means the caller must not
flag the row for site review. Leave the Returns block's D-13 sentence intact; it is already
correct.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; python manage.py makemigrations --check --dry-run &amp;&amp; python manage.py migrate &amp;&amp; python manage.py shell -c "from solsys_code.models import CampaignRun; qs = CampaignRun.objects.filter(site_needs_review=True).exclude(telescope_class=''); print('still-flagged classed rows:', list(qs.values_list('pk', 'telescope_class'))); raise SystemExit(1 if qs.exists() else 0)" &amp;&amp; ruff check solsys_code/ &amp;&amp; ruff format --check solsys_code/</automated>
  </verify>
  <done>`makemigrations --check` reports no model changes (only the hand-authored data migration was added); `migrate` applies 0012; no CampaignRun in the dev DB has both a telescope_class and site_needs_review=True; ruff lint and format are clean for `solsys_code/`.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Regression tests for the corrected rule at all four writers</name>
  <files>solsys_code/tests/test_import_campaign_csv.py, solsys_code/tests/test_campaign_approval.py, solsys_code/tests/test_canonical_record_migration.py, solsys_code/tests/test_repair_stale_campaign_run_sites.py</files>
  <behavior>
    - Importer: a CSV row with a blank/unresolvable Site Code whose Telescope / Instrument names a class (e.g. "Generic 1m robotic telescope") imports with telescope_class='1m0', site=None, and site_needs_review=False; the printed summary counts it as 0 for site_needs_review.
    - Importer: a CSV row with a blank Site Code whose instrument text names no class imports with telescope_class='' and site_needs_review=True — the genuine-failure branch still works (extend or sit alongside test_unresolvable_site_flags_needs_review_without_skipping_row at line 777 rather than duplicating it).
    - Importer: re-importing a class-carrying row leaves telescope_class and site_needs_review stable (guard against a future regression that clears the class).
    - Approve path: approving a PENDING_REVIEW run that already carries a telescope_class and whose site does not resolve leaves site_needs_review False and telescope_class unchanged; the run does not appear in the approval queue's review_table rows.
    - Approve path (control): approving a run with NO telescope_class and an unresolvable site still sets site_needs_review=True and still appears in review_table.
    - Resolve-after-derive (the sequence CR-01 flagged, now asserted with the opposite expectation): when a site IS resolved for a run that carries a telescope_class, the class is NOT cleared. Drive this through a real code path and name CR-01's rejection in the test docstring so nobody "fixes" it back.
    - Repair command: a candidate row carrying a telescope_class is skipped entirely — untouched site, untouched flag, untouched class — and reported under its own counter in the summary output.
    - Migration 0012: seed rows against the pre-0012 historical model with site_needs_review=True in three shapes (one aperture class, one SPACE, one blank class), migrate to 0012, then assert the two classed rows are unflagged and the blank-class row is still flagged.
  </behavior>
  <action>
Write the tests described in `<behavior>`, following each module's existing conventions.

For the migration test, add a new `TransactionTestCase` class to
`solsys_code/tests/test_canonical_record_migration.py` modelled on
`TestSourceAndTelescopeClassBackfill` (line 107): `migrate_from` is
`[('solsys_code', '0011_backfill_campaignrun_telescope_class')]`, `migrate_to` is
`[('solsys_code', '0012_unflag_class_wide_campaignrun_site_review')]`, the same
`MigrationExecutor` setUp/tearDown pair, and `apps.get_model` for every model. Give each
seeded row a distinct observing window so no two collide on
`unique_campaign_run_resolved_window`. Set `telescope_class` explicitly on the seeded rows
rather than relying on 0011 to derive it — the migration under test is 0012.

For Target fixtures anywhere in these tests use `NonSiderealTargetFactory`, per CLAUDE.md.

Existing tests that may now assert the old behaviour must be revised deliberately, not
deleted: check `test_repair_stale_campaign_run_sites.py::test_juice_stays_site_less_no_site_code`
(line 128, asserts `site_needs_review` stays True) — if its fixture carries no
`telescope_class` it is unaffected and must stay green as the genuine-failure control; only
if it does carry one should it move to the new skip expectation. Same discipline for
`test_import_campaign_csv.py::test_siteless_row_derives_telescope_class_from_instrument`
(line 574) and `test_reimport_keeps_source_and_telescope_class_stable` (line 604).

Avoid the `assert (expr).exists(), (msg)` construct anywhere in these files: pre-commit pins
ruff 0.2.1 while the venv has 0.15.20 and the two format it differently, producing a commit
standoff (27-REVIEW-FIX.md IN-03). Use `self.assertTrue(...)` / `self.assertFalse(...)`.

Run ONLY the named test modules — never the whole `solsys_code` suite, because
`test_views.TestEphemeris` segfaults in native ASSIST.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; python manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_campaign_approval solsys_code.tests.test_canonical_record_migration solsys_code.tests.test_repair_stale_campaign_run_sites solsys_code.tests.test_campaign_views solsys_code.tests.test_calendar_utils solsys_code.tests.test_campaign_models &amp;&amp; ruff check solsys_code/ &amp;&amp; ruff format --check solsys_code/</automated>
  </verify>
  <done>All seven named test modules pass, including new tests at all four writers and the 0012 migration test; no pre-existing test was deleted to make the suite green; ruff clean.</done>
</task>

<task type="auto">
  <name>Task 3: Refresh the paired docs and record the CR-01 rejection</name>
  <files>docs/notebooks/pre_executed/fixtures/campaign_sample.csv, docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst, .planning/phases/27-the-canonical-run-record/27-REVIEW-FIX.md</files>
  <action>
Paired docs are part of this deliverable (CLAUDE.md), because documented review-queue
behaviour changed.

(a) Fixture. `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` currently has 8 data
rows; exactly one ("Fay Review" / "Generic 1m robotic telescope", blank Site Code) exercised
the flag, and after this change it is no longer flagged — leaving the notebook with no
evidence for the genuine-failure branch at all. Add ONE synthetic, PII-free row (same 14
columns, `@example.com`-style address) with a blank Site Code and a Telescope / Instrument
string that names no telescope class and no space observatory, so `derive_telescope_class`
returns `''` and the row is still flagged. Verify the chosen instrument text derives `''`
before committing to it: it must contain no metre-aperture phrase (a digit optionally
followed by a decimal, then optional space, then a word-bounded `m`), no `0m4`/`1m0`/`2m0`/`4m0`
token, and no `juice` token — "Unassigned facility" satisfies all three. Give it a
telescope/instrument text and window that collide with no existing row's natural key.

(b) Notebook `import_campaign_csv_demo.ipynb`. Update the prose that is now wrong: the fixture
bullet list ("a blank `Site Code`, exercising the `site_needs_review` flag..."), the
post-import narrative claiming exactly one `site_needs_review`, the paragraph introducing the
site-less row inspection, and the requirements/evidence table row for D-08/D-09. The new
narrative is the contrast: the classed row is deliberately NOT flagged because its
`telescope_class` answers "why is there no site" (D-06), while the new class-less row IS
flagged — and the class is permanent, never cleared when a site later resolves. Extend the
site-less inspection code cell to print the new row alongside the existing one so both
branches show in executed output.

Then REGENERATE this notebook — it is the only notebook whose behaviour changed, and
re-executing the others is actively wrong (27-REVIEW-FIX.md IN-03: they were originally run
against the maintainer's populated dev DB and a clean-DB re-run rewrites unrelated narrative
output). Order matters: Task 1's `python manage.py migrate` must already have run against the
dev DB, so the demo row is unflagged before the importer sees it and the cell reports no churn
for it. Execute with the kernel CWD at `docs/notebooks/pre_executed/` (cell 2 asserts
`parents[2]` is the repo root): `jupyter nbconvert --to notebook --execute --inplace
import_campaign_csv_demo.ipynb`.

After execution, `git diff` the notebook and confirm the ONLY output changes are: the new
fixture row appearing in the inspection table, `site_needs_review` values/counters changing,
and the counts shifting by the one added row. If unrelated narrative output churns (Observatory
seeding flipping updated to created, or the previously-existing 8 rows flipping from unchanged
to created), the notebook ran against a different database than the committed outputs came
from — restore those cells' outputs from HEAD and report it rather than accepting the churn.
Never hand-edit executed output to fabricate a result; match the prose to the REAL regenerated
numbers, not to any number predicted in this plan.

(c) Runbook `docs/runbooks/telescope_runs_calendar.rst`. Three places: the "What the command
now writes (CANON-01/CANON-02)" note (~lines 200-213) — a row that gets a derived
`telescope_class` is deliberately NOT flagged for site review, because the class is the answer
to "why is there no site", and it is permanent (never cleared when a site later resolves); the
`repair_stale_campaign_run_sites` section (~lines 215-245) — it now skips class-carrying rows
and reports them under their own counter; the "`import_campaign_csv` unresolved rows"
troubleshooting section (~lines 403-424) — only rows with no class signal surface in "Sites
Needing Review", and per-site detail for a class-wide campaign arrives later on the linked
ObservationRecords (CANON-04), not by resolving the run to one site. Keep the Sphinx build
warning-free.

(d) `.planning/phases/27-the-canonical-run-record/27-REVIEW-FIX.md`. CR-01 currently sits
under "Skipped Issues" with "skipped by explicit instruction". Retitle that subsection to mark
it REJECTED and replace the reason with the decision: CR-01's premise is invalid —
`telescope_class` and a resolved site are not mutually exclusive, the field is a permanent
"why is there no site" fact (D-06), and clearing it would destroy correct data; the real
defect was the inverse, that classed rows were being flagged for site review, closed by quick
task 260730-jty (link this plan's directory). Update the Summary bullet at line 22 to
distinguish the rejected finding from the still-skipped WR-05. Leave the frontmatter counters
as they are — they describe the original fix run.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; grep -c "site_needs_review" docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb &amp;&amp; grep -q "REJECTED" .planning/phases/27-the-canonical-run-record/27-REVIEW-FIX.md &amp;&amp; ruff check docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb &amp;&amp; ruff format --check docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb &amp;&amp; sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D "exclude_patterns=notebooks/*,_build"</automated>
    <human-check>Read `git diff docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` and confirm every changed executed output is attributable to this change (new fixture row, site_needs_review values, shifted counts) — no unrelated narrative churn.</human-check>
  </verify>
  <done>The fixture has one added class-less row; the notebook's executed output shows the classed row unflagged AND the new row flagged, with prose matching the real numbers; the runbook states the corrected rule in all three places and Sphinx builds clean; 27-REVIEW-FIX.md records CR-01 as REJECTED with reasoning.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| staff browser → CampaignRunDecisionView | Staff POST `site_selection` free text crossing into site resolution |
| data migration → live dev/production DB | 0012 rewrites persisted rows with no user in the loop |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-jty-01 | Tampering | migration 0012 `.update()` | mitigate | Filter is a derived rule (`site_needs_review=True` AND non-blank `telescope_class`), never a pk list; writes only `site_needs_review`, never `site`, `telescope_class`, or `approval_status`; pks logged at INFO before the write |
| T-jty-02 | Elevation of Privilege | `_resolve_site` business-logic guard | accept | Unchanged: the existing server-side guard (`approval_status == APPROVED` and `site_needs_review`) still runs; this change only shrinks the set of eligible rows, it never widens it |
| T-jty-03 | Information Disclosure | demo fixture CSV row | mitigate | New row uses synthetic placeholder name/`@example.com` address only, matching CAMP-05's no-real-PII rule for this fixture |
| T-jty-SC | Tampering | npm/pip/cargo installs | accept | No package installs in this task |
</threat_model>

<verification>
- `python manage.py makemigrations --check --dry-run` reports no pending model changes.
- `python manage.py migrate` applies 0012 to the dev DB; the four live rows (pk 26/29/30/37) come back `site_needs_review=False` with their `telescope_class` values intact.
- The seven named Django test modules pass under `python manage.py test` (never the whole `solsys_code` suite).
- `ruff check solsys_code/` and `ruff format --check solsys_code/` are clean. Repo-wide `ruff check .` / `ruff format --check .` have 1 lint and 3 format failures that pre-date this task (`sync_gemini_observation_calendar_demo.ipynb`, `src/fomo/settings.py`, two `.planning/quick/260619-f7u/` scripts) — do not attempt to fix them here, just do not add new ones.
- Sphinx builds with no new warnings.
</verification>

<success_criteria>
- No code path flags a class-carrying CampaignRun for site review, and no code path clears `telescope_class`.
- The "Sites Needing Review" queue contains only rows a staff member can actually resolve.
- A genuinely unresolvable, class-less row is still flagged, still queued, and still resolvable.
- `models.py` and `calendar_utils.py` describe D-06's framing, with the never-cleared rule stated outright.
- The paired notebook and runbook document the change; CR-01 is on record as REJECTED.
</success_criteria>

<output>
Create `.planning/quick/260730-jty-stop-flagging-class-wide-and-space-runs-/260730-jty-SUMMARY.md` when done.
</output>
