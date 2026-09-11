---
phase: 34-the-observation-projector-trigger
reviewed: 2026-09-11T15:42:08Z
depth: deep
iteration: 4
files_reviewed: 29
files_reviewed_list:
  - CLAUDE.md
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
  - docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json
  - docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb
  - docs/notebooks.rst
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/apps.py
  - solsys_code/calendar_utils.py
  - solsys_code/campaign_attribution.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/management/commands/project_observation_calendar.py
  - solsys_code/models.py
  - solsys_code/observation_projector.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/helpers.py
  - solsys_code/tests/test_calendar_display_extras.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_campaign_attribution.py
  - solsys_code/tests/test_campaign_attribution_views.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_load_telescope_runs.py
  - solsys_code/tests/test_observation_projector.py
  - solsys_code/tests/test_observation_projector_signals.py
  - solsys_code/tests/test_project_observation_calendar.py
  - solsys_code/views.py
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/event_form.html
findings:
  critical: 0
  warning: 3
  info: 9
  total: 12
status: issues_found
---

# Phase 34: Code Review Report (re-review after fix pass, iteration 4 — final)

**Reviewed:** 2026-09-11T15:42:08Z
**Depth:** deep
**Files Reviewed:** 29
**Status:** issues_found

## Summary

Re-review of the current tree after the third fix pass (`3dfa45c`, `00c3c0b`, `a432094`,
`427d99c`). Every one of the four findings that pass was given was verified against the
**code**, not against `34-REVIEW-FIX.md`, by tracing each call site and each run mode.

**Verification environment.** `pre-commit run ruff --files` and `ruff-format --files` are
clean over all twenty changed Python files. `python manage.py test` across
`test_observation_projector`, `test_observation_projector_signals`,
`test_project_observation_calendar`, `test_calendar_display_extras`,
`test_calendar_template`, `test_calendar_utils`, `test_campaign_attribution`,
`test_campaign_attribution_views`, `test_campaign_reconciler`, `test_load_telescope_runs`
runs **464 tests green in 56 s** (up from 459 — the four fix commits added 5 tests). No
source file was modified by this review.

### The four findings are closed — what I checked, and how

- **CR-01 (`--dry-run` vs real-run disagreement) — closed, both modes, both call paths.**
  `project_queryset()` now detects the one write failure a dry run can see without writing
  (`CalendarEvent.objects.filter(url=url).count() > 1`,
  `observation_projector.py:503-520`) and counts it `unprojectable` ahead of the preview.
  I confirmed the detection is exactly equivalent to the real failure rather than a
  heuristic: `insert_or_create_calendar_event()`'s url-keyed branch is
  `CalendarEvent.objects.get_or_create(**lookup, defaults=fields)`
  (`calendar_utils.py:572`), whose `.get()` raises `MultipleObjectsReturned` for `count() >
  1` and never for `count() <= 1` — so the predicate is exact in both directions. I also
  checked for the false-positive risk the fix could have introduced: `get_or_create` can
  never *create* a second row at an existing url, so a sweep cannot manufacture a duplicate
  mid-run, and two records sharing a url in one sweep (the LCO/SOAR shared-id case) still
  converge to `count() == 1`. `test_dry_run_agrees_with_the_real_run_on_a_duplicate_url_failure`
  asserts all three surfaces agree (the function, the command's `--dry-run`, and the real
  run) and that the dry run wrote nothing. The `project_queryset()` docstring
  (`:410-433`) and the runbook (`:183-198`) were both narrowed to the honest "lower bound"
  claim.
- **WR-01 (series-decoration leak) — closed, and it does not over-hide.** The viewer check
  is now unconditional and first (`calendar_display_extras.py:676-682`), with
  `meta.run.is_publicly_visible` layered on as an *additional* constraint. I walked all six
  cells of (`run` none/approved/pending) × (anonymous/authenticated): an authenticated
  viewer still sees the decoration for `run is None` **and** for an approved run — the
  specific regression risk flagged for this pass — and only the pending-review case is
  hidden from them, which is the rule the docstring and the runbook both now state.
  `test_viewer_gate_applies_regardless_of_run_and_run_visibility_is_a_second_gate` pins all
  six cells at the tag level, and `test_grouped_and_attributed_event_hides_group_name_from_anonymous_viewer`
  pins the previously-leaking case end to end through the real unauthenticated modal view,
  while asserting the unrelated campaign block is *still* visible (so the fix did not
  over-reach into `campaign_decoration()`). `test_modal_query_count_does_not_grow_with_group_size`
  now logs in and is therefore no longer vacuous. `event_form.html:148` is the tag's only
  call site and is top-level, so `takes_context=True` receives a real request context.
- **WR-02 (`'ogg': 'F65'` bridge) — closed, with no scoring change for any covered case.**
  I traced every label through the new resolution order. `'FTN'`/`'FTS'`/`'SOAR'` resolve at
  the new label-keyed step 1 to `F65`/`E10`/`I33` — the same obscodes as before, so
  `test_telescope_match_d07_renamed_observed_labels_still_resolve_site_level` passes on the
  same values. `'COJ-1m0'`/`'COJ-2m0'` still resolve through the site-keyed step 2 (`'coj'`
  is retained). `'FTS'` is also a key of `telescope_runs.SITES` with obscode `'E10'`
  (`telescope_runs.py:21`), identical to what step 1 now returns, so pre-empting step 3
  changes nothing. `'OGG-0m4'` now correctly degrades to `TELESCOPE_MATCH_APERTURE_ONLY`
  instead of the false `TELESCOPE_MATCH_NONE`, pinned by a new test. Removing `'sor'` from
  the site table is safe: `SITE_TELESCOPE_MAP` emits no `SOR-*` label any more and `'SOAR'`
  is covered by step 1. See WR-02 below for the residue this fix left behind.
- **WR-03 (m2m receiver) — closed.** `receiver_on_group_membership_changed()`
  (`:665-700`) now carries the same "second, outer layer of defence" explanation on both
  its docstring and its inline comment, reads `project_record()`'s `(record_action, stage)`
  return value, and logs a membership-specific warning. The local is named `record_action`
  so it cannot shadow the signal's own `action` parameter. `test_group_add_with_write_failure_logs_a_membership_specific_warning`
  asserts the membership change survives and both log lines are present.

**Other cross-cutting checks that came back clean.** No exception *value* is logged
anywhere in the new code (every path carries only `type(exc).__name__` or a fixed string) —
SYNC-09/D-13 holds. `apps.py ready()` uses `dispatch_uid` on all three `connect()` calls so
a second `ready()` cannot double-fire, and its function-local imports avoid pulling
`solsys_code.views`/SPICE into app loading. `receiver_on_record_delete()`'s
`instance.calendar_event_meta` matches the `related_name` at `models.py:72`. The
`load_telescope_runs` WR-07 `url: ''` lookup is safe: `CalendarEvent.url` is
`URLField(blank=True, default="")` (non-nullable), and the reconciler only ever looks up and
re-keys events it already found by a `RUN:` url, so it never leaves a classical event with a
non-blank url that the new lookup would miss. `record_time_window()` returns timezone-aware
UTC on both branches, so `_window_start_or_max()`'s `datetime.max` sentinel is always
comparable and `list.sort()` cannot raise. No credential, API key or portal response body
appears in any committed notebook output or in
`project_observation_calendar_demo.sched06-baseline.json`. `docs/notebooks.rst` correctly
re-points the toctree and no `.rst`/`.html` in the toctree references the deleted
`sync_lco_observation_calendar_demo`. Every `Target` fixture in the new and edited tests
uses `NonSiderealTargetFactory`. `project_observation_calendar_demo.ipynb`'s committed
output does **not** contradict its source after these four commits (it never invokes
`--proposal`, never calls `observation_series_decoration()`, and its `--dry-run` cell shows
`unprojectable: 0` over a corpus with no duplicate urls — still the correct answer).

**Deliberately not re-litigated,** per this pass's scope: the LCO/SOAR shared-portal-url
design (user-resolved), D-14's "deleting a record deletes its event, audit trail included"
(`34-01-PLAN.md:396` records this as a locked, costly-reversibility decision the user
accepted), and the general decision not to re-execute the two live-portal notebooks.

**What this pass found.** No blockers. Three warnings, all of them residue from this last
fix pass rather than new defects in the phase's own logic: one committed notebook cell whose
executed output is now demonstrably unproducible from its own source, one branch the WR-02
fix rendered unreachable while leaving a test docstring crediting it as the mechanism under
test, and one module docstring the CR-01 fix corrected in two of three places. Seven of the
prior Info findings are unchanged and re-reported; two are new.

## Narrative Findings (AI reviewer)

## Critical Issues

None. The four findings in this pass's fix scope are closed at every call site and in both
run modes, and no new incorrect-behaviour, security or data-loss defect was found.

## Warnings

### WR-01: `campaign_lifecycle_demo.ipynb` cell 22's committed output is now unproducible from its own source — three separate lines contradict current code

**Severity:** WARNING
**File:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` cells 21-22
(JSON lines 983-1085)

**Issue:** This is the one notebook case this pass's scope explicitly asks to be reported:
committed output that now contradicts its own committed source. The cell's source is
unchanged and still reads

```python
resolved_obscode = LCO_SITE_CODE_TO_OBSCODE.get(_extract_lco_site_code(telescope_label))
print(f'{telescope_label:<5} -> site code {resolved_site_code!r} -> obscode {resolved_obscode!r}')
...
match_score, evidence = telescope_match_score(ftn_demo_run, telescope_code='FTN', ...)
print(f'FTN match level: {match_score} ({evidence})')
```

while its committed output reads

```
FTN   -> site code 'ogg' -> obscode 'F65'
FTS   -> site code 'coj' -> obscode 'E10'
SOAR  -> site code 'sor' -> obscode 'I33'

FTN match level: 1.0 (orphan LCO site code 'ogg' resolves to obscode F65, matching the run's site obscode F65)
```

Under `a432094`, re-running that exact source produces `obscode None` for FTN and for SOAR
(`'ogg'`/`'sor'` were removed from `LCO_SITE_CODE_TO_OBSCODE`,
`campaign_attribution.py:76-78`) and an evidence string of
`"orphan observed telescope 'FTN' resolves to obscode F65, matching the run's site obscode
F65"` (`campaign_attribution.py:361-364`). Three of the five printed lines are now wrong.

The prose caveat `a432094` added to cell 21 covers only the first two and then asserts
"The cell's actual point, the `telescope_match_score()` call below the loop, is unaffected"
— but the *evidence string that call printed* is the fourth stale line, and it is the line a
reader looks at to learn how the bridge works. A reader following the notebook now learns a
mechanism (`'ogg'` → site-keyed → F65) that the code explicitly removed as incorrect, from a
cell that is presented as executed proof. CLAUDE.md's paired-docs rule names this notebook as
`campaign_attribution.py`'s paired artifact and requires "cells/prose exercising the new
behavior with real executed output", not a prose annotation on stale output.

**Fix:** rewrite the loop to demonstrate the bridge through the public surface (which also
closes IN-05 — the cell would no longer need the `_extract_lco_site_code` private import)
and re-execute **this cell only**, which touches no live portal:

```python
from solsys_code.campaign_attribution import OBSERVED_TELESCOPE_OBSCODES, telescope_match_score

for telescope_label, obscode in OBSERVED_TELESCOPE_OBSCODES.items():
    print(f'{telescope_label:<5} -> obscode {obscode!r}  (label-keyed: one label, one telescope)')
```

then drop the caveat paragraph from cell 21, per the fix report's own follow-up note that
"the caveat added by this pass's commit should be removed along with the code it describes".
If re-execution genuinely cannot be scoped to one cell, the minimum is to extend the caveat
to name the evidence-string change too, and to stop claiming the `telescope_match_score()`
demonstration is "unaffected" when its printed output is one of the stale lines.

### WR-02: the WR-02 fix made `_extract_lco_site_code()`'s observed-label branch unreachable for every in-repo caller, and left the D-07 regression test's docstring crediting it as the mechanism under test

**Severity:** WARNING
**File:** `solsys_code/campaign_attribution.py:256-284` (the
`OBSERVED_TELESCOPE_SITE_CODES` consultation at `:281-283`), `:359-384`;
`solsys_code/tests/test_campaign_attribution.py:132-139`

**Issue:** `_extract_lco_site_code()` has exactly one production caller — `:372`, inside
`telescope_match_score()` — and its return value is used only under
`if lco_site_code and lco_site_code in LCO_SITE_CODE_TO_OBSCODE and run.site_id is not None`.
Trace the three observed labels through it after `a432094`:

- `'FTN'` → `'ogg'`, which `a432094` removed from `LCO_SITE_CODE_TO_OBSCODE` → condition
  false, always.
- `'SOAR'` → `'sor'`, likewise removed → condition false, always.
- `'FTS'` → `'coj'`, which *is* in the table — but `'FTS'` reaches `:372` only when step 1
  fell through, and step 1 falls through for a table-resident label only when
  `run.site_id is None`, which is exactly the condition `:373` also requires to be false.

So the `OBSERVED_TELESCOPE_SITE_CODES.get()` branch at `:281-283` can no longer change the
outcome of any scoring call. Its only remaining reader in the repository is the demo
notebook's private-helper import (WR-01/IN-05 above) — a consumer the codebase itself
documents as an anti-pattern (`calendar_utils.update_calendar_event_key_and_fields()`'s
docstring). That is a branch with no live consumer, kept alive by the one import that is
already filed for removal.

Worse for maintenance, `test_telescope_match_d07_renamed_observed_labels_still_resolve_site_level`
(`test_campaign_attribution.py:132-139`) still says in its own docstring that this is
"exactly the asymmetric regression this bridge (campaign_attribution's own
OBSERVED_TELESCOPE_SITE_CODES consultation in `_extract_lco_site_code`) closes." That
sentence is now false — the test passes because of `OBSERVED_TELESCOPE_OBSCODES` at
`:359-369`, and would keep passing if the branch it names were deleted outright. A
maintainer told by a green test which mechanism is load-bearing is being told the wrong one;
the fix report's claim that this test "continues to pass unchanged, confirming the new
label-based step reproduces the prior correct behaviour" is true about the assertion and
false about the docstring.

**Fix:** update the test docstring to name the actual mechanism
(`OBSERVED_TELESCOPE_OBSCODES`, checked as step 1 before any site-code path), and either
delete the now-unreachable branch at `:281-283` together with the notebook import that keeps
it alive, or — if the site-code-shaped return is meant to stay as a public bridge for future
callers — promote it out of a leading-underscore private helper so it has a supported
consumer, and say in its docstring that it is currently exercised only from outside this
module.

### WR-03: `project_observation_calendar.py`'s module docstring still claims "one documented exception" to dry-run/real-run agreement — the same sentence the CR-01 fix corrected in two other places

**Severity:** WARNING
**File:** `solsys_code/management/commands/project_observation_calendar.py:7-13`

**Issue:** The command module's docstring reads:

> This command shares the receiver's own comparison rule … so a ``--dry-run`` count agrees
> with what a real sweep would do for every field derived from a record's own already-stored
> state -- **with one documented exception (WR-02)**: a dry run never performs the one-time
> observed-site lookup …

`3dfa45c` corrected exactly this claim in `project_queryset()`'s docstring
(`observation_projector.py:410-433`, now "one override in each mode" plus an explicit
lower-bound caveat) and in the runbook (`telescope_runs_calendar.rst:183-198`, now "with two
exceptions"), but did not touch the command module that an operator reading `--help`-adjacent
source lands on first. The file that *defines* the operator-facing counters is now the one
place still asserting the pre-fix invariant. This is an incomplete fix, not a pre-existing
wart: the same sentence was rewritten twice in the same commit.

It also carries a review ID (`WR-02`) that resolves only against `.planning/`, which is not
shipped — see IN-04.

**Fix:** bring the module docstring in line with the two it already diverges from:

```python
``--dry-run`` agrees with a real sweep for every field derived from a record's own
already-stored state, with two exceptions: a dry run never performs the one-time
observed-site lookup (so ``site_lookups`` is always 0 and a coarse-to-observed token change
reports ``unchanged``), and a dry run can only predict a write failure it can detect without
writing (today, a duplicate calendar-event url), so its ``unprojectable`` count is a lower
bound on the real sweep's.
```

## Info

### IN-01: `write_event_meta()` silently reverts any admin-set `is_verified=False`, which the WR-06 note still does not mention

**Severity:** INFO
**File:** `solsys_code/observation_projector.py:328-337` (`'is_verified': True` at `:332`);
`solsys_code/models.py:21-30`, `:46-51`; `solsys_code/admin.py:115`;
`src/templates/tom_calendar/partials/calendar.html:245-258`

**Issue:** Unchanged from the previous pass and re-verified. The model docstring says a
`False` value "can only be a historical row … or one set directly (the admin form, a test
fixture)" and stops there. It omits the consequence: `write_event_meta()` writes
`is_verified: True` unconditionally on every projection, and `is_verified` is *not* in
`CalendarEventMetaInline.readonly_fields` (`admin.py:115` lists only
`confirmed_by`/`confirmed_at`/`observation_record`/`observation_group`), so an admin who
clears the box on a projector-owned row has it silently restored by the next
`ObservationRecord.save()` or sweep. For exactly the events the two dead `calendar.html`
branches were written for, the field is effectively read-only-`True`.

**Fix:** add one sentence to the model docstring and the two template comments ("an
admin-set `False` on a projector-owned event is reverted by the next projection"), or make
`is_verified` read-only on the inline for rows whose `observation_record` is set.

### IN-02: the LCO/SOAR shared-URL test pins the url but not which record ends up owning the companion row

**Severity:** INFO
**File:** `solsys_code/tests/test_observation_projector.py:462-479`

**Issue:** Unchanged. Not re-litigating the shared-portal-url decision.
`CalendarEventMeta.observation_record` is a `OneToOneField` (`models.py:67-74`), so when an
LCO and a SOAR record share an `observation_id`, `write_event_meta()` hands the single
companion row to whichever projected last — silently, with no log line and no counter. The
test asserts only `count() == 1` on the url, so the ownership outcome stays untested and
undocumented.

**Fix:** add
`self.assertEqual(CalendarEventMeta.objects.get(event__url=lco_url).observation_record_id, soar_record.pk)`
and a one-line comment stating last-writer-wins is the accepted outcome for this
expected-never-to-happen pairing.

### IN-03: `event_form.html` still references the retired `sync_lco_observation_calendar` command

**Severity:** INFO
**File:** `src/templates/tom_calendar/partials/event_form.html:109`

**Issue:** Unchanged, and the file has now been edited by this phase and by two subsequent
fix passes without the stale name being noticed. The comment still reads "…raw
sync_lco_observation_calendar/sync_gemini_observation_calendar/load_telescope_runs output"
for a command this phase deleted (D-18). (`docs/design/*.rst:108`/`:262` and
`load_telescope_runs_demo.ipynb:539` carry the same stale name but are outside this review's
file scope.)

**Fix:** replace with "the observation projector / `sync_gemini_observation_calendar`".

### IN-04: review-finding IDs are embedded throughout shipped source, templates and tests

**Severity:** INFO
**File:** `solsys_code/observation_projector.py:298`, `:303`, `:378`, `:653`;
`solsys_code/templatetags/calendar_display_extras.py:518`, `:582`, `:686`;
`solsys_code/management/commands/project_observation_calendar.py:10`, `:24`, `:93`;
`solsys_code/management/commands/load_telescope_runs.py:209`; `solsys_code/models.py:22`,
`:46`; `src/templates/tom_calendar/partials/calendar.html:248`, `:255`, `:276`; and ~20
test docstrings

**Issue:** Unchanged in kind. These identifiers resolve only against `.planning/`, which is
not shipped, and they will outlive the review they name. Recorded honestly: this pass's four
commits added essentially no new IDs to source (only `3dfa45c`'s test at
`test_project_observation_calendar.py:273` reuses one), which is an improvement on the prior
pass; and the practice is pre-existing and widespread across `campaign_views.py`,
`campaign_utils.py` and `import_campaign_csv.py`, so this is a repo-wide convention question
rather than a Phase 34 novelty. The worst single instance remains the ten-line
`{% comment %}` review postmortem at `calendar.html:245-256`.

**Fix:** keep the reasoning, drop the finding IDs. Shorten the `calendar.html` comment to
one line.

### IN-05: the demo notebook imports a private cross-module helper the codebase explicitly calls out as an anti-pattern

**Severity:** INFO
**File:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` cell 22 (JSON line
1028)

**Issue:** Unchanged. The D-07 cell still does
`from solsys_code.campaign_attribution import (…, _extract_lco_site_code, …)`.
`calendar_utils.update_calendar_event_key_and_fields()`'s own docstring names "a
cross-module import of a private helper" as "the exact anti-pattern the retired
`backfill_range_calendar_events` command exemplified and the v2.2 milestone's locked
constraints call out". A committed demo notebook is documentation of how to use the module.
As of WR-02 above, this import is now also the *only* consumer keeping a dead branch alive.

**Fix:** demonstrate the bridge through the public surface — see WR-01's suggested cell,
which closes both findings at once.

### IN-06: `CalendarEvent.url` is the one externally-sourced 200-character column the truncation fix did not bound

**Severity:** INFO
**File:** `solsys_code/observation_projector.py:214-224`, `:298-308`

**Issue:** Unchanged. The WR-01 truncation comment (`:298-304`) enumerates
"telescope/instrument/proposal … CharField(max_length=200)" and notes `title` is truncated in
`title_for()`. It misses `url`: `CalendarEvent.url` is a `URLField` (tom_calendar
`models.py:33`) defaulting to `max_length=200`, and `event_url()` writes
`facility.get_observation_url(record.observation_id)` into it with no bound, from an
`observation_id` that is `CharField(max_length=255)`. A ~165-character `observation_id`
overflows the column and raises `DataError` on PostgreSQL. Info rather than Warning because
the value is the identity key (truncating it would be worse than failing) and, after this
phase's savepoint plus the CR-01 fix, the overflow degrades cleanly to a rolled-back
`unprojectable` with a stderr line. It is also precisely the "no cheap pre-write signal"
case the CR-01 fix's new lower-bound caveat now documents.

**Fix:** extend the comment to say `url` is deliberately *not* truncated because it is the
identity key, and that an over-length id is handled as `unprojectable` instead — so the
omission reads as a decision rather than an oversight.

### IN-07: `[F]` is the legend label for both `FAILURE_LIMIT_REACHED` and `NOT_ATTEMPTED`, and reads as "Failed" for both

**Severity:** INFO
**File:** `solsys_code/observation_projector.py:87-92`;
`solsys_code/templatetags/calendar_display_extras.py:130-138`

**Issue:** Unchanged. `_FAILURE_MARKER_BY_STATUS` maps `NOT_ATTEMPTED` to `'[F]'`, and
`_OBSERVATION_STATUS_LEGEND` renders `[F]` as "Failed" on the public calendar, so a
never-attempted observation is shown to a calendar reader as a failure. Note the runbook's
own marker table (`telescope_runs_calendar.rst:81-82`) already says "Failed (failure limit
reached, or never attempted)" — the on-page legend is the less accurate of the two shipped
descriptions. Phase 37 (STATUS-01/02) owns the final vocabulary, which is why this is Info.

**Fix:** relabel `[F]` in `_OBSERVATION_STATUS_LEGEND` as "Failed / not attempted", matching
the wording the runbook already uses, or give `NOT_ATTEMPTED` its own marker.

### IN-08: the new dry-run `unprojectable` path is the only one of four that logs nothing

**Severity:** INFO
**File:** `solsys_code/observation_projector.py:503-520`

**Issue:** New, introduced by `3dfa45c`. `project_queryset()` has four places that count a
row `unprojectable`; three log a `logger.warning` first — the `event_fields_for()` failure
(`:493`), the real-run write failure (`:539`), and the outer catch-all (`:559-561`). The new
dry-run duplicate-url branch counts and appends the row but logs nothing. The command's
stderr line still fires (it keys off `row['action']`), so an operator running the command
sees it; anyone calling `project_queryset(..., dry_run=True)` as a library function and
reading logs rather than the return value does not. Minor and consistent with "a dry run
writes nothing", but the asymmetry is unexplained at the site.

**Fix:** either add the matching `logger.warning('dry run: duplicate url would fail
observation_id=%r', record.observation_id)`, or add one comment line saying a dry run
deliberately stays silent and reports only through its return value.

### IN-09: the runbook's ring description implies an inconsistent-record entry gets the queued ring, not the terminal ring

**Severity:** INFO
**File:** `docs/runbooks/telescope_runs_calendar.rst:104-106`;
`solsys_code/templatetags/calendar_display_extras.py:129, 201-206`

**Issue:** New. The runbook says "a Queued or an Inconsistent record entry is ringed, a
Scheduled or Observed entry is not, and an expired, cancelled or failed entry carries the
terminal ring" — grouping `[?]` with `[Q]` and listing the terminal ring separately. In the
code, `'[?] '` is a member of `_TERMINAL_PREFIXES` and therefore gets the **terminal** ring,
not the queued one; only `'[QUEUED] '`/`'[Q] '` get the queued ring. The tag's own docstring
states this correctly ("'[?] ' … reads as terminal here even though nothing actually
failed"); the runbook sentence reads the other way.

**Fix:** reword to "a Queued entry is ringed; a Scheduled or Observed entry is not; an
expired, cancelled, failed **or inconsistent** entry carries the terminal ring — an
inconsistent record is ringed because it needs an operator's eye, not because anything
failed."

---

_Reviewed: 2026-09-11T15:42:08Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Iteration: 4 (final re-review after fix pass `3dfa45c`, `00c3c0b`, `a432094`, `427d99c`)_
