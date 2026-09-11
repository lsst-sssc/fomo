---
phase: 34-the-observation-projector-trigger
reviewed: 2026-09-11T14:36:44Z
depth: deep
iteration: 2
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
  critical: 1
  warning: 4
  info: 5
  total: 10
status: issues_found
---

# Phase 34: Code Review Report (re-review after fix pass)

**Reviewed:** 2026-09-11T14:36:44Z
**Depth:** deep
**Files Reviewed:** 29
**Status:** issues_found

## Summary

This is a re-review of the current code after the 17-finding fix pass
(`5767a7a`..`0c71828`, plus `4eead9b`/`0a87174`/`f6861fc`). I re-verified every earlier
finding against the code rather than against `34-REVIEW-FIX.md`.

**Verification environment.** `pre-commit run ruff` and `pre-commit run ruff-format` are
clean on every changed Python file. `python manage.py test` over
`test_observation_projector`, `test_observation_projector_signals`,
`test_project_observation_calendar`, `test_calendar_display_extras`,
`test_calendar_template`, `test_calendar_utils`, `test_campaign_attribution`,
`test_load_telescope_runs` runs 327 tests green in 158 s. No source file was modified by
this review.

**Earlier findings genuinely closed (14 of 17):** CR-01's blank/whitespace
`observation_id` rejection is real and tested; CR-02's single-`try` relocation is real and
tested; WR-01's three `[:200]` truncations are real and tested; WR-02's softened
dry-run claims now name the exception in both docstrings and the runbook; WR-04's widened
`except Exception` and its `{'start': 12345}` regression test are correct (and
`record_time_window()` returns timezone-aware datetimes, so the `datetime.max` fallback is
genuinely comparable); WR-05's prefetch trim is correct — `calendar.html` contains no
reference to `observation_series_decoration` or `observation_group`, and the replacement
`test_modal_query_count_does_not_grow_with_group_size` measures a view that really does
render the tag; WR-06's comments/docstring are in place; WR-07's `url=''` lookup is
correct and the fields dict carries no colliding `url` key, so `create(**lookup, **fields)`
is safe; WR-08's module global is gone and the DB re-derivation is behaviourally equivalent
in every case I traced; WR-09's per-segment assertion is correct and the committed output
still satisfies it; IN-01 is accurate (`telescope_runs.SITES['FTS'] == 'E10' ==
LCO_SITE_CODE_TO_OBSCODE['coj']`, so the step-1/step-2 rerouting really is score- and
obscode-identical); IN-02, IN-03, IN-04, IN-05 are all applied as described. CR-01's
LCO/SOAR shared-portal-URL analysis is sound and I am not re-litigating it.

**What is not closed, and what the fix pass broke.** The CR-02 fix moved
`insert_or_create_calendar_event()` and `write_event_meta()` inside `project_record()`'s
`try`, but `project_queryset()` discards `project_record()`'s return value
(`observation_projector.py:478`). The sweep now counts and reports `created`/`updated` for
records whose write actually failed, and reports `unprojectable: 0, failed: 0`. I
reproduced this on a real test database (CR-01 below). The runbook's own counter
description — "`unprojectable` counts a record the sweep could not project at all" — is
now false, and TRIG-03/D-17's per-record failure isolation has no operator-visible signal
left.

Three further fixes are narrower than the finding they close: WR-03's visibility gate only
fires when a non-public `CampaignRun` is attached, so the un-attributed grouped event — the
common case — still publishes the internal portal RequestGroup name to an anonymous
visitor (the phase's own test asserts exactly that); WR-01's `transaction.atomic()`
savepoint was added to one of the three `project_record()` call sites; and the savepoint's
documented mechanism (`connection.needs_rollback`) is not the mechanism that actually
applies, because `project_record()` swallows the exception before the `atomic` block exits.

No leaked credentials, API keys or portal response bodies appear in the committed notebook
outputs or the `.sched06-baseline.json` (proposal code `KEY2026B-004`, portal request IDs,
target names and window times only). Every `Target` fixture in the new and edited tests
uses `NonSiderealTargetFactory` (CLAUDE.md convention), and the paired-docs rule is
satisfied: `project_observation_calendar_demo.ipynb`, `campaign_lifecycle_demo.ipynb` (a
genuine re-execution — every `iopub` timestamp and the scratch-DB path changed),
`sync_gemini_observation_calendar_demo.ipynb` and `docs/runbooks/telescope_runs_calendar.rst`
were all updated in-phase. No committed notebook output contradicts its own cell source.

## Critical Issues

### CR-01: The sweep reports `created`/`updated` and `failed: 0` for records whose projection actually failed — a regression introduced by the CR-02 fix

**Severity:** BLOCKER
**File:** `solsys_code/observation_projector.py:471-481` (specifically `:478`);
`solsys_code/management/commands/project_observation_calendar.py:200-215`;
`docs/runbooks/telescope_runs_calendar.rst` (sweep counter description)

**Issue:** `project_queryset()` counts the action from the *preview*, then calls the real
writer and throws its answer away:

```python
action = preview_calendar_event_action(before, fields)
facility_counters[action] += 1
if not dry_run:
    project_record(record)          # <-- return value discarded
rows.append({..., 'action': action})
```

Before commit `e3b274b` (the CR-02 fix), `insert_or_create_calendar_event()` and
`write_event_meta()` sat *outside* `project_record()`'s `try`, so a write failure
propagated into `project_queryset()`'s own outer `except` at line 482 and was at least
counted `unprojectable` and printed as a `-- skipping` line. Now that `project_record()`
swallows everything and returns `('unprojectable', <ExceptionName>)`, that signal is gone
entirely: the counter, the `rows` entry, the command's `failed` tally
(`project_observation_calendar.py:201-206`, which only fires on `row['action'] ==
'unprojectable'`) and the stderr line all report success.

Reproduced on a real Django test database (one LCO record, two pre-existing
duplicate-`url` `CalendarEvent`s — the exact input CR-02's own regression test constructs):

```
COUNTERS: {'LCO': {'created': 0, 'updated': 1, 'unchanged': 0, 'unprojectable': 0,
                   'site_lookups': 0, 'site_lookup_failed': 0}}
ROWS: [{'observation_id': 'sweep-dup-url', 'status': 'PENDING',
        'stage': 'queued', 'action': 'updated'}]
TITLES AFTER SWEEP (proof nothing was written): ['dup 0', 'dup 1']
```

The log line `unprojectable observation_id='sweep-dup-url': MultipleObjectsReturned` is
emitted at `warning` level and is the only trace — it never reaches the command's stdout
or stderr. This defeats TRIG-03/D-17's stated per-record failure isolation and falsifies
the runbook sentence "``unprojectable`` counts a record the sweep could not project at
all", the docstring at `observation_projector.py:433-435`, and the notebook's
`failed: 0 … unprojectable: 0` evidence for the whole real corpus. The same class of
failure is reachable for any write-time error (a `DataError`, a duplicate url, an FK
violation), not just `MultipleObjectsReturned`.

**Fix:** count what the writer actually did, not what the preview predicted:

```python
            action = preview_calendar_event_action(before, fields)
            if dry_run:
                facility_counters[action] += 1
            else:
                real_action, real_stage = project_record(record)
                if real_action == 'unprojectable':
                    logger.warning(
                        'sweep write failed for observation_id=%r: %s', record.observation_id, real_stage
                    )
                    facility_counters['unprojectable'] += 1
                    rows.append({
                        'observation_id': record.observation_id,
                        'status': record.status,
                        'stage': real_stage,
                        'action': 'unprojectable',
                    })
                    continue
                # keep the preview's action: project_record() can legitimately report
                # 'unchanged' when pre_fields_hook's own save already wrote this event
                # (see the docstring's counting rule).
                facility_counters[action] += 1
```

Add a regression test asserting `project_queryset()` over a duplicate-url record returns
`unprojectable: 1` and a `rows` entry with `action == 'unprojectable'`, and assert the
command's stdout carries `failed: 1`.

## Warnings

### WR-01: The WR-03 visibility gate closes only the attributed case — an un-attributed grouped event still publishes the portal RequestGroup name to anonymous visitors

**Severity:** WARNING
**File:** `solsys_code/templatetags/calendar_display_extras.py:626-630`;
`solsys_code/tests/test_calendar_template.py:1188-1213`

**Issue:** The gate added by `9f96920` is
`if meta.run is not None and not meta.run.is_publicly_visible: return None`. It therefore
suppresses the decoration only for an event that is *already* attributed to a
non-public `CampaignRun`. For the far more common projector-owned event — one with an
`observation_group` and `run is None` — the tag still returns `group_name` and a
`tom_observations:detail` link, and `event_form.html:146-158` renders both. The rendering
view (`tom_calendar.views.update_event`, wired at `solsys_code/calendar_urls.py:18`)
carries no authentication decorator, and `AUTH_STRATEGY='READ_ONLY'` means
`AuthStrategyMiddleware` does not block anonymous requests either.

The phase's own test proves the leak is live, not hypothetical —
`test_grouped_event_modal_shows_group_name_and_night_n_of_n` builds a
`CalendarEventMeta(observation_record=r1, observation_group=group)` with **no** `run`,
issues `self.client.get(...)` with no login, and asserts
`assertIn('Series Modal Group', content)`. The value at stake is not free text:
`backfill_lco_observations._group_name()` builds it as
`<LCO RequestGroup name> (<portal RequestGroup id>)`, so an anonymous visitor reads an
internal portal identifier off a public page. That was the substance of the original
WR-03; only its narrowest sub-case was closed.

**Fix:** gate on visibility rather than on attribution, e.g.

```python
    if meta.run is not None and not meta.run.is_publicly_visible:
        return None
    if meta.run is None and not _viewer_is_authenticated(context):
        return None
```

(a `takes_context=True` simple_tag, or simply wrapping the whole block in
`{% if user.is_authenticated %}` in `event_form.html`, which needs no tag change). Either
way, add an anonymous-client test asserting the group name is *absent* for a grouped,
un-attributed event, and record the decision next to `campaign_decoration()`'s gate so the
two rules stay legible side by side.

### WR-02: WR-01's savepoint protection was applied to one of three `project_record()` call sites

**Severity:** WARNING
**File:** `solsys_code/observation_projector.py:588-598` (m2m receiver),
`:473-478` (sweep), vs. `:523-525` (post_save receiver)

**Issue:** `receiver_on_record_save()` now wraps its call in `transaction.atomic()`, but
`receiver_on_group_membership_changed()` calls `project_record(record)` bare inside its own
`try`, and `project_queryset()` calls it bare too. The m2m receiver has exactly the hazard
WR-01 described: it runs inline inside the caller's transaction (Django's
`ManyRelatedManager.add()`/`.remove()`/`.clear()` all open an `atomic` block and send
`m2m_changed` inside it), and `project_record()` catches `Exception` — including
`DatabaseError` — without a savepoint. On PostgreSQL (CLAUDE.md's documented production
target) that leaves the caller's transaction aborted, so
`backfill_lco_observations`' `group.observation_records.add(...)` would fail later with a
confusing `TransactionManagementError` instead of the real error. The sweep has the same
shape: one bad row's database error can break the transaction for every later row.

**Fix:** move the savepoint into `project_record()` itself so every caller inherits it,
rather than repeating it per receiver:

```python
def project_record(record: ObservationRecord) -> tuple[str, str]:
    try:
        with transaction.atomic():
            facility = facility_for(record)
            fields, stage = event_fields_for(record, facility)
            event, action = insert_or_create_calendar_event({'url': event_url(record, facility)}, fields)
            write_event_meta(event, record)
    except Exception as exc:  # noqa: BLE001
        ...
```

and drop the now-redundant `with transaction.atomic():` from
`receiver_on_record_save()`.

### WR-03: The savepoint's documented mechanism does not apply — `needs_rollback` is never set, because the exception is swallowed inside the `atomic` block

**Severity:** WARNING
**File:** `solsys_code/observation_projector.py:503-510` (docstring), `:523-525`

**Issue:** The docstring asserts "A savepoint lets a database error here roll back only the
projector's own work", and `34-REVIEW-FIX.md` states the block's `__exit__` "issues a
ROLLBACK TO SAVEPOINT when the connection is marked as needing one". Neither is what
happens. In Django 5.2's `db/transaction.py`, `connection.needs_rollback` is set only by
`Atomic.__exit__` when an exception *propagates out* of an inner atomic block, or
explicitly by `set_rollback()` / `mark_for_rollback_on_error()`. Here `project_record()`
catches the exception *inside* the `with`, so `__exit__` sees `exc_type is None` and
`needs_rollback is False` and takes the **commit** branch — `savepoint_commit(sid)`, i.e.
`RELEASE SAVEPOINT`. Recovery on PostgreSQL happens only incidentally, through
`__exit__`'s `except DatabaseError` fallback (the `RELEASE` fails on an aborted
transaction, Django then rolls back to the savepoint and re-raises). On SQLite the whole
question is moot, so no test in this repo can distinguish a working fix from a broken one.

Django documents the supported idiom for exactly this situation ("catching a database
error inside an atomic block"), and it is not a bare savepoint.

**Fix:** make the intent explicit rather than relying on the fallback path:

```python
from django.db import transaction

    try:
        with transaction.atomic():
            with transaction.mark_for_rollback_on_error():
                action, stage = project_record(instance)
```

or, preferably, combine with WR-02 and let the exception propagate out of a
`transaction.atomic()` block *inside* `project_record()` before catching it, which is the
path `needs_rollback` actually covers. Correct the docstring either way — it currently
teaches a mechanism a future maintainer will not find in Django's source.

### WR-04: `--proposal` whose segments are all empty silently sweeps the entire corpus instead of nothing

**Severity:** WARNING
**File:** `solsys_code/management/commands/project_observation_calendar.py:177-180`

**Issue:**

```python
        if proposal_raw:
            codes = _parse_proposal_arg(proposal_raw)
            if codes:
                records = records.filter(parameters__proposal__in=codes)
```

`_parse_proposal_arg()` strips each comma-separated segment and drops empty ones, so
`--proposal ','`, `--proposal ' '` or `--proposal ',,,'` returns `[]`, the `if codes:` guard
skips the filter, and the sweep silently widens from "these proposals" to **every LCO/SOAR
record in the database**. That is the opposite of the operator's intent and, unlike the
`--facility` flag (whose `choices=` argparse validation rejects a bad value outright), it
is not reported anywhere — the summary line names no proposal scope. On a real run this
rewrites the whole corpus and can fire the one-time observed-site lookup against the live
portal for records the operator never asked about.

**Fix:** fail closed rather than widening:

```python
        if proposal_raw is not None:
            codes = _parse_proposal_arg(proposal_raw)
            if not codes:
                raise CommandError(f'--proposal {proposal_raw!r} names no usable proposal code.')
            records = records.filter(parameters__proposal__in=codes)
```

and add a test asserting `call_command('project_observation_calendar', '--proposal', ',,')`
raises `CommandError`.

## Info

### IN-01: `write_event_meta()` silently reverts any admin-set `is_verified=False`, which the WR-06 note does not mention

**Severity:** INFO
**File:** `solsys_code/observation_projector.py:328-335`; `solsys_code/models.py:21-30`;
`src/templates/tom_calendar/partials/calendar.html:245-258`

**Issue:** The WR-06 fix documents the two dead template branches as "kept only because
several tests (and any historical/admin-set row from before this phase) still construct a
`CalendarEventMeta` with `is_verified=False` directly". For a projector-owned event that
justification does not hold: `write_event_meta()` writes `is_verified: True`
unconditionally on every projection, so an admin who sets the flag `False` through the
inline has it silently reverted by the next `ObservationRecord.save()` or sweep. The field
is effectively read-only-`True` for exactly the events the branches were written for.

**Fix:** say so in the model docstring and the two template comments ("an admin-set
`False` on a projector-owned event is reverted by the next projection"), or make
`is_verified` non-editable on the admin inline for rows whose `observation_record` is set.

### IN-02: The new LCO/SOAR shared-URL test pins the url but not which record ends up owning the companion row

**Severity:** INFO
**File:** `solsys_code/tests/test_observation_projector.py:462-478`

**Issue:** Not re-litigating the shared-portal-URL decision — the analysis (SOAR is
scheduled through the same LCO portal, so one `observation_id` is one request) is correct.
But the consequence the test leaves unstated is that `CalendarEventMeta.observation_record`
is a `OneToOneField`, so when an LCO record and a SOAR record carry the same
`observation_id`, `write_event_meta()`'s line 325-327 clears the first record's claim and
hands the single companion row to whichever record projected last — silently, with no log
line and no counter. `test_lco_and_soar_records_sharing_an_observation_id_get_one_shared_url`
asserts only `count() == 1` on the url, so the ownership outcome is untested and
undocumented.

**Fix:** extend the test with
`self.assertEqual(CalendarEventMeta.objects.get(event__url=lco_url).observation_record_id,
soar_record.pk)` and a one-line comment stating that last-writer-wins is the accepted
outcome for this (expected-never-to-happen) pairing.

### IN-03: `event_form.html` still references the retired `sync_lco_observation_calendar` command

**Severity:** INFO
**File:** `src/templates/tom_calendar/partials/event_form.html:109`

**Issue:** The comment reads "…raw sync_lco_observation_calendar/sync_gemini_observation_calendar/
…" but `sync_lco_observation_calendar` was deleted in this phase (D-18). The phase edited
this file, so the stale name was in front of the author. (`docs/design/*.rst` and
`load_telescope_runs_demo.ipynb` carry the same stale name but are outside this review's
file scope.)

**Fix:** replace with "the observation projector / `sync_gemini_observation_calendar`".

### IN-04: Review-finding IDs (`WR-06`, `CR-02`, `IN-05`, …) are embedded throughout shipped source and templates

**Severity:** INFO
**File:** `solsys_code/observation_projector.py` (10 sites),
`solsys_code/templatetags/calendar_display_extras.py`, `solsys_code/models.py`,
`solsys_code/views.py`, `solsys_code/management/commands/load_telescope_runs.py`,
`src/templates/tom_calendar/partials/calendar.html:246-256`

**Issue:** The fix pass left its own bookkeeping in the code: comments such as "WR-06
(Phase 34 review): …", "see 34-REVIEW-FIX.md WR-06 for the decision…" and "CR-02: every
write this function makes…". These identifiers resolve only against `.planning/`, which is
not part of the shipped source tree, and they will outlive the review they refer to. The
codebase's existing convention is to cite durable decision IDs (`D-07`, `PROJ-01`,
`SYNC-04`), not per-review finding numbers. The `calendar.html` `{% comment %}` block is
the worst instance — a ten-line review postmortem inside a production template.

**Fix:** keep the *reasoning* and drop the finding IDs, e.g. "`telescope`/`instrument`/
`proposal` are externally sourced and write into `CharField(max_length=200)`; PostgreSQL
raises `DataError` on overflow" with no "WR-01:" prefix. Shorten the `calendar.html`
comment to one line.

### IN-05: The demo notebook imports a private cross-module helper the codebase explicitly calls out as an anti-pattern

**Severity:** INFO
**File:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` (cell 22)

**Issue:** The new D-07 cell does
`from solsys_code.campaign_attribution import (..., _extract_lco_site_code, ...)`.
`calendar_utils.update_calendar_event_key_and_fields()`'s own docstring names "a
cross-module import of a private helper" as "the exact anti-pattern the retired
`backfill_range_calendar_events` command exemplified and the v2.2 milestone's locked
constraints call out". A committed demo notebook is documentation of how to use the module,
so it teaches the pattern it forbids.

**Fix:** demonstrate the bridge through the public surface —
`telescope_match_score(run, telescope_code='FTN', …)` already does, and the loop over
`OBSERVED_TELESCOPE_SITE_CODES` can assert on the returned evidence string ("orphan LCO
site code 'ogg' resolves to obscode F65") instead of calling the private extractor
directly.

---

_Reviewed: 2026-09-11T14:36:44Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Iteration: 2 (re-review after fix pass `5767a7a`..`f6861fc`)_
