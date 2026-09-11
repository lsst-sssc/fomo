---
phase: 34-the-observation-projector-trigger
reviewed: 2026-09-11T05:29:34Z
depth: deep
files_reviewed: 27
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
  - solsys_code/observation_projector.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/helpers.py
  - solsys_code/tests/test_calendar_display_extras.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_campaign_attribution.py
  - solsys_code/tests/test_campaign_attribution_views.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_observation_projector.py
  - solsys_code/tests/test_observation_projector_signals.py
  - solsys_code/tests/test_project_observation_calendar.py
  - solsys_code/views.py
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/event_form.html
findings:
  critical: 2
  warning: 9
  info: 6
  total: 17
status: issues_found
---

# Phase 34: Code Review Report

**Reviewed:** 2026-09-11T05:29:34Z
**Depth:** deep
**Files Reviewed:** 27
**Status:** issues_found

## Summary

Reviewed the new observation projector (`observation_projector.py`), its sweep command, the
display-layer additions (status rings, legend, request-time series decoration), the
attribution/calendar_utils edits, the view prefetch change, both templates, and the committed
notebook/runbook artifacts. `pre-commit run ruff` is clean on every changed Python file and
`python manage.py test solsys_code.tests.test_observation_projector
test_observation_projector_signals test_project_observation_calendar
test_calendar_display_extras` passes (157 tests).

The phase-defining invariants that hold: the `post_save` path makes no network call (proven by
a mocked `make_request` test and by reading every call in `event_fields_for()`); the reconciler
never adopts a facility-URL-keyed event (`_attributed_nights()` reads attributed rows only, and
the per-night adopt contract was retired in Phase 33); the series tag performs no write.

The invariants that do **not** hold, and are the substance of this review: **"exactly one
CalendarEvent per record" has no enforcement at all.** `CalendarEvent.url` is a plain
non-unique `URLField` (tom_calendar `models.py:34`), `event_url()` is not namespaced by
facility (LCO and SOAR share `portal_url` in `settings.py:230/238`), and nothing rejects a
blank `observation_id`. I reproduced both failure modes against a real test database: two
records with a blank `observation_id` silently collapse onto one event and steal each other's
one-to-one companion link (CR-01), and a single duplicate url row — creatable through the
calendar's own *unauthenticated* event form, which exposes `url` as an editable field —
permanently breaks that record's projection with nothing but a log line (CR-02).
`project_record()`'s "Never raises (TRIG-02)" docstring is also literally false: three of its
four calls sit outside its `try`.

Secondary theme: several claims made in docstrings, the threat register and the notebook are
stronger than the code supports — the dry-run/real-run equivalence (WR-02), the
"query count does not grow" mitigation (WR-05), and the second-sweep notebook assertion
(WR-09) are each weaker in fact than in prose.

No leaked credentials, portal response bodies, or submitter contact details were found in the
committed notebook outputs or the `.json` baseline (they contain proposal code
`KEY2026B-004`, portal request IDs and target names only — the same data the calendar already
publishes).

## Critical Issues

### CR-01: The projector's identity key is not unique — records with a blank or duplicate `observation_id` silently overwrite each other's event and steal the companion link

**File:** `solsys_code/observation_projector.py:213-223` (`event_url`), `299-321`
(`write_event_meta`), `341`

**Issue:** `event_url()` returns `facility.get_observation_url(record.observation_id)` with no
validation. `ObservationRecord.observation_id` is `CharField(max_length=255)` with no unique
constraint and no `blank=False` enforcement at the ORM level, so `''` is a perfectly storable
value, and `get_observation_url('')` yields `https://observe.lco.global/requests/` for *every*
such record. The key is also not namespaced by facility: `settings.py:230` and `:238` give LCO
and SOAR the identical `portal_url`, so an LCO record and a SOAR record with the same
`observation_id` also collide.

Reproduced on a real test database (two LCO records, blank `observation_id`, different
statuses):

```
r1 url: https://observe.lco.global/requests/
r2 url: https://observe.lco.global/requests/
events with that url: 1
meta rows: [(1, 2)]          # record 1's companion link was silently taken over
titles: ['[O] 2m0 <target>'] # record 1's [Q] title was overwritten by record 2
```

The takeover is completely silent: `write_event_meta()` line 311-313 deliberately nulls the
other row's `observation_record` claim (the code path written for the legacy takeover sweep
absorbs this case), no counter is incremented, and nothing is logged. The sibling ingest
command already treats this exact input as a hazard —
`backfill_lco_observations.py:537-542` skips a request whose `id` is `None` — so the
projector is the one consumer of that data with no guard.

**Fix:** reject an unusable identity key before it can collide, and namespace the lookup:

```python
def event_fields_for(record, facility):
    if not (record.observation_id or '').strip():
        raise ValueError(f'record pk={record.pk} has no observation_id; cannot key an event')
    ...
```

and, for the facility collision, either include the facility in the key
(`{'url': event_url(...), 'telescope': ...}` is not enough — prefer keying on the companion
row's `observation_record` or a `LCO:`/`SOAR:`-prefixed url namespace) or add a
`UniqueConstraint` on `CalendarEvent.url` for non-blank urls via a FOMO migration so the
database refuses the collision instead of absorbing it.

### CR-02: A single duplicate-url CalendarEvent permanently breaks a record's projection, and `project_record()` does not catch it despite promising "Never raises"

**File:** `solsys_code/observation_projector.py:324-343`;
`solsys_code/calendar_utils.py:572` (`get_or_create(**lookup, defaults=fields)`)

**Issue:** `insert_or_create_calendar_event()` calls `CalendarEvent.objects.get_or_create(url=...)`
on a **non-unique** field. Two consequences, both reproduced:

1. Once two rows share the url, every projection of that record raises
   `MultipleObjectsReturned: get() returned more than one CalendarEvent -- it returned 2!`.
2. `project_record()` catches **only** `event_fields_for()` (lines 336-340). `facility_for()`
   (line 335, can raise `ImportError` from `get_service_class`),
   `insert_or_create_calendar_event()` and `write_event_meta()` (lines 341-342) are all
   outside the `try`, so the docstring's "Create/update/leave-unchanged the record's event.
   Never raises (TRIG-02)" is false. `receiver_on_record_save()` swallows the exception and
   logs at `warning`, so the record's calendar entry silently stops updating **forever** —
   no error surfaces to the operator, and the month view keeps showing the stale entry.

The duplicate is reachable through the UI, not just by a race: `event_form.html:50-58`
renders `form.url` as an editable field, and `tom_calendar.views.update_event` /
`create_event` carry **no authentication decorator at all** (installed tom_calendar
`views.py:187`, wired at `solsys_code/calendar_urls.py:18-19`), so any visitor can create a
second event carrying a projector-owned portal url. `get_or_create()` on a non-unique column
is additionally documented-racy: two concurrent saves of the same record (submission +
`updatestatus`) can both create.

There is no test for either case — every existing test starts from a clean, single-event
state.

**Fix:** make the lookup tolerant and move the writes inside the guard:

```python
def project_record(record) -> tuple[str, str]:
    try:
        facility = facility_for(record)
        fields, stage = event_fields_for(record, facility)
        event, action = insert_or_create_calendar_event({'url': event_url(record, facility)}, fields)
        write_event_meta(event, record)
    except Exception as exc:  # noqa: BLE001 -- a projector must never break the triggering save
        logger.warning('unprojectable observation_id=%r: %s', record.observation_id, type(exc).__name__)
        return 'unprojectable', type(exc).__name__
    return action, stage
```

plus, in `insert_or_create_calendar_event()`, replace `get_or_create` with an explicit
`filter(**lookup).order_by('pk').first()` + `create()` (matching the tolerance the
`start_time_tolerance` branch at line 564 already uses), and/or add the unique constraint
proposed in CR-01 so duplicates cannot exist in the first place.

## Warnings

### WR-01: The never-raise wrappers cannot protect the caller's save from a database error, and three CharFields are written unbounded

**File:** `solsys_code/observation_projector.py:487-491`, `449-461`, `286-295`

**Issue:** `receiver_on_record_save()` runs inline in the caller's transaction (TRIG-02 says so
explicitly) and catches `Exception`. Per Django's own documented rule, catching a *database*
error inside an `atomic` block leaves the transaction unusable — every subsequent query raises
`TransactionManagementError`. So for the one class of failure that actually threatens the
operator's save, the broad `except` converts a clear error into a confusing one later rather
than protecting anything.

That class of failure is reachable: `title` is truncated to 200 (line 210) but `telescope`,
`instrument` and `proposal` are written straight through (lines 291-293) into
`CharField(max_length=200)` columns. `instrument` is whatever
`parameters['c_N_instrument_type']`/`['instrument_type']` holds, and `proposal` is whatever
`parameters['proposal']` holds — both externally sourced. SQLite silently accepts over-length
values; PostgreSQL (the documented production target, CLAUDE.md "Database") raises `DataError`.

**Fix:** truncate the three fields the same way the title already is
(`token[:200]`, `instrument[:200]`, `proposal[:200]`), and wrap the projector's own work in
`transaction.atomic()` (a savepoint) inside the receiver so a rolled-back savepoint leaves the
caller's transaction usable:

```python
try:
    with transaction.atomic():
        action, stage = project_record(instance)
except Exception as exc:  # noqa: BLE001
    ...
```

### WR-02: `--dry-run` provably *can* disagree with a real run — the "structurally unable to disagree" claim is false whenever a site lookup fires

**File:** `solsys_code/observation_projector.py:367-375` (docstring),
`solsys_code/management/commands/project_observation_calendar.py:6-9` (module docstring),
`:186-189`

**Issue:** The dry-run path never calls `pre_fields_hook` (line 417 and the command's
`pre_fields_hook=None if dry_run else hook`), so the observed-telescope token is never
resolved in a dry run. A record whose *only* pending change is the coarse→observed token
(`'2m0'` → `'FTS'`) is therefore reported `unchanged` by `--dry-run` and `updated` by the real
run. The committed notebook shows the divergence directly:

```
--dry-run : LCO: created: 3, updated: 156, ..., site_lookups: 0
real sweep: LCO: created: 3, updated: 156, ..., site_lookups: 59
```

(the `updated` counts coincide here only because every event was changing anyway for the
one-time takeover). The docstrings assert the opposite — "This is what keeps a dry-run count
structurally unable to disagree with what a real run would do" and "a `--dry-run` count can
never disagree with what a real sweep would do".

**Fix:** either soften both docstrings and the runbook to state the one exception explicitly
("a dry run reports no site lookups and therefore under-reports `updated` for records whose
observed telescope is not yet resolved"), or make the dry run *predict* the lookup by counting
a would-be lookup into a `site_lookups` dry-run counter without performing it.

### WR-03: `observation_series_decoration()` publishes observation-group identity on an unauthenticated view with no visibility gate, unlike its sibling tag

**File:** `solsys_code/templatetags/calendar_display_extras.py:560-646`;
`src/templates/tom_calendar/partials/event_form.html:146-158`

**Issue:** `campaign_decoration()` — the tag the new one is explicitly modelled on — gates its
output on `run.is_publicly_visible` (line 536) precisely "to keep a pending-review run's
campaign name off the public calendar". The new tag has no equivalent gate: for any event with
an `observation_group` link it returns the group's name and a `tom_observations:detail` link,
and `tom_calendar.views.update_event` (which renders `event_form.html`) has no
`login_required`, while `AUTH_STRATEGY='READ_ONLY'` means `AuthStrategyMiddleware` does not
block anonymous requests either. The phase's own test proves the anonymous path:
`test_calendar_template.py::test_grouped_event_modal_shows_group_name_and_night_n_of_n` asserts
the group name is in the response body of a `self.client.get()` with no login.

The group name is not free text of unknown provenance — `backfill_lco_observations._group_name()`
builds it as `<LCO RequestGroup name> (<portal RequestGroup id>)`, so the modal now publishes an
internal portal identifier. T-34-13 in `34-03-PLAN.md` considered only "does the dict contain a
PII field name", never "should this be visible to an anonymous visitor at all".

**Fix:** mirror the sibling gate, e.g. return `None` when the companion row's `run` exists and
is not publicly visible, or gate the whole block in the template on
`{% if user.is_authenticated %}`. Whichever is chosen, record the decision next to
`campaign_decoration()`'s gate so the two tags' visibility rules stay legible side by side.

### WR-04: `_window_start_or_max()` catches a narrower exception set than the code path it guards, so the modal can still 500

**File:** `solsys_code/templatetags/calendar_display_extras.py:558-576`

**Issue:** The helper catches `(KeyError, ValueError)`, but `record_time_window()` calls
`datetime.fromisoformat(record.parameters['start'])`, which raises `TypeError` — not
`ValueError` — when the stored value is a JSON number, boolean or `null` rather than a string.
That `TypeError` escapes both the helper and `observation_series_decoration()` (whose docstring
says "Never raises"), and because it fires inside `list.sort()` it takes down the whole modal
response. The projector itself catches this case correctly (`event_fields_for()` is wrapped in
a bare `except Exception`), so the two modules disagree about what an unparsable window means.

**Fix:**

```python
    try:
        start, _ = record_time_window(record)
    except Exception:  # noqa: BLE001 -- a request-time decoration must never 500 the modal
        return datetime.max.replace(tzinfo=dt_timezone.utc)
```

and add a test with `parameters={'start': 12345, 'end': 12346}`.

### WR-05: The PROJ-05 prefetch widening targets the wrong view, joins a relation nothing reads, and its regression test is vacuous

**File:** `solsys_code/views.py:114-135`;
`solsys_code/tests/test_calendar_template.py::test_month_view_query_count_does_not_grow_with_second_grouped_event`

**Issue:** `observation_series_decoration()` is referenced **only** from
`event_form.html:148`, rendered by `tom_calendar.views.update_event` — a different view from
`fomo_render_calendar`. `calendar.html` never calls it. So:

* the comment at `views.py:118-124` ("the modal's series tag ... dereferences the companion
  row's own record and group per event — without this the month view pays a query per
  attributed event") describes something that cannot happen;
* `select_related('observation_record__target')` adds two LEFT JOINs per companion row for data
  no code reads — the tag uses `meta.observation_record_id` only, never the record or its
  target;
* the `assertEqual(multi_count, single_count)` test passes no matter what, because the view it
  measures never invokes the tag. T-34-15's stated mitigation ("Task 2 adds an
  `assertNumQueries` regression asserting the count does not grow per grouped event") is
  therefore not implemented, and the tag's real per-modal fan-out
  (`meta.observation_group.observation_records` + one `record_time_window()` parse per member)
  is unmeasured.

**Fix:** drop `observation_record__target` from the prefetch (keep `observation_group` only if
a month-cell consumer is actually added), correct the comment to say the modal view is where
the tag runs, and move the query-count assertion onto
`reverse('calendar:update-event', args=[event.id])` with a group of 2 vs. a group of 10.

### WR-06: `is_verified` is now write-only-`True`, leaving two dead template branches and a misleading model field

**File:** `solsys_code/observation_projector.py:314-321`;
`src/templates/tom_calendar/partials/calendar.html:247-250, 265-270`;
`solsys_code/models.py:36-38`

**Issue:** With the old sync command deleted, no code anywhere writes `is_verified=False`
(`campaign_views.py:1219` writes `True`; the reconciler never writes it; the projector writes
`True` unconditionally). The `{% if event.telescope_label_meta.is_verified == False %}`
branches in `calendar.html` — two near-duplicate `<div>` renderings plus the tooltip
"Telescope label is an estimate — could not be verified against the LCO API" — are now
unreachable dead markup, and `models.py`'s `verbose_name`
("Whether the telescope label was live-verified against the LCO API") now describes a flag that
is set to `True` without any API call ever being made.

Retiring `[UNVERIFIED]`/`is_verified=False` was an explicit decision (34-DISCUSSION-LOG.md,
D-06), so the *write* is intended — what was not carried through is the cleanup: the takeover
sweep has already overwritten the pre-existing `False` rows in the real developer database
(the notebook's first sweep was not run in a transaction), so the old signal is gone and the UI
branches can never fire again.

**Fix:** delete both dead branches and the tooltip from `calendar.html` in this phase (or file
them explicitly into Phase 37 with a code comment at each branch saying so), and update the
field's `verbose_name`/docstring to describe what `is_verified` now means.

### WR-07: The D-07 telescope rename moves the projector's token into `load_telescope_runs`' classical lookup vocabulary

**File:** `solsys_code/calendar_utils.py:54-69`;
`solsys_code/management/commands/load_telescope_runs.py:208-217`

**Issue:** `SITE_TELESCOPE_MAP` now emits `'FTS'`, `'FTN'` and `'SOAR'` as
`CalendarEvent.telescope` values for projector-owned events. `'FTS'` is simultaneously a key of
`telescope_runs.SITES` — the classical vocabulary `load_telescope_runs` writes — and that
command's find-or-create lookup is
`{'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': ...}` with a
±5-minute window and **no url restriction**. Before this rename the projector could never
produce a `telescope` value in that vocabulary (it wrote `'COJ-2m0'`/`'2m0'`), so the two
writers were structurally separated; now separation rests only on the two `instrument` strings
differing. If a classical schedule line ever names an instrument string equal to an LCO
`instrument_type`, `load_telescope_runs` will adopt and rewrite a projector-owned event's
title/description/target_list, and the next record save will rewrite it back — two writers
flapping on one row.

**Fix:** add `url=''` (or `url__exact=''`) to `load_telescope_runs`' lookup dict so the
classical writer can only ever match a blank-url event — the same namespace discipline the
reconciler and the projector both apply — and assert it in
`test_load_telescope_runs.py`.

### WR-08: `_cleared_group_members` is an unbounded, process-lifetime, non-thread-safe module global

**File:** `solsys_code/observation_projector.py:504, 532-546`

**Issue:** The `pre_clear` branch stores a list of every member pk under
`(sender, instance.pk)` and only the forward `post_clear` branch ever pops it. If anything
between the two signals raises (a database error during the through-table delete, a
`pre_clear` receiver further down the chain raising), the entry is never popped and leaks for
the life of the process. Two threads clearing the same group concurrently also overwrite each
other's captured list, so one clear can re-project the other's member set. Neither case is
tested.

**Fix:** scope the capture to the signal pair with a `try/finally` at the call site or key it
by a per-call token, and at minimum bound it — e.g. pop with a fallback and drop entries older
than the current request — or re-derive the members in `post_clear` from
`CalendarEventMeta.objects.filter(observation_group_id=instance.pk)` and remove the global
entirely.

### WR-09: The notebook's convergence assertion claims more than it checks

**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` (cell 10)

**Issue:**

```python
for facility_token in ('created: 0', 'updated: 0', 'site_lookups: 0'):
    assert facility_token in second_sweep_summary, \
        f'expected {facility_token!r} in every facility line of the second sweep'
```

`second_sweep_summary` is the single joined string containing both the LCO and the SOAR
segments, so a substring test passes when **either** facility matches. The SOAR segment is
all-zeros by construction in this database, which means this assertion would pass even if the
LCO sweep had not converged at all — precisely the failure it exists to catch. (The committed
output happens to be genuinely converged; the assertion just does not prove it.)

**Fix:** split the summary on `' | '` and assert per segment:

```python
for segment in second_sweep_summary.split(' | ')[1:]:
    for facility_token in ('created: 0', 'updated: 0', 'site_lookups: 0'):
        assert facility_token in segment, f'{facility_token!r} missing from {segment!r}'
```

## Info

### IN-01: `telescope_match_score()`'s documented resolution order is now stale

**File:** `solsys_code/campaign_attribution.py:305-310`

**Issue:** Step 2's worked example is `'FTS'`, but `_extract_lco_site_code()` now resolves
`'FTS'` at step 1 via `OBSERVED_TELESCOPE_SITE_CODES`, so `'FTS'` can never reach the
classical-alias branch. Only `Magellan-Clay`, `Magellan-Baade` and `NTT` still reach step 2.
The evidence string a classical FTS orphan produces also changed wording (from "classical site
alias" to "LCO site code 'coj'") — same score, same obscode, different operator-facing text.

**Fix:** replace the `'FTS'` example in the step-2 docstring with `'NTT'` and note the evidence
wording change.

### IN-02: The runbook's ring description omits the failure markers

**File:** `docs/runbooks/telescope_runs_calendar.rst` (status-legend section)

**Issue:** "The ring drawn around a month cell follows the same vocabulary: a Queued or an
Inconsistent record entry is ringed, a Scheduled or Observed entry is not." `[X] `, `[C] ` and
`[F] ` are also members of `_TERMINAL_PREFIXES` and are also ringed, which the sentence leaves
out.

**Fix:** add "…and an expired, cancelled or failed entry carries the terminal ring."

### IN-03: `[?]` is unreachable for a record that is both inconsistent and in a failure state

**File:** `solsys_code/observation_projector.py:207-210`

**Issue:** `title_for()` resolves the failure marker first, so an inconsistent record (half-set
schedule) whose status is `CANCELED`/`WINDOW_EXPIRED` renders `[C]`/`[X]` and the data problem
the `[?]` marker exists to surface is hidden. Precedence is documented ("A failure marker wins
over a stage marker"), so this is a note, not a defect — but the runbook's `[?]` row promises
"projected anyway so the data problem is visible on the calendar rather than only in a log",
which is not true for this combination.

**Fix:** either let `'inconsistent'` win over the failure marker, or add the caveat to the
runbook row.

### IN-04: Two hand-maintained copies of the same six counter keys

**File:** `solsys_code/observation_projector.py:350`;
`solsys_code/management/commands/project_observation_calendar.py:25`

**Issue:** `_SWEEP_COUNTER_KEYS` and `_COUNTER_KEYS` are byte-identical tuples with comments on
both sides explaining that they are deliberately separate copies. Adding a seventh counter
requires editing both, and a miss produces a `KeyError` in the summary f-string rather than a
missing column.

**Fix:** export the tuple from `observation_projector` and import it in the command (it is
already a public-ish module constant in all but the leading underscore).

### IN-05: `observed_enclosure` is written and never read; `select_related('target')` is fetched and never used

**File:** `solsys_code/management/commands/project_observation_calendar.py:90` (enclosure
write); `solsys_code/templatetags/calendar_display_extras.py:619`

**Issue:** `OBSERVED_SITE_PARAMETER_KEYS`' third key is stored on every successful lookup but
no consumer reads `observed_enclosure` anywhere in the codebase. Separately, the series tag's
`observation_records.select_related('target')` joins a relation the tag never dereferences
(only `member.pk` and `record_time_window(member)` are used).

**Fix:** keep the enclosure write if a Phase 35/36 consumer is planned and say so in a comment;
drop the unused `select_related('target')`.

### IN-06: Re-executing the demo notebook overwrites the SCHED-06 baseline it tells you to diff against

**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` (cells 15, 17)

**Issue:** Cell 15 unconditionally writes
`project_observation_calendar_demo.sched06-baseline.json`, while cell 17's instructions say to
re-execute the notebook and then diff "by eye against the JSON file this run wrote". The
re-execution replaces that file first, so the comparison only survives because the file is
committed and `git diff` recovers it — which the instructions never say.

**Fix:** either write the re-run snapshot to a second filename
(`…sched06-rerun.json`) or amend the instruction to "compare with `git diff` on the baseline
file, which this run has just overwritten".

---

_Reviewed: 2026-09-11T05:29:34Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
