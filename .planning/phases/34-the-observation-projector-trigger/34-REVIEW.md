---
phase: 34-the-observation-projector-trigger
reviewed: 2026-09-11T00:00:00Z
depth: deep
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
  warning: 10
  info: 5
  total: 16
status: issues_found
---

# Phase 34: Code Review Report

**Reviewed:** 2026-09-11
**Depth:** deep
**Files Reviewed:** 29
**Status:** issues_found

## Summary

This is a re-review after three prior review/fix cycles (`34-REVIEW.iter2.md`, `34-REVIEW.iter3.md`
and the matching `34-REVIEW-FIX*.md`). Verification of the previously-reported findings was done
against the current tree, not from the fix reports:

- **CR-01 (iter3, sweep counted the preview instead of the write)** — resolved. `project_queryset()`
  now takes `project_record()`'s `'unprojectable'` return over the preview's prediction
  (`observation_projector.py:534-548`) and the dry run detects the same duplicate-url condition
  without writing (`:502-518`). Covered by `test_project_observation_calendar.py:273-345`.
- **WR-02/WR-03 (savepoint not real)** — resolved. `project_record()`'s `except` now sits outside
  the `with transaction.atomic()` block (`observation_projector.py:365-374`), so `Atomic.__exit__`
  sees the exception in flight and rolls the savepoint back.
- **WR-01 (viewer gate on the series decoration)** — resolved. `observation_series_decoration()`
  checks `_viewer_is_authenticated(context)` unconditionally before the run-visibility gate
  (`calendar_display_extras.py:684-694`), and is covered end-to-end by real anonymous/authenticated
  client requests (`test_calendar_template.py:1192-1385`).
- **WR-04 (`--proposal` with only empty segments)** — resolved, fails closed with `CommandError`
  (`project_observation_calendar.py:181-187`).
- **WR-02 (`ogg` bridged as a site-level alias)** — resolved; `LCO_SITE_CODE_TO_OBSCODE` keeps only
  `coj`, and the label-keyed `OBSERVED_TELESCOPE_OBSCODES` handles the three single-telescope labels.

Gates were re-run and are clean: `pre-commit run ruff --all-files` and `ruff-format --all-files` both
pass, and `python manage.py test` over the whole `solsys_code` suite minus the two ASSIST-importing
modules (`test_views`, `test_ephem_utils`) is **1077 tests, OK**.

What the fix cycles did *not* close, and what this pass found new, is below. The one blocker is a
committed pre-executed demo notebook whose code cell now raises `AssertionError` if re-executed and
whose committed output is factually wrong — the last fix pass documented this in a markdown caveat
instead of fixing it, which CLAUDE.md's paired-docs rule explicitly treats as a must-have gap, not a
nice-to-have. The warnings cluster around invariants the projector *documents* as module-wide but
enforces only at some call sites, a documented-behaviour contradiction in the one-time site lookup,
and two new dead/unguarded lookup tables.

## Critical Issues

### CR-01: `campaign_lifecycle_demo.ipynb`'s D-07 cell raises `AssertionError` if re-executed, and its committed output is false

**File:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb:1044-1053` (cell 22), caveat at
`:989-1002`

**Issue:** The committed cell reads:

```python
from solsys_code.calendar_utils import OBSERVED_TELESCOPE_SITE_CODES
...
for telescope_label, expected_site_code in OBSERVED_TELESCOPE_SITE_CODES.items():
    resolved_site_code = _extract_lco_site_code(telescope_label)
    resolved_obscode = LCO_SITE_CODE_TO_OBSCODE.get(resolved_site_code)
    print(...)
    assert resolved_site_code == expected_site_code
```

Commit `36c7eae` removed `_extract_lco_site_code()`'s `OBSERVED_TELESCOPE_SITE_CODES` consultation,
so `_extract_lco_site_code('FTN')` now returns `None` (`campaign_attribution.py:274-278`:
`'FTN'.split('-', 1)[0].lower()` is `'ftn'`, which is not in `_LCO_SITE_CODES`). The loop therefore
raises `AssertionError` on its **first** iteration, and the `telescope_match_score()` call below it —
the cell's actual point — never runs. The committed output block still shows:

```
FTN   -> site code 'ogg' -> obscode 'F65'
FTS   -> site code 'coj' -> obscode 'E10'
SOAR  -> site code 'sor' -> obscode 'I33'
FTN match level: 1.0 (orphan LCO site code 'ogg' resolves to obscode F65, ...)
```

Every one of those four lines is now wrong: three cannot be produced at all, and the fourth's evidence
wording changed to the label-keyed form. This notebook is wired into the Sphinx toctree and ships on
ReadTheDocs, so a reader is shown executed output that the code provably cannot produce. CLAUDE.md's
paired-docs rule is explicit that a stale paired notebook is "a must-have gap, not a nice-to-have",
and that the executor must "add or update cells/prose exercising the new behavior with **real
executed output**". A prose caveat saying "this would crash" is not that.

Secondary defect in the same cell: it imports the module-private `_extract_lco_site_code` across
module boundaries — the exact anti-pattern `calendar_utils.update_calendar_event_key_and_fields()`'s
own docstring (`calendar_utils.py:585-591`) exists to prevent.

**Fix:** Rewrite the cell to demonstrate the bridge that actually exists, drop the private import, and
re-execute the notebook:

```python
from solsys_code.campaign_attribution import (
    OBSERVED_TELESCOPE_OBSCODES,
    TELESCOPE_MATCH_SITE,
    telescope_match_score,
)

for telescope_label, expected_obscode in OBSERVED_TELESCOPE_OBSCODES.items():
    print(f'{telescope_label:<5} -> obscode {expected_obscode!r} (label-keyed bridge)')
assert set(OBSERVED_TELESCOPE_OBSCODES) == {'FTN', 'FTS', 'SOAR'}
```

then

```console
jupyter nbconvert --to notebook --execute --inplace \
  docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
```

If re-execution genuinely cannot be done in this pass (the notebook touches the live LCO portal and
the developer database), the cell must at minimum be reduced to something that cannot crash — an
unexecutable markdown listing, or a loop with the assertion removed — rather than left as committed
code that is known to raise.

## Warnings

### WR-01: `project_record()`/`project_queryset()` do not enforce `PROJECTED_FACILITIES`, so a mis-scoped queryset can adopt and overwrite a classical blank-url event

**File:** `solsys_code/observation_projector.py:338-374` (`project_record`), `:390-566`
(`project_queryset`), invariant claimed at `:11-15`

**Issue:** The module docstring states ownership rule 1 as holding "across every function in this
module": it owns only events "whose `url` is a facility observation URL ... never a `RUN:` reconciler
event, a `GEM:` Gemini echo event, or a **blank-url classical event**." Both signal receivers enforce
this (`:591-592`, `:671-672`, `:724-725`), but `project_record()` and `project_queryset()` do not —
neither checks `record.facility in PROJECTED_FACILITIES`, and neither asserts that `event_url()`
returned a non-blank string.

`GEMFacility.get_observation_url()` returns `''` unconditionally
(`tom_observations/facilities/gemini.py:490-492`). A caller handing `project_queryset()` an
unfiltered `ObservationRecord` queryset therefore reaches
`insert_or_create_calendar_event({'url': ''}, fields)` → `CalendarEvent.objects.get_or_create(url='')`,
which **adopts the first existing blank-url `CalendarEvent`** — i.e. a `load_telescope_runs`-owned
classical night — and rewrites its title, description, start/end, telescope, instrument and
target_list, then stamps a `CalendarEventMeta` claiming it. With more than one blank-url event
present it instead raises `MultipleObjectsReturned` and is counted `unprojectable`.

Today this is latent, not live: the only production caller is the command, which filters
(`project_observation_calendar.py:176-178`), and a real Gemini record happens to fail earlier at
`extract_instrument()` because Gemini parameters carry no `instrument_type` key. Both of those are
incidental, not designed — the invariant the docstring asserts is simply not enforced where the write
happens. `TestNamespaceIsolation` in both test modules only ever feeds LCO records, so nothing pins
this.

**Fix:** Enforce the invariant at the write, not at the callers:

```python
def event_url(record: ObservationRecord, facility: Any) -> str:
    if record.facility not in PROJECTED_FACILITIES:
        raise ValueError(f'{record.facility!r} is not a projected facility; refusing to key an event')
    url = facility.get_observation_url(record.observation_id)
    if not (url or '').strip():
        raise ValueError(f'record pk={record.pk} resolved a blank observation url; refusing to key an event')
    return url
```

Both raises land in `project_record()`'s / `project_queryset()`'s existing catch and are reported
`unprojectable`. Add a regression test feeding a `GEM` record plus one pre-existing blank-url event to
`project_queryset()` and asserting the blank-url event is byte-identical afterwards.

### WR-02: the `pre_delete` receiver destroys campaign attribution and its human-confirmation audit stamps, contradicting `models.py`'s own documented `SET_NULL` rationale

**File:** `solsys_code/observation_projector.py:705-736`, contradicted comment at
`solsys_code/models.py:62-67`

**Issue:** `models.py:62-67` (a comment this phase did edit the surrounding docstring of) states the
design intent explicitly:

> Both are `SET_NULL`: deleting the record or group clears only the link, so this row's attribution
> and audit history (`run`, `is_verified`, `confirmed_by`, `confirmed_at`) and the CalendarEvent
> itself all survive.

Phase 34's new `receiver_on_record_delete` makes that false for every projector-owned event: it
deletes `meta.event` (`:730`), and `CalendarEventMeta.event` is `on_delete=CASCADE`
(`models.py:39-45`), so the companion row — including `run`, `confirmed_by` and `confirmed_at`,
which the module docstring at `observation_projector.py:16-19` says "must survive a projection
untouched" — is destroyed with it. Deleting one `ObservationRecord` silently discards a staff
member's confirmed campaign attribution and the audit trail of who confirmed it and when.

`test_observation_projector_signals.py:203-211` asserts the companion row *is* deleted, but no test
constructs an attributed, staff-confirmed event first, so the loss is unpinned in either direction.

**Fix:** Decide the rule once and make code and docs agree. Either (a) keep the delete but null the
link instead of cascading:

```python
if meta.event.url == event_url(instance, facility):
    if meta.run_id is None:
        meta.event.delete()
    else:
        # attribution survives: drop only the projector's own claim on the event
        CalendarEventMeta.objects.filter(pk=meta.pk).update(observation_record=None, observation_group=None)
```

or (b) keep the cascade and correct `models.py:62-67` to say that deleting a *projected* record also
deletes its event and companion row. Add a test that deletes a record whose event carries
`run`/`confirmed_by`/`confirmed_at` and asserts the chosen outcome.

### WR-03: `resolve_observed_site()`'s "at most once per record, ever" is contradicted three lines later, and the runbook repeats the false claim

**File:** `solsys_code/management/commands/project_observation_calendar.py:47-56`,
`docs/runbooks/telescope_runs_calendar.rst:152`

**Issue:** The docstring says:

> Calls the portal-block resolver at most once per record, ever

and then, in the next paragraph:

> both leave the coarse aperture token in place, store nothing, and **are retried on the next sweep**

Only a *successful* lookup writes `observed_site` (`:93-103`); the failure path stores nothing, so
`record.parameters.get(site_key)` stays falsy and the lookup fires again on every subsequent sweep.
The runbook repeats the wrong half at line 152 ("a single live portal call per record, ever") while
lines 183 and 1155 state the retry behaviour correctly — the same document contradicts itself.

Operationally this matters once Phase 36 runs the sweep from cron: `resolve_placement_block()` uses a
10-second timeout with no retry/backoff and no negative cache (`calendar_utils.py:99`, `:316-328`), so
a corpus with M permanently-unresolvable completed records (archived requests returning 404, a portal
outage) adds up to 10 × M seconds of blocking HTTP to **every** scheduled sweep, forever, with only a
`site_lookup_failed` counter to show for it.

**Fix:** Correct both docs, and bound the retry. Minimal doc fix:

```
Calls the portal-block resolver at most once per record per sweep, and never again once a
lookup has succeeded: a record whose ``parameters`` already carries ``observed_site`` is never
looked up again; a failed lookup stores nothing and is retried on the next sweep.
```

Minimal behaviour fix (pick one): record a failure stamp in `parameters` (e.g.
`observed_site_lookup_failed_at`) and skip a record whose last failure is younger than N days, or add
a `--skip-site-lookup` flag the cron can pass.

### WR-04: `'[?]'` doubles as both the "inconsistent record" marker and the unknown-stage fallback, and the calendar legend authoritatively mislabels the latter

**File:** `solsys_code/observation_projector.py:94-99` and `:208-211`,
`solsys_code/templatetags/calendar_display_extras.py:136-143`

**Issue:** `_STAGE_MARKER` has no `'inconsistent'` entry; `'[?]'` is produced only by
`_STAGE_MARKER.get(stage, '[?]')` falling through. The calendar legend
(`_OBSERVATION_STATUS_LEGEND`) then tells every calendar visitor, authoritatively, that `[?]` means
**"Inconsistent record"**. Any future stage string added to `stage_for()` and forgotten in
`_STAGE_MARKER` will therefore be rendered on the operator-facing calendar with a specific and wrong
diagnosis ("this record has a half-set schedule") instead of "unknown". A silent mislabel is worse
than a visible unknown, and nothing in the test suite would catch it — the only `[?]` test drives the
genuinely-inconsistent path.

**Fix:** Make the intended mapping explicit and give the fallback its own symbol:

```python
_STAGE_MARKER = {
    'queued': '[Q]',
    'placed': '[S]',
    'observed': '[O]',
    'completed-no-block': '[O]',
    'inconsistent': '[?]',
}
_UNKNOWN_STAGE_MARKER = '[!]'
...
marker = _STAGE_MARKER.get(stage, _UNKNOWN_STAGE_MARKER)
```

Add `[!]` to `_OBSERVATION_STATUS_LEGEND` and `_TERMINAL_PREFIXES` (as `'[!] '`), plus a test asserting
every value of `_STAGE_MARKER` and the unknown marker appear in the legend.

### WR-05: `calendar_utils.OBSERVED_TELESCOPE_SITE_CODES` is a dead public export with no production consumer

**File:** `solsys_code/calendar_utils.py:71-84`

**Issue:** The constant was introduced in this phase as the bridge `_extract_lco_site_code()` would
consult; commit `36c7eae` then removed that consultation as dead code. A repo-wide search finds no
remaining production reader — the only references are its own tests
(`test_calendar_utils.py:16,326,337-339`), a pointer comment in `campaign_attribution.py:266`, and the
crashing notebook cell in CR-01. Its own comment already concedes this ("this table's remaining
consumers are its own tests below and any caller that genuinely needs the classical site code").
Shipping a public constant whose only exercise is a test asserting it equals itself is exactly the
"unused export" class of defect, and it keeps a third label→site table alive alongside
`SITE_TELESCOPE_MAP` and `OBSERVED_TELESCOPE_OBSCODES`.

**Fix:** Delete `OBSERVED_TELESCOPE_SITE_CODES`, its test block, and the pointer comment at
`campaign_attribution.py:263-267`; fold the "every non-`SITECODE-CLASS` label must be known" check in
`test_calendar_utils.py:326-333` onto a literal `frozenset({'FTN', 'FTS', 'SOAR'})` or onto
`campaign_attribution.OBSERVED_TELESCOPE_OBSCODES`. If it is being kept as a deliberate future
extension point, say so in the comment and drop the "remaining consumers are its own tests" wording,
which reads as an admission rather than a rationale.

### WR-06: nothing guards `OBSERVED_TELESCOPE_OBSCODES` against drift from `SITE_TELESCOPE_MAP`

**File:** `solsys_code/campaign_attribution.py:277-289`, `solsys_code/calendar_utils.py:54-69`

**Issue:** `test_calendar_utils.py:326-333` protects `SITE_TELESCOPE_MAP` against unknown labels: any
value that is neither `^[A-Z]{3}-(0m4|1m0|2m0|4m0)$` nor a key of `OBSERVED_TELESCOPE_SITE_CODES` fails
the test. No equivalent guard exists for `OBSERVED_TELESCOPE_OBSCODES`, which lives in a different
module. Adding a fourth single-telescope site to `SITE_TELESCOPE_MAP` (e.g. a new 2m0 named `'XYZ'`)
plus its `OBSERVED_TELESCOPE_SITE_CODES` entry would keep every test green while silently degrading
that telescope's attribution from `TELESCOPE_MATCH_SITE` to aperture-only, because
`telescope_match_score()`'s step 1 would find no entry and step 2's site-keyed table has no `'xyz'`
either. The failure is invisible: a lower score, not an error.

**Fix:** Add a cross-table test in `test_campaign_attribution.py`:

```python
def test_every_non_sitecode_telescope_label_has_an_obscode_bridge(self):
    label_pattern = re.compile(r'^[A-Z]{3}-(0m4|1m0|2m0|4m0)$')
    unprefixed = {label for label in SITE_TELESCOPE_MAP.values() if not label_pattern.match(label)}
    self.assertEqual(unprefixed, set(OBSERVED_TELESCOPE_OBSCODES))
```

### WR-07: `load_telescope_runs.py` changed behaviour but its paired notebook was not updated and still cites the deleted sync command

**File:** `solsys_code/management/commands/load_telescope_runs.py:206-226`,
`docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb:539`

**Issue:** This phase added `'url': ''` to `load_telescope_runs`' find-or-create lookup key — a new
lookup parameter that changes which existing events the command can match. CLAUDE.md's paired-docs
rule names `load_telescope_runs.py -> load_telescope_runs_demo.ipynb` and fires on exactly this
("new parameters ... not pure refactors or typo fixes"), requiring the notebook in `files_modified`
up front. The notebook is not in this phase's diff at all. It is also independently stale: line 539
still points a reader at `sync_lco_observation_calendar.py`'s `_FAILURE_PREFIX_BY_STATUS`, a file this
phase deleted (the source comment at `load_telescope_runs.py:24-28` was updated; the notebook citing
it was not).

The same stale citation survives in `src/templates/tom_calendar/partials/event_form.html:109` — a file
this phase *did* edit.

**Fix:** Update `load_telescope_runs_demo.ipynb`: replace the `sync_lco_observation_calendar.py`
citation with `observation_projector._FAILURE_MARKER_BY_STATUS`, add a cell (or prose + executed
output) showing that the blank-url lookup key keeps a classical night from adopting a projector-owned
`FTS`/`FTN`/`SOAR` event, and re-execute with
`jupyter nbconvert --to notebook --execute --inplace`. Fix the `event_form.html:109` reference in the
same pass.

### WR-08: the request-window parse is duplicated in `event_fields_for()` instead of shared, and both copies silently reinterpret a non-UTC ISO offset

**File:** `solsys_code/observation_projector.py:276-284`, duplicating
`solsys_code/calendar_utils.py:482-486`

**Issue:** The `stage == 'inconsistent'` branch re-implements `record_time_window()`'s parse
byte-for-byte rather than calling a shared helper, so the two copies must now be kept in sync by hand
— and one of them already has a fix the other does not know about (`_window_start_or_max()` at
`calendar_display_extras.py:566-590` had to add a bare `except Exception` precisely because this parse
raises `TypeError`, not `ValueError`, on a JSON number).

Both copies use `datetime.fromisoformat(s).replace(tzinfo=dt_timezone.utc)`. `replace()` **overwrites**
an offset rather than converting it: `'2026-09-01T00:00:00+10:00'` becomes `00:00 UTC`, a silent
10-hour shift of the event on the calendar, with no error and no log line. The comment at
`calendar_utils.py:483-484` asserts these strings are naive, but the strings come from
`record.parameters`, which the LCO portal and `backfill_lco_observations` populate — not a value this
code controls.

**Fix:** Extract one parser and use it from both places:

```python
def parse_request_window_bound(raw: str) -> datetime:
    """Parse a parameters['start']/['end'] ISO string as UTC, honouring an explicit offset."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt_timezone.utc)
    return parsed.astimezone(dt_timezone.utc)
```

Call it from `record_time_window()` and from `event_fields_for()`'s inconsistent branch; add a test
with an offset-carrying `parameters['start']`.

### WR-09: the projector publishes every target's name to the unauthenticated calendar, while the same phase gates the observation-group name behind login

**File:** `solsys_code/observation_projector.py:194-211` (`title_for`),
`src/templates/tom_calendar/partials/calendar.html:244-290`

**Issue:** The retired sync command's title was `f'{prefix} {telescope} {instrument}'` — no target
name. The projector's is `f'{marker} {token} {target_name}'`, and `record.target.name` is written into
`CalendarEvent.title`, which `calendar.html` renders in every month cell. `/calendar/` carries no
`login_required` (`solsys_code/calendar_urls.py:17`), so this phase newly publishes the name of every
LCO/SOAR target under observation to anonymous visitors.

`CalendarEvent` has no permission filtering of any kind, so django-guardian object-level target
permissions (`TARGET_PERMISSIONS_ONLY = True`, `settings.py:361`) are bypassed for any target whose
permission is not the `OPEN` default. This sits directly beside the opposite decision made in the same
phase: `observation_series_decoration()` gates the observation-group name behind authentication
*and* `run.is_publicly_visible` (`calendar_display_extras.py:684-696`) on the grounds that it is "an
internal portal RequestGroup identifier that must not be published". The two rules were not reconciled,
and no `34-*` artifact records the target-name decision as deliberate.

**Fix:** Make the choice explicit. Either document in `title_for()`'s docstring and the runbook that
target names are intentionally public under `AUTH_STRATEGY = 'READ_ONLY'`, or render the title through
a gate the way the series block is:

```python
# title_for(): keep the name out of the stored title
return f'{marker} {token}'[:200]
```

and move the target name into `target_list_block.html`/a `takes_context=True` tag that applies the
same `_viewer_is_authenticated()` rule. Whichever is chosen, add a test asserting the anonymous
month view does or does not contain a target's name, so the decision is pinned.

### WR-10: a duplicate-url `CalendarEvent` wedges a record's projection permanently, and any anonymous visitor can create one

**File:** `solsys_code/observation_projector.py:338-374` (docstring at `:344-349` acknowledges the
path), root cause in `solsys_code/calendar_urls.py:17-22` (not in this phase's diff)

**Issue:** `project_record()`'s own docstring names the failure mode: a duplicate-url row makes
`get_or_create()` raise `MultipleObjectsReturned`, "reachable through the unauthenticated event form".
`CalendarEvent.url` has no unique constraint, and FOMO's own URL conf registers
`create/`, `update/<id>/` and `delete/<id>/` with **no authentication decorator** — upstream
`tom_calendar.views.delete_event` does not even check `request.method`, so a bare GET to
`/calendar/delete/<id>/` destroys a row.

The consequence for Phase 34 specifically: one anonymous POST to `/calendar/create/` with `url` set to
a record's LCO request URL permanently pins that record `unprojectable` — every later save and every
sweep logs a warning and writes nothing, and the projector has no disambiguation or recovery path. The
event an operator then sees on the calendar is frozen at whatever state it had when the duplicate
appeared, with no visible marker that it has stopped tracking.

This is pre-existing (the URL conf predates this phase and is out of the review's file scope), but
Phase 34 is what makes the calendar an automatically-maintained projection of observation state, so
the exposure is newly load-bearing.

**Fix:** Two independent mitigations, both cheap:

1. Guard the destructive endpoints in FOMO's own conf:
   ```python
   from django.contrib.auth.decorators import login_required
   path('create/', login_required(create_event), name='create-event'),
   path('update/<int:event_id>/', login_required(update_event), name='update-event'),
   path('delete/<int:event_id>/', login_required(require_POST(delete_event)), name='delete-event'),
   ```
2. Give the projector a recovery path instead of a permanent wedge — on `MultipleObjectsReturned`,
   adopt the lowest-pk event and log the duplicates' pks so an operator can act:
   ```python
   except CalendarEvent.MultipleObjectsReturned:
       logger.warning('duplicate url for observation_id=%r: pks=%s', record.observation_id,
                      list(CalendarEvent.objects.filter(url=url).values_list('pk', flat=True)))
   ```

## Info

### IN-01: stale `sync_lco_observation_calendar` reference in a template this phase edited

**File:** `src/templates/tom_calendar/partials/event_form.html:109`
**Issue:** The WR-03 comment block still names `raw sync_lco_observation_calendar/...` output as a
source of unlinked events; that command was deleted in this phase.
**Fix:** Replace with "raw observation-projector / `sync_gemini_observation_calendar` /
`load_telescope_runs` output".

### IN-02: `_extract_lco_site_code()` silently narrowed its whitespace tolerance

**File:** `solsys_code/campaign_attribution.py:276-278`
**Issue:** The strip/split order was inverted (`split(...).strip()` → `strip().split(...)`), so an
internal space such as `'coj -2m0'` now yields `'coj '` and resolves to `None` where it previously
resolved to `'coj'`. Almost certainly harmless for real data, but it is an undocumented behaviour
change riding along with a comment-only commit.
**Fix:** `candidate = telescope_code.split('-', 1)[0].strip().lower()` restores the old tolerance, or
note the narrowing in the docstring if it was deliberate.

### IN-03: status legend reuses the telescope-legend CSS class

**File:** `src/templates/tom_calendar/partials/calendar.html:329-336`
**Issue:** The new marker legend entries are rendered with `class="cal-legend-telescope mr-3"`, a class
whose name now describes neither what it wraps nor why. A later restyle of the telescope legend will
silently restyle the status legend too.
**Fix:** Add `cal-legend-status` alongside it (or instead of it) and give it its own rule.

### IN-04: `test_telescope_01_verified_dict_covers_all_sites` does not verify the inverse mapping it exists to protect

**File:** `solsys_code/tests/test_calendar_utils.py:337-340`
**Issue:** The loop only asserts each `OBSERVED_TELESCOPE_SITE_CODES` value is *some* key of
`SITE_TELESCOPE_MAP`. Writing `'FTN': 'coj'` would pass. The one property worth pinning — that the
table really is `SITE_TELESCOPE_MAP`'s inverse — is untested.
**Fix:** `self.assertIn(label, {v for (site_code, _ap), v in SITE_TELESCOPE_MAP.items() if site_code == site})`
(moot if WR-05 deletes the table).

### IN-05: two `is_verified == False` template branches are retained as documented dead code

**File:** `src/templates/tom_calendar/partials/calendar.html:247-258`, `:276`,
`solsys_code/models.py:21-30`
**Issue:** No current writer sets `is_verified=False`; the branches are unreachable from production
writes and are kept only for historical/admin rows and test fixtures. This was reviewed and
consciously deferred in a prior fix pass — recorded here only so it is not re-discovered as new.
**Fix:** None this phase. File the removal (branches, field, `verbose_name`, migration) into the phase
that owns the status-vocabulary cleanup (Phase 37).

---

_Reviewed: 2026-09-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
