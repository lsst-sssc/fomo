---
phase: 08-telescope-label-verification-sidecar
reviewed: 2026-06-24T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - solsys_code/migrations/0001_calendareventtelescopelabel.py
  - solsys_code/models.py
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
  - solsys_code/tests/test_load_telescope_runs.py
  - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
  - solsys_code/tests/test_calendar_template.py
  - src/templates/tom_calendar/partials/calendar.html
findings:
  critical: 0
  warning: 3
  info: 3
  total: 6
status: issues_found
---

# Phase 08: Code Review Report

**Reviewed:** 2026-06-24T00:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 8 adds a `CalendarEventTelescopeLabel` OneToOne sidecar model, writes it
from `sync_lco_observation_calendar.py` right after the existing `CalendarEvent`
upsert, and renders a dashed-border + tooltip visual cue in `calendar.html` for
fallback-labeled events. The migration, model, and write-side logic
(`update_or_create` keyed on `event`, `is_verified = not telescope_api_failed`)
are correct and match the documented "no row means verified by default"
contract — confirmed by `test_load_telescope_runs.py`'s
`test_display_01_no_sidecar_row_for_classically_scheduled_event` and the
`DoesNotExist`-on-no-row check in both the notebook and the test suite.

No critical/blocker-level defects were found. The main issue worth fixing is a
latent ordering bug in the all-day branch of `calendar.html`'s `{% if %}`
chain: the `[QUEUED] ` title-prefix check is evaluated before the
`is_verified == False` check, so any all-day event whose title happens to
start with `[QUEUED] ` would never get the dashed-border/tooltip treatment
even if its sidecar row says `is_verified=False`. This can't currently be
triggered by `sync_lco_observation_calendar.py` itself (banner-stage records
are hard-coded `telescope_api_failed=False`), but it is exactly the kind of
order-dependent, string-prefix-driven branching that breaks silently the next
time someone adds a queued+fallback code path — and the test suite's
`test_dashed_border_count_matches_fallback_event_count_only` doesn't exercise
this combination, so a regression here would go undetected by the existing
tests. A few smaller correctness/robustness gaps are also listed below.

## Warnings

### WR-01: All-day branch's `[QUEUED]` check masks the fallback dashed-border cue

**File:** `src/templates/tom_calendar/partials/calendar.html:158-165`
**Issue:** The all-day event rendering branch is:
```django
{% if event.title|slice:":9" == "[QUEUED] " %}
<div class="cal-event-all-day" style="background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);">
{% elif event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day" style="background-color: {{ event.color }}; border: 2px dashed rgba(0, 0, 0, 0.65);" title="...">
{% else %}
<div class="cal-event-all-day" style="background-color: {{ event.color }};">
{% endif %}
```
Because this is `if/elif/else` rather than two independent conditions, an
all-day event whose title starts with `[QUEUED] ` always takes the first
branch — even if it has a sidecar row with `is_verified=False`. The timed
branch a few lines below (174-182) checks `is_verified == False` first with
no competing condition, so the two branches are inconsistent with each other
for events that could plausibly carry both signals. Today
`_build_event_fields()` always sets `telescope_api_failed=False` for
banner-stage (`scheduled_start is None`, i.e. `[QUEUED]`) records, so this
can't fire yet from this codebase's only writer — but it's a textbook
"works by accident" coupling: the template depends on an invariant
(`[QUEUED]` implies `is_verified=True`) that lives in a completely different
module and is not enforced or asserted anywhere near the template. The next
feature that creates a queued+fallback-labeled event (or that adds a sidecar
row through some other code path) will silently lose the dashed-border cue
with no test failure, because the existing test
(`test_dashed_border_count_matches_fallback_event_count_only`) only exercises
all-day events that are either fully verified, fully fallback (non-queued),
or row-less — never the queued+fallback combination.
**Fix:** Make the verification check independent of the title-prefix check,
e.g. nest or reorder so `is_verified == False` is evaluated regardless of the
`[QUEUED]` prefix (or combine the conditions explicitly):
```django
{% if event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day" style="background-color: {% if event.title|slice:":9" == "[QUEUED] " %}rgba(0, 0, 0, 0.45){% else %}{{ event.color }}{% endif %}; border: 2px dashed rgba(0, 0, 0, 0.65);"
     title="Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0).">
{% elif event.title|slice:":9" == "[QUEUED] " %}
<div class="cal-event-all-day" style="background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);">
{% else %}
<div class="cal-event-all-day" style="background-color: {{ event.color }};">
{% endif %}
```
Add a test case for a `[QUEUED]`-titled event with an `is_verified=False`
sidecar row to lock in whichever behavior is intended.

### WR-02: `is_verified=True` row written for both successfully-resolved and never-attempted placed records — `is_verified` overloads two distinct meanings

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:472, 630-637`
**Issue:** `telescope_api_failed` is `True` only when a *placed* record's label
was a fallback (`record.scheduled_start is not None and label_was_fallback`).
For a banner-stage (`[QUEUED]`) record, `telescope_api_failed` is always
`False` because no API call is attempted at all — `is_verified=True` is then
written to the sidecar even though the label was never live-verified; it's
simply "not yet checked." This matches the documented design (`is_verified`
reflects "the outcome of the most recent sync run that included this
record", and a banner-stage record never attempts verification by design),
but it means a `[QUEUED]` event's sidecar row says `is_verified=True` for a
fundamentally different reason than a placed+successfully-resolved event's
row does (one never tried; the other tried and succeeded). Combined with
WR-01, a future reader of the sidecar table cannot distinguish "verified
against the live API" from "verification was never attempted" by querying
`is_verified` alone — both read `True`. This is a modeling gap, not a bug in
the current code path, but it increases the risk that whoever builds the next
stage on top of this sidecar (e.g. a "show estimate" banner on `[QUEUED]`
events, which today get a different background color but no tooltip) reaches
for `is_verified` and gets a misleading answer.
**Fix:** Consider documenting this explicitly in the model docstring (it's
partially done already) or, if a future consumer needs to distinguish
"verified" from "not yet attempted", add a third state (e.g.
`nullable BooleanField` with `None` = not attempted) rather than overloading
`True`. No code change required for this phase, but flag the ambiguity for
whoever extends this sidecar next.

### WR-03: `_resolve_placement_block` iterates the full block list even after finding `COMPLETED`, but only `break`s on `COMPLETED` — `PENDING` selection is order-dependent and silently picks the *last* `PENDING` block

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:189-196`
**Issue:**
```python
current_block = None
for block in blocks:
    if block.get('state') == 'COMPLETED':
        current_block = block
        break
    elif block.get('state') == 'PENDING':
        current_block = block
return current_block
```
If the API returns multiple blocks with no `COMPLETED` block, but more than
one `PENDING` block, this silently keeps overwriting `current_block` and
returns the *last* `PENDING` block encountered, not the first. This mirrors
"the same COMPLETED-first-else-PENDING block that `OCSFacility.get_observation_status()`
selects" per the docstring, but if the upstream selection logic in
`OCSFacility.get_observation_status()` actually selects the *first* `PENDING`
block (or a different tie-break rule) rather than the last, this function and
that one could silently disagree on multi-block placed records, undermining
the stated Pitfall-3 guarantee that "telescope resolution and timing always
come from the same block." This wasn't verified against the actual
`OCSFacility.get_observation_status()` implementation as part of this
review — flagging as a warning rather than a blocker because no test exercises
more than one `PENDING` block in the mocked response list, so the actual
behavior of `OCSFacility.get_observation_status()`'s tie-break vs. this
function's tie-break was not cross-checked here.
**Fix:** Add a test with two `PENDING` blocks (and confirm against
`OCSFacility.get_observation_status()`'s real selection order) to lock in
which `PENDING` block wins, or explicitly comment why "last PENDING wins" is
intentional and matches upstream.

## Info

### IN-01: Migration's generated timestamp predates `models.py`'s author-visible content with no makemigrations check enforced in this diff

**File:** `solsys_code/migrations/0001_calendareventtelescopelabel.py:1`
**Issue:** The migration header comment says "Generated by Django 5.2.14 on
2026-06-25 05:32" — one day after `currentDate` context (2026-06-24). This is
likely just system-clock skew in the dev environment and not a real defect,
but per the project's known `tom_jpl` makemigrations CI gotcha (any
model/Meta change needs `makemigrations` or `check_migrations` CI job fails),
worth a sanity check that `./manage.py makemigrations --check` passes for
`solsys_code` with this migration in place, since the OneToOneField target
`tom_calendar.calendarevent` lives in an external installed app whose own
migration state must already include `0005_calendarevent_instrument` for this
dependency to resolve. (Verified locally during review: this dependency does
exist in the installed `tom_calendar` package, so the migration itself is
sound — flagging only as a heads-up for CI.)
**Fix:** No action needed beyond confirming CI's migration-check job passes.

### IN-02: `CalendarEventTelescopeLabel.__str__` calls `self.event.title`, an extra query per row, only exercised in non-hot-path code

**File:** `solsys_code/models.py:24-25`
**Issue:** `__str__` dereferences `self.event.title`, which (if `event` isn't
already prefetched) issues an extra DB query per `__str__()` call. This is
idiomatic for admin/debug display and not a problem for `update_or_create`'s
write path (which never calls `__str__`), but the Django admin list view (if
ever registered for this model) would N+1 here. Not flagged as a Warning
since the model currently has no admin registration in scope of this phase.
**Fix:** If/when this model gets a `ModelAdmin`, use
`list_select_related = ('event',)` or similar to avoid N+1 in the admin list
view.

### IN-03: `_observations_block_response` test helper and the inline malformed-block construction in `test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped` duplicate the same shape

**File:** `solsys_code/tests/test_sync_lco_observation_calendar.py:1096-1112`
**Issue:** The test comment explains why the existing helper can't be reused
("always populates all four keys together") and builds the malformed
`MagicMock` inline instead — this is a reasonable, well-justified choice, not
a defect. Noting only that if more "missing individual key" test cases are
added later, a small helper taking `**overrides` (vs. the current fixed
four-keyword-arg helper) would avoid repeating this inline-mock pattern.
**Fix:** Optional refactor only if more malformed-block test cases are added;
no action required now.

---

_Reviewed: 2026-06-24T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
