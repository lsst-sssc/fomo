---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
reviewed: 2026-09-19T00:00:00Z
depth: deep
files_reviewed: 37
files_reviewed_list:
  - solsys_code/status_vocabulary.py
  - solsys_code/campaign_tally.py
  - solsys_code/proposal_allocation.py
  - solsys_code/campaign_gap.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_views.py
  - solsys_code/observation_projector.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/allocation_projector.py
  - solsys_code/calendar_utils.py
  - solsys_code/models.py
  - solsys_code/admin.py
  - solsys_code/unattended.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/migrations/0023_proposal_time_allocation_and_campaignrun_proposal_code.py
  - solsys_code/tests/test_status_vocabulary.py
  - solsys_code/tests/test_campaign_tally.py
  - solsys_code/tests/test_proposal_allocation.py
  - solsys_code/tests/test_campaign_gap.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_calendar_display_extras.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_load_telescope_runs.py
  - solsys_code/tests/test_unattended.py
  - solsys_code/tests/test_admin.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_views.py
  - solsys_code/tests/test_write_and_reconcile.py
  - src/templates/campaigns/campaign_list.html
  - src/templates/campaigns/campaignrun_gap_analysis.html
  - src/templates/campaigns/campaignrun_table.html
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/event_form.html
  - docs/runbooks/telescope_runs_calendar.rst
findings:
  critical: 3
  warning: 12
  info: 7
  total: 22
status: issues_found
---

# Phase 37: Code Review Report

**Reviewed:** 2026-09-19
**Depth:** deep
**Files Reviewed:** 37
**Status:** issues_found

## Summary

The vocabulary consolidation (37-01) is largely real — the three parallel prefix maps are
gone and `status_vocabulary.py` is genuinely the single definition — but two bare marker
literals survived in `observation_projector.py` and a third in `calendar.html`, i.e. the
exact drift disease the phase set out to cure (WR-09).

The serious problems are in the tally cache. Its freshness stamp is
`Max(ObservationRecord.modified)` over the run's *links*, which is blind to the two events
that actually change the numbers it reports: creating or deleting a `CampaignRunObservation`
(no timestamp on that model, no touch of the record), and any change to
`CampaignRun.run_status` or to the set of `ALLOC:` calendar events. The result is a public
page showing wrong group/record/night counts for up to an hour (CR-01), and — worse — the
campaign table's unused count and the calendar's `[U]` marker visibly disagreeing for up to
an hour, which is precisely what D-15 said must be impossible by construction (CR-02). Both
contradict the module docstrings and the runbook text shipped in the same phase.

Third, the public tally path calls `observation_projector.facility_for()` with no guard;
that helper raises `ImportError` for a facility name absent from `TOM_FACILITY_CLASSES`,
which turns a stale facility name on one linked record into a 500 on the anonymous campaign
list, the campaign run table and the calendar pop-up — on code paths whose own docstrings
promise "never raises" (CR-03).

Beyond that: a new always-broken `Progress` column leaked into three staff approval-queue
tables (WR-01), several `.only()` field restrictions omit fields their own callees read and
so reintroduce per-row queries (WR-02, WR-03), the table computes tallies for the whole
filtered queryset rather than the rendered page (WR-04), the public campaign list loops
roll-ups over an unpaginated campaign list (WR-05), and the new portal fetch interpolates an
untrusted proposal code into a credentialed URL path without quoting (WR-07).

Credential handling in `proposal_allocation.py` is otherwise good: the exception set matches
`resolve_placement_block()`, the caught exception is never stringified, and `raise ... from
None` suppresses the chained context that would have carried the response body.

## Critical Issues

### CR-01: Tally cache key is blind to link creation/deletion — public counts stay wrong for up to an hour

**File:** `solsys_code/campaign_tally.py:70-90`, `solsys_code/campaign_tally.py:93-132`, `solsys_code/campaign_tally.py:232-256`, `solsys_code/campaign_tally.py:259-290`

**Issue:** The freshness segment of the cache key is `records_version =
Max('observation_record__modified')` over the run's `CampaignRunObservation` rows.
`CampaignRunObservation` (`solsys_code/models.py:582-643`) carries **no** `created`/`modified`
column, and creating a link does not save the `ObservationRecord` (the only receiver wired on
that signal is `allocation_projector.receiver_on_run_observation_save`, `solsys_code/apps.py:64-75`,
which re-projects calendar events, not the record). Therefore:

- Staff confirms an attribution (creates a link) to a record whose `modified` is older than
  the run's current max → `records_version` is unchanged → the cached tally is returned with
  the **old** `groups`/`records`/`nights_*` values for up to `TALLY_CACHE_TTL_SECONDS`
  (3600 s).
- Staff removes a link to a non-newest record → `records_version` is unchanged → same stale
  result, now over-counting.

This directly contradicts the module's own contract at `campaign_tally.py:42-47` ("It is NOT
what makes the tally see a record-driven change ... that happens immediately, with no TTL
wait") and the runbook text shipped in this phase
(`docs/runbooks/telescope_runs_calendar.rst:1999-2006`). `test_campaign_tally.py:354`
(`test_saving_a_linked_record_is_reflected_with_no_clock_advance`) only exercises saving an
*already-linked* record, so the hole is untested.

**Fix:** Fold the link set itself into the key, not just the record timestamps. The cheapest
correct form keeps `link_counts_for_runs()` at two queries:

```python
# campaign_tally.py
record_rows = (
    CampaignRunObservation.objects.filter(run_id__in=run_pks)
    .values('run_id')
    .annotate(
        records=Count('observation_record', distinct=True),
        records_version=Max('observation_record__modified'),
        link_version=Max('observation_record_id'),   # changes on every new link
    )
)
...

def build_tally_cache_key(run_pk, records_version, records_count=0, link_version=None):
    version_segment = records_version.isoformat() if records_version is not None else _NO_RECORDS_VERSION_TOKEN
    return f'campaign_tally:{run_pk}:{version_segment}:{records_count}:{link_version or 0}'
```

`records_count` alone already catches both create and delete; `link_version` additionally
catches a delete-then-create pair that leaves the count unchanged. Alternatively (cleaner,
but needs a migration) add `created`/`modified` to `CampaignRunObservation` and take
`Max('modified')` from the link table. Add a regression test that creates a
`CampaignRunObservation` for an untouched pre-existing record and asserts the tally moves on
the next call.

---

### CR-02: The calendar's `[U]` marker and the table's unused count disagree for up to an hour — D-15 is not satisfied by construction

**File:** `solsys_code/campaign_tally.py:323-342`, `solsys_code/campaign_tally.py:345-371`, `solsys_code/templatetags/calendar_display_extras.py:625-679`

**Issue:** D-15 requires the table and the calendar to agree *by construction*. They share
the predicate `is_unused_allocation_night()`, but not the freshness of its inputs:

- `unused_night_decoration()` evaluates the rule **live**, per event, on every calendar
  render (`calendar_display_extras.py:674`).
- `tally_segments()`'s unused figure comes from `unused_nights_for_run()` **through the TTL
  cache**, whose key (`build_tally_cache_key`) contains only `run_pk` and the linked-record
  timestamp.

Neither `CampaignRun.run_status` nor the set of `ALLOC:` `CalendarEvent` rows is in the key.
So:

1. Staff clicks "Mark Cancelled". The reconciler re-titles the nights to `[C]` and the
   calendar drops every `[U]` immediately (`is_unused_allocation_night` returns `False` at
   `campaign_tally.py:318-319`). The campaign table keeps serving the cached non-zero
   "Unused awarded night" count for up to an hour.
2. An allocation night elapses past its projected sunrise. The calendar paints `[U]` at once;
   the table's count lags by up to an hour.
3. A night is retired (its `ALLOC:` event deleted by the Phase 35 handoff when a placed
   record claims it). The calendar stops showing `[U]`; the table still counts it.

Case 1 is the damaging one: a cancelled run is publicly advertised as having wasted awarded
nights after staff explicitly said otherwise, and the two surfaces contradict each other on
screen. The same key gap makes `campaign_rollup()`'s summary strip disagree with the sum of
the rows rendered directly beneath it (`campaign_tally.py:443-520`,
`src/templates/campaigns/campaignrun_table.html:76-87`).

**Fix:** Either make the cache key cover every input the tally reads, or stop caching the
unused half. The smaller change is the latter — `unused_nights_for_run()` is one indexed
`CalendarEvent` query:

```python
def tallies_for_runs(runs) -> dict[int, dict[str, Any]]:
    ...
        cached = cache.get(key)
        if cached is not None:
            tally = dict(cached)
            _apply_unused_fields(tally, run)   # always live: matches the calendar exactly
            result[run.pk] = tally
            continue
        ...
        tally = _combine_tally(counts, nights)
        cache.set(key, tally, timeout=TALLY_CACHE_TTL_SECONDS)   # cache WITHOUT unused_*
        _apply_unused_fields(tally, run)
        result[run.pk] = tally
```

Apply the same split in `get_or_compute_tally()` and `get_or_compute_rollup()`. If the
caching must stay, add `run.run_status` and a cheap allocation-event version
(`Max(CalendarEvent.modified)` over `allocation_events(run)`) to both cache keys, and add a
test that flips `run_status` to `CANCELLED` and asserts the table count and the calendar
decoration agree on the very next render.

---

### CR-03: `facility_for()` raises `ImportError` for an unknown facility — unguarded on three public pages that promise "never raises"

**File:** `solsys_code/campaign_tally.py:166-167`, `solsys_code/campaign_gap.py:220`, `solsys_code/templatetags/calendar_display_extras.py:611`

**Issue:** `observation_projector.facility_for()` (`observation_projector.py:62-76`) calls
`tom_observations.facility.get_service_class(name)`, which raises `ImportError` for any
facility name not in `settings.TOM_FACILITY_CLASSES` / the `observation_facilities()`
integration point (verified in the installed
`tom_observations/facility.py:110-120`). Three new Phase 37 call sites invoke it with no
guard, over records whose `facility` value is whatever TOM stored historically:

- `campaign_tally.night_counts_for_run()` line 167 — reached from the **anonymous** campaign
  run table (`campaign_views.py:203-204`) and the **anonymous** campaign list
  (`campaign_views.py:276-277`).
- `campaign_gap.observation_claimed_dates()` line 220 — reached from the gap-analysis page.
- `calendar_display_extras.run_tally()` line 611 — reached from the calendar pop-up.

Every one of these advertises the opposite. `night_counts_for_run`'s docstring
(`campaign_tally.py:146`) says "never raises"; `run_tally`'s docstring
(`calendar_display_extras.py:585`) says "This tag never raises". A single record left behind
by removing `tom_eso` or `tom_gemini` from the facility list — both names are explicitly
modelled in `status_vocabulary.OBSERVED_STATES_BY_FACILITY` — takes the public campaign list
to a 500 for every visitor. Note the observation projector itself only survives this because
its signal receivers swallow everything; these new callers have no such net.

**Fix:** Give the shared helper a non-raising variant and use it on every display path:

```python
# observation_projector.py
def facility_for_or_none(record: ObservationRecord) -> Any | None:
    """facility_for(), returning None instead of raising for an unconfigured facility."""
    try:
        return facility_for(record)
    except ImportError:
        logger.debug('facility_for: no configured facility named %r (record pk=%s)', record.facility, record.pk)
        return None
```

Then in `night_counts_for_run()` / `observation_claimed_dates()`:

```python
facility = facility_for_or_none(record)
if facility is None:
    continue          # or increment the existing site_unknown_count in campaign_gap
state = classify_record(record, facility)
```

and wrap the `get_or_compute_tally()` call in `run_tally()` in the same guard so the tag's
"never raises" contract becomes true. Add a test with `record.facility = 'NOT_CONFIGURED'`
asserting a zero tally rather than an exception.

## Warnings

### WR-01: `ApprovalQueueTable` inherits a `Progress` column that is permanently "Progress not available"

**File:** `solsys_code/campaign_tables.py:99`, `solsys_code/campaign_tables.py:156-162`, `solsys_code/campaign_tables.py:324-380`, `solsys_code/campaign_views.py:408-435`

**Issue:** `progress` is declared on `CampaignRunTable`, so `ApprovalQueueTable` inherits it,
and its `Meta.sequence` ends in `'...'` so the column lands at the far right. The three
approval-queue tables are constructed without a `tallies` kwarg
(`campaign_views.py:408`, `:415`, `:428`), so `self.tallies == {}` and `render_progress()`
falls into its not-available branch for **every** row. Confirmed empirically:

```
approval cols: ['actions', 'approval_status', 'telescope_instrument', 'site', 'window_start',
 'telescope_class', 'filters_bandpass', 'run_status', 'open_to_collaboration',
 'observation_details', 'comments', 'contact_person', 'contact_email', 'progress']
tallies attr: {}
```

The `__init__` docstring claims the optional kwarg keeps `ApprovalQueueTable` "working
unchanged" — it does not; it adds a wide, permanently-dead column to three staff pages.

**Fix:** Either exclude it in the subclass, or feed it. Excluding is the smaller change:

```python
class Meta(CampaignRunTable.Meta):  # noqa: D106
    exclude = ('weather', 'observation_outcome', 'publication_plans', 'progress')
```

If staff should see progress in the queue, pass `tallies=campaign_tally.tallies_for_runs(...)`
at each of the three construction sites in `ApprovalQueueView` instead.

---

### WR-02: `night_counts_for_run()`'s `.only()` omits `parameters`, which `record_time_window()` reads — a hidden per-record query

**File:** `solsys_code/campaign_tally.py:157-161`

**Issue:** The comment says the field list is "exactly what `classify_record()`/
`record_time_window()` read", but `record_time_window()` reads `record.parameters['start']`/
`['end']` whenever both schedule fields are `None` (`calendar_utils.py:543-550`). That branch
is reachable for records that survive the `_NIGHT_CLAIMING_STATES` filter: a
`WINDOW_EXPIRED`/`CANCELED`/`FAILURE_LIMIT_REACHED`/`NOT_ATTEMPTED` record with no placed
block classifies as a failure state (`status_vocabulary.py:266-268`), and a `completed-no-block`
observed record likewise. Each such record triggers a deferred-field refresh — one extra
`SELECT` per record, on a public page. `campaign_gap.observation_claimed_dates()` gets this
right (`campaign_gap.py:215` includes `'parameters'`), which makes the two modules
inconsistent.

**Fix:**

```python
    ).only('pk', 'status', 'facility', 'scheduled_start', 'scheduled_end', 'parameters')
```

and correct the comment above it.

---

### WR-03: `campaign_rollup()`'s `.only()` triggers a deferred load of `run_status` plus a `site` query on every run

**File:** `solsys_code/campaign_tally.py:466-470`

**Issue:** `.only('pk', 'proposal_code', 'site_id')` defers `run_status`, but the roll-up path
reads it: `_apply_unused_fields()` → `unused_nights_for_run()` →
`is_unused_allocation_night(event.end_time, run.run_status)` (`campaign_tally.py:342`). That
is one extra `SELECT` per run. Separately, `night_counts_for_run()` reads `run.site.timezone`
(`campaign_tally.py:148`) with no `select_related('site')` on this queryset — a second extra
`SELECT` per run. Both are invisible N+1s on the anonymous campaign list, which loops this
over every campaign (see WR-05).

**Fix:**

```python
    runs = list(
        CampaignRun.objects.filter(campaign=campaign)
        .exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
        .select_related('site')
        .only('pk', 'proposal_code', 'run_status', 'site_id',
              'site__timezone', 'site__obscode')
    )
```

---

### WR-04: The campaign table computes tallies for the entire filtered queryset, not the 25-row page

**File:** `solsys_code/campaign_views.py:203-204`

**Issue:** `self.object_list` is the full filtered queryset; `table_pagination = {'per_page': 25}`
means only 25 rows are rendered. `tallies_for_runs()` is therefore asked for every run in the
campaign, and on a cold cache performs 3-5 queries per run
(`night_counts_for_run` + `allocation_events` + the proposal-allocation lookups). A campaign
with 500 runs pays ~2000 queries to render 25 cells. This is the per-row query loop D-08 set
out to prevent, displaced one level up.

**Fix:** Scope the tally pass to the rendered page. `get_table_kwargs()` runs before the table
exists, so derive the page slice from the request:

```python
def get_table_kwargs(self):
    per_page = self.table_pagination['per_page']
    try:
        page = max(int(self.request.GET.get('page', 1)), 1)
    except (TypeError, ValueError):
        page = 1
    page_pks = list(self.object_list.values_list('pk', flat=True)[(page - 1) * per_page: page * per_page])
    runs = CampaignRun.objects.filter(pk__in=page_pks).select_related('site')
    return {'order_by': (), 'tallies': campaign_tally.tallies_for_runs(runs)}
```

(Alternatively override `get_table()` and slice from `table.page.object_list`.)

---

### WR-05: `CampaignListView` loops a roll-up over an unpaginated campaign list on a public page

**File:** `solsys_code/campaign_views.py:276-277`

**Issue:** `CampaignListView` declares no `paginate_by`, so `context['campaigns']` is every
campaign. Each iteration calls `get_or_compute_rollup(campaign)`, which on a cold cache costs
`campaign_records_version()` (1 query) + `campaign_rollup()` (1 runs query + 2 aggregate
queries + WR-03's 2 deferred queries per run + `night_counts_for_run`'s query per run +
`allocation_events`' query per run + up to 2 proposal-allocation queries per run). The page is
reachable anonymously, so a single unauthenticated GET after a cache flush fans out to
O(campaigns × runs) queries — an easy accidental (or deliberate) amplification.

**Fix:** Bound the work. Either paginate the list (`paginate_by = 50`) and keep the loop, or
replace the per-campaign roll-up with a single annotated aggregate for the only value the
template actually renders (`campaign.rollup.nights_observed`,
`src/templates/campaigns/campaign_list.html:48`), or gate the loop on a cache hit only and
render "—" on a miss.

---

### WR-06: `unused_night_decoration()` has no `is_publicly_visible` gate, unlike every sibling tally surface

**File:** `solsys_code/templatetags/calendar_display_extras.py:661-679`

**Issue:** `run_tally()` gates on `run is None or not run.is_publicly_visible`
(`calendar_display_extras.py:607-609`) and `campaign_rollup()` excludes
`PENDING_REVIEW` at the queryset level (`campaign_tally.py:467-468`), but
`unused_night_decoration()` only checks `run is None` (line 671). A pending-review run's
allocation night therefore gets the public `[U]` token, the dashed muted chip, and the
"This awarded night passed with nothing scheduled or observed." tooltip on the anonymous
calendar — leaking the fact that an unreviewed run exists and asserting a judgement about it.
It also makes the calendar's `[U]` set a superset of what the table counts, another D-15
divergence.

**Fix:**

```python
    run = meta.run
    if run is None or not run.is_publicly_visible:
        return None
```

---

### WR-07: Proposal code is interpolated into the credentialed portal URL path without quoting or validation

**File:** `solsys_code/proposal_allocation.py:104-109`

**Issue:**

```python
urljoin(facility.facility_settings.get_setting('portal_url'), f'/api/proposals/{proposal_code}/')
```

`proposal_code` is not percent-encoded and not validated. Its two sources are both
operator-supplied free text: `WatchedProposal.proposal_code` (admin-editable) and
`CampaignRun.proposal_code`, which comes verbatim from the bracketed `[proposal]` token in a
classical schedule file — `telescope_runs._resolve_proposal()` (`telescope_runs.py:401-437`)
accepts any non-empty text between `[` and `]` with no charset or length constraint. A value
containing `..`, `?` or `#` redirects the authenticated request to a different portal endpoint
(`/api/proposals/../requestgroups/` normalises server-side), and a value over 100 characters
will raise `DataError` on PostgreSQL against `max_length=100`.

**Fix:** Quote and validate:

```python
import re
from urllib.parse import quote

_PROPOSAL_CODE_RE = re.compile(r'^[A-Za-z0-9._\-]{1,100}$')

def fetch_proposal_allocations(proposal_code: str, facility: LCOFacility) -> list[dict[str, Any]]:
    if not _PROPOSAL_CODE_RE.match(proposal_code or ''):
        raise PortalUnavailable('ValueError')
    url = urljoin(
        facility.facility_settings.get_setting('portal_url'),
        f'/api/proposals/{quote(proposal_code, safe="")}/',
    )
```

and reject an over-long/ill-formed token in `_resolve_proposal()` so it never reaches the DB.

---

### WR-08: `unused_hours_for()` sums across every semester and instrument type, and stale rows are never pruned

**File:** `solsys_code/proposal_allocation.py:181-187`, `solsys_code/proposal_allocation.py:129-166`

**Issue:** The sum filters only on `proposal_code` and `allocation_type__in=('std',)`. A
proposal carrying allocations in two semesters contributes both, so a proposal that finished
2026A with 40 unused standard hours and has 100 hours in 2026B reports 14 estimated unused
nights rather than 10 — a figure published on a public page as the run's wasted time.
Compounding it, `store_proposal_allocations()` only ever calls `update_or_create()`; a row
whose (semester, instrument_type, allocation_type) key disappears from the portal response is
never deleted, so retired semesters accumulate forever and the estimate only ever grows.
Neither behaviour is tested (`test_proposal_allocation.py` has no cross-semester case).

**Fix:** Scope the sum to the current/most-recent semester, and prune on refresh:

```python
def unused_hours_for(proposal_code: str, semester: str | None = None) -> float | None:
    rows = ProposalTimeAllocation.objects.filter(
        proposal_code=proposal_code, allocation_type__in=ESTIMATE_ALLOCATION_TYPES
    )
    if semester is None:
        semester = rows.order_by('-semester').values_list('semester', flat=True).first()
    if semester is None:
        return None
    rows = rows.filter(semester=semester)
    ...
```

and in `store_proposal_allocations()`, after the loop, delete rows for this
`proposal_code` whose key was not in the response. If summing across semesters is the
intended rule, say so explicitly in the docstring and add a test that pins it.

---

### WR-09: Bare marker literals survive in `observation_projector.py` and `calendar.html` — the exact drift STATUS-01 set out to remove

**File:** `solsys_code/observation_projector.py:98`, `solsys_code/observation_projector.py:207`, `src/templates/tom_calendar/partials/calendar.html:356`

**Issue:** After the consolidation, three hardcoded markers remain outside
`status_vocabulary`:

- `observation_projector.py:98` — `return FAILURE_MARKER_BY_STATUS.get(status, '[F]')`
- `observation_projector.py:207` — `marker = STAGE_MARKER.get(stage, '[?]')`
- `calendar.html:356` — `{% if entry.marker == '[U]' %}` gates the click-to-filter treatment

The first two are exactly the "one module still carries its own bare quoted marker" pattern
the phase docstring (`status_vocabulary.py:3-9`) declares eliminated. The third is worse in
kind: if `MARKER[DisplayState.UNUSED]` ever changes, the legend silently degrades from a
filterable swatch to an inert entry with no test or error to catch it — the same
byte-identical-by-convention coupling the phase removed from Python.

**Fix:**

```python
# observation_projector.py
from solsys_code.status_vocabulary import (
    FAILURE_MARKER_BY_STATUS, MARKER, STAGE_MARKER, DisplayState, failed_states_for, observed_states_for,
)
...
    return FAILURE_MARKER_BY_STATUS.get(status, MARKER[DisplayState.FAILED])
...
    marker = STAGE_MARKER.get(stage, MARKER[DisplayState.INCONSISTENT])
```

For the template, expose the marker rather than comparing to a literal — e.g. add
`'filterable': state is DisplayState.UNUSED` to each `LEGEND` entry in
`status_vocabulary.LEGEND` and branch on `{% if entry.filterable %}`.

---

### WR-10: `ProposalTimeAllocationAdmin` is read-only per field but leaves add and delete unguarded

**File:** `solsys_code/admin.py:485-511`

**Issue:** The docstring says "a staff user cannot hand-edit a figure the public tallies
present as portal-sourced (T-37-07)", and the model docstring says the rows are "Written only
by the unattended runner's proposal-allocation step". Neither `has_add_permission()` nor
`has_delete_permission()` is overridden, so a staff user can **delete** rows — which silently
changes the public unused-nights estimate, or flips it from a number to "not yet known" —
and the "Add" button is rendered despite every field being read-only, producing a form that
cannot satisfy the non-null `fetched_at`.

**Fix:**

```python
class ProposalTimeAllocationAdmin(admin.ModelAdmin):  # noqa: D101
    ...
    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
```

---

### WR-11: The gap page's new "Claimed nights" list is not bounded by the requested date range

**File:** `src/templates/campaigns/campaignrun_gap_analysis.html:48-63`, `solsys_code/campaign_gap.py:282-290`

**Issue:** `claimed_dates()`'s own WR-05 note states that `claimed_dates` is campaign/site-wide
and "NOT scoped to `[start, end]`", and `_compute_gap()` passes it straight through
(`campaign_gap.py:391`). The new template block renders that unbounded set as a flat list
headed "Claimed nights" on a page whose whole premise is the user-selected date range. A user
asking about the next 30 days is shown claimed nights from years ago and years ahead, with no
indication they are out of range.

**Fix:** Bound the displayed lists in `_compute_gap()`, where `start`/`end` are in scope:

```python
    in_range = {d for d in claimed if start <= d <= end}
    return {
        ...
        'claimed_dates': sorted(in_range),
        'observation_claimed_dates': sorted(d for d in observation_claimed if start <= d <= end),
        ...
    }
```

(`gap = obs - claimed` must keep using the *unbounded* `claimed`, as today.)

---

### WR-12: The runbook still documents the retired `[CANCELLED]`/`[WEATHERED]` prefixes as current

**File:** `docs/runbooks/telescope_runs_calendar.rst:1277-1279`

**Issue:** The phase rewrote the marker table (lines 136-200) and added a "One-time title
change (Phase 37)" note (lines 687-699) explaining that `[CANCELLED]`/`[WEATHERED]` are
rewritten to `[C]`/`[W]`. Line 1277-1279 still tells the operator that a declined-retirement
night "picks up its ``[CANCELLED]`` / ``[WEATHERED]`` prefix on the next sweep". Under
CLAUDE.md's paired-docs rule the runbook is part of the deliverable, and an operator reading
that passage will look for a title that no writer produces.

**Fix:** Replace the two markers in that sentence with ``[C]`` / ``[W]``:

```rst
   ...so the entry picks up its ``[C]`` / ``[W]`` marker on the next
   sweep instead of sitting on the calendar as an ordinary observing night...
```

## Info

### IN-01: A cancelled night is labelled "Expired/failed" in the tally segment

**File:** `solsys_code/campaign_tally.py:293-297`, `solsys_code/campaign_tally.py:59-67`

**Issue:** `_NIGHT_CLAIMING_STATES` folds `WINDOW_EXPIRED`, `CANCELLED` and `FAILED` into
`nights_failed`, which `tally_segments()` renders as `[X/F] Expired/failed`. A portal-cancelled
night is neither expired nor failed, and the combined marker omits `[C]` even though `[C]` is
one of the three states being summed.

**Fix:** Either widen the token/label to `[X/C/F]` / "Expired, cancelled or failed", or split
cancelled into its own segment.

---

### IN-02: `status_border_css()`'s `'[QUEUED] '` branch is dead code

**File:** `solsys_code/templatetags/calendar_display_extras.py:183-184`

**Issue:** A repo-wide grep finds no producer of a `'[QUEUED] '`-prefixed title outside test
fixtures; every current writer emits `'[Q] '` via `STAGE_MARKER`. The branch is retained
"deliberately" per the docstring, but with `RETIRED_TITLE_PREFIXES` deleted in 37-07 on the
grounds that the database holds no legacy spellings, the same argument retires this one.

**Fix:** Delete the branch (the `RING_QUEUED_STATES` check immediately below covers `'[Q] '`),
or document the specific legacy rows it exists for.

---

### IN-03: The `[U]` legend swatch never updates `aria-pressed`

**File:** `src/templates/tom_calendar/partials/calendar.html:356-360`, `:400-420`

**Issue:** The new filter control is `role="button" aria-pressed="false"`, but the click
handler only toggles the `is-active` class; `aria-pressed` stays `"false"` forever, so a
screen-reader user cannot tell the filter is on. It is also not keyboard-focusable
(`<span role="button">` with no `tabindex`).

**Fix:** Add `tabindex="0"`, a keydown handler for Enter/Space, and
`el.setAttribute('aria-pressed', String(matched))` alongside each `classList.toggle('is-active', ...)`.

---

### IN-04: The portal fetch depends on `LCOFacility._portal_headers()`, a private upstream method

**File:** `solsys_code/proposal_allocation.py:107`

**Issue:** `facility._portal_headers()` is a leading-underscore method of a third-party class;
an upstream rename breaks the unattended step with an `AttributeError` that
`fetch_proposal_allocations()`'s except clause does not catch (it lists
`RequestException`/`ImproperCredentialsException`/`ValidationError`/`ValueError`), so it
escapes `refresh_all()`'s per-proposal isolation and aborts the whole step.

**Fix:** Pin the dependency in a comment, and add `AttributeError` to the caught set (or wrap
the header construction) so one upstream change degrades to "portal unavailable" rather than
an unattended-runner traceback.

---

### IN-05: `ProposalTimeAllocation.save()` strips `proposal_code` while `update_or_create()` looks it up unstripped

**File:** `solsys_code/models.py:836-844`, `solsys_code/proposal_allocation.py:154-164`

**Issue:** `update_or_create(proposal_code=' X ', ...)` fails to match the stored `'X'`,
creates a second instance, whose `save()` strips it back to `'X'`, and the unique constraint
then raises `IntegrityError`. Today both code sources happen to be pre-stripped
(`_resolve_proposal()` strips, `WatchedProposal.save()` strips), so this is latent rather than
live.

**Fix:** Strip at the call site — `proposal_code = (proposal_code or '').strip()` at the top of
`store_proposal_allocations()` and `proposal_codes_to_fetch()`.

---

### IN-06: Stale comments and a now-vacuous assertion reference the retired bracket-word prefixes

**File:** `solsys_code/campaign_views.py:805`, `solsys_code/campaign_views.py:840`, `solsys_code/models.py:173`, `solsys_code/tests/test_campaign_approval.py:593`

**Issue:** Three code comments still describe the writer as producing
`[CANCELLED]`/`[WEATHERED]`/`[EXPIRED]` titles. `test_campaign_approval.py:593`
(`self.assertFalse(event.title.startswith('[CANCELLED]'))`) is now trivially true for every
possible title and no longer distinguishes the weathered marker from the cancelled one.

**Fix:** Update the three comments to `[C]`/`[W]`/`[X]`, and change the assertion to
`self.assertFalse(event.title.startswith(RUN_STATUS_MARKER[CampaignRun.RunStatus.CANCELLED]))`.

---

### IN-07: `get_table_kwargs()` builds an unrestricted `CampaignRun` queryset on non-staff requests

**File:** `solsys_code/campaign_views.py:203`

**Issue:** `CampaignRun.objects.filter(pk__in=pks).select_related('site')` selects every column
— including `contact_person`/`contact_email` — into the request process for anonymous
visitors, on a view whose surrounding code goes to considerable length
(`ALLOWED_FIELDS_FOR_NON_STAFF`, the `.values()`-before-`.annotate()` gate,
`campaign_gap.claimed_dates()`'s own `.only()` note at `campaign_gap.py:297-302`) to keep
those columns out of a public request. Nothing renders them today, so this is defence in
depth, not a live leak.

**Fix:** Mirror the established discipline:

```python
runs = (CampaignRun.objects.filter(pk__in=pks)
        .select_related('site')
        .only('pk', 'proposal_code', 'run_status', 'site_id', 'site__timezone', 'site__obscode'))
```

(which also fixes WR-03's deferred-field loads on this path).

---

_Reviewed: 2026-09-19_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
