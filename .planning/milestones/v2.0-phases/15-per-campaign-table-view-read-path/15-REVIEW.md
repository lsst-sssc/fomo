---
phase: 15-per-campaign-table-view-read-path
reviewed: 2026-07-03T00:00:00Z
depth: deep
files_reviewed: 12
files_reviewed_list:
  - solsys_code/apps.py
  - solsys_code/campaign_filters.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_urls.py
  - solsys_code/campaign_views.py
  - solsys_code/tests/test_campaign_views.py
  - src/fomo/urls.py
  - src/templates/campaigns/campaign_list.html
  - src/templates/campaigns/campaignrun_table.html
  - src/templates/solsys_code/partials/campaign_links.html
  - src/templates/solsys_code/partials/campaigns_nav_link.html
  - src/templatetags/solsys_code_extras.py
findings:
  critical: 0
  warning: 2
  info: 3
  total: 5
status: issues_found
---

# Phase 15: Code Review Report

**Reviewed:** 2026-07-03T00:00:00Z
**Depth:** deep
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Reviewed the per-campaign CampaignRun table read path (django-tables2 + django-filter),
navigation wiring (target-detail button, navbar entry), and their test suite, at deep depth
(cross-file tracing of the queryset → filterset → table → template pipeline, plus empirical
verification via the Django test runner rather than static reading alone).

Both must-have security properties called out in the review brief were traced end-to-end and
hold up under adversarial scrutiny:

1. **PII gating is enforced at the queryset layer, not just the template.**
   `CampaignRunTableView.get_queryset()` (`solsys_code/campaign_views.py:62-68`) returns a
   `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` queryset for non-staff requests — the SQL `SELECT`
   itself never names `contact_person`/`contact_email`, confirmed by inspecting
   `ALLOWED_FIELDS_FOR_NON_STAFF` (`campaign_views.py:26-44`), which omits both fields. The
   `exclude=('contact_person', 'contact_email')` in `get_table_kwargs()` (`campaign_views.py:70-74`)
   is genuinely a second, redundant layer, not the only one. Confirmed empirically:
   `TestContactFieldGating` in `solsys_code/tests/test_campaign_views.py:137-167` asserts both
   the context-row dict shape and rendered HTML content, and passes.
2. **Campaign discovery goes through `TargetList` membership, never `CampaignRun.target`.**
   `campaign_links()` (`src/templatetags/solsys_code_extras.py:17-29`) filters
   `TargetList.objects.filter(targets=target, campaign_runs__isnull=False)` — membership via the
   `TargetList.targets` M2M, existence-only check on `campaign_runs`, never scoped by the run's
   optional `target` FK. `CampaignListView.queryset` (`campaign_views.py:91-93`) does the same.
   Verified the `run_count` annotation isn't corrupted by combining `.filter()` and
   `.annotate(Count(...))` on the same reverse relation (a classic Django ORM join-fanout
   gotcha) by reproducing it directly against the dev DB — count came back correct.

No critical/blocker-level defects were found. Two warnings and three info-level gaps remain,
detailed below — none of which touch the two must-have properties, but the pagination-ordering
warning does undermine VIEW-01's "the table lists all runs for a campaign" guarantee under
realistic data (multiple runs on the same night, which this exact feature exists to coordinate).

## Warnings

### WR-01: Table's default sort has no tiebreaker — pagination is not guaranteed stable when multiple runs share the same `obs_date`

**File:** `solsys_code/campaign_tables.py:71`
**Issue:** `CampaignRunTable.Meta.order_by = ('-obs_date',)` sorts on a single, frequently-tied
column. `obs_date` is a `DateField` (day granularity) and this feature's own purpose is
coordinating *multiple telescopes observing the same target on the same night* — i.e., rows
sharing `obs_date` are the expected common case, not an edge case. `SingleTableMixin` paginates
with `LIMIT`/`OFFSET` per page. SQL makes no guarantee about row order among ties for
`LIMIT`/`OFFSET` queries unless the `ORDER BY` fully determines a total order; PostgreSQL's own
docs explicitly warn that omitting a tiebreaker means "you will get an unpredictable subset of
the query's rows" across separate paginated requests. CLAUDE.md's own Architectural Constraints
section states production is expected to move off SQLite to PostgreSQL, where this failure mode
is documented and real (it happened to pass in a manual SQLite reproduction here only because
SQLite's planner incidentally fell back to rowid order in this instance — not a guarantee). The
practical consequence: a run can silently disappear from both page 1 and page 2, or appear on
both, depending on the DB's tie-resolution for a given query plan.
**Fix:**
```python
class Meta:  # noqa: D106
    ...
    order_by = ('-obs_date', '-pk')  # D-10 + deterministic tiebreaker
```

### WR-02: No regression test proves PII-excluded columns can't be reached via sort/order query-param tampering

**File:** `solsys_code/tests/test_campaign_views.py:137-167` (`TestContactFieldGating`)
**Issue:** The queryset-layer gating (`.values()` never selecting `contact_person`/
`contact_email`) is the real security boundary, and it's correct today. But the test suite only
exercises the default (unsorted) request. Nothing asserts that a non-staff request with
`?sort=contact_person` or `?sort=-contact_email` still returns 200 without a 500/`KeyError` and
without leaking the columns. Today this is safe by construction — `django_tables2.Table.order_by`
validates requested sort aliases against `self.columns` and silently drops any alias not present
(and `contact_person`/`contact_email` are removed from `self.columns` by the `exclude=` kwarg
before sorting is applied) — but that safety is emergent from two independently-maintained
components (`get_queryset`'s field list and `get_table_kwargs`'s exclude tuple) rather than
something a test locks in. A future change to either (e.g., someone drops the `exclude=` kwarg
while forgetting the `.values()` restriction still applies, or vice versa) would not be caught by
the existing suite.
**Fix:** Add a test asserting a non-staff `?sort=contact_person` (and `-contact_email`) request
still returns 200 and the response body contains neither `CONTACT_PERSON` nor `CONTACT_EMAIL`
strings, alongside the existing `TestContactFieldGating` cases.

## Info

### IN-01: `is_staff` gating logic is duplicated across two independent methods with no shared source of truth

**File:** `solsys_code/campaign_views.py:62-68`, `:70-74`
**Issue:** `get_queryset()` and `get_table_kwargs()` each independently evaluate
`self.request.user.is_staff` to decide the PII boundary. The comment on `get_table_kwargs`
("belt-and-suspenders") makes clear this duplication is intentional as defense-in-depth, which is
reasonable — but nothing ties the two checks together, so a future change to the authorization
rule (e.g., adding an `is_active` check, a feature flag, or a `django-guardian` per-object
permission) requires remembering to update both call sites in lockstep. A drift between them
would silently reopen exactly the gap D-13 is designed to close.
**Fix:** Factor the boundary into a single `self._is_staff_request()` helper (or a cached
property) used by both methods, so there is one place to change the rule.

### IN-02: `CampaignRunTableView` doesn't require the pk to actually be a "campaign" (>=1 `CampaignRun`), unlike `CampaignListView`

**File:** `solsys_code/campaign_views.py:47-59` vs `:83-96`
**Issue:** `CampaignListView` and `campaign_links()` both define "campaign" operationally as "a
`TargetList` with `campaign_runs__isnull=False`" (per D-03/D-01, documented in both docstrings).
`CampaignRunTableView.get_queryset()`/`get_context_data()`, however, accept *any* `TargetList`
pk — `get_object_or_404(TargetList, pk=...)` succeeds for a zero-run `TargetList` (e.g. an
unrelated saved search created via ordinary TOM target-list UI), rendering an (empty) table
titled with that list's name. This isn't a PII leak (no `CampaignRun` rows exist to render) and
`TargetList` has no built-in privacy/ownership model elsewhere in this TOM installation, so the
practical exposure is low — but it is an inconsistency with the "campaign" concept the rest of
this phase establishes: a `TargetList` that intentionally isn't surfaced on `/campaigns/` (because
it has 0 runs, or was never meant to be a campaign) is still directly reachable and readable at
`/campaigns/<pk>/` by URL guessing.
**Fix:** If this matters for the product surface, filter `get_context_data`'s
`get_object_or_404` (or `get_queryset`) through the same `campaign_runs__isnull=False` predicate
used elsewhere, returning 404 for non-campaign `TargetList` pks. Low priority given the lack of
any existing `TargetList` privacy boundary.

### IN-03: `TestCampaignRunFilterSet` doesn't cover the explicit `open_to_collaboration=false` case, only `true` and unset

**File:** `solsys_code/tests/test_campaign_views.py:189-194`
**Issue:** `test_open_to_collaboration_filter` only exercises `?open_to_collaboration=true`. There
is no test for `?open_to_collaboration=false` (expected: the 29 non-`open_to_collaboration` rows)
or for the "Any" sentinel (`unknown`) round-tripping correctly through
`CampaignRunFilterSet`'s `NullBooleanField`/`BooleanWidget` handling used by the custom
`<select>` in `campaignrun_table.html:25-33`. Given the custom widget markup deliberately
diverges from django-filter's default rendering (to get "Any/Yes/No" option labels instead of the
default Unknown/Yes/No), a regression in that hand-rolled `<select>` (e.g. mismatched `value=`
strings) wouldn't be caught by the current suite.
**Fix:** Add `test_open_to_collaboration_false_filter` asserting `?open_to_collaboration=false`
returns the 29 rows lacking the seeded `open_to_collaboration=True` row.

---

_Reviewed: 2026-07-03T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
