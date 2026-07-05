---
phase: 17-coverage-gap-analysis-deferrable-to-v2-1
reviewed: 2026-07-04T22:59:03Z
depth: deep
files_reviewed: 7
files_reviewed_list:
  - solsys_code/campaign_forms.py
  - solsys_code/campaign_gap.py
  - solsys_code/campaign_urls.py
  - solsys_code/campaign_views.py
  - solsys_code/tests/test_campaign_gap.py
  - src/templates/campaigns/campaignrun_gap_analysis.html
  - src/templates/campaigns/campaignrun_table.html
findings:
  critical: 2
  warning: 5
  info: 2
  total: 9
status: issues_found
---

# Phase 17: Code Review Report

**Reviewed:** 2026-07-04T22:59:03Z
**Depth:** deep
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed the coverage-gap analysis feature (GAP-01/GAP-02): `campaign_gap.py`'s pure-logic
observable/claimed/gap computation, `CampaignGapAnalysisView`'s server-side IDOR re-validation
and date-range clamping, the paired form/URL wiring, and both new/changed templates. The module
boundary discipline is good — `campaign_gap.py` and `campaign_views.py` correctly avoid
module-scope imports of the heavy SPICE-loading ephemeris module (verified by grep and by the
suite's own `TestNoHeavyEphemerisImport`), and the cache-key construction in
`build_gap_cache_key()` correctly encodes a null target as the literal `'none'` so it can never
collide with a real target pk (confirmed against `TestBuildGapCacheKey`).

However, two of the areas this review was specifically asked to scrutinize — server-side IDOR
re-validation and the observable/claimed date computation — each have a reproducible unhandled
crash on realistic input, neither of which is covered by the (otherwise fairly thorough) test
suite. Both were confirmed live against this checkout (see CR-01/CR-02 for exact repro steps and
tracebacks). There's also a real, if less urgent, gap between this feature's PII handling and the
"restrict the queryset, not just the rendered output" principle the surrounding `CampaignRunTableView`
code (D-13) explicitly documents and relies on.

## Critical Issues

### CR-01: Non-numeric `target`/`site` GET params crash the view with an unhandled 500 instead of the documented 400

**File:** `solsys_code/campaign_views.py:397` and `solsys_code/campaign_views.py:410`
**Issue:** `CampaignGapAnalysisView.get()` re-derives the campaign's allowed target/site sets and
validates the submitted pk against them — but only for pks that happen to be integers. Both
`campaign.targets.filter(pk=target_pk).exists()` (line 397) and
`allowed_sites.filter(pk=site_pk).exists()` (line 410) pass the raw `request.GET.get(...)` string
straight into `.filter(pk=...)`. Django's `IntegerField.get_prep_value()` raises a bare
`ValueError` when the value can't be parsed as an int, and nothing in this view (or anywhere
upstream) catches it. A request like `?site=abc` (or `?target=abc` on a multi-target campaign)
produces an unhandled 500, not the `HttpResponseBadRequest` the class docstring and
`T-17-01`/Pitfall 3 explicitly promise ("never trusting the dropdown alone" / "HttpResponseBadRequest
on mismatch"). Reproduced live against this checkout:

```
File "solsys_code/campaign_views.py", line 410, in get
    if not allowed_sites.filter(pk=site_pk).exists():
...
ValueError: Field 'id' expected a number but got 'abc'.
```

The existing `test_rejects_out_of_scope_target_and_site` test only exercises **valid-but-wrong**
integer pks (a real target/site belonging to a different campaign), so it never caught this — a
non-numeric value takes an entirely different, uncaught code path.

**Fix:**
```python
def _as_pk_or_none(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None

target_pk = _as_pk_or_none(request.GET.get('target'))
if target_pk is None and request.GET.get('target'):
    context['idor_error'] = True
    return self.render_to_response(context, status=400)
...
site_pk = _as_pk_or_none(request.GET.get('site'))
if site_pk is None:
    if request.GET.get('site'):
        context['idor_error'] = True
        return self.render_to_response(context, status=400)
    return self.render_to_response(context)
```
Apply the same guard to both the `target` and `site` lookups before they ever reach `.filter(pk=...)`.

### CR-02: `claimed_dates()` crashes with an unhandled `ValueError` for any site with a blank `timezone`, unlike the parallel, documented per-record skip on the observable side

**File:** `solsys_code/campaign_gap.py:100` (`_observing_night_date`), reached from
`solsys_code/campaign_gap.py:181` (`claimed_dates`)
**Issue:** `claimed_dates()`'s module docstring and `observable_dates()`'s D-03 handling both
establish "a per-date/per-record log+skip discipline" for messy data — but that discipline is
only implemented on the *observable* side (`observable_dates()` wraps `sun_event()` in
`try/except ValueError`). The *claimed* side has no equivalent protection:
`_observing_night_date(run.ut_start, site.timezone)` calls `ZoneInfo(tz_name)` directly, and when
`site.timezone == ''` this raises `ValueError: ZoneInfo keys must be normalized relative paths,
got: ` with no surrounding `try/except` anywhere in `claimed_dates()`, `_compute_gap()`,
`get_or_compute_gap()`, or the view. This propagates uncaught all the way to a 500 for the whole
gap-analysis page — not a single skipped date, but a total failure to render.

This is not a hypothetical: `Observatory.timezone` is `blank=True, default=''`
(`solsys_code/solsys_code_observatory/models.py:55`), and neither Tier 2
(`MPCObscodeFetcher.to_observatory()`, `solsys_code/solsys_code_observatory/utils.py:64-84`) nor
Tier 3 (`resolve_site()`'s placeholder-create path, `solsys_code/campaign_utils.py:155-160`) ever
sets it. Both tiers are reached by the ordinary "approve a run" flow
(`CampaignRunDecisionView.post()` → `resolve_site()`), so any approved `CampaignRun` whose site
was auto-resolved via MPC lookup or created as an unresolved placeholder — and which has
`ut_start` set but no `obs_date` — will have `site.timezone == ''` and will crash this view.
Reproduced live against this checkout:

```
File "solsys_code/campaign_gap.py", line 181, in claimed_dates
    claimed.add(_observing_night_date(run.ut_start, site.timezone))
  File "solsys_code/campaign_gap.py", line 100, in _observing_night_date
    local = ut_start.astimezone(ZoneInfo(tz_name))
ValueError: ZoneInfo keys must be normalized relative paths, got:
```

No test in `test_campaign_gap.py` covers a blank-timezone site (all fixture `Observatory` rows
set an explicit `timezone=`), so this is currently uncaught by the suite.

**Fix:** Wrap the per-run date derivation the same way `observable_dates()` wraps `sun_event()`,
and route runs whose site has no usable timezone into `undated_runs` (or a new, explicitly
surfaced bucket) rather than letting the whole request fail:
```python
for run in qs:
    if run.obs_date is not None:
        claimed.add(run.obs_date)
    elif run.ut_start is not None:
        try:
            claimed.add(_observing_night_date(run.ut_start, site.timezone))
        except ValueError:
            logger.debug('Could not derive observing night for run=%s (site timezone unset?)', run.pk)
            undated_runs.append(run)
    else:
        undated_runs.append(run)
```
(`_observing_night_date` itself already raises `ValueError` for the missing-timezone case per the
error message above, so no change is needed there beyond catching it at the call site.)

## Warnings

### WR-01: Public, unauthenticated gap-analysis view fetches full `CampaignRun` rows (including `contact_person`/`contact_email`) into the cached result, contradicting this codebase's own D-13 "restrict the queryset, not just the rendered output" rule

**File:** `solsys_code/campaign_gap.py:170` and `:176` (`unattributed_runs`/`undated_runs`
construction in `claimed_dates()`)
**Issue:** `CampaignGapAnalysisView` has no `StaffRequiredMixin` (intentionally — same posture as
`CampaignRunTableView`), yet `CampaignRunTableView.get_queryset()` explicitly restricts non-staff
requests to a `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` queryset specifically so
`contact_person`/`contact_email` are "never fetched for non-staff... per 15-RESEARCH.md Pitfall 1's
'restrict the queryset, not just the rendered table' recommendation" (see the module docstring
and `ALLOWED_FIELDS_FOR_NON_STAFF` comment at the top of `campaign_views.py`). `claimed_dates()`
does not follow that same discipline: `qs` and `unattributed_runs = list(qs.filter(target__isnull=True))`
fetch full model instances with every field, including `contact_person`/`contact_email`, and these
full objects are placed into the `_compute_gap()` result dict, which is then pickled into Django's
cache backend (`FileBasedCache` in `src/fomo/settings.py`, writing to a shared OS temp directory)
for up to an hour, servable to any anonymous visitor who can compute the cache key. The current
template only ever renders `|length`, so there's no live PII leak in the rendered response today —
but the underlying queryset itself carries the PII fields into a cache that this exact codebase's
own precedent says should never happen for a public-facing path, and it takes only one future
template change (e.g. someone adding a "contact: {{ run.contact_person }}" convenience line to the
undated/unattributed banner) to turn this into a real leak with no queryset-level guard to catch it.
**Fix:** Restrict the `undated_runs`/`unattributed_runs` collection to the same PII-safe field set
`ALLOWED_FIELDS_FOR_NON_STAFF` already defines, e.g. `qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` (or
a subset of just `pk`/`telescope_instrument`/`obs_date` if that's all this feature will ever need),
so contact fields are never fetched on this code path at all — matching the "restrict the
queryset" pattern rather than relying on the template to stay conservative.

### WR-02: `clamp_date_range()` doesn't enforce `end >= start`, so a past `end_date` silently produces a misleading "No gaps found" instead of a validation error

**File:** `solsys_code/campaign_gap.py:44-60`
**Issue:** `clamp_date_range()` only clamps the *upper* bound (`min(requested_end, max_end)`); it
never checks that `requested_end >= today`. A client submitting `?end_date=2020-01-01` (or any
past date) gets `start=today, end=<past date>`, so `end < start`. Downstream, `n_days =
(end-start).days + 1` goes negative or zero, `range(n_days)` is simply empty, `observable_dates()`
returns an empty set, and `_compute_gap()` reports `gap_dates: []`, `unknown_date_count: 0`. The
template then renders the "No gaps found... every observable night in this window is already
claimed" success message, which is factually wrong — there was no window to search at all. No
crash, but a client-supplied out-of-range value is silently accepted and produces an actively
misleading result rather than a validation error.
**Fix:** In `clamp_date_range()`, also floor `requested_end` at `start`:
```python
if requested_end is None:
    return start, default_end
return start, max(start, min(requested_end, max_end))
```
and consider surfacing a distinct "not a usable range" state in the result so the template doesn't
conflate "searched and found nothing" with "nothing was searched."

### WR-03: `end_date` is parsed by hand from raw `request.GET`, bypassing the bound `CampaignGapAnalysisForm` entirely and silently falling back to the 90-day default on any parse failure

**File:** `solsys_code/campaign_views.py:419-425`
**Issue:** `CampaignGapAnalysisForm` is instantiated and rendered (`form = CampaignGapAnalysisForm(request.GET or None, campaign=campaign)`), but `form.is_valid()`/`form.cleaned_data` is never
called anywhere in `CampaignGapAnalysisView.get()` (confirmed by grep — the only `form.cleaned_data`
usages in this file are in the unrelated `CampaignRunSubmissionView`). Instead, the view re-parses
`request.GET.get('end_date')` directly with `date.fromisoformat(raw_end_date)`, which only accepts
strict `YYYY-MM-DD` — stricter than Django's `DateField`, which accepts a configurable list of
input formats. Any `ValueError` here (including a value the form's own field would have accepted,
or simply a malformed value) is caught and silently discarded (`requested_end = None`), falling
back to the 90-day default window with **no error surfaced to the user or recorded in context** —
the user asked for a specific end date and silently got a different one. This is the same "Silent
Fallback" anti-pattern this project's own CLAUDE.md architecture notes call out by name (see
"Silent Fallback in MPC Parallax Conversion" under Anti-Patterns) — applied here to date parsing
in a brand-new feature that could have used the already-bound, already-validated form instead.
**Fix:** Call `form.is_valid()` and read `form.cleaned_data.get('end_date')` instead of re-parsing
raw `request.GET`; if the form is invalid, render the errors instead of silently substituting the
default window.

### WR-04: Redundant TOCTOU-prone `.filter(...).exists()` + `.get(...)` pairs for target/site validation

**File:** `solsys_code/campaign_views.py:397-402` and `:410-414`
**Issue:** Both the target and site validation blocks run two separate queries where one would
do (`campaign.targets.filter(pk=target_pk).exists()` then, on the next line,
`campaign.targets.get(pk=target_pk)`; same pattern for `allowed_sites`). Besides the redundant
round-trip, this is a narrow TOCTOU window: if the row is deleted between the `.exists()` check
and the `.get()` call, `.get()` raises `Target.DoesNotExist`/`Observatory.DoesNotExist`, which is
not caught anywhere and would crash the request the same way CR-01/CR-02 do. Low likelihood, but
avoidable for free.
**Fix:** Collapse to a single query, e.g. `target = campaign.targets.filter(pk=target_pk).first()`
then check `if target is None: ... 400 ...` — one query, no TOCTOU window.

### WR-05: `claimed_dates()` ignores the `[start, end]` window entirely — correctness currently rests on an implicit invariant, not an explicit one

**File:** `solsys_code/campaign_gap.py:135-185`
**Issue:** Unlike `observable_dates(site, start, end)`, `claimed_dates(campaign, target, site)`
takes no date-range parameters at all — it queries and returns *every* approved, non-excluded
`CampaignRun` for the campaign/site combination regardless of the requested window. `_compute_gap()`
still produces the correct `gap_dates` because `gap = obs - claimed` is only ever evaluated against
the range-bounded `obs` set, but the cached result's own `claimed_dates` key (and `unattributed_runs`
that feed it) are silently unbounded even though `build_gap_cache_key()` includes the date range as
one of its four dimensions — implying, misleadingly, that the whole cached payload is range-scoped.
Nothing renders `result['claimed_dates']` today, so this isn't currently user-visible, but it's a
latent trap for the next consumer (e.g. a future Stage 2-4 feature reusing this cached dict) who
reasonably assumes a date-range-keyed cache entry only contains data within that range.
**Fix:** Either scope the query explicitly (`.filter(Q(obs_date__range=(start, end)) | Q(obs_date__isnull=True, ut_start__date__range=(start, end)))`, being careful with the observing-night convention) or, at minimum, document at the point of construction that `claimed_dates`/`unattributed_runs`/`undated_runs` are campaign/site-wide, not range-scoped, so a future caller doesn't assume otherwise.

## Info

### IN-01: Template hardcodes "UTC" next to a timezone-aware datetime Django will auto-localize

**File:** `src/templates/campaigns/campaignrun_gap_analysis.html:29`
**Issue:** `<small>Last computed: {{ result.computed_at }} UTC</small>` hardcodes the "UTC" label,
but with `USE_TZ=True` Django's template rendering auto-converts an aware `datetime` to the
project's active `TIME_ZONE` (currently `'UTC'` in `src/fomo/settings.py`, so this is correct
today by coincidence of that setting, not by construction). If `TIME_ZONE` is ever changed away
from `'UTC'`, this label would silently become wrong with no test catching it (no test in
`test_campaign_gap.py` asserts on the rendered timezone label).
**Fix:** `{% load tz %}{% localtime off %}{{ result.computed_at }} UTC{% endlocaltime %}`, or format
explicitly with `result.computed_at|date:"c"` plus an explicit UTC conversion in the view.

### IN-02: `campaign.targets.count()` is queried independently three times per request for the same fact

**File:** `solsys_code/campaign_views.py:390` (`CampaignGapAnalysisView.get()`),
`solsys_code/campaign_forms.py:86` (`CampaignGapAnalysisForm.__init__`), and
`solsys_code/campaign_views.py:356` (`gap_analysis_available()`)
**Issue:** Each of these independently runs `campaign.targets.count()` (or `.count() == 0`) against
the same `campaign` within a single request, purely for code-quality/DRY reasons (not flagged as a
performance issue, out of scope for this review) — a future edit to one call site's semantics
(e.g. adding a target-status filter) could easily drift from the other two since there's no single
source of truth.
**Fix:** Compute once in the view and pass the count (or a `single_target: bool`) through to the
form constructor and to `gap_analysis_available()` rather than recomputing independently.

---

_Reviewed: 2026-07-04T22:59:03Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
