---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
reviewed: 2026-07-15T00:00:00Z
depth: deep
files_reviewed: 10
files_reviewed_list:
  - solsys_code/campaign_forms.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_urls.py
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_site_search.py
  - solsys_code/tests/test_campaign_submission.py
  - src/templates/campaigns/approval_queue.html
  - src/templates/campaigns/partials/site_search_results.html
findings:
  critical: 1
  warning: 2
  info: 2
  total: 5
status: issues_found
---

# Phase 22: Code Review Report

**Reviewed:** 2026-07-15T00:00:00Z
**Depth:** deep
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Reviewed the full diff introducing the shared anonymous/throttled HTMX site-search endpoint
(`SiteSearchView`), its wiring into the public submission form and approval-queue widgets, and
the new "Sites Needing Review" table + `resolve_site` decision action with the extracted
`_project_calendar_event()` helper. Ran the full `solsys_code.tests.test_campaign_*` suite (100
tests, all passing), `ruff check`/`ruff format --check` (clean), and independently verified
against Django 5.2's actual `QuerySet.annotate()` source that the `.values()`-before-`.annotate()`
PII-gating trick in `CampaignRunTableView.get_queryset()` really does work as claimed.

The items specifically called out in the prompt all check out as implemented correctly:
- **XSS escaping**: `site_search_results.html` runs `display`, `obscode`, and every `input_id`
  occurrence through `|escapejs` inside the `onclick=` JS-string context. Confirmed against
  Django's actual `_js_escapes` table that `'`, `"`, `<`, `>`, `&` are all hex-escaped and the
  filter's output is `mark_safe`d (so it isn't double-HTML-escaped) — this genuinely closes the
  JS-string-context XSS gap the in-line pre-implementation review (22-REVIEWS.md finding 2)
  flagged, and the server-side `_INPUT_ID_RE` allowlist in `campaign_views.py` is real
  belt-and-suspenders on top of it.
- **htmx `hx-trigger` grammar**: `input[this.value.length >= 2] changed delay:300ms` — filter
  immediately after the event name, modifiers after — is used consistently in both the form
  widget and the table widget, matches htmx's documented grammar, and is exercised by dedicated
  regression tests (`assertNotContains(response, 'delay:300ms[')`).
- **Per-IP throttle**: `_check_and_increment_throttle()`'s `add()`-then-`incr()` logic was traced
  by hand against `SITE_SEARCH_THROTTLE_LIMIT` — off-by-one behavior is correct (exactly `LIMIT`
  requests allowed, the `LIMIT+1`th rejected), and this matches the passing throttle tests.
- **Concurrency-safe site claim in `resolve_site`**: the conditional
  `.filter(pk=pk, approval_status=APPROVED, site_needs_review=True, site__isnull=True).update(site=site)`
  correctly makes exactly one of two racing POSTs win, and `site_needs_review` is genuinely never
  cleared before `_project_calendar_event()` returns without raising — for the *exception* case.

That last point is also where the deep review found a real gap the inline review missed: a
**Critical** issue where `_project_calendar_event()`'s internal `sun_event()` `ValueError` handling
means the "never clear the flag on a projection failure" guarantee silently does not apply to the
single most likely real-world failure mode for this exact feature (see CR-01). This was
reproduced with a standalone test against the actual code (not just read) before being written up
and discarded, per verifier convention.

## Critical Issues

### CR-01: `resolve_site` silently reports success and permanently drops the retry surface when the newly-resolved site has no timezone set

**File:** `solsys_code/campaign_views.py:405-421` (`_project_calendar_event`'s `sun_event()`
`ValueError` handling) and `solsys_code/campaign_views.py:592-609` (`_resolve_site`'s
flag-clearing/messaging)

**Issue:** `_project_calendar_event()` catches `sun_event()`'s `ValueError` internally and returns
`False` *without raising* (`campaign_views.py:413-421`). `_resolve_site()` treats a `False`,
non-raising return identically to the deliberate "no projection needed" cases (range/TBD run,
missing telescope/site) — it unconditionally clears `site_needs_review` and shows the plain
success message `'Site resolved.'` (`campaign_views.py:604-609`). It only takes the "stay in the
retry surface" branch when `_project_calendar_event()` *raises* (the `except Exception:` block at
line ~594).

`sun_event()` raises exactly this `ValueError` whenever `Observatory.timezone` is blank
(`solsys_code/telescope_runs.py:254-257`). Crucially, `Observatory.timezone` defaults to `''`
(`solsys_code_observatory/models.py:55`) and `MPCObscodeFetcher.to_observatory()` — the function
`resolve_site()`'s Tier 2 calls for *any obscode not already a local `Observatory` row*
(`solsys_code_observatory/utils.py:85-124`) — never sets `timezone` on the `Observatory` it
creates. Tier 2 (MPC lookup creating a brand-new local `Observatory`) is precisely the common case
this phase's "Sites Needing Review" + Resolve feature exists to serve: an approved run whose
submitted site wasn't already a known local `Observatory`.

Net effect: resolving a ground-based run's still-unmatched site via a genuine, valid MPC obscode
will, in the ordinary case, silently fail to create the `CalendarEvent` — yet report
`'Site resolved.'` (implying full success) and permanently remove the row from the "Sites Needing
Review" table (the only surface gated on `site_needs_review=True`). There is no further way to
retry or even notice the missing calendar entry short of manually cross-checking the calendar.
This directly contradicts the feature's own stated design goal, documented in `_resolve_site()`'s
own docstring: *"a failed projection leaves the run visible in the Sites Needing Review table (its
retry surface) instead of vanishing into a dead end."* — the ordering guarantee holds only for
exceptions that propagate, not for this internally-swallowed `ValueError`, which is the realistic
failure mode for exactly the sites this feature is built to resolve.

Reproduced directly against the code (test written, run, and discarded per verifier convention):
patching `resolve_site()` to create a fresh Tier-2-style `Observatory` (blank `timezone`, matching
what `to_observatory()` actually produces) and POSTing `action=resolve_site` for a single-night
ground run yields `site_needs_review == False`, `CalendarEvent.objects.count() == 0`, and
`messages == ['Site resolved.']` — the silent-failure state described above, confirmed live.

**Fix:** Distinguish "skip by design" from "projection attempted but failed" in
`_project_calendar_event()`'s contract, and have `_resolve_site()` treat the latter like the
existing `except Exception` path (keep `site_needs_review=True`, warn instead of claiming
success). E.g., re-raise the `ValueError` from `_project_calendar_event()` instead of swallowing it
when reached via a resolved site (only swallow it in call sites that don't need the retry
guarantee, or thread a tri-state result through):

```python
# campaign_views.py, _project_calendar_event()
try:
    sunset, sunrise = sun_event(run.site, run.window_start, kind='sun')
except ValueError:
    logger.debug(
        'sun_event(sun) raised for site=%s date=%s; skipping projection.',
        run.site,
        run.window_start,
    )
    raise  # let callers that need the retry guarantee (resolve_site) see this as a failure
```

and drop the (now redundant) approve-branch's separate handling, or catch-and-swallow explicitly
only in the approve() call site where there's no retry surface to protect. At minimum,
`_resolve_site()` must not report `'Site resolved.'` (a claim about the calendar entry as well as
the site) when no `CalendarEvent` was actually created due to a real failure rather than a
by-design skip.

## Warnings

### WR-01: `ApprovalQueueTable.render_actions()`/`render_site()` ignore `show_actions` entirely in resolve mode

**File:** `solsys_code/campaign_tables.py:269, 286-297`

**Issue:** `render_site()`'s early-return guard is `if self.mode != 'resolve' and not self.show_actions: ...` and `render_actions()` checks `if self.mode == 'resolve': ...` *before* checking `self.show_actions` at all. Both mean a hypothetical `ApprovalQueueTable(..., mode='resolve', show_actions=False)` would still render a fully interactive Resolve form and live-search widget — `show_actions` is a complete no-op for resolve-mode tables. No current caller constructs that combination (the one `mode='resolve'` call site in `ApprovalQueueView` never passes `show_actions`), so this isn't exploitable today, but it's a latent footgun: a future maintainer adding a read-only "resolved sites" audit view by reusing this class with `mode='resolve', show_actions=False` (the natural-looking API) would silently get live action buttons instead of a read-only render.

**Fix:** Make `show_actions` gate resolve-mode rendering too, e.g. `if not self.show_actions: return ''` as the very first check in `render_actions()`, and mirror it in `render_site()`'s guard (`if not self.show_actions: return super().render_site(record)` before the mode check).

### WR-02: `SiteSearchView`'s per-IP throttle collapses to a single shared bucket when `REMOTE_ADDR` is absent

**File:** `solsys_code/campaign_views.py:731-738`, `solsys_code/campaign_utils.py:406-407`

**Issue:** `client_ip = request.META.get('REMOTE_ADDR', '')` defaults to the empty string when the key is missing, and `_check_and_increment_throttle` keys the cache on `f'site_search_throttle:{client_ip}'`. If `REMOTE_ADDR` is ever absent from `request.META` (e.g. certain ASGI/test-client configurations, or a misconfigured deployment in front of a proxy that strips it), every such anonymous request shares one throttle bucket (`site_search_throttle:`), so one client exhausting it would incorrectly 429 all other clients missing that header, or (more likely in practice) all clients behind the same reverse-proxy IP share one bucket regardless of the header being present — a known, documented limitation (Assumption A2: no `X-Forwarded-For` handling), but the *empty-string* collapse case specifically is a step further (silently merging distinct-but-header-less clients into the same key) and isn't mentioned in the docstring's caveat.

**Fix:** At minimum, document the empty-string case explicitly alongside the existing Assumption A2 note, or treat a missing `REMOTE_ADDR` as "no throttle key available" (e.g., skip throttling and log instead of silently sharing a bucket) so the failure mode is "no rate limit" rather than "cross-client interference."

## Info

### IN-01: `_resolve_site()`'s business-logic bypass guard has no dedicated non-staff/anonymous regression test

**File:** `solsys_code/tests/test_campaign_approval.py` (class `TestSitesNeedingReview`)

**Issue:** `TestStaffGating` exercises anonymous/non-staff access only against the `approve`/`reject` actions of `campaigns:decide`, not `resolve_site`. Because `StaffRequiredMixin` gates the whole view uniformly, this is very likely fine in practice, but there's no explicit test proving a non-staff POST with `action=resolve_site` is rejected the same way — a future refactor that moves `_resolve_site()` out from under `StaffRequiredMixin` (e.g. into its own view class) would not be caught by the existing test suite.

**Fix:** Add a `test_non_staff_post_resolve_site_redirects_and_makes_no_change` case mirroring the existing `approve`/`reject` staff-gating tests.

### IN-02: `_check_and_increment_throttle`'s cache backend note documents `FileBasedCache` non-atomicity but the project's actual default cache backend isn't stated

**File:** `solsys_code/campaign_utils.py:391-395`

**Issue:** The docstring reasons about `FileBasedCache`'s `incr()` non-atomicity specifically, but doesn't note which `CACHES` backend this project actually runs (e.g. `LocMemCache`, which is what the test suite exercises, has different atomicity characteristics — its `incr()` uses a lock and is effectively atomic within a single process, but not across multiple worker processes). A reader auditing throttle correctness has to go find `settings.py` to know which caveat actually applies in production; a one-line pointer (or an explicit "we assume backend X in production") would save that lookup.

**Fix:** Add a one-line cross-reference to the configured `CACHES` backend in `settings.py`, or state explicitly that the analysis is backend-agnostic (i.e., holds for both file-based and multi-process shared caches).

---

_Reviewed: 2026-07-15T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
