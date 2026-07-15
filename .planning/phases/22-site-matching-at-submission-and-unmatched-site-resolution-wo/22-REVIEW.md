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
  critical: 2
  warning: 3
  info: 1
  total: 6
status: issues_found
---

# Phase 22: Code Review Report

**Reviewed:** 2026-07-15
**Depth:** deep
**Files Reviewed:** 10
**Status:** issues_found

## Summary

This review covers all six plans of Phase 22 (22-01 through 22-06), including the three
gap-closure plans (22-04/05/06) executed to fix UAT bugs found in the first three plans, and
supersedes the prior 22-REVIEW.md (1 Critical + 2 Warning, all fixed per 22-REVIEW-FIX.md — those
findings (`_project_calendar_event()`'s swallowed `ValueError`, `show_actions`/resolve-mode
gating, the missing-`REMOTE_ADDR` throttle bucket) are confirmed fixed in the current code and
are not repeated here).

The three invariants this review was specifically asked to re-verify against the 22-06
gap-closure changes are all correctly preserved:

- **CR-01 (a genuinely-resolved projection-failed retry row must still render plain text):**
  `ApprovalQueueTable.render_site()`'s `is_correctable_placeholder` gate requires
  `is_placeholder_observatory(site_obj)` to be True, so a retry-state row whose site is a real,
  non-placeholder Observatory still falls through to `super().render_site(record)` (plain text).
  Confirmed by `test_retry_row_renders_plain_text_site_and_resolve_button_no_input`.
- **WR-01 (a read-only table with `show_actions=False` must never render an interactive
  widget):** both branches of `render_site()` gate on `self.show_actions` before rendering the
  widget, for both `mode='pending'` and `mode='resolve'`, including the new placeholder branch.
  Confirmed by `test_placeholder_row_read_only_table_never_renders_widget`.
- **D-06 (a genuinely-resolved site must never be silently re-resolved/replaced):**
  `_resolve_site()`'s guard is `if run.site is None or is_placeholder_observatory(run.site):` — a
  real, non-placeholder site skips the branch entirely and `resolve_site()` is never called for
  it. Confirmed by `test_genuine_site_still_never_re_resolved_when_replacing_placeholder_would_apply`.

However, tracing the placeholder-awareness that 22-06 introduces (`is_placeholder_observatory()`)
back to *how a site is decided to be "genuinely resolved" in the first place* surfaced a real,
provable gap that 22-06 did not close: that awareness lives only at the display/eligibility layer
(`render_site()`/`_resolve_site()`'s own guard), not inside `resolve_site()` itself or the local
candidate pool that feeds the correction widget. Together (CR-01/CR-02 below) these let a
Sites-Needing-Review row be silently "resolved" back to a still-placeholder site — clearing
`site_needs_review` and reporting success for a run whose site is still unusable — which is
exactly the D-06/D-08 invariant this phase exists to protect, reopened one layer down.

## Critical Issues

### CR-01: `resolve_site()` reports a placeholder hit as a genuine resolution (Tier 1, and the Tier 2 race-recovery re-fetch)

**File:** `solsys_code/campaign_utils.py:157-161` (Tier 1), also `:179` (Tier 2 success) and
`:192-193` (Tier 2 IntegrityError race-recovery re-fetch)

**Issue:** `resolve_site()`'s Tier 1 branch is:

```python
try:
    return Observatory.objects.get(obscode=code), False
except Observatory.DoesNotExist:
    pass
```

This returns `needs_review=False` for *any* existing `Observatory` row matching `code`,
including one that is itself a tier-3 placeholder (`NEEDS REVIEW: `-prefixed name, created by an
earlier `resolve_site()` call for the same obscode with `create_placeholder=True`). This is
directly reachable, not a narrow edge case:

1. `import_campaign_csv.py` calls `resolve_site(site_raw)` fresh, once per CSV row, in a loop
   (`solsys_code/management/commands/import_campaign_csv.py:161`), and creates every row with
   `approval_status=APPROVED` directly (bootstrap rows are pre-vetted — the approve-time
   site-resolution code in `campaign_views.py` never runs for them, since they're never
   `PENDING_REVIEW`). For a `Site Code` that repeats across multiple CSV rows — the ordinary
   case, e.g. a telescope observing the same still-unconfigured site across several nights —
   only the **first** row's `resolve_site()` call goes through Tier 3 (`needs_review=True`);
   every subsequent row's call hits Tier 1 against the placeholder Tier 3 just created, and gets
   `needs_review=False`. Those rows are created `APPROVED` with a bogus site (blank `timezone`,
   default `lat`/`lon`/`altitude`) and `site_needs_review=False`: they never appear in "Sites
   Needing Review" and never get a `CalendarEvent`, with no path in the UI to ever notice. This
   is precisely the dead end D-07/D-08 (this phase's stated purpose) was built to close,
   reopened for every CSV-imported run after the first one sharing an unresolved site code.
2. The same Tier 1 branch is reused by both `CampaignRunDecisionView.post()`'s approve branch
   (`campaign_views.py:496`) and `_resolve_site()` (`campaign_views.py:595`) via
   `resolve_site(obscode_selection, create_placeholder=False)`. If a staff member submits (or is
   *offered* — see CR-02 below) a `site_selection` value that maps to an obscode already carrying
   a placeholder Observatory, `resolve_site()` reports success (`needs_review=False`) even though
   nothing was actually corrected — `_resolve_site()` then clears `site_needs_review` and reports
   "Site resolved — run added to the calendar." (or "Site resolved.") for a run still pointing at
   the same unusable placeholder.

No existing test exercises "Tier 1 hits a pre-existing placeholder" — every `resolve_site()` /
`TestApprovalSiteResolution` / `TestPlaceholderSiteReplacement` test either starts from an empty
`Observatory` table or fixtures a genuinely-resolved Observatory, so this shipped undetected
through three gap-closure plans of test-writing (see IN-01 below).

**Fix:** Every success path in `resolve_site()` should derive `needs_review` from whether the
matched/returned Observatory is itself a placeholder, not report `False` unconditionally:

```python
# Tier 1
try:
    obs = Observatory.objects.get(obscode=code)
except Observatory.DoesNotExist:
    pass
else:
    return obs, is_placeholder_observatory(obs)
```

Apply the same correction to the Tier 2 IntegrityError race-recovery re-fetch at line 193 (it can
race against a concurrent Tier 3 create for the same code, so the re-fetched row can be a
placeholder). The Tier 2 success path at line 179 (`fetcher.to_observatory()`) never itself
produces a placeholder-named row, so it's lower-value to change, but doing so anyway keeps every
success path consistent. `is_placeholder_observatory` is defined later in the same module
(`campaign_utils.py:222`), which is fine — it only needs to exist by call time, not definition
time.

### CR-02: Placeholder Observatories pollute the site-search candidate pool used by the correction widget

**File:** `solsys_code/campaign_utils.py:263-278` (`_local_observatory_candidates()`)

**Issue:**

```python
def _local_observatory_candidates() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for obs in Observatory.objects.all():
        for candidate in (obs.obscode, obs.name or '', obs.short_name or '', obs.old_names or ''):
            if candidate and candidate not in mapping:
                mapping[candidate] = obs.obscode
    return mapping
```

`Observatory.objects.all()` includes tier-3 placeholder rows (e.g. `name='NEEDS REVIEW: DCT'`,
`short_name='DCT'`, `obscode='DCT'`). These candidates flow unfiltered into
`build_site_candidates()`'s merged pool, which backs *both* the public submission form's live
search and the approval-queue/Sites-Needing-Review correction widget
(`SiteSearchView`/`substring_or_fuzzy_match_candidates`). Concretely: once a placeholder
`'NEEDS REVIEW: DCT'` (obscode `DCT`) exists, typing `dct` (or `needs review`) into *any*
site-search box — including that exact run's own "Sites Needing Review" correction widget —
surfaces `"NEEDS REVIEW: DCT (DCT)"` as a clickable suggestion. Clicking it fills the input with
that literal placeholder display string; submitting it maps back to obscode `DCT` via
`build_site_candidates().get(selection, selection)` and calls
`resolve_site('DCT', create_placeholder=False)`, which (pre-CR-01-fix) resolves via Tier 1 back
to the *very same placeholder* and reports success. `site_needs_review` is then cleared and
"Site resolved — run added to the calendar." (or "Site resolved.") is shown, even though the site
was never actually corrected — precisely the scenario 22-06 was written to prevent (UAT gap 2B),
now reachable *through* the correction UI itself rather than around it.

**Fix:** Exclude placeholders from the local candidate pool, e.g.:

```python
def _local_observatory_candidates() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for obs in Observatory.objects.exclude(name__startswith=NEEDS_REVIEW_NAME_PREFIX):
        for candidate in (obs.obscode, obs.name or '', obs.short_name or '', obs.old_names or ''):
            if candidate and candidate not in mapping:
                mapping[candidate] = obs.obscode
    return mapping
```

This is complementary to CR-01's fix, not redundant with it: CR-02 stops the placeholder from
being *offered* as a suggestion at all; CR-01 stops it from being silently *accepted* as a
genuine resolution if a staff member types its obscode directly (e.g. from memory, without using
the widget) or a stale/cached suggestion slips through.

## Warnings

### WR-01: `CampaignRunDecisionView.post()`'s approve branch isn't placeholder-aware, unlike `_resolve_site()`

**File:** `solsys_code/campaign_views.py:475`

**Issue:** 22-06 taught `_resolve_site()` to treat `run.site is None or
is_placeholder_observatory(run.site)` as "not yet genuinely resolved"
(`campaign_views.py:587`), but the approve branch's own inline site-resolution still uses only
`if run.site is None:` (line 475). In every currently-reachable state this self-heals — a
placeholder's `observations_type` defaults to `OPTICAL_OBSTYPE` and `timezone` defaults to `''`,
so `_project_calendar_event()` always raises `ValueError` for a placeholder-sited run, leaving
`site_needs_review` untouched and the row visible in "Sites Needing Review" — but it's a fragile
coincidence, not a designed invariant, and it will start silently under-resolving once CR-01 is
fixed at the `resolve_site()` layer: a run whose `site` is already a CSV-imported placeholder
(not `None`) would never re-enter the resolution branch here even though `is_placeholder_observatory`
would now correctly say it isn't genuinely resolved.

**Fix:** Mirror `_resolve_site()`'s guard here: `if run.site is None or
is_placeholder_observatory(run.site):`.

### WR-02: Placeholder detection is a magic string prefix on a user-editable field

**File:** `solsys_code/campaign_utils.py:222-236` (`is_placeholder_observatory()`)

**Issue:** "Is this Observatory a placeholder" is decided purely by
`observatory.name.startswith('NEEDS REVIEW: ')` — a convention on a plain, staff-editable
`CharField`, not a dedicated model flag. Nothing in `Observatory`/`CreateObservatoryForm`
prevents a staff member from creating (or renaming) a genuine, fully-configured Observatory whose
name happens to start with that exact string (e.g. copy-pasting a Sites-Needing-Review row's
display text into "Create new Observatory" without editing it). Such a record would be
permanently treated by `render_site()`/`_resolve_site()` as an eligible-for-replacement
placeholder no matter how correctly it's configured — a genuinely-resolved site perpetually
reopened for "correction", the inverse of D-06's intent.

**Fix:** Either add a dedicated boolean field (e.g. `Observatory.is_placeholder`), set explicitly
by `resolve_site()`'s tier-3 branch and checked by `is_placeholder_observatory()` instead of a
string prefix, or validate/reject the reserved `NEEDS_REVIEW_NAME_PREFIX` in
`CreateObservatoryForm` so it can never be assigned to a manually-created record.

### WR-03: Replaced placeholder Observatory rows are never cleaned up

**File:** `solsys_code/campaign_views.py:621-626` (`_resolve_site()`'s claim update)

**Issue:** When `_resolve_site()` successfully replaces a placeholder with a real site, the
conditional `.update(site=site)` repoints the `CampaignRun`, but the orphaned placeholder
`Observatory` row itself is never deleted or flagged as superseded. It stays in the database
indefinitely, continuing to satisfy `is_placeholder_observatory()` and (until CR-02 is fixed)
continuing to pollute the search pool for the next, unrelated resolution attempt.

**Fix:** Not a hard blocker, but worth a follow-up: delete the orphaned placeholder (guarding
against any other `CampaignRun` still referencing it) once a replacement succeeds, or track
superseded placeholders for periodic cleanup.

## Info

### IN-01: No test exercises `resolve_site()` hitting a pre-existing placeholder via Tier 1

**File:** `solsys_code/tests/test_campaign_approval.py`

**Issue:** Every existing `resolve_site()` test (`TestApprovalSiteResolution`,
`TestPlaceholderSiteReplacement`, `TestIsPlaceholderObservatory`) either starts from an empty
`Observatory` table or fixtures a genuinely-resolved Observatory. None fixtures an *existing
placeholder* (`Observatory.objects.create(name=f'{NEEDS_REVIEW_NAME_PREFIX}...')`) and then calls
`resolve_site()` again for the same obscode — precisely the CR-01/CR-02 scenario. This gap in
coverage is why the bug shipped undetected through three gap-closure plans of test-writing aimed
squarely at this exact placeholder-vs-genuine distinction.

**Fix:** Add a regression test such as:

```python
def test_resolve_site_tier1_hit_on_existing_placeholder_still_flags_review(self):
    Observatory.objects.create(obscode='DCT', name=f'{NEEDS_REVIEW_NAME_PREFIX}DCT', short_name='DCT')
    site, needs_review = resolve_site('DCT', create_placeholder=False)
    self.assertTrue(needs_review)  # currently fails: returns False
```

---

_Reviewed: 2026-07-15_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
