---
phase: 25-range-window-calendarevent-projection-allow-approved-site-re
reviewed: 2026-07-18T00:00:00Z
depth: deep
files_reviewed: 4
files_reviewed_list:
  - solsys_code/campaign_views.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/management/commands/backfill_range_calendar_events.py
  - solsys_code/tests/test_backfill_range_calendar_events.py
findings:
  critical: 0
  warning: 7
  info: 1
  total: 8
status: issues_found
---

# Phase 25: Code Review Report

**Reviewed:** 2026-07-18T00:00:00Z
**Depth:** deep
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed `campaign_views.py`'s new range-window `CalendarEvent` projection
(`_project_calendar_event()`'s per-night loop, `_set_run_status()`'s combined-key event
update, and `_resolve_site()`'s placeholder-aware re-resolution), the paired
`backfill_range_calendar_events` management command, and both test files, at deep depth
(cross-file call-chain tracing between the view helper, the backfill command, and
`telescope_runs.sun_event()`/`calendar_utils.insert_or_create_calendar_event()`).

`diff_base` in the config resolves to an unrelated `gh-pages` deploy commit with no merge
base to this branch, so it could not be used to scope a real diff; per the workflow's
documented fallback order, the explicit `files:` list was used directly and each file
was reviewed in full.

No crash-level or security-level defects were found in the reviewed range-window logic
itself — the per-night loop, the combined bare-key/prefixed-key `CalendarEvent` lookups,
and the placeholder-replacement/orphan-cleanup guards in `_resolve_site()` all match
their documented invariants and are exercised by the (large, thorough) test suite. The
issues found are logic/robustness gaps in the **backfill command's bookkeeping** (it
silently miscounts and under-clears state that the equivalent UI path handles correctly),
a **window-ordering** edge case that can produce a silently-empty-but-"successful"
projection, an **orphaned-event** edge case on a specific revert-then-reject sequence, and
one **test-reliability** gap where a test's own assertions can't actually distinguish the
behavior it claims to verify.

## Warnings

### WR-01: Backfill command ignores `_project_calendar_event()`'s return value, over-counting successes

**File:** `solsys_code/management/commands/backfill_range_calendar_events.py:81-88`
**Issue:** `_project_calendar_event()` (`solsys_code/campaign_views.py:404-430`) returns
`False` — without raising — when its top-level guard fails:
```python
if not (run.telescope_instrument and run.site and run.window_start and run.window_end):
    return False
```
The backfill command's candidate query (`backfill_range_calendar_events.py:50-56`) filters
on `site__isnull=False` and `window_start__isnull=False` (and the model's
`campaign_run_window_start_end_null_together` `CheckConstraint` guarantees `window_end`
is then also non-null), but it does **not** filter on `telescope_instrument` being
non-blank. `CampaignRun.telescope_instrument` is a `CharField` with no `null=True`/
`blank=True`/default guard at the DB level (confirmed directly by
`test_campaign_approval.py:428-433`'s `telescope_instrument=''` fixture, which the ORM
happily persists). For a range-window `APPROVED`, site-resolved run with a blank
`telescope_instrument`, `_project_calendar_event(run)` returns `False` silently — but the
command does:
```python
try:
    _project_calendar_event(run)
    backfilled_count += 1
except ValueError as exc:
    ...
```
never inspecting the return value, so `backfilled_count` (and the `--dry-run` "Would
backfill..." message) is incremented/printed even though zero `CalendarEvent`s were
created. Operators reading the summary line get a false positive.
**Fix:**
```python
projected = _project_calendar_event(run)
if projected:
    backfilled_count += 1
else:
    skipped_count += 1  # or a new counter, e.g. skipped_incomplete_count
```

### WR-02: Backfill command never clears `site_needs_review`, unlike the UI's `_resolve_site()`

**File:** `solsys_code/management/commands/backfill_range_calendar_events.py:38-103`
**Issue:** The candidate query at lines 50-56 filters only on `approval_status=APPROVED`,
`site__isnull=False`, and a resolved range window — it does not exclude
`site_needs_review=True` rows. A range-window run can legitimately be in that state (the
"projection-failed retry state" `campaign_views.py:521-533`/`642-646` documents at length,
and that `ApprovalQueueView`'s `review_table` surfaces for staff). When the backfill
command successfully projects such a run's `CalendarEvent`s (line 82,
`_project_calendar_event(run)`), it never does the equivalent of `_resolve_site()`'s
final step (`campaign_views.py:731-733`):
```python
run.site_needs_review = False
run.save(update_fields=['site_needs_review'])
```
so the run stays listed in the "Sites Needing Review — action required" queue
(`ApprovalQueueView`, `campaign_views.py:345-364`) forever after a successful backfill,
even though its calendar entries now exist. This is exactly the class of "dead end" state
Phase 22 (`is_placeholder_observatory`/finding 8c, referenced throughout
`campaign_views.py`) was written to eliminate for the UI path — the backfill command
reintroduces it for any candidate it processes that happens to have
`site_needs_review=True`. Untested: every fixture in
`test_backfill_range_calendar_events.py` uses `_make_approved_run()`, which leaves
`site_needs_review` at its model default (`False`); no test exercises a
`site_needs_review=True` candidate.
**Fix:** after a successful (non-raising) `_project_calendar_event(run)` call, mirror
`_resolve_site()`:
```python
_project_calendar_event(run)
if run.site_needs_review:
    run.site_needs_review = False
    run.save(update_fields=['site_needs_review'])
backfilled_count += 1
```

### WR-03: No window-ordering validation — a reversed range silently "succeeds" with zero events

**File:** `solsys_code/campaign_views.py:468-487`
**Issue:**
```python
n_nights = (run.window_end - run.window_start).days + 1
is_range = n_nights > 1
for i in range(n_nights):
    ...
return True
```
If `window_end < window_start`, `n_nights <= 0`, `range(n_nights)` iterates zero times,
and the function still unconditionally `return True` — the documented contract is "True
when `insert_or_create_calendar_event()` was actually called," but here it's called zero
times. Callers rely on this bool: `_resolve_site()` (`campaign_views.py:721,734-737`)
reports "Site resolved — run added to the calendar." and clears `site_needs_review`, both
incorrectly, for a run with zero actual calendar coverage. The model's
`campaign_run_window_start_end_null_together` `CheckConstraint`
(`solsys_code/models.py:153-160`) only enforces that the two fields are null together or
set together — it does **not** enforce `window_end >= window_start` — and
`window_start`/`window_end` are editable (not `readonly_fields`) in
`CampaignRunAdmin` (`solsys_code/admin.py:6-27`), so a reversed range is directly
reachable by a staff user editing the row in Django admin, bypassing
`campaign_utils.parse_obs_window()`'s own reversed-range-to-TBD guard
(`campaign_utils.py:690-694`) that the normal submission/CSV-import paths go through.
**Fix:**
```python
if n_nights <= 0:
    logger.warning('CampaignRun %s has window_end < window_start (%s..%s); skipping projection.',
                    run.pk, run.window_start, run.window_end)
    return False
```

### WR-04: Backfill idempotency check can't distinguish "complete" from "partially projected then failed"

**File:** `solsys_code/management/commands/backfill_range_calendar_events.py:63-71`
**Issue:**
```python
already = CalendarEvent.objects.filter(
    Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')
).exists()
if already:
    skipped_count += 1
    continue
```
`_project_calendar_event()`'s ground-based loop deliberately accepts partial projection
on a mid-window `sun_event()` `ValueError` (`campaign_views.py:462-467`, "A mid-loop
`sun_event()` ValueError re-raises immediately, leaving any already-created earlier
nights' events in place"). If that happens during a `backfill_range_calendar_events` run,
the earlier nights' `CalendarEvent` rows now exist, so on any future re-run of this same
command the `already` check above is `True` and the run is silently skipped as
"already backfilled" — permanently — even though it's missing events for the later
nights. Unlike the UI retry surface (`site_needs_review=True` rows staying listed in
"Sites Needing Review"), a range-window backfill candidate carries no equivalent
persistent flag, so there is no operator-facing indication that a partially-failed
candidate needs attention beyond the one-time stderr line printed on the run that first
hit the failure. `test_backfill_skips_and_continues_on_sun_event_valueerror`
(`test_backfill_range_calendar_events.py:116-124`) only exercises a `ValueError` that
fires on every call (full failure, zero events for both runs), so this partial-then-stuck
interaction is entirely untested.
**Fix:** either (a) have the command compare the actual per-night event count against
`n_nights` rather than a bare existence check, so a partial candidate is retried, or (b)
report partial candidates distinctly (e.g. a `partial_count` bucket) so operators know to
investigate rather than assume "skipped == done".

### WR-05: Orphaned CalendarEvents can persist after a revert-then-reject sequence

**File:** `solsys_code/campaign_views.py:518-598`
**Issue:** On approve, if `_project_calendar_event(run)` raises a **non**-`ValueError`
exception partway through a range-window projection (e.g. `insert_or_create_calendar_event`
itself failing on a later night — the documented "MAY RAISE on any other unexpected
failure" case, `campaign_views.py:410-418`), the earlier nights' `CalendarEvent`s are
already committed (no `transaction.atomic()` wrap, by design). The outer
`except Exception:` block (lines 580-595) then reverts `approval_status` back to
`PENDING_REVIEW`, but leaves `run.site` set and the already-created partial
`CalendarEvent`s in place. If a staff member subsequently re-approves the reverted run,
`insert_or_create_calendar_event()`'s upsert semantics self-heal this correctly (existing
events get updated/left unchanged, remaining nights get created). But if staff instead
choose **reject** on the reverted (now-`PENDING_REVIEW`-again) run, the reject branch
(lines 596-598, `elif updated_count == 1: messages.success(request, 'Run rejected.')`)
performs no calendar cleanup at all — the earlier-created `CalendarEvent` rows remain in
the calendar, now attached (via their `CAMPAIGN:{pk}[:date]` URL) to a run whose
`approval_status` is `REJECTED`. This is a narrow but real scenario: it requires an
unexpected mid-range projection failure followed by a staff "reject" (rather than
re-approve) decision, but nothing in the reject path guards against it, and no test
covers a reject following a reverted partial projection.
**Fix:** on `reject`, delete any `CalendarEvent`s matching the same combined
bare-key/prefixed-key lookup `_set_run_status()` already uses (`campaign_views.py:788-790`)
before/after the conditional status update, so a rejected run can never retain stray
calendar entries regardless of how it got there.

### WR-06: No upper bound on range-window size before synchronous per-night projection

**File:** `solsys_code/campaign_views.py:468-486`; `solsys_code/management/commands/backfill_range_calendar_events.py:81-88`
**Issue:** Neither `CampaignRunSubmissionForm`'s `clean()` (`campaign_forms.py:89-114`,
via `parse_obs_window()`) nor `_project_calendar_event()` itself impose any maximum on
`(window_end - window_start).days`. Each night in the loop calls
`telescope_runs.sun_event()`, which performs an iterative coarse-scan-plus-bisection
root-find over a 24h window (`telescope_runs.py:_find_crossing`) — not free. A
mistakenly-large window (a typo'd year, or a staff `site_selection`/admin edit producing
an accidentally huge range) would make a single `approve`/`resolve_site` POST, or a single
`backfill_range_calendar_events` candidate, run hundreds or thousands of these
computations synchronously and block that request/command thread with no validation
catching the obviously-wrong size earlier in the pipeline.
**Fix:** validate a sane maximum window length (e.g. a few hundred nights) in
`CampaignRunSubmissionForm.clean()` and/or as an explicit guard at the top of
`_project_calendar_event()`'s ground branch, returning a form error / logged skip rather
than silently accepting an unbounded range.

### WR-07 (test reliability): `test_backfill_skips_and_continues_on_sun_event_valueerror` can't prove what it claims

**File:** `solsys_code/tests/test_backfill_range_calendar_events.py:116-124`
**Issue:**
```python
with patch('solsys_code.campaign_views.sun_event', side_effect=ValueError('blank timezone')):
    call_command('backfill_range_calendar_events', stdout=StringIO(), stderr=StringIO())

self.assertEqual(self._event_count(run_a), 0)
self.assertEqual(self._event_count(run_b), 0)
```
The `stdout=StringIO()`/`stderr=StringIO()` arguments are anonymous — never assigned to a
variable, so their contents are never inspected. The test's name and the module docstring
(`test_backfill_range_calendar_events.py:6-8`, "a per-candidate `sun_event()` ValueError
is reported and skipped, never aborting the whole backfill run") claim this proves the
command *continues* to the next candidate after a failure. But because `sun_event` is
patched to raise unconditionally for every call, the observed result — both `run_a` and
`run_b` end up with 0 events — is identical whether the command (a) processes each
candidate independently and both happen to fail, or (b) aborts entirely after `run_a`'s
first failure (e.g. via a hypothetical regression that turns the per-candidate
`try/except` into a `return`/`raise` instead of `continue`). Since `call_command()` is not
wrapped in `assertRaises`, a regression of that kind would still let this test pass, as
long as no exception propagates out of `handle()`. As written, the test proves "the
command doesn't crash" and "no events exist for either run," but not "the command
continued past `run_a`'s failure to actually attempt `run_b`."
**Fix:** capture the stdout in a variable and assert on the summary counts (e.g.
`self.assertIn('failed: 2', out.getvalue())`), or give `run_a`/`run_b` distinguishable
`side_effect`s (e.g. fail only for `run_a`'s dates, succeed for `run_b`'s) and assert
`run_b` *does* get its 4 events — that positively proves continuation rather than merely
being consistent with early abort.

## Info

### IN-01: Duplicated Observatory fixture block across 6 test classes

**File:** `solsys_code/tests/test_campaign_approval.py:194-203, 318-327, 465-474, 900-909, 1268-1277, 1670-1679`
**Issue:** The identical `Observatory.objects.create(obscode='F65', name='Faulkes
Telescope South', short_name='FTS', lat=-31.2727, lon=149.0644, altitude=1149.0,
timezone='Australia/Sydney', observations_type=Observatory.OPTICAL_OBSTYPE)` block is
copy-pasted verbatim into six different `setUpTestData()`/`setUp()` methods
(`TestApproval`, `TestCalendarProjection`, `TestRunStatusChange`,
`TestSitesNeedingReview`, `TestPlaceholderSiteReplacement`, `TestCalendarNoChurn`). Any
future change to this fixture shape (e.g. a new required `Observatory` field) requires
editing all six call sites in lockstep.
**Fix:** extract a small module-level helper, e.g. `_create_fts_ground_site()`, and call
it from each `setUpTestData()`/`setUp()`.

---

_Reviewed: 2026-07-18T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
