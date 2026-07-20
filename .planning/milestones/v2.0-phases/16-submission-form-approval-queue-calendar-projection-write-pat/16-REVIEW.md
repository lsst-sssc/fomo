---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
reviewed: 2026-07-04T00:00:00Z
depth: deep
files_reviewed: 15
files_reviewed_list:
  - solsys_code/campaign_forms.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_urls.py
  - solsys_code/campaign_views.py
  - solsys_code/mixins.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_forms.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_campaign_views.py
  - src/fomo/settings.py
  - src/templates/campaigns/approval_queue.html
  - src/templates/campaigns/campaign_list.html
  - src/templates/campaigns/campaignrun_submit_form.html
  - src/templates/campaigns/campaignrun_table.html
  - src/templates/campaigns/submission_thanks.html
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---

# Phase 16: Code Review Report

**Reviewed:** 2026-07-04T00:00:00Z
**Depth:** deep
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Re-review of the full phase-16 file scope (plans 16-01 through 16-05) after the 16-05
gap-closure plan landed. 16-05's own change — `ApprovalQueueTable.Meta` gaining an
`exclude=('weather', 'observation_outcome', 'publication_plans')` + a `sequence` that
front-loads `actions`/`approval_status` — was traced against the installed
`django_tables2` source (`TableOptions` explicitly applies `exclude` on top of inherited
`fields`, so combining them is a supported code path, not an accidental foot-gun) and
verified empirically: `TestApprovalQueueColumns` passes, and the full listed test suite
(61 tests across the four `test_campaign_*` modules) passes, `ruff check`/`ruff format
--check` are clean on every reviewed `solsys_code`/template file. `CampaignRunTable`
itself is untouched by 16-05, confirmed both by reading the diff and by
`test_campaign_run_table_unchanged_by_approval_queue_trim`.

Cross-file call-chain tracing (`CampaignRunDecisionView.post` → `resolve_site` →
`insert_or_create_calendar_event`, and `ApprovalQueueTable.render_actions` →
`reverse('campaigns:decide')` → `CampaignRunDecisionView`) turned up no new Critical
issues, but did surface one residual defect in the **already-applied** CR-01 fix from the
prior review cycle (16-REVIEW-FIX.md), plus a persistent test-coverage gap around that
same fix, and one previously-identified-and-consciously-deferred issue that is still live
in the shipped code. No new Critical/blocking defects were found in this pass; findings
below are Warning/Info.

## Warnings

### WR-01: CR-01's approve-failure recovery reverts `approval_status` but leaves `site`/`site_needs_review` already committed — partial, misleading "reset" state

**File:** `solsys_code/campaign_views.py:282-323` (`CampaignRunDecisionView.post`)

**Issue:** The prior review cycle's CR-01 fix wraps `resolve_site()` + `run.save(update_fields=['site', 'site_needs_review'])` + `insert_or_create_calendar_event(...)` in a `try/except Exception`, and on failure reverts only `approval_status` back to `PENDING_REVIEW`:

```python
run.site, run.site_needs_review = site, needs_review
run.save(update_fields=['site', 'site_needs_review'])          # <-- already committed
...
insert_or_create_calendar_event(...)                             # <-- raises here
except Exception:
    ...
    CampaignRun.objects.filter(pk=pk).update(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
    messages.error(request, '... This run has been reset to pending review -- please try again.')
```

If `resolve_site`/`run.save()` succeed but `insert_or_create_calendar_event` then raises (e.g. a
DB write failure), the row ends up `PENDING_REVIEW` with `site`/`site_needs_review` already
populated from the resolution attempt (and, in the tier-3 fallback case, a real placeholder
`Observatory` row may already have been created as a side effect of `resolve_site`). This
contradicts the rest of the codebase's invariant that site resolution happens "at approval
time" (D-07) — a pending row now shows a resolved site in the approval queue's Site column
(`ApprovalQueueTable.render_site`), which will look inconsistent next to every other pending
row (which correctly shows blank). The user-facing message ("reset to pending review") also
overstates what actually happened: only one of the two side effects was rolled back.

Practical severity is limited (retrying is idempotent — `resolve_site`'s tier-1 lookup will
find the just-created `Observatory` on the next attempt), which is why this is a Warning
rather than a Critical, but it's a real, reproducible gap in the fix that shipped for the
previously-flagged CR-01.

**Fix:** Either revert the site fields too in the `except` block, or (better) restructure so
`run.save()` only happens after the calendar-projection step has succeeded:

```python
except Exception:
    logger.exception(...)
    CampaignRun.objects.filter(pk=pk).update(
        approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        site=None,
        site_needs_review=False,
    )
```

or move `run.save(update_fields=['site', 'site_needs_review'])` to after
`insert_or_create_calendar_event(...)` succeeds (or fold both into one `run.save()` call at the
end of the `try` block), so a failure partway through never leaves a persisted half-committed
field.

### WR-02: No test exercises the CR-01 exception/revert path (still true after the 16-05 gap closure)

**File:** `solsys_code/tests/test_campaign_approval.py` (whole file; would naturally live near
`TestCalendarProjection`/`TestCalendarNoChurn`, lines 140-221)

**Issue:** The prior REVIEW-FIX.md explicitly flagged this as needing follow-up ("no existing
test exercising the `resolve_site`-raises path ... consider adding a regression test"). This
re-review confirms the gap is still open: no test in `test_campaign_approval.py` (or anywhere
else in the reviewed scope) monkeypatches/mocks `resolve_site` or
`insert_or_create_calendar_event` to raise and asserts the revert-to-`PENDING_REVIEW` +
`messages.error` behavior actually fires. The only evidence this path works is the reviewer's
own ad hoc reproduction from the previous cycle (not committed as a test), so a future refactor
of `CampaignRunDecisionView.post` could silently break this safety net with no test failing.
It's also the code path directly implicated by WR-01 above — testing it would have caught that
partial-revert gap immediately.

**Fix:** Add a test that monkeypatches `solsys_code.campaign_views.resolve_site` (or
`insert_or_create_calendar_event`) to raise, POSTs an approve action, and asserts: (a) the
response redirects with an error message, (b) `run.approval_status` is back to
`PENDING_REVIEW`, and (c) — once WR-01 is fixed — `run.site`/`run.site_needs_review` are also
reverted.

### WR-03: Public submission form's `campaign` field still exposes every `TargetList` in the system (carried forward, deliberately deferred)

**File:** `solsys_code/campaign_forms.py:19`

**Issue:** `campaign = forms.ModelChoiceField(queryset=TargetList.objects.all(), required=True)`
still lists every `TargetList` row in the system (including unrelated saved searches/groupings
that were never intended as public campaigns) in an anonymous, unauthenticated dropdown. This
was raised as WR-02 in the prior review cycle and explicitly left unfixed per
16-REVIEW-FIX.md's documented rationale (scoping the queryset to
`campaign_runs__isnull=False` would break the supported "submit the first run for a brand-new
campaign" workflow, and there is no `TargetList.is_public_campaign`-style flag to distinguish
a legitimate campaign-in-waiting from an arbitrary private `TargetList`). That rationale is
sound given the current schema, so this is re-surfaced here only to confirm the underlying
privacy/discovery risk (any anonymous visitor can enumerate every `TargetList` name in the
system and attach a `PENDING_REVIEW` run to any of them) is still live in the shipped code, not
a false-positive from a stale review — it still has no dedicated test and no tracked follow-up
task in the reviewed scope.

**Fix:** As previously recommended: a schema-level fix (e.g. a `TargetList.is_public_campaign`
flag, or a separate allow-list model) routed through phase planning rather than a mechanical
code-review fix, since narrowing the queryset naively would regress the new-campaign workflow.

## Info

### IN-01: `CampaignRunSubmissionView.success_url` is still dead code

**File:** `solsys_code/campaign_views.py:142`

**Issue:** Unchanged from the prior review (IN-01, out of the fixer's `critical_warning` scope
last time): `success_url = reverse_lazy('campaigns:submission_thanks')` is declared but never
read — both `form_valid` branches call `redirect('campaigns:submission_thanks')` directly, so a
future URL-name rename could update `success_url` while leaving the actual redirect target
stale (or vice versa), with nothing to catch the divergence.

**Fix:** Remove `success_url`, or use `redirect(self.success_url)` in both branches so there is
one source of truth.

### IN-02: Honeypot `alt_contact_info` field still has no `max_length`

**File:** `solsys_code/campaign_forms.py:32`

**Issue:** Unchanged from the prior review (IN-02). Every other free-text field on the form has
an explicit `max_length=255` matching its model counterpart; `alt_contact_info` is a bare
`forms.CharField(required=False, widget=forms.HiddenInput())` with no bound, so a bot or manual
POST can submit an arbitrarily large value. Low risk (never persisted), but inconsistent with
the rest of the form's validation discipline.

**Fix:** Add `max_length=255` for consistency.

### IN-03: CSRF-token minting in `ApprovalQueueTable.render_actions` is never exercised with CSRF enforcement enabled

**File:** `solsys_code/campaign_tables.py:171-194`

**Issue:** `render_actions` mints a per-row CSRF token via `get_token(self.request)` and embeds
it as a hidden `csrfmiddlewaretoken` input in each Approve/Reject mini-form — the only CSRF
protection these forms get, since they aren't rendered via `{% csrf_token %}` in the template.
None of the tests in `test_campaign_approval.py` (or elsewhere in scope) use
`Client(enforce_csrf_checks=True)`; Django's default test client disables CSRF checking
entirely, so every `self.client.post(...)` call in the suite would pass even if
`render_actions` emitted an empty or stale token. The reasoning that this works is sound (this
is documented Django public API usage), but it is unverified by the test suite itself.

**Fix:** Add at least one test using `Client(enforce_csrf_checks=True)` that renders the
approval-queue page, extracts the token from the rendered Approve form, and confirms a POST
with that token succeeds (and, ideally, that a POST without it is rejected with 403).

### IN-04: WR-01's "run no longer exists" branch (from the prior review cycle) has no dedicated test

**File:** `solsys_code/campaign_views.py:327-334`

**Issue:** The prior review's WR-01 fix (distinguishing an already-decided row from a
nonexistent `pk` in the decision endpoint's fallback branch) shipped in
`solsys_code/campaign_views.py`, but `test_campaign_approval.py` still has no test posting to
`campaigns:decide` with a `pk` that doesn't exist. `test_invalid_action_returns_bad_request`
only covers a bad `action` value against a real, existing row — the `messages.error(request,
'This run no longer exists.')` branch (line 334) is currently reachable only by manual
reasoning/inspection, not by any test in the suite.

**Fix:** Add a test posting `{'action': 'approve'}` to `reverse('campaigns:decide', kwargs=
{'pk': 999999})` (a nonexistent pk) as a staff user, asserting the redirect and the
`'This run no longer exists.'` message.

---

_Reviewed: 2026-07-04T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
