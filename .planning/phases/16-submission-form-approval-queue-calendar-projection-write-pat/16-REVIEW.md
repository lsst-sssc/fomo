---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
reviewed: 2026-07-04T14:07:32Z
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
  critical: 1
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 16: Code Review Report

**Reviewed:** 2026-07-04T14:07:32Z
**Depth:** deep
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Reviewed the public campaign-run submission form, the staff approval queue, the atomic
approve/reject decision endpoint, calendar projection on approval, and supporting
templates/tests at deep depth (cross-file call-chain tracing into `campaign_utils.resolve_site`,
`calendar_utils.insert_or_create_calendar_event`, and the `CalendarEvent`/`Observatory` models).

The three explicitly-called-out concerns were verified empirically, not just read:

- **Staff gating** (`StaffRequiredMixin` on both `ApprovalQueueView` and
  `CampaignRunDecisionView`, correct MRO ordering) — confirmed correct; anonymous/non-staff
  GET and POST both redirect with no state change (test suite + direct verification).
- **Atomic double-approve guard** (`.filter(pk=pk, approval_status=PENDING_REVIEW).update(...)`
  then `updated_count == 1`) — the concurrency-safety property itself is correct (single
  UPDATE statement, verified with a live test DB). However, see CR-01 below: the guard
  protects against *concurrent* double-processing, but the *side effects that follow it*
  (site resolution + calendar projection) are not covered by the same transaction, and a
  failure there leaves the row permanently stuck in a half-approved state that the guard
  itself then makes unrecoverable.
- **Honeypot trip path** — verified via both the existing test suite and direct reasoning:
  no `CampaignRun` row and no email are created when `alt_contact_info` is filled, and the
  redirect response is byte-identical to a genuine submission.
- **PII leakage into emails/templates** — verified: `_notify_staff`'s subject/body contain
  no contact/telescope/campaign data (only a bare link), and the non-staff table path
  excludes `contact_person`/`contact_email` at the queryset level (`.values()`), not just the
  rendered table.

One finding is classified Critical (a real, reproducible data-consistency defect, not a
theoretical one — reproduced against a live test database below). Three Warnings and two
Info items round out lower-severity gaps.

## Critical Issues

### CR-01: Approve side-effects are not transactionally coupled to the atomic status update — a mid-flow failure leaves the run permanently stuck "approved" with no calendar event and no recovery path

**File:** `solsys_code/campaign_views.py:279-308` (`CampaignRunDecisionView.post`)

**Issue:** The atomic conditional `.update()` (line 279-281) is its own auto-committed
statement. Everything that follows it on approve — `resolve_site(run.site_raw)` (a network
call to the MPC Obscodes API on a Tier-2 miss), `run.save(update_fields=[...])`, and
`insert_or_create_calendar_event(...)` (a further DB write) — runs *outside* any
`transaction.atomic()` block. If any of these raises (network timeout/DNS failure not
caught by `resolve_site`'s guards, a `sqlite3.OperationalError: database is locked` under
concurrent writes — a documented limitation of this project's dev DB per CLAUDE.md — or a
worker/request timeout while `resolve_site` is blocked on the MPC API call), the
`approval_status` transition to `APPROVED` has **already committed** and is never rolled
back. The row is now permanently `APPROVED` with `site`/`site_needs_review` unresolved and
no `CalendarEvent` ever created.

Worse, this state is **unrecoverable through the UI**: the double-approve guard
(`filter(..., approval_status=PENDING_REVIEW)`) means a second POST to the same `decide`
endpoint no longer matches the row (`updated_count == 0`), so it silently falls into the
"already decided by someone else" warning branch — the calendar projection can never be
retried. The run will appear as a normal "Approved" entry in the recently-decided table
with no visible indication that its calendar projection never happened, silently defeating
the entire purpose of this phase (CAL-01/02/03).

Reproduced directly against a live test database (staff-authenticated POST to
`/campaigns/<pk>/decide/` with `resolve_site` monkeypatched to raise `RuntimeError`):

```
status 500
approval_status after failure: approved
```

**Fix:** Wrap the whole post-update side-effect sequence (site resolution, save, and
calendar projection) in the same `transaction.atomic()` block as the conditional update, so
any exception rolls the `approval_status` change back to `PENDING_REVIEW` and the run can be
re-decided:

```python
def post(self, request, pk):
    action = request.POST.get('action')
    if action not in ('approve', 'reject'):
        return HttpResponseBadRequest()
    new_status = CampaignRun.ApprovalStatus.APPROVED if action == 'approve' else CampaignRun.ApprovalStatus.REJECTED

    with transaction.atomic():
        updated_count = CampaignRun.objects.filter(
            pk=pk, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
        ).update(approval_status=new_status)

        if updated_count == 1 and action == 'approve':
            run = CampaignRun.objects.select_for_update().get(pk=pk)
            site, needs_review = resolve_site(run.site_raw)
            run.site, run.site_needs_review = site, needs_review
            run.save(update_fields=['site', 'site_needs_review'])
            if run.telescope_instrument and run.ut_start and run.ut_end:
                insert_or_create_calendar_event(...)
    # messages / redirect outside the atomic block, after the transaction has either
    # fully committed or fully rolled back.
```

If a fully-atomic wrap is deemed too broad (e.g. the MPC network call inside a long-held
transaction is itself undesirable), at minimum catch exceptions from this block, explicitly
revert `approval_status` back to `PENDING_REVIEW` on failure, and surface an error message —
never leave `APPROVED` and "not projected" as a silent, permanent combination.

## Warnings

### WR-01: Decision endpoint returns a misleading "already decided by someone else" message for a `pk` that never existed

**File:** `solsys_code/campaign_views.py:279-312`

**Issue:** `CampaignRun.objects.filter(pk=pk, approval_status=PENDING_REVIEW).update(...)`
returns `updated_count == 0` both when the row exists but was already decided, and when
`pk` does not exist at all (deleted row, stale/bookmarked link, or a tampered URL). Both
cases fall into the same `else: messages.warning(request, 'This run was already decided by
someone else.')` branch, which is factually wrong for a nonexistent row. Reproduced
directly: POSTing `action=approve` to `/campaigns/99999/decide/` (no such `CampaignRun`)
returns HTTP 200 with exactly that message, rather than a 404. No test exercises this path
(`test_invalid_action_returns_bad_request` only covers a bad `action` value, not a bad `pk`).

**Fix:** Distinguish the two cases, e.g.:

```python
run_exists = CampaignRun.objects.filter(pk=pk).exists()
...
else:
    if run_exists:
        messages.warning(request, 'This run was already decided by someone else.')
    else:
        messages.error(request, 'This run no longer exists.')
```

or use `get_object_or_404(CampaignRun, pk=pk)` up front to fail fast with a real 404 before
attempting the conditional update.

### WR-02: Public submission form's `campaign` field exposes every `TargetList` in the system, not just actual campaigns

**File:** `solsys_code/campaign_forms.py:19`

**Issue:** `campaign = forms.ModelChoiceField(queryset=TargetList.objects.all(), required=True)`
lists **every** `TargetList` row in the entire TOM instance in an anonymous, unauthenticated
dropdown — including private saved searches or ad-hoc groupings that were never intended as
public "campaigns" (per `CampaignListView`'s own docstring, "campaign" is purely operational:
a `TargetList` with `campaign_runs__isnull=False`; there is no actual campaign flag). This
lets any anonymous visitor: (a) discover the names of every `TargetList` in the system via
the rendered `<select>`, and (b) attach a `PENDING_REVIEW` `CampaignRun` to any of them,
effectively bootstrapping an unrelated saved search into appearing on the public campaigns
list once approved (or as target-list spam even while pending, visible to staff in the
queue). No test exercises this — `test_missing_campaign_invalid` only checks that omitting
the field errors, not that the field's choices are scoped.

**Fix:** Scope the queryset to `TargetList` rows that are already legitimate campaigns (or a
curated subset), e.g.:

```python
campaign = forms.ModelChoiceField(
    queryset=TargetList.objects.filter(campaign_runs__isnull=False).distinct(),
    required=True,
)
```

(with a documented decision on how a *brand-new* campaign gets its first run submitted, if
that's a use case this form needs to support.)

### WR-03: `_notify_staff`'s `NoReverseMatch` fallback branch is now dead code, misleadingly implying a pending future state

**File:** `solsys_code/campaign_views.py:191-199`

**Issue:** The `try: reverse('campaigns:approval_queue') / except NoReverseMatch: ...`
fallback's comment explains it exists because "`campaigns:approval_queue` is added by Plan
03 (Wave 3), which has not landed yet at this plan's point in the phase's sequential wave
order." All four plans have now landed in this same reviewed changeset —
`solsys_code/campaign_urls.py:27` already defines `campaigns:approval_queue`, and
`src/fomo/urls.py:26` already mounts `campaign_urls` at `'campaigns/'`. `reverse(...)` will
therefore always succeed in the shipped codebase, making the `except NoReverseMatch` branch
permanently unreachable, untested, and now actively misleading to a future reader (it
describes a build-order constraint that no longer exists).

**Fix:** Remove the `try/except NoReverseMatch` fallback now that Plan 03 has landed, and
call `reverse('campaigns:approval_queue')` directly:

```python
queue_url = self.request.build_absolute_uri(reverse('campaigns:approval_queue'))
```

(Drop the now-unused `NoReverseMatch` import from the top of the file too.)

## Info

### IN-01: `CampaignRunSubmissionView.success_url` is dead code

**File:** `solsys_code/campaign_views.py:138`

**Issue:** `success_url = reverse_lazy('campaigns:submission_thanks')` is set at the class
level, but `form_valid` never calls `super().form_valid()` or references
`self.success_url`/`self.get_success_url()` — both the honeypot path and the genuine-create
path return `redirect('campaigns:submission_thanks')` explicitly. `success_url` is therefore
unused in practice; it only matters if some future refactor accidentally calls
`super().form_valid(form)`, in which case it would silently start working (or, if left
stale relative to a URL rename, silently redirect somewhere wrong).

**Fix:** Either remove `success_url` (since `form_valid` is fully overridden), or use it
consistently — e.g. `return redirect(self.success_url)` in both branches instead of the
literal string, so there is a single source of truth for the destination URL.

### IN-02: Honeypot `alt_contact_info` field has no `max_length`

**File:** `solsys_code/campaign_forms.py:32`

**Issue:** Every other free-text field on the form (`telescope_instrument`, `site_raw`,
`filters_bandpass`, `contact_person`) has an explicit `max_length=255`, matching the
corresponding model field's `CharField(max_length=255)`. `alt_contact_info` is a bare
`forms.CharField(required=False, widget=forms.HiddenInput())` with no bound — a bot (or a
manual POST) can submit an arbitrarily large value for this field. It is never persisted, so
this isn't a storage/overflow risk, but it is inconsistent with the rest of the form's
validation discipline and offers a trivial (if minor) unbounded-input vector on a public,
unauthenticated endpoint.

**Fix:** Add a reasonable `max_length` (e.g. `max_length=255`) for consistency, even though
the value is discarded rather than saved.

---

_Reviewed: 2026-07-04T14:07:32Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
