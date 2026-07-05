---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
fixed_at: 2026-07-04T14:30:00Z
review_path: .planning/phases/16-submission-form-approval-queue-calendar-projection-write-pat/16-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 3
skipped: 1
status: partial
---

# Phase 16: Code Review Fix Report

**Fixed at:** 2026-07-04T14:30:00Z
**Source review:** .planning/phases/16-submission-form-approval-queue-calendar-projection-write-pat/16-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (CR-01, WR-01, WR-02, WR-03 -- fix_scope=critical_warning; IN-01/IN-02 out of scope)
- Fixed: 3
- Skipped: 1

## Fixed Issues

### CR-01: Approve side-effects are not transactionally coupled to the atomic status update

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** `4241b69`
**Applied fix:** Used the review's documented fallback (not the fully-atomic-wrap suggestion --
see below) since wrapping `resolve_site()`'s network call to the MPC Obscodes API inside a held
`transaction.atomic()` block would trade one bug (a permanently-stuck-approved row) for another
(a held DB transaction spanning an external HTTP call, worse under this project's SQLite dev DB
per CLAUDE.md's documented concurrent-write limitation). Instead, wrapped `resolve_site()` +
`run.save()` + `insert_or_create_calendar_event()` in a `try/except Exception` block: on any
failure, `CampaignRun.objects.filter(pk=pk).update(approval_status=PENDING_REVIEW)` explicitly
reverts the row so the double-approve guard no longer makes the half-approved state permanent,
logs the exception (`logger.exception`, matching this project's `logging.getLogger(__name__)`
convention), and surfaces a `messages.error(...)` to the staff user instead of a silent 500/stuck
row. Added `import logging` + a module-level `logger` (this file had no logger previously).

**Note:** this is an error-handling/logic change with no existing test exercising the
`resolve_site`-raises path (the reviewer's own reproduction used a monkeypatch, not a committed
test). Flagging per verification_strategy guidance: **fixed: requires human verification** --
please confirm the revert-to-pending-review semantics are the desired recovery behavior (vs., a
different failure UX) and consider adding a regression test for this path.

### WR-01: Decision endpoint returns a misleading "already decided" message for a nonexistent pk

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** `53358f8`
**Applied fix:** Split the `else` branch (where `updated_count == 0`) into two cases using the
review's first suggested approach: `CampaignRun.objects.filter(pk=pk).exists()` distinguishes a
row that exists but was already decided (`messages.warning(...)`, unchanged wording) from a `pk`
that never existed / was deleted (`messages.error(request, 'This run no longer exists.')`).
Chose this over the review's alternative (`get_object_or_404` up front, returning a real 404)
to preserve the existing redirect-with-message UX for staff clicking a stale approval-queue
link, rather than changing the response shape for that path.

### WR-03: `_notify_staff`'s `NoReverseMatch` fallback is now dead code

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** `a2975c9`
**Applied fix:** Removed the `try/except NoReverseMatch` fallback around
`reverse('campaigns:approval_queue')` and the hardcoded `/campaigns/approval-queue/` fallback
URL, calling `reverse(...)` directly, exactly as REVIEW.md's fix suggested. Also removed the
now-unused `NoReverseMatch` import from `django.urls` (kept `reverse`, `reverse_lazy`).

## Skipped Issues

### WR-02: Public submission form's `campaign` field exposes every `TargetList` in the system

**File:** `solsys_code/campaign_forms.py:19`
**Reason:** Applying REVIEW.md's literal suggested fix (`queryset=TargetList.objects.filter(
campaign_runs__isnull=False).distinct()`) would break the phase's core, intentionally-designed
workflow: a brand-new campaign's *first-ever* `CampaignRun` is submitted against a `TargetList`
that, by definition, has zero existing `CampaignRun`s at submission time (there is no
`is_campaign` flag on `TargetList` -- per this phase's own `CampaignListView` docstring and
`16-CONTEXT.md` D-10, "campaign" is purely operational: a `TargetList` becomes one only once it
has >= 1 `CampaignRun`). Scoping the form's queryset that way would make it impossible to ever
submit a run for a genuinely new campaign through the public form -- a functional regression,
not a fix -- and would immediately break the existing test suite's own bootstrap fixture pattern
(`test_campaign_submission.py::test_minimal_valid_submission_creates_pending_run` and four
sibling tests in `test_campaign_forms.py` all create a fresh `TargetList` with zero
`CampaignRun`s and then submit the form's first run against it in the same test).

REVIEW.md's own fix note anticipates exactly this tension: "(with a documented decision on how a
*brand-new* campaign gets its first run submitted, if that's a use case this form needs to
support.)" -- confirmed via `16-CONTEXT.md` that it is a supported use case (D-10 explicitly
describes a campaign appearing in the public list before any run is approved). Resolving the
underlying privacy/discovery concern (any anonymous visitor can see and attach a
`PENDING_REVIEW` run to *any* `TargetList` in the system, including private saved searches
unrelated to campaigns) needs either a new model field (e.g. `TargetList.is_public_campaign`) or
a separate allow-list mechanism -- a schema/product decision out of scope for a mechanical
code-review fix pass. Left unfixed; recommend routing this to phase planning (a follow-up
`/gsd:quick` task or a discussion note) rather than resolving it unilaterally here.

**Original issue:** `campaign = forms.ModelChoiceField(queryset=TargetList.objects.all(),
required=True)` lists every `TargetList` in the system in an anonymous, unauthenticated dropdown,
letting any visitor discover private saved-search names and attach a `CampaignRun` to any of
them.

## Verification

- `ruff check solsys_code/campaign_views.py` and `ruff format --check solsys_code/campaign_views.py`: clean.
- `python manage.py test solsys_code` (full Django suite, 300 tests): **OK**, all passing after
  all three fixes were applied, run in the isolated fix worktree.

---

_Fixed: 2026-07-04T14:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
