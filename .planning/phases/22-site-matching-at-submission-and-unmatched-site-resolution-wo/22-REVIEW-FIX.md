---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
fixed_at: 2026-07-15T14:31:05Z
review_path: .planning/phases/22-site-matching-at-submission-and-unmatched-site-resolution-wo/22-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 22: Code Review Fix Report

**Fixed at:** 2026-07-15T14:31:05Z
**Source review:** .planning/phases/22-site-matching-at-submission-and-unmatched-site-resolution-wo/22-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3 (CR-01, WR-01, WR-02 -- IN-01/IN-02 out of scope for this run)
- Fixed: 3
- Skipped: 0

## Fixed Issues

### CR-01: `resolve_site` silently reports success and permanently drops the retry surface when the newly-resolved site has no timezone set

**Files modified:** `solsys_code/campaign_views.py`, `solsys_code/tests/test_campaign_approval.py`
**Commit:** `b1aae9f`
**Applied fix:** `_project_calendar_event()` no longer swallows `sun_event()`'s `ValueError`
into a bare `return False` -- it now re-raises, so the exception genuinely propagates to
callers. `_resolve_site()`'s existing non-reverting `except Exception:` block (already
correct) now actually catches this case, keeping `site_needs_review=True` and warning
instead of claiming `'Site resolved.'`. The `approve()` call site has no retry surface to
protect (unlike the "Sites Needing Review" table), so it wraps its
`_project_calendar_event(run)` call in its own `try/except ValueError` to preserve its
original behavior (approval still succeeds silently without a `CalendarEvent` on this
specific, expected failure mode); anything else `_project_calendar_event()` raises still
falls through to the broader revert-to-`PENDING_REVIEW` except block, unchanged.
Updated `_project_calendar_event()`'s docstring and the ground-based-observatory comment
block to describe the new raise-vs-return-False contract. Added a regression test,
`test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event`, that fixtures a
local Observatory with a blank `timezone` (matching what
`MPCObscodeFetcher.to_observatory()` actually produces for Tier 2) and drives the real
`resolve_site` POST path end-to-end, asserting `site_needs_review` stays `True`, no
`CalendarEvent` is created, the message is not `'Site resolved.'`, and the run remains
listed in `review_table`.
Re-ran `solsys_code.tests.test_campaign_approval` (61 tests), the combined
`test_campaign_approval`/`test_campaign_site_search`/`test_campaign_submission` run (101
tests), and the full `solsys_code` suite (471 tests) -- all pass. `ruff check`/`ruff format
--check` clean on both touched files.

### WR-01: `ApprovalQueueTable.render_actions()`/`render_site()` ignore `show_actions` entirely in resolve mode

**Files modified:** `solsys_code/campaign_tables.py`
**Commit:** `5d40764`
**Applied fix:** Added `if not self.show_actions: return ''` as the first check in
`render_actions()` (before the `self.mode == 'resolve'` branch), and changed
`render_site()`'s guard from `if self.mode != 'resolve' and not self.show_actions:` to
`if not self.show_actions:` so a read-only table (`show_actions=False`) falls back to the
plain-text render regardless of `mode`. Removed the now-redundant second
`show_actions` check later in `render_actions()`'s pending-mode branch. Updated
`render_site()`'s docstring to describe the new uniform gating instead of the old
mode-conditional early-return. Re-ran `solsys_code.tests.test_campaign_approval` (61
tests, all pass) and the full `solsys_code` suite (471 tests, all pass). `ruff
check`/`ruff format --check` clean.

### WR-02: `SiteSearchView`'s per-IP throttle collapses distinct clients into a single shared bucket when `REMOTE_ADDR` is absent

**Files modified:** `solsys_code/campaign_views.py`, `solsys_code/campaign_utils.py`
**Commit:** `a679f4d`
**Applied fix:** `SiteSearchView.get()` no longer defaults a missing `REMOTE_ADDR` to the
empty string before throttling. It now checks `client_ip` truthiness: when present, the
existing `_check_and_increment_throttle(client_ip)` path runs unchanged; when absent
(falsy), it skips throttling entirely for that request and logs a warning, so the failure
mode becomes "no rate limit for this request" rather than "silently shares one
`site_search_throttle:` bucket across every client missing that header." Updated
`_check_and_increment_throttle()`'s docstring in `campaign_utils.py` to document that
callers must never pass a falsy `client_ip` and to explain why (this closes the gap
noted in the finding: the empty-string collapse case wasn't previously called out
alongside the existing Assumption A2 caveat). Re-ran
`solsys_code.tests.test_campaign_site_search` (17 tests) and the full `solsys_code` suite
(471 tests) -- all pass. `ruff check`/`ruff format --check` clean.

## Skipped Issues

None -- all in-scope findings were fixed.

---

_Fixed: 2026-07-15T14:31:05Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
