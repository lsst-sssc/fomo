---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
verified: 2026-07-15T18:00:00Z
status: human_needed
score: 20/20 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "On the public 'Submit an Observing Run' form, type 'faulkes' (2+ chars) into the Observing site field in a real browser and wait ~300ms."
    expected: "An hx-get fires to /campaigns/site-search/, a suggestion list appears below the field showing both Faulkes sites as 'Display Name (obscode)', and clicking one fills the input with that exact text. Typing 1 character does nothing; no request fires before 2 characters."
    why_human: "htmx's hx-trigger debounce/length-filter grammar and the onclick fill-in behavior execute in the browser's JS engine; Django's test client never runs JavaScript, so the markup-level tests (correct hx-trigger string, correct ids) prove the widget is wired correctly but cannot prove the browser actually debounces, fires, and fills as intended end-to-end."
  - test: "In the staff approval queue, use the pending-row site input and the Sites Needing Review row's site input to search and pick a suggestion, then submit Approve/Resolve."
    expected: "Suggestions render and fill the correct row's input (not a different row's), the 'Create new Observatory' link still works, and submitting resolves/approves as expected."
    why_human: "Multiple ApprovalQueueTable rows on one page share one endpoint and per-row id-suffixed containers; confirming no cross-row leakage in a live multi-row table is a rendering/interaction check outside what grep/Django TestCase assertions can observe."
  - test: "Resolve a Sites Needing Review row end-to-end against a real (or realistic Tier-2) MPC obscode with a blank Observatory.timezone and confirm the CR-01 fix's user-facing behavior: a warning is shown, the row stays in the table, and a later Resolve retries."
    expected: "The banner and row persistence behave as CR-01's fix describes when driven through the actual UI, not just the regression test's direct POST."
    why_human: "The regression test (test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event) proves the model/message state via a direct POST; a human pass over the real page confirms the message banner and table re-render look correct together in situ."
---

# Phase 22: Site Matching at Submission and Unmatched-Site Resolution Workflow Verification Report

**Phase Goal:** Close the Phase 21 functionality gap: submitters and staff get live fuzzy matching against the merged local `Observatory` + full MPC candidate pool wherever a site is entered, and approved runs with unresolved sites get a resolution workflow instead of a dead end.
**Verified:** 2026-07-15T18:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

All 20 must-haves declared across the three plans' frontmatter (`must_haves.truths`) were checked directly against the current codebase (not SUMMARY.md claims), cross-referenced with a fresh, independent full-suite test run and a fresh `ruff` run.

| # | Truth (Plan) | Status | Evidence |
|---|---|---|---|
| 1 | Anonymous GET to `campaigns:site_search` returns 200 HTML fragment (D-01/D-03) (P01) | VERIFIED | `SiteSearchView` at `campaign_views.py:743` has no `StaffRequiredMixin`, `http_method_names=['get']`, renders `site_search_results.html` via `render()`. Test suite green. |
| 2 | Suggestion renders as "Display Name (obscode)" (D-05) (P01) | VERIFIED | `site_search_results.html:26` — `{{ display }} ({{ obscode }})` visible text node; onclick sets the same text via escaped vars. |
| 3 | `substring_or_fuzzy_match_candidates('faulkes', pool)` returns both Faulkes sites (P01) | VERIFIED | `campaign_utils.py:337` implements substring-containment-first, sorted-shortest-first, `limit=8`; test `test_substring_hit_surfaces_all_faulkes_candidates` passing in full suite. |
| 4 | Zero-substring-hit query falls back to `fuzzy_match_candidates()` (P01) | VERIFIED | Same function; falls back to `fuzzy_match_candidates(text, candidate_pool)[:limit]` when containment is empty; dedicated fallback test passing. |
| 5 | (LIMIT+1)th request from one IP returns 429 (P01) | VERIFIED | `_check_and_increment_throttle()` at `campaign_utils.py:383` (`cache.add`/`incr`, `SITE_SEARCH_THROTTLE_LIMIT=40`); `SiteSearchView.get()` returns `HttpResponse(status=429)` when exceeded; tests passing. |
| 6 | Blank/1-char query returns empty fragment WITHOUT calling `build_site_candidates()` (finding 4) (P01) | VERIFIED | `campaign_views.py:792-798` — the `len(query.strip()) < 2` gate returns before `substring_or_fuzzy_match_candidates(query, build_site_candidates())` is ever reached; mock-based assert_not_called test present and passing. |
| 7 | Hostile `input_id` replaced server-side + both onclick occurrences `\|escapejs`-escaped (finding 2) (P01) | VERIFIED | `_INPUT_ID_RE = re.compile(r'^[-A-Za-z0-9_:.]+$')` at `campaign_views.py:60`, `fullmatch` guard at line 784; `site_search_results.html` has 5 `\|escapejs` occurrences (input_id x2, display, obscode, and the container-clear line). |
| 8 | Public form's `site_raw` renders hx-get with corrected trigger, no Create-new link (D-09) (P02) | VERIFIED | `campaign_forms.py:37-41` — `hx-get`/`hx-trigger='input[this.value.length >= 2] changed delay:300ms'`/`hx-target`/`hx-vals`; `grep -c "Create new Observatory" campaign_forms.py` = 0. |
| 9 | Approval-queue pending row's site input is a live-search widget, keeps Create-new link (D-10) (P02) | VERIFIED | `campaign_tables.py:212-245` `_render_site_search_widget()` emits `hx-get`, the corrected trigger, and `<a ... >Create new Observatory</a>`; `grep -c "fuzzy_match_candidates"` / `datalist` = 0 (datalist branch fully removed). |
| 10 | Three-way id wiring (hx-vals input_id = input id = suggestions-container suffix) (finding 7) (P02) | VERIFIED | Both surfaces derive `input_id`/`container_id`/`hx-vals` from one shared string (`id_site_raw` on the form; `site-input-{pk}` in `_render_site_search_widget`). |
| 11 | Resolved rows / decided table keep plain-text `render_site` fallback (P02) | VERIFIED | `render_site()` early-returns `super().render_site(record)` when `site_short_name` is set or `not self.show_actions` (`campaign_tables.py:266-274`). |
| 12 | `campaign_tables.py` no longer imports `fuzzy_match_candidates` (finding 8) (P02) | VERIFIED | Import block at `campaign_tables.py:10-17` has no such import; `grep -c` = 0. |
| 13 | Third "Sites Needing Review" table lists APPROVED + `site_needs_review=True` runs, reusing the once-per-request `candidate_pool` (D-07) (P03) | VERIFIED | `campaign_views.py:325,345-357` — `candidate_pool = build_site_candidates()` computed once and passed to both `pending_table` and `review_table`; `review_qs` filters exactly `approval_status=APPROVED, site_needs_review=True`, no row cap. |
| 14 | POST `action=resolve_site` resolves via pool-mapping + `resolve_site(create_placeholder=False)`, then fires deferred projection (D-08) (P03) | VERIFIED | `_resolve_site()` at `campaign_views.py:548-634` implements this exact flow; regression test `test_resolve_success_single_night_ground_run_projects_calendar_event` passing. |
| 15 | Projection failure saves site but KEEPS `site_needs_review=True`, run stays APPROVED (finding 3) (P03) | VERIFIED | `_resolve_site()` lines 612-625 — non-reverting `except Exception`, flag never touched on failure; test `test_resolve_retryable_projection_failure_stays_approved_site_saved_flag_stays_true` passing; **also independently confirmed the CR-01 follow-up fix closes the specific blank-`timezone` failure mode** (`test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event` — see Review Fix Verification below). |
| 16 | `site_needs_review` flips to False ONLY after `_project_calendar_event()` returns without raising (finding 3) (P03) | VERIFIED | Lines 627-629 — flag clear is strictly after the `try/except` around `_project_calendar_event(run)`. |
| 17 | Site write is a single conditional queryset update (concurrency-safe) (finding 5) (P03) | VERIFIED | Lines 599-609 — `.filter(pk=pk, approval_status=APPROVED, site_needs_review=True, site__isnull=True).update(site=site)`; `claimed == 0` path warns and returns without projecting; `test_resolve_lost_race_no_op_warns` passing. |
| 18 | `resolve_site()` never re-resolves an already-set site; retry re-attempts projection only; non-eligible run rejected with a warning (finding 8) (P03) | VERIFIED | Lines 569-573 (`if run.site is None:` guard) + line 565 (state-validation guard); tests `test_resolve_never_re_resolves_already_set_site_but_retries_projection`, `test_resolve_rejects_pending_review_run`, `test_resolve_rejects_already_resolved_run` all passing. |
| 19 | `_project_calendar_event()` returns bool distinguishing created vs. skipped-by-design (finding 6) (P03) | VERIFIED | `campaign_views.py:370` `-> bool` signature; returns `True` after both `insert_or_create_calendar_event()` call sites, `False` on the by-design skip guard; drives the two distinct success messages at lines 630-633. |
| 20 | Single-night ground run gets dip-corrected event; range/TBD run clears flag with no event (P03) | VERIFIED | `_project_calendar_event()` guard (`window_start == window_end`) + `sun_event(..., kind='sun')` branch; `test_resolve_range_tbd_run_clears_flag_with_no_calendar_event` passing. |

**Score:** 20/20 truths verified (0 present-but-behavior-unverified)

### Review-Fix Verification (Critical + 2 Warnings)

Per the given context, all 3 findings from 22-REVIEW.md (deep-depth review) were claimed fixed in 22-REVIEW-FIX.md. Each was independently re-checked against the current code (not the fix report's narrative):

| Finding | Commit | Independently confirmed in current code? |
|---|---|---|
| **CR-01** (Critical): `_project_calendar_event()` swallowed `sun_event()`'s `ValueError` on a blank-`timezone` (Tier-2) site, silently reporting success and dropping the retry surface | `b1aae9f` | **Confirmed.** `campaign_views.py:420-429` now `raise`s instead of `return False`; `approve()`'s call site (lines 511-518) explicitly catches-and-swallows only that `ValueError` to preserve prior approve behavior; `_resolve_site()`'s existing non-reverting `except Exception` (lines 616-625) now genuinely catches it. Regression test `test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event` present and passing in the full suite. |
| **WR-01** (Warning): `show_actions` was a no-op in resolve mode — a hypothetical `mode='resolve', show_actions=False` table would still render live widgets | `5d40764` | **Confirmed.** `render_actions()` now has `if not self.show_actions: return ''` as its first line (`campaign_tables.py:288`); `render_site()`'s guard is now the unconditional `if not self.show_actions:` (line 273), independent of `self.mode`. |
| **WR-02** (Warning): missing `REMOTE_ADDR` collapsed to an empty-string cache key, cross-throttling unrelated anonymous clients | `a679f4d` | **Confirmed.** `SiteSearchView.get()` (`campaign_views.py:760-777`) no longer defaults to `''`; a falsy `client_ip` now skips throttling and logs a warning instead of sharing one bucket. |

All three fixes are real code changes present on the current tree, not just claims in the fix report.

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `solsys_code/campaign_utils.py :: substring_or_fuzzy_match_candidates()` | Substring-first matcher | VERIFIED | Line 337, below `fuzzy_match_candidates()` (line 306) — helper not deleted/renamed. |
| `solsys_code/campaign_utils.py :: _check_and_increment_throttle()` + constants | Per-IP throttle | VERIFIED | Lines 379-425. |
| `solsys_code/campaign_views.py :: SiteSearchView` | Anonymous throttled endpoint | VERIFIED | Line 743. |
| `solsys_code/campaign_views.py :: _INPUT_ID_RE` | Allowlist regex | VERIFIED | Line 60. |
| `solsys_code/campaign_urls.py :: path('site-search/', ...)` | URL route | VERIFIED | Line 30, `name='site_search'`. |
| `src/templates/campaigns/partials/site_search_results.html` | Suggestion fragment | VERIFIED | Present; 5 `\|escapejs` occurrences. |
| `solsys_code/tests/test_campaign_site_search.py` | New test module | VERIFIED | Present; exercised in the 471-test full-suite run. |
| `solsys_code/campaign_forms.py :: CampaignRunSubmissionForm.site_raw` widget | HTMX widget, no create-link | VERIFIED | Lines 31-42, 132-133. |
| `solsys_code/campaign_tables.py :: ApprovalQueueTable.render_site()` HTMX widget | Pending-row widget swap | VERIFIED | Lines 247-281. |
| `solsys_code/campaign_views.py :: _project_calendar_event()` | Extracted bool-returning helper | VERIFIED | Line 370. |
| `solsys_code/campaign_views.py :: CampaignRunDecisionView.post()` resolve_site branch | New decision action | VERIFIED | Lines 453-456, `_resolve_site()` at 548-634. |
| `solsys_code/campaign_views.py :: ApprovalQueueView.get_context_data()` review_table | Third table | VERIFIED | Lines 325, 345-366. |
| `solsys_code/campaign_tables.py` resolve-mode render_site()/render_actions() | Resolve-mode rendering | VERIFIED | Lines 201-314. |
| `src/templates/campaigns/approval_queue.html` Sites Needing Review block | Third-table render | VERIFIED | Lines 14-15. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `campaigns:site_search` URL | `SiteSearchView.as_view()` | URL resolver | WIRED | Confirmed via `campaign_urls.py:30` and passing `reverse()` calls throughout the test suite. |
| `SiteSearchView.get()` | `substring_or_fuzzy_match_candidates()` + `build_site_candidates()` | Direct call, gated by throttle + min-length | WIRED | `campaign_views.py:792-800`. |
| `site_raw` widget (form) / render_site() (table) | `campaigns:site_search` | `reverse_lazy`/`reverse` in `hx-get` | WIRED | `campaign_forms.py:37`, `campaign_tables.py:226`. |
| `ApprovalQueueView.get_context_data()` review_qs | `review_table` | `ApprovalQueueTable(..., mode='resolve', candidate_pool=candidate_pool)` | WIRED | `campaign_views.py:345-357`, reuses the single `candidate_pool` (no second `build_site_candidates()` call). |
| `CampaignRunDecisionView.post()` resolve_site | conditional site-claim update | `.filter(...).update(site=site)` | WIRED | `campaign_views.py:599-604`. |
| resolve_site claim | `_project_calendar_event(run)` | direct call inside non-reverting try/except | WIRED | `campaign_views.py:616-625`. |
| approve branch | `_project_calendar_event(run)` | direct call inside UNCHANGED revert-on-failure except | WIRED | `campaign_views.py:511-518` (return value ignored, per design). |

### Requirements Coverage

Phase 22 has no formal `.planning/REQUIREMENTS.md` IDs. Confirmed by inspection: `REQUIREMENTS.md` lists 13 v1 requirements (SCHED/ASSET/IMPORT/SITE/VIEW), all mapped to Phases 18-21 and marked Complete, with a footer noting it was "Last updated 2026-07-05 after roadmap creation (Phases 18-21 mapped)." `ROADMAP.md` confirms Phase 22 was added 2026-07-14 — after that requirements pass — specifically to close the Phase 21 site-matching functionality gap ("Requirements: TBD" in the Phase 22 roadmap entry). This is a genuine scope gap in the requirements doc (a phase added post-hoc to close a functionality gap, not a new v1/v2 requirement), not an oversight by this phase's planning — the phase instead tracks D-01..D-10 in 22-CONTEXT.md, which was exhaustively cross-checked against the code above. No orphaned or unmapped requirement IDs exist for Phase 22 to report.

### Anti-Patterns Found

Scanned all files modified across the three plans (`campaign_utils.py`, `campaign_views.py`, `campaign_urls.py`, `campaign_forms.py`, `campaign_tables.py`, the two test files, `site_search_results.html`, `approval_queue.html`) for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER` and empty-implementation patterns.

No blocking anti-patterns found. All `TBD` occurrences are the pre-existing domain concept ("TBD run" = a campaign run with no scheduled window yet), not a debt marker — none reference unresolved work in this phase's own code. No `FIXME`/`XXX`/`HACK`/`PLACEHOLDER` matches in any touched file.

### Behavioral Spot-Checks / Independent Test Re-Run

- Full `solsys_code` Django test suite independently re-run by this verifier (not the executor/fixer): `python manage.py test solsys_code` → **471 tests, all pass** (matches the orchestrator's post-fix re-run claim).
- Targeted re-run of the three campaign test modules: `python manage.py test solsys_code.tests.test_campaign_site_search solsys_code.tests.test_campaign_submission solsys_code.tests.test_campaign_approval` → **101 tests, all pass**.
- `ruff check` on all touched Python files → clean.
- `ruff format --check` on all touched Python files → clean (8 files already formatted).

### Human Verification Required

The code-level and Django-test-level evidence for all 20 must-haves and all 3 review-fix items is thorough and independently reproduced. The one class of behavior this cannot prove is actual browser-executed htmx interaction (Django's test client never runs JavaScript) — the exact hx-trigger grammar was previously the subject of a review finding (22-REVIEWS.md finding 1) that the pre-implementation planning caught only through careful hand-tracing of htmx's documented grammar, not through a browser test. Three items are flagged for a human UAT pass (see frontmatter `human_verification` for full detail):

1. Public form live-search: typing 'faulkes' actually debounces, fires, and renders/fills suggestions in a real browser.
2. Multi-row approval-queue behavior: per-row id-suffixed widgets don't cross-fill between rows in a live multi-row table.
3. Sites Needing Review resolve UX: the CR-01 fix's warning/retry banner reads correctly in situ, driven through the real UI rather than a direct test-client POST.

### Gaps Summary

None. All 20 declared must-haves across the three PLAN.md files are verified present, substantive, and wired in the current codebase — not merely claimed in SUMMARY.md. All three code-review findings (1 Critical, 2 Warning) are confirmed fixed in the current code, not just in the fix report's narrative. The full 471-test Django suite passes on an independent re-run, and `ruff check`/`ruff format --check` are clean. The only open item is browser-level UAT of the htmx interaction, which no static or Django-test-client check can substitute for.

---

_Verified: 2026-07-15T18:00:00Z_
_Verifier: Claude (gsd-verifier)_
