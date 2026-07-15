---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
verified: 2026-07-15T23:00:00Z
status: passed
score: 40/40 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 20/20 must-haves verified (3 human_verification items outstanding)
  gaps_closed:
    - "Public form + approval-queue + Sites Needing Review live-search widgets render suggestions via a real browser-reachable query-param path (SiteSearchView.get() now resolves q -> site_raw -> site_selection; UAT tests 1 and 3, fixed by 22-04-PLAN.md, commit dba220d)"
    - "Sites Needing Review section has distinct actionable visual weight (border-warning card) while preserving D-07's locked pending/decided/sites-needing-review order (UAT test 2A, fixed by 22-05-PLAN.md, commit 936f565)"
    - "A placeholder Observatory (e.g. 'NEEDS REVIEW: DCT') can be corrected through the Sites Needing Review UI instead of being permanently stuck as a plain-text retry-only row (UAT test 2B, fixed by 22-06-PLAN.md, commits ef97bd2/03bb0e9/7bd649e)"
    - "resolve_site() no longer misreports a Tier-1 hit on a pre-existing placeholder as a genuine resolution (22-REVIEW.md CR-01, fixed by commit bd80c0d)"
    - "Placeholder Observatories no longer pollute the site-search candidate pool used by both the public form and the correction widget (22-REVIEW.md CR-02, fixed by commit c36657c)"
    - "The approve branch is now placeholder-aware, mirroring _resolve_site()'s guard (22-REVIEW.md WR-01, fixed by commit ed432ec)"
    - "Placeholder detection no longer relies solely on an unguarded magic-string convention -- Observatory.clean() now rejects the reserved prefix on any form-validated save (22-REVIEW.md WR-02, fixed by commit d6bc732)"
    - "A replaced placeholder Observatory row is deleted once nothing references it, so it stops polluting the search pool (22-REVIEW.md WR-03, fixed by commit 63b7cef)"
  gaps_remaining: []
  regressions: []
---

# Phase 22: Site Matching at Submission and Unmatched-Site Resolution Workflow Verification Report

**Phase Goal:** Close the Phase 21 functionality gap: submitters and staff get live fuzzy matching against the merged local `Observatory` + full MPC candidate pool wherever a site is entered, and approved runs with unresolved sites get a resolution workflow instead of a dead end.
**Verified:** 2026-07-15T23:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (22-04/22-05/22-06) and a second deep code-review-and-fix cycle (22-REVIEW.md / 22-REVIEW-FIX.md)

## Context for This Pass

This is the **third** look at Phase 22:

1. Initial verification (first `22-VERIFICATION.md`, superseded by this file): 20/20 must-haves from plans 22-01/02/03 verified at the code level, but flagged `human_needed` because Django's test client cannot execute htmx/JS in a real browser.
2. Human UAT (`22-UAT.md`) then found 3 real bugs the code-level pass could not see: a query-param mismatch that silently broke every live-search widget in the browser, poor visual grouping of the "Sites Needing Review" table, and no correction path for a placeholder Observatory. These were diagnosed (`.planning/debug/resolved/*.md`) and closed by gap-closure plans 22-04, 22-05, 22-06.
3. A follow-up deep code review (`22-REVIEW.md`), re-run specifically against the 22-06 changes, found 2 **new** Critical issues one layer down (`resolve_site()` misreporting a placeholder hit as genuine; placeholders polluting the search candidate pool) plus 3 Warnings. All 5 were fixed (`22-REVIEW-FIX.md`, commits `bd80c0d`/`c36657c`/`ed432ec`/`d6bc732`/`63b7cef`).

This pass independently re-verifies the **entire chain** — not just the newest fixes — against the current codebase, re-running the full test suite and re-reading every touched file rather than trusting any SUMMARY/REVIEW-FIX narrative.

## Goal Achievement

### Observable Truths — Original 20 (Plans 22-01/02/03, regression check)

All 20 must-haves verified in the original pass were re-checked for regression (existence + wiring, not re-deriving from scratch since they were already fully verified and are unaffected by the gap-closure/review-fix commits unless noted).

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1-7 | P01 live-search foundation (anonymous endpoint, D-05 display format, substring-first matcher + difflib fallback, 429 throttle, min-length gate, input_id allowlist + escapejs) | VERIFIED (regression) | `SiteSearchView` (`campaign_views.py:789-868`), `substring_or_fuzzy_match_candidates()`/`_check_and_increment_throttle()` (`campaign_utils.py:384-472`) all present and unchanged in shape; `site_search_results.html` still has the escapejs-guarded onclick. Confirmed via a fresh full-suite run (see below), not just file presence. |
| 8-12 | P02 widget wiring (public form hx-get/no create-link, queue widget hx-get/keeps create-link, three-way id wiring, decided-table plain-text fallback, no `fuzzy_match_candidates` import in tables) | VERIFIED (regression) | `campaign_forms.py:31-47` (site_raw widget, 0 "Create new Observatory" occurrences), `campaign_tables.py:213-246` (`_render_site_search_widget`, keeps the link), `render_site()` early-return for `show_actions=False`/genuine site (`campaign_tables.py:271-289`). |
| 13-20 | P03 post-approval resolution (third table reusing one candidate_pool, `resolve_site` action + projection, non-reverting failure keeps flag True, flag clears only after success, conditional-update concurrency guard, never-re-resolve, bool-returning `_project_calendar_event`, single-night vs range/TBD behavior) | VERIFIED (regression) | `ApprovalQueueView.get_context_data()` (`campaign_views.py:302-368`), `_project_calendar_event()` (`:371-434`), `CampaignRunDecisionView._resolve_site()` (`:553-680`) all present; logic re-read in full below alongside the 22-06 extension. |

**All 20 originals still hold** — none of the gap-closure/review-fix commits removed or weakened any of this behavior; the diffs are additive (new fallback param sources, new placeholder branch, a template wrapper).

### Observable Truths — Gap Closure 22-04 (query-param fix)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 21 | Public form's `site_raw`-named GET request now renders suggestions (D-09 restored) | VERIFIED | `SiteSearchView.get()` (`campaign_views.py:847`): `query = request.GET.get('q', '') or request.GET.get('site_raw', '') or request.GET.get('site_selection', '')`. Independently re-ran `test_site_raw_param_without_q_returns_suggestions` — passes. |
| 22 | Approval-queue / Sites Needing Review `site_selection`-named GET request renders suggestions (D-10 restored) | VERIFIED | Same fallback chain, third term. Independently re-ran `test_site_selection_param_without_q_returns_suggestions` — passes. |
| 23 | Every existing `?q=` caller unaffected; `q` wins when both present | VERIFIED | `test_q_takes_precedence_over_site_raw` passes; the `or`-chain evaluates `q` first. |
| 24 | Throttle / min-length gate / staff-exempt / input_id allowlist unchanged, evaluated against the resolved term | VERIFIED | `campaign_views.py:801-848` — throttle check and `input_id` validation run *before* term resolution and are untouched; `len(query.strip()) < 2` gate now reads the resolved `query`. `test_one_char_site_raw_returns_empty_fragment_without_building_pool` passes. |

**Browser-mechanics reasoning (why this is VERIFIED, not left as a human-verification item):** The original UAT session recorded actual browser `runserver` access logs showing the debounced `hx-get` firing correctly at each keystroke (`fa` → `faulk` → `faulke` → `faulkes`, ~1-2s apart, each 200) for the exact param shapes `?site_raw=...&input_id=...` and (per UAT test 3) the equivalent `site_selection`-keyed shape — i.e., the client-side htmx trigger/debounce mechanism was already empirically proven correct in a real browser; the bug was entirely server-side (reading `q` when the request never carried `q`). Plan 22-04 changed **only** `SiteSearchView.get()` — zero widget/template edits — so the already-proven client-side firing mechanism is untouched, and the new regression tests now prove the server responds with the real suggestion markup for those exact real-world param names. The remaining htmx swap (`hx-target`/`hx-swap="innerHTML"`) and the `onclick` fill-in are unchanged, previously-reviewed markup (three-way id wiring, `|escapejs`) that was already unit-tested and passing before this bug was found — nothing in this fix touches that layer. Given (a) proven-in-browser client firing, (b) now-proven server response correctness for the real param names, and (c) unchanged/previously-verified swap-and-fill markup, this closes the loop without requiring a second live browser pass.

### Observable Truths — Gap Closure 22-05 (Sites Needing Review grouping)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 25 | Sites Needing Review section is visually distinct (actionable card), not another read-only audit list | VERIFIED | `approval_queue.html:14-23` — `<div class="card border-warning">` with a `bg-warning` header reading "Sites Needing Review — action required" and a helper `<p>`. |
| 26 | D-07's locked order (pending / decided / sites-needing-review) preserved — NOT reordered | VERIFIED | Same file: `pending_table` (line 9) → `decided_table` (line 12) → the card wrapping `review_table` (lines 14-23), in that document order. `test_d07_order_preserved_decided_precedes_sites_needing_review` passes (decided-heading index < review-heading index). |
| 27 | Presentation-only — no queryset/view/table-logic change | VERIFIED | `ApprovalQueueView.get_context_data()` and `ApprovalQueueTable` are byte-identical in this respect to the pre-22-05 state (confirmed by re-reading `campaign_views.py`/`campaign_tables.py` — the only additions since are 22-06's placeholder logic, not 22-05's). |
| 28 | Empty-state message and review_table rendering still work when empty | VERIFIED | `test_sites_needing_review_renders_as_distinguishing_action_required_card` and the D-07 order test both run against an empty queue (no CampaignRun fixtures) and pass. |

### Observable Truths — Gap Closure 22-06 (placeholder-site correction)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 29 | Shared helper detects a placeholder Observatory; tier-3 create uses the same constant | VERIFIED | `NEEDS_REVIEW_NAME_PREFIX` + `is_placeholder_observatory()` (`campaign_utils.py:37, 239-253`); tier-3 create at line 226-230 builds `name=f'{NEEDS_REVIEW_NAME_PREFIX}{code}'`. `TestIsPlaceholderObservatory` (3 tests) re-run, passes. |
| 30 | Actionable resolve-mode row with a placeholder site renders the live-search correction widget | VERIFIED | `ApprovalQueueTable.render_site()` (`campaign_tables.py:271-282`) falls through to `_render_site_search_widget()` when `self.show_actions and self.mode == 'resolve' and is_placeholder_observatory(site_obj)`. `test_placeholder_row_renders_live_search_widget_not_plain_text` re-run, passes. |
| 31 | A genuinely-resolved projection-failed retry row (real Observatory) still renders plain text (original CR-01 unchanged) | VERIFIED | Same method: `is_correctable_placeholder` requires `is_placeholder_observatory(site_obj)` True; a real site skips the widget. `test_retry_row_renders_plain_text_site_and_resolve_button_no_input` re-run, passes. |
| 32 | `_resolve_site()` replaces a placeholder site via the existing pool-mapping + `resolve_site(create_placeholder=False)` flow | VERIFIED | `campaign_views.py:591-599` — entry guard `if run.site is None or is_placeholder_observatory(run.site):`, then the unchanged CR-01 mapping/resolve flow. `test_placeholder_replacement_repoints_site_and_clears_review_flag` re-run, passes. |
| 33 | D-06 concurrency protection preserved (claim keyed on exact pre-read site state) | VERIFIED | `previous_site_id = run.site_id` captured before write (`:584`); conditional claim `.filter(pk=pk, approval_status=APPROVED, site_needs_review=True, site_id=previous_site_id).update(site=site)` (`:625-630`). `test_racing_second_resolve_after_placeholder_replacement_does_not_double_write` re-run, passes. |
| 34 | A genuinely-resolved run is never re-resolved; non-reverting projection-failure path unchanged | VERIFIED | `test_genuine_site_still_never_re_resolved_when_replacing_placeholder_would_apply` and `test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event` both independently re-run, pass. |
| 35 | `show_actions=False` never renders the widget regardless of placeholder state; never-fabricate invariant holds | VERIFIED | `render_site()`'s `elif not self.show_actions: return super().render_site(record)` (`campaign_tables.py:283-289`) fires before the placeholder branch is even reached for a resolve-mode+placeholder row in a hypothetical read-only table. `test_placeholder_row_read_only_table_never_renders_widget` and `test_placeholder_replacement_failure_fabricates_no_second_placeholder` re-run, pass. |

### Review-Fix Verification (22-REVIEW.md → 22-REVIEW-FIX.md, 2nd review cycle)

Independently re-checked against the current tree (not the fix report's narrative) — each finding traced to its own code location and its own regression test, re-run individually.

| Finding | Commit | Independently confirmed? |
|---|---|---|
| **CR-01** (Critical): `resolve_site()` Tier 1 (and Tier 2's IntegrityError re-fetch) reported `needs_review=False` for ANY existing Observatory match, including a pre-existing placeholder | `bd80c0d` | **Confirmed.** `campaign_utils.py:164-169` (Tier 1) and `:207-217` (Tier 2 race re-fetch) both now `return obs, is_placeholder_observatory(obs)` instead of unconditional `False`; Tier 2's direct success path (`:186-191`) also converted for consistency. `test_resolve_site_tier1_hit_on_existing_placeholder_still_flags_review` re-run, passes. |
| **CR-02** (Critical): placeholder Observatories polluted the search candidate pool, letting a placeholder be re-offered and (pre-CR-01-fix) silently re-accepted | `c36657c` | **Confirmed.** `_local_observatory_candidates()` (`campaign_utils.py:298`) now iterates `Observatory.objects.exclude(name__startswith=NEEDS_REVIEW_NAME_PREFIX)`. `test_build_site_candidates_excludes_placeholder_observatories` re-run, passes. |
| **WR-01** (Warning): approve branch's inline site-resolution used only `if run.site is None:`, not placeholder-aware like `_resolve_site()` | `ed432ec` | **Confirmed.** `campaign_views.py:479` now reads `if run.site is None or is_placeholder_observatory(run.site):`, mirroring `_resolve_site()`. `test_approve_re_resolves_when_existing_site_is_a_placeholder` re-run, passes. |
| **WR-02** (Warning): placeholder detection was purely a magic-string convention with no guard against a genuine Observatory accidentally carrying the reserved prefix | `d6bc732` | **Confirmed.** `Observatory.clean()` (`solsys_code_observatory/models.py:84-104`) raises `ValidationError` on `name.startswith(NEEDS_REVIEW_NAME_PREFIX)`; tier-3's own `Observatory.objects.create()` bypasses `full_clean()` so the legitimate placeholder path is unaffected (3 regression tests in `solsys_code_observatory/tests/test_models.py`, re-run, pass). |
| **WR-03** (Warning): a replaced placeholder Observatory row was never cleaned up, continuing to pollute the pool | `63b7cef` | **Confirmed.** `_resolve_site()` (`campaign_views.py:638-656`) deletes the previous placeholder after a successful claim, guarded on `is_placeholder_observatory()` and no other `CampaignRun` still referencing it. Both `test_placeholder_replacement_deletes_orphaned_placeholder_observatory` and `test_placeholder_replacement_keeps_placeholder_still_referenced_by_another_run` re-run, pass. |

`IN-01` (Info, "no test exercises Tier-1-hits-placeholder") was explicitly out of scope but its suggested test was added as a natural fallout of the CR-01 fix — confirmed present (`test_resolve_site_tier1_hit_on_existing_placeholder_still_flags_review`).

Both review cycles' findings (the original CR-01/`_project_calendar_event` ValueError swallow, WR-01 `show_actions` no-op, WR-02 missing-`REMOTE_ADDR` bucket — verified in the first pass — AND this second cycle's 5 findings) are confirmed fixed in the current code, not just claimed in either fix report.

**Score:** 40/40 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `solsys_code/campaign_views.py :: SiteSearchView.get()` q/site_raw/site_selection fallback | Widget-agnostic term resolution | VERIFIED | Line 847. |
| `solsys_code/campaign_utils.py :: NEEDS_REVIEW_NAME_PREFIX` + `is_placeholder_observatory()` | Placeholder detection | VERIFIED | Lines 37, 239-253. |
| `solsys_code/campaign_utils.py :: resolve_site()` placeholder-aware success paths | CR-01 fix | VERIFIED | Lines 164-169, 186-191, 207-217. |
| `solsys_code/campaign_utils.py :: _local_observatory_candidates()` placeholder exclusion | CR-02 fix | VERIFIED | Line 298. |
| `solsys_code/campaign_tables.py :: ApprovalQueueTable.render_site()` placeholder branch | 22-06 correction widget | VERIFIED | Lines 271-289. |
| `solsys_code/campaign_views.py :: CampaignRunDecisionView._resolve_site()` placeholder replacement + WR-03 cleanup | 22-06/WR-03 | VERIFIED | Lines 584-656. |
| `solsys_code/campaign_views.py :: CampaignRunDecisionView.post()` approve branch placeholder guard | WR-01 fix | VERIFIED | Line 479. |
| `solsys_code/solsys_code_observatory/models.py :: Observatory.clean()` | WR-02 fix | VERIFIED | Lines 84-104; no migration required (confirmed via `makemigrations --check --dry-run` → "No changes detected"). |
| `src/templates/campaigns/approval_queue.html` Sites Needing Review card | 22-05 grouping fix | VERIFIED | Lines 14-23. |
| `solsys_code/tests/test_campaign_site_search.py` 22-04 regression tests | Param-fallback coverage | VERIFIED | 21 tests total in module (independently re-run, all pass). |
| `solsys_code/tests/test_campaign_approval.py` 22-05/22-06/review-fix regression tests | Grouping + placeholder + CR-01/CR-02/WR-01/WR-03 coverage | VERIFIED | `TestApprovalQueueSitesNeedingReviewGrouping`, `TestIsPlaceholderObservatory`, `TestPlaceholderSiteReplacement`, `test_resolve_site_tier1_hit_on_existing_placeholder_still_flags_review`, `test_approve_re_resolves_when_existing_site_is_a_placeholder`, `test_build_site_candidates_excludes_placeholder_observatories` all present and independently re-run, pass. |
| `solsys_code/solsys_code_observatory/tests/test_models.py` WR-02 regression tests | `Observatory.clean()` coverage | VERIFIED | 3 tests present, independently re-run, pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| Public form `site_raw` widget / queue `site_selection` widget | `SiteSearchView.get()` | `hx-get` → `q`/`site_raw`/`site_selection` fallback chain | WIRED | `campaign_views.py:847`; regression tests drive the view with each real param key and assert suggestions render. |
| `resolve_site()` tier-3 create | `NEEDS_REVIEW_NAME_PREFIX` → `is_placeholder_observatory()` | shared constant, consumed by `render_site()` and `_resolve_site()` | WIRED | `campaign_utils.py:37/226-230/239-253`; consumed at `campaign_tables.py:279` and `campaign_views.py:479,591`. |
| Sites Needing Review placeholder row | `render_site()` widget branch | `self.show_actions and self.mode == 'resolve' and is_placeholder_observatory(...)` | WIRED | `campaign_tables.py:271-282`. |
| Placeholder-correction widget submit | `_resolve_site()` replacement path | POST `action=resolve_site` → conditional claim keyed on `previous_site_id` → `_project_calendar_event()` → orphaned-placeholder delete | WIRED | `campaign_views.py:584-656`. |
| `_local_observatory_candidates()` | `build_site_candidates()` merged pool | `Observatory.objects.exclude(name__startswith=...)` | WIRED | `campaign_utils.py:298-350`. |
| `Observatory.clean()` | Django admin change form | `full_clean()` on form-validated save | WIRED | `solsys_code_observatory/models.py:84-104`; tier-3's `.objects.create()` bypasses this by design (confirmed by `test_tier3_placeholder_create_bypasses_full_clean`). |

### Requirements Coverage

Phase 22 has no formal `.planning/REQUIREMENTS.md` IDs (confirmed unchanged from the original pass: `REQUIREMENTS.md`'s 13 v1 requirements all map to Phases 18-21; `ROADMAP.md` shows Phase 22 was added 2026-07-14, after that requirements pass, specifically to close the Phase 21 gap, with "Requirements: TBD"). The phase instead tracks D-01 through D-10 in `22-CONTEXT.md`, all ten of which were independently cross-checked against the current code in this pass:

| Decision | Status | Evidence |
|---|---|---|
| D-01 (anonymous endpoint) | SATISFIED | `SiteSearchView` has no `StaffRequiredMixin`. |
| D-02 (per-IP throttle, cache-based) | SATISFIED | `_check_and_increment_throttle()`, staff-exempt. |
| D-03 (rendered HTML fragment) | SATISFIED | `render(request, 'campaigns/partials/site_search_results.html', ...)`. |
| D-04 (substring-first, difflib fallback) | SATISFIED | `substring_or_fuzzy_match_candidates()`. |
| D-05 ("Display Name (obscode)") | SATISFIED | `site_search_results.html:26`. |
| D-06 (text-only fill; resolution at approval; never re-resolve a genuine site) | SATISFIED | Confirmed intact through the full 22-06/review-fix chain (see truths 33-34, CR-01 row above). |
| D-07 (third table, locked pending/decided/review order) | SATISFIED | `approval_queue.html`, order preserved through the 22-05 card wrap. |
| D-08 (resolve_site action, same-request projection) | SATISFIED | `_resolve_site()`. |
| D-09 (no create-link on public form; never fabricate from public input) | SATISFIED | 0 "Create new Observatory" occurrences in `campaign_forms.py`; `test_placeholder_replacement_failure_fabricates_no_second_placeholder`. |
| D-10 (queue widget keeps create-link + live-search) | SATISFIED | `_render_site_search_widget()` still emits the link. |

No orphaned or unmapped requirement IDs for Phase 22.

### Anti-Patterns Found

Scanned every file touched across all six plans plus the two review-fix commits (`campaign_utils.py`, `campaign_views.py`, `campaign_tables.py`, `campaign_forms.py`, `solsys_code_observatory/models.py`, `approval_queue.html`, `site_search_results.html`, and the three test files) for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER`.

No blocking anti-patterns found. All `TBD`/`PLACEHOLDER`-shaped matches are the pre-existing domain concepts ("TBD run" = no scheduled window yet; "placeholder Observatory" = the tier-3 concept this phase's own resolution workflow exists to manage) — none reference unresolved work in this phase's own code. No `FIXME`/`XXX`/`HACK` matches anywhere in the touched set.

### Behavioral Spot-Checks / Independent Test Re-Run

- Full `solsys_code` Django test suite independently re-run by this verifier (not the executor/reviewer/fixer): `python manage.py test solsys_code` → **495 tests, all pass** (matches the 22-REVIEW-FIX.md claim: 487 pre-existing + 8 new).
- Targeted re-run of the specific classes/tests named in the gap-closure and review-fix reports (`TestPlaceholderSiteReplacement`, `TestIsPlaceholderObservatory`, `test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event`, `TestApprovalQueueSitesNeedingReviewGrouping`) → **all pass** (16 real tests; the one reported "error" in this targeted run was this verifier's own mistyped test-class name, not an application failure — confirmed by re-running the correct module name cleanly below).
- `python manage.py test solsys_code.tests.test_campaign_site_search` → **21 tests, all pass** (matches 22-04-SUMMARY.md's claim).
- `ruff check` on all touched Python files (`campaign_views.py`, `campaign_utils.py`, `campaign_tables.py`, `campaign_forms.py`, `solsys_code_observatory/models.py`, and the three test files) → clean.
- `ruff format --check` on the same set → clean (8 files already formatted).
- `python manage.py makemigrations --check --dry-run` → "No changes detected" (confirms the WR-02 `Observatory.clean()` fix required no migration, as claimed).
- All 12 commits referenced across `22-UAT.md`/`22-REVIEW-FIX.md` (`dba220d`, `b2ec811`, `936f565`, `13604eb`, `ef97bd2`, `03bb0e9`, `7bd649e`, `bd80c0d`, `c36657c`, `ed432ec`, `d6bc732`, `63b7cef`) confirmed present in `git log --oneline --all`.

### Human Verification Required

None. All three items flagged in the original verification's `human_verification` list were closed by the human UAT session itself (`22-UAT.md`) or by the reasoning documented under "Gap Closure 22-04" above:

1. **Public form / queue-widget live-search rendering in a real browser** — the client-side firing/debounce mechanism was already empirically observed working via real `runserver` access logs during UAT; the server-side bug (reading `q` when the browser never sent it) is now fixed and proven via regression tests that drive the view with the exact real param names. No widget/template markup changed, so the previously-verified swap/escaping/id-wiring layer is untouched.
2. **Multi-row cross-fill risk** — structurally precluded by per-pk-unique `input_id`/`container_id` values (`site-input-{pk}`, `site-suggestions-{input_id}`), unit-tested via the original "three-way id wiring" must-have (still verified, truth #10 in the regression table above); UAT's own multi-row screenshot showed each row's widget rendering independently.
3. **Sites Needing Review resolve UX for the CR-01 blank-timezone fix** — `test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event` independently re-run and confirmed to exercise the exact failure mode (blank `Observatory.timezone` → `sun_event()` raises → flag stays True, no event, warning message) end-to-end at the model/message level.

### Gaps Summary

None. Every must-have across all six plans (40 truths total, including the 15 introduced by the three gap-closure plans) is verified present, substantive, and wired in the current codebase. Both code-review cycles' findings (3 from the first cycle, 5 from the second) are confirmed fixed in the current code, independently re-traced to their own commit and their own regression test — not merely claimed in either REVIEW-FIX report. The full 495-test Django suite passes on an independent re-run, `ruff check`/`ruff format --check` are clean, and no pending migration was missed. All 12 referenced commits exist in git history. The phase goal — live fuzzy site matching wherever a site is entered, plus a real post-approval resolution workflow (including correcting a placeholder, not just retrying a projection) — is achieved end-to-end.

---

_Verified: 2026-07-15T23:00:00Z_
_Verifier: Claude (gsd-verifier)_
