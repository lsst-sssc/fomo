---
phase: 22
reviewers: [codex]
reviewed_at: 2026-07-14T20:24:16Z
plans_reviewed: [22-01-PLAN.md, 22-02-PLAN.md, 22-03-PLAN.md]
---

# Cross-AI Plan Review — Phase 22

> Reviewer availability: gemini/opencode/qwen/cursor/antigravity/coderabbit and local servers not installed; claude skipped (self — this session runs inside Claude Code). Codex (gpt-5.5) was the sole external reviewer; the "Consensus Summary" therefore reflects a single independent reviewer, not cross-reviewer agreement.

## Codex Review

## Plan 22-01 Review

**Summary**  
Plan 01 is a solid foundation for the endpoint and matcher, and it aligns with existing source structure: `build_site_candidates()` already provides the cached merged pool in [solsys_code/campaign_utils.py](/home/tlister/git/fomo_devel/solsys_code/campaign_utils.py:258), `fuzzy_match_candidates()` is isolated and easy to extend beside in [solsys_code/campaign_utils.py](/home/tlister/git/fomo_devel/solsys_code/campaign_utils.py:306), and `campaign_urls.py` has a flat namespace where `site-search/` fits cleanly before `<int:pk>/` routes in [solsys_code/campaign_urls.py](/home/tlister/git/fomo_devel/solsys_code/campaign_urls.py:20). The main gap is security hardening around untrusted `input_id` and server-side query gating.

**Strengths**
- Good dependency ordering: endpoint is delivered before widget consumers, matching the existing separation between utilities, views, URLs, and templates.
- Matcher placement is appropriate: `fuzzy_match_candidates()` is currently a small pure helper in [solsys_code/campaign_utils.py](/home/tlister/git/fomo_devel/solsys_code/campaign_utils.py:306), so adding `substring_or_fuzzy_match_candidates()` below it is low-risk.
- Uses existing candidate-pool cache instead of adding a second MPC fetch path; this is important because `build_site_candidates()` already handles MPC failure and local fallback in [solsys_code/campaign_utils.py](/home/tlister/git/fomo_devel/solsys_code/campaign_utils.py:283).
- GET-only anonymous endpoint is consistent with the public/read-only posture already used by `CampaignGapAnalysisView` in [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:509).

**Concerns**
- **HIGH:** The planned partial escapes `display` and `obscode` for JavaScript, but not `input_id`. `input_id` is planned to come directly from `request.GET`, then be interpolated inside `document.getElementById('{{ input_id }}')`. Current table code consistently uses `format_html()` for dynamic HTML in [solsys_code/campaign_tables.py](/home/tlister/git/fomo_devel/solsys_code/campaign_tables.py:236); the new partial needs equivalent JS-context escaping for `input_id`, not only visible HTML autoescaping.
- **MEDIUM:** The endpoint plan calls `build_site_candidates()` even for blank or one-character direct requests. On cache miss, that can call `MPCObscodeFetcher().query_all()` in [solsys_code/campaign_utils.py](/home/tlister/git/fomo_devel/solsys_code/campaign_utils.py:285). Client-side min-length gating in later plans does not protect direct anonymous requests.
- **LOW:** The plan puts `_check_and_increment_throttle()` in `campaign_utils.py`; that is acceptable, but it is endpoint-specific rather than campaign-domain logic. It may be cleaner in `campaign_views.py` unless future reuse is expected.

**Suggestions**
- Escape `input_id` in JS context too, or better, constrain it server-side to a conservative pattern like `[-A-Za-z0-9_:.]+` before rendering.
- Add server-side minimum query length before building the candidate pool: if `len(query.strip()) < 2`, return an empty fragment without calling `build_site_candidates()`.
- Add a test with hostile `input_id` and hostile candidate text to prove both the DOM-id and suggestion values are escaped in their correct contexts.

**Risk Assessment: MEDIUM**  
The endpoint and matcher are straightforward, but this is the first anonymous high-frequency surface over `build_site_candidates()`. With server-side query gating and `input_id` escaping, risk drops to low.

---

## Plan 22-02 Review

**Summary**  
Plan 02 correctly targets the two existing entry points: `CampaignRunSubmissionForm.site_raw` in [solsys_code/campaign_forms.py](/home/tlister/git/fomo_devel/solsys_code/campaign_forms.py:24) and the actionable approval-row `render_site()` branch in [solsys_code/campaign_tables.py](/home/tlister/git/fomo_devel/solsys_code/campaign_tables.py:208). It preserves the right product boundary: public form gets text-only search and no create link, while staff queue keeps the Create Observatory flow. The key defect is the proposed `hx-trigger` syntax.

**Strengths**
- Correctly uses `reverse_lazy` for form class attributes; current form fields are class-level declarations in [solsys_code/campaign_forms.py](/home/tlister/git/fomo_devel/solsys_code/campaign_forms.py:19), so eager `reverse()` would be fragile.
- The crispy layout insertion point is right: `site_raw` is currently followed by `obs_date` in [solsys_code/campaign_forms.py](/home/tlister/git/fomo_devel/solsys_code/campaign_forms.py:109), so the suggestions container can sit directly beneath the site field.
- The approval-queue replacement is scoped to the actionable unresolved branch; current code already preserves resolved/read-only fallback with `if site_short_name or not self.show_actions` in [solsys_code/campaign_tables.py](/home/tlister/git/fomo_devel/solsys_code/campaign_tables.py:219).
- Keeping the Create Observatory link in staff rows matches the existing round-trip URL construction in [solsys_code/campaign_tables.py](/home/tlister/git/fomo_devel/solsys_code/campaign_tables.py:232).

**Concerns**
- **HIGH:** The planned `hx-trigger="keyup changed delay:300ms[this.value.length >= 2]"` appears malformed. htmx documents event filters as following the event name before modifiers, e.g. `click[ctrlKey]`, and modifiers like `changed delay:1s` after that. See htmx official docs lines 20-32 and 47-50: https://htmx.org/attributes/hx-trigger/. The safer form is likely `keyup[this.value.length >= 2] changed delay:300ms` or use the `input` event: `input[this.value.length >= 2] changed delay:300ms`. The proposed tests only assert the same bad string, so they would not catch a non-firing widget.
- **MEDIUM:** The plan relies on inline `onclick` from Plan 01 for selection. Without a browser-level or template-level test that verifies the generated `input_id` exactly matches the rendered input IDs, a small mismatch between `site-suggestions-id_site_raw` and `site-input-{pk}` will silently break selection.
- **LOW:** The plan says leave `fuzzy_match_candidates` import if unsure. After replacing the datalist, [solsys_code/campaign_tables.py](/home/tlister/git/fomo_devel/solsys_code/campaign_tables.py:17) may become unused. Ruff may catch this, but the plan should explicitly remove it if no caller remains.

**Suggestions**
- Correct the htmx trigger syntax in all three places: form, approval queue, and Plan 03 review rows. Add a source comment or test name explaining the event-filter placement.
- Prefer the `input` event over `keyup` if paste, autocomplete, or mobile input should trigger search.
- Extend markup tests to assert the rendered `hx-vals` `input_id`, actual input `id`, and suggestion container suffix all match.

**Risk Assessment: MEDIUM-HIGH**  
The implementation is mostly simple, but the trigger syntax issue could make the feature look implemented while the browser never sends requests. Existing Django tests would not catch that unless they are changed to assert the corrected contract.

---

## Plan 22-03 Review

**Summary**  
Plan 03 addresses the real remaining workflow gap: current `ApprovalQueueView` only builds pending and decided tables in [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:289), while approved unresolved runs have no staff surface. The extraction of the calendar projection block from [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:391) is well-motivated. The main risk is failure-state handling: as written, a projection failure after resolution can remove the row from the only retry workflow.

**Strengths**
- Correctly reuses existing approval queue infrastructure rather than adding a new page; `approval_queue.html` currently has two table blocks in [src/templates/campaigns/approval_queue.html](/home/tlister/git/fomo_devel/src/templates/campaigns/approval_queue.html:8), so a third block is low-friction.
- The projection helper extraction has a clean target: all CalendarEvent projection logic is currently inline in one contiguous block in [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:395).
- Preserving the approve-path revert behavior is important. Existing code deliberately reverts `approval_status` on side-effect failure in [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:450).
- Server-side state validation for `resolve_site` is necessary because the existing decision endpoint accepts raw POST actions in [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:344).

**Concerns**
- **HIGH:** Projection failure during `resolve_site` can strand the run again. The plan saves `site_needs_review=False` before `_project_calendar_event(run)`. The new review table is planned to filter only `approval_status=APPROVED, site_needs_review=True`; once the flag is false, the row disappears even if no `CalendarEvent` was created. Current projection failures are real possibilities because `insert_or_create_calendar_event()` is an external write called in [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:414) and [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:449). Keep the row reviewable on projection failure, or add another retry surface.
- **MEDIUM:** The D-06 never-re-resolve guard is not concurrency-safe in the proposed branch. Current approve/reject uses a conditional `.filter(...).update(...)` guard in [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:350), but the new branch only “re-fetches fresh” and checks `run.site is None`. Two staff POSTs can both pass that check and race. Use `transaction.atomic()` plus `select_for_update()` or a conditional update pattern.
- **MEDIUM:** `_project_calendar_event(run) -> None` gives no signal about whether an event was actually created. The plan wants different success messages for “added to calendar” vs “site resolved,” but the helper returns `None` for both “created event” and “skipped because range/TBD/missing telescope.” Current inline logic only creates an event under the guard in [solsys_code/campaign_views.py](/home/tlister/git/fomo_devel/solsys_code/campaign_views.py:395).
- **LOW:** The plan’s “already has `run.site` set” test says not to call `resolve_site`, but does not explicitly require clearing `site_needs_review` or keeping the row actionable. If such a row exists, the resolve action may project repeatedly or remain in the review table depending on implementation.

**Suggestions**
- Make `_project_calendar_event()` return a boolean: `True` when `insert_or_create_calendar_event()` was called, `False` when skipped.
- On projection exception in `resolve_site`, preserve retryability. For example: save the resolved site, but keep `site_needs_review=True` and message “Site was resolved, but calendar projection failed; retry Resolve.”
- Wrap `resolve_site` action in a transaction and lock the row, or use a conditional update that proves the row is still `APPROVED`, `site_needs_review=True`, and `site__isnull=True`.
- Add tests asserting that projection failure leaves the run visible in `review_table`, not merely `APPROVED`.

**Risk Assessment: HIGH**  
This plan touches the most sensitive workflow: approval state, site resolution, and CalendarEvent projection. The structure is good, but failure handling needs tightening so the new workflow cannot recreate the “approved but no event and no UI path” dead end.

---

## Overall Assessment

The three-wave decomposition is sound and maps well to the existing code. The biggest fixes before execution are: correct the htmx trigger syntax, escape or validate `input_id`, enforce query length server-side, and preserve retryability when deferred projection fails after site resolution. With those changes, the phase should achieve its goal without new dependencies or migrations.

---

## Consensus Summary

Single external reviewer (Codex) — findings below are that reviewer's verified, source-cited conclusions rather than multi-reviewer consensus. All findings cite concrete `file:line` evidence from the working tree.

### Agreed Strengths
- Three-wave decomposition (endpoint → widgets → resolution) is sound and maps cleanly onto existing code structure (`campaign_utils.py`, `campaign_forms.py`, `campaign_tables.py`, `campaign_views.py`).
- Reuses the cached `build_site_candidates()` pool instead of adding a second MPC fetch path; endpoint posture matches the existing public read-only `CampaignGapAnalysisView`.
- `_project_calendar_event()` extraction has a clean, contiguous inline target in `campaign_views.py`; approve-path revert behavior deliberately preserved.
- No new dependencies, no migrations.

### Agreed Concerns
1. **HIGH (22-02, propagates to 22-03):** Planned `hx-trigger="keyup changed delay:300ms[this.value.length >= 2]"` is malformed htmx syntax — the event filter must follow the event name (`keyup[this.value.length >= 2] changed delay:300ms`, or better the `input` event). The planned tests assert the same bad string, so they would not catch a widget that never fires.
2. **HIGH (22-01):** `input_id` from `request.GET` is interpolated into `document.getElementById('{{ input_id }}')` without JS-context escaping or server-side validation — XSS/JS-injection vector. Constrain to a conservative pattern (e.g. `[-A-Za-z0-9_:.]+`) and/or escape in JS context; add hostile-`input_id` test.
3. **HIGH (22-03):** Projection failure during `resolve_site` can strand the run: plan saves `site_needs_review=False` before `_project_calendar_event(run)`, and the review table filters on `site_needs_review=True` — on projection failure the row disappears from the only retry surface, recreating the "approved but no event and no UI path" dead end.
4. **MEDIUM (22-01):** Endpoint calls `build_site_candidates()` even for blank/1-char direct requests; on cache miss that triggers `MPCObscodeFetcher().query_all()`. Add server-side min-length gate (return empty fragment for <2 chars) before building the pool.
5. **MEDIUM (22-03):** D-06 never-re-resolve guard is not concurrency-safe (plain re-fetch + `run.site is None` check; two staff POSTs can race). Use `transaction.atomic()` + `select_for_update()` or a conditional `.filter(...).update(...)`.
6. **MEDIUM (22-03):** `_project_calendar_event() -> None` can't distinguish "created event" from "skipped" — but the plan wants different success messages. Return a boolean.
7. **MEDIUM (22-02):** Selection relies on inline `onclick` with generated `input_id`; no test asserts the rendered `hx-vals` `input_id`, actual input `id`, and suggestion-container suffix all match — a mismatch silently breaks click-to-fill.
8. **LOW:** `_check_and_increment_throttle()` placement in `campaign_utils.py` (endpoint-specific, arguably view-layer); potential unused `fuzzy_match_candidates` import in `campaign_tables.py` after datalist removal; "already has `run.site` set" edge needs explicit expected state for `site_needs_review`.

### Divergent Views
None — single reviewer.

### Risk Assessment Roll-up
- 22-01: MEDIUM (drops to LOW with input_id hardening + server-side query gate)
- 22-02: MEDIUM-HIGH (hx-trigger syntax could ship a silently dead widget)
- 22-03: HIGH (approval state + resolution + projection failure handling)
