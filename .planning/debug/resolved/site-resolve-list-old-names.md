---
status: resolved
trigger: "On the DCT placeholder row (CampaignRun pk=21, site='NEEDS REVIEW: DCT'), typing 'G37'/'Discovery Chan', selecting 'Lowell Discovery Telescope (G37)', and clicking Resolve fails with 'Could not resolve that site. Try a different search term or an exact MPC code, or use Create new Observatory.' — the placeholder-replacement action (22-06's _resolve_site()) fails for a genuine MPC candidate. Suspected: MPCObscodeFetcher.to_observatory() assigns list-shaped old_names to Observatory.old_names and raises, caught by resolve_site()'s tier-2 except, returning (None, True) with create_placeholder=False."
created: 2026-07-16T07:45:00Z
updated: 2026-07-16T09:00:00Z
resolution: "Confirmed by human browser verification: 'G37'/'Discovery Chan' resolve to the real Lowell Discovery Telescope Observatory, no error. Committed as commit 9f99996 (fix) + adfcd4f (tests). The follow-on 'Site resolved, but the calendar entry couldn't be created automatically' message the user saw afterward is expected, by-design CR-01 behavior (Tier-2 MPC-resolved sites never have a timezone), not a defect."
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

reasoning_checkpoint:
  hypothesis: "The site-search suggestion widget (site_search_results.html) writes the COMBINED display string `f'{display} ({obscode})'` (e.g. 'Lowell Discovery Telescope (G37)') into the site_selection input on click. Both CampaignRunDecisionView._resolve_site() and its approve sibling map that selection back to an obscode via `build_site_candidates().get(selection, selection)`, but the pool is keyed on bare display strings and bare obscodes only — never the combined form. So the exact lookup misses, the 32-char combined string is passed literally to resolve_site(), which rejects it as oversized (len > _MAX_OBSCODE_LEN=4) and returns (None, True); with create_placeholder=False, _resolve_site() renders the generic 'Could not resolve that site' message. NOT the user's suspected list-old_names path — G37's single-code old_names is None and to_observatory() succeeds."
  confirming_evidence:
    - "Live single-code query('G37').obs_data has old_names=None (NoneType); to_observatory() SUCCEEDS, saving old_names=''. The list-old_names hypothesis is disconfirmed for G37 (single-code endpoint returns None, not a list)."
    - "site_search_results.html line 24: onclick sets `inputEl.value = displayText + ' (' + obscodeText + ')'` — the combined 'display (obscode)' form."
    - "build_site_candidates() pool: get('Lowell Discovery Telescope (G37)') -> MISS; get('Lowell Discovery Telescope') -> 'G37'; get('G37') -> 'G37'. The combined form is never a key."
    - "resolve_site('Lowell Discovery Telescope (G37)', create_placeholder=False) -> (None, True) [len 32 > _MAX_OBSCODE_LEN=4, rejected at campaign_utils.py:161]. CONTROL: resolve_site('G37', create_placeholder=False) -> (Observatory G37, False)."
    - "campaign_tables.py _render_site_search_widget() (name='site_selection', hx-get to campaigns:site_search -> site_search_results.html) backs BOTH the resolve-mode placeholder-correction row AND the pending/approve row, so both _resolve_site() (line 594-599) and approve (line 490-500) share the identical broken mapping."
  falsification_test: "If _resolve_site()'s selection->obscode mapping were patched to round-trip the combined 'display (obscode)' form back to its obscode and Resolve on the DCT/G37 row STILL failed, the hypothesis would be wrong."
  fix_rationale: "Add a shared selection_to_obscode() helper in campaign_utils.py that (1) tries an exact pool hit, then (2) parses the widget's trailing ' (obscode)' contract to recover the obscode, else (3) returns the selection unchanged. Replace the `build_site_candidates().get(selection, selection)` line at BOTH call sites. This addresses the root cause (the widget's output format doesn't round-trip through the pool's key space) rather than a symptom, and is backward-compatible: bare obscodes / bare display strings / pool keys all behave exactly as before."
  blind_spots: "The approve path shares the bug but its symptom is milder (run approved with site=None, silently landing in Sites Needing Review) — fixing it is in-scope but not the user's reported symptom; verifying it needs its own check. The regex must handle a display that itself contains parentheses (greedy .* + anchored final group)."
next_action: AWAITING HUMAN VERIFY — fix applied + self-verified (selection_to_obscode() round-trips 'Lowell Discovery Telescope (G37)' -> 'G37'; end-to-end resolve_site returns the real G37 Observatory; 110 tests OK incl. 7 new; ruff clean). Fix left UNCOMMITTED pending browser confirmation, consistent with how the G96 fix was handled. User to re-test in browser: DCT row (CampaignRun pk=21), type 'G37'/'Discovery Chan', select 'Lowell Discovery Telescope (G37)', click Resolve -> should resolve (no 'Could not resolve that site'). On "confirmed fixed" the continuation agent commits + archives.

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: |
  On the DCT placeholder row (CampaignRun pk=21, site='NEEDS REVIEW: DCT', site_needs_review=True),
  typing 'G37' or 'Discovery Chan' into the site-search widget, selecting
  'Lowell Discovery Telescope (G37)', and clicking Resolve should replace the placeholder
  Observatory with a real G37 (Lowell Discovery Telescope) Observatory row, clear
  site_needs_review, and project the pending CalendarEvent.
actual: |
  Clicking Resolve shows the error "Could not resolve that site. Try a different search term
  or an exact MPC code, or use Create new Observatory." G37 is a genuine MPC obscode that
  should resolve via tier-2 (query MPC + to_observatory()).
errors: |
  User-visible: "Could not resolve that site. Try a different search term or an exact MPC
  code, or use Create new Observatory." (campaign_views.py:605-609). No 500 reported.
timeline: |
  Found during human browser verification of the site-search-mpc-no-match (G96) fix. This is
  the latent blind-spot flagged in that session: to_observatory() assigns
  self.obs_data['old_names'] (list-shaped for 60/2712 bulk records) to Observatory.old_names.
  Same root-cause family (MPC record fields being lists), explicitly deferred as out-of-scope
  during the G96 investigation.
reproduction: |
  1. Go to the staff approval queue (/campaigns/approval-queue/).
  2. In "Sites Needing Review", find CampaignRun pk=21 (site='NEEDS REVIEW: DCT' placeholder).
  3. Type 'G37' or 'Discovery Chan' into that row's site-search input.
  4. Select "Lowell Discovery Telescope (G37)" from the suggestion list.
  5. Click Resolve. -> "Could not resolve that site" error.

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-07-16T07:50:00Z
  checked: Live MPCObscodeFetcher().query('G37') (single-code endpoint) + to_observatory() in a rolled-back transaction
  found: obs_data['old_names'] is None (NoneType), not a list. All keys present (obscode, name_utf8, short_name, longitude, rhocosphi, rhosinphi, observations_type, uses_two_line_observations, created_at, updated_at). to_observatory() SUCCEEDED, obs.old_names saved as ''.
  implication: The user's suspected list-old_names failure does NOT occur for G37 — the single-code query() endpoint returns old_names=None (the list shape is a query_all()/bulk-only phenomenon). to_observatory() is not the failure point. Root cause is elsewhere.

- timestamp: 2026-07-16T07:52:00Z
  checked: src/templates/campaigns/partials/site_search_results.html (the suggestion fragment rendered by SiteSearchView for the site_selection widget)
  found: line 24 onclick sets `inputEl.value = displayText + ' (' + obscodeText + ')'` — clicking a suggestion writes the COMBINED 'display (obscode)' string into the input, e.g. 'Lowell Discovery Telescope (G37)'.
  implication: The value submitted as site_selection is the combined form, not the bare display string or bare obscode.

- timestamp: 2026-07-16T07:55:00Z
  checked: build_site_candidates() pool lookups + resolve_site() end-to-end (Django shell, rolled back)
  found: _MAX_OBSCODE_LEN=4. pool.get('Lowell Discovery Telescope (G37)') -> MISS; pool.get('Lowell Discovery Telescope') -> 'G37'; pool.get('G37') -> 'G37'. obscode_selection computed by _resolve_site() = 'Lowell Discovery Telescope (G37)' (len 32). resolve_site(combined, create_placeholder=False) -> (None, True). CONTROL resolve_site('G37', create_placeholder=False) -> (Observatory 'G37: Lowell Discovery Telescope', False).
  implication: ROOT CAUSE. The combined 'display (obscode)' selection is never a pool key, so `build_site_candidates().get(selection, selection)` returns it unchanged; resolve_site() then rejects the 32-char string as an oversized obscode (len > _MAX_OBSCODE_LEN) and returns (None, True) -> _resolve_site() shows 'Could not resolve that site'. The bare obscode 'G37' resolves fine, proving the widget-value round-trip is the sole failure.

- timestamp: 2026-07-16T07:57:00Z
  checked: solsys_code/campaign_tables.py _render_site_search_widget() + render_site(); campaign_views.py approve path (line 490-500) and _resolve_site() (line 594-599)
  found: The SAME hx-get live-search widget (name='site_selection', backed by site_search_results.html) is rendered for both the pending/approve row and the resolve-mode placeholder-correction row. Both view handlers map the selection via the identical `build_site_candidates().get(selection, selection)` line. The approve-path code comment claiming a native <datalist> is stale/inaccurate.
  implication: Both call sites share the identical mapping bug. Fix must be shared (one helper, both call sites). Approve-path symptom is milder (approves with site=None -> silently lands in Sites Needing Review) but is the same defect.

## Eliminated
<!-- APPEND only - prevents re-investigating -->

- hypothesis: to_observatory() assigns a list-shaped old_names to Observatory.old_names and raises, which resolve_site()'s tier-2 except swallows -> (None, True) (the user's suspected root cause, and the blind-spot flagged in the G96 session).
  evidence: For G37 the single-code query() endpoint returns old_names=None (not a list); to_observatory() SUCCEEDS and resolve_site('G37', create_placeholder=False) returns the real Observatory. The list shape is a query_all()/bulk-only phenomenon and never reaches to_observatory() (which is fed by the single-code query()). So this latent issue, while real in the abstract, is NOT what breaks the G37/DCT Resolve action.
  timestamp: 2026-07-16T07:55:00Z

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: |
  The site-search suggestion widget writes the COMBINED display string into the
  site_selection input on click — src/templates/campaigns/partials/site_search_results.html
  line 24 sets `inputEl.value = displayText + ' (' + obscodeText + ')'`, e.g.
  'Lowell Discovery Telescope (G37)'. Both CampaignRunDecisionView._resolve_site()
  (campaign_views.py:594-599, the placeholder-correction / Resolve action) and its approve
  sibling (campaign_views.py:490-500) then map that selection back to an obscode with
  `build_site_candidates().get(selection, selection)`. The candidate pool is keyed on bare
  display strings and bare obscodes only (obscode, name_utf8, short_name, each old_name) —
  never the combined 'display (obscode)' form — so the exact lookup MISSES and the 32-char
  combined string is passed through literally to resolve_site(), which rejects it as an
  oversized obscode (len > _MAX_OBSCODE_LEN=4, campaign_utils.py:161) and returns (None, True).
  With create_placeholder=False, _resolve_site() renders the generic "Could not resolve that
  site" message. (This is NOT the user's suspected list-old_names path: G37's single-code
  old_names is None and to_observatory() succeeds.)
fix: |
  (Applied, self-verified, NOT yet committed — pending human browser verification.)
  1. solsys_code/campaign_utils.py — new selection_to_obscode(selection) helper +
     _SELECTION_DISPLAY_OBSCODE_RE. Maps a widget selection back to an obscode: (a) exact
     pool hit on the whole selection (bare obscode / bare display string); else (b) parse
     the widget's trailing ' (obscode)' token (the 'display (obscode)' contract from
     site_search_results.html), preferring a pool hit on the leading display part; else
     (c) return the selection unchanged. Backward-compatible with the old
     build_site_candidates().get(selection, selection) for bare obscodes/display strings.
  2. solsys_code/campaign_views.py — both CampaignRunDecisionView call sites
     (_resolve_site() ~line 594-599 and the approve path ~line 490-500) now call
     selection_to_obscode(selection) instead of build_site_candidates().get(selection,
     selection), so the combined 'display (obscode)' widget value round-trips to its obscode.
     Stale 'datalist' comment on the approve path corrected to describe the live-search widget.
  3. solsys_code/tests/test_campaign_approval.py — new TestSelectionToObscode unit class
     (6 cases: combined form, bare obscode, bare display, display-with-inner-parens,
     unknown-display fallback to parenthesized obscode, unmappable passthrough) + new
     integration test test_placeholder_replacement_via_combined_widget_selection_resolves
     (the exact reported symptom: POST 'Faulkes Telescope South (F65)' to the resolve_site
     action on a placeholder run -> resolves, no 'Could not resolve that site'). Retargeted
     TestSiteSelectionNameCandidateResolution's build_site_candidates patch from the view
     import site to campaign_utils (where selection_to_obscode now looks it up).
verification: |
  - Root cause reproduced (Django shell): resolve_site('Lowell Discovery Telescope (G37)',
    create_placeholder=False) -> (None, True) [len 32 > _MAX_OBSCODE_LEN=4]; the combined
    form is never a pool key. CONTROL resolve_site('G37', ...) -> real Observatory.
    Disconfirmed the list-old_names hypothesis: G37 single-code old_names is None and
    to_observatory() succeeds.
  - Post-fix (Django shell): selection_to_obscode('Lowell Discovery Telescope (G37)') -> 'G37';
    bare 'G37' -> 'G37'; bare 'Lowell Discovery Telescope' -> 'G37'; display-with-parens
    'Weird (Name) With Parens (G37)' -> 'G37'; unknown -> passthrough. End-to-end resolve via
    the combined selection returns the real G37 Observatory (needs_review=False).
  - Suites: ./manage.py test solsys_code.tests.test_campaign_approval
    solsys_code.tests.test_campaign_site_search -> Ran 110 tests, OK (7 new).
  - Quality gates: ruff check solsys_code/ -> All checks passed; ruff format --check -> clean.
  - PENDING: human confirmation in the real browser (DCT row, CampaignRun pk=21).
files_changed:
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/tests/test_campaign_approval.py
