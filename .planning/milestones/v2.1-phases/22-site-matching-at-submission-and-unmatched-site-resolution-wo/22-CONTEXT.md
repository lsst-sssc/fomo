# Phase 22: Site Matching at Submission and Unmatched-Site Resolution Workflow - Context

**Gathered:** 2026-07-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the Phase 21 functionality gap in two pieces:

1. **Live fuzzy site matching wherever a site is entered.** The public 'Submit an
   Observing Run' form's Observing site field (`CampaignRunSubmissionForm.site_raw`,
   today a bare free-text CharField) becomes an HTMX live-search autocomplete backed by
   a new endpoint matching against the merged local `Observatory` + full MPC candidate
   pool (`build_site_candidates()`). The same live-search widget replaces the approval
   queue's static per-row datalist (which today only shows the ≤5 difflib matches of the
   originally-submitted `site_raw`).
2. **Post-approval unmatched-site resolution workflow.** "Site failure never blocks
   approval" stays, but approved runs with `site_needs_review=True` (currently a dead
   end: no CalendarEvent, no UI to fix them) get a "Sites needing review" surface where
   staff resolve the site; resolving fires the deferred CalendarEvent projection that
   approval skipped.

Not in scope: changing the approval decision flow itself (approve/reject semantics),
submitter self-service editing, satellite/occultation/radar projection support, or
`Observatory.obscode` schema changes.

</domain>

<decisions>
## Implementation Decisions

### Live-search endpoint (access, protection, response shape)
- **D-01:** The endpoint is **open to anonymous users** — it backs the public
  submission form; the candidate pool is public MPC data.
- **D-02:** Abuse protection is a **zero-dependency per-IP throttle** built on the
  existing Django cache framework (the same cache `build_site_candidates()` already
  uses): a small counter (~10 lines) rejecting with HTTP 429 over the limit. No
  `django-ratelimit`, no DRF APIView. Exact rate (e.g. 30–60 req/min) is Claude's
  discretion during planning.
- **D-03:** The endpoint returns a **rendered HTML fragment** (HTMX convention:
  server-side partial template, `hx-get` swap). No JSON API, no custom JS —
  `django_htmx` is already installed.

### Matching strategy & selection semantics
- **D-04:** Matching is **substring-first with difflib fallback**: case-insensitive
  containment over the cached candidate pool (typing `faulkes` → both
  "Haleakala-Faulkes Telescope North" (F65) and "Siding Spring-Faulkes Telescope
  South" (E10); `lowell` → all Lowell sites), falling back to
  `difflib.get_close_matches` for typo tolerance only when containment finds nothing.
  Implemented as a new small helper alongside `fuzzy_match_candidates()` — the existing
  difflib-only helper is NOT sufficient (whole-string similarity scores short partial
  queries like 'Faulkes' vs long official MPC strings below the 0.6 cutoff).
- **D-05:** Suggestions render as **"Display Name (obscode)"** so the submitter sees
  the resolved MPC site pre-submit.
- **D-06:** Picking a suggestion **fills `site_raw` as text only; resolution stays at
  approval**. The picked display string is guaranteed exact-matchable at approve time
  via the existing pool-mapping → obscode → `resolve_site()` flow
  (`CampaignRunDecisionView` CR-01 path). Zero model changes, no migrations, and no DB
  writes triggered by anonymous traffic (resolve_site tier 2 creates Observatory rows —
  that must never fire from a public submission).

### Post-approval resolution surface & action
- **D-07:** "Sites needing review" is a **third table on the existing approval-queue
  page** (pending / decided / sites-needing-review), listing approved runs with
  `site_needs_review=True`. Reuses the `ApprovalQueueTable` machinery and the page's
  once-per-request candidate pool. No new page or navbar entry.
- **D-08:** Resolution is a **new `resolve_site` action on
  `CampaignRunDecisionView.post()`** alongside approve/reject. It resolves via the
  existing display-string→obscode pool mapping and
  `resolve_site(..., create_placeholder=False)`, honoring the Phase 21 D-06
  never-re-resolve guard, then **fires the calendar projection in the same request**.
  The projection block currently inlined in the approve branch is factored into a
  shared helper so approve and resolve_site both call it (same single-night +
  resolved-site + telescope preconditions; range/TBD runs simply clear the flag and get
  no event, per the existing rule).

### Widget behavior
- **D-09:** The public form gets **no "Create new Observatory" link** — site creation
  stays a vetted staff action (approval queue + existing `CreateObservatory` flow).
  Public submitters just type free text when nothing matches; free text never blocks
  submission.
- **D-10:** The queue's inline site input keeps its "Create new Observatory" link and
  gains the same live-search widget (replacing the static ≤5-candidate datalist), so
  staff typing something different from the original `site_raw` get live suggestions.

### Claude's Discretion
- Exact throttle rate and cache-key scheme (D-02).
- Live-search fine-tuning: minimum characters before searching (~2), `hx-trigger`
  debounce delay (~300 ms), suggestion count cap (~8), "no matches — free text is
  fine" hint copy, and how a picked suggestion populates the input.
- Endpoint URL naming and whether the queue and form share one endpoint or pass a
  context flag.
- Resolve-failure UX in the sites-needing-review table (row stays with an error
  message; exact copy TBD).
- Whether the sites-needing-review table caps row count / orders by recency (mirror
  the decided table's -pk convention unless there's a reason not to).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Prior-phase decisions this phase builds on (and one it supersedes)
- `.planning/phases/21-site-disambiguation-submitter-contact-opt-in/21-CONTEXT.md` —
  Phase 21's locked decisions: D-01/D-02 (merged local+MPC candidate pool, cached),
  D-05 (reuse `CreateObservatory` flow), D-06 (never re-resolve an already-set
  `run.site`). **Phase 22's HTMX endpoint deliberately supersedes Phase 21 D-04's
  "no new endpoint, no AJAX" choice** — that was scoped to Phase 21, not a permanent
  architecture decision.
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`
  Criterion 4 — locks difflib (not rapidfuzz) as the fuzzy library; D-04's substring
  pass is an addition for partial-name matching, not a library replacement.
- `.planning/ROADMAP.md` §"Phase 22" — phase goal and dependency note.

### Existing code this phase extends
- `solsys_code/campaign_utils.py` — `build_site_candidates()` (cached merged pool),
  `fuzzy_match_candidates()` (difflib helper the new substring-first matcher sits
  alongside), `resolve_site()` (3-tier resolver; `create_placeholder=False` for public
  input).
- `solsys_code/campaign_views.py` — `CampaignRunSubmissionView` (public form POST),
  `ApprovalQueueView.get_context_data()` (builds tables + candidate pool once per
  request), `CampaignRunDecisionView.post()` (approve/reject; gains `resolve_site`
  action; contains the CR-01 display-string→obscode mapping and the inline calendar
  projection block to factor out).
- `solsys_code/campaign_tables.py` — `ApprovalQueueTable.render_site()` (the static
  datalist rendering D-10 replaces; note the parent-vs-subclass override reasoning in
  its docstring) and `render_actions()` (the single decide-form the site input targets
  via HTML5 `form=`).
- `solsys_code/campaign_forms.py` — `CampaignRunSubmissionForm.site_raw` (the field
  gaining the widget) and its crispy-forms Layout.
- `solsys_code/solsys_code_observatory/utils.py` — `MPCObscodeFetcher` (`query_all()`
  feeds the pool; do not add a second fetch path).

### Project conventions
- `CLAUDE.md` — GSD conventions; NonSiderealTargetFactory for any Target fixtures;
  Django app tests under `solsys_code/tests/` run via `./manage.py test`; ruff
  single-quote/120-col style. (The demo-notebook rule lists specific modules —
  `telescope_runs.py` and the three sync/load commands — none of which this phase
  touches; no notebook update is required unless plans end up modifying one of them.)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `build_site_candidates()` — 24h-cached merged pool with graceful local-only fallback
  on MPC outage; the live-search endpoint calls this, never a fresh MPC fetch.
- `fuzzy_match_candidates()` — stays as the typo-tolerance fallback inside the new
  substring-first matcher.
- The CR-01 mapping in `CampaignRunDecisionView.post()` (`build_site_candidates().get(selection, selection)`)
  — already converts a picked display string to its obscode at approve time; D-06
  relies on this existing path unchanged.
- `django_htmx` middleware — already installed and configured in settings.
- Django cache framework — already used by the pool cache; D-02's throttle reuses it.

### Established Patterns
- "Never fabricate from public input" (Phase 14 D-09 → quick 260705-l1v → Phase 21):
  `create_placeholder=False` for anything user-typed; D-06/D-09 continue this.
- Approve side-effect failure handling: the broad `except Exception` in
  `CampaignRunDecisionView.post()` reverts approval to PENDING_REVIEW; the factored
  projection helper (D-08) must preserve this revert discipline for the approve path
  while the new resolve_site action needs its own (non-reverting) failure message path.
- `sun_event()` ValueError handling: logged + skipped, never propagated (ground-based
  projection branch) — the shared helper keeps this.
- Tables built once per request with a single `build_site_candidates()` call
  (Pitfall 5 in `ApprovalQueueView`); the third table must not add per-row pool reads.

### Integration Points
- New endpoint URL in `src/fomo/urls.py` or the campaigns URL conf (wherever
  `campaigns:` namespace routes live).
- New partial template for the suggestion list; submission-form template and
  approval-queue template gain HTMX attributes on the site inputs.
- `CampaignRunDecisionView.post()` gains the third action; `ApprovalQueueView`
  gains the third table's queryset (approved + `site_needs_review=True`).

</code_context>

<specifics>
## Specific Ideas

- Operator's motivating example: typing 'Faulkes' in the form should surface both
  Faulkes telescopes — "Haleakala-Faulkes Telescope North (F65)" and "Siding
  Spring-Faulkes Telescope South (E10)"; typing 'Lowell' should surface all MPC
  Lowell sites. This is the acceptance-level behavior for the substring matcher
  (D-04) and drove the substring-first decision — plain difflib demonstrably cannot
  do this.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

### Reviewed Todos (not folded)
- **"Extract site/telescope mapping and instrument extraction into own module"**
  (`.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`)
  — weak keyword match (score 0.4) on the phase-matcher; already reviewed and rejected
  as not-relevant in Phases 13, 18, and 21 (it concerns the LCO calendar-sync
  management command, not campaign site resolution). Still not relevant; not folded.

</deferred>

---

*Phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo*
*Context gathered: 2026-07-14*
