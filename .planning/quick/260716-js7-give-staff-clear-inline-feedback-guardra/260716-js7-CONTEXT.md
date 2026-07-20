# Quick Task 260716-js7: Give staff clear inline feedback/guardrails on the Pending Review row before they click Approve with an unresolved or ambiguous Observing Site, instead of silently approving into Sites Needing Review and forcing a separate scroll-and-fix later. The form already technically supports picking+approving in one submission (site_selection has form="decide-form-{pk}"), but nothing signals whether the current field value will actually resolve before the click. - Context

**Gathered:** 2026-07-16
**Status:** Ready for planning

<domain>
## Task Boundary

Give staff clear inline feedback/guardrails on the approval-queue Pending Review row before
they click Approve with an unresolved or ambiguous Observing Site, instead of silently
approving into Sites Needing Review and forcing a separate scroll-and-fix later.

Existing mechanics confirmed before this discussion (do not re-litigate): the pending row's
site-search `<input name="site_selection">` already carries `form="decide-form-{pk}"`, so it
submits alongside the Approve/Reject buttons in one POST, and
`CampaignRunDecisionView.post()`'s approve branch already reads `site_selection` and resolves
it via `selection_to_obscode()` + `resolve_site()`. The gap is purely UX: nothing tells staff,
before they click, whether the current field value will actually resolve.

</domain>

<decisions>
## Implementation Decisions

### Approve-click behavior when the site won't resolve
- **Confirm before approving.** Clicking Approve when the current site field value is not
  known to resolve pops a `confirm()` dialog first (mirroring the existing Reject
  confirmation's pattern), e.g. "This site doesn't look resolved yet — approve anyway? It
  will land in Sites Needing Review." — the user can still proceed.
- This is NOT a hard block. D-06 ("site failure never blocks approval") is preserved —
  nothing on the server changes, and a staff member who confirms can still approve with an
  unresolved site exactly as today. This is a client-side nudge only.
- The confirm dialog should NOT fire when the field is empty (site_raw was blank at
  submission) or when it's the read-only decided/rejected state — only for the pending-row
  live-search widget.

### How "will it resolve" is determined (client-side, no new endpoint)
- **Reuse the existing search response.** No new validation endpoint. Track resolution state
  client-side from the SAME `hx-get` suggestion fetch the widget already performs
  (`campaigns:site_search` → `site_search_results.html`).
- Concretely: mark the field as "known resolved" only when the user has clicked an actual
  suggestion from the dropdown (the existing onclick handler that fills the combined
  `display (obscode)` string) — clicking a real suggestion is the only client-side signal we
  trust. Any subsequent manual edit to the input's text after a click must clear the
  "resolved" flag (so a stale confirmed state can't survive further typing).
- **Known, accepted limitation (Claude's discretion approved this default):** if a row's
  original `site_raw` prefill text happens to exactly match a real MPC/local candidate (the
  user's own "Benedetto" example — resolves via an exact-match fallback at submit time even
  without clicking a suggestion) but the field was never touched/clicked, the confirm dialog
  will still fire once on the first Approve click. This is an acceptable false-positive: the
  confirm is a nudge, not a block, one extra click, and the underlying resolve_site() call at
  submit time is completely unaffected — approval still succeeds via the exact-match
  fallback exactly as before. Do NOT add a page-load or pre-fetch validation call to avoid
  this edge case — that would require a new endpoint/call, contradicting the "reuse existing
  search response" decision.

### Claude's Discretion
- Exact wording of the confirm() message.
- Whether the "known resolved" state is tracked via a `data-*` attribute on the input, a
  sibling hidden field, or a small JS module-scoped map keyed by input id — implementation
  detail, follow whatever fits the existing `_render_site_search_widget()` / htmx pattern most
  simply.
- Whether to extend the same treatment to the "Sites Needing Review" resolve-mode row's
  Resolve button (out of scope unless it falls out naturally at near-zero extra cost, since
  the user's stated pain point was specifically about the Pending Review → Approve flow).

</decisions>

<specifics>
## Specific Ideas

The user's own reproduction case: CampaignRun pk=32, submitted with `site_raw='Benedetto'`
(exactly matches a real MPC candidate for obscode 434) but never clicked a suggestion. This
is the scenario this task should visibly change — clicking Approve on that row should now
show the confirm dialog (since no suggestion was ever clicked), giving staff a moment to
notice and actually pick the site before proceeding, even though today's silent-approve
behavior would have happened to work out fine for this specific site_raw text.

</specifics>

<canonical_refs>
## Canonical References

- D-06 (locked, Phase 21/22 CONTEXT.md): site resolution failure never blocks approval —
  this task's confirm-not-block design explicitly preserves that decision.
- `solsys_code/campaign_tables.py::ApprovalQueueTable._render_site_search_widget()` /
  `render_actions()` — the existing form wiring this task builds on.
- `src/templates/campaigns/partials/site_search_results.html` — the suggestion fragment
  whose onclick handler currently fills the combined `display (obscode)` string; this is the
  natural place to also flip the "known resolved" signal.

</canonical_refs>
