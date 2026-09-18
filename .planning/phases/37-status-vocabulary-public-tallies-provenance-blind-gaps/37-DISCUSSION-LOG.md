# Phase 37: Status Vocabulary, Public Tallies & Provenance-Blind Gaps - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-18
**Phase:** 37-status-vocabulary-public-tallies-provenance-blind-gaps
**Areas discussed:** Status vocabulary wording, Tally content & placement, Unused-night appearance, Provenance-blind gap claims

---

## Todo cross-reference

| Todo | Action |
|------|--------|
| Add TTL cache to attribution banner count | User asked for a re-review of "items 2 and 3"; item 1 folded as a pattern for the public tallies (not as its own fix) |
| Guard attribution dismiss with is_offered_candidate | Re-reviewed: still open, unrelated — reviewed, not folded |
| Skip sun_event computation for existing reconciler nights | Re-reviewed: overtaken by Phase 35 (allocation projector D-13, `test_allocation_projector.py:860`) — close |

---

## Status vocabulary wording

| Option | Description | Selected |
|--------|-------------|----------|
| Short letters everywhere | Keep Phase 34's [Q] [S] [O] [X] [C] [F] [?] as final; migrate run-level [CANCELLED]/[WEATHERED] | ✓ |
| Short letters for observations, words for runs | Two spellings, defined in one module | |
| Words everywhere | [QUEUED] [SCHEDULED] … — self-explanatory but reverses 34 D-01's compactness | |

| Option | Description | Selected |
|--------|-------------|----------|
| Share the letters: run cancelled → [C], weathered → [W] | One vocabulary literally; modal's "Run status:" line says which layer | ✓ |
| Separate letters for run-level states | e.g. [RC]/[W]; bigger legend, two-letter markers | |

| Option | Description | Selected |
|--------|-------------|----------|
| Scheduled | Matches the LCO portal's word; [S] stays | ✓ |
| Placed | The internal code word; would suggest [P] | |

| Option | Description | Selected |
|--------|-------------|----------|
| One legend lists every state incl. unused and [?] | Legend is the single explanation of everything visible | ✓ |
| Legend lists title markers only | Unused explained elsewhere | |

| Option | Description | Selected |
|--------|-------------|----------|
| Facility lists + a FOMO override table for known misfits | Default from TOM lists; override Gemini | |
| LCO/SOAR only; everything else unknown | Other facilities → [?] | |
| *(Other)* LCO/SOAR OCS vocabulary canonical; map misfits onto it | | ✓ |

**User's choice:** Short letters everywhere; shared [C] and new [W]; "Scheduled"; full legend; LCO/SOAR vocabulary as the canonical model with other facilities mapped onto it.
**Notes:** "tom_gemini is a limited subset (currently... may change in the future with GPP support) for only scheduling (disruptive) ToOs so TRIGGERED/ON_HOLD makes sense for that limited subset of functionality. I assume ESO will (eventually) get a more fleshed out set read from the ESO Phase 2 but that feels a long way in the future."

---

## Tally content & placement

| Option | Description | Selected |
|--------|-------------|----------|
| Nights from linked records only; no 'unused' figure for container runs | Unused omitted for queue/class-wide runs | |
| Same five counters for every run, unused = 0 | Uniform columns | |
| Records/groups only for container runs; nights only for allocation runs | Two tally shapes | |
| *(Other)* Estimate unused nights from the portal's unused time ÷ 10 h | | ✓ |

Follow-up (plain text): the figure comes from the proposal API's `timeallocation_set` (no manual book-keeping); attached to every run carrying the proposal code (a proposal may be one run or many, and the proposal itself cannot tell); "a night = 10 hours" is a fixed rule of thumb used previously in the NOAO/NOIRLab proposal process for LCO.

| Option | Description | Selected |
|--------|-------------|----------|
| In the unattended runner tick, cached on the record | No credentialed call from a public page load; Phase 36 failure isolation | ✓ |
| On page load with a short TTL cache | Fresh-ish; public page can hit the API | |

| Option | Description | Selected |
|--------|-------------|----------|
| New small model keyed by proposal code | e.g. ProposalTimeAllocation; WatchedProposal not overloaded | ✓ |
| Extra fields on WatchedProposal | Fewer tables; only watched proposals get a figure | |

| Option | Description | Selected |
|--------|-------------|----------|
| One compact 'Progress' cell | Single column with counts and reused letters | ✓ |
| Separate sortable columns | Six new columns; wider table | |

| Option | Description | Selected |
|--------|-------------|----------|
| The calendar pop-up's attributed-run block | Add tally to event_form.html's existing block | ✓ |
| A new public run detail page | New view/template/PII gating | |

| Option | Description | Selected |
|--------|-------------|----------|
| Header strip on campaign runs page + badge on campaign list | Unused counted once per distinct proposal | ✓ |
| Campaign runs page header only | List unchanged | |

**User's choice:** Portal-sourced unused estimate at 10 h/night, fetched by the runner into a new per-proposal model; one compact Progress cell; pop-up block as "run detail"; header strip + list badge roll-up.

---

## Unused-night appearance

| Option | Description | Selected |
|--------|-------------|----------|
| Derived at display time, no write | Template tag like campaign_decoration(); no projector churn | ✓ |
| Written by the allocation projector on the next tick | Self-describing titles; time-dependent writes | |

| Option | Description | Selected |
|--------|-------------|----------|
| Muted chip + visible [U] token prepended at render | Style + text, never colour-alone | ✓ |
| Muted chip only, no token | Tooltip/aria only | |
| Status ring in a new colour | Colour-only, fourth ring colour | |

| Option | Description | Selected |
|--------|-------------|----------|
| Staff run status always wins; only truly empty nights read unused | Unlinked same-night event does not rescue | ✓ |
| Any observation event on that site-night rescues it | Per-cell query; masks missing attribution | |

| Option | Description | Selected |
|--------|-------------|----------|
| Yes to both (counts in tally; legend [U] is click-to-filter) | Same classifier for table and calendar | ✓ |
| Counts in the tally; legend entry informational only | | |

**User's choice:** Display-time derivation, muted + [U] token, staff status wins, counted and filterable.

---

## Provenance-blind gap claims

| Option | Description | Selected |
|--------|-------------|----------|
| Observed and scheduled blocks only ([O]/[S]) | Queued windows and failures claim nothing | ✓ |
| Any non-failed observation event, including queued windows | Over-reports coverage | |
| Observed only ([O]) | Strictest | |

| Option | Description | Selected |
|--------|-------------|----------|
| observed_site → Observatory; else linked run's site; else not assignable (reported, never dropped) | | ✓ |
| Any observation on the campaign claims the night at every site | Hides an idle site | |

| Option | Description | Selected |
|--------|-------------|----------|
| Attributed to the campaign's runs OR target in the campaign | Provenance-blind union | ✓ |
| Attribution link only | Unattributed observations still read as gaps | |

| Option | Description | Selected |
|--------|-------------|----------|
| Both: run windows (intent) + observation blocks (what happened) | Keep today's window claims, add blocks | ✓ |
| Events for past nights, windows for future nights | Elapsed awarded nights show as gaps | |

**User's choice:** [O]/[S] blocks claim; site from observed_site → run site → unassignable; link OR target membership; run windows still claim.

---

## Claude's Discretion

- Vocabulary module name/location; legacy title migration mechanics; the exact "night has ended" instant; TALLY-03 enforcement mechanism; pre-fetch tally wording and proposal-code discovery; gap cache invalidation; portal API specifics.

## Deferred Ideas

- Per-class hours-per-night table; a stored [U] marker; same-site-night rescue rule; a public run detail page.
