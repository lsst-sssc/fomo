# Phase 9: Proposal Color & Status Visual Treatment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-25
**Phase:** 9-Proposal Color & Status Visual Treatment
**Areas discussed:** Legend design (DISPLAY-07), Empty-proposal / classical-event treatment, Color collision / overflow behavior

---

## Pending Todos Review

| Todo | Score | Folded? |
|------|-------|---------|
| Status-aware calendar event coloring (telescope/proposal-keyed, alpha by confidence) | 0.9 | ✓ Yes |
| Extract site/telescope mapping and instrument extraction into own module | 0.6 | No |

**Notes:** The first todo is the direct origin of DISPLAY-04/05/06/07 and is `resolves_phase: 9` in its
own frontmatter; its "striping" suggestion for status treatment is carried forward as an explicit
option for the `/gsd:sketch` session, alongside research's border-style recommendation. The second is
unrelated (Phase 7 SITE_TELESCOPE_MAP refactor) — reviewed and not folded, consistent with Phase 8's
same decision.

---

## Legend Design (DISPLAY-07)

| Option | Description | Selected |
|--------|-------------|----------|
| Existing footer row | Mirrors target_list_block.html's swatch+label pattern in the existing d-flex footer row | ✓ |
| New row below the footer | Own dedicated line, more breathing room | |
| Collapsible/toggle panel | Hidden by default, click to expand | |

**User's choice:** Existing footer row.

| Option | Description | Selected |
|--------|-------------|----------|
| Only proposals visible this month | Computed from rendered month's context, stays short | ✓ |
| All proposals ever synced | Full historical list, could grow long | |

**User's choice:** Only proposals visible this month.

| Option | Description | Selected |
|--------|-------------|----------|
| Swatch + label only | Mirrors target_list_block.html exactly, no link | |
| Swatch + label + click-to-filter | New capability — filtering, beyond DISPLAY-07's original "identification only" wording | ✓ |

**User's choice:** Swatch + label + click-to-filter.

**Notes:** Claude flagged this as scope creep beyond the locked DISPLAY-07 text and offered to ship the
plain legend now while deferring filtering to a backlog todo. User explicitly chose to keep filtering in
Phase 9 itself rather than defer it.

| Option | Description | Selected |
|--------|-------------|----------|
| Ship swatch+label now, defer filtering | Stays within locked DISPLAY-07 scope; filtering becomes a backlog todo | |
| Keep filtering in Phase 9 itself | Expands DISPLAY-07's scope; requires REQUIREMENTS.md amendment | ✓ |

**User's choice:** Keep filtering in Phase 9 itself.
**Resulting action:** Claude amended `.planning/REQUIREMENTS.md` (DISPLAY-07 text) and
`.planning/ROADMAP.md` (Phase 9 success criteria, added #5) in the same session to reflect the
confirmed scope expansion, rather than leaving CONTEXT.md as the only record of it.

| Option | Description | Selected |
|--------|-------------|----------|
| Toggle highlight | Client-side CSS dim/highlight, no server round-trip, click again to clear | ✓ |
| htmx-driven hide/show | Server round-trip per click, mirrors UTC-offset selector's htmx pattern | |
| URL query param + page filter | Bookmarkable, but loses in-place feel | |

**User's choice:** Toggle highlight (client-side CSS, no server round-trip).

---

## Empty-Proposal / Classical-Event Treatment

| Option | Description | Selected |
|--------|-------------|----------|
| Grey/black (matches today's look) | Reuses existing dark/black-ish tone | |
| Distinct neutral hue from new palette | Deliberately chosen neutral, consistent with new palette design | |
| You decide | Claude picks a sensible neutral during implementation | ✓ |

**User's choice:** You decide.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, label it e.g. "Classical schedule" | Legend stays complete — every color has a matching entry | ✓ |
| No, omit it | Legend only lists actual proposal codes | |

**User's choice:** Yes, include a "Classical schedule"-style legend entry (exact wording Claude's discretion).

---

## Color Collision / Overflow Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, acceptable | Title text and legend already disambiguate; matches research's framing | ✓ |
| No, needs a mitigation | Larger palette / collision detection / warning — added complexity for a low-probability edge case | |

**User's choice:** Yes, acceptable — no mitigation needed.

| Option | Description | Selected |
|--------|-------------|----------|
| One swatch, multiple proposal codes listed | Makes the collision visible, avoids duplicate-looking swatches | ✓ |
| Separate entries per proposal, same color repeated | Simpler, but visually implies more distinct colors than the palette has | |

**User's choice:** One swatch, multiple proposal codes listed.

---

## Claude's Discretion

- Exact neutral-slot color for no-proposal/classical-schedule events.
- Exact legend label copy for the neutral slot (e.g. "Classical schedule").
- Client-side implementation mechanism for the click-to-filter toggle (plain CSS class toggle, no new
  JS framework/dependency).
- Palette representation (literal hex/rgba, independent of `tom_calendar.utils.BOOTSTRAP_COLORS`, per
  research's Bootstrap5-migration-safety recommendation) — technical detail, not reopened with the user.

## Deferred Ideas

None — the one idea that might have been deferred (click-to-filter) was instead absorbed into Phase 9's
scope at the user's explicit request, with REQUIREMENTS.md/ROADMAP.md amended in this session rather
than parked for later.

**Note (unrelated to phase scope):** user reported the AskUserQuestion multi-select/submit UI is
unusable on a phone via remote control (submit control renders off the bottom of the screen). This is
Claude Code app/tooling feedback, not a decision about this project — recorded here for visibility only,
not actionable from within this repo.
