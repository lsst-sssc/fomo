# Requirements: Telescope Runs Calendar — v1.4 Calendar Visual Clarity

**Defined:** 2026-06-24
**Core Value:** Make `CalendarEvent` color and status convey real meaning (proposal identity, queued/placed/failed state) and add a dedicated field for fallback-resolved telescope labels.

## v1 Requirements

Requirements for milestone v1.4. Each maps to roadmap phases.

### Telescope Label Verification (DISPLAY-02)

- [ ] **DISPLAY-01**: A new `CalendarEventTelescopeLabel` sidecar model (`OneToOneField(primary_key=True)` on `CalendarEvent`) records whether a record's telescope label was live-resolved or fallback-guessed; `sync_lco_observation_calendar.py` writes it via `update_or_create` immediately after the existing `CalendarEvent.objects.get_or_create(...)` block, populated from the existing TELESCOPE-03/04 fallback-vs-verified logic. No sidecar row is created for classically-scheduled events (`load_telescope_runs.py` unchanged); the template treats a missing row as "verified" by documented default.
- [ ] **DISPLAY-02**: The calendar UI shows a visual cue (border/badge style, sharing the same visual vocabulary as the status-treatment border style where convergent) distinguishing a fallback-resolved label from a verified one — discoverable without reading the title text.
- [ ] **DISPLAY-03**: Hovering a fallback-labeled event shows a tooltip (title attribute) with the verification detail, beyond the visual cue alone.

### Calendar Color & Status (DISPLAY-01)

- [ ] **DISPLAY-04**: `CalendarEvent` color is hashed deterministically from a normalized (`.strip().upper()`) `proposal` string into a small, curated, colorblind-vetted palette (extending the `BOOTSTRAP_COLORS` precedent), replacing today's meaningless `pk`-based color. The same proposal renders identically across telescopes, htmx month-grid re-renders, and process restarts. Applies to both the all-day and timed-event render branches. Events with an empty `proposal` (classical schedule) get a dedicated neutral palette slot, not a raw hash of the empty string.
- [ ] **DISPLAY-05**: The existing `[QUEUED]` template override (`calendar.html:158-161`), which currently replaces `event.color` with flat grey, is fixed so the proposal color survives under a status modifier (dimmed/bordered, not discarded).
- [ ] **DISPLAY-06**: A status visual treatment — mechanism (border-style/opacity/stripe) decided via a `/gsd:sketch` session during phase planning, research favors border-style — is layered orthogonally on top of (not instead of) the proposal color, distinguishing queued/placed/terminal-failure states for both all-day and timed events. The existing `[QUEUED]`/`[UNVERIFIED]`/terminal-prefix text convention remains as the accessible fallback channel.
- [ ] **DISPLAY-07**: A small on-page legend/key maps proposal codes to their rendered colors, so a proposal's color can be identified without hovering or clicking into an event.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Calendar Color & Status

- **DISPLAY-08**: WCAG contrast-ratio-aware text color switching (white vs. black per background), deferred until palette size or proposal count makes manual contrast-checking unwieldy.
- **DISPLAY-09**: Batching template tag to eliminate the N+1 query from the `CalendarEventTelescopeLabel` reverse `OneToOneField` accessor read per-event in the month-grid loop — current calendar-event volume doesn't justify the added scope; revisit if volume grows.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Raw hash-to-HSL color generation (`hash(proposal) % 360`) | Adjacent hues can be visually confusable (worse for colorblind users) and produce muddy/low-contrast colors; a curated palette avoids both — see DISPLAY-04 |
| Telescope-keyed color (reverting to YSE_PZ's approach) | One LCO proposal spans 2m0/1m0/0m4 across telescopes; telescope-keying would fragment a single proposal's nights into multiple colors, the opposite of the goal |
| Striping/hatching used for *both* proposal identity and status | Stacking two encoded visual dimensions on an already-truncated (~16-18 char) event block overloads a tiny target |
| Per-event custom inline color picker | Reintroduces the exact per-user/per-edit arbitrariness problem DISPLAY-04 exists to fix |
| Replacing free-text `telescope`/`proposal` `CharField`s with FK models | Already explicitly out of scope per `tom_calendar_vs_yse_pz_calendar.rst` and prior `PROJECT.md` Out of Scope; DISPLAY-04 needs the string value, not relational identity |
| Icon/glyph badge for status (e.g. clock/checkmark/warning) | Competes for the same cramped ~16-18 character event-block width that titles already truncate into; border-style treatment (DISPLAY-06) achieves the same goal without that tradeoff |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DISPLAY-01 | TBD | Pending |
| DISPLAY-02 | TBD | Pending |
| DISPLAY-03 | TBD | Pending |
| DISPLAY-04 | TBD | Pending |
| DISPLAY-05 | TBD | Pending |
| DISPLAY-06 | TBD | Pending |
| DISPLAY-07 | TBD | Pending |

**Coverage:**
- v1 requirements: 7 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 7 ⚠️ (resolved by `/gsd-roadmapper`)

---
*Requirements defined: 2026-06-24*
*Last updated: 2026-06-24 after initial v1.4 definition*
