# Feature Research

**Domain:** Calendar/scheduling UI visual encoding (color-by-category + status/confidence visual language) for a Django/htmx/Bootstrap4 TOM calendar
**Researched:** 2026-06-24
**Confidence:** MEDIUM-HIGH (general calendar/accessibility conventions are well-documented and cross-checked across multiple independent sources; astronomy-domain-specific precedent is thinner — most telescope schedulers don't publish their color semantics)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist once a calendar uses color at all. Missing these makes the new coloring feel arbitrary or actively misleading — worse than today's meaningless `pk`-based color, which at least nobody trusts.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Deterministic color per logical group (proposal) | Once color claims to mean something, the same proposal must render identically every time, on every page load, for every telescope it spans — otherwise users learn to distrust it within a day, same failure mode as today's `pk`-keyed color | LOW | Hash `proposal` string -> fixed palette index or HSL hue. Pure function, no DB state, no migration. See Differentiators for the palette-vs-hash choice. |
| Non-color redundant signal for status | WCAG 2.1 SC 1.4.1 requires status not be conveyed by color alone; also directly serves the colorblind-accessibility requirement called out in the question — since color is being spent entirely on proposal identity, status needs an orthogonal channel | LOW-MEDIUM | Already partially true: title prefixes (`[QUEUED]`, `[UNVERIFIED]`, `[EXPIRED]` etc.) already exist as the non-color channel. DISPLAY-01 needs a *visual* (not just textual) echo of that, but the text fallback already satisfies the strict WCAG letter — the visual layer is about scannability, not the only compliance path. |
| Legible text on top of arbitrary background colors | A proposal-keyed palette is unpredictable in lightness; white-on-light-yellow or black-on-dark-purple both happen | LOW | `tom_calendar`'s existing `.cal-event-all-day` template hardcodes `color: #fff !important` — fine for a curated 9-color Bootstrap palette, breaks for an open-ended hash space. Needs either: (a) stick to a small curated palette (sidesteps this entirely), or (b) compute per-color text contrast (WCAG contrast ratio) and switch white/black. (a) is far cheaper. |
| Stable color across whole-grid re-renders and pagination | htmx swaps the month partial on every Prev/Next/Today click; if color depended on row order or query-result position rather than the `proposal` value itself, it would visibly flicker between colors per request | LOW | Must hash on `proposal` field value, never on queryset position/pk — this is the exact bug being fixed, so it's also the exact bug to not reintroduce. |

### Differentiators (Competitive Advantage)

Features that go beyond "color means something" into "this calendar is genuinely easier to scan than the LCO portal or a spreadsheet."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Curated fixed palette, hash-selected (not raw HSL-from-hash) | A small (8-12 color) curated, pre-vetted-for-contrast-and-colorblind-distinguishability palette beats raw `hash(proposal) -> HSL` because an unconstrained hash can produce two visually similar hues for two different proposals (hash collisions in *perceptual* space, not just numeric space), and can produce muddy/illegible colors. `tom_calendar.utils.BOOTSTRAP_COLORS` (9 entries, already used for `target_list_color`) is sitting right there as a precedent to extend, not replace. | LOW | Reuse the existing `BOOTSTRAP_COLORS` pattern: `BOOTSTRAP_COLORS[hash(proposal) % len(BOOTSTRAP_COLORS)]` — same shape as the existing `target_list_color()` helper, just keyed on `proposal` string hash instead of `pk`. With ~9-12 active proposals expected at once, an 8-12 entry palette gives low collision *frequency*, and any collision just means two proposals share a color on a given month view (graceful degradation), not a crash. |
| Colorblind-safe palette curation | The question's framing is explicit: color is the *primary* signal for proposal identity here (unlike status, which already has a text-prefix fallback), so the palette itself must be chosen for protanopia/deuteranopia/tritanopia distinguishability, not just "looks nice" | LOW-MEDIUM | Don't assume Bootstrap's default `--red`/`--green`/`--teal`/`--orange` etc. are mutually distinguishable for deuteranopia (red/green confusion is the most common form) — red and green adjacent in a palette is the classic failure. Needs an explicit check (e.g. against a known colorblind-safe set like ColorBrewer's "qualitative" schemes, or a CVD simulator) before finalizing which Bootstrap CSS vars to include/exclude/reorder. This is a vetting task, not new code. |
| Status as a structural CSS treatment (opacity / border / stripe), applied orthogonally to the color | Lets color answer "whose program is this" and the structural treatment answer "what state is it in" independently — two questions, two channels, no entanglement. This is the core ask of DISPLAY-01's second half. | MEDIUM | See dedicated comparison below — this is the one the user explicitly wants framed as sketch-session options, not decided here. |
| Dedicated `telescope_label_verified` boolean (or enum) field, separate from `telescope`/title text | Makes "was this label live-resolved or did the API fail" a queryable, testable fact instead of a string-parsing problem (`title.startswith('[UNVERIFIED]')`). Directly what DISPLAY-02 asks for. | LOW | Straightforward Django model field (`BooleanField` or `CharField` choices `verified`/`fallback`) — but `CalendarEvent` is an upstream `tom_calendar` model, so adding a field to it directly means a migration against a third-party app's table. Flag for roadmap/architecture research whether that's via a FOMO-local migration against the vendored model, a sibling one-to-one FOMO model, or a `description`-adjacent structured field. Not resolved here — it's an architecture decision, not a feature-landscape one. |
| Visual badge/icon for the verified/fallback distinction on the grid | A small inline glyph (e.g. a dotted-outline icon, a "?"/"~" prefix glyph, or a distinct border-style) lets the eye catch "this label might be wrong" without reading text | LOW | Precedent: dashed/dotted outlines are an established low-cost UI vocabulary for "provisional/estimated/unconfirmed" data (vs. solid for confirmed) — same semantic shape as DISPLAY-02's verified/fallback distinction, can reuse the *same visual primitive* DISPLAY-01 uses for queued/placed if convergent, see Dependency note below. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Raw hash-to-HSL color generation (e.g. `hash(proposal) % 360` for hue) | Feels elegant — "infinite" proposals, no palette to maintain, no collisions in numeric space | Two different proposal codes can hash to adjacent/visually-confusable hues (e.g. hue 118 vs hue 124, both "green" to most eyes, *worse* for colorblind users); no control over saturation/lightness means some hashes produce muddy, low-contrast, or eye-straining colors; defeats the explicit colorblind-safety requirement because nothing curates the output space | Hash into a small **curated, vetted palette** (extend `BOOTSTRAP_COLORS`), not into the full color wheel |
| Telescope-keyed color (reverting to what YSE_PZ does, or hashing on `telescope` instead of `proposal`) | Telescope is already a populated field, simpler hash key, matches YSE_PZ's documented precedent | Explicitly rejected per milestone context — one LCO proposal spans 2m0/1m0/0m4 across telescopes, and the user wants "this is one science program" to read as one color regardless of which telescope it lands on; telescope-keyed color would fragment a single proposal's nights into 2-3 different colors, the opposite of the goal | Hash on `proposal`, already decided |
| Striping/hatching used for *both* signals (color-as-stripe-pattern for proposal AND status) | Tempting to get two-dimensional information density (color hue x stripe density) onto one block | Stacking two encoded visual dimensions on a block that's already ~16-18 truncated characters wide in the month grid (`truncatechars:16`/`18` in the existing template) overloads a tiny target; also stripe-pattern-as-primary-identity-signal has far weaker established precedent than solid-color-as-identity, while stripe-as-status-overlay (Outlook tentative convention) is well precedented | Color stays solely the proposal-identity channel (solid fill); status gets a *structural* treatment (opacity/border/overlay-stripe) layered *on top of*, not *instead of*, that fill — see status comparison below |
| Per-event custom inline color picker / user-assignable colors | Feels empowering, lets users "fix" a color they don't like | Reintroduces the exact problem being solved — color becomes per-user/per-edit arbitrary instead of deterministic-and-meaningful; breaks "same proposal = same color regardless of who's looking" | Deterministic hash-from-proposal is the whole point; if a specific collision is genuinely bad, fix the curated palette, not give users an escape hatch |
| Replacing free-text `telescope`/`proposal` `CharField`s with FK models (YSE_PZ's `Telescope` model) as a side effect of this work | The sibling-TOM comparison doc already surfaces this as "the one idea worth borrowing" from YSE_PZ, so it's tempting to fold into this milestone | Already explicitly out of scope per `tom_calendar_vs_yse_pz_calendar.rst`'s conclusion (keep the generic model) and per `PROJECT.md`'s Out of Scope ("Replacing `SITES`'s hardcoded telescope-name -> obscode mapping... not required"); DISPLAY-01/02 need the *string value* of `proposal`, not a relational identity — a hash works fine on a `CharField` | Hash directly on the existing `proposal: CharField` value; defer any FK-ification to a separate, explicitly-scoped future decision |

## Status Visual Language: Options for the `/gsd:sketch` Session

This is the open design question the downstream consumer flagged explicitly — framed as **options to bring into sketch**, not a final pick.

| Option | What it looks like | Accessibility (colorblind-safety) | Complexity | Fit with existing code |
|--------|--------------------|-----------------------------------|------------|--------------------------|
| **Opacity reduction** (queued = translucent, placed = full, terminal-failure = ?) | `opacity: 0.5` or `rgba()` alpha on the same proposal hue | Colorblind-neutral — opacity doesn't depend on hue perception at all, so it's *safer* than any color-based status signal. Tradeoff: opacity reduces contrast for *everyone*, can make small calendar-grid text harder to read at a glance, especially combined with the white-text-on-color convention already in the template | LOW — this exact mechanism is already shipped for `[QUEUED]` today (`rgba(0,0,0,0.45)`), just needs generalizing to preserve `event.color` underneath instead of overriding it with grey | Direct extension of existing `[QUEUED]` override block in `src/templates/tom_calendar/partials/calendar.html:158-161` — **but today's shipped code replaces `event.color` with a flat grey/black instead of dimming it**, which would erase the proposal-color signal for every queued event. This is a concrete bug DISPLAY-01 must fix, not just extend. |
| **Border style** (solid border = placed/confirmed, dashed border = queued/tentative, thick/double border = terminal-failure) | CSS `border: 1px dashed ...` vs `border: 1px solid ...` | Colorblind-neutral — border style is a shape/pattern cue, fully independent of hue. Established precedent (Outlook's tentative-vs-confirmed border treatment; the general dashed-outline-for-provisional-data convention). Best precedent match for DISPLAY-02's verified/fallback distinction. | LOW — pure CSS, one extra class/conditional in the template, no JS, no extra markup beyond a conditional class string | Cleanly orthogonal to `event.color` (border != background-fill), so color and status never compete for the same pixels. Easiest "redundant non-color channel" to implement well. |
| **Diagonal stripe / hatching overlay** (e.g. `repeating-linear-gradient` over the solid fill for queued/tentative) | CSS `background-image: repeating-linear-gradient(45deg, ...)` layered over the solid proposal-color fill | Colorblind-neutral — a geometric pattern, not a hue. Strong, well-established real-world precedent specifically for *this exact* tentative-vs-confirmed distinction (Outlook free/busy striping, multiple resource-scheduling tools). Tradeoff: at the month-grid's small event-block size (~16-18 char width, ~1 line tall) a diagonal stripe pattern may visually compress into noise or just look "smudged" rather than a clean stripe — needs a real visual check at actual rendered size, not just a mock at full scale. | MEDIUM — `repeating-linear-gradient` is GPU-accelerated, no extra images, but tuning stripe width/angle for legibility inside a tiny block takes iteration, and combining it with the white event-text color needs a contrast check | No existing precedent in this codebase to extend (unlike opacity); would be wholly new CSS. Most "visually rich" option but highest tuning cost relative to payoff at this block size. |
| **Icon/glyph badge** (e.g. a small clock icon for queued, checkmark for placed, X/warning for terminal-failure) | A `<span>` glyph prepended/appended inside the event block, alongside the title text | Colorblind-neutral — shape-based. But adds a second tiny element competing for the same ~16-18 character width that's already truncating titles; on an all-day month-grid block this is the tightest real estate in the whole UI | MEDIUM — needs icon assets or a unicode/emoji set (the calendar already uses an emoji for moon phase, so there's a precedent for emoji-as-glyph in this exact template), plus truncation-width accounting | Has a loose precedent in-template (`day.moon.emoji`) but that's a per-day icon outside the cramped event block, not inside it — fitting a badge into the already-truncated `cal-event-title`/`cal-event-all-day` would likely force shortening the visible title text further, a real tradeoff against legibility of the *other* primary signal (which proposal/program this is, via title text as a secondary check on color). |
| **Text-prefix only (status quo)** | `[QUEUED]`/`[UNVERIFIED]`/`[EXPIRED]` etc., already shipped | Fully colorblind-safe (it's literally text), and already exists | LOW (already done) | Already shipped; satisfies WCAG 1.4.1's letter but not DISPLAY-01's ask for a *visual* (scannable-without-reading) status language — keeping this is necessary as the fallback/tooltip layer regardless of which visual treatment is chosen, but isn't sufficient on its own per the milestone's intent |

**Working recommendation to carry into the sketch session** (per the downstream consumer's framing — bring options, not a final decision):

- **Border style is the strongest single option to lead with**: lowest complexity, cleanest orthogonality to the color channel, real precedent (Outlook), and no risk of visually fighting with the proposal-color fill the way opacity (dims the very color you're trying to make legible) or stripes (visually busy at small size) can.
- **Opacity is the cheapest to ship** because it's a one-line generalization of code already in production (the `[QUEUED]` override) — but the *existing* implementation needs fixing regardless of which option wins, because it currently destroys the color signal it's supposed to coexist with. That fix is in scope for DISPLAY-01 either way.
- **Stripe/hatching is the most "designed" but the riskiest at this UI's actual block size** — worth a quick low-fidelity mockup at real pixel dimensions in the sketch session before committing, not ruling out, just flagging the size risk found in research.
- **Icon badges are the weakest fit** given how cramped `cal-event-title`/`cal-event-all-day` already are (`truncatechars:16`/`18`); would likely need to drop characters from the title to make room, trading one kind of legibility for another.
- Consider: **border-style for DISPLAY-01's status states (queued/placed/terminal-failure) AND for DISPLAY-02's verified/fallback distinction**, using two different border *properties* (e.g. solid-vs-dashed for verified/fallback, color-of-border or thickness for queued/placed/failed) so the two independent facts (program identity via fill color, schedule-state via border style, label-confidence via border weight/dash) each get their own channel without needing four-way visual combinations to be individually legible. This convergence is exactly the kind of cross-cutting call worth making explicit in the sketch session rather than deciding by default here.

## Feature Dependencies

```
DISPLAY-01 (proposal-keyed color)
    └──requires──> proposal field already populated on CalendarEvent (SYNC-05, validated v1.2)
    └──requires──> a curated, colorblind-vetted palette (extends tom_calendar.utils.BOOTSTRAP_COLORS or a FOMO-local equivalent)
    └──requires──> fixing the existing [QUEUED] template override that currently destroys event.color (src/templates/tom_calendar/partials/calendar.html:158-161)

DISPLAY-01 status visual treatment
    └──requires──> DISPLAY-01's color layer (status treatment is applied ON TOP OF / ORTHOGONAL TO color, not instead of it)
    └──enhances──> existing title-prefix convention ([QUEUED]/[UNVERIFIED]/terminal prefixes) — visual layer is additive, text stays as accessible fallback/tooltip

DISPLAY-02 (verified vs fallback field)
    └──requires──> a new persisted field (boolean/enum), distinct from the existing TELESCOPE-03/04 fallback-label logic in sync_lco_observation_calendar.py
    └──enhances──> the existing [UNVERIFIED] title prefix (becomes queryable/testable, not just string-parseable)

DISPLAY-02 visual badge/border
    └──shares-visual-primitive-with──> DISPLAY-01's status border-style option (both are "is this fact about the event fully trustworthy" signals — dashed/dotted-for-uncertain is a natural shared vocabulary)
```

### Dependency Notes

- **DISPLAY-01's color layer requires fixing the existing `[QUEUED]` override before/alongside adding it:** the current shipped code (`src/templates/tom_calendar/partials/calendar.html:158-161`) already special-cases queued events with a hardcoded grey fill that *replaces* `event.color` entirely. Any new proposal-color hash will be invisible on every queued event until this is generalized to dim/border/stripe *around* the proposal color rather than over it. This is a concrete, code-located gap research surfaced — flag for the phase that implements DISPLAY-01.
- **DISPLAY-01's status treatment enhances rather than replaces the text-prefix convention:** keep `[QUEUED]`/`[UNVERIFIED]`/terminal prefixes in the title regardless of which visual treatment ships — they're the existing, already-tested, fully-accessible fallback channel (screen readers, colorblind users, anyone before the new CSS lands). The new visual treatment is for faster at-a-glance scanning, not a replacement.
- **DISPLAY-02 shares its visual vocabulary with DISPLAY-01's status treatment if border-style is chosen for both:** a dashed border for "fallback-resolved telescope label" and a dashed border for "queued/tentative schedule state" are semantically similar ("this fact may not be fully reliable yet") — worth deciding in the sketch session whether they should look identical (simpler, reinforces one mental model: dashed = provisional) or be visually distinguished (avoids conflating two different kinds of uncertainty: schedule-state uncertainty vs. label-resolution uncertainty).
- **DISPLAY-02's new field location is an architecture question, not resolved here:** `CalendarEvent` is an upstream `tom_calendar` model. Adding a field to it means either (a) a FOMO-local migration against the vendored third-party table, (b) a separate FOMO model in a one-to-one relationship, or (c) packing the fact into an existing free-text field (`description`) with a parsed convention — weakest option, repeats the very string-parsing problem DISPLAY-02 exists to avoid. Flag this explicitly for ARCHITECTURE.md / roadmap phase planning.

## MVP Definition

### Launch With (v1 — this milestone, DISPLAY-01/02)

- [ ] Deterministic, curated-palette, colorblind-vetted color hashed from `proposal` — replaces `pk % 9` — essential because it's the explicit headline ask and the cheapest, lowest-risk piece
- [ ] Fix the existing `[QUEUED]` override so it dims/borders the proposal color rather than discarding it — essential, otherwise the new color signal is invisible for every queued event, defeating the point
- [ ] One status-visual-treatment option, chosen in the sketch session (border-style is the research-favored starting point) — essential to DISPLAY-01's stated scope, but the *specific* mechanism is explicitly deferred to sketch
- [ ] Dedicated `telescope_label_verified`-equivalent field on the relevant model, populated from the existing fallback/verified logic in `sync_lco_observation_calendar.py` (TELESCOPE-03/04) — essential, this is DISPLAY-02's literal ask
- [ ] A minimal visual cue for the verified/fallback distinction (even just reusing the chosen status border-style vocabulary) — essential for DISPLAY-02 to be visually discoverable, not just stored

### Add After Validation (v1.x)

- [ ] Tooltip/title-attribute surfacing the verification field on hover, beyond the visual cue — add once the basic field + visual cue round-trip is confirmed useful in practice
- [ ] A small on-page legend/key mapping colors to proposals (since an open-ended hash space means users can't memorize "blue = LTP2025A-004" the way a tiny fixed set might allow) — add if user feedback says color alone isn't enough to identify a proposal without hovering/clicking into the event

### Future Consideration (v2+)

- [ ] WCAG contrast-ratio-aware text color switching (white vs black per background) if/when the palette grows past what's manually contrast-checked — defer until palette size or proposal count makes manual curation unwieldy
- [ ] FK-backed `Telescope`/`Proposal` models (YSE_PZ-style) instead of free-text `CharField`s — already explicitly deferred per the sibling-TOM comparison doc and `PROJECT.md`'s Out of Scope; revisit only if a future milestone needs relational integrity (e.g. per-proposal metadata beyond a color)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Proposal-keyed deterministic color (curated palette) | HIGH | LOW | P1 |
| Fix `[QUEUED]` override to preserve color signal | HIGH | LOW | P1 |
| Status visual treatment (border-style favored) | HIGH | LOW-MEDIUM | P1 |
| Colorblind-safety palette vetting | HIGH (explicit requirement) | LOW | P1 |
| `telescope_label_verified` field | HIGH | LOW | P1 |
| Verified/fallback visual cue | MEDIUM-HIGH | LOW | P1 |
| Tooltip surfacing verification detail | MEDIUM | LOW | P2 |
| On-page color legend | MEDIUM | LOW-MEDIUM | P2 |
| Contrast-aware text color switching | LOW (only matters if palette grows) | MEDIUM | P3 |
| FK-backed Telescope/Proposal models | LOW (not requested, explicitly deferred twice already) | HIGH | P3 |

**Priority key:**
- P1: Must have for this milestone (DISPLAY-01/02)
- P2: Should have, add when possible
- P3: Nice to have, future consideration / explicitly out of scope for now

## Competitor / Precedent Feature Analysis

| Concern | Google Calendar | Outlook (free/busy) | YSE_PZ (sibling TOM, already documented) | FOMO's planned approach |
|---------|------------------|----------------------|--------------------------------------------|---------------------------|
| Color-by-category | User-assignable calendar colors, ~5-7 recommended categories, no enforced colorblind vetting | Calendar-level colors, not status-level | Computed per-view in Python, fixed palette cycled by user or telescope (no colorblind vetting documented) | Deterministic hash from `proposal` into a small curated, colorblind-vetted palette — same *mechanism* as YSE_PZ (palette cycling) but keyed on proposal not telescope, and with explicit accessibility vetting neither precedent documents doing |
| Tentative/uncertain status | Not really a first-class concept (events are either on the calendar or not) | Diagonal-stripe/hash-mark border for tentative vs. solid for confirmed — direct, well-established precedent | No documented equivalent (read-only renders, no booking-confidence concept) | Recommend border-style (precedented by Outlook) as the lead sketch-session option, layered on top of (not replacing) the proposal color |
| Verified vs unverified label | N/A | N/A | N/A — no equivalent concept | Novel within this domain; closest general precedent is the dashed-outline-for-provisional-data UI convention, not a calendar-specific one |

## Sources

- [eventColor - Docs | FullCalendar](https://fullcalendar.io/docs/eventColor) — MEDIUM confidence (vendor docs via web search summary)
- [Event color customization - Demos | FullCalendar](https://fullcalendar.io/docs/event-colors-demo) — MEDIUM confidence
- [How to apply multiple colors for events based on type of events - Issue #137, fullcalendar-angular](https://github.com/fullcalendar/fullcalendar-angular/issues/137) — MEDIUM confidence (community discussion)
- [How to Color Code Google Calendar (Events, Calendars, and Categories)](https://www.usecarly.com/blog/how-to-color-code-google-calendar/) — MEDIUM confidence
- [I have a colleague who is color blind... - Google Calendar Community](https://support.google.com/calendar/thread/254793017/) — MEDIUM confidence (real user-reported colorblind pain point, corroborates the milestone's stated concern)
- [colorhash on PyPI](https://pypi.org/project/colorhash/) and [GitHub - dimostenis/color-hash-python](https://github.com/dimostenis/color-hash-python) — MEDIUM confidence, basis for the hash-to-HSL approach and its tradeoffs (used to support the anti-feature recommendation against raw hash-to-hue)
- [Colorblind-Friendly Data Visualization | Colorblind](https://colorblind.io/guides/data-visualization) — MEDIUM confidence
- [Designing for Color Blindness: A Complete Guide | Colorblind](https://colorblind.io/guides/designing-for-color-blindness) — MEDIUM confidence
- [Making data visualizations accessible - TPGi (Vispero)](https://www.tpgi.com/making-data-visualizations-accessible/) — HIGH confidence (TPGi is a recognized accessibility consultancy; cites WCAG 2.1 SC 1.4.1 directly)
- [Understanding Outlook's Calendar patchwork colors - Slipstick Systems](https://www.slipstick.com/outlook/calendar/understanding-outlooks-calendar-patchwork-colors/) — MEDIUM confidence
- [Free/Busy shows slashed lines in Scheduling Assistant - Microsoft Support](https://support.microsoft.com/en-us/topic/free-busy-shows-slashed-lines-in-scheduling-assistant-da1383a8-54fa-4e89-a2d2-214ae7d82615) — HIGH confidence (official Microsoft support doc, confirms the diagonal-stripe-for-tentative convention directly)
- [Visualize Booking Status: Three Ways to Distinguish Tentative and Confirmed Bookings - Teamup.com](https://www.teamup.com/learn/manage-availability/three-ways-to-visualize-booking-status/) — MEDIUM confidence
- [Stripes in CSS - CSS-Tricks](https://css-tricks.com/stripes-css/) — HIGH confidence (CSS-Tricks is an authoritative front-end reference) — basis for the `repeating-linear-gradient` feasibility/performance notes
- [About - Scheduler Visualization - Las Cumbres Observatory](https://schedule.lco.global/help/) — HIGH confidence (LCO's own published documentation of their scheduler visualization tool, directly relevant domain precedent)
- `docs/design/tom_calendar_vs_yse_pz_calendar.rst` (in-repo, already-completed sibling-TOM comparison) — HIGH confidence, primary internal source for YSE_PZ's per-view Python color-cycling approach
- `tom_calendar` installed package source (`models.py`, `utils.py`, `templates/tom_calendar/partials/calendar.html`) — HIGH confidence, direct inspection confirming `CalendarEvent.color` is a read-only `pk`-keyed property and the `BOOTSTRAP_COLORS`/`target_list_color()` precedent
- `src/templates/tom_calendar/partials/calendar.html` (FOMO project-level template override) — HIGH confidence, direct inspection confirming the existing `[QUEUED]` block currently discards `event.color`
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — HIGH confidence, direct inspection confirming the existing `[UNVERIFIED]`/terminal-prefix title-string convention DISPLAY-01/02 build on

---
*Feature research for: calendar/scheduling UI color-by-category and status/confidence visual language, for FOMO v1.4 "Calendar Visual Clarity" (DISPLAY-01/02)*
*Researched: 2026-06-24*
