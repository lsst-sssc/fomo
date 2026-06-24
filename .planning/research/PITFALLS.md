# Pitfalls Research

**Domain:** Django calendar visual treatment + sidecar-model field addition on a third-party model (FOMO v1.4 — DISPLAY-01 proposal-keyed color/status styling, DISPLAY-02 verified/fallback telescope-label field)
**Researched:** 2026-06-24
**Confidence:** HIGH (grounded in direct reads of this repo's `sync_lco_observation_calendar.py`, `load_telescope_runs.py`, `src/templates/tom_calendar/partials/calendar.html`, and the installed `tom_calendar.models.CalendarEvent`; consistent with `STACK.md` and `FEATURES.md` written today)

## Critical Pitfalls

### Pitfall 1: Hashing the raw `proposal` string without normalizing case/whitespace first

**What goes wrong:**
`CalendarEvent.proposal` is `models.CharField(max_length=200, blank=True, default="")` — free text, populated from `ObservationRecord.parameters['proposal']` by the sync command and left empty (`''`) by `load_telescope_runs`. Nothing in this codebase guarantees one canonical casing/whitespace form for "the same" proposal across records. If `proposal_color()` hashes the literal string (`'LTP2025A-004'` vs `'ltp2025a-004'` vs `'LTP2025A-004 '` with a trailing space, the last from a hand-edited classical schedule file or a copy-pasted proposal code), the same real-world proposal silently gets two different colors on the calendar — defeating DISPLAY-01's entire premise ("same proposal renders the same color regardless of telescope/site").

**Why it happens:**
The hash function (`hashlib.sha256`) is correctly deterministic per STACK.md, but determinism on the *exact byte string* is not the same as determinism on the *logical proposal identity*. This is a normalization problem, not a hashing problem, and it's easy to ship the hash call without ever looking at what raw values exist in the DB.

**How to avoid:**
Normalize before hashing: `proposal.strip().upper()` (or `.casefold()` for stricter Unicode correctness, though LCO proposal codes are ASCII) as the very first line of `proposal_color()`, not as a database migration or input-validation fix. Do this in the template tag itself so it's a pure function of the string and doesn't require touching the two management commands' write paths. Spot-check real data first: `CalendarEvent.objects.values_list('proposal', flat=True).distinct()` against the dev DB to confirm there isn't already inconsistent casing in flight (the v1.3 audit found real-data surprises twice already — `EXTRACT-01`/`TELESCOPE-01` — so don't assume the data is clean without checking).

**Warning signs:**
Two calendar events that the user knows belong to the same proposal render in visibly different colors; a quick `Counter(CalendarEvent.objects.values_list('proposal', flat=True))` shows near-duplicate keys differing only in case/whitespace.

**Phase to address:**
DISPLAY-01 phase, first task — normalize inside `proposal_color()` before any palette-index logic is written, not as a follow-up fix. This is exactly the kind of "fixed after the fact via quick task" gap CLAUDE.md already flags as having happened twice on this project (Phase 5/6 notebook gaps) — don't repeat the pattern here.

---

### Pitfall 2: Shipping the new proposal-color hash without fixing the existing `[QUEUED]` grey override

**What goes wrong:**
`src/templates/tom_calendar/partials/calendar.html:158-161` already special-cases `[QUEUED]`-prefixed events with a hardcoded flat grey (`rgba(0, 0, 0, 0.45)`), bypassing `{{ event.color }}` entirely for that branch. If DISPLAY-01's new `{% proposal_color event.proposal %}` tag is wired in only on the `{% else %}` branch (mirroring today's structure), every queued event — a large fraction of real calendar traffic, since records spend time queued before being placed — stays invisible to the new proposal coloring. The feature would ship and pass tests for placed/terminal events while being silently broken for the queued state, which is the most common transient state.

**Why it happens:**
This is an existing, already-confirmed bug (found by the sibling FEATURES.md research today) that sits directly in the file DISPLAY-01 must edit anyway. It's easy to treat "the new color logic" as additive scope and leave the pre-existing grey override untouched, especially if the task is scoped narrowly as "add a color tag" rather than "make color meaningful across all states."

**How to avoid:**
Treat fixing the `[QUEUED]` override as in-scope for the DISPLAY-01 phase, not a separate pre-existing-bug ticket. The status-driven visual treatment (opacity/border/striping) should be layered *on top of* the proposal color, not replace it outright — e.g., proposal color as `background-color`, queued state as a `border`/`opacity` modifier on the same element. Write a test (or at minimum a UAT checklist item) that explicitly checks a `[QUEUED]`-titled event still shows its proposal-derived color, not flat grey.

**Warning signs:**
All queued events on the rendered calendar look identical regardless of proposal; a manual UAT pass only checks placed/terminal events and never checks the queued state.

**Phase to address:**
DISPLAY-01 phase — same task that introduces `proposal_color()`, since both touch the same template lines (158-161) and an isolated "just add color" change without revisiting this block would be incomplete by construction.

---

### Pitfall 3: Treating the sidecar row write as automatically covered by the existing no-churn comparison

**What goes wrong:**
The existing no-churn guarantee in `sync_lco_observation_calendar.py` works by diffing `fields` against the live `CalendarEvent` object's attributes (`changed = any(getattr(event, field_name) != value for field_name, value in fields.items())`) *before* calling `.save()` — and that `fields` dict has nothing to do with the new sidecar model. STACK.md's recommended integration point is `CalendarEventTelescopeLabel.objects.update_or_create(event=event, defaults={'is_verified': not telescope_api_failed})`, called unconditionally for every record on every run. `update_or_create` itself does an internal compare-then-save (Django only writes if the row doesn't exist or a field differs), so the sidecar row's *own* `modified`/`id` won't churn on an unchanged value — but if the model is later given its own `modified`/timestamp field (a natural future addition), this needs the same explicit "compare before write" discipline as `CalendarEvent`'s loop, not an assumption that `update_or_create` alone is equivalent. More subtly: the *parent* `CalendarEvent`'s own no-churn block only ever looks at `fields` (the 7-ish core fields it always compared) — adding the sidecar write doesn't touch `CalendarEvent.modified`, so the parent's no-churn guarantee is actually safe by construction here. The real risk is the opposite mistake: someone "fixing" the sidecar integration by stuffing `is_verified` into the parent `fields` dict (since the sidecar model has no such column) or trying to route it through `CalendarEvent.save()`, which would not work (no such field on `CalendarEvent`) and either errors or is silently dropped.

**Why it happens:**
The codebase's one well-known idiom for "don't churn `modified`" is the inline diff-then-`setattr`-then-`save()` block on `CalendarEvent`. A developer pattern-matching on "no-churn" without reading STACK.md's specific sidecar guidance may try to retrofit the sidecar update into that same loop/dict, or worry that a *separate* `update_or_create` call breaks the "no-churn" contract because it's a second write per record, when in fact it's a logically separate, smaller, independently no-churn write (Django's `update_or_create` already does its own compare against current field values before persisting).

**How to avoid:**
Keep the sidecar write as a separate statement immediately after the existing `get_or_create`/diff/`save()` block (as STACK.md specifies), not merged into the `fields` dict or the `changed` comparison. Confirm via test that re-running the sync command twice with no change in `telescope_api_failed` produces zero `CalendarEventTelescopeLabel.objects.filter(...).count()` change and, if a timestamp field is later added to the sidecar, that it doesn't move either — mirror the existing `test_sync_lco_observation_calendar.py` no-churn test pattern (assert identical `modified`) for the sidecar's own state if/when it gains one.

**Warning signs:**
A code review where the sidecar's `is_verified` value is being assigned via `setattr(event, ...)` or appears as a key inside `fields`; a test that asserts `CalendarEvent.modified` is unchanged but never separately asserts anything about the sidecar row.

**Phase to address:**
DISPLAY-02 phase — write the no-churn-preserving integration test alongside the sidecar's first `update_or_create` call, in the same task, not as a separate hardening pass.

---

### Pitfall 4: The sidecar flag goes stale because nothing re-evaluates it outside the sync command's per-record loop

**What goes wrong:**
`is_verified` is only ever set inside `sync_lco_observation_calendar.py`'s per-record loop, driven by that run's `telescope_api_failed` value. If a record was fallback-labeled on run N (API timed out, or returned an unmapped `(site, telescope_code)` pair) and the LCO API becomes reachable/the mapping gets fixed by run N+1, `update_or_create` will correctly flip `is_verified` to `True` *only if the record is included in run N+1's queryset*. But `load_telescope_runs.py`-created events (classical schedule, no sidecar row at all per STACK.md's recommended option (a)) and any `CalendarEvent` created through a path other than these two commands (e.g. the upstream `tom_calendar` `EventForm`/`create_event` view, explicitly named in STACK.md as a code path this milestone doesn't control) will never get a sidecar row created or updated by anything. If a future feature or admin action creates `CalendarEvent` rows through that view and somehow sets a `telescope`/`proposal` value that *looks* like it came from fallback resolution, there's no mechanism to ever mark it verified or unverified — the template's `{{ event.telescope_label_meta.is_verified|default:True }}` will silently default to "verified" for something that was never resolved at all, which is a different (less severe, but real) gap from "stale due to non-rerun."

**Why it happens:**
The sidecar's only writer is the sync command's per-record loop; there's no signal-based or periodic re-evaluation, and `--proposal` filtering means a given run may not even touch all previously-synced records (e.g. running with a narrower `--proposal` filter than a prior run). A record outside this run's filtered queryset keeps whatever `is_verified` value it had from its last sync, which is correct behavior (it wasn't re-resolved, so nothing should change) but easy to misdescribe as "stale" if not understood precisely.

**How to avoid:**
Document the actual contract precisely (and matches STACK.md's framing): `is_verified` reflects the outcome of the *most recent run that included this record*, not "currently verified right now." This is the same staleness semantic the project already accepts for `CalendarEvent.telescope`/`instrument` themselves (also only updated when a record is in-scope for a given run) — DISPLAY-02 isn't introducing a new staleness risk, it's persisting a signal that already had this same lifecycle. Don't try to "fix" staleness with a background re-check job — that's out of scope and unrequested. Do make sure the default for "no sidecar row at all" (`OneToOneField.DoesNotExist`) reads as "verified" in the template per STACK.md's choice (a), and confirm in a test that a record never touched by the sync command (a `load_telescope_runs` event) renders as verified, not as an error or a false "fallback" indicator.

**Warning signs:**
A user reports "the telescope label says verified but I know that record had an API failure last week" — check whether a later run actually re-included that record before treating it as a bug.

**Phase to address:**
DISPLAY-02 phase — write this contract into the phase's success criteria / a code comment next to the `update_or_create` call (e.g. "`is_verified` reflects the most recent sync run that included this record, not real-time state") so it isn't mis-debugged later as staleness when it's actually correct, narrow-scope behavior.

---

### Pitfall 5: N+1 queries from the sidecar's reverse accessor in the month-grid template loop

**What goes wrong:**
`tom_calendar`'s `render_calendar()` view (confirmed in STACK.md to be a plain function-based view with a fixed context dict, no `extra_context` hook) builds `weeks` → `day.events`/`day.all_day_events` lists of `CalendarEvent` objects without any `select_related`/`prefetch_related` for a reverse `OneToOneField` accessor that doesn't exist yet in the upstream queryset. If the overridden `calendar.html` reads `event.telescope_label_meta.is_verified` per event inside the nested `{% for week %}{% for day %}{% for event %}` loop (as STACK.md's template-access recommendation suggests), each access that isn't already prefetched triggers one extra `SELECT ... FROM solsys_code_calendareventtelescopelabel WHERE event_id = ...` — one query per event rendered, every month view, on every page load. A typical month can have dozens of events (one per night from `load_telescope_runs` runs alone, plus all synced queue/placed events), so this is a real, measurable N+1, not a hypothetical one.

**Why it happens:**
STACK.md correctly notes there is "no Python-level hook" on `render_calendar()` for adding `select_related('telescope_label_meta')` to its queryset — the view function builds and queries `CalendarEvent` itself, upstream, outside this project's control. A template-only fix (reading `event.telescope_label_meta` directly) is the path of least resistance and looks like it "just works" in dev with a handful of test events, hiding the N+1 until a real month's worth of synced data is loaded.

**How to avoid:**
Since the view itself can't be patched without forking it (out of bounds, same reasoning STACK.md gives for not editing `tom_calendar` directly), the practical mitigation is a custom template tag (consistent with DISPLAY-01's own pattern) that batches the lookup once per render rather than once per event — e.g. a `{% load_telescope_verification day.events day.all_day_events %}` tag, or simpler: a tag called once at the top of the partial that takes the full flattened event-id list visible in `weeks` and returns a `dict[event_id, is_verified]` via one `CalendarEventTelescopeLabel.objects.filter(event_id__in=ids).values_list('event_id', 'is_verified')` query, then template lookups become dict access (`{{ verified_map|get_item:event.id }}` via a small filter) instead of per-event ORM traversal. This keeps the fix entirely in the template-tag layer DISPLAY-01 is already adding, with no upstream view changes. At minimum, if the per-event reverse-accessor approach is kept for simplicity, measure it: run `django.db import connection; len(connection.queries)` before/after rendering a populated month in a test or notebook, and treat anything scaling 1:1 with event count as a regression to fix before shipping, not an acceptable cost.

**Warning signs:**
Django Debug Toolbar (if installed) or a query-count assertion in a test shows query count scaling linearly with the number of events in the rendered month; page load on a month with many synced events is noticeably slower than a sparse month.

**Phase to address:**
DISPLAY-02 phase — since this pitfall is specifically about *displaying* the sidecar's flag (the read side), it belongs with whichever phase wires `is_verified` into the template, which per the milestone scope is DISPLAY-02 itself (or a combined DISPLAY-01+02 template-integration task, since both need the same calendar.html edits). Write a query-count regression test (`assertNumQueries`) against a month with multiple synced events as part of this phase's success criteria, not left to manual perception of slowness.

---

### Pitfall 6: Template override silently drifts from upstream `tom_calendar` on a future `tomtoolkit` upgrade

**What goes wrong:**
`src/templates/tom_calendar/partials/calendar.html` is a full copy-and-modify override of the third-party package's bundled template, not an `{% extends %}`/block-based override. STACK.md's own Version Compatibility table already flags that `tomtoolkit==3.0.0a9` → `3.0.0a10`+ renames CSS variables in this exact area (`var(--red)` → `var(--bs-red)`, part of a Bootstrap4→Bootstrap5 migration) and that upstream's `tom_calendar` model/template history is "actively maintained, evolving" (confirmed via the sibling research's GitHub commit-history check). If DISPLAY-01/02 add more logic into this already-forked file and the project later upgrades `tomtoolkit` past `3.0.0a9`, any upstream change to `calendar.html` (new fields rendered, new block structure, a `target_list_block.html` include change, a moon-phase or HTMX-attribute change) is invisible to this project — Django always resolves the project-level override first, so upstream's improved/fixed template is silently never used, and the project's copy can diverge further from what `render_calendar()`'s context dict actually provides on the new version.

**Why it happens:**
Template overrides via app-loader precedence are the standard, often only, customization mechanism for a third-party Django app's UI (confirmed by STACK.md: no `extra_context` hook on `render_calendar()`), but they have no built-in staleness detection — Django doesn't warn when an overridden template's upstream original changes.

**How to avoid:**
This is an accepted, unavoidable cost of the override approach (not a reason to avoid it — there's no alternative customization seam per STACK.md), but make the future-upgrade risk concrete and actionable rather than silent: add a one-line comment at the top of the overridden `calendar.html` noting it is a full fork of `tom_calendar/templates/tom_calendar/partials/calendar.html` as of `tomtoolkit==3.0.0a9`, and that any `tomtoolkit` version bump should include a manual diff of the installed package's original against this file (`diff <(pip show -f tomtoolkit | ...) src/templates/tom_calendar/partials/calendar.html` or simpler: locate the installed copy under `site-packages/tom_calendar/templates/...` and diff directly) before assuming the override still matches upstream's context/structure. This is exactly the kind of cross-cutting risk that belongs in PROJECT.md's Constraints/Context section (it already documents the `var(--red)`/`var(--bs-red)` constraint there) so it survives past this milestone, not just in a code comment.

**Warning signs:**
A future `tomtoolkit` upgrade changelog mentions `tom_calendar` template or `CalendarEvent` model changes; the calendar page looks visually broken or missing a feature present in upstream's docs/changelog after an upgrade, with no obvious project-side code change to explain it.

**Phase to address:**
DISPLAY-01 phase (since it's the phase actively expanding this file) — add the upstream-fork comment and a PROJECT.md Constraints note as part of the phase's documentation task, not deferred. Re-verification itself (the actual diff-on-upgrade) is not this milestone's job — it's a standing constraint for whenever `tomtoolkit` is next upgraded, same treatment as the existing Bootstrap4/5 CSS-variable constraint.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip normalizing `proposal` before hashing ("the data looks clean right now") | Saves one `.strip().upper()` line and a data spot-check | Silent color-identity breakage the moment any record has different casing/whitespace — hard to notice visually since it just looks like "two unrelated colors," not an error | Never — the normalization is a 1-line cost; there's no scenario where skipping it is worth the risk |
| Reuse Python's built-in `hash()` "just to prototype quickly" then forget to swap to `hashlib` | Marginally less code to write during a spike | Color reassignment on every server restart in any environment, including production — already flagged as a confirmed-empirically pitfall in STACK.md | Only acceptable in a throwaway local notebook experiment never merged; never in the shipped template tag |
| Create a `CalendarEventTelescopeLabel` row unconditionally for every event, including `load_telescope_runs` events, "for symmetry" | Simpler mental model (every event always has a sidecar row) | Extra write per classical-schedule event with no information value (these are never fallback-resolved), and a slightly larger surface for the no-churn discipline to cover for no benefit | Acceptable only if a future requirement needs to *query* "all events that were ever evaluated for verification status" in a single un-joined table scan; not needed for this milestone — STACK.md's recommended option (a), skip the row, is correct here |
| Combine the status-prefix-parsing logic and the proposal-color logic into one big template tag/function "since they're both about event styling" | Fewer files/tags to register | Couples two independently-evolvable concerns (color palette logic vs. status-prefix vocabulary) — a future change to one (e.g. a new terminal-state prefix) risks an unrelated regression in the other if they share a function body | Acceptable only as two separate `simple_tag`s/filters in the same module file — never as one function handling both concerns |

## Integration Gotchas

Common mistakes when connecting to this codebase's existing sync infrastructure.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Sidecar write vs. existing `get_or_create`/diff/`save()` block | Folding `is_verified` into the `fields` dict passed to `get_or_create`/compared in `changed = any(...)` | Keep `CalendarEventTelescopeLabel.objects.update_or_create(...)` as a separate statement immediately after the existing block, per STACK.md — `fields` only ever describes `CalendarEvent`'s own columns |
| `load_telescope_runs.py`'s separate `get_or_create`/`.save()` block (line ~91/109) | Assuming DISPLAY-02 needs symmetric sidecar-creation logic added here too | Per STACK.md option (a): classical-schedule events never go through telescope-label resolution, so no sidecar row, no code change needed in this file at all — confirm this explicitly in the phase scope so nobody "completes the pattern" here unnecessarily |
| Per-record facility dispatch loop (`facilities.get(record.facility)`) | Assuming `telescope_api_failed` means the same thing identically for LCO and SOAR records when computing `is_verified` | It already does — both facilities funnel through the same `_coarse_telescope_label`/`telescope_api_failed` contract fixed in Phase 07.1; no facility-specific branching needed in the sidecar write itself, only confirm the existing facility-aware fallback labeling (already correct) feeds the same boolean for both |
| Template tag loading in `calendar.html` | Forgetting `{% load tz calendar_tags %}` needs the new tag library name appended (e.g. `{% load tz calendar_tags calendar_display_extras %}`) | Add the new library to the existing `{% load %}` line at the top of the partial; a missing load produces a clear `TemplateSyntaxError` at render time (fails loudly, not a silent gotcha) but is easy to forget when copy-pasting tag usage from a notebook/test into the template |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-event reverse-accessor (`event.telescope_label_meta`) read in the nested month-grid loop, no batching | Page load time grows with event count; query count in tests scales 1:1 with events rendered | Batch-fetch `is_verified` for all visible event IDs in one query via a template tag, or annotate the queryset upstream if a future hook allows it | Noticeable once a month has more than a handful of synced events — a single busy proposal across LCO+SOAR queue + classical nights can easily produce 20-40+ events in one month view |
| `hashlib.sha256` computed fresh per-event per-render with no caching | Negligible at current scale (STACK.md correctly notes this) | None needed now; if ever revisited, memoize per-request via a dict keyed on the normalized proposal string | Not expected to break within this project's realistic scale (dozens of events/month) — explicitly not a real risk per STACK.md, listed here only for completeness against the question's prompt |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Letting `proposal_color()` or the status-prefix tag interpolate raw `proposal`/`title` text into inline `style="..."` attributes without escaping | Stored-XSS-adjacent risk if a `proposal` string ever contains a quote/angle-bracket from upstream data (low likelihood given LCO proposal-code format, but `CalendarEvent.proposal` is unconstrained free text) | Never echo the raw proposal string into the `style` attribute — only ever interpolate the tag's *computed* hex/rgba output (a value from a fixed internal palette list, never derived from the string itself beyond the hash index), exactly as STACK.md's recommended `int(hash, 16) % len(PALETTE)` approach already does; Django's autoescaping already protects `{{ event.title }}` text content elsewhere in the template, this is specifically about the `style="background-color: ..."` interpolation path |

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Picking a palette where two adjacent Bootstrap4-style hues are visually near-identical at small calendar-cell size (9 colors is a real collision-rate constraint: with even ~10 active proposals, birthday-paradox math says a collision is likely) | Two genuinely different proposals look like the same color at a glance, defeating the point of DISPLAY-01 just as badly as a normalization bug would | Accept that some collision is mathematically inevitable with a 9-bucket palette (this is a deliberate STACK.md-confirmed tradeoff, not a bug) but choose 9 colors that are maximally distinguishable from each other (avoid e.g. two similar blues) and pair the color with the existing title-prefix/status treatment so a collision in color alone doesn't fully erase distinguishability — hovering/clicking still reveals the actual proposal via the event's title/detail |
| Layering status striping in a way that makes the underlying proposal color unreadable (e.g. a busy diagonal-stripe pattern with high color contrast) | Color becomes hard to perceive at the small calendar-cell sizes already in use (`.cal-event-all-day` padding `1px 5px`, `font-size: 0.75rem`) | Prefer opacity or border treatments over full diagonal-stripe CSS gradients for this UI's cell size — a `border` (e.g. `border: 2px dashed` for queued, solid for placed) or `opacity` reduction preserves the base proposal color's hue while still signaling status, whereas a repeating-linear-gradient stripe pattern at this scale tends to either disappear (too fine) or overwhelm the base color (too coarse); if striping is chosen anyway during `/gsd:sketch`, test it at the actual `.cal-event-all-day`/`.cal-event-timed` rendered size, not a mockup at a larger size |
| Diagonal CSS gradient stripes (`repeating-linear-gradient`) requiring vendor-prefix or rendering quirks | Possible visual inconsistency across browsers, though modern evergreen browsers all support unprefixed `repeating-linear-gradient` now | If striping is chosen, use unprefixed `repeating-linear-gradient(45deg, ...)` (no `-webkit-`/`-moz-` prefixes needed for current Chrome/Firefox/Safari/Edge) rather than reaching for an SVG background-image pattern, which is unnecessary complexity for a CSS-only effect already well-supported; this is a minor concern relative to the cell-size-legibility issue above |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Proposal color hashing:** Often missing normalization — verify `proposal_color('LTP2025A-004')` and `proposal_color('ltp2025a-004 ')` return the *same* color, not just that the function runs without error.
- [ ] **Queued-event coloring:** Often missing the fix to the pre-existing `[QUEUED]` grey override at `calendar.html:158-161` — verify a `[QUEUED]`-titled event still shows its proposal's color (with a status modifier layered on, not a full color replacement), not flat grey.
- [ ] **Both event-display branches colored:** Often missing the timed-event (`day.events`/`cal-event-timed`) branch — verify the new color tag is wired into *both* the all-day and timed-event loops, since today only the all-day branch calls `{{ event.color }}` at all (confirmed in STACK.md).
- [ ] **Empty-proposal events:** Often missing a deliberate decision — verify classical-schedule events (`proposal=''`) render with an intentional, consistent treatment (a shared "no proposal" slot, or visually distinct from a hashed color) rather than an accidental, unremarked-upon palette slot.
- [ ] **Sidecar no-churn:** Often missing a regression test — verify re-running `sync_lco_observation_calendar` twice with no change in resolution outcome produces zero new/updated `CalendarEventTelescopeLabel` rows (not just zero new `CalendarEvent` rows).
- [ ] **Sidecar absence handling:** Often missing the "no sidecar row" path — verify a `load_telescope_runs`-created event (no sidecar row at all) renders as "verified" in the template, not as a template error (`RelatedObjectDoesNotExist`) or an incorrect "fallback" indicator.
- [ ] **N+1 on sidecar read:** Often missing a query-count check — verify rendering a month with multiple synced events doesn't issue one extra query per event for the sidecar lookup (`assertNumQueries` in a test, not just visual perception of speed).
- [ ] **Migration generated and applied:** Often missing in a "code works in my shell" demo — verify `./manage.py makemigrations solsys_code` actually produced a migration file for the new sidecar model and it's committed, and `./manage.py migrate` runs clean on a fresh DB.
- [ ] **Demo notebook sync (per CLAUDE.md convention):** Often missing — if DISPLAY-02 changes `sync_lco_observation_calendar.py` behavior (new sidecar write), `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` must be in `files_modified` with a regenerated, re-executed cell exercising the new sidecar behavior — this exact gap has already bitten this project twice (Phase 5/6 quick-task fixes named in CLAUDE.md).

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Un-normalized proposal hashing shipped, discovered post-release (colors don't match for "same" proposal) | LOW | Add `.strip().upper()` (or chosen normalization) to `proposal_color()`; no data migration needed since the function is pure/computed-on-read, not persisted — next render is correct immediately |
| `[QUEUED]` grey override never fixed, shipped alongside new color logic | LOW | Same file, same block (`calendar.html:158-161`) — replace the flat grey with the proposal color plus a queued-specific border/opacity modifier; no model/migration changes needed, pure template fix |
| Sidecar model's `OneToOneField` target chosen as a separate `id` PK + `unique=True` FK instead of `primary_key=True` (diverges from STACK.md's recommended idiom) | MEDIUM | Requires a data migration to drop the redundant `id` column and promote the FK to PK, or accept the extra column permanently — cheaper to get right the first time per STACK.md's explicit guidance, but recoverable via a standard Django migration if missed |
| N+1 discovered late (after a real month of synced data exists) | LOW-MEDIUM | Purely a template/tag-layer fix (batch query via `filter(event_id__in=...)`) — no schema or data change needed, just swap the per-event accessor for the batched-dict lookup in the template |
| Upstream `tomtoolkit` upgrade breaks the forked `calendar.html` silently (Pitfall 6 materializes) | MEDIUM-HIGH | Diff the new installed package's original `calendar.html` against the project's override, manually re-apply the project's customizations (proposal color tag, status treatment, sidecar read) on top of upstream's new structure — cost scales with how much upstream changed; the upfront fork-comment/PROJECT.md note (Pitfall 6's prevention) doesn't prevent this but makes the diff-and-reapply faster to start |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Un-normalized proposal hashing (Pitfall 1) | DISPLAY-01 phase, first task | Test asserting `proposal_color()` returns identical output for case/whitespace variants of the same proposal string |
| `[QUEUED]` grey override left unfixed (Pitfall 2) | DISPLAY-01 phase, same task that wires in the color tag | UAT/test checking a `[QUEUED]`-titled event shows proposal color, not flat grey |
| Sidecar write conflated with `CalendarEvent`'s no-churn diff block (Pitfall 3) | DISPLAY-02 phase | Test asserting an unchanged re-run produces zero sidecar row writes (same pattern as existing `CalendarEvent.modified`-unchanged test) |
| Sidecar staleness contract undocumented (Pitfall 4) | DISPLAY-02 phase | Code comment + a test confirming a record outside the current run's `--proposal` filter keeps its prior `is_verified` value unchanged (correct behavior, documented as such) |
| N+1 from per-event sidecar reverse-accessor reads (Pitfall 5) | DISPLAY-02 phase (template/read-side integration task) | `assertNumQueries` test rendering a month with multiple synced events |
| Template-fork drift on future `tomtoolkit` upgrade (Pitfall 6) | DISPLAY-01 phase (documentation task) | PROJECT.md Constraints section gains a note alongside the existing Bootstrap4/5 CSS-variable constraint; verified by review, not a runtime test |
| 9-color palette collision rate / striping legibility at small cell size (UX Pitfalls) | DISPLAY-01 phase (visual-treatment task, decided via `/gsd:sketch`) | Manual visual check at actual rendered `.cal-event-all-day`/`.cal-event-timed` size, not a larger mockup |

## Sources

- Direct read: `/home/tlister/git/fomo_devel/src/templates/tom_calendar/partials/calendar.html` (existing `[QUEUED]` grey override at lines 158-161, confirmed `{% load tz calendar_tags %}`, confirmed only the all-day branch references `{{ event.color }}`) — HIGH confidence.
- Direct read: `/home/tlister/git/fomo_devel/solsys_code/management/commands/sync_lco_observation_calendar.py` (per-record loop structure, `get_or_create`/diff/`save()` no-churn block, `telescope_api_failed` boolean already computed per record, per-facility counters) — HIGH confidence.
- Direct read: `/home/tlister/git/fomo_devel/solsys_code/management/commands/load_telescope_runs.py` (separate `get_or_create`/`.save()` block, confirms no telescope-label resolution path exists here at all) — HIGH confidence.
- Direct read: installed `tom_calendar/models.py` (`CalendarEvent.proposal`/`telescope` are blank-default `CharField`s, no normalization or choices constraint) — HIGH confidence.
- `.planning/research/STACK.md` (this milestone, same date) — sidecar-model recommendation, integration point, `hashlib` vs `hash()` empirical determinism finding, template-tag pattern, "no Python-level hook on `render_calendar()`" finding — HIGH confidence, built on directly, not contradicted.
- `.planning/research/FEATURES.md` (sibling research, this milestone, same date, referenced but not re-read in full per task instructions) — source of the `[QUEUED]` grey-override bug finding reused here as Pitfall 2.
- `.planning/PROJECT.md` Key Decisions table — facility-aware `_coarse_telescope_label` history, no-churn pattern rationale, terminal-state-prefix precedent, all used to ground Pitfalls 2-4 in this project's actual prior bug history rather than generic Django advice.

---
*Pitfalls research for: FOMO v1.4 Calendar Visual Clarity (DISPLAY-01 proposal-keyed color/status treatment, DISPLAY-02 verified/fallback telescope-label sidecar field)*
*Researched: 2026-06-24*
