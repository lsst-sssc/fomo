# Phase 12: Display Polish - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Two targeted display fixes for the calendar UI:

1. **DISPLAY-08**: WCAG-AA-compliant text color on all-day calendar events — computed
   from relative luminance of the proposal palette background, not hardcoded. Covers
   all 8 `PROPOSAL_PALETTE` entries and `NEUTRAL_SLOT_COLOR`.

2. **DISPLAY-09**: Eliminate N+1 queries in the calendar template for
   `CalendarEventTelescopeLabel` sidecar data and `active_todos.count` — both loaded
   via a single FOMO-local wrapper view that shadows the `tom_calendar` URL.

No new features, no new models, no new commands.

</domain>

<decisions>
## Implementation Decisions

### DISPLAY-08 — WCAG Text Color

- **D-01:** Compute text color via a new `text_color_for_bg(hex_color: str) -> str`
  template tag in `calendar_display_extras.py`. It implements the WCAG relative
  luminance formula and returns `'#fff'` or `'#000'` — whichever achieves ≥ 4.5:1
  contrast against the given background hex.

- **D-02:** Coverage: apply to all 8 `PROPOSAL_PALETTE` entries AND
  `NEUTRAL_SLOT_COLOR` (`#5a6268`). Both are rendered as all-day event backgrounds;
  both should use the computed text color for consistency. Tests cover all 9.

- **D-03:** CSS mechanism: remove `color: #fff !important` from the `.cal-event-all-day`
  and `.cal-event-all-day a` CSS rules in `calendar.html`. Replace with
  `color: {{ text_color }}` in the inline `style` attribute on the event `<div>`.
  No CSS `!important` battles.

- **D-04:** Timed events (`.cal-event-timed`) are excluded — they have a transparent
  background; the palette color is used only for the `▌` bullet, not the text
  background. No text-color change needed for timed events.

### DISPLAY-09 — N+1 Elimination

- **D-05:** Fix BOTH N+1 patterns in the event loops simultaneously:
  - `event.telescope_label_meta.is_verified` — OneToOneField reverse accessor
  - `event.active_todos.count` — filtered related manager called twice per event

- **D-06:** Prefetch injection mechanism: a **FOMO-local wrapper view** in
  `solsys_code/views.py` that shadows the `tom_calendar` `render_calendar` function.
  Added to `src/fomo/urls.py` BEFORE `path('', include('tom_common.urls'))` with
  namespace `'calendar'`, so the URL resolver hits the FOMO view first.

  The wrapper view replicates `render_calendar`'s queryset with these additions:
  ```python
  events = CalendarEvent.objects.filter(...).prefetch_related(
      'telescope_label_meta',
  ).annotate(
      active_todo_count=Count('todos', filter=Q(todos__is_completed=False))
  )
  ```

- **D-07:** `active_todos.count` fixed via **Count annotation** (not Prefetch+to_attr).
  The template is updated from `event.active_todos.count` to `event.active_todo_count`
  at both occurrences (all-day events and timed events). Cleaner than to_attr: one
  extra SQL column, zero extra queries, no change to the property or model.

### Claude's Discretion

- **Wrapper view structure**: The wrapper replicates only the queryset-building portion
  of `render_calendar` (the `CalendarEvent.objects.filter(...)` + `list()` lines),
  calling the rest of the function's context-building and rendering logic directly from
  `tom_calendar.views`. If that's not cleanly extractable, the planner may choose to
  shadow the full `render_calendar` function body with the prefetch/annotation added
  — accepting the maintenance burden against the third-party function.

- **`text_color_for_bg` helper placement**: Pure Python helper function in
  `calendar_display_extras.py` (not in `calendar_utils.py`) — it's display logic,
  not a model utility.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Requirements
- `.planning/REQUIREMENTS.md` §v1.6 Requirements — DISPLAY-08, DISPLAY-09
  (exact success criteria and acceptance gates)
- `.planning/ROADMAP.md` Phase 12 section — Goal, Depends on, Success Criteria

### Source Files to Modify
- `solsys_code/templatetags/calendar_display_extras.py` —
  add `text_color_for_bg` tag; existing `PROPOSAL_PALETTE`, `NEUTRAL_SLOT_COLOR`,
  `proposal_color` are the context for the new tag
- `src/templates/tom_calendar/partials/calendar.html` —
  DISPLAY-08 CSS change (remove hardcoded `#fff !important`), inline text-color
  style; DISPLAY-09 template variable change (`active_todo_count`)
- `src/fomo/urls.py` —
  add FOMO-local calendar URL before `tom_common.urls` include
- `solsys_code/views.py` —
  new FOMO wrapper view function that shadows `tom_calendar.views.render_calendar`
  with prefetch + annotation added

### Third-Party Code (read-only — do not modify)
- `tom_calendar/views.py` (in venv) —
  `render_calendar` function is the upstream being shadowed; understand its queryset
  and context-building logic to replicate safely
- `tom_calendar/models.py` (in venv) —
  `CalendarEvent.active_todos` property and `EventTodo.todos` FK

### Tests
- `solsys_code/tests/test_calendar_display_extras.py` —
  existing Phase 9 tests; new DISPLAY-08 parametrized tests go here
- `solsys_code/tests/test_calendar_template.py` —
  existing Phase 9 integration tests; DISPLAY-09 N+1 assertion goes here

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `calendar_display_extras.py::proposal_color` — already calls `hashlib.sha256` and
  indexes into `PROPOSAL_PALETTE`; `text_color_for_bg` is a peer tag in the same module
- `PROPOSAL_PALETTE` (8 dark hex colors) — the existing palette was designed to pass
  WCAG AA with white text; `text_color_for_bg` will formally verify this (all 8 should
  return `'#fff'`)
- `NEUTRAL_SLOT_COLOR = '#5a6268'` — medium grey, also returns `'#fff'` when computed

### Established Patterns
- `@register.simple_tag` with a return value — existing pattern in `calendar_display_extras.py`
  for `proposal_color` and `status_border_css`; `text_color_for_bg` follows the same pattern
- No `__all__` — public/private by leading-underscore convention (Phase 11)
- Google-style docstrings; `D103` enforced by ruff — `text_color_for_bg` needs one
- Absolute imports in `solsys_code/` (Phase 11 decision)
- `Count` annotation pattern: `CalendarEvent.objects.filter(...).annotate(
  active_todo_count=Count('todos', filter=Q(todos__is_completed=False)))`

### Integration Points
- `calendar.html` line 85: `.cal-event-all-day { color: #fff !important; }` — remove this
- `calendar.html` line 94: `.cal-event-all-day a { color: #fff !important; }` — remove this
- `calendar.html` lines 187-192 and 202-214: all-day and timed event divs — add
  `{% text_color_for_bg bg_color as text_color %}` then `color: {{ text_color }};` inline
- `calendar.html` lines 193-195 and 219-221: `event.active_todos.count` → `event.active_todo_count`
- `src/fomo/urls.py`: new `path('calendar/', include('solsys_code.calendar_urls', namespace='calendar'))`
  inserted before `path('', include('tom_common.urls'))`

</code_context>

<specifics>
## Specific Ideas

- The `text_color_for_bg` tag name follows the existing `status_border_css` naming
  convention (describes what it returns, not what it receives).
- The WCAG relative luminance formula for an sRGB channel `c`:
  `L = c/255; return L/12.92 if L <= 0.04045 else ((L + 0.055)/1.055)**2.4`
  Then `luminance = 0.2126*R + 0.7152*G + 0.0722*B`.
  Contrast ratio = `(L_lighter + 0.05) / (L_darker + 0.05)`. Choose white (`L=1.0`) or
  black (`L=0.0`) based on which achieves ≥ 4.5:1 against the background.
- The todo reviewed but not folded: the Phase 11 "Extract site/telescope mapping" todo
  (`2026-06-23-extract-site-telescope-mapping...`) matched phase 12 in the todo scanner
  but is already resolved in Phase 11 — not applicable here.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 12-Display Polish*
*Context gathered: 2026-06-27*
