# Phase 12: Display Polish - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-27
**Phase:** 12-display-polish
**Areas discussed:** DISPLAY-09 N+1 scope, DISPLAY-08 neutral slot

---

## DISPLAY-09 N+1 scope

### Question 1: Fix active_todos N+1 alongside telescope-label?

| Option | Description | Selected |
|--------|-------------|----------|
| Both (Recommended) | Prefetch telescope_label_meta + fix active_todos in one pass — same loops, cleaner now | ✓ |
| Telescope-label only | Stay strictly within DISPLAY-09 spec scope | |

**User's choice:** Both — fix telescope_label_meta AND active_todos.count together.

---

### Question 2: Prefetch injection mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Template tag side-effect (Recommended) | `{% prefetch_calendar_sidecars weeks %}` calling `prefetch_related_objects()` — no URL/view changes | |
| URL shadow / wrapper view | FOMO-local wrapper in urls.py shadowing tom_calendar URL, adds prefetch to queryset | ✓ |

**User's choice:** URL shadow / wrapper view.

---

### Question 3: How to fix active_todos in the wrapper view

| Option | Description | Selected |
|--------|-------------|----------|
| Count annotation (Recommended) | `annotate(active_todo_count=Count('todos', filter=Q(...)))` + template change to `event.active_todo_count` | ✓ |
| Prefetch with to_attr | `Prefetch('todos', ..., to_attr='active_todos_list')` + `\|length` in template | |
| Skip active_todos fix | Only fix telescope_label_meta | |

**User's choice:** Count annotation — one extra SQL column, zero extra queries, clean template update.

---

## DISPLAY-08 neutral slot

### Question 4: Cover NEUTRAL_SLOT_COLOR alongside palette entries?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, include it (Recommended) | #5a6268 is also rendered as all-day event background; consistent to compute its text color too | ✓ |
| Palette only | Strictly DISPLAY-08 scope — NEUTRAL_SLOT_COLOR not in PROPOSAL_PALETTE | |

**User's choice:** Include NEUTRAL_SLOT_COLOR — consistent coverage, trivial extra test.

---

### Question 5: CSS mechanism for removing hardcoded #fff

| Option | Description | Selected |
|--------|-------------|----------|
| Remove from CSS, add inline style (Recommended) | Delete `color: #fff !important` from CSS class; add `color: {{ text_color }}` inline | ✓ |
| Keep CSS, override with inline !important | Leave CSS intact; use `color: {{ text_color }} !important` in inline style | |

**User's choice:** Remove from CSS — no `!important` battles, cleaner markup.

---

## Claude's Discretion

- **`text_color_for_bg` placement**: Helper function lives in `calendar_display_extras.py` (display logic, not model utility)
- **Wrapper view structure**: If the tom_calendar render_calendar queryset isn't cleanly extractable, shadow the full function body (accepting maintenance burden vs. fragile partial override)
- **Test structure**: Parametrize over all 9 color values (8 palette + neutral slot) in `test_calendar_display_extras.py`

## Deferred Ideas

None — discussion stayed within phase scope.
