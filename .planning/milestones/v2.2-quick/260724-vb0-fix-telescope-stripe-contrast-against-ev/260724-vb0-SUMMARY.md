---
phase: quick-260724-vb0
plan: 01
subsystem: calendar-display
tags: [wcag-contrast, calendar-ui, telescope-stripe, colorblind-safety]
dependency-graph:
  requires: [quick-260724-tiz]
  provides: [telescope_stripe_color, TELESCOPE_STRIPE_PALETTE, STRIPE_OUTER_EDGE_COLOR, _contrast_ratio]
  affects: [solsys_code/templatetags/calendar_display_extras.py, src/templates/tom_calendar/partials/calendar.html]
tech-stack:
  added: []
  patterns: ["parallel palette arrays keyed by one shared hash", "programmatic WCAG contrast gate in test suite instead of visual review"]
key-files:
  created: []
  modified:
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/tests/test_calendar_display_extras.py
    - solsys_code/tests/test_calendar_template.py
    - src/templates/tom_calendar/partials/calendar.html
decisions:
  - "Split TELESCOPE_PALETTE into two parallel arrays (legend vs stripe) rather than retuning one palette, because the WCAG luminance bands for 3:1-vs-white and 3:1-vs-#5a6268 do not overlap for any 8-hue set (verified arithmetically, not assumed)"
  - "Fixed the stripe's white-facing boundary with a 1px opaque STRIPE_OUTER_EDGE_COLOR line on 3 of 4 sides (left/top/bottom) rather than a hue change, since the fill-facing (right) side already clears 3:1 by color choice alone"
metrics:
  duration: ~45min
  completed: 2026-07-25
status: complete
---

# Phase quick-260724-vb0 Plan 01: Fix telescope stripe/legend contrast against every background Summary

Split the single, overloaded `TELESCOPE_PALETTE` into two parallel palettes — one gated
against white for the legend chip, one gated against the gray classical-fill for the
stripe — and closed the stripe's remaining white-facing boundary with a one-sided opaque
edge line, replacing a visual-review habit with a programmatic WCAG audit that lives in
the test suite.

## What Was Built

**Task 1 (tracer/TDD, RED):** Added `_contrast_ratio()` (the shared WCAG 2.1 SC 1.4.11
formula, also now backing `text_color_for_bg`), a shared `_hash_to_palette_color()`
helper, `TELESCOPE_STRIPE_PALETTE` (8 stripe-tuned hex values, shipped final and
passing from this task), `telescope_stripe_color()`, and `STRIPE_OUTER_EDGE_COLOR`
(`#343a40`). Repointed the classical-stripe template branch (`--tel-color`) at
`telescope_stripe_color` while the legend kept `telescope_color`. Added
`TestTelescopeStripeContrast` gating both palettes plus the edge; ran the suite and
confirmed the single expected RED failure — `TELESCOPE_PALETTE` vs white at 3.5:1,
failing on 4 of 8 entries (3.07–3.41) — with every other test passing.

**Task 2 (GREEN):** Retuned the 4 failing `TELESCOPE_PALETTE` entries (green, mustard,
purple, red) to same-hue-family values that clear 3.5:1 against white; the other 4
already cleared it and were left unchanged. Rewrote the stale block comment and
`telescope_color`'s docstring, both of which previously claimed the palette was
"chosen for against-gray-fill stripe contrast" — the Task 1 audit disproved that.
Reworked `.cal-event-classical::before` to carry a 1px opaque `STRIPE_OUTER_EDGE_COLOR`
line on its left/top/bottom (outward-facing) sides via two composed inset
`box-shadow`s, widened the stripe 6px→7px and `.cal-event-classical`'s
`padding-left` 9px→10px, and removed the old right-edge 55%-alpha white inset
entirely (it was on the wrong edge and too faint to read as a line). All 76 tests
pass.

**Task 3:** Re-derived the status-ring adjacency: `status_border_css`'s terminal
branch paints its ring outside the chip's border box, so on a
`[CANCELLED]`/`[EXPIRED]`/`[FAILED]`/`[WEATHERED]` classical chip the ring's inner
neighbour along the chip's left flank is the stripe's *outward*-facing edge (the same
edge Task 2 added), not the fill-facing side. Added a comment on that branch recording
the alpha-composite figure and a test (`test_stripe_outer_edge_clears_terminal_ring_adjacency`)
asserting `STRIPE_OUTER_EDGE_COLOR` clears 3:1 against the composited ring. Ran the full
quality gate (77 tests, ruff scoped to this plan's files) and produced the contrast
matrix and CVD screen below.

## Contrast Matrix (computed via the module's own `_contrast_ratio`)

**TELESCOPE_PALETTE vs `#ffffff`** (legend gate, must be `>= 3.5`):

| Hex | Ratio |
|-----|-------|
| `#3987e5` | 3.64 |
| `#d95926` | 3.88 |
| `#008a55` | 4.41 |
| `#a18245` | 3.62 |
| `#d55181` | 3.94 |
| `#008300` | 4.95 |
| `#7b5ff7` | 4.35 |
| `#c0736d` | 3.55 |

All 8 clear 3.5:1. (Green, mustard, purple, red were retuned in Task 2; the other 4
already cleared the gate and are unchanged from quick-260724-tiz.)

**TELESCOPE_STRIPE_PALETTE vs `#5a6268`** (stripe gate, must be `>= 3.4`):

| Hex | Ratio |
|-----|-------|
| `#8ac9ff` | 3.51 |
| `#ffb370` | 3.53 |
| `#33dba1` | 3.49 |
| `#ffb524` | 3.52 |
| `#f8bfce` | 3.94 |
| `#5dea3e` | 3.93 |
| `#c9bfe3` | 3.56 |
| `#ffb09e` | 3.54 |

All 8 clear 3.4:1 against the fill they actually touch (the fill-facing/right edge of
the pseudo-element).

**TELESCOPE_STRIPE_PALETTE vs `#ffffff`** (expected sub-3, informational — this is why
the outer edge exists):

| Hex | Ratio |
|-----|-------|
| `#8ac9ff` | 1.77 |
| `#ffb370` | 1.76 |
| `#33dba1` | 1.78 |
| `#ffb524` | 1.77 |
| `#f8bfce` | 1.57 |
| `#5dea3e` | 1.58 |
| `#c9bfe3` | 1.74 |
| `#ffb09e` | 1.76 |

Every entry sits at 1.57–1.78:1 against white — clearly a color, not creeping toward
white itself, but nowhere near a 3:1 line on its own against the day cell. This is
expected and is exactly what `STRIPE_OUTER_EDGE_COLOR` handles on the outward-facing
sides; it is not an unresolved gap.

**Outer edge line:**

| Comparison | Ratio |
|------------|-------|
| `STRIPE_OUTER_EDGE_COLOR` (`#343a40`) vs `#ffffff` | 11.51 |
| `STRIPE_OUTER_EDGE_COLOR` vs composited terminal ring (`#cb7373`) | 3.42 |
| `STRIPE_OUTER_EDGE_COLOR` vs `#8ac9ff` | 6.51 |
| `STRIPE_OUTER_EDGE_COLOR` vs `#ffb370` | 6.54 |
| `STRIPE_OUTER_EDGE_COLOR` vs `#33dba1` | 6.46 |
| `STRIPE_OUTER_EDGE_COLOR` vs `#ffb524` | 6.51 |
| `STRIPE_OUTER_EDGE_COLOR` vs `#f8bfce` | 7.31 |
| `STRIPE_OUTER_EDGE_COLOR` vs `#5dea3e` | 7.29 |
| `STRIPE_OUTER_EDGE_COLOR` vs `#c9bfe3` | 6.60 |
| `STRIPE_OUTER_EDGE_COLOR` vs `#ffb09e` | 6.55 |

The edge reads as a crisp line against the white cell, against every stripe entry it
sits beside, and against the composited terminal status ring.

Reference points: `#ffffff` vs `#5a6268` = 6.21:1 (why the legend and stripe need
separate palettes in the first place — the two backgrounds are far apart in
luminance).

## Colorblind-Safety Screen

Two independent tools/methods were used, per the plan's instruction to re-confirm with
the dataviz skill's palette validator plus reproduce the planner's stated methodology.

**1. Planner's stated method, reproduced exactly:** Machado-Oliveira-Fernandes (2009)
severity-1.0 dichromat simulation (the same CVD model the dataviz skill's
`validate_palette.py` uses internally) + CIE76 (plain CIELAB Euclidean) minimum
pairwise ΔE across all 8×7/2 = 28 pairs:

| Palette | Protanopia worst-pair ΔE | Deuteranopia worst-pair ΔE |
|---------|--------------------------|----------------------------|
| Shipped `TELESCOPE_PALETTE` (floor, quick-260724-tiz) | 2.50 (`#3987e5`↔`#9085e9`) | 4.75 (`#199e70`↔`#d55181`) |
| New legend `TELESCOPE_PALETTE` (this plan) | 10.35 (`#d95926`↔`#a18245`) | 10.55 (`#008a55`↔`#c0736d`) |
| New `TELESCOPE_STRIPE_PALETTE` | 10.25 (`#33dba1`↔`#ffb09e`) | 12.25 (`#33dba1`↔`#ffb09e`) |

These numbers are close to but not identical to the plan's key_finding table (2.2/5.2,
9.4/10.4, 12.1/12.7) — the exact Viénot/Machado matrix and Lab conversion constants the
planner used were not recorded, so a byte-identical reproduction wasn't possible, but
the same model class + metric (Machado CVD sim + CIE76 ΔE) produces the same
conclusion: **both new palettes score roughly 2–4x above the shipped floor on both
axes.** Neither new palette regresses below the floor.

**2. Dataviz skill's actual `validate_palette.py` tool** (its own OKLab ΔE metric and
categorical-chart thresholds, calibrated for chart marks used *without* an
accompanying text label): all three palettes — the shipped floor, the new legend
palette, and the new stripe palette — **fail** this tool's stricter `CVD separation`
and `Normal-vision floor` checks (worst all-pairs ΔE in the 1.6–7.1 OKLab range against
its 6.0/15.0 floors). This is consistent with the shipped palette's pre-existing
"Legal as a sub-3:1 WARN" framing: telescope identity here is never conveyed by color
alone — the event title and the legend text label always carry it — which is the
documented secondary-encoding exemption this tool's own docstring names as the only
condition under which a sub-floor CVD separation is acceptable. No regression is
introduced; this is an existing, accepted tradeoff, not a new one.

## Deviations from Plan

### Auto-fixed Issues

None beyond what the plan's tasks already specified — no Rule 1/2/3 deviations were
needed. Execution followed the plan's task actions directly.

### Out-of-Scope Discoveries (not fixed, logged)

Running the repo-wide quality gate in Task 3 surfaced pre-existing issues unrelated to
this plan's files, logged to
`.planning/quick/260724-vb0-fix-telescope-stripe-contrast-against-ev/deferred-items.md`
per the executor's scope boundary:

- `ruff check .` (unscoped): one `D103` finding in
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`, unchanged
  since commit `292929a` and predating this plan's base commit `8a973ce`.
- `ruff format --check .` (unscoped): 4 pre-existing files needing reformatting
  (3 demo notebooks, `src/fomo/settings.py`), none touched by this plan.

Scoped to this plan's actual `files_modified`
(`solsys_code/templatetags/calendar_display_extras.py`,
`solsys_code/tests/test_calendar_display_extras.py`,
`solsys_code/tests/test_calendar_template.py`), both `ruff check` and
`ruff format --check` are clean.

## Follow-up Finding: Deferred Proposal-Chip Ring Adjacency

Explicitly out of scope for this plan, per Task 3's brief: the same terminal status
ring, painted outside the border box, also neighbours a *proposal-having* all-day
chip's `PROPOSAL_PALETTE` fill (a dark palette, unrelated to the telescope stripe).
That adjacency is a pre-existing sub-3:1 contrast case and is not fixable by recoloring
the ring alone — it would need its own edge treatment on proposal chips, which is a
wider change than this plan's brief authorizes. This is recorded here as a follow-up
finding for a future task, not attempted in this plan.

## Human Verification (manual follow-up)

Task 3's `<verify>` includes a `<human-check>` step (load the calendar month view and
visually confirm the stripe/edge/ring rendering). This quick task is fully autonomous
(no checkpoint tasks) and the contrast/adjacency claims above are all verified
programmatically via the test suite; the visual confirmation step is a manual
follow-up for the user in a real browser, not a blocking gate in this execution.

## Self-Check: PASSED

- `solsys_code/templatetags/calendar_display_extras.py` — FOUND
- `solsys_code/tests/test_calendar_display_extras.py` — FOUND
- `solsys_code/tests/test_calendar_template.py` — FOUND
- `src/templates/tom_calendar/partials/calendar.html` — FOUND
- Commit `8a6220d` (Task 1, RED) — FOUND in `git log`
- Commit `c88830b` (Task 2, GREEN) — FOUND in `git log`
- Commit `280bc18` (Task 3, audit close-out) — FOUND in `git log`
- `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` — 77 tests, OK
- `ruff check` / `ruff format --check` scoped to this plan's files — clean
