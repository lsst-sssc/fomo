# Phase 40 — UI Review

**Audited:** 2026-10-09
**Baseline:** 40-UI-SPEC.md (approved 2026-10-09), Bootstrap 5.3.3 conventions
**Screenshots:** not captured (code-only audit; no dev server started against the live developer database, per orchestrator instruction)
**Interaction captures:** off (`interaction_capture: false`)

---

## Pillar Scores

| Pillar | Score | Key Finding |
|--------|-------|-------------|
| 1. Copywriting | 3/4 | Copy matches the contract, F3 corrections applied; the per-row reason input is placeholder-only with no label |
| 2. Visuals | 3/4 | Clear hierarchy and High-band marker; the sole High candidate shows a checkbox instead of a band badge and that checkbox has no accessible name |
| 3. Color | 3/4 | Accent stays within the reserved list, but every candidate row carries a green Confirm and a red Dismiss, so the destructive colour is repeated on every row |
| 4. Typography | 4/4 | Only `fw-bold` / `fw-normal`, sizes from h4, h5, `small` and `btn-sm`; matches the contract exactly |
| 5. Spacing | 3/4 | Multiples of 4 used throughout; two inline `style` attributes (documented exceptions) including a hard-coded `5rem` rather than a token |
| 6. Experience Design | 3/4 | Return state, empty states and Undo confirm all work; the band `select` auto-submits on change with no submit fallback, and the `btn-sm` targets are below 44px |

**Overall: 19/24**

No pillar scored 1. No defect found breaks task completion, so there are no BLOCKERs. All findings below are WARNINGs or minor recommendations.

---

## Top 3 Priority Fixes

1. **Band filter reloads the page on every keystroke-level change (keyboard users)** — `select#band-filter` uses `onchange="this.form.submit()"` with no submit button. Arrowing through the options on a focused closed select (common on Linux/Chromium and Firefox) fires `change` on each step and reloads the page, so a keyboard user cannot reach "High" without reloading on the way. Fix: add a visible "Apply" submit button (`btn btn-sm btn-outline-secondary`) inside the same `form.d-flex` and drop the `onchange` attribute, or move the auto-submit behind a `click`/`Enter` handler.

2. **Unlabelled controls on each candidate row** — the sole High-band checkbox (`form-check-input`, `name=candidate_ids`) has no accessible name, and the per-row reason `input` (`form-control form-control-sm`, `required`) relies on its placeholder only. Screen readers announce neither as meaningful, and the sole High candidate's band is conveyed only by the green bar and score (colour-only cue, WCAG 1.4.1). Fix: add `aria-label="Select candidate: {{ candidate.run.label }}"` to the checkbox and `aria-label="Reason for dismissing this candidate"` to the reason input (or a visually hidden `<label class="visually-hidden">`). Show the High badge alongside the checkbox rather than replacing it.

3. **Red Dismiss button repeated on every candidate row** — `btn-danger` on every row, next to `btn-success` Confirm, puts two saturated colours in each of up to dozens of rows. This works against the 60/30/10 split the contract sets (accent reserved for three named elements) and makes the High-band green bar less distinct. Fix: render Dismiss as `btn btn-sm btn-outline-danger` so it stays destructive in meaning but loses the solid fill; keep `btn-success` for Confirm, which is the one primary action the contract reserves accent for.

---

## Detailed Findings

### Pillar 1: Copywriting (3/4)

- Copy matches the Copywriting Contract: "Confirm", "Dismiss", "Undo", "Confirm selected", "Attribution", the two worklist headings, the empty-state and done-panel text, and the "Dismissed" / "Confirmed" section headings all match `40-UI-SPEC.md` verbatim.
- F3 corrections are in place: `campaign_tables.py:789-790` reads "...It will move this pair to the Dismissed section above.", and the flash in `campaign_views.py` reads "Confirmation undone — the pair is now in Dismissed." No prompt string contains an apostrophe or double quote.
- `attribution_queue.html:113-116` the per-row reason input uses `placeholder="Why doesn't this candidate match? (required)"` with no `<label>`. The placeholder disappears on typing and is not a reliable accessible name. Recommend a visually hidden label.
- Row-level verbs ("Confirm", "Dismiss") are single words; accepted per F7 because the candidate row supplies the noun. No change required.
- Generic labels check: no "Submit", "OK", "Save" or "Click here" on the attribution page. `campaignrun_submit_form.html:13` has a "Cancel" link, which is an appropriate neutral secondary action.

### Pillar 2: Visuals (3/4)

- Clear focal hierarchy: page title `h4.fw-bold.mb-4`, section headings `h5.fw-bold.mb-3`, group header rows `tr.bg-light` with `small.fw-bold`, and the High-band 4px green bar. The contract's order (Dismissed above Confirmed) is in the template.
- Empty and done panels (`div.text-center.text-muted.p-4.bg-light.mb-4`) are present for both worklists and the attribution-complete state.
- The sole High-band candidate shows a checkbox instead of the "High" badge (`attribution_queue.html` candidate-row branch), so its band is communicated by colour alone. This is F6 in the spec, accepted as out of scope for this phase, but it remains a real visual-accessibility gap on the page this phase rebuilt.
- No icons on the attribution page (Font Awesome is used only on gap analysis), consistent with the contract.
- Long text (dismissal reason, orphan title, run label) wraps with no truncation; backstopped in the spec, no visual test pins it.

### Pillar 3: Color (3/4)

- Accent (`text-bg-success`, `btn-success`, `border-success`) is confined to the three reserved elements: the High-band bar, the High badge on non-sole High rows, and per-row Confirm. Compliant.
- Confirm selected is `btn-primary` and Undo is `btn-outline-secondary`, both as specified.
- Count check on `attribution_queue.html`: `btn-success` appears once per candidate row template (and once more for the records branch, lines 113 and 202), and `btn-danger` appears on the same two rows (lines 116 and 205). Every candidate row therefore carries one green and one red solid button. This is the distribution issue behind Priority Fix 3.
- No hard-coded hex colours or `style="color..."` in the attribution template. The only inline style is `gap` and `scroll-margin-top`, both spacing.

### Pillar 4: Typography (4/4)

- Authored sizes are limited to the declared set: `h4` (24px), `h5` (20px), `small` (14px), `btn-sm` (14px), body (16px). No `text-lg`, `text-xl`, `fs-*` or `text-xs` on the attribution template.
- Weights are limited to `fw-bold` and `fw-normal`. No `font-weight-*` (Bootstrap 4) remains; the class guard `TestTemplatesUseBootstrap5ClassNames` covers the repo template directories.
- Labels (`Filter by confidence:`, evidence labels) use `small fw-bold`, consistent with the contract.

### Pillar 5: Spacing (3/4)

- Spacing utilities (`mb-3`, `mb-4`, `mt-4`, `p-3`, `p-4`, `gap-2`, `me-*`) sit on the 4px scale with no arbitrary `[...px]` values in the attribution template.
- Two inline `style` attributes on the attribution template:
  - `style="gap: 0.5rem;"` on the per-candidate forms (lines 107, 196). Documented exception in the spec (predates this phase, equals `gap-2`). Accepted.
  - `style="scroll-margin-top: 5rem;"` on both section divs (lines 260, 271). Added per F1 and documented. It is a hard-coded `5rem` that duplicates the base template's `body` padding rather than a named token. Acceptable, but if the navbar height changes this value will drift; a comment tying it to the base template's padding would help.
- The reason input is `form-control-sm` with no `max-width`, so it can stretch the action cell and push Dismiss off the cell width on narrow viewports. Not measured (no screenshots); recommend checking at 768px.

### Pillar 6: Experience Design (3/4)

**Strengths (verified in code):**
- Every decide form carries a CSRF token and the validated return state (`attribution_return_state.html`, autoescaped, one include on every form).
- Undo on Dismissed and Confirmed keeps its page and section open (D-24, D-25, D-26); `dismissed_section_open` is computed server-side from the validated state (IN-15).
- Both Undo buttons use native `confirm()` prompts with copy that matches the contract.
- Empty states exist for the two worklists, Dismissed and Confirmed tables.
- Confirm uses `formnovalidate`, so the required reason never blocks it; Dismiss is gated by the required reason and reversible through Undo.

**Gaps:**
- **Band filter auto-submit** (`onchange="this.form.submit()"`): no submit fallback and no visible button. Keyboard users trigger a full reload per option step. This is F6 in the spec; it is not a task blocker but it degrades keyboard use of the page's first control. (Priority Fix 1.)
- **Unlabelled controls** on each candidate row (checkbox and reason input). (Priority Fix 2.)
- **Touch and click targets**: `btn-sm` controls are about 31px tall, below 44px. Accepted in the spec for a desktop staff tool; noted for Phase 41.
- **No feedback during the POST round trip**: each action is a full page reload with no disabled state on the clicked button, so a double-click on Confirm can submit twice. The server returns "already confirmed or dismissed by someone else" (warning) for the second request, so the data is safe, but the staff member sees a confusing warning. Recommend disabling the button on submit (a single `onsubmit` handler or `data-` attribute), which does not require HTMX.
- **Destructive action without confirmation**: Dismiss has no browser prompt by design (the reason is the gate, and Undo is the reversal). Accepted by the spec.
- **Registry audit**: not applicable. `components.json` absent, no third-party registries (spec Registry Safety section).

---

## Files Audited

- `src/templates/campaigns/attribution_queue.html`
- `src/templates/campaigns/partials/attribution_return_state.html`
- `src/templates/campaigns/campaign_list.html` (class and copy scan only)
- `src/templates/campaigns/campaignrun_table.html` (class and copy scan only)
- `src/templates/campaigns/approval_queue.html` (class and copy scan only)
- `src/templates/campaigns/campaignrun_gap_analysis.html` (class and copy scan only)
- `src/templates/campaigns/campaignrun_submit_form.html` (copy scan only)
- `src/templates/campaigns/submission_thanks.html` (class scan only)
- `solsys_code/campaign_tables.py` (undo prompt strings, Undo button markup, Create-new-Observatory link class)
- `.planning/phases/40-notebook-isolation-attribution-page/40-UI-SPEC.md`
- `40-01` to `40-04` SUMMARY.md (40-03 and 40-04 read in full)
