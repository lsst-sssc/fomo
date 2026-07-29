---
phase: 9
slug: proposal-color-status-visual-treatment
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-25
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase` / `Client`) |
| **Config file** | none — discovery via `./manage.py test solsys_code` (pyproject.toml testpaths deliberately excludes solsys_code/, per CLAUDE.md) |
| **Quick run command** | `./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template -v 2` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~30 seconds (quick), ~60 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template -v 2`
- **After every plan wave:** Run `./manage.py test solsys_code`
- **Before `/gsd-verify-work`:** Full suite must be green; click-to-filter behavior confirmed manually in browser (UAT)
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-xx-01 | TBD | 0 | DISPLAY-04 | — | N/A | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras -v 2` | ❌ W0 (new file) | ⬜ pending |
| 09-xx-02 | TBD | 1 | DISPLAY-04 | — | `proposal_color` output is a fixed palette hex, never echoes raw `proposal` string into style | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras -v 2` | ❌ W0 | ⬜ pending |
| 09-xx-03 | TBD | 1 | DISPLAY-04 | — | Empty proposal → neutral slot `#5a6268`, not hash-of-empty-string | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras -v 2` | ❌ W0 | ⬜ pending |
| 09-xx-04 | TBD | 1 | DISPLAY-04 | — | Both all-day and timed branches render `proposal_color` output (not event.color) | integration | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing | ⬜ pending |
| 09-xx-05 | TBD | 1 | DISPLAY-05 | — | `[QUEUED]`-titled event style contains proposal hex color, not flat-grey literal `rgba(0, 0, 0, 0.45)` | integration | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing | ⬜ pending |
| 09-xx-06 | TBD | 1 | DISPLAY-06 | — | Queued and terminal-failure events render different box-shadow CSS markers; neither uses `dashed` border style | integration | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing | ⬜ pending |
| 09-xx-07 | TBD | 1 | DISPLAY-06 | — | Event that is both fallback-labeled (Phase 8, `is_verified=False`) AND queued/terminal-failure renders BOTH Phase 8 dashed border AND Phase 9 box-shadow ring composed, not overwriting | integration (Pitfall 3 regression) | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing | ⬜ pending |
| 09-xx-08 | TBD | 1 | DISPLAY-07 | — | `visible_proposals` returns legend grouped by color (colliding proposals share one entry), current-month-only, neutral slot appears as Classical schedule | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras -v 2` | ❌ W0 | ⬜ pending |
| 09-xx-09 | TBD | 1 | DISPLAY-07 | — | Rendered footer row contains `.cal-legend-swatch` entries for each visible proposal; neutral slot present if classical events exist | integration | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing | ⬜ pending |

*Task IDs are provisional (TBD) — will be filled in after PLAN.md is created during execution.*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_calendar_display_extras.py` — new file; unit tests for `proposal_color` (normalization, hashing, neutral slot, casing variants), `status_border_css` (queued/placed/terminal-failure mapping, no dash-style output), `visible_proposals` (collision grouping, current-month-only, neutral slot, ordering). Covers DISPLAY-04/06/07's pure-function logic.
- [ ] No new `conftest.py` or fixtures — `test_calendar_template.py`'s existing `setUp()` pattern (`CalendarEvent.objects.create(...)` + sidecar rows) extended with `proposal=` values for color-collision/empty-proposal cases.
- [ ] No framework install — Django test runner already configured and working.

*(Click-to-filter (D-03, DISPLAY-07) has no automated-test Wave 0 gap — it is explicitly manual-only per the table above, since this project's toolchain has no browser-automation test runner.)*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Click a legend swatch → proposal events highlight, others dim; click again → clears | DISPLAY-07 (D-03) | No Selenium/Playwright in this project; adding a JS test runner is out of scope for a single CSS-class-toggle behavior | Load calendar page, confirm visible month has ≥ 2 proposals. Click one legend swatch → verify that proposal's events are at full opacity and others are dimmed. Click same swatch → verify all events restored. Click Prev/Next month and then repeat to verify htmx-swap survival. |
| Status ring visually distinct at typical event-block width (~16-18 chars) | DISPLAY-06 | Visual inspection of rendered output | In browser, confirm queued events show a visible ring (not confused with placed events), terminal-failure events show a thicker/darker ring; neither ring uses dashed border style. |
| Colorblind CVD pass on the final palette | DISPLAY-04 (A1 assumption) | Requires human/tool judgment on color distinguishability | Before committing final `PROPOSAL_PALETTE` hex values, run through a CVD simulator (e.g. Coblis or Chromatic Vision Simulator) for deuteranopia + protanopia. No two palette entries should be visually identical under either deficiency type. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (`test_calendar_display_extras.py` new file)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
