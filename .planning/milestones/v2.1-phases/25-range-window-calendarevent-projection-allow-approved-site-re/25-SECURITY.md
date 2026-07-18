---
phase: 25
slug: range-window-calendarevent-projection-allow-approved-site-re
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-07-18
---

# Phase 25 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| staff user -> CampaignRunDecisionView.post() | Authenticated staff (StaffRequiredMixin) submit approve / resolve_site / mark_cancelled / mark_weather_failure actions; only server-side model fields (run.pk, window_start/window_end, campaign.name, telescope_instrument) feed projection and title/url construction — no request free-text reaches the new code paths. | Staff session -> trusted DB model fields only |
| CLI operator -> backfill_range_calendar_events | CLI-only management command requiring shell access already equivalent to full DB access; writes CalendarEvent rows for already-APPROVED runs. All values feeding projection (run.pk, window dates, site) are trusted server-side model fields — no request/network input. | Shell access -> trusted DB model fields only |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-25-01 | Elevation of Privilege | `_set_run_status()` / `_resolve_site()` staff-only status/site actions | medium | mitigate | `StaffRequiredMixin` + the `approval_status == APPROVED` business-logic guard and the conditional `.update()` / `updated_count == 0` short-circuit preserved byte-exact (D-04) — verified via `git diff aeda66e df676f1`: the guard block is untouched, only the calendar-sync block below it changed. | closed |
| T-25-02 | Tampering | New per-night `CalendarEvent.url` key `CAMPAIGN:{pk}:{date.isoformat()}` and the `_set_run_status()` `url__startswith` lookup | low | mitigate | URL key built entirely server-side from trusted model fields (`pk` int, `window_start`/`window_end` DateField), never from request text — no injection surface. Trailing-colon prefix `f'CAMPAIGN:{run.pk}:'` prevents a one-vs-two-digit pk substring collision that would cross-mutate another run's events — confirmed present in `campaign_views.py:788-790` and exercised by `test_mark_range_window_run_updates_every_night_event`. | closed |
| T-25-03 | Information Disclosure | Per-night event titles carrying the window-context suffix | low | accept | Titles expose only campaign name, telescope_instrument, and window dates — already public on the per-campaign calendar for single-night runs; a range run's window dates are no more sensitive. No PII (contact_person/contact_email) added to any title or description. | closed |
| T-25-04 | Tampering (accidental, not adversarial) | Backfill command writing production CalendarEvent rows | low | mitigate | Idempotent by construction: pre-check skips runs with an existing `CAMPAIGN:{pk}*` event and delegates to `insert_or_create_calendar_event()`'s no-churn contract, so a re-run creates no duplicates. `--dry-run` provides a no-write preview. Confirmed by 5 passing tests plus a live UAT run against the real dev DB (pk=34 backfilled once, correctly, no duplicate rows). | closed |
| T-25-05 | Denial of Service | Per-candidate `sun_event()` ValueError aborting the whole backfill | low | mitigate | Each candidate's `_project_calendar_event()` call is wrapped in try/except ValueError that logs, reports, and continues — one un-projectable run cannot abort the backfill of the remaining qualifying runs. Verified by test AND observed live during UAT: pk=27/29 failed with "Observatory 'FTN' has no timezone set" and were skipped/logged while pk=34 still backfilled successfully in the same run (`candidates: 3, backfilled: 1, failed: 2`). | closed |

*Status: open · closed · open — below {block_on} threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-25-01 | T-25-03 | Range-window event titles expose only campaign name, telescope_instrument, and window dates — data already public on the calendar for single-night runs. No new sensitive fields introduced. | Phase 25 plan author (25-01-PLAN.md threat model) | 2026-07-18 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-18 | 5 | 5 | 0 | Claude (orchestrator, short-circuit path — threats_open: 0, register_authored_at_plan_time: true, asvs_level: 1) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-18
