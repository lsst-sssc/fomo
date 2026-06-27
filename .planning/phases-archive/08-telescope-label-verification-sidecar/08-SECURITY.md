---
phase: 08
slug: telescope-label-verification-sidecar
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-06-25
---

# Phase 08 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| LCO API → sync command | External API response is parsed into `telescope_api_failed`; only a derived boolean (`not telescope_api_failed`) reaches the new field — no raw API string is persisted into `is_verified`. | Derived boolean only |
| sync command → DB (sidecar row) | Trusted, internal management-command write off the request path; no user-supplied input reaches `is_verified`. | Internal write |
| DB (CalendarEvent / sidecar) → rendered HTML | Template interpolates `{{ event.color }}` (existing) and a fixed tooltip string + fixed CSS literal (new). The new `title=`/`style=` additions interpolate NO DB-sourced free text. | Fixed strings only |
| User browser → calendar view | Existing read-only calendar view (TOM `AUTH_STRATEGY='READ_ONLY'`), unchanged — no new view or permission boundary added. | Read-only HTTP |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-08-01 | Tampering | `is_verified` BooleanField write in sync command | low | accept | Computed internally from `telescope_api_failed` (trusted), never from external/user input; a BooleanField cannot carry injected markup. | closed |
| T-08-02 | Repudiation | Orphaned sidecar row after CalendarEvent deletion | low | mitigate | `on_delete=models.CASCADE` on the `OneToOneField` (`solsys_code/models.py:13-15`) deletes the sidecar row automatically when its parent `CalendarEvent` is deleted — verified in code. | closed |
| T-08-03 | Information disclosure | Demo notebook executed output | low | accept | Notebook uses mocked/fixture data, not live credentials or real proposal data; no secrets are committed in the executed output. | closed |
| T-08-04 | Tampering | New `title=`/`style=` attributes in `calendar.html` | low | mitigate | Both new attributes interpolate only a fixed, hardcoded tooltip sentence and a fixed dashed-border CSS literal (`calendar.html:160-177`) — never `event.proposal`/`event.title`/any DB free text. Verified in template source. | closed |
| T-08-05 | Denial of service | Reverse-O2O accessor read per event in the month-grid loop (N+1) | low | accept | Per CONTEXT.md locked discretion (DISPLAY-09 deferred to v2): accepted as-is for current calendar-event volume; batching deferred to a future phase. | closed |
| T-08-06 | Denial of service | Missing sidecar row raising on template read (would 500) | low | mitigate | The `== False` comparison + Django's silenced `ObjectDoesNotExist` degrades a missing row to the verified branch; confirmed by `test_calendar_renders_200_including_no_sidecar_row_events` in `solsys_code/tests/test_calendar_template.py`. | closed |
| T-08-SC | Tampering | npm/pip/cargo installs (Plan 01) | n/a | accept | No new packages installed; Django `OneToOneField`/`update_or_create` are core. | closed |
| T-08-SC | Tampering | npm/pip/cargo installs (Plan 02) | n/a | accept | No new packages installed. | closed |

*Status: open · closed · open — below {block_on} threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|--------------|------|
| AR-08-01 | T-08-01 | `is_verified` is derived internally from a trusted boolean, never from external/user input. | Plan 08-01 threat model | 2026-06-25 |
| AR-08-02 | T-08-03 | Demo notebook uses mocked/fixture data only; no secrets in committed output. | Plan 08-01 threat model | 2026-06-25 |
| AR-08-03 | T-08-05 | Per-event reverse-accessor read accepted at current calendar-event volume; batching deferred to v2 (DISPLAY-09). | Plan 08-02 threat model | 2026-06-25 |
| AR-08-04 | T-08-SC | No new package installs in either plan. | Plan 08-01 / 08-02 threat models | 2026-06-25 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-25 | 8 | 8 | 0 | gsd-secure-phase (orchestrator, L1 grep-depth, register authored at plan time) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-25
