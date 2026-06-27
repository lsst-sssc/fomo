---
phase: 10
slug: gemini-calendar-sync-command
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-27
---

# Phase 10 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| `ObservationRecord.parameters` → command processing | `parameters` JSON from a prior Gemini submission may contain a `password` key; crosses into log/stderr/stdout sinks and CalendarEvent persistence | Credential (password string) |
| Notebook fixture params → committed notebook output | Synthetic ObservationRecord params carry a `password` placeholder; command stdout is printed into committed cell output | Credential placeholder (never a real key) |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-10-01 | Information Disclosure | `handle()` per-record loop | mitigate | `safe_params = {k: v for k, v in (record.parameters or {}).items() if k != 'password'}` as first statement of each iteration; only `safe_params` referenced downstream; verified by `test_gem_secure_01_password_not_in_output` (15/15 tests pass) | closed |
| T-10-02 | Information Disclosure | except clauses / tracebacks | mitigate | `except (KeyError, ValueError) as exc` interpolates only `observation_id!r` and `type(exc).__name__` — never `safe_params` or `record.parameters`; WR-03 fix applied in code review | closed |
| T-10-03 | Tampering | CalendarEvent write path | accept | `url` key is server-derived `GEM:{prog}/{observation_id}` from trusted DB rows; no external input path; internal-only calendar | closed |
| T-10-SC | Tampering | package installs (Plans 01 + 02) | accept | No new packages installed in either plan; existing Django/tom_calendar/tom_observations/jupyter deps only | closed |
| T-10-N1 | Information Disclosure | committed notebook cell output | mitigate | Fixtures use `'password': '[redacted]'` only (never a real key); command strips `password` before any output; verified by `jupyter nbconvert` re-execution — literal `password` absent from all cell outputs | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-10-01 | T-10-03 | CalendarEvent URL is server-derived from trusted ObservationRecord DB rows; no external write path; internal calendar only — tampering risk is negligible | gsd-orchestrator | 2026-06-27 |
| AR-10-02 | T-10-SC | No new packages installed in Phase 10; all deps already audited in prior phases | gsd-orchestrator | 2026-06-27 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-27 | 5 | 5 | 0 | gsd-orchestrator (short-circuit: register_authored_at_plan_time=true, threats_open=0; T-10-01/T-10-02 verified by 15/15 passing tests; T-10-N1 verified by notebook re-execution; T-10-03/T-10-SC accepted) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-27
