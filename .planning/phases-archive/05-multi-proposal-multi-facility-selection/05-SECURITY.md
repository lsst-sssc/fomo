---
phase: 05
slug: multi-proposal-multi-facility-selection
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-19
---

# Phase 05 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| operator CLI -> ORM | The `--proposal` free-text argument crosses into an ORM `JSONField` filter (`parameters__proposal__in`). Operator-invoked, but still untrusted free text. | Untrusted string -> parameterized ORM lookup |
| settings/credentials -> logs | `api_key` lives in `FACILITIES`; the D-07 skip path and D-08 summary write to stdout/stderr — a credential must never cross into log output. | Credential/settings value -> process stdout/stderr |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-05-01 | Tampering | `--proposal` value used in `parameters__proposal__in` ORM lookup (Task 2) | accept (already mitigated) | Django ORM parameterizes `JSONField.__in` bound values — no string interpolation. `_parse_proposal_arg` is pure string manipulation (split/strip/dedupe) with no SQL/shell/HTML sink. Verified live: `_parse_proposal_arg` and the `.filter(parameters__proposal__in=codes)` call site use the ORM's `__in` lookup exclusively (`solsys_code/management/commands/sync_lco_observation_calendar.py:162,234,236`). | closed |
| T-05-02 | Information Disclosure | D-07 skip-and-log path + D-08 summary line (Task 2) | mitigate | Log ONLY `record.observation_id`, `record.facility`, and the exception message. Never write `facility.facility_settings`, `api_key`, or any raw settings dict to stdout/stderr. Verified independently: `grep -v '^#' solsys_code/management/commands/sync_lco_observation_calendar.py \| grep -c api_key` returns `0`. | closed |
| T-05-SC | Tampering | npm/pip/cargo installs | accept (N/A) | No package installs this phase — `SOARFacility` is an existing import from already-installed `tom_observations`. Verified: `git diff`/`git log` on `pyproject.toml` between the phase's base commit and HEAD show zero changes. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-19 | 3 | 3 | 0 | /gsd:secure-phase orchestrator (plan-time register, short-circuit per threats_open=0 + register_authored_at_plan_time=true) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-19
