---
phase: 24
slug: operator-and-usage-runbook-documentation-for-the-telescope-r
status: verified
threats_open: 0
asvs_level: 1
created: 2026-07-18
---

# Phase 24 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| author prose → public hosted docs (ReadTheDocs) | Hand-written runbook/troubleshooting content is published to a permanently-hosted, publicly-readable site. Any real personal data written into an example crosses from private dev artifacts into public disclosure. | Potential PII (contact_person/contact_email) if an illustrative example were copied verbatim from real data |
| doc source → Sphinx build pipeline | RST is consumed by the existing pre-commit `sphinx-build` hook and CI; no code execution, no user input accepted. | None — build-time only, no runtime data |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-24-01 | Information Disclosure | `docs/runbooks/telescope_runs_calendar.rst` troubleshooting examples | high | mitigate | Verified directly: `grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'` against the shipped file returns zero matches — no email addresses of any kind appear in the runbook, real or placeholder. The one verbatim string quoted is `Observatory 'FTN' (obscode=F65) has no timezone set` (an MPC obscode + telescope short-name, not personal data). Independently re-confirmed by both `24-REVIEW.md` (code reviewer) and `24-VERIFICATION.md` (verifier). | closed |
| T-24-02 | Information Disclosure | `docs/installation.rst` onboarding subsection | low | accept | Generic Django/manage.py orientation content only; no real data involved. Below the `block_on: high` threshold. | closed |
| T-24-SC | Tampering | pip installs (supply chain) | low | accept | Zero packages installed this phase — verified via `git diff 77ae8d4..HEAD --stat -- pyproject.toml docs/requirements.txt` (empty). D-03 explicitly rejected adding `sphinx-django-command`. Below the `block_on: high` threshold. | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-24-01 | T-24-02 | Generic Django/manage.py onboarding content carries no real data; low severity, below block-on threshold. | 24-01-PLAN.md threat model (plan-time disposition) | 2026-07-18 |
| AR-24-02 | T-24-SC | No dependency was added in this phase (docs-only); low severity, below block-on threshold. | 24-01-PLAN.md threat model (plan-time disposition) | 2026-07-18 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-18 | 3 | 3 | 0 | `/gsd-secure-phase` orchestrator — register authored at plan time (`24-01-PLAN.md` `<threat_model>`), ASVS level 1 short-circuit: threats_open reached 0 via direct L1 grep-depth verification (no auditor subagent needed) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-18
