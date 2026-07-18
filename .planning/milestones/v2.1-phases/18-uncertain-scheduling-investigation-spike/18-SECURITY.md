---
phase: 18
slug: uncertain-scheduling-investigation-spike
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-07-09
---

# Phase 18 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Real 3I/ATLAS CSV (PII: names, emails) → committed `18-DECISION.md` | Real Contact Person / Email values must be redacted before crossing into a version-controlled file (D-01) | PII (names, emails) |
| Real CSV on local disk → repo / `.planning/` | The CSV must be read in place only; never copied into the repo or `.planning/`, never written back (D-01/D-02) | Real observational schedule data |
| PyPI (`pip install rapidfuzz`) → local venv | Scratch, package-manager install crossing into the local environment; gated by a blocking human checkpoint | Third-party package code |
| `resolve_site()` Tier 2 (MPC API create) → `Observatory` table | `resolve_site()` can persist an `Observatory` row (Tier 2) even with `create_placeholder=False`; the probe must roll this back | DB write (Observatory rows) |
| `18-DECISION.md` findings → `docs/design/uncertain_scheduling_spike.rst` | Content summarized from the decision doc into a second committed file; any residual PII would be duplicated | PII (if not filtered) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-18-01 | Information Disclosure | Verbatim CSV cell text pasted into `18-DECISION.md` | high | mitigate | D-01 redaction: Contact Person / Email replaced with `<REDACTED>`; verified via email-address grep (clean) and 3 "redacted per D-01" notes present in the committed file | closed |
| T-18-02 | Information Disclosure | Real CSV copied/written into the repo or `.planning/` | high | mitigate | Probe (`fuzzy_match_probe.py`) reads the CSV in place from its local path only; verified no CSV file was ever added to the repo (`git log --all --diff-filter=A` shows no real-data CSV) | closed |
| T-18-03 | Tampering | Accidental `Observatory`-row writes from `resolve_site()` Tier 2 | medium | mitigate | Every `resolve_site()` call in `fuzzy_match_probe.py` passes `create_placeholder=False` AND is wrapped in a `transaction.atomic()` block ending in `transaction.set_rollback(True)` — verified directly in the probe source on disk; `Observatory.objects.count()` confirmed unchanged (8) at start and end of the probe run per `18-01-SUMMARY.md` | closed |
| T-18-SC-01 | Tampering | Package install (`pip install rapidfuzz`, PyPI) — Plan 18-01 | high | mitigate | Task 1 `checkpoint:human-verify gate="blocking-human"` genuinely halted execution and required explicit human confirmation (obtained live during this phase's execution — RESEARCH.md's legitimacy audit reviewed, install confirmed scratch-only, never added to `pyproject.toml`) before the install proceeded | closed |
| T-18-05 | Information Disclosure | Copying evidence from `18-DECISION.md` into the durable `.rst` | high | mitigate | `docs/design/uncertain_scheduling_spike.rst` is a prose+table summary only — verified no raw redacted CSV blocks pasted in (only column-name references in prose), no email-address pattern present | closed |
| T-18-SC-02 | Tampering | Package installs (npm/pip/cargo) — Plan 18-02 | low | accept | No package installs in Plan 18-02 — it is documentation-only (completing `18-DECISION.md`'s Recommendation section and writing the durable `.rst`); verified no install commands appear in its commits | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on (high) count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-18-01 | T-18-SC-02 | Plan 18-02 is documentation-only (no code, no package installs); the low-severity "package install" threat category from the plan's threat model template does not materialize in this plan's actual scope | gsd-secure-phase (retroactive, L1 grep-depth) | 2026-07-09 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-09 | 6 | 6 | 0 | gsd-secure-phase orchestrator (L1 grep-depth, short-circuit per ASVS level 1 + plan-time-authored register — no auditor sub-agent spawn required) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-09
