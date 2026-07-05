---
phase: 17
slug: coverage-gap-analysis-deferrable-to-v2-1
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-07-05
---

# Phase 17 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

Register built from the `<threat_model>` blocks in all three plan files (`17-01-PLAN.md`,
`17-02-PLAN.md`, `17-03-PLAN.md`) — `register_authored_at_plan_time: true`. No plan's SUMMARY.md
carried a `## Threat Flags` section. Verified by grep-level (ASVS L1) inspection of the current
code, post code-review-fix (commits `8d93714`, `2b7a7e8`), not by re-scanning for new threats.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| client GET params → gap computation | campaign/target/site/date-range values arriving from the request cross into cache-key construction and the ORM query | user-controlled pk/date values |
| computed result → shared cache backend | one campaign's gap result must never be readable under another campaign/target/site's key | cached gap-result dict (dates, run metadata) |
| rendered trigger button → server | the D-14 disabled button is a UX affordance only; a user can still hand-craft a GET to the gap URL | none (client-side only, no trust placed here) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-17-01 | Tampering / Elevation of Privilege | `CampaignGapAnalysisView` target_pk/site_pk (IDOR) | high | mitigate | Server-side re-derives `campaign.targets`/allowed-site sets and validates submitted pks before any query; `_as_pk_or_none()` guard (added in code-review fix `2b7a7e8`, CR-01) rejects non-numeric pks with a 400 instead of crashing. Verified: `solsys_code/campaign_views.py:361,417-439`; `TestGapAnalysisView.test_rejects_out_of_scope_target_and_site` passes. | closed |
| T-17-02 | Denial of Service | `clamp_date_range` (unbounded date range) | high | mitigate | 180-day hard cap enforced server-side via `form.cleaned_data.get('end_date')` (form-validated, per code-review fix `2b7a7e8`, WR-03) regardless of client input; `end` also floored at `start` (fix `8d93714`, WR-02). Verified: `solsys_code/campaign_gap.py` `clamp_date_range`; `solsys_code/campaign_views.py:446-447`. | closed |
| T-17-03 | Information Disclosure | `build_gap_cache_key` (cross-campaign cache collision) | medium | mitigate | Cache key includes all four dimensions (campaign, target, site, date range); null target encoded as literal `'none'`, never omitted. Verified: `solsys_code/campaign_gap.py:66-74` docstring + implementation. | closed |
| T-17-04 | Tampering | D-14 client-side button gating bypass | low | accept | Hidden/disabled "Show Coverage Gaps" button is defense-in-depth only; a direct GET to `campaigns:gap_analysis` is fully re-validated server-side by T-17-01's mitigation. No client-side gate is relied upon for security. | closed |
| T-17-SC | Tampering | pip installs (all 3 plans) | low | accept | No new packages introduced by this phase (RESEARCH.md Package Legitimacy Audit: none). | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on (high) count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-17-01 | T-17-04 | D-14 button gating is a UX affordance, not a security boundary — the real access control lives server-side in `CampaignGapAnalysisView` (T-17-01) | plan-time (17-03-PLAN.md threat model) | 2026-07-05 |
| AR-17-02 | T-17-SC | No new third-party packages introduced across all three plans (template + logic + test changes only) | plan-time (17-01/02/03-PLAN.md threat models) | 2026-07-05 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-05 | 5 | 5 | 0 | /gsd-secure-phase (grep-level, ASVS L1, short-circuit per register_authored_at_plan_time=true) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-05
