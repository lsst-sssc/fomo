---
phase: 15
slug: per-campaign-table-view-read-path
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-07-03
---

# Phase 15 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| anonymous/non-staff browser → `CampaignRunTableView` | Unauthenticated identity + attacker-controlled GET params (`run_status`, `open_to_collaboration`, sort, page, `pk`) cross into the queryset layer | `CampaignRun` rows incl. PII fields (contact_person/contact_email) |
| anonymous browser → target-detail page (`campaign_links` inclusion tag) | Unauthenticated request triggers a per-render `TargetList` membership query scoped by the rendered target | Campaign name/link only, no PII |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-15-01 | Information Disclosure | `CampaignRunTableView` contact_person/contact_email exposure to anonymous/non-staff | high | mitigate | `get_queryset()` returns `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` (`solsys_code/campaign_views.py:22-68`) excluding both contact fields so SQL never selects them for non-staff; `get_table_kwargs()` adds `exclude=('contact_person','contact_email')` as defense in depth. Verified present in code (grep) and independently re-proven by `15-VERIFICATION.md` truth #3 (`TestContactFieldGating` — 4/4 pass, anonymous context rows are dicts missing both keys). | closed |
| T-15-02 | Tampering | `run_status`/`open_to_collaboration` GET params driving DB filter | low | mitigate | `CampaignRunFilterSet.run_status` is an explicit `MultipleChoiceFilter(choices=CampaignRun.RunStatus.choices)` (`solsys_code/campaign_filters.py:20-21`); `open_to_collaboration` is an auto `BooleanFilter`. django-filter builds parameterized ORM filters — no raw SQL interpolation (ASVS V5). Verified present in code. | closed |
| T-15-03 | Tampering | Arbitrary/invalid campaign `pk` in URL | low | mitigate | `path('<int:pk>/', ...)` integer converter (`solsys_code/campaign_urls.py:14`) + `get_object_or_404(TargetList, pk=...)` (`solsys_code/campaign_views.py:79`) returns 404, no internals leaked. Verified present in code. | closed |
| T-15-04 | Information Disclosure | `campaign_links` inclusion tag surfacing campaigns unrelated to the rendered target, or non-campaign `TargetList`s | low | mitigate | Query is `TargetList.objects.filter(targets=target, campaign_runs__isnull=False).distinct()` (`src/templatetags/solsys_code_extras.py:25`) — scoped to the specific target's memberships AND to lists with ≥1 run; no cross-target or non-campaign leakage (D-01/Pitfall 3). Verified present in code and independently re-proven by `15-VERIFICATION.md` truth D3. | closed |
| T-15-05 | Information Disclosure | Navbar/campaign links exposing the read path to anonymous users | low | accept | Intentional per D-04 (campaign list + table are open to anonymous, matching `AUTH_STRATEGY='READ_ONLY'`); contact PII gating is enforced separately by T-15-01, not here. | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on (high) count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

No package-manager installs occurred in either plan (django-tables2/django-filter were already
installed and in `INSTALLED_APPS` per `15-RESEARCH.md`'s Package Legitimacy Audit), so no
supply-chain (T-15-SC) threat or install checkpoint applies.

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-15-01 | T-15-05 | Campaign list/table read path is intentionally open to anonymous visitors per decision D-04, matching FOMO's `AUTH_STRATEGY='READ_ONLY'` posture for `OPEN` targets. Contact PII — the only sensitive data in scope — is independently gated by T-15-01's mitigation, so exposing discoverability of the read path itself carries no incremental disclosure risk. | 15-02-PLAN.md (plan-time disposition) | 2026-07-03 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-03 | 5 | 5 | 0 | /gsd-secure-phase (orchestrator, L1 grep-depth — short-circuit per asvs_level==1 and register_authored_at_plan_time==true; mitigations independently cross-checked against `15-VERIFICATION.md`) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-03
