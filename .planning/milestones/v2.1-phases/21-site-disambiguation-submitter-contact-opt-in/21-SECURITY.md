---
phase: 21
slug: site-disambiguation-submitter-contact-opt-in
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-07-11
---

# Phase 21 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| External MPC API → backend | Untrusted external strings (`name_utf8`, `short_name`, `old_names`) enter the cached candidate pool | Free-text site names/codes |
| Submitter `site_raw` → fuzzy matcher → rendered HTML | Submitter-controlled free text is fed to `difflib` and rendered into `<input>`/`<datalist>` markup | Free-text site code/name |
| Anonymous visitor → per-campaign table queryset | Untrusted-role request must never receive PII for a run that did not opt in | `contact_person`/`contact_email` |
| Public submitter → submission form | Submitter sets a self-disclosure flag on their own contact info | Boolean opt-in flag |
| Staff `site_selection` POST → `resolve_site()` | Staff free text routed through the existing length-guarded resolver | Free-text site code/name |
| `?next=` query param → post-create redirect | Untrusted redirect target must be validated before use | URL/path string |
| Staff browser → approve/reject decision endpoint | The refactored single form still carries per-row CSRF and staff-only submission | CSRF token, staff session |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-21-01 | Tampering | `site_raw`/MPC-sourced candidate strings entering the pool and rendered in `render_site()`'s inline `<input>`/`<datalist>` | high | mitigate | All interpolation via `format_html`/`format_html_join` positional args (Django auto-escapes); never `mark_safe` or string concatenation. Verified live: no `mark_safe` in `campaign_tables.py`; `test_site_raw_script_injection_is_escaped_not_rendered_raw` asserts `<script>` is escaped in the response. | closed |
| T-21-02 | Information Disclosure | `CampaignRunTableView.get_queryset` non-staff branch | high | mitigate | Per-row PII gated at the SQL `SELECT` via `Case`/`When` conditional annotation; `ALLOWED_FIELDS_FOR_NON_STAFF` not widened. Verified live: `Case`/`When` present in `campaign_views.py`, opted-out rows' contact fields excluded from the non-staff `.values()` dict at the queryset level. | closed |
| T-21-03 | Tampering | Refactored single-`<form>` `render_actions()` / approve-reject decision POST | low | accept | CSRF mechanism unchanged — one `get_token(self.request)` mint per form; `StaffRequiredMixin` + `http_method_names=['post']` unchanged. Verified live: `get_token` call present, minted once per row form. | closed |
| T-21-04 | Denial of Service | Per-request bulk MPC fetch (1.5 MB/~1.3s) and staff `site_selection` free text in `post()` | high | mitigate | `build_site_candidates()` caches the merged pool (`django.core.cache`, 24h TTL); pool built once per request in `get_context_data`, not per row. `site_selection` routed through `resolve_site()`'s existing `_MAX_OBSCODE_LEN` guard — oversized input flagged, no tier attempted. Verified live: `cache.get`/`cache.set` and `_MAX_OBSCODE_LEN` guard both present. | closed |
| T-21-05 | Information Disclosure | Inverted/misapplied opt-in condition in the `Case`/`When` annotation | high | mitigate | Tested both directions (opted-in exposes, opted-out blanks) at the queryset layer. Verified live: `test_opted_in_row_exposes_contact_in_non_staff_values`, `test_opted_out_row_blanks_contact_in_non_staff_values`, plus anonymous-visitor and staff-sees-both variants all present in `test_campaign_views.py`. | closed |
| T-21-06 | Tampering | `CreateObservatory.get_success_url` `?next=` (open redirect) | high | mitigate | `?next=` validated with `url_has_allowed_host_and_scheme(allowed_hosts={request.get_host()})`; off-host/bad-scheme targets fall back to the detail redirect. Verified live: guard present in `solsys_code_observatory/views.py`; the round-trip itself was confirmed reachable after `21-REVIEW-FIX.md`'s CR-02 fix added the hidden `next` field to `observatory_create.html`. | closed |
| T-21-BF | Availability | MPC API outage/malformed response during candidate build | medium | mitigate | Narrow `except` around the bulk fetch falls back to the local-only `Observatory` pool, never raises into the approval-queue render. Verified live: except clause now also catches `AttributeError` per `21-REVIEW-FIX.md`'s WR-01 (bulk-endpoint shape drift no longer crashes the page). | closed |
| T-21-SAT | (documented limitation) | satellite-type `site_selection` (obscodes 250/274/289) | low | accept | Falls through to `(None, True)` via `resolve_site()`'s pre-existing `to_observatory()` TypeError path — safe (no crash/fabrication), documented with a code comment, not fixed this phase (pre-existing Phase 18-flagged bug, out of scope). Verified live: comment present at `campaign_views.py:357`. | closed |
| T-21-SC | Tampering | npm/pip/cargo installs | n/a | accept | No packages installed this phase — `difflib` is stdlib; `django.core.cache`/`requests` pre-existing. No supply-chain checkpoint required. | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on (high) count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-21-01 | T-21-03 | CSRF mechanism carried through unchanged from the pre-existing two-form design; the single-form refactor doesn't weaken it | GSD security audit (plan-time disposition) | 2026-07-11 |
| AR-21-02 | T-21-SAT | Pre-existing bug (Phase 18-flagged `to_observatory()` TypeError on satellite-type MPC records) intentionally left unfixed this phase — safe fallback, no crash or fabrication | GSD security audit (plan-time disposition) | 2026-07-11 |
| AR-21-03 | T-21-SC | No packages installed this phase | GSD security audit (plan-time disposition) | 2026-07-11 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-11 | 9 | 9 | 0 | /gsd-secure-phase orchestrator (L1 grep-depth, short-circuited per plan-time-authored register + ASVS L1) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-11
