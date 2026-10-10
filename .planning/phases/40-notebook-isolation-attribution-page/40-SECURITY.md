---
phase: "40"
slug: "notebook-isolation-attribution-page"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-09"
---

# Phase 40 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Register authored at plan time (40-01..40-04 `<threat_model>` blocks); verified at ASVS L1 (grep-depth evidence), short-circuit per secure-phase §3 (`threats_open: 0`, plan-time register, `asvs_level: 1`).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| notebook kernel -> developer database | A notebook run on the developer machine can reach `src/fomo_db.sqlite3`, which the live cron runner also writes | database writes (high) |
| settings resolution -> notebook writes | `local_settings.py` (imported last by settings.py) can override `DATABASES`, so the variable alone does not decide where Django writes | database routing (high) |
| /tmp scratch directory | A snapshot holds a full copy of the developer database (password hashes, tokens, sessions) for the run's duration | credentials at rest (medium) |
| committed notebook output | Pre-executed notebooks are committed with output | snapshot text (medium) |
| staff browser -> AttributionDecisionView | POSTed hidden fields (band, page, sort) are client-controlled and can be edited | redirect query values (low) |
| GET query -> attribution page forms | Query values are echoed into hidden inputs on the next render | reflected values (low) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-40-01 | Tampering | `setup_scratch_django` routing | high | mitigate | `check_resolved_database` raises `RuntimeError` before `migrate` (`docs/notebooks/fomo_nb_utils.py:105,121`); override subprocess test in `test_notebook_helper.py` | closed |
| T-40-02 | Tampering | snapshot copy of a database another process writes | high | mitigate | `?mode=ro` URI + `Connection.backup()` (`fomo_nb_utils.py:83-84`); test pins `mode=ro` (`test_notebook_helper.py:158`) and the import allow-list | closed |
| T-40-03 | Information disclosure | leftover `/tmp/fomo-notebook-db-*` snapshot | medium | mitigate | `tempfile.mkdtemp(prefix=SCRATCH_DIR_PREFIX)` (0700) with cleanup on setup failure (`fomo_nb_utils.py:150-156`); teardown `shutil.rmtree(path)` without `ignore_errors` (`:238`) | closed |
| T-40-04 | Tampering | `teardown_scratch_django` removing the wrong path | medium | mitigate | `SCRATCH_DIR_PREFIX = 'fomo-notebook-db-'` check before removal; `test_teardown_refuses_a_directory_without_the_scratch_prefix` (`test_notebook_helper.py:213`) | closed |
| T-40-05 | Elevation of privilege | `python -O` stripping the database guard | high | mitigate | `raise RuntimeError`, no `ast.Assert` in the helper (static test `test_notebook_helper.py:105`); subprocess runs with `-O` / `PYTHONOPTIMIZE=1` | closed |
| T-40-06 | Tampering | guard bypass by aliasing, nesting or stale output | high | mitigate | AST rules over top-level statements, import-based Django detection and alias bans (`test_pre_executed_notebooks.py:183,202`), execution counts 1..N; per-bypass synthetic tests | closed |
| T-40-07 | Information disclosure | credential-shaped text in committed snapshot output | medium | mitigate | `test_no_credential_shaped_text_in_any_notebook` (`test_pre_executed_notebooks.py:762`) | closed |
| T-40-08 | Tampering | a write hidden inside an `assert` | medium | mitigate | Rule R11 (`test_pre_executed_notebooks.py:58,435`); audit 236 asserts, none wrapping a write (40-02-SUMMARY) | closed |
| T-40-09 | Tampering | template edits (XSS surface) | low | mitigate | Class names only; `test_dismissal_reason_xss_payload_is_escaped` (`test_campaign_attribution_views.py:832`) passes | closed |
| T-40-10 | Spoofing | Bootstrap CSS supplied by a CDN | low | accept | Unchanged from before the phase; see AR-40-01 | closed |
| T-40-11 | Tampering | `_redirect` open redirect / header injection | high | mitigate | Path from `reverse('campaigns:attribution')`; query only from `attribution_return_state()` (`campaign_views.py:1209`; band whitelist `:1227`, page `[1-9][0-9]{0,5}` `:1198`, sort `-?[a-z_]{1,40}` `:1199`, both `fullmatch`) via `urlencode` (`:1475`); fragment from constant map `_ATTRIBUTION_RETURN_FRAGMENTS` (`:1203,1476`); `test_tampered_return_state_is_dropped` (`test_campaign_attribution_views.py:1652`) | closed |
| T-40-12 | Tampering | reflected values in hidden inputs (XSS) | medium | mitigate | Only validated values emitted via `format_html_join` (`campaign_tables.py:596`) and template autoescape; `attribution_return_state.html` has no `safe` filter (0 occurrences) | closed |
| T-40-13 | Tampering | CSRF on the decide endpoint | medium | mitigate | `test_every_attribution_form_keeps_its_csrf_token` (`test_attribution_template.py:424`) | closed |
| T-40-14 | Elevation of privilege | staff-only attribution actions | high | mitigate | `StaffRequiredMixin` on `AttributionQueueView` / `AttributionDecisionView` (`campaign_views.py:1292,1418`); `TestAttributionStaffGating` (`test_campaign_attribution_views.py:148`) passes | closed |
| T-40-15 | Denial of service | oversized page numbers | low | mitigate | Page values limited to 1-6 ASCII digits with no leading zero before use (`_RETURN_PAGE_RE`, `campaign_views.py:1198`); django-tables2 clamps out-of-range pages | closed |
| T-40-SC | Tampering | npm/pip/cargo installs | low | accept | No package installed by any phase-40 plan; see AR-40-02 | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-40-01 | T-40-10 | Bootstrap CSS comes from the CDN via `tom_common/base.html` / django_bootstrap5 exactly as before; 40-03 adds no external resource | plan-time threat model (40-03) | 2026-10-09 |
| AR-40-02 | T-40-SC | 40-01, 40-02, 40-03 and 40-04 install no package (RESEARCH Package Legitimacy Audit: none) | plan-time threat model (40-01..40-04) | 2026-10-09 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-09 | 16 | 16 | 0 | secure-phase State B, orchestrator (L1 grep evidence; short-circuit) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-09
