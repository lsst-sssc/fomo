---
phase: 14
slug: campaign-data-model-bootstrap-import
status: verified
threats_open: 0
asvs_level: 1
created: 2026-07-03
---

# Phase 14 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Operator CSV file → parser/DB | Untrusted, messy free-text cells from the 3I/ATLAS coordination sheet cross into `CampaignRun` DB writes via `import_campaign_csv` | Free-text (telescope, times, status, comments); PII (contact person/email) once the live sheet is imported |
| `import_campaign_csv` → MPC Obscodes API | One outbound HTTPS GET per unresolved, valid-length Site Code, via `MPCObscodeFetcher.query()` | Site code only (JSON body param, not URL-interpolated) |
| CLI filepath → filesystem | Operator supplies an arbitrary path to read (CLI trust boundary, not web-facing) | Local file read |
| Repository git history ← notebook/fixture | Committed `import_campaign_csv_demo.ipynb` output and `campaign_sample.csv` fixture rows are permanent in git history | Fixture/demo data only — real PII here would be effectively unrecoverable |
| Notebook execution → external network | A live MPC API call inside a committed pre-executed notebook would be a flaky, uncontrolled dependency | None in practice — fixture is scoped to locally-seeded site codes |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-14-01 | Tampering | CSV field parsing (V5 input validation) | medium | mitigate | Site Code length-checked against `Observatory.obscode.max_length` before any DB write (no truncation/fabrication); targeted UT-time regexes (`_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`, no permissive parser that "succeeds" on garbage) — am/pm markers now correctly applied (post-review fix CR-01, `campaign_utils.py:40-59`); `map_observation_status` safe `REQUESTED` default with negation-aware matching (post-review fix WR-08); upfront CSV header validation raises `CommandError` on missing required columns before any row processing (post-review fix WR-09, `import_campaign_csv.py`) | closed |
| T-14-02 | Information Disclosure | `CampaignRun.contact_person`/`contact_email` (V8 PII at rest) | high | mitigate | Demo fixture is synthetic, `@example.com`-only (verified — no real address pattern present); skipped-row stderr log no longer dumps the full row dict, only natural-key fields (post-review fix WR-06, `import_campaign_csv.py`); display-side auth gating for real imported PII is explicitly deferred to Phase 15 (VIEW-03) per the original Plan 14-01 mitigation plan, not an in-scope gap for this phase | closed |
| T-14-03 | Information Disclosure | `import_campaign_csv <filepath>` reading arbitrary paths | low | accept | Same accepted trust boundary as the existing `load_telescope_runs <filepath>` CLI; operator-run tool, not a web upload; `open()` wrapped in `OSError`→`CommandError` | closed |
| T-14-04 | Tampering / Information Disclosure | SSRF via Site Code into `MPCObscodeFetcher` | low | accept | Verified: `requests.get('https://data.minorplanetcenter.net/api/obscodes', json={'obscode': obscode}, ...)` — URL is a hardcoded literal; `obscode` is a JSON body param, never URL-interpolated (`solsys_code_observatory/utils.py:47-49`) — no attacker-controlled URL construction is possible | closed |
| T-14-05 | Tampering | `contact_email` format (V5) | low | mitigate | Verified: `contact_email = models.EmailField(...)` (`models.py:100`) — model-layer format validation on `full_clean()` rather than a bare `CharField` | closed |
| T-14-06 | Tampering | CSV/spreadsheet formula injection in free-text cells (e.g. `=HYPERLINK(...)`) | low | accept | Verified: no CSV/spreadsheet export or writer sink exists anywhere in `campaign_utils.py` or `import_campaign_csv.py` (grep confirmed) — this phase only imports CSV, never re-exports; flagged for any future CSV-export phase | closed |
| T-14-07 | Denial of Service | Flaky notebook re-execution from a live MPC API call | medium | mitigate | Verified: `campaign_sample.csv`'s non-blank Site Codes (`{309, F65, 705}`) exactly match the notebook's locally-seeded `Observatory` obscodes — every site resolution hits tier 1 offline; tier-2/tier-3 network paths are covered separately by Plan 02's mocked tests | closed |
| T-14-SC | Tampering | pip installs | low | accept | No new packages added across the 3 plans or the post-review fix pass — stdlib (`csv`, `re`, `datetime`) plus already-installed `requests`/Django only | closed |

*Status: open · closed · open — below `high` threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above `workflow.security_block_on` (`high`) count toward `threats_open`*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-14-01 | T-14-03 | Operator-run CLI tool reading a local path is the same accepted trust boundary as the pre-existing `load_telescope_runs` command; not a web-facing upload surface | Plan 14-02 threat model (discuss-phase carried forward) | 2026-07-02 |
| AR-14-02 | T-14-04 | `requests.get` targets a hardcoded MPC URL with `obscode` passed as a JSON body parameter, not URL-interpolated — no attacker-controlled URL construction is possible; verified directly against `solsys_code_observatory/utils.py` | Plan 14-02 threat model, re-verified during `/gsd-secure-phase` | 2026-07-03 |
| AR-14-03 | T-14-06 | This phase only imports CSV data, never re-exports to a spreadsheet — no formula-injection sink exists yet; flagged for any future CSV-export phase | Plan 14-02 threat model | 2026-07-02 |
| AR-14-04 | T-14-SC | No new third-party packages introduced by this phase's plans or its post-review fix pass | Plans 14-01/14-02/14-03 threat models | 2026-07-02 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-03 | 8 | 8 | 0 | Claude (orchestrator, `/gsd-secure-phase` — L1/ASVS-1 grep-depth verification; register authored at plan time across all 3 plans, so auditor spawn short-circuited per workflow rule) |

**Note on scope:** T-14-01 and T-14-02's mitigation evidence includes two fixes (CR-01 am/pm parsing, WR-06 PII-safe logging) that were applied during the post-plan-execution deep code review (`14-REVIEW.md`/`14-REVIEW-FIX.md`), not originally anticipated in the Plan 14-01/14-02 threat models. Both are now verified present in the current codebase and are reflected in this register's mitigation column.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-03
