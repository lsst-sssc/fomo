---
phase: 19
slug: window-schema-migration
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-07-10
---

# Phase 19 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| concurrent writers → `CampaignRun` table | Two concurrent submissions/imports may both pass an app-level uniqueness check before either commits (race) | Observing-run scheduling data |
| migration `RunPython` → stored rows | One-time destructive dedup deletes rows during the hard-cutover migration; must leave an audit trail | Existing `CampaignRun` rows (incl. contact fields on deleted duplicates) |
| `claimed_dates()` queryset → cached gap result | Fetched `CampaignRun` fields flow into `undated_runs`/`unattributed_runs` and are cached | Must never include contact PII |
| non-staff request → per-campaign table SELECT | The rendered queryset must never fetch contact PII for non-staff | Contact person/email |
| public submitter → `CampaignRunSubmissionForm` | Untrusted free-text/date input crosses into `CampaignRun.create()` | Free-text observing-run details, contact info |
| approve click → `sun_event()`/Observatory data | Messy `Observatory` records (blank timezone) can raise mid-projection | Site geodetic/timezone data |
| CSV file → `CampaignRun` rows | Batch import of untrusted sheet data; concurrent/repeat imports must not create colliding rows | Bulk historical observing-run + contact data |
| skip logging → operator | Skipped-duplicate diagnostics must name only natural-key fields, never row PII | Log output visible to operators |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-19-01a | Tampering/Repudiation | `CampaignRun` natural key, both branches (`models.py`) | high | mitigate | Two DB-level partial `UniqueConstraint`s (`unique_campaign_run_resolved_window`, `unique_campaign_run_tbd_natural_key`) replace the single old constraint — verified present in `models.py`/migration `0004` | closed |
| T-19-03 | Repudiation | Migration `0004` `dedupe_tbd_collisions`/`dedupe_resolved_window_collisions` `RunPython` | medium | mitigate | `logger.warning()` of each deleted row's pk/campaign/telescope_instrument/window before deletion — verified present in both dedup functions | closed |
| T-19-02a | Information Disclosure | `campaign_gap.claimed_dates()` queryset column set | high | mitigate | `.only('pk', 'window_start', 'window_end')` restriction — never `contact_person`/`contact_email` — verified present | closed |
| T-19-02b | Information Disclosure | `campaign_views.ALLOWED_FIELDS_FOR_NON_STAFF` | high | mitigate | Explicit enumerated allowlist (never introspected from `_meta`); includes `window_start`/`window_end`, excludes contact fields — verified present | closed |
| T-19-04 | Denial of Service | `CampaignRunDecisionView.post` D-06 `sun_event()` call site | medium | mitigate | Wrapped in `except ValueError: logger.debug(...)` — never reaches the broad `except Exception` approval-revert path — verified present | closed |
| T-19-05 | Tampering | `CampaignRunSubmissionView` duplicate submission | low | mitigate | Resolved-window `UniqueConstraint` backs the create; `IntegrityError` caught inside its own savepoint, degrades to a friendly form error, never a 500 — verified present | closed |
| T-19-01b | Tampering/Repudiation | `import_campaign_csv` natural-key lookup | high | mitigate | Backed by the resolved-window `UniqueConstraint`; genuine same-date collisions are logged and skipped, never silently merged or crashing the batch — verified present | closed |
| T-19-06 | Information Disclosure | `import_campaign_csv` duplicate-skip log line | low | mitigate | Logs only natural-key fields (telescope_instrument, obs_date) — never Contact Person/Email — verified present | closed |
| T-19-SC | Tampering | Package installs | low | accept | No new packages introduced this phase — confirmed zero diff on `pyproject.toml`/`requirements.txt` across the phase | closed |

*Status: open · closed · open — below {block_on} threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-19-01 | T-19-SC | No new third-party packages were introduced by phase 19 (pure Django schema migration + consumer rewrite) — no supply-chain surface to mitigate | Automated (grep-level verification, ASVS L1) | 2026-07-10 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-10 | 9 | 9 | 0 | `/gsd-secure-phase` orchestrator (direct code verification, ASVS L1 short-circuit — no auditor subagent spawn needed) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-10
