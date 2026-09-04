---
phase: "31"
slug: "foundation-spikes-run-identity-unattended-invocation"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-09-02"
verified: "2026-09-02"
---

# Phase 31 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

This phase's threat model was authored at plan time — every one of its five `PLAN.md` files
carries a `<threat_model>` block (trust boundaries + STRIDE register), so
`register_authored_at_plan_time: true`. `threats_open: 0` at plan time (no `<threat_model>` entry
carries a status other than `mitigate`/`accept` with a stated plan) plus `asvs_level: 1` means this
verification runs at L1 grep-depth: each mitigation's claim was re-checked directly against the
committed artifacts and command output rather than by spawning `gsd-security-auditor` for deeper
L2/L3 boundary-placement or end-to-end trace analysis.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Real dev database → disposable probe copy | A mis-pointed connection in `tmp/31_constraint_probe.py` could write into the operator's live `src/fomo_db.sqlite3` instead of the disposable copy | Full `CampaignRun` table contents (read); schema-editor writes (disposable copy only) |
| Machine-local settings override / process environment → committed decision document | `src/fomo/local_settings.py` is git-excluded and holds a live credential on the real host; this phase's evidence discipline is "quote output verbatim," which is exactly the path by which such a value could reach a committed file | Credential values (must never cross) |
| Real `CampaignRun` rows (`contact_person`, `contact_email`) → committed decision document | The model carries submitter contact fields that are PII, gated in the app's own read paths but not automatically gated in an ad hoc investigation probe | Submitter PII (must never cross as values; field names only) |
| Plan 31-01 probe transcripts → committed `31-DECISION.md` recommendation (31-02) | The recommendation restates measurements into a durable verdict; an unlabelled inference presented as a measurement is the disclosure/repudiation risk | Structural facts and counts only |
| The `31-DECISION.md` recommendation → Phase 32 adapter code | Whatever this section says becomes the schema all three future adapters write through | Design decisions (intended to cross) |
| Operator-supplied classical schedule file → committed decision document | The file may carry PI names, proposal titles and contact details; only structural facts (status word, proposal-code-shaped-token presence/format) may be quoted | PII / proposal metadata (must never cross as values) |
| Operator-supplied file → `parse_run_line` | A malformed or hostile line reaches a parser that raises `ValueError` on an unknown status; must be handled as data, not a crash | Untrusted structured text |
| Crontab lines → committed transcript (31-04) | A cron line can carry an inline environment assignment, so the raw crontab is untrusted content from a credential standpoint | Potential credential values (must never cross unredacted) |
| The FOMO process → third-party heartbeat service (`hc-ping.com`) | An outbound request from an institutional host to an external service | Bare ping, no payload |
| `31-DECISION.md` (internal evidence) → published Sphinx documentation (31-05) | The design page is built for publication; anything summarised onto it leaves the repository's internal planning context | Structural verdicts only, no raw evidence |
| Git-excluded probe transcripts and the disposable database copy → the git index | A full copy of the operator's real dev data sat under `tmp/` until plan 31-05 removed it | Full `CampaignRun` table contents (must never be committed) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-31-01 | Information Disclosure | `31-DECISION.md` verbatim evidence quoting (31-01) | high | mitigate | Probe scripts printed counts/field names only, never `contact_person`/`contact_email` values; neither probe read `local_settings.py`/`FACILITIES`/env vars. Re-verified: independent PII/credential grep over the final `31-DECISION.md` is clean. | closed |
| T-31-02 | Tampering | Write-probe target selection in `tmp/31_constraint_probe.py` | high | mitigate | Hard guard raised `SystemExit` unless the resolved DB basename was `31-spike-db-copy.sqlite3`; transcript confirms `GUARD_DISPOSABLE_COPY=OK`. Live-DB fingerprint re-compared and unchanged. | closed |
| T-31-03 | Tampering | Accidental source edit during investigation | medium | mitigate | Every task's acceptance criteria required `git status --porcelain -- solsys_code src` empty; independently re-checked at audit time — empty. No migration file was written (schema changes applied in-process via `schema_editor()` on the disposable copy only). | closed |
| T-31-SC (31-01) | Tampering | Package-manager installs | low | accept | This plan installed no package; `flock` is a pre-existing OS binary. | closed |
| T-31-04 | Tampering | Data integrity: chosen schema shape presented as measured rather than argued | high | mitigate | Task 3's second verify command required ≥3 candidate-shape references and both rejected options cited with specific measured numbers from 31-01's evidence; confirmed present in the committed SCHEMA-01 section. | closed |
| T-31-05 | Repudiation | An unlabelled inference recorded as a measurement | medium | mitigate | Every claim in both recommendation subsections carries the fixed confidence-tag vocabulary (`Confirmed against real rows` / `Constructed-input code-path check`); confirmed present. | closed |
| T-31-06 | Information Disclosure | Contact fields quoted while arguing the TBD-branch constraint | medium | mitigate | The recommendation names the `contact_person` field, never a value; the underlying 31-01 probe already prints only counts for that grouping. | closed |
| T-31-SC (31-02) | Tampering | Package-manager installs | low | accept | This plan installed no package; it wrote prose into an existing document. | closed |
| T-31-07 | Information Disclosure | Operator-supplied schedule file quoted into `31-DECISION.md` | high | mitigate | Raw files stayed in git-excluded `tmp/31-classical-samples/`; only structural facts (status word, proposal-code-shaped-token format) were quoted. Independently re-verified: PII/credential grep over the final document is clean, including a manual check for the specific PI name and proposal title present in the operator's supplied file (both absent). | closed |
| T-31-08 | Denial of Service | `parse_run_line` raising on an unrecognised status in a supplied file | low | accept | The parser's `ValueError` on an unknown status is documented behaviour and itself a recorded finding (1 of 3 real lines rejected); the inspection counted rejections rather than aborting. | closed |
| T-31-09 | Repudiation | A sufficiency verdict recorded with a confidence tag stronger than its evidence | medium | mitigate | Task 3 pinned the tag to task 2's measured sample count (1 real file, 3 lines) — confirmed the verdict carries `Constructed-input code-path check` per the actual reported findings, not upgraded on the strength of the small sample. | closed |
| T-31-SC (31-03) | Tampering | Package-manager installs | low | accept | This plan installed no package. | closed |
| T-31-10 | Information Disclosure | Crontab and settings output quoted into `31-DECISION.md` | high | mitigate | Crontab redacted at capture time (before anything written to transcript) per task 1's action; no probe read the machine-local override or the facilities settings dict. Independently re-verified: credential/hex/email grep over the final document is clean. | closed |
| T-31-11 | Information Disclosure | A credential passed as a management-command argument, visible in a process listing | high | mitigate | Recorded as an explicit prohibition in the SCHED-07 recommendation: credentials travel only as environment variables, never as a positional/keyword argument. Confirmed present in the committed section. | closed |
| T-31-12 | Information Disclosure | An exception's string form forwarded into a staff-notification email | medium | mitigate | Recorded as a named Phase 34 obligation against SCHED-10 (not fixed here — this phase writes no code); confirmed the obligation is named explicitly with the requirement ID attached in the committed section. | closed |
| T-31-13 | Denial of Service (data integrity) | Overlapping unattended invocations contending for the single-writer database | high | mitigate | Recommendation specifies a per-command file lock with a fail-fast flag. Task 1's real probe counted 0/3 existing cron entries with any overlap guard, establishing this as a live condition, not a hypothetical — confirmed in the committed transcript and recommendation. | closed |
| T-31-14 | Information Disclosure | Outbound heartbeat ping to a third-party service from an institutional host | low | accept | The ping is a bare request carrying no payload. Technical reachability confirmed by probe (HTTP 301); policy acceptability explicitly recorded as unresolved and routed to the operator, per RESEARCH.md Assumption A2 — confirmed neither scope was silently upgraded to settled. | closed |
| T-31-15 | Tampering | A configurable heartbeat target reachable from untrusted input in a later phase | low | mitigate | Recorded as a constraint for Phase 34: the ping target is a fixed value or environment variable, never derived from user-controlled input. Confirmed recorded. | closed |
| T-31-SC (31-04) | Tampering | Package-manager installs | low | accept | This plan installed no pip/npm package; the lock utility is a pre-existing OS binary and the heartbeat check needs no client library. The Task 2 container probe pulled a standard public `python:3.11-slim` base image (removed immediately after use) — a container-runtime image pull for a disposable, deleted probe, not a project dependency install, and outside this threat's pip/npm-supply-chain scope per RESEARCH.md's Package Legitimacy Audit. | closed |
| T-31-16 | Information Disclosure | Published design page and committed decision document | high | mitigate | Task 2's final gate searched both artifacts for credential-shaped tokens, long hex strings and email addresses, with existence guards. Independently re-verified at audit time over both `31-DECISION.md` and `docs/design/run_identity_and_unattended_invocation_spike.rst`: clean, including a manual check for the specific PI name/proposal title from 31-03's operator-supplied file (absent from both). | closed |
| T-31-17 | Information Disclosure | `tmp/31-spike-db-copy.sqlite3`, a full copy of the real dev database | medium | mitigate | Created inside git-excluded `tmp/`; task 2's verify command required it deleted. Independently re-confirmed at audit time: file does not exist, `git ls-files tmp/` is empty. | closed |
| T-31-18 | Tampering | An accidental source edit surviving to the phase seal | medium | mitigate | Task 2 asserted an empty `git status` for `solsys_code/`/`src/` and separately for migrations, plus a targeted regression run (143 tests, `OK`). Independently re-confirmed at audit time: `git status --porcelain -- solsys_code src` is empty. | closed |
| T-31-19 | Denial of Service | Running the test suite unqualified | low | mitigate | The regression run named exactly six explicit module labels (confirmed in the executor's report), avoiding both `test_views.TestEphemeris` (native ASSIST segfault) and the ephemeris-utilities kernel download. | closed |
| T-31-SC (31-05) | Tampering | Package-manager installs | low | accept | This plan installed no package; it ran the documentation build and test runner already present in the environment. | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above `workflow.security_block_on` (`high`) count toward `threats_open`*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-31-01 | T-31-SC (all 5 plans) | No plan in this phase installs a pip/npm package; `flock` and container base images used are pre-existing/standard, not new supply-chain dependencies | GSD security gate (ASVS L1, `security_block_on: high` — low severity, non-blocking) | 2026-09-02 |
| AR-31-02 | T-31-08 | `parse_run_line`'s `ValueError` on an unrecognised status is documented, pre-existing parser behaviour; the inspection treats a rejection as a counted finding, not a crash to fix | GSD security gate (low severity, non-blocking) | 2026-09-02 |
| AR-31-03 | T-31-14 | Outbound heartbeat ping is a bare, payload-free request; policy acceptability (as opposed to technical reachability) is explicitly out of scope for this phase and routed to the operator/a future phase | GSD security gate (low severity, non-blocking) | 2026-09-02 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-02 | 23 | 23 | 0 | Orchestrator (L1 grep-depth verification; `register_authored_at_plan_time: true`, `threats_open: 0` at plan time, `asvs_level: 1` — short-circuit rule applied, no `gsd-security-auditor` subagent dispatch needed) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-02
