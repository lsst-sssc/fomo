---
phase: "36"
slug: "unattended-operation"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-09-18"
verified: "2026-09-18"
---

# Phase 36 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

Register origin: authored at plan time — every one of the nine plans (36-01 … 36-09) carries a
`<threat_model>` block; the register below is their union (T-36-01 and T-36-04 recur across plans and
are listed once with the union of their evidence). Verification depth: ASVS L1 (grep-depth mitigation
presence, `workflow.security_asvs_level: 1`), run at HEAD `ee92c90` on 2026-09-18 after UAT rounds 1–4
completed with zero open issues.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| cron/host → Django runner process | environment cron supplies (`FOMO_HEARTBEAT_URL`, `FOMO_BASE_URL`, `FOMO_STATE_DIR`); lock/log/state paths the process writes | heartbeat ping URL (secret), base URL, paths |
| Django runner process → external heartbeat endpoint | outbound HTTPS where the URL's path component is the ping token | the ping URL itself |
| Django runner process → SMTP relay | `send_mail()` with host credentials from `local_settings.py`; a relay error's `str()` can embed them | mail host/user/password |
| Django runner / discovery sweep → LCO/SOAR portal | API-keyed HTTPS via the facility classes and `make_request`; a portal exception's `str()` can embed request/response content | LCO API key, portal responses |
| stock tomtoolkit → FOMO | `update_all_observation_statuses()` returns message strings FOMO did not build | untrusted message text |
| staff admin user → `WatchedProposal` rows → portal query parameter | operator-supplied proposal code becomes an outbound query; `last_run_summary` is stored and rendered in the admin | proposal codes, sweep summaries |
| `check_unattended` output → terminal / redirected log / runbook paste | the command reads every credential-bearing setting; anything it prints can travel | setting NAMES and set/unset status only |
| standard output + standard error → one destination (terminal, `2>&1`) | the single-sink condition G-36-5 exposed | preflight result lines |
| committed docs, notebooks, deploy templates, `settings.py` → public repository | anything written here is published permanently | placeholders only — never values |
| operator's `local_settings.py` → Django settings import | the only channel by which a real credential enters the process | LCO API key, mail password |
| docs build → on-disk HTML trees | `sphinx-build` renders `local_settings.py` into gitignored build trees (CR-03) | that host's credentials, as HTML, gitignored |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-36-01 | Information Disclosure | `run_tick()` per-step except clauses, `notify_staff()`, `step_status_refresh()` failure-list handling | high | mitigate | Class-name-only reporting (D-17): `type(exc).__name__` in `unattended.py`; message half of the tomtoolkit tuple discarded, class re-derived by bounded re-check (D-03). `TestCredentialHygiene` (incl. `test_status_refresh_portal_error_leaks_nothing`) seeds fake API key / mail password / heartbeat URL into forced failures and asserts absence from logs, stdout, stderr, `mail.outbox` | closed |
| T-36-02 | Information Disclosure | `deploy/cron/fomo.crontab.example`, cron argument vector | high | mitigate | Template carries placeholder paths and variable NAMES only (`FOMO_HEARTBEAT_URL`); no UUID-shaped string in the file; credentials arrive via environment / `local_settings.py`, never CLI args (D-15) | closed |
| T-36-03 | Denial of Service (operator visibility) | the schedule itself — never-invoked, hung, or lock-contended tick | medium | mitigate | `ping_heartbeat('start')` before the first step and `ping_heartbeat(str(exit_code))` after the last (`unattended.py:816,862`, D-12); crontab template's `lock held` skip line leaves a log trail on contention | closed |
| T-36-04 | Information Disclosure | failure email body; `check_unattended` report and printed cron line; runbook setup section; notebook committed output | high | mitigate | Body = step names, counters, exception classes, log path, two `FOMO_BASE_URL` links (D-14) — `test_failure_email_body_carries_no_secret_and_no_traceback`. Preflight prints NAME + set/unset or counts only — `TestNoValueLeakage`. Docs/notebook name settings never values; no UUID in runbook or crontab (re-grepped this audit) | closed |
| T-36-05 | Tampering | suppression state file under `FOMO_STATE_DIR` | low | accept | See Accepted Risks AR-36-01 | closed |
| T-36-06 | Information Disclosure | per-row failure path writing `WatchedProposal.last_run_summary` | high | mitigate | Summary carries `type(exc).__name__` only; `TestPerProposalIsolation.test_failure_summary_carries_no_exception_message` seeds a fake key into the raised error and asserts absence from row, stderr, logs | closed |
| T-36-07 | Tampering | `WatchedProposal` create/edit/delete | low | accept | See Accepted Risks AR-36-02 | closed |
| T-36-08 | Input Validation / Spoofing of queried scope | `proposal_code` → portal query parameter | medium | mitigate | `unique=True` plus `.strip()` on save in the `WatchedProposal` model; unusable code isolated as a per-row failure (D-09) | closed |
| T-36-09 | Denial of Service | one failing proposal starving the others | medium | mitigate | Per-row try/except/continue; `TestPerProposalIsolation` | closed |
| T-36-10 | Information Disclosure | `step_project_sweep()` observed-site hook, `step_discovery()` per-row except | high | mitigate | Class-name-only rule (D-17); dedicated `TestCredentialHygiene` cases force both paths | closed |
| T-36-11 | Information Disclosure | failure email body assembled in `run_tick()` | medium | mitigate | D-14 fixed body shape; `test_failure_email_body_carries_no_secret_and_no_traceback` | closed |
| T-36-12 | Tampering | step ordering drifting from D-01 | medium | mitigate | `STEPS` is the single ordered declaration and the source of `--step` choices; `test_all_four_steps_run_in_order` | closed |
| T-36-13 | Denial of Service | one failing step starving the other three | medium | mitigate | Per-step try/except + per-step `command_lock()` skip-and-log (D-02); `test_step_failure_does_not_abort_the_tick`, `test_facility_exception_is_isolated_per_facility` | closed |
| T-36-14 | Information Disclosure | `check_unattended --send-test-email` failure path | medium | mitigate | Raised `send_mail()` reported by class name only in `check_unattended.py` | closed |
| T-36-15 | Tampering | printed cron line drifting from the committed template | medium | mitigate | `TestCronLine.test_line_matches_the_committed_template_shape`; interval owned by one constant in `unattended.py` (WR-36, IN-40) | closed |
| T-36-16 | Elevation of Privilege | preflight creating directories / installing cron on the operator's behalf | low | mitigate | Read-only by construction; `test_command_writes_nothing` | closed |
| T-36-17 | Information Disclosure | unpatched notebook cell making a live authenticated portal call | high | mitigate | Every cell patches `backfill_lco_observations.make_request` (13 occurrences in the committed notebook) | closed |
| T-36-18 | Tampering | notebook execution leaving demo rows behind | low | mitigate | Cleanup cell deletes the `WatchedProposal` rows it created | closed |
| T-36-19 | Repudiation | runbook going stale after a future `unattended.py` change | medium | mitigate | `CLAUDE.md` paired-docs map names `unattended.py`, `notifications.py`, `run_unattended.py`, `check_unattended.py` → runbook section | closed |
| T-36-20 | Information Disclosure | corrected heartbeat paragraph, crontab comment, `check_heartbeat()` `[ok]` detail | high | mitigate | Names knobs never values; `test_line_carries_no_setting_value`, `test_output_never_contains_a_seeded_value` | closed |
| T-36-21 | Repudiation | dead-man layer that reports green but can never fire | high | mitigate | Runbook states Period/Grace values to set and the alert arithmetic; troubleshooting entry for the silent never-alerted symptom; confirmed live in UAT round 3 Test 4 (late ~15 min, alert ~35 min) | closed |
| T-36-22 | Denial of Service | grace shrunk below a slow tick's runtime | medium | mitigate | Runbook explains grace also bounds the `/start`-to-completion gap; ~20-minute recommendation retained | closed |
| T-36-23 | Tampering | refactor silently dropping the preflight's interval reminder | medium | mitigate | `test_set_heartbeat_reminds_about_the_check_period` (comment names G-36-3) | closed |
| T-36-24 | Information Disclosure | ping-URL provenance sentence; crontab variable listing | high | mitigate | Placeholder ``https://hc-ping.com/<uuid>`` only, with the "URL is itself the credential" warning; no UUID-shaped string in runbook or crontab (re-grepped this audit) | closed |
| T-36-25 | Repudiation | dead-man layer never created because the procedure never asks | high | mitigate | Create-and-configure step precedes the export step; slice gate asserts position (Period < ping-URL < `4. Export`) — re-run green this audit and recorded as a VALIDATION.md row | closed |
| T-36-26 | Tampering | guidance drifting across runbook / crontab / preflight reminder | medium | mitigate | One canonical home + cross-reference; crontab points at the subsection (3 references); preflight reminder pinned by test | closed |
| T-36-27 | Repudiation | sufficiency verdict from a reader already taught the values out of band | medium | mitigate | UAT round 3 administered the SC-5 read-through (Tests 2–3) before the live-heartbeat test (Test 4); 36-VERIFICATION.md records the order and the contamination rule | closed |
| T-36-28 | Information Disclosure | runbook step 2 API-key instruction; fold-test placeholder literal | high | mitigate | Angle-bracket placeholder on the right-hand side only; `_FAKE_` convention in `test_settings_api_key_fold.py`; no UUID in either | closed |
| T-36-29 | Spoofing | SOAR facility calling the portal with an empty `api_key` while believed configured | high | mitigate | Fold fills both `FACILITIES['LCO']`/`['SOAR']`; `TestFlatKeyReachesBothFacilities`, `TestSoarAccessorReadsFoldTarget` | closed |
| T-36-30 | Denial of Service | documented configuration step whose literal execution stops Django | high | mitigate | Step 2 names the only survivable assignment form and the `NameError` symptom; slice gate rejects any `FACILITIES[` subscript; fatal form pinned by an executable `NameError` case in the fold test | closed |
| T-36-31 | Information Disclosure | `sphinx-build` hook rendering `local_settings.py` into gitignored build trees (CR-03) | medium | accept | See Accepted Risks AR-36-03 | closed |
| T-36-32 | Tampering | step-2 instruction drifting from the fold code | medium | mitigate | The test executes the real fold tail out of the live `settings.py` (`exec` of the sliced source) rather than re-implementing it | closed |
| T-36-33 | Information Disclosure | preflight result lines whose routing 36-09 changed | medium | mitigate | Content unchanged (same f-string, same `CheckResult` fields); `TestNoValueLeakage` untouched and green (reads both captures) | closed |
| T-36-34 | Denial of Service | stdout-only redirect capturing a clean-looking log while warnings went elsewhere | medium | mitigate | Runbook step 6 states the routing, what a bare `>` drops, and shows `2>&1`; crontab template already merges streams; hard-failure exit code unaffected | closed |
| T-36-35 | Tampering | escape bytes entering a redirected log via a style argument on the stderr write | low | mitigate | No `style_func` in `check_unattended.py`; `test_merged_capture_has_no_escape_bytes` | closed |
| T-36-36 | Information Disclosure | plan 36-09 running the `sphinx-build` hook on a configured checkout (CR-03) | medium | accept | See Accepted Risks AR-36-04 | closed |
| T-36-37 | Tampering | operator's uncommitted runbook edit destroyed by an executor tidying its tree | medium | mitigate | Committed first as its own commit `db8a1cc` with the operator's phrase present; confirmed by UAT round 4 Test 2 (P1) | closed |
| T-36-SC | Tampering (supply chain) | npm/pip/cargo installs | high | accept | See Accepted Risks AR-36-05 | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-36-01 | T-36-05 | A local actor who can write the suppression state file can suppress one failure email, but the independent heartbeat layer (D-12) still fires — the reason SCHED-09 requires two layers. The file lives in an operator-owned directory (`FOMO_STATE_DIR`, validated by CR-05/WR-33/WR-34) with no wider exposure | plan 36-01 threat model; confirmed by secure-phase audit | 2026-09-18 |
| AR-36-02 | T-36-07 | `WatchedProposal` CRUD is gated by Django admin's existing `is_staff` permission model (ASVS V4, standard control reused unchanged); no anonymous path reaches the model | plan 36-02 threat model; confirmed by secure-phase audit | 2026-09-18 |
| AR-36-03 | T-36-31 | CR-03 (`sphinx-build` renders `local_settings.py` into gitignored build trees) is an open ship decision whose root-cause file this phase never touched; plan 36-08 used a read-only `docutils` parse instead of the build, added no credential, and the runbook records the standing rule: never serve those trees from a configured checkout. `docs/conf.py` now excludes `local_settings.py` from the autoapi/viewcode scan (CR-03 fix `e2ed553`) | plan 36-08 threat model; confirmed by secure-phase audit | 2026-09-18 |
| AR-36-04 | T-36-36 | Plan 36-09 ran the `sphinx-build` hook because the housekeeping commit required the docs build; both build trees are gitignored and the doc tasks asserted nothing under them is stageable. Residual unchanged from before the plan (a configured host that also serves its own build tree). CR-03 remains an open ship decision, not resolved here | plan 36-09 threat model; confirmed by secure-phase audit | 2026-09-18 |
| AR-36-05 | T-36-SC | No package is installed by any plan in this phase — 36-RESEARCH.md §"Package Legitimacy Audit" records zero new dependencies (`requests`, Django/tomtoolkit pre-existing; `fcntl` stdlib; `docutils` an already-installed Sphinx dependency used read-only), so there is no install step to gate | plans 36-01…36-09 threat models; confirmed by secure-phase audit | 2026-09-18 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-18 | 38 | 38 | 0 | /gsd-secure-phase 36 (orchestrator, ASVS L1 grep-depth; register authored at plan time, so no auditor subagent was required) |

Evidence run at HEAD `ee92c90`: every named test class/method in the Mitigation column exists in
`solsys_code/tests/test_unattended.py`, `test_check_unattended.py`, `test_backfill_lco_observations.py`
or `test_settings_api_key_fold.py`, and those modules ran green in the same session (122 + 9 tests);
UUID-pattern grep over the runbook, the crontab template and the fold test returns nothing; `style_func`
is absent from `check_unattended.py`; the runbook fresh-host slice gate and flat-setting gate print
`True`; `ping_heartbeat('start')` / `ping_heartbeat(str(exit_code))` bracket the tick at
`solsys_code/unattended.py:816,862`.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-18
