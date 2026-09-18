---
status: complete
phase: 36-unattended-operation
source: [36-VERIFICATION.md]
started: 2026-09-17T17:25:00Z
updated: 2026-09-18T00:48:39Z
---

## Current Test

[testing complete]

## Tests

### 1. Fresh-host preflight
expected: On the real FOMO host, after creating `/var/lock/fomo` and `/var/log/fomo` writable by the cron account, putting the real `EMAIL_BACKEND`/`EMAIL_HOST_*` and the LCO/SOAR `api_key` in `local_settings.py`, and exporting `FOMO_HEARTBEAT_URL` and `FOMO_BASE_URL` for the cron daemon, `python manage.py check_unattended` reports `[ok]` on every hard check, exits 0, and prints a cron line with this host's real interpreter and `manage.py` paths. (On the dev checkout it correctly exits 1 naming `FOMO_LOCK_DIR`/`FOMO_LOG_FILE`/`EMAIL_BACKEND`.)
result: pass
note: "Only a [WARN] for no active WatchedProposal; every other check [ok], exit 0, cron line printed."

### 2. The real crontab
expected: After installing the printed cron line in the FOMO service account's crontab (`crontab -e`) and waiting ~45 minutes, three `START` / per-step / `END` banners appear in `/var/log/fomo/unattended.log`, roughly 15 minutes apart, with nobody typing anything.
result: pass
note: "Three START+END sequences observed in /var/log/fomo/unattended.log."

### 3. The heartbeat's dead-man half
expected: With a healthchecks-compatible check pointed at `FOMO_HEARTBEAT_URL` (grace period ~20 minutes), disabling the crontab line and waiting past the grace period makes the heartbeat service raise an alert even though FOMO itself logged nothing and sent no email — the second, independent layer of SC 3.
result: issue
reported: "I have the healthchecks.io set at Grace 20 mins and last ping was 28 minutes ago but it is still green - not sure if I need to wait more or not... ? / set period to 15 minutes, now have orange pling / went red and got the alert email, nothing from FOMO"
severity: major
note: "Mechanism works: with Period=15 min, Grace=20 min the check went Late then Down and the alert email arrived with FOMO logging nothing and sending nothing. The issue is the runbook: docs/runbooks/telescope_runs_calendar.rst:1544-1547 says only 'grace period ~20 minutes' and never mentions the check's Period (default 1 day), so an operator following it literally gets a check that stays green for ~24 h after the schedule stops — the dead-man layer is silently disabled." 

### 4. Real mail delivery
expected: With the real (non-console) email backend configured, `python manage.py check_unattended --send-test-email` delivers one message to the mailbox of every staff user with an email on file — the same recipient rule the failure notice uses.
result: pass

### 5. Log rotation under a live writer
expected: With `deploy/logrotate/fomo.example` installed as `/etc/logrotate.d/fomo`, `logrotate -d /etc/logrotate.d/fomo` parses the stanza, and forcing one real rotation while a tick is running shows `copytruncate` keeping cron's still-open append redirect writing to the live file rather than the rotated-away inode.
result: pass
note: "Forced rotation (possibly twice) mid-tick: START banner went with the rotated copy; the per-step lines and `=== FOMO unattended run END 2026-09-18T00:30:54 exit=0 ===` landed in the truncated live file." 

### 6. Runbook sufficiency
expected: Someone who has not read this phase's source, given only the runbook's "How do I run everything unattended?" section, reaches a working, checked schedule on a fresh host — no source reading, no questions back (SC 5).
result: pass
note: "User verdict: pass. Questions that came up during this UAT and may merit runbook clarification: where FOMO_HEARTBEAT_URL / FOMO_BASE_URL are read (env var vs local_settings.py, cron vs web process) and that installing /etc/logrotate.d/fomo and forcing a rotation need sudo. The heartbeat Period omission is tracked as G-36-3." 

## Summary

total: 6
passed: 5
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-36-3
  truth: "An operator following the runbook's heartbeat guidance gets a healthchecks-compatible check that alerts within ~20 minutes of the cron schedule stopping."
  status: failed
  reason: "User reported: healthchecks.io check set to Grace 20 min per the runbook stayed green 28 minutes after the last ping; it only went Late/Down after the user discovered and set Period to 15 minutes themselves. Runbook (docs/runbooks/telescope_runs_calendar.rst:1544-1547) mentions only the grace period, not the check's Period/schedule, whose default is 1 day."
  severity: major
  test: 3
  artifacts: []  # Filled by diagnosis
  missing: []    # Filled by diagnosis

## Operational Findings

- finding_id: F-36-1
  title: "One LCO record's observed-site lookup fails on every tick"
  detail: "Tick ending 2026-09-18T00:30:54Z on the real host logged `observation_id='4276100': observed-site lookup unavailable -- using fallback label.` and `site_lookup_failed: 1` in the project_sweep summary (159 unchanged, 0 unprojectable). Tick still exit=0. Seen while verifying Test 5; not a Phase 36 defect."
  action: "Check record 4276100 in the LCO portal / resolve_placement_block() path when convenient."
