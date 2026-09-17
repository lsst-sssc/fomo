---
status: testing
phase: 36-unattended-operation
source: [36-VERIFICATION.md]
started: 2026-09-17T17:25:00Z
updated: 2026-09-17T17:25:00Z
---

## Current Test

number: 1
name: Fresh-host preflight
expected: |
  On the real FOMO host — with /var/lock/fomo and /var/log/fomo writable by the cron account, the real
  EMAIL_BACKEND/EMAIL_HOST_* and the LCO/SOAR api_key in local_settings.py, and FOMO_HEARTBEAT_URL /
  FOMO_BASE_URL exported for the cron daemon — `python manage.py check_unattended` reports [ok] on every
  hard check, exits 0, and prints a cron line carrying this host's real interpreter and manage.py paths.
awaiting: user response

## Tests

### 1. Fresh-host preflight
expected: On the real FOMO host, after creating `/var/lock/fomo` and `/var/log/fomo` writable by the cron account, putting the real `EMAIL_BACKEND`/`EMAIL_HOST_*` and the LCO/SOAR `api_key` in `local_settings.py`, and exporting `FOMO_HEARTBEAT_URL` and `FOMO_BASE_URL` for the cron daemon, `python manage.py check_unattended` reports `[ok]` on every hard check, exits 0, and prints a cron line with this host's real interpreter and `manage.py` paths. (On the dev checkout it correctly exits 1 naming `FOMO_LOCK_DIR`/`FOMO_LOG_FILE`/`EMAIL_BACKEND`.)
result: [pending]

### 2. The real crontab
expected: After installing the printed cron line in the FOMO service account's crontab (`crontab -e`) and waiting ~45 minutes, three `START` / per-step / `END` banners appear in `/var/log/fomo/unattended.log`, roughly 15 minutes apart, with nobody typing anything.
result: [pending]

### 3. The heartbeat's dead-man half
expected: With a healthchecks-compatible check pointed at `FOMO_HEARTBEAT_URL` (grace period ~20 minutes), disabling the crontab line and waiting past the grace period makes the heartbeat service raise an alert even though FOMO itself logged nothing and sent no email — the second, independent layer of SC 3.
result: [pending]

### 4. Real mail delivery
expected: With the real (non-console) email backend configured, `python manage.py check_unattended --send-test-email` delivers one message to the mailbox of every staff user with an email on file — the same recipient rule the failure notice uses.
result: [pending]

### 5. Log rotation under a live writer
expected: With `deploy/logrotate/fomo.example` installed as `/etc/logrotate.d/fomo`, `logrotate -d /etc/logrotate.d/fomo` parses the stanza, and forcing one real rotation while a tick is running shows `copytruncate` keeping cron's still-open append redirect writing to the live file rather than the rotated-away inode.
result: [pending]

### 6. Runbook sufficiency
expected: Someone who has not read this phase's source, given only the runbook's "How do I run everything unattended?" section, reaches a working, checked schedule on a fresh host — no source reading, no questions back (SC 5).
result: [pending]

## Summary

total: 6
passed: 0
issues: 0
pending: 6
skipped: 0
blocked: 0

## Gaps
