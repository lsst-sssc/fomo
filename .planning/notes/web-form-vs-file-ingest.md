---
title: Web form vs file ingest — division of purpose
date: 2026-06-16
context: Explored during Phase 03 UAT session (issue #37 telescope runs calendar)
---

# Note: Web form vs file ingest — division of purpose

## The two workflows are intentionally separate

### File-based ingest (`load_telescope_runs`)

**Use case:** Pre-planned, PI-coordinated observing programs where the schedule is known
in advance and loaded in bulk.

**Example:** "The Didymos follow-up campaign next month — we're working directly with the
PI and loading their full schedule from a text file."

**Characteristics:**
- Operator-run (CLI / cron), not self-service
- Bulk: one file covers many runs across a season
- No approval queue needed — the PI relationship is already established
- Target may or may not be specified per line

### Web form (future, see seeds/target-linked-run-submission-form.md)

**Use case:** Ad-hoc coordination around urgent, rare objects where multiple PIs and
community members need to register time quickly without CLI access.

**Example:** "4I/Borisov-class interstellar object detected — DDT programs activated,
IAWN campaign launched, several groups want their runs on the shared calendar."

**Characteristics:**
- Self-service web UI, accessible to PIs and community members
- Per-run submissions, not bulk
- Requires admin approval before appearing on the calendar (spam/error control)
- **Target is mandatory** — runs are coordinated around a specific FOMO Target, not
  generalized Rubin follow-up
- Optional proposal code bridges calendar-only visibility and FOMO observation submission

## Why not merge them?

The workflows differ in trust model, urgency, and audience. Merging them would either
make the file-based path unnecessarily heavyweight (approval queues for known PIs) or
make the web form too permissive (no oversight for community submissions).
