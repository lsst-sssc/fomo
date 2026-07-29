# Phase 3: Classical Calendar Ingest - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-13
**Phase:** 3-Classical Calendar Ingest
**Areas discussed:** CLI input & per-line errors, Idempotency / upsert key, Title & description format, Status-dependent behavior

---

## CLI input

| Option | Description | Selected |
|--------|-------------|----------|
| Positional file path argument | `./manage.py load_telescope_runs schedule.txt` — mirrors typical Django management command conventions | ✓ |
| `--file` option | `./manage.py load_telescope_runs --file schedule.txt` — matches fetch_jplsbdb_objects's flag-based style | |
| stdin | `cat schedule.txt \| ./manage.py load_telescope_runs` — flexible but less discoverable | |

**User's choice:** Positional file path argument (Recommended)
**Notes:** D-01.

---

## Line errors

| Option | Description | Selected |
|--------|-------------|----------|
| Skip and log, continue | Write error to stderr with line number/text, continue; summary reports created/skipped/errored counts | ✓ |
| Abort entire run | Any unparseable line stops the whole command | |

**User's choice:** Skip and log, continue (Recommended)
**Notes:** D-02. Covers Phase 2's ambiguous-`'Magellan'` ValueError case for the two sample lines that don't resolve.

---

## Upsert key

| Option | Description | Selected |
|--------|-------------|----------|
| telescope + instrument + start_time | get_or_create on these three; update title/description/end_time on match | ✓ |
| telescope + start_time only | Simpler but instrument changes update in place rather than signal a change | |
| telescope + instrument + start_time + end_time | Most specific; any time change creates a new event | |

**User's choice:** telescope + instrument + start_time (Recommended)
**Notes:** D-03.

---

## Unchanged re-run

| Option | Description | Selected |
|--------|-------------|----------|
| Update fields anyway (idempotent overwrite) | Always overwrite from freshly parsed line | |
| Leave untouched if unchanged | Compare computed fields to existing; only write if different | ✓ |

**User's choice:** Leave untouched if unchanged (Recommended)
**Notes:** D-04.

---

## Title format

| Option | Description | Selected |
|--------|-------------|----------|
| "{telescope} {instrument}" | e.g. 'NTT EFOSC2' — clean, queryable; status not in title | ✓ |
| "{telescope} {instrument} ({status})" | e.g. 'NTT EFOSC2 (allocation)' — surfaces status at a glance | |

**User's choice:** "{telescope} {instrument}" (Recommended)
**Notes:** D-05.

---

## Description format

| Option | Description | Selected |
|--------|-------------|----------|
| Dark window + original line + status | Covers INGEST-02 directly | ✓ |
| Same, plus night index | Adds "Night 2 of 5 (date)" | |

**User's choice:** Dark window + original line + status (Recommended)
**Notes:** D-06. Exact formatting left to Claude's discretion.

---

## Status behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Informational only | All statuses create/update events the same way; status is text in description | ✓ |
| 'cancelled' deletes matching events | Adds a delete path not covered by current requirements | |

**User's choice:** Informational only (Recommended)
**Notes:** D-07. 'cancelled'-status deletion noted as a deferred idea.

---

## Claude's Discretion

- Per-night iteration approach across month/year boundaries within a single run
- Exact description field wording/layout beyond the three required pieces (dark window, status, source line)
- Exact stdout/stderr summary message wording (counts required)
- get_or_create+conditional-save vs update_or_create+pre-comparison implementation choice

## Deferred Ideas

- 'cancelled'-status lines deleting/striking-through previously-created events
- Resolving Magellan-Clay vs Magellan-Baade ambiguity (carried forward from Phase 2)
