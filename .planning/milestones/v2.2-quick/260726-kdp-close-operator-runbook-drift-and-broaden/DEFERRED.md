# Deferred doc gaps (quick task 260726-kdp)

Found during the Finding 3 audit of `docs/runbooks/telescope_runs_calendar.rst` and
deliberately left out of scope. Audit rule applied: correct the runbook only where existing
prose is now *false*; where the runbook is merely *silent*, defer here instead of silently
dropping the gap.

1. **`fetch_jplsbdb_objects` has no operator documentation anywhere.** Excluded by the task
   brief — it is a JPL target-ingest command, not part of the telescope-runs-calendar
   operator story, so it does not belong in this runbook. Would need its own page.

2. **`7b1e873` (`260722-uyz`)** — `sync_lco_observation_calendar` now sets each
   CalendarEvent's campaign association from the record's Target's campaign. The runbook is
   silent on this rather than wrong about it: omission, not contradiction.

3. **`83d024c` (`260722-hpw`)** — `import_campaign_csv` now scans the first several leading
   rows for the real header before reading data, tolerating a title/blank row above the
   header, and fails fast with a clear error if no header is found. Omission, not
   contradiction.

4. **The LCO sync's other title prefixes** — `[EXPIRED]`, `[FAILED]`, `[CANCELLED]` for
   terminal-failure statuses, and `[UNVERIFIED]` for an unresolved telescope label — are
   undocumented in the runbook body (`[UNVERIFIED]` appears only in Troubleshooting).
   Omission, not contradiction; deliberately not expanded during the `[QUEUED]` fix in Task 1
   of this quick task to keep that correction surgical.
