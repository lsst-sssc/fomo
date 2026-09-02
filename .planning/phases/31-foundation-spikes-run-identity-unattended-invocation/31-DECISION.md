# Phase 31: Foundation Spikes — Run Identity & Unattended Invocation - Decision

**Investigated:** 2026-09-01
**Status:** In progress. This document is built up across all five plans of Phase 31.
Plan 31-01 records the SCHEMA-01 dev-DB snapshot (block B/C/D/E of
`tmp/31_dbsnapshot_probe.py`), the SCHEMA-01 read-path blast-radius inventory, and the
SCHEMA-02 candidate-shape constraint probe. Plans 31-02 through 31-04 append the
SCHEMA-01/02 recommendation checkpoint, the SCHEMA-03 classical-schedule-file findings,
and the SCHED-07 scheduling-mechanism track respectively. Plan 31-05 completes
`## Recommendation` and `## Durable summary` and publishes
`docs/design/run_identity_and_unattended_invocation_spike.rst`.

This phase is **investigation-only**, following the Phase 18 (uncertain-scheduling) and
Phase 26 (canonical-record) precedents. No `CampaignRun` schema migration is applied, no
adapter code is written, and the only committed artifacts of this phase are this file and
one `docs/design/` page. **Evidence posture, stated explicitly per track:** the
schema/identity track follows Phase 26's disposable-file-copy posture — write-probes run
for real against `tmp/31-spike-db-copy.sqlite3`, with no rollback anywhere in the
procedure, because the whole file is throwaway — while read-only probes touch the real
`src/fomo_db.sqlite3` under the fingerprint-before/after discipline only, and never
select `contact_person`/`contact_email` values. The scheduling track (SCHED-07, plan
31-04) verifies real host/container facts directly rather than reasoning from
documentation.

## Findings

### Schema/identity track (SCHEMA-01/02/03)

#### SCHEMA-01 evidence - real dev-DB snapshot

Executed via `tmp/31_dbsnapshot_probe.py` (`python manage.py shell < tmp/31_dbsnapshot_probe.py`,
captured verbatim to `tmp/31-dbsnapshot.txt`) against the real, unmodified
`src/fomo_db.sqlite3`. The script never calls a write-style ORM method and never selects
`contact_person`/`contact_email` values.

```
FINGERPRINT_BEFORE=1208320 1788271090
TOTAL_CAMPAIGNRUN_ROWS=49
NULL_CAMPAIGN_ROWS=0
DISTINCT_CAMPAIGNS=5
RESOLVED_WINDOW_ROWS=43
TBD_WINDOW_ROWS=6
SOURCE_classical_file=1
SOURCE_csv_import=11
SOURCE_eso_queue=7
SOURCE_lco_queue=5
SOURCE_legacy=24
SOURCE_web=1
DUP_TELINST_WINDOW_TUPLES_IGNORING_CAMPAIGN=4
DUP_TELINST_CONTACT_TUPLES_IGNORING_CAMPAIGN=0
FINGERPRINT_AFTER=1208320 1788271090
FINGERPRINT_UNCHANGED=PASS
=== Block (E): CampaignRun._meta.constraints ===
  UniqueConstraint name='unique_campaign_run_resolved_window' fields=('campaign', 'telescope_instrument', 'window_start', 'window_end') condition=<Q: (AND: ('window_start__isnull', False))>
  UniqueConstraint name='unique_campaign_run_tbd_natural_key' fields=('campaign', 'telescope_instrument', 'contact_person') condition=<Q: (AND: ('window_start__isnull', True))>
  CheckConstraint name='campaign_run_window_start_end_null_together' fields=None condition=<Q: (OR: (AND: ('window_end__isnull', True), ('window_start__isnull', True)), (AND: ('window_end__isnull', False), ('window_start__isnull', False)))>
```

The dated fingerprint of `src/fomo_db.sqlite3` (`1208320 1788271090`) is identical before
and after this probe ran — the real dev database was not modified.

Tag: **Confirmed against real rows** (the probe read the real, current 49 `CampaignRun`
rows in `src/fomo_db.sqlite3`, not a copy or a synthetic fixture).

**Reading for D-06's low-migration-risk premise:** the null-campaign count is **0 of 49**
rows — every existing `CampaignRun` row today has a real `campaign` FK value, confirming
D-06's premise directly rather than assuming it (this matches the 0/49 figure RESEARCH.md
already found during the research pass; this probe re-confirms it fresh, one dated snapshot
later, via the app's own ORM rather than a raw `sqlite3` CLI query). Nullability/sentinel
design genuinely only has to serve *new* adapter-written rows — there is no backfill
population to worry about.

Block (C) shows **4 pre-existing `(telescope_instrument, window_start, window_end)` tuples**
already occur more than once across the real resolved-window population, ignoring `campaign`
entirely. This is real-data confirmation of RESEARCH.md Pitfall 2's collision risk: if every
non-campaign run were collapsed onto one shared `campaign` value (Option B), at least 4
existing telescope/instrument/window combinations already recur across *different* campaigns
today — meaning `unique_campaign_run_resolved_window` would already have refused to create a
second row for at least 4 real-shaped situations had they all shared one `campaign` value.
This is direct evidence against Option B's viability without folding a new identity field into
the constraint itself (RESEARCH.md Pitfall 2's proposed mitigation (b)). The TBD branch shows
**0** such pre-existing collisions when grouped by `(telescope_instrument, contact_person)`
ignoring `campaign` — the TBD population is small (6 rows) and evidently more discriminated by
contact person than the resolved-window population is by window alone.

The per-source breakdown (`SOURCE_classical_file=1`, `SOURCE_csv_import=11`,
`SOURCE_eso_queue=7`, `SOURCE_lco_queue=5`, `SOURCE_legacy=24`, `SOURCE_web=1`) shows the
existing 49 rows spread across five of the seven declared `Source` values (`gemini_queue`
and `csv_import`'s sibling `legacy` value account for none/most respectively) — D-06's
per-ingest-path picture is measured directly rather than assumed: every existing row traces
to a source that already required a real `campaign`, none traces to an adapter-write path
that doesn't exist yet (`lco_queue`/`gemini_queue` here are pre-v2.3 legacy-labeled rows, not
output of the not-yet-built ADAPT-01..03 adapters).

**No local precedent for the schema shape itself:** none of the three candidate schema
shapes (nullable FK / single sentinel `TargetList` / per-proposal auto-created default
`TargetList`) has any precedent anywhere in this codebase. The precedent that exists (Phase
18's and Phase 26's `{N}-DECISION.md` structure, re-applied here) is for the *investigation
process and document structure*, not for the schema shape itself — this is a first-principles
design decision, per 31-PATTERNS.md's "No Analog Found" entry.
