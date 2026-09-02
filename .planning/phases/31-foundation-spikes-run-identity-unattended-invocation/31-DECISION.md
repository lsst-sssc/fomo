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

#### SCHEMA-01 evidence - campaign FK read-path blast radius

Grep-derived inventory of every non-test, non-migration site in `solsys_code/` and `src/`
that reads `CampaignRun.campaign`, so Option A's (nullable FK) blast radius is counted
rather than assumed (RESEARCH.md Pitfall 1 / Open Question 2). Four access classes,
reproduced with these exact commands (run from the repository root):

```bash
# Class (a): direct attribute read of the related object (raises AttributeError if null)
grep -rn '\.campaign\.' --include='*.py' solsys_code src | grep -v '/migrations/' | grep -v '/tests/'

# Class (b): related-object assignment/pass-through, not followed by a dot (survives null)
grep -rn '\.campaign\b' --include='*.py' solsys_code src | grep -v '/migrations/' | grep -v '/tests/' | grep -v '\.campaign\.'

# Class (c): queryset traversals assuming a joinable row
grep -rn "select_related(.campaign" --include='*.py' solsys_code src | grep -v '/tests/'
grep -rn "run__campaign" --include='*.py' solsys_code src | grep -v '/tests/'
grep -rn "campaign__[a-z]" --include='*.py' solsys_code src | grep -v '/tests/'

# Class (d): template-variable reads of the FK's related object
grep -rn '\.campaign\.' src/templates solsys_code/*/templates 2>/dev/null
grep -rln 'run\.campaign\|record\.run\.campaign' src/templates solsys_code 2>/dev/null
```

| File:Line | Class | Expression | Verdict under a null `campaign` |
|---|---|---|---|
| `solsys_code/models.py:352` | (a) | `self.campaign.name` (`CampaignRun.__str__`) | Raises `AttributeError` — this label is rendered on the admin changelist, change-form title, delete-confirmation page, admin history `object_repr`, and the `CalendarEventMetaAdmin.run` autocomplete JSON (per the docstring at `models.py:319-335`) |
| `solsys_code/campaign_reconciler.py:176` | (a) | `run.campaign.name` (`event_title()`) | Raises `AttributeError` — called on every `reconcile_run()` invocation to build the projected `CalendarEvent`'s title |
| `solsys_code/campaign_tables.py:467` | (a) | `record.run.campaign.name` (`DismissalHistoryTable.render_run`) | Raises `AttributeError` — rendered per row of the dismissal-history table |
| `solsys_code/campaign_tables.py:538` | (a) | `record.run.campaign.name` (second dismissal-table render method) | Raises `AttributeError` — same risk as the row above, second table instance |
| `solsys_code/campaign_attribution.py:397` | (a) | `run.campaign.name` (candidate-evidence string builder) | Raises `AttributeError` — called while building attribution-queue candidate evidence text |
| `solsys_code/campaign_reconciler.py:261` | (b) | `'target_list': run.campaign,` (whole-window `CalendarEvent` fields dict) | Survives — assigns `None` into the fields dict; whether the downstream `CalendarEvent.target_list` write itself then raises is a write-path question, out of this read-path inventory's scope and covered by Task 3's constraint probe |
| `solsys_code/campaign_reconciler.py:398` | (b) | `'target_list': run.campaign,` (per-night `CalendarEvent` fields dict) | Survives — same pass-through as the row above, per-night branch |
| `solsys_code/campaign_views.py:366` | (c) | `.select_related('campaign', 'site')` (approval-queue queryset) | Survives at query time (`LEFT OUTER JOIN`, no raise for a null FK); the risk moves to wherever the resulting rows are later read |
| `solsys_code/campaign_views.py:378` | (c) | `.select_related('campaign', 'site')` (second approval-queue queryset) | Survives at query time, same as above |
| `solsys_code/campaign_views.py:405` | (c) | `runs_needing_site_review().select_related('campaign', 'site')` | Survives at query time |
| `solsys_code/admin.py:225` | (c) | `.select_related('campaign', 'site')` (`CampaignRunAdmin.get_queryset`) | Survives at query time |
| `solsys_code/campaign_attribution.py:572` | (c) | `_eligible_runs_for_event(event).select_related('campaign', 'site')` | Survives at query time |
| `solsys_code/campaign_attribution.py:645` | (c) | `_eligible_runs_for_record(record).select_related('campaign', 'site')` | Survives at query time |
| `solsys_code/campaign_views.py:981` | (c) | `.select_related('event', 'run__campaign', 'dismissed_by')` | Survives at query time |
| `solsys_code/campaign_views.py:982` | (c) | `.select_related('observation_record', 'run__campaign', 'dismissed_by')` | Survives at query time |
| `solsys_code/campaign_views.py:1001` | (c) | `.select_related('event', 'run__campaign', 'confirmed_by')` | Survives at query time |
| `solsys_code/campaign_views.py:1002` | (c) | `.select_related('observation_record', 'run__campaign', 'confirmed_by')` | Survives at query time |
| `solsys_code/admin.py:290` | (c) | `list_select_related = ['event', 'run', 'run__campaign', 'run__site']` (`CalendarEventMetaAdmin`) | Survives at query time |
| `solsys_code/campaign_attribution.py:541` | (c) | `CampaignRun.objects.filter(campaign__in=record.target.targetlist_set.all())` | Survives — a null `campaign` simply never matches this `__in` filter, excluding the row rather than raising |
| — | (d) | none found | **Zero** genuine `{{ run.campaign.<attr> }}`-style template traversals exist anywhere under `src/templates/` or a `solsys_code/*/templates/` directory — recorded as an explicit zero count, not a search gap. The one adjacent template site, `src/templates/tom_calendar/partials/event_form.html:136` (`{% url 'campaigns:table' run.campaign_id %}`), reads the raw FK id column (`campaign_id`), not the related object, so it is not a class (d) site; a null id there could still raise `NoReverseMatch` from the `{% url %}` tag (a raise, not a silent blank), noted here for completeness but not counted in class (d)'s total |

**Totals:** Class (a) 5 sites, class (b) 2 sites, class (c) 12 sites, class (d) 0 sites (1
adjacent non-class-(d) template site noted above). **32 additional occurrences of
`.campaign.`** were excluded because they live under `solsys_code/*/tests/` (test fixtures
constructing `CampaignRun` rows directly, e.g. `self.campaign.pk`/`self.campaign.name`
where `self.campaign` is a test's own `TargetList` fixture, not a `CampaignRun.campaign`
FK read) — 10 distinct test files.

**Reading for the D-05 comparison:** under Option A, **5 sites** (all class (a)) would need
an explicit null guard to avoid a new `AttributeError`, plus the 2 class (b) sites whose
downstream write behavior is Task 3's concern, not this read-path inventory's. Of the 5,
one (`campaign_reconciler.py:176`, `event_title()`) sits on a genuinely hot path — it runs
on every `reconcile_run()` invocation, the core sync entry point every adapter will call.
The other four (`models.py:352`, the two `campaign_tables.py` dismissal-table renderers,
and `campaign_attribution.py:397`) are staff-facing UI render paths (admin labels,
dismissal-history table, attribution-queue evidence text) — invoked per page view, not per
sync tick, and are judged cold relative to the reconciler's hot path. This hot/cold split is
a code-reading judgement, not a profiling measurement — no profiler was run.

Tag: **Confirmed against real rows** for the grep-derived inventory itself (the greps ran
against the real, current source tree). Tag: **Constructed-input code-path check** for the
hot-versus-cold judgement, since no profiling was run.
