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

#### SCHEMA-02 evidence - candidate-shape constraint probe

Executed via `tmp/31_constraint_probe.py` (`python manage.py shell < tmp/31_constraint_probe.py`,
captured verbatim to `tmp/31-constraint-probe.txt`) against a disposable copy of the dev
database (`cp src/fomo_db.sqlite3 tmp/31-spike-db-copy.sqlite3`). The DB-path guard printed
`GUARD_DISPOSABLE_COPY=OK` before any write; the schema changes below were applied
in-process via `connection.schema_editor()` — no migration file was written
(`git status --porcelain -- solsys_code/migrations` prints nothing). Five lettered blocks,
19 `PASS:` lines, zero `FAIL:` lines:

```
RESOLVED_DB_NAME=/home/tlister/git/fomo_devel/tmp/31-spike-db-copy.sqlite3
GUARD_DISPOSABLE_COPY=OK
=== Block (A): constraint inventory ===
  UniqueConstraint name='unique_campaign_run_resolved_window' fields=('campaign', 'telescope_instrument', 'window_start', 'window_end') condition=<Q: (AND: ('window_start__isnull', False))>
  UniqueConstraint name='unique_campaign_run_tbd_natural_key' fields=('campaign', 'telescope_instrument', 'contact_person') condition=<Q: (AND: ('window_start__isnull', True))>
  CheckConstraint name='campaign_run_window_start_end_null_together' fields=() condition=<Q: (OR: (AND: ('window_end__isnull', True), ('window_start__isnull', True)), (AND: ('window_end__isnull', False), ('window_start__isnull', False)))>
PASS: source_identifier does not (yet) appear in any existing CampaignRun constraint field set.
=== Block (B): Option A -- nullable campaign FK ===
PASS: created CampaignRun pk=59 with campaign=None, no IntegrityError.
PASS: constraint no longer fires, second row created (pk=60) -- two null-campaign rows sharing telescope_instrument/window_start/window_end did NOT collide on unique_campaign_run_resolved_window. NULL is never treated as equal in a unique index, so this constraint has silently stopped discriminating for exactly the rows Option A gives identity to.
=== Block (C): Option B -- single shared sentinel TargetList ===
Sentinel TargetList pk=7 name='SPIKE-No Campaign'
PASS: created first sentinel-campaign resolved-window run pk=61, no IntegrityError.
PASS: unique_campaign_run_resolved_window fires for two genuinely distinct sentinel-campaign runs: UNIQUE constraint failed: solsys_code_campaignrun.campaign_id, solsys_code_campaignrun.telescope_instrument, solsys_code_campaignrun.window_start, solsys_code_campaignrun.window_end
PASS: created first sentinel-campaign TBD run pk=62, no IntegrityError.
PASS: unique_campaign_run_tbd_natural_key fires for two genuinely distinct sentinel-campaign TBD runs: UNIQUE constraint failed: solsys_code_campaignrun.campaign_id, solsys_code_campaignrun.telescope_instrument, solsys_code_campaignrun.contact_person
=== Block (D): Option C -- per-proposal auto-created default TargetList ===
PASS: two runs under DIFFERENT proposal placeholders (pk=63, pk=64), same telescope_instrument/window, did not collide -- different campaign value each.
PASS: two runs under the SAME proposal placeholder collide on unique_campaign_run_resolved_window: UNIQUE constraint failed: solsys_code_campaignrun.campaign_id, solsys_code_campaignrun.telescope_instrument, solsys_code_campaignrun.window_start, solsys_code_campaignrun.window_end
OPEN QUESTION (Option C, recorded per RESEARCH.md): once a proposal later acquires a real coordinated campaign, does its placeholder TargetList's existing CampaignRun rows get re-pointed, or does the placeholder persist forever alongside the real campaign? Not settled by this probe -- a design question for whichever plan implements Option C, if it wins.
=== Block (E): candidate source_identifier field ===
PASS: source_identifier CharField(null=True) and its partial UniqueConstraint added via schema_editor.
PASS: lco row pk=65 created=True source_identifier='https://observe.lco.global/requests/4247146'
PASS: gemini row pk=66 created=True source_identifier='GEM:GS-2026A-Q-1/GS-2026A-Q-1-0001'
PASS: classical row pk=67 created=True source_identifier='CLASSICAL:SPIKE-NTT:SPIKE-EFOSC2:2026-09-05'
PASS: lco find-or-create on source_identifier created nothing new on the second pass (pk=65).
PASS: gemini find-or-create on source_identifier created nothing new on the second pass (pk=66).
PASS: classical find-or-create on source_identifier created nothing new on the second pass (pk=67).
PASS: CampaignRun count unchanged across the re-write pass (58 before, 58 after).
PASS: unique_campaign_run_resolved_window still fires unmodified with source_identifier present: UNIQUE constraint failed: solsys_code_campaignrun.campaign_id, solsys_code_campaignrun.telescope_instrument, solsys_code_campaignrun.window_start, solsys_code_campaignrun.window_end
PASS: unique_campaign_run_tbd_natural_key still fires unmodified with source_identifier present: UNIQUE constraint failed: solsys_code_campaignrun.campaign_id, solsys_code_campaignrun.telescope_instrument, solsys_code_campaignrun.contact_person
=== Summary ===
FINAL_CAMPAIGNRUN_COUNT=58
```

**Comparison table (D-05's three candidates against both existing partial constraints):**

| Shape | Collides with `unique_campaign_run_resolved_window`? | Collides with `unique_campaign_run_tbd_natural_key`? | What it costs |
|---|---|---|---|
| Option A — nullable `campaign` FK | **No** — never fires for any two null-campaign rows, regardless of telescope/instrument/window overlap (Block B) | Same — `contact_person` alone still discriminates the TBD branch, but the FK itself no longer contributes any discrimination | Cheapest migration (single `AlterField`), but silently loses the resolved-window constraint's discriminating power for every non-campaign row, on top of the 5-site read-path blast radius the prior section measured |
| Option B — single shared sentinel `TargetList` | **Yes** — two genuinely distinct non-campaign runs sharing telescope/instrument/window collide and are correctly refused (Block C) | **Yes** — two genuinely distinct non-campaign TBD runs sharing telescope/instrument/contact_person collide and are correctly refused (Block C) | No schema change to `campaign` at all, but the collision is real: the SCHEMA-01 snapshot already found 4 pre-existing telescope/instrument/window tuples that recur across different real campaigns today — under Option B, any two *new* non-campaign runs sharing those same real-shaped combinations would be refused a second row, exactly the failure mode RESEARCH.md Pitfall 2 predicted, now confirmed constructed-input |
| Option C — per-proposal auto-created placeholder `TargetList` | **Splits, does not eliminate:** two runs under *different* placeholders never collide (Block D); two runs under the *same* placeholder collide exactly like Option B (Block D) | Same split — not exercised separately in Block D's TBD case, but the mechanism is identical (`campaign` is still part of both constraints' field tuples) | Reduces Option B's collision surface by proposal, at the cost of one `TargetList` row per proposal ID and the open question this probe explicitly could not settle: what happens to a placeholder's existing rows once its proposal later gets a real coordinated campaign |

**Block (B)'s null-campaign observation, stated explicitly:** two `CampaignRun` rows with
`campaign=None`, an identical `telescope_instrument`, and an identical resolved window both
saved with **zero** `IntegrityError` — SQLite (and every backend) never treats `NULL` as
equal to `NULL` in a unique index, so `unique_campaign_run_resolved_window` silently stops
discriminating for exactly the population Option A is meant to serve. This is Option A's
load-bearing cost: it is cheap to migrate and reads none of Option B/C's collision risk, but
it also provides **no** duplicate-prevention at all for non-campaign rows unless a new field
(e.g. `source_identifier`) is added to do that job instead.

**Block (E)'s per-ingest-path identity values, as actually written:**

| Ingest path | `source_identifier` value written | Confidence |
|---|---|---|
| LCO (`sync_lco_observation_calendar.py`) | `https://observe.lco.global/requests/4247146` — a **real** LCO portal request `url` copied from an existing `CalendarEvent` row in the dev DB (via the disposable copy) | **Confirmed against real rows** |
| Gemini (`sync_gemini_observation_calendar.py`) | `GEM:GS-2026A-Q-1/GS-2026A-Q-1-0001` — constructed per the adapter's own `f'GEM:{prog}/{obsid}'` pattern (`sync_gemini_observation_calendar.py:150`); no real `GEM:`-namespaced row exists in this dev DB to copy from (0/0, matching 26-DECISION.md's own finding) | **Constructed-input code-path check** |
| Classical (`load_telescope_runs.py`) | `CLASSICAL:SPIKE-NTT:SPIKE-EFOSC2:2026-09-05` — synthesized per RESEARCH.md Pitfall 3's suggested deterministic key (`f'CLASSICAL:{telescope}:{instrument}:{start_time.isoformat()}'`); the classical adapter has no natural single-string value to write today, unlike LCO/Gemini | **Constructed-input code-path check** |

Each of the three rows was re-written a second time through a `get_or_create()` lookup keyed
on `source_identifier` alone; the `CampaignRun` count was identical before and after
(`58` both times), confirming find-or-create idempotency on the new field for all three
adapter shapes. Both original constraints (`unique_campaign_run_resolved_window`,
`unique_campaign_run_tbd_natural_key`) were then re-exercised against a genuine duplicate
with `source_identifier` now present in the schema, and both still raised `IntegrityError`
unmodified — proving the candidate field is additive alongside the existing constraints,
never a silent replacement for them.

**Evidence posture:** writes for real against `tmp/31-spike-db-copy.sqlite3`, a disposable
file copy — no rollback anywhere in the positive-case writes, only in the negative-control
blocks (`transaction.atomic()`), and only to protect the connection from a poisoned
transaction, not to undo a real write. `src/fomo_db.sqlite3`'s fingerprint
(`1208320 1788271090`) is unchanged, matching Task 1's recorded value exactly.

Tag: **Confirmed against real rows** for Block (A)'s constraint inventory (read from the
live model against the real schema shape) and for the LCO row in Block (E) (copied from a
real dev-DB `CalendarEvent.url`). Tag: **Constructed-input code-path check** for Blocks
(B)/(C)/(D) (all synthetic `telescope_instrument`/window values on the disposable copy) and
for the Gemini/classical rows in Block (E) (no real row exists to confirm against).

## Recommendation

### SCHEMA-01 - schema shape for a non-campaign run

The chosen schema shape is **Option A - make `CampaignRun.campaign` nullable**, selected by
the project owner at plan 31-02's Task 1 checkpoint (`checkpoint:decision`,
`gate="blocking-human"`, D-05's one-way door) after reviewing all three evidence
subsections above.

The Django field declaration changes as follows, and only as follows — `related_name` and
`on_delete` are unchanged from today's `models.py:167-173`:

```python
campaign = models.ForeignKey(
    TargetList,
    on_delete=models.PROTECT,   # unchanged here -- SET_NULL/SET() for a placeholder-row
                                 # cleanup case is a separate follow-on decision
                                 # (RESEARCH.md Pattern 2), not settled by this plan
    null=True,                  # was: null=False
    blank=True,                 # new
    related_name='campaign_runs',   # unchanged
    verbose_name='Campaign target list',  # unchanged
)
```

Migration: a single `AlterField` — no `RunPython` backfill step, since D-06 confirms below
that no existing row needs one.

#### Why not the other two

**Option B (single shared sentinel `TargetList`)** is rejected because the SCHEMA-01
dev-DB snapshot found **4 pre-existing `(telescope_instrument, window_start, window_end)`
tuples** that already recur across different real campaigns today, ignoring `campaign`
entirely (`DUP_TELINST_WINDOW_TUPLES_IGNORING_CAMPAIGN=4`). Collapsing every non-campaign
run at every facility onto one shared `campaign` value would make
`unique_campaign_run_resolved_window` refuse a second row for at least those 4 real-shaped
situations — a real collision risk, not a hypothetical one, confirmed further by the
constraint probe's Block (C): two genuinely distinct sentinel-campaign resolved-window runs
correctly collide (`UNIQUE constraint failed`), and the same for the TBD branch.

**Option C (per-proposal auto-created placeholder)** is rejected because the constraint
probe's Block (D) shows it only *splits* Option B's collision risk, it does not eliminate
it: two runs under different placeholders (pk=63, pk=64) did not collide, but two runs
under the *same* placeholder collide exactly like Option B (`UNIQUE constraint failed`,
same as Block C). Since the SCHEMA-01 snapshot's 4 pre-existing colliding tuples are not
known to be spread one-per-proposal, some of those 4 could recur within a single
proposal's own placeholder, reproducing Option B's failure mode at proposal scale. Option
C also carries an open design question the probe explicitly could not settle: what happens
to a placeholder's existing `CampaignRun` rows once its proposal later acquires a real
coordinated campaign.

#### What Phase 32 inherits

Because the chosen shape leaves `campaign` nullable, Phase 32 inherits the full read-site
blast-radius inventory from the "campaign FK read-path blast radius" evidence above as an
explicit obligation — every one of the 5 class-(a) sites must gain a `None`-guard before
any adapter can write a null-campaign row, named here in priority order:

1. `solsys_code/models.py:352` (`CampaignRun.__str__`) — named first: it is rendered on
   the admin changelist, change-form title, delete-confirmation page, admin history
   `object_repr`, and the `CalendarEventMetaAdmin.run` autocomplete JSON.
2. `solsys_code/campaign_reconciler.py:176` (`event_title()`) — the hot site: called on
   every `reconcile_run()` invocation.
3. `solsys_code/campaign_tables.py:467` (`DismissalHistoryTable.render_run`)
4. `solsys_code/campaign_tables.py:538` (second dismissal-table render method)
5. `solsys_code/campaign_attribution.py:397` (candidate-evidence string builder)

The 2 class-(b) pass-through sites (`campaign_reconciler.py:261,398`) survive a null
`campaign` at the read layer, but their downstream `CalendarEvent.target_list` write
behavior is Task 3's/Phase 32's concern, not resolved here. The `on_delete=PROTECT`
question RESEARCH.md Pattern 2 raises (what a placeholder/real campaign deletion should do
to the runs pointing at it) is explicitly **not** decided by this plan — it stays
`PROTECT`, unchanged, and is a follow-on question for whichever plan next touches
`campaign` deletion behavior.

**What happens if a non-campaign run's proposal later acquires a real campaign:** under
Option A this is the simple case the other two candidates complicate — the row's
`campaign` FK is simply updated from `NULL` to the real `TargetList`; no re-migration of
`source_identifier` or any other field is needed, since `source_identifier` (proposed
below, SCHEMA-02) is an additive, independent field that Option A does not tie to
campaign-presence. The row keeps the same primary key and the same `source_identifier`
value throughout — nothing is re-pointed or duplicated the way Option C's placeholder rows
would be — and it only gains real constraint-discrimination via `campaign` for the first
time once the FK is set to a real value.

D-06 is confirmed directly, not assumed: **0 of the 49** existing `CampaignRun` rows have
a null `campaign` today (`NULL_CAMPAIGN_ROWS=0`, `TOTAL_CAMPAIGNRUN_ROWS=49`), so no
existing row needs migrating under this shape — the nullable-FK migration serves only new,
adapter-written rows.

Tag: **Confirmed against real rows** for the 0/49 null-campaign count and the 4
pre-existing colliding tuples (both read from the real, unmodified `src/fomo_db.sqlite3`).
Tag: **Constructed-input code-path check** for the Block (C)/(D) collision demonstrations
(both run against the disposable `tmp/31-spike-db-copy.sqlite3` copy, not real rows).

### SCHEMA-02 - write-time identity field and constraint

The proposed write-time identity field is **`source_identifier`**: a
`CharField(max_length=255, null=True, blank=True)` on `CampaignRun`. It is not separately
`db_index`ed via an explicit `db_index=True` — its index is the one the proposed partial
`UniqueConstraint` below already creates, so a redundant plain index is not needed.

The proposed uniqueness constraint, written out for Phase 32 to transcribe verbatim:

```python
models.UniqueConstraint(
    fields=('source_identifier',),
    condition=models.Q(source_identifier__isnull=False),
    name='unique_campaign_run_source_identifier',
),
```

Both existing constraints, quoted verbatim from `solsys_code/models.py:288-303`, alongside
the non-collision argument for each:

```python
models.UniqueConstraint(
    fields=('campaign', 'telescope_instrument', 'window_start', 'window_end'),
    condition=models.Q(window_start__isnull=False),
    name='unique_campaign_run_resolved_window',
),
models.UniqueConstraint(
    fields=('campaign', 'telescope_instrument', 'contact_person'),
    condition=models.Q(window_start__isnull=True),
    name='unique_campaign_run_tbd_natural_key',
),
```

- `unique_campaign_run_resolved_window`'s field tuple (`campaign`, `telescope_instrument`,
  `window_start`, `window_end`) shares **no field** with the proposed constraint's field
  tuple (`source_identifier`) — the two constraints are disjoint by field set, so a row
  can violate at most one independently of the other; there is no combination of field
  values that could make satisfying one constraint force a violation of the other.
- `unique_campaign_run_tbd_natural_key`'s field tuple (`campaign`, `telescope_instrument`,
  `contact_person`) is likewise disjoint from `(source_identifier,)`, for the same reason.

This is confirmed empirically, not just argued: the constraint probe's Block (E)
re-exercised both existing constraints against a genuine duplicate *after* adding
`source_identifier` to the schema, and both still raised `IntegrityError` unmodified
(`unique_campaign_run_resolved_window still fires unmodified with source_identifier
present`; `unique_campaign_run_tbd_natural_key still fires unmodified with
source_identifier present`) — the new field is additive, never a silent replacement for
either.

**Per-ingest-path table** — the value each of the three ingest paths would write into
`source_identifier`:

| Management command | `CampaignRun.Source` value | `source_identifier` value written | Where the value comes from | Available every time, or only sometimes? | Block (E) probe result |
|---|---|---|---|---|---|
| `load_telescope_runs.py` | `CLASSICAL_FILE` | Synthesized deterministic key: `f'CLASSICAL:{parsed.telescope}:{parsed.instrument}:{start_time.isoformat()}'` | `load_telescope_runs.py:207-216` (the `insert_or_create_calendar_event()` call site, inside the per-night loop where `parsed.telescope`/`parsed.instrument`/`start_time` are all already computed) | Every case — telescope, instrument and `start_time` are computed for every processed night before this call | `CLASSICAL:SPIKE-NTT:SPIKE-EFOSC2:2026-09-05` created (pk=67), find-or-create idempotent on the second pass — Tag: **Constructed-input code-path check** (no real classical schedule file has been run through this path yet; see SCHEMA-03/plan 31-03) |
| `sync_lco_observation_calendar.py` | `LCO_QUEUE` | The LCO portal request URL already extracted for the `CalendarEvent` lookup | `sync_lco_observation_calendar.py:329,341` (`url = fields.pop('url')`, then `insert_or_create_calendar_event({'url': url}, fields)`) | Every case where a `CampaignRun` would actually be written — records whose URL extraction fails are `continue`d past before reaching this point | `https://observe.lco.global/requests/4247146` created (pk=65), copied from a real dev-DB `CalendarEvent.url` — find-or-create idempotent on the second pass — Tag: **Confirmed against real rows** |
| `sync_gemini_observation_calendar.py` | `GEMINI_QUEUE` | The constructed key already built for the `CalendarEvent` lookup | `sync_gemini_observation_calendar.py:150` (`url = f'GEM:{prog}/{record.observation_id}'`), used at line 163 | Every case — `prog` and `record.observation_id` are always present on the record | `GEM:GS-2026A-Q-1/GS-2026A-Q-1-0001` created (pk=66), find-or-create idempotent on the second pass — Tag: **Constructed-input code-path check** (no real `GEM:`-namespaced row exists in this dev DB to confirm against) |

The classical row states explicitly: `source_identifier` is **not** left blank for the
classical path — it carries the synthesized deterministic key above, because the
constraint probe confirmed that key is idempotent under `get_or_create()` (Block (E): the
second write pass created no new row) and additive alongside both existing constraints.
This is a default recommendation, not a final answer for the classical facility
specifically: RESEARCH.md Pitfall 3 and D-07 both note that a real classical schedule file
might carry a proposal code for some run states, which plan 31-03's SCHEMA-03
investigation is tasked with checking against real files; if plan 31-03 finds a better
facility-specific key, it supersedes this default without touching the `source_identifier`
field or constraint itself, only the value the classical adapter writes into it. Until
then, the existing 5-minute telescope/instrument/start_time tolerance match
(`load_telescope_runs.py:207-216`, unchanged by this plan) continues to do the actual
`CalendarEvent`-level matching work; `source_identifier` on `CampaignRun` is an additional,
independent identity surface, not a replacement for that tolerance match.

#### Phase 32 guidance (not work done in this phase)

- **Promote, don't add alongside, once adapters ship:** per this plan's
  `<assumption_delta_decision>`, if Phase 32's adapters write through `source_identifier`
  as their general write-time identity, Phase 32 should **promote** it to the primary
  lookup key for adapter-written rows and demote the `campaign`-plus-window tuple to a
  detail of the campaign-submission variant specifically — not add `source_identifier`
  alongside a still-required `campaign`-plus-window lookup. Adding alongside is accepted
  only if named explicitly as debt, with the condition that would force a later promotion
  stated.
- **Suggested invariant test (not a task in this phase, since this phase writes no source
  code):** an invariant test asserting every `CampaignRun` written by any ingest path is
  findable again through the primary identity key, for every supported
  `CampaignRun.Source` value — this test would go red the moment a later phase
  reintroduces the campaign-is-always-present assumption this plan's evidence has just
  disproven for `LCO_QUEUE`/`GEMINI_QUEUE`/`CLASSICAL_FILE` rows.

**WR-05 finding (`26-DECISION.md` lines 835-859):** a `get_or_create()`/find-or-create
lookup is only race-safe when its lookup fields are backed by a real database constraint.
`source_identifier` **is** proposed as part of a real partial `UniqueConstraint`
(`unique_campaign_run_source_identifier`, above) — so a find-or-create keyed on it alone
would be race-safe once Phase 32 implements it, unlike a design where the field existed
with no backing constraint at all. This closes the WR-05 gap directly rather than leaving
it open for Phase 32 to notice on its own.

Tag: **Confirmed against real rows** for the two existing constraints (read live from
`CampaignRun.Meta.constraints` against the real schema) and for the LCO row's value.
Tag: **Constructed-input code-path check** for the proposed `source_identifier` field/
constraint declaration itself (added via `schema_editor()` against the disposable copy,
never migrated for real) and for the Gemini/classical rows' values.
