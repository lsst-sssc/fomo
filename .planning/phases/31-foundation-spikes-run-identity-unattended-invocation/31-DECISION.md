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

#### SCHEMA-03 evidence - classical schedule-file sample inspection

Dated 2026-09-02. **1 real classical schedule file** was obtained from the operator and
inspected: `tmp/31-classical-samples/didymos_2026_july_classical_runs.txt` (git-excluded,
never staged; `git ls-files tmp/` prints nothing). `CLASSICAL_SAMPLE_FILES=1`. The file
contains 10 total lines: **3 real classical run lines**, 1 blank separator line, a 5-line
block the file itself marks "not classical-schedule format -- informational only" (a Gemini
queue-program note, excluded from this inspection per its own label and per this plan's
content-handling rule — no field from that block, including any name or title, is quoted
anywhere below), and a trailing blank line.

Each of the 3 real classical run lines was run through `solsys_code.telescope_runs.
parse_run_line()` in a read-only `python manage.py shell` session (a pure-function parse; no
database row was touched):

| Line | Bare status word present? | `parse_run_line` outcome | Proposal-code-shaped token present? |
|---|---|---|---|
| 1 | Yes — `allocation` | **Rejected**: `ValueError` ("Unrecognized status 'EFOSC2'...") | **Yes** — a dot-delimited three-segment token (`digits.alphanumeric.digits`-shaped, resembling an ESO Tatoo/proposal-ID format), positioned before the telescope token |
| 2 | No (defaults to `allocation`, per `ParsedRun.status`'s documented default) | Parsed successfully | No |
| 3 | No (defaults to `allocation`) | Parsed successfully | No — the line's only trailing token is a partial-night window matching the documented `(BoN\|\d{4})-(EoN\|\d{4})` grammar, structurally distinct from a proposal code |

**1 of 3 real lines was rejected** by `parse_run_line`. The rejection is not actually a
status-vocabulary gap (`allocation` is a real `KNOWN_STATUSES` member and is found correctly
by `_resolve_status()` before the failure) — it is positional. Today's grammar is
`telescope instrument [status] daterange [(status)]`, with `telescope` = token 0 and
`instrument` = token 1 of whatever remains once the status word is removed. Line 1 places
its proposal-code-shaped token *before* the telescope name, so after status removal the
remaining tokens are `[proposal-code, NTT, EFOSC2]`. `parse_run_line` reads token 0 (the
proposal code) as the telescope, token 1 (`NTT`) as the instrument, leaving `EFOSC2` as an
unconsumed third token — which the parser treats as an unrecognized status-shaped leftover,
raising `ValueError: Unrecognized status 'EFOSC2'...`. This is a genuine, real-sample
confirmed parser gap: any classical line carrying a leading proposal code in this position
is rejected outright by today's grammar, independent of which status word it carries.

**Distinct status words actually observed:** only `allocation` — the sole bare status word
literally present in any of the 3 real lines (line 1). Lines 2 and 3 carry no status word at
all and fall to `ParsedRun`'s documented default of `'allocation'`. No other
`KNOWN_STATUSES` member (`proposed`, `confirmed`, `cancelled`, `not confirmed`) occurs
anywhere in this sample.

**Reconciling D-07 against the parser's real vocabulary:** `KNOWN_STATUSES` contains
`{'allocation', 'proposed', 'confirmed', 'cancelled', 'not confirmed'}` — neither `planned`
nor `observed` (D-07's remembered words) is a member. `CampaignRun.RunStatus`
(`solsys_code/models.py:96-106`), by contrast, *does* declare exactly `PLANNED` and
`OBSERVED` (alongside `REQUESTED`, `REDUCED`, `PUBLISHED`, `CANCELLED`, `NOT_AWARDED`,
`WEATHER_TECH_FAILURE`). This real sample supports the conclusion that D-07's recollection
belongs to `CampaignRun.RunStatus`'s lifecycle vocabulary, not to the classical
schedule-file's `KNOWN_STATUSES` vocabulary: none of the 3 real lines contains `planned` or
`observed` as a file-level status word, and none could, since neither word is a member of
`KNOWN_STATUSES` at all.

**Proposal-code / status correlation (weak, single data point):** the one line carrying a
proposal-code-shaped token (line 1) is also the one line carrying an explicit status word
(`allocation`); both lines with no proposal code (2, 3) fall to the default status rather
than stating one explicitly. This is loosely suggestive of D-07's recollection (code present
alongside a stated status, absent when status is merely implied), but rests on n=1 versus
n=2 within a single 3-line file — far too thin to generalize, and stated here as an
observation, not a conclusion.

**`ParsedRun` field count:** 9 fields (`telescope`, `instrument`, `status`, `year`, `month`,
`day1`, `day2`, `start_window`, `end_window`). None of them is a proposal code today. Any
facility-specific key incorporating a proposal code would require a parser/grammar change in
a later phase — both a new field and new positional handling, since line 1 shows the code
does not sit in a position today's grammar reserves for anything — stated here as a
consequence for Phase 32, not as work done in this plan.

Tag: **Confirmed against real rows** — every fact in this section is read directly out of
the one real operator-supplied file and its live `parse_run_line()` outcomes; no
documentation example or synthetic line was needed to fill a gap.

### Scheduling track (SCHED-07)

#### SCHED-07 evidence - interim host verification

Dated 2026-09-02. Captured live, during this plan, to `tmp/31-host-probe.txt` (git-excluded;
`git ls-files tmp/` prints nothing) via three probe commands run in sequence from the
repository root, each appending to the same transcript. The crontab block was piped through
a substitution that replaces the value part of any assignment whose name contains a
secret-sounding word (`KEY`, `TOKEN`, `PASSWORD`, `SECRET`, `CREDENTIAL`) with `REDACTED`
before anything was written to the transcript — no such assignment appears in the real
crontab captured here, so the substitution had nothing to redact, which is itself recorded
rather than silently assumed.

```
=== uname ===
Linux [hostname redacted] [kernel build string redacted] x86_64 GNU/Linux
=== init ===
systemd 252 (252-67.el9_8.4.rocky.0.1)
=== flock ===
flock from util-linux 2.37.4
flock: probe exit 0
=== crontab ===
[1 unrelated commented-out cron entry redacted]
0 * * * * /home/tlister/venv/fomo311_venv/bin/python /home/tlister/git/fomo_fresh/manage.py rundataquery 1
17 * * * * /home/tlister/venv/fomo311_venv/bin/python /home/tlister/git/fomo_fresh/manage.py updatescout --skip-designations
47 4 * * * /home/tlister/venv/fomo311_venv/bin/python /home/tlister/git/fomo_fresh/manage.py updatescout --skip-reconcile

[2 flock-guarded cron entries for an unrelated project redacted]
crontab: probe exit 0
=== heartbeat egress ===
301
heartbeat: curl exit 0
```

**Correction, recorded during the code-review fix pass:** the block above redacts the
developer hostname, the exact kernel build string, one unrelated commented-out cron entry
(`update_sentry_risk.py`), and two `flock`-guarded cron entries belonging to a different,
unrelated project (`scout-alert-bridge`) that were reproduced verbatim here in an earlier
version of this document. None of the redacted material was a credential or changes any
finding below — the evidence this section rests on is unchanged: `flock` present, 3 active
`manage.py` entries, 0 guarded. The unredacted transcript remains available locally at
`tmp/31-host-probe.txt` (git-excluded, never committed).

Tag: **Confirmed against real rows** for all three probe outputs — this transcript was
produced live during this plan's execution, not carried over from RESEARCH.md's earlier
session, per the roadmap's own bar for Success Criterion 4.

**First finding — lock utility.** `flock` is present: `flock from util-linux 2.37.4`, probe
exit `0`. This is the technical half of D-03's first open question, confirmed for the
interim host: the lock utility this mechanism depends on is already installed here, nothing
needs adding to this host for the cron+flock mechanism to function.

**Second finding — existing cron precedent, the one that matters most.** The real crontab
already invokes Django management commands today: **3 active entries** target
`/home/tlister/git/fomo_fresh/manage.py` (`rundataquery 1`, hourly at `:00`;
`updatescout --skip-designations`, hourly at `:17`; `updatescout --skip-reconcile`, daily at
`4:47`). **Correction, recorded during the code-review fix pass:** neither command exists in
this repository (`grep -rn "rundataquery\|updatescout" --include=*.py .` returns nothing) —
`updatescout` belongs to the third-party `tom_jpl` plugin and `rundataquery` to
`tom_dataservices`, both invoked from `/home/tlister/git/fomo_fresh`, a sibling checkout of
this same project, not FOMO's own management commands. **0 of these 3 carry any overlap
guard** — none is wrapped in `flock` or any other serialization mechanism. This confirms
RESEARCH.md Pitfall 4's finding directly, one session later: the concurrent-write-contention
risk in `.planning/codebase/CONCERNS.md:159` is a live condition on this host today, not a
hypothetical this phase is inventing. This is the strongest single piece of evidence for
D-02's cron+flock choice — the ratio is **0 guarded / 3 total Django management-command
entries (from the `tom_jpl` and `tom_dataservices` plugins, not FOMO's own management
commands)**. (The crontab also carries two `flock`-guarded entries for a different project,
`scout-alert-bridge`, which invoke a shell script rather than a Django management command and
are noted here only to be excluded from the 0/3 count above — they do not change that ratio;
a fifth line, the `update_sentry_risk.py` cron job, is commented out (`#`-prefixed) and
therefore not an active entry at all.) This is exactly RESEARCH.md's own observation, carried
forward rather than silently resolved; see the human-check below.

**Third finding — outbound heartbeat egress.** The heartbeat check against `hc-ping.com`
returned HTTP status **301** (a redirect response), with `curl` itself exiting `0`. A
three-digit response proves egress: this host can reach a third-party heartbeat service over
HTTPS today. This resolves only the **technical** half of D-03's second open question —
whether an outbound ping to a third-party service is acceptable on policy or compliance
grounds is a separate question this probe cannot answer and does not attempt to; RESEARCH.md
logs it as Assumption A2, and it remains open here, routed to the operator via the human-check
below and to Phase 34/task 3's recommendation as an explicit unresolved item.

**Scope caveat governing the whole track:** these facts are about the shell this plan ran in.
RESEARCH.md Open Question 3 records strong circumstantial evidence that this is the machine
D-01 describes (the kernel string and crontab paths above are consistent with it), but only
the operator can confirm it — and even a confirmed match proves nothing about the container
image or the AWS target, which no probe run from this shell can reach.

| Scope | Status |
|---|---|
| Interim host (Rocky 9 / WSL2, D-01) | **Reached** — flock present, 0/3 unguarded FOMO cron entries confirmed, heartbeat egress confirmed (HTTP 301) |
| FOMO container image | Not reached by this task — see the next evidence subsection |
| Eventual AWS Kubernetes target | Not reached by this task — see the next evidence subsection |

#### SCHED-07 evidence - container image and AWS scope

Dated 2026-09-02. Captured live to `tmp/31-container-probe.txt` (git-excluded; `git ls-files
tmp/` prints nothing).

**This repository tracks no container build file at all.** A tracked-file search for any
Dockerfile, Containerfile, docker-compose file, Helm chart or Kubernetes manifest returned
`NONE`:

```
=== container build files tracked in this repo ===
NONE
```

This is a genuine absence, not an oversight of the search: `git ls-files` enumerates every
file this repository actually tracks, so a build definition living anywhere in this repo
would have appeared here. **Consequence:** if the mechanism is cron plus a file lock inside
the FOMO container, no such container is defined anywhere in this codebase today — something
must both define that container image and, inside it, install a cron daemon and the lock
utility, and run the daemon as the container's entry point. Whoever writes the container
definition inherits that requirement; it is named here as an explicit **Phase 34
dependency**, not left implied.

Because no build definition exists, there is nothing to build a real FOMO image from. A local
image inventory (via `docker image ls`, the runtime that responded on this host) was captured
to establish what *is* available locally rather than assuming nothing is:

```
=== local image inventory ===
scout-alert-bridge:uv-test
scout-alert-bridge-bridge:latest
postgres:16-alpine
[internal-registry]/neoexchange:test_new_pyslalib
quay.io/minio/minio:RELEASE.2025-02-07T23-21-09Z
apache/kafka:3.8.0
rockylinux:9
rockylinux:8
rabbitmq:3.10.6-management-alpine
nginx:1.21-alpine
dannygoldstein/zuds-demo:latest
dannygoldstein/zuds:latest
dannygoldstein/zuds-db:0.1dev
```

No image in this inventory plausibly belongs to FOMO (no `fomo`-named repository present;
the closest neighbor, `[internal-registry]/neoexchange:test_new_pyslalib`, is a different LCO
project). **Branch taken: no FOMO image exists, so the two checks were run inside a
representative stand-in base image instead** — `python:3.11-slim`, pulled fresh for this
task (egress to Docker Hub succeeded, itself confirming outbound network access works from
this host beyond just `hc-ping.com`) — with the container removed afterward (`docker run
--rm`, confirmed via `docker ps -a --filter ancestor=python:3.11-slim` returning no rows
post-run):

```
=== stand-in base image: python:3.11-slim ===
=== flock (base image) ===
flock from util-linux 2.41.5
flock: probe exit 0
=== heartbeat egress (base image) ===
bash: line 1: curl: command not found
heartbeat: curl exit 127
```

This stand-in result is a genuinely mixed, useful finding, labeled for exactly what it is:
`flock` **is** present in this particular slim Debian-based image (util-linux ships as part
of Debian's base layer), but `curl` is **not** — a concrete, real example of RESEARCH.md's
warning that slim base images often omit tools an unattended-invocation mechanism needs,
here caught for the heartbeat check's own client rather than for `flock` itself. This proves
nothing about FOMO's actual image, which does not exist yet: FOMO's own container definition,
whenever written, must explicitly ensure both `flock` and an HTTP client (`curl`, or Python's
already-available `requests`, since the heartbeat call happens from inside a Django management
command rather than a shell script) are present, rather than assuming either survives from
whatever base image is chosen.

Tag: **Constructed-input code-path check** for both checks above — they ran inside a stand-in
base image, not an actual FOMO image, per this task's instruction to tag any check against a
substitute image this way rather than upgrading it to "Confirmed against real rows."

**Container row of the scope table:** status remains **unconfirmed** — a stand-in check is
informative but is not evidence about an image that does not exist. No tag upgrade is applied.

**AWS row.** Nothing this phase can run reaches LCO's AWS Kubernetes cluster; its status is
**unconfirmed by construction** — no probe was attempted, and none could succeed from this
shell. Two concrete things a later phase (Phase 34, or a deployment phase) must check there,
one technical and one policy:

1. **Technical:** whether the container's base image, once one is written, ships the lock
   utility (`flock`) — the same gate this task just demonstrated a slim base image can fail
   silently for an adjacent tool (`curl`).
2. **Policy:** whether outbound egress to a third-party heartbeat service (`hc-ping.com` or
   equivalent) is permitted by the AWS cluster's network policy *and* by institutional policy
   — this is the half of D-03's second open question no technical probe run from any shell can
   settle; it requires an explicit answer from whoever owns that deployment's network and
   compliance posture.

Updated three-scope table:

| Scope | Status |
|---|---|
| Interim host (Rocky 9 / WSL2, D-01) | **Reached** — flock present, 0/3 unguarded FOMO cron entries confirmed, heartbeat egress confirmed (HTTP 301) |
| FOMO container image | **Unconfirmed** — no container build definition exists in this repository; checks ran against a stand-in `python:3.11-slim` image only (flock present, curl absent), which proves nothing about an image that does not exist yet |
| Eventual AWS Kubernetes target | **Unconfirmed by construction** — no probe reaches this scope from any developer shell; two concrete follow-ups (lock-utility presence, network/institutional policy for outbound egress) are named above for a later phase |

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

**Correction, recorded during the code-review fix pass:** the synthesized key as written
above (`f'CLASSICAL:{telescope}:{instrument}:{start_time.isoformat()}'`) is exact-string
equality on a computed datetime, not a tolerance match — it is **not** a mirror of the
5-minute tolerance match it is meant to accompany. A `start_time` recomputation that drifts
by even one second (an edited site altitude, an ephemeris or library update) produces a
different key, and a find-or-create keyed on it then creates a duplicate row — precisely
the failure `source_identifier` exists to prevent. Phase 32 must quantise `start_time` to
the same granularity the tolerance match implies (e.g. floor to the nearest 5-minute
boundary) before formatting the key, and a 5-minute bucket still splits a pair straddling a
boundary — an open gap, not fully closed by quantisation alone. Separately, the probe's
`classical_start` value (`tmp/31_constraint_probe.py`) is a `datetime.date`
(`date(2026, 9, 5)`), not the `datetime` the loader actually computes, so
`CLASSICAL:SPIKE-NTT:SPIKE-EFOSC2:2026-09-05` in the Block (E) transcript above was
produced with a value shape `load_telescope_runs.py` will never write. The round-trip
result is real but must be re-confirmed with a genuine datetime before Phase 32 relies on
it as proof the key works end-to-end.

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

### SCHEMA-03 - classical write-time identity surface

No — the existing 5-minute telescope/instrument/start_time tolerance match
(`_START_TIME_MATCH_TOLERANCE = timedelta(minutes=5)`, `load_telescope_runs.py:22`, applied
via the `{'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time':
start_time}` lookup passed to `insert_or_create_calendar_event()` at
`load_telescope_runs.py:207-216`) is **not sufficient** as a write-time identity surface on
its own. The concrete failing case: **two genuinely distinct proposals allocated the same
telescope, the same instrument and the same night, with no partial-night window
distinguishing them.** Neither `status` nor any proposal identifier is part of that lookup's
field set, so two run lines sharing telescope + instrument + full night resolve to the
identical `start_time` (a pure function of the site's sunset/sunrise geometry for that
calendar date, per `sun_event()`) — the second `insert_or_create_calendar_event()` call
therefore *updates* the first `CalendarEvent` row rather than creating a second one, silently
overwriting whichever proposal's title/description got there first.

Each of the four candidate cases, checked against the loader's actual lookup code:

1. **Instrument swap on the same telescope, same night** — ruled out. `instrument` is
   already one of the three tolerance-match fields (`load_telescope_runs.py:207-216`), so
   two runs differing only by instrument never share a lookup key, regardless of night
   overlap.
2. **Night-convention shift** (the Las Campanas both-inclusive vs. ESO noon-to-noon rules
   `_iter_run_nights` dispatches on) — ruled out for today's fixed `SITES`/
   `ESO_NOON_TO_NOON_SITES` configuration. The convention is a static per-site lookup, not
   something that varies between two ingests of the same schedule line, so re-ingesting an
   unchanged line always computes the same night sequence and the same `start_time` (within
   the ~2-second IERS drift the 5-minute tolerance is explicitly sized to absorb, per the
   comment at `load_telescope_runs.py:12-21`). The residual risk is a *future code change*
   reclassifying a site's convention, which would shift every existing event's expected
   `start_time` by up to a full night — a code-deployment risk, not a data-variability one,
   and unrelated to proposal identity; noted here, not treated as this plan's failing case.
3. **Re-issued schedule file, run's status changed but nothing else did** — ruled out.
   `status` is deliberately excluded from the tolerance-match lookup, so a line whose only
   change is its status word (e.g. `allocation` -> `confirmed`) correctly matches the
   existing `CalendarEvent` and updates it (new title prefix / description) instead of
   creating a duplicate — this is the intended behavior, confirmed by reading the lookup
   dict directly rather than assumed.
4. **Two different proposals allocated the same telescope, instrument and night** — **named
   as the failing case** (above). This also covers the sequential variant of the same gap: a
   re-issued schedule file that *reassigns* a night from one proposal to a genuinely
   different one (not merely relabeling its status) would silently overwrite the record via
   the identical identity gap — which matters specifically because `CampaignRun`'s whole
   purpose is proposal/campaign attribution, so an overwritten row loses the earlier
   proposal's attribution history rather than merely losing a display label.

**Consequence for the SCHEMA-02 field:** the classical path does **not** leave
`source_identifier` blank — per plan 31-02's recommendation it writes the synthesized
deterministic key `f'CLASSICAL:{telescope}:{instrument}:{start_time.isoformat()}'`. That
formula uses exactly the same three fields as the tolerance-match lookup above, so it
inherits the identical blind spot: two different proposals sharing telescope + instrument +
night would also produce the identical synthesized `source_identifier` string, and a
`get_or_create()` keyed on it would treat them as the same row. This is **not** a
contradiction of plan 31-02's nullable-vs-synthesized decision — the field stays populated,
exactly as planned, with no compatibility break — but it is an *incomplete* mitigation:
promoting `source_identifier` to the primary identity key later (per SCHEMA-02's own
Phase 32 guidance) would carry this exact gap forward unless a proposal-code component is
folded into the formula.

A proposal code is **not** currently a reliable substitute for closing this gap. Only 1 of
the 3 real lines in this plan's sample carried a proposal-code-shaped token at all (2 of 3
had none), and where present it did not even parse under today's grammar — it triggered a
`ValueError` because its position (prefixed before the telescope token) breaks the
documented `telescope instrument [status] daterange` format. A code available only
sometimes, and only ever observed in an unparseable position, is not usable as a
fallback-free identity key, per RESEARCH.md Pitfall 3's own warning against exactly this.

**Recommendation for now:** keep the tolerance match (and its `source_identifier` mirror) as
today's default — 2 of the 3 real sample lines parse cleanly and behave correctly under it —
and accept the identified gap as a documented, low-observed-frequency risk (classical
scheduling committees generally allocate one telescope+instrument to one proposal per full
night; an actual double-booking or an unmarked proposal reassignment is the trigger, and
neither was observed in this sample). Phase 32 inherits two explicit, separable follow-on
items, not a closed question: (a) extend `ParsedRun`/`parse_run_line`'s grammar to recognize
an optional leading proposal-code token — needed regardless of whether it becomes part of
the identity key, since today's grammar rejects that shape outright; and (b) once a proposal
code is reliably extractable, decide whether to fold it into both the `CalendarEvent` lookup
and the `source_identifier` formula, or to keep the current pair with the
double-booking/reassignment gap documented as an accepted risk.

**Compatibility with plan 31-02:** this verdict is compatible with SCHEMA-01's nullable-FK
decision — `campaign` nullability is orthogonal to this finding, which concerns
`CalendarEvent`/`source_identifier` matching, not `CampaignRun.campaign`. It is also
compatible with SCHEMA-02's decision to populate (not blank) `source_identifier` for the
classical path — no contradiction there — but it does not fully resolve the gap that
decision was meant to help close; Phase 32 inherits the two follow-on items above rather
than a settled answer.

Tag: **Confirmed against real rows** for the parse facts this verdict rests on (1 real file,
3 real lines, 1 rejected, 1 proposal-code-shaped token observed in an unparseable position).
Tag: **Constructed-input code-path check** for the sufficiency verdict and the
two-proposals-same-night failing case itself — no actual collision between two proposals
was observed in this 3-line sample; the failing case is reasoned directly from the loader's
own lookup/tolerance code, not from an observed collision. Phase 32 should re-check this
verdict against the next real classical schedule file it sees, specifically watching for two
distinct full-night entries sharing one telescope and instrument.

### SCHED-07 - unattended invocation mechanism

The mechanism is **cron plus a per-command file lock (`flock`)** on the interim host
(Rocky 9 / WSL2) today, per D-02.

**Correction, recorded during the code-review fix pass:** this is *not* the same mechanism
that carries over unchanged once deployed inside LCO's AWS Kubernetes cluster. A `flock`
lock lives on a single filesystem in a single container; on Kubernetes, two pods of the
same workload (a rolling update overlapping old and new, a restarted pod, a CronJob with
`concurrencyPolicy: Allow`, or any replica count above one) each get their own writable
layer and therefore their own lock file, so two concurrent runs of the same management
command would both acquire "the" lock and both proceed. The idiomatic and correct
construct on Kubernetes is a `CronJob` per command with `concurrencyPolicy: Forbid` (and
`startingDeadlineSeconds` set); `flock` is retained there only as an in-container
belt-and-braces guard against two processes inside the *same* pod, never as the
cross-pod migration story on its own. This is carried forward as an open item for
whoever owns the AWS deployment, not settled by this spike. A task queue (Celery/huey/APScheduler)
was not chosen: the project's own v2.3 Out-of-Scope table already states that no
broker/worker infrastructure buys anything for a single-server, few-jobs-an-hour deployment,
and this phase's evidence confirms no blocker was found to the simpler choice — flock is
already installed on the interim host (`flock from util-linux 2.37.4`, task 1) and cron
already runs this project's management commands there today. The one condition that would
reopen this question: if a future job count or latency requirement genuinely needs
concurrent, cross-host worker distribution that a single per-host file lock cannot express —
nothing found during this phase's investigation suggests that condition exists today.

**Overlap prevention.** The exact invocation shape Phase 34 will write into a cron line, per
component:

```
/usr/bin/flock -n /var/lock/fomo/<command-name>.lock \
    /path/to/venv/bin/python /path/to/checkout/manage.py <command-name> [args]
```

- **Lock utility, absolute path:** `/usr/bin/flock` (confirmed present, task 1's transcript
  above; util-linux 2.37.4 on the interim host).
- **Fail-fast flag:** `-n` (`--nb`, non-blocking) — a second invocation that finds the lock
  already held gives up immediately rather than queuing behind it.
- **Lock file path convention:** one lock file per command, not one shared lock across all
  scheduled commands — e.g. `/var/lock/fomo/sync_lco_observation_calendar.lock`,
  `/var/lock/fomo/sync_gemini_observation_calendar.lock`, one per management command name.
  This is deliberate: a slow-running sync for one facility must never starve or delay an
  unrelated command's own scheduled tick just because they happen to share a lock file.
- **Interpreter, absolute path:** the project's virtualenv `python`, e.g.
  `/home/tlister/venv/fomo311_venv/bin/python`, matching the exact pattern already in use by
  this host's real crontab entries (task 1's transcript) — not the bare `python3` on `PATH`,
  which may resolve to the wrong interpreter or environment under cron's minimal `PATH`.
- **`manage.py`, absolute path:** the project checkout's `manage.py`, e.g.
  `/path/to/checkout/manage.py` — cron does not run with the working directory Phase 34's
  command expects, so a relative path is not reliable.
- **Command name:** the specific management command being scheduled (e.g.
  `sync_lco_observation_calendar`).

**Adjacency question, answered directly:** when the previous run is still holding the lock as
the next tick fires, the new invocation is **skipped**, not queued and not blocked — `flock
-n` fails fast and exits immediately rather than waiting for the lock to release. A silently
skipped tick is indistinguishable from a tick that ran and found nothing to do, so a skipped
tick must be visible somewhere — this is exactly what the missed-invocation visibility layer
below exists to catch, since a `flock -n` failure exit code, left unobserved, is itself a
silent failure mode.

This overlap-prevention need is not hypothetical: task 1's transcript shows the real crontab
already has **0 of 3** existing Django management-command entries (from the `tom_jpl` and
`tom_dataservices` plugins, not FOMO's own management commands, invoked from a sibling
checkout) carrying any overlap guard at all (`rundataquery`, `updatescout
--skip-designations`, `updatescout --skip-reconcile`, all unguarded) — making the
concurrent-write risk in `.planning/codebase/CONCERNS.md:159` a live condition on this host
today, not a hypothetical this phase is inventing.

**Credential handling.** Credentials reach an unattended run as **environment variables**,
extending the pattern this project already uses for its alert-stream credentials —
`src/fomo/settings.py:309-314` reads `FINK_CREDENTIAL_URL`, `FINK_CREDENTIAL_USERNAME`,
`FINK_CREDENTIAL_GROUP_ID`, and related `FINK_*` names via `os.getenv(...)` with a
descriptive placeholder default when unset. New LCO/Gemini credentials for Phase 34's
unattended run should follow the identical `<SERVICE>_CREDENTIAL_<FIELD>`-style naming
convention, not any value itself.

A credential must **never** be passed as a positional or keyword argument to a management
command, because a process listing (`ps aux`) exposes the full argument vector to any local
user of the same host — and the same applies to a cron line itself, which is world-readable
to the owning user's own processes (and, depending on `crontab` file permissions, potentially
to other local users). Environment-variable injection at the container/host level, read
inside the process via `os.getenv()`, avoids both exposure paths.

One further finding this phase surfaced, not previously recorded: `src/fomo/settings.py:400`
imports `fomo.local_settings` at the end of the settings module — a git-excluded,
machine-local override file that, per `.planning/codebase/CONCERNS.md`'s existing "Hardcoded
API Keys in settings.py" finding, holds a live credential on the real host. This means **any
probe, transcript, or documentation page that quotes settings output verbatim is one step
away from committing a secret** — this phase's own task 1/task 2 probes deliberately never
read `local_settings.py` or any facility credential dict for exactly this reason. This is
named here as a **standing constraint on Phase 34's own evidence-gathering**, not just on its
code: any future spike or debug session that dumps live settings values for diagnostic
purposes must apply the same redaction discipline this phase's crontab capture used.

**Missed-invocation visibility.** SCHED-09 requires two independent layers:

The first lives **inside** the Django command: extend the existing staff-notification
helper, `campaign_views.py:326-343`'s `_notify_staff()` — correcting the record where
31-CONTEXT.md's canonical_refs section described it as a "`mail_admins()`-based
staff-notification idiom": it does not call `mail_admins()` at all (zero matches anywhere in
this codebase, confirmed by RESEARCH.md's own grep this milestone). The real mechanism emails
every `is_staff=True` user with a non-blank email on file via `django.core.mail.send_mail()`.
The concrete adaptation Phase 34 needs: `_notify_staff()` builds its approval-queue link via
`self.request.build_absolute_uri(...)`, but a management command has no `request` object —
so the extension must either build the link another way (e.g. a hardcoded or
settings-derived base URL) or drop the link entirely for a scheduler-context failure
notification, rather than assuming `self.request` is available.

The second layer lives **outside** the process entirely: an external dead-man's-switch (e.g.
healthchecks.io, pinged at `hc-ping.com`) that fires when an expected ping fails to arrive —
the only mechanism that can catch the scheduler itself never invoking the command at all,
since anything living inside the process cannot observe its own non-execution. Task 1's
transcript recorded HTTP status **301** from this host to `hc-ping.com` — proof that outbound
egress to a heartbeat service works from the interim host today. As stated in task 1's
findings, this resolves only the technical half of the question; whether pinging a
third-party service is acceptable on policy/compliance grounds remains **unresolved** and is
routed to the operator, not assumed either way.

**Empty-input question, answered directly:** the heartbeat must fire even on a run that found
nothing to do — a zero-work sweep and a dead scheduler are otherwise indistinguishable from
the outside. The heartbeat ping belongs on the command's normal successful-completion path,
not gated on "did this run do anything," so a quiet night still reports as alive.

**SCHED-10 obligation, recorded as a named Phase 34 requirement.** An exception's string form
from an HTTP client (e.g. `requests`) can carry the full request, including any query-string
credentials, verbatim into its `str()` representation. Any exception raised by an adapter
must therefore be **sanitised before it reaches a notification body** — whether the in-command
`_notify_staff()`-style email or any log line — rather than forwarded via a bare `str(exc)`.
This is recorded here as a named obligation against **SCHED-10** specifically, not fixed by
this phase (which ships no code), so Phase 34 cannot inherit it silently.

**Open item carried forward for Phase 34, not answered here:** when several scheduled
commands are due at the same minute, whether their relative invocation order needs
specifying depends on whether they contend for the same database write lock — the command
set is not final until Phase 34 defines it, so this plan records the question rather than
guessing an answer to it.

Tag: **Confirmed against real rows** for the flock version, the cron-entry guard ratio
(0/3), and the heartbeat status code (301) — all quoted from task 1's live transcript above.
Tag: **Constructed-input code-path check** for the invocation-shape recommendation itself (no
Phase 34 cron line has been written yet) and for the `_notify_staff()` adaptation guidance
(reasoned from reading the existing code, not from having built the extension). The container
and AWS scopes remain labelled **unconfirmed** per task 2's findings above — nothing in this
section upgrades either scope's status on the strength of the interim-host evidence.
