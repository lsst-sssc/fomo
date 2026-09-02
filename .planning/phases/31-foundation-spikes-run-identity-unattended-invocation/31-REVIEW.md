---
phase: 31-foundation-spikes-run-identity-unattended-invocation
reviewed: 2026-09-02T00:00:00Z
depth: deep
files_reviewed: 8
files_reviewed_list:
  - docs/design/design.rst
  - docs/design/run_identity_and_unattended_invocation_spike.rst
  - tmp/31_constraint_probe.py
  - tmp/31-constraint-probe.txt
  - tmp/31-container-probe.txt
  - tmp/31_dbsnapshot_probe.py
  - tmp/31-dbsnapshot.txt
  - tmp/31-host-probe.txt
findings:
  critical: 6
  warning: 13
  info: 7
  total: 26
status: issues_found
---

# Phase 31: Code Review Report

**Reviewed:** 2026-09-02
**Depth:** deep
**Files Reviewed:** 8
**Status:** issues_found

## Summary

This is an investigation-only spike. No application source under `solsys_code/` or `src/`
changed (confirmed). The reviewable surface is therefore: one published Sphinx page plus a
one-line toctree edit, two disposable probe scripts that write against a copy of the real
dev database, and four captured transcripts.

For a spike, the *evidence* and the *recommendation derived from it* are the deliverable, so
they were reviewed as production artifacts: an unfalsifiable assertion in a probe, or a
recommendation the evidence does not actually support, is a defect with the same downstream
cost as a logic bug — Phases 32 and 34 are explicitly stated to execute these decisions
without re-deriving them.

Key concerns, in order of severity:

1. A stray `</content>` tool-wrapper tag is **committed** at the end of the published Sphinx
   page and will render as literal text on the built docs site.
2. The disposable-copy guard in `tmp/31_constraint_probe.py` is tautological — it validates a
   value it assigned itself one line earlier — and does not resolve symlinks, check the file
   exists, or compare against the real database path. The script it protects performs
   `alter_field` / `add_field` / `add_constraint` schema surgery.
3. Two of the phase's headline recommendations (the classical `source_identifier` key, and
   the new partial unique constraint the doc calls "proven, not just argued") are **not**
   supported by the transcripts submitted as their evidence.
4. The unattended-invocation decision asserts that `flock -n` on a local lock file transfers
   unchanged to Kubernetes. It does not — a container-local file lock gives no mutual
   exclusion across pods.

The two `.rst` files are otherwise structurally valid RST (verified by a docutils parse; the
only `toctree` "unknown directive" error is the expected Sphinx-only directive), the toctree
addition resolves to a real file, and the phase numbering (32/34) matches `ROADMAP.md`.

**Explicit answers to the three questions raised in the review brief:**

- **(1) The disposable-copy guard is not correct.** See CR-02 — it is bypassable and gives a
  false sense of safety. It happens to have worked on the one recorded run (the transcript's
  pk/count arithmetic independently confirms it hit a real 49-row copy), but that is luck of
  invocation, not the guard.
- **(2) No unredacted credential, API key, or long hex secret was found** in either
  transcript. A targeted scan for `password|secret|token|api[_-]?key|BEGIN .*PRIVATE` and for
  24+ character opaque strings returned only field names, constraint names and image tags.
  The captured crontab contains no environment assignment lines at all. However, the
  transcripts do disclose non-secret infrastructure detail that is now committed in a public
  repo (WR-08), and they carry no marker proving redaction occurred versus nothing existing
  to redact (WR-09).
- **(3) RST correctness is fine; internal consistency is not.** Syntax parses clean apart from
  the committed stray tag (CR-01). Several statements in the published page contradict its own
  evidence, its own companion document, or this repository (CR-04, CR-05, CR-06, WR-01 through
  WR-04).

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Stray `</content>` tool-wrapper tag committed into the published Sphinx page

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:198`
**Issue:** The last line of the committed file is a literal `</content>` tag — an artifact of
the tool wrapper used to author it, not RST. Verified on disk (`tail -c 40` shows
`...(Phase 34).\n</content>\n`) and in git (`git show HEAD:docs/design/run_identity_and_unattended_invocation_spike.rst | tail -1`).
A docutils parse emits `(WARNING/2) Bullet list ends without a blank line; unexpected
unindent` at exactly line 198, and Sphinx will render the tag as a visible paragraph at the
bottom of the published page. The pre-commit `sphinx-build` hook
(`.pre-commit-config.yaml:68-88`) does not pass `-W`, so a warning does not fail the gate —
which is why this shipped.
**Fix:** Delete the final line.

```diff
 * Whether several scheduled commands due at the same minute need a specified relative
   invocation order — depends on a command set this milestone has not finalised yet
   (Phase 34).
-</content>
```

Then re-run `pre-commit run sphinx-build --all-files`. Consider adding `-W` to the hook args
so a docutils warning fails the build rather than passing silently. See also IN-06 — the same
tag is present in `31-05-SUMMARY.md:244`.

### CR-02: Disposable-copy guard is tautological and bypassable; script performs destructive schema surgery

**File:** `tmp/31_constraint_probe.py:19-32`
**Issue:** The guard is:

```python
COPY_PATH = os.path.abspath('tmp/31-spike-db-copy.sqlite3')
conn.settings_dict['NAME'] = COPY_PATH
conn.connect()
resolved_name = conn.settings_dict['NAME']
if os.path.basename(resolved_name) != '31-spike-db-copy.sqlite3':
    raise SystemExit(1)
```

It reads back the value it assigned two statements earlier and checks only its **basename**.
The only way it can fail is if Django mutated the dict in between. It provides no protection
against the hazard it exists for. Four concrete gaps:

1. **No symlink/realpath resolution.** If `tmp/31-spike-db-copy.sqlite3` is a symlink or
   hardlink to `src/fomo_db.sqlite3`, the basename check passes and the script proceeds to
   `schema_editor.alter_field` on `CampaignRun.campaign` (line 58-64), `add_field`
   (line 160), `add_constraint` (line 167), and nine `objects.create()` calls — against the
   real dev database, out of band from Django migrations.
2. **No existence check.** SQLite creates the database file on connect. If the operator
   forgets to make the copy, the script prints `GUARD_DISPOSABLE_COPY=OK` and runs against a
   brand-new empty database.
3. **CWD-relative.** `os.path.abspath('tmp/...')` resolves against the process working
   directory. Invoked from anywhere but the repo root it silently targets a different path —
   and the basename check still passes.
4. **Never compared against the real path.** The one check that would actually be sound —
   `realpath(COPY_PATH) != realpath(original settings NAME)` — is not performed. Note also
   that `conn.settings_dict` in Django *is* the same dict object as
   `settings.DATABASES['default']`, so the original value must be captured before the
   assignment.

**Fix:**

```python
from django.conf import settings

REAL_DB = os.path.realpath(str(settings.DATABASES['default']['NAME']))
COPY_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp', '31-spike-db-copy.sqlite3')
)

if not os.path.isfile(COPY_PATH):
    raise SystemExit(f'ABORT: disposable copy does not exist at {COPY_PATH}; refusing to let sqlite create it.')
if COPY_PATH == REAL_DB:
    raise SystemExit(f'ABORT: copy path resolves to the real database ({REAL_DB}).')
if os.path.samefile(COPY_PATH, REAL_DB) if os.path.exists(REAL_DB) else False:
    raise SystemExit('ABORT: copy path is a link to the real database.')

conn = connections['default']
conn.close()
conn.settings_dict['NAME'] = COPY_PATH
conn.connect()
print(f'GUARD_DISPOSABLE_COPY=OK real={REAL_DB} copy={COPY_PATH}')
```

Printing both resolved paths also fixes the audit gap in WR-11.

### CR-03: Block B asserts nothing — both branches print `PASS`, including the one labelled "unexpectedly"

**File:** `tmp/31_constraint_probe.py:72-83`
**Issue:** The try/except around the second null-campaign insert prints `PASS:` in the success
branch (line 76-81) **and** `PASS:` in the `IntegrityError` branch (line 83) — where the
message itself reads "unexpectedly fired". This check cannot fail. It is the only evidence
gathered for Option A (the schema shape that was ultimately chosen), and every other block in
the same script correctly uses a `FAIL:` label for the unexpected branch (lines 100, 118, 143,
213, 226). Had the constraint behaved differently than the author assumed, the transcript
would have recorded it as a pass and the decision would have been made on a false reading.
The transcript's `PASS:` lines are quoted verbatim into `31-DECISION.md:191` and are the
phase's audit trail.
**Fix:**

```python
try:
    run_b2 = CampaignRun.objects.create(
        campaign=None, telescope_instrument=TELINST, window_start=WSTART, window_end=WEND
    )
    print(
        f'PASS: second null-campaign row created (pk={run_b2.pk}) -- unique_campaign_run_resolved_window '
        'does NOT discriminate when campaign IS NULL, as SQL NULL-inequality predicts.'
    )
except IntegrityError as exc:
    print(f'FAIL: unique_campaign_run_resolved_window fired for two null-campaign rows, contradicting the premise: {exc}')
```

### CR-04: Recommended classical `source_identifier` key cannot reproduce the tolerance match it claims to mirror, and was never tested with a real start time

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:80-87` (and
`:88-103`); evidence at `tmp/31_constraint_probe.py:173-175`, `tmp/31-constraint-probe.txt:27`
**Issue:** Two independent problems in the same recommendation:

1. **Semantic mismatch.** The doc specifies
   `` `CLASSICAL:{telescope}:{instrument}:{start-time}` ``, "matching the same three fields
   its existing tolerance match already uses", and row 4 of the table calls it the tolerance
   match's "``source_identifier`` mirror". It is not a mirror. The real matcher
   (`solsys_code/management/commands/load_telescope_runs.py:215` with
   `_START_TIME_MATCH_TOLERANCE = timedelta(minutes=5)` at line 22) matches a **datetime
   within ±5 minutes**. A string key is exact equality. `start_time` is a *computed* value
   (dark-window / sunset derived from `Observatory` coordinates), so any recomputation that
   shifts it by seconds — an edited site altitude, an ephemeris or library update — produces a
   different key. Find-or-create keyed on that string then creates a **duplicate row** on the
   next sync, which is precisely the failure `source_identifier` is being introduced to
   prevent. Conversely, truncating to a date cannot distinguish two runs on the same night.
2. **The probe never exercised it.** `classical_start = date(2026, 9, 5)` (line 174) is a
   `datetime.date`, and `.isoformat()` yields `2026-09-05` — a *date*, not the datetime the
   loader computes. The transcript records
   `source_identifier='CLASSICAL:SPIKE-NTT:SPIKE-EFOSC2:2026-09-05'`. So the recorded "round
   tripped cleanly" result was obtained with a value shape the adapter will never produce.

**Fix:** Before Phase 32 can implement this, the doc must state the canonical serialisation and
the quantisation rule, and the probe must be re-run with a real datetime. Suggested key:

```python
# Quantise to the same granularity the tolerance match implies, so a sub-tolerance
# drift maps to the same key rather than minting a new one.
bucket = start_time.replace(second=0, microsecond=0)
bucket = bucket.replace(minute=(bucket.minute // 5) * 5)
classical_identity = f'CLASSICAL:{telescope}:{instrument}:{bucket.isoformat()}'
```

Note a 5-minute bucket still splits a pair straddling a boundary; if that is unacceptable, the
honest answer is that `source_identifier` cannot replace the tolerance match for the classical
path and the doc should say so explicitly rather than implying equivalence.

### CR-05: The new partial unique constraint — the central SCHEMA-02 recommendation — is never negative-tested, yet the doc calls it "proven, not just argued"

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:70-78`; evidence at
`tmp/31_constraint_probe.py:156-204`
**Issue:** The doc states: *"A new ``source_identifier`` field ... with its own partial unique
constraint ... — proven, not just argued: after adding the field to a disposable copy of the
database, both existing constraints still refused a genuine duplicate exactly as before."*

What Block E actually proves is only the second clause. `spike_unique_source_identifier` is
added at line 162-167 and then **never tested**. The only writes against it use
`get_or_create` (lines 183 and 192), which by construction issues a `SELECT` first and never
attempts a duplicate `INSERT` — so the constraint is never given a chance to fire. Two
properties the whole recommendation rests on are unverified:

- that two rows with the *same* non-null `source_identifier` are rejected;
- that many rows with `source_identifier IS NULL` coexist (partially implied, since the
  constraint was added over 55 existing NULL rows without error, but never asserted).

The transcript reflects this: `tmp/31-constraint-probe.txt:24-33` contains no line naming
`spike_unique_source_identifier` in a firing assertion. Every `PASS` about a constraint firing
names one of the two pre-existing constraints.
**Fix:** Add negative controls before quoting the result as proven:

```python
# Positive control: duplicate non-null source_identifier must be rejected.
try:
    with transaction.atomic():
        CampaignRun.objects.create(
            source_identifier=lco_identity, campaign=sentinel,
            telescope_instrument='SPIKE-DUP/SPIKE-DUP', window_start=WSTART, window_end=WEND,
        )
    print('FAIL: spike_unique_source_identifier did NOT fire for a duplicate source_identifier.')
except IntegrityError as exc:
    print(f'PASS: spike_unique_source_identifier rejects a duplicate source_identifier: {exc}')

# Negative control: multiple NULLs must coexist under the partial constraint.
CampaignRun.objects.create(source_identifier=None, campaign=sentinel,
                           telescope_instrument='SPIKE-N1/SPIKE-N1', window_start=WSTART, window_end=WEND)
CampaignRun.objects.create(source_identifier=None, campaign=sentinel,
                           telescope_instrument='SPIKE-N2/SPIKE-N2', window_start=WSTART, window_end=WEND)
print('PASS: two NULL source_identifier rows coexist under the partial constraint.')
```

Until then, soften the doc from "proven" to "the additive property is proven; the new
constraint's own behaviour is asserted from Django/SQLite semantics, not measured."

### CR-06: `flock -n` on a local lock file is asserted to transfer unchanged to Kubernetes — it gives no cross-pod mutual exclusion

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:124-143`
**Issue:** The invocation decision states: *"Cron plus a per-command file lock (``flock``),
running inside the FOMO container — the same mechanism on today's interim host and on the
eventual AWS Kubernetes deployment, so migrating between them needs no scheduling redesign."*
The overlap-prevention row then commits Phase 34 to `flock -n` against a per-command lock file.

A `flock` lock lives on a single filesystem in a single container. On Kubernetes, two pods of
the same workload (a rolling update overlapping old and new, a restarted pod, a CronJob with
`concurrencyPolicy: Allow`, or any replica count above one) each get their own writable layer
and therefore their own lock file. Two concurrent runs of the same management command will
both acquire "the" lock and both proceed. The stated benefit — that the mechanism migrates
with no scheduling redesign — is the opposite of true: on Kubernetes the idiomatic and correct
construct is a `CronJob` with `concurrencyPolicy: Forbid`, or a shared/distributed lock
(database row lock, ReadWriteMany volume, lease). Neither this page nor `31-DECISION.md`
records this anywhere; a grep of `31-DECISION.md` for `cronjob|concurrencyPolicy|replica|pod|
distributed lock` returns nothing relevant. Phase 34 is stated to be "built directly against
the invocation mechanism this spike verifies", so this propagates as-is.
**Fix:** Split the decision by target instead of claiming one mechanism covers both:

> **Interim host:** cron + `flock -n` per command name. Verified.
> **AWS Kubernetes:** a `CronJob` per command with `concurrencyPolicy: Forbid` and
> `startingDeadlineSeconds` set. `flock` is retained only as an in-container belt-and-braces
> guard against two processes inside the *same* pod; it provides no protection across pods, so
> it cannot be the migration story on its own. Carry this as an open item for whoever owns the
> AWS deployment.

## Warnings

### WR-01: Missed-invocation layer 1 is unimplementable as written, and the helper it names swallows its own failures

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:152-162`
**Issue:** The decision names *"an in-command extension of this project's existing
staff-notification email"*. That helper is `campaign_views.py:326` `_notify_staff()`, and two
properties make it unusable as written for a scheduled command:

1. It is a **view method** that builds its link with `self.request.build_absolute_uri(...)`
   (`campaign_views.py:337`). A management command has no `request`. `31-DECISION.md:906-911`
   records this adaptation requirement explicitly — the published RST drops it entirely.
2. It calls `send_mail(..., fail_silently=True)` (`campaign_views.py:345`). Building a
   *failure-alerting* layer on a helper that discards its own delivery errors means the alert
   layer fails silently — the exact failure mode the two-layer design exists to eliminate.
   Neither document mentions `fail_silently` at all.

The doc's own preamble at lines 107-114 claims the mechanism was corrected "from this phase's
own source-code reading", which raises the bar: the same reading passed over both properties.
**Fix:** Extend the decision cell to record what Phase 34 must actually build — extract
`_notify_staff()` into a request-free helper taking an explicit base URL (from settings), and
use `fail_silently=False` with a caught-and-logged exception on the scheduler path so a mail
outage is at least visible in the command's stderr and log.

### WR-02: "the real crontab already runs this project's own management commands today" is false

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:128-131` and `:137-142`
**Issue:** The captured crontab (`tmp/31-host-probe.txt:10-12`) shows three active `manage.py`
entries: `rundataquery 1`, `updatescout --skip-designations`, `updatescout --skip-reconcile`.
Neither command exists in this repository — `grep -rn "rundataquery\|updatescout" --include=*.py .`
returns nothing, and this app's commands are `backfill_lco_observation_records`,
`fetch_jplsbdb_objects`, `import_campaign_csv`, `load_telescope_runs`, `reconcile_campaign_runs`,
`repair_stale_campaign_run_sites`, `sync_gemini_observation_calendar`,
`sync_lco_observation_calendar`. Locating them on disk shows `updatescout` belongs to the
third-party `tom_jpl` plugin and `rundataquery` to `tom_dataservices`. They are also invoked
from a *different checkout* (`/home/tlister/git/fomo_fresh/manage.py`).
`31-DECISION.md:378-390` compounds this by calling the ratio "0 guarded / 3 total **FOMO
management-command** entries" and naming it "the strongest single piece of evidence" for the
cron+flock choice.

The underlying conclusion survives (a FOMO deployment's crontab does run Django management
commands unguarded), but the label does not, and the doc is the durable record Phase 34 reads.
**Fix:** Reword to what the transcript supports: *"a sibling FOMO checkout's crontab already
invokes Django management commands (from the `tom_jpl` and `tom_dataservices` plugins, not
FOMO's own) unguarded, 0 of 3."*

### WR-03: Nullable-FK cost is understated, and contradicted by the same document's Future scope

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:66-68` vs `:167-171`
**Issue:** The decision cell says the nullable shape *"costs a single field-level migration and
needs no backfill: 0 of the 49 real ``CampaignRun`` rows today have a null campaign, so the
change only ever affects rows a future adapter writes."* Sixty lines later the Future-scope
section points at *"the grep-derived inventory of every site that reads
``CampaignRun.campaign`` and would need a null guard under the chosen schema shape."* Those two
statements cannot both be true: an inventory of read sites needing null guards is exactly a
cost beyond "a single field-level migration", and it affects existing code, not only rows a
future adapter writes.
**Fix:** Amend the cell to *"costs a single field-level migration, no data backfill, and a null
guard at each existing `CampaignRun.campaign` read site (inventory in 31-DECISION.md)."*

### WR-04: The chosen option's own demonstrated downside is omitted from the decision table

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:57-69`; evidence at
`tmp/31-constraint-probe.txt:12`
**Issue:** Block B measured a real, load-bearing consequence of the nullable shape: with
`campaign IS NULL`, `unique_campaign_run_resolved_window` *"has silently stopped
discriminating for exactly the rows Option A gives identity to"* — two identical non-campaign
runs were both accepted. The decision cell lists the drawbacks of the two **rejected** options
at length and none of the chosen one's. A Phase 32 implementer reading only the published page
would not learn that the existing uniqueness guarantee is void for every row the new adapters
write, and that the *only* thing standing between them and duplicate rows is the new,
never-negative-tested `source_identifier` constraint (CR-05).
**Fix:** Add to the chosen-shape cell: *"Accepted cost: SQL NULL-inequality means
`unique_campaign_run_resolved_window` does not fire for null-campaign rows (measured, not
argued), so duplicate protection for adapter-written rows rests entirely on the
`source_identifier` constraint."*

### WR-05: `31_dbsnapshot_probe.py` module docstring's privacy claim is factually wrong

**File:** `tmp/31_dbsnapshot_probe.py:4-5` vs `:54-56`
**Issue:** The docstring states *"Never selects contact_person or contact_email values (only
counts touch CampaignRun rows)."* Line 55 executes
`values_list('telescope_instrument', 'contact_person')` and materialises every
`contact_person` value into a `Counter` key. The inline comment at lines 52-53 quietly narrows
the claim to *"values themselves are never printed"* — a different and much weaker guarantee.
A reviewer trusting the docstring would wrongly conclude PII never left the database.
**Fix:** Correct the docstring to the true guarantee, or make it true by hashing before
counting:

```python
tbd_tuples = Counter(
    (telinst, hashlib.sha256((contact or '').encode()).hexdigest()[:12])
    for telinst, contact in CampaignRun.objects.filter(window_start__isnull=True).values_list(
        'telescope_instrument', 'contact_person'
    )
)
```

### WR-06: Block B omits the `transaction.atomic()` wrapper every other block uses

**File:** `tmp/31_constraint_probe.py:72-83`
**Issue:** The module docstring (lines 6-9) states rollback exists *"only to protect the
connection from a poisoned transaction after an expected IntegrityError."* Blocks C, D and E
all wrap their expected-failure `create()` in `with transaction.atomic():` (lines 96, 110, 139,
209, 218). Block B does not. On the recorded run this was harmless because the insert
succeeded, but if the `IntegrityError` path had been taken, the connection would have been left
in a broken-transaction state and every subsequent block's result — the entire remainder of the
evidence — would have been unreliable rather than merely wrong. Combined with CR-03 (both
branches labelled `PASS`), a poisoned run could have produced a transcript that still read as
all-pass.
**Fix:** Wrap the create in `with transaction.atomic():` to match the other four blocks.

### WR-07: The four-collision figure is measured over campaign-bearing rows and extrapolated to non-campaign runs

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:59-64`; evidence at
`tmp/31_dbsnapshot_probe.py:44-50`, `tmp/31-dbsnapshot.txt:15`
**Issue:** The doc argues a shared sentinel *"collides: four real telescope/instrument/window
combinations already recur across different real campaigns today, so collapsing every
non-campaign run onto one shared campaign value would refuse a second row for at least those
four combinations."* The measurement (`DUP_TELINST_WINDOW_TUPLES_IGNORING_CAMPAIGN=4`) is taken
over the 49 existing rows, **all of which have real campaigns** (`NULL_CAMPAIGN_ROWS=0`). Those
rows would not move to a sentinel. The number is therefore an argument by analogy about the
likely recurrence rate of future non-campaign runs, not a measurement of the collision the
sentence describes. The conclusion is plausible; the phrasing overclaims what was measured.
**Fix:** Reword to *"four such combinations recur among today's 49 campaign-bearing rows, so
the recurrence rate for a single shared campaign value is demonstrably non-zero; a sentinel
would refuse the second write for any combination that recurs the same way."*

### WR-08: Transcripts disclose developer-host infrastructure detail that is now committed in a public repository

**File:** `tmp/31-host-probe.txt:2-16`, `tmp/31-container-probe.txt:4-16`
**Issue:** The captured transcripts include the developer hostname (`tlister-thinkmate`), exact
kernel patch level (`5.14.0-687.42.1.el9_8.x86_64`), init version, absolute home and virtualenv
paths, an internal container registry hostname (`docker.lco.global/neoexchange:test_new_pyslalib`),
and the developer's complete personal crontab including an unrelated project's paths and log
locations. The `tmp/` files themselves are gitignored (`.gitignore:145`), but these transcripts
are reproduced **verbatim** into `31-DECISION.md:356-366` and `:476-478`, which is committed —
and the origin remote is `https://github.com/lsst-sssc/fomo.git`. The stated capture-time
redaction requirement covered secret-shaped environment assignments only; nothing else was
filtered. None of it is a credential, so this is disclosure hygiene rather than a breach, but
the kernel patch level and internal registry name are the kind of detail that does not belong
in a public repo by accident.
**Fix:** In `31-DECISION.md`, redact the hostname, kernel build string and unrelated crontab
lines, keeping only what the evidence needs (`flock` present; N active `manage.py` entries;
0 guarded). Extend the capture-time redaction rule from "secret-shaped env assignments" to
"hostnames, kernel build strings, absolute home paths, internal registry hostnames, and cron
lines belonging to unrelated projects."

### WR-09: Transcripts record no commands and no redaction marker, so redaction is unauditable and the capture is unreproducible

**File:** `tmp/31-host-probe.txt:1-20`, `tmp/31-container-probe.txt:1-23`
**Issue:** Each transcript contains only section headers (`=== crontab ===`) and output. The
commands that produced them are not recorded, so nobody can reproduce the capture or tell what
filtering was applied on the way to the file. Critically, the crontab section shows no
environment-assignment lines **and** no redaction placeholder — so it is impossible to
determine whether the mandated redaction removed a line or whether none existed. For an
artifact whose entire value is being trusted verbatim evidence, "cannot tell whether redaction
happened" is a defect.
**Fix:** Emit the command with each header and an explicit redaction accounting, e.g.:

```text
=== crontab ($ crontab -l | sed -E 's/^([A-Z_]+)=.*/\1=[REDACTED]/') ===
...
crontab: 0 environment-assignment lines redacted
```

### WR-10: `source_identifier` mixes three cardinality granularities in one global uniqueness namespace, with no rule recorded

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:80-87`
**Issue:** The three per-ingest-path values are at different granularities: the LCO value is
per-**request** (`facility.get_observation_url(record.observation_id)` →
`https://observe.lco.global/requests/<id>`, confirmed at
`solsys_code/management/commands/sync_lco_observation_calendar.py:193`); the Gemini value is
per-**observation**; the classical value is per-**night**. All three share one global partial
unique constraint. That silently fixes a cardinality rule the doc never states: **one
`CampaignRun` per LCO request**. If Phase 32's adapter instead maps several requests on the
same telescope/night onto one run — which is what "run" means in the classical path — the
second request's write is rejected by the constraint. This is adjacent to, but distinct from,
the open item already listed at lines 180-182.
**Fix:** Add one sentence to the per-ingest-path cell stating the cardinality each key implies
(*"one CampaignRun per LCO request, per Gemini observation, and per classical
telescope/instrument/night"*), and note that if Phase 32 wants many-requests-to-one-run, the
LCO key must move up to the request-group URL.

### WR-11: The guard's `OK` line is printed before any evidence the database is the intended copy

**File:** `tmp/31_constraint_probe.py:28-32`
**Issue:** `RESOLVED_DB_NAME=` prints only the path string that was just assigned, then
`GUARD_DISPOSABLE_COPY=OK`. Because SQLite silently creates a missing file (CR-02 gap 2), a run
against a brand-new empty database emits the same two lines. Blocks C and D would still print
their `PASS` messages on an empty database. The only thing in the recorded transcript that
distinguishes a real-copy run from an empty-database run is incidental — the pk sequence
starting at 59 and `FINAL_CAMPAIGNRUN_COUNT=58`, which happen to reconcile with the 49 rows in
`tmp/31-dbsnapshot.txt` (49 + 9 created = 58). That reconciliation is load-bearing but
accidental.
**Fix:** Make it deliberate — print and assert the pre-existing row count before any write:

```python
baseline = CampaignRun.objects.count()
print(f'BASELINE_CAMPAIGNRUN_ROWS={baseline}')
if baseline == 0:
    raise SystemExit('ABORT: 0 pre-existing rows -- this is not a copy of the dev DB.')
```

### WR-12: Model class and global settings are mutated with no restore

**File:** `tmp/31_constraint_probe.py:24`, `:65`, `:159`
**Issue:** Three unrestored global mutations: `conn.settings_dict['NAME'] = COPY_PATH` (which,
because Django's `settings_dict` is the same object as `settings.DATABASES['default']`, silently
re-points the *global* database setting for the rest of the process);
`CampaignRun._meta.get_field('campaign').null = True`; and
`CampaignRun.add_to_class('source_identifier', ...)`, which permanently attaches a field to the
model class. None is undone, and there is no `try/finally`. In `manage.py shell < script` this
is bounded by process exit, but if the script is ever pasted into an interactive shell — the
usual next step when a probe misbehaves — every subsequent ORM statement in that session
operates against the disposable copy with a mutated model, which is exactly the confusion the
guard exists to prevent.
**Fix:** Wrap the body in `try/finally` restoring the original `NAME`, or add a prominent
docstring line: *"This process is permanently poisoned after it runs — never paste into an
interactive shell, always invoke as a one-shot `manage.py shell < ...`."*

### WR-13: Recommended field definition omits the width the probe used

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:71`
**Issue:** The doc specifies only *"a new ``source_identifier`` field (nullable
``CharField``)"*. The probe used `models.CharField(max_length=500, null=True, blank=True)`
(`tmp/31_constraint_probe.py:157`). `max_length` is mandatory on a Django `CharField` and is
baked into the migration, so Phase 32 must either re-derive it or guess. 500 is not obviously
right or wrong for an LCO portal URL, but it is the number that was actually validated.
**Fix:** Record it: *"nullable `CharField(max_length=500)` — the width validated by the probe;
sized for a full LCO portal request URL."*

## Info

### IN-01: Four lines exceed the project's 120-column limit

**File:** `tmp/31_constraint_probe.py:105` (121), `:120` (122), `:194` (130), `:199` (122)
**Issue:** `pyproject.toml` sets ruff's line length to 120. Because `tmp/` is gitignored
(`.gitignore:145`), `pre-commit run ruff --all-files` never sees these files, so the project's
own style gate is not applied to code that reads the production database.
**Fix:** Either format the probes with the pinned ruff before committing evidence, or note in
the header that `tmp/` probes are exempt from the style gate by design.

### IN-02: Read-only fingerprint has one-second granularity and no content check

**File:** `tmp/31_dbsnapshot_probe.py:20-21`, `:61-64`
**Issue:** The integrity proof is `f'{st.st_size} {int(st.st_mtime)}'`. Truncating mtime to
whole seconds means a write completing within the same second as the baseline `stat`, and
leaving the file the same size (an in-place page update, which SQLite does routinely), reports
`FINGERPRINT_UNCHANGED=PASS`. The read-only guarantee actually rests on the script issuing no
write-style ORM calls; the fingerprint is decoration.
**Fix:** Use `st_mtime_ns` and add a content hash for a real check:
`hashlib.sha256(open(db_path,'rb').read()).hexdigest()`.

### IN-03: A hardcoded literal is annotated as coming from the dev database

**File:** `tmp/31_constraint_probe.py:170`
**Issue:** `lco_identity = 'https://observe.lco.global/requests/4247146'  # real LCO portal
request url, this dev DB` — the value is typed in, never queried. The comment asserts a
provenance the script does not establish, and the transcript inherits the claim.
**Fix:** Query it, so the provenance is real:
`lco_identity = CalendarEvent.objects.filter(url__contains='/requests/').values_list('url', flat=True).first()`
(with a fallback literal if none exists), and print which branch was taken.

### IN-04: `field_names_in_constraints.update(fields)` is fragile against a `fields=None` constraint

**File:** `tmp/31_constraint_probe.py:45-49`
**Issue:** `fields = getattr(constraint, 'fields', ())` is safe only because `CheckConstraint`
has no `fields` attribute at all. If any constraint type ever exposes `fields = None`,
`set.update(None)` raises `TypeError` and the probe dies in Block A. The sibling script's own
output (`tmp/31-dbsnapshot.txt:22`) prints `fields=None` for the `CheckConstraint`, which makes
`None` look like a live shape to a future reader of these transcripts.
**Fix:** `fields = getattr(constraint, 'fields', None) or ()`.

### IN-05: RST omits the two guarded crontab entries its companion explicitly excludes

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:137-142`
**Issue:** The doc says the crontab *"runs three FOMO management commands with zero overlap
guards between them."* The transcript (`tmp/31-host-probe.txt:15-16`) also contains two
`flock -n` guarded entries for `scout-alert-bridge`, which `31-DECISION.md:387-392` deliberately
excludes from the ratio. An RST-only reader cannot reconstruct the count or see that the
mechanism being recommended is already in production use on the same host for another project —
which is arguably the strongest supporting evidence available and is currently invisible.
**Fix:** Add a parenthetical: *"(two further entries for an unrelated project on the same host
are already `flock -n` guarded, which is where the confidence in the mechanism comes from.)"*

### IN-06: Same stray `</content>` tag in the phase summary

**File:** `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-05-SUMMARY.md:244`
**Issue:** The tool-wrapper leak behind CR-01 also landed in the phase summary. Outside the
declared review scope, but the same root cause and worth cleaning in the same pass — a repo-wide
`grep -rn '</content>'` is the check.
**Fix:** Delete the line; add `</content>` / `</result>` to a pre-commit forbidden-string hook.

### IN-07: RST does not record that a `flock -n` skip is itself a silent failure

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst:136-143`
**Issue:** The overlap-prevention decision says a contended tick *"is skipped outright, not
queued and not blocked"*, without noting that `flock -n` exits non-zero and, unlogged, that skip
is invisible — a run that silently never happens. `31-DECISION.md:862-865` records this; the
published page does not, and the missed-invocation row two cells below never connects the two.
**Fix:** Append: *"a `flock -n` non-zero exit must be logged, otherwise a permanently contended
lock is indistinguishable from a healthy no-op."*

---

_Reviewed: 2026-09-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
