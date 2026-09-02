# Phase 31: Foundation Spikes — Run Identity & Unattended Invocation - Pattern Map

**Mapped:** 2026-09-01
**Files analyzed:** 4 (this phase ships no `solsys_code/` source edits — see CONTEXT.md
"Phase Boundary" / RESEARCH.md "Validation Architecture": investigation-only, no schema
migration, no adapter code)
**Analogs found:** 4 / 4

This phase's "files to create" are entirely planning/decision artifacts and disposable
investigation scripts, not application code. There is no controller/service/model file
being added or modified. The table below classifies each deliverable against the two
closest prior investigation-only phases (18 and 26), per CONTEXT.md's explicit precedent
instruction.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md` | config (decision doc, not code) | batch (evidence accumulated across plans, written once per finding) | `.planning/milestones/v2.2-phases/26-canonical-record-spike/26-DECISION.md` (secondary: `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`) | exact — same artifact type, same authoring project, more directly comparable scope (two independent tracks in one phase, same as 26's multi-plan build-up) |
| `docs/design/run_identity_and_unattended_invocation_spike.rst` (suggested filename, per RESEARCH.md's Architecture Patterns section) | config (durable durable-summary doc) | batch | `docs/design/canonical_record_spike.rst` (secondary: `docs/design/uncertain_scheduling_spike.rst`) | exact — same doc family, same authoring team, same "Background / Key finding / Decisions / Future scope" section shape required by RESEARCH.md |
| `tmp/31_constraint_probe.py` (disposable, git-excluded — confirmed via `tmp/` in `.gitignore:145`) | test / utility (throwaway investigation script, never committed) | transform (writes real rows into a **disposable copy** of the dev DB, asserts constraint behavior, prints PASS/FAIL) | `tmp/26_integrity_check.py` referenced verbatim in `26-DECISION.md` (not git-tracked — it is itself a disposable script under the gitignored `tmp/` directory, exactly like this phase's own probe; there is no tracked source to point to, which is the expected/correct state for this artifact type) | role-match — same throwaway-probe pattern, but the analog script itself is untracked by design; extract its *documented* structure/output-format from `26-DECISION.md`'s Findings section (quoted below) rather than its file contents |
| Container-level shell verification (flock/network reachability, Pitfall 5) — not a file, a recorded terminal transcript inside `31-DECISION.md` | utility (verification transcript, no file artifact) | request-response (shell command -> stdout, quoted verbatim) | `18-DECISION.md`'s live probe transcripts (`crontab -l`, `flock --version`, `curl` checks — quoted verbatim in RESEARCH.md's own "Standard Stack" table, sourced from this same investigation style) | role-match — same "run it for real, quote verbatim" evidentiary discipline as both precedents |

## Pattern Assignments

### `.planning/phases/31-.../31-DECISION.md` (decision doc)

**Analog:** `26-DECISION.md` (header/status pattern) + `18-DECISION.md` (single-topic
sub-shape, useful for this phase's SCHED-07 track which is more self-contained than the
26 precedent's five-plan sprawl)

**Header pattern** (`26-DECISION.md:1-30`):
```markdown
# Phase 26: Canonical-Record Spike - Decision

**Investigated:** 2026-07-27
**Status:** Complete. This document was built up across all five plans of Phase 26.
Plan 26-01 records the D-04 dated snapshot, D-16, SPIKE-02's four-adapter identity
mapping, ...

This phase is **investigation-only**, following the Phase 13 (ESO) and Phase 18
(uncertain-scheduling) precedents. No `CampaignRun` schema migration, no
`campaign_reconciler.py` module, and no attribution UI ships from this phase. Every
`solsys_code/` edit this plan makes lives on a dedicated scratch git branch
(`spike/26-canonical-record-probe`) that is never merged and is deleted in plan 26-03;
the scratch DB copy (`tmp/26-spike-db-copy.sqlite3`) and `local_settings.py` are
git-excluded and removed at the same point. The sole committed artifact of this plan is
this file.

**Evidence posture, stated explicitly because it differs from Phase 18's:** Phase 18
wrapped every write in a rolled-back `transaction.atomic()` block because it ran against
the live `Observatory` table. Phase 26 instead writes for real against a disposable file
copy of the dev DB ... there is no rollback anywhere in this procedure, because the whole
file is throwaway.
```
Apply this shape to `31-DECISION.md`: state up front that no `CampaignRun` migration and
no adapter code ships from this phase; state the evidence posture per track (the schema
track's constraint probe should follow Phase 26's "disposable file copy, no rollback"
posture since it also targets a copy of `src/fomo_db.sqlite3`, not the live `Observatory`
table Phase 18 touched).

**Per-finding evidence pattern, with confidence tagging** (`26-DECISION.md:366-434`,
the `SPIKE-01` section — the single closest-matching subsection to this phase's own
SCHEMA-02 constraint-coexistence question):
```markdown
### SPIKE-01 criterion 1 — source vocabulary and constraint coexistence

Executed via `tmp/26_integrity_check.py` (`python manage.py shell < tmp/26_integrity_check.py`,
captured verbatim to `tmp/26-integrity-check.txt`) against the migrated scratch copy, after
confirming the DB-path guard. Four blocks, five PASS lines, zero FAIL lines:

\`\`\`
=== Block (A): constraint inventory ===
  UniqueConstraint name='unique_campaign_run_resolved_window' fields=('campaign', 'telescope_instrument', 'window_start', 'window_end') condition=<Q: (AND: ('window_start__isnull', False))>
  ...
PASS: neither source nor telescope_class appears in any CampaignRun constraint field set.
\`\`\`

Tag: **Confirmed against real rows** (real `CampaignRun` pk=1, its real 11 LCO-sourced
companion rows, and the real TBD-window run pk=4 used for the second negative control).
`src/fomo_db.sqlite3`'s fingerprint (`946176 1785094461`) was unchanged throughout this task.
```
This is the exact structure Phase 31's `31-DECISION.md` should copy for SCHEMA-02's
`source_identifier`-vs-existing-constraints coexistence check: dated dev-DB fingerprint
before/after, verbatim script output blocks, explicit PASS/FAIL lines, and a confidence
tag drawn from the same fixed vocabulary (**Confirmed against real rows** /
**Constructed-input code-path check** — see RESEARCH.md's own citation of this vocabulary
in its Validation Architecture section).

**Fingerprint-before/after discipline** (`26-DECISION.md:34-40`):
```markdown
Read-only probe (`tmp/26_snapshot_probe.py`, never selects `contact_person`/
`contact_email`, never calls a write-style ORM method) run against the real, unmodified
`src/fomo_db.sqlite3` via `./manage.py shell -c "exec(open(...).read())"`. The real DB's
`stat -c '%s %Y'` fingerprint (`946176 1785094461`) was recorded before this probe ran
and is identical afterward — this snapshot made zero writes.
```
Apply verbatim for D-08's real-dev-DB requirement: any read-only probe against
`src/fomo_db.sqlite3` (e.g. the "0/49 null-campaign rows" count) must record
`stat -c '%s %Y'` before and after and state explicitly that it is unchanged.

**Open-question / honest-gap framing** (`18-DECISION.md`'s Future-scope-style honesty,
applied via `26-DECISION.md`'s D-16 section, `26-DECISION.md:62-72`):
```markdown
### D-16 — PROJECT.md's Phase 25 claim does not reproduce

As of 2026-07-27, the maximum `CampaignRun` pk is 31 ... PROJECT.md's "Current Milestone"
section states `CampaignRun` pk=34 ... this claim does not reproduce against the live dev
DB ...
```
Apply this same "state the gap explicitly rather than papering over it" discipline for
SCHEMA-03's classical-schedule-file gap: RESEARCH.md's Open Question 1 already establishes
that no real classical schedule file exists in this repo — `31-DECISION.md` must record
this as an explicit, dated "0 real files available" finding (mirroring 26-DECISION.md's
own "0 real `GEM:`/`CAMPAIGN:` rows" honesty for SPIKE-02's Gemini/campaign-projection
subsections, `26-DECISION.md:106-129`) rather than reasoning from the three
`docs/design/telescope_runs_calendar.rst` example lines as if they were representative.

---

### `docs/design/run_identity_and_unattended_invocation_spike.rst` (durable summary)

**Analog:** `docs/design/canonical_record_spike.rst` (primary structural template — most
recent, two-track-in-one-phase precedent already anticipated by RESEARCH.md's own
Architecture Patterns section) and `docs/design/uncertain_scheduling_spike.rst`
(secondary, for its more compact single-topic "Decisions" list-table shape)

**Overall section shape** (`canonical_record_spike.rst:1-36`):
```rst
Canonical-Record Spike
=======================

This document records the investigation spike that settled how FOMO's calendar and
``CampaignRun`` model connect to each other for v2.2's canonical-record and reconciler
work. It was written after a live investigation (2026-07-27) that read the real dev
database, applied a throwaway migration to a disposable copy of it, and measured the
actual behaviour of the existing calendar-sync code rather than reasoning from
documentation alone. No ``CampaignRun`` schema migration, no reconciler module, and no
attribution UI was built during this spike — the deliverable is this durable summary and
its full-detail companion, ``26-DECISION.md`` (originally at
``.planning/phases/26-canonical-record-spike/26-DECISION.md``; this project's
milestone-archival workflow moves completed phase directories to
``.planning/phases-archive/`` once their milestone closes, so check there first if the
original path no longer resolves).

Background
----------

Today, a calendar event created by any of FOMO's sync commands ... has no stored link
back to the ``CampaignRun`` that actually produced it ... this spike settled four concrete
questions against the real dev database so none of that downstream work has to re-derive
them from scratch:

* How to record which pipeline created a ``CampaignRun`` ...
* ...
```
Apply this shape for the Phase 31 page: opening paragraph naming the investigation date,
what was and was not built, and a pointer back to `31-DECISION.md` with the same
archival-path caveat ("originally at `.planning/phases/31-.../31-DECISION.md`; check
`.planning/phases-archive/` once the milestone closes"). Background section: bulleted list
of the four concrete questions this phase answers (SCHEMA-01/02/03, SCHED-07), matching
the four-bullet pattern above.

**"Decisions" list-table pattern** (`canonical_record_spike.rst:99-131`):
```rst
.. list-table::
   :header-rows: 1
   :widths: 22 50 12

   * - Topic
     - Decision
     - Phase
   * - ``source`` vocabulary
     - Six values: the five ways a run can be created ... The three adapter-produced
       values are declared now but not written by any code path yet — that starts with a
       later milestone's adapter rewiring.
     - 27
```
Apply this per-topic three-column table shape (Topic / Decision / Phase) for both of
Phase 31's tracks: one table for the schema-shape verdict (SCHEMA-01/02/03), one for the
scheduling verdict (SCHED-07) — matching RESEARCH.md's own suggested structure ("two
`Decisions` tables, since Success Criterion 5 asks for 'a `docs/design/` page' — singular
— carrying both verdicts").

**Domain-correction / dated-addendum pattern** (`canonical_record_spike.rst:38-83`):
```rst
Domain correction: queue windows are not sets of owned nights
---------------------------------------------------------------

**Recorded 2026-07-27, after this spike's measurements were taken, from the project
owner (a professional astronomer) — read this before the Decisions section below, which
it qualifies.**
```
Reusable if Phase 31's investigation surfaces a similar late-breaking correction (e.g. if
the operator later clarifies something about the classical schedule-file format after
initial findings are recorded) — bold-dated attribution line, explicit "this qualifies the
section below" framing, never silently rewriting an earlier finding in place.

**Compact single-topic table variant** (`uncertain_scheduling_spike.rst:56-99`) — useful
if the scheduling track ends up simpler than the schema track and doesn't need the
two-table split:
```rst
.. list-table::
   :header-rows: 1
   :widths: 22 48 12

   * - SCHED-01 criterion
     - Decision
     - Phase
   * - Window field schema
     - Nullable ``window_start``/``window_end`` ``DateField`` pair, confirmed
       against real single-date, ranged, and TBD cell shapes.
     - 19
```

**"Future scope" closing pattern** (`canonical_record_spike.rst:308-330`,
`uncertain_scheduling_spike.rst:101-111`):
```rst
Future scope
------------

See ``26-DECISION.md`` (path note above) for the full evidence each of these decisions
rests on — including the real constraint-coexistence test results, the per-adapter
key-construction code citations, ... These are recommendations for Phases 27-29 to
implement, plus **one** still-open question for Phase 29 to settle from the evidence
above ... none of it is implemented in this spike.
```
Apply verbatim shape: point to `31-DECISION.md` for full evidence, explicitly name which
findings are still open for Phase 32/34 to settle (e.g. SCHEMA-03's classical
proposal-code question if no real schedule file is obtained in time), and restate "none of
it is implemented in this spike."

---

### `tmp/31_constraint_probe.py` (disposable probe script, never committed)

**Analog:** `tmp/26_integrity_check.py` (untracked, but its structure and invocation are
fully documented in `26-DECISION.md:366-434`, quoted above) — this is the correct
reference because the analog file itself is intentionally git-excluded, exactly as this
phase's own probe must be; do not attempt to point the planner at a tracked path for it.

**Invocation pattern** (documented in `26-DECISION.md:368`):
```
python manage.py shell < tmp/26_integrity_check.py
```
captured verbatim to a companion `tmp/*-check.txt` transcript file (also git-excluded).

**Structural pattern** (four lettered blocks, inferred from `26-DECISION.md`'s quoted
output and its own description "Four blocks, five PASS lines, zero FAIL lines"):
1. Block (A): constraint inventory — print every relevant `Meta.constraints` entry via
   `Model._meta.constraints`, confirm the new candidate field (`source_identifier` or
   equivalent) does **not** appear in any existing constraint's field tuple, unless the
   design intentionally folds it in (Pitfall 2 in RESEARCH.md explicitly requires deciding
   this, not assuming it).
2. Block (B): positive case — write a real row exercising the new field, confirm no
   `IntegrityError`.
3. Block (C)/(D): negative controls — attempt a genuine duplicate insert that **should**
   still collide on each existing partial `UniqueConstraint`, wrapped in
   `transaction.atomic()` solely to prevent a poisoned connection after the expected
   failure (never to roll back a real write):
```python
# Pattern, per 26-DECISION.md:387-392 ("Block (C)"/"Block (D)")
try:
    with transaction.atomic():
        CampaignRun.objects.create(**duplicate_kwargs)
except IntegrityError as exc:
    print(f'PASS: unique_campaign_run_resolved_window still fires unmodified: {exc}')
```

**Disposable-copy discipline** (`26-DECISION.md:24-30`, restated at `366-368`):
```markdown
Phase 26 instead writes for real against a disposable file copy of the dev DB
(`tmp/26-spike-db-copy.sqlite3`) — there is no rollback anywhere in this procedure,
because the whole file is throwaway.
```
Apply the same for Phase 31: `cp src/fomo_db.sqlite3 tmp/31-spike-db-copy.sqlite3`, point
`local_settings.py` (itself git-excluded) at the copy, run the probe for real, never touch
the live file for anything but read-only fingerprinted snapshots.

---

## Shared Patterns

### Confidence-tagging vocabulary (applies to every finding in `31-DECISION.md`)
**Source:** `26-DECISION.md` (used throughout — e.g. lines 94, 113-115, 199-202, 432-434)
**Apply to:** Every dated finding in both tracks of `31-DECISION.md`
```markdown
Tag: **Confirmed against real rows** (...)
Tag: **Constructed-input code-path check** (...)
```
Two fixed tags: **Confirmed against real rows** (a real, unmodified — or disposably-copied
— DB row was read or written) vs. **Constructed-input code-path check** (reasoned from
source code / a synthetic input because no real row exists, e.g. Phase 26's 0-real-`GEM:`-
row Gemini finding). Phase 31's classical-schedule-file gap (Open Question 1) and the
container-level flock/network check (Pitfall 5, not yet run this session) are both
currently **unconfirmed** and must be labeled as such explicitly, never silently upgraded
to "Confirmed" without the actual container-level or operator-supplied-file evidence.

### DB-fingerprint-before/after discipline
**Source:** `26-DECISION.md:34-40`, restated at 361-364, 584-585
**Apply to:** Any step in the schema track that touches `src/fomo_db.sqlite3` (even
read-only)
```
`stat -c '%s %Y'` recorded before and after; must be identical for any read-only probe.
```

### Existing constraints the new identity field must not collide with (verbatim, current)
**Source:** `solsys_code/models.py:288-303` (`CampaignRun.Meta.constraints`, read this
session — confirmed unchanged since Phase 26/27)
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
**Apply to:** SCHEMA-02's `source_identifier`-vs-constraint coexistence check — the probe
script must print these two constraints' field tuples verbatim (Block A) before asserting
the new field is absent from (or deliberately present in) either.

### Credential-handling pattern to extend (informs SCHED-07/D-04 findings only — not code
this phase writes)
**Source:** `src/fomo/settings.py:309-312` (`FINK_CREDENTIAL_*`, read this session)
```python
'URL': os.getenv('FINK_CREDENTIAL_URL', 'set FINK_CREDENTIAL_URL value in environment'),
'USERNAME': os.getenv('FINK_CREDENTIAL_USERNAME', 'set FINK_CREDENTIAL_USERNAME value in environment'),
```
**Apply to:** `31-DECISION.md`'s SCHED-07/D-04 findings section should cite this exact
pattern as the one D-04 extends for LCO/Gemini credentials — note it in the decision doc,
do not write new settings.py code this phase.

### Staff-notification idiom to note as Phase 34's intended consumer (informational only)
**Source:** `solsys_code/campaign_views.py:326-345` (`_notify_staff`, read this session)
```python
def _notify_staff(self, run):
    recipients = list(User.objects.filter(is_staff=True).exclude(email='').values_list('email', flat=True))
    if not recipients:
        return
    queue_url = self.request.build_absolute_uri(reverse('campaigns:approval_queue'))
    send_mail(
        subject='FOMO: new campaign run submission pending review',
        message=f'A new run submission is pending review: {queue_url}',
        from_email=None,
        recipient_list=recipients,
        fail_silently=True,
    )
```
**Apply to:** `31-DECISION.md`'s SCHED-07 findings should note this as the reusable
mechanism Phase 34's scheduler entry point will extend for in-command failure
notification — and must record the correction RESEARCH.md already made: this uses
`send_mail()` to staff users, not Django's `mail_admins()` (zero matches for
`mail_admins` anywhere in the codebase). Not implemented by this phase.

## No Analog Found

None — every deliverable this phase produces has a directly applicable structural analog
in Phase 18 or Phase 26's artifacts, per CONTEXT.md's explicit precedent instruction. The
one item with no local precedent at all is the **content** of the schema-shape decision
itself (RESEARCH.md: "None of the three candidate schema shapes ... has any precedent
anywhere else in this codebase — there is no existing 'default row' or 'sentinel
campaign' pattern to copy") — this is a first-principles design decision, not a
pattern-copying exercise, and `31-DECISION.md` should say so explicitly rather than
implying a local precedent exists for the schema shape itself (only for the *investigation
process and document structure* do precedents exist).

## Metadata

**Analog search scope:** `.planning/milestones/v2.1-phases/18-*/`,
`.planning/milestones/v2.2-phases/26-*/`, `docs/design/*.rst`, `solsys_code/models.py`,
`solsys_code/campaign_views.py`, `src/fomo/settings.py`, `.gitignore` (`tmp/` exclusion
confirmed)
**Files scanned:** 8 (both DECISION.md precedents in full, both docs/design/*.rst
precedents in full, models.py CampaignRun Meta block, campaign_views._notify_staff,
settings.py FINK_CREDENTIAL_* block, .gitignore)
**Pattern extraction date:** 2026-09-01
**Tracked-source gate:** All four analog paths cited above (`26-DECISION.md`,
`18-DECISION.md`, `docs/design/canonical_record_spike.rst`,
`docs/design/uncertain_scheduling_spike.rst`) confirmed git-tracked via
`git ls-files`. `tmp/26_integrity_check.py` is correctly **not** cited as a path to copy
from — it is untracked by design (inside gitignored `tmp/`), and its structure was
extracted instead from its documented output in the tracked `26-DECISION.md`.
