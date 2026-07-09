# Phase 18: Uncertain-Scheduling Investigation Spike - Pattern Map

**Mapped:** 2026-07-09
**Files analyzed:** 3 (1 throwaway probe script, 1-2 decision docs, 1 read-only reference set)
**Analogs found:** 3 / 3

This is an investigation-only spike (mirrors Phase 13's ESO feasibility spike). No production
code is created or modified. The two files this phase actually produces are:

1. A throwaway, git-excluded probe script (working name `fuzzy_match_probe.py`, repo root) —
   analog: Phase 13's `eso_p2_probe.py`.
2. A committed decision doc, `18-DECISION.md` (phase directory) — analog: Phase 13's
   `13-DECISION.md`.

Both are read-only with respect to the application: the probe script only reads
`Observatory` rows via the Django ORM and the real 3I/ATLAS CSV from its local path; it makes
no DB writes and no application code changes. The decision doc is prose/Markdown.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|-----------------|----------------|
| `fuzzy_match_probe.py` (repo root, git-excluded, throwaway) | utility / investigation script | request-response (ad-hoc, run manually) | `eso_p2_probe.py` (repo root, git-excluded) | exact |
| `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` | doc (decision record) | transform (evidence → recommendation) | `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-DECISION.md` | exact |
| (read-only reference, not modified) `solsys_code/campaign_utils.py` | service (helper functions) | CRUD / transform | n/a — this *is* the code being exercised, not created | n/a |
| (read-only reference, not modified) `solsys_code/models.py:CampaignRun` | model | CRUD | n/a — read-only reference for the natural-key discussion | n/a |

## Pattern Assignments

### `fuzzy_match_probe.py` (utility, investigation script)

**Analog:** `eso_p2_probe.py` (repo root)

**Git-exclusion mechanism** — verified present today in `.git/info/exclude`:
```
eso_p2_probe.py
```
Add a new line for this phase's script:
```
fuzzy_match_probe.py
```
Use `.git/info/exclude`, **not** `.gitignore` — keeps the exclusion local/untracked, matching
the "throwaway, never staged" intent (per Phase 13 precedent and this phase's RESEARCH.md).

**Header/docstring pattern** (`eso_p2_probe.py` lines 1-45):
```python
#!/usr/bin/env python
"""Throwaway, READ-ONLY probe script for the real ESO Phase 2 API (Paranal/VLT
production), used to gather evidence for Phase 13 ("ESO Feasibility Spike",
13-01-PLAN.md Task 1) of the v1.7 ESO/VLT Calendar Sync milestone.

THIS FILE IS NOT A DELIVERABLE (CONTEXT.md D-09). It is registered in
`.git/info/exclude` and MUST NEVER be staged or committed. It exists purely
so the operator ... can run a read-only walk ... and copy the (redacted)
output into `.planning/phases/13-eso-feasibility-spike/13-DECISION.md`.

READ-ONLY ONLY, NO WRITES, EVER (D-08). ...

How to run
----------
1. Set credentials WITHOUT committing them, e.g.: ...
2. Run it in the project venv: ...
3. Copy the printed blocks you need for ESO-02 evidence. REDACT any
   credential-adjacent content ... before pasting anything into 13-DECISION.md
"""
```
Adapt directly for Phase 18: swap "ESO Phase 2 API" for "rapidfuzz vs. difflib fuzzy-match
comparison and `resolve_site()`/`parse_obs_window()` live exercise against the real 3I/ATLAS
CSV," swap the phase/plan references, and swap the redaction target from
"credential-adjacent content" to "Contact Person/Email" (per D-01).

**Credential/config-loading pattern** (env var with local fallback, `eso_p2_probe.py`
lines 49-79) — for Phase 18 the equivalent "external input" is the real CSV path (D-01,
`/mnt/c/Users/liste/OneDrive/Documents/Asteroids/3I/3I_ATLAS Observations and Observing
Plans - Sheet1.csv`) rather than credentials, but the same discipline applies: read it from
its fixed local path, never hard-code assumptions that it hasn't changed (D-02), never write
it back.

**Django ORM access inside a throwaway script** — per this phase's RESEARCH.md "Probe-script
harness shape" code example (verbatim recommended pattern, not yet on disk — write exactly
this shape):
```python
# fuzzy_match_probe.py -- throwaway, git-excluded per .git/info/exclude (Phase 13 convention)
# Run via: ./manage.py shell < fuzzy_match_probe.py  (needs Django ORM for Observatory)
import difflib
from rapidfuzz import fuzz, process

from solsys_code.solsys_code_observatory.models import Observatory

candidate_pool = list(
    Observatory.objects.values_list('obscode', 'name', 'short_name', 'old_names')
)
flat_candidates = sorted({s for row in candidate_pool for s in row if s})

D09_TEST_CODES = ['X09', 'N50', 'X07', 'C65']  # real messy Site Code values, per D-09

for raw in D09_TEST_CODES:
    rf_best = process.extractOne(raw, flat_candidates, scorer=fuzz.WRatio, score_cutoff=60)
    dl_best = difflib.get_close_matches(raw, flat_candidates, n=1, cutoff=0.6)
    print(f'{raw!r}: rapidfuzz={rf_best!r} difflib={dl_best!r}')
```
This is a good starting skeleton; the plan should extend it to also import and call
`resolve_site()`/`parse_obs_window()` directly (both already importable, no Django setup
issue) against real CSV rows read via the stdlib `csv` module (matching D-10's confirmed
embedded-newline-safe approach — do not hand-roll CSV parsing).

**Verification-gate pattern** (Phase 13's `13-01-PLAN.md`, referenced in this phase's
RESEARCH.md "Architecture Patterns" section) — adapt the same shape:
```bash
test -f fuzzy_match_probe.py && grep -q 'fuzzy_match_probe.py' .git/info/exclude && \
  python -c "import ast; ast.parse(open('fuzzy_match_probe.py').read())"
```

---

### `18-DECISION.md` (doc, decision record)

**Analog:** `13-DECISION.md`

**Header/status pattern** (`13-DECISION.md` lines 1-11):
```markdown
# Phase 13: ESO Feasibility Spike - Decision

**Investigated:** 2026-07-01
**Status:** Complete. Findings recorded (ESO-01/02/03) against real Paranal production
credentials; Recommendation (ESO-04: Bypass) and Future-sync sketch (ESO-05) completed in
Plan 02.

This phase is investigation-only. No `sync_eso_observation_calendar` command, no
`FACILITIES['ESO']` settings change, and no other application code is shipped from this
plan — the sole committed deliverable is this findings record, built from a throwaway,
git-excluded probe script (`eso_p2_probe.py`, D-09) run by the operator against real ESO
Phase 2 production credentials (D-01/D-03), never against a demo/sandbox environment.
```
Adapt for Phase 18: swap ESO-01/02/03 finding IDs for this phase's own findings, keyed to
SCHED-01's 5 criteria (window schema — already locked; TBD natural key; CSV range/TBD
parsing rules; fuzzy-match library choice; `resolve_site()` MPC-code confirmation). State
plainly that no `CampaignRun` migration, no CSV importer change, and no fuzzy-match UI code
ships this phase.

**Findings-with-verbatim-evidence pattern** (`13-DECISION.md` lines 13-116, `ESO-01`
section) — each finding is: a heading (`### ESO-01 — <short title>`), a plain-English
summary sentence, then verbatim command/output blocks in fenced code blocks, followed by an
explicit "what this confirms / doesn't confirm" paragraph distinguishing confirmed-fact from
still-open. Example structure to copy:
```markdown
### ESO-01 — Credential obtainability & usability

Valid ESO Phase 2 production credentials for Paranal (VLT) ARE obtainable and usable. ...

**La Silla stretch result (D-06):** ...

```
ESOAPI.__init__: Error creating API connections: (500, 'POST', 'production_lasilla', ...)
```

credential-adjacent fields redacted per D-04 (none were present in this diagnostic; ...).

Per D-06 this is a valid, documented finding, not a phase blocker: ...
```
For Phase 18, apply the equivalent redaction discipline to `Contact Person`/`Email` values
per D-01 — replace with `<REDACTED>` or omit, keep everything else (dates, site codes,
telescope/instrument names, real people's names describing a finding — explicitly allowed
unredacted per D-01) verbatim.

**Distinguishing confidence levels** — Phase 18's D-09 explicitly requires the decision doc
to separate "confirmed against a real row" (e.g. `resolve_site('250')` against Jewitt's/
Noonan's real Hubble rows) from "confirmed via constructed input only" (e.g.
`resolve_site('274')`, since no real row in the current snapshot types plain `274`). Mirror
`13-DECISION.md`'s ESO-01 pattern of explicitly flagging a "confirmed working" vs. a
"revised/reasoned finding" (the La Silla P2 bypass discussion) rather than blurring the two.

**Recommendation section pattern** — `13-DECISION.md` closes with an explicit
recommendation section (ESO-04, not shown above but present later in the file) stating the
chosen approach and rationale. Phase 18's doc should close similarly: state the chosen
fuzzy-match library (or "split verdict, use X for case A / Y for case B" per RESEARCH.md's
Open Questions §1) with the D-09 corpus scores as evidence, and the natural-key
recommendation (D-06's `contact_person`-in-key decision) restated as a locked finding for
Phase 19 to implement.

---

## Shared Patterns

### Redaction discipline (PII-gated real data)
**Source:** `13-DECISION.md` D-04 precedent (credential-adjacent redaction), extended per
Phase 18's own D-01 (`Contact Person`/`Email` redaction)
**Apply to:** `18-DECISION.md` — any verbatim cell text quoted from the real 3I/ATLAS CSV.
Replace `Contact Person`/`Email` values with `<REDACTED>`; real people's names used to
*describe* a finding (not quoted as a raw cell value) are fine unredacted per D-01.

### Throwaway/git-excluded script discipline
**Source:** `eso_p2_probe.py` + `.git/info/exclude` (existing line: `eso_p2_probe.py`)
**Apply to:** `fuzzy_match_probe.py` — add its filename as a new line in
`.git/info/exclude` (not `.gitignore`), never `git add` it, read-only against the DB (only
`Observatory.objects.values_list(...)`, no writes) and read-only against the real CSV
(never write it back).

### "Never raise on non-key/messy fields, flag instead" discipline
**Source:** `solsys_code/campaign_utils.py:resolve_site()` (lines 85-183) and
`parse_obs_window()` (lines 186-244) — both already implement "return a usable value plus
an explicit `needs_review`/`ut_needs_review` flag rather than raising" for exactly the kind
of messy real-world input this phase's decision doc documents (D-05's TBD/garbage-text UT
cells, D-07's blank site codes).
**Apply to:** The decision doc's *recommendation* text for Phase 20's future CSV
range/TBD-parsing extension — cite this existing discipline by name (and the three regex
names `_HHMM_RANGE`, `_APPROX_HOUR`, `_BARE_HOUR_UTC`) so Phase 20's planner doesn't have to
rediscover them.

## No Analog Found

None. Both files this phase produces have exact Phase 13 analogs; all "existing code" this
phase investigates (`campaign_utils.py`, `models.py:CampaignRun`,
`solsys_code_observatory/models.py:Observatory`) is read-only reference material, not a file
being created or modified, so no analog search was needed for it.

## Metadata

**Analog search scope:** `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/`,
repo root (`eso_p2_probe.py`, `.git/info/exclude`), `solsys_code/campaign_utils.py`,
`solsys_code/models.py`.
**Files scanned:** 6 (`13-DECISION.md`, `13-CONTEXT.md` [via RESEARCH.md citation],
`eso_p2_probe.py`, `.git/info/exclude`, `solsys_code/campaign_utils.py`,
`solsys_code/models.py`).
**Pattern extraction date:** 2026-07-09
