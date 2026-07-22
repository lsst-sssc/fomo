# Phase 18: Uncertain-Scheduling Investigation Spike - Research

**Researched:** 2026-07-08
**Domain:** Fuzzy string matching (library API comparison) + throwaway-investigation-script conventions
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Phase boundary:** Investigation-only, mirroring Phase 13's ESO feasibility spike. Settles
5 open design decisions — window field schema (already locked), the replacement natural key
for TBD rows, the CSV range/TBD text-parsing rules, the fuzzy-match library choice
(`rapidfuzz` vs. stdlib `difflib`), and whether `resolve_site()` correctly resolves real
space-observatory MPC codes — against the real 3I/ATLAS coordination sheet. Deliverable is a
decision doc (or docs); no `CampaignRun` schema migration, no CSV importer changes, no
fuzzy-match UI code ships this phase (that's Phases 19-21).

**Already locked (not open for this phase to reconsider):**
- Window schema is a nullable `window_start`/`window_end` `DateField` pair (not datetime,
  not a `django.contrib.postgres.fields.DateRangeField`) — per SCHED-01/PROJECT.md.
- `Observatory.obscode` widening is presumed unnecessary — real space-observatory MPC codes
  (250=Hubble, 274=JWST, 289=Nancy Grace Roman) are standard 3-char codes that already fit
  `CharField(max_length=4)`. The spike's job is to confirm this, default answer is "no
  widening needed," not to design a widening migration.

**Real-data access:**
- D-01: The real 3I/ATLAS sheet CSV export lives locally at
  `/mnt/c/Users/liste/OneDrive/Documents/Asteroids/3I/3I_ATLAS Observations and Observing
  Plans - Sheet1.csv`. Read it directly from that path during Plan 18 execution — do NOT
  copy it into the repo or `.planning/` (PII-gated). Verbatim cell text quoted into a
  committed decision doc must have `Contact Person`/`Email` redacted, matching Phase 13's
  D-04 precedent. Real people's names describing a finding are acceptable without redaction.
- D-02: This discussion already read the real CSV once (2026-07-08 snapshot). Findings
  D-03..D-09 are real, not constructed — Plan 18 execution should re-read the file directly
  (it's a live publicly-editable Google Sheet export) rather than trusting CONTEXT.md alone.

**CSV range/TBD parsing rules (SCHED-01 criterion 3):**
- D-03: Real `Obs. Date` column shapes beyond exact `YYYY-MM-DD`: blank entirely; `" to "`
  full-date range (`2025-07-05 to 2025-09-22`); compact same-month range
  (`2025-11-02 -25`); month-known-day-TBD marker (`2025-12-?`).
- D-04: A multi-day window is sometimes typed into `UT Time Range` instead of `Obs. Date`
  (e.g. `2025-11-27 to 2025-12-10` with `Obs. Date` = exact `2025-11-27`). A parser that
  only inspects `Obs. Date` for range syntax will silently miss these. Any range-detection
  logic built in Phase 20 must check BOTH columns.
- D-05: Real `UT Time Range` free-text shapes beyond already-handled cases: literal `TBD`
  text, blank, `~7:00:00 AM` (approx marker with seconds), and one row with a stray
  copy-paste artifact (genuine unparseable garbage). Confirms "never raise on this column,
  fall back and flag needs-review" is still the right posture, extended to range/TBD
  detection.
- D-06 (locks TBD-collision open item): Real collision — Belyakov's JWST/MIRI row and
  Cordiner's JWST/NIRSpec row are both `Telescope/Instrument = "JWST"`, both
  `Obs. Date = "2025-12-?"` (→ `window_start = window_end = NULL`). **Decision:** extend the
  TBD-row natural key with `contact_person` (already a `CampaignRun` field) so these two
  distinct rows don't collide under `(campaign, telescope_instrument, window_start)` alone.
  Phase 19 designs the exact mechanism (e.g. conditional/partial `UniqueConstraint`
  including `contact_person` only when `window_start IS NULL`); this phase records the
  decision and evidence.
- D-07: Some real rows have entirely blank `Site Code` for legitimate ground-network entries
  (`"LCO 1m"`, `"LCO 2m"`) and space missions with no MPC site concept (Swift, JUICE). Flag
  for Phase 20's ASSET-01/02 research: when site never resolves, there's no `Observatory` to
  read `observations_type` from. Not this phase's decision to solve.

**Fuzzy-match library & resolve_site() confirmation (SCHED-01 criteria 4-5):**
- D-08: Both should be live-tested, not reasoned from documentation alone. `rapidfuzz` is
  NOT added to `pyproject.toml` this phase — install temporarily/scratch-only for the
  comparison (like Phase 13's git-excluded `eso_p2_probe.py`).
- D-09: Real messy `Site Code` test corpus: `X09` (Sam Deen, "Deep Random Survey / 43cm"),
  `N50` (HCT), `X07` (Josep Trigo-Rodríguez, "Deep Sky Chile"), `C65` ("Telescope Joan Oró,
  Montsec, Catalonia"), and blank/missing codes (Swift, JUICE, "LCO 1m"/"LCO 2m" — not a
  fuzzy-match case). **Important:** the real JWST rows all use `Site Code = "500@-170"` (JPL
  Horizons/SPICE notation), never the correct standard MPC code `274` — this is exactly the
  over-length code `resolve_site()` already flags for manual review (confirms that behavior
  is correct and needed). `resolve_site('274')` works is a code-path check using constructed
  input (no real row types plain `274`); `resolve_site('250')` (Hubble) can be confirmed
  directly against real rows (Jewitt's, Noonan's). Document both as different confidence
  levels — don't conflate "confirmed against a real row" with "confirmed via constructed
  input." The `500@-170`-vs-`274` mismatch means a straight fuzzy-string match against
  `Observatory.name`/`short_name`/`old_names` won't bridge JPL/SPICE notation to an MPC
  code — a distinct future problem, not this phase's to solve.
- D-10: The real sheet's `Telescope/Instrument` column has at least one embedded-newline
  quoted CSV cell (`"Hubble\nWFC3/UVIS"`) — confirms the importer must keep using Python's
  `csv` module (already handles this correctly), not naive line-based parsing.

### Claude's Discretion
- Exact wording/structure of the decision doc(s) beyond D-01..D-10.
- Whether to produce a full-detail doc plus a durable summary (Phase 13's D-10 pattern) or
  a single doc — narrower scope than Phase 13's, a single doc may suffice.
- How exactly to redact quoted real-sheet examples while keeping them useful as evidence.
- Exact regex/parsing-rule design implementing D-03/D-04/D-05 shapes — this phase documents
  the shapes; designing the actual parsing rules is shared between this spike's decision
  doc (recommending an approach) and Phase 19/20's planning (implementing it).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. (D-07's "blank site code" finding is adjacent
to this phase's SCHED-01 scope but is explicitly logged as a note for Phase 20's ASSET-01/02
research, not solved here.)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SCHED-01 | A phase-time investigation spike settles the window field schema, the replacement natural key for TBD rows, the CSV range/TBD text-parsing rules, the fuzzy-match library choice (`rapidfuzz` vs. stdlib `difflib`), and confirms whether `resolve_site()` correctly resolves real space-observatory MPC codes — before implementation begins | This document supplies the verified API surface for both fuzzy-match candidates (function signatures, return shapes, parameters, known short-string gotchas) so the planner can write a concrete live-comparison task against the D-09 real corpus, plus the exact existing `resolve_site()`/`parse_obs_window()` signatures and the throwaway-script convention (`.git/info/exclude`) the probe script should follow. The window-schema, natural-key, and CSV-parsing-rule decisions are already fully evidenced in CONTEXT.md D-03..D-07/D-10 and are not re-researched here per the phase-specific research focus. |

</phase_requirements>

## Summary

This is an investigation-only spike phase, structurally identical to Phase 13's ESO
feasibility spike (which shipped no RESEARCH.md at all — CONTEXT.md's real-CSV findings
D-01..D-10 were sufficient grounding for planning). The one genuine research gap CONTEXT.md
leaves open is criterion 4: **the concrete, current API surface of the two fuzzy-match
library candidates**, needed so the planner can write a specific, executable live-comparison
task (not "compare the libraries" but "call `X` with these exact parameters against the D-09
corpus and record the scores").

Both candidates were verified this session: `rapidfuzz` 3.14.5 is current on PyPI (this dev
venv already has 3.14.3 installed, pulled in transitively as a dependency of `cleo`/`poetry`
— not a project dependency, and not reliable to assume present in any other environment).
stdlib `difflib` needs no installation and is guaranteed available in any Python 3.10+
environment already targeted by this project.

**Primary recommendation:** Write a throwaway, git-excluded probe script (following Phase
13's `eso_p2_probe.py` convention exactly — added to `.git/info/exclude`, never staged) that
runs both `rapidfuzz.process.extractOne(..., scorer=fuzz.WRatio)` and
`difflib.get_close_matches(...)` against the D-09 real `Site Code` corpus, using
`Observatory.name`/`short_name`/`old_names` as the candidate pool, and records verbatim
scores/matches as the decision doc's live-test evidence per D-08. `pip install rapidfuzz`
before running (already present in this venv, but the probe/decision doc should not assume
that's true elsewhere) — do not add it to `pyproject.toml`.

## Architectural Responsibility Map

This phase ships no application code — the map below describes the tier each *future*
capability (decided here, built in Phases 19-21) belongs to, so the decision doc's
recommendations point the right direction for downstream planning.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Window schema (`window_start`/`window_end`) | Database / Storage | — | Django model fields + migration on `CampaignRun`; a pure persistence-layer decision (Phase 19). |
| TBD natural-key constraint | Database / Storage | — | Conditional/partial `UniqueConstraint` on `CampaignRun.Meta` (Phase 19). |
| CSV range/TBD text parsing | API / Backend | — | Lives in `solsys_code/campaign_utils.py:parse_obs_window()` and the `import_campaign_csv` management command — server-side batch ingestion, no browser/CDN involvement (Phase 20). |
| Fuzzy site-name matching | API / Backend | Frontend Server (SSR) | Match computation happens server-side (`resolve_site()`/a new fuzzy-match helper in `campaign_utils.py`); the dropdown of candidates it feeds is rendered into the existing Django-templated approval-queue page (Phase 21). |
| `resolve_site()` MPC-code confirmation | API / Backend | — | Existing `solsys_code/campaign_utils.py:resolve_site()` — no new tier, this phase only validates current behavior (this phase, code-path check only). |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `rapidfuzz` | 3.14.5 (latest on PyPI; 3.14.3 already installed in this dev venv) [VERIFIED: pip index versions rapidfuzz / pypi.org] | C++-backed fuzzy string matching — `process.extractOne`/`fuzz.WRatio` etc. | MIT-licensed, ~40M downloads/week [VERIFIED: pypistats.org], drop-in successor to the GPL-licensed `fuzzywuzzy`/`thefuzz`. Purpose-built for exactly this "best match from a small candidate list" use case. |
| `difflib` | stdlib (bundled with every Python 3.10-3.12 this project targets) [VERIFIED: docs.python.org] | `get_close_matches()`/`SequenceMatcher` — Ratcliff-Obershelp sequence similarity | Zero install, zero new dependency surface. Already the kind of "boring stdlib tool" this codebase's `ruff`/ecosystem conventions favor when it's sufficient. |

### Supporting
None — this phase's scope is a live-test comparison, not new production code.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rapidfuzz` | `fuzzywuzzy`/`thefuzz` | GPL-licensed (thefuzz is MIT but wraps python-Levenshtein, a C extension with historically flaky wheel availability); rapidfuzz is the maintained, faster successor — not worth live-testing separately since it's the documented reason rapidfuzz exists [CITED: pypi.org/project/rapidfuzz]. |
| `difflib` | `Levenshtein`/`python-Levenshtein` (standalone) | Redundant with `rapidfuzz`, which already implements Levenshtein-family scorers internally (`fuzz.ratio` etc.) with a faster C++ core — no reason to add a third candidate to this phase's comparison. |

**Installation (scratch-only, per D-08 — NOT added to `pyproject.toml`):**
```bash
pip install rapidfuzz
```
`difflib` requires no installation (stdlib).

**Version verification:** `pip index versions rapidfuzz` returned `3.14.5` as latest, with
`3.14.3` already importable in this exact dev venv (`pip show rapidfuzz` →
`Location: /home/tlister/venv/fomo_venv/lib/python3.12/site-packages`,
`Required-by: cleo`). This is an *incidental* transitive dependency of Poetry's `cleo` CLI
package, not a project dependency — do not rely on it being present in CI or a fresh
`./.setup_dev.sh` environment; the probe script/decision doc should still explicitly
`pip install rapidfuzz` (or note the version used) rather than assuming it's already there.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `rapidfuzz` | PyPI | Actively maintained since ~2020, latest release 2026-04-07 [VERIFIED: pypi.org] | ~40M/week [VERIFIED: pypistats.org] | `github.com/rapidfuzz/RapidFuzz` (MIT license, author Max Bachmann) [VERIFIED: pypi.org] | `[SUS]` (automated gate) → manually cross-verified `[OK]` | Approved, scratch-only per D-08 |

**Automated gate detail:** `gsd-tools query package-legitimacy check --ecosystem pypi
rapidfuzz` returned verdict `SUS` with reason `unknown-downloads` — the checker's download
lookup didn't resolve a number, which is a **tooling/API-reachability limitation of the
check itself**, not a real signal about the package. Manual cross-verification via
`pypi.org/project/rapidfuzz/` and `pypistats.org/packages/rapidfuzz` (both fetched live this
session) shows ~40M weekly downloads, an active MIT-licensed GitHub repo, a well-known
maintainer, and the package is already transitively installed in this exact dev venv via
`cleo`/`poetry` — this is not a slopsquat or hallucinated package. Per protocol the planner
should still add a lightweight `checkpoint:human-verify` before the `pip install rapidfuzz`
step, but the evidence above should make that checkpoint trivial to clear (Tim can confirm
"yes, this is the real, well-known rapidfuzz" in seconds).

**Packages removed due to `[SLOP]` verdict:** none.
**Packages flagged as suspicious `[SUS]`:** `rapidfuzz` — automated-gate false positive per
above; planner should still gate the `pip install` behind `checkpoint:human-verify` per
protocol, with this section's evidence attached.

## Architecture Patterns

### Throwaway Investigation Script Convention (from Phase 13 precedent)

**What:** A single, self-contained, git-excluded Python script at the repo root, run
manually (not via a management command), that exercises a candidate library/API live and
prints/captures results for the decision doc.

**Established precedent (Phase 13):**
- File: `eso_p2_probe.py` at the repo root.
- Git exclusion: added as a bare filename to `.git/info/exclude` (already present today:
  `cat .git/info/exclude` shows `eso_p2_probe.py` as the last line) — **not** `.gitignore`
  (keeps the exclusion local/untracked, matching the "throwaway, never staged" intent).
  This phase's probe script should follow the identical mechanism — add its own filename
  (e.g. `fuzzy_match_probe.py`) as a new line in `.git/info/exclude`.
- Content discipline: read-only/side-effect-free, no writes to the DB beyond what's already
  safe (Phase 13's guardrail was "no ESO API writes"; this phase's equivalent is "read the
  real CSV read-only, never write it back, never commit it").
- Verification gate used in Phase 13's plan (for reference — same shape applies here):
  ```bash
  test -f fuzzy_match_probe.py && grep -q 'fuzzy_match_probe.py' .git/info/exclude && \
    python -c "import ast; ast.parse(open('fuzzy_match_probe.py').read())"
  ```
- Run manually: `python fuzzy_match_probe.py` (or via `./manage.py shell < ...` if Django
  ORM access to `Observatory` records is needed — likely required here, since the candidate
  pool is `Observatory.objects.values_list('name', 'short_name', 'old_names')`, not a static
  list).

**When to use:** Any phase-time investigation that needs to run real code against real data
(a live API, a live CSV, a live DB query) but produces no shippable application code — the
decision doc is the deliverable, not the script.

### Existing Code This Phase's Findings Extend

**`solsys_code/campaign_utils.py:resolve_site()`** (lines 85-183) — current signature:
```python
def resolve_site(site_code_raw: str, *, create_placeholder: bool = True) -> tuple[Observatory | None, bool]:
```
3-tier resolution: (1) exact `Observatory.objects.get(obscode=code)`, (2) live MPC Obscodes
API via `MPCObscodeFetcher`, (3) placeholder creation (skippable via `create_placeholder=False`).
A blank or over-`_MAX_OBSCODE_LEN` code (`Observatory._meta.get_field('obscode').max_length`,
currently `4`) never reaches tier 1/2/3 — flagged immediately with `(None, True)`. This is
exactly why `"500@-170"` (JWST's real Site Code value, D-09) is already handled correctly
today — it's 8 characters, over the length guard, so it's flagged for manual review rather
than truncated/fabricated. **This phase's job for criterion 5 is to confirm this behavior
against the real rows, not change it.**

**`solsys_code/campaign_utils.py:parse_obs_window()`** (lines 186-244) — current signature:
```python
def parse_obs_window(obs_date_raw: str, ut_range_raw: str) -> tuple[date, datetime, datetime | None, bool]:
```
`obs_date_raw` must parse as exact `%Y-%m-%d` or raises (true natural-key failure);
`ut_range_raw` is always best-effort via three narrowly-scoped regexes
(`_HHMM_RANGE`, `_APPROX_HOUR`, `_BARE_HOUR_UTC`), falling back to midnight UTC with
`ut_needs_review=True` rather than raising. This is the exact function Phase 20 will extend
to accept range/TBD `Obs. Date` shapes (D-03/D-04) — this phase's decision doc should
reference these three regex names directly so Phase 20's planner doesn't have to rediscover
them.

**`solsys_code/models.py:CampaignRun`** (lines 31-128) — current natural key:
```python
models.UniqueConstraint(
    fields=['campaign', 'telescope_instrument', 'ut_start'],
    name='unique_campaign_run_natural_key',
)
```
`contact_person` (line 99, `CharField(max_length=255, blank=True, default='')`) already
exists on the model — D-06's decision to fold it into the TBD-row key needs no new field,
only a new/adjusted constraint (Phase 19's job).

**`solsys_code/solsys_code_observatory/models.py:Observatory`** (lines 11+) — fields
relevant to a future fuzzy-match candidate pool: `obscode` (`CharField(max_length=4)`),
`name` (`CharField(max_length=255, unique=True)`), `short_name` (`CharField(max_length=255)`),
`old_names` (`TextField(blank=True)`, free-text "any previous names used"). A future
fuzzy-match UI (Phase 21) would build its candidate-string pool from `name`/`short_name`/
`old_names` — this phase's probe script should use the same fields for its live comparison,
so the D-08 evidence directly informs Phase 21's design rather than testing against a
made-up candidate list.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fuzzy string similarity scoring | A custom Levenshtein-distance or token-overlap function | `rapidfuzz.fuzz.*` / `rapidfuzz.process.extractOne` or stdlib `difflib.get_close_matches` | Both already implement well-tested, edge-case-hardened similarity algorithms (Levenshtein-family in rapidfuzz, Ratcliff-Obershelp in difflib); a hand-rolled version would need to independently solve normalization, Unicode handling, and scoring-scale consistency that these libraries already solve. |

**Key insight:** This phase's actual decision isn't "should we use a library" (settled —
CONTEXT.md's phase boundary already frames it as "which of these two libraries," not
"library vs. hand-rolled") — it's "which of the two already-standard choices fits this
codebase's specific need" (matching short, often-abbreviated site codes/names against a
small, static `Observatory` candidate pool). The live-test comparison against D-09's real
messy corpus is what actually answers that, not documentation alone (per D-08).

## Common Pitfalls

### Pitfall 1: `difflib.SequenceMatcher.ratio()` is order-dependent
**What goes wrong:** `SequenceMatcher(None, a, b).ratio()` can return a different score than
`SequenceMatcher(None, b, a).ratio()` for the same pair of strings — documented explicitly
by Python's own stdlib docs with a worked example (`'tide'` vs `'diet'`: `0.25` one way,
`0.5` the other) [CITED: docs.python.org/3/library/difflib.html].
**Why it happens:** The Ratcliff-Obershelp algorithm's longest-matching-block recursion is
not symmetric in its treatment of the two input sequences.
**How to avoid:** When live-testing `difflib.get_close_matches(word, possibilities)` against
the D-09 corpus, be consistent about which argument is the raw sheet `Site Code` (the
`word`) and which is the `Observatory` candidate list (`possibilities`) — don't swap them
between test runs, and note this ordering explicitly in the decision doc so a future
implementer doesn't accidentally reverse it.
**Warning signs:** A score that looks surprisingly asymmetric between two visually-similar
test cases in the D-09 corpus is worth double-checking argument order before concluding
difflib is "worse" than rapidfuzz on that case.

### Pitfall 2: `rapidfuzz.fuzz.WRatio`'s length-ratio scaling can penalize short-vs-long comparisons
**What goes wrong:** `WRatio` (the default `process.extractOne` scorer) applies a length-ratio
rule: if one string is more than 1.5x the length of the other, it falls back to
`partial_ratio`-style comparisons (scaled by 0.9 so only a true full match can reach 100)
rather than a plain full-string ratio [CITED: rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html].
**Why it happens:** WRatio is designed as a general-purpose "best of several strategies"
scorer, tuned for cases like matching a short query against a long document — not
specifically for two short strings (e.g. a 3-character site code vs. a long observatory
name like `"Telescope Joan Oró, Montsec, Catalonia"`, D-09).
**How to avoid:** When comparing a short raw code (`C65`) against long candidate names, also
try `fuzz.token_sort_ratio`/`fuzz.token_set_ratio` (word-order-insensitive) alongside the
default `WRatio`, since the real D-09 corpus mixes short codes and long human-readable names
in the same candidate pool. Record which scorer performs best per case in the decision doc,
not just the default.
**Warning signs:** A candidate that a human would obviously pick (e.g. matching `"Deep Sky
Chile"` from the sheet's contact-person note against an `Observatory.old_names` entry)
scoring unexpectedly low under the default scorer.

### Pitfall 3: `score_cutoff=None` (rapidfuzz) / `cutoff=0.6` (difflib) are different defaults, not directly comparable
**What goes wrong:** `rapidfuzz.process.extractOne`'s `score_cutoff` defaults to `None`
(no filtering — always returns the single best match, however poor)
[CITED: rapidfuzz.github.io/RapidFuzz/Usage/process.html], while
`difflib.get_close_matches`'s `cutoff` defaults to `0.6` on a 0-1 scale (candidates below it
are silently dropped, and the function can return an *empty list*, not just a worse match)
[CITED: docs.python.org/3/library/difflib.html]. A live comparison that doesn't normalize
these will make rapidfuzz look artificially "more permissive" purely because of its default,
not its underlying algorithm.
**Why it happens:** The two libraries chose different default philosophies — rapidfuzz
returns "best available" by default, difflib returns "good enough or nothing."
**How to avoid:** When live-testing, either set an explicit `score_cutoff` on rapidfuzz
comparable to difflib's `0.6` (rapidfuzz scores are 0-100, so the equivalent is
`score_cutoff=60`), or explicitly note in the decision doc that "no result" is a valid,
meaningful outcome for difflib on a bad site code and a forced low-confidence match is the
equivalent outcome for rapidfuzz with no cutoff set.
**Warning signs:** difflib returning `[]` for a genuinely bad D-07/D-09 case (blank/garbage
code) while rapidfuzz returns *something* — that's expected default behavior, not a bug or a
disqualifying weakness of difflib.

### Pitfall 4: rapidfuzz being importable in this dev venv today is incidental, not guaranteed
**What goes wrong:** A probe script (or a future accidental production import) that assumes
`import rapidfuzz` just works because it happens to succeed in this shell session could fail
in CI, in a teammate's fresh `./.setup_dev.sh` environment, or after a `poetry`/`cleo`
version bump drops the transitive pull-in.
**Why it happens:** `rapidfuzz` is currently present only as a transitive dependency of
`cleo` (required by `poetry`, a dev/packaging tool) — confirmed via `pip show rapidfuzz` →
`Required-by: cleo`. It has never been an explicit project dependency.
**How to avoid:** The probe script should not assume rapidfuzz is pre-installed; document
the explicit `pip install rapidfuzz` step D-08 already requires, and never let this
incidental availability leak into a false sense that "it's already a dependency, no action
needed."
**Warning signs:** None during this phase (script is throwaway and git-excluded either way)
— but flag it for Phase 21's planner if the fuzzy-match UI phase ultimately picks rapidfuzz:
that phase must add `rapidfuzz` to `pyproject.toml` explicitly, not rely on the `cleo`
transitive pull-in.

## Code Examples

### rapidfuzz: best-match lookup against a small candidate list
```python
# Source: https://rapidfuzz.github.io/RapidFuzz/Usage/process.html (verified live, this venv: rapidfuzz 3.14.3)
from rapidfuzz import fuzz, process

candidates = ['X09', 'N50', 'X07', 'C65', 'Deep Random Survey', 'Deep Sky Chile',
              'Telescope Joan Oro, Montsec, Catalonia']  # e.g. Observatory obscode/name/short_name/old_names

# Default scorer (WRatio) -- returns (match, score, index) or None if score_cutoff not met
best = process.extractOne('X09', candidates, scorer=fuzz.WRatio, score_cutoff=60)
# best == ('X09', 100.0, 0) for an exact hit; try token_sort_ratio for long-name cases:
best_long_name = process.extractOne(
    'Deep Sky Chile', candidates, scorer=fuzz.token_sort_ratio, score_cutoff=60
)
```

### difflib: best-N-matches lookup (stdlib, no install)
```python
# Source: https://docs.python.org/3/library/difflib.html (stdlib, Python 3.12.3 confirmed installed this session)
import difflib

candidates = ['X09', 'N50', 'X07', 'C65', 'Deep Random Survey', 'Deep Sky Chile',
              'Telescope Joan Oro, Montsec, Catalonia']

matches = difflib.get_close_matches('X09', candidates, n=3, cutoff=0.6)
# matches == ['X09'] for an exact hit; returns [] (not a low-confidence guess) if nothing
# clears the 0.6 cutoff -- e.g. for D-09's genuinely blank/unresolvable codes.
```

### Probe-script harness shape (both libraries, live against real Observatory data)
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

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `fuzzywuzzy` (GPL, pure-Python, depends on `python-Levenshtein` C extension for speed) | `rapidfuzz` (MIT, C++-backed, no GPL dependency chain) | rapidfuzz has been the recommended successor since ~2021; `fuzzywuzzy` itself now recommends switching in its own README [CITED: pypi.org/project/rapidfuzz] | Not directly relevant to this phase's binary choice (rapidfuzz vs. difflib), but rules out "should we consider fuzzywuzzy instead" as a live option — it's already superseded. |

**Deprecated/outdated:** `fuzzywuzzy`/`thefuzz` as a first-choice recommendation — rapidfuzz
is a drop-in-compatible, faster, permissively-licensed replacement; no reason to evaluate it
separately in this phase's live comparison.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | rapidfuzz's `process.extractOne`/`fuzz.WRatio` and difflib's `get_close_matches`/`SequenceMatcher` are the right pair of entry points to live-test (rather than e.g. rapidfuzz's lower-level `Levenshtein` module or `fuzz.ratio` alone) | Code Examples, Standard Stack | Low — both are the documented top-level, purpose-built APIs for "best match from a candidate list," confirmed via each library's official Usage docs this session; if wrong, the probe script can trivially add another scorer/function to the same harness without redesign. |

**All other claims in this research were verified this session** (via `pip show`,
`pip index versions`, `pypi.org`, `pypistats.org`, `rapidfuzz.github.io`,
`docs.python.org`, and direct reads of `campaign_utils.py`/`models.py`) or cited from
CONTEXT.md's already-locked D-01..D-10 findings — no other claim needs user confirmation
before becoming a locked decision.

## Open Questions

1. **Will the live-test comparison actually produce a clear winner, or a split verdict
   (e.g. rapidfuzz better for short-code matches, difflib fine for exact/near-exact cases)?**
   - What we know: D-09's real corpus mixes very short codes (`X09`, `C65`) with long
     human-readable names (`"Telescope Joan Oró, Montsec, Catalonia"`) and several
     no-code/blank cases (D-07) that neither library can meaningfully "match" at all.
   - What's unclear: Whether the eventual Phase 21 UI needs one library or could reasonably
     use difflib (zero new dependency) if its match quality proves close enough on this
     specific corpus.
   - Recommendation: The plan should treat "record scores for all D-09 cases under both
     libraries, then decide" as the concrete deliverable — don't presuppose rapidfuzz wins
     just because it's the more full-featured library; D-08 explicitly calls for live
     evidence over documentation-based reasoning.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|--------------|-----------|---------|----------|
| `rapidfuzz` (PyPI) | Live-comparison probe script (D-08) | Yes, incidentally (transitive dep of `cleo`/`poetry`) [VERIFIED: pip show] | 3.14.3 installed / 3.14.5 latest | Explicit `pip install rapidfuzz` if run in an environment without `poetry`/`cleo` installed — do not rely on the transitive pull-in (Pitfall 4). |
| `difflib` (stdlib) | Live-comparison probe script (D-08) | Yes, always | Bundled with Python 3.12.3 (this session) / 3.10-3.12 (project's supported range) | None needed — stdlib. |
| Real 3I/ATLAS CSV | All of D-01..D-10's evidence, and Plan 18's re-read (D-02) | Assumed yes (Tim's local machine path per D-01) — not independently verified this research session since it's outside `.planning/`/repo scope and PII-gated | — (live Google Sheet export, no fixed version) | None — this phase is blocked without it; D-01 already documents the exact path. |

**Missing dependencies with no fallback:** none within this phase's control (the real CSV's
presence is an operator-side given, not something this research session verifies or can
substitute for).

**Missing dependencies with fallback:** `rapidfuzz`'s current incidental availability — has
a trivial fallback (`pip install rapidfuzz`), not a blocker.

## Validation Architecture

This phase produces no shippable application code — only a decision doc (or docs) and a
throwaway, git-excluded probe script (never committed, per D-08/D-09's Phase 13 precedent).
There is no `PLAN.md` task in this phase that changes `solsys_code/` production behavior, so
there is nothing here for an automated test suite to cover. Phase 13 (the only prior
spike-shaped phase) established this exact precedent: no test-map section, no Wave 0 gaps,
because a decision doc has no runtime behavior to assert against.

**Sampling rate / gate:** N/A this phase. The existing `./manage.py test solsys_code` and
`pytest` suites remain green throughout (nothing in this phase's scope touches
`solsys_code/models.py`, `campaign_utils.py`, or any tested module — confirmed by this
phase's Architecture Patterns section, which documents current behavior read-only). The
planner should still include a final `ruff check .`/`ruff format --check .` verification
step (CLAUDE.md's baseline quality gate) since the decision doc's Markdown and any
supporting `.rst` durable-summary content don't trigger ruff, but a stray leftover
`fuzzy_match_probe.py` accidentally staged would — the gate is "probe script is git-excluded
and not staged," already captured in the Architecture Patterns verification-gate example
above.

## Security Domain

`security_enforcement` is enabled project-wide (`.planning/config.json`), but this phase
ships no application code with any of V2 (Authentication), V3 (Session Management), V4
(Access Control), V5 (Input Validation), or V6 (Cryptography) surface area — it produces a
decision doc and a throwaway, never-committed, read-only probe script. No ASVS category
applies to code delivered *by this phase*. The one real security/privacy-adjacent concern is
already fully covered by CONTEXT.md's locked decisions, not a new finding:

- **PII handling (D-01):** the real 3I/ATLAS CSV contains real names/emails. It must be read
  directly from its local path, never copied into the repo/`.planning/`, and any verbatim
  cell text quoted into a committed decision doc must have `Contact Person`/`Email`
  redacted — matching Phase 13's D-04 API-response-redaction precedent for credential-
  adjacent content. This is a data-handling discipline for the decision-doc-writing task,
  not a code-level ASVS control.

No threat-pattern table is included — there is no STRIDE-relevant attack surface (no new
endpoint, no new user input path, no new stored-data field) introduced by this phase.

## Sources

### Primary (HIGH confidence)
- `pip show rapidfuzz`, `pip index versions rapidfuzz` (this session, this dev venv) —
  confirmed installed version 3.14.3, latest 3.14.5, transitive dependency of `cleo`.
- `pypi.org/project/rapidfuzz/` — package metadata, license, author, Python version support
  (fetched live this session).
- `pypistats.org/packages/rapidfuzz` — weekly/daily download counts (fetched live this
  session).
- `docs.python.org/3/library/difflib.html` — `get_close_matches`/`SequenceMatcher` exact
  signatures, defaults, and documented order-dependency caveat (fetched live this session).
- `rapidfuzz.github.io/RapidFuzz/Usage/process.html` and `.../Usage/fuzz.html` —
  `process.extractOne`/`process.extract` signatures, `WRatio` length-scaling behavior
  (fetched live this session).
- Direct reads of `solsys_code/campaign_utils.py` and `solsys_code/models.py` (this
  session) — exact current `resolve_site()`/`parse_obs_window()` signatures and
  `CampaignRun`'s natural-key constraint.
- `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-01-PLAN.md` and
  `13-CONTEXT.md`/`13-DECISION.md` (this session) — throwaway-script convention
  (`.git/info/exclude`), redaction precedent, decision-doc structure precedent.
- `gsd-tools query package-legitimacy check --ecosystem pypi rapidfuzz` (this session) —
  automated verdict `SUS` (reason: `unknown-downloads`), cross-verified manually via the
  PyPI/pypistats sources above.

### Secondary (MEDIUM confidence)
None — all findings this session were either directly verified via tool/fetch or already
locked in CONTEXT.md as real-data findings from the discussion phase.

### Tertiary (LOW confidence)
None.

## Metadata

**Confidence breakdown:**
- Fuzzy-match library API surface: HIGH — both libraries' documentation fetched live this
  session and cross-checked against the actually-installed version in this dev venv.
- Window schema / natural key / CSV parsing rules: HIGH — already fully evidenced in
  CONTEXT.md against the real sheet (D-01..D-07/D-10); not re-researched here per the
  phase-specific research focus (explicitly out of scope for this document).
- Package legitimacy: HIGH after manual cross-verification (automated gate's `SUS` verdict
  was a tooling limitation, not a real signal — documented explicitly above).

**Research date:** 2026-07-08
**Valid until:** 30 days (library API surfaces are stable; the real CSV itself is a live,
publicly-editable sheet that may have changed further by execution time, per D-02 — Plan 18
must re-read it directly rather than trusting this document or CONTEXT.md as final).
