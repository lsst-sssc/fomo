# Phase 21: Site Disambiguation & Submitter Contact Opt-In - Research

**Researched:** 2026-07-11
**Domain:** Django form/view/table UI (approval queue), MPC Observatory Codes API integration, Django low-level caching, PII-gated queryset design
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Candidate pool is widened to the **live MPC Obscodes list**, not just the
  local `Observatory` table (~8 rows today) that Phase 18's spike tested against. The
  spike's evidence (`18-DECISION.md` Criterion 4) explicitly flagged the local table as
  too narrow a candidate pool to meaningfully fuzzy-match arbitrary external site codes —
  this decision acts on that finding.
- **D-02:** Fetch via the existing `MPCObscodeFetcher`
  (`solsys_code/solsys_code_observatory/utils.py`), **cached locally** (DB table or
  Django cache) rather than fetched live on every approval-queue page render. Refresh
  periodically (management command or lazy-on-stale), not per-request — avoids adding
  network latency/failure risk to a page staff load often. Exact cache mechanism
  (dedicated cache table vs. Django's cache framework vs. a periodic sync command
  populating a lookup table) is Claude's discretion during planning/research.
- **D-03:** The fuzzy match still only runs after `resolve_site()`'s own tier 1 (exact
  match) and tier 2 (live single-obscode MPC lookup) have both missed — per SITE-01's
  literal wording and the spike's note. This is not a new decision, just confirmed scope.
- **D-04:** Site column becomes an inline `<select>` of fuzzy-matched candidates plus a
  free-text fallback input, directly in the existing approval-queue table row. The
  chosen/typed value rides along with the **existing** approve/reject form POST to
  `CampaignRunDecisionView` — no new endpoint, no AJAX, no separate "resolve site" action
  decoupled from the decision.
- **D-05:** SITE-02's "create a new Observatory" (when no fuzzy-matched candidate is
  correct) reuses the **existing `CreateObservatory` form/flow**
  (`solsys_code_observatory`), not a new lightweight inline form. Link/redirect to it and
  return to the approval queue afterward — avoids duplicating obscode-driven creation
  validation that already exists.
- **D-06:** `CampaignRunDecisionView.post()` (lines 291-413 of `campaign_views.py`)
  currently *always* calls `resolve_site(run.site_raw, create_placeholder=False)` on
  approve, unconditionally overwriting `run.site`/`run.site_needs_review`. Fix: **skip the
  `resolve_site()` call whenever `run.site` is already set** (not `None`) at approve time
  — trust an already-resolved site (whether resolved at CSV-import time, tier 1/2
  auto-resolution, or staff's new manual-resolution UI from D-04) rather than adding a new
  `site_manually_resolved` field/migration. Only a run with `site=None` still gets
  auto-resolved on approve.
- **D-07:** A single checkbox on `CampaignRunSubmissionForm`
  (`solsys_code/campaign_forms.py`), placed immediately after the existing
  `contact_person`/`contact_email` fields, default **unchecked** (opt-out, matches
  today's staff-only behavior when unset). Set once at submission — there is no submitter
  self-service edit view today (confirmed), so no "editable after submission" mechanism
  is built this phase. Exact field name, verbose label, and help text are Claude's
  discretion (mirror `open_to_collaboration`'s existing style/placement precedent).
- **D-08:** When set, the new flag adds `contact_person`/`contact_email` to
  `ALLOWED_FIELDS_FOR_NON_STAFF` (or an equivalent per-row conditional) so anonymous
  visitors see them on the per-campaign table for that run only — runs that didn't opt in
  stay staff-only exactly as today.

### Claude's Discretion

- Exact MPC-list cache mechanism (D-02) — dedicated table, Django cache, or a periodic
  sync command.
- Fuzzy-match candidate count/threshold shown in the dropdown (not discussed — use
  judgement, consistent with `difflib.get_close_matches`'s `cutoff`/`n` defaults unless
  research surfaces a reason to change them).
- VIEW-05 checkbox field name, verbose label, help text (D-07).
- Whether the per-row "opted in" state needs its own visible indicator in the approval
  queue, or is purely a per-campaign-table display concern.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. Two weakly-matched pending todos
(`2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` and
`2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md`) were
reviewed and confirmed not relevant to this phase's `campaign_utils.py`/approval-queue/
submission-form scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SITE-01 | When a submitted `site_raw` doesn't resolve via `resolve_site()`'s existing tier 1 (exact `Observatory` match) or tier 2 (live MPC Obscodes API), the approval queue's Site column presents a dropdown of fuzzy-matched `Observatory` candidates for staff to pick from | Pattern 2/3 (bulk MPC fetch + candidate-pool flattening, live-verified 2,710 codes), Pattern 1 (cache reuse), Pattern 4 (`<datalist>`/`form=` UI wiring), Pitfalls 1/2/5 (which corpus codes genuinely need fuzzy matching, acronym-gap limitation, per-request pool construction) |
| SITE-02 | Staff can type a code directly and resolve it to an existing `Observatory` or explicitly create a new one; no placeholder `Observatory` is ever auto-fabricated | Pattern 4 (free-text input in the same control as the dropdown), Don't Hand-Roll (reuse `CreateObservatory`), Pitfall 6 (`CreateObservatory`'s missing `?next=`/`?obscode=` support — a real gap to close), Pitfall 4 (satellite-obscode `to_observatory()` limitation, pre-existing, not a regression) |
| SITE-03 | Approving a run whose site a staff member already manually resolved does not silently re-resolve or overwrite that choice | Architecture Diagram (D-06 guard), Pitfall 3 (concrete, already-reachable reproduction path via the calendar-projection revert flow — not just a hypothetical introduced by the new UI) |
| VIEW-05 | A submitter who opts in (single combined flag, default opt-out) has their `contact_person`/`contact_email` shown publicly on the per-campaign table; leaving it unset keeps them staff-only exactly as today | Pattern 5 (`Case`/`When` queryset-level PII gating, preserves 15-RESEARCH.md Pitfall 1's discipline), Code Examples (migration shape mirroring `open_to_collaboration`), Assumption A1 (field-name recommendation) |
</phase_requirements>

## Summary

This phase adds two independent, structurally small features to the existing approval-queue
and submission-form code, both already scoped precisely by CONTEXT.md's D-01..D-08. Research
this session focused on de-risking the three genuinely open implementation questions: (1) how
to widen the fuzzy-match candidate pool to the live MPC Obscodes API and cache it idiomatically
for this codebase, (2) how to wire a per-row `<select>`+free-text site input into the *existing*
approve/reject POST without a new endpoint or AJAX, and (3) how to expose `contact_person`/
`contact_email` per-row for opted-in runs only while preserving `ALLOWED_FIELDS_FOR_NON_STAFF`'s
existing "restrict the queryset itself, not just the template" security discipline.

All three now have a concrete, verified answer. The MPC Obscodes API **does** support a bulk
"all observatory codes" query (confirmed live this session: omitting the `obscode` key in the
POST body returns all 2,710 codes as one ~1.5 MB JSON payload in ~1.3 s) — `MPCObscodeFetcher`
needs a small addition to support this mode; the result should be flattened to a lightweight
`candidate_string -> obscode` mapping and cached via Django's **existing low-level cache
framework** (`django.core.cache.cache`, already used by `campaign_gap.py` for an identically-shaped
"expensive computation, TTL cache" problem — this is the natural fit, not a new DB table or a
periodic sync command, since this repo has no cron/celery infrastructure at all). The inline
site-selection UI is best built as a single `<input type="text" list="...">` bound to a
`<datalist>` (HTML5, no JS) whose value rides into the row's approve/reject form via the HTML5
`form="..."` attribute — this also requires collapsing the *existing* two-separate-`<form>`
Approve/Reject rendering in `ApprovalQueueTable.render_actions()` into one `<form>` with two
named submit buttons, a real (if small) refactor of already-shipped code. The PII-gating
question resolves cleanly with a `Case`/`When` SQL-level conditional annotation, keeping
`contact_person`/`contact_email` values physically excluded from the non-staff SELECT for
runs that haven't opted in, with **zero change** needed to `render_*` methods.

A significant, freshly-verified finding reframes SITE-01's actual value: three of Phase 18's
four "difflib/rapidfuzz both miss" corpus codes (`X09`, `N50`, `X07`, `C65`) turn out to be
**real, valid MPC obscodes** whose live-API names correctly match the submitter's described
site (`X09` = "Deep Random Survey, Rio Hurtado" for Sam Deen's "Deep Random Survey"; `C65` =
"Observatori Astronòmic del Montsec" for the Joan Oró/Montsec row, etc.) — `resolve_site()`'s
**existing, unmodified tier 2** (a live single-code MPC lookup) already resolves all four
correctly, no fuzzy match needed. The real gap fuzzy-matching closes is the acronym/nickname
case (e.g. `'DCT'` for the Lowell Discovery Telescope, MPC code `G37`) — confirmed this session
that even the widened 5,636-string candidate pool cannot bridge this via `difflib` (character-
sequence matching finds no shared substring between `'DCT'` and `'Lowell Discovery Telescope'`).
Plan around this: the free-text/create-new fallback (D-05) is not a rare escape hatch, it is the
primary path for nickname-style site text.

Also found, independent of any new UI: **SITE-03's clobbering bug is already reachable today**
via `CampaignRunDecisionView.post()`'s existing `except Exception` revert path — `run.site` is
saved *before* calendar-projection code that can fail; on failure `approval_status` reverts to
`PENDING_REVIEW` but `run.site` stays set, so a second Approve click re-runs `resolve_site()`
unconditionally and can silently overwrite the already-good resolution. D-06's fix (skip
`resolve_site()` when `run.site` is already set) closes this concretely, not just defensively.

**Primary recommendation:** Reuse `django.core.cache.cache` (already imported in
`campaign_gap.py`) for the MPC candidate pool with a long TTL (e.g. 24h, lazy-refresh-on-miss);
build the site-selection UI as one native `<input list=...>`+`<datalist>` per pending row wired
via the HTML5 `form=` attribute into a refactored single-`<form>`-per-row `render_actions()`;
gate `contact_person`/`contact_email` at the queryset level via `Case`/`When`, not by adding
them unconditionally to `ALLOWED_FIELDS_FOR_NON_STAFF`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| MPC obscode candidate fetch + flatten | API / Backend | Database / Storage (cache) | `MPCObscodeFetcher` extension is a backend integration; the result is persisted only in the cache layer, never as new `Observatory` rows (avoids polluting the table with 2,700 unused rows). |
| MPC candidate pool caching | Database / Storage | API / Backend | Django's low-level cache framework (`django.core.cache.cache`) is itself a storage-tier concern already used identically by `campaign_gap.py`; the backend view/table layer only calls `cache.get`/`cache.set`. |
| Fuzzy-match execution (`difflib`) | API / Backend | — | Pure server-side computation against the cached candidate pool + `Observatory.objects` queryset; no client-side matching. |
| Site-selection input UI (datalist + free text) | Browser / Client | Frontend Server (SSR) | The `<datalist>`/`<input list=...>` browser-native autocomplete is a client-tier capability; Django server-renders the `<option>` list per row (SSR), no AJAX round-trip. |
| Approve/reject decision + clobber guard | API / Backend | — | `CampaignRunDecisionView.post()` — pure server-side state transition + conditional `resolve_site()` call, already staff-gated. |
| "Create new Observatory" flow | API / Backend | Browser / Client (redirect) | Reuses the existing `CreateObservatory` `CreateView`; client only follows a link/redirect, all validation/creation server-side. |
| Contact opt-in flag storage | Database / Storage | — | New `BooleanField` on `CampaignRun`, default `False`, mirrors `open_to_collaboration`. |
| Per-row PII exposure gating | Database / Storage (query) | API / Backend (queryset construction) | Must be enforced in the SQL `SELECT` itself (a `Case`/`When` annotation), not merely in template rendering — matches the existing `ALLOWED_FIELDS_FOR_NON_STAFF` discipline (15-RESEARCH.md Pitfall 1). |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `difflib` (stdlib) | bundled, Python 3.10-3.12 [VERIFIED: docs.python.org, already locked by 18-DECISION.md] | Fuzzy string matching (`get_close_matches`) against the widened candidate pool | Already the locked library choice from Phase 18's spike — not re-decided here. Zero new dependency. |
| `django.core.cache.cache` | Django (already installed, version pinned by `tomtoolkit`) [VERIFIED: `src/fomo/settings.py` `CACHES` block, `solsys_code/campaign_gap.py` existing usage] | TTL-cached MPC obscode candidate pool | Already the established pattern in this exact codebase for "expensive computation, cache with TTL" (`campaign_gap.py`'s `GAP_CACHE_TTL_SECONDS = 3600` / `cache.get`/`cache.set`). No new infrastructure. |
| `requests` (already a transitive/direct dependency via `MPCObscodeFetcher`) | already installed [VERIFIED: `solsys_code/solsys_code_observatory/utils.py`] | Bulk MPC Obscodes API fetch (no-`obscode`-key POST body) | `MPCObscodeFetcher` already wraps `requests.get`; the bulk-fetch mode is a documented variant of the same endpoint, not a new integration. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| HTML5 `<datalist>` + `form=` attribute | browser-native (Chrome/Firefox/Safari/Edge, no polyfill needed) [CITED: developer.mozilla.org/en-US/docs/Web/HTML/Element/datalist, .../Element/input#form] | Per-row dropdown-of-candidates + free-text fallback in one control, submitted into a form declared in a *different* table cell | The approval-queue table's Site column and Actions column are separate `<td>`s; `form="decide-form-{pk}"` is the standards-based way to associate an `<input>` outside a `<form>` element with that form for submission purposes. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Django's low-level cache framework | A dedicated `MPCObscodeCache`/`ObservatoryCandidate` DB model + periodic sync management command | Would give queryable/indexed candidates and survive a cache-backend swap, but this repo has **no cron/celery/scheduled-task infrastructure at all** [VERIFIED: grep for `crontab`/`celery`/cron config in `pyproject.toml`/`settings.py`/repo — none found] — a "periodic sync command" would need external ops setup with zero existing precedent, whereas `cache.get`/`cache.set` needs none. Reconsider only if a future need for indexed/filterable candidate queries emerges. |
| `<datalist>` + `form=` attribute (no AJAX) | A JS-driven `<select>` with an `onchange` handler copying into a hidden input | Achieves the same "no new endpoint" goal but adds a new JS dependency/pattern to a codebase that currently ships **zero custom JS** in the campaigns templates (grep confirmed: `approval_queue.html` has no `<script>` block) — the native HTML5 approach needs none. |
| `Case`/`When` queryset-level PII gating | Keep `ALLOWED_FIELDS_FOR_NON_STAFF` static and gate only in `render_*` template methods | Would reintroduce the exact anti-pattern 15-RESEARCH.md Pitfall 1 already fixed for VIEW-03 — the raw PII would sit in every non-staff `.values()` dict in memory (and in any future serialization of that queryset) even for opted-out runs, not just fail to render. |

**Installation:** No new packages. `difflib` is stdlib; `django.core.cache` and `requests` are
already installed and imported elsewhere in this codebase.

**Version verification:** No new external packages are introduced this phase — `difflib` is
part of the Python 3.10-3.12 standard library `pyproject.toml` already targets; Django's cache
framework and `requests` are already pinned via existing `pyproject.toml`/`tomtoolkit`
dependencies. No `npm view`/`pip index versions` check applies.

## Package Legitimacy Audit

**No new external packages are installed by this phase.** `difflib` is Python stdlib (no
registry entry to audit); `django.core.cache`, `requests`, and `django-tables2` are pre-existing
project dependencies already in use elsewhere in `solsys_code/`. `rapidfuzz` was explicitly
**not** added to `pyproject.toml` per Phase 18's split verdict (18-DECISION.md Criterion 4) and
remains out of scope here.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| — | — | — | — | — | — | No packages to audit this phase |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
Staff loads /campaigns/approval-queue/ (ApprovalQueueView.get_context_data)
        |
        v
For each PENDING_REVIEW row with run.site is None:
        |
        +--> candidate pool lookup (per-request, built once, reused across rows)
        |         |
        |         +--> cache.get('mpc_obscode_candidates')
        |         |         |
        |         |    [miss/stale]                [hit]
        |         |         |                         |
        |         |         v                         |
        |         |   MPCObscodeFetcher.query_all()    |
        |         |   (bulk POST, no obscode key)      |
        |         |         |                         |
        |         |         v                         |
        |         |   flatten -> {string: obscode}     |
        |         |         |                         |
        |         |         v                         |
        |         |   cache.set(..., TTL=24h)          |
        |         |         |                         |
        |         |         +-----------<--------------+
        |         v
        |   merged pool = local Observatory candidates U cached MPC candidates
        |
        +--> difflib.get_close_matches(run.site_raw, merged_pool, n=5, cutoff=0.6)
        |
        v
CampaignRunTable.render_site(record) renders, per row:
   - resolved (run.site set)         -> plain text (existing behavior, UNCHANGED)
   - unresolved, no fuzzy candidates -> <input list=...> (free text only) + "Create new" link
   - unresolved, fuzzy candidates    -> <input list=...> + <datalist> of matches + "Create new" link
        |
        v (staff types/selects a value, clicks Approve or Reject)
        |
   Single <form id="decide-form-{pk}"> in the Actions <td> (refactored from 2 forms -> 1)
   <input name="site_selection" form="decide-form-{pk}"> lives in the Site <td>, associated
   via the HTML5 form= attribute (works across table cells — no JS needed)
        |
        v  POST /campaigns/<pk>/decide/  {action: approve|reject, site_selection: "..."}
        |
CampaignRunDecisionView.post()
   .filter(pk=pk, approval_status=PENDING_REVIEW).update(approval_status=...)  [existing guard]
        |
   if action == approve and run.site is None:        <-- D-06 guard (NEW)
        selection = request.POST.get('site_selection', '').strip() or run.site_raw
        site, needs_review = resolve_site(selection, create_placeholder=False)  [UNCHANGED fn]
        run.site, run.site_needs_review = site, needs_review; run.save(...)
   else:
        # run.site already set (prior manual resolution, or a future non-decide write path)
        # -- skip resolve_site() entirely, trust the existing value (SITE-03 fix)
        |
   [existing calendar-projection logic, UNCHANGED]

--- separately, for SITE-02's "create new" path ---

Staff clicks "Create new Observatory" next to an unresolved row
        |
        v
GET /observatory/create/?obscode=<typed-code>&next=/campaigns/approval-queue/
        |
   CreateObservatory (existing CreateView, form pre-filled from ?obscode=, EXTENDED
   with a ?next= redirect target instead of always redirecting to the detail page)
        |
   MPCObscodeFetcher().query(obscode).to_observatory()  [existing, UNCHANGED]
        |
   redirect back to approval queue (?next= target) instead of the Observatory detail page
        |
   staff re-selects the now-locally-resolvable obscode in the datalist and clicks Approve
   (resolve_site() tier 1 now hits on the next attempt -- no special-case code needed)

--- VIEW-05, independent flow ---

CampaignRunSubmissionForm (public) gains one BooleanField, default False
        |
   CampaignRunSubmissionView.form_valid() persists it onto the new CampaignRun.<field>
        |
CampaignRunTableView.get_queryset() (non-staff branch)
   .annotate(
       _contact_person=Case(When(<field>=True, then=F('contact_person')), default=Value('')),
       _contact_email=Case(When(<field>=True, then=F('contact_email')), default=Value('')),
   )
   .values(*ALLOWED_FIELDS_FOR_NON_STAFF, contact_person=F('_contact_person'), contact_email=F('_contact_email'))
        |
   dict rows now always carry 'contact_person'/'contact_email' keys, but the SQL-level
   CASE expression means the actual PII string is only ever selected for opted-in rows --
   render_* methods and get_table_kwargs()'s exclude=(...) need NO changes beyond removing
   'contact_person'/'contact_email' from that exclude tuple.
```

### Recommended Project Structure

No new files. All changes land in already-identified existing modules:
```
solsys_code/
├── campaign_views.py         # CampaignRunDecisionView.post() D-06 guard; CampaignRunTableView Case/When queryset
├── campaign_tables.py        # render_site() gains the input+datalist branch; render_actions() single-form refactor
├── campaign_utils.py         # resolve_site() UNCHANGED; new build_site_candidates()/fuzzy_match_candidates() helpers
├── campaign_forms.py         # CampaignRunSubmissionForm gains the opt-in BooleanField
├── models.py                 # CampaignRun gains the opt-in field
├── migrations/000N_....py    # AddField migration, mirrors 0006's shape
└── solsys_code_observatory/
    ├── utils.py               # MPCObscodeFetcher gains a bulk-fetch mode
    └── views.py               # CreateObservatory: ?obscode= prefill + ?next= redirect support
```

### Pattern 1: Django low-level cache for an expensive, rarely-changing external list
**What:** `cache.get(key)` / on miss, compute+`cache.set(key, value, timeout=TTL)`.
**When to use:** Exactly this repo's existing precedent — `campaign_gap.py`'s
`get_or_compute_gap()`. Not a new pattern, a direct reuse.
**Example:**
```python
# Source: solsys_code/campaign_gap.py (existing, verbatim pattern to mirror)
from django.core.cache import cache

GAP_CACHE_TTL_SECONDS = 3600  # existing precedent -- MPC candidate pool should use a
                               # much longer TTL (e.g. 86400) since the MPC obscode list
                               # changes far less often than gap-analysis results

def get_or_compute_gap(campaign, target, site, start, end):
    key = build_gap_cache_key(campaign.pk, target.pk if target else None, site.pk, start, end)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _compute(...)
    cache.set(key, result, timeout=GAP_CACHE_TTL_SECONDS)
    return result
```

### Pattern 2: Bulk MPC Obscodes fetch (verified live this session)
**What:** Omitting the `obscode` key from the POST body returns every registered observatory
code as one dict keyed by 3-char code.
**When to use:** Building the widened D-01 candidate pool. Confirmed live this session:
`curl -X GET -H "Content-Type: application/json" -d '{}' https://data.minorplanetcenter.net/api/obscodes`
returned **2,710** codes, ~1.5 MB, in ~1.3s — cache this, never call it per-request.
**Example:**
```python
# Source: verified live this session against the real MPC API (data.minorplanetcenter.net)
import requests

def fetch_all_obscodes(timeout: float = 30) -> dict:
    """Returns {obscode: {name_utf8, short_name, old_names, obscode, observations_type, ...}}."""
    response = requests.get(
        'https://data.minorplanetcenter.net/api/obscodes', json={}, timeout=timeout
    )
    response.raise_for_status()
    return response.json()  # dict keyed by 3-char obscode, ~2710 entries as of 2026-07-11
```
This is a new method on `MPCObscodeFetcher` (e.g. `query_all(timeout=30)` storing the full dict
on `self.obs_data`), sibling to the existing single-code `query()` — do not repurpose `query()`
itself, its `self.obs_data` shape (flat dict of one observatory) and `to_observatory()` contract
must stay unchanged for its two existing callers (`resolve_site()` tier 2, `CreateObservatory`).

### Pattern 3: Flattening the bulk response into a fuzzy-matchable candidate pool
**What:** `difflib.get_close_matches` needs a flat sequence of strings; build a
`candidate_string -> obscode` mapping so a match can be resolved back to an actionable code.
**Example (verified live this session against the real 2,710-entry payload):**
```python
def build_candidate_pool(obscode_dict: dict) -> dict[str, str]:
    """Maps each candidate display string (obscode, name_utf8, short_name) to its obscode.
    First-seen wins on collision (rare -- distinct sites practically never share a name)."""
    mapping: dict[str, str] = {}
    for code, rec in obscode_dict.items():
        for s in (code, rec.get('name_utf8') or '', rec.get('short_name') or ''):
            if s and s not in mapping:
                mapping[s] = code
    return mapping

# candidates = list(mapping.keys())  # 5,636 strings from the live 2,710-entry payload
# difflib.get_close_matches('X09', candidates, n=5, cutoff=0.6) -> ['X09']  # exact hit
# -- resolve back to an obscode via mapping['X09'] -> 'X09' (itself, in this case)
```

### Pattern 4: HTML5 `form=` attribute to submit a value from a different table cell
**What:** `<input form="target-form-id">` associates the input with a `<form>` element
elsewhere in the DOM (including a different `<td>` in the same `<tr>`), without JS.
**When to use:** D-04's "no new endpoint, no AJAX" requirement, given django-tables2 renders
one `<td>` per column — the Site column and the Actions column (where the form lives today)
are structurally separate cells.
**Example:**
```python
# Source: solsys_code/campaign_tables.py -- pattern for the refactored render_actions()
# (currently renders TWO independent <form> tags; must become ONE <form>, two named
# submit buttons, so a single site_selection input can target it)
def render_actions(self, record):
    if not self.show_actions:
        return ''
    decide_url = reverse('campaigns:decide', kwargs={'pk': record.pk})
    csrf_token = get_token(self.request) if self.request is not None else ''
    form_id = f'decide-form-{record.pk}'
    return format_html(
        '<form id="{0}" method="post" action="{1}">'
        '<input type="hidden" name="csrfmiddlewaretoken" value="{2}">'
        '<div class="d-flex" style="gap: 0.5rem;">'
        '<button type="submit" name="action" value="approve" class="btn btn-sm btn-success">Approve</button>'
        '<button type="submit" name="action" value="reject" class="btn btn-sm btn-danger" '
        'onclick="return confirm(\'Reject this submission? '
        'The submitter will not be automatically notified.\')">Reject</button>'
        '</div></form>',
        form_id, decide_url, csrf_token,
    )

# In render_site(), for an unresolved pending row:
def render_site(self, record):
    ...
    pk = Accessor('pk').resolve(record, quiet=True)
    site_raw = Accessor('site_raw').resolve(record, quiet=True) or ''
    datalist_id = f'site-candidates-{pk}'
    options = format_html_join('', '<option value="{}">{}</option>', candidate_pairs)
    return format_html(
        '<input type="text" name="site_selection" value="{0}" list="{1}" '
        'form="decide-form-{2}" class="form-control form-control-sm">'
        '<datalist id="{1}">{3}</datalist> '
        '<a href="{4}" class="small">Create new</a>',
        site_raw, datalist_id, pk, options, create_url,
    )
```
`format_html`/`format_html_join` are already the established escaping mechanism in this file
(`render_window_start`'s tooltip) — reuse, never `mark_safe`/f-string interpolation of
`site_raw` or any submitter-controlled text (stored-XSS defense, matches T-20-03 precedent).

### Pattern 5: Queryset-level conditional PII exposure (`Case`/`When`)
**What:** Replace a raw column selection with a conditional expression evaluated in SQL, so
the underlying value is genuinely absent from the result set for rows that don't qualify —
not merely hidden by later Python/template logic.
**Example:**
```python
# Source: Django ORM docs (conditional expressions) — new pattern for this codebase,
# not previously used in campaign_views.py, but a standard, well-documented ORM feature.
from django.db.models import Case, CharField, EmailField, F, Value, When

def get_queryset(self):
    campaign_pk = self.kwargs['pk']
    qs = CampaignRun.objects.filter(campaign_id=campaign_pk)
    if self.request.user.is_staff:
        return qs.select_related('site').order_by(F('window_start').desc(nulls_last=True))
    qs = qs.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
    qs = qs.annotate(
        _public_contact_person=Case(
            When(contact_opt_in=True, then=F('contact_person')),
            default=Value(''), output_field=CharField(),
        ),
        _public_contact_email=Case(
            When(contact_opt_in=True, then=F('contact_email')),
            default=Value(''), output_field=EmailField(),
        ),
    )
    qs = qs.order_by(F('window_start').desc(nulls_last=True))
    return qs.values(
        *[f for f in ALLOWED_FIELDS_FOR_NON_STAFF if f not in ('contact_person', 'contact_email')],
        contact_person=F('_public_contact_person'),
        contact_email=F('_public_contact_email'),
    )
```
`get_table_kwargs()`'s non-staff branch must then **stop** excluding `contact_person`/
`contact_email` from the table (they're now always safe to render — blank for opted-out rows,
populated for opted-in ones) — remove those two from its `exclude=(...)` tuple. No change to
`CampaignRunTable`'s column definitions or any `render_*` method is needed.

### Anti-Patterns to Avoid
- **Adding `contact_person`/`contact_email` unconditionally to `ALLOWED_FIELDS_FOR_NON_STAFF`
  and gating only in a template/render method:** reintroduces exactly the defense-in-depth gap
  15-RESEARCH.md Pitfall 1 was written to close — the raw PII string would be present in every
  non-staff `.values()` dict in memory regardless of opt-in state.
- **Calling `resolve_site()` a second time "just to be safe" even when `run.site` is already
  set:** this is the literal SITE-03 bug (D-06) — trust the existing value once set, don't
  re-derive it.
- **Fetching the full MPC bulk payload per approval-queue page load:** 1.5 MB / ~1.3 s per
  request (verified this session) for a page staff load frequently — must be cached.
- **Treating the widened MPC pool as a fix for acronym/nickname mismatches** (e.g. `'DCT'`):
  verified this session that `difflib` cannot bridge these even against the full 5,636-string
  pool — the free-text/create-new fallback is load-bearing, not a rare escape hatch.
- **A JS `onchange` handler to sync a `<select>` into a hidden input:** works, but introduces
  a JS pattern this template currently has none of; the native `<datalist>` + `form=` approach
  achieves the same UX with zero JS.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fuzzy string similarity | A custom edit-distance/substring-overlap scorer | `difflib.get_close_matches` (already locked, 18-DECISION.md) | Already decided; re-deriving this here would contradict the locked phase boundary. |
| MPC observatory metadata lookup/creation | Direct `requests` calls scattered across new code | `MPCObscodeFetcher` (extended with a bulk mode) | Centralizes the API contract (field names, error shapes) in one place, matching the existing single-code caller's precedent. |
| "Create a new Observatory from an obscode" | A bespoke inline form in the approval-queue template | Existing `CreateObservatory` `CreateView` (D-05) | Already validates/creates via `MPCObscodeFetcher.to_observatory()`; duplicating this risks diverging validation between the two creation paths. |
| Expensive-computation caching | A hand-rolled in-process dict/module-level cache, or a new DB table | Django's `django.core.cache.cache` (already used identically by `campaign_gap.py`) | Already the established, TTL-aware, backend-agnostic mechanism in this codebase; a module-level dict wouldn't survive process restarts or work across multiple worker processes. |

**Key insight:** every "don't hand-roll" item in this phase already has an established,
in-repo precedent to mirror (`resolve_site()`, `MPCObscodeFetcher`, `CreateObservatory`,
`campaign_gap.py`'s cache pattern) — this phase is almost entirely "extend/reuse", not "design
from scratch."

## Common Pitfalls

### Pitfall 1: Treating Phase 18's "fuzzy match misses" corpus as evidence that fuzzy matching is broken
**What goes wrong:** Assuming `X09`/`N50`/`X07`/`C65` (Phase 18's four spike test codes) need
the fuzzy-match dropdown to resolve correctly.
**Why it happens:** Phase 18's spike only tested fuzzy matching against the narrow 8-row local
`Observatory` table; it never actually ran these codes through `resolve_site()`'s tier 2 (a
live single-code MPC lookup).
**How to avoid:** Verified live this session — all four are real, valid MPC obscodes whose
live-API `name_utf8` genuinely matches the submitter's described site (e.g. `X09` = "Deep
Random Survey, Rio Hurtado" for Sam Deen's "Deep Random Survey / 43cm"). `resolve_site()`'s
existing, **unmodified** tier 2 already resolves all four correctly with no Phase 21 code
change. Don't build test fixtures around these four codes as "must be fuzzy-matched" cases —
they're tier-2 hits. Use a genuinely non-obscode nickname (e.g. `'DCT'`) as the fuzzy-match
test case instead.
**Warning signs:** A test asserting the fuzzy-match dropdown is *needed* to resolve `X09`/etc.
will pass today even with zero Phase 21 changes (tier 2 alone resolves it) — a false signal
that the feature works when it was never exercised.

### Pitfall 2: `difflib` cannot bridge acronym/nickname gaps, even against the widened pool
**What goes wrong:** Expecting the widened MPC candidate pool to fuzzy-match `'DCT'` to
`'Lowell Discovery Telescope'` (MPC code `G37`).
**Why it happens:** `difflib.SequenceMatcher`'s Ratcliff-Obershelp algorithm finds matching
*contiguous character blocks* — `'DCT'` shares no contiguous 2+ character substring with
`'Lowell Discovery Telescope'` in the right positions, so it scores below the `0.6` cutoff.
Verified live this session: `difflib.get_close_matches('DCT', <5636-string pool>, n=8,
cutoff=0.6)` returns `[]`; lowering `cutoff=0.3` still surfaces only unrelated `SARA-CT`/`T35`-
style codes, never the correct `G37`.
**How to avoid:** Design the UI (D-05) so the free-text input and "Create new Observatory" link
are always visible and prominent for unresolved rows, not a secondary/collapsed option only
shown when the dropdown is empty — this is the primary path for nickname-style input, not a
fallback for rare cases.
**Warning signs:** A UI that visually de-emphasizes the free-text/create-new option (e.g.
behind a toggle) will make the common `'DCT'`-style case harder to use, not easier.

### Pitfall 3: SITE-03's clobbering bug is already live-reachable, independent of any new UI
**What goes wrong:** Treating D-06's fix as purely defensive/future-proofing for the new
manual-resolution UI.
**Why it happens:** `CampaignRunDecisionView.post()` today saves `run.site`/
`run.site_needs_review` via `run.save(update_fields=[...])` **before** the calendar-projection
code that can raise into the outer `except Exception:` handler. That handler reverts
`approval_status` back to `PENDING_REVIEW` via a bare `.update()` call — but never resets
`run.site` back to `None`. A second Approve click on the same row (e.g. after a transient
calendar-projection failure) re-enters the `updated_count == 1` branch and calls
`resolve_site()` **again**, unconditionally, even though the site was already correctly
resolved on the first attempt.
**How to avoid:** D-06's `if run.site is None:` guard (skip `resolve_site()` entirely when
already set) closes this reachable path, not just a hypothetical one. Write a regression test
that: (1) approves a run where site resolution succeeds but calendar projection raises
(mock `insert_or_create_calendar_event` or `sun_event` to raise a non-`ValueError` exception),
confirms `approval_status` reverts to `PENDING_REVIEW` while `run.site` stays set, then (2)
approves again and asserts `resolve_site`/`MPCObscodeFetcher.query` is **not** called a second
time (`@patch` + `assert_not_called()`), and `run.site` is unchanged.
**Warning signs:** No existing test in `test_campaign_approval.py`'s
`TestCalendarProjection`/`TestApproval` classes currently exercises this two-attempt sequence —
confirmed by reading the full file's class list this session.

### Pitfall 4: `MPCObscodeFetcher.to_observatory()` still can't handle satellite-type records
**What goes wrong:** Assuming the widened candidate pool lets staff resolve `250`/`274`/`289`
(Hubble/JWST/Roman) end-to-end via the new UI.
**Why it happens:** Phase 18's spike already found (and this session's live re-verification
confirms, `longitude: null` for all three in the current live payload) that
`MPCObscodeFetcher.to_observatory()` raises `TypeError: float() argument must be a string or a
real number, not 'NoneType'` on `elong = float(self.obs_data['longitude'])` for any
`observations_type='satellite'` record. This bug is unrelated to and unfixed by this phase.
**How to avoid:** If a staff member selects `250`/`274`/`289` from the fuzzy-match dropdown
(they will appear as valid candidates once the widened pool includes them, e.g. matching
free-text "Hubble" or "JWST"), `resolve_site()` will still fall through to `(None, True)` via
the same `except (KeyError, ValueError, TypeError): pass` path documented in 18-DECISION.md —
safe (no crash, no fabrication) but the run stays `site_needs_review=True` even after staff
"resolved" it via the dropdown. This is expected, pre-existing behavior, not a regression to
fix in this phase — but worth a code comment at the call site so a future maintainer doesn't
mistake it for a new bug.
**Warning signs:** A UAT test that selects "James Webb Space Telescope" from the dropdown and
expects the run's site column to show a resolved short_name afterward will fail — this is a
real, documented (not silently swallowed) limitation, not this phase's regression.

### Pitfall 5: Candidate pool must be built once per request, not once per row
**What goes wrong:** Calling `cache.get('mpc_obscode_candidates')` and rebuilding the merged
local+MPC pool inside a per-row loop (e.g. inside `render_site()`, which django-tables2 calls
once per row per render).
**Why it happens:** `render_site()` doesn't naturally have access to "all rows" — it's easy to
default to a lazy per-call fetch.
**How to avoid:** Build the merged candidate pool once in `ApprovalQueueView.get_context_data()`
(or a helper it calls) before constructing the tables, and pass it into
`ApprovalQueueTable.__init__` (mirroring the existing `request=` kwarg pattern) so
`render_site()` reads from `self.candidate_pool` rather than re-fetching. `cache.get()` itself
is cheap on a hit, but rebuilding the flattened `{string: obscode}` dict (5,636 entries) 20+
times per page load for a page with that many pending rows is wasted CPU for no benefit.
**Warning signs:** A noticeably slower approval-queue page load as the pending-review count
grows, even though caching is "in place."

### Pitfall 6: `CreateObservatory`'s current `get_success_url()` always redirects to the Observatory detail page
**What goes wrong:** Assuming D-05's "link/redirect to it and return to the approval queue
afterward" already works out of the box.
**Why it happens:** `CreateObservatory.get_success_url()` unconditionally returns
`reverse_lazy('solsys_code.solsys_code_observatory:detail', kwargs={'pk': self.kwargs['pk']})`
— there is no `?next=`/redirect-target support today [VERIFIED: direct read of
`solsys_code/solsys_code_observatory/views.py`]. A commented-out URL pattern
(`path('create/<str:mpccode>/', ...)`) in `urls.py` shows an earlier, unused design for
obscode prefill that also isn't wired up.
**How to avoid:** This phase needs a small, explicit extension to `CreateObservatory`: read
`request.GET.get('obscode')` as the form's `initial` value, and read `request.GET.get('next')`
(validated with Django's `django.utils.http.url_has_allowed_host_and_scheme`) as the redirect
target in `get_success_url()`, falling back to the existing detail-page behavior when `next` is
absent. This is a real, scoped code change beyond "just reuse the existing view" — flag it
explicitly in the plan's `files_modified` for `solsys_code_observatory/views.py`.
**Warning signs:** A plan that lists `CreateObservatory` under "no changes needed, reused as-is"
will produce a UX where staff creating a new Observatory get bounced to its detail page and
have to navigate back to the approval queue manually, and lose their place if there were
multiple unresolved rows.

## Code Examples

Verified patterns from official sources and this session's live testing:

### Fetching the full MPC obscode list (verified live, 2026-07-11)
```python
# Source: verified live this session against https://data.minorplanetcenter.net/api/obscodes
import requests

response = requests.get(
    'https://data.minorplanetcenter.net/api/obscodes',
    json={},  # omitting 'obscode' entirely triggers the bulk-list mode
    timeout=30,
)
data = response.json()
len(data)          # 2710 (2026-07-11 snapshot)
data['250']['name_utf8']   # 'Hubble Space Telescope'
data['250']['longitude']   # None -- satellite-type records have no fixed geodetic position
```

### `difflib.get_close_matches` against the widened pool (verified live, 2026-07-11)
```python
# Source: verified live this session, real corpus codes from 18-DECISION.md Criterion 4
import difflib
matches = difflib.get_close_matches('X09', candidate_strings, n=5, cutoff=0.6)
# ['X09', 'Z09', 'X93', 'X91', 'X90'] -- exact hit first (self-match), plus near-miss codes
matches = difflib.get_close_matches('DCT', candidate_strings, n=8, cutoff=0.6)
# [] -- acronym/nickname gap, confirmed not bridgeable by difflib even at cutoff=0.3
```

### `open_to_collaboration`-style migration (mirror for the VIEW-05 opt-in field)
```python
# Source: solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py
# (existing shape to mirror for the new opt-in field's migration)
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0006_campaignrun_original_obs_date_raw_and_window_needs_review'),
    ]

    operations = [
        migrations.AddField(
            model_name='campaignrun',
            name='contact_public_opt_in',  # naming: Claude's discretion (D-07); this is a
                                            # recommended concrete choice mirroring
                                            # open_to_collaboration's "state describes intent"
                                            # naming style
            field=models.BooleanField(
                default=False,
                verbose_name='Show contact info publicly?',
            ),
        ),
    ]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Fuzzy-match candidate pool = local `Observatory` table only (8 rows) | Widened to the live MPC Obscodes bulk list (2,710 codes), cached | This phase (D-01) | Fixes the Phase 18 spike's documented "too narrow a pool" finding — but see Pitfall 1/2: most value is for near-typo cases against valid codes, not acronym translation. |
| `CampaignRunDecisionView.post()` always calls `resolve_site()` on approve | Skips `resolve_site()` when `run.site` is already set | This phase (D-06/SITE-03) | Closes an already-reachable clobbering bug (Pitfall 3), not just a hypothetical one introduced by the new UI. |
| `contact_person`/`contact_email` always staff-only on the per-campaign table | Per-row conditional public exposure via a single opt-in flag | This phase (VIEW-05) | First per-row (not per-campaign or per-user-role) PII exposure rule in this codebase — sets the `Case`/`When` pattern precedent for any future similar need. |

**Deprecated/outdated:** None — no library or pattern in this phase replaces a previously
deprecated approach; all three sub-features are net-new capability on top of stable existing
code.

## Runtime State Inventory

Not applicable — this is a feature-addition phase (new field, new UI, new bug fix), not a
rename/refactor/migration phase. No existing stored data, live service config, OS-registered
state, secrets, or build artifacts reference anything being renamed or moved. The one schema
change (VIEW-05's new `BooleanField`) is purely additive (`default=False`), needing no backfill
or data migration — every existing `CampaignRun` row correctly defaults to the opted-out state
that already describes its actual behavior today.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The recommended opt-in field name `contact_public_opt_in` and verbose label "Show contact info publicly?" (D-07 explicitly left to Claude's discretion) | Code Examples, Architecture Patterns | Low — purely cosmetic; easy to rename during planning/discussion if the user prefers different wording. No functional impact either way. |
| A2 | A 24-hour TTL for the MPC candidate-pool cache is an appropriate balance (long enough to avoid repeated 1.3s bulk fetches, short enough that a newly-registered MPC site becomes matchable within a day) | Architecture Patterns Pattern 1 | Low-Medium — if wrong, staff might not see a very-recently-registered MPC site in the dropdown for up to 24h; the free-text/create-new fallback (D-05) still works regardless, so this never blocks resolution, only delays dropdown convenience. Easy to tune via a single constant. |
| A3 | `n=5` (candidate count) and `cutoff=0.6` (difflib default) are reasonable defaults for the dropdown's candidate count, per D-01's "not discussed -- use judgement, consistent with difflib defaults" discretion note | Code Examples | Low — CONTEXT.md explicitly delegates this to Claude's discretion; `cutoff=0.6` is difflib's own documented default and was the exact value used in Phase 18's spike, so no norm is being invented. |

**If this table is empty:** N/A — three low-risk discretionary claims are logged above; none
require user confirmation before planning proceeds (all three are already explicitly flagged
in CONTEXT.md as "Claude's discretion").

## Open Questions

1. **Should the per-row "opted in" state get a visible indicator in the approval queue itself?**
   - What we know: CONTEXT.md explicitly lists this as Claude's discretion, unresolved during
     discuss-phase.
   - What's unclear: Whether staff reviewing a pending submission benefit from seeing "this
     submitter opted into public contact display" before approving, vs. it being purely a
     per-campaign-table (post-approval) display concern.
   - Recommendation: Given the approval queue's `ApprovalQueueTable` already excludes the three
     post-observation columns for a triage-focused view (weather/observation_outcome/
     publication_plans, per the UAT-14 gap-closure precedent in `campaign_tables.py`), adding
     another column here would work against that established "triage-focused, minimal columns"
     design intent. Recommend: no new approval-queue column: the opt-in state only affects the
     `CampaignRunTable` on the per-campaign page. If a plan-checker or discuss-phase disagrees,
     it's a small, additive one-column change, not a redesign.

2. **Exact candidate-string composition for the fuzzy-match pool (D-01's local ∪ MPC merge)**
   - What we know: The local `Observatory` table's existing candidate composition
     (`obscode`, `name`, `short_name`, `old_names`) from Phase 18's spike; the live MPC bulk
     payload's field names (`obscode`, `name_utf8`, `short_name`, `old_names`) confirmed this
     session.
   - What's unclear: Whether `old_names` (a free-text field, sometimes multi-value/
     newline-or-comma-delimited in the live data, confirmed mostly `None` in this session's
     spot-checks) should be split into separate candidate strings or included as one long
     string (which would rarely score above `cutoff=0.6` against a short submitted code
     anyway).
   - Recommendation: Include `old_names` as one whole string (not split) for simplicity — spot
     checks this session found it `None` for the vast majority of records including all of
     `250`/`274`/`289`/`X09`/`N50`/`X07`/`C65`/`DCT`'s real MPC counterpart, so its practical
     impact on match quality is low either way. Revisit only if a real submitted site text is
     found that only matches via a specific historical name.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| MPC Obscodes API (`data.minorplanetcenter.net`) | D-01/D-02 bulk candidate fetch | ✓ [VERIFIED: live `curl` this session, 2026-07-11, 200 OK, 2,710 codes, ~1.3s] | — (external HTTP API, no version) | Network failure at fetch/refresh time: fall back to the local-only `Observatory` candidate pool (matches `resolve_site()`'s existing `requests.exceptions.RequestException` -> fall-through-and-continue precedent) rather than blocking the approval-queue page render. |
| Django cache backend (`FileBasedCache`) | D-02 candidate-pool caching | ✓ [VERIFIED: `src/fomo/settings.py` `CACHES` block, already in production use by `campaign_gap.py`] | Django-bundled | None needed — already configured and already used identically elsewhere in this codebase. |
| `difflib` (stdlib) | Fuzzy matching | ✓ [VERIFIED: bundled with Python 3.10-3.12, already used by Phase 18's probe script] | bundled | None needed — stdlib. |

**Missing dependencies with no fallback:** none.

**Missing dependencies with fallback:** MPC API network failure — fall back to local-only
candidate pool (see above); still never blocks approval (matches the existing `create_placeholder=False`
"flag for review, don't block" discipline).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (`django.test`) via `./manage.py test` -- **not** pytest (this repo's pytest suite (`testpaths = ["tests","src","docs"]`) does not collect `solsys_code/tests/`, per CLAUDE.md "Testing" section) |
| Config file | none dedicated -- driven by `src/fomo/settings.py` (`DJANGO_SETTINGS_MODULE`) |
| Quick run command | `./manage.py test solsys_code.tests.test_campaign_approval` (site-resolution/clobber-guard changes) and `./manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_forms solsys_code.tests.test_campaign_submission` (VIEW-05 opt-in changes) |
| Full suite command | `./manage.py test solsys_code` (332 tests as of Phase 20's completion per `260705-l1v-SUMMARY.md`) plus `ruff check .` / `ruff format --check .` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SITE-01 | Unresolved pending row's Site column presents fuzzy-matched candidates (via a rendered `<datalist>`) | unit | `./manage.py test solsys_code.tests.test_campaign_approval.TestSiteFuzzyMatch` | ❌ Wave 0 -- new test class |
| SITE-02 | Staff-typed/selected code resolves to an existing Observatory or explicit create-new; no placeholder ever auto-fabricated | unit | `./manage.py test solsys_code.tests.test_campaign_approval.TestApprovalSiteResolution` (extend existing class) | ✅ existing class, add cases |
| SITE-03 | Approving a run with `run.site` already set does not call `resolve_site()` again | unit | `./manage.py test solsys_code.tests.test_campaign_approval.TestApproval` (extend, per Pitfall 3's regression scenario) | ✅ existing class, add cases |
| VIEW-05 | Opted-in run's `contact_person`/`contact_email` visible to anonymous visitors on the per-campaign table; opted-out stays staff-only | unit | `./manage.py test solsys_code.tests.test_campaign_views` | ✅ existing file, add cases |
| VIEW-05 | Submission form's opt-in checkbox persists onto the created `CampaignRun` | unit | `./manage.py test solsys_code.tests.test_campaign_submission solsys_code.tests.test_campaign_forms` | ✅ existing files, add cases |

### Sampling Rate
- **Per task commit:** the relevant module's quick-run command above (never the full 332-test suite per commit)
- **Per wave merge:** `./manage.py test solsys_code`
- **Phase gate:** Full suite green (`./manage.py test solsys_code`) plus `ruff check .`/`ruff format --check .` before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `TestSiteFuzzyMatch` class in `solsys_code/tests/test_campaign_approval.py` -- covers SITE-01 (candidate pool building, `difflib` invocation, datalist rendering), mocking `MPCObscodeFetcher`'s bulk-fetch method (never hit the real live API in tests -- mirror the existing `@patch('requests.get', ...)` pattern already used elsewhere in this test module for tier-2 mocking)
- [ ] A regression test for Pitfall 3's exact reproduction sequence (approve succeeds at site-resolution but fails at calendar projection, reverts to PENDING_REVIEW, approve again, assert `resolve_site`/underlying `requests.get` is NOT called a second time)
- [ ] Test fixtures for the bulk MPC response shape -- a small hand-built dict mirroring the real `{obscode: {name_utf8, short_name, old_names, observations_type, longitude, ...}}` shape (no need to fixture all 2,710 real entries; 5-10 representative entries suffice, including one `observations_type='satellite'` entry with `longitude: None` to exercise Pitfall 4's existing-bug-preserving behavior)
- [ ] Framework install: none -- Django `TestCase` already the established framework for this module

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Unchanged this phase -- `StaffRequiredMixin` already gates the approval queue/decision endpoints; the submission form remains intentionally anonymous-accessible (public intake, unchanged). |
| V3 Session Management | no | Unchanged -- no new session state introduced. |
| V4 Access Control | yes | `StaffRequiredMixin` (existing, unchanged) continues to gate `CampaignRunDecisionView`/`ApprovalQueueView`; the new `site_selection` POST field only has effect when combined with staff-only access to the endpoint that reads it -- never exposed on the public submission form. |
| V5 Input Validation | yes | `resolve_site()`'s existing length-guard (`_MAX_OBSCODE_LEN`) and blank-check apply unchanged to the new `site_selection` free-text field (it's passed through the same function, not a new validation path); the new opt-in `BooleanField` needs no custom validation (Django `forms.BooleanField` is inherently binary). |
| V6 Cryptography | no | No new cryptographic material this phase. |
| V8 Data Protection | yes | The core VIEW-05 concern -- per-row PII (`contact_person`/`contact_email`) exposure gated at the SQL `SELECT` level via `Case`/`When` (Pattern 5), preserving the existing "restrict the queryset, not just the template" invariant (15-RESEARCH.md Pitfall 1) that this phase must not regress. |

### Known Threat Patterns for {stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Stored XSS via submitter-controlled free text (`site_raw`, a fuzzy-matched candidate's `name_utf8` sourced from the external MPC API) rendered into the new `<input value="...">`/`<datalist><option>` markup | Tampering | `format_html`/`format_html_join` with positional arguments (Django auto-escapes), never `mark_safe` or f-string/`%`-interpolation of untrusted text -- exact existing pattern already used by `render_window_start()`'s tooltip (T-20-03 precedent); apply identically to the new site-input rendering. |
| Over-broad PII exposure if the `Case`/`When` annotation is misapplied (e.g. condition inverted, or applied only in the template rather than the queryset) | Information Disclosure | Verify with a DB-backed test asserting the raw `.values()` dict itself (not just the rendered HTML) never contains a non-empty `contact_person`/`contact_email` string for an opted-out row -- mirrors the existing `TestApprovalQueueColumns`-style column-content assertions already in this test module. |
| CSRF on the refactored single-`<form>` `render_actions()` | Tampering | Unchanged mechanism -- `get_token(self.request)` continues to mint a real CSRF token per row's form; the refactor from two forms to one does not remove or weaken this, since both the old and new shape mint exactly one token per form instance. |
| A staff member submitting an oversized/malformed `site_selection` value (accidentally pasting a huge block of text into the free-text input) | Tampering / Denial of Service | Already handled by `resolve_site()`'s existing length guard (`len(code) > _MAX_OBSCODE_LEN` -> immediate flag, no tier attempted, no network call) -- no new guard needed since `site_selection` is routed through the same function. |

## Sources

### Primary (HIGH confidence)
- `solsys_code/campaign_views.py`, `campaign_tables.py`, `campaign_utils.py`, `campaign_forms.py`,
  `models.py`, `solsys_code_observatory/utils.py`, `solsys_code_observatory/views.py`,
  `solsys_code_observatory/models.py`, `solsys_code_observatory/forms.py` -- direct reads this
  session, exact current behavior of `resolve_site()`, `CampaignRunDecisionView.post()`,
  `render_site()`/`render_actions()`, `MPCObscodeFetcher`, `CreateObservatory`.
- Live `curl`/Python verification this session against `https://data.minorplanetcenter.net/api/obscodes`
  (bulk-fetch mode, 2,710 codes, field shapes, `250`/`274`/`289`/`X09`/`N50`/`X07`/`C65`/`DCT`
  real lookups) -- direct network calls, not documentation-sourced.
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` -- the locked
  fuzzy-match library choice and the exact D-09 corpus this session's live re-verification builds on.
- `solsys_code/tests/test_campaign_approval.py` -- existing test class/fixture conventions
  (`CampaignApprovalTestBase`, `@patch` usage patterns).
- `solsys_code/migrations/0002_campaignrun.py`, `0006_campaignrun_original_obs_date_raw_and_window_needs_review.py`
  -- exact migration shape to mirror for the new opt-in field.

### Secondary (MEDIUM confidence)
- `https://www.minorplanetcenter.net/mpcops/documentation/obscodes-api/` [CITED, via WebFetch
  this session] -- confirms the bulk-query capability is documented API behavior, not an
  undocumented quirk this session happened to discover.
- MDN `<datalist>`/`<input form=...>` documentation [CITED] -- confirms cross-browser support
  for the HTML5 mechanism Pattern 4 relies on.

### Tertiary (LOW confidence)
- None -- every claim in this research is either a direct code read, a live-verified API call,
  or a cited official-documentation reference this session.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies; both `difflib` and Django's cache framework are
  already locked/established in this exact codebase.
- Architecture: HIGH -- every pattern (cache reuse, `Case`/`When`, `form=` attribute, bulk MPC
  fetch) was verified live this session, not assumed from training data.
- Pitfalls: HIGH -- Pitfalls 1-3 are freshly-discovered, live-verified findings (not
  documentation-sourced guesses) that materially change how the planner should scope SITE-01/03.

**Research date:** 2026-07-11
**Valid until:** 2026-08-10 (30 days -- the MPC obscode candidate count/specific codes will
drift over time, but the architectural findings and pitfalls are stable regardless)
