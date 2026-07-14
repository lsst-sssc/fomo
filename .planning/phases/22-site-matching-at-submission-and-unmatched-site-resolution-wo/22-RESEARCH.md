# Phase 22: Site Matching at Submission and Unmatched-Site Resolution Workflow - Research

**Researched:** 2026-07-14
**Domain:** Django/HTMX live-search UI + post-approval workflow extension (no new external dependencies)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Live-search endpoint (access, protection, response shape)**
- **D-01:** The endpoint is **open to anonymous users** — it backs the public submission form; the candidate pool is public MPC data.
- **D-02:** Abuse protection is a **zero-dependency per-IP throttle** built on the existing Django cache framework (the same cache `build_site_candidates()` already uses): a small counter (~10 lines) rejecting with HTTP 429 over the limit. No `django-ratelimit`, no DRF APIView. Exact rate (e.g. 30–60 req/min) is Claude's discretion during planning.
- **D-03:** The endpoint returns a **rendered HTML fragment** (HTMX convention: server-side partial template, `hx-get` swap). No JSON API, no custom JS — `django_htmx` is already installed.

**Matching strategy & selection semantics**
- **D-04:** Matching is **substring-first with difflib fallback**: case-insensitive containment over the cached candidate pool (typing `faulkes` → both "Haleakala-Faulkes Telescope North" (F65) and "Siding Spring-Faulkes Telescope South" (E10); `lowell` → all Lowell sites), falling back to `difflib.get_close_matches` for typo tolerance only when containment finds nothing. Implemented as a new small helper alongside `fuzzy_match_candidates()` — the existing difflib-only helper is NOT sufficient (whole-string similarity scores short partial queries like 'Faulkes' vs long official MPC strings below the 0.6 cutoff).
- **D-05:** Suggestions render as **"Display Name (obscode)"** so the submitter sees the resolved MPC site pre-submit.
- **D-06:** Picking a suggestion **fills `site_raw` as text only; resolution stays at approval**. The picked display string is guaranteed exact-matchable at approve time via the existing pool-mapping → obscode → `resolve_site()` flow (`CampaignRunDecisionView` CR-01 path). Zero model changes, no migrations, and no DB writes triggered by anonymous traffic (resolve_site tier 2 creates Observatory rows — that must never fire from a public submission).

**Post-approval resolution surface & action**
- **D-07:** "Sites needing review" is a **third table on the existing approval-queue page** (pending / decided / sites-needing-review), listing approved runs with `site_needs_review=True`. Reuses the `ApprovalQueueTable` machinery and the page's once-per-request candidate pool. No new page or navbar entry.
- **D-08:** Resolution is a **new `resolve_site` action on `CampaignRunDecisionView.post()`** alongside approve/reject. It resolves via the existing display-string→obscode pool mapping and `resolve_site(..., create_placeholder=False)`, honoring the Phase 21 D-06 never-re-resolve guard, then **fires the calendar projection in the same request**. The projection block currently inlined in the approve branch is factored into a shared helper so approve and resolve_site both call it (same single-night + resolved-site + telescope preconditions; range/TBD runs simply clear the flag and get no event, per the existing rule).

**Widget behavior**
- **D-09:** The public form gets **no "Create new Observatory" link** — site creation stays a vetted staff action (approval queue + existing `CreateObservatory` flow). Public submitters just type free text when nothing matches; free text never blocks submission.
- **D-10:** The queue's inline site input keeps its "Create new Observatory" link and gains the same live-search widget (replacing the static ≤5-candidate datalist), so staff typing something different from the original `site_raw` get live suggestions.

### Claude's Discretion
- Exact throttle rate and cache-key scheme (D-02).
- Live-search fine-tuning: minimum characters before searching (~2), `hx-trigger` debounce delay (~300 ms), suggestion count cap (~8), "no matches — free text is fine" hint copy, and how a picked suggestion populates the input.
- Endpoint URL naming and whether the queue and form share one endpoint or pass a context flag.
- Resolve-failure UX in the sites-needing-review table (row stays with an error message; exact copy TBD).
- Whether the sites-needing-review table caps row count / orders by recency (mirror the decided table's -pk convention unless there's a reason not to).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.

Not in scope: changing the approval decision flow itself (approve/reject semantics), submitter self-service editing, satellite/occultation/radar projection support, or `Observatory.obscode` schema changes.
</user_constraints>

<phase_requirements>
## Phase Requirements

No formal `REQUIREMENTS.md` IDs are mapped to this phase — it closes a functionality gap identified after Phase 21 shipped (v2.1's 13 v1 requirements are all already `Complete`; see `.planning/REQUIREMENTS.md` traceability table and `.planning/STATE.md` "Roadmap Evolution"). CONTEXT.md's locked decisions (D-01 through D-10, above) are the operative acceptance criteria for this phase; use them as the requirement identifiers in the Validation Architecture test map below.

| ID | Description | Research Support |
|----|-------------|------------------|
| D-01..D-03 | Live-search endpoint: anonymous, throttled, HTML-fragment response | Architecture Patterns (HTMX fragment endpoint), Security Domain |
| D-04..D-06 | Substring-first+difflib matching, "Display Name (obscode)" rendering, text-only fill | Architecture Patterns (substring-first matcher), Code Examples |
| D-07..D-08 | Sites-needing-review table + `resolve_site` decision action + shared projection helper | Architecture Patterns (shared calendar-projection helper), Code Examples |
| D-09..D-10 | Widget placement differences (form vs. queue) | Architecture Patterns, Code Examples |
</phase_requirements>

## Summary

This phase adds no new dependencies — `django-htmx` 1.23.2 is already installed and its
`htmx.min.js` asset is already loaded on every page via `tom_common/base.html`
`[VERIFIED: pip show django-htmx / installed tom_common base.html]`; `difflib` is stdlib and
already used by `fuzzy_match_candidates()`; the Django cache framework (currently
`FileBasedCache`) already backs `build_site_candidates()`'s 24h pool cache
`[VERIFIED: src/fomo/settings.py CACHES]`. The work is entirely new/extended Python in three
existing files (`campaign_utils.py`, `campaign_views.py`, `campaign_tables.py`) plus one new
view, one new URL, one new partial template, and template edits to the two existing forms.

The riskiest new surface is the live-search endpoint itself: it is the **first anonymous,
unauthenticated endpoint in this codebase that runs a substring/fuzzy scan over the full
merged candidate pool (~5,000+ strings from the MPC bulk obscode list) on every keystroke**.
Phase 21's DoS mitigation for the same pool (`T-21-04`) only had to protect a staff-gated page
load; this phase's endpoint is reachable by anyone, repeatedly, with no login. D-02's per-IP
throttle is therefore load-bearing, not optional polish — it is the single control standing
between "public search box" and "unauthenticated request amplification against an in-memory
scan." The plan should treat it with the same weight as the substring-matching logic itself.

The second risk area is correctly factoring `CampaignRunDecisionView.post()`'s calendar
projection so `approve` keeps its existing revert-on-failure discipline (Phase 21 P04:
"Kept the except Exception revert block byte-for-byte unchanged") while the new `resolve_site`
action gets its own **non-reverting** failure path — the run is already `APPROVED`; a failed
projection must never un-approve it, only report a friendly retry message. Getting this
extraction wrong (e.g. re-wrapping `resolve_site` in the same broad revert-to-PENDING_REVIEW
`except Exception` used for `approve`) would silently break "site failure never blocks
approval" for the exact rows this phase exists to fix.

**Primary recommendation:** One shared GET-only live-search endpoint (`campaigns:site_search`)
returning an HTML fragment, driven by `hx-trigger="keyup changed delay:300ms[this.value.length >= 2]"`
(htmx has no `hx-min-length` attribute — verified against official docs, see Pitfall 1), gated
by a ~10-line `cache.add()`/`cache.incr()` per-IP throttle; a new `substring_or_fuzzy_match_candidates()`
helper in `campaign_utils.py` sitting alongside (not replacing) `fuzzy_match_candidates()`; and a
`_project_calendar_event(run)` helper extracted from the approve branch's existing inline block,
called from both `approve` (inside its unchanged revert-on-failure `try/except`) and the new
`resolve_site` action (inside its own non-reverting `try/except`).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Substring-first + difflib-fallback matching algorithm | API / Backend | — | Pure Python function in `campaign_utils.py`; no I/O beyond the already-cached pool dict |
| Live-search HTTP endpoint (throttle, query handling) | API / Backend | — | New `View` in `campaign_views.py`; owns request validation and the per-IP counter |
| Site-candidate pool caching | Database / Storage | API / Backend | Django cache framework (`FileBasedCache`) is the persistence/caching layer; `build_site_candidates()` (backend) is the sole writer/reader — unchanged this phase |
| Suggestion-list HTML rendering | Frontend Server (SSR) | — | New Django partial template, server-rendered fragment returned to `hx-get` |
| Debounced live-search interaction (typing → request) | Browser / Client | — | Pure HTML attributes (`hx-trigger`, `hx-get`, `hx-target`) — no custom JS per D-03 |
| Suggestion pick → fill `site_raw` text | Browser / Client | — | Inline `onclick`/`hx-on:click` setting the input's `.value`, mirroring the existing inline-`onclick` convention already used in `ApprovalQueueTable.render_actions()`'s Reject-confirm button |
| "Sites needing review" queryset + table | API / Backend | Frontend Server (SSR) | `ApprovalQueueView.get_context_data()` builds the queryset; `ApprovalQueueTable` (reused) renders it |
| `resolve_site` decision action + calendar projection | API / Backend | Database / Storage | `CampaignRunDecisionView.post()`; writes `CampaignRun.site`/`site_needs_review` and (via `insert_or_create_calendar_event`) `CalendarEvent` rows |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `django-htmx` | 1.23.2 (already installed) | `request.htmx` detection; `htmx.min.js` already loaded on every page via `tom_common/base.html` | Already the project's chosen HTMX integration (Phase 22 supersedes Phase 21 D-04's "no new endpoint" scoping, not the library choice) `[VERIFIED: pip show django-htmx; tom_common/templates/tom_common/base.html]` |
| `difflib` (stdlib) | Python 3.10+ stdlib | Typo-tolerant fallback inside the new substring-first matcher | Locked by Phase 18 D-04 spike decision (`rapidfuzz` deferred until a real advantage is demonstrated); already used by `fuzzy_match_candidates()` `[VERIFIED: 18-DECISION.md; solsys_code/campaign_utils.py]` |
| `django.core.cache` | Django 5.2.15 (installed) | Site-candidate pool cache (existing, unchanged) + new per-IP throttle counter | Zero-dependency per D-02; `cache.add()`/`cache.incr()` confirmed present and behave as documented by direct inspection of the installed Django source `[VERIFIED: django.core.cache.backends.base.BaseCache / filebased.FileBasedCache source, installed 5.2.15]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `django.utils.html.format_html` / `format_html_join` | Django 5.2.15 (installed) | Auto-escaped HTML fragment construction for suggestion `<li>`/`<button>` markup | Every place untrusted MPC/local candidate strings are interpolated into HTML — same convention already used in `campaign_tables.py`'s `render_site()`/`render_actions()` (T-21-01 precedent) |
| `django_tables2` | already installed | Reuse `ApprovalQueueTable` for the "Sites needing review" table | D-07 explicitly reuses the existing table machinery rather than a new table class |
| `crispy_forms` | already installed | `CampaignRunSubmissionForm`'s existing Layout; inject the suggestions container via `HTML(...)` next to `site_raw` | Consistent with the form's existing crispy-forms `Layout` construction |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled `cache.add()`/`cache.incr()` throttle | `django-ratelimit` or DRF `APIView` + `SimpleRateThrottle` | Rejected by D-02 — extra dependency for ~10 lines of logic; DRF isn't otherwise used in this app |
| `difflib` substring-first matcher | `rapidfuzz` | Rejected — Phase 18 D-04 explicitly deferred `rapidfuzz` until a real demonstrated advantage; this phase's containment-first approach solves the acronym/partial-match gap without a new library |
| Server-rendered HTML fragment (`hx-get` swap) | JSON API + client-side JS rendering | Rejected by D-03 — no custom JS, matches the project's existing zero-JS-framework posture |

**Installation:**
No new packages required — `django-htmx`, `difflib` (stdlib), and `django.core.cache` are already present in the environment.

**Version verification:**
```bash
pip show django-htmx   # Version: 1.23.2
pip show django        # Version: 5.2.15
```
Both confirmed installed and current in this project's virtualenv at research time.

## Package Legitimacy Audit

**No new external packages are installed by this phase.** `django-htmx` is a pre-existing
project dependency (already in `pyproject.toml`/environment, used since before this phase);
`difflib` is Python stdlib. Per the Phase 21 precedent (`T-21-SC`/`AR-21-03`: "No packages
installed this phase — no supply-chain checkpoint required"), the same disposition applies
here. The `package-legitimacy check` gate and its table are omitted — there is nothing to
audit.

## Architecture Patterns

### System Architecture Diagram

```
[Public submission form]              [Approval queue: pending row]     [Approval queue: sites-needing-review row]
   site_raw <input hx-get>               site_selection <input hx-get>      site_selection <input hx-get>
        |  keyup (debounced, >=2 chars)       |  keyup (debounced)              |  keyup (debounced)
        v                                     v                                v
   +------------------------------------------------------------------------------------------+
   |  GET /campaigns/site-search/?q=<text>&input_id=<dom-id>   (campaigns:site_search, anon OK) |
   |    1. per-IP throttle check (cache.add/incr; 429 if over limit)                            |
   |    2. build_site_candidates()  <-- 24h-cached merged local+MPC pool (unchanged, no re-fetch)|
   |    3. substring_or_fuzzy_match_candidates(q, pool)  <-- new: containment first, difflib     |
   |       fallback only if containment finds nothing                                            |
   |    4. render partial template -> HTML fragment: "Display Name (obscode)" list items         |
   +------------------------------------------------------------------------------------------+
        |  HTML fragment (hx-swap=innerHTML into #site-suggestions-<id>)
        v
   [Suggestion list rendered under the input]
        |  click a suggestion (inline onclick / hx-on:click sets input.value; no round trip)
        v
   [site_raw / site_selection now holds "Display Name (obscode)" text -- unresolved until approval/resolve]

   --- separate flow: post-approval resolution ---

[ApprovalQueueView.get_context_data()]
   pending_qs   -> ApprovalQueueTable (existing, Approve/Reject actions)
   decided_qs   -> ApprovalQueueTable (existing, read-only)
   review_qs = CampaignRun.objects.filter(approval_status=APPROVED, site_needs_review=True)
             -> ApprovalQueueTable (reused, "Resolve" action only)   <-- NEW third table (D-07)
        |
        v  staff types/picks a site, submits the row's decide-form (action=resolve_site)
   CampaignRunDecisionView.post()
        1. action == 'resolve_site' branch (NEW, alongside approve/reject)
        2. guard: run.site is None (D-06 never-re-resolve, mirrors approve's existing guard)
        3. selection -> build_site_candidates().get(selection, selection) -> obscode
        4. resolve_site(obscode, create_placeholder=False) -> (site, needs_review)
        5. save run.site / run.site_needs_review
        6. if resolved: _project_calendar_event(run)  <-- SHARED helper, also called by approve
           - non-reverting try/except (unlike approve's revert-to-PENDING_REVIEW)
        7. redirect back to campaigns:approval_queue with a messages.* banner
```

### Recommended Project Structure
```
solsys_code/
├── campaign_utils.py         # + substring_or_fuzzy_match_candidates(), throttle helper (or inline in view)
├── campaign_views.py         # + SiteSearchView; CampaignRunDecisionView.post() gains resolve_site branch
│                              #   + shared _project_calendar_event(run) module-level helper
├── campaign_tables.py        # ApprovalQueueTable.render_site()/render_actions() gain resolve-mode support
├── campaign_forms.py         # CampaignRunSubmissionForm.site_raw widget gains hx-* attrs
└── campaign_urls.py          # + path('site-search/', SiteSearchView.as_view(), name='site_search')

src/templates/campaigns/
├── campaignrun_submit_form.html   # unchanged structurally; suggestions container injected via crispy Layout HTML()
├── approval_queue.html            # + third {% render_table %} block for "Sites needing review"
└── partials/                      # NEW directory (mirrors src/templates/solsys_code/partials/)
    └── site_search_results.html   # NEW: the hx-get fragment template, shared by both callers
```

### Pattern 1: Shared GET-only HTML-fragment live-search endpoint
**What:** A single `View` (not `FormView`/`TemplateView`) handling `GET` only, no auth
required, returning a small rendered partial — never JSON.
**When to use:** Any place D-03/D-10 need live suggestions (both the public form and the
approval-queue rows share this one endpoint, differentiated only by a `input_id` GET param
the fragment echoes back into each suggestion's click-to-fill target, per the "Claude's
Discretion" note on whether to share one endpoint).
**Example:**
```python
# Source: pattern derived from this project's own request.htmx usage
# (solsys_code/views.py:156) + django-htmx 1.23.2 docs conventions.
class SiteSearchView(View):
    http_method_names = ['get']

    def get(self, request):
        client_ip = request.META.get('REMOTE_ADDR', '')
        if not _check_and_increment_throttle(client_ip):
            return HttpResponse(status=429)
        query = request.GET.get('q', '')
        input_id = request.GET.get('input_id', 'id_site_raw')
        candidates = substring_or_fuzzy_match_candidates(query, build_site_candidates())
        return render(
            request,
            'campaigns/partials/site_search_results.html',
            {'candidates': candidates, 'input_id': input_id, 'query': query},
        )
```
`django.http.HttpResponse(status=429)` is the correct primitive — Django 5.2.15 has **no**
`HttpResponseTooManyRequests` class (confirmed by listing `django.http`'s exported names in
the installed environment; see Pitfall 4).

### Pattern 2: Substring-first, difflib-fallback matcher (D-04)
**What:** New helper alongside (not replacing) `fuzzy_match_candidates()`.
**When to use:** Every live-search call site; `fuzzy_match_candidates()` itself stays
unchanged so `ApprovalQueueTable.render_site()`'s existing static-datalist call site (being
replaced by the live widget this phase, but harmless if anything else still calls it)
continues to work identically.
**Example:**
```python
# Source: derived directly from this repo's campaign_utils.py conventions
# (docstring style, "never raise" discipline, existing fuzzy_match_candidates()).
def substring_or_fuzzy_match_candidates(
    site_raw: str, candidate_pool: dict[str, str], *, limit: int = 8
) -> list[tuple[str, str]]:
    """Substring-first, difflib-fallback site match (D-04).

    Case-insensitive containment over candidate_pool first (bridges short partial queries
    like 'Faulkes' against long official MPC strings that difflib's 0.6 cutoff can't reach);
    falls back to fuzzy_match_candidates() for typo tolerance only when containment finds
    nothing at all. Never raises.
    """
    text = (site_raw or '').strip()
    if not text:
        return []
    needle = text.lower()
    hits = [(candidate, obscode) for candidate, obscode in candidate_pool.items() if needle in candidate.lower()]
    if hits:
        hits.sort(key=lambda pair: (len(pair[0]), pair[0]))  # shortest/most-specific first
        return hits[:limit]
    return fuzzy_match_candidates(text, candidate_pool)[:limit]
```
Note: `fuzzy_match_candidates()`'s `n=5` is currently hardcoded
(`solsys_code/campaign_utils.py:327`). If `limit` should exceed 5 in the fallback branch too,
add an optional `n: int = 5` parameter to `fuzzy_match_candidates()` rather than
reimplementing the difflib call — the existing single call site
(`ApprovalQueueTable.render_site()`, being replaced this phase) is unaffected by a
backward-compatible default.

### Pattern 3: Per-IP throttle via `cache.add()` + `cache.incr()` (D-02)
**What:** A fixed-window counter keyed by client IP, using only `django.core.cache`.
**When to use:** `SiteSearchView.get()`, before any pool scan.
**Example:**
```python
# Source: derived from Django's own documented cache API
# (django.core.cache.backends.base.BaseCache.add/incr, verified against the
# installed Django 5.2.15 source) plus the general per-IP-cache-key pattern
# confirmed by web search against multiple secondary sources.
SITE_SEARCH_THROTTLE_LIMIT = 40  # requests per window; Claude's discretion, within D-02's 30-60/min guidance
SITE_SEARCH_THROTTLE_WINDOW_SECONDS = 60


def _check_and_increment_throttle(client_ip: str) -> bool:
    """Return False (reject) once client_ip exceeds the per-window request count."""
    key = f'site_search_throttle:{client_ip}'
    added = cache.add(key, 1, timeout=SITE_SEARCH_THROTTLE_WINDOW_SECONDS)
    if added:
        return True
    try:
        count = cache.incr(key)
    except ValueError:
        # Key expired between add() and incr() (race) -- treat as a fresh window.
        cache.set(key, 1, timeout=SITE_SEARCH_THROTTLE_WINDOW_SECONDS)
        return True
    return count <= SITE_SEARCH_THROTTLE_LIMIT
```
`REMOTE_ADDR` is the only client-IP source available — `settings.py` configures no
`X-Forwarded-For`/proxy-header handling (`ALLOWED_HOSTS = []`, no `USE_X_FORWARDED_HOST`),
consistent with this project's dev-oriented deployment posture. If FOMO is later deployed
behind a reverse proxy, this throttle's IP source will need revisiting — out of scope for
this phase, flagged in Open Questions.

### Pattern 4: Shared, revert-agnostic calendar-projection helper (D-08)
**What:** Extract the approve branch's existing inline projection block (the
`if run.telescope_instrument and run.site and ...:` block in `CampaignRunDecisionView.post()`,
lines ~395-449) into a module-level function that raises on failure and does no
error-handling of its own — callers decide revert vs. non-reverting behavior.
**When to use:** Called from both the `approve` action (inside its existing, unchanged
`except Exception: ... revert to PENDING_REVIEW` block) and the new `resolve_site` action
(inside its own `try/except` that reports failure via `messages.error` but does **not**
change `approval_status` or `site`/`site_needs_review`, which are already correctly saved).
**Example:**
```python
# Source: extracted directly from the existing approve branch in
# solsys_code/campaign_views.py:391-449 (verbatim logic, only wrapped as a function).
def _project_calendar_event(run: CampaignRun) -> None:
    """CAL-01/CAL-02 CalendarEvent projection (D-08). May raise -- callers own error handling.

    Only a single concrete night (window_start == window_end) with a resolved site and a
    telescope_instrument gets projected; a range, TBD run, or unresolved site produces no
    event and is not an error (this function simply returns).
    """
    if not (run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end):
        return
    event_fields = {
        'title': f'{run.campaign.name}: {run.telescope_instrument}',
        'description': run.observation_details,
        'target_list': run.campaign,
        'telescope': run.telescope_instrument,
    }
    if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        event_fields['start_time'] = datetime.combine(run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc)
        event_fields['end_time'] = datetime.combine(run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc)
        insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
        return
    try:
        sunset, sunrise = sun_event(run.site, run.window_start, kind='sun')
    except ValueError:
        logger.debug('sun_event(sun) raised for site=%s date=%s; skipping projection.', run.site, run.window_start)
        return
    event_fields['start_time'] = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    event_fields['end_time'] = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
```
The `resolve_site` action's own guard before calling this helper must re-check
`run.site is None` (D-06 never-re-resolve) exactly like the approve branch already does —
copy the existing guard's comment/rationale, don't restate it differently.

### Anti-Patterns to Avoid
- **`hx-min-length` attribute:** Does not exist in htmx (verified against official docs). Use
  the event-filter bracket syntax instead: `hx-trigger="keyup changed delay:300ms[this.value.length >= 2]"`.
- **Re-wrapping `resolve_site` in the approve branch's broad `except Exception: revert to PENDING_REVIEW`:**
  The run is already `APPROVED` when `resolve_site` runs; reverting it would resurrect a
  decided run back into the pending queue, which is exactly the "dead end" this phase is
  supposed to close, not reintroduce elsewhere.
- **Calling `build_site_candidates()` per suggestion-list render inside a loop:** Build/pass
  the pool once per request (existing Pitfall 5 discipline in `ApprovalQueueView`); the new
  third table must reuse the same `candidate_pool` variable already computed for the other two.
- **JSON response + client-side `fetch()`/JS templating:** Contradicts D-03 explicitly; every
  other HTMX integration point in this codebase (calendar view, TOM Toolkit's own
  `bootstrap_htmx.html`) returns rendered HTML fragments, not JSON.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|--------------|-----|
| Typo-tolerant fuzzy string matching | Custom Levenshtein/edit-distance implementation | `difflib.get_close_matches` (already wrapped by `fuzzy_match_candidates()`) | Locked by Phase 18 D-04; stdlib, already proven against this exact candidate pool shape |
| Debounced search-as-you-type | Custom JS `setTimeout`/`fetch` debounce | htmx `hx-trigger="keyup changed delay:300ms[...]"` | Zero-JS, declarative, already the project's only client-interactivity mechanism (`django-htmx` is installed for exactly this) |
| Per-IP rate limiting | Generic middleware-based rate-limit framework (sliding-window store, LRU eviction, etc.) | `django.core.cache` `add()`/`incr()`, ~10 lines, fixed window | D-02 explicitly forbids `django-ratelimit`/DRF for this; the existing cache backend already does the heavy lifting |
| Escaping untrusted MPC/local site-name strings in HTML | Manual string concatenation / f-string HTML building | `django.utils.html.format_html`/`format_html_join` (or plain Django template auto-escaping) | Established codebase convention (T-21-01, closed in Phase 21); a regression here reopens a closed XSS threat |

**Key insight:** Every piece of this phase's "new" functionality (fuzzy matching, debouncing,
throttling, escaping) already has a proven, in-repo or stdlib solution from Phase 18/21 or
Django itself — the actual net-new work is wiring them together into one endpoint and one
decision-view branch, not inventing new mechanisms.

## Common Pitfalls

### Pitfall 1: `hx-min-length` is not a real htmx attribute
**What goes wrong:** A plan or implementation uses `hx-min-length="2"` expecting htmx to gate
requests on input length, and it silently does nothing (htmx ignores unknown attributes) —
every keystroke fires a request from character 1.
**Why it happens:** Several search-engine summaries of htmx patterns reference
`hx-min-length` as if it were a documented attribute; it is not, per htmx's own
`hx-trigger` reference page.
**How to avoid:** Use htmx's documented event-filter bracket syntax instead:
`hx-trigger="keyup changed delay:300ms[this.value.length >= 2]"`.
**Warning signs:** Network tab shows a request firing on the very first keystroke.

### Pitfall 2: `resolve_site`'s never-re-resolve guard must be re-implemented, not assumed
**What goes wrong:** The `resolve_site` decision action resolves and overwrites an already-set
`run.site` a second time (e.g. two staff members submit the row's form in quick succession),
silently clobbering a correct resolution with a worse one — the exact bug class SITE-03/D-06
closed in Phase 21.
**Why it happens:** The new action is structurally separate code from the approve branch;
copying the *behavior* without copying the *guard* (`if run.site is None:`) is an easy gap.
**How to avoid:** Re-check `run.site is None` (freshly fetched from the DB, not a stale
in-memory object) immediately before calling `resolve_site()` in the new branch, mirroring
the approve branch's existing comment/rationale.
**Warning signs:** A resolved site changes on a second click of "Resolve" for the same row.

### Pitfall 3: FileBasedCache's `incr()` is not atomic across processes
**What goes wrong:** Under concurrent requests from the same IP (e.g. a script hammering the
endpoint, or two browser tabs), the throttle's get-then-set `incr()` can under-count,
allowing slightly more than `SITE_SEARCH_THROTTLE_LIMIT` requests through.
**Why it happens:** `django.core.cache.backends.base.BaseCache.incr()` is implemented as a
plain `get()` + `set()` pair (confirmed by reading the installed Django 5.2.15 source) — not
atomic, unlike Memcached/Redis backends' native `INCR`.
**How to avoid:** Accept it — this project's `CACHES` backend is `FileBasedCache` (single
dev-server deployment, `src/fomo/settings.py`), and a soft-throttle that's occasionally off by
a few requests still achieves D-02's stated goal ("abuse protection", not hard SLA
enforcement). Do not pull in Redis/Memcached just to make this atomic — out of scope.
**Warning signs:** None expected at this project's traffic scale; flag only if load testing
reveals it matters.

### Pitfall 4: `HttpResponseTooManyRequests` does not exist in Django 5.2
**What goes wrong:** Code imports `from django.http import HttpResponseTooManyRequests` and
raises an `ImportError` at module load (which, given this module also gets imported by
`campaign_urls.py`, would break every campaigns URL, not just the new endpoint).
**Why it happens:** Some other frameworks/versions have this convenience class; Django's
`django.http` module does not (confirmed by listing its exported names in the installed
5.2.15 environment).
**How to avoid:** Use `HttpResponse(status=429)` directly.
**Warning signs:** `ImportError: cannot import name 'HttpResponseTooManyRequests'` at
`./manage.py runserver` startup.

### Pitfall 5: Staff typing quickly in the approval queue can trip the same anonymous throttle
**What goes wrong:** If the throttle counts every request to `campaigns:site_search`
regardless of caller, a staff member actively resolving several rows in the queue (D-10's
live-search widget) can hit the same per-IP limit meant for anonymous abuse, and their own
UI stops offering suggestions.
**Why it happens:** D-01/D-02 describe the throttle as protecting the *public, anonymous*
form; D-10 reuses the exact same endpoint for staff. The endpoint doesn't currently
distinguish the two callers.
**How to avoid:** Consider exempting `request.user.is_staff` from the throttle (staff are
already authenticated; there's no anonymous-abuse concern for them), or give staff sessions a
materially higher limit. This wasn't explicitly locked in CONTEXT.md — flagged as a planning
decision, not a re-litigation of D-02's zero-dependency-throttle requirement itself.
**Warning signs:** A staff member reports suggestions "stopping" mid-session while actively
triaging the approval queue.

### Pitfall 6: `reverse_lazy()` must be used for the URL inside the form field's widget attrs
**What goes wrong:** `CampaignRunSubmissionForm.site_raw`'s widget needs the search
endpoint's URL as an `hx-get` attribute value. Using `reverse('campaigns:site_search')` at
class-body/module-import time raises `NoReverseMatch` (or worse, a silent wrong value if
called before `campaign_urls.py` is fully loaded), because Django's URL resolver isn't ready
at import time for every app-loading order.
**Why it happens:** `CampaignRunSubmissionForm` fields are defined as class attributes,
evaluated once at class-definition/import time.
**How to avoid:** Use `reverse_lazy()`, which defers resolution until the value is actually
rendered into a string (matches the existing `success_url = reverse_lazy(...)` pattern
already used in `CampaignRunSubmissionView`).
**Warning signs:** `django.urls.exceptions.NoReverseMatch` at Django startup/first import of
`campaign_forms.py`.

## Code Examples

### Suggestion-fragment partial template (pattern for `site_search_results.html`)
```html
{# Source: pattern derived from this project's existing format_html/auto-escaping
   convention in campaign_tables.py (T-21-01) -- Django template auto-escaping is on
   by default, so {{ candidate }} below is safe without extra escaping calls. #}
{% if candidates %}
<ul class="list-group" id="site-suggestions-list">
  {% for display, obscode in candidates %}
  <li class="list-group-item list-group-item-action"
      style="cursor:pointer;"
      onclick="document.getElementById('{{ input_id }}').value = '{{ display|escapejs }} ({{ obscode|escapejs }})'; document.getElementById('site-suggestions-{{ input_id }}').innerHTML = '';">
    {{ display }} ({{ obscode }})
  </li>
  {% endfor %}
</ul>
{% elif query %}
<p class="text-muted small mb-0">No matches — free text is fine, a staff member will resolve it.</p>
{% endif %}
```
Note the `|escapejs` filter on the values embedded inside the `onclick=` JS-context string —
Django's default HTML auto-escaping is not sufficient inside an inline event-handler
attribute's JS string literal; this is a distinct escaping context from the visible
`{{ display }} ({{ obscode }})` text node, which auto-escapes correctly on its own.

### Form widget wiring (`CampaignRunSubmissionForm.site_raw`)
```python
# Source: pattern derived from this project's existing CampaignRunSubmissionForm structure
# (solsys_code/campaign_forms.py) plus reverse_lazy precedent already used in campaign_views.py.
from django.urls import reverse_lazy

site_raw = forms.CharField(
    max_length=255,
    required=False,
    label='Observing site',
    widget=forms.TextInput(attrs={
        'hx-get': reverse_lazy('campaigns:site_search'),
        'hx-trigger': "keyup changed delay:300ms[this.value.length >= 2]",
        'hx-target': '#site-suggestions-id_site_raw',
        'hx-swap': 'innerHTML',
        'hx-vals': '{"input_id": "id_site_raw"}',
        'autocomplete': 'off',
    }),
)
```
The crispy `Layout` needs a matching `HTML('<div id="site-suggestions-id_site_raw"></div>')`
placed immediately after `'site_raw'` in `CampaignRunSubmissionForm.__init__`'s `Layout(...)`
call — crispy-forms' `HTML()` layout object is the standard way to inject a raw template
snippet between declared fields.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 21 D-04: static per-row `<datalist>` of the top-5 `fuzzy_match_candidates()` results, computed once against the originally-submitted `site_raw` | Phase 22 D-01–D-06: live HTMX endpoint recomputing matches against whatever the user is currently typing, substring-first | This phase | Staff/submitters can now search for a site by typing something *different* from the original `site_raw` (e.g. correcting a submitter's typo); the old datalist only ever showed matches for the text as originally submitted |
| Phase 21: unresolved sites (`site_needs_review=True`) had no UI — dead end after approval | Phase 22 D-07/D-08: "Sites needing review" table + `resolve_site` action + deferred calendar projection | This phase | Closes the only remaining gap in the site-resolution feature; previously the only fix was a manual DB edit |

**Deprecated/outdated:**
- The static per-row datalist rendering in `ApprovalQueueTable.render_site()`
  (`solsys_code/campaign_tables.py:208-246`) is superseded by the live-search widget for the
  pending-queue row case; the "read-only, resolved" fallback rendering in the same method
  (calling `super().render_site(record)`) is unchanged and still needed for resolved/decided
  rows.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | 30–60 req/min (settled on 40/min, 60s fixed window) is an adequate per-IP throttle rate for D-02 | Pattern 3 / Code Examples | If too low, legitimate rapid typing (or staff triaging many rows) gets 429'd; if too high, the anonymous endpoint remains meaningfully scannable. Easy to tune post-launch since it's a single constant. |
| A2 | `REMOTE_ADDR` alone (no `X-Forwarded-For` handling) is the correct client-IP source for this deployment | Pattern 3 | If FOMO is deployed behind a reverse proxy/load balancer in production, every request would appear to come from the proxy's IP, making the per-IP throttle effectively a single global throttle. `settings.py` currently has no proxy-header configuration at all, so this matches the project's current deployment posture, but should be revisited if that changes. |
| A3 | Exempting staff (`request.user.is_staff`) from the anonymous throttle is a reasonable design choice, not explicitly locked by CONTEXT.md | Pitfall 5 | If not implemented, staff could occasionally be rate-limited during their own legitimate use of the D-10 widget in the approval queue — a UX annoyance, not a correctness bug (free-text fallback always still works). |
| A4 | A single shared endpoint (`campaigns:site_search`) serving both the public form and the approval-queue rows, differentiated by an `input_id` GET param, is simpler than two separate endpoints | Pattern 1 | If the two callers' needs diverge later (e.g. different result formatting), a shared endpoint could need an `if` branch — low risk, easy to split later since it's one `View`. |

## Open Questions

1. **Should the throttle exempt authenticated staff?**
   - What we know: D-02 locks the mechanism (zero-dependency cache-based per-IP throttle) but not whether it applies uniformly to staff and anonymous callers.
   - What's unclear: Whether staff hitting their own rate limit during active queue triage is an acceptable tradeoff or a real annoyance worth special-casing.
   - Recommendation: Default to exempting `request.user.is_staff` (Pitfall 5/A3) unless the planner decides the added `if` branch isn't worth it for a first cut — either is defensible; flag for `/gsd-discuss-phase` follow-up if genuinely contentious, otherwise let the planner decide (it's within "Claude's Discretion" scope per CONTEXT.md's "endpoint URL naming and whether the queue and form share one endpoint" framing, which implicitly covers this kind of endpoint-behavior detail).

2. **Exact suggestion count cap and debounce delay**
   - What we know: CONTEXT.md gives approximate defaults (~2 min chars, ~300ms delay, ~8 suggestion cap) explicitly as "Claude's Discretion."
   - What's unclear: Nothing blocking — these are tuning constants with no correctness implications either way.
   - Recommendation: Use the stated approximate defaults (2 chars, 300ms, 8 results) unless UAT surfaces a reason to change them.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `django-htmx` | Live-search widget (`request.htmx`, `hx-*` attrs already loaded client-side) | ✓ | 1.23.2 | — |
| `django.core.cache` (FileBasedCache backend) | Pool cache (existing) + new throttle counter | ✓ | Django 5.2.15 built-in | — |
| MPC Obscodes API (network) | `build_site_candidates()`'s 24h-cached bulk fetch (unchanged this phase) | ✓ (network-dependent) | — | Existing graceful local-only-pool fallback on outage (`campaign_utils.py`), unchanged |
| `difflib` | Fuzzy-match fallback | ✓ | Python 3.10+ stdlib | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None — all dependencies are already installed and
working in this environment; the only "network" dependency (MPC API) already has an
established graceful-degradation fallback from Phase 21, unchanged by this phase.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (unittest-based), run via `./manage.py test` |
| Config file | none — Django's built-in test runner, configured by `DJANGO_SETTINGS_MODULE=src.fomo.settings` (set in `manage.py`) |
| Quick run command | `./manage.py test solsys_code.tests.test_campaign_approval` (extend this existing module, or add a new `test_campaign_site_search.py`) |
| Full suite command | `./manage.py test solsys_code` |

Per CLAUDE.md: this is the **Django app test suite**, distinct from `python -m pytest`
(`testpaths = ["tests", "src", "docs"]` does not collect `solsys_code/`). All new tests for
this phase are DB-dependent (`Observatory`/`CampaignRun` fixtures, cache-backed throttle) and
belong under `solsys_code/tests/`.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D-01 | Anonymous GET to the live-search endpoint succeeds (no login redirect) | integration | `./manage.py test solsys_code.tests.test_campaign_site_search -v2` | ❌ Wave 0 |
| D-02 | Nth+1 request from the same IP within the window returns 429 | integration | same module | ❌ Wave 0 |
| D-03 | Response `Content-Type` is HTML (not `application/json`); body contains rendered `<li>`/suggestion markup | integration | same module | ❌ Wave 0 |
| D-04 | `substring_or_fuzzy_match_candidates('faulkes', pool)` returns both Faulkes sites; a query that finds no substring match falls back to `fuzzy_match_candidates()` | unit | `./manage.py test solsys_code.tests.test_campaign_utils` (or extend existing utils test module if one exists — verify at plan time) | ❌ Wave 0 (new test class in existing or new module) |
| D-05 | Suggestion display text is `"{name} ({obscode})"` | unit | same module | ❌ Wave 0 |
| D-06 | Picking a suggestion in a live browser test isn't automatable via Django `TestCase` (no JS execution) — verify the *rendered* `onclick`/`hx-*` markup is present and correctly escaped instead | unit (markup assertion) | `test_campaign_site_search` or `test_campaign_forms` | ❌ Wave 0 |
| D-07 | `ApprovalQueueView` context contains a third table for `approval_status=APPROVED, site_needs_review=True` rows | integration | extend `solsys_code.tests.test_campaign_approval` | ❌ Wave 0 (new test class) |
| D-08 | POST `action=resolve_site` on an approved+needs-review row resolves the site and creates the deferred `CalendarEvent`; a projection failure does not revert `approval_status` | integration | extend `solsys_code.tests.test_campaign_approval` (mirrors existing `TestCalendarProjection`/`TestApproval` patterns) | ❌ Wave 0 (new test class) |
| D-09 | Public submission form template contains no "Create new Observatory" link | unit (template/markup) | extend `solsys_code.tests.test_campaign_submission` | ❌ Wave 0 |
| D-10 | Approval-queue pending row keeps its "Create new Observatory" link alongside the new live-search widget | unit (template/markup) | extend `solsys_code.tests.test_campaign_approval` (`TestApprovalQueueSiteVisibility` or a new class) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_submission` (plus any new site-search-specific module)
- **Per wave merge:** `./manage.py test solsys_code`
- **Phase gate:** Full `./manage.py test solsys_code` green, plus `ruff check .` and `ruff format --check .` (per CLAUDE.md quality gates) before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_campaign_site_search.py` — new test module covering the
      live-search endpoint (anonymous access, throttle 429, fragment content) and the new
      `substring_or_fuzzy_match_candidates()` helper (D-01..D-06).
- [ ] Extend `solsys_code/tests/test_campaign_approval.py` with a `TestSitesNeedingReview`
      (or similar) class covering D-07/D-08: queryset membership, the `resolve_site` action's
      success path, its non-reverting failure path, and the D-06 never-re-resolve guard
      re-applied in the new branch.
- [ ] Extend `solsys_code/tests/test_campaign_submission.py` and/or
      `test_campaign_approval.py`'s `TestApprovalQueueSiteVisibility` with markup assertions
      for D-09/D-10 (presence/absence of the "Create new Observatory" link per surface).
- [ ] No new test framework/config needed — `./manage.py test` already covers everything;
      the existing `CampaignApprovalTestBase` fixture (`campaign`, `staff_user`,
      `non_staff_user`, `_make_pending_run()`) and `BULK_MPC_FIXTURE` in
      `test_campaign_approval.py` are directly reusable for the new test classes.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Live-search endpoint is deliberately unauthenticated (D-01); `resolve_site` action stays behind existing `StaffRequiredMixin` |
| V3 Session Management | No | No session-state changes introduced by this phase |
| V4 Access Control | Yes (for the resolve action + review table only) | `StaffRequiredMixin` (existing, unchanged) gates `ApprovalQueueView` and `CampaignRunDecisionView`; the live-search endpoint is intentionally public, matching `build_site_candidates()`'s already-public data (MPC bulk obscode list) |
| V5 Input Validation | Yes | `_MAX_OBSCODE_LEN` guard (existing, unchanged) still applies inside `resolve_site()`; the live-search query string itself needs no server-side length cap beyond what the throttle/substring-scan naturally bounds (a very long query simply matches nothing) |
| V6 Cryptography | No | Not applicable |
| V11 Business Logic / Anti-automation | Yes | D-02's per-IP throttle is the anti-automation control for the new anonymous endpoint |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Reflected/stored XSS via untrusted MPC `name_utf8`/`short_name`/`old_names` or submitter free text rendered into the suggestion fragment | Tampering | `format_html`/`format_html_join` or Django template auto-escaping for the visible text node; `|escapejs` for any value embedded inside an inline `onclick=` JS-string context (distinct escaping context — see Code Examples). Mirrors the closed `T-21-01` mitigation; do not regress it. |
| Denial of Service via unthrottled anonymous scan of the ~5,000+-string candidate pool per keystroke | Denial of Service | D-02's per-IP throttle (`cache.add`/`cache.incr`, ~10 lines) plus the existing `build_site_candidates()` 24h cache (no MPC re-fetch per request) plus the 2-character minimum-length gate (fewer, cheaper requests than character-1 firing). This is the direct anonymous-facing analogue of Phase 21's `T-21-04` (mitigated there only for staff-gated page loads). |
| Business-logic bypass: a staff session POSTs `action=resolve_site` for a `pk` that is not actually `APPROVED`+`site_needs_review=True` | Tampering / Elevation of Privilege (low severity — already requires staff auth) | The new branch must validate the row's actual state (`approval_status == APPROVED and site_needs_review is True`) before acting, not just trust that the UI only offers the button on eligible rows — mirrors the existing `updated_count == 1` pattern already used to detect a stale/already-decided approve/reject POST. |
| CSRF on the new endpoint | Tampering | Not applicable — the live-search endpoint is `GET`-only (read-only, idempotent); Django's CSRF protection only applies to unsafe methods. The `resolve_site` POST reuses the existing per-row CSRF token minting already in `ApprovalQueueTable.render_actions()` — no new CSRF surface. |

## Sources

### Primary (HIGH confidence)
- Direct repository inspection: `solsys_code/campaign_utils.py`, `campaign_views.py`,
  `campaign_tables.py`, `campaign_forms.py`, `campaign_urls.py`, `models.py`,
  `solsys_code_observatory/models.py`/`urls.py`, `mixins.py`,
  `solsys_code/tests/test_campaign_approval.py` — read in full at research time.
- Direct installed-environment inspection: `pip show django-htmx` (1.23.2), `pip show django`
  (5.2.15), `python3 -c "import inspect; ... BaseCache.incr/add"`,
  `python3 -c "import django.http as h; print(dir(h))"` (confirms no
  `HttpResponseTooManyRequests`), `tom_common/templates/tom_common/base.html` (`htmx.min.js`
  already loaded site-wide), `tom_common/templates/tom_common/bootstrap_htmx.html` (existing
  in-repo HTMX pattern precedent).
- `.planning/phases/21-site-disambiguation-submitter-contact-opt-in/21-CONTEXT.md`,
  `21-SECURITY.md` — prior-phase locked decisions and closed threat register this phase
  builds on/supersedes.
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` — locks
  `difflib` over `rapidfuzz`.

### Secondary (MEDIUM confidence)
- `https://htmx.org/attributes/hx-trigger/` (fetched directly, WebFetch) — confirms the
  `delay:` and event-filter-bracket syntax, and confirms `hx-min-length` is **not** a real
  htmx attribute (correcting a general web-search summary that implied otherwise).

### Tertiary (LOW confidence)
- General web search on "Django per-IP rate limiting using cache framework" — informed the
  general shape of the `cache.add()`/`cache.incr()` throttle pattern, but the specific
  behavior claims (`incr()` is get-then-set, not atomic on `FileBasedCache`) were verified
  directly against the installed Django source, not taken from search results alone.
- General web search on htmx debounce patterns — the `delay:` modifier and `changed`
  qualifier were corroborated by the direct htmx.org fetch above; treat the search-summary's
  mention of `hx-min-length` as incorrect (see Pitfall 1).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; every library/version claim verified directly
  against the installed environment.
- Architecture: HIGH — every pattern is either a direct extraction of existing, already-read
  code (`_project_calendar_event`) or a small addition following an established in-repo
  convention (`format_html`, `reverse_lazy`, `StaffRequiredMixin`, `request.htmx`).
- Pitfalls: MEDIUM-HIGH — the htmx/`HttpResponseTooManyRequests` pitfalls are directly
  verified; the throttle-exemption-for-staff pitfall (A3/Pitfall 5) is a design judgment call,
  not a verified fact, and is flagged as such in Open Questions.

**Research date:** 2026-07-14
**Valid until:** 2026-08-13 (30 days — stable stack, no fast-moving external dependencies;
re-verify `django-htmx` version if the milestone extends past that window)
