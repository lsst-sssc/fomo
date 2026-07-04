# Phase 17: Coverage-Gap Analysis (Deferrable to v2.1) - Research

**Researched:** 2026-07-04
**Domain:** Django low-level caching, per-campaign gap-computation view design, `telescope_runs.py` performance characteristics
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**GAP-01: research-spike decision (dark-window-only vs. target-altitude)**
- **D-01:** Locked now: dark-window-only, reusing `telescope_runs.py`'s `sun_event()`/
  `get_site()` — not true target-altitude/airmass filtering via `ephem_utils`. Pre-milestone
  research (`.planning/research/ARCHITECTURE.md`'s explicit "Decision", `PITFALLS.md`,
  `SUMMARY.md`, `STACK.md`) already unanimously recommends this; re-running the same research
  question at plan time would not surface new information. `gsd-phase-researcher` should treat
  this as settled and focus research effort on implementation details (view/cache design), not
  re-litigating the SPICE-cost tradeoff.
- **D-02:** GAP-01's success criterion ("a phase-time research spike produces an explicit
  decision") is satisfied with a short written decision doc during execution
  (`17-GAP-01-DECISION.md` or similar), mirroring Phase 13's `13-DECISION.md` precedent — even
  though the decision itself was reached quickly via this discussion rather than a multi-day
  spike.
- **D-03:** A per-date `sun_event(kind='dark')` `ValueError` skips that one date as "unknown",
  does not abort the whole gap request. Matches this codebase's existing per-line/per-record
  "log+skip, never abort" convention.
- **D-04:** Any non-zero dark window counts as "observable" — no minimum-duration threshold.

**"Claimed" definition**
- **D-05:** A date is "claimed" (excluded from the gap list) when a `CampaignRun` has
  `approval_status='approved'` and `run_status` is not in
  `{cancelled, not_awarded, weather_tech_failure}`.
- **D-06:** The claimed date is `obs_date` if set, else derived from `ut_start`.
- **D-07:** When deriving from `ut_start`, convert to the site's local calendar date, using the
  same local-noon-anchored "observing night" convention `sun_event()` already uses.
- **D-08:** A `CampaignRun` with neither `obs_date` nor `ut_start` set cannot be attributed to
  any date. It is flagged separately as "undated, needs review" alongside the gap-analysis
  result.

**Trigger, caching & date range**
- **D-09:** Gap computation is triggered by a button on the per-campaign table page that loads
  a separate gap-analysis section/page via a normal (non-htmx) request.
- **D-10:** Results are cached via Django's low-level cache framework (`cache.set`/`cache.get`),
  keyed by `(campaign, target, site, date range)`, with a 1-hour TTL. Display a "last computed
  at" timestamp alongside the result.
- **D-11:** Default date-range window is the next 90 days from today; max allowed span is
  180 days, enforced server-side regardless of any client-supplied range.

**Target + site selection**
- **D-12:** Target selection: auto-use the sole `Target` when `campaign.targets.count() == 1`;
  otherwise show a dropdown to pick one.
- **D-13:** Site selection: a dropdown of `Observatory` records already used by this campaign's
  `CampaignRun`s (i.e. distinct non-null `.site` values among the campaign's runs) — not every
  `Observatory` in the DB, and not restricted to `telescope_runs.py`'s 4-entry `SITES` dict.
- **D-14:** If a campaign has zero `Target`s, or its `CampaignRun`s have no resolved site at all
  (all `site=None`/`site_needs_review=True`), the gap-analysis button is hidden/disabled with an
  explanatory message rather than shown and failing on click.

### Claude's Discretion
- Exact URL names/paths for the gap-analysis view/section.
- Exact wording of the "last computed at" display and the "undated, needs review" flag (D-08).
- Whether the gap-analysis result lives on its own page or as a section appended to the existing
  per-campaign table page.
- Internal structure/naming of the `17-GAP-01-DECISION.md` artifact from D-02.
- Exact cache key format for D-10 — any format that's stable and collision-free is fine.

### Deferred Ideas (OUT OF SCOPE)
- Auto-calculate/suggest `ut_start`/`ut_end` on the Phase 16 public submission form via JS,
  based on the entered site (MPC code) + `obs_date`, likely reusing `telescope_runs.sun_event()`.
  Not this phase's scope — candidate for a future phase or quick task.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GAP-01 | Phase-time research spike decides dark-window-only vs. target-altitude filtering (the `ephem_utils`/SPICE cost decision) before implementation | Already locked by CONTEXT.md D-01/D-02 (pre-milestone research unanimous). This research file's contribution is documenting the concrete performance cost of the chosen approach (see "Critical Finding: `sun_event()` Per-Call Cost" below) and the `17-GAP-01-DECISION.md` artifact's expected shape (see Architecture Patterns). |
| GAP-02 | User can view observable-but-unclaimed dates for a campaign target + site; computed on explicit request or cached, never inline in the table view, never importing `ephem_utils` at module scope | Covered by Standard Stack (Django low-level cache), Architecture Patterns (view/query design), Code Examples (cache key builder, claimed-date query, gap-set computation), and Common Pitfalls (cache test-isolation, per-call latency, IDOR on target/site params). |
</phase_requirements>

## Summary

This phase's open technical questions are narrow: everything about *whether* to build
dark-window-only gap analysis is already locked by CONTEXT.md D-01–D-14. What remains is HOW —
concrete Django cache usage, view/query structure, and integration into the existing
`campaign_views.py`/`campaign_tables.py`/`campaign_urls.py` trio from Phases 15/16.

The most consequential finding from this research session is **not** a stack or pattern
question — it's a directly-measured performance number. `telescope_runs.sun_event()` (the only
ephemeris call this phase is allowed to make) costs **~520ms per call** on this machine, measured
via `cProfile` against real seeded `Observatory` records. A D-11 90-day default window therefore
costs **~47 seconds** of synchronous computation on a cache miss; a 180-day max-span request
costs **~94 seconds**. This is an order of magnitude more expensive than the discussion's framing
("a few hundred `sun_event()` calls per request... computationally reasonable") assumed, and it
must inform how the planner scopes the view (see "Critical Finding" below) — the request/response
cycle, cache strategy, and UX copy all need to account for a worst-case near-2-minute synchronous
wait, not a sub-second one.

Everything else is straightforward: `CACHES` is already configured in `src/fomo/settings.py`
(a `FileBasedCache` pointed at the system temp dir) — no settings change needed to use
`django.core.cache.cache`. Django's test runner does **not** auto-isolate the cache between test
cases the way it does for email (confirmed by reading `django/test/signals.py` directly — there
**is** a `setting_changed` receiver that correctly resets `django.core.cache.cache` when
`override_settings(CACHES=...)` is used, so the standard override-to-locmem-plus-clear pattern
works cleanly). No new third-party package is needed for this phase; `stdlib datetime` plus
Django's ORM plus the existing cache framework are sufficient, matching pre-milestone
`STACK.md`'s conclusion.

**Primary recommendation:** Ship a synchronous `TemplateView`/`FormView`-style gap-analysis view
(GET-triggered per D-09), a small new `campaign_gap.py` module (mirroring `campaign_utils.py`'s
role) holding the pure query/computation logic, cache the result dict via
`django.core.cache.cache` keyed on a stable delimited string, and communicate the (accurately
measured) potential ~1-minute wait to the user explicitly rather than assuming the computation is
fast.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Compute per-date dark-window observability | Backend/API | — | Pure function call into `telescope_runs.sun_event()`; no DB write, no template concern |
| Query "claimed" dates from `CampaignRun` | Backend/API | Database/Storage | Django ORM query against existing model; the DB does the filtering, the view shapes the result |
| Cache the combined gap result | Database/Storage | Backend/API | Django's low-level cache framework (`FileBasedCache`, already configured) is a storage-tier concern accessed from the view layer |
| Target/site selection dropdowns | Frontend Server (SSR) | Backend/API | Rendered server-side (Django template + form), but the *choices offered* must be validated server-side too (see Security Domain — IDOR) |
| "Last computed at" / "undated, needs review" display | Frontend Server (SSR) | — | Pure template/context concern once the view has assembled the result dict |
| Gap-analysis trigger button | Frontend Server (SSR) | — | A link/button in the existing `campaignrun_table.html` template (D-09), no JS |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django cache framework (`django.core.cache.cache`) | Bundled with Django 5.2.15 (already installed, confirmed via `pip show` in prior milestone research) | Low-level `cache.get`/`cache.set` for D-10's 1-hour TTL result cache | Already configured in `src/fomo/settings.py` (`CACHES['default']` = `FileBasedCache`, `LOCATION=tempfile.gettempdir()`) — zero new dependency, zero new settings change [VERIFIED: direct read of `src/fomo/settings.py:190-198`] |
| `solsys_code.telescope_runs.sun_event()` / `get_site()` | Existing module, no version to track (this repo's own code) | Per-site dark-window UTC crossing times — the only ephemeris dependency this phase introduces (D-01) | Already tested to ≤2 min vs. LCO skycalc reference (Stage 1's own success criterion); explicitly does not import `ephem_utils` [VERIFIED: direct read of `solsys_code/telescope_runs.py`] |
| stdlib `datetime` | 3.10+ (project baseline) | Date-range iteration, set-difference between the full observable-night list and the claimed-date set | Pre-milestone `STACK.md` already confirmed no interval-arithmetic package is needed at this scale — one campaign, one date range [CITED: `.planning/research/STACK.md`] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `django-tables2` 3.0.0 | Already installed (Phase 15/16 dependency) | Optional: render the "undated, needs review" list or the observable-dates list as a small table for visual consistency with the rest of the campaign pages | Only if the planner wants tabular rendering; a plain unordered list of dates is equally valid and simpler for this feature's small row counts (at most 180 dates) |
| `django-crispy-forms` 2.4 + `crispy-bootstrap4` | Already installed | Rendering the target/site-selection dropdown form (D-12/D-13), matching `CampaignRunSubmissionForm`'s existing styling | Use if the selection UI is a real Django form (recommended — see Architecture Patterns) rather than raw `<select>` HTML |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Django's low-level `cache` API (D-10, locked) | A dedicated `GapAnalysisResult` model row with a `computed_at` timestamp | Rejected by D-10 itself — "not a dedicated persistent model." A cache-backed approach needs no migration and the TTL handles staleness without invalidation signals. |
| `FileBasedCache` (already configured, reused as-is) | Switching to `LocMemCache` or `django-redis` for production | Out of scope for this phase — D-10 doesn't ask for a cache-backend change, and `FileBasedCache` already works correctly for a single-process dev deployment. Only relevant if FOMO ever moves to multi-worker production deployment, where `FileBasedCache`'s per-file-on-disk model still technically works (shared filesystem) but a proper networked cache (Redis/Memcached) becomes preferable — not a decision this phase needs to make. |
| A synchronous view with the measured ~47-94s worst-case latency (recommended, see Critical Finding) | A Celery/RQ background task + polling/websocket UI | No task queue is installed anywhere in this codebase (verified: no `celery`/`django_rq`/`channels` in `pyproject.toml` or `INSTALLED_APPS`). Introducing one is a large, out-of-scope infrastructure addition for a "deferrable to v2.1" phase. Recommend synchronous with honest UX copy instead (see Common Pitfalls). |

**Installation:**
```bash
# Nothing to install. django.core.cache is bundled with Django; CACHES is already
# configured in src/fomo/settings.py. Confirm with:
python3 -c "from django.core.cache import cache; print(cache)"
```

**Version verification:** No new packages recommended for this phase; Django 5.2.15 is already
the pinned/installed version (confirmed by prior-milestone `STACK.md`'s `pip show` check,
2026-07-02). No re-verification needed since nothing new is being added.

## Package Legitimacy Audit

**No external packages are introduced by this phase.** Everything needed (Django's bundled cache
framework, `stdlib datetime`, this repo's own `telescope_runs.py`, and already-installed
`django-tables2`/`django-crispy-forms` for optional UI polish) is either stdlib, Django-bundled,
or already vetted and in use by Phases 14-16. The Package Legitimacy Gate protocol is not
applicable here — skip the check-and-verdict table.

**Packages removed due to [SLOP] verdict:** none (no new packages proposed)
**Packages flagged as suspicious [SUS]:** none (no new packages proposed)

## Architecture Patterns

### System Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Per-campaign table page (existing, Phase 15/16)                          │
│ campaignrun_table.html                                                    │
│  [Show coverage gaps ▸] button (D-09, NEW) -- hidden/disabled if D-14     │
│  applies (no targets, or no CampaignRun has a resolved site)              │
└───────────────────────────┬────────────────────────────────────────────┘
                            │ GET (non-htmx, D-09)
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ CampaignGapAnalysisView (NEW, campaign_views.py)                          │
│  1. Resolve target (D-12: auto if campaign.targets.count()==1, else       │
│     validate submitted target_pk belongs to this campaign)                │
│  2. Resolve site (D-13: validate submitted site_pk is among this          │
│     campaign's distinct CampaignRun.site values)                          │
│  3. Resolve + clamp date range (D-11: default next 90d, hard cap 180d)    │
│  4. Build cache key from (campaign, target, site, start, end) -- D-10     │
│  5. cache.get(key) -- HIT: use cached dict, show "last computed at"       │
│     MISS: compute (step 6-8), cache.set(key, result, timeout=3600)        │
└───────────────────────────┬────────────────────────────────────────────┘
                            │ (cache miss path only)
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ campaign_gap.py (NEW module, mirrors campaign_utils.py's role)            │
│  6. claimed_dates(campaign, target, site) -> set[date]                    │
│     Query CampaignRun.objects.filter(campaign=, site=)                    │
│     [+ target= if campaign has >1 target, per D-12's disambiguation       │
│     need -- see Open Questions], approval_status=APPROVED,                │
│     run_status not in {CANCELLED, NOT_AWARDED, WEATHER_TECH_FAILURE}      │
│     (D-05). Derive date via D-06/D-07 (obs_date, else ut_start's site-    │
│     local calendar date). Separately collect undated rows (D-08).         │
│                                                                            │
│  7. observable_dates(site, start, end) -> set[date]                       │
│     For each date in the range: telescope_runs.sun_event(site, date,      │
│     kind='dark'); ValueError -> skip as unknown (D-03); any non-zero      │
│     dark window counts (D-04). This is the ~520ms/call cost (see          │
│     Critical Finding) -- the loop's total cost is O(days in range).       │
│                                                                            │
│  8. gap_dates = observable_dates - claimed_dates (plain set difference,   │
│     stdlib only, per STACK.md)                                            │
└───────────────────────────┬────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ campaigns/campaignrun_gap_analysis.html (NEW template, or a section of    │
│ the existing table template -- Claude's Discretion)                      │
│  Renders: gap_dates list, "last computed at" timestamp, undated-runs      │
│  flag (D-08), target/site selection form for re-running with different   │
│  parameters                                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
solsys_code/
├── campaign_gap.py          # NEW: claimed_dates(), observable_dates(), gap
│                             #      cache-key builder -- pure logic, no view
│                             #      concerns, mirrors campaign_utils.py's role
├── campaign_views.py         # MODIFIED: + CampaignGapAnalysisView
├── campaign_urls.py          # MODIFIED: + one new path
├── campaign_forms.py         # MODIFIED: + a small target/site/date-range
│                             #      selection form (D-12/D-13/D-11), OR
│                             #      Claude's Discretion: plain GET params
│                             #      validated in the view if a full form is
│                             #      judged unnecessary for this small UI
└── tests/
    └── test_campaign_gap.py  # NEW: DB-dependent tests (Observatory/
                              #      CampaignRun fixtures) -- runs via
                              #      ./manage.py test solsys_code
```

### Pattern 1: Cache-or-compute with an explicit "last computed at" timestamp

**What:** `cache.get(key)` returning `None` on miss; on hit, the cached value must itself carry
a `computed_at` timestamp (not just the gap-date list) so the template can show D-10's "last
computed at" UI per `PITFALLS.md`'s explicit recommendation.

**When:** Every request to the gap-analysis view.

**Example:**
```python
# Source: this repo's own solsys_code/campaign_gap.py (recommended new module),
# following the cache-key pattern confirmed against Django's own cache docs
# (docs.djangoproject.com/en/5.2/topics/cache/#the-low-level-cache-api)
from datetime import date, timedelta

from django.core.cache import cache
from django.utils import timezone

GAP_CACHE_TTL_SECONDS = 3600  # D-10


def build_gap_cache_key(campaign_pk: int, target_pk: int | None, site_pk: int, start: date, end: date) -> str:
    """Stable, collision-free cache key for a gap-analysis request (D-10).

    target_pk may be None for a single-target campaign that has no CampaignRun.target
    disambiguation need (D-12) -- represented explicitly as the string 'none' rather than
    omitted, so a null-target request never collides with a differently-scoped request.
    """
    return f'campaign_gap:{campaign_pk}:{target_pk if target_pk is not None else "none"}:{site_pk}:{start.isoformat()}:{end.isoformat()}'


def get_or_compute_gap(campaign, target, site, start: date, end: date) -> dict:
    """Cache-or-compute wrapper. Returns a dict with gap_dates, undated_runs, computed_at."""
    key = build_gap_cache_key(campaign.pk, target.pk if target else None, site.pk, start, end)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _compute_gap(campaign, target, site, start, end)  # -> {'gap_dates': [...], 'undated_runs': [...]}
    result['computed_at'] = timezone.now()
    cache.set(key, result, timeout=GAP_CACHE_TTL_SECONDS)
    return result
```

### Pattern 2: Per-date dark-window loop with per-date skip-on-error (D-03)

**What:** Iterate the requested date range calling `telescope_runs.sun_event(site, d,
kind='dark')`, catching `ValueError` per date rather than letting one bad date abort the whole
computation — matches this codebase's established per-record log-and-skip convention
(`load_telescope_runs`, `import_campaign_csv`).

**When:** Building the observable-nights side of the gap computation.

**Example:**
```python
# Source: this repo's own solsys_code/telescope_runs.py (existing sun_event contract)
# and the log+skip convention already established in load_telescope_runs.py/campaign_utils.py
import logging
from datetime import date, timedelta

from solsys_code.telescope_runs import sun_event

logger = logging.getLogger(__name__)


def observable_dates(site, start: date, end: date) -> set[date]:
    """D-04: any non-zero dark window counts as observable. D-03: skip unknown dates."""
    observable = set()
    n_days = (end - start).days + 1
    for i in range(n_days):
        d = start + timedelta(days=i)
        try:
            sun_event(site, d, kind='dark')  # non-zero dark window exists if this doesn't raise
            observable.add(d)
        except ValueError:
            logger.debug('sun_event(dark) raised for site=%s date=%s; skipping as unknown (D-03).', site, d)
    return observable
```

### Pattern 3: Server-side clamping of a client-supplied date range (D-11)

**What:** The 90-day default / 180-day max span must be enforced server-side regardless of any
GET parameter the client sends — never trust a client-supplied `end` date directly.

**Example:**
```python
from datetime import date, timedelta

DEFAULT_WINDOW_DAYS = 90  # D-11
MAX_WINDOW_DAYS = 180  # D-11


def clamp_date_range(today: date, requested_end: date | None) -> tuple[date, date]:
    """Enforce D-11's 90-day default / 180-day max span, independent of client input."""
    start = today
    default_end = start + timedelta(days=DEFAULT_WINDOW_DAYS)
    max_end = start + timedelta(days=MAX_WINDOW_DAYS)
    if requested_end is None:
        return start, default_end
    return start, min(requested_end, max_end)
```

### Anti-Patterns to Avoid

- **Recomputing on every table-view page load:** Explicitly forbidden by the phase's own third
  success criterion and by pre-milestone `PITFALLS.md` Pitfall 5. The gap view must be a
  separate, explicitly-triggered endpoint (D-09), never called from
  `CampaignRunTableView.get_context_data()`.
- **Trusting dropdown-restricted `target_pk`/`site_pk` GET params as pre-validated:** D-12/D-13
  scope the *dropdown's offered choices*, not what a request can actually submit. A user (or a
  crafted request) could submit any `target_pk`/`site_pk` regardless of what the rendered
  `<select>` contained — the view must re-validate server-side that the submitted target belongs
  to this campaign and the submitted site is among this campaign's used sites, exactly like
  D-14's underlying "never fabricate/guess" discipline. See Security Domain.
- **Importing `solsys_code.views` or `solsys_code.ephem_utils`, even transitively:** Already
  covered by D-01 and the phase's own third success criterion; verify with
  `grep -rn "import.*ephem_utils\|from solsys_code.views import" solsys_code/campaign_gap.py
  solsys_code/tests/test_campaign_gap.py` returning zero hits before considering the phase done.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Result caching with TTL | A custom in-memory dict + manual expiry timestamps | Django's `django.core.cache.cache` (`cache.get`/`cache.set(timeout=3600)`) | Already configured (`FileBasedCache`), zero new code for expiry logic, and it's the mechanism D-10 explicitly names |
| Per-date dark-window computation | A new lightweight sun-altitude formula | `telescope_runs.sun_event(site, date, kind='dark')` | Already tested to ≤2 min accuracy vs. LCO skycalc; D-01 explicitly locks this reuse |
| Date-range gap-set computation | A dependency like `python-intervals` | stdlib `set` difference (`observable_dates - claimed_dates`) | Pre-milestone `STACK.md` already confirmed this is sufficient at this scale — one target, one site, at most 180 dates |
| Access control for the (non-staff-gated) gap view | A new permission class or `django-guardian` object permission | Straightforward server-side validation that submitted `target_pk`/`site_pk` belong to the requested campaign (plain ORM `.filter()` checks) | This phase's gap view is public/read-only like Phase 15's table view, not staff-only — the concern is parameter tampering (IDOR), not authorization tiers, and doesn't need a permissions framework |

**Key insight:** Every "don't hand-roll" item in this phase already has an existing, in-repo,
already-vetted answer (Django's cache framework, `telescope_runs.py`, stdlib `set`). The only
genuinely new code this phase writes is the *composition* of those three things into one view —
which is exactly the "genuinely novel synthesis" `FEATURES.md` already flagged pre-milestone.

## Common Pitfalls

### Pitfall 1: `sun_event()`'s real per-call cost is ~520ms, not "a few hundred cheap calls" (CRITICAL — new finding this session)

**What goes wrong:** CONTEXT.md's own framing of D-11 describes the 90/180-day window as
bounding "the worst-case number of `sun_event()` calls per request" as if the per-call cost were
negligible. Measured directly in this repo via `cProfile` against real seeded `Observatory`
records (`obscode='268'`, Magellan-Clay), a single `sun_event(site, date, kind='dark')` call
costs **~520ms** (60-call average: 523.9ms/call; profiled single-call breakdown: 21 internal
`_solar_altitude()` evaluations — 1 vectorized coarse scan + 2×10 bisection-refinement
single-point calls — each paying astropy/erfa's `transform_to`/`epv00` fixed per-call overhead,
independent of the coarse-scan's vectorized point count).

**Why it happens:** `_find_crossing()`'s bisection refinement (`solsys_code/telescope_runs.py:163-204`)
evaluates `_solar_altitude()` once per bisection iteration (10 iterations × 2 crossings = 20
calls) as single-point (non-vectorized) `Time` objects, and each such call re-pays astropy's full
coordinate-transform pipeline overhead (dominated by `erfa.core.epv00`, confirmed in the
`cProfile` trace) rather than amortizing it across a batch.

**Concrete cost projection (measured, this machine):**
- D-11 default window (90 days): **~47 seconds** synchronous, cache-miss cost
- D-11 max window (180 days): **~94 seconds** synchronous, cache-miss cost

**How to avoid:** This is not a phase-blocking defect — it's a design input the planner must
account for explicitly, not silently assume away:
1. **Recommended minimum:** Ship the view synchronously as planned (D-09's non-htmx button,
   D-10's 1-hour cache), but make the near-1-minute worst-case latency an explicit, tested UX
   decision — e.g. copy reading "Computing observable nights may take up to a minute for a full
   180-day window; results are cached for an hour afterward" rather than a bare spinner with no
   explanation. This matches this codebase's own existing precedent (`MakeEphemerisView` is
   already a synchronous, occasionally-slow view with no async infrastructure) and D-09's framing
   of this as a rare, deliberate action.
2. **Optional (if time allows, not required for phase completion):** A batched variant of the
   coarse-scan-plus-bisection loop that vectorizes across the *entire requested date range* in a
   handful of large `Time` arrays rather than one `sun_event()` call per date, cutting the
   ~21-calls-per-date cost down dramatically. This would mean adding a new function to
   `telescope_runs.py` (or a wrapper in `campaign_gap.py`) rather than a strict reuse of
   `sun_event()` as-is — a larger scope change than CONTEXT.md's discussion anticipated. Flag this
   as a "should, not must" to the planner; do not silently attempt it without calling it out as
   new scope beyond D-01's "reuse sun_event()/get_site()" framing.
3. **Do not** reach for a task queue (Celery/RQ) to solve this — none is installed anywhere in
   this codebase (verified: no `celery`/`django_rq`/`channels` in `pyproject.toml`), and adding
   one is out of scope for a phase explicitly framed as deferrable/lightweight.

**Warning signs:** A plan that estimates gap-view response time in "a couple seconds" without
having measured it; a UAT/manual test that only exercises a small 5-10 day range (which would
hide this cost — 5 days × ~520ms ≈ 2.6s, well within "feels fine" territory, while the D-11
default 90-day case is 18x slower).

### Pitfall 2: Django's low-level cache is NOT auto-isolated between test cases

**What goes wrong:** Unlike `EMAIL_BACKEND` (which Django's test runner transparently replaces
with a locmem backend regardless of settings — confirmed by this codebase's existing
`test_campaign_submission.py` using `django.core.mail.outbox` with zero `override_settings`
calls), the cache framework has no equivalent automatic test isolation. A value written via
`cache.set()` in one test remains visible to a later test unless explicitly cleared — this
repo's `CACHES['default']` is `FileBasedCache` pointed at `tempfile.gettempdir()`, a location
shared across every test run (and the dev server) on the machine.

**Why it happens:** Django deliberately does not treat the cache like the database (which is
wrapped in a per-test transaction and rolled back automatically) — confirmed via Django's own
ticket tracker (tickets #11505, #17995, #20075: cache-flush-between-tests has been an open,
unresolved feature request for over a decade) [CITED: code.djangoproject.com, verified by
websearch, cross-checked against multiple independent sources].

**How to avoid:** Use `@override_settings(CACHES={'default': {'BACKEND':
'django.core.cache.backends.locmem.LocMemCache'}})` on the test class (or an individual test),
matching the exact pattern this codebase already uses for `@override_settings(FACILITIES=...)`
in `test_sync_gemini_observation_calendar.py`. This is fully sufficient on its own — **direct
read of `django/test/signals.py` in this repo's installed Django (5.2.15) confirms** a
`@receiver(setting_changed)` handler (`clear_cache_handlers`) specifically resets
`django.core.cache.caches`'s internal connection/settings state whenever the `CACHES` setting
changes, and `django.core.cache.cache` is itself a `ConnectionProxy` that looks up
`caches['default']` fresh on every access — so no manual "reset the global" workaround is needed
[VERIFIED: direct read of `django/test/signals.py:26-33` and `django/core/cache/__init__.py:56-58`
in this repo's active venv]. Additionally call `cache.clear()` in `setUp()`/`tearDown()` as a
belt-and-suspenders measure for tests that don't use `override_settings` (e.g. a test asserting
the *production* `FileBasedCache` config itself behaves correctly).

**Warning signs:** A gap-analysis test that passes in isolation but fails (or passes for the
wrong reason — stale cached data from an earlier test) when run as part of the full
`./manage.py test solsys_code` suite; a test asserting "last computed at" changed between two
calls that's actually reading a cache entry left over from a previous test's run.

### Pitfall 3: Trusting client-supplied `target_pk`/`site_pk` as already scoped to the campaign (IDOR)

**What goes wrong:** D-12/D-13 describe *dropdown* population rules ("auto-use the sole Target",
"a dropdown of Observatory records already used by this campaign's CampaignRuns") — these
constrain what's *offered* in the UI, not what a raw HTTP request can submit. A request that
submits a `target_pk` belonging to a different campaign's target, or a `site_pk` never actually
used by this campaign, must be rejected server-side, not merely "not offered" client-side.

**Why it happens:** It's easy to conflate "the dropdown only shows valid choices" with "therefore
the server only ever receives valid choices" — the latter doesn't follow, since any GET
parameter can be hand-crafted regardless of what a `<select>` rendered.

**How to avoid:** In the view, always re-derive the *allowed* target/site sets from the campaign
(same query D-12/D-13 use to populate the dropdown) and validate the submitted `pk` is a member
of that set before using it in the cache key or the `campaign_gap.py` query — return `400 Bad
Request` (mirroring `CampaignRunDecisionView`'s existing `HttpResponseBadRequest()` pattern for
an invalid `action` value) rather than silently falling back to a default or, worse, running the
query with an out-of-scope target/site.

**Warning signs:** A view that does `Target.objects.get(pk=request.GET['target_pk'])` with no
`campaign.targets.filter(pk=...)` membership check first; a `site_pk` used to build the cache
key and run the query without confirming it appears in `campaign.campaign_runs.values('site')`.

### Pitfall 4: Ambiguous `CampaignRun.target` attribution for multi-target campaigns

**What goes wrong:** `CampaignRun.target` is nullable (`null=True, blank=True`) and, per D-12's
own precedent (`import_campaign_csv.py`'s `campaign.targets.first() if
campaign.targets.count() == 1 else None`), is commonly left unset even for real imported data.
For a single-target campaign this is harmless (every run implicitly "belongs" to the one
target). For a multi-target campaign, though, a `CampaignRun` with `target=None` is genuinely
ambiguous — the "claimed" query for a *specific* selected target has no principled way to decide
whether an unattributed run should count as claiming that target's date or not.

**Why it happens:** This exact question is not addressed by any of CONTEXT.md's locked
decisions (D-05 through D-08 define claimed-vs-unclaimed logic assuming an unambiguous
target; D-12 only addresses *which target to select for the gap view*, not how to filter
`CampaignRun.target=None` rows once a specific target is selected).

**How to avoid (recommendation, not a locked decision — planner's call):** For a single-target
campaign (the common case, `campaign.targets.count() == 1`), don't filter the claimed-date query
by `target` at all — filter by `campaign` + `site` only, since essentially every real run will
have `target=None` in this case (per D-12's own precedent) and the single target is implied. For
a multi-target campaign, filter `target=<selected target>` strictly and treat `target=None` rows
as *not* claiming any specific target's dates — consistent with this codebase's established
"never guess/fabricate" discipline (D-08's own "undated, needs review" flag is the model to
follow: an unattributed row is a data-quality signal, not something to silently resolve either
way). Consider surfacing target-unattributed multi-target-campaign runs alongside D-08's
"undated, needs review" flag rather than silently dropping them from consideration.

**Warning signs:** A gap-analysis result for a multi-target campaign that silently either
over-counts (treats every `target=None` run as claiming every target's dates) or under-counts
(ignores `target=None` rows entirely with no visible flag) with no way for a user to notice the
discrepancy.

## Code Examples

Verified patterns from this repo's own source (no external library APIs beyond Django's
documented cache API):

### Django's low-level cache API contract (confirmed against this repo's installed Django 5.2.15)
```python
# Source: django/core/cache/__init__.py (installed package, read directly)
from django.core.cache import cache

cache.set('some_key', {'gap_dates': [...]}, timeout=3600)  # seconds
value = cache.get('some_key')  # None if missing or expired
```

### Test isolation pattern to copy (mirrors this repo's own FACILITIES override precedent)
```python
# Source: solsys_code/tests/test_sync_gemini_observation_calendar.py (this repo, existing
# @override_settings(FACILITIES=GEM_SETTINGS) precedent), adapted for CACHES
from django.core.cache import cache
from django.test import TestCase, override_settings

TEST_CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}


@override_settings(CACHES=TEST_CACHES)
class TestCampaignGapAnalysis(TestCase):
    def setUp(self):
        cache.clear()  # belt-and-suspenders; override_settings already resets the backend
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| N/A — this is new functionality, not a replacement of an existing gap-analysis feature | N/A | N/A | N/A |

**Deprecated/outdated:** Nothing applicable — no prior version of this feature exists in FOMO.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Multi-target-campaign `CampaignRun.target=None` handling recommendation (exclude from a specific target's claimed-date set, flag separately) | Common Pitfalls, Pitfall 4 | Low-medium — this is presented as a recommendation, not a locked decision; if the planner/user decides differently (e.g. treat `target=None` as claiming for every target in a multi-target campaign), the query in `campaign_gap.py` simply needs a different filter, no architectural rework |
| A2 | The ~520ms/call `sun_event()` cost is representative of the deployment environment, not just this dev machine | Common Pitfalls, Pitfall 1 | Medium — if production hardware is meaningfully faster/slower, the 47s/94s projections shift proportionally, but the *order of magnitude* (tens of seconds, not sub-second) is very unlikely to change since the cost is dominated by fixed-overhead astropy/erfa transform calls, not I/O or DB access |
| A3 | Optional batched/vectorized `sun_event()` optimization (Pitfall 1, item 2) is out of this phase's minimum scope | Common Pitfalls, Pitfall 1; Standard Stack alternatives | Low — explicitly framed as "should, not must"; if the planner includes it anyway, no other research finding in this document depends on it being absent |

**If this table is empty:** N/A — see entries above; all other claims in this research were
either verified via direct source read/measurement in this repo, or cited against Django's own
source/documentation/ticket tracker.

## Open Questions

1. **Should the target/site/date-range selection be a real Django `Form` or plain validated GET params?**
   - What we know: `CampaignRunSubmissionForm` (Phase 16) is a plain `forms.Form` rendered via
     crispy-forms; the gap-analysis selection UI is much smaller (2-3 fields: target, site,
     optionally a custom end date).
   - What's unclear: Whether the extra structure of a `forms.Form` (validation, crispy rendering)
     is worth it for a 2-3-field GET-request selector, versus reading/validating `request.GET`
     directly in the view (as `CampaignRunDecisionView` does for its `action` POST param).
   - Recommendation: Use a small `forms.Form` for consistency with this codebase's established
     pattern (every other user input in `campaign_*` is form-mediated) and because crispy-forms
     is already a project dependency — but this is genuinely Claude's Discretion per CONTEXT.md,
     not a blocking question.

2. **Multi-target `CampaignRun.target=None` attribution (see Pitfall 4)**
   - What we know: CONTEXT.md's locked decisions don't address this case.
   - What's unclear: Whether to exclude, include, or flag `target=None` rows when a specific
     target is selected for a multi-target campaign.
   - Recommendation: See Pitfall 4's recommendation (exclude + flag). Since real campaign data
     seen so far (3I/ATLAS) is a single-target campaign, this edge case may not be exercised by
     real data at all during this phase — worth a small dedicated unit test regardless, since the
     model structurally allows it.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Django cache framework (`django.core.cache`) | D-10's result caching | ✓ | Bundled with Django 5.2.15 (already installed) | — |
| `CACHES` setting configured | Cache framework to function at all | ✓ | `FileBasedCache`, `LOCATION=tempfile.gettempdir()` (confirmed in `src/fomo/settings.py:190-198`) | — |
| `telescope_runs.py`'s seeded `Observatory` records (Magellan-Clay/Baade, NTT, FTS) | Any manual/dev testing of the gap computation | ✓ | Seeded via Stage 1's `CreateObservatory` form / demo notebook | Tests should create their own fixture `Observatory` rows rather than depend on dev-DB seed data being present |
| Task queue (Celery/RQ/channels) | Only relevant if the planner chooses the "optional" async mitigation for Pitfall 1 | ✗ | — | Not required — synchronous view with honest UX copy is the recommended approach (see Pitfall 1) |

**Missing dependencies with no fallback:** None — everything the recommended (minimum-scope)
approach needs is already installed and configured.

**Missing dependencies with fallback:** Task queue infrastructure is absent but not needed for
the recommended synchronous approach.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (`django.test.TestCase`), per this codebase's established two-suite split |
| Config file | None dedicated — governed by `manage.py`'s `DJANGO_SETTINGS_MODULE=src.fomo.settings` |
| Quick run command | `./manage.py test solsys_code.tests.test_campaign_gap` (new file, once created) |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GAP-01 | `17-GAP-01-DECISION.md` documents the dark-window-only decision (D-02) | manual-only (a written artifact, not code) | N/A — verified by document review, not a test command | N/A |
| GAP-02 | `observable_dates()` returns dates with a non-zero dark window, skipping `ValueError` dates (D-03/D-04) | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestObservableDates -v 2` | ❌ Wave 0 |
| GAP-02 | `claimed_dates()` derives dates per D-05/D-06/D-07 and excludes cancelled/not-awarded/weather-failure runs | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClaimedDates -v 2` | ❌ Wave 0 |
| GAP-02 | Undated `CampaignRun`s (D-08) are flagged, not silently dropped or counted as claiming a date | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClaimedDates::test_undated_runs_flagged -v 2` | ❌ Wave 0 |
| GAP-02 | Gap view never computes inline on the table-view GET; is a separate endpoint (D-09) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_table_view_does_not_trigger_computation -v 2` | ❌ Wave 0 |
| GAP-02 | Cache hit avoids recomputation; TTL and "last computed at" behave correctly (D-10) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_cache_hit_skips_recomputation -v 2` | ❌ Wave 0 |
| GAP-02 | Date range clamps to 180-day max regardless of client input (D-11) | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClampDateRange -v 2` | ❌ Wave 0 |
| GAP-02 | Submitted `target_pk`/`site_pk` outside this campaign's scope are rejected (Pitfall 3, IDOR) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView::test_rejects_out_of_scope_target_and_site -v 2` | ❌ Wave 0 |
| GAP-02 | Gap-analysis button hidden/disabled when D-14 applies (no targets, or no resolved site) | integration | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisButton -v 2` | ❌ Wave 0 |
| GAP-01 (transitively) | No module in this phase imports `ephem_utils`/`solsys_code.views` at module scope | static check (not a pytest/Django test) | `grep -rn "import.*ephem_utils\|from solsys_code.views import" solsys_code/campaign_gap.py solsys_code/tests/test_campaign_gap.py` (expect zero output) | N/A |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_campaign_gap`
- **Per wave merge:** `./manage.py test solsys_code`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_campaign_gap.py` — new file, covers all GAP-02 rows above
- [ ] No new shared fixtures/conftest needed — existing `Observatory`/`CampaignRun`/`TargetList`
      factories (`NonSiderealTargetFactory` per CLAUDE.md) already cover this phase's fixture
      needs
- [ ] Framework install: none — Django `TestCase` is already the established framework for this
      codebase's DB-dependent tests

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Gap-analysis view is public/read-only, same posture as Phase 15's table view (`AUTH_STRATEGY='READ_ONLY'`) — no new auth surface |
| V3 Session Management | No | No session state introduced by this phase |
| V4 Access Control | Yes | Server-side re-validation that submitted `target_pk` belongs to the requested campaign and `site_pk` is among that campaign's actually-used sites — an object-level authorization check (see Pitfall 3, IDOR), not a role-based gate |
| V5 Input Validation | Yes | Date-range clamping (D-11, server-side regardless of client input) and `target_pk`/`site_pk` membership validation; both are plain Django ORM `.filter()`/`in` checks, no new validation library needed |
| V6 Cryptography | No | No secrets, tokens, or crypto operations in this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| IDOR via crafted `target_pk`/`site_pk` GET params (Pitfall 3) | Tampering / Elevation of Privilege (cross-campaign data leakage) | Re-derive the allowed target/site sets from the campaign server-side and validate membership before using either value in a query or cache key; `400 Bad Request` on mismatch |
| Unbounded date-range request causing resource exhaustion (D-11's own concern, `PITFALLS.md` Pitfall 5) | Denial of Service | Server-side clamp to 180 days regardless of client-supplied `end` param (Pattern 3 above) — this is *also* a security control, not just a UX one, given the ~520ms/call cost measured in Pitfall 1: an unbounded range could otherwise be used to tie up a worker process for minutes per request |
| Cache-key collision across campaigns/targets/sites leaking one campaign's gap result into another's view | Information Disclosure | The recommended cache key (`build_gap_cache_key`) includes all four dimensions (campaign, target, site, date range) explicitly — never omit any of them, even for the "auto-selected sole target" case (represent as `'none'` explicitly rather than omitting the segment) |

## Sources

### Primary (HIGH confidence)
- `solsys_code/telescope_runs.py` (this repo, read directly) — `sun_event()`/`get_site()`
  contract, `_find_crossing()`'s bisection-refinement structure
- `src/fomo/settings.py` (this repo, read directly, lines 190-198) — confirmed `CACHES` is
  already configured (`FileBasedCache`, `LOCATION=tempfile.gettempdir()`)
- `solsys_code/models.py` (this repo, read directly) — `CampaignRun` field definitions
  (`approval_status`, `run_status`, `obs_date`, `ut_start`/`ut_end`, `site` FK, `target` FK
  nullable)
- `solsys_code/campaign_views.py`, `campaign_tables.py`, `campaign_urls.py`, `campaign_utils.py`
  (this repo, read directly) — existing view/query/no-churn patterns this phase's new code
  mirrors
- Direct `cProfile` measurement against this repo's real seeded `Observatory` records
  (`obscode='268'`, Magellan-Clay) — 60-call average 523.9ms/call, single-call profile showing 21
  internal `_solar_altitude()` evaluations dominated by `erfa.core.epv00`/`transform_to` overhead
- `django/core/cache/__init__.py` (installed Django 5.2.15, read directly) — confirmed
  `cache = ConnectionProxy(caches, DEFAULT_CACHE_ALIAS)` — `cache` is a live proxy, not a bound
  reference, so `override_settings(CACHES=...)` correctly affects it
- `django/test/signals.py` (installed Django 5.2.15, read directly) — confirmed
  `clear_cache_handlers` `@receiver(setting_changed)` resets `caches`'s internal state when the
  `CACHES` setting changes via `override_settings`
- `.planning/phases/17-.../17-CONTEXT.md` — all D-01 through D-14 locked decisions
- `.planning/research/{ARCHITECTURE,PITFALLS,SUMMARY,STACK,FEATURES}.md` — pre-milestone research
  already settling GAP-01's dark-window-vs-altitude question
- `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-DECISION.md` — precedent shape
  for the `17-GAP-01-DECISION.md` artifact D-02 requires

### Secondary (MEDIUM confidence)
- [Django's cache framework — official docs](https://docs.djangoproject.com/en/6.0/topics/cache/)
  — low-level cache API shape, `cache.set`/`cache.get` signatures, `KEY_PREFIX` mechanism
  (via WebSearch, cross-referenced against this repo's own installed Django source)
- [TIL Django doesn't flush caches between tests — David Winterbottom](https://til.codeinthehole.com/posts/django-doesnt-flush-caches-between-tests/)
  and Django's own ticket tracker (#11505, #17995, #20075) — corroborates the cache-test-isolation
  gotcha (cross-checked against this repo's direct source read, which additionally clarified that
  `override_settings(CACHES=...)` alone IS sufficient, contrary to this source's more cautious
  framing)

### Tertiary (LOW confidence)
- None — every non-primary claim in this research was cross-checked against direct source reads
  in this repo's own installed packages/codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages, everything already installed/configured, confirmed by
  direct file reads
- Architecture: HIGH — directly mirrors Phases 15/16's established `campaign_*` module patterns
- Pitfalls: HIGH for the performance finding (directly measured in this repo) and the cache-test-
  isolation finding (directly confirmed via Django source read); MEDIUM for the multi-target
  attribution recommendation (a reasoned recommendation, not a locked decision or empirically
  testable fact)

**Research date:** 2026-07-04
**Valid until:** 30 days (stable Django/astropy versions; the `sun_event()` performance
measurement is machine-dependent and should be re-verified on the actual deployment target if
the ~1-minute worst-case latency becomes a real UX concern)
