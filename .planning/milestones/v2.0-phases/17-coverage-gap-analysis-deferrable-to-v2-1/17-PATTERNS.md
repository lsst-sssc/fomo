# Phase 17: Coverage-Gap Analysis (Deferrable to v2.1) - Pattern Map

**Mapped:** 2026-07-04
**Files analyzed:** 8 (new: 4, modified: 3, decision doc: 1)
**Analogs found:** 8 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/campaign_gap.py` | service/utility | transform (date-set computation) + CRUD (read `CampaignRun`) | `solsys_code/campaign_utils.py` | exact (same role: pure-logic helper module paired with a views/urls module, "never raise for expected messy data" discipline) |
| `solsys_code/campaign_views.py` (MODIFIED: + `CampaignGapAnalysisView`) | controller | request-response, cache-or-compute | `solsys_code/campaign_views.py` itself — `CampaignRunTableView`/`ApprovalQueueView` | exact (same file, same module conventions, same "never import `solsys_code.views`" discipline) |
| `solsys_code/campaign_urls.py` (MODIFIED: + 1 path) | route | request-response | `solsys_code/campaign_urls.py` itself | exact |
| `solsys_code/campaign_forms.py` (MODIFIED: + gap selection form, if planner chooses a `Form`) | component/form | request-response | `solsys_code/campaign_forms.py` — `CampaignRunSubmissionForm` | exact |
| `campaigns/campaignrun_gap_analysis.html` (NEW template, or section of `campaignrun_table.html`) | component (template) | request-response | `src/templates/campaigns/campaignrun_table.html` (existing table template `CampaignRunTableView` renders) | role-match |
| `solsys_code/tests/test_campaign_gap.py` | test | CRUD (DB-dependent) | `solsys_code/tests/test_campaign_views.py` | exact |
| `.planning/phases/17-.../17-GAP-01-DECISION.md` | config/doc (decision artifact) | N/A | `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-DECISION.md` | exact |
| Data source for cache: `django.core.cache.cache` (no new file) | — | — | `src/fomo/settings.py:190-198` (`CACHES` config, already present) | exact (config already exists, no changes needed) |

## Pattern Assignments

### `solsys_code/campaign_gap.py` (service/utility, transform + CRUD-read)

**Analog:** `solsys_code/campaign_utils.py`

**Imports pattern** (lines 1-19 of `campaign_utils.py`):
```python
"""Shared helpers for the campaign-coordination CSV bootstrap import.
...mirrors calendar_utils.py's role...every function here is structured as
"never raise for expected messy data; return a usable value plus an explicit flag"...
"""

import re
from datetime import date, datetime
from datetime import timezone as dt_timezone
from typing import Any

import requests
from django.db.utils import IntegrityError
from tom_dataservices.dataservices import MissingDataException

from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.solsys_code_observatory.utils import MPCObscodeFetcher
```
For `campaign_gap.py`, mirror this shape but import `telescope_runs.sun_event`/`get_site` (never
`ephem_utils`), `django.core.cache.cache`, and `CampaignRun`:
```python
import logging
from datetime import date, timedelta

from django.core.cache import cache
from django.utils import timezone

from solsys_code.models import CampaignRun
from solsys_code.telescope_runs import sun_event  # NOT ephem_utils -- CLAUDE.md constraint

logger = logging.getLogger(__name__)
```

**"Never raise for expected messy data" pattern to copy** — `_STATUS_MAP`/status translation
(`campaign_utils.py` lines 60-72): a substring-match, most-specific-first table with a
conservative default, no exception for unrecognized input. Reuse this same shape for D-05's
run_status exclusion set (`{cancelled, not_awarded, weather_tech_failure}`) — a plain frozenset
membership check, not a translation table, since these are exact enum members already:
```python
_EXCLUDED_RUN_STATUSES = {
    CampaignRun.RunStatus.CANCELLED,
    CampaignRun.RunStatus.NOT_AWARDED,
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE,
}
```

**Per-date log+skip pattern** (D-03) — directly from RESEARCH.md's own Pattern 2 (already
verified against this repo's conventions in `load_telescope_runs.py`/`campaign_utils.py`):
```python
def observable_dates(site, start: date, end: date) -> set[date]:
    observable = set()
    n_days = (end - start).days + 1
    for i in range(n_days):
        d = start + timedelta(days=i)
        try:
            sun_event(site, d, kind='dark')
            observable.add(d)
        except ValueError:
            logger.debug('sun_event(dark) raised for site=%s date=%s; skipping as unknown (D-03).', site, d)
    return observable
```

**Cache-or-compute pattern** (D-10) — RESEARCH.md's Pattern 1, verified against
`django/core/cache/__init__.py` in this repo's installed Django:
```python
GAP_CACHE_TTL_SECONDS = 3600  # D-10


def build_gap_cache_key(campaign_pk: int, target_pk: int | None, site_pk: int, start: date, end: date) -> str:
    return f'campaign_gap:{campaign_pk}:{target_pk if target_pk is not None else "none"}:{site_pk}:{start.isoformat()}:{end.isoformat()}'


def get_or_compute_gap(campaign, target, site, start: date, end: date) -> dict:
    key = build_gap_cache_key(campaign.pk, target.pk if target else None, site.pk, start, end)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _compute_gap(campaign, target, site, start, end)
    result['computed_at'] = timezone.now()
    cache.set(key, result, timeout=GAP_CACHE_TTL_SECONDS)
    return result
```

**Claimed-date query pattern (D-05/D-06/D-07)** — copy the `resolve_site()` 3-tier discipline's
"never silently guess" spirit and the `import_campaign_csv.py` single-target auto-assign
precedent (line 69):
```python
# solsys_code/management/commands/import_campaign_csv.py:69
auto_target = campaign.targets.first() if campaign.targets.count() == 1 else None
```
Directly reusable for D-12 in the new view (see below).

---

### `solsys_code/campaign_views.py` (MODIFIED: + `CampaignGapAnalysisView`)

**Analog:** same file — `CampaignRunTableView` (lines 63-100) and `CampaignRunDecisionView`
(lines 260-335), both already present in this file.

**Class docstring / "never import `solsys_code.views`" discipline** (file header, lines 1-9):
```python
"""Views for the per-campaign table read path (VIEW-01/02/03/04) and the public submission
write path (SUBMIT-01/04/05).

Views: ... Deliberately does not import ``solsys_code.views`` -- that module imports
``.ephem_utils`` at module load time, which triggers a ~1.6 GB SPICE kernel download
(CLAUDE.md "Heavy import side effect").
"""
```
Copy verbatim as the module-docstring convention: `CampaignGapAnalysisView` must state
explicitly (and be verifiable via the grep in RESEARCH.md's Anti-Patterns section) that it
imports `telescope_runs`, never `ephem_utils`/`solsys_code.views`.

**`get_context_data` + `get_object_or_404` pattern** (`CampaignRunTableView`, lines 96-100):
```python
def get_context_data(self, **kwargs):
    """Add the campaign (TargetList) to context for the page heading."""
    context = super().get_context_data(**kwargs)
    context['campaign'] = get_object_or_404(TargetList, pk=self.kwargs['pk'])
    return context
```

**HttpResponseBadRequest / IDOR-rejection pattern** (`CampaignRunDecisionView.post`, lines
272-277) — directly reusable for Pitfall 3's server-side `target_pk`/`site_pk` membership
validation:
```python
def post(self, request, pk):
    action = request.POST.get('action')
    if action not in ('approve', 'reject'):
        return HttpResponseBadRequest()
```
`CampaignGapAnalysisView` should mirror this exact "validate against the actual server-derived
allowed set, `HttpResponseBadRequest()` on mismatch" shape for `target_pk`/`site_pk` GET params.

**Error-recovery / try-except-log-message pattern** (`CampaignRunDecisionView`, lines 282-323):
copy the `logger.exception(...)` + `messages.error(request, ...)` + redirect shape if the gap
computation needs a user-facing failure path (e.g. `Observatory.timezone` unset mid-loop — though
D-03 already handles this per-date via `ValueError`, so this may not be needed at the top level).

---

### `solsys_code/campaign_urls.py` (MODIFIED: + 1 path)

**Analog:** same file (lines 1-29) — flat `app_name` + `urlpatterns` list, one `path()` per view,
`<int:pk>/` reused as the campaign-scoping segment:
```python
app_name = 'campaigns'

urlpatterns = [
    path('', CampaignListView.as_view(), name='list'),
    ...
    path('<int:pk>/', CampaignRunTableView.as_view(), name='table'),
]
```
Add e.g. `path('<int:pk>/gaps/', CampaignGapAnalysisView.as_view(), name='gap_analysis')` —
matches the existing `<int:pk>/` campaign-scoping convention; exact segment name is Claude's
Discretion per CONTEXT.md.

---

### `solsys_code/campaign_forms.py` (MODIFIED: + gap selection form, if chosen)

**Analog:** same file — `CampaignRunSubmissionForm` (lines 1-60).

**Plain `forms.Form` + crispy layout pattern** (lines 15-60):
```python
class CampaignRunSubmissionForm(forms.Form):
    campaign = forms.ModelChoiceField(queryset=TargetList.objects.all(), required=True)
    ...
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            'campaign',
            Fieldset('Run details', ...),
            FormActions(Submit('submit', 'Submit run for review')),
        )
```
A new `CampaignGapAnalysisForm` (if the planner opts for a real `Form` per RESEARCH.md's Open
Question 1) should follow this exact shape: plain `forms.Form` (not `ModelForm`), a `ModelChoiceField`
for target/site scoped to the campaign's own querysets (populated in the view via `__init__(self,
*args, campaign=None, **kwargs)` rather than a class-level unscoped queryset — note
`CampaignRunSubmissionForm`'s `campaign` field uses `TargetList.objects.all()` unscoped since it's
public intake; the gap form's target/site fields MUST instead be scoped per-campaign at
instantiation time, matching D-12/D-13's dropdown-population rules).

---

### `campaigns/campaignrun_gap_analysis.html` (NEW template, or table-page section)

**Analog:** `src/templates/campaigns/campaignrun_table.html` (the template
`CampaignRunTableView` renders) — read this template directly before writing the new one/section
to match its heading/breadcrumb/table-styling conventions. (Not re-read in full here since its
content is template-layer boilerplate rather than load-bearing logic; the planner/executor should
`Read` it directly when writing the gap-analysis template to match block names and Bootstrap 4
markup conventions already established.)

---

### `solsys_code/tests/test_campaign_gap.py` (NEW)

**Analog:** `solsys_code/tests/test_campaign_views.py` (lines 1-60+).

**Fixture-base pattern to copy** (lines 1-52):
```python
from datetime import date, datetime, timedelta, timezone

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.models import CampaignRun

class CampaignViewTestBase(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.empty_campaign = TargetList.objects.create(name='Empty Campaign')
        cls.staff_user = User.objects.create_user(username='staffcoordinator', password='pw', is_staff=True)
        ...
```
Copy this shared-fixture-base shape (`setUpTestData`, `TargetList.objects.create`, staff `User`
fixture) for `TestCampaignGapAnalysis`'s base class. **Always use
`tom_targets.tests.factories.NonSiderealTargetFactory`** when fixturing a `Target` — never
`SiderealTargetFactory` (CLAUDE.md, applies to this subagent's PATTERNS.md too).

**Cache test-isolation pattern** (RESEARCH.md's own Code Examples section, mirrored from
`solsys_code/tests/test_sync_gemini_observation_calendar.py:61-62`'s
`@override_settings(FACILITIES=GEM_SETTINGS)` precedent):
```python
from django.core.cache import cache
from django.test import TestCase, override_settings

TEST_CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}


@override_settings(CACHES=TEST_CACHES)
class TestCampaignGapAnalysis(TestCase):
    def setUp(self):
        cache.clear()  # belt-and-suspenders
```

---

### `.planning/phases/17-.../17-GAP-01-DECISION.md` (NEW decision artifact, D-02)

**Analog:** `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-DECISION.md`.

**Shape to loosely follow** (lines 1-11 + "Findings" section headers):
```markdown
# Phase 13: ESO Feasibility Spike - Decision

**Investigated:** 2026-07-01
**Status:** Complete. Findings recorded (...); Recommendation (...) completed in Plan 02.

This phase is investigation-only. ...

## Findings

### ESO-01 — Credential obtainability & usability
...
```
For `17-GAP-01-DECISION.md`: title `# Phase 17: Coverage-Gap Analysis - GAP-01 Decision`,
`**Decided:**` date, `**Status:**` line, then a `## Decision` section stating dark-window-only
(D-01) with citations to the pre-milestone research docs already unanimous on this (per D-01's own
text) — this is a "decision was reached via discussion, documented after the fact" artifact
(CONTEXT.md D-02), not a multi-day investigation log like Phase 13's, so it can be much shorter;
follow the *shape* (title/status/decision-with-rationale headers), not the *length*.

## Shared Patterns

### "Never import solsys_code.views / ephem_utils at module scope"
**Source:** `solsys_code/campaign_views.py` lines 1-9 (module docstring) and RESEARCH.md's
Anti-Patterns section (grep-based verification command).
**Apply to:** `campaign_gap.py`, `campaign_views.py`'s new view, `test_campaign_gap.py`.
```text
grep -rn "import.*ephem_utils\|from solsys_code.views import" solsys_code/campaign_gap.py \
    solsys_code/tests/test_campaign_gap.py
# must return zero output
```

### Cache-based result storage with explicit "computed_at" (D-10)
**Source:** `src/fomo/settings.py:190-198` (`CACHES` already configured, no changes needed) +
RESEARCH.md Pattern 1.
**Apply to:** `campaign_gap.py` (`get_or_compute_gap`), `campaign_views.py`'s new view (rendering
"last computed at" in context), the new template.

### Server-side IDOR re-validation of GET params scoped by a campaign
**Source:** `solsys_code/campaign_views.py` — `CampaignRunDecisionView.post`'s
`HttpResponseBadRequest()` pattern (lines 274-276), and `import_campaign_csv.py:69`'s
`campaign.targets.first() if campaign.targets.count() == 1 else None` (D-12 precedent).
**Apply to:** `CampaignGapAnalysisView` — re-derive allowed target/site sets from the campaign
server-side before trusting any submitted `target_pk`/`site_pk`.

### Per-record "log+skip, never abort" error handling
**Source:** `solsys_code/campaign_utils.py`'s status-translation discipline; RESEARCH.md Pattern 2
(directly reusable code).
**Apply to:** `campaign_gap.py`'s `observable_dates()` (D-03's per-date `ValueError` skip).

### Server-side clamping of client-supplied ranges (D-11)
**Source:** RESEARCH.md Pattern 3 (new code, no direct existing-codebase analog for date-range
clamping specifically, but follows the same "never trust client input, always re-derive/clamp
server-side" discipline as the IDOR pattern above).
**Apply to:** `campaign_gap.py` or `campaign_views.py`'s new view — `clamp_date_range()`.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Date-range clamping logic (`clamp_date_range` in `campaign_gap.py`) | utility | transform | No existing date-range-clamping code in this codebase; this is genuinely new logic (RESEARCH.md Pattern 3 supplies the reference implementation instead of an in-repo analog) |
| Optional batched/vectorized `sun_event()` variant (RESEARCH.md Pitfall 1, item 2 — "should not must") | utility | batch/transform | Explicitly out of this phase's minimum scope per RESEARCH.md's Assumptions Log (A3); no analog needed unless planner chooses to include it |

## Metadata

**Analog search scope:** `solsys_code/` (campaign_*.py trio, telescope_runs.py, models.py,
tests/), `src/templates/campaigns/`, `src/fomo/settings.py`, `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/`
**Files scanned:** `telescope_runs.py`, `campaign_views.py`, `campaign_utils.py`,
`campaign_urls.py`, `campaign_forms.py`, `models.py`, `test_campaign_views.py`,
`test_sync_gemini_observation_calendar.py`, `import_campaign_csv.py`, `settings.py`,
`13-DECISION.md`
**Pattern extraction date:** 2026-07-04
