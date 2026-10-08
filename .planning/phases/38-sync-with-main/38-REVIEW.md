---
phase: 38-sync-with-main
reviewed: 2026-10-08T00:06:19Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - solsys_code/tests/test_urls.py
  - src/fomo/urls.py
findings:
  critical: 0
  warning: 0
  info: 2
  total: 2
status: issues_found
---

# Phase 38: Code Review Report (re-review after gap-closure plans 38-05 and 38-06)

**Reviewed:** 2026-10-08T00:06:19Z
**Depth:** deep
**Files Reviewed:** 2
**Status:** issues_found

## Summary

This is an incremental re-review. It covers only the source changes made since the previous review
(`9d916d1`): commits `32dafa2` (test) and `a4d77f2` (fix) from plan 38-05. Plan 38-06 changed no
source files. The previous review's other findings (WR-01, WR-02, IN-01 to IN-04) concern files
outside this scope. They are not re-reported here and remain as recorded in
`38-REVIEW-DISPOSITION.md`. The new findings are numbered from IN-05 so they do not reuse an ID that
already has a disposition row.

**CR-01 is fixed.** I checked this against the installed tomtoolkit 3.1.0 / Django 5.2.17
environment as well as the diff:

- `src/fomo/urls.py` no longer contains the `alerts/` include or its incorrect comment. Every
  remaining shadow route (`targets/`, `targets/export/`, `calendar/`, `campaigns/`,
  `users/<pk>/delete/`) still comes before `path('', include('tom_common.urls'))`.
- No `alerts:` or `tom_alerts:` reversal is left that could now raise `NoReverseMatch`. I searched
  `src/templates`, `solsys_code` and every installed site-package outside `tom_alerts` itself.
  tomtoolkit 3.1.0's `tom_common.urls` does not register `tom_alerts`. `tom_alerts` is not
  installed, so the plugin loop (`include_url_paths`) cannot add it either.
- `python manage.py test solsys_code.tests.test_urls` passes (4 tests). The committed red evidence
  (`38-05-red-evidence.json`) shows all three alerts tests failing for the right reasons against
  the unfixed urlconf (`Resolver404 not raised`, `NoReverseMatch not raised`, `500 != 404`), while
  the route-order test passed.
- `pre-commit run ruff` and `ruff-format` pass on both files.

**How I tested the new tests.** I ran `solsys_code.tests.test_urls` against three mutated copies of
the urlconf, built in the session scratchpad and loaded through `ROOT_URLCONF`. No source file was
touched.

| Mutation | Result |
|----------|--------|
| `path('alerts/', include('tom_alerts.urls'))` restored with no `namespace=` | Caught by 2 of 3 alerts tests; the namespace test passes without testing anything (IN-06) |
| `path('brokers/', include('tom_alerts.urls', namespace='alerts'))` | Caught by the namespace test only |
| `targets/export/` shadow moved after `tom_common.urls` | **Not caught** by this module (IN-05) |

Both gaps are Info rather than Warning. A restored `alerts/` route is caught in every form a merge
would plausibly produce, and the export shadow route is already checked by behaviour elsewhere
(`solsys_code/tests/test_scout_views.py:350-357`).

## Narrative Findings (AI reviewer)

## Info

### IN-05: `TestProjectRoutesStillResolve` does not check the `targets/export/` shadow route, though its docstring says it checks the routes that come before `tom_common.urls`

**File:** `solsys_code/tests/test_urls.py:38-56` (route at `src/fomo/urls.py:19`)
**Issue:** The class docstring says "Routes registered before tom_common.urls still win over
tom_common's own". The test checks `/targets/`, `/calendar/` and `/users/1/delete/`, which are real
shadows of tom_common routes. It also checks `/scout/rubin-too*` and `/campaigns/`, which have no
tom_common counterpart, so no ordering mistake could affect them. It does not check
`/targets/export/`. That route shadows `tom_targets.urls`' `export` and is the one named in the
comment at `src/fomo/urls.py:14-17`.

When I moved the `targets/export/` line below `tom_common.urls`, this module still passed. The
reorder is caught today only by the behaviour test
`TestScoutTargetFilter.test_export_honours_scout_filter` in `test_scout_views.py`, which this
module's docstring does not mention. Separately, all six checks sit in one test method, so the
first failure hides the rest.

**Fix:** Add the missing shadow route, and use `subTest` so each route is reported on its own:

```python
from solsys_code.scout_views import ScoutTargetExportView, ScoutTargetListView

    def test_project_routes_resolve_to_fomo_and_main_views(self) -> None:
        cases = [
            ('/targets/', ScoutTargetListView),
            ('/targets/export/', ScoutTargetExportView),
            ('/campaigns/', CampaignListView),
            ('/users/1/delete/', ProtectedUserDeleteView),
        ]
        for path_, view_class in cases:
            with self.subTest(path=path_):
                self.assertIs(resolve(path_).func.view_class, view_class)
        self.assertIs(resolve('/calendar/').func, fomo_render_calendar)
```

Alternatively, narrow the docstring so it does not claim to cover every shadow route.

### IN-06: The namespace test only checks the `alerts` instance namespace, not the `tom_alerts` app namespace that `tom_alerts` itself uses

**File:** `solsys_code/tests/test_urls.py:24-26`
**Issue:** `tom_alerts/urls.py` declares `app_name = 'tom_alerts'`, and its own views reverse
`tom_alerts:list`, `tom_alerts:run` and so on. `reverse('alerts:list')` only fails while there is no
*instance* namespace called `alerts`. If a later merge restores the include without
`namespace='alerts'` (for example `include('tom_alerts.urls')`), the app namespace `tom_alerts` comes
back, but this test still passes because it is not testing the right name. I confirmed this with
mutation 1 above.

The path tests at lines 20-35 still catch that case at the `/alerts/` prefix. So the only form of
restoration that nothing catches is an include with no namespace under a different prefix. The gap
is small, but the test's stated purpose ("the alerts/ include ... must stay gone") covers it.

**Fix:** Assert that neither namespace reverses:

```python
    def test_alerts_namespace_cannot_be_reversed(self) -> None:
        for name in ('alerts:list', 'tom_alerts:list'):
            with self.subTest(name=name), self.assertRaises(NoReverseMatch):
                reverse(name)
```

---

_Reviewed: 2026-10-08T00:06:19Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
