# Phase 39: Calendar Write Access - Pattern Map

**Mapped:** 2026-10-08
**Files analyzed:** 9
**Analogs found:** 8 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `solsys_code/calendar_urls.py` (modify) | route/config | request-response | `solsys_code/mixins.py` (`user_passes_test` gate) + itself | role-match |
| `solsys_code/calendar_views.py` (optional new, method-aware wrapper) | middleware (view decorator) | request-response | `solsys_code/mixins.py:StaffRequiredMixin` | partial |
| `src/templates/tom_calendar/partials/calendar.html` (modify) | component (template) | request-response | itself (lines 217-225, 232-238); `event_form.html:224` for `request.user` gating | exact |
| `src/templates/tom_calendar/partials/event_form.html` (modify) | component (template) | request-response | itself (`:224` `request.user.is_staff` branch) | exact |
| `solsys_code/tests/test_calendar_template.py` (modify) | test | request-response | `EventModalSeriesDecorationTest` (:1302+) | exact |
| new test class for the five write routes (same file or new `test_calendar_write_access.py`) | test | request-response | `EventModalSeriesDecorationTest` | role-match |
| `solsys_code/tests/test_bootstrap5_rendering.py` (modify) | test (Playwright) | request-response | `TestBootstrap5Rendering` (:44-100, :277-290) | exact |
| `docs/runbooks/telescope_runs_calendar.rst` (modify) | docs | n/a | existing pop-up section (~line 2567) | exact |
| the two v2.4 review ledgers (modify) | docs | n/a | existing WR-05 rows | exact |

## Pattern Assignments

### `solsys_code/calendar_urls.py` (route, request-response)

**Current file (whole body, lines 9-26):**
```python
from django.urls import path
from tom_calendar.views import create_event, create_todo, delete_event, update_event, update_todo

from solsys_code.views import fomo_render_calendar

app_name = 'calendar'

urlpatterns = [
    path('', fomo_render_calendar, name='calendar'),
    path('create/', create_event, name='create-event'),
    path('update/<int:event_id>/', update_event, name='update-event'),
    path('delete/<int:event_id>/', delete_event, name='delete-event'),
    path('todo/create/<int:event_id>/', create_todo, name='create-todo'),
    path('todo/update/<int:todo_id>/', update_todo, name='update-todo'),
]
```
Real URL paths are `create/`, `update/<id>/`, `delete/<id>/`, `todo/create/<id>/`, `todo/update/<id>/` (not `create-event/`). Tests must hit these real paths, and reverse by name `calendar:create-event` etc.

**Guard analog** (`solsys_code/mixins.py:1-12`, the repo's only redirect-to-LOGIN_URL pattern):
```python
from django.contrib.auth.decorators import user_passes_test
from django.utils.decorators import method_decorator

class StaffRequiredMixin:
    @method_decorator(user_passes_test(lambda u: u.is_staff))
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```
Function-view equivalent for D-01/D-02/D-06: `django.contrib.auth.decorators.login_required` wrapped directly: `path('create/', login_required(create_event), name='create-event')`, same for `delete_event`, `create_todo`, `update_todo`. `login_required` gives the 302 to `settings.LOGIN_URL` (`/accounts/login/`, settings.py:123) with `?next=` (D-08).

**Method-aware wrapper for `update_event` (D-04)** (no existing analog in the repo; write with `functools.wraps`):
```python
def login_required_for_post(view):
    guarded = login_required(view)
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if request.method == 'POST':
            return guarded(request, *args, **kwargs)
        return view(request, *args, **kwargs)
    return wrapper
```
Place inline in `calendar_urls.py` or in `solsys_code/calendar_views.py`. Must not duplicate upstream logic. Update the module docstring (it currently says "delegate ... unchanged").

---

### `src/templates/tom_calendar/partials/calendar.html` (template)

**Two click targets to remove for anonymous (D-07)**, lines 217-225 and 232-238:
```html
<button class="btn btn-outline-secondary btn-sm"
        hx-get="{% url 'calendar:create-event' %}"
        hx-target="#cal-modal-body"
        hx-on::after-request="bootstrap.Modal.getOrCreateInstance(document.getElementById('cal-modal')).show();"
>
  + New Event
</button>
...
<div class="cal-day{% if ... %} ... border-right border-bottom p-1"
     hx-get="{% url 'calendar:create-event' %}?date={{ day.date|date:'Y-m-d' }}"
     hx-target="#cal-modal-body"
     hx-on::after-request="...show();"
>
```
Wrap the button in `{% if request.user.is_authenticated %}...{% endif %}` and wrap just the three `hx-*` attribute lines on the day cell in the same `{% if %}` (inside the tag). Event rows (`hx-get="{% url 'calendar:update-event' event.id %}"`, lines 263/302/309) stay untouched. Note the template is rendered by `fomo_render_calendar` (check that `request` is in context; `event_form.html` already uses `request.user.is_staff` at :224, and `calendar_display_extras._viewer_is_authenticated(context)` reads `context['user']` as a fallback pattern).

**Bootstrap 4 tidy (D-11)** lines to change: 34 `var(--white)` -> `var(--bs-white)`; 227 `border-left` -> `border-start`; 229 `font-weight-bold border-right` -> `fw-bold border-end`; 232 `border-right` -> `border-end` (day cell class string); 334 `mr-2` -> `me-2`; 340, 347, 362(?), 366 `mr-3` -> `me-3`. Lines 28/35/85/159 `font-weight:` are CSS properties, leave. Keep `data-url`. Re-grep after editing: `grep -n "border-left\|border-right\|mr-\|ml-\|font-weight-\|--white"`.

---

### `src/templates/tom_calendar/partials/event_form.html` (template)

**Header comment (lines 1-26, `{% comment %}`)**: rewrite per D-10 as a numbered list pinned to tomtoolkit 3.1.0 (currently says 3.0.1 and "exact copy ... with one new block"). Keep syntax safe: `TemplateCommentSyntaxSweepTest` (test_calendar_template.py:894) sweeps templates for comment syntax problems, so do not put `{% ... %}` or `#}` text inside the comment.

**Structure to branch on** (line numbers current):
- 27 `{% load django_bootstrap5 attribution_display_extras calendar_display_extras %}`
- 28-103 the `<form ...hx-post...>` including Save (93), Save and edit (95), Delete (97-101) buttons
- 104-~222 the series, attributed-run blocks (render once, for both editors and visitors)
- 224 `{% elif not event.telescope_label_meta.run and request.user.is_staff %}` (staff candidate hint)
- 253-261 todo list: `{% include 'tom_calendar/partials/todos.html' with event=event %}` when `action == "update"`

Existing gating idiom to copy (`request.user` read directly, no context flag):
```
{% elif not event.telescope_label_meta.run and request.user.is_staff %}
```
Apply `{% if request.user.is_authenticated %}<form>...</form>{% else %}<div class="card ..."> plain-text title/start/end/description/URL(is_web_url link)/targets/user/proposal/telescope/instrument + read-only todo list {% endif %}` (D-05). The upstream `todos.html` is not overridden; for the read-only todo list either inline a loop over `event.todos.all` (check the related name in installed `tom_calendar/models.py`) in the else branch, or add a third override and list it in the header.

**Existing anonymous series gating to be aware of:** `solsys_code/templatetags/calendar_display_extras.py:756-774` `_viewer_is_authenticated(context)` and :841 hide the observation-series group name from anonymous viewers (see test at test_calendar_template.py:1440+ "hides_group_name_from_anonymous_viewer"). CONTEXT says the series line "renders unchanged" for visitors, so keep that filter as is; do not change its visibility rule.

---

### Tests: `solsys_code/tests/test_calendar_template.py` (and/or new module) (test, request-response)

**Analog:** `EventModalSeriesDecorationTest` (lines 1302-1335, 1420-1440). Fixture + login idiom:
```python
@classmethod
def setUpTestData(cls) -> None:
    cls.target = NonSiderealTargetFactory.create()   # never SiderealTargetFactory (CLAUDE.md)
    ...
    cls.authenticated_user = User.objects.create_user(username='seriesmodalviewer', password='pw')

def test_...(self):
    event = CalendarEvent.objects.create(title=..., start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc), end_time=...)
    self.client.force_login(self.authenticated_user)
    response = self.client.get(self._modal_url(event))
    self.assertEqual(response.status_code, 200)
    content = response.content.decode()
    self.assertIn(..., content)
```
Anonymous variant: omit `force_login` (the `TestCase` client is anonymous by default). Other `force_login(self.staff_user)` uses at :551, :833, :1197.

New test cases to write (D-08/D-09, security gate):
- For each of the 5 routes, anonymous `POST` to the real path (`reverse('calendar:create-event')` etc., plus a literal `/calendar/create/` to prove the shadowing claim) -> `assertEqual(response.status_code, 302)`, `assertTrue(response['Location'].startswith(settings.LOGIN_URL))`, `assertIn('next=', ...)`, and `CalendarEvent.objects.count()` / `EventTodo.objects.count()` and the target row's fields unchanged.
- Anonymous `GET` create -> 302; anonymous `GET` update -> 200 with no `<form`, no `Delete`, no `hx-post`; logged-in `GET` update contains the form (editor unchanged).
- Anonymous `GET /calendar/` (`reverse('calendar:calendar')`) contains no `calendar:create-event` URL and no "+ New Event"; logged-in contains both.
- Optional: send header `HTTP_HX_REQUEST='true'` and assert middleware `HX-Redirect` (`tom_common.middleware.HTMXRedirectMiddleware`, in `TOMTOOLKIT_MIDDLEWARE`).
- Template comment sweep (`TemplateCommentSyntaxSweepTest`, :894) already covers the edited templates.

### `solsys_code/tests/test_bootstrap5_rendering.py` (Playwright, `@tag('functional')`)

**Class scaffold to copy** (lines 44-75): `@tag('functional') class ...(StaticLiveServerTestCase)` with `setUpClass` setting `DJANGO_ALLOW_ASYNC_UNSAFE`, `sync_playwright().start()`, `chromium.launch(headless=True)`; `tearDownClass` closing browser/stopping playwright and restoring env; `setUp` doing `self.page = self.browser.new_page()` and building fixtures in `setUp` (NOT `setUpTestData`, TransactionTestCase). Reuse the existing `attributed_event` fixture (:80-100, month `CAL_YEAR=2026`, `CAL_MONTH=8`) and the existing calendar modal-open tests in this class as models for navigating to `/calendar/?month=8&year=2026` and clicking an event.

**Session-cookie hand-off** (lines 277-290), for the logged-in round trip; skip it for the anonymous test (fresh `new_page()` has no cookie):
```python
self.client.force_login(staff)
self.page.context.add_cookies([{
    'name': settings.SESSION_COOKIE_NAME,
    'value': self.client.cookies[settings.SESSION_COOKIE_NAME].value,
    'url': self.live_server_url,
}])
page_errors = []
self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))
self.page.goto(f'{self.live_server_url}{reverse(...)}')
...
assert page_errors == []
```
Put the new test inside the existing class (shares browser) or a sibling class with the same scaffold.

---

### Docs and ledgers
- `docs/runbooks/telescope_runs_calendar.rst` ~line 2567 ("Clicking a calendar entry opens a pop-up"): add read-only-when-logged-out sentence; grep the file for "add events"/"New Event" for other lines to update. Paired doc per CLAUDE.md; no notebook pairs with these templates.
- `.planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md`: WR-05 row -> `fixed`.
- `.planning/milestones/v2.4-phases/33-series-identity-reconciler-inversion/33-REVIEW.md` WR-05: add "fixed in Phase 39 (ACCESS-01)".

## Shared Patterns

### Login redirect semantics
**Source:** `django.contrib.auth.decorators.login_required` (as used via `user_passes_test` in `solsys_code/mixins.py`). **Apply to:** all five write routes; uses `LOGIN_URL = '/accounts/login/'` (`src/fomo/settings.py:123`). `AUTH_STRATEGY = 'READ_ONLY'` (:340).

### Template-side viewer gating
**Source:** `event_form.html:224` (`request.user.is_staff`) and `calendar_display_extras.py:756-774` `_viewer_is_authenticated`. **Apply to:** both template edits; read `request.user.is_authenticated` directly, no new context variable required.

### Test conventions
`django.test.TestCase`, `User.objects.create_user(username=..., password='pw')`, `self.client.force_login(user)`, `NonSiderealTargetFactory`, Django runner only (`python manage.py test solsys_code.tests.test_calendar_template`). Single quotes, 120 cols, ruff via pre-commit.

## No Analog Found

| File | Role | Reason |
|---|---|---|
| method-aware `functools.wraps` login wrapper for `update_event` | decorator | No `functools.wraps` view decorator or `login_required` function-view usage exists in `solsys_code/`; use the sketch above. |

## Metadata

**Analog search scope:** `solsys_code/` (py, templatetags, tests), `src/templates/tom_calendar/partials/`, `src/fomo/settings.py`
**Tracked-source check:** `git ls-files src/templates/tom_calendar` lists calendar.html, campaign_chip.html, event_form.html (all tracked); all analog paths are tracked source, none are `.gsd` mirrors.
**Pattern extraction date:** 2026-10-08
