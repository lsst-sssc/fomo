"""Login guards for the calendar write views (ACCESS-01, Phase 33 review WR-05).

tomtoolkit 3.1.0's ``tom_calendar.views`` functions carry no login check and act on any HTTP method, so
an anonymous request could create, change or delete a calendar event or todo. FOMO wraps those views in its
own URL conf (``solsys_code/calendar_urls.py``) with the two decorators below; the vendored package is never
edited and no upstream view body is re-implemented here. The decorators only decide allow-or-redirect.

Decisions: any logged-in user may write (D-01); the event pop-up is the GET of update-event, so that route
stays open for reading (D-04); a refused write is a redirect to the login page, never a 403 (D-08). The
login ``next`` is the calendar page, not the refused URL: delete-event and update-todo act on a plain GET
upstream, so a ``next`` pointing at them would replay the write as soon as the visitor logged in.

The decorators keep no state between requests: they decide from ``request.user`` and ``request.method`` alone.
"""

from collections.abc import Callable
from functools import wraps

from django.contrib.auth.views import redirect_to_login
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.urls import reverse

_READ_METHODS = ('GET', 'HEAD')


def _login_redirect() -> HttpResponseRedirect:
    """Redirect to the login page, returning to the calendar page afterwards.

    ``reverse`` is evaluated per call, never at import time, because this module is imported while the URL
    conf itself is still loading.

    Returns:
        A redirect to ``settings.LOGIN_URL`` whose ``next`` is the calendar page (never the refused URL).
    """
    return redirect_to_login(reverse('calendar:calendar'))


def write_requires_login(view: Callable[..., HttpResponse]) -> Callable[..., HttpResponse]:
    """Require a signed-in user for every HTTP method on the wrapped view.

    Args:
        view: The upstream ``tom_calendar`` view function to protect.

    Returns:
        A view that calls ``view`` for a signed-in user and otherwise returns a login redirect.
    """

    @wraps(view)
    def wrapper(request: HttpRequest, *args, **kwargs) -> HttpResponse:
        if request.user.is_authenticated:
            return view(request, *args, **kwargs)
        return _login_redirect()

    wrapper.calendar_guard = 'write_requires_login'
    return wrapper


def read_open_write_requires_login(view: Callable[..., HttpResponse]) -> Callable[..., HttpResponse]:
    """Leave GET and HEAD open on the wrapped view, require a signed-in user for any other method.

    Used for update-event, whose GET renders the read-only event pop-up and whose POST saves.

    Args:
        view: The upstream ``tom_calendar`` view function to protect.

    Returns:
        A view that calls ``view`` for GET, HEAD or a signed-in user, and otherwise returns a login redirect.
    """

    @wraps(view)
    def wrapper(request: HttpRequest, *args, **kwargs) -> HttpResponse:
        if request.method in _READ_METHODS or request.user.is_authenticated:
            return view(request, *args, **kwargs)
        return _login_redirect()

    wrapper.calendar_guard = 'read_open_write_requires_login'
    return wrapper
