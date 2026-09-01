"""django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import include, path

from solsys_code.views import Ephemeris, MakeEphemerisView

urlpatterns = [
    path('observatory/', include('solsys_code.solsys_code_observatory.urls', namespace='solsys_code_observatory')),
    path('ephem/<int:pk>/', Ephemeris.as_view(), name='ephem'),
    path('targets/<int:pk>/makeephem/', MakeEphemerisView.as_view(), name='makeephem'),
    # DISPLAY-09 — must precede tom_common.urls: tomtoolkit 3.0 now also registers a 'calendar'
    # namespace there (tom_calendar.urls) with identical URL names, which triggers Django's
    # urls.W005 "namespace isn't unique" check warning. That's expected and benign — this entry,
    # being first, is what `calendar:*` reversal and dispatch actually resolve to; the tom_common
    # one is fully shadowed. See solsys_code/calendar_urls.py's module docstring.
    path('calendar/', include('solsys_code.calendar_urls', namespace='calendar')),
    path('campaigns/', include('solsys_code.campaign_urls', namespace='campaigns')),  # VIEW-01 — before tom_common
    # tomtoolkit 3.0.0 final dropped the 'alerts/' include from tom_common.urls (previously
    # registered there in 2.x/3.0.0a9) -- tom_alerts is still an installed app, so its urls
    # must now be wired up at the project level to keep the 'alerts' namespace resolvable.
    path('alerts/', include('tom_alerts.urls', namespace='alerts')),
    path('', include('tom_common.urls')),
]
