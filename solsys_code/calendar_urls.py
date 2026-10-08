"""FOMO-local calendar URL conf — full replacement of tom_calendar.urls for /calendar/.

Shadows the entire tom_calendar URL namespace so that all calendar:* reversals resolve
through this module.  The root path ('') is served by fomo_render_calendar, which injects
prefetch_related + Count annotation (DISPLAY-09).

The five write routes (create, update, delete, todo create, todo update) delegate to tomtoolkit
3.1.0's tom_calendar view functions, which carry no login check and act on any HTTP method.  They
are wrapped in the guards in solsys_code/calendar_access.py (Phase 39, ACCESS-01): an anonymous
caller is redirected to login with next set to the calendar page.  update-event's GET and HEAD stay
open because that is the event pop-up (D-04).  delete-event, create-todo and update-todo also
refuse non-POST methods (require_POST), because upstream acts on a plain GET (it deletes an event or
blanks a todo).  The upstream view bodies run unchanged.
"""

from django.urls import path
from django.views.decorators.http import require_POST
from tom_calendar.views import create_event, create_todo, delete_event, update_event, update_todo

from solsys_code.calendar_access import read_open_write_requires_login, write_requires_login
from solsys_code.views import fomo_render_calendar

app_name = 'calendar'

urlpatterns = [
    path('', fomo_render_calendar, name='calendar'),
    path('create/', write_requires_login(create_event), name='create-event'),
    path('update/<int:event_id>/', read_open_write_requires_login(update_event), name='update-event'),
    path('delete/<int:event_id>/', write_requires_login(require_POST(delete_event)), name='delete-event'),
    path('todo/create/<int:event_id>/', write_requires_login(require_POST(create_todo)), name='create-todo'),
    path('todo/update/<int:todo_id>/', write_requires_login(require_POST(update_todo)), name='update-todo'),
]
