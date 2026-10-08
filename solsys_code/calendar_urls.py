"""FOMO-local calendar URL conf — full replacement of tom_calendar.urls for /calendar/.

Shadows the entire tom_calendar URL namespace so that all calendar:* reversals resolve
through this module.  The root path ('') is served by fomo_render_calendar, which injects
prefetch_related + Count annotation (DISPLAY-09).  The sub-paths (create, update, delete, todo)
delegate to the upstream tom_calendar view functions, and update-event is wrapped in
read_open_write_requires_login (solsys_code/calendar_access.py, ACCESS-01) so its GET stays the open
pop-up while any other method needs a login.
"""

from django.urls import path
from tom_calendar.views import create_event, create_todo, delete_event, update_event, update_todo

from solsys_code.calendar_access import read_open_write_requires_login
from solsys_code.views import fomo_render_calendar

app_name = 'calendar'

urlpatterns = [
    path('', fomo_render_calendar, name='calendar'),
    path('create/', create_event, name='create-event'),
    path('update/<int:event_id>/', read_open_write_requires_login(update_event), name='update-event'),
    path('delete/<int:event_id>/', delete_event, name='delete-event'),
    path('todo/create/<int:event_id>/', create_todo, name='create-todo'),
    path('todo/update/<int:todo_id>/', update_todo, name='update-todo'),
]
