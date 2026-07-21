"""FOMO-local calendar URL conf — full replacement of tom_calendar.urls for /calendar/.

Shadows the entire tom_calendar URL namespace so that all calendar:* reversals resolve
through this module.  The root path ('') is served by fomo_render_calendar, which injects
prefetch_related + Count annotation (DISPLAY-09).  All sub-paths (create, update, delete,
todo) delegate to the upstream tom_calendar view functions unchanged.
"""

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
