from django.urls import path

from solsys_code.views import fomo_render_calendar

app_name = 'calendar'

urlpatterns = [
    path('', fomo_render_calendar, name='calendar'),
]
