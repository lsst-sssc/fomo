from django.db import models
from tom_calendar.models import CalendarEvent


class CalendarEventTelescopeLabel(models.Model):
    """Sidecar record of whether a CalendarEvent's telescope label was live-verified
    against the LCO API or fallback-guessed (TELESCOPE-03/04). One row per
    CalendarEvent at most; no row at all means "verified" by documented default
    (e.g. classically-scheduled events from load_telescope_runs, which never go
    through telescope-label resolution).
    """

    event = models.OneToOneField(
        CalendarEvent,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='telescope_label_meta',
        verbose_name='Calendar event',
    )
    is_verified = models.BooleanField(
        default=True, verbose_name='Whether the telescope label was live-verified against the LCO API'
    )

    def __str__(self):
        return f'{"Verified" if self.is_verified else "Fallback"} label for {self.event.title}'
