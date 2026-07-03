from django.db import models
from tom_calendar.models import CalendarEvent
from tom_targets.models import Target, TargetList

from solsys_code.solsys_code_observatory.models import Observatory


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


class CampaignRun(models.Model):
    """A single target-linked observing run within a coordination campaign (e.g. 3I/ATLAS).

    Replaces the ad-hoc Google Sheet the community previously used to coordinate follow-up
    observations of a rare/urgent object. Status is split into two independent fields
    (``approval_status``/``run_status``) rather than one flat vocabulary, so a DDT/proposal
    request whose real-world outcome is still pending can be represented independently of
    admin review state (D-02). The campaign container (``TargetList``) itself carries no
    status field in this milestone (D-01) -- status lives entirely on ``CampaignRun``.
    """

    class ApprovalStatus(models.TextChoices):
        """Admin review state for a CampaignRun (independent of real-world run outcome)."""

        PENDING_REVIEW = 'pending_review', 'Pending Review'
        APPROVED = 'approved', 'Approved'
        REJECTED = 'rejected', 'Rejected'

    class RunStatus(models.TextChoices):
        """Real-world lifecycle state of a CampaignRun, independent of admin review state."""

        REQUESTED = 'requested', 'Requested'
        PLANNED = 'planned', 'Planned'
        OBSERVED = 'observed', 'Observed'
        REDUCED = 'reduced', 'Reduced'
        PUBLISHED = 'published', 'Published'
        CANCELLED = 'cancelled', 'Cancelled'
        NOT_AWARDED = 'not_awarded', 'Not Awarded'
        WEATHER_TECH_FAILURE = 'weather_tech_failure', 'Weather/Technical Failure'

    campaign = models.ForeignKey(
        TargetList,
        on_delete=models.PROTECT,
        null=False,
        related_name='campaign_runs',
        verbose_name='Campaign target list',
    )
    target = models.ForeignKey(
        Target,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='campaign_runs',
        verbose_name='Observed target',
    )
    telescope_instrument = models.CharField(max_length=255, verbose_name='Telescope / instrument')
    site = models.ForeignKey(
        Observatory,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='campaign_runs',
        verbose_name='Resolved observing site',
    )
    site_raw = models.CharField(max_length=255, blank=True, default='', verbose_name='Original site code text')
    site_needs_review = models.BooleanField(
        default=False, verbose_name='Whether the site could not be automatically resolved and needs manual review'
    )
    obs_date = models.DateField(null=True, blank=True, verbose_name='Observation date')
    ut_start = models.DateTimeField(null=True, blank=True, verbose_name='UT start time')
    ut_end = models.DateTimeField(null=True, blank=True, verbose_name='UT end time')
    filters_bandpass = models.CharField(max_length=255, blank=True, default='', verbose_name='Filter(s) / bandpass')
    observation_details = models.TextField(blank=True, default='', verbose_name='Observation details')
    weather = models.TextField(blank=True, default='', verbose_name='Weather conditions or forecast')
    observation_outcome = models.TextField(blank=True, default='', verbose_name='Observation outcome')
    publication_plans = models.TextField(blank=True, default='', verbose_name='Publication plans')
    open_to_collaboration = models.BooleanField(default=False, verbose_name='Open to collaboration?')
    comments = models.TextField(blank=True, default='', verbose_name='Other comments')
    contact_person = models.CharField(max_length=255, blank=True, default='', verbose_name='Contact person')
    contact_email = models.EmailField(blank=True, default='', verbose_name='Contact email')
    approval_status = models.CharField(
        max_length=20,
        choices=ApprovalStatus,
        default=ApprovalStatus.PENDING_REVIEW,
        verbose_name='Approval status',
    )
    run_status = models.CharField(
        max_length=30,
        choices=RunStatus,
        default=RunStatus.REQUESTED,
        verbose_name='Run status',
    )

    class Meta:  # noqa: D106
        constraints = [
            # WR-05: backs the natural key insert_or_create_campaign_run's docstring and
            # import_campaign_csv's D-04 comment both describe as relied on for
            # idempotent re-imports. get_or_create() is only race-safe when its lookup
            # fields are backed by a real DB constraint; without one, two concurrent
            # imports could both miss the existing row and both attempt to create it.
            models.UniqueConstraint(
                fields=['campaign', 'telescope_instrument', 'ut_start'],
                name='unique_campaign_run_natural_key',
            ),
        ]

    def __str__(self):
        return f'{self.campaign.name}: {self.telescope_instrument} on {self.obs_date}'
