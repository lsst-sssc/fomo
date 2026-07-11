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
    window_start = models.DateField(null=True, blank=True, verbose_name='Observing window start')
    window_end = models.DateField(null=True, blank=True, verbose_name='Observing window end')
    original_obs_date_raw = models.CharField(
        max_length=255, blank=True, default='', verbose_name='Original Obs. Date text (TBD rows only)'
    )
    window_needs_review = models.BooleanField(
        default=False,
        verbose_name='Whether the observing window could not be automatically resolved and needs manual review',
    )
    filters_bandpass = models.CharField(max_length=255, blank=True, default='', verbose_name='Filter(s) / bandpass')
    observation_details = models.TextField(blank=True, default='', verbose_name='Observation details')
    weather = models.TextField(blank=True, default='', verbose_name='Weather conditions or forecast')
    observation_outcome = models.TextField(blank=True, default='', verbose_name='Observation outcome')
    publication_plans = models.TextField(blank=True, default='', verbose_name='Publication plans')
    open_to_collaboration = models.BooleanField(default=False, verbose_name='Open to collaboration?')
    comments = models.TextField(blank=True, default='', verbose_name='Other comments')
    contact_person = models.CharField(max_length=255, blank=True, default='', verbose_name='Contact person')
    contact_email = models.EmailField(blank=True, default='', verbose_name='Contact email')
    contact_public_opt_in = models.BooleanField(default=False, verbose_name='Show contact info publicly?')
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
            # Resolved-window branch: a concrete single night (window_start == window_end)
            # or range. window_end is included (not just window_start) so a range starting
            # on the same day as an existing single-night entry is not treated as the same
            # row.
            models.UniqueConstraint(
                fields=('campaign', 'telescope_instrument', 'window_start', 'window_end'),
                condition=models.Q(window_start__isnull=False),
                name='unique_campaign_run_resolved_window',
            ),
            # TBD branch: window_start/window_end are deliberately NOT in this constraint's
            # field tuple -- they're both NULL for every row this constraint applies to (per
            # its own condition), and NULL is never considered equal by a unique constraint
            # on any backend, so including them here would silently defeat the whole point
            # of this constraint. contact_person is the natural-key discriminator instead
            # (never NULL: CharField(blank=True, default='')).
            models.UniqueConstraint(
                fields=('campaign', 'telescope_instrument', 'contact_person'),
                condition=models.Q(window_start__isnull=True),
                name='unique_campaign_run_tbd_natural_key',
            ),
            # WR-02: every reader of window_start/window_end (render_window_start,
            # CampaignRunDecisionView.post, claimed_dates) assumes the two fields are either
            # both NULL (TBD) or both set (resolved) -- neither partial UniqueConstraint above
            # enforces that pairing. Without this, a row with window_start set and
            # window_end NULL (or vice versa) would silently persist and crash
            # claimed_dates()'s date-arithmetic on read.
            models.CheckConstraint(
                condition=(
                    models.Q(window_start__isnull=True, window_end__isnull=True)
                    | models.Q(window_start__isnull=False, window_end__isnull=False)
                ),
                name='campaign_run_window_start_end_null_together',
            ),
        ]

    def __str__(self):
        return f'{self.campaign.name}: {self.telescope_instrument} on {self.window_start}'
