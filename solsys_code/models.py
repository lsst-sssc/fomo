from django.conf import settings
from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import Target, TargetList

from solsys_code.solsys_code_observatory.models import Observatory


class CalendarEventMeta(models.Model):
    """General companion record for a CalendarEvent (Phase 27 CANON-03): carries whether the
    event's telescope label was live-verified against the LCO API or fallback-guessed
    (TELESCOPE-03/04), plus which CampaignRun, if any, owns this event. One row per
    CalendarEvent at most; no row at all means "verified" by documented default (e.g.
    classically-scheduled events from load_telescope_runs, which never go through
    telescope-label resolution). 26-DECISION chose this general name over
    `CalendarEventRunLink` precisely so a third field added in a future version needs no
    second rename.

    A row whose ``run`` is unset means "not owned by any CampaignRun" -- never "touch me".
    This is the ownership rule the Phase 29 reconciler reads.
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
    run = models.ForeignKey(
        'CampaignRun',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='calendar_event_metas',
        verbose_name='Owning campaign run',
    )
    # D-12 (28-CONTEXT.md): Phase 28 deliberately reopens Phase 27's D-05, which left this FK
    # bare on purpose and accepted the resulting audit asymmetry with the observation link.
    # ROADMAP Phase 28 criterion 4 requires both a confirmation and an undo to be attributable
    # to a person and a time, and 27-CONTEXT.md itself named this phase as the place to
    # revisit the gap if the undo flow proved it painful -- it did, so these two fields close
    # that asymmetry, mirroring CampaignRunObservation.confirmed_by/confirmed_at exactly.
    #
    # Consequence readers need: an event link written before Phase 28 (e.g. by the admin FK
    # picker) has both fields NULL, so NULL means "confirmed before audit fields existed", not
    # "unconfirmed" -- the `run` FK being set is still what means "attributed".
    confirmed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='confirmed_calendar_event_metas',
        verbose_name='Confirmed by',
    )
    confirmed_at = models.DateTimeField(null=True, blank=True, verbose_name='Confirmed at')

    def __str__(self):
        """Verified/Fallback prefix + event title + event start (Task 1, 27.1-02).

        Measured against the live dev DB (2026-07-30): 7 of the 11 real companion rows
        share the identical event title ``[EXPIRED] 2m0 2M0-SCICAM-MUSCAT``; their start
        date-times (2026-07-07 through 2026-07-20) are all distinct, which is what makes
        the 11 rows distinguishable in the admin changelist and autocomplete. Do not
        "simplify" the date back out.
        """
        prefix = 'Verified' if self.is_verified else 'Fallback'
        start = self.event.start_time.strftime('%Y-%m-%d %H:%M')
        return f'{prefix} label for {self.event.title} ({start})'


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

    class Source(models.TextChoices):
        """Which FOMO ingest path created this row (CANON-01, 26-DECISION.md Criterion 1).

        CLASSICAL_FILE/LCO_QUEUE/GEMINI_QUEUE are declared now but not produced by any code
        path until v2.3's ADAPT-01..03 rewires the three calendar-sync adapters to write
        CampaignRuns -- declaring them early costs nothing because TextChoices values are
        validation-only, and it lets the vocabulary be settled once rather than twice.

        Derivation rule (verbatim, 26-DECISION.md Criterion 1): ``approval_status ==
        APPROVED`` together with ``source != WEB`` means no approval was required -- a
        different fact from a human having approved the run. A fourth NOT_REQUIRED
        approval_status value was considered and rejected because every existing reader of
        approval_status would have to handle it for a distinction source already carries.

        ESO_QUEUE added in plan 29-06 (explicit user-directed deviation, not part of this
        phase's original scope): the real 3I/ATLAS dev-DB data contains ESO VLT rows
        (obscode 309) which 26-DECISION.md's own "Run-type inventory" rule classifies as a
        shared-queue-scheduled network alongside LCO/Gemini/SOAR, but the vocabulary had no
        dedicated slot for it -- mapping those rows onto LCO_QUEUE would have been
        semantically wrong (they are not LCO-network runs), so the user chose to add a real
        value instead of overloading an existing one or leaving the rows under-classified.
        """

        WEB = 'web', 'Web submission'
        CLASSICAL_FILE = 'classical_file', 'Classical run file'
        LCO_QUEUE = 'lco_queue', 'LCO queue'
        GEMINI_QUEUE = 'gemini_queue', 'Gemini queue'
        ESO_QUEUE = 'eso_queue', 'ESO queue'
        CSV_IMPORT = 'csv_import', 'CSV import'
        LEGACY = 'legacy', 'Legacy (pre-v2.2)'

    class TelescopeClass(models.TextChoices):
        """Distinguishes a class-wide telescope allocation from a run whose site failed to
        resolve (CANON-02, D-11/D-12/D-20/D-21) -- today both read as ``site=None``.

        Three telescope-class allocations plus SPACE, where SPACE means specifically *a
        space observatory that has a JPL Horizons code but no MPC obscode assigned at all*
        -- JUICE (site_raw='500@-28') is the case. Swift (C52), HST (250) and JWST (274) are
        NOT SPACE: they resolve to a real Observatory like any ground site (D-11 falsified
        26-DECISION.md Criterion 3's "space missions are permanently site-less" premise).

        "Unresolved" is deliberately NOT a value here -- site_needs_review already carries
        exactly that meaning and is already wired into the approval queue's site-resolution
        work list (D-11). 4m0 is deliberately excluded to match CANON-02's wording, even
        though calendar_utils' aperture set has it for SOAR (D-12); the subset-assertion
        test in test_calendar_utils.py names 4m0 as the known exclusion so nobody "fixes"
        the discrepancy by adding it here.
        """

        # D-21: stored lowercase, matching calendar_utils.aperture_class_from_telescope_code's
        # existing vocabulary -- that module is where the class vocabulary already lives
        # (D-20), so its casing wins over CONTEXT.md's uppercase prose styling.
        TWO_M0 = '2m0', '2m0 class allocation'
        ONE_M0 = '1m0', '1m0 class allocation'
        ZERO_M4 = '0m4', '0m4 class allocation'
        # D-21: SPACE keeps CONTEXT.md's literal uppercase casing -- it has no calendar_utils
        # counterpart to match the way 2m0/1m0/0m4 do. Do not lowercase this for consistency.
        SPACE = 'SPACE', 'Space observatory with no MPC code'

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
    # D-06 (26-CONTEXT.md:94): corrected meaning -- True means the site did not resolve AND
    # no telescope_class explains why. A class-carrying run (telescope_class non-blank) is
    # never flagged, because the class already answers "why is there no site"; only a
    # genuinely unresolvable, class-less row belongs in the staff "Sites Needing Review"
    # queue. verbose_name is left unchanged deliberately -- editing it would force an extra
    # AlterField migration for no behavioural gain.
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
    # The LEGACY default is what backfills every pre-milestone row -- no RunPython step at
    # all, exactly as 26-DECISION.md Criterion 4 locked. Every NEW write path must set source
    # explicitly rather than relying on this default: Plan 05 sets WEB on the submission
    # path, Plan 06 sets CSV_IMPORT on the importer path.
    source = models.CharField(
        max_length=20,
        choices=Source,
        default=Source.LEGACY,
        verbose_name='Ingest source',
    )
    # D-06 (26-CONTEXT.md:94): this field records WHY there is no site -- it is a PERMANENT,
    # correct campaign-level fact, not a placeholder cleared once a site becomes known. A
    # class-wide campaign (e.g. "LOOK Project Comet Followup 2026B", following up many
    # targets across the LCO 1m0 network) legitimately keeps site=None forever; its per-site
    # detail lives on the linked ObservationRecords via CampaignRunObservation (CANON-04),
    # never on the run itself -- the run-level `site` field is for single-site runs only.
    # The value is NEVER cleared by any writer once a site later resolves elsewhere for the
    # same run (Phase 27 code-review finding CR-01 proposed clearing it; the user REJECTED
    # CR-01 -- see 27-REVIEW-FIX.md -- because telescope_class and a resolved site are not
    # mutually exclusive). Inference still only happens at write time for site-less rows
    # (D-13): blank is the correct value for a genuine site-resolution failure, since
    # site_needs_review already carries "unresolved" -- telescope_class is never inferred
    # for a run whose site DID resolve.
    telescope_class = models.CharField(
        max_length=10,
        choices=TelescopeClass,
        blank=True,
        default='',
        verbose_name='Telescope class allocation',
    )

    @property
    def is_publicly_visible(self) -> bool:
        """D-09/D-10: whether this run should be visible to a non-staff reader.

        Exists so the 'pending_review' literal never appears in a template where nothing
        would catch it drifting from ApprovalStatus's TextChoices -- Plan 05's calendar-modal
        template override is its consumer. Negative constraint: do NOT refactor
        CampaignRunTableView.get_queryset() to use this -- D-10 keeps the queryset-level
        exclude() because a Python property cannot be used in a .filter(), so this is
        deliberately one definition in meaning and two in code; the queryset form is the one
        that keeps pending rows out of the SQL SELECT entirely.
        """
        return self.approval_status != self.ApprovalStatus.PENDING_REVIEW

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
        """Discriminating admin-picker label (Task 1, 27.1-02 / criterion 4).

        Load-bearing: this label is rendered on the Phase 27 WR-03 admin FK picker -- the
        only mechanism that can create a run<->event link until Phase 28's attribution
        queue ships -- as well as the `CampaignRunAdmin` changelist, change-form title,
        delete-confirmation page, admin history `object_repr`, and the
        `CalendarEventMetaAdmin.run` autocomplete JSON. It is built from four parts so two
        otherwise-identical rows (e.g. the real dev-DB pks 27/28, both
        ``3I/ATLAS: JWST on None`` under the old label) are always distinguishable:
        ``#pk``, campaign name, telescope/instrument, and a window/site discriminator.

        Deliberately excludes the two submitter-contact fields: this label is rendered
        into surfaces broader than the change form (the changelist and the autocomplete
        JSON endpoint), and `admin.py`'s T-jpd-02 PII gate must not be undone by widening
        what `__str__` exposes.
        """
        if self.window_start is None:
            window_label = 'TBD'
        elif self.window_start == self.window_end:
            window_label = str(self.window_start)
        else:
            window_label = f'{self.window_start}..{self.window_end}'

        if self.site_id is not None:
            site_label = self.site.obscode
        elif self.telescope_class:
            site_label = f'class {self.telescope_class}'
        elif self.site_raw:
            site_label = f'raw {self.site_raw}'
        else:
            site_label = 'no site'

        return f'#{self.pk} {self.campaign.name} | {self.telescope_instrument} | {window_label} | {site_label}'


@receiver(pre_delete, sender=CampaignRun)
def _delete_owned_calendar_events_on_campaign_run_delete(sender, instance, **kwargs):  # noqa: D103
    """WR-01 fix (29-REVIEW.md): cascade-delete a deleted run's calendar events.

    ``CalendarEventMeta.run`` uses ``on_delete=SET_NULL``, so without this, deleting a
    ``CampaignRun`` (from the Django admin -- either the single-object delete confirmation
    page or the changelist's bulk "Delete selected" action -- or directly via the ORM) left
    its ``RUN:``-namespaced ``CalendarEvent`` rows on the shared calendar forever,
    referencing a run that no longer exists, with no reconciler entry point that could ever
    touch them again (``reconcile_run()`` requires a live ``CampaignRun`` instance).

    A signal (not an override of ``CampaignRun.delete()``) so this also fires for the admin
    changelist's bulk delete action, which calls ``QuerySet.delete()`` -- that bypasses any
    instance-level ``delete()`` override, but still sends ``pre_delete``/``post_delete`` per
    object. ``pre_delete`` (not ``post_delete``) so ``campaign_reconciler.owned_events()``'s
    lookup (keyed on ``instance.pk``) still resolves normally, before the run row itself is
    gone. Deleting the ``CalendarEvent`` rows (not just detaching them) cascades to their
    ``CalendarEventMeta`` companion rows via that FK's ``on_delete=CASCADE``.

    Imported lazily (function-local, not module-level) to avoid a circular import:
    ``campaign_reconciler`` imports ``CalendarEventMeta``/``CampaignRun`` from this module.
    """
    from solsys_code.campaign_reconciler import owned_events

    owned_events(instance).delete()


class CampaignRunObservation(models.Model):
    """Links a CampaignRun to an ObservationRecord that realises it (CANON-04).

    D-01: a row exists only once a staff member confirms the attribution. Phase 28 computes
    attribution candidates on the fly and writes nothing until confirmation -- this keeps
    ATTRIB-03 ("no association without explicit staff confirmation") structural rather than a
    rule code must remember, and it is what Phase 28 uses to compute attribution candidates
    without ever writing one itself.

    No boolean confirmation flag (D-03): under D-01 the row's existence already means "a
    staff member confirmed this", so a flag would be redundant state that could contradict
    the row. Consequence for Phase 28: it computes candidates on the fly and writes nothing
    until confirmation, which is what keeps ATTRIB-03 structural rather than a rule code must
    remember.
    """

    run = models.ForeignKey(
        CampaignRun,
        on_delete=models.CASCADE,
        related_name='observation_links',
        verbose_name='Campaign run',
    )
    # CASCADE (D-04), not SET_NULL: the ObservationRecord is on the other side of the
    # relation and is untouched by this, and a run-less observation link would carry nothing
    # and mean nothing -- unlike CalendarEventMeta.run, which also carries is_verified and
    # must survive its owning run's deletion.
    observation_record = models.ForeignKey(
        ObservationRecord,
        on_delete=models.CASCADE,
        related_name='campaign_run_links',
        verbose_name='Observation record',
    )
    # D-03.
    confirmed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='confirmed_campaign_run_observations',
        verbose_name='Confirmed by',
    )
    confirmed_at = models.DateTimeField(null=True, blank=True, verbose_name='Confirmed at')

    class Meta:  # noqa: D106
        constraints = [
            # D-02: one run per observation record, expressed so it can be broadened cheaply.
            # A real DB constraint (not app-level validation) because two concurrent admin
            # saves could both miss an existing row. A named UniqueConstraint rather than a
            # OneToOneField: behaviour today is identical, but broadening to many-runs-per-
            # record later is a single RemoveConstraint with no field change and no reader
            # rewrites, since the reverse accessor is already a manager -- a OneToOneField
            # would instead need an AlterField that changes the accessor from an object to a
            # manager and breaks every reader at once. No condition= needed -- unlike
            # CampaignRun's two-branch design there is no branching case here.
            models.UniqueConstraint(
                fields=('observation_record',),
                name='unique_campaign_run_observation_record',
            ),
        ]

    def __str__(self):
        return f'{self.run}: {self.observation_record}'


class CalendarEventDismissal(models.Model):
    """Records a staff member's rejection of a suggested (CalendarEvent, CampaignRun) pair
    (28-CONTEXT.md D-05/D-06/D-08).

    A dismissal is NOT an association -- persisting one does not weaken Phase 27 D-01's
    invariant that an unconfirmed guess can never be mistaken for ownership. Without a
    dismissal record the attribution queue could never drain (ATTRIB-06 would be unreachable
    for any orphan carrying a wrong candidate), since a rejected candidate would return on
    every page load.

    The unique constraint below is per (event, run) PAIR, not per event, because D-05
    requires dismissing one wrong candidate to leave the event's other candidates -- and any
    candidate the matcher surfaces later as new runs arrive -- still offered.
    """

    event = models.ForeignKey(
        CalendarEvent,
        on_delete=models.CASCADE,
        related_name='attribution_dismissals',
        verbose_name='Calendar event',
    )
    run = models.ForeignKey(
        CampaignRun,
        on_delete=models.CASCADE,
        related_name='calendar_event_dismissals',
        verbose_name='Campaign run',
    )
    dismissed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='dismissed_calendar_event_attributions',
        verbose_name='Dismissed by',
    )
    dismissed_at = models.DateTimeField(null=True, blank=True, verbose_name='Dismissed at')
    reason = models.TextField(blank=True, default='', verbose_name='Why this candidate was rejected')

    class Meta:  # noqa: D106
        constraints = [
            # D-05/D-08: named per-pair UniqueConstraint, not a bare unique_together (this
            # codebase never uses that shape). Per (event, run) pair, not per event -- see
            # class docstring.
            models.UniqueConstraint(fields=('event', 'run'), name='unique_calendar_event_dismissal_pair'),
        ]

    # Both FKs are CASCADE, not SET_NULL -- a genuinely new judgement call flagged by
    # 28-PATTERNS.md (this is NOT the same choice CampaignRunObservation.observation_record
    # made for the same reason: a dismissal row records "this pair was rejected"; if either
    # side of the pair is deleted the pair can never be suggested again, so the row carries
    # nothing and means nothing. This is deliberately NOT the "survives as an audit trail
    # forever" alternative -- D-06/D-07's audit-trail requirement is about surviving the
    # *dismissing user's* deletion (dismissed_by is SET_NULL, below), not the orphan's or
    # run's.
    def __str__(self):
        return f'dismissed {self.event} for {self.run}'


class ObservationRecordDismissal(models.Model):
    """Records a staff member's rejection of a suggested (ObservationRecord, CampaignRun)
    pair (28-CONTEXT.md D-05/D-06/D-08). See CalendarEventDismissal's docstring for the full
    rationale -- this model is identical apart from the orphan-side FK.
    """

    observation_record = models.ForeignKey(
        ObservationRecord,
        on_delete=models.CASCADE,
        related_name='attribution_dismissals',
        verbose_name='Observation record',
    )
    run = models.ForeignKey(
        CampaignRun,
        on_delete=models.CASCADE,
        related_name='observation_record_dismissals',
        verbose_name='Campaign run',
    )
    dismissed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='dismissed_observation_record_attributions',
        verbose_name='Dismissed by',
    )
    dismissed_at = models.DateTimeField(null=True, blank=True, verbose_name='Dismissed at')
    reason = models.TextField(blank=True, default='', verbose_name='Why this candidate was rejected')

    class Meta:  # noqa: D106
        constraints = [
            models.UniqueConstraint(
                fields=('observation_record', 'run'), name='unique_observation_record_dismissal_pair'
            ),
        ]

    # See CalendarEventDismissal's on_delete comment -- same reasoning applies here: both FKs
    # are CASCADE because an orphaned dismissal row (either side gone) carries nothing and
    # means nothing.
    def __str__(self):
        return f'dismissed {self.observation_record} for {self.run}'
