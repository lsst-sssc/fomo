from django.contrib import admin
from django.forms.models import BaseInlineFormSet
from django.utils import timezone
from tom_targets.models import Target

from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation


class CalendarEventMetaInlineFormSet(BaseInlineFormSet):
    """WR-08: freeze `event` on rows that already exist, so it can never be re-pointed.

    `CalendarEventMeta.event` is an explicitly declared ``OneToOneField(primary_key=True)``.
    Django's ``BaseModelFormSet.add_fields()`` only injects a hidden ``id`` field when
    ``pk_is_not_editable(pk)``; such a field IS editable and IS present in ``form.fields``,
    so no hidden pk is added and `event` renders as a live ``<select>`` on every existing
    inline row. Changing it would make ``instance.pk`` a value absent from the table, so
    ``instance.save()`` would issue an UPDATE matching 0 rows and fall back to an INSERT --
    leaving the original row behind as a duplicate with an orphaned is_verified/run history,
    which is exactly what migration 0008's header comment was written to protect.

    ``disabled=True`` (not ``readonly_fields``) is used deliberately: readonly_fields on the
    inline would be keyed on the PARENT ``CampaignRun`` object and would therefore also strip
    the widget from the blank "Add another" row, making it impossible to link a new event from
    an existing run's change page. Per-form ``disabled`` leaves add and delete working.

    Disabling the widget is necessary but not sufficient, because
    ``BaseModelFormSet._construct_form()`` resolves each initial form's instance from the
    *submitted* pk (``_existing_object(pk)``) before any field ever sees it -- a submitted pk
    outside the queryset yields ``instance=None`` and therefore a brand-new object. So the
    submitted pk is first normalised back to the pk of the row this form slot actually
    belongs to; after that the widget's disabled state is what stops a staff user reaching
    the tampering path by accident in the first place.
    """

    def __init__(self, *args, **kwargs):
        """Normalise every initial form's submitted `event` pk back to its own row's pk."""
        super().__init__(*args, **kwargs)
        if self.is_bound:
            self._freeze_submitted_event_pks()

    def _freeze_submitted_event_pks(self):
        """Rewrite the POSTed pk of each existing-row slot to that row's real pk.

        Mirrors the data-rewriting precedent in Django's own
        ``BaseInlineFormSet._construct_form()`` ``save_as_new`` branch. Runs before any form
        is constructed, so ``_construct_form()`` resolves the intended instance rather than
        falling through to a fresh, unsaved one.
        """
        pk_name = self.model._meta.pk.name
        queryset = self.get_queryset()
        for index in range(self.initial_form_count()):
            try:
                frozen_pk = str(queryset[index].pk)
            except IndexError:
                # More initial forms declared in the management form than rows exist --
                # tampered or stale POST. Leave it alone; Django's own validation reports it.
                break
            pk_key = f'{self.add_prefix(index)}-{pk_name}'
            if self.data.get(pk_key) != frozen_pk:
                # QueryDict is immutable; .copy() yields a mutable one (and works for the
                # plain-dict form data used in tests too).
                self.data = self.data.copy()
                self.data[pk_key] = frozen_pk
        # No return statement -- this is an in-place normalisation

    def add_fields(self, form, index):
        """Disable the identity field on any form bound to an already-saved row."""
        super().add_fields(form, index)
        if form.instance.pk is not None and 'event' in form.fields:
            form.fields['event'].disabled = True
        # No return statement -- BaseInlineFormSet.add_fields() returns None


class CalendarEventMetaInline(admin.TabularInline):
    """D-06: a row appearing here means the calendar event is owned by this run. Removing
    the `run` value on a row un-owns the event without deleting the companion row itself
    (CalendarEventMeta.run is SET_NULL, not CASCADE) -- the row, and its is_verified
    history, survive.

    WR-08: `event` is this model's primary key, so it is frozen on existing rows via
    CalendarEventMetaInlineFormSet -- add and delete are the only operations on the link
    itself; re-pointing it would duplicate the row rather than move it.
    """

    model = CalendarEventMeta
    formset = CalendarEventMetaInlineFormSet
    fk_name = 'run'
    extra = 0


class CampaignRunObservationInline(admin.TabularInline):
    """D-06/CANON-05: confirmed observation-record attributions for this run.

    confirmed_by/confirmed_at are read-only here -- CampaignRunAdmin.save_formset is the
    only place that sets them (D-07), so they can never be hand-typed, which is what keeps
    D-03's audit trail trustworthy.
    """

    model = CampaignRunObservation
    fk_name = 'run'
    extra = 0
    readonly_fields = ['confirmed_by', 'confirmed_at']


class CampaignRunAdmin(admin.ModelAdmin):  # noqa: D101
    """D-06: `inlines` below satisfies CANON-05 (staff see/edit a run's linked events and
    observation records) without a new view/URL/template -- no run-detail view exists today
    (campaign_urls.py: list/table/submit/approval-queue/decide/gaps/site-search). The real
    staff-facing run-detail page is deferred to Phase 28.
    """

    list_display = [
        'pk',
        'campaign',
        'telescope_instrument',
        'approval_status',
        'run_status',
        'site',
        'window_start',
        'window_end',
        'source',
        'telescope_class',
    ]
    # D-19: filtering by source is how staff audit "which runs came from the CSV import";
    # by telescope_class, how they find class-wide runs.
    list_filter = ['approval_status', 'run_status', 'campaign', 'source', 'telescope_class']
    search_fields = ['telescope_instrument', 'site_raw', 'contact_person']
    inlines = [CalendarEventMetaInline, CampaignRunObservationInline]
    # approval_status must stay read-only here: its transition triggers the calendar-
    # projection side effect and the D-06 clobber guard that live in
    # CampaignRunDecisionView.post(), not on the model. source stays editable by default
    # here (D-19) -- it has no such side-effecting transition, so the class-level omission
    # is a decision, not an oversight. get_readonly_fields below narrows this further:
    # source is withheld on every web-sourced row, at any approval status (27.1-05, closing
    # criterion 6) -- see that method's docstring for the cost and the residual left open.
    # Django excludes readonly fields from the generated ModelForm entirely, so a POSTed
    # source value on such a row cannot bind -- it is not merely ignored on render.
    readonly_fields = ['approval_status']

    def get_readonly_fields(self, request, obj=None):
        """Withhold `source` on every web-sourced run, at any approval status (27.1-05).

        **The rule.** `source` is withheld whenever the run's source is web, regardless of
        approval status. The add-form gate is unchanged -- a run can still be created with
        any source. This closes `27.1-REVIEW.md` WR-03 and Phase 27.1 success criterion 6.

        **What it closes.** The narrower lock this replaced only withheld `source` once a
        WEB run was already APPROVED, so a legal two-step sequence reached the identical
        outcome without ever touching a protected row: edit `source` while the run is still
        PENDING_REVIEW (editable by design, D-19), then approve it through
        `CampaignRunDecisionView`, which only requires `approval_status ==
        PENDING_REVIEW` and does not itself inspect `source`. The row would land on APPROVED
        + non-web, which under `CampaignRun.Source`'s own derivation rule reads as "no
        approval was required" -- the same provenance loss the narrower lock existed to
        prevent, reached by a longer route.

        **Why widening costs nothing real.** `source = web` is written by exactly one code
        path in the repository, the public submission view (`campaign_views.py`), so a web
        label can never be a mislabel -- D-19's correction use case, which is about
        correcting a mislabelled *ingest* source, never applied to a web row in the first
        place. Every non-web row of every approval status keeps `source` editable, so D-19
        survives for `legacy` (a pre-v2.2 bulk backfill guess), `csv_import`, and the queue
        sources.

        **Parity.** `import_campaign_csv.py` already refuses to rewrite `source` on a row
        whose existing `source` is web, with no approval-status condition at all. This
        change makes the admin write path enforce that same single rule instead of a
        weaker variant of it.

        **The accepted cost.** A genuinely mis-recorded `source` on a web row can no longer
        be corrected through the admin change form at all -- it needs a shell or a data migration.
        That is the intended friction, and it matches the friction the CSV path already
        imposes.

        **The residual, left open deliberately.** The check keys on the row's *current*
        `source`, so it is a one-way ratchet: a non-web row can still be relabelled *to*
        web, after which it locks. That direction fabricates provenance rather than
        destroying it, requires a deliberate staff act, and is recorded by Django's admin
        `LogEntry` with actor and timestamp -- the destructive direction left nothing
        behind. Locking it too would re-open D-19 for the sources D-19 legitimately covers,
        so it is accepted rather than mitigated.

        **WR-10: what "accepted" actually costs, stated plainly.** Two consequences follow
        from that ratchet and are neither hypothetical nor recoverable in the UI:

        1. It is a trapdoor with no admin-side exit. The instant the relabel saves,
           `obj.source == WEB` makes `source` readonly, so the staff user cannot undo their
           own mis-click here at all -- and the most likely way to land on a wrongly-web row
           is exactly that: an accidental selection in a `legacy`/`csv_import` row's `source`
           dropdown, not a genuinely-web row that was mis-recorded.
        2. The CSV path cannot repair it either. `import_campaign_csv` pops both `source`
           and `approval_status` for any row whose *existing* source is web, so re-importing
           the sheet that originally produced the row will not restore `csv_import`. The
           parity noted above therefore also removes the second repair route.

        A mis-clicked APPROVED row consequently reads permanently as "a human approved this
        public submission" -- a fact that never happened -- and `LogEntry.change_message`
        records only that `source` changed, not what it changed *from*. Both directions of
        the ratchet are pinned by
        `test_relabel_to_web_locks_the_row_and_cannot_be_undone` so a refactor cannot move
        them silently. If this edge ever proves too sharp in practice, the alternative is to
        withhold `source` whenever `approval_status == APPROVED` regardless of source, which
        costs D-19 only the ability to relabel already-approved backfill.
        """
        readonly = list(super().get_readonly_fields(request, obj))
        if obj is not None and obj.source == CampaignRun.Source.WEB:
            readonly.append('source')
        return readonly

    def get_queryset(self, request):
        """select_related the FKs `CampaignRun.__str__` now dereferences (Task 1).

        Not cosmetic: Django's `AutocompleteJsonView` calls the target ModelAdmin's
        `get_queryset()` and then `str()` on each result, so without this every
        autocomplete keystroke costs two extra queries per result row. `list_select_related`
        would not cover this -- the autocomplete endpoint does not read it.
        """
        return super().get_queryset(request).select_related('campaign', 'site')

    def save_formset(self, request, form, formset, change):
        """D-07: stamp confirmed_by/confirmed_at on newly created CampaignRunObservation
        rows only.

        Under D-01, a CampaignRunObservation row's existence *is* the claim that a human
        confirmed the attribution -- a row created here without confirmed_by would look
        confirmed while carrying no attribution, exactly the hole D-07 exists to close, and
        the one Phase 28's ATTRIB-03 depends on being closed. Follows Django's own
        save_formset idiom (formset.save(commit=False) + manual instance.save() +
        formset.deleted_objects cleanup + formset.save_m2m()) rather than the base
        implementation's bare formset.save(), since request.user is only available here.
        """
        instances = formset.save(commit=False)
        for instance in instances:
            # isinstance gate: save_formset is called once per inline formset -- both
            # CalendarEventMetaInline's and CampaignRunObservationInline's formsets flow
            # through this same method. CalendarEventMeta has no confirmed_by/confirmed_at
            # fields at all (D-05 declines audit fields on the event side), so it must never
            # be stamped.
            if isinstance(instance, CampaignRunObservation) and instance.pk is None:
                # The pk-is-unset check above is the gate: a NEWLY CREATED row gets
                # stamped; an EXISTING row edited through the admin keeps its original
                # confirmed_by/confirmed_at, so editing a link never silently
                # re-attributes someone else's confirmation.
                instance.confirmed_by = request.user
                instance.confirmed_at = timezone.now()
            instance.save()
        # Stock idiom: formset.save(commit=False) does not itself delete rows marked for
        # deletion -- it only populates formset.deleted_objects. Deleting them here
        # preserves the base ModelAdmin.save_formset() behaviour this override replaces.
        for obj in formset.deleted_objects:
            obj.delete()
        formset.save_m2m()


class CalendarEventMetaAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['event', 'event_start', 'is_verified', 'run']
    list_filter = ['is_verified']
    search_fields = ['event__title']
    # 27.1-02 criterion 4: a search box instead of a 44-option flat <select> for the run
    # picker -- safe because CampaignRunAdmin.search_fields already exists, which is what
    # Django's admin.E040 check requires.
    autocomplete_fields = ['run']
    # Covers the new event_start column and CampaignRun.__str__'s campaign/site
    # dereferences (Task 1), so an 11-row changelist does not turn into 30+ queries.
    list_select_related = ['event', 'run', 'run__campaign', 'run__site']

    def get_readonly_fields(self, request, obj=None):
        """CR-02: extend the WR-08 primary-key freeze to the standalone change form.

        ``CalendarEventMetaInlineFormSet`` froze `event` only on the inline. This ModelAdmin
        is a second, independent write path onto the same row, and 27.1-02 made it the
        primary staff surface for hand-linking a run to an event. Because
        ``CalendarEventMeta.event`` is an explicitly declared
        ``OneToOneField(primary_key=True)``, Django treats it as editable and renders it as a
        live ``<select>`` here too -- so re-pointing it made ``instance.pk`` a value absent
        from the table, turning ``save()``'s 0-row UPDATE into an INSERT and leaving the
        original row behind as a duplicate with orphaned is_verified/run history (the exact
        failure migration 0008's header comment exists to prevent).

        Returning it as readonly on change (``obj is not None``) drops the field from the
        form entirely, so a submitted `event` value is never bound; the instance keeps
        resolving from the URL pk. Add stays editable so a new link can still be created,
        which keeps re-pointing available as delete + re-add -- matching the inline's
        add-and-delete-only contract rather than being stricter than it.
        """
        readonly = list(super().get_readonly_fields(request, obj))
        if obj is not None and 'event' not in readonly:
            readonly.append('event')
        return readonly

    @admin.display(description='Event start', ordering='event__start_time')
    def event_start(self, obj):
        """Return the owning CalendarEvent's start time for the changelist column."""
        return obj.event.start_time


class TargetAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['name', 'type', 'ra', 'dec']
    list_filter = ['type']
    search_fields = ['name']


admin.site.register(CampaignRun, CampaignRunAdmin)
admin.site.register(CalendarEventMeta, CalendarEventMetaAdmin)
admin.site.unregister(Target)
admin.site.register(Target, TargetAdmin)
