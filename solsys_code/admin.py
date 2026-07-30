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
    # CampaignRunDecisionView.post(), not on the model. source is deliberately NOT added
    # here (D-19) -- it has no such side-effecting transition, so the omission is a decision.
    readonly_fields = ['approval_status']

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
    list_display = ['event', 'is_verified']
    list_filter = ['is_verified']
    search_fields = ['event__title']


class TargetAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['name', 'type', 'ra', 'dec']
    list_filter = ['type']
    search_fields = ['name']


admin.site.register(CampaignRun, CampaignRunAdmin)
admin.site.register(CalendarEventMeta, CalendarEventMetaAdmin)
admin.site.unregister(Target)
admin.site.register(Target, TargetAdmin)
