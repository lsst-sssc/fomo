from django.contrib import admin
from django.utils import timezone
from tom_targets.models import Target

from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation


class CalendarEventMetaInline(admin.TabularInline):
    """D-06: a row appearing here means the calendar event is owned by this run. Removing
    the `run` value on a row un-owns the event without deleting the companion row itself
    (CalendarEventMeta.run is SET_NULL, not CASCADE) -- the row, and its is_verified
    history, survive.
    """

    model = CalendarEventMeta
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
