from django.contrib import admin
from tom_targets.models import Target

from solsys_code.models import CalendarEventTelescopeLabel, CampaignRun


class CampaignRunAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = [
        'pk',
        'campaign',
        'telescope_instrument',
        'approval_status',
        'run_status',
        'site',
        'window_start',
        'window_end',
    ]
    list_filter = ['approval_status', 'run_status', 'campaign']
    search_fields = ['telescope_instrument', 'site_raw', 'contact_person']
    # approval_status must stay read-only here: its normal transition triggers the
    # calendar-projection side effect and the D-06 `if run.site is None` clobber guard that
    # live entirely in CampaignRunDecisionView.post(), not on the model. Admin must never be
    # able to silently flip a run to APPROVED without going through that real approval-queue
    # flow.
    readonly_fields = ['approval_status']


class CalendarEventTelescopeLabelAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['event', 'is_verified']
    list_filter = ['is_verified']
    search_fields = ['event__title']


class TargetAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['name', 'type', 'ra', 'dec']
    list_filter = ['type']
    search_fields = ['name']


admin.site.register(CampaignRun, CampaignRunAdmin)
admin.site.register(CalendarEventTelescopeLabel, CalendarEventTelescopeLabelAdmin)
admin.site.unregister(Target)
admin.site.register(Target, TargetAdmin)
