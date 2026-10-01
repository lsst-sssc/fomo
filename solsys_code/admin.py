from django.contrib import admin
from tom_targets.admin import TargetAdmin
from tom_targets.models import Target, TargetName


class TargetNameInline(admin.TabularInline):
    """Edit a Target's aliases alongside the Target itself."""

    model = TargetName
    extra = 0


class SolsysTargetAdmin(TargetAdmin):
    """TOM Toolkit's TargetAdmin with the search, filters and columns it omits.

    The default columns suit the non-sidereal catalogue, where H and the orbit
    scheme carry what coordinates carry for a sidereal target; see
    get_list_display for the sidereal case.
    """

    search_fields = ('name', 'aliases__name')
    list_display = ('name', 'type', 'scheme', 'abs_mag', 'created')
    list_filter = ('type', 'scheme', 'created')
    ordering = ('-created',)
    inlines = TargetAdmin.inlines + [TargetNameInline]

    def get_list_display(self, request):
        """Swap in coordinates while the changelist is filtered to sidereal targets.

        The two target types have disjoint useful fields -- ra/dec are null on every
        non-sidereal row, scheme/abs_mag on every sidereal one -- so any single fixed
        column set is half empty. Columns cannot vary per row (list_display is one
        header row for the table), but they can follow the `type` filter, which is
        already in list_filter.
        """
        if request.GET.get('type__exact') == Target.SIDEREAL:
            return ('name', 'type', 'ra', 'dec', 'created')
        return self.list_display


admin.site.unregister(Target)
admin.site.register(Target, SolsysTargetAdmin)
