from django.contrib import admin
from tom_targets.admin import TargetAdmin
from tom_targets.models import Target, TargetName


class TargetNameInline(admin.TabularInline):
    """Edit a Target's aliases alongside the Target itself."""

    model = TargetName
    extra = 0


class SolsysTargetAdmin(TargetAdmin):
    """TOM Toolkit's TargetAdmin with the search, filters and columns it omits."""

    search_fields = ('name', 'aliases__name')
    list_display = ('name', 'type', 'created', 'modified')
    list_filter = ('type', 'scheme', 'created')
    ordering = ('-created',)
    inlines = TargetAdmin.inlines + [TargetNameInline]


admin.site.unregister(Target)
admin.site.register(Target, SolsysTargetAdmin)
