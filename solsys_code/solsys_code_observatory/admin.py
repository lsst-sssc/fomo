from django.contrib import admin

from solsys_code.solsys_code_observatory.models import Observatory


class ObservatoryAdmin(admin.ModelAdmin):  # noqa: D101
    model = Observatory
    list_display = ['obscode', 'name', 'lon', 'lat', 'altitude']
    ordering = ['obscode']


admin.site.register(Observatory, ObservatoryAdmin)
