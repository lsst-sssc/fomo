"""django-tables2 tables extending the TOM Toolkit defaults."""

import django_tables2 as tables
from tom_targets.tables import TargetTable


class ScoutTargetTable(TargetTable):
    """``TargetTable`` with an 'Origin' column.

    The value comes from the ``origin`` annotation added by
    :class:`solsys_code.scout_views.ScoutTargetListView`, so it sorts in the DB.
    """

    origin = tables.Column(verbose_name='Origin')

    class Meta(TargetTable.Meta):
        """Inherit model/fields/HTMX attrs; only fix where the new column sits."""

        sequence = ('selection', 'name', 'origin', '...')
