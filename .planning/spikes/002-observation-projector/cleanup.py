"""Spike 002 cleanup: delete exactly the CalendarEvent rows the projector wrote for KEY2026B-004.

Run from the repo root::

    python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/002-observation-projector/cleanup.py', run_name='__main__')"

Selects by the projector's own URL keys AND the spike marker in the description, so a
pre-existing event for one of these requests written by anything else would be left alone.
"""

import os
import sys

SPIKE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SPIKE_DIR)

import projector  # noqa: E402
from tom_calendar.models import CalendarEvent  # noqa: E402
from tom_observations.models import ObservationRecord  # noqa: E402

PROPOSAL = 'KEY2026B-004'


def main() -> None:
    records = ObservationRecord.objects.filter(facility='LCO', parameters__proposal=PROPOSAL)
    urls = {projector.event_url(r, projector.facility_for(r)) for r in records}
    qs = CalendarEvent.objects.filter(url__in=urls, description__contains=projector.SPIKE_MARK)
    n = qs.count()
    qs.delete()
    print(f'deleted {n} spike-owned events; remaining in namespace: {CalendarEvent.objects.filter(url__in=urls).count()}')


if __name__ == '__main__':
    main()
