from datetime import datetime, timedelta, timezone

from django import template

from solsys_code.forms import EphemerisForm

register = template.Library()


@register.inclusion_tag('fomo/partials/ephemeris_form.html')
def ephemeris_form(target):
    """
    Renders a form for requesting an ephemeris for a Target.
    """
    dt = datetime.now(timezone.utc)
    start_date = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=20)
    return {'form': EphemerisForm(initial={'target_id': target.id, 'start_date': start_date, 'end_date': end_date})}
