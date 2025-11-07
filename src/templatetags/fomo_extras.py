from django import template

from solsys_code.forms import EphemerisForm

register = template.Library()


@register.inclusion_tag('fomo/partials/ephemeris_form.html')
def ephemeris_form(target):
    """
    Renders a form for requesting an ephemeris for a Target.
    """

    return {'form': EphemerisForm(initial={'target_id': target.id})}
