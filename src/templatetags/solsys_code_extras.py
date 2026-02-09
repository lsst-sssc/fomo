from django import template

register = template.Library()


@register.inclusion_tag('solsys_code/partials/ephem_button.html', takes_context=True)
def ephem_button(context):
    """
    Returns the app specific context for making a target detail button.
    """

    context = {'button_text': 'Ephemeris'}
    return context
