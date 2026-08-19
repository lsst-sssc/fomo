from django import template

register = template.Library()


@register.inclusion_tag('solsys_code/partials/ephem_button.html', takes_context=True)
def ephem_button(context):
    """
    Returns the app specific context for making a target detail button.
    """

    context = {'button_text': 'Ephemeris'}
    return context


@register.inclusion_tag('solsys_code/partials/ssodnet_card.html', takes_context=True)
def ssodnet_card(context):
    """
    Returns the app specific context for rendering the SsODNet ssoCard on the
    Target detail page.

    TODO: this is a scaffold. Fetch SsODNet data for context['target'] here (e.g.
    via SsODNetDataService().query_service(...), see solsys_code/ssodnet.py) and add
    it to the returned context so ssodnet_card.html has something to render.
    """
    target = context['target']
    return {'target': target}
