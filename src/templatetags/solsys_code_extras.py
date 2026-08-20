from django import template

from solsys_code.ssodnet import SsODNetDataService, build_card_context

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
    """
    target = context['target']
    service = SsODNetDataService()
    rock = service.query_service(service.build_query_parameters_from_target(target))
    return {'target': target, 'ssodnet': build_card_context(rock)}
