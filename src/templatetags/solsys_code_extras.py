from django import template
from tom_targets.models import TargetList

register = template.Library()


@register.inclusion_tag('solsys_code/partials/ephem_button.html', takes_context=True)
def ephem_button(context):
    """
    Returns the app specific context for making a target detail button.
    """

    context = {'button_text': 'Ephemeris'}
    return context


@register.inclusion_tag('solsys_code/partials/campaign_links.html', takes_context=True)
def campaign_links(context):
    """
    Returns every campaign (TargetList with >= 1 CampaignRun) the rendered target belongs to,
    discovered via TargetList membership -- never via the run's optional target FK (D-01).
    """
    target = context.get('target')
    campaigns = (
        TargetList.objects.filter(targets=target, campaign_runs__isnull=False).distinct()
        if target
        else TargetList.objects.none()
    )
    return {'campaigns': campaigns}


@register.inclusion_tag('solsys_code/partials/campaigns_nav_link.html', takes_context=True)
def campaigns_nav_link(context):
    """
    Static navbar entry linking to the campaigns list page (D-03). No per-request data needed;
    the partial reads `request` from the surrounding page context for the active-nav check.
    """
    return {}
