"""
Template tags for non-sidereal visibility: TOM's airmass "Plan" panel plus the multi-site cadence window.
"""

from django import template
from plotly import graph_objs as go
from plotly import offline
from plotly.colors import qualitative
from tom_observations.facility import get_service_class

from solsys_code.ephem_utils import get_nonsidereal_visibility
from solsys_code.forms import NonSiderealVisibilityForm
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.visibility import cadence_window, visibility_windows

register = template.Library()

# LCO site code -> MPC code of one dome at that site (inter-dome differences are negligible for visibility)
LCO_SITE_OBSCODES = {'coj': 'Q63', 'cpt': 'K91', 'lsc': 'W85', 'tfn': 'Z31', 'elp': 'V38', 'ogg': 'T04'}
DEFAULT_SITES = 'lsc,cpt,coj'


def lco_observatories(sitecodes):
    """
    Maps LCO site codes to ``Observatory`` rows keyed by upper-case site code, creating any that are missing
    from the LCO facility's site list.

    :param sitecodes: LCO site codes, e.g. ``['lsc', 'cpt', 'coj']``
    :type sitecodes: list[str]
    :rtype: dict[str, Observatory]
    """
    lco_sites = get_service_class('LCO')().get_observing_sites()
    by_sitecode = {details['sitecode']: (name, details) for name, details in lco_sites.items()}
    observatories = {}
    for sitecode in sitecodes:
        name, details = by_sitecode[sitecode]
        observatory, _ = Observatory.objects.get_or_create(
            obscode=LCO_SITE_OBSCODES[sitecode],
            defaults={
                'name': f'{name}-LCO',
                'lat': details['latitude'],
                'lon': details['longitude'],
                'altitude': details['elevation'],
            },
        )
        observatories[sitecode.upper()] = observatory
    return observatories


def airmass_figure(visibility, width=600, height=400):
    """
    Airmass against time, one line per site, in the same style as TOM's ``target_plan`` tag.

    :param visibility: ``{site: (times, airmasses)}`` as returned by ``get_nonsidereal_visibility``
    :rtype: plotly.graph_objs.Figure
    """
    data = [
        go.Scatter(x=list(times), y=airmasses, mode='lines', name=site)
        for site, (times, airmasses) in visibility.items()
    ]
    layout = go.Layout(
        xaxis={'title': 'UTC'}, yaxis={'autorange': 'reversed', 'title': 'Airmass'}, width=width, height=height
    )
    return go.Figure(data=data, layout=layout)


def cadence_figure(windows, window, width=600, height=300):
    """
    Horizontal bars of the visibility windows per site, an "All sites" row of the merged coverage inside
    the cadence window and a dashed line at its midpoint.

    :param windows: ``{site: [(start, end), ...]}`` as returned by ``visibility_windows``
    :param window: The ``CadenceWindow`` computed from those intervals
    :rtype: plotly.graph_objs.Figure
    """
    fig = go.Figure()
    rows = list(windows.items()) + [('All sites', window.coverage)]
    for i, (row, intervals) in enumerate(rows):
        color = qualitative.Plotly[i % len(qualitative.Plotly)]
        for start, end in intervals:
            fig.add_trace(
                go.Bar(
                    y=[row],
                    base=[start],
                    x=[(end - start).total_seconds() * 1000],
                    orientation='h',
                    marker_color=color,
                    showlegend=False,
                    hovertemplate=f'{row}: {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC<extra></extra>',
                )
            )
    fig.add_shape(type='line', x0=window.midpoint, x1=window.midpoint, y0=0, y1=1, yref='paper', line={'dash': 'dash'})
    fig.add_annotation(
        x=window.midpoint,
        y=1,
        yref='paper',
        yanchor='bottom',
        showarrow=False,
        text=f'midpoint {window.midpoint:%H:%M} UTC',
    )
    fig.update_layout(
        barmode='overlay',
        xaxis={'type': 'date', 'title': 'UTC'},
        yaxis={'autorange': 'reversed'},
        width=width,
        height=height,
        margin={'t': 40},
    )
    return fig


def window_summary(window):
    """One-line description of a ``CadenceWindow`` in the terms of an LCO cadence request."""
    hours = window.duration.total_seconds() / 3600
    summary = (
        f'{window.start:%H:%M}–{window.end:%H:%M} UTC ({hours:.1f} h): '
        f'midpoint {window.midpoint:%H:%M} UTC ± {hours / 2:.1f} h'
    )
    if window.gaps:
        gaps = ', '.join(f'{start:%H:%M}–{end:%H:%M}' for start, end in window.gaps)
        summary += f'; gaps within the window: {gaps} UTC'
    return summary


@register.inclusion_tag('solsys_code/partials/nonsidereal_target_plan.html', takes_context=True)
def nonsidereal_target_plan(context, sites=DEFAULT_SITES, interval=15, width=600, height=400):
    """
    Non-sidereal counterpart of TOM's ``target_plan`` tag: a form for the date range and airmass limit,
    airmass against time for each LCO site and the merged cadence window (see ``solsys_code.visibility``).

    :param sites: Comma-separated LCO site codes to sample (keys of ``LCO_SITE_OBSCODES``)
    :param interval: Sampling interval in minutes
    """
    request = context['request']
    target = context['object']
    result = {
        'form': NonSiderealVisibilityForm(),
        'target': target,
        'airmass_graph': '',
        'window_graph': '',
        'window_summary': '',
        'message': '',
    }
    if not all(request.GET.get(x) for x in ['start_time', 'end_time']):
        return result
    form = NonSiderealVisibilityForm(
        {
            'start_time': request.GET['start_time'],
            'end_time': request.GET['end_time'],
            'airmass': request.GET.get('airmass', 2.5),
        }
    )
    result['form'] = form
    if not form.is_valid():
        return result

    airmass_limit = form.cleaned_data['airmass']
    visibility = get_nonsidereal_visibility(
        target,
        lco_observatories(sites.split(',')),
        form.cleaned_data['start_time'],
        form.cleaned_data['end_time'],
        interval,
        None if airmass_limit is None else float(airmass_limit),
    )
    result['airmass_graph'] = offline.plot(
        airmass_figure(visibility, width, height), output_type='div', show_link=False
    )

    windows = visibility_windows(visibility)
    intervals = [site_interval for site_intervals in windows.values() for site_interval in site_intervals]
    if intervals:
        window = cadence_window(intervals)
        result['window_summary'] = window_summary(window)
        result['window_graph'] = offline.plot(
            cadence_figure(windows, window, width, height * 3 // 4), output_type='div', show_link=False
        )
    else:
        result['message'] = 'Target is not observable from the selected sites in this date range.'
    return result
