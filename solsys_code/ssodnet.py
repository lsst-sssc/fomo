"""
SsODNet DataService integration for FOMO.

Pulls a Target's ssoCard (dynamical + physical properties, each with a literature
reference) from IMCCE's SsODNet service (https://ssp.imcce.fr/webservices/ssodnet/)
and surfaces it on the Target detail page.

THIS IS PARTIALLY IMPLEMENTED. query_service() fetches ssoCard data via `rocks`;
rendering it on the target detail page is still a TODO (see the note at the
bottom of this file and SSODNET_HANDOFF.md).

Before writing the real logic, run this in the FOMO dev environment to confirm the
exact method signatures/hooks available in the installed tomtoolkit version (this
scaffold was written from TOM Toolkit docs + the tom_fink plugin as reference, not
from the installed source directly):

    python -c "import tom_dataservices.dataservices as m; help(m.DataService)"

Reference implementations to compare against:
  - tom_fink.fink.FinkDataService -- closest existing example in this dependency set
    of build_query_parameters_from_target() -> query_service() enriching an EXISTING
    target (as opposed to tom_jpl's ScoutDataService / our own JPLSBDBQuery, which
    both *search* and create NEW targets -- that's not what this does).
    https://github.com/TOMToolkit/tom_fink
  - Fink portal's own SsODNet card renderer (not a TOM Toolkit plugin, but the
    reference for *how* to fetch/parse ssoCard data via the `rocks` package):
    https://github.com/astrolabsoftware/ztf.fink-portal.org/blob/master/apps/sso/cards.py
"""

import logging
import math

import rocks
from tom_dataservices.dataservices import DataService

logger = logging.getLogger(__name__)


class SsODNetDataService(DataService):
    """
    Per-target lookup of a Solar System object's ssoCard from IMCCE's SsODNet.

    Unlike tom_fink/tom_jpl, this service is NOT meant to search for or create new
    Targets -- it enriches an EXISTING Target's detail page with ssoCard data.
    """

    name = 'SsODNet'
    info_url = 'https://ssp.imcce.fr/webservices/ssodnet/'

    # RESOLVED (see SSODNET_HANDOFF.md): confirmed no config needed -- SsODNet's
    # ssoCard API is public, no API key/contact-email/rate-limit config required.
    # No `configuration()` override needed; settings.DATA_SERVICES['SsODNet'] can
    # stay absent.

    def build_query_parameters_from_target(self, target, **kwargs):
        """
        Turn an existing FOMO Target into the identifier SsODNet needs.

        RESOLVED (see SSODNET_HANDOFF.md): confirmed `target.name` resolves fine via
        SsODNet's quaero resolver as-is -- no prefix handling or alias fallback
        needed.
        """
        return {'name': target.name}

    def query_service(self, query_parameters, **kwargs):
        """
        Fetch the ssoCard for the resolved name.

        Returns the `rocks.Rock` instance for `query_parameters['name']`, or None if
        SsODNet has no ssoCard for that identifier (or the lookup fails outright).

        CONFIRMED (2026-08-19, real `rocks` call): an unresolvable name raises
        `KeyError: 'ssocard'` rather than returning a Rock with an empty id -- the
        `except Exception` below catches that and returns None. The `rock.id_ is
        None` check is kept as a defensive fallback in case some other not-found
        path doesn't raise, but the exception path is the one that's actually
        confirmed to fire.
        """
        name = query_parameters.get('name')
        if not name:
            return None

        try:
            rock = rocks.Rock(name)
        except Exception:
            logger.exception('SsODNet lookup failed for %s', name)
            return None

        not_found = rock is None or getattr(rock, 'id_', None) is None
        if not_found:
            logger.debug('No SsODNet ssoCard found for %s', name)
            return None

        return rock


def _clean_float(value):
    """
    SsODNet represents "no data" for a numeric property as NaN, not None. Convert
    that to None so templates can use a simple {% if %} rather than NaN-checking.
    """
    try:
        return None if math.isnan(value) else value
    except TypeError:
        return value


def _references(bibrefs):
    """
    Turn a rocks `bibref` list (Bibref namedtuple-likes) into plain dicts for the
    template, dropping empty placeholder entries (rocks fills unused slots with
    Bibref(shortbib='', bibcode='', ...) rather than omitting them).
    """
    return [
        {'shortbib': b.shortbib, 'bibcode': b.bibcode, 'doi': b.doi}
        for b in (bibrefs or [])
        if getattr(b, 'shortbib', '')
    ]


def build_card_context(rock):
    """
    Shape a `rocks.Rock` (as returned by SsODNetDataService.query_service()) into the
    flat dict solsys_code/partials/ssodnet_card.html expects.

    Mirrors the fields the Fink portal's sso/cards.py shows for the "at a glance"
    case (see module docstring): name/class/parent body/dynamical system, then
    physical parameters -- taxonomy, absolute magnitude (H) + slope parameter (G),
    diameter, albedo -- each with its SsODNet reference(s). Returns None if `rock`
    is None (SsODNet has no card for this target).

    Orbital/dynamical properties (moid, proper elements, Yarkovsky, ...) and the
    more structurally complex ones (spin -- a list of possibly-multiple solutions;
    mass/density -- not wanted for this card / not requested) are deliberately left
    out of this first pass. Worth a follow-up if spin is wanted later.
    """
    if rock is None:
        return None

    physical = rock.parameters.physical
    return {
        'name': rock.name,
        'number': rock.number,
        'class_': rock.class_ or None,
        'parent': rock.parent or None,
        'system': rock.system or None,
        'taxonomy': {
            'value': physical.taxonomy.class_.value or None,
            'references': _references(physical.taxonomy.bibref),
        },
        'absolute_magnitude': {
            # H and G share one reference list on the parent absolute_magnitude
            # object (confirmed against a live rocks.Rock('Eros') call).
            'H': _clean_float(physical.absolute_magnitude.H.value),
            'G': _clean_float(physical.absolute_magnitude.G.value),
            'references': _references(physical.absolute_magnitude.bibref),
        },
        'diameter': {
            'value': _clean_float(physical.diameter.value),
            'unit': 'km',
            'references': _references(physical.diameter.bibref),
        },
        'albedo': {
            'value': _clean_float(physical.albedo.value),
            'references': _references(physical.albedo.bibref),
        },
    }
