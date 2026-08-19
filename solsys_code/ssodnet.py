"""
SsODNet DataService integration for FOMO.

Pulls a Target's ssoCard (dynamical + physical properties, each with a literature
reference) from IMCCE's SsODNet service (https://ssp.imcce.fr/webservices/ssodnet/)
and surfaces it on the Target detail page.

THIS IS A SCAFFOLD, NOT A WORKING IMPLEMENTATION. See the TODOs below.

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

        TODO: call `rocks.Rock(query_parameters['name'])`, same as the Fink portal's
        `get_sso_data()` (see reference link in the module docstring above), including
        its fallback behaviour for objects `rocks` can't resolve. Handle "not found"
        by logging at debug level and returning None -- don't raise -- so the target
        detail page can show "no SsODNet data" instead of erroring.
        """
        raise NotImplementedError

    # TODO: figure out the right hook for RENDERING this. ssoCard data (nested
    # dynamical/physical property blocks, each with a value + reference) doesn't fit
    # to_reduced_datums (that's for photometry/spectroscopy time series) or
    # to_target/create_target_from_query (that's for creating NEW targets from a
    # search -- not what this does). Likely path: don't rely on the DataService's own
    # query-form UI at all -- call query_service() directly from the
    # solsys_code_extras.ssodnet_card template tag below, and let
    # SolsysCodeConfig.target_detail_buttons() (apps.py) inject the rendered card into
    # the target detail page, the same way the Ephemeris button is injected today.
