import logging

from tom_observations.facility import BaseManualObservationFacility, BaseManualObservationForm

logger = logging.getLogger(__name__)

# XXX Todo, pull from solsys_observatory
RUBIN_SITES = {
    # Coordinates from NOIRLab pages at:
    # https://noirlab.edu/public/programs/vera-c-rubin-observatory/simonyi-survey-telescope/
    # https://noirlab.edu/public/programs/vera-c-rubin-observatory/rubin-auxtel/
    'Simonyi Survey Telescope': {
        'sitecode': 'X05',
        'latitude': -30.24491667,
        'longitude': -70.74916667,
        'elevation': 2663.0,
    },
    'AuxTel': {'sitecode': 'VRO-AUXTEL', 'latitude': -30.24479722, 'longitude': -70.74772222, 'elevation': 2647.0},
}


class VROFacility(BaseManualObservationFacility):
    """
    Stub of facility for the Vera C. Rubin Observatory
    """

    name = 'VRO'
    observation_types = [('OBSERVATION', 'Manual Observation')]
    observation_forms = {
        'IMAGING': BaseManualObservationForm,
        'SPECTROSCOPY': None,
    }

    def get_form(self, observation_type):
        """
        This method takes in an observation type and returns the form type that matches it.
        """
        return super().get_form(observation_type)

    def get_observing_sites(self):
        """
        Return a dictionary of dictionaries that contain the information
        necessary to be used in the planning (visibility) tool. Each per-site
        dictionary must contain sitecode, latitude, longitude and elevation
        but these needs to be in a dictionary as well, even thoough its a
        single telecope as code calls 'sites.items()' on what is returned
        from here.
        """
        return {'Simonyi Survey Telescope': RUBIN_SITES['Simonyi Survey Telescope']}

    def get_observation_url(self, observation_id):
        """
        Maybe return a top-level RSP link?
        """
        return super().get_observation_url(observation_id)

    def get_terminal_observing_states(self):
        """
        Returns the states for which an observation is not expected
        to change. Not likely to be applicable as we can't request Rubin obs.
        """
        return [
            'Completed',
        ]

    def submit_observation(self, observation_payload):
        """
        Not applicable as we can't request Rubin observations.
        """
        raise NotImplementedError

    def validate_observation(self, observation_payload):
        """
        Not applicable as we can't request Rubin observations.
        """
        raise NotImplementedError
