import logging
from datetime import datetime, timezone

import requests
from tom_catalogs.harvester import MissingDataException

from solsys_code.solsys_code_observatory.models import Observatory

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class MPCObscodeFetcher:
    """
    The ``MPCObscodeFetcher`` is the interface to the Minor Planet Center (MPC)
    Observatory Codes API
    (https://www.minorplanetcenter.net/mpcops/documentation/obscodes-api/)
    """

    def query(self, obscode: str, dbg: bool = False):
        """Query the MPC obscodes API for the specific <obscode>.
        If successful, the JSON response data is stored in self.obs_data.

        :param obscode: 3 character MPC observatory code to search for
        :type term: str
        :param dbg: Turns on basic print dump of the key-value pairs (or error response)
        :type term: bool
        """
        self.obs_data = None

        response = requests.get('https://data.minorplanetcenter.net/api/obscodes', json={'obscode': obscode})

        if response.ok:
            self.obs_data = response.json()
            if dbg:
                for key, value in response.json().items():
                    logger.debug(f'{key:<27}: {value}')
        else:
            logging.error('Error: ', response.status_code, response.content)
            if dbg:
                print('Error: ', response.status_code, response.content)

    def to_observatory(self):
        """
        Instantiates a ``Observatory`` object with the data from the obscode query search result.

        :returns: ``Observatory` representation of the entry from the ObsCodes API

        """
        if not self.obs_data:
            raise MissingDataException('No observatory data. Did you call query()?')
        else:
            obs = Observatory()

            obs.obscode = self.obs_data['obscode']
            obs.name = self.obs_data['name_utf8']
            obs.short_name = self.obs_data['short_name']
            if self.obs_data['old_names']:
                obs.old_names = self.obs_data['old_names']
            elong = float(self.obs_data['longitude'])
            obs.lon = elong
            # Convert parallax constants to longitude (again), latitude and altitude
            obs.from_parallax_constants(elong, float(self.obs_data['rhocosphi']), float(self.obs_data['rhosinphi']))
            try:
                created_time = datetime.strptime(self.obs_data['created_at'], '%a, %d %b %Y %H:%M:%S %Z')
                created_time = created_time.replace(tzinfo=timezone.utc)
                obs.created = created_time
            except ValueError:
                pass
            try:
                modified_time = datetime.strptime(self.obs_data['updated_at'], '%a, %d %b %Y %H:%M:%S %Z')
                modified_time = modified_time.replace(tzinfo=timezone.utc)
                obs.modified = modified_time
            except ValueError:
                pass
            # Make dictionary of choices using dict comprehension
            obstype_choices = {choice[1].lower(): choice[0] for choice in Observatory.OBSTYPE_CHOICES}
            obs.observations_type = obstype_choices.get(self.obs_data['observations_type'], 0)
            obs.uses_two_line_obs = self.obs_data['uses_two_line_observations']

            obs.save()
        return obs
