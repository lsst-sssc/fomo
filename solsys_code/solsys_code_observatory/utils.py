import logging
from datetime import datetime, timezone

import requests
from tom_dataservices.dataservices import MissingDataException

from solsys_code.solsys_code_observatory.models import Observatory

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

_timezone_finder = None


def _get_timezone_finder():
    """Return a lazily-constructed, module-cached ``TimezoneFinder`` instance.

    ``TimezoneFinder`` loads its boundary-polygon data on construction, which is
    relatively expensive, so it is built once here and reused across every
    ``to_observatory()`` call rather than per-call. The import itself is deferred
    to inside this function (rather than module level) so that importing
    ``utils.py`` -- which happens broadly across the codebase -- doesn't pay the
    polygon-data load cost unless a timezone lookup is actually needed.

    :returns: shared ``TimezoneFinder`` instance
    """
    global _timezone_finder
    if _timezone_finder is None:
        from timezonefinder import TimezoneFinder

        _timezone_finder = TimezoneFinder()
    return _timezone_finder


class MPCObscodeFetcher:
    """
    The ``MPCObscodeFetcher`` is the interface to the Minor Planet Center (MPC)
    Observatory Codes API
    (https://www.minorplanetcenter.net/mpcops/documentation/obscodes-api/)
    """

    def _flatten_error_dict(self, error_dict):
        """Flattens down the passed `error_dict` for outputting as an error message"""
        non_field_errors = []
        for k, v in error_dict.items():
            if isinstance(v, list):
                for i in v:
                    if isinstance(i, str):
                        non_field_errors.append(f'{k}: {i}')
            elif isinstance(v, str):
                non_field_errors.append(f'{k}: {v}')
        return non_field_errors

    def query(self, obscode: str, dbg: bool = False, timeout: float = 10):
        """Query the MPC obscodes API for the specific <obscode>.
        If successful, the JSON response data is stored in self.obs_data.

        :param obscode: 3 character MPC observatory code to search for
        :type term: str
        :param dbg: Turns on basic print dump of the key-value pairs (or error response)
        :type term: bool
        :param timeout: request timeout in seconds, passed through to ``requests.get``.
            Callers that need "never hang" behavior (e.g. a synchronous per-row import
            loop) should rely on this rather than the default of no timeout at all.
        :type timeout: float
        """
        self.obs_data = None

        response = requests.get(
            'https://data.minorplanetcenter.net/api/obscodes', json={'obscode': obscode}, timeout=timeout
        )

        if response.ok:
            self.obs_data = response.json()
            if dbg:
                for key, value in response.json().items():
                    logger.debug(f'{key:<27}: {value}')
        else:
            json_resp = response.json()
            errors = self._flatten_error_dict(json_resp)
            logging.error(f'Error: {response.status_code} Message: {". ".join(errors)}')
            if dbg:
                print('Error: ', response.status_code, self._flatten_error_dict(json_resp))
            return json_resp

    def query_all(self, timeout: float = 30) -> dict:
        """Query the MPC obscodes API for every registered observatory code (bulk mode).

        Omitting the ``obscode`` key from the POST body triggers the bulk-list response
        (confirmed live: 2,710 codes, ~1.5 MB, ~1.3s as of 2026-07-11). Stores the result
        on ``self.obs_data`` like ``query()`` does, but here it is a dict keyed by 3-char
        obscode rather than a single flat observatory dict -- do **not** call
        ``to_observatory()`` on a ``query_all()`` result, its ``self.obs_data`` shape
        contract is for ``query()`` only. This is a distinct, sibling method: ``query()``
        itself is unmodified.

        :param timeout: request timeout in seconds, passed through to ``requests.get``.
        :type timeout: float
        :returns: dict keyed by obscode, e.g. {'X09': {'name_utf8': ..., 'longitude': ..., ...}}
        :rtype: dict
        """
        response = requests.get('https://data.minorplanetcenter.net/api/obscodes', json={}, timeout=timeout)
        response.raise_for_status()
        self.obs_data = response.json()
        return self.obs_data

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
            # Backfill timezone from the resolved coordinates when the MPC record doesn't
            # supply one (it never does in live data). A value already present on the record
            # is authoritative and is never overwritten by the coordinate lookup.
            obs.timezone = self.obs_data.get('timezone', '') or ''
            if not obs.timezone and obs.lat is not None and obs.lon is not None:
                tz_name = _get_timezone_finder().timezone_at(lat=obs.lat, lng=obs.lon)
                # A coordinate with no timezone polygon (e.g. open ocean) leaves timezone
                # blank rather than fabricating a guess, preserving the CR-01
                # resolve-fails-gracefully / stays-retryable behavior.
                if tz_name:
                    obs.timezone = tz_name
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
