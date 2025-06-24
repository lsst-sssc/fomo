import requests
from typing import Optional, Dict, Any
from astropy.table import QTable
import logging
import json
import urllib.parse


class JPLSBDBQuery:
    """
    The ``JPLSBDBQuery`` provides an interface to JPL's Small Body Database Query
    via its API interface (https://ssd.jpl.nasa.gov/tools/sbdb_query.html)
    """

    base_url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

    def __init__(self, orbit_class=None, orbital_constraints=None):
        """
        orbit_class: str or None (e.g. "IEO", "TJN", etc.)
        orbital_constraints: list of constraint strings, e.g. ["q|LT|1.3", "i|LT|10.5"]
        """
        self.orbit_class = orbit_class
        self.orbital_constraints_raw = orbital_constraints or []
        self.orbital_constraints = self._translate_constraints(self.orbital_constraints_raw)

    def _translate_constraints(self, constraints):
        translated = []
        for c in constraints:
            # Handle range (e.g. 6<=H<=7)
            if "<=" in c and c.count("<=") == 2:
                parts = c.split("<=")
                min_val = parts[0].strip()
                field = parts[1].strip()
                max_val = parts[2].strip()
                translated.append(f"{field}|RG|{min_val}|{max_val}")
                continue

            # Handle <, <=, >, >=
            if "<=" in c:
                field, value = c.split("<=")
                translated.append(f"{field.strip()}|LE|{value.strip()}")
            elif ">=" in c:
                field, value = c.split(">=")
                translated.append(f"{field.strip()}|GE|{value.strip()}")
            elif "<" in c:
                field, value = c.split("<")
                translated.append(f"{field.strip()}|LT|{value.strip()}")
            elif ">" in c:
                field, value = c.split(">")
                translated.append(f"{field.strip()}|GT|{value.strip()}")
            else:
                raise ValueError(f"Unsupported constraint format: {c}")

        return translated

    def build_query_url(self):
        # Base query fields
        params = {
            "fields": "full_name,first_obs,epoch,e,a,q,i,om,w"
        }

        # Add sb-class if provided
        if self.orbit_class:
            params["sb-class"] = self.orbit_class

        # Add sb-cdata if constraints provided
        if self.orbital_constraints:
            constraint_obj = {"AND": self.orbital_constraints}
            json_str = json.dumps(constraint_obj, separators=(',', ':'))
            encoded_cdata = urllib.parse.quote(json_str)
            params["sb-cdata"] = encoded_cdata

        # Build URL
        query_parts = [f"{key}={urllib.parse.quote(str(value))}" for key, value in params.items()]
        url = f"{self.base_url}?" + "&".join(query_parts)
        return url

    def run_query(self) -> Optional[Dict[str, Any]]:
        """
        Execute the query and return results as JSON (if successful).
        """
        url = self.build_query_url()
        resp = requests.get(url)

        if resp.ok:
            return resp.json()
        else:
            logger = logging.getLogger(__name__)
            logger.debug(f"Query failed with status {resp.status_code}")
            return None

    def parse_results(self, results: Dict[str, Any]) -> QTable:
        """
        Parse JSON results into an Astropy QTable.
        """
        if not results or "data" not in results:
            logger = logging.getLogger(__name__)
            logger.debug(f"No data found in results")
            return QTable()

        data = results["data"]
        columns = results["fields"]
        table = QTable(rows=data, names=columns)

        return table