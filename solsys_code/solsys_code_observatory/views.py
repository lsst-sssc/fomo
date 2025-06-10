from datetime import datetime, timezone
from typing import Any

import requests
from django.urls import reverse_lazy
from django.views.generic import CreateView, DetailView, ListView
from tom_catalogs.harvester import MissingDataException

from solsys_code.solsys_code_observatory.forms import CreateObservatoryForm
from solsys_code.solsys_code_observatory.models import Observatory


class MPCObscodeFetcher:
    """
    The ``MPCObscodeFetcher`` is the interface to the Minor Planet Center (MPC)
    Observatory Codes API
    (https://www.minorplanetcenter.net/mpcops/documentation/obscodes-api/)
    """

    def query(self, obscode: str):
        """Query the MPC obscodes API for the specific <obscode>
        XXX needs more work

        :param obscode: 3 character MPC observatory code to search for
        :type term: str
        """
        self.obs_data = None

        response = requests.get('https://data.minorplanetcenter.net/api/obscodes', json={'obscode': obscode})

        if response.ok:
            self.obs_data = response.json()
            for key, value in response.json().items():
                print(f'{key:<27}: {value}')
        else:
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
            obs.save()
        return obs


class CreateObservatory(CreateView):
    """Creates an Observatory object by retrieving the MPC code from the form
    and querying the MPC Obscodes API
    (https://www.minorplanetcenter.net/mpcops/documentation/obscodes-api/)"""

    model = Observatory
    form_class = CreateObservatoryForm
    template_name = 'solsys_code_observatory/observatory_create.html'
    success_url = reverse_lazy('solsys_code_observatory:<pk>')

    def get_data(self, obscode):  # noqa: D102
        pass


class ObservatoryDetailView(DetailView):
    """Detailed view of one specific Observatory"""

    model = Observatory
    template_name = 'solsys_code_observatory/observatory_detail.html'

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D102
        return super().get_context_data(**kwargs)


class ObservatoryList(ListView):
    """Overview of all Observatory site with basic parameters and link to details"""

    model = Observatory
    template_name = 'solsys_code_observatory/observatory_list.html'

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D102
        context = super().get_context_data(**kwargs)

        context['num_obs'] = Observatory.objects.count()
        context['observatory_list'] = Observatory.objects.all()

        return context
