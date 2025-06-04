from typing import Any

import requests
from django.urls import reverse_lazy
from django.views.generic import CreateView, DetailView, ListView

from solsys_code.solsys_code_observatory.forms import CreateObservatoryForm
from solsys_code.solsys_code_observatory.models import Observatory


class CreateObservatory(CreateView):
    """Creates an Observatory object by retrieving the MPC code from the form
    and querying the MPC Obscodes API
    (https://www.minorplanetcenter.net/mpcops/documentation/obscodes-api/)"""

    model = Observatory
    form_class = CreateObservatoryForm
    template_name = 'solsys_code_observatory/observatory_create.html'
    success_url = reverse_lazy('solsys_code_observatory:<pk>')

    def get_data(self, obscode):
        """Query the MPC obscodes API for the specific <obscode>
        XXX needs more work
        """
        resp = requests.get('https://data.minorplanetcenter.net/api/obscodes', json={'obscode': obscode})

        if resp.ok:
            for key, value in resp.json().items():
                print(f'{key:<27}: {value}')
        else:
            print('Error: ', resp.status_code, resp.content)


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

        print(context['observatory_list'])
        return context
