from typing import Any

from django.contrib import messages
from django.db.utils import IntegrityError
from django.forms import BaseModelForm
from django.http import HttpResponse
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView, DetailView, ListView

from solsys_code.solsys_code_observatory.forms import CreateObservatoryForm
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.solsys_code_observatory.utils import MPCObscodeFetcher


class CreateObservatory(CreateView):
    """Creates an Observatory object by retrieving the MPC code from the form
    and querying the MPC Obscodes API
    (https://www.minorplanetcenter.net/mpcops/documentation/obscodes-api/)"""

    model = Observatory
    form_class = CreateObservatoryForm
    template_name = 'solsys_code_observatory/observatory_create.html'

    def get_success_url(self):
        """Create a custom success_url to redirect to the detail page for the
        newly created Observatory.
        """
        return reverse_lazy('solsys_code_observatory:detail', kwargs={'pk': self.kwargs['pk']})

    def get_context_data(self, **kwargs):  # noqa: D102
        context = super().get_context_data(**kwargs)
        return context

    def form_valid(self, form: BaseModelForm) -> HttpResponse:
        """
        Runs after the form validation (which ensures that the obscode is uppercased (required))
        Performs the query through the MPC API using ``MPCObscodeFetcher()`` and then tries to
        create the ``Observatory`` through ``MPCObscodeFetcher.to_observatory(). This means
        we shouldn't/don't call the superclass's ``form_valid`` method as this will
        attempt to create a duplicate (which raises an IntegrityError)
        """
        obs = MPCObscodeFetcher()
        obs.query(form.cleaned_data['obscode'])
        try:
            obs = obs.to_observatory()
            self.object = obs
            self.kwargs['pk'] = obs.pk
        except IntegrityError:
            print('Attempt to create duplicate Observatory')
            messages.error(self.request, 'Attempt to create duplicate Observatory')

        return redirect(self.get_success_url())


class ObservatoryDetailView(DetailView):
    """Detailed view of one specific Observatory"""

    model = Observatory
    template_name = 'solsys_code_observatory/observatory_detail.html'

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D102
        return super().get_context_data(**kwargs)


class ObservatoryList(ListView):
    """Overview of all Observatory sites with basic parameters and link to details"""

    model = Observatory
    template_name = 'solsys_code_observatory/observatory_list.html'

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D102
        context = super().get_context_data(**kwargs)

        context['num_obs'] = Observatory.objects.count()
        context['observatory_list'] = Observatory.objects.all()

        return context
