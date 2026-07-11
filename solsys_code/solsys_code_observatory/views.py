from typing import Any

from django.contrib import messages
from django.db.utils import IntegrityError
from django.forms import BaseModelForm
from django.http import HttpResponse
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.generic import CreateView, DetailView, ListView
from tom_dataservices.dataservices import MissingDataException

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
        """Redirect to a validated ``?next=`` target (SITE-02/D-05) when present -- e.g. back
        to the approval queue for the "Create new Observatory" round-trip from Plan 21-03 --
        falling back to the detail page for the newly created Observatory otherwise.
        Validated with ``url_has_allowed_host_and_scheme`` so an off-host/bad-scheme ``next``
        can never be used as an open redirect (T-21-06).
        """
        next_url = self.request.GET.get('next') or self.request.POST.get('next')
        if next_url and url_has_allowed_host_and_scheme(
            next_url, allowed_hosts={self.request.get_host()}, require_https=self.request.is_secure()
        ):
            return next_url
        return reverse_lazy('solsys_code_observatory:detail', kwargs={'pk': self.kwargs['pk']})

    def get_initial(self):
        """Pre-fill the ``obscode`` field from ``?obscode=`` (SITE-02/D-05) -- e.g. the typed
        text from an unresolved approval-queue row via Plan 21-03's "Create new Observatory"
        link -- so staff don't have to retype it.
        """
        initial = super().get_initial()
        initial['obscode'] = self.request.GET.get('obscode', '')
        return initial

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
        errors = obs.query(form.cleaned_data['obscode'])
        try:
            obs = obs.to_observatory()
            self.object = obs
            self.kwargs['pk'] = obs.pk
        except MissingDataException:
            if errors:
                form.add_error('obscode', errors.get('message', 'Invalid MPC site code'))
            return self.form_invalid(form)

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
