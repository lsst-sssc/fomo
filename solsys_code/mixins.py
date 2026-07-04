from django.contrib.auth.decorators import user_passes_test
from django.utils.decorators import method_decorator


class StaffRequiredMixin:
    """Redirect to LOGIN_URL unless request.user.is_staff (D-01 approval-queue gate)."""

    @method_decorator(user_passes_test(lambda u: u.is_staff))
    def dispatch(self, *args, **kwargs):
        """Redirect non-staff/anonymous requests to LOGIN_URL before dispatching."""
        return super().dispatch(*args, **kwargs)
