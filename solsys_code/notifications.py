"""Request-free staff-notification helper (D-11, 36-CONTEXT.md).

Shared by the campaign submission notice (``campaign_views.CampaignRunSubmissionView``)
and the unattended runner (``solsys_code.unattended``), so a recipient-rule or body change
can never diverge between the two callers. Lifted out of the former
``CampaignRunSubmissionView._notify_staff()`` body, made request-free by taking
``settings.FOMO_BASE_URL`` in place of ``request.build_absolute_uri``.

This module imports only ``django.conf``, ``django.contrib.auth.models`` and
``django.core.mail`` -- it must stay safe to import from a management-command-only
process (the unattended runner) with no web request in flight.
"""

from django.conf import settings
from django.contrib.auth.models import User
from django.core.mail import send_mail


def staff_recipients() -> list[str]:
    """Return the email addresses of every staff user with an email on file (D-13).

    Returns:
        list[str]: staff user email addresses, in queryset order. Empty when no staff
            user has one set -- not an error, just nobody to notify.
    """
    return list(User.objects.filter(is_staff=True).exclude(email='').values_list('email', flat=True))


def absolute_url(path: str) -> str:
    """Join ``settings.FOMO_BASE_URL`` with a path, in place of a request-bound
    ``build_absolute_uri`` call.

    Args:
        path: a URL path, typically the output of ``reverse()`` (a caller with an active
            request/URL-conf context) or a literal path (a caller, such as the unattended
            runner, that must not trigger URL-conf resolution -- see
            ``solsys_code.unattended``'s module docstring for why).

    Returns:
        str: the absolute URL.
    """
    return f"{settings.FOMO_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def notify_staff(subject: str, message: str, *, fail_silently: bool = False) -> bool:
    """Email every staff recipient with an email on file (D-11/D-13).

    Args:
        subject: email subject.
        message: email body. Callers are responsible for keeping it free of PII and of
            any credential value (D-16/D-17) -- this helper does not scrub content.
        fail_silently: passed through to ``send_mail()``. The unattended runner calls
            this with ``fail_silently=False`` and catches the exception itself, logging
            only its class name (D-17); the campaign submission notice keeps its
            existing outage-tolerant ``fail_silently=True`` semantics at its own call
            site -- a mail outage must never break a submission.

    Returns:
        bool: True if there was at least one recipient (an attempt was made, whether or
            not it succeeded), False if there were no recipients (not an error, and
            never attempted).

    Raises:
        Exception: whatever ``send_mail()`` raised, only when ``fail_silently`` is
            False. ``fail_silently`` is handled by this function's own try/except
            rather than delegated to ``send_mail()``'s own parameter of the same name,
            so behavior is identical regardless of which layer (this function,
            ``send_mail()`` itself, or the configured email backend) a caller mocks or
            a real outage occurs at.
    """
    recipients = staff_recipients()
    if not recipients:
        return False
    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=None,
            recipient_list=recipients,
            fail_silently=False,
        )
    except Exception:
        if not fail_silently:
            raise
    return True
