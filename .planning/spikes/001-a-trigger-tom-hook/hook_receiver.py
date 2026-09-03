"""Receiver registered into settings.HOOKS['observation_change_state'] by spike.py.

Lives in its own module so TOM's ``run_hook`` can resolve it by dotted path
(``hook_receiver.receiver``) after spike.py puts this directory on ``sys.path``.
"""

FIRED: list[dict] = []


def receiver(record, previous_status):
    """Record every invocation TOM makes; never touches the database."""
    FIRED.append(
        {
            'pk': record.pk,
            'observation_id': record.observation_id,
            'status': record.status,
            'previous_status': previous_status,
            'scheduled_start': None if record.scheduled_start is None else record.scheduled_start.isoformat(),
        }
    )
