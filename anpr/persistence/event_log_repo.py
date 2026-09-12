"""MySQL-backed implementation of the ``EventLogRepository`` interface.

``MySqlEventLogRepository`` provides append-only access to the ``event_log``
table. It exposes exactly one operation - ``append`` - and no
``find_open_entries``/``close_open_entry`` reads: entry/exit correlation is
retired in the monitoring migration (Req 5.2), so ``handle_detection`` writes
exactly one flat record per detection and never reads back open entries.
Trimming the surface to a single write makes the append-only guarantee
structural rather than merely conventional (Req 4.7).

``append`` persists the *full* ``EventRecord`` schema (design Data Models
section): every field is written, and absent values are written explicitly as
an empty string, NULL, or the ``NA_SENTINEL`` (``"N/A"``) rather than being
omitted. In the flat match-and-log flow ``direction`` and ``event_kind`` are
written as ``NULL`` (via ``_enum_value(None)``) and ``entry_state`` as the
``"N/A"`` value from ``EntryState.NA`` (Req 5.3, 5.4).

Persistence goes through the shared :class:`~anpr.persistence.db.Database`
helper, which owns the single MySQL connection, the access lock, and the
reconnecting liveness check. Every statement uses PyMySQL ``%s`` placeholders
(Req 1.2) and the application issues no DDL (Req 1.3). The ``Database`` instance
is injected via the constructor, keeping this class free of any coupling to the
``ConfigProvider`` (Requirement 14.4).

This concrete class structurally satisfies the ``EventLogRepository`` Protocol
in ``anpr.core.interfaces``.
"""

from __future__ import annotations

from typing import Optional

from anpr.core.models import EventRecord
from anpr.persistence.db import Database

# Columns inserted for every record, in ``EventRecord`` field order.
_COLUMNS = (
    "id, timestamp, ocr_plate, guard_plate, normalized_plate, classification, "
    "direction, grant_method, event_kind, entry_state, closed_by_event_id, "
    "image_ref, detection_confidence, ocr_confidence, processing_latency_ms, "
    "environment_label"
)


class MySqlEventLogRepository:
    """Append-only access to the ``event_log`` table backed by MySQL.

    Parameters
    ----------
    db:
        A shared :class:`~anpr.persistence.db.Database` instance. Sharing the
        instance lets the resident repository and image store reuse the same
        connection, lock, and liveness check.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def append(self, record: EventRecord) -> None:
        """Write exactly one ``EventRecord`` (Req 4.7).

        The insert uses the full column list with PyMySQL ``%s`` placeholders
        (Req 1.2) and goes through the guarded :meth:`Database.execute`, which
        acquires the access lock, pings with reconnect, executes, commits, and
        releases the lock on every path including error (Req 8.1-8.7). In the
        flat match-and-log flow ``direction``/``event_kind`` are written as
        ``NULL`` and ``entry_state`` as the ``"N/A"`` sentinel (Req 5.3, 5.4).
        """
        self._db.execute(
            f"INSERT INTO event_log ({_COLUMNS}) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                record.id,
                record.timestamp,
                record.ocr_plate,
                record.guard_plate,
                record.normalized_plate,
                _enum_value(record.classification),
                _enum_value(record.direction),
                _enum_value(record.grant_method),
                _enum_value(record.event_kind),
                _enum_value(record.entry_state),
                record.closed_by_event_id,
                record.image_ref,
                _metric_value(record.detection_confidence),
                _metric_value(record.ocr_confidence),
                _metric_value(record.processing_latency_ms),
                _enum_value(record.environment_label),
            ),
        )


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------
def _enum_value(member) -> Optional[str]:
    """Return an ``(str, Enum)`` member's ``.value``, or ``None`` when absent."""
    return member.value if member is not None else None


def _metric_value(value: "float | int | str") -> "float | int | str":
    """Persist a numeric metric as-is, or the literal ``"N/A"`` sentinel.

    The value is stored unchanged: a float/int is bound directly and the
    ``NA_SENTINEL`` string is stored verbatim so it round-trips as ``"N/A"``
    (Req 15.5, 16.4).
    """
    return value
