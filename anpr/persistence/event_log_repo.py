"""MySQL-backed implementation of the ``EventLogRepository`` interface.

``MySqlEventLogRepository`` provides append-only access to the ``ANPR_Log``
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

from datetime import datetime, timezone
from typing import Optional

from anpr.core.models import EventRecord
from anpr.persistence.db import Database

# Columns inserted for every record, in ``ANPR_Log`` schema naming.
#
# ``Log_ID`` is deliberately absent: it is AUTO_INCREMENT and assigned by MySQL,
# so the application no longer supplies its own event id. ``append`` returns the
# generated value and stamps it onto the record.
_COLUMNS = (
    "Inserted_Time, Camera_ID, License_Plate_Number, Normalized_Plate, "
    "Guard_Plate, Vehicle_ID, Classification, Direction, Event_Kind, "
    "Grant_Method, Entry_State, Detection_Confidence, OCR_Confidence, "
    "Processing_Time_MS, Environment_Label, Image_Ref, Closed_By_Log_ID"
)

_PLACEHOLDERS = ", ".join(["%s"] * len(_COLUMNS.split(",")))


class MySqlEventLogRepository:
    """Append-only access to the ``ANPR_Log`` table backed by MySQL.

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
    def append(self, record: EventRecord) -> int:
        """Write exactly one ``EventRecord`` and return its ``Log_ID`` (Req 4.7).

        The insert uses the full column list with PyMySQL ``%s`` placeholders
        (Req 1.2) and goes through the guarded
        :meth:`Database.execute_returning_id`, which acquires the access lock,
        pings with reconnect, executes, reads ``lastrowid`` inside the lock,
        commits, and releases the lock on every path including error
        (Req 8.1-8.7). In the flat match-and-log flow ``Direction``/``Event_Kind``
        are written as ``NULL`` and ``Entry_State`` as the ``"N/A"`` sentinel
        (Req 5.3, 5.4).

        The generated ``Log_ID`` is both returned and stamped onto
        ``record.log_id`` so the caller can attach the ``Images`` row, whose
        ``Log_ID`` FK is NOT NULL.
        """
        log_id = self._db.execute_returning_id(
            f"INSERT INTO ANPR_Log ({_COLUMNS}) VALUES ({_PLACEHOLDERS})",
            (
                _datetime_value(record.timestamp),
                record.camera_id,
                record.ocr_plate,
                record.normalized_plate,
                record.guard_plate,
                record.vehicle_id,
                _enum_value(record.classification),
                _enum_value(record.direction),
                _enum_value(record.event_kind),
                _enum_value(record.grant_method),
                _enum_value(record.entry_state),
                _metric_value(record.detection_confidence),
                _metric_value(record.ocr_confidence),
                _metric_value(record.processing_latency_ms),
                _enum_value(record.environment_label),
                record.image_ref,
                _int_or_none(record.closed_by_event_id),
            ),
        )
        record.log_id = log_id
        return log_id


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------
def _enum_value(member) -> Optional[str]:
    """Return an ``(str, Enum)`` member's ``.value``, or ``None`` when absent."""
    return member.value if member is not None else None


def _metric_value(value: "float | int | str | None") -> "float | int | None":
    """Bind a numeric metric, mapping an absent value to SQL ``NULL``.

    The ``ANPR_Log`` metric columns are ``DECIMAL`` (``Detection_Confidence`` and
    ``OCR_Confidence`` are ``DECIMAL(5,4)``, ``Processing_Time_MS`` is
    ``DECIMAL(10,2)``), so the domain's ``NA_SENTINEL`` string cannot be stored
    there: under MySQL's default ``STRICT_TRANS_TABLES`` binding ``"N/A"`` to a
    DECIMAL column raises error 1366 and the whole insert fails, which would drop
    any event with a missing metric (an OCR timeout, for instance).

    All three columns are nullable, so an absent metric becomes ``NULL``. Absence
    is still represented explicitly rather than omitted (Req 15.5, 16.4); the
    representation is simply SQL ``NULL`` instead of the ``"N/A"`` string, and
    readers render ``NULL`` back as "N/A" for display.

    Any non-numeric string is treated as absent, so an unexpected sentinel can
    never turn a single event into a failed insert.
    """
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, bool):  # bool is an int subclass; not a real metric
        return None
    return value


def _datetime_value(value: "str | datetime | None") -> Optional[datetime]:
    """Coerce a record timestamp to a naive UTC ``datetime`` for ``DATETIME``.

    ``EventRecord.timestamp`` is ISO-8601 *with a UTC offset* by design
    (Req 10.3), but ``ANPR_Log.Inserted_Time`` is a plain MySQL ``DATETIME``,
    which has no timezone component and rejects the trailing offset. The value is
    therefore converted to UTC and made naive before binding.

    ``None`` is returned when the timestamp is missing or unparseable, letting the
    column's ``DEFAULT CURRENT_TIMESTAMP`` supply the time rather than failing the
    insert.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _int_or_none(value) -> Optional[int]:
    """Coerce a value to ``int``, or ``None`` when absent/non-numeric.

    ``Closed_By_Log_ID`` is an ``INT`` FK, whereas the retired correlation flow
    carried a string event id. It is always ``None`` in the flat flow (Req 5.2).
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
