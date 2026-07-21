"""SQLite-backed implementation of the ``EventLogRepository`` interface.

``SqliteEventLogRepository`` provides append-and-correlate access to the
``Event_Log``. It persists the *full* ``EventRecord`` schema (design Data
Models section): every field is stored, and absent values are written
explicitly as an empty string, NULL, or the ``NA_SENTINEL`` (``"N/A"``) rather
than being omitted (Req 10.4, 10.8, 15.5, 16.4).

Three operations back the entry/exit correlation flow:

* ``append`` writes exactly one record inside a single transaction, so exactly
  one record is persisted per access attempt (Req 10.1, 10.2). A technical
  fault rolls the transaction back, leaving no partial record and every
  previously written record byte-for-byte unchanged (Req 8.3, 10.10).
* ``find_open_entries`` returns the Open_Entry_Records (open entry events) for
  an exact normalized-plate match, used by outbound exit correlation (Req 9).
* ``close_open_entry`` closes an Open_Entry_Record, recording the exit event
  that closed it (Req 9.2, 9.3).

Persistence goes through the shared :class:`~anpr.persistence.db.Database`
helper, which owns the SQLite connection and table creation. The database
location comes from configuration (``database.location``) and is injected via
the constructor, keeping this class free of any coupling to the
``ConfigProvider`` (Requirement 14.4).

This concrete class structurally satisfies the ``EventLogRepository`` Protocol
in ``anpr.core.interfaces``.
"""

from __future__ import annotations

from typing import Optional

from anpr.core.models import (
    Classification,
    DirectionOutcome,
    EntryState,
    EnvironmentLabel,
    EventKind,
    EventRecord,
    GrantMethod,
    NA_SENTINEL,
)
from anpr.persistence.db import Database

# Columns selected/inserted for every record, in ``EventRecord`` field order.
_COLUMNS = (
    "id, timestamp, ocr_plate, guard_plate, normalized_plate, classification, "
    "direction, grant_method, event_kind, entry_state, closed_by_event_id, "
    "image_ref, detection_confidence, ocr_confidence, processing_latency_ms, "
    "environment_label"
)


class SqliteEventLogRepository:
    """Append-and-correlate access to the ``Event_Log`` backed by SQLite.

    Parameters
    ----------
    db:
        A shared :class:`~anpr.persistence.db.Database` instance, or a database
        location string. Passing a string is a convenience that constructs the
        ``Database`` internally; passing an instance lets this repository share
        the same connection and schema as the resident repository.
    """

    def __init__(self, db: Database | str) -> None:
        self._db = db if isinstance(db, Database) else Database(db)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def append(self, record: EventRecord) -> None:
        """Write exactly one ``EventRecord`` atomically.

        The insert runs inside a single ``with connection`` transaction block:
        on success it commits one row; on any technical fault the transaction
        is rolled back so no partial record is persisted and previously written
        records are left byte-for-byte unchanged (Req 8.3, 10.1, 10.2, 10.10).
        """
        with self._db.connection as conn:
            conn.execute(
                f"INSERT INTO event_log ({_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def close_open_entry(self, entry_id: str, exit_id: str) -> None:
        """Close an Open_Entry_Record, referencing the closing exit event.

        Sets the entry record's ``entry_state`` to ``CLOSED`` and its
        ``closed_by_event_id`` to ``exit_id`` (Req 9.2, 9.3). The update runs in
        a single transaction; when no record with ``entry_id`` exists the
        database is left unchanged.
        """
        with self._db.connection as conn:
            conn.execute(
                "UPDATE event_log SET entry_state = ?, closed_by_event_id = ? "
                "WHERE id = ?",
                (EntryState.CLOSED.value, exit_id, entry_id),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def find_open_entries(self, normalized_plate: str) -> list[EventRecord]:
        """Return Open_Entry_Records for an exact normalized-plate match.

        An Open_Entry_Record is an entry event (``event_kind == ENTRY``) that is
        still open (``entry_state == OPEN``). Results are ordered by timestamp
        then id so callers can apply the earliest-entry tie-break (Req 9.3).
        """
        cursor = self._db.connection.execute(
            f"SELECT {_COLUMNS} FROM event_log "
            "WHERE normalized_plate = ? AND entry_state = ? AND event_kind = ? "
            "ORDER BY timestamp, id",
            (normalized_plate, EntryState.OPEN.value, EventKind.ENTRY.value),
        )
        return [_row_to_record(row) for row in cursor.fetchall()]


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------
def _enum_value(member) -> Optional[str]:
    """Return an ``(str, Enum)`` member's ``.value``, or ``None`` when absent."""
    return member.value if member is not None else None


def _metric_value(value: "float | int | str") -> "float | int | str":
    """Persist a numeric metric as-is, or the literal ``"N/A"`` sentinel.

    The value is stored unchanged: a float/int goes into the (untyped) SQLite
    column directly, and the ``NA_SENTINEL`` string is stored verbatim so it
    round-trips as ``"N/A"`` (Req 15.5, 16.4).
    """
    return value


def _row_to_record(row) -> EventRecord:
    """Map a ``sqlite3.Row`` onto a fully-reconstructed :class:`EventRecord`.

    Enum columns are rebuilt from their stored ``.value`` (or ``None`` when the
    column is NULL); metric columns return their numeric value or the
    ``NA_SENTINEL`` string exactly as stored.
    """
    return EventRecord(
        id=row["id"],
        timestamp=row["timestamp"],
        ocr_plate=row["ocr_plate"],
        guard_plate=row["guard_plate"],
        normalized_plate=row["normalized_plate"],
        classification=_to_enum(Classification, row["classification"]),
        direction=_to_enum(DirectionOutcome, row["direction"]),
        grant_method=_to_enum(GrantMethod, row["grant_method"]),
        event_kind=_to_enum(EventKind, row["event_kind"]),
        entry_state=_to_enum(EntryState, row["entry_state"]),
        closed_by_event_id=row["closed_by_event_id"],
        image_ref=row["image_ref"],
        detection_confidence=_to_metric(row["detection_confidence"]),
        ocr_confidence=_to_metric(row["ocr_confidence"]),
        processing_latency_ms=_to_metric(row["processing_latency_ms"]),
        environment_label=_to_enum(EnvironmentLabel, row["environment_label"]),
    )


def _to_enum(enum_cls, value):
    """Reconstruct an enum member from its stored value, or ``None`` when NULL."""
    return enum_cls(value) if value is not None else None


def _to_metric(value: "float | int | str") -> "float | int | str":
    """Return a stored metric verbatim (numeric value or ``NA_SENTINEL``)."""
    return NA_SENTINEL if value == NA_SENTINEL else value
