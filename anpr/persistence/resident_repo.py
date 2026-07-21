"""SQLite-backed implementation of the ``ResidentRepository`` interface.

``SqliteResidentRepository`` provides CRUD access to the ``Resident_Database``
(Requirement 13.1-13.4). It stores each record's normalized plate exactly as
supplied by the caller: the Admin_Dashboard applies the ``Plate_Normalizer``
and enforces the format/duplicate rules *before* calling this repository
(design Admin_Dashboard section, Req 13.5-13.7), so the repository itself is a
faithful, policy-free data store. In particular, duplicate normalized plates
are permitted (Req 13.6) - the table does not declare the column unique.

Persistence goes through the shared :class:`~anpr.persistence.db.Database`
helper, which owns the SQLite connection and table creation. The database
location comes from configuration (``database.location``) and is injected via
the constructor, keeping this class free of any coupling to the
``ConfigProvider`` (Requirement 14.4).

This concrete class structurally satisfies the ``ResidentRepository`` Protocol
in ``anpr.core.interfaces``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from anpr.core.models import ResidentRecord
from anpr.persistence.db import Database

# Columns selected for every read, in the order the ``ResidentRecord`` fields
# are constructed below.
_COLUMNS = "id, normalized_plate, created_at, updated_at"


class SqliteResidentRepository:
    """CRUD access to the ``Resident_Database`` backed by SQLite.

    Parameters
    ----------
    db:
        A shared :class:`~anpr.persistence.db.Database` instance, or a database
        location string. Passing a string is a convenience that constructs the
        ``Database`` internally; passing an instance lets the event-log
        repository share the same connection and schema.
    """

    def __init__(self, db: Database | str) -> None:
        self._db = db if isinstance(db, Database) else Database(db)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        """Return a resident record for an exact normalized-plate match.

        Because duplicate normalized plates are permitted (Req 13.6), more than
        one record may match; the earliest-created record is returned for a
        deterministic result. Returns ``None`` when no record matches.
        """
        cursor = self._db.connection.execute(
            f"SELECT {_COLUMNS} FROM residents "
            "WHERE normalized_plate = ? ORDER BY created_at, id LIMIT 1",
            (normalized_plate,),
        )
        row = cursor.fetchone()
        return _row_to_record(row) if row is not None else None

    def list_all(self) -> list[ResidentRecord]:
        """Return all resident records, ordered by creation time then id."""
        cursor = self._db.connection.execute(
            f"SELECT {_COLUMNS} FROM residents ORDER BY created_at, id"
        )
        return [_row_to_record(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def create(self, plate: str) -> ResidentRecord:
        """Create and return a new resident record for ``plate``.

        A UUID primary key and matching ``created_at``/``updated_at`` ISO-8601
        timestamps (with timezone) are generated for the new record.
        """
        now = _now_iso()
        record = ResidentRecord(
            id=str(uuid.uuid4()),
            normalized_plate=plate,
            created_at=now,
            updated_at=now,
        )
        with self._db.connection as conn:
            conn.execute(
                "INSERT INTO residents (id, normalized_plate, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    record.id,
                    record.normalized_plate,
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def update(self, id: str, plate: str) -> Optional[ResidentRecord]:
        """Update an existing record's plate; return ``None`` when absent.

        Leaves ``created_at`` untouched and refreshes ``updated_at``. When no
        record with ``id`` exists the database is left unchanged (Req 13.8).
        """
        updated_at = _now_iso()
        with self._db.connection as conn:
            cursor = conn.execute(
                "UPDATE residents SET normalized_plate = ?, updated_at = ? WHERE id = ?",
                (plate, updated_at, id),
            )
            if cursor.rowcount == 0:
                return None
        return self._get(id)

    def delete(self, id: str) -> bool:
        """Delete the record with ``id``; return ``True`` when one was removed.

        Returns ``False`` (leaving the database unchanged) when no matching
        record exists (Req 13.8).
        """
        with self._db.connection as conn:
            cursor = conn.execute("DELETE FROM residents WHERE id = ?", (id,))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get(self, id: str) -> Optional[ResidentRecord]:
        """Fetch a single record by primary key, or ``None`` when absent."""
        cursor = self._db.connection.execute(
            f"SELECT {_COLUMNS} FROM residents WHERE id = ?", (id,)
        )
        row = cursor.fetchone()
        return _row_to_record(row) if row is not None else None


def _now_iso() -> str:
    """Return the current time as an ISO-8601 string with a timezone offset."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_record(row) -> ResidentRecord:
    """Map a ``sqlite3.Row`` onto a :class:`ResidentRecord`."""
    return ResidentRecord(
        id=row["id"],
        normalized_plate=row["normalized_plate"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
