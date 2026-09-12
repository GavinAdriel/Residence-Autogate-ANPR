"""MySQL-backed implementation of the ``ResidentRepository`` interface.

``MySqlResidentRepository`` provides read-only access to the ``residents`` table
(Req 3.1, 3.2). It exposes exactly one operation - ``find_by_plate`` - and no
create/update/delete/list_all writes: the resident whitelist is managed
externally through phpMyAdmin (Req 3.4, 6.5), so the application never needs
write operations. Trimming the surface to a single read makes the read-only
guarantee structural rather than merely conventional.

Persistence goes through the shared :class:`~anpr.persistence.db.Database`
helper, which owns the single MySQL connection, the access lock, and the
reconnecting liveness check. Every statement uses PyMySQL ``%s`` placeholders
(Req 1.2) and the application issues no DDL (Req 1.3). The ``Database`` instance
is injected via the constructor, keeping this class free of any coupling to the
``ConfigProvider`` (Requirement 14.4).

This concrete class structurally satisfies the ``ResidentRepository`` Protocol
in ``anpr.core.interfaces``.
"""

from __future__ import annotations

from typing import Optional

from anpr.core.models import ResidentRecord
from anpr.persistence.db import Database

# Columns selected for every read, in the order the ``ResidentRecord`` fields
# are constructed below.
_COLUMNS = "id, normalized_plate, created_at, updated_at"


class MySqlResidentRepository:
    """Read-only access to the ``residents`` table backed by MySQL.

    Parameters
    ----------
    db:
        A shared :class:`~anpr.persistence.db.Database` instance. Sharing the
        instance lets the event-log repository and image store reuse the same
        connection, lock, and liveness check.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        """Return a resident record for an exact normalized-plate match.

        The empty string returns ``None`` without querying (Req 3.7). Otherwise
        an exact-match lookup is performed; because duplicate normalized plates
        are permitted, more than one record may match, and the earliest-created
        record wins with ties broken by the lowest ``id``
        (``ORDER BY created_at ASC, id ASC LIMIT 1``) for a deterministic result
        (Req 3.3, 3.5). Returns ``None`` when no record matches (Req 3.6).
        """
        if normalized_plate == "":
            return None
        rows = self._db.query(
            f"SELECT {_COLUMNS} FROM residents "
            "WHERE normalized_plate = %s "
            "ORDER BY created_at ASC, id ASC LIMIT 1",
            (normalized_plate,),
        )
        return _row_to_record(rows[0]) if rows else None


def _row_to_record(row) -> ResidentRecord:
    """Map a DictCursor row onto a :class:`ResidentRecord`."""
    return ResidentRecord(
        id=row["id"],
        normalized_plate=row["normalized_plate"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
