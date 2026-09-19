"""MySQL-backed implementation of the ``ResidentRepository`` interface.

``MySqlResidentRepository`` provides read-only access to the resident whitelist
(Req 3.1, 3.2). It exposes exactly one operation - ``find_by_plate`` - and no
create/update/delete/list_all writes: the resident whitelist is managed
externally through phpMyAdmin (Req 3.4, 6.5), so the application never needs
write operations. Trimming the surface to a single read makes the read-only
guarantee structural rather than merely conventional.

Schema note: in the ``anpr_system`` schema the whitelist is *normalized across
two tables*. ``Vehicle`` holds the plate (``Normalized_Plate``) and ``Resident``
holds the owner; there is no plate column on ``Resident`` at all. A plate lookup
therefore joins ``Vehicle`` to ``Resident`` on ``Resident_ID``. The join is an
INNER JOIN, which is safe because ``Vehicle.Resident_ID`` is NOT NULL with an FK
to ``Resident``: every vehicle is guaranteed an owner, so the join can never
drop a registered plate.

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

# Columns selected for every read, aliased to the ``ResidentRecord`` field names
# so the row mapping stays independent of the schema's Capitalized_Snake naming.
# ``v.Vehicle_ID`` is carried because the access controller writes it to the
# nullable ``ANPR_Log.Vehicle_ID`` FK, and ``r.Resident_Name`` so the monitoring
# views can show the owner without a second query.
_COLUMNS = (
    "r.Resident_ID AS id, "
    "v.Normalized_Plate AS normalized_plate, "
    "v.Vehicle_ID AS vehicle_id, "
    "v.License_Plate_Number AS license_plate, "
    "r.Resident_Name AS resident_name, "
    "r.Created_At AS created_at, "
    "r.Updated_At AS updated_at"
)


class MySqlResidentRepository:
    """Read-only whitelist lookup over ``Vehicle`` joined to ``Resident``.

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
            f"SELECT {_COLUMNS} "
            "FROM Vehicle AS v "
            "JOIN Resident AS r ON r.Resident_ID = v.Resident_ID "
            "WHERE v.Normalized_Plate = %s "
            "ORDER BY v.Created_At ASC, v.Vehicle_ID ASC LIMIT 1",
            (normalized_plate,),
        )
        return _row_to_record(rows[0]) if rows else None


def _row_to_record(row) -> ResidentRecord:
    """Map a DictCursor row onto a :class:`ResidentRecord`.

    ``id`` and the timestamps are coerced to ``str`` because MySQL returns
    ``Resident_ID`` as an ``int`` and ``Created_At``/``Updated_At`` as
    ``datetime`` objects, while :class:`ResidentRecord` declares them as strings.
    """
    return ResidentRecord(
        id=str(row["id"]),
        normalized_plate=row["normalized_plate"],
        created_at=_as_text(row["created_at"]),
        updated_at=_as_text(row["updated_at"]),
        vehicle_id=(
            int(row["vehicle_id"]) if row.get("vehicle_id") is not None else None
        ),
        resident_name=row.get("resident_name") or "",
        license_plate=row.get("license_plate") or "",
    )


def _as_text(value) -> str:
    """Render a MySQL ``DATETIME`` (or already-string) value as ISO-8601 text."""
    if value is None:
        return ""
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)
