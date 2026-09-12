"""Property-based test for resident lookup selection semantics.

Feature: mysql-monitoring-migration, Property 1
Property 1: Resident lookup selects the earliest-created, lowest-id match.

``MySqlResidentRepository.find_by_plate`` (Req 3.3, 3.5, 3.6, 3.7) returns:

* ``None`` for the empty string, without querying the database (Req 3.7);
* ``None`` when no resident row matches the queried plate (Req 3.6);
* otherwise the single matching resident, and because duplicate normalized
  plates are permitted the earliest-created row wins with ties broken by the
  lowest ``id`` (Req 3.3, 3.5).

The database performs the ``ORDER BY created_at ASC, id ASC LIMIT 1`` ordering,
so the fake DB-API stub (``tests/persistence/fake_dbapi.py``) is programmed to
return rows in exactly that order -- i.e. this test sorts the generated matching
rows by ``(created_at, id)`` and queues them, mirroring what the real SQL would
hand back. The property then asserts the repository maps the first such row (or
``None``) per the selection/None semantics above.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("pymysql")
pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from anpr.persistence.resident_repo import MySqlResidentRepository
from tests.persistence.fake_dbapi import fake_database

# A small pool of plates so duplicates arise naturally across generated rows.
_PLATE_POOL = ["B1234XYZ", "D5678AB", "F9012CD", "AA1BB", "Z0000ZZ"]
# A plate that is deliberately never inserted, to exercise the no-match path.
_ABSENT_PLATE = "N0MATCH99"
# A small pool of ISO-8601 timestamps; the small size forces ``created_at`` ties
# so the lowest-``id`` tiebreak (Req 3.5) is exercised.
_TS_POOL = [
    "2023-12-31T23:59:59+07:00",
    "2024-01-01T08:00:00+07:00",
    "2024-01-01T09:00:00+07:00",
    "2024-02-15T12:30:00+07:00",
]


@st.composite
def _resident_rows(draw) -> list[dict]:
    """Generate a set of resident rows with unique ids and duplicate plates."""
    ids = draw(
        st.lists(st.integers(min_value=1, max_value=10_000), unique=True, max_size=8)
    )
    rows: list[dict] = []
    for row_id in ids:
        created = draw(st.sampled_from(_TS_POOL))
        rows.append(
            {
                "id": row_id,
                "normalized_plate": draw(st.sampled_from(_PLATE_POOL)),
                "created_at": created,
                "updated_at": created,
            }
        )
    return rows


@pytest.mark.property
@settings(max_examples=200)
@given(
    rows=_resident_rows(),
    query=st.sampled_from(_PLATE_POOL + [_ABSENT_PLATE, ""]),
)
def test_find_by_plate_selects_earliest_lowest_id(rows, query) -> None:
    """Validates: Requirements 3.3, 3.5, 3.6, 3.7

    For any set of resident rows and any queried plate, ``find_by_plate``
    returns the earliest-created / lowest-id matching row, ``None`` on no match,
    and ``None`` on the empty string (without touching the database).
    """
    db, conn = fake_database()
    repo = MySqlResidentRepository(db)

    # Empty string short-circuits: None and no SQL is ever issued (Req 3.7).
    if query == "":
        assert repo.find_by_plate(query) is None
        assert conn.executed == [], "empty-string lookup must not query the DB"
        return

    # The DB does the ORDER BY + LIMIT 1, so model the fake by handing back the
    # matching rows in (created_at, id) order -- what the real SQL would return.
    matches = [r for r in rows if r["normalized_plate"] == query]
    ordered = sorted(matches, key=lambda r: (r["created_at"], r["id"]))
    conn.queue_rows(*ordered)

    result = repo.find_by_plate(query)

    if not ordered:
        # No matching row -> None (Req 3.6).
        assert result is None
    else:
        # The earliest-created / lowest-id row wins (Req 3.3, 3.5).
        expected = ordered[0]
        assert result is not None
        assert result.id == expected["id"]
        assert result.normalized_plate == query
        assert result.created_at == expected["created_at"]
        assert result.updated_at == expected["updated_at"]

    # Exactly one read-only SELECT with %s and the deterministic ordering, and
    # no DDL / write statement was issued for the lookup (Req 3.4 corollary).
    assert len(conn.executed) == 1
    sql = conn.executed[0].sql
    assert conn.executed[0].params == (query,)
    upper = sql.upper()
    assert upper.startswith("SELECT")
    assert "%S" in upper  # PyMySQL placeholder, uppercased here
    assert "ORDER BY CREATED_AT ASC, ID ASC" in upper
    assert "LIMIT 1" in upper
    # Word-boundary match so the ``updated_at`` column does not trip "UPDATE".
    for forbidden in ("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"):
        assert re.search(rf"\b{forbidden}\b", upper) is None
