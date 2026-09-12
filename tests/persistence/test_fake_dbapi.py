"""Unit tests for the fake DB-API stub and the ``Database`` factory seam.

These verify the shared test fixture behaves like the slice of PyMySQL the
persistence layer touches -- capturing SQL/params, serving canned dict rows,
recording call order -- and that :class:`anpr.persistence.db.Database` routes
through an injected fake connection with **no live MySQL** (Req 10.1, 10.3).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymysql")

from anpr.persistence.db import DatabaseError, Database
from tests.persistence.fake_dbapi import FakeConnection, fake_database


@pytest.mark.unit
def test_cursor_captures_sql_and_params() -> None:
    conn = FakeConnection()
    conn.queue_rows({"id": 1, "normalized_plate": "B1234XYZ"})

    with conn.cursor() as cur:
        cur.execute("SELECT * FROM residents WHERE normalized_plate = %s", ("B1234XYZ",))
        rows = cur.fetchall()

    assert rows == [{"id": 1, "normalized_plate": "B1234XYZ"}]
    assert conn.executed_sql == [
        "SELECT * FROM residents WHERE normalized_plate = %s"
    ]
    assert conn.executed[0].params == ("B1234XYZ",)


@pytest.mark.unit
def test_fetchone_returns_first_row_or_none() -> None:
    conn = FakeConnection()
    conn.queue_rows({"id": 1}, {"id": 2})
    conn.queue_result(rows=[])  # empty read for the second execute

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM residents", ())
        assert cur.fetchone() == {"id": 1}

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM residents WHERE id = %s", (99,))
        assert cur.fetchone() is None


@pytest.mark.unit
def test_rowcount_defaults_to_row_count_and_honors_override() -> None:
    conn = FakeConnection()
    conn.queue_rows({"a": 1}, {"a": 2})  # rowcount defaults to len(rows) == 2
    conn.queue_result(rowcount=1)  # write result: one affected row, no rows

    with conn.cursor() as cur:
        cur.execute("SELECT a FROM t", ())
        assert cur.rowcount == 2

    with conn.cursor() as cur:
        cur.execute("INSERT INTO t (a) VALUES (%s)", (5,))
        assert cur.rowcount == 1


@pytest.mark.unit
def test_database_uses_injected_fake_connection() -> None:
    """The factory seam opens the fake connection, not a real MySQL server."""
    db, conn = fake_database()

    # Eager connect used the injected factory and recorded the connect kwargs.
    assert conn.connect_kwargs is not None
    assert conn.connect_kwargs["host"] == "127.0.0.1"
    assert conn.connect_kwargs["db"] == "anpr"

    conn.queue_rows({"id": 7})
    rows = db.query("SELECT id FROM residents WHERE id = %s", (7,))

    assert rows == [{"id": 7}]
    # A read pings (reconnect=True) before executing and never commits.
    assert conn.pings == [True]
    assert [c[0] for c in conn.calls if c[0] in ("ping", "execute")] == [
        "ping",
        "execute",
    ]
    assert conn.commits == 0


@pytest.mark.unit
def test_database_execute_commits_and_reports_rowcount() -> None:
    db, conn = fake_database()
    conn.queue_result(rowcount=1)

    affected = db.execute("INSERT INTO event_log (id) VALUES (%s)", (1,))

    assert affected == 1
    assert conn.commits == 1
    assert conn.rollbacks == 0
    assert conn.executed_sql == ["INSERT INTO event_log (id) VALUES (%s)"]


@pytest.mark.unit
def test_database_execute_error_rolls_back() -> None:
    import pymysql

    db, conn = fake_database()
    conn.queue_error(pymysql.err.OperationalError(1213, "deadlock"))

    with pytest.raises(DatabaseError):
        db.execute("INSERT INTO event_log (id) VALUES (%s)", (1,))

    assert conn.rollbacks == 1
    assert conn.commits == 0


@pytest.mark.unit
def test_ping_failure_aborts_without_executing() -> None:
    import pymysql

    conn = FakeConnection(ping_error=pymysql.err.OperationalError(2006, "gone away"))
    db = Database(
        host="127.0.0.1",
        port=3306,
        name="anpr",
        user="anpr",
        password="pw",
        connect_factory=conn.as_factory(),
    )

    with pytest.raises(DatabaseError):
        db.query("SELECT 1", ())

    # Liveness check ran but no statement executed (Req 8.7).
    assert conn.pings == [True]
    assert conn.executed == []
