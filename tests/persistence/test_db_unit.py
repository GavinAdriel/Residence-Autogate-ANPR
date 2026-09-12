"""Unit tests for the MySQL ``Database`` helper (``anpr/persistence/db.py``).

Example-based checks for the guarded data-access helper, complementing the
property-based lock/liveness suites. They run entirely against the shared fake
DB-API stub (``tests/persistence/fake_dbapi.py``) with **no live MySQL**
(Req 10.1, 10.3), and cover:

* Parameterization / schema safety -- every statement the helper issues uses the
  PyMySQL ``%s`` placeholder and never any DDL (Req 1.2, 1.3).
* Failed-liveness abort -- a failing reconnecting ``ping`` aborts the operation
  without executing any statement and still releases ``DB_Access_Lock`` so the
  connection is not left locked (Req 8.7).
* Blocking-lock semantics -- while one thread holds ``DB_Access_Lock`` mid-operation,
  a second thread requesting a database operation blocks until the lock is
  released, then proceeds (Req 8.3).
"""

from __future__ import annotations

import re
import threading

import pytest

pytest.importorskip("pymysql")

import pymysql

from anpr.persistence.db import DatabaseError, Database
from tests.persistence.fake_dbapi import FakeConnection, fake_database

# DDL verbs the application must never issue: the schema is owned by the Docker
# init SQL, so the helper never creates/alters/drops any schema object (Req 1.3).
_DDL_PATTERN = re.compile(
    r"\b(CREATE|ALTER|DROP|TRUNCATE|RENAME)\b", re.IGNORECASE
)


def _assert_no_ddl(conn: FakeConnection) -> None:
    for sql in conn.executed_sql:
        assert not _DDL_PATTERN.search(sql), f"DDL statement issued: {sql!r}"


def _assert_placeholders_are_pymysql(conn: FakeConnection) -> None:
    """Every parameterized statement uses ``%s`` and never the SQLite ``?``."""
    for stmt in conn.executed:
        assert "?" not in stmt.sql, f"SQLite '?' placeholder found: {stmt.sql!r}"
        # A statement carrying bound params must use the PyMySQL ``%s`` placeholder.
        if stmt.params:
            assert "%s" in stmt.sql, f"parameterized statement lacks %s: {stmt.sql!r}"


# ----------------------------------------------------------------------------
# Req 1.2 / 1.3: %s placeholders only, and no DDL is ever issued
# ----------------------------------------------------------------------------
@pytest.mark.unit
def test_guarded_api_uses_pymysql_placeholders_and_no_ddl() -> None:
    """Validates: Requirements 1.2, 1.3

    Reads and writes routed through the guarded API carry their SQL verbatim,
    using ``%s`` placeholders and never emitting any DDL keyword.
    """
    db, conn = fake_database()
    conn.queue_rows({"id": 1, "normalized_plate": "B1234XYZ"})  # for the read
    conn.queue_result(rowcount=1)  # for the write

    db.query(
        "SELECT id, normalized_plate FROM residents WHERE normalized_plate = %s",
        ("B1234XYZ",),
    )
    db.execute(
        "INSERT INTO event_log (normalized_plate) VALUES (%s)",
        ("B1234XYZ",),
    )

    _assert_placeholders_are_pymysql(conn)
    _assert_no_ddl(conn)


@pytest.mark.unit
def test_check_required_tables_reads_information_schema_without_ddl() -> None:
    """Validates: Requirements 1.2, 1.3

    The table-presence check queries ``information_schema`` (a read) with a ``%s``
    placeholder and issues no schema-mutating statement (Req 1.5's mechanism,
    proven here to stay read-only per Req 1.3).
    """
    db, conn = fake_database()
    conn.queue_rows(
        {"table_name": "residents"},
        {"table_name": "event_log"},
        {"table_name": "images"},
    )

    db.check_required_tables()  # all three present -> returns cleanly

    assert len(conn.executed) == 1
    stmt = conn.executed[0]
    assert stmt.sql.strip().upper().startswith("SELECT")
    assert "information_schema.tables" in stmt.sql.lower()
    _assert_placeholders_are_pymysql(conn)
    _assert_no_ddl(conn)


# ----------------------------------------------------------------------------
# Req 8.7: failed liveness check aborts without executing and releases the lock
# ----------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("op", ["query", "execute"])
def test_failed_ping_aborts_operation_without_executing(op: str) -> None:
    """Validates: Requirements 8.7

    A failing reconnecting liveness check raises a ``DatabaseError`` and aborts
    the operation: no statement is executed, and neither commit nor rollback is
    attempted on the connection.
    """
    conn = FakeConnection(
        ping_error=pymysql.err.OperationalError(2006, "server gone away")
    )
    db = Database(
        host="127.0.0.1",
        port=3306,
        name="anpr",
        user="anpr",
        password="pw",
        connect_factory=conn.as_factory(),
    )

    with pytest.raises(DatabaseError):
        if op == "query":
            db.query("SELECT 1 FROM residents WHERE id = %s", (1,))
        else:
            db.execute("INSERT INTO event_log (id) VALUES (%s)", (1,))

    # Liveness check ran, but the statement never executed (Req 8.7).
    assert conn.pings == [True]
    assert conn.executed == []
    assert conn.commits == 0
    assert conn.rollbacks == 0


@pytest.mark.unit
def test_failed_ping_releases_lock_so_next_operation_proceeds() -> None:
    """Validates: Requirements 8.7

    After a liveness-check failure aborts an operation, ``DB_Access_Lock`` is
    released: it can be acquired non-blockingly, and a subsequent operation
    (once the connection recovers) runs normally.
    """
    conn = FakeConnection(
        ping_error=pymysql.err.OperationalError(2006, "server gone away")
    )
    db = Database(
        host="127.0.0.1",
        port=3306,
        name="anpr",
        user="anpr",
        password="pw",
        connect_factory=conn.as_factory(),
    )

    with pytest.raises(DatabaseError):
        db.query("SELECT 1 FROM residents WHERE id = %s", (1,))

    # The lock must not be held after the aborted op: acquire it non-blockingly.
    acquired = db._lock.acquire(blocking=False)
    assert acquired is True, "DB_Access_Lock was left held after a failed ping"
    db._lock.release()

    # Once the connection recovers, the next operation proceeds normally.
    conn._ping_error = None
    conn.queue_rows({"id": 1})
    assert db.query("SELECT id FROM residents WHERE id = %s", (1,)) == [{"id": 1}]


# ----------------------------------------------------------------------------
# Req 8.3: blocking-lock semantics across two threads
# ----------------------------------------------------------------------------
class _BlockingConnection(FakeConnection):
    """Fake connection whose first ``execute`` blocks until explicitly released.

    Thread A runs an operation that stalls inside the first ``execute`` while
    holding ``DB_Access_Lock`` (the block happens after the statement is
    recorded), giving the test a deterministic window during which thread B must
    block on the lock.
    """

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self._entered = entered
        self._release = release
        self._blocked_once = False

    def _record_execute(self, sql, params) -> None:
        super()._record_execute(sql, params)
        if not self._blocked_once:
            self._blocked_once = True
            self._entered.set()  # signal: A holds the lock and is mid-operation
            # Hold here (lock still held) until the test releases A.
            assert self._release.wait(timeout=5.0), "release signal never arrived"


@pytest.mark.unit
def test_second_thread_blocks_until_lock_released() -> None:
    """Validates: Requirements 8.3

    While thread A holds ``DB_Access_Lock`` mid-operation, thread B's database
    operation blocks; once A releases the lock, B proceeds and completes.
    """
    entered = threading.Event()
    release = threading.Event()
    conn = _BlockingConnection(entered, release)
    db = Database(
        host="127.0.0.1",
        port=3306,
        name="anpr",
        user="anpr",
        password="pw",
        connect_factory=conn.as_factory(),
    )
    conn.queue_result(rowcount=1)  # thread A's write
    conn.queue_rows({"id": 1})  # thread B's read

    b_done = threading.Event()
    errors: list[BaseException] = []

    def run_a() -> None:
        try:
            db.execute("INSERT INTO event_log (id) VALUES (%s)", (1,))
        except BaseException as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    def run_b() -> None:
        try:
            db.query("SELECT id FROM residents WHERE id = %s", (1,))
            b_done.set()
        except BaseException as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    thread_a = threading.Thread(target=run_a)
    thread_a.start()
    # Wait until A holds the lock and is stalled inside its operation.
    assert entered.wait(timeout=5.0), "thread A never entered its operation"

    thread_b = threading.Thread(target=run_b)
    thread_b.start()

    # B must not complete while A holds the lock (no wait bound is imposed).
    assert not b_done.wait(timeout=0.3), "thread B did not block on the held lock"

    # Release A; the lock frees and B must now proceed to completion.
    release.set()
    thread_a.join(timeout=5.0)
    assert b_done.wait(timeout=5.0), "thread B did not proceed after lock release"
    thread_b.join(timeout=5.0)

    assert not errors, f"unexpected thread errors: {errors}"
    assert thread_a.is_alive() is False
    assert thread_b.is_alive() is False
