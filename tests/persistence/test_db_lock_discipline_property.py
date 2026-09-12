"""Property-based test for the ``Database`` lock discipline.

Feature: mysql-monitoring-migration, Property 9
Property 9: Every database operation acquires and releases the lock.

For any database operation (``query`` or ``execute``) on the ``Database`` helper,
the ``DB_Access_Lock`` is held while the underlying statement executes and is
released afterward -- including when the statement raises, in which case the lock
is released before the error propagates so the connection never remains locked
(Req 8.1, 8.2, 8.4, 8.5).

The test instruments the lock so its held-state is observable, and uses a
lock-observing fake connection that snapshots the held-state at each ``ping`` and
each ``execute``. Sequences of read/write ops -- some of which raise -- are
generated with Hypothesis, and after every op the lock must be released while the
underlying statement must have run with the lock held.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("pymysql")
pytest.importorskip("hypothesis")

import pymysql
from hypothesis import given, settings
from hypothesis import strategies as st

from anpr.persistence.db import DatabaseError, Database
from tests.persistence.fake_dbapi import FakeConnection


class InstrumentedLock:
    """A ``threading.Lock`` wrapper that records whether it is currently held.

    Drop-in for the ``Database``'s ``DB_Access_Lock``: it supports the context
    manager protocol (``with self._lock:``) and the ``acquire``/``release`` API,
    exposing ``held`` so a collaborator can observe the held-state at any instant
    and ``acquire_count``/``release_count`` so the test can assert balance.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.held = False
        self.acquire_count = 0
        self.release_count = 0

    def acquire(self, *args, **kwargs) -> bool:
        acquired = self._lock.acquire(*args, **kwargs)
        if acquired:
            self.held = True
            self.acquire_count += 1
        return acquired

    def release(self) -> None:
        self.held = False
        self.release_count += 1
        self._lock.release()

    def __enter__(self) -> "InstrumentedLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


class LockObservingConnection(FakeConnection):
    """Fake connection that snapshots the lock held-state during ping/execute.

    Each ``ping`` and each ``execute`` records whether the instrumented lock was
    held at that instant, so the test can assert the underlying statement always
    ran while the lock was held.
    """

    def __init__(self, lock: InstrumentedLock, **kwargs) -> None:
        super().__init__(**kwargs)
        self._observed_lock = lock
        self.held_during_ping: list[bool] = []
        self.held_during_execute: list[bool] = []

    def ping(self, reconnect: bool = False) -> None:
        self.held_during_ping.append(self._observed_lock.held)
        super().ping(reconnect=reconnect)

    def _record_execute(self, sql, params) -> None:
        self.held_during_execute.append(self._observed_lock.held)
        super()._record_execute(sql, params)


def _build_instrumented_database() -> tuple[Database, LockObservingConnection, InstrumentedLock]:
    """Wire a ``Database`` onto an instrumented lock and observing connection.

    The lock is not touched during eager connect, so replacing ``db._lock`` after
    construction is safe and leaves the guarded execute/query paths using it.
    """
    lock = InstrumentedLock()
    conn = LockObservingConnection(lock)
    db = Database(
        host="127.0.0.1",
        port=3306,
        name="anpr",
        user="anpr",
        password="pw",
        connect_factory=conn.as_factory(),
    )
    db._lock = lock  # instrument the DB_Access_Lock (test seam)
    return db, conn, lock


# An op is ("read"|"write", raises?) -- reads/writes, some raising a driver error.
op_strategy = st.tuples(
    st.sampled_from(["read", "write"]),
    st.booleans(),
)


@pytest.mark.property
@settings(max_examples=200)
@given(ops=st.lists(op_strategy, min_size=1, max_size=25))
def test_lock_held_during_and_released_after_every_operation(ops) -> None:
    """Validates: Requirements 8.1, 8.2, 8.4, 8.5

    For every generated read/write op (raising or not): the lock is held while
    the underlying statement executes, and is released once the op returns or
    raises. No op leaves the lock held.
    """
    db, conn, lock = _build_instrumented_database()

    for kind, raises in ops:
        if raises:
            # The next execute raises a driver error; the guarded path must roll
            # back (writes), surface a DatabaseError, and still release the lock.
            conn.queue_error(pymysql.err.OperationalError(1213, "deadlock"))
        elif kind == "read":
            conn.queue_rows({"id": 1})
        else:
            conn.queue_result(rowcount=1)

        if raises:
            with pytest.raises(DatabaseError):
                if kind == "read":
                    db.query("SELECT id FROM residents WHERE id = %s", (1,))
                else:
                    db.execute("INSERT INTO event_log (id) VALUES (%s)", (1,))
        else:
            if kind == "read":
                db.query("SELECT id FROM residents WHERE id = %s", (1,))
            else:
                db.execute("INSERT INTO event_log (id) VALUES (%s)", (1,))

        # Req 8.4 / 8.5: the lock is released after the op on every path,
        # including when the statement raised.
        assert lock.held is False

    # Req 8.1 / 8.2: the underlying statement always ran with the lock held,
    # as did the reconnecting liveness check that guards it.
    assert conn.held_during_execute, "expected at least one executed statement"
    assert all(conn.held_during_execute)
    assert all(conn.held_during_ping)

    # Balanced acquire/release across the whole sequence (one pair per op).
    assert lock.acquire_count == len(ops)
    assert lock.release_count == len(ops)
