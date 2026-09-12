"""Property-based test for the reconnecting-liveness-before-operation discipline.

Feature: mysql-monitoring-migration, Property 10
Property 10: Reconnecting liveness check precedes every operation.

Requirement 8.6 states the ``Database`` helper SHALL perform a reconnecting
liveness check (``ping(reconnect=True)``) before every statement it executes, so
a dropped connection is transparently re-established before any read or write.

This test drives an arbitrary sequence of ``query``/``execute`` operations
against a :class:`FakeConnection` (no live MySQL) whose ordered ``calls`` log
records every ``ping`` and ``execute`` in the order they happened. It then
asserts that in that recorded order every executed statement is immediately
preceded by a ``ping(reconnect=True)`` -- for every generated operation, no more
and no fewer.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymysql")
pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.persistence.fake_dbapi import fake_database

# One generated operation is either a read (``query``) or a write (``execute``).
_operation = st.sampled_from(["query", "execute"])


@pytest.mark.property
@settings(max_examples=200)
@given(operations=st.lists(_operation, min_size=1, max_size=20))
def test_ping_reconnect_precedes_every_operation(operations) -> None:
    """Validates: Requirements 8.6

    For any sequence of read/write operations, every executed statement in the
    recorded call order is immediately preceded by ``ping(reconnect=True)``.
    """
    db, conn = fake_database()

    for op in operations:
        if op == "query":
            # A read serves canned rows; execute() drives the write path.
            conn.queue_rows({"id": 1})
            db.query("SELECT id FROM residents WHERE id = %s", (1,))
        else:
            conn.queue_result(rowcount=1)
            db.execute("INSERT INTO event_log (id) VALUES (%s)", (1,))

    # Keep only the ping/execute events, preserving their recorded order.
    ping_and_exec = [c for c in conn.calls if c[0] in ("ping", "execute")]

    # Exactly one ping and one execute per generated operation.
    assert conn.pings == [True] * len(operations), (
        "every operation must ping with reconnect=True; "
        f"recorded pings: {conn.pings}"
    )
    assert [c[0] for c in ping_and_exec].count("execute") == len(operations)

    # Every execute is immediately preceded by a reconnecting liveness check.
    for i, (kind, _detail) in enumerate(ping_and_exec):
        if kind == "execute":
            assert i > 0, "an execute occurred with no preceding ping"
            prev_kind, prev_detail = ping_and_exec[i - 1]
            assert (prev_kind, prev_detail) == ("ping", True), (
                "execute was not immediately preceded by ping(reconnect=True); "
                f"preceding call was {ping_and_exec[i - 1]!r}\n"
                f"full ping/execute order: {ping_and_exec}"
            )
