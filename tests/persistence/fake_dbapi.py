"""Fake DB-API stub for persistence tests (no live MySQL).

This module models the tiny slice of the PyMySQL surface that
:class:`anpr.persistence.db.Database` (and the repositories / image store built
on top of it) actually touch, so that repository and helper tests run with **no
live MySQL server** (Req 10.1, 10.3; Design: Testing Strategy "Fake DB-API
stub"). It is a shared test fixture used by the persistence, access-controller,
and image-store test suites.

The two stubs mirror the real driver:

* :class:`FakeConnection` -- ``ping(reconnect=...)``, ``cursor()``, ``commit()``,
  ``rollback()`` (plus ``close()``), matching how the ``Database`` helper drives
  a connection.
* :class:`FakeCursor` -- ``execute(sql, params)``, ``fetchall()``, ``fetchone()``,
  and a ``rowcount`` attribute, usable as a context manager exactly like a real
  PyMySQL cursor (``with conn.cursor() as cur:``).

Every executed SQL string and its params are captured on the connection so tests
can assert ``%s`` placeholders (Req 1.2) and that no DDL is ever issued
(Req 1.3). Reads are driven by *canned* dict rows queued on the connection, and
an ordered ``calls`` log records the ping/execute/commit/rollback sequence so
tests can assert the reconnecting-liveness-before-operation discipline (Req 8.6)
and lock/error-path behavior (Req 8.5, 8.7).

The connection doubles as the ``connect_factory`` seam on ``Database``: pass
``connect_factory=fake.as_factory()`` (or use :func:`fake_database`) so the
``Database`` opens this fake instead of a real MySQL connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ExecutedStatement:
    """A single captured ``execute`` call: the raw SQL and its bound params."""

    sql: str
    params: Any


@dataclass
class CannedResult:
    """A canned result served to one ``execute`` call.

    ``rows`` are the dict rows returned by ``fetchall``/``fetchone``; ``rowcount``
    is what the cursor reports (defaults to ``len(rows)`` when not given, so a
    read result needs no explicit count while a write result can set it, e.g.
    ``rowcount=1`` for a single-row INSERT).
    """

    rows: list[dict] = field(default_factory=list)
    rowcount: Optional[int] = None

    def resolved_rowcount(self) -> int:
        return self.rowcount if self.rowcount is not None else len(self.rows)


class FakeCursor:
    """Minimal stand-in for a PyMySQL ``DictCursor``.

    Captures each ``execute`` on the owning connection, serves the next canned
    result, and exposes ``fetchall``/``fetchone``/``rowcount``. Supports the
    context-manager protocol so ``with conn.cursor() as cur:`` works unchanged.
    """

    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection
        self._rows: list[dict] = []
        self.rowcount: int = -1

    def execute(self, sql: str, params: Any = None) -> int:
        """Record the statement, then serve the next canned result.

        If the connection has a queued error for this position, that exception is
        raised (after the statement is still recorded) so tests can drive the
        write-error rollback path and the lock-release-on-error property.
        """
        self._connection._record_execute(sql, params)
        result = self._connection._next_result()
        if isinstance(result, BaseException):
            raise result
        self._rows = list(result.rows)
        self.rowcount = result.resolved_rowcount()
        return self.rowcount

    def fetchall(self) -> list[dict]:
        return list(self._rows)

    def fetchone(self) -> Optional[dict]:
        return self._rows[0] if self._rows else None

    def close(self) -> None:
        self._connection.calls.append(("cursor_close", None))

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class FakeConnection:
    """Fake PyMySQL connection capturing all traffic and serving canned rows.

    Parameters
    ----------
    ping_error:
        Optional exception raised by :meth:`ping` to drive the failed-liveness
        path (the operation must abort without executing, Req 8.7).

    Queue canned results with :meth:`queue_result` / :meth:`queue_rows` /
    :meth:`queue_error` (consumed FIFO, one per ``execute``). When the queue is
    empty, ``execute`` serves an empty read with ``rowcount == 0``.
    """

    def __init__(self, ping_error: Optional[BaseException] = None) -> None:
        # Captured traffic (Req 1.2, 1.3): every executed statement + params.
        self.executed: list[ExecutedStatement] = []
        # Ordered call log for sequencing asserts (Req 8.6): ("ping"|"execute"|
        # "commit"|"rollback"|"cursor_close", detail).
        self.calls: list[tuple[str, Any]] = []
        self.pings: list[bool] = []
        self.commits: int = 0
        self.rollbacks: int = 0
        self.closed: bool = False
        # Connect kwargs recorded when used as a ``connect_factory`` seam.
        self.connect_kwargs: Optional[dict] = None
        self._ping_error = ping_error
        self._results: list[Any] = []  # FIFO of CannedResult | BaseException

    # ------------------------------------------------------------------
    # Result programming
    # ------------------------------------------------------------------
    def queue_result(
        self, rows: Optional[list[dict]] = None, rowcount: Optional[int] = None
    ) -> "FakeConnection":
        """Queue one canned result (returns self for chaining)."""
        self._results.append(CannedResult(rows=list(rows or []), rowcount=rowcount))
        return self

    def queue_rows(self, *rows: dict) -> "FakeConnection":
        """Queue one read result made of the given dict rows."""
        return self.queue_result(rows=list(rows))

    def queue_error(self, exc: BaseException) -> "FakeConnection":
        """Queue an exception to be raised by the next ``execute``."""
        self._results.append(exc)
        return self

    # ------------------------------------------------------------------
    # PyMySQL connection surface
    # ------------------------------------------------------------------
    def ping(self, reconnect: bool = False) -> None:
        self.calls.append(("ping", reconnect))
        self.pings.append(reconnect)
        if self._ping_error is not None:
            raise self._ping_error

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.calls.append(("commit", None))
        self.commits += 1

    def rollback(self) -> None:
        self.calls.append(("rollback", None))
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True

    # ------------------------------------------------------------------
    # Factory seam helper
    # ------------------------------------------------------------------
    def as_factory(self) -> Callable[..., "FakeConnection"]:
        """Return a ``connect_factory`` that yields this connection.

        The returned callable records the connection kwargs the ``Database``
        would have passed to ``pymysql.connect`` so tests can assert them.
        """

        def _factory(**kwargs: Any) -> "FakeConnection":
            self.connect_kwargs = dict(kwargs)
            return self

        return _factory

    # ------------------------------------------------------------------
    # Internal hooks used by FakeCursor
    # ------------------------------------------------------------------
    def _record_execute(self, sql: str, params: Any) -> None:
        self.calls.append(("execute", sql))
        self.executed.append(ExecutedStatement(sql=sql, params=params))

    def _next_result(self) -> Any:
        if self._results:
            return self._results.pop(0)
        return CannedResult(rows=[], rowcount=0)

    # ------------------------------------------------------------------
    # Convenience assertions for tests
    # ------------------------------------------------------------------
    @property
    def executed_sql(self) -> list[str]:
        """The captured SQL strings, in execution order."""
        return [stmt.sql for stmt in self.executed]


def fake_database(
    connection: Optional[FakeConnection] = None,
    *,
    host: str = "127.0.0.1",
    port: int = 3306,
    name: str = "anpr",
    user: str = "anpr",
    password: str = "anprpassword",
):
    """Build a ``Database`` wired to a :class:`FakeConnection` (no live MySQL).

    Returns ``(database, connection)``. Imported lazily so this helper module has
    no import-time dependency on a working ``pymysql`` install beyond what the
    ``Database`` module itself already requires.
    """
    from anpr.persistence.db import Database

    conn = connection if connection is not None else FakeConnection()
    db = Database(
        host=host,
        port=port,
        name=name,
        user=user,
        password=password,
        connect_factory=conn.as_factory(),
    )
    return db, conn
