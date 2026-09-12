"""Thin MySQL data-access helper shared by the persistence repositories.

This module owns the single concern of *how* the ANPR Autogate System talks to
its Docker-hosted MySQL 8.0 database: opening one PyMySQL connection, serializing
every access through a shared lock, performing a reconnecting liveness check
before each statement, and verifying that the externally-owned schema is present.
Both :class:`~anpr.persistence.resident_repo.MySqlResidentRepository` and
:class:`~anpr.persistence.event_log_repo.MySqlEventLogRepository` (and the
:class:`~anpr.imaging.store.DiskImageStore`) sit on top of this helper so that
connection handling, locking, and liveness live in exactly one place.

The connection settings are environment-specific and come from the configuration
keys ``database.{host,port,name,user,password}`` (Req 2.1); they are passed in by
the caller rather than read here, keeping this helper free of any coupling to the
``ConfigProvider``.

The schema is owned by the Docker init SQL (``docker/mysql/init/01-schema.sql``),
not by the application: this helper issues **no** DDL and never creates, alters,
or drops any schema object (Req 1.3). It only verifies that the required tables
exist (Req 1.5).

No cloud or internet access is performed: the only egress is the local MySQL
connection over the configured address (Req 1.6).
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import pymysql
import pymysql.cursors

# Connection timeout for the eager connect in ``__init__`` (Req 1.1).
CONNECT_TIMEOUT_S = 10

# Tables owned by the Docker init schema that must exist before startup
# proceeds (Req 1.5). The application never creates them.
REQUIRED_TABLES = ("residents", "event_log", "images")


class DatabaseError(RuntimeError):
    """Raised for query/connection faults reported to callers.

    The composition root maps this to a startup failure when raised during
    startup; at runtime it propagates to the calling component to indicate the
    operation did not execute (Req 8.7).
    """


class MissingTableError(RuntimeError):
    """Raised by :meth:`Database.check_required_tables` when a required table is
    absent from the MySQL database (Req 1.5)."""


class Database:
    """Manages a single MySQL connection shared across repositories.

    Parameters
    ----------
    host, port, name, user, password:
        MySQL connection settings from config ``database.*``. ``port`` is coerced
        to ``int`` and ``name`` maps to the PyMySQL ``db`` parameter.

    The connection is opened eagerly (Req 1.1, 9.2); a failure to connect raises
    :class:`ConnectionError` naming the configured host and port (Req 1.4). All
    access to the shared connection is serialized through ``DB_Access_Lock`` and
    guarded by a reconnecting liveness check (Req 8).

    ``connect_factory`` is a testing seam: it defaults to :func:`pymysql.connect`
    so production wiring opens a real MySQL connection, but tests may inject a
    fake DB-API stub (see ``tests/persistence/fake_dbapi.py``) so repository and
    helper tests run with no live MySQL (Req 10.1, 10.3).
    """

    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        user: str,
        password: str,
        connect_factory: Callable[..., Any] = pymysql.connect,
    ) -> None:
        self._params = dict(
            host=host,
            port=int(port),
            db=name,
            user=user,
            password=password,
            charset="utf8mb4",
            connect_timeout=CONNECT_TIMEOUT_S,
            cursorclass=pymysql.cursors.DictCursor,  # rows behave like mappings
        )
        # Factory seam (Req 10.1, 10.3): production passes ``pymysql.connect``;
        # tests inject a fake connection factory so no live MySQL is required.
        self._connect_factory = connect_factory
        # DB_Access_Lock: serializes all access to the single connection so that
        # at most one thread executes a database operation at any instant
        # (Req 8.1, 8.3). The default blocking acquire imposes no wait bound.
        self._lock = threading.Lock()
        self._conn = self._connect()  # eager connect (Req 1.1, 9.2)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def _connect(self) -> Any:
        """Open the connection via the factory, naming host/port on failure (Req 1.1, 1.4)."""
        try:
            return self._connect_factory(**self._params)
        except pymysql.MySQLError as exc:
            raise ConnectionError(
                f"Cannot connect to MySQL at {self._params['host']}:"
                f"{self._params['port']}: {exc}"
            ) from exc

    def _ping_or_raise(self) -> None:
        """Reconnecting liveness check before each operation (Req 8.6).

        On failure the operation is aborted; the caller's ``with self._lock``
        still releases the lock, and a :class:`DatabaseError` propagates to
        indicate the operation did not execute (Req 8.7).
        """
        try:
            self._conn.ping(reconnect=True)
        except pymysql.MySQLError as exc:
            raise DatabaseError(
                f"MySQL connection could not be re-established: {exc}"
            ) from exc

    @property
    def name(self) -> str:
        """Return the configured database name."""
        return self._params["db"]

    def close(self) -> None:
        """Close the underlying MySQL connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Guarded execute/query API (Req 8.1-8.7)
    # ------------------------------------------------------------------
    def execute(self, sql: str, params: tuple = ()) -> int:
        """Run a write (INSERT/UPDATE/DELETE); return the affected row count.

        Acquires ``DB_Access_Lock``, performs a reconnecting liveness check,
        executes the statement, commits on success, and rolls back on error.
        The lock is released on every path including error (Req 8.1-8.7). Uses
        ``%s`` placeholders only (Req 1.2).
        """
        with self._lock:  # Req 8.1, 8.3, 8.4
            self._ping_or_raise()  # Req 8.6, 8.7
            try:
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                    rowcount = cur.rowcount
                self._conn.commit()
                return rowcount
            except pymysql.MySQLError as exc:
                self._conn.rollback()
                raise DatabaseError(str(exc)) from exc
        # lock released by the `with` block even when the body raises (Req 8.5)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a read (SELECT); return all rows as dict mappings.

        Same lock / liveness / release policy as :meth:`execute` (Req 8.1-8.7).
        Uses ``%s`` placeholders only (Req 1.2).
        """
        with self._lock:  # Req 8.1, 8.3, 8.4
            self._ping_or_raise()  # Req 8.6, 8.7
            try:
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall()
            except pymysql.MySQLError as exc:
                raise DatabaseError(str(exc)) from exc
        # lock released by the `with` block even when the body raises (Req 8.5)

    # ------------------------------------------------------------------
    # Table-presence check (Req 1.5) - read-only, no DDL (Req 1.3)
    # ------------------------------------------------------------------
    def check_required_tables(self) -> None:
        """Verify each required table exists; raise naming the first missing one.

        Uses ``information_schema`` (a read, not DDL) so the application issues
        no CREATE/ALTER/DROP (Req 1.3, 1.5).
        """
        rows = self.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s",
            (self._params["db"],),
        )
        present = {r["table_name"] for r in rows if "table_name" in r}
        present |= {r["TABLE_NAME"] for r in rows if "TABLE_NAME" in r}
        for table in REQUIRED_TABLES:
            if table not in present:
                raise MissingTableError(
                    f"Required MySQL table '{table}' is absent from database "
                    f"'{self._params['db']}'."
                )
