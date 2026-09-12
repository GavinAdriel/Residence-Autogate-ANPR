"""Unit tests for :class:`~anpr.imaging.store.DiskImageStore` against the fake stub.

These example-based tests exercise the image store's database interactions
through the shared fake DB-API stub (``tests/persistence/fake_dbapi.py``) so they
run with **no live MySQL** (Req 10.1, 10.3). They pin the SQL contract that the
store now rides on top of the guarded MySQL :class:`~anpr.persistence.db.Database`
API (task 7.1):

* The image-reference insert uses ``%s`` placeholders and the MySQL upsert
  ``ON DUPLICATE KEY UPDATE`` (Req 1.2, 8.1) - it never issues the SQLite
  ``INSERT OR REPLACE`` or ``?`` placeholder.
* Retention reads via the guarded ``query`` (a ``SELECT ... FROM images``) and
  deletes via the guarded ``execute`` (repeated ``DELETE ... WHERE event_id = %s``)
  (Req 1.2, 8.1).
* No DDL (CREATE/ALTER/DROP/TRUNCATE) is ever issued (Req 1.3).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("pymysql")

from anpr.core.models import ImageRef
from anpr.imaging.store import DiskImageStore
from tests.persistence.fake_dbapi import fake_database

# DDL verbs the application must never issue: the schema is owned by the Docker
# init SQL, so the store never creates/alters/drops any schema object (Req 1.3).
_DDL_PATTERN = re.compile(r"\b(CREATE|ALTER|DROP|TRUNCATE|RENAME)\b", re.IGNORECASE)


def _assert_no_ddl(executed_sql: list[str]) -> None:
    for sql in executed_sql:
        assert not _DDL_PATTERN.search(sql), f"unexpected DDL statement: {sql!r}"


def _assert_no_sqlite_placeholders(executed_sql: list[str]) -> None:
    for sql in executed_sql:
        assert "?" not in sql, f"SQLite '?' placeholder found: {sql!r}"


# ----------------------------------------------------------------------
# _insert_reference: %s placeholders + ON DUPLICATE KEY UPDATE upsert
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_insert_reference_uses_placeholders_and_upsert(tmp_path) -> None:
    """The image-reference insert is a parameterized MySQL upsert (Req 1.2, 8.1)."""
    db, conn = fake_database()
    conn.queue_result(rowcount=1)  # the guarded write

    store = DiskImageStore(db=db, image_dir=str(tmp_path))
    ref = ImageRef(
        event_id="evt-1",
        snapshot_path=str(tmp_path / "evt-1.jpg"),
        thumbnail_path=str(tmp_path / "evt-1_thumb.jpg"),
        captured_at="2024-01-01T08:30:00+07:00",
    )

    store._insert_reference(ref)

    # Exactly one guarded write, routed through Database.execute (which commits).
    assert len(conn.executed) == 1
    assert conn.commits == 1

    sql = conn.executed[0].sql
    assert sql.lstrip().upper().startswith("INSERT INTO IMAGES")
    # MySQL upsert semantics replace the prior sqlite INSERT OR REPLACE (Req 1.2).
    assert "ON DUPLICATE KEY UPDATE" in sql.upper()
    assert "INSERT OR REPLACE" not in sql.upper()
    # One %s per inserted column and no SQLite '?' placeholder (Req 1.2).
    assert sql.count("%s") == 4
    assert "?" not in sql

    # Params are bound positionally in ImageRef field order.
    assert conn.executed[0].params == (
        "evt-1",
        str(tmp_path / "evt-1.jpg"),
        str(tmp_path / "evt-1_thumb.jpg"),
        "2024-01-01T08:30:00+07:00",
    )


@pytest.mark.unit
def test_insert_reference_issues_no_ddl(tmp_path) -> None:
    """The insert path never issues DDL (Req 1.3)."""
    db, conn = fake_database()
    conn.queue_result(rowcount=1)

    store = DiskImageStore(db=db, image_dir=str(tmp_path))
    store._insert_reference(
        ImageRef(
            event_id="evt-1",
            snapshot_path=str(tmp_path / "evt-1.jpg"),
            thumbnail_path=str(tmp_path / "evt-1_thumb.jpg"),
            captured_at="2024-01-01T08:30:00+07:00",
        )
    )

    _assert_no_ddl(conn.executed_sql)
    _assert_no_sqlite_placeholders(conn.executed_sql)


# ----------------------------------------------------------------------
# run_retention: SELECT via query, DELETE via execute, %s placeholders
# ----------------------------------------------------------------------
def _row(event_id: str, captured_at: datetime) -> dict:
    """A canned ``images`` retention row with an ISO-8601 capture time."""
    return {
        "event_id": event_id,
        "snapshot_path": f"/tmp/{event_id}.jpg",
        "thumbnail_path": f"/tmp/{event_id}_thumb.jpg",
        "captured_at": captured_at.isoformat(),
    }


@pytest.mark.unit
def test_run_retention_selects_then_deletes_expired_via_guarded_api(tmp_path) -> None:
    """Retention reads via ``query`` and deletes expired rows via ``execute`` (Req 1.2, 8.1)."""
    db, conn = fake_database()

    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    # One row well past the 30-day window (expired) and one within it (kept).
    expired = _row("evt-old", now - timedelta(days=100))
    fresh = _row("evt-new", now - timedelta(days=1))
    conn.queue_rows(expired, fresh)  # served to the retention SELECT
    conn.queue_result(rowcount=1)  # served to the single DELETE

    store = DiskImageStore(db=db, image_dir=str(tmp_path), retention_days=30)
    deleted = store.run_retention(now)

    assert deleted == 1

    # First statement is the read-only SELECT issued via Database.query.
    select_sql = conn.executed[0].sql
    assert select_sql.lstrip().upper().startswith("SELECT")
    assert "FROM images" in select_sql

    # The second statement is the parameterized DELETE issued via Database.execute.
    delete_stmt = conn.executed[1]
    assert delete_stmt.sql.strip() == "DELETE FROM images WHERE event_id = %s"
    assert delete_stmt.params == ("evt-old",)

    # Only the DELETE goes through the write path, so exactly one commit ran;
    # the SELECT (query) does not commit.
    assert conn.commits == 1
    _assert_no_sqlite_placeholders(conn.executed_sql)


@pytest.mark.unit
def test_run_retention_deletes_each_expired_row_individually(tmp_path) -> None:
    """Each expired row is deleted by its own guarded ``execute`` (Req 8.1)."""
    db, conn = fake_database()

    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    old_rows = [_row(f"evt-{i}", now - timedelta(days=100 + i)) for i in range(3)]
    conn.queue_rows(*old_rows)
    for _ in old_rows:
        conn.queue_result(rowcount=1)  # one canned result per DELETE

    store = DiskImageStore(db=db, image_dir=str(tmp_path), retention_days=30)
    deleted = store.run_retention(now)

    assert deleted == 3
    delete_stmts = [s for s in conn.executed if s.sql.strip().upper().startswith("DELETE")]
    assert len(delete_stmts) == 3
    assert [s.params for s in delete_stmts] == [("evt-0",), ("evt-1",), ("evt-2",)]
    for stmt in delete_stmts:
        assert stmt.sql.strip() == "DELETE FROM images WHERE event_id = %s"
    # One commit per DELETE (three guarded writes).
    assert conn.commits == 3


@pytest.mark.unit
def test_run_retention_no_expired_rows_issues_no_delete(tmp_path) -> None:
    """When nothing is expired only the SELECT runs and no DELETE is issued."""
    db, conn = fake_database()

    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    conn.queue_rows(_row("evt-new", now - timedelta(days=1)))

    store = DiskImageStore(db=db, image_dir=str(tmp_path), retention_days=30)
    deleted = store.run_retention(now)

    assert deleted == 0
    assert len(conn.executed) == 1  # the SELECT only
    assert conn.executed[0].sql.lstrip().upper().startswith("SELECT")
    assert conn.commits == 0  # no write path taken


@pytest.mark.unit
def test_run_retention_issues_no_ddl(tmp_path) -> None:
    """The retention path never issues DDL (Req 1.3)."""
    db, conn = fake_database()

    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    conn.queue_rows(_row("evt-old", now - timedelta(days=100)))
    conn.queue_result(rowcount=1)

    store = DiskImageStore(db=db, image_dir=str(tmp_path), retention_days=30)
    store.run_retention(now)

    _assert_no_ddl(conn.executed_sql)
    _assert_no_sqlite_placeholders(conn.executed_sql)
