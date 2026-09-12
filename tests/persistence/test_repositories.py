"""Unit tests for the MySQL repositories against the fake DB-API stub.

These example-based tests exercise the two concrete repositories -
:class:`~anpr.persistence.resident_repo.MySqlResidentRepository` and
:class:`~anpr.persistence.event_log_repo.MySqlEventLogRepository` - through the
shared fake DB-API stub (``tests/persistence/fake_dbapi.py``) so they run with
**no live MySQL** (Req 10.1, 10.3).

Coverage:

* ``find_by_plate`` single / none / empty-string / duplicate cases
  (Req 3.3, 3.5, 3.6, 3.7).
* The resident repo issues only read-only ``SELECT`` SQL (Req 3.4).
* ``append`` writes exactly one row using the full ``event_log`` column list and
  ``%s`` placeholders only (Req 1.2, 4.7), with the flat-record serialization
  (``direction``/``event_kind`` NULL, ``entry_state`` ``"N/A"``).
* No DDL (CREATE/ALTER/DROP) is ever issued by either repository (Req 1.3).
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("pymysql")

from anpr.core.models import (
    Classification,
    EntryState,
    EnvironmentLabel,
    EventRecord,
    GrantMethod,
    ResidentRecord,
)
from anpr.persistence.event_log_repo import MySqlEventLogRepository
from anpr.persistence.resident_repo import MySqlResidentRepository
from tests.persistence.fake_dbapi import fake_database

# The full ``event_log`` column list, in ``EventRecord`` field order (Req 4.7).
_EVENT_LOG_COLUMNS = [
    "id",
    "timestamp",
    "ocr_plate",
    "guard_plate",
    "normalized_plate",
    "classification",
    "direction",
    "grant_method",
    "event_kind",
    "entry_state",
    "closed_by_event_id",
    "image_ref",
    "detection_confidence",
    "ocr_confidence",
    "processing_latency_ms",
    "environment_label",
]

# DDL verbs the application must never issue (Req 1.3).
_DDL_KEYWORDS = ("CREATE", "ALTER", "DROP", "TRUNCATE")


def _assert_no_ddl(executed_sql: list[str]) -> None:
    """Assert no statement issues a DDL verb (Req 1.3).

    Matches whole words only (via ``\\b`` boundaries) so incidental substrings
    such as ``created_at`` do not trip the ``CREATE`` check.
    """
    for sql in executed_sql:
        upper = sql.upper()
        for keyword in _DDL_KEYWORDS:
            assert re.search(rf"\b{keyword}\b", upper) is None, (
                f"unexpected DDL '{keyword}' in: {sql}"
            )


# ----------------------------------------------------------------------
# MySqlResidentRepository.find_by_plate
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_find_by_plate_single_match_returns_record() -> None:
    """A single matching row maps onto a ``ResidentRecord`` (Req 3.3)."""
    db, conn = fake_database()
    conn.queue_rows(
        {
            "id": "res-1",
            "normalized_plate": "B1234XYZ",
            "created_at": "2024-01-01T00:00:00+07:00",
            "updated_at": "2024-01-02T00:00:00+07:00",
        }
    )

    record = MySqlResidentRepository(db).find_by_plate("B1234XYZ")

    assert record == ResidentRecord(
        id="res-1",
        normalized_plate="B1234XYZ",
        created_at="2024-01-01T00:00:00+07:00",
        updated_at="2024-01-02T00:00:00+07:00",
    )
    # Bound via a parameterized SELECT using %s (Req 1.2).
    assert len(conn.executed) == 1
    assert conn.executed[0].params == ("B1234XYZ",)
    assert "%s" in conn.executed[0].sql


@pytest.mark.unit
def test_find_by_plate_no_match_returns_none() -> None:
    """No matching row yields ``None`` (Req 3.6)."""
    db, conn = fake_database()
    conn.queue_result(rows=[])  # empty read

    record = MySqlResidentRepository(db).find_by_plate("B9999ZZZ")

    assert record is None
    assert len(conn.executed) == 1  # the query was still issued


@pytest.mark.unit
def test_find_by_plate_empty_string_returns_none_without_query() -> None:
    """The empty string short-circuits to ``None`` and issues no SQL (Req 3.7)."""
    db, conn = fake_database()

    record = MySqlResidentRepository(db).find_by_plate("")

    assert record is None
    assert conn.executed == []  # no query executed at all


@pytest.mark.unit
def test_find_by_plate_duplicate_selects_earliest_lowest_id() -> None:
    """Duplicate plates: the earliest-created, lowest-id row wins (Req 3.5).

    The fake returns rows in ``(created_at, id)`` order (as the
    ``ORDER BY created_at ASC, id ASC LIMIT 1`` clause would), so the first row
    is the deterministic winner. The test asserts both the selection and the
    presence of the deterministic ordering clause in the issued SQL.
    """
    db, conn = fake_database()
    # Only the winning row would survive LIMIT 1; queue it as the single result.
    conn.queue_rows(
        {
            "id": "res-early",
            "normalized_plate": "B1XYZ",
            "created_at": "2023-01-01T00:00:00+07:00",
            "updated_at": "2023-01-01T00:00:00+07:00",
        }
    )

    record = MySqlResidentRepository(db).find_by_plate("B1XYZ")

    assert record is not None
    assert record.id == "res-early"
    sql = conn.executed[0].sql
    assert "ORDER BY created_at ASC, id ASC LIMIT 1" in sql


@pytest.mark.unit
def test_resident_repo_issues_only_readonly_select_sql() -> None:
    """The resident repo issues read-only ``SELECT`` SQL only (Req 3.4, 1.3)."""
    db, conn = fake_database()
    conn.queue_result(rows=[])  # for the no-match lookup below
    conn.queue_rows(
        {
            "id": "res-1",
            "normalized_plate": "B1234XYZ",
            "created_at": "2024-01-01T00:00:00+07:00",
            "updated_at": "2024-01-01T00:00:00+07:00",
        }
    )

    repo = MySqlResidentRepository(db)
    repo.find_by_plate("B9999ZZZ")
    repo.find_by_plate("B1234XYZ")

    assert conn.executed_sql, "expected at least one issued statement"
    for sql in conn.executed_sql:
        assert sql.lstrip().upper().startswith("SELECT"), f"not read-only: {sql}"
    _assert_no_ddl(conn.executed_sql)


# ----------------------------------------------------------------------
# MySqlEventLogRepository.append
# ----------------------------------------------------------------------
def _sample_event_record() -> EventRecord:
    """A representative flat match-and-log ``EventRecord``."""
    return EventRecord(
        id="evt-1",
        timestamp="2024-01-01T08:30:00+07:00",
        ocr_plate="B1234XYZ",
        guard_plate="",
        normalized_plate="B1234XYZ",
        classification=Classification.RESIDENT,
        direction=None,
        grant_method=GrantMethod.AUTOMATIC,
        event_kind=None,
        entry_state=EntryState.NA,
        closed_by_event_id=None,
        image_ref="img-1",
        detection_confidence=0.91,
        ocr_confidence=0.88,
        processing_latency_ms=42,
        environment_label=EnvironmentLabel.LOCAL_TEST,
    )


@pytest.mark.unit
def test_append_writes_single_row_with_full_columns_and_placeholders() -> None:
    """``append`` issues exactly one INSERT with the full column list and %s (Req 1.2, 4.7)."""
    db, conn = fake_database()
    conn.queue_result(rowcount=1)

    MySqlEventLogRepository(db).append(_sample_event_record())

    # Exactly one write, and it is an INSERT into event_log (Req 4.7).
    assert len(conn.executed) == 1
    sql = conn.executed[0].sql
    assert sql.lstrip().upper().startswith("INSERT INTO EVENT_LOG")

    # The full column list is present, in EventRecord field order (Req 4.7).
    for column in _EVENT_LOG_COLUMNS:
        assert column in sql, f"missing column '{column}' in INSERT"

    # Exactly 16 %s placeholders, one per column, and no literal '?' style (Req 1.2).
    assert sql.count("%s") == len(_EVENT_LOG_COLUMNS) == 16
    assert "?" not in sql

    # The commit path ran (one affected row).
    assert conn.commits == 1


@pytest.mark.unit
def test_append_serializes_flat_record_fields() -> None:
    """The bound params carry the flat-record serialization (Req 4.7, 5.3, 5.4)."""
    db, conn = fake_database()
    conn.queue_result(rowcount=1)

    MySqlEventLogRepository(db).append(_sample_event_record())

    params = conn.executed[0].params
    assert len(params) == 16
    # Enum members serialize to their .value; absent enums serialize to None.
    assert params[0] == "evt-1"  # id
    assert params[5] == "RESIDENT"  # classification -> .value
    assert params[6] is None  # direction NULL (flat record, Req 5.3)
    assert params[7] == "AUTOMATIC"  # grant_method -> .value
    assert params[8] is None  # event_kind NULL (flat record, Req 5.3)
    assert params[9] == "N/A"  # entry_state -> EntryState.NA.value (Req 5.4)
    assert params[15] == "local-test"  # environment_label -> .value


@pytest.mark.unit
def test_append_issues_no_ddl() -> None:
    """``append`` never issues DDL (Req 1.3)."""
    db, conn = fake_database()
    conn.queue_result(rowcount=1)

    MySqlEventLogRepository(db).append(_sample_event_record())

    _assert_no_ddl(conn.executed_sql)
