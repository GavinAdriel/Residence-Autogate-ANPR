"""Unit tests for the MySQL repositories against the fake DB-API stub.

These example-based tests exercise the two concrete repositories -
:class:`~anpr.persistence.resident_repo.MySqlResidentRepository` and
:class:`~anpr.persistence.event_log_repo.MySqlEventLogRepository` - through the
shared fake DB-API stub (``tests/persistence/fake_dbapi.py``) so they run with
**no live MySQL** (Req 10.1, 10.3).

Coverage, retargeted onto the ``anpr_system`` schema:

* ``find_by_plate`` single / none / empty-string / duplicate cases
  (Req 3.3, 3.5, 3.6, 3.7). The plate lives on ``Vehicle`` and the owner on
  ``Resident``, so the lookup is a join across both tables.
* The resident repo issues only read-only ``SELECT`` SQL (Req 3.4).
* ``append`` writes exactly one row using the full ``ANPR_Log`` column list and
  ``%s`` placeholders only (Req 1.2, 4.7), with the flat-record serialization
  (``Direction``/``Event_Kind`` NULL, ``Entry_State`` ``"N/A"``), and returns the
  generated ``Log_ID``.
* Absent metrics bind as SQL ``NULL`` rather than the ``"N/A"`` string, because
  the ``ANPR_Log`` metric columns are ``DECIMAL``.
* No DDL (CREATE/ALTER/DROP) is ever issued by either repository (Req 1.3).
"""

from __future__ import annotations

import re
from datetime import datetime

import pytest

pytest.importorskip("pymysql")

from anpr.core.models import (
    NA_SENTINEL,
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

# The full ``ANPR_Log`` column list, in insert order. ``Log_ID`` is absent
# because it is AUTO_INCREMENT and assigned by MySQL (Req 4.7).
_ANPR_LOG_COLUMNS = [
    "Inserted_Time",
    "Camera_ID",
    "License_Plate_Number",
    "Normalized_Plate",
    "Guard_Plate",
    "Vehicle_ID",
    "Classification",
    "Direction",
    "Event_Kind",
    "Grant_Method",
    "Entry_State",
    "Detection_Confidence",
    "OCR_Confidence",
    "Processing_Time_MS",
    "Environment_Label",
    "Image_Ref",
    "Closed_By_Log_ID",
]

# DDL verbs the application must never issue (Req 1.3).
_DDL_KEYWORDS = ("CREATE", "ALTER", "DROP", "TRUNCATE")


def _assert_no_ddl(executed_sql: list[str]) -> None:
    """Assert no statement issues a DDL verb (Req 1.3).

    Matches whole words only (via ``\\b`` boundaries) so incidental substrings
    such as ``Created_At`` do not trip the ``CREATE`` check.
    """
    for sql in executed_sql:
        upper = sql.upper()
        for keyword in _DDL_KEYWORDS:
            assert re.search(rf"\b{keyword}\b", upper) is None, (
                f"unexpected DDL '{keyword}' in: {sql}"
            )


def _resident_row(**overrides) -> dict:
    """A canned joined ``Vehicle``/``Resident`` row keyed by the query aliases."""
    row = {
        "id": 1,
        "normalized_plate": "B1234XYZ",
        "vehicle_id": 7,
        "license_plate": "B 1234 XYZ",
        "resident_name": "John Doe",
        "created_at": "2024-01-01T00:00:00+07:00",
        "updated_at": "2024-01-02T00:00:00+07:00",
    }
    row.update(overrides)
    return row


# ----------------------------------------------------------------------
# MySqlResidentRepository.find_by_plate
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_find_by_plate_single_match_returns_record() -> None:
    """A single matching row maps onto a ``ResidentRecord`` (Req 3.3)."""
    db, conn = fake_database()
    conn.queue_rows(_resident_row())

    record = MySqlResidentRepository(db).find_by_plate("B1234XYZ")

    assert record == ResidentRecord(
        id="1",  # Resident_ID arrives as an int and is coerced to str
        normalized_plate="B1234XYZ",
        created_at="2024-01-01T00:00:00+07:00",
        updated_at="2024-01-02T00:00:00+07:00",
        vehicle_id=7,
        resident_name="John Doe",
        license_plate="B 1234 XYZ",
    )
    # Bound via a parameterized SELECT using %s (Req 1.2).
    assert len(conn.executed) == 1
    assert conn.executed[0].params == ("B1234XYZ",)
    assert "%s" in conn.executed[0].sql


@pytest.mark.unit
def test_find_by_plate_joins_vehicle_to_resident() -> None:
    """The lookup joins ``Vehicle`` to ``Resident`` on ``Resident_ID``.

    The plate column exists only on ``Vehicle`` in this schema, so filtering on
    ``Resident`` alone is impossible; the join is what makes the whitelist
    lookup work at all.
    """
    db, conn = fake_database()
    conn.queue_rows(_resident_row())

    MySqlResidentRepository(db).find_by_plate("B1234XYZ")

    sql = conn.executed[0].sql
    assert "FROM Vehicle" in sql
    assert "JOIN Resident" in sql
    assert "r.Resident_ID = v.Resident_ID" in sql
    # The filter is applied to the Vehicle-side plate column.
    assert "v.Normalized_Plate = %s" in sql


@pytest.mark.unit
def test_find_by_plate_carries_vehicle_id_for_log_fk() -> None:
    """``Vehicle_ID`` is selected so it can populate ``ANPR_Log.Vehicle_ID``."""
    db, conn = fake_database()
    conn.queue_rows(_resident_row(vehicle_id=99))

    record = MySqlResidentRepository(db).find_by_plate("B1234XYZ")

    assert record is not None
    assert record.vehicle_id == 99
    assert "v.Vehicle_ID AS vehicle_id" in conn.executed[0].sql


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

    The fake returns rows in ``(Created_At, Vehicle_ID)`` order (as the
    ``ORDER BY ... LIMIT 1`` clause would), so the first row is the
    deterministic winner. The test asserts both the selection and the presence of
    the deterministic ordering clause in the issued SQL. Ordering is by the
    *vehicle* columns because the vehicle row is what carries the plate.
    """
    db, conn = fake_database()
    # Only the winning row would survive LIMIT 1; queue it as the single result.
    conn.queue_rows(
        _resident_row(
            id="res-early",
            normalized_plate="B1XYZ",
            created_at="2023-01-01T00:00:00+07:00",
            updated_at="2023-01-01T00:00:00+07:00",
        )
    )

    record = MySqlResidentRepository(db).find_by_plate("B1XYZ")

    assert record is not None
    assert record.id == "res-early"
    sql = conn.executed[0].sql
    assert "ORDER BY v.Created_At ASC, v.Vehicle_ID ASC LIMIT 1" in sql


@pytest.mark.unit
def test_resident_repo_issues_only_readonly_select_sql() -> None:
    """The resident repo issues read-only ``SELECT`` SQL only (Req 3.4, 1.3)."""
    db, conn = fake_database()
    conn.queue_result(rows=[])  # for the no-match lookup below
    conn.queue_rows(_resident_row())

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
def _sample_event_record(**overrides) -> EventRecord:
    """A representative flat match-and-log ``EventRecord``."""
    record = EventRecord(
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
        camera_id=1,
        vehicle_id=7,
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


@pytest.mark.unit
def test_append_writes_single_row_with_full_columns_and_placeholders() -> None:
    """``append`` issues exactly one INSERT with the full column list and %s (Req 1.2, 4.7)."""
    db, conn = fake_database()
    conn.queue_insert_id(123)

    MySqlEventLogRepository(db).append(_sample_event_record())

    # Exactly one write, and it is an INSERT into ANPR_Log (Req 4.7).
    assert len(conn.executed) == 1
    sql = conn.executed[0].sql
    assert sql.lstrip().upper().startswith("INSERT INTO ANPR_LOG")

    # The full column list is present, in insert order (Req 4.7).
    for column in _ANPR_LOG_COLUMNS:
        assert column in sql, f"missing column '{column}' in INSERT"

    # Log_ID is never supplied by the application: it is AUTO_INCREMENT.
    assert "Log_ID," not in sql
    assert "(Log_ID" not in sql

    # Exactly one %s per column, and no literal '?' style (Req 1.2).
    assert sql.count("%s") == len(_ANPR_LOG_COLUMNS) == 17
    assert "?" not in sql

    # The commit path ran (one affected row).
    assert conn.commits == 1


@pytest.mark.unit
def test_append_returns_and_stamps_generated_log_id() -> None:
    """``append`` returns the generated ``Log_ID`` and stamps it on the record.

    The id is required to attach the ``Images`` row, whose ``Log_ID`` foreign key
    is NOT NULL, so it must be surfaced rather than discarded.
    """
    db, conn = fake_database()
    conn.queue_insert_id(4242)

    record = _sample_event_record()
    returned = MySqlEventLogRepository(db).append(record)

    assert returned == 4242
    assert record.log_id == 4242


@pytest.mark.unit
def test_append_serializes_flat_record_fields() -> None:
    """The bound params carry the flat-record serialization (Req 4.7, 5.3, 5.4)."""
    db, conn = fake_database()
    conn.queue_insert_id(1)

    MySqlEventLogRepository(db).append(_sample_event_record())

    params = conn.executed[0].params
    assert len(params) == 17
    # Inserted_Time is a naive UTC datetime, not the offset-bearing string:
    # 08:30+07:00 is 01:30 UTC, and DATETIME has no offset component.
    assert params[0] == datetime(2024, 1, 1, 1, 30, 0)
    assert params[1] == 1  # Camera_ID (NOT NULL FK)
    assert params[2] == "B1234XYZ"  # License_Plate_Number (from ocr_plate)
    assert params[5] == 7  # Vehicle_ID (from the matched resident)
    # Enum members serialize to their .value; absent enums serialize to None.
    assert params[6] == "RESIDENT"  # Classification -> .value
    assert params[7] is None  # Direction NULL (flat record, Req 5.3)
    assert params[8] is None  # Event_Kind NULL (flat record, Req 5.3)
    assert params[9] == "AUTOMATIC"  # Grant_Method -> .value
    assert params[10] == "N/A"  # Entry_State -> EntryState.NA.value (Req 5.4)
    assert params[14] == "local-test"  # Environment_Label -> .value
    assert params[16] is None  # Closed_By_Log_ID (correlation retired, Req 5.2)


@pytest.mark.unit
def test_append_binds_absent_metrics_as_null_not_na_string() -> None:
    """An absent metric binds as SQL ``NULL``, never the ``"N/A"`` string.

    ``Detection_Confidence`` and ``OCR_Confidence`` are ``DECIMAL(5,4)`` and
    ``Processing_Time_MS`` is ``DECIMAL(10,2)``. Binding ``"N/A"`` to a DECIMAL
    column raises MySQL error 1366 under the default ``STRICT_TRANS_TABLES``,
    which would drop the whole event -- so the sentinel must be translated to
    ``NULL`` at the persistence boundary.
    """
    db, conn = fake_database()
    conn.queue_insert_id(1)

    MySqlEventLogRepository(db).append(
        _sample_event_record(
            detection_confidence=NA_SENTINEL,
            ocr_confidence=NA_SENTINEL,
            processing_latency_ms=NA_SENTINEL,
        )
    )

    params = conn.executed[0].params
    metric_params = params[11:14]
    assert metric_params == (None, None, None), (
        "Detection_Confidence, OCR_Confidence and Processing_Time_MS must bind "
        f"as NULL; got {metric_params!r}"
    )
    assert NA_SENTINEL not in metric_params
    # Entry_State legitimately keeps the "N/A" string: it is VARCHAR(32), not
    # DECIMAL, so the sentinel is a valid value there (Req 5.4).
    assert params[10] == NA_SENTINEL


@pytest.mark.unit
def test_append_preserves_numeric_metrics() -> None:
    """Present metrics are bound unchanged as numbers."""
    db, conn = fake_database()
    conn.queue_insert_id(1)

    MySqlEventLogRepository(db).append(_sample_event_record())

    params = conn.executed[0].params
    assert params[11] == 0.91  # Detection_Confidence
    assert params[12] == 0.88  # OCR_Confidence
    assert params[13] == 42  # Processing_Time_MS


@pytest.mark.unit
def test_append_issues_no_ddl() -> None:
    """``append`` never issues DDL (Req 1.3)."""
    db, conn = fake_database()
    conn.queue_insert_id(1)

    MySqlEventLogRepository(db).append(_sample_event_record())

    _assert_no_ddl(conn.executed_sql)
