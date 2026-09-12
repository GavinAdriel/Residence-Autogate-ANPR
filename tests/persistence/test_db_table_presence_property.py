"""Property-based test for the required-table presence check.

Feature: mysql-monitoring-migration, Property 11
Property 11: Table-presence check fails exactly when a required table is absent.

Requirement 1.5 states that, at startup, IF any of the ``residents``,
``event_log``, or ``images`` tables is absent from the MySQL database THEN the
system SHALL log an error identifying the missing table and halt startup. The
``Database.check_required_tables()`` helper enforces this by querying
``information_schema`` (a read, never DDL) and raising :class:`MissingTableError`
naming the first missing required table.

This test drives ``check_required_tables`` against a :class:`FakeConnection`
(no live MySQL) whose canned ``information_schema`` rows are an arbitrary subset
of the three required tables plus arbitrary extra (non-required) table names. It
asserts the biconditional: the check raises ``MissingTableError`` naming a
missing required table **iff** at least one required table is absent, and returns
cleanly **iff** all three required tables are present -- regardless of any extra
tables that happen to exist.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymysql")
pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from anpr.persistence.db import REQUIRED_TABLES, MissingTableError
from tests.persistence.fake_dbapi import fake_database

# Extra, non-required table names that may coexist in the schema. Constrained to
# identifiers disjoint from the required set so they never accidentally satisfy a
# required-table check.
_extra_table = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1,
    max_size=12,
).filter(lambda name: name not in REQUIRED_TABLES)


@pytest.mark.property
@settings(max_examples=200)
@given(
    present_required=st.lists(
        st.sampled_from(REQUIRED_TABLES), unique=True, max_size=len(REQUIRED_TABLES)
    ),
    extras=st.lists(_extra_table, unique=True, max_size=5),
)
def test_check_required_tables_fails_iff_required_table_absent(
    present_required, extras
) -> None:
    """Validates: Requirements 1.5

    For any subset of the required tables reported present (plus arbitrary extra
    tables), ``check_required_tables`` raises ``MissingTableError`` naming a
    missing required table iff a required table is absent, and returns cleanly
    iff all three required tables are present.
    """
    present_set = set(present_required)
    missing = [t for t in REQUIRED_TABLES if t not in present_set]

    db, conn = fake_database()
    # Canned information_schema rows: the present required tables plus extras, in
    # the DictCursor shape the check reads (``table_name`` column).
    present_rows = [{"table_name": name} for name in list(present_set) + extras]
    conn.queue_rows(*present_rows)

    if missing:
        with pytest.raises(MissingTableError) as exc_info:
            db.check_required_tables()
        # The error names the first missing required table (Req 1.5).
        assert missing[0] in str(exc_info.value), (
            f"MissingTableError should name the first absent required table "
            f"{missing[0]!r}; message was: {exc_info.value}"
        )
        # It names an actually-absent required table, never a present one.
        named = [t for t in REQUIRED_TABLES if t in str(exc_info.value)]
        assert all(t in missing for t in named), (
            f"error named a present table; missing={missing}, named={named}"
        )
    else:
        # All three required tables present: returns cleanly (no raise).
        assert db.check_required_tables() is None
