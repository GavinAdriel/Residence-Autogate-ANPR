"""Thin SQLite data-access helper shared by the persistence repositories.

This module owns the single concern of *how* the ANPR Autogate System talks to
its SQLite database file: opening the connection, applying pragmatic defaults,
and creating the tables that back the ``Resident_Database`` and (in a later
task) the ``Event_Log``. Both :class:`~anpr.persistence.resident_repo.
SqliteResidentRepository` and the event-log repository sit on top of this
helper so connection handling and schema creation live in exactly one place.

The database location is environment-specific and comes from the configuration
key ``database.location`` (Requirement 14.2); it is passed in by the caller
rather than read here, keeping this helper free of any coupling to the
``ConfigProvider``. The special value ``":memory:"`` is honoured for tests.

No cloud or network access is performed: persistence is a local SQLite file
(Requirement 14.3).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# DDL for the Resident_Database table.
#
# ``normalized_plate`` is intentionally *not* declared UNIQUE: the design
# permits duplicate normalized plates, with the Admin_Dashboard reporting the
# conflict while still allowing the submission to proceed (Req 13.6). An index
# keeps the exact-match lookup in ``find_by_plate`` efficient without enforcing
# uniqueness.
_RESIDENTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS residents (
    id              TEXT PRIMARY KEY,
    normalized_plate TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
)
"""

_RESIDENTS_PLATE_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_residents_normalized_plate
    ON residents (normalized_plate)
"""

# DDL for the Event_Log table (design Data Models -> EventRecord).
#
# Every field of the ``EventRecord`` schema is persisted so a future reporting
# dashboard has all it needs; absent values are stored explicitly as empty
# string, NULL, or the ``"N/A"`` sentinel rather than being omitted (Req 10.4,
# 10.8, 15.5, 16.4). SQLite's dynamic typing lets the metric columns hold either
# a numeric value or the literal ``"N/A"`` string.
#
# ``closed_by_event_id`` is a self-referential foreign key to the exit event
# that closed an Open_Entry_Record (Req 9.2, 9.3); it stays NULL until closed.
_EVENT_LOG_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS event_log (
    id                    TEXT PRIMARY KEY,
    timestamp             TEXT NOT NULL,
    ocr_plate             TEXT NOT NULL DEFAULT '',
    guard_plate           TEXT NOT NULL DEFAULT '',
    normalized_plate      TEXT NOT NULL DEFAULT '',
    classification        TEXT,
    direction             TEXT,
    grant_method          TEXT NOT NULL,
    event_kind            TEXT,
    entry_state           TEXT NOT NULL,
    closed_by_event_id    TEXT,
    image_ref             TEXT,
    detection_confidence            NOT NULL,
    ocr_confidence                  NOT NULL,
    processing_latency_ms           NOT NULL,
    environment_label     TEXT,
    FOREIGN KEY (closed_by_event_id) REFERENCES event_log (id)
)
"""

# Index supporting the Open_Entry_Record correlation query in
# ``find_open_entries`` (exact normalized-plate match on open entry events).
_EVENT_LOG_OPEN_ENTRY_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_event_log_open_entries
    ON event_log (normalized_plate, entry_state, event_kind)
"""

# DDL for the Image_Store reference table (design Data Models -> ImageRef).
#
# Requirement 11.3 mandates that image *files* live on disk while the database
# stores only *references* to those files (never the binaries themselves). Each
# row records the on-disk snapshot/thumbnail paths and the capture time for an
# event, so ``ImageStore.get_thumbnail`` can resolve a reference back to its
# thumbnail file and ``ImageStore.run_retention`` can select images older than
# the retention period (Req 11.6, 11.9).
_IMAGES_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS images (
    event_id        TEXT PRIMARY KEY,
    snapshot_path   TEXT NOT NULL,
    thumbnail_path  TEXT NOT NULL,
    captured_at     TEXT NOT NULL
)
"""

# Index supporting the retention query (select images older than a cutoff by
# capture time) in ``ImageStore.run_retention`` (Req 11.6).
_IMAGES_CAPTURED_AT_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_images_captured_at
    ON images (captured_at)
"""


class Database:
    """Manages a single SQLite connection and the shared table schema.

    Parameters
    ----------
    location:
        Filesystem path to the SQLite database file (from config
        ``database.location``), or ``":memory:"`` for an in-memory database.
        Parent directories for a file-based location are created on demand.

    The connection is opened eagerly and the schema is created idempotently so
    a freshly configured deployment starts with a valid, empty database.
    """

    def __init__(self, location: str) -> None:
        self._location = location
        self._conn = self._connect(location)
        self._initialize_schema()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def _connect(location: str) -> sqlite3.Connection:
        """Open the SQLite connection, creating parent dirs when needed."""
        if location != ":memory:":
            path = Path(location)
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(location)
        # Rows behave like mappings so repositories can read columns by name.
        conn.row_factory = sqlite3.Row
        # Enforce foreign-key constraints (used by the event-log repo).
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the live SQLite connection shared across repositories."""
        return self._conn

    @property
    def location(self) -> str:
        """Return the configured database location."""
        return self._location

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _initialize_schema(self) -> None:
        """Create all shared tables and indexes if they do not yet exist."""
        with self._conn:
            self._conn.execute(_RESIDENTS_TABLE_DDL)
            self._conn.execute(_RESIDENTS_PLATE_INDEX_DDL)
            self._conn.execute(_EVENT_LOG_TABLE_DDL)
            self._conn.execute(_EVENT_LOG_OPEN_ENTRY_INDEX_DDL)
            self._conn.execute(_IMAGES_TABLE_DDL)
            self._conn.execute(_IMAGES_CAPTURED_AT_INDEX_DDL)
