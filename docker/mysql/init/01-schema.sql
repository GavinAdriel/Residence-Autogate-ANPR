-- ANPR Autogate System - MySQL schema.

CREATE DATABASE IF NOT EXISTS anpr
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE anpr;

-- Resident_Database.
-- normalized_plate is intentionally NOT unique (duplicates are permitted;
-- the Admin_Dashboard reports the conflict). An index keeps lookups fast.
CREATE TABLE IF NOT EXISTS residents (
    id               VARCHAR(64)  NOT NULL,
    normalized_plate VARCHAR(32)  NOT NULL,
    created_at       VARCHAR(64)  NOT NULL,
    updated_at       VARCHAR(64)  NOT NULL,
    PRIMARY KEY (id),
    KEY idx_residents_normalized_plate (normalized_plate)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Event_Log.
-- closed_by_event_id is a self-referential FK to the exit event that closed an
-- Open_Entry_Record; stays NULL until closed.
CREATE TABLE IF NOT EXISTS event_log (
    id                     VARCHAR(64)  NOT NULL,
    timestamp              VARCHAR(64)  NOT NULL,
    ocr_plate              VARCHAR(32)  NOT NULL DEFAULT '',
    guard_plate            VARCHAR(32)  NOT NULL DEFAULT '',
    normalized_plate       VARCHAR(32)  NOT NULL DEFAULT '',
    classification         VARCHAR(32)  NULL,
    direction              VARCHAR(32)  NULL,
    grant_method           VARCHAR(32)  NOT NULL,
    event_kind             VARCHAR(32)  NULL,
    entry_state            VARCHAR(32)  NOT NULL,
    closed_by_event_id     VARCHAR(64)  NULL,
    image_ref              VARCHAR(512) NULL,
    detection_confidence   VARCHAR(32)  NOT NULL,
    ocr_confidence         VARCHAR(32)  NOT NULL,
    processing_latency_ms  VARCHAR(32)  NOT NULL,
    environment_label      VARCHAR(32)  NULL,
    PRIMARY KEY (id),
    KEY idx_event_log_open_entries (normalized_plate, entry_state, event_kind),
    KEY idx_event_log_closed_by (closed_by_event_id),
    CONSTRAINT fk_event_log_closed_by
        FOREIGN KEY (closed_by_event_id) REFERENCES event_log (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Image_Store references (files live on disk; DB stores paths only).
CREATE TABLE IF NOT EXISTS images (
    event_id       VARCHAR(64)  NOT NULL,
    snapshot_path  VARCHAR(512) NOT NULL,
    thumbnail_path VARCHAR(512) NOT NULL,
    captured_at    VARCHAR(64)  NOT NULL,
    PRIMARY KEY (event_id),
    KEY idx_images_captured_at (captured_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
