"""Shared value objects and enumerations for the ANPR Autogate System.

These dataclasses and enums are the vocabulary shared across every component.
They are pure data containers with no behavior and no I/O, so they can be
imported freely by both the pure-core logic and the I/O adapters without
creating coupling to any concrete implementation (Requirement 14.4).

See .kiro/specs/anpr-autogate-system/design.md (Data Models and Components
sections) for the authoritative field definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Primitive type aliases
# ---------------------------------------------------------------------------

# A per-frame centroid point in pixel coordinates: (x, y).
Point = tuple[int, int]

# Sentinel used in the Event_Log for numeric metrics that are unavailable.
# Per the design data model, absent metric values are stored explicitly as the
# string "N/A" rather than being omitted (Req 10.4, 15.5, 16.4).
NA_SENTINEL = "N/A"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Classification(str, Enum):
    """Resident/guest classification produced by the Access_Controller."""

    RESIDENT = "RESIDENT"
    GUEST = "GUEST"


class DirectionOutcome(str, Enum):
    """Direction resolution outcome.

    In this phase the single-camera resolver only ever emits INBOUND or
    UNDETERMINED; OUTBOUND is retained for the deferred dual-camera / outbound
    trajectory work.
    """

    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"
    UNDETERMINED = "UNDETERMINED"


class GrantMethod(str, Enum):
    """How (or whether) a gate-open was granted for an access attempt."""

    AUTOMATIC = "AUTOMATIC"
    MANUAL = "MANUAL"
    NONE = "NONE"


class EventKind(str, Enum):
    """The kind of Event_Log record."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    MANUAL = "MANUAL"
    ANOMALY = "ANOMALY"
    REVIEW = "REVIEW"


class EntryState(str, Enum):
    """Open/closed tracking state for Open_Entry_Record correlation.

    NA represents the design's "N/A" state for records that are not entry
    records and therefore have no open/closed lifecycle.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    NA = "N/A"


class EnvironmentLabel(str, Enum):
    """Deployment environment label persisted with every event (Req 15.2)."""

    LOCAL_TEST = "local-test"
    FIELD = "field"


# ---------------------------------------------------------------------------
# Geometry / detection value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Detection:
    """A single detected vehicle produced by the YOLO26_Detector wrapper."""

    box: BoundingBox
    confidence: float  # inclusive 0.0..1.0
    track_id: Optional[int] = None  # persistent tracker id, when available


# ---------------------------------------------------------------------------
# OCR / normalization value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcrResult:
    """Result of reading a cropped plate region with the OCR_Engine."""

    text: Optional[str]  # None => no readable text
    confidence: float  # inclusive 0.0..1.0
    timed_out: bool = False


@dataclass(frozen=True)
class NormalizationResult:
    """Result of normalizing raw OCR text with the Plate_Normalizer."""

    normalized: str  # uppercase, alnum-only (may be empty)
    is_valid: bool  # conforms to Indonesian_Plate_Format
    raw: str  # original text retained
    reason: Optional[str] = None  # non-empty rejection reason when invalid


# ---------------------------------------------------------------------------
# Direction value objects
# ---------------------------------------------------------------------------


@dataclass
class TrackHistory:
    """Per-track motion history fed to the Direction_Resolver."""

    track_id: int
    centroids: list[Point] = field(default_factory=list)
    camera_id: Optional[str] = None
    confidence: float = 0.0  # 0.00..1.00


# ---------------------------------------------------------------------------
# Pipeline event object
# ---------------------------------------------------------------------------


@dataclass
class DetectionEvent:
    """Accumulator value object flowing through the pipeline.

    A single DetectionEvent is created when a detection is retained and is
    enriched as it passes through OCR -> normalization -> direction -> access
    control, until it is written as exactly one Event_Log record when it
    results in an access attempt. All fields are optional so the event can be
    built incrementally.
    """

    # Identity / timing
    event_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    acquired_at: Optional[datetime] = None  # frame acquisition time (t0)

    # Detection
    box: Optional[BoundingBox] = None
    detection_confidence: Optional[float] = None
    track_id: Optional[int] = None
    track: Optional[TrackHistory] = None

    # OCR / normalization
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_timed_out: bool = False
    normalized_plate: Optional[str] = None
    is_format_valid: Optional[bool] = None
    normalization_reason: Optional[str] = None

    # Direction / classification
    direction: Optional[DirectionOutcome] = None
    classification: Optional[Classification] = None

    # Guard-provided correction (manual entry / confirmation)
    guard_plate: Optional[str] = None

    # Image reference
    image_ref: Optional["ImageRef"] = None

    # Metrics
    processing_latency_ms: Optional[int] = None

    # Flags for the manual-review queue
    needs_manual_review: bool = False
    manual_review_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Access-control / gate value objects
# ---------------------------------------------------------------------------


@dataclass
class AccessDecision:
    """Outcome of an Access_Controller decision for an access attempt."""

    classification: Optional[Classification] = None
    grant_method: GrantMethod = GrantMethod.NONE
    gate_requested: bool = False
    surfaced_to_guard: bool = False
    reason: Optional[str] = None
    event_record: Optional["EventRecord"] = None


@dataclass(frozen=True)
class GateResult:
    """Completion result of a gate-open action."""

    success: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Persistence records
# ---------------------------------------------------------------------------


@dataclass
class EventRecord:
    """Full Event_Log schema (design Data Models section).

    Absent values are represented explicitly (empty string, None, or the
    ``NA_SENTINEL`` for numeric metrics) rather than being omitted
    (Req 10.4, 10.8, 15.5, 16.4).
    """

    id: str
    timestamp: str  # ISO-8601 with date, time, and tz offset (Req 10.3)
    ocr_plate: str = ""
    guard_plate: str = ""  # empty when no guard correction (Req 10.4)
    normalized_plate: str = ""  # shared entry/exit correlation key (Req 8.1)
    classification: Optional[Classification] = None
    direction: Optional[DirectionOutcome] = None
    grant_method: GrantMethod = GrantMethod.NONE
    event_kind: Optional[EventKind] = None
    entry_state: EntryState = EntryState.NA
    closed_by_event_id: Optional[str] = None
    image_ref: Optional[str] = None  # empty/None when capture disabled/failed
    # Metrics: a valid float in 0.0..1.0 / ms, or NA_SENTINEL when unavailable.
    detection_confidence: "float | str" = NA_SENTINEL
    ocr_confidence: "float | str" = NA_SENTINEL
    processing_latency_ms: "int | str" = NA_SENTINEL
    environment_label: Optional[EnvironmentLabel] = None


@dataclass
class ResidentRecord:
    """A registered resident plate record in the Resident_Database."""

    id: str
    normalized_plate: str  # uppercase, alnum, Indonesian_Plate_Format
    created_at: str  # ISO-8601 with tz
    updated_at: str  # ISO-8601 with tz


@dataclass(frozen=True)
class ImageRef:
    """Reference to a stored snapshot + thumbnail for an event."""

    event_id: str
    snapshot_path: str  # compressed snapshot, smaller than original (Req 11.4)
    thumbnail_path: str  # longest edge <= configured max (Req 11.4)
    captured_at: str  # ISO-8601


# ---------------------------------------------------------------------------
# Configuration error value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigError:
    """A single configuration validation failure.

    Names the offending setting so the composition root can refuse to start
    the affected component (Req 14.5).
    """

    key: str
    message: str
