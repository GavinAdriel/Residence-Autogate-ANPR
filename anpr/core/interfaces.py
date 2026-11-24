"""Abstract component interfaces for the ANPR Autogate System.

Every environment-specific component is defined here as a ``typing.Protocol``
so that mock implementations (simulated gate, webcam source) and field
implementations (relay/PLC gate, IP camera source) are structurally
interchangeable and are selected purely by configuration in the composition
root. No core component imports a concrete peer implementation, satisfying the
loose-coupling requirement (Requirement 14.4).

These protocols contain signatures only; concrete implementations live in the
respective sub-packages (config, pipeline, detection, ocr, gate, direction,
persistence, imaging).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

try:  # numpy is a runtime dependency (via OpenCV/Ultralytics) but keep the
    # scaffold importable before dependencies are installed.
    import numpy as np

    Frame = np.ndarray
except ModuleNotFoundError:  # pragma: no cover - exercised only pre-install
    Frame = Any  # type: ignore[misc,assignment]

from anpr.core.models import (
    ConfigError,
    Detection,
    DirectionOutcome,
    EventRecord,
    GateResult,
    ImageRef,
    OcrResult,
    ResidentRecord,
)

# ``Frame`` is a NumPy image array (H x W x C) at runtime; defined above with a
# graceful fallback to ``Any`` when numpy is not yet installed.


@runtime_checkable
class ConfigProvider(Protocol):
    """Loads layered configuration (file + environment overrides)."""

    def get(self, key: str) -> Any:
        """Return the resolved value for a dotted config key."""
        ...

    def get_section(self, name: str) -> Mapping[str, Any]:
        """Return a mapping of all keys within a config section."""
        ...

    def validate(self) -> list[ConfigError]:
        """Return validation errors; an empty list means the config is valid."""
        ...


@runtime_checkable
class VideoSource(Protocol):
    """A source of video frames (webcam or IP camera) behind OpenCV."""

    def open(self) -> None:
        """Open the source; raises when the source is unavailable."""
        ...

    def read(self) -> Optional[Frame]:
        """Return the next frame, or None when no frame is available now."""
        ...

    def close(self) -> None:
        """Release the source."""
        ...

    @property
    def descriptor(self) -> str:
        """Human-readable source identifier used in logs."""
        ...


@runtime_checkable
class VehicleDetector(Protocol):
    """Vehicle detector wrapping the YOLO26 weights."""

    def load(self) -> None:
        """Load weights; raises a weights-load error on failure."""
        ...

    def detect(self, frame: Frame) -> list[Detection]:
        """Detect vehicles in a frame and return retained detections."""
        ...


@runtime_checkable
class OcrEngine(Protocol):
    """OCR engine wrapping PaddleOCR."""

    def read_plate(self, crop: Frame) -> OcrResult:
        """Read candidate plate text from a cropped region."""
        ...


@runtime_checkable
class GateController(Protocol):
    """Abstraction over the physical/simulated gate-open action."""

    def open_gate(self, event_id: str) -> GateResult:
        """Request a gate-open for the given event; returns success/failure."""
        ...


@runtime_checkable
class DirectionResolver(Protocol):
    """Resolves travel direction from the camera that produced a detection."""

    def resolve_for_camera(self, camera_id: Optional[int]) -> DirectionOutcome:
        """Return INBOUND/OUTBOUND for a mapped camera id, else UNDETERMINED."""
        ...


@runtime_checkable
class ResidentRepository(Protocol):
    """Read-only access to the resident whitelist.

    The whitelist is managed externally (phpMyAdmin), so the port exposes only
    the single read used by the access controller (Req 3.1, 3.2, 3.4).
    """

    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        """Return the resident record for an exact normalized-plate match."""
        ...


@runtime_checkable
class EventLogRepository(Protocol):
    """Append + entry/exit correlation access to the Event_Log.

    The base flow writes exactly one record per detection (Req 4.7). The
    dual-camera exit feature additionally reads back the open entry for an
    exiting plate and closes it, reviving the correlation the monitoring
    migration had parked (dual-camera-exit-correlation Req 5). The two
    correlation methods are used only on the OUTBOUND path; single-camera /
    flat-log deployments call only :meth:`append`.
    """

    def append(self, record: EventRecord) -> int:
        """Write exactly one record and return its generated ``Log_ID``.

        Exactly-once per detection. The returned id is the MySQL-assigned
        ``ANPR_Log.Log_ID``, required to attach child rows such as the ``Images``
        reference, whose ``Log_ID`` foreign key is NOT NULL.
        """
        ...

    def find_open_entry(self, normalized_plate: str) -> Optional[EventRecord]:
        """Return the most recent OPEN entry for a plate, or None.

        Looks up the Open_Entry_Record for ``normalized_plate``
        (``Event_Kind = ENTRY`` AND ``Entry_State = OPEN``), most recent first,
        for exit correlation (Req 5.1). Returns ``None`` when no open entry
        exists (an Exit_Anomaly).
        """
        ...

    def close_entry(self, entry_log_id: int, closed_by_log_id: int) -> int:
        """Close an open entry, returning the number of rows affected.

        Sets ``Entry_State = CLOSED`` and ``Closed_By_Log_ID`` on the entry whose
        ``Log_ID`` is ``entry_log_id``, guarded by ``Entry_State = OPEN`` so the
        close is idempotent and race-safe: closing an already-closed entry
        affects zero rows and is not an error (Req 5.2, 5.6).
        """
        ...


@runtime_checkable
class ImageStore(Protocol):
    """Captures, serves, and retains event frame images on disk."""

    def capture_and_store(self, frame: Frame, event_id: str) -> Optional[ImageRef]:
        """Write a snapshot + thumbnail to disk; None on failure.

        Writes files only. The database row is added by
        :meth:`record_reference` once the event has a ``Log_ID``.
        """
        ...

    def record_reference(self, ref: ImageRef, log_id: int) -> bool:
        """Record a stored capture against a logged event's ``Log_ID``.

        Returns ``False`` on a database fault, leaving the event logged and the
        files on disk.
        """
        ...

    def get_thumbnail(self, ref: ImageRef) -> bytes:
        """Return the thumbnail bytes for a stored image reference."""
        ...

    def run_retention(self, now: datetime) -> int:
        """Delete images older than the retention period; return count deleted."""
        ...
