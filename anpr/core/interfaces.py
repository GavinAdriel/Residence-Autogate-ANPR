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
    TrackHistory,
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
    """Resolves the travel direction of a tracked vehicle."""

    def resolve(self, track: TrackHistory) -> DirectionOutcome:
        """Resolve a track history into a direction outcome."""
        ...


@runtime_checkable
class ResidentRepository(Protocol):
    """CRUD access to the Resident_Database."""

    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        """Return the resident record for an exact normalized-plate match."""
        ...

    def create(self, plate: str) -> ResidentRecord:
        """Create and return a new resident record."""
        ...

    def update(self, id: str, plate: str) -> Optional[ResidentRecord]:
        """Update an existing record; return None when it does not exist."""
        ...

    def delete(self, id: str) -> bool:
        """Delete a record; return True when a record was deleted."""
        ...

    def list_all(self) -> list[ResidentRecord]:
        """Return all resident records."""
        ...


@runtime_checkable
class EventLogRepository(Protocol):
    """Append-and-correlate access to the Event_Log."""

    def append(self, record: EventRecord) -> None:
        """Write exactly one record atomically (exactly-once per attempt)."""
        ...

    def find_open_entries(self, normalized_plate: str) -> list[EventRecord]:
        """Return Open_Entry_Records with an exact normalized-plate match."""
        ...

    def close_open_entry(self, entry_id: str, exit_id: str) -> None:
        """Close an Open_Entry_Record, referencing the closing exit event."""
        ...


@runtime_checkable
class ImageStore(Protocol):
    """Captures, serves, and retains event frame images on disk."""

    def capture_and_store(self, frame: Frame, event_id: str) -> Optional[ImageRef]:
        """Capture and persist a snapshot + thumbnail; None on failure."""
        ...

    def get_thumbnail(self, ref: ImageRef) -> bytes:
        """Return the thumbnail bytes for a stored image reference."""
        ...

    def run_retention(self, now: datetime) -> int:
        """Delete images older than the retention period; return count deleted."""
        ...
