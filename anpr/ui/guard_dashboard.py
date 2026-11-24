"""PyQt5 Guard monitoring dashboard for the ANPR Autogate System.

:class:`GuardDashboard` is the security guard's live operational surface. It is a
three-panel PyQt5 window (Requirement 12):

* **Left panel** -- the live camera feed with YOLO26 bounding-box overlays,
  refreshing at no less than the configured minimum frame rate and rendering
  each overlay within the configured maximum overlay-to-frame latency (Req 12.1).
* **Top-right panel** -- on a new detection, updated within the configured
  maximum update latency to show the detected plate number, its Resident/Guest
  tag, and the detection timestamp (Req 12.2).
* **Bottom-right panel** -- a manual plate-number input accepting up to the
  configured maximum character length, plus an "Open Gate" control (Req 12.3),
  and the manual-review queue of events awaiting a guard decision (Req 12.7).

Activating "Open Gate" requests the gate-open action through the injected
``GateController`` (Req 12.4). When the gate reports success, an Event_Log record
marked *manually granted* (``grant_method = MANUAL``) is written, carrying the
guard-entered plate value normalized by the ``PlateNormalizer`` where one was
provided (Req 12.5). When the gate reports failure, an error identifying the
failed action is presented and no manually-granted record is written (Req 12.6).

Loose coupling (Requirement 14.4): the dashboard depends only on the Protocol
interfaces in ``anpr.core.interfaces`` (``GateController``, ``EventLogRepository``,
``ImageStore``, ``ConfigProvider``) and on the pure ``PlateNormalizer`` -- never
on a concrete implementation -- so simulated and field adapters are structurally
interchangeable and chosen only in the composition root.

Design note -- separation of concerns: the two pieces of behaviour that carry
correctness properties are implemented as *pure* helper classes with no Qt
dependency, so they can be exercised without a running GUI:

* :class:`ManualReviewQueue` -- retains surfaced events until they are explicitly
  resolved and removes each exactly once on resolution (Property 27, Req 12.7).
* :class:`ManualGrantService` -- performs the gate-open + manual-grant recording
  transaction and produces the manually-granted Event_Log record with the
  normalized guard plate (Property 26, Req 12.4-12.6).

The heavy PyQt5 import is guarded (mirroring the ``numpy``/``cv2``/``yaml``
fallbacks used elsewhere in the codebase) so this module stays importable and
byte-compilable before PyQt5 is installed; :class:`GuardDashboard` raises a clear
error at construction time when PyQt5 is unavailable. The GUI tests
(``pytest-qt``) are a separate task.

See .kiro/specs/anpr-autogate-system/design.md (Guard_Dashboard section) and
requirements.md (Requirement 12) for the authoritative acceptance criteria.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

from anpr.core.interfaces import (
    ConfigProvider,
    EventLogRepository,
    Frame,
    GateController,
    ImageStore,
)
from anpr.core.models import (
    NA_SENTINEL,
    Classification,
    DetectionEvent,
    DirectionOutcome,
    EntryState,
    EnvironmentLabel,
    EventKind,
    EventRecord,
    GrantMethod,
    ImageRef,
)
from anpr.core.normalizer import PlateNormalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded PyQt5 import (kept importable before the GUI dependency is installed)
# ---------------------------------------------------------------------------
try:  # PyQt5 is a pinned runtime dependency; guard so the module stays
    # importable/byte-compilable pre-install (matches the codebase pattern).
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
    from PyQt5.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    _PYQT5_AVAILABLE = True
    _PYQT5_IMPORT_ERROR: Optional[BaseException] = None
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only pre-install
    _PYQT5_AVAILABLE = False
    _PYQT5_IMPORT_ERROR = exc
    # Fallback so the ``GuardDashboard`` class body (which subclasses the main
    # window) can still be defined and the module imported. Construction guards
    # against the missing dependency and raises a clear error.
    QMainWindow = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Configuration keys and defaults (design Configuration schema: ui.guard.*)
# ---------------------------------------------------------------------------

# Minimum feed refresh rate in fps; the feed refreshes at >= this rate (Req 12.1).
DEFAULT_MIN_FPS = 10
# Maximum overlay-to-frame render latency budget in ms (Req 12.1).
DEFAULT_MAX_OVERLAY_LATENCY_MS = 200
# Maximum top-right update latency budget in ms on a new detection (Req 12.2).
DEFAULT_MAX_UPDATE_LATENCY_MS = 500
# Maximum surfacing latency budget in ms for manual-review events (Req 12.7).
DEFAULT_MAX_SURFACING_LATENCY_MS = 1000
# Maximum accepted manual plate-input length in characters (Req 12.3).
DEFAULT_MANUAL_PLATE_MAX_LEN = 12


@dataclass(frozen=True)
class GuardConfig:
    """Resolved ``ui.guard.*`` timing/limit settings for the dashboard."""

    min_fps: int = DEFAULT_MIN_FPS
    max_overlay_latency_ms: int = DEFAULT_MAX_OVERLAY_LATENCY_MS
    max_update_latency_ms: int = DEFAULT_MAX_UPDATE_LATENCY_MS
    max_surfacing_latency_ms: int = DEFAULT_MAX_SURFACING_LATENCY_MS
    manual_plate_max_len: int = DEFAULT_MANUAL_PLATE_MAX_LEN

    @property
    def feed_interval_ms(self) -> int:
        """Timer interval that guarantees at least ``min_fps`` refreshes/sec."""
        fps = self.min_fps if self.min_fps and self.min_fps > 0 else DEFAULT_MIN_FPS
        return max(1, int(1000 // fps))

    @classmethod
    def from_config(cls, config: Optional[ConfigProvider]) -> "GuardConfig":
        """Build from a ``ConfigProvider``, applying documented defaults.

        Absent keys fall back to their defaults so the dashboard is usable even
        with a partial configuration.
        """
        if config is None:
            return cls()
        return cls(
            min_fps=int(_cfg(config, "ui.guard.min_fps", DEFAULT_MIN_FPS)),
            max_overlay_latency_ms=int(
                _cfg(config, "ui.guard.max_overlay_latency_ms", DEFAULT_MAX_OVERLAY_LATENCY_MS)
            ),
            max_update_latency_ms=int(
                _cfg(config, "ui.guard.max_update_latency_ms", DEFAULT_MAX_UPDATE_LATENCY_MS)
            ),
            max_surfacing_latency_ms=int(
                _cfg(
                    config,
                    "ui.guard.max_surfacing_latency_ms",
                    DEFAULT_MAX_SURFACING_LATENCY_MS,
                )
            ),
            manual_plate_max_len=int(
                _cfg(config, "ui.guard.manual_plate_max_len", DEFAULT_MANUAL_PLATE_MAX_LEN)
            ),
        )


def _cfg(config: ConfigProvider, key: str, default: Any) -> Any:
    """Read a dotted config key, falling back to ``default`` when absent/None."""
    try:
        value = config.get(key)
    except KeyError:
        return default
    return default if value is None else value


def _now_hms() -> str:
    """Return the current local time as ``HH:MM:SS`` for status messages."""
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Visual style helpers
# ---------------------------------------------------------------------------

# Recent-activity rows kept; a camera dot turns red after this many seconds
# without a frame.
_MAX_ACTIVITY = 10
_FEED_STALE_S = 3.0

# Badge / banner colours: badges are a single hex; banners are (bg, border, fg).
_GREY = "#6b7280"
_GREEN = "#16a34a"
_ORANGE = "#ea580c"
_BLUE = "#2563eb"
_PURPLE = "#7c3aed"
_GREY_BANNER = ("#f3f4f6", "#d1d5db", "#4b5563")
_RED_BANNER = ("#fee2e2", "#dc2626", "#991b1b")
_GREEN_BANNER = ("#dcfce7", "#16a34a", "#166534")

_DASHBOARD_STYLE = """
QWidget#central { background: #f3f4f6; font-family: 'Segoe UI', Arial; font-size: 13px; }
QFrame#statusBar { background: #111827; border-radius: 6px; }
QLabel#appTitle { color: white; font-size: 16px; font-weight: bold; }
QLabel#barText { color: #e5e7eb; margin-right: 12px; }
QLabel#dot { font-size: 16px; }
QFrame#card { background: white; border-radius: 6px; }
QFrame#panel { background: #1f2937; border-radius: 6px; }
QLabel#panelTitle { color: #9ca3af; }
QLabel#feed { background: #111827; color: #9ca3af; }
QLabel#caption { color: #6b7280; font-size: 11px; font-weight: bold; }
QLabel#plate { font-size: 34px; font-weight: bold; color: #111827; }
QLabel#muted { color: #6b7280; }
QLabel#badge { color: white; font-weight: bold; padding: 4px 14px; border-radius: 12px; }
QPushButton { padding: 6px 14px; border-radius: 4px; border: 1px solid #d1d5db; background: #f9fafb; }
QPushButton#primary { background: #16a34a; color: white; font-weight: bold; border: none; padding: 8px 18px; }
QLineEdit { padding: 6px; border: 1px solid #d1d5db; border-radius: 4px; }
QListWidget { border: 1px solid #e5e7eb; border-radius: 4px; font-family: 'Segoe UI', Arial; font-size: 12px; }
"""


def _set_dot(dot: Any, ok: Optional[bool]) -> None:
    """Colour a status dot: green (ok), red (down) or grey (unknown)."""
    colour = _GREY if ok is None else (_GREEN if ok else "#dc2626")
    dot.setStyleSheet(f"color: {colour};")


def _set_badge(label: Any, text: str, colour: str) -> None:
    """Set a pill badge's text and background colour."""
    label.setText(text)
    label.setStyleSheet(f"background: {colour};")


def _caption(text: str, parent: Any) -> Any:
    """Small grey caption label used as a card heading."""
    label = QLabel(text, parent)
    label.setObjectName("caption")
    return label


def _titled(title: str, content: Any, parent: Any) -> Any:
    """Wrap ``content`` in a dark panel with a title line (live-feed panes)."""
    frame = QFrame(parent)
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    heading = QLabel(title, frame)
    heading.setObjectName("panelTitle")
    layout.addWidget(heading)
    layout.addWidget(content, 1)
    return frame


def _outcome_text(event: DetectionEvent) -> str:
    """One-line outcome for a classified detection."""
    if getattr(event, "is_exit_anomaly", False):
        return "Exit without entry ticket - guard decision needed"
    if event.classification is Classification.RESIDENT:
        return "Resident matched - gate open requested"
    if event.classification is Classification.GUEST:
        return "Guest - waiting for guard"
    return ""


def _direction_style(direction: Optional[DirectionOutcome]) -> tuple[str, str]:
    """Return ``(text, colour)`` for the direction badge."""
    if direction is DirectionOutcome.INBOUND:
        return "INBOUND", _BLUE
    if direction is DirectionOutcome.OUTBOUND:
        return "OUTBOUND", _PURPLE
    return "DIRECTION ?", _GREY


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with tz offset."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Manual-review queue (pure; no Qt dependency) -- Req 12.7 / Property 27
# ---------------------------------------------------------------------------


@dataclass
class ManualReviewEntry:
    """A single event surfaced to the guard for a manual decision.

    ``token`` is a stable, per-queue identifier assigned on surfacing so an
    event can be resolved unambiguously even when several detections share
    (or lack) an ``event_id``.
    """

    token: int
    event: DetectionEvent
    reason: Optional[str] = None
    surfaced_at: Optional[float] = None


class ManualReviewQueue:
    """Retains surfaced events until the guard explicitly resolves them.

    Backs Requirement 12.7 and Property 27: an event surfaced for a manual
    decision (OCR failure, low-confidence, format-invalid, direction-undetermined
    or Exit_Anomaly) remains queued until it is explicitly resolved, and is
    removed *exactly once* on resolution. This class is pure (no Qt) so the
    retention invariant is testable without a GUI.
    """

    def __init__(self) -> None:
        self._entries: list[ManualReviewEntry] = []
        self._next_token = 0

    def surface(
        self,
        event: DetectionEvent,
        reason: Optional[str] = None,
        *,
        surfaced_at: Optional[float] = None,
    ) -> int:
        """Add ``event`` to the queue and return its stable resolution token.

        The reason defaults to the event's own ``manual_review_reason`` when the
        caller does not supply one, so the pipeline-surfaced reason is preserved.
        """
        token = self._next_token
        self._next_token += 1
        self._entries.append(
            ManualReviewEntry(
                token=token,
                event=event,
                reason=reason
                if reason is not None
                else getattr(event, "manual_review_reason", None),
                surfaced_at=surfaced_at,
            )
        )
        return token

    def resolve(self, token: int) -> bool:
        """Remove the entry with ``token``; return True the first time only.

        Resolving the same token again returns False, so an event is removed
        exactly once on resolution (Property 27).
        """
        for index, entry in enumerate(self._entries):
            if entry.token == token:
                del self._entries[index]
                return True
        return False

    @property
    def pending(self) -> list[ManualReviewEntry]:
        """A snapshot list of the currently unresolved entries, in FIFO order."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, token: object) -> bool:
        return any(entry.token == token for entry in self._entries)


# ---------------------------------------------------------------------------
# Manual-grant service (pure; no Qt dependency) -- Req 12.4-12.6 / Property 26
# ---------------------------------------------------------------------------


@dataclass
class ManualGrantResult:
    """Outcome of a guard-initiated Open Gate action.

    ``gate_success`` reports whether the ``GateController`` opened the gate;
    ``recorded`` reports whether the manually-granted Event_Log record was
    persisted. ``record`` is the built record (persisted when ``recorded`` is
    True). ``detail`` carries a human-readable message for the guard.
    """

    gate_success: bool
    recorded: bool
    detail: str
    record: Optional[EventRecord] = None


class ManualGrantService:
    """Performs the gate-open + manual-grant recording transaction (Req 12.4-12.6).

    Depends only on the ``GateController`` and ``EventLogRepository`` Protocols,
    the pure ``PlateNormalizer``, and (optionally) the ``ImageStore`` Protocol,
    so it carries no coupling to concrete adapters (Req 14.4).
    """

    def __init__(
        self,
        gate_controller: GateController,
        event_log_repo: EventLogRepository,
        normalizer: PlateNormalizer,
        *,
        image_store: Optional[ImageStore] = None,
        environment_label: Optional[EnvironmentLabel] = None,
        camera_id: Optional[int] = None,
        id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
        clock_iso: Callable[[], str] = _now_iso,
    ) -> None:
        self._gate = gate_controller
        self._event_log = event_log_repo
        self._normalizer = normalizer
        self._image_store = image_store
        self._environment_label = environment_label
        # ANPR_Log.Camera_ID is NOT NULL with an FK to Camera, so a manual-grant
        # record must carry a camera id just like an automatic detection does.
        self._camera_id = camera_id
        self._id_factory = id_factory
        self._clock_iso = clock_iso

    def record_manual_grant(
        self,
        plate: Optional[str],
        *,
        frame: Optional[Frame] = None,
        event_id: Optional[str] = None,
        classification: Optional[Classification] = None,
        is_exit: bool = False,
        direction: Optional[DirectionOutcome] = None,
        camera_id: Optional[int] = None,
    ) -> ManualGrantResult:
        """Request the gate-open and, on success, record the manual grant.

        Requirement 12.4: request the gate-open through the ``GateController``.
        Requirement 12.6: on gate failure, return a failure result and write no
        manually-granted record. Requirement 12.5 / Property 26: on gate
        success, write exactly one Event_Log record marked ``grant_method =
        MANUAL`` whose guard-corrected plate value is the ``PlateNormalizer``
        output of the guard input where a plate was provided (empty otherwise).

        ``classification`` tags the written row (e.g. ``Classification.GUEST``
        when the guard permits a surfaced guest). It is the *only* row written
        for that vehicle -- guests are no longer auto-logged at detection time,
        so this manual grant is the guest's single Event_Log record.
        """
        eid = event_id or self._id_factory()

        # Req 12.4: request the gate-open action through the abstraction.
        gate_result = self._gate.open_gate(eid)

        # Req 12.6: gate-open failed -> surface an error, do NOT mark granted.
        if not gate_result.success:
            return ManualGrantResult(
                gate_success=False,
                recorded=False,
                detail=gate_result.detail or "Gate-open action failed.",
                record=None,
            )

        # Req 12.5 / Property 26: normalize the guard-entered plate where one was
        # provided; store the normalized value as the guard-corrected plate and
        # reuse it as the entry/exit correlation key.
        provided = plate is not None and str(plate).strip() != ""
        normalized_plate = ""
        if provided:
            normalized_plate = self._normalizer.normalize(str(plate)).normalized

        # The files are written now so the snapshot path can go into
        # ANPR_Log.Image_Ref; the Images row is inserted after the log write,
        # once a Log_ID exists for its NOT NULL foreign key.
        captured = self._maybe_capture(frame, eid)

        # A guard-permitted Exit_Anomaly is recorded as an OUTBOUND EXIT row
        # (still Grant_Method=MANUAL); any other manual grant stays a plain
        # MANUAL event with no direction (dual-camera-exit-correlation Req 5.4).
        # A guard-permitted INBOUND guest (dual-camera mode, where the detection
        # carries a direction) is the guest's entry ticket: it is written as an
        # ENTRY/OPEN row so a later OUTBOUND read can correlate to it and
        # auto-authorize the exit (Req 4.1, 5.1). Without a direction (single
        # camera) or for a bare manual open it stays a plain MANUAL event.
        entry_state = EntryState.NA
        if is_exit:
            direction = DirectionOutcome.OUTBOUND
            event_kind = EventKind.EXIT
        elif (
            classification is Classification.GUEST
            and direction is DirectionOutcome.INBOUND
        ):
            event_kind = EventKind.ENTRY
            entry_state = EntryState.OPEN
        else:
            direction = None
            event_kind = EventKind.MANUAL

        record = EventRecord(
            id=eid,
            timestamp=self._clock_iso(),
            ocr_plate="",
            guard_plate=normalized_plate,
            normalized_plate=normalized_plate,
            classification=classification,
            direction=direction,
            grant_method=GrantMethod.MANUAL,
            event_kind=event_kind,
            entry_state=entry_state,
            image_ref=(captured.snapshot_path if captured is not None else None),
            # Prefer the camera that saw the vehicle (dual-camera mode) so an
            # Exit_Anomaly permit is attributed to the outbound camera.
            camera_id=(camera_id if camera_id is not None else self._camera_id),
            detection_confidence=NA_SENTINEL,
            ocr_confidence=NA_SENTINEL,
            processing_latency_ms=NA_SENTINEL,
            environment_label=self._environment_label,
        )

        try:
            self._event_log.append(record)
        except Exception as exc:  # noqa: BLE001 - a write fault must not crash the UI
            # The gate opened, but the record could not be persisted. Report it
            # so the guard is informed; the repository's atomic write leaves no
            # partial record (Req 10.10).
            logger.error("Manual-grant Event_Log write failed for event %s: %s", eid, exc)
            return ManualGrantResult(
                gate_success=True,
                recorded=False,
                detail=f"Gate opened, but recording the manual grant failed: {exc}",
                record=record,
            )

        # Link the capture to the freshly written log row (Images.Log_ID FK).
        self._maybe_link_image(captured, record)

        return ManualGrantResult(
            gate_success=True,
            recorded=True,
            detail=gate_result.detail or "Gate opened; manual grant recorded.",
            record=record,
        )

    def _maybe_capture(
        self, frame: Optional[Frame], event_id: str
    ) -> Optional[ImageRef]:
        """Write an image for the grant to disk when a store/frame exist.

        Returns the reference so the caller can both put the snapshot path on the
        record and, after the log write, insert the ``Images`` row. A capture
        failure never blocks the grant: the record is kept without an image
        reference (Req 11.7).
        """
        if frame is None or self._image_store is None:
            return None
        try:
            return self._image_store.capture_and_store(frame, event_id)
        except Exception as exc:  # noqa: BLE001 - capture must not crash the UI
            logger.error("Manual-grant image capture failed for event %s: %s", event_id, exc)
            return None

    def _maybe_link_image(
        self, captured: Optional[ImageRef], record: EventRecord
    ) -> None:
        """Insert the ``Images`` row for a capture once ``Log_ID`` is known.

        Never raises: the gate has already opened and the grant is already
        recorded, so a failure to record the image reference must not turn a
        successful grant into an error (Req 11.7).
        """
        if captured is None or self._image_store is None or record.log_id is None:
            return
        try:
            self._image_store.record_reference(captured, record.log_id)
        except Exception as exc:  # noqa: BLE001 - linkage must not crash the UI
            logger.error(
                "Manual-grant image linkage failed for log %s: %s",
                record.log_id,
                exc,
            )


# ---------------------------------------------------------------------------
# Guard dashboard (PyQt5 three-panel window)
# ---------------------------------------------------------------------------


class GuardDashboard(QMainWindow):  # type: ignore[misc,valid-type]
    """The PyQt5 three-panel Guard monitoring window (Requirement 12).

    Parameters
    ----------
    gate_controller:
        The gate-open abstraction (Req 12.4).
    event_log_repo:
        Append access to the Event_Log for manual grants (Req 12.5).
    normalizer:
        The pure plate normalizer applied to guard input (Req 12.5).
    config:
        Optional ``ConfigProvider`` supplying ``ui.guard.*`` timing/limit
        settings; documented defaults are used when absent.
    image_store:
        Optional image store; when present a snapshot is captured for a manual
        grant (Req 11.1, 11.7).
    frame_provider:
        Optional callable returning ``(frame, detections)`` for the live feed;
        polled by the refresh timer. ``detections`` is a list of
        :class:`~anpr.core.models.Detection`.
    environment_label:
        Optional deployment environment stamped onto manual-grant records.
    on_resolve_track:
        Optional callback invoked with a resolved guest's ``track_id`` when the
        guard permits it via Open Gate. Wired by the composition root to
        ``TrackAggregator.mark_resolved`` so the shared dedup state marks the
        guest's track resolved and the pipeline stops re-surfacing it.

    Threading note: ``on_detection`` and ``surface_event`` update Qt widgets and
    must be invoked on the GUI thread. The composition root (Task 18) is
    responsible for marshalling pipeline callbacks onto the GUI thread.
    """

    def __init__(
        self,
        gate_controller: GateController,
        event_log_repo: EventLogRepository,
        normalizer: PlateNormalizer,
        *,
        config: Optional[ConfigProvider] = None,
        image_store: Optional[ImageStore] = None,
        frame_provider: Optional[Callable[[], Optional[tuple[Frame, list]]]] = None,
        primary_feed_title: str = "Live Feed",
        extra_feeds: Optional[
            Sequence[tuple[str, Callable[[], Optional[tuple[Frame, list]]]]]
        ] = None,
        environment_label: Optional[EnvironmentLabel] = None,
        camera_id: Optional[int] = None,
        on_resolve_track: Optional[Callable[[int], None]] = None,
        on_guest_entry: Optional[Callable[[str], None]] = None,
        parent: Optional[Any] = None,
    ) -> None:
        if not _PYQT5_AVAILABLE:  # pragma: no cover - exercised only pre-install
            raise RuntimeError(
                "PyQt5 is required to construct the GuardDashboard but is not "
                f"installed: {_PYQT5_IMPORT_ERROR}"
            )
        super().__init__(parent)

        self._guard_config = GuardConfig.from_config(config)
        self._frame_provider = frame_provider
        # Additional live-feed panes (e.g. the outbound camera in dual-camera
        # mode): (title, provider) pairs rendered below the primary feed.
        self._primary_feed_title = primary_feed_title
        self._extra_feeds = list(extra_feeds or [])
        self._extra_feed_labels: list[Any] = []
        # Invoked with a resolved guest's ``track_id`` when the guard permits it,
        # so the composition root can mark that track resolved on the shared
        # TrackAggregator -- the pipeline then stops re-surfacing/re-processing
        # the guest. ``None`` keeps the dashboard standalone (e.g. in tests).
        self._on_resolve_track = on_resolve_track
        # Invoked with the normalized plate when a guard permit writes an INBOUND
        # guest ENTRY ticket, so the composition root can record it for
        # same-pass reconciliation with the outbound camera.
        self._on_guest_entry = on_guest_entry

        # Pure collaborators (testable without a GUI).
        self._review_queue = ManualReviewQueue()
        self._grant_service = ManualGrantService(
            gate_controller,
            event_log_repo,
            normalizer,
            image_store=image_store,
            environment_label=environment_label,
            camera_id=camera_id,
        )

        # Most recent frame/detections, reused when capturing a manual-grant
        # image on Open Gate.
        self._current_frame: Optional[Frame] = None
        self._current_detections: list = []

        # The guest currently at the gate (if any). ``on_detection`` keeps the
        # plate input auto-filled with this guest's best read so the guard only
        # has to press Open Gate; a single guest is handled at a time. The edit
        # flag suppresses auto-fill once the guard starts correcting the plate,
        # so a fresh frame never clobbers an in-progress edit.
        self._current_guest_event: Optional[DetectionEvent] = None
        self._guard_edited_plate: bool = False

        # De-duplication state for the error banner and activity list.
        self._last_error_message: Optional[str] = None
        self._activity_keys: "deque[tuple]" = deque(maxlen=50)

        # Last action outcome, retained for tests and status display.
        self.last_grant_result: Optional[ManualGrantResult] = None

        self.setWindowTitle("ANPR Autogate - Guard Dashboard")
        self._build_ui()

        # Feed refresh timer: fires often enough to guarantee >= min fps
        # (Req 12.1). Not started automatically so a dashboard can be built
        # without a running feed; call :meth:`start_feed`.
        self._feed_timer = QTimer(self)
        self._feed_timer.setInterval(self._guard_config.feed_interval_ms)
        self._feed_timer.timeout.connect(self._refresh_feed)

    # ------------------------------------------------------------------
    # Read-only surface
    # ------------------------------------------------------------------
    @property
    def guard_config(self) -> GuardConfig:
        """The resolved ``ui.guard.*`` timing/limit configuration."""
        return self._guard_config

    @property
    def review_queue(self) -> ManualReviewQueue:
        """The manual-review queue backing the bottom-right list (Req 12.7)."""
        return self._review_queue

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Assemble the dashboard: status bar, live feeds, and info cards."""
        central = QWidget(self)
        central.setObjectName("central")
        central.setStyleSheet(_DASHBOARD_STYLE)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # --- Top status bar: title + camera / DB health dots ---------------
        bar = QFrame(central)
        bar.setObjectName("statusBar")
        bar_layout = QHBoxLayout(bar)
        title = QLabel("ANPR Autogate", bar)
        title.setObjectName("appTitle")
        bar_layout.addWidget(title)
        bar_layout.addStretch(1)

        self._cam_dots: list[Any] = []
        camera_titles = [self._primary_feed_title] + [t for t, _ in self._extra_feeds]
        for cam_title in camera_titles:
            dot = QLabel("\u25CF", bar)
            dot.setObjectName("dot")
            text = QLabel(cam_title.replace("Live Feed", "Camera").replace(" - ", ": "), bar)
            text.setObjectName("barText")
            bar_layout.addWidget(dot)
            bar_layout.addWidget(text)
            self._cam_dots.append(dot)
        self._db_dot = QLabel("\u25CF", bar)
        self._db_dot.setObjectName("dot")
        db_text = QLabel("DB", bar)
        db_text.setObjectName("barText")
        bar_layout.addWidget(self._db_dot)
        bar_layout.addWidget(db_text)
        outer.addWidget(bar)
        for dot in self._cam_dots + [self._db_dot]:
            _set_dot(dot, None)
        self._feed_last_seen: list[Optional[float]] = [None] * len(self._cam_dots)

        body = QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, 1)

        # --- Left: live feeds ----------------------------------------------
        feed_column = QVBoxLayout()
        multi = bool(self._extra_feeds)
        min_size = (400, 260) if multi else (480, 360)

        self._feed_label = QLabel("No feed", central)
        self._feed_label.setObjectName("feed")
        self._feed_label.setAlignment(Qt.AlignCenter)
        self._feed_label.setMinimumSize(*min_size)
        self._feed_label.setScaledContents(False)
        feed_column.addWidget(_titled(self._primary_feed_title, self._feed_label, central), 1)

        for extra_title, _provider in self._extra_feeds:
            extra_label = QLabel("No feed", central)
            extra_label.setObjectName("feed")
            extra_label.setAlignment(Qt.AlignCenter)
            extra_label.setMinimumSize(*min_size)
            extra_label.setScaledContents(False)
            self._extra_feed_labels.append(extra_label)
            feed_column.addWidget(_titled(extra_title, extra_label, central), 1)
        body.addLayout(feed_column, 2)

        # --- Right column: cards -------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(10)

        # Current vehicle card (Req 12.2).
        vehicle = QFrame(central)
        vehicle.setObjectName("card")
        v_layout = QVBoxLayout(vehicle)
        v_layout.addWidget(_caption("CURRENT VEHICLE", vehicle))
        self._plate_value = QLabel("-", vehicle)
        self._plate_value.setObjectName("plate")
        v_layout.addWidget(self._plate_value)
        badges = QHBoxLayout()
        self._tag_value = QLabel("-", vehicle)
        self._direction_value = QLabel("-", vehicle)
        for badge in (self._tag_value, self._direction_value):
            badge.setObjectName("badge")
            badge.setAlignment(Qt.AlignCenter)
            badges.addWidget(badge)
        badges.addStretch(1)
        _set_badge(self._tag_value, "-", _GREY)
        _set_badge(self._direction_value, "-", _GREY)
        v_layout.addLayout(badges)
        self._timestamp_value = QLabel("Time: -", vehicle)
        self._timestamp_value.setObjectName("muted")
        v_layout.addWidget(self._timestamp_value)
        self._outcome_value = QLabel("Waiting for a vehicle...", vehicle)
        self._outcome_value.setObjectName("muted")
        self._outcome_value.setWordWrap(True)
        v_layout.addWidget(self._outcome_value)
        right.addWidget(vehicle)

        # Current error banner (replaced on each new error, not a list).
        self._error_label = QLabel("", central)
        self._error_label.setWordWrap(True)
        self._error_label.setObjectName("banner")
        right.addWidget(self._error_label)
        self.clear_error()

        # Latest DB insert banner.
        self._db_label = QLabel("", central)
        self._db_label.setWordWrap(True)
        self._db_label.setObjectName("banner")
        right.addWidget(self._db_label)
        self._set_banner(self._db_label, "NO DB INSERTS YET", "", _GREY_BANNER)

        # Guest / manual control card (Req 12.3-12.7).
        control = QFrame(central)
        control.setObjectName("card")
        c_layout = QVBoxLayout(control)
        c_layout.addWidget(_caption("GUEST / MANUAL CONTROL", control))

        row = QHBoxLayout()
        self._plate_input = QLineEdit(control)
        self._plate_input.setPlaceholderText("Plate (auto-filled for a guest, editable)")
        # Req 12.3: accept up to the configured maximum character length.
        self._plate_input.setMaxLength(self._guard_config.manual_plate_max_len)
        # ``textEdited`` fires only on user keystrokes (not programmatic
        # ``setText``) so an in-progress correction is never clobbered.
        self._plate_input.textEdited.connect(self._on_plate_text_edited)
        row.addWidget(self._plate_input, 1)
        self._open_gate_button = QPushButton("Open Gate", control)
        self._open_gate_button.setObjectName("primary")
        self._open_gate_button.clicked.connect(self._on_open_gate_clicked)
        row.addWidget(self._open_gate_button)
        c_layout.addLayout(row)

        self._status_label = QLabel("", control)
        self._status_label.setWordWrap(True)
        c_layout.addWidget(self._status_label)

        c_layout.addWidget(_caption("Guest / exit-anomaly queue", control))
        self._review_list = QListWidget(control)
        self._review_list.setMaximumHeight(90)
        self._review_list.setWordWrap(True)
        self._review_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Selecting a surfaced guest pre-fills the manual plate input (Req 12.3).
        self._review_list.currentItemChanged.connect(self._on_review_selection_changed)
        c_layout.addWidget(self._review_list)
        self._resolve_button = QPushButton("Resolve selected", control)
        self._resolve_button.clicked.connect(self._on_resolve_clicked)
        c_layout.addWidget(self._resolve_button)
        right.addWidget(control)

        # Recent activity (in-memory history, newest first).
        activity = QFrame(central)
        activity.setObjectName("card")
        a_layout = QVBoxLayout(activity)
        a_layout.addWidget(_caption("RECENT ACTIVITY", activity))
        self._activity_list = QListWidget(activity)
        self._activity_list.setObjectName("activity")
        self._activity_list.setWordWrap(True)
        self._activity_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        a_layout.addWidget(self._activity_list)
        right.addWidget(activity, 1)

        body.addLayout(right, 1)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Banners, status dots and recent activity
    # ------------------------------------------------------------------
    def _set_banner(self, label: Any, head: str, detail: str, colors: tuple) -> None:
        """Style a banner label and set its heading + detail text."""
        bg, border, fg = colors
        label.setStyleSheet(
            f"background:{bg}; border:1px solid {border}; color:{fg}; "
            "border-radius:6px; padding:8px;"
        )
        label.setText(f"<b>{head}</b><br>{detail}" if detail else f"<b>{head}</b>")

    def _add_activity(self, text: str, *, error: bool = False) -> None:
        """Prepend a line to the recent-activity list (capped, newest first)."""
        item = QListWidgetItem(f"{_now_hms()}   {text}")
        if error:
            item.setForeground(QColor("#b91c1c"))
        self._activity_list.insertItem(0, item)
        while self._activity_list.count() > _MAX_ACTIVITY:
            self._activity_list.takeItem(self._activity_list.count() - 1)

    def _note_feed(self, index: int, alive: bool) -> None:
        """Record camera liveness and refresh its status dot."""
        if index >= len(self._cam_dots):
            return
        now = time.monotonic()
        if alive:
            self._feed_last_seen[index] = now
        last = self._feed_last_seen[index]
        if last is None:
            _set_dot(self._cam_dots[index], None)
        else:
            _set_dot(self._cam_dots[index], (now - last) <= _FEED_STALE_S)


    # ------------------------------------------------------------------
    # Live feed (Req 12.1)
    # ------------------------------------------------------------------
    def start_feed(self) -> None:
        """Start polling the frame provider at >= the configured min fps."""
        if not self._feed_timer.isActive():
            self._feed_timer.start()

    def stop_feed(self) -> None:
        """Stop the live-feed refresh timer."""
        self._feed_timer.stop()

    def _refresh_feed(self) -> None:
        """Timer slot: pull the latest frame + detections and render them."""
        # Extra feeds first: they render only and never touch the frame retained
        # for manual-grant capture.
        for index, ((title, provider), label) in enumerate(
            zip(self._extra_feeds, self._extra_feed_labels), start=1
        ):
            try:
                extra_payload = provider()
            except Exception as exc:  # noqa: BLE001 - must not crash the UI
                logger.error("Live-feed provider for %r failed: %s", title, exc)
                self._note_feed(index, False)
                continue
            if extra_payload is None:
                self._note_feed(index, False)
                continue
            extra_frame, extra_detections = extra_payload
            if extra_frame is None:
                self._note_feed(index, False)
                continue
            self._note_feed(index, True)
            pixmap = self._frame_to_pixmap(extra_frame, list(extra_detections or []))
            if pixmap is not None:
                label.setPixmap(
                    pixmap.scaled(
                        label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )

        if self._frame_provider is None:
            return
        try:
            payload = self._frame_provider()
        except Exception as exc:  # noqa: BLE001 - a provider fault must not crash the UI
            logger.error("Live-feed frame provider failed: %s", exc)
            self._note_feed(0, False)
            return
        if payload is None or payload[0] is None:
            self._note_feed(0, False)
            return
        self._note_feed(0, True)
        frame, detections = payload
        self.render_frame(frame, detections or [])

    def render_frame(self, frame: Optional[Frame], detections: list) -> None:
        """Render a frame with bounding-box overlays onto the left panel.

        Measures the overlay-to-frame render latency and logs when it exceeds
        the configured maximum (Req 12.1). The frame/detections are retained so
        an Open Gate action can capture the current frame.
        """
        self._current_frame = frame
        self._current_detections = list(detections or [])
        if frame is None:
            return

        started = time.monotonic()
        pixmap = self._frame_to_pixmap(frame, self._current_detections)
        if pixmap is not None:
            self._feed_label.setPixmap(
                pixmap.scaled(
                    self._feed_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms > self._guard_config.max_overlay_latency_ms:
            logger.warning(
                "Overlay render took %.0f ms, exceeding the %d ms budget.",
                elapsed_ms,
                self._guard_config.max_overlay_latency_ms,
            )

    def _frame_to_pixmap(self, frame: Frame, detections: list) -> Optional[Any]:
        """Convert a BGR/grayscale frame + detections into a QPixmap w/ overlays."""
        try:
            import numpy as np
        except ModuleNotFoundError:  # pragma: no cover - numpy is a runtime dep
            logger.error("NumPy is required to render the live feed.")
            return None

        arr = np.ascontiguousarray(frame)
        if arr.ndim == 3 and arr.shape[2] == 3:
            height, width = int(arr.shape[0]), int(arr.shape[1])
            # OpenCV frames are BGR; QImage.Format_RGB888 expects RGB.
            rgb = np.ascontiguousarray(arr[:, :, ::-1])
            image = QImage(
                rgb.data, width, height, 3 * width, QImage.Format_RGB888
            ).copy()
        elif arr.ndim == 2:
            height, width = int(arr.shape[0]), int(arr.shape[1])
            image = QImage(
                arr.data, width, height, width, QImage.Format_Grayscale8
            ).copy()
        else:
            logger.error("Unsupported frame shape for rendering: %s", getattr(arr, "shape", None))
            return None

        pixmap = QPixmap.fromImage(image)
        self._draw_overlays(pixmap, detections)
        return pixmap

    def _draw_overlays(self, pixmap: Any, detections: list) -> None:
        """Draw YOLO26 bounding boxes (and confidence labels) onto ``pixmap``."""
        if not detections:
            return
        painter = QPainter(pixmap)
        try:
            pen = QPen(QColor(0, 220, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            for detection in detections:
                box = getattr(detection, "box", None)
                if box is None:
                    continue
                painter.drawRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
                confidence = getattr(detection, "confidence", None)
                if confidence is not None:
                    painter.drawText(box.x1, max(0, box.y1 - 4), f"{confidence:.2f}")
        finally:
            painter.end()

    # ------------------------------------------------------------------
    # Top-right detection summary (Req 12.2)
    # ------------------------------------------------------------------
    def on_detection(self, event: DetectionEvent) -> None:
        """Update the top-right panel and auto-fill the plate for a guest.

        Shows the detected plate number, the Resident/Guest tag, and the
        detection timestamp (Req 12.2). For a **guest** at the gate this also
        auto-fills the manual plate input with the current best read so the
        guard only has to press Open Gate -- no queue selection needed. The
        input keeps refreshing to the best read on later frames of the same
        guest, unless the guard has started editing it (then the edit is
        preserved). A non-guest or empty detection clears the current guest and,
        when the guard has not typed anything, clears the input so no stale
        plate can be approved by accident.
        """
        started = time.monotonic()

        plate = event.normalized_plate or event.ocr_text or "-"
        self._plate_value.setText(plate)
        tag_colour = {
            Classification.RESIDENT: _GREEN,
            Classification.GUEST: _ORANGE,
        }.get(event.classification, _GREY)
        _set_badge(
            self._tag_value, _classification_tag(event.classification).upper(), tag_colour
        )
        dir_text, dir_colour = _direction_style(event.direction)
        _set_badge(self._direction_value, dir_text, dir_colour)
        self._timestamp_value.setText(f"Time: {_format_timestamp(event.timestamp)}")

        self._update_guest_autofill(event)

        # A vehicle that was classified (resident/guest) means the pipeline is
        # working again, so the previous error no longer applies.
        if event.classification is not None:
            self.clear_error()
            self._outcome_value.setText(_outcome_text(event))
            self._log_detection_activity(event, plate)

        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms > self._guard_config.max_update_latency_ms:
            logger.warning(
                "Detection panel update took %.0f ms, exceeding the %d ms budget.",
                elapsed_ms,
                self._guard_config.max_update_latency_ms,
            )

    # ------------------------------------------------------------------
    # Current-error and DB-insert messages
    # ------------------------------------------------------------------
    def show_error(self, message: str) -> None:
        """Show ``message`` as the current error (replaces any previous one)."""
        self._set_banner(
            self._error_label, f"ERROR {_now_hms()}", message, _RED_BANNER
        )
        # Repeated identical errors (e.g. a camera retry loop) update the banner
        # but are not re-added to the activity list.
        if message != self._last_error_message:
            self._last_error_message = message
            self._add_activity(f"ERROR  {message}", error=True)
        lowered = message.lower()
        if "db" in lowered or "lookup" in lowered or "insert" in lowered:
            _set_dot(self._db_dot, False)

    def clear_error(self) -> None:
        """Clear the current-error message."""
        self._last_error_message = None
        self._set_banner(self._error_label, "NO ERRORS", "", _GREY_BANNER)

    def show_db_insert(self, record: Any) -> None:
        """Show a clear confirmation that a row was inserted into the DB."""
        kind = getattr(getattr(record, "event_kind", None), "value", None)
        direction = getattr(getattr(record, "direction", None), "value", None)
        classification = getattr(getattr(record, "classification", None), "value", None)
        parts = [
            f"Log #{getattr(record, 'log_id', None)}",
            getattr(record, "normalized_plate", "") or "-",
        ]
        parts.extend(v for v in (classification, direction, kind) if v)
        self._set_banner(
            self._db_label,
            f"SAVED TO DB {_now_hms()}",
            " - ".join(str(p) for p in parts),
            _GREEN_BANNER,
        )
        _set_dot(self._db_dot, True)

    def _log_detection_activity(self, event: DetectionEvent, plate: str) -> None:
        """Add one activity line per vehicle track (not per frame)."""
        key = (event.camera_id, event.track_id, event.classification)
        if event.track_id is not None:
            if key in self._activity_keys:
                return
            self._activity_keys.append(key)
        direction = getattr(event.direction, "value", None)
        parts = [plate, _classification_tag(event.classification)]
        if direction:
            parts.append(direction)
        parts.append(_outcome_text(event))
        self._add_activity("   ".join(parts))

    def _update_guest_autofill(self, event: DetectionEvent) -> None:
        """Keep the plate input tracking the current guest's best read.

        A guest detection becomes (or stays) the current guest and auto-fills
        the input with its best-read normalized plate, as long as the guard has
        not started editing the field. Any non-guest detection means no guest is
        awaiting a manual grant, so the current guest is cleared and the input is
        reset (unless the guard has typed into it).
        """
        is_guest = event.classification is Classification.GUEST
        if not is_guest:
            # No guest at the gate right now.
            self._current_guest_event = None
            if not self._guard_edited_plate:
                self._clear_plate_input()
            return

        self._current_guest_event = event
        if self._guard_edited_plate:
            # Preserve an in-progress correction; do not clobber it.
            return
        plate = event.normalized_plate or ""
        if plate:
            # Programmatic set: does not trip the ``textEdited`` guard.
            self._plate_input.setText(plate)

    def _on_plate_text_edited(self, _text: str) -> None:
        """Mark that the guard has manually edited the plate input."""
        self._guard_edited_plate = True

    def _clear_plate_input(self) -> None:
        """Clear the plate input and reset the manual-edit flag."""
        self._plate_input.clear()
        self._guard_edited_plate = False

    # ------------------------------------------------------------------
    # Manual-review queue (Req 12.7)
    # ------------------------------------------------------------------
    def surface_event(self, event: DetectionEvent, reason: Optional[str] = None) -> int:
        """Surface an event for a manual decision and render it in the queue.

        Handles OCR-failure, low-confidence, format-invalid, direction-
        undetermined, and Exit_Anomaly events (Req 12.7). Returns the queue
        token so the caller can resolve the event later. The event is retained
        until explicitly resolved.
        """
        surfaced_at = time.monotonic()
        token = self._review_queue.surface(event, reason, surfaced_at=surfaced_at)

        item = QListWidgetItem(_review_label(event, reason))
        item.setData(Qt.UserRole, token)
        self._review_list.addItem(item)

        elapsed_ms = (time.monotonic() - surfaced_at) * 1000.0
        if elapsed_ms > self._guard_config.max_surfacing_latency_ms:
            logger.warning(
                "Surfacing event took %.0f ms, exceeding the %d ms budget.",
                elapsed_ms,
                self._guard_config.max_surfacing_latency_ms,
            )
        return token

    def resolve_event(self, token: int) -> bool:
        """Resolve the queued event with ``token``, removing it from the list.

        Returns True the first time the token is resolved and False afterwards,
        so an event is removed exactly once (Req 12.7 / Property 27).
        """
        removed = self._review_queue.resolve(token)
        if removed:
            for row in range(self._review_list.count()):
                item = self._review_list.item(row)
                if item is not None and item.data(Qt.UserRole) == token:
                    self._review_list.takeItem(row)
                    break
        return removed

    def _on_resolve_clicked(self) -> None:
        """Button slot: resolve the currently selected manual-review entry."""
        item = self._review_list.currentItem()
        if item is None:
            return
        token = item.data(Qt.UserRole)
        if token is not None:
            self.resolve_event(int(token))

    def _on_review_selection_changed(self, current: Any, _previous: Any) -> None:
        """Pre-fill the plate input from the selected guest's best read (Req 12.3).

        Selecting a surfaced guest copies its best-read normalized plate into the
        manual input so the guard can permit it with a single Open Gate click,
        while still being free to edit the value. A non-guest surfaced entry
        (e.g. a lookup fault) leaves the input untouched.
        """
        if current is None:
            return
        token = current.data(Qt.UserRole)
        if token is None:
            return
        event = self._event_for_token(int(token))
        plate = getattr(event, "normalized_plate", None) if event is not None else None
        if plate:
            self._plate_input.setText(plate)

    def _event_for_token(self, token: int) -> Optional[DetectionEvent]:
        """Return the surfaced ``DetectionEvent`` for ``token`` (or ``None``)."""
        for entry in self._review_queue.pending:
            if entry.token == token:
                return entry.event
        return None

    # ------------------------------------------------------------------
    # Open Gate (Req 12.4-12.6)
    # ------------------------------------------------------------------
    def _on_open_gate_clicked(self) -> None:
        """Button slot: perform a guard-initiated Open Gate action.

        Requests the gate-open through the ``GateController`` (Req 12.4). On
        success writes a manually-granted Event_Log record with the normalized
        guard plate (Req 12.5) and confirms to the guard. On failure presents an
        error identifying the failed action and marks nothing granted (Req 12.6).
        """
        plate_text = self._plate_input.text().strip()

        # Resolve which guest this grant is for. The primary flow is the guest
        # currently at the gate (auto-filled by ``on_detection``); whatever plate
        # is in the input -- the auto-filled best read or the guard's correction
        # -- is what gets recorded. If no current guest is set, fall back to a
        # selected review-queue entry (legacy path).
        guest_event = self._current_guest_event
        selected_token: Optional[int] = None
        if guest_event is None:
            selected_item = self._review_list.currentItem()
            if selected_item is not None:
                raw_token = selected_item.data(Qt.UserRole)
                if raw_token is not None:
                    selected_token = int(raw_token)
                    guest_event = self._event_for_token(selected_token)

        # A guest grant (current guest or selected entry) is tagged GUEST; a
        # bare manual open with no guest context stays untagged.
        classification = (
            Classification.GUEST if guest_event is not None else None
        )
        # A guard-permitted Exit_Anomaly records an OUTBOUND EXIT row (Req 5.4).
        is_exit = bool(getattr(guest_event, "is_exit_anomaly", False))

        result = self._grant_service.record_manual_grant(
            plate_text or None,
            frame=self._current_frame,
            classification=classification,
            is_exit=is_exit,
            direction=getattr(guest_event, "direction", None),
            camera_id=getattr(guest_event, "camera_id", None),
        )
        self.last_grant_result = result

        if not result.gate_success:
            # Req 12.6: error identifying the failed action; not marked granted.
            self._status_label.setText(f"ERROR: Open Gate failed. {result.detail}")
            self.show_error(f"Open Gate failed. {result.detail}")
        elif not result.recorded:
            self._status_label.setText(f"ERROR: {result.detail}")
            self.show_error(result.detail)
        else:
            if result.record is not None:
                self.show_db_insert(result.record)
            granted_plate = result.record.normalized_plate if result.record else ""
            suffix = f" (plate {granted_plate})" if granted_plate else ""
            self._status_label.setText(f"Gate opened; manual grant recorded{suffix}.")
            self._add_activity(f"{granted_plate or '-'}   Granted by guard")
            record = result.record
            if (
                self._on_guest_entry is not None
                and record is not None
                and record.event_kind is EventKind.ENTRY
                and record.normalized_plate
            ):
                try:
                    self._on_guest_entry(record.normalized_plate)
                except Exception:  # noqa: BLE001 - a callback fault must not crash the UI
                    logger.error("on_guest_entry callback failed.", exc_info=True)
            # One row written for this guest: resolve its track so the pipeline
            # stops re-surfacing/re-processing it, drop any queue entry, and
            # clear the input so the next guest starts clean (decision 2a). The
            # vehicle now has access and will drive through.
            self._finish_guest_grant(guest_event, selected_token)

    def _finish_guest_grant(
        self, guest_event: Optional[DetectionEvent], selected_token: Optional[int]
    ) -> None:
        """Clean up after a successful guest grant.

        Marks the guest's track resolved via ``on_resolve_track`` (so the
        pipeline skips it thereafter), removes any matching review-queue entry,
        clears the current-guest reference, and resets the plate input so no
        stale plate lingers for the next vehicle.
        """
        track_id = (
            getattr(guest_event, "track_id", None) if guest_event is not None else None
        )
        if track_id is not None and self._on_resolve_track is not None:
            try:
                self._on_resolve_track(int(track_id))
            except Exception:  # noqa: BLE001 - a callback fault must not crash the UI
                logger.error(
                    "on_resolve_track callback failed for track %s.",
                    track_id,
                    exc_info=True,
                )

        # Remove a matching queue entry if one exists (legacy selection path, or
        # the same guest that was also surfaced into the queue).
        token = selected_token
        if token is None and track_id is not None:
            token = self._token_for_track(track_id)
        if token is not None:
            self.resolve_event(token)

        self._current_guest_event = None
        self._clear_plate_input()

    def _token_for_track(self, track_id: int) -> Optional[int]:
        """Return the review-queue token whose event has ``track_id`` (or None)."""
        for entry in self._review_queue.pending:
            if getattr(entry.event, "track_id", None) == track_id:
                return entry.token
        return None

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override name
        """Stop the feed timer on window close."""
        self.stop_feed()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Display helpers (pure)
# ---------------------------------------------------------------------------
def _classification_tag(classification: Optional[Classification]) -> str:
    """Render the Resident/Guest tag for the detection summary (Req 12.2)."""
    if classification == Classification.RESIDENT:
        return "Resident"
    if classification == Classification.GUEST:
        return "Guest"
    return "Unknown"


def _format_timestamp(timestamp: Optional[datetime]) -> str:
    """Render a detection timestamp for display (Req 12.2)."""
    if timestamp is None:
        return "-"
    return timestamp.isoformat()


def _review_label(event: DetectionEvent, reason: Optional[str]) -> str:
    """Build a concise one-line label for a manual-review queue entry.

    An Exit_Anomaly -- a vehicle leaving with no matching entry ticket -- is
    prefixed with a distinct ``[EXIT - NO TICKET]`` tag so the guard immediately
    sees it differs from a normal guest entry (dual-camera-exit-correlation
    Req 5.3).
    """
    plate = event.normalized_plate or event.ocr_text or "(no plate)"
    text = reason if reason is not None else getattr(event, "manual_review_reason", None)
    base = f"{plate} - {text}" if text else str(plate)
    if getattr(event, "is_exit_anomaly", False):
        return f"[EXIT - NO TICKET] {base}"
    return base
