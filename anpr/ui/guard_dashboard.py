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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

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
    EntryState,
    EnvironmentLabel,
    EventKind,
    EventRecord,
    GrantMethod,
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
        QGroupBox,
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
        id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
        clock_iso: Callable[[], str] = _now_iso,
    ) -> None:
        self._gate = gate_controller
        self._event_log = event_log_repo
        self._normalizer = normalizer
        self._image_store = image_store
        self._environment_label = environment_label
        self._id_factory = id_factory
        self._clock_iso = clock_iso

    def record_manual_grant(
        self,
        plate: Optional[str],
        *,
        frame: Optional[Frame] = None,
        event_id: Optional[str] = None,
    ) -> ManualGrantResult:
        """Request the gate-open and, on success, record the manual grant.

        Requirement 12.4: request the gate-open through the ``GateController``.
        Requirement 12.6: on gate failure, return a failure result and write no
        manually-granted record. Requirement 12.5 / Property 26: on gate
        success, write exactly one Event_Log record marked ``grant_method =
        MANUAL`` whose guard-corrected plate value is the ``PlateNormalizer``
        output of the guard input where a plate was provided (empty otherwise).
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

        image_ref = self._maybe_capture(frame, eid)

        record = EventRecord(
            id=eid,
            timestamp=self._clock_iso(),
            ocr_plate="",
            guard_plate=normalized_plate,
            normalized_plate=normalized_plate,
            classification=None,
            direction=None,
            grant_method=GrantMethod.MANUAL,
            event_kind=EventKind.MANUAL,
            entry_state=EntryState.NA,
            image_ref=image_ref,
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

        return ManualGrantResult(
            gate_success=True,
            recorded=True,
            detail=gate_result.detail or "Gate opened; manual grant recorded.",
            record=record,
        )

    def _maybe_capture(self, frame: Optional[Frame], event_id: str) -> Optional[str]:
        """Capture and store an image for the grant when a store/frame exist.

        A capture failure never blocks the grant: the record is kept without an
        image reference (Req 11.7).
        """
        if frame is None or self._image_store is None:
            return None
        try:
            ref = self._image_store.capture_and_store(frame, event_id)
        except Exception as exc:  # noqa: BLE001 - capture must not crash the UI
            logger.error("Manual-grant image capture failed for event %s: %s", event_id, exc)
            return None
        return ref.snapshot_path if ref is not None else None


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
        environment_label: Optional[EnvironmentLabel] = None,
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

        # Pure collaborators (testable without a GUI).
        self._review_queue = ManualReviewQueue()
        self._grant_service = ManualGrantService(
            gate_controller,
            event_log_repo,
            normalizer,
            image_store=image_store,
            environment_label=environment_label,
        )

        # Most recent frame/detections, reused when capturing a manual-grant
        # image on Open Gate.
        self._current_frame: Optional[Frame] = None
        self._current_detections: list = []

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
        """Assemble the three-panel layout (left feed, top/bottom right)."""
        central = QWidget(self)
        root = QHBoxLayout(central)

        # --- Left panel: live feed with overlays (Req 12.1) ----------------
        left_panel = QGroupBox("Live Feed", central)
        left_layout = QVBoxLayout(left_panel)
        self._feed_label = QLabel("No feed", left_panel)
        self._feed_label.setAlignment(Qt.AlignCenter)
        self._feed_label.setMinimumSize(480, 360)
        self._feed_label.setScaledContents(False)
        left_layout.addWidget(self._feed_label)
        root.addWidget(left_panel, 2)

        # --- Right column: split into top-right and bottom-right -----------
        right_column = QVBoxLayout()

        # Top-right: latest detection summary (Req 12.2).
        detection_panel = QGroupBox("Latest Detection", central)
        detection_layout = QVBoxLayout(detection_panel)
        self._plate_value = QLabel("Plate: -", detection_panel)
        self._tag_value = QLabel("Tag: -", detection_panel)
        self._timestamp_value = QLabel("Time: -", detection_panel)
        detection_layout.addWidget(self._plate_value)
        detection_layout.addWidget(self._tag_value)
        detection_layout.addWidget(self._timestamp_value)
        right_column.addWidget(detection_panel, 1)

        # Bottom-right: manual controls + manual-review queue (Req 12.3, 12.7).
        control_panel = QGroupBox("Manual Control", central)
        control_layout = QVBoxLayout(control_panel)

        self._plate_input = QLineEdit(control_panel)
        self._plate_input.setPlaceholderText("Enter plate for manual grant")
        # Req 12.3: accept up to the configured maximum character length.
        self._plate_input.setMaxLength(self._guard_config.manual_plate_max_len)
        control_layout.addWidget(self._plate_input)

        self._open_gate_button = QPushButton("Open Gate", control_panel)
        self._open_gate_button.clicked.connect(self._on_open_gate_clicked)
        control_layout.addWidget(self._open_gate_button)

        self._status_label = QLabel("", control_panel)
        self._status_label.setWordWrap(True)
        control_layout.addWidget(self._status_label)

        control_layout.addWidget(QLabel("Manual-review queue:", control_panel))
        self._review_list = QListWidget(control_panel)
        control_layout.addWidget(self._review_list, 1)

        self._resolve_button = QPushButton("Resolve selected", control_panel)
        self._resolve_button.clicked.connect(self._on_resolve_clicked)
        control_layout.addWidget(self._resolve_button)

        right_column.addWidget(control_panel, 2)
        root.addLayout(right_column, 1)

        self.setCentralWidget(central)

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
        if self._frame_provider is None:
            return
        try:
            payload = self._frame_provider()
        except Exception as exc:  # noqa: BLE001 - a provider fault must not crash the UI
            logger.error("Live-feed frame provider failed: %s", exc)
            return
        if payload is None:
            return
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
        """Update the top-right panel for a new detection within the budget.

        Shows the detected plate number, the Resident/Guest tag, and the
        detection timestamp (Req 12.2). Logs when the update exceeds the
        configured maximum update latency.
        """
        started = time.monotonic()

        plate = event.normalized_plate or event.ocr_text or "-"
        self._plate_value.setText(f"Plate: {plate}")
        self._tag_value.setText(f"Tag: {_classification_tag(event.classification)}")
        self._timestamp_value.setText(f"Time: {_format_timestamp(event.timestamp)}")

        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms > self._guard_config.max_update_latency_ms:
            logger.warning(
                "Detection panel update took %.0f ms, exceeding the %d ms budget.",
                elapsed_ms,
                self._guard_config.max_update_latency_ms,
            )

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
        result = self._grant_service.record_manual_grant(
            plate_text or None, frame=self._current_frame
        )
        self.last_grant_result = result

        if not result.gate_success:
            # Req 12.6: error identifying the failed action; not marked granted.
            self._status_label.setText(f"ERROR: Open Gate failed. {result.detail}")
        elif not result.recorded:
            self._status_label.setText(f"ERROR: {result.detail}")
        else:
            granted_plate = result.record.normalized_plate if result.record else ""
            suffix = f" (plate {granted_plate})" if granted_plate else ""
            self._status_label.setText(f"Gate opened; manual grant recorded{suffix}.")

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
    """Build a concise one-line label for a manual-review queue entry."""
    plate = event.normalized_plate or event.ocr_text or "(no plate)"
    text = reason if reason is not None else getattr(event, "manual_review_reason", None)
    return f"{plate} - {text}" if text else str(plate)
