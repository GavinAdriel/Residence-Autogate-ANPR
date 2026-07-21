"""Detection pipeline: the ingest -> decision loop of the ANPR Autogate System.

:class:`DetectionPipeline` drives the whole ingest-to-decision flow at the
configured frame rate. It is the orchestration seam that wires together the
thin :class:`~anpr.core.interfaces.VideoSource`, the ``VehicleDetector``, the
``OcrEngine``, the ``PlateNormalizer``, the ``DirectionResolver`` and the
``AccessController`` -- depending only on their Protocol interfaces so mock and
field implementations stay structurally interchangeable and are selected purely
by configuration in the composition root (Requirement 14.4).

Responsibilities implemented here (task 12.2):

* **Frame-rate throttling** -- limit ingestion to the configured target frame
  rate between 1 and 60 fps (Req 1.1).
* **Connect with retry** -- open the source, and on a
  :class:`~anpr.pipeline.video_source.SourceUnavailable` failure log the
  offending source and retry at the configurable 1-60 s reconnect interval
  (Req 1.4).
* **Reconnect on interruption / inactivity** -- when a connected source stops
  yielding frames, or no new frame arrives within the configurable 1-30 s
  inactivity timeout, treat the source as a connection failure and reconnect
  using the same log-and-retry behavior (Req 1.5, 1.6).
* **One crop per retained box** -- crop exactly one vehicle/plate region for
  each retained detection box and pass each crop to the OCR engine (Req 3.1).
* **OCR gating** -- on no readable text or an OCR timeout, record an OCR
  failure, retain the crop, and surface the event for manual entry (Req 3.3,
  3.5); mark a reading whose confidence is strictly below the OCR threshold
  (default 0.70) as low-confidence and surface it for manual confirmation
  (Req 3.4).
* **Format gating** -- when normalization marks a plate format-invalid, surface
  the raw text together with the rejection reason for manual confirmation
  (Req 4.5).
* **Routing** -- route a format-valid, confident, inbound detection through the
  direction resolver into the ``AccessController``; surface direction-
  undetermined events for manual resolution (Req 7.4).
* **Event assembly** -- build a :class:`~anpr.core.models.DetectionEvent`
  accumulating the detection and OCR confidences and the end-to-end processing
  latency (frame acquisition -> access decision) (design Event flow, Req 15.1).

Testability: the monotonic clock, the wall-clock, the ``sleep`` function, and
the OCR-timeout invoker are all injectable, so throttling, the reconnect
interval, the inactivity timeout, and OCR-timeout handling can be exercised
deterministically with a fake clock and a mock :class:`VideoSource` without any
real camera, model, or wall-clock delay. Events surfaced for manual handling are
exposed both via an injectable callback and via the
:attr:`DetectionPipeline.manual_review_queue`, so the Guard_Dashboard task can
consume them (Req 3.3, 3.4, 3.5, 4.5, 7.4).

See .kiro/specs/anpr-autogate-system/design.md (DetectionPipeline section) and
requirements.md (Requirements 1, 3) for the authoritative acceptance criteria.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Optional

from anpr.core.access_controller import AccessController
from anpr.core.interfaces import (
    OcrEngine,
    VehicleDetector,
    VideoSource,
)
from anpr.core.models import (
    BoundingBox,
    Detection,
    DetectionEvent,
    DirectionOutcome,
    OcrResult,
    Point,
    TrackHistory,
)
from anpr.core.normalizer import PlateNormalizer
from anpr.direction.resolver import DirectionResolver
from anpr.pipeline.video_source import SourceUnavailable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults and bounds (design Configuration schema)
# ---------------------------------------------------------------------------

# Target frame rate is configurable in the inclusive range 1..60 fps (Req 1.1).
DEFAULT_TARGET_FPS = 15
MIN_TARGET_FPS = 1
MAX_TARGET_FPS = 60

# Reconnect interval is configurable in the inclusive range 1..60 s (Req 1.4/1.5).
DEFAULT_RECONNECT_INTERVAL_S = 5.0
MIN_RECONNECT_INTERVAL_S = 1.0
MAX_RECONNECT_INTERVAL_S = 60.0

# Inactivity timeout is configurable in the inclusive range 1..30 s (Req 1.6).
DEFAULT_INACTIVITY_TIMEOUT_S = 10.0
MIN_INACTIVITY_TIMEOUT_S = 1.0
MAX_INACTIVITY_TIMEOUT_S = 30.0

# OCR confidence threshold is configurable in the inclusive range 0..1,
# defaulting to 0.70 (Req 3.4).
DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0.70
MIN_OCR_CONFIDENCE_THRESHOLD = 0.0
MAX_OCR_CONFIDENCE_THRESHOLD = 1.0

# OCR processing timeout, defaulting to 2000 ms (Req 3.5).
DEFAULT_OCR_TIMEOUT_MS = 2000


# Human-readable reasons attached to events surfaced for manual handling. Kept
# as constants so the Guard_Dashboard task and tests can match them exactly.
REVIEW_OCR_NO_TEXT = "OCR produced no readable text; surfaced for manual entry."
REVIEW_OCR_TIMEOUT = "OCR timed out; surfaced for manual entry."
REVIEW_OCR_LOW_CONFIDENCE = (
    "OCR recognition confidence is below the configured threshold; surfaced "
    "for manual confirmation."
)
REVIEW_FORMAT_INVALID = "Plate is format-invalid; surfaced for manual confirmation."
REVIEW_DIRECTION_UNDETERMINED = (
    "Travel direction is undetermined; surfaced for manual resolution."
)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def _default_ocr_invoker(engine: OcrEngine, crop: Any, timeout_s: float) -> OcrResult:
    """Invoke ``engine.read_plate`` under a wall-clock timeout (Req 3.5).

    Runs the (potentially slow) OCR read on a daemon worker thread and waits up
    to ``timeout_s`` seconds for it. When the worker does not finish in time, a
    timed-out :class:`OcrResult` is returned so the pipeline records an OCR
    failure and surfaces the event for manual entry; the abandoned worker is a
    daemon so it never blocks shutdown. Injectable so tests can supply a
    deterministic invoker with no real threads or delays.
    """
    holder: dict[str, OcrResult] = {}

    def _target() -> None:
        holder["result"] = engine.read_plate(crop)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout_s if timeout_s > 0 else None)
    if worker.is_alive():
        return OcrResult(text=None, confidence=0.0, timed_out=True)
    return holder.get("result", OcrResult(text=None, confidence=0.0, timed_out=True))


class DetectionPipeline:
    """Drives the ingest -> detect -> read -> normalize -> decide loop.

    Parameters
    ----------
    video_source:
        The frame source (webcam or IP camera) behind the ``VideoSource``
        Protocol; the pipeline owns the retry/reconnect loop and the inactivity
        watchdog around it (Req 1.4-1.6).
    detector:
        The vehicle detector wrapping the YOLO26 weights.
    ocr_engine:
        The OCR engine wrapping PaddleOCR.
    normalizer:
        The pure plate normalizer.
    direction_resolver:
        The configured direction resolver (single-camera trajectory this phase).
    access_controller:
        The access-control orchestrator that classifies, drives the gate, and
        logs the access attempt.
    target_fps, reconnect_interval_s, inactivity_timeout_s,
    ocr_confidence_threshold, ocr_timeout_ms:
        Tuning values (clamped to their valid ranges); normally supplied from
        configuration via :meth:`from_config`.
    on_manual_review:
        Optional callback invoked with each event surfaced for manual handling
        (Req 3.3, 3.4, 3.5, 4.5, 7.4). Every such event is also appended to
        :attr:`manual_review_queue`.
    on_event:
        Optional callback invoked with every fully assembled
        :class:`DetectionEvent` (surfaced or routed) for the live feed.
    monotonic, wall_clock, sleep, ocr_invoker:
        Injectable time / sleep / OCR-timeout seams for deterministic testing.
    """

    def __init__(
        self,
        video_source: VideoSource,
        detector: VehicleDetector,
        ocr_engine: OcrEngine,
        normalizer: PlateNormalizer,
        direction_resolver: DirectionResolver,
        access_controller: AccessController,
        *,
        target_fps: float = DEFAULT_TARGET_FPS,
        reconnect_interval_s: float = DEFAULT_RECONNECT_INTERVAL_S,
        inactivity_timeout_s: float = DEFAULT_INACTIVITY_TIMEOUT_S,
        ocr_confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
        ocr_timeout_ms: float = DEFAULT_OCR_TIMEOUT_MS,
        on_manual_review: Optional[Callable[[DetectionEvent], None]] = None,
        on_event: Optional[Callable[[DetectionEvent], None]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], None] = time.sleep,
        ocr_invoker: Callable[[OcrEngine, Any, float], OcrResult] = _default_ocr_invoker,
    ) -> None:
        self._source = video_source
        self._detector = detector
        self._ocr = ocr_engine
        self._normalizer = normalizer
        self._direction = direction_resolver
        self._access = access_controller

        self._target_fps = _clamp(float(target_fps), MIN_TARGET_FPS, MAX_TARGET_FPS)
        self._frame_interval_s = 1.0 / self._target_fps
        self._reconnect_interval_s = _clamp(
            float(reconnect_interval_s), MIN_RECONNECT_INTERVAL_S, MAX_RECONNECT_INTERVAL_S
        )
        self._inactivity_timeout_s = _clamp(
            float(inactivity_timeout_s), MIN_INACTIVITY_TIMEOUT_S, MAX_INACTIVITY_TIMEOUT_S
        )
        self._ocr_confidence_threshold = _clamp(
            float(ocr_confidence_threshold),
            MIN_OCR_CONFIDENCE_THRESHOLD,
            MAX_OCR_CONFIDENCE_THRESHOLD,
        )
        self._ocr_timeout_s = max(0.0, float(ocr_timeout_ms) / 1000.0)

        self._on_manual_review = on_manual_review
        self._on_event = on_event

        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._ocr_invoker = ocr_invoker

        # Loop / connection state.
        self._running = False
        self._connected = False
        self._last_activity: Optional[float] = None
        self._last_cycle: Optional[float] = None

        # Per-track centroid history for the single-camera trajectory resolver.
        self._track_histories: dict[int, TrackHistory] = {}

        # Events awaiting a guard decision; retained until the guard resolves
        # them (Req 12.7 -- surfacing side implemented here).
        self.manual_review_queue: Deque[DetectionEvent] = deque()

        # Most recently processed frame + detections, exposed to the
        # Guard_Dashboard live feed via :meth:`latest_feed` (Req 12.1). Written
        # on the pipeline thread and read on the GUI thread, so guarded by a
        # lock; the frame reference is swapped wholesale (cheap under the GIL).
        self._feed_lock = threading.Lock()
        self._latest_frame: Optional[Any] = None
        self._latest_detections: list[Detection] = []

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: Any,
        video_source: VideoSource,
        detector: VehicleDetector,
        ocr_engine: OcrEngine,
        normalizer: PlateNormalizer,
        direction_resolver: DirectionResolver,
        access_controller: AccessController,
        **overrides: Any,
    ) -> "DetectionPipeline":
        """Build a pipeline from a ``ConfigProvider`` and injected collaborators.

        Reads ``camera.target_fps`` (Req 1.1), ``camera.reconnect_interval_s``
        (Req 1.4/1.5), ``camera.inactivity_timeout_s`` (Req 1.6),
        ``ocr.confidence_threshold`` (Req 3.4) and ``ocr.timeout_ms`` (Req 3.5),
        falling back to the documented defaults when a key is absent. Any keyword
        override is forwarded to the constructor (e.g. injectable clocks in
        tests).
        """
        return cls(
            video_source,
            detector,
            ocr_engine,
            normalizer,
            direction_resolver,
            access_controller,
            target_fps=_cfg(config, "camera.target_fps", DEFAULT_TARGET_FPS),
            reconnect_interval_s=_cfg(
                config, "camera.reconnect_interval_s", DEFAULT_RECONNECT_INTERVAL_S
            ),
            inactivity_timeout_s=_cfg(
                config, "camera.inactivity_timeout_s", DEFAULT_INACTIVITY_TIMEOUT_S
            ),
            ocr_confidence_threshold=_cfg(
                config, "ocr.confidence_threshold", DEFAULT_OCR_CONFIDENCE_THRESHOLD
            ),
            ocr_timeout_ms=_cfg(config, "ocr.timeout_ms", DEFAULT_OCR_TIMEOUT_MS),
            **overrides,
        )

    # ------------------------------------------------------------------
    # Introspection (used by tests and the composition root)
    # ------------------------------------------------------------------
    @property
    def target_fps(self) -> float:
        """Effective (clamped) target frame rate in fps."""
        return self._target_fps

    @property
    def reconnect_interval_s(self) -> float:
        """Effective (clamped) reconnect interval in seconds."""
        return self._reconnect_interval_s

    @property
    def inactivity_timeout_s(self) -> float:
        """Effective (clamped) inactivity timeout in seconds."""
        return self._inactivity_timeout_s

    @property
    def ocr_confidence_threshold(self) -> float:
        """Effective (clamped) OCR confidence threshold in the 0..1 range."""
        return self._ocr_confidence_threshold

    @property
    def is_running(self) -> bool:
        """Whether the ingest loop is currently running."""
        return self._running

    def latest_feed(self) -> Optional[tuple[Any, list]]:
        """Return the most recent ``(frame, detections)`` for the live feed.

        A thread-safe snapshot consumed by the Guard_Dashboard's frame provider
        (Req 12.1). Returns ``None`` until the first frame has been processed.
        The detections list is copied so the caller can iterate it without
        racing the pipeline thread.
        """
        with self._feed_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame, list(self._latest_detections)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Run the ingest loop until :meth:`stop` is called.

        Blocks the calling thread, repeatedly running one loop iteration. The
        composition root runs this on a dedicated thread; tests drive
        :meth:`run_once` (or :meth:`process_frame`) directly instead.
        """
        self._running = True
        try:
            while self._running:
                self.run_once()
        finally:
            self._teardown()

    def run(self, max_iterations: int) -> None:
        """Run a bounded number of loop iterations (deterministic for tests)."""
        self._running = True
        try:
            for _ in range(max_iterations):
                if not self._running:
                    break
                self.run_once()
        finally:
            self._teardown()

    def stop(self) -> None:
        """Signal the ingest loop to stop after the current iteration."""
        self._running = False

    def _teardown(self) -> None:
        """Release the source and reset connection state on loop exit."""
        try:
            self._source.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            logger.debug("Ignoring error while closing the video source.", exc_info=True)
        self._connected = False

    # ------------------------------------------------------------------
    # One loop iteration
    # ------------------------------------------------------------------
    def run_once(self) -> list[DetectionEvent]:
        """Run a single ingest iteration and return the events it produced.

        Ensures the source is connected (retrying on failure, Req 1.4), throttles
        to the target frame rate (Req 1.1), reads one frame, applies the
        inactivity / interruption watchdog (Req 1.5, 1.6), and -- when a frame is
        available -- processes it into zero or more :class:`DetectionEvent`.
        """
        self._ensure_connected()
        if not self._running or not self._connected:
            return []

        self._throttle()

        try:
            frame = self._source.read()
        except Exception as exc:  # noqa: BLE001 - a read fault is an interruption
            logger.error(
                "Video source '%s' read failed; treating as interruption: %s",
                self._descriptor(),
                exc,
            )
            self._handle_disconnect("read fault")
            return []

        acquired_at = self._monotonic()

        if frame is None:
            self._handle_no_frame(acquired_at)
            return []

        # A frame arrived: reset the inactivity watchdog and process it.
        self._last_activity = acquired_at
        return self.process_frame(frame, acquired_at)

    # ------------------------------------------------------------------
    # Connection management (Req 1.4, 1.5, 1.6)
    # ------------------------------------------------------------------
    def _ensure_connected(self) -> None:
        """Open the source, retrying on failure at the reconnect interval.

        On a :class:`SourceUnavailable` the offending source is logged and the
        pipeline waits the configurable 1-60 s reconnect interval before
        retrying, until it connects or the pipeline is stopped (Req 1.4).
        """
        while self._running and not self._connected:
            try:
                self._source.open()
            except SourceUnavailable as exc:
                logger.error(
                    "Failed to open video source '%s'; retrying in %.0f s: %s",
                    self._descriptor(),
                    self._reconnect_interval_s,
                    exc,
                )
                self._sleep(self._reconnect_interval_s)
                continue
            except Exception as exc:  # noqa: BLE001 - any open fault is unavailability
                logger.error(
                    "Unexpected error opening video source '%s'; retrying in "
                    "%.0f s: %s",
                    self._descriptor(),
                    self._reconnect_interval_s,
                    exc,
                )
                self._sleep(self._reconnect_interval_s)
                continue

            self._connected = True
            now = self._monotonic()
            self._last_activity = now
            # Reset throttle so the first post-connect frame is read promptly.
            self._last_cycle = None
            logger.info("Connected to video source '%s'.", self._descriptor())

    def _handle_no_frame(self, now: float) -> None:
        """Apply the inactivity / interruption watchdog for a missing frame.

        When no frame has arrived for longer than the configurable inactivity
        timeout, the connected source is treated as a connection failure and the
        pipeline reconnects using the same log-and-retry behavior as an initial
        open failure (Req 1.5, 1.6).
        """
        last = self._last_activity if self._last_activity is not None else now
        if now - last >= self._inactivity_timeout_s:
            self._handle_disconnect(
                f"no frame within inactivity timeout of "
                f"{self._inactivity_timeout_s:.0f} s"
            )

    def _handle_disconnect(self, reason: str) -> None:
        """Log an interruption, release the source, and mark it disconnected.

        The next :meth:`run_once` will re-enter :meth:`_ensure_connected` and
        reconnect at the configured interval (Req 1.5, 1.6).
        """
        logger.error(
            "Video source '%s' interrupted (%s); will reconnect in %.0f s.",
            self._descriptor(),
            reason,
            self._reconnect_interval_s,
        )
        try:
            self._source.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            logger.debug("Ignoring error while closing the video source.", exc_info=True)
        self._connected = False
        self._last_activity = None
        self._sleep(self._reconnect_interval_s)

    def _throttle(self) -> None:
        """Sleep as needed to hold ingestion at the target frame rate (Req 1.1)."""
        now = self._monotonic()
        if self._last_cycle is not None:
            elapsed = now - self._last_cycle
            remaining = self._frame_interval_s - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last_cycle = self._monotonic()

    # ------------------------------------------------------------------
    # Frame processing (Req 3.1, 3.3, 3.4, 3.5, 4.5, 7.x)
    # ------------------------------------------------------------------
    def process_frame(self, frame: Any, acquired_at: float) -> list[DetectionEvent]:
        """Detect, crop, read, normalize, resolve, and route one frame.

        Runs the detector, then for each retained detection box crops exactly
        one region (Req 3.1), reads it with the OCR engine under the OCR timeout
        (Req 3.5), and gates the result through OCR-failure / low-confidence /
        format-invalid / direction handling, assembling a
        :class:`DetectionEvent` per detection with accumulated confidences and
        end-to-end latency. An empty frame yields no events (Req 2.3).
        """
        detections = self._detector.detect(frame)

        # Publish the frame + detections for the live feed on every processed
        # frame (even with no detections) so the Guard_Dashboard shows a
        # continuous feed, not just detection moments (Req 12.1).
        with self._feed_lock:
            self._latest_frame = frame
            self._latest_detections = list(detections)

        events: list[DetectionEvent] = []
        for detection in detections:
            events.append(self._process_detection(frame, detection, acquired_at))
        return events

    def _process_detection(
        self, frame: Any, detection: Detection, acquired_at: float
    ) -> DetectionEvent:
        """Process one retained detection into a routed / surfaced event."""
        event = DetectionEvent(
            timestamp=self._wall_clock(),
            acquired_at=self._wall_clock_from_monotonic(acquired_at),
            box=detection.box,
            detection_confidence=detection.confidence,
            track_id=detection.track_id,
        )

        # Req 3.1: crop exactly one region for this box and pass it to the OCR
        # engine. The crop is kept on the event so a surfaced OCR failure retains
        # it for manual entry (Req 3.3).
        crop = self._crop_region(frame, detection.box)
        event.retained_crop = crop  # type: ignore[attr-defined]

        ocr_result = self._ocr_invoker(self._ocr, crop, self._ocr_timeout_s)
        event.ocr_text = ocr_result.text
        event.ocr_confidence = ocr_result.confidence
        event.ocr_timed_out = ocr_result.timed_out

        # Req 3.5: an OCR timeout is an OCR failure surfaced for manual entry.
        if ocr_result.timed_out:
            return self._surface(event, acquired_at, REVIEW_OCR_TIMEOUT)

        # Req 3.3: no readable text is an OCR failure surfaced for manual entry;
        # the crop is retained on the event above.
        if ocr_result.text is None:
            return self._surface(event, acquired_at, REVIEW_OCR_NO_TEXT)

        # Req 3.4: confidence strictly below the OCR threshold is low-confidence,
        # retaining the text and confidence, surfaced for manual confirmation.
        if ocr_result.confidence < self._ocr_confidence_threshold:
            return self._surface(event, acquired_at, REVIEW_OCR_LOW_CONFIDENCE)

        # Normalize the confident reading.
        normalization = self._normalizer.normalize(ocr_result.text)
        event.normalized_plate = normalization.normalized
        event.is_format_valid = normalization.is_valid
        event.normalization_reason = normalization.reason

        # Req 4.5: a format-invalid plate surfaces the raw text + reason.
        if not normalization.is_valid:
            reason = normalization.reason or REVIEW_FORMAT_INVALID
            return self._surface(event, acquired_at, reason)

        # Direction resolution over the accumulated track history.
        track = self._update_track_history(detection)
        event.track = track
        direction = self._direction.resolve(track)
        event.direction = direction

        if direction == DirectionOutcome.INBOUND:
            self._finalize_latency(event, acquired_at)
            decision = self._access.handle_inbound(event)
            self._apply_decision(event, decision)
            self._emit(event)
            return event

        if direction == DirectionOutcome.OUTBOUND:
            # Outbound trajectory resolution is deferred this phase; the
            # access-control logic exists and is exercised when fed an outbound
            # event (design Phase Scope). Route it through so the outbound flow
            # stays wired.
            self._finalize_latency(event, acquired_at)
            decision = self._access.handle_outbound(event)
            self._apply_decision(event, decision)
            self._emit(event)
            return event

        # Req 7.4: direction-undetermined applies no direction-specific rules
        # and is surfaced for manual resolution.
        return self._surface(event, acquired_at, REVIEW_DIRECTION_UNDETERMINED)

    # ------------------------------------------------------------------
    # Track history
    # ------------------------------------------------------------------
    def _update_track_history(self, detection: Detection) -> TrackHistory:
        """Append this detection's centroid to its track history.

        Maintains a per-``track_id`` centroid trail so the single-camera
        trajectory resolver has the consecutive frames it needs (Req 7.3). A
        detection without a tracker id gets a fresh single-point history (which
        the resolver treats as undetermined). The track's confidence is set to
        the latest detection confidence as a proxy alignment weight.
        """
        centroid = _centroid(detection.box)
        track_id = detection.track_id
        if track_id is None:
            return TrackHistory(
                track_id=-1, centroids=[centroid], confidence=detection.confidence
            )

        history = self._track_histories.get(track_id)
        if history is None:
            history = TrackHistory(track_id=track_id, centroids=[])
            self._track_histories[track_id] = history
        history.centroids.append(centroid)
        history.confidence = detection.confidence
        return history

    # ------------------------------------------------------------------
    # Surfacing / emitting
    # ------------------------------------------------------------------
    def _surface(
        self, event: DetectionEvent, acquired_at: float, reason: str
    ) -> DetectionEvent:
        """Mark an event for manual handling and surface it (Req 3.3-4.5, 7.4).

        Finalizes the end-to-end latency, flags the event for the manual-review
        queue, appends it, notifies the optional callback, and emits it on the
        general event sink. Returns the same event for convenience.
        """
        self._finalize_latency(event, acquired_at)
        event.needs_manual_review = True
        event.manual_review_reason = reason
        self.manual_review_queue.append(event)
        if self._on_manual_review is not None:
            self._on_manual_review(event)
        logger.info(
            "Surfaced detection for manual handling (plate=%r): %s",
            event.normalized_plate or event.ocr_text,
            reason,
        )
        self._emit(event)
        return event

    def _apply_decision(self, event: DetectionEvent, decision: Any) -> None:
        """Reflect an ``AccessDecision`` onto the event's review flags.

        When the access controller surfaced the event to the guard (guest, DB
        fault, exit anomaly, or exit write/close fault), mirror that onto the
        pipeline's manual-review queue so the Guard_Dashboard sees a single,
        unified stream of events needing attention.
        """
        event.classification = decision.classification
        if getattr(decision, "surfaced_to_guard", False):
            event.needs_manual_review = True
            event.manual_review_reason = decision.reason
            self.manual_review_queue.append(event)
            if self._on_manual_review is not None:
                self._on_manual_review(event)

    def _emit(self, event: DetectionEvent) -> None:
        """Notify the optional general event sink (live feed)."""
        if self._on_event is not None:
            self._on_event(event)

    # ------------------------------------------------------------------
    # Latency / timing helpers
    # ------------------------------------------------------------------
    def _finalize_latency(self, event: DetectionEvent, acquired_at: float) -> None:
        """Set ``processing_latency_ms`` = decision time - acquisition time.

        Computed from the monotonic clock so it is immune to wall-clock jumps,
        and only set once so a downstream access decision keeps the same value
        (Req 15.1).
        """
        if event.processing_latency_ms is not None:
            return
        elapsed_ms = int(round((self._monotonic() - acquired_at) * 1000.0))
        event.processing_latency_ms = max(0, elapsed_ms)

    def _wall_clock_from_monotonic(self, acquired_monotonic: float) -> datetime:
        """Best-effort wall-clock acquisition time for the event record.

        The pipeline measures latency on the monotonic clock but records a
        wall-clock ``acquired_at`` for auditability; the small offset between the
        two reads is immaterial at one-second timestamp precision (Req 10.3).
        """
        return self._wall_clock()

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _crop_region(self, frame: Any, box: BoundingBox) -> Any:
        """Crop exactly one region for a bounding box (Req 3.1).

        Slices the frame to ``[y1:y2, x1:x2]`` for an ``(H, W, C)`` array. On any
        slicing error (an unusual frame type) the whole frame is returned so the
        OCR engine still receives exactly one region per box rather than raising.
        """
        try:
            return frame[box.y1:box.y2, box.x1:box.x2]
        except Exception:  # noqa: BLE001 - tolerate non-array frames in tests
            return frame

    def _descriptor(self) -> str:
        """Return the source descriptor for logs, tolerating its absence."""
        try:
            return self._source.descriptor
        except Exception:  # noqa: BLE001 - a missing descriptor must not crash logs
            return "<unknown source>"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _centroid(box: BoundingBox) -> Point:
    """Return the integer pixel centroid ``(cx, cy)`` of a bounding box."""
    return ((box.x1 + box.x2) // 2, (box.y1 + box.y2) // 2)


def _cfg(config: Any, key: str, default: Any) -> Any:
    """Read a dotted config key, falling back to ``default`` when absent/None."""
    try:
        value = config.get(key)
    except (KeyError, AttributeError):
        return default
    return default if value is None else value
