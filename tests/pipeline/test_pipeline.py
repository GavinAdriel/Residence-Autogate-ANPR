"""Unit tests for the DetectionPipeline ingest -> decision loop.

These example-based tests exercise the pipeline's throttling, connect/reconnect
behavior, cropping, OCR gating, normalization gating, and routing using injected
fakes (a mock ``VideoSource``, detector, OCR engine, and access controller) and
a fake clock, so no real camera, model, or wall-clock delay is needed.

Direction resolution is retired in the monitoring migration: a confident,
format-valid reading is routed directly through
:meth:`AccessController.handle_detection` (Req 5.1), with no direction resolver
in the pipeline.

Covers Requirements 1.1 (throttle), 1.4/1.5/1.6 (connect/reconnect/inactivity),
3.1 (one crop per box), 3.3/3.5 (OCR failure surfacing), 3.4 (low-confidence
gating), 4.5 (format-invalid surfacing), 5.1 (flat match-and-log routing),
7.5/7.6/7.7 (surfacing retention + reconnect/inactivity), and the end-to-end
latency assembly.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from anpr.core.models import (
    AccessDecision,
    BoundingBox,
    Classification,
    Detection,
    GrantMethod,
    OcrResult,
)
from anpr.core.normalizer import PlateNormalizer
from anpr.pipeline.pipeline import (
    DetectionPipeline,
    REVIEW_FORMAT_INVALID,
    REVIEW_OCR_LOW_CONFIDENCE,
    REVIEW_OCR_NO_TEXT,
    REVIEW_OCR_TIMEOUT,
)
from anpr.pipeline.video_source import SourceUnavailable


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances virtual time."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self.slept.append(secs)
        self.t += secs


class FakeSource:
    """Mock VideoSource with scripted open results and frame reads."""

    def __init__(self, *, open_errors: int = 0, frames=None) -> None:
        self._open_errors = open_errors
        self._frames = list(frames) if frames is not None else []
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1
        if self.open_calls <= self._open_errors:
            raise SourceUnavailable(self.descriptor)

    def read(self):
        if not self._frames:
            return None
        return self._frames.pop(0)

    def close(self) -> None:
        self.close_calls += 1

    @property
    def descriptor(self) -> str:
        return "fake:source"


class FakeDetector:
    """Detector returning a fixed list of detections for any frame."""

    def __init__(self, detections=None) -> None:
        self._detections = list(detections) if detections is not None else []
        self.frames_seen: list = []

    def load(self) -> None:  # pragma: no cover - not used here
        pass

    def detect(self, frame):
        self.frames_seen.append(frame)
        return list(self._detections)


class FakeOcr:
    """OCR engine returning a scripted result and recording each crop."""

    def __init__(self, result: OcrResult) -> None:
        self._result = result
        self.crops: list = []

    def read_plate(self, crop) -> OcrResult:
        self.crops.append(crop)
        return self._result


class FakeAccessController:
    """Access controller recording flat ``handle_detection`` calls."""

    def __init__(self, decision: AccessDecision | None = None) -> None:
        self._decision = decision or AccessDecision(
            classification=Classification.RESIDENT,
            grant_method=GrantMethod.AUTOMATIC,
            gate_requested=True,
        )
        self.detection_events: list = []

    def handle_detection(self, ev) -> AccessDecision:
        self.detection_events.append(ev)
        return self._decision


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


# Sentinel: tell make_pipeline to use the pipeline's real (thread-based) OCR
# invoker instead of the default synchronous test invoker.
USE_REAL_OCR_INVOKER = object()

# Marker meaning "caller did not specify an invoker" -> use synchronous invoker.
_NOT_PROVIDED = object()


def make_pipeline(
    *,
    source=None,
    detector=None,
    ocr=None,
    access=None,
    clock=None,
    ocr_invoker=_NOT_PROVIDED,
    **overrides,
):
    clock = clock or FakeClock()
    source = source or FakeSource()
    detector = detector or FakeDetector()
    ocr = ocr or FakeOcr(OcrResult(text="B1234CD", confidence=0.9))
    access = access or FakeAccessController()

    kwargs = dict(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        wall_clock=lambda: __import__("datetime").datetime(2024, 1, 1),
        **overrides,
    )
    if ocr_invoker is _NOT_PROVIDED:
        # Default to a synchronous OCR invoker so tests need no real threads.
        kwargs["ocr_invoker"] = lambda engine, crop, timeout_s: engine.read_plate(crop)
    elif ocr_invoker is not USE_REAL_OCR_INVOKER:
        kwargs["ocr_invoker"] = ocr_invoker
    # else: leave ocr_invoker unset so the pipeline uses its real default.

    pipeline = DetectionPipeline(
        source,
        detector,
        ocr,
        PlateNormalizer(),
        access,
        **kwargs,
    )
    return pipeline, clock, source, detector, ocr, access


def a_frame(width=200, height=100):
    """A distinct HxWxC frame whose pixels encode their coordinates."""
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


# ---------------------------------------------------------------------------
# Config clamping / from_config
# ---------------------------------------------------------------------------


class DictConfig:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, key):
        return self._m[key]


@pytest.mark.unit
def test_from_config_reads_camera_and_ocr_settings():
    config = DictConfig(
        {
            "camera.target_fps": 30,
            "camera.reconnect_interval_s": 7,
            "camera.inactivity_timeout_s": 12,
            "ocr.confidence_threshold": 0.8,
            "ocr.timeout_ms": 1500,
        }
    )
    pipeline = DetectionPipeline.from_config(
        config,
        FakeSource(),
        FakeDetector(),
        FakeOcr(OcrResult(text="B1CD", confidence=0.9)),
        PlateNormalizer(),
        FakeAccessController(),
    )
    assert pipeline.target_fps == 30
    assert pipeline.reconnect_interval_s == 7
    assert pipeline.inactivity_timeout_s == 12
    assert pipeline.ocr_confidence_threshold == 0.8


@pytest.mark.unit
def test_config_values_are_clamped_to_valid_ranges():
    pipeline, *_ = make_pipeline(
        target_fps=999,
        reconnect_interval_s=0.1,
        inactivity_timeout_s=100,
        ocr_confidence_threshold=5.0,
    )
    assert pipeline.target_fps == 60  # clamped to max
    assert pipeline.reconnect_interval_s == 1.0  # clamped to min
    assert pipeline.inactivity_timeout_s == 30.0  # clamped to max
    assert pipeline.ocr_confidence_threshold == 1.0  # clamped to max


# ---------------------------------------------------------------------------
# Throttling (Req 1.1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_throttle_sleeps_to_hold_target_fps():
    pipeline, clock, *_ = make_pipeline(target_fps=10)  # interval = 0.1 s
    # First throttle establishes the cycle start; no sleep.
    pipeline._throttle()
    assert clock.slept == []
    # Advance only 0.02 s of the 0.1 s interval, then throttle again.
    clock.t += 0.02
    pipeline._throttle()
    assert clock.slept == [pytest.approx(0.08)]


@pytest.mark.unit
def test_throttle_does_not_sleep_when_interval_already_elapsed():
    pipeline, clock, *_ = make_pipeline(target_fps=10)
    pipeline._throttle()
    clock.t += 0.5  # far beyond the 0.1 s interval
    pipeline._throttle()
    assert clock.slept == []


# ---------------------------------------------------------------------------
# Connect with retry (Req 1.4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ensure_connected_retries_at_interval_until_open_succeeds():
    source = FakeSource(open_errors=2)
    pipeline, clock, *_ = make_pipeline(source=source, reconnect_interval_s=5)
    pipeline._running = True
    pipeline._ensure_connected()
    assert pipeline._connected is True
    assert source.open_calls == 3  # two failures + one success
    assert clock.slept == [5.0, 5.0]  # slept the reconnect interval each failure


@pytest.mark.unit
def test_ensure_connected_stops_retrying_when_pipeline_stopped():
    source = FakeSource(open_errors=100)

    pipeline, clock, *_ = make_pipeline(source=source, reconnect_interval_s=3)

    # Stop the pipeline from within sleep so the retry loop terminates.
    def stopping_sleep(secs):
        clock.slept.append(secs)
        pipeline._running = False

    pipeline._sleep = stopping_sleep
    pipeline._running = True
    pipeline._ensure_connected()
    assert pipeline._connected is False
    assert source.open_calls == 1


# ---------------------------------------------------------------------------
# Reconnect on inactivity / interruption (Req 7.6, 7.7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inactivity_timeout_triggers_disconnect_and_reconnect_wait():
    source = FakeSource(frames=[])  # connected but yields no frames
    pipeline, clock, *_ = make_pipeline(
        source=source, inactivity_timeout_s=10, reconnect_interval_s=5
    )
    pipeline._running = True
    pipeline._connected = True
    pipeline._last_activity = 0.0
    clock.t = 11.0  # 11 s since last activity > 10 s timeout

    events = pipeline.run_once()

    assert events == []
    assert pipeline._connected is False  # treated as connection failure
    assert source.close_calls == 1  # source released before reconnect
    assert 5.0 in clock.slept  # waited the reconnect interval


@pytest.mark.unit
def test_no_frame_within_timeout_does_not_disconnect():
    source = FakeSource(frames=[])
    pipeline, clock, *_ = make_pipeline(
        source=source, inactivity_timeout_s=10, reconnect_interval_s=5
    )
    pipeline._running = True
    pipeline._connected = True
    pipeline._last_activity = 0.0
    clock.t = 3.0  # only 3 s elapsed, under the timeout

    events = pipeline.run_once()

    assert events == []
    assert pipeline._connected is True  # still connected


@pytest.mark.unit
def test_read_fault_is_treated_as_interruption():
    class RaisingSource(FakeSource):
        def read(self):
            raise RuntimeError("stream dropped")

    source = RaisingSource()
    pipeline, clock, *_ = make_pipeline(source=source, reconnect_interval_s=5)
    pipeline._running = True
    pipeline._connected = True
    pipeline._last_activity = 0.0

    events = pipeline.run_once()

    assert events == []
    assert pipeline._connected is False
    assert source.close_calls == 1


# ---------------------------------------------------------------------------
# One crop per retained box (Req 3.1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_one_crop_per_retained_box_passed_to_ocr():
    detections = [
        Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1),
        Detection(box=BoundingBox(30, 10, 60, 40), confidence=0.8, track_id=2),
        Detection(box=BoundingBox(100, 50, 150, 90), confidence=0.7, track_id=3),
    ]
    ocr = FakeOcr(OcrResult(text=None, confidence=0.0))  # surface all, keep simple
    pipeline, *_ = make_pipeline(detector=FakeDetector(detections), ocr=ocr)

    frame = a_frame()
    events = pipeline.process_frame(frame, acquired_at=0.0)

    # Exactly one crop per box, in order.
    assert len(ocr.crops) == 3
    assert len(events) == 3
    # Each crop matches the exact region sliced from the frame for its box.
    for det, crop in zip(detections, ocr.crops):
        expected = frame[det.box.y1:det.box.y2, det.box.x1:det.box.x2]
        assert np.array_equal(crop, expected)


@pytest.mark.unit
def test_empty_frame_yields_no_events():
    pipeline, *_ = make_pipeline(detector=FakeDetector([]))
    assert pipeline.process_frame(a_frame(), acquired_at=0.0) == []


# ---------------------------------------------------------------------------
# OCR failure surfacing (Req 3.3, 3.5, 7.5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ocr_no_text_surfaces_for_manual_entry_and_retains_crop():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1)
    ocr = FakeOcr(OcrResult(text=None, confidence=0.0))
    access = FakeAccessController()
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]), ocr=ocr, access=access
    )

    frame = a_frame()
    [event] = pipeline.process_frame(frame, acquired_at=0.0)

    assert event.needs_manual_review is True
    assert event.manual_review_reason == REVIEW_OCR_NO_TEXT
    assert list(pipeline.manual_review_queue) == [event]
    # Surfaced without ever reaching the access controller (Req 7.2/7.3).
    assert access.detection_events == []
    # Crop retained on the event for manual entry (Req 3.3).
    expected = frame[0:10, 0:20]
    assert np.array_equal(event.retained_crop, expected)


@pytest.mark.unit
def test_ocr_timeout_surfaces_for_manual_entry():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1)
    ocr = FakeOcr(OcrResult(text=None, confidence=0.0, timed_out=True))
    access = FakeAccessController()
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]), ocr=ocr, access=access
    )

    [event] = pipeline.process_frame(a_frame(), acquired_at=0.0)

    assert event.ocr_timed_out is True
    assert event.needs_manual_review is True
    assert event.manual_review_reason == REVIEW_OCR_TIMEOUT
    assert access.detection_events == []


@pytest.mark.unit
def test_default_ocr_invoker_times_out_a_slow_read():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1)

    class SlowOcr:
        def read_plate(self, crop):
            time.sleep(0.2)
            return OcrResult(text="B1234CD", confidence=0.99)

    # Use the real (thread-based) default invoker with a tiny timeout.
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]),
        ocr=SlowOcr(),
        ocr_timeout_ms=10,  # 0.01 s << 0.2 s read
        ocr_invoker=USE_REAL_OCR_INVOKER,  # force the real thread-based invoker
    )
    [event] = pipeline.process_frame(a_frame(), acquired_at=0.0)
    assert event.ocr_timed_out is True
    assert event.manual_review_reason == REVIEW_OCR_TIMEOUT


# ---------------------------------------------------------------------------
# Low-confidence gating (Req 3.4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confidence_strictly_below_threshold_is_low_confidence():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1)
    ocr = FakeOcr(OcrResult(text="B1234CD", confidence=0.69))  # below 0.70
    access = FakeAccessController()
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]),
        ocr=ocr,
        access=access,
        ocr_confidence_threshold=0.70,
    )

    [event] = pipeline.process_frame(a_frame(), acquired_at=0.0)

    assert event.needs_manual_review is True
    assert event.manual_review_reason == REVIEW_OCR_LOW_CONFIDENCE
    assert event.ocr_confidence == 0.69  # text + confidence retained
    # Low-confidence reads never reach the access controller (Req 7.4).
    assert access.detection_events == []


@pytest.mark.unit
def test_confidence_exactly_at_threshold_is_not_low_confidence():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1)
    ocr = FakeOcr(OcrResult(text="B1234CD", confidence=0.70))  # exactly at threshold
    access = FakeAccessController()
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]),
        ocr=ocr,
        access=access,
        ocr_confidence_threshold=0.70,
    )

    [event] = pipeline.process_frame(a_frame(), acquired_at=0.0)

    # At/above threshold is retained and routed, not surfaced as low-confidence.
    assert event.manual_review_reason != REVIEW_OCR_LOW_CONFIDENCE
    assert len(access.detection_events) == 1


# ---------------------------------------------------------------------------
# Format-invalid surfacing (Req 4.5, 7.5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_invalid_surfaces_raw_and_reason():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1)
    ocr = FakeOcr(OcrResult(text="123456", confidence=0.95))  # not Indonesian format
    access = FakeAccessController()
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]), ocr=ocr, access=access
    )

    [event] = pipeline.process_frame(a_frame(), acquired_at=0.0)

    assert event.is_format_valid is False
    assert event.ocr_text == "123456"  # raw retained
    assert event.needs_manual_review is True
    # Surfaces the normalizer's specific, non-empty rejection reason (Req 4.5).
    assert event.manual_review_reason
    assert event.manual_review_reason == event.normalization_reason
    # Format-invalid plates never reach the access controller (Req 7.4).
    assert access.detection_events == []


# ---------------------------------------------------------------------------
# Routing valid detections (flat match-and-log) + latency (Req 5.1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confident_valid_detection_is_routed_through_handle_detection():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.85, track_id=7)
    ocr = FakeOcr(OcrResult(text="B1234CD", confidence=0.95))
    access = FakeAccessController(
        AccessDecision(
            classification=Classification.RESIDENT,
            grant_method=GrantMethod.AUTOMATIC,
            gate_requested=True,
        )
    )
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]),
        ocr=ocr,
        access=access,
    )

    [event] = pipeline.process_frame(a_frame(), acquired_at=0.0)

    # Routed directly through the flat match-and-log, exactly once (Req 5.1).
    assert len(access.detection_events) == 1
    assert access.detection_events[0] is event
    assert event.normalized_plate == "B1234CD"
    assert event.detection_confidence == 0.85
    assert event.ocr_confidence == 0.95
    assert event.classification == Classification.RESIDENT
    # Not surfaced for manual review since it was routed and granted.
    assert event.needs_manual_review is False


@pytest.mark.unit
def test_guest_decision_is_mirrored_to_manual_review_queue():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.85, track_id=7)
    ocr = FakeOcr(OcrResult(text="B1234CD", confidence=0.95))
    access = FakeAccessController(
        AccessDecision(
            classification=Classification.GUEST,
            grant_method=GrantMethod.NONE,
            gate_requested=False,
            surfaced_to_guard=True,
            reason="No resident match; surfaced for manual decision.",
        )
    )
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]), ocr=ocr, access=access
    )

    [event] = pipeline.process_frame(a_frame(), acquired_at=0.0)

    # Still routed through handle_detection, then mirrored onto the review queue.
    assert len(access.detection_events) == 1
    assert event.classification == Classification.GUEST
    assert event.needs_manual_review is True
    assert list(pipeline.manual_review_queue) == [event]


@pytest.mark.unit
def test_processing_latency_is_decision_minus_acquisition():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.85, track_id=7)
    ocr = FakeOcr(OcrResult(text="B1234CD", confidence=0.95))
    clock = FakeClock(start=100.0)

    # Advance the clock by 0.25 s during OCR so latency = 250 ms.
    def slow_invoker(engine, crop, timeout_s):
        clock.t += 0.25
        return engine.read_plate(crop)

    pipeline, _clock, *_ = make_pipeline(
        detector=FakeDetector([detection]),
        ocr=ocr,
        clock=clock,
        ocr_invoker=slow_invoker,
    )

    [event] = pipeline.process_frame(a_frame(), acquired_at=100.0)

    assert event.processing_latency_ms == 250


@pytest.mark.unit
def test_on_manual_review_callback_is_invoked():
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1)
    ocr = FakeOcr(OcrResult(text=None, confidence=0.0))
    seen: list = []
    pipeline, *_ = make_pipeline(
        detector=FakeDetector([detection]),
        ocr=ocr,
        on_manual_review=seen.append,
    )

    [event] = pipeline.process_frame(a_frame(), acquired_at=0.0)
    assert seen == [event]
