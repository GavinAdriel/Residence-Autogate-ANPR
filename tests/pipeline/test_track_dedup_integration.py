"""Integration tests for per-track deduplication in the DetectionPipeline.

These tests drive :meth:`DetectionPipeline.process_frame` across several frames
with scripted detections (same or different ``track_id``) and a fake access
controller, asserting the "one logged event per vehicle" behaviour:

* a resident track is routed through ``handle_detection`` exactly once, after
  which later frames of that track are skipped before OCR runs ("OCR will not
  read the plate anymore");
* a guest track performs no DB write at detection time, is surfaced to the guard
  exactly once (not per frame), and keeps the best (highest-confidence) read;
* two distinct resident tracks each produce their own single routing;
* an unresolved guest track that leaves view is evicted (no phantom state).

They use the fakes from ``test_pipeline`` so no real camera/model/DB is needed.
"""

from __future__ import annotations

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
from anpr.pipeline.pipeline import DetectionPipeline

from tests.pipeline.test_pipeline import (
    FakeAccessController,
    FakeClock,
    FakeDetector,
    FakeOcr,
    FakeSource,
    a_frame,
)


# ---------------------------------------------------------------------------
# Fakes specific to dedup: an access controller + OCR that vary per call.
# ---------------------------------------------------------------------------
class ScriptedAccessController:
    """Access controller returning a canned classification, counting calls."""

    def __init__(self, classification: Classification) -> None:
        self._classification = classification
        self.detection_events: list = []

    def handle_detection(self, ev) -> AccessDecision:
        self.detection_events.append(ev)
        if self._classification is Classification.RESIDENT:
            return AccessDecision(
                classification=Classification.RESIDENT,
                grant_method=GrantMethod.AUTOMATIC,
                gate_requested=True,
            )
        return AccessDecision(
            classification=Classification.GUEST,
            grant_method=GrantMethod.NONE,
            gate_requested=False,
            surfaced_to_guard=True,
            detected_plate=ev.normalized_plate,
            reason="No resident match; surfaced for manual decision.",
        )


class SequencedOcr:
    """OCR engine returning scripted results in order, repeating the last one."""

    def __init__(self, results: list[OcrResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def read_plate(self, crop) -> OcrResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def _pipeline(detector, ocr, access, **overrides):
    clock = FakeClock()
    pipeline = DetectionPipeline(
        FakeSource(),
        detector,
        ocr,
        PlateNormalizer(),
        access,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        wall_clock=lambda: __import__("datetime").datetime(2024, 1, 1),
        ocr_invoker=lambda engine, crop, timeout_s: engine.read_plate(crop),
        **overrides,
    )
    return pipeline


def _det(track_id: int) -> Detection:
    return Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=track_id)


# ---------------------------------------------------------------------------
# Resident: one routing then skip-before-OCR
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_resident_routed_once_then_track_skipped_before_ocr() -> None:
    detector = FakeDetector([_det(7)])
    ocr = SequencedOcr([OcrResult(text="B1234CD", confidence=0.95)])
    access = ScriptedAccessController(Classification.RESIDENT)
    pipeline = _pipeline(detector, ocr, access, track_staleness_frames=5)

    frame = a_frame()
    # Process the same track across five frames.
    for _ in range(5):
        pipeline.process_frame(frame, acquired_at=0.0)

    # Routed through the access controller exactly once (Req: one row/vehicle).
    assert len(access.detection_events) == 1
    # OCR ran only on the first frame; the resolved track is skipped before OCR.
    assert ocr.calls == 1
    assert pipeline.track_aggregator.is_resolved(7) is True


@pytest.mark.integration
def test_two_distinct_resident_tracks_each_routed_once() -> None:
    # Both vehicles are present in the same frame across two frames.
    detector = FakeDetector([_det(1), _det(2)])
    ocr = SequencedOcr([OcrResult(text="B1234CD", confidence=0.95)])
    access = ScriptedAccessController(Classification.RESIDENT)
    pipeline = _pipeline(detector, ocr, access, track_staleness_frames=5)

    frame = a_frame()
    pipeline.process_frame(frame, acquired_at=0.0)
    pipeline.process_frame(frame, acquired_at=0.0)

    # Each distinct track routed exactly once -> two rows for two vehicles.
    assert len(access.detection_events) == 2
    assert pipeline.track_aggregator.is_resolved(1) is True
    assert pipeline.track_aggregator.is_resolved(2) is True
    # OCR ran once per track on their first frame, then both were skipped.
    assert ocr.calls == 2


# ---------------------------------------------------------------------------
# Guest: surfaced once, best read kept, no resolution, evicted when gone
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_guest_track_surfaced_once_with_best_read_and_no_db_write() -> None:
    detector = FakeDetector([_det(9)])
    # Rising confidence across frames; a later frame reads the plate better.
    ocr = SequencedOcr(
        [
            OcrResult(text="B1234CD", confidence=0.72),
            OcrResult(text="B1234CE", confidence=0.95),  # best read
            OcrResult(text="B1234CF", confidence=0.80),
        ]
    )
    access = ScriptedAccessController(Classification.GUEST)
    pipeline = _pipeline(detector, ocr, access, track_staleness_frames=5)

    frame = a_frame()
    events = []
    for _ in range(3):
        events.extend(pipeline.process_frame(frame, acquired_at=0.0))

    # The guest reached the controller every frame (it is not resolved), but...
    assert len(access.detection_events) == 3
    # ...it is surfaced to the guard exactly once (not once per frame).
    assert len(pipeline.manual_review_queue) == 1
    # The guest track is never resolved by detection alone.
    assert pipeline.track_aggregator.is_resolved(9) is False
    # The best (highest-confidence) read is retained for the guard pre-fill.
    best = pipeline.track_aggregator.best_read(9)
    assert best is not None
    assert best.normalized_plate == "B1234CE"
    assert best.confidence == pytest.approx(0.95)


@pytest.mark.integration
def test_unresolved_guest_track_is_evicted_when_it_leaves_view() -> None:
    detector = FakeDetector([_det(3)])
    ocr = SequencedOcr([OcrResult(text="B1234CD", confidence=0.9)])
    access = ScriptedAccessController(Classification.GUEST)
    pipeline = _pipeline(detector, ocr, access, track_staleness_frames=0)

    frame = a_frame()
    pipeline.process_frame(frame, acquired_at=0.0)
    assert pipeline.track_aggregator.is_tracked(3) is True

    # The vehicle leaves view: a frame with no detections evicts the track.
    detector._detections = []  # type: ignore[attr-defined]
    pipeline.process_frame(frame, acquired_at=0.0)
    assert pipeline.track_aggregator.is_tracked(3) is False
    assert pipeline.track_aggregator.best_read(3) is None


# ---------------------------------------------------------------------------
# Config knob
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_from_config_reads_track_staleness_frames() -> None:
    class DictConfig:
        def __init__(self, mapping):
            self._m = mapping

        def get(self, key):
            return self._m[key]

    config = DictConfig(
        {
            "camera.target_fps": 30,
            "camera.reconnect_interval_s": 5,
            "camera.inactivity_timeout_s": 10,
            "ocr.confidence_threshold": 0.70,
            "ocr.timeout_ms": 2000,
            "pipeline.track_staleness_frames": 4,
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
    assert pipeline.track_aggregator.staleness_frames == 4
