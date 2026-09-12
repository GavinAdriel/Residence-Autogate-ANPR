"""Property-based test for the detection pipeline's gating predicate.

Feature: mysql-monitoring-migration, Property 4
Property 4: Pipeline invokes ``handle_detection`` iff confident and
format-valid.

``DetectionPipeline._process_detection`` (Req 5.1, 7.2-7.4) routes a single
retained detection to ``AccessController.handle_detection`` *only* when the OCR
read is not timed out, has readable text, has confidence at or above the
configured threshold, and normalizes to a format-valid plate. In every other
case the event is surfaced for manual handling and ``handle_detection`` is never
called. Direction resolution is retired in the monitoring migration, so no
direction is ever resolved and ``event.direction`` stays ``None``.

For any generated OCR result (text / confidence / timed-out), configured
threshold, and normalization validity this test asserts:

* When confident-and-valid: ``handle_detection`` is called exactly once, the
  event is not surfaced for manual review, and no direction is resolved.
* Otherwise: ``handle_detection`` is not called, the event is surfaced (queued
  with the matching manual-review reason), and no direction is resolved.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from anpr.core.models import (
    AccessDecision,
    BoundingBox,
    Classification,
    Detection,
    GrantMethod,
    NormalizationResult,
    OcrResult,
)
from anpr.pipeline.pipeline import (
    DetectionPipeline,
    REVIEW_FORMAT_INVALID,
    REVIEW_OCR_LOW_CONFIDENCE,
    REVIEW_OCR_NO_TEXT,
    REVIEW_OCR_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Fakes for the injected ports.
# ---------------------------------------------------------------------------


class FakeDetector:
    """Detector returning a single fixed detection for any frame."""

    def __init__(self, detection: Detection) -> None:
        self._detection = detection

    def detect(self, frame):
        return [self._detection]


class FakeOcr:
    """OCR engine returning a scripted :class:`OcrResult`."""

    def __init__(self, result: OcrResult) -> None:
        self._result = result

    def read_plate(self, crop) -> OcrResult:
        return self._result


class FakeNormalizer:
    """Normalizer returning a controllable validity, recording each call.

    The pipeline only normalizes once the OCR gating passes, so ``calls`` also
    doubles as a witness that normalization was (or was not) reached.
    """

    _INVALID_REASON = "fake: not the Indonesian plate format."

    def __init__(self, is_valid: bool) -> None:
        self._is_valid = is_valid
        self.calls: list[Optional[str]] = []

    def normalize(self, raw: str) -> NormalizationResult:
        self.calls.append(raw)
        return NormalizationResult(
            normalized=(raw or "").upper(),
            is_valid=self._is_valid,
            raw=raw or "",
            reason=None if self._is_valid else self._INVALID_REASON,
        )


class CountingAccessController:
    """Access controller with a counting ``handle_detection`` seam.

    Exposes only ``handle_detection`` (the flat match-and-log entry point);
    ``handle_inbound``/``handle_outbound`` deliberately do not exist, so any
    attempt to resolve direction and route by it would raise rather than pass
    silently.
    """

    def __init__(self) -> None:
        self.events: list = []

    def handle_detection(self, ev) -> AccessDecision:
        self.events.append(ev)
        # A granted resident decision that is *not* surfaced to the guard, so a
        # routed event stays off the manual-review queue.
        return AccessDecision(
            classification=Classification.RESIDENT,
            grant_method=GrantMethod.AUTOMATIC,
            gate_requested=True,
            surfaced_to_guard=False,
        )


def _a_frame(width: int = 40, height: int = 20):
    """A small distinct HxWxC frame so cropping produces a real array."""
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def _make_pipeline(ocr_result: OcrResult, is_valid: bool, threshold: float):
    detection = Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=1)
    detector = FakeDetector(detection)
    ocr = FakeOcr(ocr_result)
    normalizer = FakeNormalizer(is_valid)
    access = CountingAccessController()
    pipeline = DetectionPipeline(
        object(),  # video source is unused; process_frame is driven directly
        detector,
        ocr,
        normalizer,
        access,
        ocr_confidence_threshold=threshold,
        wall_clock=lambda: __import__("datetime").datetime(2024, 1, 1),
        # Synchronous OCR invoker: no real threads or wall-clock delay.
        ocr_invoker=lambda engine, crop, timeout_s: engine.read_plate(crop),
    )
    return pipeline, normalizer, access


# ---------------------------------------------------------------------------
# Property.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    text=st.one_of(st.none(), st.text(max_size=12)),
    confidence=st.floats(min_value=0.0, max_value=1.0),
    timed_out=st.booleans(),
    is_valid=st.booleans(),
    threshold=st.floats(min_value=0.0, max_value=1.0),
)
def test_pipeline_routes_to_handle_detection_iff_confident_and_valid(
    text, confidence, timed_out, is_valid, threshold
) -> None:
    """Validates: Requirements 5.1, 7.2, 7.3, 7.4

    ``handle_detection`` is called exactly once iff the read is not timed out,
    has text, is confident (>= threshold), and normalizes valid; otherwise the
    event is surfaced and ``handle_detection`` is not called. Direction is never
    resolved in either branch.
    """
    ocr_result = OcrResult(text=text, confidence=confidence, timed_out=timed_out)
    pipeline, normalizer, access = _make_pipeline(ocr_result, is_valid, threshold)

    effective_threshold = pipeline.ocr_confidence_threshold
    confident_and_valid = (
        (not timed_out)
        and (text is not None)
        and (confidence >= effective_threshold)
        and is_valid
    )

    [event] = pipeline.process_frame(_a_frame(), acquired_at=0.0)

    # Direction resolution is retired: no direction is ever resolved (Req 7.x),
    # and the pipeline holds no direction resolver seam.
    assert event.direction is None
    assert not hasattr(pipeline, "_direction")

    if confident_and_valid:
        # Routed exactly once through the flat match-and-log (Req 5.1, 7.2-7.4).
        assert len(access.events) == 1
        assert access.events[0] is event
        # A granted, non-surfaced decision stays off the manual-review queue.
        assert event.needs_manual_review is False
        assert list(pipeline.manual_review_queue) == []
    else:
        # Not confident-and-valid: never routed, always surfaced for review.
        assert access.events == []
        assert event.needs_manual_review is True
        assert list(pipeline.manual_review_queue) == [event]

        # The surfacing reason matches the specific gate that rejected it, in
        # the same precedence the pipeline applies.
        if timed_out:
            expected_reason = REVIEW_OCR_TIMEOUT
        elif text is None:
            expected_reason = REVIEW_OCR_NO_TEXT
        elif confidence < effective_threshold:
            expected_reason = REVIEW_OCR_LOW_CONFIDENCE
        else:  # normalization rejected the plate as format-invalid
            expected_reason = normalizer._INVALID_REASON or REVIEW_FORMAT_INVALID
        assert event.manual_review_reason == expected_reason
