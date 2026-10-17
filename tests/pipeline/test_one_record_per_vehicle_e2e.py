"""End-to-end integration tests for one-record-per-vehicle + guest grant.

These wire the *real* :class:`DetectionPipeline`, the *real*
:class:`AccessController` (over in-memory fake repositories/gate), a shared
:class:`TrackAggregator`, and -- for the guest leg -- the *real*
:class:`GuardDashboard`, to prove the full flow end to end:

* **Resident vehicle (multi-frame)** -> exactly one ``ANPR_Log`` row, the gate
  opened once, and OCR stops after the first confident match.
* **Guest vehicle (multi-frame)** -> no row during detection, surfaced to the
  guard exactly once with the best-read plate; the guard's Open Gate then writes
  the single ``grant_method=MANUAL`` row tagged GUEST and marks the track
  resolved on the shared aggregator so the pipeline skips it thereafter.
* **Un-permitted guest** -> after it leaves view it is evicted and no row is
  ever written (no phantom entry).

The GUI leg is skipped when PyQt5 / pytest-qt are unavailable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from anpr.core.access_controller import AccessController
from anpr.core.models import (
    BoundingBox,
    Classification,
    Detection,
    EventRecord,
    GateResult,
    GrantMethod,
    OcrResult,
    ResidentRecord,
)
from anpr.core.normalizer import PlateNormalizer
from anpr.pipeline.pipeline import DetectionPipeline
from anpr.pipeline.track_aggregator import TrackAggregator

from tests.pipeline.test_pipeline import FakeClock, FakeDetector, FakeSource, a_frame
from tests.pipeline.test_track_dedup_integration import SequencedOcr


# ---------------------------------------------------------------------------
# In-memory ports (shared by pipeline-side AccessController and dashboard-side)
# ---------------------------------------------------------------------------
class FakeResidentRepo:
    """Resident lookup keyed by an exact normalized-plate match."""

    def __init__(self, residents: dict[str, ResidentRecord]) -> None:
        self._residents = residents
        self.queried: list[str] = []

    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        self.queried.append(normalized_plate)
        return self._residents.get(normalized_plate)


class CountingEventLogRepo:
    """Append-only event-log repo recording every record and assigning Log_IDs."""

    def __init__(self) -> None:
        self.records: list[EventRecord] = []

    def append(self, record: EventRecord) -> int:
        self.records.append(record)
        record.log_id = len(self.records)
        return record.log_id


class FakeGate:
    """Gate reporting a canned success and counting open calls."""

    def __init__(self, *, success: bool = True) -> None:
        self._success = success
        self.open_calls: list[str] = []

    def open_gate(self, event_id: str) -> GateResult:
        self.open_calls.append(event_id)
        return GateResult(success=self._success, detail="")


def _resident(plate: str) -> ResidentRecord:
    return ResidentRecord(
        id="res-1",
        normalized_plate=plate,
        created_at="2024-01-01T00:00:00+07:00",
        updated_at="2024-01-01T00:00:00+07:00",
        vehicle_id=42,
    )


def _det(track_id: int) -> Detection:
    return Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=track_id)


def _build(
    *,
    residents: dict[str, ResidentRecord],
    ocr: SequencedOcr,
    detector: FakeDetector,
    aggregator: TrackAggregator,
) -> tuple[DetectionPipeline, CountingEventLogRepo, FakeGate, FakeResidentRepo]:
    resident_repo = FakeResidentRepo(residents)
    event_log = CountingEventLogRepo()
    gate = FakeGate(success=True)
    controller = AccessController(resident_repo, event_log, gate)
    clock = FakeClock()
    pipeline = DetectionPipeline(
        FakeSource(),
        detector,
        ocr,
        PlateNormalizer(),
        controller,
        track_aggregator=aggregator,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        wall_clock=lambda: __import__("datetime").datetime(2024, 1, 1),
        ocr_invoker=lambda engine, crop, timeout_s: engine.read_plate(crop),
    )
    return pipeline, event_log, gate, resident_repo


# ---------------------------------------------------------------------------
# Resident: one row, one gate-open, OCR stops after match
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_resident_vehicle_logs_exactly_one_row_and_stops_ocr() -> None:
    aggregator = TrackAggregator(staleness_frames=10)
    detector = FakeDetector([_det(1)])
    ocr = SequencedOcr([OcrResult(text="B1234CD", confidence=0.95)])
    pipeline, event_log, gate, _repo = _build(
        residents={"B1234CD": _resident("B1234CD")},
        ocr=ocr,
        detector=detector,
        aggregator=aggregator,
    )

    frame = a_frame()
    for _ in range(6):
        pipeline.process_frame(frame, acquired_at=0.0)

    # Exactly one row for the vehicle, gate opened exactly once.
    assert len(event_log.records) == 1
    assert event_log.records[0].classification is Classification.RESIDENT
    assert event_log.records[0].grant_method is GrantMethod.AUTOMATIC
    assert len(gate.open_calls) == 1
    # OCR stopped after the first confident match (resolved track skipped).
    assert ocr.calls == 1
    assert aggregator.is_resolved(1) is True


# ---------------------------------------------------------------------------
# Guest: no row at detection, surfaced once; guard grant writes one row
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_guest_vehicle_surfaced_once_then_guard_grant_writes_one_row() -> None:
    pytest.importorskip("PyQt5")
    pytest.importorskip("pytestqt")
    from PyQt5.QtWidgets import QApplication  # noqa: F401

    from anpr.ui.guard_dashboard import GuardDashboard

    aggregator = TrackAggregator(staleness_frames=10)
    detector = FakeDetector([_det(9)])
    ocr = SequencedOcr(
        [
            OcrResult(text="B9999ZZ", confidence=0.75),
            OcrResult(text="B9999ZY", confidence=0.95),  # best read
        ]
    )
    pipeline, event_log, gate, _repo = _build(
        residents={},  # no residents -> every match is a guest
        ocr=ocr,
        detector=detector,
        aggregator=aggregator,
    )

    frame = a_frame()
    surfaced: list = []
    pipeline._on_manual_review = surfaced.append  # type: ignore[attr-defined]
    for _ in range(3):
        pipeline.process_frame(frame, acquired_at=0.0)

    # No DB row written during detection; surfaced to the guard exactly once.
    assert event_log.records == []
    assert len(surfaced) == 1
    assert aggregator.is_resolved(9) is False
    best = aggregator.best_read(9)
    assert best is not None and best.normalized_plate == "B9999ZY"

    # The guard permits the guest via the real dashboard, sharing the gate,
    # event log, and aggregator (resolve callback) with the pipeline.
    app = QApplication.instance() or QApplication([])
    dashboard = GuardDashboard(
        gate,
        event_log,
        PlateNormalizer(),
        on_resolve_track=aggregator.mark_resolved,
    )
    try:
        surfaced_event = surfaced[0]
        dashboard.surface_event(surfaced_event, surfaced_event.manual_review_reason)
        dashboard._review_list.setCurrentRow(0)
        # Pre-filled with the best read.
        assert dashboard._plate_input.text() == "B9999ZY"

        dashboard._on_open_gate_clicked()
    finally:
        dashboard.close()

    # Exactly one guest row written by the grant, tagged GUEST + MANUAL.
    assert len(event_log.records) == 1
    record = event_log.records[0]
    assert record.classification is Classification.GUEST
    assert record.grant_method is GrantMethod.MANUAL
    assert record.normalized_plate == "B9999ZY"
    # The track is now resolved on the shared aggregator.
    assert aggregator.is_resolved(9) is True

    # A later frame of the same (now resolved) track is skipped: no new OCR,
    # no new row.
    detector._detections = [_det(9)]  # type: ignore[attr-defined]
    ocr_calls_before = ocr.calls
    pipeline.process_frame(frame, acquired_at=0.0)
    assert ocr.calls == ocr_calls_before
    assert len(event_log.records) == 1


# ---------------------------------------------------------------------------
# Un-permitted guest: evicted, never logged
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_unpermitted_guest_leaves_no_row_after_eviction() -> None:
    aggregator = TrackAggregator(staleness_frames=0)
    detector = FakeDetector([_det(3)])
    ocr = SequencedOcr([OcrResult(text="B9999ZZ", confidence=0.9)])
    pipeline, event_log, gate, _repo = _build(
        residents={},
        ocr=ocr,
        detector=detector,
        aggregator=aggregator,
    )

    frame = a_frame()
    pipeline.process_frame(frame, acquired_at=0.0)
    assert aggregator.is_tracked(3) is True
    assert event_log.records == []

    # The guest reverses out before the guard permits it: no detections ->
    # evicted, and nothing was ever logged.
    detector._detections = []  # type: ignore[attr-defined]
    pipeline.process_frame(frame, acquired_at=0.0)
    assert aggregator.is_tracked(3) is False
    assert event_log.records == []
    assert gate.open_calls == []
