"""GUI tests (pytest-qt) for the auto-fill guest grant flow on the dashboard.

The guard no longer selects a queue entry: when a guest is detected, the plate
input auto-fills with the current best read, and pressing Open Gate records that
plate (or the guard's correction of it) as the guest's single Event_Log row,
resolves the guest's track, and clears the field for the next vehicle.

Covered:
* a guest detection auto-fills the plate input;
* the field keeps tracking the best read on later frames (decision 1a);
* the guard's in-progress edit is not clobbered by a new frame (decision 3a);
* Open Gate writes one GUEST/MANUAL row with the field value, resolves the
  track, and clears the field (decision 2a);
* a resident detection does not auto-fill and clears the field;
* an un-edited field clears when the guest leaves (non-guest detection).

Skipped when PyQt5 / pytest-qt are unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt")

from anpr.core.models import (
    Classification,
    DetectionEvent,
    EventRecord,
    GateResult,
    GrantMethod,
)
from anpr.core.normalizer import PlateNormalizer
from anpr.ui.guard_dashboard import GuardDashboard


class FakeGate:
    def __init__(self, *, success: bool = True, detail: str = "") -> None:
        self._success = success
        self._detail = detail
        self.open_calls: list[str] = []

    def open_gate(self, event_id: str) -> GateResult:
        self.open_calls.append(event_id)
        return GateResult(success=self._success, detail=self._detail)


class CountingEventLogRepository:
    def __init__(self) -> None:
        self.records: list[EventRecord] = []

    def append(self, record: EventRecord) -> int:
        self.records.append(record)
        record.log_id = len(self.records)
        return record.log_id


def _guest(track_id: int, plate: str, confidence: float = 0.9) -> DetectionEvent:
    return DetectionEvent(
        track_id=track_id,
        normalized_plate=plate,
        ocr_confidence=confidence,
        classification=Classification.GUEST,
    )


def _resident(track_id: int, plate: str) -> DetectionEvent:
    return DetectionEvent(
        track_id=track_id,
        normalized_plate=plate,
        classification=Classification.RESIDENT,
    )


def _dashboard(qtbot, *, gate=None, resolved=None):
    gate = gate or FakeGate(success=True)
    log = CountingEventLogRepository()
    resolved = resolved if resolved is not None else []
    dashboard = GuardDashboard(
        gate, log, PlateNormalizer(), on_resolve_track=resolved.append
    )
    qtbot.addWidget(dashboard)
    dashboard._test_gate = gate  # type: ignore[attr-defined]
    dashboard._test_log = log  # type: ignore[attr-defined]
    dashboard._test_resolved = resolved  # type: ignore[attr-defined]
    return dashboard


@pytest.mark.gui
def test_guest_detection_autofills_plate_input(qtbot) -> None:
    dashboard = _dashboard(qtbot)
    dashboard.on_detection(_guest(7, "B1234CD"))
    assert dashboard._plate_input.text() == "B1234CD"


@pytest.mark.gui
def test_autofill_tracks_best_read_across_frames(qtbot) -> None:
    dashboard = _dashboard(qtbot)
    dashboard.on_detection(_guest(7, "B1234CD"))
    # A later frame of the same guest carries a refined (best) read.
    dashboard.on_detection(_guest(7, "B1234CE"))
    assert dashboard._plate_input.text() == "B1234CE"


@pytest.mark.gui
def test_guard_edit_is_not_clobbered_by_new_frame(qtbot) -> None:
    dashboard = _dashboard(qtbot)
    dashboard.on_detection(_guest(7, "B1234CD"))

    # Guard corrects the plate (simulate a user keystroke edit).
    dashboard._plate_input.setText("B9999ZZ")
    dashboard._on_plate_text_edited("B9999ZZ")

    # A new frame arrives; it must not overwrite the in-progress correction.
    dashboard.on_detection(_guest(7, "B1234CE"))
    assert dashboard._plate_input.text() == "B9999ZZ"


@pytest.mark.gui
def test_open_gate_records_field_value_resolves_track_and_clears(qtbot) -> None:
    resolved: list[int] = []
    dashboard = _dashboard(qtbot, resolved=resolved)
    dashboard.on_detection(_guest(7, "B1234CD"))

    dashboard._on_open_gate_clicked()

    log = dashboard._test_log  # type: ignore[attr-defined]
    gate = dashboard._test_gate  # type: ignore[attr-defined]
    assert len(gate.open_calls) == 1
    assert len(log.records) == 1
    record = log.records[0]
    assert record.classification is Classification.GUEST
    assert record.grant_method is GrantMethod.MANUAL
    assert record.normalized_plate == "B1234CD"
    # Track resolved and the field cleared for the next vehicle (decision 2a).
    assert resolved == [7]
    assert dashboard._plate_input.text() == ""
    assert dashboard._current_guest_event is None


@pytest.mark.gui
def test_open_gate_records_guard_corrected_plate(qtbot) -> None:
    dashboard = _dashboard(qtbot)
    dashboard.on_detection(_guest(7, "B1234CD"))

    # Guard corrects the plate before opening the gate.
    dashboard._plate_input.setText("B5678XY")
    dashboard._on_plate_text_edited("B5678XY")

    dashboard._on_open_gate_clicked()

    log = dashboard._test_log  # type: ignore[attr-defined]
    assert log.records[0].normalized_plate == "B5678XY"
    assert dashboard._test_resolved == [7]  # type: ignore[attr-defined]


@pytest.mark.gui
def test_resident_detection_does_not_autofill_and_clears_field(qtbot) -> None:
    dashboard = _dashboard(qtbot)
    dashboard.on_detection(_guest(7, "B1234CD"))
    assert dashboard._plate_input.text() == "B1234CD"

    # A resident vehicle is handled automatically; no guest awaits a grant.
    dashboard.on_detection(_resident(8, "B0001AA"))
    assert dashboard._plate_input.text() == ""
    assert dashboard._current_guest_event is None


@pytest.mark.gui
def test_field_clears_when_guest_leaves(qtbot) -> None:
    dashboard = _dashboard(qtbot)
    dashboard.on_detection(_guest(7, "B1234CD"))
    assert dashboard._plate_input.text() == "B1234CD"

    # No guest at the gate anymore (e.g. an unknown/none classification).
    dashboard.on_detection(DetectionEvent(track_id=None, classification=None))
    assert dashboard._plate_input.text() == ""


@pytest.mark.gui
def test_guard_edit_preserved_when_guest_leaves(qtbot) -> None:
    dashboard = _dashboard(qtbot)
    dashboard.on_detection(_guest(7, "B1234CD"))
    dashboard._plate_input.setText("B9999ZZ")
    dashboard._on_plate_text_edited("B9999ZZ")

    # Even if the guest classification drops, an in-progress edit is preserved.
    dashboard.on_detection(DetectionEvent(track_id=None, classification=None))
    assert dashboard._plate_input.text() == "B9999ZZ"
