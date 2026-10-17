"""GUI tests (pytest-qt) for the guard-granted guest flow on the dashboard.

These exercise the end of the guest lifecycle on the real :class:`GuardDashboard`
widget: a surfaced guest is selected (pre-filling the plate input with its
best-read plate), Open Gate writes the guest's single MANUAL row tagged GUEST,
the review entry is resolved exactly once, and the shared-aggregator resolve
callback fires with the guest's ``track_id``.

Skipped entirely when PyQt5 / pytest-qt are unavailable.
"""

from __future__ import annotations

from typing import Optional

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


def _guest_event(track_id: int, plate: str) -> DetectionEvent:
    return DetectionEvent(
        track_id=track_id,
        normalized_plate=plate,
        classification=Classification.GUEST,
        needs_manual_review=True,
        manual_review_reason="No resident match; surfaced for manual decision.",
    )


def _dashboard(qtbot, *, resolved_tracks: Optional[list] = None) -> GuardDashboard:
    gate = FakeGate(success=True)
    log = CountingEventLogRepository()
    resolved = resolved_tracks if resolved_tracks is not None else []
    dashboard = GuardDashboard(
        gate,
        log,
        PlateNormalizer(),
        on_resolve_track=resolved.append,
    )
    qtbot.addWidget(dashboard)
    # Attach fakes for assertions.
    dashboard._test_gate = gate  # type: ignore[attr-defined]
    dashboard._test_log = log  # type: ignore[attr-defined]
    return dashboard


@pytest.mark.gui
def test_selecting_surfaced_guest_prefills_plate_input(qtbot) -> None:
    dashboard = _dashboard(qtbot)
    dashboard.surface_event(_guest_event(7, "B1234CD"))

    # Select the only review entry.
    dashboard._review_list.setCurrentRow(0)

    assert dashboard._plate_input.text() == "B1234CD"


@pytest.mark.gui
def test_open_gate_writes_one_guest_row_and_resolves(qtbot) -> None:
    resolved_tracks: list[int] = []
    dashboard = _dashboard(qtbot, resolved_tracks=resolved_tracks)
    dashboard.surface_event(_guest_event(7, "B1234CD"))
    dashboard._review_list.setCurrentRow(0)

    assert len(dashboard.review_queue) == 1

    # Permit the guest.
    dashboard._on_open_gate_clicked()

    log = dashboard._test_log  # type: ignore[attr-defined]
    gate = dashboard._test_gate  # type: ignore[attr-defined]

    # Exactly one gate-open and one guest-tagged MANUAL row.
    assert len(gate.open_calls) == 1
    assert len(log.records) == 1
    record = log.records[0]
    assert record.grant_method is GrantMethod.MANUAL
    assert record.classification is Classification.GUEST
    assert record.normalized_plate == "B1234CD"

    # The entry is resolved (removed exactly once) and the track callback fired.
    assert len(dashboard.review_queue) == 0
    assert resolved_tracks == [7]


@pytest.mark.gui
def test_guest_not_written_or_resolved_on_gate_failure(qtbot) -> None:
    resolved_tracks: list[int] = []
    gate = FakeGate(success=False, detail="relay jammed")
    log = CountingEventLogRepository()
    dashboard = GuardDashboard(
        gate, log, PlateNormalizer(), on_resolve_track=resolved_tracks.append
    )
    qtbot.addWidget(dashboard)
    dashboard.surface_event(_guest_event(9, "B1234CD"))
    dashboard._review_list.setCurrentRow(0)

    dashboard._on_open_gate_clicked()

    # Gate failed: no row, entry retained, track not resolved (Req 12.6).
    assert log.records == []
    assert len(dashboard.review_queue) == 1
    assert resolved_tracks == []


@pytest.mark.gui
def test_resolving_same_grant_removes_entry_exactly_once(qtbot) -> None:
    resolved_tracks: list[int] = []
    dashboard = _dashboard(qtbot, resolved_tracks=resolved_tracks)
    token = dashboard.surface_event(_guest_event(5, "B1234CD"))
    dashboard._review_list.setCurrentRow(0)

    dashboard._on_open_gate_clicked()
    # Already resolved; resolving the same token again removes nothing.
    assert dashboard.resolve_event(token) is False
    assert len(dashboard.review_queue) == 0
    assert resolved_tracks == [5]
