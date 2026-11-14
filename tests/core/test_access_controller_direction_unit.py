"""Direction-aware AccessController tests (dual-camera-exit-correlation).

Covers the INBOUND/OUTBOUND x resident/guest matrix and exit correlation
(Requirements 3, 4.1, 5), plus backward compatibility: ``handle_detection(ev)``
with no direction is unchanged flat behavior (Req 7.1).

Pure-logic tests with in-memory fakes for the three ports; the event-log fake
additionally implements ``find_open_entry`` / ``close_entry`` for the OUTBOUND
exit path.
"""

from __future__ import annotations

from typing import Optional

import pytest

from anpr.core.access_controller import AccessController
from anpr.core.models import (
    Classification,
    DetectionEvent,
    DirectionOutcome,
    EntryState,
    EventKind,
    EventRecord,
    GateResult,
    GrantMethod,
    ResidentRecord,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeResidentRepository:
    def __init__(self, *, record: Optional[ResidentRecord] = None) -> None:
        self._record = record

    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        return self._record


class FakeGate:
    def __init__(self, *, success: bool = True) -> None:
        self._success = success
        self.open_calls: list[str] = []

    def open_gate(self, event_id: str) -> GateResult:
        self.open_calls.append(event_id)
        return GateResult(success=self._success)


class FakeEventLog:
    """Append + find_open_entry + close_entry, with a controllable open entry."""

    def __init__(self, *, open_entry: Optional[EventRecord] = None) -> None:
        self.records: list[EventRecord] = []
        self._open_entry = open_entry
        self.find_calls: list[str] = []
        self.close_calls: list[tuple[int, int]] = []
        self._next_log_id = 1000

    def append(self, record: EventRecord) -> int:
        record.log_id = self._next_log_id
        self._next_log_id += 1
        self.records.append(record)
        return record.log_id

    def find_open_entry(self, normalized_plate: str) -> Optional[EventRecord]:
        self.find_calls.append(normalized_plate)
        return self._open_entry

    def close_entry(self, entry_log_id: int, closed_by_log_id: int) -> int:
        self.close_calls.append((entry_log_id, closed_by_log_id))
        return 1


def _resident(plate: str = "B1234XYZ") -> ResidentRecord:
    return ResidentRecord(
        id="res-1",
        normalized_plate=plate,
        created_at="2024-01-01T00:00:00+07:00",
        updated_at="2024-01-01T00:00:00+07:00",
        vehicle_id=42,
    )


def _event(plate: str = "B1234XYZ", camera_id: Optional[int] = None) -> DetectionEvent:
    return DetectionEvent(
        event_id="evt-1",
        ocr_text=plate,
        normalized_plate=plate,
        is_format_valid=True,
        camera_id=camera_id,
    )


def _open_entry(plate: str = "B1234XYZ", log_id: int = 10) -> EventRecord:
    return EventRecord(
        id=str(log_id),
        timestamp="2026-01-01T08:00:00+07:00",
        normalized_plate=plate,
        classification=Classification.GUEST,
        direction=DirectionOutcome.INBOUND,
        event_kind=EventKind.ENTRY,
        entry_state=EntryState.OPEN,
        log_id=log_id,
    )


def _controller(repo, gate, log) -> AccessController:
    return AccessController(repo, log, gate, camera_id=1)


# ----------------------------------------------------------------------
# INBOUND resident / guest (Req 3.1, 4.1, 4.2)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_inbound_resident_writes_entry_open_and_gates() -> None:
    """INBOUND resident: gate + one ENTRY/OPEN/INBOUND row (Req 3.1)."""
    gate, log = FakeGate(), FakeEventLog()
    ctrl = _controller(FakeResidentRepository(record=_resident()), gate, log)

    decision = ctrl.handle_detection(_event(), DirectionOutcome.INBOUND)

    assert decision.classification is Classification.RESIDENT
    assert decision.grant_method is GrantMethod.AUTOMATIC
    assert gate.open_calls  # gate opened
    assert len(log.records) == 1
    row = log.records[0]
    assert row.direction is DirectionOutcome.INBOUND
    assert row.event_kind is EventKind.ENTRY
    assert row.entry_state is EntryState.OPEN


@pytest.mark.unit
def test_inbound_guest_surfaces_without_row() -> None:
    """INBOUND guest: surfaced, no row written here (Req 4.1)."""
    gate, log = FakeGate(), FakeEventLog()
    ctrl = _controller(FakeResidentRepository(record=None), gate, log)

    decision = ctrl.handle_detection(_event(), DirectionOutcome.INBOUND)

    assert decision.classification is Classification.GUEST
    assert decision.surfaced_to_guard is True
    assert decision.gate_requested is False
    assert log.records == []


# ----------------------------------------------------------------------
# OUTBOUND resident (Req 3.2, 3.3)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_outbound_resident_writes_exit_and_closes_open_entry() -> None:
    """OUTBOUND resident: EXIT row + best-effort close of a prior entry (Req 3.2, 3.3)."""
    gate = FakeGate()
    log = FakeEventLog(open_entry=_open_entry())
    ctrl = _controller(FakeResidentRepository(record=_resident()), gate, log)

    decision = ctrl.handle_detection(_event(), DirectionOutcome.OUTBOUND)

    assert decision.classification is Classification.RESIDENT
    assert gate.open_calls
    row = log.records[0]
    assert row.direction is DirectionOutcome.OUTBOUND
    assert row.event_kind is EventKind.EXIT
    assert row.entry_state is EntryState.NA
    # the prior open entry was closed with the exit row's log id
    assert log.close_calls == [(10, row.log_id)]


@pytest.mark.unit
def test_outbound_resident_without_open_entry_still_exits() -> None:
    """OUTBOUND resident with no open entry still gates + writes EXIT (no close)."""
    gate = FakeGate()
    log = FakeEventLog(open_entry=None)
    ctrl = _controller(FakeResidentRepository(record=_resident()), gate, log)

    decision = ctrl.handle_detection(_event(), DirectionOutcome.OUTBOUND)

    assert decision.grant_method is GrantMethod.AUTOMATIC
    assert len(log.records) == 1
    assert log.close_calls == []


# ----------------------------------------------------------------------
# OUTBOUND guest: exit correlation (Req 5)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_outbound_guest_with_open_entry_exits_and_closes() -> None:
    """OUTBOUND guest matched to an open entry: gate + EXIT row + close (Req 5.2)."""
    gate = FakeGate()
    log = FakeEventLog(open_entry=_open_entry())
    ctrl = _controller(FakeResidentRepository(record=None), gate, log)

    decision = ctrl.handle_detection(_event(), DirectionOutcome.OUTBOUND)

    assert decision.classification is Classification.GUEST
    assert decision.grant_method is GrantMethod.AUTOMATIC
    assert gate.open_calls
    assert log.find_calls == ["B1234XYZ"]
    row = log.records[0]
    assert row.event_kind is EventKind.EXIT
    assert log.close_calls == [(10, row.log_id)]
    assert decision.is_exit_anomaly is False


@pytest.mark.unit
def test_outbound_guest_without_open_entry_is_exit_anomaly() -> None:
    """OUTBOUND guest with no entry ticket: Exit_Anomaly, surfaced, no row (Req 5.3)."""
    gate = FakeGate()
    log = FakeEventLog(open_entry=None)
    ctrl = _controller(FakeResidentRepository(record=None), gate, log)

    decision = ctrl.handle_detection(_event(), DirectionOutcome.OUTBOUND)

    assert decision.classification is Classification.GUEST
    assert decision.is_exit_anomaly is True
    assert decision.surfaced_to_guard is True
    assert decision.gate_requested is False
    assert gate.open_calls == []
    assert log.records == []  # no automatic row
    assert decision.detected_plate == "B1234XYZ"


@pytest.mark.unit
def test_outbound_guest_close_failure_does_not_fail_the_exit() -> None:
    """A close that raises never fails the already-granted exit (Req 5.6)."""
    gate = FakeGate()
    log = FakeEventLog(open_entry=_open_entry())

    def _raise(entry_log_id: int, closed_by_log_id: int) -> int:
        raise RuntimeError("db hiccup")

    log.close_entry = _raise  # type: ignore[assignment]
    ctrl = _controller(FakeResidentRepository(record=None), gate, log)

    decision = ctrl.handle_detection(_event(), DirectionOutcome.OUTBOUND)

    # Exit still authorized and logged despite the close failure.
    assert decision.grant_method is GrantMethod.AUTOMATIC
    assert len(log.records) == 1


# ----------------------------------------------------------------------
# Camera id + backward compatibility
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_event_camera_id_overrides_controller_camera_id() -> None:
    """A per-detection camera_id is written in preference to the controller's."""
    gate, log = FakeGate(), FakeEventLog()
    ctrl = _controller(FakeResidentRepository(record=_resident()), gate, log)

    ctrl.handle_detection(_event(camera_id=2), DirectionOutcome.OUTBOUND)

    assert log.records[0].camera_id == 2


@pytest.mark.unit
def test_no_direction_is_unchanged_flat_behavior() -> None:
    """handle_detection(ev) with no direction keeps flat single-camera behavior (Req 7.1)."""
    gate, log = FakeGate(), FakeEventLog()
    ctrl = _controller(FakeResidentRepository(record=_resident()), gate, log)

    ctrl.handle_detection(_event())  # no direction arg

    row = log.records[0]
    assert row.direction is None
    assert row.event_kind is None
    assert row.entry_state is EntryState.NA
