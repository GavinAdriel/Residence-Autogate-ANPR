"""End-to-end exit-correlation integration (dual-camera-exit-correlation, Task 8.1).

Drives the real :class:`AccessController` across a full inbound-then-outbound
pass using an in-memory event log that reproduces the MySQL correlation
semantics (open ENTRY rows, most-recent-open lookup, idempotent close). This
exercises the correlation logic end-to-end -- entry opened, then closed by the
matching exit -- and the Exit_Anomaly path, without a live MySQL, torch, or
OpenCV.

Covers Requirements 3.1, 4.2 (via the guard-entry analogue), 5.1, 5.2, 5.3.
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

pytestmark = pytest.mark.integration


class InMemoryEventLog:
    """An event log reproducing the MySQL open-entry correlation semantics.

    ``append`` assigns a monotonic Log_ID; ``find_open_entry`` returns the most
    recent OPEN ENTRY for a plate; ``close_entry`` is guarded by OPEN state so a
    second close affects zero rows (idempotent), mirroring the real SQL.
    """

    def __init__(self) -> None:
        self.rows: list[EventRecord] = []
        self._next_id = 1

    def append(self, record: EventRecord) -> int:
        record.log_id = self._next_id
        self._next_id += 1
        self.rows.append(record)
        return record.log_id

    def find_open_entry(self, normalized_plate: str) -> Optional[EventRecord]:
        matches = [
            r
            for r in self.rows
            if r.normalized_plate == normalized_plate
            and r.event_kind is EventKind.ENTRY
            and r.entry_state is EntryState.OPEN
        ]
        return matches[-1] if matches else None

    def close_entry(self, entry_log_id: int, closed_by_log_id: int) -> int:
        for r in self.rows:
            if r.log_id == entry_log_id and r.entry_state is EntryState.OPEN:
                r.entry_state = EntryState.CLOSED
                r.closed_by_event_id = str(closed_by_log_id)
                return 1
        return 0


class FakeResidentRepo:
    def __init__(self, plates: set[str]) -> None:
        self._plates = plates

    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        if normalized_plate in self._plates:
            return ResidentRecord(
                id="r1",
                normalized_plate=normalized_plate,
                created_at="2026-01-01T00:00:00+07:00",
                updated_at="2026-01-01T00:00:00+07:00",
                vehicle_id=7,
            )
        return None


class FakeGate:
    def __init__(self) -> None:
        self.opens = 0

    def open_gate(self, event_id: str) -> GateResult:
        self.opens += 1
        return GateResult(success=True)


def _ev(plate: str, camera_id: int) -> DetectionEvent:
    return DetectionEvent(
        event_id=f"evt-{plate}-{camera_id}",
        ocr_text=plate,
        normalized_plate=plate,
        is_format_valid=True,
        camera_id=camera_id,
    )


def _controller(residents: set[str], log: InMemoryEventLog, gate: FakeGate):
    return AccessController(FakeResidentRepo(residents), log, gate)


# ---------------------------------------------------------------------------
# Guest round trip: entry opened, then closed by the matching exit (Req 5.1, 5.2)
# ---------------------------------------------------------------------------
def test_guest_entry_opened_then_closed_by_matching_exit() -> None:
    plate = "B1234CD"
    log, gate = InMemoryEventLog(), FakeGate()
    ctrl = _controller(set(), log, gate)  # no residents -> guest

    # 1. INBOUND guest surfaced; the guard permits entry -> one OPEN ENTRY row.
    inbound_decision = ctrl.handle_detection(_ev(plate, 1), DirectionOutcome.INBOUND)
    assert inbound_decision.classification is Classification.GUEST
    assert inbound_decision.surfaced_to_guard is True
    assert log.rows == []  # not auto-logged; the guard writes the entry

    # Simulate the guard's entry permit writing the open ENTRY row.
    entry = EventRecord(
        id="entry-1",
        timestamp="2026-01-01T08:00:00+07:00",
        normalized_plate=plate,
        classification=Classification.GUEST,
        direction=DirectionOutcome.INBOUND,
        grant_method=GrantMethod.MANUAL,
        event_kind=EventKind.ENTRY,
        entry_state=EntryState.OPEN,
        camera_id=1,
    )
    log.append(entry)
    assert log.find_open_entry(plate) is not None

    # 2. OUTBOUND guest later: correlated to the open entry -> gate + EXIT + close.
    outbound_decision = ctrl.handle_detection(_ev(plate, 2), DirectionOutcome.OUTBOUND)

    assert outbound_decision.grant_method is GrantMethod.AUTOMATIC
    assert outbound_decision.is_exit_anomaly is False
    assert gate.opens == 1

    # An EXIT row was written and the entry is now CLOSED by it.
    exit_rows = [r for r in log.rows if r.event_kind is EventKind.EXIT]
    assert len(exit_rows) == 1
    exit_row = exit_rows[0]
    assert exit_row.direction is DirectionOutcome.OUTBOUND
    assert entry.entry_state is EntryState.CLOSED
    assert entry.closed_by_event_id == str(exit_row.log_id)

    # 3. The entry is no longer open: a second exit would be an anomaly.
    assert log.find_open_entry(plate) is None


# ---------------------------------------------------------------------------
# Exit with no entry ticket is an Exit_Anomaly (Req 5.3)
# ---------------------------------------------------------------------------
def test_unmatched_exit_is_an_anomaly() -> None:
    log, gate = InMemoryEventLog(), FakeGate()
    ctrl = _controller(set(), log, gate)

    decision = ctrl.handle_detection(_ev("X9999ZZ", 2), DirectionOutcome.OUTBOUND)

    assert decision.is_exit_anomaly is True
    assert decision.surfaced_to_guard is True
    assert decision.gate_requested is False
    assert gate.opens == 0
    assert log.rows == []  # no automatic row for an anomaly


# ---------------------------------------------------------------------------
# Resident round trip: enters (ENTRY/OPEN), exits (EXIT) + closes own entry
# ---------------------------------------------------------------------------
def test_resident_entry_then_exit_closes_own_entry() -> None:
    plate = "D5RES"
    log, gate = InMemoryEventLog(), FakeGate()
    ctrl = _controller({plate}, log, gate)  # resident

    ctrl.handle_detection(_ev(plate, 1), DirectionOutcome.INBOUND)
    entry_rows = [r for r in log.rows if r.event_kind is EventKind.ENTRY]
    assert len(entry_rows) == 1
    assert entry_rows[0].entry_state is EntryState.OPEN

    ctrl.handle_detection(_ev(plate, 2), DirectionOutcome.OUTBOUND)
    assert entry_rows[0].entry_state is EntryState.CLOSED
    assert gate.opens == 2  # opened on entry and on exit
