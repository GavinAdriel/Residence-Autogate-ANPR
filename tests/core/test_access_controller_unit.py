"""Unit tests for the flat match-and-log :class:`AccessController`.

These example-based tests exercise ``AccessController.handle_detection`` across
the three access outcomes using in-memory fakes for the three ports
(``ResidentRepository``, ``EventLogRepository``, ``GateController``) so they run
with no live MySQL and no framework dependencies (core stays pure).

Coverage:

* Resident + gate-success -> ``RESIDENT``/``AUTOMATIC``, one gate-open, surfaced
  ``False`` (Req 4.2, 4.3).
* Resident + gate-failure -> ``RESIDENT``/``NONE`` with ``gate_requested=True``
  (Req 4.4).
* Guest (no match) -> ``GUEST``/``NONE``, no gate-open, surfaced ``True``
  (Req 4.5).
* Resident-lookup fault -> classification ``None``, no gate-open, surfaced
  ``True`` (Req 4.6).
* Exactly one flat ``EventRecord`` is appended in every outcome (counting
  event-log repo).
* The public surface is flat: ``handle_detection`` exists and
  ``handle_inbound``/``handle_outbound`` do not (Req 4.8).
"""

from __future__ import annotations

from typing import Optional

import pytest

from anpr.core.access_controller import AccessController
from anpr.core.models import (
    Classification,
    DetectionEvent,
    EntryState,
    EventRecord,
    GateResult,
    GrantMethod,
    ResidentRecord,
)


# ----------------------------------------------------------------------
# In-memory fakes for the three ports
# ----------------------------------------------------------------------
class FakeResidentRepository:
    """A resident repo that returns a canned match, ``None``, or raises."""

    def __init__(
        self,
        *,
        record: Optional[ResidentRecord] = None,
        raises: Optional[Exception] = None,
    ) -> None:
        self._record = record
        self._raises = raises
        self.queried_plates: list[str] = []

    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        self.queried_plates.append(normalized_plate)
        if self._raises is not None:
            raise self._raises
        return self._record


class FakeGate:
    """A gate that reports a canned success/failure and counts open calls."""

    def __init__(self, *, success: bool = True, detail: str = "") -> None:
        self._success = success
        self._detail = detail
        self.open_calls: list[str] = []

    def open_gate(self, event_id: str) -> GateResult:
        self.open_calls.append(event_id)
        return GateResult(success=self._success, detail=self._detail)


class CountingEventLogRepository:
    """An event-log repo that counts and retains every appended record."""

    def __init__(self) -> None:
        self.records: list[EventRecord] = []

    def append(self, record: EventRecord) -> None:
        self.records.append(record)

    @property
    def append_count(self) -> int:
        return len(self.records)


def _resident_record(plate: str = "B1234XYZ") -> ResidentRecord:
    return ResidentRecord(
        id="res-1",
        normalized_plate=plate,
        created_at="2024-01-01T00:00:00+07:00",
        updated_at="2024-01-01T00:00:00+07:00",
    )


def _event(plate: str = "B1234XYZ") -> DetectionEvent:
    return DetectionEvent(
        event_id="evt-1",
        ocr_text=plate,
        normalized_plate=plate,
        is_format_valid=True,
    )


# ----------------------------------------------------------------------
# Outcome 1: Resident + gate success (Req 4.2, 4.3)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_resident_gate_success_grants_automatic() -> None:
    """A resident match with a successful gate-open grants AUTOMATIC (Req 4.3)."""
    repo = FakeResidentRepository(record=_resident_record())
    gate = FakeGate(success=True)
    log = CountingEventLogRepository()
    controller = AccessController(repo, log, gate)

    decision = controller.handle_detection(_event())

    assert decision.classification is Classification.RESIDENT
    assert decision.grant_method is GrantMethod.AUTOMATIC
    assert decision.gate_requested is True
    assert decision.surfaced_to_guard is False
    # Exactly one gate-open was requested (Req 4.2).
    assert gate.open_calls == ["evt-1"]
    # Exactly one flat record appended.
    assert log.append_count == 1


# ----------------------------------------------------------------------
# Outcome 2: Resident + gate failure (Req 4.4)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_resident_gate_failure_yields_none_with_gate_requested() -> None:
    """A resident match whose gate-open fails yields NONE but records the attempt (Req 4.4)."""
    repo = FakeResidentRepository(record=_resident_record())
    gate = FakeGate(success=False, detail="relay timeout")
    log = CountingEventLogRepository()
    controller = AccessController(repo, log, gate)

    decision = controller.handle_detection(_event())

    assert decision.classification is Classification.RESIDENT
    assert decision.grant_method is GrantMethod.NONE
    # The gate-open was still requested even though it failed (Req 4.4).
    assert decision.gate_requested is True
    assert gate.open_calls == ["evt-1"]
    assert log.append_count == 1


# ----------------------------------------------------------------------
# Outcome 3: Guest / no match (Req 4.5)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_guest_no_match_surfaces_without_gate() -> None:
    """No resident match classifies Guest, requests no gate, and surfaces (Req 4.5)."""
    repo = FakeResidentRepository(record=None)
    gate = FakeGate(success=True)
    log = CountingEventLogRepository()
    controller = AccessController(repo, log, gate)

    decision = controller.handle_detection(_event("B9999ZZZ"))

    assert decision.classification is Classification.GUEST
    assert decision.grant_method is GrantMethod.NONE
    assert decision.gate_requested is False
    assert decision.surfaced_to_guard is True
    # No automatic gate-open for a guest (Req 4.5).
    assert gate.open_calls == []
    assert log.append_count == 1


# ----------------------------------------------------------------------
# Outcome 4: Resident-lookup fault (Req 4.6)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_lookup_fault_surfaces_without_gate_and_no_classification() -> None:
    """A resident-lookup fault surfaces, opens no gate, and leaves classification unset (Req 4.6)."""
    repo = FakeResidentRepository(raises=RuntimeError("db down"))
    gate = FakeGate(success=True)
    log = CountingEventLogRepository()
    controller = AccessController(repo, log, gate)

    decision = controller.handle_detection(_event())

    assert decision.classification is None
    assert decision.grant_method is GrantMethod.NONE
    assert decision.gate_requested is False
    assert decision.surfaced_to_guard is True
    # No automatic gate-open on a lookup fault (Req 4.6).
    assert gate.open_calls == []
    # The failure is still logged with the normalized plate (Req 4.6).
    assert log.append_count == 1
    assert log.records[0].normalized_plate == "B1234XYZ"


# ----------------------------------------------------------------------
# Every outcome appends exactly one flat record
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_every_outcome_appends_exactly_one_flat_record() -> None:
    """Resident, guest, and fault inputs each append exactly one flat record."""
    scenarios = [
        (FakeResidentRepository(record=_resident_record()), FakeGate(success=True)),
        (FakeResidentRepository(record=_resident_record()), FakeGate(success=False)),
        (FakeResidentRepository(record=None), FakeGate(success=True)),
        (FakeResidentRepository(raises=RuntimeError("db down")), FakeGate(success=True)),
    ]

    for repo, gate in scenarios:
        log = CountingEventLogRepository()
        controller = AccessController(repo, log, gate)

        controller.handle_detection(_event())

        assert log.append_count == 1
        record = log.records[0]
        # Flat fields: no entry/exit correlation (Req 5.3, 5.4).
        assert record.direction is None
        assert record.event_kind is None
        assert record.entry_state is EntryState.NA
        assert record.closed_by_event_id is None


# ----------------------------------------------------------------------
# Flat public surface (Req 4.8)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_controller_exposes_only_handle_detection() -> None:
    """``handle_detection`` exists; the retired entry/exit handlers do not (Req 4.8)."""
    controller = AccessController(
        FakeResidentRepository(record=None),
        CountingEventLogRepository(),
        FakeGate(),
    )

    assert hasattr(controller, "handle_detection")
    assert callable(getattr(controller, "handle_detection"))
    assert not hasattr(controller, "handle_inbound")
    assert not hasattr(controller, "handle_outbound")
