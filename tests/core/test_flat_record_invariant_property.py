"""Property-based test for the single flat Event_Log record invariant.

Feature: mysql-monitoring-migration, Property 3
Property 3: Exactly one flat EventRecord per detection.

``AccessController.handle_detection(ev)`` appends *exactly one*
:class:`~anpr.core.models.EventRecord` in every outcome -- Resident (gate
success or failure), Guest, and resident-lookup fault -- via
``_append_flat_record`` (Req 4.7). Every such record is *flat*: entry/exit
correlation is retired, so ``direction`` and ``event_kind`` are ``None``,
``entry_state`` is ``EntryState.NA`` (serialized "N/A"), and
``closed_by_event_id`` is ``None`` (Req 5.2, 5.3, 5.4). The carried fields
(normalized plate, classification, grant method) match the returned decision.

The test drives the controller with a *counting* fake ``EventLogRepository``
that records every appended record, a fake ``ResidentRepository`` that returns a
match / ``None`` / raises, and a fake ``GateController`` -- no live MySQL and no
concrete peer implementations, per the ports-and-adapters design.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

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


# ---------------------------------------------------------------------------
# Fakes (no concrete peer implementations, no I/O)
# ---------------------------------------------------------------------------
class CountingEventLogRepo:
    """Fake ``EventLogRepository`` that records every appended record."""

    def __init__(self) -> None:
        self.records: list[EventRecord] = []

    def append(self, record: EventRecord) -> None:
        self.records.append(record)


class FakeResidentRepo:
    """Fake ``ResidentRepository`` with match / None / raising behavior."""

    def __init__(self, mode: str, resident: Optional[ResidentRecord]) -> None:
        self._mode = mode
        self._resident = resident

    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        if self._mode == "fault":
            raise RuntimeError("simulated resident-lookup fault")
        if self._mode == "resident":
            return self._resident
        return None  # guest: no match


class FakeGate:
    """Fake ``GateController`` recording the number of open requests."""

    def __init__(self, success: bool) -> None:
        self._success = success
        self.calls = 0

    def open_gate(self, event_id: str) -> GateResult:
        self.calls += 1
        return GateResult(success=self._success, detail="" if self._success else "jammed")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
_PLATES = ["B1234XYZ", "D5678AB", "F9012CD", "AA1BB", "Z0000ZZ"]


def _resident_for(plate: str) -> ResidentRecord:
    return ResidentRecord(
        id="res-1",
        normalized_plate=plate,
        created_at="2024-01-01T08:00:00+07:00",
        updated_at="2024-01-01T08:00:00+07:00",
    )


@st.composite
def _detection_events(draw) -> DetectionEvent:
    """Generate a confident, format-valid detection event."""
    plate = draw(st.sampled_from(_PLATES))
    return DetectionEvent(
        event_id=draw(st.one_of(st.none(), st.just("evt-fixed"))),
        timestamp=draw(
            st.one_of(st.none(), st.just(datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)))
        ),
        normalized_plate=draw(st.one_of(st.just(plate), st.none())),
        ocr_text=draw(st.one_of(st.none(), st.text(max_size=12))),
        guard_plate=draw(st.one_of(st.none(), st.text(max_size=12))),
        detection_confidence=draw(st.one_of(st.none(), st.floats(0.0, 1.0))),
        ocr_confidence=draw(st.one_of(st.none(), st.floats(0.0, 1.0))),
        processing_latency_ms=draw(st.one_of(st.none(), st.integers(0, 5000))),
    )


@pytest.mark.property
@settings(max_examples=200)
@given(
    ev=_detection_events(),
    mode=st.sampled_from(["resident", "guest", "fault"]),
    gate_success=st.booleans(),
)
def test_exactly_one_flat_record_per_detection(ev, mode, gate_success) -> None:
    """Validates: Requirements 4.7, 5.2, 5.3, 5.4

    Across resident / guest / fault inputs, ``handle_detection`` appends exactly
    one flat record whose correlation fields are cleared and whose plate,
    classification, and grant method match the returned decision.
    """
    normalized_plate = ev.normalized_plate or ""
    resident = _resident_for(normalized_plate) if mode == "resident" else None

    event_log = CountingEventLogRepo()
    gate = FakeGate(success=gate_success)
    controller = AccessController(
        resident_repo=FakeResidentRepo(mode, resident),
        event_log_repo=event_log,
        gate_controller=gate,
    )

    decision = controller.handle_detection(ev)

    # Exactly one record appended, in every outcome (Req 4.7).
    assert len(event_log.records) == 1
    record = event_log.records[0]

    # The decision references the same record that was appended.
    assert decision.event_record is record

    # Flat record: entry/exit correlation retired (Req 5.2, 5.3, 5.4).
    assert record.direction is None
    assert record.event_kind is None
    assert record.entry_state is EntryState.NA
    assert record.entry_state == "N/A"
    assert record.closed_by_event_id is None

    # Carried fields match the decision (plate / classification / grant).
    assert record.normalized_plate == normalized_plate
    assert record.classification == decision.classification
    assert record.grant_method == decision.grant_method

    # Sanity: the classification carried matches the outcome under test.
    if mode == "resident":
        assert record.classification is Classification.RESIDENT
        expected_grant = GrantMethod.AUTOMATIC if gate_success else GrantMethod.NONE
        assert record.grant_method is expected_grant
    elif mode == "guest":
        assert record.classification is Classification.GUEST
        assert record.grant_method is GrantMethod.NONE
    else:  # fault
        assert record.classification is None
        assert record.grant_method is GrantMethod.NONE
