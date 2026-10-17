"""Unit tests for the guest manual-grant tagging on :class:`ManualGrantService`.

Guests are no longer auto-logged at detection time: the guard's Open Gate action
writes the guest's single Event_Log row. These Qt-free tests assert that the
service writes exactly one ``grant_method=MANUAL`` record, tags it with the
supplied classification (``GUEST`` for a guard-permitted guest), and carries the
normalized guard/best-read plate.
"""

from __future__ import annotations

from typing import Optional

import pytest

from anpr.core.models import (
    Classification,
    EventRecord,
    GateResult,
    GrantMethod,
)
from anpr.core.normalizer import PlateNormalizer
from anpr.ui.guard_dashboard import ManualGrantService


class FakeGate:
    """Gate that reports a canned success/failure and counts open calls."""

    def __init__(self, *, success: bool = True, detail: str = "") -> None:
        self._success = success
        self._detail = detail
        self.open_calls: list[str] = []

    def open_gate(self, event_id: str) -> GateResult:
        self.open_calls.append(event_id)
        return GateResult(success=self._success, detail=self._detail)


class CountingEventLogRepository:
    """Event-log repo that counts and retains every appended record."""

    def __init__(self) -> None:
        self.records: list[EventRecord] = []

    def append(self, record: EventRecord) -> int:
        self.records.append(record)
        record.log_id = len(self.records)
        return record.log_id


def _service(gate: Optional[FakeGate] = None, log=None) -> tuple:
    gate = gate or FakeGate(success=True)
    log = log or CountingEventLogRepository()
    service = ManualGrantService(gate, log, PlateNormalizer())
    return service, gate, log


@pytest.mark.unit
def test_guest_manual_grant_writes_one_guest_tagged_record() -> None:
    """A guard-permitted guest writes exactly one MANUAL row tagged GUEST."""
    service, gate, log = _service()

    result = service.record_manual_grant(
        "b 1234 cd", classification=Classification.GUEST
    )

    assert result.gate_success is True
    assert result.recorded is True
    assert len(gate.open_calls) == 1
    assert len(log.records) == 1
    record = log.records[0]
    assert record.grant_method is GrantMethod.MANUAL
    assert record.classification is Classification.GUEST
    # The guard/best-read plate is normalized onto the record (Req 12.5).
    assert record.normalized_plate == "B1234CD"
    assert record.guard_plate == "B1234CD"


@pytest.mark.unit
def test_manual_grant_without_classification_leaves_it_unset() -> None:
    """A plain manual grant (no selected guest) leaves classification None."""
    service, _gate, log = _service()

    service.record_manual_grant("B1234CD")

    assert log.records[0].classification is None


@pytest.mark.unit
def test_guest_manual_grant_not_written_on_gate_failure() -> None:
    """A failed gate writes no guest row even when a classification is given."""
    service, gate, log = _service(gate=FakeGate(success=False, detail="jammed"))

    result = service.record_manual_grant(
        "B1234CD", classification=Classification.GUEST
    )

    assert result.gate_success is False
    assert result.recorded is False
    assert len(gate.open_calls) == 1
    assert log.records == []
