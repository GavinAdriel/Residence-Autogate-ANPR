"""Property-based test for the access-controller outcome mapping.

Feature: mysql-monitoring-migration, Property 2
Property 2: ``handle_detection`` outcome mapping.

``AccessController.handle_detection`` (Req 4.1-4.6) maps a confident,
format-valid :class:`DetectionEvent` to an :class:`AccessDecision` via a flat
match-and-log over three collaborators (resident repo, gate, event-log repo).
For any generated detection this test asserts, per outcome, the classification,
the number of gate-open calls, the grant method, and whether the event was
surfaced to the guard:

* Resident match + gate success (Req 4.2, 4.3): classification ``RESIDENT``,
  the gate is opened exactly once, grant ``AUTOMATIC``, ``gate_requested`` True,
  not surfaced.
* Resident match + gate failure (Req 4.2, 4.4): classification ``RESIDENT``,
  the gate is opened exactly once, grant ``NONE``, ``gate_requested`` True.
* Guest / no match (Req 4.5): classification ``GUEST``, the gate is never
  opened, grant ``NONE``, surfaced to the guard.
* Resident-lookup fault / repo raises (Req 4.6): classification unset
  (``None``), the gate is never opened, grant ``NONE``, surfaced to the guard.
"""

from __future__ import annotations

from typing import Optional

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from anpr.core.access_controller import AccessController
from anpr.core.models import (
    Classification,
    DetectionEvent,
    GateResult,
    GrantMethod,
    ResidentRecord,
)


# ---------------------------------------------------------------------------
# In-memory fakes for the three injected ports.
# ---------------------------------------------------------------------------


class FakeResidentRepository:
    """Resident lookup that returns a match, ``None``, or raises.

    Configured by ``mode``: ``"match"`` returns a canned ``ResidentRecord``,
    ``"none"`` returns ``None`` (guest), and ``"raise"`` raises to model a
    database-lookup fault (Req 4.6).
    """

    def __init__(self, mode: str, record: Optional[ResidentRecord] = None) -> None:
        self._mode = mode
        self._record = record
        self.calls: list[str] = []

    def find_by_plate(self, normalized_plate: str) -> Optional[ResidentRecord]:
        self.calls.append(normalized_plate)
        if self._mode == "raise":
            raise RuntimeError("simulated resident-lookup fault")
        if self._mode == "match":
            return self._record
        return None


class FakeEventLogRepository:
    """Append-only event-log repo that records every appended record."""

    def __init__(self) -> None:
        self.records: list = []

    def append(self, record) -> None:
        self.records.append(record)


class FakeGateController:
    """Gate that returns a configured success/failure and counts calls."""

    def __init__(self, success: bool) -> None:
        self._success = success
        self.calls: list[str] = []

    def open_gate(self, event_id: str) -> GateResult:
        self.calls.append(event_id)
        return GateResult(
            success=self._success,
            detail="opened" if self._success else "gate jammed",
        )


def _make_record() -> ResidentRecord:
    return ResidentRecord(
        id="res-1",
        normalized_plate="B1234XYZ",
        created_at="2024-01-01T08:00:00+07:00",
        updated_at="2024-01-01T08:00:00+07:00",
    )


# ---------------------------------------------------------------------------
# Generators for the detection event under test.
# ---------------------------------------------------------------------------


@st.composite
def _detections(draw) -> DetectionEvent:
    """Generate a confident, format-valid detection with varied fields."""
    return DetectionEvent(
        event_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=12))),
        ocr_text=draw(st.one_of(st.none(), st.text(max_size=12))),
        guard_plate=draw(st.one_of(st.none(), st.text(max_size=12))),
        normalized_plate=draw(
            st.one_of(st.none(), st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", max_size=10))
        ),
        detection_confidence=draw(st.one_of(st.none(), st.floats(0.0, 1.0))),
        ocr_confidence=draw(st.one_of(st.none(), st.floats(0.0, 1.0))),
        processing_latency_ms=draw(st.one_of(st.none(), st.integers(0, 5000))),
    )


@pytest.mark.property
@given(
    ev=_detections(),
    repo_mode=st.sampled_from(["match", "none", "raise"]),
    gate_success=st.booleans(),
)
def test_handle_detection_outcome_mapping(ev, repo_mode, gate_success) -> None:
    """Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6

    For any detection, the resident-repo mode and gate result fully determine
    the classification, gate-call count, grant method, and surfacing.
    """
    resident_repo = FakeResidentRepository(repo_mode, record=_make_record())
    event_log_repo = FakeEventLogRepository()
    gate = FakeGateController(success=gate_success)
    controller = AccessController(resident_repo, event_log_repo, gate)

    decision = controller.handle_detection(ev)

    # The resident repo is always consulted exactly once (Req 4.1).
    assert len(resident_repo.calls) == 1

    if repo_mode == "match":
        # --- Resident (Req 4.2, 4.3, 4.4) ------------------------------
        assert decision.classification is Classification.RESIDENT
        # The gate is opened exactly once for a resident match (Req 4.2).
        assert len(gate.calls) == 1
        assert decision.gate_requested is True
        # A resident gate-open is not itself surfaced for manual review.
        assert decision.surfaced_to_guard is False
        if gate_success:
            assert decision.grant_method is GrantMethod.AUTOMATIC  # Req 4.3
        else:
            assert decision.grant_method is GrantMethod.NONE  # Req 4.4
    elif repo_mode == "none":
        # --- Guest (Req 4.5) -------------------------------------------
        assert decision.classification is Classification.GUEST
        # No automatic gate-open is requested for a guest (Req 4.5).
        assert len(gate.calls) == 0
        assert decision.gate_requested is False
        assert decision.grant_method is GrantMethod.NONE
        assert decision.surfaced_to_guard is True
    else:  # repo_mode == "raise"
        # --- Resident-lookup fault (Req 4.6) ---------------------------
        # Classification is left unset on a lookup fault.
        assert decision.classification is None
        # No gate-open is requested when the lookup faults (Req 4.6).
        assert len(gate.calls) == 0
        assert decision.gate_requested is False
        assert decision.grant_method is GrantMethod.NONE
        assert decision.surfaced_to_guard is True
