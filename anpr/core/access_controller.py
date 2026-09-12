"""Access-control decision logic for the ANPR Autogate System.

The :class:`AccessController` turns a confident, format-valid
:class:`DetectionEvent` into an :class:`AccessDecision`: it classifies a
recognized plate as Resident or Guest, requests the gate-open action for
residents, records the access attempt as exactly one flat record in the
``event_log``, and surfaces events to the Guard_Dashboard when a manual
decision is required.

The controller is pure orchestration over injected collaborators and depends
only on the Protocol interfaces defined in ``anpr.core.interfaces``
(``ResidentRepository``, ``EventLogRepository``, ``GateController``) -- never on
a concrete implementation -- so simulated and field adapters are structurally
interchangeable and chosen only in the composition root.

Flat match-and-log (Requirement 4, 5):

* Query the ``residents`` table for an *exact* normalized-plate match on
  ``ev.normalized_plate`` (Req 4.1).
* A match classifies the plate as Resident and requests a single gate-open via
  ``GateController.open_gate`` (Req 4.2). Gate success grants ``AUTOMATIC``
  (Req 4.3); gate failure yields ``NONE`` with ``gate_requested=True``
  (Req 4.4).
* No match classifies the plate as Guest: no automatic gate-open is requested
  and the event is surfaced to the guard for a manual decision (Req 4.5).
* A resident-lookup fault requests no automatic gate-open, logs the failure with
  the normalized plate string, and surfaces the event to the guard (Req 4.6).
* In **every** outcome exactly one flat :class:`EventRecord` is appended with
  ``direction=None``, ``event_kind=None``, ``entry_state=EntryState.NA``, and
  ``closed_by_event_id=None`` -- entry/exit correlation is retired (Req 4.7,
  5.2, 5.3, 5.4).

See .kiro/specs/mysql-monitoring-migration/design.md (Access controller) and
requirements.md (Requirements 4 and 5) for the authoritative definitions.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from anpr.core.interfaces import (
    EventLogRepository,
    GateController,
    ResidentRepository,
)
from anpr.core.models import (
    NA_SENTINEL,
    AccessDecision,
    Classification,
    DetectionEvent,
    EntryState,
    EnvironmentLabel,
    EventRecord,
    GrantMethod,
)

logger = logging.getLogger(__name__)


def _default_clock() -> datetime:
    """Return the current time as a timezone-aware UTC ``datetime``."""
    return datetime.now(timezone.utc)


def _default_id_factory() -> str:
    """Return a fresh unique event identifier."""
    return str(uuid.uuid4())


def _metric(value: Optional[float | int]) -> "float | int | str":
    """Return a numeric metric, or ``NA_SENTINEL`` when it is unavailable."""
    return value if value is not None else NA_SENTINEL


class AccessController:
    """Classifies plates and orchestrates gate/logging for access attempts.

    Parameters
    ----------
    resident_repo:
        Exact-match lookup into the ``residents`` table (Req 4.1).
    event_log_repo:
        Append-only access to the ``event_log`` (Req 4.7, 5.2).
    gate_controller:
        The gate-open abstraction shared with Manual_Override.
    environment_label:
        Deployment environment stamped onto every written record; optional so
        pure-logic tests need not supply one.
    clock:
        Timezone-aware time source; injectable for deterministic tests.
    id_factory:
        Unique event-id generator; injectable for deterministic tests.
    """

    def __init__(
        self,
        resident_repo: ResidentRepository,
        event_log_repo: EventLogRepository,
        gate_controller: GateController,
        *,
        environment_label: Optional[EnvironmentLabel] = None,
        clock: Callable[[], datetime] = _default_clock,
        id_factory: Callable[[], str] = _default_id_factory,
    ) -> None:
        self._resident_repo = resident_repo
        self._event_log_repo = event_log_repo
        self._gate_controller = gate_controller
        self._environment_label = environment_label
        self._clock = clock
        self._id_factory = id_factory

    # ------------------------------------------------------------------
    # Flat match-and-log
    # ------------------------------------------------------------------
    def handle_detection(self, ev: DetectionEvent) -> AccessDecision:
        """Flat match-and-log for one confident, format-valid detection (Req 4).

        Returns an :class:`AccessDecision` describing the classification, the
        grant method, whether a gate-open was requested, and whether the event
        was surfaced to the guard for a manual decision. In every outcome
        exactly one flat :class:`EventRecord` is appended (Req 4.7).
        """
        normalized_plate = ev.normalized_plate or ""
        decision = AccessDecision()

        # --- Resident lookup (Req 4.1) -------------------------------------
        try:
            resident = self._resident_repo.find_by_plate(normalized_plate)
        except Exception as exc:  # noqa: BLE001 - any query fault is a DB fault
            # Resident-lookup fault (Req 4.6): no auto-open, log with the
            # normalized plate, surface to the guard for a manual decision.
            logger.error(
                "Resident lookup failed for plate %r: %s",
                normalized_plate,
                exc,
            )
            decision.classification = None  # unset (Req 4.7)
            decision.grant_method = GrantMethod.NONE
            decision.gate_requested = False  # no gate (Req 4.6)
            decision.surfaced_to_guard = True  # surface (Req 4.6)
            decision.reason = (
                "Resident lookup failed; surfaced for manual decision."
            )
            # Exactly one flat record for the detection (Req 4.7).
            self._append_flat_record(ev, normalized_plate, decision)
            return decision

        if resident is not None:
            # --- Resident (Req 4.2, 4.3, 4.4) ------------------------------
            decision.classification = Classification.RESIDENT
            gate_result = self._gate_controller.open_gate(self._event_id(ev))
            decision.gate_requested = True
            if gate_result.success:
                decision.grant_method = GrantMethod.AUTOMATIC  # (Req 4.3)
                decision.reason = (
                    "Resident matched; automatic gate-open requested."
                )
            else:
                # The gate-open was requested but the gate reported failure;
                # the attempt still occurred (Req 4.4).
                decision.grant_method = GrantMethod.NONE
                decision.reason = (
                    f"Resident matched; gate-open failed: {gate_result.detail}"
                )
        else:
            # --- Guest (Req 4.5) -------------------------------------------
            decision.classification = Classification.GUEST
            decision.grant_method = GrantMethod.NONE
            decision.gate_requested = False
            decision.surfaced_to_guard = True
            decision.reason = "No resident match; surfaced for manual decision."

        # Exactly one flat record for the detection (Req 4.7).
        self._append_flat_record(ev, normalized_plate, decision)
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _append_flat_record(
        self,
        ev: DetectionEvent,
        normalized_plate: str,
        decision: AccessDecision,
    ) -> None:
        """Build and append exactly one flat record (Req 4.7, 5.2, 5.3, 5.4).

        The record carries ``direction=None``, ``event_kind=None``,
        ``entry_state=EntryState.NA`` ("N/A"), and ``closed_by_event_id=None``:
        entry/exit correlation is retired in the monitoring migration.
        """
        record = EventRecord(
            id=self._event_id(ev),
            timestamp=self._timestamp(ev),
            ocr_plate=ev.ocr_text or "",
            guard_plate=ev.guard_plate or "",
            normalized_plate=normalized_plate,
            classification=decision.classification,
            direction=None,  # Req 5.3
            grant_method=decision.grant_method,
            event_kind=None,  # Req 5.3
            entry_state=EntryState.NA,  # Req 5.4 ("N/A")
            closed_by_event_id=None,
            image_ref=(ev.image_ref.snapshot_path if ev.image_ref else None),
            detection_confidence=_metric(ev.detection_confidence),
            ocr_confidence=_metric(ev.ocr_confidence),
            processing_latency_ms=_metric(ev.processing_latency_ms),
            environment_label=self._environment_label,
        )
        self._event_log_repo.append(record)
        decision.event_record = record

    def _event_id(self, ev: DetectionEvent) -> str:
        """Return the event's id, generating a fresh one when absent."""
        return ev.event_id or self._id_factory()

    def _timestamp(self, ev: DetectionEvent) -> str:
        """Return an ISO-8601 timestamp for the record (Req 5.4).

        Prefers the event's own timestamp when present, otherwise stamps the
        injected clock's current time.
        """
        ts = ev.timestamp or self._clock()
        return ts.isoformat()
