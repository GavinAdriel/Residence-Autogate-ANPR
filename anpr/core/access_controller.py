"""Access-control decision logic for the ANPR Autogate System.

The :class:`AccessController` turns a confident, format-valid
:class:`DetectionEvent` into an :class:`AccessDecision`: it classifies a
recognized plate as Resident or Guest, requests the gate-open action for
residents, and surfaces events to the Guard_Dashboard when a manual decision is
required.

Deduplicated, action-based logging (supersedes the earlier "log every
detection" flow): the detection pipeline collapses the many per-frame
detections of one vehicle into a single logged event per physical pass. The
controller participates as follows:

* **Resident** -- a match writes exactly one flat ``event_log`` record and
  requests the gate-open. The pipeline then marks the track resolved and stops
  reading it, so a resident yields exactly one row.
* **Guest** -- a non-match is NOT auto-logged. The controller returns a decision
  carrying the detected plate and surfaced to the guard; the single guest row is
  written only when the guard presses the entry-permit control. A guest the
  guard never permits yields no row at all.
* **Resident-lookup fault** -- still logs exactly one safety record naming the
  normalized plate and surfaces to the guard (Req 4.6), since this is an error
  path, not a routine guest.

The controller is pure orchestration over injected collaborators and depends
only on the Protocol interfaces defined in ``anpr.core.interfaces``
(``ResidentRepository``, ``EventLogRepository``, ``GateController``) -- never on
a concrete implementation -- so simulated and field adapters are structurally
interchangeable and chosen only in the composition root.

Match-and-log (Requirement 4, 5):

* Query the ``residents`` table for an *exact* normalized-plate match on
  ``ev.normalized_plate`` (Req 4.1).
* A match classifies the plate as Resident and requests a single gate-open via
  ``GateController.open_gate`` (Req 4.2). Gate success grants ``AUTOMATIC``
  (Req 4.3); gate failure yields ``NONE`` with ``gate_requested=True``
  (Req 4.4). A resident writes exactly one flat record.
* No match classifies the plate as Guest: no automatic gate-open is requested,
  **no record is written here**, and the event is surfaced to the guard for a
  manual decision carrying the detected plate (Req 4.5). The guest row is
  written later by the guard's entry-permit action.
* A resident-lookup fault requests no automatic gate-open, logs one safety
  record with the normalized plate string, and surfaces the event to the guard
  (Req 4.6).
* Every record that *is* written is flat: ``direction=None``,
  ``event_kind=None``, ``entry_state=EntryState.NA``, and
  ``closed_by_event_id=None`` -- entry/exit correlation is retired (Req 5.2,
  5.3, 5.4). The earlier "exactly one record in every outcome" guarantee no
  longer holds for guests, which are logged only on the guard's action.

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
    ImageStore,
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
    ResidentRecord,
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
    camera_id:
        Identifier of the camera that produced the detection, written to
        ``ANPR_Log.Camera_ID``. That column is NOT NULL with a foreign key to
        ``Camera``, so a real deployment must supply an id matching a seeded
        ``Camera`` row or the insert is rejected. Optional here so pure-logic
        tests, which use a fake repository, need not supply one.
    image_store:
        Optional store used to link a captured image to the logged event once its
        ``Log_ID`` is known (``Images.Log_ID`` is a NOT NULL foreign key, so the
        link can only be made after the log write).
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
        camera_id: Optional[int] = None,
        image_store: Optional["ImageStore"] = None,
        clock: Callable[[], datetime] = _default_clock,
        id_factory: Callable[[], str] = _default_id_factory,
    ) -> None:
        self._resident_repo = resident_repo
        self._event_log_repo = event_log_repo
        self._gate_controller = gate_controller
        self._environment_label = environment_label
        self._camera_id = camera_id
        self._image_store = image_store
        self._clock = clock
        self._id_factory = id_factory

    # ------------------------------------------------------------------
    # Flat match-and-log
    # ------------------------------------------------------------------
    def handle_detection(self, ev: DetectionEvent) -> AccessDecision:
        """Match-and-log for one confident, format-valid detection (Req 4).

        Returns an :class:`AccessDecision` describing the classification, the
        grant method, whether a gate-open was requested, whether the event was
        surfaced to the guard, and the detected plate for a guard pre-fill. A
        resident appends exactly one flat :class:`EventRecord` and requests the
        gate; a guest appends nothing here (it is surfaced and logged later by
        the guard's entry-permit action, Req 4.5); a resident-lookup fault
        appends one safety record and surfaces (Req 4.6).
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
            # Exactly one flat record for the detection (Req 4.7). No resident is
            # known, so Vehicle_ID is written as NULL.
            self._append_flat_record(ev, normalized_plate, decision, None)
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
            # A guest is NOT auto-logged: no record is written at detection
            # time. The event is surfaced to the guard carrying the detected
            # plate, and the single guest row is written only when the guard
            # presses the entry-permit control (see ManualGrantService). This
            # retires the old "every detection is logged" behaviour so each
            # physical vehicle yields exactly one row -- and a guest the guard
            # never permits yields none.
            decision.classification = Classification.GUEST
            decision.grant_method = GrantMethod.NONE
            decision.gate_requested = False
            decision.surfaced_to_guard = True
            decision.detected_plate = normalized_plate
            decision.reason = "No resident match; surfaced for manual decision."
            return decision

        # Resident only: write exactly one flat record for the detection and
        # attach any captured image. ``resident`` is the matched whitelist row,
        # supplying Vehicle_ID.
        self._append_flat_record(ev, normalized_plate, decision, resident)
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _append_flat_record(
        self,
        ev: DetectionEvent,
        normalized_plate: str,
        decision: AccessDecision,
        resident: Optional[ResidentRecord],
    ) -> None:
        """Build and append exactly one flat record (Req 4.7, 5.2, 5.3, 5.4).

        The record carries ``direction=None``, ``event_kind=None``,
        ``entry_state=EntryState.NA`` ("N/A"), and ``closed_by_event_id=None``:
        entry/exit correlation is retired in the monitoring migration.

        ``camera_id`` is stamped from configuration (``ANPR_Log.Camera_ID`` is NOT
        NULL) and ``vehicle_id`` from the matched whitelist row, or left NULL for a
        guest or a failed lookup. After the write the generated ``Log_ID`` is used
        to link any captured image, which cannot happen earlier because
        ``Images.Log_ID`` is a NOT NULL foreign key.
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
            camera_id=self._camera_id,
            vehicle_id=(resident.vehicle_id if resident is not None else None),
        )
        self._event_log_repo.append(record)
        decision.event_record = record
        self._link_image(ev, record)

    def _link_image(self, ev: DetectionEvent, record: EventRecord) -> None:
        """Attach a captured image to the logged event via its ``Log_ID``.

        A missing store, missing capture, or missing ``Log_ID`` is simply a no-op.
        Failures never propagate: the event is already logged and the gate
        decision already made, so a bookkeeping fault must not surface as an
        access-control error (Req 11.7).
        """
        if self._image_store is None or ev.image_ref is None:
            return
        if record.log_id is None:
            return
        try:
            self._image_store.record_reference(ev.image_ref, record.log_id)
        except Exception as exc:  # noqa: BLE001 - image linkage is non-critical
            logger.error(
                "Failed to link image for log %s: %s", record.log_id, exc
            )

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
