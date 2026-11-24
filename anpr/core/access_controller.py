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
* The direction passed in comes from the camera that saw the vehicle: INBOUND
  writes an ``ENTRY/OPEN`` row, OUTBOUND an ``EXIT`` row that closes the open
  entry, and ``None`` / ``UNDETERMINED`` a flat row (``direction=None``,
  ``event_kind=None``, ``entry_state=EntryState.NA``). Guests are logged only on
  the guard's action, so "exactly one record in every outcome" does not hold
  for them.

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
    DirectionOutcome,
    EntryState,
    EnvironmentLabel,
    EventKind,
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


def _logged_direction(
    direction: Optional[DirectionOutcome],
) -> Optional[DirectionOutcome]:
    """Return the direction to persist in ``ANPR_Log.Direction``.

    INBOUND/OUTBOUND are persisted as-is. ``None`` (flat/single-camera) and
    ``UNDETERMINED`` persist as NULL, matching the flat-record behavior and
    leaving an undetermined-direction event without a misleading direction.
    """
    if direction in (DirectionOutcome.INBOUND, DirectionOutcome.OUTBOUND):
        return direction
    return None


def _correlation_fields(
    direction: Optional[DirectionOutcome],
) -> "tuple[Optional[EventKind], EntryState]":
    """Map a direction to the ``(Event_Kind, Entry_State)`` to persist.

    * INBOUND -> ``(ENTRY, OPEN)`` -- this row becomes an Open_Entry_Record.
    * OUTBOUND -> ``(EXIT, N/A)`` -- an exit has no open lifecycle of its own.
    * None / UNDETERMINED -> ``(None, N/A)`` -- the flat single-camera record.
    """
    if direction is DirectionOutcome.INBOUND:
        return EventKind.ENTRY, EntryState.OPEN
    if direction is DirectionOutcome.OUTBOUND:
        return EventKind.EXIT, EntryState.NA
    return None, EntryState.NA


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
    def handle_detection(
        self,
        ev: DetectionEvent,
        direction: Optional[DirectionOutcome] = None,
    ) -> AccessDecision:
        """Match-and-log for one confident, format-valid detection (Req 4).

        ``direction`` drives direction-specific behavior
        (dual-camera-exit-correlation):

        * ``None`` / ``UNDETERMINED`` -> the existing flat match-and-log: a
          resident is logged + gated, a guest is surfaced, a lookup fault logs a
          safety record and surfaces. No direction/entry-state is set. This
          preserves the single-camera behavior and every existing caller/test.
        * ``INBOUND`` -> a resident entry writes one ``ENTRY/OPEN/INBOUND`` row
          and gates; a guest is surfaced (its entry ticket is written by the
          guard's permit action).
        * ``OUTBOUND`` -> a resident exit gates, writes one ``EXIT/N-A/OUTBOUND``
          row, and best-effort closes any open entry; a guest exit is correlated
          to its Open_Entry_Record (Req 5): found -> gate + EXIT row + close; none
          -> Exit_Anomaly, surfaced to the guard with no automatic grant.

        Returns an :class:`AccessDecision` describing the classification, grant
        method, gate request, guard surfacing, and the detected plate.
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
            # Exactly one record for the detection (Req 4.7). No resident is
            # known, so Vehicle_ID is written as NULL.
            self._append_record(ev, normalized_plate, decision, None, direction)
            return decision

        if resident is not None:
            return self._handle_resident(ev, normalized_plate, resident, direction)
        return self._handle_guest(ev, normalized_plate, direction)

    # ------------------------------------------------------------------
    # Resident / guest handlers
    # ------------------------------------------------------------------
    def _handle_resident(
        self,
        ev: DetectionEvent,
        normalized_plate: str,
        resident: ResidentRecord,
        direction: Optional[DirectionOutcome],
    ) -> AccessDecision:
        """Resident match in either direction: gate + one directional row.

        A resident is authorized by the whitelist match alone, independent of
        any entry ticket (Req 3.4). An OUTBOUND resident additionally closes any
        prior Open_Entry_Record for its plate, best-effort (Req 3.3).
        """
        decision = AccessDecision()
        decision.classification = Classification.RESIDENT
        gate_result = self._gate_controller.open_gate(self._event_id(ev))
        decision.gate_requested = True
        if gate_result.success:
            decision.grant_method = GrantMethod.AUTOMATIC  # (Req 4.3)
            decision.reason = "Resident matched; automatic gate-open requested."
        else:
            # The gate-open was requested but the gate reported failure; the
            # attempt still occurred (Req 4.4).
            decision.grant_method = GrantMethod.NONE
            decision.reason = (
                f"Resident matched; gate-open failed: {gate_result.detail}"
            )

        record = self._append_record(
            ev, normalized_plate, decision, resident, direction
        )
        # OUTBOUND resident: best-effort close of a prior open entry (Req 3.3).
        if direction is DirectionOutcome.OUTBOUND and record is not None:
            self._close_open_entry(normalized_plate, record)
        return decision

    def _handle_guest(
        self,
        ev: DetectionEvent,
        normalized_plate: str,
        direction: Optional[DirectionOutcome],
    ) -> AccessDecision:
        """Guest (no resident match): direction-specific handling.

        * INBOUND / None / UNDETERMINED -> surface for a manual decision; no row
          is written here (the guard's permit action writes it, Req 4.5 / 4.1).
        * OUTBOUND -> exit correlation (Req 5): look up the Open_Entry_Record; if
          found, gate + write one EXIT row + close the entry; if none, it is an
          Exit_Anomaly surfaced to the guard with no automatic grant.
        """
        if direction is DirectionOutcome.OUTBOUND:
            return self._handle_guest_exit(ev, normalized_plate)

        # INBOUND / undetermined guest: surface, write nothing now (Req 4.5).
        decision = AccessDecision()
        decision.classification = Classification.GUEST
        decision.grant_method = GrantMethod.NONE
        decision.gate_requested = False
        decision.surfaced_to_guard = True
        decision.detected_plate = normalized_plate
        decision.reason = "No resident match; surfaced for manual decision."
        return decision

    def _handle_guest_exit(
        self, ev: DetectionEvent, normalized_plate: str
    ) -> AccessDecision:
        """Correlate an exiting guest to its open entry ticket (Req 5).

        Finds the most recent Open_Entry_Record for the plate. When found, the
        exit is authorized: the gate is opened, one EXIT row is written, and the
        entry is closed (best-effort). When none is found it is an Exit_Anomaly:
        no automatic gate-open, surfaced to the guard, and no row is written here
        (the guard's manual permit writes the exit row, Req 5.3).
        """
        decision = AccessDecision()
        decision.classification = Classification.GUEST
        decision.detected_plate = normalized_plate

        try:
            open_entry = self._find_open_entry(normalized_plate)
        except Exception as exc:  # noqa: BLE001 - a lookup fault is a DB fault
            # Treat like the resident-lookup fault: no auto-open, surface, log.
            logger.error(
                "Open-entry lookup failed for plate %r: %s", normalized_plate, exc
            )
            decision.grant_method = GrantMethod.NONE
            decision.gate_requested = False
            decision.surfaced_to_guard = True
            decision.reason = (
                "Exit open-entry lookup failed; surfaced for manual decision."
            )
            return decision

        if open_entry is None:
            # Exit_Anomaly (Req 5.3): no entry ticket -> surface, no auto grant.
            decision.grant_method = GrantMethod.NONE
            decision.gate_requested = False
            decision.surfaced_to_guard = True
            decision.is_exit_anomaly = True
            decision.reason = (
                "Exit with no matching entry ticket (Exit_Anomaly); "
                "surfaced for manual decision."
            )
            return decision

        # Matched open entry (Req 5.2): gate + one EXIT row + close the entry.
        gate_result = self._gate_controller.open_gate(self._event_id(ev))
        decision.gate_requested = True
        if gate_result.success:
            decision.grant_method = GrantMethod.AUTOMATIC
            decision.reason = (
                "Exiting guest matched an open entry; automatic gate-open "
                "requested and entry closed."
            )
        else:
            decision.grant_method = GrantMethod.NONE
            decision.reason = (
                f"Exiting guest matched an open entry; gate-open failed: "
                f"{gate_result.detail}"
            )

        exit_record = self._append_record(
            ev, normalized_plate, decision, None, DirectionOutcome.OUTBOUND
        )
        if exit_record is not None:
            self._close_entry_record(open_entry, exit_record)
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _append_record(
        self,
        ev: DetectionEvent,
        normalized_plate: str,
        decision: AccessDecision,
        resident: Optional[ResidentRecord],
        direction: Optional[DirectionOutcome],
    ) -> Optional[EventRecord]:
        """Build and append exactly one record; return it (Req 4.7, 5).

        The ``direction`` determines the correlation fields:

        * ``None`` / ``UNDETERMINED`` -> flat record: ``direction=None``,
          ``event_kind=None``, ``entry_state=N/A`` (unchanged single-camera
          behavior).
        * ``INBOUND`` -> ``Event_Kind=ENTRY``, ``Entry_State=OPEN`` (this becomes
          the plate's Open_Entry_Record).
        * ``OUTBOUND`` -> ``Event_Kind=EXIT``, ``Entry_State=N/A`` (an exit never
          has its own open lifecycle; it closes a prior entry instead).

        ``camera_id`` prefers the per-detection id stamped by the pipeline
        (dual-camera mode) and falls back to the controller's configured id
        (single-camera mode); ``ANPR_Log.Camera_ID`` is NOT NULL. ``vehicle_id``
        comes from the matched whitelist row or is NULL for a guest/failed
        lookup. After the write the generated ``Log_ID`` links any captured
        image (``Images.Log_ID`` is a NOT NULL FK, so it can only follow the
        insert). The appended record is returned so the caller can close a
        correlated entry with the exit's ``Log_ID``.
        """
        event_kind, entry_state = _correlation_fields(direction)
        record = EventRecord(
            id=self._event_id(ev),
            timestamp=self._timestamp(ev),
            ocr_plate=ev.ocr_text or "",
            guard_plate=ev.guard_plate or "",
            normalized_plate=normalized_plate,
            classification=decision.classification,
            direction=_logged_direction(direction),
            grant_method=decision.grant_method,
            event_kind=event_kind,
            entry_state=entry_state,
            closed_by_event_id=None,
            image_ref=(ev.image_ref.snapshot_path if ev.image_ref else None),
            detection_confidence=_metric(ev.detection_confidence),
            ocr_confidence=_metric(ev.ocr_confidence),
            processing_latency_ms=_metric(ev.processing_latency_ms),
            environment_label=self._environment_label,
            camera_id=(ev.camera_id if ev.camera_id is not None else self._camera_id),
            vehicle_id=(resident.vehicle_id if resident is not None else None),
        )
        self._event_log_repo.append(record)
        decision.event_record = record
        self._link_image(ev, record)
        return record

    # ------------------------------------------------------------------
    # Exit correlation helpers (Req 5)
    # ------------------------------------------------------------------
    def _find_open_entry(self, normalized_plate: str) -> Optional[EventRecord]:
        """Find the Open_Entry_Record for a plate via the repository (Req 5.1).

        Raises when the repository has no ``find_open_entry`` (a flat-mode
        repository), which only happens if OUTBOUND handling is reached without
        the dual-camera repository wired -- a misconfiguration surfaced to the
        guard by the caller's fault handling rather than silently granting.
        """
        finder = getattr(self._event_log_repo, "find_open_entry", None)
        if not callable(finder):
            raise AttributeError(
                "event-log repository has no find_open_entry; exit correlation "
                "requires the dual-camera repository surface."
            )
        return finder(normalized_plate)

    def _close_entry_record(
        self, open_entry: EventRecord, exit_record: EventRecord
    ) -> None:
        """Close a matched open entry with the exit row's Log_ID (Req 5.2, 5.6).

        Best-effort: a missing close capability, a missing Log_ID, or a close
        affecting zero rows (already closed / race) is logged, never fatal -- the
        exit was already authorized and must not be blocked by a bookkeeping
        fault (Req 5.6).
        """
        closer = getattr(self._event_log_repo, "close_entry", None)
        if not callable(closer):
            logger.warning(
                "No close_entry on repository; leaving entry %s open.",
                open_entry.log_id,
            )
            return
        if open_entry.log_id is None or exit_record.log_id is None:
            logger.warning(
                "Cannot close entry: missing Log_ID (entry=%s, exit=%s).",
                open_entry.log_id,
                exit_record.log_id,
            )
            return
        try:
            affected = closer(open_entry.log_id, exit_record.log_id)
            if not affected:
                logger.info(
                    "Entry %s was already closed (0 rows affected).",
                    open_entry.log_id,
                )
        except Exception as exc:  # noqa: BLE001 - close must never fail the exit
            logger.error("Failed to close entry %s: %s", open_entry.log_id, exc)

    def _close_open_entry(
        self, normalized_plate: str, exit_record: EventRecord
    ) -> None:
        """Best-effort close of any open entry for a plate (resident exit, Req 3.3).

        Looks up the plate's Open_Entry_Record and closes it with the exit row's
        Log_ID. Any fault (lookup or close) is logged and swallowed so a resident
        exit is never blocked.
        """
        try:
            open_entry = self._find_open_entry(normalized_plate)
        except Exception as exc:  # noqa: BLE001 - lookup fault must not fail exit
            logger.error(
                "Open-entry lookup for resident exit %r failed: %s",
                normalized_plate,
                exc,
            )
            return
        if open_entry is not None:
            self._close_entry_record(open_entry, exit_record)

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
