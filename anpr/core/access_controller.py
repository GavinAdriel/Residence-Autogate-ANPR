"""Access-control decision logic for the ANPR Autogate System.

The :class:`AccessController` turns an enriched :class:`DetectionEvent` into an
:class:`AccessDecision`: it classifies a recognized plate as Resident or Guest,
requests the gate-open action for residents, records the access attempt in the
``Event_Log``, and surfaces events to the Guard_Dashboard when a manual decision
is required.

The controller is pure orchestration over injected collaborators and depends
only on the Protocol interfaces defined in ``anpr.core.interfaces``
(``ResidentRepository``, ``EventLogRepository``, ``GateController``) -- never on
a concrete implementation -- so simulated and field adapters are structurally
interchangeable and chosen only in the composition root (Requirement 14.4).

Inbound handling (Requirement 5, 8):

* Query the ``Resident_Database`` for an *exact* normalized-plate match on
  ``ev.normalized_plate`` (Req 5.1).
* A match classifies the plate as Resident and requests the gate-open action
  via ``GateController.open_gate`` (Req 5.2, 5.4). Because that call is a direct
  synchronous invocation there is no artificial delay, so the classification-to
  -request limit (configurable, <= 2 s) is trivially met.
* No match classifies the plate as Guest: no automatic gate-open is requested
  and the event is surfaced to the guard for a manual decision (Req 5.3, 5.5).
* A ``Resident_Database`` query fault requests no automatic gate-open, logs the
  failure together with the normalized plate string, and surfaces the event to
  the guard (Req 5.6).
* When the inbound event results in an access attempt, an Entry_Event is
  recorded as an Open_Entry_Record on the *same* normalized plate used for
  resident matching, so it is correlatable with a later Exit_Event (Req 8.1,
  8.2). If recording the Entry_Event raises a technical fault, the recording is
  abandoned without retry, no partial record or Open_Entry_Record is created,
  and -- per Req 8.3 -- no alert is surfaced to operators.

Outbound handling (Requirement 9):

* Query the ``Event_Log`` for Open_Entry_Records with an *exact* normalized
  -plate match on ``ev.normalized_plate`` (Req 9.1); the repository returns
  them ordered earliest-first.
* Exactly one open entry requests the gate-open, records an Exit_Event, and
  closes the matching Open_Entry_Record (Req 9.2). More than one closes the
  earliest-timestamp record (the first returned) and records an Exit_Event
  referencing it (Req 9.3).
* No open entry is an Exit_Anomaly: no automatic gate-open is requested and the
  event is surfaced to the guard as an alert requiring a Manual_Override
  decision, with a single anomaly record written to the Event_Log (Req 9.4,
  9.5).
* On an exit write/close fault the recording is abandoned, the matching
  Open_Entry_Record is left open, the failure is logged with the plate, and --
  unlike the inbound entry fault (Req 8.3) -- the event *is* surfaced to the
  guard for manual resolution (Req 9.6). A completed entry followed by a
  completed exit leaves zero Open_Entry_Records for that plate (Req 9.7).

See .kiro/specs/anpr-autogate-system/design.md (Access_Controller section) and
requirements.md (Requirements 5, 8, and 9) for the authoritative definitions.
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
    DirectionOutcome,
    EntryState,
    EnvironmentLabel,
    EventKind,
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


class AccessController:
    """Classifies plates and orchestrates gate/logging for access attempts.

    Parameters
    ----------
    resident_repo:
        Exact-match lookup into the ``Resident_Database`` (Req 5.1).
    event_log_repo:
        Append-and-correlate access to the ``Event_Log`` (Req 8, 9, 10).
    gate_controller:
        The gate-open abstraction shared with Manual_Override (Req 6.1).
    environment_label:
        Deployment environment stamped onto every written record (Req 15.2);
        optional so pure-logic tests need not supply one.
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
    # Inbound
    # ------------------------------------------------------------------
    def handle_inbound(self, ev: DetectionEvent) -> AccessDecision:
        """Classify an inbound detection and drive gate/logging (Req 5, 8).

        Returns an :class:`AccessDecision` describing the classification, the
        grant method, whether a gate-open was requested, and whether the event
        was surfaced to the guard for a manual decision.
        """
        normalized_plate = ev.normalized_plate or ""

        decision = AccessDecision()

        # --- Resident lookup (Req 5.1) -------------------------------------
        try:
            resident = self._resident_repo.find_by_plate(normalized_plate)
        except Exception as exc:  # noqa: BLE001 - any query fault is a DB fault
            # Resident_Database query fault (Req 5.6): no auto-open, log with
            # the normalized plate, surface to the guard for a manual decision.
            logger.error(
                "Resident_Database query failed for plate %r: %s",
                normalized_plate,
                exc,
            )
            decision.classification = None
            decision.grant_method = GrantMethod.NONE
            decision.gate_requested = False
            decision.surfaced_to_guard = True
            decision.reason = (
                "Resident_Database query failed; surfaced for manual decision."
            )
            self._record_entry_event(ev, normalized_plate, decision)
            return decision

        if resident is not None:
            # --- Resident (Req 5.2, 5.4) -----------------------------------
            decision.classification = Classification.RESIDENT
            gate_result = self._gate_controller.open_gate(self._event_id(ev))
            decision.gate_requested = True
            if gate_result.success:
                decision.grant_method = GrantMethod.AUTOMATIC
                decision.reason = "Resident matched; automatic gate-open requested."
            else:
                # The gate-open was requested within the limit but the gate
                # reported failure; the attempt still occurred.
                decision.grant_method = GrantMethod.NONE
                decision.reason = (
                    f"Resident matched; gate-open requested but failed: "
                    f"{gate_result.detail}"
                )
        else:
            # --- Guest (Req 5.3, 5.5) --------------------------------------
            decision.classification = Classification.GUEST
            decision.grant_method = GrantMethod.NONE
            decision.gate_requested = False
            decision.surfaced_to_guard = True
            decision.reason = "No resident match; surfaced for manual decision."

        # --- Record the Entry_Event as an Open_Entry_Record (Req 8.1, 8.2) -
        self._record_entry_event(ev, normalized_plate, decision)
        return decision

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------
    def handle_outbound(self, ev: DetectionEvent) -> AccessDecision:
        """Validate an outbound detection against recorded entries (Req 9).

        Queries the ``Event_Log`` for Open_Entry_Records whose normalized plate
        exactly matches ``ev.normalized_plate`` (Req 9.1) and drives the exit
        flow:

        * Exactly one open entry (Req 9.2): request the gate-open, record an
          Exit_Event, and close the matching Open_Entry_Record.
        * More than one open entry (Req 9.3): close the record with the
          earliest entry timestamp (the first returned, since results are
          ordered by timestamp then id), record an Exit_Event referencing it,
          and request the gate-open.
        * No open entry (Req 9.4, 9.5): classify as an Exit_Anomaly -- no
          automatic gate-open -- and surface it to the guard as an alert
          requiring a Manual_Override decision. An anomaly record is written.

        On an exit write/close fault (Req 9.6) the matching Open_Entry_Record is
        left open and -- unlike the inbound entry fault (Req 8.3) -- the event
        *is* surfaced to the guard for manual resolution.

        Returns an :class:`AccessDecision` describing the classification, grant
        method, whether a gate-open was requested, and whether the event was
        surfaced to the guard.
        """
        normalized_plate = ev.normalized_plate or ""
        decision = AccessDecision()

        open_entries = self._event_log_repo.find_open_entries(normalized_plate)

        if not open_entries:
            # --- Exit_Anomaly (Req 9.4, 9.5) -------------------------------
            self._record_exit_anomaly(ev, normalized_plate, decision)
            return decision

        # --- Matched entry: single (Req 9.2) or earliest of many (Req 9.3) -
        # The repo orders results by timestamp then id, so the first record is
        # the earliest entry and is the one to close in either case.
        entry = open_entries[0]
        # Carry the entry's classification onto the exit decision when known.
        decision.classification = entry.classification

        exit_id = self._event_id(ev)

        # Request the gate-open first so the persisted Exit_Event reflects the
        # actual grant outcome (Req 9.2, 9.3).
        gate_result = self._gate_controller.open_gate(exit_id)
        decision.gate_requested = True
        if gate_result.success:
            decision.grant_method = GrantMethod.AUTOMATIC
        else:
            decision.grant_method = GrantMethod.NONE

        exit_record = self._build_record(
            ev,
            normalized_plate,
            event_id=exit_id,
            classification=entry.classification,
            grant_method=decision.grant_method,
            event_kind=EventKind.EXIT,
            entry_state=EntryState.NA,
        )

        # Order of operations (Req 9.6, 9.7): append the Exit_Event, then close
        # the Open_Entry_Record referencing that same exit id. If the append
        # fails the entry stays open; if the close fails it is a single UPDATE
        # so nothing is closed -- in either case leave the entry open and
        # surface to the guard for manual resolution.
        try:
            self._event_log_repo.append(exit_record)
            self._event_log_repo.close_open_entry(entry.id, exit_id)
        except Exception as exc:  # noqa: BLE001 - any exit write/close fault
            logger.error(
                "Exit_Event write/close failed for plate %r: %s",
                normalized_plate,
                exc,
            )
            decision.event_record = None
            decision.surfaced_to_guard = True
            decision.reason = (
                "Exit_Event write/close failed; Open_Entry_Record left open "
                "and surfaced for manual resolution."
            )
            return decision

        decision.event_record = exit_record
        if gate_result.success:
            decision.reason = (
                "Outbound matched Open_Entry_Record; automatic gate-open "
                "requested and entry closed."
            )
        else:
            decision.reason = (
                f"Outbound matched Open_Entry_Record; gate-open requested but "
                f"failed: {gate_result.detail}"
            )
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _event_id(self, ev: DetectionEvent) -> str:
        """Return the event's id, generating a fresh one when absent."""
        return ev.event_id or self._id_factory()

    def _record_entry_event(
        self,
        ev: DetectionEvent,
        normalized_plate: str,
        decision: AccessDecision,
    ) -> None:
        """Append the inbound Entry_Event as an Open_Entry_Record (Req 8.1-8.3).

        Builds an :class:`EventRecord` on the *same* normalized plate used for
        resident matching and appends it via ``EventLogRepository.append``. Per
        Req 8.3, if the append raises a technical fault the recording is
        abandoned without retry -- the repository's atomic write leaves no
        partial record and no Open_Entry_Record -- and *no* alert is surfaced to
        operators (the guard surfacing state is left unchanged).
        """
        record = EventRecord(
            id=self._event_id(ev),
            timestamp=self._timestamp(ev),
            ocr_plate=ev.ocr_text or "",
            guard_plate=ev.guard_plate or "",
            normalized_plate=normalized_plate,
            classification=decision.classification,
            direction=ev.direction or DirectionOutcome.INBOUND,
            grant_method=decision.grant_method,
            event_kind=EventKind.ENTRY,
            entry_state=EntryState.OPEN,
            image_ref=(ev.image_ref.snapshot_path if ev.image_ref else None),
            detection_confidence=(
                ev.detection_confidence
                if ev.detection_confidence is not None
                else NA_SENTINEL
            ),
            ocr_confidence=(
                ev.ocr_confidence if ev.ocr_confidence is not None else NA_SENTINEL
            ),
            processing_latency_ms=(
                ev.processing_latency_ms
                if ev.processing_latency_ms is not None
                else NA_SENTINEL
            ),
            environment_label=self._environment_label,
        )

        try:
            self._event_log_repo.append(record)
        except Exception as exc:  # noqa: BLE001 - any append fault (Req 8.3)
            # Req 8.3: abandon the recording without retrying; the repo's atomic
            # write leaves no partial record and no Open_Entry_Record; do NOT
            # surface an alert to operators for an entry recording fault.
            logger.error(
                "Entry_Event recording failed for plate %r: %s",
                normalized_plate,
                exc,
            )
            decision.event_record = None
            return

        decision.event_record = record

    def _record_exit_anomaly(
        self,
        ev: DetectionEvent,
        normalized_plate: str,
        decision: AccessDecision,
    ) -> None:
        """Classify and record an Exit_Anomaly (Req 9.4, 9.5).

        An outbound detection with no matching Open_Entry_Record is an
        Exit_Anomaly: no automatic gate-open is requested and the event is
        surfaced to the guard as an alert requiring a Manual_Override decision.
        A single anomaly Event_Log record is written with
        ``event_kind == ANOMALY``, ``direction == OUTBOUND``,
        ``grant_method == NONE``, and ``entry_state == N/A`` (Req 10.1).
        """
        decision.grant_method = GrantMethod.NONE
        decision.gate_requested = False
        decision.surfaced_to_guard = True
        decision.reason = (
            "Exit_Anomaly: no matching Open_Entry_Record; surfaced for "
            "Manual_Override decision."
        )

        record = self._build_record(
            ev,
            normalized_plate,
            event_id=self._event_id(ev),
            classification=decision.classification,
            grant_method=GrantMethod.NONE,
            event_kind=EventKind.ANOMALY,
            entry_state=EntryState.NA,
        )
        self._event_log_repo.append(record)
        decision.event_record = record

    def _build_record(
        self,
        ev: DetectionEvent,
        normalized_plate: str,
        *,
        event_id: str,
        classification: Optional[Classification],
        grant_method: GrantMethod,
        event_kind: EventKind,
        entry_state: EntryState,
    ) -> EventRecord:
        """Build an outbound :class:`EventRecord` for the Event_Log.

        Shared by the Exit_Event and Exit_Anomaly paths (Req 9). The record is
        stamped as ``OUTBOUND`` and carries the same metric/image fields as the
        inbound entry record, with absent numeric metrics represented by the
        ``NA_SENTINEL`` (Req 10.4).
        """
        return EventRecord(
            id=event_id,
            timestamp=self._timestamp(ev),
            ocr_plate=ev.ocr_text or "",
            guard_plate=ev.guard_plate or "",
            normalized_plate=normalized_plate,
            classification=classification,
            direction=DirectionOutcome.OUTBOUND,
            grant_method=grant_method,
            event_kind=event_kind,
            entry_state=entry_state,
            image_ref=(ev.image_ref.snapshot_path if ev.image_ref else None),
            detection_confidence=(
                ev.detection_confidence
                if ev.detection_confidence is not None
                else NA_SENTINEL
            ),
            ocr_confidence=(
                ev.ocr_confidence if ev.ocr_confidence is not None else NA_SENTINEL
            ),
            processing_latency_ms=(
                ev.processing_latency_ms
                if ev.processing_latency_ms is not None
                else NA_SENTINEL
            ),
            environment_label=self._environment_label,
        )

    def _timestamp(self, ev: DetectionEvent) -> str:
        """Return an ISO-8601 timestamp for the record (Req 10.3).

        Prefers the event's own timestamp when present, otherwise stamps the
        injected clock's current time.
        """
        ts = ev.timestamp or self._clock()
        return ts.isoformat()
