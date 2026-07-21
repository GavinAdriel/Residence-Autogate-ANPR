"""Gate controller implementations for the ANPR Autogate System.

The gate-open action lives behind the ``GateController`` interface (defined in
``anpr.core.interfaces``) so that the ``Access_Controller`` and Manual_Override
flows depend only on the abstraction, never on a concrete gate. Two concrete
controllers are provided and selected purely by configuration in the
composition root (Requirement 6.4):

* :class:`SimulatedGate` (``gate.mode = simulation``) logs the gate-open action
  as a logged event, drives no physical hardware, and always reports success
  (Requirement 6.2).
* :class:`HardwareGate` (``gate.mode = hardware``) issues the gate-open signal
  through a configured hardware interface abstraction and reports success only
  when the interface acknowledges the signal within a configurable response
  timeout between 1 and 30 seconds (default 3 s) (Requirement 6.3). On any
  failure -- a response-timeout being exceeded or an interface error -- it logs
  the failure together with the associated event identifier and reports failure
  to the caller (Requirement 6.5).

Both controllers return a :class:`~anpr.core.models.GateResult` from a single
``open_gate(event_id)`` operation (Requirement 6.1).

The hardware interface and the clock/sleep primitives used by
:class:`HardwareGate` are injected, so the acknowledgement, timeout, and
interface-error branches can be unit tested deterministically with a fake clock
and a mock interface. See .kiro/specs/anpr-autogate-system/requirements.md
(Requirement 6) for the authoritative acceptance criteria.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Protocol, runtime_checkable

from anpr.core.models import GateResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response-timeout bounds for the hardware gate (Req 6.3)
# ---------------------------------------------------------------------------

# The hardware acknowledgement response timeout is configurable between 1 and
# 30 seconds and defaults to 3 seconds.
MIN_RESPONSE_TIMEOUT_S = 1.0
MAX_RESPONSE_TIMEOUT_S = 30.0
DEFAULT_RESPONSE_TIMEOUT_S = 3.0

# Default interval between acknowledgement polls while waiting on the hardware.
_DEFAULT_POLL_INTERVAL_S = 0.05


class HardwareInterfaceError(Exception):
    """Raised by a hardware interface when it cannot issue the gate signal."""


@runtime_checkable
class HardwareInterface(Protocol):
    """Abstraction over the physical gate signalling hardware (relay/PLC).

    This is the seam that a field relay/PLC driver implements. It is kept
    deliberately thin -- issue a signal, then report whether an acknowledgement
    has arrived -- so the :class:`HardwareGate` owns the timeout policy and can
    be exercised with a mock in unit tests.
    """

    def send_open_signal(self, event_id: str) -> None:
        """Issue the gate-open signal.

        Raises :class:`HardwareInterfaceError` (or any exception) when the
        signal cannot be issued.
        """
        ...

    def poll_acknowledgement(self, event_id: str) -> bool:
        """Return True once the hardware has acknowledged the open signal."""
        ...


def clamp_response_timeout(timeout_s: Optional[float]) -> float:
    """Clamp a configured response timeout to the valid 1-30 s range (Req 6.3).

    ``None`` or a non-numeric value falls back to the 3 s default; values below
    1 s or above 30 s are clamped to the nearest bound so a misconfiguration can
    never produce an out-of-range wait.
    """
    if timeout_s is None or isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        return DEFAULT_RESPONSE_TIMEOUT_S
    return float(min(MAX_RESPONSE_TIMEOUT_S, max(MIN_RESPONSE_TIMEOUT_S, timeout_s)))


class SimulatedGate:
    """Simulated gate controller (``gate.mode = simulation``).

    Records the gate-open action as a logged event, drives no physical
    hardware, and always reports success (Requirement 6.2). Structurally
    satisfies the ``GateController`` Protocol.
    """

    def open_gate(self, event_id: str) -> GateResult:
        """Log the simulated gate-open action and report success (Req 6.2)."""
        logger.info("Simulated gate-open action for event %s", event_id)
        return GateResult(
            success=True,
            detail=f"Simulated gate-open succeeded for event {event_id}.",
        )


class HardwareGate:
    """Hardware gate controller (``gate.mode = hardware``).

    Issues the gate-open signal through an injected :class:`HardwareInterface`
    and reports success only when the interface acknowledges within the
    configured response timeout (1-30 s, default 3 s) (Requirement 6.3). Any
    failure -- interface error or timeout -- is logged with the associated event
    identifier and reported as failure to the caller (Requirement 6.5).

    Parameters
    ----------
    interface:
        The hardware signalling abstraction (relay/PLC driver).
    response_timeout_s:
        Configured acknowledgement timeout; clamped to the 1-30 s range.
    clock:
        Monotonic time source returning seconds; injectable for tests.
    sleep:
        Sleep primitive used between acknowledgement polls; injectable so a
        fake clock can advance time deterministically in tests.
    poll_interval_s:
        Interval between acknowledgement polls while awaiting the ack.
    """

    def __init__(
        self,
        interface: HardwareInterface,
        response_timeout_s: Optional[float] = DEFAULT_RESPONSE_TIMEOUT_S,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self._interface = interface
        self._response_timeout_s = clamp_response_timeout(response_timeout_s)
        self._clock = clock
        self._sleep = sleep
        self._poll_interval_s = poll_interval_s

    @property
    def response_timeout_s(self) -> float:
        """The effective (clamped) acknowledgement response timeout in seconds."""
        return self._response_timeout_s

    def open_gate(self, event_id: str) -> GateResult:
        """Issue the gate-open signal and await acknowledgement (Req 6.3, 6.5).

        Returns a successful :class:`GateResult` when the hardware interface
        acknowledges within the response timeout, otherwise logs the failure
        with the event identifier and returns a failure result.
        """
        # Issue the signal. An interface error here is a hardware failure: log
        # it with the event id and report failure (Req 6.5).
        try:
            self._interface.send_open_signal(event_id)
        except Exception as exc:  # noqa: BLE001 - any interface fault is a failure
            detail = f"Hardware interface error issuing gate-open signal: {exc}"
            logger.error("Gate-open failed for event %s: %s", event_id, detail)
            return GateResult(success=False, detail=detail)

        # Poll for acknowledgement until the response timeout elapses (Req 6.3).
        deadline = self._clock() + self._response_timeout_s
        while True:
            try:
                acknowledged = self._interface.poll_acknowledgement(event_id)
            except Exception as exc:  # noqa: BLE001 - any interface fault is a failure
                detail = f"Hardware interface error awaiting acknowledgement: {exc}"
                logger.error("Gate-open failed for event %s: %s", event_id, detail)
                return GateResult(success=False, detail=detail)

            if acknowledged:
                logger.info("Gate-open acknowledged by hardware for event %s", event_id)
                return GateResult(
                    success=True,
                    detail=f"Hardware gate-open acknowledged for event {event_id}.",
                )

            # Timed out waiting for the acknowledgement (Req 6.5).
            if self._clock() >= deadline:
                detail = (
                    "Hardware gate-open acknowledgement timed out after "
                    f"{self._response_timeout_s:g} s."
                )
                logger.error("Gate-open failed for event %s: %s", event_id, detail)
                return GateResult(success=False, detail=detail)

            self._sleep(self._poll_interval_s)
