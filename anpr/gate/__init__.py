"""Gate controller sub-package.

Exposes the concrete gate controllers and the hardware interface abstraction so
the composition root can select an implementation from configuration
(Requirement 6.4).
"""

from anpr.gate.controller import (
    DEFAULT_RESPONSE_TIMEOUT_S,
    MAX_RESPONSE_TIMEOUT_S,
    MIN_RESPONSE_TIMEOUT_S,
    HardwareGate,
    HardwareInterface,
    HardwareInterfaceError,
    SimulatedGate,
    clamp_response_timeout,
)

__all__ = [
    "SimulatedGate",
    "HardwareGate",
    "HardwareInterface",
    "HardwareInterfaceError",
    "clamp_response_timeout",
    "MIN_RESPONSE_TIMEOUT_S",
    "MAX_RESPONSE_TIMEOUT_S",
    "DEFAULT_RESPONSE_TIMEOUT_S",
]
