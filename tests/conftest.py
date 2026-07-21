"""Shared pytest configuration for the ANPR Autogate System test suite.

Registers a Hypothesis settings profile that runs a minimum of 100 examples,
per the design's property-based testing strategy (each of the 33 correctness
properties runs at least 100 generated iterations). The profile is loaded
automatically when Hypothesis is installed; the import is guarded so the test
session still collects before dependencies are installed.
"""

from __future__ import annotations

try:
    from hypothesis import HealthCheck, settings

    # Minimum 100 examples for every property-based test (design requirement).
    settings.register_profile(
        "anpr",
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.load_profile("anpr")
except ModuleNotFoundError:  # pragma: no cover - before deps are installed
    pass
