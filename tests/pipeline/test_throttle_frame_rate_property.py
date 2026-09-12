"""Property-based test for the detection pipeline's frame-rate throttling.

Feature: mysql-monitoring-migration, Property 12
Property 12: Ingestion holds at the configured frame rate.

``DetectionPipeline._throttle`` (Req 7.1, preserved unchanged by the monitoring
migration) sleeps as needed so consecutive ingest cycles are separated by at
least the configured frame interval ``1 / target_fps``. Frames that would arrive
faster than the configured rate are discarded by the throttle sleep, so the
pipeline never processes more than ``target_fps`` frames per second.

Reusing the existing unit-test approach (a deterministic fake clock whose
``sleep`` advances virtual time), this test drives many throttle cycles with
arbitrary per-cycle "work" durations (the time a frame's detect/OCR/normalize
work would take) and asserts, for any generated schedule:

* every consecutive pair of processed cycles is at least one frame interval
  apart (the mechanism that discards excess frames);
* the effective rate over the whole run never exceeds ``target_fps``; and
* each throttle sleep is non-negative and never exceeds one frame interval.

No real camera, model, or wall-clock delay is involved.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from anpr.pipeline.pipeline import DetectionPipeline


# Tolerance for float accumulation across many virtual-clock additions.
_EPS = 1e-9


class FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances virtual time.

    Mirrors the fake clock used by the pipeline's example-based tests: reads
    return the current virtual time and ``sleep`` both records the requested
    duration and advances the clock, so throttling is exercised with no real
    delay.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.t = start
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self.slept.append(secs)
        self.t += secs


def _make_pipeline(target_fps: int, clock: FakeClock) -> DetectionPipeline:
    """Build a pipeline wired only for throttling (ports are inert stubs).

    ``_throttle`` touches only the injected monotonic/sleep seams and the
    derived frame interval, so the video source, detector, OCR engine,
    normalizer, and access controller are never invoked and can be plain
    sentinels.
    """
    return DetectionPipeline(
        object(),  # video_source
        object(),  # detector
        object(),  # ocr_engine
        object(),  # normalizer
        object(),  # access_controller
        target_fps=target_fps,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    target_fps=st.integers(min_value=1, max_value=60),
    work_durations=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=40,
    ),
)
def test_throttle_holds_at_configured_frame_rate(target_fps, work_durations) -> None:
    """Validates: Requirements 7.1

    For any configured frame rate and any schedule of per-cycle work durations
    (including frames arriving faster than the rate, i.e. zero work), the
    throttle keeps consecutive processed cycles at least one frame interval
    apart and holds the effective ingestion rate at no more than ``target_fps``.
    """
    interval = 1.0 / target_fps
    clock = FakeClock()
    pipeline = _make_pipeline(target_fps, clock)

    # The effective (clamped) rate matches what we asked for, so ``interval`` is
    # the real throttle target.
    assert pipeline.target_fps == float(target_fps)

    # Drive one throttle per cycle, then advance the clock by the frame's work
    # to model the ingest loop's detect/read/process time between throttles.
    cycle_starts: list[float] = []
    for work in work_durations:
        pipeline._throttle()
        # After ``_throttle`` returns, the virtual clock sits exactly at the
        # cycle-start mark it recorded internally.
        cycle_starts.append(clock.monotonic())
        clock.t += work

    # (1) Excess frames are discarded: consecutive processed cycles are always
    #     at least one frame interval apart.
    for previous, current in zip(cycle_starts, cycle_starts[1:]):
        assert current - previous >= interval - _EPS

    # (2) Ingestion holds at no more than the configured rate over the whole
    #     run: with N marks spanning duration D there are N-1 intervals, and
    #     D >= (N-1) * interval, so the effective rate (N-1)/D <= target_fps.
    if len(cycle_starts) >= 2:
        span = cycle_starts[-1] - cycle_starts[0]
        processed_intervals = len(cycle_starts) - 1
        assert processed_intervals <= target_fps * span + _EPS

    # (3) Throttling only ever waits, and never waits longer than one frame
    #     interval (it discards excess by sleeping the remaining time).
    for slept in clock.slept:
        assert slept >= 0.0
        assert slept <= interval + _EPS
