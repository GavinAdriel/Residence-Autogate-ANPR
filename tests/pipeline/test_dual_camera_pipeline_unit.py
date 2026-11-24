"""Dual-camera pipeline behavior (dual-camera-exit-correlation).

Covers camera-scoped tracking (Req 6.1), one-decision-per-pass direction
stamping (Req 2.4, 6.2), and the reconciliation window that suppresses a
same-pass OUTBOUND double glimpse while still allowing a genuine later exit
(Req 6.3, 6.4).

Reuses the fakes/builders from ``test_pipeline`` and adds a direction-aware fake
access controller.
"""

from __future__ import annotations

import pytest

from anpr.core.models import (
    AccessDecision,
    BoundingBox,
    Classification,
    Detection,
    DirectionOutcome,
    GrantMethod,
    OcrResult,
)
from anpr.direction.resolver import CameraDirectionResolver
from anpr.pipeline.pipeline import RecentEntryRegistry
from tests.pipeline.test_pipeline import (
    FakeClock,
    FakeDetector,
    FakeOcr,
    FakeSource,
    a_frame,
    make_pipeline,
)


class DirectionAwareAccess:
    """Fake controller accepting an optional direction; records (ev, direction)."""

    def __init__(self, decision: AccessDecision | None = None) -> None:
        self._decision = decision or AccessDecision(
            classification=Classification.RESIDENT,
            grant_method=GrantMethod.AUTOMATIC,
            gate_requested=True,
        )
        self.calls: list[tuple[object, object]] = []

    def handle_detection(self, ev, direction=None) -> AccessDecision:
        self.calls.append((ev, direction))
        return self._decision


def _det(track_id: int) -> Detection:
    return Detection(box=BoundingBox(0, 0, 20, 10), confidence=0.9, track_id=track_id)


def _dual_pipeline(*, camera_id, resolver, access, registry, clock=None, window=3.0):
    """Build a pipeline wired for dual-camera mode."""
    return make_pipeline(
        detector=FakeDetector([_det(1)]),
        ocr=FakeOcr(OcrResult(text="B1234CD", confidence=0.95)),
        access=access,
        clock=clock,
        camera_id=camera_id,
        direction_resolver=resolver,
        recent_entry_registry=registry,
        reconciliation_window_s=window,
    )


# ---------------------------------------------------------------------------
# Direction stamping (Req 2.4)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_inbound_camera_stamps_inbound_direction() -> None:
    """The inbound camera's events are passed to the controller as INBOUND."""
    resolver = CameraDirectionResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    access = DirectionAwareAccess()
    registry = RecentEntryRegistry()
    pipeline, *_ = _dual_pipeline(
        camera_id=1, resolver=resolver, access=access, registry=registry
    )

    pipeline.process_frame(a_frame(), acquired_at=0.0)

    assert len(access.calls) == 1
    ev, direction = access.calls[0]
    assert direction is DirectionOutcome.INBOUND
    assert ev.camera_id == 1
    assert ev.direction is DirectionOutcome.INBOUND


@pytest.mark.unit
def test_outbound_camera_stamps_outbound_direction() -> None:
    """The outbound camera's events are passed to the controller as OUTBOUND."""
    resolver = CameraDirectionResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    access = DirectionAwareAccess()
    registry = RecentEntryRegistry()
    pipeline, *_ = _dual_pipeline(
        camera_id=2, resolver=resolver, access=access, registry=registry
    )

    pipeline.process_frame(a_frame(), acquired_at=0.0)

    _, direction = access.calls[0]
    assert direction is DirectionOutcome.OUTBOUND


# ---------------------------------------------------------------------------
# Camera-scoped tracking (Req 6.1)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_two_cameras_same_track_id_do_not_collide() -> None:
    """Identical track_id on two pipelines (own aggregators) are independent (Req 6.1)."""
    resolver = CameraDirectionResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    registry = RecentEntryRegistry()

    in_access = DirectionAwareAccess()
    out_access = DirectionAwareAccess(
        AccessDecision(classification=Classification.GUEST, surfaced_to_guard=True)
    )
    in_pipe, *_ = _dual_pipeline(
        camera_id=1, resolver=resolver, access=in_access, registry=registry
    )
    out_pipe, *_ = _dual_pipeline(
        camera_id=2, resolver=resolver, access=out_access, registry=registry
    )

    # Both pipelines see a detection with track_id=1 but they are different cars.
    in_pipe.process_frame(a_frame(), acquired_at=0.0)
    out_pipe.process_frame(a_frame(), acquired_at=10.0)

    # Each pipeline processed its own track_id=1 independently (no cross-skip).
    assert len(in_access.calls) == 1
    assert len(out_access.calls) == 1
    # Separate aggregators: the inbound resident track is resolved only on the
    # inbound pipeline, not the outbound one.
    assert in_pipe.track_aggregator.is_resolved(1) is True
    assert out_pipe.track_aggregator.is_resolved(1) is False


# ---------------------------------------------------------------------------
# Reconciliation window (Req 6.2, 6.3, 6.4)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_same_pass_outbound_glimpse_is_suppressed() -> None:
    """An OUTBOUND read within the window of the INBOUND entry is not exited (Req 6.3)."""
    resolver = CameraDirectionResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    registry = RecentEntryRegistry()
    clock = FakeClock()

    # Inbound resident enters at t=0 -> records a recent entry.
    in_access = DirectionAwareAccess()  # RESIDENT by default
    in_pipe, *_ = _dual_pipeline(
        camera_id=1, resolver=resolver, access=in_access, registry=registry, clock=clock
    )
    in_pipe.process_frame(a_frame(), acquired_at=0.0)
    assert len(in_access.calls) == 1  # entry handled

    # Outbound camera glimpses the SAME plate 1s later (within the 3s window).
    out_access = DirectionAwareAccess()
    out_pipe, *_ = _dual_pipeline(
        camera_id=2, resolver=resolver, access=out_access, registry=registry, window=3.0
    )
    out_pipe.process_frame(a_frame(), acquired_at=1.0)

    # The outbound read was suppressed: the controller was NOT asked to exit it.
    assert out_access.calls == []


@pytest.mark.unit
def test_genuine_later_exit_outside_window_correlates() -> None:
    """An OUTBOUND read after the window is a real exit and reaches the controller (Req 6.4)."""
    resolver = CameraDirectionResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    registry = RecentEntryRegistry()

    in_access = DirectionAwareAccess()
    in_pipe, *_ = _dual_pipeline(
        camera_id=1, resolver=resolver, access=in_access, registry=registry
    )
    in_pipe.process_frame(a_frame(), acquired_at=0.0)

    out_access = DirectionAwareAccess()
    out_pipe, *_ = _dual_pipeline(
        camera_id=2, resolver=resolver, access=out_access, registry=registry, window=3.0
    )
    # 100 seconds later: well outside the window -> a genuine exit.
    out_pipe.process_frame(a_frame(), acquired_at=100.0)

    assert len(out_access.calls) == 1
    _, direction = out_access.calls[0]
    assert direction is DirectionOutcome.OUTBOUND
