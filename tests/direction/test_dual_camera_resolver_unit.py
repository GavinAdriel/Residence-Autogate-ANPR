"""Unit tests for the dual-camera direction resolver and its config.

Covers the dual-camera-exit-correlation spec, Requirements 2.1-2.3 (direction by
camera, unknown camera -> UNDETERMINED) and 1.5 / 7.5 (incomplete dual-camera
config refuses startup naming the offending key).
"""

from __future__ import annotations

import pytest

from anpr.config.provider import ConfigProvider
from anpr.core.models import DirectionOutcome, TrackHistory
from anpr.direction.resolver import (
    DualCameraResolver,
    SingleCameraTrajectoryResolver,
    create_direction_resolver,
)


# ---------------------------------------------------------------------------
# Resolver mapping (Req 2.1, 2.2, 2.3)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_inbound_camera_resolves_inbound() -> None:
    """A detection from the inbound camera resolves to INBOUND (Req 2.1)."""
    r = DualCameraResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    assert r.resolve_for_camera(1) is DirectionOutcome.INBOUND


@pytest.mark.unit
def test_outbound_camera_resolves_outbound() -> None:
    """A detection from the outbound camera resolves to OUTBOUND (Req 2.2)."""
    r = DualCameraResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    assert r.resolve_for_camera(2) is DirectionOutcome.OUTBOUND


@pytest.mark.unit
def test_unmapped_camera_resolves_undetermined() -> None:
    """An unmapped camera id resolves to UNDETERMINED for guard review (Req 2.3)."""
    r = DualCameraResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    assert r.resolve_for_camera(9) is DirectionOutcome.UNDETERMINED


@pytest.mark.unit
def test_missing_camera_id_resolves_undetermined() -> None:
    """A missing (None) camera id resolves to UNDETERMINED (Req 2.3)."""
    r = DualCameraResolver({1: DirectionOutcome.INBOUND})
    assert r.resolve_for_camera(None) is DirectionOutcome.UNDETERMINED


@pytest.mark.unit
def test_resolve_delegates_via_track_camera_id() -> None:
    """The Protocol ``resolve(track)`` delegates using ``track.camera_id``."""
    r = DualCameraResolver({2: DirectionOutcome.OUTBOUND})
    track = TrackHistory(track_id=5, camera_id=2)
    assert r.resolve(track) is DirectionOutcome.OUTBOUND


@pytest.mark.unit
def test_map_accepts_string_direction_values() -> None:
    """String direction values from config coerce to DirectionOutcome."""
    r = DualCameraResolver({1: "INBOUND", 2: "outbound"})
    assert r.resolve_for_camera(1) is DirectionOutcome.INBOUND
    assert r.resolve_for_camera(2) is DirectionOutcome.OUTBOUND


# ---------------------------------------------------------------------------
# Factory selection + config map (Req 2.5)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_factory_builds_single_resolver_by_default() -> None:
    """Default mode builds the single-camera trajectory resolver (unchanged)."""
    resolver = create_direction_resolver(ConfigProvider(env={}))
    assert isinstance(resolver, SingleCameraTrajectoryResolver)


@pytest.mark.unit
def test_factory_builds_dual_resolver_with_camera_map() -> None:
    """dual_camera mode builds a resolver mapping the two configured camera ids."""
    env = {
        "ANPR_DIRECTION__MODE": "dual_camera",
        "ANPR_DIRECTION__CAMERAS__INBOUND__STREAM_URL": "rtsp://in/stream",
        "ANPR_DIRECTION__CAMERAS__OUTBOUND__STREAM_URL": "rtsp://out/stream",
    }
    resolver = create_direction_resolver(ConfigProvider(env=env))
    assert isinstance(resolver, DualCameraResolver)
    assert resolver.camera_direction_map == {
        1: DirectionOutcome.INBOUND,
        2: DirectionOutcome.OUTBOUND,
    }


# ---------------------------------------------------------------------------
# Dual-camera config validation (Req 1.5, 7.5)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_valid_dual_camera_config_validates_cleanly() -> None:
    """A complete dual-camera config reports no error."""
    env = {
        "ANPR_DIRECTION__MODE": "dual_camera",
        "ANPR_DIRECTION__CAMERAS__INBOUND__STREAM_URL": "rtsp://in/stream",
        "ANPR_DIRECTION__CAMERAS__OUTBOUND__STREAM_URL": "rtsp://out/stream",
    }
    errors = ConfigProvider(env=env).validate()
    assert errors == [], f"unexpected errors: {errors}"


@pytest.mark.unit
def test_incomplete_dual_camera_config_refuses_startup_naming_keys() -> None:
    """Missing ip stream URLs in dual mode are reported per offending key (Req 1.5, 7.5)."""
    env = {"ANPR_DIRECTION__MODE": "dual_camera"}  # type ip but no stream urls
    errors = ConfigProvider(env=env).validate()
    keys = {e.key for e in errors}
    assert "direction.cameras.inbound.stream_url" in keys
    assert "direction.cameras.outbound.stream_url" in keys


@pytest.mark.unit
def test_invalid_direction_mode_is_reported() -> None:
    """An unknown direction.mode is a configuration error."""
    errors = ConfigProvider(env={"ANPR_DIRECTION__MODE": "sideways"}).validate()
    assert any(e.key == "direction.mode" for e in errors)
