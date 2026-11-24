"""Unit tests for the camera-id direction resolver and the ``cameras`` config.

Direction comes only from which camera saw the vehicle: ``cameras.inbound`` is
INBOUND and ``cameras.outbound`` is OUTBOUND (dual-camera-exit-correlation
Requirements 2.1-2.3). An incomplete ``cameras`` block refuses startup naming the
offending key (Requirements 1.5 / 7.5).
"""

from __future__ import annotations

import pytest

from anpr.config.provider import ConfigProvider
from anpr.core.models import DirectionOutcome
from anpr.direction.resolver import CameraDirectionResolver, create_direction_resolver


# ---------------------------------------------------------------------------
# Resolver mapping (Req 2.1, 2.2, 2.3)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_inbound_camera_resolves_inbound() -> None:
    r = CameraDirectionResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    assert r.resolve_for_camera(1) is DirectionOutcome.INBOUND


@pytest.mark.unit
def test_outbound_camera_resolves_outbound() -> None:
    r = CameraDirectionResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    assert r.resolve_for_camera(2) is DirectionOutcome.OUTBOUND


@pytest.mark.unit
def test_unmapped_camera_resolves_undetermined() -> None:
    r = CameraDirectionResolver({1: DirectionOutcome.INBOUND, 2: DirectionOutcome.OUTBOUND})
    assert r.resolve_for_camera(9) is DirectionOutcome.UNDETERMINED


@pytest.mark.unit
def test_missing_camera_id_resolves_undetermined() -> None:
    r = CameraDirectionResolver({1: DirectionOutcome.INBOUND})
    assert r.resolve_for_camera(None) is DirectionOutcome.UNDETERMINED


@pytest.mark.unit
def test_map_accepts_string_direction_values() -> None:
    r = CameraDirectionResolver({1: "INBOUND", 2: "outbound"})
    assert r.resolve_for_camera(1) is DirectionOutcome.INBOUND
    assert r.resolve_for_camera(2) is DirectionOutcome.OUTBOUND


# ---------------------------------------------------------------------------
# Factory: the map comes from cameras.<side>.id
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_factory_maps_both_configured_cameras() -> None:
    resolver = create_direction_resolver(ConfigProvider(env={}))
    assert resolver.camera_direction_map == {
        1: DirectionOutcome.INBOUND,
        2: DirectionOutcome.OUTBOUND,
    }


@pytest.mark.unit
def test_factory_maps_custom_ids() -> None:
    env = {"ANPR_CAMERAS__INBOUND__ID": "7", "ANPR_CAMERAS__OUTBOUND__ID": "8"}
    resolver = create_direction_resolver(ConfigProvider(env=env))
    assert resolver.camera_direction_map == {
        7: DirectionOutcome.INBOUND,
        8: DirectionOutcome.OUTBOUND,
    }


# ---------------------------------------------------------------------------
# cameras config validation (Req 1.5, 7.5)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_default_cameras_config_validates_cleanly() -> None:
    errors = ConfigProvider(env={}).validate()
    assert errors == [], f"unexpected errors: {errors}"


@pytest.mark.unit
def test_ip_cameras_without_stream_url_are_reported_per_key() -> None:
    env = {
        "ANPR_CAMERAS__INBOUND__TYPE": "ip",
        "ANPR_CAMERAS__OUTBOUND__TYPE": "ip",
    }
    keys = {e.key for e in ConfigProvider(env=env).validate()}
    # The default stream_url is an empty string, which is not a usable address.
    assert "cameras.inbound.stream_url" in keys
    assert "cameras.outbound.stream_url" in keys


@pytest.mark.unit
def test_ip_cameras_with_stream_url_validate() -> None:
    env = {
        "ANPR_CAMERAS__INBOUND__TYPE": "ip",
        "ANPR_CAMERAS__INBOUND__STREAM_URL": "rtsp://in/stream",
        "ANPR_CAMERAS__OUTBOUND__TYPE": "ip",
        "ANPR_CAMERAS__OUTBOUND__STREAM_URL": "rtsp://out/stream",
    }
    assert ConfigProvider(env=env).validate() == []


@pytest.mark.unit
def test_invalid_camera_type_is_reported() -> None:
    errors = ConfigProvider(env={"ANPR_CAMERAS__INBOUND__TYPE": "usb"}).validate()
    assert any(e.key == "cameras.inbound.type" for e in errors)


@pytest.mark.unit
def test_non_integer_camera_id_is_reported() -> None:
    errors = ConfigProvider(env={"ANPR_CAMERAS__OUTBOUND__ID": "abc"}).validate()
    assert any(e.key == "cameras.outbound.id" for e in errors)


@pytest.mark.unit
def test_unknown_camera_name_is_reported() -> None:
    errors = ConfigProvider(env={"ANPR_CAMERAS__SIDEWAYS__ID": "3"}).validate()
    assert any(e.key == "cameras.sideways" for e in errors)


@pytest.mark.unit
def test_no_cameras_is_reported(tmp_path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("environment:\n  label: local-test\n", encoding="utf-8")
    errors = ConfigProvider(config_path=cfg, env={}).validate()
    assert any(e.key == "cameras" for e in errors)
