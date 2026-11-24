"""Headless wiring test for camera composition (Task 6.2).

Exercises :func:`anpr.main.build_camera_pipelines` with fakes and no live MySQL,
PyQt5, camera, or weights. One pipeline is built per configured ``cameras.*``
entry and the key name is the direction:

* only ``cameras.inbound`` (or only ``cameras.outbound``) -> exactly one primary
  pipeline with a fixed direction;
* both -> two pipelines (inbound primary + outbound), sharing the
  detector/OCR/normalizer/access-controller, each with its OWN TrackAggregator
  (Req 6.1), both carrying the shared CameraDirectionResolver and the right
  camera id (Req 2.4);
* the composition imports no PyQt5 (Req 7.3).
"""

from __future__ import annotations

import sys

import pytest

import anpr.main as main
from anpr.config.provider import ConfigProvider
from anpr.core.access_controller import AccessController
from anpr.core.models import DirectionOutcome
from anpr.direction.resolver import CameraDirectionResolver
from anpr.pipeline.track_aggregator import TrackAggregator

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Lightweight fakes for the shared collaborators (no heavy deps).
# ---------------------------------------------------------------------------
class _FakeDetector:
    def load(self) -> None:  # pragma: no cover - not called here
        pass

    def detect(self, frame):  # pragma: no cover - not called here
        return []


class _FakeOcr:
    def read_plate(self, crop):  # pragma: no cover - not called here
        return None


class _FakeResidentRepo:
    def find_by_plate(self, plate):  # pragma: no cover
        return None


class _FakeEventLog:
    def append(self, record):  # pragma: no cover
        return 1

    def find_open_entry(self, plate):  # pragma: no cover
        return None

    def close_entry(self, entry_log_id, closed_by_log_id):  # pragma: no cover
        return 0


class _FakeGate:
    def open_gate(self, event_id):  # pragma: no cover
        from anpr.core.models import GateResult

        return GateResult(success=True)


def _collaborators():
    from anpr.core.normalizer import PlateNormalizer

    detector = _FakeDetector()
    ocr = _FakeOcr()
    normalizer = PlateNormalizer()
    access = AccessController(_FakeResidentRepo(), _FakeEventLog(), _FakeGate())
    return detector, ocr, normalizer, access


def _pyqt_modules() -> set[str]:
    return {m for m in sys.modules if m == "PyQt5" or m.startswith("PyQt5.")}


# ---------------------------------------------------------------------------
# Single-camera (one ``cameras.*`` entry, fixed direction)
# ---------------------------------------------------------------------------
def _single_config(tmp_path, side: str, cam_id: int) -> ConfigProvider:
    cfg = tmp_path / "single.yaml"
    cfg.write_text(
        "cameras:\n"
        f"  {side}:\n"
        f"    id: {cam_id}\n"
        "    type: ip\n"
        "    stream_url: rtsp://only/stream\n",
        encoding="utf-8",
    )
    return ConfigProvider(config_path=cfg, env={})


@pytest.mark.parametrize(
    ("side", "cam_id", "expected"),
    [
        ("inbound", 1, DirectionOutcome.INBOUND),
        ("outbound", 2, DirectionOutcome.OUTBOUND),
    ],
)
def test_single_camera_builds_one_primary_pipeline_with_fixed_direction(
    tmp_path, side, cam_id, expected
) -> None:
    config = _single_config(tmp_path, side, cam_id)
    detector, ocr, normalizer, access = _collaborators()

    pipelines = main.build_camera_pipelines(config, detector, ocr, normalizer, access)

    assert len(pipelines) == 1
    cp = pipelines[0]
    assert cp.is_primary is True
    assert cp.label == side
    assert cp.camera_id == cam_id
    assert cp.pipeline._resolve_direction() is expected


def test_no_cameras_configured_halts_startup(tmp_path) -> None:
    cfg = tmp_path / "none.yaml"
    cfg.write_text("environment:\n  label: local-test\n", encoding="utf-8")
    detector, ocr, normalizer, access = _collaborators()

    with pytest.raises(main.StartupError):
        main.build_camera_pipelines(
            ConfigProvider(config_path=cfg, env={}), detector, ocr, normalizer, access
        )


# ---------------------------------------------------------------------------
# Two cameras (Req 2.4, 6.1, 7.3)
# ---------------------------------------------------------------------------
def _dual_config() -> ConfigProvider:
    return ConfigProvider(
        env={
            "ANPR_CAMERAS__INBOUND__TYPE": "ip",
            "ANPR_CAMERAS__INBOUND__STREAM_URL": "rtsp://in/stream",
            "ANPR_CAMERAS__OUTBOUND__TYPE": "ip",
            "ANPR_CAMERAS__OUTBOUND__STREAM_URL": "rtsp://out/stream",
        }
    )


def test_dual_camera_mode_builds_two_pipelines_with_shared_collaborators() -> None:
    config = _dual_config()
    detector, ocr, normalizer, access = _collaborators()

    pipelines = main.build_camera_pipelines(config, detector, ocr, normalizer, access)

    assert len(pipelines) == 2
    labels = {cp.label for cp in pipelines}
    assert labels == {"inbound", "outbound"}

    by_label = {cp.label: cp for cp in pipelines}
    assert by_label["inbound"].is_primary is True
    assert by_label["outbound"].is_primary is False
    assert by_label["inbound"].camera_id == 1
    assert by_label["outbound"].camera_id == 2

    # Shared collaborators: both pipelines hold the very same detector/ocr/etc.
    in_p, out_p = by_label["inbound"].pipeline, by_label["outbound"].pipeline
    assert in_p._detector is out_p._detector is detector
    assert in_p._ocr is out_p._ocr is ocr
    assert in_p._access is out_p._access is access

    # Distinct aggregators so identical tracker ids never collide (Req 6.1).
    assert in_p.track_aggregator is not out_p.track_aggregator
    assert isinstance(in_p.track_aggregator, TrackAggregator)

    # Both share one CameraDirectionResolver and one RecentEntryRegistry.
    assert isinstance(in_p._direction_resolver, CameraDirectionResolver)
    assert in_p._direction_resolver is out_p._direction_resolver
    assert in_p._recent_entries is out_p._recent_entries


def test_dual_camera_direction_resolution_per_pipeline() -> None:
    config = _dual_config()
    detector, ocr, normalizer, access = _collaborators()

    pipelines = main.build_camera_pipelines(config, detector, ocr, normalizer, access)
    by_label = {cp.label: cp for cp in pipelines}

    assert (
        by_label["inbound"].pipeline._resolve_direction() is DirectionOutcome.INBOUND
    )
    assert (
        by_label["outbound"].pipeline._resolve_direction() is DirectionOutcome.OUTBOUND
    )


def test_dual_camera_composition_imports_no_pyqt5() -> None:
    before = _pyqt_modules()
    config = _dual_config()
    detector, ocr, normalizer, access = _collaborators()

    main.build_camera_pipelines(config, detector, ocr, normalizer, access)

    assert _pyqt_modules() - before == set()
