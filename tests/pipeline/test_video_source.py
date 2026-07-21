"""Unit tests for the OpenCV-backed video sources (Task 12.1).

These example-based tests exercise the ``open`` / ``read`` / ``close`` /
``descriptor`` surface of :class:`WebcamVideoSource` and
:class:`IpCameraVideoSource` (Requirements 1.2, 1.3) using an injected fake
capture, so no real camera or OpenCV runtime is needed.
"""

from __future__ import annotations

import pytest

from anpr.pipeline.video_source import (
    IpCameraVideoSource,
    SourceUnavailable,
    WebcamVideoSource,
)


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture`` for deterministic tests."""

    def __init__(self, *, opened: bool = True, frames=None) -> None:
        self._opened = opened
        # Each frame is yielded as a successful (True, frame) read; when the
        # list is exhausted, reads report failure as (False, None).
        self._frames = list(frames) if frames is not None else []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API name
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


@pytest.mark.unit
def test_webcam_descriptor_uses_device_index():
    source = WebcamVideoSource(2, capture_factory=lambda arg: FakeCapture())
    assert source.descriptor == "webcam:2"
    assert source.device_index == 2


@pytest.mark.unit
def test_ip_camera_descriptor_uses_stream_url():
    url = "rtsp://cam.local/stream"
    source = IpCameraVideoSource(url, capture_factory=lambda arg: FakeCapture())
    assert source.descriptor == f"ip:{url}"
    assert source.stream_url == url


@pytest.mark.unit
def test_webcam_passes_device_index_to_capture_factory():
    seen = {}

    def factory(arg):
        seen["arg"] = arg
        return FakeCapture()

    WebcamVideoSource(3, capture_factory=factory).open()
    assert seen["arg"] == 3


@pytest.mark.unit
def test_ip_camera_passes_stream_url_to_capture_factory():
    seen = {}

    def factory(arg):
        seen["arg"] = arg
        return FakeCapture()

    IpCameraVideoSource("http://cam/stream", capture_factory=factory).open()
    assert seen["arg"] == "http://cam/stream"


@pytest.mark.unit
def test_open_raises_source_unavailable_when_capture_not_opened():
    source = WebcamVideoSource(0, capture_factory=lambda arg: FakeCapture(opened=False))
    with pytest.raises(SourceUnavailable) as excinfo:
        source.open()
    assert excinfo.value.descriptor == "webcam:0"


@pytest.mark.unit
def test_open_raises_source_unavailable_when_factory_errors():
    def factory(arg):
        raise OSError("device busy")

    source = IpCameraVideoSource("rtsp://x", capture_factory=factory)
    with pytest.raises(SourceUnavailable):
        source.open()


@pytest.mark.unit
def test_open_releases_capture_that_failed_to_open():
    capture = FakeCapture(opened=False)
    source = WebcamVideoSource(0, capture_factory=lambda arg: capture)
    with pytest.raises(SourceUnavailable):
        source.open()
    assert capture.released is True


@pytest.mark.unit
def test_read_returns_frame_on_successful_capture_read():
    frame = object()
    source = WebcamVideoSource(0, capture_factory=lambda arg: FakeCapture(frames=[frame]))
    source.open()
    assert source.read() is frame


@pytest.mark.unit
def test_read_returns_none_when_no_frame_available():
    source = WebcamVideoSource(0, capture_factory=lambda arg: FakeCapture(frames=[]))
    source.open()
    assert source.read() is None


@pytest.mark.unit
def test_read_returns_none_before_open():
    source = WebcamVideoSource(0, capture_factory=lambda arg: FakeCapture())
    assert source.read() is None


@pytest.mark.unit
def test_close_releases_capture_and_is_idempotent():
    capture = FakeCapture()
    source = WebcamVideoSource(0, capture_factory=lambda arg: capture)
    source.open()
    source.close()
    assert capture.released is True
    # Closing again must not raise.
    source.close()
