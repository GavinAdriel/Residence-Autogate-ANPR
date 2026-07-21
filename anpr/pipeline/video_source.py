"""OpenCV-backed video sources for the ANPR Autogate System.

:class:`WebcamVideoSource` and :class:`IpCameraVideoSource` are the two concrete
frame sources for this phase. Both wrap OpenCV ``cv2.VideoCapture`` and
structurally satisfy the ``VideoSource`` Protocol defined in
``anpr.core.interfaces`` (``open`` / ``read`` / ``close`` / ``descriptor``), so
the ``DetectionPipeline`` depends only on the abstraction and never on OpenCV or
on a concrete source directly (Requirement 14.4). The composition root selects
between them purely from configuration (Requirement 1.2, 1.3).

* :class:`WebcamVideoSource` ingests frames from a USB webcam **device index**
  (``camera.device_index``) (Req 1.2). Its descriptor reads ``"webcam:<index>"``.
* :class:`IpCameraVideoSource` ingests frames from a network **stream URL**
  (``camera.stream_url``, RTSP/HTTP) (Req 1.3). Its descriptor reads
  ``"ip:<url>"``.

The sources are kept deliberately thin: the retry/reconnect loop and the
inactivity watchdog live in the ``DetectionPipeline`` (Req 1.4-1.6), not here.
Each source only knows how to:

* :meth:`open` -- construct the underlying capture and raise
  :class:`SourceUnavailable` when it fails to open (``cap.isOpened()`` is
  ``False``), so the pipeline's retry loop has a single, explicit failure signal.
* :meth:`read` -- return the current frame when ``cap.read()`` succeeds, or
  ``None`` when no frame is available right now (the pipeline decides whether a
  ``None`` run means an interruption/inactivity timeout).
* :meth:`close` -- release the underlying capture.

The ``cv2.VideoCapture`` factory is injectable so the open/read/close branches
can be exercised deterministically in tests without a real camera or the heavy
OpenCV runtime dependency, matching the guarded-import + injectable-factory
pattern used by the detector and OCR wrappers. See
.kiro/specs/anpr-autogate-system/requirements.md (Requirement 1) and design.md
(VideoSource section) for the authoritative acceptance criteria.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# A capture factory builds an OpenCV-like capture object from a source argument
# (a webcam device index or a stream URL). Injectable for tests.
CaptureFactory = Callable[[Any], Any]


class SourceUnavailable(RuntimeError):
    """Raised when a configured video source cannot be opened.

    Carries the source :attr:`descriptor` so the ``DetectionPipeline`` can log
    the offending source and apply its retry/reconnect behavior (Req 1.4-1.6).
    """

    def __init__(self, descriptor: str, cause: Optional[BaseException] = None) -> None:
        self.descriptor = descriptor
        message = f"Video source '{descriptor}' could not be opened."
        if cause is not None:
            message = f"{message} {cause}"
        super().__init__(message)


def _default_capture_factory(source: Any) -> Any:
    """Construct an OpenCV ``VideoCapture`` for a device index or stream URL.

    Imported lazily so the package stays importable before the heavy OpenCV
    dependency is installed (matching the guarded-import pattern used by the
    detector and OCR wrappers).
    """
    import cv2  # local import: heavy optional dependency

    return cv2.VideoCapture(source)


class _OpenCvVideoSource:
    """Shared OpenCV ``VideoCapture`` plumbing for the concrete video sources.

    Subclasses supply the capture argument (device index or stream URL) and a
    human-readable :attr:`descriptor`; this base owns the open/read/close
    lifecycle around the underlying capture object.

    Parameters
    ----------
    capture_arg:
        The value passed to the capture factory -- a webcam device index (int)
        or an IP-camera stream URL (str).
    capture_factory:
        Callable that builds the underlying capture from ``capture_arg``.
        Injectable so the lifecycle branches can be unit tested without a real
        camera or the OpenCV runtime dependency.
    """

    def __init__(
        self,
        capture_arg: Any,
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
    ) -> None:
        self._capture_arg = capture_arg
        self._capture_factory = capture_factory
        self._capture: Any = None

    # ------------------------------------------------------------------
    # Descriptor (overridden by subclasses)
    # ------------------------------------------------------------------
    @property
    def descriptor(self) -> str:  # pragma: no cover - overridden by subclasses
        """Human-readable source identifier used in logs."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle (Req 1.2, 1.3; failure signalling for Req 1.4-1.6)
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Open the underlying capture, raising when the source is unavailable.

        Builds the capture via the injected factory and verifies it opened
        (``cap.isOpened()``). On any failure -- a factory error or a capture
        that reports it did not open -- the capture is released and a
        :class:`SourceUnavailable` is raised carrying this source's descriptor
        so the pipeline can log it and retry (Req 1.4).
        """
        try:
            capture = self._capture_factory(self._capture_arg)
        except Exception as exc:  # noqa: BLE001 - any factory fault is unavailability
            logger.error(
                "Failed to construct video capture for source '%s': %s",
                self.descriptor,
                exc,
            )
            raise SourceUnavailable(self.descriptor, exc) from exc

        if capture is None or not _is_opened(capture):
            logger.error("Video source '%s' failed to open.", self.descriptor)
            _release(capture)
            raise SourceUnavailable(self.descriptor)

        self._capture = capture
        logger.info("Opened video source '%s'.", self.descriptor)

    def read(self) -> Optional[Any]:
        """Return the current frame, or ``None`` when none is available now.

        Returns the frame when ``cap.read()`` reports success and yields a
        non-``None`` frame; otherwise returns ``None`` (no frame available right
        now). Returns ``None`` when the source has not been opened. The pipeline
        interprets a run of ``None`` reads as an interruption or inactivity
        timeout (Req 1.5, 1.6).
        """
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        """Release the underlying capture, if any."""
        if self._capture is None:
            return
        _release(self._capture)
        self._capture = None
        logger.info("Closed video source '%s'.", self.descriptor)


class WebcamVideoSource(_OpenCvVideoSource):
    """USB-webcam frame source addressed by device index (Req 1.2).

    Parameters
    ----------
    device_index:
        The webcam device index supplied by the Configuration_Provider
        (``camera.device_index``).
    capture_factory:
        Injectable capture factory (see :class:`_OpenCvVideoSource`).
    """

    def __init__(
        self,
        device_index: int,
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
    ) -> None:
        super().__init__(int(device_index), capture_factory=capture_factory)
        self._device_index = int(device_index)

    @property
    def device_index(self) -> int:
        """The configured webcam device index."""
        return self._device_index

    @property
    def descriptor(self) -> str:
        """Human-readable id, e.g. ``"webcam:0"``."""
        return f"webcam:{self._device_index}"


class IpCameraVideoSource(_OpenCvVideoSource):
    """IP-camera frame source addressed by network stream URL (Req 1.3).

    Parameters
    ----------
    stream_url:
        The RTSP/HTTP network stream address supplied by the
        Configuration_Provider (``camera.stream_url``).
    capture_factory:
        Injectable capture factory (see :class:`_OpenCvVideoSource`).
    """

    def __init__(
        self,
        stream_url: str,
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
    ) -> None:
        super().__init__(stream_url, capture_factory=capture_factory)
        self._stream_url = stream_url

    @property
    def stream_url(self) -> str:
        """The configured IP-camera stream URL."""
        return self._stream_url

    @property
    def descriptor(self) -> str:
        """Human-readable id, e.g. ``"ip:rtsp://camera/stream"``."""
        return f"ip:{self._stream_url}"


# ---------------------------------------------------------------------------
# Capture helpers (tolerant of the OpenCV VideoCapture surface)
# ---------------------------------------------------------------------------


def _is_opened(capture: Any) -> bool:
    """Return whether a capture reports it is open (``cap.isOpened()``).

    Tolerates capture objects without ``isOpened`` (treated as not open) so a
    malformed factory result surfaces as unavailability rather than an
    ``AttributeError``.
    """
    is_opened = getattr(capture, "isOpened", None)
    if not callable(is_opened):
        return False
    try:
        return bool(is_opened())
    except Exception:  # noqa: BLE001 - a faulty probe means "not open"
        return False


def _release(capture: Any) -> None:
    """Release a capture object, ignoring absence of a ``release`` method."""
    if capture is None:
        return
    release = getattr(capture, "release", None)
    if callable(release):
        try:
            release()
        except Exception:  # noqa: BLE001 - releasing must never raise
            logger.debug("Ignoring error while releasing a video capture.", exc_info=True)
