"""YOLO26 vehicle-detection wrapper for the ANPR Autogate System.

:class:`YoloVehicleDetector` wraps the pre-trained Ultralytics YOLO26 model
loaded from the ``best.pt`` weights and structurally satisfies the
``VehicleDetector`` Protocol defined in ``anpr.core.interfaces``. It is the
single I/O adapter that turns raw frames into the pure
:class:`~anpr.core.models.Detection` value objects the rest of the pipeline
consumes, so no downstream component depends on Ultralytics directly
(Requirement 14.4).

Loading (Requirement 2.1, 2.5, 2.6):

* :meth:`load` loads the vehicle-detection weights from the configured
  ``model.weights_path`` (defaulting to the ``best.pt`` weights present in the
  workspace) at startup, before any frame is processed (Req 2.1).
* If the specified weights file cannot be loaded, the failure is logged with
  the offending weights path and re-raised as a :class:`WeightsLoadError` so the
  composition root halts the Detection_Pipeline startup (Req 2.5).
* If the loaded-weights indicator reports weights as loaded while no valid
  weights file is active, the active weights path is reset to the default
  ``best.pt`` weights path before loading (Req 2.6).

Detection (Requirement 2.2, 2.3, 2.4):

* :meth:`detect` runs Ultralytics **track** mode (``persist=True``) so each
  emitted :class:`Detection` carries a persistent ``track_id`` across frames,
  which the single-camera trajectory ``Direction_Resolver`` relies on.
* Every emitted bounding box is clamped to lie within the frame's pixel bounds
  ``[0, width] x [0, height]`` and every confidence is clamped to the inclusive
  range ``0.0`` to ``1.0`` (Req 2.2).
* Detections with confidence strictly below the configurable detection
  threshold (inclusive ``[0.0, 1.0]``, default ``0.5``) are excluded; those at
  or above it are retained (Req 2.4).
* A frame with no detectable vehicle yields zero detections and nothing flows
  downstream for that frame (Req 2.3).

The Ultralytics model factory is injectable so the load/indicator branches and
the detection-parsing logic can be exercised deterministically in tests without
the heavy runtime dependency. See
.kiro/specs/anpr-autogate-system/requirements.md (Requirement 2) and design.md
(YOLO26_Detector section) for the authoritative acceptance criteria.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from anpr.core.models import BoundingBox, Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (design Configuration schema: model.weights_path, detection_threshold)
# ---------------------------------------------------------------------------

# Default weights file present in the workspace (Req 2.1, 2.6).
DEFAULT_WEIGHTS_PATH = "best.pt"

# Detection threshold is configurable in the inclusive range 0.0..1.0 and
# defaults to 0.5 (Req 2.4).
DEFAULT_DETECTION_THRESHOLD = 0.5
MIN_DETECTION_THRESHOLD = 0.0
MAX_DETECTION_THRESHOLD = 1.0


class WeightsLoadError(RuntimeError):
    """Raised when the configured vehicle-detection weights cannot be loaded.

    Carries the offending weights path so the composition root can log it and
    halt the Detection_Pipeline startup (Req 2.5).
    """

    def __init__(self, weights_path: str, cause: Optional[BaseException] = None) -> None:
        self.weights_path = weights_path
        message = f"Failed to load vehicle-detection weights from '{weights_path}'."
        if cause is not None:
            message = f"{message} {cause}"
        super().__init__(message)


def clamp_detection_threshold(threshold: Optional[float]) -> float:
    """Clamp a configured detection threshold to the valid 0.0-1.0 range (Req 2.4).

    ``None`` or a non-numeric value falls back to the 0.5 default; values below
    0.0 or above 1.0 are clamped to the nearest bound so a misconfiguration can
    never produce an out-of-range threshold.
    """
    if threshold is None or isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return DEFAULT_DETECTION_THRESHOLD
    return float(min(MAX_DETECTION_THRESHOLD, max(MIN_DETECTION_THRESHOLD, threshold)))


def _default_model_factory(weights_path: str) -> Any:
    """Construct an Ultralytics ``YOLO`` model from a weights path.

    Imported lazily so the package stays importable before the heavy
    Ultralytics dependency is installed (matching the guarded-import pattern
    used elsewhere in the codebase).
    """
    from ultralytics import YOLO  # local import: heavy optional dependency

    return YOLO(weights_path)


class YoloVehicleDetector:
    """Ultralytics YOLO26 detector wrapper (structurally a ``VehicleDetector``).

    Parameters
    ----------
    weights_path:
        Filesystem path to the vehicle-detection weights (from config
        ``model.weights_path``). Defaults to the workspace ``best.pt`` weights.
    detection_threshold:
        Configured detection threshold; clamped to the inclusive 0.0-1.0 range
        (default 0.5).
    model_factory:
        Callable that builds the underlying detection model from a weights
        path. Injectable so the load and detection branches can be unit tested
        without the Ultralytics runtime dependency.
    """

    def __init__(
        self,
        weights_path: Optional[str] = DEFAULT_WEIGHTS_PATH,
        detection_threshold: Optional[float] = DEFAULT_DETECTION_THRESHOLD,
        *,
        model_factory: Callable[[str], Any] = _default_model_factory,
    ) -> None:
        self._weights_path = weights_path if weights_path else DEFAULT_WEIGHTS_PATH
        self._detection_threshold = clamp_detection_threshold(detection_threshold)
        self._model_factory = model_factory
        self._model: Any = None
        # The "loaded-weights indicator": True once weights have been loaded.
        self._loaded = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def weights_path(self) -> str:
        """The active weights file path (may be reset to the default, Req 2.6)."""
        return self._weights_path

    @property
    def detection_threshold(self) -> float:
        """The effective (clamped) detection threshold in the 0.0-1.0 range."""
        return self._detection_threshold

    @property
    def is_loaded(self) -> bool:
        """The loaded-weights indicator (True once weights are loaded)."""
        return self._loaded

    # ------------------------------------------------------------------
    # Loading (Req 2.1, 2.5, 2.6)
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load vehicle-detection weights, halting startup on failure.

        Reconciles the loaded-weights indicator with the active weights file
        (Req 2.6), then loads the weights from the active path (Req 2.1). On any
        failure the offending path is logged and a :class:`WeightsLoadError` is
        raised so the composition root halts the pipeline startup (Req 2.5).
        """
        # Req 2.6: if the loaded indicator is set while no valid weights file is
        # active, reset the active weights path to the default best.pt path.
        self._reconcile_active_weights()

        try:
            self._model = self._model_factory(self._weights_path)
        except Exception as exc:  # noqa: BLE001 - any load fault halts startup
            logger.error(
                "Failed to load vehicle-detection weights from '%s': %s",
                self._weights_path,
                exc,
            )
            self._model = None
            self._loaded = False
            raise WeightsLoadError(self._weights_path, exc) from exc

        self._loaded = True
        logger.info("Loaded vehicle-detection weights from '%s'.", self._weights_path)

    def _reconcile_active_weights(self) -> None:
        """Reset the active weights path to the default when inconsistent (Req 2.6).

        When the loaded-weights indicator reports weights as loaded but no valid
        weights file is active, the active weights path is reset to the default
        ``best.pt`` path so a later load uses a known-good default.
        """
        if self._loaded and not self._is_valid_weights_file(self._weights_path):
            logger.warning(
                "Loaded-weights indicator is set while no valid weights file is "
                "active ('%s'); resetting active weights path to default '%s'.",
                self._weights_path,
                DEFAULT_WEIGHTS_PATH,
            )
            self._weights_path = DEFAULT_WEIGHTS_PATH

    @staticmethod
    def _is_valid_weights_file(weights_path: Optional[str]) -> bool:
        """Return True when ``weights_path`` names an existing file."""
        if not weights_path:
            return False
        try:
            return Path(weights_path).is_file()
        except OSError:  # pragma: no cover - defensive against odd paths
            return False

    # ------------------------------------------------------------------
    # Detection (Req 2.2, 2.3, 2.4)
    # ------------------------------------------------------------------
    def detect(self, frame: Any) -> list[Detection]:
        """Detect vehicles in ``frame`` and return the retained detections.

        Runs Ultralytics track mode, clamps every box to the frame bounds and
        every confidence to ``[0.0, 1.0]`` (Req 2.2), and retains only
        detections whose confidence is ``>=`` the detection threshold, excluding
        those strictly below it (Req 2.4). An empty frame (``None`` or zero
        pixels) yields zero detections (Req 2.3).
        """
        if self._model is None:
            raise RuntimeError(
                "Detector weights are not loaded; call load() before detect()."
            )

        dims = _frame_dimensions(frame)
        if dims is None:
            # Empty / absent frame -> nothing detectable, nothing downstream.
            return []
        width, height = dims

        results = self._model.track(frame, persist=True, verbose=False)

        detections: list[Detection] = []
        for box, confidence, track_id in _iter_raw_boxes(results):
            if confidence < self._detection_threshold:
                # Req 2.4: strictly below the threshold is excluded.
                continue
            detections.append(
                Detection(
                    box=_clamp_box(box, width, height),
                    confidence=_clamp_confidence(confidence),
                    track_id=track_id,
                )
            )
        return detections


# ---------------------------------------------------------------------------
# Parsing / clamping helpers
# ---------------------------------------------------------------------------


def _frame_dimensions(frame: Any) -> Optional[tuple[int, int]]:
    """Return ``(width, height)`` for a frame, or ``None`` when it is empty.

    A frame is an ``(H, W, C)`` NumPy array; an absent frame or one with zero
    width/height is treated as empty (Req 2.3).
    """
    if frame is None:
        return None
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _iter_raw_boxes(results: Any):
    """Yield ``(xyxy, confidence, track_id)`` tuples from Ultralytics results.

    Tolerates the absence of boxes (no detections) and the absence of tracker
    ids (``track_id`` is ``None`` when the tracker did not assign one).
    """
    if results is None:
        return
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = _to_list(getattr(boxes, "xyxy", None))
        confs = _to_list(getattr(boxes, "conf", None))
        ids = _to_list(getattr(boxes, "id", None))
        for index, coords in enumerate(xyxy):
            coords_list = _to_list(coords)
            if coords_list is None or len(coords_list) < 4:
                continue
            confidence = float(confs[index]) if confs is not None and index < len(confs) else 0.0
            track_id = (
                int(ids[index]) if ids is not None and index < len(ids) and ids[index] is not None
                else None
            )
            yield coords_list, confidence, track_id


def _to_list(value: Any) -> Optional[list]:
    """Convert a tensor/ndarray/sequence to a plain Python list.

    Handles Ultralytics tensors (which expose ``tolist``) and plain sequences,
    returning ``None`` when the value is absent.
    """
    if value is None:
        return None
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_list()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clamp_box(coords: list, width: int, height: int) -> BoundingBox:
    """Clamp raw ``(x1, y1, x2, y2)`` coordinates to the frame bounds (Req 2.2).

    Coordinates are rounded to integer pixels and constrained to
    ``[0, width] x [0, height]`` so every emitted box lies within the frame.
    """
    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    return BoundingBox(
        x1=_clamp_int(x1, 0, width),
        y1=_clamp_int(y1, 0, height),
        x2=_clamp_int(x2, 0, width),
        y2=_clamp_int(y2, 0, height),
    )


def _clamp_int(value: Any, low: int, high: int) -> int:
    """Round ``value`` to an int and clamp it to the inclusive ``[low, high]``."""
    return int(min(high, max(low, round(float(value)))))


def _clamp_confidence(confidence: float) -> float:
    """Clamp a confidence value to the inclusive range 0.0..1.0 (Req 2.2)."""
    return float(min(MAX_DETECTION_THRESHOLD, max(MIN_DETECTION_THRESHOLD, confidence)))
