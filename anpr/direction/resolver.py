"""Direction resolution logic for the ANPR Autogate System.

The Direction_Resolver decides whether a tracked vehicle at the single shared
gate is moving *inbound* or whether its direction is *undetermined*. The
concrete resolver is chosen purely by configuration (``direction.mode``) in the
composition root, so this module depends only on the shared value objects and
never on a concrete peer implementation (Requirement 14.4).

Two resolvers are defined, mirroring the two configurable modes:

* :class:`SingleCameraTrajectoryResolver` (``direction.mode =
  single_camera_trajectory``) is the mode exercised end-to-end in this phase.
  It inspects the centroid trajectory of a :class:`TrackHistory` over a
  configurable minimum number of consecutive frames and resolves the event as
  ``INBOUND`` when the motion matches the configured inbound reference
  orientation with confidence strictly above the direction threshold, or
  ``UNDETERMINED`` otherwise. It never emits ``OUTBOUND`` in this phase
  (Requirements 7.2, 7.3, 7.4, 7.5).
* :class:`DualCameraResolver` (``direction.mode = dual_camera``) is *deferred*:
  its resolution logic is not implemented in this phase, but the class and the
  ``DirectionResolver`` interface surface are retained so the eventual field
  deployment can drop it in without a configuration or interface change
  (Requirements 7.1, 7.6, 7.7).

The trajectory math is a pure, deterministic function of its inputs (no I/O),
so it is directly property-testable. See
.kiro/specs/anpr-autogate-system/requirements.md (Requirement 7) and design.md
(Direction_Resolver section) for the authoritative acceptance criteria.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

from anpr.core.interfaces import ConfigProvider, DirectionResolver
from anpr.core.models import DirectionOutcome, Point, TrackHistory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration keys and defaults (design Configuration / Direction section)
# ---------------------------------------------------------------------------

MODE_SINGLE_CAMERA_TRAJECTORY = "single_camera_trajectory"
MODE_DUAL_CAMERA = "dual_camera"

# Default direction mode when configuration is silent (Req 7.1).
DEFAULT_MODE = MODE_SINGLE_CAMERA_TRAJECTORY

# Minimum consecutive frames of trajectory needed before a direction is
# resolved; fewer frames than this yields UNDETERMINED (Req 7.3). Must be at
# least 2 for a motion vector to exist.
DEFAULT_MIN_CONSECUTIVE_FRAMES = 5

# Inbound reference orientation the motion vector is compared against (Req 7.3).
DEFAULT_INBOUND_REFERENCE_ORIENTATION: tuple[float, float] = (1.0, 0.0)

# Direction confidence threshold: a resolution confidence at or below this
# value yields UNDETERMINED, so INBOUND requires confidence strictly ABOVE it
# (Req 7.5).
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Pure trajectory helpers
# ---------------------------------------------------------------------------


def _net_motion_vector(centroids: Sequence[Point]) -> tuple[float, float]:
    """Return the net displacement across ``centroids`` as ``(dx, dy)``.

    The motion vector is the straight-line displacement from the first to the
    last centroid in the supplied window. This aggregates per-frame jitter into
    the overall direction of travel, which is what the inbound-orientation
    comparison cares about.
    """
    first_x, first_y = centroids[0]
    last_x, last_y = centroids[-1]
    return (float(last_x - first_x), float(last_y - first_y))


def _magnitude(vector: tuple[float, float]) -> float:
    """Euclidean magnitude of a 2-D vector."""
    return math.hypot(vector[0], vector[1])


def _cosine_similarity(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Cosine similarity of two 2-D vectors in ``[-1.0, 1.0]``.

    Returns ``0.0`` when either vector has zero magnitude (no defined
    direction), so a stationary track or a degenerate reference orientation can
    never be mistaken for a directional match.
    """
    mag_a = _magnitude(a)
    mag_b = _magnitude(b)
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    dot = a[0] * b[0] + a[1] * b[1]
    # Clamp to guard against floating-point drift pushing the ratio slightly
    # outside the mathematically valid [-1, 1] range.
    return max(-1.0, min(1.0, dot / (mag_a * mag_b)))


def _clamp01(value: float) -> float:
    """Clamp a value to the inclusive ``[0.0, 1.0]`` range."""
    return max(0.0, min(1.0, value))


class SingleCameraTrajectoryResolver:
    """Single-camera trajectory direction resolver (``single_camera_trajectory``).

    Over the most recent ``min_consecutive_frames`` centroids of a
    :class:`TrackHistory`, the resolver computes the net motion vector and
    compares its direction to the configured inbound reference orientation via
    cosine similarity:

    * The motion is considered a **directional match** for the inbound
      reference when the cosine similarity is positive, i.e. the vehicle is
      travelling within 90 degrees of the inbound reference direction.
    * The **resolution confidence** combines how well the motion aligns with
      the reference (the alignment score, ``max(0, cosine_similarity)``) with
      the track's own reported confidence, as their product. Both must be high
      for a confident inbound resolution, and any motion pointing away from or
      perpendicular to the inbound reference collapses the alignment score to
      ``0`` so it can never resolve as inbound.

    The event resolves to :data:`DirectionOutcome.INBOUND` only when the motion
    is a directional match *and* the resolution confidence is strictly above
    the configured threshold (Req 7.3). Every other case
    -- too few frames, a stationary track, a non-matching direction, or a
    confidence at or below the threshold -- resolves to
    :data:`DirectionOutcome.UNDETERMINED` (Req 7.4, 7.5). ``OUTBOUND`` is never
    emitted in this phase (Req 7.2).

    This class structurally satisfies the ``DirectionResolver`` Protocol.
    """

    def __init__(
        self,
        min_consecutive_frames: int = DEFAULT_MIN_CONSECUTIVE_FRAMES,
        inbound_reference_orientation: tuple[float, float] = DEFAULT_INBOUND_REFERENCE_ORIENTATION,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        # A motion vector needs at least two centroids to exist; never allow a
        # window smaller than 2 regardless of misconfiguration.
        self._min_consecutive_frames = max(2, int(min_consecutive_frames))
        self._inbound_reference_orientation = (
            float(inbound_reference_orientation[0]),
            float(inbound_reference_orientation[1]),
        )
        self._confidence_threshold = float(confidence_threshold)

    @property
    def min_consecutive_frames(self) -> int:
        """Effective minimum consecutive frames required to resolve a direction."""
        return self._min_consecutive_frames

    @property
    def inbound_reference_orientation(self) -> tuple[float, float]:
        """The configured inbound reference orientation vector."""
        return self._inbound_reference_orientation

    @property
    def confidence_threshold(self) -> float:
        """The configured direction confidence threshold."""
        return self._confidence_threshold

    def resolve(self, track: TrackHistory) -> DirectionOutcome:
        """Resolve ``track`` into INBOUND or UNDETERMINED (Req 7.2-7.5)."""
        centroids = track.centroids

        # Req 7.3: need at least the configured minimum consecutive frames of
        # trajectory before any direction can be resolved.
        if centroids is None or len(centroids) < self._min_consecutive_frames:
            return DirectionOutcome.UNDETERMINED

        # Use the most recent window of centroids so the resolution reflects the
        # vehicle's current direction of travel.
        window = centroids[-self._min_consecutive_frames:]
        motion = _net_motion_vector(window)

        # A stationary track has no direction of travel -> UNDETERMINED.
        if _magnitude(motion) == 0.0:
            return DirectionOutcome.UNDETERMINED

        # Compare the motion direction to the configured inbound reference.
        similarity = _cosine_similarity(motion, self._inbound_reference_orientation)

        # Directional match: the motion points within 90 degrees of the inbound
        # reference (positive cosine similarity). Anything else (perpendicular
        # or opposing) is not an inbound trajectory (Req 7.4).
        is_directional_match = similarity > 0.0

        # Resolution confidence combines geometric alignment with the track's
        # own confidence; both must be strong for a confident inbound call.
        alignment_score = _clamp01(similarity)
        resolution_confidence = _clamp01(track.confidence) * alignment_score

        # Req 7.3: matching inbound orientation with confidence strictly above
        # the threshold resolves as INBOUND. Req 7.5: confidence at or below the
        # threshold resolves as UNDETERMINED.
        if is_directional_match and resolution_confidence > self._confidence_threshold:
            return DirectionOutcome.INBOUND

        # Req 7.4 / 7.5: non-matching direction or insufficient confidence is
        # logged as undetermined and surfaced to the guard; no direction rules.
        return DirectionOutcome.UNDETERMINED


class DualCameraResolver:
    """Dual-camera direction resolver (``dual_camera``).

    Direction is derived from the identity of the camera feed that produced the
    detection, not from motion trajectory: the Inbound_Camera resolves to
    ``INBOUND`` and the Outbound_Camera to ``OUTBOUND``
    (dual-camera-exit-correlation Req 2.1, 2.2). A camera id not present in the
    configured mapping resolves to ``UNDETERMINED`` so the event is surfaced to
    the guard rather than acted on automatically (Req 2.3).

    The mapping is keyed by ``Camera.Camera_ID`` (the integer id stamped onto
    each detection and written to ``ANPR_Log.Camera_ID``), with values being
    ``DirectionOutcome`` members. ``create_direction_resolver`` builds it from
    the two configured camera ids.

    This class provides :meth:`resolve_for_camera` -- the direct id->direction
    lookup the pipeline uses, since the pipeline knows its own camera id -- and
    also satisfies the trajectory-shaped ``DirectionResolver`` Protocol via
    :meth:`resolve`, which delegates using ``track.camera_id``.
    """

    def __init__(
        self, camera_direction_map: dict[int, DirectionOutcome] | None = None
    ) -> None:
        # Normalize keys to int (camera ids) and values to DirectionOutcome so
        # the map is tolerant of config-sourced strings.
        self._camera_direction_map: dict[int, DirectionOutcome] = {}
        for key, value in dict(camera_direction_map or {}).items():
            try:
                cam_id = int(key)
            except (TypeError, ValueError):
                continue
            self._camera_direction_map[cam_id] = _as_direction(value)

    @property
    def camera_direction_map(self) -> dict[int, DirectionOutcome]:
        """The camera-id -> direction mapping used for resolution."""
        return dict(self._camera_direction_map)

    def resolve_for_camera(self, camera_id: "int | None") -> DirectionOutcome:
        """Resolve direction from the producing camera's id (Req 2.1-2.3).

        An id mapped to inbound yields ``INBOUND``, to outbound ``OUTBOUND``; an
        unmapped or missing id yields ``UNDETERMINED`` so the event is surfaced
        to the guard.
        """
        if camera_id is None:
            return DirectionOutcome.UNDETERMINED
        try:
            cam_id = int(camera_id)
        except (TypeError, ValueError):
            return DirectionOutcome.UNDETERMINED
        return self._camera_direction_map.get(cam_id, DirectionOutcome.UNDETERMINED)

    def resolve(self, track: TrackHistory) -> DirectionOutcome:
        """Resolve using the track's ``camera_id`` (Protocol compatibility).

        Delegates to :meth:`resolve_for_camera`. In practice the pipeline calls
        :meth:`resolve_for_camera` directly with its own camera id; this method
        keeps the ``DirectionResolver`` Protocol satisfied.
        """
        return self.resolve_for_camera(getattr(track, "camera_id", None))


# ---------------------------------------------------------------------------
# Factory / selector
# ---------------------------------------------------------------------------


def _get(config: ConfigProvider, key: str, default):
    """Read a dotted config key, falling back to ``default`` when absent."""
    try:
        value = config.get(key)
    except KeyError:
        return default
    return default if value is None else value


def _as_orientation(value) -> tuple[float, float]:
    """Coerce a configured orientation into a ``(dx, dy)`` float tuple."""
    try:
        x, y = value
        return (float(x), float(y))
    except (TypeError, ValueError):
        return DEFAULT_INBOUND_REFERENCE_ORIENTATION


def _as_direction(value) -> DirectionOutcome:
    """Coerce a configured value into a :class:`DirectionOutcome`.

    Accepts a ``DirectionOutcome`` member or its string value
    (case-insensitive); anything unrecognized becomes ``UNDETERMINED`` so a
    misconfigured mapping fails safe to manual resolution rather than silently
    picking a direction.
    """
    if isinstance(value, DirectionOutcome):
        return value
    try:
        return DirectionOutcome(str(value).strip().upper())
    except (TypeError, ValueError, AttributeError):
        return DirectionOutcome.UNDETERMINED


def create_direction_resolver(config: ConfigProvider) -> DirectionResolver:
    """Build the configured :class:`DirectionResolver` (Req 7.1).

    Selects the resolver mode from ``direction.mode`` and constructs the
    matching resolver from the ``direction.*`` configuration, applying the
    documented defaults (``min_consecutive_frames=5``,
    ``inbound_reference_orientation=[1, 0]``, ``confidence_threshold=0.60``)
    when a value is absent.

    Raises
    ------
    ValueError
        When ``direction.mode`` is not one of the supported modes.
    """
    mode = _get(config, "direction.mode", DEFAULT_MODE)

    if mode == MODE_SINGLE_CAMERA_TRAJECTORY:
        return SingleCameraTrajectoryResolver(
            min_consecutive_frames=int(
                _get(config, "direction.min_consecutive_frames", DEFAULT_MIN_CONSECUTIVE_FRAMES)
            ),
            inbound_reference_orientation=_as_orientation(
                _get(
                    config,
                    "direction.inbound_reference_orientation",
                    DEFAULT_INBOUND_REFERENCE_ORIENTATION,
                )
            ),
            confidence_threshold=float(
                _get(config, "direction.confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
            ),
        )

    if mode == MODE_DUAL_CAMERA:
        return DualCameraResolver(
            camera_direction_map=_build_camera_direction_map(config)
        )

    raise ValueError(
        f"Unsupported direction.mode '{mode}'; expected one of "
        f"'{MODE_SINGLE_CAMERA_TRAJECTORY}' or '{MODE_DUAL_CAMERA}'."
    )


def _build_camera_direction_map(config: ConfigProvider) -> dict[int, DirectionOutcome]:
    """Build the camera-id -> direction map from the two configured cameras.

    Reads ``direction.cameras.inbound.id`` -> INBOUND and
    ``direction.cameras.outbound.id`` -> OUTBOUND. An explicit
    ``direction.camera_direction_map`` (ids -> direction strings), when present,
    is merged on top so an advanced configuration can override or extend the
    mapping. Non-integer ids are skipped (and are caught by config validation
    in dual mode).
    """
    mapping: dict[int, DirectionOutcome] = {}

    def _put(cam_id, direction: DirectionOutcome) -> None:
        if cam_id is None:
            return
        try:
            mapping[int(cam_id)] = direction
        except (TypeError, ValueError):
            pass

    _put(_get(config, "direction.cameras.inbound.id", None), DirectionOutcome.INBOUND)
    _put(_get(config, "direction.cameras.outbound.id", None), DirectionOutcome.OUTBOUND)

    for key, value in dict(_get(config, "direction.camera_direction_map", {}) or {}).items():
        try:
            mapping[int(key)] = _as_direction(value)
        except (TypeError, ValueError):
            continue

    return mapping
