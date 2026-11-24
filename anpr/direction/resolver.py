"""Direction resolution for the ANPR Autogate System.

Direction is a property of the camera, not of vehicle motion. The configuration
names each camera by the direction it watches (``cameras.inbound`` and
``cameras.outbound``), so a detection from the inbound camera is ``INBOUND`` and
a detection from the outbound camera is ``OUTBOUND``. The same resolver serves a
single-camera gate (one side configured) and a two-camera lane (both sides).

A camera id that is missing or not in the configured mapping resolves to
``UNDETERMINED``, so the event is surfaced to the guard instead of being acted on
automatically (dual-camera-exit-correlation Req 2.3).

This module depends only on the shared value objects and never on a concrete
peer implementation (Requirement 14.4). See
.kiro/specs/dual-camera-exit-correlation for the acceptance criteria.
"""

from __future__ import annotations

import logging

from anpr.core.interfaces import ConfigProvider
from anpr.core.models import DirectionOutcome

logger = logging.getLogger(__name__)


class CameraDirectionResolver:
    """Resolves direction from the id of the camera that produced a detection.

    The mapping is keyed by ``Camera.Camera_ID`` (the integer id stamped onto
    each detection and written to ``ANPR_Log.Camera_ID``), with values being
    :class:`DirectionOutcome` members. :func:`create_direction_resolver` builds
    it from the configured ``cameras.*.id`` values.

    This class structurally satisfies the ``DirectionResolver`` Protocol.
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
        """Resolve direction from the producing camera's id.

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


def _get(config: ConfigProvider, key: str, default):
    """Read a dotted config key, falling back to ``default`` when absent."""
    try:
        value = config.get(key)
    except KeyError:
        return default
    return default if value is None else value


def _as_direction(value) -> DirectionOutcome:
    """Coerce a value into a :class:`DirectionOutcome`.

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


def create_direction_resolver(config: ConfigProvider) -> CameraDirectionResolver:
    """Build the resolver from ``cameras.inbound.id`` and ``cameras.outbound.id``.

    Only the sides present in the configuration are mapped. Non-integer ids are
    skipped (config validation reports them at startup).
    """
    mapping: dict[int, DirectionOutcome] = {}
    for side, outcome in (
        ("inbound", DirectionOutcome.INBOUND),
        ("outbound", DirectionOutcome.OUTBOUND),
    ):
        raw = _get(config, f"cameras.{side}.id", None)
        if raw is None:
            continue
        try:
            mapping[int(raw)] = outcome
        except (TypeError, ValueError):
            logger.error("Invalid cameras.%s.id %r; expected an integer.", side, raw)
    return CameraDirectionResolver(camera_direction_map=mapping)
