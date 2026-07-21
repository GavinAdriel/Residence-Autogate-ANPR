"""On-disk image capture, compression, thumbnailing, and retention.

The ``Image_Store`` captures the frame image for a detection event, writes an
efficiently compressed snapshot plus a small thumbnail to a configured storage
location, and records a *reference* to those on-disk files in the database.
Full image binaries are never stored in the database (Requirement 11.3): the
``images`` table holds only the snapshot/thumbnail paths and the capture time,
so the store can serve thumbnails and prune old images later.

Behavioural contract (design Image_Store section, Req 11):

* :meth:`DiskImageStore.capture_and_store` — when capture is enabled, produces a
  compressed snapshot whose file size is *strictly smaller* than the original
  captured frame, plus a thumbnail whose longest edge does not exceed the
  configured maximum (default 320 px), persists both to the configured storage
  location, records the reference in the database, and completes within the
  capture window (≤1000 ms) (Req 11.1-11.4). On a capture-encoding failure it
  returns ``None`` so the caller keeps the Event_Log record without an image
  reference and marks the image unavailable (Req 11.7). On a storage
  failure (location unavailable or disk full) it logs an error identifying the
  storage location and returns ``None`` (Req 11.8).
* :meth:`DiskImageStore.get_thumbnail` — serves the thumbnail bytes for a stored
  reference so the Guard_Dashboard shows the small image (Req 11.5).
* :meth:`DiskImageStore.run_retention` — deletes stored images older than the
  configurable retention period (1-365 days) and returns the count deleted
  (Req 11.6). The evaluation cadence is bounded by
  :attr:`retention_interval_h` (≤24 h) which the composition-root scheduler
  honours (Req 11.9).

OpenCV (``cv2``) performs the JPEG encoding and thumbnail resizing, matching the
rest of the project. It is imported lazily inside the methods that need it so
this module stays importable before the heavy dependency is installed (the same
pattern used by the video source and detector wrappers).

This concrete class structurally satisfies the ``ImageStore`` Protocol in
``anpr.core.interfaces``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from anpr.core.interfaces import ConfigProvider, Frame
from anpr.core.models import ImageRef
from anpr.persistence.db import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration keys and defaults (design Configuration / storage section)
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_DIR = "./images"
DEFAULT_CAPTURE_ENABLED = True
# Thumbnail longest-edge maximum in pixels (Req 11.4).
DEFAULT_THUMBNAIL_MAX_PX = 320
# Retention period in days; the valid range is 1..365 (Req 11.6).
DEFAULT_RETENTION_DAYS = 30
# Retention evaluation interval in hours; must not exceed 24 (Req 11.9).
DEFAULT_RETENTION_INTERVAL_H = 24

# Upper bound on the capture window (Req 11.1). Capture is expected to finish
# well within this; exceeding it is logged but does not invalidate the image.
CAPTURE_WINDOW_MS = 1000

# JPEG quality ladder tried, in descending order, when compressing the
# snapshot. A high quality is preferred; lower rungs are used only as needed to
# guarantee the snapshot file is strictly smaller than the original frame
# (Req 11.4). Thumbnails use the first (highest) quality since their small
# pixel dimensions already keep them tiny.
_JPEG_QUALITY_LADDER = (90, 75, 60, 45, 30, 20, 10, 5)


class ImageCaptureError(RuntimeError):
    """Raised internally when encoding a captured frame fails (Req 11.7)."""


class DiskImageStore:
    """Captures, serves, and retains event frame images as files on disk.

    Parameters
    ----------
    db:
        A shared :class:`~anpr.persistence.db.Database` instance, or a database
        location string. Passing a string constructs the ``Database``
        internally; passing an instance shares the connection/schema with the
        repositories.
    image_dir:
        Storage location for image files (config ``storage.image_dir``).
    capture_enabled:
        Whether capture is active (config ``storage.capture_enabled``, Req 11.1).
    thumbnail_max_px:
        Thumbnail longest-edge maximum in pixels (config
        ``storage.thumbnail_max_px``, default 320) (Req 11.4).
    retention_days:
        Retention period in days, clamped to the valid 1..365 range
        (config ``storage.retention_days``) (Req 11.6).
    retention_interval_h:
        Retention evaluation interval in hours, clamped to at most 24
        (config ``storage.retention_interval_h``) (Req 11.9).
    """

    def __init__(
        self,
        db: Database | str,
        image_dir: str = DEFAULT_IMAGE_DIR,
        capture_enabled: bool = DEFAULT_CAPTURE_ENABLED,
        thumbnail_max_px: int = DEFAULT_THUMBNAIL_MAX_PX,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        retention_interval_h: int = DEFAULT_RETENTION_INTERVAL_H,
    ) -> None:
        self._db = db if isinstance(db, Database) else Database(db)
        self._image_dir = Path(image_dir)
        self._capture_enabled = bool(capture_enabled)
        # Guard against misconfiguration: a thumbnail edge must be positive.
        self._thumbnail_max_px = max(1, int(thumbnail_max_px))
        # Clamp to the design's valid range so a stray value can never widen the
        # retention period beyond 365 days or below 1 (Req 11.6).
        self._retention_days = _clamp(int(retention_days), 1, 365)
        # The evaluation interval must never exceed 24 h (Req 11.9).
        self._retention_interval_h = _clamp(int(retention_interval_h), 1, 24)

    # ------------------------------------------------------------------
    # Read-only configuration surface
    # ------------------------------------------------------------------
    @property
    def image_dir(self) -> Path:
        """The configured storage location for image files."""
        return self._image_dir

    @property
    def capture_enabled(self) -> bool:
        """Whether image capture is enabled (Req 11.1)."""
        return self._capture_enabled

    @property
    def thumbnail_max_px(self) -> int:
        """The thumbnail longest-edge maximum in pixels (Req 11.4)."""
        return self._thumbnail_max_px

    @property
    def retention_days(self) -> int:
        """The effective retention period in days (1..365) (Req 11.6)."""
        return self._retention_days

    @property
    def retention_interval_h(self) -> int:
        """The retention evaluation interval in hours (≤24) (Req 11.9)."""
        return self._retention_interval_h

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def capture_and_store(self, frame: Frame, event_id: str) -> Optional[ImageRef]:
        """Capture and persist a snapshot + thumbnail for ``event_id``.

        Returns the stored :class:`~anpr.core.models.ImageRef` on success, or
        ``None`` when capture is disabled, the frame cannot be encoded
        (Req 11.7), or the storage location is unavailable/full (Req 11.8). In
        every ``None`` case the caller keeps the Event_Log record without an
        image reference.
        """
        # Req 11.1: only capture when capture is enabled by configuration.
        if not self._capture_enabled:
            return None

        started = time.monotonic()

        # 1) Encode both artifacts in memory before touching disk, so an
        #    encoding failure (Req 11.7) is cleanly separable from a storage
        #    failure (Req 11.8) and never leaves a half-written pair on disk.
        try:
            snapshot_bytes = self._encode_snapshot(frame)
            thumbnail_bytes = self._encode_thumbnail(frame)
        except ImageCaptureError as exc:
            # Req 11.7: capture failed -> keep the record without an image ref
            # and mark the image unavailable (the caller does so on None).
            logger.error("Image capture failed for event %s: %s", event_id, exc)
            return None

        # 2) Persist both files to the configured storage location (Req 11.2).
        snapshot_path = self._image_dir / f"{event_id}.jpg"
        thumbnail_path = self._image_dir / f"{event_id}_thumb.jpg"
        try:
            self._image_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(snapshot_bytes)
            thumbnail_path.write_bytes(thumbnail_bytes)
        except OSError as exc:
            # Req 11.8: storage unavailable/full -> log the storage location and
            # keep the record without an image ref (the caller does so on None).
            logger.error(
                "Image storage unavailable for event %s at storage location '%s': %s",
                event_id,
                self._image_dir,
                exc,
            )
            _best_effort_unlink(snapshot_path, thumbnail_path)
            return None

        captured_at = _now_iso()
        ref = ImageRef(
            event_id=event_id,
            snapshot_path=str(snapshot_path),
            thumbnail_path=str(thumbnail_path),
            captured_at=captured_at,
        )

        # 3) Record the reference in the database (Req 11.3: references only).
        try:
            self._insert_reference(ref)
        except Exception as exc:  # noqa: BLE001 - treat any DB fault as storage failure
            logger.error(
                "Failed to record image reference for event %s at storage location '%s': %s",
                event_id,
                self._image_dir,
                exc,
            )
            _best_effort_unlink(snapshot_path, thumbnail_path)
            return None

        # Req 11.1: capture is expected to complete within the capture window.
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms > CAPTURE_WINDOW_MS:
            logger.warning(
                "Image capture for event %s took %.0f ms, exceeding the %d ms capture window.",
                event_id,
                elapsed_ms,
                CAPTURE_WINDOW_MS,
            )
        return ref

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------
    def get_thumbnail(self, ref: ImageRef) -> bytes:
        """Return the thumbnail bytes for a stored image reference (Req 11.5)."""
        return Path(ref.thumbnail_path).read_bytes()

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------
    def run_retention(self, now: datetime) -> int:
        """Delete images older than the retention period; return count deleted.

        An image is *older than the retention period* when its capture time is
        strictly before ``now`` minus the configured retention days (Req 11.6).
        Both the on-disk files and the database reference row are removed. The
        returned integer is the number of image records deleted.
        """
        cutoff = _as_aware(now) - timedelta(days=self._retention_days)

        rows = self._db.connection.execute(
            "SELECT event_id, snapshot_path, thumbnail_path, captured_at FROM images"
        ).fetchall()

        expired_ids: list[str] = []
        for row in rows:
            captured_at = _parse_iso(row["captured_at"])
            if captured_at is None:
                # A reference we cannot interpret is left untouched rather than
                # risking deletion of an image that may still be within period.
                logger.warning(
                    "Skipping retention for image %s: unparseable capture time '%s'.",
                    row["event_id"],
                    row["captured_at"],
                )
                continue
            if captured_at < cutoff:
                _best_effort_unlink(Path(row["snapshot_path"]), Path(row["thumbnail_path"]))
                expired_ids.append(row["event_id"])

        if not expired_ids:
            return 0

        with self._db.connection as conn:
            conn.executemany(
                "DELETE FROM images WHERE event_id = ?",
                [(event_id,) for event_id in expired_ids],
            )
        return len(expired_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode_snapshot(self, frame: Frame) -> bytes:
        """Encode ``frame`` as a JPEG strictly smaller than the original.

        The "original captured image" is the raw, uncompressed frame; its size
        is ``frame.nbytes``. The quality ladder is walked from high to low and
        the first encoding whose byte length is strictly below that raw size is
        returned, guaranteeing the compressed snapshot shrinks (Req 11.4).
        """
        import cv2

        original_size = _frame_nbytes(frame)
        best: Optional[bytes] = None
        for quality in _JPEG_QUALITY_LADDER:
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                raise ImageCaptureError("OpenCV failed to encode the snapshot frame.")
            data = buffer.tobytes()
            if best is None or len(data) < len(best):
                best = data
            if len(data) < original_size:
                return data

        # No quality produced a file strictly smaller than the raw frame. This
        # only happens for degenerate, tiny inputs that are not real capture
        # frames; treat it as a capture failure (Req 11.7).
        raise ImageCaptureError(
            f"Could not compress snapshot below the original size of {original_size} bytes "
            f"(smallest encoding was {0 if best is None else len(best)} bytes)."
        )

    def _encode_thumbnail(self, frame: Frame) -> bytes:
        """Encode a JPEG thumbnail whose longest edge ≤ the configured max.

        The frame is downscaled with area interpolation so the longer of its
        two edges equals the configured maximum; a frame already within the
        maximum is used as-is (Req 11.4).
        """
        import cv2

        try:
            height, width = int(frame.shape[0]), int(frame.shape[1])
        except (AttributeError, IndexError, ValueError) as exc:
            raise ImageCaptureError(f"Frame has no usable image dimensions: {exc}") from exc

        if height <= 0 or width <= 0:
            raise ImageCaptureError("Frame has zero-sized dimensions.")

        longest_edge = max(height, width)
        if longest_edge > self._thumbnail_max_px:
            scale = self._thumbnail_max_px / longest_edge
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            thumbnail = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        else:
            thumbnail = frame

        ok, buffer = cv2.imencode(
            ".jpg", thumbnail, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY_LADDER[0]]
        )
        if not ok:
            raise ImageCaptureError("OpenCV failed to encode the thumbnail.")
        return buffer.tobytes()

    def _insert_reference(self, ref: ImageRef) -> None:
        """Insert (or replace) the on-disk image reference row (Req 11.3)."""
        with self._db.connection as conn:
            conn.execute(
                "INSERT OR REPLACE INTO images "
                "(event_id, snapshot_path, thumbnail_path, captured_at) "
                "VALUES (?, ?, ?, ?)",
                (ref.event_id, ref.snapshot_path, ref.thumbnail_path, ref.captured_at),
            )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def _clamp(value: int, low: int, high: int) -> int:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def _frame_nbytes(frame: Frame) -> int:
    """Return the raw byte size of a frame (the original captured image)."""
    nbytes = getattr(frame, "nbytes", None)
    if isinstance(nbytes, int) and nbytes > 0:
        return nbytes
    raise ImageCaptureError("Frame is empty or not a valid image array.")


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with tz offset."""
    return datetime.now(timezone.utc).isoformat()


def _as_aware(dt: datetime) -> datetime:
    """Return ``dt`` as a timezone-aware datetime, assuming UTC when naive."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _parse_iso(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into an aware datetime, or ``None``."""
    try:
        return _as_aware(datetime.fromisoformat(value))
    except (TypeError, ValueError):
        return None


def _best_effort_unlink(*paths: Path) -> None:
    """Remove image files if present, logging (not raising) on failure."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - unusual filesystem state
            logger.warning("Could not remove image file '%s': %s", path, exc)


# ----------------------------------------------------------------------
# Factory / selector
# ----------------------------------------------------------------------
def _get(config: ConfigProvider, key: str, default):
    """Read a dotted config key, falling back to ``default`` when absent."""
    try:
        value = config.get(key)
    except KeyError:
        return default
    return default if value is None else value


def create_image_store(config: ConfigProvider, db: Database | str) -> DiskImageStore:
    """Build a :class:`DiskImageStore` from ``storage.*`` configuration.

    Reads the storage location, capture flag, thumbnail maximum, retention
    period, and retention interval from configuration, applying the documented
    defaults when a value is absent (Req 11.1-11.6, 11.9).
    """
    return DiskImageStore(
        db=db,
        image_dir=str(_get(config, "storage.image_dir", DEFAULT_IMAGE_DIR)),
        capture_enabled=bool(_get(config, "storage.capture_enabled", DEFAULT_CAPTURE_ENABLED)),
        thumbnail_max_px=int(_get(config, "storage.thumbnail_max_px", DEFAULT_THUMBNAIL_MAX_PX)),
        retention_days=int(_get(config, "storage.retention_days", DEFAULT_RETENTION_DAYS)),
        retention_interval_h=int(
            _get(config, "storage.retention_interval_h", DEFAULT_RETENTION_INTERVAL_H)
        ),
    )
