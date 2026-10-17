"""Per-track deduplication state for the ANPR Autogate detection pipeline.

:class:`TrackAggregator` collapses the many per-frame detections that a single
vehicle produces (the YOLO tracker emits one :class:`~anpr.core.models.Detection`
per frame, each carrying a persistent ``track_id``) into **exactly one logged
event per physical vehicle pass**. It is a pure, in-memory data structure with
no I/O and no dependency on any Protocol or concrete peer, so it lives in the
pipeline layer and leaves ``core/`` untouched (ports-and-adapters: dedup is a
pipeline concern, not domain logic).

Two track lifecycles are tracked, keyed by ``track_id``:

* **Resident** -- the first confident, format-valid read that matches the
  resident whitelist permits the vehicle, writes one record, opens the gate, and
  immediately marks the track *resolved*. The pipeline then skips every later
  frame of that track before OCR runs ("OCR will not read the plate anymore").
* **Guest** -- no database write happens during detection. The aggregator keeps
  the single best (strictly-higher OCR confidence) read seen for the track so
  far, and the pipeline surfaces the guest to the guard exactly once. The guest
  record is written only when the guard presses the entry-permit button, after
  which the track is marked *resolved*. A guest the guard never permits writes
  no record at all -- it simply ages out via :meth:`evict_stale`.

Because residents stop OCR on their first match and guests are written only once
at button-press, no "update an existing record" path is ever needed: every
written row is a single ``append``.

Eviction: the pipeline receives no explicit "track ended" signal from the
tracker, so track end is *inferred*. After each processed frame the pipeline
calls :meth:`evict_stale` with the ``track_id`` set present in that frame; a
track absent for more than a small configurable grace window (in frames) is
forgotten, freeing its state and -- for an unresolved guest -- discarding its
best read so a vehicle that reverses out leaves no phantom entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Default number of consecutive frames a track may be absent from the active set
# before it is evicted. One frame of grace absorbs a single-frame tracker gap
# (a momentary miss) without prematurely forgetting a still-present vehicle.
DEFAULT_STALENESS_FRAMES = 1


@dataclass(frozen=True)
class GuestRead:
    """A single OCR read retained as the best-so-far for a guest track.

    ``confidence`` is the OCR recognition confidence used to pick the best read;
    ``normalized_plate`` is the normalizer output (used to prefill the guard's
    manual grant input); ``ocr_text`` retains the raw read for display/audit.
    """

    confidence: float
    normalized_plate: str
    ocr_text: str = ""


@dataclass
class _TrackState:
    """Mutable per-track bookkeeping held by the aggregator.

    ``resolved`` marks a track that has already been permitted/logged (resident
    at detection, or guest at the guard's button press); once resolved the
    pipeline skips the track entirely. ``best_read`` is the best guest read seen
    so far (``None`` until the first confident guest read, and irrelevant once
    resolved). ``surfaced`` records whether the guest has already been shown to
    the guard, so it is surfaced exactly once. ``frames_since_seen`` drives
    eviction: it is reset to zero whenever the track is observed and incremented
    for every frame the track is absent.
    """

    resolved: bool = False
    surfaced: bool = False
    best_read: Optional[GuestRead] = None
    frames_since_seen: int = 0


class TrackAggregator:
    """Collapses per-frame detections into one logged event per track_id.

    Parameters
    ----------
    staleness_frames:
        Number of consecutive frames a track may be absent from the active set
        before :meth:`evict_stale` forgets it. Clamped to a minimum of ``0``
        (evict as soon as a track is missing from a frame). Defaults to
        :data:`DEFAULT_STALENESS_FRAMES`.
    """

    def __init__(self, *, staleness_frames: int = DEFAULT_STALENESS_FRAMES) -> None:
        self._staleness_frames = max(0, int(staleness_frames))
        self._tracks: dict[int, _TrackState] = {}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def staleness_frames(self) -> int:
        """Effective (clamped) eviction grace window in frames."""
        return self._staleness_frames

    def is_tracked(self, track_id: int) -> bool:
        """Whether any state is currently held for ``track_id``."""
        return track_id in self._tracks

    def is_resolved(self, track_id: int) -> bool:
        """Whether ``track_id`` has already been permitted/logged.

        A resolved track is skipped by the pipeline before OCR runs, so a
        resident is logged once and its later frames are ignored, and a
        guard-permitted guest is never re-surfaced or re-processed. An unknown
        track is not resolved.
        """
        state = self._tracks.get(track_id)
        return state is not None and state.resolved

    def is_surfaced(self, track_id: int) -> bool:
        """Whether the guest on ``track_id`` has already been shown to the guard."""
        state = self._tracks.get(track_id)
        return state is not None and state.surfaced

    def best_read(self, track_id: int) -> Optional[GuestRead]:
        """Return the best guest read retained for ``track_id`` (or ``None``)."""
        state = self._tracks.get(track_id)
        return state.best_read if state is not None else None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def touch(self, track_id: int) -> None:
        """Register that ``track_id`` was seen this frame, resetting staleness.

        Creates fresh state for a previously unseen track. Used by the pipeline
        for every observed detection (including resolved tracks it is about to
        skip) so an in-view vehicle is never evicted mid-pass.
        """
        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackState()
            self._tracks[track_id] = state
        state.frames_since_seen = 0

    def observe_guest_read(self, track_id: int, read: GuestRead) -> Optional[GuestRead]:
        """Record a guest read for ``track_id``, keeping the best so far.

        The retained read is replaced only by a *strictly* higher OCR
        confidence; an equal or lower confidence leaves the existing best read
        in place (so the first read wins ties and the result is stable). Also
        resets the staleness counter. Returns the resulting best read.

        Observing a read on a resolved track is a no-op that returns the
        existing best read unchanged (which may be ``None``): once a track is
        permitted its plate is fixed and no later read is adopted. In practice
        the pipeline skips resolved tracks before OCR, so this guard is purely
        defensive.
        """
        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackState()
            self._tracks[track_id] = state
        state.frames_since_seen = 0

        if state.resolved:
            return state.best_read

        if state.best_read is None or read.confidence > state.best_read.confidence:
            state.best_read = read
        return state.best_read

    def mark_surfaced(self, track_id: int) -> None:
        """Mark the guest on ``track_id`` as shown to the guard (surface once).

        Idempotent; creates state for an unknown track so a surface decision is
        never lost.
        """
        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackState()
            self._tracks[track_id] = state
        state.surfaced = True

    def mark_resolved(self, track_id: int) -> None:
        """Mark ``track_id`` as permitted/logged so it is skipped thereafter.

        Idempotent; creates state for an unknown track so a resolution is never
        lost (e.g. a guard grant arriving after the track briefly aged out).
        """
        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackState()
            self._tracks[track_id] = state
        state.resolved = True

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------
    def evict_stale(self, active_track_ids: "set[int] | frozenset[int]") -> list[int]:
        """Forget tracks absent beyond the staleness window; return evicted ids.

        For every tracked id not in ``active_track_ids`` the absence counter is
        incremented; a track whose absence exceeds ``staleness_frames`` is
        removed. Tracks present in ``active_track_ids`` have their counter reset.
        Returns the list of evicted ``track_id``s (useful for logging/tests).

        Evicting an unresolved guest discards its best read, which is the
        intended "no phantom entry for a vehicle that never entered" behavior.
        """
        evicted: list[int] = []
        for track_id, state in list(self._tracks.items()):
            if track_id in active_track_ids:
                state.frames_since_seen = 0
                continue
            state.frames_since_seen += 1
            if state.frames_since_seen > self._staleness_frames:
                del self._tracks[track_id]
                evicted.append(track_id)
        return evicted

    def __len__(self) -> int:
        return len(self._tracks)
