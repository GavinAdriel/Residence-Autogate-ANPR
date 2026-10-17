"""Unit tests for the pipeline :class:`TrackAggregator`.

Example-based coverage of the per-``track_id`` deduplication state that collapses
many per-frame detections into one logged event per vehicle:

* resolved tracks report resolved and are skipped regardless of later reads;
* a guest track keeps the strictly-higher-confidence best read (ties keep the
  first);
* surfacing is recorded once;
* eviction removes only tracks absent beyond the staleness window and discards an
  unresolved guest's best read.

The aggregator is pure (no I/O, no Protocol/concrete peer), so these tests need
no camera, model, database, or GUI.
"""

from __future__ import annotations

import pytest

from anpr.pipeline.track_aggregator import (
    DEFAULT_STALENESS_FRAMES,
    GuestRead,
    TrackAggregator,
)


@pytest.mark.unit
def test_unknown_track_is_not_resolved_or_surfaced() -> None:
    agg = TrackAggregator()
    assert agg.is_resolved(1) is False
    assert agg.is_surfaced(1) is False
    assert agg.best_read(1) is None
    assert agg.is_tracked(1) is False


@pytest.mark.unit
def test_mark_resolved_makes_track_resolved_and_skipped() -> None:
    agg = TrackAggregator()
    agg.mark_resolved(7)
    assert agg.is_resolved(7) is True
    # A later read on a resolved track is not adopted (best read stays None).
    returned = agg.observe_guest_read(7, GuestRead(0.99, "B9ZZZ"))
    assert returned is None
    assert agg.best_read(7) is None


@pytest.mark.unit
def test_guest_best_read_keeps_strictly_higher_confidence() -> None:
    agg = TrackAggregator()
    agg.observe_guest_read(3, GuestRead(0.72, "B1234XY"))
    agg.observe_guest_read(3, GuestRead(0.60, "WRONG"))  # lower -> ignored
    assert agg.best_read(3).normalized_plate == "B1234XY"

    agg.observe_guest_read(3, GuestRead(0.91, "B1234XYZ"))  # higher -> wins
    assert agg.best_read(3).normalized_plate == "B1234XYZ"
    assert agg.best_read(3).confidence == pytest.approx(0.91)


@pytest.mark.unit
def test_guest_best_read_tie_keeps_first() -> None:
    agg = TrackAggregator()
    agg.observe_guest_read(4, GuestRead(0.80, "FIRST"))
    agg.observe_guest_read(4, GuestRead(0.80, "SECOND"))  # equal -> keep first
    assert agg.best_read(4).normalized_plate == "FIRST"


@pytest.mark.unit
def test_mark_surfaced_is_recorded_and_idempotent() -> None:
    agg = TrackAggregator()
    assert agg.is_surfaced(5) is False
    agg.mark_surfaced(5)
    agg.mark_surfaced(5)
    assert agg.is_surfaced(5) is True


@pytest.mark.unit
def test_touch_resets_staleness_so_present_track_is_not_evicted() -> None:
    agg = TrackAggregator(staleness_frames=1)
    agg.touch(1)
    # Frame where track 1 is still present: not evicted.
    assert agg.evict_stale({1}) == []
    assert agg.is_tracked(1) is True


@pytest.mark.unit
def test_evict_stale_removes_track_absent_beyond_window() -> None:
    agg = TrackAggregator(staleness_frames=1)
    agg.observe_guest_read(2, GuestRead(0.9, "B1XY"))
    # Absent one frame: within the 1-frame grace, still tracked.
    assert agg.evict_stale(set()) == []
    assert agg.is_tracked(2) is True
    # Absent a second frame: exceeds the window, evicted and best read gone.
    assert agg.evict_stale(set()) == [2]
    assert agg.is_tracked(2) is False
    assert agg.best_read(2) is None


@pytest.mark.unit
def test_evict_stale_only_removes_absent_tracks() -> None:
    agg = TrackAggregator(staleness_frames=0)
    agg.touch(1)
    agg.touch(2)
    # Track 2 present, track 1 absent with a zero-frame window -> only 1 evicted.
    evicted = agg.evict_stale({2})
    assert evicted == [1]
    assert agg.is_tracked(2) is True
    assert agg.is_tracked(1) is False


@pytest.mark.unit
def test_resolved_track_survives_until_absent_then_evicts() -> None:
    agg = TrackAggregator(staleness_frames=0)
    agg.mark_resolved(9)
    # Still present -> stays resolved.
    assert agg.evict_stale({9}) == []
    assert agg.is_resolved(9) is True
    # Absent with a zero window -> evicted; a brand-new track id is unresolved.
    assert agg.evict_stale(set()) == [9]
    assert agg.is_resolved(9) is False


@pytest.mark.unit
def test_negative_staleness_is_clamped_to_zero() -> None:
    agg = TrackAggregator(staleness_frames=-5)
    assert agg.staleness_frames == 0


@pytest.mark.unit
def test_default_staleness_window() -> None:
    agg = TrackAggregator()
    assert agg.staleness_frames == DEFAULT_STALENESS_FRAMES
