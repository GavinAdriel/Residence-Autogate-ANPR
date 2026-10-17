"""Property-based tests for the pipeline :class:`TrackAggregator`.

Hypothesis exercises the two invariants that make "one logged event per vehicle"
correct, over randomized read sequences and frame-presence schedules (minimum
100 examples via the registered ``anpr`` profile):

* **Best-read is the maximum confidence seen** for an unresolved guest track,
  with ties resolved to the earliest read (stability).
* **Eviction removes exactly the tracks absent beyond the staleness window** and
  never a track present in the current frame.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
import pytest

from anpr.pipeline.track_aggregator import GuestRead, TrackAggregator


# A sequence of (confidence, plate) guest reads for a single track.
_reads = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=1, max_size=9),
    ),
    min_size=1,
    max_size=30,
)


@pytest.mark.property
@given(reads=_reads)
def test_best_read_is_max_confidence_with_first_wins_ties(reads) -> None:
    """The retained best read has the maximum confidence, ties kept earliest."""
    agg = TrackAggregator()
    for confidence, plate in reads:
        agg.observe_guest_read(1, GuestRead(confidence, plate))

    best = agg.best_read(1)
    max_conf = max(c for c, _ in reads)
    assert best is not None
    assert best.confidence == pytest.approx(max_conf)

    # The best read must be the *first* read that attained the max confidence:
    # replacement happens only on a strictly higher confidence.
    first_max_plate = next(p for c, p in reads if c == max_conf)
    assert best.normalized_plate == first_max_plate


@pytest.mark.property
@given(
    track_ids=st.lists(st.integers(min_value=0, max_value=20), min_size=1, max_size=12, unique=True),
    active=st.lists(st.integers(min_value=0, max_value=20), max_size=12),
    staleness=st.integers(min_value=0, max_value=3),
)
def test_eviction_never_removes_present_tracks(track_ids, active, staleness) -> None:
    """A track present in the active set is never evicted, whatever the window."""
    agg = TrackAggregator(staleness_frames=staleness)
    for tid in track_ids:
        agg.touch(tid)

    active_set = set(active)
    evicted = agg.evict_stale(active_set)

    # Nothing present this frame may be evicted.
    assert active_set.isdisjoint(set(evicted))
    # Everything present this frame (and previously tracked) remains tracked.
    for tid in track_ids:
        if tid in active_set:
            assert agg.is_tracked(tid)


@pytest.mark.property
@given(absent_frames=st.integers(min_value=1, max_value=10), staleness=st.integers(min_value=0, max_value=5))
def test_track_evicted_exactly_after_window_exceeded(absent_frames, staleness) -> None:
    """A continuously-absent track is evicted on the frame the window is exceeded."""
    agg = TrackAggregator(staleness_frames=staleness)
    agg.touch(1)

    evicted_on = None
    for frame in range(1, absent_frames + 1):
        evicted = agg.evict_stale(set())
        if evicted == [1]:
            evicted_on = frame
            break

    if absent_frames > staleness:
        # Eviction happens on the first absent frame strictly beyond the window.
        assert evicted_on == staleness + 1
    else:
        # Never absent long enough to exceed the window.
        assert evicted_on is None
        assert agg.is_tracked(1)
