"""Property-based test for manual-review queue retention.

Feature: mysql-monitoring-migration, Property 5
Property 5: Surfaced events persist until explicitly resolved.

Requirement 7.5 states that WHEN an event is surfaced for a manual decision, THE
Guard_Dashboard SHALL retain and display the event as unresolved until the Guard
records a resolution for it, and SHALL NOT automatically dismiss or remove the
event.

This reuses/extends the existing ``ManualReviewQueue`` retention property
(design Property 27, Req 12.7) -- the pure, Qt-free queue that backs the Guard
dashboard's manual-decision surface -- and asserts, for any sequence of surfaced
events, that each event:

* stays in the queue until it is explicitly resolved (persistence),
* is never auto-dismissed -- surfacing other events, querying, or resolving an
  unrelated/unknown token never removes it, and
* is removed exactly once on resolution (``resolve`` returns True the first time
  and False thereafter, and the length drops by exactly one).
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from anpr.core.models import DetectionEvent
from anpr.ui.guard_dashboard import ManualReviewQueue


def _surfaced_event(reason) -> DetectionEvent:
    """Build a minimal event as the pipeline would surface it for review."""
    return DetectionEvent(needs_manual_review=True, manual_review_reason=reason)


# A surfaced event is characterised (for retention purposes) only by whether it
# carries a review reason; generate absent/empty/arbitrary reasons so identical
# and distinct events both appear in the same queue.
reason_strategy = st.one_of(st.none(), st.text(max_size=24))


@pytest.mark.property
@given(
    reasons=st.lists(reason_strategy, max_size=20),
    data=st.data(),
)
def test_surfaced_events_persist_until_explicitly_resolved(reasons, data) -> None:
    """Validates: Requirements 7.5

    Feature: mysql-monitoring-migration, Property 5

    For any sequence of surfaced events, each event remains queued until it is
    explicitly resolved, is never auto-dismissed, and is removed exactly once on
    resolution.
    """
    queue = ManualReviewQueue()
    events = [_surfaced_event(reason) for reason in reasons]

    # --- Surfacing: every event stays; adding others never auto-dismisses. ---
    tokens: list[int] = []
    for index, event in enumerate(events):
        token = queue.surface(event)
        tokens.append(token)
        # Persistence + no auto-dismiss: all previously surfaced events remain.
        assert len(queue) == index + 1
        for prior in tokens:
            assert prior in queue

    # Tokens are unique so every surfaced event is independently resolvable.
    assert len(set(tokens)) == len(tokens)

    # The queue displays exactly the surfaced events, in FIFO order, with the
    # original event objects and reasons preserved (retained "as unresolved").
    pending = queue.pending
    assert [entry.token for entry in pending] == tokens
    assert [entry.event for entry in pending] == events
    assert [entry.reason for entry in pending] == reasons

    if not tokens:
        # Nothing surfaced: resolving anything removes nothing.
        assert queue.resolve(0) is False
        assert len(queue) == 0
        return

    # Resolving a token that was never issued dismisses nothing (no auto-drop).
    assert queue.resolve(max(tokens) + 1) is False
    assert queue.resolve(-1) is False
    assert len(queue) == len(tokens)
    for token in tokens:
        assert token in queue

    # --- Resolution: explicit only, exactly once, others untouched. ---
    remaining = set(tokens)
    for token in data.draw(st.permutations(tokens)):
        assert token in queue
        before = len(queue)

        # Removed exactly once: the first resolve reports success.
        assert queue.resolve(token) is True
        assert len(queue) == before - 1
        assert token not in queue
        remaining.discard(token)

        # No auto-dismiss: every not-yet-resolved event is still present.
        for other in remaining:
            assert other in queue

        # Exactly once: resolving the same token again removes nothing.
        assert queue.resolve(token) is False
        assert len(queue) == len(remaining)

    # All events left only via explicit resolution.
    assert len(queue) == 0
    assert queue.pending == []
