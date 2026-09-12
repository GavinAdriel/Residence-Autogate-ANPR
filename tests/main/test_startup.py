"""Startup ordering and failure-mode tests for the composition root.

These tests exercise :func:`anpr.main.build_application` -- the single non-GUI
composition root -- with **no live MySQL** (a fake ``Database`` is injected in
place of the real PyMySQL-backed one) and without loading the heavy detector
weights. They assert the documented startup ordering and the fail-safe startup
behavior:

* the ordered builders run in the sequence
  ``validate -> connect -> table-check -> persistence/imaging -> video/gate ->
  weights -> pipeline`` (Req 9.6);
* a MySQL connect failure surfaces as :class:`~anpr.main.StartupError` and no
  downstream component (and no dashboard) is ever built (Req 1.4, 9.7);
* a missing required table surfaces as :class:`StartupError` (Req 1.5);
* any step failing halts every subsequent step so no partially-wired system
  starts (Req 9.8);
* an *integration* variant assembles the whole non-GUI object graph over a fake
  ``Database`` (simulation gate, webcam camera) and tears it down cleanly with
  no live MySQL.

The fake ``Database`` subclasses the real
:class:`anpr.persistence.db.Database` (overriding ``__init__`` so it never opens
a socket) so that the ``DiskImageStore``'s ``isinstance(db, Database)`` check
still passes and no real connection is attempted anywhere in the graph.
"""

from __future__ import annotations

import sys

import pytest

import anpr.main as main
from anpr.gate.controller import SimulatedGate
from anpr.imaging.store import DiskImageStore
from anpr.main import StartupError
from anpr.persistence.db import Database as RealDatabase
from anpr.persistence.db import MissingTableError
from anpr.persistence.event_log_repo import MySqlEventLogRepository
from anpr.persistence.resident_repo import MySqlResidentRepository
from anpr.pipeline.video_source import WebcamVideoSource


# ---------------------------------------------------------------------------
# Fake Database (no live MySQL)
# ---------------------------------------------------------------------------
def _make_fake_db_class(order=None, *, connect_error=None, table_error=None):
    """Build a fake ``Database`` class wired for a specific test scenario.

    The returned class subclasses the real :class:`Database` (so the image
    store's ``isinstance`` check passes) but never calls ``super().__init__``,
    so no socket is ever opened. ``order`` (when given) records ``"connect"`` on
    construction and ``"table-check"`` on :meth:`check_required_tables` so the
    startup ordering can be asserted. ``connect_error`` / ``table_error`` drive
    the failure-mode paths.
    """

    class _FakeDatabase(RealDatabase):
        instances: list = []

        def __init__(self, **kwargs):  # noqa: D401 - intentionally no super()
            self.kwargs = kwargs
            self.closed = False
            if order is not None:
                order.append("connect")
            if connect_error is not None:
                raise connect_error
            _FakeDatabase.instances.append(self)

        def check_required_tables(self) -> None:
            if table_error is not None:
                raise table_error
            if order is not None:
                order.append("table-check")

        def query(self, sql, params=()):  # pragma: no cover - not hit in assembly
            return []

        def execute(self, sql, params=()):  # pragma: no cover - not hit in assembly
            return 0

        @property
        def name(self) -> str:
            return self.kwargs.get("name", "anpr")

        def close(self) -> None:
            self.closed = True

    return _FakeDatabase


# The ordered builders enforced by the composition root, in call order. The
# connect/table-check sub-steps live inside ``build_repositories`` and are
# recorded by the fake ``Database`` itself.
ORDERED_BUILDERS = [
    "validate_config_or_halt",
    "build_repositories",
    "build_image_store",
    "build_video_source",
    "build_gate",
    "build_detector",
    "load_detector_weights_or_halt",
    "build_pipeline",
]

# Harmless stand-in return values so a recording spy can replace a real builder
# without constructing anything real.
_FAKE_RETURNS = {
    "validate_config_or_halt": None,
    "build_repositories": None,  # replaced per-test with a fake 3-tuple
    "build_image_store": object(),
    "build_video_source": object(),
    "build_gate": object(),
    "build_detector": object(),
    "load_detector_weights_or_halt": None,
    "build_pipeline": object(),
}


def _spy(monkeypatch, name, order, *, delegate=True, result=None, raises=None):
    """Install a recording wrapper around ``anpr.main.<name>``.

    Records ``name`` into ``order`` when invoked. ``delegate`` calls through to
    the original builder (real construction); otherwise it returns ``result``.
    ``raises`` makes the wrapper raise the given exception after recording.
    """
    original = getattr(main, name)

    def wrapper(*args, **kwargs):
        order.append(name)
        if raises is not None:
            raise raises
        if delegate:
            return original(*args, **kwargs)
        return result

    monkeypatch.setattr(main, name, wrapper)
    return original


# ---------------------------------------------------------------------------
# Startup ordering (Req 9.6)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_build_application_runs_startup_steps_in_documented_order(monkeypatch):
    """validate -> connect -> table-check -> persistence/imaging -> video/gate
    -> weights -> pipeline (Req 9.6)."""
    order: list[str] = []

    # Inject the fake DB so build_repositories connects + checks tables against
    # it (recording "connect"/"table-check") with no live MySQL.
    monkeypatch.setattr(main, "Database", _make_fake_db_class(order))

    # Wrap the ordered builders so each records its own step. build_repositories
    # runs for real (delegated) so the connect/table-check sub-steps are
    # recorded by the fake DB in their natural place. The weights load is
    # recorded but not delegated so no heavy YOLO weights are read.
    _spy(monkeypatch, "validate_config_or_halt", order, delegate=True)
    _spy(monkeypatch, "build_image_store", order, delegate=True)
    _spy(monkeypatch, "build_video_source", order, delegate=True)
    _spy(monkeypatch, "build_gate", order, delegate=True)
    _spy(monkeypatch, "build_detector", order, delegate=True)
    _spy(monkeypatch, "load_detector_weights_or_halt", order, delegate=False)
    _spy(monkeypatch, "build_pipeline", order, delegate=True)

    app = main.build_application(load_weights=True)

    assert order == [
        "validate_config_or_halt",
        "connect",
        "table-check",
        "build_image_store",
        "build_video_source",
        "build_gate",
        "build_detector",
        "load_detector_weights_or_halt",
        "build_pipeline",
    ]
    # The assembled graph is wired onto the injected fake DB.
    assert isinstance(app.database, RealDatabase)
    assert app.database.kwargs["host"] == "127.0.0.1"


@pytest.mark.smoke
def test_connect_precedes_table_check_which_precedes_repositories(monkeypatch):
    """The MySQL connect happens before the table check, which happens before
    any persistence/imaging is wired (Req 9.6)."""
    order: list[str] = []
    monkeypatch.setattr(main, "Database", _make_fake_db_class(order))
    _spy(monkeypatch, "build_image_store", order, delegate=True)
    _spy(monkeypatch, "load_detector_weights_or_halt", order, delegate=False)

    main.build_application(load_weights=False)

    assert order.index("connect") < order.index("table-check")
    assert order.index("table-check") < order.index("build_image_store")


# ---------------------------------------------------------------------------
# Failure mode: MySQL connect failure (Req 1.4, 9.7)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_connect_failure_raises_startup_error_and_builds_no_dashboard(monkeypatch):
    """A ``ConnectionError`` from the injected ``Database`` halts startup as a
    ``StartupError``; nothing downstream (and no dashboard) is built
    (Req 1.4, 9.7)."""
    order: list[str] = []
    monkeypatch.setattr(
        main,
        "Database",
        _make_fake_db_class(
            order, connect_error=ConnectionError("cannot reach 127.0.0.1:3306")
        ),
    )
    # Downstream builders must never run once the connect fails.
    _spy(monkeypatch, "build_image_store", order, delegate=False, result=object())
    _spy(monkeypatch, "build_video_source", order, delegate=False, result=object())
    _spy(monkeypatch, "build_gate", order, delegate=False, result=object())
    _spy(monkeypatch, "build_pipeline", order, delegate=False, result=object())

    pyqt_before = {m for m in sys.modules if m.startswith("PyQt5")}

    with pytest.raises(StartupError):
        main.build_application(load_weights=False)

    # Connect was attempted, but table-check and every downstream step were not.
    assert order == ["connect"]
    # No dashboard / GUI is imported by the non-GUI composition root.
    pyqt_after = {m for m in sys.modules if m.startswith("PyQt5")}
    assert pyqt_after == pyqt_before


# ---------------------------------------------------------------------------
# Failure mode: missing required table (Req 1.5)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_missing_table_raises_startup_error(monkeypatch):
    """A ``MissingTableError`` from ``check_required_tables`` halts startup as a
    ``StartupError`` (Req 1.5)."""
    order: list[str] = []
    missing = MissingTableError("Required MySQL table 'event_log' is absent.")
    monkeypatch.setattr(
        main, "Database", _make_fake_db_class(order, table_error=missing)
    )
    _spy(monkeypatch, "build_image_store", order, delegate=False, result=object())
    _spy(monkeypatch, "build_pipeline", order, delegate=False, result=object())

    with pytest.raises(StartupError) as exc_info:
        main.build_application(load_weights=False)

    # Connected, then the table check failed; nothing downstream was built.
    assert order == ["connect"]
    assert "event_log" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Failure mode: any step failing halts every subsequent step (Req 9.8)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
@pytest.mark.parametrize("failing_index", range(len(ORDERED_BUILDERS)))
def test_step_failure_halts_all_subsequent_steps(monkeypatch, failing_index):
    """Whichever ordered builder raises, no later builder is invoked (Req 9.8)."""
    order: list[str] = []
    # Inject a working fake DB in case build_repositories is delegated; here we
    # replace every ordered builder with a recording spy so the fail point is
    # deterministic and no real construction happens.
    fake_db = _make_fake_db_class(None)()
    fake_triple = (fake_db, object(), object())

    for index, name in enumerate(ORDERED_BUILDERS):
        result = fake_triple if name == "build_repositories" else _FAKE_RETURNS[name]
        if index == failing_index:
            _spy(
                monkeypatch,
                name,
                order,
                delegate=False,
                raises=StartupError(f"{name} failed"),
            )
        else:
            _spy(monkeypatch, name, order, delegate=False, result=result)

    with pytest.raises(StartupError):
        main.build_application(load_weights=True)

    # Every step up to and including the failing one ran; none after it did.
    assert order == ORDERED_BUILDERS[: failing_index + 1]


# ---------------------------------------------------------------------------
# Integration: assemble and tear down the non-GUI graph with no live MySQL
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_non_gui_graph_assembles_and_stops_cleanly(monkeypatch):
    """The full non-GUI object graph assembles over a fake ``Database``
    (simulation gate, webcam camera) and tears down cleanly with no live
    MySQL and no PyQt5 import."""
    monkeypatch.setattr(main, "Database", _make_fake_db_class(None))

    pyqt_before = {m for m in sys.modules if m.startswith("PyQt5")}

    # Default config selects camera.type=webcam and gate.mode=simulation; skip
    # loading the heavy detector weights.
    app = main.build_application(load_weights=False)

    # MySQL-backed repositories built over the shared (fake) connection.
    assert isinstance(app.database, RealDatabase)
    assert isinstance(app.resident_repo, MySqlResidentRepository)
    assert isinstance(app.event_log_repo, MySqlEventLogRepository)
    assert isinstance(app.image_store, DiskImageStore)
    # Config-selected concrete video/gate implementations.
    assert isinstance(app.video_source, WebcamVideoSource)
    assert isinstance(app.gate, SimulatedGate)
    # Pipeline assembled and not yet running.
    assert app.pipeline is not None
    assert app.pipeline.is_running is False

    # The non-GUI graph must be PyQt5-free.
    pyqt_after = {m for m in sys.modules if m.startswith("PyQt5")}
    assert pyqt_after == pyqt_before

    # Tear down cleanly: stopping an unstarted pipeline is a no-op, and closing
    # the (fake) database succeeds without touching a real socket.
    app.pipeline.stop()
    assert app.pipeline.is_running is False
    app.database.close()
    assert app.database.closed is True
