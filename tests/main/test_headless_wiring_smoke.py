"""Headless wiring smoke test for the composition root (task 8.3).

Assembles the non-GUI object graph via :func:`anpr.main.build_application` with
an injected/monkeypatched :class:`~anpr.persistence.db.Database` (so **no live
MySQL** is required) and asserts the migration's structural guarantees:

* the MySQL-backed repositories are built from ``database.*`` and share the one
  ``Database`` connection (Req 9.1);
* the pipeline is wired to the selected collaborators -- detector, OCR engine,
  normalizer, access controller, and video source (Req 9.2);
* neither the :class:`~anpr.main.Application` graph nor the pipeline carries a
  direction resolver -- entry/exit correlation is retired (Req 9.3);
* assembling the graph pulls in **no** ``PyQt5`` module: the non-GUI builders
  stay import-clean so the wiring is headless-testable and the Guard dashboard
  is the only place Qt is imported (Req 9.4, 10.2, 10.5).

The fake DB-API stub lives in ``tests/persistence/fake_dbapi.py``; here we go one
level up and replace the ``Database`` *class* that ``anpr.main`` references with
a recording subclass, so ``build_repositories`` neither opens a socket nor runs
a table check against a real server, while remaining an ``isinstance`` of the
real ``Database`` (which the image store relies on).
"""

from __future__ import annotations

import dataclasses
import sys

import pytest

import anpr.main as main_mod
from anpr.core.access_controller import AccessController
from anpr.core.normalizer import PlateNormalizer
from anpr.detection.detector import YoloVehicleDetector
from anpr.gate.controller import SimulatedGate
from anpr.persistence.db import Database
from anpr.persistence.event_log_repo import MySqlEventLogRepository
from anpr.persistence.resident_repo import MySqlResidentRepository
from anpr.pipeline.pipeline import DetectionPipeline
from anpr.pipeline.video_source import WebcamVideoSource

pytestmark = pytest.mark.smoke


# Connection settings pushed through the environment overlay so the assertions
# can prove the repositories are built from ``database.*`` and not hardcoded
# defaults (Req 9.1). Non-default values on every key make the check meaningful.
_ENV = {
    "ANPR_CAMERA__TYPE": "webcam",
    "ANPR_GATE__MODE": "simulation",
    "ANPR_DATABASE__HOST": "203.0.113.7",
    "ANPR_DATABASE__PORT": "13306",
    "ANPR_DATABASE__NAME": "anpr_smoke",
    "ANPR_DATABASE__USER": "anpr_smoke_user",
    "ANPR_DATABASE__PASSWORD": "s3cr3t-smoke",
}

_EXPECTED_PARAMS = {
    "host": "203.0.113.7",
    "port": 13306,
    "name": "anpr_smoke",
    "user": "anpr_smoke_user",
    "password": "s3cr3t-smoke",
}


class _RecordingDatabase(Database):
    """Drop-in for :class:`Database` that records params and skips MySQL.

    Subclasses the real ``Database`` so it satisfies the ``isinstance`` check the
    image store performs, but overrides ``__init__`` to avoid opening a socket
    and ``check_required_tables``/``query``/``execute`` to avoid touching a live
    server. It captures the connection settings passed by ``build_repositories``
    so the test can assert they came from ``database.*`` (Req 9.1).
    """

    instances: "list[_RecordingDatabase]" = []

    def __init__(self, host, port, name, user, password, connect_factory=None):
        # Intentionally do not call super().__init__ (which would connect).
        self.connect_params = {
            "host": host,
            "port": int(port),
            "name": name,
            "user": user,
            "password": password,
        }
        self.tables_checked = False
        _RecordingDatabase.instances.append(self)

    @property
    def name(self) -> str:  # mirrors Database.name without _params
        return self.connect_params["name"]

    def check_required_tables(self) -> None:
        self.tables_checked = True

    def query(self, sql, params=()):  # never reached in this smoke test
        return []

    def execute(self, sql, params=()):  # never reached in this smoke test
        return 0

    def close(self) -> None:
        pass


def _pyqt5_modules() -> set[str]:
    """Return the currently-loaded module names belonging to PyQt5."""
    return {
        name
        for name in sys.modules
        if name == "PyQt5" or name.startswith("PyQt5.")
    }


@pytest.fixture()
def application(monkeypatch):
    """Build the non-GUI graph with the recording DB and no live MySQL."""
    _RecordingDatabase.instances.clear()
    monkeypatch.setattr(main_mod, "Database", _RecordingDatabase)

    # Snapshot any PyQt5 modules already loaded by earlier tests so we can prove
    # build_application itself introduces none (Req 9.4, 10.2, 10.5).
    pyqt_before = _pyqt5_modules()

    app = main_mod.build_application(env=_ENV, load_weights=False)
    return app, pyqt_before


def test_repositories_built_from_database_config(application):
    """MySQL-backed repos share the one Database built from ``database.*`` (Req 9.1)."""
    app, _ = application

    # Exactly one Database was constructed, from the configured connection keys.
    assert len(_RecordingDatabase.instances) == 1
    db = _RecordingDatabase.instances[0]
    assert app.database is db
    assert db.connect_params == _EXPECTED_PARAMS
    # Required-table verification ran before wiring proceeded (Req 1.5, 9.6).
    assert db.tables_checked is True

    # The repositories are the MySQL adapters, sharing the single connection.
    assert isinstance(app.resident_repo, MySqlResidentRepository)
    assert isinstance(app.event_log_repo, MySqlEventLogRepository)
    assert app.resident_repo._db is db
    assert app.event_log_repo._db is db
    # The image store rides the same shared Database instance too.
    assert app.image_store._db is db


def test_pipeline_collaborators_wired(application):
    """The pipeline is wired to the selected collaborators (Req 9.2)."""
    app, _ = application

    assert isinstance(app.pipeline, DetectionPipeline)
    # Concrete selections driven by config.
    assert isinstance(app.video_source, WebcamVideoSource)
    assert isinstance(app.gate, SimulatedGate)
    assert isinstance(app.detector, YoloVehicleDetector)
    assert isinstance(app.normalizer, PlateNormalizer)
    assert isinstance(app.access_controller, AccessController)

    # Each collaborator on the Application is the very object the pipeline holds.
    assert app.pipeline._source is app.video_source
    assert app.pipeline._detector is app.detector
    assert app.pipeline._ocr is app.ocr_engine
    assert app.pipeline._normalizer is app.normalizer
    assert app.pipeline._access is app.access_controller

    # The access controller is wired to the same repos + gate.
    assert app.access_controller._resident_repo is app.resident_repo
    assert app.access_controller._event_log_repo is app.event_log_repo
    assert app.access_controller._gate_controller is app.gate


def test_no_direction_resolver_on_application_or_pipeline(application):
    """Direction resolution is retired: none on Application or pipeline (Req 9.3)."""
    app, _ = application

    field_names = {f.name for f in dataclasses.fields(main_mod.Application)}
    assert "direction_resolver" not in field_names
    assert not hasattr(app, "direction_resolver")

    # The pipeline neither stores a resolver nor keeps a direction seam.
    assert not hasattr(app.pipeline, "_direction")
    assert not hasattr(app.pipeline, "direction_resolver")

    # The composition root no longer exposes a direction builder.
    assert not hasattr(main_mod, "build_direction_resolver")


def test_assembly_imports_no_pyqt5(application):
    """Assembling the non-GUI graph pulls in no PyQt5 module (Req 9.4, 10.2, 10.5)."""
    _, pyqt_before = application

    pyqt_after = _pyqt5_modules()
    # build_application must not have imported any PyQt5 module. Compared as a
    # delta so an earlier GUI test that already loaded PyQt5 cannot mask a
    # regression here (the headless graph itself stays Qt-free).
    introduced = pyqt_after - pyqt_before
    assert introduced == set(), (
        f"build_application imported PyQt5 module(s): {sorted(introduced)}"
    )
