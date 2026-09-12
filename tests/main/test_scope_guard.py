"""Scope-guard smoke test for the MySQL monitoring migration (task 9.2).

Locks in the structural boundaries the migration establishes so a future change
cannot silently reintroduce a retired surface:

* the Admin dashboard module ``anpr/ui/admin_dashboard.py`` is gone and importing
  the ``anpr.ui`` package does not pull it back in (Req 6.2, 6.3);
* the ``anpr.direction`` package stays importable (deferred, not deleted) but the
  detection pipeline ``anpr.pipeline.pipeline`` does not import it -- entry/exit
  direction resolution is retired from the running system (Req 5.6);
* the ``ResidentRepository`` and ``EventLogRepository`` ports in
  ``anpr.core.interfaces`` expose exactly one method each -- ``find_by_plate`` and
  ``append`` -- with no whitelist-mutation or entry/exit-correlation methods
  (Req 3.1, 3.2, 4.7).

No live MySQL or PyQt5 is required: every assertion is structural.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import anpr.core.interfaces as interfaces

pytestmark = pytest.mark.smoke


def _public_members(protocol_cls: type) -> set[str]:
    """Return the declared (non-dunder) members of a Protocol class.

    ``typing.Protocol`` injects only dunder / single-underscore machinery, so
    filtering names that start with ``_`` leaves exactly the members the port
    declares.
    """
    return {name for name in vars(protocol_cls) if not name.startswith("_")}


# ---------------------------------------------------------------------------
# Admin dashboard removal (Req 6.2, 6.3)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_admin_dashboard_module_is_absent() -> None:
    """The Admin dashboard module file no longer exists on disk (Req 6.2)."""
    import anpr.ui as ui_pkg

    ui_dir = Path(ui_pkg.__file__).parent
    assert not (ui_dir / "admin_dashboard.py").exists(), (
        "anpr/ui/admin_dashboard.py must be deleted (Req 6.2)"
    )
    # importlib agrees the submodule cannot be resolved.
    assert importlib.util.find_spec("anpr.ui.admin_dashboard") is None


@pytest.mark.smoke
def test_importing_ui_package_does_not_pull_in_admin_dashboard() -> None:
    """Importing ``anpr.ui`` must not import an Admin dashboard (Req 6.3)."""
    # Drop any stale entry, then import the package fresh.
    sys.modules.pop("anpr.ui.admin_dashboard", None)
    ui_pkg = importlib.import_module("anpr.ui")
    importlib.reload(ui_pkg)

    assert "anpr.ui.admin_dashboard" not in sys.modules
    assert not hasattr(ui_pkg, "AdminDashboard")


# ---------------------------------------------------------------------------
# Direction retired from the pipeline but still importable (Req 5.6)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_direction_package_remains_importable() -> None:
    """``anpr.direction`` is deferred, not deleted, so it still imports."""
    module = importlib.import_module("anpr.direction")
    assert module is not None
    # The resolver module and factory are still present in the deferred package.
    resolver = importlib.import_module("anpr.direction.resolver")
    assert resolver is not None


@pytest.mark.smoke
def test_pipeline_source_has_no_direction_import() -> None:
    """The pipeline module source imports nothing from ``anpr.direction``."""
    import anpr.pipeline.pipeline as pipeline_mod

    source = Path(pipeline_mod.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # narrative comments may still mention direction
        assert "import anpr.direction" not in stripped
        assert "from anpr.direction" not in stripped


@pytest.mark.smoke
def test_pipeline_import_does_not_load_direction() -> None:
    """A fresh import of the pipeline must not transitively load direction.

    Run in a clean interpreter so results are independent of whatever other
    tests already imported ``anpr.direction`` into this process's
    ``sys.modules``.
    """
    code = (
        "import sys; import anpr.pipeline.pipeline; "
        "mods = [m for m in sys.modules if m == 'anpr.direction' "
        "or m.startswith('anpr.direction.')]; "
        "print(mods); "
        "sys.exit(1 if mods else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "importing anpr.pipeline.pipeline pulled in anpr.direction: "
        f"{result.stdout.strip()}"
    )


# ---------------------------------------------------------------------------
# Trimmed repository ports (Req 3.1, 3.2, 4.7)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_resident_repository_exposes_only_find_by_plate() -> None:
    """The whitelist port is read-only: only ``find_by_plate`` (Req 3.1, 3.2)."""
    members = _public_members(interfaces.ResidentRepository)
    assert members == {"find_by_plate"}, members
    for retired in ("create", "update", "delete", "list_all"):
        assert not hasattr(interfaces.ResidentRepository, retired), retired


@pytest.mark.smoke
def test_event_log_repository_exposes_only_append() -> None:
    """The event-log port is append-only: only ``append`` (Req 4.7)."""
    members = _public_members(interfaces.EventLogRepository)
    assert members == {"append"}, members
    for retired in ("find_open_entries", "close_open_entry"):
        assert not hasattr(interfaces.EventLogRepository, retired), retired
