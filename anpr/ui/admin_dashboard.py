"""PyQt5 Admin resident-management dashboard for the ANPR Autogate System.

:class:`AdminDashboard` is the administrator's surface for keeping the
``Resident_Database`` accurate (Requirement 13). It is a PyQt5 window that lets
the Admin create, read, update, and delete resident plate records:

* **Resident list** -- displays the normalized license plate value of every
  existing resident record (Req 13.2), refreshable on demand.
* **Plate input + actions** -- a plate-number input (bounded to the configured
  maximum length) and Create / Update / Delete controls (Req 13.1, 13.3, 13.4).

The correctness-bearing behaviour lives in a *pure* helper class,
:class:`ResidentCrudService`, that has no Qt dependency so it can be exercised
without a running GUI (Properties 28-31):

* On create/update the ``PlateNormalizer`` is applied **before any rejection
  check and before persisting** (Req 13.5).
* A submission whose normalized plate duplicates an existing record reports the
  conflict but is still allowed to proceed (Req 13.6).
* A submission whose normalized plate is marked format-invalid is rejected,
  reported as non-conforming, and never persisted (Req 13.7).
* An update or delete of a non-existent record reports not-found and leaves the
  database unchanged (Req 13.8).
* A successful create/update/delete is confirmed to the Admin (Req 13.9).

Loose coupling (Requirement 14.4): the dashboard and service depend only on the
``ResidentRepository`` Protocol in ``anpr.core.interfaces`` and on the pure
``PlateNormalizer`` -- never on a concrete implementation -- so the SQLite
repository and any mock are structurally interchangeable and chosen only in the
composition root.

The heavy PyQt5 import is guarded (mirroring ``guard_dashboard`` and the
``numpy``/``cv2``/``yaml`` fallbacks used elsewhere) so this module stays
importable and byte-compilable before PyQt5 is installed; :class:`AdminDashboard`
raises a clear error at construction time when PyQt5 is unavailable. The GUI
tests (``pytest-qt``) are a separate task.

See .kiro/specs/anpr-autogate-system/design.md (Admin_Dashboard section) and
requirements.md (Requirement 13) for the authoritative acceptance criteria.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from anpr.core.interfaces import ConfigProvider, ResidentRepository
from anpr.core.models import ResidentRecord
from anpr.core.normalizer import PlateNormalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded PyQt5 import (kept importable before the GUI dependency is installed)
# ---------------------------------------------------------------------------
try:  # PyQt5 is a pinned runtime dependency; guard so the module stays
    # importable/byte-compilable pre-install (matches the codebase pattern).
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    _PYQT5_AVAILABLE = True
    _PYQT5_IMPORT_ERROR: Optional[BaseException] = None
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only pre-install
    _PYQT5_AVAILABLE = False
    _PYQT5_IMPORT_ERROR = exc
    # Fallback so the ``AdminDashboard`` class body (which subclasses the main
    # window) can still be defined and the module imported. Construction guards
    # against the missing dependency and raises a clear error.
    QMainWindow = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Configuration keys and defaults (design Configuration schema: ui.admin.*)
# ---------------------------------------------------------------------------

# Maximum accepted plate-input length in characters. The Plate_Normalizer marks
# anything longer than 12 alphanumeric characters as format-invalid (Req 4.2),
# so this bounds the raw input widget to the same ceiling by default.
DEFAULT_PLATE_MAX_LEN = 12


@dataclass(frozen=True)
class AdminConfig:
    """Resolved ``ui.admin.*`` limit settings for the dashboard."""

    plate_max_len: int = DEFAULT_PLATE_MAX_LEN

    @classmethod
    def from_config(cls, config: Optional[ConfigProvider]) -> "AdminConfig":
        """Build from a ``ConfigProvider``, applying documented defaults.

        Absent keys fall back to their defaults so the dashboard is usable even
        with a partial configuration.
        """
        if config is None:
            return cls()
        return cls(
            plate_max_len=int(
                _cfg(config, "ui.admin.plate_max_len", DEFAULT_PLATE_MAX_LEN)
            ),
        )


def _cfg(config: ConfigProvider, key: str, default: Any) -> Any:
    """Read a dotted config key, falling back to ``default`` when absent/None."""
    try:
        value = config.get(key)
    except KeyError:
        return default
    return default if value is None else value


# ---------------------------------------------------------------------------
# Resident CRUD service (pure; no Qt dependency) -- Req 13.5-13.9 / Prop 28-31
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidentCrudResult:
    """Outcome of an Admin create/update/delete operation.

    ``success`` reports whether the operation completed and (for create/update)
    persisted a value (Req 13.9). The flags describe *why* an operation did or
    did not persist so the dashboard can present the right message to the Admin:

    * ``format_invalid`` -- the normalized plate did not conform to the
      Indonesian_Plate_Format; the submission was rejected and nothing was
      persisted (Req 13.7).
    * ``duplicate`` -- the normalized plate duplicates an existing record; the
      conflict is reported but the submission still proceeded (Req 13.6).
    * ``not_found`` -- an update/delete targeted a record that does not exist;
      the database was left unchanged (Req 13.8).

    ``normalized_plate`` is the ``PlateNormalizer`` output for the submitted
    value (empty for delete). ``record`` is the persisted record on success.
    ``detail`` is a human-readable message for the Admin.
    """

    success: bool
    detail: str
    normalized_plate: str = ""
    record: Optional[ResidentRecord] = None
    duplicate: bool = False
    format_invalid: bool = False
    not_found: bool = False


class ResidentCrudService:
    """Applies the Admin CRUD policy over a ``ResidentRepository`` (Req 13.5-13.9).

    Depends only on the ``ResidentRepository`` Protocol and the pure
    ``PlateNormalizer``, so it carries no coupling to concrete adapters
    (Req 14.4). All plate policy (normalize-before-persist, duplicate reporting,
    format rejection, not-found handling) lives here rather than in the widgets,
    keeping the correctness-bearing behaviour testable without a GUI.
    """

    def __init__(
        self,
        repository: ResidentRepository,
        normalizer: PlateNormalizer,
    ) -> None:
        self._repo = repository
        self._normalizer = normalizer

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def list_residents(self) -> list[ResidentRecord]:
        """Return every resident record for display (Req 13.2)."""
        return self._repo.list_all()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def create_resident(self, raw_plate: str) -> ResidentCrudResult:
        """Create a resident record from a raw plate value.

        Requirement 13.5: the ``PlateNormalizer`` is applied before any
        rejection check and before persisting. Requirement 13.7: a
        format-invalid plate is rejected and never persisted. Requirement 13.6:
        a normalized plate that duplicates an existing record reports the
        conflict but still proceeds. Requirement 13.9: a successful create is
        confirmed.
        """
        # Req 13.5: normalize first, before evaluating any rejection condition.
        result = self._normalizer.normalize(raw_plate if raw_plate is not None else "")
        normalized = result.normalized

        # Req 13.7: reject a format-invalid plate; do NOT persist.
        if not result.is_valid:
            return ResidentCrudResult(
                success=False,
                detail=_nonconforming_detail(result.reason),
                normalized_plate=normalized,
                format_invalid=True,
            )

        # Req 13.6: report a duplicate conflict but allow the submission.
        duplicate = self._has_plate(normalized)

        record = self._repo.create(normalized)

        detail = f"Resident '{normalized}' created."
        if duplicate:
            detail += (
                " Note: this plate duplicates an existing record; "
                "the record was created anyway."
            )
        return ResidentCrudResult(
            success=True,
            detail=detail,
            normalized_plate=normalized,
            record=record,
            duplicate=duplicate,
        )

    def update_resident(self, record_id: str, raw_plate: str) -> ResidentCrudResult:
        """Update an existing resident record's plate value.

        Applies the same normalize-before-persist (Req 13.5), format-rejection
        (Req 13.7), and duplicate-reporting (Req 13.6) policy as create. An
        update targeting a record that does not exist reports not-found and
        leaves the database unchanged (Req 13.8); a successful update is
        confirmed (Req 13.9).
        """
        # Req 13.5: normalize first, before evaluating any rejection condition.
        result = self._normalizer.normalize(raw_plate if raw_plate is not None else "")
        normalized = result.normalized

        # Req 13.7: reject a format-invalid plate; do NOT persist.
        if not result.is_valid:
            return ResidentCrudResult(
                success=False,
                detail=_nonconforming_detail(result.reason),
                normalized_plate=normalized,
                format_invalid=True,
            )

        # Req 13.6: a duplicate is another record sharing the normalized plate.
        duplicate = self._has_plate(normalized, exclude_id=record_id)

        updated = self._repo.update(record_id, normalized)

        # Req 13.8: no such record -> report not-found, DB left unchanged.
        if updated is None:
            return ResidentCrudResult(
                success=False,
                detail=f"No resident record with id '{record_id}' was found.",
                normalized_plate=normalized,
                not_found=True,
            )

        detail = f"Resident '{normalized}' updated."
        if duplicate:
            detail += (
                " Note: this plate duplicates an existing record; "
                "the update was applied anyway."
            )
        return ResidentCrudResult(
            success=True,
            detail=detail,
            normalized_plate=normalized,
            record=updated,
            duplicate=duplicate,
        )

    def delete_resident(self, record_id: str) -> ResidentCrudResult:
        """Delete a resident record by id.

        Requirement 13.8: deleting a record that does not exist reports
        not-found and leaves the database unchanged. Requirement 13.9: a
        successful delete is confirmed.
        """
        deleted = self._repo.delete(record_id)
        if not deleted:
            return ResidentCrudResult(
                success=False,
                detail=f"No resident record with id '{record_id}' was found.",
                not_found=True,
            )
        return ResidentCrudResult(
            success=True,
            detail=f"Resident record '{record_id}' deleted.",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _has_plate(self, normalized: str, *, exclude_id: Optional[str] = None) -> bool:
        """Return True when another record already has ``normalized`` plate.

        Used to report a duplicate conflict (Req 13.6). ``exclude_id`` skips the
        record being updated so re-saving its own plate is not a self-conflict.
        """
        for record in self._repo.list_all():
            if record.normalized_plate == normalized and record.id != exclude_id:
                return True
        return False


def _nonconforming_detail(reason: Optional[str]) -> str:
    """Build the Req 13.7 non-conformance message, appending the reason."""
    message = "Plate does not conform to the Indonesian plate format; not saved."
    if reason:
        message += f" ({reason})"
    return message


# ---------------------------------------------------------------------------
# Admin dashboard (PyQt5 window)
# ---------------------------------------------------------------------------


class AdminDashboard(QMainWindow):  # type: ignore[misc,valid-type]
    """The PyQt5 resident-management window (Requirement 13).

    Parameters
    ----------
    repository:
        The ``ResidentRepository`` abstraction backing CRUD (Req 13.1-13.4).
    normalizer:
        The pure plate normalizer applied before rejection/persist (Req 13.5).
    config:
        Optional ``ConfigProvider`` supplying ``ui.admin.*`` limits; documented
        defaults are used when absent.

    The dashboard delegates all plate policy to the pure
    :class:`ResidentCrudService`; the widgets only collect input, present the
    resident list, and show the operation result to the Admin.
    """

    def __init__(
        self,
        repository: ResidentRepository,
        normalizer: PlateNormalizer,
        *,
        config: Optional[ConfigProvider] = None,
        parent: Optional[Any] = None,
    ) -> None:
        if not _PYQT5_AVAILABLE:  # pragma: no cover - exercised only pre-install
            raise RuntimeError(
                "PyQt5 is required to construct the AdminDashboard but is not "
                f"installed: {_PYQT5_IMPORT_ERROR}"
            )
        super().__init__(parent)

        self._admin_config = AdminConfig.from_config(config)
        self._service = ResidentCrudService(repository, normalizer)

        # Last operation outcome, retained for tests and status display.
        self.last_result: Optional[ResidentCrudResult] = None

        self.setWindowTitle("ANPR Autogate - Admin Dashboard")
        self._build_ui()
        self.refresh_residents()

    # ------------------------------------------------------------------
    # Read-only surface
    # ------------------------------------------------------------------
    @property
    def admin_config(self) -> AdminConfig:
        """The resolved ``ui.admin.*`` limit configuration."""
        return self._admin_config

    @property
    def service(self) -> ResidentCrudService:
        """The pure CRUD service backing the dashboard (Req 13.5-13.9)."""
        return self._service

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Assemble the resident list and the create/update/delete controls."""
        central = QWidget(self)
        root = QHBoxLayout(central)

        # --- Left panel: resident list (Req 13.2) -------------------------
        list_panel = QGroupBox("Residents", central)
        list_layout = QVBoxLayout(list_panel)
        self._resident_list = QListWidget(list_panel)
        list_layout.addWidget(self._resident_list, 1)
        self._refresh_button = QPushButton("Refresh", list_panel)
        self._refresh_button.clicked.connect(self.refresh_residents)
        list_layout.addWidget(self._refresh_button)
        root.addWidget(list_panel, 2)

        # --- Right panel: CRUD controls (Req 13.1, 13.3, 13.4) ------------
        control_panel = QGroupBox("Manage Resident", central)
        control_layout = QVBoxLayout(control_panel)

        control_layout.addWidget(QLabel("Plate value:", control_panel))
        self._plate_input = QLineEdit(control_panel)
        self._plate_input.setPlaceholderText("Enter plate value")
        # Bound the raw input to the configured maximum length.
        self._plate_input.setMaxLength(self._admin_config.plate_max_len)
        control_layout.addWidget(self._plate_input)

        self._create_button = QPushButton("Create", control_panel)
        self._create_button.clicked.connect(self._on_create_clicked)
        control_layout.addWidget(self._create_button)

        self._update_button = QPushButton("Update selected", control_panel)
        self._update_button.clicked.connect(self._on_update_clicked)
        control_layout.addWidget(self._update_button)

        self._delete_button = QPushButton("Delete selected", control_panel)
        self._delete_button.clicked.connect(self._on_delete_clicked)
        control_layout.addWidget(self._delete_button)

        self._status_label = QLabel("", control_panel)
        self._status_label.setWordWrap(True)
        control_layout.addWidget(self._status_label)

        control_layout.addStretch(1)
        root.addWidget(control_panel, 1)

        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Resident list (Req 13.2)
    # ------------------------------------------------------------------
    def refresh_residents(self) -> None:
        """Reload the resident list, showing each record's normalized plate."""
        self._resident_list.clear()
        for record in self._service.list_residents():
            item = QListWidgetItem(record.normalized_plate)
            item.setData(Qt.UserRole, record.id)
            self._resident_list.addItem(item)

    def _selected_record_id(self) -> Optional[str]:
        """Return the id of the currently selected resident, or None."""
        item = self._resident_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        return str(value) if value is not None else None

    # ------------------------------------------------------------------
    # CRUD button slots (Req 13.1, 13.3, 13.4, 13.6-13.9)
    # ------------------------------------------------------------------
    def _on_create_clicked(self) -> None:
        """Button slot: create a resident from the plate input."""
        result = self._service.create_resident(self._plate_input.text())
        self._apply_result(result, clear_input_on_success=True)

    def _on_update_clicked(self) -> None:
        """Button slot: update the selected resident with the plate input."""
        record_id = self._selected_record_id()
        if record_id is None:
            self._status_label.setText("Select a resident to update.")
            return
        result = self._service.update_resident(record_id, self._plate_input.text())
        self._apply_result(result, clear_input_on_success=True)

    def _on_delete_clicked(self) -> None:
        """Button slot: delete the selected resident."""
        record_id = self._selected_record_id()
        if record_id is None:
            self._status_label.setText("Select a resident to delete.")
            return
        result = self._service.delete_resident(record_id)
        self._apply_result(result, clear_input_on_success=False)

    def _apply_result(
        self, result: ResidentCrudResult, *, clear_input_on_success: bool
    ) -> None:
        """Present ``result`` to the Admin and refresh the list on success.

        Confirms a completed operation (Req 13.9), reports a duplicate conflict
        (Req 13.6), a non-conforming plate (Req 13.7), or a not-found record
        (Req 13.8). The list is only reloaded when the database changed.
        """
        self.last_result = result
        prefix = "" if result.success else "ERROR: "
        self._status_label.setText(f"{prefix}{result.detail}")
        if result.success:
            if clear_input_on_success:
                self._plate_input.clear()
            self.refresh_residents()
