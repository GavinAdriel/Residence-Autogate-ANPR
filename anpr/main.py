"""Composition root for the ANPR Autogate System.

This is the single place where configuration is read and concrete component
implementations are selected and wired together (Requirement 14.2, 14.4). No
core component imports a concrete peer implementation: every concrete choice --
which :class:`VideoSource`, which :class:`GateController`, which
:class:`DirectionResolver`, the SQLite repositories, and the image store -- is
made here from configuration values alone.

Startup ordering (see design.md "Architecture" and Requirements 2.1, 14.5):

1. Load configuration through the :class:`~anpr.config.provider.ConfigProvider`
   and run :meth:`ConfigProvider.validate`. Every reported
   :class:`~anpr.core.models.ConfigError` is logged naming the offending
   setting, and startup is refused so an affected component never starts with a
   missing/invalid required value (Req 14.5).
2. Build the persistence layer (a shared SQLite :class:`Database` plus the
   resident and event-log repositories) from ``database.location``.
3. Build the image store from ``storage.*``.
4. Select the video source from ``camera.type`` (Req 1.2, 1.3).
5. Select the gate from ``gate.mode`` (Req 6.4).
6. Select the direction resolver from ``direction.mode`` (Req 7.1).
7. Load the detector weights *before* the first frame / before the pipeline
   starts; on a weights-load failure log the offending path and halt pipeline
   startup (Req 2.1, 2.5).
8. Construct the :class:`~anpr.pipeline.pipeline.DetectionPipeline`, wiring the
   detector, OCR engine, normalizer, direction resolver, access controller,
   repositories, gate, and image store.
9. Launch the Guard and Admin dashboards and start the ingest loop.

Offline by default (Req 14.3): the composition root performs no outbound cloud
or internet connection; the only network egress is the optional IP-camera
stream selected via ``camera.type = ip``.

The non-GUI builders and :func:`build_application` are deliberately free of any
PyQt5 dependency so the wiring can be exercised headlessly; the dashboards are
constructed only inside :func:`main`, behind the same guarded PyQt5 import used
elsewhere in the codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from anpr.config.provider import ConfigProvider
from anpr.core.access_controller import AccessController
from anpr.core.interfaces import (
    DirectionResolver,
    EventLogRepository,
    GateController,
    ImageStore,
    OcrEngine,
    ResidentRepository,
    VideoSource,
)
from anpr.core.models import ConfigError, EnvironmentLabel
from anpr.core.normalizer import PlateNormalizer
from anpr.detection.detector import (
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_WEIGHTS_PATH,
    WeightsLoadError,
    YoloVehicleDetector,
)
from anpr.direction.resolver import create_direction_resolver
from anpr.gate.controller import (
    DEFAULT_RESPONSE_TIMEOUT_S,
    HardwareGate,
    HardwareInterface,
    SimulatedGate,
)
from anpr.imaging.store import DiskImageStore, create_image_store
from anpr.ocr.engine import PaddleOcrEngine
from anpr.persistence.db import Database
from anpr.persistence.event_log_repo import SqliteEventLogRepository
from anpr.persistence.resident_repo import SqliteResidentRepository
from anpr.pipeline.pipeline import DetectionPipeline
from anpr.pipeline.video_source import IpCameraVideoSource, WebcamVideoSource

logger = logging.getLogger(__name__)

# Config selector values (mirrors the schema in default_config.yaml).
CAMERA_WEBCAM = "webcam"
CAMERA_IP = "ip"
GATE_SIMULATION = "simulation"
GATE_HARDWARE = "hardware"


class StartupError(RuntimeError):
    """Raised when the system cannot start with the resolved configuration.

    Carries the human-readable reason (e.g. the list of configuration errors or
    an unsupported selector value) so the entrypoint can log it and exit without
    starting a partially-wired system (Req 14.5).
    """


def _cfg(config: Any, key: str, default: Any) -> Any:
    """Read a dotted config key, falling back to ``default`` when absent/None."""
    try:
        value = config.get(key)
    except (KeyError, AttributeError):
        return default
    return default if value is None else value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def create_config_provider(
    config_path: Optional[str] = None,
    env: Optional[Any] = None,
) -> ConfigProvider:
    """Construct the :class:`ConfigProvider` (file + environment overrides)."""
    return ConfigProvider(config_path=config_path, env=env)


def validate_config_or_halt(config: ConfigProvider) -> None:
    """Validate configuration and refuse to start on any error (Req 14.5).

    Every :class:`ConfigError` is logged naming the offending setting; if any
    error is present a :class:`StartupError` is raised so no affected component
    starts with a missing or invalid required value.
    """
    errors: list[ConfigError] = config.validate()
    if not errors:
        return
    for error in errors:
        logger.error(
            "Configuration error for '%s': %s", error.key, error.message
        )
    summary = "; ".join(f"{e.key}: {e.message}" for e in errors)
    raise StartupError(
        f"Refusing to start: {len(errors)} configuration error(s): {summary}"
    )


def resolve_environment_label(config: ConfigProvider) -> Optional[EnvironmentLabel]:
    """Resolve ``environment.label`` into an :class:`EnvironmentLabel` (Req 15.2)."""
    label = _cfg(config, "environment.label", None)
    if label is None:
        return None
    try:
        return EnvironmentLabel(label)
    except ValueError:
        logger.warning("Unknown environment.label %r; leaving unset.", label)
        return None


# ---------------------------------------------------------------------------
# Concrete-implementation selection (purely from config values, Req 14.2, 14.4)
# ---------------------------------------------------------------------------
def build_video_source(config: ConfigProvider) -> VideoSource:
    """Select the video source from ``camera.type`` (Req 1.2, 1.3, 14.4).

    ``webcam`` -> :class:`WebcamVideoSource` addressed by ``camera.device_index``;
    ``ip`` -> :class:`IpCameraVideoSource` addressed by ``camera.stream_url``.
    """
    camera_type = _cfg(config, "camera.type", CAMERA_WEBCAM)
    if camera_type == CAMERA_WEBCAM:
        device_index = int(_cfg(config, "camera.device_index", 0))
        logger.info("Selecting WebcamVideoSource (device_index=%d).", device_index)
        return WebcamVideoSource(device_index)
    if camera_type == CAMERA_IP:
        stream_url = str(_cfg(config, "camera.stream_url", ""))
        logger.info("Selecting IpCameraVideoSource (stream_url=%r).", stream_url)
        return IpCameraVideoSource(stream_url)
    raise StartupError(
        f"Unsupported camera.type {camera_type!r}; expected "
        f"{CAMERA_WEBCAM!r} or {CAMERA_IP!r}."
    )


def build_gate(
    config: ConfigProvider,
    *,
    hardware_interface: Optional[HardwareInterface] = None,
) -> GateController:
    """Select the gate controller from ``gate.mode`` (Req 6.4, 14.4).

    ``simulation`` -> :class:`SimulatedGate`; ``hardware`` -> :class:`HardwareGate`
    driven by an injected :class:`HardwareInterface` (the field relay/PLC driver).
    The hardware interface is not shipped in this phase, so ``hardware`` mode
    requires the caller to supply the driver; otherwise gate startup is halted
    with a clear error (Req 14.5).
    """
    mode = _cfg(config, "gate.mode", GATE_SIMULATION)
    if mode == GATE_SIMULATION:
        logger.info("Selecting SimulatedGate (gate.mode=simulation).")
        return SimulatedGate()
    if mode == GATE_HARDWARE:
        if hardware_interface is None:
            raise StartupError(
                "gate.mode 'hardware' requires a hardware interface driver "
                "(relay/PLC), which is not provided in this phase; supply one "
                "via build_application(hardware_interface=...) or use "
                "gate.mode 'simulation'."
            )
        timeout = _cfg(
            config, "gate.hardware_response_timeout_s", DEFAULT_RESPONSE_TIMEOUT_S
        )
        logger.info(
            "Selecting HardwareGate (gate.mode=hardware, response_timeout_s=%s).",
            timeout,
        )
        return HardwareGate(hardware_interface, response_timeout_s=timeout)
    raise StartupError(
        f"Unsupported gate.mode {mode!r}; expected "
        f"{GATE_SIMULATION!r} or {GATE_HARDWARE!r}."
    )


def build_direction_resolver(config: ConfigProvider) -> DirectionResolver:
    """Select the direction resolver from ``direction.mode`` (Req 7.1, 14.4)."""
    try:
        return create_direction_resolver(config)
    except ValueError as exc:
        # create_direction_resolver raises ValueError on an unsupported mode.
        raise StartupError(str(exc)) from exc


def build_repositories(
    config: ConfigProvider,
) -> tuple[Database, ResidentRepository, EventLogRepository]:
    """Build the shared SQLite database and its repositories (Req 14.2).

    A single :class:`Database` connection (from ``database.location``) is shared
    by both repositories so they use the same schema and connection. Persistence
    is a local SQLite file -- no network access (Req 14.3).
    """
    location = str(_cfg(config, "database.location", "./anpr.db"))
    logger.info("Opening SQLite database at %r.", location)
    database = Database(location)
    resident_repo = SqliteResidentRepository(database)
    event_log_repo = SqliteEventLogRepository(database)
    return database, resident_repo, event_log_repo


def build_image_store(config: ConfigProvider, database: Database) -> DiskImageStore:
    """Build the on-disk image store from ``storage.*`` (Req 11)."""
    return create_image_store(config, database)


def build_detector(config: ConfigProvider) -> YoloVehicleDetector:
    """Build the vehicle detector from ``model.*`` (weights not yet loaded)."""
    return YoloVehicleDetector(
        weights_path=str(_cfg(config, "model.weights_path", DEFAULT_WEIGHTS_PATH)),
        detection_threshold=_cfg(
            config, "model.detection_threshold", DEFAULT_DETECTION_THRESHOLD
        ),
    )


def build_ocr_engine(config: ConfigProvider) -> OcrEngine:
    """Build the OCR engine (PaddleOCR wrapper; lazily loaded on first use)."""
    return PaddleOcrEngine()


def load_detector_weights_or_halt(detector: YoloVehicleDetector) -> None:
    """Load detector weights before the first frame, halting on failure.

    Requirement 2.1: weights are loaded at startup, before any frame is
    processed. Requirement 2.5: on a weights-load failure the offending path is
    logged and pipeline startup is halted (surfaced here as a
    :class:`StartupError`).
    """
    try:
        detector.load()
    except WeightsLoadError as exc:
        logger.error(
            "Halting pipeline startup: could not load detector weights from "
            "'%s': %s",
            exc.weights_path,
            exc,
        )
        raise StartupError(
            f"Detector weights could not be loaded from '{exc.weights_path}'."
        ) from exc


def build_access_controller(
    resident_repo: ResidentRepository,
    event_log_repo: EventLogRepository,
    gate: GateController,
    environment_label: Optional[EnvironmentLabel],
) -> AccessController:
    """Wire the pure access-control orchestrator over its collaborators."""
    return AccessController(
        resident_repo,
        event_log_repo,
        gate,
        environment_label=environment_label,
    )


def build_pipeline(
    config: ConfigProvider,
    video_source: VideoSource,
    detector: YoloVehicleDetector,
    ocr_engine: OcrEngine,
    normalizer: PlateNormalizer,
    direction_resolver: DirectionResolver,
    access_controller: AccessController,
    *,
    on_event: Optional[Callable[[Any], None]] = None,
    on_manual_review: Optional[Callable[[Any], None]] = None,
) -> DetectionPipeline:
    """Construct the :class:`DetectionPipeline` from config and collaborators.

    Camera/OCR tuning values are read from configuration by
    :meth:`DetectionPipeline.from_config`; the live-feed and manual-review
    callbacks (wired to the Guard_Dashboard by :func:`main`) are forwarded here.
    """
    return DetectionPipeline.from_config(
        config,
        video_source,
        detector,
        ocr_engine,
        normalizer,
        direction_resolver,
        access_controller,
        on_event=on_event,
        on_manual_review=on_manual_review,
    )


# ---------------------------------------------------------------------------
# Application assembly (non-GUI; headless-testable)
# ---------------------------------------------------------------------------
@dataclass
class Application:
    """The fully-wired, non-GUI object graph produced by the composition root.

    Holds every selected concrete implementation plus the assembled pipeline so
    the entrypoint (or a headless test) can launch the dashboards and start the
    ingest loop. GUI widgets are intentionally *not* part of this graph.
    """

    config: ConfigProvider
    environment_label: Optional[EnvironmentLabel]
    database: Database
    resident_repo: ResidentRepository
    event_log_repo: EventLogRepository
    image_store: DiskImageStore
    video_source: VideoSource
    gate: GateController
    direction_resolver: DirectionResolver
    normalizer: PlateNormalizer
    detector: YoloVehicleDetector
    ocr_engine: OcrEngine
    access_controller: AccessController
    pipeline: DetectionPipeline


def build_application(
    config_path: Optional[str] = None,
    *,
    env: Optional[Any] = None,
    load_weights: bool = True,
    hardware_interface: Optional[HardwareInterface] = None,
    on_event: Optional[Callable[[Any], None]] = None,
    on_manual_review: Optional[Callable[[Any], None]] = None,
) -> Application:
    """Read configuration and wire the whole non-GUI object graph in order.

    Runs the startup ordering documented in the module docstring: validate
    configuration (Req 14.5), build persistence/imaging, select the video
    source/gate/direction resolver purely from config (Req 14.2, 14.4), load the
    detector weights before the pipeline is constructed (Req 2.1), then assemble
    the pipeline. Raises :class:`StartupError` when configuration is invalid, a
    selector value is unsupported, or the detector weights cannot be loaded.

    ``load_weights`` may be set ``False`` for headless wiring tests that inject a
    detector model factory of their own.
    """
    config = create_config_provider(config_path, env=env)

    # Step 1: refuse to start on any configuration error (Req 14.5).
    validate_config_or_halt(config)
    environment_label = resolve_environment_label(config)

    # Step 2-3: persistence + imaging (local SQLite/disk only, Req 14.3).
    database, resident_repo, event_log_repo = build_repositories(config)
    image_store = build_image_store(config, database)

    # Step 4-6: concrete selection purely from config (Req 14.2, 14.4).
    video_source = build_video_source(config)
    gate = build_gate(config, hardware_interface=hardware_interface)
    direction_resolver = build_direction_resolver(config)

    # Pure collaborators.
    normalizer = PlateNormalizer()
    ocr_engine = build_ocr_engine(config)
    access_controller = build_access_controller(
        resident_repo, event_log_repo, gate, environment_label
    )

    # Step 7: load detector weights BEFORE constructing/starting the pipeline
    # so the first frame is never processed against unloaded weights (Req 2.1).
    detector = build_detector(config)
    if load_weights:
        load_detector_weights_or_halt(detector)

    # Step 8: assemble the pipeline over the abstractions.
    pipeline = build_pipeline(
        config,
        video_source,
        detector,
        ocr_engine,
        normalizer,
        direction_resolver,
        access_controller,
        on_event=on_event,
        on_manual_review=on_manual_review,
    )

    return Application(
        config=config,
        environment_label=environment_label,
        database=database,
        resident_repo=resident_repo,
        event_log_repo=event_log_repo,
        image_store=image_store,
        video_source=video_source,
        gate=gate,
        direction_resolver=direction_resolver,
        normalizer=normalizer,
        detector=detector,
        ocr_engine=ocr_engine,
        access_controller=access_controller,
        pipeline=pipeline,
    )


# ---------------------------------------------------------------------------
# Entrypoint (GUI): launches the Guard and Admin dashboards + ingest loop
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    """Configure basic console logging when the app owns the root logger."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def main(
    config_path: Optional[str] = None,
    *,
    hardware_interface: Optional[HardwareInterface] = None,
) -> int:
    """Compose the system, launch the dashboards, and run the ingest loop.

    Wires the non-GUI object graph in the documented startup order, constructs
    the Guard and Admin dashboards behind a guarded PyQt5 import, marshals the
    pipeline's live-feed and manual-review callbacks onto the GUI thread via a
    Qt signal bridge, starts the :class:`DetectionPipeline` on a background
    thread, and runs the Qt event loop. Returns the Qt exit code.

    Raises :class:`StartupError` when configuration is invalid, a selector value
    is unsupported, the detector weights cannot be loaded, or PyQt5 is not
    installed.
    """
    import threading

    _configure_logging()

    # On Windows, importing PyQt5 before torch makes torch's native DLLs
    # (c10.dll and its dependencies) fail to initialize with WinError 1114,
    # because Qt loads conflicting runtime DLLs first. Pre-import torch here so
    # its DLLs load before Qt's; the detector (via ultralytics) needs torch
    # anyway. Guarded so the entrypoint still runs if torch is absent.
    try:
        import torch  # noqa: F401  (pre-imported for Windows DLL load ordering)
    except ImportError:
        pass

    # --- Guarded PyQt5 import (matches the codebase's pre-install pattern) ---
    try:
        from PyQt5.QtCore import QObject, pyqtSignal
        from PyQt5.QtWidgets import QApplication
    except ModuleNotFoundError as exc:  # pragma: no cover - only pre-install
        raise StartupError(
            "PyQt5 is required to launch the Guard and Admin dashboards but is "
            f"not installed: {exc}"
        ) from exc

    from anpr.ui.admin_dashboard import AdminDashboard
    from anpr.ui.guard_dashboard import GuardDashboard

    # --- Configuration + non-GUI components (Req 14.2, 14.4, 14.5) ----------
    config = create_config_provider(config_path)
    validate_config_or_halt(config)
    environment_label = resolve_environment_label(config)

    database, resident_repo, event_log_repo = build_repositories(config)
    image_store = build_image_store(config, database)
    video_source = build_video_source(config)
    gate = build_gate(config, hardware_interface=hardware_interface)
    direction_resolver = build_direction_resolver(config)
    normalizer = PlateNormalizer()
    ocr_engine = build_ocr_engine(config)
    access_controller = build_access_controller(
        resident_repo, event_log_repo, gate, environment_label
    )

    # Load detector weights BEFORE the pipeline starts (Req 2.1, 2.5).
    detector = build_detector(config)
    load_detector_weights_or_halt(detector)

    # --- GUI: dashboards + cross-thread callback bridge ---------------------
    app = QApplication.instance() or QApplication([])

    class _EventBridge(QObject):
        """Marshals pipeline callbacks (background thread) onto the GUI thread.

        Emitting a signal from the pipeline thread to a slot owned by the GUI
        thread queues the call safely (Qt AutoConnection), so the dashboards are
        only ever touched on the GUI thread.
        """

        detection = pyqtSignal(object)
        review = pyqtSignal(object)

    bridge = _EventBridge()

    # Assemble the pipeline with the bridge-backed callbacks (Req 12.2, 12.7).
    # Built before the dashboards so its live-feed accessor can be wired into
    # the Guard_Dashboard as the frame provider.
    pipeline = build_pipeline(
        config,
        video_source,
        detector,
        ocr_engine,
        normalizer,
        direction_resolver,
        access_controller,
        on_event=bridge.detection.emit,
        on_manual_review=bridge.review.emit,
    )

    guard_dashboard = GuardDashboard(
        gate,
        event_log_repo,
        normalizer,
        config=config,
        image_store=image_store,
        environment_label=environment_label,
        frame_provider=pipeline.latest_feed,
    )
    admin_dashboard = AdminDashboard(resident_repo, normalizer, config=config)

    bridge.detection.connect(guard_dashboard.on_detection)
    bridge.review.connect(
        lambda ev: guard_dashboard.surface_event(
            ev, getattr(ev, "manual_review_reason", None)
        )
    )

    guard_dashboard.show()
    admin_dashboard.show()

    # Start polling the pipeline's latest frame so the live feed renders
    # continuously at >= the configured min fps (Req 12.1).
    guard_dashboard.start_feed()

    # Run the blocking ingest loop on a background daemon thread so the Qt event
    # loop stays responsive; stop it cleanly when the app quits.
    pipeline_thread = threading.Thread(
        target=pipeline.start, name="detection-pipeline", daemon=True
    )
    app.aboutToQuit.connect(pipeline.stop)
    pipeline_thread.start()

    logger.info("ANPR Autogate System started; entering Qt event loop.")
    exit_code = app.exec_()
    pipeline.stop()
    pipeline_thread.join(timeout=5.0)
    return exit_code


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except StartupError as error:
        logging.getLogger(__name__).error("Startup aborted: %s", error)
        sys.exit(1)
