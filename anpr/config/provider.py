"""Layered configuration loading and validation.

``ConfigProvider`` loads settings from a YAML config file and then overlays
environment variables so that, where a setting is defined in both, the
environment-variable value wins (Requirement 14.1). The resolved configuration
supplies every environment-specific setting (camera source, model weights,
storage location, MySQL connection settings, network addresses, gate mode, ...)
so switching between the local-test and field deployments is config-only with
no code changes (Requirement 14.2).

``validate`` reports every missing or invalid required setting as a
``ConfigError`` naming the offending key, letting the composition root refuse
to start the affected component (Requirement 14.5).

This concrete class structurally satisfies the ``ConfigProvider`` Protocol in
``anpr.core.interfaces``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

try:  # PyYAML is a pinned runtime dependency; guard the import so the package
    # stays importable before dependencies are installed (matches the scaffold).
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only pre-install
    yaml = None  # type: ignore[assignment]

from anpr.core.models import ConfigError

# Default config file shipped alongside this module (design Configuration
# section). Loaded when no explicit path is supplied.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.yaml")

# Environment-override convention (Requirement 14.1): keys are prefixed with
# ``ANPR_`` and use ``__`` (double underscore) as the section separator, so
# ``ANPR_PIPELINE__TARGET_FPS`` maps to the dotted key ``pipeline.target_fps``.
ENV_PREFIX = "ANPR_"
ENV_SEPARATOR = "__"

# Sentinel distinguishing "key absent" from a legitimately stored ``None``.
_MISSING = object()


class ConfigProvider:
    """Loads layered configuration (file + environment overrides).

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML config file. When omitted, the shipped
        default config file is loaded.
    env:
        Optional environment mapping (defaults to ``os.environ``). Injectable to
        keep the provider testable.
    """

    def __init__(
        self,
        config_path: Optional[str | os.PathLike[str]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Loading / merging
    # ------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        """Load the file config then overlay environment overrides."""
        base = self._load_file()
        self._apply_env_overrides(base)
        return base

    def _load_file(self) -> dict[str, Any]:
        if yaml is None:  # pragma: no cover - exercised only pre-install
            raise RuntimeError(
                "PyYAML is required to load configuration but is not installed."
            )
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")
        with open(self._config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Config file must contain a mapping at the top level: {self._config_path}"
            )
        return loaded

    def _apply_env_overrides(self, base: MutableMapping[str, Any]) -> None:
        """Overlay ANPR_-prefixed environment variables (env wins, Req 14.1)."""
        for raw_key, raw_value in self._env.items():
            if not raw_key.startswith(ENV_PREFIX):
                continue
            path_part = raw_key[len(ENV_PREFIX):]
            if not path_part:
                continue
            segments = [seg.lower() for seg in path_part.split(ENV_SEPARATOR)]
            existing = self._lookup(base, segments)
            value = _coerce(raw_value, existing)
            self._set(base, segments, value)

    @staticmethod
    def _lookup(base: Mapping[str, Any], segments: list[str]) -> Any:
        """Return the current value at ``segments`` or ``_MISSING``."""
        node: Any = base
        for segment in segments:
            if isinstance(node, Mapping) and segment in node:
                node = node[segment]
            else:
                return _MISSING
        return node

    @staticmethod
    def _set(base: MutableMapping[str, Any], segments: list[str], value: Any) -> None:
        """Set ``value`` at the nested ``segments`` path, creating dicts."""
        node: MutableMapping[str, Any] = base
        for segment in segments[:-1]:
            child = node.get(segment)
            if not isinstance(child, MutableMapping):
                child = {}
                node[segment] = child
            node = child
        node[segments[-1]] = value

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------
    def get(self, key: str) -> Any:
        """Return the resolved value for a dotted config key.

        Raises ``KeyError`` when the key is not present.
        """
        segments = key.split(".")
        value = self._lookup(self._data, segments)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def get_section(self, name: str) -> Mapping[str, Any]:
        """Return a mapping of all keys within a config section.

        Raises ``KeyError`` when the section is absent, or ``TypeError`` when
        the named key is a scalar rather than a section mapping.
        """
        value = self._lookup(self._data, name.split("."))
        if value is _MISSING:
            raise KeyError(name)
        if not isinstance(value, Mapping):
            raise TypeError(f"Config key '{name}' is not a section mapping.")
        return value

    def validate(self) -> list[ConfigError]:
        """Return validation errors; an empty list means the config is valid.

        Every missing required key and every invalid value is reported so the
        caller sees all problems at once (Requirement 14.5).
        """
        errors: list[ConfigError] = []

        def check(key: str, predicate, message: str) -> None:
            value = self._lookup(self._data, key.split("."))
            if value is _MISSING:
                errors.append(ConfigError(key=key, message="missing required setting"))
                return
            try:
                ok = predicate(value)
            except (TypeError, ValueError):
                ok = False
            if not ok:
                errors.append(ConfigError(key=key, message=message))

        # environment
        check(
            "environment.label",
            lambda v: v in {"local-test", "field"},
            "must be one of {local-test, field}",
        )
        # cameras (the key name is the direction: inbound | outbound)
        self._validate_cameras(errors)
        # pipeline settings shared by every camera
        check("pipeline.target_fps", lambda v: _is_num(v) and 1 <= v <= 60, "must be between 1 and 60")
        check(
            "pipeline.reconnect_interval_s",
            lambda v: _is_num(v) and 1 <= v <= 60,
            "must be between 1 and 60",
        )
        check(
            "pipeline.inactivity_timeout_s",
            lambda v: _is_num(v) and 1 <= v <= 30,
            "must be between 1 and 30",
        )
        check(
            "pipeline.reconciliation_window_s",
            lambda v: _is_num(v) and 1 <= v <= 30,
            "must be between 1 and 30",
        )
        # model
        check(
            "model.weights_path",
            lambda v: isinstance(v, str) and v.strip() != "",
            "must be a non-empty path",
        )
        check(
            "model.detection_threshold",
            lambda v: _is_num(v) and 0 <= v <= 1,
            "must be between 0 and 1",
        )
        # ocr
        check(
            "ocr.confidence_threshold",
            lambda v: _is_num(v) and 0 <= v <= 1,
            "must be between 0 and 1",
        )
        check("ocr.timeout_ms", lambda v: _is_num(v) and v > 0, "must be greater than 0")
        # pipeline (per-track deduplication). An integer grace window in frames;
        # 0 evicts a track as soon as it is absent from a frame.
        check(
            "pipeline.track_staleness_frames",
            lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 600,
            "must be an integer between 0 and 600",
        )
        # gate
        check("gate.mode", lambda v: v in {"simulation", "hardware"}, "must be one of {simulation, hardware}")
        check(
            "gate.hardware_response_timeout_s",
            lambda v: _is_num(v) and 1 <= v <= 30,
            "must be between 1 and 30",
        )
        # storage
        check(
            "storage.image_dir",
            lambda v: isinstance(v, str) and v.strip() != "",
            "must be a non-empty path",
        )
        check("storage.thumbnail_max_px", lambda v: _is_num(v) and v > 0, "must be greater than 0")
        check(
            "storage.retention_days",
            lambda v: _is_num(v) and 1 <= v <= 365,
            "must be between 1 and 365",
        )
        check(
            "storage.retention_interval_h",
            lambda v: _is_num(v) and 0 < v <= 24,
            "must be between 1 and 24",
        )
        # database (MySQL connection settings, Requirements 2.1-2.4). The env
        # overlay coerces ANPR_DATABASE__PORT / ANPR_DATABASE__PASSWORD toward
        # the file value's type via ``_coerce`` before validation runs here.
        check(
            "database.host",
            lambda v: isinstance(v, str) and v.strip() != "",
            "must be a non-empty string",
        )
        check(
            "database.port",
            lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 65535,
            "must be an integer between 1 and 65535",
        )
        check(
            "database.name",
            lambda v: isinstance(v, str) and v.strip() != "",
            "must be a non-empty string",
        )
        check(
            "database.user",
            lambda v: isinstance(v, str) and v.strip() != "",
            "must be a non-empty string",
        )
        check(
            "database.password",
            lambda v: isinstance(v, str),
            "must be a string",
        )
        # Optional TLS CA-bundle path. Empty string disables TLS (local Docker);
        # a non-empty path enables a verified, encrypted link to a remote
        # endpoint such as AWS RDS. Only the type is constrained here; the path's
        # existence is checked when the connection is opened.
        check(
            "database.ssl_ca",
            lambda v: isinstance(v, str),
            "must be a string",
        )
        return errors

    def _validate_cameras(self, errors: "list[ConfigError]") -> None:
        """Validate ``cameras.{inbound,outbound}``; at least one side is required.

        The key name is the direction, so a side that is absent is simply not
        used. Each configured side needs a valid ``type`` (webcam | ip), an
        integer ``id`` matching a seeded ``Camera`` row, a non-empty
        ``stream_url`` for ``ip`` and an integer ``device_index`` for ``webcam``.
        Any gap is reported naming the offending key so startup is refused
        (Req 1.5).
        """
        sides = [
            side
            for side in ("inbound", "outbound")
            if self._lookup(self._data, ["cameras", side]) is not _MISSING
        ]
        cameras = self._lookup(self._data, ["cameras"])
        if isinstance(cameras, Mapping):
            for name in cameras:
                if name not in ("inbound", "outbound"):
                    errors.append(
                        ConfigError(
                            key=f"cameras.{name}",
                            message="unknown camera; expected 'inbound' or 'outbound'",
                        )
                    )
        if not sides:
            errors.append(
                ConfigError(
                    key="cameras",
                    message="configure at least one of 'inbound' or 'outbound'",
                )
            )
            return

        for side in sides:
            base = ["cameras", side]
            key = f"cameras.{side}"
            cam_type = self._lookup(self._data, base + ["type"])
            cam_id = self._lookup(self._data, base + ["id"])

            if cam_type is _MISSING:
                errors.append(ConfigError(key=f"{key}.type", message="missing required setting"))
            elif cam_type not in {"webcam", "ip"}:
                errors.append(
                    ConfigError(key=f"{key}.type", message="must be one of {webcam, ip}")
                )

            if cam_id is _MISSING:
                errors.append(ConfigError(key=f"{key}.id", message="missing required setting"))
            elif not (isinstance(cam_id, int) and not isinstance(cam_id, bool)):
                errors.append(
                    ConfigError(key=f"{key}.id", message="must be an integer Camera.Camera_ID")
                )

            # Address requirement depends on the source type.
            if cam_type == "ip":
                addr = self._lookup(self._data, base + ["stream_url"])
                if not (isinstance(addr, str) and addr.strip()):
                    errors.append(
                        ConfigError(
                            key=f"{key}.stream_url",
                            message="must be a non-empty stream URL when type is 'ip'",
                        )
                    )
            elif cam_type == "webcam":
                index = self._lookup(self._data, base + ["device_index"])
                if index is not _MISSING and not (
                    isinstance(index, int) and not isinstance(index, bool) and index >= 0
                ):
                    errors.append(
                        ConfigError(
                            key=f"{key}.device_index",
                            message="must be a non-negative integer",
                        )
                    )


def _is_num(value: Any) -> bool:
    """True for real numbers, excluding bool (which is an int subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce(raw: str, existing: Any) -> Any:
    """Coerce an environment string to match the type of ``existing``.

    Environment values are always strings; coerce them toward the type of the
    corresponding default/file value where reasonable so numeric and boolean
    settings keep their expected types. Falls back to the raw string when no
    default exists or coercion fails.
    """
    if existing is _MISSING or existing is None:
        return raw
    if isinstance(existing, bool):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return raw
    if isinstance(existing, int):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(existing, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw
