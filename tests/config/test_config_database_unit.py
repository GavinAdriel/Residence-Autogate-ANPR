"""Unit tests for the migrated MySQL connection configuration.

Feature: mysql-monitoring-migration

These example-based tests lock in the config-layer contract introduced by the
SQLite -> MySQL migration:

* Requirement 2.1 -- ``default_config.yaml`` exposes the MySQL connection keys
  ``database.{host,port,name,user,password}`` and no longer supplies the old
  ``database.location`` key.
* Requirement 2.6 -- the configuration defines no ``direction`` section, and
  ``ConfigProvider.validate()`` reports no ``direction.*`` error.
* Requirement 2.7 -- when one or more MySQL connection values are invalid,
  ``validate()`` names each offending setting, and determining that verdict
  opens no database connection (startup halts before any connection is opened).

Every test runs to completion without a live MySQL server (Req 10.3): the
configuration layer never connects to a database, and the one connection-boundary
assertion spies on the driver rather than requiring one to be reachable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yaml")

import copy
import os
import tempfile

import yaml

from anpr.config.provider import DEFAULT_CONFIG_PATH, ConfigProvider

# The five MySQL connection settings the migration must supply (Req 2.1).
_CONNECTION_KEYS = ("host", "port", "name", "user", "password")


def _base_config() -> dict:
    """Load the shipped default config as a valid baseline to mutate."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_temp_config(config: dict) -> str:
    """Serialize ``config`` to a temp YAML file and return its path."""
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", encoding="utf-8", delete=False
    )
    yaml.safe_dump(config, handle, allow_unicode=True)
    handle.close()
    return handle.name


# ---------------------------------------------------------------------------
# Requirement 2.1 -- database connection keys present, database.location absent
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_default_config_exposes_mysql_connection_keys() -> None:
    """Validates: Requirements 2.1

    The default configuration's ``database`` section exposes exactly the MySQL
    connection settings and no ``database.location``.
    """
    provider = ConfigProvider(env={})
    database = provider.get_section("database")

    for key in _CONNECTION_KEYS:
        assert key in database, f"database.{key} is missing from default_config.yaml"

    # host/name/user are non-empty strings; port is an int in range; password is
    # a string (empty allowed). These match the shipped docker-compose values.
    assert isinstance(database["host"], str) and database["host"].strip() != ""
    assert isinstance(database["name"], str) and database["name"].strip() != ""
    assert isinstance(database["user"], str) and database["user"].strip() != ""
    assert (
        isinstance(database["port"], int)
        and not isinstance(database["port"], bool)
        and 1 <= database["port"] <= 65535
    )
    assert isinstance(database["password"], str)


@pytest.mark.unit
def test_default_config_has_no_database_location() -> None:
    """Validates: Requirements 2.1

    The retired ``database.location`` setting is absent from both the section
    mapping and dotted-key lookup.
    """
    provider = ConfigProvider(env={})

    assert "location" not in provider.get_section("database")
    with pytest.raises(KeyError):
        provider.get("database.location")


# ---------------------------------------------------------------------------
# Requirement 2.6 -- no direction section, no direction.* validation error
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_default_config_has_no_direction_section() -> None:
    """Validates: Requirements 2.6

    The configuration defines no ``direction`` section.
    """
    provider = ConfigProvider(env={})

    with pytest.raises(KeyError):
        provider.get_section("direction")


@pytest.mark.unit
def test_validate_emits_no_direction_error() -> None:
    """Validates: Requirements 2.6

    Validating the shipped (valid) configuration reports no error, and in
    particular no error whose key targets the retired ``direction`` section.
    """
    provider = ConfigProvider(env={})
    errors = provider.validate()

    direction_errors = [e for e in errors if e.key.startswith("direction")]
    assert direction_errors == [], f"unexpected direction.* errors: {direction_errors}"
    # The shipped default config is fully valid.
    assert errors == [], f"default config unexpectedly invalid: {errors}"


# ---------------------------------------------------------------------------
# Requirement 2.7 -- invalid config halts before any DB connection is opened
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_invalid_connection_config_reports_each_offending_key() -> None:
    """Validates: Requirements 2.7

    When multiple MySQL connection values are invalid, ``validate()`` names each
    offending setting so the composition root can log them and refuse to start.
    """
    config = copy.deepcopy(_base_config())
    config["database"]["host"] = "   "   # whitespace-only -> invalid
    config["database"].pop("name", None)  # absent -> invalid
    config["database"]["port"] = 70000    # out of range -> invalid

    path = _write_temp_config(config)
    try:
        errors = ConfigProvider(config_path=path, env={}).validate()
    finally:
        os.unlink(path)

    reported = {e.key for e in errors}
    assert {"database.host", "database.name", "database.port"} <= reported, (
        f"validate() did not name every offending setting; reported: {sorted(reported)}"
    )


@pytest.mark.unit
def test_validation_opens_no_database_connection(monkeypatch) -> None:
    """Validates: Requirements 2.7

    Determining that a configuration is invalid opens no MySQL connection: the
    configuration layer reaches its verdict without the driver's ``connect``
    ever being called, so startup halts before any database connection is
    opened. The assertion runs whether or not the PyMySQL driver is installed
    (it never requires a reachable server, per Req 10.3).
    """
    connect_calls: list = []

    try:  # Guard the driver import so the test runs pre-install too.
        import pymysql

        monkeypatch.setattr(
            pymysql,
            "connect",
            lambda *a, **k: connect_calls.append((a, k)),
        )
    except ModuleNotFoundError:
        pass

    config = copy.deepcopy(_base_config())
    config["database"]["host"] = ""  # invalid -> would halt startup

    path = _write_temp_config(config)
    try:
        errors = ConfigProvider(config_path=path, env={}).validate()
    finally:
        os.unlink(path)

    # The invalid setting is surfaced (the halt trigger) ...
    assert any(e.key == "database.host" for e in errors)
    # ... and no database connection was opened to reach that verdict.
    assert connect_calls == [], "config validation must not open a DB connection"
