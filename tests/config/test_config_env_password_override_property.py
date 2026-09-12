"""Property-based test for the MySQL ``database.password`` env override.

Feature: mysql-monitoring-migration, Property 8
Property 8: Environment password override wins.

Requirement 2.5 states that WHERE the environment variable
``ANPR_DATABASE__PASSWORD`` is defined, THE Configuration_Provider SHALL use that
environment-variable value as ``database.password`` in preference to the
configuration-file value, regardless of whether ``database.password`` is present
in the configuration file, and SHALL accept an empty environment-variable value
as valid.

This test generates arbitrary file-side password values (including "absent") and
arbitrary ``ANPR_DATABASE__PASSWORD`` values (including empty), writes the file
value into an otherwise valid config, overlays the env value, and asserts the
resolved ``database.password`` equals the env value in every case -- and that
``validate()`` reports no ``database.password`` error for it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yaml")
pytest.importorskip("hypothesis")

import copy
import os
import tempfile

import yaml
from hypothesis import given
from hypothesis import strategies as st

from anpr.config.provider import DEFAULT_CONFIG_PATH, ConfigProvider


def _base_config() -> dict:
    """Load the shipped default config as a valid baseline to mutate."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# Loaded once; each example deep-copies and overrides only the file-side
# password so the rest of the configuration stays valid and the only variables
# under test are the file password and the env password.
_BASE_CONFIG = _base_config()

# Sentinel meaning "remove database.password from the config file entirely" so
# the env override is exercised both with and without a file-side value present
# (Req 2.5: "regardless of whether database.password is present in the file").
_ABSENT = object()


# File-side password: either absent, or an arbitrary string (incl. empty and
# whitespace-only). ``_coerce`` types the env string toward the file value's
# type; since the file value (when present) is a string, the env value stays a
# string, and when absent the raw env string is used as-is.
file_password_strategy = st.one_of(
    st.just(_ABSENT),
    st.text(),
    st.sampled_from(["", " ", "\t", "anprpassword"]),
)

# Env-side password: arbitrary string, explicitly including empty and
# whitespace-only so the "empty env value is valid" boundary is always covered.
env_password_strategy = st.one_of(
    st.text(),
    st.sampled_from(["", " ", "   ", "\t", "\n", " \t \n "]),
)


@pytest.mark.property
@given(file_password=file_password_strategy, env_password=env_password_strategy)
def test_env_password_override_wins(file_password, env_password: str) -> None:
    """Validates: Requirements 2.5

    For any file-side ``database.password`` (present or absent) and any
    ``ANPR_DATABASE__PASSWORD`` env value (empty included), the resolved
    ``database.password`` equals the env value and is accepted as valid.
    """
    config = copy.deepcopy(_BASE_CONFIG)
    if file_password is _ABSENT:
        config["database"].pop("password", None)
    else:
        config["database"]["password"] = file_password

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", encoding="utf-8", delete=False
    )
    try:
        yaml.safe_dump(config, handle, allow_unicode=True)
        handle.close()

        provider = ConfigProvider(
            config_path=handle.name,
            env={"ANPR_DATABASE__PASSWORD": env_password},
        )
        resolved = provider.get("database.password")
        errors = provider.validate()
    finally:
        os.unlink(handle.name)

    assert resolved == env_password, (
        "env ANPR_DATABASE__PASSWORD did not win over the file value.\n"
        f"  file password: {file_password!r}\n"
        f"  env password:  {env_password!r}\n"
        f"  resolved:      {resolved!r}"
    )
    assert isinstance(resolved, str), (
        f"resolved database.password should stay a string, got {type(resolved)!r}"
    )
    password_errors = [err for err in errors if err.key == "database.password"]
    assert password_errors == [], (
        f"env-overridden database.password rejected for value {env_password!r}: "
        f"{password_errors}"
    )
