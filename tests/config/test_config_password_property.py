"""Property-based test for MySQL ``database.password`` acceptance.

Feature: mysql-monitoring-migration, Property 7
Property 7: Any string is a valid password.

Requirement 2.4 states the Configuration_Provider SHALL accept an empty string
as a valid ``database.password`` and SHALL treat any string value (including one
containing whitespace) as valid. This test generates arbitrary password strings
-- including empty and whitespace-only values -- writes each into an otherwise
valid config file, and asserts ``ConfigProvider.validate()`` never reports a
``database.password`` error.
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


# Loaded once; each example deep-copies and overrides only the password so the
# rest of the configuration stays valid and the only variable is the password.
_BASE_CONFIG = _base_config()


# Arbitrary strings, explicitly including empty and whitespace-only examples so
# the boundary cases from Requirement 2.4 are always exercised.
password_strategy = st.one_of(
    st.text(),
    st.sampled_from(["", " ", "   ", "\t", "\n", " \t \n "]),
)


@pytest.mark.property
@given(password=password_strategy)
def test_any_string_is_a_valid_password(password: str) -> None:
    """Validates: Requirements 2.4

    For any string ``database.password`` (empty or whitespace-only included),
    ``validate()`` reports no error naming ``database.password``.
    """
    config = copy.deepcopy(_BASE_CONFIG)
    config["database"]["password"] = password

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", encoding="utf-8", delete=False
    )
    try:
        yaml.safe_dump(config, handle, allow_unicode=True)
        handle.close()

        provider = ConfigProvider(config_path=handle.name, env={})
        errors = provider.validate()
    finally:
        os.unlink(handle.name)

    password_errors = [err for err in errors if err.key == "database.password"]
    assert password_errors == [], (
        f"database.password rejected for value {password!r}: {password_errors}"
    )
