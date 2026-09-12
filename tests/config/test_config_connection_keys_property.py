"""Property-based test for MySQL connection-key validation.

Feature: mysql-monitoring-migration, Property 6
Property 6: Config validation flags exactly the invalid connection keys.

Requirement 2.2 states the Configuration_Provider SHALL report a validation
error naming ``database.host``, ``database.name``, or ``database.user`` when that
value is absent, is not a string, or contains no non-whitespace characters.
Requirement 2.3 states it SHALL report an error naming ``database.port`` when the
port is absent, is not an integer, or falls outside the inclusive range 1..65535.

This test independently classifies each generated value as valid or invalid (by
the category it was drawn from, not by re-using the production predicate), writes
it into an otherwise valid config file, and asserts ``ConfigProvider.validate()``
names *exactly* the set of connection keys that were made invalid -- no more and
no fewer.
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


# Loaded once; each example deep-copies and overrides only the database
# connection keys so the rest of the configuration stays valid and the only
# variables under test are host/name/user/port.
_BASE_CONFIG = _base_config()

# Sentinel meaning "remove this key from the config entirely" (the absent case).
_ABSENT = object()

# The four connection keys this property governs.
_CONNECTION_KEYS = {
    "database.host",
    "database.name",
    "database.user",
    "database.port",
}

# --- string-field value categories (host / name / user) --------------------

# Valid: at least one non-whitespace character. Printable ASCII (no spaces) so
# the value round-trips cleanly through YAML and is unambiguously non-empty.
_valid_string = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
).filter(lambda s: s.strip() != "")

# Whitespace-only (including empty): invalid per Req 2.2.
_whitespace_string = st.sampled_from(["", " ", "   ", "\t", "\n", " \t\n "])

# Not a string at all: invalid per Req 2.2 (bool, int, float, None, list, dict).
_non_string = st.sampled_from([0, 1, 3306, -5, 3.14, True, False, None, [1, 2], {"k": "v"}])


def _string_field() -> st.SearchStrategy:
    """Draw a (category, value) pair for a string connection key."""
    return st.one_of(
        st.tuples(st.just("valid"), _valid_string),
        st.tuples(st.just("absent"), st.just(_ABSENT)),
        st.tuples(st.just("whitespace"), _whitespace_string),
        st.tuples(st.just("non_string"), _non_string),
    )


# --- port value categories -------------------------------------------------

# Valid: an int within the inclusive range 1..65535.
_valid_port = st.integers(min_value=1, max_value=65535)

# Out of range: an int <= 0 or > 65535: invalid per Req 2.3.
_out_of_range_port = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=65536),
)

# Not an integer: strings, floats, bool (excluded), None, containers: invalid.
_non_int_port = st.sampled_from(["3306", "", "abc", 3306.0, 0.5, True, False, None, [3306], {"p": 1}])


def _port_field() -> st.SearchStrategy:
    """Draw a (category, value) pair for ``database.port``."""
    return st.one_of(
        st.tuples(st.just("valid"), _valid_port),
        st.tuples(st.just("absent"), st.just(_ABSENT)),
        st.tuples(st.just("non_int"), _non_int_port),
        st.tuples(st.just("out_of_range"), _out_of_range_port),
    )


@pytest.mark.property
@given(host=_string_field(), name=_string_field(), user=_string_field(), port=_port_field())
def test_validate_flags_exactly_invalid_connection_keys(host, name, user, port) -> None:
    """Validates: Requirements 2.2, 2.3

    For any combination of absent / non-string / whitespace host/name/user and
    absent / non-int / out-of-range port, ``validate()`` names exactly the
    connection keys that were made invalid.
    """
    config = copy.deepcopy(_BASE_CONFIG)

    expected_invalid: set[str] = set()
    fields = {"host": host, "name": name, "user": user, "port": port}
    for key, (category, value) in fields.items():
        dotted = f"database.{key}"
        if value is _ABSENT:
            config["database"].pop(key, None)
        else:
            config["database"][key] = value
        if category != "valid":
            expected_invalid.add(dotted)

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

    reported = {err.key for err in errors if err.key in _CONNECTION_KEYS}

    assert reported == expected_invalid, (
        "validate() did not flag exactly the invalid connection keys.\n"
        f"  expected invalid: {sorted(expected_invalid)}\n"
        f"  reported invalid: {sorted(reported)}\n"
        f"  field categories: "
        f"{ {k: c for k, (c, _v) in fields.items()} }"
    )
