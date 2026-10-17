"""One-time AWS RDS bootstrap + connection verifier for the ANPR database.

This throwaway operational script does three things against the MySQL endpoint
described by the ``ANPR_DATABASE__*`` environment variables:

1. Connects over TLS (verifying the server cert against the CA bundle).
2. Loads ``docker/mysql/init/01-schema.sql`` so the externally-owned schema
   exists on RDS (RDS does not auto-run init SQL the way the local Docker
   container does). The script is idempotent because the schema uses
   ``CREATE TABLE IF NOT EXISTS`` and ``INSERT IGNORE``.
3. Verifies every table the application requires is present, reusing the app's
   own ``REQUIRED_TABLES`` constant so this check tracks the real requirement.

Credentials are read from the environment and never printed, so the password
never appears in logs, chat, or a committed file. Run it from a terminal where
the ``ANPR_DATABASE__*`` variables are already set:

    # PowerShell
    $env:ANPR_DATABASE__HOST     = "your-endpoint.rds.amazonaws.com"
    $env:ANPR_DATABASE__USER     = "your-master-user"
    $env:ANPR_DATABASE__PASSWORD = "your-master-password"
    $env:ANPR_DATABASE__NAME     = "anpr_system"
    $env:ANPR_DATABASE__SSL_CA   = "global-bundle.pem"

    .venv\\Scripts\\python.exe scripts\\aws_db_bootstrap.py

Exit code 0 means the schema is loaded and every required table is present.
Any failure (bad endpoint, wrong credentials, TLS problem, missing table)
prints a short message and exits non-zero.

This script is operational tooling, not part of the application runtime: the
app itself never issues DDL (it only verifies tables exist).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pymysql

from anpr.persistence.db import REQUIRED_TABLES

# Repo-root-relative path to the externally-owned schema. The app treats this
# schema as owned by the DB operator; this script is the operator action that
# applies it to a fresh RDS instance.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "docker" / "mysql" / "init" / "01-schema.sql"


def _env(name: str, default: str | None = None) -> str | None:
    """Read an ``ANPR_DATABASE__*`` variable; return ``default`` when unset."""
    return os.environ.get(name, default)


def _redact_host(host: str) -> str:
    """Return a host safe to print (the endpoint is not secret, but be tidy)."""
    return host


def _connect_params() -> dict:
    """Assemble PyMySQL connect kwargs from the environment.

    Mirrors the application's own ``database.*`` keys. ``ssl_ca`` is optional;
    when set the connection is made over TLS with server-cert verification,
    which is what a public RDS endpoint requires.
    """
    host = _env("ANPR_DATABASE__HOST")
    user = _env("ANPR_DATABASE__USER")
    password = _env("ANPR_DATABASE__PASSWORD")
    name = _env("ANPR_DATABASE__NAME", "anpr_system")
    port = int(_env("ANPR_DATABASE__PORT", "3306"))
    ssl_ca = _env("ANPR_DATABASE__SSL_CA")

    missing = [
        var
        for var, val in (
            ("ANPR_DATABASE__HOST", host),
            ("ANPR_DATABASE__USER", user),
            ("ANPR_DATABASE__PASSWORD", password),
        )
        if not val
    ]
    if missing:
        print(
            "Missing required environment variables: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Set ANPR_DATABASE__HOST/USER/PASSWORD (and optionally __NAME, "
            "__PORT, __SSL_CA) before running.",
            file=sys.stderr,
        )
        sys.exit(2)

    params = dict(
        host=host,
        port=port,
        user=user,
        password=password,
        charset="utf8mb4",
        connect_timeout=10,
        cursorclass=pymysql.cursors.DictCursor,
        # autocommit so the DDL in the schema file is persisted statement by
        # statement without an explicit transaction wrapper.
        autocommit=True,
    )
    if ssl_ca:
        ca_path = Path(ssl_ca)
        if not ca_path.is_absolute():
            ca_path = REPO_ROOT / ca_path
        if not ca_path.exists():
            print(f"CA bundle not found: {ca_path}", file=sys.stderr)
            sys.exit(2)
        params["ssl"] = {"ca": str(ca_path)}
    return params


def _load_schema(conn) -> None:
    """Execute every statement in the schema file (idempotent DDL)."""
    if not SCHEMA_PATH.exists():
        print(f"Schema file not found: {SCHEMA_PATH}", file=sys.stderr)
        sys.exit(2)
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = _split_statements(sql)
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    print(f"Applied {len(statements)} schema statement(s) from {SCHEMA_PATH.name}.")


def _split_statements(sql: str) -> list[str]:
    """Split a SQL script into individual statements on semicolons.

    The ANPR schema uses only plain statements (no stored routines, triggers,
    or custom DELIMITER blocks), so a naive split on ``;`` after stripping line
    comments is sufficient. Blank fragments are dropped.
    """
    lines = []
    for raw in sql.splitlines():
        stripped = raw.strip()
        if stripped.startswith("--"):
            continue  # drop full-line comments
        lines.append(raw)
    joined = "\n".join(lines)
    return [s.strip() for s in joined.split(";") if s.strip()]


def _verify_tables(conn, db_name: str) -> None:
    """Raise/exit if any application-required table is absent.

    The lookup is tolerant of the two things that commonly cause a false
    "missing" after a successful load against RDS:

    * the schema's own ``CREATE DATABASE anpr_system`` means the tables may live
      in ``anpr_system`` even when the connection's ``ANPR_DATABASE__NAME`` points
      elsewhere, so the search spans every non-system schema and reports which
      schema actually holds the tables; and
    * ``lower_case_table_names`` differences fold ``Resident`` to ``resident``,
      so the required-table match is case-insensitive.

    When the required tables are found in a schema other than ``db_name`` the
    mismatch is reported explicitly, because the application connects to
    ``ANPR_DATABASE__NAME`` and would otherwise see an empty database.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_schema AS s, table_name AS t "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN "
            "('mysql','information_schema','performance_schema','sys')"
        )
        rows = cur.fetchall()

    # schema -> {lowercased table name}
    by_schema: dict[str, set[str]] = {}
    for r in rows:
        schema = r.get("s") or r.get("table_schema") or r.get("TABLE_SCHEMA")
        table = r.get("t") or r.get("table_name") or r.get("TABLE_NAME")
        if schema is None or table is None:
            continue
        by_schema.setdefault(schema, set()).add(table.lower())

    print("Databases found on this server: " + (", ".join(sorted(by_schema)) or "(none)"))

    required_lc = {t.lower() for t in REQUIRED_TABLES}
    # Which schema (if any) holds every required table?
    holding = [s for s, tables in by_schema.items() if required_lc.issubset(tables)]

    if not holding:
        # Report the best-effort view so the cause is obvious.
        target = by_schema.get(db_name, set())
        missing = [t for t in REQUIRED_TABLES if t.lower() not in target]
        print(
            f"Required table(s) missing from database '{db_name}': "
            + ", ".join(missing),
            file=sys.stderr,
        )
        for s, tables in sorted(by_schema.items()):
            print(f"  - '{s}' contains: {', '.join(sorted(tables)) or '(empty)'}", file=sys.stderr)
        sys.exit(1)

    if db_name not in holding:
        print(
            f"Required tables exist, but in {holding!r}, not in the configured "
            f"ANPR_DATABASE__NAME='{db_name}'. Point the app at one of {holding} "
            f"(set ANPR_DATABASE__NAME), or re-create the schema in '{db_name}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"All required tables present in '{db_name}': " + ", ".join(REQUIRED_TABLES) + ".")


def main() -> int:
    params = _connect_params()
    db_name = _env("ANPR_DATABASE__NAME", "anpr_system")
    tls = " over TLS" if "ssl" in params else " (no TLS)"
    print(f"Connecting to MySQL at {_redact_host(params['host'])}:{params['port']}{tls} ...")
    try:
        conn = pymysql.connect(**params)
    except pymysql.MySQLError as exc:
        print(f"Could not connect: {exc}", file=sys.stderr)
        return 1
    try:
        _load_schema(conn)
        # Re-select the target database in case the script's USE ran against a
        # different default; information_schema is queried by name regardless.
        _verify_tables(conn, db_name)
    finally:
        conn.close()
    print("\nBootstrap complete. The app can now point at this endpoint via the")
    print("same ANPR_DATABASE__* variables (see AWS_SETUP.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
