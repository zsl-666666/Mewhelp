from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from scripts.init_db import apply_schema, read_schema_statements, table_state

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def test_read_schema_statements_returns_four_create_tables():
    statements = read_schema_statements(SCHEMA_PATH)
    assert len(statements) == 4
    assert statements[0].startswith("CREATE TABLE conversations")
    assert statements[1].startswith("CREATE TABLE messages")
    assert statements[2].startswith("CREATE TABLE faq")
    assert statements[3].startswith("CREATE TABLE tickets")


def test_table_state():
    tables = {"conversations", "messages", "faq", "tickets"}
    assert table_state(set()) == "empty"
    assert table_state(tables) == "ready"
    assert table_state({"conversations"}) == "partial"


EXPECTED_TABLES = ("conversations", "messages", "faq", "tickets")


def _create_tables(engine, names: tuple[str, ...]) -> None:
    with engine.begin() as connection:
        for name in names:
            connection.execute(text(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY)"))


def test_table_state_treats_superset_as_ready():
    tables = {"conversations", "messages", "faq", "tickets", "audit_log"}
    assert table_state(tables) == "ready"


def test_apply_schema_applies_empty_database(tmp_path):
    # 真实 DDL 是 MySQL 语法，这里用最小 SQLite 兼容 DDL 验证执行与返回值。
    schema = tmp_path / "schema.sql"
    schema.write_text(
        "CREATE TABLE sample (id INTEGER PRIMARY KEY);", encoding="utf-8"
    )
    engine = create_engine("sqlite://")

    assert apply_schema(engine, schema) == "applied"
    assert set(inspect(engine).get_table_names()) == {"sample"}


def test_apply_schema_skips_when_all_tables_present():
    engine = create_engine("sqlite://")
    _create_tables(engine, EXPECTED_TABLES)

    assert apply_schema(engine) == "skipped"
    assert set(inspect(engine).get_table_names()) == set(EXPECTED_TABLES)


def test_apply_schema_rejects_partial_schema_without_creating_tables():
    engine = create_engine("sqlite://")
    _create_tables(engine, ("conversations",))

    with pytest.raises(RuntimeError, match="conversations"):
        apply_schema(engine)

    assert set(inspect(engine).get_table_names()) == {"conversations"}
