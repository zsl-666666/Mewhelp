from pathlib import Path

from scripts.init_db import read_schema_statements, table_state

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
