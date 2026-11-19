from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def test_schema_has_four_tables_in_order_and_target_collation():
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    assert sql.count("CREATE TABLE") == 4
    positions = [
        sql.index(f"CREATE TABLE {name}")
        for name in ("conversations", "messages", "faq", "tickets")
    ]
    assert positions == sorted(positions)
    table_suffix = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    assert sql.count(table_suffix) == 4
