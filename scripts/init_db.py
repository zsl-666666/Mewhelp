"""幂等建表 + FAQ 种子脚本。

用法::

    uv run python -m scripts.init_db

只读 ``db/schema.sql`` 建表，表为空时灌入种子 FAQ。
四张表齐全时跳过；结构不完整直接报错，不会自动补建或删表。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.db import get_engine, get_session_factory
from app.repositories.faq import seed_faq

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"

EXPECTED_TABLES = frozenset({"conversations", "messages", "faq", "tickets"})

TableState = Literal["empty", "ready", "partial"]


def read_schema_statements(path: Path) -> list[str]:
    """读取 DDL，去掉整行 ``--`` 注释，返回所有 ``CREATE TABLE`` 语句。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    body = "\n".join(line for line in lines if not line.lstrip().startswith("--"))
    return [
        statement.strip()
        for statement in body.split(";")
        if statement.strip().startswith("CREATE TABLE")
    ]


def table_state(existing: set[str]) -> TableState:
    """根据现有表名判断建表状态。"""
    if not existing:
        return "empty"
    if EXPECTED_TABLES <= set(existing):
        return "ready"
    return "partial"


def apply_schema(engine: Engine, path: Path = SCHEMA_PATH) -> str:
    """空库建表；已就绪跳过；结构不完整直接报错。"""
    existing = set(inspect(engine).get_table_names())
    state = table_state(existing)

    if state == "ready":
        return "skipped"
    if state == "partial":
        present = ", ".join(sorted(existing & EXPECTED_TABLES)) or "无"
        missing = ", ".join(sorted(EXPECTED_TABLES - existing))
        raise RuntimeError(
            f"数据库表结构不完整：已存在 [{present}]，缺少 [{missing}]。"
            "脚本不会自动补建或删除表，请人工核对后重试。"
        )

    with engine.begin() as connection:
        for statement in read_schema_statements(path):
            connection.execute(text(statement))
    return "applied"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    state = apply_schema(get_engine())
    with get_session_factory()() as session:
        seeded = seed_faq(session)

    print(f"表结构：{state}")
    if seeded:
        print(f"FAQ 种子：写入 {seeded} 条")
    else:
        print("FAQ 种子：已存在，跳过写入")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
