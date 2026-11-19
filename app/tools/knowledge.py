"""FAQ 工具：按关键词检索本地 FAQ 表。"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field
from sqlalchemy.orm import Session, sessionmaker

from app.repositories.faq import search_faq


def build_query_faq_tool(session_factory: sessionmaker[Session]) -> BaseTool:
    """构造绑定 ``session_factory`` 的 ``query_faq`` 工具。"""

    @tool
    def query_faq(
        keyword: Annotated[str, Field(min_length=1)],
        limit: Annotated[int, Field(ge=1, le=10)] = 3,
    ) -> str:
        """按关键词查询常见问题（FAQ）及标准答案。

        请使用用户原话中的关键词直接检索，不要改写或替换为同义词。
        """
        with session_factory() as session:
            rows = search_faq(session, keyword, limit)
            data = [
                {
                    "question": row.question,
                    "answer": row.answer,
                    "category": row.category,
                }
                for row in rows
            ]

        if not data:
            return json.dumps(
                {"ok": True, "data": [], "message": "未找到相关 FAQ"},
                ensure_ascii=False,
            )
        return json.dumps({"ok": True, "data": data}, ensure_ascii=False)

    return query_faq
