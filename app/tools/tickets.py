"""工单工具：创建人工工单并将会话状态改为已转人工。"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, InjectedToolArg, tool
from sqlalchemy.orm import Session, sessionmaker

from app.repositories.tickets import create_ticket_record


def build_create_ticket_tool(
    session_factory: sessionmaker[Session],
) -> BaseTool:
    """构造绑定 ``session_factory`` 的 ``create_ticket`` 工具。"""

    @tool
    def create_ticket(
        description: str,
        ticket_type: Literal["售后", "投诉", "咨询"],
        conversation_id: Annotated[int, InjectedToolArg],
    ) -> str:
        """创建人工工单，并把当前会话转交人工客服。

        ``ticket_type`` 按语义选择：退货/退款/物流等商品服务问题选“售后”，
        不满/投诉/服务态度问题选“投诉”，一般咨询选“咨询”。
        """
        with session_factory() as session:
            ticket = create_ticket_record(
                session,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
            data = {
                "ticket_no": ticket.ticket_no,
                "ticket_type": ticket.ticket_type,
                "status": ticket.status,
            }
        return json.dumps({"ok": True, "data": data}, ensure_ascii=False)

    return create_ticket
