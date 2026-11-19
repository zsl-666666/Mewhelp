"""工单仓储：创建工单并在同一事务内置会话为已转人工。"""

from __future__ import annotations

import secrets
from datetime import datetime

from sqlalchemy.orm import Session

from app.models import Conversation, Ticket

ALLOWED_TICKET_TYPES = frozenset({"售后", "投诉", "咨询"})


def _generate_ticket_no() -> str:
    """生成 ``T`` + 14 位时间戳 + 4 位十六进制，共 19 位。"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"T{timestamp}{secrets.token_hex(2).upper()}"


def create_ticket_record(
    session: Session,
    conversation_id: int,
    description: str,
    ticket_type: str,
) -> Ticket:
    """创建工单并把所属会话状态改为“已转人工”。

    参数非法或会话不存在时抛出 ValueError，且不产生任何写入。
    """
    if ticket_type not in ALLOWED_TICKET_TYPES:
        raise ValueError(f"不支持的工单类型: {ticket_type}")

    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise ValueError(f"会话不存在: {conversation_id}")

    ticket = Ticket(
        ticket_no=_generate_ticket_no(),
        conversation_id=conversation_id,
        description=description,
        ticket_type=ticket_type,
        status="待处理",
    )
    session.add(ticket)
    conversation.status = "已转人工"
    session.commit()
    session.refresh(ticket)
    return ticket


def get_ticket(session: Session, ticket_no: str) -> Ticket | None:
    """按工单号查询工单，不存在返回 None。"""
    return session.get(Ticket, ticket_no)
