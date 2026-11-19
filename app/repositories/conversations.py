"""会话与消息仓储。

负责会话的创建/查询、消息追加、历史重建，以及会话状态更新。
历史重建必须与聊天编排层的消息类型严格对应：user → HumanMessage、
assistant → AIMessage（含 tool_calls）、tool → ToolMessage（content 取
JSON envelope 中的 ``result``）。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Conversation, Message


def create_conversation(session: Session, user_id: str) -> Conversation:
    """新建一个进行中的会话并返回。"""
    conversation = Conversation(user_id=user_id, status="进行中")
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return conversation


def get_conversation(session: Session, conversation_id: int) -> Conversation | None:
    """按主键查询会话，不存在返回 None。"""
    return session.get(Conversation, conversation_id)


def append_message(
    session: Session,
    conversation_id: int,
    *,
    role: str,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
) -> Message:
    """追加一条消息并提交，返回已刷新的 ORM 对象。"""
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    return message


def _tool_message_content(raw: str | None) -> str:
    """从工具消息的 JSON envelope 中取出 ``result``，失败时退回原文。"""
    if raw is None:
        return ""
    try:
        payload = json.loads(raw)
        result = payload["result"]
    except (json.JSONDecodeError, TypeError, KeyError):
        return raw
    return result if isinstance(result, str) else str(result)


def load_history(session: Session, conversation_id: int) -> list[BaseMessage]:
    """按消息 id 升序重建 LangChain 消息历史。"""
    rows = session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    ).all()

    history: list[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            history.append(HumanMessage(content=row.content or ""))
        elif row.role == "assistant":
            history.append(
                AIMessage(content=row.content or "", tool_calls=row.tool_calls or [])
            )
        elif row.role == "tool":
            history.append(
                ToolMessage(
                    content=_tool_message_content(row.content),
                    tool_call_id=row.tool_call_id or "",
                )
            )
    return history


def set_conversation_status(
    session: Session, conversation_id: int, status: str
) -> None:
    """更新会话状态并提交；会话不存在时抛出 ValueError。"""
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise ValueError(f"会话不存在: {conversation_id}")
    conversation.status = status
    session.commit()
