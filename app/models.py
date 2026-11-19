"""Chapter 2 ORM 模型。

映射 `db/schema.sql` 中的四张表：faq / conversations / messages / tickets。
``db/schema.sql`` 是唯一事实来源，MySQL 业务表由该 DDL 创建，本模块只负责
查询与写入映射；SQLite 单测才通过 ``Base.metadata.create_all`` 建表。

MySQL 的 ENUM 用 ``String`` + ``CheckConstraint`` 做可移植映射，
BIGINT UNSIGNED 用 ``BigInteger().with_variant(Integer, "sqlite")`` 映射，
使 SQLite 自增主键正常工作。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# SQLite 只对 INTEGER PRIMARY KEY 提供自增，故 BIGINT 在 SQLite 降级为 Integer。
BigIntId = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    """所有 ORM 模型的声明式基类。"""


class Faq(Base):
    __tablename__ = "faq"
    __table_args__ = (Index("idx_category", "category"),)

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('进行中','已转人工','已结束')",
            name="ck_conversations_status",
        ),
        Index("idx_user_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="进行中", server_default="进行中"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # 只建这两条一对多关系；不配置级联删除，与 DDL 的默认 RESTRICT 一致。
    messages: Mapped[list["Message"]] = relationship()
    tickets: Mapped[list["Ticket"]] = relationship()


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user','assistant','tool')",
            name="ck_messages_role",
        ),
        # SQLite 的索引名是库级全局唯一，MySQL 是表级；
        # 因此这里的模型索引名加表前缀，避免与 tickets 的同类索引冲突。
        Index("idx_messages_conversation_id", "conversation_id"),
    )

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigIntId, ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint(
            "ticket_type IN ('售后','投诉','咨询')",
            name="ck_tickets_type",
        ),
        CheckConstraint(
            "status IN ('待处理','已处理')",
            name="ck_tickets_status",
        ),
        Index("idx_tickets_conversation_id", "conversation_id"),
    )

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigIntId, ForeignKey("conversations.id"), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="待处理", server_default="待处理"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
