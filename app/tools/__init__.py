"""工具注册表与全部业务工具。"""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.knowledge import build_query_faq_tool
from app.tools.registry import ToolRegistry
from app.tools.tickets import build_create_ticket_tool

__all__ = ["build_tool_registry"]


def build_tool_registry(session_factory: sessionmaker[Session]) -> ToolRegistry:
    """注册五个业务工具，返回可绑定给模型的注册表。"""
    registry = ToolRegistry()
    registry.register(query_order, label="查订单")
    registry.register(query_product, label="查商品")
    registry.register(query_logistics, label="查物流")
    registry.register(build_query_faq_tool(session_factory), label="查 FAQ")
    registry.register(
        build_create_ticket_tool(session_factory),
        label="转人工",
        retryable=False,
    )
    return registry
