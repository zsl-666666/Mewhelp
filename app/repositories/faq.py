"""FAQ 仓储：LIKE 检索与幂等种子写入。"""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import Faq
from app.seed_data import FAQ_SEED

__all__ = ["FAQ_SEED", "search_faq", "seed_faq"]


def search_faq(session: Session, keyword: str, limit: int = 3) -> list[Faq]:
    """按问题或答案做 LIKE 检索，转义 ``%`` / ``_`` 等通配符。"""
    statement = (
        select(Faq)
        .where(
            or_(
                Faq.question.contains(keyword, autoescape=True),
                Faq.answer.contains(keyword, autoescape=True),
            )
        )
        .order_by(Faq.id)
        .limit(limit)
    )
    return list(session.scalars(statement).all())


def seed_faq(session: Session) -> int:
    """表为空时写入种子 FAQ 并返回条数；表非空返回 0（幂等）。"""
    count = session.scalar(select(func.count()).select_from(Faq))
    if count:
        return 0
    session.add_all([Faq(**row) for row in FAQ_SEED])
    session.commit()
    return len(FAQ_SEED)
