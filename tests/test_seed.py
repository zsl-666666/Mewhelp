from sqlalchemy import select

from app.models import Faq
from app.repositories.faq import FAQ_SEED, seed_faq


def test_seed_faq_is_idempotent_and_contains_known_limitation(session_factory):
    with session_factory() as session:
        assert seed_faq(session) == len(FAQ_SEED)
        assert seed_faq(session) == 0
        rows = session.scalars(select(Faq)).all()

    assert any("退货政策" in row.question for row in rows)
    assert all("邮费" not in (row.question + row.answer + row.category) for row in rows)
